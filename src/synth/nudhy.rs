//! NuDHy state container + MCMC kernel for the configuration-style null model.
//!
//! Phase 13b — see `docs/synth_nudhy_algorithm.md` for the algorithm spec.
//!
//! ## What this module owns
//!
//! - [`NudhyState`] — the MCMC chain state (hyperedges + dual indices).
//! - [`NudhyParams`] — calibration output / generation input (defaults
//!   computed from source state).
//! - [`nudhy_mcmc_step`] — one Chodrow 2020 §3.2 double-edge swap.
//! - [`run_nudhy_chain`] — burn-in + sample-gap loop with starvation guard.
//!
//! ## What this module does NOT own
//!
//! - The `SurrogateModel` impl (`NudhySurrogate`) — see
//!   [`super::nudhy_surrogate`].
//! - Calibration entry-point — see [`super::nudhy_surrogate::calibrate`].
//! - Generation entry-point — see [`super::nudhy_surrogate::run_generate`].
//!
//! ## Determinism contract
//!
//! Every stochastic draw inside this module flows through a single
//! [`ChaCha8Rng`] passed by `&mut`. Same seed → same swap trace →
//! same final state. The convention matches Phase 1 + 2.5 (`ChaCha8Rng`,
//! NOT `StdRng`); see `docs/synth_nudhy_algorithm.md` §3 for the reasoning.
//!
//! ## Invariants — held across every accepted swap
//!
//! 1. `entity_degree[v]` equals `entity_to_edges[v].len()`.
//! 2. `entity_to_edges[v]` is the exact sorted list of indices
//!    `i` such that `hyperedges[i]` contains `v`.
//! 3. The multiset `{|hyperedges[i]|}` is constant (size-preservation).
//! 4. Every `hyperedges[i]` is sorted ascending — enables binary-search
//!    membership and deterministic serialisation.

use std::collections::HashMap;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};

// ── State ────────────────────────────────────────────────────────────────────

/// Complete MCMC state for one chain in the NuDHy null model.
///
/// **Never mutate `hyperedges`, `entity_degree`, or `entity_to_edges`
/// directly.** All mutations go through [`NudhyState::apply_swap`] which
/// keeps the three structures in lockstep.
#[derive(Debug, Clone)]
pub struct NudhyState {
    /// One entry per situation; each inner Vec is the participant UUID list,
    /// kept sorted ascending so membership checks are O(log |e_i|) via
    /// `binary_search`.
    pub hyperedges: Vec<Vec<Uuid>>,
    /// Running hyperdegree per entity. Invariant per design doc §2.1:
    /// equals the count of hyperedges that contain this entity.
    pub entity_degree: HashMap<Uuid, u32>,
    /// Inverted index: entity → sorted Vec of hyperedge indices that
    /// contain it. Sorted to enable O(log n) `contains` and stable
    /// debugging output.
    pub entity_to_edges: HashMap<Uuid, Vec<usize>>,
}

impl NudhyState {
    /// Build a fresh state from a raw hyperedge list. Each inner Vec is
    /// sorted; the dual indices are populated in one pass.
    ///
    /// O(sum_of_edge_sizes × log(max_degree)).
    pub fn from_hyperedges(raw: Vec<Vec<Uuid>>) -> Self {
        let mut hyperedges: Vec<Vec<Uuid>> = raw
            .into_iter()
            .map(|mut e| {
                e.sort();
                e.dedup(); // hardening: a duplicate would break apply_swap's invariants
                e
            })
            .collect();

        let mut entity_degree: HashMap<Uuid, u32> = HashMap::new();
        let mut entity_to_edges: HashMap<Uuid, Vec<usize>> = HashMap::new();
        for (i, edge) in hyperedges.iter_mut().enumerate() {
            for &v in edge.iter() {
                *entity_degree.entry(v).or_insert(0) += 1;
                entity_to_edges.entry(v).or_default().push(i);
            }
        }
        // entity_to_edges values are already in ascending `i` order because
        // we visit `(i, edge)` left-to-right.

        Self {
            hyperedges,
            entity_degree,
            entity_to_edges,
        }
    }

    /// Number of hyperedges currently in the chain state.
    pub fn num_hyperedges(&self) -> usize {
        self.hyperedges.len()
    }

    /// Total (edge, node) incidences — proxy for the participation-record count.
    pub fn sum_of_edge_sizes(&self) -> usize {
        self.hyperedges.iter().map(|e| e.len()).sum()
    }

    /// O(log |e_i|) membership check via binary search on the sorted inner Vec.
    pub fn edge_contains(&self, edge_idx: usize, entity: Uuid) -> bool {
        match self.hyperedges.get(edge_idx) {
            Some(e) => e.binary_search(&entity).is_ok(),
            None => false,
        }
    }

    /// Apply an accepted double-swap.
    ///
    /// Per design doc §2.4: removes `v1` from edge `i1`, inserts `v2` in
    /// `i1`; removes `v2` from edge `i2`, inserts `v1` in `i2`. Updates
    /// `entity_to_edges` for both entities incrementally — degrees are
    /// invariant by construction, so `entity_degree` is left untouched.
    ///
    /// Panics in debug builds if any precondition (membership / non-membership)
    /// fails; releases swallow silently to keep the MCMC inner loop branch-free.
    pub fn apply_swap(&mut self, i1: usize, v1: Uuid, i2: usize, v2: Uuid) {
        // Edge 1: remove v1, insert v2.
        if let Some(edge) = self.hyperedges.get_mut(i1) {
            if let Ok(pos) = edge.binary_search(&v1) {
                edge.remove(pos);
            } else {
                debug_assert!(false, "apply_swap: v1 not in edge i1");
                return;
            }
            let insert_at = edge.binary_search(&v2).unwrap_or_else(|p| p);
            edge.insert(insert_at, v2);
        }
        // Edge 2: remove v2, insert v1.
        if let Some(edge) = self.hyperedges.get_mut(i2) {
            if let Ok(pos) = edge.binary_search(&v2) {
                edge.remove(pos);
            } else {
                debug_assert!(false, "apply_swap: v2 not in edge i2");
                return;
            }
            let insert_at = edge.binary_search(&v1).unwrap_or_else(|p| p);
            edge.insert(insert_at, v1);
        }
        // entity_to_edges: v1 leaves i1, joins i2; v2 leaves i2, joins i1.
        update_entity_index(&mut self.entity_to_edges, v1, i1, i2);
        update_entity_index(&mut self.entity_to_edges, v2, i2, i1);
    }

    /// Source-projection helper for membership-preserving emission. Returns
    /// the union of every entity UUID that appears in `hyperedges`, sorted
    /// for deterministic downstream consumption.
    pub fn entity_universe(&self) -> Vec<Uuid> {
        let mut out: Vec<Uuid> = self.entity_degree.keys().copied().collect();
        out.sort();
        out
    }
}

/// `entity_to_edges[v]`: remove `from_edge`, insert `to_edge` in sorted order.
/// Both operations are O(log n + n) on a Vec; for typical narrative degrees
/// (mean ≤ 6) this is effectively constant.
fn update_entity_index(
    index: &mut HashMap<Uuid, Vec<usize>>,
    entity: Uuid,
    from_edge: usize,
    to_edge: usize,
) {
    let slot = index.entry(entity).or_default();
    if let Ok(pos) = slot.binary_search(&from_edge) {
        slot.remove(pos);
    }
    let insert_at = slot.binary_search(&to_edge).unwrap_or_else(|p| p);
    slot.insert(insert_at, to_edge);
}

// ── Params ──────────────────────────────────────────────────────────────────

/// NuDHy-specific generation parameters.
///
/// Stored inside [`super::types::SurrogateParams::params_json`] after JSON
/// conversion. The `initial_state_json` field is the only large field
/// (O(num_situations × mean_size)); everything else is scalar.
///
/// Per design doc §14 Q1, size-1 hyperedges that can't participate in any
/// swap live in `fixed_edges_json` (separate from the chain state) and are
/// emitted unchanged during generation.
///
/// Per design doc §14 Q2, only `hyperedges` is persisted on disk; the dual
/// indices (`entity_degree`, `entity_to_edges`) are recomputed from scratch
/// via [`NudhyState::from_hyperedges`] on deserialisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NudhyParams {
    /// Number of MCMC steps for the burn-in phase.
    /// Default: `max(10_000, 10 × sum_of_edge_sizes)` (§5.1).
    pub burn_in_steps: usize,
    /// Steps between successive samples when drawing multiple surrogates
    /// from one chain. Significance / fidelity K-loops use independent
    /// chains; this field governs only rare multi-sample single-chain use.
    /// Default: `max(1_000, sum_of_edge_sizes)` (§5.1).
    pub sample_gap_steps: usize,
    /// Minimum acceptance rate over the first 1000 proposals before
    /// starvation is declared. Default: 0.01 (§5 + §9.3).
    pub accept_rejection_rate_min: f32,
    /// Serialised initial `NudhyState.hyperedges` — the chain-mutable
    /// hyperedge snapshot derived at calibration time. JSON shape:
    /// `Vec<Vec<String>>` where each inner string is a UUID.
    pub initial_state_json: serde_json::Value,
    /// Size-1 hyperedges excluded from the MCMC chain (per design doc §9.4).
    /// Emitted unchanged at generation time. Same JSON shape as
    /// `initial_state_json`. May be `null` / empty when no such edges exist.
    #[serde(default)]
    pub fixed_edges_json: serde_json::Value,
}

impl NudhyParams {
    /// Compute defaults from the chain-mutable initial state and any
    /// already-extracted fixed (size-1) edges. See §5.1 for the formulas.
    pub fn from_source_state(state: &NudhyState, fixed_edges: Vec<Vec<Uuid>>) -> Result<Self> {
        let sum_sizes = state.sum_of_edge_sizes();
        let burn_in_steps = std::cmp::max(10_000, 10 * sum_sizes);
        let sample_gap_steps = std::cmp::max(1_000, sum_sizes);
        let initial_state_json = serde_json::to_value(&state.hyperedges)?;
        let fixed_edges_json = if fixed_edges.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::to_value(&fixed_edges)?
        };
        Ok(Self {
            burn_in_steps,
            sample_gap_steps,
            accept_rejection_rate_min: 0.01,
            initial_state_json,
            fixed_edges_json,
        })
    }
}

// ── MCMC kernel ──────────────────────────────────────────────────────────────

/// One Chodrow 2020 §3.2 double-edge swap step.
///
/// Returns `true` iff the proposed swap was accepted (and thus the state
/// mutated). Returns `false` for trivial / invalid proposals (degenerate
/// state, size-1 edges, identical nodes, would-create-duplicate post-state).
///
/// Pure swap kernel: no Metropolis ratio — acceptance is a hard 0/1 on
/// validity (post-swap edges remain simple sets). See §1.2 for the
/// stationary-distribution proof sketch.
pub fn nudhy_mcmc_step(state: &mut NudhyState, rng: &mut ChaCha8Rng) -> bool {
    let num_edges = state.hyperedges.len();
    if num_edges < 2 {
        return false;
    }

    // 1. Two distinct hyperedge indices, rejection-free.
    let i1 = rng.gen_range(0..num_edges);
    let mut i2 = rng.gen_range(0..num_edges - 1);
    if i2 >= i1 {
        i2 += 1;
    }

    // 2. Read sizes — size-1 edges can't contribute (would empty the edge).
    let len1 = state.hyperedges[i1].len();
    let len2 = state.hyperedges[i2].len();
    if len1 < 2 || len2 < 2 {
        return false;
    }

    // 3. One node uniformly from each.
    let pos1 = rng.gen_range(0..len1);
    let pos2 = rng.gen_range(0..len2);
    let v1 = state.hyperedges[i1][pos1];
    let v2 = state.hyperedges[i2][pos2];
    if v1 == v2 {
        return false;
    }

    // 4. Validity check — would the swap create a duplicate inside e1' or e2'?
    if state.edge_contains(i2, v1) {
        return false;
    }
    if state.edge_contains(i1, v2) {
        return false;
    }

    // 5. Apply.
    state.apply_swap(i1, v1, i2, v2);
    true
}

/// Burn-in + sample-gap MCMC chain runner with starvation detection.
///
/// Chain steps draw from the supplied `rng`. Returns the final state after
/// `burn_in_steps + sample_gap_steps` proposals.
///
/// Errors with `TensaError::SynthFailure` when fewer than
/// `params.accept_rejection_rate_min × 1000` of the first 1000 proposals
/// were accepted (per §9.3). Detects pathological "every entity in every
/// edge" sources where no valid swap exists.
pub fn run_nudhy_chain(
    initial_state: NudhyState,
    params: &NudhyParams,
    rng: &mut ChaCha8Rng,
) -> Result<NudhyState> {
    let mut state = initial_state;
    let mut accepted: u64 = 0;
    let mut proposed: u64 = 0;
    const STARVATION_PROBE: u64 = 1000;

    for _ in 0..params.burn_in_steps {
        proposed += 1;
        if nudhy_mcmc_step(&mut state, rng) {
            accepted += 1;
        }
        if proposed == STARVATION_PROBE {
            let rate = accepted as f32 / STARVATION_PROBE as f32;
            tracing::info!(
                accepted,
                proposed = STARVATION_PROBE,
                rate,
                "NuDHy MCMC: 1000-step probe acceptance rate"
            );
            if rate < params.accept_rejection_rate_min {
                return Err(TensaError::SynthFailure(format!(
                    "NuDHy MCMC starvation: accepted {accepted} of first {STARVATION_PROBE} proposals \
                     (rate {rate:.4}, min {:.4}); source hypergraph may be too rigid for \
                     configuration-model randomisation. Consider using EATH instead.",
                    params.accept_rejection_rate_min
                )));
            }
        }
    }

    // Sample-gap phase — never errors on rate (chain is past burn-in).
    for _ in 0..params.sample_gap_steps {
        nudhy_mcmc_step(&mut state, rng);
    }

    Ok(state)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "nudhy_tests.rs"]
mod nudhy_tests;
