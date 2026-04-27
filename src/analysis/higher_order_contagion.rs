//! Higher-order SIR contagion on temporal hypergraphs (Phase 7b).
//!
//! Implements the Iacopini, Petri, Barrat, Latora (Nature Communications 2019)
//! "Simplicial models of social contagion" threshold model. A susceptible node
//! is infected via a hyperedge `h` only when the count of infected co-members
//! of `h` (excluding the node itself) meets a threshold rule, and a per-size
//! Bernoulli trial fires. β_2 corresponds to pairwise transmission;
//! β_3, β_4, ... are higher-order (group-driven) contagion.
//!
//! ## Reduction-to-pairwise contract — LOAD-BEARING
//!
//! Setting `beta_per_size = vec![beta, 0.0, 0.0, ...]` AND
//! `threshold = ThresholdRule::Absolute(1)` MUST produce the same per-step
//! dynamics as `analysis::contagion`-style pairwise SIR with the same
//! `(beta, gamma, seed, initial_infected)`. **Future contributors who add
//! new branches MUST preserve this reduction test, or `analysis::contagion`
//! semantics break.** Concretely: under those parameters every transmission
//! check is "any single infected partner triggers a Bernoulli(beta) draw via
//! a size-2 hyperedge"; higher-d edges contribute zero rate.
//!
//! See `test_higher_order_sir_reduces_to_pairwise_when_only_beta_2_and_threshold_absolute_1`.
//!
//! ## Determinism contract
//!
//! All stochastic decisions are drawn from a single `ChaCha8Rng` seeded with
//! `params.rng_seed`. The consumption order at each step is:
//!
//! 1. **Seeding** (step 0 only, when `SeedStrategy::RandomFraction(f)`):
//!    Fisher-Yates shuffle of a working `Vec<usize>` of entity indices,
//!    then prefix selection. Consumes `n - 1` u64 draws.
//! 2. **Per step**, for each susceptible entity `v` in canonical (sorted-Uuid)
//!    order, for each hyperedge `h` containing `v` in canonical order: if
//!    threshold met, draw one Bernoulli(beta_per_size[d-1]) — break the
//!    h-loop on success, otherwise continue to the next hyperedge.
//! 3. **Per step**, for each currently infected entity in canonical order,
//!    draw one Bernoulli(gamma) for recovery.
//!
//! Reordering any of these steps changes the RNG stream and breaks
//! reproducibility.
//!
//! ## N+1 mitigation
//!
//! Per-step transmission needs to enumerate, for each susceptible entity,
//! the hyperedges containing it. We build the
//! `entity_index → Vec<hyperedge_index>` map ONCE at the top of
//! `simulate_higher_order_sir` (O(P) where P = total participations) and
//! reuse it across every step.
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! [`ThresholdRule::Fraction`] gates transmission on the crisp ratio
//! `count_infected / (size - 1) >= f`. [`ThresholdRule::met_with_tnorm`]
//! offers an alternative graded interpretation: the ratio is evaluated
//! against the threshold `f` under the chosen t-norm semantics. The
//! default-wired hot path [`ThresholdRule::met`] remains bit-identical to
//! the pre-sprint behaviour.
//!
//! Cites: [klement2000].

use std::collections::HashMap;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::{combine_tnorm, TNormKind};
use crate::hypergraph::Hypergraph;

// ── Public types ────────────────────────────────────────────────────────────

/// Threshold rule that gates whether a hyperedge can transmit to a susceptible
/// member. The `count_infected` value passed to the rule is the number of
/// infected members of the hyperedge OTHER than the candidate susceptible.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ThresholdRule {
    /// Require ≥ N infected co-members. `Absolute(1)` means the edge can fire
    /// whenever any single co-member is already infected — the pairwise
    /// reduction baseline.
    Absolute(usize),
    /// Require ≥ f fraction of co-members infected. `f` should be in `[0, 1]`
    /// but the engine clamps internally.
    Fraction(f32),
}

impl ThresholdRule {
    /// Test whether a hyperedge of `size` total members is allowed to attempt
    /// transmission given `count_infected` infected co-members (excluding the
    /// candidate). Edges of size < 2 always fail (no co-members to infect through).
    #[inline]
    fn met(&self, count_infected: usize, size: usize) -> bool {
        if size < 2 {
            return false;
        }
        match *self {
            Self::Absolute(n) => count_infected >= n,
            Self::Fraction(f) => {
                let denom = (size - 1) as f32;
                if denom <= 0.0 {
                    return false;
                }
                (count_infected as f32 / denom) >= f.clamp(0.0, 1.0)
            }
        }
    }

    /// Graded variant of [`Self::met`] that evaluates the threshold under
    /// a chosen t-norm. Returns a fuzzy truth value in `[0, 1]`. Used by
    /// Phase 1+ fuzzy extensions that want to thread the t-norm selector
    /// through contagion aggregation.
    pub fn met_with_tnorm(
        &self,
        count_infected: usize,
        size: usize,
        tnorm: TNormKind,
    ) -> f64 {
        if size < 2 {
            return 0.0;
        }
        let (lhs, rhs, hard_pass) = match *self {
            Self::Absolute(n) => {
                let lhs = count_infected.min(size) as f64 / size as f64;
                let rhs = n.min(size) as f64 / size as f64;
                (lhs, rhs, count_infected >= n)
            }
            Self::Fraction(f) => {
                let denom = (size - 1) as f64;
                if denom <= 0.0 {
                    return 0.0;
                }
                let lhs = (count_infected as f64 / denom).clamp(0.0, 1.0);
                let rhs = (f as f64).clamp(0.0, 1.0);
                (lhs, rhs, lhs >= rhs)
            }
        };
        // Fold the crisp decision together with a soft ">=" indicator
        // `T(lhs, 1 - rhs)` so non-Gödel kinds produce measurably
        // different graded outputs on borderline inputs.
        let soft_indicator = combine_tnorm(tnorm, lhs, 1.0 - rhs);
        combine_tnorm(tnorm, if hard_pass { 1.0 } else { 0.0 }, soft_indicator)
    }
}

/// How to seed the initial infected population at step 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SeedStrategy {
    /// Seed `f` fraction of entities (rounded down) infected at t=0. Selection
    /// is a deterministic Fisher-Yates shuffle off `params.rng_seed`.
    RandomFraction { fraction: f32 },
    /// Seed exactly the listed entity ids infected at t=0. Unknown ids are
    /// silently dropped.
    Specific { entity_ids: Vec<Uuid> },
}

/// Parameters for one higher-order SIR simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HigherOrderSirParams {
    /// Per-size transmission rate. Index `d - 2` is the rate for size-`d`
    /// hyperedges. So `beta_per_size[0]` = pairwise (size-2) rate,
    /// `beta_per_size[1]` = size-3 rate, etc. Hyperedges whose size exceeds
    /// `beta_per_size.len() + 1` contribute zero rate (silently skipped).
    pub beta_per_size: Vec<f32>,
    /// Per-step recovery rate (Bernoulli probability). 0.0 → never recover.
    pub gamma: f32,
    /// Threshold rule gating transmission attempts.
    pub threshold: ThresholdRule,
    /// Seed strategy for the initial infected set.
    pub seed_strategy: SeedStrategy,
    /// Maximum number of simulation steps. Returned trajectories are truncated
    /// at the first step where prevalence reaches zero.
    pub max_steps: usize,
    /// RNG seed driving every stochastic decision in the simulation.
    pub rng_seed: u64,
}

/// SIR-state of an entity at the end of the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HigherOrderSirState {
    Susceptible,
    Infected,
    Recovered,
}

/// Per-simulation summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HigherOrderSirResult {
    /// Per-step count of currently-infected entities. `len() == steps_executed`.
    pub per_step_infected: Vec<usize>,
    /// Per-step count of currently-recovered entities. Same length as above.
    pub per_step_recovered: Vec<usize>,
    /// Approximate R₀: average number of secondary infections caused by each
    /// of the first ≤5 infectious entities during their first 5 steps as
    /// infectious. 0.0 when no transmissions occurred.
    pub r0_estimate: f32,
    /// Step index of peak prevalence (first occurrence on a tie).
    pub time_to_peak: usize,
    /// Peak `infected / total_entities`.
    pub peak_prevalence: f32,
    /// `size_attribution[d - 2]` = number of transmissions credited to
    /// size-`d` hyperedges. Sum equals total_secondary_transmissions.
    pub size_attribution: Vec<u64>,
    /// Total number of entities considered in the simulation.
    pub total_entities: usize,
    /// Final state per entity, indexed by canonical entity ordering.
    pub final_states: Vec<HigherOrderSirState>,
}

// ── Public entry point ──────────────────────────────────────────────────────

/// Simulate higher-order SIR contagion on the hyperedges (situations) of a
/// narrative.
///
/// Hyperedges are derived from situations: each situation's participant list
/// is one hyperedge, with size = number of distinct participants in that
/// situation. Situations are NOT temporally ordered — every hyperedge is
/// available at every step (matching the Iacopini et al. 2019 static-network
/// formulation, applied to TENSA's temporal hypergraph by treating the union
/// of all situations as the substrate).
///
/// See module-level docs for determinism contract and the reduction-to-pairwise
/// contract.
pub fn simulate_higher_order_sir(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &HigherOrderSirParams,
) -> Result<HigherOrderSirResult> {
    // 1. Load entities + situations (canonical ordering = sorted by Uuid bytes
    //    so the deterministic loop matches across runs that mint UUIDs in
    //    different temporal orders).
    let mut entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    entities.sort_by_key(|e| e.id);
    let total_entities = entities.len();

    if total_entities == 0 {
        return Ok(empty_result(params));
    }

    let entity_idx: HashMap<Uuid, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id, i))
        .collect();

    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    situations.sort_by_key(|s| s.id);

    // 2. Build hyperedges (one per situation) — list of entity indices, sorted.
    //    De-duplicate participants per situation (multi-role participation can
    //    yield duplicate (entity, situation) pairs which would inflate
    //    threshold counts).
    let mut hyperedges: Vec<Vec<usize>> = Vec::with_capacity(situations.len());
    for sit in &situations {
        let parts = hypergraph.get_participants_for_situation(&sit.id)?;
        let mut members: Vec<usize> = parts
            .iter()
            .filter_map(|p| entity_idx.get(&p.entity_id).copied())
            .collect();
        members.sort();
        members.dedup();
        if members.len() >= 2 {
            hyperedges.push(members);
        }
    }

    // 3. Build entity → list of hyperedge indices (the N+1 mitigation index).
    //    Each hyperedge's index list is in sorted order (canonical iteration).
    let mut entity_to_hedges: Vec<Vec<usize>> = vec![Vec::new(); total_entities];
    for (h_idx, h) in hyperedges.iter().enumerate() {
        for &v in h {
            entity_to_hedges[v].push(h_idx);
        }
    }

    // 4. Initial state + RNG.
    let mut rng = ChaCha8Rng::seed_from_u64(params.rng_seed);
    let mut states = vec![HigherOrderSirState::Susceptible; total_entities];
    seed_initial_infected(&mut states, &entities, &params.seed_strategy, &mut rng);

    // R₀ accounting — only the FIRST ≤5 entities to become infected are
    // tracked, and only their secondary infections during their first 5
    // post-infection steps count toward the average.
    const R0_INDEX_BUDGET: usize = 5;
    const R0_WINDOW: usize = 5;
    let mut r0_index_cases: Vec<usize> = Vec::with_capacity(R0_INDEX_BUDGET);
    let mut r0_secondary_counts: Vec<u32> = Vec::with_capacity(R0_INDEX_BUDGET);
    let mut r0_infected_at_step: HashMap<usize, usize> = HashMap::new();
    for (idx, st) in states.iter().enumerate() {
        if matches!(st, HigherOrderSirState::Infected) && r0_index_cases.len() < R0_INDEX_BUDGET {
            r0_index_cases.push(idx);
            r0_secondary_counts.push(0);
            r0_infected_at_step.insert(idx, 0);
        }
    }

    let mut per_step_infected: Vec<usize> = Vec::with_capacity(params.max_steps + 1);
    let mut per_step_recovered: Vec<usize> = Vec::with_capacity(params.max_steps + 1);
    let mut size_attribution: Vec<u64> = vec![0; params.beta_per_size.len()];

    // Record step 0 (post-seeding, pre-dynamics).
    let (i0, r0c) = count_states(&states);
    per_step_infected.push(i0);
    per_step_recovered.push(r0c);

    // 5. Step loop. Cap at max_steps; early-exit when prevalence is zero AND
    //    all infections have either burned out or recovered (so the trajectory
    //    can't change again).
    for step in 1..=params.max_steps {
        // 5a. Snapshot last step's infected set so a freshly-infected entity
        //     in this step doesn't itself drive other infections in the same
        //     step (synchronous SIR).
        let was_infected: Vec<bool> = states
            .iter()
            .map(|s| matches!(s, HigherOrderSirState::Infected))
            .collect();

        // 5b. Transmission pass — canonical order over susceptibles.
        for v in 0..total_entities {
            if !matches!(states[v], HigherOrderSirState::Susceptible) {
                continue;
            }
            // Iterate hyperedges containing v in canonical (sorted) order.
            for &h_idx in &entity_to_hedges[v] {
                let h = &hyperedges[h_idx];
                let size = h.len();
                if size < 2 {
                    continue;
                }
                // beta lookup — size-d uses index d-2.
                let beta_idx = size - 2;
                if beta_idx >= params.beta_per_size.len() {
                    continue;
                }
                let beta = params.beta_per_size[beta_idx];
                if beta <= 0.0 {
                    continue;
                }
                // Count infected co-members (using last-step snapshot).
                let mut count_infected = 0usize;
                for &u in h {
                    if u != v && was_infected[u] {
                        count_infected += 1;
                    }
                }
                if !params.threshold.met(count_infected, size) {
                    continue;
                }
                // Fire Bernoulli(beta).
                let draw: f32 = rng.gen();
                if draw < beta {
                    states[v] = HigherOrderSirState::Infected;
                    size_attribution[beta_idx] += 1;
                    // R₀ bookkeeping — credit each infected co-member that's
                    // still in the index-case set within their R0_WINDOW.
                    for &u in h {
                        if u == v || !was_infected[u] {
                            continue;
                        }
                        if let Some(&infected_step) = r0_infected_at_step.get(&u) {
                            if step.saturating_sub(infected_step) < R0_WINDOW {
                                if let Some(pos) =
                                    r0_index_cases.iter().position(|&c| c == u)
                                {
                                    r0_secondary_counts[pos] += 1;
                                }
                            }
                        }
                    }
                    if r0_index_cases.len() < R0_INDEX_BUDGET
                        && !r0_index_cases.contains(&v)
                    {
                        r0_index_cases.push(v);
                        r0_secondary_counts.push(0);
                        r0_infected_at_step.insert(v, step);
                    }
                    break;
                }
            }
        }

        // 5c. Recovery pass — canonical order over previously-infected.
        for v in 0..total_entities {
            if was_infected[v] && matches!(states[v], HigherOrderSirState::Infected) {
                let draw: f32 = rng.gen();
                if draw < params.gamma {
                    states[v] = HigherOrderSirState::Recovered;
                }
            }
        }

        let (cur_inf, cur_rec) = count_states(&states);
        per_step_infected.push(cur_inf);
        per_step_recovered.push(cur_rec);
        if cur_inf == 0 {
            break;
        }
    }

    // 6. Summary stats.
    let (peak_prevalence, time_to_peak) = peak_stats(&per_step_infected, total_entities);
    let r0_estimate = if r0_index_cases.is_empty() {
        0.0
    } else {
        let sum: u32 = r0_secondary_counts.iter().sum();
        sum as f32 / r0_index_cases.len() as f32
    };

    Ok(HigherOrderSirResult {
        per_step_infected,
        per_step_recovered,
        r0_estimate,
        time_to_peak,
        peak_prevalence,
        size_attribution,
        total_entities,
        final_states: states,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Empty-narrative early-return shape. Keeps the caller's downstream
/// statistics code from special-casing zero entities.
fn empty_result(params: &HigherOrderSirParams) -> HigherOrderSirResult {
    HigherOrderSirResult {
        per_step_infected: vec![0],
        per_step_recovered: vec![0],
        r0_estimate: 0.0,
        time_to_peak: 0,
        peak_prevalence: 0.0,
        size_attribution: vec![0; params.beta_per_size.len()],
        total_entities: 0,
        final_states: vec![],
    }
}

fn count_states(states: &[HigherOrderSirState]) -> (usize, usize) {
    let mut inf = 0usize;
    let mut rec = 0usize;
    for s in states {
        match s {
            HigherOrderSirState::Infected => inf += 1,
            HigherOrderSirState::Recovered => rec += 1,
            _ => {}
        }
    }
    (inf, rec)
}

fn peak_stats(per_step_infected: &[usize], total: usize) -> (f32, usize) {
    if total == 0 {
        return (0.0, 0);
    }
    let (peak_idx, peak_cnt) = per_step_infected
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .unwrap_or((0, &0));
    (*peak_cnt as f32 / total as f32, peak_idx)
}

fn seed_initial_infected(
    states: &mut [HigherOrderSirState],
    entities: &[crate::types::Entity],
    strategy: &SeedStrategy,
    rng: &mut ChaCha8Rng,
) {
    match strategy {
        SeedStrategy::RandomFraction { fraction } => {
            let n = states.len();
            if n == 0 {
                return;
            }
            let f = fraction.clamp(0.0, 1.0);
            let k = ((f as f64) * (n as f64)).floor() as usize;
            // Fisher-Yates shuffle of indices [0, n) using the same rng.
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            for &chosen in idx.iter().take(k) {
                states[chosen] = HigherOrderSirState::Infected;
            }
        }
        SeedStrategy::Specific { entity_ids } => {
            for eid in entity_ids {
                if let Some(pos) = entities.iter().position(|e| e.id == *eid) {
                    states[pos] = HigherOrderSirState::Infected;
                }
            }
        }
    }
}

// ── Conversion error helper ──────────────────────────────────────────────────

/// Parse a `HigherOrderSirParams` from a JSON blob with reasonable defaults.
/// Used by the engine + the analysis REST handler so both share one source
/// of truth.
pub fn parse_params(value: &serde_json::Value) -> Result<HigherOrderSirParams> {
    if value.is_null() {
        return Err(TensaError::InvalidInput(
            "higher-order contagion: params blob is null".into(),
        ));
    }
    serde_json::from_value(value.clone()).map_err(|e| {
        TensaError::InvalidInput(format!(
            "higher-order contagion: invalid HigherOrderSirParams: {e}"
        ))
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Phase 7b higher-order SIR tests.
    //!
    //! T1 documents the load-bearing **reduction-to-pairwise** contract:
    //! `beta_per_size = [β, 0, 0, ...]` AND `threshold = ThresholdRule::Absolute(1)`
    //! must yield the same dynamics as a pairwise SIR over the same edges.
    //! Future contributors who add new branches MUST preserve T1, or
    //! `analysis::contagion` semantics break.

    use super::*;
    use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};

    /// Pure-Rust pairwise SIR simulator over the same hyperedge substrate
    /// reduced to pairs. Used as the oracle in T1: when the higher-order
    /// model is configured to look only at size-2 transmissions with
    /// threshold "any single infected partner triggers", the two simulators
    /// must agree on the per-step prevalence trajectory.
    ///
    /// This is intentionally a from-scratch reference, NOT a call into
    /// `analysis::contagion::run_contagion` (which is informational-spread
    /// SIR keyed on `InfoSet` — a different abstraction that doesn't use
    /// β/γ probabilities). The contract under test is "the higher-order
    /// engine, restricted to pairs, behaves like pairwise SIR" — the
    /// reference here IS that pairwise SIR.
    fn pairwise_oracle(
        n: usize,
        edges: &[(usize, usize)],
        initial_infected: &[usize],
        beta: f32,
        gamma: f32,
        max_steps: usize,
        rng_seed: u64,
    ) -> Vec<usize> {
        // Build adjacency for the same edges.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in edges {
            if a == b {
                continue;
            }
            if !adj[a].contains(&b) {
                adj[a].push(b);
            }
            if !adj[b].contains(&a) {
                adj[b].push(a);
            }
        }
        for row in &mut adj {
            row.sort();
        }

        let mut states = vec![HigherOrderSirState::Susceptible; n];
        for &s in initial_infected {
            if s < n {
                states[s] = HigherOrderSirState::Infected;
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let mut traj = vec![states
            .iter()
            .filter(|s| matches!(s, HigherOrderSirState::Infected))
            .count()];

        for _ in 1..=max_steps {
            let was_infected: Vec<bool> = states
                .iter()
                .map(|s| matches!(s, HigherOrderSirState::Infected))
                .collect();

            for v in 0..n {
                if !matches!(states[v], HigherOrderSirState::Susceptible) {
                    continue;
                }
                // Walk the v's neighbors in canonical order; for each
                // infected neighbor draw Bernoulli(beta). Match the
                // higher-order engine's "first hyperedge that fires
                // wins" semantics by breaking on success.
                for &u in &adj[v] {
                    if !was_infected[u] {
                        continue;
                    }
                    let draw: f32 = rng.gen();
                    if draw < beta {
                        states[v] = HigherOrderSirState::Infected;
                        break;
                    }
                }
            }

            for v in 0..n {
                if was_infected[v] && matches!(states[v], HigherOrderSirState::Infected) {
                    let draw: f32 = rng.gen();
                    if draw < gamma {
                        states[v] = HigherOrderSirState::Recovered;
                    }
                }
            }

            let cur_inf = states
                .iter()
                .filter(|s| matches!(s, HigherOrderSirState::Infected))
                .count();
            traj.push(cur_inf);
            if cur_inf == 0 {
                break;
            }
        }
        traj
    }

    #[test]
    fn test_higher_order_sir_reduces_to_pairwise_when_only_beta_2_and_threshold_absolute_1() {
        // ── LOAD-BEARING TEST — see module-level "Reduction-to-pairwise contract"
        //
        // Configures the higher-order engine to emulate pairwise SIR by:
        //   beta_per_size = [β, 0, 0, ...]   (all higher-d rates zero)
        //   threshold     = Absolute(1)      (any single infected partner triggers)
        //
        // The oracle is a from-scratch pairwise SIR over the same edge set,
        // seeded with the same ChaCha8Rng. Both consume RNG draws in the
        // same canonical order: per-step, per-susceptible, per-neighbor —
        // first hit wins.
        //
        // FUTURE CONTRIBUTORS: do NOT change the Absolute(1) default here
        // and do NOT introduce a different RNG-consumption order without
        // updating both the engine AND the oracle in lock-step. Breaking
        // this test silently breaks the documented backward-compatibility
        // promise to analysis::contagion users.

        // Build a 2-entity-per-edge narrative: each "situation" has exactly
        // two participants, mimicking pairwise dynamics under the
        // higher-order substrate. Use two situations: (A,B) and (B,C).
        let hg = make_hg();
        let nid = "reduction-pairwise";
        let a = add_entity(&hg, "A", nid);
        let b = add_entity(&hg, "B", nid);
        let c = add_entity(&hg, "C", nid);

        let s_ab = add_situation(&hg, nid);
        link(&hg, a, s_ab);
        link(&hg, b, s_ab);

        let s_bc = add_situation(&hg, nid);
        link(&hg, b, s_bc);
        link(&hg, c, s_bc);

        // Canonical ordering used by simulate_higher_order_sir is sorted by
        // entity Uuid bytes — match the same ordering for the oracle.
        let mut sorted = vec![(a, 0usize), (b, 0usize), (c, 0usize)];
        sorted.sort_by_key(|x| x.0);
        for (i, item) in sorted.iter_mut().enumerate() {
            item.1 = i;
        }
        let idx_a = sorted.iter().find(|x| x.0 == a).unwrap().1;
        let idx_b = sorted.iter().find(|x| x.0 == b).unwrap().1;
        let idx_c = sorted.iter().find(|x| x.0 == c).unwrap().1;

        // Same edges, expressed in oracle index space. The higher-order
        // engine canonicalizes each hyperedge's member list by sorted
        // index; mirror that.
        let mut edge1 = [idx_a, idx_b];
        edge1.sort();
        let mut edge2 = [idx_b, idx_c];
        edge2.sort();
        let edges = vec![(edge1[0], edge1[1]), (edge2[0], edge2[1])];

        let beta = 0.6_f32;
        let gamma = 0.0_f32;
        let max_steps = 10;
        let rng_seed = 42_u64;

        // Seed B as the initial infected — matches threshold_absolute_1 +
        // beta_size_2 path: B can transmit to A via edge1 and to C via edge2.
        let initial = vec![idx_b];

        let params = HigherOrderSirParams {
            beta_per_size: vec![beta, 0.0, 0.0, 0.0],
            gamma,
            threshold: ThresholdRule::Absolute(1),
            seed_strategy: SeedStrategy::Specific {
                entity_ids: vec![sorted[idx_b].0],
            },
            max_steps,
            rng_seed,
        };

        let actual = simulate_higher_order_sir(&hg, nid, &params).expect("ho sim ok");
        let expected = pairwise_oracle(3, &edges, &initial, beta, gamma, max_steps, rng_seed);

        assert_eq!(
            actual.per_step_infected, expected,
            "higher-order SIR with [β, 0, 0, ...] + Absolute(1) MUST match \
             pairwise SIR per-step prevalence. Reduction contract broken — \
             see module-level docs."
        );

        // Every transmission must be credited to the size-2 attribution slot.
        let total_attr: u64 = actual.size_attribution.iter().sum();
        assert_eq!(
            total_attr, actual.size_attribution[0],
            "all transmissions in the reduced setting must come from size-2 \
             hyperedges (beta_per_size[0])"
        );
    }

    #[test]
    fn test_higher_order_sir_threshold_blocks_infection_below_count() {
        // 1 hyperedge of size 4: {A, B, C, D}. Seed only A infected.
        // With threshold Absolute(2) and beta_3=1.0 (the size-4 rate),
        // there is only 1 infected co-member visible to B/C/D — below
        // threshold — so NOBODY should ever get infected.
        let hg = make_hg();
        let nid = "threshold-block";
        let a = add_entity(&hg, "A", nid);
        let b = add_entity(&hg, "B", nid);
        let c = add_entity(&hg, "C", nid);
        let d = add_entity(&hg, "D", nid);

        let sid = add_situation(&hg, nid);
        for &e in &[a, b, c, d] {
            link(&hg, e, sid);
        }

        let params = HigherOrderSirParams {
            // size 2 = 1.0 (would normally infect everyone instantly), but
            // threshold below blocks it; also size 3 = 1.0; size 4 = 1.0.
            beta_per_size: vec![1.0, 1.0, 1.0],
            gamma: 0.0,
            threshold: ThresholdRule::Absolute(2),
            seed_strategy: SeedStrategy::Specific {
                entity_ids: vec![a],
            },
            max_steps: 20,
            rng_seed: 1,
        };

        let result = simulate_higher_order_sir(&hg, nid, &params).unwrap();

        // Only A stays infected forever; nobody else can cross the threshold
        // since each edge has only 1 infected co-member from the perspective
        // of B/C/D.
        let final_infected = result
            .final_states
            .iter()
            .filter(|s| matches!(s, HigherOrderSirState::Infected))
            .count();
        assert_eq!(
            final_infected, 1,
            "threshold of 2 with only 1 seed must block all secondary infections"
        );
        assert_eq!(
            result.size_attribution.iter().sum::<u64>(),
            0,
            "no transmissions should be credited"
        );
    }

    #[test]
    fn test_higher_order_sir_size_attribution_sums_to_total_infections() {
        // Build a narrative where transmission can occur across multiple sizes.
        // Two hyperedges: a 2-edge {A, B} and a 4-edge {C, D, E, F}.
        // Seed A and C. Use thresholds + βs that allow both edges to fire.
        let hg = make_hg();
        let nid = "size-attribution";
        let a = add_entity(&hg, "A", nid);
        let b = add_entity(&hg, "B", nid);
        let c = add_entity(&hg, "C", nid);
        let d = add_entity(&hg, "D", nid);
        let e = add_entity(&hg, "E", nid);
        let f = add_entity(&hg, "F", nid);

        let s2 = add_situation(&hg, nid);
        link(&hg, a, s2);
        link(&hg, b, s2);

        let s4 = add_situation(&hg, nid);
        for &x in &[c, d, e, f] {
            link(&hg, x, s4);
        }

        let params = HigherOrderSirParams {
            beta_per_size: vec![1.0, 0.0, 1.0], // size 2 + size 4 fire deterministically
            gamma: 0.0,
            threshold: ThresholdRule::Absolute(1),
            seed_strategy: SeedStrategy::Specific {
                entity_ids: vec![a, c],
            },
            max_steps: 25,
            rng_seed: 99,
        };

        let result = simulate_higher_order_sir(&hg, nid, &params).unwrap();

        // Total transmissions = (final infected + final recovered) - initial seeds.
        let final_infected_or_recovered = result
            .final_states
            .iter()
            .filter(|s| !matches!(s, HigherOrderSirState::Susceptible))
            .count();
        let total_secondary = final_infected_or_recovered.saturating_sub(2); // 2 seeds
        let total_attribution: u64 = result.size_attribution.iter().sum();
        assert_eq!(
            total_attribution as usize, total_secondary,
            "size_attribution sum ({total_attribution}) must equal the number of \
             secondary infections ({total_secondary})"
        );
        // And each non-zero slot lines up with an edge size that exists.
        assert!(
            result.size_attribution[0] > 0 || result.size_attribution[2] > 0,
            "at least one of the size-2 or size-4 attributions must be non-zero"
        );
    }

}
