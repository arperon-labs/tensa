//! `NudhySurrogate` — `SurrogateModel` impl for the configuration-style
//! null model (Phase 13b).
//!
//! Calibration is a pure read of the source narrative: extract every
//! hyperedge (situation → participant set) and store it as the initial MCMC
//! state. Generation runs the Chodrow 2020 §3.2 double-edge swap chain to
//! convergence, then emits each post-MCMC hyperedge as a synthetic Situation
//! that REUSES the source entities (no new Entity records).
//!
//! Reference: `docs/synth_nudhy_algorithm.md` (Phase 13a architect output).
//!
//! Cited papers:
//! - Preti, Fazzone, Petri, De Francisci Morales — Phys. Rev. X **14**,
//!   031032 (2024). NuDHy lineage (directed analog).
//! - Chodrow — J. Complex Networks **8**: cnaa018 (2020). Undirected
//!   double-edge-swap construction TENSA actually implements.

use std::collections::HashSet;

use chrono::{TimeZone, Utc};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::analysis::graph_projection::collect_participation_index;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::MaturityLevel;

use super::emit::{
    filter_synthetic_entities, filter_synthetic_situations, write_synthetic_situation,
    EmitContext, DEFAULT_SYNTHETIC_CONFIDENCE,
};
use super::nudhy::{run_nudhy_chain, NudhyParams, NudhyState};
use super::surrogate::SurrogateModel;
use super::types::{RunKind, SurrogateParams, SurrogateRunSummary};

// ── Surrogate ────────────────────────────────────────────────────────────────

/// Configuration-model null-model surrogate (NuDHy-flavour).
///
/// Zero-sized — every per-run param lives in
/// [`SurrogateParams::params_json`] (specifically [`NudhyParams`]) so multiple
/// workers may share one `Arc<NudhySurrogate>` without contention.
pub struct NudhySurrogate;

impl SurrogateModel for NudhySurrogate {
    fn name(&self) -> &'static str {
        "nudhy"
    }

    fn version(&self) -> &'static str {
        // v1.0 — initial release of the configuration-style null model.
        "v1.0"
    }

    fn calibrate(
        &self,
        hypergraph: &Hypergraph,
        narrative_id: &str,
    ) -> Result<serde_json::Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::SynthFailure(
                "NudhySurrogate::calibrate: narrative_id is empty".into(),
            ));
        }
        let params = calibrate_nudhy(hypergraph, narrative_id)?;
        Ok(serde_json::to_value(params)?)
    }

    fn generate(
        &self,
        params: &SurrogateParams,
        target: &Hypergraph,
        output_narrative_id: &str,
    ) -> Result<SurrogateRunSummary> {
        run_generate(params, target, output_narrative_id)
    }

    fn fidelity_metrics(&self) -> Vec<&'static str> {
        // Per design doc §10. Note the inverted interpretation: the first two
        // are EXACT invariants (Spearman ρ = 1.0); the third is a measure of
        // successful randomisation (KS expected non-zero).
        vec![
            "degree_sequence_preservation",
            "edge_size_sequence_preservation",
            "entity_pair_overlap_divergence",
        ]
    }
}

// ── Calibration ──────────────────────────────────────────────────────────────

/// Trivial calibration — read source hyperedges into a [`NudhyParams`] blob.
///
/// Per design doc §6: list non-synthetic situations + entities for the
/// narrative, build the participation index via the shared
/// [`collect_participation_index`] helper (single pass, no N+1), then split
/// off size-1 edges into `fixed_edges_json` (they can't participate in any
/// valid swap — see §9.4).
pub(crate) fn calibrate_nudhy(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<NudhyParams> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let situations = filter_synthetic_situations(situations, false);
    if situations.len() < 2 {
        return Err(TensaError::SynthFailure(format!(
            "NuDHy: narrative '{narrative_id}' has {} non-synthetic situations; \
             need >= 2 hyperedges for any MCMC swap",
            situations.len()
        )));
    }

    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entities = filter_synthetic_entities(entities, false);
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    let entity_idx: std::collections::HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let sit_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();

    // Single-pass participation index (situation → [entity_idx]).
    let sit_participants =
        collect_participation_index(hypergraph, &entity_idx, &sit_ids, None)?;

    // Build hyperedge list, mapping back to entity UUIDs in deterministic
    // order — `collect_participation_index` returns indices in scan order,
    // so we sort each edge for canonical NuDHy state.
    let hyperedges: Vec<Vec<Uuid>> = sit_ids
        .iter()
        .map(|sid| {
            let mut v: Vec<Uuid> = match sit_participants.get(sid) {
                Some(idxs) => idxs.iter().map(|&i| entity_ids[i]).collect(),
                None => Vec::new(),
            };
            v.sort();
            v.dedup();
            v
        })
        .collect();

    // Partition: |e| >= 2 goes into the chain; |e| == 1 goes into fixed
    // edges. |e| == 0 is dropped (per §9.5 — pathological data quality).
    let mut chain_edges = Vec::new();
    let mut fixed_edges = Vec::new();
    let mut empty_count = 0usize;
    for e in hyperedges {
        match e.len() {
            0 => empty_count += 1,
            1 => fixed_edges.push(e),
            _ => chain_edges.push(e),
        }
    }
    if empty_count > 0 {
        tracing::debug!(
            empty_count,
            "NuDHy calibration: dropped {} empty hyperedge(s)",
            empty_count
        );
    }
    if !fixed_edges.is_empty() {
        tracing::warn!(
            fixed = fixed_edges.len(),
            "NuDHy calibration: {} size-1 hyperedges excluded from MCMC chain \
             (will be emitted unchanged at generation)",
            fixed_edges.len()
        );
    }
    if chain_edges.len() < 2 {
        return Err(TensaError::SynthFailure(format!(
            "NuDHy: narrative '{narrative_id}' has {} chain-eligible hyperedges \
             (size >= 2); need >= 2 for any MCMC swap",
            chain_edges.len()
        )));
    }

    // Detect fixed-point ensemble (every chain edge identical) — warn but
    // do not error; the source IS a valid member of the ensemble (§9.2).
    let unique: HashSet<Vec<Uuid>> = chain_edges.iter().cloned().collect();
    if unique.len() == 1 {
        tracing::warn!(
            edges = chain_edges.len(),
            "NuDHy: all {} chain-eligible hyperedges are identical; chain is a \
             fixed point. Output will be identical to source. Statistically valid \
             but may surprise callers.",
            chain_edges.len()
        );
    }

    let state = NudhyState::from_hyperedges(chain_edges);
    NudhyParams::from_source_state(&state, fixed_edges)
}

// ── Generation ───────────────────────────────────────────────────────────────

/// Synthetic-epoch anchor for `temporal.start` on emitted situations.
/// 2020-01-01T00:00:00Z; one situation per "tick" (1 second by default).
/// Same anchor EATH uses — keeps temporal-ordering tests consistent.
fn synth_epoch() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
}

/// Mint a deterministic run_id from `seed` (matches `eath.rs::run_id_from_seed`
/// but XORs against a different mix constant so NuDHy + EATH runs with the
/// same seed don't collide).
fn run_id_from_seed(seed: u64) -> Uuid {
    // Distinct from EATH's 0xFFFF_FFFF_0000_0000 — same-seed runs across
    // models produce different run_ids by construction.
    const NUDHY_RUN_ID_MIX: u64 = 0xA5A5_A5A5_5A5A_5A5A;
    let mut sub = ChaCha8Rng::seed_from_u64(seed ^ NUDHY_RUN_ID_MIX);
    let mut bytes = [0u8; 16];
    sub.fill_bytes(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
    Uuid::from_bytes(bytes)
}

/// Generate one synthetic narrative under `output_narrative_id`. Per design
/// doc §7: parse params → build initial state → run MCMC chain → emit
/// situations referencing the source entities (no fresh `Entity` records).
fn run_generate(
    params: &SurrogateParams,
    target: &Hypergraph,
    output_narrative_id: &str,
) -> Result<SurrogateRunSummary> {
    // 1. Parse model-specific params.
    let nudhy: NudhyParams = serde_json::from_value(params.params_json.clone())
        .map_err(|e| TensaError::SynthFailure(format!("NudhyParams parse: {e}")))?;

    // 2. Reconstruct chain-eligible edges + fixed (size-1) edges from JSON.
    let chain_edges: Vec<Vec<Uuid>> =
        serde_json::from_value(nudhy.initial_state_json.clone())
            .map_err(|e| TensaError::SynthFailure(format!("initial_state_json parse: {e}")))?;
    if chain_edges.len() < 2 {
        return Err(TensaError::SynthFailure(format!(
            "NuDHy generate: initial_state_json has {} edges; need >= 2",
            chain_edges.len()
        )));
    }
    let fixed_edges: Vec<Vec<Uuid>> = if nudhy.fixed_edges_json.is_null() {
        Vec::new()
    } else {
        serde_json::from_value(nudhy.fixed_edges_json.clone())
            .map_err(|e| TensaError::SynthFailure(format!("fixed_edges_json parse: {e}")))?
    };

    // 3. Mint deterministic run_id.
    let run_id = run_id_from_seed(params.seed);
    let started_at = Utc::now();

    // 4. Persist ReproducibilityBlob BEFORE any data writes.
    let blob = super::build_reproducibility_blob(run_id, params.clone(), None);
    super::store_reproducibility_blob(target.store(), &blob)?;

    // 5. Build EmitContext with reuse_entities = Some(union of all member UUIDs).
    let mut universe: HashSet<Uuid> = HashSet::new();
    for e in &chain_edges {
        universe.extend(e.iter().copied());
    }
    for e in &fixed_edges {
        universe.extend(e.iter().copied());
    }
    let mut all_entity_ids: Vec<Uuid> = universe.into_iter().collect();
    all_entity_ids.sort(); // deterministic ordering for downstream consumers.

    let ctx = EmitContext {
        run_id,
        narrative_id: output_narrative_id.to_string(),
        maturity: MaturityLevel::Candidate,
        confidence: DEFAULT_SYNTHETIC_CONFIDENCE,
        label_prefix: format!("{}-", params.label_prefix),
        time_anchor: synth_epoch(),
        step_duration_seconds: 1,
        model: "nudhy".to_string(),
        // Phase 13b — preserve node identity (configuration-model semantics).
        // write_synthetic_entities is NEVER called from this path.
        reuse_entities: Some(all_entity_ids.clone()),
    };

    // 6. Run MCMC chain.
    let initial_state = NudhyState::from_hyperedges(chain_edges);
    let mut chain_rng = ChaCha8Rng::seed_from_u64(params.seed);
    let final_state = run_nudhy_chain(initial_state, &nudhy, &mut chain_rng)?;

    // 7. Separate sub-RNG for situation UUID minting (keeps MCMC stream pure).
    let mut sit_rng = ChaCha8Rng::seed_from_u64(params.seed.wrapping_add(0xC0FFEE));

    let mut num_situations = 0usize;
    let mut num_participations = 0usize;

    // 7a. Emit chain hyperedges (MCMC-randomised).
    for (step, members) in final_state.hyperedges.iter().enumerate() {
        write_synthetic_situation(&ctx, step, members, &mut sit_rng, target)?;
        num_situations += 1;
        num_participations += members.len();
    }

    // 7b. Emit fixed (size-1) hyperedges unchanged.
    let step_offset = final_state.hyperedges.len();
    for (i, members) in fixed_edges.iter().enumerate() {
        write_synthetic_situation(&ctx, step_offset + i, members, &mut sit_rng, target)?;
        num_situations += 1;
        num_participations += members.len();
    }

    let finished_at = Utc::now();
    let params_hash = super::hashing::canonical_params_hash(params);
    let summary = SurrogateRunSummary {
        run_id,
        model: "nudhy".into(),
        params_hash,
        // NuDHy doesn't carry a source narrative through the SurrogateModel
        // trait API (per §14 Q3). Configuration-style provenance is recorded
        // in the ReproducibilityBlob via `params_full.params_json`.
        source_narrative_id: None,
        source_state_hash: None,
        output_narrative_id: output_narrative_id.into(),
        // Configuration models reuse source entities — they do NOT mint new
        // Entity records. Report 0 for this run's contribution to the
        // entity count (the entities already exist under the source narrative).
        num_entities: 0,
        num_situations,
        num_participations,
        started_at,
        finished_at,
        duration_ms: (finished_at - started_at).num_milliseconds().max(0) as u64,
        kind: RunKind::Generation,
    };

    let key = super::key_synth_run(output_narrative_id, &run_id);
    target.store().put(&key, &serde_json::to_vec(&summary)?)?;
    super::record_lineage_run(target.store(), output_narrative_id, &run_id)?;

    Ok(summary)
}

// ── Tests (T6, T8) ───────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "nudhy_surrogate_tests.rs"]
mod nudhy_surrogate_tests;
