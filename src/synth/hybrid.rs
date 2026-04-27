//! Hybrid hypergraph generation (EATH Phase 9).
//!
//! Implements the **mixture-distribution** hybrid algorithm (Mancastroppa et
//! al. §III.4 approach a): at each step, sample a source from a weighted
//! multinomial, then recruit ONE complete hyperedge using that source's
//! calibrated [`EathParams`]. The output is a mixture of `n` independent EATH
//! processes, NOT a statistic interpolation and NOT a per-entity partition.
//!
//! See [`docs/synth_hybrid_semantics.md`](../../docs/synth_hybrid_semantics.md)
//! for the full design — worked example, rejected alternatives, and rationale
//! for shared STM + per-source LTM.
//!
//! ## Reuse
//!
//! Phase 1's `recruit_from_scratch` / `recruit_from_memory` (in
//! [`super::eath_recruit`]) and `ActivityField::step_transition` /
//! `draw_num_groups` / `sample_group_size` (in [`super::activity`]) are reused
//! verbatim. Phase 3's emit pipeline (in [`super::emit`]) is the only writer
//! path. The hybrid module multiplexes between calls into those primitives;
//! it never reimplements the EATH algorithm.

use chrono::{TimeZone, Utc};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::MaturityLevel;

use super::activity::{draw_num_groups, sample_group_size, ActivityField};
use super::calibrate::load_params;
use super::eath_recruit::{recruit_from_memory, recruit_from_scratch};
use super::emit::{write_synthetic_entities, write_synthetic_situation, EmitContext};
use super::memory::{LongTermMemory, RecentGroup, ShortTermMemory};
use super::registry::SurrogateRegistry;
use super::types::{EathParams, RunKind, SurrogateParams, SurrogateRunSummary};

// ── Public types ─────────────────────────────────────────────────────────────

/// One component of a hybrid generation: which source narrative + model to
/// pull calibrated params from, and the mixture weight (`Σ weight = 1.0`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HybridComponent {
    /// Source narrative whose calibrated params drive this component.
    pub narrative_id: String,
    /// Surrogate model name (typically `"eath"`).
    pub model: String,
    /// Mixture weight in `[0, 1]`. Sum across components must equal 1.0
    /// within [`HYBRID_WEIGHT_TOLERANCE`].
    pub weight: f32,
}

/// Inputs to [`generate_hybrid_hypergraph`]. Carried as a separate struct so
/// future fields (e.g. `label_prefix`) can land without breaking the function
/// signature — currently only used by tests for ergonomic construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridParams {
    pub sources: Vec<HybridComponent>,
    pub seed: u64,
    pub num_steps: usize,
}

// ── Constants ────────────────────────────────────────────────────────────────

/// Max allowed deviation of `Σ weight` from 1.0. Same tolerance the
/// calibrator's NaN sweep uses — tight enough to catch input mistakes but
/// loose enough for JSON-roundtrip / Studio-slider noise.
pub const HYBRID_WEIGHT_TOLERANCE: f32 = 1e-6;

/// Sentinel narrative-id that hybrid runs use as the "source" for the
/// `EmitContext` label prefix. Real lineage lives in the components manifest
/// (see [`canonical_hybrid_params`]).
const HYBRID_LABEL_PREFIX: &str = "synth-hybrid-actor-";

/// Synthetic-epoch anchor for `temporal.start` on emitted situations. Same
/// constant the EATH single-source generator uses so the two are
/// interchangeable in downstream temporal-ordering code.
fn synth_epoch() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Generate a synthetic narrative as a weighted mixture of `n` EATH processes.
///
/// See [`docs/synth_hybrid_semantics.md`](../../docs/synth_hybrid_semantics.md)
/// for the algorithm. Per-source LTM, shared STM, single main RNG. Emits one
/// `SurrogateRunSummary` (kind = `Hybrid`) plus a `ReproducibilityBlob` whose
/// `params_full.params_json` is the components manifest.
///
/// Validation:
/// 1. `components.is_empty()` → error.
/// 2. `Σ weight - 1.0` outside `±HYBRID_WEIGHT_TOLERANCE` → error.
/// 3. Any `weight < 0` or non-finite → error.
/// 4. Any source's params absent (un-calibrated) → error.
///
/// Determinism: identical `seed` + identical loaded `EathParams` for every
/// source ⇒ identical output (entities, situations, participations).
pub fn generate_hybrid_hypergraph(
    components: &[HybridComponent],
    output_narrative_id: &str,
    seed: u64,
    num_steps: usize,
    hypergraph: &Hypergraph,
    _registry: &SurrogateRegistry,
) -> Result<SurrogateRunSummary> {
    // 1. Validate inputs eagerly — fail BEFORE any KV write.
    if output_narrative_id.is_empty() {
        return Err(TensaError::InvalidInput(
            "generate_hybrid_hypergraph: output_narrative_id is empty".into(),
        ));
    }
    if components.is_empty() {
        return Err(TensaError::InvalidInput(
            "generate_hybrid_hypergraph: components list is empty".into(),
        ));
    }
    validate_weights(components)?;

    // 2. Load each source's calibrated params.
    let mut per_source_params: Vec<EathParams> = Vec::with_capacity(components.len());
    for c in components {
        let params = load_params(hypergraph.store(), &c.narrative_id, &c.model)?
            .ok_or_else(|| {
                TensaError::SynthFailure(format!(
                    "no calibrated params for ('{}', '{}'); calibrate the source first",
                    c.narrative_id, c.model
                ))
            })?;
        per_source_params.push(params);
    }

    // 3. Determine the entity count: take the MAX num_entities across sources
    //    (using `a_t_distribution.len()` as authoritative). Sources with
    //    smaller a_t_distribution have their per-entity vectors logically
    //    padded with zeros at the tail; the recruitment helpers' bounds
    //    checks treat OOR indices as zero-weight, so this is safe.
    //    Cap at the wargame-substrate inline budget — Phase 9 spec §risks.
    let n_max = per_source_params
        .iter()
        .map(|p| p.a_t_distribution.len())
        .max()
        .unwrap_or(0);
    if n_max < 2 {
        return Err(TensaError::SynthFailure(
            "every source's a_t_distribution.len() < 2; cannot generate".into(),
        ));
    }

    // 4. Mint a deterministic run_id from the seed (NOT wall-clock UUIDv7 —
    //    we need same-seed → same-uuid for the determinism contract).
    let run_id = run_id_from_seed(seed);
    let started_at = Utc::now();

    // 5. Build canonical SurrogateParams envelope for the run summary +
    //    reproducibility blob. The params_json is the components manifest;
    //    seed + num_steps replay through `generate_hybrid_hypergraph`.
    let canonical = canonical_hybrid_params(components, seed, num_steps);
    let params_hash = super::hashing::canonical_params_hash(&canonical);

    // 6. Persist ReproducibilityBlob BEFORE the loop — Phase 1 contract that
    //    a crashed run is still reproducible from inputs alone.
    let blob = super::build_reproducibility_blob(
        run_id,
        canonical.clone(),
        // Hybrids have no single source narrative; source_state_hash is None.
        None,
    );
    super::store_reproducibility_blob(hypergraph.store(), &blob)?;

    // 7. Emit context — single source of truth for provenance metadata
    //    threaded into every emitted entity / situation / participation.
    let ctx = EmitContext {
        run_id,
        narrative_id: output_narrative_id.to_string(),
        maturity: MaturityLevel::Candidate,
        // Match EATH single-source generator: explicit confidence so
        // determinism tests assert exact bit-for-bit reproducibility.
        confidence: 1.0,
        label_prefix: HYBRID_LABEL_PREFIX.to_string(),
        time_anchor: synth_epoch(),
        step_duration_seconds: 1,
        // Provenance tag: the output entities/situations carry "hybrid" as
        // the model. The components manifest in the ReproducibilityBlob is
        // the source of truth for which sources contributed.
        model: "hybrid".to_string(),
        // Hybrid mints fresh synthetic entities (same as EATH); only
        // configuration-style models (Phase 13b NuDHy) reuse source entities.
        reuse_entities: None,
    };

    // 8. Pre-build entity UUIDs from the seed, then emit the entity records.
    //    Dedicated entity sub-RNG keeps emit's UUID draws independent of the
    //    main simulation stream (same as EathSurrogate).
    let mut entity_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xE1717));
    let entity_ids = write_synthetic_entities(&ctx, n_max, &mut entity_rng, hypergraph)?;

    // 9. Per-source state. Each source has its own ActivityField + LTM +
    //    pre-computed normalisation constants.
    let mut per_source_field: Vec<ActivityField> = per_source_params
        .iter()
        .map(|_p| ActivityField::new(n_max))
        .collect();
    let mut per_source_ltm: Vec<LongTermMemory> =
        components.iter().map(|_| LongTermMemory::new()).collect();
    let per_source_mean_a_t: Vec<f32> = per_source_params
        .iter()
        .map(|p| {
            let n = p.a_t_distribution.len();
            if n == 0 {
                0.0
            } else {
                p.a_t_distribution.iter().sum::<f32>() / n as f32
            }
        })
        .collect();
    let per_source_mean_total_activity: Vec<f32> = per_source_params
        .iter()
        .map(|p| p.a_h_distribution.iter().sum::<f32>().max(1e-9))
        .collect();

    // 10. Pre-compute weight-vector cumulative sum for the per-step source
    //     pick — done once, reused every step. Stored as f32 to match RNG
    //     output type and avoid one f32→f64 cast per step.
    let cumulative_weights: Vec<f32> = {
        let mut acc = 0.0_f32;
        components
            .iter()
            .map(|c| {
                acc += c.weight;
                acc
            })
            .collect()
    };

    // 11. Shared state: STM + main RNG. STM capacity = max across sources so
    //     no source's reactivation window gets clipped.
    let stm_capacity = components
        .iter()
        .enumerate()
        .map(|(k, _)| per_source_params[k].stm_capacity)
        .max()
        .unwrap_or(7);
    let mut stm = ShortTermMemory::new(stm_capacity);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // 12. Scratch buffers reused across steps (avoid per-step allocation).
    //     Sized for the max group size across sources.
    let max_group_size = per_source_params
        .iter()
        .map(|p| p.max_group_size)
        .max()
        .unwrap_or(10);
    let mut weight_buf: Vec<f32> = vec![0.0; n_max];
    let mut member_indices: Vec<usize> = Vec::with_capacity(max_group_size);
    let mut member_uuids: Vec<Uuid> = Vec::with_capacity(max_group_size);

    let mut num_situations = 0_usize;
    let mut num_participations = 0_usize;

    // 13. Main loop.
    for tick in 0..num_steps {
        // 13a. Pick the source for this step. ALWAYS consume one f32 draw
        //      so determinism alignment doesn't depend on weight pattern.
        let source_pick: f32 = rng.gen();
        let source_k = pick_source(&cumulative_weights, source_pick);
        let params_k = &per_source_params[source_k];

        // 13b. Step the chosen source's activity field.
        let _lambda_t = per_source_field[source_k].step_transition(
            &mut rng,
            params_k,
            tick as u64,
            per_source_mean_a_t[source_k],
        );

        // 13c. Group count + per-group recruitment, all driven by source k.
        let total_activity = per_source_field[source_k].total_activity();
        let n_for_source = params_k.a_t_distribution.len().min(n_max);
        let num_groups = draw_num_groups(
            params_k,
            total_activity,
            per_source_mean_total_activity[source_k],
            n_for_source,
            &mut rng,
        );

        for _ in 0..num_groups {
            let group_size = sample_group_size(params_k, &mut rng);
            let coin: f32 = rng.gen();
            let from_scratch = stm.is_empty() || coin < params_k.p_from_scratch;

            member_indices.clear();
            // Recruit only over the source's own entity prefix [0, n_for_source).
            // Pad weight_buf to that length before each call so the
            // recruitment helper's bounds checks see the right slice.
            // (The helpers resize to n internally, so this is just for the
            // per-source-vector access at index lookup time.)
            if from_scratch {
                recruit_from_scratch(
                    group_size,
                    &per_source_field[source_k].current_activity,
                    params_k,
                    &per_source_ltm[source_k],
                    tick as u64,
                    n_for_source,
                    &mut weight_buf,
                    &mut member_indices,
                    &mut rng,
                );
            } else {
                recruit_from_memory(
                    group_size,
                    &stm,
                    &per_source_field[source_k].current_activity,
                    params_k,
                    &per_source_ltm[source_k],
                    tick as u64,
                    n_for_source,
                    &mut weight_buf,
                    &mut member_indices,
                    &mut rng,
                );
            }

            if member_indices.is_empty() {
                continue;
            }

            // Resolve indices → UUIDs and emit through the standard pipeline.
            member_uuids.clear();
            for &idx in &member_indices {
                if idx < entity_ids.len() {
                    member_uuids.push(entity_ids[idx]);
                }
            }
            if member_uuids.is_empty() {
                continue;
            }
            let _sit_id =
                write_synthetic_situation(&ctx, tick, &member_uuids, &mut rng, hypergraph)?;

            // Update memory: shared STM (every source contributes) +
            // per-source LTM (only source k advances its decay clock).
            stm.push(RecentGroup {
                members: member_indices.clone(),
                age: 0,
            });
            per_source_ltm[source_k].record_group(&member_indices, tick as u64);
            num_situations += 1;
            num_participations += member_uuids.len();
        }

        stm.age_all();
    }

    // 14. Build the run summary.
    let finished_at = Utc::now();
    let summary = SurrogateRunSummary {
        run_id,
        model: "hybrid".into(),
        params_hash,
        // Hybrids have no single source narrative — components manifest in
        // the ReproducibilityBlob is the source of truth.
        source_narrative_id: None,
        source_state_hash: None,
        output_narrative_id: output_narrative_id.into(),
        num_entities: entity_ids.len(),
        num_situations,
        num_participations,
        started_at,
        finished_at,
        duration_ms: (finished_at - started_at).num_milliseconds().max(0) as u64,
        kind: RunKind::Hybrid,
    };

    let key = super::key_synth_run(output_narrative_id, &run_id);
    hypergraph
        .store()
        .put(&key, &serde_json::to_vec(&summary)?)?;
    super::record_lineage_run(hypergraph.store(), output_narrative_id, &run_id)?;

    Ok(summary)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn validate_weights(components: &[HybridComponent]) -> Result<()> {
    let mut sum = 0.0_f32;
    for (i, c) in components.iter().enumerate() {
        if !c.weight.is_finite() {
            return Err(TensaError::InvalidInput(format!(
                "hybrid component {i} ('{}'): weight is NaN/Inf",
                c.narrative_id
            )));
        }
        if c.weight < 0.0 {
            return Err(TensaError::InvalidInput(format!(
                "hybrid component {i} ('{}'): weight {} < 0",
                c.narrative_id, c.weight
            )));
        }
        if c.narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(format!(
                "hybrid component {i}: narrative_id is empty"
            )));
        }
        if c.model.is_empty() {
            return Err(TensaError::InvalidInput(format!(
                "hybrid component {i} ('{}'): model is empty",
                c.narrative_id
            )));
        }
        sum += c.weight;
    }
    if (sum - 1.0).abs() > HYBRID_WEIGHT_TOLERANCE {
        return Err(TensaError::InvalidInput(format!(
            "hybrid weights must sum to 1.0 within ±{HYBRID_WEIGHT_TOLERANCE} (got {sum})"
        )));
    }
    Ok(())
}

/// Build the canonical `SurrogateParams` envelope for a hybrid run. The
/// `params_json` is a hybrid manifest object — distinct from a single-source
/// `EathParams` blob — so consumers (Studio, replay tooling, fidelity
/// readers) can branch on `params_json.type == "hybrid"`.
pub(crate) fn canonical_hybrid_params(
    components: &[HybridComponent],
    seed: u64,
    num_steps: usize,
) -> SurrogateParams {
    let manifest = serde_json::json!({
        "type": "hybrid",
        "sources": components,
        "num_steps": num_steps,
    });
    SurrogateParams {
        model: "hybrid".to_string(),
        params_json: manifest,
        seed,
        num_steps,
        label_prefix: "synth-hybrid".to_string(),
    }
}

/// Pick a source index given the cumulative-weight prefix sum and a uniform
/// random in `[0, 1)`. Returns the last index when `coin >= cumulative.last()`
/// (defensive fallback for floating-point drift).
fn pick_source(cumulative: &[f32], coin: f32) -> usize {
    for (k, &c) in cumulative.iter().enumerate() {
        if coin < c {
            return k;
        }
    }
    cumulative.len().saturating_sub(1)
}

/// Mint a run_id deterministically from `seed`. Distinct sub-RNG so this
/// doesn't consume from the main simulation stream.
fn run_id_from_seed(seed: u64) -> Uuid {
    // Constant differs from EATH single-source so hybrid + EATH from the
    // same seed get distinct run_ids.
    const HYBRID_RUN_ID_MIX: u64 = 0xAAAA_5555_AAAA_5555;
    let mut sub = ChaCha8Rng::seed_from_u64(seed ^ HYBRID_RUN_ID_MIX);
    let mut bytes = [0u8; 16];
    sub.fill_bytes(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
    Uuid::from_bytes(bytes)
}

#[cfg(test)]
#[path = "hybrid_tests.rs"]
mod hybrid_tests;
