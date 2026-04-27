//! EATH — Effective Active Temporal Hypergraph surrogate model.
//!
//! Phase 1 ships the algorithm core: per-tick activity dynamics, group
//! recruitment with short- and long-term memory, deterministic UUID
//! construction, and inline emit shortcuts for end-to-end testing without the
//! full `emit.rs` pipeline (Phase 3 owns that).
//!
//! See `docs/synth_eath_algorithm.md` for the algorithmic specification —
//! this file follows §2 (algorithm) + §3 (determinism) + §4 (data structures)
//! + §6 (Phase 0 integration) + §8 (inline emit) of that doc.
//!
//! Reference: Mancastroppa, Cencetti, Barrat — arXiv:2507.01124v2.

use chrono::{TimeZone, Utc};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

use super::activity::{draw_num_groups, sample_group_size, ActivityField};
use super::eath_recruit::{recruit_from_memory, recruit_from_scratch};
use super::emit::{
    write_synthetic_entities, write_synthetic_situation, EmitContext,
    DEFAULT_SYNTHETIC_CONFIDENCE,
};
use super::memory::{LongTermMemory, RecentGroup, ShortTermMemory};
use super::surrogate::SurrogateModel;
use super::types::{EathParams, RunKind, SurrogateParams, SurrogateRunSummary};
use crate::types::MaturityLevel;

// ── Constants ────────────────────────────────────────────────────────────────

/// Knuth multiplicative-hash golden-ratio constant. Used by entity-UUID
/// per-index seeding so adjacent entity indices map to well-dispersed bytes.
const RNG_SUBSEED_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

// ── EathSurrogate ────────────────────────────────────────────────────────────

/// Effective Active Temporal Hypergraph generator.
///
/// Zero-sized — all per-run state lives in `SurrogateParams.params_json`
/// (specifically [`super::types::EathParams`]) so multiple workers can share
/// one `Arc<EathSurrogate>` without contention.
pub struct EathSurrogate;

impl SurrogateModel for EathSurrogate {
    fn name(&self) -> &'static str {
        "eath"
    }

    fn version(&self) -> &'static str {
        // Phase 2 replaces the trivial fitter with the real estimator.
        "v0.2"
    }

    fn calibrate(
        &self,
        hypergraph: &Hypergraph,
        narrative_id: &str,
    ) -> Result<serde_json::Value> {
        // Phase 2 — delegate to the real fitter. Empty narrative_id is an
        // error here (the trait doesn't allow "calibrate from nothing"); the
        // fitter itself rejects empty / unknown / zero-entity narratives.
        if narrative_id.is_empty() {
            return Err(TensaError::SynthFailure(
                "EathSurrogate::calibrate: narrative_id is empty".into(),
            ));
        }
        let params = super::calibrate::fit_params_from_narrative(hypergraph, narrative_id)?;
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
        // Phase 2.5 fills these — listed here so the trait stays self-documenting.
        vec![
            "group_size_distribution",
            "interevent_time_distribution",
            "active_group_lifetime",
            "actor_activity_rate",
        ]
    }
}

// ── Validation ───────────────────────────────────────────────────────────────

fn validate_params(p: &EathParams) -> Result<()> {
    if p.a_t_distribution.len() < 2 {
        return Err(TensaError::SynthFailure(format!(
            "EathParams: num_entities = {} < 2 (need a_t_distribution.len() >= 2)",
            p.a_t_distribution.len()
        )));
    }
    if !p.a_h_distribution.is_empty() && p.a_h_distribution.len() != p.a_t_distribution.len() {
        return Err(TensaError::SynthFailure(format!(
            "EathParams: a_h_distribution.len() ({}) must match a_t_distribution.len() ({})",
            p.a_h_distribution.len(),
            p.a_t_distribution.len(),
        )));
    }
    for (i, &v) in p.a_t_distribution.iter().enumerate() {
        if !v.is_finite() {
            return Err(TensaError::SynthFailure(format!(
                "EathParams: NaN/Inf in a_t_distribution at index {i}"
            )));
        }
        if v < 0.0 {
            return Err(TensaError::SynthFailure(format!(
                "EathParams: negative value in a_t_distribution at index {i}: {v}"
            )));
        }
    }
    for (i, &v) in p.a_h_distribution.iter().enumerate() {
        if !v.is_finite() {
            return Err(TensaError::SynthFailure(format!(
                "EathParams: NaN/Inf in a_h_distribution at index {i}"
            )));
        }
        if v < 0.0 {
            return Err(TensaError::SynthFailure(format!(
                "EathParams: negative value in a_h_distribution at index {i}: {v}"
            )));
        }
    }
    for (i, &v) in p.lambda_schedule.iter().enumerate() {
        if !v.is_finite() {
            return Err(TensaError::SynthFailure(format!(
                "EathParams: NaN/Inf in lambda_schedule at index {i}"
            )));
        }
    }
    if !p.rho_low.is_finite() || !p.rho_high.is_finite() || !p.xi.is_finite() {
        return Err(TensaError::SynthFailure(
            "EathParams: NaN/Inf in rho_low / rho_high / xi".into(),
        ));
    }
    if !p.p_from_scratch.is_finite()
        || !(0.0..=1.0).contains(&p.p_from_scratch)
    {
        return Err(TensaError::SynthFailure(format!(
            "EathParams: p_from_scratch must be in [0,1], got {}",
            p.p_from_scratch
        )));
    }
    if p.max_group_size < 2 {
        return Err(TensaError::SynthFailure(format!(
            "EathParams: max_group_size must be >= 2, got {}",
            p.max_group_size
        )));
    }
    Ok(())
}

// ── Run summary scratch ──────────────────────────────────────────────────────

#[derive(Default)]
struct RunStats {
    num_situations: usize,
    num_participations: usize,
}

// ── Main generate() ──────────────────────────────────────────────────────────

/// Synthetic-epoch anchor for `temporal.start` on emitted situations.
/// 2020-01-01T00:00:00Z; one situation per "tick" (1 second by default).
/// Same anchor Phase 1's inline emit used — kept for backward compatibility
/// with downstream temporal-ordering tests (T1).
fn synth_epoch() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
}

/// Source narrative this run was calibrated against. Threaded through
/// `EathSurrogate::generate_with_source` so the `ReproducibilityBlob`
/// + `SurrogateRunSummary` carry the matching `source_state_hash`.
fn run_generate(
    params: &SurrogateParams,
    target: &Hypergraph,
    output_narrative_id: &str,
) -> Result<SurrogateRunSummary> {
    run_generate_with_source(params, target, output_narrative_id, None)
}

fn run_generate_with_source(
    params: &SurrogateParams,
    target: &Hypergraph,
    output_narrative_id: &str,
    source_narrative_id: Option<&str>,
) -> Result<SurrogateRunSummary> {
    // 1. Parse model-specific params.
    let params_eath: EathParams = serde_json::from_value(params.params_json.clone())
        .map_err(|e| TensaError::SynthFailure(format!("EathParams parse: {e}")))?;

    // 2. Validate (errors fire BEFORE any KV write — no orphaned blob).
    validate_params(&params_eath)?;

    // 3. Mint deterministic run_id from a sub-RNG seeded by params.seed.
    //    Wall-clock UUIDv7 would break the "same seed ⇒ same UUIDs" contract.
    let run_id = run_id_from_seed(params.seed);
    let started_at = Utc::now();

    // 4. Compute source_state_hash if a source narrative was provided.
    let source_state_hash: Option<String> = match source_narrative_id {
        Some(nid) => Some(super::hashing::canonical_narrative_state_hash(target, nid)?),
        None => None,
    };

    // 5. Build + persist ReproducibilityBlob BEFORE the loop.
    let blob = super::build_reproducibility_blob(
        run_id,
        params.clone(),
        source_state_hash.clone(),
    );
    super::store_reproducibility_blob(target.store(), &blob)?;

    // 6. Build the EmitContext — single source of truth for provenance
    //    metadata threaded into every emitted record.
    let ctx = EmitContext {
        run_id,
        narrative_id: output_narrative_id.to_string(),
        maturity: MaturityLevel::Candidate,
        // EATH determinism tests assert exact bit-for-bit reproducibility, so
        // override the default 0.5 confidence to 1.0 — the tests don't care
        // about the *value*, only that it's stable across runs. Production
        // calibrate-then-generate should explicitly set DEFAULT_SYNTHETIC_CONFIDENCE.
        confidence: 1.0,
        label_prefix: format!("{}-", params.label_prefix),
        time_anchor: synth_epoch(),
        step_duration_seconds: 1,
        model: "eath".to_string(),
        // EATH mints fresh synthetic entities; configuration-style models
        // (Phase 13b NuDHy) set this to Some(source_entity_uuids).
        reuse_entities: None,
    };
    let _ = DEFAULT_SYNTHETIC_CONFIDENCE; // explicit reference so the constant is publicly used.

    // 7. Pre-build entity UUIDs (deterministic from seed) and emit them.
    //    A dedicated entity sub-RNG keeps emit's UUID draws independent of
    //    the main simulation stream, preserving determinism if emit's UUID
    //    scheme changes.
    let n = params_eath.a_t_distribution.len();
    let mut entity_rng = ChaCha8Rng::seed_from_u64(params.seed.wrapping_add(0xE1717));
    let entity_ids = write_synthetic_entities(&ctx, n, &mut entity_rng, target)?;

    // 8. Pre-loop scratch.
    let mean_a_t: f32 = if n == 0 {
        0.0
    } else {
        params_eath.a_t_distribution.iter().sum::<f32>() / n as f32
    };
    let mean_total_activity: f32 = params_eath
        .a_h_distribution
        .iter()
        .sum::<f32>()
        .max(1e-9);

    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
    let mut field = ActivityField::new(n);
    let mut stm = ShortTermMemory::new(params_eath.stm_capacity);
    let mut ltm = LongTermMemory::new();

    let mut weight_buf: Vec<f32> = vec![0.0; n];
    let mut member_indices: Vec<usize> = Vec::with_capacity(params_eath.max_group_size);
    let mut member_uuids: Vec<Uuid> = Vec::with_capacity(params_eath.max_group_size);

    let mut stats = RunStats::default();

    // 9. Main loop.
    for tick in 0..params.num_steps {
        let lambda_t = field.step_transition(&mut rng, &params_eath, tick as u64, mean_a_t);
        let _ = lambda_t;

        let total_activity = field.total_activity();
        let num_groups = draw_num_groups(
            &params_eath,
            total_activity,
            mean_total_activity,
            n,
            &mut rng,
        );

        for _ in 0..num_groups {
            let group_size = sample_group_size(&params_eath, &mut rng);
            let coin: f32 = rng.gen();
            let from_scratch = stm.is_empty() || coin < params_eath.p_from_scratch;

            member_indices.clear();
            if from_scratch {
                recruit_from_scratch(
                    group_size,
                    &field.current_activity,
                    &params_eath,
                    &ltm,
                    tick as u64,
                    n,
                    &mut weight_buf,
                    &mut member_indices,
                    &mut rng,
                );
            } else {
                recruit_from_memory(
                    group_size,
                    &stm,
                    &field.current_activity,
                    &params_eath,
                    &ltm,
                    tick as u64,
                    n,
                    &mut weight_buf,
                    &mut member_indices,
                    &mut rng,
                );
            }

            if member_indices.is_empty() {
                continue;
            }

            // Resolve member indices → UUIDs and emit through Phase 3 pipeline.
            // The situation gets the per-step temporal anchor + sentinel info_set
            // on every participant.
            member_uuids.clear();
            for &idx in &member_indices {
                if idx < entity_ids.len() {
                    member_uuids.push(entity_ids[idx]);
                }
            }
            if member_uuids.is_empty() {
                continue;
            }
            // Mint the situation UUID from the MAIN simulation RNG —
            // matches the Phase 1 inline-emit ordering so determinism tests
            // (T1) keep producing the same membership trace under the same
            // seed even after Phase 3's emit refactor.
            let _sit_id = write_synthetic_situation(
                &ctx,
                tick,
                &member_uuids,
                &mut rng,
                target,
            )?;

            // Update memory.
            stm.push(RecentGroup {
                members: member_indices.clone(),
                age: 0,
            });
            ltm.record_group(&member_indices, tick as u64);
            stats.num_situations += 1;
            stats.num_participations += member_uuids.len();
        }

        stm.age_all();
    }
    let finished_at = Utc::now();
    let params_hash = super::hashing::canonical_params_hash(params);
    let summary = SurrogateRunSummary {
        run_id,
        model: "eath".into(),
        params_hash,
        source_narrative_id: source_narrative_id.map(str::to_string),
        source_state_hash,
        output_narrative_id: output_narrative_id.into(),
        num_entities: entity_ids.len(),
        num_situations: stats.num_situations,
        num_participations: stats.num_participations,
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

// Phase 4 entry: surrogate generation worker calls this directly when it has
// a source narrative ID in hand. Phase 3's wider `EathSurrogate::generate`
// trait method goes through `run_generate` (no source).
#[allow(dead_code)]
pub(crate) fn generate_with_source(
    params: &SurrogateParams,
    target: &Hypergraph,
    output_narrative_id: &str,
    source_narrative_id: &str,
) -> Result<SurrogateRunSummary> {
    run_generate_with_source(params, target, output_narrative_id, Some(source_narrative_id))
}

// ── Deterministic UUID construction ─────────────────────────────────────────

/// Mint a run_id deterministically from `seed`. Distinct sub-RNG so this
/// doesn't consume from the main simulation stream.
fn run_id_from_seed(seed: u64) -> Uuid {
    // XOR with a distinct constant (all-ones in low 32 bits) so the run_id
    // sub-RNG is independent of the per-entity sub-RNG seed transformation.
    const RUN_ID_MIX: u64 = 0xFFFF_FFFF_0000_0000;
    let mut sub = ChaCha8Rng::seed_from_u64(seed ^ RUN_ID_MIX);
    let mut bytes = [0u8; 16];
    sub.fill_bytes(&mut bytes);
    set_uuid_v4_bits(&mut bytes);
    Uuid::from_bytes(bytes)
}

/// Build the entity UUID vector deterministically from `seed`. Per-entity
/// seed is derived as `(seed + i).wrapping_mul(RNG_SUBSEED_MIX)` — Knuth
/// golden-ratio mixing so adjacent entity indices map to well-dispersed
/// UUIDs even at small seeds.
///
/// Phase 3 routes entity emission through `crate::synth::emit::write_synthetic_entities`
/// which mints UUIDs from its own sub-RNG, so this helper is no longer
/// called from the run loop. It's retained for the determinism regression
/// test in `eath_tests.rs::test_build_entity_ids_is_deterministic`.
#[allow(dead_code)]
fn build_entity_ids(n: usize, seed: u64) -> Vec<Uuid> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let per_index_seed = seed
            .wrapping_add(i as u64)
            .wrapping_mul(RNG_SUBSEED_MIX);
        let mut sub = ChaCha8Rng::seed_from_u64(per_index_seed);
        let mut bytes = [0u8; 16];
        sub.fill_bytes(&mut bytes);
        set_uuid_v4_bits(&mut bytes);
        out.push(Uuid::from_bytes(bytes));
    }
    out
}

/// Draw a v4-style situation UUID from the main RNG. The monotone tick
/// counter is NOT embedded — purely random bytes, version + variant bits set
/// per RFC 4122.
///
/// Phase 3 deletion candidate: situations are now emitted via
/// `crate::synth::emit::write_synthetic_situation` which mints its own UUIDs.
#[allow(dead_code)]
fn situation_uuid_from_rng(rng: &mut ChaCha8Rng) -> Uuid {
    let mut bytes = [0u8; 16];
    rng.fill_bytes(&mut bytes);
    set_uuid_v4_bits(&mut bytes);
    Uuid::from_bytes(bytes)
}

#[inline]
fn set_uuid_v4_bits(bytes: &mut [u8; 16]) {
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
}

// ── Tests ────────────────────────────────────────────────────────────────────
//
// Phase-1 algorithm-level tests (T1-T8) plus internal unit tests for
// `validate_params`, `trivial_calibrate`, and the deterministic UUID helpers
// all live in `eath_tests.rs` so this file stays under the 500-line cap.

#[cfg(test)]
#[path = "eath_tests.rs"]
mod eath_tests;
