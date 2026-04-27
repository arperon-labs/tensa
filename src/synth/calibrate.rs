//! Phase 2 — fit `EathParams` from a real TENSA narrative.
//!
//! Replaces the Phase-1 trivial fitter (uniform `a_T`, constant `λ_t`,
//! [1, 1, 1] group-size distribution) with a real estimator that walks every
//! entity / situation / participation in the source narrative and produces
//! per-entity activity rates, a time-bucketed `Λ_t` schedule, an empirical
//! group-size histogram, and the seven Phase-1 extension parameters
//! (`rho_low`, `rho_high`, `xi`, `order_propensity`, `max_group_size`,
//! `stm_capacity`, `num_entities`).
//!
//! Algorithm reference: `docs/synth_eath_algorithm.md` §6.2 (full fitter).
//! Phase reference: `docs/EATH_sprint.md` Phase 2 prompt.
//!
//! The fitter is intentionally O(N + S + P): one pass over entities, one over
//! situations (sorted), and one over the participation index built via
//! [`crate::analysis::graph_projection::collect_participation_index`] (which
//! itself is one pass through `ps/`). No N+1 store gets in the hot path.
//!
//! Per-field fitters live in `calibrate_fitters.rs` (sibling module) so this
//! file stays under the 500-line cap.

use std::collections::HashMap;

use chrono::Utc;
use uuid::Uuid;

use crate::analysis::graph_projection::collect_participation_index;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;
use crate::types::Situation;

use super::calibrate_fitters::{
    build_group_size_histogram, build_lambda_schedule, compute_per_entity_activity,
    estimate_p_from_scratch, estimate_rho_pair, first_non_finite,
    sort_situations_chronologically,
};
use super::fidelity::{
    run_fidelity_report, FidelityConfig, FidelityReport, ThresholdsProvenance,
};
use super::key_synth_params;
use super::types::EathParams;

// ── Tunable constants (named so /simplify doesn't flag them as magic) ────────

/// Default memory decay coefficient ω₀. Mancastroppa § III.1 reports 0.95 as
/// the empirical sweet spot across the RFID datasets in the paper.
const DEFAULT_OMEGA_DECAY: f32 = 0.95;

/// Default short-term memory ring-buffer capacity. Midpoint of the paper's
/// 5–10 range. Calibration does NOT change this (it's a generation-time
/// hyperparameter that future studies may sweep).
const DEFAULT_STM_CAPACITY: usize = 7;

/// Maximum group size cap exposed by [`EathParams::max_group_size`]. Used as
/// the upper bound for the histogram size and the order-propensity row width.
/// Prevents pathological 1000-actor groups from blowing up the propensity
/// matrix at calibration time.
const MAX_GROUP_SIZE_CAP: usize = 50;

/// `xi` clamp range. `xi` becomes the expected groups per tick after
/// normalisation; [0.1, 50.0] keeps the simulator from emitting nothing or
/// flooding it with thousands of groups per step.
const XI_CLAMP_MIN: f32 = 0.1;
const XI_CLAMP_MAX: f32 = 50.0;

/// Default model name carried in [`EathParams`]-bearing
/// [`super::types::SurrogateParams`] envelopes. Keep as a constant so the
/// persistence helpers and the EATH surrogate's `name()` agree.
pub const EATH_MODEL_NAME: &str = "eath";

// ── Public API ───────────────────────────────────────────────────────────────

/// Internal params-only fitter. Public callers go through
/// [`fit_params_from_narrative`] (wrapper) or [`calibrate_with_fidelity_report`]
/// (full calibration + fidelity assessment). This entry point skips the K=20
/// fidelity step so the calibration test suite (T1-T7 in `calibrate_tests.rs`)
/// stays cheap.
fn fit_params_only_from_narrative(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<EathParams> {
    if narrative_id.is_empty() {
        return Err(TensaError::SynthFailure(
            "fit_params_from_narrative: narrative_id is empty".into(),
        ));
    }

    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let n = entities.len();
    if n == 0 {
        return Err(TensaError::SynthFailure(format!(
            "narrative '{narrative_id}' has 0 entities; cannot calibrate"
        )));
    }

    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let num_situations = situations.len();

    // Stable entity ordering: the order returned by `list_entities_by_narrative`.
    // Phase 1 indexes into `a_t_distribution` / `a_h_distribution` /
    // `order_propensity` by *position*, so ALL produced vectors below MUST use
    // this same ordering.
    let entity_idx: HashMap<Uuid, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id, i))
        .collect();

    // Sort situations chronologically with creation-order fallback for
    // `TimeGranularity::Unknown` rows (or anywhere `temporal.start` is None).
    // The sorted order drives both the `lambda_schedule` bucketing (step 5)
    // and the consecutive-pair scan for `p_from_scratch` (step 6).
    let situation_order = sort_situations_chronologically(&situations);
    let sorted_situations: Vec<&Situation> =
        situation_order.iter().map(|&i| &situations[i]).collect();
    let situation_ids: Vec<Uuid> = sorted_situations.iter().map(|s| s.id).collect();

    // Single-pass participation index: situation_id -> [entity_idx].
    // The function does ONE prefix scan per situation under `ps/` — never
    // an entity-by-entity walk — so the cost is O(S) not O(N * S).
    let participants_by_sit =
        collect_participation_index(hypergraph, &entity_idx, &situation_ids, None)?;

    // Per-entity scratch: how many distinct situations each entity touched
    // and the cumulative observed group sizes when they participated.
    let mut sit_count_per_entity: Vec<u32> = vec![0; n];
    let mut group_size_sum_per_entity: Vec<u64> = vec![0; n];

    // Group-size histogram: capped at `max_group_size_observed.min(CAP)`.
    let mut max_group_size_observed: usize = 2;

    // First pass over the participation index: per-entity stats + the running
    // max group size. We split into two passes (this one + the histogram pass
    // inside `build_group_size_histogram`) so we can size the histogram
    // exactly once.
    for sit in &sorted_situations {
        let members = match participants_by_sit.get(&sit.id) {
            Some(v) => v,
            None => continue,
        };
        let group_size = members.len();
        if group_size > max_group_size_observed {
            max_group_size_observed = group_size;
        }
        for &idx in members {
            if idx < n {
                sit_count_per_entity[idx] = sit_count_per_entity[idx].saturating_add(1);
                group_size_sum_per_entity[idx] =
                    group_size_sum_per_entity[idx].saturating_add(group_size as u64);
            }
        }
    }

    let max_group_size = max_group_size_observed.clamp(2, MAX_GROUP_SIZE_CAP);

    let (a_t, a_h) = compute_per_entity_activity(
        n,
        num_situations,
        &sit_count_per_entity,
        &group_size_sum_per_entity,
    );

    let group_size_distribution =
        build_group_size_histogram(&sorted_situations, &participants_by_sit, max_group_size);

    let (lambda_schedule, mean_groups_per_bucket) = build_lambda_schedule(num_situations);

    let p_from_scratch = estimate_p_from_scratch(&sorted_situations, &participants_by_sit);

    let (rho_low, rho_high) = estimate_rho_pair(&sorted_situations, &participants_by_sit, n);

    // xi ≈ mean groups per tick (one situation per emitted group, so
    // mean groups per bucket IS the per-tick rate after normalisation).
    let xi = mean_groups_per_bucket.clamp(XI_CLAMP_MIN, XI_CLAMP_MAX);

    // order_propensity[i] aligns with ah[i] in this phase (Phase 4 may
    // diverge them when per-entity heterogeneity gets fitted). Phase 1's
    // `order_propensity_for` falls back to the empirical group-size
    // distribution when the propensity vector is empty, so we pass empty
    // here — the per-entity scalar lives in `a_h_distribution` already.
    let order_propensity: Vec<f32> = Vec::new();

    let params = EathParams {
        a_t_distribution: a_t,
        a_h_distribution: a_h,
        lambda_schedule,
        p_from_scratch,
        omega_decay: DEFAULT_OMEGA_DECAY,
        group_size_distribution,
        rho_low,
        rho_high,
        xi,
        order_propensity,
        max_group_size,
        stm_capacity: DEFAULT_STM_CAPACITY,
        num_entities: n,
    };

    // Defensive NaN / inf sweep BEFORE returning. If any field would carry
    // a non-finite value the calibration is unusable downstream (the
    // generator's `validate_params` would catch it too, but failing here
    // gives the caller a more actionable error message).
    //
    // NOTE: the spec calls out a `seed: u64` derived here from
    // `chrono::Utc::now().timestamp_nanos_opt()`. We deliberately defer that
    // to the caller — Phase 2's `fit_params_from_narrative` returns
    // `EathParams` (not `SurrogateParams`), and the seed lives on the
    // `SurrogateParams` envelope. Phase 4 wires calibration → generation
    // and mints the seed at that boundary via [`generate_seed`] (kept as
    // a public helper here so both call sites agree on the contract).
    if let Some(reason) = first_non_finite(&params) {
        return Err(TensaError::SynthFailure(format!(
            "calibration produced non-finite {reason}"
        )));
    }

    Ok(params)
}

/// Mint a calibration-time seed. Wall-clock IS acceptable here — calibration
/// is offline setup, NOT the deterministic generation hot path. The
/// generator's reproducibility comes from its `SurrogateParams.seed` field
/// being stored verbatim in the [`super::ReproducibilityBlob`].
///
/// Falls back to a fixed sentinel if the system clock somehow returns a
/// time outside the i64-nanosecond range — which won't happen for current
/// dates but the fallback keeps this function infallible.
pub fn generate_seed() -> u64 {
    Utc::now()
        .timestamp_nanos_opt()
        .map(|n| n as u64)
        .unwrap_or(0xDEAD_BEEF_DEAD_BEEF)
}

/// Persist a calibrated `EathParams` (or any other surrogate model's params)
/// at `syn/p/{narrative_id}/{model}` via serde_json.
///
/// `model` exists so future surrogate families can persist their own params
/// under the same prefix without colliding with EATH. For EATH callers,
/// pass [`EATH_MODEL_NAME`].
pub fn save_params(
    store: &dyn KVStore,
    narrative_id: &str,
    model: &str,
    params: &EathParams,
) -> Result<()> {
    let key = key_synth_params(narrative_id, model);
    let value = serde_json::to_vec(params)?;
    store.put(&key, &value)
}

/// Inverse of [`save_params`]. Returns `Ok(None)` when no params exist for
/// the (narrative, model) pair.
pub fn load_params(
    store: &dyn KVStore,
    narrative_id: &str,
    model: &str,
) -> Result<Option<EathParams>> {
    let key = key_synth_params(narrative_id, model);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Delete persisted params. No-op on a missing key (the underlying KVStore
/// contract is delete-as-idempotent).
pub fn delete_params(
    store: &dyn KVStore,
    narrative_id: &str,
    model: &str,
) -> Result<()> {
    let key = key_synth_params(narrative_id, model);
    store.delete(&key)
}

// ── Public calibration entry points (Phase 2.5) ──────────────────────────────

/// Calibrate `EathParams` from the source narrative AND produce a fidelity
/// report comparing the calibration against K synthetic samples.
///
/// This is the canonical Phase 2.5 entry point — every successful calibration
/// in production paths (Phase 4 inference engine, REST `POST /synth/calibrate`,
/// MCP tools) routes through here so reviewers always see a fidelity report.
///
/// The function does NOT persist the report. Callers (Phase 4 engine in
/// particular) decide whether to write to KV via
/// [`super::fidelity::save_fidelity_report`].
///
/// `base_seed` derives the per-K-sample seeds via XOR-mix with each sample
/// index — same algebra under any threading mode, so single-threaded and
/// multi-threaded fidelity runs produce IDENTICAL reports for the same
/// inputs. Use [`generate_seed`] when you don't have a specific seed in
/// mind.
pub fn calibrate_with_fidelity_report(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    config: &FidelityConfig,
) -> Result<(EathParams, FidelityReport)> {
    let params = fit_params_only_from_narrative(hypergraph, narrative_id)?;
    let run_id = Uuid::now_v7();
    let base_seed = generate_seed();
    let report = run_fidelity_report(
        hypergraph,
        narrative_id,
        &params,
        config,
        run_id,
        base_seed,
        ThresholdsProvenance::Default,
    )?;
    Ok((params, report))
}

/// Backward-compatible thin wrapper preserving the Phase 2 signature.
///
/// Calls [`fit_params_only_from_narrative`] — does NOT trigger the K=20
/// fidelity loop. Existing callers (`EathSurrogate::calibrate`, the
/// calibration test suite, ad-hoc tooling) keep their performance profile.
///
/// Production code that wants a fidelity report alongside params MUST call
/// [`calibrate_with_fidelity_report`] directly. The Phase 4 inference engine
/// will route through that entry point so reviewers always see a fidelity
/// score in the job result.
pub fn fit_params_from_narrative(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<EathParams> {
    fit_params_only_from_narrative(hypergraph, narrative_id)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "calibrate_tests.rs"]
mod calibrate_tests;
