//! EATH Extension Phase 14 — bistability-significance engine.
//!
//! Runs a bistability sweep on the SOURCE narrative, then on K surrogate
//! samples per requested null model, and reports per-model quantiles for
//! `bistable_interval` width and `max_hysteresis_gap`.
//!
//! Mirrors Phase 13c's [`super::dual_significance_engine`] structurally:
//! per-model parallelism via `std::thread::scope`, per-K sequential
//! iteration inside each thread. The reduction-to-pairwise contract is
//! preserved because every call goes through
//! [`crate::analysis::contagion_bistability::run_bistability_sweep`] which
//! dispatches to the unchanged Phase 7b simulator.
//!
//! NuDHy starvation handling: failed MCMC chains tally into
//! `SingleModelBistabilityNull.starvations` rather than aborting the run.

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use uuid::Uuid;

use crate::analysis::contagion_bistability::{
    parse_params as parse_bistability_params, run_bistability_sweep, summary_scalars,
    BistabilitySweepParams,
};
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

use super::bistability_significance_samples::{
    per_sample_seed, run_one_eath_sample, run_one_nudhy_sample,
};
use super::dual_significance_engine::DEFAULT_MODELS;
use super::nudhy::NudhyParams;
use super::nudhy_surrogate::calibrate_nudhy;
use super::registry::SurrogateRegistry;
use super::save_bistability_significance_report;
use super::significance::{guard_not_fully_synthetic, resolve_params, K_MAX};
use super::types::{
    BistabilitySignificance, BistabilitySignificanceReport, EathParams,
    SingleModelBistabilityNull,
};

// ── Type alias ──────────────────────────────────────────────────────────────

/// NuDHy calibration triple — same shape as Phase 13c's helper, hoisted to a
/// named alias for clippy + readability.
type NudhyCalibration = (NudhyParams, Vec<Vec<Uuid>>, Vec<Vec<Uuid>>);

// ── Constants ────────────────────────────────────────────────────────────────

/// p<0.05 quantile threshold: source observed value must beat 95% of samples.
const P05_QUANTILE: f32 = 0.95;
/// p<0.01 quantile threshold.
const P01_QUANTILE: f32 = 0.99;

// ── Engine ───────────────────────────────────────────────────────────────────

/// `InferenceEngine` impl for [`InferenceJobType::SurrogateBistabilitySignificance`].
pub struct SurrogateBistabilitySignificanceEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateBistabilitySignificanceEngine {
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateBistabilitySignificanceEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — `WorkerPool::engines` keys by std::mem::Discriminant.
        InferenceJobType::SurrogateBistabilitySignificance {
            narrative_id: String::new(),
            params: serde_json::Value::Null,
            k: 0,
            models: Vec::new(),
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let started = Instant::now();
        let started_at = Utc::now();

        // 1. Deconstruct payload.
        let (narrative_id, params_json, k_requested, models_requested) = match &job.job_type {
            InferenceJobType::SurrogateBistabilitySignificance {
                narrative_id,
                params,
                k,
                models,
            } => (
                narrative_id.clone(),
                params.clone(),
                *k,
                models.clone(),
            ),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateBistabilitySignificanceEngine received wrong job type: {other:?}"
                )));
            }
        };

        // 2. Validate inputs.
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateBistabilitySignificance: narrative_id is empty".into(),
            ));
        }
        let bistability_params = parse_bistability_params(&params_json)?;

        let models: Vec<String> = if models_requested.is_empty() {
            DEFAULT_MODELS.iter().map(|s| s.to_string()).collect()
        } else {
            models_requested
        };
        for m in &models {
            self.registry.get(m)?;
        }

        // 3. Refuse fully-synthetic narratives — same invariant as Phase 7.
        guard_not_fully_synthetic(hypergraph, &narrative_id)?;

        // 4. Cap K.
        let k_clamped = k_requested.clamp(1, K_MAX);
        if k_requested > K_MAX {
            tracing::warn!(
                "SurrogateBistabilitySignificance: k={k_requested} exceeds K_MAX={K_MAX}; clamping"
            );
        }
        let k_effective = k_clamped as usize;

        // 5. Source observation: bistability sweep on the REAL narrative.
        let source_report = run_bistability_sweep(hypergraph, &narrative_id, &bistability_params)?;
        let (source_width, source_gap) = summary_scalars(&source_report);

        // 6. Resolve EathParams ONCE if needed (cheap when persisted).
        let eath_params_opt = if models.iter().any(|m| m == "eath") {
            let (params, _, _) =
                resolve_params(hypergraph, &narrative_id, "eath", &job.parameters)?;
            Some(params)
        } else {
            None
        };

        // 7. NuDHy: parse calibration ONCE outside the K-loop (Phase 13b
        //    heads-up).
        let nudhy_calibration: Option<NudhyCalibration> =
            if models.iter().any(|m| m == "nudhy") {
                let np = calibrate_nudhy(hypergraph, &narrative_id)?;
                let chain_edges: Vec<Vec<Uuid>> =
                    serde_json::from_value(np.initial_state_json.clone()).map_err(|e| {
                        TensaError::SynthFailure(format!(
                            "NuDHy initial_state_json parse: {e}"
                        ))
                    })?;
                let fixed_edges: Vec<Vec<Uuid>> = if np.fixed_edges_json.is_null() {
                    Vec::new()
                } else {
                    serde_json::from_value(np.fixed_edges_json.clone()).map_err(|e| {
                        TensaError::SynthFailure(format!(
                            "NuDHy fixed_edges_json parse: {e}"
                        ))
                    })?
                };
                Some((np, chain_edges, fixed_edges))
            } else {
                None
            };

        // 8. Per-model parallelism via std::thread::scope (Phase 13c pattern).
        let mut per_model_results: Vec<Option<Result<SingleModelBistabilityNull>>> =
            (0..models.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(models.len());
            for (idx, model_name) in models.iter().enumerate() {
                let model_name = model_name.clone();
                let bistability_ref = &bistability_params;
                let narrative_id_ref = &narrative_id;
                let eath_params_ref = eath_params_opt.as_ref();
                let nudhy_ref = nudhy_calibration.as_ref();
                let handle = scope.spawn(move || {
                    let res = run_one_model(
                        model_name.as_str(),
                        narrative_id_ref,
                        hypergraph,
                        bistability_ref,
                        k_effective,
                        eath_params_ref,
                        nudhy_ref,
                        source_width,
                        source_gap,
                    );
                    (idx, res)
                });
                handles.push(handle);
            }
            for handle in handles {
                if let Ok((idx, res)) = handle.join() {
                    per_model_results[idx] = Some(res);
                }
            }
        });

        let mut per_model: Vec<SingleModelBistabilityNull> = Vec::with_capacity(models.len());
        for (i, slot) in per_model_results.into_iter().enumerate() {
            match slot {
                Some(Ok(r)) => per_model.push(r),
                Some(Err(e)) => {
                    return Err(TensaError::SynthFailure(format!(
                        "SurrogateBistabilitySignificance: model '{}' failed: {e}",
                        models[i]
                    )));
                }
                None => {
                    return Err(TensaError::SynthFailure(format!(
                        "SurrogateBistabilitySignificance: model '{}' thread panicked",
                        models[i]
                    )));
                }
            }
        }

        // 9. Combine per-model rows into the headline verdict.
        let combined = combine_per_model(&per_model);

        // 10. Build + persist the report.
        let finished_at = Utc::now();
        let duration_ms = started.elapsed().as_millis() as u64;
        let report = BistabilitySignificanceReport {
            run_id: Uuid::now_v7(),
            narrative_id: narrative_id.clone(),
            params: params_json,
            k: k_clamped,
            models: models.clone(),
            source_observation: serde_json::to_value(&source_report).map_err(|e| {
                TensaError::SynthFailure(format!(
                    "BistabilitySignificance: cannot serialize source observation: {e}"
                ))
            })?,
            per_model,
            combined,
            started_at,
            finished_at,
            duration_ms,
        };
        save_bistability_significance_report(hypergraph.store(), &report)?;

        let result_json = serde_json::json!({
            "kind": "bistability_significance_done",
            "narrative_id": report.narrative_id,
            "run_id": report.run_id,
            "k": report.k,
            "models": models,
            "report": report,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: result_json,
            confidence: 1.0,
            explanation: Some(format!(
                "bistability significance over {k_effective} samples × {} model(s)",
                models.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(finished_at),
        })
    }
}

// ── Per-model dispatch ──────────────────────────────────────────────────────

/// Run K bistability sweeps for one null model and reduce to a single
/// [`SingleModelBistabilityNull`] row.
#[allow(clippy::too_many_arguments)]
fn run_one_model(
    model_name: &str,
    narrative_id: &str,
    source_hg: &Hypergraph,
    params: &BistabilitySweepParams,
    k: usize,
    eath_params: Option<&EathParams>,
    nudhy_calibration: Option<&NudhyCalibration>,
    source_width: f32,
    source_gap: f32,
) -> Result<SingleModelBistabilityNull> {
    let mut widths: Vec<f32> = Vec::with_capacity(k);
    let mut gaps: Vec<f32> = Vec::with_capacity(k);
    let mut starvations: u16 = 0;
    let base_seed = bistability_base_seed(narrative_id, model_name);

    match model_name {
        "eath" => {
            let eath = eath_params.ok_or_else(|| {
                TensaError::SynthFailure(
                    "SurrogateBistabilitySignificance: EATH params not resolved".into(),
                )
            })?;
            for k_idx in 0..k {
                let report = run_one_eath_sample(eath, base_seed, k_idx, params)?;
                let (w, g) = summary_scalars(&report);
                widths.push(w);
                gaps.push(g);
            }
        }
        "nudhy" => {
            let cal = nudhy_calibration.ok_or_else(|| {
                TensaError::SynthFailure(
                    "SurrogateBistabilitySignificance: NuDHy calibration not resolved".into(),
                )
            })?;
            let (np, chain_edges, fixed_edges) = cal;
            for k_idx in 0..k {
                let chain_seed = per_sample_seed(base_seed, k_idx);
                match run_one_nudhy_sample(
                    np,
                    chain_edges.clone(),
                    fixed_edges.clone(),
                    chain_seed,
                    k_idx,
                    source_hg,
                    narrative_id,
                    params,
                ) {
                    Ok(report) => {
                        let (w, g) = summary_scalars(&report);
                        widths.push(w);
                        gaps.push(g);
                    }
                    Err(TensaError::SynthFailure(msg)) if msg.contains("starvation") => {
                        starvations = starvations.saturating_add(1);
                        tracing::warn!(
                            k_idx,
                            "NuDHy bistability K-chain {k_idx} starved; tallying and continuing"
                        );
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        other => {
            return Err(TensaError::SynthFailure(format!(
                "SurrogateBistabilitySignificance: model '{other}' is registered but no \
                 K-loop dispatch is wired"
            )));
        }
    }

    let (mean_w, std_w) = mean_std(&widths);
    let (mean_g, std_g) = mean_std(&gaps);
    let q_w = quantile_below(&widths, source_width);
    let q_g = quantile_below(&gaps, source_gap);

    Ok(SingleModelBistabilityNull {
        model: model_name.to_string(),
        mean_bistable_interval_width: mean_w,
        std_bistable_interval_width: std_w,
        bistable_interval_width_quantile: q_w,
        mean_max_hysteresis_gap: mean_g,
        std_max_hysteresis_gap: std_g,
        max_hysteresis_gap_quantile: q_g,
        samples_used: widths.len() as u16,
        starvations,
    })
}

// ── Reductions ──────────────────────────────────────────────────────────────

/// AND-reduce per-model rows into the headline verdict.
fn combine_per_model(rows: &[SingleModelBistabilityNull]) -> BistabilitySignificance {
    if rows.is_empty() {
        return BistabilitySignificance {
            source_bistable_wider_than_all_at_p05: false,
            source_bistable_wider_than_all_at_p01: false,
            min_quantile_across_models: 0.0,
            max_z_across_models: 0.0,
        };
    }
    let mut all_p05 = true;
    let mut all_p01 = true;
    let mut min_q: f32 = f32::INFINITY;
    let mut max_z: f32 = f32::NEG_INFINITY;
    for r in rows {
        if r.bistable_interval_width_quantile <= P05_QUANTILE {
            all_p05 = false;
        }
        if r.bistable_interval_width_quantile <= P01_QUANTILE {
            all_p01 = false;
        }
        if r.bistable_interval_width_quantile < min_q {
            min_q = r.bistable_interval_width_quantile;
        }
        if r.std_bistable_interval_width > 0.0 {
            // Width-equivalent z approximation: map quantile [0,1] → [-2, 2].
            // Directionally-correct without pulling in an inverse-normal dep.
            let z = ((r.bistable_interval_width_quantile as f64 - 0.5) / 0.5) * 2.0;
            if (z as f32) > max_z {
                max_z = z as f32;
            }
        }
    }
    // serde_json refuses NaN — coerce to 0.0 so the report round-trips.
    BistabilitySignificance {
        source_bistable_wider_than_all_at_p05: all_p05,
        source_bistable_wider_than_all_at_p01: all_p01,
        min_quantile_across_models: if min_q.is_finite() { min_q } else { 0.0 },
        max_z_across_models: if max_z.is_finite() { max_z } else { 0.0 },
    }
}

/// Population mean + stddev; returns `(0, 0)` for empty input.
fn mean_std(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() { return (0.0, 0.0); }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Empirical quantile of `target` against `samples`: fraction of samples
/// strictly less than target. Returns 0.0 for empty samples.
fn quantile_below(samples: &[f32], target: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let count = samples.iter().filter(|&&s| s < target).count();
    count as f32 / samples.len() as f32
}

// ── Determinism helpers ─────────────────────────────────────────────────────

/// Base seed from (narrative_id, model_name) — stable across reruns of the
/// same source narrative.
fn bistability_base_seed(narrative_id: &str, model_name: &str) -> u64 {
    let mut seed: u64 = 0x000B_1570_8AB1_5708;
    for (i, b) in narrative_id.bytes().enumerate() {
        seed ^= (b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15 ^ (i as u64));
        seed = seed.rotate_left(13);
    }
    for (i, b) in model_name.bytes().enumerate() {
        seed ^= (b as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9 ^ (i as u64));
        seed = seed.rotate_left(7);
    }
    seed
}

