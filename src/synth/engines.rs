//! Phase 4 — surrogate calibration + generation as `InferenceEngine` impls.
//!
//! Wires the synth substrate into the existing `WorkerPool` so calibration
//! and generation behave like every other long-running TENSA job:
//! priority-scheduled, KV-persisted, deduplicated by `(target, type)`,
//! resumable across restarts.
//!
//! Two distinct engines (NOT one combined "synth" engine):
//!
//! * [`SurrogateCalibrationEngine`] — one-shot O(dataset) fit + fidelity
//!   report. Cheap to retry. Result carries the [`super::fidelity::FidelityReport`]
//!   so reviewers see fidelity metrics without a second job submission.
//! * [`SurrogateGenerationEngine`] — variable-length forward simulation.
//!   Honors `seed_override` for deterministic replay. Persists
//!   [`super::ReproducibilityBlob`] BEFORE the loop starts (Phase 1
//!   contract) so a crashed run is still reproducible from inputs.
//!
//! Both engines share one `Arc<SurrogateRegistry>` — the registry is
//! read-only after construction so multiple workers can hold the same
//! `Arc` without contention.
//!
//! ## Engine-key resolution (Phase 0 deferral closed)
//!
//! `WorkerPool::engines` was historically keyed by the full
//! [`InferenceJobType`] value. That works for unit-only variants
//! (`PageRank`, `EntropyAnalysis`, etc.) but BREAKS for variants with
//! payload (`SurrogateCalibration { narrative_id, model }`) because the
//! engine's `job_type()` returns ONE fixed value while jobs carry
//! arbitrary field values. Phase 4 switches the map to
//! `HashMap<std::mem::Discriminant<InferenceJobType>, _>` so lookup is
//! by variant tag, not by payload-equality. See
//! [`crate::inference::worker`] for the changeover.

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

use super::calibrate::{
    calibrate_with_fidelity_report, generate_seed, load_params, save_params,
};
use super::eath::generate_with_source as eath_generate_with_source;
use super::fidelity::{save_fidelity_report, FidelityConfig, ParallelismMode};
use super::registry::SurrogateRegistry;
use super::significance::{
    guard_not_fully_synthetic, resolve_params, save_significance_report, SignificanceMetric,
    SignificanceReport, K_MAX,
};
use super::significance::adapters::{run_significance_pipeline, AdapterChoice};
use super::types::{EathParams, SurrogateParams};
use uuid::Uuid;

// ── Defaults ─────────────────────────────────────────────────────────────────

/// Default `num_steps` for a generation job when neither `params_override`
/// nor `job.parameters.num_steps` supplies one. Picked to be cheap enough
/// for unit tests while still producing a non-trivial trace.
const DEFAULT_NUM_STEPS: usize = 100;

/// Default `label_prefix` carried into the `SurrogateParams` envelope.
/// Output narratives get `{label_prefix}-{run_id}` lineage threaded through
/// every emitted entity / situation. Phase 5 lets callers override via the
/// TensaQL `LABEL_PREFIX` clause; until then this default applies.
const DEFAULT_LABEL_PREFIX: &str = "synth";

// Sentinel "no narrative" target for SurrogateCalibration / Generation jobs.
// Calibration is narrative-scoped (the narrative_id lives in the variant
// fields, NOT a UUID), but `InferenceJob.target_id` is `Uuid` and isn't
// nullable. Engines ignore `target_id` for synth jobs — the routing key is
// the variant's `narrative_id` / `output_narrative_id` field.

// ── Calibration engine ───────────────────────────────────────────────────────

/// Inference engine wrapper for [`super::SurrogateModel::calibrate`] + the
/// Phase 2.5 fidelity report.
///
/// The calibration → fidelity bundle goes through
/// [`calibrate_with_fidelity_report`], NOT through the model trait's
/// `calibrate()` method directly, so the resulting [`InferenceResult`]
/// always carries a fidelity score reviewers can act on.
pub struct SurrogateCalibrationEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateCalibrationEngine {
    /// Construct with a shared registry.
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateCalibrationEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel value with empty fields — the worker pool keys engines by
        // `std::mem::discriminant`, not by full equality, so the field values
        // here don't matter. Using empty strings makes "if you ever look at
        // the registration" obviously a sentinel, not a real config.
        InferenceJobType::SurrogateCalibration {
            narrative_id: String::new(),
            model: String::new(),
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        // Cost is owned by `inference::cost::estimate_cost`. Engine's own
        // estimator is only consulted when the planner skips the central
        // cost table — return the same flat figure here so behavior agrees.
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        // 1. Deconstruct variant payload.
        let (narrative_id, model_name) = match &job.job_type {
            InferenceJobType::SurrogateCalibration { narrative_id, model } => {
                (narrative_id.clone(), model.clone())
            }
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateCalibrationEngine received wrong job type: {other:?}"
                )));
            }
        };

        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateCalibration: narrative_id is empty".into(),
            ));
        }
        if model_name.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateCalibration: model is empty".into(),
            ));
        }

        // 2. Resolve model — error early on unknown name (no wasted fit).
        //    For now we only validate that the requested model is registered;
        //    the actual calibration goes through `calibrate_with_fidelity_report`
        //    which is EATH-specific. Future surrogate families will dispatch on
        //    `model.name()` here — until then, only "eath" is supported and
        //    other registered models would error out at the EATH-specific
        //    fidelity step.
        let model = self.registry.get(&model_name)?;
        if model.name() != "eath" {
            return Err(TensaError::SynthFailure(format!(
                "calibration for model '{}' is not yet wired through this engine \
                 (only 'eath' is supported in Phase 4)",
                model.name()
            )));
        }

        // 3. Calibrate + fidelity-report in one call so reviewers always see
        //    a fidelity score in the InferenceResult.
        let (params, report) = calibrate_with_fidelity_report(
            hypergraph,
            &narrative_id,
            &FidelityConfig::default(),
        )?;

        // 4. Persist params + fidelity report so generation jobs can pick
        //    them up later (Phase 4.5+ may add cross-narrative reuse).
        save_params(hypergraph.store(), &narrative_id, &model_name, &params)?;
        save_fidelity_report(hypergraph.store(), &report)?;

        // 5. Build the result envelope. `params_summary` is a sentinel slice
        //    of EathParams fields chosen for at-a-glance review in the API
        //    response — full params live at `syn/p/{nid}/{model}` for callers
        //    that need them.
        let params_summary = serde_json::json!({
            "num_entities": params.num_entities,
            "max_group_size": params.max_group_size,
            "stm_capacity": params.stm_capacity,
            "p_from_scratch": params.p_from_scratch,
            "rho_low": params.rho_low,
            "rho_high": params.rho_high,
            "xi": params.xi,
        });

        let result_json = serde_json::json!({
            "kind": "calibration_done",
            "narrative_id": narrative_id,
            "model": model_name,
            "params_summary": params_summary,
            "fidelity_report": report,
            "run_id": report.run_id,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: result_json,
            confidence: report.overall_score.clamp(0.0, 1.0),
            explanation: None,
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ── Generation engine ────────────────────────────────────────────────────────

/// Inference engine wrapper for [`super::SurrogateModel::generate`].
///
/// Loads calibrated params (or accepts an inline override), honors a seed
/// override for deterministic replay, and routes through
/// [`eath_generate_with_source`] when a source narrative is in scope so
/// `ReproducibilityBlob.source_state_hash` is populated.
pub struct SurrogateGenerationEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateGenerationEngine {
    /// Construct with a shared registry.
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateGenerationEngine {
    fn job_type(&self) -> InferenceJobType {
        // Same sentinel pattern as SurrogateCalibrationEngine — the worker
        // pool keys by discriminant.
        InferenceJobType::SurrogateGeneration {
            source_narrative_id: None,
            output_narrative_id: String::new(),
            model: String::new(),
            params_override: None,
            seed_override: None,
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        // Defer to the central cost table populated by `cost::estimate_cost`.
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        // 1. Deconstruct variant payload.
        let (
            source_narrative_id,
            output_narrative_id,
            model_name,
            params_override,
            seed_override,
        ) = match &job.job_type {
            InferenceJobType::SurrogateGeneration {
                source_narrative_id,
                output_narrative_id,
                model,
                params_override,
                seed_override,
            } => (
                source_narrative_id.clone(),
                output_narrative_id.clone(),
                model.clone(),
                params_override.clone(),
                *seed_override,
            ),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateGenerationEngine received wrong job type: {other:?}"
                )));
            }
        };

        if output_narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateGeneration: output_narrative_id is empty".into(),
            ));
        }
        if model_name.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateGeneration: model is empty".into(),
            ));
        }

        // 2. Resolve model — error early on unknown name (no wasted load).
        let model = self.registry.get(&model_name)?;

        // 3. Resolve EathParams: override wins over loaded params.
        let eath_params: EathParams = if let Some(raw) = params_override {
            serde_json::from_value(raw).map_err(|e| {
                TensaError::InvalidInput(format!(
                    "SurrogateGeneration: params_override is not a valid EathParams blob: {e}"
                ))
            })?
        } else {
            let nid = source_narrative_id.as_deref().ok_or_else(|| {
                TensaError::InvalidInput(
                    "SurrogateGeneration: no params_override AND no source_narrative_id; \
                     calibrate first or pass params_override"
                        .into(),
                )
            })?;
            load_params(hypergraph.store(), nid, &model_name)?.ok_or_else(|| {
                TensaError::SynthFailure(format!(
                    "no calibrated params for ('{nid}', '{model_name}'); calibrate first or \
                     provide params_override"
                ))
            })?
        };

        // 4. Resolve num_steps: job parameters override default. Phase 5
        //    will surface this via the TensaQL grammar; until then the
        //    JSON parameters slot is the only knob.
        let num_steps = job
            .parameters
            .get("num_steps")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_NUM_STEPS);

        // 5. Resolve label_prefix: job parameters override default.
        let label_prefix = job
            .parameters
            .get("label_prefix")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_LABEL_PREFIX)
            .to_string();

        // 6. Build the SurrogateParams envelope. seed_override wins over a
        //    fresh `generate_seed()`; same algebra as the docstring promise.
        let seed = seed_override.unwrap_or_else(generate_seed);
        let surrogate_params = SurrogateParams {
            model: model_name.clone(),
            params_json: serde_json::to_value(&eath_params).map_err(|e| {
                TensaError::SynthFailure(format!("EathParams → JSON failed: {e}"))
            })?,
            seed,
            num_steps,
            label_prefix,
        };

        // 7. Generate. Prefer the source-aware entry point ONLY for the EATH
        //    model — that helper is EATH-specific (knows how to compute
        //    `source_state_hash`). Future surrogate families that want the
        //    same provenance will add equivalent helpers and dispatch on
        //    `model.name()`. Cancellation: TENSA's WorkerPool currently has
        //    no per-job cancellation token (jobs run to completion or fail
        //    via panic). Logged as a Phase 12.5 follow-up — see
        //    `docs/EATH_sprint.md` Notes.
        let run_summary = if model.name() == "eath" {
            if let Some(nid) = source_narrative_id.as_deref() {
                eath_generate_with_source(&surrogate_params, hypergraph, &output_narrative_id, nid)?
            } else {
                model.generate(&surrogate_params, hypergraph, &output_narrative_id)?
            }
        } else {
            model.generate(&surrogate_params, hypergraph, &output_narrative_id)?
        };

        // 8. Build the result envelope.
        let result_json = serde_json::json!({
            "kind": "generation_done",
            "run_summary": run_summary,
            "output_narrative_id": output_narrative_id,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: result_json,
            // Generation has no inherent confidence — the surrogate produced
            // exactly what it was asked to. Reviewers should consult the
            // upstream calibration's FidelityReport for fitness, not this.
            confidence: 1.0,
            explanation: None,
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ── Contagion-significance engine (Phase 7b) ─────────────────────────────────

/// Inference engine wrapper for higher-order contagion significance over EATH
/// surrogates. Mirrors [`super::SurrogateSignificanceEngine`] structurally but
/// dispatches through the
/// [`crate::analysis::higher_order_contagion`] simulator on each K sample.
///
/// Reuses every Phase 7 invariant guard:
/// * Refuses fully-synthetic source narratives (synthetic-vs-synthetic
///   z-scores are meaningless).
/// * Auto-calibrates EathParams when none are persisted (and reports it).
/// * K is clamped at [`super::significance::K_MAX`].
///
/// Requires `job.parameters.contagion_params` to be a JSON blob deserializable
/// into [`crate::analysis::higher_order_contagion::HigherOrderSirParams`].
pub struct SurrogateContagionSignificanceEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateContagionSignificanceEngine {
    /// Construct with a shared registry.
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateContagionSignificanceEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — same convention as SurrogateSignificanceEngine. The
        // worker pool keys engines by std::mem::Discriminant, so payload
        // values are ignored at registration.
        InferenceJobType::SurrogateContagionSignificance {
            narrative_id: String::new(),
            k: 0,
            model: String::new(),
            contagion_params: serde_json::Value::Null,
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let started = Instant::now();
        let started_at = Utc::now();

        // 1. Deconstruct variant payload.
        let (narrative_id, k_requested, model_name, contagion_params_json) = match &job.job_type {
            InferenceJobType::SurrogateContagionSignificance {
                narrative_id,
                k,
                model,
                contagion_params,
            } => (
                narrative_id.clone(),
                *k,
                model.clone(),
                contagion_params.clone(),
            ),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateContagionSignificanceEngine received wrong job type: {other:?}"
                )));
            }
        };

        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateContagionSignificance: narrative_id is empty".into(),
            ));
        }
        let model = if model_name.is_empty() {
            "eath".to_string()
        } else {
            model_name
        };

        // 2. Validate model is registered.
        let model_handle = self.registry.get(&model)?;
        if model_handle.name() != "eath" {
            return Err(TensaError::SynthFailure(format!(
                "contagion significance for model '{}' is not yet wired through this engine \
                 (only 'eath' is supported in Phase 7b)",
                model_handle.name()
            )));
        }

        // 3. Refuse fully-synthetic narratives.
        guard_not_fully_synthetic(hypergraph, &narrative_id)?;

        // 4. Cap k.
        let k_clamped = k_requested.clamp(1, K_MAX);
        if k_requested > K_MAX {
            tracing::warn!(
                "SurrogateContagionSignificance: k={k_requested} exceeds K_MAX={K_MAX}; clamping"
            );
        }
        let k_effective = k_clamped as usize;

        // 5. Resolve EathParams (override → loaded → auto-calibrate).
        let (eath_params, auto_calibrated, calibration_fidelity) =
            resolve_params(hypergraph, &narrative_id, &model, &job.parameters)?;

        // 6. Resolve HigherOrderSirParams. Source-of-truth order:
        //    job.parameters.contagion_params → variant payload's contagion_params.
        let raw_params = if !job
            .parameters
            .get("contagion_params")
            .map(serde_json::Value::is_null)
            .unwrap_or(true)
        {
            job.parameters.get("contagion_params").cloned().unwrap()
        } else {
            contagion_params_json
        };
        let sir_params =
            crate::analysis::higher_order_contagion::parse_params(&raw_params)?;

        // 7. Run the K-loop pipeline through the contagion adapter.
        let pipeline = run_significance_pipeline(
            hypergraph,
            &narrative_id,
            &eath_params,
            AdapterChoice::Contagion(&sir_params),
            &serde_json::Value::Null,
            k_effective,
            ParallelismMode::Auto,
        )?;

        // 8. Build the report.
        let finished_at = Utc::now();
        let duration_ms = started.elapsed().as_millis() as u64;
        let note = if pipeline.source_was_empty {
            Some(
                "higher-order contagion produced no infections on the source narrative; \
                 z-scores compare zero baselines"
                    .to_string(),
            )
        } else {
            None
        };

        let report = SignificanceReport {
            run_id: Uuid::now_v7(),
            narrative_id: narrative_id.clone(),
            model: model.clone(),
            metric: SignificanceMetric::Contagion,
            k_samples_used: k_effective,
            source_observation: pipeline.source_observation,
            synthetic_distribution: pipeline.distribution,
            auto_calibrated,
            calibration_fidelity,
            note,
            started_at,
            finished_at,
            duration_ms,
        };

        save_significance_report(hypergraph.store(), &report)?;

        let result_json = serde_json::json!({
            "kind": "contagion_significance_done",
            "narrative_id": report.narrative_id,
            "model": report.model,
            "metric": report.metric.as_kv_str(),
            "run_id": report.run_id,
            "k_samples_used": report.k_samples_used,
            "auto_calibrated": report.auto_calibrated,
            "report": report,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: result_json,
            confidence: 1.0,
            explanation: Some(format!(
                "higher-order contagion significance over {k_effective} synthetic samples"
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(finished_at),
        })
    }
}

// ── Hybrid generation engine (EATH Phase 9) ──────────────────────────────────

/// Inference engine for hybrid (mixture-distribution) generation.
///
/// Distinct from [`SurrogateGenerationEngine`] — that engine runs ONE EATH
/// process; this one multiplexes between `n` per-source recruitment streams
/// per [`crate::synth::hybrid::generate_hybrid_hypergraph`]. Worker pool
/// dispatch keys on the `InferenceJobType::SurrogateHybridGeneration`
/// discriminant.
pub struct SurrogateHybridGenerationEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateHybridGenerationEngine {
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateHybridGenerationEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — worker pool keys by std::mem::Discriminant.
        InferenceJobType::SurrogateHybridGeneration {
            components: serde_json::Value::Null,
            output_narrative_id: String::new(),
            seed_override: None,
            num_steps: None,
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let (components_json, output_narrative_id, seed_override, num_steps) = match &job.job_type {
            InferenceJobType::SurrogateHybridGeneration {
                components,
                output_narrative_id,
                seed_override,
                num_steps,
            } => (
                components.clone(),
                output_narrative_id.clone(),
                *seed_override,
                *num_steps,
            ),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateHybridGenerationEngine received wrong job type: {other:?}"
                )));
            }
        };

        if output_narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateHybridGeneration: output_narrative_id is empty".into(),
            ));
        }

        // Parse components from JSON (the variant stores raw JSON because
        // HybridComponent lives above `types::*` in the layering).
        let components: Vec<super::hybrid::HybridComponent> =
            serde_json::from_value(components_json).map_err(|e| {
                TensaError::InvalidInput(format!(
                    "SurrogateHybridGeneration: components is not a valid HybridComponent[] blob: {e}"
                ))
            })?;

        let seed = seed_override.unwrap_or_else(super::calibrate::generate_seed);
        let num_steps = num_steps.unwrap_or(100);

        let summary = super::hybrid::generate_hybrid_hypergraph(
            &components,
            &output_narrative_id,
            seed,
            num_steps,
            hypergraph,
            &self.registry,
        )?;

        let result_json = serde_json::json!({
            "kind": "hybrid_generation_done",
            "run_summary": summary,
            "output_narrative_id": output_narrative_id,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: result_json,
            confidence: 1.0,
            explanation: None,
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ── Convenience constructor ──────────────────────────────────────────────────

/// Build all three synth engines wired to one shared registry. Callers that
/// register multiple synth engines should always use this so the registry
/// `Arc` is genuinely shared.
pub fn make_synth_engines(
    registry: Arc<SurrogateRegistry>,
) -> (
    Arc<SurrogateCalibrationEngine>,
    Arc<SurrogateGenerationEngine>,
    Arc<SurrogateHybridGenerationEngine>,
) {
    (
        Arc::new(SurrogateCalibrationEngine::new(registry.clone())),
        Arc::new(SurrogateGenerationEngine::new(registry.clone())),
        Arc::new(SurrogateHybridGenerationEngine::new(registry)),
    )
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "engines_tests.rs"]
mod engines_tests;
