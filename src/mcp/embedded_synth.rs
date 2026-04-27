//! EATH Phase 10 — EmbeddedBackend impls for the synthetic-hypergraph MCP tools.
//!
//! Seven tools:
//! 1. `calibrate_surrogate` — submit `SurrogateCalibration` job.
//! 2. `generate_synthetic_narrative` — submit `SurrogateGeneration` job.
//! 3. `generate_hybrid_narrative` — submit `SurrogateHybridGeneration` job.
//! 4. `list_synthetic_runs` — read `SurrogateRunSummary[]` from KV.
//! 5. `get_fidelity_report` — read `FidelityReport` from KV.
//! 6. `compute_pattern_significance` — submit `SurrogateSignificance` job.
//! 7. `simulate_higher_order_contagion` — submit `SurrogateContagionSignificance` job.
//!
//! Submission tools mirror the REST handlers in
//! [`crate::api::synth`] — same validation rules, same job envelopes
//! (`{ job_id, status: "Pending" }`). Read tools (4 & 5) call directly into
//! the synth library helpers, bypassing HTTP for the embedded backend.
//!
//! Job submission contract: `target_id` is a fresh `Uuid::now_v7()` so dedup
//! by `(target_id, job_type)` doesn't collapse two distinct narrative
//! calibrations or generations. Matches the contract documented in
//! [`crate::api::synth::submit_synth_job`].

use serde_json::Value;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::inference::types::InferenceJob;
use crate::synth::significance::{K_DEFAULT, K_MAX};
use crate::synth::{key_synth_run, list_runs_newest_first, SurrogateRunSummary};
use crate::synth::fidelity::load_fidelity_report;
use crate::types::{InferenceJobType, JobPriority, JobStatus};

use super::embedded::EmbeddedBackend;
use super::embedded_ext::{parse_uuid, to_json};

/// Default surrogate-model name. Mirrors `crate::api::synth::DEFAULT_MODEL`
/// (private const there). EATH is the only model shipping in Phase 4, but
/// every endpoint is still keyed by name so future surrogate families plug
/// in without route changes.
pub(crate) const DEFAULT_MODEL: &str = "eath";

/// Default page size for `list_synthetic_runs`. Mirrors
/// `crate::api::synth::DEFAULT_RUN_PAGE_LIMIT`.
pub(crate) const DEFAULT_RUN_PAGE_LIMIT: usize = 50;

impl EmbeddedBackend {
    /// Submit a synth job to the JobQueue. Returns `{ job_id, status }`
    /// matching the REST envelope from [`crate::api::synth::submit_synth_job`].
    fn submit_synth_job_impl(
        &self,
        job_type: InferenceJobType,
        parameters: Value,
    ) -> Result<Value> {
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            // Synth jobs are scoped by their narrative_id payload, not target_id;
            // engines ignore the field. Mint a fresh UUID so dedup-by-target
            // doesn't collapse two distinct narrative_id calibrations.
            target_id: Uuid::now_v7(),
            job_type,
            parameters,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let job_id = self.job_queue_arc().submit(job)?;
        Ok(serde_json::json!({"job_id": job_id, "status": "Pending"}))
    }

    // ─── Tool 1: calibrate_surrogate ─────────────────────────────

    pub(crate) async fn calibrate_surrogate_impl(
        &self,
        narrative_id: &str,
        model: Option<&str>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();
        self.submit_synth_job_impl(
            InferenceJobType::SurrogateCalibration {
                narrative_id: narrative_id.to_string(),
                model,
            },
            serde_json::json!({}),
        )
    }

    // ─── Tool 2: generate_synthetic_narrative ────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn generate_synthetic_narrative_impl(
        &self,
        source_narrative_id: &str,
        output_narrative_id: &str,
        model: Option<&str>,
        params: Option<Value>,
        seed: Option<u64>,
        num_steps: Option<usize>,
        label_prefix: Option<&str>,
    ) -> Result<Value> {
        if output_narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "output_narrative_id is empty".into(),
            ));
        }
        // We accept a non-empty source_narrative_id OR inline params (REST
        // endpoint allows source-less generation when params are inline). The
        // MCP request schema requires `source_narrative_id` as a String, so we
        // treat empty-string + no-params as the equivalent of "missing source".
        if source_narrative_id.is_empty() && params.is_none() {
            return Err(TensaError::InvalidInput(
                "missing source_narrative_id (and no inline params provided)".into(),
            ));
        }

        let source = if source_narrative_id.is_empty() {
            None
        } else {
            Some(source_narrative_id.to_string())
        };
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        let job_type = InferenceJobType::SurrogateGeneration {
            source_narrative_id: source,
            output_narrative_id: output_narrative_id.to_string(),
            model,
            params_override: params,
            seed_override: seed,
        };

        // num_steps + label_prefix piggyback on `job.parameters` because the
        // job-type variant doesn't carry them — engine reads them from the
        // JSON parameters slot (see `synth::engines::SurrogateGenerationEngine`).
        let mut params_map = serde_json::Map::new();
        if let Some(n) = num_steps {
            params_map.insert("num_steps".into(), serde_json::json!(n));
        }
        if let Some(prefix) = label_prefix {
            if !prefix.is_empty() {
                params_map.insert("label_prefix".into(), serde_json::json!(prefix));
            }
        }
        self.submit_synth_job_impl(job_type, Value::Object(params_map))
    }

    // ─── Tool 3: generate_hybrid_narrative ───────────────────────

    pub(crate) async fn generate_hybrid_narrative_impl(
        &self,
        components: Value,
        output_narrative_id: &str,
        seed: Option<u64>,
        num_steps: Option<usize>,
    ) -> Result<Value> {
        if output_narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "output_narrative_id is empty".into(),
            ));
        }
        // Validate the components shape so callers get a synchronous error
        // for the most-common mistakes (empty list, weights don't sum to 1.0).
        // The engine performs the same checks (defence in depth).
        let parsed: Vec<crate::synth::hybrid::HybridComponent> =
            serde_json::from_value(components.clone()).map_err(|e| {
                TensaError::InvalidInput(format!("invalid hybrid components: {e}"))
            })?;
        if parsed.is_empty() {
            return Err(TensaError::InvalidInput(
                "hybrid components list is empty".into(),
            ));
        }
        let sum: f32 = parsed.iter().map(|c| c.weight).sum();
        if (sum - 1.0).abs() > crate::synth::hybrid::HYBRID_WEIGHT_TOLERANCE {
            return Err(TensaError::InvalidInput(format!(
                "hybrid weights must sum to 1.0 (got {sum})"
            )));
        }

        let job_type = InferenceJobType::SurrogateHybridGeneration {
            components,
            output_narrative_id: output_narrative_id.to_string(),
            seed_override: seed,
            num_steps,
        };
        self.submit_synth_job_impl(job_type, Value::Object(serde_json::Map::new()))
    }

    // ─── Tool 4: list_synthetic_runs ─────────────────────────────

    pub(crate) async fn list_synthetic_runs_impl(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let limit = limit.unwrap_or(DEFAULT_RUN_PAGE_LIMIT).clamp(1, 1000);
        let runs = list_runs_newest_first(self.hypergraph().store(), narrative_id, limit)?;
        // Coerce into JSON array — preserves field-for-field shape.
        to_json(&runs)
    }

    // ─── Tool 5: get_fidelity_report ─────────────────────────────

    pub(crate) async fn get_fidelity_report_impl(
        &self,
        narrative_id: &str,
        run_id: &str,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let run_uuid = parse_uuid(run_id)?;
        match load_fidelity_report(self.hypergraph().store(), narrative_id, &run_uuid)? {
            Some(report) => to_json(&report),
            None => Err(TensaError::Internal(format!(
                "no fidelity report for ('{narrative_id}', '{run_id}')"
            ))),
        }
    }

    // ─── Tool 6: compute_pattern_significance ────────────────────

    pub(crate) async fn compute_pattern_significance_impl(
        &self,
        narrative_id: &str,
        metric: &str,
        k: Option<u16>,
        model: Option<&str>,
        params_override: Option<Value>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        // Accept exactly the three Phase 7 metrics. Contagion is routed to
        // `simulate_higher_order_contagion` (its own engine).
        match metric {
            "temporal_motifs" | "communities" | "patterns" => {}
            "contagion" => {
                return Err(TensaError::InvalidInput(
                    "metric 'contagion' must be submitted via simulate_higher_order_contagion"
                        .into(),
                ));
            }
            other => {
                return Err(TensaError::InvalidInput(format!(
                    "unknown metric '{other}'; expected one of: temporal_motifs, communities, patterns"
                )));
            }
        }

        let k_clamped = k.unwrap_or(K_DEFAULT).clamp(1, K_MAX);
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        let job_type = InferenceJobType::SurrogateSignificance {
            narrative_id: narrative_id.to_string(),
            metric_kind: metric.to_string(),
            k: k_clamped,
            model,
        };

        // params_override piggybacks on `job.parameters` because the variant
        // doesn't carry it — the engine reads it from there.
        let mut params = serde_json::Map::new();
        if let Some(p) = params_override {
            if !p.is_null() {
                params.insert("params_override".into(), p);
            }
        }
        self.submit_synth_job_impl(job_type, Value::Object(params))
    }

    // ─── Tool 8: compute_dual_significance (Phase 13c) ──────────

    /// Submit a `SurrogateDualSignificance` job. Mirrors
    /// `crate::api::synth::dual_significance::post_dual_significance` —
    /// validates `narrative_id`/`metric` shape, defaults `models` to
    /// `["eath", "nudhy"]` when omitted, clamps `k_per_model` at `K_MAX`.
    /// Returns `{ job_id, status: "Pending" }`.
    pub(crate) async fn compute_dual_significance_impl(
        &self,
        narrative_id: &str,
        metric: &str,
        k_per_model: Option<u16>,
        models: Option<Vec<String>>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        match metric {
            "temporal_motifs" | "communities" | "patterns" => {}
            "contagion" => {
                return Err(TensaError::InvalidInput(
                    "metric 'contagion' is not supported on the dual surface in Phase 13c; \
                     use simulate_higher_order_contagion for single-model contagion null testing"
                        .into(),
                ));
            }
            other => {
                return Err(TensaError::InvalidInput(format!(
                    "unknown metric '{other}'; expected one of: temporal_motifs, communities, patterns"
                )));
            }
        }
        let k_clamped = k_per_model.unwrap_or(K_DEFAULT).clamp(1, K_MAX);
        // Use the engine const so the default stays in lockstep with the
        // engine's expansion logic.
        let models: Vec<String> = match models {
            Some(v) if !v.is_empty() => v,
            _ => crate::synth::dual_significance_engine::DEFAULT_MODELS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };
        for m in &models {
            if m.is_empty() {
                return Err(TensaError::InvalidInput(
                    "models contains an empty string entry".into(),
                ));
            }
        }
        let job_type = InferenceJobType::SurrogateDualSignificance {
            narrative_id: narrative_id.to_string(),
            metric: metric.to_string(),
            k_per_model: k_clamped,
            models,
        };
        self.submit_synth_job_impl(job_type, Value::Object(serde_json::Map::new()))
    }

    // ─── Tool 9: compute_bistability_significance (Phase 14) ────

    /// Submit a `SurrogateBistabilitySignificance` job. Mirrors
    /// `crate::api::synth::bistability_significance::post_bistability_significance` —
    /// validates `narrative_id`/`params` shape, defaults `models` to
    /// `["eath", "nudhy"]` when omitted, clamps `k` at min(K_MAX, 500).
    /// Returns `{ job_id, status: "Pending" }`.
    pub(crate) async fn compute_bistability_significance_impl(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        models: Option<Vec<String>>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        if params.is_null() {
            return Err(TensaError::InvalidInput(
                "bistability params blob is required".into(),
            ));
        }
        let k_clamped = k.unwrap_or(K_DEFAULT.min(50)).clamp(1, K_MAX.min(500));
        let models: Vec<String> = match models {
            Some(v) if !v.is_empty() => v,
            _ => crate::synth::dual_significance_engine::DEFAULT_MODELS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };
        for m in &models {
            if m.is_empty() {
                return Err(TensaError::InvalidInput(
                    "models contains an empty string entry".into(),
                ));
            }
        }
        let job_type = InferenceJobType::SurrogateBistabilitySignificance {
            narrative_id: narrative_id.to_string(),
            params,
            k: k_clamped,
            models,
        };
        self.submit_synth_job_impl(job_type, Value::Null)
    }

    // ─── Tool 10: reconstruct_hypergraph (Phase 15c) ────────────

    /// Submit a `HypergraphReconstruction` job. Mirrors
    /// `crate::api::inference::reconstruction::submit` — validates
    /// `narrative_id`, threads the (optional) partial params blob through
    /// `job.parameters` so the engine's two-slot resolver picks it up, and
    /// returns `{ job_id, status: "Pending" }`.
    pub(crate) async fn reconstruct_hypergraph_impl(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let params_payload = params.unwrap_or_else(|| serde_json::json!({}));
        let job_type = InferenceJobType::HypergraphReconstruction {
            narrative_id: narrative_id.to_string(),
            params: params_payload.clone(),
        };
        let mut params_map = serde_json::Map::new();
        params_map.insert("narrative_id".into(), serde_json::json!(narrative_id));
        if let Value::Object(obj) = params_payload {
            for (k, v) in obj {
                params_map.insert(k, v);
            }
        }
        self.submit_synth_job_impl(job_type, Value::Object(params_map))
    }

    // ─── Tool 11: simulate_opinion_dynamics (Phase 16c) ──────────

    /// Synchronously run one opinion-dynamics simulation. Mirrors
    /// `crate::api::analysis::opinion_dynamics::run` — validates
    /// `narrative_id`, deserializes `params` over engine defaults, runs
    /// the engine, persists the report at `opd/report/{nid}/{run_id_v7}`,
    /// and returns `{ run_id, report }`.
    pub(crate) async fn simulate_opinion_dynamics_impl(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let p: crate::analysis::opinion_dynamics::OpinionDynamicsParams = match params {
            Some(v) if !v.is_null() => serde_json::from_value(v).map_err(|e| {
                TensaError::InvalidInput(format!("invalid OpinionDynamicsParams: {e}"))
            })?,
            _ => crate::analysis::opinion_dynamics::OpinionDynamicsParams::default(),
        };
        let report = crate::analysis::opinion_dynamics::simulate_opinion_dynamics(
            self.hypergraph(),
            narrative_id,
            &p,
        )?;
        let run_id = Uuid::now_v7();
        if let Err(e) = crate::analysis::opinion_dynamics::save_opinion_report(
            self.hypergraph().store(),
            narrative_id,
            run_id,
            &report,
        ) {
            tracing::warn!("MCP simulate_opinion_dynamics: persist failed: {e}");
        }
        Ok(serde_json::json!({"run_id": run_id, "report": report}))
    }

    // ─── Tool 12: simulate_opinion_phase_transition (Phase 16c) ──

    pub(crate) async fn simulate_opinion_phase_transition_impl(
        &self,
        narrative_id: &str,
        c_range: [Value; 3],
        base_params: Option<Value>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let c_start = c_range[0].as_f64().ok_or_else(|| {
            TensaError::InvalidInput("c_range[0] must be a finite number".into())
        })? as f32;
        let c_end = c_range[1].as_f64().ok_or_else(|| {
            TensaError::InvalidInput("c_range[1] must be a finite number".into())
        })? as f32;
        let num_points = c_range[2]
            .as_u64()
            .ok_or_else(|| TensaError::InvalidInput("c_range[2] must be an unsigned int".into()))?
            as usize;
        let bp: crate::analysis::opinion_dynamics::OpinionDynamicsParams = match base_params {
            Some(v) if !v.is_null() => serde_json::from_value(v).map_err(|e| {
                TensaError::InvalidInput(format!("invalid OpinionDynamicsParams: {e}"))
            })?,
            _ => crate::analysis::opinion_dynamics::OpinionDynamicsParams::default(),
        };
        let report = crate::analysis::opinion_dynamics::run_phase_transition_sweep(
            self.hypergraph(),
            narrative_id,
            (c_start, c_end, num_points),
            &bp,
        )?;
        Ok(serde_json::to_value(&report).map_err(|e| {
            TensaError::SynthFailure(format!("PhaseTransitionReport serialize: {e}"))
        })?)
    }

    // ─── Tool 7: simulate_higher_order_contagion ─────────────────

    pub(crate) async fn simulate_higher_order_contagion_impl(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        model: Option<&str>,
    ) -> Result<Value> {
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        if params.is_null() {
            return Err(TensaError::InvalidInput(
                "contagion params blob is required".into(),
            ));
        }
        // Validate shape so callers get a synchronous error rather than a
        // queued-then-failed job. Engine reparses defensively.
        let _typed: crate::analysis::higher_order_contagion::HigherOrderSirParams =
            serde_json::from_value(params.clone()).map_err(|e| {
                TensaError::InvalidInput(format!("invalid HigherOrderSirParams: {e}"))
            })?;

        let k_clamped = k.unwrap_or(K_DEFAULT).clamp(1, K_MAX);
        let model = model
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_MODEL)
            .to_string();

        let job_type = InferenceJobType::SurrogateContagionSignificance {
            narrative_id: narrative_id.to_string(),
            k: k_clamped,
            model,
            contagion_params: params.clone(),
        };

        // contagion_params lives on the variant payload AND is mirrored into
        // job.parameters for the engine to find via `parameters.contagion_params`
        // lookup (matches the engine's resolution order).
        let mut p_map = serde_json::Map::new();
        p_map.insert("contagion_params".into(), params);
        self.submit_synth_job_impl(job_type, Value::Object(p_map))
    }
}

// Suppress unused-import warning when this file's helpers are referenced
// only by the impl block above. `key_synth_run` + `SurrogateRunSummary` are
// imported for documentation lookup but not currently invoked — the
// `list_runs_newest_first` helper does the KV scan itself.
#[allow(dead_code)]
fn _unused_imports() {
    let _ = key_synth_run;
    let _: Option<SurrogateRunSummary> = None;
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::synth::fidelity::{save_fidelity_report, FidelityReport, ThresholdsProvenance};
    use crate::synth::record_lineage_run;
    use crate::synth::types::{RunKind, SurrogateRunSummary};
    use std::sync::Arc;

    fn backend() -> EmbeddedBackend {
        EmbeddedBackend::from_store(Arc::new(MemoryStore::new()))
    }

    fn write_run_summary(b: &EmbeddedBackend, narrative_id: &str) -> Uuid {
        let run_id = Uuid::now_v7();
        let summary = SurrogateRunSummary {
            run_id,
            model: "eath".to_string(),
            params_hash: "dummy".to_string(),
            source_narrative_id: Some(narrative_id.to_string()),
            source_state_hash: None,
            output_narrative_id: format!("{narrative_id}-synth"),
            num_entities: 0,
            num_situations: 0,
            num_participations: 0,
            started_at: chrono::Utc::now(),
            finished_at: chrono::Utc::now(),
            duration_ms: 0,
            kind: RunKind::Generation,
        };
        let key = key_synth_run(narrative_id, &run_id);
        b.hypergraph()
            .store()
            .put(&key, &serde_json::to_vec(&summary).unwrap())
            .unwrap();
        record_lineage_run(b.hypergraph().store(), narrative_id, &run_id).unwrap();
        run_id
    }

    fn dummy_fidelity(run_id: Uuid, narrative_id: &str) -> FidelityReport {
        FidelityReport {
            run_id,
            model: "eath".to_string(),
            narrative_id: narrative_id.to_string(),
            k_samples_used: 1,
            metrics: vec![],
            overall_score: 1.0,
            passed: true,
            thresholds_provenance: ThresholdsProvenance::Default,
            fuzzy_measure_id: None,
            fuzzy_measure_version: None,
        }
    }

    #[tokio::test]
    async fn test_calibrate_surrogate_round_trip() {
        let b = backend();
        let result = b.calibrate_surrogate_impl("hamlet", None).await.unwrap();
        assert!(result["job_id"].as_str().is_some());
        assert_eq!(result["status"], "Pending");
    }

    #[tokio::test]
    async fn test_calibrate_surrogate_default_model() {
        let b = backend();
        // None → "eath" default.
        let result = b.calibrate_surrogate_impl("hamlet", None).await.unwrap();
        let job_id = result["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(job_id).unwrap();
        match job.job_type {
            InferenceJobType::SurrogateCalibration { model, .. } => {
                assert_eq!(model, "eath");
            }
            other => panic!("expected SurrogateCalibration, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_calibrate_surrogate_empty_narrative() {
        let b = backend();
        assert!(b.calibrate_surrogate_impl("", None).await.is_err());
    }

    #[tokio::test]
    async fn test_generate_synthetic_narrative_round_trip() {
        let b = backend();
        let result = b
            .generate_synthetic_narrative_impl(
                "hamlet",
                "hamlet-synth",
                None,
                None,
                Some(42),
                Some(50),
                Some("test"),
            )
            .await
            .unwrap();
        assert!(result["job_id"].as_str().is_some());
        assert_eq!(result["status"], "Pending");
    }

    #[tokio::test]
    async fn test_generate_synthetic_narrative_missing_output() {
        let b = backend();
        let result = b
            .generate_synthetic_narrative_impl(
                "hamlet", "", None, None, None, None, None,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_synthetic_narrative_missing_source_and_params() {
        let b = backend();
        let result = b
            .generate_synthetic_narrative_impl(
                "", "out", None, None, None, None, None,
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generate_hybrid_narrative_round_trip() {
        let b = backend();
        let components = serde_json::json!([
            {"narrative_id": "a", "model": "eath", "weight": 0.5},
            {"narrative_id": "b", "model": "eath", "weight": 0.5},
        ]);
        let result = b
            .generate_hybrid_narrative_impl(components, "hybrid-out", Some(7), Some(50))
            .await
            .unwrap();
        assert!(result["job_id"].as_str().is_some());
        assert_eq!(result["status"], "Pending");
    }

    #[tokio::test]
    async fn test_generate_hybrid_narrative_bad_weights() {
        let b = backend();
        let components = serde_json::json!([
            {"narrative_id": "a", "model": "eath", "weight": 0.3},
            {"narrative_id": "b", "model": "eath", "weight": 0.3},
        ]);
        assert!(b
            .generate_hybrid_narrative_impl(components, "hybrid-out", None, None)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_generate_hybrid_narrative_empty_components() {
        let b = backend();
        let components = serde_json::json!([]);
        assert!(b
            .generate_hybrid_narrative_impl(components, "out", None, None)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_list_synthetic_runs_returns_paginated() {
        let b = backend();
        // Empty case.
        let runs = b.list_synthetic_runs_impl("hamlet", None).await.unwrap();
        assert!(runs.as_array().unwrap().is_empty());

        // Populate three runs and re-list.
        write_run_summary(&b, "hamlet");
        write_run_summary(&b, "hamlet");
        write_run_summary(&b, "hamlet");
        let runs = b
            .list_synthetic_runs_impl("hamlet", Some(2))
            .await
            .unwrap();
        let arr = runs.as_array().unwrap();
        assert_eq!(arr.len(), 2, "limit clamps to 2");
    }

    #[tokio::test]
    async fn test_get_fidelity_report_returns_metrics() {
        let b = backend();
        let run_id = Uuid::now_v7();
        let report = dummy_fidelity(run_id, "hamlet");
        save_fidelity_report(b.hypergraph().store(), &report).unwrap();

        let got = b
            .get_fidelity_report_impl("hamlet", &run_id.to_string())
            .await
            .unwrap();
        assert_eq!(got["run_id"].as_str().unwrap(), run_id.to_string());
        assert_eq!(got["model"], "eath");
        assert_eq!(got["passed"], true);
    }

    #[tokio::test]
    async fn test_get_fidelity_report_missing() {
        let b = backend();
        let run_id = Uuid::now_v7();
        let result = b
            .get_fidelity_report_impl("hamlet", &run_id.to_string())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_compute_pattern_significance_round_trip() {
        let b = backend();
        let result = b
            .compute_pattern_significance_impl(
                "hamlet",
                "temporal_motifs",
                Some(50),
                None,
                None,
            )
            .await
            .unwrap();
        assert!(result["job_id"].as_str().is_some());
        assert_eq!(result["status"], "Pending");
    }

    #[tokio::test]
    async fn test_compute_pattern_significance_unknown_metric() {
        let b = backend();
        let result = b
            .compute_pattern_significance_impl("hamlet", "nonsense", None, None, None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_compute_pattern_significance_routes_contagion_to_dedicated_tool() {
        let b = backend();
        let result = b
            .compute_pattern_significance_impl("hamlet", "contagion", None, None, None)
            .await;
        // Should reject — contagion has its own tool.
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_simulate_higher_order_contagion_round_trip() {
        let b = backend();
        let params = serde_json::json!({
            "beta_per_size": [0.05, 0.1],
            "gamma": 0.1,
            "threshold": {"kind": "absolute", "value": 1},
            "seed_strategy": {"kind": "random_fraction", "fraction": 0.05},
            "max_steps": 30,
            "rng_seed": 42
        });
        let result = b
            .simulate_higher_order_contagion_impl("hamlet", params, Some(50), None)
            .await
            .unwrap();
        assert!(result["job_id"].as_str().is_some());
        assert_eq!(result["status"], "Pending");
    }

    #[tokio::test]
    async fn test_simulate_higher_order_contagion_default_model_is_eath() {
        let b = backend();
        let params = serde_json::json!({
            "beta_per_size": [0.05],
            "gamma": 0.1,
            "threshold": {"kind": "absolute", "value": 1},
            "seed_strategy": {"kind": "random_fraction", "fraction": 0.05},
            "max_steps": 10,
            "rng_seed": 1
        });
        let result = b
            .simulate_higher_order_contagion_impl("hamlet", params, None, None)
            .await
            .unwrap();
        let job_id = result["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(job_id).unwrap();
        match job.job_type {
            InferenceJobType::SurrogateContagionSignificance { model, .. } => {
                assert_eq!(model, "eath", "default model is eath");
            }
            other => panic!(
                "expected SurrogateContagionSignificance, got {other:?}"
            ),
        }
    }

    #[tokio::test]
    async fn test_simulate_higher_order_contagion_invalid_params() {
        let b = backend();
        let params = serde_json::json!({"missing": "required fields"});
        let result = b
            .simulate_higher_order_contagion_impl("hamlet", params, None, None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_default_model_is_eath_when_omitted() {
        // Single end-to-end check across all submission tools with model=None.
        let b = backend();

        let r = b.calibrate_surrogate_impl("n", None).await.unwrap();
        let id = r["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(id).unwrap();
        match job.job_type {
            InferenceJobType::SurrogateCalibration { model, .. } => {
                assert_eq!(model, "eath")
            }
            other => panic!("calibrate: {other:?}"),
        }

        let r = b
            .generate_synthetic_narrative_impl("a", "b", None, None, None, None, None)
            .await
            .unwrap();
        let id = r["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(id).unwrap();
        match job.job_type {
            InferenceJobType::SurrogateGeneration { model, .. } => {
                assert_eq!(model, "eath")
            }
            other => panic!("generate: {other:?}"),
        }

        let r = b
            .compute_pattern_significance_impl("n", "patterns", None, None, None)
            .await
            .unwrap();
        let id = r["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(id).unwrap();
        match job.job_type {
            InferenceJobType::SurrogateSignificance { model, .. } => {
                assert_eq!(model, "eath")
            }
            other => panic!("significance: {other:?}"),
        }
    }

    // ─── Phase 13c — compute_dual_significance MCP roundtrip ─────

    /// T6 — submitting `compute_dual_significance` through the embedded
    /// backend yields a queued `SurrogateDualSignificance` job whose payload
    /// matches the request (default models, clamped k, normalized metric).
    #[tokio::test]
    async fn test_dual_sig_mcp_roundtrip() {
        let b = backend();
        // Default models when omitted.
        let r = b
            .compute_dual_significance_impl("hamlet", "patterns", Some(50), None)
            .await
            .unwrap();
        assert!(r["job_id"].as_str().is_some(), "job_id present");
        assert_eq!(r["status"], "Pending");
        let job_id = r["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(job_id).unwrap();
        match job.job_type {
            InferenceJobType::SurrogateDualSignificance {
                narrative_id,
                metric,
                k_per_model,
                models,
            } => {
                assert_eq!(narrative_id, "hamlet");
                assert_eq!(metric, "patterns");
                assert_eq!(k_per_model, 50);
                assert_eq!(
                    models,
                    vec!["eath".to_string(), "nudhy".to_string()],
                    "default models must be eath + nudhy"
                );
            }
            other => panic!("expected SurrogateDualSignificance, got {other:?}"),
        }

        // Explicit models override default + cap clamp.
        let r2 = b
            .compute_dual_significance_impl(
                "hamlet",
                "communities",
                Some(u16::MAX),
                Some(vec!["nudhy".into()]),
            )
            .await
            .unwrap();
        let job2 = b
            .job_queue_arc()
            .get_job(r2["job_id"].as_str().unwrap())
            .unwrap();
        match job2.job_type {
            InferenceJobType::SurrogateDualSignificance {
                k_per_model,
                models,
                ..
            } => {
                assert_eq!(
                    k_per_model,
                    crate::synth::significance::K_MAX,
                    "k_per_model must be clamped at K_MAX"
                );
                assert_eq!(models, vec!["nudhy".to_string()]);
            }
            other => panic!("expected SurrogateDualSignificance, got {other:?}"),
        }

        // Empty narrative_id → InvalidInput.
        let bad = b
            .compute_dual_significance_impl("", "patterns", None, None)
            .await;
        assert!(bad.is_err(), "empty narrative_id must error");

        // Contagion metric → InvalidInput with routing hint.
        let routed = b
            .compute_dual_significance_impl("hamlet", "contagion", None, None)
            .await;
        assert!(routed.is_err(), "contagion metric must reject");
    }

    // ─── Phase 15c — reconstruct_hypergraph MCP roundtrip ────────

    /// T3 — submitting `reconstruct_hypergraph` through the embedded backend
    /// yields a queued `HypergraphReconstruction` job whose payload
    /// matches the request (narrative_id threaded into both the variant
    /// payload and `job.parameters`).
    #[tokio::test]
    async fn test_reconstruct_hypergraph_mcp_roundtrip() {
        let b = backend();
        // Submit with no params blob — engine should fall back to defaults.
        let r = b.reconstruct_hypergraph_impl("disinfo-1", None).await.unwrap();
        assert!(r["job_id"].as_str().is_some(), "job_id present");
        assert_eq!(r["status"], "Pending");
        let job_id = r["job_id"].as_str().unwrap();
        let job = b.job_queue_arc().get_job(job_id).unwrap();
        match &job.job_type {
            InferenceJobType::HypergraphReconstruction { narrative_id, .. } => {
                assert_eq!(narrative_id, "disinfo-1");
            }
            other => panic!("expected HypergraphReconstruction, got {other:?}"),
        }
        // narrative_id is also mirrored into job.parameters for the engine
        // resolver — sanity-check.
        assert_eq!(
            job.parameters
                .get("narrative_id")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "disinfo-1"
        );

        // Submit with explicit params override.
        let r2 = b
            .reconstruct_hypergraph_impl(
                "disinfo-2",
                Some(serde_json::json!({"max_order": 2, "lambda_l1": 0.05})),
            )
            .await
            .unwrap();
        let job2 = b.job_queue_arc().get_job(r2["job_id"].as_str().unwrap()).unwrap();
        assert_eq!(
            job2.parameters
                .get("max_order")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            2
        );

        // Empty narrative_id → InvalidInput.
        assert!(b.reconstruct_hypergraph_impl("", None).await.is_err());
    }

    // ─── Phase 16c — opinion dynamics MCP roundtrip ──────────────

    /// T4 — submitting `simulate_opinion_dynamics` through the embedded
    /// backend yields an inline OpinionDynamicsReport (no job queue), and
    /// `simulate_opinion_phase_transition` returns a well-formed
    /// PhaseTransitionReport.
    #[tokio::test]
    async fn test_opinion_dynamics_mcp_roundtrip() {
        use crate::types::*;
        let b = backend();
        let hg = b.hypergraph();

        // Seed a triangle (3 entities + 1 size-3 situation).
        let nid = "od-mcp-narr";
        let now = chrono::Utc::now();
        let mut ids = Vec::new();
        for i in 0..3 {
            let id = Uuid::now_v7();
            let e = Entity {
                id,
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("a{i}")}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: Some(ExtractionMethod::HumanEntered),
                narrative_id: Some(nid.into()),
                created_at: now,
                updated_at: now,
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(e).unwrap();
            ids.push(id);
        }
        let sit_id = Uuid::now_v7();
        let s = Situation {
            id: sit_id,
            name: None,
            description: None,
            properties: serde_json::Value::Null,
            temporal: AllenInterval {
                start: Some(now),
                end: Some(now + chrono::Duration::seconds(60)),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("seed")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(s).unwrap();
        for id in &ids {
            hg.add_participant(Participation {
                entity_id: *id,
                situation_id: sit_id,
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        // simulate_opinion_dynamics: defaults (None) — runs synchronously.
        let r = b
            .simulate_opinion_dynamics_impl(
                nid,
                Some(serde_json::json!({
                    "confidence_bound": 0.5,
                    "convergence_rate": 0.5,
                    "max_steps": 5000,
                    "model": "pairwise_within",
                    "hyperedge_selection": "uniform_random",
                    "initial_opinion_distribution": {"kind": "uniform"},
                    "convergence_tol": 1.0e-4,
                    "convergence_window": 100,
                    "seed": 42
                })),
            )
            .await
            .unwrap();
        assert!(r["run_id"].as_str().is_some());
        let report = &r["report"];
        assert!(report["num_steps_executed"].as_u64().unwrap() > 0);

        // simulate_opinion_phase_transition: 3-point sweep.
        let pt = b
            .simulate_opinion_phase_transition_impl(
                nid,
                [
                    serde_json::json!(0.05),
                    serde_json::json!(0.5),
                    serde_json::json!(3),
                ],
                None,
            )
            .await
            .unwrap();
        assert_eq!(pt["c_values"].as_array().unwrap().len(), 3);
        assert_eq!(pt["convergence_times"].as_array().unwrap().len(), 3);
        assert!(pt["spike_threshold"].as_f64().is_some());

        // Empty narrative_id → InvalidInput.
        assert!(b.simulate_opinion_dynamics_impl("", None).await.is_err());
    }
}
