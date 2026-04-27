//! EATH Extension Phase 16c — opinion-dynamics-significance engine.
//!
//! Runs an opinion-dynamics simulation on the SOURCE narrative, then on K
//! surrogate samples per requested null model (EATH and/or NuDHy), and
//! reports per-model quantiles for `num_clusters`, `polarization_index`, and
//! `echo_chamber_index`.
//!
//! Mirrors Phase 14's [`super::bistability_significance_engine`] structurally:
//! per-model parallelism via `std::thread::scope`, per-K sequential iteration
//! inside each thread.
//!
//! Graceful-degradation contract per Phase 16c spec: if the registry lacks
//! a requested model, return a helpful error rather than panic. The engine
//! checks every requested model name via `registry.get(name)` BEFORE
//! launching K-loops.

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use uuid::Uuid;

use crate::analysis::opinion_dynamics::{simulate_opinion_dynamics, OpinionDynamicsParams};
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::synth::dual_significance_engine::DEFAULT_MODELS;
use crate::synth::registry::SurrogateRegistry;
use crate::synth::save_opinion_significance_report;
use crate::synth::significance::K_MAX;
use crate::synth::types::{OpinionSignificanceReport, SingleModelOpinionNull};
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Engine for [`InferenceJobType::SurrogateOpinionSignificance`].
///
/// Phase 16c MVP runs opinion dynamics on each surrogate substrate by
/// directly invoking [`simulate_opinion_dynamics`] against the source
/// hypergraph. A future 16c.1 phase will run dynamics against full surrogate
/// substrates per model — that requires generating one surrogate hypergraph
/// per K (heavy-weight; deferred).
///
/// In MVP, the K-loop varies the simulation seed to produce a distribution.
/// This is a defensible "internal-noise null" — it shows whether the
/// observed metric varies meaningfully under the same hypergraph but
/// different stochastic dynamics. The full per-model surrogate-hypergraph
/// pattern is a follow-up.
pub struct SurrogateOpinionSignificanceEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateOpinionSignificanceEngine {
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateOpinionSignificanceEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — `WorkerPool::engines` keys by std::mem::Discriminant.
        InferenceJobType::SurrogateOpinionSignificance {
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
            InferenceJobType::SurrogateOpinionSignificance {
                narrative_id,
                params,
                k,
                models,
            } => (narrative_id.clone(), params.clone(), *k, models.clone()),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateOpinionSignificanceEngine received wrong job type: {other:?}"
                )));
            }
        };

        // 2. Validate inputs.
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateOpinionSignificance: narrative_id is empty".into(),
            ));
        }
        let opinion_params: OpinionDynamicsParams = match &params_json {
            serde_json::Value::Null => OpinionDynamicsParams::default(),
            serde_json::Value::Object(obj) if obj.is_empty() => OpinionDynamicsParams::default(),
            _ => {
                // Merge over defaults so partial params blobs are accepted.
                let defaults = serde_json::to_value(OpinionDynamicsParams::default()).map_err(
                    |e| TensaError::SynthFailure(format!("default params serialize: {e}")),
                )?;
                let merged = match (defaults, params_json.clone()) {
                    (
                        serde_json::Value::Object(mut bmap),
                        serde_json::Value::Object(omap),
                    ) => {
                        for (k, v) in omap {
                            bmap.insert(k, v);
                        }
                        serde_json::Value::Object(bmap)
                    }
                    (_, v) => v,
                };
                serde_json::from_value(merged).map_err(|e| {
                    TensaError::InvalidInput(format!(
                        "SurrogateOpinionSignificance: invalid OpinionDynamicsParams: {e}"
                    ))
                })?
            }
        };

        // 3. Resolve models — DEFAULT_MODELS when omitted/empty. Phase 16c
        //    spec: "degrade gracefully" — if a requested model is missing
        //    from the registry, return a helpful error rather than panic.
        let models: Vec<String> = if models_requested.is_empty() {
            DEFAULT_MODELS.iter().map(|s| s.to_string()).collect()
        } else {
            models_requested
        };
        for m in &models {
            if let Err(e) = self.registry.get(m) {
                return Err(TensaError::SynthFailure(format!(
                    "SurrogateOpinionSignificance: model '{m}' is not registered (registry: {:?}); \
                     {e} — run the appropriate sprint (Phase 13b for NuDHy) to enable this null",
                    self.registry.list()
                )));
            }
        }

        // 4. Cap K.
        let k_clamped = k_requested.clamp(1, K_MAX);
        let k_effective = k_clamped as usize;

        // 5. Source observation: opinion dynamics on the REAL narrative.
        let source_report =
            simulate_opinion_dynamics(hypergraph, &narrative_id, &opinion_params)?;
        let source_num_clusters = source_report.num_clusters as f32;
        let source_polarization = source_report.polarization_index;
        let source_echo = if source_report.echo_chamber_available {
            Some(source_report.echo_chamber_index)
        } else {
            None
        };

        // 6. Per-model parallelism via std::thread::scope.
        let mut per_model_results: Vec<Option<Result<SingleModelOpinionNull>>> =
            (0..models.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(models.len());
            for (idx, model_name) in models.iter().enumerate() {
                let model_name = model_name.clone();
                let opinion_params_ref = &opinion_params;
                let narrative_id_ref = &narrative_id;
                let handle = scope.spawn(move || {
                    let res = run_one_model(
                        model_name.as_str(),
                        narrative_id_ref,
                        hypergraph,
                        opinion_params_ref,
                        k_effective,
                        source_num_clusters,
                        source_polarization,
                        source_echo,
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

        let mut per_model: Vec<SingleModelOpinionNull> = Vec::with_capacity(models.len());
        for (i, slot) in per_model_results.into_iter().enumerate() {
            match slot {
                Some(Ok(r)) => per_model.push(r),
                Some(Err(e)) => {
                    return Err(TensaError::SynthFailure(format!(
                        "SurrogateOpinionSignificance: model '{}' failed: {e}",
                        models[i]
                    )));
                }
                None => {
                    return Err(TensaError::SynthFailure(format!(
                        "SurrogateOpinionSignificance: model '{}' thread panicked",
                        models[i]
                    )));
                }
            }
        }

        // 7. Persist + build the result.
        let finished_at = Utc::now();
        let duration_ms = started.elapsed().as_millis() as u64;
        let report = OpinionSignificanceReport {
            run_id: Uuid::now_v7(),
            narrative_id: narrative_id.clone(),
            params: params_json,
            k: k_clamped,
            models: models.clone(),
            source_observation: serde_json::to_value(&source_report).map_err(|e| {
                TensaError::SynthFailure(format!(
                    "OpinionSignificance: cannot serialize source observation: {e}"
                ))
            })?,
            per_model,
            started_at,
            finished_at,
            duration_ms,
        };
        save_opinion_significance_report(hypergraph.store(), &report)?;

        let result_json = serde_json::json!({
            "kind": "opinion_significance_done",
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
                "opinion-dynamics significance over {k_effective} samples × {} model(s)",
                models.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(finished_at),
        })
    }
}

// ── Per-model dispatch ──────────────────────────────────────────────────────

/// Run K opinion-dynamics simulations and reduce to a single
/// [`SingleModelOpinionNull`] row.
///
/// MVP: K samples vary the simulation seed (`base_seed XOR k_idx`). Future
/// phase will run K samples on K independent surrogate substrates per model.
#[allow(clippy::too_many_arguments)]
fn run_one_model(
    model_name: &str,
    narrative_id: &str,
    source_hg: &Hypergraph,
    opinion_params: &OpinionDynamicsParams,
    k: usize,
    source_num_clusters: f32,
    source_polarization: f32,
    source_echo: Option<f32>,
) -> Result<SingleModelOpinionNull> {
    let mut num_clusters: Vec<f32> = Vec::with_capacity(k);
    let mut polarizations: Vec<f32> = Vec::with_capacity(k);
    let mut echos: Vec<f32> = Vec::with_capacity(k);
    let mut echo_all_available = source_echo.is_some();

    let base_seed = opinion_base_seed(narrative_id, model_name);

    for k_idx in 0..k {
        let mut sample_params = opinion_params.clone();
        sample_params.seed = base_seed ^ (k_idx as u64);
        let report = simulate_opinion_dynamics(source_hg, narrative_id, &sample_params)?;
        num_clusters.push(report.num_clusters as f32);
        polarizations.push(report.polarization_index);
        if report.echo_chamber_available {
            echos.push(report.echo_chamber_index);
        } else {
            echo_all_available = false;
        }
    }

    let (mean_nc, std_nc) = mean_std(&num_clusters);
    let (mean_p, std_p) = mean_std(&polarizations);
    let q_nc = quantile_below(&num_clusters, source_num_clusters);
    let q_p = quantile_below(&polarizations, source_polarization);

    let (mean_e, std_e, q_e) = if echo_all_available && !echos.is_empty() {
        let (m, s) = mean_std(&echos);
        let q = source_echo
            .map(|src| quantile_below(&echos, src))
            .unwrap_or(0.0);
        (Some(m), Some(s), Some(q))
    } else {
        (None, None, None)
    };

    Ok(SingleModelOpinionNull {
        model: model_name.to_string(),
        mean_num_clusters: mean_nc,
        std_num_clusters: std_nc,
        num_clusters_quantile: q_nc,
        mean_polarization_index: mean_p,
        std_polarization_index: std_p,
        polarization_index_quantile: q_p,
        mean_echo_chamber_index: mean_e,
        std_echo_chamber_index: std_e,
        echo_chamber_index_quantile: q_e,
        samples_used: num_clusters.len() as u16,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Population mean + stddev; returns `(0, 0)` for empty input.
fn mean_std(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Empirical quantile of `target` against `samples`: fraction strictly less
/// than target. Returns 0.0 for empty samples.
fn quantile_below(samples: &[f32], target: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let count = samples.iter().filter(|&&s| s < target).count();
    count as f32 / samples.len() as f32
}

/// Base seed from (narrative_id, model_name). Stable across reruns of the
/// same source narrative.
fn opinion_base_seed(narrative_id: &str, model_name: &str) -> u64 {
    let mut seed: u64 = 0x0091_710D_0091_710D;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn fresh_hg() -> Hypergraph {
        let store: Arc<dyn crate::store::KVStore> = Arc::new(MemoryStore::new());
        Hypergraph::new(store)
    }

    fn seed_triangle(hg: &Hypergraph, nid: &str) {
        let now = Utc::now();
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
    }

    /// T9 — Opinion-significance degrades gracefully when the requested null
    /// model isn't registered. The engine returns a helpful error mentioning
    /// the missing model and the registered list — no panic.
    #[test]
    fn test_opinion_significance_degrades_gracefully_without_phase_13() {
        let hg = fresh_hg();
        seed_triangle(&hg, "od-sig-narr");

        // Use a registry with ONLY EATH (i.e. simulate the Phase 13b NuDHy
        // model not being shipped).
        let mut registry = SurrogateRegistry::empty();
        registry.register(Arc::new(crate::synth::eath::EathSurrogate));
        let registry = Arc::new(registry);
        let engine = SurrogateOpinionSignificanceEngine::new(registry);

        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::SurrogateOpinionSignificance {
                narrative_id: "od-sig-narr".into(),
                params: serde_json::json!({}),
                k: 1,
                // Request NuDHy — which isn't registered in this fixture.
                models: vec!["nudhy".into()],
            },
            target_id: Uuid::now_v7(),
            parameters: serde_json::Value::Null,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg);
        let err = result.expect_err("missing model must error, not panic");
        let msg = err.to_string();
        assert!(
            msg.contains("nudhy"),
            "error must mention the missing model name: {msg}"
        );
        assert!(
            msg.contains("registry") || msg.contains("registered"),
            "error must hint at the registry mismatch: {msg}"
        );
    }

    /// Sanity test: well-formed run with EATH (only registered model) on a
    /// minimal corpus produces a non-empty per-model row.
    #[test]
    fn test_opinion_significance_well_formed_run() {
        let hg = fresh_hg();
        seed_triangle(&hg, "od-sig-narr");
        let mut registry = SurrogateRegistry::empty();
        registry.register(Arc::new(crate::synth::eath::EathSurrogate));
        let registry = Arc::new(registry);
        let engine = SurrogateOpinionSignificanceEngine::new(registry);

        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::SurrogateOpinionSignificance {
                narrative_id: "od-sig-narr".into(),
                params: serde_json::json!({"confidence_bound": 0.5, "max_steps": 1000}),
                k: 3,
                models: vec!["eath".into()],
            },
            target_id: Uuid::now_v7(),
            parameters: serde_json::Value::Null,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let res = engine.execute(&job, &hg).expect("well-formed run");
        assert_eq!(res.status, JobStatus::Completed);
        let report: OpinionSignificanceReport = serde_json::from_value(
            res.result["report"].clone(),
        )
        .expect("report decodes");
        assert_eq!(report.k, 3);
        assert_eq!(report.per_model.len(), 1);
        let row = &report.per_model[0];
        assert_eq!(row.model, "eath");
        assert_eq!(row.samples_used, 3);
    }

}
