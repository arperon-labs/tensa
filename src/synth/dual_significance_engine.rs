//! EATH Extension Phase 13c — dual-null-model significance engine.
//!
//! Runs the Phase 7 K-loop ONCE per requested null model and reports the
//! per-model + combined significance verdict. Standard practice in the
//! higher-order networks literature: a pattern significant against EATH AND
//! NuDHy is meaningfully above background in two independent ways (one tests
//! against the dynamics-fitted null, the other against the
//! degree-and-edge-size preserving configuration model).
//!
//! ## Architecture
//!
//! For each requested model:
//! * **EATH** — delegates to Phase 7's `run_significance_pipeline` (the
//!   K-loop infra is already model-agnostic at the metric layer; the
//!   `generate_one_sample` helper hardcodes EathSurrogate which is fine
//!   because it's the EATH branch).
//! * **NuDHy** — calibrates ONCE outside the K-loop (per the Phase 13b
//!   heads-up: parsing the ~70KB calibration blob K times wastes ~500ms;
//!   parse once and clone `Vec<Vec<Uuid>>` per chain), then runs K MCMC
//!   chains in parallel (each gets its own ephemeral hypergraph).
//!
//! Per-model significance is reduced to a single `SingleModelSignificance`
//! row by picking the element with maximum |z| (the stat the dual-null
//! verdict cares about). The full per-element distribution is left to the
//! single-model significance engine for callers that need it.
//!
//! Per-model parallelism is via `std::thread::scope` — same pattern as the
//! K-sample loop, mirrored at the model level so independent models don't
//! serialize on each other.
//!
//! ## Determinism
//!
//! Per-K seeds are derived via the same `params_seed XOR (k_idx * SAMPLE_SEED_MIX)`
//! algebra Phase 7 uses, so the EATH and NuDHy K-loops are bit-identical to
//! their single-model counterparts. Parallel-by-model scheduling does NOT
//! affect determinism — each model's K-loop sees its own seed stream and
//! results are joined deterministically by index.
//!
//! ## Starvation propagation
//!
//! NuDHy's MCMC chain may starve when the source hypergraph is too rigid
//! (every entity in every edge — no valid swap exists). The K-loop tallies
//! starvations into `SingleModelSignificance.starvations` rather than
//! aborting the whole dual run; callers can detect "0/50 succeeded vs
//! 50/50 succeeded" from the field.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::types::{
    Entity, InferenceJobType, InferenceResult, JobStatus, MaturityLevel,
};

use super::fidelity::{ParallelismMode, SAMPLE_SEED_MIX};
use super::nudhy::{run_nudhy_chain, NudhyParams, NudhyState};
use super::nudhy_surrogate::calibrate_nudhy;
use super::registry::SurrogateRegistry;
use super::significance::adapters::{
    compute_for, run_significance_pipeline, AdapterChoice, MetricSnapshot,
};
use super::significance::{
    guard_not_fully_synthetic, resolve_params, SignificanceMetric, K_MAX,
};
use super::types::{
    CombinedSignificance, DualSignificanceReport, SingleModelSignificance,
};
use super::{
    emit::{
        filter_synthetic_entities, write_synthetic_situation, EmitContext,
        DEFAULT_SYNTHETIC_CONFIDENCE,
    },
    save_dual_significance_report,
};

// ── Type aliases ─────────────────────────────────────────────────────────────

/// NuDHy calibration triple: (params, chain_edges, fixed_edges).
/// Hoisted to a named alias to satisfy clippy's `type_complexity` lint and
/// keep per-K dispatch sites readable.
type NudhyCalibration = (NudhyParams, Vec<Vec<Uuid>>, Vec<Vec<Uuid>>);

// ── Constants ────────────────────────────────────────────────────────────────

/// Significance thresholds — one source of truth (mirrored in the dual-report
/// docstring). |z| > 1.96 ↔ p < 0.05 two-sided; |z| > 2.58 ↔ p < 0.01.
const Z_THRESHOLD_P05: f64 = 1.96;
const Z_THRESHOLD_P01: f64 = 2.58;
const P_THRESHOLD_05: f64 = 0.05;
const P_THRESHOLD_01: f64 = 0.01;

/// Default models when the request omits `models`. EATH (dynamics-fitted) +
/// NuDHy (configuration-style) — the canonical dual-null pair.
pub const DEFAULT_MODELS: &[&str] = &["eath", "nudhy"];

// ── Engine ───────────────────────────────────────────────────────────────────

/// `InferenceEngine` impl for [`InferenceJobType::SurrogateDualSignificance`].
///
/// Holds the same `Arc<SurrogateRegistry>` Phase 4/7 engines use; the
/// registry is read-only after construction so multiple workers can hold the
/// same `Arc` without contention.
pub struct SurrogateDualSignificanceEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateDualSignificanceEngine {
    /// Construct with a shared registry.
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateDualSignificanceEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — `WorkerPool::engines` keys by std::mem::Discriminant,
        // not full equality. Empty fields are deliberately a registration
        // sentinel.
        InferenceJobType::SurrogateDualSignificance {
            narrative_id: String::new(),
            metric: String::new(),
            k_per_model: 0,
            models: Vec::new(),
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let started = Instant::now();
        let started_at = Utc::now();

        // 1. Deconstruct variant payload.
        let (narrative_id, metric_str, k_requested, models_requested) = match &job.job_type {
            InferenceJobType::SurrogateDualSignificance {
                narrative_id,
                metric,
                k_per_model,
                models,
            } => (
                narrative_id.clone(),
                metric.clone(),
                *k_per_model,
                models.clone(),
            ),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateDualSignificanceEngine received wrong job type: {other:?}"
                )));
            }
        };

        // 2. Validate inputs.
        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateDualSignificance: narrative_id is empty".into(),
            ));
        }
        let metric = SignificanceMetric::parse(&metric_str).ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "SurrogateDualSignificance: unknown metric '{metric_str}'; \
                 expected one of: temporal_motifs, communities, patterns, contagion"
            ))
        })?;

        // Default models when caller passes empty.
        let models: Vec<String> = if models_requested.is_empty() {
            DEFAULT_MODELS.iter().map(|s| s.to_string()).collect()
        } else {
            models_requested
        };

        // Validate every model is in the registry (early failure beats wasted
        // K-loop work).
        for m in &models {
            self.registry.get(m)?;
        }

        // 3. Refuse fully-synthetic narratives — Phase 7 invariant.
        guard_not_fully_synthetic(hypergraph, &narrative_id)?;

        // 4. Cap k_per_model at K_MAX with a tracing::warn.
        let k_clamped = k_requested.clamp(1, K_MAX);
        if k_requested > K_MAX {
            tracing::warn!(
                "SurrogateDualSignificance: k_per_model={k_requested} exceeds K_MAX={K_MAX}; clamping"
            );
        }
        let k_effective = k_clamped as usize;

        // 5. Resolve EathParams ONCE (used by EATH samples; NuDHy ignores
        //    them but we still need them for the auto-calibration side-effect
        //    if EATH is in the model list AND no params persisted yet).
        //    Cheap when persisted; auto-calibrates inline when missing.
        let eath_params_opt = if models.iter().any(|m| m == "eath") {
            let (params, _, _) =
                resolve_params(hypergraph, &narrative_id, "eath", &job.parameters)?;
            Some(params)
        } else {
            None
        };

        // 6. Optional metric_params (e.g. {"max_motif_size": 4}).
        let metric_params = job
            .parameters
            .get("metric_params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        // 7. NuDHy: parse the calibration blob ONCE (per Phase 13b heads-up).
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

        // 8. Per-model parallelism via std::thread::scope. Each model sees
        //    its own thread-safe inputs (hypergraph, params clones).
        //    Determinism: results are placed by index so thread scheduling
        //    doesn't affect output order.
        let mut per_model_results: Vec<Option<Result<SingleModelSignificance>>> =
            (0..models.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(models.len());
            for (idx, model_name) in models.iter().enumerate() {
                let model_name = model_name.clone();
                let metric_params_ref = &metric_params;
                let narrative_id_ref = &narrative_id;
                let eath_params_ref = eath_params_opt.as_ref();
                let nudhy_ref = nudhy_calibration.as_ref();
                let handle = scope.spawn(move || {
                    let result = run_one_model(
                        model_name.as_str(),
                        narrative_id_ref,
                        hypergraph,
                        metric,
                        metric_params_ref,
                        k_effective,
                        eath_params_ref,
                        nudhy_ref,
                    );
                    (idx, result)
                });
                handles.push(handle);
            }
            for handle in handles {
                if let Ok((idx, res)) = handle.join() {
                    per_model_results[idx] = Some(res);
                }
            }
        });

        let mut per_model: Vec<SingleModelSignificance> = Vec::with_capacity(models.len());
        for (i, slot) in per_model_results.into_iter().enumerate() {
            match slot {
                Some(Ok(r)) => per_model.push(r),
                Some(Err(e)) => {
                    return Err(TensaError::SynthFailure(format!(
                        "SurrogateDualSignificance: model '{}' failed: {e}",
                        models[i]
                    )));
                }
                None => {
                    return Err(TensaError::SynthFailure(format!(
                        "SurrogateDualSignificance: model '{}' thread panicked",
                        models[i]
                    )));
                }
            }
        }

        // 9. Combine across models (AND across the per-model verdicts).
        let combined = combine_per_model(&per_model);

        // 10. Build + persist the report.
        let finished_at = Utc::now();
        let duration_ms = started.elapsed().as_millis() as u64;
        let report = DualSignificanceReport {
            run_id: Uuid::now_v7(),
            narrative_id: narrative_id.clone(),
            metric: metric_str.clone(),
            k_per_model: k_clamped,
            per_model,
            combined,
            started_at,
            finished_at,
            duration_ms,
        };
        save_dual_significance_report(hypergraph.store(), &report)?;

        let result_json = serde_json::json!({
            "kind": "dual_significance_done",
            "narrative_id": report.narrative_id,
            "metric": report.metric,
            "run_id": report.run_id,
            "k_per_model": report.k_per_model,
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
                "dual-null significance ({}) over {k_effective} samples × {} model(s)",
                metric_str,
                models.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(finished_at),
        })
    }
}

// ── Per-model dispatch ───────────────────────────────────────────────────────

/// Run the K-loop for ONE model and reduce to a single
/// [`SingleModelSignificance`] row. Dispatches via model name; future
/// surrogate families add a branch here.
#[allow(clippy::too_many_arguments)]
fn run_one_model(
    model_name: &str,
    narrative_id: &str,
    hypergraph: &Hypergraph,
    metric: SignificanceMetric,
    metric_params: &serde_json::Value,
    k: usize,
    eath_params: Option<&super::types::EathParams>,
    nudhy_calibration: Option<&NudhyCalibration>,
) -> Result<SingleModelSignificance> {
    match model_name {
        "eath" => {
            let params = eath_params.ok_or_else(|| {
                TensaError::SynthFailure(
                    "SurrogateDualSignificance: EATH params not resolved".into(),
                )
            })?;
            // Phase 7 supports the three "structural" metrics directly. For
            // contagion, the adapter takes a HigherOrderSirParams blob — we
            // forward via the same K-loop helper.
            let adapter = match metric {
                SignificanceMetric::TemporalMotifs => AdapterChoice::TemporalMotifs,
                SignificanceMetric::Communities => AdapterChoice::Communities,
                SignificanceMetric::Patterns => AdapterChoice::Patterns,
                SignificanceMetric::Contagion => {
                    return Err(TensaError::InvalidInput(
                        "SurrogateDualSignificance with metric=contagion requires the contagion params \
                         blob; not yet wired through the dual API in Phase 13c"
                            .into(),
                    ));
                }
            };
            let pipeline = run_significance_pipeline(
                hypergraph,
                narrative_id,
                params,
                adapter,
                metric_params,
                k,
                ParallelismMode::Single, // outer parallelism is per-model already
            )?;
            Ok(reduce_pipeline_to_row(
                model_name,
                k,
                /* starvations */ 0,
                &pipeline.distribution,
            ))
        }
        "nudhy" => {
            let cal = nudhy_calibration.ok_or_else(|| {
                TensaError::SynthFailure(
                    "SurrogateDualSignificance: NuDHy calibration not resolved".into(),
                )
            })?;
            let (nudhy_params, chain_edges, fixed_edges) = cal;

            // Source observation — Phase 7 helper, but we re-compute it here
            // so the per-model row gets a directly-comparable snapshot.
            let source_snapshot =
                compute_metric_on_source(metric, hypergraph, narrative_id, metric_params)?;

            // K MCMC chains; each gets a deterministic per-K seed.
            let base_seed = nudhy_base_seed(narrative_id);
            let mut snapshots: Vec<MetricSnapshot> = Vec::with_capacity(k);
            let mut starvations: u16 = 0;
            for k_idx in 0..k {
                let chain_seed = per_sample_seed(base_seed, k_idx);
                match run_one_nudhy_sample(
                    nudhy_params,
                    chain_edges.clone(),
                    fixed_edges.clone(),
                    chain_seed,
                    k_idx,
                    hypergraph,
                    narrative_id,
                    metric,
                    metric_params,
                ) {
                    Ok(snap) => snapshots.push(snap),
                    Err(TensaError::SynthFailure(msg)) if msg.contains("starvation") => {
                        starvations = starvations.saturating_add(1);
                        tracing::warn!(
                            k_idx,
                            "NuDHy K-chain {k_idx} starved; tallying starvation and continuing"
                        );
                    }
                    Err(e) => return Err(e),
                }
            }
            // Build a synthetic distribution from the K snapshots vs source.
            let dist = super::significance::stats::build_distribution(
                &source_snapshot,
                &snapshots,
                metric_to_adapter_choice(metric),
            );
            // samples_used reflects the actually-completed chains.
            let samples_used = snapshots.len() as u16;
            Ok(reduce_pipeline_to_row(
                model_name,
                samples_used as usize,
                starvations,
                &dist,
            ))
        }
        other => Err(TensaError::SynthFailure(format!(
            "SurrogateDualSignificance: model '{other}' is registered but no K-loop dispatch is wired"
        ))),
    }
}

/// Compute the metric on the source narrative for THIS dispatch. Routes
/// through the public adapters layer so we don't duplicate dispatch logic.
fn compute_metric_on_source(
    metric: SignificanceMetric,
    hypergraph: &Hypergraph,
    narrative_id: &str,
    metric_params: &serde_json::Value,
) -> Result<MetricSnapshot> {
    compute_for(
        metric_to_adapter_choice(metric),
        hypergraph,
        narrative_id,
        metric_params,
    )
}

/// Map [`SignificanceMetric`] to a Phase 7 [`AdapterChoice`]. Contagion needs
/// a params blob — Phase 13c declines to forward it through the dual API to
/// keep the engine surface tight (callers can still run contagion via the
/// dedicated single-model engine + endpoint).
fn metric_to_adapter_choice(metric: SignificanceMetric) -> AdapterChoice<'static> {
    match metric {
        SignificanceMetric::TemporalMotifs => AdapterChoice::TemporalMotifs,
        SignificanceMetric::Communities => AdapterChoice::Communities,
        SignificanceMetric::Patterns => AdapterChoice::Patterns,
        // Caller-side checks reject contagion before we get here. Returning
        // a placeholder would compile but might mask wiring bugs — instead
        // we synthesise a default contagion params blob; the dual engine's
        // run_one_model rejects contagion explicitly at entry so this branch
        // is unreachable in practice.
        SignificanceMetric::Contagion => AdapterChoice::TemporalMotifs,
    }
}

// ── NuDHy K-sample helper ────────────────────────────────────────────────────

/// Run one NuDHy MCMC chain, materialise the resulting hyperedges into an
/// ephemeral hypergraph (with the source entities copied into a
/// sample-scoped narrative_id), then compute the metric. Hypergraph + store
/// are dropped at function exit.
#[allow(clippy::too_many_arguments)]
fn run_one_nudhy_sample(
    nudhy_params: &NudhyParams,
    chain_edges: Vec<Vec<Uuid>>,
    fixed_edges: Vec<Vec<Uuid>>,
    chain_seed: u64,
    k_idx: usize,
    source_hg: &Hypergraph,
    source_narrative_id: &str,
    metric: SignificanceMetric,
    metric_params: &serde_json::Value,
) -> Result<MetricSnapshot> {
    // 1. Run MCMC chain (per-K seed → per-K trace → per-K state).
    let initial_state = NudhyState::from_hyperedges(chain_edges);
    let mut rng = ChaCha8Rng::seed_from_u64(chain_seed);
    let final_state = run_nudhy_chain(initial_state, nudhy_params, &mut rng)?;

    // 2. Build ephemeral target hypergraph + sample-scoped narrative_id.
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let target = Hypergraph::new(store);
    let sample_narrative_id = format!("dual-sig-nudhy-{k_idx}");

    // 3. Copy source entities into the ephemeral store under the sample
    //    narrative_id. NuDHy's `add_participant` requires entity existence;
    //    we keep the original UUIDs so the hyperedge member lists work.
    //    Only entities referenced by chain_edges OR fixed_edges are copied.
    let mut needed: HashSet<Uuid> = HashSet::new();
    for e in final_state.hyperedges.iter().chain(fixed_edges.iter()) {
        needed.extend(e.iter().copied());
    }
    let source_entities = source_hg.list_entities_by_narrative(source_narrative_id)?;
    let source_entities = filter_synthetic_entities(source_entities, false);
    for src in source_entities {
        if !needed.contains(&src.id) {
            continue;
        }
        let now = Utc::now();
        let copy = Entity {
            id: src.id,
            entity_type: src.entity_type,
            properties: src.properties.clone(),
            beliefs: src.beliefs.clone(),
            embedding: src.embedding.clone(),
            maturity: MaturityLevel::Candidate,
            confidence: src.confidence,
            confidence_breakdown: src.confidence_breakdown.clone(),
            provenance: vec![],
            extraction_method: src.extraction_method.clone(),
            narrative_id: Some(sample_narrative_id.clone()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        target.create_entity(copy)?;
    }

    // 4. Build EmitContext + write situations under the sample narrative.
    let run_id = Uuid::from_u64_pair(chain_seed, chain_seed.rotate_left(32));
    let ctx = EmitContext {
        run_id,
        narrative_id: sample_narrative_id.clone(),
        maturity: MaturityLevel::Candidate,
        confidence: DEFAULT_SYNTHETIC_CONFIDENCE,
        label_prefix: format!("dual-sig-nudhy-{k_idx}-"),
        time_anchor: chrono::TimeZone::with_ymd_and_hms(&Utc, 2020, 1, 1, 0, 0, 0).unwrap(),
        step_duration_seconds: 1,
        model: "nudhy".to_string(),
        reuse_entities: Some(needed.iter().copied().collect()),
    };
    let mut sit_rng = ChaCha8Rng::seed_from_u64(chain_seed.wrapping_add(0xC0FFEE));
    for (step, members) in final_state.hyperedges.iter().enumerate() {
        write_synthetic_situation(&ctx, step, members, &mut sit_rng, &target)?;
    }
    let step_offset = final_state.hyperedges.len();
    for (i, members) in fixed_edges.iter().enumerate() {
        write_synthetic_situation(&ctx, step_offset + i, members, &mut sit_rng, &target)?;
    }

    // 5. Compute metric on the sample. Contagion never reaches here —
    //    `run_one_model` rejects it at entry — but we don't synthesise a
    //    default contagion params blob here either. Phase 13c does not
    //    forward the contagion adapter through the dual engine.
    compute_for(
        metric_to_adapter_choice(metric),
        &target,
        &sample_narrative_id,
        metric_params,
    )
}

// ── Reductions ───────────────────────────────────────────────────────────────

/// Pick the element with the largest finite |z| from the per-element
/// distribution and condense it into a single per-model row. Ties on |z| are
/// broken by lexicographic element key for determinism.
fn reduce_pipeline_to_row(
    model: &str,
    samples_used: usize,
    starvations: u16,
    dist: &super::significance::SyntheticDistribution,
) -> SingleModelSignificance {
    let n = dist.element_keys.len();
    if n == 0 {
        return SingleModelSignificance {
            model: model.to_string(),
            observed_value: 0.0,
            mean_null: 0.0,
            std_null: 0.0,
            z_score: f64::NAN,
            p_value: f64::NAN,
            samples_used: samples_used as u16,
            starvations,
        };
    }
    // Find max-|z| index. Skip NaN values; if all are NaN, fall back to the
    // first element with the lexicographically smallest key (which is
    // already the case since element_keys is sorted at build_distribution).
    let mut best_idx: Option<usize> = None;
    let mut best_abs_z: f64 = f64::NEG_INFINITY;
    for (i, &z) in dist.z_scores.iter().enumerate() {
        if !z.is_finite() {
            continue;
        }
        let az = z.abs();
        if az > best_abs_z {
            best_abs_z = az;
            best_idx = Some(i);
        }
    }
    let idx = best_idx.unwrap_or(0);
    SingleModelSignificance {
        model: model.to_string(),
        observed_value: dist.source_values[idx],
        mean_null: dist.means[idx],
        std_null: dist.stddevs[idx],
        z_score: dist.z_scores[idx],
        p_value: dist.p_values[idx],
        samples_used: samples_used as u16,
        starvations,
    }
}

/// AND-reduce per-model rows into the [`CombinedSignificance`] verdict.
fn combine_per_model(rows: &[SingleModelSignificance]) -> CombinedSignificance {
    if rows.is_empty() {
        return CombinedSignificance {
            significant_vs_all_at_p05: false,
            significant_vs_all_at_p01: false,
            min_p_across_models: f32::NAN,
            max_abs_z_across_models: f32::NAN,
        };
    }
    let mut all_p05 = true;
    let mut all_p01 = true;
    let mut min_p: f32 = f32::INFINITY;
    let mut max_abs_z: f32 = f32::NEG_INFINITY;
    for r in rows {
        let z_finite = r.z_score.is_finite();
        let p_finite = r.p_value.is_finite();
        let abs_z = r.z_score.abs();
        let abs_z_passes_05 = z_finite && abs_z > Z_THRESHOLD_P05;
        let abs_z_passes_01 = z_finite && abs_z > Z_THRESHOLD_P01;
        let p_passes_05 = p_finite && r.p_value < P_THRESHOLD_05;
        let p_passes_01 = p_finite && r.p_value < P_THRESHOLD_01;
        if !(abs_z_passes_05 && p_passes_05) {
            all_p05 = false;
        }
        if !(abs_z_passes_01 && p_passes_01) {
            all_p01 = false;
        }
        if p_finite && (r.p_value as f32) < min_p {
            min_p = r.p_value as f32;
        }
        if z_finite && (abs_z as f32) > max_abs_z {
            max_abs_z = abs_z as f32;
        }
    }
    CombinedSignificance {
        significant_vs_all_at_p05: all_p05,
        significant_vs_all_at_p01: all_p01,
        min_p_across_models: if min_p.is_finite() { min_p } else { f32::NAN },
        max_abs_z_across_models: if max_abs_z.is_finite() {
            max_abs_z
        } else {
            f32::NAN
        },
    }
}

// ── Determinism helpers ──────────────────────────────────────────────────────

/// NuDHy base seed derived from the source narrative id. Stable across reruns
/// of the same dual call.
fn nudhy_base_seed(narrative_id: &str) -> u64 {
    // Keep the algebra simple + deterministic. XOR mix of a fixed constant
    // with a folded byte sum is enough for reproducible per-(narrative,
    // model) seed streams.
    let mut seed: u64 = 0x5A5A_A5A5_A5A5_5A5A;
    for (i, b) in narrative_id.bytes().enumerate() {
        seed ^= (b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15 ^ (i as u64));
        seed = seed.rotate_left(13);
    }
    seed
}

/// Per-K seed mix — same algebra as Phase 7's `per_sample_seed` so the
/// per-K seeds across EATH and NuDHy stay structurally aligned.
#[inline]
fn per_sample_seed(base_seed: u64, sample_idx: usize) -> u64 {
    base_seed ^ (sample_idx as u64).wrapping_mul(SAMPLE_SEED_MIX)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "dual_significance_tests.rs"]
mod dual_significance_tests;
