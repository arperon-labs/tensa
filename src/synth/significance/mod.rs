//! Phase 7 — Pattern-significance integration via EATH null models.
//!
//! Treats EATH-as-calibrated-null as the reference distribution, generates K
//! ephemeral synthetic narratives, computes the same metric on each, and
//! reports per-element z-scores + one-tailed empirical p-values vs the source
//! narrative. Three metrics ship in this phase:
//!
//! * `temporal_motifs` — per-motif-type counts via
//!   [`crate::analysis::temporal_motifs::temporal_motif_census`].
//! * `communities` — `(num_communities, modularity)` via
//!   [`crate::analysis::graph_projection::build_co_graph`] +
//!   [`crate::analysis::community_detect::label_propagation`].
//! * `patterns` — per-pattern presence/absence via
//!   [`crate::narrative::pattern::mine_patterns_with_config`].
//!
//! See `docs/synth_null_model.md` for the full design contract; this module
//! implements §7 (types), §3 (KV layout), §6 (edge cases), §9 (adapters).
//!
//! ## File layout
//!
//! Submodule split per design doc §10:
//! * `mod.rs` (this file) — types, persistence, [`SignificanceEngine`].
//! * `adapters.rs` — three [`MetricAdapter`] impls + the K-loop helper.
//! * `../significance_tests.rs` — six tests per design doc §8.
//!
//! ## Determinism contract
//!
//! Per-sample seed is `base_seed XOR (k_idx * SAMPLE_SEED_MIX)` —
//! identical algebra to `fidelity.rs`. This means the K-loop produces
//! identical synthetic narratives across single- and multi-threaded runs,
//! and the first 20 samples of any significance run can in principle
//! be reused from a fidelity run with the same `(params, base_seed)`.

pub mod adapters;
pub(crate) mod stats;

use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::KVStore;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

use super::calibrate::{calibrate_with_fidelity_report, load_params, save_params};
use super::emit::{is_synthetic_entity, is_synthetic_situation};
use super::fidelity::{FidelityConfig, FidelityReport, ParallelismMode};
use super::registry::SurrogateRegistry;
use super::types::EathParams;
use super::{key_synth_sig, narrative_scan_prefix, SYNTH_SIG_PREFIX};

use adapters::{run_significance_pipeline, AdapterChoice};

// ── Constants (named so /simplify doesn't flag them as magic) ────────────────

/// Default K samples when the request omits `k`.
pub const K_DEFAULT: u16 = 100;

/// Hard cap on K. Requests with `k > K_MAX` are silently clamped — the engine
/// reports the effective K in `SignificanceReport.k_samples_used`.
pub const K_MAX: u16 = 1000;

// ── Public types ────────────────────────────────────────────────────────────

/// Canonical string identifier for a significance metric.
///
/// The `as_kv_str` form is part of the KV key encoding at
/// `syn/sig/{narrative_id}/{metric_kv_str}/{run_id_BE}` — DO NOT CHANGE
/// without a migration.
///
/// `Contagion` (Phase 7b) is the only variant that needs a per-run params
/// blob — `adapter()` requires the caller to thread through the
/// `HigherOrderSirParams`. The other three are stateless adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignificanceMetric {
    TemporalMotifs,
    Communities,
    Patterns,
    Contagion,
}

impl SignificanceMetric {
    /// Parse from the string values used in REST bodies and KV keys.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "temporal_motifs" => Some(Self::TemporalMotifs),
            "communities" => Some(Self::Communities),
            "patterns" => Some(Self::Patterns),
            "contagion" => Some(Self::Contagion),
            _ => None,
        }
    }

    /// Canonical KV string. Single source of truth — used in keys and the
    /// REST URL path.
    pub fn as_kv_str(&self) -> &'static str {
        match self {
            Self::TemporalMotifs => "temporal_motifs",
            Self::Communities => "communities",
            Self::Patterns => "patterns",
            Self::Contagion => "contagion",
        }
    }

    /// Internal mapping into the adapter type the engine dispatches on.
    /// `Contagion` is **not** included here because it requires a params
    /// blob — callers route to the contagion engine which constructs an
    /// `AdapterChoice::Contagion(params)` directly.
    fn adapter(&self) -> Option<AdapterChoice<'static>> {
        match self {
            Self::TemporalMotifs => Some(AdapterChoice::TemporalMotifs),
            Self::Communities => Some(AdapterChoice::Communities),
            Self::Patterns => Some(AdapterChoice::Patterns),
            Self::Contagion => None,
        }
    }
}

/// Per-element distribution statistics. Parallel `Vec`s indexed together —
/// `element_keys[i]` corresponds to `source_values[i]`, `means[i]`, etc.
///
/// `f64::NAN` serializes as JSON `null`; downstream consumers should treat
/// `null` in `z_scores` / `p_values` as "not computable" (e.g. zero stddev
/// or zero source observation with zero synthetic mass).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticDistribution {
    /// Human-readable label per scalar element.
    pub element_keys: Vec<String>,
    /// Metric value on the REAL source narrative, per element.
    pub source_values: Vec<f64>,
    /// Mean of the metric across K synthetic samples, per element.
    pub means: Vec<f64>,
    /// Population standard deviation (denominator K), per element.
    pub stddevs: Vec<f64>,
    /// z-score per element. `NaN` when stddev == 0.
    pub z_scores: Vec<f64>,
    /// Empirical one-tailed p-value per element: `count(m_k[i] >= M_real[i]) / K`.
    /// `NaN` when both M_real and all m_k are 0.
    pub p_values: Vec<f64>,
    /// Test direction documented for interpretability — currently always
    /// `"more_is_significant"` for the three Phase 7 metrics.
    pub direction: String,
}

/// Complete output of one significance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceReport {
    /// Unique run identifier (UUIDv7). Used in the KV key suffix.
    pub run_id: Uuid,
    /// Source narrative being tested.
    pub narrative_id: String,
    /// Surrogate model (typically `"eath"`).
    pub model: String,
    /// Which metric was tested.
    pub metric: SignificanceMetric,
    /// Effective K (may be < requested if capped).
    pub k_samples_used: usize,
    /// Raw metric output on the real narrative as a JSON blob for archival.
    pub source_observation: serde_json::Value,
    /// Full per-element distribution statistics.
    pub synthetic_distribution: SyntheticDistribution,
    /// True when EathParams were not found at `syn/p/{nid}/{model}` and were
    /// auto-calibrated inline. When true, `calibration_fidelity` is populated.
    pub auto_calibrated: bool,
    /// Fidelity report from auto-calibration. None when `auto_calibrated == false`.
    #[serde(default)]
    pub calibration_fidelity: Option<FidelityReport>,
    /// Optional free-text note (e.g. "metric returned no observations").
    #[serde(default)]
    pub note: Option<String>,
    /// Wall-clock timestamps.
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
}

// ── Persistence helpers ──────────────────────────────────────────────────────

/// Persist a [`SignificanceReport`] at
/// `syn/sig/{narrative_id}/{metric}/{run_id_BE}`.
pub fn save_significance_report(
    store: &dyn KVStore,
    report: &SignificanceReport,
) -> Result<()> {
    let key = key_synth_sig(
        &report.narrative_id,
        report.metric.as_kv_str(),
        &report.run_id,
    );
    let value = serde_json::to_vec(report)?;
    store.put(&key, &value)
}

/// Inverse of [`save_significance_report`]. `Ok(None)` when no report exists.
pub fn load_significance_report(
    store: &dyn KVStore,
    narrative_id: &str,
    metric: SignificanceMetric,
    run_id: &Uuid,
) -> Result<Option<SignificanceReport>> {
    let key = key_synth_sig(narrative_id, metric.as_kv_str(), run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List up to `limit` reports for `(narrative_id, metric)` newest first.
/// Reverses the chronological prefix scan in O(n) on the page window.
pub fn list_significance_reports(
    store: &dyn KVStore,
    narrative_id: &str,
    metric: SignificanceMetric,
    limit: usize,
) -> Result<Vec<SignificanceReport>> {
    // Build the `syn/sig/{narrative_id}/{metric}/` prefix.
    let mut prefix = narrative_scan_prefix(SYNTH_SIG_PREFIX, narrative_id);
    prefix.extend_from_slice(metric.as_kv_str().as_bytes());
    prefix.push(b'/');
    let mut pairs = store.prefix_scan(&prefix)?;
    pairs.reverse();
    pairs.truncate(limit);
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        match serde_json::from_slice::<SignificanceReport>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!("skipping malformed SignificanceReport: {e}"),
        }
    }
    Ok(out)
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// `InferenceEngine` impl for [`InferenceJobType::SurrogateSignificance`].
///
/// Holds the same `Arc<SurrogateRegistry>` Phase 4 calibration / generation
/// engines use; the registry is read-only after construction so multiple
/// workers can hold the same `Arc` without contention.
pub struct SurrogateSignificanceEngine {
    pub registry: Arc<SurrogateRegistry>,
}

impl SurrogateSignificanceEngine {
    /// Construct with a shared registry.
    pub fn new(registry: Arc<SurrogateRegistry>) -> Self {
        Self { registry }
    }
}

impl InferenceEngine for SurrogateSignificanceEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — `WorkerPool::engines` keys by std::mem::Discriminant,
        // not full equality. Empty string fields are deliberate: any future
        // reader of this can immediately tell it's a registration sentinel.
        InferenceJobType::SurrogateSignificance {
            narrative_id: String::new(),
            metric_kind: String::new(),
            k: 0,
            model: String::new(),
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        // Defer to the central `cost::estimate_cost` table — same pattern as
        // the calibration / generation engines.
        Ok(job.estimated_cost_ms)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let started = Instant::now();
        let started_at = Utc::now();

        // 1. Deconstruct variant payload.
        let (narrative_id, metric_kind, k_requested, model_name) = match &job.job_type {
            InferenceJobType::SurrogateSignificance {
                narrative_id,
                metric_kind,
                k,
                model,
            } => (narrative_id.clone(), metric_kind.clone(), *k, model.clone()),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "SurrogateSignificanceEngine received wrong job type: {other:?}"
                )));
            }
        };

        if narrative_id.is_empty() {
            return Err(TensaError::InvalidInput(
                "SurrogateSignificance: narrative_id is empty".into(),
            ));
        }
        let metric = SignificanceMetric::parse(&metric_kind).ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "SurrogateSignificance: unknown metric '{metric_kind}'; \
                 expected one of: temporal_motifs, communities, patterns. \
                 (For 'contagion', submit a SurrogateContagionSignificance job instead.)"
            ))
        })?;
        // Contagion has its own engine — see SurrogateContagionSignificanceEngine.
        let adapter = metric.adapter().ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "SurrogateSignificance: metric '{metric_kind}' is not handled by this engine; \
                 use SurrogateContagionSignificance instead"
            ))
        })?;
        let model = if model_name.is_empty() {
            "eath".to_string()
        } else {
            model_name
        };

        // 2. Validate model is registered (early failure beats wasted work).
        // Phase 13c lift: previously this guard rejected anything non-"eath";
        // the underlying K-loop is model-agnostic via the trait, so the only
        // requirement is registry membership. NudhySurrogate is now a valid
        // null model here. `registry.get()` already returns SynthFailure with
        // a "did you mean X?"-style listing on miss, so we don't need a second
        // explicit check.
        let _model_handle = self.registry.get(&model)?;

        // 3. Refuse fully-synthetic narratives — comparing synthetic vs
        //    synthetic produces z-scores without ground-truth meaning.
        guard_not_fully_synthetic(hypergraph, &narrative_id)?;

        // 4. Cap k.
        let k_clamped = k_requested.clamp(1, K_MAX);
        if k_requested > K_MAX {
            tracing::warn!(
                "SurrogateSignificance: k={k_requested} exceeds K_MAX={K_MAX}; clamping"
            );
        }
        let k_effective = k_clamped as usize;

        // 5. Resolve EathParams. params_override path is only available via
        //    job.parameters since the InferenceJobType variant doesn't carry
        //    one — POST /synth/significance threads it through there.
        let (eath_params, auto_calibrated, calibration_fidelity) = resolve_params(
            hypergraph,
            &narrative_id,
            &model,
            &job.parameters,
        )?;

        // 6. Read optional metric_params (e.g. {"max_motif_size": 4}).
        let metric_params = job
            .parameters
            .get("metric_params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        // 7. Run the K-loop pipeline (computes source observation ONCE, then
        //    K samples in parallel via std::thread::scope).
        let pipeline = run_significance_pipeline(
            hypergraph,
            &narrative_id,
            &eath_params,
            adapter,
            &metric_params,
            k_effective,
            ParallelismMode::Auto,
        )?;

        // 8. Build the report.
        let finished_at = Utc::now();
        let duration_ms = started.elapsed().as_millis() as u64;
        let note = if pipeline.source_was_empty {
            Some(
                "metric returned no observations on the source narrative; \
                 z-scores are not meaningful"
                    .to_string(),
            )
        } else {
            None
        };

        let report = SignificanceReport {
            run_id: Uuid::now_v7(),
            narrative_id: narrative_id.clone(),
            model: model.clone(),
            metric,
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

        // 9. Persist and emit the result envelope.
        save_significance_report(hypergraph.store(), &report)?;

        let result_json = serde_json::json!({
            "kind": "significance_done",
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
                "significance ({}) over {k_effective} synthetic samples",
                report.metric.as_kv_str()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(finished_at),
        })
    }
}

// ── Engine helpers ───────────────────────────────────────────────────────────

/// Refuse the job iff every entity AND every situation in the narrative
/// passes the synthetic predicate. Mixed narratives are permitted (Phase 3
/// invariant test relies on this). Empty narratives fall through to the
/// adapter (which produces a graceful zero-stats blob and a `note`).
///
/// Visible to sibling engines (Phase 7b's contagion variant reuses it).
pub(crate) fn guard_not_fully_synthetic(hypergraph: &Hypergraph, narrative_id: &str) -> Result<()> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    if entities.is_empty() {
        return Ok(());
    }
    // Authoritative full-pass check — mixed real+synthetic narratives are
    // permitted (Phase 3 invariant tests rely on this), so we need a real
    // answer, not a sample-based one. Cost is O(N) on a metadata-only walk
    // that's already in cache from the calibrate path.
    if !entities.iter().all(is_synthetic_entity) {
        return Ok(());
    }
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        // All entities synthetic but no situations — still a synthetic narrative.
        return Err(TensaError::SynthFailure(format!(
            "cannot run significance on a synthetic narrative '{narrative_id}'; \
             metric would compare synthetic vs synthetic, producing z-scores \
             without meaningful ground truth"
        )));
    }
    let all_s_synth = situations.iter().all(is_synthetic_situation);
    if all_s_synth {
        return Err(TensaError::SynthFailure(format!(
            "cannot run significance on a synthetic narrative '{narrative_id}'; \
             metric would compare synthetic vs synthetic, producing z-scores \
             without meaningful ground truth"
        )));
    }
    Ok(())
}

/// Resolve EathParams from (in order): `parameters.params_override` JSON,
/// then `load_params(store, narrative_id, model)`, then auto-calibrate inline
/// (which doubles wall-clock).
///
/// Visible to sibling engines (Phase 7b's contagion variant reuses it).
pub(crate) fn resolve_params(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    model: &str,
    parameters: &serde_json::Value,
) -> Result<(EathParams, bool, Option<FidelityReport>)> {
    if let Some(raw) = parameters.get("params_override") {
        if !raw.is_null() {
            let params: EathParams = serde_json::from_value(raw.clone()).map_err(|e| {
                TensaError::InvalidInput(format!(
                    "SurrogateSignificance: params_override is not a valid EathParams blob: {e}"
                ))
            })?;
            return Ok((params, false, None));
        }
    }

    if let Some(p) = load_params(hypergraph.store(), narrative_id, model)? {
        return Ok((p, false, None));
    }

    tracing::info!(
        "SurrogateSignificance: no params found for ({narrative_id}, {model}); \
         auto-calibrating (adds ~2s for 1k-entity narratives)"
    );
    let (params, report) =
        calibrate_with_fidelity_report(hypergraph, narrative_id, &FidelityConfig::default())?;
    save_params(hypergraph.store(), narrative_id, model, &params)?;
    Ok((params, true, Some(report)))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../significance_tests.rs"]
mod significance_tests;
