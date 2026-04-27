//! Phase 2.5 — fidelity report comparing a calibrated EATH surrogate against
//! the source narrative it was fitted to.
//!
//! Without a fidelity report, every p-value the Phase 7 significance machinery
//! emits is scientifically indefensible — a reviewer rejects on sight, and so
//! do internal consumers (disinfo team, EIC Pathfinder reviewers).
//!
//! ## Pipeline
//!
//! 1. Caller invokes [`run_fidelity_report`] with the calibrated `EathParams`,
//!    a reference `Hypergraph` (for the source-narrative truth stats), and a
//!    [`FidelityConfig`].
//! 2. We compute the source narrative's stats ONCE (group sizes, per-entity
//!    activity / propensity / inter-event times / autocorr / hyperdegree).
//! 3. We generate K samples (K ≥ 10, default 20) via [`super::eath::EathSurrogate`]
//!    into *ephemeral* `MemoryStore`-backed hypergraphs — synthetic samples
//!    NEVER pollute the user's persistent KV store.
//! 4. Per K-sample we extract the same stat shape, then average / aggregate
//!    across K to compare against source.
//! 5. We score each metric against its threshold + weight, emitting a
//!    [`FidelityReport`] with `passed` and `overall_score`.
//!
//! ## Determinism
//!
//! Same `(EathParams, K, base_seed)` ⇒ identical [`FidelityReport.metrics`].
//! Per-sample seed is `base_seed XOR (sample_idx as u64).wrapping_mul(SAMPLE_SEED_MIX)`,
//! so single- and multi-threaded runs both walk the same K seed sequence.
//!
//! ## Persistence
//!
//! [`run_fidelity_report`] is pure — it returns the report. The caller decides
//! whether to persist via [`save_fidelity_report`] (Phase 4 inference engine
//! will; many tests don't).
//!
//! Custom thresholds persist at `cfg/synth_fidelity/{narrative_id}` —
//! user-editable config sits in the existing `cfg/` namespace alongside
//! `cfg/llm` etc. (see [`SYNTH_FIDELITY_THRESHOLDS_PREFIX`]).
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! The weighted pass-fraction `aggregate_score` stays unchanged
//! (weighted arithmetic, not a t-norm). The opt-in
//! `aggregate_score_with_tnorm` in `fidelity_pipeline.rs` is available
//! for AND-style "every metric must pass" reduction under a chosen
//! t-norm family (Gödel / Goguen / Łukasiewicz / Hamacher). Default
//! thresholds and the existing `OVERALL_PASS_SCORE` gate remain
//! bit-identical — opt-in only.
//!
//! Cites: [klement2000].

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;

use super::fidelity_pipeline::{
    aggregate_score, collapse_k_samples, compute_metrics, extract_source_stats, run_k_samples,
};
use super::key_synth_fidelity;
use super::types::EathParams;

// ── Constants ────────────────────────────────────────────────────────────────

/// Minimum number of K samples. Five samples gives effectively zero KS-test
/// resolution; ten is the floor where divergence values are interpretable.
pub const MIN_K_SAMPLES: usize = 10;

/// Default K samples. Twenty is the design-doc target — embarrassingly
/// parallel so the K=20 cost is wall-clock-flat on multi-core.
pub const DEFAULT_K_SAMPLES: usize = 20;

/// Per-sample seed mixing constant. XORed with `sample_idx * MIX` to derive
/// the per-sample seed from `base_seed`; same algebra under any threading
/// strategy so multi-threaded runs produce IDENTICAL reports.
pub(super) const SAMPLE_SEED_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// Default num-steps per generated K sample. Each step samples one tick of
/// the EATH chain; 200 ticks reliably produces enough situations to compute
/// stable stats on small (≤ 100-entity) narratives.
pub(super) const DEFAULT_GEN_STEPS: usize = 200;

/// `overall_score` ≥ this threshold ⇒ `passed = true`.
pub const OVERALL_PASS_SCORE: f32 = 0.7;

/// Custom-thresholds KV prefix. Mirrors the `cfg/llm`, `cfg/rag` pattern
/// so user-editable surrogate config sits in one namespace.
pub const SYNTH_FIDELITY_THRESHOLDS_PREFIX: &[u8] = b"cfg/synth_fidelity/";

// ── Public types ────────────────────────────────────────────────────────────

/// Per-metric result. `passed` is computed at construction time so
/// downstream consumers don't re-derive it from `value` vs `threshold`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FidelityMetric {
    pub name: String,
    /// Statistic family — `"ks_divergence"`, `"spearman_rho"`, or `"mae"`.
    /// Drives how the renderer phrases the comparison.
    pub statistic: String,
    pub value: f32,
    pub threshold: f32,
    pub passed: bool,
}

/// How a metric was scored — KS / MAE pass when `value <= threshold` (lower
/// is better); Spearman passes when `value >= threshold` (higher is better).
pub(super) fn metric_passed(statistic: &str, value: f32, threshold: f32) -> bool {
    match statistic {
        "spearman_rho" => value >= threshold,
        _ => value <= threshold,
    }
}

/// Aggregated report. `Serialize + Deserialize + PartialEq` so the
/// persistence round-trip test can assert byte-equality after load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FidelityReport {
    pub run_id: Uuid,
    pub model: String,
    pub narrative_id: String,
    pub k_samples_used: usize,
    pub metrics: Vec<FidelityMetric>,
    pub overall_score: f32,
    pub passed: bool,
    pub thresholds_provenance: ThresholdsProvenance,
    /// Optional learned-measure identifier when the overall_score was
    /// reduced via a `Choquet(measure)` aggregator with provenance
    /// (Phase 2 of the Graded Acceptability sprint). Serde-defaulted so
    /// pre-Phase-2 records load unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fuzzy_measure_id: Option<String>,
    /// Version stamp paired with `fuzzy_measure_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fuzzy_measure_version: Option<u32>,
}

/// Where the thresholds came from. Default ⇒ render the ⚠ placeholder
/// warning in the markdown report. UserOverride / StudyCalibrated ⇒ no
/// warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdsProvenance {
    Default,
    UserOverride,
    StudyCalibrated,
}

/// Threading strategy for the K-sample loop. `Auto` picks `Threads(num_cpus)`
/// when the OS reports multiple cores, otherwise `Single`.
///
/// Determinism is preserved across all three modes — per-sample seeds are
/// derived from `(base_seed, sample_idx)` only, never from execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParallelismMode {
    #[default]
    Auto,
    Single,
    Threads(usize),
}

/// Configurable thresholds. Defaults are PLACEHOLDER values — the markdown
/// renderer surfaces this fact when `thresholds_provenance == Default`.
///
/// `weights` aggregates per-metric `passed` (1.0) / `failed` (0.0) into the
/// overall score: `Σ weight_i * passed_i / Σ weight_i`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FidelityThresholds {
    #[serde(default = "default_inter_event_ks")]
    pub inter_event_ks: f32,
    #[serde(default = "default_group_size_ks")]
    pub group_size_ks: f32,
    #[serde(default = "default_activity_spearman")]
    pub activity_spearman: f32,
    #[serde(default = "default_order_propensity_spearman")]
    pub order_propensity_spearman: f32,
    #[serde(default = "default_burstiness_mae")]
    pub burstiness_mae: f32,
    #[serde(default = "default_memory_autocorr_mae")]
    pub memory_autocorr_mae: f32,
    #[serde(default = "default_hyperdegree_ks")]
    pub hyperdegree_ks: f32,
    #[serde(default = "default_weights")]
    pub weights: Vec<(String, f32)>,
}

impl Default for FidelityThresholds {
    fn default() -> Self {
        Self {
            inter_event_ks: default_inter_event_ks(),
            group_size_ks: default_group_size_ks(),
            activity_spearman: default_activity_spearman(),
            order_propensity_spearman: default_order_propensity_spearman(),
            burstiness_mae: default_burstiness_mae(),
            memory_autocorr_mae: default_memory_autocorr_mae(),
            hyperdegree_ks: default_hyperdegree_ks(),
            weights: default_weights(),
        }
    }
}

// PLACEHOLDER values — see EATH_sprint.md Notes for the follow-up study.
pub fn default_inter_event_ks() -> f32 {
    0.10
}
pub fn default_group_size_ks() -> f32 {
    0.05
}
pub fn default_activity_spearman() -> f32 {
    0.7
}
pub fn default_order_propensity_spearman() -> f32 {
    0.6
}
pub fn default_burstiness_mae() -> f32 {
    0.15
}
pub fn default_memory_autocorr_mae() -> f32 {
    0.15
}
pub fn default_hyperdegree_ks() -> f32 {
    0.10
}
pub fn default_weights() -> Vec<(String, f32)> {
    vec![
        ("inter_event_time_distribution".into(), 1.0),
        ("group_size_distribution".into(), 1.0),
        ("activity_match".into(), 1.0),
        ("order_propensity_match".into(), 1.0),
        ("burstiness_parity".into(), 1.0),
        ("memory_autocorrelation".into(), 1.0),
        ("hyperdegree_distribution".into(), 1.0),
    ]
}

/// User-facing config bundle. Construct via `Default` for K=20 + Auto +
/// default thresholds; via [`FidelityConfig::new`] to override K (with
/// minimum-clamping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityConfig {
    pub k_samples: usize,
    pub thresholds: FidelityThresholds,
    pub parallelism: ParallelismMode,
}

impl Default for FidelityConfig {
    fn default() -> Self {
        Self {
            k_samples: DEFAULT_K_SAMPLES,
            thresholds: FidelityThresholds::default(),
            parallelism: ParallelismMode::Auto,
        }
    }
}

impl FidelityConfig {
    /// Construct with custom K. K below [`MIN_K_SAMPLES`] is CLAMPED with a
    /// `tracing::warn!` — the reduced statistical resolution is real, so
    /// silently dropping a too-small K would hide a config bug.
    pub fn new(
        k_samples: usize,
        thresholds: FidelityThresholds,
        parallelism: ParallelismMode,
    ) -> Self {
        let clamped = if k_samples < MIN_K_SAMPLES {
            tracing::warn!(
                "FidelityConfig: K={k_samples} requested below minimum {MIN_K_SAMPLES}; \
                 clamping to {MIN_K_SAMPLES}"
            );
            MIN_K_SAMPLES
        } else {
            k_samples
        };
        Self {
            k_samples: clamped,
            thresholds,
            parallelism,
        }
    }
}

// ── Public entry point ─────────────────────────────────────────────────────

/// Build a [`FidelityReport`] for the calibrated `params` against the
/// `narrative_id` source narrative.
///
/// `run_id` ties this report to the calibration run that produced `params`.
/// Use `Uuid::now_v7()` if you don't have one (the report is keyed by
/// `(narrative_id, run_id)` — Phase 4 inference engine will pass the
/// calibration job's run UUID here).
///
/// Determinism: same `(params, k_samples, base_seed)` ⇒ identical
/// `metrics`. Different threading modes CANNOT change the result.
pub fn run_fidelity_report(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &EathParams,
    config: &FidelityConfig,
    run_id: Uuid,
    base_seed: u64,
    thresholds_provenance: ThresholdsProvenance,
) -> Result<FidelityReport> {
    let source = extract_source_stats(hypergraph, narrative_id)?;
    let samples = run_k_samples(params, base_seed, config.k_samples, config.parallelism)?;
    let collapsed = collapse_k_samples(&samples);
    let metrics = compute_metrics(&source, &collapsed, &config.thresholds);
    let overall_score = aggregate_score(&metrics, &config.thresholds.weights);
    Ok(FidelityReport {
        run_id,
        model: "eath".into(),
        narrative_id: narrative_id.to_string(),
        k_samples_used: config.k_samples,
        metrics,
        overall_score,
        passed: overall_score >= OVERALL_PASS_SCORE,
        thresholds_provenance,
        fuzzy_measure_id: None,
        fuzzy_measure_version: None,
    })
}

// ── Persistence ────────────────────────────────────────────────────────────

/// Persist a fidelity report at `syn/fidelity/{narrative_id}/{run_id_BE}`.
/// Caller's choice — the report-builder ([`run_fidelity_report`]) is pure and
/// doesn't write KV; Phase 4 inference engine will, many tests won't.
pub fn save_fidelity_report(store: &dyn KVStore, report: &FidelityReport) -> Result<()> {
    let key = key_synth_fidelity(&report.narrative_id, &report.run_id);
    let value = serde_json::to_vec(report)?;
    store.put(&key, &value)
}

/// Inverse of [`save_fidelity_report`]. Returns `Ok(None)` when no report
/// exists for the `(narrative_id, run_id)` pair.
pub fn load_fidelity_report(
    store: &dyn KVStore,
    narrative_id: &str,
    run_id: &Uuid,
) -> Result<Option<FidelityReport>> {
    let key = key_synth_fidelity(narrative_id, run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Persist user-overridden thresholds at `cfg/synth_fidelity/{narrative_id}`.
/// Phase 6 REST routes call this from `PUT /synth/fidelity-thresholds/{nid}`.
pub fn save_thresholds(
    store: &dyn KVStore,
    narrative_id: &str,
    thresholds: &FidelityThresholds,
) -> Result<()> {
    let key = thresholds_key(narrative_id);
    let value = serde_json::to_vec(thresholds)?;
    store.put(&key, &value)
}

/// Load user-overridden thresholds. Returns `Default` when absent — never
/// `Option`, so consumer code can always proceed without unwrap branches.
pub fn load_thresholds(store: &dyn KVStore, narrative_id: &str) -> Result<FidelityThresholds> {
    let key = thresholds_key(narrative_id);
    match store.get(&key)? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Ok(FidelityThresholds::default()),
    }
}

fn thresholds_key(narrative_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(SYNTH_FIDELITY_THRESHOLDS_PREFIX.len() + narrative_id.len());
    k.extend_from_slice(SYNTH_FIDELITY_THRESHOLDS_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k
}

// ── Markdown renderer ──────────────────────────────────────────────────────

/// Render a `FidelityReport` as Markdown suitable for PR descriptions and
/// the future Studio UI.
///
/// When `thresholds_provenance == Default`, appends the load-bearing ⚠
/// placeholder warning. The exact string is asserted against in T5.
pub fn render_report_as_markdown(report: &FidelityReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Fidelity Report — {}\n\n",
        report.narrative_id
    ));
    out.push_str(&format!("- Run ID: `{}`\n", report.run_id));
    out.push_str(&format!("- Model: `{}`\n", report.model));
    out.push_str(&format!("- K samples: {}\n", report.k_samples_used));
    out.push_str(&format!(
        "- Overall score: **{:.3}** ({})\n\n",
        report.overall_score,
        if report.passed { "PASS" } else { "FAIL" }
    ));

    out.push_str("| Metric | Statistic | Value | Threshold | Passed |\n");
    out.push_str("|---|---|---|---|---|\n");
    for m in &report.metrics {
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {} |\n",
            m.name,
            m.statistic,
            m.value,
            m.threshold,
            if m.passed { "PASS" } else { "FAIL" }
        ));
    }
    out.push('\n');

    if matches!(report.thresholds_provenance, ThresholdsProvenance::Default) {
        out.push_str("> ⚠ Thresholds: default placeholder values — empirical study pending.\n");
        out.push_str("> See EATH_sprint.md Notes for follow-up sprint details.\n");
    }

    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "fidelity_tests.rs"]
mod fidelity_tests;
