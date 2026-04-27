//! Shared types for surrogate-model calibration, generation, and reproducibility.
//!
//! `SurrogateParams` is the model-agnostic envelope; the inner `params_json`
//! carries model-specific fitted parameters (`EathParams` for the first impl).
//! Future models — HAD, hyperedge configuration, narrative-conditioned
//! diffusion, etc. — get their own concrete params types in this file and
//! serialize through the same envelope.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Model-agnostic envelope for everything a [`SurrogateModel::generate`] call
/// needs. The model-specific fitted parameters live in `params_json`; the
/// model itself knows the schema (e.g. `EathParams` for `model == "eath"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateParams {
    /// Surrogate-model name. Stable identifier — same string the registry
    /// uses for lookup and the TensaQL grammar carries as a string param.
    pub model: String,
    /// Model-specific fitted params, JSON-shaped so new models can plug in
    /// without changing this envelope.
    pub params_json: serde_json::Value,
    /// Deterministic-replay seed.
    pub seed: u64,
    /// Number of steps the generator runs (semantics are model-specific —
    /// EATH treats each step as one "tick" of activity sampling).
    pub num_steps: usize,
    /// Prefix for synthetic-narrative ids (output_narrative_id =
    /// `format!("{label_prefix}-{run_id}")` is one common pattern).
    pub label_prefix: String,
}

/// EATH-specific fitted parameters (Mancastroppa, Cencetti, Barrat 2025).
///
/// Stored inside `SurrogateParams.params_json` after JSON conversion.
/// Phases 1–2 populate every field; the empty-vector defaults are stand-ins
/// so this struct deserializes cleanly from a Phase 0 stub.
///
/// Phase-1 extensions (`rho_low`, `rho_high`, `xi`, `order_propensity`,
/// `max_group_size`, `stm_capacity`, `num_entities`) are all
/// `#[serde(default)]` so older Phase-0 blobs round-trip cleanly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EathParams {
    /// Activation probability per actor, indexed by actor index. Empirical
    /// distribution used to sample which actors fire at each step.
    #[serde(default)]
    pub a_t_distribution: Vec<f32>,
    /// Per-actor "hyperactivity" — propensity to spawn larger groups.
    #[serde(default)]
    pub a_h_distribution: Vec<f32>,
    /// Λ_t schedule: time-varying global activity rate (one value per step).
    #[serde(default)]
    pub lambda_schedule: Vec<f32>,
    /// Probability that a new active group forms from scratch instead of
    /// re-activating a memorized one (Mancastroppa-Cencetti-Barrat eq. 4).
    #[serde(default)]
    pub p_from_scratch: f32,
    /// Memory decay coefficient ω for the "active history" weighting.
    #[serde(default)]
    pub omega_decay: f32,
    /// Empirical group-size distribution: `group_size_distribution[k] =
    /// count(groups of size k+2)`. Index 0 is dyads, 1 is triads, etc.
    #[serde(default)]
    pub group_size_distribution: Vec<u32>,

    // ── Phase-1 extensions (paper §III) ──────────────────────────────────────
    /// Low→High transition rate scale. r_{l→h}(i,t) =
    /// `lambda_t * rho_low * (a_T(i) / mean_a_T)`. Default 0.1
    /// (Mancastroppa § III.1 empirical sweet spot).
    #[serde(default = "default_rho_low")]
    pub rho_low: f32,
    /// High→Low transition rate scale. r_{h→l}(i,t) =
    /// `(1 - lambda_t) * rho_high`. Default 0.5 — entities spend ~2× longer
    /// in the low phase than the high phase.
    #[serde(default = "default_rho_high")]
    pub rho_high: f32,
    /// Scale factor ξ: expected groups per step ≈ `xi * (A_t / mean_A_t)`.
    /// Default 1.0 — one group per "average-activity" step in expectation.
    #[serde(default = "default_xi")]
    pub xi: f32,
    /// Per-entity order propensity φ_i(m). Flat row-major
    /// `Vec<f32>` of length `num_entities * (max_group_size - 1)`.
    /// Empty default ⇒ uniform, drawn from `group_size_distribution`.
    #[serde(default)]
    pub order_propensity: Vec<f32>,
    /// Maximum group size. Bounds STM mutation and order-propensity lookup.
    /// Default 10.
    #[serde(default = "default_max_group_size")]
    pub max_group_size: usize,
    /// Number of groups retained in short-term memory.
    /// Default 7 (midpoint of paper's 5–10 range).
    #[serde(default = "default_stm_capacity")]
    pub stm_capacity: usize,
    /// Number of entities to generate. Used as cross-check / override when
    /// `a_t_distribution` is empty (rare). Defaults to 0; the loader treats
    /// `a_t_distribution.len()` as authoritative when populated.
    #[serde(default)]
    pub num_entities: usize,
}

// ── Default-value functions for `#[serde(default = "…")]` attributes ──────────
//
// Serde requires named fns (not closures) for non-`Default::default` defaults.
// Keep these private and adjacent to `EathParams` so the contract is local.

fn default_rho_low() -> f32 {
    0.1
}

fn default_rho_high() -> f32 {
    0.5
}

fn default_xi() -> f32 {
    1.0
}

fn default_max_group_size() -> usize {
    10
}

fn default_stm_capacity() -> usize {
    7
}

/// Tag for what a run was for — pure calibration scratch, real generation,
/// or a combined calibrate-then-generate pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunKind {
    Calibration,
    Generation,
    Hybrid,
}

/// One row in the runs ledger. Persisted at `syn/r/{nid}/{run_id_BE}` so a
/// `prefix_scan` lists every run for a narrative in chronological order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateRunSummary {
    pub run_id: Uuid,
    pub model: String,
    /// Hex digest of the canonical-JSON serialization of the calibrated
    /// `SurrogateParams` (see [`super::hashing::canonical_params_hash`]).
    pub params_hash: String,
    /// The narrative we calibrated against — `None` for pure generation
    /// from hand-tuned params.
    pub source_narrative_id: Option<String>,
    /// Hex digest of the source-narrative state at calibration time
    /// (see [`super::hashing::canonical_narrative_state_hash`]). `None` when
    /// `source_narrative_id` is also `None`.
    pub source_state_hash: Option<String>,
    /// The synthetic narrative the run wrote into.
    pub output_narrative_id: String,
    pub num_entities: usize,
    pub num_situations: usize,
    pub num_participations: usize,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub kind: RunKind,
}

/// Everything needed to replay a run bit-for-bit. Persisted at
/// `syn/seed/{run_id_BE}`. Not embedded in `SurrogateRunSummary` to keep the
/// per-narrative run list cheap to scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityBlob {
    pub run_id: Uuid,
    pub model: String,
    pub params_full: SurrogateParams,
    pub seed: u64,
    /// Best-effort git SHA of the TENSA build that produced the run. Sourced
    /// from `option_env!("TENSA_GIT_HASH")` so dev builds without it set
    /// don't fail.
    pub git_sha: Option<String>,
    pub tensa_version: String,
    pub captured_at: DateTime<Utc>,
    /// Hash of the source narrative's state at run time. The Phase 11 Studio
    /// "Reproduce this run" action compares this to a freshly-computed state
    /// hash and warns when the source has drifted since the run.
    pub source_state_hash: Option<String>,
}

// ── Dual-null-model significance (Phase 13c) ────────────────────────────────

/// Per-model row inside a [`DualSignificanceReport`]. One entry per requested
/// null model.
///
/// The four scalars (`mean_null`, `std_null`, `z_score`, `p_value`) are the
/// AGGREGATE across the metric's element keys: max-|z| / matching-p, so a
/// single number per model can be compared without dragging the full
/// [`SyntheticDistribution`] into the dual report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingleModelSignificance {
    /// Surrogate model name (matches `SurrogateRegistry::get(&name)`).
    pub model: String,
    /// Source-side observed value — picked from the element with maximum |z|
    /// (single-source-of-truth across the dual reports).
    pub observed_value: f64,
    /// Mean of the null distribution at that same element.
    pub mean_null: f64,
    /// Population stddev of the null distribution at that element.
    pub std_null: f64,
    /// z-score of the observed value vs the null. Aggregated as max-|z| across
    /// metric elements (NaN-safe — finite values win over NaN).
    pub z_score: f64,
    /// Empirical one-tailed p-value at the same element. NaN when both source
    /// and all synth are zero (per stats::build_distribution semantics).
    pub p_value: f64,
    /// Effective K for this model — may be < requested when a model starves
    /// (NuDHy MCMC chains can fail; the K-loop tallies starvations).
    pub samples_used: u16,
    /// Number of K-chain failures for this model. EATH never starves
    /// (always 0); NuDHy can starve when source narrative violates MCMC
    /// preconditions (every chain edge identical, etc.). Field exists so the
    /// caller can tell "0/50 succeeded" from "50/50 succeeded".
    pub starvations: u16,
}

/// AND-reduced verdict across [`SingleModelSignificance`] entries.
///
/// "Significant vs ALL models" means the source narrative beats EVERY
/// requested null at the given threshold — the strong claim Phase 13c is
/// designed to surface. Min-p / max-|z| are exposed for callers that want a
/// single scalar per dual run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CombinedSignificance {
    /// True iff every per-model `|z| > 1.96` AND every per-model `p < 0.05`.
    pub significant_vs_all_at_p05: bool,
    /// True iff every per-model `|z| > 2.58` AND every per-model `p < 0.01`.
    pub significant_vs_all_at_p01: bool,
    /// Smallest p-value across the per-model rows. NaN when every per-model
    /// p is NaN.
    pub min_p_across_models: f32,
    /// Largest |z|-score across the per-model rows. NaN when every per-model
    /// z is NaN.
    pub max_abs_z_across_models: f32,
}

/// Output of one dual-null-model significance run. Persisted at
/// `syn/dual_sig/{narrative_id}/{metric}/{run_id_v7_BE_BIN_16}`.
///
/// `metric` is a string for grammar/registry agnosticism — the engine maps it
/// via [`crate::synth::significance::SignificanceMetric::parse`] but the
/// archived blob carries the original casing the caller passed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DualSignificanceReport {
    pub run_id: Uuid,
    pub narrative_id: String,
    pub metric: String,
    pub k_per_model: u16,
    pub per_model: Vec<SingleModelSignificance>,
    pub combined: CombinedSignificance,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
}

// ── Phase 14 — Bistability significance types ────────────────────────────────

/// Per-model row in a [`BistabilitySignificanceReport`]. Mirrors
/// [`SingleModelSignificance`] structurally but exposes the two scalars the
/// bistability sweep produces (interval width, max gap) instead of the
/// generic-metric (mean/std/z) form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingleModelBistabilityNull {
    /// Surrogate model name (matches `SurrogateRegistry::get(&name)`).
    pub model: String,
    /// Mean of `bistable_interval` width across K samples.
    pub mean_bistable_interval_width: f32,
    /// Population stddev of `bistable_interval` width across K samples.
    pub std_bistable_interval_width: f32,
    /// Empirical quantile of the source's bistable_interval_width vs the K
    /// null samples: `count(null_width < source_width) / K`. Closer to 1.0
    /// means the source is wider than most surrogates.
    pub bistable_interval_width_quantile: f32,
    /// Mean of `max_hysteresis_gap` across K samples.
    pub mean_max_hysteresis_gap: f32,
    /// Population stddev of `max_hysteresis_gap` across K samples.
    pub std_max_hysteresis_gap: f32,
    /// Empirical quantile of the source's gap vs the K null samples.
    pub max_hysteresis_gap_quantile: f32,
    /// Effective K for this model — may be < requested when a model
    /// starves (NuDHy MCMC chains can fail).
    pub samples_used: u16,
    /// Number of K-chain failures.
    pub starvations: u16,
}

/// AND-reduced verdict across [`SingleModelBistabilityNull`] entries.
///
/// The headline claim Phase 14 is designed to surface: "real narrative has a
/// wider bistable interval than 95% of EATH+NuDHy surrogates." The
/// `..._wider_than_all_at_p05` flag is true iff EVERY per-model
/// `bistable_interval_width_quantile > 0.95`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BistabilitySignificance {
    /// True iff every per-model `bistable_interval_width_quantile > 0.95`.
    pub source_bistable_wider_than_all_at_p05: bool,
    /// True iff every per-model `bistable_interval_width_quantile > 0.99`.
    pub source_bistable_wider_than_all_at_p01: bool,
    /// Smallest interval-width quantile across the per-model rows — 0.0
    /// when no models were sampled.
    pub min_quantile_across_models: f32,
    /// Largest |z|-equivalent score: `(source_width - mean) / std` per
    /// model, max across models. NaN when every model has zero std.
    pub max_z_across_models: f32,
}

/// Output of one bistability-significance run. Persisted at
/// `syn/bistability/{narrative_id}/{run_id_v7_BE_BIN_16}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BistabilitySignificanceReport {
    pub run_id: Uuid,
    pub narrative_id: String,
    /// Echo of the params the engine received — JSON to avoid pulling
    /// `BistabilitySweepParams` into this layer (lives in `analysis/`).
    pub params: serde_json::Value,
    pub k: u16,
    pub models: Vec<String>,
    /// The bistability run on the REAL narrative, serialized as JSON for
    /// the same layering reason (`BistabilityReport` lives above this
    /// module's layer constraint).
    pub source_observation: serde_json::Value,
    pub per_model: Vec<SingleModelBistabilityNull>,
    pub combined: BistabilitySignificance,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
}

// ── Phase 16c — Opinion-dynamics significance types ─────────────────────────

/// Per-model row in an [`OpinionSignificanceReport`].
///
/// Each entry compares the source narrative's three opinion-dynamics scalars
/// (num_clusters, polarization_index, echo_chamber_index) against K
/// surrogate samples from one null model. Z-scores follow the standard
/// `(observed - mean) / std` convention; quantiles are
/// `count(null < observed) / K` (so closer-to-1 means the source is more
/// extreme than most surrogates).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingleModelOpinionNull {
    pub model: String,
    pub mean_num_clusters: f32,
    pub std_num_clusters: f32,
    pub num_clusters_quantile: f32,
    pub mean_polarization_index: f32,
    pub std_polarization_index: f32,
    pub polarization_index_quantile: f32,
    /// `Some` only when every surrogate run produced a usable echo-chamber
    /// label-propagation result. `None` when label_propagation was missing
    /// for any surrogate (graceful degradation).
    pub mean_echo_chamber_index: Option<f32>,
    pub std_echo_chamber_index: Option<f32>,
    pub echo_chamber_index_quantile: Option<f32>,
    pub samples_used: u16,
}

/// Output of one opinion-dynamics-significance run. Persisted at
/// `syn/opinion_sig/{narrative_id}/{run_id_v7_BE_BIN_16}`.
///
/// Mirrors [`BistabilitySignificanceReport`] structurally so the existing
/// dual-null reading patterns translate one-for-one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpinionSignificanceReport {
    pub run_id: Uuid,
    pub narrative_id: String,
    /// Echo of the params the engine received — JSON to avoid pulling
    /// `OpinionDynamicsParams` into this layer (lives in `analysis/`).
    pub params: serde_json::Value,
    pub k: u16,
    pub models: Vec<String>,
    /// The opinion-dynamics run on the REAL narrative, serialized as JSON
    /// for the same layering reason (`OpinionDynamicsReport` lives above
    /// this module's layer constraint).
    pub source_observation: serde_json::Value,
    pub per_model: Vec<SingleModelOpinionNull>,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
}
