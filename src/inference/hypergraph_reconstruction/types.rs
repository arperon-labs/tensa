//! Data structures for SINDy hypergraph reconstruction.
//!
//! All types serialize through `serde_json` (the TENSA KV-storage default).
//! See `mod.rs` for module-level documentation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Enumerations ─────────────────────────────────────────────────────────────

/// Per-entity time-series source for the reconstruction.
///
/// Ranked by MVP readiness:
/// - [`ObservationSource::ParticipationRate`] — universal, fully implemented.
/// - [`ObservationSource::SentimentMean`] — requires `Situation.properties["sentiment"]`.
/// - [`ObservationSource::BeliefMass`] — requires populated `an/ev/` keys.
/// - [`ObservationSource::Engagement`] — multi-dimensional, deferred to Phase 15c.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObservationSource {
    /// Normalized participation count in a sliding window. One scalar per
    /// entity per time step. **MVP primary source.**
    ParticipationRate,
    /// Mean sentiment of situations the entity participated in (within the
    /// window). MVP returns `InferenceError` if sentiment data is missing.
    SentimentMean,
    /// Dempster-Shafer belief mass on a named proposition, projected to a
    /// scalar in [0, 1]. MVP returns `InferenceError` if the evidence engine
    /// has not been run.
    BeliefMass { proposition: String },
    /// Multi-dimensional engagement vector ([PR, SM, likes_rate, share_rate]).
    /// Declared per Phase 0 exhaustive-match convention; MVP stubs with
    /// `InferenceError` (full implementation is Phase 15c).
    Engagement,
}

/// Numerical derivative estimator for `dx_i/dt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DerivativeEstimator {
    /// Central finite differences. Trims first and last row (T' = T - 2).
    FiniteDiff,
    /// Savitzky-Golay polynomial smoother + differentiator.
    /// `window` must be odd, `>= 3`. `order` is polynomial degree.
    /// Default: `window=5, order=2`. Trims `(window - 1) / 2` rows from each
    /// end (T' = T - window + 1).
    SavitzkyGolay { window: usize, order: usize },
}

impl Default for DerivativeEstimator {
    fn default() -> Self {
        DerivativeEstimator::SavitzkyGolay {
            window: 5,
            order: 2,
        }
    }
}

/// A single monomial interaction term in the SINDy library.
///
/// Used as an index key for the coefficient matrix and bootstrap retention map.
/// Indices reference entity positions in the (sorted) entity vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LibraryTerm {
    /// `x_j` — pairwise interaction candidate.
    Order1(usize),
    /// `x_j * x_k` (j < k) — triadic hyperedge candidate.
    Order2(usize, usize),
    /// `x_j * x_k * x_l` (j < k < l) — 4-body candidate.
    Order3(usize, usize, usize),
    /// `x_j * x_k * x_l * x_m` (j < k < l < m) — 5-body candidate.
    Order4(usize, usize, usize, usize),
}

impl LibraryTerm {
    /// The number of factor variables in this monomial. Equivalent to the
    /// hyperedge order minus one (because the regression is for entity i, so
    /// term `x_j` represents the {i, j} pair, etc.).
    pub fn order(&self) -> usize {
        match self {
            Self::Order1(_) => 1,
            Self::Order2(_, _) => 2,
            Self::Order3(_, _, _) => 3,
            Self::Order4(_, _, _, _) => 4,
        }
    }

    /// Canonical sorted indices in ascending order.
    pub fn members(&self) -> Vec<usize> {
        match self {
            Self::Order1(j) => vec![*j],
            Self::Order2(j, k) => vec![*j, *k],
            Self::Order3(j, k, l) => vec![*j, *k, *l],
            Self::Order4(j, k, l, m) => vec![*j, *k, *l, *m],
        }
    }
}

// ── Parameter struct ─────────────────────────────────────────────────────────

/// Tunables for one reconstruction run.
///
/// `serde_json::from_value::<ReconstructionParams>(...)` honours all
/// `#[serde(default)]` attributes, so callers can pass partial JSON blobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionParams {
    /// Observation source. Default: `ParticipationRate`.
    #[serde(default = "default_observation")]
    pub observation: ObservationSource,
    /// Sliding-window length, in seconds. Default: 60.
    #[serde(default = "default_window_seconds")]
    pub window_seconds: i64,
    /// Time-bin width, in seconds. Default: 60.
    #[serde(default = "default_time_resolution")]
    pub time_resolution_seconds: i64,
    /// Maximum monomial order in the library. Hard cap: 4. Default: 3.
    #[serde(default = "default_max_order")]
    pub max_order: usize,
    /// L1 regularization strength. `0.0` ⇒ auto via λ_max heuristic.
    #[serde(default)]
    pub lambda_l1: f32,
    /// Derivative estimator. Default: `SavitzkyGolay{5, 2}`.
    #[serde(default)]
    pub derivative_estimator: DerivativeEstimator,
    /// Whether to symmetrize coefficient matrix entries before edge
    /// extraction. Default: `true`.
    #[serde(default = "default_true")]
    pub symmetrize: bool,
    /// Minimum |Pearson correlation| for a candidate term to enter the
    /// library. Default: 0.1. Lower values keep more library terms (slower
    /// LASSO solves) but recover weaker edges.
    #[serde(default = "default_pearson_threshold")]
    pub pearson_filter_threshold: f32,
    /// Number of time-axis bootstrap resamples for confidence scoring.
    /// Default: 10. Set to 0 to skip bootstrap.
    #[serde(default = "default_bootstrap_k")]
    pub bootstrap_k: usize,
    /// Hard cap on entity count. Reconstruction returns `InvalidInput` when
    /// the source narrative exceeds this. Default: 200.
    #[serde(default = "default_entity_cap")]
    pub entity_cap: usize,
    /// If `true`, use 5-fold cross-validation to pick lambda. Slower but
    /// more accurate. Phase 15b returns `InferenceError("CV not implemented
    /// in MVP")` per architect Q3 default.
    #[serde(default)]
    pub lambda_cv: bool,
    /// Master seed for the bootstrap resampler. Default: `0xCAFEF00D`.
    #[serde(default = "default_seed")]
    pub bootstrap_seed: u64,
}

impl Default for ReconstructionParams {
    fn default() -> Self {
        Self {
            observation: default_observation(),
            window_seconds: default_window_seconds(),
            time_resolution_seconds: default_time_resolution(),
            max_order: default_max_order(),
            lambda_l1: 0.0,
            derivative_estimator: DerivativeEstimator::default(),
            symmetrize: true,
            pearson_filter_threshold: default_pearson_threshold(),
            bootstrap_k: default_bootstrap_k(),
            entity_cap: default_entity_cap(),
            lambda_cv: false,
            bootstrap_seed: default_seed(),
        }
    }
}

fn default_observation() -> ObservationSource {
    ObservationSource::ParticipationRate
}
fn default_window_seconds() -> i64 {
    60
}
fn default_time_resolution() -> i64 {
    60
}
fn default_max_order() -> usize {
    3
}
fn default_true() -> bool {
    true
}
fn default_pearson_threshold() -> f32 {
    0.1
}
fn default_bootstrap_k() -> usize {
    10
}
fn default_entity_cap() -> usize {
    200
}
fn default_seed() -> u64 {
    0xCAFE_F00D
}

// ── Result structs ───────────────────────────────────────────────────────────

/// One hyperedge inferred from the LASSO solution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferredHyperedge {
    /// Member entity UUIDs, sorted ascending for canonical order.
    pub members: Vec<Uuid>,
    /// Hyperedge order (size). 2 = pairwise, 3 = triadic, etc.
    pub order: u8,
    /// Symmetrized weight (mean |coefficient| over equivalent permutations).
    pub weight: f32,
    /// Bootstrap retention frequency in [0, 1]. Defaults to 1.0 when
    /// `bootstrap_k = 0`.
    pub confidence: f32,
    /// Set when this pairwise edge's members also appear in a recovered
    /// higher-order edge with a higher weight (Taylor-expansion artifact).
    #[serde(default)]
    pub possible_masking_artifact: bool,
}

/// Diagnostics about the LASSO solve that produced this `ReconstructionResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixStats {
    pub n_entities: usize,
    pub n_library_terms: usize,
    pub n_timesteps: usize,
    /// Fraction of Ξ entries that are zero after thresholding.
    pub sparsity: f32,
    /// Approximate condition number — ratio of max/min Θ column norms.
    pub condition_number_approx: f32,
    /// The actual λ used (auto-selected when `params.lambda_l1 == 0.0`).
    pub lambda_used: f32,
    /// How many candidate library terms survived the Pearson pre-filter.
    pub pearson_filtered_pairs: usize,
}

/// Top-level output of one reconstruction run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionResult {
    pub inferred_edges: Vec<InferredHyperedge>,
    pub coefficient_matrix_stats: MatrixStats,
    /// Held-out R² averaged over entities. In [-∞, 1.0]; clamped to
    /// `[0.0, 1.0]` when used as the engine's confidence.
    pub goodness_of_fit: f32,
    pub observation_source: ObservationSource,
    pub params_used: ReconstructionParams,
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
    pub bootstrap_resamples_completed: usize,
    /// Best-effort warnings (e.g. "T < 10·N — system may be underdetermined").
    #[serde(default)]
    pub warnings: Vec<String>,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_round_trip() {
        let params = ReconstructionParams::default();
        let json = serde_json::to_value(&params).unwrap();
        let back: ReconstructionParams = serde_json::from_value(json).unwrap();
        assert_eq!(params, back);
    }

    #[test]
    fn test_partial_json_uses_defaults() {
        let json = serde_json::json!({"max_order": 2});
        let params: ReconstructionParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.max_order, 2);
        assert_eq!(params.bootstrap_k, default_bootstrap_k());
        assert!(matches!(
            params.derivative_estimator,
            DerivativeEstimator::SavitzkyGolay { window: 5, order: 2 }
        ));
    }

    #[test]
    fn test_library_term_order_and_members() {
        let t = LibraryTerm::Order2(1, 4);
        assert_eq!(t.order(), 2);
        assert_eq!(t.members(), vec![1, 4]);

        let t = LibraryTerm::Order3(0, 2, 5);
        assert_eq!(t.order(), 3);
        assert_eq!(t.members(), vec![0, 2, 5]);
    }

    #[test]
    fn test_inferred_hyperedge_serde() {
        let edge = InferredHyperedge {
            members: vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()],
            order: 3,
            weight: 0.42,
            confidence: 0.91,
            possible_masking_artifact: false,
        };
        let json = serde_json::to_value(&edge).unwrap();
        let back: InferredHyperedge = serde_json::from_value(json).unwrap();
        assert_eq!(edge, back);
    }
}
