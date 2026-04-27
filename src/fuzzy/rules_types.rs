//! Mamdani-style fuzzy rule system — type definitions.
//!
//! Phase 9. The evaluation pipeline lives in
//! [`super::rules_eval`]; KV persistence in [`super::rules_store`].
//! Split out of [`super::rules`] to keep each file under the 500-line cap.
//!
//! Cites: [mamdani1975mamdani].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::tnorm::TNormKind;

// ── Membership functions ─────────────────────────────────────────────────────

/// Shape of a fuzzy membership function μ : ℝ → [0, 1].
///
/// Three canonical shapes cover the Mamdani paper's working examples plus
/// the generalisation to smooth kernels:
///
/// * [`MembershipFunction::Triangular`] `{a, b, c}` with `a ≤ b ≤ c` —
///   peaks at `b`, zero at `a` and `c`, linear on the two legs.
/// * [`MembershipFunction::Trapezoidal`] `{a, b, c, d}` with `a ≤ b ≤ c ≤ d` —
///   zero at `a` and `d`, one on the plateau `[b, c]`.
/// * [`MembershipFunction::Gaussian`] `{mean, sigma}` with `sigma > 0` —
///   `exp(-(x - mean)² / (2·sigma²))`. Peaks at `mean`.
///
/// Shapes with degenerate parameters (`a > b`, `sigma ≤ 0`, …) return
/// `0.0` rather than `NaN` — the library never emits `NaN` / `Inf` on
/// edge cases (same invariant as the t-norm layer).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MembershipFunction {
    /// Triangle with legs `[a, b]` (rising) and `[b, c]` (falling).
    Triangular {
        a: f64,
        b: f64,
        c: f64,
    },
    /// Trapezoid with rising leg `[a, b]`, plateau `[b, c]`, falling leg `[c, d]`.
    Trapezoidal {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    },
    /// Gaussian bell `exp(-(x - mean)² / (2·sigma²))`.
    Gaussian {
        mean: f64,
        sigma: f64,
    },
}

impl MembershipFunction {
    /// Evaluate the membership degree at `x`, clamped to `[0, 1]`.
    pub fn membership(&self, x: f64) -> f64 {
        if x.is_nan() {
            return 0.0;
        }
        let mu = match *self {
            MembershipFunction::Triangular { a, b, c } => triangle_mu(a, b, c, x),
            MembershipFunction::Trapezoidal { a, b, c, d } => trapezoid_mu(a, b, c, d, x),
            MembershipFunction::Gaussian { mean, sigma } => gaussian_mu(mean, sigma, x),
        };
        if mu.is_nan() {
            0.0
        } else {
            mu.clamp(0.0, 1.0)
        }
    }

    /// Return the representative `[lo, hi]` support interval used for
    /// defuzzification discretisation. Gaussian uses `mean ± 3σ` which
    /// covers ~99.7% of the mass.
    pub fn support(&self) -> (f64, f64) {
        match *self {
            MembershipFunction::Triangular { a, c, .. } => (a.min(c), a.max(c)),
            MembershipFunction::Trapezoidal { a, d, .. } => (a.min(d), a.max(d)),
            MembershipFunction::Gaussian { mean, sigma } => {
                let s = sigma.abs().max(1e-9);
                (mean - 3.0 * s, mean + 3.0 * s)
            }
        }
    }
}

#[inline]
fn triangle_mu(a: f64, b: f64, c: f64, x: f64) -> f64 {
    if !(a <= b && b <= c) {
        return 0.0;
    }
    if x <= a || x >= c {
        return 0.0;
    }
    if (x - b).abs() < 1e-12 {
        return 1.0;
    }
    if x < b {
        if (b - a).abs() < 1e-12 {
            return 1.0;
        }
        (x - a) / (b - a)
    } else {
        if (c - b).abs() < 1e-12 {
            return 1.0;
        }
        (c - x) / (c - b)
    }
}

#[inline]
fn trapezoid_mu(a: f64, b: f64, c: f64, d: f64, x: f64) -> f64 {
    if !(a <= b && b <= c && c <= d) {
        return 0.0;
    }
    if x <= a || x >= d {
        return 0.0;
    }
    if x >= b && x <= c {
        return 1.0;
    }
    if x < b {
        if (b - a).abs() < 1e-12 {
            return 1.0;
        }
        return (x - a) / (b - a);
    }
    // x > c && x < d
    if (d - c).abs() < 1e-12 {
        return 1.0;
    }
    (d - x) / (d - c)
}

#[inline]
fn gaussian_mu(mean: f64, sigma: f64, x: f64) -> f64 {
    if sigma <= 0.0 || !sigma.is_finite() {
        return 0.0;
    }
    let z = (x - mean) / sigma;
    (-0.5 * z * z).exp()
}

// ── Conditions + outputs ─────────────────────────────────────────────────────

/// One antecedent clause of a Mamdani rule.
///
/// `variable_path` is a dot-path into the entity's JSON properties.
/// A leading `"entity.properties."` prefix is stripped at resolution
/// time so callers can write either `"entity.properties.score"` or
/// just `"score"` with identical effect. `"entity.confidence"`
/// resolves to the entity confidence. `"entity.entity_type"`
/// resolves to the entity-type string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyCondition {
    /// Dot-path into `entity.properties.*` or one of the reserved
    /// `entity.confidence` / `entity.entity_type` keys.
    pub variable_path: String,
    /// Membership function over the resolved variable value.
    pub membership: MembershipFunction,
    /// Human-readable linguistic term (e.g. `"high"`, `"highly-partisan"`).
    /// Surfaced in rule-firing traces so analysts read firings as
    /// `"age IS high (μ=0.72)"` rather than as bare numbers.
    #[serde(default)]
    pub linguistic_term: String,
}

/// Consequent of a Mamdani rule. `variable` is an opaque output-space
/// name (e.g. `"disinfo_risk"`); the defuzzified scalar is computed over
/// `membership.support()` discretised into [`DEFAULT_DEFUZZ_BINS`] bins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyOutput {
    /// Output-variable name (free-form).
    pub variable: String,
    /// Membership function over the output universe.
    pub membership: MembershipFunction,
    /// Human-readable linguistic term (e.g. `"elevated"`, `"low"`).
    #[serde(default)]
    pub linguistic_term: String,
}

/// Defuzzification strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Defuzzification {
    /// Centre-of-area over the aggregated output fuzzy set.
    Centroid,
    /// Mean-of-maxima — mean x over bins where aggregated μ is maximal.
    MeanOfMaxima,
}

impl Default for Defuzzification {
    fn default() -> Self {
        Defuzzification::Centroid
    }
}

/// Number of discretisation bins used by defuzzification. Keeps
/// Centroid / MeanOfMaxima finite-sum implementations cheap while
/// giving analysts ~1% resolution on typical output ranges.
pub const DEFAULT_DEFUZZ_BINS: usize = 100;

// ── Rule + rule set ──────────────────────────────────────────────────────────

/// A Mamdani-style fuzzy rule `IF <antecedent> THEN <consequent>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MamdaniRule {
    /// Stable v7 UUID — chronological key ordering.
    pub id: Uuid,
    /// Human-readable name (e.g. `"elevated-disinfo-risk"`).
    pub name: String,
    /// Narrative this rule is scoped to.
    pub narrative_id: String,
    /// Ordered list of antecedent conditions (combined under `tnorm`).
    pub antecedent: Vec<FuzzyCondition>,
    /// Consequent clause.
    pub consequent: FuzzyOutput,
    /// T-norm used to fold antecedent μ values into a firing strength.
    /// Default Gödel (`min`). Serde-default so legacy payloads load.
    #[serde(default = "default_tnorm")]
    pub tnorm: TNormKind,
    /// Created-at timestamp (v7 UUIDs already encode this, but we
    /// also surface it at the JSON top level for tooling).
    #[serde(default = "chrono::Utc::now")]
    pub created_at: DateTime<Utc>,
    /// Enable/disable switch. Disabled rules are ignored by rule-set
    /// evaluation (they still round-trip through KV).
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_tnorm() -> TNormKind {
    TNormKind::Godel
}

fn default_enabled() -> bool {
    true
}

/// An ordered collection of Mamdani rules plus the defuzzification
/// strategy to apply to their aggregated output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MamdaniRuleSet {
    pub rules: Vec<MamdaniRule>,
    #[serde(default)]
    pub defuzzification: Defuzzification,
    /// Optional aggregator used to fold per-rule firing strengths into
    /// a single summary scalar (reported alongside the defuzzified
    /// output). `None` preserves the classical Mamdani contract where
    /// the reducer is embedded in the aggregation-of-scaled-consequents
    /// step only.
    #[serde(default)]
    pub firing_aggregator: Option<AggregatorKind>,
}

impl MamdaniRuleSet {
    /// Convenience constructor — rules, Centroid defuzzification, no
    /// firing aggregator.
    pub fn new(rules: Vec<MamdaniRule>) -> Self {
        Self {
            rules,
            defuzzification: Defuzzification::default(),
            firing_aggregator: None,
        }
    }
}

// ── Evaluation result shape ──────────────────────────────────────────────────

/// Rule-firing summary returned by the evaluation pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiredRule {
    pub rule_id: Uuid,
    pub rule_name: String,
    pub firing_strength: f64,
    /// Per-antecedent μ values (in antecedent order) so analysts can
    /// see which clauses drove the firing strength.
    pub per_antecedent_mu: Vec<f64>,
}

/// Top-level rule-set evaluation descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleSetEvaluation {
    pub entity_id: Uuid,
    pub fired_rules: Vec<FiredRule>,
    /// Defuzzified output scalar. `None` when no rule fired with
    /// strength > 0 — there is no fuzzy set to defuzzify.
    pub defuzzified_output: Option<f64>,
    /// Defuzzification strategy used (echoes the rule set's).
    pub defuzzification: Defuzzification,
    /// Optional firing-strength aggregate when the rule set carries a
    /// `firing_aggregator`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firing_aggregate: Option<f64>,
}
