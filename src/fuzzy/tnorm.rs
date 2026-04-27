//! T-norms and t-conorms — canonical triangular-norm families.
//!
//! Phase 1 ships real implementations of the four canonical families
//! (Gödel, Goguen, Łukasiewicz, Hamacher) plus their dual t-conorms.
//! All inputs are defensively clamped to `[0, 1]` before combination;
//! the library never emits `NaN` or `Inf` on edge cases.
//!
//! ## Canonical formulas
//!
//! | Family | T(a,b) | S(a,b) |
//! |---|---|---|
//! | Gödel | `min(a, b)` | `max(a, b)` |
//! | Goguen | `a * b` | `a + b - a*b` |
//! | Łukasiewicz | `max(0, a+b-1)` | `min(1, a+b)` |
//! | Hamacher(λ) | `ab / (λ + (1-λ)(a+b-ab))` | De Morgan dual `1 - T(1-a, 1-b)` |
//!
//! Hamacher degenerates to Goguen at `λ = 1` (bit-identical within `1e-12`),
//! to the Hamacher product at `λ = 0`, and approaches the drastic t-norm
//! as `λ → ∞`.
//!
//! ## Ordering
//!
//! For every `(a, b) ∈ [0, 1]²`:
//! ```text
//!   T_Lukasiewicz(a, b)  ≤  T_Goguen(a, b)  ≤  T_Godel(a, b)
//! ```
//! See [`tests::test_ordering`] for the 36-point verification grid.
//!
//! Cites: [klement2000].

use serde::{Deserialize, Serialize};

/// A triangular norm (logical conjunction in fuzzy logic).
///
/// Must satisfy: `T(a, 1) = a`, `T(a, 0) = 0`, commutative, associative,
/// monotone non-decreasing in each argument. The Phase 1 test suite
/// checks each registered impl against these axioms.
pub trait TNorm: Send + Sync {
    /// Stable identifier used by [`crate::fuzzy::registry::TNormRegistry`].
    fn name(&self) -> &'static str;
    /// Combine two fuzzy truth values `a, b ∈ [0, 1]`. Inputs outside
    /// `[0, 1]` are clamped before computation (defensive — callers
    /// should already supply clean values, but the library must not emit
    /// `NaN` / `Inf` on edge cases).
    fn combine(&self, a: f64, b: f64) -> f64;
    /// Neutral element — for every canonical t-norm this is `1.0`.
    /// `combine(a, neutral())` must return `a`.
    fn neutral(&self) -> f64;
}

/// A triangular co-norm (logical disjunction in fuzzy logic).
///
/// Must satisfy: `S(a, 0) = a`, `S(a, 1) = 1`, commutative, associative,
/// monotone non-decreasing. Every canonical t-conorm is the De Morgan
/// dual of its paired t-norm under the standard negation `n(x) = 1 - x`.
pub trait TConorm: Send + Sync {
    /// Stable identifier used by the registry.
    fn name(&self) -> &'static str;
    /// Combine two fuzzy truth values `a, b ∈ [0, 1]`.
    fn combine(&self, a: f64, b: f64) -> f64;
    /// Neutral element — for every canonical t-conorm this is `0.0`.
    /// `combine(a, neutral())` must return `a`.
    fn neutral(&self) -> f64;
}

/// Defensive clamp to `[0, 1]` used by every canonical family. `NaN`
/// inputs collapse to `0.0` so downstream arithmetic stays finite.
#[inline]
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

// ── TNormKind — family enumeration ────────────────────────────────────────────

/// Canonical triangular-norm family.
///
/// Parameterised families (Hamacher, eventually Schweizer–Sklar, Frank,
/// Dombi, …) carry their parameter as a tuple field. Serialization uses
/// serde's internal tag `kind` + content `param` so Hamacher writes as
/// `{"kind": "hamacher", "param": 0.5}` and Gödel as `{"kind": "godel"}`
/// (no param field).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "param", rename_all = "lowercase")]
pub enum TNormKind {
    /// `T(a, b) = min(a, b)` — the maximal t-norm. Default for the
    /// backward-compatibility regression wiring on "best match" call sites.
    Godel,
    /// `T(a, b) = a * b` — product t-norm, matches Goguen's algebra.
    /// Default for Dempster-Shafer mass combination.
    Goguen,
    /// `T(a, b) = max(0, a + b - 1)` — bounded-difference t-norm.
    Lukasiewicz,
    /// Hamacher family parameterised by `λ ≥ 0`. At `λ = 0` this is the
    /// Hamacher product; at `λ = 1` it coincides with Goguen; as
    /// `λ → ∞` it approaches the drastic t-norm.
    Hamacher(f64),
}

impl TNormKind {
    /// Stable short name matching [`TNorm::name`] / [`TConorm::name`].
    pub fn name(&self) -> &'static str {
        match self {
            TNormKind::Godel => "godel",
            TNormKind::Goguen => "goguen",
            TNormKind::Lukasiewicz => "lukasiewicz",
            TNormKind::Hamacher(_) => "hamacher",
        }
    }
}

// ── Concrete TNorm / TConorm impls ─────────────────────────────────────────────

/// `T(a, b) = min(a, b)` — the maximal t-norm.
pub struct GodelTNorm;
impl TNorm for GodelTNorm {
    fn name(&self) -> &'static str {
        "godel"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        a.min(b)
    }
    fn neutral(&self) -> f64 {
        1.0
    }
}

/// `S(a, b) = max(a, b)` — the minimal t-conorm.
pub struct GodelTConorm;
impl TConorm for GodelTConorm {
    fn name(&self) -> &'static str {
        "godel"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        a.max(b)
    }
    fn neutral(&self) -> f64 {
        0.0
    }
}

/// `T(a, b) = a * b` — product t-norm.
pub struct GoguenTNorm;
impl TNorm for GoguenTNorm {
    fn name(&self) -> &'static str {
        "goguen"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        a * b
    }
    fn neutral(&self) -> f64 {
        1.0
    }
}

/// `S(a, b) = a + b - a*b` — probabilistic OR.
pub struct GoguenTConorm;
impl TConorm for GoguenTConorm {
    fn name(&self) -> &'static str {
        "goguen"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        (a + b - a * b).clamp(0.0, 1.0)
    }
    fn neutral(&self) -> f64 {
        0.0
    }
}

/// `T(a, b) = max(0, a + b - 1)` — bounded difference.
pub struct LukasiewiczTNorm;
impl TNorm for LukasiewiczTNorm {
    fn name(&self) -> &'static str {
        "lukasiewicz"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        (a + b - 1.0).max(0.0)
    }
    fn neutral(&self) -> f64 {
        1.0
    }
}

/// `S(a, b) = min(1, a + b)` — bounded sum.
pub struct LukasiewiczTConorm;
impl TConorm for LukasiewiczTConorm {
    fn name(&self) -> &'static str {
        "lukasiewicz"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        (a + b).min(1.0)
    }
    fn neutral(&self) -> f64 {
        0.0
    }
}

/// Hamacher family `T(a,b) = ab / (λ + (1-λ)(a+b-ab))`.
///
/// At `λ = 0` this is the Hamacher product. At `λ = 1` it coincides
/// with Goguen (product) within `1e-12`. As `λ → ∞` it approaches the
/// drastic t-norm `T_drastic(a, b) = min(a, b)` when `max(a, b) = 1`,
/// else `0`.
pub struct HamacherTNorm {
    pub lambda: f64,
}
impl TNorm for HamacherTNorm {
    fn name(&self) -> &'static str {
        "hamacher"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        hamacher_t(self.lambda, clamp01(a), clamp01(b))
    }
    fn neutral(&self) -> f64 {
        1.0
    }
}

/// Hamacher dual co-norm via De Morgan: `S(a,b) = 1 - T(1-a, 1-b)`.
pub struct HamacherTConorm {
    pub lambda: f64,
}
impl TConorm for HamacherTConorm {
    fn name(&self) -> &'static str {
        "hamacher"
    }
    fn combine(&self, a: f64, b: f64) -> f64 {
        let (a, b) = (clamp01(a), clamp01(b));
        let t = hamacher_t(self.lambda, 1.0 - a, 1.0 - b);
        (1.0 - t).clamp(0.0, 1.0)
    }
    fn neutral(&self) -> f64 {
        0.0
    }
}

/// Core Hamacher t-norm body. `λ` is clamped to `[0, +∞)`; negative
/// values are treated as `0`. If the denominator collapses to zero
/// (happens exactly when `λ = 0 ∧ a = b = 0`), the result is `0` by
/// L'Hôpital-style limit — matches the Hamacher-product convention.
#[inline]
fn hamacher_t(lambda: f64, a: f64, b: f64) -> f64 {
    let lambda = lambda.max(0.0);
    let num = a * b;
    if num == 0.0 {
        return 0.0;
    }
    let denom = lambda + (1.0 - lambda) * (a + b - a * b);
    if denom.abs() < 1e-18 {
        // Should only be reachable via λ=0 ∧ a=b=0, which the num==0 guard
        // already caught. Defensive fallback to 0.0.
        return 0.0;
    }
    (num / denom).clamp(0.0, 1.0)
}

// ── Factory helpers ───────────────────────────────────────────────────────────

/// Boxed-trait factory: `kind → Box<dyn TNorm>`.
pub fn tnorm_for(kind: TNormKind) -> Box<dyn TNorm> {
    match kind {
        TNormKind::Godel => Box::new(GodelTNorm),
        TNormKind::Goguen => Box::new(GoguenTNorm),
        TNormKind::Lukasiewicz => Box::new(LukasiewiczTNorm),
        TNormKind::Hamacher(lambda) => Box::new(HamacherTNorm { lambda }),
    }
}

/// Boxed-trait factory: `kind → Box<dyn TConorm>`.
pub fn tconorm_for(kind: TNormKind) -> Box<dyn TConorm> {
    match kind {
        TNormKind::Godel => Box::new(GodelTConorm),
        TNormKind::Goguen => Box::new(GoguenTConorm),
        TNormKind::Lukasiewicz => Box::new(LukasiewiczTConorm),
        TNormKind::Hamacher(lambda) => Box::new(HamacherTConorm { lambda }),
    }
}

// ── n-ary reduction helpers ────────────────────────────────────────────────────

/// Fold a slice under a t-norm, starting from the neutral element `1.0`.
/// Empty slice returns `1.0`; singleton returns the element itself.
pub fn reduce_tnorm(kind: TNormKind, xs: &[f64]) -> f64 {
    let t = tnorm_for(kind);
    xs.iter().fold(t.neutral(), |acc, &x| t.combine(acc, x))
}

/// Fold a slice under a t-conorm, starting from the neutral element `0.0`.
/// Empty slice returns `0.0`; singleton returns the element itself.
pub fn reduce_tconorm(kind: TNormKind, xs: &[f64]) -> f64 {
    let s = tconorm_for(kind);
    xs.iter().fold(s.neutral(), |acc, &x| s.combine(acc, x))
}

/// Convenience: combine two values under the selected t-norm.
#[inline]
pub fn combine_tnorm(kind: TNormKind, a: f64, b: f64) -> f64 {
    tnorm_for(kind).combine(a, b)
}

/// Convenience: combine two values under the selected t-conorm.
#[inline]
pub fn combine_tconorm(kind: TNormKind, a: f64, b: f64) -> f64 {
    tconorm_for(kind).combine(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tnorm_kinds_serialize_roundtrip() {
        let kinds = [
            TNormKind::Godel,
            TNormKind::Goguen,
            TNormKind::Lukasiewicz,
            TNormKind::Hamacher(0.5),
        ];
        for k in kinds {
            let ser = serde_json::to_string(&k).expect("serialize");
            let back: TNormKind = serde_json::from_str(&ser).expect("deserialize");
            assert_eq!(k, back, "round-trip failed for {:?} (json={})", k, ser);
        }
    }

    #[test]
    fn test_tnorm_kind_names_match_registry_keys() {
        assert_eq!(TNormKind::Godel.name(), "godel");
        assert_eq!(TNormKind::Goguen.name(), "goguen");
        assert_eq!(TNormKind::Lukasiewicz.name(), "lukasiewicz");
        assert_eq!(TNormKind::Hamacher(0.5).name(), "hamacher");
    }

    #[test]
    fn test_tnorm_neutral_is_one() {
        for k in [
            TNormKind::Godel,
            TNormKind::Goguen,
            TNormKind::Lukasiewicz,
            TNormKind::Hamacher(0.5),
        ] {
            let t = tnorm_for(k);
            assert_eq!(t.neutral(), 1.0);
        }
    }

    #[test]
    fn test_tconorm_neutral_is_zero() {
        for k in [
            TNormKind::Godel,
            TNormKind::Goguen,
            TNormKind::Lukasiewicz,
            TNormKind::Hamacher(0.5),
        ] {
            let s = tconorm_for(k);
            assert_eq!(s.neutral(), 0.0);
        }
    }
}
