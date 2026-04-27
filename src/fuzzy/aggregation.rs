//! Aggregation operators: Mean / Median / OWA (Yager) / Choquet integral /
//! t-norm reduction / t-conorm reduction.
//!
//! Phase 1 wired real bodies for `Mean`, `Median`, `TNormReduce`, and
//! `TConormReduce`. Phase 2 lands OWA + Choquet via the dedicated
//! sub-modules:
//! * [`crate::fuzzy::aggregation_owa`] — Yager OWA + linguistic-
//!   quantifier weight vectors (Most / AlmostAll / Few).
//! * [`crate::fuzzy::aggregation_choquet`] — exact Choquet for n ≤ 10 +
//!   Monte-Carlo fallback for larger universes with standard-error
//!   reporting.
//! * [`crate::fuzzy::aggregation_measure`] — `FuzzyMeasure` monotonicity
//!   check + Möbius transform + built-in symmetric measures.
//!
//! This file now holds the `Aggregator` trait + `AggregatorKind` enum +
//! the boxed-trait factory; the concrete numerics live in the sub-modules
//! so each file stays under the 500-line cap.
//!
//! Cites: [yager1988owa] [grabisch1996choquet] [bustince2016choquet]
//!        [grabisch2000fuzzymeasure] [klement2000].

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::TNormKind;

// ── Aggregator trait ──────────────────────────────────────────────────────────

/// An aggregator maps a vector of fuzzy truth values to a single value in
/// `[0, 1]`. Unlike a t-norm, aggregators are not required to be strictly
/// associative / commutative in the general case — OWA is
/// order-dependent; Choquet is measure-dependent.
pub trait Aggregator: Send + Sync {
    /// Stable identifier for the registry.
    fn name(&self) -> &'static str;
    /// Fold a slice of fuzzy values into a single aggregated value.
    fn aggregate(&self, xs: &[f64]) -> Result<f64>;
}

// ── AggregatorKind — enumerable variants ──────────────────────────────────────

/// Aggregator family.
///
/// Parameterised variants carry their parameter inline; serialization
/// uses the default externally-tagged shape so callers can express
/// configuration directly as JSON:
/// * `"Mean"`, `"Median"`
/// * `{ "Owa": [0.5, 0.3, 0.2] }`
/// * `{ "Choquet": { "n": 2, "values": [0.0, 0.3, 0.4, 1.0] } }`
/// * `{ "TNormReduce": { "kind": "godel" } }`
/// * `{ "TConormReduce": { "kind": "goguen" } }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregatorKind {
    /// Arithmetic mean.
    Mean,
    /// Sample median.
    Median,
    /// Yager OWA with per-rank weights (length must equal input length).
    Owa(Vec<f64>),
    /// Choquet integral with respect to a fuzzy measure.
    Choquet(FuzzyMeasure),
    /// Fold with a t-norm (`combine` applied left-associative, identity on
    /// the empty input, clamped to `[0, 1]`).
    TNormReduce(TNormKind),
    /// Fold with a t-conorm (dual of `TNormReduce`).
    TConormReduce(TNormKind),
}

impl AggregatorKind {
    /// Stable short name matching the registry key.
    pub fn name(&self) -> &'static str {
        match self {
            AggregatorKind::Mean => "mean",
            AggregatorKind::Median => "median",
            AggregatorKind::Owa(_) => "owa",
            AggregatorKind::Choquet(_) => "choquet",
            AggregatorKind::TNormReduce(_) => "tnorm_reduce",
            AggregatorKind::TConormReduce(_) => "tconorm_reduce",
        }
    }
}

// ── FuzzyMeasure — Choquet capacity carrier ───────────────────────────────────

/// A fuzzy (non-additive) measure over a universe of `n` elements.
///
/// Stored as a flat `Vec<f64>` indexed by the subset's binary encoding —
/// `values[0]` is `μ(∅)`, `values[2^n - 1]` is `μ(full universe)`. The
/// Möbius-transform path used by the Choquet integral requires the full
/// `2^n` table, so Phase 2 caps the exact path at `n ≤ 10` (1024 entries)
/// and falls back to Monte-Carlo above.
///
/// Phase 0 validation is minimal — length + endpoint checks only. Phase 2
/// adds monotonicity (`A ⊆ B ⇒ μ(A) ≤ μ(B)`) and normalisation
/// (`μ(∅) = 0 ∧ μ(full) = 1`) checks before the Choquet engine wires in.
///
/// ## Provenance slots (Graded Acceptability sprint, Phase 2)
///
/// `measure_id` + `measure_version` carry the identity of a learned
/// measure when one is in flight (see [`crate::fuzzy::aggregation_learn`]).
/// Both default to `None`, are `#[serde(skip_serializing_if = "Option::is_none")]`,
/// and pass through every `_tracked` workflow wire ([§4.5 of
/// `docs/choquet_learning_algorithm.md`](../../docs/choquet_learning_algorithm.md)).
/// Built-in symmetric measures and hand-rolled `Manual` measures leave
/// the slots `None` so existing wire envelopes stay bit-identical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyMeasure {
    /// Size of the universe (`μ` is a mapping from the `2^n`-element power
    /// set to `[0, 1]`). Capped at 10 to keep storage bounded.
    pub n: u8,
    /// `values.len() == 2^n`. `values[subset_mask] = μ(subset)`.
    pub values: Vec<f64>,
    /// Optional identifier of the learned measure this capacity was loaded
    /// from. Populated by the planner / REST handlers when a request
    /// resolves a `measure=<name>` reference; left `None` for built-in
    /// symmetric measures and ad-hoc inline construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure_id: Option<String>,
    /// Optional version stamp paired with `measure_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure_version: Option<u32>,
}

impl FuzzyMeasure {
    /// Phase 0 constructor — enforces the length + endpoint invariants and
    /// the n ≤ 16 cap. Monotonicity is NOT checked here (Phase 2 adds the
    /// check at [`crate::fuzzy::aggregation_measure::new_monotone`] /
    /// [`crate::fuzzy::aggregation_measure::check_monotone`]; callers that
    /// need the stricter contract route through the measure sub-module).
    ///
    /// The n cap is pushed to 16 so Monte-Carlo Choquet (2^16 = 65 536
    /// lookup entries) is reachable without overflowing the stack — the
    /// exact Choquet path in
    /// [`crate::fuzzy::aggregation_choquet::choquet_exact`] still caps
    /// itself at n ≤ [`crate::fuzzy::aggregation_choquet::EXACT_N_CAP`].
    pub fn new(n: u8, values: Vec<f64>) -> Result<Self> {
        if n > 16 {
            return Err(TensaError::InvalidInput(format!(
                "fuzzy measure n must be ≤ 16 (got {}); 2^n = {} lookup entries \
                 would otherwise exhaust the heap budget",
                n,
                1u128 << (n as u32)
            )));
        }
        let expected = 1usize << (n as usize);
        if values.len() != expected {
            return Err(TensaError::InvalidInput(format!(
                "fuzzy measure requires exactly 2^n = {} values, got {}",
                expected,
                values.len()
            )));
        }
        // Endpoints: μ(∅) = 0, μ(full) = 1.
        if (values[0] - 0.0).abs() > 1e-12 {
            return Err(TensaError::InvalidInput(format!(
                "fuzzy measure μ(∅) must equal 0.0, got {}",
                values[0]
            )));
        }
        let last = values[expected - 1];
        if (last - 1.0).abs() > 1e-12 {
            return Err(TensaError::InvalidInput(format!(
                "fuzzy measure μ(full universe) must equal 1.0, got {}",
                last
            )));
        }
        // Phase 0: individual values must still be in [0, 1].
        for (i, &v) in values.iter().enumerate() {
            if !(0.0..=1.0).contains(&v) {
                return Err(TensaError::InvalidInput(format!(
                    "fuzzy measure value at index {} out of [0, 1]: {}",
                    i, v
                )));
            }
        }
        Ok(Self {
            n,
            values,
            measure_id: None,
            measure_version: None,
        })
    }

    /// Same as [`Self::new`] but tags the result with a `(measure_id,
    /// measure_version)` pair. Phase 2 introduced this so REST / planner
    /// resolvers can stamp identity onto a learned measure without losing
    /// the validation [`Self::new`] performs.
    pub fn with_id(
        n: u8,
        values: Vec<f64>,
        measure_id: Option<String>,
        measure_version: Option<u32>,
    ) -> Result<Self> {
        let mut m = Self::new(n, values)?;
        m.measure_id = measure_id;
        m.measure_version = measure_version;
        Ok(m)
    }

    /// The trivial measure `μ(S) = 0` for `S ≠ full` and `μ(full) = 1`.
    /// Useful as a placeholder / default for registry constructors.
    ///
    /// Requires `n ≥ 1` — an empty universe has no proper subsets and the
    /// single entry would have to be simultaneously `μ(∅) = 0` and
    /// `μ(full) = 1`. Callers passing `n = 0` get an `InvalidInput` error.
    pub fn trivial(n: u8) -> Result<Self> {
        if n == 0 {
            return Err(TensaError::InvalidInput(
                "fuzzy measure n must be ≥ 1 — an empty universe has no proper subsets".into(),
            ));
        }
        let size = 1usize << (n as usize);
        let mut values = vec![0.0; size];
        values[size - 1] = 1.0;
        Self::new(n, values)
    }
}

// ── Concrete aggregator stubs — Phase 1/2 fill these in ──────────────────────

/// Arithmetic mean. Empty input returns `0.0`.
pub struct MeanAggregator;
impl Aggregator for MeanAggregator {
    fn name(&self) -> &'static str {
        "mean"
    }
    fn aggregate(&self, xs: &[f64]) -> Result<f64> {
        if xs.is_empty() {
            return Ok(0.0);
        }
        let sum: f64 = xs.iter().map(|x| x.clamp(0.0, 1.0)).sum();
        Ok(sum / xs.len() as f64)
    }
}

/// Sample median (midpoint of two middle values on even count).
/// Empty input returns `0.0`.
pub struct MedianAggregator;
impl Aggregator for MedianAggregator {
    fn name(&self) -> &'static str {
        "median"
    }
    fn aggregate(&self, xs: &[f64]) -> Result<f64> {
        if xs.is_empty() {
            return Ok(0.0);
        }
        let mut sorted: Vec<f64> = xs.iter().map(|x| x.clamp(0.0, 1.0)).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let mid = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        };
        Ok(mid)
    }
}

/// Yager OWA — weighted average over sorted inputs.
///
/// Delegates to [`crate::fuzzy::aggregation_owa::owa`]; see that module
/// for the full algorithm, error contract, and linguistic-quantifier
/// helpers.
pub struct OwaAggregator {
    pub weights: Vec<f64>,
}
impl Aggregator for OwaAggregator {
    fn name(&self) -> &'static str {
        "owa"
    }
    fn aggregate(&self, xs: &[f64]) -> Result<f64> {
        crate::fuzzy::aggregation_owa::owa(xs, &self.weights)
    }
}

/// Choquet integral over a fuzzy measure.
///
/// Delegates to [`crate::fuzzy::aggregation_choquet::choquet`], which
/// picks the exact `O(n · 2^n)` path for `n ≤ 10` and the Monte-Carlo
/// estimator above that cap. Phase 2 seeds the MC path with `0` by
/// default; callers wanting reproducible MC should invoke the sub-module
/// helpers directly.
pub struct ChoquetAggregator {
    pub measure: FuzzyMeasure,
}
impl Aggregator for ChoquetAggregator {
    fn name(&self) -> &'static str {
        "choquet"
    }
    fn aggregate(&self, xs: &[f64]) -> Result<f64> {
        let r = crate::fuzzy::aggregation_choquet::choquet(xs, &self.measure, 0)?;
        Ok(r.value)
    }
}

/// Left-fold under a t-norm. Empty input returns the t-norm neutral
/// element (`1.0`), matching standard fuzzy-logic convention.
pub struct TNormReduceAggregator {
    pub kind: TNormKind,
}
impl Aggregator for TNormReduceAggregator {
    fn name(&self) -> &'static str {
        "tnorm_reduce"
    }
    fn aggregate(&self, xs: &[f64]) -> Result<f64> {
        Ok(crate::fuzzy::tnorm::reduce_tnorm(self.kind, xs))
    }
}

/// Left-fold under a t-conorm. Empty input returns the t-conorm
/// neutral element (`0.0`).
pub struct TConormReduceAggregator {
    pub kind: TNormKind,
}
impl Aggregator for TConormReduceAggregator {
    fn name(&self) -> &'static str {
        "tconorm_reduce"
    }
    fn aggregate(&self, xs: &[f64]) -> Result<f64> {
        Ok(crate::fuzzy::tnorm::reduce_tconorm(self.kind, xs))
    }
}

/// Boxed-trait factory: `AggregatorKind → Box<dyn Aggregator>`.
pub fn aggregator_for(kind: AggregatorKind) -> Box<dyn Aggregator> {
    match kind {
        AggregatorKind::Mean => Box::new(MeanAggregator),
        AggregatorKind::Median => Box::new(MedianAggregator),
        AggregatorKind::Owa(weights) => Box::new(OwaAggregator { weights }),
        AggregatorKind::Choquet(measure) => Box::new(ChoquetAggregator { measure }),
        AggregatorKind::TNormReduce(kind) => Box::new(TNormReduceAggregator { kind }),
        AggregatorKind::TConormReduce(kind) => Box::new(TConormReduceAggregator { kind }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_measure_new_accepts_valid() {
        // n=2 → 4 entries, μ(∅)=0, μ(full)=1, intermediate ∈ [0,1].
        let m = FuzzyMeasure::new(2, vec![0.0, 0.3, 0.4, 1.0]).expect("valid");
        assert_eq!(m.n, 2);
        assert_eq!(m.values.len(), 4);
    }

    #[test]
    fn test_fuzzy_measure_rejects_wrong_length() {
        assert!(FuzzyMeasure::new(2, vec![0.0, 1.0]).is_err());
        assert!(FuzzyMeasure::new(2, vec![0.0, 0.3, 0.4, 0.5, 1.0]).is_err());
    }

    #[test]
    fn test_fuzzy_measure_rejects_wrong_endpoints() {
        assert!(FuzzyMeasure::new(2, vec![0.1, 0.3, 0.4, 1.0]).is_err());
        assert!(FuzzyMeasure::new(2, vec![0.0, 0.3, 0.4, 0.9]).is_err());
    }

    #[test]
    fn test_fuzzy_measure_rejects_out_of_range_value() {
        assert!(FuzzyMeasure::new(2, vec![0.0, -0.1, 0.4, 1.0]).is_err());
        assert!(FuzzyMeasure::new(2, vec![0.0, 1.2, 0.4, 1.0]).is_err());
    }

    #[test]
    fn test_fuzzy_measure_caps_n_at_16() {
        // n = 17 → 131 072 entries. Phase 2 lifted the cap to 16 to
        // accommodate Monte-Carlo Choquet beyond EXACT_N_CAP = 10; any
        // n above 16 is rejected regardless of whether the caller filled
        // the full 2^n vec.
        let size = 1usize << 17;
        let mut v = vec![0.0; size];
        v[size - 1] = 1.0;
        assert!(FuzzyMeasure::new(17, v).is_err());
    }

    #[test]
    fn test_fuzzy_measure_trivial_is_valid() {
        // Trivial measure — μ(S) = 0 for all S ≠ full — is well-formed
        // for every n ≥ 1. n = 0 is rejected because the single entry
        // would have to be both μ(∅) = 0 and μ(full) = 1 simultaneously.
        assert!(FuzzyMeasure::trivial(0).is_err());
        for n in 1..=4u8 {
            let m = FuzzyMeasure::trivial(n).expect("trivial must be valid");
            assert_eq!(m.values[0], 0.0);
            assert_eq!(*m.values.last().unwrap(), 1.0);
        }
    }

    #[test]
    fn test_aggregator_kind_names() {
        assert_eq!(AggregatorKind::Mean.name(), "mean");
        assert_eq!(AggregatorKind::Median.name(), "median");
        assert_eq!(AggregatorKind::Owa(vec![1.0]).name(), "owa");
        assert_eq!(
            AggregatorKind::Choquet(FuzzyMeasure::trivial(1).unwrap()).name(),
            "choquet"
        );
        assert_eq!(
            AggregatorKind::TNormReduce(TNormKind::Godel).name(),
            "tnorm_reduce"
        );
        assert_eq!(
            AggregatorKind::TConormReduce(TNormKind::Godel).name(),
            "tconorm_reduce"
        );
    }

    #[test]
    fn test_aggregator_kind_serialize_roundtrip() {
        let cases = vec![
            AggregatorKind::Mean,
            AggregatorKind::Median,
            AggregatorKind::Owa(vec![0.5, 0.3, 0.2]),
            AggregatorKind::Choquet(FuzzyMeasure::new(2, vec![0.0, 0.3, 0.4, 1.0]).unwrap()),
            AggregatorKind::TNormReduce(TNormKind::Lukasiewicz),
            AggregatorKind::TConormReduce(TNormKind::Hamacher(0.5)),
        ];
        for k in cases {
            let ser = serde_json::to_string(&k).expect("serialize");
            let back: AggregatorKind = serde_json::from_str(&ser).expect("deserialize");
            assert_eq!(k, back, "round-trip failed for {:?} (json={})", k, ser);
        }
    }
}
