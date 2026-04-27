//! Yager OWA (Ordered Weighted Averaging) operator.
//!
//! An OWA aggregator sorts the input vector in **descending** order and
//! applies a weight vector to the sorted positions. With weights `[1, 0,
//! ...]` OWA recovers `max`, with weights `[0, ..., 1]` it recovers
//! `min`, and with uniform weights it recovers the arithmetic mean.
//!
//! Linguistic-quantifier helpers translate fuzzy natural-language
//! quantifiers (e.g. *"most"*, *"few"*, *"almost all"*) into weight
//! vectors via `w_i = Q(i/n) - Q((i-1)/n)`, per Yager 1988 §5.
//!
//! Cites: [yager1988owa].
//!
//! Zadeh's original paper on linguistic quantifiers is the motivating
//! reference; Yager 1988 derives the OWA weight vector from the Q
//! function.

use crate::error::{Result, TensaError};

/// Canonical linguistic quantifiers from Yager 1988.
///
/// Each variant defines a non-decreasing `Q: [0, 1] → [0, 1]` with
/// `Q(0) = 0` and `Q(1) = 1`. The OWA weight vector is then
/// `w_i = Q(i/n) - Q((i-1)/n)` for `i = 1..=n`, guaranteeing `Σw = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantifier {
    /// `Q_most(r) = clamp((r - 0.3) / 0.5, 0, 1)` — emphasises the top ~70%.
    Most,
    /// `Q_almost_all(r) = clamp((r - 0.5) / 0.5, 0, 1)` — emphasises the top half.
    AlmostAll,
    /// `Q_few(r) = clamp(r / 0.3, 0, 1)` — emphasises the top ~30%.
    Few,
}

impl Quantifier {
    /// Evaluate `Q(r)` on `[0, 1]`. Inputs outside the range are clamped.
    #[inline]
    pub fn eval(self, r: f64) -> f64 {
        let r = r.clamp(0.0, 1.0);
        match self {
            Quantifier::Most => ((r - 0.3) / 0.5).clamp(0.0, 1.0),
            Quantifier::AlmostAll => ((r - 0.5) / 0.5).clamp(0.0, 1.0),
            Quantifier::Few => (r / 0.3).clamp(0.0, 1.0),
        }
    }
}

/// Build a length-`n` OWA weight vector from a linguistic quantifier via
/// `w_i = Q(i/n) - Q((i-1)/n)`. The resulting `Σw = Q(1) - Q(0) = 1`
/// exactly by telescoping, so the vector is always normalised.
pub fn linguistic_weights(q: Quantifier, n: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(TensaError::InvalidInput(
            "OWA: linguistic_weights requires n ≥ 1".into(),
        ));
    }
    let mut w = Vec::with_capacity(n);
    let n_f = n as f64;
    let mut prev = q.eval(0.0);
    for i in 1..=n {
        let cur = q.eval(i as f64 / n_f);
        // Clamp to non-negative — defensive against FP round-off for
        // near-equal adjacent Q values.
        w.push((cur - prev).max(0.0));
        prev = cur;
    }
    // Renormalise to kill any residual FP drift from the clamp above.
    let sum: f64 = w.iter().sum();
    if sum > 0.0 {
        for wi in &mut w {
            *wi /= sum;
        }
    }
    Ok(w)
}

/// Normalise a weight vector in-place so `Σw = 1`. Returns an error if
/// the sum is non-positive (all zeros / negative values).
pub fn owa_normalize(weights: &mut [f64]) -> Result<()> {
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return Err(TensaError::InvalidInput(format!(
            "OWA: cannot normalise weight vector with non-positive sum {}",
            sum
        )));
    }
    for w in weights.iter_mut() {
        *w /= sum;
    }
    Ok(())
}

/// Compute the Yager OWA aggregate of `xs` under the given weights.
///
/// Errors:
/// * empty `xs` → `InvalidInput("OWA: empty input")`.
/// * `|weights| != |xs|` → `InvalidInput` with both lengths.
/// * `|Σweights - 1| > 1e-9` → `InvalidInput` with a hint to call
///   [`owa_normalize`] first.
///
/// Algorithm:
/// 1. Sort `xs` in **descending** order (ties broken by insertion order —
///    `sort_by` is stable, so the resulting permutation is deterministic).
/// 2. Return `Σ w_i * x_(i)` where `x_(i)` is the `i`-th largest input.
pub fn owa(xs: &[f64], weights: &[f64]) -> Result<f64> {
    if xs.is_empty() {
        return Err(TensaError::InvalidInput("OWA: empty input".into()));
    }
    if weights.len() != xs.len() {
        return Err(TensaError::InvalidInput(format!(
            "OWA: |weights|={} must equal |xs|={}",
            weights.len(),
            xs.len()
        )));
    }
    let w_sum: f64 = weights.iter().sum();
    if (w_sum - 1.0).abs() > 1e-9 {
        return Err(TensaError::InvalidInput(format!(
            "OWA: weights must sum to 1.0 (got {}); call owa_normalize first",
            w_sum
        )));
    }

    // Indices sorted by descending xs, stable on ties to preserve
    // insertion order.
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut acc = 0.0_f64;
    for (rank, &i) in idx.iter().enumerate() {
        acc += weights[rank] * xs[i];
    }
    Ok(acc)
}
