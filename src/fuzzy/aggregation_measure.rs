//! FuzzyMeasure monotonicity check + Möbius transform + default
//! symmetric measures.
//!
//! A fuzzy (non-additive) measure `μ` on a finite universe `N = {0, ..
//! n-1}` satisfies:
//! * `μ(∅) = 0`,
//! * `μ(N) = 1`,
//! * `A ⊆ B ⟹ μ(A) ≤ μ(B)` (monotonicity).
//!
//! The Möbius transform `m(A) = Σ_{B⊆A} (-1)^(|A|-|B|) μ(B)` recovers a
//! signed measure on the power set; the inverse is `μ(A) = Σ_{B⊆A}
//! m(B)`. Phase 2 exposes both directions as helpers so the Choquet
//! engine can switch representations when needed — the exact Choquet
//! path itself uses `μ` directly, but Möbius-based sensitivity / Shapley-
//! value computations live downstream.
//!
//! Cites: [grabisch1996choquet] [grabisch2000fuzzymeasure].

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::FuzzyMeasure;

/// Verify `μ` is monotone: `A ⊆ B ⟹ μ(A) ≤ μ(B)`.
///
/// Implemented with the "add one element" characterisation — `μ` is
/// monotone iff `μ(S) ≤ μ(S ∪ {i})` for every `S` and every `i ∉ S`.
/// That gives an `O(n · 2^n)` check rather than `O(3^n)` for the naive
/// subset-of-subset enumeration.
pub fn check_monotone(measure: &FuzzyMeasure) -> Result<()> {
    let n = measure.n as usize;
    let size = 1usize << n;
    for s in 0..size {
        let mu_s = measure.values[s];
        for i in 0..n {
            let bit = 1usize << i;
            if s & bit != 0 {
                continue;
            }
            let t = s | bit;
            let mu_t = measure.values[t];
            if mu_t + 1e-12 < mu_s {
                return Err(TensaError::InvalidInput(format!(
                    "fuzzy measure violates monotonicity: μ(mask={:b})={} > μ(mask={:b})={} \
                     despite the former being a subset of the latter",
                    s, mu_s, t, mu_t
                )));
            }
        }
    }
    Ok(())
}

/// Construct a `FuzzyMeasure` enforcing monotonicity on top of the
/// length + endpoint + range checks already provided by
/// [`FuzzyMeasure::new`]. Phase 2 callers use this wrapper; legacy
/// callers that cannot guarantee monotonicity still use the more
/// permissive base constructor.
pub fn new_monotone(n: u8, values: Vec<f64>) -> Result<FuzzyMeasure> {
    let m = FuzzyMeasure::new(n, values)?;
    check_monotone(&m)?;
    Ok(m)
}

/// Möbius transform: `m(A) = Σ_{B⊆A} (-1)^(|A|-|B|) μ(B)`.
///
/// Returns `Vec<f64>` indexed by the same bitmask scheme as
/// `FuzzyMeasure::values`. Used for Shapley-value computation and
/// interaction-index analysis; not on the hot Choquet path.
pub fn mobius_from_measure(measure: &FuzzyMeasure) -> Vec<f64> {
    let n = measure.n as usize;
    let size = 1usize << n;
    let mut m = vec![0.0_f64; size];
    // `a` IS the subset bitmask we're computing for — not a loop variable
    // that happens to index `m`. Clippy's `needless_range_loop` lint
    // doesn't understand bitmask enumeration.
    #[allow(clippy::needless_range_loop)]
    for a in 0..size {
        let a_card = (a as u64).count_ones() as i32;
        // Enumerate subsets of a via the standard "b = (b - 1) & a" trick.
        let mut sum = 0.0_f64;
        // Start from b = a and descend; include b = 0.
        let mut b = a;
        loop {
            let b_card = (b as u64).count_ones() as i32;
            let parity = if (a_card - b_card) % 2 == 0 { 1.0 } else { -1.0 };
            sum += parity * measure.values[b];
            if b == 0 {
                break;
            }
            b = (b - 1) & a;
        }
        m[a] = sum;
    }
    m
}

/// Inverse Möbius transform: `μ(A) = Σ_{B⊆A} m(B)`.
///
/// Used by round-trip tests (`measure → m → measure`) and by downstream
/// Shapley-value machinery that edits the Möbius representation.
pub fn measure_from_mobius(n: u8, m: &[f64]) -> Result<FuzzyMeasure> {
    let n_us = n as usize;
    let size = 1usize << n_us;
    if m.len() != size {
        return Err(TensaError::InvalidInput(format!(
            "Möbius vector must have 2^n = {} entries, got {}",
            size,
            m.len()
        )));
    }
    let mut values = vec![0.0_f64; size];
    // `a` IS the subset bitmask, not a throwaway loop counter.
    #[allow(clippy::needless_range_loop)]
    for a in 0..size {
        let mut sum = 0.0_f64;
        let mut b = a;
        loop {
            sum += m[b];
            if b == 0 {
                break;
            }
            b = (b - 1) & a;
        }
        values[a] = sum;
    }
    FuzzyMeasure::new(n, values)
}

// ── Built-in symmetric measures ──────────────────────────────────────────────

/// Additive symmetric measure: `μ(S) = |S| / n`. Under Choquet this
/// recovers the arithmetic mean exactly, serving as the regression
/// target for both the Choquet engine and the "default aggregator =
/// Mean" fallback.
pub fn symmetric_additive(n: u8) -> Result<FuzzyMeasure> {
    if n == 0 {
        return Err(TensaError::InvalidInput(
            "symmetric_additive requires n ≥ 1".into(),
        ));
    }
    let size = 1usize << (n as usize);
    let denom = n as f64;
    let values: Vec<f64> = (0..size)
        .map(|s| (s as u64).count_ones() as f64 / denom)
        .collect();
    FuzzyMeasure::new(n, values)
}

/// Pessimistic symmetric measure: `μ(S) = 1` iff `|S| = n`, else `0`.
/// Under Choquet this recovers `min(xs)` — every input must be present
/// before any mass accrues.
pub fn symmetric_pessimistic(n: u8) -> Result<FuzzyMeasure> {
    if n == 0 {
        return Err(TensaError::InvalidInput(
            "symmetric_pessimistic requires n ≥ 1".into(),
        ));
    }
    let size = 1usize << (n as usize);
    let mut values = vec![0.0_f64; size];
    values[size - 1] = 1.0;
    FuzzyMeasure::new(n, values)
}

/// Optimistic symmetric measure: `μ(S) = 1` iff `|S| ≥ 1`, else `0`
/// (μ(∅) = 0). Under Choquet this recovers `max(xs)` — a single input
/// suffices to saturate the aggregate.
pub fn symmetric_optimistic(n: u8) -> Result<FuzzyMeasure> {
    if n == 0 {
        return Err(TensaError::InvalidInput(
            "symmetric_optimistic requires n ≥ 1".into(),
        ));
    }
    let size = 1usize << (n as usize);
    let mut values = vec![1.0_f64; size];
    values[0] = 0.0;
    FuzzyMeasure::new(n, values)
}
