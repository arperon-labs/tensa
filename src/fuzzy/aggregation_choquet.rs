//! Choquet integral over a fuzzy measure.
//!
//! For an input vector `x = (x_1, .., x_n)` and fuzzy measure `μ` on
//! `N = {0, .., n-1}`, the Choquet integral is
//!
//! ```text
//!   C_μ(x) = Σ_{i=1}^{n} (x_(i) - x_(i-1)) · μ(A_i)
//! ```
//!
//! where `x_(·)` is `x` sorted in ascending order, `x_(0) := 0`, and
//! `A_i = {j : x_j ≥ x_(i)}`. The exact path is `O(n · 2^n)` indexing
//! against `FuzzyMeasure::values`; for `n > 10` a Monte-Carlo estimator
//! over `k` random permutations returns a mean and standard-error so
//! callers can choose their confidence / latency tradeoff.
//!
//! Special cases verified in tests:
//! * `symmetric_additive ⇒ C_μ = arithmetic mean`.
//! * `symmetric_pessimistic ⇒ C_μ = min(x)`.
//! * `symmetric_optimistic ⇒ C_μ = max(x)`.
//!
//! Cites: [grabisch1996choquet] [bustince2016choquet] [grabisch2000fuzzymeasure].

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::FuzzyMeasure;

/// Exact-path cap. Above this we switch to Monte-Carlo — a 2^11 = 2048
/// lookup table per `FuzzyMeasure` is already at the ceiling of what
/// `FuzzyMeasure::new` accepts, but the sorted-subset reduction still
/// costs `O(n · 2^n)` per evaluation, which is what we're actually
/// bounding.
pub const EXACT_N_CAP: usize = 10;

/// Default number of Monte-Carlo permutations when the caller does not
/// specify one.
pub const DEFAULT_MC_PERMUTATIONS: usize = 1000;

/// Result of a Choquet evaluation. The exact path reports
/// `std_err = None`; the Monte-Carlo path reports the sample standard
/// error of the permutation estimator.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoquetResult {
    pub value: f64,
    pub std_err: Option<f64>,
}

/// Compute the exact Choquet integral of `xs` under `measure`.
///
/// Requires `measure.n as usize == xs.len()` and `xs.len() ≤ EXACT_N_CAP`.
/// Callers with `n > 10` should use [`choquet_monte_carlo`].
pub fn choquet_exact(xs: &[f64], measure: &FuzzyMeasure) -> Result<f64> {
    let n = xs.len();
    if n != measure.n as usize {
        return Err(TensaError::InvalidInput(format!(
            "Choquet: |xs|={} must equal measure.n={}",
            n, measure.n
        )));
    }
    if n == 0 {
        return Err(TensaError::InvalidInput("Choquet: empty input".into()));
    }
    if n > EXACT_N_CAP {
        return Err(TensaError::InvalidInput(format!(
            "Choquet: exact path capped at n ≤ {}, got {} — use choquet_monte_carlo",
            EXACT_N_CAP, n
        )));
    }

    // Sort indices by ascending xs, stable on ties.
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result = 0.0_f64;
    let mut prev = 0.0_f64;
    for rank in 0..n {
        let cur = xs[sorted[rank]];
        // A_rank = indices of xs with value ≥ cur, i.e. sorted[rank..].
        // Build the subset bitmask in terms of ORIGINAL indices.
        let mut mask = 0usize;
        for &j in &sorted[rank..] {
            mask |= 1 << j;
        }
        let mu = measure.values[mask];
        result += (cur - prev) * mu;
        prev = cur;
    }
    Ok(result.clamp(0.0, 1.0))
}

/// Monte-Carlo approximation of the Choquet integral via `k` random
/// permutations. Each permutation contributes one marginal-chain
/// estimate; the returned value is the sample mean, `std_err` the
/// sample standard error over the `k` estimates.
///
/// The estimator follows Grabisch & Labreuche 2010 §4.2: for a random
/// permutation `π`, the chain `∅ ⊂ {π_1} ⊂ {π_1, π_2} ⊂ ...` gives a
/// telescoping decomposition whose expectation equals `C_μ(x)` as long
/// as `μ` is a capacity. When `π` is uniform over `S_n` the estimator
/// is unbiased.
pub fn choquet_monte_carlo(
    xs: &[f64],
    measure: &FuzzyMeasure,
    k: usize,
    seed: u64,
) -> Result<ChoquetResult> {
    let n = xs.len();
    if n != measure.n as usize {
        return Err(TensaError::InvalidInput(format!(
            "Choquet: |xs|={} must equal measure.n={}",
            n, measure.n
        )));
    }
    if n == 0 {
        return Err(TensaError::InvalidInput("Choquet: empty input".into()));
    }
    let k = if k == 0 { DEFAULT_MC_PERMUTATIONS } else { k };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut samples: Vec<f64> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        // Marginal chain: μ(S_i) - μ(S_{i-1}) weighted by x_{π_i}.
        let mut prev_mask = 0usize;
        let mut prev_mu = measure.values[prev_mask];
        let mut acc = 0.0_f64;
        for &p in &perm {
            let cur_mask = prev_mask | (1 << p);
            let cur_mu = measure.values[cur_mask];
            acc += xs[p] * (cur_mu - prev_mu);
            prev_mask = cur_mask;
            prev_mu = cur_mu;
        }
        samples.push(acc.clamp(0.0, 1.0));
    }

    let mean: f64 = samples.iter().sum::<f64>() / k as f64;
    let var: f64 = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
        / (k.max(1) as f64);
    let std_err = if k > 1 {
        Some((var / k as f64).sqrt())
    } else {
        Some(0.0)
    };
    Ok(ChoquetResult {
        value: mean,
        std_err,
    })
}

/// Dispatch: exact path for `n ≤ EXACT_N_CAP`, MC fallback otherwise.
/// `seed` is only used on the MC path; exact callers can pass `0`.
pub fn choquet(xs: &[f64], measure: &FuzzyMeasure, seed: u64) -> Result<ChoquetResult> {
    if xs.len() <= EXACT_N_CAP {
        Ok(ChoquetResult {
            value: choquet_exact(xs, measure)?,
            std_err: None,
        })
    } else {
        choquet_monte_carlo(xs, measure, DEFAULT_MC_PERMUTATIONS, seed)
    }
}
