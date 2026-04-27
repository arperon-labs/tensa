//! Per-metric statistics for [`super::fidelity`] (Phase 2.5).
//!
//! Seven metrics, three statistic families:
//! - **KS divergence** (group_size, hyperdegree, inter_event) — two-sample
//!   `D = max |F_a(x) - F_b(x)|` over the empirical CDFs. Lower is better.
//! - **Spearman rank correlation** (activity, order_propensity) — Pearson
//!   correlation of ranks. Mean-rank tie-breaking. Higher is better.
//! - **Mean absolute error** (burstiness, memory_autocorr) — `mean(|a_i - b_i|)`
//!   over per-entity scalars. Lower is better.
//!
//! All inputs flow through [`SourceStats`] / [`SyntheticStats`] in
//! `fidelity.rs` so this module never touches the `Hypergraph`.
//!
//! Hand-rolled math (no `statrs`/`statistical` crate dependency) — every
//! function below is one of: a single sweep through a sorted vec, a rank
//! computation, or a scalar accumulation. ~20-line functions.

// ── Two-sample Kolmogorov-Smirnov ────────────────────────────────────────────

/// Two-sample KS statistic on raw f64 samples.
///
/// `D = max_x |F_a(x) - F_b(x)|`. Both inputs are sorted in place via clones;
/// the typical caller (per-entity inter-event time vector, group sizes) holds
/// only a few hundred to tens of thousands of values, so the sort cost is
/// negligible relative to the K=20 generation work that produced them.
///
/// Empty inputs:
/// - both empty → 0.0 (vacuously identical).
/// - one empty → 1.0 (maximally divergent — everything in the non-empty set is
///   above 100 % of the empty set's CDF).
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let mut a_s = a.to_vec();
    let mut b_s = b.to_vec();
    a_s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let n_a = a_s.len() as f64;
    let n_b = b_s.len() as f64;

    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max = 0.0_f64;
    while i < a_s.len() && j < b_s.len() {
        let va = a_s[i];
        let vb = b_s[j];
        if va <= vb {
            i += 1;
        }
        if vb <= va {
            j += 1;
        }
        let cdf_a = i as f64 / n_a;
        let cdf_b = j as f64 / n_b;
        let d = (cdf_a - cdf_b).abs();
        if d > d_max {
            d_max = d;
        }
    }
    d_max
}

// ── Spearman rank correlation ────────────────────────────────────────────────

/// Spearman rank correlation between two equal-length scalar vectors.
///
/// Computes Pearson correlation on mean-rank-tie-broken ranks. Returns 0.0
/// when either input has zero variance (degenerate input — no correlation
/// signal possible). Returns 0.0 when the vectors have different lengths
/// (caller error; we don't panic here because fidelity reports must always
/// produce a finite number).
///
/// Output range: `[-1.0, 1.0]`.
pub fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let ra = mean_ranks(a);
    let rb = mean_ranks(b);
    pearson(&ra, &rb)
}

/// Mean-rank tie breaking. `[3.0, 1.0, 1.0, 5.0]` → `[3.0, 1.5, 1.5, 4.0]`.
fn mean_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < f64::EPSILON {
            j += 1;
        }
        // `i..j` is a tie group. Assign mean rank `(i + j - 1) / 2 + 1` (1-based).
        let mean_rank = ((i + j - 1) as f64) / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = mean_rank;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation. Hand-rolled (~10 lines) so we don't pull in `statrs`.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den_a = 0.0_f64;
    let mut den_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }
    let den = (den_a * den_b).sqrt();
    if den <= f64::EPSILON {
        return 0.0;
    }
    num / den
}

// ── Mean absolute error ──────────────────────────────────────────────────────

/// `MAE(a, b) = mean(|a_i - b_i|)`. Returns 0.0 when both empty;
/// returns 1.0 when lengths differ (treated as maximally divergent — keeps
/// the report finite without panicking).
pub fn mae(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 1.0;
    }
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / n
}

// ── Per-entity burstiness coefficient ────────────────────────────────────────

/// Burstiness coefficient `B = (σ - μ) / (σ + μ)` over inter-event times.
///
/// Goh & Barabási (2008). Range `[-1.0, 1.0]`:
/// - `+1` ⇒ infinitely bursty (single huge gap).
/// - ` 0` ⇒ Poisson process (σ = μ).
/// - `-1` ⇒ perfectly periodic (σ = 0).
///
/// Returns `0.0` for fewer than two inter-event times (no variance signal) and
/// for `μ + σ < ε` (degenerate — entity never participated, all times zero).
pub fn burstiness(inter_event_times: &[f64]) -> f64 {
    if inter_event_times.len() < 2 {
        return 0.0;
    }
    let n = inter_event_times.len() as f64;
    let mean: f64 = inter_event_times.iter().sum::<f64>() / n;
    let variance: f64 = inter_event_times
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / n;
    let sigma = variance.sqrt();
    let denom = sigma + mean;
    if denom <= f64::EPSILON {
        return 0.0;
    }
    (sigma - mean) / denom
}

// ── Lag-1 autocorrelation ────────────────────────────────────────────────────

/// Lag-1 autocorrelation of a binary series (1 = active at tick t, 0 = idle).
///
/// `r_1 = mean[(x_t - μ)(x_{t+1} - μ)] / variance(x)`. Returns 0.0 for a
/// constant series (no variance) or fewer than 2 samples.
pub fn lag1_autocorr(series: &[f64]) -> f64 {
    if series.len() < 2 {
        return 0.0;
    }
    let n = series.len() as f64;
    let mean: f64 = series.iter().sum::<f64>() / n;
    let variance: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    if variance <= f64::EPSILON {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for w in series.windows(2) {
        acc += (w[0] - mean) * (w[1] - mean);
    }
    let cov = acc / (n - 1.0);
    cov / variance
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ks_identical_distributions_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = a.clone();
        assert!(ks_two_sample(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn ks_disjoint_distributions_is_one() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![10.0, 10.0, 10.0];
        assert!((ks_two_sample(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ks_empty_one_side_is_one() {
        assert!((ks_two_sample(&[1.0], &[]) - 1.0).abs() < 1e-9);
        assert!((ks_two_sample(&[], &[1.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ks_both_empty_is_zero() {
        assert!(ks_two_sample(&[], &[]).abs() < 1e-9);
    }

    #[test]
    fn spearman_perfect_monotone_is_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_rho(&a, &b);
        assert!((rho - 1.0).abs() < 1e-9, "expected rho=1, got {rho}");
    }

    #[test]
    fn spearman_perfect_anti_monotone_is_minus_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let rho = spearman_rho(&a, &b);
        assert!((rho + 1.0).abs() < 1e-9, "expected rho=-1, got {rho}");
    }

    #[test]
    fn spearman_handles_ties_via_mean_rank() {
        // Both vectors have the same tie pattern → rho should still be 1.0.
        let a = vec![1.0, 1.0, 2.0, 3.0];
        let b = vec![10.0, 10.0, 20.0, 30.0];
        let rho = spearman_rho(&a, &b);
        assert!((rho - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spearman_zero_variance_returns_zero() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(spearman_rho(&a, &b), 0.0);
    }

    #[test]
    fn mae_zero_for_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(mae(&a, &a), 0.0);
    }

    #[test]
    fn mae_unit_diff_is_one() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        assert!((mae(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn burstiness_periodic_is_minus_one() {
        // Constant inter-event times ⇒ σ = 0 ⇒ B = -1.
        let times = vec![5.0, 5.0, 5.0, 5.0];
        assert!((burstiness(&times) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn burstiness_too_few_samples_is_zero() {
        assert_eq!(burstiness(&[]), 0.0);
        assert_eq!(burstiness(&[1.0]), 0.0);
    }

    #[test]
    fn lag1_autocorr_constant_is_zero() {
        let s = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(lag1_autocorr(&s), 0.0);
    }

    #[test]
    fn lag1_autocorr_perfect_alternation_is_negative() {
        let s = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let r = lag1_autocorr(&s);
        assert!(r < 0.0, "alternating series should have negative lag-1 autocorr, got {r}");
    }
}
