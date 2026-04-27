//! Per-layer distance metrics for the Narrative Fingerprint.
//!
//! The fingerprint is a heterogeneous vector — function-word frequencies (Burrows-style),
//! probability distributions (situation density, Allen relations, game types),
//! scalar features (Gini, entropy, chain lengths), and a 256-bit WL graph SimHash.
//! Plain cosine over the concatenation is suboptimal because each component has
//! different distributional assumptions. This module provides kernels tailored to
//! each component type:
//!
//! - `burrows_cosine` — cosine on z-scored function-word frequencies (Würzburg Delta)
//! - `jensen_shannon_sim` — symmetric, bounded similarity derived from JS divergence
//!   for probability distributions
//! - `mahalanobis_sim_diag` — diagonal Mahalanobis on scalars with corpus stds
//! - `hamming_sim_u256` — Hamming-distance similarity on 256-bit SimHash signatures
//!
//! All similarity functions return values in `[0, 1]` where 1 = identical and 0 = maximally dissimilar.

use crate::analysis::stylometry::{cosine_delta, CorpusStats, ProseStyleFeatures};

/// Minimum norm below which a vector is considered zero (avoids NaN in ratios).
const EPSILON: f32 = 1e-12;

/// Burrows-Cosine similarity on the function-word layer.
///
/// Wraps `cosine_delta` (which returns a distance in `[0, 2]` on z-scored frequencies)
/// into a bounded similarity `max(0, 1 - delta / 2)`.
///
/// This is the recommended prose-layer kernel: it uses the Würzburg Delta formulation
/// that research has shown outperforms Burrows' Delta on most authorship attribution
/// tasks, and it honors corpus-level z-score normalization for Mahalanobis-compatible
/// geometry.
pub fn burrows_cosine(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &CorpusStats,
) -> f32 {
    let delta = cosine_delta(a, b, corpus_stats);
    (1.0 - delta / 2.0).clamp(0.0, 1.0)
}

/// Normalize a non-negative vector to a probability distribution.
/// Zero-sum vectors are returned unchanged (the caller should treat both sides as zero).
fn normalize_probs(v: &[f32]) -> Vec<f32> {
    let total: f32 = v.iter().map(|x| x.max(0.0)).sum();
    if total <= EPSILON {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x.max(0.0) / total).collect()
}

/// Jensen–Shannon divergence (base 2) between two probability distributions.
///
/// JS is symmetric and bounded by `log2(2) = 1.0` in base-2 units. The function
/// accepts vectors that may not be normalized and normalizes them defensively.
/// Vectors of unequal length are truncated to the shorter.
///
/// Returns 0.0 when either vector sums to (near-)zero, treating missing data as
/// "indistinguishable" rather than "maximally different".
pub fn jensen_shannon(p: &[f32], q: &[f32]) -> f32 {
    let len = p.len().min(q.len());
    if len == 0 {
        return 0.0;
    }
    let p = normalize_probs(&p[..len]);
    let q = normalize_probs(&q[..len]);
    let p_sum: f32 = p.iter().sum();
    let q_sum: f32 = q.iter().sum();
    if p_sum < EPSILON || q_sum < EPSILON {
        // Both empty → identical-by-default; one empty → no signal → 0 divergence.
        return 0.0;
    }

    let mut js = 0.0_f32;
    for i in 0..len {
        let m = 0.5 * (p[i] + q[i]);
        if m <= 0.0 {
            continue;
        }
        if p[i] > 0.0 {
            js += 0.5 * p[i] * (p[i] / m).log2();
        }
        if q[i] > 0.0 {
            js += 0.5 * q[i] * (q[i] / m).log2();
        }
    }
    js.clamp(0.0, 1.0)
}

/// Bounded similarity derived from Jensen–Shannon divergence.
///
/// Uses the square root of JS (Jensen–Shannon distance) which is a proper metric,
/// and maps it into `[0, 1]` as `1 - sqrt(JS)`.
pub fn jensen_shannon_sim(p: &[f32], q: &[f32]) -> f32 {
    let js = jensen_shannon(p, q);
    (1.0 - js.sqrt()).clamp(0.0, 1.0)
}

/// Diagonal Mahalanobis-distance similarity over scalar feature vectors.
///
/// Computes `d^2 = Σ ((a_i - b_i) / σ_i)^2` over the provided corpus standard
/// deviations, then maps through `exp(-d^2 / dim)` to produce a bounded similarity.
///
/// - Slots where `σ_i ≤ EPSILON` are skipped (the feature has no corpus variability
///   and cannot discriminate).
/// - If `stds` is shorter than `a` / `b`, missing slots default to `1.0` (unit
///   variance — equivalent to Euclidean).
/// - Returns `1.0` when every slot is skipped (no discriminative features).
pub fn mahalanobis_sim_diag(a: &[f32], b: &[f32], stds: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 1.0;
    }
    let mut d2 = 0.0_f32;
    let mut used = 0usize;
    for i in 0..len {
        let sigma = if i < stds.len() { stds[i] } else { 1.0 };
        if sigma <= EPSILON {
            continue;
        }
        let diff = (a[i] - b[i]) / sigma;
        d2 += diff * diff;
        used += 1;
    }
    if used == 0 {
        return 1.0;
    }
    (-d2 / used as f32).exp().clamp(0.0, 1.0)
}

/// Hamming-distance similarity on 256-bit SimHash signatures.
///
/// Returns `1.0 - popcount(a XOR b) / 256`. Identical signatures → 1.0,
/// fully-flipped → 0.0.
pub fn hamming_sim_u256(a: [u64; 4], b: [u64; 4]) -> f32 {
    let mut bits = 0u32;
    for i in 0..4 {
        bits += (a[i] ^ b[i]).count_ones();
    }
    1.0 - (bits as f32 / 256.0)
}

#[cfg(test)]
#[path = "similarity_metrics_tests.rs"]
mod tests;
