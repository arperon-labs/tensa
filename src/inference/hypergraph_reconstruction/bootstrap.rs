//! Bootstrap confidence scoring + AUROC utility.
//!
//! For each of `K` resamples, we draw `T'` rows of `(Θ, X_dot)` with
//! replacement, re-solve the LASSO with the same λ, extract edges, and
//! tally retention frequency per canonical edge identity.
//!
//! The participation-rate state matrix is built ONCE upstream
//! (in [`super::reconstruct`]); this routine reuses the precomputed library
//! and derivatives, so each bootstrap iteration is dominated by the LASSO
//! solve and column-norm recomputation.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::error::Result;

use super::lasso::solve_lasso;
use super::library::Library;
use super::reconstruct::{canonical_key, extract_edges};
use super::symmetrize::symmetrize_xi;
use super::types::ReconstructionParams;

/// Run K bootstrap resamples and return per-edge retention frequency.
///
/// `entity_uuids` maps column indices to UUIDs (matches the layout used by
/// `extract_edges`). Returns a `HashMap<canonical_key, frequency_in_[0,1]>`.
pub fn run_bootstrap(
    x_trimmed: &[Vec<f32>],
    x_dot: &[Vec<f32>],
    library: &Library,
    n: usize,
    params: &ReconstructionParams,
    lambda_used: f32,
    entity_uuids: &[Uuid],
) -> Result<HashMap<Vec<u8>, f32>> {
    let k = params.bootstrap_k;
    if k == 0 {
        return Ok(HashMap::new());
    }
    let t = x_trimmed.len();
    if t < 4 {
        return Ok(HashMap::new());
    }

    let mut tally: HashMap<Vec<u8>, usize> = HashMap::new();

    for sample in 0..k {
        let seed = params
            .bootstrap_seed
            .wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(sample as u64 + 1));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let resampled_indices: Vec<usize> = (0..t).map(|_| rng.gen_range(0..t)).collect();

        // Materialize resampled (Θ, X_dot) and recompute column norms.
        let theta_resampled: Vec<Vec<f32>> = resampled_indices
            .iter()
            .map(|&idx| library.theta[idx].clone())
            .collect();
        let x_dot_resampled: Vec<Vec<f32>> = resampled_indices
            .iter()
            .map(|&idx| x_dot[idx].clone())
            .collect();
        let column_norm_sq = compute_column_norms_sq(&theta_resampled, library.terms.len());

        // Per-entity LASSO solve.
        let mut xi_matrix = vec![vec![0.0_f32; library.terms.len()]; n];
        for i in 0..n {
            let y: Vec<f32> = x_dot_resampled.iter().map(|row| row[i]).collect();
            xi_matrix[i] = solve_lasso(&theta_resampled, &y, &column_norm_sq, lambda_used);
        }

        // Symmetrize and extract — same threshold as the main pipeline.
        let symmetrized = symmetrize_xi(&xi_matrix, &library.terms, params.symmetrize, n);
        let edges = extract_edges(&symmetrized, &library.terms, entity_uuids, lambda_used);

        for edge in edges {
            let key = canonical_key(&edge);
            *tally.entry(key).or_insert(0) += 1;
        }
    }

    let inv_k = 1.0_f32 / k as f32;
    Ok(tally.into_iter().map(|(k, v)| (k, v as f32 * inv_k)).collect())
}

fn compute_column_norms_sq(theta: &[Vec<f32>], l: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; l];
    for row in theta {
        for (col, &val) in row.iter().enumerate() {
            out[col] += val * val;
        }
    }
    out
}

/// Compute AUROC from sorted `(score, is_true_edge)` pairs via the
/// trapezoidal rule on the ROC curve.
///
/// `n_positives` and `n_negatives` are the totals across the input list
/// (NOT recounted from the slice — caller knows them precisely from the
/// ground truth set). Returns 0.5 (random) when either is zero.
pub fn compute_auroc(
    scored_edges: &mut [(f32, bool)],
    n_positives: usize,
    n_negatives: usize,
) -> f32 {
    if n_positives == 0 || n_negatives == 0 {
        return 0.5;
    }
    // Sort by score descending.
    scored_edges
        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut prev_tpr = 0.0_f32;
    let mut prev_fpr = 0.0_f32;
    let mut auroc = 0.0_f32;
    let mut i = 0usize;
    while i < scored_edges.len() {
        // Group ties so we plot one ROC point per unique score.
        let cur_score = scored_edges[i].0;
        let mut j = i;
        while j < scored_edges.len() && (scored_edges[j].0 - cur_score).abs() <= f32::EPSILON {
            if scored_edges[j].1 {
                tp += 1;
            } else {
                fp += 1;
            }
            j += 1;
        }
        let tpr = tp as f32 / n_positives as f32;
        let fpr = fp as f32 / n_negatives as f32;
        // Trapezoidal area.
        auroc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_tpr = tpr;
        prev_fpr = fpr;
        i = j;
    }
    // Sweep to (1, 1) — any unseen negatives push FPR up to 1; recovered
    // positives push TPR to 1. The final segment connects last point to (1, 1).
    if prev_fpr < 1.0 {
        let tpr_end = 1.0_f32;
        auroc += (1.0 - prev_fpr) * (tpr_end + prev_tpr) * 0.5;
    }
    auroc.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_auroc_perfect_ranking() {
        // 3 positives at top, 3 negatives at bottom.
        let mut pairs = vec![
            (0.9, true),
            (0.8, true),
            (0.7, true),
            (0.4, false),
            (0.3, false),
            (0.2, false),
        ];
        let auroc = compute_auroc(&mut pairs, 3, 3);
        assert!((auroc - 1.0).abs() < 1e-3, "expected 1.0, got {auroc}");
    }

    #[test]
    fn test_compute_auroc_random_is_half() {
        // Interleaved positives and negatives → AUROC ≈ 0.5.
        let mut pairs = vec![
            (0.9, true),
            (0.8, false),
            (0.7, true),
            (0.6, false),
            (0.5, true),
            (0.4, false),
        ];
        let auroc = compute_auroc(&mut pairs, 3, 3);
        assert!(
            (auroc - 0.5).abs() < 0.2,
            "expected ~0.5 for interleaved, got {auroc}"
        );
    }

    #[test]
    fn test_compute_auroc_inverse_ranking_is_zero() {
        let mut pairs = vec![
            (0.9, false),
            (0.8, false),
            (0.7, false),
            (0.4, true),
            (0.3, true),
            (0.2, true),
        ];
        let auroc = compute_auroc(&mut pairs, 3, 3);
        assert!(auroc < 0.1, "expected near 0, got {auroc}");
    }

    #[test]
    fn test_compute_column_norms_sq_basic() {
        let theta = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let norms = compute_column_norms_sq(&theta, 2);
        assert!((norms[0] - 10.0).abs() < 1e-6); // 1 + 9
        assert!((norms[1] - 20.0).abs() < 1e-6); // 4 + 16
    }

}
