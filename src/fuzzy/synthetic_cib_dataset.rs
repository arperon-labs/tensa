//! Synthetic CIB (Coordinated Inauthentic Behaviour) dataset generator.
//!
//! Produces a deterministic ranking-supervised dataset with 4 signals per
//! cluster and a non-additive ground-truth ranking score:
//!
//! ```text
//! s = sigmoid( 2.0 · temporal_correlation · content_overlap
//!              + 0.3 · network_density
//!              - 0.5 · posting_cadence )
//! ```
//!
//! The product term `temporal_correlation × content_overlap` is the
//! load-bearing non-additivity — no additive measure can recover the
//! ranking, only a Choquet capacity with super-additive interaction
//! between signals 0 and 1.
//!
//! Used as the Phase 2 acceptance demo (see
//! [`crate::fuzzy::aggregation_learn`]); also referenced by the
//! `synthetic_cib_demonstration` test, which is allowed to `println!` the
//! resulting AUC numbers (one explicit exception to the no-`println!`
//! rule per design §3.3 of `docs/choquet_learning_algorithm.md`).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Number of signals per cluster — fixed at 4 for the CIB demo.
pub const SYNTHETIC_CIB_N_SIGNALS: usize = 4;

/// Generate `n_clusters` synthetic CIB clusters.
///
/// Each cluster has 4 uniform-`[0, 1]` signals; the returned `rank` is
/// the cluster's position after sorting by ground-truth score descending
/// (rank 0 = most coordinated). Deterministic for a fixed `seed`.
pub fn generate_synthetic_cib(seed: u64, n_clusters: u32) -> Vec<(Vec<f64>, u32)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_clusters as usize;
    let mut signals: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut scores: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        let temporal_correlation: f64 = rng.gen();
        let content_overlap: f64 = rng.gen();
        let network_density: f64 = rng.gen();
        let posting_cadence: f64 = rng.gen();
        let raw = 2.0 * temporal_correlation * content_overlap
            + 0.3 * network_density
            - 0.5 * posting_cadence;
        let s = sigmoid(raw);
        signals.push(vec![
            temporal_correlation,
            content_overlap,
            network_density,
            posting_cadence,
        ]);
        scores.push(s);
    }

    // Sort indices by score descending → rank = position.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    indices
        .into_iter()
        .enumerate()
        .map(|(rank, idx)| (signals[idx].clone(), rank as u32))
        .collect()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_fixed_seed() {
        let a = generate_synthetic_cib(42, 20);
        let b = generate_synthetic_cib(42, 20);
        assert_eq!(a.len(), b.len());
        for ((xa, ra), (xb, rb)) in a.iter().zip(b.iter()) {
            assert_eq!(ra, rb);
            assert_eq!(xa, xb);
        }
    }

    #[test]
    fn returns_requested_count() {
        let d = generate_synthetic_cib(7, 50);
        assert_eq!(d.len(), 50);
        assert_eq!(d[0].0.len(), SYNTHETIC_CIB_N_SIGNALS);
    }

    #[test]
    fn ranks_are_dense_and_unique() {
        let d = generate_synthetic_cib(1, 30);
        let mut ranks: Vec<u32> = d.iter().map(|(_, r)| *r).collect();
        ranks.sort_unstable();
        for (i, r) in ranks.iter().enumerate() {
            assert_eq!(*r as usize, i, "ranks must be 0..n_clusters with no gaps");
        }
    }
}
