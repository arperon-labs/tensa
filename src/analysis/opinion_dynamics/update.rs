//! Per-edge update kernels for the two MVP variants.
//!
//! Both kernels are pure: they mutate `x` in place but never touch the
//! hypergraph, KV store, or RNG. The convergence-rate / confidence-bound /
//! size-scaling parameters all come from [`super::types::OpinionDynamicsParams`].
//!
//! The strict inequality `|x_i - x_j| < c_e` is *load-bearing* (Hickok 2022
//! §2.1). Switching to `<=` would change boundary-case dynamics — see the
//! worked example in `docs/opinion_dynamics_algorithm.md` §15.

use super::types::{ConfidenceScaling, OpinionDynamicsParams};

/// Compute the effective confidence bound `c_e` for a hyperedge of `edge_size`.
///
/// `Flat` (default when `confidence_size_scaling == None`) returns `c`
/// unchanged. The inverse variants reduce `c_e` for larger edges. For
/// `edge_size <= 1`, all variants degenerate to `params.confidence_bound`
/// (the caller has already filtered size-1 edges out — this is a safety
/// fallback only).
#[inline]
pub fn effective_c(edge_size: usize, params: &OpinionDynamicsParams) -> f32 {
    match params.confidence_size_scaling {
        None | Some(ConfidenceScaling::Flat) => params.confidence_bound,
        Some(ConfidenceScaling::InverseSqrtSize) => {
            let denom = (edge_size.saturating_sub(1) as f32).sqrt().max(1.0);
            params.confidence_bound / denom
        }
        Some(ConfidenceScaling::InverseSize) => {
            let denom = edge_size.saturating_sub(1).max(1) as f32;
            params.confidence_bound / denom
        }
    }
}

/// Hickok 2022 PairwiseWithin update.
///
/// For every ordered pair `(i, j)` with `i < j` in `edge` (sorted-UUID
/// canonical order), if `|x[j] - x[i]| < c_e`, both move toward each other
/// by `μ · (x[j] - x[i])`. Updates are *immediately visible* to subsequent
/// pairs in the same edge (Gauss-Seidel) — this is the mechanism behind the
/// opinion-jumping phenomenon (Hickok §4).
pub fn pairwise_within_update(edge: &[usize], x: &mut [f32], params: &OpinionDynamicsParams) {
    if edge.len() < 2 {
        return;
    }
    let c_e = effective_c(edge.len(), params);
    let mu = params.convergence_rate;
    for i_idx in 0..edge.len() {
        for j_idx in (i_idx + 1)..edge.len() {
            let i = edge[i_idx];
            let j = edge[j_idx];
            let diff = x[j] - x[i];
            // strict inequality per Hickok 2022 §2.1
            if diff.abs() < c_e {
                let delta = mu * diff;
                x[i] += delta;
                x[j] -= delta;
            }
        }
    }
}

/// Schawe & Hernández 2022 GroupMean update.
///
/// All-or-nothing: when `max(x[edge]) - min(x[edge]) < c_e`, every member
/// moves toward the group mean by `μ`. Otherwise, no update.
pub fn group_mean_update(edge: &[usize], x: &mut [f32], params: &OpinionDynamicsParams) {
    if edge.len() < 2 {
        return;
    }
    let c_e = effective_c(edge.len(), params);
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut sum = 0.0_f32;
    for &i in edge {
        let v = x[i];
        if v < min_x {
            min_x = v;
        }
        if v > max_x {
            max_x = v;
        }
        sum += v;
    }
    let spread = max_x - min_x;
    // strict inequality per Hickok 2022 §2.1 / Schawe-Hernández compatibility
    if spread < c_e {
        let mean = sum / edge.len() as f32;
        let mu = params.convergence_rate;
        for &i in edge {
            x[i] += mu * (mean - x[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::opinion_dynamics::types::*;

    fn p(c: f32, mu: f32) -> OpinionDynamicsParams {
        OpinionDynamicsParams {
            confidence_bound: c,
            convergence_rate: mu,
            ..Default::default()
        }
    }

    #[test]
    fn test_effective_c_flat_default() {
        let mut params = p(0.4, 0.5);
        params.confidence_size_scaling = None;
        assert_eq!(effective_c(2, &params), 0.4);
        assert_eq!(effective_c(5, &params), 0.4);
        params.confidence_size_scaling = Some(ConfidenceScaling::Flat);
        assert_eq!(effective_c(5, &params), 0.4);
    }

    #[test]
    fn test_effective_c_inverse_sqrt() {
        let mut params = p(0.4, 0.5);
        params.confidence_size_scaling = Some(ConfidenceScaling::InverseSqrtSize);
        assert_eq!(effective_c(2, &params), 0.4); // sqrt(1) = 1
        let three = effective_c(3, &params);
        assert!((three - 0.4 / (2.0_f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_effective_c_inverse_linear() {
        let mut params = p(0.6, 0.5);
        params.confidence_size_scaling = Some(ConfidenceScaling::InverseSize);
        assert!((effective_c(2, &params) - 0.6).abs() < 1e-6); // /1
        assert!((effective_c(4, &params) - 0.2).abs() < 1e-6); // /3
    }

    #[test]
    fn test_pairwise_no_op_for_size_1() {
        let mut x = vec![0.5];
        pairwise_within_update(&[0], &mut x, &p(0.3, 0.5));
        assert_eq!(x, vec![0.5]);
    }

    #[test]
    fn test_pairwise_dyadic_meets_at_midpoint_full_mu() {
        let mut x = vec![0.10, 0.30];
        pairwise_within_update(&[0, 1], &mut x, &p(0.5, 0.5));
        // Δ = 0.5 * 0.20 = 0.10 → both meet at 0.20
        assert!((x[0] - 0.20).abs() < 1e-6);
        assert!((x[1] - 0.20).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_skips_when_diff_above_c() {
        let mut x = vec![0.10, 0.50];
        pairwise_within_update(&[0, 1], &mut x, &p(0.3, 0.5));
        // diff 0.40 > c 0.30 → no update
        assert_eq!(x, vec![0.10, 0.50]);
    }

    #[test]
    fn test_pairwise_strict_inequality_at_boundary() {
        let mut x = vec![0.10, 0.40]; // diff = 0.30
        pairwise_within_update(&[0, 1], &mut x, &p(0.3, 0.5));
        // strict `<` => no update at exactly c
        assert_eq!(x, vec![0.10, 0.40]);
    }

    #[test]
    fn test_pairwise_canonical_order_gauss_seidel() {
        // A=0.05, B=0.45, C=0.55, c=0.5, mu=0.5.
        let mut x = vec![0.05_f32, 0.45, 0.55];
        pairwise_within_update(&[0, 1, 2], &mut x, &p(0.5, 0.5));
        // Pair(A,B): diff 0.40 < 0.5; delta = 0.20; A=0.25, B=0.25.
        // Pair(A,C): diff 0.55-0.25 = 0.30 < 0.50; delta = 0.15; A=0.40, C=0.40.
        // Pair(B,C): diff 0.40-0.25 = 0.15 < 0.50; delta = 0.075;
        //   B=0.25+0.075=0.325, C=0.40-0.075=0.325.
        assert!((x[0] - 0.40).abs() < 1e-6, "A={}", x[0]);
        assert!((x[1] - 0.325).abs() < 1e-6, "B={}", x[1]);
        assert!((x[2] - 0.325).abs() < 1e-6, "C={}", x[2]);
    }

    #[test]
    fn test_group_mean_no_op_when_spread_exceeds_c() {
        let mut x = vec![0.10, 0.50, 0.90];
        group_mean_update(&[0, 1, 2], &mut x, &p(0.3, 0.5));
        // spread = 0.80 > c = 0.30 → no update
        assert_eq!(x, vec![0.10, 0.50, 0.90]);
    }

    #[test]
    fn test_group_mean_pulls_toward_mean_within_c() {
        let mut x = vec![0.40, 0.50, 0.60];
        group_mean_update(&[0, 1, 2], &mut x, &p(0.3, 0.5));
        // spread = 0.20 < c = 0.30; mean = 0.50; mu = 0.5
        // x[0] += 0.5*(0.50-0.40) = 0.45
        // x[1] += 0.5*(0.50-0.50) = 0.50
        // x[2] += 0.5*(0.50-0.60) = 0.55
        assert!((x[0] - 0.45).abs() < 1e-6);
        assert!((x[1] - 0.50).abs() < 1e-6);
        assert!((x[2] - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_group_mean_full_convergence_at_mu_one() {
        let mut x = vec![0.40, 0.50, 0.60];
        group_mean_update(&[0, 1, 2], &mut x, &p(0.3, 1.0));
        // mu = 1.0 → all collapse to mean 0.50 in one step
        for &v in &x {
            assert!((v - 0.50).abs() < 1e-6);
        }
    }

    #[test]
    fn test_group_mean_strict_inequality_at_boundary() {
        let mut x = vec![0.20, 0.50]; // spread = 0.30
        group_mean_update(&[0, 1], &mut x, &p(0.3, 0.5));
        assert_eq!(x, vec![0.20, 0.50]);
    }
}
