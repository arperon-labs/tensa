//! Coordinate-descent LASSO solver — handwritten, no external deps.
//!
//! Solves `argmin_ξ ||Θ ξ - y||²₂ + λ ||ξ||₁` per entity, with column
//! normalization computed up-front so the inner loop is a tight dot product.
//!
//! References:
//! - Tibshirani, *Regression Shrinkage and Selection via the Lasso*, J. R.
//!   Stat. Soc. B **58**, 267 (1996).
//! - Friedman, Hastie, Höfling, Tibshirani, *Pathwise coordinate
//!   optimization*, Ann. Appl. Stat. **1**, 302 (2007).

use crate::error::{Result, TensaError};

use super::library::Library;

/// Maximum coordinate-descent passes before bailout.
const LASSO_MAX_ITERS: usize = 1_000;
/// Convergence tolerance on the maximum coordinate update per pass.
const LASSO_TOLERANCE: f32 = 1e-5;
/// Auto-lambda heuristic: 10% of `λ_max = max(|Θᵀ y|) / T'`.
const AUTO_LAMBDA_FRACTION: f32 = 0.1;

/// Soft-threshold operator for LASSO updates.
#[inline]
fn soft_threshold(z: f32, lambda: f32) -> f32 {
    if z > lambda {
        z - lambda
    } else if z < -lambda {
        z + lambda
    } else {
        0.0
    }
}

/// Compute the auto-selected λ for one (Θ, y) pair using the heuristic
/// `λ = 0.1 × max(|Θᵀ y|) / T'`.
pub fn auto_lambda(theta: &[Vec<f32>], y: &[f32]) -> f32 {
    if theta.is_empty() || theta[0].is_empty() {
        return 0.0;
    }
    let t = theta.len() as f32;
    let l = theta[0].len();
    let mut max_inner = 0.0_f32;
    for col in 0..l {
        let mut acc = 0.0_f32;
        for row in 0..theta.len() {
            acc += theta[row][col] * y[row];
        }
        if acc.abs() > max_inner {
            max_inner = acc.abs();
        }
    }
    AUTO_LAMBDA_FRACTION * max_inner / t.max(1.0)
}

/// Solve a single LASSO problem via coordinate descent.
///
/// Returns the coefficient vector `ξ` of length `theta[0].len()`.
///
/// `column_norm_sq` is `||Θ[:, j]||²` precomputed once and reused across
/// every entity's solve.
pub fn solve_lasso(
    theta: &[Vec<f32>],
    y: &[f32],
    column_norm_sq: &[f32],
    lambda: f32,
) -> Vec<f32> {
    if theta.is_empty() {
        return Vec::new();
    }
    let t = theta.len();
    let l = theta[0].len();
    let mut xi = vec![0.0_f32; l];
    let mut residual = y.to_vec();

    for _iter in 0..LASSO_MAX_ITERS {
        let mut max_delta = 0.0_f32;
        for j in 0..l {
            if column_norm_sq[j] <= 0.0 {
                continue;
            }
            // ρ_j = Θ[:,j]ᵀ (y - Θ ξ) + ||Θ[:,j]||² ξ_j
            //     = Θ[:,j]ᵀ residual + ||Θ[:,j]||² ξ_j
            let mut rho = 0.0_f32;
            for row in 0..t {
                rho += theta[row][j] * residual[row];
            }
            let z = rho + column_norm_sq[j] * xi[j];
            let new_xi = soft_threshold(z / column_norm_sq[j], lambda / column_norm_sq[j]);
            let delta = new_xi - xi[j];
            if delta.abs() > 0.0 {
                // Update residual: r -= delta * Θ[:, j]
                for row in 0..t {
                    residual[row] -= delta * theta[row][j];
                }
                xi[j] = new_xi;
                if delta.abs() > max_delta {
                    max_delta = delta.abs();
                }
            }
        }
        if max_delta < LASSO_TOLERANCE {
            break;
        }
    }
    xi
}

/// Solve `N` independent LASSO problems for each entity column of `y_matrix`.
///
/// Returns the coefficient matrix `Ξ[N × L]` and the actual λ used (relevant
/// when `lambda_l1 == 0.0`, in which case auto-lambda is invoked once per
/// entity).
///
/// When `lambda_cv` is `true`, this currently errors with
/// `InferenceError("CV not implemented in MVP")` per architect Q3 default.
pub fn solve_lasso_n(
    library: &Library,
    y_matrix: &[Vec<f32>],
    n_entities: usize,
    lambda_l1: f32,
    lambda_cv: bool,
) -> Result<(Vec<Vec<f32>>, f32)> {
    if lambda_cv {
        return Err(TensaError::InferenceError(
            "lambda_cv is not implemented in MVP — set false or omit. \
             5-fold CV planned for Phase 15c."
                .into(),
        ));
    }
    if y_matrix.len() != library.theta.len() {
        return Err(TensaError::InferenceError(format!(
            "solve_lasso_n: y_matrix has {} rows, library.theta has {} rows",
            y_matrix.len(),
            library.theta.len()
        )));
    }

    let l = library.terms.len();
    let column_norm_sq: Vec<f32> = library.column_norms.iter().map(|c| c * c).collect();

    let mut xi_matrix = vec![vec![0.0_f32; l]; n_entities];
    // Auto-lambda: pick λ once globally. Take the max across entities so a
    // single λ keeps determinism — per-entity lambdas would couple the
    // bootstrap to entity ordering.
    let lambda_used = if lambda_l1 > 0.0 {
        lambda_l1
    } else {
        let mut max_lambda = 0.0_f32;
        for i in 0..n_entities {
            let y: Vec<f32> = y_matrix.iter().map(|row| row[i]).collect();
            let l_i = auto_lambda(&library.theta, &y);
            if l_i > max_lambda {
                max_lambda = l_i;
            }
        }
        max_lambda.max(1e-6)
    };

    for i in 0..n_entities {
        let y: Vec<f32> = y_matrix.iter().map(|row| row[i]).collect();
        let xi = solve_lasso(&library.theta, &y, &column_norm_sq, lambda_used);
        xi_matrix[i] = xi;
    }

    Ok((xi_matrix, lambda_used))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::hypergraph_reconstruction::library::build_library;

    #[test]
    fn test_soft_threshold_basic() {
        assert_eq!(soft_threshold(0.5, 0.1), 0.4);
        assert_eq!(soft_threshold(-0.5, 0.1), -0.4);
        assert_eq!(soft_threshold(0.05, 0.1), 0.0);
    }

    #[test]
    fn test_lasso_recovers_planted_sparse_solution() {
        // Synthesize y = 2 x_0 + 0 x_1 + 0 x_2; expect xi ≈ [2, 0, 0].
        let mut x: Vec<Vec<f32>> = Vec::new();
        for tt in 0..50 {
            x.push(vec![tt as f32 / 50.0, (tt as f32 * 0.1).sin(), 0.5]);
        }
        let lib = build_library(&x, 3, 1, 0.0).unwrap();
        let y_vec: Vec<f32> = x.iter().map(|r| 2.0 * r[0]).collect();
        let column_norm_sq: Vec<f32> = lib.column_norms.iter().map(|c| c * c).collect();
        let lambda = auto_lambda(&lib.theta, &y_vec);
        let xi = solve_lasso(&lib.theta, &y_vec, &column_norm_sq, lambda);
        assert!(xi[0] > 1.0, "expected xi[0] near 2.0, got {}", xi[0]);
        assert!(xi[1].abs() < 0.5, "expected xi[1] sparse, got {}", xi[1]);
        assert!(xi[2].abs() < 0.5, "expected xi[2] sparse, got {}", xi[2]);
    }

    #[test]
    fn test_solve_lasso_n_lambda_cv_errors() {
        let x = vec![vec![1.0_f32, 2.0]; 5];
        let lib = build_library(&x, 2, 1, 0.0).unwrap();
        let err = solve_lasso_n(&lib, &x, 2, 0.0, true).unwrap_err();
        match err {
            TensaError::InferenceError(msg) => assert!(msg.contains("CV")),
            other => panic!("expected InferenceError, got {other:?}"),
        }
    }

    #[test]
    fn test_solve_lasso_n_uses_explicit_lambda() {
        let x: Vec<Vec<f32>> = (0..30)
            .map(|tt| vec![tt as f32 / 30.0, ((tt * 7) as f32).sin()])
            .collect();
        let lib = build_library(&x, 2, 1, 0.0).unwrap();
        let (_xi, lambda_used) = solve_lasso_n(&lib, &x, 2, 0.42, false).unwrap();
        assert!((lambda_used - 0.42).abs() < 1e-6);
    }
}
