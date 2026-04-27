//! Coefficient-matrix symmetrization for the SINDy reconstruction pipeline.
//!
//! After N independent LASSO solves we have an Ξ matrix whose rows are the
//! per-entity coefficient vectors. Each candidate hyperedge appears in
//! multiple regressions (for entity i, term `x_j * x_k` evidences edge
//! {i, j, k}; the same edge is also evidenced by row j, term `x_i * x_k`,
//! and row k, term `x_i * x_j`). Symmetrization combines those mirror
//! coefficients per term so the canonical edge identity carries one weight.
//!
//! We use MAX (not MEAN) for order ≥ 2 terms — a single strong piece of
//! evidence shouldn't be diluted by weak/zero mirror-row signals on noisy
//! participation-rate observations. Order-1 terms still use the symmetric
//! mean since both directions provide equally valid pairwise evidence.

use super::types::LibraryTerm;

/// Symmetrize the coefficient matrix per library term.
pub(crate) fn symmetrize_xi(
    xi_matrix: &[Vec<f32>],
    terms: &[LibraryTerm],
    enabled: bool,
    n: usize,
) -> Vec<Vec<f32>> {
    let abs_xi: Vec<Vec<f32>> = xi_matrix
        .iter()
        .map(|r| r.iter().map(|v| v.abs()).collect())
        .collect();
    if !enabled {
        return abs_xi;
    }

    let mut out = vec![vec![0.0_f32; terms.len()]; n];
    for (col, term) in terms.iter().enumerate() {
        match term {
            LibraryTerm::Order1(j) => {
                // Pairwise edge {i, j}. Pair the (i, x_j) coefficient with
                // the symmetric (j, x_i) coefficient when the latter exists.
                for i in 0..n {
                    let primary = abs_xi[i][col];
                    let mirror = if i != *j {
                        terms
                            .iter()
                            .position(|t| matches!(t, LibraryTerm::Order1(idx) if *idx == i))
                            .map(|c2| abs_xi[*j][c2])
                            .unwrap_or(0.0)
                    } else {
                        primary
                    };
                    out[i][col] = if primary > 0.0 && mirror > 0.0 {
                        0.5 * (primary + mirror)
                    } else {
                        primary.max(mirror)
                    };
                }
            }
            LibraryTerm::Order2(j, k) => {
                // Triadic edge {i, j, k}. MAX across the three regressions
                // that "see" the edge.
                for i in 0..n {
                    let mut best = abs_xi[i][col];
                    if i != *j {
                        if let Some(c2) = position_of_order2(terms, i, *k) {
                            best = best.max(abs_xi[*j][c2]);
                        }
                    }
                    if i != *k {
                        if let Some(c2) = position_of_order2(terms, i, *j) {
                            best = best.max(abs_xi[*k][c2]);
                        }
                    }
                    out[i][col] = best;
                }
            }
            LibraryTerm::Order3(j, k, l) => {
                for i in 0..n {
                    let mut best = abs_xi[i][col];
                    for &(target, a, b, c) in &[
                        (*j, i, *k, *l),
                        (*k, i, *j, *l),
                        (*l, i, *j, *k),
                    ] {
                        if target == i {
                            continue;
                        }
                        if let Some(c2) = position_of_order3(terms, a, b, c) {
                            best = best.max(abs_xi[target][c2]);
                        }
                    }
                    out[i][col] = best;
                }
            }
            LibraryTerm::Order4(_, _, _, _) => {
                // Order-4 symmetrization: |coefficient| as-is.
                for i in 0..n {
                    out[i][col] = abs_xi[i][col];
                }
            }
        }
    }
    out
}

fn position_of_order2(terms: &[LibraryTerm], a: usize, b: usize) -> Option<usize> {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    terms
        .iter()
        .position(|t| matches!(t, LibraryTerm::Order2(j, k) if *j == lo && *k == hi))
}

fn position_of_order3(terms: &[LibraryTerm], a: usize, b: usize, c: usize) -> Option<usize> {
    let mut sorted = [a, b, c];
    sorted.sort_unstable();
    terms.iter().position(|t| {
        matches!(t, LibraryTerm::Order3(j, k, l) if *j == sorted[0] && *k == sorted[1] && *l == sorted[2])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xi_row(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[test]
    fn test_symmetrize_pairwise_averages_mirror_columns() {
        let xi = vec![xi_row(&[0.0, 0.4]), xi_row(&[0.6, 0.0])];
        let terms = vec![LibraryTerm::Order1(0), LibraryTerm::Order1(1)];
        let out = symmetrize_xi(&xi, &terms, true, 2);
        assert!((out[0][1] - 0.5).abs() < 1e-6, "got {}", out[0][1]);
        assert!((out[1][0] - 0.5).abs() < 1e-6, "got {}", out[1][0]);
    }

    #[test]
    fn test_symmetrize_disabled_returns_abs_only() {
        let xi = vec![xi_row(&[-0.3, 0.5]), xi_row(&[0.2, -0.1])];
        let terms = vec![LibraryTerm::Order1(0), LibraryTerm::Order1(1)];
        let out = symmetrize_xi(&xi, &terms, false, 2);
        assert_eq!(out[0][0], 0.3);
        assert_eq!(out[0][1], 0.5);
        assert_eq!(out[1][0], 0.2);
        assert_eq!(out[1][1], 0.1);
    }
}
