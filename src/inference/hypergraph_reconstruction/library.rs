// Pearson + library construction is dense matrix code; explicit indexing
// is clearer than rewriting as iterators on small loops.
#![allow(clippy::needless_range_loop)]

//! SINDy library construction with Pearson pre-filter.
//!
//! Given the trimmed state matrix `X[T' × N]`, build the candidate
//! [`LibraryTerm`] index plus the dense library matrix `Θ[T' × L]`, where
//! `L` is the number of surviving candidate terms.
//!
//! ## Pearson pre-filter
//!
//! Per architect §2.2: full library size at `N=100, max_order=3` is ~167k
//! terms — infeasible. The filter retains a candidate term iff every
//! pairwise correlation between its constituent indices clears `ρ_min`
//! (default 0.1). This empirically retains 5–15% of candidates on
//! coordination corpora, reducing `L` by 10–20×.

use crate::error::{Result, TensaError};

use super::types::LibraryTerm;

/// Hard memory cap for the library matrix (in `f32` cells). 50M cells ≈ 200 MB.
const LIBRARY_MEM_CAP_CELLS: usize = 50_000_000;

/// Output of [`build_library`]. The library matrix is row-major and dense.
#[derive(Debug, Clone)]
pub struct Library {
    /// Surviving candidate terms in iteration order.
    pub terms: Vec<LibraryTerm>,
    /// `theta[t][l]` — value of term `l` at trimmed time-step `t`.
    pub theta: Vec<Vec<f32>>,
    /// `theta` column norms — useful for diagnostics + LASSO.
    pub column_norms: Vec<f32>,
    /// Number of candidate pairs/triples/quads that survived the Pearson
    /// filter (excludes order-1 terms which are always kept).
    pub pearson_filtered_pairs: usize,
}

/// Compute the Pearson correlation matrix of the trimmed state matrix.
///
/// Returns a flat `N × N` matrix where `pearson[i][j]` is `corr(X[:,i], X[:,j])`.
/// Diagonal is 1.0 (or 0.0 for entirely-constant columns to avoid NaN).
pub fn pearson_matrix(x: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
    if x.is_empty() {
        return vec![vec![0.0_f32; n]; n];
    }
    let t = x.len();
    let inv_t = 1.0_f32 / t as f32;

    // Mean per column.
    let mut mean = vec![0.0_f32; n];
    for row in x {
        for i in 0..n {
            mean[i] += row[i];
        }
    }
    for m in &mut mean {
        *m *= inv_t;
    }

    // Stddev per column (population).
    let mut std = vec![0.0_f32; n];
    for row in x {
        for i in 0..n {
            let d = row[i] - mean[i];
            std[i] += d * d;
        }
    }
    for s in &mut std {
        *s = (*s * inv_t).sqrt();
    }

    let mut out = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        out[i][i] = if std[i] > 0.0 { 1.0 } else { 0.0 };
        for j in (i + 1)..n {
            if std[i] <= 0.0 || std[j] <= 0.0 {
                out[i][j] = 0.0;
                out[j][i] = 0.0;
                continue;
            }
            let mut acc = 0.0_f32;
            for row in x {
                acc += (row[i] - mean[i]) * (row[j] - mean[j]);
            }
            let r = (acc * inv_t) / (std[i] * std[j]);
            out[i][j] = r;
            out[j][i] = r;
        }
    }
    out
}

/// Build the library term index plus the dense library matrix.
///
/// Strategy:
/// 1. Order-1 terms `x_j` for j in [0, n) — always retained.
/// 2. Order-2 terms `x_j * x_k` (j < k) — retained iff `|ρ_{j,k}| >= threshold`.
/// 3. Order-3 terms `x_j * x_k * x_l` (j < k < l) — retained iff all three
///    pairwise correlations clear `threshold`.
/// 4. Order-4 terms — same all-pairs rule.
///
/// The library matrix is built in one pass after term indexing. Memory
/// guard: if `T' × L > LIBRARY_MEM_CAP_CELLS` the function returns
/// `InvalidInput` with a hint to reduce `entity_cap` or `max_order`.
pub fn build_library(
    x: &[Vec<f32>],
    n: usize,
    max_order: usize,
    pearson_filter_threshold: f32,
) -> Result<Library> {
    if max_order < 1 {
        return Err(TensaError::InvalidInput(
            "max_order must be >= 1".into(),
        ));
    }
    if max_order > 4 {
        return Err(TensaError::InvalidInput(
            "max_order exceeds hard cap of 4".into(),
        ));
    }
    if x.is_empty() {
        return Err(TensaError::InvalidInput(
            "build_library: empty state matrix".into(),
        ));
    }

    let pearson = pearson_matrix(x, n);
    let threshold = pearson_filter_threshold.max(0.0);
    let mut terms: Vec<LibraryTerm> = Vec::new();
    let mut filtered_higher_order: usize = 0;

    // Order-1: keep all.
    for j in 0..n {
        terms.push(LibraryTerm::Order1(j));
    }

    // Order-2.
    if max_order >= 2 {
        for j in 0..n {
            for k in (j + 1)..n {
                if pearson[j][k].abs() >= threshold {
                    terms.push(LibraryTerm::Order2(j, k));
                    filtered_higher_order += 1;
                }
            }
        }
    }

    // Order-3.
    if max_order >= 3 {
        for j in 0..n {
            for k in (j + 1)..n {
                if pearson[j][k].abs() < threshold {
                    continue;
                }
                for l in (k + 1)..n {
                    if pearson[j][l].abs() < threshold || pearson[k][l].abs() < threshold {
                        continue;
                    }
                    terms.push(LibraryTerm::Order3(j, k, l));
                    filtered_higher_order += 1;
                }
            }
        }
    }

    // Order-4.
    if max_order >= 4 {
        for j in 0..n {
            for k in (j + 1)..n {
                if pearson[j][k].abs() < threshold {
                    continue;
                }
                for l in (k + 1)..n {
                    if pearson[j][l].abs() < threshold || pearson[k][l].abs() < threshold {
                        continue;
                    }
                    for m in (l + 1)..n {
                        if pearson[j][m].abs() < threshold
                            || pearson[k][m].abs() < threshold
                            || pearson[l][m].abs() < threshold
                        {
                            continue;
                        }
                        terms.push(LibraryTerm::Order4(j, k, l, m));
                        filtered_higher_order += 1;
                    }
                }
            }
        }
    }

    let t = x.len();
    let l = terms.len();
    if t.saturating_mul(l) > LIBRARY_MEM_CAP_CELLS {
        return Err(TensaError::InvalidInput(format!(
            "Library matrix exceeds 200 MB safety cap: T'={t} × L={l} = {} cells. \
             Reduce entity_cap, lower max_order, or raise pearson_filter_threshold.",
            t.saturating_mul(l)
        )));
    }

    let mut theta = vec![vec![0.0_f32; l]; t];
    for (col, term) in terms.iter().enumerate() {
        match term {
            LibraryTerm::Order1(j) => {
                for (row_idx, row) in x.iter().enumerate() {
                    theta[row_idx][col] = row[*j];
                }
            }
            LibraryTerm::Order2(j, k) => {
                for (row_idx, row) in x.iter().enumerate() {
                    theta[row_idx][col] = row[*j] * row[*k];
                }
            }
            LibraryTerm::Order3(j, k, l_idx) => {
                for (row_idx, row) in x.iter().enumerate() {
                    theta[row_idx][col] = row[*j] * row[*k] * row[*l_idx];
                }
            }
            LibraryTerm::Order4(j, k, l_idx, m) => {
                for (row_idx, row) in x.iter().enumerate() {
                    theta[row_idx][col] = row[*j] * row[*k] * row[*l_idx] * row[*m];
                }
            }
        }
    }

    let mut column_norms = vec![0.0_f32; l];
    for row in &theta {
        for (col, &val) in row.iter().enumerate() {
            column_norms[col] += val * val;
        }
    }
    for c in &mut column_norms {
        *c = c.sqrt();
    }

    Ok(Library {
        terms,
        theta,
        column_norms,
        pearson_filtered_pairs: filtered_higher_order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlated_pair(t: usize) -> Vec<Vec<f32>> {
        // X[:, 0] and X[:, 1] are perfectly correlated (X[:, 1] = X[:, 0]),
        // X[:, 2] is independent (alternating 0/1).
        let mut out = Vec::with_capacity(t);
        for tt in 0..t {
            let v0 = (tt as f32).sin();
            let v1 = v0;
            let v2 = if tt % 3 == 0 { 1.0 } else { -1.0 };
            out.push(vec![v0, v1, v2]);
        }
        out
    }

    #[test]
    fn test_pearson_matrix_identifies_correlated_pair() {
        let x = correlated_pair(50);
        let p = pearson_matrix(&x, 3);
        assert!((p[0][1] - 1.0).abs() < 1e-3);
        assert!(p[0][2].abs() < 0.5);
    }

    #[test]
    fn test_build_library_keeps_order_1_and_filters_pairs() {
        let x = correlated_pair(40);
        let lib = build_library(&x, 3, 2, 0.5).unwrap();
        // Order-1: 3 terms always kept.
        let order1_count = lib
            .terms
            .iter()
            .filter(|t| matches!(t, LibraryTerm::Order1(_)))
            .count();
        assert_eq!(order1_count, 3);
        // Order-2: only the {0,1} pair survives the 0.5 threshold.
        let order2_count = lib
            .terms
            .iter()
            .filter(|t| matches!(t, LibraryTerm::Order2(_, _)))
            .count();
        assert_eq!(order2_count, 1);
        assert_eq!(lib.pearson_filtered_pairs, 1);
    }

    #[test]
    fn test_build_library_max_order_cap() {
        let x = correlated_pair(20);
        let err = build_library(&x, 3, 5, 0.0).unwrap_err();
        match err {
            TensaError::InvalidInput(msg) => assert!(msg.contains("max_order")),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_theta_values_match_term_definition() {
        // X[:, 0] = [1, 2, 3], X[:, 1] = [4, 5, 6].
        let x = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        let lib = build_library(&x, 2, 2, 0.0).unwrap();
        // term 0: x_0; term 1: x_1; term 2: x_0 * x_1.
        assert_eq!(lib.theta[0][0], 1.0);
        assert_eq!(lib.theta[0][1], 4.0);
        assert_eq!(lib.theta[0][2], 4.0);
        assert_eq!(lib.theta[1][2], 10.0);
        assert_eq!(lib.theta[2][2], 18.0);
    }
}
