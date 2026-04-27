// Loop variables that index small dense matrices read more clearly with
// explicit indexing than with `iter().enumerate()`. The numerical kernels
// here are deliberately written in low-level form to keep the hot loops
// auto-vectorisable.
#![allow(clippy::needless_range_loop)]

//! Numerical derivative estimators for SINDy.
//!
//! Two estimators:
//! - [`DerivativeEstimator::FiniteDiff`]: central differences. Simple,
//!   sensitive to noise. Trims first + last row.
//! - [`DerivativeEstimator::SavitzkyGolay`]: polynomial smoother +
//!   differentiator in one pass. Default for Phase 15b — EATH binary
//!   participation signals are step-function-like and need smoothing
//!   before differentiation.
//!
//! Both produce trimmed `(X', X_dot)` pairs where `X'.len() == X_dot.len() == T'`
//! and `T' < T`. The architect's worked example (§13.4) uses
//! `SavitzkyGolay{5, 2}` for binary EATH signals.

use crate::error::{Result, TensaError};

use super::types::DerivativeEstimator;

/// Output of derivative estimation: trimmed state matrix and matching
/// derivative matrix. Both have shape `[T' × N]`.
#[derive(Debug, Clone)]
pub struct DifferentiatedSeries {
    pub x_trimmed: Vec<Vec<f32>>,
    pub x_dot: Vec<Vec<f32>>,
    /// Time-step duration used to scale the finite-difference / SG kernel.
    pub dt_seconds: f32,
}

/// Estimate `dx_i/dt` for every entity column.
///
/// `dt_seconds` is the bin width of the input X matrix (used to scale the
/// derivative — the kernel itself works in bin units).
pub fn estimate_derivative(
    x: &[Vec<f32>],
    dt_seconds: f32,
    estimator: &DerivativeEstimator,
) -> Result<DifferentiatedSeries> {
    if x.is_empty() {
        return Err(TensaError::InvalidInput(
            "estimate_derivative: empty state matrix".into(),
        ));
    }
    if dt_seconds <= 0.0 {
        return Err(TensaError::InvalidInput(
            "estimate_derivative: dt_seconds must be > 0".into(),
        ));
    }
    let n_cols = x[0].len();
    if x.iter().any(|row| row.len() != n_cols) {
        return Err(TensaError::InvalidInput(
            "estimate_derivative: ragged state matrix".into(),
        ));
    }

    match estimator {
        DerivativeEstimator::FiniteDiff => finite_diff(x, dt_seconds),
        DerivativeEstimator::SavitzkyGolay { window, order } => {
            savitzky_golay(x, dt_seconds, *window, *order)
        }
    }
}

/// Central finite differences with forward/backward at boundaries.
///
/// We trim the first and last row to keep the result bounded by interior
/// estimates only — boundary differences inflate noise on EATH-style
/// step functions.
fn finite_diff(x: &[Vec<f32>], dt: f32) -> Result<DifferentiatedSeries> {
    let t = x.len();
    if t < 3 {
        return Err(TensaError::InvalidInput(format!(
            "FiniteDiff: T={t}, need >= 3 rows for central differences"
        )));
    }
    let n = x[0].len();
    let trimmed_t = t - 2;
    let mut x_trimmed = Vec::with_capacity(trimmed_t);
    let mut x_dot = Vec::with_capacity(trimmed_t);
    for tt in 1..(t - 1) {
        x_trimmed.push(x[tt].clone());
        let mut row = vec![0.0_f32; n];
        for i in 0..n {
            row[i] = (x[tt + 1][i] - x[tt - 1][i]) / (2.0 * dt);
        }
        x_dot.push(row);
    }
    Ok(DifferentiatedSeries {
        x_trimmed,
        x_dot,
        dt_seconds: dt,
    })
}

/// Savitzky-Golay smoother + differentiator.
///
/// Pre-computes both the smoothing kernel (zero-th derivative coefficients)
/// and the first-derivative kernel via least-squares polynomial fit on a
/// `window`-wide neighbourhood, then applies them as 1D convolutions along
/// the time axis. Trim `(window-1)/2` rows from each end.
///
/// Standard reference: Press et al., *Numerical Recipes* §14.9.
fn savitzky_golay(
    x: &[Vec<f32>],
    dt: f32,
    window: usize,
    order: usize,
) -> Result<DifferentiatedSeries> {
    if window < 3 || !is_odd(window) {
        return Err(TensaError::InvalidInput(format!(
            "SavitzkyGolay: window={window}, must be odd and >= 3"
        )));
    }
    if order < 1 {
        return Err(TensaError::InvalidInput(format!(
            "SavitzkyGolay: order={order}, must be >= 1 to compute a derivative"
        )));
    }
    if order >= window {
        return Err(TensaError::InvalidInput(format!(
            "SavitzkyGolay: order ({order}) must be < window ({window})"
        )));
    }
    let t = x.len();
    if t < window {
        return Err(TensaError::InvalidInput(format!(
            "SavitzkyGolay: T={t} rows is shorter than window={window}"
        )));
    }
    let n = x[0].len();
    let half = (window - 1) / 2;

    let smooth_kernel = sg_kernel(window, order, 0)?;
    let mut deriv_kernel = sg_kernel(window, order, 1)?;
    // First-derivative kernel must be scaled by 1/dt for time-series units.
    for c in &mut deriv_kernel {
        *c /= dt;
    }

    let trimmed_t = t - window + 1;
    let mut x_trimmed = Vec::with_capacity(trimmed_t);
    let mut x_dot = Vec::with_capacity(trimmed_t);
    for centre in half..(t - half) {
        let mut sm_row = vec![0.0_f32; n];
        let mut dv_row = vec![0.0_f32; n];
        for k in 0..window {
            let row = &x[centre + k - half];
            let sm_w = smooth_kernel[k];
            let dv_w = deriv_kernel[k];
            for i in 0..n {
                sm_row[i] += sm_w * row[i];
                dv_row[i] += dv_w * row[i];
            }
        }
        x_trimmed.push(sm_row);
        x_dot.push(dv_row);
    }
    Ok(DifferentiatedSeries {
        x_trimmed,
        x_dot,
        dt_seconds: dt,
    })
}

/// Compute the Savitzky-Golay convolution coefficients for the centre point
/// of a `window`-sample symmetric neighbourhood, polynomial degree `order`,
/// returning the `derivative`-th derivative coefficients.
///
/// Solves `(A^T A) c = A^T e_d`, where `A[i,j] = i^j` for `i ∈ [-half, half]`,
/// `j ∈ [0, order]`, and `e_d` is the unit vector for the requested
/// derivative index. The convolution coefficients are then row `derivative`
/// of `(A^T A)^{-1} A^T`, scaled by `derivative!`.
fn sg_kernel(window: usize, order: usize, derivative: usize) -> Result<Vec<f32>> {
    let half = (window as isize - 1) / 2;
    let cols = order + 1;

    // Build A[i, j] = i^j, with i ranging from -half to +half.
    let mut a: Vec<Vec<f64>> = vec![vec![0.0; cols]; window];
    for (row, i) in (-half..=half).enumerate() {
        for j in 0..cols {
            a[row][j] = (i as f64).powi(j as i32);
        }
    }

    // ATA[a][b] = sum_i a[i,a] * a[i,b].
    let mut ata = vec![vec![0.0_f64; cols]; cols];
    for a_row in 0..window {
        for ca in 0..cols {
            for cb in 0..cols {
                ata[ca][cb] += a[a_row][ca] * a[a_row][cb];
            }
        }
    }

    // Solve ATA · v = e_derivative (Gauss-Jordan with partial pivoting).
    let mut rhs = vec![0.0_f64; cols];
    rhs[derivative] = 1.0;
    let v = gauss_solve(ata, rhs).ok_or_else(|| {
        TensaError::InvalidInput(format!(
            "sg_kernel: singular normal-equations matrix for window={window}, order={order}"
        ))
    })?;

    // Convolution coefficients: c[i] = sum_j a[i,j] * v[j], scaled by derivative!.
    let factorial: f64 = (1..=derivative as u32).map(|x| x as f64).product::<f64>().max(1.0);
    let mut coeffs = vec![0.0_f32; window];
    for i in 0..window {
        let mut acc = 0.0_f64;
        for j in 0..cols {
            acc += a[i][j] * v[j];
        }
        coeffs[i] = (acc * factorial) as f32;
    }
    Ok(coeffs)
}

#[inline]
fn is_odd(n: usize) -> bool {
    n & 1 == 1
}

/// Plain Gauss-Jordan elimination for very small symmetric systems.
/// Returns `None` if pivoting fails (singular matrix).
fn gauss_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|r| r.len() != n) {
        return None;
    }
    for k in 0..n {
        // Partial pivoting.
        let mut pivot = k;
        let mut best = a[k][k].abs();
        for r in (k + 1)..n {
            let cand = a[r][k].abs();
            if cand > best {
                best = cand;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }
        let inv = 1.0 / a[k][k];
        for j in k..n {
            a[k][j] *= inv;
        }
        b[k] *= inv;
        for r in 0..n {
            if r == k {
                continue;
            }
            let factor = a[r][k];
            if factor.abs() < 1e-18 {
                continue;
            }
            for j in k..n {
                a[r][j] -= factor * a[k][j];
            }
            b[r] -= factor * b[k];
        }
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_signal(t: usize, n: usize, slope: f32) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(t);
        for tt in 0..t {
            let mut row = vec![0.0_f32; n];
            for i in 0..n {
                row[i] = slope * tt as f32 + (i as f32) * 0.1;
            }
            out.push(row);
        }
        out
    }

    #[test]
    fn test_finite_diff_recovers_constant_slope() {
        let x = linear_signal(10, 3, 2.0);
        let series = estimate_derivative(&x, 1.0, &DerivativeEstimator::FiniteDiff).unwrap();
        assert_eq!(series.x_trimmed.len(), 8);
        for row in &series.x_dot {
            for &v in row {
                assert!((v - 2.0).abs() < 1e-4, "expected slope=2.0, got {v}");
            }
        }
    }

    #[test]
    fn test_savitzky_golay_recovers_constant_slope() {
        let x = linear_signal(20, 4, -1.5);
        let series = estimate_derivative(
            &x,
            1.0,
            &DerivativeEstimator::SavitzkyGolay {
                window: 5,
                order: 2,
            },
        )
        .unwrap();
        assert_eq!(series.x_trimmed.len(), 16);
        for row in &series.x_dot {
            for &v in row {
                assert!(
                    (v - (-1.5)).abs() < 1e-3,
                    "SG should recover linear slope, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_savitzky_golay_smooths_step_function() {
        // Step function x[t] = 0 for t < 5, 1 for t >= 5, length 11.
        // FiniteDiff would give a single huge spike; SG should give a
        // smoothed bump centred near t=5.
        let mut x = vec![vec![0.0_f32]; 11];
        for tt in 5..11 {
            x[tt][0] = 1.0;
        }
        let series = estimate_derivative(
            &x,
            1.0,
            &DerivativeEstimator::SavitzkyGolay {
                window: 5,
                order: 2,
            },
        )
        .unwrap();
        // Total derivative integrated should be ~1 (the step height).
        let sum_dx: f32 = series.x_dot.iter().map(|r| r[0]).sum();
        assert!(
            sum_dx > 0.5 && sum_dx < 2.0,
            "SG should preserve step magnitude (got integrated dx={sum_dx})"
        );
    }

    #[test]
    fn test_savitzky_golay_rejects_even_window() {
        let x = linear_signal(10, 1, 1.0);
        let err = estimate_derivative(
            &x,
            1.0,
            &DerivativeEstimator::SavitzkyGolay {
                window: 4,
                order: 2,
            },
        )
        .unwrap_err();
        match err {
            TensaError::InvalidInput(_) => {}
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_finite_diff_rejects_short_series() {
        let x = vec![vec![1.0_f32], vec![2.0_f32]];
        assert!(estimate_derivative(&x, 1.0, &DerivativeEstimator::FiniteDiff).is_err());
    }
}
