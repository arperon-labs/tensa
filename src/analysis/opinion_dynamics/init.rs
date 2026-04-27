//! Initial-opinion sampling and log-spaced trajectory sample-point computation.
//!
//! Both functions are pure — they consume the RNG passed in and never touch
//! the hypergraph. All Gaussian draws use Box-Muller transform on `f32`
//! uniforms produced by the same RNG; this avoids pulling in `rand_distr`.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::types::InitialOpinionDist;

/// Sample `n` initial opinions from `dist`, advancing `rng` in canonical
/// order. For [`InitialOpinionDist::Custom`], the supplied vector is returned
/// directly *and validated for length by the caller* (see
/// [`super::simulate::simulate_opinion_dynamics`]).
pub fn sample_initial_opinions(
    dist: &InitialOpinionDist,
    n: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<f32> {
    match dist {
        InitialOpinionDist::Uniform => (0..n).map(|_| rng.gen::<f32>()).collect(),
        InitialOpinionDist::Gaussian { mean, std } => (0..n)
            .map(|_| clamp_unit(box_muller(rng) * *std + *mean))
            .collect(),
        InitialOpinionDist::Bimodal {
            mode_a,
            mode_b,
            spread,
        } => (0..n)
            .map(|_| {
                let coin: f32 = rng.gen();
                let mode = if coin < 0.5 { *mode_a } else { *mode_b };
                clamp_unit(mode + box_muller(rng) * *spread)
            })
            .collect(),
        InitialOpinionDist::Custom(v) => v.clone(),
    }
}

/// Compute deterministic, deduplicated, log-spaced sample steps over
/// `[0, max_steps]`. Always includes step `0` (initial state) and step
/// `max_steps` (final state). For `max_steps = 100_000` and
/// `k_target = 30`, this yields ~30 unique snapshots.
///
/// Memory bound: `K × N × 4 bytes` for the trajectory; for `K = 30`,
/// `N = 5000`, this is 600 KB.
pub fn compute_log_sample_points(max_steps: usize, k_target: usize) -> Vec<usize> {
    if max_steps == 0 {
        return vec![0];
    }
    let mut points: Vec<usize> = Vec::with_capacity(k_target + 2);
    points.push(0);

    if k_target >= 2 && max_steps >= 1 {
        let denom = (k_target.saturating_sub(1).max(1)) as f64;
        for k in 1..k_target.saturating_sub(1) {
            let f = k as f64 / denom;
            let raw = (max_steps as f64).powf(f);
            let s = raw as usize;
            // Skip 0 (already present); skip duplicates.
            if s > *points.last().unwrap_or(&0) {
                points.push(s);
            }
        }
    }

    if max_steps != *points.last().unwrap_or(&0) {
        points.push(max_steps);
    }
    points.dedup();
    points
}

// ── Internal helpers ───────────────────────────────────────────────────────

#[inline]
fn clamp_unit(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Standard normal sample via Box-Muller. Consumes 2 uniforms; returns one
/// of the two outputs (we don't bother caching the second — Phase 16b is
/// not RNG-bound).
fn box_muller(rng: &mut ChaCha8Rng) -> f32 {
    // Reject u1 == 0 to avoid log(0).
    let u1: f32 = loop {
        let v: f32 = rng.gen();
        if v > 0.0 {
            break v;
        }
    };
    let u2: f32 = rng.gen();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_uniform_sampling_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let xs = sample_initial_opinions(&InitialOpinionDist::Uniform, 1000, &mut rng);
        assert_eq!(xs.len(), 1000);
        assert!(xs.iter().all(|&x| (0.0..=1.0).contains(&x)));
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "mean drift {}", mean);
    }

    #[test]
    fn test_gaussian_clamping() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        // Wide std so clamping definitely happens.
        let xs = sample_initial_opinions(
            &InitialOpinionDist::Gaussian {
                mean: 0.5,
                std: 0.5,
            },
            500,
            &mut rng,
        );
        assert!(xs.iter().all(|&x| (0.0..=1.0).contains(&x)));
        // At least some values should be at the boundaries.
        let at_zero = xs.iter().filter(|&&x| x == 0.0).count();
        let at_one = xs.iter().filter(|&&x| x == 1.0).count();
        assert!(at_zero + at_one > 0, "expected boundary saturation");
    }

    #[test]
    fn test_bimodal_distribution_has_two_modes() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let xs = sample_initial_opinions(
            &InitialOpinionDist::Bimodal {
                mode_a: 0.1,
                mode_b: 0.9,
                spread: 0.02,
            },
            1000,
            &mut rng,
        );
        let near_a = xs.iter().filter(|&&x| (x - 0.1).abs() < 0.1).count();
        let near_b = xs.iter().filter(|&&x| (x - 0.9).abs() < 0.1).count();
        // Roughly equal halves, both substantial.
        assert!(near_a > 350, "mode A undercount {near_a}");
        assert!(near_b > 350, "mode B undercount {near_b}");
        assert!(near_a + near_b > 950, "noise outside both modes");
    }

    #[test]
    fn test_custom_returns_supplied_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let v = vec![0.1, 0.5, 0.9];
        let xs = sample_initial_opinions(&InitialOpinionDist::Custom(v.clone()), v.len(), &mut rng);
        assert_eq!(xs, v);
    }

    #[test]
    fn test_log_sample_points_includes_zero_and_final() {
        let pts = compute_log_sample_points(100_000, 30);
        assert_eq!(pts[0], 0);
        assert_eq!(*pts.last().unwrap(), 100_000);
        assert!(pts.len() >= 5 && pts.len() <= 32);
        // strictly increasing
        for w in pts.windows(2) {
            assert!(w[0] < w[1], "non-monotone: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_log_sample_points_small_max_steps() {
        let pts = compute_log_sample_points(5, 30);
        assert_eq!(pts[0], 0);
        assert_eq!(*pts.last().unwrap(), 5);
        // strictly increasing, no duplicates
        for w in pts.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn test_log_sample_points_zero_max_steps() {
        let pts = compute_log_sample_points(0, 30);
        assert_eq!(pts, vec![0]);
    }

    #[test]
    fn test_sample_initial_deterministic_by_seed() {
        let mut a = ChaCha8Rng::seed_from_u64(99);
        let mut b = ChaCha8Rng::seed_from_u64(99);
        let xa = sample_initial_opinions(&InitialOpinionDist::Uniform, 50, &mut a);
        let xb = sample_initial_opinions(&InitialOpinionDist::Uniform, 50, &mut b);
        assert_eq!(xa, xb);
    }
}
