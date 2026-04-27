//! Phase-transition sweep: vary `c`, measure convergence time, locate the
//! spike (Hickok §5).
//!
//! On a complete hypergraph with `N(0.5, σ²)` initial opinions, convergence
//! time spikes near `c = σ²`. The sweep returns
//! [`super::types::PhaseTransitionReport`] containing the per-`c`
//! convergence times and the inferred critical `c*`.
//!
//! This module is **distinct** from `analysis::contagion_bistability`: that
//! sweeps β and measures prevalence; this sweeps `c` and measures
//! convergence time. See design doc §8.2 for the comparison table.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

use super::init::sample_initial_opinions;
use super::simulate::simulate_opinion_dynamics;
use super::types::{OpinionDynamicsParams, PhaseTransitionReport};

/// Default multiplicative spike threshold for `critical_c` detection.
pub const DEFAULT_SPIKE_THRESHOLD: f32 = 3.0;

/// Run the phase-transition sweep. `c_range = (start, end, num_points)` —
/// `num_points` must be ≥ 2; an evenly-spaced linspace from `start` to `end`
/// inclusive. Each point gets a fresh `ChaCha8Rng` seeded from
/// `base_params.seed XOR (i as u64)` to ensure run independence.
///
/// `convergence_time` is `Some(step)` on convergence, `None` on cutoff.
/// `critical_c_estimate` is the smallest `c_i` whose convergence time
/// exceeds `median(convergence_times) * spike_threshold` (default 3.0×).
pub fn run_phase_transition_sweep(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    c_range: (f32, f32, usize),
    base_params: &OpinionDynamicsParams,
) -> Result<PhaseTransitionReport> {
    let (c_start, c_end, num_points) = c_range;
    if num_points < 2 {
        return Err(TensaError::InvalidInput(format!(
            "phase transition sweep: num_points must be >= 2, got {}",
            num_points
        )));
    }
    if !c_start.is_finite() || !c_end.is_finite() {
        return Err(TensaError::InvalidInput(
            "phase transition sweep: c_start/c_end must be finite".into(),
        ));
    }
    if c_start <= 0.0 || c_end >= 1.0 || c_start >= c_end {
        return Err(TensaError::InvalidInput(format!(
            "phase transition sweep: require 0 < c_start < c_end < 1, got ({}, {})",
            c_start, c_end
        )));
    }

    // 1. c_values: linspace inclusive of both endpoints.
    let c_values: Vec<f32> = (0..num_points)
        .map(|i| {
            let t = i as f32 / (num_points - 1) as f32;
            c_start + (c_end - c_start) * t
        })
        .collect();

    // 2. initial_variance: sample once with base_params.seed and the
    //    base_params initial distribution. This is the σ² the spike should
    //    sit near (when distribution is Gaussian centred at 0.5).
    let mut diag_rng = ChaCha8Rng::seed_from_u64(base_params.seed);
    let n_for_diag = hypergraph
        .list_entities_by_narrative(narrative_id)?
        .len();
    let initial_variance = if n_for_diag == 0 {
        0.0
    } else {
        let xs = sample_initial_opinions(
            &base_params.initial_opinion_distribution,
            n_for_diag,
            &mut diag_rng,
        );
        compute_variance(&xs)
    };

    // 3. Per-c simulations.
    let mut convergence_times: Vec<Option<usize>> = Vec::with_capacity(c_values.len());
    for (i, &c) in c_values.iter().enumerate() {
        let mut params = base_params.clone();
        params.confidence_bound = c;
        params.seed = base_params.seed ^ (i as u64);
        let report = simulate_opinion_dynamics(hypergraph, narrative_id, &params)?;
        let t = if report.converged {
            Some(report.convergence_step.unwrap_or(report.num_steps_executed))
        } else {
            None
        };
        convergence_times.push(t);
    }

    // 4. Critical-c detection.
    let critical_c_estimate =
        detect_critical_c(&c_values, &convergence_times, DEFAULT_SPIKE_THRESHOLD);

    Ok(PhaseTransitionReport {
        c_values,
        convergence_times,
        critical_c_estimate,
        initial_variance,
        spike_threshold: DEFAULT_SPIKE_THRESHOLD,
    })
}

/// Identify the smallest `c_i` whose convergence time exceeds the median
/// convergence time by `spike_threshold ×`. Returns `None` when no such
/// spike exists.
///
/// `None` (cutoff) entries are treated as "spiked": if the convergence
/// timed out, that's a strong signal that the dynamics fragmented.
/// They're folded into the median as `max(times)` so the median itself
/// remains a robust lower-bound reference; spike detection then registers
/// the cutoff as an explicit spike.
pub fn detect_critical_c(
    c_values: &[f32],
    times: &[Option<usize>],
    spike_threshold: f32,
) -> Option<f32> {
    if c_values.len() != times.len() || c_values.is_empty() {
        return None;
    }

    // Median over the *converged* runs only — a robust reference for "fast"
    // convergence time. Cutoff entries (None) are an unconditional spike
    // signal: failure to converge within max_steps is the strongest possible
    // spike (the dynamics stayed fragmented).
    let converged_only: Vec<f64> = times
        .iter()
        .filter_map(|t| t.map(|v| v as f64))
        .collect();
    let median_converged = if converged_only.is_empty() {
        // Every run hit cutoff — return the smallest c (the lowest c is the
        // first observed cutoff).
        return Some(c_values[0]);
    } else {
        let mut copy = converged_only.clone();
        copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = copy.len() / 2;
        if copy.len().is_multiple_of(2) {
            0.5 * (copy[mid - 1] + copy[mid])
        } else {
            copy[mid]
        }
    };
    if median_converged <= 0.0 {
        return None;
    }
    let threshold = median_converged * spike_threshold as f64;

    // Smallest c_i whose convergence time exceeds the threshold (or that
    // hit the cutoff entirely — the strongest possible spike signal).
    for (i, t) in times.iter().enumerate() {
        match t {
            None => return Some(c_values[i]),
            Some(s) if (*s as f64) > threshold => return Some(c_values[i]),
            _ => {}
        }
    }
    None
}

#[inline]
fn compute_variance(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_critical_c_finds_spike() {
        let c_values = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
        // Most convergence times are around 1000; one spike at c=0.15.
        let times = vec![
            Some(1000),
            Some(1100),
            Some(50_000),
            Some(900),
            Some(950),
            Some(1000),
        ];
        let critical = detect_critical_c(&c_values, &times, 3.0);
        assert_eq!(critical, Some(0.15));
    }

    #[test]
    fn test_detect_critical_c_none_when_flat() {
        let c_values = vec![0.05, 0.10, 0.15, 0.20];
        let times = vec![Some(1000), Some(1050), Some(1010), Some(1020)];
        assert_eq!(detect_critical_c(&c_values, &times, 3.0), None);
    }

    #[test]
    fn test_detect_critical_c_treats_cutoff_as_spike() {
        let c_values = vec![0.05, 0.10, 0.15];
        // Two converge fast, third hits cutoff (None).
        let times = vec![Some(100), Some(120), None];
        let critical = detect_critical_c(&c_values, &times, 3.0);
        assert_eq!(critical, Some(0.15));
    }

    #[test]
    fn test_detect_critical_c_empty_returns_none() {
        assert_eq!(detect_critical_c(&[], &[], 3.0), None);
    }

    #[test]
    fn test_detect_critical_c_mismatch_returns_none() {
        assert_eq!(
            detect_critical_c(&[0.1, 0.2], &[Some(100)], 3.0),
            None
        );
    }
}
