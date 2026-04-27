//! Platform β calibration — empirical transmission rate estimation.
//!
//! Calibrates per-platform SMIR transmission rates from observed data
//! using Bayesian updating. Provides default literature-calibrated values
//! and supports progressive refinement from campaign observations.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hypergraph::Hypergraph;

// ─── Calibration Data ────────────────────────────────────────

/// Calibrated beta values for a platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCalibration {
    pub platform: String,
    /// Current calibrated beta.
    pub beta: f64,
    /// Prior beta (literature default).
    pub prior_beta: f64,
    /// Number of observations used for calibration.
    pub observation_count: usize,
    /// Uncertainty (standard deviation of posterior).
    pub uncertainty: f64,
}

/// Default literature-calibrated beta values.
///
/// Sources:
/// - Twitter: Vosoughi, Roy & Aral (2018), Science 359(6380)
/// - Facebook: post-2018 algorithm changes (reduced organic reach)
/// - Telegram: channel-based, no algorithmic suppression
/// - TikTok: algorithmic amplification of novel content
/// - YouTube: recommendation-driven, slower cascade
pub fn default_betas() -> Vec<PlatformCalibration> {
    vec![
        PlatformCalibration {
            platform: "twitter".into(),
            beta: 0.40,
            prior_beta: 0.40,
            observation_count: 0,
            uncertainty: 0.10,
        },
        PlatformCalibration {
            platform: "facebook".into(),
            beta: 0.22,
            prior_beta: 0.22,
            observation_count: 0,
            uncertainty: 0.08,
        },
        PlatformCalibration {
            platform: "telegram".into(),
            beta: 0.65,
            prior_beta: 0.65,
            observation_count: 0,
            uncertainty: 0.15,
        },
        PlatformCalibration {
            platform: "tiktok".into(),
            beta: 0.55,
            prior_beta: 0.55,
            observation_count: 0,
            uncertainty: 0.12,
        },
        PlatformCalibration {
            platform: "youtube".into(),
            beta: 0.15,
            prior_beta: 0.15,
            observation_count: 0,
            uncertainty: 0.05,
        },
    ]
}

/// Update beta estimate using a Bayesian conjugate prior.
///
/// Uses a simple normal-normal conjugate model:
/// posterior_mean = (prior_precision * prior_mean + data_precision * data_mean) / (prior_precision + data_precision)
///
/// where precision = 1/variance.
pub fn bayesian_beta_update(
    calibration: &mut PlatformCalibration,
    observed_r0: f64,
    gamma: f64,
    population: f64,
) {
    if gamma <= 0.0 || population <= 0.0 {
        return;
    }

    // Estimate beta from observed R₀: R₀ = β × S / (γ × N), so β ≈ R₀ × γ
    // (assuming S ≈ N for early-stage campaigns)
    let observed_beta = observed_r0 * gamma;

    let prior_precision = 1.0 / (calibration.uncertainty * calibration.uncertainty);
    // Data precision increases with observations
    let data_precision = (calibration.observation_count as f64 + 1.0) * 10.0;

    let total_precision = prior_precision + data_precision;
    let posterior_mean =
        (prior_precision * calibration.beta + data_precision * observed_beta) / total_precision;
    let posterior_uncertainty = (1.0 / total_precision).sqrt();

    calibration.beta = posterior_mean.clamp(0.01, 1.0);
    calibration.uncertainty = posterior_uncertainty;
    calibration.observation_count += 1;
}

// ─── Analytical SMIR Approximation ───────────────────────────

/// Fast analytical approximation of SMIR steady-state.
///
/// Used as fallback when no trained GNN surrogate is available.
/// Returns (final_misinformed_fraction, final_r0, time_to_peak_days).
pub fn analytical_smir_approximation(
    beta: f64,
    gamma: f64,
    delta: f64,
    initial_misinformed_fraction: f64,
) -> (f64, f64, f64) {
    let r0 = if gamma > 0.0 { beta / gamma } else { 0.0 };

    if r0 <= 1.0 {
        // Sub-critical: narrative dies out
        return (0.0, r0, 0.0);
    }

    // SIR final size equation approximation:
    // final_infected ≈ 1 - 1/R₀ (for large populations, no inoculation)
    // With inoculation: effective S₀ = 1 - delta/beta fraction are susceptible
    let effective_s0 = (1.0 - delta / beta.max(0.001)).max(0.0);
    let effective_r0 = r0 * effective_s0;

    let final_misinformed = if effective_r0 > 1.0 {
        (1.0 - 1.0 / effective_r0) * effective_s0
    } else {
        initial_misinformed_fraction * 0.5 // Decaying
    };

    // Time to peak: approximately 1 / (beta - gamma) in natural time units
    let growth_rate = (beta * effective_s0 - gamma).max(0.001);
    let time_to_peak = (1.0 / growth_rate).min(365.0); // Cap at 1 year

    (final_misinformed.min(1.0), effective_r0, time_to_peak)
}

// ─── KV Storage ──────────────────────────────────────────────

const CALIB_PREFIX: &[u8] = b"adv/calib/";

pub fn store_calibration(hypergraph: &Hypergraph, calibration: &PlatformCalibration) -> Result<()> {
    let mut key = CALIB_PREFIX.to_vec();
    key.extend_from_slice(calibration.platform.as_bytes());
    let value = serde_json::to_vec(calibration)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

pub fn load_calibration(
    hypergraph: &Hypergraph,
    platform: &str,
) -> Result<Option<PlatformCalibration>> {
    let mut key = CALIB_PREFIX.to_vec();
    key.extend_from_slice(platform.as_bytes());
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_default_betas_in_range() {
        let betas = default_betas();
        assert_eq!(betas.len(), 5);
        for b in &betas {
            assert!(
                b.beta > 0.0 && b.beta < 1.0,
                "{}: beta={}",
                b.platform,
                b.beta
            );
            assert!(b.uncertainty > 0.0);
        }
    }

    #[test]
    fn test_bayesian_update_moves_toward_observed() {
        let mut cal = PlatformCalibration {
            platform: "twitter".into(),
            beta: 0.40,
            prior_beta: 0.40,
            observation_count: 0,
            uncertainty: 0.10,
        };

        // Observe R₀ = 3.0 with gamma = 0.05 → observed_beta = 0.15
        bayesian_beta_update(&mut cal, 3.0, 0.05, 10_000.0);

        assert!(
            cal.beta < 0.40 && cal.beta > 0.15,
            "beta should move toward observed: {}",
            cal.beta
        );
        assert_eq!(cal.observation_count, 1);
        assert!(cal.uncertainty < 0.10, "uncertainty should decrease");
    }

    #[test]
    fn test_bayesian_update_converges_with_many_observations() {
        let mut cal = PlatformCalibration {
            platform: "test".into(),
            beta: 0.50,
            prior_beta: 0.50,
            observation_count: 0,
            uncertainty: 0.10,
        };

        // Repeatedly observe R₀ = 4.0 with gamma = 0.05 → beta = 0.20
        for _ in 0..20 {
            bayesian_beta_update(&mut cal, 4.0, 0.05, 10_000.0);
        }

        assert!(
            (cal.beta - 0.20).abs() < 0.05,
            "should converge toward 0.20: {}",
            cal.beta
        );
    }

    #[test]
    fn test_analytical_subcritical() {
        let (final_m, r0, _) = analytical_smir_approximation(0.03, 0.05, 0.0, 0.01);
        assert!(r0 < 1.0, "should be subcritical: R₀={}", r0);
        assert!(final_m < 0.01, "subcritical should die out: {}", final_m);
    }

    #[test]
    fn test_analytical_supercritical() {
        let (final_m, r0, peak_time) = analytical_smir_approximation(0.40, 0.05, 0.0, 0.01);
        assert!(r0 > 1.0, "should be supercritical: R₀={}", r0);
        assert!(
            final_m > 0.5,
            "supercritical should infect majority: {}",
            final_m
        );
        assert!(peak_time > 0.0, "peak time should be positive");
    }

    #[test]
    fn test_analytical_inoculation_reduces_final_size() {
        let (no_inoc, _, _) = analytical_smir_approximation(0.40, 0.05, 0.0, 0.01);
        let (with_inoc, _, _) = analytical_smir_approximation(0.40, 0.05, 0.05, 0.01);

        assert!(
            with_inoc < no_inoc,
            "inoculation should reduce final size: {} vs {}",
            with_inoc,
            no_inoc
        );
    }

    #[test]
    fn test_calibration_persistence() {
        let store = Arc::new(MemoryStore::new());
        let hg = crate::Hypergraph::new(store);

        let cal = PlatformCalibration {
            platform: "twitter".into(),
            beta: 0.35,
            prior_beta: 0.40,
            observation_count: 5,
            uncertainty: 0.07,
        };
        store_calibration(&hg, &cal).unwrap();

        let loaded = load_calibration(&hg, "twitter").unwrap();
        assert!(loaded.is_some());
        assert!((loaded.unwrap().beta - 0.35).abs() < 1e-10);
    }
}
