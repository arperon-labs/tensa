//! Distribution alignment + per-element statistics helpers.
//!
//! Split out of `adapters.rs` to keep that file under the 500-line cap.
//! Both pieces (alignment + stats) are pure functions — no I/O, no
//! engine state. Determinism is guaranteed by the population stddev
//! denominator (N, not N-1) per design doc §1.

use std::collections::HashSet;

use super::adapters::{AdapterChoice, MetricSnapshot};
use super::SyntheticDistribution;

/// Build the aligned [`SyntheticDistribution`] from the source snapshot and
/// K synthetic snapshots. Element keys are the union; missing entries
/// contribute 0.0.
///
/// `pub(crate)` so Phase 13c's dual-significance engine can reuse the same
/// alignment + per-element stats reduction without duplicating the algebra.
pub(crate) fn build_distribution(
    source: &MetricSnapshot,
    synthetic: &[MetricSnapshot],
    adapter: AdapterChoice,
) -> SyntheticDistribution {
    // 1. Element-key union, sorted for stable JSON output.
    let mut key_set: HashSet<String> = HashSet::new();
    for k in source.values.keys() {
        key_set.insert(k.clone());
    }
    for s in synthetic {
        for k in s.values.keys() {
            key_set.insert(k.clone());
        }
    }
    let mut element_keys: Vec<String> = key_set.into_iter().collect();
    element_keys.sort();

    // 2. Per-element vectors.
    let n = element_keys.len();
    let k = synthetic.len() as f64;
    let mut source_values = Vec::with_capacity(n);
    let mut means = Vec::with_capacity(n);
    let mut stddevs = Vec::with_capacity(n);
    let mut z_scores = Vec::with_capacity(n);
    let mut p_values = Vec::with_capacity(n);

    for key in &element_keys {
        let m_real = source.values.get(key).copied().unwrap_or(0.0);
        let samples: Vec<f64> = synthetic
            .iter()
            .map(|s| s.values.get(key).copied().unwrap_or(0.0))
            .collect();
        let (mean, stddev) = mean_stddev(&samples);
        let z = if stddev == 0.0 {
            f64::NAN
        } else {
            (m_real - mean) / stddev
        };
        let p = if synthetic.is_empty()
            || (m_real == 0.0 && samples.iter().all(|&v| v == 0.0))
        {
            // No samples to compare against, OR both source and all synth are
            // zero — empirical p-value is undefined.
            f64::NAN
        } else {
            samples.iter().filter(|&&v| v >= m_real).count() as f64 / k
        };
        source_values.push(m_real);
        means.push(mean);
        stddevs.push(stddev);
        z_scores.push(z);
        p_values.push(p);
    }

    // All four current metrics use "more is significant" — design doc §1
    // documents this as the convention; metadata field carries the explicit
    // string for downstream interpretability. Higher-order contagion (Phase 7b)
    // joins the same convention: a higher z-score on peak_prevalence /
    // r0_estimate / per-size attribution means "this narrative spreads
    // contagion more than the EATH null model would predict".
    let direction = match adapter {
        AdapterChoice::TemporalMotifs
        | AdapterChoice::Communities
        | AdapterChoice::Patterns
        | AdapterChoice::Contagion(_) => "more_is_significant".to_string(),
    };

    SyntheticDistribution {
        element_keys,
        source_values,
        means,
        stddevs,
        z_scores,
        p_values,
        direction,
    }
}

/// Population mean + stddev (denominator N, not N-1) per design doc §1.
fn mean_stddev(samples: &[f64]) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    (mean, var.sqrt())
}
