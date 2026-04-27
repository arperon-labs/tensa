//! Density-gap cluster detection, polarisation index, echo-chamber index.
//!
//! All three operate on the *final* opinion vector returned by
//! [`super::simulate::simulate_opinion_dynamics`]. The echo-chamber index is
//! the only one that touches the hypergraph: it reads pre-computed
//! label-propagation labels at `an/lp/{narrative_id}/{entity_id}` and skips
//! gracefully (returning `(0.0, false)`) when those labels are absent.

use std::collections::HashMap;

use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;

// ── Density-gap clustering ─────────────────────────────────────────────────

/// Per design doc §14 Q3: a sensible default gap that works for both
/// high-precision (small `tol`) and standard-precision runs.
fn cluster_gap_threshold(tol: f32) -> f32 {
    (tol * 2.0).max(1e-3)
}

/// Density-gap cluster detection on a 1-D opinion vector.
///
/// Returns `(cluster_sizes, cluster_means)` ordered by descending size.
/// Two opinions sit in the same cluster iff their sorted-neighbour
/// difference is below `gap_threshold = max(tol*2, 1e-3)` — the §14 Q3
/// default that handles both interactive and high-precision runs.
///
/// An empty input returns two empty vectors. A single-element input returns
/// one cluster of size 1 with the only opinion as its mean.
pub fn detect_clusters_density_gap(opinions: &[f32], tol: f32) -> (Vec<usize>, Vec<f32>) {
    if opinions.is_empty() {
        return (vec![], vec![]);
    }
    let gap = cluster_gap_threshold(tol);
    let mut sorted: Vec<f32> = opinions.to_vec();
    // f32::total_cmp gives a total order even with NaNs (should not occur
    // in practice but defensive).
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut cluster_sizes: Vec<usize> = Vec::new();
    let mut cluster_means: Vec<f32> = Vec::new();
    let mut current_size: usize = 1;
    let mut current_sum: f32 = sorted[0];
    for k in 1..sorted.len() {
        let prev = sorted[k - 1];
        let cur = sorted[k];
        if cur - prev > gap {
            cluster_sizes.push(current_size);
            cluster_means.push(current_sum / current_size as f32);
            current_size = 1;
            current_sum = cur;
        } else {
            current_size += 1;
            current_sum += cur;
        }
    }
    cluster_sizes.push(current_size);
    cluster_means.push(current_sum / current_size as f32);

    // Order by descending size; preserve mean alignment.
    let mut paired: Vec<(usize, f32)> = cluster_sizes
        .into_iter()
        .zip(cluster_means)
        .collect();
    paired.sort_by(|a, b| b.0.cmp(&a.0));
    let sizes: Vec<usize> = paired.iter().map(|(s, _)| *s).collect();
    let means: Vec<f32> = paired.iter().map(|(_, m)| *m).collect();
    (sizes, means)
}

// ── Polarisation index ─────────────────────────────────────────────────────

/// Normalised-variance polarisation index, range `[0, 1]`.
///
/// `polarization = var(x) / (mean(x) * (1 - mean(x)))` — i.e. the variance
/// normalised by the maximum possible variance for a `[0,1]`-supported
/// distribution with the same mean. Values near `1` indicate maximum
/// bimodality (mass at 0 and 1); values near `0` indicate consensus.
///
/// Returns `0.0` for empty input or for a degenerate mean at exactly 0 or 1
/// (no variance possible).
pub fn polarization_index(opinions: &[f32]) -> f32 {
    if opinions.is_empty() {
        return 0.0;
    }
    let n = opinions.len() as f32;
    let mean = opinions.iter().sum::<f32>() / n;
    if mean <= 0.0 || mean >= 1.0 {
        return 0.0;
    }
    let var = opinions.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let norm = mean * (1.0 - mean);
    (var / norm).clamp(0.0, 1.0)
}

// ── Echo-chamber index ─────────────────────────────────────────────────────

/// Echo-chamber index, range `[0, 1]`.
///
/// `1 - (mean_c σ_c) / σ_global`, where the per-community standard deviations
/// are size-weighted. `1.0` ≡ perfect echo chambers; `0.0` ≡ no within-vs-
/// between heterogeneity gap.
///
/// Reads label-propagation labels from KV at `an/lp/{narrative_id}/{eid}`.
/// When labels are missing for the narrative entirely (or KV scan returns
/// nothing), this returns `(0.0, false)` — Phase 16b never errors on a
/// missing prerequisite, it just signals unavailability via the bool.
pub fn echo_chamber_index(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    final_opinions: &HashMap<Uuid, f32>,
) -> Result<(f32, bool)> {
    if final_opinions.is_empty() {
        return Ok((0.0, false));
    }

    // Pull all labels for this narrative.
    let prefix = format!("an/lp/{}/", narrative_id);
    let pairs = hypergraph.store().prefix_scan(prefix.as_bytes())?;
    if pairs.is_empty() {
        return Ok((0.0, false));
    }

    // Decode label[entity] from KV.
    let mut entity_label: HashMap<Uuid, usize> = HashMap::with_capacity(pairs.len());
    for (key, val) in pairs {
        // key format: an/lp/{narrative_id}/{entity_uuid_string}
        let key_str = match std::str::from_utf8(&key) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let entity_str = match key_str.rsplit('/').next() {
            Some(s) => s,
            None => continue,
        };
        let entity_id = match Uuid::parse_str(entity_str) {
            Ok(id) => id,
            Err(_) => continue,
        };
        let label_f64: f64 = match serde_json::from_slice(&val) {
            Ok(v) => v,
            Err(_) => continue,
        };
        entity_label.insert(entity_id, label_f64.round() as usize);
    }

    if entity_label.is_empty() {
        return Ok((0.0, false));
    }

    // Group final opinions by community label, restricted to entities that
    // both have a label and are in the simulation's entity set.
    let mut community: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut all_opinions: Vec<f32> = Vec::with_capacity(final_opinions.len());
    for (eid, &op) in final_opinions {
        if let Some(&label) = entity_label.get(eid) {
            community.entry(label).or_default().push(op);
            all_opinions.push(op);
        }
    }

    if all_opinions.is_empty() {
        return Ok((0.0, false));
    }

    let sigma_global = std_dev(&all_opinions);
    if sigma_global <= 0.0 {
        // Everyone agrees globally — no chambers needed. Convention §9.2.
        return Ok((0.0, true));
    }

    let n = all_opinions.len() as f32;
    let mut weighted_sigma_c = 0.0_f32;
    for opinions_c in community.values() {
        let size = opinions_c.len() as f32;
        let sigma_c = std_dev(opinions_c);
        weighted_sigma_c += (size / n) * sigma_c;
    }

    let index = (1.0 - weighted_sigma_c / sigma_global).clamp(0.0, 1.0);
    Ok((index, true))
}

#[inline]
fn std_dev(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::make_hg;

    #[test]
    fn test_detect_clusters_empty() {
        let (sizes, means) = detect_clusters_density_gap(&[], 1e-4);
        assert!(sizes.is_empty());
        assert!(means.is_empty());
    }

    #[test]
    fn test_detect_clusters_single_consensus() {
        let opinions = vec![0.5; 10];
        let (sizes, means) = detect_clusters_density_gap(&opinions, 1e-4);
        assert_eq!(sizes, vec![10]);
        assert!((means[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_detect_clusters_finds_planted_bimodal() {
        // 5 at exactly 0.1, 5 at exactly 0.9: gap of 0.8 >> 1e-3 threshold.
        let mut opinions = vec![0.1_f32; 5];
        opinions.extend(std::iter::repeat(0.9_f32).take(5));
        let (sizes, means) = detect_clusters_density_gap(&opinions, 1e-4);
        assert_eq!(sizes, vec![5, 5]);
        // Means should bracket the gap.
        assert!((means[0] - 0.1).abs() < 0.05 || (means[0] - 0.9).abs() < 0.05);
        assert!((means[1] - 0.1).abs() < 0.05 || (means[1] - 0.9).abs() < 0.05);
    }

    #[test]
    fn test_polarization_extremes() {
        // Consensus → ~0
        let consensus = vec![0.5_f32; 100];
        let p_low = polarization_index(&consensus);
        assert!(p_low < 0.01, "consensus polarization {p_low}");
        // Maximally polarised → 1.0 (half at 0, half at 1).
        let mut polarised = vec![0.0_f32; 50];
        polarised.extend(std::iter::repeat(1.0_f32).take(50));
        let p_high = polarization_index(&polarised);
        assert!(p_high > 0.99, "polarised polarization {p_high}");
    }

    #[test]
    fn test_polarization_empty_returns_zero() {
        assert_eq!(polarization_index(&[]), 0.0);
    }

    #[test]
    fn test_echo_chamber_missing_labels_returns_zero_unavailable() {
        let hg = make_hg();
        let mut opinions: HashMap<Uuid, f32> = HashMap::new();
        opinions.insert(Uuid::now_v7(), 0.3);
        opinions.insert(Uuid::now_v7(), 0.7);
        let (idx, available) = echo_chamber_index(&hg, "no-such-narrative", &opinions).unwrap();
        assert_eq!(idx, 0.0);
        assert!(!available);
    }

    #[test]
    fn test_echo_chamber_with_planted_labels() {
        let hg = make_hg();
        let nid = "echo-test";
        // Two communities of 3 each. Community 0 opinions tight near 0.2,
        // community 1 tight near 0.8.
        let mut opinions: HashMap<Uuid, f32> = HashMap::new();
        let community_0: Vec<Uuid> = (0..3).map(|_| Uuid::now_v7()).collect();
        let community_1: Vec<Uuid> = (0..3).map(|_| Uuid::now_v7()).collect();
        for (i, eid) in community_0.iter().enumerate() {
            opinions.insert(*eid, 0.20 + 0.001 * i as f32);
            let key = format!("an/lp/{nid}/{eid}");
            let val = serde_json::to_vec(&0.0_f64).unwrap();
            hg.store().put(key.as_bytes(), &val).unwrap();
        }
        for (i, eid) in community_1.iter().enumerate() {
            opinions.insert(*eid, 0.80 + 0.001 * i as f32);
            let key = format!("an/lp/{nid}/{eid}");
            let val = serde_json::to_vec(&1.0_f64).unwrap();
            hg.store().put(key.as_bytes(), &val).unwrap();
        }
        let (idx, available) = echo_chamber_index(&hg, nid, &opinions).unwrap();
        assert!(available);
        assert!(idx > 0.9, "expected near-perfect echo chambers, got {idx}");
    }
}
