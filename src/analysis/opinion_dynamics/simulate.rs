//! Top-level simulation entry point: [`simulate_opinion_dynamics`].
//!
//! Pipeline (per design doc §3):
//! 1. Load entities + situations for `narrative_id`, build hyperedges.
//! 2. Validate params (`c ∈ (0, 1)`, `μ ∈ (0, 1]`, `max_steps ≥ 1`, ≥ 1 entity,
//!    ≥ 1 size-≥2 hyperedge).
//! 3. Sample initial opinions; if globally pre-converged, return immediately.
//! 4. Pre-compute log-spaced trajectory sample steps and (optionally) the
//!    `ActivityWeighted` cumulative-weight table.
//! 5. Main loop: select edge(s), update opinions, sample trajectory, check
//!    convergence streak.
//! 6. Post-process: cluster detection, polarisation, echo-chamber index.
//!
//! All RNG goes through a single `ChaCha8Rng` seeded from `params.seed`.

use std::collections::HashMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::{info, warn};
use uuid::Uuid;

use crate::analysis::graph_projection::collect_participation_index;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

use super::cluster::{detect_clusters_density_gap, echo_chamber_index, polarization_index};
use super::init::{compute_log_sample_points, sample_initial_opinions};
use super::types::{
    BcmVariant, HyperedgeSelection, InitialOpinionDist, OpinionDynamicsParams,
    OpinionDynamicsReport, OpinionTrajectory,
};
use super::update::{group_mean_update, pairwise_within_update};

/// Target number of trajectory snapshots. `K` snapshots × `N` floats = bounded
/// memory regardless of `max_steps`.
const TRAJECTORY_K: usize = 30;

/// Run a single opinion-dynamics simulation. See module-level docs.
pub fn simulate_opinion_dynamics(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &OpinionDynamicsParams,
) -> Result<OpinionDynamicsReport> {
    // ── 1. Validate params ────────────────────────────────────────────────
    validate_params(params)?;

    // ── 2. Load entities + situations + build hyperedges ──────────────────
    let mut entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    if entities.is_empty() {
        return Err(TensaError::InvalidInput(format!(
            "opinion dynamics: narrative '{}' has no entities",
            narrative_id
        )));
    }
    entities.sort_by_key(|e| e.id);
    let entity_order: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    let n = entity_order.len();
    let entity_idx: HashMap<Uuid, usize> = entity_order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Err(TensaError::InvalidInput(format!(
            "opinion dynamics: narrative '{}' has no situations; need >= 1 hyperedge",
            narrative_id
        )));
    }
    situations.sort_by_key(|s| s.id);
    let situation_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();

    // Single-pass participation index — N+1 mitigation.
    let part_index =
        collect_participation_index(hypergraph, &entity_idx, &situation_ids, None)?;

    let mut hyperedges: Vec<Vec<usize>> = Vec::with_capacity(situations.len());
    let mut activity_weights: Vec<f32> = Vec::with_capacity(situations.len());
    let mut warned_size1 = false;
    for sit in &situations {
        match part_index.get(&sit.id) {
            Some(members) => {
                let mut m = members.clone();
                m.sort();
                m.dedup();
                if m.len() < 2 {
                    if !warned_size1 {
                        warn!(
                            narrative_id = narrative_id,
                            "opinion dynamics: skipping size-{} hyperedge (need >= 2)",
                            m.len()
                        );
                        warned_size1 = true;
                    }
                    continue;
                }
                hyperedges.push(m);
                activity_weights.push(sit.confidence.max(0.0));
            }
            None => {
                if !warned_size1 {
                    warn!(
                        narrative_id = narrative_id,
                        "opinion dynamics: skipping situation with no participants"
                    );
                    warned_size1 = true;
                }
            }
        }
    }
    if hyperedges.is_empty() {
        return Err(TensaError::InvalidInput(format!(
            "opinion dynamics: narrative '{}' has no hyperedges of size >= 2",
            narrative_id
        )));
    }

    // ── 3. Initial opinions ────────────────────────────────────────────────
    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);

    if let InitialOpinionDist::Custom(v) = &params.initial_opinion_distribution {
        if v.len() != n {
            return Err(TensaError::InvalidInput(format!(
                "opinion dynamics: Custom opinion vector length {} != entity count {}",
                v.len(),
                n
            )));
        }
    }
    let mut x = sample_initial_opinions(&params.initial_opinion_distribution, n, &mut rng);

    // ── 4. Pre-compute trajectory sample points + ActivityWeighted CDF ─────
    let sample_points = compute_log_sample_points(params.max_steps, TRAJECTORY_K);
    let sample_set: std::collections::HashSet<usize> = sample_points.iter().copied().collect();

    // Cumulative weights for ActivityWeighted (kept even if unused — small).
    let cum_weights = build_cum_weights(&activity_weights);

    // ── 5. Pre-converged check ────────────────────────────────────────────
    let mut trajectory = OpinionTrajectory {
        opinion_history: vec![x.clone()],
        sample_steps: vec![0],
        final_opinions: HashMap::new(),
        entity_order: entity_order.clone(),
    };
    let mut variance_timeseries = vec![variance(&x)];

    let initial_spread = spread(&x);
    if initial_spread < params.convergence_tol {
        info!(
            narrative_id = narrative_id,
            initial_spread = initial_spread,
            "opinion dynamics: pre-converged at step 0"
        );
        return finalise(
            hypergraph,
            narrative_id,
            params,
            entity_order,
            x,
            trajectory,
            variance_timeseries,
            0,
            true,
            Some(0),
        );
    }

    // ── 6. Main loop ──────────────────────────────────────────────────────
    let mut convergence_streak: usize = 0;
    let mut converged = false;
    let mut convergence_step: Option<usize> = None;
    let mut last_executed_step: usize = 0;
    let m = hyperedges.len();

    for step in 1..=params.max_steps {
        // 6a. Select edge(s).
        match params.hyperedge_selection {
            HyperedgeSelection::UniformRandom => {
                let idx = rng.gen_range(0..m);
                apply_one(&hyperedges[idx], &mut x, params);
            }
            HyperedgeSelection::ActivityWeighted => {
                let idx = sample_weighted(&cum_weights, &mut rng).unwrap_or(0);
                apply_one(&hyperedges[idx], &mut x, params);
            }
            HyperedgeSelection::PerStepAll => {
                for edge in &hyperedges {
                    apply_one(edge, &mut x, params);
                }
            }
        }
        last_executed_step = step;

        // 6b. Trajectory sampling at log-spaced points.
        if sample_set.contains(&step) {
            trajectory.opinion_history.push(x.clone());
            trajectory.sample_steps.push(step);
            variance_timeseries.push(variance(&x));
        }

        // 6c. Convergence check.
        let s = spread(&x);
        if s < params.convergence_tol {
            convergence_streak += 1;
            if convergence_streak >= params.convergence_window {
                converged = true;
                convergence_step = Some(step);
                break;
            }
        } else {
            convergence_streak = 0;
        }
    }

    // Ensure final step recorded if not already.
    if last_executed_step > 0 && *trajectory.sample_steps.last().unwrap() != last_executed_step {
        trajectory.opinion_history.push(x.clone());
        trajectory.sample_steps.push(last_executed_step);
        variance_timeseries.push(variance(&x));
    }

    finalise(
        hypergraph,
        narrative_id,
        params,
        entity_order,
        x,
        trajectory,
        variance_timeseries,
        last_executed_step,
        converged,
        convergence_step,
    )
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn validate_params(params: &OpinionDynamicsParams) -> Result<()> {
    if !params.confidence_bound.is_finite()
        || params.confidence_bound <= 0.0
        || params.confidence_bound >= 1.0
    {
        return Err(TensaError::InvalidInput(format!(
            "opinion dynamics: confidence_bound must be in (0, 1), got {}",
            params.confidence_bound
        )));
    }
    if !params.convergence_rate.is_finite()
        || params.convergence_rate <= 0.0
        || params.convergence_rate > 1.0
    {
        return Err(TensaError::InvalidInput(format!(
            "opinion dynamics: convergence_rate must be in (0, 1], got {}",
            params.convergence_rate
        )));
    }
    if params.max_steps < 1 {
        return Err(TensaError::InvalidInput(
            "opinion dynamics: max_steps must be >= 1".into(),
        ));
    }
    if params.convergence_window < 1 {
        return Err(TensaError::InvalidInput(
            "opinion dynamics: convergence_window must be >= 1".into(),
        ));
    }
    if !params.convergence_tol.is_finite() || params.convergence_tol <= 0.0 {
        return Err(TensaError::InvalidInput(
            "opinion dynamics: convergence_tol must be > 0".into(),
        ));
    }
    Ok(())
}

#[inline]
fn apply_one(edge: &[usize], x: &mut [f32], params: &OpinionDynamicsParams) {
    match params.model {
        BcmVariant::PairwiseWithin => pairwise_within_update(edge, x, params),
        BcmVariant::GroupMean => group_mean_update(edge, x, params),
    }
}

#[inline]
fn spread(x: &[f32]) -> f32 {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    for &v in x {
        if v < min_x {
            min_x = v;
        }
        if v > max_x {
            max_x = v;
        }
    }
    if max_x.is_finite() && min_x.is_finite() {
        max_x - min_x
    } else {
        0.0
    }
}

#[inline]
fn variance(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

/// Build a strictly increasing cumulative-weight table for activity-weighted
/// sampling. If all weights are zero/finite-zero, returns a uniform CDF so
/// sampling falls back to uniform.
fn build_cum_weights(weights: &[f32]) -> Vec<f32> {
    let mut cum: Vec<f32> = Vec::with_capacity(weights.len());
    let mut acc = 0.0_f32;
    for &w in weights {
        acc += w.max(0.0);
        cum.push(acc);
    }
    if acc <= 0.0 {
        // Degenerate: replace with uniform CDF.
        cum.clear();
        for i in 0..weights.len() {
            cum.push((i + 1) as f32);
        }
    }
    cum
}

/// Inverse-CDF sampling from a strictly-increasing cumulative-weight vector.
/// Returns `None` for an empty input. Linear scan — fine for `M < ~10k`
/// (Phase 16b targets); for larger `M`, switch to binary search in 16c.
fn sample_weighted(cum_weights: &[f32], rng: &mut ChaCha8Rng) -> Option<usize> {
    if cum_weights.is_empty() {
        return None;
    }
    let total = *cum_weights.last().unwrap();
    if total <= 0.0 {
        return Some(rng.gen_range(0..cum_weights.len()));
    }
    let r: f32 = rng.gen::<f32>() * total;
    for (i, &c) in cum_weights.iter().enumerate() {
        if r < c {
            return Some(i);
        }
    }
    Some(cum_weights.len() - 1)
}

#[allow(clippy::too_many_arguments)]
fn finalise(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &OpinionDynamicsParams,
    entity_order: Vec<Uuid>,
    final_x: Vec<f32>,
    mut trajectory: OpinionTrajectory,
    variance_timeseries: Vec<f32>,
    num_steps_executed: usize,
    converged: bool,
    convergence_step: Option<usize>,
) -> Result<OpinionDynamicsReport> {
    let final_opinions: HashMap<Uuid, f32> = entity_order
        .iter()
        .copied()
        .zip(final_x.iter().copied())
        .collect();
    trajectory.final_opinions = final_opinions.clone();

    let (cluster_sizes, cluster_means) =
        detect_clusters_density_gap(&final_x, params.convergence_tol);
    let num_clusters = cluster_sizes.len();
    let polarization = polarization_index(&final_x);
    let (echo, echo_available) =
        echo_chamber_index(hypergraph, narrative_id, &final_opinions)?;

    Ok(OpinionDynamicsReport {
        num_steps_executed,
        converged,
        convergence_step,
        num_clusters,
        cluster_sizes,
        cluster_means,
        variance_timeseries,
        polarization_index: polarization,
        echo_chamber_index: echo,
        echo_chamber_available: echo_available,
        trajectory,
        params_used: params.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_params_rejects_bad_c() {
        let mut p = OpinionDynamicsParams::default();
        p.confidence_bound = 0.0;
        assert!(validate_params(&p).is_err());
        p.confidence_bound = 1.0;
        assert!(validate_params(&p).is_err());
        p.confidence_bound = -0.1;
        assert!(validate_params(&p).is_err());
        p.confidence_bound = f32::NAN;
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn test_validate_params_rejects_bad_mu() {
        let mut p = OpinionDynamicsParams::default();
        p.convergence_rate = 0.0;
        assert!(validate_params(&p).is_err());
        p.convergence_rate = 1.5;
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn test_validate_params_accepts_defaults() {
        assert!(validate_params(&OpinionDynamicsParams::default()).is_ok());
    }

    #[test]
    fn test_spread_basic() {
        assert!((spread(&[0.1, 0.5, 0.9]) - 0.8).abs() < 1e-6);
        assert_eq!(spread(&[]), 0.0);
        assert_eq!(spread(&[0.42]), 0.0);
    }

    #[test]
    fn test_variance_basic() {
        let v = variance(&[0.0, 1.0]);
        assert!((v - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_build_cum_weights_zero_falls_back_to_uniform() {
        let cum = build_cum_weights(&[0.0, 0.0, 0.0]);
        assert_eq!(cum, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sample_weighted_zero_total_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cum = vec![0.0, 0.0, 0.0];
        // total == 0 in the cum vector itself triggers the uniform branch
        let idx = sample_weighted(&cum, &mut rng);
        assert!(idx.is_some_and(|i| i < 3));
    }
}
