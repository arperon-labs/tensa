//! End-to-end orchestration: hypergraph + narrative_id + params → result.
//!
//! Stages: load → observe → differentiate → build_library (Pearson filter)
//! → solve_lasso (per entity) → symmetrize + threshold → bootstrap.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

use super::bootstrap::run_bootstrap;
use super::derivative::estimate_derivative;
use super::lasso::solve_lasso_n;
use super::library::build_library;
use super::observe::build_state_matrix;
use super::symmetrize::symmetrize_xi;
use super::types::{
    InferredHyperedge, LibraryTerm, MatrixStats, ReconstructionParams, ReconstructionResult,
};

/// Top-level reconstruction entry point.
pub fn reconstruct(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &ReconstructionParams,
) -> Result<ReconstructionResult> {
    if narrative_id.is_empty() {
        return Err(TensaError::InvalidInput(
            "reconstruct: narrative_id is required".into(),
        ));
    }
    if params.max_order > 4 {
        return Err(TensaError::InvalidInput(
            "max_order exceeds hard cap of 4".into(),
        ));
    }

    // Stage 1: entities + idx.
    let entities_full = hypergraph.list_entities_by_narrative(narrative_id)?;
    if entities_full.is_empty() {
        return Err(TensaError::InferenceError(format!(
            "reconstruct: narrative '{narrative_id}' has no entities"
        )));
    }
    if entities_full.len() > params.entity_cap {
        return Err(TensaError::InvalidInput(format!(
            "N={} exceeds entity_cap {}. Filter to a sub-narrative or raise the cap.",
            entities_full.len(),
            params.entity_cap
        )));
    }
    let entity_uuids: Vec<Uuid> = entities_full.iter().map(|e| e.id).collect();
    let entity_idx: HashMap<Uuid, usize> = entity_uuids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let n = entity_uuids.len();

    let mut warnings: Vec<String> = Vec::new();

    // Stage 2: state matrix X. The participation index is built ONCE here
    // (inside build_state_matrix → collect_participation_index) so the
    // bootstrap loop never re-scans the hypergraph.
    let observations = build_state_matrix(
        hypergraph,
        narrative_id,
        entity_uuids.clone(),
        &entity_idx,
        params.window_seconds,
        params.time_resolution_seconds,
        &params.observation,
    )?;
    if observations.x.is_empty() {
        return Err(TensaError::InferenceError(
            "reconstruct: empty observation matrix".into(),
        ));
    }
    let t_pre = observations.x.len();
    if (t_pre as i64) < (10 * n as i64) {
        warnings.push(format!(
            "T={t_pre} < 10·N={} — system may be underdetermined; lower max_order or add more time",
            10 * n
        ));
    }
    let dt_seconds = params.time_resolution_seconds as f32;

    // Stage 3: derivative.
    let series = estimate_derivative(&observations.x, dt_seconds, &params.derivative_estimator)?;
    let t_prime = series.x_dot.len();
    if t_prime < 4 {
        return Err(TensaError::InferenceError(format!(
            "reconstruct: trimmed time-series has only {t_prime} rows after derivative — need >= 4"
        )));
    }

    // Stage 4 + 5: library (with Pearson pre-filter).
    let library = build_library(
        &series.x_trimmed,
        n,
        params.max_order,
        params.pearson_filter_threshold,
    )?;
    let pearson_filtered = library.pearson_filtered_pairs;
    let l_terms = library.terms.len();

    // Stage 6: N independent LASSO solves.
    let (xi_matrix, lambda_used) = solve_lasso_n(
        &library,
        &series.x_dot,
        n,
        params.lambda_l1,
        params.lambda_cv,
    )?;

    // Stage 7: post-processing — symmetrize and extract edges.
    let symmetrized = symmetrize_xi(&xi_matrix, &library.terms, params.symmetrize, n);
    let edges = extract_edges(&symmetrized, &library.terms, &entity_uuids, lambda_used);

    // Stage 8: bootstrap confidence (one extra LASSO solve per resample).
    let confidence_map = if params.bootstrap_k > 0 {
        run_bootstrap(
            &series.x_trimmed,
            &series.x_dot,
            &library,
            n,
            params,
            lambda_used,
            &entity_uuids,
        )?
    } else {
        HashMap::new()
    };

    // Apply confidence and masking-artifact flag.
    let mut edges_with_conf: Vec<InferredHyperedge> = edges
        .into_iter()
        .map(|mut e| {
            let key = canonical_key(&e);
            e.confidence = confidence_map.get(&key).copied().unwrap_or(1.0);
            e
        })
        .collect();
    flag_masking_artifacts(&mut edges_with_conf);
    edges_with_conf.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Stage 7-aux: matrix stats.
    let total_cells = n * l_terms;
    let zero_cells = symmetrized
        .iter()
        .map(|row| row.iter().filter(|v| v.abs() < f32::EPSILON).count())
        .sum::<usize>();
    let sparsity = if total_cells == 0 {
        1.0
    } else {
        zero_cells as f32 / total_cells as f32
    };
    let condition_number_approx = approx_condition_number(&library.column_norms);

    // Goodness-of-fit on a held-out tail of 20% of rows.
    let goodness_of_fit = compute_held_out_r2(&library.theta, &series.x_dot, &xi_matrix, n);

    // Time range from observation axis.
    let time_range = {
        let first = *observations.time_axis.first().unwrap();
        let last = *observations.time_axis.last().unwrap();
        (first, last)
    };

    let bootstrap_done = if confidence_map.is_empty() && params.bootstrap_k > 0 {
        // Bootstrap was requested but produced no confidence entries — happens
        // when symmetrized matrix is all-zero. Report 0 so the caller sees it.
        0
    } else {
        params.bootstrap_k
    };

    Ok(ReconstructionResult {
        inferred_edges: edges_with_conf,
        coefficient_matrix_stats: MatrixStats {
            n_entities: n,
            n_library_terms: l_terms,
            n_timesteps: t_prime,
            sparsity,
            condition_number_approx,
            lambda_used,
            pearson_filtered_pairs: pearson_filtered,
        },
        goodness_of_fit,
        observation_source: params.observation.clone(),
        params_used: params.clone(),
        time_range,
        bootstrap_resamples_completed: bootstrap_done,
        warnings,
    })
}

/// Pull edges from the symmetrized coefficient matrix.
///
/// For each (regression-row i, library term) where |coefficient| > λ/10,
/// materialize a hyperedge `{i} ∪ term.members()` (canonicalized to a
/// sorted unique-UUID set). De-duplicate by the canonical UUID set; weight
/// = max coefficient seen for that set.
pub(crate) fn extract_edges(
    symmetrized: &[Vec<f32>],
    terms: &[LibraryTerm],
    entity_uuids: &[Uuid],
    lambda_used: f32,
) -> Vec<InferredHyperedge> {
    let mut by_set: HashMap<Vec<usize>, f32> = HashMap::new();
    let threshold = (lambda_used / 10.0).max(1e-6);

    for (col, term) in terms.iter().enumerate() {
        for (i, row) in symmetrized.iter().enumerate() {
            let coef = row[col].abs();
            if coef <= threshold {
                continue;
            }
            let mut members = term.members();
            if !members.contains(&i) {
                members.push(i);
            }
            members.sort_unstable();
            members.dedup();
            if members.len() < 2 {
                continue;
            }
            let entry = by_set.entry(members).or_insert(0.0);
            if coef > *entry {
                *entry = coef;
            }
        }
    }

    by_set
        .into_iter()
        .filter_map(|(members, weight)| {
            let uuids: Vec<Uuid> = members
                .iter()
                .filter_map(|&idx| entity_uuids.get(idx).copied())
                .collect();
            if uuids.len() < 2 {
                return None;
            }
            let order = uuids.len() as u8;
            Some(InferredHyperedge {
                members: uuids,
                order,
                weight,
                confidence: 1.0,
                possible_masking_artifact: false,
            })
        })
        .collect()
}

/// Canonical key for hashing hyperedge identities (sorted UUID byte tuples).
pub(crate) fn canonical_key(edge: &InferredHyperedge) -> Vec<u8> {
    let mut sorted: Vec<Uuid> = edge.members.clone();
    sorted.sort();
    let mut out = Vec::with_capacity(sorted.len() * 16);
    for uid in &sorted {
        out.extend_from_slice(uid.as_bytes());
    }
    out
}

/// Set the `possible_masking_artifact` flag for any pairwise edge whose
/// members appear together in a higher-order edge with greater weight.
fn flag_masking_artifacts(edges: &mut [InferredHyperedge]) {
    let higher_order: Vec<(HashSet<Uuid>, f32)> = edges
        .iter()
        .filter(|e| e.order >= 3)
        .map(|e| (e.members.iter().copied().collect(), e.weight))
        .collect();
    for edge in edges.iter_mut() {
        if edge.order != 2 {
            continue;
        }
        let pair: HashSet<Uuid> = edge.members.iter().copied().collect();
        for (set, weight) in &higher_order {
            if pair.is_subset(set) && *weight > edge.weight {
                edge.possible_masking_artifact = true;
                break;
            }
        }
    }
}

/// Approximate condition number = `max_norm / min_norm` of library columns.
fn approx_condition_number(column_norms: &[f32]) -> f32 {
    let positive: Vec<f32> = column_norms.iter().copied().filter(|c| *c > 0.0).collect();
    if positive.is_empty() {
        return 1.0;
    }
    let max = positive.iter().cloned().fold(0.0_f32, f32::max);
    let min = positive.iter().cloned().fold(f32::INFINITY, f32::min);
    if min <= 0.0 || !min.is_finite() {
        1.0
    } else {
        max / min
    }
}

/// Held-out R² over the last 20% of rows. Average across entities.
fn compute_held_out_r2(
    theta: &[Vec<f32>],
    x_dot: &[Vec<f32>],
    xi_matrix: &[Vec<f32>],
    n: usize,
) -> f32 {
    let t = theta.len();
    if t < 5 {
        return 0.0;
    }
    let split = (t * 4) / 5;
    let n_held = t - split;
    if n_held < 1 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    let mut count = 0;
    for i in 0..n {
        let mut ss_res = 0.0_f32;
        let mut ss_tot = 0.0_f32;
        let mut mean = 0.0_f32;
        for row in &x_dot[split..] {
            mean += row[i];
        }
        mean /= n_held as f32;
        for tt in split..t {
            let y_actual = x_dot[tt][i];
            let mut y_pred = 0.0_f32;
            for (col, val) in theta[tt].iter().enumerate() {
                y_pred += val * xi_matrix[i][col];
            }
            ss_res += (y_actual - y_pred).powi(2);
            ss_tot += (y_actual - mean).powi(2);
        }
        if ss_tot > 1e-9 {
            total += 1.0 - ss_res / ss_tot;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_edges_dedupes_by_uuid_set() {
        let uuids = vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];
        // term 0 = Order1(1) ⇒ "x_1 contribution to whoever's being regressed"
        // term 1 = Order1(0) ⇒ "x_0 contribution to whoever's being regressed"
        let terms = vec![LibraryTerm::Order1(1), LibraryTerm::Order1(0)];
        // Row 0 (regression for entity 0) has nonzero weight on x_1 → edge {0,1}
        // Row 1 (regression for entity 1) has nonzero weight on x_0 → edge {0,1}
        // Both routes encode the same undirected edge → must dedupe.
        let symmetrized = vec![
            vec![0.5, 0.0], // row 0 sees x_1 with weight 0.5
            vec![0.0, 0.5], // row 1 sees x_0 with weight 0.5
            vec![0.0, 0.0],
        ];
        let edges = extract_edges(&symmetrized, &terms, &uuids, 0.05);
        assert_eq!(edges.len(), 1, "expected 1 deduped edge, got {edges:?}");
        assert_eq!(edges[0].order, 2);
    }

    #[test]
    fn test_flag_masking_artifacts_marks_pair_below_triple() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        let mut edges = vec![
            InferredHyperedge {
                members: vec![a, b, c],
                order: 3,
                weight: 0.9,
                confidence: 1.0,
                possible_masking_artifact: false,
            },
            InferredHyperedge {
                members: vec![a, b],
                order: 2,
                weight: 0.4,
                confidence: 1.0,
                possible_masking_artifact: false,
            },
        ];
        flag_masking_artifacts(&mut edges);
        assert!(edges[1].possible_masking_artifact);
    }
}
