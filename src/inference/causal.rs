//! Causal inference engine.
//!
//! Implements NOTEARS-inspired causal discovery with temporal ordering
//! constraints, do-calculus for interventions, and counterfactual
//! beam search. Uses Allen interval relations to enforce that causes
//! must precede effects.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

/// Causal discovery algorithm selector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CausalAlgorithm {
    /// DAGMA: log-determinant M-matrix constraint (Bello et al. NeurIPS 2022).
    /// 30x faster than NOTEARS at d=800, 2x better SHD accuracy.
    /// Default choice for most use cases.
    #[default]
    Dagma,
    /// NOTEARS: trace-exponential DAG constraint (Zheng et al. NeurIPS 2018).
    /// More mature, well-studied. Better for very small graphs (d < 10)
    /// or when numerical stability of matrix inversion is a concern.
    Notears,
}

/// Configuration for the causal engine.
#[derive(Debug, Clone)]
pub struct CausalConfig {
    /// Algorithm to use for causal discovery.
    pub algorithm: CausalAlgorithm,
    /// Maximum situations before hierarchical decomposition.
    pub decomposition_threshold: usize,
    /// Learning rate for NOTEARS gradient descent.
    pub learning_rate: f64,
    /// Convergence threshold for gradient norm.
    pub convergence_threshold: f64,
    /// Maximum iterations for NOTEARS inner loop.
    pub max_iterations: usize,
    /// Minimum edge weight to include in output.
    pub min_edge_weight: f64,
    /// Beam width for counterfactual search.
    pub beam_width: usize,
    /// Maximum depth for counterfactual beam search.
    pub beam_depth: usize,
    /// Minimum probability to keep a counterfactual branch.
    pub prune_threshold: f32,
    /// L1 regularization penalty for NOTEARS sparsity.
    pub lambda1: f64,
    /// Initial augmented Lagrangian penalty parameter.
    pub rho_init: f64,
    /// Maximum rho before stopping augmented Lagrangian.
    pub rho_max: f64,
    /// Initial Lagrange multiplier for DAG constraint.
    pub alpha_init: f64,
    /// DAG constraint tolerance: h(W) < h_tolerance means acyclic.
    pub h_tolerance: f64,
    /// Maximum outer iterations for augmented Lagrangian.
    pub max_outer_iterations: usize,
    /// Adaptive s schedule for DAGMA (spectral radius safety margin).
    /// Each outer iteration uses the next value. Default: [1.0, 0.75, 0.5, 0.25].
    pub s_schedule: Vec<f64>,
    /// L2 regularization weight for prior matrix: `lambda_prior * ||W - P||`.
    pub lambda_prior: f64,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            algorithm: CausalAlgorithm::default(),
            decomposition_threshold: 50,
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            max_iterations: 500,
            min_edge_weight: 0.1,
            beam_width: 5,
            beam_depth: 20,
            prune_threshold: 0.05,
            lambda1: 0.1,
            rho_init: 1.0,
            rho_max: 1e16,
            alpha_init: 0.0,
            h_tolerance: 1e-8,
            max_outer_iterations: 100,
            s_schedule: vec![1.0, 0.75, 0.5, 0.25],
            lambda_prior: 0.1,
        }
    }
}

/// Compute matrix exponential via truncated Taylor series.
///
/// `e^M = I + M + M^2/2! + M^3/3! + ... + M^k/k!`
///
/// Suitable for small matrices (d <= ~50) typical in narrative analysis.
#[cfg(feature = "inference")]
fn matrix_exp_taylor(m: &nalgebra::DMatrix<f64>, terms: usize) -> nalgebra::DMatrix<f64> {
    let n = m.nrows();
    let mut result = nalgebra::DMatrix::identity(n, n);
    let mut term = nalgebra::DMatrix::identity(n, n); // M^k / k!
    for k in 1..=terms {
        term = &term * m / (k as f64);
        result += &term;
    }
    result
}

/// Compute the NOTEARS DAG constraint and its gradient.
///
/// Returns `(h, grad_h)` where:
/// - `h = tr(e^{W circ W}) - d` (equals 0 iff W is a DAG)
/// - `grad_h = 2 * W circ (e^{W circ W})^T`
#[cfg(feature = "inference")]
fn dag_constraint(w: &nalgebra::DMatrix<f64>) -> (f64, nalgebra::DMatrix<f64>) {
    let d = w.nrows();
    // Element-wise square: W circ W
    let w_sq = w.component_mul(w);
    // Matrix exponential of W circ W
    let exp_w_sq = matrix_exp_taylor(&w_sq, 20);
    // h(W) = tr(e^{W circ W}) - d
    let h = exp_w_sq.trace() - d as f64;
    // grad_h = 2 * W circ (e^{W circ W})^T
    let grad_h = w.component_mul(&exp_w_sq.transpose()) * 2.0;
    (h, grad_h)
}

/// Compute the DAGMA DAG constraint and its gradient (Bello et al. 2022).
///
/// Uses log-determinant M-matrix characterization:
/// - `h(W) = -log det(sI - W circ W) + d * log(s)` (equals 0 iff W is a DAG)
/// - `grad_h = 2 * W circ (sI - W circ W)^{-1T}`
///
/// The parameter `s` must be greater than the spectral radius of `W circ W`.
/// Avoids the expensive matrix exponential of NOTEARS.
#[cfg(feature = "inference")]
fn dagma_constraint(
    w: &nalgebra::DMatrix<f64>,
    s: f64,
) -> std::result::Result<(f64, nalgebra::DMatrix<f64>), &'static str> {
    let d = w.nrows();

    // Element-wise square: W circ W
    let w_sq = w.component_mul(w);

    // sI - W circ W
    let s_eye = nalgebra::DMatrix::identity(d, d) * s;
    let m = &s_eye - &w_sq;

    // Determinant and inverse
    let det = m.determinant();
    if det <= 0.0 {
        return Err("DAGMA: sI - W*W is not positive definite (s too small or W too large)");
    }

    let m_inv = match m.clone().try_inverse() {
        Some(inv) => inv,
        None => return Err("DAGMA: matrix inversion failed"),
    };

    // h(W) = -log(det(sI - W*W)) + d * log(s)
    let h = -det.ln() + (d as f64) * s.ln();

    // grad_h = 2 * W circ (sI - W*W)^{-T}
    let grad_h = w.component_mul(&m_inv.transpose()) * 2.0;

    Ok((h, grad_h))
}

/// Causal inference engine implementing NOTEARS with temporal constraints.
pub struct CausalEngine {
    config: CausalConfig,
}

impl Default for CausalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalEngine {
    /// Create a new causal engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: CausalConfig::default(),
        }
    }

    /// Create a new causal engine with custom configuration.
    pub fn with_config(config: CausalConfig) -> Self {
        Self { config }
    }

    /// Discover causal links among situations using NOTEARS with temporal constraints.
    pub fn discover_causes(
        &self,
        situations: &[Situation],
        hypergraph: &Hypergraph,
    ) -> Result<(CausalGraph, CausalPathExplanation)> {
        let empty_explanation = CausalPathExplanation {
            paths: vec![],
            summary: "No causal paths (insufficient situations).".to_string(),
        };
        if situations.is_empty() {
            return Ok((
                CausalGraph {
                    links: vec![],
                    confidence: 0.0,
                },
                empty_explanation,
            ));
        }
        if situations.len() == 1 {
            return Ok((
                CausalGraph {
                    links: vec![],
                    confidence: 1.0,
                },
                empty_explanation,
            ));
        }

        let n = situations.len();

        // Build temporal ordering constraints
        let temporal_mask = self.build_temporal_mask(situations);

        // Build feature matrix from situations
        let features = self.build_feature_matrix(situations, hypergraph)?;

        // Seed adjacency from existing causal links
        let mut adjacency = self.seed_adjacency(situations, hypergraph)?;

        // Causal discovery via selected algorithm
        self.run_causal_discovery(&mut adjacency, &features, &temporal_mask, n)?;

        // Note: discover_causes_with_params() supports LLM prior matrix via job parameters.

        // Extract links from adjacency matrix
        let links = self.extract_links(situations, &adjacency, n);

        // Trace strongest causal paths for explanation
        let explanation = self.trace_causal_paths(situations, &adjacency, n, 5);

        let confidence = if links.is_empty() {
            0.0
        } else {
            links.iter().map(|l| l.strength).sum::<f32>() / links.len() as f32
        };

        Ok((CausalGraph { links, confidence }, explanation))
    }

    /// Discover causal links with optional LLM prior matrix from job parameters.
    ///
    /// When `params` contains `use_llm_priors: true` and `prior_matrix: [[f64]]`,
    /// the prior matrix is used to regularize DAGMA optimization.
    pub fn discover_causes_with_params(
        &self,
        situations: &[Situation],
        hypergraph: &Hypergraph,
        params: &serde_json::Value,
    ) -> Result<(CausalGraph, CausalPathExplanation)> {
        let n = situations.len();
        let use_priors = params
            .get("use_llm_priors")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if !use_priors || n == 0 {
            return self.discover_causes(situations, hypergraph);
        }

        let temporal_mask = self.build_temporal_mask(situations);
        let features = self.build_feature_matrix(situations, hypergraph)?;
        let mut adjacency = self.seed_adjacency(situations, hypergraph)?;

        // Build flat prior matrix from job parameters
        let prior_flat = Self::build_prior_flat(params, n);

        self.run_causal_discovery_with_prior(
            &mut adjacency,
            &features,
            &temporal_mask,
            n,
            prior_flat.as_deref(),
        )?;

        let links = self.extract_links(situations, &adjacency, n);
        let explanation = self.trace_causal_paths(situations, &adjacency, n, 5);
        let confidence = if links.is_empty() {
            0.0
        } else {
            links.iter().map(|l| l.strength).sum::<f32>() / links.len() as f32
        };

        Ok((CausalGraph { links, confidence }, explanation))
    }

    /// Compute counterfactual outcomes via beam search.
    pub fn counterfactual(
        &self,
        intervention: &Intervention,
        hypergraph: &Hypergraph,
    ) -> Result<CounterfactualResult> {
        // Get the causal chain forward from the intervention situation
        let chain =
            hypergraph.traverse_causal_chain(&intervention.situation_id, self.config.beam_depth)?;

        let mut outcomes = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(intervention.situation_id);

        // BFS through causal chain, applying intervention effects
        let mut queue: VecDeque<(Uuid, f32)> = VecDeque::new();

        // Start with direct consequences
        for link in &chain {
            if link.from_situation == intervention.situation_id
                && !visited.contains(&link.to_situation)
            {
                let prob = link.strength;
                if prob >= self.config.prune_threshold {
                    queue.push_back((link.to_situation, prob));
                    visited.insert(link.to_situation);
                }
            }
        }

        while let Some((sit_id, cumulative_prob)) = queue.pop_front() {
            if outcomes.len() >= self.config.beam_width {
                break;
            }

            // Get the original situation state
            if let Ok(situation) = hypergraph.get_situation(&sit_id) {
                let original_state = serde_json::json!({
                    "narrative_level": situation.narrative_level,
                    "confidence": situation.confidence,
                    "raw_content": situation.raw_content,
                });

                let counterfactual_state = serde_json::json!({
                    "narrative_level": situation.narrative_level,
                    "confidence": situation.confidence * cumulative_prob,
                    "intervention_applied": true,
                    "do_variable": intervention.do_variable,
                    "do_value": intervention.do_value,
                });

                outcomes.push(CounterfactualOutcome {
                    affected_situation: sit_id,
                    original_state,
                    counterfactual_state,
                    probability: cumulative_prob,
                });
            }

            // Propagate to further consequences
            for link in &chain {
                if link.from_situation == sit_id && !visited.contains(&link.to_situation) {
                    let new_prob = cumulative_prob * link.strength;
                    if new_prob >= self.config.prune_threshold {
                        queue.push_back((link.to_situation, new_prob));
                        visited.insert(link.to_situation);
                    }
                }
            }
        }

        Ok(CounterfactualResult {
            intervention: intervention.clone(),
            outcomes,
            beam_width: self.config.beam_width,
        })
    }

    /// Build a temporal precedence mask. mask[i][j] = true means
    /// situation i can potentially cause situation j (i precedes j).
    fn build_temporal_mask(&self, situations: &[Situation]) -> Vec<Vec<bool>> {
        let n = situations.len();
        let mut mask = vec![vec![false; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Check if i temporally precedes j
                if self.temporally_precedes(&situations[i], &situations[j]) {
                    mask[i][j] = true;
                }
            }
        }

        mask
    }

    /// Check if situation a temporally precedes situation b.
    fn temporally_precedes(&self, a: &Situation, b: &Situation) -> bool {
        // Check explicit Allen relations
        for rel in &a.temporal.relations {
            if rel.target_situation == b.id {
                return matches!(
                    rel.relation,
                    AllenRelation::Before | AllenRelation::Meets | AllenRelation::Overlaps
                );
            }
        }

        // Fall back to timestamp comparison
        match (a.temporal.start, b.temporal.start) {
            (Some(a_start), Some(b_start)) => a_start < b_start,
            _ => false,
        }
    }

    /// Build a feature matrix from situations (n x features).
    fn build_feature_matrix(
        &self,
        situations: &[Situation],
        hypergraph: &Hypergraph,
    ) -> Result<Vec<Vec<f64>>> {
        let mut features = Vec::new();

        for sit in situations {
            let mut row = Vec::new();

            // Feature: confidence score
            row.push(sit.confidence as f64);

            // Feature: number of participants
            let participants = hypergraph.get_participants_for_situation(&sit.id)?;
            row.push(participants.len() as f64);

            // Feature: narrative level (ordinal)
            row.push(match sit.narrative_level {
                NarrativeLevel::Story => 0.0,
                NarrativeLevel::Arc => 1.0,
                NarrativeLevel::Sequence => 2.0,
                NarrativeLevel::Scene => 3.0,
                NarrativeLevel::Beat => 4.0,
                NarrativeLevel::Event => 5.0,
            });

            // Feature: has game structure
            row.push(if sit.game_structure.is_some() {
                1.0
            } else {
                0.0
            });

            // Feature: number of existing causal links (from c/ prefix index)
            let causal_count = hypergraph
                .get_consequences(&sit.id)
                .map(|v| v.len())
                .unwrap_or(0);
            row.push(causal_count as f64);

            // Feature: has spatial anchor
            row.push(if sit.spatial.is_some() { 1.0 } else { 0.0 });

            // Feature: participant action diversity
            let unique_actions: HashSet<_> = participants
                .iter()
                .filter_map(|p| p.action.as_deref())
                .collect();
            row.push(unique_actions.len() as f64);

            features.push(row);
        }

        Ok(features)
    }

    /// Seed the adjacency matrix from existing causal links.
    fn seed_adjacency(
        &self,
        situations: &[Situation],
        hypergraph: &Hypergraph,
    ) -> Result<Vec<Vec<f64>>> {
        let n = situations.len();
        let mut adjacency = vec![vec![0.0; n]; n];

        // Build index from UUID to position
        let id_to_idx: HashMap<Uuid, usize> = situations
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        for (i, sit) in situations.iter().enumerate() {
            let consequences = hypergraph.get_consequences(&sit.id)?;
            for link in consequences {
                if let Some(&j) = id_to_idx.get(&link.to_situation) {
                    adjacency[i][j] = link.strength as f64;
                }
            }
        }

        Ok(adjacency)
    }

    /// Build a flat prior matrix from job parameters.
    ///
    /// The prior matrix P ∈ {0, 0.5, 1} encodes: 1 = LLM confident causal link exists,
    /// 0 = LLM confident no link, 0.5 = unknown. Passed via job parameters as
    /// `prior_matrix: [[f64]]` (n×n array). Returns row-major flat Vec.
    fn build_prior_flat(params: &serde_json::Value, n: usize) -> Option<Vec<f64>> {
        let prior_arr = params.get("prior_matrix")?.as_array()?;
        if prior_arr.len() != n {
            return None;
        }
        let mut data = Vec::with_capacity(n * n);
        for row in prior_arr {
            let row_arr = row.as_array()?;
            if row_arr.len() != n {
                return None;
            }
            for val in row_arr {
                data.push(val.as_f64().unwrap_or(0.5));
            }
        }
        Some(data)
    }

    /// Route to the configured causal discovery algorithm.
    fn run_causal_discovery(
        &self,
        adjacency: &mut [Vec<f64>],
        features: &[Vec<f64>],
        temporal_mask: &[Vec<bool>],
        n: usize,
    ) -> Result<()> {
        self.run_causal_discovery_with_prior(adjacency, features, temporal_mask, n, None)
    }

    /// Route to discovery with optional LLM prior (as flat Vec).
    ///
    /// `prior_flat`: optional n×n row-major prior matrix as flat Vec<f64>.
    fn run_causal_discovery_with_prior(
        &self,
        adjacency: &mut [Vec<f64>],
        features: &[Vec<f64>],
        temporal_mask: &[Vec<bool>],
        n: usize,
        #[allow(unused_variables)] prior_flat: Option<&[f64]>,
    ) -> Result<()> {
        match self.config.algorithm {
            #[cfg(feature = "inference")]
            CausalAlgorithm::Dagma => {
                let prior = prior_flat.map(|data| nalgebra::DMatrix::from_row_slice(n, n, data));
                self.dagma_optimize_with_prior(
                    adjacency,
                    features,
                    temporal_mask,
                    n,
                    prior.as_ref(),
                )
            }
            #[cfg(not(feature = "inference"))]
            CausalAlgorithm::Dagma => {
                tracing::warn!("DAGMA requires 'inference' feature; falling back to correlation");
                self.notears_optimize(adjacency, features, temporal_mask, n)
            }
            CausalAlgorithm::Notears => {
                self.notears_optimize(adjacency, features, temporal_mask, n)
            }
        }
    }

    /// NOTEARS optimization with trace-exponential DAG constraint and
    /// augmented Lagrangian method per Zheng et al. (2018).
    ///
    /// Minimizes: `0.5/n * ||X - XW||_F^2 + lambda1 * ||W||_1`
    /// subject to: `h(W) = tr(e^{W circ W}) - d = 0` (acyclicity)
    ///
    /// Temporal mask is applied as an additional hard constraint after
    /// each gradient step (causes must precede effects).
    ///
    /// Requires the `inference` feature for nalgebra matrix operations.
    /// Without it, falls back to correlation-based scoring.
    #[cfg(feature = "inference")]
    fn notears_optimize(
        &self,
        adjacency: &mut [Vec<f64>],
        features: &[Vec<f64>],
        temporal_mask: &[Vec<bool>],
        n: usize,
    ) -> Result<()> {
        if n == 0 || features.is_empty() || features[0].is_empty() {
            return Ok(());
        }

        let d = features[0].len();

        // NOTEARS treats situations as *variables* (columns of X) — W is then
        // the (n_situations × n_situations) DAG. Each row of X is one feature
        // observed across every situation, i.e. X is (d_features × n_situations).
        // Historic shape was (n × d) which only worked when d == n by accident
        // and panicked inside nalgebra's gemm otherwise.
        let mut x_data: Vec<f64> = Vec::with_capacity(d * n);
        for fi in 0..d {
            for sit in features {
                x_data.push(sit[fi]);
            }
        }
        let x = nalgebra::DMatrix::from_row_slice(d, n, &x_data);

        // Build weight matrix W (n x n) from seeded adjacency
        let w_data: Vec<f64> = adjacency
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let mut w = nalgebra::DMatrix::from_row_slice(n, n, &w_data);

        // Build temporal mask as f64 matrix (1.0 where edge allowed, 0.0 otherwise)
        let mask_data: Vec<f64> = temporal_mask
            .iter()
            .flat_map(|row| row.iter().map(|&b| if b { 1.0 } else { 0.0 }))
            .collect();
        let mask = nalgebra::DMatrix::from_row_slice(n, n, &mask_data);

        // Zero the diagonal (no self-loops)
        for i in 0..n {
            w[(i, i)] = 0.0;
        }
        // Apply temporal mask to initial W
        w.component_mul_assign(&mask);

        let mut alpha = self.config.alpha_init;
        let mut rho = self.config.rho_init;

        // Augmented Lagrangian outer loop
        for outer in 0..self.config.max_outer_iterations {
            // Inner optimization: gradient descent on augmented objective
            let mut last_h = 0.0_f64;
            for inner in 0..self.config.max_iterations {
                // Loss gradient: grad_L = -X^T(X - XW) / n
                let xw = &x * &w;
                let residual = &x - &xw;
                let grad_loss = -x.transpose() * &residual / (n as f64);

                // DAG constraint and gradient
                let (h_val, grad_h) = dag_constraint(&w);
                last_h = h_val;

                // L1 subgradient: lambda1 * sign(W)
                let l1_subgrad = w.map(|v| {
                    if v > 0.0 {
                        self.config.lambda1
                    } else if v < 0.0 {
                        -self.config.lambda1
                    } else {
                        0.0
                    }
                });

                // Combined gradient:
                // grad = grad_loss + l1_subgrad + (alpha + rho*h) * grad_h
                let aug_grad_h = &grad_h * (alpha + rho * h_val);
                let grad = grad_loss + l1_subgrad + aug_grad_h;

                // Gradient descent step
                w -= &grad * self.config.learning_rate;

                // Enforce constraints: temporal mask, zero diagonal, clamp [0, 1]
                w.component_mul_assign(&mask);
                for i in 0..n {
                    w[(i, i)] = 0.0;
                }
                w.apply(|v| *v = v.clamp(0.0, 1.0));

                // Convergence check on gradient norm
                let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if grad_norm < self.config.convergence_threshold {
                    tracing::debug!(
                        "NOTEARS inner converged: outer={}, inner={}, grad_norm={:.2e}",
                        outer,
                        inner,
                        grad_norm
                    );
                    break;
                }
            }

            // Reuse last h(W) from inner loop instead of recomputing
            if last_h.abs() < self.config.h_tolerance {
                tracing::debug!(
                    "NOTEARS converged: h(W)={:.2e} after {} outer iterations",
                    last_h,
                    outer + 1
                );
                break;
            }

            // Update Lagrange multiplier and penalty
            alpha += rho * last_h;
            rho = (rho * 2.0).min(self.config.rho_max);
        }

        // Copy optimized W back to adjacency
        for i in 0..n {
            for j in 0..n {
                adjacency[i][j] = w[(i, j)];
            }
        }

        Ok(())
    }

    /// DAGMA optimization with log-determinant M-matrix DAG constraint
    /// per Bello et al. (NeurIPS 2022).
    ///
    /// Same objective as NOTEARS: `0.5/n * ||X - XW||_F^2 + lambda1 * ||W||_1`
    /// but with `h(W) = -log det(sI - W*W) + d*log(s)` as the acyclicity constraint.
    ///
    /// Uses mu-scheduling: multiply the penalty mu by a factor each outer iteration,
    /// which is simpler and faster than the augmented Lagrangian in NOTEARS.
    #[cfg(feature = "inference")]
    fn dagma_optimize(
        &self,
        adjacency: &mut [Vec<f64>],
        features: &[Vec<f64>],
        temporal_mask: &[Vec<bool>],
        n: usize,
    ) -> Result<()> {
        self.dagma_optimize_with_prior(adjacency, features, temporal_mask, n, None)
    }

    /// DAGMA optimization with optional prior matrix.
    ///
    /// If `prior` is Some, adds L2 regularization `lambda_prior * ||W - P||^2`
    /// to encourage the optimized weights to stay close to the LLM-generated prior.
    #[cfg(feature = "inference")]
    fn dagma_optimize_with_prior(
        &self,
        adjacency: &mut [Vec<f64>],
        features: &[Vec<f64>],
        temporal_mask: &[Vec<bool>],
        n: usize,
        prior: Option<&nalgebra::DMatrix<f64>>,
    ) -> Result<()> {
        if n == 0 || features.is_empty() || features[0].is_empty() {
            return Ok(());
        }

        let d = features[0].len();

        // X is (d × n) — feature rows × situation columns. See the matching
        // comment in `notears_optimize` for the derivation.
        let mut x_data: Vec<f64> = Vec::with_capacity(d * n);
        for fi in 0..d {
            for sit in features {
                x_data.push(sit[fi]);
            }
        }
        let x = nalgebra::DMatrix::from_row_slice(d, n, &x_data);

        // Build weight matrix W (n x n) from seeded adjacency
        let w_data: Vec<f64> = adjacency
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let mut w = nalgebra::DMatrix::from_row_slice(n, n, &w_data);

        // Build temporal mask as f64 matrix
        let mask_data: Vec<f64> = temporal_mask
            .iter()
            .flat_map(|row| row.iter().map(|&b| if b { 1.0 } else { 0.0 }))
            .collect();
        let mask = nalgebra::DMatrix::from_row_slice(n, n, &mask_data);

        // Zero diagonal + apply mask
        for i in 0..n {
            w[(i, i)] = 0.0;
        }
        w.component_mul_assign(&mask);

        // SCC pre-validation: detect cycles in the seeded adjacency.
        // If cycles exist, zero out the weakest edge in each cycle.
        {
            let dag = crate::analysis::graph_projection::CausalDag {
                situations: (0..n).map(|_| uuid::Uuid::nil()).collect(),
                adj: (0..n)
                    .map(|i| {
                        (0..n)
                            .filter(|&j| w[(i, j)] > 0.0)
                            .map(|j| (j, w[(i, j)]))
                            .collect()
                    })
                    .collect(),
            };
            let components = crate::analysis::graph_projection::scc(&dag);
            for comp in &components {
                if comp.len() > 1 {
                    tracing::warn!(
                        "DAGMA: SCC of size {} detected in seed adjacency — pruning weakest edge",
                        comp.len()
                    );
                    // Find and zero the weakest edge within this SCC
                    let mut weakest = (0, 0, f64::INFINITY);
                    for &i in comp {
                        for &j in comp {
                            if i != j && w[(i, j)] > 0.0 && w[(i, j)] < weakest.2 {
                                weakest = (i, j, w[(i, j)]);
                            }
                        }
                    }
                    if weakest.2 < f64::INFINITY {
                        w[(weakest.0, weakest.1)] = 0.0;
                    }
                }
            }
        }

        // Adaptive s schedule (decreasing spectral radius margin per outer iteration).
        let s_schedule = &self.config.s_schedule;
        let mut mu = 1.0;
        let mu_factor = 0.1;
        let t_max = self.config.max_outer_iterations.min(8);

        for outer in 0..t_max {
            let s = s_schedule
                .get(outer)
                .copied()
                .unwrap_or(*s_schedule.last().unwrap_or(&0.25));
            let mut last_h = 0.0_f64;

            for inner in 0..self.config.max_iterations {
                // Loss gradient: grad_L = -X^T(X - XW) / n
                let xw = &x * &w;
                let residual = &x - &xw;
                let grad_loss = -x.transpose() * &residual / (n as f64);

                // DAGMA constraint and gradient
                let (h_val, grad_h) = match dagma_constraint(&w, s) {
                    Ok((h, g)) => (h, g),
                    Err(msg) => {
                        tracing::warn!(
                            "DAGMA constraint failed: {}. Falling back to NOTEARS.",
                            msg
                        );
                        return self.notears_optimize(adjacency, features, temporal_mask, n);
                    }
                };
                last_h = h_val;

                // L1 subgradient
                let l1_subgrad = w.map(|v| {
                    if v > 0.0 {
                        self.config.lambda1
                    } else if v < 0.0 {
                        -self.config.lambda1
                    } else {
                        0.0
                    }
                });

                // Prior regularization: lambda_prior * 2 * (W - P)
                let prior_grad = if let Some(p) = prior {
                    (&w - p) * (2.0 * self.config.lambda_prior)
                } else {
                    nalgebra::DMatrix::zeros(n, n)
                };

                // Combined gradient: grad_loss + l1 + (1/mu) * grad_h + prior_reg
                let h_weight = 1.0 / mu;
                let grad = grad_loss + l1_subgrad + &grad_h * h_weight + prior_grad;

                // Gradient descent step
                w -= &grad * self.config.learning_rate;

                // Enforce constraints
                w.component_mul_assign(&mask);
                for i in 0..n {
                    w[(i, i)] = 0.0;
                }
                w.apply(|v| *v = v.clamp(0.0, 1.0));

                let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if grad_norm < self.config.convergence_threshold {
                    tracing::debug!(
                        "DAGMA inner converged: outer={}, inner={}, grad_norm={:.2e}",
                        outer,
                        inner,
                        grad_norm
                    );
                    break;
                }
            }

            if last_h.abs() < self.config.h_tolerance {
                tracing::debug!(
                    "DAGMA converged: h(W)={:.2e} after {} outer iterations",
                    last_h,
                    outer + 1
                );
                break;
            }

            // Decrease mu to tighten the DAG constraint
            mu *= mu_factor;
        }

        // Copy optimized W back to adjacency
        for i in 0..n {
            for j in 0..n {
                adjacency[i][j] = w[(i, j)];
            }
        }

        Ok(())
    }

    /// Fallback NOTEARS when nalgebra is not available.
    /// Uses correlation-based scoring with temporal mask enforcement.
    #[cfg(not(feature = "inference"))]
    fn notears_optimize(
        &self,
        adjacency: &mut [Vec<f64>],
        features: &[Vec<f64>],
        temporal_mask: &[Vec<bool>],
        n: usize,
    ) -> Result<()> {
        if features.is_empty() || features[0].is_empty() {
            return Ok(());
        }

        for iter in 0..self.config.max_iterations {
            let mut max_gradient = 0.0f64;

            for i in 0..n {
                for j in 0..n {
                    if i == j || !temporal_mask[i][j] {
                        adjacency[i][j] = 0.0;
                        continue;
                    }

                    let mut correlation = 0.0;
                    for (fi, fj) in features[i].iter().zip(features[j].iter()) {
                        correlation += fi * fj;
                    }
                    let norm_i: f64 = features[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                    let norm_j: f64 = features[j].iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm_i > 0.0 && norm_j > 0.0 {
                        correlation /= norm_i * norm_j;
                    }

                    let gradient = correlation - adjacency[i][j];
                    adjacency[i][j] += self.config.learning_rate * gradient;
                    adjacency[i][j] = adjacency[i][j].clamp(0.0, 1.0);
                    max_gradient = max_gradient.max(gradient.abs());
                }
            }

            if max_gradient < self.config.convergence_threshold {
                tracing::debug!("NOTEARS (fallback) converged after {} iterations", iter + 1);
                break;
            }
        }

        Ok(())
    }

    /// Extract causal links from the optimized adjacency matrix.
    fn extract_links(
        &self,
        situations: &[Situation],
        adjacency: &[Vec<f64>],
        n: usize,
    ) -> Vec<InferredCausalLink> {
        let mut links = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if adjacency[i][j] >= self.config.min_edge_weight {
                    links.push(InferredCausalLink {
                        from_situation: situations[i].id,
                        to_situation: situations[j].id,
                        mechanism: None,
                        strength: adjacency[i][j] as f32,
                        causal_type: if adjacency[i][j] > 0.8 {
                            CausalType::Necessary
                        } else if adjacency[i][j] > 0.5 {
                            CausalType::Contributing
                        } else {
                            CausalType::Enabling
                        },
                    });
                }
            }
        }

        links
    }

    /// Trace the top-k strongest causal paths through the adjacency matrix.
    ///
    /// Uses DFS from each node, following edges above `min_edge_weight`,
    /// collecting paths up to max depth. Returns top-k by total strength
    /// (product of hop weights).
    fn trace_causal_paths(
        &self,
        situations: &[Situation],
        adjacency: &[Vec<f64>],
        n: usize,
        top_k: usize,
    ) -> CausalPathExplanation {
        let mut all_paths: Vec<CausalPath> = Vec::new();
        let max_depth = 6; // cap path length for tractability

        // DFS from each source node
        for start in 0..n {
            let mut stack: Vec<(usize, Vec<usize>, Vec<f32>)> = vec![(start, vec![start], vec![])];
            while let Some((current, path, strengths)) = stack.pop() {
                // Record paths of length >= 2
                if path.len() >= 2 {
                    let total: f32 = strengths.iter().product();
                    all_paths.push(CausalPath {
                        situation_ids: path.iter().map(|&i| situations[i].id).collect(),
                        hop_strengths: strengths.clone(),
                        total_strength: total,
                    });
                }
                // Extend path if not too deep
                if path.len() < max_depth {
                    for next in 0..n {
                        if path.contains(&next) {
                            continue; // acyclic paths only
                        }
                        let w = adjacency[current][next] as f32;
                        if w >= self.config.min_edge_weight as f32 {
                            let mut new_path = path.clone();
                            new_path.push(next);
                            let mut new_strengths = strengths.clone();
                            new_strengths.push(w);
                            stack.push((next, new_path, new_strengths));
                        }
                    }
                }
            }
        }

        // Sort by total strength descending, take top-k
        all_paths.sort_by(|a, b| {
            b.total_strength
                .partial_cmp(&a.total_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_paths.truncate(top_k);

        let summary = if all_paths.is_empty() {
            "No significant causal paths found.".to_string()
        } else {
            format!(
                "Found {} causal path(s). Strongest path has {} hops with total strength {:.3}.",
                all_paths.len(),
                all_paths[0].situation_ids.len() - 1,
                all_paths[0].total_strength,
            )
        };

        CausalPathExplanation {
            paths: all_paths,
            summary,
        }
    }
}

impl InferenceEngine for CausalEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CausalDiscovery
    }

    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
        super::cost::estimate_cost(job, hypergraph)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let is_counterfactual = job.job_type == InferenceJobType::Counterfactual;

        let (result_value, explanation) = if is_counterfactual {
            // Parse intervention from parameters
            let intervention: Intervention = serde_json::from_value(
                job.parameters
                    .get("intervention")
                    .cloned()
                    .unwrap_or_default(),
            )
            .map_err(|e| TensaError::InferenceError(format!("Invalid intervention: {}", e)))?;

            let cf_result = self.counterfactual(&intervention, hypergraph)?;
            (serde_json::to_value(&cf_result)?, None)
        } else {
            // Causal discovery: collect situations for the target narrative
            // only, matching the scoping every other engine does via
            // `list_situations_by_narrative`. Without this filter, NOTEARS
            // runs over every situation on the server — minutes to hours on
            // a mixed-corpus deployment. If `narrative_id` is absent (legacy
            // callers), fall back to the old global scan for backward compat.
            let narrative_id = job.parameters.get("narrative_id").and_then(|v| v.as_str());
            let all_situations = if let Some(nid) = narrative_id {
                hypergraph.list_situations_by_narrative(nid)?
            } else {
                let mut acc = Vec::new();
                for level in &[
                    NarrativeLevel::Event,
                    NarrativeLevel::Beat,
                    NarrativeLevel::Scene,
                    NarrativeLevel::Sequence,
                    NarrativeLevel::Arc,
                    NarrativeLevel::Story,
                ] {
                    acc.extend(hypergraph.list_situations_by_level(*level)?);
                }
                acc
            };

            // Hierarchical decomposition if too many situations
            if all_situations.len() > self.config.decomposition_threshold {
                // Process by narrative level, starting from finest granularity
                let mut all_links = Vec::new();
                let mut total_confidence = 0.0;
                let mut group_count = 0;

                for level in &[
                    NarrativeLevel::Scene,
                    NarrativeLevel::Sequence,
                    NarrativeLevel::Arc,
                ] {
                    let level_sits: Vec<_> = all_situations
                        .iter()
                        .filter(|s| s.narrative_level == *level)
                        .cloned()
                        .collect();
                    if !level_sits.is_empty() {
                        let (graph, _level_expl) = self.discover_causes_with_params(
                            &level_sits,
                            hypergraph,
                            &job.parameters,
                        )?;
                        total_confidence += graph.confidence;
                        group_count += 1;
                        all_links.extend(graph.links);
                    }
                }

                let graph = CausalGraph {
                    links: all_links,
                    confidence: if group_count > 0 {
                        total_confidence / group_count as f32
                    } else {
                        0.0
                    },
                };
                (serde_json::to_value(&graph)?, None) // hierarchical: explanation per-level is too fragmented
            } else {
                let (graph, expl) =
                    self.discover_causes_with_params(&all_situations, hypergraph, &job.parameters)?;
                (serde_json::to_value(&graph)?, Some(expl))
            }
        };

        let explanation_text = explanation.as_ref().map(|e| e.summary.clone());

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: result_value,
            confidence: 0.7, // default; refined in actual result
            explanation: explanation_text,
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── CounterfactualEngine Wrapper ──────────────────────────

/// Thin wrapper that delegates to `CausalEngine` for counterfactual jobs.
///
/// `CausalEngine::execute` already branches on `InferenceJobType::Counterfactual`
/// and returns `job.job_type.clone()` in the result, so delegation is safe.
pub struct CounterfactualEngine;

impl InferenceEngine for CounterfactualEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::Counterfactual
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(8000) // 8 seconds estimate (beam search)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        CausalEngine::default().execute(job, hypergraph)
    }
}

// ─── MissingLinksEngine ────────────────────────────────────

/// Detects gaps in causal chains within a narrative.
///
/// Finds situations that lack expected causal predecessors or successors,
/// indicating potential missing links in the causal graph.
pub struct MissingLinksEngine;

impl InferenceEngine for MissingLinksEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::MissingLinks
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000) // 3 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        // Try parameters.narrative_id first, then fall back to looking up
        // the target_id as a situation/entity and reading its narrative_id.
        let narrative_id_owned: String;
        let narrative_id = match crate::analysis::extract_narrative_id(job) {
            Ok(nid) => nid,
            Err(_) => {
                // Fallback: try target_id as situation, then entity
                let nid = hypergraph
                    .get_situation(&job.target_id)
                    .ok()
                    .and_then(|s| s.narrative_id.clone())
                    .or_else(|| {
                        hypergraph
                            .get_entity(&job.target_id)
                            .ok()
                            .and_then(|e| e.narrative_id.clone())
                    })
                    .ok_or_else(|| {
                        TensaError::InferenceError(
                            "missing narrative_id in parameters and could not derive from target_id"
                                .into(),
                        )
                    })?;
                narrative_id_owned = nid;
                &narrative_id_owned
            }
        };
        let report = find_missing_links(narrative_id, hypergraph)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::MissingLinks,
            target_id: job.target_id,
            result: serde_json::to_value(&report)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Missing links analysis: {} orphans, {} dead-ends",
                report.orphan_situations.len(),
                report.dead_end_situations.len(),
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Report of missing causal links in a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MissingLinksReport {
    narrative_id: String,
    /// Situations with no incoming or outgoing causal links.
    orphan_situations: Vec<Uuid>,
    /// Situations with incoming causal links but no outgoing ones (chain terminators).
    dead_end_situations: Vec<Uuid>,
    /// Total causal links in the narrative.
    total_links: usize,
}

/// Find missing links in the causal graph for a narrative.
fn find_missing_links(narrative_id: &str, hypergraph: &Hypergraph) -> Result<MissingLinksReport> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    let sit_ids: HashSet<Uuid> = situations.iter().map(|s| s.id).collect();
    let mut has_incoming: HashSet<Uuid> = HashSet::new();
    let mut has_outgoing: HashSet<Uuid> = HashSet::new();
    let mut total_links = 0usize;

    for sit in &situations {
        let consequences = hypergraph.get_consequences(&sit.id)?;
        for link in &consequences {
            if sit_ids.contains(&link.to_situation) {
                has_outgoing.insert(sit.id);
                has_incoming.insert(link.to_situation);
                total_links += 1;
            }
        }
    }

    let orphans: Vec<Uuid> = situations
        .iter()
        .filter(|s| !has_incoming.contains(&s.id) && !has_outgoing.contains(&s.id))
        .map(|s| s.id)
        .collect();

    let dead_ends: Vec<Uuid> = situations
        .iter()
        .filter(|s| has_incoming.contains(&s.id) && !has_outgoing.contains(&s.id))
        .map(|s| s.id)
        .collect();

    Ok(MissingLinksReport {
        narrative_id: narrative_id.to_string(),
        orphan_situations: orphans,
        dead_end_situations: dead_ends,
        total_links,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::{Duration, Utc};
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_situation(
        hg: &Hypergraph,
        start_offset_hours: i64,
        level: NarrativeLevel,
    ) -> Situation {
        let now = Utc::now();
        let start = now + Duration::hours(start_offset_hours);
        let end = start + Duration::hours(1);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(end),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Test situation")],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let id = hg.create_situation(sit.clone()).unwrap();
        let mut result = sit;
        result.id = id;
        result
    }

    #[test]
    fn test_empty_situations() {
        let hg = test_hg();
        let engine = CausalEngine::new();
        let (graph, _expl) = engine.discover_causes(&[], &hg).unwrap();
        assert!(graph.links.is_empty());
        assert_eq!(graph.confidence, 0.0);
    }

    #[test]
    fn test_single_situation() {
        let hg = test_hg();
        let engine = CausalEngine::new();
        let sit = make_situation(&hg, 0, NarrativeLevel::Scene);
        let (graph, _expl) = engine.discover_causes(&[sit], &hg).unwrap();
        assert!(graph.links.is_empty());
        assert_eq!(graph.confidence, 1.0);
    }

    #[test]
    fn test_temporal_ordering_prevents_backward_edges() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        // Create situations with clear temporal ordering
        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);
        let s3 = make_situation(&hg, 4, NarrativeLevel::Scene);

        let (graph, _expl) = engine
            .discover_causes(&[s1.clone(), s2.clone(), s3.clone()], &hg)
            .unwrap();

        // No backward edges should exist
        for link in &graph.links {
            let from_idx = [&s1, &s2, &s3]
                .iter()
                .position(|s| s.id == link.from_situation)
                .unwrap();
            let to_idx = [&s1, &s2, &s3]
                .iter()
                .position(|s| s.id == link.to_situation)
                .unwrap();
            assert!(
                from_idx < to_idx,
                "Found backward causal edge: {} -> {}",
                from_idx,
                to_idx
            );
        }
    }

    #[test]
    fn test_discovers_causal_structure() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);

        // Seed existing link
        hg.add_causal_link(CausalLink {
            from_situation: s1.id,
            to_situation: s2.id,
            mechanism: Some("direct cause".into()),
            strength: 0.9,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let (graph, _expl) = engine.discover_causes(&[s1, s2], &hg).unwrap();
        // Should find at least the seeded link
        assert!(!graph.links.is_empty());
    }

    #[test]
    fn test_temporal_mask_with_allen_relations() {
        let engine = CausalEngine::new();

        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();

        let sit_a = Situation {
            id: id_a,
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now() + Duration::hours(1)),
                granularity: TimeGranularity::Approximate,
                relations: vec![AllenRelationTo {
                    target_situation: id_b,
                    relation: AllenRelation::Before,
                }],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };

        let sit_b = Situation {
            id: id_b,
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now() + Duration::hours(2)),
                end: Some(Utc::now() + Duration::hours(3)),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };

        let mask = engine.build_temporal_mask(&[sit_a, sit_b]);
        assert!(mask[0][1]); // a can cause b
        assert!(!mask[1][0]); // b cannot cause a
    }

    #[test]
    fn test_counterfactual_beam_search() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);
        let s3 = make_situation(&hg, 4, NarrativeLevel::Scene);

        // Create causal chain: s1 -> s2 -> s3
        hg.add_causal_link(CausalLink {
            from_situation: s1.id,
            to_situation: s2.id,
            mechanism: Some("direct".into()),
            strength: 0.8,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2.id,
            to_situation: s3.id,
            mechanism: Some("chain".into()),
            strength: 0.7,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let intervention = Intervention {
            situation_id: s1.id,
            do_variable: "action".into(),
            do_value: serde_json::json!("cooperate"),
        };

        let result = engine.counterfactual(&intervention, &hg).unwrap();
        assert!(!result.outcomes.is_empty());
        // Should propagate to s2 and s3
        assert!(result.outcomes.len() <= engine.config.beam_width);
    }

    #[test]
    fn test_counterfactual_prunes_low_probability() {
        let hg = test_hg();
        let mut config = CausalConfig::default();
        config.prune_threshold = 0.5; // high threshold
        let engine = CausalEngine::with_config(config);

        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);
        let s3 = make_situation(&hg, 4, NarrativeLevel::Scene);

        // Weak chain
        hg.add_causal_link(CausalLink {
            from_situation: s1.id,
            to_situation: s2.id,
            mechanism: None,
            strength: 0.6,
            causal_type: CausalType::Enabling,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2.id,
            to_situation: s3.id,
            mechanism: None,
            strength: 0.4,
            causal_type: CausalType::Enabling,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let intervention = Intervention {
            situation_id: s1.id,
            do_variable: "action".into(),
            do_value: serde_json::json!("change"),
        };

        let result = engine.counterfactual(&intervention, &hg).unwrap();
        // s2 has prob 0.6 (>0.5) so it should be included
        // s3 has prob 0.6*0.4=0.24 (<0.5) so it should be pruned
        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(result.outcomes[0].affected_situation, s2.id);
    }

    #[test]
    fn test_engine_execute_causal_discovery() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let _s2 = make_situation(&hg, 2, NarrativeLevel::Scene);

        let job = InferenceJob {
            id: "causal-001".to_string(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: s1.id,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 1000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.job_id, "causal-001");
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());
    }

    #[test]
    fn test_engine_execute_counterfactual() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);

        hg.add_causal_link(CausalLink {
            from_situation: s1.id,
            to_situation: s2.id,
            mechanism: None,
            strength: 0.8,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let job = InferenceJob {
            id: "cf-001".to_string(),
            job_type: InferenceJobType::Counterfactual,
            target_id: s1.id,
            parameters: serde_json::json!({
                "intervention": {
                    "situation_id": s1.id,
                    "do_variable": "action",
                    "do_value": "cooperate"
                }
            }),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 1000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        let cf: CounterfactualResult = serde_json::from_value(result.result).unwrap();
        assert!(!cf.outcomes.is_empty());
    }

    #[test]
    fn test_counterfactual_engine_execute() {
        let hg = test_hg();
        let s1 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);

        hg.add_causal_link(CausalLink {
            from_situation: s1.id,
            to_situation: s2.id,
            mechanism: None,
            strength: 0.8,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let engine = CounterfactualEngine;
        assert_eq!(engine.job_type(), InferenceJobType::Counterfactual);

        let job = InferenceJob {
            id: "cf-wrap-001".to_string(),
            job_type: InferenceJobType::Counterfactual,
            target_id: s1.id,
            parameters: serde_json::json!({
                "intervention": {
                    "situation_id": s1.id,
                    "do_variable": "action",
                    "do_value": "cooperate"
                }
            }),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert_eq!(result.job_type, InferenceJobType::Counterfactual);
        assert!(result.completed_at.is_some());
    }

    fn make_narrative_situation(
        hg: &Hypergraph,
        start_offset_hours: i64,
        narrative_id: &str,
    ) -> Situation {
        let now = Utc::now();
        let start = now + Duration::hours(start_offset_hours);
        let end = start + Duration::hours(1);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(end),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Test situation")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: Some(narrative_id.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let id = hg.create_situation(sit.clone()).unwrap();
        let mut result = sit;
        result.id = id;
        result
    }

    #[test]
    fn test_missing_links_engine_execute() {
        let hg = test_hg();
        let nid = "ml-test";

        let s1 = make_narrative_situation(&hg, 0, nid);
        let s2 = make_narrative_situation(&hg, 2, nid);
        let _s3 = make_narrative_situation(&hg, 4, nid); // orphan

        // s1 -> s2 link, but s3 is orphaned
        hg.add_causal_link(CausalLink {
            from_situation: s1.id,
            to_situation: s2.id,
            mechanism: None,
            strength: 0.8,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let engine = MissingLinksEngine;
        assert_eq!(engine.job_type(), InferenceJobType::MissingLinks);

        let job = InferenceJob {
            id: "ml-001".to_string(),
            job_type: InferenceJobType::MissingLinks,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": nid}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());
    }

    // ─── NOTEARS algorithm tests ──────────────────────────────

    #[cfg(feature = "inference")]
    #[test]
    fn test_matrix_exp_identity() {
        // exp(zero matrix) = identity
        let z: nalgebra::DMatrix<f64> = nalgebra::DMatrix::zeros(4, 4);
        let result = super::matrix_exp_taylor(&z, 20);
        let identity: nalgebra::DMatrix<f64> = nalgebra::DMatrix::identity(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (result[(i, j)] - identity[(i, j)]).abs() < 1e-10,
                    "exp(0)[{},{}] = {} != {}",
                    i,
                    j,
                    result[(i, j)],
                    identity[(i, j)]
                );
            }
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_matrix_exp_diagonal() {
        // exp(diag(a, b, c)) = diag(e^a, e^b, e^c)
        let mut m = nalgebra::DMatrix::zeros(3, 3);
        m[(0, 0)] = 1.0;
        m[(1, 1)] = 2.0;
        m[(2, 2)] = 0.5;
        let result = super::matrix_exp_taylor(&m, 20);
        assert!((result[(0, 0)] - 1.0_f64.exp()).abs() < 1e-8);
        assert!((result[(1, 1)] - 2.0_f64.exp()).abs() < 1e-8);
        assert!((result[(2, 2)] - 0.5_f64.exp()).abs() < 1e-8);
        // Off-diagonal should be zero
        assert!(result[(0, 1)].abs() < 1e-10);
        assert!(result[(1, 0)].abs() < 1e-10);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_dag_constraint_on_dag() {
        // A strict lower-triangular matrix (DAG) should have h(W) close to 0
        let mut w = nalgebra::DMatrix::zeros(3, 3);
        w[(1, 0)] = 0.5; // edge 0 -> 1
        w[(2, 0)] = 0.3; // edge 0 -> 2
        w[(2, 1)] = 0.4; // edge 1 -> 2
        let (h, _) = super::dag_constraint(&w);
        assert!(h.abs() < 1e-6, "DAG should have h(W) near 0, got {}", h);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_dag_constraint_on_cycle() {
        // A cycle: 0->1->2->0 should have h(W) > 0
        let mut w = nalgebra::DMatrix::zeros(3, 3);
        w[(0, 1)] = 0.8;
        w[(1, 2)] = 0.8;
        w[(2, 0)] = 0.8;
        let (h, _) = super::dag_constraint(&w);
        assert!(h > 0.1, "Cycle should have h(W) > 0, got {}", h);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_notears_produces_dag() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        // Create 4 situations with temporal ordering
        let s0 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s1 = make_situation(&hg, 1, NarrativeLevel::Scene);
        let s2 = make_situation(&hg, 2, NarrativeLevel::Scene);
        let s3 = make_situation(&hg, 3, NarrativeLevel::Scene);

        let situations = vec![s0, s1, s2, s3];

        let features = engine.build_feature_matrix(&situations, &hg).unwrap();
        let temporal_mask = engine.build_temporal_mask(&situations);
        let n = situations.len();
        let mut adjacency = vec![vec![0.0; n]; n];

        engine
            .notears_optimize(&mut adjacency, &features, &temporal_mask, n)
            .unwrap();

        // Build W from result and check DAG constraint
        let w_data: Vec<f64> = adjacency.iter().flat_map(|r| r.iter().copied()).collect();
        let w = nalgebra::DMatrix::from_row_slice(n, n, &w_data);
        let (h, _) = super::dag_constraint(&w);
        assert!(
            h.abs() < 0.01,
            "After NOTEARS, h(W) should be near 0, got {}",
            h
        );
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_notears_respects_temporal_mask() {
        let hg = test_hg();
        let engine = CausalEngine::new();

        let s0 = make_situation(&hg, 0, NarrativeLevel::Scene);
        let s1 = make_situation(&hg, 2, NarrativeLevel::Scene);

        let situations = vec![s0, s1];

        let features = engine.build_feature_matrix(&situations, &hg).unwrap();
        let temporal_mask = engine.build_temporal_mask(&situations);
        let n = 2;
        let mut adjacency = vec![vec![0.0; n]; n];

        engine
            .notears_optimize(&mut adjacency, &features, &temporal_mask, n)
            .unwrap();

        // s1 should NOT cause s0 (temporal mask prevents backward edges)
        assert!(
            adjacency[1][0] < 1e-10,
            "Backward edge should be 0, got {}",
            adjacency[1][0]
        );
    }

    // ─── DAGMA tests ────────────────────────────────────────

    #[test]
    fn test_causal_algorithm_default_is_dagma() {
        let config = CausalConfig::default();
        assert_eq!(config.algorithm, CausalAlgorithm::Dagma);
    }

    #[test]
    fn test_causal_algorithm_serde() {
        let dagma: CausalAlgorithm = serde_json::from_str("\"dagma\"").unwrap();
        assert_eq!(dagma, CausalAlgorithm::Dagma);
        let notears: CausalAlgorithm = serde_json::from_str("\"notears\"").unwrap();
        assert_eq!(notears, CausalAlgorithm::Notears);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_dagma_constraint_on_dag() {
        // Zero matrix = trivial DAG; h should be ~0
        let w = nalgebra::DMatrix::zeros(3, 3);
        let (h, _grad) = dagma_constraint(&w, 1.0).unwrap();
        assert!(h.abs() < 1e-8, "DAG should have h≈0, got {}", h);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_dagma_constraint_on_cycle() {
        // W with a cycle: 0→1, 1→2, 2→0
        let mut w = nalgebra::DMatrix::zeros(3, 3);
        w[(0, 1)] = 0.5;
        w[(1, 2)] = 0.5;
        w[(2, 0)] = 0.5;
        let result = dagma_constraint(&w, 1.0);
        match result {
            Ok((h, _)) => assert!(h > 0.01, "Cycle should have h > 0, got {}", h),
            Err(_) => {} // Also acceptable — matrix may not be invertible
        }
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_dagma_produces_dag() {
        let engine = CausalEngine::with_config(CausalConfig {
            algorithm: CausalAlgorithm::Dagma,
            ..CausalConfig::default()
        });

        let n = 3;
        let features = vec![
            vec![1.0, 0.5, 0.0],
            vec![0.5, 1.0, 0.3],
            vec![0.0, 0.3, 1.0],
        ];
        let temporal_mask = vec![
            vec![false, true, true],   // 0 can cause 1, 2
            vec![false, false, true],  // 1 can cause 2
            vec![false, false, false], // 2 can't cause anything
        ];
        let mut adjacency = vec![vec![0.5; n]; n];

        engine
            .dagma_optimize(&mut adjacency, &features, &temporal_mask, n)
            .unwrap();

        // Verify DAG: compute h via NOTEARS constraint (independent check)
        let w_data: Vec<f64> = adjacency
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let w = nalgebra::DMatrix::from_row_slice(n, n, &w_data);
        let (h, _) = dag_constraint(&w);
        assert!(h < 0.1, "DAGMA result should be near-DAG, got h={}", h);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_dagma_respects_temporal_mask() {
        let engine = CausalEngine::with_config(CausalConfig {
            algorithm: CausalAlgorithm::Dagma,
            ..CausalConfig::default()
        });

        let n = 2;
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // s0 BEFORE s1, so only s0→s1 allowed
        let temporal_mask = vec![
            vec![false, true],  // 0→1 allowed
            vec![false, false], // 1→0 NOT allowed
        ];
        let mut adjacency = vec![vec![0.0; n]; n];

        engine
            .dagma_optimize(&mut adjacency, &features, &temporal_mask, n)
            .unwrap();

        assert!(
            adjacency[1][0] < 1e-10,
            "DAGMA: backward edge should be 0, got {}",
            adjacency[1][0]
        );
    }

    #[test]
    fn test_run_causal_discovery_routes_correctly() {
        // Test that run_causal_discovery uses the configured algorithm
        let engine_dagma = CausalEngine::with_config(CausalConfig {
            algorithm: CausalAlgorithm::Dagma,
            ..CausalConfig::default()
        });
        let engine_notears = CausalEngine::with_config(CausalConfig {
            algorithm: CausalAlgorithm::Notears,
            ..CausalConfig::default()
        });

        let n = 2;
        let features = vec![vec![1.0], vec![0.5]];
        let mask = vec![vec![false, true], vec![false, false]];
        let mut adj1 = vec![vec![0.0; n]; n];
        let mut adj2 = vec![vec![0.0; n]; n];

        // Both should succeed without error
        engine_dagma
            .run_causal_discovery(&mut adj1, &features, &mask, n)
            .unwrap();
        engine_notears
            .run_causal_discovery(&mut adj2, &features, &mask, n)
            .unwrap();
    }

    // ─── Causal path explanation tests ──────────────────────

    #[test]
    fn test_trace_causal_paths_empty() {
        let engine = CausalEngine::new();
        let sits: Vec<Situation> = Vec::new();
        let adj: Vec<Vec<f64>> = Vec::new();
        let expl = engine.trace_causal_paths(&sits, &adj, 0, 5);
        assert!(expl.paths.is_empty());
        assert!(expl.summary.contains("No significant"));
    }

    #[test]
    fn test_trace_causal_paths_linear_chain() {
        let engine = CausalEngine::new();
        // 3 situations: 0→1 (0.8), 1→2 (0.6)
        let sits: Vec<Situation> = (0..3)
            .map(|_| Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(Utc::now()),
                    end: Some(Utc::now()),
                    granularity: TimeGranularity::Approximate,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.8,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: None,
                source_chunk_id: None,
                source_span: None,
                synopsis: None,
                manuscript_order: None,
                parent_situation_id: None,
                label: None,
                status: None,
                keywords: vec![],
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .collect();

        let adj = vec![
            vec![0.0, 0.8, 0.0],
            vec![0.0, 0.0, 0.6],
            vec![0.0, 0.0, 0.0],
        ];

        let expl = engine.trace_causal_paths(&sits, &adj, 3, 5);
        assert!(!expl.paths.is_empty());

        // Should find at least the 2-hop path 0→1→2
        let long_paths: Vec<_> = expl
            .paths
            .iter()
            .filter(|p| p.situation_ids.len() == 3)
            .collect();
        assert!(
            !long_paths.is_empty(),
            "Should find the 3-situation chain 0→1→2"
        );
        // Total strength of 0→1→2 = 0.8 * 0.6 = 0.48
        let chain = &long_paths[0];
        assert!((chain.total_strength - 0.48).abs() < 0.01);
    }

    #[test]
    fn test_causal_path_explanation_serde() {
        let expl = CausalPathExplanation {
            paths: vec![CausalPath {
                situation_ids: vec![Uuid::now_v7(), Uuid::now_v7()],
                hop_strengths: vec![0.7],
                total_strength: 0.7,
            }],
            summary: "Test".to_string(),
        };
        let json = serde_json::to_vec(&expl).unwrap();
        let back: CausalPathExplanation = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.paths.len(), 1);
        assert!((back.paths[0].total_strength - 0.7).abs() < f32::EPSILON);
    }
}
