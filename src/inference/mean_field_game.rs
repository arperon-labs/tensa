//! Mean Field Game (MFG) engine for large-population strategic analysis.
//!
//! Extends game-theoretic analysis beyond the N≤4 QRE solver by replacing
//! individual opponent tracking with a population distribution μ(a). Each
//! representative agent best-responds to the distribution, and the algorithm
//! iterates until a fixed-point equilibrium is reached.
//!
//! Key idea: EU(a_i, μ) = base_payoff(a_i) + Σ_a' μ(a') × coupling(a_i, a')
//!
//! Reference: Mean Field Games (Lasry & Lions 2007, Huang et al. 2006)

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::analysis::analysis_key;
use crate::error::{Result, TensaError};
use crate::hypergraph::keys::ANALYSIS_MEAN_FIELD;
use crate::hypergraph::Hypergraph;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

// ─── Configuration ─────────────────────────────────────────

/// Configuration for the Mean Field Game solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanFieldConfig {
    /// Maximum fixed-point iterations.
    pub max_iterations: usize,
    /// Convergence threshold for distribution delta.
    pub convergence_threshold: f64,
    /// Rationality parameter (higher = more rational, same as QRE lambda).
    pub lambda: f64,
    /// Minimum participants to use MFG (below this, result notes insufficient data).
    pub min_participants: usize,
    /// Scales the interaction term in mean field utility.
    pub coupling_strength: f64,
}

impl Default for MeanFieldConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            convergence_threshold: 1e-5,
            lambda: 1.0,
            min_participants: 5,
            coupling_strength: 0.1,
        }
    }
}

// ─── Result Types ──────────────────────────────────────────

/// Full result of a Mean Field Game analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanFieldResult {
    pub situation_id: String,
    pub num_participants: usize,
    pub equilibrium: MeanFieldEquilibrium,
    pub convergence: ConvergenceInfo,
    pub coupling_matrix: Vec<CouplingEntry>,
}

/// The converged population distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanFieldEquilibrium {
    pub distribution: Vec<ActionShare>,
    pub total_welfare: f64,
    pub entropy: f64,
    pub stability: StabilityType,
}

/// A single action's share in the equilibrium distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionShare {
    pub action: String,
    pub probability: f64,
    pub expected_payoff: f64,
}

/// Convergence diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_delta: f64,
}

/// Coupling between action pairs (how action_i's payoff is affected by action_j's prevalence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingEntry {
    pub action_i: String,
    pub action_j: String,
    pub coupling: f64,
}

/// Stability classification of the equilibrium.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilityType {
    Stable,
    Unstable,
    Saddle,
}

// ─── Engine ────────────────────────────────────────────────

/// Mean Field Game inference engine.
pub struct MeanFieldGameEngine {
    config: MeanFieldConfig,
}

impl Default for MeanFieldGameEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MeanFieldGameEngine {
    pub fn new() -> Self {
        Self {
            config: MeanFieldConfig::default(),
        }
    }

    pub fn with_config(config: MeanFieldConfig) -> Self {
        Self { config }
    }
}

impl InferenceEngine for MeanFieldGameEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::MeanFieldGame
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(4000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let situation = hypergraph.get_situation(&job.target_id)?;
        let participants = hypergraph.get_participants_for_situation(&situation.id)?;

        // Parse config overrides
        let mut config = self.config.clone();
        if let Some(v) = job.parameters.get("lambda").and_then(|v| v.as_f64()) {
            config.lambda = v;
        }
        if let Some(v) = job
            .parameters
            .get("min_participants")
            .and_then(|v| v.as_u64())
        {
            config.min_participants = v as usize;
        }
        if let Some(v) = job
            .parameters
            .get("coupling_strength")
            .and_then(|v| v.as_f64())
        {
            config.coupling_strength = v;
        }

        if participants.is_empty() {
            return Err(TensaError::InferenceError(
                "No participants in situation".into(),
            ));
        }

        // Clamp lambda to positive range
        if config.lambda <= 0.0 {
            config.lambda = 1.0;
        }

        let result = solve_situation(&participants, &config);

        let result_data = MeanFieldResult {
            situation_id: job.target_id.to_string(),
            num_participants: participants.len(),
            equilibrium: result.0,
            convergence: result.1,
            coupling_matrix: result.2,
        };

        let result_value = serde_json::to_value(&result_data)?;

        // Persist to KV
        let key = analysis_key(ANALYSIS_MEAN_FIELD, &[&job.target_id.to_string()]);
        hypergraph
            .store()
            .put(&key, result_value.to_string().as_bytes())?;

        let n_actions = result_data.equilibrium.distribution.len();
        let converged = result_data.convergence.converged;
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::MeanFieldGame,
            target_id: job.target_id,
            result: result_value,
            confidence: if converged { 0.8 } else { 0.4 },
            explanation: Some(format!(
                "Mean Field Game: {} participants, {} actions, {} iterations ({})",
                result_data.num_participants,
                n_actions,
                result_data.convergence.iterations,
                if converged {
                    "converged"
                } else {
                    "did not converge"
                },
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Core Algorithm ────────────────────────────────────────

/// Solve a situation's strategic interaction via Mean Field approximation.
fn solve_situation(
    participants: &[Participation],
    config: &MeanFieldConfig,
) -> (MeanFieldEquilibrium, ConvergenceInfo, Vec<CouplingEntry>) {
    // Extract unique actions and their base payoffs
    let (actions, base_payoffs) = extract_actions_and_payoffs(participants);

    if actions.is_empty() {
        return (
            MeanFieldEquilibrium {
                distribution: vec![],
                total_welfare: 0.0,
                entropy: 0.0,
                stability: StabilityType::Stable,
            },
            ConvergenceInfo {
                converged: true,
                iterations: 0,
                final_delta: 0.0,
            },
            vec![],
        );
    }

    // Build coupling matrix from observed payoff interactions
    let coupling = build_coupling_matrix(&actions, participants);

    // Solve mean field fixed-point
    let (distribution, convergence) = solve_mean_field(&actions, &base_payoffs, &coupling, config);

    // Compute equilibrium properties
    let total_welfare: f64 = distribution
        .iter()
        .map(|s| s.probability * s.expected_payoff)
        .sum();

    let entropy = compute_entropy(&distribution);
    let stability = classify_stability(&distribution, &coupling, config.lambda);

    let equilibrium = MeanFieldEquilibrium {
        distribution,
        total_welfare,
        entropy,
        stability,
    };

    // Convert coupling matrix to flat entries
    let coupling_entries = flatten_coupling(&actions, &coupling);

    (equilibrium, convergence, coupling_entries)
}

/// Extract unique actions and their average base payoffs from participants.
fn extract_actions_and_payoffs(participants: &[Participation]) -> (Vec<String>, Vec<f64>) {
    let mut action_payoffs: HashMap<String, (f64, usize)> = HashMap::new();

    for p in participants {
        let action = match &p.action {
            Some(a) if !a.is_empty() => a.clone(),
            _ => continue,
        };

        let payoff = p.payoff.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0);

        let entry = action_payoffs.entry(action).or_insert((0.0, 0));
        entry.0 += payoff;
        entry.1 += 1;
    }

    let mut actions: Vec<String> = action_payoffs.keys().cloned().collect();
    actions.sort(); // deterministic ordering

    let base_payoffs: Vec<f64> = actions
        .iter()
        .map(|a| {
            let (sum, count) = action_payoffs[a];
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        })
        .collect();

    (actions, base_payoffs)
}

/// Build coupling matrix C[i][j] = how action j's prevalence affects action i's payoff.
///
/// Positive coupling = complementary actions (both benefit from coexistence).
/// Negative coupling = substitution (competing for same resource).
///
/// Derived from co-occurrence payoff patterns: when two participants with different
/// actions both have positive payoffs, their actions are complementary.
fn build_coupling_matrix(actions: &[String], participants: &[Participation]) -> Vec<Vec<f64>> {
    let n = actions.len();
    let mut coupling = vec![vec![0.0_f64; n]; n];
    let mut counts = vec![vec![0_usize; n]; n];

    let action_idx: HashMap<&str, usize> = actions
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();

    // Collect action-payoff pairs
    let action_payoff_pairs: Vec<(usize, f64)> = participants
        .iter()
        .filter_map(|p| {
            let action = p.action.as_ref()?;
            let idx = *action_idx.get(action.as_str())?;
            let payoff = p.payoff.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0);
            Some((idx, payoff))
        })
        .collect();

    // For each pair of participants, compute coupling from their payoff interaction.
    // Same-action pairs: positive if both benefit, negative if both suffer.
    // Cross-action pairs: positive if complementary, negative if competitive.
    for i in 0..action_payoff_pairs.len() {
        for j in (i + 1)..action_payoff_pairs.len() {
            let (ai, pi) = action_payoff_pairs[i];
            let (aj, pj) = action_payoff_pairs[j];

            // For same-action pairs, use the average payoff sign (both negative = negative coupling).
            // For cross-action pairs, product captures complementarity vs competition.
            let interaction = if ai == aj {
                // Same action: sign = positive if avg payoff positive, else negative
                let avg = (pi + pj) / 2.0;
                avg.abs() * avg.signum()
            } else {
                pi * pj
            };

            coupling[ai][aj] += interaction;
            coupling[aj][ai] += interaction;
            counts[ai][aj] += 1;
            counts[aj][ai] += 1;
        }
    }

    // Normalize by pair count
    for i in 0..n {
        for j in 0..n {
            if counts[i][j] > 0 {
                coupling[i][j] /= counts[i][j] as f64;
            }
        }
    }

    coupling
}

/// Solve mean field fixed-point: find distribution μ where μ = softmax(λ · EU(·, μ)).
fn solve_mean_field(
    actions: &[String],
    base_payoffs: &[f64],
    coupling: &[Vec<f64>],
    config: &MeanFieldConfig,
) -> (Vec<ActionShare>, ConvergenceInfo) {
    let n = actions.len();

    // Initialize: uniform distribution
    let mut mu: Vec<f64> = vec![1.0 / n as f64; n];

    let mut iterations = 0;
    let mut final_delta = 0.0;
    let mut converged = false;
    let mut last_eu: Vec<f64> = vec![0.0; n];

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        let eu: Vec<f64> = (0..n)
            .map(|i| compute_mean_field_utility(i, &mu, base_payoffs, coupling, config))
            .collect();
        last_eu = eu.clone();
        let eu = last_eu.as_slice();

        // Softmax best response: μ_new(a) = exp(λ·EU(a,μ)) / Σ exp(λ·EU(a',μ))
        let max_eu = eu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = eu
            .iter()
            .map(|e| (config.lambda * (e - max_eu)).exp())
            .collect();
        let sum_exp: f64 = exp_vals.iter().sum();

        let mu_new: Vec<f64> = if sum_exp > 0.0 {
            exp_vals.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / n as f64; n]
        };

        // Convergence check
        final_delta = mu
            .iter()
            .zip(mu_new.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        mu = mu_new;

        if final_delta < config.convergence_threshold {
            converged = true;
            break;
        }
    }

    // Build result using last iteration's utilities (avoid recomputation)
    let distribution: Vec<ActionShare> = actions
        .iter()
        .enumerate()
        .map(|(i, action)| ActionShare {
            action: action.clone(),
            probability: mu[i],
            expected_payoff: last_eu[i],
        })
        .collect();

    let convergence = ConvergenceInfo {
        converged,
        iterations,
        final_delta,
    };

    (distribution, convergence)
}

/// EU(a_i, μ) = base_payoff(a_i) + coupling_strength × Σ_j μ(j) × coupling[i][j]
fn compute_mean_field_utility(
    action_idx: usize,
    distribution: &[f64],
    base_payoffs: &[f64],
    coupling: &[Vec<f64>],
    config: &MeanFieldConfig,
) -> f64 {
    let base = base_payoffs[action_idx];
    let interaction: f64 = distribution
        .iter()
        .enumerate()
        .map(|(j, mu_j)| mu_j * coupling[action_idx][j])
        .sum();

    base + config.coupling_strength * interaction
}

/// Shannon entropy of the distribution: -Σ μ(a) ln μ(a).
fn compute_entropy(distribution: &[ActionShare]) -> f64 {
    -distribution
        .iter()
        .map(|s| {
            if s.probability > 0.0 {
                s.probability * s.probability.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// Classify equilibrium stability via Jacobian diagonal dominance.
///
/// For a softmax fixed-point, the Jacobian ∂μ_new/∂μ has entries involving
/// λ × coupling. If all coupling eigenvalues have magnitude < 1/λ, the
/// equilibrium is stable (contraction mapping).
fn classify_stability(
    distribution: &[ActionShare],
    coupling: &[Vec<f64>],
    lambda: f64,
) -> StabilityType {
    if distribution.is_empty() || coupling.is_empty() {
        return StabilityType::Stable;
    }

    let n = distribution.len();

    // Approximate: check if max |λ × coupling[i][j]| < 1 for all i,j
    // This is a sufficient condition for contraction (Banach fixed-point theorem)
    let mut max_coupling_magnitude = 0.0_f64;
    let mut has_positive = false;
    let mut has_negative = false;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mag = (lambda * coupling[i][j]).abs();
                max_coupling_magnitude = max_coupling_magnitude.max(mag);
                if coupling[i][j] > 0.0 {
                    has_positive = true;
                }
                if coupling[i][j] < 0.0 {
                    has_negative = true;
                }
            }
        }
    }

    if max_coupling_magnitude < 1.0 {
        StabilityType::Stable
    } else if has_positive && has_negative {
        StabilityType::Saddle
    } else {
        StabilityType::Unstable
    }
}

/// Flatten coupling matrix to vector of entries for serialization.
fn flatten_coupling(actions: &[String], coupling: &[Vec<f64>]) -> Vec<CouplingEntry> {
    let mut entries = Vec::new();
    for (i, ai) in actions.iter().enumerate() {
        for (j, aj) in actions.iter().enumerate() {
            if coupling[i][j].abs() > 1e-10 {
                entries.push(CouplingEntry {
                    action_i: ai.clone(),
                    action_j: aj.clone(),
                    coupling: coupling[i][j],
                });
            }
        }
    }
    entries
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::make_hg;
    use uuid::Uuid;

    fn make_participation(action: &str, payoff: f64) -> Participation {
        Participation {
            entity_id: Uuid::now_v7(),
            situation_id: Uuid::now_v7(),
            role: Role::Protagonist,
            info_set: None,
            action: Some(action.to_string()),
            payoff: Some(serde_json::json!(payoff)),
            seq: 0,
        }
    }

    fn make_situation(hg: &Hypergraph) -> Uuid {
        crate::analysis::test_helpers::add_situation(hg, "test")
    }

    #[test]
    fn test_empty_participants() {
        let hg = make_hg();
        let sid = make_situation(&hg);
        let engine = MeanFieldGameEngine::new();
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::MeanFieldGame,
            target_id: sid,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let result = engine.execute(&job, &hg);
        assert!(result.is_err(), "Should error with no participants");
    }

    #[test]
    fn test_below_threshold() {
        // With only participants that have no actions, should produce empty equilibrium
        let participants = vec![Participation {
            entity_id: Uuid::now_v7(),
            situation_id: Uuid::now_v7(),
            role: Role::Protagonist,
            info_set: None,
            action: None, // no action
            payoff: None,
            seq: 0,
        }];
        let config = MeanFieldConfig::default();
        let (eq, conv, _) = solve_situation(&participants, &config);
        assert!(eq.distribution.is_empty());
        assert!(conv.converged);
    }

    #[test]
    fn test_uniform_equilibrium() {
        // Symmetric game: all actions have same payoff, no coupling
        let participants: Vec<Participation> = (0..10)
            .map(|i| make_participation(if i % 2 == 0 { "cooperate" } else { "defect" }, 1.0))
            .collect();

        let config = MeanFieldConfig {
            coupling_strength: 0.0, // no interaction
            ..Default::default()
        };
        let (eq, conv, _) = solve_situation(&participants, &config);

        assert!(conv.converged);
        assert_eq!(eq.distribution.len(), 2);
        // With equal payoffs and no coupling, distribution should be uniform
        for share in &eq.distribution {
            assert!(
                (share.probability - 0.5).abs() < 0.01,
                "Expected ~0.5, got {}",
                share.probability
            );
        }
    }

    #[test]
    fn test_dominant_action() {
        // One action has much higher payoff
        let mut participants = Vec::new();
        for _ in 0..5 {
            participants.push(make_participation("good", 10.0));
        }
        for _ in 0..5 {
            participants.push(make_participation("bad", 1.0));
        }

        let config = MeanFieldConfig {
            lambda: 5.0, // high rationality
            coupling_strength: 0.0,
            ..Default::default()
        };
        let (eq, conv, _) = solve_situation(&participants, &config);

        assert!(conv.converged);
        let good = eq.distribution.iter().find(|s| s.action == "good").unwrap();
        assert!(
            good.probability > 0.9,
            "Good action should dominate, got {}",
            good.probability
        );
    }

    #[test]
    fn test_coupling_matrix_construction() {
        let participants = vec![
            make_participation("cooperate", 3.0),
            make_participation("cooperate", 4.0),
            make_participation("defect", -2.0),
        ];

        let actions = vec!["cooperate".to_string(), "defect".to_string()];
        let coupling = build_coupling_matrix(&actions, &participants);

        // cooperate-cooperate: positive (3*4 = 12, normalized)
        assert!(
            coupling[0][0] > 0.0,
            "cooperate-cooperate coupling should be positive"
        );
        // cooperate-defect: negative (3*-2 and 4*-2, both negative)
        assert!(
            coupling[0][1] < 0.0,
            "cooperate-defect coupling should be negative"
        );
    }

    #[test]
    fn test_convergence() {
        let mut participants = Vec::new();
        for _ in 0..8 {
            participants.push(make_participation("attack", 2.0));
        }
        for _ in 0..4 {
            participants.push(make_participation("defend", 3.0));
        }

        let config = MeanFieldConfig::default();
        let (_, conv, _) = solve_situation(&participants, &config);

        assert!(conv.converged, "Should converge within max_iterations");
        assert!(conv.iterations < 500, "Should converge quickly");
    }

    #[test]
    fn test_mean_field_utility_no_coupling() {
        let base_payoffs = vec![5.0, 3.0, 1.0];
        let coupling = vec![vec![0.0; 3]; 3];
        let mu = vec![0.33, 0.33, 0.34];
        let config = MeanFieldConfig::default();

        let eu = compute_mean_field_utility(0, &mu, &base_payoffs, &coupling, &config);
        assert!(
            (eu - 5.0).abs() < 0.01,
            "With zero coupling, EU should be base payoff"
        );
    }

    #[test]
    fn test_mean_field_utility_with_coupling() {
        let base_payoffs = vec![5.0, 3.0];
        let coupling = vec![vec![0.0, 10.0], vec![10.0, 0.0]];
        let mu = vec![0.5, 0.5];
        let config = MeanFieldConfig {
            coupling_strength: 1.0,
            ..Default::default()
        };

        let eu = compute_mean_field_utility(0, &mu, &base_payoffs, &coupling, &config);
        // EU(0) = 5.0 + 1.0 * (0.5*0.0 + 0.5*10.0) = 5.0 + 5.0 = 10.0
        assert!(
            (eu - 10.0).abs() < 0.01,
            "Expected 10.0 with coupling, got {}",
            eu
        );
    }

    #[test]
    fn test_stability_classification() {
        let dist = vec![
            ActionShare {
                action: "a".into(),
                probability: 0.5,
                expected_payoff: 1.0,
            },
            ActionShare {
                action: "b".into(),
                probability: 0.5,
                expected_payoff: 1.0,
            },
        ];

        // Weak coupling → stable
        let weak_coupling = vec![vec![0.0, 0.1], vec![0.1, 0.0]];
        assert_eq!(
            classify_stability(&dist, &weak_coupling, 1.0),
            StabilityType::Stable
        );

        // Strong coupling → unstable
        let strong_coupling = vec![vec![0.0, 5.0], vec![5.0, 0.0]];
        assert_eq!(
            classify_stability(&dist, &strong_coupling, 1.0),
            StabilityType::Unstable
        );

        // Mixed coupling → saddle
        let mixed_coupling = vec![vec![0.0, 5.0], vec![-5.0, 0.0]];
        assert_eq!(
            classify_stability(&dist, &mixed_coupling, 1.0),
            StabilityType::Saddle
        );
    }

    #[test]
    fn test_engine_execute() {
        let hg = make_hg();
        let sid = make_situation(&hg);

        // Add participants
        for i in 0..6 {
            let eid = Uuid::now_v7();
            let entity = Entity {
                id: eid,
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("Player{}", i)}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("test".into()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(entity).unwrap();
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: sid,
                role: Role::Protagonist,
                info_set: None,
                action: Some(if i % 2 == 0 { "attack" } else { "defend" }.into()),
                payoff: Some(serde_json::json!(if i % 2 == 0 { 2.0 } else { 3.0 })),
                seq: 0,
            })
            .unwrap();
        }

        let engine = MeanFieldGameEngine::new();
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::MeanFieldGame,
            target_id: sid,
            parameters: serde_json::json!({}),
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
        assert!(result.explanation.is_some());

        let parsed: MeanFieldResult = serde_json::from_value(result.result).unwrap();
        assert_eq!(parsed.num_participants, 6);
        assert!(!parsed.equilibrium.distribution.is_empty());
    }

    #[test]
    fn test_result_serde() {
        let result = MeanFieldResult {
            situation_id: Uuid::now_v7().to_string(),
            num_participants: 10,
            equilibrium: MeanFieldEquilibrium {
                distribution: vec![
                    ActionShare {
                        action: "cooperate".into(),
                        probability: 0.6,
                        expected_payoff: 3.5,
                    },
                    ActionShare {
                        action: "defect".into(),
                        probability: 0.4,
                        expected_payoff: 2.1,
                    },
                ],
                total_welfare: 2.94,
                entropy: 0.67,
                stability: StabilityType::Stable,
            },
            convergence: ConvergenceInfo {
                converged: true,
                iterations: 42,
                final_delta: 1e-7,
            },
            coupling_matrix: vec![CouplingEntry {
                action_i: "cooperate".into(),
                action_j: "defect".into(),
                coupling: -0.5,
            }],
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: MeanFieldResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.num_participants, 10);
        assert_eq!(parsed.equilibrium.distribution.len(), 2);
        assert_eq!(parsed.equilibrium.stability, StabilityType::Stable);
    }

    #[test]
    fn test_kv_persistence() {
        let hg = make_hg();
        let sid = make_situation(&hg);

        for i in 0..5 {
            let eid = Uuid::now_v7();
            let entity = Entity {
                id: eid,
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("E{}", i)}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("test".into()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(entity).unwrap();
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: sid,
                role: Role::Protagonist,
                info_set: None,
                action: Some("act".into()),
                payoff: Some(serde_json::json!(1.0)),
                seq: 0,
            })
            .unwrap();
        }

        let engine = MeanFieldGameEngine::new();
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::MeanFieldGame,
            target_id: sid,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        engine.execute(&job, &hg).unwrap();

        let key = analysis_key(ANALYSIS_MEAN_FIELD, &[&sid.to_string()]);
        let stored = hg.store().get(&key).unwrap();
        assert!(stored.is_some(), "Result should be persisted in KV");

        let parsed: MeanFieldResult = serde_json::from_slice(&stored.unwrap()).unwrap();
        assert_eq!(parsed.num_participants, 5);
    }
}
