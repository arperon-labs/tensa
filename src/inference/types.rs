//! Inference-specific types for Phase 2 async inference workers.
//!
//! These types define the inputs and outputs for causal discovery,
//! game-theoretic analysis, motivation inference, and counterfactual
//! reasoning engines.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{CausalType, GameStructure, InferenceJobType, JobPriority, JobStatus};

// ─── Job Definition ──────────────────────────────────────────

/// An inference job submitted to the job queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJob {
    pub id: String,
    pub job_type: InferenceJobType,
    pub target_id: Uuid,
    pub parameters: serde_json::Value,
    pub priority: JobPriority,
    pub status: JobStatus,
    pub estimated_cost_ms: u64,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

// ─── Causal Engine Output ────────────────────────────────────

/// Result of causal discovery (NOTEARS algorithm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    pub links: Vec<InferredCausalLink>,
    pub confidence: f32,
}

/// A single inferred causal link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredCausalLink {
    pub from_situation: Uuid,
    pub to_situation: Uuid,
    pub mechanism: Option<String>,
    pub strength: f32,
    pub causal_type: CausalType,
}

// ─── Causal Path Explanation ────────────────────────────────

/// Structured explanation of a causal discovery result.
///
/// Contains the top-k strongest causal paths found in the adjacency matrix,
/// with per-hop strength and an overall summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPathExplanation {
    /// The strongest causal paths in the discovered graph.
    pub paths: Vec<CausalPath>,
    /// Human-readable summary of the causal structure.
    pub summary: String,
}

/// A single causal path (chain of situations connected by causal links).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPath {
    /// Ordered sequence of situation UUIDs forming the path.
    pub situation_ids: Vec<Uuid>,
    /// Strength at each hop (len = situation_ids.len() - 1).
    pub hop_strengths: Vec<f32>,
    /// Overall path strength (product of hop strengths).
    pub total_strength: f32,
}

// ─── Counterfactual Output ───────────────────────────────────

/// Result of a counterfactual query (beam search).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    pub intervention: Intervention,
    pub outcomes: Vec<CounterfactualOutcome>,
    pub beam_width: usize,
}

/// A do-calculus intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intervention {
    pub situation_id: Uuid,
    pub do_variable: String,
    pub do_value: serde_json::Value,
}

/// One possible outcome of a counterfactual intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualOutcome {
    pub affected_situation: Uuid,
    pub original_state: serde_json::Value,
    pub counterfactual_state: serde_json::Value,
    pub probability: f32,
}

// ─── Game-Theoretic Output ───────────────────────────────────

/// Result of game-theoretic analysis for a situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameAnalysis {
    pub game_structure: GameStructure,
    pub equilibria: Vec<Equilibrium>,
    pub sub_games: Vec<Uuid>,
}

/// An equilibrium solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Equilibrium {
    pub eq_type: EquilibriumType,
    pub strategy_profile: Vec<ActorStrategy>,
    pub lambda: f32,
}

/// Type of equilibrium found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EquilibriumType {
    Nash,
    QRE,
    Dominant,
}

/// An actor's strategy in an equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorStrategy {
    pub entity_id: Uuid,
    pub action: String,
    pub probability: f32,
    pub expected_payoff: f64,
}

// ─── Motivation Inference Output ─────────────────────────────

/// Result of motivation inference (MaxEnt IRL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationProfile {
    pub entity_id: Uuid,
    pub reward_weights: Vec<FeatureWeight>,
    pub archetype: Option<MotivationArchetype>,
    /// Per-archetype affinity scores (0.0–1.0) for all 7 core archetypes.
    #[serde(default)]
    pub archetype_scores: Vec<ArchetypeScore>,
    pub trajectory_length: usize,
    pub confidence: f32,
}

/// A single archetype affinity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeScore {
    pub archetype: String,
    pub score: f64,
}

/// A single feature weight from IRL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureWeight {
    pub feature_name: String,
    pub weight: f64,
    pub confidence: f32,
}

/// Behavioral archetype for sparse-data actors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotivationArchetype {
    PowerSeeking,
    Altruistic,
    SelfPreserving,
    StatusDriven,
    Vengeful,
    Loyal,
    Opportunistic,
    Ideological,
    Custom(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_job_serialization() {
        let job = InferenceJob {
            id: "job-001".to_string(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"locality_window": 20}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 5000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let json = serde_json::to_vec(&job).unwrap();
        let deserialized: InferenceJob = serde_json::from_slice(&json).unwrap();
        assert_eq!(deserialized.id, "job-001");
        assert_eq!(deserialized.job_type, InferenceJobType::CausalDiscovery);
    }

    #[test]
    fn test_causal_graph_serialization() {
        let graph = CausalGraph {
            links: vec![InferredCausalLink {
                from_situation: Uuid::now_v7(),
                to_situation: Uuid::now_v7(),
                mechanism: Some("direct action".into()),
                strength: 0.85,
                causal_type: CausalType::Contributing,
            }],
            confidence: 0.75,
        };
        let json = serde_json::to_value(&graph).unwrap();
        assert_eq!(graph.links.len(), 1);
        assert!(json["links"].is_array());
    }

    #[test]
    fn test_motivation_profile_serialization() {
        let profile = MotivationProfile {
            entity_id: Uuid::now_v7(),
            reward_weights: vec![
                FeatureWeight {
                    feature_name: "power".into(),
                    weight: 0.8,
                    confidence: 0.9,
                },
                FeatureWeight {
                    feature_name: "loyalty".into(),
                    weight: -0.3,
                    confidence: 0.7,
                },
            ],
            archetype: Some(MotivationArchetype::PowerSeeking),
            archetype_scores: vec![ArchetypeScore {
                archetype: "PowerSeeking".into(),
                score: 0.9,
            }],
            trajectory_length: 15,
            confidence: 0.82,
        };
        let json = serde_json::to_vec(&profile).unwrap();
        let deserialized: MotivationProfile = serde_json::from_slice(&json).unwrap();
        assert_eq!(deserialized.reward_weights.len(), 2);
        assert_eq!(
            deserialized.archetype,
            Some(MotivationArchetype::PowerSeeking)
        );
    }

    #[test]
    fn test_game_analysis_serialization() {
        use crate::types::{GameClassification, InfoStructureType, MaturityLevel};

        let analysis = GameAnalysis {
            game_structure: GameStructure {
                game_type: GameClassification::PrisonersDilemma,
                info_structure: InfoStructureType::Complete,
                description: Some("Classic 2-player dilemma".into()),
                maturity: MaturityLevel::Candidate,
            },
            equilibria: vec![Equilibrium {
                eq_type: EquilibriumType::QRE,
                strategy_profile: vec![ActorStrategy {
                    entity_id: Uuid::now_v7(),
                    action: "cooperate".into(),
                    probability: 0.6,
                    expected_payoff: 3.0,
                }],
                lambda: 1.5,
            }],
            sub_games: vec![],
        };
        let json = serde_json::to_vec(&analysis).unwrap();
        let deserialized: GameAnalysis = serde_json::from_slice(&json).unwrap();
        assert_eq!(deserialized.equilibria.len(), 1);
        assert_eq!(deserialized.equilibria[0].eq_type, EquilibriumType::QRE);
    }

    #[test]
    fn test_counterfactual_result_serialization() {
        let sit_id = Uuid::now_v7();
        let result = CounterfactualResult {
            intervention: Intervention {
                situation_id: sit_id,
                do_variable: "action".into(),
                do_value: serde_json::json!("cooperate"),
            },
            outcomes: vec![CounterfactualOutcome {
                affected_situation: Uuid::now_v7(),
                original_state: serde_json::json!({"outcome": "betrayal"}),
                counterfactual_state: serde_json::json!({"outcome": "mutual_cooperation"}),
                probability: 0.72,
            }],
            beam_width: 5,
        };
        let json = serde_json::to_vec(&result).unwrap();
        let deserialized: CounterfactualResult = serde_json::from_slice(&json).unwrap();
        assert_eq!(deserialized.intervention.situation_id, sit_id);
        assert_eq!(deserialized.outcomes.len(), 1);
    }
}
