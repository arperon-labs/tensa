//! Adversary policy generator.
//!
//! Reverses IRL: instead of "what rewards explain observed behavior?"
//! asks "given these rewards + constraints, what actions would this actor take?"
//!
//! Takes learned IRL reward weights from `inference::motivation` and generates
//! synthetic adversary action sequences, subject to bounded rationality (SUQR)
//! and operational constraints.

use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::InferenceJobType;

use super::suqr::SubjectiveUtilityQR;
use super::types::*;

// ─── Policy Generator ────────────────────────────────────────

/// Adversary policy generator: turns IRL reward weights into forward actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryPolicyGenerator {
    /// Bounded rationality model (includes feature weights from IRL).
    pub rationality: SubjectiveUtilityQR,
    /// Operational constraints (budget, platforms, timing, opsec).
    pub constraints: OperationalConstraints,
    /// Adversarial archetype label (from disinfo archetypes if available).
    pub archetype: Option<String>,
    /// Narrative context for action generation.
    pub narrative_id: Option<String>,
}

/// Cached policy stored in KV at `adv/policy/{narrative_id}/{actor_id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPolicy {
    /// Policy generator configuration.
    pub policy: AdversaryPolicyGenerator,
    /// Entity this policy is for.
    pub entity_id: Uuid,
    /// When this policy was generated.
    pub created_at: DateTime<Utc>,
    /// Last set of generated actions.
    pub last_actions: Vec<AdversaryAction>,
}

// ─── Action Space ────────────────────────────────────────────

/// Feature vector for an action in the adversary's action space.
///
/// Each dimension corresponds to a reward feature that the IRL
/// weights are learned over.
#[derive(Debug, Clone)]
struct ActionFeatures {
    action_type: ActionType,
    platform: Platform,
    /// Feature vector aligned with reward_weights.
    features: Vec<f64>,
}

/// Default action space with feature vectors.
///
/// Features (aligned with typical IRL reward dimensions):
/// 0: reach potential (how many people see it)
/// 1: disruption potential (how much it shifts beliefs)
/// 2: coordination cost (resource expenditure)
/// 3: detection risk (probability of being caught)
/// 4: amplification multiplier (viral potential)
fn build_action_space(constraints: &OperationalConstraints) -> Vec<ActionFeatures> {
    let mut actions = Vec::new();

    for platform in &constraints.platforms {
        // Post: moderate reach, moderate disruption, low cost, moderate risk
        actions.push(ActionFeatures {
            action_type: ActionType::Post,
            platform: platform.clone(),
            features: vec![0.5, 0.4, 0.2, 0.3 * (1.0 - constraints.opsec_level), 0.3],
        });

        // Amplify: high reach, low disruption, low cost, low risk
        actions.push(ActionFeatures {
            action_type: ActionType::Amplify,
            platform: platform.clone(),
            features: vec![0.7, 0.2, 0.1, 0.1 * (1.0 - constraints.opsec_level), 0.8],
        });

        // Reply: low reach, moderate disruption, low cost, moderate risk
        actions.push(ActionFeatures {
            action_type: ActionType::Reply,
            platform: platform.clone(),
            features: vec![0.2, 0.5, 0.1, 0.4 * (1.0 - constraints.opsec_level), 0.1],
        });

        // Coordinate: very high reach, high disruption, high cost, high risk
        actions.push(ActionFeatures {
            action_type: ActionType::Coordinate,
            platform: platform.clone(),
            features: vec![0.9, 0.7, 0.8, 0.7 * (1.0 - constraints.opsec_level), 0.9],
        });
    }

    // CreateAccount: no immediate reach, no disruption, moderate cost, moderate risk
    if let Some(platform) = constraints.platforms.first() {
        actions.push(ActionFeatures {
            action_type: ActionType::CreateAccount,
            platform: platform.clone(),
            features: vec![0.0, 0.0, 0.4, 0.5 * (1.0 - constraints.opsec_level), 0.0],
        });
    }

    // Observe: zero everything (safe fallback)
    if let Some(platform) = constraints.platforms.first() {
        actions.push(ActionFeatures {
            action_type: ActionType::Observe,
            platform: platform.clone(),
            features: vec![0.0, 0.0, 0.0, 0.0, 0.0],
        });
    }

    actions
}

impl AdversaryPolicyGenerator {
    /// Generate a turn's worth of adversary actions.
    ///
    /// Uses SUQR softmax over the action space weighted by IRL reward weights
    /// to produce a ranked list of actions, filtered by operational constraints.
    pub fn generate_turn(&self, current_time: DateTime<Utc>) -> Vec<AdversaryAction> {
        let action_space = build_action_space(&self.constraints);
        if action_space.is_empty() {
            return vec![];
        }

        // Check working hours constraint (handles cross-midnight windows like 22..6)
        if let Some((start, end)) = self.constraints.working_hours {
            let hour = current_time.hour() as u8;
            let in_window = if start <= end {
                hour >= start && hour < end
            } else {
                hour >= start || hour < end
            };
            if !in_window {
                return vec![];
            }
        }

        // Build SUQR player from action space
        let player = super::suqr::SuqrPlayer {
            entity_id: Uuid::nil(),
            actions: action_space
                .iter()
                .map(|a| format!("{}:{}", a.action_type, a.platform.as_index_str()))
                .collect(),
            features: action_space.iter().map(|a| a.features.clone()).collect(),
        };

        let strategies = self.rationality.solve_single(&player);

        // Convert to AdversaryActions, filtered by budget
        let mut result: Vec<AdversaryAction> = strategies
            .iter()
            .zip(action_space.iter())
            .filter(|(s, _)| s.probability > 0.01) // Drop negligible actions
            .map(|(s, af)| AdversaryAction {
                action_type: af.action_type.clone(),
                target_narrative: self.narrative_id.clone().unwrap_or_default(),
                target_platform: af.platform.clone(),
                content_template: String::new(),
                timing: ActionTiming::Immediate,
                expected_reward: s.subjective_utility,
                confidence: s.probability,
            })
            .collect();

        // Sort by expected reward descending
        result.sort_by(|a, b| {
            b.expected_reward
                .partial_cmp(&a.expected_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to daily budget
        result.truncate(self.constraints.budget_posts_per_day);

        result
    }
}

// ─── KV Storage ──────────────────────────────────────────────

/// KV key prefix for adversary policies.
const POLICY_PREFIX: &[u8] = b"adv/policy/";

/// Store a cached policy in the KV store.
pub fn store_policy(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entity_id: &Uuid,
    policy: &CachedPolicy,
) -> Result<()> {
    let key = policy_key(narrative_id, entity_id);
    let value = serde_json::to_vec(policy)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

/// Load a cached policy from the KV store.
pub fn load_policy(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entity_id: &Uuid,
) -> Result<Option<CachedPolicy>> {
    let key = policy_key(narrative_id, entity_id);
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

fn policy_key(narrative_id: &str, entity_id: &Uuid) -> Vec<u8> {
    let mut key = POLICY_PREFIX.to_vec();
    key.extend_from_slice(narrative_id.as_bytes());
    key.push(b'/');
    key.extend_from_slice(entity_id.as_bytes());
    key
}

// ─── Inference Engines ───────────────────────────────────────

use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::InferenceResult;

/// Engine for generating adversary policies.
pub struct AdversaryPolicyEngine;

impl InferenceEngine for AdversaryPolicyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::AdversaryPolicy
    }

    fn estimate_cost(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<u64> {
        crate::inference::cost::estimate_cost(job, hg)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        // Try to load IRL reward weights from motivation inference
        let reward_weights = job
            .parameters
            .get("reward_weights")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())
            .unwrap_or_else(|| vec![1.0; 5]); // Default: uniform weights

        let lambda = job
            .parameters
            .get("lambda")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let lambda_cap = job
            .parameters
            .get("lambda_cap")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.6);

        let generator = AdversaryPolicyGenerator {
            rationality: SubjectiveUtilityQR::new(lambda, reward_weights, lambda_cap),
            constraints: OperationalConstraints::default(),
            archetype: job
                .parameters
                .get("archetype")
                .and_then(|v| v.as_str())
                .map(String::from),
            narrative_id: Some(narrative_id.to_string()),
        };

        let actions = generator.generate_turn(Utc::now());

        // Cache the policy
        let cached = CachedPolicy {
            policy: generator,
            entity_id: job.target_id,
            created_at: Utc::now(),
            last_actions: actions.clone(),
        };
        let _ = store_policy(hypergraph, narrative_id, &job.target_id, &cached);

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: serde_json::to_value(&cached)?,
            confidence: 0.6,
            explanation: Some("Adversary policy generated from IRL reward weights + SUQR".into()),
            status: crate::types::JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Engine for Cognitive Hierarchy analysis.
pub struct CognitiveHierarchyEngine;

impl InferenceEngine for CognitiveHierarchyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CognitiveHierarchy
    }

    fn estimate_cost(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<u64> {
        crate::inference::cost::estimate_cost(job, hg)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let tau = job
            .parameters
            .get("tau")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.5);

        let max_level = job
            .parameters
            .get("max_level")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let ch = super::cognitive_hierarchy::CognitiveHierarchy { tau, max_level };

        // Get situation participants and build a CH player
        let situation = hypergraph.get_situation(&job.target_id).map_err(|_| {
            TensaError::InferenceError(format!("Target situation not found: {}", job.target_id))
        })?;

        let participants = hypergraph.get_participants_for_situation(&situation.id)?;

        let strategies: Vec<_> = participants
            .iter()
            .filter(|p| p.action.is_some())
            .map(|p| {
                let action = p.action.clone().unwrap_or_default();
                let payoff = p.payoff.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0);

                let player = super::cognitive_hierarchy::ChPlayer {
                    entity_id: p.entity_id,
                    actions: vec![action.clone(), format!("not_{}", action)],
                    payoffs: vec![vec![payoff], vec![-payoff * 0.5]],
                };

                ch.solve(&player)
            })
            .collect();

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: serde_json::to_value(&strategies)?,
            confidence: 0.5,
            explanation: Some(format!(
                "Cognitive Hierarchy analysis (τ={}, max_level={})",
                tau, max_level
            )),
            status: crate::types::JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_generate_turn_produces_actions() {
        let gen = AdversaryPolicyGenerator {
            rationality: SubjectiveUtilityQR::new(2.0, vec![0.5, 0.3, -0.2, -0.4, 0.6], 4.6),
            constraints: OperationalConstraints {
                working_hours: None, // 24/7 so test doesn't depend on current time
                ..Default::default()
            },
            archetype: Some("state_actor".into()),
            narrative_id: Some("test-narrative".into()),
        };

        let actions = gen.generate_turn(Utc::now());
        assert!(!actions.is_empty(), "should generate at least one action");

        // Actions should be sorted by expected_reward descending
        for w in actions.windows(2) {
            assert!(
                w[0].expected_reward >= w[1].expected_reward,
                "actions should be sorted by reward"
            );
        }
    }

    #[test]
    fn test_constraints_reduce_action_space() {
        let constrained = AdversaryPolicyGenerator {
            rationality: SubjectiveUtilityQR::default(),
            constraints: OperationalConstraints {
                budget_posts_per_day: 2,
                platforms: vec![Platform::Twitter],
                ..Default::default()
            },
            archetype: None,
            narrative_id: None,
        };

        let actions = constrained.generate_turn(Utc::now());
        assert!(
            actions.len() <= 2,
            "budget should limit to 2 actions, got {}",
            actions.len()
        );
    }

    #[test]
    fn test_low_lambda_produces_diverse_actions() {
        let gen = AdversaryPolicyGenerator {
            rationality: SubjectiveUtilityQR::new(0.1, vec![1.0; 5], 4.6),
            constraints: OperationalConstraints {
                budget_posts_per_day: 50,
                platforms: vec![Platform::Twitter, Platform::Facebook],
                working_hours: None,
                ..Default::default()
            },
            archetype: None,
            narrative_id: None,
        };

        let actions = gen.generate_turn(Utc::now());
        // With low lambda, many action types should have non-negligible probability
        let unique_types: std::collections::HashSet<_> = actions
            .iter()
            .map(|a| format!("{:?}", a.action_type))
            .collect();
        assert!(
            unique_types.len() >= 3,
            "low lambda should produce diverse actions, got {} types",
            unique_types.len()
        );
    }

    #[test]
    fn test_policy_persistence_roundtrip() {
        let hg = test_hg();
        let entity_id = Uuid::now_v7();
        let narrative_id = "test-nar";

        let cached = CachedPolicy {
            policy: AdversaryPolicyGenerator {
                rationality: SubjectiveUtilityQR::default(),
                constraints: OperationalConstraints::default(),
                archetype: None,
                narrative_id: Some(narrative_id.to_string()),
            },
            entity_id,
            created_at: Utc::now(),
            last_actions: vec![],
        };

        store_policy(&hg, narrative_id, &entity_id, &cached).unwrap();
        let loaded = load_policy(&hg, narrative_id, &entity_id).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.entity_id, entity_id);
        assert_eq!(loaded.policy.rationality.feature_weights, vec![1.0]);
    }

    #[test]
    fn test_policy_load_nonexistent() {
        let hg = test_hg();
        let result = load_policy(&hg, "nope", &Uuid::now_v7()).unwrap();
        assert!(result.is_none());
    }
}
