//! Motivation inference engine.
//!
//! Implements Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
//! to infer actor motivation vectors from observed behavior trajectories.
//! Falls back to archetype classification for actors with sparse data
//! (<5 observed actions).

use chrono::Utc;
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

/// Maximum number of features for IRL (computational tractability).
const MAX_FEATURES: usize = 20;

/// Minimum actions required for full IRL (below this, use archetypes).
const MIN_ACTIONS_FOR_IRL: usize = 5;

/// Configuration for the motivation engine.
#[derive(Debug, Clone)]
pub struct MotivationConfig {
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Convergence threshold for gradient norm.
    pub convergence_threshold: f64,
    /// Maximum iterations for IRL.
    pub max_iterations: usize,
    /// Maximum trajectory enumerations before switching to logistic approximation.
    pub max_trajectories: usize,
}

impl Default for MotivationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            convergence_threshold: 1e-6,
            max_iterations: 200,
            max_trajectories: 10_000,
        }
    }
}

/// Motivation inference engine using MaxEnt IRL.
pub struct MotivationEngine {
    config: MotivationConfig,
}

impl Default for MotivationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MotivationEngine {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: MotivationConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: MotivationConfig) -> Self {
        Self { config }
    }

    /// Infer motivation profile for an actor entity.
    pub fn infer_motivation(
        &self,
        entity_id: &Uuid,
        hypergraph: &Hypergraph,
    ) -> Result<MotivationProfile> {
        // Get all situations this entity participates in
        let participations = hypergraph.get_situations_for_entity(entity_id)?;

        if participations.is_empty() {
            return Ok(MotivationProfile {
                entity_id: *entity_id,
                reward_weights: vec![],
                archetype: None,
                archetype_scores: vec![],
                trajectory_length: 0,
                confidence: 0.0,
            });
        }

        // Extract trajectory: (participation, situation) pairs sorted by time
        let mut trajectory = Vec::new();
        for p in &participations {
            if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
                trajectory.push((p.clone(), sit));
            }
        }

        // Sort by temporal start
        trajectory.sort_by(|(_, a), (_, b)| {
            let a_time = a.temporal.start.unwrap_or(a.created_at);
            let b_time = b.temporal.start.unwrap_or(b.created_at);
            a_time.cmp(&b_time)
        });

        let action_count = trajectory
            .iter()
            .filter(|(p, _)| p.action.is_some())
            .count();

        if action_count < MIN_ACTIONS_FOR_IRL {
            // Sparse data: compute scores, derive dominant archetype from argmax
            let archetype_scores = self.compute_sparse_archetype_scores(&trajectory);
            let archetype = Self::dominant_archetype(&archetype_scores);
            return Ok(MotivationProfile {
                entity_id: *entity_id,
                reward_weights: vec![],
                archetype,
                archetype_scores,
                trajectory_length: trajectory.len(),
                confidence: 0.3 + (action_count as f32 * 0.1).min(0.2),
            });
        }

        // Full IRL
        let features = self.extract_features(&trajectory, hypergraph)?;
        let observed_features = self.compute_observed_features(&trajectory, &features);
        let weights = self.maxent_irl(&features, &observed_features)?;

        // Build reward weights
        let feature_names = self.feature_names();
        let reward_weights: Vec<FeatureWeight> = weights
            .iter()
            .enumerate()
            .take(feature_names.len().min(weights.len()))
            .map(|(i, &w)| FeatureWeight {
                feature_name: feature_names[i].clone(),
                weight: w,
                confidence: self.weight_confidence(w, action_count),
            })
            .collect();

        // Determine archetype from weights
        let archetype = self.archetype_from_weights(&reward_weights);
        let archetype_scores = self.compute_irl_archetype_scores(&reward_weights);

        let confidence = self.trajectory_confidence(action_count, trajectory.len());

        Ok(MotivationProfile {
            entity_id: *entity_id,
            reward_weights,
            archetype: Some(archetype),
            archetype_scores,
            trajectory_length: trajectory.len(),
            confidence,
        })
    }

    /// Extract features from trajectory for IRL.
    fn extract_features(
        &self,
        trajectory: &[(Participation, Situation)],
        hypergraph: &Hypergraph,
    ) -> Result<Vec<Vec<f64>>> {
        let mut features = Vec::new();

        for (p, sit) in trajectory {
            let mut row = Vec::new();

            // Feature 1: Role is protagonist
            row.push(if p.role == Role::Protagonist {
                1.0
            } else {
                0.0
            });

            // Feature 2: Role is antagonist
            row.push(if p.role == Role::Antagonist { 1.0 } else { 0.0 });

            // Feature 3: Has action
            row.push(if p.action.is_some() { 1.0 } else { 0.0 });

            // Feature 4: Payoff value
            row.push(p.payoff.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0));

            // Feature 5: Knowledge gained (learns count)
            row.push(
                p.info_set
                    .as_ref()
                    .map(|is| is.learns.len() as f64)
                    .unwrap_or(0.0),
            );

            // Feature 6: Knowledge revealed
            row.push(
                p.info_set
                    .as_ref()
                    .map(|is| is.reveals.len() as f64)
                    .unwrap_or(0.0),
            );

            // Feature 7: Situation confidence
            row.push(sit.confidence as f64);

            // Feature 8: Number of co-participants
            let co_participants = hypergraph.get_participants_for_situation(&sit.id)?;
            row.push(co_participants.len() as f64 - 1.0);

            // Feature 9: Narrative level (higher = more granular)
            row.push(match sit.narrative_level {
                NarrativeLevel::Story => 0.0,
                NarrativeLevel::Arc => 0.2,
                NarrativeLevel::Sequence => 0.4,
                NarrativeLevel::Scene => 0.6,
                NarrativeLevel::Beat => 0.8,
                NarrativeLevel::Event => 1.0,
            });

            // Feature 10: Has game structure
            row.push(if sit.game_structure.is_some() {
                1.0
            } else {
                0.0
            });

            // Truncate to MAX_FEATURES
            row.truncate(MAX_FEATURES);
            features.push(row);
        }

        Ok(features)
    }

    /// Compute observed feature expectations (average over trajectory).
    fn compute_observed_features(
        &self,
        trajectory: &[(Participation, Situation)],
        features: &[Vec<f64>],
    ) -> Vec<f64> {
        if features.is_empty() || features[0].is_empty() {
            return vec![];
        }

        let d = features[0].len();
        let n = features.len() as f64;
        let mut expected = vec![0.0; d];

        for row in features {
            for (i, &v) in row.iter().enumerate().take(d) {
                expected[i] += v / n;
            }
        }

        // Weight by actions (features from action steps count more)
        for (idx, (p, _)) in trajectory.iter().enumerate() {
            if p.action.is_some() && idx < features.len() {
                for (i, &v) in features[idx].iter().enumerate().take(d) {
                    expected[i] += v / (n * 2.0); // extra weight for action steps
                }
            }
        }

        expected
    }

    /// MaxEnt IRL per Ziebart et al. (2008): find reward weights θ such that
    /// the expected features under the induced policy match observed features.
    ///
    /// **Trajectory model:** at each timestep t, the entity chose to participate
    /// (observed action) or could have opted out (counterfactual). This gives
    /// 2^T possible trajectories.
    ///
    /// For T ≤ 20: full trajectory enumeration.
    /// For T > 20: per-step logistic (sigmoid) decomposition.
    ///
    /// - Z(θ) = Σ_τ exp(θᵀ f(τ)) over all trajectories τ
    /// - L(θ) = θᵀ f_observed - log Z(θ)
    /// - ∇L = f_observed - E_θ[f]
    fn maxent_irl(
        &self,
        step_features: &[Vec<f64>],
        observed_features: &[f64],
    ) -> Result<Vec<f64>> {
        if observed_features.is_empty() || step_features.is_empty() {
            return Ok(vec![]);
        }

        let n_features = step_features.first().map_or(0, |r| r.len());
        let d = n_features.min(MAX_FEATURES).min(observed_features.len());
        let t = step_features.len();
        let mut theta = vec![0.0; d];

        // Determine whether full enumeration or logistic approximation
        let use_enumeration = t <= 20
            && (1_u64.checked_shl(t as u32).unwrap_or(u64::MAX)
                <= self.config.max_trajectories as u64);

        if use_enumeration {
            // Full trajectory enumeration: 2^T trajectories
            let n_trajectories = 1_u64 << t;

            // Precompute feature vectors for all trajectories (avoid recomputing per iteration)
            let mut trajectory_features: Vec<Vec<f64>> =
                Vec::with_capacity(n_trajectories as usize);
            for k in 0..n_trajectories {
                let mut f_tau = vec![0.0_f64; d];
                for (step, features) in step_features.iter().enumerate() {
                    if (k >> step) & 1 == 1 {
                        for (i, &f) in features.iter().enumerate().take(d) {
                            f_tau[i] += f;
                        }
                    }
                }
                trajectory_features.push(f_tau);
            }

            for _iter in 0..self.config.max_iterations {
                // Compute rewards for all trajectories
                let rewards: Vec<f64> = trajectory_features
                    .iter()
                    .map(|f_tau| theta.iter().zip(f_tau.iter()).map(|(t, f)| t * f).sum())
                    .collect();

                // Log-sum-exp for numerical stability
                let max_reward = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut z = 0.0_f64;
                let mut expected = vec![0.0_f64; d];

                for (k, f_tau) in trajectory_features.iter().enumerate() {
                    let exp_reward = (rewards[k] - max_reward).exp();
                    z += exp_reward;
                    for i in 0..d {
                        expected[i] += f_tau[i] * exp_reward;
                    }
                }

                // Normalize: E_θ[f] = Σ_τ f(τ) P(τ|θ)
                if z > 0.0 {
                    for e in expected.iter_mut() {
                        *e /= z;
                    }
                }

                // Gradient: ∇L = f_observed - E_θ[f]
                let mut max_grad = 0.0_f64;
                for i in 0..d {
                    let grad = observed_features[i] - expected[i];
                    theta[i] += self.config.learning_rate * grad;
                    max_grad = max_grad.max(grad.abs());
                }

                if max_grad < self.config.convergence_threshold {
                    break;
                }
            }
        } else {
            // Per-step logistic decomposition for large trajectories.
            // At each step: P(participate | θ) = σ(θᵀ f_t) = 1 / (1 + exp(-θᵀ f_t))
            // This factorizes the trajectory probability as a product of independent choices.
            for _iter in 0..self.config.max_iterations {
                let mut expected = vec![0.0_f64; d];

                for features in step_features {
                    let reward_t: f64 = theta.iter().zip(features.iter()).map(|(t, f)| t * f).sum();
                    let p_t = 1.0 / (1.0 + (-reward_t).exp()); // sigmoid

                    for (i, &f) in features.iter().enumerate().take(d) {
                        expected[i] += f * p_t;
                    }
                }

                // Normalize by trajectory length
                if t > 0 {
                    for e in expected.iter_mut() {
                        *e /= t as f64;
                    }
                }

                // Gradient: ∇L = f_observed - E_θ[f]
                let mut max_grad = 0.0_f64;
                for i in 0..d {
                    let grad = observed_features[i] - expected[i];
                    theta[i] += self.config.learning_rate * grad;
                    max_grad = max_grad.max(grad.abs());
                }

                if max_grad < self.config.convergence_threshold {
                    break;
                }
            }
        }

        Ok(theta)
    }

    /// Feature names matching the extraction order.
    fn feature_names(&self) -> Vec<String> {
        vec![
            "protagonist_role".into(),
            "antagonist_role".into(),
            "has_action".into(),
            "payoff_value".into(),
            "knowledge_gained".into(),
            "knowledge_revealed".into(),
            "situation_confidence".into(),
            "co_participant_count".into(),
            "narrative_granularity".into(),
            "has_game_structure".into(),
        ]
    }

    /// Confidence for an individual weight based on magnitude and data size.
    fn weight_confidence(&self, weight: f64, action_count: usize) -> f32 {
        let magnitude_factor = (weight.abs() * 2.0).min(1.0);
        let data_factor = (action_count as f64 / 20.0).min(1.0);
        (magnitude_factor * data_factor) as f32
    }

    /// Overall confidence based on trajectory size.
    fn trajectory_confidence(&self, action_count: usize, total_steps: usize) -> f32 {
        let action_ratio = if total_steps > 0 {
            action_count as f32 / total_steps as f32
        } else {
            0.0
        };
        let data_factor = (action_count as f32 / 15.0).min(1.0);
        (0.3 + 0.5 * data_factor + 0.2 * action_ratio).min(1.0)
    }

    /// The 7 core archetype names in canonical order.
    /// Excludes `Ideological` and `Custom` which are open-ended variants
    /// not representable as fixed radar axes.
    const ARCHETYPE_NAMES: [&'static str; 7] = [
        "PowerSeeking",
        "Altruistic",
        "SelfPreserving",
        "StatusDriven",
        "Vengeful",
        "Loyal",
        "Opportunistic",
    ];

    /// Pick the dominant archetype from scored results (argmax).
    fn dominant_archetype(scores: &[ArchetypeScore]) -> Option<MotivationArchetype> {
        scores
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|s| match s.archetype.as_str() {
                "PowerSeeking" => Some(MotivationArchetype::PowerSeeking),
                "Altruistic" => Some(MotivationArchetype::Altruistic),
                "SelfPreserving" => Some(MotivationArchetype::SelfPreserving),
                "StatusDriven" => Some(MotivationArchetype::StatusDriven),
                "Vengeful" => Some(MotivationArchetype::Vengeful),
                "Loyal" => Some(MotivationArchetype::Loyal),
                "Opportunistic" => Some(MotivationArchetype::Opportunistic),
                _ => None,
            })
    }

    /// Normalize raw scores so the maximum becomes 1.0, preserving ratios.
    fn normalize_to_max(raw: &[f64]) -> Vec<f64> {
        let max_val = raw.iter().cloned().fold(0.0f64, f64::max).max(0.01);
        raw.iter().map(|&v| (v / max_val).min(1.0)).collect()
    }

    /// Compute per-archetype affinity scores from sparse trajectory data.
    /// Uses keyword matching and role signals to produce soft scores.
    fn compute_sparse_archetype_scores(
        &self,
        trajectory: &[(Participation, Situation)],
    ) -> Vec<ArchetypeScore> {
        let mut scores = [0.0f64; 7];
        let total = trajectory.len().max(1) as f64;
        let mut protagonist_count = 0usize;
        let mut antagonist_count = 0usize;

        for (p, _) in trajectory {
            if p.role == Role::Protagonist {
                protagonist_count += 1;
            }
            if p.role == Role::Antagonist {
                antagonist_count += 1;
            }

            if let Some(action) = &p.action {
                let a = action.to_lowercase();
                // PowerSeeking
                if a.contains("attack")
                    || a.contains("fight")
                    || a.contains("destroy")
                    || a.contains("seize")
                    || a.contains("dominate")
                {
                    scores[0] += 1.0;
                }
                // Altruistic
                if a.contains("help")
                    || a.contains("save")
                    || a.contains("protect")
                    || a.contains("support")
                    || a.contains("heal")
                {
                    scores[1] += 1.0;
                }
                // SelfPreserving
                if a.contains("flee")
                    || a.contains("hide")
                    || a.contains("defend")
                    || a.contains("survive")
                    || a.contains("escape")
                {
                    scores[2] += 1.0;
                }
                // StatusDriven
                if a.contains("boast")
                    || a.contains("display")
                    || a.contains("impress")
                    || a.contains("claim")
                    || a.contains("announce")
                {
                    scores[3] += 1.0;
                }
                // Vengeful
                if a.contains("revenge")
                    || a.contains("avenge")
                    || a.contains("retaliate")
                    || a.contains("punish")
                {
                    scores[4] += 1.0;
                }
                // Loyal
                if a.contains("follow")
                    || a.contains("serve")
                    || a.contains("loyal")
                    || a.contains("obey")
                    || a.contains("pledge")
                {
                    scores[5] += 1.0;
                }
                // Opportunistic
                if a.contains("exploit")
                    || a.contains("steal")
                    || a.contains("betray")
                    || a.contains("trade")
                    || a.contains("bargain")
                {
                    scores[6] += 1.0;
                }
            }
        }

        // Add role-based signals
        let prot = protagonist_count as f64 / total;
        let antag = antagonist_count as f64 / total;
        scores[0] += antag * 0.5; // PowerSeeking
        scores[1] += prot * 0.3; // Altruistic
        scores[2] += prot * 0.4; // SelfPreserving
        scores[5] += prot * 0.2; // Loyal
        scores[6] += antag * 0.3; // Opportunistic

        let normalized = Self::normalize_to_max(&scores);
        Self::ARCHETYPE_NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| ArchetypeScore {
                archetype: (*name).to_string(),
                score: normalized[i],
            })
            .collect()
    }

    /// Compute per-archetype affinity scores from IRL reward weights.
    /// Maps the 10 IRL features to 7 archetype dimensions via weighted sums.
    fn compute_irl_archetype_scores(&self, weights: &[FeatureWeight]) -> Vec<ArchetypeScore> {
        let get = |name: &str| -> f64 {
            weights
                .iter()
                .find(|w| w.feature_name == name)
                .map(|w| w.weight)
                .unwrap_or(0.0)
        };

        let protagonist = get("protagonist_role");
        let antagonist = get("antagonist_role");
        let payoff = get("payoff_value");
        let knowledge = get("knowledge_gained");
        let reveal = get("knowledge_revealed");
        let action = get("has_action");
        let confidence = get("situation_confidence");
        let copart = get("co_participant_count");
        let granularity = get("narrative_granularity");
        let game = get("has_game_structure");

        // Raw affinity: positive = aligned, then sigmoid to [0, 1]
        let sig = |x: f64| 1.0 / (1.0 + (-x * 4.0).exp());
        let raw = [
            sig(antagonist * 0.4 + payoff * 0.3 + action * 0.3), // PowerSeeking
            sig(protagonist * 0.3 + reveal * 0.4 - payoff * 0.3), // Altruistic
            sig(protagonist * 0.4 + confidence * 0.3 - reveal * 0.3), // SelfPreserving
            sig(copart * 0.4 + granularity * 0.3 + protagonist * 0.3), // StatusDriven
            sig(action * 0.4 + antagonist * 0.3 + knowledge * 0.3), // Vengeful
            sig(protagonist * 0.3 - payoff * 0.3 + copart * 0.4), // Loyal
            sig(payoff * 0.4 + game * 0.3 + action * 0.3),       // Opportunistic
        ];

        let normalized = Self::normalize_to_max(&raw);
        Self::ARCHETYPE_NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| ArchetypeScore {
                archetype: (*name).to_string(),
                score: normalized[i],
            })
            .collect()
    }

    /// Determine archetype from learned reward weights.
    fn archetype_from_weights(&self, weights: &[FeatureWeight]) -> MotivationArchetype {
        let get_weight = |name: &str| -> f64 {
            weights
                .iter()
                .find(|w| w.feature_name == name)
                .map(|w| w.weight)
                .unwrap_or(0.0)
        };

        let protagonist_w = get_weight("protagonist_role");
        let antagonist_w = get_weight("antagonist_role");
        let payoff_w = get_weight("payoff_value");
        let knowledge_w = get_weight("knowledge_gained");
        let reveal_w = get_weight("knowledge_revealed");

        if antagonist_w > protagonist_w && payoff_w > 0.3 {
            MotivationArchetype::PowerSeeking
        } else if reveal_w > 0.5 && knowledge_w < 0.0 {
            MotivationArchetype::Altruistic
        } else if knowledge_w > 0.5 {
            MotivationArchetype::Ideological
        } else if payoff_w > 0.5 {
            MotivationArchetype::Opportunistic
        } else if protagonist_w > 0.3 {
            MotivationArchetype::SelfPreserving
        } else {
            MotivationArchetype::Opportunistic
        }
    }
}

impl InferenceEngine for MotivationEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::MotivationInference
    }

    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
        super::cost::estimate_cost(job, hypergraph)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let profile = self.infer_motivation(&job.target_id, hypergraph)?;
        let confidence = profile.confidence;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: serde_json::to_value(&profile)?,
            confidence,
            explanation: None,
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
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

    fn make_entity(hg: &Hypergraph, name: &str) -> Uuid {
        hg.create_entity(Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap()
    }

    fn make_situation_at(hg: &Hypergraph, offset_hours: i64) -> Uuid {
        let start = Utc::now() + Duration::hours(offset_hours);
        hg.create_situation(Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(start + Duration::hours(1)),
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
            raw_content: vec![ContentBlock::text("Test")],
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
        })
        .unwrap()
    }

    #[test]
    fn test_empty_trajectory() {
        let hg = test_hg();
        let engine = MotivationEngine::new();
        let entity_id = make_entity(&hg, "Nobody");

        let profile = engine.infer_motivation(&entity_id, &hg).unwrap();
        assert_eq!(profile.trajectory_length, 0);
        assert_eq!(profile.confidence, 0.0);
        assert!(profile.reward_weights.is_empty());
    }

    #[test]
    fn test_archetype_sparse_data() {
        let hg = test_hg();
        let engine = MotivationEngine::new();
        let entity_id = make_entity(&hg, "Hero");

        // Add only 3 actions (below MIN_ACTIONS_FOR_IRL)
        for i in 0..3 {
            let sit_id = make_situation_at(&hg, i);
            hg.add_participant(Participation {
                entity_id,
                situation_id: sit_id,
                role: Role::Protagonist,
                info_set: None,
                action: Some("help others".into()),
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let profile = engine.infer_motivation(&entity_id, &hg).unwrap();
        assert!(profile.archetype.is_some());
        assert!(profile.reward_weights.is_empty()); // No IRL for sparse data
        assert_eq!(profile.trajectory_length, 3);
    }

    #[test]
    fn test_archetype_vengeful() {
        let hg = test_hg();
        let engine = MotivationEngine::new();
        let entity_id = make_entity(&hg, "Avenger");

        for i in 0..3 {
            let sit_id = make_situation_at(&hg, i);
            hg.add_participant(Participation {
                entity_id,
                situation_id: sit_id,
                role: Role::Antagonist,
                info_set: None,
                action: Some("revenge".into()),
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let profile = engine.infer_motivation(&entity_id, &hg).unwrap();
        assert_eq!(profile.archetype, Some(MotivationArchetype::Vengeful));
    }

    #[test]
    fn test_full_irl_with_sufficient_data() {
        let hg = test_hg();
        let engine = MotivationEngine::new();
        let entity_id = make_entity(&hg, "Complex Actor");

        // Add 7 actions (above MIN_ACTIONS_FOR_IRL)
        for i in 0..7 {
            let sit_id = make_situation_at(&hg, i);
            hg.add_participant(Participation {
                entity_id,
                situation_id: sit_id,
                role: if i % 2 == 0 {
                    Role::Protagonist
                } else {
                    Role::Antagonist
                },
                info_set: Some(InfoSet {
                    knows_before: vec![],
                    learns: vec![KnowledgeFact {
                        about_entity: Uuid::now_v7(),
                        fact: "something".into(),
                        confidence: 0.8,
                    }],
                    reveals: vec![],
                    beliefs_about_others: vec![],
                }),
                action: Some(format!("action_{}", i)),
                payoff: Some(serde_json::json!(i as f64 * 0.5)),
                seq: 0,
            })
            .unwrap();
        }

        let profile = engine.infer_motivation(&entity_id, &hg).unwrap();
        assert!(!profile.reward_weights.is_empty());
        assert!(profile.reward_weights.len() <= MAX_FEATURES);
        assert!(profile.archetype.is_some());
        assert!(profile.confidence > 0.3);
    }

    #[test]
    fn test_feature_budget_cap() {
        let engine = MotivationEngine::new();
        let feature_names = engine.feature_names();
        assert!(feature_names.len() <= MAX_FEATURES);
    }

    #[test]
    fn test_confidence_decreases_with_fewer_observations() {
        let hg = test_hg();
        let engine = MotivationEngine::new();

        // Actor with many actions
        let many_entity = make_entity(&hg, "Many");
        for i in 0..10 {
            let sit_id = make_situation_at(&hg, i);
            hg.add_participant(Participation {
                entity_id: many_entity,
                situation_id: sit_id,
                role: Role::Protagonist,
                info_set: None,
                action: Some(format!("act_{}", i)),
                payoff: Some(serde_json::json!(1.0)),
                seq: 0,
            })
            .unwrap();
        }

        // Actor with few actions
        let few_entity = make_entity(&hg, "Few");
        for i in 0..5 {
            let sit_id = make_situation_at(&hg, i + 100);
            hg.add_participant(Participation {
                entity_id: few_entity,
                situation_id: sit_id,
                role: Role::Protagonist,
                info_set: None,
                action: Some(format!("act_{}", i)),
                payoff: Some(serde_json::json!(1.0)),
                seq: 0,
            })
            .unwrap();
        }

        let many_profile = engine.infer_motivation(&many_entity, &hg).unwrap();
        let few_profile = engine.infer_motivation(&few_entity, &hg).unwrap();

        assert!(
            many_profile.confidence > few_profile.confidence,
            "More observations should yield higher confidence: {} > {}",
            many_profile.confidence,
            few_profile.confidence
        );
    }

    #[test]
    fn test_engine_execute() {
        let hg = test_hg();
        let engine = MotivationEngine::new();
        let entity_id = make_entity(&hg, "Test");

        let sit_id = make_situation_at(&hg, 0);
        hg.add_participant(Participation {
            entity_id,
            situation_id: sit_id,
            role: Role::Protagonist,
            info_set: None,
            action: Some("act".into()),
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let job = InferenceJob {
            id: "motiv-001".to_string(),
            job_type: InferenceJobType::MotivationInference,
            target_id: entity_id,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 1500,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        let profile: MotivationProfile = serde_json::from_value(result.result).unwrap();
        assert_eq!(profile.entity_id, entity_id);
    }
}
