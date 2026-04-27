//! TGN-style temporal entity embeddings.
//!
//! Computes trajectory embeddings that evolve as entities participate in
//! situations. Unlike static embeddings (ONNX/Hash) that capture *what*
//! an entity is, trajectory embeddings capture *how* it evolves.
//!
//! Algorithm: feature-based trajectory compression via exponential
//! weighted average over participation features + trajectory statistics.
//!
//! Use cases:
//! - Trajectory similarity: "find entities with similar behavioral arcs"
//! - Role drift detection: entity changes from Witness to Protagonist
//! - Predictive participation: what will this entity do next?
//!
//! Reference: Rossi et al., "Temporal Graph Networks" (Twitter Research, 2020)
//! This implements a simplified, training-free variant.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id, load_sorted_situations};
use crate::error::Result;
use crate::hypergraph::keys::ANALYSIS_TRAJECTORY;
use crate::hypergraph::Hypergraph;
use crate::ingestion::embed::l2_normalize;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

// ─── Configuration ─────────────────────────────────────────

/// Feature dimensions (fixed by extraction).
const STEP_FEATURE_DIM: usize = 25;
/// Trajectory statistics dimensions.
const STAT_DIM: usize = 8;
/// Total embedding dimension.
pub const TRAJECTORY_DIM: usize = STEP_FEATURE_DIM + STAT_DIM; // 33

/// Configuration for the trajectory embedding engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryConfig {
    /// Exponential decay factor for weighting recent events more.
    pub decay_factor: f64,
    /// Minimum participation steps to compute a trajectory (below this, entity is skipped).
    pub min_steps: usize,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.9,
            min_steps: 2,
        }
    }
}

// ─── Result Types ──────────────────────────────────────────

/// Full result of a trajectory embedding job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryResult {
    pub narrative_id: String,
    pub entity_embeddings: Vec<EntityTrajectoryEmbedding>,
    pub dimension: usize,
    pub entities_skipped: usize,
}

/// A single entity's trajectory embedding with diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTrajectoryEmbedding {
    pub entity_id: Uuid,
    pub entity_name: String,
    pub embedding: Vec<f32>,
    pub trajectory_length: usize,
    pub temporal_span_hours: f64,
    pub role_entropy: f64,
}

// ─── Engine ────────────────────────────────────────────────

/// Trajectory embedding inference engine.
pub struct TrajectoryEmbeddingEngine;

impl InferenceEngine for TrajectoryEmbeddingEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::TrajectoryEmbedding
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;

        let mut config = TrajectoryConfig::default();
        if let Some(v) = job.parameters.get("decay_factor").and_then(|v| v.as_f64()) {
            config.decay_factor = v.clamp(0.01, 0.999);
        }
        if let Some(v) = job.parameters.get("min_steps").and_then(|v| v.as_u64()) {
            config.min_steps = (v as usize).max(1);
        }

        let result = compute_narrative_trajectories(hypergraph, narrative_id, &config)?;

        let result_value = serde_json::to_value(&result)?;
        let key = analysis_key(ANALYSIS_TRAJECTORY, &[narrative_id]);
        hypergraph
            .store()
            .put(&key, result_value.to_string().as_bytes())?;

        let n_embedded = result.entity_embeddings.len();
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::TrajectoryEmbedding,
            target_id: job.target_id,
            result: result_value,
            confidence: if n_embedded > 0 { 0.85 } else { 0.0 },
            explanation: Some(format!(
                "Trajectory embeddings: {} entities embedded ({}d), {} skipped (< {} steps)",
                n_embedded, TRAJECTORY_DIM, result.entities_skipped, config.min_steps,
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Core Algorithm ────────────────────────────────────────

/// Compute trajectory embeddings for all entities in a narrative.
fn compute_narrative_trajectories(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    config: &TrajectoryConfig,
) -> Result<TrajectoryResult> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let situations = load_sorted_situations(hypergraph, narrative_id)?;

    if entities.is_empty() {
        return Ok(TrajectoryResult {
            narrative_id: narrative_id.to_string(),
            entity_embeddings: vec![],
            dimension: TRAJECTORY_DIM,
            entities_skipped: 0,
        });
    }

    // Build situation lookup
    let sit_map: HashMap<Uuid, &Situation> = situations.iter().map(|s| (s.id, s)).collect();

    let mut embeddings = Vec::new();
    let mut skipped = 0;

    for entity in &entities {
        // Get all participations for this entity
        let participations = hypergraph.get_situations_for_entity(&entity.id)?;

        if participations.len() < config.min_steps {
            skipped += 1;
            continue;
        }

        // Build chronological participation steps with situation data
        let mut steps: Vec<(&Participation, &Situation)> = Vec::new();
        for p in &participations {
            if let Some(sit) = sit_map.get(&p.situation_id) {
                steps.push((p, sit));
            }
        }

        // Sort by situation temporal start
        steps.sort_by(|a, b| a.1.temporal.start.cmp(&b.1.temporal.start));

        if steps.len() < config.min_steps {
            skipped += 1;
            continue;
        }

        // Extract per-step features
        let feature_vectors: Vec<Vec<f64>> = steps
            .iter()
            .enumerate()
            .map(|(i, (p, sit))| {
                extract_step_features(
                    &entity.entity_type,
                    p,
                    sit,
                    entity.confidence,
                    i,
                    steps.len(),
                )
            })
            .collect();

        // Exponential weighted average
        let compressed = exponential_weighted_average(&feature_vectors, config.decay_factor);

        // Compute trajectory diagnostics (used in both stats and output)
        let temporal_span = compute_temporal_span(&steps);
        let role_entropy = compute_role_entropy(&steps);

        // Compute trajectory statistics
        let stats = compute_trajectory_stats(&steps, entity, role_entropy, temporal_span);

        // Combine: compressed features + statistics
        let mut embedding: Vec<f32> = compressed.iter().map(|v| *v as f32).collect();
        embedding.extend(stats.iter().map(|v| *v as f32));

        // L2 normalize
        l2_normalize(&mut embedding);

        let entity_name = entity
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed")
            .to_string();

        embeddings.push(EntityTrajectoryEmbedding {
            entity_id: entity.id,
            entity_name,
            embedding,
            trajectory_length: steps.len(),
            temporal_span_hours: temporal_span,
            role_entropy,
        });
    }

    Ok(TrajectoryResult {
        narrative_id: narrative_id.to_string(),
        entity_embeddings: embeddings,
        dimension: TRAJECTORY_DIM,
        entities_skipped: skipped,
    })
}

/// Extract a 25-dimensional feature vector for a single participation step.
fn extract_step_features(
    entity_type: &EntityType,
    participation: &Participation,
    situation: &Situation,
    confidence: f32,
    step_index: usize,
    total_steps: usize,
) -> Vec<f64> {
    let mut features = vec![0.0_f64; STEP_FEATURE_DIM];

    // Entity type one-hot (5 dims: indices 0-4)
    let et_idx = match entity_type {
        EntityType::Actor => 0,
        EntityType::Location => 1,
        EntityType::Artifact => 2,
        EntityType::Concept => 3,
        EntityType::Organization => 4,
    };
    features[et_idx] = 1.0;

    // Role one-hot (10 dims: indices 5-14)
    let role_idx = match &participation.role {
        Role::Protagonist => 5,
        Role::Antagonist => 6,
        Role::Witness => 7,
        Role::Target => 8,
        Role::Instrument => 9,
        Role::Confidant => 10,
        Role::Informant => 11,
        Role::Recipient => 12,
        Role::Bystander => 13,
        Role::SubjectOfDiscussion | Role::Facilitator | Role::Custom(_) => 14,
    };
    features[role_idx] = 1.0;

    // Narrative level one-hot (6 dims: indices 15-20)
    let level_idx = match situation.narrative_level {
        NarrativeLevel::Story => 15,
        NarrativeLevel::Arc => 16,
        NarrativeLevel::Sequence => 17,
        NarrativeLevel::Scene => 18,
        NarrativeLevel::Beat => 19,
        NarrativeLevel::Event => 20,
    };
    features[level_idx] = 1.0;

    // Confidence (index 21)
    features[21] = confidence as f64;

    // Temporal position normalized [0,1] (index 22)
    let t_pos = if total_steps > 1 {
        step_index as f64 / (total_steps - 1) as f64
    } else {
        0.5
    };
    features[22] = t_pos;

    // Has payoff (index 23)
    features[23] = if participation.payoff.is_some() {
        1.0
    } else {
        0.0
    };

    // Payoff value (index 24)
    features[24] = participation
        .payoff
        .as_ref()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .clamp(-10.0, 10.0)
        / 10.0; // normalize to [-1, 1]

    features
}

/// Compress a sequence of feature vectors using exponential weighted average.
/// Recent events (higher index) are weighted more with decay_factor < 1.
fn exponential_weighted_average(vectors: &[Vec<f64>], decay: f64) -> Vec<f64> {
    if vectors.is_empty() {
        return vec![0.0; STEP_FEATURE_DIM];
    }

    let n = vectors.len();
    let dim = vectors[0].len();
    let mut result = vec![0.0_f64; dim];
    let mut weight_sum = 0.0_f64;

    for (t, vec) in vectors.iter().enumerate() {
        let weight = decay.powi((n - 1 - t) as i32);
        weight_sum += weight;
        for (i, val) in vec.iter().enumerate() {
            result[i] += weight * val;
        }
    }

    if weight_sum > 0.0 {
        for v in &mut result {
            *v /= weight_sum;
        }
    }

    result
}

/// Compute 8-dimensional trajectory statistics.
fn compute_trajectory_stats(
    steps: &[(&Participation, &Situation)],
    entity: &Entity,
    role_entropy: f64,
    temporal_span: f64,
) -> Vec<f64> {
    let mut stats = vec![0.0_f64; STAT_DIM];

    let n = steps.len() as f64;

    // 0: Trajectory length (log-normalized, cap at 100)
    stats[0] = (n.ln() / 100.0_f64.ln()).min(1.0);

    // 1: Role entropy (max = ln(10) ≈ 2.303 for 10 role categories)
    stats[1] = (role_entropy / 10.0_f64.ln()).clamp(0.0, 1.0);

    // 2: Role drift (cosine distance between first-half and second-half role distributions)
    stats[2] = compute_role_drift(steps);

    // 3: Temporal span (hours, log-normalized)
    let span = temporal_span;
    stats[3] = if span > 0.0 {
        (span.ln() / 10000.0_f64.ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // 4: Mean confidence (entity confidence as proxy)
    stats[4] = entity.confidence as f64;

    // 5: Participation density (events per hour)
    let density = if span > 0.0 { n / span } else { 0.0 };
    stats[5] = (density * 10.0).min(1.0); // normalize

    // 6: Unique situations (normalized by total steps)
    let unique_sits: std::collections::HashSet<Uuid> = steps.iter().map(|(_, s)| s.id).collect();
    stats[6] = unique_sits.len() as f64 / n.max(1.0);

    // 7: Has causal involvement (any situation has causes)
    stats[7] = if steps.iter().any(|(_, s)| !s.causes.is_empty()) {
        1.0
    } else {
        0.0
    };

    stats
}

/// Compute Shannon entropy of role distribution.
fn compute_role_entropy(steps: &[(&Participation, &Situation)]) -> f64 {
    let mut role_counts: HashMap<&Role, usize> = HashMap::new();
    for (p, _) in steps {
        *role_counts.entry(&p.role).or_default() += 1;
    }

    let total = steps.len() as f64;
    if total == 0.0 {
        return 0.0;
    }

    -role_counts
        .values()
        .map(|&count| {
            let p = count as f64 / total;
            if p > 0.0 {
                p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// Compute role drift: cosine distance between first-half and second-half role distributions.
fn compute_role_drift(steps: &[(&Participation, &Situation)]) -> f64 {
    if steps.len() < 4 {
        return 0.0;
    }

    let mid = steps.len() / 2;
    let first_half = &steps[..mid];
    let second_half = &steps[mid..];

    let dist_a = role_distribution(first_half);
    let dist_b = role_distribution(second_half);

    // Cosine distance = 1 - cosine_similarity
    let dot: f64 = dist_a.iter().zip(dist_b.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f64 = dist_a.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm_b: f64 = dist_b.iter().map(|b| b * b).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // maximum drift
    }

    (1.0 - dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

/// Build a 10-dim role frequency distribution.
fn role_distribution(steps: &[(&Participation, &Situation)]) -> Vec<f64> {
    let mut dist = vec![0.0_f64; 10];
    let n = steps.len() as f64;
    if n == 0.0 {
        return dist;
    }

    for (p, _) in steps {
        let idx = match &p.role {
            Role::Protagonist => 0,
            Role::Antagonist => 1,
            Role::Witness => 2,
            Role::Target => 3,
            Role::Instrument => 4,
            Role::Confidant => 5,
            Role::Informant => 6,
            Role::Recipient => 7,
            Role::Bystander => 8,
            Role::SubjectOfDiscussion | Role::Facilitator | Role::Custom(_) => 9,
        };
        dist[idx] += 1.0 / n;
    }

    dist
}

/// Compute temporal span in hours between first and last participation.
fn compute_temporal_span(steps: &[(&Participation, &Situation)]) -> f64 {
    if steps.len() < 2 {
        return 0.0;
    }

    let first = steps
        .first()
        .and_then(|(_, s)| s.temporal.start)
        .unwrap_or_default();
    let last = steps
        .last()
        .and_then(|(_, s)| s.temporal.start)
        .unwrap_or_default();

    let diff = last.signed_duration_since(first);
    (diff.num_milliseconds() as f64 / 3_600_000.0).max(0.0)
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, make_hg};
    use chrono::Duration;

    fn add_situation_at(
        hg: &Hypergraph,
        narrative: &str,
        level: NarrativeLevel,
        base: chrono::DateTime<Utc>,
        offset_min: i64,
    ) -> Uuid {
        let start = base + Duration::minutes(offset_min);
        let end = start + Duration::minutes(5);
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
            raw_content: vec![ContentBlock::text("test")],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
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
        hg.create_situation(sit).unwrap()
    }

    fn link_with_role(hg: &Hypergraph, entity: Uuid, situation: Uuid, role: Role) {
        hg.add_participant(Participation {
            entity_id: entity,
            situation_id: situation,
            role,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    #[test]
    fn test_empty_narrative() {
        let hg = make_hg();
        let result =
            compute_narrative_trajectories(&hg, "empty", &TrajectoryConfig::default()).unwrap();
        assert!(result.entity_embeddings.is_empty());
        assert_eq!(result.entities_skipped, 0);
    }

    #[test]
    fn test_single_step_below_minimum() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Solo", "n1");
        let s = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 0);
        link_with_role(&hg, e, s, Role::Protagonist);

        let result =
            compute_narrative_trajectories(&hg, "n1", &TrajectoryConfig::default()).unwrap();
        assert!(result.entity_embeddings.is_empty());
        assert_eq!(result.entities_skipped, 1);
    }

    #[test]
    fn test_two_step_trajectory() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Alice", "n1");
        let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 0);
        let s2 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 60);
        link_with_role(&hg, e, s1, Role::Protagonist);
        link_with_role(&hg, e, s2, Role::Protagonist);

        let result =
            compute_narrative_trajectories(&hg, "n1", &TrajectoryConfig::default()).unwrap();
        assert_eq!(result.entity_embeddings.len(), 1);
        assert_eq!(result.entity_embeddings[0].trajectory_length, 2);
        assert_eq!(result.entity_embeddings[0].entity_name, "Alice");
    }

    #[test]
    fn test_feature_extraction() {
        let p = Participation {
            entity_id: Uuid::now_v7(),
            situation_id: Uuid::now_v7(),
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: Some(serde_json::json!(5.0)),
            seq: 0,
        };
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
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
        };

        let features = extract_step_features(&EntityType::Actor, &p, &sit, 0.9, 0, 5);
        assert_eq!(features.len(), STEP_FEATURE_DIM);
        assert_eq!(features[0], 1.0); // Actor one-hot
        assert_eq!(features[5], 1.0); // Protagonist one-hot
        assert_eq!(features[18], 1.0); // Scene one-hot
        assert!((features[21] - 0.9).abs() < 0.01); // confidence
        assert_eq!(features[23], 1.0); // has payoff
        assert!((features[24] - 0.5).abs() < 0.01); // payoff 5.0/10.0
    }

    #[test]
    fn test_exponential_decay() {
        // Two feature vectors: first = [1, 0], second = [0, 1]
        // With decay 0.5, second (more recent) should be weighted more
        let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = exponential_weighted_average(&vecs, 0.5);
        // weight_0 = 0.5^1 = 0.5, weight_1 = 0.5^0 = 1.0
        // result[0] = 0.5*1 / 1.5 = 0.333
        // result[1] = 1.0*1 / 1.5 = 0.667
        assert!(
            result[1] > result[0],
            "Recent event should be weighted more"
        );
    }

    #[test]
    fn test_role_entropy() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Multi", "n1");

        // Same role repeatedly → zero entropy
        let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 0);
        let s2 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 10);
        link_with_role(&hg, e, s1, Role::Protagonist);
        link_with_role(&hg, e, s2, Role::Protagonist);

        let result =
            compute_narrative_trajectories(&hg, "n1", &TrajectoryConfig::default()).unwrap();
        assert!(!result.entity_embeddings.is_empty());
        let entropy_single = result.entity_embeddings[0].role_entropy;
        assert!(
            entropy_single < 0.01,
            "Single role should have ~0 entropy, got {}",
            entropy_single
        );
    }

    #[test]
    fn test_role_drift() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Drifter", "n1");

        // First half: Witness, second half: Protagonist
        for i in 0..4 {
            let s = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, i * 10);
            let role = if i < 2 {
                Role::Witness
            } else {
                Role::Protagonist
            };
            link_with_role(&hg, e, s, role);
        }

        let result =
            compute_narrative_trajectories(&hg, "n1", &TrajectoryConfig::default()).unwrap();
        assert_eq!(result.entity_embeddings.len(), 1);
        // Role entropy should be positive (2 different roles)
        assert!(result.entity_embeddings[0].role_entropy > 0.0);
    }

    #[test]
    fn test_trajectory_similarity() {
        let hg = make_hg();
        let base = Utc::now();

        // Two entities with identical trajectories
        for name in ["Alice", "Bob"] {
            let e = add_entity(&hg, name, "sim");
            for i in 0..3 {
                let s = add_situation_at(&hg, "sim", NarrativeLevel::Scene, base, i * 10);
                link_with_role(&hg, e, s, Role::Protagonist);
            }
        }

        let result =
            compute_narrative_trajectories(&hg, "sim", &TrajectoryConfig::default()).unwrap();
        assert_eq!(result.entity_embeddings.len(), 2);

        let emb_a = &result.entity_embeddings[0].embedding;
        let emb_b = &result.entity_embeddings[1].embedding;
        let sim = crate::ingestion::embed::cosine_similarity(emb_a, emb_b);
        assert!(
            sim > 0.9,
            "Similar trajectories should have high cosine similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_trajectory_dissimilarity() {
        let hg = make_hg();
        let base = Utc::now();

        // Entity A: Actor/Protagonist in Scene
        let ea = add_entity(&hg, "Hero", "diff");
        for i in 0..3 {
            let s = add_situation_at(&hg, "diff", NarrativeLevel::Scene, base, i * 10);
            link_with_role(&hg, ea, s, Role::Protagonist);
        }

        // Entity B: Location/Bystander in Event (very different)
        let eb = hg
            .create_entity(Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Location,
                properties: serde_json::json!({"name": "Dungeon"}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.3,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("diff".into()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();
        for i in 0..3 {
            let s = add_situation_at(&hg, "diff", NarrativeLevel::Event, base, i * 100);
            link_with_role(&hg, eb, s, Role::Bystander);
        }

        let result =
            compute_narrative_trajectories(&hg, "diff", &TrajectoryConfig::default()).unwrap();
        assert_eq!(result.entity_embeddings.len(), 2);

        let emb_hero = result
            .entity_embeddings
            .iter()
            .find(|e| e.entity_name == "Hero")
            .unwrap();
        let emb_loc = result
            .entity_embeddings
            .iter()
            .find(|e| e.entity_name == "Dungeon")
            .unwrap();

        let sim =
            crate::ingestion::embed::cosine_similarity(&emb_hero.embedding, &emb_loc.embedding);
        assert!(
            sim < 0.5,
            "Different trajectories should have low similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_embedding_dimension() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Dimcheck", "n1");
        let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 0);
        let s2 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 10);
        link_with_role(&hg, e, s1, Role::Protagonist);
        link_with_role(&hg, e, s2, Role::Antagonist);

        let result =
            compute_narrative_trajectories(&hg, "n1", &TrajectoryConfig::default()).unwrap();
        assert_eq!(result.entity_embeddings[0].embedding.len(), TRAJECTORY_DIM);
        assert_eq!(TRAJECTORY_DIM, 33);
    }

    #[test]
    fn test_engine_execute() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Engine", "traj-test");
        let s1 = add_situation_at(&hg, "traj-test", NarrativeLevel::Scene, base, 0);
        let s2 = add_situation_at(&hg, "traj-test", NarrativeLevel::Scene, base, 60);
        link_with_role(&hg, e, s1, Role::Protagonist);
        link_with_role(&hg, e, s2, Role::Witness);

        let engine = TrajectoryEmbeddingEngine;
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::TrajectoryEmbedding,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": "traj-test"}),
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

        let parsed: TrajectoryResult = serde_json::from_value(result.result).unwrap();
        assert_eq!(parsed.narrative_id, "traj-test");
        assert_eq!(parsed.entity_embeddings.len(), 1);
        assert_eq!(parsed.dimension, 33);
    }

    #[test]
    fn test_result_serde() {
        let result = TrajectoryResult {
            narrative_id: "test".into(),
            entity_embeddings: vec![EntityTrajectoryEmbedding {
                entity_id: Uuid::now_v7(),
                entity_name: "Alice".into(),
                embedding: vec![0.1; 33],
                trajectory_length: 5,
                temporal_span_hours: 24.0,
                role_entropy: 0.69,
            }],
            dimension: 33,
            entities_skipped: 2,
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: TrajectoryResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entity_embeddings.len(), 1);
        assert_eq!(parsed.entity_embeddings[0].entity_name, "Alice");
        assert_eq!(parsed.dimension, 33);
    }
}
