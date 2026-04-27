//! Information entropy analysis for narrative situations.
//!
//! Computes self-information (surprise) of each situation, mutual information
//! between entity pairs, and KL divergence for deception detection.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id, load_sorted_situations};
use crate::error::Result;
use crate::hypergraph::{keys, Hypergraph};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

// ─── Data Structures ────────────────────────────────────────

/// Entropy result for a single situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationEntropy {
    pub situation_id: Uuid,
    pub self_information: f64,
    pub feature_vector: SituationFeatures,
}

/// Feature vector extracted from a situation for entropy computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationFeatures {
    pub participant_count: usize,
    pub unique_role_count: usize,
    pub narrative_level_ordinal: usize,
    pub has_game_structure: bool,
    pub causal_in_degree: usize,
    pub causal_out_degree: usize,
    pub normalized_temporal_position: f64,
}

/// Mutual information between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualInformationResult {
    pub entity_a: Uuid,
    pub entity_b: Uuid,
    pub mutual_information: f64,
    pub joint_appearances: usize,
    pub total_situations: usize,
}

/// KL divergence for deception detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlDivergenceResult {
    pub entity_id: Uuid,
    pub kl_divergence: f64,
    pub action_distribution: HashMap<String, f64>,
    pub expected_distribution: HashMap<String, f64>,
}

/// Full entropy analysis for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyAnalysis {
    pub narrative_id: String,
    pub situation_entropies: Vec<SituationEntropy>,
    pub mutual_information: Vec<MutualInformationResult>,
    pub kl_divergences: Vec<KlDivergenceResult>,
}

// ─── Feature Extraction ─────────────────────────────────────

fn extract_features(
    situation: &Situation,
    situation_index: usize,
    total_situations: usize,
    causal_in: usize,
    causal_out: usize,
    participant_count: usize,
    unique_role_count: usize,
) -> SituationFeatures {
    let normalized_pos = if total_situations > 1 {
        situation_index as f64 / (total_situations - 1) as f64
    } else {
        0.5
    };

    SituationFeatures {
        participant_count,
        unique_role_count,
        narrative_level_ordinal: situation.narrative_level.ordinal(),
        has_game_structure: situation.game_structure.is_some(),
        causal_in_degree: causal_in,
        causal_out_degree: causal_out,
        normalized_temporal_position: normalized_pos,
    }
}

// ─── Self-Information ───────────────────────────────────────

/// Discretize a feature value into a bucket for probability estimation.
fn discretize_count(count: usize) -> usize {
    match count {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=6 => 3,
        _ => 4,
    }
}

/// Compute a feature signature for a situation (for frequency estimation).
fn feature_signature(features: &SituationFeatures) -> String {
    format!(
        "pc{}:rc{}:nl{}:gs{}:ci{}:co{}",
        discretize_count(features.participant_count),
        discretize_count(features.unique_role_count),
        features.narrative_level_ordinal,
        features.has_game_structure as u8,
        discretize_count(features.causal_in_degree),
        discretize_count(features.causal_out_degree),
    )
}

/// Compute self-information for each situation based on feature frequency.
fn compute_self_information(features_list: &[SituationFeatures]) -> Vec<f64> {
    if features_list.is_empty() {
        return vec![];
    }

    let n = features_list.len() as f64;

    // Count feature signature frequencies.
    let mut sig_counts: HashMap<String, usize> = HashMap::new();
    let signatures: Vec<String> = features_list.iter().map(|f| feature_signature(f)).collect();
    for sig in &signatures {
        *sig_counts.entry(sig.clone()).or_insert(0) += 1;
    }

    // Self-information: -log2(P(signature))
    signatures
        .iter()
        .map(|sig| {
            let count = sig_counts[sig] as f64;
            let p = count / n;
            -p.log2()
        })
        .collect()
}

// ─── Mutual Information ─────────────────────────────────────

/// Compute mutual information between all entity pairs.
fn compute_mutual_information(
    entity_ids: &[Uuid],
    entity_participations: &HashMap<Uuid, Vec<Participation>>,
    situation_ids: &std::collections::HashSet<Uuid>,
) -> Vec<MutualInformationResult> {
    let total = situation_ids.len();
    if total == 0 || entity_ids.len() < 2 {
        return vec![];
    }

    // Build entity → set of situations.
    let mut entity_sits: HashMap<Uuid, std::collections::HashSet<Uuid>> = HashMap::new();
    for &eid in entity_ids {
        let sit_set: std::collections::HashSet<Uuid> = entity_participations
            .get(&eid)
            .map(|ps| {
                ps.iter()
                    .filter(|p| situation_ids.contains(&p.situation_id))
                    .map(|p| p.situation_id)
                    .collect()
            })
            .unwrap_or_default();
        entity_sits.insert(eid, sit_set);
    }

    let mut results = Vec::new();
    let n = total as f64;

    for i in 0..entity_ids.len() {
        for j in (i + 1)..entity_ids.len() {
            let a = entity_ids[i];
            let b = entity_ids[j];
            let sits_a = &entity_sits[&a];
            let sits_b = &entity_sits[&b];

            let joint = sits_a.intersection(sits_b).count();

            // P(A=1), P(B=1)
            let pa = sits_a.len() as f64 / n;
            let pb = sits_b.len() as f64 / n;

            if pa == 0.0 || pb == 0.0 || pa == 1.0 || pb == 1.0 {
                results.push(MutualInformationResult {
                    entity_a: a,
                    entity_b: b,
                    mutual_information: 0.0,
                    joint_appearances: joint,
                    total_situations: total,
                });
                continue;
            }

            // Compute MI from the 2x2 contingency table.
            let p11 = joint as f64 / n;
            let p10 = (sits_a.len().saturating_sub(joint)) as f64 / n;
            let p01 = (sits_b.len().saturating_sub(joint)) as f64 / n;
            let p00 = (total + joint).saturating_sub(sits_a.len() + sits_b.len()) as f64 / n;

            let mut mi = 0.0;
            for &(pxy, px, py) in &[
                (p11, pa, pb),
                (p10, pa, 1.0 - pb),
                (p01, 1.0 - pa, pb),
                (p00, 1.0 - pa, 1.0 - pb),
            ] {
                if pxy > 0.0 && px > 0.0 && py > 0.0 {
                    mi += pxy * (pxy / (px * py)).log2();
                }
            }

            results.push(MutualInformationResult {
                entity_a: a,
                entity_b: b,
                mutual_information: mi.max(0.0),
                joint_appearances: joint,
                total_situations: total,
            });
        }
    }

    results
}

// ─── KL Divergence ──────────────────────────────────────────

/// Compute KL divergence for an entity by comparing action distribution
/// against a uniform baseline (if no motivation data available).
fn compute_kl_divergence(
    entity_id: Uuid,
    participations: &[Participation],
    situation_ids: &std::collections::HashSet<Uuid>,
) -> Option<KlDivergenceResult> {
    let relevant: Vec<_> = participations
        .iter()
        .filter(|p| situation_ids.contains(&p.situation_id))
        .collect();

    // Build action distribution.
    let mut action_counts: HashMap<String, usize> = HashMap::new();
    for p in &relevant {
        let action = p.action.clone().unwrap_or_else(|| "none".to_string());
        *action_counts.entry(action).or_insert(0) += 1;
    }

    if action_counts.is_empty() || action_counts.len() < 2 {
        return None;
    }

    let total = relevant.len() as f64;
    let action_dist: HashMap<String, f64> = action_counts
        .iter()
        .map(|(k, v)| (k.clone(), *v as f64 / total))
        .collect();

    // Expected distribution: uniform over observed actions.
    let uniform_p = 1.0 / action_counts.len() as f64;
    let expected_dist: HashMap<String, f64> = action_counts
        .keys()
        .map(|k| (k.clone(), uniform_p))
        .collect();

    // KL(actual || expected) = Σ actual(x) * log(actual(x) / expected(x))
    let mut kl = 0.0;
    for (action, &p) in &action_dist {
        let q = expected_dist[action];
        if p > 0.0 && q > 0.0 {
            kl += p * (p / q).ln();
        }
    }

    Some(KlDivergenceResult {
        entity_id,
        kl_divergence: kl.max(0.0),
        action_distribution: action_dist,
        expected_distribution: expected_dist,
    })
}

// ─── Main Entry Point ───────────────────────────────────────

/// Run entropy analysis on a narrative.
pub fn run_entropy(hypergraph: &Hypergraph, narrative_id: &str) -> Result<EntropyAnalysis> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    let situations = load_sorted_situations(hypergraph, narrative_id)?;

    let total = situations.len();
    let sit_id_set: std::collections::HashSet<Uuid> = situations.iter().map(|s| s.id).collect();

    // Pre-load all participation data once (avoids N+1 queries in MI + KL).
    let mut entity_participations: HashMap<Uuid, Vec<Participation>> = HashMap::new();
    for &eid in &entity_ids {
        entity_participations.insert(eid, hypergraph.get_situations_for_entity(&eid)?);
    }

    // Pre-load causal degrees for all situations (avoids 2 KV calls per situation).
    let mut causal_in_counts: HashMap<Uuid, usize> = HashMap::new();
    let mut causal_out_counts: HashMap<Uuid, usize> = HashMap::new();
    for sit in &situations {
        causal_in_counts.insert(sit.id, hypergraph.get_antecedents(&sit.id)?.len());
        causal_out_counts.insert(sit.id, hypergraph.get_consequences(&sit.id)?.len());
    }

    // Extract features using pre-loaded data.
    let mut features_list = Vec::new();
    for (i, sit) in situations.iter().enumerate() {
        let participants = hypergraph.get_participants_for_situation(&sit.id)?;
        let unique_roles: std::collections::HashSet<_> =
            participants.iter().map(|p| &p.role).collect();
        let features = extract_features(
            sit,
            i,
            total,
            *causal_in_counts.get(&sit.id).unwrap_or(&0),
            *causal_out_counts.get(&sit.id).unwrap_or(&0),
            participants.len(),
            unique_roles.len(),
        );
        features_list.push(features);
    }

    // Self-information.
    let self_info = compute_self_information(&features_list);
    let mut situation_entropies: Vec<SituationEntropy> = Vec::with_capacity(total);
    for (i, sit) in situations.iter().enumerate() {
        let entropy = SituationEntropy {
            situation_id: sit.id,
            self_information: self_info.get(i).copied().unwrap_or(0.0),
            feature_vector: features_list[i].clone(),
        };

        let key = analysis_key(keys::ANALYSIS_ENTROPY, &[narrative_id, &sit.id.to_string()]);
        let bytes = serde_json::to_vec(&entropy)?;
        hypergraph.store().put(&key, &bytes)?;

        situation_entropies.push(entropy);
    }

    let mutual_information =
        compute_mutual_information(&entity_ids, &entity_participations, &sit_id_set);

    // Store MI results.
    for mi in &mutual_information {
        let key = analysis_key(
            keys::ANALYSIS_MUTUAL_INFO,
            &[
                narrative_id,
                &mi.entity_a.to_string(),
                &mi.entity_b.to_string(),
            ],
        );
        let bytes = serde_json::to_vec(mi)?;
        hypergraph.store().put(&key, &bytes)?;
    }

    // KL divergence (reuses pre-loaded participation data).
    let mut kl_divergences = Vec::new();
    for &eid in &entity_ids {
        let participations = entity_participations
            .get(&eid)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        if let Some(kl) = compute_kl_divergence(eid, participations, &sit_id_set) {
            kl_divergences.push(kl);
        }
    }

    Ok(EntropyAnalysis {
        narrative_id: narrative_id.to_string(),
        situation_entropies,
        mutual_information,
        kl_divergences,
    })
}

// ─── InferenceEngine ────────────────────────────────────────

pub struct EntropyEngine;

impl InferenceEngine for EntropyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::EntropyAnalysis
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let analysis = run_entropy(hypergraph, narrative_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::EntropyAnalysis,
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Entropy analysis: {} situations, {} MI pairs",
                analysis.situation_entropies.len(),
                analysis.mutual_information.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod tests;
