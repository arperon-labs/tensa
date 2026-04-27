//! Psychological reward model for narratives.
//!
//! Models why people share disinformation through 8 psychological reward
//! dimensions derived from Self-Determination Theory (SDT) and empirical
//! research on moral-emotional contagion.
//!
//! ## References
//!
//! - Brady, Wills, Jost, Tucker & Van Bavel (2017). PNAS 114(28): moral
//!   outrage increases sharing by ~20% per moral-emotional word.
//! - Vosoughi, Roy & Aral (2018). Science 359(6380): false news spreads
//!   via novelty-driven engagement.
//! - McLoughlin, Brady et al. (2024). Science 386(6725): outrage > accuracy
//!   as sharing driver.
//! - Leonard & Philippe (2021). Frontiers in Psychology 12: SDT needs
//!   (autonomy, competence, relatedness) predict conspiracy belief.
//! - Pennycook & Rand (2019): inattention to accuracy, not motivated
//!   reasoning, drives sharing of false news.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

// ─── Reward Profile ──────────────────────────────────────────

/// 8-dimensional psychological reward profile.
///
/// Each dimension ∈ [0.0, 1.0] represents how strongly a narrative
/// satisfies a particular psychological need.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardProfile {
    /// In-group identity reinforcement (Kahan 2017: identity-protective cognition).
    pub tribal_signaling: f64,
    /// Sense-making under uncertainty (conspiracy as anxiety reduction).
    pub anxiety_relief: f64,
    /// Moral-emotional amplification (Brady 2017: +20% sharing per moral word).
    pub moral_outrage: f64,
    /// Likes, shares, community belonging.
    pub social_validation: f64,
    /// Novelty-driven engagement (Vosoughi 2018: false > true on novelty).
    pub novelty_surprise: f64,
    /// SDT: feeling knowledgeable/informed.
    pub competence_display: f64,
    /// SDT: resisting "mainstream" control.
    pub autonomy_assertion: f64,
    /// SDT: connection to like-minded community.
    pub relatedness: f64,
}

impl RewardProfile {
    /// Number of dimensions.
    pub const DIMS: usize = 8;

    /// Convert to a flat vector for mathematical operations.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.tribal_signaling,
            self.anxiety_relief,
            self.moral_outrage,
            self.social_validation,
            self.novelty_surprise,
            self.competence_display,
            self.autonomy_assertion,
            self.relatedness,
        ]
    }

    /// Construct from a flat vector. Returns error if len < 8.
    pub fn from_vec(v: &[f64]) -> crate::error::Result<Self> {
        if v.len() < Self::DIMS {
            return Err(crate::TensaError::InferenceError(format!(
                "RewardProfile::from_vec: need {} values, got {}",
                Self::DIMS,
                v.len()
            )));
        }
        Ok(Self {
            tribal_signaling: v[0],
            anxiety_relief: v[1],
            moral_outrage: v[2],
            social_validation: v[3],
            novelty_surprise: v[4],
            competence_display: v[5],
            autonomy_assertion: v[6],
            relatedness: v[7],
        })
    }

    /// Cosine similarity between two reward profiles.
    pub fn cosine_similarity(&self, other: &RewardProfile) -> f64 {
        let a = self.to_vec();
        let b = other.to_vec();

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// L2 distance between two profiles.
    pub fn distance(&self, other: &RewardProfile) -> f64 {
        let a = self.to_vec();
        let b = other.to_vec();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Phase 2 aggregator-selectable scalar score over the 8-dim reward
    /// profile. `AggregatorKind::Mean` reproduces the legacy "flat
    /// arithmetic mean" view; Choquet over a measure encoding reward-
    /// dimension interactions (e.g. "tribal × moral-outrage together are
    /// more than the sum of their parts") captures the non-additive
    /// utilities the adversarial model theoretically expects.
    ///
    /// Cites: [yager1988owa] [grabisch1996choquet].
    pub fn score_with_aggregator(
        &self,
        agg: &crate::fuzzy::aggregation::AggregatorKind,
    ) -> crate::error::Result<f64> {
        self.score_with_aggregator_tracked(agg, None, None)
            .map(|(score, _, _)| score)
    }

    /// Provenance-tracking sibling of [`Self::score_with_aggregator`].
    ///
    /// Returns `(score, measure_id, measure_version)`. Slot resolution
    /// follows [`crate::fuzzy::aggregation_learn::resolve_measure_provenance`].
    /// `RewardFingerprint` is not extended — the wargame session log
    /// already captures provenance — so callers that need the IDs must
    /// route through this method directly.
    ///
    /// Cites: [grabisch1996choquet], [bustince2016choquet].
    pub fn score_with_aggregator_tracked(
        &self,
        agg: &crate::fuzzy::aggregation::AggregatorKind,
        measure_id: Option<String>,
        measure_version: Option<u32>,
    ) -> crate::error::Result<(f64, Option<String>, Option<u32>)> {
        let xs = self.to_vec();
        let aggregator = crate::fuzzy::aggregation::aggregator_for(agg.clone());
        let score = aggregator.aggregate(&xs)?;
        let (id_out, ver_out) =
            crate::fuzzy::aggregation_learn::resolve_measure_provenance(
                agg,
                measure_id,
                measure_version,
            );
        Ok((score, id_out, ver_out))
    }

    /// Dimension labels for display.
    pub fn dimension_labels() -> &'static [&'static str] {
        &[
            "tribal_signaling",
            "anxiety_relief",
            "moral_outrage",
            "social_validation",
            "novelty_surprise",
            "competence_display",
            "autonomy_assertion",
            "relatedness",
        ]
    }
}

impl Default for RewardProfile {
    fn default() -> Self {
        Self {
            tribal_signaling: 0.0,
            anxiety_relief: 0.0,
            moral_outrage: 0.0,
            social_validation: 0.0,
            novelty_surprise: 0.0,
            competence_display: 0.0,
            autonomy_assertion: 0.0,
            relatedness: 0.0,
        }
    }
}

// ─── Reward Fingerprint ──────────────────────────────────────

/// Full reward fingerprint for a narrative — what needs it satisfies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardFingerprint {
    /// What psychological needs does consuming this narrative satisfy?
    pub narrative_rewards: RewardProfile,
    /// What psychological needs does *sharing* this narrative satisfy?
    pub sharing_rewards: RewardProfile,
    /// Dominant reward channel.
    pub dominant_reward: DominantReward,
    /// Overall reward intensity (0.0-1.0).
    pub reward_intensity: f64,
    /// Narrative ID this fingerprint is for.
    pub narrative_id: String,
    /// When this fingerprint was computed.
    pub computed_at: chrono::DateTime<chrono::Utc>,
}

/// The dominant psychological reward a narrative provides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DominantReward {
    /// Kahan: identity-protective cognition drives sharing.
    IdentityProtective,
    /// McLoughlin: outrage > accuracy as sharing driver.
    OutrageAmplified,
    /// Vosoughi: novelty-driven sharing.
    NoveltySeeking,
    /// Sense-making under uncertainty.
    AnxietyReducing,
    /// Competence/knowledge display.
    StatusSeeking,
}

// ─── Heuristic Reward Computation ────────────────────────────

/// Keyword lists for heuristic reward classification.
const TRIBAL_KEYWORDS: &[&str] = &[
    "us",
    "them",
    "our",
    "enemy",
    "traitor",
    "patriot",
    "real",
    "true",
    "awakened",
    "sheep",
    "elite",
    "establishment",
];
const ANXIETY_KEYWORDS: &[&str] = &[
    "threat",
    "danger",
    "crisis",
    "warning",
    "urgent",
    "alarming",
    "catastrophe",
    "collapse",
    "hidden",
    "secret",
    "coverup",
];
const OUTRAGE_KEYWORDS: &[&str] = &[
    "outrage",
    "disgrace",
    "scandal",
    "corrupt",
    "evil",
    "immoral",
    "shameful",
    "unforgivable",
    "criminal",
    "betrayal",
];
const NOVELTY_KEYWORDS: &[&str] = &[
    "breaking",
    "exclusive",
    "revealed",
    "shocking",
    "unprecedented",
    "never before",
    "first time",
    "bombshell",
    "leaked",
];
const COMPETENCE_KEYWORDS: &[&str] = &[
    "research",
    "study",
    "data",
    "evidence",
    "proof",
    "expert",
    "analysis",
    "investigation",
    "discovered",
    "confirmed",
];
const AUTONOMY_KEYWORDS: &[&str] = &[
    "censored",
    "silenced",
    "banned",
    "suppressed",
    "mainstream",
    "think for yourself",
    "question everything",
    "wake up",
    "truth",
];

/// Collect all raw_content text from a narrative's situations.
pub fn collect_narrative_text(hypergraph: &Hypergraph, narrative_id: &str) -> Result<String> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let mut text = String::new();
    for sit in &situations {
        for block in &sit.raw_content {
            if !block.content.is_empty() {
                text.push(' ');
                text.push_str(&block.content);
            }
        }
    }
    Ok(text)
}

/// Compute a reward fingerprint from narrative content using keyword heuristics.
///
/// For production use, this should be replaced with LLM-based classification.
pub fn compute_reward_fingerprint(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<RewardFingerprint> {
    let all_text = collect_narrative_text(hypergraph, narrative_id)?;
    compute_reward_fingerprint_from_text(&all_text, narrative_id)
}

/// Compute a reward fingerprint from pre-collected text.
pub fn compute_reward_fingerprint_from_text(
    all_text: &str,
    narrative_id: &str,
) -> Result<RewardFingerprint> {
    let lower = all_text.to_lowercase();
    let word_count = lower.split_whitespace().count().max(1) as f64;

    // Count keyword hits per dimension, normalized by word count
    let tribal = keyword_density(&lower, TRIBAL_KEYWORDS, word_count);
    let anxiety = keyword_density(&lower, ANXIETY_KEYWORDS, word_count);
    let outrage = keyword_density(&lower, OUTRAGE_KEYWORDS, word_count);
    let novelty = keyword_density(&lower, NOVELTY_KEYWORDS, word_count);
    let competence = keyword_density(&lower, COMPETENCE_KEYWORDS, word_count);
    let autonomy = keyword_density(&lower, AUTONOMY_KEYWORDS, word_count);

    // Social validation and relatedness are harder to compute from text alone;
    // approximate from engagement signals (future: use actual sharing data)
    let social = (tribal + outrage) * 0.5; // correlated with tribal + outrage
    let relatedness = tribal * 0.7; // correlated with tribal signaling

    let narrative_rewards = RewardProfile {
        tribal_signaling: tribal.min(1.0),
        anxiety_relief: anxiety.min(1.0),
        moral_outrage: outrage.min(1.0),
        social_validation: social.min(1.0),
        novelty_surprise: novelty.min(1.0),
        competence_display: competence.min(1.0),
        autonomy_assertion: autonomy.min(1.0),
        relatedness: relatedness.min(1.0),
    };

    // Sharing rewards are slightly different — outrage and novelty are amplified
    let sharing_rewards = RewardProfile {
        tribal_signaling: (tribal * 1.2).min(1.0),
        anxiety_relief: anxiety.min(1.0),
        moral_outrage: (outrage * 1.3).min(1.0), // Brady 2017: moral content +20-30%
        social_validation: (social * 1.5).min(1.0),
        novelty_surprise: (novelty * 1.4).min(1.0), // Vosoughi 2018: novelty drives sharing
        competence_display: (competence * 1.1).min(1.0),
        autonomy_assertion: autonomy.min(1.0),
        relatedness: (relatedness * 1.3).min(1.0),
    };

    // Determine dominant reward
    let vals = narrative_rewards.to_vec();
    let labels = [
        DominantReward::IdentityProtective, // tribal
        DominantReward::AnxietyReducing,    // anxiety
        DominantReward::OutrageAmplified,   // outrage
        DominantReward::StatusSeeking,      // social (approximate)
        DominantReward::NoveltySeeking,     // novelty
        DominantReward::StatusSeeking,      // competence
        DominantReward::IdentityProtective, // autonomy (maps to identity)
        DominantReward::IdentityProtective, // relatedness (maps to identity)
    ];
    let max_idx = vals
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let dominant_reward = labels[max_idx].clone();
    let reward_intensity = vals.iter().sum::<f64>() / RewardProfile::DIMS as f64;

    Ok(RewardFingerprint {
        narrative_rewards,
        sharing_rewards,
        dominant_reward,
        reward_intensity: reward_intensity.min(1.0),
        narrative_id: narrative_id.to_string(),
        computed_at: Utc::now(),
    })
}

/// Count keyword density: hits / word_count, scaled to [0, 1].
fn keyword_density(text: &str, keywords: &[&str], word_count: f64) -> f64 {
    let hits: usize = keywords.iter().map(|kw| text.matches(kw).count()).sum();
    // Scale: 2% keyword density (2 hits per 100 words) saturates at 1.0
    (hits as f64 / word_count * 50.0).min(1.0)
}

// ─── KV Storage ──────────────────────────────────────────────

const REWARD_PREFIX: &[u8] = b"adv/reward/";

pub fn store_reward_fingerprint(
    hypergraph: &Hypergraph,
    fingerprint: &RewardFingerprint,
) -> Result<()> {
    let mut key = REWARD_PREFIX.to_vec();
    key.extend_from_slice(fingerprint.narrative_id.as_bytes());
    let value = serde_json::to_vec(fingerprint)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

pub fn load_reward_fingerprint(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<RewardFingerprint>> {
    let mut key = REWARD_PREFIX.to_vec();
    key.extend_from_slice(narrative_id.as_bytes());
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

// ─── Inference Engine ────────────────────────────────────────

use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;

/// Engine for computing psychological reward fingerprints.
pub struct RewardFingerprintEngine;

impl InferenceEngine for RewardFingerprintEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::RewardFingerprint
    }

    fn estimate_cost(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<u64> {
        crate::inference::cost::estimate_cost(job, hg)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InferenceError("narrative_id required".into()))?;

        let fingerprint = compute_reward_fingerprint(hypergraph, narrative_id)?;

        if let Err(e) = store_reward_fingerprint(hypergraph, &fingerprint) {
            tracing::warn!("Failed to cache reward fingerprint: {}", e);
        }

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: serde_json::to_value(&fingerprint)?,
            confidence: 0.6,
            explanation: Some(format!(
                "Reward fingerprint: dominant={:?}, intensity={:.2}",
                fingerprint.dominant_reward, fingerprint.reward_intensity
            )),
            status: JobStatus::Completed,
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
    use crate::types::*;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn add_situation(hg: &Hypergraph, narrative_id: &str, text: &str) {
        hg.create_situation(Situation {
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
            raw_content: vec![ContentBlock::text(text)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
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
        })
        .unwrap();
    }

    #[test]
    fn test_high_outrage_content() {
        let hg = test_hg();
        add_situation(
            &hg,
            "outrage-nar",
            "This is an outrage! A disgraceful scandal of corrupt officials \
             committing criminal betrayal. Shameful immoral behavior!",
        );

        let fp = compute_reward_fingerprint(&hg, "outrage-nar").unwrap();
        assert!(
            fp.narrative_rewards.moral_outrage > fp.narrative_rewards.novelty_surprise,
            "outrage content should score high on moral_outrage: {} vs {}",
            fp.narrative_rewards.moral_outrage,
            fp.narrative_rewards.novelty_surprise
        );
        assert_eq!(fp.dominant_reward, DominantReward::OutrageAmplified);
    }

    #[test]
    fn test_conspiracy_content_high_anxiety() {
        let hg = test_hg();
        add_situation(
            &hg,
            "conspiracy-nar",
            "A hidden threat is being covered up. Secret danger lurks in the \
             crisis being concealed from you. Alarming coverup of the warning \
             signs of catastrophe.",
        );

        let fp = compute_reward_fingerprint(&hg, "conspiracy-nar").unwrap();
        assert!(
            fp.narrative_rewards.anxiety_relief > 0.1,
            "conspiracy content should score on anxiety_relief: {}",
            fp.narrative_rewards.anxiety_relief
        );
    }

    #[test]
    fn test_reward_profile_cosine_similarity() {
        let a = RewardProfile {
            moral_outrage: 0.9,
            tribal_signaling: 0.7,
            ..Default::default()
        };
        let b = RewardProfile {
            moral_outrage: 0.8,
            tribal_signaling: 0.6,
            ..Default::default()
        };
        let c = RewardProfile {
            novelty_surprise: 0.9,
            competence_display: 0.7,
            ..Default::default()
        };

        let sim_ab = a.cosine_similarity(&b);
        let sim_ac = a.cosine_similarity(&c);

        assert!(
            sim_ab > sim_ac,
            "similar profiles should have higher cosine sim: {} vs {}",
            sim_ab,
            sim_ac
        );
        assert!(sim_ab > 0.99, "very similar profiles: {}", sim_ab);
    }

    #[test]
    fn test_reward_fingerprint_persistence() {
        let hg = test_hg();
        add_situation(&hg, "persist-nar", "Test content for persistence.");

        let fp = compute_reward_fingerprint(&hg, "persist-nar").unwrap();
        store_reward_fingerprint(&hg, &fp).unwrap();

        let loaded = load_reward_fingerprint(&hg, "persist-nar").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().narrative_id, "persist-nar");
    }

    #[test]
    fn test_empty_narrative() {
        let hg = test_hg();
        let fp = compute_reward_fingerprint(&hg, "empty-nar").unwrap();
        assert!(
            fp.reward_intensity < 0.01,
            "empty narrative should have near-zero intensity: {}",
            fp.reward_intensity
        );
    }

    #[test]
    fn test_dominant_reward_identification() {
        let hg = test_hg();
        add_situation(
            &hg,
            "novelty-nar",
            "Breaking news! Exclusive bombshell revealed for the first time! \
             Unprecedented shocking leaked documents. Breaking exclusive bombshell \
             revealed shocking unprecedented leaked breaking exclusive.",
        );

        let fp = compute_reward_fingerprint(&hg, "novelty-nar").unwrap();
        assert_eq!(fp.dominant_reward, DominantReward::NoveltySeeking);
    }

    #[test]
    fn test_reward_profile_score_with_aggregator_mean_matches_vec_mean() {
        // Phase 2: `score_with_aggregator(Mean)` = arithmetic mean of the
        // eight dimensions. For a uniform-0.5 profile, that's 0.5.
        use crate::fuzzy::aggregation::AggregatorKind;
        let rp = RewardProfile {
            tribal_signaling: 0.5,
            anxiety_relief: 0.5,
            moral_outrage: 0.5,
            social_validation: 0.5,
            novelty_surprise: 0.5,
            competence_display: 0.5,
            autonomy_assertion: 0.5,
            relatedness: 0.5,
        };
        let mean = rp.score_with_aggregator(&AggregatorKind::Mean).unwrap();
        assert!((mean - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_reward_profile_score_with_aggregator_choquet_pessimistic_recovers_min() {
        use crate::fuzzy::aggregation::AggregatorKind;
        use crate::fuzzy::aggregation_measure::symmetric_pessimistic;
        let rp = RewardProfile {
            tribal_signaling: 0.9,
            anxiety_relief: 0.8,
            moral_outrage: 0.1, // minimum
            social_validation: 0.5,
            novelty_surprise: 0.7,
            competence_display: 0.6,
            autonomy_assertion: 0.4,
            relatedness: 0.3,
        };
        let pessimistic = symmetric_pessimistic(8).unwrap();
        let score = rp
            .score_with_aggregator(&AggregatorKind::Choquet(pessimistic))
            .unwrap();
        // Choquet ⊗ symmetric_pessimistic = min of the vector = 0.1.
        assert!(
            (score - 0.1).abs() < 1e-12,
            "pessimistic should recover min (0.1), got {}",
            score
        );

        // Mean path diverges — confirm the switch actually changes something.
        let mean = rp.score_with_aggregator(&AggregatorKind::Mean).unwrap();
        assert!((mean - score).abs() > 0.3);
    }

    #[test]
    fn test_sharing_amplifies_outrage() {
        let hg = test_hg();
        add_situation(
            &hg,
            "share-nar",
            "Outrage scandal corrupt criminal betrayal disgrace immoral evil",
        );

        let fp = compute_reward_fingerprint(&hg, "share-nar").unwrap();
        assert!(
            fp.sharing_rewards.moral_outrage >= fp.narrative_rewards.moral_outrage,
            "sharing should amplify outrage: {} vs {}",
            fp.sharing_rewards.moral_outrage,
            fp.narrative_rewards.moral_outrage
        );
    }
}
