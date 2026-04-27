//! Counter-narrative scoring — reward parity + conclusion redirect.
//!
//! Scores candidate counter-narratives on two dimensions:
//! 1. **Reward parity**: does it satisfy the same psychological needs?
//! 2. **Conclusion redirect**: does it lead to a different (correct) conclusion?
//!
//! The key insight from Braddock & Horgan (2016): effective counter-narratives
//! must substitute the psychological rewards of the original, not just
//! present factual corrections.

use serde::{Deserialize, Serialize};

use super::reward_model::{RewardFingerprint, RewardProfile};

// ─── Scoring ─────────────────────────────────────────────────

/// Score for a candidate counter-narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterNarrativeScore {
    /// How well the counter matches the target's psychological reward profile (0.0-1.0).
    pub reward_parity: f64,
    /// How far the counter's conclusion diverges from the original (0.0-1.0).
    pub conclusion_redirect: f64,
    /// Weighted composite score (higher = better counter-narrative).
    pub composite: f64,
    /// Per-dimension reward parity breakdown.
    pub per_reward_match: Vec<(String, f64)>,
    /// Estimated R₀ reduction from deploying this counter.
    pub predicted_r0_reduction: f64,
}

/// Weights for the composite score calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for reward parity (default 0.6).
    pub reward_parity_weight: f64,
    /// Weight for conclusion redirect (default 0.3).
    pub redirect_weight: f64,
    /// Weight for predicted R₀ reduction (default 0.1).
    pub r0_weight: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            reward_parity_weight: 0.6,
            redirect_weight: 0.3,
            r0_weight: 0.1,
        }
    }
}

/// Score a candidate counter-narrative against the target narrative's reward profile.
pub fn score_counter_narrative(
    target: &RewardFingerprint,
    counter_rewards: &RewardProfile,
    conclusion_distance: f64,
    predicted_r0_reduction: f64,
    weights: &ScoringWeights,
) -> CounterNarrativeScore {
    // Reward parity: cosine similarity between target sharing rewards and counter rewards
    let parity = target
        .sharing_rewards
        .cosine_similarity(counter_rewards)
        .max(0.0); // Clamp negative similarity to 0

    // Per-dimension match
    let target_vec = target.sharing_rewards.to_vec();
    let counter_vec = counter_rewards.to_vec();
    let labels = RewardProfile::dimension_labels();
    let per_reward_match: Vec<(String, f64)> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let match_score = 1.0 - (target_vec[i] - counter_vec[i]).abs();
            (label.to_string(), match_score.max(0.0))
        })
        .collect();

    // Composite: weighted sum
    let composite = weights.reward_parity_weight * parity
        + weights.redirect_weight * conclusion_distance
        + weights.r0_weight * predicted_r0_reduction;

    CounterNarrativeScore {
        reward_parity: parity,
        conclusion_redirect: conclusion_distance,
        composite,
        per_reward_match,
        predicted_r0_reduction,
    }
}

/// Estimate conclusion distance between two texts using keyword overlap.
///
/// Returns 0.0 for identical conclusions, 1.0 for completely different.
/// This is a heuristic; production systems should use semantic embeddings.
pub fn estimate_conclusion_distance(original_text: &str, counter_text: &str) -> f64 {
    let orig_lower = original_text.to_lowercase();
    let counter_lower = counter_text.to_lowercase();
    let orig_words: std::collections::HashSet<&str> = orig_lower.split_whitespace().collect();
    let counter_words: std::collections::HashSet<&str> = counter_lower.split_whitespace().collect();

    if orig_words.is_empty() && counter_words.is_empty() {
        return 0.0;
    }

    let intersection = orig_words.intersection(&counter_words).count() as f64;
    let union = orig_words.union(&counter_words).count() as f64;

    if union == 0.0 {
        return 1.0;
    }

    // Jaccard distance = 1 - Jaccard similarity
    1.0 - (intersection / union)
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_matched_counter_scores_higher() {
        let target = RewardFingerprint {
            narrative_rewards: RewardProfile {
                moral_outrage: 0.8,
                tribal_signaling: 0.6,
                ..Default::default()
            },
            sharing_rewards: RewardProfile {
                moral_outrage: 0.9,
                tribal_signaling: 0.7,
                ..Default::default()
            },
            dominant_reward: super::super::reward_model::DominantReward::OutrageAmplified,
            reward_intensity: 0.5,
            narrative_id: "test".into(),
            computed_at: chrono::Utc::now(),
        };

        // Counter that matches the outrage reward profile
        let matched_counter = RewardProfile {
            moral_outrage: 0.85,
            tribal_signaling: 0.6,
            ..Default::default()
        };

        // Factual-only counter with no emotional match
        let factual_counter = RewardProfile {
            competence_display: 0.9,
            ..Default::default()
        };

        let weights = ScoringWeights::default();

        let matched_score = score_counter_narrative(&target, &matched_counter, 0.7, 0.3, &weights);
        let factual_score = score_counter_narrative(&target, &factual_counter, 0.8, 0.4, &weights);

        assert!(
            matched_score.reward_parity > factual_score.reward_parity,
            "reward-matched counter should have higher parity: {} vs {}",
            matched_score.reward_parity,
            factual_score.reward_parity
        );
    }

    #[test]
    fn test_conclusion_distance() {
        let same = estimate_conclusion_distance("the cat sat on the mat", "the cat sat on the mat");
        assert!(same < 0.01, "same text should have ~0 distance: {}", same);

        let different = estimate_conclusion_distance(
            "vaccines cause autism according to hidden research",
            "vaccines are safe and effective per extensive clinical trials",
        );
        assert!(
            different > 0.5,
            "different conclusions should have high distance: {}",
            different
        );
    }

    #[test]
    fn test_composite_score_weights() {
        let target = RewardFingerprint {
            narrative_rewards: RewardProfile::default(),
            sharing_rewards: RewardProfile {
                moral_outrage: 0.5,
                ..Default::default()
            },
            dominant_reward: super::super::reward_model::DominantReward::OutrageAmplified,
            reward_intensity: 0.3,
            narrative_id: "test".into(),
            computed_at: chrono::Utc::now(),
        };

        let counter = RewardProfile {
            moral_outrage: 0.5,
            ..Default::default()
        };

        let weights = ScoringWeights {
            reward_parity_weight: 1.0,
            redirect_weight: 0.0,
            r0_weight: 0.0,
        };

        let score = score_counter_narrative(&target, &counter, 0.0, 0.0, &weights);
        assert!(
            (score.composite - score.reward_parity).abs() < 1e-10,
            "with only parity weight, composite should equal parity"
        );
    }

    #[test]
    fn test_per_reward_match_dimensions() {
        let target = RewardFingerprint {
            narrative_rewards: RewardProfile::default(),
            sharing_rewards: RewardProfile {
                moral_outrage: 0.8,
                novelty_surprise: 0.6,
                ..Default::default()
            },
            dominant_reward: super::super::reward_model::DominantReward::OutrageAmplified,
            reward_intensity: 0.3,
            narrative_id: "test".into(),
            computed_at: chrono::Utc::now(),
        };

        let counter = RewardProfile {
            moral_outrage: 0.7,
            novelty_surprise: 0.1,
            ..Default::default()
        };

        let score =
            score_counter_narrative(&target, &counter, 0.5, 0.2, &ScoringWeights::default());

        assert_eq!(score.per_reward_match.len(), 8);

        // Outrage match should be high (0.8 vs 0.7 = 0.9 match)
        let outrage_match = score
            .per_reward_match
            .iter()
            .find(|(l, _)| l == "moral_outrage")
            .unwrap()
            .1;
        assert!(
            outrage_match > 0.8,
            "outrage match should be high: {}",
            outrage_match
        );
    }
}
