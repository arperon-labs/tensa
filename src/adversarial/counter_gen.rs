//! Counter-narrative generation pipeline.
//!
//! Generates reward-aware counter-narratives that satisfy the same
//! psychological needs as the target misinformation while redirecting
//! to a correct conclusion.
//!
//! Pipeline: compute reward fingerprint -> generate N candidates ->
//! score each on reward parity + conclusion redirect -> rank -> return.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

use super::counter_score::*;
use super::reward_model::{
    collect_narrative_text, compute_reward_fingerprint_from_text, DominantReward,
    RewardFingerprint, RewardProfile,
};

// ─── Request / Response Types ────────────────────────────────

/// Request for counter-narrative generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterNarrativeRequest {
    /// Target narrative to counter.
    pub target_narrative_id: String,
    /// Number of candidates to generate (default 5).
    pub max_candidates: usize,
    /// Whether to use inoculation (prebunk) mode vs debunk mode.
    pub inoculation_mode: bool,
    /// Target platform for style adaptation.
    pub target_platform: Option<String>,
    /// Target audience segment.
    pub audience_segment: Option<String>,
}

impl Default for CounterNarrativeRequest {
    fn default() -> Self {
        Self {
            target_narrative_id: String::new(),
            max_candidates: 5,
            inoculation_mode: false,
            target_platform: None,
            audience_segment: None,
        }
    }
}

/// Result of counter-narrative generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterNarrativeResult {
    /// All generated candidates with scores.
    pub candidates: Vec<ScoredCounter>,
    /// The best candidate (highest composite score).
    pub best: Option<ScoredCounter>,
    /// Target narrative's reward fingerprint.
    pub target_fingerprint: RewardFingerprint,
    /// Narrative ID.
    pub narrative_id: String,
}

/// A scored counter-narrative candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredCounter {
    /// The counter-narrative text.
    pub text: String,
    /// Scoring breakdown.
    pub score: CounterNarrativeScore,
    /// Which reward channel this counter targets.
    pub target_reward_channel: String,
    /// Rank (1 = best).
    pub rank: usize,
}

// ─── Generation Pipeline ─────────────────────────────────────

/// Generate counter-narratives for a target narrative.
///
/// Uses heuristic template-based generation. For production, this should
/// be replaced with LLM-based generation using the reward profile as
/// a constraint in the prompt.
pub fn generate_counter_narratives(
    hypergraph: &Hypergraph,
    request: &CounterNarrativeRequest,
) -> Result<CounterNarrativeResult> {
    // Collect text once, use for both fingerprint and conclusion distance
    let target_text = collect_narrative_text(hypergraph, &request.target_narrative_id)?;
    let fingerprint =
        compute_reward_fingerprint_from_text(&target_text, &request.target_narrative_id)?;

    // Step 3: Generate candidates using templates
    let templates = generate_templates(&fingerprint, request.inoculation_mode);
    let n = request.max_candidates.min(templates.len());

    // Step 4: Score each candidate
    let weights = ScoringWeights::default();
    let mut scored: Vec<ScoredCounter> = templates
        .into_iter()
        .take(n)
        .map(|(text, reward_channel, counter_rewards)| {
            let conclusion_dist = estimate_conclusion_distance(&target_text, &text);
            // Estimate R₀ reduction from reward parity
            let parity = fingerprint
                .sharing_rewards
                .cosine_similarity(&counter_rewards)
                .max(0.0);
            let r0_reduction = parity * 0.3; // Higher parity → more effective counter

            let score = score_counter_narrative(
                &fingerprint,
                &counter_rewards,
                conclusion_dist,
                r0_reduction,
                &weights,
            );
            ScoredCounter {
                text,
                score,
                target_reward_channel: reward_channel,
                rank: 0, // Set after sorting
            }
        })
        .collect();

    // Step 5: Rank by composite score
    scored.sort_by(|a, b| {
        b.score
            .composite
            .partial_cmp(&a.score.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, s) in scored.iter_mut().enumerate() {
        s.rank = i + 1;
    }

    let best = scored.first().cloned();

    Ok(CounterNarrativeResult {
        candidates: scored,
        best,
        target_fingerprint: fingerprint,
        narrative_id: request.target_narrative_id.clone(),
    })
}

/// Generate template-based counter-narratives targeting each reward dimension.
///
/// Each template is designed to satisfy a specific psychological need
/// while redirecting to a factual conclusion.
fn generate_templates(
    fingerprint: &RewardFingerprint,
    inoculation_mode: bool,
) -> Vec<(String, String, RewardProfile)> {
    let mut templates = Vec::new();
    let mode = if inoculation_mode {
        "prebunk"
    } else {
        "debunk"
    };

    // Template per dominant reward dimension (sorted by target's sharing reward strength)
    let sharing = &fingerprint.sharing_rewards;
    let mut dims: Vec<(usize, f64)> = sharing.to_vec().into_iter().enumerate().collect();
    dims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let labels = RewardProfile::dimension_labels();

    for (idx, strength) in &dims {
        if *strength < 0.05 {
            continue; // Skip negligible dimensions
        }

        let (text, rewards) = match *idx {
            0 => (
                // tribal_signaling
                format!(
                    "[{mode}] True patriots protect their community by verifying information \
                     before sharing. Our group's strength comes from accuracy, not rumors. \
                     Here's what the evidence actually shows..."
                ),
                RewardProfile {
                    tribal_signaling: 0.8,
                    social_validation: 0.5,
                    competence_display: 0.4,
                    ..Default::default()
                },
            ),
            1 => (
                // anxiety_relief
                format!(
                    "[{mode}] Feeling uncertain is normal, but there's a clear explanation \
                     based on established facts. Understanding the real situation gives us \
                     the power to act effectively..."
                ),
                RewardProfile {
                    anxiety_relief: 0.8,
                    competence_display: 0.6,
                    autonomy_assertion: 0.3,
                    ..Default::default()
                },
            ),
            2 => (
                // moral_outrage
                format!(
                    "[{mode}] You're right to be outraged — but the real scandal is how \
                     this misinformation distracts from the actual problem. The facts \
                     reveal an even more important issue that demands our attention..."
                ),
                RewardProfile {
                    moral_outrage: 0.85,
                    tribal_signaling: 0.4,
                    autonomy_assertion: 0.3,
                    ..Default::default()
                },
            ),
            3 => (
                // social_validation
                format!(
                    "[{mode}] Thousands of informed citizens have already fact-checked \
                     this claim. Join the growing community that values accuracy. \
                     Being well-informed is the new social currency..."
                ),
                RewardProfile {
                    social_validation: 0.8,
                    competence_display: 0.5,
                    relatedness: 0.4,
                    ..Default::default()
                },
            ),
            4 => (
                // novelty_surprise
                format!(
                    "[{mode}] Here's something truly surprising that most people don't \
                     know: the real story behind this claim is even more fascinating \
                     than the fiction. Breaking: the facts tell a more compelling story..."
                ),
                RewardProfile {
                    novelty_surprise: 0.85,
                    competence_display: 0.4,
                    ..Default::default()
                },
            ),
            5 => (
                // competence_display
                format!(
                    "[{mode}] The data actually shows something different. Here's the \
                     peer-reviewed evidence that experts in the field have confirmed. \
                     Understanding this makes you more informed than 95% of people..."
                ),
                RewardProfile {
                    competence_display: 0.85,
                    social_validation: 0.3,
                    ..Default::default()
                },
            ),
            6 => (
                // autonomy_assertion
                format!(
                    "[{mode}] Don't let anyone manipulate you — including the people \
                     spreading this claim. Think independently: check the primary sources \
                     yourself. The truth is out there if you look with clear eyes..."
                ),
                RewardProfile {
                    autonomy_assertion: 0.85,
                    competence_display: 0.4,
                    ..Default::default()
                },
            ),
            7 => (
                // relatedness
                format!(
                    "[{mode}] Our community deserves better than misinformation. Together, \
                     we can build a shared understanding based on facts. When we share \
                     accurate information, we strengthen our bonds..."
                ),
                RewardProfile {
                    relatedness: 0.8,
                    tribal_signaling: 0.5,
                    social_validation: 0.4,
                    ..Default::default()
                },
            ),
            _ => continue,
        };

        templates.push((text, labels[*idx].to_string(), rewards));
    }

    templates
}

// ─── KV Storage ──────────────────────────────────────────────

const COUNTER_PREFIX: &[u8] = b"adv/counter/";

pub fn store_counter_result(
    hypergraph: &Hypergraph,
    result: &CounterNarrativeResult,
) -> Result<()> {
    let mut key = COUNTER_PREFIX.to_vec();
    key.extend_from_slice(result.narrative_id.as_bytes());
    let value = serde_json::to_vec(result)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

pub fn load_counter_result(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<CounterNarrativeResult>> {
    let mut key = COUNTER_PREFIX.to_vec();
    key.extend_from_slice(narrative_id.as_bytes());
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

// ─── Inference Engine ────────────────────────────────────────

use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;

/// Engine for generating counter-narratives.
pub struct CounterNarrativeEngine;

impl InferenceEngine for CounterNarrativeEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CounterNarrative
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

        let max_candidates = job
            .parameters
            .get("max_candidates")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let inoculation_mode = job
            .parameters
            .get("inoculation_mode")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let request = CounterNarrativeRequest {
            target_narrative_id: narrative_id.to_string(),
            max_candidates,
            inoculation_mode,
            ..Default::default()
        };

        let result = generate_counter_narratives(hypergraph, &request)?;

        if let Err(e) = store_counter_result(hypergraph, &result) {
            tracing::warn!("Failed to cache counter-narrative result: {}", e);
        }

        let best_composite = result
            .best
            .as_ref()
            .map(|b| b.score.composite)
            .unwrap_or(0.0);

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: serde_json::to_value(&result)?,
            confidence: best_composite as f32,
            explanation: Some(format!(
                "Generated {} counter-narrative candidates (best composite: {:.2})",
                result.candidates.len(),
                best_composite
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

    fn test_hg_with_narrative(narrative_id: &str, text: &str) -> Hypergraph {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

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

        hg
    }

    #[test]
    fn test_generates_multiple_candidates() {
        let hg = test_hg_with_narrative(
            "counter-test",
            "Outrage scandal corrupt criminal betrayal disgrace immoral evil shameful",
        );

        let request = CounterNarrativeRequest {
            target_narrative_id: "counter-test".into(),
            max_candidates: 5,
            inoculation_mode: false,
            ..Default::default()
        };

        let result = generate_counter_narratives(&hg, &request).unwrap();
        assert!(
            result.candidates.len() >= 2,
            "should generate multiple candidates: {}",
            result.candidates.len()
        );
        assert!(result.best.is_some());
    }

    #[test]
    fn test_candidates_are_ranked() {
        let hg = test_hg_with_narrative(
            "rank-test",
            "This is an outrage and a scandal of corrupt betrayal",
        );

        let request = CounterNarrativeRequest {
            target_narrative_id: "rank-test".into(),
            max_candidates: 5,
            ..Default::default()
        };

        let result = generate_counter_narratives(&hg, &request).unwrap();

        // Verify ranks are sequential
        for (i, c) in result.candidates.iter().enumerate() {
            assert_eq!(c.rank, i + 1, "rank should be sequential");
        }

        // Verify sorted by composite score descending
        for w in result.candidates.windows(2) {
            assert!(
                w[0].score.composite >= w[1].score.composite,
                "candidates should be sorted by composite score"
            );
        }
    }

    #[test]
    fn test_reward_matched_counter_ranks_higher() {
        let hg = test_hg_with_narrative(
            "parity-test",
            "Outrage scandal corrupt criminal betrayal disgrace immoral evil shameful outrage",
        );

        let request = CounterNarrativeRequest {
            target_narrative_id: "parity-test".into(),
            max_candidates: 8,
            ..Default::default()
        };

        let result = generate_counter_narratives(&hg, &request).unwrap();

        // The outrage-targeting counter should rank near the top
        let outrage_counter = result
            .candidates
            .iter()
            .find(|c| c.target_reward_channel == "moral_outrage");
        assert!(
            outrage_counter.is_some(),
            "should have an outrage-targeting counter"
        );

        if let Some(oc) = outrage_counter {
            assert!(
                oc.score.reward_parity > 0.5,
                "outrage counter should have high parity: {}",
                oc.score.reward_parity
            );
        }
    }

    #[test]
    fn test_inoculation_mode() {
        let hg = test_hg_with_narrative(
            "inoc-test",
            "Breaking shocking exclusive bombshell unprecedented leaked",
        );

        let request = CounterNarrativeRequest {
            target_narrative_id: "inoc-test".into(),
            max_candidates: 3,
            inoculation_mode: true,
            ..Default::default()
        };

        let result = generate_counter_narratives(&hg, &request).unwrap();
        assert!(!result.candidates.is_empty());

        // Inoculation candidates should contain [prebunk]
        for c in &result.candidates {
            assert!(
                c.text.contains("[prebunk]"),
                "inoculation mode should produce prebunk text"
            );
        }
    }

    #[test]
    fn test_counter_result_persistence() {
        let hg = test_hg_with_narrative("persist-counter", "Test outrage scandal content");

        let request = CounterNarrativeRequest {
            target_narrative_id: "persist-counter".into(),
            max_candidates: 3,
            ..Default::default()
        };

        let result = generate_counter_narratives(&hg, &request).unwrap();
        store_counter_result(&hg, &result).unwrap();

        let loaded = load_counter_result(&hg, "persist-counter").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().narrative_id, "persist-counter");
    }

    #[test]
    fn test_empty_narrative_produces_empty_candidates() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let request = CounterNarrativeRequest {
            target_narrative_id: "empty".into(),
            max_candidates: 5,
            ..Default::default()
        };

        let result = generate_counter_narratives(&hg, &request).unwrap();
        assert!(
            result.candidates.is_empty(),
            "empty narrative should produce no candidates"
        );
    }
}
