//! Scene-sequel rhythm classification (Sprint D9.4, Swain/Bickham).
//!
//! Micro-scale structural analysis: classify each situation as Action Scene
//! (goal/conflict/disaster) or Sequel (reaction/dilemma/decision), compute
//! alternation rhythm, and derive a composite pacing score.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use chrono::Utc;

use crate::inference::InferenceEngine;
use crate::types::InferenceResult;
use crate::types::{InferenceJobType, JobStatus, Situation};

// ─── Types ──────────────────────────────────────────────────

/// Scene type in the Swain/Bickham model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneType {
    /// Character pursues goal → faces opposition → outcome (usually negative).
    ActionScene {
        goal: Option<String>,
        conflict: Option<String>,
        disaster: Option<String>,
    },
    /// Character processes → weighs options → decides next action.
    Sequel {
        reaction: Option<String>,
        dilemma: Option<String>,
        decision: Option<String>,
    },
    /// Mix of action and reflection.
    Hybrid,
    /// Cannot be classified.
    Unclassified,
}

impl SceneType {
    /// Returns true if this is an action scene.
    pub fn is_action(&self) -> bool {
        matches!(self, SceneType::ActionScene { .. })
    }

    /// Returns true if this is a sequel.
    pub fn is_sequel(&self) -> bool {
        matches!(self, SceneType::Sequel { .. })
    }

    /// Numeric code for sequence analysis: 1.0 = action, 0.0 = sequel, 0.5 = hybrid/unknown.
    pub fn numeric(&self) -> f64 {
        match self {
            SceneType::ActionScene { .. } => 1.0,
            SceneType::Sequel { .. } => 0.0,
            SceneType::Hybrid => 0.5,
            SceneType::Unclassified => 0.5,
        }
    }
}

/// Classification of a situation's scene type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneClassification {
    pub situation_id: Uuid,
    pub chapter: usize,
    pub scene_type: SceneType,
    /// Confidence of classification.
    pub confidence: f64,
}

/// Scene-sequel rhythm analysis for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSequelAnalysis {
    pub narrative_id: String,
    pub classifications: Vec<SceneClassification>,
    /// Autocorrelation of scene type sequence. Negative = good alternation.
    pub autocorrelation: f64,
    /// Longest streak of consecutive action scenes.
    pub max_action_streak: usize,
    /// Longest streak of consecutive sequel scenes.
    pub max_sequel_streak: usize,
    /// Composite pacing score (0.0 = stalled, 1.0 = breathless).
    pub pacing_score: f64,
    /// Fraction that are action scenes.
    pub action_fraction: f64,
    /// Fraction that are sequels.
    pub sequel_fraction: f64,
}

// ─── KV ─────────────────────────────────────────────────────

fn scene_sequel_key(narrative_id: &str) -> Vec<u8> {
    format!("ss/{}", narrative_id).into_bytes()
}

pub fn store_analysis(hg: &Hypergraph, analysis: &SceneSequelAnalysis) -> Result<()> {
    let key = scene_sequel_key(&analysis.narrative_id);
    let val = serde_json::to_vec(analysis)?;
    hg.store().put(&key, &val)
}

pub fn load_analysis(hg: &Hypergraph, narrative_id: &str) -> Result<Option<SceneSequelAnalysis>> {
    let key = scene_sequel_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

// ─── Classification ─────────────────────────────────────────

/// Classify a situation as action scene or sequel using heuristic analysis.
pub fn classify_scene_type(
    hg: &Hypergraph,
    sit: &Situation,
    chapter: usize,
) -> SceneClassification {
    let text: String = sit
        .raw_content
        .iter()
        .map(|cb| cb.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let lower = text.to_lowercase();

    // Heuristic signals for action scenes
    let action_signals = [
        "attack", "fight", "run", "chase", "grab", "shout", "scream", "explode", "shoot", "sword",
        "punch", "dodge", "escape", "confront", "argue", "demand", "threaten", "slam", "crash",
        "pursue",
    ];
    let action_score: f64 = action_signals.iter().filter(|w| lower.contains(*w)).count() as f64;

    // Dialogue markers (scenes tend to have more dialogue)
    let dialogue_count = text.matches('"').count() / 2; // pairs
    let dialogue_score = (dialogue_count as f64 * 0.3).min(2.0);

    // Heuristic signals for sequels
    let sequel_signals = [
        "think",
        "wonder",
        "consider",
        "remember",
        "feel",
        "reflect",
        "realize",
        "decide",
        "ponder",
        "contemplate",
        "weigh",
        "hesitate",
        "question",
        "doubt",
        "hope",
        "fear",
        "regret",
        "recall",
    ];
    let sequel_score: f64 = sequel_signals.iter().filter(|w| lower.contains(*w)).count() as f64;

    // Participant action count
    let action_count = if let Ok(participants) = hg.get_participants_for_situation(&sit.id) {
        participants.iter().filter(|p| p.action.is_some()).count() as f64
    } else {
        0.0
    };

    let total_action = action_score + dialogue_score + action_count * 0.5;
    let total_sequel = sequel_score;

    if total_action > total_sequel + 1.0 {
        SceneClassification {
            situation_id: sit.id,
            chapter,
            scene_type: SceneType::ActionScene {
                goal: None,
                conflict: None,
                disaster: None,
            },
            confidence: (total_action / (total_action + total_sequel + 1.0)).min(0.95),
        }
    } else if total_sequel > total_action + 0.5 {
        SceneClassification {
            situation_id: sit.id,
            chapter,
            scene_type: SceneType::Sequel {
                reaction: None,
                dilemma: None,
                decision: None,
            },
            confidence: (total_sequel / (total_action + total_sequel + 1.0)).min(0.95),
        }
    } else if total_action > 0.0 || total_sequel > 0.0 {
        SceneClassification {
            situation_id: sit.id,
            chapter,
            scene_type: SceneType::Hybrid,
            confidence: 0.4,
        }
    } else {
        SceneClassification {
            situation_id: sit.id,
            chapter,
            scene_type: SceneType::Unclassified,
            confidence: 0.1,
        }
    }
}

/// Analyze scene-sequel rhythm for an entire narrative.
pub fn analyze_scene_sequel(hg: &Hypergraph, narrative_id: &str) -> Result<SceneSequelAnalysis> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(SceneSequelAnalysis {
            narrative_id: narrative_id.to_string(),
            classifications: Vec::new(),
            autocorrelation: 0.0,
            max_action_streak: 0,
            max_sequel_streak: 0,
            pacing_score: 0.5,
            action_fraction: 0.0,
            sequel_fraction: 0.0,
        });
    }

    // Sort by temporal order
    let mut sorted: Vec<&Situation> = situations.iter().collect();
    sorted.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    let mut classifications = Vec::new();
    for (chapter, sit) in sorted.iter().enumerate() {
        classifications.push(classify_scene_type(hg, sit, chapter));
    }

    // Compute sequence statistics
    let sequence: Vec<f64> = classifications
        .iter()
        .map(|c| c.scene_type.numeric())
        .collect();

    let autocorrelation = compute_autocorrelation(&sequence);

    let (max_action_streak, max_sequel_streak) = compute_streaks(&classifications);

    let action_count = classifications
        .iter()
        .filter(|c| c.scene_type.is_action())
        .count();
    let sequel_count = classifications
        .iter()
        .filter(|c| c.scene_type.is_sequel())
        .count();
    let n = classifications.len() as f64;

    let action_fraction = action_count as f64 / n;
    let sequel_fraction = sequel_count as f64 / n;

    // Pacing score: high action fraction + good alternation (negative autocorrelation) = high pacing
    let alternation_bonus = (-autocorrelation).max(0.0) * 0.3;
    let pacing_score = (action_fraction * 0.7 + alternation_bonus).clamp(0.0, 1.0);

    let analysis = SceneSequelAnalysis {
        narrative_id: narrative_id.to_string(),
        classifications,
        autocorrelation,
        max_action_streak,
        max_sequel_streak,
        pacing_score,
        action_fraction,
        sequel_fraction,
    };

    store_analysis(hg, &analysis)?;
    Ok(analysis)
}

/// Compute lag-1 autocorrelation of a numeric sequence.
fn compute_autocorrelation(seq: &[f64]) -> f64 {
    if seq.len() < 3 {
        return 0.0;
    }

    let n = seq.len() as f64;
    let mean = seq.iter().sum::<f64>() / n;
    let variance: f64 = seq.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    if variance < 1e-10 {
        return 0.0;
    }

    let covariance: f64 = seq
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / (n - 1.0);

    covariance / variance
}

/// Compute longest action and sequel streaks.
fn compute_streaks(classifications: &[SceneClassification]) -> (usize, usize) {
    let mut max_action = 0usize;
    let mut max_sequel = 0usize;
    let mut current_action = 0usize;
    let mut current_sequel = 0usize;

    for cls in classifications {
        if cls.scene_type.is_action() {
            current_action += 1;
            current_sequel = 0;
            max_action = max_action.max(current_action);
        } else if cls.scene_type.is_sequel() {
            current_sequel += 1;
            current_action = 0;
            max_sequel = max_sequel.max(current_sequel);
        } else {
            current_action = 0;
            current_sequel = 0;
        }
    }

    (max_action, max_sequel)
}

// ─── Inference Engine ───────────────────────────────────────

pub struct SceneSequelEngine;

impl InferenceEngine for SceneSequelEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::SceneSequel
    }

    fn estimate_cost(
        &self,
        _job: &crate::inference::types::InferenceJob,
        _hg: &Hypergraph,
    ) -> Result<u64> {
        Ok(3000)
    }

    fn execute(
        &self,
        job: &crate::inference::types::InferenceJob,
        hg: &Hypergraph,
    ) -> Result<InferenceResult> {
        let narrative_id = crate::analysis::extract_narrative_id(job)?;

        let analysis = analyze_scene_sequel(hg, narrative_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::SceneSequel,
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence: 0.7,
            explanation: Some(format!(
                "Pacing: {:.2}, action: {:.0}%, sequel: {:.0}%, autocorr: {:.2}",
                analysis.pacing_score,
                analysis.action_fraction * 100.0,
                analysis.sequel_fraction * 100.0,
                analysis.autocorrelation
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::{DateTime, Utc};
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_sit(hg: &Hypergraph, nid: &str, hour: i64, text: &str) -> Uuid {
        let start = DateTime::from_timestamp(1700000000 + hour * 3600, 0).unwrap();
        hg.create_situation(Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: crate::types::AllenInterval {
                start: Some(start),
                end: Some(start + chrono::Duration::hours(1)),
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
            narrative_id: Some(nid.into()),
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
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
    fn test_classify_action_scene() {
        let hg = test_hg();
        let nid = "action-test";
        let sid = make_sit(
            &hg,
            nid,
            0,
            r#""Stop!" he shouted. The sword flashed as he attacked. She dodged and ran, crashing through the door."#,
        );
        let sit = hg.get_situation(&sid).unwrap();
        let cls = classify_scene_type(&hg, &sit, 0);
        assert!(
            cls.scene_type.is_action(),
            "Should classify as action scene: {:?}",
            cls.scene_type
        );
    }

    #[test]
    fn test_classify_sequel() {
        let hg = test_hg();
        let nid = "sequel-test";
        let sid = make_sit(&hg, nid, 0,
            "She sat alone, thinking about what had happened. She wondered if she had made the right decision. She felt doubt creep in as she contemplated her options.");
        let sit = hg.get_situation(&sid).unwrap();
        let cls = classify_scene_type(&hg, &sit, 0);
        assert!(
            cls.scene_type.is_sequel(),
            "Should classify as sequel: {:?}",
            cls.scene_type
        );
    }

    #[test]
    fn test_autocorrelation_alternating() {
        // Perfect alternation: 1, 0, 1, 0, 1, 0
        let seq = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let ac = compute_autocorrelation(&seq);
        assert!(
            ac < 0.0,
            "Alternating sequence should have negative autocorrelation, got {}",
            ac
        );
    }

    #[test]
    fn test_autocorrelation_clustering() {
        // Clustered: 1, 1, 1, 0, 0, 0
        let seq = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let ac = compute_autocorrelation(&seq);
        assert!(
            ac > 0.0,
            "Clustered sequence should have positive autocorrelation, got {}",
            ac
        );
    }

    #[test]
    fn test_streaks() {
        let classifications = vec![
            SceneClassification {
                situation_id: Uuid::nil(),
                chapter: 0,
                scene_type: SceneType::ActionScene {
                    goal: None,
                    conflict: None,
                    disaster: None,
                },
                confidence: 0.8,
            },
            SceneClassification {
                situation_id: Uuid::nil(),
                chapter: 1,
                scene_type: SceneType::ActionScene {
                    goal: None,
                    conflict: None,
                    disaster: None,
                },
                confidence: 0.8,
            },
            SceneClassification {
                situation_id: Uuid::nil(),
                chapter: 2,
                scene_type: SceneType::ActionScene {
                    goal: None,
                    conflict: None,
                    disaster: None,
                },
                confidence: 0.8,
            },
            SceneClassification {
                situation_id: Uuid::nil(),
                chapter: 3,
                scene_type: SceneType::Sequel {
                    reaction: None,
                    dilemma: None,
                    decision: None,
                },
                confidence: 0.7,
            },
            SceneClassification {
                situation_id: Uuid::nil(),
                chapter: 4,
                scene_type: SceneType::Sequel {
                    reaction: None,
                    dilemma: None,
                    decision: None,
                },
                confidence: 0.7,
            },
        ];
        let (max_action, max_sequel) = compute_streaks(&classifications);
        assert_eq!(max_action, 3);
        assert_eq!(max_sequel, 2);
    }

    #[test]
    fn test_full_analysis() {
        let hg = test_hg();
        let nid = "rhythm-test";

        // Action scenes
        make_sit(
            &hg,
            nid,
            0,
            r#""Attack!" he shouted, drawing his sword. The fight was fierce."#,
        );
        // Sequel
        make_sit(
            &hg,
            nid,
            1,
            "She sat and wondered about her decision. She felt doubt and considered her options.",
        );
        // Action
        make_sit(
            &hg,
            nid,
            2,
            r#"The chase was on. He ran and dodged through the crowd. "Stop him!" they shouted."#,
        );

        let analysis = analyze_scene_sequel(&hg, nid).unwrap();
        assert_eq!(analysis.classifications.len(), 3);
        assert!(analysis.pacing_score > 0.0);
    }

    #[test]
    fn test_kv_persistence() {
        let hg = test_hg();
        let nid = "persist-ss";
        make_sit(&hg, nid, 0, "Action scene with fighting and shouting");
        let analysis = analyze_scene_sequel(&hg, nid).unwrap();
        let loaded = load_analysis(&hg, nid).unwrap().unwrap();
        assert_eq!(analysis.classifications.len(), loaded.classifications.len());
    }
}
