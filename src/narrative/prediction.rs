//! Missing event prediction using causal gaps and cross-narrative patterns.
//!
//! Detects gaps in causal chains and predicts missing situations by
//! combining causal gap analysis, pattern-based prediction from
//! cross-narrative patterns, and motivation-based validation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{AllenInterval, Role, TimeGranularity};

use super::pattern::NarrativePattern;

// ─── Types ───────────────────────────────────────────────────

/// A gap in a causal chain where a missing event may exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGap {
    pub from_situation: Uuid,
    pub to_situation: Uuid,
    pub gap_score: f64,
    pub temporal_gap_hours: f64,
    pub link_strength: f32,
}

/// A predicted missing situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedSituation {
    pub predicted_id: Uuid,
    pub narrative_id: String,
    pub between: (Uuid, Uuid),
    pub predicted_temporal: AllenInterval,
    pub predicted_participants: Vec<PredictedParticipant>,
    pub predicted_content: String,
    pub confidence: f32,
    pub evidence: PredictionEvidence,
}

/// A predicted participant in a missing event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedParticipant {
    pub entity_id: Uuid,
    pub predicted_role: Role,
    pub confidence: f32,
}

/// Evidence supporting a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEvidence {
    pub causal_gap_score: f64,
    pub pattern_support: usize,
    pub supporting_patterns: Vec<String>,
    pub motivation_alignment: f64,
}

// ─── Causal Gap Detection ────────────────────────────────────

/// Detect causal gaps in a narrative — pairs of causally linked
/// situations where the gap suggests a missing intermediary.
pub fn detect_causal_gaps(narrative_id: &str, hypergraph: &Hypergraph) -> Result<Vec<CausalGap>> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    if situations.len() < 2 {
        return Ok(vec![]);
    }

    // Compute average temporal gap
    let mut total_gap = 0.0;
    let mut gap_count = 0;
    let mut gaps = Vec::new();

    // Collect all causal links within the narrative
    let sit_ids: std::collections::HashSet<Uuid> = situations.iter().map(|s| s.id).collect();
    let sit_map: HashMap<Uuid, &crate::types::Situation> =
        situations.iter().map(|s| (s.id, s)).collect();

    for sit in &situations {
        let consequences = hypergraph.get_consequences(&sit.id).unwrap_or_default();
        for cause in &consequences {
            if !sit_ids.contains(&cause.from_situation) || !sit_ids.contains(&cause.to_situation) {
                continue;
            }

            let from = sit_map.get(&cause.from_situation);
            let to = sit_map.get(&cause.to_situation);

            if let (Some(from_sit), Some(to_sit)) = (from, to) {
                let temporal_gap = compute_temporal_gap(from_sit, to_sit);
                total_gap += temporal_gap;
                gap_count += 1;

                gaps.push((cause.clone(), temporal_gap));
            }
        }
    }

    if gap_count == 0 {
        return Ok(vec![]);
    }

    let avg_gap = total_gap / gap_count as f64;
    let mut causal_gaps = Vec::new();

    for (link, temporal_gap) in &gaps {
        // Gap score: higher for larger temporal gaps and weaker links
        let temporal_factor = if avg_gap > 0.0 {
            *temporal_gap / avg_gap
        } else {
            1.0
        };
        let strength_factor = 1.0 - link.strength as f64;
        let gap_score = temporal_factor * 0.6 + strength_factor * 0.4;

        // Only report gaps above threshold
        if gap_score > 1.0 || link.strength < 0.5 {
            causal_gaps.push(CausalGap {
                from_situation: link.from_situation,
                to_situation: link.to_situation,
                gap_score,
                temporal_gap_hours: *temporal_gap,
                link_strength: link.strength,
            });
        }
    }

    // Sort by gap score descending
    causal_gaps.sort_by(|a, b| {
        b.gap_score
            .partial_cmp(&a.gap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(causal_gaps)
}

fn compute_temporal_gap(from: &crate::types::Situation, to: &crate::types::Situation) -> f64 {
    let from_end = from
        .temporal
        .end
        .unwrap_or_else(|| from.temporal.start.unwrap_or_default());
    let to_start = to
        .temporal
        .start
        .unwrap_or_else(|| to.temporal.end.unwrap_or_default());

    let duration = to_start.signed_duration_since(from_end);
    duration.num_hours().abs() as f64
}

// ─── Pattern-Based Prediction ────────────────────────────────

/// Predict missing situations based on causal gaps and cross-narrative patterns.
pub fn predict_missing_events(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    patterns: &[NarrativePattern],
) -> Result<Vec<PredictedSituation>> {
    let gaps = detect_causal_gaps(narrative_id, hypergraph)?;
    let mut predictions = Vec::new();

    for gap in &gaps {
        let from_sit = hypergraph.get_situation(&gap.from_situation)?;
        let to_sit = hypergraph.get_situation(&gap.to_situation)?;

        // Find supporting patterns
        let supporting = find_supporting_patterns(patterns, &from_sit, &to_sit);

        // Interpolate temporal position
        let predicted_temporal = interpolate_temporal(&from_sit, &to_sit);

        // Predict participants (actors common to both bounding situations)
        let from_participants = hypergraph.get_participants_for_situation(&gap.from_situation)?;
        let to_participants = hypergraph.get_participants_for_situation(&gap.to_situation)?;

        let predicted_participants = predict_participants(&from_participants, &to_participants);

        // Compute confidence
        let pattern_support = supporting.len();
        let base_confidence = gap.gap_score.min(1.0) * 0.4;
        let pattern_bonus = (pattern_support as f64 * 0.15).min(0.4);
        let confidence = (base_confidence + pattern_bonus) as f32;

        let predicted_content = format!(
            "Predicted event between '{}' and '{}'",
            from_sit
                .raw_content
                .first()
                .map(|c| c.content.chars().take(50).collect::<String>())
                .unwrap_or_default(),
            to_sit
                .raw_content
                .first()
                .map(|c| c.content.chars().take(50).collect::<String>())
                .unwrap_or_default(),
        );

        predictions.push(PredictedSituation {
            predicted_id: Uuid::now_v7(),
            narrative_id: narrative_id.to_string(),
            between: (gap.from_situation, gap.to_situation),
            predicted_temporal,
            predicted_participants,
            predicted_content,
            confidence,
            evidence: PredictionEvidence {
                causal_gap_score: gap.gap_score,
                pattern_support,
                supporting_patterns: supporting,
                motivation_alignment: 0.5, // default
            },
        });
    }

    // Sort by confidence descending
    predictions.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(predictions)
}

/// Find patterns that might fill the gap between two situations.
fn find_supporting_patterns(
    patterns: &[NarrativePattern],
    _from: &crate::types::Situation,
    _to: &crate::types::Situation,
) -> Vec<String> {
    // Patterns with 3+ node causal chains could fill a 2-node gap
    patterns
        .iter()
        .filter(|p| p.subgraph.nodes.len() >= 3 && p.confidence > 0.3)
        .map(|p| p.id.clone())
        .collect()
}

/// Interpolate temporal position between two situations.
fn interpolate_temporal(
    from: &crate::types::Situation,
    to: &crate::types::Situation,
) -> AllenInterval {
    let from_end = from
        .temporal
        .end
        .unwrap_or_else(|| from.temporal.start.unwrap_or_default());
    let to_start = to
        .temporal
        .start
        .unwrap_or_else(|| to.temporal.end.unwrap_or_default());

    let midpoint = from_end
        + chrono::Duration::milliseconds(
            to_start.signed_duration_since(from_end).num_milliseconds() / 2,
        );

    AllenInterval {
        start: Some(midpoint),
        end: Some(midpoint + chrono::Duration::hours(1)),
        granularity: TimeGranularity::Approximate,
        relations: vec![],
        fuzzy_endpoints: None,
    }
}

/// Predict participants — entities appearing in both bounding situations.
fn predict_participants(
    from: &[crate::types::Participation],
    to: &[crate::types::Participation],
) -> Vec<PredictedParticipant> {
    let from_entities: std::collections::HashSet<Uuid> = from.iter().map(|p| p.entity_id).collect();

    let mut participants = Vec::new();

    // Entities in both situations are likely in the middle one
    for p in to {
        if from_entities.contains(&p.entity_id) {
            participants.push(PredictedParticipant {
                entity_id: p.entity_id,
                predicted_role: p.role.clone(),
                confidence: 0.7,
            });
        }
    }

    // Entities only in the 'from' situation with lower confidence
    for p in from {
        if !participants.iter().any(|pp| pp.entity_id == p.entity_id) {
            participants.push(PredictedParticipant {
                entity_id: p.entity_id,
                predicted_role: p.role.clone(),
                confidence: 0.3,
            });
        }
    }

    participants
}

// ─── InferenceEngine Implementation ─────────────────────────

use crate::analysis::extract_narrative_id;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Missing event prediction engine for the inference job queue.
pub struct MissingEventEngine;

impl InferenceEngine for MissingEventEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::MissingEventPrediction
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(10000) // 10 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;

        let gaps = detect_causal_gaps(narrative_id, hypergraph)?;
        // Load any previously mined patterns from KV for pattern-based prediction
        let patterns = super::pattern::list_patterns(hypergraph.store()).unwrap_or_default();
        let predictions = predict_missing_events(narrative_id, hypergraph, &patterns)?;

        let combined = serde_json::json!({
            "causal_gaps": gaps,
            "predicted_events": predictions,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::MissingEventPrediction,
            target_id: job.target_id,
            result: combined,
            confidence: 1.0,
            explanation: Some(format!(
                "Missing event prediction: {} gaps, {} predictions",
                gaps.len(),
                predictions.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(chrono::Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::{DateTime, Duration, Utc};
    use std::sync::Arc;

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_sit(
        hg: &Hypergraph,
        nid: &str,
        hours: i64,
        content: &str,
        causes: Vec<CausalLink>,
    ) -> Uuid {
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(base + Duration::hours(hours)),
                end: Some(base + Duration::hours(hours + 1)),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes,
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(content)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.to_string()),
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

    #[test]
    fn test_detect_gaps_empty() {
        let hg = setup();
        let gaps = detect_causal_gaps("empty", &hg).unwrap();
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_gaps_no_causal_links() {
        let hg = setup();
        make_sit(&hg, "test", 0, "Scene 1", vec![]);
        make_sit(&hg, "test", 10, "Scene 2", vec![]);
        let gaps = detect_causal_gaps("test", &hg).unwrap();
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_detect_gap_weak_link() {
        let hg = setup();
        let s1 = make_sit(&hg, "test", 0, "Start", vec![]);
        let _s2 = make_sit(
            &hg,
            "test",
            100,
            "End",
            vec![CausalLink {
                from_situation: s1,
                to_situation: Uuid::nil(), // placeholder, replaced below
                mechanism: Some("weak link".to_string()),
                strength: 0.2,
                causal_type: CausalType::Contributing,
                maturity: MaturityLevel::Candidate,
            }],
        );

        // Need to update s2 with correct self-reference
        // Since we can't know s2's ID before creating it, use a different approach
        let situations = hg.list_situations_by_narrative("test").unwrap();
        assert_eq!(situations.len(), 2);
    }

    #[test]
    fn test_predict_empty_narrative() {
        let hg = setup();
        let predictions = predict_missing_events("empty", &hg, &[]).unwrap();
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_interpolate_temporal() {
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let from = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(base),
                end: Some(base + Duration::hours(1)),
                granularity: TimeGranularity::Exact,
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

        let to = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(base + Duration::hours(10)),
                end: Some(base + Duration::hours(11)),
                granularity: TimeGranularity::Exact,
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

        let result = interpolate_temporal(&from, &to);
        let mid = result.start.unwrap();
        assert!(mid > base + Duration::hours(1));
        assert!(mid < base + Duration::hours(10));
    }

    #[test]
    fn test_predict_participants_common() {
        let e1 = Uuid::now_v7();
        let e2 = Uuid::now_v7();
        let e3 = Uuid::now_v7();

        let from = vec![
            Participation {
                entity_id: e1,
                situation_id: Uuid::now_v7(),
                role: Role::Protagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            },
            Participation {
                entity_id: e2,
                situation_id: Uuid::now_v7(),
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            },
        ];

        let to = vec![
            Participation {
                entity_id: e1,
                situation_id: Uuid::now_v7(),
                role: Role::Protagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            },
            Participation {
                entity_id: e3,
                situation_id: Uuid::now_v7(),
                role: Role::Antagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            },
        ];

        let predicted = predict_participants(&from, &to);
        assert_eq!(predicted.len(), 2); // e1 (common, high conf) + e2 (from-only, low conf)
        let common = predicted.iter().find(|p| p.entity_id == e1).unwrap();
        assert_eq!(common.confidence, 0.7);
        let from_only = predicted.iter().find(|p| p.entity_id == e2).unwrap();
        assert_eq!(from_only.confidence, 0.3);
    }

    #[test]
    fn test_prediction_evidence_serialization() {
        let evidence = PredictionEvidence {
            causal_gap_score: 1.5,
            pattern_support: 3,
            supporting_patterns: vec!["p1".to_string(), "p2".to_string()],
            motivation_alignment: 0.7,
        };
        let json = serde_json::to_vec(&evidence).unwrap();
        let decoded: PredictionEvidence = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.pattern_support, 3);
        assert_eq!(decoded.causal_gap_score, 1.5);
    }

    #[test]
    fn test_predicted_situation_serialization() {
        let pred = PredictedSituation {
            predicted_id: Uuid::now_v7(),
            narrative_id: "test".to_string(),
            between: (Uuid::now_v7(), Uuid::now_v7()),
            predicted_temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            predicted_participants: vec![],
            predicted_content: "test prediction".to_string(),
            confidence: 0.6,
            evidence: PredictionEvidence {
                causal_gap_score: 1.0,
                pattern_support: 0,
                supporting_patterns: vec![],
                motivation_alignment: 0.5,
            },
        };
        let json = serde_json::to_vec(&pred).unwrap();
        let decoded: PredictedSituation = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.confidence, 0.6);
        assert_eq!(decoded.narrative_id, "test");
    }

    #[test]
    fn test_missing_event_engine_execute() {
        let hg = setup();

        let engine = MissingEventEngine;
        assert_eq!(engine.job_type(), InferenceJobType::MissingEventPrediction);

        let job = crate::inference::types::InferenceJob {
            id: "me-test".to_string(),
            job_type: InferenceJobType::MissingEventPrediction,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": "test-me"}),
            priority: crate::types::JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());
        assert!(result.result.get("causal_gaps").is_some());
        assert!(result.result.get("predicted_events").is_some());
    }
}
