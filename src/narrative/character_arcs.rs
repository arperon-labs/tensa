//! Character arc detection and tracking (Sprint D9.4).
//!
//! Macro-scale structural analysis: detect character transformation arcs,
//! classify arc types, extract want/need/lie/truth, and compute arc completeness.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::InferenceEngine;
use crate::types::InferenceResult;
use crate::types::{InferenceJobType, JobStatus};

// ─── Types ──────────────────────────────────────────────────

/// Character arc archetype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArcType {
    /// Pursues want → discovers need → transforms (most common).
    PositiveChange,
    /// Starts good → corrupted by pursuing want.
    NegativeCorruption,
    /// Discovers truth but it destroys them.
    NegativeDisillusionment,
    /// Already knows truth → tests it against challenges → remains unchanged.
    Flat,
    /// Loses innocence but gains wisdom.
    PositiveDisillusionment,
}

/// A character's narrative arc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterArc {
    pub character_id: Uuid,
    pub narrative_id: String,
    pub arc_type: ArcType,
    /// Conscious desire.
    pub want: String,
    /// Unconscious necessity.
    pub need: String,
    /// False belief (e.g., "I don't need anyone").
    pub lie: Option<String>,
    /// Reality discovered (e.g., "connection is strength").
    pub truth: Option<String>,
    /// Motivation at each situation point (situation_id, valence score).
    pub motivation_trajectory: Vec<(Uuid, f64)>,
    /// Distance between initial and final motivation vectors.
    pub transformation_magnitude: f64,
    /// Arc completeness (0.0–1.0): midpoint turn? dark night? resolution?
    pub completeness: f64,
    /// Midpoint situation (where the arc turns).
    pub midpoint: Option<Uuid>,
    /// Dark night of the soul (motivation furthest from start and end).
    pub dark_night: Option<Uuid>,
}

// ─── KV ─────────────────────────────────────────────────────

fn arc_key(narrative_id: &str, character_id: &Uuid) -> Vec<u8> {
    format!("ca/{}/{}", narrative_id, character_id).into_bytes()
}

pub fn store_character_arc(hg: &Hypergraph, arc: &CharacterArc) -> Result<()> {
    let key = arc_key(&arc.narrative_id, &arc.character_id);
    let val = serde_json::to_vec(arc)?;
    hg.store().put(&key, &val)
}

pub fn load_character_arc(
    hg: &Hypergraph,
    narrative_id: &str,
    character_id: &Uuid,
) -> Result<Option<CharacterArc>> {
    let key = arc_key(narrative_id, character_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn list_character_arcs(hg: &Hypergraph, narrative_id: &str) -> Result<Vec<CharacterArc>> {
    let prefix = format!("ca/{}/", narrative_id).into_bytes();
    let items = hg.store().prefix_scan(&prefix)?;
    let mut out = Vec::new();
    for (_k, v) in items {
        if let Ok(arc) = serde_json::from_slice::<CharacterArc>(&v) {
            out.push(arc);
        }
    }
    Ok(out)
}

// ─── Detection ──────────────────────────────────────────────

/// Detect the character arc for a given entity in a narrative.
pub fn detect_character_arc(
    hg: &Hypergraph,
    narrative_id: &str,
    character_id: Uuid,
) -> Result<CharacterArc> {
    // Get all situations this character participates in, ordered temporally
    let participations = hg.get_situations_for_entity(&character_id)?;
    let mut sit_data: Vec<(Uuid, f64, DateTime<Utc>)> = Vec::new();

    for p in &participations {
        if let Ok(sit) = hg.get_situation(&p.situation_id) {
            if sit.narrative_id.as_deref() != Some(narrative_id) {
                continue;
            }
            // Compute a valence score from participation
            let valence = compute_participation_valence(p, &sit);
            let time = sit.temporal.start.unwrap_or(sit.created_at);
            sit_data.push((sit.id, valence, time));
        }
    }

    sit_data.sort_by(|a, b| a.2.cmp(&b.2));

    if sit_data.is_empty() {
        return Ok(CharacterArc {
            character_id,
            narrative_id: narrative_id.to_string(),
            arc_type: ArcType::Flat,
            want: String::new(),
            need: String::new(),
            lie: None,
            truth: None,
            motivation_trajectory: Vec::new(),
            transformation_magnitude: 0.0,
            completeness: 0.0,
            midpoint: None,
            dark_night: None,
        });
    }

    let trajectory: Vec<(Uuid, f64)> = sit_data.iter().map(|(id, v, _)| (*id, *v)).collect();

    let first_valence = trajectory.first().map(|(_, v)| *v).unwrap_or(0.0);
    let last_valence = trajectory.last().map(|(_, v)| *v).unwrap_or(0.0);

    // Transformation magnitude
    let magnitude = (last_valence - first_valence).abs();

    // Find midpoint (point of greatest directional change)
    let mid_idx = trajectory.len() / 2;
    let midpoint = trajectory.get(mid_idx).map(|(id, _)| *id);

    // Find dark night (minimum valence point)
    let dark_night = trajectory
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| *id);

    // Classify arc type
    let arc_type = classify_arc_type(first_valence, last_valence, &trajectory);

    // Extract want/need from entity properties (heuristic)
    let entity = hg.get_entity(&character_id)?;
    let want = entity
        .properties
        .get("want")
        .or(entity.properties.get("goal"))
        .and_then(|v| v.as_str())
        .unwrap_or("(not specified)")
        .to_string();
    let need = entity
        .properties
        .get("need")
        .and_then(|v| v.as_str())
        .unwrap_or("(not specified)")
        .to_string();
    let lie = entity
        .properties
        .get("lie")
        .and_then(|v| v.as_str())
        .map(String::from);
    let truth = entity
        .properties
        .get("truth")
        .and_then(|v| v.as_str())
        .map(String::from);

    // Completeness: does it have a midpoint turn + dark night + resolution?
    let has_midpoint_turn = if trajectory.len() >= 3 {
        let mid_val = trajectory[mid_idx].1;
        (mid_val - first_valence).abs() > 0.2 && (mid_val - last_valence).abs() > 0.2
    } else {
        false
    };
    let has_dark_night = dark_night.is_some()
        && trajectory
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, v)| *v < first_valence.min(last_valence) - 0.1)
            .unwrap_or(false);
    let has_resolution = magnitude > 0.3 || matches!(arc_type, ArcType::Flat);

    let completeness = [has_midpoint_turn, has_dark_night, has_resolution]
        .iter()
        .filter(|&&x| x)
        .count() as f64
        / 3.0;

    let arc = CharacterArc {
        character_id,
        narrative_id: narrative_id.to_string(),
        arc_type,
        want,
        need,
        lie,
        truth,
        motivation_trajectory: trajectory,
        transformation_magnitude: magnitude,
        completeness,
        midpoint,
        dark_night,
    };

    store_character_arc(hg, &arc)?;
    Ok(arc)
}

/// Compute a valence score for a participation.
fn compute_participation_valence(
    p: &crate::types::Participation,
    sit: &crate::types::Situation,
) -> f64 {
    let mut valence = 0.0;

    // Payoff contributes to valence
    if let Some(ref payoff) = p.payoff {
        if let Some(n) = payoff.as_f64() {
            valence += n;
        } else if let Some(n) = payoff.as_i64() {
            valence += n as f64;
        }
    }

    // Role-based baseline
    valence += match p.role {
        crate::types::Role::Protagonist => 0.5,
        crate::types::Role::Antagonist => -0.5,
        crate::types::Role::Target => -1.0,
        _ => 0.0,
    };

    // Content-based sentiment (quick heuristic)
    let text: String = sit
        .raw_content
        .iter()
        .map(|cb| cb.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let lower = text.to_lowercase();

    let positive_words = [
        "happy", "love", "succeed", "win", "joy", "hope", "peace", "victory", "friend", "trust",
    ];
    let negative_words = [
        "sad", "hate", "fail", "lose", "grief", "fear", "war", "defeat", "betray", "death",
    ];

    let pos = positive_words.iter().filter(|w| lower.contains(*w)).count() as f64;
    let neg = negative_words.iter().filter(|w| lower.contains(*w)).count() as f64;

    valence += (pos - neg) * 0.3;
    valence.clamp(-3.0, 3.0)
}

/// Classify arc type from valence trajectory.
fn classify_arc_type(first: f64, last: f64, trajectory: &[(Uuid, f64)]) -> ArcType {
    let delta = last - first;
    let min_valence = trajectory.iter().map(|(_, v)| *v).fold(f64::MAX, f64::min);
    let max_valence = trajectory.iter().map(|(_, v)| *v).fold(f64::MIN, f64::max);

    if delta.abs() < 0.3 {
        ArcType::Flat
    } else if delta > 0.5 {
        if min_valence < first - 0.5 {
            ArcType::PositiveDisillusionment // Dropped then rose
        } else {
            ArcType::PositiveChange
        }
    } else if delta < -0.5 {
        if max_valence > first + 0.5 {
            ArcType::NegativeDisillusionment // Rose then crashed
        } else {
            ArcType::NegativeCorruption
        }
    } else if delta > 0.0 {
        ArcType::PositiveChange
    } else {
        ArcType::NegativeCorruption
    }
}

// ─── Inference Engine ───────────────────────────────────────

pub struct CharacterArcEngine;

impl InferenceEngine for CharacterArcEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CharacterArc
    }

    fn estimate_cost(
        &self,
        _job: &crate::inference::types::InferenceJob,
        _hg: &Hypergraph,
    ) -> Result<u64> {
        Ok(5000)
    }

    fn execute(
        &self,
        job: &crate::inference::types::InferenceJob,
        hg: &Hypergraph,
    ) -> Result<InferenceResult> {
        let narrative_id = crate::analysis::extract_narrative_id(job)?;

        // If target_id is non-nil, detect arc for that specific character
        if job.target_id != Uuid::nil() {
            let arc = detect_character_arc(hg, narrative_id, job.target_id)?;
            return Ok(InferenceResult {
                job_id: job.id.clone(),
                job_type: InferenceJobType::CharacterArc,
                target_id: job.target_id,
                result: serde_json::to_value(&arc)?,
                confidence: 0.7,
                explanation: Some(format!(
                    "Arc type: {:?}, completeness: {:.0}%",
                    arc.arc_type,
                    arc.completeness * 100.0
                )),
                status: JobStatus::Completed,
                created_at: job.created_at,
                completed_at: Some(Utc::now()),
            });
        }

        // Otherwise detect all character arcs in the narrative
        let entities = hg
            .list_entities_by_narrative(narrative_id)?
            .into_iter()
            .filter(|e| e.entity_type == crate::types::EntityType::Actor)
            .collect::<Vec<_>>();
        let mut arcs = Vec::new();
        for e in &entities {
            let arc = detect_character_arc(hg, narrative_id, e.id)?;
            if !arc.motivation_trajectory.is_empty() {
                arcs.push(arc);
            }
        }

        let avg_completeness = if arcs.is_empty() {
            0.0
        } else {
            arcs.iter().map(|a| a.completeness).sum::<f64>() / arcs.len() as f64
        };

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::CharacterArc,
            target_id: job.target_id,
            result: serde_json::to_value(&arcs)?,
            confidence: 0.7,
            explanation: Some(format!(
                "{} character arcs, avg completeness {:.0}%",
                arcs.len(),
                avg_completeness * 100.0
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
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_arc_type_classification() {
        let id = Uuid::nil();
        // Flat arc
        assert_eq!(
            classify_arc_type(0.5, 0.5, &[(id, 0.5), (id, 0.6), (id, 0.5)]),
            ArcType::Flat
        );

        // Positive change
        assert_eq!(
            classify_arc_type(-1.0, 1.5, &[(id, -1.0), (id, 0.0), (id, 1.5)]),
            ArcType::PositiveChange
        );

        // Negative corruption
        assert_eq!(
            classify_arc_type(1.5, -1.0, &[(id, 1.5), (id, 0.0), (id, -1.0)]),
            ArcType::NegativeCorruption
        );

        // Positive disillusionment (drop then rise)
        assert_eq!(
            classify_arc_type(0.0, 1.0, &[(id, 0.0), (id, -1.0), (id, 1.0)]),
            ArcType::PositiveDisillusionment
        );
    }

    #[test]
    fn test_detect_character_arc() {
        let hg = test_hg();
        let nid = "arc-test";

        let char_id = hg.create_entity(Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Hero", "want": "Find the treasure", "need": "Learn humility"}),
            beliefs: None, embedding: None,
            narrative_id: Some(nid.into()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9, confidence_breakdown: None,
            provenance: vec![], extraction_method: None,
            created_at: Utc::now(), updated_at: Utc::now(),
            deleted_at: None, transaction_time: None,
        }).unwrap();

        // Create situations with increasing temporal order
        for i in 0..5 {
            let start = DateTime::from_timestamp(1700000000 + i * 3600, 0).unwrap();
            let sid = hg
                .create_situation(Situation {
                    id: Uuid::now_v7(),
                    properties: serde_json::Value::Null,
                    name: Some(format!("Scene {}", i)),
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
                    raw_content: vec![ContentBlock::text(if i < 3 {
                        "The hero struggles and fails"
                    } else {
                        "The hero succeeds with joy and victory"
                    })],
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
                .unwrap();

            hg.add_participant(Participation {
                entity_id: char_id,
                situation_id: sid,
                role: Role::Protagonist,
                info_set: None,
                action: Some(format!("act_{}", i)),
                payoff: Some(serde_json::json!(if i < 3 { -1 } else { 2 })),
                seq: 0,
            })
            .unwrap();
        }

        let arc = detect_character_arc(&hg, nid, char_id).unwrap();
        assert_eq!(arc.want, "Find the treasure");
        assert_eq!(arc.need, "Learn humility");
        assert_eq!(arc.motivation_trajectory.len(), 5);
        assert!(arc.transformation_magnitude > 0.0);
    }

    #[test]
    fn test_arc_kv_persistence() {
        let hg = test_hg();
        let arc = CharacterArc {
            character_id: Uuid::now_v7(),
            narrative_id: "persist".to_string(),
            arc_type: ArcType::PositiveChange,
            want: "Freedom".into(),
            need: "Responsibility".into(),
            lie: Some("Rules are for fools".into()),
            truth: Some("Structure enables freedom".into()),
            motivation_trajectory: vec![
                (Uuid::nil(), -1.0),
                (Uuid::nil(), 0.5),
                (Uuid::nil(), 2.0),
            ],
            transformation_magnitude: 3.0,
            completeness: 0.67,
            midpoint: None,
            dark_night: None,
        };
        store_character_arc(&hg, &arc).unwrap();
        let loaded = load_character_arc(&hg, "persist", &arc.character_id)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.arc_type, ArcType::PositiveChange);
        assert_eq!(loaded.want, "Freedom");
    }

    #[test]
    fn test_empty_character() {
        let hg = test_hg();
        let char_id = Uuid::now_v7();
        hg.create_entity(Entity {
            id: char_id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Nobody"}),
            beliefs: None,
            embedding: None,
            narrative_id: Some("empty".into()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.5,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap();

        let arc = detect_character_arc(&hg, "empty", char_id).unwrap();
        assert_eq!(arc.arc_type, ArcType::Flat);
        assert_eq!(arc.motivation_trajectory.len(), 0);
    }
}
