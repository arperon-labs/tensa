//! Narrative skeleton extraction and transplantation.
//!
//! Extracts the pure structural skeleton — arc shapes, game types,
//! commitment patterns, knowledge flows, causal topology — strips all
//! content, and enables re-skinning in a different domain.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::character_arcs::ArcType;
use crate::narrative::commitments::CommitmentType;

// ─── Types ──────────────────────────────────────────────────

/// A content-free narrative skeleton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSkeleton {
    pub source_narrative: String,
    /// Entity roles and arc types (no names, no content).
    pub entity_slots: Vec<EntitySlot>,
    /// Situation structure (no content, only relationships).
    pub situation_slots: Vec<SituationSlot>,
    /// Commitment pattern (type + relative positioning).
    pub commitment_pattern: Vec<CommitmentSlot>,
    /// Causal topology (edges between situation slots).
    pub causal_edges: Vec<(usize, usize)>,
    /// Total chapters/situation count.
    pub chapter_count: usize,
}

/// An entity slot in the skeleton (role, not identity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySlot {
    pub slot_id: usize,
    pub entity_type: String,
    pub role: String,
    pub arc_type: Option<ArcType>,
    /// Situation indices this entity participates in.
    pub participates_in: Vec<usize>,
}

/// A situation slot (structure, not content).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationSlot {
    pub slot_id: usize,
    pub chapter: usize,
    pub participant_slots: Vec<usize>,
    pub narrative_level: String,
    /// Relative temporal position (0.0 = start, 1.0 = end).
    pub relative_position: f64,
}

/// A commitment pattern slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentSlot {
    pub commitment_type: CommitmentType,
    /// Chapter as fraction of total (0.0–1.0).
    pub setup_position: f64,
    pub payoff_position: f64,
}

// ─── Extraction ─────────────────────────────────────────────

/// Extract the structural skeleton from a narrative.
pub fn extract_skeleton(hg: &Hypergraph, narrative_id: &str) -> Result<NarrativeSkeleton> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let mut sorted_sits = situations.clone();
    sorted_sits.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));
    let chapter_count = sorted_sits.len();

    let entities = hg.list_entities_by_narrative(narrative_id)?;

    // Build entity slots
    let entity_id_to_slot: HashMap<Uuid, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id, i))
        .collect();

    let sit_id_to_slot: HashMap<Uuid, usize> = sorted_sits
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i))
        .collect();

    let mut entity_slots: Vec<EntitySlot> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let role = e
                .properties
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            EntitySlot {
                slot_id: i,
                entity_type: format!("{:?}", e.entity_type),
                role,
                arc_type: None,
                participates_in: Vec::new(),
            }
        })
        .collect();

    // Build participation mapping
    for (sit_idx, sit) in sorted_sits.iter().enumerate() {
        if let Ok(participants) = hg.get_participants_for_situation(&sit.id) {
            for p in &participants {
                if let Some(&ent_slot) = entity_id_to_slot.get(&p.entity_id) {
                    entity_slots[ent_slot].participates_in.push(sit_idx);
                }
            }
        }
    }

    // Enrich with arc types
    let arcs = crate::narrative::character_arcs::list_character_arcs(hg, narrative_id)?;
    for arc in &arcs {
        if let Some(&slot) = entity_id_to_slot.get(&arc.character_id) {
            entity_slots[slot].arc_type = Some(arc.arc_type.clone());
        }
    }

    // Build situation slots
    let situation_slots: Vec<SituationSlot> = sorted_sits
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let participants: Vec<usize> = hg
                .get_participants_for_situation(&s.id)
                .unwrap_or_default()
                .iter()
                .filter_map(|p| entity_id_to_slot.get(&p.entity_id).copied())
                .collect();

            SituationSlot {
                slot_id: i,
                chapter: i,
                participant_slots: participants,
                narrative_level: format!("{:?}", s.narrative_level),
                relative_position: if chapter_count > 1 {
                    i as f64 / (chapter_count - 1) as f64
                } else {
                    0.0
                },
            }
        })
        .collect();

    // Build causal edges
    let mut causal_edges = Vec::new();
    for sit in &sorted_sits {
        if let Some(&to_slot) = sit_id_to_slot.get(&sit.id) {
            for cause in &sit.causes {
                if let Some(&from_slot) = sit_id_to_slot.get(&cause.from_situation) {
                    causal_edges.push((from_slot, to_slot));
                }
            }
        }
    }

    // Build commitment pattern
    let commitments = crate::narrative::commitments::list_commitments(hg, narrative_id)?;
    let commitment_pattern: Vec<CommitmentSlot> = commitments
        .iter()
        .map(|c| CommitmentSlot {
            commitment_type: c.commitment_type.clone(),
            setup_position: if chapter_count > 0 {
                c.setup_chapter as f64 / chapter_count as f64
            } else {
                0.0
            },
            payoff_position: c
                .payoff_chapter
                .map(|p| {
                    if chapter_count > 0 {
                        p as f64 / chapter_count as f64
                    } else {
                        0.0
                    }
                })
                .unwrap_or(1.0),
        })
        .collect();

    Ok(NarrativeSkeleton {
        source_narrative: narrative_id.to_string(),
        entity_slots,
        situation_slots,
        commitment_pattern,
        causal_edges,
        chapter_count,
    })
}

/// Compare structural similarity between two skeletons.
pub fn skeleton_similarity(a: &NarrativeSkeleton, b: &NarrativeSkeleton) -> f64 {
    let mut score = 0.0;
    let mut weight = 0.0;

    // Entity count similarity
    let ent_ratio = a.entity_slots.len().min(b.entity_slots.len()) as f64
        / a.entity_slots.len().max(b.entity_slots.len()).max(1) as f64;
    score += ent_ratio * 0.15;
    weight += 0.15;

    // Situation count similarity
    let sit_ratio = a.situation_slots.len().min(b.situation_slots.len()) as f64
        / a.situation_slots.len().max(b.situation_slots.len()).max(1) as f64;
    score += sit_ratio * 0.15;
    weight += 0.15;

    // Causal density similarity
    let causal_a = a.causal_edges.len() as f64 / a.situation_slots.len().max(1) as f64;
    let causal_b = b.causal_edges.len() as f64 / b.situation_slots.len().max(1) as f64;
    let causal_sim = 1.0 - (causal_a - causal_b).abs().min(1.0);
    score += causal_sim * 0.2;
    weight += 0.2;

    // Commitment pattern similarity
    let comm_ratio = a.commitment_pattern.len().min(b.commitment_pattern.len()) as f64
        / a.commitment_pattern
            .len()
            .max(b.commitment_pattern.len())
            .max(1) as f64;
    score += comm_ratio * 0.2;
    weight += 0.2;

    // Arc type distribution similarity
    let arcs_a: Vec<_> = a
        .entity_slots
        .iter()
        .filter_map(|e| e.arc_type.as_ref())
        .collect();
    let arcs_b: Vec<_> = b
        .entity_slots
        .iter()
        .filter_map(|e| e.arc_type.as_ref())
        .collect();
    let arc_overlap = arcs_a.iter().filter(|a| arcs_b.contains(a)).count();
    let arc_sim = arc_overlap as f64 / arcs_a.len().max(arcs_b.len()).max(1) as f64;
    score += arc_sim * 0.3;
    weight += 0.3;

    if weight > 0.0 {
        score / weight
    } else {
        0.0
    }
}

// ─── KV ─────────────────────────────────────────────────────

fn skeleton_key(narrative_id: &str) -> Vec<u8> {
    format!("nskel/{}", narrative_id).into_bytes()
}

pub fn store_skeleton(hg: &Hypergraph, skel: &NarrativeSkeleton) -> Result<()> {
    let key = skeleton_key(&skel.source_narrative);
    let val = serde_json::to_vec(skel)?;
    hg.store().put(&key, &val)
}

pub fn load_skeleton(hg: &Hypergraph, narrative_id: &str) -> Result<Option<NarrativeSkeleton>> {
    let key = skeleton_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_extract_empty_skeleton() {
        let hg = test_hg();
        let skel = extract_skeleton(&hg, "empty").unwrap();
        assert_eq!(skel.chapter_count, 0);
        assert!(skel.entity_slots.is_empty());
    }

    #[test]
    fn test_skeleton_similarity_identical() {
        let skel = NarrativeSkeleton {
            source_narrative: "test".into(),
            entity_slots: vec![EntitySlot {
                slot_id: 0,
                entity_type: "Actor".into(),
                role: "protagonist".into(),
                arc_type: Some(ArcType::PositiveChange),
                participates_in: vec![0, 1, 2],
            }],
            situation_slots: vec![
                SituationSlot {
                    slot_id: 0,
                    chapter: 0,
                    participant_slots: vec![0],
                    narrative_level: "Scene".into(),
                    relative_position: 0.0,
                },
                SituationSlot {
                    slot_id: 1,
                    chapter: 1,
                    participant_slots: vec![0],
                    narrative_level: "Scene".into(),
                    relative_position: 0.5,
                },
                SituationSlot {
                    slot_id: 2,
                    chapter: 2,
                    participant_slots: vec![0],
                    narrative_level: "Scene".into(),
                    relative_position: 1.0,
                },
            ],
            commitment_pattern: vec![CommitmentSlot {
                commitment_type: CommitmentType::ChekhovsGun,
                setup_position: 0.0,
                payoff_position: 0.8,
            }],
            causal_edges: vec![(0, 1), (1, 2)],
            chapter_count: 3,
        };
        let sim = skeleton_similarity(&skel, &skel);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "identical skeleton should have sim ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_skeleton_kv_persistence() {
        let hg = test_hg();
        let skel = NarrativeSkeleton {
            source_narrative: "persist-test".into(),
            entity_slots: vec![],
            situation_slots: vec![],
            commitment_pattern: vec![],
            causal_edges: vec![],
            chapter_count: 5,
        };
        store_skeleton(&hg, &skel).unwrap();
        let loaded = load_skeleton(&hg, "persist-test").unwrap().unwrap();
        assert_eq!(loaded.chapter_count, 5);
    }
}
