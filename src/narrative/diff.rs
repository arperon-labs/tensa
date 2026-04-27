//! Narrative diff — structural comparison of two narrative versions.
//!
//! Compares two narratives at the structural level, not textual: entity
//! changes, causal chain breaks, commitment status changes, dramatic
//! irony shifts, emotional arc changes, and overall fingerprint distance.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// A structural diff between two versions of a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeDiff {
    pub narrative_a: String,
    pub narrative_b: String,
    pub entity_changes: EntityChanges,
    pub situation_changes: SituationChanges,
    pub commitment_changes: Vec<CommitmentChange>,
    pub arc_changes: Vec<ArcChange>,
    pub pacing_delta: PacingDelta,
    /// Overall structural distance (0.0 = identical, higher = more different).
    pub structural_distance: f64,
    pub summary: String,
}

/// Entity additions, removals, and role changes between versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityChanges {
    pub added: Vec<EntitySummary>,
    pub removed: Vec<EntitySummary>,
    /// Entities present in both but with changed properties.
    pub modified: Vec<EntityModification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    pub id: Uuid,
    pub name: String,
    pub entity_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityModification {
    pub id: Uuid,
    pub name: String,
    pub changes: Vec<String>,
}

/// Situation additions, removals, and reorderings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationChanges {
    pub added_count: usize,
    pub removed_count: usize,
    pub reordered_count: usize,
    /// Causal links that were broken by the revision.
    pub broken_causal_links: usize,
    /// New causal links created by the revision.
    pub new_causal_links: usize,
}

/// A change in commitment status between versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentChange {
    pub element: String,
    pub status_a: String,
    pub status_b: String,
    pub description: String,
}

/// A change in character arc between versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcChange {
    pub character_name: String,
    pub arc_type_a: String,
    pub arc_type_b: String,
    pub completeness_delta: f64,
}

/// Pacing metrics delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacingDelta {
    pub pacing_score_a: f64,
    pub pacing_score_b: f64,
    pub rhythm_improvement: f64,
}

// ─── Diff Computation ───────────────────────────────────────

/// Compare two narratives structurally.
pub fn diff_narratives(
    hg: &Hypergraph,
    narrative_a: &str,
    narrative_b: &str,
) -> Result<NarrativeDiff> {
    let entity_changes = diff_entities(hg, narrative_a, narrative_b)?;
    let situation_changes = diff_situations(hg, narrative_a, narrative_b)?;
    let commitment_changes = diff_commitments(hg, narrative_a, narrative_b)?;
    let arc_changes = diff_arcs(hg, narrative_a, narrative_b)?;
    let pacing_delta = diff_pacing(hg, narrative_a, narrative_b)?;

    // Composite structural distance
    let entity_dist = (entity_changes.added.len() + entity_changes.removed.len()) as f64 * 0.1;
    let sit_dist = (situation_changes.added_count + situation_changes.removed_count) as f64 * 0.05
        + situation_changes.broken_causal_links as f64 * 0.15;
    let commit_dist = commitment_changes.len() as f64 * 0.1;
    let arc_dist = arc_changes
        .iter()
        .map(|a| a.completeness_delta.abs())
        .sum::<f64>()
        * 0.2;
    let pace_dist = (pacing_delta.pacing_score_a - pacing_delta.pacing_score_b).abs() * 0.3;

    let structural_distance =
        (entity_dist + sit_dist + commit_dist + arc_dist + pace_dist).min(1.0);

    let summary = format!(
        "Entities: +{}/-{}/~{}. Situations: +{}/-{}. Causal links: {} broken, {} new. {} commitment changes. Pacing: {:.2} → {:.2}.",
        entity_changes.added.len(),
        entity_changes.removed.len(),
        entity_changes.modified.len(),
        situation_changes.added_count,
        situation_changes.removed_count,
        situation_changes.broken_causal_links,
        situation_changes.new_causal_links,
        commitment_changes.len(),
        pacing_delta.pacing_score_a,
        pacing_delta.pacing_score_b,
    );

    Ok(NarrativeDiff {
        narrative_a: narrative_a.to_string(),
        narrative_b: narrative_b.to_string(),
        entity_changes,
        situation_changes,
        commitment_changes,
        arc_changes,
        pacing_delta,
        structural_distance,
        summary,
    })
}

fn diff_entities(hg: &Hypergraph, na: &str, nb: &str) -> Result<EntityChanges> {
    let ents_a = hg.list_entities_by_narrative(na)?;
    let ents_b = hg.list_entities_by_narrative(nb)?;

    let names_a: HashMap<String, &crate::types::Entity> = ents_a
        .iter()
        .filter_map(|e| {
            e.properties
                .get("name")
                .and_then(|n| n.as_str())
                .map(|n| (n.to_string(), e))
        })
        .collect();
    let names_b: HashMap<String, &crate::types::Entity> = ents_b
        .iter()
        .filter_map(|e| {
            e.properties
                .get("name")
                .and_then(|n| n.as_str())
                .map(|n| (n.to_string(), e))
        })
        .collect();

    let set_a: HashSet<&str> = names_a.keys().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = names_b.keys().map(|s| s.as_str()).collect();

    let added: Vec<EntitySummary> = set_b
        .difference(&set_a)
        .filter_map(|name| {
            names_b.get(*name).map(|e| EntitySummary {
                id: e.id,
                name: name.to_string(),
                entity_type: format!("{:?}", e.entity_type),
            })
        })
        .collect();

    let removed: Vec<EntitySummary> = set_a
        .difference(&set_b)
        .filter_map(|name| {
            names_a.get(*name).map(|e| EntitySummary {
                id: e.id,
                name: name.to_string(),
                entity_type: format!("{:?}", e.entity_type),
            })
        })
        .collect();

    let modified: Vec<EntityModification> = set_a
        .intersection(&set_b)
        .filter_map(|name| {
            let ea = names_a.get(*name)?;
            let eb = names_b.get(*name)?;
            let mut changes = Vec::new();
            if ea.entity_type != eb.entity_type {
                changes.push(format!("type: {:?} → {:?}", ea.entity_type, eb.entity_type));
            }
            if (ea.confidence - eb.confidence).abs() > 0.05 {
                changes.push(format!(
                    "confidence: {:.2} → {:.2}",
                    ea.confidence, eb.confidence
                ));
            }
            if changes.is_empty() {
                None
            } else {
                Some(EntityModification {
                    id: eb.id,
                    name: name.to_string(),
                    changes,
                })
            }
        })
        .collect();

    Ok(EntityChanges {
        added,
        removed,
        modified,
    })
}

fn diff_situations(hg: &Hypergraph, na: &str, nb: &str) -> Result<SituationChanges> {
    let sits_a = hg.list_situations_by_narrative(na)?;
    let sits_b = hg.list_situations_by_narrative(nb)?;

    let names_a: HashSet<String> = sits_a.iter().filter_map(|s| s.name.clone()).collect();
    let names_b: HashSet<String> = sits_b.iter().filter_map(|s| s.name.clone()).collect();

    let added_count = names_b.difference(&names_a).count();
    let removed_count = names_a.difference(&names_b).count();

    // Count causal link changes
    let causal_a: HashSet<(Uuid, Uuid)> = sits_a
        .iter()
        .flat_map(|s| s.causes.iter().map(|c| (c.from_situation, s.id)))
        .collect();
    let causal_b: HashSet<(Uuid, Uuid)> = sits_b
        .iter()
        .flat_map(|s| s.causes.iter().map(|c| (c.from_situation, s.id)))
        .collect();

    let broken_causal_links = causal_a.difference(&causal_b).count();
    let new_causal_links = causal_b.difference(&causal_a).count();

    Ok(SituationChanges {
        added_count,
        removed_count,
        reordered_count: 0, // Would need fabula comparison for this
        broken_causal_links,
        new_causal_links,
    })
}

fn diff_commitments(hg: &Hypergraph, na: &str, nb: &str) -> Result<Vec<CommitmentChange>> {
    let comms_a = crate::narrative::commitments::list_commitments(hg, na)?;
    let comms_b = crate::narrative::commitments::list_commitments(hg, nb)?;

    let map_a: HashMap<&str, &crate::narrative::commitments::NarrativeCommitment> = comms_a
        .iter()
        .map(|c| (c.tracked_element.as_str(), c))
        .collect();
    let map_b: HashMap<&str, &crate::narrative::commitments::NarrativeCommitment> = comms_b
        .iter()
        .map(|c| (c.tracked_element.as_str(), c))
        .collect();

    let mut changes = Vec::new();
    for (element, ca) in &map_a {
        if let Some(cb) = map_b.get(element) {
            if ca.status != cb.status {
                changes.push(CommitmentChange {
                    element: element.to_string(),
                    status_a: format!("{:?}", ca.status),
                    status_b: format!("{:?}", cb.status),
                    description: format!(
                        "'{}' changed from {:?} to {:?}",
                        element, ca.status, cb.status
                    ),
                });
            }
        } else {
            changes.push(CommitmentChange {
                element: element.to_string(),
                status_a: format!("{:?}", ca.status),
                status_b: "(removed)".into(),
                description: format!("'{}' was removed in version B", element),
            });
        }
    }
    for (element, _cb) in &map_b {
        if !map_a.contains_key(element) {
            changes.push(CommitmentChange {
                element: element.to_string(),
                status_a: "(new)".into(),
                status_b: "(added)".into(),
                description: format!("'{}' was added in version B", element),
            });
        }
    }

    Ok(changes)
}

fn diff_arcs(hg: &Hypergraph, na: &str, nb: &str) -> Result<Vec<ArcChange>> {
    let arcs_a = crate::narrative::character_arcs::list_character_arcs(hg, na)?;
    let arcs_b = crate::narrative::character_arcs::list_character_arcs(hg, nb)?;

    let mut changes = Vec::new();

    for arc_a in &arcs_a {
        if let Some(arc_b) = arcs_b.iter().find(|b| b.character_id == arc_a.character_id) {
            if arc_a.arc_type != arc_b.arc_type
                || (arc_a.completeness - arc_b.completeness).abs() > 0.1
            {
                let name = hg
                    .get_entity(&arc_a.character_id)
                    .ok()
                    .and_then(|e| {
                        e.properties
                            .get("name")
                            .and_then(|n| n.as_str())
                            .map(String::from)
                    })
                    .unwrap_or_else(|| arc_a.character_id.to_string());

                changes.push(ArcChange {
                    character_name: name,
                    arc_type_a: format!("{:?}", arc_a.arc_type),
                    arc_type_b: format!("{:?}", arc_b.arc_type),
                    completeness_delta: arc_b.completeness - arc_a.completeness,
                });
            }
        }
    }

    Ok(changes)
}

fn diff_pacing(hg: &Hypergraph, na: &str, nb: &str) -> Result<PacingDelta> {
    let ss_a = crate::narrative::scene_sequel::load_analysis(hg, na)?;
    let ss_b = crate::narrative::scene_sequel::load_analysis(hg, nb)?;

    let score_a = ss_a.map(|s| s.pacing_score).unwrap_or(0.5);
    let score_b = ss_b.map(|s| s.pacing_score).unwrap_or(0.5);

    Ok(PacingDelta {
        pacing_score_a: score_a,
        pacing_score_b: score_b,
        rhythm_improvement: score_b - score_a,
    })
}

// ─── KV ─────────────────────────────────────────────────────

fn diff_key(na: &str, nb: &str) -> Vec<u8> {
    format!("ndiff/{}:{}", na, nb).into_bytes()
}

pub fn store_diff(hg: &Hypergraph, diff: &NarrativeDiff) -> Result<()> {
    let key = diff_key(&diff.narrative_a, &diff.narrative_b);
    let val = serde_json::to_vec(diff)?;
    hg.store().put(&key, &val)
}

pub fn load_diff(hg: &Hypergraph, na: &str, nb: &str) -> Result<Option<NarrativeDiff>> {
    let key = diff_key(na, nb);
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
    fn test_diff_empty_narratives() {
        let hg = test_hg();
        let diff = diff_narratives(&hg, "story-v1", "story-v2").unwrap();
        assert_eq!(diff.structural_distance, 0.0);
        assert!(diff.entity_changes.added.is_empty());
    }

    #[test]
    fn test_diff_kv_persistence() {
        let hg = test_hg();
        let diff = NarrativeDiff {
            narrative_a: "v1".into(),
            narrative_b: "v2".into(),
            entity_changes: EntityChanges {
                added: vec![],
                removed: vec![EntitySummary {
                    id: Uuid::nil(),
                    name: "Removed Character".into(),
                    entity_type: "Actor".into(),
                }],
                modified: vec![],
            },
            situation_changes: SituationChanges {
                added_count: 2,
                removed_count: 1,
                reordered_count: 0,
                broken_causal_links: 1,
                new_causal_links: 2,
            },
            commitment_changes: vec![],
            arc_changes: vec![],
            pacing_delta: PacingDelta {
                pacing_score_a: 0.6,
                pacing_score_b: 0.75,
                rhythm_improvement: 0.15,
            },
            structural_distance: 0.35,
            summary: "Test diff".into(),
        };
        store_diff(&hg, &diff).unwrap();
        let loaded = load_diff(&hg, "v1", "v2").unwrap().unwrap();
        assert_eq!(loaded.entity_changes.removed.len(), 1);
        assert_eq!(loaded.structural_distance, 0.35);
    }
}
