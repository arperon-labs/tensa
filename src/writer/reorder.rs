//! Batch reorder helper used by `POST /narratives/:id/reorder` (Sprint W8).
//!
//! Accepts an ordered list of `{situation_id, parent_id?}` and rewrites both
//! `manuscript_order` and `parent_situation_id` in a single pass. The order
//! values are re-densified to `STEP, 2*STEP, …` so later drag-inserts have
//! room to slot a new scene between two existing ones without shifting all
//! neighbours.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::writer::scene::check_parent_cycle;
use crate::Hypergraph;

/// Gap between densified positions; leaves 999 slots between neighbours.
pub const REORDER_STEP: u32 = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReorderEntry {
    pub situation_id: Uuid,
    #[serde(default)]
    pub parent_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReorderReport {
    pub updated: Vec<ReorderedSituation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReorderedSituation {
    pub situation_id: Uuid,
    pub manuscript_order: u32,
    pub parent_situation_id: Option<Uuid>,
}

/// Apply a reorder proposal to a narrative.
///
/// Validates:
/// - every id exists and belongs to `narrative_id`
/// - no duplicate ids in the input
/// - every `parent_id`, if set, exists (may or may not be in the input list)
/// - no parent cycles after the reorder
pub fn apply_reorder(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entries: &[ReorderEntry],
) -> Result<ReorderReport> {
    let mut seen: HashSet<Uuid> = HashSet::with_capacity(entries.len());
    for e in entries {
        if !seen.insert(e.situation_id) {
            return Err(TensaError::InvalidQuery(format!(
                "duplicate situation_id {} in reorder payload",
                e.situation_id
            )));
        }
    }

    // Phase 1: load + validate narrative membership + validate any parent id exists.
    for e in entries {
        let sit = hypergraph.get_situation(&e.situation_id)?;
        if sit.narrative_id.as_deref() != Some(narrative_id) {
            return Err(TensaError::InvalidQuery(format!(
                "situation {} does not belong to narrative {narrative_id}",
                e.situation_id
            )));
        }
        if let Some(pid) = e.parent_id {
            if pid == e.situation_id {
                return Err(TensaError::InvalidQuery(format!(
                    "situation {} cannot be its own parent",
                    e.situation_id
                )));
            }
            // Make sure the parent exists.
            let _ = hypergraph.get_situation(&pid)?;
        }
    }

    // Phase 2: apply in order, densifying positions. For cycle checking we use the
    // live KV state after each write so an internally-consistent batch is accepted
    // even when two scenes swap parents within the same payload.
    let mut report = ReorderReport {
        updated: Vec::with_capacity(entries.len()),
    };
    for (idx, entry) in entries.iter().enumerate() {
        let new_order = (idx as u32 + 1).saturating_mul(REORDER_STEP);
        if let Some(pid) = entry.parent_id {
            // Validate cycles against *current* KV state before writing this row.
            check_parent_cycle(hypergraph, &entry.situation_id, &pid)?;
        }
        hypergraph.update_situation(&entry.situation_id, |s| {
            s.manuscript_order = Some(new_order);
            s.parent_situation_id = entry.parent_id;
        })?;
        report.updated.push(ReorderedSituation {
            situation_id: entry.situation_id,
            manuscript_order: new_order,
            parent_situation_id: entry.parent_id,
        });
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Situation,
        TimeGranularity,
    };
    use chrono::Utc;
    use std::sync::Arc;

    fn make_scene(parent: Option<Uuid>, narrative: &str) -> Situation {
        Situation {
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
            raw_content: vec![ContentBlock::text("test")],
            narrative_level: NarrativeLevel::Scene,
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
            parent_situation_id: parent,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_reorder_densifies_order() {
        let hg = setup();
        let a = make_scene(None, "n1");
        let b = make_scene(None, "n1");
        let c = make_scene(None, "n1");
        let ids = [a.id, b.id, c.id];
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();
        hg.create_situation(c).unwrap();

        let report = apply_reorder(
            &hg,
            "n1",
            &[
                ReorderEntry {
                    situation_id: ids[2],
                    parent_id: None,
                },
                ReorderEntry {
                    situation_id: ids[0],
                    parent_id: None,
                },
                ReorderEntry {
                    situation_id: ids[1],
                    parent_id: None,
                },
            ],
        )
        .unwrap();
        let orders: Vec<u32> = report.updated.iter().map(|r| r.manuscript_order).collect();
        assert_eq!(
            orders,
            vec![REORDER_STEP, 2 * REORDER_STEP, 3 * REORDER_STEP]
        );

        // Confirmed via KV.
        let fresh_c = hg.get_situation(&ids[2]).unwrap();
        assert_eq!(fresh_c.manuscript_order, Some(REORDER_STEP));
    }

    #[test]
    fn test_reorder_rejects_foreign_narrative() {
        let hg = setup();
        let a = make_scene(None, "n1");
        let b = make_scene(None, "n2");
        let ids = [a.id, b.id];
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();

        let err = apply_reorder(
            &hg,
            "n1",
            &[
                ReorderEntry {
                    situation_id: ids[0],
                    parent_id: None,
                },
                ReorderEntry {
                    situation_id: ids[1],
                    parent_id: None,
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(err, TensaError::InvalidQuery(_)));
    }

    #[test]
    fn test_reorder_rejects_duplicate_ids() {
        let hg = setup();
        let a = make_scene(None, "n1");
        let a_id = a.id;
        hg.create_situation(a).unwrap();
        let err = apply_reorder(
            &hg,
            "n1",
            &[
                ReorderEntry {
                    situation_id: a_id,
                    parent_id: None,
                },
                ReorderEntry {
                    situation_id: a_id,
                    parent_id: None,
                },
            ],
        )
        .unwrap_err();
        assert!(matches!(err, TensaError::InvalidQuery(_)));
    }

    #[test]
    fn test_reorder_reparents_within_batch() {
        let hg = setup();
        let root = make_scene(None, "n1");
        let chapter = make_scene(None, "n1");
        let scene = make_scene(None, "n1");
        let ids = [root.id, chapter.id, scene.id];
        hg.create_situation(root).unwrap();
        hg.create_situation(chapter).unwrap();
        hg.create_situation(scene).unwrap();

        apply_reorder(
            &hg,
            "n1",
            &[
                ReorderEntry {
                    situation_id: ids[0],
                    parent_id: None,
                },
                ReorderEntry {
                    situation_id: ids[1],
                    parent_id: Some(ids[0]),
                },
                ReorderEntry {
                    situation_id: ids[2],
                    parent_id: Some(ids[1]),
                },
            ],
        )
        .unwrap();

        assert_eq!(
            hg.get_situation(&ids[1]).unwrap().parent_situation_id,
            Some(ids[0])
        );
        assert_eq!(
            hg.get_situation(&ids[2]).unwrap().parent_situation_id,
            Some(ids[1])
        );
    }
}
