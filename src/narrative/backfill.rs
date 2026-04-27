//! One-shot backfill for missing causal adjacency links.
//!
//! Writer- and legacy-ingested narratives often have zero causal edges in the
//! `c/` / `cr/` KV prefixes — Workshop's causal detectors, graph-projection,
//! and narrative-diameter analyses all degenerate when the graph is empty.
//! [`backfill_adjacent_causal_links`] walks the narrative's situations in
//! manuscript order and synthesises weak sequential `Enabling` links between
//! same-level adjacent pairs.
//!
//! Safe to run multiple times: pairs that already have an edge (in either
//! direction) are skipped.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::causal_helpers::{add_sequential_link, CausalIndex, MECHANISM_BACKFILL};
use crate::types::{MaturityLevel, NarrativeLevel};
use crate::writer::scene::manuscript_sort_key;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillReport {
    pub narrative_id: String,
    pub situations_total: usize,
    /// New causal edges written to the store.
    pub links_added: usize,
    /// Adjacent pairs that already had an edge (either direction).
    pub pairs_skipped_existing: usize,
    /// Adjacent pairs skipped because they cross narrative levels
    /// (synthesising across hierarchy would produce spurious edges).
    pub pairs_skipped_cross_level: usize,
}

pub fn backfill_adjacent_causal_links(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<BackfillReport> {
    let mut situations = hg.list_situations_by_narrative(narrative_id)?;
    situations.sort_by(|a, b| manuscript_sort_key(a).cmp(&manuscript_sort_key(b)));

    let mut report = BackfillReport {
        narrative_id: narrative_id.to_string(),
        situations_total: situations.len(),
        links_added: 0,
        pairs_skipped_existing: 0,
        pairs_skipped_cross_level: 0,
    };
    if situations.len() < 2 {
        return Ok(report);
    }

    // One prefix scan per situation, then all pair-adjacency probes hit the
    // in-memory map instead of re-scanning `cr/` twice per pair.
    let index = CausalIndex::build(hg, &situations)?;

    for pair in situations.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if !is_significant_level(a.narrative_level) || !is_significant_level(b.narrative_level) {
            // Beat/Event/Line — backfilling adjacency at those levels
            // would drown the store.
            continue;
        }
        if a.narrative_level != b.narrative_level {
            report.pairs_skipped_cross_level += 1;
            continue;
        }
        if index.edge_exists_either_direction(&a.id, &b.id) {
            report.pairs_skipped_existing += 1;
            continue;
        }
        if add_sequential_link(
            hg,
            a.id,
            b.id,
            MECHANISM_BACKFILL,
            0.3,
            MaturityLevel::Candidate,
        )? {
            report.links_added += 1;
        } else {
            // Cycle via a non-adjacent ancestor — treat as a skip, not an
            // abort, so one pair doesn't stop the whole repair.
            report.pairs_skipped_existing += 1;
        }
    }
    Ok(report)
}

fn is_significant_level(level: NarrativeLevel) -> bool {
    matches!(
        level,
        NarrativeLevel::Scene | NarrativeLevel::Sequence | NarrativeLevel::Arc
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, CausalLink, CausalType, ContentBlock, ExtractionMethod, MaturityLevel,
        NarrativeLevel, Situation, TimeGranularity,
    };
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    fn make_scene(nid: &str, order: u32, level: NarrativeLevel) -> Situation {
        let now = Utc::now();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(format!("scene-{}", order)),
            description: None,
            temporal: AllenInterval {
                start: Some(now),
                end: None,
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
            raw_content: vec![ContentBlock::text("x")],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: Some(order),
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn backfill_adds_adjacent_links_in_empty_graph() {
        let hg = setup();
        for i in 0..3 {
            hg.create_situation(make_scene("n1", i, NarrativeLevel::Scene))
                .unwrap();
        }
        let report = backfill_adjacent_causal_links(&hg, "n1").unwrap();
        assert_eq!(report.situations_total, 3);
        assert_eq!(report.links_added, 2);
        assert_eq!(report.pairs_skipped_existing, 0);
    }

    #[test]
    fn backfill_is_idempotent() {
        let hg = setup();
        for i in 0..3 {
            hg.create_situation(make_scene("n1", i, NarrativeLevel::Scene))
                .unwrap();
        }
        let first = backfill_adjacent_causal_links(&hg, "n1").unwrap();
        let second = backfill_adjacent_causal_links(&hg, "n1").unwrap();
        assert_eq!(first.links_added, 2);
        assert_eq!(second.links_added, 0);
        assert_eq!(second.pairs_skipped_existing, 2);
    }

    #[test]
    fn backfill_skips_cross_level_pairs() {
        let hg = setup();
        hg.create_situation(make_scene("n1", 0, NarrativeLevel::Arc))
            .unwrap();
        hg.create_situation(make_scene("n1", 1, NarrativeLevel::Scene))
            .unwrap();
        hg.create_situation(make_scene("n1", 2, NarrativeLevel::Scene))
            .unwrap();
        let report = backfill_adjacent_causal_links(&hg, "n1").unwrap();
        assert_eq!(report.links_added, 1);
        assert_eq!(report.pairs_skipped_cross_level, 1);
    }

    #[test]
    fn backfill_respects_existing_reverse_edge() {
        let hg = setup();
        let s0 = make_scene("n1", 0, NarrativeLevel::Scene);
        let s1 = make_scene("n1", 1, NarrativeLevel::Scene);
        let (id0, id1) = (s0.id, s1.id);
        hg.create_situation(s0).unwrap();
        hg.create_situation(s1).unwrap();
        // Reverse edge 1 -> 0 already exists. Backfill would be a cycle the
        // other way; we skip it.
        hg.add_causal_link(CausalLink {
            from_situation: id1,
            to_situation: id0,
            mechanism: Some("test".into()),
            strength: 0.9,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Validated,
        })
        .unwrap();
        let report = backfill_adjacent_causal_links(&hg, "n1").unwrap();
        assert_eq!(report.links_added, 0);
        assert_eq!(report.pairs_skipped_existing, 1);
    }

    #[test]
    fn backfill_handles_single_situation() {
        let hg = setup();
        hg.create_situation(make_scene("n1", 0, NarrativeLevel::Scene))
            .unwrap();
        let report = backfill_adjacent_causal_links(&hg, "n1").unwrap();
        assert_eq!(report.situations_total, 1);
        assert_eq!(report.links_added, 0);
    }
}
