//! Shared primitives for emitting and indexing auto-generated causal edges.
//!
//! Four code paths emit weak sequential `Enabling` links — writer's
//! `apply_proposal`, the plan materializer, the ingestion chunk-level
//! fallback, and the one-shot backfill. All four build the same `CausalLink`
//! shape and swallow the same error variants; [`add_sequential_link`]
//! collapses that boilerplate. [`CausalIndex`] pre-fetches antecedent edges
//! once so detectors and idempotency probes don't re-scan `cr/` per
//! situation.

use std::collections::HashMap;

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::{CausalLink, CausalType, MaturityLevel, Situation};

/// Writer / materializer placement — chapter N enables chapter N+1.
pub const MECHANISM_SEQUENTIAL: &str = "sequential";
/// Ingestion per-chunk fallback when the LLM committed zero causal edges.
pub const MECHANISM_FALLBACK: &str = "sequential (fallback)";
/// One-shot repair for empty causal graphs.
pub const MECHANISM_BACKFILL: &str = "sequential (backfill)";

/// Emit a weak `Enabling` causal link. Swallows `CausalCycle` and
/// `SituationNotFound` so one bad pair in a batch doesn't abort the rest.
/// Returns `Ok(true)` when a new edge was written.
pub fn add_sequential_link(
    hg: &Hypergraph,
    from: Uuid,
    to: Uuid,
    mechanism: &'static str,
    strength: f32,
    maturity: MaturityLevel,
) -> Result<bool> {
    let link = CausalLink {
        from_situation: from,
        to_situation: to,
        mechanism: Some(mechanism.to_string()),
        strength,
        causal_type: CausalType::Enabling,
        maturity,
    };
    match hg.add_causal_link(link) {
        Ok(()) => Ok(true),
        Err(TensaError::CausalCycle { .. }) | Err(TensaError::SituationNotFound(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Pre-fetched map of causal antecedents per situation id. Building once
/// and querying many times avoids the per-situation `cr/` prefix scan that
/// each naive detector would otherwise do.
#[derive(Debug, Default)]
pub struct CausalIndex {
    antecedents: HashMap<Uuid, Vec<CausalLink>>,
}

impl CausalIndex {
    pub fn build(hg: &Hypergraph, situations: &[Situation]) -> Result<Self> {
        let mut antecedents: HashMap<Uuid, Vec<CausalLink>> = HashMap::new();
        for sit in situations {
            let edges = hg.get_antecedents(&sit.id)?;
            if !edges.is_empty() {
                antecedents.insert(sit.id, edges);
            }
        }
        Ok(Self { antecedents })
    }

    pub fn antecedents_of(&self, id: &Uuid) -> &[CausalLink] {
        self.antecedents
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// True when a causal edge exists between `a` and `b` in either
    /// direction. Used to keep backfill and adjacency-link emission
    /// idempotent.
    pub fn edge_exists_either_direction(&self, a: &Uuid, b: &Uuid) -> bool {
        self.antecedents_of(b)
            .iter()
            .any(|l| &l.from_situation == a)
            || self
                .antecedents_of(a)
                .iter()
                .any(|l| &l.from_situation == b)
    }

    pub fn total_edges(&self) -> usize {
        self.antecedents.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, NarrativeLevel, TimeGranularity,
    };
    use chrono::Utc;
    use std::sync::Arc;

    fn make_sit() -> Situation {
        let now = Utc::now();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
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
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("n1".into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
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

    #[test]
    fn add_sequential_link_writes_enabling_edge() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let a = make_sit();
        let b = make_sit();
        let (aid, bid) = (a.id, b.id);
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();
        let added = add_sequential_link(
            &hg,
            aid,
            bid,
            MECHANISM_SEQUENTIAL,
            0.5,
            MaturityLevel::Candidate,
        )
        .unwrap();
        assert!(added);
        let edges = hg.get_antecedents(&bid).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].causal_type, CausalType::Enabling);
    }

    #[test]
    fn add_sequential_link_swallows_cycle() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let a = make_sit();
        let b = make_sit();
        let (aid, bid) = (a.id, b.id);
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();
        add_sequential_link(
            &hg,
            aid,
            bid,
            MECHANISM_SEQUENTIAL,
            0.5,
            MaturityLevel::Candidate,
        )
        .unwrap();
        let added = add_sequential_link(
            &hg,
            bid,
            aid,
            MECHANISM_SEQUENTIAL,
            0.5,
            MaturityLevel::Candidate,
        )
        .unwrap();
        assert!(
            !added,
            "reverse edge should be rejected as cycle, not error"
        );
    }

    #[test]
    fn causal_index_roundtrip() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let a = make_sit();
        let b = make_sit();
        let c = make_sit();
        let (aid, bid, cid) = (a.id, b.id, c.id);
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();
        hg.create_situation(c).unwrap();
        add_sequential_link(
            &hg,
            aid,
            bid,
            MECHANISM_SEQUENTIAL,
            0.5,
            MaturityLevel::Candidate,
        )
        .unwrap();

        let sits = hg.list_situations_by_narrative("n1").unwrap();
        let idx = CausalIndex::build(&hg, &sits).unwrap();
        assert_eq!(idx.total_edges(), 1);
        assert!(idx.edge_exists_either_direction(&aid, &bid));
        assert!(idx.edge_exists_either_direction(&bid, &aid));
        assert!(!idx.edge_exists_either_direction(&aid, &cid));
    }
}
