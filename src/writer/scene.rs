//! Scene-schema helpers for the binder / corkboard / outliner views.
//!
//! Sprint W7 adds `synopsis`, `manuscript_order`, `parent_situation_id`, `label`,
//! `status`, and `keywords` to [`Situation`]. This module centralises the invariants
//! the backend must enforce:
//!
//! - [`check_parent_cycle`] — reject `parent_situation_id` assignments that would
//!   create a cycle in the scene tree.
//! - [`manuscript_sort_key`] — the total-ordering key used by manuscript export
//!   and every Studio view that renders scenes in narrated order.
//! - [`word_count`] — pure helper used by the Outliner column.

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::types::{ContentType, Situation};
use crate::Hypergraph;

/// Maximum depth of the parent-situation chain. Paranoid bound: manuscript binders
/// almost never exceed Part → Chapter → Scene → Beat → Line (5), so 16 is plenty
/// without letting a malformed KV produce an infinite traversal.
pub const MAX_PARENT_DEPTH: usize = 16;

/// Walk upward from `proposed_parent` to verify that setting
/// `update_target.parent_situation_id = Some(proposed_parent)` would not create a
/// cycle in the scene tree, and that the chain stays under [`MAX_PARENT_DEPTH`].
///
/// Self-parenting is rejected by the caller before this helper runs.
///
/// # Errors
///
/// - [`TensaError::InvalidQuery`] if a cycle is detected or the chain exceeds the
///   depth bound.
/// - [`TensaError::SituationNotFound`] if a link in the chain has been deleted.
pub fn check_parent_cycle(
    hypergraph: &Hypergraph,
    update_target: &Uuid,
    proposed_parent: &Uuid,
) -> Result<()> {
    let mut current = *proposed_parent;
    for depth in 0..MAX_PARENT_DEPTH {
        if current == *update_target {
            return Err(TensaError::InvalidQuery(format!(
                "parent_situation_id would create a cycle (depth {depth})"
            )));
        }
        // Deleted / missing parents are a data error, not a cycle; surface clearly.
        let parent_sit = hypergraph.get_situation(&current)?;
        match parent_sit.parent_situation_id {
            None => return Ok(()),
            Some(next) => current = next,
        }
    }
    Err(TensaError::InvalidQuery(format!(
        "parent_situation_id chain exceeds max depth {MAX_PARENT_DEPTH}"
    )))
}

/// Total-ordering key for manuscript narrated order.
///
/// Primary key: `manuscript_order` (writer-curated). Situations without one sort
/// *after* all ordered ones using their temporal start as the secondary key, which
/// keeps ingested material in chronological order until the writer touches it.
/// The tie-break on `id` (v7 UUID → time-ordered) guarantees a stable sort.
pub fn manuscript_sort_key(sit: &Situation) -> (u32, i64, Uuid) {
    let order_bucket = sit.manuscript_order.unwrap_or(u32::MAX);
    let time_key = sit
        .temporal
        .start
        .unwrap_or(sit.created_at)
        .timestamp_nanos_opt()
        .unwrap_or(i64::MAX);
    (order_bucket, time_key, sit.id)
}

/// Word count of a situation's prose content. Used by the Outliner column and by
/// writing-session statistics.
///
/// Only `Text` and `Dialogue` blocks contribute; other content types (observation,
/// document, media) carry metadata rather than prose and are counted zero.
pub fn word_count(sit: &Situation) -> usize {
    sit.raw_content
        .iter()
        .filter(|b| matches!(b.content_type, ContentType::Text | ContentType::Dialogue))
        .map(|b| count_words(&b.content))
        .sum()
}

fn count_words(s: &str) -> usize {
    s.split_whitespace().filter(|w| !w.is_empty()).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ContentType, ExtractionMethod, MaturityLevel, NarrativeLevel,
        TimeGranularity,
    };
    use chrono::Utc;
    use std::sync::Arc;

    fn make_scene(parent: Option<Uuid>, level: NarrativeLevel) -> Situation {
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
            raw_content: vec![ContentBlock::text("Hello there friend, how are you?")],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("n1".into()),
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
    fn test_word_count_counts_dialogue_and_text() {
        let s = make_scene(None, NarrativeLevel::Scene);
        // "Hello there friend, how are you?" — 6 words
        assert_eq!(word_count(&s), 6);
    }

    #[test]
    fn test_word_count_ignores_non_prose_blocks() {
        let mut s = make_scene(None, NarrativeLevel::Scene);
        let wc_without = word_count(&s);
        s.raw_content.push(ContentBlock {
            content_type: ContentType::Observation,
            content: "these three words".into(),
            source: None,
        });
        let wc_with = word_count(&s);
        // Observation blocks are metadata, not prose.
        assert_eq!(wc_with, wc_without);
    }

    #[test]
    fn test_manuscript_sort_key_prefers_ordered_before_unordered() {
        let mut ordered = make_scene(None, NarrativeLevel::Scene);
        ordered.manuscript_order = Some(10);
        let unordered = make_scene(None, NarrativeLevel::Scene);
        assert!(manuscript_sort_key(&ordered) < manuscript_sort_key(&unordered));
    }

    #[test]
    fn test_manuscript_sort_key_stable_within_bucket() {
        let mut a = make_scene(None, NarrativeLevel::Scene);
        let mut b = make_scene(None, NarrativeLevel::Scene);
        a.manuscript_order = Some(5);
        b.manuscript_order = Some(5);
        // Same bucket → time + id break the tie deterministically.
        let ka = manuscript_sort_key(&a);
        let kb = manuscript_sort_key(&b);
        assert_ne!(ka, kb);
    }

    #[test]
    fn test_check_parent_cycle_rejects_direct_cycle() {
        let hg = setup();
        let a = make_scene(None, NarrativeLevel::Scene);
        let b = make_scene(Some(a.id), NarrativeLevel::Scene);
        let a_id = a.id;
        let b_id = b.id;
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();

        // Making A a child of B would close the loop A→B→A.
        let err = check_parent_cycle(&hg, &a_id, &b_id).unwrap_err();
        assert!(matches!(err, TensaError::InvalidQuery(_)));
    }

    #[test]
    fn test_check_parent_cycle_accepts_valid_reparent() {
        let hg = setup();
        let root = make_scene(None, NarrativeLevel::Arc);
        let chapter = make_scene(Some(root.id), NarrativeLevel::Sequence);
        let scene = make_scene(None, NarrativeLevel::Scene);
        let root_id = root.id;
        let chapter_id = chapter.id;
        let scene_id = scene.id;
        hg.create_situation(root).unwrap();
        hg.create_situation(chapter).unwrap();
        hg.create_situation(scene).unwrap();

        // Reparenting scene under chapter is fine.
        check_parent_cycle(&hg, &scene_id, &chapter_id).unwrap();
        // And under root is fine.
        check_parent_cycle(&hg, &scene_id, &root_id).unwrap();
    }

    #[test]
    fn test_check_parent_cycle_rejects_indirect_cycle() {
        let hg = setup();
        // Build chain: A ← B ← C (C parent=B, B parent=A).
        let a = make_scene(None, NarrativeLevel::Arc);
        let b = make_scene(Some(a.id), NarrativeLevel::Sequence);
        let c = make_scene(Some(b.id), NarrativeLevel::Scene);
        let a_id = a.id;
        let c_id = c.id;
        hg.create_situation(a).unwrap();
        hg.create_situation(b).unwrap();
        hg.create_situation(c).unwrap();

        // Making A's parent = C closes A→?→C→B→A.
        let err = check_parent_cycle(&hg, &a_id, &c_id).unwrap_err();
        assert!(matches!(err, TensaError::InvalidQuery(_)));
    }
}
