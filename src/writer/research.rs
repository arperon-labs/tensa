//! Research-panel module: Sprint W9.
//!
//! TENSA's research model is not a dumb file drawer. Every scene has an
//! implicit research context assembled from the hypergraph's existing
//! knowledge: sources attributed to the scene, sources attributed to the
//! participant entities, contentions between participants, narrative-scoped
//! pinned facts, and top-k semantically-similar source chunks. Writers can
//! also pin per-scene `ResearchNote` items (quotes, clippings, links,
//! freeform notes) as a lightweight margin-of-the-page workspace.
//!
//! All of this is surfaced in the Studio Research panel as the writer moves
//! the caret between scenes, and is consumed by W10's fact-check overlay
//! and W11's cited generation.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::narrative::continuity;
use crate::source::{ContentionLink, SourceAttribution};
use crate::store::KVStore;
use crate::types::PinnedFact;
use crate::Hypergraph;

/// Freeform writer-authored research snippet pinned to a single scene.
///
/// This is intentionally lighter than a registered `Source`: a source is a
/// first-class entity in the system (publisher, article, trust score,
/// attributions, contentions). A research note is the margin of a writer's
/// notebook — quote, clipping, link, or one-sentence reminder — associated
/// with one scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchNote {
    pub id: Uuid,
    pub narrative_id: String,
    pub situation_id: Uuid,
    pub kind: ResearchNoteKind,
    /// Plain body of the note / quote / URL / clipping.
    pub body: String,
    /// If this note was promoted from a source chunk (W9 "promote chunk to note"),
    /// this points to the originating ChunkStore chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_chunk_id: Option<Uuid>,
    /// Registered source id this note is attached to, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchNoteKind {
    /// Direct quotation from a source.
    Quote,
    /// Paraphrased clipping or excerpt.
    Clipping,
    /// URL / external reference.
    Link,
    /// Writer's freeform note (nothing to cite).
    Note,
}

/// Aggregated research context for a single scene, assembled from the
/// hypergraph on demand.
///
/// Consumed by the Studio Research panel and by `writer::factcheck` (W10)
/// as the evidence pool to match atomic claims against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneResearchContext {
    pub situation_id: Uuid,
    pub narrative_id: Option<String>,
    /// Sources attributed directly to this scene.
    pub scene_sources: Vec<SourceAttribution>,
    /// Sources attributed to any of this scene's participant entities.
    pub participant_sources: Vec<ParticipantSource>,
    /// Contentions this scene is part of.
    pub contentions: Vec<ContentionLink>,
    /// Pinned facts in the scene's narrative whose `entity_id` overlaps
    /// participants, plus all narrative-wide pinned facts.
    pub pinned_facts: Vec<PinnedFact>,
    /// Writer-authored research notes attached to this scene.
    pub notes: Vec<ResearchNote>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantSource {
    pub entity_id: Uuid,
    pub attribution: SourceAttribution,
}

/// Build the scene research context.
pub fn build_scene_research_context(
    hypergraph: &Hypergraph,
    situation_id: &Uuid,
) -> Result<SceneResearchContext> {
    let situation = hypergraph.get_situation(situation_id)?;
    let narrative_id = situation.narrative_id.clone();

    // (a) Attributions on this scene.
    let scene_sources = hypergraph.get_attributions_for_target(situation_id)?;

    // (b) Attributions on participants. De-duplicate entity ids first — an
    //     actor with multiple roles (seq>0) appears multiple times in the
    //     participation list, and each extra pass does a redundant prefix scan.
    let participants = hypergraph.get_participants_for_situation(situation_id)?;
    let mut seen_entities: std::collections::HashSet<Uuid> =
        std::collections::HashSet::with_capacity(participants.len());
    let mut participant_sources = Vec::new();
    for p in &participants {
        if !seen_entities.insert(p.entity_id) {
            continue;
        }
        for a in hypergraph.get_attributions_for_target(&p.entity_id)? {
            participant_sources.push(ParticipantSource {
                entity_id: p.entity_id,
                attribution: a,
            });
        }
    }

    // (c) Contentions involving this scene.
    let contentions = hypergraph.get_contentions_for_situation(situation_id)?;

    // (d) Pinned facts: narrative-wide + entity-scoped for scene participants.
    let mut pinned_facts = Vec::new();
    if let Some(nid) = &narrative_id {
        let all = continuity::list_pinned_facts(hypergraph.store(), nid)?;
        let participant_ids: std::collections::HashSet<Uuid> =
            participants.iter().map(|p| p.entity_id).collect();
        for f in all {
            let relevant = match f.entity_id {
                None => true, // narrative-wide fact
                Some(eid) => participant_ids.contains(&eid),
            };
            if relevant {
                pinned_facts.push(f);
            }
        }
    }

    // (e) Research notes scoped to this scene.
    let notes = list_notes_for_situation(hypergraph.store(), situation_id)?;

    Ok(SceneResearchContext {
        situation_id: *situation_id,
        narrative_id,
        scene_sources,
        participant_sources,
        contentions,
        pinned_facts,
        notes,
    })
}

// ─── Research-note persistence ────────────────────────────────

/// KV key: `rn/{note_uuid_bytes}` — the authoritative record.
pub(crate) fn note_key(note_id: &Uuid) -> Vec<u8> {
    let mut k = b"rn/".to_vec();
    k.extend_from_slice(note_id.as_bytes());
    k
}

/// KV key: `rn/s/{situation_uuid_bytes}/{note_uuid_bytes}` — situation→note index.
pub(crate) fn situation_index_key(situation_id: &Uuid, note_id: &Uuid) -> Vec<u8> {
    let mut k = b"rn/s/".to_vec();
    k.extend_from_slice(situation_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(note_id.as_bytes());
    k
}

pub(crate) fn situation_index_prefix(situation_id: &Uuid) -> Vec<u8> {
    let mut k = b"rn/s/".to_vec();
    k.extend_from_slice(situation_id.as_bytes());
    k.push(b'/');
    k
}

/// KV key: `rn/n/{narrative_id}/{note_uuid_bytes}` — narrative→note index.
pub(crate) fn narrative_index_key(narrative_id: &str, note_id: &Uuid) -> Vec<u8> {
    let mut k = format!("rn/n/{}/", narrative_id).into_bytes();
    k.extend_from_slice(note_id.as_bytes());
    k
}

pub(crate) fn narrative_index_prefix(narrative_id: &str) -> Vec<u8> {
    format!("rn/n/{}/", narrative_id).into_bytes()
}

pub fn create_research_note(store: &dyn KVStore, mut note: ResearchNote) -> Result<ResearchNote> {
    if note.body.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "research note body cannot be empty".into(),
        ));
    }
    let now = Utc::now();
    if note.id.is_nil() {
        note.id = Uuid::now_v7();
    }
    note.created_at = now;
    note.updated_at = now;
    let bytes = serde_json::to_vec(&note)?;
    store.put(&note_key(&note.id), &bytes)?;
    store.put(&situation_index_key(&note.situation_id, &note.id), &[])?;
    store.put(&narrative_index_key(&note.narrative_id, &note.id), &[])?;
    Ok(note)
}

pub fn get_research_note(store: &dyn KVStore, note_id: &Uuid) -> Result<ResearchNote> {
    match store.get(&note_key(note_id))? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::QueryError(format!(
            "research note {note_id} not found"
        ))),
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct ResearchNotePatch {
    #[serde(default)]
    pub kind: Option<ResearchNoteKind>,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub author: Option<Option<String>>,
    #[serde(default)]
    pub source_id: Option<Option<Uuid>>,
}

pub fn update_research_note(
    store: &dyn KVStore,
    note_id: &Uuid,
    patch: ResearchNotePatch,
) -> Result<ResearchNote> {
    let mut note = get_research_note(store, note_id)?;
    if let Some(k) = patch.kind {
        note.kind = k;
    }
    if let Some(b) = patch.body {
        if b.trim().is_empty() {
            return Err(TensaError::InvalidQuery(
                "research note body cannot be empty".into(),
            ));
        }
        note.body = b;
    }
    if let Some(a) = patch.author {
        note.author = a;
    }
    if let Some(s) = patch.source_id {
        note.source_id = s;
    }
    note.updated_at = Utc::now();
    let bytes = serde_json::to_vec(&note)?;
    store.put(&note_key(note_id), &bytes)?;
    Ok(note)
}

pub fn delete_research_note(store: &dyn KVStore, note_id: &Uuid) -> Result<()> {
    let note = get_research_note(store, note_id)?;
    store.delete(&note_key(note_id))?;
    store.delete(&situation_index_key(&note.situation_id, note_id))?;
    store.delete(&narrative_index_key(&note.narrative_id, note_id))?;
    Ok(())
}

pub fn list_notes_for_situation(
    store: &dyn KVStore,
    situation_id: &Uuid,
) -> Result<Vec<ResearchNote>> {
    let prefix = situation_index_prefix(situation_id);
    let mut out = crate::store::scan_uuid_index(store, &prefix, |id| get_research_note(store, id))?;
    out.sort_by_key(|n| n.created_at);
    Ok(out)
}

pub fn list_notes_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<ResearchNote>> {
    let prefix = narrative_index_prefix(narrative_id);
    let mut out = crate::store::scan_uuid_index(store, &prefix, |id| get_research_note(store, id))?;
    out.sort_by_key(|n| n.created_at);
    Ok(out)
}

// ─── Chunk → note promotion ───────────────────────────────────

/// Minimal descriptor of a source chunk the writer wants to promote.
#[derive(Debug, Clone, Deserialize)]
pub struct PromoteChunkRequest {
    pub situation_id: Uuid,
    pub narrative_id: String,
    pub chunk_id: Uuid,
    /// Verbatim quote / excerpt extracted from the chunk.
    pub body: String,
    #[serde(default)]
    pub source_id: Option<Uuid>,
    #[serde(default)]
    pub kind: Option<ResearchNoteKind>,
    #[serde(default)]
    pub author: Option<String>,
}

/// Promote a ChunkStore chunk into a scene-scoped research note.
///
/// The caller has already selected the excerpt. We simply persist it as a
/// note and record the back-link to `source_chunk_id` so a later action
/// can navigate from the note back to the origin chunk.
pub fn promote_chunk_to_note(
    store: &dyn KVStore,
    req: PromoteChunkRequest,
) -> Result<ResearchNote> {
    let note = ResearchNote {
        id: Uuid::now_v7(),
        narrative_id: req.narrative_id,
        situation_id: req.situation_id,
        kind: req.kind.unwrap_or(ResearchNoteKind::Quote),
        body: req.body,
        source_chunk_id: Some(req.chunk_id),
        source_id: req.source_id,
        author: req.author,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    create_research_note(store, note)
}

// Unused import guard for `Arc` — available for future caller helpers that
// need to hand the hypergraph's store through a shared reference.
#[allow(dead_code)]
fn _unused_arc_hint(_: Arc<dyn KVStore>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Situation,
        TimeGranularity,
    };

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_scene(narrative: &str) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some("scene".into()),
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
            raw_content: vec![ContentBlock::text("body")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.into()),
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
        }
    }

    #[test]
    fn test_create_and_list_note() {
        let hg = setup();
        let store = hg.store();
        let scene = make_scene("n1");
        let scene_id = scene.id;
        hg.create_situation(scene).unwrap();

        let note = ResearchNote {
            id: Uuid::nil(),
            narrative_id: "n1".into(),
            situation_id: scene_id,
            kind: ResearchNoteKind::Quote,
            body: "A memorable quote.".into(),
            source_chunk_id: None,
            source_id: None,
            author: Some("writer".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let saved = create_research_note(store, note).unwrap();
        assert!(!saved.id.is_nil());

        let listed = list_notes_for_situation(store, &scene_id).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].body, "A memorable quote.");

        let by_narrative = list_notes_for_narrative(store, "n1").unwrap();
        assert_eq!(by_narrative.len(), 1);
    }

    #[test]
    fn test_create_rejects_empty_body() {
        let hg = setup();
        let scene = make_scene("n1");
        let note = ResearchNote {
            id: Uuid::nil(),
            narrative_id: "n1".into(),
            situation_id: scene.id,
            kind: ResearchNoteKind::Note,
            body: "   ".into(),
            source_chunk_id: None,
            source_id: None,
            author: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let err = create_research_note(hg.store(), note).unwrap_err();
        assert!(matches!(err, TensaError::InvalidQuery(_)));
    }

    #[test]
    fn test_update_and_delete_note() {
        let hg = setup();
        let scene = make_scene("n1");
        let scene_id = scene.id;
        hg.create_situation(scene).unwrap();

        let note = create_research_note(
            hg.store(),
            ResearchNote {
                id: Uuid::nil(),
                narrative_id: "n1".into(),
                situation_id: scene_id,
                kind: ResearchNoteKind::Note,
                body: "v1".into(),
                source_chunk_id: None,
                source_id: None,
                author: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();

        let updated = update_research_note(
            hg.store(),
            &note.id,
            ResearchNotePatch {
                body: Some("v2".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(updated.body, "v2");

        delete_research_note(hg.store(), &note.id).unwrap();
        let empty = list_notes_for_situation(hg.store(), &scene_id).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_promote_chunk_to_note_sets_back_link() {
        let hg = setup();
        let scene = make_scene("n1");
        let scene_id = scene.id;
        hg.create_situation(scene).unwrap();

        let chunk_id = Uuid::now_v7();
        let note = promote_chunk_to_note(
            hg.store(),
            PromoteChunkRequest {
                situation_id: scene_id,
                narrative_id: "n1".into(),
                chunk_id,
                body: "Important quote.".into(),
                source_id: None,
                kind: None,
                author: None,
            },
        )
        .unwrap();
        assert_eq!(note.source_chunk_id, Some(chunk_id));
        assert_eq!(note.kind, ResearchNoteKind::Quote);
    }

    #[test]
    fn test_build_scene_research_context_aggregates() {
        let hg = setup();
        let scene = make_scene("n1");
        let scene_id = scene.id;
        hg.create_situation(scene).unwrap();

        // Add a note; context should surface it.
        create_research_note(
            hg.store(),
            ResearchNote {
                id: Uuid::nil(),
                narrative_id: "n1".into(),
                situation_id: scene_id,
                kind: ResearchNoteKind::Note,
                body: "Research hint.".into(),
                source_chunk_id: None,
                source_id: None,
                author: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();

        let ctx = build_scene_research_context(&hg, &scene_id).unwrap();
        assert_eq!(ctx.situation_id, scene_id);
        assert_eq!(ctx.notes.len(), 1);
        assert!(ctx.scene_sources.is_empty());
        assert!(ctx.contentions.is_empty());
    }
}
