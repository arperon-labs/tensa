//! Sprint W12 — structured inline annotations: comments, footnotes, citations.
//!
//! Annotations live on a byte-span of a situation's concatenated prose
//! content. Citations carry `source_id` + `chunk_id` so they survive into
//! compile output (W13). Span maintenance: on a prose edit that changes
//! byte offsets, annotations whose span falls outside the new buffer are
//! marked `detached=true` rather than deleted, so the writer can re-anchor
//! them if desired.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub id: Uuid,
    pub situation_id: Uuid,
    pub kind: AnnotationKind,
    /// Byte range in the situation's concatenated prose (Text + Dialogue blocks,
    /// joined with "\n\n"). Detached annotations keep their old span for reference.
    pub span: (usize, usize),
    pub body: String,
    /// Citation payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// True if the prose was edited such that this annotation's span no longer
    /// maps to valid text. The UI surfaces detached annotations with a warning
    /// so the writer can re-anchor or delete them.
    #[serde(default)]
    pub detached: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationKind {
    /// Reviewer / author margin comment (not rendered in compile).
    Comment,
    /// Footnote rendered at the bottom of the page / end of chapter in compile.
    Footnote,
    /// Citation — source attribution; rendered as a numeric ref + bibliography entry.
    Citation,
}

// ─── KV persistence ──────────────────────────────────────────

pub(crate) fn annotation_key(id: &Uuid) -> Vec<u8> {
    let mut k = b"ann/".to_vec();
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn situation_index_key(situation_id: &Uuid, id: &Uuid) -> Vec<u8> {
    let mut k = b"ann/s/".to_vec();
    k.extend_from_slice(situation_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn situation_index_prefix(situation_id: &Uuid) -> Vec<u8> {
    let mut k = b"ann/s/".to_vec();
    k.extend_from_slice(situation_id.as_bytes());
    k.push(b'/');
    k
}

pub fn create_annotation(store: &dyn KVStore, mut ann: Annotation) -> Result<Annotation> {
    if ann.body.trim().is_empty() && ann.kind != AnnotationKind::Citation {
        return Err(TensaError::InvalidQuery(
            "annotation body cannot be empty unless kind=Citation".into(),
        ));
    }
    if ann.span.0 > ann.span.1 {
        return Err(TensaError::InvalidQuery(
            "annotation span.start must be <= span.end".into(),
        ));
    }
    let now = Utc::now();
    if ann.id.is_nil() {
        ann.id = Uuid::now_v7();
    }
    ann.created_at = now;
    ann.updated_at = now;
    let bytes = serde_json::to_vec(&ann)?;
    store.put(&annotation_key(&ann.id), &bytes)?;
    store.put(&situation_index_key(&ann.situation_id, &ann.id), &[])?;
    Ok(ann)
}

pub fn get_annotation(store: &dyn KVStore, id: &Uuid) -> Result<Annotation> {
    match store.get(&annotation_key(id))? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::QueryError(format!("annotation {id} not found"))),
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct AnnotationPatch {
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub span: Option<(usize, usize)>,
    #[serde(default)]
    pub source_id: Option<Option<Uuid>>,
    #[serde(default)]
    pub chunk_id: Option<Option<Uuid>>,
    #[serde(default)]
    pub detached: Option<bool>,
}

pub fn update_annotation(
    store: &dyn KVStore,
    id: &Uuid,
    patch: AnnotationPatch,
) -> Result<Annotation> {
    let mut ann = get_annotation(store, id)?;
    if let Some(body) = patch.body {
        ann.body = body;
    }
    if let Some(span) = patch.span {
        if span.0 > span.1 {
            return Err(TensaError::InvalidQuery(
                "annotation span.start must be <= span.end".into(),
            ));
        }
        ann.span = span;
        ann.detached = false;
    }
    if let Some(s) = patch.source_id {
        ann.source_id = s;
    }
    if let Some(c) = patch.chunk_id {
        ann.chunk_id = c;
    }
    if let Some(d) = patch.detached {
        ann.detached = d;
    }
    ann.updated_at = Utc::now();
    let bytes = serde_json::to_vec(&ann)?;
    store.put(&annotation_key(id), &bytes)?;
    Ok(ann)
}

pub fn delete_annotation(store: &dyn KVStore, id: &Uuid) -> Result<()> {
    let ann = get_annotation(store, id)?;
    store.delete(&annotation_key(id))?;
    store.delete(&situation_index_key(&ann.situation_id, id))?;
    Ok(())
}

pub fn list_annotations_for_situation(
    store: &dyn KVStore,
    situation_id: &Uuid,
) -> Result<Vec<Annotation>> {
    let prefix = situation_index_prefix(situation_id);
    let mut out = crate::store::scan_uuid_index(store, &prefix, |id| get_annotation(store, id))?;
    out.sort_by_key(|a| a.span.0);
    Ok(out)
}

/// Batch-load all annotations for a set of scenes in a single pass.
///
/// Walks the global `ann/` prefix once, deserialising every annotation record
/// and bucketing by `situation_id`. Scenes not in `scene_ids` are dropped.
/// Used by the compile pipeline to avoid the N+1 `list_annotations_for_situation`
/// pattern when rendering many scenes.
pub fn list_annotations_for_scenes(
    store: &dyn KVStore,
    scene_ids: &std::collections::HashSet<Uuid>,
) -> Result<std::collections::HashMap<Uuid, Vec<Annotation>>> {
    let mut out: std::collections::HashMap<Uuid, Vec<Annotation>> =
        std::collections::HashMap::with_capacity(scene_ids.len());
    let entries = store.prefix_scan(b"ann/")?;
    for (_key, value) in entries {
        // The `ann/s/…` index entries have empty values; skip them cheaply.
        if value.is_empty() {
            continue;
        }
        let ann: Annotation = match serde_json::from_slice(&value) {
            Ok(a) => a,
            Err(_) => continue,
        };
        if scene_ids.contains(&ann.situation_id) {
            out.entry(ann.situation_id).or_default().push(ann);
        }
    }
    for v in out.values_mut() {
        v.sort_by_key(|a| a.span.0);
    }
    Ok(out)
}

/// Detach-or-shift all annotations on a situation after a prose edit.
///
/// Invariant: given the *old* and *new* prose strings, any annotation whose
/// original span substring is still present in the new text at a single
/// unambiguous location is re-anchored to that location. Otherwise it's
/// flagged `detached=true` and left in place — the UI surfaces a warning
/// so the writer can decide.
pub fn reconcile_spans_after_edit(
    store: &dyn KVStore,
    situation_id: &Uuid,
    old_prose: &str,
    new_prose: &str,
) -> Result<ReconciliationReport> {
    let annotations = list_annotations_for_situation(store, situation_id)?;
    let mut moved = 0u32;
    let mut detached = 0u32;
    let mut unchanged = 0u32;
    for mut ann in annotations {
        if ann.detached {
            continue;
        }
        let (s, e) = ann.span;
        let e = e.min(old_prose.len());
        let s = s.min(e);
        if s == e {
            unchanged += 1;
            continue;
        }
        let old_substr = &old_prose[s..e];
        let matches: Vec<_> = find_all(new_prose, old_substr).collect();
        match matches.len() {
            1 => {
                let start = matches[0];
                let new_span = (start, start + old_substr.len());
                if new_span != ann.span {
                    ann.span = new_span;
                    ann.updated_at = Utc::now();
                    let bytes = serde_json::to_vec(&ann)?;
                    store.put(&annotation_key(&ann.id), &bytes)?;
                    moved += 1;
                } else {
                    unchanged += 1;
                }
            }
            _ => {
                ann.detached = true;
                ann.updated_at = Utc::now();
                let bytes = serde_json::to_vec(&ann)?;
                store.put(&annotation_key(&ann.id), &bytes)?;
                detached += 1;
            }
        }
    }
    Ok(ReconciliationReport {
        moved,
        detached,
        unchanged,
    })
}

fn find_all<'a>(haystack: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= haystack.len() || needle.is_empty() {
            return None;
        }
        match haystack[start..].find(needle) {
            Some(rel) => {
                let abs = start + rel;
                start = abs + needle.len().max(1);
                Some(abs)
            }
            None => None,
        }
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationReport {
    pub moved: u32,
    pub detached: u32,
    pub unchanged: u32,
}

/// Sprint W12 canonical renderer — produces inline markers plus concatenated
/// footnote and bibliography bodies from a list of annotations. Comments are
/// always skipped (they're editor-only). Footnotes can be suppressed with
/// `include_footnotes=false` so the compile pipeline can honour its
/// `FootnoteStyle::None` setting.
///
/// Returns `(markers, footnote_body, bibliography_body)`. Callers that want
/// a full Markdown block with section headers can use
/// [`render_annotations_markdown`] below.
pub fn render_annotation_parts(
    annotations: &[Annotation],
    include_footnotes: bool,
) -> (Vec<(usize, String)>, String, String) {
    let mut markers: Vec<(usize, String)> = Vec::new();
    let mut footnote_body = String::new();
    let mut bib_body = String::new();
    let mut footnote_counter = 0u32;
    let mut citation_counter = 0u32;
    for ann in annotations {
        if ann.detached {
            continue;
        }
        match ann.kind {
            AnnotationKind::Footnote if include_footnotes => {
                footnote_counter += 1;
                let marker = format!("[^{}]", footnote_counter);
                markers.push((ann.span.0, marker.clone()));
                footnote_body.push_str(&format!("{}: {}\n", marker, ann.body));
            }
            AnnotationKind::Citation => {
                citation_counter += 1;
                let marker = format!("[^ref:{}]", citation_counter);
                markers.push((ann.span.0, marker.clone()));
                let src = ann
                    .source_id
                    .map(|id| format!("source={}", id))
                    .unwrap_or_else(|| "source=unknown".into());
                let chunk = ann
                    .chunk_id
                    .map(|id| format!(", chunk={}", id))
                    .unwrap_or_default();
                bib_body.push_str(&format!("{}: {} — {}{}\n", marker, ann.body, src, chunk));
            }
            // Footnotes with include_footnotes=false: skip.
            // Comments: editor-only, never shipped.
            _ => {}
        }
    }
    (markers, footnote_body, bib_body)
}

/// Render annotations into a complete Markdown section (markers + section
/// headers + bodies). Used by the direct-export path; compile profiles call
/// [`render_annotation_parts`] and splice the fragments into their own layout.
pub fn render_annotations_markdown(annotations: &[Annotation]) -> (Vec<(usize, String)>, String) {
    let (markers, footnote_body, bib_body) = render_annotation_parts(annotations, true);
    let mut block = String::new();
    if !footnote_body.is_empty() {
        block.push_str("\n\n---\n\n### Footnotes\n\n");
        block.push_str(&footnote_body);
    }
    if !bib_body.is_empty() {
        block.push_str("\n\n### References\n\n");
        block.push_str(&bib_body);
    }
    (markers, block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn setup() -> std::sync::Arc<MemoryStore> {
        std::sync::Arc::new(MemoryStore::new())
    }

    fn make_ann(sid: Uuid, kind: AnnotationKind, span: (usize, usize), body: &str) -> Annotation {
        Annotation {
            id: Uuid::nil(),
            situation_id: sid,
            kind,
            span,
            body: body.into(),
            source_id: None,
            chunk_id: None,
            author: None,
            detached: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_crud_and_list_by_situation() {
        let store = setup();
        let sid = Uuid::now_v7();
        let a = create_annotation(
            &*store,
            make_ann(sid, AnnotationKind::Comment, (0, 10), "nice"),
        )
        .unwrap();
        let _b = create_annotation(
            &*store,
            make_ann(sid, AnnotationKind::Footnote, (20, 30), "source note"),
        )
        .unwrap();
        let list = list_annotations_for_situation(&*store, &sid).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].span.0, 0);
        assert_eq!(list[1].span.0, 20);
        delete_annotation(&*store, &a.id).unwrap();
        let after = list_annotations_for_situation(&*store, &sid).unwrap();
        assert_eq!(after.len(), 1);
    }

    #[test]
    fn test_reconcile_reanchors_single_match() {
        let store = setup();
        let sid = Uuid::now_v7();
        let old = "Hello beautiful world.";
        let new = "Greetings, beautiful world.";
        let ann = create_annotation(
            &*store,
            make_ann(sid, AnnotationKind::Footnote, (6, 15), "on beautiful"),
        )
        .unwrap();
        let report = reconcile_spans_after_edit(&*store, &sid, old, new).unwrap();
        assert_eq!(report.moved, 1);
        let fresh = get_annotation(&*store, &ann.id).unwrap();
        assert!(!fresh.detached);
        assert_eq!(&new[fresh.span.0..fresh.span.1], "beautiful");
    }

    #[test]
    fn test_reconcile_detaches_on_ambiguous_match() {
        let store = setup();
        let sid = Uuid::now_v7();
        let old = "Alpha bravo.";
        let new = "Alpha bravo. Alpha bravo.";
        let ann = create_annotation(
            &*store,
            make_ann(sid, AnnotationKind::Comment, (0, 5), "first alpha"),
        )
        .unwrap();
        let report = reconcile_spans_after_edit(&*store, &sid, old, new).unwrap();
        assert_eq!(report.detached, 1);
        let fresh = get_annotation(&*store, &ann.id).unwrap();
        assert!(fresh.detached);
    }

    #[test]
    fn test_render_footnote_and_citation_produce_block() {
        let sid = Uuid::now_v7();
        let anns = vec![
            Annotation {
                id: Uuid::now_v7(),
                situation_id: sid,
                kind: AnnotationKind::Footnote,
                span: (5, 10),
                body: "See appendix A.".into(),
                source_id: None,
                chunk_id: None,
                author: None,
                detached: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            Annotation {
                id: Uuid::now_v7(),
                situation_id: sid,
                kind: AnnotationKind::Citation,
                span: (12, 18),
                body: "Author, Title (1999)".into(),
                source_id: Some(Uuid::now_v7()),
                chunk_id: None,
                author: None,
                detached: false,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        ];
        let (markers, block) = render_annotations_markdown(&anns);
        assert_eq!(markers.len(), 2);
        assert!(block.contains("Footnotes"));
        assert!(block.contains("References"));
    }

    #[test]
    fn test_comments_never_ship() {
        let sid = Uuid::now_v7();
        let anns = vec![Annotation {
            id: Uuid::now_v7(),
            situation_id: sid,
            kind: AnnotationKind::Comment,
            span: (0, 1),
            body: "editor margin note".into(),
            source_id: None,
            chunk_id: None,
            author: None,
            detached: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }];
        let (markers, block) = render_annotations_markdown(&anns);
        assert!(markers.is_empty());
        assert!(block.is_empty());
    }
}
