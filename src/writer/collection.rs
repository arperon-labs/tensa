//! Sprint W15 — Collections (saved searches).
//!
//! A `Collection` is a named, persisted filter that resolves to a set of
//! situation ids at query time. It shows up in the Studio Binder as a
//! virtual folder alongside real hierarchy nodes — writer-tool parity.
//!
//! The filter is structured (not a free-form TensaQL string) so we can
//! efficiently re-resolve as situations change. Supported predicates:
//! labels IN, status IN, keywords ANY, min/max manuscript_order, min/max
//! word count, free-text substring match against synopsis + name.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;
use crate::writer::scene::word_count;
use crate::Hypergraph;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: Uuid,
    pub narrative_id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub query: CollectionQuery,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CollectionQuery {
    /// Match if label is in this set (empty = any label).
    #[serde(default)]
    pub labels: Vec<String>,
    /// Match if status is in this set (empty = any status).
    #[serde(default)]
    pub statuses: Vec<String>,
    /// Match if scene's keywords intersect this set (empty = any).
    #[serde(default)]
    pub keywords_any: Vec<String>,
    /// Free-text case-insensitive substring match against (name + synopsis + description).
    #[serde(default)]
    pub text: Option<String>,
    /// Inclusive lower bound on `manuscript_order` (None = unbounded).
    #[serde(default)]
    pub min_order: Option<u32>,
    /// Inclusive upper bound on `manuscript_order` (None = unbounded).
    #[serde(default)]
    pub max_order: Option<u32>,
    /// Inclusive lower bound on word_count.
    #[serde(default)]
    pub min_words: Option<usize>,
    /// Inclusive upper bound on word_count.
    #[serde(default)]
    pub max_words: Option<usize>,
}

// ─── KV persistence ──────────────────────────────────────────

pub(crate) fn collection_key(id: &Uuid) -> Vec<u8> {
    let mut k = b"col/".to_vec();
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn narrative_index_key(narrative_id: &str, id: &Uuid) -> Vec<u8> {
    let mut k = format!("col/n/{}/", narrative_id).into_bytes();
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn narrative_index_prefix(narrative_id: &str) -> Vec<u8> {
    format!("col/n/{}/", narrative_id).into_bytes()
}

pub fn create_collection(store: &dyn KVStore, mut c: Collection) -> Result<Collection> {
    if c.name.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "collection name cannot be empty".into(),
        ));
    }
    if c.narrative_id.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "collection narrative_id is required".into(),
        ));
    }
    let now = Utc::now();
    if c.id.is_nil() {
        c.id = Uuid::now_v7();
    }
    c.created_at = now;
    c.updated_at = now;
    let bytes = serde_json::to_vec(&c)?;
    store.put(&collection_key(&c.id), &bytes)?;
    store.put(&narrative_index_key(&c.narrative_id, &c.id), &[])?;
    Ok(c)
}

pub fn get_collection(store: &dyn KVStore, id: &Uuid) -> Result<Collection> {
    match store.get(&collection_key(id))? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::QueryError(format!("collection {id} not found"))),
    }
}

pub fn list_collections_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<Collection>> {
    let prefix = narrative_index_prefix(narrative_id);
    let mut out = crate::store::scan_uuid_index(store, &prefix, |id| get_collection(store, id))?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct CollectionPatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub query: Option<CollectionQuery>,
}

pub fn update_collection(
    store: &dyn KVStore,
    id: &Uuid,
    patch: CollectionPatch,
) -> Result<Collection> {
    let mut c = get_collection(store, id)?;
    if let Some(n) = patch.name {
        if n.trim().is_empty() {
            return Err(TensaError::InvalidQuery(
                "collection name cannot be empty".into(),
            ));
        }
        c.name = n;
    }
    if let Some(d) = patch.description {
        c.description = d;
    }
    if let Some(q) = patch.query {
        c.query = q;
    }
    c.updated_at = Utc::now();
    let bytes = serde_json::to_vec(&c)?;
    store.put(&collection_key(id), &bytes)?;
    Ok(c)
}

pub fn delete_collection(store: &dyn KVStore, id: &Uuid) -> Result<()> {
    let c = get_collection(store, id)?;
    store.delete(&collection_key(id))?;
    store.delete(&narrative_index_key(&c.narrative_id, id))?;
    Ok(())
}

// ─── Resolution ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionResolution {
    pub collection_id: Uuid,
    pub narrative_id: String,
    pub matches: Vec<Uuid>,
    pub match_count: usize,
}

/// Resolve a saved query to the matching situation ids in the narrative.
pub fn resolve_collection(
    hypergraph: &Hypergraph,
    collection: &Collection,
) -> Result<CollectionResolution> {
    let candidates = hypergraph.list_situations_by_narrative(&collection.narrative_id)?;
    let mut matches = Vec::new();
    let text = collection
        .query
        .text
        .as_deref()
        .map(|t| t.to_lowercase())
        .filter(|t| !t.is_empty());
    let kw_any: std::collections::HashSet<String> = collection
        .query
        .keywords_any
        .iter()
        .map(|s| s.to_lowercase())
        .collect();
    for sit in &candidates {
        if !collection.query.labels.is_empty() {
            let Some(l) = &sit.label else {
                continue;
            };
            if !collection.query.labels.contains(l) {
                continue;
            }
        }
        if !collection.query.statuses.is_empty() {
            let Some(st) = &sit.status else {
                continue;
            };
            if !collection.query.statuses.contains(st) {
                continue;
            }
        }
        if !kw_any.is_empty() {
            let sit_kw_lower: std::collections::HashSet<String> =
                sit.keywords.iter().map(|k| k.to_lowercase()).collect();
            if sit_kw_lower.is_disjoint(&kw_any) {
                continue;
            }
        }
        if let Some(min) = collection.query.min_order {
            if sit.manuscript_order.unwrap_or(u32::MAX) < min {
                continue;
            }
        }
        if let Some(max) = collection.query.max_order {
            if sit.manuscript_order.unwrap_or(0) > max {
                continue;
            }
        }
        if collection.query.min_words.is_some() || collection.query.max_words.is_some() {
            let wc = word_count(sit);
            if let Some(min) = collection.query.min_words {
                if wc < min {
                    continue;
                }
            }
            if let Some(max) = collection.query.max_words {
                if wc > max {
                    continue;
                }
            }
        }
        if let Some(q) = &text {
            let hay = [
                sit.name.as_deref().unwrap_or(""),
                sit.synopsis.as_deref().unwrap_or(""),
                sit.description.as_deref().unwrap_or(""),
            ]
            .join(" ")
            .to_lowercase();
            if !hay.contains(q) {
                continue;
            }
        }
        matches.push(sit.id);
    }
    let match_count = matches.len();
    Ok(CollectionResolution {
        collection_id: collection.id,
        narrative_id: collection.narrative_id.clone(),
        matches,
        match_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Situation,
        TimeGranularity,
    };
    use std::sync::Arc;

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_scene(
        narrative: &str,
        name: &str,
        label: Option<&str>,
        status: Option<&str>,
        keywords: Vec<&str>,
        order: Option<u32>,
        word_count_target: usize,
    ) -> Situation {
        let words: Vec<String> = (0..word_count_target).map(|i| format!("word{i}")).collect();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(name.into()),
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
            raw_content: vec![ContentBlock::text(&words.join(" "))],
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
            manuscript_order: order,
            parent_situation_id: None,
            label: label.map(String::from),
            status: status.map(String::from),
            keywords: keywords.into_iter().map(String::from).collect(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn default_collection(narrative: &str) -> Collection {
        Collection {
            id: Uuid::nil(),
            narrative_id: narrative.into(),
            name: "Needs revision".into(),
            description: None,
            query: CollectionQuery::default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_crud_collection() {
        let hg = setup();
        let c = create_collection(hg.store(), default_collection("n1")).unwrap();
        let list = list_collections_for_narrative(hg.store(), "n1").unwrap();
        assert_eq!(list.len(), 1);
        delete_collection(hg.store(), &c.id).unwrap();
        let list = list_collections_for_narrative(hg.store(), "n1").unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_empty_query_matches_all() {
        let hg = setup();
        hg.create_situation(make_scene("n1", "A", None, None, vec![], Some(1000), 5))
            .unwrap();
        hg.create_situation(make_scene("n1", "B", None, None, vec![], Some(2000), 5))
            .unwrap();
        let c = create_collection(hg.store(), default_collection("n1")).unwrap();
        let res = resolve_collection(&hg, &c).unwrap();
        assert_eq!(res.match_count, 2);
    }

    #[test]
    fn test_label_and_status_filter() {
        let hg = setup();
        hg.create_situation(make_scene(
            "n1",
            "A",
            Some("Draft"),
            Some("first-draft"),
            vec![],
            None,
            5,
        ))
        .unwrap();
        hg.create_situation(make_scene(
            "n1",
            "B",
            Some("Final"),
            Some("final"),
            vec![],
            None,
            5,
        ))
        .unwrap();
        let mut c = default_collection("n1");
        c.query.labels = vec!["Final".into()];
        let saved = create_collection(hg.store(), c).unwrap();
        let res = resolve_collection(&hg, &saved).unwrap();
        assert_eq!(res.match_count, 1);
    }

    #[test]
    fn test_keyword_any_and_text_substring() {
        let hg = setup();
        hg.create_situation(make_scene(
            "n1",
            "the dark tower",
            None,
            None,
            vec!["magic", "villain"],
            None,
            5,
        ))
        .unwrap();
        hg.create_situation(make_scene(
            "n1",
            "harbour scene",
            None,
            None,
            vec!["ship"],
            None,
            5,
        ))
        .unwrap();
        let mut c = default_collection("n1");
        c.query.keywords_any = vec!["villain".into()];
        c.query.text = Some("tower".into());
        let saved = create_collection(hg.store(), c).unwrap();
        let res = resolve_collection(&hg, &saved).unwrap();
        assert_eq!(res.match_count, 1);
    }

    #[test]
    fn test_word_count_bounds() {
        let hg = setup();
        hg.create_situation(make_scene("n1", "short", None, None, vec![], None, 3))
            .unwrap();
        hg.create_situation(make_scene("n1", "medium", None, None, vec![], None, 10))
            .unwrap();
        hg.create_situation(make_scene("n1", "long", None, None, vec![], None, 50))
            .unwrap();
        let mut c = default_collection("n1");
        c.query.min_words = Some(5);
        c.query.max_words = Some(20);
        let saved = create_collection(hg.store(), c).unwrap();
        let res = resolve_collection(&hg, &saved).unwrap();
        assert_eq!(res.match_count, 1);
    }

    #[test]
    fn test_order_bounds() {
        let hg = setup();
        hg.create_situation(make_scene("n1", "a", None, None, vec![], Some(500), 5))
            .unwrap();
        hg.create_situation(make_scene("n1", "b", None, None, vec![], Some(1500), 5))
            .unwrap();
        hg.create_situation(make_scene("n1", "c", None, None, vec![], Some(3000), 5))
            .unwrap();
        let mut c = default_collection("n1");
        c.query.min_order = Some(1000);
        c.query.max_order = Some(2500);
        let saved = create_collection(hg.store(), c).unwrap();
        let res = resolve_collection(&hg, &saved).unwrap();
        assert_eq!(res.match_count, 1);
    }
}
