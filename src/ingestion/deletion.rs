//! Smart document deletion with cascade.
//!
//! Tracks which entities and situations were created from each source
//! (via the `si/` KV prefix) and supports cascade deletion: removing a
//! source deletes all entities, situations, and participations it produced.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::ingestion::doc_status::DocStatusTracker;
use crate::store::KVStore;

/// KV prefix for source-to-item index.
const SOURCE_INDEX_PREFIX: &str = "si/";

/// Report of what was deleted during a cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionReport {
    /// The source that was deleted.
    pub source_id: String,
    /// Number of entities deleted.
    pub entities_deleted: usize,
    /// Number of situations deleted.
    pub situations_deleted: usize,
    /// Number of participations removed.
    pub participations_removed: usize,
}

/// Source index for tracking which entities/situations came from which source.
///
/// Uses the `si/{source_id}/{uuid}` KV prefix with value `e` for entity
/// or `s` for situation.
pub struct SourceIndex {
    store: Arc<dyn KVStore>,
}

impl SourceIndex {
    /// Create a new source index backed by the given KV store.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Record that an entity was created from a source.
    pub fn record_entity(&self, source_id: &str, entity_id: Uuid) -> Result<()> {
        let key = format!("{}{}/{}", SOURCE_INDEX_PREFIX, source_id, entity_id);
        self.store.put(key.as_bytes(), b"e")
    }

    /// Record that a situation was created from a source.
    pub fn record_situation(&self, source_id: &str, situation_id: Uuid) -> Result<()> {
        let key = format!("{}{}/{}", SOURCE_INDEX_PREFIX, source_id, situation_id);
        self.store.put(key.as_bytes(), b"s")
    }

    /// List all entity and situation IDs associated with a source.
    ///
    /// Returns `(entity_ids, situation_ids)`.
    pub fn items_from_source(&self, source_id: &str) -> Result<(Vec<Uuid>, Vec<Uuid>)> {
        let prefix = format!("{}{}/", SOURCE_INDEX_PREFIX, source_id);
        let entries = self.store.prefix_scan(prefix.as_bytes())?;
        let mut entities = Vec::new();
        let mut situations = Vec::new();

        for (key, value) in &entries {
            if let Ok(key_str) = std::str::from_utf8(key) {
                if let Some(uuid_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                        if value == b"e" {
                            entities.push(uuid);
                        } else if value == b"s" {
                            situations.push(uuid);
                        }
                    }
                }
            }
        }

        Ok((entities, situations))
    }

    /// Remove all index entries for a source. Returns the number of entries removed.
    pub fn remove_source(&self, source_id: &str) -> Result<usize> {
        let prefix = format!("{}{}/", SOURCE_INDEX_PREFIX, source_id);
        let entries = self.store.prefix_scan(prefix.as_bytes())?;
        let count = entries.len();
        for (key, _) in &entries {
            self.store.delete(key)?;
        }
        Ok(count)
    }

    /// Return all items from a source AND delete the index entries in a single scan.
    ///
    /// Combines `items_from_source` and `remove_source` to avoid two prefix scans.
    /// Returns `(entity_ids, situation_ids, entries_removed)`.
    pub fn drain_source(&self, source_id: &str) -> Result<(Vec<Uuid>, Vec<Uuid>, usize)> {
        let prefix = format!("{}{}/", SOURCE_INDEX_PREFIX, source_id);
        let entries = self.store.prefix_scan(prefix.as_bytes())?;
        let mut entities = Vec::new();
        let mut situations = Vec::new();
        let count = entries.len();

        for (key, value) in &entries {
            if let Ok(key_str) = std::str::from_utf8(key) {
                if let Some(uuid_str) = key_str.strip_prefix(&prefix) {
                    if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                        if value == b"e" {
                            entities.push(uuid);
                        } else if value == b"s" {
                            situations.push(uuid);
                        }
                    }
                }
            }
            self.store.delete(key)?;
        }

        Ok((entities, situations, count))
    }

    /// List all tracked source IDs.
    pub fn list_sources(&self) -> Result<Vec<String>> {
        let entries = self.store.prefix_scan(SOURCE_INDEX_PREFIX.as_bytes())?;
        let mut sources = std::collections::BTreeSet::new();
        for (key, _) in &entries {
            if let Ok(key_str) = std::str::from_utf8(key) {
                if let Some(rest) = key_str.strip_prefix(SOURCE_INDEX_PREFIX) {
                    if let Some(slash_pos) = rest.find('/') {
                        sources.insert(rest[..slash_pos].to_string());
                    }
                }
            }
        }
        Ok(sources.into_iter().collect())
    }
}

/// Cascade-delete all entities, situations, and participations from a source.
///
/// 1. Looks up all items in the source index
/// 2. Removes participations for each situation
/// 3. Deletes situations
/// 4. Deletes entities
/// 5. Cleans up the source index
/// 6. Optionally removes the document status record
pub fn cascade_delete_source(
    source_id: &str,
    hypergraph: &Hypergraph,
    source_index: &SourceIndex,
    doc_tracker: Option<&DocStatusTracker>,
) -> Result<DeletionReport> {
    // Drain returns items AND deletes index entries in a single prefix scan
    let (entity_ids, situation_ids, _) = source_index.drain_source(source_id)?;

    let mut entities_deleted = 0;
    let mut situations_deleted = 0;
    let mut participations_removed = 0;

    // Delete situations first (they reference entities via participation)
    for sid in &situation_ids {
        // Remove all participations for this situation
        if let Ok(participants) = hypergraph.get_participants_for_situation(sid) {
            for p in &participants {
                if hypergraph
                    .remove_participant(&p.entity_id, sid, None)
                    .is_ok()
                {
                    participations_removed += 1;
                }
            }
        }
        if hypergraph.delete_situation(sid).is_ok() {
            situations_deleted += 1;
        }
    }

    // Delete entities
    for eid in &entity_ids {
        // Also remove any participations this entity has
        if let Ok(entity_situations) = hypergraph.get_situations_for_entity(eid) {
            for p in &entity_situations {
                if hypergraph
                    .remove_participant(eid, &p.situation_id, None)
                    .is_ok()
                {
                    participations_removed += 1;
                }
            }
        }
        if hypergraph.delete_entity(eid).is_ok() {
            entities_deleted += 1;
        }
    }

    // Try to remove matching doc status entries
    if let Some(tracker) = doc_tracker {
        if let Ok(all_statuses) = tracker.list_all() {
            for status in &all_statuses {
                if status.source_id == source_id {
                    let _ = tracker.remove(&status.source_hash);
                }
            }
        }
    }

    Ok(DeletionReport {
        source_id: source_id.to_string(),
        entities_deleted,
        situations_deleted,
        participations_removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;

    fn make_entity(name: &str) -> Entity {
        Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.85,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("test-narrative".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn make_situation(text: &str) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
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
            raw_content: vec![ContentBlock::text(text)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.75,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: Some("test-narrative".into()),
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
    fn test_source_index_record_and_list() {
        let store = Arc::new(MemoryStore::new());
        let idx = SourceIndex::new(store);

        let e1 = Uuid::now_v7();
        let e2 = Uuid::now_v7();
        let s1 = Uuid::now_v7();

        idx.record_entity("doc-1", e1).unwrap();
        idx.record_entity("doc-1", e2).unwrap();
        idx.record_situation("doc-1", s1).unwrap();

        let (entities, situations) = idx.items_from_source("doc-1").unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(situations.len(), 1);
        assert!(entities.contains(&e1));
        assert!(entities.contains(&e2));
        assert!(situations.contains(&s1));
    }

    #[test]
    fn test_source_index_remove() {
        let store = Arc::new(MemoryStore::new());
        let idx = SourceIndex::new(store);

        idx.record_entity("doc-2", Uuid::now_v7()).unwrap();
        idx.record_situation("doc-2", Uuid::now_v7()).unwrap();

        let removed = idx.remove_source("doc-2").unwrap();
        assert_eq!(removed, 2);

        let (entities, situations) = idx.items_from_source("doc-2").unwrap();
        assert!(entities.is_empty());
        assert!(situations.is_empty());
    }

    #[test]
    fn test_cascade_delete() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let source_idx = SourceIndex::new(store);

        let entity = make_entity("Alice");
        let eid = entity.id;
        hg.create_entity(entity).unwrap();
        source_idx.record_entity("source-a", eid).unwrap();

        let situation = make_situation("Alice walked in.");
        let sid = situation.id;
        hg.create_situation(situation).unwrap();
        source_idx.record_situation("source-a", sid).unwrap();

        // Verify they exist
        assert!(hg.get_entity(&eid).is_ok());
        assert!(hg.get_situation(&sid).is_ok());

        let report = cascade_delete_source("source-a", &hg, &source_idx, None).unwrap();
        assert_eq!(report.entities_deleted, 1);
        assert_eq!(report.situations_deleted, 1);

        // Verify they are gone (soft-deleted)
        assert!(hg.get_entity(&eid).is_err());
        assert!(hg.get_situation(&sid).is_err());

        // Source index cleaned up
        let (e, s) = source_idx.items_from_source("source-a").unwrap();
        assert!(e.is_empty());
        assert!(s.is_empty());
    }

    #[test]
    fn test_cascade_delete_unknown_source() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let source_idx = SourceIndex::new(store);

        let report = cascade_delete_source("nonexistent", &hg, &source_idx, None).unwrap();
        assert_eq!(report.entities_deleted, 0);
        assert_eq!(report.situations_deleted, 0);
        assert_eq!(report.participations_removed, 0);
    }

    #[test]
    fn test_cascade_delete_with_participations() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let source_idx = SourceIndex::new(store);

        let entity = make_entity("Bob");
        let eid = entity.id;
        hg.create_entity(entity).unwrap();
        source_idx.record_entity("source-b", eid).unwrap();

        let situation = make_situation("Bob entered the room.");
        let sid = situation.id;
        hg.create_situation(situation).unwrap();
        source_idx.record_situation("source-b", sid).unwrap();

        // Add participation
        hg.add_participant(Participation {
            entity_id: eid,
            situation_id: sid,
            role: Role::Protagonist,
            info_set: None,
            action: Some("entered".into()),
            payoff: None,
            seq: 0,
        })
        .unwrap();

        // Verify participation exists
        let participants = hg.get_participants_for_situation(&sid).unwrap();
        assert_eq!(participants.len(), 1);

        let report = cascade_delete_source("source-b", &hg, &source_idx, None).unwrap();
        assert_eq!(report.entities_deleted, 1);
        assert_eq!(report.situations_deleted, 1);
        assert!(report.participations_removed >= 1);
    }

    #[test]
    fn test_source_index_list_sources() {
        let store = Arc::new(MemoryStore::new());
        let idx = SourceIndex::new(store);

        idx.record_entity("alpha", Uuid::now_v7()).unwrap();
        idx.record_entity("beta", Uuid::now_v7()).unwrap();
        idx.record_situation("alpha", Uuid::now_v7()).unwrap();

        let sources = idx.list_sources().unwrap();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"alpha".to_string()));
        assert!(sources.contains(&"beta".to_string()));
    }

    #[test]
    fn test_cascade_delete_with_doc_tracker() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let source_idx = SourceIndex::new(store.clone());
        let tracker = DocStatusTracker::new(store);

        // Record a doc status
        let status = crate::ingestion::doc_status::DocumentStatus {
            source_hash: DocStatusTracker::content_hash("test content"),
            source_id: "tracked-doc".to_string(),
            narrative_id: Some("test-narrative".into()),
            ingested_at: Utc::now(),
            chunk_count: 1,
            entity_count: 1,
            situation_count: 0,
            job_id: "j1".into(),
        };
        tracker.record(&status).unwrap();
        assert!(tracker.is_ingested("test content").unwrap().is_some());

        let entity = make_entity("Charlie");
        let eid = entity.id;
        hg.create_entity(entity).unwrap();
        source_idx.record_entity("tracked-doc", eid).unwrap();

        let report =
            cascade_delete_source("tracked-doc", &hg, &source_idx, Some(&tracker)).unwrap();
        assert_eq!(report.entities_deleted, 1);

        // Doc status should be removed
        assert!(tracker.is_ingested("test content").unwrap().is_none());
    }
}
