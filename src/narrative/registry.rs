//! Narrative registry — KV-backed CRUD for narrative metadata.
//!
//! Narratives are stored at `nr/{narrative_id}` in the KV store.
//! The registry provides create/get/update/delete/list operations.

use std::sync::Arc;

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::hypergraph::keys;
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;

use super::types::{Narrative, NarrativeMergeReport};

/// Registry for managing narrative metadata in the KV store.
pub struct NarrativeRegistry {
    store: Arc<dyn KVStore>,
}

impl NarrativeRegistry {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Create a new narrative. Returns `NarrativeExists` if the ID is taken.
    /// If `project_id` is set, also writes a project-narrative index entry.
    pub fn create(&self, narrative: Narrative) -> Result<String> {
        let key = keys::narrative_key(&narrative.id);
        if self.store.get(&key)?.is_some() {
            return Err(TensaError::NarrativeExists(narrative.id));
        }
        // Write project-narrative index
        if let Some(ref pid) = narrative.project_id {
            let idx_key = keys::project_narrative_index_key(pid, &narrative.id);
            self.store.put(&idx_key, &[])?;
        }
        let bytes = serde_json::to_vec(&narrative)?;
        self.store.put(&key, &bytes)?;
        Ok(narrative.id)
    }

    /// Get a narrative by ID. Returns `NarrativeNotFound` if missing.
    pub fn get(&self, id: &str) -> Result<Narrative> {
        let key = keys::narrative_key(id);
        match self.store.get(&key)? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Err(TensaError::NarrativeNotFound(id.to_string())),
        }
    }

    /// Update a narrative by applying a closure. Automatically updates `updated_at`.
    pub fn update(&self, id: &str, updater: impl FnOnce(&mut Narrative)) -> Result<Narrative> {
        let mut narrative = self.get(id)?;
        updater(&mut narrative);
        narrative.updated_at = Utc::now();
        let key = keys::narrative_key(id);
        let bytes = serde_json::to_vec(&narrative)?;
        self.store.put(&key, &bytes)?;
        Ok(narrative)
    }

    /// Delete a narrative by ID. Returns `NarrativeNotFound` if missing.
    /// Also removes the project-narrative index entry if `project_id` is set.
    pub fn delete(&self, id: &str) -> Result<()> {
        let narrative = self.get(id)?;
        if let Some(ref pid) = narrative.project_id {
            let idx_key = keys::project_narrative_index_key(pid, id);
            let _ = self.store.delete(&idx_key);
        }
        self.store.delete(&keys::narrative_key(id))
    }

    /// List all narratives. Optionally filter by genre and/or tag.
    pub fn list(
        &self,
        genre_filter: Option<&str>,
        tag_filter: Option<&str>,
    ) -> Result<Vec<Narrative>> {
        let pairs = self.store.prefix_scan(&keys::narrative_prefix())?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let narrative: Narrative = serde_json::from_slice(&value)?;
            if let Some(genre) = genre_filter {
                if narrative.genre.as_deref() != Some(genre) {
                    continue;
                }
            }
            if let Some(tag) = tag_filter {
                if !narrative.tags.iter().any(|t| t == tag) {
                    continue;
                }
            }
            result.push(narrative);
        }
        Ok(result)
    }

    /// Merge source narrative into target. Moves all entities, situations,
    /// and chunks from source to target, combines metadata (tags, authors),
    /// then deletes the source narrative record.
    ///
    /// Entity deduplication is NOT automatic — run `merge_entities` separately
    /// if needed after the narrative merge.
    pub fn merge_narratives(
        &self,
        target_id: &str,
        source_id: &str,
        hypergraph: &Hypergraph,
    ) -> Result<NarrativeMergeReport> {
        if target_id == source_id {
            return Err(TensaError::NarrativeMergeError(
                "Cannot merge a narrative with itself".into(),
            ));
        }
        let _target = self.get(target_id)?;
        let _source = self.get(source_id)?;

        // 1. Re-assign entities
        let entities = hypergraph.list_entities_by_narrative(source_id)?;
        let entities_moved = entities.len();
        let target_id_str = target_id.to_string();
        for ent in &entities {
            let tid = target_id_str.clone();
            hypergraph.update_entity_no_snapshot(&ent.id, move |e| {
                e.narrative_id = Some(tid);
            })?;
        }

        // 2. Re-assign situations
        let situations = hypergraph.list_situations_by_narrative(source_id)?;
        let situations_moved = situations.len();
        for sit in &situations {
            let tid = target_id_str.clone();
            hypergraph.update_situation(&sit.id, move |s| {
                s.narrative_id = Some(tid);
            })?;
        }

        // 3. Move chunk narrative index entries
        let chunk_prefix = keys::chunk_narrative_prefix(source_id);
        let chunk_pairs = self.store.prefix_scan(&chunk_prefix)?;
        let chunks_moved = chunk_pairs.len();
        for (old_key, value) in &chunk_pairs {
            // Extract chunk_index bytes from old key suffix
            let suffix = &old_key[chunk_prefix.len()..];
            // Rewrite as ch/n/{target_id}/{chunk_index}
            let mut new_key = keys::chunk_narrative_prefix(target_id);
            new_key.extend_from_slice(suffix);
            self.store.put(&new_key, value)?;
            self.store.delete(old_key)?;
        }

        // Also update chunk records' narrative_id if stored
        let chunk_record_prefix = keys::chunk_narrative_prefix(source_id);
        // ChunkRecords store narrative_id in JSON — scan ch/r/ and patch
        let cr_prefix = keys::CHUNK_RECORD.to_vec();
        let cr_pairs = self.store.prefix_scan(&cr_prefix)?;
        for (key, value) in cr_pairs {
            if let Ok(mut record) = serde_json::from_slice::<serde_json::Value>(&value) {
                if record.get("narrative_id").and_then(|v| v.as_str()) == Some(source_id) {
                    record["narrative_id"] = serde_json::json!(target_id);
                    let bytes = serde_json::to_vec(&record)?;
                    self.store.put(&key, &bytes)?;
                }
            }
        }
        // Suppress unused variable warning
        let _ = chunk_record_prefix;

        // 4. Merge metadata into target (combine tags, authors)
        let source = self.get(source_id).ok();
        self.update(target_id, |t| {
            if let Some(ref src) = source {
                // Merge tags (dedup)
                for tag in &src.tags {
                    if !t.tags.contains(tag) {
                        t.tags.push(tag.clone());
                    }
                }
                // Merge authors (dedup)
                for author in &src.authors {
                    if !t.authors.contains(author) {
                        t.authors.push(author.clone());
                    }
                }
                // Accumulate counts
                t.entity_count = t.entity_count.saturating_add(src.entity_count);
                t.situation_count = t.situation_count.saturating_add(src.situation_count);
            }
        })?;

        // 5. Delete source narrative
        self.delete(source_id)?;

        Ok(NarrativeMergeReport {
            target_id: target_id.to_string(),
            source_id: source_id.to_string(),
            entities_moved,
            situations_moved,
            chunks_moved,
            source_deleted: true,
        })
    }

    /// List narratives with cursor-based pagination.
    /// Returns `(items, next_cursor)`. The cursor is the narrative ID.
    pub fn list_paginated(
        &self,
        limit: usize,
        after: Option<&str>,
    ) -> Result<(Vec<Narrative>, Option<String>)> {
        let prefix = keys::narrative_prefix();
        let start = match after {
            Some(cursor) => {
                let mut k = keys::narrative_key(cursor);
                k.push(0); // byte after cursor key to exclude it
                k
            }
            None => prefix.clone(),
        };
        let mut end = prefix;
        end.push(0xFF);

        let pairs = self.store.range(&start, &end)?;
        let mut result = Vec::with_capacity(limit + 1);
        for (_key, value) in pairs.iter().take(limit + 1) {
            result.push(serde_json::from_slice::<Narrative>(value)?);
        }

        let next_cursor = if result.len() > limit {
            result.pop(); // discard the extra item
            result.last().map(|n| n.id.clone())
        } else {
            None
        };

        Ok((result, next_cursor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::SourceReference;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    fn test_narrative(id: &str, title: &str) -> Narrative {
        Narrative {
            id: id.to_string(),
            title: title.to_string(),
            genre: None,
            tags: vec![],
            source: None,
            project_id: None,
            description: None,
            authors: vec![],
            language: None,
            publication_date: None,
            cover_url: None,
            custom_properties: std::collections::HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_create_narrative() {
        let reg = NarrativeRegistry::new(test_store());
        let id = reg.create(test_narrative("hamlet", "Hamlet")).unwrap();
        assert_eq!(id, "hamlet");
    }

    #[test]
    fn test_get_narrative() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("hamlet", "Hamlet")).unwrap();
        let n = reg.get("hamlet").unwrap();
        assert_eq!(n.title, "Hamlet");
    }

    #[test]
    fn test_get_narrative_not_found() {
        let reg = NarrativeRegistry::new(test_store());
        let result = reg.get("nonexistent");
        assert!(matches!(result, Err(TensaError::NarrativeNotFound(_))));
    }

    #[test]
    fn test_create_duplicate_narrative() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("hamlet", "Hamlet")).unwrap();
        let result = reg.create(test_narrative("hamlet", "Hamlet 2"));
        assert!(matches!(result, Err(TensaError::NarrativeExists(_))));
    }

    #[test]
    fn test_update_narrative() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("hamlet", "Hamlet")).unwrap();
        let updated = reg
            .update("hamlet", |n| {
                n.genre = Some("tragedy".to_string());
            })
            .unwrap();
        assert_eq!(updated.genre, Some("tragedy".to_string()));
        // Verify persisted
        let retrieved = reg.get("hamlet").unwrap();
        assert_eq!(retrieved.genre, Some("tragedy".to_string()));
    }

    #[test]
    fn test_delete_narrative() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("hamlet", "Hamlet")).unwrap();
        reg.delete("hamlet").unwrap();
        assert!(matches!(
            reg.get("hamlet"),
            Err(TensaError::NarrativeNotFound(_))
        ));
    }

    #[test]
    fn test_delete_narrative_not_found() {
        let reg = NarrativeRegistry::new(test_store());
        assert!(matches!(
            reg.delete("nonexistent"),
            Err(TensaError::NarrativeNotFound(_))
        ));
    }

    #[test]
    fn test_list_narratives_all() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("hamlet", "Hamlet")).unwrap();
        reg.create(test_narrative("macbeth", "Macbeth")).unwrap();
        let all = reg.list(None, None).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_list_narratives_empty() {
        let reg = NarrativeRegistry::new(test_store());
        let all = reg.list(None, None).unwrap();
        assert!(all.is_empty());
    }

    #[test]
    fn test_list_narratives_filter_genre() {
        let reg = NarrativeRegistry::new(test_store());
        let mut n1 = test_narrative("hamlet", "Hamlet");
        n1.genre = Some("tragedy".to_string());
        let mut n2 = test_narrative("comedy", "A Comedy");
        n2.genre = Some("comedy".to_string());
        reg.create(n1).unwrap();
        reg.create(n2).unwrap();

        let tragedies = reg.list(Some("tragedy"), None).unwrap();
        assert_eq!(tragedies.len(), 1);
        assert_eq!(tragedies[0].id, "hamlet");
    }

    #[test]
    fn test_list_narratives_filter_tag() {
        let reg = NarrativeRegistry::new(test_store());
        let mut n1 = test_narrative("hamlet", "Hamlet");
        n1.tags = vec!["shakespeare".to_string()];
        let mut n2 = test_narrative("dostoevsky", "C&P");
        n2.tags = vec!["russian".to_string()];
        reg.create(n1).unwrap();
        reg.create(n2).unwrap();

        let shakespeare = reg.list(None, Some("shakespeare")).unwrap();
        assert_eq!(shakespeare.len(), 1);
        assert_eq!(shakespeare[0].id, "hamlet");
    }

    #[test]
    fn test_narrative_with_source() {
        let reg = NarrativeRegistry::new(test_store());
        let mut n = test_narrative("hamlet", "Hamlet");
        n.source = Some(SourceReference {
            source_type: "gutenberg".to_string(),
            source_id: Some("PG1524".to_string()),
            description: Some("Project Gutenberg".to_string()),
            timestamp: Utc::now(),
            registered_source: None,
        });
        reg.create(n).unwrap();
        let retrieved = reg.get("hamlet").unwrap();
        assert!(retrieved.source.is_some());
        assert_eq!(retrieved.source.unwrap().source_type, "gutenberg");
    }

    #[test]
    fn test_list_paginated_basic() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("alpha", "Alpha")).unwrap();
        reg.create(test_narrative("beta", "Beta")).unwrap();
        reg.create(test_narrative("gamma", "Gamma")).unwrap();
        reg.create(test_narrative("delta", "Delta")).unwrap();
        reg.create(test_narrative("epsilon", "Epsilon")).unwrap();

        // First page
        let (page1, cursor1) = reg.list_paginated(2, None).unwrap();
        assert_eq!(page1.len(), 2);
        assert!(cursor1.is_some());

        // Second page
        let (page2, cursor2) = reg.list_paginated(2, cursor1.as_deref()).unwrap();
        assert_eq!(page2.len(), 2);
        assert!(cursor2.is_some());

        // Third page (last item)
        let (page3, cursor3) = reg.list_paginated(2, cursor2.as_deref()).unwrap();
        assert_eq!(page3.len(), 1);
        assert!(cursor3.is_none());

        // Verify no overlap
        let all_ids: Vec<String> = page1
            .iter()
            .chain(page2.iter())
            .chain(page3.iter())
            .map(|n| n.id.clone())
            .collect();
        assert_eq!(all_ids.len(), 5);
    }

    #[test]
    fn test_list_paginated_empty() {
        let reg = NarrativeRegistry::new(test_store());
        let (items, cursor) = reg.list_paginated(50, None).unwrap();
        assert!(items.is_empty());
        assert!(cursor.is_none());
    }

    #[test]
    fn test_list_paginated_invalid_cursor() {
        let reg = NarrativeRegistry::new(test_store());
        reg.create(test_narrative("alpha", "Alpha")).unwrap();
        // Invalid cursor — returns empty since cursor is never found
        let (items, cursor) = reg.list_paginated(50, Some("nonexistent")).unwrap();
        assert!(items.is_empty());
        assert!(cursor.is_none());
    }

    #[test]
    fn test_merge_narratives_basic() {
        let store = test_store();
        let hg = crate::hypergraph::Hypergraph::new(store.clone());
        let reg = NarrativeRegistry::new(store.clone());

        // Create target and source narratives
        let mut target = test_narrative("target", "Target");
        target.tags = vec!["fantasy".to_string()];
        target.authors = vec!["Author A".to_string()];
        reg.create(target).unwrap();

        let mut source = test_narrative("source", "Source");
        source.tags = vec!["fantasy".to_string(), "epic".to_string()];
        source.authors = vec!["Author B".to_string()];
        reg.create(source).unwrap();

        // Create entities in source narrative
        let ent = crate::types::Entity {
            id: uuid::Uuid::now_v7(),
            entity_type: crate::types::EntityType::Actor,
            properties: serde_json::json!({"name": "Gandalf"}),
            beliefs: None,
            embedding: None,
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("source".to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let ent_id = hg.create_entity(ent).unwrap();

        // Merge
        let report = reg.merge_narratives("target", "source", &hg).unwrap();
        assert_eq!(report.entities_moved, 1);
        assert!(report.source_deleted);

        // Entity is now in target
        let moved = hg.get_entity(&ent_id).unwrap();
        assert_eq!(moved.narrative_id, Some("target".to_string()));

        // Source narrative is gone
        assert!(matches!(
            reg.get("source"),
            Err(TensaError::NarrativeNotFound(_))
        ));

        // Tags merged and deduped
        let updated_target = reg.get("target").unwrap();
        assert!(updated_target.tags.contains(&"fantasy".to_string()));
        assert!(updated_target.tags.contains(&"epic".to_string()));
        assert!(updated_target.authors.contains(&"Author A".to_string()));
        assert!(updated_target.authors.contains(&"Author B".to_string()));
    }

    #[test]
    fn test_merge_narratives_self_rejected() {
        let store = test_store();
        let hg = crate::hypergraph::Hypergraph::new(store.clone());
        let reg = NarrativeRegistry::new(store);
        reg.create(test_narrative("alpha", "Alpha")).unwrap();
        let result = reg.merge_narratives("alpha", "alpha", &hg);
        assert!(matches!(result, Err(TensaError::NarrativeMergeError(_))));
    }

    #[test]
    fn test_merge_narratives_missing_source() {
        let store = test_store();
        let hg = crate::hypergraph::Hypergraph::new(store.clone());
        let reg = NarrativeRegistry::new(store);
        reg.create(test_narrative("alpha", "Alpha")).unwrap();
        let result = reg.merge_narratives("alpha", "nonexistent", &hg);
        assert!(matches!(result, Err(TensaError::NarrativeNotFound(_))));
    }

    #[test]
    fn test_update_narrative_updates_timestamp() {
        let reg = NarrativeRegistry::new(test_store());
        let n = test_narrative("hamlet", "Hamlet");
        let original_time = n.updated_at;
        reg.create(n).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let updated = reg
            .update("hamlet", |n| {
                n.entity_count = 5;
            })
            .unwrap();
        assert!(updated.updated_at > original_time);
    }
}
