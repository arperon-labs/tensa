use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;
use crate::temporal::interval::relation_between;
use crate::types::{AllenInterval, AllenRelation, TimeGranularity};

const META_ALLEN_TREE: &[u8] = b"meta/allen_tree";

/// Entry in the interval tree: a situation's temporal extent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalEntry {
    pub situation_id: Uuid,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// In-memory interval index backed by a sorted Vec.
/// Persisted to KV store at `meta/allen_tree`.
pub struct IntervalTree {
    entries: Vec<IntervalEntry>,
}

impl IntervalTree {
    /// Create a new empty tree.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert a situation's interval. Maintains sorted order by start time.
    pub fn insert(&mut self, entry: IntervalEntry) {
        let pos = self
            .entries
            .binary_search_by(|e| e.start.cmp(&entry.start))
            .unwrap_or_else(|pos| pos);
        self.entries.insert(pos, entry);
    }

    /// Remove a situation's interval by situation_id. Returns true if found.
    pub fn delete(&mut self, situation_id: &Uuid) -> bool {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|e| e.situation_id == *situation_id)
        {
            self.entries.remove(pos);
            true
        } else {
            false
        }
    }

    /// Point query: which situations are active at a given time?
    pub fn point_query(&self, time: &DateTime<Utc>) -> Vec<&IntervalEntry> {
        self.entries
            .iter()
            .filter(|e| e.start <= *time && e.end >= *time)
            .collect()
    }

    /// Interval query: which situations overlap with [start, end]?
    pub fn interval_query(
        &self,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Vec<&IntervalEntry> {
        self.entries
            .iter()
            .filter(|e| e.start <= *end && e.end >= *start)
            .collect()
    }

    /// Allen relation query: find all situations with a given Allen relation
    /// to the reference interval.
    pub fn allen_relation_query(
        &self,
        reference: &IntervalEntry,
        relation: AllenRelation,
    ) -> Vec<&IntervalEntry> {
        let ref_interval = AllenInterval {
            start: Some(reference.start),
            end: Some(reference.end),
            granularity: TimeGranularity::Exact,
            relations: vec![],
            fuzzy_endpoints: None,
        };
        self.entries
            .iter()
            .filter(|e| {
                let entry_interval = AllenInterval {
                    start: Some(e.start),
                    end: Some(e.end),
                    granularity: TimeGranularity::Exact,
                    relations: vec![],
                    fuzzy_endpoints: None,
                };
                relation_between(&ref_interval, &entry_interval)
                    .map(|r| r == relation)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize and persist to KV store at `meta/allen_tree`.
    pub fn save(&self, store: &dyn KVStore) -> Result<()> {
        let bytes = serde_json::to_vec(&self.entries)?;
        store.put(META_ALLEN_TREE, &bytes)
    }

    /// Load from KV store. Returns empty tree if key not found.
    pub fn load(store: &dyn KVStore) -> Result<Self> {
        match store.get(META_ALLEN_TREE)? {
            Some(bytes) => {
                let entries: Vec<IntervalEntry> = serde_json::from_slice(&bytes)?;
                Ok(Self { entries })
            }
            None => Ok(Self::new()),
        }
    }

    /// Rebuild tree from all situations in the store.
    /// Scans all `s/` keys, extracts temporal intervals, inserts valid ones.
    pub fn rebuild(store: &dyn KVStore) -> Result<Self> {
        use crate::types::Situation;
        let pairs = store.prefix_scan(b"s/")?;
        let mut tree = Self::new();
        for (_key, value) in pairs {
            let sit: Situation = serde_json::from_slice(&value)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            if let (Some(start), Some(end)) = (sit.temporal.start, sit.temporal.end) {
                tree.insert(IntervalEntry {
                    situation_id: sit.id,
                    start,
                    end,
                });
            }
        }
        Ok(tree)
    }
}

impl Default for IntervalTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Fuzzy Sprint Phase 5 — graded-Allen query over the situation store.
///
/// Lives on the `temporal` layer so callers don't reach into the `fuzzy`
/// module directly. Iterates over all situations in `narrative_id` since
/// no crisp index path covers graded filtering.
///
/// Cites: [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen].
pub fn fuzzy_relation_query(
    store: &dyn KVStore,
    narrative_id: &str,
    reference: &AllenInterval,
    rel: AllenRelation,
    threshold: f64,
) -> Result<Vec<Uuid>> {
    crate::fuzzy::allen::fuzzy_relation_query_situations(
        store,
        narrative_id,
        reference,
        rel,
        threshold,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Duration;
    use std::sync::Arc;

    fn make_entry(id: Uuid, start_offset: i64, end_offset: i64) -> IntervalEntry {
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        IntervalEntry {
            situation_id: id,
            start: base + Duration::hours(start_offset),
            end: base + Duration::hours(end_offset),
        }
    }

    #[test]
    fn test_insert_and_point_query() {
        let mut tree = IntervalTree::new();
        let id = Uuid::now_v7();
        tree.insert(make_entry(id, 0, 10));

        let base = DateTime::parse_from_rfc3339("2025-01-01T05:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let results = tree.point_query(&base);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].situation_id, id);
    }

    #[test]
    fn test_point_query_no_match() {
        let mut tree = IntervalTree::new();
        tree.insert(make_entry(Uuid::now_v7(), 0, 5));

        let time = DateTime::parse_from_rfc3339("2025-01-01T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let results = tree.point_query(&time);
        assert!(results.is_empty());
    }

    #[test]
    fn test_interval_overlap_query() {
        let mut tree = IntervalTree::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();
        tree.insert(make_entry(id1, 0, 5));
        tree.insert(make_entry(id2, 3, 8));
        tree.insert(make_entry(id3, 10, 15));

        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let results =
            tree.interval_query(&(base + Duration::hours(4)), &(base + Duration::hours(6)));
        assert_eq!(results.len(), 2); // id1 and id2 overlap [4,6]
    }

    #[test]
    fn test_allen_relation_query_before() {
        let mut tree = IntervalTree::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();
        tree.insert(make_entry(id1, 0, 2));
        tree.insert(make_entry(id2, 5, 8));
        tree.insert(make_entry(id3, 10, 12));

        let reference = make_entry(Uuid::now_v7(), 3, 4);
        let results = tree.allen_relation_query(&reference, AllenRelation::After);
        // Reference [3,4] is AFTER [0,2] => id1
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].situation_id, id1);
    }

    #[test]
    fn test_allen_relation_query_during() {
        let mut tree = IntervalTree::new();
        let id = Uuid::now_v7();
        tree.insert(make_entry(id, 0, 10));

        let reference = make_entry(Uuid::now_v7(), 0, 10);
        let results = tree.allen_relation_query(&reference, AllenRelation::Contains);
        // Reference [0,10] contains nothing of [0,10] (that's Equals)
        assert!(results.is_empty());

        let results_eq = tree.allen_relation_query(&reference, AllenRelation::Equals);
        assert_eq!(results_eq.len(), 1);
    }

    #[test]
    fn test_bulk_insert() {
        let mut tree = IntervalTree::new();
        for i in 0..100 {
            tree.insert(make_entry(Uuid::now_v7(), i * 2, i * 2 + 1));
        }
        assert_eq!(tree.len(), 100);
    }

    #[test]
    fn test_delete_from_tree() {
        let mut tree = IntervalTree::new();
        let id = Uuid::now_v7();
        tree.insert(make_entry(id, 0, 5));
        assert_eq!(tree.len(), 1);
        assert!(tree.delete(&id));
        assert_eq!(tree.len(), 0);
        assert!(!tree.delete(&id)); // already gone
    }

    #[test]
    fn test_persistence_to_kv_and_rebuild() {
        let store = Arc::new(MemoryStore::new());
        let mut tree = IntervalTree::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        tree.insert(make_entry(id1, 0, 5));
        tree.insert(make_entry(id2, 3, 8));
        tree.save(store.as_ref()).unwrap();

        let loaded = IntervalTree::load(store.as_ref()).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_empty_tree_queries() {
        let tree = IntervalTree::new();
        let time = Utc::now();
        assert!(tree.point_query(&time).is_empty());
        assert!(tree
            .interval_query(&time, &(time + Duration::hours(1)))
            .is_empty());
    }

    #[test]
    fn test_load_empty_store() {
        let store = MemoryStore::new();
        let tree = IntervalTree::load(&store).unwrap();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_large_tree_performance() {
        let mut tree = IntervalTree::new();
        for i in 0..1000 {
            tree.insert(make_entry(Uuid::now_v7(), i, i + 10));
        }
        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let results = tree.point_query(&(base + Duration::hours(500)));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_rebuild_from_situations() {
        use crate::hypergraph::Hypergraph;
        use crate::types::*;

        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());

        let base = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        for i in 0..3 {
            let sit = Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(base + Duration::hours(i * 10)),
                    end: Some(base + Duration::hours(i * 10 + 5)),
                    granularity: TimeGranularity::Exact,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![ContentBlock::text("Test")],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.8,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: None,
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
            };
            hg.create_situation(sit).unwrap();
        }

        let tree = IntervalTree::rebuild(store.as_ref()).unwrap();
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_sorted_insertion_order() {
        let mut tree = IntervalTree::new();
        // Insert out of order
        tree.insert(make_entry(Uuid::now_v7(), 10, 15));
        tree.insert(make_entry(Uuid::now_v7(), 0, 5));
        tree.insert(make_entry(Uuid::now_v7(), 5, 10));

        // Verify sorted by start
        for i in 1..tree.entries.len() {
            assert!(tree.entries[i - 1].start <= tree.entries[i].start);
        }
    }
}
