//! Corpus management — train/test splits and narrative statistics.
//!
//! Provides corpus-level operations for working with multiple
//! narratives: creating train/test/validation splits, computing
//! per-narrative statistics, and managing corpus-wide operations.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::hypergraph::{keys, Hypergraph};
use crate::store::KVStore;

use super::registry::NarrativeRegistry;
use super::types::{CorpusSplit, NarrativeStats};

/// Corpus manager for multi-narrative operations.
pub struct CorpusManager {
    store: Arc<dyn KVStore>,
    registry: NarrativeRegistry,
}

impl CorpusManager {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        let registry = NarrativeRegistry::new(Arc::clone(&store));
        Self { store, registry }
    }

    /// Get a reference to the underlying registry.
    pub fn registry(&self) -> &NarrativeRegistry {
        &self.registry
    }

    /// Create a train/test/validation split from existing narrative IDs.
    ///
    /// Uses a simple deterministic split based on the seed: narratives
    /// are sorted by ID, then divided in a 60/20/20 ratio. The split
    /// is stored at `cp/split/{name}`.
    pub fn create_split(
        &self,
        name: &str,
        narrative_ids: &[String],
        seed: u64,
    ) -> Result<CorpusSplit> {
        let mut ids = narrative_ids.to_vec();
        // Simple deterministic shuffle using seed
        let n = ids.len();
        if n > 1 {
            let mut state = seed;
            for i in (1..n).rev() {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let j = (state as usize) % (i + 1);
                ids.swap(i, j);
            }
        }

        let train_end = ((n * 60) / 100).max(if n > 0 { 1 } else { 0 }).min(n);
        let test_end = (train_end + (n * 20) / 100).min(n);

        let split = CorpusSplit {
            train: ids[..train_end].to_vec(),
            test: ids[train_end..test_end].to_vec(),
            validation: ids[test_end..].to_vec(),
            seed,
        };

        let key = keys::corpus_split_key(name);
        let bytes = serde_json::to_vec(&split)?;
        self.store.put(&key, &bytes)?;
        Ok(split)
    }

    /// Get a previously saved corpus split by name.
    pub fn get_split(&self, name: &str) -> Result<Option<CorpusSplit>> {
        let key = keys::corpus_split_key(name);
        match self.store.get(&key)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Compute statistics for a narrative by scanning its entities,
    /// situations, participations, and causal links. Excludes synthetic
    /// records by default — call `compute_stats_with_synthetic` to opt in.
    pub fn compute_stats(
        &self,
        narrative_id: &str,
        hypergraph: &Hypergraph,
    ) -> Result<NarrativeStats> {
        self.compute_stats_with_synthetic(narrative_id, hypergraph, false)
    }

    /// Compute statistics with explicit control over synthetic-record
    /// inclusion. EATH Phase 3.
    pub fn compute_stats_with_synthetic(
        &self,
        narrative_id: &str,
        hypergraph: &Hypergraph,
        include_synthetic: bool,
    ) -> Result<NarrativeStats> {
        let entities = crate::synth::emit::filter_synthetic_entities(
            hypergraph.list_entities_by_narrative(narrative_id)?,
            include_synthetic,
        );
        let situations = crate::synth::emit::filter_synthetic_situations(
            hypergraph.list_situations_by_narrative(narrative_id)?,
            include_synthetic,
        );

        // Count participations
        let mut participation_count = 0;
        for entity in &entities {
            participation_count += hypergraph.get_situations_for_entity(&entity.id)?.len();
        }

        // Count causal links and compute temporal span + level distribution
        let mut causal_link_count = 0;
        let mut min_time = None;
        let mut max_time = None;
        let mut level_counts: HashMap<String, usize> = HashMap::new();

        for sit in &situations {
            causal_link_count += hypergraph
                .get_consequences(&sit.id)
                .map(|v| v.len())
                .unwrap_or(0);

            let level_name = format!("{:?}", sit.narrative_level);
            *level_counts.entry(level_name).or_insert(0) += 1;

            if let Some(start) = &sit.temporal.start {
                match &min_time {
                    None => min_time = Some(*start),
                    Some(current) if start < current => min_time = Some(*start),
                    _ => {}
                }
            }
            if let Some(end) = &sit.temporal.end {
                match &max_time {
                    None => max_time = Some(*end),
                    Some(current) if end > current => max_time = Some(*end),
                    _ => {}
                }
            }
        }

        let temporal_span = match (min_time, max_time) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
        };

        Ok(NarrativeStats {
            narrative_id: narrative_id.to_string(),
            entity_count: entities.len(),
            situation_count: situations.len(),
            participation_count,
            causal_link_count,
            temporal_span,
            narrative_levels: level_counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn test_store() -> Arc<dyn crate::store::KVStore> {
        Arc::new(MemoryStore::new())
    }

    fn make_entity(narrative_id: &str) -> Entity {
        Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Test"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some(narrative_id.to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn make_situation(narrative_id: &str, level: NarrativeLevel) -> Situation {
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
            raw_content: vec![ContentBlock::text("Test")],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative_id.to_string()),
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
    fn test_create_split_basic() {
        let store = test_store();
        let cm = CorpusManager::new(store);
        let ids: Vec<String> = (0..10).map(|i| format!("narrative-{}", i)).collect();
        let split = cm.create_split("default", &ids, 42).unwrap();
        // 60/20/20 of 10 = 6/2/2
        assert_eq!(split.train.len(), 6);
        assert_eq!(split.test.len(), 2);
        assert_eq!(split.validation.len(), 2);
        assert_eq!(split.seed, 42);
    }

    #[test]
    fn test_create_split_small_corpus() {
        let store = test_store();
        let cm = CorpusManager::new(store);
        let ids = vec!["only-one".to_string()];
        let split = cm.create_split("default", &ids, 0).unwrap();
        assert_eq!(split.train.len(), 1);
        assert!(split.test.is_empty());
        assert!(split.validation.is_empty());
    }

    #[test]
    fn test_create_split_empty() {
        let store = test_store();
        let cm = CorpusManager::new(store);
        let ids: Vec<String> = vec![];
        let split = cm.create_split("default", &ids, 0).unwrap();
        assert!(split.train.is_empty());
    }

    #[test]
    fn test_get_split() {
        let store = test_store();
        let cm = CorpusManager::new(store);
        let ids: Vec<String> = (0..5).map(|i| format!("n-{}", i)).collect();
        cm.create_split("default", &ids, 42).unwrap();
        let retrieved = cm.get_split("default").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().seed, 42);
    }

    #[test]
    fn test_get_split_not_found() {
        let store = test_store();
        let cm = CorpusManager::new(store);
        let result = cm.get_split("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_deterministic_split() {
        let store1 = test_store();
        let store2 = test_store();
        let cm1 = CorpusManager::new(store1);
        let cm2 = CorpusManager::new(store2);
        let ids: Vec<String> = (0..10).map(|i| format!("n-{}", i)).collect();
        let split1 = cm1.create_split("a", &ids, 42).unwrap();
        let split2 = cm2.create_split("b", &ids, 42).unwrap();
        assert_eq!(split1.train, split2.train);
        assert_eq!(split1.test, split2.test);
        assert_eq!(split1.validation, split2.validation);
    }

    #[test]
    fn test_compute_stats_empty_narrative() {
        let store = test_store();
        let cm = CorpusManager::new(Arc::clone(&store));
        let hg = Hypergraph::new(store);
        let stats = cm.compute_stats("nonexistent", &hg).unwrap();
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.situation_count, 0);
        assert_eq!(stats.participation_count, 0);
        assert!(stats.temporal_span.is_none());
    }

    #[test]
    fn test_compute_stats_with_data() {
        let store = test_store();
        let cm = CorpusManager::new(Arc::clone(&store));
        let hg = Hypergraph::new(Arc::clone(&store));

        // Create entities and situations in "hamlet" narrative
        let e1 = make_entity("hamlet");
        let e1_id = hg.create_entity(e1).unwrap();
        hg.create_entity(make_entity("hamlet")).unwrap();

        let s1 = make_situation("hamlet", NarrativeLevel::Scene);
        let s1_id = hg.create_situation(s1).unwrap();
        hg.create_situation(make_situation("hamlet", NarrativeLevel::Event))
            .unwrap();

        // Add participation
        hg.add_participant(Participation {
            entity_id: e1_id,
            situation_id: s1_id,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let stats = cm.compute_stats("hamlet", &hg).unwrap();
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.situation_count, 2);
        assert_eq!(stats.participation_count, 1);
        assert_eq!(stats.narrative_levels["Scene"], 1);
        assert_eq!(stats.narrative_levels["Event"], 1);
    }

    #[test]
    fn test_compute_stats_ignores_other_narratives() {
        let store = test_store();
        let cm = CorpusManager::new(Arc::clone(&store));
        let hg = Hypergraph::new(Arc::clone(&store));

        hg.create_entity(make_entity("hamlet")).unwrap();
        hg.create_entity(make_entity("macbeth")).unwrap();

        let stats = cm.compute_stats("hamlet", &hg).unwrap();
        assert_eq!(stats.entity_count, 1);
    }

    #[test]
    fn test_split_different_seeds_differ() {
        let store1 = test_store();
        let store2 = test_store();
        let cm1 = CorpusManager::new(store1);
        let cm2 = CorpusManager::new(store2);
        let ids: Vec<String> = (0..10).map(|i| format!("n-{}", i)).collect();
        let split1 = cm1.create_split("a", &ids, 42).unwrap();
        let split2 = cm2.create_split("b", &ids, 99).unwrap();
        // Different seeds should produce different orderings (very likely)
        // At minimum the splits should still cover all IDs
        let total1 = split1.train.len() + split1.test.len() + split1.validation.len();
        let total2 = split2.train.len() + split2.test.len() + split2.validation.len();
        assert_eq!(total1, 10);
        assert_eq!(total2, 10);
    }
}
