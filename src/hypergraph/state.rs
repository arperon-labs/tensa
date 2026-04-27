use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::types::*;
use chrono::{DateTime, Utc};
use uuid::Uuid;

impl Hypergraph {
    /// Create a state version snapshot for an entity at a given situation.
    /// Stored at `sv/{entity_id}/{situation_id}`.
    pub fn create_state_version(&self, state: StateVersion) -> Result<()> {
        let key = keys::state_version_key(&state.entity_id, &state.situation_id);
        let bytes = serde_json::to_vec(&state)?;
        self.store.put(&key, &bytes)
    }

    /// Get entity state at a specific situation.
    pub fn get_state_at_situation(
        &self,
        entity_id: &Uuid,
        situation_id: &Uuid,
    ) -> Result<StateVersion> {
        let key = keys::state_version_key(entity_id, situation_id);
        match self.store.get(&key)? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Err(TensaError::NotFound(format!(
                "State version for entity {} at situation {}",
                entity_id, situation_id
            ))),
        }
    }

    /// Get full state history for an entity (all state versions).
    /// Ordered by situation UUID (v7 UUIDs sort chronologically).
    pub fn get_state_history(&self, entity_id: &Uuid) -> Result<Vec<StateVersion>> {
        let prefix = keys::state_version_prefix(entity_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let sv: StateVersion = serde_json::from_slice(&value)?;
            result.push(sv);
        }
        Ok(result)
    }

    /// Get the latest state version for an entity at or before a given time.
    /// Returns `None` if no state versions exist before the given time.
    /// Uses early termination: since keys are v7-UUID-sorted (chronological),
    /// stops deserializing once a version exceeding `at` is found.
    pub fn get_state_at_time(
        &self,
        entity_id: &Uuid,
        at: DateTime<Utc>,
    ) -> Result<Option<StateVersion>> {
        let prefix = keys::state_version_prefix(entity_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut candidate: Option<StateVersion> = None;
        for (_key, value) in pairs {
            let sv: StateVersion = serde_json::from_slice(&value)?;
            if sv.timestamp > at {
                break;
            }
            candidate = Some(sv);
        }
        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;
    use std::sync::Arc;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    fn test_state(entity_id: Uuid, situation_id: Uuid) -> StateVersion {
        StateVersion {
            entity_id,
            situation_id,
            properties: serde_json::json!({"mood": "neutral"}),
            beliefs: None,
            embedding: None,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_create_state_version() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let sid = Uuid::now_v7();
        let state = test_state(eid, sid);
        hg.create_state_version(state).unwrap();
        let retrieved = hg.get_state_at_situation(&eid, &sid).unwrap();
        assert_eq!(retrieved.entity_id, eid);
        assert_eq!(retrieved.situation_id, sid);
    }

    #[test]
    fn test_get_state_at_situation() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let sid = Uuid::now_v7();
        let mut state = test_state(eid, sid);
        state.properties = serde_json::json!({"mood": "anxious"});
        hg.create_state_version(state).unwrap();

        let retrieved = hg.get_state_at_situation(&eid, &sid).unwrap();
        assert_eq!(retrieved.properties["mood"], "anxious");
    }

    #[test]
    fn test_get_state_not_found() {
        let hg = Hypergraph::new(test_store());
        let result = hg.get_state_at_situation(&Uuid::now_v7(), &Uuid::now_v7());
        assert!(matches!(result, Err(TensaError::NotFound(_))));
    }

    #[test]
    fn test_get_state_history_for_entity() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let s1 = Uuid::now_v7();
        let s2 = Uuid::now_v7();
        let s3 = Uuid::now_v7();

        hg.create_state_version(test_state(eid, s1)).unwrap();
        hg.create_state_version(test_state(eid, s2)).unwrap();
        hg.create_state_version(test_state(eid, s3)).unwrap();

        let history = hg.get_state_history(&eid).unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_state_history_ordered_by_situation() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        // v7 UUIDs created in sequence are chronologically ordered
        let s1 = Uuid::now_v7();
        let s2 = Uuid::now_v7();

        let mut st1 = test_state(eid, s1);
        st1.properties = serde_json::json!({"step": 1});
        let mut st2 = test_state(eid, s2);
        st2.properties = serde_json::json!({"step": 2});

        hg.create_state_version(st1).unwrap();
        hg.create_state_version(st2).unwrap();

        let history = hg.get_state_history(&eid).unwrap();
        assert_eq!(history[0].properties["step"], 1);
        assert_eq!(history[1].properties["step"], 2);
    }

    #[test]
    fn test_multiple_state_versions() {
        let hg = Hypergraph::new(test_store());
        let e1 = Uuid::now_v7();
        let e2 = Uuid::now_v7();
        let sid = Uuid::now_v7();

        hg.create_state_version(test_state(e1, sid)).unwrap();
        hg.create_state_version(test_state(e2, sid)).unwrap();

        // Each entity has its own history
        assert_eq!(hg.get_state_history(&e1).unwrap().len(), 1);
        assert_eq!(hg.get_state_history(&e2).unwrap().len(), 1);
    }

    #[test]
    fn test_state_with_embedding_change() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let s1 = Uuid::now_v7();
        let s2 = Uuid::now_v7();

        let mut st1 = test_state(eid, s1);
        st1.embedding = Some(vec![0.1, 0.2]);
        let mut st2 = test_state(eid, s2);
        st2.embedding = Some(vec![0.3, 0.4]);

        hg.create_state_version(st1).unwrap();
        hg.create_state_version(st2).unwrap();

        let history = hg.get_state_history(&eid).unwrap();
        assert_eq!(history[0].embedding.as_ref().unwrap(), &vec![0.1, 0.2]);
        assert_eq!(history[1].embedding.as_ref().unwrap(), &vec![0.3, 0.4]);
    }

    #[test]
    fn test_state_with_belief_update() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let sid = Uuid::now_v7();

        let mut state = test_state(eid, sid);
        state.beliefs = Some(serde_json::json!({"guilt": 0.9}));
        hg.create_state_version(state).unwrap();

        let retrieved = hg.get_state_at_situation(&eid, &sid).unwrap();
        assert_eq!(retrieved.beliefs.unwrap()["guilt"], 0.9);
    }

    #[test]
    fn test_state_serialization_roundtrip() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let sid = Uuid::now_v7();

        let mut state = test_state(eid, sid);
        state.properties = serde_json::json!({
            "location": "apartment",
            "health": 100,
            "inventory": ["axe", "money"]
        });
        state.beliefs = Some(serde_json::json!({"paranoia": 0.6}));
        state.embedding = Some(vec![0.5; 10]);
        hg.create_state_version(state).unwrap();

        let retrieved = hg.get_state_at_situation(&eid, &sid).unwrap();
        assert_eq!(retrieved.properties["location"], "apartment");
        assert_eq!(retrieved.embedding.as_ref().unwrap().len(), 10);
    }

    #[test]
    fn test_state_history_empty() {
        let hg = Hypergraph::new(test_store());
        let history = hg.get_state_history(&Uuid::now_v7()).unwrap();
        assert!(history.is_empty());
    }

    // ─── get_state_at_time Tests ───────────────────────────────

    #[test]
    fn test_get_state_at_time_exact_match() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let t1 = Utc::now() - chrono::Duration::hours(1);
        let mut sv = test_state(eid, Uuid::now_v7());
        sv.timestamp = t1;
        sv.properties = serde_json::json!({"at_t1": true});
        hg.create_state_version(sv).unwrap();

        let result = hg.get_state_at_time(&eid, t1).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().properties["at_t1"], true);
    }

    #[test]
    fn test_get_state_at_time_between_versions() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let base = Utc::now();
        let t1 = base - chrono::Duration::hours(3);
        let t2 = base - chrono::Duration::hours(2);
        let t3 = base - chrono::Duration::hours(1);

        let mut sv1 = test_state(eid, Uuid::now_v7());
        sv1.timestamp = t1;
        sv1.properties = serde_json::json!({"version": 1});
        hg.create_state_version(sv1).unwrap();

        let mut sv3 = test_state(eid, Uuid::now_v7());
        sv3.timestamp = t3;
        sv3.properties = serde_json::json!({"version": 3});
        hg.create_state_version(sv3).unwrap();

        // Query at t2 (between t1 and t3) should return t1 version
        let result = hg.get_state_at_time(&eid, t2).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().properties["version"], 1);
    }

    #[test]
    fn test_get_state_at_time_before_any() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let base = Utc::now();
        let t_version = base - chrono::Duration::hours(1);
        let t_query = base - chrono::Duration::hours(2);

        let mut sv = test_state(eid, Uuid::now_v7());
        sv.timestamp = t_version;
        hg.create_state_version(sv).unwrap();

        let result = hg.get_state_at_time(&eid, t_query).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_state_at_time_after_all() {
        let hg = Hypergraph::new(test_store());
        let eid = Uuid::now_v7();
        let base = Utc::now();
        let t1 = base - chrono::Duration::hours(3);
        let t2 = base - chrono::Duration::hours(2);
        let t3 = base - chrono::Duration::hours(1);

        let mut sv1 = test_state(eid, Uuid::now_v7());
        sv1.timestamp = t1;
        sv1.properties = serde_json::json!({"version": 1});
        hg.create_state_version(sv1).unwrap();

        let mut sv2 = test_state(eid, Uuid::now_v7());
        sv2.timestamp = t2;
        sv2.properties = serde_json::json!({"version": 2});
        hg.create_state_version(sv2).unwrap();

        // Query at t3 (after all versions) should return t2 version
        let result = hg.get_state_at_time(&eid, t3).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().properties["version"], 2);
    }

    #[test]
    fn test_get_state_at_time_empty_history() {
        let hg = Hypergraph::new(test_store());
        let result = hg.get_state_at_time(&Uuid::now_v7(), Utc::now()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_state_at_time_with_auto_snapshot() {
        let hg = Hypergraph::new(test_store());
        let entity = crate::types::Entity {
            id: Uuid::now_v7(),
            entity_type: crate::types::EntityType::Actor,
            properties: serde_json::json!({"name": "Alice", "score": 10}),
            beliefs: None,
            embedding: None,
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let eid = hg.create_entity(entity).unwrap();

        // Record the time before update
        let before_update = Utc::now();

        // Small sleep to ensure timestamp ordering
        std::thread::sleep(std::time::Duration::from_millis(5));

        // update_entity auto-snapshots the old state
        hg.update_entity(&eid, |e| {
            e.properties = serde_json::json!({"name": "Alice", "score": 99});
        })
        .unwrap();

        let after_update = Utc::now();

        // Query at before_update should return the snapshot (old state)
        let result = hg.get_state_at_time(&eid, before_update).unwrap();
        assert!(result.is_some());
        let snap = result.unwrap();
        assert_eq!(snap.properties["score"], 10);

        // Query at after_update should also return the snapshot (it's the only one,
        // and its timestamp is the entity's old updated_at which is <= after_update)
        let result2 = hg.get_state_at_time(&eid, after_update).unwrap();
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().properties["score"], 10);
    }
}
