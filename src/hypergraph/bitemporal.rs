//! Bi-temporal versioning support.
//!
//! Stores historical versions of entities at `bt/e/{entity_id}/{tx_time_be}`
//! enabling queries like "what did we know about entity X at time T?"
//!
//! Transaction time is recorded automatically on every create/update.
//! The `bt/` prefix is separate from the main `e/` prefix to avoid
//! interfering with current-state queries.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;
use crate::store::KVStore;
use crate::types::Entity;

/// KV prefix for bitemporal entity versions.
const BT_ENTITY_PREFIX: &str = "bt/e/";

/// Store a bitemporal snapshot of an entity at the current transaction time.
pub fn store_bitemporal_snapshot(store: &dyn KVStore, entity: &Entity) -> Result<()> {
    let tx_millis = entity
        .transaction_time
        .unwrap_or_else(chrono::Utc::now)
        .timestamp_millis();
    let key = format!(
        "{}{}/{:016x}",
        BT_ENTITY_PREFIX, entity.id, tx_millis as u64
    );
    let value = serde_json::to_vec(entity)?;
    store.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Retrieve the entity version that was current at a given transaction time.
///
/// Scans bitemporal versions for this entity and returns the latest version
/// whose transaction_time <= the requested time.
pub fn get_entity_as_of(
    store: &dyn KVStore,
    entity_id: &Uuid,
    as_of: DateTime<Utc>,
) -> Result<Option<Entity>> {
    let prefix = format!("{}{}/", BT_ENTITY_PREFIX, entity_id);
    let pairs = store.prefix_scan(prefix.as_bytes())?;

    let as_of_millis = as_of.timestamp_millis();
    let mut best: Option<Entity> = None;

    for (_key, value) in pairs {
        if let Ok(entity) = serde_json::from_slice::<Entity>(&value) {
            let tx_millis = entity
                .transaction_time
                .unwrap_or_else(chrono::Utc::now)
                .timestamp_millis();
            if tx_millis <= as_of_millis {
                // This version was known at or before the requested time
                match &best {
                    None => best = Some(entity),
                    Some(current_best) => {
                        if entity.transaction_time > current_best.transaction_time {
                            best = Some(entity);
                        }
                    }
                }
            }
        }
    }

    Ok(best)
}

/// List all bitemporal versions of an entity, sorted by transaction time.
pub fn list_entity_versions(store: &dyn KVStore, entity_id: &Uuid) -> Result<Vec<Entity>> {
    let prefix = format!("{}{}/", BT_ENTITY_PREFIX, entity_id);
    let pairs = store.prefix_scan(prefix.as_bytes())?;

    let mut versions: Vec<Entity> = pairs
        .into_iter()
        .filter_map(|(_key, value)| serde_json::from_slice(&value).ok())
        .collect();
    versions.sort_by_key(|e| e.transaction_time);
    Ok(versions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Duration;
    use std::sync::Arc;

    fn make_entity(name: &str, tx_time: DateTime<Utc>) -> Entity {
        Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: tx_time,
            updated_at: tx_time,
            deleted_at: None,
            transaction_time: Some(tx_time),
        }
    }

    #[test]
    fn test_bitemporal_store_and_retrieve() {
        let store = Arc::new(MemoryStore::new());
        let now = Utc::now();

        let mut entity = make_entity("Alice", now);
        let id = entity.id;

        // Store v1
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        // Update and store v2
        entity.properties = serde_json::json!({"name": "Alice", "role": "detective"});
        entity.transaction_time = Some(now + Duration::hours(1));
        entity.updated_at = now + Duration::hours(1);
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        // Query as of before v2: should get v1
        let v1 = get_entity_as_of(store.as_ref(), &id, now + Duration::minutes(30))
            .unwrap()
            .expect("v1 should exist");
        assert!(v1.properties.get("role").is_none());

        // Query as of after v2: should get v2
        let v2 = get_entity_as_of(store.as_ref(), &id, now + Duration::hours(2))
            .unwrap()
            .expect("v2 should exist");
        assert_eq!(v2.properties["role"], "detective");
    }

    #[test]
    fn test_bitemporal_as_of_before_any_version() {
        let store = Arc::new(MemoryStore::new());
        let now = Utc::now();

        let entity = make_entity("Bob", now);
        let id = entity.id;
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        // Query before entity existed
        let result = get_entity_as_of(store.as_ref(), &id, now - Duration::hours(1)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_bitemporal_list_versions() {
        let store = Arc::new(MemoryStore::new());
        let now = Utc::now();

        let mut entity = make_entity("Charlie", now);
        let id = entity.id;
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        entity.transaction_time = Some(now + Duration::hours(1));
        entity.confidence = 0.9;
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        entity.transaction_time = Some(now + Duration::hours(2));
        entity.confidence = 0.95;
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        let versions = list_entity_versions(store.as_ref(), &id).unwrap();
        assert_eq!(versions.len(), 3);
        assert!(versions[0].transaction_time < versions[2].transaction_time);
    }

    #[test]
    fn test_bitemporal_nonexistent_entity() {
        let store = Arc::new(MemoryStore::new());
        let result = get_entity_as_of(store.as_ref(), &Uuid::now_v7(), Utc::now()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_bitemporal_exact_time_match() {
        let store = Arc::new(MemoryStore::new());
        let now = Utc::now();

        let entity = make_entity("Dave", now);
        let id = entity.id;
        store_bitemporal_snapshot(store.as_ref(), &entity).unwrap();

        // Query at exact transaction time
        let result = get_entity_as_of(store.as_ref(), &id, now)
            .unwrap()
            .expect("should find at exact time");
        assert_eq!(result.id, id);
    }
}
