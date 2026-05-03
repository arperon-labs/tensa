//! KV operations for image metadata + bytes.

use std::sync::Arc;

use uuid::Uuid;

use super::types::ImageRecord;
use super::{IMG_BYTES_PREFIX, IMG_INDEX_PREFIX, IMG_META_PREFIX};
use crate::error::{Result, TensaError};
use crate::store::KVStore;

/// `img/m/{narrative_id}/{image_id_v7_BE_BIN_16}` — narrative is a string so we
/// can prefix-scan all images for a given narrative when exporting.
pub fn image_meta_key(narrative_id: &str, image_id: &Uuid) -> Vec<u8> {
    let mut key = Vec::with_capacity(IMG_META_PREFIX.len() + narrative_id.len() + 1 + 16);
    key.extend_from_slice(IMG_META_PREFIX);
    key.extend_from_slice(narrative_id.as_bytes());
    key.push(b'/');
    key.extend_from_slice(image_id.as_bytes());
    key
}

/// `img/b/{image_id_v7_BE_BIN_16}` — bytes live under a flat prefix so we
/// can fetch them by id alone.
pub fn image_bytes_key(image_id: &Uuid) -> Vec<u8> {
    let mut key = Vec::with_capacity(IMG_BYTES_PREFIX.len() + 16);
    key.extend_from_slice(IMG_BYTES_PREFIX);
    key.extend_from_slice(image_id.as_bytes());
    key
}

/// `img/i/{image_id_v7_BE_BIN_16}` — global index pointing back at the
/// narrative_id, so a fetch-by-id can locate the metadata row in O(log N).
pub fn image_index_key(image_id: &Uuid) -> Vec<u8> {
    let mut key = Vec::with_capacity(IMG_INDEX_PREFIX.len() + 16);
    key.extend_from_slice(IMG_INDEX_PREFIX);
    key.extend_from_slice(image_id.as_bytes());
    key
}

/// Persist record + bytes + global id-index atomically (best-effort batch).
pub fn save_image(store: &dyn KVStore, record: &ImageRecord, bytes: &[u8]) -> Result<()> {
    let nid = record.narrative_id.as_deref().unwrap_or("");
    let meta_key = image_meta_key(nid, &record.id);
    let bytes_key = image_bytes_key(&record.id);
    let index_key = image_index_key(&record.id);
    let meta_bytes = serde_json::to_vec(record)
        .map_err(|e| TensaError::Internal(format!("serialize ImageRecord: {e}")))?;
    let nid_bytes = nid.as_bytes();
    store.batch_put(vec![
        (meta_key.as_slice(), meta_bytes.as_slice()),
        (bytes_key.as_slice(), bytes),
        (index_key.as_slice(), nid_bytes),
    ])?;
    Ok(())
}

/// Resolve `image_id` → `narrative_id` via the global index. Empty string
/// means "the image was saved without a narrative" (pre-narrative entities).
pub fn resolve_narrative_for_image(
    store: &dyn KVStore,
    image_id: &Uuid,
) -> Result<Option<String>> {
    Ok(store
        .get(&image_index_key(image_id))?
        .map(|b| String::from_utf8_lossy(&b).to_string()))
}

/// Convenience: load the metadata row for an image by id alone.
pub fn load_image_by_id(store: &dyn KVStore, image_id: &Uuid) -> Result<Option<ImageRecord>> {
    let Some(nid) = resolve_narrative_for_image(store, image_id)? else {
        return Ok(None);
    };
    load_image_meta(store, &nid, image_id)
}

/// Load a single record by narrative + id.
pub fn load_image_meta(
    store: &dyn KVStore,
    narrative_id: &str,
    image_id: &Uuid,
) -> Result<Option<ImageRecord>> {
    let key = image_meta_key(narrative_id, image_id);
    match store.get(&key)? {
        Some(bytes) => {
            let record: ImageRecord = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Internal(format!("deserialize ImageRecord: {e}")))?;
            Ok(Some(record))
        }
        None => Ok(None),
    }
}

/// Load raw bytes by image id alone.
pub fn load_image_bytes(store: &dyn KVStore, image_id: &Uuid) -> Result<Option<Vec<u8>>> {
    store.get(&image_bytes_key(image_id))
}

/// Delete record + bytes + index entry for an image.
pub fn delete_image(store: &dyn KVStore, narrative_id: &str, image_id: &Uuid) -> Result<()> {
    store.delete(&image_meta_key(narrative_id, image_id))?;
    store.delete(&image_bytes_key(image_id))?;
    store.delete(&image_index_key(image_id))?;
    Ok(())
}

/// List all image metadata records for a narrative.
pub fn list_narrative_images(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<ImageRecord>> {
    let mut prefix = Vec::with_capacity(IMG_META_PREFIX.len() + narrative_id.len() + 1);
    prefix.extend_from_slice(IMG_META_PREFIX);
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');

    let mut out = Vec::new();
    for (_, value) in store.prefix_scan(&prefix)? {
        match serde_json::from_slice::<ImageRecord>(&value) {
            Ok(record) => out.push(record),
            Err(e) => tracing::warn!("Skipping corrupt ImageRecord: {e}"),
        }
    }
    Ok(out)
}

/// List images attached to a single entity, sorted oldest → newest by created_at.
pub fn list_entity_images(
    store: &dyn KVStore,
    narrative_id: &str,
    entity_id: &Uuid,
) -> Result<Vec<ImageRecord>> {
    let mut all = list_narrative_images(store, narrative_id)?;
    all.retain(|r| r.entity_id == *entity_id);
    all.sort_by_key(|r| r.created_at);
    Ok(all)
}

/// Helper: walk every image in the store regardless of narrative. Used by
/// the archive exporter when no narrative filter is applied.
pub fn list_all_images(store: &dyn KVStore) -> Result<Vec<ImageRecord>> {
    let mut out = Vec::new();
    for (_, value) in store.prefix_scan(IMG_META_PREFIX)? {
        if let Ok(record) = serde_json::from_slice::<ImageRecord>(&value) {
            out.push(record);
        }
    }
    Ok(out)
}

/// Convenience for tests: count metadata rows.
pub fn count_images(store: &Arc<dyn KVStore>) -> usize {
    store
        .prefix_scan(IMG_META_PREFIX)
        .map(|rows| rows.len())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;

    use crate::images::types::ImageSource;

    fn fixture(entity: Uuid, narrative: &str) -> ImageRecord {
        ImageRecord {
            id: Uuid::now_v7(),
            entity_id: entity,
            narrative_id: Some(narrative.into()),
            mime: "image/png".into(),
            bytes_len: 4,
            caption: Some("test".into()),
            source: ImageSource::Upload,
            prompt: None,
            style: None,
            place: None,
            era: None,
            provider: None,
            model: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let store = MemoryStore::new();
        let entity = Uuid::now_v7();
        let record = fixture(entity, "n1");
        let bytes = vec![1u8, 2, 3, 4];

        save_image(&store, &record, &bytes).unwrap();

        let loaded = load_image_meta(&store, "n1", &record.id).unwrap().unwrap();
        assert_eq!(loaded.id, record.id);
        assert_eq!(loaded.bytes_len, 4);

        let loaded_bytes = load_image_bytes(&store, &record.id).unwrap().unwrap();
        assert_eq!(loaded_bytes, bytes);
    }

    #[test]
    fn test_list_filters_by_entity() {
        let store = MemoryStore::new();
        let entity_a = Uuid::now_v7();
        let entity_b = Uuid::now_v7();
        save_image(&store, &fixture(entity_a, "n1"), &[1]).unwrap();
        save_image(&store, &fixture(entity_a, "n1"), &[2]).unwrap();
        save_image(&store, &fixture(entity_b, "n1"), &[3]).unwrap();

        let only_a = list_entity_images(&store, "n1", &entity_a).unwrap();
        assert_eq!(only_a.len(), 2);
        let only_b = list_entity_images(&store, "n1", &entity_b).unwrap();
        assert_eq!(only_b.len(), 1);
    }

    #[test]
    fn test_delete_removes_both_keys() {
        let store = MemoryStore::new();
        let record = fixture(Uuid::now_v7(), "n1");
        save_image(&store, &record, &[9, 9]).unwrap();
        delete_image(&store, "n1", &record.id).unwrap();
        assert!(load_image_meta(&store, "n1", &record.id).unwrap().is_none());
        assert!(load_image_bytes(&store, &record.id).unwrap().is_none());
    }

    #[test]
    fn test_list_narrative_isolates_namespaces() {
        let store = MemoryStore::new();
        save_image(&store, &fixture(Uuid::now_v7(), "alpha"), &[1]).unwrap();
        save_image(&store, &fixture(Uuid::now_v7(), "beta"), &[2]).unwrap();
        save_image(&store, &fixture(Uuid::now_v7(), "alpha"), &[3]).unwrap();
        assert_eq!(list_narrative_images(&store, "alpha").unwrap().len(), 2);
        assert_eq!(list_narrative_images(&store, "beta").unwrap().len(), 1);
    }

    #[test]
    fn test_load_image_by_id_uses_global_index() {
        let store = MemoryStore::new();
        let entity = Uuid::now_v7();
        let record = fixture(entity, "alpha");
        save_image(&store, &record, &[7, 7, 7]).unwrap();

        let loaded = load_image_by_id(&store, &record.id).unwrap().unwrap();
        assert_eq!(loaded.id, record.id);
        assert_eq!(loaded.narrative_id.as_deref(), Some("alpha"));

        delete_image(&store, "alpha", &record.id).unwrap();
        assert!(load_image_by_id(&store, &record.id).unwrap().is_none());
        assert!(resolve_narrative_for_image(&store, &record.id)
            .unwrap()
            .is_none());
    }
}
