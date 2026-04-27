//! Validation queue for human-in-the-loop review.
//!
//! Items that fall within the review confidence band are stored here
//! for human approval, editing, or rejection before being committed
//! to the hypergraph.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;

/// Type of item in the validation queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueItemType {
    Entity,
    Situation,
    Participation,
    CausalLink,
}

/// Status of a validation queue item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueItemStatus {
    Pending,
    Approved,
    Edited,
    Rejected,
}

/// An item in the validation queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationQueueItem {
    pub id: Uuid,
    pub item_type: QueueItemType,
    pub extracted_data: serde_json::Value,
    pub source_text: String,
    pub chunk_id: usize,
    /// UUID of the source chunk record in the ChunkStore.
    #[serde(default)]
    pub source_chunk_id: Option<Uuid>,
    /// Narrative this queued extraction belongs to. Set from
    /// `IngestionConfig.narrative_id` at enqueue time so the Studio
    /// validation queue can filter by the active narrative without
    /// having to peek inside `extracted_data`.
    #[serde(default)]
    pub narrative_id: Option<String>,
    pub confidence: f32,
    pub status: QueueItemStatus,
    pub reviewer_notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub reviewed_at: Option<DateTime<Utc>>,
}

/// KV-backed validation queue.
pub struct ValidationQueue {
    store: Arc<dyn KVStore>,
}

impl ValidationQueue {
    /// Create a new validation queue backed by the given store.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Enqueue an item for review.
    pub fn enqueue(&self, item: ValidationQueueItem) -> Result<Uuid> {
        let id = item.id;
        let key = queue_item_key(&id);
        let value = serde_json::to_vec(&item)?;
        self.store.put(&key, &value)?;

        // Also store in the pending index for efficient listing
        let pending_key = pending_index_key(&item.created_at, &id);
        self.store.put(&pending_key, id.as_bytes())?;

        Ok(id)
    }

    /// Get a queue item by ID.
    pub fn get(&self, id: &Uuid) -> Result<ValidationQueueItem> {
        let key = queue_item_key(id);
        match self.store.get(&key)? {
            Some(data) => Ok(serde_json::from_slice(&data)?),
            None => Err(TensaError::ValidationQueueError(format!(
                "Queue item not found: {}",
                id
            ))),
        }
    }

    /// List pending items, ordered by creation time (oldest first).
    pub fn list_pending(&self, limit: usize) -> Result<Vec<ValidationQueueItem>> {
        let prefix = b"vq/p/".to_vec();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut items = Vec::new();

        for (_key, id_bytes) in pairs {
            if items.len() >= limit {
                break;
            }
            if id_bytes.len() == 16 {
                let id = Uuid::from_bytes(
                    id_bytes
                        .try_into()
                        .map_err(|_| TensaError::ValidationQueueError("Invalid UUID".into()))?,
                );
                match self.get(&id) {
                    Ok(item) if item.status == QueueItemStatus::Pending => {
                        items.push(item);
                    }
                    _ => continue,
                }
            }
        }

        Ok(items)
    }

    /// Approve a queue item and update its status.
    pub fn approve(&self, id: &Uuid, reviewer: &str) -> Result<ValidationQueueItem> {
        let mut item = self.get(id)?;
        if item.status != QueueItemStatus::Pending {
            return Err(TensaError::ValidationQueueError(format!(
                "Item {} is not pending (status: {:?})",
                id, item.status
            )));
        }
        item.status = QueueItemStatus::Approved;
        item.reviewer_notes = Some(format!("Approved by {}", reviewer));
        item.reviewed_at = Some(Utc::now());
        self.update_item(&item)?;
        Ok(item)
    }

    /// Reject a queue item.
    pub fn reject(
        &self,
        id: &Uuid,
        reviewer: &str,
        notes: Option<String>,
    ) -> Result<ValidationQueueItem> {
        let mut item = self.get(id)?;
        if item.status != QueueItemStatus::Pending {
            return Err(TensaError::ValidationQueueError(format!(
                "Item {} is not pending (status: {:?})",
                id, item.status
            )));
        }
        item.status = QueueItemStatus::Rejected;
        item.reviewer_notes = Some(notes.unwrap_or_else(|| format!("Rejected by {}", reviewer)));
        item.reviewed_at = Some(Utc::now());
        self.update_item(&item)?;
        Ok(item)
    }

    /// Edit the extracted data and approve.
    pub fn edit_and_approve(
        &self,
        id: &Uuid,
        reviewer: &str,
        edited_data: serde_json::Value,
    ) -> Result<ValidationQueueItem> {
        let mut item = self.get(id)?;
        if item.status != QueueItemStatus::Pending {
            return Err(TensaError::ValidationQueueError(format!(
                "Item {} is not pending (status: {:?})",
                id, item.status
            )));
        }
        item.extracted_data = edited_data;
        item.status = QueueItemStatus::Edited;
        item.reviewer_notes = Some(format!("Edited and approved by {}", reviewer));
        item.reviewed_at = Some(Utc::now());
        self.update_item(&item)?;
        Ok(item)
    }

    fn update_item(&self, item: &ValidationQueueItem) -> Result<()> {
        let key = queue_item_key(&item.id);
        let value = serde_json::to_vec(item)?;
        self.store.put(&key, &value)?;
        Ok(())
    }
}

/// Build the primary key for a queue item: vq/i/{uuid}
fn queue_item_key(id: &Uuid) -> Vec<u8> {
    let mut key = b"vq/i/".to_vec();
    key.extend_from_slice(id.as_bytes());
    key
}

/// Build the pending index key: vq/p/{timestamp_be}/{uuid}
fn pending_index_key(created_at: &DateTime<Utc>, id: &Uuid) -> Vec<u8> {
    let mut key = b"vq/p/".to_vec();
    key.extend_from_slice(&created_at.timestamp_millis().to_be_bytes());
    key.push(b'/');
    key.extend_from_slice(id.as_bytes());
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn setup() -> ValidationQueue {
        let store = Arc::new(MemoryStore::new());
        ValidationQueue::new(store)
    }

    fn make_item(confidence: f32) -> ValidationQueueItem {
        ValidationQueueItem {
            id: Uuid::now_v7(),
            item_type: QueueItemType::Entity,
            extracted_data: serde_json::json!({"name": "Test"}),
            source_text: "Test text".to_string(),
            chunk_id: 0,
            source_chunk_id: None,
            narrative_id: None,
            confidence,
            status: QueueItemStatus::Pending,
            reviewer_notes: None,
            created_at: Utc::now(),
            reviewed_at: None,
        }
    }

    #[test]
    fn test_enqueue_and_get() {
        let queue = setup();
        let item = make_item(0.5);
        let id = item.id;
        queue.enqueue(item).unwrap();
        let retrieved = queue.get(&id).unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.status, QueueItemStatus::Pending);
    }

    #[test]
    fn test_list_pending() {
        let queue = setup();
        queue.enqueue(make_item(0.5)).unwrap();
        queue.enqueue(make_item(0.6)).unwrap();
        queue.enqueue(make_item(0.7)).unwrap();

        let pending = queue.list_pending(10).unwrap();
        assert_eq!(pending.len(), 3);
    }

    #[test]
    fn test_list_pending_limit() {
        let queue = setup();
        for _ in 0..5 {
            queue.enqueue(make_item(0.5)).unwrap();
        }
        let pending = queue.list_pending(2).unwrap();
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_approve() {
        let queue = setup();
        let item = make_item(0.5);
        let id = item.id;
        queue.enqueue(item).unwrap();

        let approved = queue.approve(&id, "reviewer1").unwrap();
        assert_eq!(approved.status, QueueItemStatus::Approved);
        assert!(approved.reviewer_notes.unwrap().contains("reviewer1"));
        assert!(approved.reviewed_at.is_some());
    }

    #[test]
    fn test_reject() {
        let queue = setup();
        let item = make_item(0.5);
        let id = item.id;
        queue.enqueue(item).unwrap();

        let rejected = queue
            .reject(&id, "reviewer1", Some("Low quality".into()))
            .unwrap();
        assert_eq!(rejected.status, QueueItemStatus::Rejected);
        assert!(rejected.reviewer_notes.unwrap().contains("Low quality"));
    }

    #[test]
    fn test_edit_and_approve() {
        let queue = setup();
        let item = make_item(0.5);
        let id = item.id;
        queue.enqueue(item).unwrap();

        let edited = queue
            .edit_and_approve(&id, "reviewer1", serde_json::json!({"name": "Fixed"}))
            .unwrap();
        assert_eq!(edited.status, QueueItemStatus::Edited);
        assert_eq!(edited.extracted_data["name"], "Fixed");
    }

    #[test]
    fn test_approve_nonexistent_fails() {
        let queue = setup();
        let result = queue.approve(&Uuid::now_v7(), "reviewer1");
        assert!(result.is_err());
    }

    #[test]
    fn test_approve_already_approved_fails() {
        let queue = setup();
        let item = make_item(0.5);
        let id = item.id;
        queue.enqueue(item).unwrap();
        queue.approve(&id, "reviewer1").unwrap();
        // Second approve should fail
        assert!(queue.approve(&id, "reviewer2").is_err());
    }

    #[test]
    fn test_list_pending_excludes_approved() {
        let queue = setup();
        let item1 = make_item(0.5);
        let id1 = item1.id;
        queue.enqueue(item1).unwrap();
        queue.enqueue(make_item(0.6)).unwrap();

        queue.approve(&id1, "reviewer1").unwrap();

        let pending = queue.list_pending(10).unwrap();
        assert_eq!(pending.len(), 1);
    }
}
