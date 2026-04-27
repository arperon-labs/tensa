//! Document status tracking for ingestion deduplication.
//!
//! Tracks which documents have been ingested, keyed by content hash,
//! using the `ds/` KV prefix.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::ingestion::hex_encode;
use crate::store::KVStore;

const PREFIX: &[u8] = b"ds/";

/// Status record for an ingested document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatus {
    /// SHA-256 hex hash of the document text.
    pub source_hash: String,
    /// User-provided name/identifier for the source.
    pub source_id: String,
    /// Narrative the document was ingested into.
    pub narrative_id: Option<String>,
    /// When the document was ingested.
    pub ingested_at: DateTime<Utc>,
    /// Number of text chunks produced.
    pub chunk_count: usize,
    /// Number of entities extracted.
    pub entity_count: usize,
    /// Number of situations extracted.
    pub situation_count: usize,
    /// Ingestion job identifier.
    pub job_id: String,
}

/// KV-backed tracker for ingested document status.
pub struct DocStatusTracker {
    store: Arc<dyn KVStore>,
}

impl DocStatusTracker {
    /// Create a new tracker backed by the given KV store.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Compute the SHA-256 hex hash of document text.
    pub fn content_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hex_encode(&hasher.finalize())
    }

    fn key_for_hash(hash: &str) -> Vec<u8> {
        let mut key = Vec::with_capacity(3 + hash.len());
        key.extend_from_slice(PREFIX);
        key.extend_from_slice(hash.as_bytes());
        key
    }

    /// Record a document status entry.
    pub fn record(&self, status: &DocumentStatus) -> Result<()> {
        let key = Self::key_for_hash(&status.source_hash);
        let val = serde_json::to_vec(status)?;
        self.store.put(&key, &val)?;
        Ok(())
    }

    /// Check if text has already been ingested. Returns the status if found.
    pub fn is_ingested(&self, text: &str) -> Result<Option<DocumentStatus>> {
        let hash = Self::content_hash(text);
        self.get_by_hash(&hash)
    }

    /// Look up a document status by its content hash directly.
    pub fn get_by_hash(&self, hash: &str) -> Result<Option<DocumentStatus>> {
        let key = Self::key_for_hash(hash);
        match self.store.get(&key)? {
            Some(bytes) => {
                let status: DocumentStatus = serde_json::from_slice(&bytes)?;
                Ok(Some(status))
            }
            None => Ok(None),
        }
    }

    /// List all recorded document statuses.
    pub fn list_all(&self) -> Result<Vec<DocumentStatus>> {
        let entries = self.store.prefix_scan(PREFIX)?;
        let mut statuses = Vec::with_capacity(entries.len());
        for (_, val) in entries {
            let status: DocumentStatus = serde_json::from_slice(&val)?;
            statuses.push(status);
        }
        Ok(statuses)
    }

    /// Remove a document status record by content hash.
    pub fn remove(&self, source_hash: &str) -> Result<()> {
        let key = Self::key_for_hash(source_hash);
        self.store.delete(&key)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn make_status(text: &str, source_id: &str) -> DocumentStatus {
        DocumentStatus {
            source_hash: DocStatusTracker::content_hash(text),
            source_id: source_id.to_string(),
            narrative_id: Some("test-narrative".into()),
            ingested_at: Utc::now(),
            chunk_count: 3,
            entity_count: 5,
            situation_count: 2,
            job_id: "job-001".into(),
        }
    }

    #[test]
    fn test_record_and_lookup() {
        let store = Arc::new(MemoryStore::new());
        let tracker = DocStatusTracker::new(store);
        let status = make_status("Hello world", "doc1");
        tracker.record(&status).unwrap();
        let found = tracker.is_ingested("Hello world").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().source_id, "doc1");
    }

    #[test]
    fn test_not_ingested() {
        let store = Arc::new(MemoryStore::new());
        let tracker = DocStatusTracker::new(store);
        let found = tracker.is_ingested("unknown text").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_list_all() {
        let store = Arc::new(MemoryStore::new());
        let tracker = DocStatusTracker::new(store);
        tracker.record(&make_status("text1", "doc1")).unwrap();
        tracker.record(&make_status("text2", "doc2")).unwrap();
        tracker.record(&make_status("text3", "doc3")).unwrap();
        let all = tracker.list_all().unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_remove() {
        let store = Arc::new(MemoryStore::new());
        let tracker = DocStatusTracker::new(store);
        let status = make_status("removable", "doc-x");
        tracker.record(&status).unwrap();
        assert!(tracker.is_ingested("removable").unwrap().is_some());
        tracker.remove(&status.source_hash).unwrap();
        assert!(tracker.is_ingested("removable").unwrap().is_none());
    }
}
