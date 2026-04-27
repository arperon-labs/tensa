//! Pluggable vector storage trait.
//!
//! Defines an abstraction over vector storage backends so callers
//! can swap between in-memory brute-force, external vector DBs, etc.
//! The default implementation wraps `VectorIndex` behind a `RwLock`.

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::ingestion::vector::{SearchResult, VectorIndex};
use crate::store::KVStore;

/// Trait for pluggable vector storage backends.
///
/// All methods take `&self` (not `&mut self`) so implementations can be
/// shared behind `Arc` for concurrent access. Mutable backends should
/// use internal locking (e.g., `RwLock`).
pub trait VectorStore: Send + Sync {
    /// Add a vector with the given ID.
    fn add(&self, id: Uuid, embedding: &[f32]) -> Result<()>;

    /// Remove a vector by ID. Returns true if it existed.
    fn remove(&self, id: &Uuid) -> Result<bool>;

    /// Search for the k nearest neighbors by cosine similarity.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Number of vectors stored.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist the index to durable storage (if supported).
    fn save(&self, store: &dyn KVStore) -> Result<()>;
}

/// In-memory vector store wrapping `VectorIndex` behind a `RwLock`.
///
/// This is the default implementation used when no external vector DB
/// is configured. Thread-safe for concurrent reads and writes.
pub struct InMemoryVectorStore {
    inner: std::sync::RwLock<VectorIndex>,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store with the given dimensionality.
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: std::sync::RwLock::new(VectorIndex::new(dimension)),
        }
    }

    /// Load from KV store, falling back to empty index on miss.
    pub fn load(store: &dyn KVStore, dimension: usize) -> Result<Self> {
        let idx = VectorIndex::load(store, dimension)?;
        Ok(Self {
            inner: std::sync::RwLock::new(idx),
        })
    }

    /// Return the configured dimensionality.
    pub fn dimension(&self) -> usize {
        self.inner.read().map(|v| v.dimension()).unwrap_or(0)
    }

    /// Get a read lock on the underlying VectorIndex.
    ///
    /// Useful for callers that need direct access (e.g., existing code
    /// that operates on `&VectorIndex`).
    pub fn read_inner(&self) -> Result<std::sync::RwLockReadGuard<'_, VectorIndex>> {
        self.inner
            .read()
            .map_err(|_| TensaError::Internal("VectorStore lock poisoned".into()))
    }

    /// Get a write lock on the underlying VectorIndex.
    pub fn write_inner(&self) -> Result<std::sync::RwLockWriteGuard<'_, VectorIndex>> {
        self.inner
            .write()
            .map_err(|_| TensaError::Internal("VectorStore lock poisoned".into()))
    }
}

impl VectorStore for InMemoryVectorStore {
    fn add(&self, id: Uuid, embedding: &[f32]) -> Result<()> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| TensaError::Internal("VectorStore lock poisoned".into()))?;
        guard.add(id, embedding)
    }

    fn remove(&self, id: &Uuid) -> Result<bool> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| TensaError::Internal("VectorStore lock poisoned".into()))?;
        Ok(guard.remove(id))
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let guard = self
            .inner
            .read()
            .map_err(|_| TensaError::Internal("VectorStore lock poisoned".into()))?;
        guard.search(query, k)
    }

    fn len(&self) -> usize {
        self.inner.read().map(|v| v.len()).unwrap_or(0)
    }

    fn save(&self, store: &dyn KVStore) -> Result<()> {
        let guard = self
            .inner
            .read()
            .map_err(|_| TensaError::Internal("VectorStore lock poisoned".into()))?;
        guard.save(store)
    }
}

// ─── Vector Store Configuration ──────────────────────────────

/// Configuration for selecting and connecting to a vector store backend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "backend")]
pub enum VectorStoreConfig {
    /// In-memory brute-force search (default, no external dependencies).
    #[serde(rename = "memory")]
    InMemory {
        /// Embedding dimension (e.g. 384 for all-MiniLM-L6-v2).
        dimension: usize,
    },
    /// Qdrant vector database (requires `qdrant` feature and running Qdrant server).
    #[serde(rename = "qdrant")]
    Qdrant {
        /// Qdrant server URL (e.g. "http://localhost:6334").
        url: String,
        /// Collection name.
        collection: String,
        /// Embedding dimension.
        dimension: usize,
    },
    /// PostgreSQL with pgvector extension (requires `pgvector` feature).
    #[serde(rename = "pgvector")]
    PgVector {
        /// PostgreSQL connection string.
        connection_string: String,
        /// Table name for vector storage.
        table: String,
        /// Embedding dimension.
        dimension: usize,
    },
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self::InMemory { dimension: 384 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_in_memory_add_search() {
        let store = InMemoryVectorStore::new(3);
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        store.add(id1, &[1.0, 0.0, 0.0]).unwrap();
        store.add(id2, &[0.9, 0.1, 0.0]).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_in_memory_remove() {
        let store = InMemoryVectorStore::new(3);
        let id = Uuid::now_v7();
        store.add(id, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(store.len(), 1);

        let removed = store.remove(&id).unwrap();
        assert!(removed);
        assert_eq!(store.len(), 0);

        // Search should return nothing
        let results = store.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_in_memory_len() {
        let store = InMemoryVectorStore::new(3);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.add(Uuid::now_v7(), &[1.0, 0.0, 0.0]).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_in_memory_save_and_load() {
        let kv = Arc::new(MemoryStore::new());
        let store = InMemoryVectorStore::new(3);
        let id = Uuid::now_v7();
        store.add(id, &[1.0, 0.5, 0.2]).unwrap();
        store.save(kv.as_ref()).unwrap();

        let loaded = InMemoryVectorStore::load(kv.as_ref(), 3).unwrap();
        assert_eq!(loaded.len(), 1);
        let results = loaded.search(&[1.0, 0.5, 0.2], 1).unwrap();
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn test_in_memory_dimension() {
        let store = InMemoryVectorStore::new(128);
        assert_eq!(store.dimension(), 128);
    }

    #[test]
    fn test_vector_store_config_default() {
        let config = VectorStoreConfig::default();
        matches!(config, VectorStoreConfig::InMemory { dimension: 384 });
    }

    #[test]
    fn test_vector_store_config_serde_memory() {
        let config = VectorStoreConfig::InMemory { dimension: 384 };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("memory"));
        let parsed: VectorStoreConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            VectorStoreConfig::InMemory { dimension } => assert_eq!(dimension, 384),
            _ => panic!("Expected InMemory"),
        }
    }

    #[test]
    fn test_vector_store_config_serde_qdrant() {
        let json = r#"{"backend":"qdrant","url":"http://localhost:6334","collection":"test","dimension":384}"#;
        let parsed: VectorStoreConfig = serde_json::from_str(json).unwrap();
        match parsed {
            VectorStoreConfig::Qdrant {
                url,
                collection,
                dimension,
            } => {
                assert_eq!(url, "http://localhost:6334");
                assert_eq!(collection, "test");
                assert_eq!(dimension, 384);
            }
            _ => panic!("Expected Qdrant"),
        }
    }

    #[test]
    fn test_vector_store_config_serde_pgvector() {
        let json = r#"{"backend":"pgvector","connection_string":"postgres://localhost/tensa","table":"vectors","dimension":768}"#;
        let parsed: VectorStoreConfig = serde_json::from_str(json).unwrap();
        match parsed {
            VectorStoreConfig::PgVector {
                connection_string,
                table,
                dimension,
            } => {
                assert_eq!(connection_string, "postgres://localhost/tensa");
                assert_eq!(table, "vectors");
                assert_eq!(dimension, 768);
            }
            _ => panic!("Expected PgVector"),
        }
    }

    #[test]
    fn test_in_memory_wrong_dimension() {
        let store = InMemoryVectorStore::new(3);
        assert!(store.add(Uuid::now_v7(), &[1.0, 0.0]).is_err());
    }
}
