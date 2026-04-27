//! In-memory vector index for nearest-neighbor search.
//!
//! Uses a brute-force approach suitable for moderate data sizes (< 100K).
//! When the `embedding` feature is enabled, this can be swapped for usearch.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::ingestion::embed::cosine_similarity;

/// A nearest-neighbor vector index.
pub struct VectorIndex {
    dimension: usize,
    vectors: HashMap<Uuid, Vec<f32>>,
}

/// A search result: (uuid, similarity_score).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: Uuid,
    pub score: f32,
}

impl VectorIndex {
    /// Create a new empty vector index.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: HashMap::new(),
        }
    }

    /// Return the configured dimensionality.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: Uuid, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dimension {
            return Err(TensaError::EmbeddingError(format!(
                "Expected dimension {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }
        self.vectors.insert(id, embedding.to_vec());
        Ok(())
    }

    /// Remove a vector from the index.
    pub fn remove(&mut self, id: &Uuid) -> bool {
        self.vectors.remove(id).is_some()
    }

    /// Search for the k nearest neighbors by cosine similarity.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(TensaError::EmbeddingError(format!(
                "Query dimension {} != index dimension {}",
                query.len(),
                self.dimension
            )));
        }

        let mut scored: Vec<SearchResult> = self
            .vectors
            .iter()
            .map(|(id, vec)| SearchResult {
                id: *id,
                score: cosine_similarity(query, vec),
            })
            .collect();

        // Sort by descending similarity
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        Ok(scored)
    }

    /// Persist the index to the KV store.
    pub fn save(&self, store: &dyn crate::store::KVStore) -> Result<()> {
        let data = serde_json::to_vec(&self.to_serializable())?;
        store.put(b"meta/vector_index", &data)?;
        Ok(())
    }

    /// Load the index from the KV store.
    pub fn load(store: &dyn crate::store::KVStore, dimension: usize) -> Result<Self> {
        match store.get(b"meta/vector_index")? {
            Some(data) => {
                let s: SerializableIndex = serde_json::from_slice(&data)?;
                Ok(Self::from_serializable(s, dimension))
            }
            None => Ok(Self::new(dimension)),
        }
    }

    fn to_serializable(&self) -> SerializableIndex {
        SerializableIndex {
            entries: self
                .vectors
                .iter()
                .map(|(id, vec)| (*id, vec.clone()))
                .collect(),
        }
    }

    fn from_serializable(s: SerializableIndex, dimension: usize) -> Self {
        let mut vectors = HashMap::new();
        for (id, vec) in s.entries {
            if vec.len() == dimension {
                vectors.insert(id, vec);
            }
        }
        Self { dimension, vectors }
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableIndex {
    entries: Vec<(Uuid, Vec<f32>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_vector_add_and_search() {
        let mut idx = VectorIndex::new(3);
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        idx.add(id1, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(id2, &[0.9, 0.1, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id1); // exact match first
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_vector_search_empty_index() {
        let idx = VectorIndex::new(3);
        let results = idx.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_vector_remove() {
        let mut idx = VectorIndex::new(3);
        let id = Uuid::now_v7();
        idx.add(id, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.remove(&id));
        assert_eq!(idx.len(), 0);
        assert!(!idx.remove(&id));
    }

    #[test]
    fn test_vector_search_k_limit() {
        let mut idx = VectorIndex::new(2);
        for _ in 0..10 {
            idx.add(Uuid::now_v7(), &[1.0, 0.0]).unwrap();
        }
        let results = idx.search(&[1.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_vector_wrong_dimension_add() {
        let mut idx = VectorIndex::new(3);
        assert!(idx.add(Uuid::now_v7(), &[1.0, 0.0]).is_err());
    }

    #[test]
    fn test_vector_wrong_dimension_search() {
        let idx = VectorIndex::new(3);
        assert!(idx.search(&[1.0], 5).is_err());
    }

    #[test]
    fn test_vector_save_and_load() {
        let store = Arc::new(MemoryStore::new());
        let mut idx = VectorIndex::new(3);
        let id = Uuid::now_v7();
        idx.add(id, &[1.0, 0.5, 0.2]).unwrap();
        idx.save(store.as_ref()).unwrap();

        let loaded = VectorIndex::load(store.as_ref(), 3).unwrap();
        assert_eq!(loaded.len(), 1);
        let results = loaded.search(&[1.0, 0.5, 0.2], 1).unwrap();
        assert_eq!(results[0].id, id);
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vector_load_empty_store() {
        let store = Arc::new(MemoryStore::new());
        let loaded = VectorIndex::load(store.as_ref(), 3).unwrap();
        assert!(loaded.is_empty());
    }
}
