//! BM25 (Okapi BM25) keyword index for hybrid retrieval.
//!
//! Provides a lightweight inverted index that scores documents by term frequency
//! and inverse document frequency. Used alongside vector search in hybrid mode
//! to combine semantic similarity with keyword precision.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::store::KVStore;

/// KV prefix for persisted BM25 index.
const BM25_PREFIX: &str = "bm25/idx";

/// BM25 parameters.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// A lightweight BM25 inverted index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Index {
    /// Term → set of document UUIDs containing the term.
    inverted: HashMap<String, HashSet<Uuid>>,
    /// Document UUID → token count.
    doc_lengths: HashMap<Uuid, usize>,
    /// Total number of documents.
    doc_count: usize,
    /// Sum of all document lengths (for avgdl).
    total_length: usize,
}

/// A single BM25 search result.
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub id: Uuid,
    pub score: f64,
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl BM25Index {
    /// Create a new empty BM25 index.
    pub fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_count: 0,
            total_length: 0,
        }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, id: Uuid, text: &str) {
        let tokens = tokenize(text);
        let token_count = tokens.len();

        // Remove old entry if re-indexing
        if self.doc_lengths.contains_key(&id) {
            self.remove_document(&id);
        }

        self.doc_lengths.insert(id, token_count);
        self.doc_count += 1;
        self.total_length += token_count;

        for token in tokens {
            self.inverted.entry(token).or_default().insert(id);
        }
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, id: &Uuid) {
        if let Some(old_len) = self.doc_lengths.remove(id) {
            self.doc_count = self.doc_count.saturating_sub(1);
            self.total_length = self.total_length.saturating_sub(old_len);
            // Remove from all posting lists
            self.inverted.retain(|_, docs| {
                docs.remove(id);
                !docs.is_empty()
            });
        }
    }

    /// Search the index for documents matching the query, returning top-k results.
    pub fn search(&self, query: &str, k: usize) -> Vec<BM25Result> {
        if self.doc_count == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        let avgdl = self.total_length as f64 / self.doc_count.max(1) as f64;

        // Accumulate BM25 scores per document
        let mut scores: HashMap<Uuid, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(posting_list) = self.inverted.get(token) {
                let df = posting_list.len() as f64;
                let idf = ((self.doc_count as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &doc_id in posting_list {
                    let dl = *self.doc_lengths.get(&doc_id).unwrap_or(&1) as f64;
                    // Count term frequency in this document
                    let tf = self.term_frequency(token, &doc_id);
                    let tf_score = (tf * (BM25_K1 + 1.0))
                        / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl));
                    *scores.entry(doc_id).or_default() += idf * tf_score;
                }
            }
        }

        // Sort by score descending, take top k
        let mut results: Vec<BM25Result> = scores
            .into_iter()
            .map(|(id, score)| BM25Result { id, score })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Count how many times a term appears in a document.
    /// This is approximate — we only know set membership, not exact count.
    /// For BM25 accuracy, TF=1.0 for presence (boolean BM25 variant).
    fn term_frequency(&self, _token: &str, _doc_id: &Uuid) -> f64 {
        // Boolean model: tf = 1 if term is present
        1.0
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.doc_count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }

    /// Number of unique terms in the index.
    pub fn term_count(&self) -> usize {
        self.inverted.len()
    }

    /// Save the index to KV store.
    pub fn save(&self, store: &dyn KVStore) -> Result<()> {
        let data = serde_json::to_vec(self)?;
        store.put(BM25_PREFIX.as_bytes(), &data)?;
        Ok(())
    }

    /// Load the index from KV store.
    pub fn load(store: &dyn KVStore) -> Result<Option<Self>> {
        match store.get(BM25_PREFIX.as_bytes())? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }
}

/// Tokenize text into lowercase terms, stripping punctuation.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| s.len() >= 2) // skip single chars
        .map(String::from)
        .collect()
}

/// Merge vector search results with BM25 results using linear combination.
///
/// `alpha` controls the weight: `score = alpha * vector_score + (1 - alpha) * bm25_score`.
/// Both score sets are normalized to [0, 1] before merging.
pub fn merge_hybrid_results(
    vector_results: &[(Uuid, f32)],
    bm25_results: &[BM25Result],
    alpha: f32,
) -> Vec<(Uuid, f32)> {
    let mut combined: HashMap<Uuid, (f32, f32)> = HashMap::new();

    // Normalize vector scores
    let v_max = vector_results
        .iter()
        .map(|(_, s)| *s)
        .fold(0.0f32, f32::max);
    let v_max = if v_max > 0.0 { v_max } else { 1.0 };

    for &(id, score) in vector_results {
        combined.entry(id).or_default().0 = score / v_max;
    }

    // Normalize BM25 scores
    let b_max = bm25_results.iter().map(|r| r.score).fold(0.0f64, f64::max);
    let b_max = if b_max > 0.0 { b_max } else { 1.0 };

    for result in bm25_results {
        combined.entry(result.id).or_default().1 = (result.score / b_max) as f32;
    }

    // Compute weighted combination
    let mut merged: Vec<(Uuid, f32)> = combined
        .into_iter()
        .map(|(id, (v_score, b_score))| {
            let final_score = alpha * v_score + (1.0 - alpha) * b_score;
            (id, final_score)
        })
        .collect();

    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_empty_index() {
        let idx = BM25Index::new();
        assert!(idx.is_empty());
        assert_eq!(idx.search("hello world", 10).len(), 0);
    }

    #[test]
    fn test_bm25_add_and_search() {
        let mut idx = BM25Index::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        idx.add_document(id1, "The quick brown fox jumps over the lazy dog");
        idx.add_document(id2, "A fast brown fox leaps over a sleepy hound");
        idx.add_document(id3, "The weather is sunny and warm today");

        let results = idx.search("brown fox", 10);
        assert_eq!(results.len(), 2); // id1 and id2 match
                                      // Both should score > 0
        assert!(results[0].score > 0.0);
        assert!(results[1].score > 0.0);
    }

    #[test]
    fn test_bm25_ranking() {
        let mut idx = BM25Index::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        // id1 has more query terms
        idx.add_document(id1, "murder weapon found at crime scene evidence");
        idx.add_document(id2, "sunny day at the park with children playing");

        let results = idx.search("murder crime evidence", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, id1); // id1 should rank higher
    }

    #[test]
    fn test_bm25_remove_document() {
        let mut idx = BM25Index::new();
        let id1 = Uuid::now_v7();
        idx.add_document(id1, "test document with some words");
        assert_eq!(idx.len(), 1);

        idx.remove_document(&id1);
        assert_eq!(idx.len(), 0);
        assert!(idx.search("test", 10).is_empty());
    }

    #[test]
    fn test_bm25_reindex() {
        let mut idx = BM25Index::new();
        let id1 = Uuid::now_v7();
        idx.add_document(id1, "old content about cats");
        idx.add_document(id1, "new content about dogs");
        assert_eq!(idx.len(), 1); // should not duplicate

        let results = idx.search("dogs", 10);
        assert_eq!(results.len(), 1);
        let results = idx.search("cats", 10);
        assert_eq!(results.len(), 0); // old content removed
    }

    #[test]
    fn test_bm25_persistence() {
        use crate::store::memory::MemoryStore;
        use std::sync::Arc;

        let store = Arc::new(MemoryStore::new());
        let mut idx = BM25Index::new();
        let id1 = Uuid::now_v7();
        idx.add_document(id1, "persisted content for testing");
        idx.save(store.as_ref()).unwrap();

        let loaded = BM25Index::load(store.as_ref()).unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        let results = loaded.search("persisted testing", 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_bm25_no_match() {
        let mut idx = BM25Index::new();
        idx.add_document(Uuid::now_v7(), "alpha beta gamma delta");
        let results = idx.search("zebra", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! It's a test-case.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"it's".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single chars should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_merge_hybrid_results() {
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        let vector = vec![(id1, 0.9f32), (id2, 0.5f32)];
        let bm25 = vec![
            BM25Result {
                id: id2,
                score: 3.0,
            },
            BM25Result {
                id: id3,
                score: 2.0,
            },
        ];

        let merged = merge_hybrid_results(&vector, &bm25, 0.7);
        assert_eq!(merged.len(), 3); // all 3 unique IDs
                                     // id2 should score well (present in both)
        let id2_score = merged.iter().find(|(id, _)| *id == id2).unwrap().1;
        assert!(id2_score > 0.0);
    }

    #[test]
    fn test_merge_hybrid_empty() {
        let merged = merge_hybrid_results(&[], &[], 0.7);
        assert!(merged.is_empty());
    }
}
