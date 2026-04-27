//! Embedding model abstraction for generating vector representations.
//!
//! Provides a trait-based interface so callers are decoupled from the
//! specific embedding implementation (ONNX, Python sidecar, etc.).

use crate::error::{Result, TensaError};

#[cfg(feature = "embedding")]
pub use crate::ingestion::onnx_embedder::OnnxEmbedder;

/// Trait for generating text embeddings.
pub trait EmbeddingProvider: Send + Sync {
    /// Return the dimensionality of produced embeddings.
    fn dimension(&self) -> usize;

    /// Human-readable provider name (e.g. "hash", "onnx").
    fn provider_name(&self) -> &'static str {
        "unknown"
    }

    /// Embed a single text string.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of texts (default: call embed_text in a loop).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_text(t)).collect()
    }
}

/// L2-normalize a vector in place.
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// Cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// A simple in-memory embedding provider that hashes text to a
/// deterministic pseudo-embedding. Useful for tests and development
/// when no real model is available.
pub struct HashEmbedding {
    dimension: usize,
}

impl HashEmbedding {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingProvider for HashEmbedding {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn provider_name(&self) -> &'static str {
        "hash"
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(TensaError::EmbeddingError("Cannot embed empty text".into()));
        }
        // Deterministic pseudo-embedding from text hash
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();
        for (i, slot) in embedding.iter_mut().enumerate() {
            let mut h: u32 = 0x811c_9dc5; // FNV offset basis
            for &b in bytes {
                h ^= b as u32;
                h = h.wrapping_mul(0x0100_0193); // FNV prime
            }
            h ^= i as u32;
            h = h.wrapping_mul(0x0100_0193);
            *slot = (h as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        // L2 normalize
        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_embed_correct_dimension() {
        let model = HashEmbedding::new(384);
        let emb = model.embed_text("Hello world").unwrap();
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn test_hash_embed_deterministic() {
        let model = HashEmbedding::new(128);
        let e1 = model.embed_text("Test text").unwrap();
        let e2 = model.embed_text("Test text").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_hash_embed_normalized() {
        let model = HashEmbedding::new(256);
        let emb = model.embed_text("Normalize me").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hash_embed_empty_text_error() {
        let model = HashEmbedding::new(128);
        assert!(model.embed_text("").is_err());
    }

    #[test]
    fn test_embed_batch() {
        let model = HashEmbedding::new(64);
        let texts = vec!["Hello", "World"];
        let results = model.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 64);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
