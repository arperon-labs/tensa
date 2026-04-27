//! Style encoder architecture and inference (Sprint D9.5).
//!
//! Contrastive learning: same-author chunks → similar embeddings,
//! different-author chunks → distant embeddings. Uses triplet loss / InfoNCE.
//!
//! Training requires ONNX/candle infrastructure (separate binary).
//! This module provides the inference-time encoding trait and a hash-based
//! fallback encoder for testing.

use serde::{Deserialize, Serialize};

use crate::error::Result;

// ─── Encoder Trait ──────────────────────────────────────────

/// Trait for style encoders that convert text chunks to dense style vectors.
pub trait StyleEncoder: Send + Sync {
    /// Encode a text chunk into a style embedding vector.
    fn encode(&self, text: &str) -> Result<Vec<f32>>;

    /// Output dimension of the encoder.
    fn dim(&self) -> usize;

    /// Name of the encoder model.
    fn model_name(&self) -> &str;
}

// ─── Hash-Based Fallback Encoder ────────────────────────────

/// Deterministic hash-based encoder for testing.
///
/// Produces consistent but non-semantic embeddings from text.
/// Useful for integration testing without requiring a trained model.
#[derive(Debug, Clone)]
pub struct HashStyleEncoder {
    dim: usize,
}

impl HashStyleEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl StyleEncoder for HashStyleEncoder {
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0f32; self.dim];

        // Simple character-level hash features
        let bytes = text.as_bytes();
        for (i, chunk) in bytes.chunks(4).enumerate() {
            let idx = i % self.dim;
            let val: u32 = chunk
                .iter()
                .fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
            vector[idx] += (val as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        // Additional text-level features in early dimensions
        let word_count = text.split_whitespace().count();
        let avg_word_len = if word_count > 0 {
            text.split_whitespace().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32
        } else {
            0.0
        };
        let sentence_count = text.matches('.').count().max(1);
        let avg_sentence_len = word_count as f32 / sentence_count as f32;

        if self.dim > 3 {
            vector[0] = avg_word_len / 10.0;
            vector[1] = avg_sentence_len / 30.0;
            vector[2] = (text.matches('"').count() as f32) / (word_count as f32 + 1.0);
            vector[3] = (text.matches(',').count() as f32) / (word_count as f32 + 1.0);
        }

        // L2 normalize
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in &mut vector {
                *v /= norm;
            }
        }

        Ok(vector)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "hash-style-encoder"
    }
}

/// Training configuration for the contrastive style encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderTrainingConfig {
    /// Base transformer model to fine-tune.
    pub base_model: String,
    /// Output embedding dimension.
    pub output_dim: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Triplet loss margin.
    pub margin: f64,
    /// Minimum chunk size in tokens.
    pub min_chunk_tokens: usize,
    /// Maximum chunk size in tokens.
    pub max_chunk_tokens: usize,
    /// Minimum authors in training corpus.
    pub min_authors: usize,
    /// Minimum words per author.
    pub min_words_per_author: usize,
}

impl Default for EncoderTrainingConfig {
    fn default() -> Self {
        Self {
            base_model: "BAAI/bge-m3".into(),
            output_dim: 256,
            epochs: 10,
            batch_size: 32,
            learning_rate: 2e-5,
            margin: 0.5,
            min_chunk_tokens: 512,
            max_chunk_tokens: 2048,
            min_authors: 50,
            min_words_per_author: 50_000,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_encoder_deterministic() {
        let encoder = HashStyleEncoder::new(64);
        let v1 = encoder.encode("The quick brown fox").unwrap();
        let v2 = encoder.encode("The quick brown fox").unwrap();
        assert_eq!(v1, v2, "Same input should produce same output");
    }

    #[test]
    fn test_hash_encoder_different_inputs() {
        let encoder = HashStyleEncoder::new(64);
        let v1 = encoder
            .encode("The quick brown fox jumps over the lazy dog")
            .unwrap();
        let v2 = encoder
            .encode("A completely different text about something else entirely")
            .unwrap();
        // Different inputs should produce different vectors
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_hash_encoder_normalized() {
        let encoder = HashStyleEncoder::new(64);
        let v = encoder
            .encode("Some text for normalization testing")
            .unwrap();
        let norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Vector should be L2-normalized, norm={}",
            norm
        );
    }

    #[test]
    fn test_hash_encoder_dimension() {
        let encoder = HashStyleEncoder::new(256);
        let v = encoder.encode("Test").unwrap();
        assert_eq!(v.len(), 256);
    }

    #[test]
    fn test_default_training_config() {
        let config = EncoderTrainingConfig::default();
        assert_eq!(config.output_dim, 256);
        assert_eq!(config.min_authors, 50);
    }
}
