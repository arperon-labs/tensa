//! Reranker trait and built-in implementations.
//!
//! Rerankers take a query and a set of retrieved documents and produce
//! a relevance-ordered list of `(original_index, score)` pairs.
//!
//! Built-in implementations:
//! - `NoopReranker`: preserves original order
//! - `TermOverlapReranker`: keyword overlap scoring (optionally
//!   aggregated under a configurable t-norm / t-conorm — default Gödel
//!   preserves the pre-sprint "fraction of matched terms" scoring)
//! - `CrossEncoderReranker`: ONNX cross-encoder model (feature-gated `embedding`)
//!
//! Cites: [klement2000].

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::fuzzy::tnorm::TNormKind;

/// Trait for reranking retrieved documents by relevance to a query.
pub trait Reranker: Send + Sync {
    /// Rerank documents. Returns `(original_index, score)` pairs sorted by
    /// descending score.
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>>;
}

/// No-op reranker that preserves original order with uniform scores.
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        Ok((0..documents.len()).map(|i| (i, 1.0)).collect())
    }
}

/// Simple TF-IDF-like reranker using term overlap scoring.
///
/// Counts how many query terms (length > 2) appear in each document and
/// produces a score in `[0.0, 1.0]` representing the fraction of query
/// terms found.
///
/// Accepts an optional `score_fusion` t-norm slot. `None` (default)
/// preserves the pre-sprint numerics exactly — the per-term hits are
/// summed and divided by total term count. When set, per-term match
/// indicators are instead folded under the chosen t-conorm (for an
/// OR-style score) and the final score is clamped to `[0, 1]`.
#[derive(Default)]
pub struct TermOverlapReranker {
    /// Optional t-conorm override for score fusion. `None` preserves the
    /// pre-sprint fraction-of-terms score; `Some(kind)` folds per-term
    /// 0/1 indicators under that t-conorm.
    pub score_fusion: Option<TNormKind>,
}

impl TermOverlapReranker {
    /// Default constructor — fraction-of-terms scoring (pre-sprint).
    pub fn new() -> Self {
        Self { score_fusion: None }
    }

    /// Constructor with an explicit score-fusion t-conorm override.
    pub fn with_fusion(kind: TNormKind) -> Self {
        Self {
            score_fusion: Some(kind),
        }
    }
}

impl Reranker for TermOverlapReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|s| s.to_string())
            .collect();

        if query_terms.is_empty() {
            return Ok((0..documents.len()).map(|i| (i, 1.0)).collect());
        }

        let mut scored: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let doc_lower = doc.to_lowercase();
                let hits: Vec<bool> = query_terms
                    .iter()
                    .map(|t| doc_lower.contains(t.as_str()))
                    .collect();

                let score = match self.score_fusion {
                    None => {
                        let matches = hits.iter().filter(|h| **h).count();
                        matches as f32 / query_terms.len() as f32
                    }
                    Some(kind) => {
                        let xs: Vec<f64> =
                            hits.iter().map(|h| if *h { 1.0 } else { 0.0 }).collect();
                        crate::fuzzy::tnorm::reduce_tconorm(kind, &xs) as f32
                    }
                };
                (i, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

/// Reranker type selector for configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RerankerType {
    /// No reranking.
    None,
    /// Term overlap scoring (default).
    #[default]
    TermOverlap,
    /// ONNX cross-encoder model (requires `embedding` feature + model file).
    CrossEncoder,
}

/// ONNX-based cross-encoder reranker.
///
/// Scores query-document pairs jointly through a cross-encoder model,
/// producing much more accurate relevance scores than bi-encoder similarity.
/// Feature-gated behind `embedding`.
///
/// The model expects `[CLS] query [SEP] document [SEP]` input and produces
/// a single logit score per pair.
#[cfg(feature = "embedding")]
pub struct CrossEncoderReranker {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
}

#[cfg(feature = "embedding")]
impl CrossEncoderReranker {
    /// Load cross-encoder model from a directory containing `model.onnx` and `tokenizer.json`.
    pub fn from_directory(dir: &str) -> Result<Self> {
        use crate::error::TensaError;
        use ort::session::builder::GraphOptimizationLevel;

        let dir_path = std::path::Path::new(dir);
        let model_path = dir_path.join("model.onnx");
        let tokenizer_path = dir_path.join("tokenizer.json");

        if !model_path.exists() {
            return Err(TensaError::EmbeddingError(format!(
                "Cross-encoder model not found: {}",
                model_path.display()
            )));
        }

        let model_str = model_path
            .to_str()
            .ok_or_else(|| TensaError::EmbeddingError("Non-UTF-8 model path".to_string()))?;
        let tokenizer_str = tokenizer_path
            .to_str()
            .ok_or_else(|| TensaError::EmbeddingError("Non-UTF-8 tokenizer path".to_string()))?;

        let session = ort::session::Session::builder()
            .map_err(|e| TensaError::EmbeddingError(format!("ONNX session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TensaError::EmbeddingError(format!("ONNX optimization: {e}")))?
            .commit_from_file(model_str)
            .map_err(|e| TensaError::EmbeddingError(format!("ONNX model load: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_str)
            .map_err(|e| TensaError::EmbeddingError(format!("Tokenizer load: {e}")))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            max_length: 512,
        })
    }
}

#[cfg(feature = "embedding")]
impl Reranker for CrossEncoderReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        use crate::error::TensaError;
        use ndarray::Array2;
        use ort::value::Tensor;

        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());

        // Score each query-document pair
        for (i, doc) in documents.iter().enumerate() {
            let pair_text = format!("{} [SEP] {}", query, doc);

            let encoding = self
                .tokenizer
                .encode(pair_text.as_str(), true)
                .map_err(|e| TensaError::EmbeddingError(format!("Tokenization: {e}")))?;

            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();
            let len = ids.len().min(self.max_length);

            let ids_i64: Vec<i64> = ids[..len].iter().map(|&x| x as i64).collect();
            let attn_i64: Vec<i64> = attention[..len].iter().map(|&x| x as i64).collect();

            let ids_array = Array2::from_shape_vec((1, len), ids_i64)
                .map_err(|e| TensaError::EmbeddingError(format!("Shape error: {e}")))?;
            let attn_array = Array2::from_shape_vec((1, len), attn_i64)
                .map_err(|e| TensaError::EmbeddingError(format!("Shape error: {e}")))?;

            let ids_tensor = Tensor::from_array(ids_array)
                .map_err(|e| TensaError::EmbeddingError(format!("Tensor: {e}")))?;
            let attn_tensor = Tensor::from_array(attn_array)
                .map_err(|e| TensaError::EmbeddingError(format!("Tensor: {e}")))?;

            let mut session = self
                .session
                .lock()
                .map_err(|e| TensaError::EmbeddingError(format!("Lock: {e}")))?;

            let outputs = session
                .run(ort::inputs![ids_tensor, attn_tensor])
                .map_err(|e| TensaError::EmbeddingError(format!("Inference: {e}")))?;

            // Extract logit score from output (typically shape [1, 1] or [1])
            if let Some((_name, output)) = outputs.iter().next() {
                if let Ok(tensor) = output.try_extract_tensor::<f32>() {
                    let score = *tensor.1.iter().next().unwrap_or(&0.0);
                    scored.push((i, score));
                } else {
                    scored.push((i, 0.0));
                }
            } else {
                scored.push((i, 0.0));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_preserves_order() {
        let reranker = NoopReranker;
        let docs = vec!["alpha bravo", "charlie delta", "echo foxtrot"];
        let result = reranker.rerank("any query", &docs).unwrap();
        assert_eq!(result.len(), 3);
        for (i, (idx, score)) in result.iter().enumerate() {
            assert_eq!(*idx, i);
            assert!((score - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_term_overlap_scores() {
        let reranker = TermOverlapReranker::new();
        let docs = vec![
            "The cat sat on the mat",         // matches: cat, sat, mat
            "A dog ran through the park",     // matches: none of cat/sat/mat
            "The cat and the mat were dirty", // matches: cat, mat
        ];
        let result = reranker.rerank("cat sat on the mat", &docs).unwrap();
        // "cat sat mat" are the query terms (len>2), "the" and "on" are filtered
        // doc0 should score highest (cat, sat, mat all present)
        assert_eq!(result[0].0, 0);
        // doc2 has cat+mat => second
        assert_eq!(result[1].0, 2);
        // doc1 has none => last
        assert_eq!(result[2].0, 1);
        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 > result[2].1);
    }

    #[test]
    fn test_term_overlap_empty_query() {
        let reranker = TermOverlapReranker::new();
        let docs = vec!["some document", "another one"];
        let result = reranker.rerank("", &docs).unwrap();
        assert_eq!(result.len(), 2);
        // All uniform scores
        for (_idx, score) in &result {
            assert!((score - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_reranker_empty_docs() {
        let noop = NoopReranker;
        assert!(noop.rerank("query", &[]).unwrap().is_empty());

        let term = TermOverlapReranker::new();
        assert!(term.rerank("query", &[]).unwrap().is_empty());
    }

    #[test]
    fn test_term_overlap_short_words_filtered() {
        let reranker = TermOverlapReranker::new();
        let docs = vec!["is it an ox?"];
        // All query terms <= 2 chars, so uniform scores
        let result = reranker.rerank("is it an ox", &docs).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_reranker_type_default() {
        let rt = RerankerType::default();
        assert_eq!(rt, RerankerType::TermOverlap);
    }

    #[test]
    fn test_reranker_type_serde() {
        let rt = RerankerType::CrossEncoder;
        let json = serde_json::to_string(&rt).unwrap();
        assert_eq!(json, "\"cross_encoder\"");

        let back: RerankerType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RerankerType::CrossEncoder);
    }

    #[test]
    fn test_reranker_type_none() {
        let rt: RerankerType = serde_json::from_str("\"none\"").unwrap();
        assert_eq!(rt, RerankerType::None);
    }
}
