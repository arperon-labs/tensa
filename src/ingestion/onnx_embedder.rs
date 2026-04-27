//! ONNX-based semantic embedding provider.
//!
//! Loads a sentence-transformer model (e.g. all-MiniLM-L6-v2) via the `ort`
//! crate and a HuggingFace tokenizer. Feature-gated behind `embedding`.
//!
//! Model directory must contain `model.onnx` and `tokenizer.json`.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Axis};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{Result, TensaError};
use crate::ingestion::embed::{l2_normalize, EmbeddingProvider};

/// Semantic embedding provider backed by an ONNX Runtime session.
///
/// Wraps an ONNX session and HuggingFace tokenizer for semantic text embedding.
/// The session is behind a Mutex because `Session::run` requires `&mut self`.
pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    dimension: Mutex<usize>,
    max_length: usize,
    /// Number of inputs the model expects (2 or 3).
    num_inputs: usize,
    /// Target dimension for Matryoshka truncation. None = use full model output.
    target_dimension: Option<usize>,
}

impl OnnxEmbedder {
    /// Load model and tokenizer from a directory containing `model.onnx` and `tokenizer.json`.
    pub fn from_directory(dir: &str) -> Result<Self> {
        let dir_path = Path::new(dir);
        let model_path = dir_path.join("model.onnx");
        let tokenizer_path = dir_path.join("tokenizer.json");

        if !model_path.exists() {
            return Err(TensaError::EmbeddingError(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(TensaError::EmbeddingError(format!(
                "Tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        let model_str = model_path.to_str().ok_or_else(|| {
            TensaError::EmbeddingError(format!("Non-UTF-8 model path: {}", model_path.display()))
        })?;
        let tokenizer_str = tokenizer_path.to_str().ok_or_else(|| {
            TensaError::EmbeddingError(format!(
                "Non-UTF-8 tokenizer path: {}",
                tokenizer_path.display()
            ))
        })?;
        Self::from_paths(model_str, tokenizer_str)
    }

    /// Load from explicit model and tokenizer file paths.
    pub fn from_paths(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| TensaError::EmbeddingError(format!("ONNX session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TensaError::EmbeddingError(format!("ONNX optimization: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| TensaError::EmbeddingError(format!("ONNX model load: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| TensaError::EmbeddingError(format!("Tokenizer load: {e}")))?;

        // Detect number of model inputs (2 for mpnet-style, 3 for BERT-style)
        let num_inputs = session.inputs().len();

        // Dimension will be detected from first inference output.
        // Start with 0 as sentinel.
        let dimension = 0;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dimension: Mutex::new(dimension),
            max_length: 256,
            num_inputs,
            target_dimension: None,
        })
    }

    /// Set the maximum token sequence length (default: 256).
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set target dimension for Matryoshka Representation Learning truncation.
    /// When set, output embeddings are truncated to this dimension and re-normalized.
    /// Useful for models like ModernBERT-embed or nomic-embed-v1.5 that support
    /// variable dimensions (e.g., 256/512/768) with minimal quality loss.
    pub fn with_target_dimension(mut self, dim: usize) -> Self {
        self.target_dimension = Some(dim);
        self
    }

    /// Tokenize, run inference, mean-pool, and normalize.
    fn infer(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| TensaError::EmbeddingError(format!("Tokenization: {e}")))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate to max_length
        let len = input_ids.len().min(self.max_length);
        let input_ids = &input_ids[..len];
        let attention_mask = &attention_mask[..len];

        // Build ONNX input tensors [1, seq_len]
        let ids_array =
            Array2::from_shape_vec((1, len), input_ids.iter().map(|&x| x as i64).collect())
                .map_err(|e| TensaError::EmbeddingError(format!("Array shape: {e}")))?;
        let mask_array =
            Array2::from_shape_vec((1, len), attention_mask.iter().map(|&x| x as i64).collect())
                .map_err(|e| TensaError::EmbeddingError(format!("Mask array shape: {e}")))?;

        let ids_tensor = Tensor::<i64>::from_array(ids_array)
            .map_err(|e| TensaError::EmbeddingError(format!("ids tensor: {e}")))?;
        let mask_tensor = Tensor::<i64>::from_array(mask_array)
            .map_err(|e| TensaError::EmbeddingError(format!("mask tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| TensaError::EmbeddingError("Session lock poisoned".into()))?;

        // Some models (BERT) take 3 inputs; others (mpnet) take only 2
        let outputs = if self.num_inputs >= 3 {
            let type_ids_array = Array2::<i64>::zeros((1, len));
            let type_ids_tensor = Tensor::<i64>::from_array(type_ids_array)
                .map_err(|e| TensaError::EmbeddingError(format!("type_ids tensor: {e}")))?;
            session.run(ort::inputs![ids_tensor, mask_tensor, type_ids_tensor])
        } else {
            session.run(ort::inputs![ids_tensor, mask_tensor])
        }
        .map_err(|e| TensaError::EmbeddingError(format!("ONNX inference: {e}")))?;

        // Extract first output tensor (last_hidden_state or sentence_embedding)
        let (_name, output_value) = outputs
            .iter()
            .next()
            .ok_or_else(|| TensaError::EmbeddingError("No output tensor".into()))?;

        let (shape, flat_data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| TensaError::EmbeddingError(format!("Tensor extraction: {e}")))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let result = if dims.len() == 3 {
            // [batch=1, seq_len, dim] -> mean pool over seq_len with attention mask
            let seq_len = dims[1];
            let dim = dims[2];
            let embeddings =
                Array2::from_shape_vec((seq_len, dim), flat_data[..seq_len * dim].to_vec())
                    .map_err(|e| TensaError::EmbeddingError(format!("Reshape: {e}")))?;
            let mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
            let mut pooled = mean_pool(&embeddings, &mask_i64);
            l2_normalize(&mut pooled);
            pooled
        } else if dims.len() == 2 {
            // [batch=1, dim] -> already pooled (some models output this directly)
            let mut vec: Vec<f32> = flat_data.to_vec();
            l2_normalize(&mut vec);
            vec
        } else {
            return Err(TensaError::EmbeddingError(format!(
                "Unexpected output shape: {:?}",
                dims
            )));
        };

        // Matryoshka truncation: if target_dimension is set and smaller than output,
        // truncate and re-normalize for dimensionality reduction.
        let result = if let Some(target) = self.target_dimension {
            if target < result.len() {
                let mut truncated = result[..target].to_vec();
                l2_normalize(&mut truncated);
                truncated
            } else {
                result
            }
        } else {
            result
        };

        // Auto-detect dimension from first inference
        let mut dim_guard = self.dimension.lock().unwrap();
        if *dim_guard == 0 {
            *dim_guard = result.len();
        }

        Ok(result)
    }
}

impl EmbeddingProvider for OnnxEmbedder {
    fn dimension(&self) -> usize {
        if let Some(target) = self.target_dimension {
            return target;
        }
        let d = *self.dimension.lock().unwrap();
        if d == 0 {
            384
        } else {
            d
        } // fallback before first inference
    }

    fn provider_name(&self) -> &'static str {
        "onnx"
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(TensaError::EmbeddingError("Cannot embed empty text".into()));
        }
        self.infer(text)
    }
}

/// Mean pooling: average token embeddings weighted by attention mask.
pub(crate) fn mean_pool(embeddings: &Array2<f32>, attention_mask: &[i64]) -> Vec<f32> {
    let dim = embeddings.ncols();
    let mut sum = Array1::<f32>::zeros(dim);
    let mut count = 0.0f32;

    for (i, row) in embeddings.axis_iter(Axis(0)).enumerate() {
        let mask_val = attention_mask.get(i).copied().unwrap_or(0) as f32;
        sum = sum + &(&row * mask_val);
        count += mask_val;
    }

    if count > 0.0 {
        sum /= count;
    }

    sum.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pool_basic() {
        // 2 tokens, dim=3, all unmasked
        let embeddings =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]).unwrap();
        let mask = vec![1i64, 1];
        let result = mean_pool(&embeddings, &mask);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_with_mask() {
        // 3 tokens, dim=2, only first 2 unmasked (padding token masked out)
        let embeddings =
            Array2::from_shape_vec((3, 2), vec![2.0, 4.0, 6.0, 8.0, 100.0, 100.0]).unwrap();
        let mask = vec![1i64, 1, 0]; // third token is padding
        let result = mean_pool(&embeddings, &mask);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 4.0).abs() < 1e-6); // (2+6)/2
        assert!((result[1] - 6.0).abs() < 1e-6); // (4+8)/2
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]); // no division by zero
    }

    #[test]
    fn test_from_directory_missing() {
        let result = OnnxEmbedder::from_directory("/nonexistent/path/to/model");
        assert!(result.is_err());
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(msg.contains("Model file not found"));
    }

    #[test]
    fn test_from_directory_missing_tokenizer() {
        // Create a temp dir with model.onnx but no tokenizer.json
        let tmp = std::env::temp_dir().join("tensa_test_onnx_no_tok");
        let _ = std::fs::create_dir_all(&tmp);
        let model_path = tmp.join("model.onnx");
        let _ = std::fs::write(&model_path, b"fake");
        let result = OnnxEmbedder::from_directory(tmp.to_str().unwrap());
        assert!(result.is_err());
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error"),
        };
        assert!(msg.contains("Tokenizer file not found"));
        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Tests requiring a real ONNX model (run with --ignored) ──

    fn model_dir() -> Option<String> {
        std::env::var("TENSA_EMBEDDING_MODEL").ok()
    }

    #[test]
    #[ignore]
    fn test_onnx_embedder_loads() {
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load model");
        assert!(embedder.dimension() > 0);
    }

    #[test]
    #[ignore]
    fn test_onnx_embed_dimension() {
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load");
        let emb = embedder.embed_text("Hello world").unwrap();
        assert_eq!(emb.len(), embedder.dimension());
    }

    #[test]
    #[ignore]
    fn test_onnx_embed_deterministic() {
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load");
        let e1 = embedder.embed_text("Test text").unwrap();
        let e2 = embedder.embed_text("Test text").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    #[ignore]
    fn test_onnx_embed_normalized() {
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load");
        let emb = embedder.embed_text("Normalize this").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    #[ignore]
    fn test_onnx_embed_empty_error() {
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load");
        assert!(embedder.embed_text("").is_err());
    }

    #[test]
    #[ignore]
    fn test_onnx_embed_batch() {
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load");
        let results = embedder.embed_batch(&["Hello", "World"]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), embedder.dimension());
    }

    #[test]
    #[ignore]
    fn test_onnx_semantic_similarity() {
        use crate::ingestion::embed::cosine_similarity;
        let dir = model_dir().expect("Set TENSA_EMBEDDING_MODEL");
        let embedder = OnnxEmbedder::from_directory(&dir).expect("Failed to load");
        let cat = embedder.embed_text("cat").unwrap();
        let kitten = embedder.embed_text("kitten").unwrap();
        let airplane = embedder.embed_text("airplane").unwrap();
        let sim_close = cosine_similarity(&cat, &kitten);
        let sim_far = cosine_similarity(&cat, &airplane);
        assert!(
            sim_close > sim_far,
            "cat-kitten ({:.3}) should be more similar than cat-airplane ({:.3})",
            sim_close,
            sim_far
        );
    }
}
