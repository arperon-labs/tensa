//! Style embeddings: dense voice vectors from author corpora (Sprint D9.5).
//!
//! KV persistence at `se/` (style embedding) and `se/blend/` prefixes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// Source of a style embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StyleEmbeddingSource {
    SingleAuthor { author_id: Uuid },
    Blended { sources: Vec<(Uuid, f64)> },
    GenreComposite { genre: String, corpus_size: usize },
    Custom { label: String },
}

/// A dense style embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleEmbedding {
    pub id: Uuid,
    /// Dense vector (256 or 512 dimensions).
    pub vector: Vec<f32>,
    pub source: StyleEmbeddingSource,
    pub base_model: String,
    pub training_corpus_size: usize,
    pub created_at: DateTime<Utc>,
}

impl StyleEmbedding {
    /// Dimension of the embedding vector.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &StyleEmbedding) -> f64 {
        crate::ingestion::embed::cosine_similarity(&self.vector, &other.vector) as f64
    }
}

/// A blend recipe specifying how multiple author styles are combined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleBlendRecipe {
    pub id: Uuid,
    pub label: String,
    pub components: Vec<(Uuid, f64)>,
    pub result: StyleEmbedding,
    pub created_at: DateTime<Utc>,
}

// ─── KV Operations ──────────────────────────────────────────

fn embedding_key(id: &Uuid) -> Vec<u8> {
    format!("se/{}", id).into_bytes()
}

fn blend_key(id: &Uuid) -> Vec<u8> {
    format!("se/blend/{}", id).into_bytes()
}

pub fn store_embedding(hg: &Hypergraph, emb: &StyleEmbedding) -> Result<()> {
    let key = embedding_key(&emb.id);
    let val = serde_json::to_vec(emb)?;
    hg.store().put(&key, &val)
}

pub fn load_embedding(hg: &Hypergraph, id: &Uuid) -> Result<Option<StyleEmbedding>> {
    let key = embedding_key(id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn list_embeddings(hg: &Hypergraph) -> Result<Vec<StyleEmbedding>> {
    let prefix = b"se/";
    let items = hg.store().prefix_scan(prefix)?;
    let mut out = Vec::new();
    for (k, v) in items {
        let key_str = String::from_utf8_lossy(&k);
        if key_str.contains("/blend/") {
            continue;
        }
        if let Ok(emb) = serde_json::from_slice::<StyleEmbedding>(&v) {
            out.push(emb);
        }
    }
    Ok(out)
}

pub fn store_blend(hg: &Hypergraph, recipe: &StyleBlendRecipe) -> Result<()> {
    let key = blend_key(&recipe.id);
    let val = serde_json::to_vec(recipe)?;
    hg.store().put(&key, &val)
}

pub fn load_blend(hg: &Hypergraph, id: &Uuid) -> Result<Option<StyleBlendRecipe>> {
    let key = blend_key(id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

// ─── Core Operations ────────────────────────────────────────

/// Blend multiple style embeddings with weighted average.
///
/// Because style embeddings live in a learned homogeneous space, linear
/// interpolation is valid (unlike fingerprints which need per-layer handling).
pub fn blend_styles(styles: &[(StyleEmbedding, f64)]) -> Result<StyleEmbedding> {
    if styles.is_empty() {
        return Err(TensaError::InvalidQuery(
            "cannot blend zero style embeddings".into(),
        ));
    }

    let dim = styles[0].0.dim();
    if dim == 0 {
        return Err(TensaError::InvalidQuery(
            "cannot blend zero-dimensional embeddings".into(),
        ));
    }

    // Normalize weights
    let total_weight: f64 = styles.iter().map(|(_, w)| w).sum();
    if total_weight < 1e-10 {
        return Err(TensaError::InvalidQuery("blend weights sum to zero".into()));
    }

    let mut blended = vec![0.0f32; dim];
    for (emb, weight) in styles {
        if emb.dim() != dim {
            return Err(TensaError::InvalidQuery(format!(
                "dimension mismatch: expected {}, got {}",
                dim,
                emb.dim()
            )));
        }
        let norm_w = (*weight / total_weight) as f32;
        for (i, v) in emb.vector.iter().enumerate() {
            blended[i] += v * norm_w;
        }
    }

    let sources: Vec<(Uuid, f64)> = styles
        .iter()
        .filter_map(|(emb, w)| {
            if let StyleEmbeddingSource::SingleAuthor { author_id } = &emb.source {
                Some((*author_id, *w / total_weight))
            } else {
                None
            }
        })
        .collect();

    Ok(StyleEmbedding {
        id: Uuid::now_v7(),
        vector: blended,
        source: StyleEmbeddingSource::Blended { sources },
        base_model: styles[0].0.base_model.clone(),
        training_corpus_size: styles.iter().map(|(e, _)| e.training_corpus_size).sum(),
        created_at: Utc::now(),
    })
}

/// Encode an author corpus by averaging embeddings with outlier removal.
///
/// Chunks deviating >2σ from the mean are removed before final averaging.
pub fn average_with_outlier_removal(embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
    if embeddings.is_empty() {
        return Err(TensaError::InvalidQuery("no embeddings to average".into()));
    }

    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;

    // Compute mean
    let mut mean = vec![0.0f32; dim];
    for emb in embeddings {
        for (i, v) in emb.iter().enumerate() {
            mean[i] += v / n;
        }
    }

    // Compute distances from mean
    let distances: Vec<f64> = embeddings
        .iter()
        .map(|emb| {
            emb.iter()
                .zip(mean.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;
    let std_dist = (distances
        .iter()
        .map(|d| (d - mean_dist).powi(2))
        .sum::<f64>()
        / distances.len() as f64)
        .sqrt();

    let threshold = mean_dist + 2.0 * std_dist;

    // Re-average excluding outliers
    let mut filtered_mean = vec![0.0f32; dim];
    let mut count = 0.0f32;
    for (emb, dist) in embeddings.iter().zip(distances.iter()) {
        if *dist <= threshold {
            for (i, v) in emb.iter().enumerate() {
                filtered_mean[i] += v;
            }
            count += 1.0;
        }
    }

    if count < 1.0 {
        return Ok(mean); // Fallback if all are outliers
    }

    for v in &mut filtered_mean {
        *v /= count;
    }

    Ok(filtered_mean)
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_emb(id: Uuid, vector: Vec<f32>) -> StyleEmbedding {
        StyleEmbedding {
            id,
            vector,
            source: StyleEmbeddingSource::SingleAuthor { author_id: id },
            base_model: "test-model".into(),
            training_corpus_size: 1000,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let emb = make_emb(Uuid::now_v7(), vec![1.0, 0.0, 0.0, 0.0]);
        assert!((emb.cosine_similarity(&emb) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = make_emb(Uuid::now_v7(), vec![1.0, 0.0, 0.0, 0.0]);
        let b = make_emb(Uuid::now_v7(), vec![0.0, 1.0, 0.0, 0.0]);
        assert!(a.cosine_similarity(&b).abs() < 0.001);
    }

    #[test]
    fn test_blend_50_50_equidistant() {
        let a = make_emb(Uuid::now_v7(), vec![1.0, 0.0, 0.0, 0.0]);
        let b = make_emb(Uuid::now_v7(), vec![0.0, 1.0, 0.0, 0.0]);
        let blended = blend_styles(&[(a.clone(), 0.5), (b.clone(), 0.5)]).unwrap();

        assert_eq!(blended.vector, vec![0.5, 0.5, 0.0, 0.0]);

        // Blended should be equidistant from both sources
        let dist_a = blended.cosine_similarity(&a);
        let dist_b = blended.cosine_similarity(&b);
        assert!((dist_a - dist_b).abs() < 0.01);
    }

    #[test]
    fn test_blend_weighted() {
        let a = make_emb(Uuid::now_v7(), vec![1.0, 0.0]);
        let b = make_emb(Uuid::now_v7(), vec![0.0, 1.0]);
        let blended = blend_styles(&[(a, 0.75), (b, 0.25)]).unwrap();
        assert!((blended.vector[0] - 0.75).abs() < 0.01);
        assert!((blended.vector[1] - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_outlier_removal() {
        // 6 normal points + 1 extreme outlier — 2σ threshold should exclude it
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![0.9, -0.1],
            vec![1.0, 0.05],
            vec![0.95, -0.05],
            vec![1.05, 0.0],
            vec![100.0, 100.0], // Outlier
        ];
        let avg = average_with_outlier_removal(&embeddings).unwrap();
        // The outlier should be removed, average should be near [1.0, 0.0]
        assert!(avg[0] < 2.0, "Outlier should be removed, got {}", avg[0]);
    }

    #[test]
    fn test_embedding_kv() {
        let hg = test_hg();
        let emb = make_emb(Uuid::now_v7(), vec![0.5, 0.5, 0.5, 0.5]);
        store_embedding(&hg, &emb).unwrap();
        let loaded = load_embedding(&hg, &emb.id).unwrap().unwrap();
        assert_eq!(loaded.vector, emb.vector);
    }

    #[test]
    fn test_blend_recipe_kv() {
        let hg = test_hg();
        let emb = make_emb(Uuid::now_v7(), vec![0.5, 0.5]);
        let recipe = StyleBlendRecipe {
            id: Uuid::now_v7(),
            label: "hemingway-marquez".into(),
            components: vec![(Uuid::now_v7(), 0.6), (Uuid::now_v7(), 0.4)],
            result: emb,
            created_at: Utc::now(),
        };
        store_blend(&hg, &recipe).unwrap();
        let loaded = load_blend(&hg, &recipe.id).unwrap().unwrap();
        assert_eq!(loaded.label, "hemingway-marquez");
    }
}
