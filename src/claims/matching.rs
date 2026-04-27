//! Claim matching -- match new claims against known fact-checks.
//!
//! Uses text similarity (normalized Levenshtein) as primary matcher,
//! with embedding cosine similarity when embeddings are available.

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::Hypergraph;

use super::types::{Claim, FactCheck, MatchMethod, MatchedFactCheck};

/// Match a claim against all stored fact-checks for the same narrative.
///
/// Returns matches sorted by similarity (highest first), filtered by
/// `min_similarity` threshold.
pub fn match_claim(
    hypergraph: &Hypergraph,
    claim: &Claim,
    min_similarity: f64,
) -> Result<Vec<MatchedFactCheck>> {
    let fact_checks = list_fact_checks(hypergraph, claim.narrative_id.as_deref())?;
    let mut matches = Vec::new();

    for fc in &fact_checks {
        // Load the claim this fact-check is about
        let checked_claim = match super::detection::load_claim(hypergraph, &fc.claim_id)? {
            Some(c) => c,
            None => continue,
        };

        // Text similarity (normalized Levenshtein)
        let text_sim = strsim::normalized_levenshtein(&claim.text, &checked_claim.text);

        // Embedding similarity if both have embeddings
        let embed_sim = match (&claim.embedding, &checked_claim.embedding) {
            (Some(a), Some(b)) if !a.is_empty() && a.len() == b.len() => {
                Some(cosine_similarity(a, b))
            }
            _ => None,
        };

        // Take the max of text and embedding similarity
        let (similarity, method) = match embed_sim {
            Some(es) if es > text_sim => (es, MatchMethod::Embedding),
            _ => (
                text_sim,
                if text_sim > 0.95 {
                    MatchMethod::ExactText
                } else {
                    MatchMethod::KeywordRerank
                },
            ),
        };

        if similarity >= min_similarity {
            matches.push(MatchedFactCheck {
                claim_id: claim.id,
                fact_check_id: fc.id,
                similarity,
                method,
                matched_at: Utc::now(),
            });
        }
    }

    matches.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(matches)
}

/// Cosine similarity between two vectors.
pub(crate) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a < 1e-10 || mag_b < 1e-10 {
        return 0.0;
    }
    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

/// List all fact-checks. Skips `fc/sync/` sub-prefix (sync audit records).
pub fn list_fact_checks(
    hypergraph: &Hypergraph,
    _narrative_id: Option<&str>,
) -> Result<Vec<FactCheck>> {
    let prefix = b"fc/";
    let pairs = hypergraph
        .store()
        .prefix_scan(prefix)
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    let mut checks = Vec::new();
    for (key, value) in pairs {
        // Skip sync result records at fc/sync/
        if key.starts_with(b"fc/sync/") {
            continue;
        }
        if let Ok(fc) = serde_json::from_slice::<FactCheck>(&value) {
            checks.push(fc);
        }
    }
    Ok(checks)
}

/// Store a matched fact-check record at `cl/m/{claim_id}/{fact_check_id}`.
pub fn store_match(hypergraph: &Hypergraph, m: &MatchedFactCheck) -> Result<()> {
    let key = format!("cl/m/{}/{}", m.claim_id, m.fact_check_id);
    let value = serde_json::to_vec(m).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 1.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_match_claim_empty_store() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let claim = super::super::types::Claim {
            id: Uuid::now_v7(),
            text: "Test claim".to_string(),
            original_text: "Test claim".to_string(),
            language: "en".to_string(),
            source_situation_id: None,
            source_entity_id: None,
            narrative_id: Some("test".into()),
            embedding: None,
            confidence: 0.8,
            category: super::super::types::ClaimCategory::Factual,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        let matches = match_claim(&hg, &claim, 0.5).unwrap();
        assert!(matches.is_empty());
    }
}
