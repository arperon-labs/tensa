//! Claim mutation tracking -- detect how claims evolve over time.
//!
//! Compares claims within the same narrative to detect paraphrases,
//! detail shifts, and semantic drift.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::Hypergraph;

use super::matching::cosine_similarity;
use super::types::{Claim, MutationEvent, MutationType};

/// Thresholds for mutation classification.
const PARAPHRASE_TEXT_THRESHOLD: f64 = 0.85;
const DETAIL_SHIFT_TEXT_THRESHOLD: f64 = 0.60;
const SEMANTIC_DRIFT_EMBED_THRESHOLD: f64 = 0.70;

/// Track mutations of a claim by comparing it against all other claims
/// in the same narrative. Returns mutation events sorted by detection time.
pub fn track_mutations(hypergraph: &Hypergraph, claim: &Claim) -> Result<Vec<MutationEvent>> {
    let narrative_id = match &claim.narrative_id {
        Some(nid) => nid.clone(),
        None => return Ok(Vec::new()),
    };

    let all_claims = super::detection::list_claims_for_narrative(hypergraph, &narrative_id)?;
    let mut events = Vec::new();

    for other in &all_claims {
        if other.id == claim.id {
            continue;
        }

        let text_sim = strsim::normalized_levenshtein(&claim.text, &other.text);
        let text_distance = 1.0 - text_sim;

        // Skip if too dissimilar to be a mutation
        if text_sim < 0.30 {
            continue;
        }

        let embedding_drift = match (&claim.embedding, &other.embedding) {
            (Some(a), Some(b)) if !a.is_empty() && a.len() == b.len() => {
                1.0 - cosine_similarity(a, b)
            }
            _ => text_distance, // fallback to text distance
        };

        let mutation_type = classify_mutation(text_sim, embedding_drift);

        events.push(MutationEvent {
            id: Uuid::now_v7(),
            original_claim_id: if claim.created_at <= other.created_at {
                claim.id
            } else {
                other.id
            },
            mutated_claim_id: if claim.created_at <= other.created_at {
                other.id
            } else {
                claim.id
            },
            embedding_drift,
            text_distance,
            mutation_type,
            detected_at: Utc::now(),
        });
    }

    events.sort_by(|a, b| a.detected_at.cmp(&b.detected_at));
    Ok(events)
}

/// Classify the type of mutation based on text similarity and embedding drift.
fn classify_mutation(text_sim: f64, embedding_drift: f64) -> MutationType {
    if text_sim >= PARAPHRASE_TEXT_THRESHOLD {
        MutationType::Paraphrase
    } else if embedding_drift > (1.0 - SEMANTIC_DRIFT_EMBED_THRESHOLD) {
        MutationType::SemanticDrift
    } else if text_sim >= DETAIL_SHIFT_TEXT_THRESHOLD {
        MutationType::DetailShift
    } else {
        MutationType::ContextShift
    }
}

/// Compute the average embedding drift for all mutations of claims in a narrative.
/// Used for DisinfoFingerprint axis #8 (claim_mutation_rate).
///
/// Uses a single claims load + O(n²/2) pairwise comparison instead of
/// re-loading claims per iteration.
pub fn narrative_mutation_rate(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Option<f64>> {
    let claims = super::detection::list_claims_for_narrative(hypergraph, narrative_id)?;
    if claims.len() < 2 {
        return Ok(None);
    }

    let mut total_drift = 0.0;
    let mut count = 0usize;

    // Pairwise comparison without re-loading claims from KV
    for i in 0..claims.len() {
        for j in (i + 1)..claims.len() {
            let text_sim = strsim::normalized_levenshtein(&claims[i].text, &claims[j].text);
            if text_sim < 0.30 {
                continue;
            }
            let embedding_drift = match (&claims[i].embedding, &claims[j].embedding) {
                (Some(a), Some(b)) if !a.is_empty() && a.len() == b.len() => {
                    1.0 - super::matching::cosine_similarity(a, b)
                }
                _ => 1.0 - text_sim,
            };
            total_drift += embedding_drift;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(None);
    }

    let avg = total_drift / count as f64;
    let rate = (2.0 * avg).tanh();
    Ok(Some(rate.clamp(0.0, 1.0)))
}

/// Store a mutation event at `cl/mut/{original_id}/{mutated_id}`.
pub fn store_mutation(hypergraph: &Hypergraph, event: &MutationEvent) -> Result<()> {
    let key = format!(
        "cl/mut/{}/{}",
        event.original_claim_id, event.mutated_claim_id
    );
    let value = serde_json::to_vec(event).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_paraphrase() {
        assert_eq!(classify_mutation(0.90, 0.05), MutationType::Paraphrase);
    }

    #[test]
    fn test_classify_semantic_drift() {
        assert_eq!(classify_mutation(0.40, 0.50), MutationType::SemanticDrift);
    }

    #[test]
    fn test_classify_detail_shift() {
        assert_eq!(classify_mutation(0.70, 0.15), MutationType::DetailShift);
    }

    #[test]
    fn test_narrative_mutation_rate_empty() {
        let store = std::sync::Arc::new(crate::store::memory::MemoryStore::new());
        let hg = Hypergraph::new(store);
        let rate = narrative_mutation_rate(&hg, "empty-narrative").unwrap();
        assert!(rate.is_none());
    }
}
