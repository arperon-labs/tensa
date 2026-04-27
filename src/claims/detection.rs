//! Claim detection -- extract verifiable claims from text.
//!
//! Uses regex-based heuristic patterns to identify check-worthy sentences.
//! A future sprint will add LLM-assisted detection for higher quality.

use chrono::Utc;
use regex::Regex;
use std::sync::OnceLock;
use uuid::Uuid;

use crate::error::Result;
use crate::Hypergraph;

use super::types::{Claim, ClaimCategory};

/// Regex patterns for different claim categories.
struct ClaimPatterns {
    numerical: Regex,
    quote: Regex,
    causal: Regex,
    comparison: Regex,
    predictive: Regex,
    factual: Regex,
}

fn patterns() -> &'static ClaimPatterns {
    static PATTERNS: OnceLock<ClaimPatterns> = OnceLock::new();
    PATTERNS.get_or_init(|| ClaimPatterns {
        numerical: Regex::new(
            r"(?i)\b\d+[\.,]?\d*\s*(%|percent|million|billion|thousand|hundred|times|fold)\b",
        )
        .unwrap(),
        quote: Regex::new(
            r#"(?i)(said|stated|claimed|according to|told|announced)\s+"#,
        )
        .unwrap(),
        causal: Regex::new(
            r"(?i)\b(because|caused|leads?\s+to|results?\s+in|due\s+to|responsible\s+for|triggers?)\b",
        )
        .unwrap(),
        comparison: Regex::new(
            r"(?i)\b(more\s+than|less\s+than|greater|fewer|better|worse|higher|lower|largest|smallest|most|least)\b",
        )
        .unwrap(),
        predictive: Regex::new(
            r"(?i)\b(will|shall|going\s+to|expected\s+to|predicted|forecast|projected|likely\s+to)\b",
        )
        .unwrap(),
        factual: Regex::new(
            r"(?i)\b(is|are|was|were|has|have|had)\s+(a|an|the|always|never|every|no|all)\b",
        )
        .unwrap(),
    })
}

/// Split text into sentences (simple heuristic).
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if (c == '.' || c == '!' || c == '?') && i + 1 < text.len() {
            let next = text[i + 1..].chars().next();
            if next == Some(' ') || next == Some('\n') {
                let s = text[start..=i].trim();
                if !s.is_empty() && s.len() > 10 {
                    sentences.push(s);
                }
                start = i + 1;
            }
        }
    }
    // Don't forget the last segment
    let last = text[start..].trim();
    if !last.is_empty() && last.len() > 10 {
        sentences.push(last);
    }
    sentences
}

/// Classify a sentence into a claim category based on regex patterns.
fn classify_sentence(sentence: &str) -> Option<(ClaimCategory, f64)> {
    let pats = patterns();
    // Priority order: numerical > quote > causal > comparison > predictive > factual
    if pats.numerical.is_match(sentence) {
        return Some((ClaimCategory::Numerical, 0.85));
    }
    if pats.quote.is_match(sentence) {
        return Some((ClaimCategory::Quote, 0.75));
    }
    if pats.causal.is_match(sentence) {
        return Some((ClaimCategory::Causal, 0.70));
    }
    if pats.comparison.is_match(sentence) {
        return Some((ClaimCategory::Comparison, 0.65));
    }
    if pats.predictive.is_match(sentence) {
        return Some((ClaimCategory::Predictive, 0.60));
    }
    if pats.factual.is_match(sentence) {
        return Some((ClaimCategory::Factual, 0.50));
    }
    None
}

/// Detect verifiable claims in raw text using regex-based heuristics.
///
/// Returns a list of `Claim` structs with confidence scores and categories.
/// Each claim is assigned a UUIDv7.
pub fn detect_claims(
    text: &str,
    narrative_id: Option<&str>,
    source_situation_id: Option<Uuid>,
    source_entity_id: Option<Uuid>,
) -> Vec<Claim> {
    let sentences = split_sentences(text);
    let mut claims = Vec::new();
    for sentence in sentences {
        if let Some((category, confidence)) = classify_sentence(sentence) {
            claims.push(Claim {
                id: Uuid::now_v7(),
                text: sentence.to_string(),
                original_text: sentence.to_string(),
                language: "en".to_string(),
                source_situation_id,
                source_entity_id,
                narrative_id: narrative_id.map(String::from),
                embedding: None,
                confidence,
                category,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            });
        }
    }
    claims
}

/// Persist a claim to the KV store at `cl/{claim_uuid}`.
pub fn store_claim(hypergraph: &Hypergraph, claim: &Claim) -> Result<()> {
    let key = format!("cl/{}", claim.id);
    let value =
        serde_json::to_vec(claim).map_err(|e| crate::TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| crate::TensaError::Internal(e.to_string()))
}

/// Load a claim by UUID from `cl/{claim_uuid}`.
pub fn load_claim(hypergraph: &Hypergraph, claim_id: &Uuid) -> Result<Option<Claim>> {
    let key = format!("cl/{}", claim_id);
    let value = hypergraph
        .store()
        .get(key.as_bytes())
        .map_err(|e| crate::TensaError::Internal(e.to_string()))?;
    match value {
        Some(bytes) => {
            let claim: Claim = serde_json::from_slice(&bytes)
                .map_err(|e| crate::TensaError::Serialization(e.to_string()))?;
            Ok(Some(claim))
        }
        None => Ok(None),
    }
}

/// List claims, optionally filtered by narrative. Pass `None` to list all.
pub fn list_claims(hypergraph: &Hypergraph, narrative_id: Option<&str>) -> Result<Vec<Claim>> {
    let prefix = b"cl/";
    let pairs = hypergraph
        .store()
        .prefix_scan(prefix)
        .map_err(|e| crate::TensaError::Internal(e.to_string()))?;
    let mut claims = Vec::new();
    for (key, value) in pairs {
        // Skip sub-prefixes (cl/m/, cl/mut/, cl/n/)
        let key_str = String::from_utf8_lossy(&key);
        if key_str.starts_with("cl/m") || key_str.starts_with("cl/n") {
            continue;
        }
        if let Ok(claim) = serde_json::from_slice::<Claim>(&value) {
            match narrative_id {
                Some(nid) => {
                    if claim.narrative_id.as_deref() == Some(nid) {
                        claims.push(claim);
                    }
                }
                None => claims.push(claim),
            }
        }
    }
    Ok(claims)
}

/// List all claims for a narrative (convenience wrapper).
pub fn list_claims_for_narrative(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<Claim>> {
    list_claims(hypergraph, Some(narrative_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_numerical_claim() {
        let text = "The vaccine was administered to over 5 million people in the first week. This caused widespread concern.";
        let claims = detect_claims(text, Some("test"), None, None);
        assert!(!claims.is_empty());
        assert!(claims
            .iter()
            .any(|c| c.category == ClaimCategory::Numerical));
    }

    #[test]
    fn test_detect_causal_claim() {
        let text = "The policy change leads to increased unemployment in the region. Citizens protested loudly.";
        let claims = detect_claims(text, Some("test"), None, None);
        assert!(claims.iter().any(|c| c.category == ClaimCategory::Causal));
    }

    #[test]
    fn test_detect_quote_claim() {
        let text = r#"The president said "we will prevail" during the address. Applause followed."#;
        let claims = detect_claims(text, Some("test"), None, None);
        assert!(claims.iter().any(|c| c.category == ClaimCategory::Quote));
    }

    #[test]
    fn test_no_claims_in_short_text() {
        let text = "Hello.";
        let claims = detect_claims(text, None, None, None);
        assert!(claims.is_empty());
    }

    #[test]
    fn test_store_and_load_claim() {
        let store = crate::store::memory::MemoryStore::new();
        let hg = Hypergraph::new(std::sync::Arc::new(store));
        let claims = detect_claims(
            "There are more than 100 cases reported daily. Officials confirmed this trend.",
            Some("test-nar"),
            None,
            None,
        );
        assert!(!claims.is_empty());
        let claim = &claims[0];
        store_claim(&hg, claim).unwrap();
        let loaded = load_claim(&hg, &claim.id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().id, claim.id);
    }
}
