//! Fact-check ingestion -- create attack relations in the argumentation framework.
//!
//! When a fact-check is ingested, it creates an argumentation attack
//! against the original claim and triggers DS evidence fusion update.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::Hypergraph;

use super::types::{FactCheck, FactCheckVerdict};

/// Ingest a fact-check: persist it and wire argumentation attack relation.
///
/// Returns the fact-check ID. The caller is responsible for any DS fusion
/// update if needed.
pub fn ingest_fact_check(
    hypergraph: &Hypergraph,
    claim_id: Uuid,
    verdict: FactCheckVerdict,
    source: &str,
    url: Option<&str>,
    language: &str,
    explanation: Option<&str>,
    confidence: f64,
) -> Result<FactCheck> {
    // Verify the claim exists
    let _claim = super::detection::load_claim(hypergraph, &claim_id)?
        .ok_or_else(|| TensaError::NotFound(format!("Claim {} not found", claim_id)))?;

    let fc = FactCheck {
        id: Uuid::now_v7(),
        claim_id,
        verdict,
        source: source.to_string(),
        url: url.map(String::from),
        language: language.to_string(),
        explanation: explanation.map(String::from),
        confidence,
        created_at: Utc::now(),
    };

    store_fact_check(hypergraph, &fc)?;
    Ok(fc)
}

/// Persist a fact-check at `fc/{fact_check_uuid}`.
pub fn store_fact_check(hypergraph: &Hypergraph, fc: &FactCheck) -> Result<()> {
    let key = format!("fc/{}", fc.id);
    let value = serde_json::to_vec(fc).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

/// Load a fact-check by UUID.
pub fn load_fact_check(hypergraph: &Hypergraph, fc_id: &Uuid) -> Result<Option<FactCheck>> {
    let key = format!("fc/{}", fc_id);
    let value = hypergraph
        .store()
        .get(key.as_bytes())
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    match value {
        Some(bytes) => {
            let fc: FactCheck = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(fc))
        }
        None => Ok(None),
    }
}

/// List all fact-checks for a given claim.
pub fn list_fact_checks_for_claim(
    hypergraph: &Hypergraph,
    claim_id: &Uuid,
) -> Result<Vec<FactCheck>> {
    let prefix = b"fc/";
    let pairs = hypergraph
        .store()
        .prefix_scan(prefix)
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    let mut checks = Vec::new();
    for (_, value) in pairs {
        if let Ok(fc) = serde_json::from_slice::<FactCheck>(&value) {
            if fc.claim_id == *claim_id {
                checks.push(fc);
            }
        }
    }
    Ok(checks)
}

/// Compute the counter-narrative resistance metric for DisinfoFingerprint
/// axis #9. This is the ratio of undefeated (no fact-check) claims to total
/// claims in the narrative.
///
/// Loads all fact-checks once rather than per-claim to avoid O(N×F) KV scans.
pub fn counter_narrative_resistance(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<f64>> {
    let claims = super::detection::list_claims_for_narrative(hypergraph, narrative_id)?;
    if claims.is_empty() {
        return Ok(None);
    }

    // Build set of claim IDs that have attacking fact-checks (single scan)
    let claim_ids: std::collections::HashSet<Uuid> = claims.iter().map(|c| c.id).collect();
    let all_fcs = super::matching::list_fact_checks(hypergraph, None)?;
    let mut attacked: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for fc in &all_fcs {
        if claim_ids.contains(&fc.claim_id)
            && matches!(
                fc.verdict,
                FactCheckVerdict::False
                    | FactCheckVerdict::Misleading
                    | FactCheckVerdict::OutOfContext
            )
        {
            attacked.insert(fc.claim_id);
        }
    }

    let undefeated = claims.iter().filter(|c| !attacked.contains(&c.id)).count();
    let ratio = undefeated as f64 / claims.len() as f64;
    Ok(Some(ratio.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::super::types::Claim;
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_ingest_fact_check_missing_claim() {
        let hg = setup();
        let result = ingest_fact_check(
            &hg,
            Uuid::now_v7(),
            FactCheckVerdict::False,
            "TestChecker",
            None,
            "en",
            None,
            0.9,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_ingest_and_load_fact_check() {
        let hg = setup();
        // First store a claim
        let claim = Claim {
            id: Uuid::now_v7(),
            text: "5 million affected".to_string(),
            original_text: "5 million affected".to_string(),
            language: "en".to_string(),
            source_situation_id: None,
            source_entity_id: None,
            narrative_id: Some("test".into()),
            embedding: None,
            confidence: 0.8,
            category: super::super::types::ClaimCategory::Numerical,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        super::super::detection::store_claim(&hg, &claim).unwrap();

        // Ingest a fact-check
        let fc = ingest_fact_check(
            &hg,
            claim.id,
            FactCheckVerdict::False,
            "Reuters",
            Some("https://reuters.com/check/1"),
            "en",
            Some("The number is inflated"),
            0.95,
        )
        .unwrap();

        let loaded = load_fact_check(&hg, &fc.id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().verdict, FactCheckVerdict::False);
    }

    #[test]
    fn test_counter_narrative_resistance_no_claims() {
        let hg = setup();
        let r = counter_narrative_resistance(&hg, "empty").unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn test_counter_narrative_resistance_all_undefeated() {
        let hg = setup();
        let claim = Claim {
            id: Uuid::now_v7(),
            text: "claim".to_string(),
            original_text: "claim".to_string(),
            language: "en".to_string(),
            source_situation_id: None,
            source_entity_id: None,
            narrative_id: Some("test-nar".into()),
            embedding: None,
            confidence: 0.8,
            category: super::super::types::ClaimCategory::Factual,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        super::super::detection::store_claim(&hg, &claim).unwrap();
        let r = counter_narrative_resistance(&hg, "test-nar").unwrap();
        assert_eq!(r, Some(1.0)); // no fact-checks = fully undefeated
    }
}
