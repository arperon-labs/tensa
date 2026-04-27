//! Inference engines for claim pipeline (Sprint D4).

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Engine for `INFER CLAIM_ORIGIN` — trace a claim back to its earliest appearance.
pub struct ClaimOriginEngine;

impl InferenceEngine for ClaimOriginEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ClaimOrigin
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(10)
    }

    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let claim_id_str = job
            .parameters
            .get("claim_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("Missing claim_id parameter".into()))?;
        let claim_id: Uuid = claim_id_str
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid claim UUID: {}", e)))?;

        let claim = crate::claims::detection::load_claim(hg, &claim_id)?
            .ok_or_else(|| TensaError::NotFound(format!("Claim {} not found", claim_id)))?;

        let mutations = crate::claims::track_mutations(hg, &claim)?;

        let mut chain: Vec<serde_json::Value> = Vec::new();
        chain.push(serde_json::json!({
            "claim_id": claim.id.to_string(),
            "timestamp": claim.created_at,
            "similarity": 1.0,
        }));

        for m in &mutations {
            if m.original_claim_id != claim.id {
                chain.push(serde_json::json!({
                    "claim_id": m.original_claim_id.to_string(),
                    "similarity": 1.0 - m.embedding_drift,
                    "mutation_type": m.mutation_type,
                }));
            }
        }

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: self.job_type(),
            target_id: job.target_id,
            result: serde_json::json!({
                "claim_id": claim_id.to_string(),
                "chain_length": chain.len(),
                "chain": chain,
            }),
            confidence: 1.0,
            explanation: Some(format!(
                "Traced origin of claim {} through {} appearances",
                claim_id,
                chain.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Engine for `INFER CLAIM_MATCH` — match claims against fact-checks.
pub struct ClaimMatchEngine;

impl InferenceEngine for ClaimMatchEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ClaimMatch
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(20)
    }

    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("Missing narrative_id parameter".into()))?;

        let min_similarity = job
            .parameters
            .get("min_similarity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let claims = crate::claims::detection::list_claims_for_narrative(hg, narrative_id)?;
        let mut all_matches = Vec::new();

        for claim in &claims {
            let matches = crate::claims::match_claim(hg, claim, min_similarity)?;
            for m in matches {
                all_matches.push(serde_json::json!({
                    "claim_id": m.claim_id.to_string(),
                    "fact_check_id": m.fact_check_id.to_string(),
                    "similarity": m.similarity,
                    "method": m.method,
                }));
            }
        }

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: self.job_type(),
            target_id: job.target_id,
            result: serde_json::json!({
                "narrative_id": narrative_id,
                "claims_checked": claims.len(),
                "matches_found": all_matches.len(),
                "matches": all_matches,
            }),
            confidence: 1.0,
            explanation: Some(format!(
                "Matched {} claims, found {} fact-check matches",
                claims.len(),
                all_matches.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}
