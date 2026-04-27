//! Inference engines for archetype + fusion (Sprint D5).

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Engine for `INFER ARCHETYPE` — classify an actor into adversarial archetypes.
pub struct ArchetypeEngine;

impl InferenceEngine for ArchetypeEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ArchetypeClassification
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(15)
    }

    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let actor_id_str = job
            .parameters
            .get("actor_id")
            .or_else(|| job.parameters.get("target_id"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("Missing actor_id parameter".into()))?;
        let actor_id: Uuid = actor_id_str
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid actor UUID: {}", e)))?;

        let dist = crate::disinfo::archetypes::classify_actor_archetype(hg, actor_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: self.job_type(),
            target_id: job.target_id,
            result: serde_json::to_value(&dist)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: dist.confidence as f32,
            explanation: Some(format!(
                "Actor {} classified as {} with confidence {:.2}",
                actor_id, dist.primary, dist.confidence
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Engine for `INFER DISINFO_ASSESSMENT` — DS fusion of disinfo signals.
pub struct DisinfoAssessmentEngine;

impl InferenceEngine for DisinfoAssessmentEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::DisinfoAssessment
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(10)
    }

    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let target_id = job
            .parameters
            .get("target_id")
            .or_else(|| job.parameters.get("narrative_id"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("Missing target_id parameter".into()))?;

        // Parse signals from parameters
        let signals_val = job
            .parameters
            .get("signals")
            .ok_or_else(|| TensaError::InvalidQuery("Missing signals parameter".into()))?;
        let signals: Vec<crate::disinfo::fusion::DisinfoSignal> =
            serde_json::from_value(signals_val.clone())
                .map_err(|e| TensaError::InvalidQuery(format!("Invalid signals: {}", e)))?;

        let assessment = crate::disinfo::fusion::fuse_disinfo_signals(target_id, &signals)?;
        crate::disinfo::fusion::store_assessment(hg, &assessment)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: self.job_type(),
            target_id: job.target_id,
            result: serde_json::to_value(&assessment)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: assessment.confidence as f32,
            explanation: Some(format!(
                "Assessment for {}: verdict={:?}, confidence={:.2}, conflict={:.2}",
                target_id, assessment.verdict, assessment.confidence, assessment.conflict
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}
