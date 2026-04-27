//! Inference engines for the disinfo fingerprints (Sprint D1).
//!
//! Both engines wrap synchronous compute functions in
//! [`super::fingerprints`]. The job queue is heavyweight overkill for a
//! single fingerprint compute, but routing through the engine layer lets
//! TensaQL `INFER BEHAVIORAL_FINGERPRINT FOR e:Actor ...` and
//! `INFER DISINFO_FINGERPRINT FOR n:Narrative ...` work end-to-end.
//!
//! Job parameters expected:
//! - `BehavioralFingerprintEngine`: `{ actor_id: "<uuid>" }` (or `target_id`).
//! - `DisinfoFingerprintEngine`: `{ narrative_id: "<id>" }`.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

use super::fingerprints::{
    compute_behavioral_fingerprint, compute_disinfo_fingerprint, store_behavioral_fingerprint,
    store_disinfo_fingerprint,
};

pub struct BehavioralFingerprintEngine;

impl InferenceEngine for BehavioralFingerprintEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::BehavioralFingerprint
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        // Reads a handful of situations; lightweight.
        Ok(200)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let actor_id = resolve_actor_id(job)?;
        let fp = compute_behavioral_fingerprint(hypergraph, &actor_id)?;
        store_behavioral_fingerprint(hypergraph, &fp)?;
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::BehavioralFingerprint,
            target_id: actor_id,
            result: serde_json::to_value(&fp)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: 1.0,
            explanation: Some(format!(
                "Behavioral fingerprint computed: {}/{} axes filled",
                fp.computed_axes(),
                super::fingerprints::BEHAVIORAL_AXIS_COUNT
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

pub struct DisinfoFingerprintEngine;

impl InferenceEngine for DisinfoFingerprintEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::DisinfoFingerprint
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(500)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TensaError::InferenceError(
                    "DisinfoFingerprint job requires narrative_id parameter".into(),
                )
            })?;
        let fp = compute_disinfo_fingerprint(hypergraph, narrative_id)?;
        store_disinfo_fingerprint(hypergraph, &fp)?;
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::DisinfoFingerprint,
            target_id: job.target_id,
            result: serde_json::to_value(&fp)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: 1.0,
            explanation: Some(format!(
                "Disinfo fingerprint computed: {}/{} axes filled",
                fp.computed_axes(),
                super::fingerprints::DISINFO_AXIS_COUNT
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

fn resolve_actor_id(job: &InferenceJob) -> Result<Uuid> {
    if let Some(s) = job.parameters.get("actor_id").and_then(|v| v.as_str()) {
        return Uuid::parse_str(s)
            .map_err(|e| TensaError::InferenceError(format!("invalid actor_id '{s}': {e}")));
    }
    if let Some(s) = job.parameters.get("target_id").and_then(|v| v.as_str()) {
        return Uuid::parse_str(s)
            .map_err(|e| TensaError::InferenceError(format!("invalid target_id '{s}': {e}")));
    }
    if !job.target_id.is_nil() {
        return Ok(job.target_id);
    }
    Err(TensaError::InferenceError(
        "BehavioralFingerprint job requires actor_id or target_id".into(),
    ))
}

// ─── Sprint D2: spread engines ─────────────────────────────────

/// Engine for `INFER SPREAD_VELOCITY FOR n:Narrative ...`.
///
/// Job parameters:
/// - `narrative_id` (string, required)
/// - `fact` (string, required) — the KnowledgeFact text being modeled
/// - `about_entity` (UUID string, required) — the subject of the fact
/// - `narrative_kind` (string, optional, defaults "default") — bucket for baseline
/// - `beta_overrides` (optional map { platform_str: f64 })
///
/// Runs `run_smir_contagion`, detects cross-platform jumps, then feeds each
/// per-platform R₀ through the velocity monitor. Returns the SMIR result with
/// the firing alerts (if any) attached for the API/Studio response.
pub struct SpreadVelocityEngine;

impl InferenceEngine for SpreadVelocityEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::SpreadVelocity
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(4000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        use crate::analysis::contagion::compute_spread_r0_payload;

        let (narrative_id, fact, about_entity) = parse_smir_params(job)?;
        let narrative_kind = job
            .parameters
            .get("narrative_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let beta_overrides = parse_beta_overrides(job);

        let payload = compute_spread_r0_payload(
            hypergraph,
            &narrative_id,
            &fact,
            about_entity,
            narrative_kind,
            &beta_overrides,
        )?;

        // Surface aggregated stats for the explanation line — pull them out
        // of the JSON envelope so we don't need a second SMIR run.
        let r0_overall = payload
            .get("smir")
            .and_then(|s| s.get("r0_overall"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let platform_count = payload
            .get("smir")
            .and_then(|s| s.get("r0_by_platform"))
            .and_then(|v| v.as_object())
            .map(|m| m.len())
            .unwrap_or(0);
        let alert_count = payload
            .get("alerts")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::SpreadVelocity,
            target_id: job.target_id,
            result: payload,
            confidence: 1.0,
            explanation: Some(format!(
                "SMIR R₀ = {r0_overall:.2} across {platform_count} platforms; {alert_count} alert(s) fired"
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Engine for `INFER SPREAD_INTERVENTION ...`.
///
/// Job parameters: same as SpreadVelocity, plus `intervention` (object,
/// required) shaped per [`crate::analysis::spread_intervention::Intervention`].
pub struct SpreadInterventionEngine;

impl InferenceEngine for SpreadInterventionEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::SpreadIntervention
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        use crate::analysis::spread_intervention::{parse_intervention, simulate_intervention};

        let (narrative_id, fact, about_entity) = parse_smir_params(job)?;
        let intervention_param = job
            .parameters
            .get("intervention")
            .ok_or_else(|| TensaError::InferenceError("missing intervention parameter".into()))?;
        let intervention = parse_intervention(intervention_param)?;
        let beta_overrides = parse_beta_overrides(job);

        let projection = simulate_intervention(
            hypergraph,
            &narrative_id,
            &fact,
            about_entity,
            intervention,
            &beta_overrides,
        )?;

        let explanation = format!(
            "Projection: R₀ {:.2} → {:.2} (Δ {:.2}); audience saved {}",
            projection.baseline_r0,
            projection.projected_r0,
            projection.r0_delta,
            projection.audience_saved
        );

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::SpreadIntervention,
            target_id: job.target_id,
            result: serde_json::to_value(&projection)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: 1.0,
            explanation: Some(explanation),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

fn parse_smir_params(job: &InferenceJob) -> Result<(String, String, Uuid)> {
    let narrative_id = job
        .parameters
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TensaError::InferenceError("missing narrative_id".into()))?
        .to_string();
    let fact = job
        .parameters
        .get("fact")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TensaError::InferenceError("missing fact".into()))?
        .to_string();
    let about_entity_str = job
        .parameters
        .get("about_entity")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TensaError::InferenceError("missing about_entity".into()))?;
    let about_entity = Uuid::parse_str(about_entity_str)
        .map_err(|e| TensaError::InferenceError(format!("invalid about_entity UUID: {e}")))?;
    Ok((narrative_id, fact, about_entity))
}

fn parse_beta_overrides(job: &InferenceJob) -> Vec<(crate::types::Platform, f64)> {
    let map = match job
        .parameters
        .get("beta_overrides")
        .and_then(|v| v.as_object())
    {
        Some(m) => m,
        None => return vec![],
    };
    crate::analysis::contagion::parse_beta_overrides(
        map.iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?))),
    )
}

// ─── Sprint D3: CIB detection + superspreader engines ─────────

/// Engine for `INFER CIB FOR e:Actor WHERE e.narrative_id = "..."`.
///
/// Job parameters:
/// - `narrative_id` (string, required)
/// - `cross_platform` (bool, optional — defaults false) — when true, runs
///   [`crate::analysis::cib::detect_cross_platform_cib`] instead.
/// - `similarity_threshold` (f64, optional)
/// - `alpha` (f64, optional)
/// - `bootstrap_iter` (usize, optional)
/// - `min_cluster_size` (usize, optional)
/// - `seed` (u64, optional)
pub struct CibDetectionEngine;

impl InferenceEngine for CibDetectionEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CibDetection
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(6000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        use crate::analysis::cib::{detect_cib_clusters, detect_cross_platform_cib, CibConfig};

        let narrative_id = crate::analysis::extract_narrative_id(job)?.to_string();
        let config = CibConfig::from_json(&job.parameters);

        let cross_platform = job
            .parameters
            .get("cross_platform")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let result = if cross_platform {
            detect_cross_platform_cib(hypergraph, &narrative_id, &config)?
        } else {
            detect_cib_clusters(hypergraph, &narrative_id, &config)?
        };

        let flagged = result.clusters.len();
        let explanation = format!(
            "CIB detection: {} cluster(s) flagged at α = {:.3} over {} actors",
            flagged, config.alpha, result.network_size
        );
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::CibDetection,
            target_id: job.target_id,
            result: serde_json::to_value(&result)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: 1.0,
            explanation: Some(explanation),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Engine for `INFER SUPERSPREADERS FOR n:Narrative ... METHOD pagerank TOP 10`.
///
/// Job parameters:
/// - `narrative_id` (string, required)
/// - `method` (string, optional — `"pagerank"` | `"eigenvector"` | `"harmonic"`; default pagerank)
/// - `top_n` (usize, optional — default 10)
pub struct SuperspreadersEngine;

impl InferenceEngine for SuperspreadersEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::Superspreaders
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        use std::str::FromStr;

        use crate::analysis::cib::{rank_superspreaders, SuperspreaderMethod};

        let narrative_id = crate::analysis::extract_narrative_id(job)?.to_string();
        let method = match job.parameters.get("method").and_then(|v| v.as_str()) {
            Some(s) => SuperspreaderMethod::from_str(s)?,
            None => SuperspreaderMethod::PageRank,
        };
        let top_n = job
            .parameters
            .get("top_n")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let ranking = rank_superspreaders(hypergraph, &narrative_id, method, top_n)?;
        let explanation = format!(
            "Superspreader ranking ({}): top {} of {} actors",
            method.as_str(),
            ranking.scores.len(),
            ranking.network_size
        );
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::Superspreaders,
            target_id: job.target_id,
            result: serde_json::to_value(&ranking)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            confidence: 1.0,
            explanation: Some(explanation),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, make_hg};
    use crate::inference::types::InferenceJob;
    use crate::types::JobPriority;

    fn job_for(_narrative: &str, params: serde_json::Value) -> InferenceJob {
        InferenceJob {
            id: "test-job".into(),
            job_type: InferenceJobType::BehavioralFingerprint,
            target_id: Uuid::nil(),
            parameters: params,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    #[test]
    fn behavioral_engine_runs_with_actor_id_param() {
        let hg = make_hg();
        let actor = add_entity(&hg, "Alice", "test-narr");
        let engine = BehavioralFingerprintEngine;
        let job = job_for(
            "test-narr",
            serde_json::json!({"actor_id": actor.to_string()}),
        );
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.job_type, InferenceJobType::BehavioralFingerprint);
        assert_eq!(result.target_id, actor);
        assert_eq!(result.status, JobStatus::Completed);
    }

    #[test]
    fn behavioral_engine_falls_back_to_target_id() {
        let hg = make_hg();
        let actor = add_entity(&hg, "Alice", "test-narr");
        let engine = BehavioralFingerprintEngine;
        let mut job = job_for("test-narr", serde_json::json!({}));
        job.target_id = actor;
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.target_id, actor);
    }

    #[test]
    fn behavioral_engine_errors_without_id() {
        let hg = make_hg();
        let engine = BehavioralFingerprintEngine;
        let job = job_for("test-narr", serde_json::json!({}));
        // job.target_id is nil here.
        let err = engine.execute(&job, &hg).unwrap_err();
        match err {
            TensaError::InferenceError(msg) => assert!(msg.contains("actor_id")),
            other => panic!("expected InferenceError, got {other:?}"),
        }
    }

    #[test]
    fn disinfo_engine_runs_on_empty_narrative() {
        let hg = make_hg();
        let engine = DisinfoFingerprintEngine;
        let job = job_for(
            "empty-narr",
            serde_json::json!({"narrative_id": "empty-narr"}),
        );
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.job_type, InferenceJobType::DisinfoFingerprint);
        assert_eq!(result.status, JobStatus::Completed);
    }

    #[test]
    fn disinfo_engine_errors_without_narrative_id() {
        let hg = make_hg();
        let engine = DisinfoFingerprintEngine;
        let job = job_for("ignored", serde_json::json!({}));
        assert!(engine.execute(&job, &hg).is_err());
    }
}
