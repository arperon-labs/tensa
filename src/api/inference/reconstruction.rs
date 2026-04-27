//! `POST /inference/hypergraph-reconstruction` + companion routes.
//!
//! EATH Extension Phase 15c. Wraps the reconstruction engine submitted via
//! the worker pool, exposes the completed result via the existing
//! `JobQueue::get_result` pathway, and adds an opt-in materialization route
//! that converts inferred hyperedges (above a confidence threshold) into
//! [`Situation`](crate::types::Situation) records under a caller-chosen
//! output narrative.
//!
//! See `docs/synth_reconstruction_algorithm.md` §13.7 for the analyst
//! workflow rationale: filter by `confidence > 0.7`, not by `weight > ε`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::inference::hypergraph_reconstruction::{
    materialize_reconstruction, ReconstructionResult, DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD,
};
use crate::inference::types::InferenceJob;
use crate::types::{InferenceJobType, JobPriority, JobStatus};

/// Body for `POST /inference/hypergraph-reconstruction`.
///
/// `params` is the JSON serialization of
/// [`crate::inference::hypergraph_reconstruction::ReconstructionParams`] —
/// every field has a serde default, so callers can pass `{}` (or omit the
/// field entirely) to get the engine's recommended defaults
/// (max_order=3, observation=ParticipationRate, lambda auto-selected, etc.).
#[derive(Debug, Deserialize)]
pub struct SubmitReconstructionBody {
    /// Source narrative whose entity time-series to reconstruct from.
    /// Required.
    pub narrative_id: String,
    /// Optional partial `ReconstructionParams` blob. Missing fields fall
    /// back to engine defaults via `serde(default)`.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

/// `POST /inference/hypergraph-reconstruction` — queue a reconstruction job.
///
/// Returns `{ job_id, status: "Pending" }` with 201, mirroring `POST /jobs`
/// and the synth submission envelope.
pub async fn submit(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SubmitReconstructionBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    let job_queue = match &state.job_queue {
        Some(q) => q.clone(),
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response();
        }
    };

    // Build the variant payload + the JSON parameters mirror so the engine's
    // two-slot resolver (variant payload OR `job.parameters`) finds them.
    let params_payload = body.params.clone().unwrap_or(serde_json::json!({}));
    let job_type = InferenceJobType::HypergraphReconstruction {
        narrative_id: body.narrative_id.clone(),
        params: params_payload.clone(),
    };
    let mut params_map = serde_json::Map::new();
    params_map.insert("narrative_id".into(), serde_json::json!(body.narrative_id));
    if let serde_json::Value::Object(obj) = params_payload {
        for (k, v) in obj {
            params_map.insert(k, v);
        }
    }

    let job = InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        target_id: Uuid::now_v7(),
        parameters: serde_json::Value::Object(params_map),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };

    match job_queue.submit(job) {
        Ok(id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"job_id": id, "status": "Pending"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /inference/hypergraph-reconstruction/{job_id}` — fetch the
/// completed [`ReconstructionResult`] (404 until the job finishes).
///
/// Delegates to the existing `JobQueue::get_result` pathway so polling
/// behaviour mirrors `GET /jobs/{id}/result`.
pub async fn get_result(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q.clone(),
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response();
        }
    };
    match job_queue.get_result(&job_id) {
        Ok(result) => json_ok(&result),
        Err(TensaError::JobNotFound(_)) | Err(TensaError::NotFound(_)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no reconstruction result for job '{job_id}'")
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Body for `POST /inference/hypergraph-reconstruction/{job_id}/materialize`.
///
/// `opt_in: true` is required — materialization mutates the hypergraph by
/// creating new Situations, so callers must explicitly acknowledge that.
/// `confidence_threshold` defaults to
/// [`DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD`] (0.7) when omitted.
#[derive(Debug, Deserialize)]
pub struct MaterializeBody {
    /// Where to write the materialized situations. Required.
    pub output_narrative_id: String,
    /// Required gate: callers must pass `true` to actually create
    /// situations. Defends against accidental materialization on misrouted
    /// requests.
    pub opt_in: bool,
    /// Minimum bootstrap confidence for an inferred hyperedge to be
    /// materialized. Default `0.7` per architect §13.7.
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
}

/// `POST /inference/hypergraph-reconstruction/{job_id}/materialize` —
/// commit inferred hyperedges as [`Situation`](crate::types::Situation)s.
///
/// Returns the [`MaterializationReport`](crate::inference::hypergraph_reconstruction::MaterializationReport).
pub async fn materialize(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(body): Json<MaterializeBody>,
) -> impl IntoResponse {
    if !body.opt_in {
        return error_response(TensaError::InvalidInput(
            "materialization requires opt_in=true (defends against accidental commits)".into(),
        ))
        .into_response();
    }
    if body.output_narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput(
            "output_narrative_id is empty".into(),
        ))
        .into_response();
    }

    let job_queue = match &state.job_queue {
        Some(q) => q.clone(),
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response();
        }
    };

    // Retrieve the completed job + result.
    let job = match job_queue.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };
    let inference_result = match job_queue.get_result(&job_id) {
        Ok(r) => r,
        Err(e) => return error_response(e).into_response(),
    };

    // Pull the source narrative id off the job's variant payload (Phase 15b
    // engine writes both into the variant and into job.parameters).
    let source_narrative_id = match &job.job_type {
        InferenceJobType::HypergraphReconstruction { narrative_id, .. } if !narrative_id.is_empty() => {
            narrative_id.clone()
        }
        _ => job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    };
    if source_narrative_id.is_empty() {
        return error_response(TensaError::InferenceError(
            "job is missing source narrative_id".into(),
        ))
        .into_response();
    }

    // Decode the result payload — engine wraps it as
    // `{ kind: "reconstruction_done", result: ReconstructionResult }`.
    let payload = inference_result.result;
    let result_value = payload
        .get("result")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let result: ReconstructionResult = match serde_json::from_value(result_value) {
        Ok(r) => r,
        Err(e) => {
            return error_response(TensaError::InferenceError(format!(
                "result blob does not decode as ReconstructionResult: {e}"
            )))
            .into_response();
        }
    };

    let threshold = body
        .confidence_threshold
        .unwrap_or(DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD);

    match materialize_reconstruction(
        &state.hypergraph,
        &result,
        &source_narrative_id,
        &body.output_narrative_id,
        &job_id,
        Some(threshold),
    ) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}
