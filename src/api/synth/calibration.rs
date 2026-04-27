//! `/synth/calibrate` + `/synth/params/{narrative}/{model}` handlers.
//!
//! Submits calibration jobs to the worker pool and exposes the params CRUD
//! surface. Unchanged from Phase 6 except for the module relocation — the
//! routing contract (paths, HTTP verbs, response envelopes) is frozen.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::synth::calibrate::{delete_params, load_params, save_params};
use crate::synth::types::EathParams;
use crate::types::InferenceJobType;

use super::{submit_synth_job, DEFAULT_MODEL};

// ── Calibration ──────────────────────────────────────────────────────────────

/// POST /synth/calibrate/{narrative_id}
///
/// Submits a `SurrogateCalibration` job to the worker pool. Default model is
/// `"eath"`. Returns the queued job id with `201 Created`, matching the
/// envelope returned by `POST /jobs`.
///
/// Body is optional — clients that want the EATH default can POST an empty
/// body. We accept any JSON value and tolerate `null`/missing-fields rather
/// than rejecting empty bodies as a 415.
pub async fn calibrate(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    body: Option<Json<serde_json::Value>>,
) -> impl IntoResponse {
    if narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }

    let model = body
        .and_then(|Json(v)| v.get("model").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    submit_synth_job(
        &state,
        InferenceJobType::SurrogateCalibration {
            narrative_id,
            model,
        },
        serde_json::json!({}),
    )
}

// ── Params CRUD ──────────────────────────────────────────────────────────────

/// GET /synth/params/{narrative_id}/{model}
///
/// Returns the persisted per-model fitted params, or 404 when none exists.
/// The persisted blob is `EathParams` (the inner model-specific shape under
/// the `SurrogateParams` envelope) — see calibrate engine.
pub async fn get_params(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, model)): Path<(String, String)>,
) -> impl IntoResponse {
    match load_params(state.hypergraph.store(), &narrative_id, &model) {
        Ok(Some(params)) => json_ok(&params),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no params for ('{narrative_id}', '{model}')")
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /synth/params/{narrative_id}/{model}
///
/// Upserts the per-model fitted params. Body must deserialize as `EathParams`
/// — extra fields are ignored, missing fields fall back to `serde(default)`.
pub async fn put_params(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, model)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let params: EathParams = match serde_json::from_value(body) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("invalid EathParams body: {e}")})),
            )
                .into_response();
        }
    };
    match save_params(state.hypergraph.store(), &narrative_id, &model, &params) {
        Ok(()) => json_ok(&serde_json::json!({"ok": true})),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /synth/params/{narrative_id}/{model}
///
/// Idempotent — succeeds even when no params exist (matches the underlying
/// KVStore contract).
pub async fn delete_params_route(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, model)): Path<(String, String)>,
) -> impl IntoResponse {
    match delete_params(state.hypergraph.store(), &narrative_id, &model) {
        Ok(()) => json_ok(&serde_json::json!({"ok": true})),
        Err(e) => error_response(e).into_response(),
    }
}
