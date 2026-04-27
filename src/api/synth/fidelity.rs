//! `/synth/fidelity/*` + `/synth/fidelity-thresholds/{narrative}` handlers.
//!
//! Reads previously-persisted FidelityReport blobs and lets clients tune the
//! per-narrative passing thresholds. Unchanged from Phase 6 except for the
//! module relocation.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::synth::fidelity::{
    load_fidelity_report, load_thresholds, save_thresholds, FidelityThresholds,
};

/// GET /synth/fidelity/{narrative_id}/{run_id}
///
/// Returns the fidelity report produced when the calibration job for `run_id`
/// completed. 404 when no report exists at that (narrative, run_id) pair.
pub async fn get_fidelity(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, run_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_fidelity_report(state.hypergraph.store(), &narrative_id, &run_id) {
        Ok(Some(report)) => json_ok(&report),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no fidelity report for ('{narrative_id}', '{run_id}')")
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /synth/fidelity-thresholds/{narrative_id}
///
/// Per-narrative custom thresholds — falls back to defaults when none have
/// been persisted. Default is the `FidelityThresholds::default()` PLACEHOLDER
/// value set; see `fidelity.rs` for the calibration-study follow-up note.
pub async fn get_fidelity_thresholds(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match load_thresholds(state.hypergraph.store(), &narrative_id) {
        Ok(thresholds) => json_ok(&thresholds),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /synth/fidelity-thresholds/{narrative_id}
///
/// Upserts the per-narrative thresholds. Body must deserialize cleanly as
/// `FidelityThresholds` — every field is `serde(default)` so partial bodies
/// fall back to defaults field-by-field.
pub async fn put_fidelity_thresholds(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let thresholds: FidelityThresholds = match serde_json::from_value(body) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("invalid FidelityThresholds body: {e}")
                })),
            )
                .into_response();
        }
    };
    match save_thresholds(state.hypergraph.store(), &narrative_id, &thresholds) {
        Ok(()) => json_ok(&serde_json::json!({"ok": true})),
        Err(e) => error_response(e).into_response(),
    }
}
