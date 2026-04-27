//! EATH Extension Phase 16c — `POST /synth/opinion-significance` plus
//! `GET /synth/opinion-significance/{narrative_id}/{run_id}` and
//! `GET /synth/opinion-significance/{narrative_id}` listings.
//!
//! Mirrors Phase 14's [`super::bistability_significance`] structurally:
//! same job envelope shape, same default-models list, but submits a
//! [`InferenceJobType::SurrogateOpinionSignificance`] job and reads from the
//! disjoint `syn/opinion_sig/` KV slice.
//!
//! Default `models` when the body omits the field: `["eath", "nudhy"]`.
//! The engine validates each requested model against the registry and
//! returns a helpful error when one is missing — this is the Phase 16c
//! "degrade gracefully without Phase 13" contract; the REST handler doesn't
//! need extra logic beyond submitting the job.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::synth::dual_significance_engine::DEFAULT_MODELS;
use crate::synth::significance::{K_DEFAULT, K_MAX};
use crate::synth::{list_opinion_significance_reports, load_opinion_significance_report};
use crate::types::InferenceJobType;

use super::submit_synth_job;

/// Default page size for the listing endpoint when `?limit=` is omitted.
const DEFAULT_REPORT_PAGE_LIMIT: usize = 50;

// ── POST /synth/opinion-significance ────────────────────────────────────────

#[derive(Deserialize)]
pub struct OpinionSignificanceBody {
    pub narrative_id: String,
    /// `OpinionDynamicsParams` blob — see
    /// [`crate::analysis::opinion_dynamics::OpinionDynamicsParams`]. May be
    /// `null`/omitted to use the engine's documented defaults.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Number of synthetic samples per model. Defaults to 50; clamped at K_MAX.
    #[serde(default)]
    pub k: Option<u16>,
    /// Surrogate models to compare against. Defaults to `["eath", "nudhy"]`
    /// when omitted or empty. Engine validates each entry against the
    /// registry; missing models return a helpful error rather than panic.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// POST /synth/opinion-significance — submits a `SurrogateOpinionSignificance`
/// job. Returns `{ job_id, status }`.
pub async fn post_opinion_significance(
    State(state): State<Arc<AppState>>,
    Json(body): Json<OpinionSignificanceBody>,
) -> impl IntoResponse {
    if body.narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }

    let k = body.k.unwrap_or(K_DEFAULT.min(50)).clamp(1, K_MAX.min(500));

    let models: Vec<String> = match body.models {
        Some(v) if !v.is_empty() => v,
        _ => DEFAULT_MODELS.iter().map(|s| s.to_string()).collect(),
    };
    for m in &models {
        if m.is_empty() {
            return error_response(TensaError::InvalidInput(
                "models contains an empty string entry".into(),
            ))
            .into_response();
        }
    }

    let params = body.params.unwrap_or(serde_json::Value::Null);

    let job_type = InferenceJobType::SurrogateOpinionSignificance {
        narrative_id: body.narrative_id,
        params,
        k,
        models,
    };

    submit_synth_job(&state, job_type, serde_json::Value::Null)
}

// ── GET /synth/opinion-significance/{narrative_id}/{run_id} ────────────────

pub async fn get_opinion_significance_result(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, run_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_opinion_significance_report(state.hypergraph.store(), &narrative_id, &run_id) {
        Ok(Some(report)) => json_ok(&report),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!(
                    "no opinion-significance report for ('{narrative_id}', '{run_id}')"
                )
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── GET /synth/opinion-significance/{narrative_id} ─────────────────────────

#[derive(Deserialize, Default)]
pub struct OpinionSignificanceListQuery {
    pub limit: Option<usize>,
}

pub async fn list_opinion_significance_results(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<OpinionSignificanceListQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(DEFAULT_REPORT_PAGE_LIMIT).clamp(1, 1000);
    match list_opinion_significance_reports(state.hypergraph.store(), &narrative_id, limit) {
        Ok(reports) => json_ok(&reports),
        Err(e) => error_response(e).into_response(),
    }
}
