//! Phase 7 significance handlers: `POST /synth/significance` plus
//! `GET /synth/significance/{narrative}/{metric}/{run_id}` and
//! `GET /synth/significance/{narrative}/{metric}` listings.
//!
//! All three are job-aware: POST queues a `SurrogateSignificance` job through
//! the existing worker pool (the engine handles the K-loop in
//! `crate::synth::significance::SurrogateSignificanceEngine`); the two GETs
//! read previously-completed reports out of KV at
//! `syn/sig/{narrative}/{metric}/{run_id_BE}`.

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
use crate::synth::significance::{
    list_significance_reports, load_significance_report, SignificanceMetric, K_DEFAULT, K_MAX,
};
use crate::types::InferenceJobType;

use super::{submit_synth_job, DEFAULT_MODEL};

/// Default page size for the listing endpoint when `?limit=` is omitted.
const DEFAULT_REPORT_PAGE_LIMIT: usize = 50;

// ── POST /synth/significance ────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SignificanceBody {
    /// Source narrative being tested. Required.
    pub narrative_id: String,
    /// `temporal_motifs` | `communities` | `patterns`.
    pub metric: String,
    /// Number of synthetic samples. Defaults to 100; clamped at 1000.
    #[serde(default)]
    pub k: Option<u16>,
    /// Surrogate model. Defaults to `"eath"`.
    #[serde(default)]
    pub model: Option<String>,
    /// Inline EathParams override. When present, the engine skips
    /// `load_params` and uses this directly. Wins over auto-calibration.
    #[serde(default)]
    pub params_override: Option<serde_json::Value>,
    /// Optional metric-specific config (e.g. `{"max_motif_size": 4}` for
    /// temporal_motifs). Threaded into `job.parameters.metric_params`.
    #[serde(default)]
    pub metric_params: Option<serde_json::Value>,
}

/// POST /synth/significance
///
/// Submits a `SurrogateSignificance` job. Validates synchronously that the
/// narrative_id is non-empty and the metric is one of the three known
/// values; returns 400 otherwise. The engine performs the heavier checks
/// (synthetic-narrative refusal, K-loop, persistence).
pub async fn post_significance(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SignificanceBody>,
) -> impl IntoResponse {
    if body.narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    match SignificanceMetric::parse(&body.metric) {
        // Phase 7 metrics route through this engine.
        Some(SignificanceMetric::TemporalMotifs)
        | Some(SignificanceMetric::Communities)
        | Some(SignificanceMetric::Patterns) => {}
        // Phase 7b — contagion has a dedicated engine + endpoint.
        Some(SignificanceMetric::Contagion) => {
            return error_response(TensaError::InvalidInput(
                "metric 'contagion' must be submitted via POST /synth/contagion-significance"
                    .into(),
            ))
            .into_response();
        }
        None => {
            return error_response(TensaError::InvalidInput(format!(
                "unknown metric '{}'; expected one of: temporal_motifs, communities, patterns",
                body.metric
            )))
            .into_response();
        }
    }

    let k = body.k.unwrap_or(K_DEFAULT).clamp(1, K_MAX);
    let model = body
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let job_type = InferenceJobType::SurrogateSignificance {
        narrative_id: body.narrative_id,
        metric_kind: body.metric,
        k,
        model,
    };

    // params_override + metric_params piggyback on `job.parameters` because
    // the variant doesn't carry them — the engine reads both from there.
    let mut params = serde_json::Map::new();
    if let Some(p) = body.params_override {
        if !p.is_null() {
            params.insert("params_override".into(), p);
        }
    }
    if let Some(mp) = body.metric_params {
        if !mp.is_null() {
            params.insert("metric_params".into(), mp);
        }
    }

    submit_synth_job(&state, job_type, serde_json::Value::Object(params))
}

// ── GET /synth/significance/{narrative_id}/{metric}/{run_id} ────────────────

/// GET handler for a single completed report. 404 when no report exists
/// at the (narrative, metric, run_id) triple — typically because the job
/// hasn't completed yet, or `metric` doesn't match what the job emitted.
pub async fn get_significance_result(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, metric, run_id)): Path<(String, String, Uuid)>,
) -> impl IntoResponse {
    let parsed = match SignificanceMetric::parse(&metric) {
        Some(m) => m,
        None => {
            return error_response(TensaError::InvalidInput(format!(
                "unknown metric '{metric}'"
            )))
            .into_response();
        }
    };
    match load_significance_report(state.hypergraph.store(), &narrative_id, parsed, &run_id) {
        Ok(Some(report)) => json_ok(&report),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!(
                    "no significance report for ('{narrative_id}', '{metric}', '{run_id}')"
                )
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── GET /synth/significance/{narrative_id}/{metric} ─────────────────────────

#[derive(Deserialize, Default)]
pub struct SignificanceListQuery {
    pub limit: Option<usize>,
}

/// Returns reports for `(narrative_id, metric)` newest first. `?limit=`
/// clamps to [1, 1000]; defaults to 50.
pub async fn list_significance_results(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, metric)): Path<(String, String)>,
    Query(q): Query<SignificanceListQuery>,
) -> impl IntoResponse {
    let parsed = match SignificanceMetric::parse(&metric) {
        Some(m) => m,
        None => {
            return error_response(TensaError::InvalidInput(format!(
                "unknown metric '{metric}'"
            )))
            .into_response();
        }
    };
    let limit = q.limit.unwrap_or(DEFAULT_REPORT_PAGE_LIMIT).clamp(1, 1000);
    match list_significance_reports(state.hypergraph.store(), &narrative_id, parsed, limit) {
        Ok(reports) => json_ok(&reports),
        Err(e) => error_response(e).into_response(),
    }
}
