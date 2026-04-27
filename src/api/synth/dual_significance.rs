//! EATH Extension Phase 13c — `POST /synth/dual-significance` plus
//! `GET /synth/dual-significance/{narrative_id}/{metric}/{run_id}` and
//! `GET /synth/dual-significance/{narrative_id}/{metric}` listings.
//!
//! Mirrors the Phase 7 [`super::significance`] module structurally — same
//! validation rules, same job envelope shape — but submits a
//! [`InferenceJobType::SurrogateDualSignificance`] job and reads from the
//! disjoint `syn/dual_sig/` KV slice.
//!
//! Default `models` when the body omits the field: `["eath", "nudhy"]` (the
//! canonical dual-null pair). Per-model name validation lives in the engine
//! (early failure on unknown model name).

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
use crate::synth::significance::{SignificanceMetric, K_DEFAULT, K_MAX};
use crate::synth::{list_dual_significance_reports, load_dual_significance_report};
use crate::types::InferenceJobType;

use super::submit_synth_job;

/// Default page size for the listing endpoint when `?limit=` is omitted.
/// Frozen to match the Phase 7 single-significance listing.
const DEFAULT_REPORT_PAGE_LIMIT: usize = 50;

// ── POST /synth/dual-significance ───────────────────────────────────────────

#[derive(Deserialize)]
pub struct DualSignificanceBody {
    /// Source narrative being tested. Required.
    pub narrative_id: String,
    /// Metric to test: `temporal_motifs` | `communities` | `patterns`.
    /// Contagion is intentionally NOT supported on the dual surface in
    /// Phase 13c — it requires a `HigherOrderSirParams` blob; future
    /// phases may add dual contagion.
    pub metric: String,
    /// Per-model number of synthetic samples. Defaults to 100; clamped at 1000.
    #[serde(default)]
    pub k_per_model: Option<u16>,
    /// Surrogate models to compare against. Defaults to `["eath", "nudhy"]`
    /// when omitted or empty. Each name is validated against the registry
    /// at engine entry; unknown names return SynthFailure with a list of
    /// known options.
    #[serde(default)]
    pub models: Option<Vec<String>>,
    /// Optional metric-specific config (e.g. `{"max_motif_size": 4}` for
    /// temporal_motifs). Threaded into `job.parameters.metric_params`.
    #[serde(default)]
    pub metric_params: Option<serde_json::Value>,
}

/// POST /synth/dual-significance
///
/// Submits a `SurrogateDualSignificance` job. Validates synchronously that
/// `narrative_id` is non-empty, the `metric` is one of the three structural
/// values, and (when supplied) every entry in `models` is a non-empty
/// string. The engine performs the heavier validation (registry lookup,
/// fully-synthetic refusal, K-loop dispatch).
pub async fn post_dual_significance(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DualSignificanceBody>,
) -> impl IntoResponse {
    if body.narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    match SignificanceMetric::parse(&body.metric) {
        Some(SignificanceMetric::TemporalMotifs)
        | Some(SignificanceMetric::Communities)
        | Some(SignificanceMetric::Patterns) => {}
        Some(SignificanceMetric::Contagion) => {
            return error_response(TensaError::InvalidInput(
                "metric 'contagion' is not supported on the dual surface in Phase 13c; \
                 use POST /synth/contagion-significance for single-model contagion null testing"
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

    let k_per_model = body.k_per_model.unwrap_or(K_DEFAULT).clamp(1, K_MAX);

    // Default models when the caller omits the field or passes an empty list.
    let models: Vec<String> = match body.models {
        Some(v) if !v.is_empty() => v,
        _ => DEFAULT_MODELS.iter().map(|s| s.to_string()).collect(),
    };
    // Per-name shape validation — empty model names are an obvious caller
    // bug. Registry membership is checked engine-side.
    for m in &models {
        if m.is_empty() {
            return error_response(TensaError::InvalidInput(
                "models contains an empty string entry".into(),
            ))
            .into_response();
        }
    }

    let job_type = InferenceJobType::SurrogateDualSignificance {
        narrative_id: body.narrative_id,
        metric: body.metric,
        k_per_model,
        models,
    };

    let mut params = serde_json::Map::new();
    if let Some(mp) = body.metric_params {
        if !mp.is_null() {
            params.insert("metric_params".into(), mp);
        }
    }

    submit_synth_job(&state, job_type, serde_json::Value::Object(params))
}

// ── GET /synth/dual-significance/{narrative_id}/{metric}/{run_id} ──────────

pub async fn get_dual_significance_result(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, metric, run_id)): Path<(String, String, Uuid)>,
) -> impl IntoResponse {
    match load_dual_significance_report(state.hypergraph.store(), &narrative_id, &metric, &run_id) {
        Ok(Some(report)) => json_ok(&report),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!(
                    "no dual-significance report for ('{narrative_id}', '{metric}', '{run_id}')"
                )
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── GET /synth/dual-significance/{narrative_id}/{metric} ────────────────────

#[derive(Deserialize, Default)]
pub struct DualSignificanceListQuery {
    pub limit: Option<usize>,
}

pub async fn list_dual_significance_results(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, metric)): Path<(String, String)>,
    Query(q): Query<DualSignificanceListQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(DEFAULT_REPORT_PAGE_LIMIT).clamp(1, 1000);
    match list_dual_significance_reports(state.hypergraph.store(), &narrative_id, &metric, limit) {
        Ok(reports) => json_ok(&reports),
        Err(e) => error_response(e).into_response(),
    }
}
