//! EATH Extension Phase 14 — `POST /synth/bistability-significance` plus
//! `GET /synth/bistability-significance/{narrative_id}/{run_id}` and
//! `GET /synth/bistability-significance/{narrative_id}` listings.
//!
//! Mirrors Phase 13c's [`super::dual_significance`] structurally: same job
//! envelope shape, same default-models list, but submits a
//! [`InferenceJobType::SurrogateBistabilitySignificance`] job and reads from
//! the disjoint `syn/bistability/` KV slice.
//!
//! Default `models` when the body omits the field: `["eath", "nudhy"]` (the
//! canonical dual-null pair).

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
use crate::synth::{list_bistability_significance_reports, load_bistability_significance_report};
use crate::types::InferenceJobType;

use super::submit_synth_job;

/// Default page size for the listing endpoint when `?limit=` is omitted.
const DEFAULT_REPORT_PAGE_LIMIT: usize = 50;

// ── POST /synth/bistability-significance ────────────────────────────────────

#[derive(Deserialize)]
pub struct BistabilitySignificanceBody {
    /// Source narrative being tested. Required.
    pub narrative_id: String,
    /// `BistabilitySweepParams` blob — see
    /// [`crate::analysis::contagion_bistability::BistabilitySweepParams`].
    pub params: serde_json::Value,
    /// Number of synthetic samples per model. Defaults to 50 (smaller than
    /// the standard 100 because each sample runs an entire forward-backward
    /// sweep). Clamped at 500.
    #[serde(default)]
    pub k: Option<u16>,
    /// Surrogate models to compare against. Defaults to `["eath", "nudhy"]`
    /// when omitted or empty. Each name validated against the registry at
    /// engine entry.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// POST /synth/bistability-significance
///
/// Submits a `SurrogateBistabilitySignificance` job. Validates synchronously
/// that `narrative_id` is non-empty, `params` is a non-null JSON object, and
/// (when supplied) every entry in `models` is a non-empty string. The
/// engine performs heavier validation (registry lookup, fully-synthetic
/// refusal, sweep param parsing).
pub async fn post_bistability_significance(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BistabilitySignificanceBody>,
) -> impl IntoResponse {
    if body.narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    if body.params.is_null() {
        return error_response(TensaError::InvalidInput(
            "bistability params blob is required".into(),
        ))
        .into_response();
    }

    // Default K is smaller than significance K because each sample is a full
    // sweep — 50 × 2 models × 20 betas × 10 replicates is already 20k
    // simulations.
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

    let job_type = InferenceJobType::SurrogateBistabilitySignificance {
        narrative_id: body.narrative_id,
        params: body.params,
        k,
        models,
    };

    submit_synth_job(&state, job_type, serde_json::Value::Null)
}

// ── GET /synth/bistability-significance/{narrative_id}/{run_id} ─────────────

pub async fn get_bistability_significance_result(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, run_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_bistability_significance_report(state.hypergraph.store(), &narrative_id, &run_id) {
        Ok(Some(report)) => json_ok(&report),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!(
                    "no bistability-significance report for ('{narrative_id}', '{run_id}')"
                )
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── GET /synth/bistability-significance/{narrative_id} ──────────────────────

#[derive(Deserialize, Default)]
pub struct BistabilitySignificanceListQuery {
    pub limit: Option<usize>,
}

pub async fn list_bistability_significance_results(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<BistabilitySignificanceListQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(DEFAULT_REPORT_PAGE_LIMIT).clamp(1, 1000);
    match list_bistability_significance_reports(state.hypergraph.store(), &narrative_id, limit) {
        Ok(reports) => json_ok(&reports),
        Err(e) => error_response(e).into_response(),
    }
}
