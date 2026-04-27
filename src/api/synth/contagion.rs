//! Phase 7b — `POST /synth/contagion-significance` plus
//! `GET /synth/contagion-significance/{narrative_id}/{run_id}` and
//! `GET /synth/contagion-significance/{narrative_id}` listings.
//!
//! Mirrors `super::significance` structurally — the K-loop infrastructure is
//! shared (Phase 7's [`crate::synth::significance::adapters::run_significance_pipeline`]
//! handles the per-sample dispatch). The contagion-specific bits are: the
//! [`crate::analysis::higher_order_contagion::HigherOrderSirParams`] body, the
//! dedicated job variant, and the dedicated endpoint URL pair.
//!
//! All three endpoints persist + read from the same `syn/sig/{nid}/contagion/`
//! KV slice the engine writes to via
//! [`crate::synth::significance::save_significance_report`].

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

/// Default page size — same value as the Phase 7 significance listing.
const DEFAULT_REPORT_PAGE_LIMIT: usize = 50;

// ── POST /synth/contagion-significance ──────────────────────────────────────

/// REST body for `POST /synth/contagion-significance`.
///
/// `params` is the
/// [`crate::analysis::higher_order_contagion::HigherOrderSirParams`] blob
/// (raw JSON — the engine deserializes it). Phase 7b doesn't expose every
/// field as a top-level body knob to keep the API surface tight; if a future
/// phase needs that ergonomics, the body can grow flat fields without
/// changing the wire format under `params`.
#[derive(Deserialize)]
pub struct ContagionSignificanceBody {
    /// Source narrative being tested. Required.
    pub narrative_id: String,
    /// Higher-order SIR simulation parameters.
    pub params: serde_json::Value,
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
}

/// POST /synth/contagion-significance — queue a SurrogateContagionSignificance
/// job. Returns `{ job_id, status }`.
pub async fn post_contagion_significance(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ContagionSignificanceBody>,
) -> impl IntoResponse {
    if body.narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    if body.params.is_null() {
        return error_response(TensaError::InvalidInput(
            "contagion params blob is required".into(),
        ))
        .into_response();
    }

    let k = body.k.unwrap_or(K_DEFAULT).clamp(1, K_MAX);
    let model = body
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let job_type = InferenceJobType::SurrogateContagionSignificance {
        narrative_id: body.narrative_id,
        k,
        model,
        contagion_params: body.params.clone(),
    };

    // Thread params_override through job.parameters so the engine can pick
    // it up via the same Phase 7 resolver. contagion_params lives on the
    // variant payload and is also mirrored into job.parameters for the
    // engine to find via parameters.contagion_params lookup (matches the
    // engine's resolution order).
    let mut params = serde_json::Map::new();
    if let Some(p) = body.params_override {
        if !p.is_null() {
            params.insert("params_override".into(), p);
        }
    }
    params.insert("contagion_params".into(), body.params);

    submit_synth_job(&state, job_type, serde_json::Value::Object(params))
}

// ── GET /synth/contagion-significance/{narrative_id}/{run_id} ───────────────

pub async fn get_contagion_significance_result(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, run_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_significance_report(
        state.hypergraph.store(),
        &narrative_id,
        SignificanceMetric::Contagion,
        &run_id,
    ) {
        Ok(Some(report)) => json_ok(&report),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!(
                    "no contagion-significance report for ('{narrative_id}', '{run_id}')"
                )
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── GET /synth/contagion-significance/{narrative_id} ────────────────────────

#[derive(Deserialize, Default)]
pub struct ContagionSignificanceListQuery {
    pub limit: Option<usize>,
}

pub async fn list_contagion_significance_results(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<ContagionSignificanceListQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(DEFAULT_REPORT_PAGE_LIMIT).clamp(1, 1000);
    match list_significance_reports(
        state.hypergraph.store(),
        &narrative_id,
        SignificanceMetric::Contagion,
        limit,
    ) {
        Ok(reports) => json_ok(&reports),
        Err(e) => error_response(e).into_response(),
    }
}
