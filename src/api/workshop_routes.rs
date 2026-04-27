//! Workshop endpoints — Sprint W3 (v0.49.2).
//!
//! - `POST /narratives/:id/workshop/estimate` — cost estimate only
//! - `POST /narratives/:id/workshop/run`      — run analysis, persist report
//! - `GET  /narratives/:id/workshop/reports`  — list past reports
//! - `GET  /workshop/reports/:report_id`      — fetch a report

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::workshop::{
    estimate_cost, get_report, list_reports, run_workshop, WorkshopRequest, WorkshopTier,
};

fn registry_for(state: &AppState) -> NarrativeRegistry {
    NarrativeRegistry::new(state.hypergraph.store_arc())
}

pub async fn estimate_handler(
    State(_state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(mut body): Json<WorkshopRequest>,
) -> impl IntoResponse {
    body.narrative_id = narrative_id;
    json_ok(&estimate_cost(&body))
}

pub async fn run_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(mut body): Json<WorkshopRequest>,
) -> impl IntoResponse {
    body.narrative_id = narrative_id;
    let registry = registry_for(&state);

    // Standard tier needs an extractor. Cheap/Deep don't.
    let needs_llm = matches!(body.tier, WorkshopTier::Standard);
    let extractor_lock = if needs_llm {
        match state.extractor.read() {
            Ok(g) => Some(g),
            Err(_) => {
                return error_response(TensaError::Internal("extractor lock poisoned".into()))
                    .into_response()
            }
        }
    } else {
        None
    };
    let extractor_session: Option<&dyn crate::ingestion::llm::SessionCapableExtractor> =
        if let Some(guard) = extractor_lock.as_ref() {
            match guard.as_ref() {
                Some(arc) => match arc.as_session() {
                    Some(s) => Some(s),
                    None => {
                        return error_response(TensaError::LlmError(
                            "active LLM provider does not support session-style calls (required for Standard tier)".into(),
                        ))
                        .into_response()
                    }
                },
                None => {
                    return error_response(TensaError::LlmError(
                        "no LLM extractor configured (see /settings/llm) \u{2014} required for Standard tier".into(),
                    ))
                    .into_response()
                }
            }
        } else {
            None
        };

    match run_workshop(&state.hypergraph, &registry, extractor_session, body) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn list_reports_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match list_reports(state.hypergraph.store(), &narrative_id) {
        Ok(list) => json_ok(&list),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn get_report_handler(
    State(state): State<Arc<AppState>>,
    Path(report_id): Path<Uuid>,
) -> impl IntoResponse {
    match get_report(state.hypergraph.store(), &report_id) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}
