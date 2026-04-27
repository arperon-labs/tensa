//! Import endpoints for structured JSON/CSV data (Sprint P3.6 — F-CE2).

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::routes::error_response;
use crate::api::server::AppState;
use crate::ingestion::gate::ConfidenceGate;
use crate::ingestion::structured::{
    parse_csv_to_import, process_structured_import, CsvImportRequest, ImportReport,
    StructuredImport,
};

/// Run import and return the response.
fn _import_response(report: crate::error::Result<ImportReport>) -> axum::response::Response {
    match report {
        Ok(r) => (StatusCode::OK, Json(r)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /import/json — Import structured data from JSON payload.
/// Query params: ?analyze=true triggers community detection + summary generation after import.
pub async fn import_json(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(body): Json<StructuredImport>,
) -> impl IntoResponse {
    let gate = ConfidenceGate::default();
    let narrative_id = body.narrative_id.clone();
    let result =
        process_structured_import(&body, &state.hypergraph, &gate, &state.validation_queue);
    let analyze = params.get("analyze").map(|v| v == "true").unwrap_or(false);
    match result {
        Ok(report) => {
            if analyze {
                if let Some(nid) = narrative_id.as_deref() {
                    run_post_import_analysis(&state, nid);
                }
            }
            (StatusCode::OK, Json(report)).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /import/csv — Import entity data from CSV with column mapping.
/// Query params: ?analyze=true triggers community detection + summary generation after import.
pub async fn import_csv(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(body): Json<CsvImportRequest>,
) -> impl IntoResponse {
    let narrative_id = body.narrative_id.clone();
    let parsed = match parse_csv_to_import(&body.csv_data, &body.mapping, body.narrative_id.clone())
    {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };

    let gate = ConfidenceGate::default();
    let result =
        process_structured_import(&parsed, &state.hypergraph, &gate, &state.validation_queue);
    let analyze = params.get("analyze").map(|v| v == "true").unwrap_or(false);
    match result {
        Ok(report) => {
            if analyze {
                if let Some(nid) = narrative_id.as_deref() {
                    run_post_import_analysis(&state, nid);
                }
            }
            (StatusCode::OK, Json(report)).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /import/stix — Import STIX 2.1 JSON bundle.
pub async fn import_stix(State(state): State<Arc<AppState>>, body: String) -> impl IntoResponse {
    match crate::ingestion::stix::import_stix_bundle(&state.hypergraph, &body, "stix-import") {
        Ok(report) => (StatusCode::OK, Json(report)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Run community detection + summary generation after data import (BYOG pipeline).
fn run_post_import_analysis(state: &Arc<AppState>, narrative_id: &str) {
    // Run centrality analysis (includes Leiden community detection + hierarchy)
    if let Err(e) = crate::analysis::centrality::run_centrality(&state.hypergraph, narrative_id) {
        tracing::warn!("Post-import centrality analysis failed: {}", e);
        return;
    }
    // Generate community summaries if extractor is available
    let ext_guard = state.extractor.read().unwrap();
    if let Some(ext) = ext_guard.as_ref() {
        if let Err(e) = crate::analysis::community::generate_summaries(
            narrative_id,
            &state.hypergraph,
            ext.as_ref(),
            state.hypergraph.store(),
        ) {
            tracing::warn!("Post-import community summary generation failed: {}", e);
        }
    }
}
