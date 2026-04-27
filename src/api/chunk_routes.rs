//! HTTP handlers for chunk storage endpoints.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

/// Query parameters for listing chunks.
#[derive(Deserialize)]
pub struct ChunkListParams {
    pub job_id: Option<String>,
    pub narrative_id: Option<String>,
}

/// GET /chunks?job_id=...&narrative_id=...
pub async fn list_chunks(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ChunkListParams>,
) -> impl IntoResponse {
    if let Some(job_id) = params.job_id {
        match state.hypergraph.list_chunks_by_job(&job_id) {
            Ok(chunks) => json_ok(&chunks),
            Err(e) => error_response(e).into_response(),
        }
    } else if let Some(narrative_id) = params.narrative_id {
        match state.hypergraph.list_chunks_by_narrative(&narrative_id) {
            Ok(chunks) => json_ok(&chunks),
            Err(e) => error_response(e).into_response(),
        }
    } else {
        error_response(TensaError::InvalidQuery(
            "Provide job_id or narrative_id query parameter".into(),
        ))
        .into_response()
    }
}

/// GET /chunks/:id
pub async fn get_chunk(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(_) => {
            return error_response(TensaError::InvalidQuery(format!(
                "Invalid chunk UUID: {id}"
            )))
            .into_response()
        }
    };
    match state.hypergraph.get_chunk(&uuid) {
        Ok(chunk) => json_ok(&chunk),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/chunks
pub async fn list_narrative_chunks(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.hypergraph.list_chunks_by_narrative(&id) {
        Ok(chunks) => json_ok(&chunks),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /ingest/jobs/:id/chunks
pub async fn list_job_chunks(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.hypergraph.list_chunks_by_job(&id) {
        Ok(chunks) => json_ok(&chunks),
        Err(e) => error_response(e).into_response(),
    }
}
