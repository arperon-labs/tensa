//! HTTP handlers for source intelligence endpoints.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::error_response;
use crate::api::server::AppState;
use crate::source::*;

// ─── Source CRUD ─────────────────────────────────────────────

/// POST /sources
pub async fn create_source(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let source: Source = match serde_json::from_value(body) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    match state.hypergraph.create_source(source) {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Query params for GET /sources: supports pagination plus an optional
/// `narrative_id` filter that restricts the result set to sources
/// attributed to any entity or situation in that narrative.
#[derive(Deserialize)]
pub struct ListSourcesParams {
    pub limit: Option<usize>,
    pub after: Option<String>,
    pub narrative_id: Option<String>,
}

/// GET /sources — narrative filter returns the full matching set (no
/// pagination cursor) since the filter itself bounds the result size.
pub async fn list_sources(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<ListSourcesParams>,
) -> impl IntoResponse {
    if let Some(nid) = params.narrative_id.as_deref() {
        return match state.hypergraph.list_sources_for_narrative(nid) {
            Ok(sources) => {
                let resp = crate::api::routes::PaginatedResponse {
                    data: sources,
                    next_cursor: None,
                };
                crate::api::routes::json_ok(&resp)
            }
            Err(e) => error_response(e).into_response(),
        };
    }

    let limit = crate::api::routes::clamp_limit(params.limit);
    let after_uuid = params.after.as_deref().and_then(|s| s.parse::<Uuid>().ok());
    match state
        .hypergraph
        .list_sources_paginated(limit, after_uuid.as_ref())
    {
        Ok((sources, next_cursor)) => {
            let resp = crate::api::routes::PaginatedResponse {
                data: sources,
                next_cursor: next_cursor.map(|id| id.to_string()),
            };
            crate::api::routes::json_ok(&resp)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /sources/:id
pub async fn get_source(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.get_source(&id) {
        Ok(source) => crate::api::routes::json_ok(&source),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /sources/:id
pub async fn update_source(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Pre-validate complex fields before entering the updater closure
    let bias_profile = if let Some(bp) = body.get("bias_profile") {
        match serde_json::from_value::<BiasProfile>(bp.clone()) {
            Ok(profile) => Some(profile),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("Invalid bias_profile: {}", e)})),
                )
                    .into_response()
            }
        }
    } else {
        None
    };
    let tags = if let Some(t) = body.get("tags") {
        match serde_json::from_value::<Vec<String>>(t.clone()) {
            Ok(tags) => Some(tags),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": format!("Invalid tags: {}", e)})),
                )
                    .into_response()
            }
        }
    } else {
        None
    };

    let trust_changed = body.get("trust_score").and_then(|v| v.as_f64()).is_some();

    match state.hypergraph.update_source(&id, |source| {
        if let Some(name) = body.get("name").and_then(|v| v.as_str()) {
            source.name = name.to_string();
        }
        if let Some(trust) = body.get("trust_score").and_then(|v| v.as_f64()) {
            source.trust_score = trust as f32;
        }
        if let Some(url) = body.get("url") {
            source.url = url.as_str().map(|s| s.to_string());
        }
        if let Some(desc) = body.get("description") {
            source.description = desc.as_str().map(|s| s.to_string());
        }
        if let Some(profile) = bias_profile {
            source.bias_profile = profile;
        }
        if let Some(t) = tags {
            source.tags = t;
        }
        if let Some(v) = body.get("author") {
            source.author = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("publication") {
            source.publication = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("ingested_by") {
            source.ingested_by = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("ingestion_notes") {
            source.ingestion_notes = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("publication_date") {
            source.publication_date = v.as_str().and_then(|s| s.parse().ok());
        }
    }) {
        Ok(source) => {
            // Best-effort: propagate trust changes to all attributed targets
            if trust_changed {
                if let Err(e) = state.hypergraph.propagate_source_trust_change(&id) {
                    tracing::warn!("Failed to propagate trust change for source {}: {}", id, e);
                }
            }
            crate::api::routes::json_ok(&source)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /sources/:id
pub async fn delete_source(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.delete_source(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Source Attribution ──────────────────────────────────────

/// POST /sources/:id/attributions
pub async fn add_attribution(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut attr: SourceAttribution = match serde_json::from_value(body) {
        Ok(a) => a,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    attr.source_id = source_id; // path param takes precedence
    match state.hypergraph.add_attribution(attr) {
        Ok(()) => StatusCode::CREATED.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /sources/:id/attributions
pub async fn list_attributions_for_source(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.get_attributions_for_source(&source_id) {
        Ok(attrs) => crate::api::routes::json_ok(&attrs),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /entities/:id/attributions or /situations/:id/attributions
pub async fn list_attributions_for_target(
    State(state): State<Arc<AppState>>,
    Path(target_id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.get_attributions_for_target(&target_id) {
        Ok(attrs) => crate::api::routes::json_ok(&attrs),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Deserialize)]
pub struct AttributionPath {
    pub source_id: Uuid,
    pub target_id: Uuid,
}

/// DELETE /sources/:source_id/attributions/:target_id
pub async fn remove_attribution(
    State(state): State<Arc<AppState>>,
    Path(path): Path<AttributionPath>,
) -> impl IntoResponse {
    match state
        .hypergraph
        .remove_attribution(&path.source_id, &path.target_id)
    {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Contention ──────────────────────────────────────────────

/// POST /contentions
pub async fn add_contention(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let link: ContentionLink = match serde_json::from_value(body) {
        Ok(l) => l,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    match state.hypergraph.add_contention(link) {
        Ok(()) => StatusCode::CREATED.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /situations/:id/contentions
pub async fn list_contentions(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.get_contentions_for_situation(&id) {
        Ok(links) => crate::api::routes::json_ok(&links),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Deserialize)]
pub struct ResolveContentionBody {
    pub situation_a: Uuid,
    pub situation_b: Uuid,
    pub resolution: String,
}

/// POST /contentions/resolve
pub async fn resolve_contention(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ResolveContentionBody>,
) -> impl IntoResponse {
    match state
        .hypergraph
        .resolve_contention(&body.situation_a, &body.situation_b, body.resolution)
    {
        Ok(link) => crate::api::routes::json_ok(&link),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Confidence Recompute ────────────────────────────────────

fn recompute_confidence_response(
    state: &AppState,
    id: Uuid,
    current_confidence: f32,
) -> axum::response::Response {
    match state
        .hypergraph
        .recompute_confidence(&id, current_confidence)
    {
        Ok(breakdown) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": id,
                "breakdown": breakdown,
                "composite": breakdown.composite(),
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /entities/:id/recompute-confidence
pub async fn recompute_entity_confidence(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let entity = match state.hypergraph.get_entity(&id) {
        Ok(e) => e,
        Err(e) => return error_response(e).into_response(),
    };
    recompute_confidence_response(&state, id, entity.confidence)
}

/// POST /situations/:id/recompute-confidence
pub async fn recompute_situation_confidence(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let sit = match state.hypergraph.get_situation(&id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    recompute_confidence_response(&state, id, sit.confidence)
}
