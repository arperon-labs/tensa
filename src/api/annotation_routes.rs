//! Sprint W12 — REST for annotations / footnotes / citations.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::writer::annotation::{
    create_annotation, delete_annotation, get_annotation, list_annotations_for_scenes,
    list_annotations_for_situation, reconcile_spans_after_edit, update_annotation, Annotation,
    AnnotationKind, AnnotationPatch,
};

#[derive(Debug, Deserialize)]
pub struct CreateAnnotationBody {
    pub kind: AnnotationKind,
    pub span: (usize, usize),
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub source_id: Option<Uuid>,
    #[serde(default)]
    pub chunk_id: Option<Uuid>,
    #[serde(default)]
    pub author: Option<String>,
}

/// POST /situations/:id/annotations
pub async fn create_for_situation(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<CreateAnnotationBody>,
) -> impl IntoResponse {
    // Validate the scene exists.
    if let Err(e) = state.hypergraph.get_situation(&situation_id) {
        return error_response(e).into_response();
    }
    let ann = Annotation {
        id: Uuid::nil(),
        situation_id,
        kind: body.kind,
        span: body.span,
        body: body.body,
        source_id: body.source_id,
        chunk_id: body.chunk_id,
        author: body.author,
        detached: false,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    match create_annotation(state.hypergraph.store(), ann) {
        Ok(a) => json_ok(&a),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /situations/:id/annotations
pub async fn list_for_situation(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
) -> impl IntoResponse {
    match list_annotations_for_situation(state.hypergraph.store(), &situation_id) {
        Ok(anns) => json_ok(&anns),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/annotations — flatten every scene's annotations in
/// a single server-side call so MCP/HTTP callers don't issue N+1 round-trips.
pub async fn list_for_narrative(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let sits = match state.hypergraph.list_situations_by_narrative(&narrative_id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let scene_ids: std::collections::HashSet<Uuid> = sits.iter().map(|s| s.id).collect();
    match list_annotations_for_scenes(state.hypergraph.store(), &scene_ids) {
        Ok(anns) => json_ok(&anns),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /annotations/:id
pub async fn get_one(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match get_annotation(state.hypergraph.store(), &id) {
        Ok(a) => json_ok(&a),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /annotations/:id
pub async fn update_one(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(patch): Json<AnnotationPatch>,
) -> impl IntoResponse {
    match update_annotation(state.hypergraph.store(), &id, patch) {
        Ok(a) => json_ok(&a),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /annotations/:id
pub async fn delete_one(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match delete_annotation(state.hypergraph.store(), &id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /situations/:id/annotations/reconcile — after a prose edit, try to
/// re-anchor annotations to the new text. Body: `{old_prose, new_prose}`.
#[derive(Debug, Deserialize)]
pub struct ReconcileBody {
    pub old_prose: String,
    pub new_prose: String,
}

pub async fn reconcile(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<ReconcileBody>,
) -> impl IntoResponse {
    if body.old_prose.is_empty() && body.new_prose.is_empty() {
        return error_response(TensaError::InvalidQuery(
            "both old_prose and new_prose are empty".into(),
        ))
        .into_response();
    }
    match reconcile_spans_after_edit(
        state.hypergraph.store(),
        &situation_id,
        &body.old_prose,
        &body.new_prose,
    ) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}
