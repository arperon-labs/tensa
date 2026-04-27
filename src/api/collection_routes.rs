//! Sprint W15 — REST routes for collections (saved searches).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::writer::collection::{
    create_collection, delete_collection, get_collection, list_collections_for_narrative,
    resolve_collection, update_collection, Collection, CollectionPatch, CollectionQuery,
};

#[derive(Debug, Deserialize)]
pub struct CreateCollectionBody {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub query: CollectionQuery,
}

/// POST /narratives/:id/collections
pub async fn create_route(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<CreateCollectionBody>,
) -> impl IntoResponse {
    let c = Collection {
        id: Uuid::nil(),
        narrative_id,
        name: body.name,
        description: body.description,
        query: body.query,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    match create_collection(state.hypergraph.store(), c) {
        Ok(saved) => json_ok(&saved),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/collections
pub async fn list_route(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match list_collections_for_narrative(state.hypergraph.store(), &narrative_id) {
        Ok(list) => json_ok(&list),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /collections/:id
pub async fn get_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match get_collection(state.hypergraph.store(), &id) {
        Ok(c) => json_ok(&c),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /collections/:id
pub async fn update_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(patch): Json<CollectionPatch>,
) -> impl IntoResponse {
    match update_collection(state.hypergraph.store(), &id, patch) {
        Ok(c) => json_ok(&c),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /collections/:id
pub async fn delete_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match delete_collection(state.hypergraph.store(), &id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /collections/:id/resolve
pub async fn resolve_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match get_collection(state.hypergraph.store(), &id) {
        Ok(c) => match resolve_collection(&state.hypergraph, &c) {
            Ok(r) => json_ok(&r),
            Err(e) => error_response(e).into_response(),
        },
        Err(e) => error_response(e).into_response(),
    }
}
