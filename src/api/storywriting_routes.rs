//! Storywriting workspace endpoints (v0.48.0).
//!
//! Adds the endpoints that the Studio Storywriting Workspace depends on but
//! which don't fit into the existing route files:
//!
//! - `POST   /participations/bulk`
//! - `DELETE /participations/:entity_id/:situation_id`
//! - `POST   /narratives/:id/arcs`
//! - `GET    /narratives/:id/arcs`
//! - `PUT    /narratives/:id/arcs/:arc_id`
//! - `DELETE /narratives/:id/arcs/:arc_id`
//! - `POST   /narratives/:id/situations/reorder`
//!
//! The expansion of `PUT /situations/:id` to support full-field patches
//! lives in `routes.rs` (critical-path core route).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::bulk_routes::BulkItemResult;
use crate::api::routes::{deserialize_optional_option, error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::hypergraph::keys;
use crate::types::{AllenInterval, Participation, UserArc};

/// Cap for bulk operations — matches the existing bulk-entity/situation cap.
const MAX_BULK_PARTICIPATIONS: usize = 500;
/// Cap for batch reorder operations.
const MAX_REORDER_ITEMS: usize = 2000;

// ─── POST /participations/bulk ────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BulkParticipationRequest {
    pub items: Vec<Participation>,
}

#[derive(Debug, Serialize)]
pub struct BulkParticipationResponse {
    pub created: usize,
    pub errors: Vec<BulkItemResult>,
}

/// Create multiple participations in one request. Never aborts early — per-item
/// errors are reported in the response.
pub async fn bulk_create_participations(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BulkParticipationRequest>,
) -> impl IntoResponse {
    if req.items.len() > MAX_BULK_PARTICIPATIONS {
        return error_response(TensaError::InvalidQuery(format!(
            "bulk participation request exceeds maximum of {} items",
            MAX_BULK_PARTICIPATIONS
        )))
        .into_response();
    }

    let mut created = 0usize;
    let mut errors = Vec::new();
    for (i, participation) in req.items.into_iter().enumerate() {
        match state.hypergraph.add_participant(participation) {
            Ok(()) => created += 1,
            Err(e) => errors.push(BulkItemResult {
                index: i,
                id: None,
                error: Some(e.to_string()),
            }),
        }
    }

    json_ok(&BulkParticipationResponse { created, errors })
}

// ─── DELETE /participations/:entity_id/:situation_id ─────────────────

#[derive(Debug, Deserialize)]
pub struct RemoveParticipationQuery {
    pub seq: Option<u16>,
}

/// Remove a participation. With `?seq=N` deletes just that seq; without it
/// deletes all participations for the (entity, situation) pair.
pub async fn remove_participation(
    State(state): State<Arc<AppState>>,
    Path((entity_id, situation_id)): Path<(Uuid, Uuid)>,
    Query(q): Query<RemoveParticipationQuery>,
) -> impl IntoResponse {
    match state
        .hypergraph
        .remove_participant(&entity_id, &situation_id, q.seq)
    {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── UserArc CRUD ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateArcRequest {
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    pub arc_type: String,
    #[serde(default)]
    pub situation_ids: Vec<Uuid>,
    #[serde(default)]
    pub order: u32,
}

#[derive(Debug, Deserialize)]
pub struct UpdateArcRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub arc_type: Option<String>,
    #[serde(default)]
    pub situation_ids: Option<Vec<Uuid>>,
    #[serde(default)]
    pub order: Option<u32>,
}

/// Require the narrative to exist before mutating its arcs.
fn assert_narrative_exists(state: &AppState, narrative_id: &str) -> Result<(), TensaError> {
    let key = keys::narrative_key(narrative_id);
    match state.hypergraph.store().get(&key)? {
        Some(_) => Ok(()),
        None => Err(TensaError::NarrativeNotFound(narrative_id.to_string())),
    }
}

/// POST /narratives/:id/arcs — create a user-defined arc scaffold.
pub async fn create_arc(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(req): Json<CreateArcRequest>,
) -> impl IntoResponse {
    if let Err(e) = assert_narrative_exists(&state, &narrative_id) {
        return error_response(e).into_response();
    }
    if req.title.trim().is_empty() {
        return error_response(TensaError::InvalidQuery("arc title is required".into()))
            .into_response();
    }
    if req.arc_type.trim().is_empty() {
        return error_response(TensaError::InvalidQuery("arc_type is required".into()))
            .into_response();
    }
    let now = Utc::now();
    let arc = UserArc {
        id: Uuid::now_v7(),
        narrative_id: narrative_id.clone(),
        title: req.title,
        description: req.description,
        arc_type: req.arc_type,
        situation_ids: req.situation_ids,
        order: req.order,
        created_at: now,
        updated_at: now,
    };
    let key = keys::user_arc_key(&narrative_id, &arc.id);
    let bytes = match serde_json::to_vec(&arc) {
        Ok(b) => b,
        Err(e) => return error_response(TensaError::Serialization(e.to_string())).into_response(),
    };
    if let Err(e) = state.hypergraph.store().put(&key, &bytes) {
        return error_response(e).into_response();
    }
    (StatusCode::CREATED, json_ok(&arc)).into_response()
}

/// GET /narratives/:id/arcs — list all arcs for a narrative, sorted by `order`.
pub async fn list_arcs(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let prefix = keys::user_arc_prefix(&narrative_id);
    let pairs = match state.hypergraph.store().prefix_scan(&prefix) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    let mut arcs: Vec<UserArc> = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        match serde_json::from_slice::<UserArc>(&value) {
            Ok(arc) => arcs.push(arc),
            Err(e) => {
                return error_response(TensaError::Serialization(e.to_string())).into_response()
            }
        }
    }
    arcs.sort_by(|a, b| a.order.cmp(&b.order).then(a.created_at.cmp(&b.created_at)));
    json_ok(&arcs)
}

/// PUT /narratives/:id/arcs/:arc_id — patch an arc.
pub async fn update_arc(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, arc_id)): Path<(String, Uuid)>,
    Json(req): Json<UpdateArcRequest>,
) -> impl IntoResponse {
    let key = keys::user_arc_key(&narrative_id, &arc_id);
    let existing = match state.hypergraph.store().get(&key) {
        Ok(Some(bytes)) => match serde_json::from_slice::<UserArc>(&bytes) {
            Ok(arc) => arc,
            Err(e) => {
                return error_response(TensaError::Serialization(e.to_string())).into_response()
            }
        },
        Ok(None) => {
            return error_response(TensaError::NotFound(format!(
                "arc {} not found in narrative {}",
                arc_id, narrative_id
            )))
            .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    let mut arc = existing;
    if let Some(title) = req.title {
        if title.trim().is_empty() {
            return error_response(TensaError::InvalidQuery("title cannot be empty".into()))
                .into_response();
        }
        arc.title = title;
    }
    if let Some(description) = req.description {
        arc.description = description;
    }
    if let Some(arc_type) = req.arc_type {
        if arc_type.trim().is_empty() {
            return error_response(TensaError::InvalidQuery("arc_type cannot be empty".into()))
                .into_response();
        }
        arc.arc_type = arc_type;
    }
    if let Some(situation_ids) = req.situation_ids {
        arc.situation_ids = situation_ids;
    }
    if let Some(order) = req.order {
        arc.order = order;
    }
    arc.updated_at = Utc::now();
    let bytes = match serde_json::to_vec(&arc) {
        Ok(b) => b,
        Err(e) => return error_response(TensaError::Serialization(e.to_string())).into_response(),
    };
    if let Err(e) = state.hypergraph.store().put(&key, &bytes) {
        return error_response(e).into_response();
    }
    json_ok(&arc)
}

/// DELETE /narratives/:id/arcs/:arc_id — hard-delete an arc.
pub async fn delete_arc(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, arc_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    let key = keys::user_arc_key(&narrative_id, &arc_id);
    match state.hypergraph.store().get(&key) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return error_response(TensaError::NotFound(format!(
                "arc {} not found in narrative {}",
                arc_id, narrative_id
            )))
            .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    }
    if let Err(e) = state.hypergraph.store().delete(&key) {
        return error_response(e).into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

// ─── POST /narratives/:id/situations/reorder ─────────────────────────

#[derive(Debug, Deserialize)]
pub struct ReorderItem {
    pub situation_id: Uuid,
    #[serde(default)]
    pub new_start: Option<DateTime<Utc>>,
    #[serde(default)]
    pub new_end: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct ReorderRequest {
    pub items: Vec<ReorderItem>,
}

#[derive(Debug, Serialize)]
pub struct ReorderResponse {
    pub updated: usize,
    pub errors: Vec<BulkItemResult>,
}

/// Update `temporal.start` / `temporal.end` for many situations in one request.
/// Sugar over N `PUT /situations/:id` calls; writes temporal in place.
pub async fn reorder_situations(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(req): Json<ReorderRequest>,
) -> impl IntoResponse {
    if req.items.len() > MAX_REORDER_ITEMS {
        return error_response(TensaError::InvalidQuery(format!(
            "reorder batch exceeds maximum of {} items",
            MAX_REORDER_ITEMS
        )))
        .into_response();
    }
    let mut updated = 0usize;
    let mut errors = Vec::new();
    for (i, item) in req.items.into_iter().enumerate() {
        // Validate narrative membership up-front so we don't trigger a spurious
        // update_situation (which would bump updated_at and churn indexes).
        let existing = match state.hypergraph.get_situation(&item.situation_id) {
            Ok(s) => s,
            Err(e) => {
                errors.push(BulkItemResult {
                    index: i,
                    id: None,
                    error: Some(e.to_string()),
                });
                continue;
            }
        };
        if existing.narrative_id.as_deref() != Some(narrative_id.as_str()) {
            errors.push(BulkItemResult {
                index: i,
                id: None,
                error: Some(format!(
                    "situation {} does not belong to narrative {}",
                    item.situation_id, narrative_id
                )),
            });
            continue;
        }
        let result = state.hypergraph.update_situation(&item.situation_id, |s| {
            s.temporal = AllenInterval {
                start: item.new_start.or(s.temporal.start),
                end: item.new_end.or(s.temporal.end),
                granularity: s.temporal.granularity,
                relations: s.temporal.relations.clone(),
                fuzzy_endpoints: None,
            };
        });
        match result {
            Ok(_) => updated += 1,
            Err(e) => errors.push(BulkItemResult {
                index: i,
                id: None,
                error: Some(e.to_string()),
            }),
        }
    }
    json_ok(&ReorderResponse { updated, errors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_arc_key_is_prefixed_by_narrative() {
        let narrative = "hamlet";
        let id = Uuid::now_v7();
        let k = keys::user_arc_key(narrative, &id);
        let p = keys::user_arc_prefix(narrative);
        assert!(k.starts_with(&p));
        assert!(k.starts_with(b"ua/hamlet/"));
    }

    #[test]
    fn update_arc_description_cleared_with_null() {
        let json = serde_json::json!({ "description": null });
        let req: UpdateArcRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.description, Some(None));
    }

    #[test]
    fn update_arc_description_set() {
        let json = serde_json::json!({ "description": "x" });
        let req: UpdateArcRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.description, Some(Some("x".into())));
    }

    #[test]
    fn update_arc_description_omitted() {
        let json = serde_json::json!({});
        let req: UpdateArcRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.description, None);
    }
}
