//! Project API routes — CRUD for projects with cascade delete.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::routes::{cascade_delete_narrative, error_response};
use crate::api::server::AppState;

use crate::narrative::project::ProjectRegistry;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::types::Project;

/// POST /projects
pub async fn create_project(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Project>,
) -> impl IntoResponse {
    let reg = ProjectRegistry::new(state.hypergraph.store_arc());
    match reg.create(body) {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /projects
pub async fn list_projects(
    State(state): State<Arc<AppState>>,
    Query(params): Query<crate::api::routes::PaginationParams>,
) -> impl IntoResponse {
    let reg = ProjectRegistry::new(state.hypergraph.store_arc());
    let limit = crate::api::routes::clamp_limit(params.limit);
    match reg.list_paginated(limit, params.after.as_deref()) {
        Ok((projects, next_cursor)) => {
            let resp = crate::api::routes::PaginatedResponse {
                data: projects,
                next_cursor,
            };
            crate::api::routes::json_ok(&resp)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /projects/:id
pub async fn get_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let reg = ProjectRegistry::new(state.hypergraph.store_arc());
    match reg.get(&id) {
        Ok(project) => Json(serde_json::json!(project)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /projects/:id
pub async fn update_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let reg = ProjectRegistry::new(state.hypergraph.store_arc());
    match reg.update(&id, |p| {
        if let Some(title) = body.get("title").and_then(|v| v.as_str()) {
            p.title = title.to_string();
        }
        if let Some(desc) = body.get("description") {
            p.description = desc.as_str().map(|s| s.to_string());
        }
        if let Some(tags) = body.get("tags").and_then(|v| v.as_array()) {
            p.tags = tags
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
    }) {
        Ok(project) => Json(serde_json::json!(project)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /projects/:id?cascade=true
///
/// Without `cascade=true`, deletes only the project metadata.
/// With `cascade=true`, cascade-deletes all narratives (and their
/// entities/situations/participations/causal links) in the project.
pub async fn delete_project(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let cascade = params.get("cascade").map(|v| v == "true").unwrap_or(false);
    let project_reg = ProjectRegistry::new(state.hypergraph.store_arc());
    let narrative_reg = NarrativeRegistry::new(state.hypergraph.store_arc());

    if let Err(e) = project_reg.get(&id) {
        return error_response(e).into_response();
    }

    if cascade {
        let narrative_ids = match project_reg.list_narrative_ids(&id) {
            Ok(ids) => ids,
            Err(e) => return error_response(e).into_response(),
        };

        let mut total_entities = 0u64;
        let mut total_situations = 0u64;
        let mut total_participations = 0u64;
        let mut total_causal_links = 0u64;
        let mut narratives_deleted = 0u64;

        for nid in &narrative_ids {
            // Cascade delete narrative contents
            if let Ok(stats) = cascade_delete_narrative(&state.hypergraph, nid) {
                total_entities += stats["entities_deleted"].as_u64().unwrap_or(0);
                total_situations += stats["situations_deleted"].as_u64().unwrap_or(0);
                total_participations += stats["participations_removed"].as_u64().unwrap_or(0);
                total_causal_links += stats["causal_links_removed"].as_u64().unwrap_or(0);
            }
            // Delete narrative metadata
            let _ = narrative_reg.delete(nid);
            narratives_deleted += 1;
        }

        // Delete the project itself
        if let Err(e) = project_reg.delete(&id) {
            return error_response(e).into_response();
        }

        Json(serde_json::json!({
            "deleted": true,
            "cascade": {
                "narratives_deleted": narratives_deleted,
                "entities_deleted": total_entities,
                "situations_deleted": total_situations,
                "participations_removed": total_participations,
                "causal_links_removed": total_causal_links,
            }
        }))
        .into_response()
    } else {
        match project_reg.delete(&id) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => error_response(e).into_response(),
        }
    }
}

/// GET /projects/:id/narratives
pub async fn list_project_narratives(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let project_reg = ProjectRegistry::new(state.hypergraph.store_arc());
    let narrative_reg = NarrativeRegistry::new(state.hypergraph.store_arc());

    let narrative_ids = match project_reg.list_narrative_ids(&id) {
        Ok(ids) => ids,
        Err(e) => return error_response(e).into_response(),
    };

    let mut narratives = Vec::new();
    for nid in &narrative_ids {
        if let Ok(n) = narrative_reg.get(nid) {
            narratives.push(n);
        }
    }

    Json(serde_json::json!(narratives)).into_response()
}
