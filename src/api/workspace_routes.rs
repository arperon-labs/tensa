use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::routes::json_ok;
use crate::api::server::AppState;

/// Workspace metadata stored at `_ws/{id}` in the root KV store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceMeta {
    /// Unique workspace identifier (URL-safe slug).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// When the workspace was created.
    pub created_at: DateTime<Utc>,
}

/// Create a new workspace.
///
/// `POST /workspaces`
/// Body: `{ "id": "my-workspace", "name": "My Workspace" }`
pub async fn create_workspace(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let id = body
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "id is required"})),
        )
            .into_response();
    }
    // Validate: no slashes, no dots, reasonable length, alphanumeric + hyphens + underscores
    if id.contains('/') || id.contains('.') || id.len() > 64 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Invalid workspace id: must not contain / or . and be at most 64 chars"})),
        )
            .into_response();
    }
    // Check if workspace already exists
    let key = format!("_ws/{}", id);
    match state.root_store.get(key.as_bytes()) {
        Ok(Some(_)) => {
            return (
                StatusCode::CONFLICT,
                Json(json!({"error": format!("Workspace '{}' already exists", id)})),
            )
                .into_response();
        }
        Ok(None) => {}
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    }
    let meta = WorkspaceMeta {
        id: id.clone(),
        name: body
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&id)
            .to_string(),
        created_at: Utc::now(),
    };
    let value = match serde_json::to_vec(&meta) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };
    match state.root_store.put(key.as_bytes(), &value) {
        Ok(_) => json_ok(&meta),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// List all workspaces.
///
/// `GET /workspaces`
pub async fn list_workspaces(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.root_store.prefix_scan(b"_ws/") {
        Ok(entries) => {
            let workspaces: Vec<WorkspaceMeta> = entries
                .iter()
                .filter_map(|(_, v)| serde_json::from_slice(v).ok())
                .collect();
            json_ok(&workspaces)
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Get a single workspace by id.
///
/// `GET /workspaces/:id`
pub async fn get_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let key = format!("_ws/{}", id);
    match state.root_store.get(key.as_bytes()) {
        Ok(Some(data)) => match serde_json::from_slice::<WorkspaceMeta>(&data) {
            Ok(meta) => json_ok(&meta),
            Err(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "corrupt workspace metadata"})),
            )
                .into_response(),
        },
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("Workspace '{}' not found", id)})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Delete a workspace and all its data.
///
/// `DELETE /workspaces/:id`
///
/// The `default` workspace cannot be deleted.
pub async fn delete_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if id == "default" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Cannot delete the default workspace"})),
        )
            .into_response();
    }
    // Verify workspace exists
    let key = format!("_ws/{}", id);
    match state.root_store.get(key.as_bytes()) {
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("Workspace '{}' not found", id)})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    }
    // Delete workspace metadata
    if let Err(e) = state.root_store.delete(key.as_bytes()) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response();
    }
    // Delete all data in the workspace (prefix scan w/{id}/ and delete all)
    let ws_prefix = format!("w/{}/", id);
    if let Ok(entries) = state.root_store.prefix_scan(ws_prefix.as_bytes()) {
        for (k, _) in entries {
            let _ = state.root_store.delete(&k);
        }
    }
    json_ok(&json!({"deleted": id}))
}
