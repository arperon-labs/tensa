//! Sprint W13 — REST routes for compile profiles + compile endpoint.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::export::compile::{
    compile, create_profile, delete_profile, get_profile, list_profiles_for_narrative,
    update_profile, CompileFormat, CompileProfile, ProfilePatch,
};

/// Force a fresh server-assigned id + narrative on incoming profile creations.
fn normalize_for_create(mut p: CompileProfile, narrative_id: String) -> CompileProfile {
    p.id = Uuid::nil();
    p.narrative_id = narrative_id;
    p.created_at = chrono::Utc::now();
    p.updated_at = chrono::Utc::now();
    p
}

/// GET /narratives/:id/compile-profiles
pub async fn list_profiles(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match list_profiles_for_narrative(state.hypergraph.store(), &narrative_id) {
        Ok(profiles) => json_ok(&profiles),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/compile-profiles
///
/// Accepts a `CompileProfile` body directly. Client-supplied `id`,
/// `narrative_id`, `created_at`, `updated_at` are overwritten server-side.
pub async fn create_profile_route(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<CompileProfile>,
) -> impl IntoResponse {
    let p = normalize_for_create(body, narrative_id);
    match create_profile(state.hypergraph.store(), p) {
        Ok(saved) => json_ok(&saved),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /compile-profiles/:id
pub async fn get_profile_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match get_profile(state.hypergraph.store(), &id) {
        Ok(p) => json_ok(&p),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /compile-profiles/:id
pub async fn update_profile_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(patch): Json<ProfilePatch>,
) -> impl IntoResponse {
    match update_profile(state.hypergraph.store(), &id, patch) {
        Ok(p) => json_ok(&p),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /compile-profiles/:id
pub async fn delete_profile_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match delete_profile(state.hypergraph.store(), &id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CompileQuery {
    /// `markdown` (default) | `epub` | `docx`.
    #[serde(default)]
    pub format: Option<String>,
    /// Profile id to use. If absent, the default profile is synthesised.
    #[serde(default)]
    pub profile_id: Option<Uuid>,
}

/// POST /narratives/:id/compile — emits bytes for the chosen format.
pub async fn compile_route(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<CompileQuery>,
) -> impl IntoResponse {
    let format = match q.format.as_deref().unwrap_or("markdown") {
        "markdown" | "md" => CompileFormat::Markdown,
        "epub" => CompileFormat::Epub,
        "docx" => CompileFormat::Docx,
        other => {
            return error_response(TensaError::InvalidQuery(format!(
                "unknown compile format: {other}"
            )))
            .into_response()
        }
    };
    let data = match crate::export::collect_narrative_data(&narrative_id, &state.hypergraph) {
        Ok(d) => d,
        Err(e) => return error_response(e).into_response(),
    };
    let profile = match q.profile_id {
        Some(pid) => match get_profile(state.hypergraph.store(), &pid) {
            Ok(p) => p,
            Err(e) => return error_response(e).into_response(),
        },
        None => CompileProfile {
            id: Uuid::nil(),
            narrative_id: narrative_id.clone(),
            name: data.narrative_id.clone(),
            description: None,
            include_labels: vec![],
            exclude_labels: vec![],
            include_statuses: vec![],
            heading_templates: vec![],
            front_matter_md: None,
            back_matter_md: None,
            footnote_style: Default::default(),
            include_comments: false,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        },
    };
    match compile(state.hypergraph.store(), &data, &profile, format) {
        Ok(bytes) => {
            let mime = match format {
                CompileFormat::Markdown => "text/markdown",
                CompileFormat::Epub => "application/epub+zip",
                CompileFormat::Docx => {
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                }
            };
            let ext = match format {
                CompileFormat::Markdown => "md",
                CompileFormat::Epub => "epub",
                CompileFormat::Docx => "docx",
            };
            axum::response::Response::builder()
                .header(header::CONTENT_TYPE, mime)
                .header(
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{}.{}\"", narrative_id, ext),
                )
                .body(Body::from(bytes))
                .unwrap()
        }
        Err(e) => error_response(e).into_response(),
    }
}
