//! Continuity endpoints — Sprint W4 (v0.49.3).
//!
//! - `GET    /narratives/:id/pinned-facts`       — list
//! - `POST   /narratives/:id/pinned-facts`       — create
//! - `PUT    /narratives/:id/pinned-facts/:fact_id` — update
//! - `DELETE /narratives/:id/pinned-facts/:fact_id` — delete
//! - `POST   /narratives/:id/continuity-check`   — check a proposal
//! - `GET    /narratives/:id/workspace`          — workspace summary

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{deserialize_optional_option, error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::continuity::{
    check_edit_proposal, check_generation_proposal, create_pinned_fact, delete_pinned_fact,
    list_pinned_facts, update_pinned_fact, PinnedFactPatch,
};
use crate::narrative::editing::EditProposal;
use crate::narrative::generation::GenerationProposal;
use crate::narrative::workspace::get_workspace_summary;
use crate::types::PinnedFact;

// ─── Pinned facts ─────────────────────────────────────────────────

pub async fn list_facts_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match list_pinned_facts(state.hypergraph.store(), &narrative_id) {
        Ok(list) => json_ok(&list),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateFactBody {
    #[serde(default)]
    pub entity_id: Option<Uuid>,
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub note: Option<String>,
}

pub async fn create_fact_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<CreateFactBody>,
) -> impl IntoResponse {
    let now = chrono::Utc::now();
    let fact = PinnedFact {
        id: Uuid::nil(),
        narrative_id,
        entity_id: body.entity_id,
        key: body.key,
        value: body.value,
        note: body.note,
        created_at: now,
        updated_at: now,
    };
    match create_pinned_fact(state.hypergraph.store(), fact) {
        Ok(f) => (StatusCode::CREATED, json_ok(&f)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct UpdateFactBody {
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub note: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub entity_id: Option<Option<Uuid>>,
}

impl From<UpdateFactBody> for PinnedFactPatch {
    fn from(b: UpdateFactBody) -> Self {
        PinnedFactPatch {
            key: b.key,
            value: b.value,
            note: b.note,
            entity_id: b.entity_id,
        }
    }
}

pub async fn update_fact_handler(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, fact_id)): Path<(String, Uuid)>,
    Json(body): Json<UpdateFactBody>,
) -> impl IntoResponse {
    match update_pinned_fact(
        state.hypergraph.store(),
        &narrative_id,
        &fact_id,
        body.into(),
    ) {
        Ok(f) => json_ok(&f),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn delete_fact_handler(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, fact_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    // 404 existence probe — byte-level `get`, no full deserialize. `store.delete`
    // is a no-op for missing keys so we can't get 404 from the delete itself.
    let key = crate::hypergraph::keys::pinned_fact_key(&narrative_id, &fact_id);
    match state.hypergraph.store().get(&key) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return error_response(TensaError::NotFound(format!("pinned fact {}", fact_id)))
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    }
    match delete_pinned_fact(state.hypergraph.store(), &narrative_id, &fact_id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Continuity check ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContinuityCheckBody {
    Generation { proposal: GenerationProposal },
    Edit { proposal: EditProposal },
}

pub async fn check_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<ContinuityCheckBody>,
) -> impl IntoResponse {
    let result = match body {
        ContinuityCheckBody::Generation { mut proposal } => {
            if proposal.narrative_id != narrative_id {
                proposal.narrative_id = narrative_id;
            }
            check_generation_proposal(&state.hypergraph, &proposal)
        }
        ContinuityCheckBody::Edit { mut proposal } => {
            if proposal.narrative_id != narrative_id {
                proposal.narrative_id = narrative_id;
            }
            check_edit_proposal(&state.hypergraph, &proposal)
        }
    };
    match result {
        Ok(warnings) => json_ok(&warnings),
        Err(e) => error_response(e).into_response(),
    }
}

/// Body for `POST /narratives/:id/continuity/check-prose`.
#[derive(Debug, Deserialize)]
pub struct CheckProseBody {
    pub prose: String,
}

/// `POST /narratives/:id/continuity/check-prose` — deterministic scan of a
/// block of prose against the narrative's pinned facts. Simpler sibling of
/// `check_handler`: takes bare prose, no structured proposal required.
/// Mirrors the `check_continuity` MCP tool's intent.
pub async fn check_prose_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<CheckProseBody>,
) -> impl IntoResponse {
    match crate::narrative::continuity::check_prose(&state.hypergraph, &narrative_id, &body.prose) {
        Ok(warnings) => json_ok(&warnings),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Workspace summary ────────────────────────────────────────────

pub async fn workspace_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match get_workspace_summary(&state.hypergraph, &narrative_id) {
        Ok(s) => json_ok(&s),
        Err(e) => error_response(e).into_response(),
    }
}
