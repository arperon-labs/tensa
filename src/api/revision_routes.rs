//! Narrative revision endpoints (v0.48.1).
//!
//! "git for narratives" — commit, list, restore, diff over the full authored
//! state of a narrative. See `src/narrative/revision.rs` for the core logic.
//!
//! - `POST   /narratives/:id/revisions`          — commit current state
//! - `GET    /narratives/:id/revisions`          — list history
//! - `GET    /narratives/:id/revisions/head`     — fetch HEAD (full snapshot)
//! - `GET    /revisions/:rev_id`                 — fetch a revision (full snapshot)
//! - `POST   /revisions/:rev_id/restore`         — restore (auto-commits first)
//! - `GET    /narratives/:id/diff?from=&to=`     — diff two revisions

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::revision::{
    commit_narrative, diff_revisions, get_revision, list_revisions, restore_revision, summary_of,
    CommitOutcome,
};

fn registry_for(state: &AppState) -> NarrativeRegistry {
    NarrativeRegistry::new(state.hypergraph.store_arc())
}

// ─── POST /narratives/:id/revisions ──────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CommitRequest {
    pub message: String,
    #[serde(default)]
    pub author: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum CommitResponse {
    Committed {
        revision: crate::types::RevisionSummary,
    },
    NoChange {
        revision: crate::types::RevisionSummary,
    },
}

pub async fn commit_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(req): Json<CommitRequest>,
) -> impl IntoResponse {
    if req.message.trim().is_empty() {
        return error_response(TensaError::InvalidQuery(
            "commit message is required".into(),
        ))
        .into_response();
    }
    let registry = registry_for(&state);
    match commit_narrative(
        &state.hypergraph,
        &registry,
        &narrative_id,
        req.message,
        req.author,
    ) {
        Ok(CommitOutcome::Committed(rev)) => (
            StatusCode::CREATED,
            json_ok(&CommitResponse::Committed {
                revision: summary_of(&rev),
            }),
        )
            .into_response(),
        Ok(CommitOutcome::NoChange(rev)) => json_ok(&CommitResponse::NoChange {
            revision: summary_of(&rev),
        })
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── GET /narratives/:id/revisions ───────────────────────────────

pub async fn list_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match list_revisions(state.hypergraph.store(), &narrative_id) {
        Ok(list) => json_ok(&list),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── GET /narratives/:id/revisions/head ──────────────────────────

pub async fn head_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let list = match list_revisions(state.hypergraph.store(), &narrative_id) {
        Ok(l) => l,
        Err(e) => return error_response(e).into_response(),
    };
    let Some(latest) = list.last() else {
        return error_response(TensaError::NotFound(format!(
            "no revisions yet for narrative {}",
            narrative_id
        )))
        .into_response();
    };
    match get_revision(state.hypergraph.store(), &latest.id) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── GET /revisions/:rev_id ──────────────────────────────────────

pub async fn get_handler(
    State(state): State<Arc<AppState>>,
    Path(rev_id): Path<Uuid>,
) -> impl IntoResponse {
    match get_revision(state.hypergraph.store(), &rev_id) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── POST /revisions/:rev_id/restore ─────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub struct RestoreRequest {
    #[serde(default)]
    pub author: Option<String>,
}

pub async fn restore_handler(
    State(state): State<Arc<AppState>>,
    Path(rev_id): Path<Uuid>,
    body: Option<Json<RestoreRequest>>,
) -> impl IntoResponse {
    let author = body.and_then(|Json(b)| b.author);
    let registry = registry_for(&state);
    match restore_revision(&state.hypergraph, &registry, &rev_id, author) {
        Ok(report) => json_ok(&serde_json::json!({
            "restored_from": report.restored_from,
            "auto_commit": report.auto_commit,
            "situations_restored": report.situations_restored,
            "entities_restored": report.entities_restored,
            "participations_restored": report.participations_restored,
            "user_arcs_restored": report.user_arcs_restored,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── GET /narratives/:id/diff?from=&to= ──────────────────────────

#[derive(Debug, Deserialize)]
pub struct DiffQuery {
    pub from: Uuid,
    pub to: Uuid,
}

pub async fn diff_handler(
    State(state): State<Arc<AppState>>,
    Path(_narrative_id): Path<String>,
    Query(q): Query<DiffQuery>,
) -> impl IntoResponse {
    match diff_revisions(state.hypergraph.store(), &q.from, &q.to) {
        Ok(d) => json_ok(&d),
        Err(e) => error_response(e).into_response(),
    }
}
