//! Edit endpoints — Sprint W2 (v0.49.1).
//!
//! - `POST /situations/:id/edit?dry_run=true`  — prompt + estimate only
//! - `POST /situations/:id/edit`               — returns `EditProposal`
//! - `POST /situations/:id/edit/apply`         — applies a proposal + commits revision

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::editing::{
    apply_edit, build_prompt_for_situation, propose_edit_for_situation, EditOperation, EditProposal,
};
use crate::narrative::registry::NarrativeRegistry;

fn registry_for(state: &AppState) -> NarrativeRegistry {
    NarrativeRegistry::new(state.hypergraph.store_arc())
}

#[derive(Debug, Deserialize, Default)]
pub struct DryRunQuery {
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum EditResponse {
    Prompt {
        prompt: crate::narrative::editing::EditPrompt,
    },
    Proposal {
        proposal: EditProposal,
    },
}

pub async fn edit_handler(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Query(q): Query<DryRunQuery>,
    Json(op): Json<EditOperation>,
) -> impl IntoResponse {
    let registry = registry_for(&state);

    if q.dry_run {
        return match build_prompt_for_situation(&state.hypergraph, &registry, &situation_id, &op) {
            Ok(prompt) => json_ok(&EditResponse::Prompt { prompt }),
            Err(e) => error_response(e).into_response(),
        };
    }

    let extractor_guard = match state.extractor.read() {
        Ok(g) => g,
        Err(_) => {
            return error_response(TensaError::Internal("extractor lock poisoned".into()))
                .into_response()
        }
    };
    let Some(extractor_arc) = extractor_guard.as_ref() else {
        return error_response(TensaError::LlmError(
            "no LLM extractor configured (see /settings/llm)".into(),
        ))
        .into_response();
    };
    let Some(session) = extractor_arc.as_session() else {
        return error_response(TensaError::LlmError(
            "active LLM provider does not support session-style calls".into(),
        ))
        .into_response();
    };

    match propose_edit_for_situation(&state.hypergraph, &registry, session, &situation_id, &op) {
        Ok(proposal) => json_ok(&EditResponse::Proposal { proposal }),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct ApplyBody {
    pub proposal: EditProposal,
    #[serde(default)]
    pub author: Option<String>,
}

pub async fn apply_handler(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<ApplyBody>,
) -> impl IntoResponse {
    // Path wins over proposal.situation_id to prevent redirect attacks.
    let mut proposal = body.proposal;
    if proposal.situation_id != situation_id {
        proposal.situation_id = situation_id;
    }
    let registry = registry_for(&state);
    match apply_edit(&state.hypergraph, &registry, proposal, body.author) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}
