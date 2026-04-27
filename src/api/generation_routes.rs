//! Generation endpoints — Sprint W1 (v0.49.0).
//!
//! Every endpoint routes through the proposal → apply → commit pipeline in
//! [src/narrative/generation.rs](../narrative/generation.rs). Dry-run mode
//! returns the assembled prompt + token estimate without calling the LLM.
//!
//! - `POST /narratives/:id/generate?dry_run=true`  — prompt + estimate only
//! - `POST /narratives/:id/generate`               — full generation, returns proposal
//! - `POST /narratives/:id/generate/apply`         — apply a proposal (writes + commits)
//! - `POST /narratives/:id/generate/estimate`      — cost estimate only (no LLM)

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::generation::{
    apply_proposal, build_prompt_for_narrative, estimate_tokens, generate_for_narrative,
    GenerationPrompt, GenerationProposal, GenerationRequest,
};
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::revision::gather_snapshot;

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
pub enum GenerateResponse {
    Prompt { prompt: GenerationPrompt },
    Proposal { proposal: GenerationProposal },
}

pub async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<DryRunQuery>,
    Json(req): Json<GenerationRequest>,
) -> impl IntoResponse {
    let registry = registry_for(&state);

    if q.dry_run {
        return match build_prompt_for_narrative(&state.hypergraph, &registry, &narrative_id, &req) {
            Ok(prompt) => json_ok(&GenerateResponse::Prompt { prompt }),
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
            "active LLM provider does not support session-style calls (required for generation)"
                .into(),
        ))
        .into_response();
    };

    let cache = state.llm_cache.as_ref();
    match generate_for_narrative(
        &state.hypergraph,
        &registry,
        session,
        cache,
        &narrative_id,
        &req,
    ) {
        Ok(proposal) => json_ok(&GenerateResponse::Proposal { proposal }),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct ApplyBody {
    pub proposal: GenerationProposal,
    #[serde(default)]
    pub author: Option<String>,
}

pub async fn apply_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<ApplyBody>,
) -> impl IntoResponse {
    // Guard: path id wins over body's narrative_id so the client can't redirect
    // the apply to a different narrative than the one in the URL.
    let mut proposal = body.proposal;
    if proposal.narrative_id != narrative_id {
        proposal.narrative_id = narrative_id;
    }
    let registry = registry_for(&state);
    match apply_proposal(&state.hypergraph, &registry, proposal, body.author) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Serialize)]
pub struct EstimateResponse {
    pub prompt_tokens: u32,
    pub expected_response_tokens: u32,
}

pub async fn estimate_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(req): Json<GenerationRequest>,
) -> impl IntoResponse {
    let registry = registry_for(&state);
    // Ensure narrative exists for a clear 404.
    if let Err(e) = registry.get(&narrative_id) {
        return error_response(e).into_response();
    }
    let snapshot = match gather_snapshot(&state.hypergraph, &registry, &narrative_id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let est = estimate_tokens(&req, &snapshot);
    json_ok(&EstimateResponse {
        prompt_tokens: est.prompt_tokens,
        expected_response_tokens: est.expected_response_tokens,
    })
}
