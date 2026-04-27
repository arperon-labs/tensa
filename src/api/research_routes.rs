//! REST routes for Sprint W9: scene research context + research notes.

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
use crate::writer::cited_generation::{
    build_research_prompt, hallucination_guard, parse_cited_text, CitedGenerationRequest,
};
use crate::writer::factcheck::{run_factcheck, FactCheckTier};
use crate::writer::research::{
    build_scene_research_context, create_research_note, delete_research_note, get_research_note,
    list_notes_for_narrative, promote_chunk_to_note, update_research_note, PromoteChunkRequest,
    ResearchNote, ResearchNoteKind, ResearchNotePatch,
};

/// GET /situations/:id/research-context — aggregated research view.
pub async fn get_scene_research_context(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
) -> impl IntoResponse {
    match build_scene_research_context(&state.hypergraph, &situation_id) {
        Ok(ctx) => json_ok(&ctx),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /situations/:id/research-notes — create a research note pinned to this scene.
#[derive(Debug, Deserialize)]
pub struct CreateResearchNoteBody {
    pub narrative_id: String,
    pub kind: ResearchNoteKind,
    pub body: String,
    #[serde(default)]
    pub source_chunk_id: Option<Uuid>,
    #[serde(default)]
    pub source_id: Option<Uuid>,
    #[serde(default)]
    pub author: Option<String>,
}

pub async fn create_note_for_situation(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<CreateResearchNoteBody>,
) -> impl IntoResponse {
    // Validate the scene exists and the narrative matches.
    match state.hypergraph.get_situation(&situation_id) {
        Ok(sit) => {
            if sit.narrative_id.as_deref() != Some(&body.narrative_id) {
                return error_response(TensaError::InvalidQuery(format!(
                    "situation {situation_id} does not belong to narrative {}",
                    body.narrative_id
                )))
                .into_response();
            }
        }
        Err(e) => return error_response(e).into_response(),
    }
    let note = ResearchNote {
        id: Uuid::nil(),
        narrative_id: body.narrative_id,
        situation_id,
        kind: body.kind,
        body: body.body,
        source_chunk_id: body.source_chunk_id,
        source_id: body.source_id,
        author: body.author,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    match create_research_note(state.hypergraph.store(), note) {
        Ok(saved) => json_ok(&saved),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /situations/:id/research-notes — list notes pinned to this scene.
pub async fn list_notes_for_situation(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
) -> impl IntoResponse {
    match crate::writer::research::list_notes_for_situation(state.hypergraph.store(), &situation_id)
    {
        Ok(notes) => json_ok(&notes),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/research-notes — list notes across the whole narrative.
pub async fn list_notes_in_narrative(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match list_notes_for_narrative(state.hypergraph.store(), &narrative_id) {
        Ok(notes) => json_ok(&notes),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /research-notes/:id — patch an existing note.
pub async fn update_note(
    State(state): State<Arc<AppState>>,
    Path(note_id): Path<Uuid>,
    Json(patch): Json<ResearchNotePatch>,
) -> impl IntoResponse {
    match update_research_note(state.hypergraph.store(), &note_id, patch) {
        Ok(n) => json_ok(&n),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /research-notes/:id
pub async fn delete_note(
    State(state): State<Arc<AppState>>,
    Path(note_id): Path<Uuid>,
) -> impl IntoResponse {
    match delete_research_note(state.hypergraph.store(), &note_id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /research-notes/:id
pub async fn get_note(
    State(state): State<Arc<AppState>>,
    Path(note_id): Path<Uuid>,
) -> impl IntoResponse {
    match get_research_note(state.hypergraph.store(), &note_id) {
        Ok(n) => json_ok(&n),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /situations/:id/research-notes/from-chunk — promote a chunk to a note.
#[derive(Debug, Deserialize)]
pub struct PromoteChunkBody {
    pub narrative_id: String,
    pub chunk_id: Uuid,
    pub body: String,
    #[serde(default)]
    pub source_id: Option<Uuid>,
    #[serde(default)]
    pub kind: Option<ResearchNoteKind>,
    #[serde(default)]
    pub author: Option<String>,
}

/// POST /situations/:id/factcheck — Sprint W10 inline fact-check.
///
/// Body: `{ "text": "...", "tier": "Fast"|"Standard"|"Deep" }`.
/// Returns a `FactCheckReport` with per-claim verdicts against the scene's
/// research context.
#[derive(Debug, Deserialize)]
pub struct FactCheckBody {
    pub text: String,
    #[serde(default)]
    pub tier: Option<FactCheckTier>,
}

/// POST /situations/:id/generation-prompt — Sprint W11.
/// Returns the research addendum + pending contentions for a cited-generation request.
pub async fn build_generation_prompt(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(mut body): Json<CitedGenerationRequest>,
) -> impl IntoResponse {
    body.situation_id = situation_id;
    match build_research_prompt(&state.hypergraph, &body) {
        Ok(add) => json_ok(&add),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /generation/parse-citations — Sprint W11 helper.
/// Accepts raw LLM output; returns clean text + per-span citation lists.
#[derive(Debug, Deserialize)]
pub struct ParseCitationsBody {
    pub raw: String,
}

pub async fn parse_citations_route(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<ParseCitationsBody>,
) -> impl IntoResponse {
    let parsed = parse_cited_text(&body.raw);
    json_ok(&parsed)
}

/// POST /situations/:id/hallucination-guard — Sprint W11.
/// Runs fact-check over freshly-generated text; flags `Contradicted` verdicts.
pub async fn hallucination_guard_route(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<FactCheckBody>,
) -> impl IntoResponse {
    if body.text.trim().is_empty() {
        return error_response(TensaError::InvalidQuery(
            "hallucination-guard body 'text' cannot be empty".into(),
        ))
        .into_response();
    }
    match hallucination_guard(&state.hypergraph, &situation_id, &body.text) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn factcheck_scene(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<FactCheckBody>,
) -> impl IntoResponse {
    if body.text.trim().is_empty() {
        return error_response(TensaError::InvalidQuery(
            "factcheck body 'text' cannot be empty".into(),
        ))
        .into_response();
    }
    let tier = body.tier.unwrap_or_default();
    match run_factcheck(&state.hypergraph, &situation_id, &body.text, tier) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn promote_chunk(
    State(state): State<Arc<AppState>>,
    Path(situation_id): Path<Uuid>,
    Json(body): Json<PromoteChunkBody>,
) -> impl IntoResponse {
    match state.hypergraph.get_situation(&situation_id) {
        Ok(sit) => {
            if sit.narrative_id.as_deref() != Some(&body.narrative_id) {
                return error_response(TensaError::InvalidQuery(format!(
                    "situation {situation_id} does not belong to narrative {}",
                    body.narrative_id
                )))
                .into_response();
            }
        }
        Err(e) => return error_response(e).into_response(),
    }
    match promote_chunk_to_note(
        state.hypergraph.store(),
        PromoteChunkRequest {
            situation_id,
            narrative_id: body.narrative_id,
            chunk_id: body.chunk_id,
            body: body.body,
            source_id: body.source_id,
            kind: body.kind,
            author: body.author,
        },
    ) {
        Ok(n) => json_ok(&n),
        Err(e) => error_response(e).into_response(),
    }
}
