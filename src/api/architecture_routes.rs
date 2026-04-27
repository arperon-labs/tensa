//! Narrative architecture endpoints — Sprint W14 (v0.69.0).
//!
//! Exposes the D9 narrative-architecture analysers that were previously
//! stub-only in the MCP surface:
//!
//! - `GET    /narratives/:id/commitments`              — detect Chekhov's guns / foreshadowing
//! - `GET    /narratives/:id/commitment-rhythm`        — promise rhythm (tension curve)
//! - `GET    /narratives/:id/fabula`                   — chronological event order
//! - `GET    /narratives/:id/sjuzet`                   — discourse/telling order
//! - `GET    /narratives/:id/sjuzet/reorderings`       — candidate reorderings
//! - `GET    /narratives/:id/dramatic-irony`           — reader/character knowledge gaps
//! - `GET    /narratives/:id/focalization`             — focalization × irony interactions
//! - `GET    /narratives/:id/character-arc`            — detect arc(s); `?character_id=UUID`
//! - `GET    /narratives/:id/subplots`                 — community-based subplot detection
//! - `GET    /narratives/:id/scene-sequel`             — Swain/Bickham scene-sequel rhythm
//!
//! - `POST   /narratives/plan`                         — generate a NarrativePlan (in-memory)
//! - `POST   /plans/:plan_id/materialize`              — materialize plan → hypergraph
//! - `GET    /narratives/:id/validate-materialized`    — consistency issues
//! - `POST   /narratives/:id/generate-chapter`         — prepare a chapter prompt (non-LLM)
//! - `POST   /narratives/:id/generate-narrative`       — prepare full-narrative prompts
//!
//! The plan / materialize / validate / generate-chapter / generate-narrative
//! endpoints are feature-gated behind `generation`, matching the backend
//! module's cfg. The detection endpoints compile unconditionally.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
#[cfg(feature = "generation")]
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::character_arcs::{detect_character_arc, list_character_arcs};
use crate::narrative::commitments::{compute_promise_rhythm, detect_commitments};
use crate::narrative::dramatic_irony::{
    compute_dramatic_irony_map, compute_focalization_irony_interaction,
};
use crate::narrative::fabula_sjuzet::{extract_fabula, extract_sjuzet, suggest_reordering};
use crate::narrative::scene_sequel::analyze_scene_sequel;
use crate::narrative::subplots::detect_subplots;

// ─── D9.1 Commitments ─────────────────────────────────────────────

pub async fn commitments_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match detect_commitments(&state.hypergraph, &narrative_id) {
        Ok(cs) => json_ok(&cs),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn commitment_rhythm_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match compute_promise_rhythm(&state.hypergraph, &narrative_id) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── D9.2 Fabula / Sjuzet ─────────────────────────────────────────

pub async fn fabula_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match extract_fabula(&state.hypergraph, &narrative_id) {
        Ok(f) => json_ok(&f),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn sjuzet_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match extract_sjuzet(&state.hypergraph, &narrative_id) {
        Ok(s) => json_ok(&s),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn sjuzet_reorderings_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let fabula = match extract_fabula(&state.hypergraph, &narrative_id) {
        Ok(f) => f,
        Err(e) => return error_response(e).into_response(),
    };
    let sjuzet = match extract_sjuzet(&state.hypergraph, &narrative_id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    match suggest_reordering(&fabula, &sjuzet, &state.hypergraph, &narrative_id) {
        Ok(cs) => json_ok(&cs),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── D9.3 Dramatic irony / focalization ───────────────────────────

pub async fn dramatic_irony_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match compute_dramatic_irony_map(&state.hypergraph, &narrative_id) {
        Ok(m) => json_ok(&m),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn focalization_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match compute_focalization_irony_interaction(&state.hypergraph, &narrative_id) {
        Ok(ix) => json_ok(&ix),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── D9.4 Character arcs / subplots / scene-sequel ────────────────

#[derive(Debug, Deserialize)]
pub struct CharacterArcQuery {
    #[serde(default)]
    pub character_id: Option<Uuid>,
}

pub async fn character_arc_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<CharacterArcQuery>,
) -> impl IntoResponse {
    match q.character_id {
        Some(uuid) => match detect_character_arc(&state.hypergraph, &narrative_id, uuid) {
            Ok(arc) => json_ok(&arc),
            Err(e) => error_response(e).into_response(),
        },
        None => match list_character_arcs(&state.hypergraph, &narrative_id) {
            Ok(arcs) => json_ok(&arcs),
            Err(e) => error_response(e).into_response(),
        },
    }
}

pub async fn subplots_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match detect_subplots(&state.hypergraph, &narrative_id) {
        Ok(a) => json_ok(&a),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn scene_sequel_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match analyze_scene_sequel(&state.hypergraph, &narrative_id) {
        Ok(a) => json_ok(&a),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── D9.6 Plan / Materialize / Validate ───────────────────────────

#[cfg(feature = "generation")]
#[derive(Debug, Deserialize)]
pub struct GeneratePlanBody {
    pub premise: String,
    #[serde(default = "default_genre")]
    pub genre: String,
    #[serde(default = "default_chapters")]
    pub chapter_count: usize,
    #[serde(default = "default_subplots")]
    pub subplot_count: usize,
    #[serde(default = "default_protagonists")]
    pub protagonist_count: usize,
    #[serde(default = "default_density")]
    pub commitment_density: f64,
}

#[cfg(feature = "generation")]
fn default_genre() -> String {
    "literary fiction".into()
}
#[cfg(feature = "generation")]
fn default_chapters() -> usize {
    12
}
#[cfg(feature = "generation")]
fn default_subplots() -> usize {
    2
}
#[cfg(feature = "generation")]
fn default_protagonists() -> usize {
    1
}
#[cfg(feature = "generation")]
fn default_density() -> f64 {
    0.5
}

/// POST /narratives/plan — generate a narrative plan and persist it.
#[cfg(feature = "generation")]
pub async fn generate_plan_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GeneratePlanBody>,
) -> impl IntoResponse {
    use crate::generation::planner::{generate_plan, store_plan};
    use crate::generation::types::PlanConfig;

    let config = PlanConfig {
        genre: body.genre,
        chapter_count: body.chapter_count,
        protagonist_count: body.protagonist_count,
        subplot_count: body.subplot_count,
        commitment_density: body.commitment_density,
        premise: body.premise,
        constraints: Vec::new(),
    };
    let plan = match generate_plan(config) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    if let Err(e) = store_plan(&state.hypergraph, &plan) {
        return error_response(e).into_response();
    }
    json_ok(&plan)
}

/// POST /plans/:plan_id/materialize — materialize a stored plan.
#[cfg(feature = "generation")]
pub async fn materialize_plan_handler(
    State(state): State<Arc<AppState>>,
    Path(plan_id): Path<Uuid>,
) -> impl IntoResponse {
    use crate::generation::materializer::materialize_plan;
    use crate::generation::planner::load_plan;

    let plan = match load_plan(&state.hypergraph, &plan_id) {
        Ok(Some(p)) => p,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!("plan {plan_id} not found")))
                .into_response();
        }
        Err(e) => return error_response(e).into_response(),
    };
    match materialize_plan(&state.hypergraph, &plan) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/validate-materialized — run consistency checks.
#[cfg(feature = "generation")]
pub async fn validate_materialized_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    use crate::generation::materializer::validate_materialized;

    match validate_materialized(&state.hypergraph, &narrative_id) {
        Ok(issues) => json_ok(&issues),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── D9.7 Chapter / Narrative prompt preparation ──────────────────

#[cfg(feature = "generation")]
#[derive(Debug, Deserialize)]
pub struct GenerateChapterBody {
    pub chapter: usize,
    #[serde(default)]
    pub voice_description: Option<String>,
}

/// POST /narratives/:id/generate-chapter — build a chapter prompt (non-LLM).
#[cfg(feature = "generation")]
pub async fn generate_chapter_prep_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<GenerateChapterBody>,
) -> impl IntoResponse {
    use crate::generation::engine::GenerationEngine;
    use crate::generation::types::StyleTarget;

    let style = StyleTarget {
        voice_description: body.voice_description,
        ..StyleTarget::default()
    };
    let engine = GenerationEngine::new(narrative_id);
    match engine.prepare_chapter(&state.hypergraph, body.chapter, &style, &[]) {
        Ok((prompt, chapter)) => json_ok(&serde_json::json!({
            "prompt": prompt,
            "chapter": chapter,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

#[cfg(feature = "generation")]
#[derive(Debug, Deserialize)]
pub struct GenerateNarrativeBody {
    pub chapter_count: usize,
    #[serde(default)]
    pub voice_description: Option<String>,
}

/// POST /narratives/:id/generate-narrative — build prompts for all chapters (non-LLM).
#[cfg(feature = "generation")]
pub async fn generate_narrative_prep_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<GenerateNarrativeBody>,
) -> impl IntoResponse {
    use crate::generation::engine::GenerationEngine;
    use crate::generation::types::StyleTarget;

    let style = StyleTarget {
        voice_description: body.voice_description,
        ..StyleTarget::default()
    };
    let engine = GenerationEngine::new(narrative_id);
    match engine.prepare_full_narrative(&state.hypergraph, &style, body.chapter_count) {
        Ok(result) => json_ok(&result),
        Err(e) => error_response(e).into_response(),
    }
}
