//! Narrative plan endpoints (v0.48.2).
//!
//! The writer's canonical plot/style/length/setting document for a narrative.
//! See [src/narrative/plan.rs](../narrative/plan.rs) for the core logic and
//! [docs/TENSA_WRITER_WORKFLOWS_SPRINT_PLAN.md](../../docs/TENSA_WRITER_WORKFLOWS_SPRINT_PLAN.md)
//! for how it integrates with W1–W3.
//!
//! - `GET    /narratives/:id/plan`  — returns `NarrativePlan | null`
//! - `PUT    /narratives/:id/plan`  — partial patch (null to clear nullable fields)
//! - `POST   /narratives/:id/plan`  — full replace
//! - `DELETE /narratives/:id/plan`  — hard delete

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::api::routes::{deserialize_optional_option, error_response, json_ok};
use crate::api::server::AppState;
use crate::narrative::plan::{self, PlanPatch};
use crate::types::{LengthTargets, NarrativePlan, PlotBeat, SettingNotes, StyleTargets};

// ─── GET ────────────────────────────────────────────────────────

pub async fn get_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match plan::get_plan(state.hypergraph.store(), &narrative_id) {
        Ok(Some(p)) => json_ok(&p),
        // Return 200 + null rather than 404 — "no plan yet" is a normal state
        // and the client should render the empty form.
        Ok(None) => json_ok(&serde_json::Value::Null),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── POST (full replace) ────────────────────────────────────────

pub async fn upsert_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(mut body): Json<NarrativePlan>,
) -> impl IntoResponse {
    // Path wins over body id so writers can't accidentally retarget.
    body.narrative_id = narrative_id;
    match plan::upsert_plan(state.hypergraph.store(), body) {
        Ok(p) => (StatusCode::OK, json_ok(&p)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── PUT (partial patch) ────────────────────────────────────────

#[derive(Deserialize, Default)]
pub struct PlanPatchBody {
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub logline: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub synopsis: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub premise: Option<Option<String>>,
    #[serde(default)]
    pub themes: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub central_conflict: Option<Option<String>>,
    #[serde(default)]
    pub plot_beats: Option<Vec<PlotBeat>>,
    #[serde(default)]
    pub style: Option<StyleTargets>,
    #[serde(default)]
    pub length: Option<LengthTargets>,
    #[serde(default)]
    pub setting: Option<SettingNotes>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub target_audience: Option<Option<String>>,
    #[serde(default)]
    pub comp_titles: Option<Vec<String>>,
    #[serde(default)]
    pub content_warnings: Option<Vec<String>>,
    #[serde(default)]
    pub custom: Option<std::collections::HashMap<String, serde_json::Value>>,
}

impl From<PlanPatchBody> for PlanPatch {
    fn from(b: PlanPatchBody) -> Self {
        PlanPatch {
            logline: b.logline,
            synopsis: b.synopsis,
            premise: b.premise,
            themes: b.themes,
            central_conflict: b.central_conflict,
            plot_beats: b.plot_beats,
            style: b.style,
            length: b.length,
            setting: b.setting,
            notes: b.notes,
            target_audience: b.target_audience,
            comp_titles: b.comp_titles,
            content_warnings: b.content_warnings,
            custom: b.custom,
        }
    }
}

pub async fn patch_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<PlanPatchBody>,
) -> impl IntoResponse {
    match plan::patch_plan(state.hypergraph.store(), &narrative_id, body.into()) {
        Ok(p) => json_ok(&p),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── DELETE ─────────────────────────────────────────────────────

pub async fn delete_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match plan::delete_plan(state.hypergraph.store(), &narrative_id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}
