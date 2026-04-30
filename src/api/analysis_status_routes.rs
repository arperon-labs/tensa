//! REST routes for the analysis-status registry.
//!
//! - `GET    /narratives/:id/analysis-status`              — list rows for a narrative
//! - `POST   /narratives/:id/analysis-status`              — upsert (used by skill / manual)
//! - `PATCH  /narratives/:id/analysis-status/:job_type`    — toggle the `locked` flag
//! - `DELETE /narratives/:id/analysis-status/:job_type`    — remove the row
//!
//! Scope is the optional query / body parameter `scope` (default `"story"`).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use std::str::FromStr;

use crate::analysis_status::{
    AnalysisSource, AnalysisStatusEntry, AnalysisStatusStore, ResultRef,
};
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::types::InferenceJobType;

#[derive(Debug, Deserialize)]
pub struct ScopeQuery {
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertBody {
    pub job_type: InferenceJobType,
    #[serde(default)]
    pub scope: Option<String>,
    pub source: AnalysisSource,
    #[serde(default)]
    pub skill: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub locked: Option<bool>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub result_refs: Vec<ResultRef>,
    #[serde(default)]
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct LockBody {
    pub locked: bool,
    #[serde(default)]
    pub scope: Option<String>,
}

pub async fn list_status(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let store = AnalysisStatusStore::new(state.hypergraph.store_arc());
    match store.list_for_narrative(&narrative_id) {
        Ok(rows) => json_ok(&serde_json::json!({
            "narrative_id": narrative_id,
            "entries": rows,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn upsert_status(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<UpsertBody>,
) -> impl IntoResponse {
    let scope = body.scope.unwrap_or_else(|| "story".to_string());
    let locked = body.locked.unwrap_or_else(|| body.source.default_locked());
    let entry = AnalysisStatusEntry {
        narrative_id: narrative_id.clone(),
        job_type: body.job_type,
        scope,
        source: body.source,
        skill: body.skill,
        model: body.model,
        completed_at: body.completed_at.unwrap_or_else(chrono::Utc::now),
        locked,
        summary: body.summary,
        confidence: body.confidence,
        result_refs: body.result_refs,
    };
    let store = AnalysisStatusStore::new(state.hypergraph.store_arc());
    match store.upsert(&entry) {
        Ok(()) => (StatusCode::OK, Json(serde_json::to_value(&entry).unwrap_or_default()))
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn set_lock(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, job_type_str)): Path<(String, String)>,
    Json(body): Json<LockBody>,
) -> impl IntoResponse {
    let jt = match InferenceJobType::from_str(&job_type_str) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };
    let scope = body.scope.unwrap_or_else(|| "story".to_string());
    let store = AnalysisStatusStore::new(state.hypergraph.store_arc());
    match store.set_locked(&narrative_id, &jt, &scope, body.locked) {
        Ok(updated) => json_ok(&updated),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn delete_status(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, job_type_str)): Path<(String, String)>,
    Query(q): Query<ScopeQuery>,
) -> impl IntoResponse {
    let jt = match InferenceJobType::from_str(&job_type_str) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };
    let scope = q.scope.unwrap_or_else(|| "story".to_string());
    let store = AnalysisStatusStore::new(state.hypergraph.store_arc());
    match store.delete(&narrative_id, &jt, &scope) {
        Ok(existed) => json_ok(&serde_json::json!({"deleted": existed})),
        Err(e) => error_response(e).into_response(),
    }
}
