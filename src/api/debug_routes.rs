//! API endpoints for the narrative debugger.

use std::str::FromStr;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::narrative::debug::{
    diagnose_chapter, diagnose_narrative, diagnose_narrative_with, load_diagnosis, store_diagnosis,
    DiagnosticConfig, GenrePreset, PathologySeverity,
};
use crate::narrative::debug_fixes::{apply_fix, auto_repair, suggest_fixes, SuggestedFix};

#[derive(Debug, Deserialize, Default)]
pub struct DiagnoseBody {
    pub genre: Option<String>,
    pub config: Option<DiagnosticConfig>,
}

/// POST /narratives/:id/diagnose — Run full structural diagnosis.
pub async fn diagnose(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<Option<DiagnoseBody>>,
) -> impl IntoResponse {
    let body = body.unwrap_or_default();
    let cfg = if let Some(c) = body.config {
        c
    } else {
        let g = body
            .genre
            .as_deref()
            .and_then(|s| GenrePreset::from_str(s).ok())
            .unwrap_or_default();
        DiagnosticConfig::for_genre(g)
    };
    match diagnose_narrative_with(&state.hypergraph, &narrative_id, &cfg) {
        Ok(diag) => {
            let _ = store_diagnosis(&state.hypergraph, &diag);
            json_ok(&diag)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/diagnostics — Latest stored diagnosis.
pub async fn get_diagnostics(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match load_diagnosis(&state.hypergraph, &narrative_id) {
        Ok(Some(diag)) => json_ok(&diag),
        Ok(None) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "no diagnosis on file; run POST /narratives/:id/diagnose first" })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/health-score — quick health snapshot.
pub async fn get_health_score(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match diagnose_narrative(&state.hypergraph, &narrative_id) {
        Ok(diag) => json_ok(&serde_json::json!({
            "narrative_id": narrative_id,
            "health_score": diag.health_score,
            "errors": diag.error_count,
            "warnings": diag.warning_count,
            "infos": diag.info_count,
            "worst_chapter": diag.summary.worst_chapter,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/diagnose-chapter/:n — Diagnose a single chapter.
pub async fn diagnose_single_chapter(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, chapter)): Path<(String, usize)>,
) -> impl IntoResponse {
    match diagnose_chapter(&state.hypergraph, &narrative_id, chapter) {
        Ok(paths) => json_ok(&serde_json::json!({
            "narrative_id": narrative_id,
            "chapter": chapter,
            "pathologies": paths,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/suggest-fixes — Suggest fixes for latest diagnosis.
pub async fn suggest_narrative_fixes(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let diag = match load_diagnosis(&state.hypergraph, &narrative_id) {
        Ok(Some(d)) => d,
        Ok(None) => match diagnose_narrative(&state.hypergraph, &narrative_id) {
            Ok(d) => d,
            Err(e) => return error_response(e).into_response(),
        },
        Err(e) => return error_response(e).into_response(),
    };
    match suggest_fixes(&state.hypergraph, &diag.pathologies) {
        Ok(fixes) => json_ok(&serde_json::json!({
            "narrative_id": narrative_id,
            "pathology_count": diag.pathologies.len(),
            "fixes": fixes,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/apply-fix — Apply a single supplied fix.
pub async fn apply_narrative_fix(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(fix): Json<SuggestedFix>,
) -> impl IntoResponse {
    match apply_fix(&state.hypergraph, &narrative_id, &fix) {
        Ok(res) => json_ok(&res),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct AutoRepairBody {
    pub max_severity: Option<String>,
    pub max_iterations: Option<usize>,
}

/// POST /narratives/:id/auto-repair — Diagnose+fix loop.
pub async fn auto_repair_narrative(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<Option<AutoRepairBody>>,
) -> impl IntoResponse {
    let body = body.unwrap_or_default();
    let sev = body
        .max_severity
        .as_deref()
        .and_then(|s| PathologySeverity::from_str(s).ok())
        .unwrap_or(PathologySeverity::Warning);
    let iters = body.max_iterations.unwrap_or(5).clamp(1, 20);
    match auto_repair(&state.hypergraph, &narrative_id, sev, iters) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}
