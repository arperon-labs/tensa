//! API endpoints for narrative adaptation: essentiality, compression, expansion, diff.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::narrative::compression::{
    compress_narrative, compress_to_novella, compress_to_screenplay_outline,
    compress_to_short_story, preview_compression, CompressionConfig,
};
use crate::narrative::diff::diff_narratives;
use crate::narrative::essentiality::{compute_essentiality, store_essentiality};
use crate::narrative::expansion::{
    add_subplot_to, expand_narrative, expand_to_novel, preview_expansion, ExpansionConfig,
};

// ─── Essentiality ────────────────────────────────────────────

/// POST /narratives/:id/essentiality — compute essentiality for situations/entities/subplots.
pub async fn essentiality(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match compute_essentiality(&state.hypergraph, &narrative_id) {
        Ok(report) => {
            let _ = store_essentiality(&state.hypergraph, &report);
            json_ok(&report)
        }
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Compression ────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub struct CompressBody {
    #[serde(default)]
    pub config: Option<CompressionConfig>,
    /// Preset: novella | short_story | screenplay_outline. Overrides config.
    pub preset: Option<String>,
}

/// POST /narratives/:id/compress — execute (plan-only) compression.
pub async fn compress(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<Option<CompressBody>>,
) -> impl IntoResponse {
    let body = body.unwrap_or_default();
    let result = match body.preset.as_deref() {
        Some("novella") => compress_to_novella(&state.hypergraph, &narrative_id),
        Some("short_story") => compress_to_short_story(&state.hypergraph, &narrative_id),
        Some("screenplay_outline") => {
            compress_to_screenplay_outline(&state.hypergraph, &narrative_id)
        }
        _ => compress_narrative(
            &state.hypergraph,
            &narrative_id,
            &body.config.unwrap_or_default(),
        ),
    };
    match result {
        Ok(plan) => json_ok(&plan),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/compress/preview — dry-run preview.
pub async fn compress_preview(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<Option<CompressBody>>,
) -> impl IntoResponse {
    let body = body.unwrap_or_default();
    let cfg = body.config.unwrap_or_default();
    match preview_compression(&state.hypergraph, &narrative_id, &cfg) {
        Ok(p) => json_ok(&p),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Expansion ──────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub struct ExpandBody {
    #[serde(default)]
    pub config: Option<ExpansionConfig>,
    /// Preset: novel (requires target_chapters).
    pub preset: Option<String>,
    pub target_chapters: Option<usize>,
}

pub async fn expand(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<Option<ExpandBody>>,
) -> impl IntoResponse {
    let body = body.unwrap_or_default();
    let result = if body.preset.as_deref() == Some("novel") {
        let target = body.target_chapters.unwrap_or(30);
        expand_to_novel(&state.hypergraph, &narrative_id, target)
    } else {
        expand_narrative(
            &state.hypergraph,
            &narrative_id,
            &body.config.unwrap_or_default(),
        )
    };
    match result {
        Ok(plan) => json_ok(&plan),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn expand_preview(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<Option<ExpandBody>>,
) -> impl IntoResponse {
    let body = body.unwrap_or_default();
    let cfg = body.config.unwrap_or_default();
    match preview_expansion(&state.hypergraph, &narrative_id, &cfg) {
        Ok(p) => json_ok(&p),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct AddSubplotBody {
    pub theme: String,
    pub relation: String,
}

pub async fn add_subplot(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<AddSubplotBody>,
) -> impl IntoResponse {
    match add_subplot_to(
        &state.hypergraph,
        &narrative_id,
        &body.theme,
        &body.relation,
    ) {
        Ok(plan) => json_ok(&plan),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Diff ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DiffBody {
    pub narrative_a: String,
    pub narrative_b: String,
}

/// POST /narratives/diff — structural diff.
pub async fn diff(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DiffBody>,
) -> impl IntoResponse {
    match diff_narratives(&state.hypergraph, &body.narrative_a, &body.narrative_b) {
        Ok(d) => json_ok(&d),
        Err(e) => error_response(e).into_response(),
    }
}
