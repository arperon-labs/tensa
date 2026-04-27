//! Multilingual & MCP source API routes (Sprint D6).

use std::sync::Arc;

use axum::extract::State;
use axum::response::Response;
use axum::Json;
use serde::Deserialize;

use crate::api::routes::json_ok;
use crate::api::server::AppState;

#[derive(Debug, Deserialize)]
pub struct LangDetectRequest {
    pub text: String,
}

/// POST /lang/detect — detect language of text.
pub async fn detect_language(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<LangDetectRequest>,
) -> Response {
    let result = crate::disinfo::multilingual::detect_language(&req.text);
    let normalized = crate::disinfo::multilingual::normalize_for_matching(&req.text);
    json_ok(&serde_json::json!({
        "language": result.language,
        "confidence": result.confidence,
        "normalized_text": normalized,
    }))
}

/// GET /ingest/mcp-sources — list configured MCP sources (placeholder).
pub async fn list_mcp_sources(State(_state): State<Arc<AppState>>) -> Response {
    // Placeholder — returns empty list until orchestrator is fully connected
    json_ok(&serde_json::json!({
        "sources": [],
        "note": "MCP source registry not yet loaded. Configure via tensa-mcp-sources.toml.",
    }))
}
