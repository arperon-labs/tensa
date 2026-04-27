//! REST routes for narrative/situation template library (Sprint W15).
//!
//! - `GET  /narrative-templates` — list builtin + stored templates
//! - `POST /narrative-templates/:id/instantiate` — bind slots to entities
//!
//! The ingestion-config `/templates` routes live in `routes.rs`; these are
//! the narrative scaffold templates from `src/narrative/templates.rs`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::narrative::templates::{
    builtin_templates, instantiate_template, list_templates, load_template, SlotBindings,
};

/// GET /narrative-templates — builtin + stored templates, deduplicated by id.
pub async fn list_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut out = builtin_templates();
    match list_templates(&state.hypergraph) {
        Ok(stored) => {
            let existing: std::collections::HashSet<Uuid> = out.iter().map(|t| t.id).collect();
            for t in stored {
                if !existing.contains(&t.id) {
                    out.push(t);
                }
            }
            json_ok(&out)
        }
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct InstantiateBody {
    /// Map of `slot_id -> entity_uuid`.
    pub bindings: SlotBindings,
}

/// POST /narrative-templates/:id/instantiate
pub async fn instantiate_handler(
    State(state): State<Arc<AppState>>,
    Path(template_id): Path<Uuid>,
    Json(body): Json<InstantiateBody>,
) -> impl IntoResponse {
    // Look up the template: try stored first, then fall back to builtins.
    let tpl = match load_template(&state.hypergraph, &template_id) {
        Ok(Some(t)) => t,
        Ok(None) => match builtin_templates()
            .into_iter()
            .find(|t| t.id == template_id)
        {
            Some(t) => t,
            None => {
                return error_response(TensaError::NotFound(format!(
                    "template {} not found",
                    template_id
                )))
                .into_response()
            }
        },
        Err(e) => return error_response(e).into_response(),
    };
    match instantiate_template(&tpl, &body.bindings) {
        Ok(inst) => json_ok(&inst),
        Err(e) => error_response(e).into_response(),
    }
}
