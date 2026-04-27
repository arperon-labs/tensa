//! Actor archetype + disinfo assessment API routes (Sprint D5).
//!
//! Endpoints:
//! - `POST /actors/:id/archetype`   — classify an actor into adversarial archetypes.
//! - `GET  /actors/:id/archetype`   — get cached archetype classification.
//! - `POST /analysis/disinfo-assess` — fuse multiple disinfo signals via DS.
//! - `GET  /analysis/disinfo-assess/:id` — get cached disinfo assessment.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

#[derive(Debug, Deserialize)]
pub struct ArchetypeRequest {
    #[serde(default)]
    pub force: bool,
}

/// POST /actors/:id/archetype — classify an actor into adversarial archetypes.
pub async fn classify_archetype(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ArchetypeRequest>,
) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };

    // Check for cached result unless force-recompute requested
    if !req.force {
        if let Ok(Some(cached)) =
            crate::disinfo::archetypes::load_archetype(&state.hypergraph, &uuid)
        {
            return json_ok(&cached);
        }
    }

    match crate::disinfo::archetypes::classify_actor_archetype(&state.hypergraph, uuid) {
        Ok(dist) => json_ok(&dist),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /actors/:id/archetype — get cached archetype classification.
pub async fn get_archetype(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    match crate::disinfo::archetypes::load_archetype(&state.hypergraph, &uuid) {
        Ok(Some(dist)) => json_ok(&dist),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "No archetype for actor {}",
            id
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct DisinfoAssessRequest {
    pub target_id: String,
    pub signals: Vec<crate::disinfo::fusion::DisinfoSignal>,
}

/// POST /analysis/disinfo-assess — fuse multiple disinfo signals via DS.
pub async fn assess_disinfo(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DisinfoAssessRequest>,
) -> Response {
    match crate::disinfo::fusion::fuse_disinfo_signals(&req.target_id, &req.signals) {
        Ok(assessment) => {
            let _ = crate::disinfo::fusion::store_assessment(&state.hypergraph, &assessment);
            json_ok(&assessment)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /analysis/disinfo-assess/:id — get cached disinfo assessment.
pub async fn get_assessment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    match crate::disinfo::fusion::load_assessment(&state.hypergraph, &id) {
        Ok(Some(a)) => json_ok(&a),
        Ok(None) => error_response(TensaError::NotFound(format!("No assessment for {}", id)))
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}
