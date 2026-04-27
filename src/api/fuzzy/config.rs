//! Per-workspace default t-norm + aggregator persistence.
//!
//! `GET /fuzzy/config` / `PUT /fuzzy/config` — round-trip the
//! [`super::FuzzyWorkspaceConfig`] struct. A `PUT` body with
//! `{"reset": true}` restores the built-in Godel / Mean default (Phase
//! 4 test T8).

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

use super::{load_workspace_config, save_workspace_config, FuzzyWorkspaceConfig};

/// GET /fuzzy/config — returns the current default (or Godel/Mean if none).
pub async fn get_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = load_workspace_config(state.hypergraph.store());
    json_ok(&cfg)
}

/// Body shape for `PUT /fuzzy/config`. Every field is optional; `reset =
/// true` short-circuits to the Godel/Mean factory default.
#[derive(Debug, Deserialize, Default)]
pub struct PutConfigBody {
    #[serde(default)]
    pub tnorm: Option<String>,
    #[serde(default)]
    pub aggregator: Option<String>,
    #[serde(default)]
    pub measure: Option<String>,
    #[serde(default)]
    pub reset: bool,
}

/// PUT /fuzzy/config — validates via the registries before writing.
pub async fn put_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PutConfigBody>,
) -> impl IntoResponse {
    if body.reset {
        let defaults = FuzzyWorkspaceConfig::default();
        if let Err(e) = save_workspace_config(state.hypergraph.store(), &defaults) {
            return error_response(e).into_response();
        }
        tracing::info!("fuzzy config reset to Godel/Mean default");
        return json_ok(&defaults);
    }

    let mut cfg = load_workspace_config(state.hypergraph.store());
    if let Some(t) = body.tnorm {
        cfg.tnorm = t;
    }
    if let Some(a) = body.aggregator {
        cfg.aggregator = a;
    }
    if body.measure.is_some() {
        cfg.measure = body.measure;
    }

    if cfg.tnorm.trim().is_empty() || cfg.aggregator.trim().is_empty() {
        return error_response(TensaError::InvalidInput(
            "fuzzy config must carry non-empty tnorm and aggregator fields".into(),
        ))
        .into_response();
    }

    if let Err(e) = save_workspace_config(state.hypergraph.store(), &cfg) {
        return error_response(e).into_response();
    }
    tracing::info!("fuzzy config updated: tnorm={} aggregator={}", cfg.tnorm, cfg.aggregator);
    json_ok(&cfg)
}
