//! HTTP handlers for the disinfo spread-dynamics extension (Sprint D2).
//!
//! Endpoints:
//! - `POST /spread/r0` — body `{narrative_id, fact, about_entity, narrative_kind?, beta_overrides?}`.
//!   Runs SMIR + per-platform R₀, detects cross-platform jumps, fires velocity
//!   alerts. Returns the same payload the inference engine produces (so MCP +
//!   REST + Studio share one shape).
//! - `GET /spread/r0/:narrative_id` — load the most recent persisted SMIR result.
//! - `GET /spread/velocity/:narrative_id?limit=N` — recent velocity alerts.
//! - `GET /spread/jumps/:narrative_id` — persisted cross-platform jumps.
//! - `POST /spread/intervention` — body `{narrative_id, fact, about_entity,
//!   intervention, beta_overrides?}`. Counterfactual `RemoveTopAmplifiers` /
//!   `DebunkAt` projection.
//!
//! Cfg-gated behind `disinfo`.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::analysis::contagion::{
    compute_spread_r0_payload, list_cross_platform_jumps, load_smir_result, parse_beta_overrides,
};
use crate::analysis::spread_intervention::{simulate_intervention, Intervention};
use crate::analysis::velocity_monitor::VelocityMonitor;
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::types::Platform;

#[derive(Debug, Deserialize)]
pub struct SpreadR0Request {
    pub narrative_id: String,
    pub fact: String,
    pub about_entity: String,
    #[serde(default)]
    pub narrative_kind: Option<String>,
    /// Optional `{ "twitter": 0.5, "telegram": 0.3 }` overrides.
    #[serde(default)]
    pub beta_overrides: Option<std::collections::HashMap<String, f64>>,
}

fn parse_uuid(s: &str) -> Result<Uuid, TensaError> {
    Uuid::parse_str(s).map_err(|e| TensaError::InvalidQuery(format!("invalid UUID '{s}': {e}")))
}

fn parse_beta_map(
    overrides: Option<std::collections::HashMap<String, f64>>,
) -> Vec<(Platform, f64)> {
    parse_beta_overrides(overrides.unwrap_or_default())
}

/// `POST /spread/r0` — run SMIR + jumps + velocity check, return aggregated payload.
pub async fn run_spread_r0(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpreadR0Request>,
) -> axum::response::Response {
    let about_entity = match parse_uuid(&req.about_entity) {
        Ok(u) => u,
        Err(e) => return error_response(e).into_response(),
    };
    let beta_overrides = parse_beta_map(req.beta_overrides);
    let narrative_kind = req.narrative_kind.unwrap_or_else(|| "default".to_string());
    match compute_spread_r0_payload(
        &state.hypergraph,
        &req.narrative_id,
        &req.fact,
        about_entity,
        &narrative_kind,
        &beta_overrides,
    ) {
        Ok(payload) => json_ok(&payload),
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /spread/r0/:narrative_id` — load most recent persisted SMIR result.
pub async fn get_spread_r0(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> axum::response::Response {
    match load_smir_result(&state.hypergraph, &narrative_id) {
        Ok(Some(result)) => json_ok(&result),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "no SMIR result for narrative '{narrative_id}' — POST /spread/r0 to compute"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct VelocityQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

/// `GET /spread/velocity/:narrative_id?limit=N` — recent velocity alerts.
pub async fn list_velocity_alerts(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<VelocityQuery>,
) -> axum::response::Response {
    let monitor = VelocityMonitor::new(&state.hypergraph);
    match monitor.recent_alerts(&narrative_id, q.limit.min(500)) {
        Ok(alerts) => json_ok(&serde_json::json!({"alerts": alerts})),
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /spread/jumps/:narrative_id` — persisted cross-platform jumps.
pub async fn get_cross_platform_jumps(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> axum::response::Response {
    match list_cross_platform_jumps(&state.hypergraph, &narrative_id) {
        Ok(jumps) => json_ok(&serde_json::json!({"jumps": jumps})),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct InterventionRequest {
    pub narrative_id: String,
    pub fact: String,
    pub about_entity: String,
    pub intervention: Intervention,
    #[serde(default)]
    pub beta_overrides: Option<std::collections::HashMap<String, f64>>,
}

/// `POST /spread/intervention` — counterfactual projection.
pub async fn run_intervention(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InterventionRequest>,
) -> axum::response::Response {
    let about_entity = match parse_uuid(&req.about_entity) {
        Ok(u) => u,
        Err(e) => return error_response(e).into_response(),
    };
    let beta_overrides = parse_beta_map(req.beta_overrides);
    match simulate_intervention(
        &state.hypergraph,
        &req.narrative_id,
        &req.fact,
        about_entity,
        req.intervention,
        &beta_overrides,
    ) {
        Ok(projection) => json_ok(&projection),
        Err(e) => error_response(e).into_response(),
    }
}
