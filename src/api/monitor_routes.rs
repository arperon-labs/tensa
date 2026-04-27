//! HTTP handlers for the disinfo monitor subscriptions (Sprint D7.3).
//!
//! Endpoints:
//! - `POST   /monitor/subscriptions` — create a new subscription
//! - `GET    /monitor/subscriptions` — list subscriptions (?active_only=true)
//! - `DELETE /monitor/subscriptions/:id` — delete a subscription
//! - `GET    /monitor/alerts` — list recent alerts (?limit=N)
//!
//! Cfg-gated behind `disinfo`.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::disinfo::monitor;
use crate::error::TensaError;

// ─── Create subscription ──────────────────────────────────────

pub async fn create_subscription(
    State(state): State<Arc<AppState>>,
    Json(body): Json<monitor::MonitorSubscription>,
) -> axum::response::Response {
    match monitor::create_subscription(&state.hypergraph, body) {
        Ok(sub) => json_ok(&sub),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── List subscriptions ───────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ListSubsQuery {
    #[serde(default)]
    pub active_only: Option<bool>,
}

pub async fn list_subscriptions(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ListSubsQuery>,
) -> axum::response::Response {
    let active_only = q.active_only.unwrap_or(false);
    match monitor::list_subscriptions(&state.hypergraph, active_only) {
        Ok(subs) => json_ok(&subs),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Delete subscription ──────────────────────────────────────

pub async fn delete_subscription(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    let uuid = match Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("invalid UUID: {e}")))
                .into_response();
        }
    };
    match monitor::delete_subscription(&state.hypergraph, &uuid) {
        Ok(()) => json_ok(&serde_json::json!({"deleted": id})),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── List alerts ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ListAlertsQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    100
}

pub async fn list_alerts(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ListAlertsQuery>,
) -> axum::response::Response {
    match monitor::list_alerts(&state.hypergraph, q.limit) {
        Ok(alerts) => json_ok(&alerts),
        Err(e) => error_response(e).into_response(),
    }
}
