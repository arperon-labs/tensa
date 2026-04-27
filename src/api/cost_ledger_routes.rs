//! Cost ledger endpoints — Sprint W5 (v0.49.4).
//!
//! - `GET /narratives/:id/cost-ledger?limit=N`          — recent entries
//! - `GET /narratives/:id/cost-ledger/summary?window=30d` — rolled-up totals

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use serde::Deserialize;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::narrative::cost_ledger;

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 500;

#[derive(Debug, Deserialize, Default)]
pub struct LedgerListQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

pub async fn list_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<LedgerListQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    match cost_ledger::list(state.hypergraph.store(), &narrative_id, limit) {
        Ok(entries) => json_ok(&entries),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct SummaryQuery {
    /// Window label: "7d", "30d", "24h", "all" (default "30d").
    #[serde(default)]
    pub window: Option<String>,
}

pub async fn summary_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<SummaryQuery>,
) -> impl IntoResponse {
    let label = q.window.as_deref().unwrap_or("30d");
    let window = cost_ledger::parse_window(label);
    match cost_ledger::summary(state.hypergraph.store(), &narrative_id, window) {
        Ok(s) => json_ok(&s),
        Err(e) => error_response(e).into_response(),
    }
}
