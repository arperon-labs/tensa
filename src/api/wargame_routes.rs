//! REST surface for the adversarial wargame subsystem.
//!
//! Thin HTTP wrapper over [`crate::adversarial::session::WargameSession`]
//! CRUD + the turn / auto-play primitives in [`crate::adversarial::wargame`].
//! Mirrors the response shape consumed by [studio/src/views/Wargame.tsx](../../../studio/src/views/Wargame.tsx)
//! — every list / state endpoint emits the [`WargameSummaryResponse`] shape
//! (a [`SessionSummary`] augmented with `narrative_id`) so the Studio's
//! `WargameSession` TS interface stays a one-to-one mirror.
//!
//! The same lifecycle is also reachable via the MCP tools
//! (`create_wargame`, `submit_wargame_move`, `auto_play_wargame`,
//! `get_wargame_state`); this module exists so the Studio Wargame view
//! does not need to talk MCP.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::adversarial::session::{
    delete_session, list_sessions, load_session, store_session, BackgroundSubstrate, Objective,
    SessionStatus, SessionSummary, WargameConfig, WargameSession,
};
use crate::adversarial::wargame::{TurnResult, WargameMove};
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

// ─── Request / response shapes ─────────────────────────────────────────

/// Body for `POST /wargame/sessions`. All scheduling fields are optional —
/// omitted fields fall back to [`WargameConfig::default`].
#[derive(Debug, Deserialize)]
pub struct CreateWargameRequest {
    pub narrative_id: String,
    #[serde(default)]
    pub max_turns: Option<usize>,
    #[serde(default)]
    pub time_step_minutes: Option<u64>,
    #[serde(default)]
    pub auto_red: Option<bool>,
    #[serde(default)]
    pub auto_blue: Option<bool>,
    #[serde(default)]
    pub red_objectives: Option<Vec<Objective>>,
    #[serde(default)]
    pub blue_objectives: Option<Vec<Objective>>,
    #[serde(default)]
    pub background: Option<BackgroundSubstrate>,
}

/// Body for `POST /wargame/sessions/:id/auto-play`.
#[derive(Debug, Deserialize)]
pub struct AutoPlayRequest {
    #[serde(default = "default_auto_play_turns")]
    pub num_turns: usize,
}

fn default_auto_play_turns() -> usize {
    1
}

/// Body for `POST /wargame/sessions/:id/moves`.
#[derive(Debug, Deserialize)]
pub struct SubmitMovesRequest {
    #[serde(default)]
    pub red_moves: Vec<WargameMove>,
    #[serde(default)]
    pub blue_moves: Vec<WargameMove>,
}

/// Studio-facing summary: `SessionSummary` with `narrative_id` attached.
/// The Studio's `WargameSession` TS interface expects exactly this shape.
#[derive(Debug, Serialize)]
pub struct WargameSummaryResponse {
    pub session_id: String,
    pub narrative_id: String,
    pub status: SessionStatus,
    pub turn: usize,
    pub max_turns: usize,
    pub r0: f64,
    pub misinformed: f64,
    pub susceptible: f64,
    pub red_objectives_met: usize,
    pub blue_objectives_met: usize,
    pub total_moves: usize,
}

impl WargameSummaryResponse {
    fn from_session(session: &WargameSession) -> Self {
        let SessionSummary {
            session_id,
            status,
            turn,
            max_turns,
            r0,
            misinformed,
            susceptible,
            red_objectives_met,
            blue_objectives_met,
            total_moves,
        } = session.summary();
        Self {
            session_id,
            narrative_id: session.narrative_id.clone(),
            status,
            turn,
            max_turns,
            r0,
            misinformed,
            susceptible,
            red_objectives_met,
            blue_objectives_met,
            total_moves,
        }
    }
}

// ─── Handlers ──────────────────────────────────────────────────────────

/// `POST /wargame/sessions` — create a new wargame session.
pub async fn create_wargame_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateWargameRequest>,
) -> impl IntoResponse {
    if req.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }

    let defaults = WargameConfig::default();
    let config = WargameConfig {
        max_turns: req.max_turns.unwrap_or(defaults.max_turns),
        time_step_minutes: req.time_step_minutes.unwrap_or(defaults.time_step_minutes),
        red_objectives: req.red_objectives.unwrap_or(defaults.red_objectives),
        blue_objectives: req.blue_objectives.unwrap_or(defaults.blue_objectives),
        auto_red: req.auto_red.unwrap_or(defaults.auto_red),
        auto_blue: req.auto_blue.unwrap_or(defaults.auto_blue),
        background: req.background.or(defaults.background),
    };

    let session = match WargameSession::create(&state.hypergraph, &req.narrative_id, config) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };

    if let Err(e) = store_session(&state.hypergraph, &session) {
        return error_response(e).into_response();
    }

    (
        StatusCode::CREATED,
        Json(serde_json::json!({"session_id": session.session_id})),
    )
        .into_response()
}

/// `GET /wargame/sessions` — list all sessions as Studio-facing summaries.
pub async fn list_wargame_sessions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match list_sessions(&state.hypergraph) {
        Ok(sessions) => {
            let summaries: Vec<WargameSummaryResponse> = sessions
                .iter()
                .map(WargameSummaryResponse::from_session)
                .collect();
            json_ok(&summaries)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /wargame/sessions/:id/state` — current summary for one session.
pub async fn get_wargame_state(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match load_session(&state.hypergraph, &session_id) {
        Ok(Some(session)) => json_ok(&WargameSummaryResponse::from_session(&session)),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "wargame session {session_id} not found"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// `POST /wargame/sessions/:id/auto-play` — advance N turns using the
/// auto-play heuristics for whichever team(s) have `auto_red` / `auto_blue`
/// set on the session config.
///
/// `num_turns` × `advance_turn` is CPU-bound and unbounded by `num_turns`,
/// so the loop runs on `spawn_blocking` to keep the tokio worker free —
/// matches the convention in `routes::ingest_text` (routes.rs:2869).
pub async fn auto_play_wargame(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<AutoPlayRequest>,
) -> impl IntoResponse {
    let session = match load_session(&state.hypergraph, &session_id) {
        Ok(Some(s)) => s,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!(
                "wargame session {session_id} not found"
            )))
            .into_response();
        }
        Err(e) => return error_response(e).into_response(),
    };

    let join = tokio::task::spawn_blocking(move || {
        let mut session = session;
        let results = session.auto_play(req.num_turns)?;
        Ok::<_, TensaError>((session, results))
    })
    .await;

    let (session, results) = match join {
        Ok(Ok(pair)) => pair,
        Ok(Err(e)) => return error_response(e).into_response(),
        Err(e) => {
            return error_response(TensaError::Internal(format!(
                "auto_play task panicked: {e}"
            )))
            .into_response();
        }
    };

    if let Err(e) = store_session(&state.hypergraph, &session) {
        return error_response(e).into_response();
    }

    json_ok(&results)
}

/// `POST /wargame/sessions/:id/moves` — submit explicit moves for one turn.
pub async fn submit_wargame_moves(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<SubmitMovesRequest>,
) -> impl IntoResponse {
    let mut session = match load_session(&state.hypergraph, &session_id) {
        Ok(Some(s)) => s,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!(
                "wargame session {session_id} not found"
            )))
            .into_response();
        }
        Err(e) => return error_response(e).into_response(),
    };

    let result = match session.play_turn(&req.red_moves, &req.blue_moves) {
        Ok(r) => r,
        Err(e) => return error_response(e).into_response(),
    };

    if let Err(e) = store_session(&state.hypergraph, &session) {
        return error_response(e).into_response();
    }

    json_ok(&result)
}

/// `DELETE /wargame/sessions/:id` — idempotent delete.
pub async fn delete_wargame_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match delete_session(&state.hypergraph, &session_id) {
        Ok(()) => json_ok(&serde_json::json!({"ok": true})),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adversarial::session::WargameSession;
    use crate::hypergraph::Hypergraph;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn t_summary_response_carries_narrative_id() {
        // WargameSummaryResponse::from_session must surface the session's
        // substrate narrative_id alongside the SessionSummary fields the
        // Studio Wargame view binds to.
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let config = WargameConfig {
            max_turns: 7,
            ..Default::default()
        };
        let session = WargameSession::create(&hg, "narr-1", config)
            .expect("session create on empty narrative should succeed");

        let resp = WargameSummaryResponse::from_session(&session);
        assert_eq!(resp.narrative_id, "narr-1");
        assert_eq!(resp.session_id, session.session_id);
        assert_eq!(resp.max_turns, 7);
        assert_eq!(resp.turn, 0);
    }
}
