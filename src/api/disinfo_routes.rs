//! HTTP handlers for the disinfo extension (Sprint D1).
//!
//! Endpoints:
//! - `GET  /entities/:id/behavioral-fingerprint` — load (compute on first
//!   access) a per-actor [`BehavioralFingerprint`].
//! - `POST /entities/:id/behavioral-fingerprint/compute` — force recompute
//!   and persist.
//! - `GET  /narratives/:id/disinfo-fingerprint` — load (compute on first
//!   access) a per-narrative [`DisinformationFingerprint`].
//! - `POST /narratives/:id/disinfo-fingerprint/compute` — force recompute
//!   and persist.
//! - `POST /fingerprints/compare` — compare two persisted fingerprints
//!   (behavioral or disinfo) and return per-axis distances + composite +
//!   p-value + 95% CI + same-source verdict.
//!
//! All handlers respect the `disinfo` Cargo feature; when that feature is off
//! the module is not compiled into the API binary.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::disinfo::{
    behavioral_envelope, compare_fingerprints, disinfo_envelope, ensure_behavioral_fingerprint,
    ensure_disinfo_fingerprint, ComparisonKind, ComparisonTask,
};
use crate::error::TensaError;

fn parse_uuid(s: &str) -> Result<Uuid, TensaError> {
    Uuid::parse_str(s).map_err(|e| TensaError::InvalidQuery(format!("invalid UUID '{s}': {e}")))
}

// ─── Behavioral Fingerprint ────────────────────────────────

pub async fn get_behavioral_fingerprint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    handle_behavioral(&state, &id, false)
}

pub async fn compute_behavioral_fingerprint_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    handle_behavioral(&state, &id, true)
}

fn handle_behavioral(state: &AppState, id: &str, force: bool) -> axum::response::Response {
    let actor_id = match parse_uuid(id) {
        Ok(u) => u,
        Err(e) => return error_response(e).into_response(),
    };
    match ensure_behavioral_fingerprint(&state.hypergraph, &actor_id, force) {
        Ok(fp) => json_ok(&behavioral_envelope(&fp)),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Disinfo Fingerprint ───────────────────────────────────

pub async fn get_disinfo_fingerprint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    handle_disinfo(&state, &id, false)
}

pub async fn compute_disinfo_fingerprint_route(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    handle_disinfo(&state, &id, true)
}

fn handle_disinfo(state: &AppState, id: &str, force: bool) -> axum::response::Response {
    match ensure_disinfo_fingerprint(&state.hypergraph, id, force) {
        Ok(fp) => json_ok(&disinfo_envelope(&fp)),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Compare ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompareRequest {
    /// `"behavioral"` or `"disinfo"`. (Narrative-content comparisons go via
    /// the existing `/style/compare` endpoint.)
    pub kind: ComparisonKind,
    #[serde(default)]
    pub task: ComparisonTask,
    /// First fingerprint id — actor UUID for behavioral, narrative ID for disinfo.
    pub a_id: String,
    pub b_id: String,
}

pub async fn compare_fingerprints_route(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompareRequest>,
) -> axum::response::Response {
    let pair: Result<(serde_json::Value, serde_json::Value), TensaError> = match req.kind {
        ComparisonKind::Behavioral => {
            let a_uuid = match parse_uuid(&req.a_id) {
                Ok(u) => u,
                Err(e) => return error_response(e).into_response(),
            };
            let b_uuid = match parse_uuid(&req.b_id) {
                Ok(u) => u,
                Err(e) => return error_response(e).into_response(),
            };
            load_behavioral_pair(&state, &a_uuid, &b_uuid)
        }
        ComparisonKind::Disinfo => load_disinfo_pair(&state, &req.a_id, &req.b_id),
        ComparisonKind::Narrative => {
            return error_response(TensaError::InvalidQuery(
                "narrative-content comparison: use POST /style/compare instead".into(),
            ))
            .into_response();
        }
    };
    let (a_value, b_value) = match pair {
        Ok(v) => v,
        Err(e) => return error_response(e).into_response(),
    };
    match compare_fingerprints(req.kind, req.task, &a_value, &b_value) {
        Ok(comparison) => json_ok(&comparison),
        Err(e) => error_response(e).into_response(),
    }
}

fn load_behavioral_pair(
    state: &AppState,
    a: &Uuid,
    b: &Uuid,
) -> Result<(serde_json::Value, serde_json::Value), TensaError> {
    let fp_a = ensure_behavioral_fingerprint(&state.hypergraph, a, false)?;
    let fp_b = ensure_behavioral_fingerprint(&state.hypergraph, b, false)?;
    Ok((to_value(&fp_a)?, to_value(&fp_b)?))
}

fn load_disinfo_pair(
    state: &AppState,
    a: &str,
    b: &str,
) -> Result<(serde_json::Value, serde_json::Value), TensaError> {
    let fp_a = ensure_disinfo_fingerprint(&state.hypergraph, a, false)?;
    let fp_b = ensure_disinfo_fingerprint(&state.hypergraph, b, false)?;
    Ok((to_value(&fp_a)?, to_value(&fp_b)?))
}

fn to_value<T: serde::Serialize>(value: &T) -> Result<serde_json::Value, TensaError> {
    serde_json::to_value(value).map_err(|e| TensaError::Serialization(e.to_string()))
}
