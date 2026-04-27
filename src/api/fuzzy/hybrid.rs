//! `POST /fuzzy/hybrid/probability`, `GET /fuzzy/hybrid/probability/{nid}`
//! (list), `GET /fuzzy/hybrid/probability/{nid}/{query_id}`,
//! `DELETE /fuzzy/hybrid/probability/{nid}/{query_id}`.
//!
//! Fuzzy Sprint Phase 10 — synchronous Cao–Holčapek base-case fuzzy-
//! probability evaluation. The endpoint validates the discrete
//! distribution (`Σ P = 1 ± 1e-9`, `P ∈ [0, 1]`, no duplicate UUIDs),
//! dispatches on the event predicate kind, and persists a
//! [`HybridProbabilityReport`] at `fz/hybrid/{nid}/{query_id_BE_16}` so
//! callers can re-read the run later without recomputing.
//!
//! Synchronous path: the numeric work is `O(|outcomes| + |entities|)`
//! per request, fitting well inside a normal HTTP lifetime.
//!
//! Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::hybrid::{
    build_hybrid_report, delete_hybrid_result, fuzzy_probability, list_hybrid_results_for_narrative,
    load_hybrid_result, save_hybrid_result, FuzzyEvent, HybridProbabilityReport, ProbDist,
};
use crate::fuzzy::registry::TNormRegistry;
use crate::fuzzy::tnorm::TNormKind;

/// Body for `POST /fuzzy/hybrid/probability`.
#[derive(Debug, Deserialize)]
pub struct HybridProbabilityBody {
    pub narrative_id: String,
    pub event: FuzzyEvent,
    pub distribution: ProbDist,
    /// Optional t-norm override. Phase 10 accepts but does not consume
    /// it (base case has no composition step). Surface tolerates
    /// unknown names by returning 400.
    #[serde(default)]
    pub tnorm: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HybridProbabilityResponse {
    pub value: f64,
    pub event_kind: String,
    pub distribution_summary: String,
    pub query_id: Uuid,
    pub narrative_id: String,
    pub tnorm: String,
}

/// POST /fuzzy/hybrid/probability
pub async fn compute_probability(
    State(state): State<Arc<AppState>>,
    Json(body): Json<HybridProbabilityBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }

    let tnorm: TNormKind = match body.tnorm.as_deref() {
        None | Some("") => TNormKind::Godel,
        Some(name) => match TNormRegistry::default().get(name) {
            Ok(k) => k,
            Err(e) => return error_response(e).into_response(),
        },
    };

    let hg = &state.hypergraph;
    let value = match fuzzy_probability(hg, &body.narrative_id, &body.event, &body.distribution, tnorm) {
        Ok(v) => v,
        Err(e) => return error_response(e).into_response(),
    };

    let report = build_hybrid_report(
        &body.narrative_id,
        &body.event,
        &body.distribution,
        tnorm,
        value,
    );
    let query_id = report.query_id;
    if let Err(e) = save_hybrid_result(hg.store(), &body.narrative_id, &report) {
        tracing::warn!(
            narrative_id = %body.narrative_id,
            "failed to persist hybrid-probability report ({e}); returning inline anyway"
        );
    }

    json_ok(&HybridProbabilityResponse {
        value,
        event_kind: report.event_kind.clone(),
        distribution_summary: report.distribution_summary.clone(),
        query_id,
        narrative_id: body.narrative_id,
        tnorm: report.tnorm.clone(),
    })
    .into_response()
}

/// GET /fuzzy/hybrid/probability/{nid}/{query_id}
pub async fn get_probability(
    State(state): State<Arc<AppState>>,
    Path((nid, query_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_hybrid_result(state.hypergraph.store(), &nid, &query_id) {
        Ok(Some(r)) => json_ok(&r).into_response(),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "no hybrid-probability record at {nid}/{query_id}"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /fuzzy/hybrid/probability/{nid} — list every report for a
/// narrative (newest first, max 200 per request).
pub async fn list_probability(
    State(state): State<Arc<AppState>>,
    Path(nid): Path<String>,
) -> impl IntoResponse {
    match list_hybrid_results_for_narrative(state.hypergraph.store(), &nid) {
        Ok(reports) => {
            let page: Vec<HybridProbabilityReport> = reports.into_iter().take(200).collect();
            json_ok(&page).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /fuzzy/hybrid/probability/{nid}/{query_id}
pub async fn delete_probability(
    State(state): State<Arc<AppState>>,
    Path((nid, query_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    if let Err(e) = delete_hybrid_result(state.hypergraph.store(), &nid, &query_id) {
        return error_response(e).into_response();
    }
    json_ok(&serde_json::json!({"deleted": true, "narrative_id": nid, "query_id": query_id}))
        .into_response()
}
