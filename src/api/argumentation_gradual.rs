//! `POST /analysis/argumentation/gradual` — synchronous gradual-semantics
//! evaluation on a narrative's argumentation framework.
//!
//! Phase 3 of the Graded Acceptability sprint. Mirrors the synchronous
//! pattern established by `POST /analysis/higher-order-contagion` (Fuzzy
//! Sprint Phase 7b) and `POST /analysis/contagion-bistability` (EATH
//! Phase 14): the read-back endpoint at `GET /narratives/:id/arguments`
//! stays cacheable + idempotent (it returns stored blobs only), while
//! this endpoint runs the gradual-semantics pass on demand and returns
//! the result inline.
//!
//! The handler delegates to
//! [`crate::analysis::argumentation::run_argumentation_with_gradual`] so
//! the framework-construction + crisp-semantics path is shared with the
//! job-queue route (`POST /jobs` with `job_type =
//! ArgumentationAnalysis`); only the surface differs.
//!
//! Cites: [amgoud2013ranking], [besnard2001hcategoriser],
//!        [amgoud2017weighted].

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::analysis::argumentation::run_argumentation_with_gradual;
use crate::analysis::argumentation_gradual::{GradualResult, GradualSemanticsKind};
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::tnorm::TNormKind;

/// Request body for `POST /analysis/argumentation/gradual`.
#[derive(Debug, Deserialize)]
pub struct GradualArgumentationRequest {
    /// Source narrative whose contention edges form the framework.
    pub narrative_id: String,
    /// Which gradual semantics to run. Serde rejects unknown variants
    /// with a 422-style decode error which axum surfaces as HTTP 400.
    pub gradual_semantics: GradualSemanticsKind,
    /// Optional t-norm override for the influence step. `None` =
    /// canonical Gödel formula (bit-identical to the cited paper).
    #[serde(default)]
    pub tnorm: Option<TNormKind>,
}

/// Response body for `POST /analysis/argumentation/gradual`.
///
/// Echoes the resolved `narrative_id` + the full [`GradualResult`].
/// `iterations` and `converged` are duplicated at the envelope level so
/// callers that only care about telemetry can skip the per-argument
/// acceptability map.
#[derive(Debug, Serialize)]
pub struct GradualArgumentationResponse {
    pub narrative_id: String,
    pub gradual: GradualResult,
    pub iterations: u32,
    pub converged: bool,
}

/// `POST /analysis/argumentation/gradual` — run gradual semantics on a
/// narrative's argumentation framework synchronously.
///
/// Failure modes:
/// * `narrative_id` empty → HTTP 400 (`InvalidInput`).
/// * Gradual evaluation fails (e.g. weighted-h-categoriser construction
///   constraints) → HTTP 400 via `error_response`.
/// * Underlying hypergraph load failure → HTTP 500 via `error_response`.
pub async fn run_gradual(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GradualArgumentationRequest>,
) -> impl IntoResponse {
    let narrative_id = req.narrative_id.trim();
    if narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput(
            "narrative_id is required".into(),
        ))
        .into_response();
    }

    let result = match run_argumentation_with_gradual(
        &state.hypergraph,
        narrative_id,
        Some(req.gradual_semantics),
        req.tnorm,
    ) {
        Ok(r) => r,
        Err(e) => return error_response(e).into_response(),
    };

    // `run_argumentation_with_gradual` always returns a populated
    // `gradual` field when the caller passes `Some(kind)` — pull it out
    // and surface a clear 500 rather than panicking if the invariant
    // ever drifts.
    let gradual = match result.gradual {
        Some(g) => g,
        None => {
            return error_response(TensaError::InferenceError(
                "argumentation engine returned no gradual result for a gradual request".into(),
            ))
            .into_response();
        }
    };

    let response = GradualArgumentationResponse {
        narrative_id: narrative_id.to_string(),
        iterations: gradual.iterations,
        converged: gradual.converged,
        gradual,
    };
    json_ok(&response)
}
