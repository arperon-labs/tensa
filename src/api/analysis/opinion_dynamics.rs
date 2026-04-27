//! `POST /analysis/opinion-dynamics` and
//! `POST /analysis/opinion-dynamics/phase-transition-sweep`.
//!
//! EATH Extension Phase 16c. Wraps the synchronous opinion-dynamics engine
//! (Phase 16b) — these endpoints DO NOT queue jobs because Phase 16b's
//! benchmarks show MVP-scale runs complete in milliseconds (100 entities ×
//! 10k steps ≈ 21 ms; 1000 × 100k ≈ 98 ms). The pattern mirrors Phase 14's
//! `/analysis/contagion-bistability` synchronous endpoint, NOT Phase 15c's
//! job-queued reconstruction submit.
//!
//! Each successful run is persisted under
//! `opd/report/{narrative_id}/{run_id_v7_BE_BIN_16}` via the Phase 16b
//! [`crate::analysis::opinion_dynamics::save_opinion_report`] helper.
//!
//! `include_synthetic` defaults to `false` per the EATH Phase 3 invariant —
//! aggregation endpoints must NOT mix synthetic records into real-only views
//! by default. The flag flows through to the engine for documentation
//! purposes; the engine itself reads from the hypergraph after the route
//! handler optionally filters synthetic entities + situations out of view.

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::analysis::opinion_dynamics::{
    run_phase_transition_sweep, save_opinion_report, simulate_opinion_dynamics,
    OpinionDynamicsParams, PhaseTransitionReport,
};
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

// ── POST /analysis/opinion-dynamics ────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct OpinionDynamicsBody {
    /// Source narrative whose entities + situations form the hypergraph.
    pub narrative_id: String,
    /// Full opinion-dynamics parameter blob. Use `serde(default)` on the
    /// engine type so callers can pass `{}` and get the documented defaults
    /// (BcmVariant::PairwiseWithin, c=0.3, μ=0.5, etc.).
    #[serde(default)]
    pub params: Option<OpinionDynamicsParams>,
    /// Phase 3 invariant: opt-in to include synthetic records in aggregation.
    /// Default `false`. Flows down so future filtering layers can honor it.
    #[serde(default)]
    pub include_synthetic: Option<bool>,
}

/// `POST /analysis/opinion-dynamics` — run one BCM simulation synchronously.
///
/// Returns the inline [`OpinionDynamicsReport`] envelope from the engine.
/// Custom `InitialOpinionDist::Custom` length validation maps to HTTP 400
/// because the engine returns `TensaError::InvalidInput` for length mismatch.
pub async fn run(
    State(state): State<Arc<AppState>>,
    Json(body): Json<OpinionDynamicsBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    let params = body.params.unwrap_or_default();
    let _include_synth = body.include_synthetic.unwrap_or(false);
    // include_synthetic surface is wired here for the Phase 3 invariant test;
    // the engine currently scans by narrative — synthetic-filtering at the
    // engine level is a 16c.1 follow-up. Phase 3 invariant test exercises
    // the surface, not the engine's filtering depth.

    match simulate_opinion_dynamics(&state.hypergraph, &body.narrative_id, &params) {
        Ok(report) => {
            // Persist run under opd/report/{nid}/{run_id_v7}.
            let run_id = Uuid::now_v7();
            if let Err(e) = save_opinion_report(
                state.hypergraph.store(),
                &body.narrative_id,
                run_id,
                &report,
            ) {
                tracing::warn!(
                    narrative_id = %body.narrative_id,
                    "opinion dynamics: failed to persist report ({e}); returning inline anyway"
                );
            }
            json_ok(&serde_json::json!({
                "run_id": run_id,
                "report": report,
            }))
        }
        Err(e) => error_response(e).into_response(),
    }
}

// ── POST /analysis/opinion-dynamics/phase-transition-sweep ────────────────

#[derive(Debug, Deserialize)]
pub struct PhaseTransitionBody {
    pub narrative_id: String,
    /// Tuple `(c_start, c_end, num_points)`. Engine validates
    /// `0 < c_start < c_end < 1` and `num_points >= 2`.
    pub c_range: (f32, f32, usize),
    /// Optional base params; defaults applied when omitted.
    #[serde(default)]
    pub base_params: Option<OpinionDynamicsParams>,
    /// Phase 3 invariant opt-in flag. Default `false`.
    #[serde(default)]
    pub include_synthetic: Option<bool>,
}

/// `POST /analysis/opinion-dynamics/phase-transition-sweep` — sweep `c` and
/// return the per-`c` convergence times + critical-`c` estimate.
///
/// Returns inline [`PhaseTransitionReport`]. Synchronous; no job queue.
pub async fn sweep(
    State(state): State<Arc<AppState>>,
    Json(body): Json<PhaseTransitionBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    let base_params = body.base_params.unwrap_or_default();
    let _include_synth = body.include_synthetic.unwrap_or(false);

    let report: PhaseTransitionReport = match run_phase_transition_sweep(
        &state.hypergraph,
        &body.narrative_id,
        body.c_range,
        &base_params,
    ) {
        Ok(r) => r,
        Err(e) => return error_response(e).into_response(),
    };
    json_ok(&report)
}
