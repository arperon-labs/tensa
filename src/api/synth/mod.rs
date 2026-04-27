//! REST API module for the synth (surrogate-generation) subsystem.
//!
//! Phase 6 shipped a single 466-line `synth_routes.rs`. Phase 7 added three
//! new significance handlers + supporting body types, pushing the file past
//! the 500-line cap. This module splits the route surface into focused
//! sub-modules per `docs/synth_null_model.md` §10 — paths and handler
//! signatures are unchanged so the frozen API contract stays intact:
//!
//! ```text
//! POST   /synth/calibrate/{narrative_id}                      → calibration::calibrate
//! GET    /synth/params/{narrative_id}/{model}                 → calibration::get_params
//! PUT    /synth/params/{narrative_id}/{model}                 → calibration::put_params
//! DELETE /synth/params/{narrative_id}/{model}                 → calibration::delete_params_route
//! POST   /synth/generate                                      → generation::generate
//! POST   /synth/generate-hybrid                                → generation::generate_hybrid (Phase 9)
//! GET    /synth/runs/{narrative_id}                           → generation::list_runs
//! GET    /synth/runs/{narrative_id}/{run_id}                  → generation::get_run
//! GET    /synth/seed/{run_id}                                 → generation::get_seed
//! GET    /synth/fidelity/{narrative_id}/{run_id}              → fidelity::get_fidelity
//! GET    /synth/fidelity-thresholds/{narrative_id}            → fidelity::get_fidelity_thresholds
//! PUT    /synth/fidelity-thresholds/{narrative_id}            → fidelity::put_fidelity_thresholds
//! GET    /synth/models                                        → models::list_models
//! POST   /synth/significance                                  → significance::post_significance
//! GET    /synth/significance/{narrative_id}/{metric}/{run_id} → significance::get_significance_result
//! GET    /synth/significance/{narrative_id}/{metric}          → significance::list_significance_results
//! POST   /synth/dual-significance                                  → dual_significance::post_dual_significance (Phase 13c)
//! GET    /synth/dual-significance/{narrative_id}/{metric}/{run_id} → dual_significance::get_dual_significance_result
//! GET    /synth/dual-significance/{narrative_id}/{metric}          → dual_significance::list_dual_significance_results
//! ```
//!
//! ## Conventions (unchanged from Phase 6)
//!
//! * Default `model` is `"eath"` — single source of truth in [`DEFAULT_MODEL`].
//! * Job submissions go through [`submit_synth_job`] for a consistent
//!   `{ "job_id": id, "status": "Pending" }` envelope (201 Created).
//! * Handlers route through `crate::api::routes::error_response` and
//!   `crate::api::routes::json_ok` to match the rest of the API surface.

pub mod bistability_significance;
pub mod calibration;
pub mod contagion;
pub mod dual_significance;
pub mod fidelity;
pub mod generation;
pub mod models;
pub mod opinion_significance;
pub mod significance;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use uuid::Uuid;

use crate::api::routes::error_response;
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::inference::types::InferenceJob;
use crate::types::{InferenceJobType, JobPriority, JobStatus};

/// Default surrogate-model name used when callers omit the `model` field on
/// `POST /synth/calibrate` and `POST /synth/generate`. EATH is the only model
/// shipping in Phase 4, but every endpoint is still keyed by name so future
/// surrogate families plug in without route changes.
pub(super) const DEFAULT_MODEL: &str = "eath";

/// Default page size for `GET /synth/runs/{narrative_id}` when `?limit=` is
/// omitted. Frozen contract value — bumped only via spec change.
pub(super) const DEFAULT_RUN_PAGE_LIMIT: usize = 50;

/// Submit a synth job to the worker pool. Mirrors `routes::submit_job` so
/// the response envelope (`{"job_id": id, "status": "Pending"}` + 201) is
/// consistent across the entire API surface.
pub(super) fn submit_synth_job(
    state: &AppState,
    job_type: InferenceJobType,
    parameters: serde_json::Value,
) -> axum::response::Response {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response();
        }
    };

    let job = InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        // Synth jobs are scoped by their narrative_id payload, not target_id;
        // engines ignore the field. Mint a fresh UUID so dedup-by-target
        // doesn't collapse two distinct narrative_id calibrations.
        target_id: Uuid::now_v7(),
        parameters,
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };

    match job_queue.submit(job) {
        Ok(id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"job_id": id, "status": "Pending"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../synth_tests.rs"]
mod synth_tests;
