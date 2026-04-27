//! `/synth/generate`, `/synth/runs/*`, `/synth/seed/{run_id}` handlers.
//!
//! Submits generation jobs and exposes the run / reproducibility blob
//! readback surface. Unchanged from Phase 6 except for module relocation —
//! frozen routing contract.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::synth::{
    key_synth_run, list_runs_newest_first, load_reproducibility_blob, SurrogateRunSummary,
};
use crate::types::InferenceJobType;

use super::{submit_synth_job, DEFAULT_MODEL, DEFAULT_RUN_PAGE_LIMIT};

// ── Generate ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct GenerateBody {
    /// Source narrative whose calibrated params drive generation. `None` is
    /// only valid when `params` is set (inline params + no source).
    #[serde(default)]
    pub source_narrative_id: Option<String>,
    /// Where to write the synthetic narrative.
    pub output_narrative_id: String,
    /// Surrogate model name. Defaults to `"eath"`.
    #[serde(default)]
    pub model: Option<String>,
    /// Inline param override — wins over loaded params when present.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Deterministic-replay seed. Engine generates one when absent.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Number of generation steps. Engine defaults to 100 when absent.
    #[serde(default)]
    pub num_steps: Option<usize>,
    /// Synthetic-narrative label prefix. Engine defaults to `"synth"` when absent.
    #[serde(default)]
    pub label_prefix: Option<String>,
}

/// POST /synth/generate
///
/// Submits a `SurrogateGeneration` job to the worker pool. Returns 400 when
/// neither `source_narrative_id` nor inline `params` is supplied — the
/// engine would error at runtime, but a synchronous 400 is friendlier to
/// API clients than a queued-then-failed job.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GenerateBody>,
) -> impl IntoResponse {
    if body.output_narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput(
            "output_narrative_id is empty".into(),
        ))
        .into_response();
    }
    if body.source_narrative_id.is_none() && body.params.is_none() {
        return error_response(TensaError::InvalidInput(
            "missing source_narrative_id (and no inline params provided)".into(),
        ))
        .into_response();
    }

    let model = body.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let job_type = InferenceJobType::SurrogateGeneration {
        source_narrative_id: body.source_narrative_id,
        output_narrative_id: body.output_narrative_id,
        model,
        params_override: body.params,
        seed_override: body.seed,
    };

    // num_steps + label_prefix piggyback on `job.parameters` because the
    // job-type variant doesn't carry them — engine reads them from the JSON
    // parameters slot. See `synth::engines::SurrogateGenerationEngine`.
    let mut params = serde_json::Map::new();
    if let Some(n) = body.num_steps {
        params.insert("num_steps".into(), serde_json::json!(n));
    }
    if let Some(prefix) = body.label_prefix {
        params.insert("label_prefix".into(), serde_json::json!(prefix));
    }

    submit_synth_job(&state, job_type, serde_json::Value::Object(params))
}

// ── Runs ─────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
pub struct RunsListQuery {
    pub limit: Option<usize>,
}

/// GET /synth/runs/{narrative_id}
///
/// Newest-first listing — backed by `list_runs_newest_first`, which reverses
/// the chronological prefix scan in O(n) on the page window (no client-side
/// sort needed).
pub async fn list_runs(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Query(q): Query<RunsListQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(DEFAULT_RUN_PAGE_LIMIT).clamp(1, 1000);
    match list_runs_newest_first(state.hypergraph.store(), &narrative_id, limit) {
        Ok(runs) => json_ok(&runs),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /synth/runs/{narrative_id}/{run_id}
///
/// Direct KV fetch via `key_synth_run` — does NOT scan the lineage index.
/// Returns 404 when no run exists at the (narrative, run_id) pair.
pub async fn get_run(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, run_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    let key = key_synth_run(&narrative_id, &run_id);
    match state.hypergraph.store().get(&key) {
        Ok(Some(bytes)) => match serde_json::from_slice::<SurrogateRunSummary>(&bytes) {
            Ok(summary) => json_ok(&summary),
            Err(e) => error_response(TensaError::Serialization(e.to_string())).into_response(),
        },
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no run for ('{narrative_id}', '{run_id}')")
            })),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── Reproducibility ──────────────────────────────────────────────────────────

/// GET /synth/seed/{run_id}
///
/// Returns the `ReproducibilityBlob` for a run — the input bundle (params,
/// seed, source state hash, tensa version, git sha) needed to replay the
/// exact same generation.
pub async fn get_seed(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<Uuid>,
) -> impl IntoResponse {
    match load_reproducibility_blob(state.hypergraph.store(), &run_id) {
        Ok(Some(blob)) => json_ok(&blob),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("no reproducibility blob for '{run_id}'")})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ── Hybrid generation (EATH Phase 9) ─────────────────────────────────────────

/// Request body for `POST /synth/generate-hybrid`. Components are validated
/// engine-side (weights sum to 1.0 within `1e-6`); the route only checks the
/// minimum-viable shape so a synchronous 400 catches obvious mistakes.
#[derive(Deserialize)]
pub struct GenerateHybridBody {
    /// Mixture components — each carries `(narrative_id, model, weight)`.
    pub components: Vec<crate::synth::hybrid::HybridComponent>,
    /// Where to write the synthetic output. Required (no default — hybrid
    /// has no canonical "source" to derive a name from).
    pub output_narrative_id: String,
    /// Deterministic-replay seed. Engine generates one when absent.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Number of generation steps. Engine defaults to 100 when absent.
    #[serde(default)]
    pub num_steps: Option<usize>,
}

/// POST /synth/generate-hybrid
///
/// Submits a `SurrogateHybridGeneration` job to the worker pool. Quick
/// shape-checks here (non-empty components, non-empty output id, weights
/// sum-to-1.0) so the user gets a synchronous 400 rather than waiting for
/// a queued-then-failed job. Engine performs the same checks (defence in
/// depth + invariant for direct engine callers).
pub async fn generate_hybrid(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GenerateHybridBody>,
) -> impl IntoResponse {
    if body.output_narrative_id.is_empty() {
        return error_response(TensaError::InvalidInput(
            "output_narrative_id is empty".into(),
        ))
        .into_response();
    }
    if body.components.is_empty() {
        return error_response(TensaError::InvalidInput(
            "hybrid components list is empty".into(),
        ))
        .into_response();
    }
    let sum: f32 = body.components.iter().map(|c| c.weight).sum();
    if (sum - 1.0).abs() > crate::synth::hybrid::HYBRID_WEIGHT_TOLERANCE {
        return error_response(TensaError::InvalidInput(format!(
            "hybrid weights must sum to 1.0 (got {sum})"
        )))
        .into_response();
    }

    let components_json = match serde_json::to_value(&body.components) {
        Ok(v) => v,
        Err(e) => {
            return error_response(TensaError::InvalidInput(format!(
                "components serialize: {e}"
            )))
            .into_response();
        }
    };

    let job_type = InferenceJobType::SurrogateHybridGeneration {
        components: components_json,
        output_narrative_id: body.output_narrative_id,
        seed_override: body.seed,
        num_steps: body.num_steps,
    };

    submit_synth_job(&state, job_type, serde_json::Value::Object(serde_json::Map::new()))
}
