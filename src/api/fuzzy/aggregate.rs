//! `POST /fuzzy/aggregate` — one-shot aggregation over a caller-supplied
//! vector. Synchronous endpoint (no job queue) because the exact Choquet
//! path is `O(n · 2^n)` with `n ≤ 10` and the Mean/Median/OWA paths are
//! linear in `|xs|`.
//!
//! Caps:
//! * `|xs| ≤ 1000` — above this the caller should route through the
//!   async job queue.
//! * Choquet exact path: `n ≤ 10` (see [`crate::fuzzy::
//!   aggregation_choquet::EXACT_N_CAP`]).
//! * Choquet Monte-Carlo path: `n ≤ 16` (FuzzyMeasure storage cap).

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::aggregation::{
    Aggregator, FuzzyMeasure, MeanAggregator, MedianAggregator, OwaAggregator,
    TConormReduceAggregator, TNormReduceAggregator,
};
use crate::fuzzy::aggregation_choquet::{choquet, EXACT_N_CAP};
use crate::fuzzy::registry::{AggregatorRegistry, TNormRegistry};

use super::measure::load_measure;

/// Maximum `|xs|` the synchronous endpoint will service. Above this we
/// expect callers to chunk the input or route through the worker pool.
pub const MAX_AGGREGATE_LEN: usize = 1000;

/// Body for `POST /fuzzy/aggregate`.
#[derive(Debug, Deserialize)]
pub struct AggregateBody {
    pub xs: Vec<f64>,
    /// One of `mean` / `median` / `owa` / `choquet` / `tnorm_reduce` /
    /// `tconorm_reduce`. Resolved via [`AggregatorRegistry`].
    pub aggregator: String,
    /// Required by `tnorm_reduce` / `tconorm_reduce`; optional default
    /// for other aggregators. Resolved via [`TNormRegistry`].
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Named reference to a persisted measure (CRUD under
    /// `/fuzzy/measures`). Required by `choquet`.
    #[serde(default)]
    pub measure: Option<String>,
    /// Explicit OWA weights. Required by `owa`. Must have length equal
    /// to `|xs|` and sum to `1.0 ± 1e-9`.
    #[serde(default)]
    pub owa_weights: Option<Vec<f64>>,
    /// Optional seed for Choquet Monte-Carlo path.
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct AggregateResult {
    pub value: f64,
    pub aggregator_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tnorm_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measure_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_err: Option<f64>,
}

/// POST /fuzzy/aggregate — compute a single aggregated value.
pub async fn aggregate(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AggregateBody>,
) -> impl IntoResponse {
    // Shape checks.
    if body.xs.is_empty() {
        return error_response(TensaError::InvalidInput(
            "aggregate: xs must be non-empty".into(),
        ))
        .into_response();
    }
    if body.xs.len() > MAX_AGGREGATE_LEN {
        return error_response(TensaError::InvalidInput(format!(
            "aggregate: |xs|={} exceeds the synchronous cap of {}; chunk the input or use the async worker pool",
            body.xs.len(),
            MAX_AGGREGATE_LEN
        )))
        .into_response();
    }
    for (i, v) in body.xs.iter().enumerate() {
        if !v.is_finite() {
            return error_response(TensaError::InvalidInput(format!(
                "aggregate: xs[{i}] is not finite ({v})"
            )))
            .into_response();
        }
    }

    let agg_name = body.aggregator.trim().to_lowercase();
    let registry = AggregatorRegistry::default();
    if registry.get(&agg_name).is_err() {
        return error_response(TensaError::InvalidInput(format!(
            "aggregate: unknown aggregator '{agg_name}'"
        )))
        .into_response();
    }

    // Dispatch per kind — we build the concrete `AggregatorKind` here so
    // parameter validation sits next to the endpoint's contract.
    let (value, tnorm_name, measure_name, std_err) = match agg_name.as_str() {
        "mean" => {
            let r = match MeanAggregator.aggregate(&body.xs) {
                Ok(v) => v,
                Err(e) => return error_response(e).into_response(),
            };
            (r, None, None, None)
        }
        "median" => {
            let r = match MedianAggregator.aggregate(&body.xs) {
                Ok(v) => v,
                Err(e) => return error_response(e).into_response(),
            };
            (r, None, None, None)
        }
        "owa" => {
            let weights = match body.owa_weights.clone() {
                Some(w) => w,
                None => {
                    return error_response(TensaError::InvalidInput(
                        "aggregate: owa requires 'owa_weights' of length |xs|".into(),
                    ))
                    .into_response()
                }
            };
            let agg = OwaAggregator { weights };
            let r = match agg.aggregate(&body.xs) {
                Ok(v) => v,
                Err(e) => return error_response(e).into_response(),
            };
            (r, None, None, None)
        }
        "choquet" => {
            let measure_name = match body.measure.clone() {
                Some(m) => m,
                None => {
                    return error_response(TensaError::InvalidInput(
                        "aggregate: choquet requires a 'measure' reference (POST /fuzzy/measures first)"
                            .into(),
                    ))
                    .into_response()
                }
            };
            let measure: FuzzyMeasure =
                match load_measure(state.hypergraph.store(), &measure_name) {
                    Ok(Some(m)) => m,
                    Ok(None) => {
                        return error_response(TensaError::InvalidInput(format!(
                            "aggregate: no measure named '{measure_name}' — create one via POST /fuzzy/measures"
                        )))
                        .into_response()
                    }
                    Err(e) => return error_response(e).into_response(),
                };
            let n = body.xs.len();
            if n != measure.n as usize {
                return error_response(TensaError::InvalidInput(format!(
                    "aggregate: choquet requires |xs|={n} to equal measure.n={}",
                    measure.n
                )))
                .into_response();
            }
            if n > 16 {
                return error_response(TensaError::InvalidInput(format!(
                    "aggregate: choquet |xs|={n} exceeds the cap of 16; exact path caps at n ≤ {}",
                    EXACT_N_CAP
                )))
                .into_response();
            }
            let seed = body.seed.unwrap_or(0);
            let r = match choquet(&body.xs, &measure, seed) {
                Ok(v) => v,
                Err(e) => return error_response(e).into_response(),
            };
            (r.value, None, Some(measure_name), r.std_err)
        }
        "tnorm_reduce" => {
            let (kind, name) = match resolve_tnorm(&body.tnorm) {
                Ok(v) => v,
                Err(e) => return error_response(e).into_response(),
            };
            let r = TNormReduceAggregator { kind }
                .aggregate(&body.xs)
                .unwrap_or(0.0);
            (r, Some(name), None, None)
        }
        "tconorm_reduce" => {
            let (kind, name) = match resolve_tnorm(&body.tnorm) {
                Ok(v) => v,
                Err(e) => return error_response(e).into_response(),
            };
            let r = TConormReduceAggregator { kind }
                .aggregate(&body.xs)
                .unwrap_or(0.0);
            (r, Some(name), None, None)
        }
        other => {
            return error_response(TensaError::InvalidInput(format!(
                "aggregate: aggregator '{other}' is registered but not reachable — please file a bug"
            )))
            .into_response()
        }
    };

    // Belt + braces: every canonical aggregator must produce a finite value.
    debug_assert!(value.is_finite(), "aggregator produced non-finite value");

    let resp = AggregateResult {
        value,
        aggregator_name: agg_name,
        tnorm_name,
        measure_name,
        std_err,
    };
    json_ok(&resp)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn resolve_tnorm(
    name: &Option<String>,
) -> Result<(crate::fuzzy::tnorm::TNormKind, String), TensaError> {
    let n = name.as_deref().unwrap_or("godel").trim().to_lowercase();
    let kind = TNormRegistry::default().get(&n)?;
    Ok((kind, n))
}
