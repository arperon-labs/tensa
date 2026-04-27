//! `POST /fuzzy/measures/learn` — Phase 2 ranking-supervised PGD fit.
//!
//! Accepts a `(input_vec, rank)` dataset and a target name; runs the PGD
//! solver in [`crate::fuzzy::aggregation_learn::learn_choquet_measure`];
//! persists the resulting `StoredMeasure` with `MeasureProvenance::Learned`
//! at both the versionless `fz/tn/measures/{name}` key (latest pointer)
//! and a versioned `fz/tn/measures/{name}/v{version}` key (history slice).
//!
//! Re-training under an existing name increments the version. Old
//! versions are preserved at their versioned keys so paper-figure
//! callers can replay specific fits.
//!
//! Cites: [grabisch1996choquet], [bustince2016choquet].

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::error_response;
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::aggregation_learn::{
    learn_choquet_measure, LearnedChoquetMeasureOutput, LearnedMeasureProvenance,
    MeasureProvenance,
};

use super::measure::{measure_key, versioned_measure_key, StoredMeasure};

/// Body for `POST /fuzzy/measures/learn`.
#[derive(Debug, Deserialize)]
pub struct LearnMeasureRequest {
    /// Persistence name. Must be `[A-Za-z0-9._-]` — `/`, whitespace, and
    /// newlines are rejected to keep KV key encoding clean.
    pub name: String,
    /// Universe size. Capped at 6 by the in-tree PGD path; larger
    /// universes return InvalidInput with a pointer to k-additive.
    pub n: u8,
    /// `(input_vec, rank)` pairs. Lower rank = more strongly coordinated.
    pub dataset: Vec<(Vec<f64>, u32)>,
    /// Caller-supplied dataset identifier — drives the deterministic
    /// 50 / 50 train / test split seed.
    pub dataset_id: String,
}

/// Reply shape for `POST /fuzzy/measures/learn`.
#[derive(Debug, Serialize)]
pub struct LearnedMeasureSummary {
    pub name: String,
    pub version: u32,
    pub n: u8,
    pub provenance: LearnedMeasureProvenance,
    pub train_auc: f64,
    pub test_auc: f64,
}

// Phase 3: the two key builders live in `super::measure` as the single
// source of truth — see [`super::measure::measure_key`] and
// [`super::measure::versioned_measure_key`]. We re-import them above so
// the persistence path here stays identical to the read path in CRUD.

/// POST /fuzzy/measures/learn — fit + persist.
pub async fn learn_measure_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LearnMeasureRequest>,
) -> impl IntoResponse {
    let name = req.name.trim().to_string();
    if name.is_empty() {
        return error_response(TensaError::InvalidInput(
            "measure name must be non-empty".into(),
        ))
        .into_response();
    }
    if name.contains('/') || name.contains('\n') || name.contains(' ') {
        return error_response(TensaError::InvalidInput(format!(
            "measure name '{name}' contains invalid characters ('/', whitespace, or newlines)"
        )))
        .into_response();
    }

    let LearnedChoquetMeasureOutput {
        mut measure,
        provenance,
        train_auc,
        test_auc,
    } = match learn_choquet_measure(req.n, &req.dataset, &req.dataset_id) {
        Ok(o) => o,
        Err(e) => return error_response(e).into_response(),
    };

    // Look up existing record to determine the next version.
    let store = state.hypergraph.store();
    let prev_version = match store.get(&measure_key(&name)) {
        Ok(Some(bytes)) => match serde_json::from_slice::<StoredMeasure>(&bytes) {
            Ok(prev) => Some(prev.version),
            Err(e) => {
                return error_response(TensaError::Serialization(e.to_string())).into_response();
            }
        },
        Ok(None) => None,
        Err(e) => return error_response(e).into_response(),
    };
    let next_version = prev_version.map(|v| v.saturating_add(1)).unwrap_or(1);

    // Stamp the measure with its name + version so downstream consumers
    // see provenance even when only the inline FuzzyMeasure escapes.
    measure.measure_id = Some(name.clone());
    measure.measure_version = Some(next_version);

    let stored = StoredMeasure {
        name: name.clone(),
        measure,
        version: next_version,
        provenance: MeasureProvenance::Learned(provenance.clone()),
    };
    let bytes = match serde_json::to_vec(&stored) {
        Ok(b) => b,
        Err(e) => return error_response(TensaError::Serialization(e.to_string())).into_response(),
    };

    // Write versionless (latest pointer) AND versioned (history) keys.
    if let Err(e) = store.put(&measure_key(&name), &bytes) {
        return error_response(e).into_response();
    }
    if let Err(e) = store.put(&versioned_measure_key(&name, next_version), &bytes) {
        return error_response(e).into_response();
    }

    let summary = LearnedMeasureSummary {
        name,
        version: next_version,
        n: req.n,
        provenance,
        train_auc,
        test_auc,
    };
    (
        StatusCode::CREATED,
        Json(serde_json::to_value(&summary).unwrap_or_default()),
    )
        .into_response()
}
