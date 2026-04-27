//! SINDy hypergraph reconstruction from observed entity dynamics.
//!
//! Implements the Taylor-expanded Hypergraph Identification via SINDy
//! (THIS) method: given a per-entity time-series observation function,
//! infer the latent hyperedges that best explain the observed joint
//! dynamics under a sparse dynamical-systems assumption.
//!
//! ## Pipeline
//!
//! ```text
//! load → observe → differentiate → build_library (Pearson pre-filter)
//!     → solve_lasso (coordinate descent, per-entity) → symmetrize+threshold
//!     → bootstrap (K resamples for confidence) → ReconstructionResult
//! ```
//!
//! ## Method references
//!
//! - **Primary:** Delabays, De Pasquale, Dörfler, Zhang —
//!   *Hypergraph reconstruction from dynamics*, Nat. Commun. **16**, 2691
//!   (2025), arXiv:2402.00078.
//! - **Method parent (SINDy):** Brunton, Proctor, Kutz —
//!   *Discovering governing equations from data by sparse identification of
//!   nonlinear dynamical systems*, PNAS **113**, 3932 (2016),
//!   arXiv:1509.03580.
//! - **Pairwise baseline (ARNI):** Casadiego, Nitzan, Hallerberg, Timme —
//!   *Model-free inference of direct network interactions from nonlinear
//!   collective dynamics*, Nat. Commun. **8**, 2192 (2017).
//!
//! ## Validation
//!
//! Phase 15b validates against EATH-generated synthetic narratives with
//! planted hyperedges (Mancastroppa, Cencetti, Barrat — arXiv:2507.01124).
//! Test 1 (`reconstruct_tests::test_reconstruction_recovers_planted_eath_structure_auroc_gt_0_85`)
//! is load-bearing: AUROC must clear 0.85 against the planted ground truth.
//!
//! ## MVP scope
//!
//! - `ObservationSource::ParticipationRate` is fully implemented.
//! - `SentimentMean` and `BeliefMass` are declared but return
//!   `InferenceError("PrerequisiteMissing: ...")` until the underlying data
//!   sources are populated.
//! - `Engagement` (multi-dimensional) is declared but stubbed per Phase 0
//!   exhaustive-match convention.
//! - Lambda auto-selection uses the `λ_max × 0.1` heuristic. Cross-validation
//!   is opt-in via `lambda_cv: bool` (returns `InferenceError` until Phase
//!   15c implements it).
//! - Output is undirected. Streaming/online/temporal-tracking forms are
//!   deferred to follow-up phases.

pub mod bootstrap;
pub mod derivative;
pub mod library;
pub mod materialize;
pub mod observe;
pub mod reconstruct;
pub mod types;

mod lasso;
mod symmetrize;

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

pub use self::materialize::{
    materialize_reconstruction, MaterializationReport, ReconstructedSituationRef,
    DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD,
};
pub use self::reconstruct::reconstruct;
pub use self::types::{
    DerivativeEstimator, InferredHyperedge, LibraryTerm, MatrixStats, ObservationSource,
    ReconstructionParams, ReconstructionResult,
};

/// Inference engine for SINDy hypergraph reconstruction.
///
/// Consumes a [`InferenceJobType::HypergraphReconstruction`] job and produces
/// an `InferenceResult` whose `result` field carries a JSON-serialized
/// [`ReconstructionResult`] under `kind = "reconstruction_done"`.
pub struct ReconstructionEngine;

impl InferenceEngine for ReconstructionEngine {
    fn job_type(&self) -> InferenceJobType {
        // Sentinel — the worker pool keys engines by discriminant, so the
        // payload values here are never compared. See `worker.rs` module doc.
        InferenceJobType::HypergraphReconstruction {
            narrative_id: String::new(),
            params: serde_json::Value::Null,
        }
    }

    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
        // Delegate to the central cost estimator so the heuristic stays in
        // one place.
        crate::inference::cost::estimate_cost(job, hypergraph)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let (narrative_id, params) = match &job.job_type {
            InferenceJobType::HypergraphReconstruction {
                narrative_id,
                params,
            } => (narrative_id.clone(), params.clone()),
            other => {
                return Err(TensaError::InferenceError(format!(
                    "ReconstructionEngine: unexpected job type {other:?}"
                )));
            }
        };

        // Allow both job.job_type.params and job.parameters to carry the
        // params blob — the descriptor dispatch path stuffs it into
        // `job.parameters` rather than rebuilding the variant.
        let effective_params: ReconstructionParams = if params.is_null() {
            if job.parameters.is_null() || job.parameters == serde_json::Value::Object(Default::default()) {
                ReconstructionParams::default()
            } else {
                serde_json::from_value(job.parameters.clone()).map_err(|e| {
                    TensaError::InferenceError(format!(
                        "ReconstructionEngine: malformed params: {e}"
                    ))
                })?
            }
        } else {
            serde_json::from_value(params).map_err(|e| {
                TensaError::InferenceError(format!(
                    "ReconstructionEngine: malformed params: {e}"
                ))
            })?
        };

        let effective_narrative = if narrative_id.is_empty() {
            job.parameters
                .get("narrative_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            narrative_id
        };

        if effective_narrative.is_empty() {
            return Err(TensaError::InferenceError(
                "ReconstructionEngine: narrative_id is required".into(),
            ));
        }

        let result = reconstruct(hypergraph, &effective_narrative, &effective_params)?;
        let confidence = result.goodness_of_fit.clamp(0.0, 1.0);
        let n_edges = result.inferred_edges.len();
        let payload = serde_json::json!({
            "kind": "reconstruction_done",
            "result": result,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: payload,
            confidence,
            explanation: Some(format!(
                "SINDy reconstruction recovered {n_edges} hyperedge(s) from narrative '{effective_narrative}'"
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod engine_tests;

#[cfg(test)]
#[path = "reconstruct_tests.rs"]
mod reconstruct_tests;
