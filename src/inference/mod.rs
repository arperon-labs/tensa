//! Inference engine layer (Phase 2).
//!
//! Provides async inference workers for causal discovery, game-theoretic
//! analysis, motivation inference, and counterfactual reasoning. All
//! inference results enter the hypergraph at `MaturityLevel::Candidate`
//! with confidence scores.

pub mod causal;
pub mod cost;
pub mod dispatch;
pub mod enrichment;
pub mod explain;
pub mod game;
pub mod hawkes;
pub mod hypergraph_reconstruction;
pub mod jobs;
pub mod mean_field_game;
pub mod motivation;
pub mod netinf;
pub mod simulation;
pub mod temporal_ilp;
pub mod trajectory;
pub mod types;
pub mod worker;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{InferenceJobType, InferenceResult};

use self::types::InferenceJob;

/// Trait for inference engines. Each engine handles one type of
/// inference job and produces results with confidence scores.
///
/// Implementations should be stateless — all data is read from
/// the hypergraph and job parameters. This enables easy mocking
/// in tests and swapping algorithm implementations.
pub trait InferenceEngine: Send + Sync {
    /// The type of job this engine handles.
    fn job_type(&self) -> InferenceJobType;

    /// Estimate execution cost in milliseconds.
    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64>;

    /// Execute the inference job. Returns an InferenceResult with
    /// the computed data serialized as JSON in the `result` field.
    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult>;
}
