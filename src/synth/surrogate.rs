//! The `SurrogateModel` trait — single behavioral contract every surrogate
//! plug-in implements. Phase 0 ships only the trait + an EATH stub; Phases 1–3
//! flesh out the EATH impl, and future families (HAD, hyperedge config, etc.)
//! plug in by implementing the same trait and registering with
//! [`crate::synth::SurrogateRegistry`].
//!
//! Trait method bodies must be pure with respect to `&self` so multiple
//! threads can call the same registered model concurrently — engine code in
//! `inference/worker.rs` clones `Arc<dyn SurrogateModel>` across worker tasks.

use crate::error::Result;
use crate::hypergraph::Hypergraph;

use super::types::{SurrogateParams, SurrogateRunSummary};

/// A pluggable surrogate / null-model generator for temporal hypergraphs.
pub trait SurrogateModel: Send + Sync {
    /// Stable identifier used by the registry, the TensaQL grammar, and KV
    /// keys. **Must** match `&'static str` so it can be embedded in keys
    /// without allocation.
    fn name(&self) -> &'static str;

    /// Algorithm version. Bump when the calibration or generation logic
    /// changes in a way that would invalidate a previously-stored
    /// `SurrogateParams`. Used by Phase 11 "Reproduce this run" to warn when
    /// model code has drifted since a run was recorded.
    fn version(&self) -> &'static str;

    /// Fit model parameters against a real narrative. Returns the
    /// model-specific JSON blob that goes into `SurrogateParams.params_json`.
    /// Implementations should be deterministic given identical input — any
    /// stochastic seeds belong in the returned params, not in side-effects.
    fn calibrate(&self, hypergraph: &Hypergraph, narrative_id: &str)
        -> Result<serde_json::Value>;

    /// Generate a synthetic narrative into `target` using `params`. Writes
    /// entities + situations + participations under `output_narrative_id`.
    /// Returns the run summary for the caller to persist via
    /// [`super::key_synth_run`].
    fn generate(
        &self,
        params: &SurrogateParams,
        target: &Hypergraph,
        output_narrative_id: &str,
    ) -> Result<SurrogateRunSummary>;

    /// Names of the fidelity metrics this model emits. Phase 2.5 uses this
    /// to know which slots in `FidelityReport` to populate. Default empty
    /// — implementations that produce metrics override.
    fn fidelity_metrics(&self) -> Vec<&'static str> {
        Vec::new()
    }
}
