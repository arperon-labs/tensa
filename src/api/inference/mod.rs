//! REST API module for `/inference/*` routes.
//!
//! New top-level API category introduced in EATH Extension Phase 15c. Holds
//! the SINDy hypergraph reconstruction surface (job submission + result
//! polling + opt-in materialization). The module pattern mirrors
//! [`crate::api::synth`] so the existing [`crate::api::routes::error_response`]
//! / [`crate::api::routes::json_ok`] envelopes stay consistent across the
//! API.
//!
//! ```text
//! POST /inference/hypergraph-reconstruction                  → reconstruction::submit
//! GET  /inference/hypergraph-reconstruction/{job_id}         → reconstruction::get_result
//! POST /inference/hypergraph-reconstruction/{job_id}/materialize → reconstruction::materialize
//! ```

pub mod reconstruction;

#[cfg(test)]
#[path = "reconstruction_tests.rs"]
mod reconstruction_tests;
