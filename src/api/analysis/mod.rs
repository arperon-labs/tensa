//! REST API module for `/analysis/*` routes that need their own subdirectory.
//!
//! New top-level API category introduced in EATH Extension Phase 16c. Holds
//! the opinion-dynamics REST surface (synchronous simulation + phase-transition
//! sweep). The module pattern mirrors [`crate::api::synth`] and
//! [`crate::api::inference`] so the existing
//! [`crate::api::routes::error_response`] / [`crate::api::routes::json_ok`]
//! envelopes stay consistent across the API.
//!
//! ```text
//! POST /analysis/opinion-dynamics                          → opinion_dynamics::run
//! POST /analysis/opinion-dynamics/phase-transition-sweep   → opinion_dynamics::sweep
//! ```

pub mod fuzzy_allen;
pub mod opinion_dynamics;

#[cfg(test)]
#[path = "opinion_dynamics_tests.rs"]
mod opinion_dynamics_tests;
