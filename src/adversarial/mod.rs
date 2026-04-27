//! Adversarial Narrative Wargaming module.
//!
//! Extends TENSA from detection/analysis to forward simulation and
//! adversarial wargaming. Feature-gated behind `--features adversarial`.
//!
//! ## Architecture
//!
//! - **types**: Shared types (Team, Platform, OperationalConstraints, actions)
//! - **suqr**: Subjective Utility Quantal Response bounded rationality model
//!
//! ## Sprint Roadmap
//!
//! - **D9**: SUQR + cognitive hierarchy + IRL policy generator
//! - **D10**: Simulation state engine + wargame loop
//! - **D11**: DISARM TTP taxonomy + STIX 2.1 interoperability
//! - **D12**: Reward-aware counter-narrative generation
//! - **D13**: Historical campaign retrodiction + benchmarks
//! - **D14**: Real-time ingestion + intervention optimizer
//! - **D15**: GNN surrogate + platform β calibration
//! - **D16**: Studio wargaming UI
//! - **D17**: Ethics governance layer

pub mod calibration;
pub mod cognitive_hierarchy;
pub mod counter_gen;
pub mod counter_score;
pub mod disarm;
pub mod governance;
pub mod opinion_shift_reward;
pub mod optimizer;
pub mod policy_gen;
pub mod retrodiction;
pub mod reward_model;
pub mod session;
pub mod sim_state;
pub mod stix_export;
pub mod suqr;
pub mod types;
pub mod wargame;

#[cfg(test)]
mod wargame_substrate_tests;

pub use suqr::*;
pub use types::*;
