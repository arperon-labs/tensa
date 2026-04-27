//! Narrative generation engine (Sprint D9.6–D9.7).
//!
//! Three-stage pipeline: **Plan → Materialize → Generate.**
//!
//! 1. `planner` — generate a formal NarrativePlan from a premise + target fingerprint
//! 2. `materializer` — write the plan into the TENSA hypergraph as real entities,
//!    situations, Allen relations, causal edges, commitments, and knowledge states
//! 3. `engine` — generate prose by querying the materialized hypergraph for each
//!    situation's specification, then re-ingest generated text back into the graph
//! 4. `prompt_builder` — construct LLM prompts from hypergraph queries
//!
//! Feature-gated behind `generation`.

pub mod engine;
pub mod job_engine;
pub mod materializer;
pub mod planner;
pub mod prompt_builder;
pub mod types;
