//! Cross-narrative learning layer (Phase 3).
//!
//! Provides corpus management, structural pattern mining,
//! arc classification, and missing event prediction across
//! multiple narratives stored in the hypergraph.

pub mod arc;
pub mod backfill;
pub mod causal_helpers;
pub mod character_arcs;
pub mod commitments;
pub mod compression;
pub mod continuity;
pub mod corpus;
pub mod cost_ledger;
pub mod debug;
pub mod debug_fixes;
pub mod dedup;
pub mod diff;
pub mod dramatic_irony;
pub mod editing;
pub mod essentiality;
pub mod expansion;
pub mod fabula_sjuzet;
pub mod generation;
pub mod name_backfill;
pub mod pattern;
pub mod plan;
pub mod prediction;
pub mod project;
pub mod propp;
pub mod registry;
pub mod revision;
pub mod rhythm_transfer;
pub mod scene_sequel;
pub mod similarity;
pub mod skeleton;
pub mod subgraph;
pub mod subplots;
pub mod taxonomy;
pub mod templates;
pub mod three_process;
pub mod types;
pub mod workshop;
pub mod workspace;
pub mod writer_common;

pub use arc::{ActorArcEngine, ArcEngine};
pub use character_arcs::CharacterArcEngine;
pub use commitments::CommitmentEngine;
pub use dramatic_irony::DramaticIronyEngine;
pub use fabula_sjuzet::FabulaSjuzetEngine;
pub use pattern::PatternMiningEngine;
pub use prediction::MissingEventEngine;
pub use scene_sequel::SceneSequelEngine;
pub use subplots::SubplotEngine;
