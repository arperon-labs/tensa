//! Claims & fact-check pipeline (Disinfo Sprint D4).
//!
//! Detects verifiable claims in narrative text, matches them against known
//! fact-checks, tracks claim mutations over time, and integrates with the
//! Dung argumentation framework for attack/defense modeling.

pub mod detection;
pub mod fact_check;
pub mod matching;
pub mod mutation;
pub mod sync;
pub mod types;

pub use detection::detect_claims;
pub use fact_check::ingest_fact_check;
pub use matching::match_claim;
pub use mutation::track_mutations;
pub use sync::{execute_sync, FactCheckSource, SyncResult};
pub use types::*;
