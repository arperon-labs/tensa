#[cfg(feature = "adversarial")]
pub mod adversarial;
pub mod analysis;
pub mod analysis_status;
pub mod api;
#[cfg(feature = "disinfo")]
pub mod claims;
#[cfg(feature = "disinfo")]
pub mod disinfo;
pub mod error;
pub mod export;
pub mod fuzzy;
#[cfg(feature = "generation")]
pub mod generation;
pub mod hypergraph;
pub mod inference;
pub mod ingestion;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod narrative;
pub mod query;
#[cfg(feature = "disinfo")]
pub mod scheduler;
pub mod source;
pub mod store;
#[cfg(feature = "studio-chat")]
pub mod studio_chat;
pub mod synth;
#[cfg(feature = "generation")]
pub mod style;
pub mod temporal;
pub mod text_util;
pub mod types;
pub mod writer;

pub use error::{Result, TensaError};
pub use hypergraph::Hypergraph;
pub use store::KVStore;
pub use types::*;

/// Build label: "v0.14.0+abc1234"
pub fn build_label() -> String {
    format!("v{}+{}", env!("CARGO_PKG_VERSION"), env!("TENSA_GIT_HASH"))
}
