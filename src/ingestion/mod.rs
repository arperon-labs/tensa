//! Text ingestion pipeline — Phase 1
//!
//! Transforms raw narrative text into structured hypergraph data through
//! LLM-powered extraction, confidence gating, and entity resolution.

/// Encode bytes as lowercase hex string.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub mod archive;
#[cfg(feature = "bedrock")]
pub mod bedrock;
pub mod chunker;
#[cfg(feature = "server")]
pub mod config;
pub mod deletion;
#[cfg(feature = "disinfo")]
pub mod discovery;
pub mod doc_status;
pub mod docparse;
pub mod embed;
pub mod extraction;
pub mod gate;
#[cfg(feature = "gemini")]
pub mod gemini;
pub mod geocode;
pub mod jobs;
pub mod llm;
pub mod llm_cache;
pub mod nlp_extract;
#[cfg(feature = "embedding")]
pub mod onnx_embedder;
pub mod pipeline;
pub mod prompt_tuning;
pub mod queue;
pub mod resolve;
pub mod span_resolve;
pub mod stix;
pub mod structured;
pub mod vector;
pub mod vector_store;
pub mod web;
