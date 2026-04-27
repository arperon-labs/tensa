//! MCP (Model Context Protocol) server for TENSA.
//!
//! Exposes TENSA's temporal hypergraph engine as MCP tools,
//! allowing Claude and other MCP clients to query narratives,
//! create entities, run inference, and ingest text.
//!
//! Supports two modes:
//! - **Embedded**: Direct library access (default)
//! - **HTTP**: Client to the running REST API

pub mod backend;
pub mod embedded;
pub mod embedded_ext;
pub mod embedded_fuzzy;
pub mod embedded_fuzzy_ext;
#[cfg(test)]
mod embedded_fuzzy_tests;
pub mod embedded_graded;
#[cfg(test)]
mod embedded_graded_tests;
pub mod embedded_synth;
pub mod embedded_w15;
pub mod error;
pub mod http;
pub mod server;
pub mod types;
