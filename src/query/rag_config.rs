//! RAG (Retrieval-Augmented Generation) configuration.
//!
//! Controls retrieval mode and token budget allocation for ASK queries.

use serde::{Deserialize, Serialize};

use super::token_budget::TokenBudget;

/// Retrieval mode for RAG queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMode {
    /// Search only within the specified narrative's local context.
    Local,
    /// Search across all narratives using global summaries.
    Global,
    /// Combine local entity/situation retrieval with global chunk search.
    Hybrid,
    /// Blend all retrieval strategies with dynamic weighting.
    Mix,
    /// DRIFT: Dynamic Reasoning with Flexible Traversal.
    /// Three-phase: community hierarchy primer → follow-up drill-down → leaf entity retrieval.
    Drift,
    /// LazyGraphRAG: no pre-computation. Vector search → local subgraph → on-demand mini-community summary.
    Lazy,
    /// Personalized PageRank: seed from query-relevant entities, run PPR with restart α=0.15.
    Ppr,
}

impl RetrievalMode {
    /// Parse a mode string into a `RetrievalMode`, defaulting to `Hybrid`
    /// for unrecognized values.
    pub fn from_str_or_default(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "local" => Self::Local,
            "global" => Self::Global,
            "mix" => Self::Mix,
            "drift" => Self::Drift,
            "lazy" => Self::Lazy,
            "ppr" => Self::Ppr,
            _ => Self::Hybrid,
        }
    }
}

impl Default for RetrievalMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Traversal backbone for DRIFT retrieval. `Community` (default) walks
/// the hierarchical-Leiden community tree — the pre-Fuzzy-Phase-8
/// behavior. `Lattice` walks a persisted FCA concept lattice keyed by
/// `lattice_id`.
/// Cites: [belohlavek2004fuzzyfca].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DriftPath {
    Community,
    Lattice { lattice_id: String },
}

impl Default for DriftPath {
    fn default() -> Self {
        Self::Community
    }
}

/// Configuration for RAG query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Token budget for context assembly.
    pub budget: TokenBudget,
    /// Default retrieval mode when not specified in the query.
    pub default_mode: RetrievalMode,
    /// Weight for vector vs BM25 in hybrid search: `alpha * vector + (1-alpha) * bm25`.
    /// Range [0.0, 1.0]. Default: 0.7 (70% vector, 30% BM25).
    #[serde(default = "default_hybrid_alpha")]
    pub hybrid_alpha: f32,
    /// Reranker type for post-retrieval relevance scoring.
    #[serde(default)]
    pub reranker_type: crate::query::reranker::RerankerType,
    /// Traversal backbone for DRIFT retrieval. `Default = Community`
    /// preserves pre-Fuzzy-Phase-8 behavior bit-identically.
    /// Cites: [belohlavek2004fuzzyfca].
    #[serde(default)]
    pub drift_path: DriftPath,
}

fn default_hybrid_alpha() -> f32 {
    0.7
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            budget: TokenBudget::default(),
            default_mode: RetrievalMode::Hybrid,
            hybrid_alpha: default_hybrid_alpha(),
            reranker_type: crate::query::reranker::RerankerType::default(),
            drift_path: DriftPath::default(),
        }
    }
}
