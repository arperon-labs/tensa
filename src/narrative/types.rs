//! Types for cross-narrative learning (Phase 3).
//!
//! Defines narrative metadata, corpus splits, statistics,
//! and pattern-related types used across the narrative module.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::SourceReference;

// ─── Taxonomy ──────────────────────────────────────────────

/// A single entry in a taxonomy category (e.g. genre, content_type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyEntry {
    pub category: String,
    pub value: String,
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
    pub is_builtin: bool,
}

// ─── Project ────────────────────────────────────────────────

/// A project is a top-level container that groups related narratives.
/// For example, a "Geopolitics" project might contain narratives for
/// "Ukraine", "Middle East", and "South China Sea".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Human-readable slug, e.g. "geopolitics".
    pub id: String,
    /// Display title.
    pub title: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Free-form tags for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Cached count of narratives in this project.
    #[serde(default)]
    pub narrative_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ─── Narrative Metadata ──────────────────────────────────────

/// A narrative is a named, isolated collection of entities and
/// situations within the hypergraph. Each entity/situation carries
/// an optional `narrative_id` that associates it with a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Narrative {
    /// Human-readable slug, e.g. "crime-and-punishment".
    pub id: String,
    /// Display title.
    pub title: String,
    /// Optional genre tag (e.g. "novel", "investigation", "geopolitical").
    pub genre: Option<String>,
    /// Free-form tags for filtering.
    pub tags: Vec<String>,
    /// Provenance of the narrative source.
    pub source: Option<SourceReference>,
    /// Optional parent project.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Author(s) of the narrative source material.
    #[serde(default)]
    pub authors: Vec<String>,
    /// ISO 639-1 language code (e.g. "en", "ru", "de").
    #[serde(default)]
    pub language: Option<String>,
    /// Publication date of the original source material.
    #[serde(default)]
    pub publication_date: Option<DateTime<Utc>>,
    /// URL to a cover image.
    #[serde(default)]
    pub cover_url: Option<String>,
    /// Arbitrary user-defined key-value metadata. Values may be any JSON —
    /// strings, numbers, booleans, arrays, or nested objects.
    #[serde(default)]
    pub custom_properties: HashMap<String, serde_json::Value>,
    /// Cached count of entities in this narrative.
    pub entity_count: usize,
    /// Cached count of situations in this narrative.
    pub situation_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ─── Corpus Management ───────────────────────────────────────

/// A train/test/validation split over narrative IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusSplit {
    pub train: Vec<String>,
    pub test: Vec<String>,
    pub validation: Vec<String>,
    /// Seed used for randomization.
    pub seed: u64,
}

/// Report returned by a narrative merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeMergeReport {
    pub target_id: String,
    pub source_id: String,
    pub entities_moved: usize,
    pub situations_moved: usize,
    pub chunks_moved: usize,
    pub source_deleted: bool,
}

/// Statistics for a single narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeStats {
    pub narrative_id: String,
    pub entity_count: usize,
    pub situation_count: usize,
    pub participation_count: usize,
    pub causal_link_count: usize,
    /// Earliest and latest situation timestamps, if any.
    pub temporal_span: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Count of situations per narrative level.
    pub narrative_levels: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_narrative_serialization_roundtrip() {
        let narrative = Narrative {
            id: "hamlet".to_string(),
            title: "Hamlet".to_string(),
            genre: Some("tragedy".to_string()),
            tags: vec!["shakespeare".to_string(), "revenge".to_string()],
            source: None,
            project_id: None,
            description: None,
            authors: vec!["Shakespeare".to_string()],
            language: Some("en".to_string()),
            publication_date: None,
            cover_url: None,
            custom_properties: HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_vec(&narrative).unwrap();
        let decoded: Narrative = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.id, "hamlet");
        assert_eq!(decoded.genre, Some("tragedy".to_string()));
        assert_eq!(decoded.tags.len(), 2);
    }

    #[test]
    fn test_corpus_split_serialization() {
        let split = CorpusSplit {
            train: vec!["a".to_string(), "b".to_string()],
            test: vec!["c".to_string()],
            validation: vec!["d".to_string()],
            seed: 42,
        };
        let json = serde_json::to_vec(&split).unwrap();
        let decoded: CorpusSplit = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.train.len(), 2);
        assert_eq!(decoded.seed, 42);
    }

    #[test]
    fn test_narrative_stats_serialization() {
        let stats = NarrativeStats {
            narrative_id: "test".to_string(),
            entity_count: 10,
            situation_count: 25,
            participation_count: 50,
            causal_link_count: 8,
            temporal_span: None,
            narrative_levels: HashMap::from([("Scene".to_string(), 15), ("Event".to_string(), 10)]),
        };
        let json = serde_json::to_vec(&stats).unwrap();
        let decoded: NarrativeStats = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.entity_count, 10);
        assert_eq!(decoded.narrative_levels["Scene"], 15);
    }

    #[test]
    fn test_narrative_without_optional_fields() {
        let narrative = Narrative {
            id: "minimal".to_string(),
            title: "Minimal".to_string(),
            genre: None,
            tags: vec![],
            source: None,
            project_id: None,
            description: None,
            authors: vec![],
            language: None,
            publication_date: None,
            cover_url: None,
            custom_properties: HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_vec(&narrative).unwrap();
        let decoded: Narrative = serde_json::from_slice(&json).unwrap();
        assert!(decoded.genre.is_none());
        assert!(decoded.tags.is_empty());
    }
}
