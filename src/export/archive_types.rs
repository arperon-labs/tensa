//! Types for the Tensa Narrative Archive (.tensa) format.
//!
//! Defines the manifest, layer flags, export/import options, and report
//! types for the ZIP-based lossless narrative exchange format.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Current archive format version.
pub const ARCHIVE_VERSION: &str = "1.0.0";

/// MIME type for .tensa archives.
pub const ARCHIVE_CONTENT_TYPE: &str = "application/x-tensa-archive";

// ─── Manifest ───────────────────────────────────────────────

/// Top-level manifest at `manifest.json` in the archive root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveManifest {
    /// Semver format version (e.g. "1.0.0").
    pub tensa_archive_version: String,
    /// When the archive was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Tool that created the archive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<ArchiveCreatedBy>,
    /// List of narrative slugs included.
    pub narratives: Vec<String>,
    /// Which optional layers are present.
    pub layers: ArchiveLayers,
    /// When false, dangling UUID references produce warnings instead of errors.
    #[serde(default = "default_true")]
    pub strict_references: bool,
    /// Opaque identifier for the originating Tensa instance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_namespace: Option<String>,
    /// Human-readable description of the archive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

pub fn default_true() -> bool {
    true
}

/// Identifies the tool that created the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveCreatedBy {
    pub tool: String,
    pub version: String,
}

/// Boolean flags indicating which optional layers are present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveLayers {
    /// Core graph data (entities, situations, participations) — always true.
    #[serde(default = "default_true")]
    pub core: bool,
    /// Source intelligence (sources, attributions, contentions).
    #[serde(default)]
    pub sources: bool,
    /// Original text chunks from ingestion.
    #[serde(default)]
    pub chunks: bool,
    /// Entity state version history.
    #[serde(default)]
    pub state_versions: bool,
    /// Inference job results.
    #[serde(default)]
    pub inference: bool,
    /// Analysis data (communities, style profiles).
    #[serde(default)]
    pub analysis: bool,
    /// LLM prompt tuning data.
    #[serde(default)]
    pub tuning: bool,
    /// Binary embedding vectors.
    #[serde(default)]
    pub embeddings: bool,
    /// Custom taxonomy entries.
    #[serde(default)]
    pub taxonomy: bool,
    /// Project containers.
    #[serde(default)]
    pub projects: bool,
}

impl Default for ArchiveLayers {
    fn default() -> Self {
        Self {
            core: true,
            sources: false,
            chunks: false,
            state_versions: false,
            inference: false,
            analysis: false,
            tuning: false,
            embeddings: false,
            taxonomy: false,
            projects: false,
        }
    }
}

// ─── Embedding Index ────────────────────────────────────────

/// Maps UUIDs to byte offsets in binary embedding files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingIndex {
    /// Dimension of all embeddings in this narrative.
    pub dimension: usize,
    /// Entity UUID → offset mapping.
    #[serde(default)]
    pub entities: HashMap<String, EmbeddingEntry>,
    /// Situation UUID → offset mapping.
    #[serde(default)]
    pub situations: HashMap<String, EmbeddingEntry>,
    /// Chunk UUID → offset mapping.
    #[serde(default)]
    pub chunks: HashMap<String, EmbeddingEntry>,
}

/// A single embedding's location in a binary file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEntry {
    /// Which .bin file contains this embedding.
    pub file: String,
    /// Byte offset within the file.
    pub offset: usize,
}

// ─── Export Options ─────────────────────────────────────────

/// Options controlling what layers are included in an archive export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveExportOptions {
    /// Include source intelligence layer.
    #[serde(default = "default_true")]
    pub include_sources: bool,
    /// Include original text chunks.
    #[serde(default = "default_true")]
    pub include_chunks: bool,
    /// Include entity state version history.
    #[serde(default = "default_true")]
    pub include_state_versions: bool,
    /// Include inference results.
    #[serde(default)]
    pub include_inference: bool,
    /// Include analysis data (communities, style profiles).
    #[serde(default = "default_true")]
    pub include_analysis: bool,
    /// Include tuned prompts.
    #[serde(default = "default_true")]
    pub include_tuning: bool,
    /// Include binary embedding vectors.
    #[serde(default)]
    pub include_embeddings: bool,
    /// Include taxonomy entries.
    #[serde(default = "default_true")]
    pub include_taxonomy: bool,
    /// Include project containers.
    #[serde(default = "default_true")]
    pub include_projects: bool,
    /// Pretty-print JSON (larger but human-readable).
    #[serde(default = "default_true")]
    pub pretty: bool,
    /// EATH Phase 3 — when false (default), synthetic entities and
    /// situations are stripped before export. Set true to preserve
    /// synthetic records alongside empirical ones (e.g. when exporting
    /// a calibrated surrogate run for reproduction).
    #[serde(default)]
    pub include_synthetic: bool,
}

impl Default for ArchiveExportOptions {
    fn default() -> Self {
        Self {
            include_sources: true,
            include_chunks: true,
            include_state_versions: true,
            include_inference: false,
            include_analysis: true,
            include_tuning: true,
            include_embeddings: false,
            include_taxonomy: true,
            include_projects: true,
            pretty: true,
            include_synthetic: false,
        }
    }
}

// ─── Import Options ─────────────────────────────────────────

/// Options controlling archive import behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveImportOptions {
    /// When true, skip records whose UUIDs already exist instead of remapping.
    pub merge_mode: bool,
    /// When true, reject the archive if dangling references are found.
    pub strict_references: bool,
    /// Override the narrative slug on import.
    pub target_narrative_id: Option<String>,
}

impl Default for ArchiveImportOptions {
    fn default() -> Self {
        Self {
            merge_mode: false,
            strict_references: true,
            target_narrative_id: None,
        }
    }
}

// ─── Import Report ──────────────────────────────────────────

/// Detailed report from an archive import operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveImportReport {
    pub narratives_imported: usize,
    pub entities_created: usize,
    pub entities_skipped: usize,
    pub situations_created: usize,
    pub situations_skipped: usize,
    pub participations_created: usize,
    pub causal_links_created: usize,
    pub sources_created: usize,
    pub attributions_created: usize,
    pub contentions_created: usize,
    pub chunks_created: usize,
    pub state_versions_created: usize,
    pub inference_results_created: usize,
    pub communities_created: usize,
    pub prompts_created: usize,
    pub taxonomy_entries_created: usize,
    pub projects_created: usize,
    /// Old UUID → new UUID mappings for clashed records.
    pub id_remaps: HashMap<String, String>,
    /// Non-fatal warnings (e.g. dangling references in relaxed mode).
    pub warnings: Vec<String>,
    /// Fatal errors that prevented import of specific items.
    pub errors: Vec<String>,
}

impl Default for ArchiveImportReport {
    fn default() -> Self {
        Self {
            narratives_imported: 0,
            entities_created: 0,
            entities_skipped: 0,
            situations_created: 0,
            situations_skipped: 0,
            participations_created: 0,
            causal_links_created: 0,
            sources_created: 0,
            attributions_created: 0,
            contentions_created: 0,
            chunks_created: 0,
            state_versions_created: 0,
            inference_results_created: 0,
            communities_created: 0,
            prompts_created: 0,
            taxonomy_entries_created: 0,
            projects_created: 0,
            id_remaps: HashMap::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_roundtrip() {
        let manifest = ArchiveManifest {
            tensa_archive_version: ARCHIVE_VERSION.to_string(),
            created_at: chrono::Utc::now(),
            created_by: Some(ArchiveCreatedBy {
                tool: "tensa".to_string(),
                version: "0.14.2".to_string(),
            }),
            narratives: vec!["test-narrative".to_string()],
            layers: ArchiveLayers {
                core: true,
                sources: true,
                ..Default::default()
            },
            strict_references: true,
            id_namespace: None,
            description: Some("Test archive".to_string()),
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: ArchiveManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tensa_archive_version, ARCHIVE_VERSION);
        assert_eq!(parsed.narratives.len(), 1);
        assert!(parsed.layers.core);
        assert!(parsed.layers.sources);
        assert!(!parsed.layers.chunks);
    }

    #[test]
    fn test_layers_default() {
        let layers = ArchiveLayers::default();
        assert!(layers.core);
        assert!(!layers.sources);
        assert!(!layers.embeddings);
    }

    #[test]
    fn test_import_report_default() {
        let report = ArchiveImportReport::default();
        assert_eq!(report.entities_created, 0);
        assert!(report.id_remaps.is_empty());
        assert!(report.warnings.is_empty());
    }

    #[test]
    fn test_export_options_default() {
        let opts = ArchiveExportOptions::default();
        assert!(opts.include_sources);
        assert!(opts.include_chunks);
        assert!(!opts.include_embeddings); // Large, off by default
        assert!(!opts.include_inference); // Regenerable, off by default
        assert!(opts.pretty);
    }
}
