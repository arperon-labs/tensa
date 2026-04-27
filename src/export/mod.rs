//! Narrative export in multiple formats (Sprint P3.6 — F-AP4, Sprint P3.8).
//!
//! Supports CSV, GraphML, JSON, Manuscript, Report, and Archive export of narrative data.

pub mod archive;
pub mod archive_types;
pub mod compile;
pub mod csv_export;
#[cfg(feature = "disinfo")]
pub mod disinfo_report;
pub mod graphml;
#[cfg(feature = "disinfo")]
pub mod maltego;
pub mod manuscript;
#[cfg(feature = "disinfo")]
pub mod misp;
pub mod report;
#[cfg(feature = "disinfo")]
pub mod situation_report;
pub mod stix;

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::*;

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Csv,
    #[serde(alias = "graphml")]
    GraphML,
    Json,
    Manuscript,
    Report,
    Archive,
    Stix,
}

impl std::str::FromStr for ExportFormat {
    type Err = TensaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "csv" => Ok(Self::Csv),
            "graphml" => Ok(Self::GraphML),
            "json" => Ok(Self::Json),
            "manuscript" => Ok(Self::Manuscript),
            "report" => Ok(Self::Report),
            "archive" | "tensa" => Ok(Self::Archive),
            "stix" => Ok(Self::Stix),
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown export format: '{}'. Use csv, graphml, json, manuscript, report, archive, or stix.",
                other
            ))),
        }
    }
}

/// Output of an export operation.
pub struct ExportOutput {
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

/// Collected narrative data for export.
#[derive(Debug, Serialize, Deserialize)]
pub struct NarrativeExport {
    pub narrative_id: String,
    pub entities: Vec<Entity>,
    pub situations: Vec<Situation>,
    pub participations: Vec<Participation>,
    pub causal_links: Vec<CausalLink>,
}

/// Sort situations for manuscript narrated order.
///
/// Primary key is writer-curated `manuscript_order` (Sprint W7). Situations without
/// one sort after all ordered ones, using `temporal.start` / `created_at` as the
/// secondary key so newly-ingested material stays in chronological order until the
/// writer rearranges it. `id` (v7 UUID) is the stable tie-break.
pub(crate) fn sort_situations_by_time(situations: &[Situation]) -> Vec<&Situation> {
    let mut sorted: Vec<&Situation> = situations.iter().collect();
    sorted.sort_by_key(|s| crate::writer::scene::manuscript_sort_key(s));
    sorted
}

/// Truncate a string to at most `max_chars` characters, appending "..." if truncated.
/// Safe for multi-byte UTF-8 — never slices mid-codepoint.
pub(crate) fn truncate_with_ellipsis(text: &str, max_chars: usize) -> String {
    let mut char_indices = text.char_indices();
    match char_indices.nth(max_chars) {
        Some((byte_pos, _)) => format!("{}...", &text[..byte_pos]),
        None => text.to_string(),
    }
}

/// Extract a short preview from the first content block of a situation.
/// Returns `fallback()` if no content blocks exist.
pub(crate) fn situation_preview(
    sit: &Situation,
    max_chars: usize,
    fallback: impl FnOnce() -> String,
) -> String {
    sit.raw_content
        .first()
        .map(|b| truncate_with_ellipsis(b.content.trim(), max_chars))
        .unwrap_or_else(fallback)
}

/// Extract the `name` field from entity properties, returning `fallback` if absent.
pub(crate) fn entity_display_name<'a>(entity: &'a Entity, fallback: &'a str) -> &'a str {
    entity
        .properties
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback)
}

/// Collect all data for a narrative from the hypergraph.
pub fn collect_narrative_data(
    narrative_id: &str,
    hypergraph: &Hypergraph,
) -> Result<NarrativeExport> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    let mut participations = Vec::new();
    let mut causal_links = Vec::new();
    let sit_ids: HashSet<_> = situations.iter().map(|s| s.id).collect();

    for sit in &situations {
        participations.extend(hypergraph.get_participants_for_situation(&sit.id)?);

        for link in hypergraph.get_consequences(&sit.id)? {
            if sit_ids.contains(&link.to_situation) {
                causal_links.push(link);
            }
        }
    }

    Ok(NarrativeExport {
        narrative_id: narrative_id.to_string(),
        entities,
        situations,
        participations,
        causal_links,
    })
}

/// Export a narrative to the requested format.
///
/// When `source_mode` is true and chunks are available, manuscript export
/// uses original ingested text instead of LLM-extracted content.
pub fn export_narrative(
    narrative_id: &str,
    format: ExportFormat,
    hypergraph: &Hypergraph,
    source_mode: bool,
) -> Result<ExportOutput> {
    let data = collect_narrative_data(narrative_id, hypergraph)?;

    match format {
        ExportFormat::Json => {
            let json = serde_json::to_vec_pretty(&data)
                .map_err(|e| TensaError::ExportError(format!("JSON serialization: {e}")))?;
            Ok(ExportOutput {
                content_type: "application/json",
                body: json,
            })
        }
        ExportFormat::Csv => {
            let csv_str = csv_export::export_csv(&data)?;
            Ok(ExportOutput {
                content_type: "text/csv",
                body: csv_str.into_bytes(),
            })
        }
        ExportFormat::GraphML => {
            let xml = graphml::export_graphml(&data)?;
            Ok(ExportOutput {
                content_type: "application/xml",
                body: xml.into_bytes(),
            })
        }
        ExportFormat::Manuscript => {
            let md = if source_mode {
                // Try chunk-based export first
                let chunks = hypergraph.list_chunks_by_narrative(narrative_id)?;
                if chunks.is_empty() {
                    manuscript::export_manuscript(&data)?
                } else {
                    manuscript::export_manuscript_from_chunks(&chunks)?
                }
            } else {
                manuscript::export_manuscript(&data)?
            };
            Ok(ExportOutput {
                content_type: "text/markdown",
                body: md.into_bytes(),
            })
        }
        ExportFormat::Report => {
            let md = report::export_report(&data)?;
            Ok(ExportOutput {
                content_type: "text/markdown",
                body: md.into_bytes(),
            })
        }
        ExportFormat::Archive => {
            let opts = archive_types::ArchiveExportOptions::default();
            let bytes = archive::export_archive(&[narrative_id], hypergraph, &opts)?;
            Ok(ExportOutput {
                content_type: archive_types::ARCHIVE_CONTENT_TYPE,
                body: bytes,
            })
        }
        ExportFormat::Stix => {
            let json_str = stix::export_stix(&data)?;
            Ok(ExportOutput {
                content_type: "application/json",
                body: json_str.into_bytes(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::Hypergraph;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    fn setup_narrative(hg: &Hypergraph) {
        let now = Utc::now();
        let e = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Alice"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("test-nar".into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(e).unwrap();
    }

    #[test]
    fn test_export_json_roundtrip() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        setup_narrative(&hg);

        let output = export_narrative("test-nar", ExportFormat::Json, &hg, false).unwrap();
        assert_eq!(output.content_type, "application/json");
        let parsed: NarrativeExport = serde_json::from_slice(&output.body).unwrap();
        assert_eq!(parsed.narrative_id, "test-nar");
        assert_eq!(parsed.entities.len(), 1);
    }

    #[test]
    fn test_export_empty_narrative() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let output = export_narrative("empty", ExportFormat::Json, &hg, false).unwrap();
        let parsed: NarrativeExport = serde_json::from_slice(&output.body).unwrap();
        assert!(parsed.entities.is_empty());
        assert!(parsed.situations.is_empty());
    }

    #[test]
    fn test_export_format_dispatch() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let csv_out = export_narrative("x", ExportFormat::Csv, &hg, false).unwrap();
        assert_eq!(csv_out.content_type, "text/csv");

        let gml_out = export_narrative("x", ExportFormat::GraphML, &hg, false).unwrap();
        assert_eq!(gml_out.content_type, "application/xml");

        let json_out = export_narrative("x", ExportFormat::Json, &hg, false).unwrap();
        assert_eq!(json_out.content_type, "application/json");
    }

    #[test]
    fn test_export_manuscript_dispatch() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let out = export_narrative("x", ExportFormat::Manuscript, &hg, false).unwrap();
        assert_eq!(out.content_type, "text/markdown");
    }

    #[test]
    fn test_export_report_dispatch() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let out = export_narrative("x", ExportFormat::Report, &hg, false).unwrap();
        assert_eq!(out.content_type, "text/markdown");
    }
}
