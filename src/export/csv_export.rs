//! CSV export for narrative data (Sprint P3.6 — F-AP4).
//!
//! Exports entities and situations as CSV rows with a unified schema.

use crate::error::{Result, TensaError};
use crate::export::{entity_display_name, NarrativeExport};

/// Export narrative data as CSV string.
///
/// Output format:
/// ```csv
/// type,id,name,entity_type,narrative_level,confidence,created_at,properties
/// entity,<uuid>,Alice,Actor,,0.9,2026-01-01T00:00:00Z,"{...}"
/// situation,<uuid>,,,,0.8,2026-01-01T00:00:00Z,"{...}"
/// ```
pub fn export_csv(data: &NarrativeExport) -> Result<String> {
    let mut writer = csv::Writer::from_writer(Vec::new());

    writer
        .write_record([
            "type",
            "id",
            "name",
            "entity_type",
            "narrative_level",
            "confidence",
            "created_at",
            "properties",
        ])
        .map_err(|e| TensaError::ExportError(format!("CSV header: {e}")))?;

    for entity in &data.entities {
        let name = entity_display_name(entity, "");
        let entity_type = format!("{:?}", entity.entity_type);
        let props = serde_json::to_string(&entity.properties).unwrap_or_default();

        writer
            .write_record([
                "entity",
                &entity.id.to_string(),
                name,
                &entity_type,
                "",
                &format!("{:.2}", entity.confidence),
                &entity.created_at.to_rfc3339(),
                &props,
            ])
            .map_err(|e| TensaError::ExportError(format!("CSV entity row: {e}")))?;
    }

    for sit in &data.situations {
        let description = sit
            .raw_content
            .first()
            .map(|c| c.content.as_str())
            .unwrap_or("");
        let level = format!("{:?}", sit.narrative_level);

        writer
            .write_record([
                "situation",
                &sit.id.to_string(),
                description,
                "",
                &level,
                &format!("{:.2}", sit.confidence),
                &sit.created_at.to_rfc3339(),
                "",
            ])
            .map_err(|e| TensaError::ExportError(format!("CSV situation row: {e}")))?;
    }

    let bytes = writer
        .into_inner()
        .map_err(|e| TensaError::ExportError(format!("CSV flush: {e}")))?;
    String::from_utf8(bytes).map_err(|e| TensaError::ExportError(format!("CSV encoding: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::NarrativeExport;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_export(entities: Vec<Entity>, situations: Vec<Situation>) -> NarrativeExport {
        NarrativeExport {
            narrative_id: "test".into(),
            entities,
            situations,
            participations: vec![],
            causal_links: vec![],
        }
    }

    fn make_entity(name: &str) -> Entity {
        let now = Utc::now();
        Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("test".into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn make_situation(desc: &str) -> Situation {
        let now = Utc::now();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(desc)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("test".into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[test]
    fn test_csv_export_entities() {
        let data = make_export(vec![make_entity("Alice"), make_entity("Bob")], vec![]);
        let csv = export_csv(&data).unwrap();
        assert!(csv.contains("entity"));
        assert!(csv.contains("Alice"));
        assert!(csv.contains("Bob"));
        // Count entity rows (minus header)
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 entities
    }

    #[test]
    fn test_csv_export_situations() {
        let data = make_export(
            vec![],
            vec![make_situation("Meeting"), make_situation("Departure")],
        );
        let csv = export_csv(&data).unwrap();
        assert!(csv.contains("situation"));
        assert!(csv.contains("Meeting"));
        assert!(csv.contains("Departure"));
    }

    #[test]
    fn test_csv_export_mixed() {
        let data = make_export(vec![make_entity("Alice")], vec![make_situation("Meeting")]);
        let csv = export_csv(&data).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 1 entity + 1 situation
        assert!(lines[1].starts_with("entity"));
        assert!(lines[2].starts_with("situation"));
    }

    #[test]
    fn test_csv_export_empty() {
        let data = make_export(vec![], vec![]);
        let csv = export_csv(&data).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 1); // header only
        assert!(lines[0].contains("type"));
    }

    #[test]
    fn test_csv_export_special_chars() {
        let entity = {
            let mut e = make_entity("Alice, \"the Great\"");
            e.properties = serde_json::json!({"name": "Alice, \"the Great\""});
            e
        };
        let data = make_export(vec![entity], vec![]);
        let csv = export_csv(&data).unwrap();
        // csv crate should properly quote/escape
        assert!(csv.contains("Alice"));
        // Should be parseable back
        let mut reader = csv::Reader::from_reader(csv.as_bytes());
        let records: Vec<_> = reader.records().collect();
        assert_eq!(records.len(), 1);
        assert!(records[0].is_ok());
    }
}
