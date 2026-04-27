//! Maltego transform export (Sprint D6).
//!
//! Exports narrative entities as a Maltego-compatible transform result set.
//! Each TENSA entity maps to a Maltego `MaltegoEntity` with type, value,
//! and additional fields from properties.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::EntityType;

use super::{collect_narrative_data, entity_display_name};

/// Top-level Maltego transform result.
#[derive(Debug, Serialize, Deserialize)]
pub struct MaltegoTransformResult {
    pub entities: Vec<MaltegoEntity>,
    pub messages: Vec<String>,
}

/// A single entity in Maltego transform output.
#[derive(Debug, Serialize, Deserialize)]
pub struct MaltegoEntity {
    /// Maltego entity type (e.g. "maltego.Person", "maltego.Location").
    #[serde(rename = "type")]
    pub entity_type: String,
    /// Primary display value.
    pub value: String,
    /// Weight (higher = more central). Mapped from TENSA confidence.
    pub weight: i32,
    /// Additional properties.
    pub properties: Vec<MaltegoProperty>,
}

/// A key-value property on a Maltego entity.
#[derive(Debug, Serialize, Deserialize)]
pub struct MaltegoProperty {
    pub name: String,
    pub display_name: String,
    pub value: String,
}

/// Map TENSA EntityType to Maltego entity type string.
fn maltego_entity_type(et: &EntityType) -> &'static str {
    match et {
        EntityType::Actor => "maltego.Person",
        EntityType::Location => "maltego.Location",
        EntityType::Organization => "maltego.Organization",
        EntityType::Artifact => "maltego.Document",
        EntityType::Concept => "maltego.Phrase",
    }
}

/// Export narrative entities as a Maltego transform result.
pub fn export_maltego(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<MaltegoTransformResult> {
    let data = collect_narrative_data(narrative_id, hypergraph)?;

    let entities: Vec<MaltegoEntity> = data
        .entities
        .iter()
        .map(|e| {
            let name = entity_display_name(e, "(unnamed)");
            let weight = (e.confidence * 100.0) as i32;

            let mut properties = vec![
                MaltegoProperty {
                    name: "tensa.entity_id".to_string(),
                    display_name: "TENSA ID".to_string(),
                    value: e.id.to_string(),
                },
                MaltegoProperty {
                    name: "tensa.confidence".to_string(),
                    display_name: "Confidence".to_string(),
                    value: format!("{:.2}", e.confidence),
                },
                MaltegoProperty {
                    name: "tensa.maturity".to_string(),
                    display_name: "Maturity".to_string(),
                    value: format!("{:?}", e.maturity),
                },
            ];

            // Add narrative_id if present
            if let Some(ref nid) = e.narrative_id {
                properties.push(MaltegoProperty {
                    name: "tensa.narrative_id".to_string(),
                    display_name: "Narrative".to_string(),
                    value: nid.clone(),
                });
            }

            // Add platform if present
            if let Some(platform) = e.properties.get("platform").and_then(|v| v.as_str()) {
                properties.push(MaltegoProperty {
                    name: "tensa.platform".to_string(),
                    display_name: "Platform".to_string(),
                    value: platform.to_string(),
                });
            }

            MaltegoEntity {
                entity_type: maltego_entity_type(&e.entity_type).to_string(),
                value: name.to_string(),
                weight,
                properties,
            }
        })
        .collect();

    let messages = vec![format!(
        "Exported {} entities from narrative '{}'",
        entities.len(),
        narrative_id
    )];

    Ok(MaltegoTransformResult { entities, messages })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_export_maltego_empty() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let result = export_maltego(&hg, "empty").unwrap();
        assert!(result.entities.is_empty());
    }

    #[test]
    fn test_export_maltego_with_entities() {
        use crate::types::*;
        use chrono::Utc;
        use uuid::Uuid;

        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Agent Smith", "platform": "twitter"}),
            beliefs: None,
            embedding: None,
            narrative_id: Some("test-maltego".to_string()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: None,
            provenance: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap();

        let result = export_maltego(&hg, "test-maltego").unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].entity_type, "maltego.Person");
        assert_eq!(result.entities[0].value, "Agent Smith");
        assert_eq!(result.entities[0].weight, 90);

        // Should have platform property
        let has_platform = result.entities[0]
            .properties
            .iter()
            .any(|p| p.name == "tensa.platform" && p.value == "twitter");
        assert!(has_platform);
    }
}
