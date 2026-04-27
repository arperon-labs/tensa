//! MISP event export (Sprint D6).
//!
//! Exports a TENSA narrative as a MISP-compatible JSON event with attributes
//! mapped from entities and situations. This enables interoperability with
//! MISP threat intelligence sharing platforms.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::EntityType;

use super::{collect_narrative_data, entity_display_name, situation_preview};

/// MISP event envelope.
#[derive(Debug, Serialize, Deserialize)]
pub struct MispEvent {
    pub info: String,
    pub distribution: u8,
    pub threat_level_id: u8,
    pub analysis: u8,
    pub date: String,
    #[serde(rename = "Attribute")]
    pub attributes: Vec<MispAttribute>,
    #[serde(rename = "Tag")]
    pub tags: Vec<MispTag>,
}

/// MISP attribute (indicator-level datum).
#[derive(Debug, Serialize, Deserialize)]
pub struct MispAttribute {
    pub r#type: String,
    pub category: String,
    pub value: String,
    pub comment: String,
    pub to_ids: bool,
}

/// MISP tag (free-form label).
#[derive(Debug, Serialize, Deserialize)]
pub struct MispTag {
    pub name: String,
}

/// Export a TENSA narrative as a MISP event JSON.
pub fn export_misp(hypergraph: &Hypergraph, narrative_id: &str) -> Result<MispEvent> {
    let data = collect_narrative_data(narrative_id, hypergraph)?;

    let date = data
        .situations
        .first()
        .and_then(|s| s.temporal.start.map(|t| t.format("%Y-%m-%d").to_string()))
        .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%d").to_string());

    let mut attributes = Vec::new();

    // Map entities to MISP attributes
    for entity in &data.entities {
        let name = entity_display_name(entity, "(unnamed)");
        let (attr_type, category) = match entity.entity_type {
            EntityType::Actor => ("threat-actor", "Attribution"),
            EntityType::Location => ("text", "Targeting data"),
            EntityType::Organization => ("target-org", "Targeting data"),
            EntityType::Artifact => ("filename", "Payload delivery"),
            EntityType::Concept => ("text", "Other"),
        };

        attributes.push(MispAttribute {
            r#type: attr_type.to_string(),
            category: category.to_string(),
            value: name.to_string(),
            comment: format!(
                "TENSA entity {} (confidence: {:.2})",
                entity.id, entity.confidence
            ),
            to_ids: false,
        });

        // Extract URLs from entity properties as network-activity indicators
        if let Some(url) = entity.properties.get("url").and_then(|v| v.as_str()) {
            attributes.push(MispAttribute {
                r#type: "url".to_string(),
                category: "Network activity".to_string(),
                value: url.to_string(),
                comment: format!("URL associated with {}", name),
                to_ids: true,
            });
        }
    }

    // Map situations as internal-comment attributes
    for sit in &data.situations {
        let preview = situation_preview(sit, 200, || format!("situation-{}", sit.id));
        attributes.push(MispAttribute {
            r#type: "comment".to_string(),
            category: "Internal reference".to_string(),
            value: preview,
            comment: format!(
                "TENSA situation {} (confidence: {:.2})",
                sit.id, sit.confidence
            ),
            to_ids: false,
        });
    }

    let mut tags = vec![
        MispTag {
            name: format!("tensa:narrative={}", narrative_id),
        },
        MispTag {
            name: "tensa:export".to_string(),
        },
    ];

    // Add entity type tags
    let entity_types: std::collections::HashSet<String> = data
        .entities
        .iter()
        .map(|e| format!("tensa:entity-type={}", e.entity_type.as_index_str()))
        .collect();
    for tag in entity_types {
        tags.push(MispTag { name: tag });
    }

    Ok(MispEvent {
        info: format!("TENSA Narrative: {}", narrative_id),
        distribution: 0,    // Your organisation only
        threat_level_id: 2, // Medium
        analysis: 1,        // Ongoing
        date,
        attributes,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_export_misp_empty_narrative() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let event = export_misp(&hg, "empty-nar").unwrap();
        assert!(event.attributes.is_empty());
        assert_eq!(event.info, "TENSA Narrative: empty-nar");
        assert!(event.tags.len() >= 2);
    }

    #[test]
    fn test_export_misp_with_entity() {
        use crate::types::*;
        use chrono::Utc;
        use uuid::Uuid;

        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Bot Account X"}),
            beliefs: None,
            embedding: None,
            narrative_id: Some("test-misp".to_string()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.85,
            confidence_breakdown: None,
            extraction_method: None,
            provenance: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap();

        let event = export_misp(&hg, "test-misp").unwrap();
        assert_eq!(event.attributes.len(), 1);
        assert_eq!(event.attributes[0].value, "Bot Account X");
        assert_eq!(event.attributes[0].r#type, "threat-actor");
    }
}
