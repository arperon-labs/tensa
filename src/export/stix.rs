//! STIX 2.1 export — map TENSA narrative data to STIX bundle format.
//!
//! Mapping: Entity(Actor) → threat-actor, Entity(Artifact) → indicator,
//! Entity(Location) → location, Situation → observed-data,
//! Participation → relationship SRO, Narrative → campaign.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::Result;
use crate::export::NarrativeExport;
use crate::types::*;

/// STIX 2.1 bundle wrapper.
#[derive(Debug, Serialize, Deserialize)]
pub struct StixBundle {
    #[serde(rename = "type")]
    pub bundle_type: String,
    pub id: String,
    pub objects: Vec<Value>,
}

// ─── ID Conversion ─────────────────────────────────────────

/// Generate a deterministic UUID from input bytes via SHA-256.
/// Takes the first 16 bytes of the hash and sets version/variant bits.
pub(crate) fn deterministic_uuid(input: &[u8]) -> Uuid {
    let hash = Sha256::digest(input);
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hash[..16]);
    // Set version 5 (name-based SHA) and variant bits for compatibility.
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

/// Convert a TENSA UUID to a STIX identifier: `type--uuid`.
pub fn uuid_to_stix_id(stix_type: &str, id: Uuid) -> String {
    format!("{}--{}", stix_type, id)
}

/// Parse a STIX identifier `type--uuid` back to a UUID.
/// Returns `None` if the format is invalid.
pub fn stix_id_to_uuid(stix_id: &str) -> Option<Uuid> {
    let parts: Vec<&str> = stix_id.splitn(2, "--").collect();
    if parts.len() == 2 {
        Uuid::parse_str(parts[1]).ok()
    } else {
        None
    }
}

/// Map TENSA EntityType to STIX SDO type.
fn entity_type_to_stix(et: &EntityType) -> &'static str {
    match et {
        EntityType::Actor => "threat-actor",
        EntityType::Organization => "identity",
        EntityType::Location => "location",
        EntityType::Artifact => "indicator",
        EntityType::Concept => "note",
    }
}

/// Map STIX SDO type to TENSA EntityType.
pub fn stix_type_to_entity_type(stix_type: &str) -> Option<EntityType> {
    match stix_type {
        "threat-actor" => Some(EntityType::Actor),
        "identity" => Some(EntityType::Organization),
        "location" => Some(EntityType::Location),
        "indicator" | "malware" | "tool" => Some(EntityType::Artifact),
        "note" => Some(EntityType::Concept),
        "campaign" => None, // maps to Narrative, not Entity
        "relationship" => None,
        "observed-data" | "sighting" => None, // maps to Situation
        _ => None,
    }
}

/// Convert a STIX confidence (0-100) to TENSA confidence (0.0-1.0).
pub fn stix_confidence_to_tensa(stix_conf: u32) -> f32 {
    (stix_conf.min(100) as f32) / 100.0
}

/// Convert TENSA confidence (0.0-1.0) to STIX confidence (0-100).
fn tensa_confidence_to_stix(conf: f32) -> u32 {
    (conf.clamp(0.0, 1.0) * 100.0).round() as u32
}

/// Convert an optional DateTime to STIX timestamp string.
fn to_stix_timestamp(dt: Option<DateTime<Utc>>) -> Value {
    match dt {
        Some(t) => json!(t.to_rfc3339()),
        None => Value::Null,
    }
}

// ─── Export ────────────────────────────────────────────────

/// Export a `NarrativeExport` as a STIX 2.1 JSON bundle.
pub fn export_stix(data: &NarrativeExport) -> Result<String> {
    let mut objects: Vec<Value> = Vec::new();
    let now = Utc::now().to_rfc3339();

    // Campaign SDO for the narrative itself.
    objects.push(json!({
        "type": "campaign",
        "spec_version": "2.1",
        "id": format!("campaign--{}", deterministic_uuid(data.narrative_id.as_bytes())),
        "name": data.narrative_id,
        "created": &now,
        "modified": &now,
    }));

    // Entities → SDOs.
    for entity in &data.entities {
        let stix_type = entity_type_to_stix(&entity.entity_type);
        let name = super::entity_display_name(entity, "Unknown");

        let mut obj = json!({
            "type": stix_type,
            "spec_version": "2.1",
            "id": uuid_to_stix_id(stix_type, entity.id),
            "created": entity.created_at.to_rfc3339(),
            "modified": entity.updated_at.to_rfc3339(),
            "confidence": tensa_confidence_to_stix(entity.confidence),
        });

        // Add name/description based on type.
        match stix_type {
            "threat-actor" => {
                obj["name"] = json!(name);
                obj["threat_actor_types"] = json!(["unknown"]);
            }
            "identity" => {
                obj["name"] = json!(name);
                obj["identity_class"] = json!("organization");
            }
            "location" => {
                obj["name"] = json!(name);
                if let Some(lat) = entity.properties.get("latitude") {
                    obj["latitude"] = lat.clone();
                }
                if let Some(lon) = entity.properties.get("longitude") {
                    obj["longitude"] = lon.clone();
                }
            }
            "indicator" => {
                obj["name"] = json!(name);
                obj["pattern_type"] = json!("stix");
                obj["pattern"] = json!(format!("[artifact:payload_bin = '{}']", name));
                obj["valid_from"] = json!(entity.created_at.to_rfc3339());
            }
            _ => {
                obj["content"] = json!(name);
            }
        }

        objects.push(obj);
    }

    // Situations → observed-data SDOs.
    for sit in &data.situations {
        let content: String = sit
            .raw_content
            .iter()
            .map(|b| b.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let mut obj = json!({
            "type": "observed-data",
            "spec_version": "2.1",
            "id": uuid_to_stix_id("observed-data", sit.id),
            "created": sit.created_at.to_rfc3339(),
            "modified": sit.updated_at.to_rfc3339(),
            "confidence": tensa_confidence_to_stix(sit.confidence),
            "number_observed": 1,
            "object_refs": [],
        });
        obj["first_observed"] = to_stix_timestamp(sit.temporal.start);
        obj["last_observed"] = to_stix_timestamp(sit.temporal.end.or(sit.temporal.start));

        // Store content as extension.
        obj["x_tensa_content"] = json!(content);
        if let Some(name) = &sit.name {
            obj["x_tensa_name"] = json!(name);
        }

        objects.push(obj);
    }

    // Participations → relationship SROs.
    for p in &data.participations {
        let entity_stix_type = data
            .entities
            .iter()
            .find(|e| e.id == p.entity_id)
            .map(|e| entity_type_to_stix(&e.entity_type))
            .unwrap_or("identity");

        let rel_type = match p.role {
            Role::Protagonist => "attributed-to",
            Role::Antagonist => "targets",
            Role::Witness => "related-to",
            Role::Target => "targets",
            Role::Instrument => "uses",
            _ => "related-to",
        };

        objects.push(json!({
            "type": "relationship",
            "spec_version": "2.1",
            "id": uuid_to_stix_id("relationship", deterministic_uuid(
                format!("{}:{}:{}", p.entity_id, p.situation_id, p.seq).as_bytes()
            )),
            "relationship_type": rel_type,
            "source_ref": uuid_to_stix_id(entity_stix_type, p.entity_id),
            "target_ref": uuid_to_stix_id("observed-data", p.situation_id),
            "created": &now,
            "modified": &now,
        }));
    }

    // Causal links → relationship SROs.
    for link in &data.causal_links {
        objects.push(json!({
            "type": "relationship",
            "spec_version": "2.1",
            "id": uuid_to_stix_id("relationship", deterministic_uuid(
                format!("cause:{}:{}", link.from_situation, link.to_situation).as_bytes()
            )),
            "relationship_type": "causes",
            "source_ref": uuid_to_stix_id("observed-data", link.from_situation),
            "target_ref": uuid_to_stix_id("observed-data", link.to_situation),
            "x_tensa_mechanism": link.mechanism.as_deref().unwrap_or(""),
            "x_tensa_strength": link.strength,
            "created": &now,
            "modified": &now,
        }));
    }

    let bundle = StixBundle {
        bundle_type: "bundle".into(),
        id: format!(
            "bundle--{}",
            deterministic_uuid(format!("bundle:{}", data.narrative_id).as_bytes())
        ),
        objects,
    };

    serde_json::to_string_pretty(&bundle)
        .map_err(|e| crate::error::TensaError::ExportError(format!("STIX JSON: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::NarrativeExport;
    use chrono::Utc;

    fn make_export() -> NarrativeExport {
        let now = Utc::now();
        NarrativeExport {
            narrative_id: "test-case".into(),
            entities: vec![
                Entity {
                    id: Uuid::now_v7(),
                    entity_type: EntityType::Actor,
                    properties: json!({"name": "Suspect Alpha"}),
                    beliefs: None,
                    embedding: None,
                    maturity: MaturityLevel::Validated,
                    confidence: 0.85,
                    confidence_breakdown: None,
                    provenance: vec![],
                    extraction_method: None,
                    narrative_id: Some("test-case".into()),
                    created_at: now,
                    updated_at: now,
                    deleted_at: None,
                    transaction_time: None,
                },
                Entity {
                    id: Uuid::now_v7(),
                    entity_type: EntityType::Location,
                    properties: json!({"name": "Safe House", "latitude": 48.8566, "longitude": 2.3522}),
                    beliefs: None,
                    embedding: None,
                    maturity: MaturityLevel::Candidate,
                    confidence: 0.7,
                    confidence_breakdown: None,
                    provenance: vec![],
                    extraction_method: None,
                    narrative_id: Some("test-case".into()),
                    created_at: now,
                    updated_at: now,
                    deleted_at: None,
                    transaction_time: None,
                },
            ],
            situations: vec![Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: Some("Meeting".into()),
                description: None,
                temporal: AllenInterval {
                    start: Some(now),
                    end: Some(now),
                    granularity: TimeGranularity::Approximate,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![ContentBlock::text("Suspect met contact at safe house")],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.8,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some("test-case".into()),
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
            }],
            participations: vec![],
            causal_links: vec![],
        }
    }

    #[test]
    fn test_stix_export_narrative() {
        let data = make_export();
        let json_str = export_stix(&data).unwrap();
        let bundle: StixBundle = serde_json::from_str(&json_str).unwrap();
        assert_eq!(bundle.bundle_type, "bundle");
        // campaign + 2 entities + 1 situation = 4 objects
        assert_eq!(bundle.objects.len(), 4);
        // First object should be the campaign.
        assert_eq!(bundle.objects[0]["type"], "campaign");
        assert_eq!(bundle.objects[0]["name"], "test-case");
    }

    #[test]
    fn test_stix_id_to_uuid() {
        let id = Uuid::now_v7();
        let stix_id = uuid_to_stix_id("threat-actor", id);
        assert!(stix_id.starts_with("threat-actor--"));
        let parsed = stix_id_to_uuid(&stix_id).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn test_stix_confidence_scaling() {
        assert_eq!(stix_confidence_to_tensa(85), 0.85);
        assert_eq!(stix_confidence_to_tensa(0), 0.0);
        assert_eq!(stix_confidence_to_tensa(100), 1.0);
        assert_eq!(stix_confidence_to_tensa(150), 1.0); // clamped
        assert_eq!(tensa_confidence_to_stix(0.85), 85);
        assert_eq!(tensa_confidence_to_stix(1.0), 100);
    }

    #[test]
    fn test_stix_roundtrip() {
        let data = make_export();
        let json_str = export_stix(&data).unwrap();
        let bundle: StixBundle = serde_json::from_str(&json_str).unwrap();
        // Verify entity IDs are recoverable.
        let entity_obj = &bundle.objects[1]; // first entity after campaign
        let stix_id = entity_obj["id"].as_str().unwrap();
        let recovered = stix_id_to_uuid(stix_id).unwrap();
        assert_eq!(recovered, data.entities[0].id);
    }

    #[test]
    fn test_stix_timestamp_to_allen() {
        // Situations with timestamps should produce first_observed/last_observed.
        let data = make_export();
        let json_str = export_stix(&data).unwrap();
        let bundle: StixBundle = serde_json::from_str(&json_str).unwrap();
        let sit_obj = bundle
            .objects
            .iter()
            .find(|o| o["type"] == "observed-data")
            .unwrap();
        assert!(sit_obj["first_observed"].is_string());
        assert!(sit_obj["last_observed"].is_string());
    }
}
