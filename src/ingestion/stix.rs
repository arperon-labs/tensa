//! STIX 2.1 bundle import — parse STIX JSON into TENSA entities and situations.
//!
//! Mapping: campaign → Narrative metadata, threat-actor → Entity(Actor),
//! identity → Entity(Organization), location → Entity(Location),
//! indicator → Entity(Object), observed-data/sighting → Situation,
//! relationship → Participation or CausalLink.

use chrono::{DateTime, Utc};
use serde_json::Value;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::export::stix::{stix_confidence_to_tensa, stix_id_to_uuid, stix_type_to_entity_type};
use crate::hypergraph::Hypergraph;
use crate::types::*;

/// Result of importing a STIX bundle.
#[derive(Debug, serde::Serialize)]
pub struct StixImportReport {
    pub entities_created: usize,
    pub situations_created: usize,
    pub relationships_created: usize,
    pub skipped: usize,
    pub narrative_id: Option<String>,
}

/// Import a STIX 2.1 JSON bundle into the hypergraph.
///
/// The `narrative_id` parameter assigns all imported objects to a narrative.
/// If a campaign SDO is found, its name is used as the narrative_id instead.
pub fn import_stix_bundle(
    hypergraph: &Hypergraph,
    bundle_json: &str,
    default_narrative_id: &str,
) -> Result<StixImportReport> {
    let bundle: Value = serde_json::from_str(bundle_json)
        .map_err(|e| TensaError::ParseError(format!("Invalid STIX JSON: {e}")))?;

    let objects = bundle
        .get("objects")
        .and_then(|v| v.as_array())
        .ok_or_else(|| TensaError::ParseError("STIX bundle missing 'objects' array".into()))?;

    let mut narrative_id = default_narrative_id.to_string();
    let mut entities_created = 0;
    let mut situations_created = 0;
    let mut relationships_created = 0;
    let mut skipped = 0;

    // First pass: find campaign to set narrative_id.
    for obj in objects {
        if obj.get("type").and_then(|t| t.as_str()) == Some("campaign") {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                narrative_id = name.to_string();
            }
        }
    }

    // Second pass: import SDOs (entities + situations).
    for obj in objects {
        let stix_type = match obj.get("type").and_then(|t| t.as_str()) {
            Some(t) => t,
            None => {
                skipped += 1;
                continue;
            }
        };

        let stix_id = obj.get("id").and_then(|i| i.as_str()).unwrap_or("");

        match stix_type {
            "campaign" => continue,     // handled above
            "relationship" => continue, // handled in third pass
            "observed-data" | "sighting" => {
                let id = stix_id_to_uuid(stix_id).unwrap_or_else(Uuid::now_v7);
                let confidence = obj
                    .get("confidence")
                    .and_then(|c| c.as_u64())
                    .map(|c| stix_confidence_to_tensa(c as u32))
                    .unwrap_or(0.5);

                let content = obj
                    .get("x_tensa_content")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("description").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();

                let start = parse_stix_timestamp(obj.get("first_observed"));
                let end = parse_stix_timestamp(
                    obj.get("last_observed")
                        .or_else(|| obj.get("first_observed")),
                );

                let sit = Situation {
                    id,
                    properties: serde_json::Value::Null,
                    name: obj
                        .get("x_tensa_name")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    description: None,
                    temporal: AllenInterval {
                        start,
                        end,
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
                    raw_content: if content.is_empty() {
                        vec![]
                    } else {
                        vec![ContentBlock::text(&content)]
                    },
                    narrative_level: NarrativeLevel::Event,
                    discourse: None,
                    maturity: MaturityLevel::Candidate,
                    confidence,
                    confidence_breakdown: None,
                    extraction_method: ExtractionMethod::StructuredImport,
                    provenance: vec![],
                    narrative_id: Some(narrative_id.clone()),
                    source_chunk_id: None,
                    source_span: None,
                    synopsis: None,
                    manuscript_order: None,
                    parent_situation_id: None,
                    label: None,
                    status: None,
                    keywords: vec![],
                    created_at: parse_stix_timestamp(obj.get("created")).unwrap_or_else(Utc::now),
                    updated_at: parse_stix_timestamp(obj.get("modified")).unwrap_or_else(Utc::now),
                    deleted_at: None,
                    transaction_time: None,
                };
                hypergraph.create_situation(sit)?;
                situations_created += 1;
            }
            _ => {
                // Try mapping to an entity type.
                match stix_type_to_entity_type(stix_type) {
                    Some(entity_type) => {
                        let id = stix_id_to_uuid(stix_id).unwrap_or_else(Uuid::now_v7);
                        let confidence = obj
                            .get("confidence")
                            .and_then(|c| c.as_u64())
                            .map(|c| stix_confidence_to_tensa(c as u32))
                            .unwrap_or(0.5);

                        let name = obj
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("Unknown");

                        let mut props = serde_json::json!({"name": name});
                        if let Some(desc) = obj.get("description").and_then(|d| d.as_str()) {
                            props["description"] = serde_json::json!(desc);
                        }
                        if let Some(lat) = obj.get("latitude") {
                            props["latitude"] = lat.clone();
                        }
                        if let Some(lon) = obj.get("longitude") {
                            props["longitude"] = lon.clone();
                        }

                        let entity = Entity {
                            id,
                            entity_type,
                            properties: props,
                            beliefs: None,
                            embedding: None,
                            maturity: MaturityLevel::Candidate,
                            confidence,
                            confidence_breakdown: None,
                            provenance: vec![],
                            extraction_method: Some(ExtractionMethod::StructuredImport),
                            narrative_id: Some(narrative_id.clone()),
                            created_at: parse_stix_timestamp(obj.get("created"))
                                .unwrap_or_else(Utc::now),
                            updated_at: parse_stix_timestamp(obj.get("modified"))
                                .unwrap_or_else(Utc::now),
                            deleted_at: None,
                            transaction_time: None,
                        };
                        hypergraph.create_entity(entity)?;
                        entities_created += 1;
                    }
                    None => {
                        skipped += 1;
                    }
                }
            }
        }
    }

    // Third pass: import relationships.
    for obj in objects {
        if obj.get("type").and_then(|t| t.as_str()) != Some("relationship") {
            continue;
        }

        let rel_type = obj
            .get("relationship_type")
            .and_then(|r| r.as_str())
            .unwrap_or("related-to");

        let source_id = obj
            .get("source_ref")
            .and_then(|r| r.as_str())
            .and_then(stix_id_to_uuid);
        let target_id = obj
            .get("target_ref")
            .and_then(|r| r.as_str())
            .and_then(stix_id_to_uuid);

        let (source_id, target_id) = match (source_id, target_id) {
            (Some(s), Some(t)) => (s, t),
            _ => {
                skipped += 1;
                continue;
            }
        };

        // "causes" relationships become causal links.
        if rel_type == "causes" {
            let link = CausalLink {
                from_situation: source_id,
                to_situation: target_id,
                mechanism: obj
                    .get("x_tensa_mechanism")
                    .and_then(|m| m.as_str())
                    .filter(|s| !s.is_empty())
                    .map(String::from),
                strength: obj
                    .get("x_tensa_strength")
                    .and_then(|s| s.as_f64())
                    .map(|s| s as f32)
                    .unwrap_or(0.5),
                causal_type: CausalType::Contributing,
                maturity: MaturityLevel::Candidate,
            };
            let _ = hypergraph.add_causal_link(link);
            relationships_created += 1;
            continue;
        }

        // Other relationships → participations (entity → situation).
        let role = match rel_type {
            "attributed-to" => Role::Protagonist,
            "targets" => Role::Target,
            "uses" => Role::Instrument,
            _ => Role::Witness,
        };

        let participation = Participation {
            entity_id: source_id,
            situation_id: target_id,
            role,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        };
        let _ = hypergraph.add_participant(participation);
        relationships_created += 1;
    }

    Ok(StixImportReport {
        entities_created,
        situations_created,
        relationships_created,
        skipped,
        narrative_id: Some(narrative_id),
    })
}

/// Parse a STIX timestamp string to `DateTime<Utc>`.
fn parse_stix_timestamp(value: Option<&Value>) -> Option<DateTime<Utc>> {
    value
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn make_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn sample_bundle() -> String {
        serde_json::json!({
            "type": "bundle",
            "id": "bundle--test-001",
            "objects": [
                {
                    "type": "campaign",
                    "spec_version": "2.1",
                    "id": "campaign--11111111-1111-7111-8111-111111111111",
                    "name": "Operation Nightfall",
                    "created": "2025-01-01T00:00:00Z",
                    "modified": "2025-01-01T00:00:00Z"
                },
                {
                    "type": "threat-actor",
                    "spec_version": "2.1",
                    "id": "threat-actor--22222222-2222-7222-8222-222222222222",
                    "name": "APT-42",
                    "confidence": 85,
                    "created": "2025-01-01T00:00:00Z",
                    "modified": "2025-01-01T00:00:00Z"
                },
                {
                    "type": "observed-data",
                    "spec_version": "2.1",
                    "id": "observed-data--33333333-3333-7333-8333-333333333333",
                    "first_observed": "2025-06-15T10:00:00Z",
                    "last_observed": "2025-06-15T11:00:00Z",
                    "number_observed": 1,
                    "confidence": 70,
                    "x_tensa_content": "Suspicious network activity detected",
                    "created": "2025-06-15T12:00:00Z",
                    "modified": "2025-06-15T12:00:00Z"
                },
                {
                    "type": "relationship",
                    "spec_version": "2.1",
                    "id": "relationship--44444444-4444-7444-8444-444444444444",
                    "relationship_type": "attributed-to",
                    "source_ref": "threat-actor--22222222-2222-7222-8222-222222222222",
                    "target_ref": "observed-data--33333333-3333-7333-8333-333333333333",
                    "created": "2025-06-15T12:00:00Z",
                    "modified": "2025-06-15T12:00:00Z"
                }
            ]
        })
        .to_string()
    }

    #[test]
    fn test_stix_import_campaign() {
        let hg = make_hg();
        let report = import_stix_bundle(&hg, &sample_bundle(), "fallback").unwrap();
        assert_eq!(
            report.narrative_id.as_deref(),
            Some("Operation Nightfall"),
            "Campaign name should override default narrative_id"
        );
    }

    #[test]
    fn test_stix_import_threat_actor() {
        let hg = make_hg();
        let report = import_stix_bundle(&hg, &sample_bundle(), "test").unwrap();
        assert_eq!(report.entities_created, 1);
        let entities = hg
            .list_entities_by_narrative("Operation Nightfall")
            .unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, EntityType::Actor);
        assert!((entities[0].confidence - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_stix_import_relationship() {
        let hg = make_hg();
        let report = import_stix_bundle(&hg, &sample_bundle(), "test").unwrap();
        assert_eq!(report.relationships_created, 1);
    }

    #[test]
    fn test_stix_unknown_type_skipped() {
        let hg = make_hg();
        let bundle = serde_json::json!({
            "type": "bundle",
            "id": "bundle--skip",
            "objects": [
                {"type": "malware-analysis", "id": "malware-analysis--fake", "name": "test"},
                {"type": "grouping", "id": "grouping--fake", "name": "test"}
            ]
        })
        .to_string();
        let report = import_stix_bundle(&hg, &bundle, "skip-test").unwrap();
        assert_eq!(report.skipped, 2);
        assert_eq!(report.entities_created, 0);
    }

    #[test]
    fn test_stix_export_grammar() {
        // Verify the export format string "stix" parses correctly.
        let format: crate::export::ExportFormat = "stix".parse().unwrap();
        assert_eq!(format, crate::export::ExportFormat::Stix);
    }
}
