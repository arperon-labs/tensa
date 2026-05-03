//! Structured data import (Sprint P3.6 — F-CE2).
//!
//! Import entities, situations, participations, and causal links from
//! JSON or CSV payloads without LLM extraction. Uses confidence gating
//! for items with confidence < 1.0.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::gate::{ConfidenceGate, GateDecision};
use crate::ingestion::queue::{
    QueueItemStatus, QueueItemType, ValidationQueue, ValidationQueueItem,
};
use crate::types::*;

// ─── Import Types ──────────────────────────────────────────

/// Top-level structured import payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredImport {
    #[serde(default)]
    pub entities: Vec<ImportEntity>,
    #[serde(default)]
    pub situations: Vec<ImportSituation>,
    #[serde(default)]
    pub participations: Vec<ImportParticipation>,
    #[serde(default)]
    pub causal_links: Vec<ImportCausalLink>,
    #[serde(default)]
    pub narrative_id: Option<String>,
}

/// A simplified entity for import (lighter than full Entity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportEntity {
    pub name: String,
    pub entity_type: EntityType,
    #[serde(default = "default_properties")]
    pub properties: serde_json::Value,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub id: Option<Uuid>,
}

/// A simplified situation for import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSituation {
    pub description: String,
    #[serde(default)]
    pub temporal_marker: Option<String>,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default = "default_narrative_level")]
    pub narrative_level: NarrativeLevel,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// A participation reference (by entity name and situation index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportParticipation {
    pub entity_name: String,
    pub situation_index: usize,
    #[serde(default = "default_role")]
    pub role: Role,
    #[serde(default)]
    pub action: Option<String>,
}

/// A causal link reference (by situation indices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportCausalLink {
    pub from_situation_index: usize,
    pub to_situation_index: usize,
    #[serde(default)]
    pub mechanism: Option<String>,
    #[serde(default = "default_causal_type")]
    pub causal_type: CausalType,
    #[serde(default = "default_strength")]
    pub strength: f32,
}

fn default_properties() -> serde_json::Value {
    serde_json::json!({})
}
fn default_confidence() -> f32 {
    1.0
}
fn default_narrative_level() -> NarrativeLevel {
    NarrativeLevel::Scene
}
fn default_role() -> Role {
    Role::Witness
}
fn default_causal_type() -> CausalType {
    CausalType::Contributing
}
fn default_strength() -> f32 {
    0.5
}

/// Report from a structured import operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    pub entities_created: usize,
    pub situations_created: usize,
    pub participations_created: usize,
    pub causal_links_created: usize,
    pub items_queued: usize,
    pub items_rejected: usize,
    pub errors: Vec<String>,
    pub entity_id_map: HashMap<String, Uuid>,
    pub situation_id_map: HashMap<usize, Uuid>,
}

// ─── CSV Column Mapping ────────────────────────────────────

/// Request body for CSV import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvImportRequest {
    pub csv_data: String,
    pub mapping: CsvColumnMapping,
    #[serde(default)]
    pub narrative_id: Option<String>,
}

/// Column mapping configuration for CSV entity import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvColumnMapping {
    pub name_col: String,
    pub type_col: String,
    #[serde(default)]
    pub confidence_col: Option<String>,
    #[serde(default)]
    pub property_cols: Vec<String>,
}

// ─── Core Import Logic ─────────────────────────────────────

/// Process a structured import payload, applying confidence gating.
pub fn process_structured_import(
    import: &StructuredImport,
    hypergraph: &Hypergraph,
    gate: &ConfidenceGate,
    queue: &ValidationQueue,
) -> Result<ImportReport> {
    let mut report = ImportReport {
        entities_created: 0,
        situations_created: 0,
        participations_created: 0,
        causal_links_created: 0,
        items_queued: 0,
        items_rejected: 0,
        errors: vec![],
        entity_id_map: HashMap::new(),
        situation_id_map: HashMap::new(),
    };
    let now = Utc::now();

    // --- Phase 1: Entities ---
    for imp_ent in &import.entities {
        // Dedup by name within this import
        if report.entity_id_map.contains_key(&imp_ent.name) {
            continue;
        }

        match gate.decide(imp_ent.confidence) {
            GateDecision::AutoCommit => {
                let id = imp_ent.id.unwrap_or_else(Uuid::now_v7);
                let mut props = imp_ent.properties.clone();
                if let Some(obj) = props.as_object_mut() {
                    obj.entry("name".to_string())
                        .or_insert_with(|| serde_json::Value::String(imp_ent.name.clone()));
                }
                let entity = Entity {
                    id,
                    entity_type: imp_ent.entity_type.clone(),
                    properties: props,
                    beliefs: None,
                    embedding: None,
                    maturity: MaturityLevel::Candidate,
                    confidence: imp_ent.confidence,
                    confidence_breakdown: None,
                    provenance: vec![SourceReference {
                        source_type: "structured_import".into(),
                        source_id: None,
                        description: Some("Bulk JSON/CSV import".into()),
                        timestamp: now,
                        registered_source: None,
                    }],
                    extraction_method: Some(ExtractionMethod::StructuredImport),
                    narrative_id: import.narrative_id.clone(),
                    created_at: now,
                    updated_at: now,
                    deleted_at: None,
                    transaction_time: None,
                };
                match hypergraph.create_entity(entity) {
                    Ok(created_id) => {
                        report
                            .entity_id_map
                            .insert(imp_ent.name.clone(), created_id);
                        report.entities_created += 1;
                    }
                    Err(e) => report
                        .errors
                        .push(format!("Entity '{}': {e}", imp_ent.name)),
                }
            }
            GateDecision::QueueForReview => {
                let item = ValidationQueueItem {
                    id: Uuid::now_v7(),
                    item_type: QueueItemType::Entity,
                    extracted_data: serde_json::json!({
                        "name": imp_ent.name,
                        "entity_type": imp_ent.entity_type,
                        "properties": imp_ent.properties,
                    }),
                    source_text: format!("Structured import: entity '{}'", imp_ent.name),
                    chunk_id: 0,
                    source_chunk_id: None,
                    narrative_id: import.narrative_id.clone(),
                    confidence: imp_ent.confidence,
                    status: QueueItemStatus::Pending,
                    reviewer_notes: None,
                    created_at: now,
                    reviewed_at: None,
                };
                match queue.enqueue(item) {
                    Ok(_) => report.items_queued += 1,
                    Err(e) => report
                        .errors
                        .push(format!("Queue entity '{}': {e}", imp_ent.name)),
                }
            }
            GateDecision::Reject => {
                report.items_rejected += 1;
            }
        }
    }

    // --- Phase 2: Situations ---
    for (idx, imp_sit) in import.situations.iter().enumerate() {
        match gate.decide(imp_sit.confidence) {
            GateDecision::AutoCommit => {
                let id = Uuid::now_v7();
                let spatial = imp_sit.location.as_ref().map(|loc| SpatialAnchor {
                    latitude: None,
                    longitude: None,
                    precision: SpatialPrecision::Unknown,
                    location_entity: None,
                    location_name: Some(loc.clone()),
                    description: Some(loc.clone()),
                    geo_provenance: None,
                });
                let situation = Situation {
                    id,
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
                    spatial,
                    game_structure: None,
                    causes: vec![],
                    deterministic: None,
                    probabilistic: None,
                    embedding: None,
                    raw_content: vec![ContentBlock::text(&imp_sit.description)],
                    narrative_level: imp_sit.narrative_level,
                    discourse: None,
                    maturity: MaturityLevel::Candidate,
                    confidence: imp_sit.confidence,
                    confidence_breakdown: None,
                    extraction_method: ExtractionMethod::StructuredImport,
                    provenance: vec![SourceReference {
                        source_type: "structured_import".into(),
                        source_id: None,
                        description: Some("Bulk JSON/CSV import".into()),
                        timestamp: now,
                        registered_source: None,
                    }],
                    narrative_id: import.narrative_id.clone(),
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
                };
                match hypergraph.create_situation(situation) {
                    Ok(created_id) => {
                        report.situation_id_map.insert(idx, created_id);
                        report.situations_created += 1;
                    }
                    Err(e) => report
                        .errors
                        .push(format!("Situation [{idx}] '{}': {e}", imp_sit.description)),
                }
            }
            GateDecision::QueueForReview => {
                let item = ValidationQueueItem {
                    id: Uuid::now_v7(),
                    item_type: QueueItemType::Situation,
                    extracted_data: serde_json::json!({
                        "index": idx,
                        "description": imp_sit.description,
                    }),
                    source_text: format!("Structured import: situation '{}'", imp_sit.description),
                    chunk_id: 0,
                    source_chunk_id: None,
                    narrative_id: import.narrative_id.clone(),
                    confidence: imp_sit.confidence,
                    status: QueueItemStatus::Pending,
                    reviewer_notes: None,
                    created_at: now,
                    reviewed_at: None,
                };
                match queue.enqueue(item) {
                    Ok(_) => report.items_queued += 1,
                    Err(e) => report.errors.push(format!("Queue situation [{idx}]: {e}")),
                }
            }
            GateDecision::Reject => {
                report.items_rejected += 1;
            }
        }
    }

    // --- Phase 3: Participations ---
    for imp_part in &import.participations {
        let entity_id = match report.entity_id_map.get(&imp_part.entity_name) {
            Some(id) => *id,
            None => {
                report.errors.push(format!(
                    "Participation: entity '{}' not found in import",
                    imp_part.entity_name
                ));
                continue;
            }
        };
        let situation_id = match report.situation_id_map.get(&imp_part.situation_index) {
            Some(id) => *id,
            None => {
                report.errors.push(format!(
                    "Participation: situation index {} not found in import",
                    imp_part.situation_index
                ));
                continue;
            }
        };
        let participation = Participation {
            entity_id,
            situation_id,
            role: imp_part.role.clone(),
            info_set: None,
            action: imp_part.action.clone(),
            payoff: None,
            seq: 0,
        };
        match hypergraph.add_participant(participation) {
            Ok(()) => report.participations_created += 1,
            Err(e) => report.errors.push(format!(
                "Participation '{}' -> [{}]: {e}",
                imp_part.entity_name, imp_part.situation_index
            )),
        }
    }

    // --- Phase 4: Causal Links ---
    for imp_link in &import.causal_links {
        let from_id = match report.situation_id_map.get(&imp_link.from_situation_index) {
            Some(id) => *id,
            None => {
                report.errors.push(format!(
                    "Causal link: from_situation index {} not found",
                    imp_link.from_situation_index
                ));
                continue;
            }
        };
        let to_id = match report.situation_id_map.get(&imp_link.to_situation_index) {
            Some(id) => *id,
            None => {
                report.errors.push(format!(
                    "Causal link: to_situation index {} not found",
                    imp_link.to_situation_index
                ));
                continue;
            }
        };
        let link = CausalLink {
            from_situation: from_id,
            to_situation: to_id,
            mechanism: imp_link.mechanism.clone(),
            strength: imp_link.strength,
            causal_type: imp_link.causal_type,
            maturity: MaturityLevel::Candidate,
        };
        match hypergraph.add_causal_link(link) {
            Ok(()) => report.causal_links_created += 1,
            Err(e) => report.errors.push(format!(
                "Causal [{} -> {}]: {e}",
                imp_link.from_situation_index, imp_link.to_situation_index
            )),
        }
    }

    Ok(report)
}

// ─── CSV Parsing ───────────────────────────────────────────

/// Parse CSV text into a `StructuredImport` using column mapping.
pub fn parse_csv_to_import(
    csv_text: &str,
    mapping: &CsvColumnMapping,
    narrative_id: Option<String>,
) -> Result<StructuredImport> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_text.as_bytes());

    let headers = reader
        .headers()
        .map_err(|e| TensaError::ImportError(format!("CSV header error: {e}")))?
        .clone();

    let name_idx = headers
        .iter()
        .position(|h| h == mapping.name_col)
        .ok_or_else(|| {
            TensaError::ImportError(format!("Column '{}' not found in CSV", mapping.name_col))
        })?;
    let type_idx = headers
        .iter()
        .position(|h| h == mapping.type_col)
        .ok_or_else(|| {
            TensaError::ImportError(format!("Column '{}' not found in CSV", mapping.type_col))
        })?;
    let conf_idx = mapping
        .confidence_col
        .as_ref()
        .and_then(|col| headers.iter().position(|h| h == col.as_str()));
    let prop_indices: Vec<(usize, String)> = mapping
        .property_cols
        .iter()
        .filter_map(|col| {
            headers
                .iter()
                .position(|h| h == col.as_str())
                .map(|idx| (idx, col.clone()))
        })
        .collect();

    let mut entities = Vec::new();

    for result in reader.records() {
        let record =
            result.map_err(|e| TensaError::ImportError(format!("CSV record error: {e}")))?;

        let name = record.get(name_idx).unwrap_or_default().trim().to_string();
        if name.is_empty() {
            continue;
        }

        let type_str = record.get(type_idx).unwrap_or_default().trim().to_string();
        let entity_type: EntityType = serde_json::from_value(serde_json::Value::String(
            type_str.clone(),
        ))
        .unwrap_or_else(|_| {
            // Fallback: try case-insensitive match
            match type_str.to_lowercase().as_str() {
                "actor" | "" => EntityType::Actor,
                "location" => EntityType::Location,
                "artifact" => EntityType::Artifact,
                "concept" => EntityType::Concept,
                "organization" => EntityType::Organization,
                _ => EntityType::Actor, // default for CSV compatibility
            }
        });

        let confidence = conf_idx
            .and_then(|idx| record.get(idx))
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(1.0);

        let mut properties = serde_json::Map::new();
        for (idx, col_name) in &prop_indices {
            if let Some(val) = record.get(*idx) {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    properties.insert(col_name.clone(), serde_json::Value::String(trimmed.into()));
                }
            }
        }

        entities.push(ImportEntity {
            name,
            entity_type,
            properties: serde_json::Value::Object(properties),
            confidence,
            id: None,
        });
    }

    Ok(StructuredImport {
        entities,
        situations: vec![],
        participations: vec![],
        causal_links: vec![],
        narrative_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::Hypergraph;
    use crate::ingestion::gate::ConfidenceGate;
    use crate::ingestion::queue::ValidationQueue;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn setup() -> (Hypergraph, ConfidenceGate, ValidationQueue) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let gate = ConfidenceGate::default();
        let queue = ValidationQueue::new(store);
        (hg, gate, queue)
    }

    #[test]
    fn test_import_json_entities_only() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![
                ImportEntity {
                    name: "Alice".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"age": 30}),
                    confidence: 0.9,
                    id: None,
                },
                ImportEntity {
                    name: "Berlin".into(),
                    entity_type: EntityType::Location,
                    properties: serde_json::json!({}),
                    confidence: 0.85,
                    id: None,
                },
                ImportEntity {
                    name: "ACME".into(),
                    entity_type: EntityType::Organization,
                    properties: serde_json::json!({}),
                    confidence: 0.95,
                    id: None,
                },
            ],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 3);
        assert_eq!(report.entity_id_map.len(), 3);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_import_json_situations_only() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![],
            situations: vec![
                ImportSituation {
                    description: "Meeting at dawn".into(),
                    temporal_marker: None,
                    location: Some("Berlin".into()),
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.9,
                },
                ImportSituation {
                    description: "Chase through streets".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.85,
                },
                ImportSituation {
                    description: "Resolution".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.8,
                },
            ],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.situations_created, 3);
        assert_eq!(report.situation_id_map.len(), 3);
    }

    #[test]
    fn test_import_json_full() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![
                ImportEntity {
                    name: "Alice".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.9,
                    id: None,
                },
                ImportEntity {
                    name: "Bob".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.9,
                    id: None,
                },
            ],
            situations: vec![
                ImportSituation {
                    description: "Meeting".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.9,
                },
                ImportSituation {
                    description: "Conflict".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.9,
                },
            ],
            participations: vec![
                ImportParticipation {
                    entity_name: "Alice".into(),
                    situation_index: 0,
                    role: Role::Protagonist,
                    action: Some("greets".into()),
                },
                ImportParticipation {
                    entity_name: "Bob".into(),
                    situation_index: 0,
                    role: Role::Witness,
                    action: None,
                },
            ],
            causal_links: vec![ImportCausalLink {
                from_situation_index: 0,
                to_situation_index: 1,
                mechanism: Some("provocation".into()),
                causal_type: CausalType::Contributing,
                strength: 0.7,
            }],
            narrative_id: Some("test-story".into()),
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 2);
        assert_eq!(report.situations_created, 2);
        assert_eq!(report.participations_created, 2);
        assert_eq!(report.causal_links_created, 1);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn test_import_json_confidence_gating_auto_commit() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![ImportEntity {
                name: "High".into(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.95,
                id: None,
            }],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 1);
        assert_eq!(report.items_queued, 0);
        assert_eq!(report.items_rejected, 0);
    }

    #[test]
    fn test_import_json_confidence_gating_queue() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![ImportEntity {
                name: "Medium".into(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.5,
                id: None,
            }],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 0);
        assert_eq!(report.items_queued, 1);
    }

    #[test]
    fn test_import_json_confidence_gating_reject() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![ImportEntity {
                name: "Low".into(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.1,
                id: None,
            }],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 0);
        assert_eq!(report.items_rejected, 1);
    }

    #[test]
    fn test_import_json_mixed_confidence() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![
                ImportEntity {
                    name: "High".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.9,
                    id: None,
                },
                ImportEntity {
                    name: "Medium".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.5,
                    id: None,
                },
                ImportEntity {
                    name: "Low".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.1,
                    id: None,
                },
            ],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 1);
        assert_eq!(report.items_queued, 1);
        assert_eq!(report.items_rejected, 1);
    }

    #[test]
    fn test_import_json_entity_name_dedup() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![
                ImportEntity {
                    name: "Alice".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"version": 1}),
                    confidence: 0.9,
                    id: None,
                },
                ImportEntity {
                    name: "Alice".into(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"version": 2}),
                    confidence: 0.9,
                    id: None,
                },
            ],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.entities_created, 1);
        assert_eq!(report.entity_id_map.len(), 1);
    }

    #[test]
    fn test_import_json_invalid_participation_ref() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![],
            situations: vec![ImportSituation {
                description: "Scene".into(),
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Scene,
                confidence: 0.9,
            }],
            participations: vec![ImportParticipation {
                entity_name: "NonExistent".into(),
                situation_index: 0,
                role: Role::Witness,
                action: None,
            }],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.participations_created, 0);
        assert!(!report.errors.is_empty());
        assert!(report.errors[0].contains("not found"));
    }

    #[test]
    fn test_import_json_causal_link_cycle() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![],
            situations: vec![
                ImportSituation {
                    description: "A".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.9,
                },
                ImportSituation {
                    description: "B".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    confidence: 0.9,
                },
            ],
            // A -> B then B -> A should detect cycle
            causal_links: vec![
                ImportCausalLink {
                    from_situation_index: 0,
                    to_situation_index: 1,
                    mechanism: None,
                    causal_type: CausalType::Contributing,
                    strength: 0.5,
                },
                ImportCausalLink {
                    from_situation_index: 1,
                    to_situation_index: 0,
                    mechanism: None,
                    causal_type: CausalType::Contributing,
                    strength: 0.5,
                },
            ],
            participations: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        assert_eq!(report.causal_links_created, 1);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_import_json_extraction_method_set() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![ImportEntity {
                name: "Test".into(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.9,
                id: None,
            }],
            situations: vec![ImportSituation {
                description: "Scene".into(),
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Scene,
                confidence: 0.9,
            }],
            participations: vec![],
            causal_links: vec![],
            narrative_id: None,
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        let eid = report.entity_id_map["Test"];
        let entity = hg.get_entity(&eid).unwrap();
        assert_eq!(
            entity.extraction_method,
            Some(ExtractionMethod::StructuredImport)
        );

        let sid = report.situation_id_map[&0];
        let situation = hg.get_situation(&sid).unwrap();
        assert_eq!(
            situation.extraction_method,
            ExtractionMethod::StructuredImport
        );
    }

    #[test]
    fn test_import_json_narrative_id_propagated() {
        let (hg, gate, queue) = setup();
        let import = StructuredImport {
            entities: vec![ImportEntity {
                name: "Test".into(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.9,
                id: None,
            }],
            situations: vec![ImportSituation {
                description: "Scene".into(),
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Scene,
                confidence: 0.9,
            }],
            participations: vec![],
            causal_links: vec![],
            narrative_id: Some("my-narrative".into()),
        };
        let report = process_structured_import(&import, &hg, &gate, &queue).unwrap();
        let eid = report.entity_id_map["Test"];
        let entity = hg.get_entity(&eid).unwrap();
        assert_eq!(entity.narrative_id, Some("my-narrative".into()));

        let sid = report.situation_id_map[&0];
        let situation = hg.get_situation(&sid).unwrap();
        assert_eq!(situation.narrative_id, Some("my-narrative".into()));
    }

    #[test]
    fn test_csv_parse_basic() {
        let csv_data = "name,type,age\nAlice,Actor,30\nBerlin,Location,\n";
        let mapping = CsvColumnMapping {
            name_col: "name".into(),
            type_col: "type".into(),
            confidence_col: None,
            property_cols: vec!["age".into()],
        };
        let result = parse_csv_to_import(csv_data, &mapping, None).unwrap();
        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.entities[0].name, "Alice");
        assert_eq!(result.entities[0].entity_type, EntityType::Actor);
        assert_eq!(result.entities[1].name, "Berlin");
        assert_eq!(result.entities[1].entity_type, EntityType::Location);
    }

    #[test]
    fn test_csv_parse_with_mapping() {
        let csv_data =
            "entity_name,entity_type,conf,department\nAlice,Actor,0.75,Engineering\nBob,Actor,0.9,Sales\n";
        let mapping = CsvColumnMapping {
            name_col: "entity_name".into(),
            type_col: "entity_type".into(),
            confidence_col: Some("conf".into()),
            property_cols: vec!["department".into()],
        };
        let result = parse_csv_to_import(csv_data, &mapping, Some("test".into())).unwrap();
        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.entities[0].confidence, 0.75);
        assert_eq!(result.entities[1].confidence, 0.9);
        assert_eq!(result.narrative_id, Some("test".into()));
        let props = result.entities[0].properties.as_object().unwrap();
        assert_eq!(props["department"], "Engineering");
    }

    #[test]
    fn test_csv_parse_empty() {
        let csv_data = "name,type\n";
        let mapping = CsvColumnMapping {
            name_col: "name".into(),
            type_col: "type".into(),
            confidence_col: None,
            property_cols: vec![],
        };
        let result = parse_csv_to_import(csv_data, &mapping, None).unwrap();
        assert!(result.entities.is_empty());
    }
}
