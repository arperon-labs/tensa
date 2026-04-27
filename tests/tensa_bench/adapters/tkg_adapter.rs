//! TKG adapter: maps temporal triples (s, r, o, t) to TENSA's hypergraph.
//!
//! Each entity ID becomes a TENSA Entity; each triple becomes a Situation
//! with two Participations (subject + object). No LLM calls — direct CRUD.

use crate::tensa_bench::datasets::icews::{IcewsDataset, TemporalTriple};
use crate::tensa_bench::metrics::ranking::RankingAccumulator;
use chrono::{NaiveTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tensa::analysis::graph_projection::{build_co_graph, CoGraph};
use tensa::analysis::link_prediction;
use tensa::hypergraph::Hypergraph;
use tensa::store::memory::MemoryStore;
use tensa::types::*;
use uuid::Uuid;

/// Mapping from TKG entity IDs to TENSA UUIDs.
pub struct TkgMapping {
    pub entity_map: HashMap<u32, Uuid>,
    pub relation_names: HashMap<u32, String>,
    pub entity_names: HashMap<u32, String>,
    pub narrative_id: String,
}

/// Ingest all train triples into a fresh TENSA hypergraph.
///
/// Returns the hypergraph and the entity ID mapping.
pub fn ingest_tkg_triples(
    dataset: &IcewsDataset,
    train_triples: &[TemporalTriple],
    narrative_id: &str,
) -> Result<(Hypergraph, TkgMapping), String> {
    let store: Arc<dyn tensa::store::KVStore> = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);

    let mut entity_map: HashMap<u32, Uuid> = HashMap::with_capacity(dataset.entity_names.len());

    for (&eid, name_ref) in &dataset.entity_names {
        let name = name_ref.clone();

        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Concept,
            properties: serde_json::json!({
                "name": name,
                "tkg_id": eid,
            }),
            beliefs: None,
            embedding: None,
            narrative_id: Some(narrative_id.to_string()),
            maturity: MaturityLevel::Validated,
            confidence: 1.0,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::StructuredImport),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };

        let uuid = hg
            .create_entity(entity)
            .map_err(|e| format!("Failed to create entity {}: {}", eid, e))?;
        entity_map.insert(eid, uuid);
    }

    // Create situations from train triples
    for triple in train_triples {
        ingest_single_triple(
            &hg,
            triple,
            &entity_map,
            &dataset.relation_names,
            narrative_id,
        )?;
    }

    let mapping = TkgMapping {
        entity_map,
        relation_names: dataset.relation_names.clone(),
        entity_names: dataset.entity_names.clone(),
        narrative_id: narrative_id.to_string(),
    };

    Ok((hg, mapping))
}

/// Ingest a single temporal triple as a Situation + 2 Participations.
fn ingest_single_triple(
    hg: &Hypergraph,
    triple: &TemporalTriple,
    entity_map: &HashMap<u32, Uuid>,
    relation_names: &HashMap<u32, String>,
    narrative_id: &str,
) -> Result<Uuid, String> {
    let subject_uuid = entity_map
        .get(&triple.subject)
        .ok_or_else(|| format!("Unknown subject entity: {}", triple.subject))?;
    let object_uuid = entity_map
        .get(&triple.object)
        .ok_or_else(|| format!("Unknown object entity: {}", triple.object))?;

    let relation_name = relation_names
        .get(&triple.relation)
        .cloned()
        .unwrap_or_else(|| format!("relation_{}", triple.relation));

    let start_dt = triple
        .timestamp
        .and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap())
        .and_utc();
    let end_dt = triple
        .timestamp
        .and_time(NaiveTime::from_hms_opt(23, 59, 59).unwrap())
        .and_utc();

    let situation = Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: Some(start_dt),
            end: Some(end_dt),
            granularity: TimeGranularity::Day,
            relations: vec![],
            fuzzy_endpoints: None,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: Some(serde_json::json!({
            "relation_id": triple.relation,
            "relation_name": relation_name,
            "subject_id": triple.subject,
            "object_id": triple.object,
        })),
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text(&format!(
            "{} → {} → {}",
            triple.subject, relation_name, triple.object
        ))],
        narrative_level: NarrativeLevel::Event,
        narrative_id: Some(narrative_id.to_string()),
        discourse: None,
        maturity: MaturityLevel::Validated,
        confidence: 1.0,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::StructuredImport,
        provenance: vec![],
        source_span: None,
        source_chunk_id: None,
        synopsis: None,
        manuscript_order: None,
        parent_situation_id: None,
        label: None,
        status: None,
        keywords: vec![],
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };

    let sit_id = hg
        .create_situation(situation)
        .map_err(|e| format!("Failed to create situation: {}", e))?;

    hg.add_participant(Participation {
        entity_id: *subject_uuid,
        situation_id: sit_id,
        seq: 0,
        role: Role::Protagonist,
        info_set: None,
        action: Some("subject".to_string()),
        payoff: None,
    })
    .map_err(|e| format!("Failed to add subject participation: {}", e))?;

    hg.add_participant(Participation {
        entity_id: *object_uuid,
        situation_id: sit_id,
        seq: 0,
        role: Role::Target,
        info_set: None,
        action: Some("object".to_string()),
        payoff: None,
    })
    .map_err(|e| format!("Failed to add object participation: {}", e))?;

    Ok(sit_id)
}

/// Score candidates for a link prediction query (s, r, ?, t).
///
/// `uuid_to_idx`: precomputed Uuid→co_graph index lookup (build once per evaluation).
/// Returns scores for all candidate entities. Higher score = more likely.
pub fn score_candidates_object(
    co_graph: &CoGraph,
    mapping: &TkgMapping,
    uuid_to_idx: &HashMap<Uuid, usize>,
    subject: u32,
) -> Vec<(u32, f64)> {
    let subject_uuid = match mapping.entity_map.get(&subject) {
        Some(u) => *u,
        None => return vec![],
    };
    let subject_idx = match uuid_to_idx.get(&subject_uuid) {
        Some(&i) => i,
        None => return vec![],
    };

    let mut scores: Vec<(u32, f64)> = Vec::with_capacity(mapping.entity_map.len());

    for (&eid, &uuid) in &mapping.entity_map {
        if eid == subject {
            continue;
        }
        let candidate_idx = match uuid_to_idx.get(&uuid) {
            Some(&i) => i,
            None => {
                scores.push((eid, 0.0));
                continue;
            }
        };

        let aa = link_prediction::adamic_adar(co_graph, subject_idx, candidate_idx);
        let cn = link_prediction::common_neighbors(co_graph, subject_idx, candidate_idx) as f64;
        let ra = link_prediction::resource_allocation(co_graph, subject_idx, candidate_idx);
        let score = aa * 0.4 + cn * 0.3 + ra * 0.3;
        scores.push((eid, score));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Run link prediction evaluation on test triples.
///
/// For each test triple (s, r, ?, t), rank all entities and record the rank
/// of the correct object. Returns a RankingAccumulator with the results.
pub fn evaluate_link_prediction(
    hg: &Hypergraph,
    mapping: &TkgMapping,
    test_triples: &[TemporalTriple],
) -> Result<RankingAccumulator, String> {
    let co_graph = build_co_graph(hg, &mapping.narrative_id)
        .map_err(|e| format!("Failed to build co-graph: {}", e))?;

    // Precompute Uuid→co_graph index; scoring per triple is then O(V) instead of O(V²).
    let uuid_to_idx: HashMap<Uuid, usize> = co_graph
        .entities
        .iter()
        .enumerate()
        .map(|(i, &u)| (u, i))
        .collect();

    let mut acc = RankingAccumulator::new();

    for triple in test_triples {
        let scores = score_candidates_object(&co_graph, mapping, &uuid_to_idx, triple.subject);

        let rank = scores
            .iter()
            .position(|(eid, _)| *eid == triple.object)
            .map(|pos| pos + 1)
            .unwrap_or(scores.len() + 1);

        acc.add(rank);
    }

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_small_graph() {
        let dataset = IcewsDataset {
            triples: vec![],
            entity_names: {
                let mut m = HashMap::new();
                m.insert(0, "USA".to_string());
                m.insert(1, "Russia".to_string());
                m.insert(2, "China".to_string());
                m
            },
            relation_names: {
                let mut m = HashMap::new();
                m.insert(0, "cooperate".to_string());
                m.insert(1, "conflict".to_string());
                m
            },
        };

        let train = vec![
            TemporalTriple {
                subject: 0,
                relation: 0,
                object: 1,
                timestamp: chrono::NaiveDate::from_ymd_opt(2014, 1, 1).unwrap(),
            },
            TemporalTriple {
                subject: 1,
                relation: 1,
                object: 2,
                timestamp: chrono::NaiveDate::from_ymd_opt(2014, 1, 2).unwrap(),
            },
        ];

        let (hg, mapping) = ingest_tkg_triples(&dataset, &train, "test-tkg").unwrap();
        assert_eq!(mapping.entity_map.len(), 3);

        // Verify entities exist
        for &uuid in mapping.entity_map.values() {
            assert!(hg.get_entity(&uuid).is_ok());
        }
    }
}
