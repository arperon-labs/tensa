//! Temporal Correlation Graph (TCG) anomaly detection.
//!
//! Builds three distinct views of narrative structure and detects anomalies
//! visible only across multiple views:
//!
//! - **Similarity Graph**: entities connected by co-occurrence in situations
//! - **Causality Graph**: entities connected via causal link chains
//! - **Synchronization Graph**: entities whose situations overlap temporally
//!
//! Cross-view anomalies:
//! - Present in Similarity but absent from Causality → planted information
//! - Present in Synchronization but absent from Similarity → coordinated but independent
//! - Present in Causality but absent from Synchronization → delayed causal effects

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::KVStore;
use crate::types::*;

use super::extract_narrative_id;

/// KV prefix for persisted TCG anomaly reports.
pub const TCG_PREFIX: &str = "an/tcg/";

/// Edge in a correlation graph (undirected, weighted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEdge {
    pub entity_a: Uuid,
    pub entity_b: Uuid,
    pub weight: f64,
}

/// A TCG anomaly flagging an entity pair present in one view but absent from another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCGAnomaly {
    pub entity_a: Uuid,
    pub entity_b: Uuid,
    pub present_in: String,
    pub absent_from: String,
    pub anomaly_type: String,
    pub similarity_weight: f64,
    pub causality_weight: f64,
    pub synchronization_weight: f64,
}

/// Full TCG anomaly report for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCGReport {
    pub narrative_id: String,
    pub similarity_edges: usize,
    pub causality_edges: usize,
    pub synchronization_edges: usize,
    pub anomalies: Vec<TCGAnomaly>,
}

/// TCG anomaly detection engine.
pub struct TCGAnomalyEngine;

impl InferenceEngine for TCGAnomalyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::TCGAnomaly
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000) // 5 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let report = detect_tcg_anomalies(narrative_id, hypergraph)?;

        // Persist report
        persist_tcg_report(hypergraph.store(), &report)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::TCGAnomaly,
            target_id: job.target_id,
            result: serde_json::to_value(&report)?,
            confidence: 1.0,
            explanation: Some(format!(
                "TCG analysis: {} anomalies across {} similarity, {} causality, {} synchronization edges",
                report.anomalies.len(),
                report.similarity_edges,
                report.causality_edges,
                report.synchronization_edges,
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Type alias for an edge set keyed by sorted entity pair.
type EdgeMap = HashMap<(Uuid, Uuid), f64>;

/// Normalize an entity pair so (a, b) and (b, a) map to the same key.
fn edge_key(a: Uuid, b: Uuid) -> (Uuid, Uuid) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Build the three correlation graphs and detect cross-view anomalies.
fn detect_tcg_anomalies(narrative_id: &str, hypergraph: &Hypergraph) -> Result<TCGReport> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    situations.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    if entities.len() < 2 {
        return Ok(TCGReport {
            narrative_id: narrative_id.to_string(),
            similarity_edges: 0,
            causality_edges: 0,
            synchronization_edges: 0,
            anomalies: vec![],
        });
    }

    // ── 1. Similarity Graph: entity co-occurrence in situations ──
    let similarity = build_similarity_graph(&situations, hypergraph);

    // ── 2. Causality Graph: entities linked via causal chains ──
    let causality = build_causality_graph(&situations, hypergraph);

    // ── 3. Synchronization Graph: entities with temporally overlapping situations ──
    let synchronization = build_synchronization_graph(&entities, &situations, hypergraph);

    // ── Cross-view anomaly detection ──
    let sim_set: HashSet<(Uuid, Uuid)> = similarity.keys().copied().collect();
    let caus_set: HashSet<(Uuid, Uuid)> = causality.keys().copied().collect();
    let sync_set: HashSet<(Uuid, Uuid)> = synchronization.keys().copied().collect();

    let mut anomalies = Vec::new();

    // Similarity without Causality → planted information
    for &(a, b) in sim_set.difference(&caus_set) {
        if !sync_set.contains(&(a, b)) {
            // Only flag if also not synchronized (truly disconnected)
            anomalies.push(TCGAnomaly {
                entity_a: a,
                entity_b: b,
                present_in: "similarity".to_string(),
                absent_from: "causality".to_string(),
                anomaly_type: "planted_information".to_string(),
                similarity_weight: *similarity.get(&(a, b)).unwrap_or(&0.0),
                causality_weight: 0.0,
                synchronization_weight: *synchronization.get(&(a, b)).unwrap_or(&0.0),
            });
        }
    }

    // Synchronization without Similarity → coordinated but independent
    for &(a, b) in sync_set.difference(&sim_set) {
        anomalies.push(TCGAnomaly {
            entity_a: a,
            entity_b: b,
            present_in: "synchronization".to_string(),
            absent_from: "similarity".to_string(),
            anomaly_type: "coordinated_independent".to_string(),
            similarity_weight: 0.0,
            causality_weight: *causality.get(&(a, b)).unwrap_or(&0.0),
            synchronization_weight: *synchronization.get(&(a, b)).unwrap_or(&0.0),
        });
    }

    // Causality without Synchronization → delayed causal effects
    for &(a, b) in caus_set.difference(&sync_set) {
        anomalies.push(TCGAnomaly {
            entity_a: a,
            entity_b: b,
            present_in: "causality".to_string(),
            absent_from: "synchronization".to_string(),
            anomaly_type: "delayed_causal".to_string(),
            similarity_weight: *similarity.get(&(a, b)).unwrap_or(&0.0),
            causality_weight: *causality.get(&(a, b)).unwrap_or(&0.0),
            synchronization_weight: 0.0,
        });
    }

    Ok(TCGReport {
        narrative_id: narrative_id.to_string(),
        similarity_edges: similarity.len(),
        causality_edges: causality.len(),
        synchronization_edges: synchronization.len(),
        anomalies,
    })
}

/// Build Similarity Graph: entities connected by co-occurrence in situations.
fn build_similarity_graph(situations: &[Situation], hypergraph: &Hypergraph) -> EdgeMap {
    let mut edges: EdgeMap = HashMap::new();

    for sit in situations {
        let participants = hypergraph
            .get_participants_for_situation(&sit.id)
            .unwrap_or_default();
        let entity_ids: Vec<Uuid> = participants.iter().map(|p| p.entity_id).collect();

        // All pairs of entities in the same situation
        for i in 0..entity_ids.len() {
            for j in (i + 1)..entity_ids.len() {
                let key = edge_key(entity_ids[i], entity_ids[j]);
                *edges.entry(key).or_default() += 1.0;
            }
        }
    }

    edges
}

/// Build Causality Graph: entities connected via causal link chains.
fn build_causality_graph(situations: &[Situation], hypergraph: &Hypergraph) -> EdgeMap {
    let mut edges: EdgeMap = HashMap::new();

    // For each causal link, find entities in both situations
    for sit in situations {
        let forward_links = hypergraph.get_consequences(&sit.id).unwrap_or_default();

        for link in &forward_links {
            let from_participants = hypergraph
                .get_participants_for_situation(&link.from_situation)
                .unwrap_or_default();
            let to_participants = hypergraph
                .get_participants_for_situation(&link.to_situation)
                .unwrap_or_default();

            for fp in &from_participants {
                for tp in &to_participants {
                    if fp.entity_id != tp.entity_id {
                        let key = edge_key(fp.entity_id, tp.entity_id);
                        *edges.entry(key).or_default() += link.strength as f64;
                    }
                }
            }
        }
    }

    edges
}

/// Build Synchronization Graph: entities whose situations overlap temporally.
fn build_synchronization_graph(
    _entities: &[Entity],
    situations: &[Situation],
    hypergraph: &Hypergraph,
) -> EdgeMap {
    let mut edges: EdgeMap = HashMap::new();

    // For each pair of situations that overlap temporally,
    // connect entities from situation A to entities from situation B
    for i in 0..situations.len() {
        let _start_i = situations[i].temporal.start.unwrap_or_default();
        let end_i = situations[i]
            .temporal
            .end
            .unwrap_or(situations[i].temporal.start.unwrap_or_default());

        for j in (i + 1)..situations.len() {
            let start_j = situations[j].temporal.start.unwrap_or_default();
            // Early termination: situations are sorted by start time,
            // so if start_j > end_i, no more overlaps possible
            if start_j > end_i {
                break;
            }

            // Overlapping: connect entities across situations
            let parts_i = hypergraph
                .get_participants_for_situation(&situations[i].id)
                .unwrap_or_default();
            let parts_j = hypergraph
                .get_participants_for_situation(&situations[j].id)
                .unwrap_or_default();

            for pi in &parts_i {
                for pj in &parts_j {
                    if pi.entity_id != pj.entity_id {
                        let key = edge_key(pi.entity_id, pj.entity_id);
                        *edges.entry(key).or_default() += 1.0;
                    }
                }
            }
        }
    }

    edges
}

/// Persist a TCG report to KV.
pub fn persist_tcg_report(store: &dyn KVStore, report: &TCGReport) -> Result<()> {
    let key = format!("{}{}", TCG_PREFIX, report.narrative_id);
    let value = serde_json::to_vec(report)?;
    store.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Load a persisted TCG report from KV.
pub fn get_tcg_report(store: &dyn KVStore, narrative_id: &str) -> Result<Option<TCGReport>> {
    let key = format!("{}{}", TCG_PREFIX, narrative_id);
    match store.get(key.as_bytes())? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_tcg_empty_narrative() {
        let hg = make_hg();
        let report = detect_tcg_anomalies("empty", &hg).unwrap();
        assert!(report.anomalies.is_empty());
        assert_eq!(report.similarity_edges, 0);
    }

    #[test]
    fn test_tcg_similarity_graph() {
        let hg = make_hg();
        let a = add_entity(&hg, "Alice", "tcg-test");
        let b = add_entity(&hg, "Bob", "tcg-test");
        let s1 = add_situation(&hg, "tcg-test");
        link(&hg, a, s1);
        link(&hg, b, s1);

        let sim =
            build_similarity_graph(&hg.list_situations_by_narrative("tcg-test").unwrap(), &hg);
        assert_eq!(sim.len(), 1);
        let key = edge_key(a, b);
        assert_eq!(*sim.get(&key).unwrap(), 1.0);
    }

    #[test]
    fn test_tcg_causality_graph() {
        let hg = make_hg();
        let a = add_entity(&hg, "Alice", "tcg-test");
        let b = add_entity(&hg, "Bob", "tcg-test");
        let s1 = add_situation(&hg, "tcg-test");
        let s2 = add_situation(&hg, "tcg-test");
        link(&hg, a, s1);
        link(&hg, b, s2);
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            causal_type: CausalType::Contributing,
            strength: 0.8,
            mechanism: None,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let caus =
            build_causality_graph(&hg.list_situations_by_narrative("tcg-test").unwrap(), &hg);
        assert_eq!(caus.len(), 1);
    }

    #[test]
    fn test_tcg_cross_view_anomaly() {
        let hg = make_hg();
        let nid = "tcg-cross";
        let a = add_entity(&hg, "Alice", nid);
        let b = add_entity(&hg, "Bob", nid);
        let c = add_entity(&hg, "Charlie", nid);

        // Alice and Bob co-occur (similarity) but no causal link
        let s1 = add_situation(&hg, nid);
        link(&hg, a, s1);
        link(&hg, b, s1);

        // Charlie and Bob are causally linked but don't co-occur
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);
        link(&hg, c, s2);
        link(&hg, b, s3);
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            causal_type: CausalType::Necessary,
            strength: 0.9,
            mechanism: None,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let report = detect_tcg_anomalies(nid, &hg).unwrap();
        // Should find anomalies (cross-view mismatches)
        assert!(
            !report.anomalies.is_empty(),
            "Expected cross-view anomalies, got {}",
            report.anomalies.len()
        );
    }

    #[test]
    fn test_tcg_engine_execute() {
        let hg = make_hg();
        let nid = "tcg-eng";
        let a = add_entity(&hg, "Alice", nid);
        let b = add_entity(&hg, "Bob", nid);
        let s1 = add_situation(&hg, nid);
        link(&hg, a, s1);
        link(&hg, b, s1);

        let engine = TCGAnomalyEngine;
        assert_eq!(engine.job_type(), InferenceJobType::TCGAnomaly);

        let job = InferenceJob {
            id: "tcg-001".to_string(),
            job_type: InferenceJobType::TCGAnomaly,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": nid}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.explanation.is_some());
    }

    #[test]
    fn test_tcg_persistence() {
        let store = std::sync::Arc::new(crate::store::memory::MemoryStore::new());
        let report = TCGReport {
            narrative_id: "persist-test".to_string(),
            similarity_edges: 5,
            causality_edges: 3,
            synchronization_edges: 2,
            anomalies: vec![TCGAnomaly {
                entity_a: Uuid::now_v7(),
                entity_b: Uuid::now_v7(),
                present_in: "similarity".to_string(),
                absent_from: "causality".to_string(),
                anomaly_type: "planted_information".to_string(),
                similarity_weight: 2.0,
                causality_weight: 0.0,
                synchronization_weight: 0.0,
            }],
        };
        persist_tcg_report(store.as_ref(), &report).unwrap();
        let loaded = get_tcg_report(store.as_ref(), "persist-test")
            .unwrap()
            .expect("should exist");
        assert_eq!(loaded.anomalies.len(), 1);
        assert_eq!(loaded.anomalies[0].anomaly_type, "planted_information");
    }

    #[test]
    fn test_edge_key_symmetry() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        assert_eq!(edge_key(a, b), edge_key(b, a));
    }
}
