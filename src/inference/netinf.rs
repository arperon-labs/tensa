//! NetInf — infer hidden diffusion networks from observed information cascades.
//!
//! Implements a simplified Gomez-Rodriguez algorithm: given contagion traces
//! (from `analysis::contagion`), greedily add edges that maximize cascade
//! likelihood via submodular optimization.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id, load_sorted_situations};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

const NETINF_PREFIX: &[u8] = b"an/ni/";

/// An inferred diffusion edge between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredEdge {
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    /// Marginal likelihood gain from adding this edge.
    pub weight: f64,
}

/// Result of NetInf inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetInfResult {
    pub narrative_id: String,
    pub edges: Vec<InferredEdge>,
    pub num_cascades: usize,
}

/// A single cascade trace: ordered list of (entity_id, time_step).
/// Entities appear in the order they were "infected" (learned the fact).
#[derive(Debug, Clone)]
pub struct CascadeTrace {
    pub events: Vec<(Uuid, usize)>,
}

/// Extract cascade traces from the hypergraph contagion data.
///
/// Each fact that spreads through the narrative generates a cascade.
/// The cascade is the sequence of entities learning the fact, ordered
/// by the situation they learned it in.
pub fn extract_cascades(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Vec<CascadeTrace>> {
    let sorted = load_sorted_situations(hypergraph, narrative_id)?;

    // Track which facts each entity learns, by situation order.
    let mut fact_events: HashMap<String, Vec<(Uuid, usize)>> = HashMap::new();

    for (time_step, sit) in sorted.iter().enumerate() {
        let participants = hypergraph.get_participants_for_situation(&sit.id)?;
        for p in &participants {
            if let Some(info) = &p.info_set {
                for kf in &info.learns {
                    let key = format!("{}:{}", kf.about_entity, kf.fact);
                    fact_events
                        .entry(key)
                        .or_default()
                        .push((p.entity_id, time_step));
                }
            }
        }
    }

    // Each fact's learning sequence = a cascade.
    let cascades: Vec<CascadeTrace> = fact_events
        .into_values()
        .filter(|events| events.len() >= 2) // need at least 2 nodes for a cascade
        .map(|mut events| {
            events.sort_by_key(|&(_, t)| t);
            CascadeTrace { events }
        })
        .collect();

    Ok(cascades)
}

/// Compute the likelihood improvement from adding edge (u → v) given cascades.
///
/// For each cascade where u appears before v, the edge explains v's infection.
/// Score = number of cascades where u precedes v, weighted by time proximity.
fn edge_marginal_gain(
    u: Uuid,
    v: Uuid,
    cascades: &[CascadeTrace],
    existing_parents: &HashMap<Uuid, HashSet<Uuid>>,
) -> f64 {
    let mut gain = 0.0;
    let v_parents = existing_parents.get(&v);

    for cascade in cascades {
        let u_pos = cascade.events.iter().position(|&(eid, _)| eid == u);
        let v_pos = cascade.events.iter().position(|&(eid, _)| eid == v);

        if let (Some(up), Some(vp)) = (u_pos, v_pos) {
            if up < vp {
                // u appears before v → potential parent.
                // Only count if v doesn't already have a parent in this cascade.
                let has_existing = v_parents.map_or(false, |parents| {
                    cascade.events[..vp]
                        .iter()
                        .any(|&(eid, _)| parents.contains(&eid))
                });
                if !has_existing {
                    // Weight by inverse time difference (closer = stronger signal).
                    let dt = (vp - up) as f64;
                    gain += 1.0 / (1.0 + dt);
                }
            }
        }
    }
    gain
}

/// Run the NetInf algorithm: greedily infer a diffusion network from cascades.
///
/// `max_edges` limits the number of inferred edges.
pub fn run_netinf(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    max_edges: usize,
) -> Result<NetInfResult> {
    let cascades = extract_cascades(hypergraph, narrative_id)?;
    if cascades.is_empty() {
        return Ok(NetInfResult {
            narrative_id: narrative_id.to_string(),
            edges: vec![],
            num_cascades: 0,
        });
    }

    // Collect all entities that appear in cascades.
    let mut entities: HashSet<Uuid> = HashSet::new();
    for c in &cascades {
        for &(eid, _) in &c.events {
            entities.insert(eid);
        }
    }
    let mut entity_vec: Vec<Uuid> = entities.into_iter().collect();
    entity_vec.sort();

    // Greedy edge addition.
    let mut inferred_edges: Vec<InferredEdge> = Vec::new();
    let mut existing_parents: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();

    for _ in 0..max_edges {
        let mut best_edge: Option<(Uuid, Uuid, f64)> = None;

        for &u in &entity_vec {
            for &v in &entity_vec {
                if u == v {
                    continue;
                }
                // Skip if edge already exists.
                if existing_parents.get(&v).map_or(false, |p| p.contains(&u)) {
                    continue;
                }

                let gain = edge_marginal_gain(u, v, &cascades, &existing_parents);
                if gain > 0.0 {
                    if best_edge.map_or(true, |(_, _, g)| gain > g) {
                        best_edge = Some((u, v, gain));
                    }
                }
            }
        }

        match best_edge {
            Some((u, v, weight)) => {
                existing_parents.entry(v).or_default().insert(u);
                inferred_edges.push(InferredEdge {
                    from_entity: u,
                    to_entity: v,
                    weight,
                });
            }
            None => break, // no more beneficial edges
        }
    }

    // Store result.
    let result = NetInfResult {
        narrative_id: narrative_id.to_string(),
        edges: inferred_edges,
        num_cascades: cascades.len(),
    };
    let key = analysis_key(NETINF_PREFIX, &[narrative_id]);
    let bytes = serde_json::to_vec(&result)?;
    hypergraph.store().put(&key, &bytes)?;

    Ok(result)
}

// ─── InferenceEngine ───────────────────────────────────────

/// NetInf inference engine.
pub struct NetInfEngine;

impl InferenceEngine for NetInfEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::NetworkInference
    }
    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(15000)
    }
    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let max_edges = job
            .parameters
            .get("max_edges")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let result = run_netinf(hg, narrative_id, max_edges)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::NetworkInference,
            target_id: job.target_id,
            result: serde_json::to_value(&result)?,
            confidence: 1.0,
            explanation: Some(format!(
                "NetInf: {} edges inferred from {} cascades",
                result.edges.len(),
                result.num_cascades
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;
    use chrono::Duration;

    fn kf(about: Uuid, fact: &str) -> KnowledgeFact {
        KnowledgeFact {
            about_entity: about,
            fact: fact.to_string(),
            confidence: 1.0,
        }
    }

    /// Create a situation with a specific time offset (hours from a base).
    fn add_timed_sit(hg: &Hypergraph, narrative: &str, hour: i64) -> Uuid {
        let base = chrono::Utc::now() - Duration::hours(100);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(base + Duration::hours(hour)),
                end: Some(base + Duration::hours(hour + 1)),
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
            raw_content: vec![ContentBlock::text("test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_netinf_chain() {
        // A → B → C cascade: A learns first, B second, C third.
        let hg = make_hg();
        let n = "ni_chain";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let c = add_entity(&hg, "C", n);
        let target = uuid::Uuid::now_v7();

        let s1 = add_timed_sit(&hg, n, 0);
        hg.add_participant(Participation {
            entity_id: a,
            situation_id: s1,
            role: Role::Protagonist,
            info_set: Some(InfoSet {
                knows_before: vec![],
                learns: vec![kf(target, "secret")],
                reveals: vec![],
                beliefs_about_others: vec![],
            }),
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let s2 = add_timed_sit(&hg, n, 10);
        hg.add_participant(Participation {
            entity_id: b,
            situation_id: s2,
            role: Role::Protagonist,
            info_set: Some(InfoSet {
                knows_before: vec![],
                learns: vec![kf(target, "secret")],
                reveals: vec![],
                beliefs_about_others: vec![],
            }),
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let s3 = add_timed_sit(&hg, n, 20);
        hg.add_participant(Participation {
            entity_id: c,
            situation_id: s3,
            role: Role::Protagonist,
            info_set: Some(InfoSet {
                knows_before: vec![],
                learns: vec![kf(target, "secret")],
                reveals: vec![],
                beliefs_about_others: vec![],
            }),
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let result = run_netinf(&hg, n, 10).unwrap();
        assert_eq!(result.num_cascades, 1);
        assert!(!result.edges.is_empty(), "Should infer at least one edge");
        // A should appear as a source in inferred edges (A learned first).
        let a_is_source = result.edges.iter().any(|e| e.from_entity == a);
        assert!(a_is_source, "A should be inferred as a source entity");
    }

    #[test]
    fn test_netinf_star() {
        // A tells B and C simultaneously (star pattern).
        let hg = make_hg();
        let n = "ni_star";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let c = add_entity(&hg, "C", n);
        let target = uuid::Uuid::now_v7();

        let s1 = add_timed_sit(&hg, n, 0);
        hg.add_participant(Participation {
            entity_id: a,
            situation_id: s1,
            role: Role::Protagonist,
            info_set: Some(InfoSet {
                knows_before: vec![],
                learns: vec![kf(target, "info")],
                reveals: vec![],
                beliefs_about_others: vec![],
            }),
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let s2 = add_timed_sit(&hg, n, 10);
        for eid in [b, c] {
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: s2,
                role: Role::Recipient,
                info_set: Some(InfoSet {
                    knows_before: vec![],
                    learns: vec![kf(target, "info")],
                    reveals: vec![],
                    beliefs_about_others: vec![],
                }),
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let result = run_netinf(&hg, n, 10).unwrap();
        assert_eq!(result.num_cascades, 1);
        // Should infer A as parent (A learned first, B and C learned later).
        assert!(!result.edges.is_empty(), "Should infer edges");
        let a_edges: Vec<_> = result.edges.iter().filter(|e| e.from_entity == a).collect();
        assert!(!a_edges.is_empty(), "A should be inferred as a source");
    }

    #[test]
    fn test_netinf_empty() {
        let hg = make_hg();
        let result = run_netinf(&hg, "empty", 10).unwrap();
        assert!(result.edges.is_empty());
        assert_eq!(result.num_cascades, 0);
    }
}
