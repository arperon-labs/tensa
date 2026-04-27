//! Subgraph extraction and pattern matching for cross-narrative analysis.
//!
//! Extracts in-memory graph representations from the hypergraph
//! and supports VF2-lite subgraph isomorphism for pattern matching.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{AllenRelation, EntityType, NarrativeLevel, Role};

// ─── Graph Representation ────────────────────────────────────

/// In-memory graph representation of a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeGraph {
    pub narrative_id: String,
    pub nodes: Vec<GraphNode>,
    pub adjacency: Vec<Vec<(usize, EdgeLabel)>>,
}

/// A node in the narrative graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: Uuid,
    pub node_type: GraphNodeType,
    pub label: String,
    pub features: Vec<f64>,
}

/// Node type distinguishes entities from situations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphNodeType {
    Entity(EntityType),
    Situation(NarrativeLevel),
}

/// Edge label for adjacency.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeLabel {
    Participation(Role),
    Causal,
    Temporal(AllenRelation),
}

// ─── Pattern Types ───────────────────────────────────────────

/// A structural pattern template for matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSubgraph {
    pub nodes: Vec<PatternNode>,
    pub edges: Vec<PatternEdge>,
}

/// A node in a pattern template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    pub label: String,
    pub node_type: Option<GraphNodeType>,
}

/// An edge in a pattern template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEdge {
    pub from_idx: usize,
    pub to_idx: usize,
    pub edge_type: EdgeLabel,
}

/// A match of a pattern against a narrative graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    pub pattern_id: String,
    pub narrative_id: String,
    pub node_mapping: HashMap<usize, Uuid>,
    pub similarity_score: f64,
}

// ─── Extraction ──────────────────────────────────────────────

impl NarrativeGraph {
    /// Extract a narrative graph from the hypergraph by scanning all
    /// entities and situations with the given narrative_id.
    pub fn extract(narrative_id: &str, hypergraph: &Hypergraph) -> Result<Self> {
        let mut nodes = Vec::new();
        let mut id_to_idx: HashMap<Uuid, usize> = HashMap::new();

        // Add entities
        let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
        for entity in &entities {
            let idx = nodes.len();
            id_to_idx.insert(entity.id, idx);
            nodes.push(GraphNode {
                id: entity.id,
                node_type: GraphNodeType::Entity(entity.entity_type.clone()),
                label: entity
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                features: vec![entity.confidence as f64],
            });
        }

        // Add situations
        let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
        for sit in &situations {
            let idx = nodes.len();
            id_to_idx.insert(sit.id, idx);
            nodes.push(GraphNode {
                id: sit.id,
                node_type: GraphNodeType::Situation(sit.narrative_level),
                label: format!("{:?}", sit.narrative_level),
                features: vec![sit.confidence as f64],
            });
        }

        let n = nodes.len();
        let mut adjacency: Vec<Vec<(usize, EdgeLabel)>> = vec![vec![]; n];

        // Add participation edges (entity <-> situation)
        for entity in &entities {
            if let Some(&e_idx) = id_to_idx.get(&entity.id) {
                let participations = hypergraph.get_situations_for_entity(&entity.id)?;
                for participation in &participations {
                    if let Some(&s_idx) = id_to_idx.get(&participation.situation_id) {
                        adjacency[e_idx]
                            .push((s_idx, EdgeLabel::Participation(participation.role.clone())));
                        adjacency[s_idx]
                            .push((e_idx, EdgeLabel::Participation(participation.role.clone())));
                    }
                }
            }
        }

        // Add causal edges (situation -> situation) from c/ prefix index
        for sit in &situations {
            if let Some(&from_idx) = id_to_idx.get(&sit.id) {
                let consequences = hypergraph.get_consequences(&sit.id).unwrap_or_default();
                for cause in &consequences {
                    if let Some(&to_idx) = id_to_idx.get(&cause.to_situation) {
                        adjacency[from_idx].push((to_idx, EdgeLabel::Causal));
                    }
                }
            }
        }

        Ok(NarrativeGraph {
            narrative_id: narrative_id.to_string(),
            nodes,
            adjacency,
        })
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges (counting each direction once).
    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum::<usize>() / 2
    }
}

// ─── Pattern Matching (VF2-lite) ─────────────────────────────

/// Match a pattern against a narrative graph using backtracking.
/// Returns all matches with a similarity score above threshold.
pub fn match_pattern(
    pattern: &PatternSubgraph,
    graph: &NarrativeGraph,
    pattern_id: &str,
    threshold: f64,
) -> Vec<PatternMatch> {
    if pattern.nodes.is_empty() || graph.nodes.is_empty() {
        return vec![];
    }

    let mut matches = Vec::new();
    let mut mapping: Vec<Option<usize>> = vec![None; pattern.nodes.len()];
    let mut used: Vec<bool> = vec![false; graph.nodes.len()];

    backtrack(
        pattern,
        graph,
        0,
        &mut mapping,
        &mut used,
        &mut matches,
        pattern_id,
        threshold,
    );
    matches
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    pattern: &PatternSubgraph,
    graph: &NarrativeGraph,
    depth: usize,
    mapping: &mut Vec<Option<usize>>,
    used: &mut Vec<bool>,
    matches: &mut Vec<PatternMatch>,
    pattern_id: &str,
    threshold: f64,
) {
    if depth == pattern.nodes.len() {
        // Check edge compatibility
        let score = compute_match_score(pattern, graph, mapping);
        if score >= threshold {
            let node_mapping: HashMap<usize, Uuid> = mapping
                .iter()
                .enumerate()
                .filter_map(|(i, m)| m.map(|g_idx| (i, graph.nodes[g_idx].id)))
                .collect();
            matches.push(PatternMatch {
                pattern_id: pattern_id.to_string(),
                narrative_id: graph.narrative_id.clone(),
                node_mapping,
                similarity_score: score,
            });
        }
        return;
    }

    let p_node = &pattern.nodes[depth];

    for g_idx in 0..graph.nodes.len() {
        if used[g_idx] {
            continue;
        }

        // Node type compatibility
        if let Some(ref ptype) = p_node.node_type {
            if &graph.nodes[g_idx].node_type != ptype {
                continue;
            }
        }

        mapping[depth] = Some(g_idx);
        used[g_idx] = true;

        // Check partial edge constraints
        if check_partial_edges(pattern, graph, mapping, depth) {
            backtrack(
                pattern,
                graph,
                depth + 1,
                mapping,
                used,
                matches,
                pattern_id,
                threshold,
            );
        }

        mapping[depth] = None;
        used[g_idx] = false;
    }
}

/// Check edges for already-mapped nodes.
fn check_partial_edges(
    pattern: &PatternSubgraph,
    graph: &NarrativeGraph,
    mapping: &[Option<usize>],
    current_depth: usize,
) -> bool {
    for edge in &pattern.edges {
        let from_mapped = if edge.from_idx <= current_depth {
            mapping[edge.from_idx]
        } else {
            None
        };
        let to_mapped = if edge.to_idx <= current_depth {
            mapping[edge.to_idx]
        } else {
            None
        };

        // Only check if both endpoints are mapped
        if let (Some(g_from), Some(g_to)) = (from_mapped, to_mapped) {
            let has_edge = graph.adjacency[g_from]
                .iter()
                .any(|(neighbor, label)| *neighbor == g_to && *label == edge.edge_type);
            if !has_edge {
                return false;
            }
        }
    }
    true
}

/// Compute match score based on edge coverage and feature similarity.
fn compute_match_score(
    pattern: &PatternSubgraph,
    graph: &NarrativeGraph,
    mapping: &[Option<usize>],
) -> f64 {
    if pattern.edges.is_empty() {
        return 1.0;
    }

    let mut matched_edges = 0;
    for edge in &pattern.edges {
        if let (Some(g_from), Some(g_to)) = (mapping[edge.from_idx], mapping[edge.to_idx]) {
            if graph.adjacency[g_from]
                .iter()
                .any(|(n, l)| *n == g_to && *l == edge.edge_type)
            {
                matched_edges += 1;
            }
        }
    }
    matched_edges as f64 / pattern.edges.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn setup() -> (Hypergraph, String) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        (hg, "test-narrative".to_string())
    }

    fn make_entity(hg: &Hypergraph, name: &str, nid: &str) -> Uuid {
        let entity = Entity {
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
            narrative_id: Some(nid.to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap()
    }

    fn make_situation(hg: &Hypergraph, nid: &str) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
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
            raw_content: vec![ContentBlock::text("Test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.to_string()),
            source_chunk_id: None,
            source_span: None,
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
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_extract_empty_narrative() {
        let (hg, nid) = setup();
        let graph = NarrativeGraph::extract(&nid, &hg).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_extract_entities_only() {
        let (hg, nid) = setup();
        make_entity(&hg, "Alice", &nid);
        make_entity(&hg, "Bob", &nid);
        let graph = NarrativeGraph::extract(&nid, &hg).unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_extract_with_participation() {
        let (hg, nid) = setup();
        let e_id = make_entity(&hg, "Alice", &nid);
        let s_id = make_situation(&hg, &nid);
        hg.add_participant(Participation {
            entity_id: e_id,
            situation_id: s_id,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let graph = NarrativeGraph::extract(&nid, &hg).unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1); // bidirectional counted once
    }

    #[test]
    fn test_extract_ignores_other_narratives() {
        let (hg, nid) = setup();
        make_entity(&hg, "Alice", &nid);
        make_entity(&hg, "Bob", "other-narrative");
        let graph = NarrativeGraph::extract(&nid, &hg).unwrap();
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_graph_node_types() {
        let (hg, nid) = setup();
        make_entity(&hg, "Alice", &nid);
        make_situation(&hg, &nid);
        let graph = NarrativeGraph::extract(&nid, &hg).unwrap();
        assert!(matches!(
            graph.nodes[0].node_type,
            GraphNodeType::Entity(EntityType::Actor)
        ));
        assert!(matches!(
            graph.nodes[1].node_type,
            GraphNodeType::Situation(NarrativeLevel::Scene)
        ));
    }

    #[test]
    fn test_match_pattern_empty() {
        let pattern = PatternSubgraph {
            nodes: vec![],
            edges: vec![],
        };
        let graph = NarrativeGraph {
            narrative_id: "test".to_string(),
            nodes: vec![],
            adjacency: vec![],
        };
        let matches = match_pattern(&pattern, &graph, "p1", 0.5);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_pattern_single_node() {
        let pattern = PatternSubgraph {
            nodes: vec![PatternNode {
                label: "actor".to_string(),
                node_type: Some(GraphNodeType::Entity(EntityType::Actor)),
            }],
            edges: vec![],
        };
        let graph = NarrativeGraph {
            narrative_id: "test".to_string(),
            nodes: vec![GraphNode {
                id: Uuid::now_v7(),
                node_type: GraphNodeType::Entity(EntityType::Actor),
                label: "Alice".to_string(),
                features: vec![0.9],
            }],
            adjacency: vec![vec![]],
        };
        let matches = match_pattern(&pattern, &graph, "p1", 0.5);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].similarity_score, 1.0);
    }

    #[test]
    fn test_match_pattern_type_mismatch() {
        let pattern = PatternSubgraph {
            nodes: vec![PatternNode {
                label: "location".to_string(),
                node_type: Some(GraphNodeType::Entity(EntityType::Location)),
            }],
            edges: vec![],
        };
        let graph = NarrativeGraph {
            narrative_id: "test".to_string(),
            nodes: vec![GraphNode {
                id: Uuid::now_v7(),
                node_type: GraphNodeType::Entity(EntityType::Actor),
                label: "Alice".to_string(),
                features: vec![0.9],
            }],
            adjacency: vec![vec![]],
        };
        let matches = match_pattern(&pattern, &graph, "p1", 0.5);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_pattern_with_edge() {
        let id_a = Uuid::now_v7();
        let id_s = Uuid::now_v7();

        let pattern = PatternSubgraph {
            nodes: vec![
                PatternNode {
                    label: "actor".to_string(),
                    node_type: Some(GraphNodeType::Entity(EntityType::Actor)),
                },
                PatternNode {
                    label: "scene".to_string(),
                    node_type: Some(GraphNodeType::Situation(NarrativeLevel::Scene)),
                },
            ],
            edges: vec![PatternEdge {
                from_idx: 0,
                to_idx: 1,
                edge_type: EdgeLabel::Participation(Role::Protagonist),
            }],
        };

        let graph = NarrativeGraph {
            narrative_id: "test".to_string(),
            nodes: vec![
                GraphNode {
                    id: id_a,
                    node_type: GraphNodeType::Entity(EntityType::Actor),
                    label: "Alice".to_string(),
                    features: vec![0.9],
                },
                GraphNode {
                    id: id_s,
                    node_type: GraphNodeType::Situation(NarrativeLevel::Scene),
                    label: "Scene".to_string(),
                    features: vec![0.8],
                },
            ],
            adjacency: vec![
                vec![(1, EdgeLabel::Participation(Role::Protagonist))],
                vec![(0, EdgeLabel::Participation(Role::Protagonist))],
            ],
        };

        let matches = match_pattern(&pattern, &graph, "p1", 0.5);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].node_mapping[&0], id_a);
        assert_eq!(matches[0].node_mapping[&1], id_s);
    }

    #[test]
    fn test_match_pattern_below_threshold() {
        let pattern = PatternSubgraph {
            nodes: vec![
                PatternNode {
                    label: "a".to_string(),
                    node_type: Some(GraphNodeType::Entity(EntityType::Actor)),
                },
                PatternNode {
                    label: "b".to_string(),
                    node_type: Some(GraphNodeType::Entity(EntityType::Actor)),
                },
            ],
            edges: vec![PatternEdge {
                from_idx: 0,
                to_idx: 1,
                edge_type: EdgeLabel::Causal,
            }],
        };

        let graph = NarrativeGraph {
            narrative_id: "test".to_string(),
            nodes: vec![
                GraphNode {
                    id: Uuid::now_v7(),
                    node_type: GraphNodeType::Entity(EntityType::Actor),
                    label: "A".to_string(),
                    features: vec![],
                },
                GraphNode {
                    id: Uuid::now_v7(),
                    node_type: GraphNodeType::Entity(EntityType::Actor),
                    label: "B".to_string(),
                    features: vec![],
                },
            ],
            adjacency: vec![vec![], vec![]], // no edges
        };

        let matches = match_pattern(&pattern, &graph, "p1", 0.5);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let graph = NarrativeGraph {
            narrative_id: "test".to_string(),
            nodes: vec![GraphNode {
                id: Uuid::now_v7(),
                node_type: GraphNodeType::Entity(EntityType::Actor),
                label: "Alice".to_string(),
                features: vec![0.5, 0.7],
            }],
            adjacency: vec![vec![]],
        };
        let json = serde_json::to_vec(&graph).unwrap();
        let decoded: NarrativeGraph = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.narrative_id, "test");
        assert_eq!(decoded.nodes.len(), 1);
    }
}
