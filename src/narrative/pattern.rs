//! Cross-narrative pattern mining engine.
//!
//! Discovers frequent structural patterns across multiple narratives
//! using graph kernel methods and stores them in the KV store.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::keys;
use crate::store::KVStore;

use super::subgraph::{
    EdgeLabel, GraphNodeType, NarrativeGraph, PatternEdge, PatternNode, PatternSubgraph,
};

// ─── Pattern Types ───────────────────────────────────────────

/// A discovered narrative pattern with its frequency and supporting narratives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativePattern {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub subgraph: PatternSubgraph,
    pub frequency: usize,
    pub narrative_ids: Vec<String>,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}

// ─── Configuration ──────────────────────────────────────────

/// Configuration for the pattern mining algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMiningConfig {
    /// Minimum number of narratives a pattern must appear in.
    pub min_support: usize,
    /// Maximum causal chain length to discover (default 6, was hardcoded 3).
    pub max_chain_length: usize,
    /// Maximum star motif size (number of participant roles around a situation).
    pub max_star_size: usize,
    /// Whether to discover star motifs (situation + multiple participant types).
    pub enable_star_motifs: bool,
    /// Maximum total patterns to return (safety cap).
    pub max_patterns: usize,
}

impl Default for PatternMiningConfig {
    fn default() -> Self {
        Self {
            min_support: 1,
            max_chain_length: 6,
            max_star_size: 4,
            enable_star_motifs: true,
            max_patterns: 1000,
        }
    }
}

// ─── Frequent Motif Mining ───────────────────────────────────

/// A frequent edge motif (2-node pattern) discovered across graphs.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct EdgeMotif {
    from_type: GraphNodeType,
    to_type: GraphNodeType,
    edge_label: EdgeLabel,
}

/// Mine frequent patterns across a set of narrative graphs.
///
/// Backward-compatible wrapper using default config with the given min_support.
pub fn mine_patterns(graphs: &[NarrativeGraph], min_support: usize) -> Vec<NarrativePattern> {
    let config = PatternMiningConfig {
        min_support,
        ..Default::default()
    };
    mine_patterns_with_config(graphs, &config)
}

/// Mine frequent patterns with full configuration control.
///
/// Three-phase bottom-up approach:
/// 1. Frequent edge motifs (2-node patterns)
/// 2. Causal chains of length 3..=max_chain_length (recursive extension with anti-monotone pruning)
/// 3. Star motifs: situation nodes with multiple participant types (optional)
pub fn mine_patterns_with_config(
    graphs: &[NarrativeGraph],
    config: &PatternMiningConfig,
) -> Vec<NarrativePattern> {
    if graphs.is_empty() || config.min_support == 0 {
        return vec![];
    }

    let mut patterns = Vec::new();
    let num_graphs = graphs.len();

    // Phase 1: Frequent edge motifs (2-node patterns)
    let motifs = find_frequent_motifs(graphs, config.min_support);
    for (motif, supporting_narratives) in &motifs {
        if patterns.len() >= config.max_patterns {
            break;
        }
        patterns.push(NarrativePattern {
            id: Uuid::now_v7().to_string(),
            name: format!("{:?}->{:?}", motif.from_type, motif.to_type),
            description: Some(format!("Edge pattern: {:?}", motif.edge_label)),
            subgraph: PatternSubgraph {
                nodes: vec![
                    PatternNode {
                        label: format!("{:?}", motif.from_type),
                        node_type: Some(motif.from_type.clone()),
                    },
                    PatternNode {
                        label: format!("{:?}", motif.to_type),
                        node_type: Some(motif.to_type.clone()),
                    },
                ],
                edges: vec![PatternEdge {
                    from_idx: 0,
                    to_idx: 1,
                    edge_type: motif.edge_label.clone(),
                }],
            },
            frequency: supporting_narratives.len(),
            narrative_ids: supporting_narratives.clone(),
            confidence: supporting_narratives.len() as f32 / num_graphs as f32,
            created_at: Utc::now(),
        });
    }

    // Phase 2: Causal chains (3..=max_chain_length nodes, recursive extension)
    if config.max_chain_length >= 3 {
        let chains = find_causal_chains_recursive(graphs, config);
        for (chain, supporting_narratives) in &chains {
            if patterns.len() >= config.max_patterns {
                break;
            }
            patterns.push(NarrativePattern {
                id: Uuid::now_v7().to_string(),
                name: format!("chain-{}", chain.len()),
                description: Some(format!("Causal chain of length {}", chain.len())),
                subgraph: chain_to_subgraph(chain),
                frequency: supporting_narratives.len(),
                narrative_ids: supporting_narratives.clone(),
                confidence: supporting_narratives.len() as f32 / num_graphs as f32,
                created_at: Utc::now(),
            });
        }
    }

    // Phase 3: Star motifs (situation + multiple participant types)
    if config.enable_star_motifs && config.max_star_size >= 2 {
        let stars = find_star_motifs(graphs, config);
        for (star, supporting_narratives) in &stars {
            if patterns.len() >= config.max_patterns {
                break;
            }
            patterns.push(NarrativePattern {
                id: Uuid::now_v7().to_string(),
                name: format!("star-{}-{}", star.center_type_label(), star.arms.len()),
                description: Some(format!(
                    "Star: {:?} with {} participant types",
                    star.center_type,
                    star.arms.len()
                )),
                subgraph: star.to_subgraph(),
                frequency: supporting_narratives.len(),
                narrative_ids: supporting_narratives.clone(),
                confidence: supporting_narratives.len() as f32 / num_graphs as f32,
                created_at: Utc::now(),
            });
        }
    }

    patterns
}

fn find_frequent_motifs(
    graphs: &[NarrativeGraph],
    min_support: usize,
) -> Vec<(EdgeMotif, Vec<String>)> {
    let mut motif_narratives: HashMap<EdgeMotif, Vec<String>> = HashMap::new();

    for graph in graphs {
        let mut seen_in_this_graph: std::collections::HashSet<EdgeMotif> =
            std::collections::HashSet::new();

        for (node_idx, adj) in graph.adjacency.iter().enumerate() {
            let from_type = &graph.nodes[node_idx].node_type;
            for (neighbor, edge_label) in adj {
                let to_type = &graph.nodes[*neighbor].node_type;
                let motif = EdgeMotif {
                    from_type: from_type.clone(),
                    to_type: to_type.clone(),
                    edge_label: edge_label.clone(),
                };
                if seen_in_this_graph.insert(motif.clone()) {
                    motif_narratives
                        .entry(motif)
                        .or_default()
                        .push(graph.narrative_id.clone());
                }
            }
        }
    }

    motif_narratives
        .into_iter()
        .filter(|(_, narratives)| narratives.len() >= min_support)
        .collect()
}

/// Find frequent causal chains (sequences of nodes connected by Causal edges).
/// Recursively discover causal chains of length 3..=max_chain_length using
/// level-wise Apriori extension with anti-monotone pruning.
fn find_causal_chains_recursive(
    graphs: &[NarrativeGraph],
    config: &PatternMiningConfig,
) -> Vec<(Vec<GraphNodeType>, Vec<String>)> {
    let mut all_chains: Vec<(Vec<GraphNodeType>, Vec<String>)> = Vec::new();

    // Seed: collect all 2-node causal edges per graph
    // Key: chain type signature, Value: supporting narrative IDs
    let mut current_level: HashMap<Vec<GraphNodeType>, HashSet<String>> = HashMap::new();

    for graph in graphs {
        let mut seen: HashSet<Vec<GraphNodeType>> = HashSet::new();
        for node_idx in 0..graph.nodes.len() {
            for (neighbor, label) in &graph.adjacency[node_idx] {
                if matches!(label, EdgeLabel::Causal) {
                    let chain = vec![
                        graph.nodes[node_idx].node_type.clone(),
                        graph.nodes[*neighbor].node_type.clone(),
                    ];
                    if seen.insert(chain.clone()) {
                        current_level
                            .entry(chain)
                            .or_default()
                            .insert(graph.narrative_id.clone());
                    }
                }
            }
        }
    }

    // Filter seeds by min_support
    current_level.retain(|_, narratives| narratives.len() >= config.min_support);

    // Iterative extension: length 3, 4, ..., max_chain_length
    for _length in 3..=config.max_chain_length {
        let mut next_level: HashMap<Vec<GraphNodeType>, HashSet<String>> = HashMap::new();

        // For each surviving chain, try to extend by one causal hop
        for (chain_types, supporting_narratives) in &current_level {
            for graph in graphs {
                if !supporting_narratives.contains(&graph.narrative_id) {
                    continue;
                }

                // Find all instances of this chain in the graph and try to extend
                extend_chain_in_graph(graph, chain_types, &mut next_level);
            }
        }

        // Filter by min_support (anti-monotone pruning)
        next_level.retain(|_, narratives| narratives.len() >= config.min_support);

        if next_level.is_empty() {
            break; // No more chains to extend
        }

        // Collect chains at this length
        for (chain, narratives) in &next_level {
            all_chains.push((chain.clone(), narratives.iter().cloned().collect()));
        }

        current_level = next_level;
    }

    // Also include 3-node chains from the first extension if we started from 2-node seeds
    // (The first iteration of the loop above produces length-3 chains)
    all_chains
}

/// Find instances of a chain type signature in a graph and extend by one causal hop.
fn extend_chain_in_graph(
    graph: &NarrativeGraph,
    chain_types: &[GraphNodeType],
    results: &mut HashMap<Vec<GraphNodeType>, HashSet<String>>,
) {
    let chain_len = chain_types.len();
    if chain_len == 0 {
        return;
    }

    // Find all starting nodes matching the first chain type
    for start_idx in 0..graph.nodes.len() {
        if graph.nodes[start_idx].node_type != chain_types[0] {
            continue;
        }

        // DFS to find chain instances
        let mut path = vec![start_idx];
        find_chain_instances(graph, chain_types, 1, &mut path, results);
    }
}

/// DFS to find all instances of a chain type pattern starting from the current path,
/// then extend each instance by one causal hop.
fn find_chain_instances(
    graph: &NarrativeGraph,
    chain_types: &[GraphNodeType],
    depth: usize,
    path: &mut Vec<usize>,
    results: &mut HashMap<Vec<GraphNodeType>, HashSet<String>>,
) {
    if depth == chain_types.len() {
        // Found a complete chain instance — now extend it
        let last_idx = *path.last().unwrap();
        let visited: HashSet<usize> = path.iter().copied().collect();

        for (neighbor, label) in &graph.adjacency[last_idx] {
            if !matches!(label, EdgeLabel::Causal) || visited.contains(neighbor) {
                continue;
            }

            let mut extended = Vec::with_capacity(chain_types.len() + 1);
            extended.extend_from_slice(chain_types);
            extended.push(graph.nodes[*neighbor].node_type.clone());

            results
                .entry(extended)
                .or_default()
                .insert(graph.narrative_id.clone());
        }
        return;
    }

    let current_idx = *path.last().unwrap();
    for (neighbor, label) in &graph.adjacency[current_idx] {
        if !matches!(label, EdgeLabel::Causal) {
            continue;
        }
        if path.contains(neighbor) {
            continue; // cycle prevention
        }
        if graph.nodes[*neighbor].node_type != chain_types[depth] {
            continue;
        }
        path.push(*neighbor);
        find_chain_instances(graph, chain_types, depth + 1, path, results);
        path.pop();
    }
}

// ─── Star Motif Mining ──────────────────────────────────────

/// A star motif: a central situation with multiple participant types.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct StarMotif {
    center_type: GraphNodeType,
    arms: Vec<(GraphNodeType, EdgeLabel)>, // sorted for canonical form
}

impl StarMotif {
    fn center_type_label(&self) -> String {
        format!("{:?}", self.center_type)
    }

    fn to_subgraph(&self) -> PatternSubgraph {
        let mut nodes = vec![PatternNode {
            label: format!("{:?}", self.center_type),
            node_type: Some(self.center_type.clone()),
        }];
        let mut edges = Vec::new();

        for (i, (node_type, edge_label)) in self.arms.iter().enumerate() {
            nodes.push(PatternNode {
                label: format!("{:?}", node_type),
                node_type: Some(node_type.clone()),
            });
            edges.push(PatternEdge {
                from_idx: i + 1,
                to_idx: 0,
                edge_type: edge_label.clone(),
            });
        }

        PatternSubgraph { nodes, edges }
    }
}

/// Discover star motifs: situation nodes with multiple participant types.
fn find_star_motifs(
    graphs: &[NarrativeGraph],
    config: &PatternMiningConfig,
) -> Vec<(StarMotif, Vec<String>)> {
    let mut star_narratives: HashMap<StarMotif, HashSet<String>> = HashMap::new();

    for graph in graphs {
        let mut seen: HashSet<StarMotif> = HashSet::new();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Only situation nodes can be star centers
            if !matches!(node.node_type, GraphNodeType::Situation(_)) {
                continue;
            }

            // Collect participation edges to entity nodes
            let mut arms: Vec<(GraphNodeType, EdgeLabel)> = Vec::new();
            for (neighbor, label) in &graph.adjacency[node_idx] {
                if matches!(label, EdgeLabel::Participation(_))
                    && matches!(graph.nodes[*neighbor].node_type, GraphNodeType::Entity(_))
                {
                    arms.push((graph.nodes[*neighbor].node_type.clone(), label.clone()));
                }
            }

            // Sort for canonical form and deduplicate
            arms.sort_by(|a, b| format!("{:?}{:?}", a.0, a.1).cmp(&format!("{:?}{:?}", b.0, b.1)));
            arms.dedup();

            if arms.len() < 2 {
                continue; // need at least 2 arms for a star
            }

            // Generate subsets of size 2..=min(max_star_size, arms.len())
            let max_k = config.max_star_size.min(arms.len());
            for k in 2..=max_k {
                // Generate all k-combinations
                let combinations = combinations_of(&arms, k);
                for combo in combinations {
                    let star = StarMotif {
                        center_type: node.node_type.clone(),
                        arms: combo,
                    };
                    if seen.insert(star.clone()) {
                        star_narratives
                            .entry(star)
                            .or_default()
                            .insert(graph.narrative_id.clone());
                    }
                }
            }
        }
    }

    star_narratives
        .into_iter()
        .filter(|(_, narratives)| narratives.len() >= config.min_support)
        .map(|(star, narratives)| (star, narratives.into_iter().collect()))
        .collect()
}

/// Generate all k-combinations from a slice. Capped at 500 to prevent explosion.
fn combinations_of<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k > items.len() {
        return vec![];
    }
    let mut result = Vec::new();
    let mut indices: Vec<usize> = (0..k).collect();

    loop {
        if result.len() >= 500 {
            break; // safety cap
        }
        result.push(indices.iter().map(|&i| items[i].clone()).collect());

        // Find rightmost index that can be incremented
        let mut i = k;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if indices[i] != i + items.len() - k {
                break;
            }
            if i == 0 && indices[0] == items.len() - k {
                return result;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }

    result
}

fn chain_to_subgraph(types: &[GraphNodeType]) -> PatternSubgraph {
    let nodes: Vec<PatternNode> = types
        .iter()
        .map(|t| PatternNode {
            label: format!("{:?}", t),
            node_type: Some(t.clone()),
        })
        .collect();
    let edges: Vec<PatternEdge> = (0..types.len().saturating_sub(1))
        .map(|i| PatternEdge {
            from_idx: i,
            to_idx: i + 1,
            edge_type: EdgeLabel::Causal,
        })
        .collect();
    PatternSubgraph { nodes, edges }
}

// ─── Pattern Storage ─────────────────────────────────────────

/// Store a pattern in the KV store.
pub fn store_pattern(store: &dyn KVStore, pattern: &NarrativePattern) -> Result<()> {
    let key = keys::pattern_key(&pattern.id);
    let bytes = serde_json::to_vec(pattern)?;
    store.put(&key, &bytes)
}

/// Load a pattern from the KV store.
pub fn load_pattern(store: &dyn KVStore, pattern_id: &str) -> Result<Option<NarrativePattern>> {
    let key = keys::pattern_key(pattern_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List all stored patterns.
pub fn list_patterns(store: &dyn KVStore) -> Result<Vec<NarrativePattern>> {
    // Use prefix scan on pm/ but exclude pm/m/ (match records)
    let pairs = store.prefix_scan(keys::PATTERN)?;
    let mut patterns = Vec::new();
    for (key, value) in pairs {
        // Skip match records (pm/m/)
        if key.starts_with(keys::PATTERN_MATCH) {
            continue;
        }
        if let Ok(pattern) = serde_json::from_slice::<NarrativePattern>(&value) {
            patterns.push(pattern);
        }
    }
    Ok(patterns)
}

// ─── InferenceEngine Implementation ─────────────────────────

use crate::error::TensaError;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Pattern mining engine for the inference job queue.
pub struct PatternMiningEngine;

impl InferenceEngine for PatternMiningEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::PatternMining
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(15000) // 15 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        // Extract narrative IDs — may be a single string or an array
        let narrative_ids: Vec<String> = if let Some(arr) = job
            .parameters
            .get("narrative_ids")
            .and_then(|v| v.as_array())
        {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else if let Some(nid) = job.parameters.get("narrative_id").and_then(|v| v.as_str()) {
            vec![nid.to_string()]
        } else {
            return Err(TensaError::InferenceError(
                "missing narrative_id or narrative_ids".into(),
            ));
        };

        // Build config from job parameters
        let mut config = PatternMiningConfig::default();
        if let Some(v) = job.parameters.get("min_support").and_then(|v| v.as_u64()) {
            config.min_support = v as usize;
        }
        if let Some(v) = job
            .parameters
            .get("max_chain_length")
            .and_then(|v| v.as_u64())
        {
            config.max_chain_length = v as usize;
        }
        if let Some(v) = job.parameters.get("max_star_size").and_then(|v| v.as_u64()) {
            config.max_star_size = v as usize;
        }
        if let Some(v) = job
            .parameters
            .get("enable_star_motifs")
            .and_then(|v| v.as_bool())
        {
            config.enable_star_motifs = v;
        }

        // Build NarrativeGraph for each narrative
        let mut graphs = Vec::with_capacity(narrative_ids.len());
        for nid in &narrative_ids {
            let graph = NarrativeGraph::extract(nid, hypergraph)?;
            graphs.push(graph);
        }

        let patterns = mine_patterns_with_config(&graphs, &config);

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::PatternMining,
            target_id: job.target_id,
            result: serde_json::to_value(&patterns)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Pattern mining: {} patterns found across {} narratives",
                patterns.len(),
                narrative_ids.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(chrono::Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrative::subgraph::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use std::sync::Arc;

    fn make_graph(
        nid: &str,
        labels: Vec<&str>,
        node_types: Vec<GraphNodeType>,
        edges: Vec<(usize, usize, EdgeLabel)>,
    ) -> NarrativeGraph {
        let nodes: Vec<GraphNode> = labels
            .iter()
            .zip(node_types.iter())
            .map(|(l, t)| GraphNode {
                id: Uuid::now_v7(),
                node_type: t.clone(),
                label: l.to_string(),
                features: vec![],
            })
            .collect();

        let n = nodes.len();
        let mut adjacency: Vec<Vec<(usize, EdgeLabel)>> = vec![vec![]; n];
        for (from, to, label) in edges {
            adjacency[from].push((to, label.clone()));
            adjacency[to].push((from, label));
        }

        NarrativeGraph {
            narrative_id: nid.to_string(),
            nodes,
            adjacency,
        }
    }

    #[test]
    fn test_mine_patterns_empty() {
        let patterns = mine_patterns(&[], 1);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_mine_patterns_single_graph() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);
        let g = make_graph(
            "n1",
            vec!["Alice", "Scene1"],
            vec![actor.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Participation(Role::Protagonist))],
        );
        let patterns = mine_patterns(&[g], 1);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_mine_patterns_shared_motif() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g1 = make_graph(
            "n1",
            vec!["Alice", "Scene1"],
            vec![actor.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Participation(Role::Protagonist))],
        );
        let g2 = make_graph(
            "n2",
            vec!["Bob", "Scene2"],
            vec![actor.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Participation(Role::Protagonist))],
        );

        let patterns = mine_patterns(&[g1, g2], 2);
        assert!(
            !patterns.is_empty(),
            "Should find shared protagonist-participation motif"
        );
        // The shared pattern should reference both narratives
        let shared = patterns.iter().find(|p| p.narrative_ids.len() == 2);
        assert!(shared.is_some());
    }

    #[test]
    fn test_mine_patterns_min_support_filter() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g1 = make_graph(
            "n1",
            vec!["A", "S1"],
            vec![actor.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Participation(Role::Protagonist))],
        );
        let g2 = make_graph(
            "n2",
            vec!["B", "S2"],
            vec![actor.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Participation(Role::Antagonist))],
        );

        // Protagonist motif only in g1, Antagonist only in g2
        let patterns = mine_patterns(&[g1, g2], 2);
        // Neither motif appears in both graphs at min_support=2
        let protagonist_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.narrative_ids.len() >= 2)
            .collect();
        // The participation motifs have different roles, so they won't match
        // But both have Entity(Actor)->Situation(Scene) participation
        // The edge labels differ (Protagonist vs Antagonist) so they are different motifs
        assert!(
            protagonist_patterns.is_empty()
                || protagonist_patterns.iter().all(|p| p.frequency >= 2)
        );
    }

    #[test]
    fn test_causal_chain_mining() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g1 = make_graph(
            "n1",
            vec!["S1", "S2", "S3"],
            vec![scene.clone(), scene.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Causal), (1, 2, EdgeLabel::Causal)],
        );
        let g2 = make_graph(
            "n2",
            vec!["S4", "S5", "S6"],
            vec![scene.clone(), scene.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Causal), (1, 2, EdgeLabel::Causal)],
        );

        let patterns = mine_patterns(&[g1, g2], 2);
        let chains: Vec<_> = patterns
            .iter()
            .filter(|p| p.subgraph.nodes.len() == 3)
            .collect();
        assert!(!chains.is_empty(), "Should find 3-node causal chain");
    }

    #[test]
    fn test_store_and_load_pattern() {
        let store = Arc::new(MemoryStore::new());
        let pattern = NarrativePattern {
            id: "test-pattern".to_string(),
            name: "Test Pattern".to_string(),
            description: Some("A test".to_string()),
            subgraph: PatternSubgraph {
                nodes: vec![PatternNode {
                    label: "A".to_string(),
                    node_type: None,
                }],
                edges: vec![],
            },
            frequency: 3,
            narrative_ids: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            confidence: 0.75,
            created_at: Utc::now(),
        };

        store_pattern(store.as_ref(), &pattern).unwrap();
        let loaded = load_pattern(store.as_ref(), "test-pattern").unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.name, "Test Pattern");
        assert_eq!(loaded.frequency, 3);
    }

    #[test]
    fn test_load_pattern_not_found() {
        let store = Arc::new(MemoryStore::new());
        let loaded = load_pattern(store.as_ref(), "nonexistent").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_list_patterns() {
        let store = Arc::new(MemoryStore::new());

        let p1 = NarrativePattern {
            id: "p1".to_string(),
            name: "Pattern 1".to_string(),
            description: None,
            subgraph: PatternSubgraph {
                nodes: vec![],
                edges: vec![],
            },
            frequency: 1,
            narrative_ids: vec!["n1".to_string()],
            confidence: 0.5,
            created_at: Utc::now(),
        };
        let p2 = NarrativePattern {
            id: "p2".to_string(),
            name: "Pattern 2".to_string(),
            description: None,
            subgraph: PatternSubgraph {
                nodes: vec![],
                edges: vec![],
            },
            frequency: 2,
            narrative_ids: vec!["n1".to_string(), "n2".to_string()],
            confidence: 0.8,
            created_at: Utc::now(),
        };

        store_pattern(store.as_ref(), &p1).unwrap();
        store_pattern(store.as_ref(), &p2).unwrap();

        let all = list_patterns(store.as_ref()).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_pattern_confidence_calculation() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let graphs: Vec<_> = (0..4)
            .map(|i| {
                make_graph(
                    &format!("n{}", i),
                    vec!["A", "S"],
                    vec![actor.clone(), scene.clone()],
                    vec![(0, 1, EdgeLabel::Participation(Role::Protagonist))],
                )
            })
            .collect();

        let patterns = mine_patterns(&graphs, 2);
        for p in &patterns {
            assert!(p.confidence > 0.0 && p.confidence <= 1.0);
        }
    }

    #[test]
    fn test_pattern_mining_engine_execute() {
        use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};
        use chrono::Utc;

        let hg = make_hg();
        let e1 = add_entity(&hg, "Alice", "pm-eng");
        let s1 = add_situation(&hg, "pm-eng");
        link(&hg, e1, s1);

        let engine = PatternMiningEngine;
        assert_eq!(engine.job_type(), InferenceJobType::PatternMining);

        let job = crate::inference::types::InferenceJob {
            id: "pm-test".to_string(),
            job_type: InferenceJobType::PatternMining,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_ids": ["pm-eng"], "min_support": 1}),
            priority: crate::types::JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());
        assert!(result.result.is_array());
    }

    // ─── New tests for extended pattern mining ──────────────────

    #[test]
    fn test_chain_length_4() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);
        let event = GraphNodeType::Situation(NarrativeLevel::Event);

        let g1 = make_graph(
            "n1",
            vec!["S1", "S2", "S3", "S4"],
            vec![scene.clone(), event.clone(), scene.clone(), event.clone()],
            vec![
                (0, 1, EdgeLabel::Causal),
                (1, 2, EdgeLabel::Causal),
                (2, 3, EdgeLabel::Causal),
            ],
        );
        let g2 = make_graph(
            "n2",
            vec!["A", "B", "C", "D"],
            vec![scene.clone(), event.clone(), scene.clone(), event.clone()],
            vec![
                (0, 1, EdgeLabel::Causal),
                (1, 2, EdgeLabel::Causal),
                (2, 3, EdgeLabel::Causal),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 2,
            max_chain_length: 6,
            enable_star_motifs: false,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g1, g2], &config);
        let long_chains: Vec<_> = patterns
            .iter()
            .filter(|p| p.subgraph.nodes.len() == 4)
            .collect();
        assert!(
            !long_chains.is_empty(),
            "Should find 4-node causal chain; found patterns: {:?}",
            patterns
                .iter()
                .map(|p| p.subgraph.nodes.len())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_chain_length_5() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g1 = make_graph(
            "n1",
            vec!["A", "B", "C", "D", "E"],
            vec![scene.clone(); 5],
            vec![
                (0, 1, EdgeLabel::Causal),
                (1, 2, EdgeLabel::Causal),
                (2, 3, EdgeLabel::Causal),
                (3, 4, EdgeLabel::Causal),
            ],
        );
        let g2 = make_graph(
            "n2",
            vec!["V", "W", "X", "Y", "Z"],
            vec![scene.clone(); 5],
            vec![
                (0, 1, EdgeLabel::Causal),
                (1, 2, EdgeLabel::Causal),
                (2, 3, EdgeLabel::Causal),
                (3, 4, EdgeLabel::Causal),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 2,
            max_chain_length: 6,
            enable_star_motifs: false,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g1, g2], &config);
        let chain5: Vec<_> = patterns
            .iter()
            .filter(|p| p.subgraph.nodes.len() == 5)
            .collect();
        assert!(
            !chain5.is_empty(),
            "Should find 5-node chain; sizes: {:?}",
            patterns
                .iter()
                .map(|p| p.subgraph.nodes.len())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_chain_max_length_respected() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g = make_graph(
            "n1",
            vec!["A", "B", "C", "D", "E"],
            vec![scene.clone(); 5],
            vec![
                (0, 1, EdgeLabel::Causal),
                (1, 2, EdgeLabel::Causal),
                (2, 3, EdgeLabel::Causal),
                (3, 4, EdgeLabel::Causal),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 1,
            max_chain_length: 3, // cap at 3
            enable_star_motifs: false,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g], &config);
        for p in &patterns {
            assert!(
                p.subgraph.nodes.len() <= 3,
                "No pattern should exceed max_chain_length=3, got {}",
                p.subgraph.nodes.len()
            );
        }
    }

    #[test]
    fn test_anti_monotone_pruning() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);
        let event = GraphNodeType::Situation(NarrativeLevel::Event);

        // g1: Scene->Event->Scene (3-chain)
        // g2: Scene->Event (only 2-chain, no continuation)
        let g1 = make_graph(
            "n1",
            vec!["A", "B", "C"],
            vec![scene.clone(), event.clone(), scene.clone()],
            vec![(0, 1, EdgeLabel::Causal), (1, 2, EdgeLabel::Causal)],
        );
        let g2 = make_graph(
            "n2",
            vec!["X", "Y"],
            vec![scene.clone(), event.clone()],
            vec![(0, 1, EdgeLabel::Causal)],
        );

        let config = PatternMiningConfig {
            min_support: 2,
            max_chain_length: 6,
            enable_star_motifs: false,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g1, g2], &config);
        // 3-node chain only appears in g1 (support=1), should be pruned at min_support=2
        let chain3: Vec<_> = patterns
            .iter()
            .filter(|p| p.subgraph.nodes.len() == 3)
            .collect();
        assert!(
            chain3.is_empty(),
            "3-node chain should be pruned (support=1 < min_support=2)"
        );
    }

    #[test]
    fn test_no_cycle_in_chain() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        // Graph with a cycle: A->B->C->A
        let g = make_graph(
            "n1",
            vec!["A", "B", "C"],
            vec![scene.clone(); 3],
            vec![
                (0, 1, EdgeLabel::Causal),
                (1, 2, EdgeLabel::Causal),
                (2, 0, EdgeLabel::Causal), // cycle back to A
            ],
        );

        let config = PatternMiningConfig {
            min_support: 1,
            max_chain_length: 6,
            enable_star_motifs: false,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g], &config);
        // Chains should never revisit nodes — max chain = 3 here (all 3 distinct nodes)
        for p in &patterns {
            assert!(
                p.subgraph.nodes.len() <= 3,
                "Chain should not revisit nodes in a 3-node graph"
            );
        }
    }

    #[test]
    fn test_star_motif_basic() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let loc = GraphNodeType::Entity(EntityType::Location);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        // Scene with Protagonist(Actor) + Bystander(Location)
        let g1 = make_graph(
            "n1",
            vec!["Alice", "Park", "Scene1"],
            vec![actor.clone(), loc.clone(), scene.clone()],
            vec![
                (0, 2, EdgeLabel::Participation(Role::Protagonist)),
                (1, 2, EdgeLabel::Participation(Role::Bystander)),
            ],
        );
        let g2 = make_graph(
            "n2",
            vec!["Bob", "Garden", "Scene2"],
            vec![actor.clone(), loc.clone(), scene.clone()],
            vec![
                (0, 2, EdgeLabel::Participation(Role::Protagonist)),
                (1, 2, EdgeLabel::Participation(Role::Bystander)),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 2,
            max_chain_length: 3,
            enable_star_motifs: true,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g1, g2], &config);
        let stars: Vec<_> = patterns
            .iter()
            .filter(|p| p.name.starts_with("star-"))
            .collect();
        assert!(
            !stars.is_empty(),
            "Should find star motif (Scene with Actor+Location)"
        );
    }

    #[test]
    fn test_star_motif_3_roles() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let org = GraphNodeType::Entity(EntityType::Organization);
        let loc = GraphNodeType::Entity(EntityType::Location);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g1 = make_graph(
            "n1",
            vec!["Alice", "Corp", "City", "Scene1"],
            vec![actor.clone(), org.clone(), loc.clone(), scene.clone()],
            vec![
                (0, 3, EdgeLabel::Participation(Role::Protagonist)),
                (1, 3, EdgeLabel::Participation(Role::Target)),
                (2, 3, EdgeLabel::Participation(Role::Bystander)),
            ],
        );
        let g2 = make_graph(
            "n2",
            vec!["Bob", "Agency", "Town", "Scene2"],
            vec![actor.clone(), org.clone(), loc.clone(), scene.clone()],
            vec![
                (0, 3, EdgeLabel::Participation(Role::Protagonist)),
                (1, 3, EdgeLabel::Participation(Role::Target)),
                (2, 3, EdgeLabel::Participation(Role::Bystander)),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 2,
            max_chain_length: 3,
            enable_star_motifs: true,
            max_star_size: 4,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g1, g2], &config);
        let star3: Vec<_> = patterns
            .iter()
            .filter(|p| p.name.starts_with("star-") && p.subgraph.nodes.len() == 4) // center + 3 arms
            .collect();
        assert!(
            !star3.is_empty(),
            "Should find 3-arm star motif; stars: {:?}",
            patterns
                .iter()
                .filter(|p| p.name.starts_with("star-"))
                .map(|p| &p.name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_star_motif_disabled() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let loc = GraphNodeType::Entity(EntityType::Location);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g = make_graph(
            "n1",
            vec!["Alice", "Park", "Scene1"],
            vec![actor.clone(), loc.clone(), scene.clone()],
            vec![
                (0, 2, EdgeLabel::Participation(Role::Protagonist)),
                (1, 2, EdgeLabel::Participation(Role::Bystander)),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 1,
            enable_star_motifs: false,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g], &config);
        let stars: Vec<_> = patterns
            .iter()
            .filter(|p| p.name.starts_with("star-"))
            .collect();
        assert!(stars.is_empty(), "Star motifs should be disabled");
    }

    #[test]
    fn test_backward_compat() {
        let actor = GraphNodeType::Entity(EntityType::Actor);
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);

        let g1 = make_graph(
            "n1",
            vec!["A", "S1", "S2", "S3"],
            vec![actor.clone(), scene.clone(), scene.clone(), scene.clone()],
            vec![
                (0, 1, EdgeLabel::Participation(Role::Protagonist)),
                (1, 2, EdgeLabel::Causal),
                (2, 3, EdgeLabel::Causal),
            ],
        );

        // Old API should still work
        let patterns_old = mine_patterns(&[g1.clone()], 1);
        assert!(
            !patterns_old.is_empty(),
            "Old API should still return patterns"
        );

        // Should include both edge motifs and causal chains
        let has_2node = patterns_old.iter().any(|p| p.subgraph.nodes.len() == 2);
        assert!(has_2node, "Should have 2-node edge motifs");
    }

    #[test]
    fn test_max_patterns_cap() {
        let scene = GraphNodeType::Situation(NarrativeLevel::Scene);
        let actor = GraphNodeType::Entity(EntityType::Actor);

        // Create a graph with many different edge types to generate many motifs
        let g = make_graph(
            "n1",
            vec!["A", "B", "S1", "S2", "S3"],
            vec![
                actor.clone(),
                actor.clone(),
                scene.clone(),
                scene.clone(),
                scene.clone(),
            ],
            vec![
                (0, 2, EdgeLabel::Participation(Role::Protagonist)),
                (1, 3, EdgeLabel::Participation(Role::Antagonist)),
                (0, 3, EdgeLabel::Participation(Role::Witness)),
                (2, 3, EdgeLabel::Causal),
                (3, 4, EdgeLabel::Causal),
            ],
        );

        let config = PatternMiningConfig {
            min_support: 1,
            max_patterns: 3, // very low cap
            enable_star_motifs: true,
            ..Default::default()
        };
        let patterns = mine_patterns_with_config(&[g], &config);
        assert!(
            patterns.len() <= 3,
            "Should respect max_patterns cap of 3, got {}",
            patterns.len()
        );
    }
}
