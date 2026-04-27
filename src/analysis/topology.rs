//! Structural topology analysis: articulation points, bridges, and k-core decomposition.
//!
//! - **INFER TOPOLOGY** runs articulation points + bridges together (single job).
//!   Results stored at `an/tp/{narrative_id}/{entity_id}` as `TopologyResult` JSON.
//! - **INFER KCORE** computes k-core decomposition.
//!   Results stored at `an/kc/{narrative_id}/{entity_id}` as scalar core number.

use serde::{Deserialize, Serialize};

use crate::analysis::graph_projection::{self, CoGraph};
use crate::analysis::{
    analysis_key, extract_narrative_id, make_engine_result, store_entity_scores,
};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult};

// ─── Articulation Points & Bridges ─────────────────────────────

/// Per-entity topology flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyResult {
    pub is_articulation_point: bool,
    pub is_bridge_endpoint: bool,
}

/// Find articulation points and bridges in an undirected co-graph using
/// iterative Tarjan's DFS algorithm.
///
/// Returns (is_articulation_point[i], bridge_edges) where bridge_edges
/// are pairs of node indices.
pub fn find_articulation_points_and_bridges(graph: &CoGraph) -> (Vec<bool>, Vec<(usize, usize)>) {
    let n = graph.entities.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let mut disc = vec![0u32; n];
    let mut low = vec![0u32; n];
    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    let mut is_ap = vec![false; n];
    let mut bridges: Vec<(usize, usize)> = Vec::new();
    let mut timer: u32 = 1;

    // Iterative DFS to avoid stack overflow on large graphs
    for start in 0..n {
        if visited[start] {
            continue;
        }

        // Stack entries: (node, neighbor_iterator_index, is_root)
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        visited[start] = true;
        disc[start] = timer;
        low[start] = timer;
        timer += 1;
        let mut root_children = 0u32;

        while let Some(&mut (u, ref mut ni)) = stack.last_mut() {
            let neighbors = &graph.adj[u];
            if *ni < neighbors.len() {
                let (v, _) = neighbors[*ni];
                *ni += 1;

                if !visited[v] {
                    visited[v] = true;
                    disc[v] = timer;
                    low[v] = timer;
                    timer += 1;
                    parent[v] = u;

                    if u == start {
                        root_children += 1;
                    }

                    stack.push((v, 0));
                } else if v != parent[u] {
                    // Back edge — update low
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                // All neighbors processed — backtrack
                stack.pop();
                if let Some(&(p, _)) = stack.last() {
                    low[p] = low[p].min(low[u]);

                    // Bridge check: edge (p, u) is a bridge if low[u] > disc[p]
                    if low[u] > disc[p] {
                        let bridge = if p < u { (p, u) } else { (u, p) };
                        bridges.push(bridge);
                    }

                    // Articulation point check (non-root)
                    if parent[p] != usize::MAX && low[u] >= disc[p] {
                        is_ap[p] = true;
                    }
                }
            }
        }

        // Root is an articulation point if it has 2+ children in DFS tree
        if root_children >= 2 {
            is_ap[start] = true;
        }
    }

    (is_ap, bridges)
}

/// Which graph projection to run articulation-point / bridge detection on.
///
/// Default (`Cooccurrence`) is the participation co-graph — two entities edge
/// when they share at least one situation. In dense late-stage narratives
/// (e.g. post-arrest raids with many co-participants), this graph saturates
/// and articulation points become meaningless.
///
/// `Causal` builds an entity-level graph from the situation-level causal DAG
/// — two entities edge when one participates in a cause-situation whose
/// consequence-situation the other participates in. Sparser, surfaces actors
/// who actually bridge distinct causal chains rather than those who happen
/// to share a meeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TopologyProjection {
    #[default]
    Cooccurrence,
    Causal,
}

impl TopologyProjection {
    fn from_job(job: &InferenceJob) -> Self {
        match job
            .parameters
            .get("projection")
            .and_then(|v| v.as_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("causal") => Self::Causal,
            _ => Self::Cooccurrence,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cooccurrence => "cooccurrence",
            Self::Causal => "causal",
        }
    }
}

/// INFER TOPOLOGY engine — runs articulation points + bridges together.
///
/// Accepts an optional `projection` parameter (`"cooccurrence"` default, or
/// `"causal"`) to switch the underlying graph — see `TopologyProjection`.
pub struct TopologyEngine;

impl InferenceEngine for TopologyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::Topology
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let projection = TopologyProjection::from_job(job);
        let graph = match projection {
            TopologyProjection::Cooccurrence => {
                graph_projection::build_co_graph(hypergraph, narrative_id)?
            }
            TopologyProjection::Causal => {
                graph_projection::build_causal_entity_graph(hypergraph, narrative_id)?
            }
        };
        let (is_ap, bridges) = find_articulation_points_and_bridges(&graph);

        // Mark bridge endpoints
        let mut is_bridge_endpoint = vec![false; graph.entities.len()];
        for &(a, b) in &bridges {
            is_bridge_endpoint[a] = true;
            is_bridge_endpoint[b] = true;
        }

        // Store per-entity topology results
        for (i, &eid) in graph.entities.iter().enumerate() {
            let result = TopologyResult {
                is_articulation_point: is_ap[i],
                is_bridge_endpoint: is_bridge_endpoint[i],
            };
            let key = analysis_key(b"an/tp/", &[narrative_id, &eid.to_string()]);
            let bytes = serde_json::to_vec(&result)?;
            hypergraph.store().put(&key, &bytes)?;
        }

        let result_map: Vec<serde_json::Value> = graph
            .entities
            .iter()
            .enumerate()
            .map(|(i, eid)| {
                serde_json::json!({
                    "entity_id": eid,
                    "is_articulation_point": is_ap[i],
                    "is_bridge_endpoint": is_bridge_endpoint[i],
                })
            })
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::Topology,
            narrative_id,
            result_map,
            &format!(
                "Topology analysis complete ({} projection, {} entities, {} bridges)",
                projection.label(),
                graph.entities.len(),
                bridges.len(),
            ),
        ))
    }
}

// ─── K-Core Decomposition ──────────────────────────────────────

/// Compute k-core decomposition: iteratively remove nodes with degree < k.
///
/// Returns the core number for each node (the maximum k such that the node
/// belongs to the k-core).
pub fn compute_kcore(graph: &CoGraph) -> Vec<usize> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    // Compute initial degrees (unweighted)
    let mut degree: Vec<usize> = graph.adj.iter().map(|nb| nb.len()).collect();
    let mut core = vec![0usize; n];
    let mut removed = vec![false; n];
    let mut remaining = n;

    // Iterative peeling: remove nodes with minimum degree
    let mut k = 0;
    while remaining > 0 {
        // Find the minimum degree among remaining nodes
        let min_deg = (0..n)
            .filter(|&i| !removed[i])
            .map(|i| degree[i])
            .min()
            .unwrap_or(0);

        k = k.max(min_deg);

        // Collect all nodes with degree <= k among remaining
        let mut to_remove: Vec<usize> = (0..n).filter(|&i| !removed[i] && degree[i] <= k).collect();

        while let Some(u) = to_remove.pop() {
            if removed[u] {
                continue;
            }
            removed[u] = true;
            core[u] = k;
            remaining -= 1;

            // Reduce degree of remaining neighbors
            for &(v, _) in &graph.adj[u] {
                if !removed[v] {
                    degree[v] = degree[v].saturating_sub(1);
                    if degree[v] <= k {
                        to_remove.push(v);
                    }
                }
            }
        }
    }

    core
}

/// K-Core decomposition engine.
pub struct KCoreEngine;

impl InferenceEngine for KCoreEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::KCore
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
        let cores = compute_kcore(&graph);

        // Store as f64 for consistency with other scalar stores
        let scores: Vec<f64> = cores.iter().map(|&c| c as f64).collect();
        store_entity_scores(
            hypergraph,
            &graph.entities,
            &scores,
            b"an/kc/",
            narrative_id,
        )?;

        let result_map: Vec<serde_json::Value> = graph
            .entities
            .iter()
            .enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "kcore": cores[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::KCore,
            narrative_id,
            result_map,
            "K-Core decomposition complete",
        ))
    }
}

// ─── Inline Graph Functions (Level 2) ──────────────────────────

/// Count triangles containing node `node_idx`. O(d²) per node.
pub fn triangles(graph: &CoGraph, node_idx: usize) -> u64 {
    let neighbors: std::collections::HashSet<usize> =
        graph.adj[node_idx].iter().map(|&(v, _)| v).collect();
    let mut count = 0u64;
    let nb_vec: Vec<usize> = neighbors.iter().copied().collect();
    for i in 0..nb_vec.len() {
        for j in (i + 1)..nb_vec.len() {
            if graph.adj[nb_vec[i]].iter().any(|&(v, _)| v == nb_vec[j]) {
                count += 1;
            }
        }
    }
    count
}

/// Local clustering coefficient for node `node_idx`.
/// `edges_among_neighbors / (degree * (degree-1) / 2)`.
pub fn clustering_coefficient(graph: &CoGraph, node_idx: usize) -> f64 {
    let degree = graph.adj[node_idx].len();
    if degree < 2 {
        return 0.0;
    }
    let tri = triangles(graph, node_idx);
    let possible = (degree * (degree - 1)) / 2;
    tri as f64 / possible as f64
}

#[cfg(test)]
#[path = "topology_tests.rs"]
mod tests;
