//! Network centrality analysis on the entity co-participation graph.
//!
//! Implements betweenness, closeness, and degree centrality using Brandes'
//! algorithm, plus Louvain community detection. Operates on the bipartite
//! entity-situation participation graph projected onto an entity co-occurrence
//! graph where edge weight = number of shared situations.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::graph_projection::{self, CoGraph};
use crate::analysis::{analysis_key, extract_narrative_id};
use crate::error::Result;
use crate::hypergraph::{keys, Hypergraph};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

// Re-export for backward compatibility with tests
pub use crate::analysis::graph_projection::build_co_graph;

// ─── Data Structures ────────────────────────────────────────

/// Centrality results for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityResult {
    pub entity_id: Uuid,
    pub betweenness: f64,
    pub closeness: f64,
    pub degree: f64,
    pub community_id: usize,
    pub narrative_id: String,
}

/// Full centrality analysis output for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityAnalysis {
    pub narrative_id: String,
    pub results: Vec<CentralityResult>,
    pub num_communities: usize,
    pub modularity: f64,
}

// ─── Betweenness Centrality (Brandes, weighted) ─────────────

/// Compute betweenness centrality using Brandes' algorithm with Dijkstra
/// for weighted shortest paths. Edge weights are inverse co-participation
/// counts (higher co-participation = shorter distance).
fn compute_betweenness(graph: &CoGraph) -> Vec<f64> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let mut cb = vec![0.0_f64; n];

    for s in 0..n {
        // Dijkstra from s using inverse weight as distance
        let mut stack: Vec<usize> = Vec::new();
        let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0_f64; n];
        sigma[s] = 1.0;
        let mut dist: Vec<f64> = vec![f64::INFINITY; n];
        dist[s] = 0.0;

        // Min-heap: (distance, node)
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(std::cmp::Reverse((OrdF64::new(0.0), s)));

        while let Some(std::cmp::Reverse((OrdF64(d), v))) = heap.pop() {
            if d > dist[v] {
                continue; // stale entry
            }
            stack.push(v);
            for &(w, weight) in &graph.adj[v] {
                // Distance = inverse weight (more shared situations = closer)
                let edge_dist = 1.0 / (weight as f64);
                let new_dist = dist[v] + edge_dist;

                if new_dist < dist[w] - 1e-10 {
                    dist[w] = new_dist;
                    sigma[w] = sigma[v];
                    predecessors[w] = vec![v];
                    heap.push(std::cmp::Reverse((OrdF64::new(new_dist), w)));
                } else if (new_dist - dist[w]).abs() < 1e-10 && !predecessors[w].contains(&v) {
                    // Equal-length path
                    sigma[w] += sigma[v];
                    predecessors[w].push(v);
                }
            }
        }

        // Back-propagation
        let mut delta = vec![0.0_f64; n];
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // Normalize: for undirected graph, divide by 2
    let norm = if n > 2 {
        ((n - 1) * (n - 2)) as f64
    } else {
        1.0
    };
    for val in &mut cb {
        *val /= norm;
    }
    cb
}

/// Newtype for f64 with total Ord (for BinaryHeap).
/// Panics on NaN in debug mode to catch data corruption early.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct OrdF64(pub(crate) f64);

impl OrdF64 {
    pub(crate) fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan(), "NaN in OrdF64 — likely a zero-weight edge");
        Self(v)
    }
}

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── Closeness Centrality (Wasserman-Faust) ──────────────────

/// Compute closeness centrality with Wasserman-Faust normalization
/// for disconnected graphs.
///
/// For a node v in a component of size n_c within a graph of N total nodes:
/// `C(v) = ((n_c - 1) / (N - 1)) × ((n_c - 1) / farness(v))`
///
/// This correctly handles disconnected graphs by scaling closeness by
/// the component's fraction of the total graph.
fn compute_closeness(graph: &CoGraph) -> Vec<f64> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    // Compute WCC for Wasserman-Faust normalization
    let components = graph_projection::wcc(graph);

    // Map each node to its component size
    let mut component_size = vec![0usize; n];
    for component in &components {
        let size = component.len();
        for &node in component {
            component_size[node] = size;
        }
    }

    let mut closeness = vec![0.0_f64; n];
    let big_n = n as f64;

    for s in 0..n {
        // BFS
        let mut dist: Vec<i64> = vec![-1; n];
        dist[s] = 0;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            for &(w, _) in &graph.adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
            }
        }

        let total_dist: i64 = dist.iter().filter(|&&d| d > 0).sum();
        let n_c = component_size[s] as f64;

        if n_c > 1.0 && total_dist > 0 {
            // Wasserman-Faust: scale by component fraction
            let component_closeness = (n_c - 1.0) / total_dist as f64;
            let scale = (n_c - 1.0) / (big_n - 1.0);
            closeness[s] = scale * component_closeness;
        }
    }

    closeness
}

// ─── Degree Centrality ──────────────────────────────────────

/// Compute normalized degree centrality.
fn compute_degree(graph: &CoGraph) -> Vec<f64> {
    let n = graph.entities.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    let norm = (n - 1) as f64;
    graph
        .adj
        .iter()
        .map(|neighbors| neighbors.len() as f64 / norm)
        .collect()
}

// ─── Louvain Community Detection ────────────────────────────

/// Phase 1: local moves using standard Louvain ΔQ formula.
/// Returns (community_assignment, did_change).
fn louvain_phase1(adj: &[Vec<(usize, usize)>], k: &[f64], m2: f64) -> (Vec<usize>, bool) {
    let n = adj.len();
    let mut community: Vec<usize> = (0..n).collect();

    let mut sigma_tot: HashMap<usize, f64> = HashMap::new();
    for j in 0..n {
        *sigma_tot.entry(community[j]).or_insert(0.0) += k[j];
    }

    let mut any_change = false;
    let mut improved = true;
    let mut max_iterations = 100;
    while improved && max_iterations > 0 {
        improved = false;
        max_iterations -= 1;

        for i in 0..n {
            let current_comm = community[i];
            let ki = k[i];

            let mut ki_in: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                *ki_in.entry(community[j]).or_insert(0.0) += w as f64;
            }

            let ki_in_current = ki_in.get(&current_comm).copied().unwrap_or(0.0);
            let sigma_current_minus_i = sigma_tot.get(&current_comm).copied().unwrap_or(0.0) - ki;
            let remove_cost = ki_in_current / m2 - (sigma_current_minus_i * ki) / (m2 * m2);

            let mut best_comm = current_comm;
            let mut best_gain = 0.0;

            let candidate_comms: HashSet<usize> = ki_in.keys().copied().collect();
            for &c in &candidate_comms {
                if c == current_comm {
                    continue;
                }
                let ki_in_c = ki_in.get(&c).copied().unwrap_or(0.0);
                let sigma_c = sigma_tot.get(&c).copied().unwrap_or(0.0);
                let add_gain = ki_in_c / m2 - (sigma_c * ki) / (m2 * m2);
                let delta_q = add_gain - remove_cost;

                if delta_q > best_gain {
                    best_gain = delta_q;
                    best_comm = c;
                }
            }

            if best_comm != current_comm {
                *sigma_tot.entry(current_comm).or_insert(0.0) -= ki;
                *sigma_tot.entry(best_comm).or_insert(0.0) += ki;
                community[i] = best_comm;
                improved = true;
                any_change = true;
            }
        }
    }

    (community, any_change)
}

/// Renumber community labels to contiguous 0..n.
fn renumber_communities(community: &mut [usize]) -> usize {
    let mut comm_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for c in community.iter_mut() {
        let new_id = comm_map.entry(*c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *c = *new_id;
    }
    next_id
}

/// Detect communities using the Louvain algorithm with Phase 2 super-graph contraction.
///
/// Phase 1: local node moves to maximize modularity.
/// Phase 2: collapse communities into super-nodes, aggregate edges, repeat.
/// Returns (community_assignment, modularity, num_communities).
fn louvain(graph: &CoGraph) -> (Vec<usize>, f64, usize) {
    let n = graph.entities.len();
    if n == 0 {
        return (vec![], 0.0, 0);
    }

    let total_weight: f64 = graph
        .adj
        .iter()
        .flat_map(|neighbors| neighbors.iter().map(|&(_, w)| w))
        .sum::<usize>() as f64
        / 2.0;

    if total_weight == 0.0 {
        return ((0..n).collect(), 0.0, n);
    }

    let m2 = 2.0 * total_weight;

    // Working adjacency list (starts as a clone of the input graph adj)
    let mut cur_adj = graph.adj.clone();
    let mut cur_n = n;

    // Maps current super-node index back to original nodes
    let mut node_map: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let max_phases = 10;
    for _ in 0..max_phases {
        let k: Vec<f64> = cur_adj
            .iter()
            .map(|neighbors| neighbors.iter().map(|&(_, w)| w as f64).sum())
            .collect();

        let (mut phase1_comm, changed) = louvain_phase1(&cur_adj, &k, m2);
        if !changed {
            break;
        }

        let num_comm = renumber_communities(&mut phase1_comm);
        if num_comm >= cur_n {
            break; // No reduction
        }

        // Phase 2: Build super-graph by collapsing communities into super-nodes.
        // Intra-community edges become self-loops, folded into degree (k) vector.
        let mut super_edges: HashMap<(usize, usize), usize> = HashMap::new();
        let mut self_loops: Vec<usize> = vec![0; num_comm];
        for i in 0..cur_n {
            let ci = phase1_comm[i];
            for &(j, w) in &cur_adj[i] {
                if i >= j {
                    continue; // only count each undirected edge once
                }
                let cj = phase1_comm[j];
                if ci == cj {
                    // Intra-community edge → self-loop on super-node
                    self_loops[ci] += w;
                } else {
                    let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                    *super_edges.entry(key).or_insert(0) += w;
                }
            }
        }

        let mut super_adj: Vec<Vec<(usize, usize)>> = vec![vec![]; num_comm];
        for (&(ci, cj), &w) in &super_edges {
            super_adj[ci].push((cj, w));
            super_adj[cj].push((ci, w));
        }
        // Add self-loops as edges to self (counted in sigma_tot via k vector)
        for (ci, &sl) in self_loops.iter().enumerate() {
            if sl > 0 {
                super_adj[ci].push((ci, 2 * sl)); // 2× because k sums all adjacency entries
            }
        }

        // Update node_map: merge original node sets by community
        let mut new_node_map: Vec<Vec<usize>> = vec![vec![]; num_comm];
        for (old_idx, comm) in phase1_comm.iter().enumerate() {
            new_node_map[*comm].extend_from_slice(&node_map[old_idx]);
        }
        node_map = new_node_map;

        cur_adj = super_adj;
        cur_n = num_comm;
    }

    // Map final super-node communities back to original nodes
    let mut community = vec![0usize; n];
    for (super_idx, original_nodes) in node_map.iter().enumerate() {
        for &orig in original_nodes {
            community[orig] = super_idx;
        }
    }

    let num_communities = renumber_communities(&mut community);
    let k: Vec<f64> = graph
        .adj
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w as f64).sum())
        .collect();
    let modularity = compute_modularity(&community, graph, m2, &k);

    (community, modularity, num_communities)
}

// ─── Leiden Community Detection ────────────────────────────

/// Detect communities using the Leiden algorithm (Louvain + refinement).
///
/// The refinement step guarantees that all detected communities are
/// connected subgraphs, fixing a known weakness of plain Louvain.
/// Returns (community_assignment, modularity, num_communities).
pub fn leiden(graph: &CoGraph) -> (Vec<usize>, f64, usize) {
    // Phase 1: Run standard Louvain local moves
    let (mut community, _, _) = louvain(graph);
    let n = graph.entities.len();
    if n == 0 {
        return (vec![], 0.0, 0);
    }

    // Phase 2: Refinement — check connectivity within each community.
    // For each community, run BFS to find connected components.
    // If a community has multiple components, split them.
    let mut comm_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in community.iter().enumerate() {
        comm_nodes.entry(c).or_default().push(i);
    }

    let mut next_comm_id = *community.iter().max().unwrap_or(&0) + 1;
    for (_comm, nodes) in &comm_nodes {
        if nodes.len() <= 1 {
            continue;
        }
        // BFS to find connected components within this community
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components: Vec<Vec<usize>> = Vec::new();

        for &start in nodes {
            if visited.contains(&start) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited.insert(start);
            while let Some(node) = queue.pop_front() {
                component.push(node);
                for &(neighbor, _) in &graph.adj[node] {
                    if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(component);
        }

        // If multiple components, assign each to a new community ID
        // Keep the largest component in the original community
        if components.len() > 1 {
            // Sort by size descending — largest keeps the original ID
            let mut sorted: Vec<(usize, Vec<usize>)> = components.into_iter().enumerate().collect();
            sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
            // Skip the first (largest) — it keeps the original community
            for (_idx, component) in sorted.into_iter().skip(1) {
                for node in component {
                    community[node] = next_comm_id;
                }
                next_comm_id += 1;
            }
        }
    }

    // Renumber communities to be contiguous
    let mut comm_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for c in &mut community {
        let new_id = comm_map.entry(*c).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *c = *new_id;
    }
    let num_communities = next_id;

    // Compute total weight for modularity
    let total_weight: f64 = graph
        .adj
        .iter()
        .flat_map(|neighbors| neighbors.iter().map(|&(_, w)| w))
        .sum::<usize>() as f64
        / 2.0;

    if total_weight == 0.0 {
        return (community, 0.0, num_communities);
    }

    let m2 = 2.0 * total_weight;
    let k: Vec<f64> = graph
        .adj
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w as f64).sum())
        .collect();
    let modularity = compute_modularity(&community, graph, m2, &k);

    (community, modularity, num_communities)
}

/// A single level of community hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityLevel {
    /// Level in the hierarchy (0 = most granular / leaf).
    pub level: usize,
    /// Community assignments: entity index → community ID at this level.
    pub assignments: Vec<usize>,
    /// Number of communities at this level.
    pub num_communities: usize,
    /// Modularity at this level.
    pub modularity: f64,
}

/// Run hierarchical Leiden: recursively subdivides communities larger than
/// `max_community_size` to produce a multi-level hierarchy.
///
/// Returns levels ordered from most granular (level 0) to coarsest (highest level).
pub fn hierarchical_leiden(graph: &CoGraph, max_community_size: usize) -> Vec<CommunityLevel> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    // Level 0: run Leiden on the full graph
    let (base_communities, base_mod, base_num) = leiden(graph);

    let mut levels = vec![CommunityLevel {
        level: 0,
        assignments: base_communities.clone(),
        num_communities: base_num,
        modularity: base_mod,
    }];

    // Check if any community exceeds the threshold
    let mut comm_sizes: HashMap<usize, usize> = HashMap::new();
    for &c in &base_communities {
        *comm_sizes.entry(c).or_insert(0) += 1;
    }

    let needs_subdivision = comm_sizes.values().any(|&size| size > max_community_size);
    if !needs_subdivision || base_num <= 1 {
        return levels;
    }

    // Build a coarsened "super-graph" where each community becomes a node.
    // Edge weight = total inter-community edge weight.
    let mut super_edges: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..n {
        let ci = base_communities[i];
        for &(j, w) in &graph.adj[i] {
            let cj = base_communities[j];
            if ci != cj {
                let key = if ci < cj { (ci, cj) } else { (cj, ci) };
                *super_edges.entry(key).or_insert(0) += w;
            }
        }
    }

    // Build super-graph adjacency
    let mut super_adj: Vec<Vec<(usize, usize)>> = vec![vec![]; base_num];
    for (&(ci, cj), &w) in &super_edges {
        super_adj[ci].push((cj, w));
        super_adj[cj].push((ci, w));
    }

    let super_graph = CoGraph {
        entities: (0..base_num).map(|_| Uuid::nil()).collect(),
        adj: super_adj,
    };

    // Run Leiden on the super-graph
    let (super_communities, super_mod, super_num) = leiden(&super_graph);

    // Map back to original entities: each entity's level-1 community =
    // super_communities[base_communities[i]]
    let level1_assignments: Vec<usize> = (0..n)
        .map(|i| super_communities[base_communities[i]])
        .collect();

    levels.push(CommunityLevel {
        level: 1,
        assignments: level1_assignments,
        num_communities: super_num,
        modularity: super_mod,
    });

    levels
}

fn compute_modularity(community: &[usize], graph: &CoGraph, m2: f64, k: &[f64]) -> f64 {
    let mut q = 0.0;
    for i in 0..community.len() {
        for &(j, w) in &graph.adj[i] {
            if community[i] == community[j] {
                q += w as f64 - k[i] * k[j] / m2;
            }
        }
    }
    q / m2
}

// ─── InferenceEngine Implementation ─────────────────────────

/// Centrality analysis engine.
pub struct CentralityEngine;

impl InferenceEngine for CentralityEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CentralityAnalysis
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000) // 5 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let analysis = run_centrality(hypergraph, narrative_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::CentralityAnalysis,
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence: 1.0,
            explanation: Some("Network centrality analysis complete".into()),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Run centrality analysis on a narrative and return results.
///
/// Uses the Leiden algorithm for community detection (guarantees connected communities).
/// Also computes hierarchical community levels and stores them in KV.
pub fn run_centrality(hypergraph: &Hypergraph, narrative_id: &str) -> Result<CentralityAnalysis> {
    let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;

    if graph.entities.is_empty() {
        return Ok(CentralityAnalysis {
            narrative_id: narrative_id.to_string(),
            results: vec![],
            num_communities: 0,
            modularity: 0.0,
        });
    }

    let betweenness = compute_betweenness(&graph);
    let closeness = compute_closeness(&graph);
    let degree = compute_degree(&graph);
    // Use Leiden (Louvain + refinement) for base community detection
    let (communities, modularity, num_communities) = leiden(&graph);

    // Compute hierarchical levels (max community size = 10)
    let hierarchy = hierarchical_leiden(&graph, 10);
    // Store hierarchy in KV for later use by DRIFT search and community browsing
    let hierarchy_key = format!("an/ch/{}", narrative_id);
    let hierarchy_bytes = serde_json::to_vec(&hierarchy)?;
    hypergraph
        .store()
        .put(hierarchy_key.as_bytes(), &hierarchy_bytes)?;

    let results: Vec<CentralityResult> = graph
        .entities
        .iter()
        .enumerate()
        .map(|(i, &eid)| CentralityResult {
            entity_id: eid,
            betweenness: betweenness[i],
            closeness: closeness[i],
            degree: degree[i],
            community_id: communities[i],
            narrative_id: narrative_id.to_string(),
        })
        .collect();

    // Store results in KV
    for result in &results {
        let key = analysis_key(
            keys::ANALYSIS_CENTRALITY,
            &[narrative_id, &result.entity_id.to_string()],
        );
        let bytes = serde_json::to_vec(result)?;
        hypergraph.store().put(&key, &bytes)?;
    }

    Ok(CentralityAnalysis {
        narrative_id: narrative_id.to_string(),
        results,
        num_communities,
        modularity,
    })
}

#[cfg(test)]
#[path = "centrality_tests.rs"]
mod tests;
