//! Graph pathfinding algorithms: Dijkstra, Yen's K-shortest, narrative diameter, max-flow.
//!
//! These are Level 3 algorithms invoked synchronously during PATH query execution.
//! They operate on `CoGraph` (entity paths) and `CausalDag` (causal chains).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::analysis::centrality::OrdF64;
use crate::analysis::graph_projection::{self, CausalDag, CoGraph};
use crate::error::Result;

/// A weighted shortest path result.
#[derive(Debug, Clone)]
pub struct ShortestPath {
    pub path: Vec<usize>,
    pub total_weight: f64,
}

// ─── Dijkstra ──────────────────────────────────────────────────

/// Dijkstra's shortest path on a weighted co-graph.
///
/// Edge weight = inverse co-participation count (more shared situations = shorter).
/// Returns None if target is unreachable.
pub fn dijkstra(graph: &CoGraph, source: usize, target: usize) -> Option<ShortestPath> {
    let n = graph.entities.len();
    if source >= n || target >= n {
        return None;
    }

    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    dist[source] = 0.0;

    let mut heap: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((OrdF64::new(0.0), source)));

    while let Some(Reverse((OrdF64(d), u))) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        if u == target {
            break;
        }
        for &(v, w) in &graph.adj[u] {
            let edge_w = 1.0 / w as f64;
            let new_dist = dist[u] + edge_w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                prev[v] = Some(u);
                heap.push(Reverse((OrdF64::new(new_dist), v)));
            }
        }
    }

    if dist[target].is_infinite() {
        return None;
    }

    // Reconstruct path
    let mut path = vec![target];
    let mut cur = target;
    while let Some(p) = prev[cur] {
        path.push(p);
        cur = p;
    }
    path.reverse();

    Some(ShortestPath {
        path,
        total_weight: dist[target],
    })
}

/// Dijkstra with excluded edges — used as subroutine in Yen's algorithm.
fn dijkstra_excluding(
    graph: &CoGraph,
    source: usize,
    target: usize,
    excluded_nodes: &HashSet<usize>,
    excluded_edges: &HashSet<(usize, usize)>,
) -> Option<ShortestPath> {
    let n = graph.entities.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    dist[source] = 0.0;

    let mut heap: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((OrdF64::new(0.0), source)));

    while let Some(Reverse((OrdF64(d), u))) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        if u == target {
            break;
        }
        for &(v, w) in &graph.adj[u] {
            if excluded_nodes.contains(&v) {
                continue;
            }
            let edge_key = if u < v { (u, v) } else { (v, u) };
            if excluded_edges.contains(&edge_key) {
                continue;
            }
            let edge_w = 1.0 / w as f64;
            let new_dist = dist[u] + edge_w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                prev[v] = Some(u);
                heap.push(Reverse((OrdF64::new(new_dist), v)));
            }
        }
    }

    if dist[target].is_infinite() {
        return None;
    }

    let mut path = vec![target];
    let mut cur = target;
    while let Some(p) = prev[cur] {
        path.push(p);
        cur = p;
    }
    path.reverse();

    Some(ShortestPath {
        path,
        total_weight: dist[target],
    })
}

// ─── Yen's K-Shortest Paths ───────────────────────────────────

/// Find K shortest paths between source and target using Yen's algorithm.
///
/// Uses Dijkstra as subroutine. Returns up to k paths sorted by weight.
pub fn yen_k_shortest(
    graph: &CoGraph,
    source: usize,
    target: usize,
    k: usize,
) -> Vec<ShortestPath> {
    if k == 0 {
        return vec![];
    }

    let mut result: Vec<ShortestPath> = Vec::new();
    let mut candidates: BinaryHeap<Reverse<(OrdF64, Vec<usize>)>> = BinaryHeap::new();

    // Find the shortest path first
    let Some(first) = dijkstra(graph, source, target) else {
        return vec![];
    };
    result.push(first);

    for _ki in 1..k {
        let prev_path = &result.last().unwrap().path;

        for i in 0..prev_path.len().saturating_sub(1) {
            let spur_node = prev_path[i];
            let root_path = &prev_path[..=i];

            // Exclude edges that share the same root path prefix in previous results
            let mut excluded_edges: HashSet<(usize, usize)> = HashSet::new();
            for existing in &result {
                if existing.path.len() > i && existing.path[..=i] == *root_path {
                    let u = existing.path[i];
                    let v = existing.path[i + 1];
                    let edge = if u < v { (u, v) } else { (v, u) };
                    excluded_edges.insert(edge);
                }
            }

            // Exclude root path nodes (except spur node)
            let excluded_nodes: HashSet<usize> = root_path[..i].iter().copied().collect();

            if let Some(spur_path) =
                dijkstra_excluding(graph, spur_node, target, &excluded_nodes, &excluded_edges)
            {
                // Concatenate root + spur
                let mut total_path: Vec<usize> = root_path.to_vec();
                total_path.extend_from_slice(&spur_path.path[1..]); // skip spur_node duplicate

                // Compute total weight
                let mut total_weight = 0.0;
                for w in 0..total_path.len().saturating_sub(1) {
                    let u = total_path[w];
                    let v = total_path[w + 1];
                    let edge_w = graph.adj[u]
                        .iter()
                        .find(|&&(nb, _)| nb == v)
                        .map(|&(_, wt)| 1.0 / wt as f64)
                        .unwrap_or(f64::INFINITY);
                    total_weight += edge_w;
                }

                // Add to candidates if not already in results
                let is_dup = result.iter().any(|r| r.path == total_path);
                if !is_dup {
                    candidates.push(Reverse((OrdF64::new(total_weight), total_path)));
                }
            }
        }

        // Pick the best candidate
        if let Some(Reverse((OrdF64(w), path))) = candidates.pop() {
            result.push(ShortestPath {
                path,
                total_weight: w,
            });
        } else {
            break; // No more paths
        }
    }

    result
}

// ─── Narrative Diameter ────────────────────────────────────────

/// Compute the longest path in a causal DAG via DP on topological sort.
///
/// Returns (longest_path_indices, total_weight). Only works on DAGs
/// (guaranteed by topological_sort returning error on cycles).
pub fn narrative_diameter(dag: &CausalDag) -> Result<Option<ShortestPath>> {
    let n = dag.situations.len();
    if n == 0 {
        return Ok(None);
    }

    let topo = graph_projection::topological_sort(dag)?;

    let mut dist = vec![0.0_f64; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];

    // DP forward pass: dist[v] = max(dist[u] + weight(u,v)) for each predecessor u
    for &u in &topo {
        for &(v, w) in &dag.adj[u] {
            let new_dist = dist[u] + w;
            if new_dist > dist[v] {
                dist[v] = new_dist;
                prev[v] = Some(u);
            }
        }
    }

    // Find the node with maximum distance
    let (max_idx, &max_dist) = dist
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    if max_dist == 0.0 {
        return Ok(None); // No edges
    }

    // Reconstruct longest path
    let mut path = vec![max_idx];
    let mut cur = max_idx;
    while let Some(p) = prev[cur] {
        path.push(p);
        cur = p;
    }
    path.reverse();

    Ok(Some(ShortestPath {
        path,
        total_weight: max_dist,
    }))
}

// ─── Max-Flow / Min-Cut (Edmonds-Karp) ─────────────────────────

/// Compute max-flow between source and sink using Edmonds-Karp (BFS-based Ford-Fulkerson).
///
/// Edge capacities = co-participation weight. Returns (max_flow_value, min_cut_edges).
pub fn max_flow(graph: &CoGraph, source: usize, sink: usize) -> (f64, Vec<(usize, usize)>) {
    let n = graph.entities.len();
    if n == 0 || source == sink || source >= n || sink >= n {
        return (0.0, vec![]);
    }

    // Build residual graph as adjacency matrix (sparse → dense for small graphs)
    let mut capacity: HashMap<(usize, usize), f64> = HashMap::new();
    for u in 0..n {
        for &(v, w) in &graph.adj[u] {
            *capacity.entry((u, v)).or_insert(0.0) += w as f64;
        }
    }

    // Build adjacency set for BFS (includes reverse edges) — O(1) dedup
    let mut adj_set: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(u, v) in capacity.keys() {
        adj_set[u].insert(v);
        adj_set[v].insert(u);
    }
    let adj: Vec<Vec<usize>> = adj_set
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();

    let mut total_flow = 0.0;

    // BFS to find augmenting paths
    loop {
        let mut parent: Vec<Option<usize>> = vec![None; n];
        parent[source] = Some(source);
        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            if u == sink {
                break;
            }
            for &v in &adj[u] {
                let residual = capacity.get(&(u, v)).copied().unwrap_or(0.0);
                if parent[v].is_none() && residual > 0.0 {
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        if parent[sink].is_none() {
            break; // No augmenting path
        }

        // Find bottleneck
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v].unwrap();
            bottleneck = bottleneck.min(capacity.get(&(u, v)).copied().unwrap_or(0.0));
            v = u;
        }

        // Update residual capacities
        v = sink;
        while v != source {
            let u = parent[v].unwrap();
            *capacity.entry((u, v)).or_insert(0.0) -= bottleneck;
            *capacity.entry((v, u)).or_insert(0.0) += bottleneck;
            v = u;
        }

        total_flow += bottleneck;
    }

    // Find min-cut: BFS from source in residual graph to find reachable nodes
    let mut reachable = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(source);
    reachable[source] = true;
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if !reachable[v] && capacity.get(&(u, v)).copied().unwrap_or(0.0) > 0.0 {
                reachable[v] = true;
                queue.push_back(v);
            }
        }
    }

    // Min-cut edges: edges from reachable to non-reachable
    let mut cut_edges = Vec::new();
    for u in 0..n {
        if reachable[u] {
            for &(v, _) in &graph.adj[u] {
                if !reachable[v] {
                    let edge = if u < v { (u, v) } else { (v, u) };
                    if !cut_edges.contains(&edge) {
                        cut_edges.push(edge);
                    }
                }
            }
        }
    }

    (total_flow, cut_edges)
}

// ─── PCST (Prize-Collecting Steiner Tree) 2-Approximation ──────

/// Find a connected subgraph containing the prize nodes with minimum cost.
///
/// Heuristic 2-approximation: compute shortest paths between all prize pairs,
/// build a complete graph on prizes weighted by shortest-path distances,
/// find MST of that complete graph, map MST edges back to original paths.
///
/// Returns the set of node indices in the Steiner tree.
pub fn pcst_approximation(graph: &CoGraph, prize_nodes: &[usize]) -> Vec<usize> {
    if prize_nodes.len() <= 1 {
        return prize_nodes.to_vec();
    }

    // Cap prize nodes to avoid O(k^2) Dijkstra explosion
    let k = prize_nodes.len().min(50);

    // Compute shortest paths between all pairs of prize nodes
    let mut pair_paths: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut pair_weights: HashMap<(usize, usize), f64> = HashMap::new();

    for i in 0..k {
        for j in (i + 1)..k {
            if let Some(sp) = dijkstra(graph, prize_nodes[i], prize_nodes[j]) {
                let key = (i, j);
                pair_paths.insert(key, sp.path.clone());
                pair_weights.insert(key, sp.total_weight);
            }
        }
    }

    // Build MST on the complete graph of prizes using Kruskal's
    let mut edges: Vec<(f64, usize, usize)> =
        pair_weights.iter().map(|(&(i, j), &w)| (w, i, j)).collect();
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find for Kruskal's
    let mut parent: Vec<usize> = (0..k).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    let mut tree_nodes: HashSet<usize> = HashSet::new();
    for &(_, i, j) in &edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            parent[ri] = rj;
            // Add all nodes on the path between prize[i] and prize[j]
            if let Some(path) = pair_paths.get(&(i, j)) {
                tree_nodes.extend(path.iter().copied());
            }
        }
    }

    // Always include all prize nodes
    tree_nodes.extend(prize_nodes.iter().copied());
    tree_nodes.into_iter().collect()
}

#[cfg(test)]
#[path = "pathfinding_tests.rs"]
mod tests;
