//! Link prediction functions for co-participation graphs.
//!
//! All functions operate on `CoGraph` and compute synchronously during query execution.
//! These are Level 2 inline graph functions — no job queue, no KV storage.

use crate::analysis::graph_projection::CoGraph;

/// Common neighbors: `|N(a) ∩ N(b)|`.
pub fn common_neighbors(graph: &CoGraph, a: usize, b: usize) -> u64 {
    let na = graph.neighbor_set(a);
    let nb = graph.neighbor_set(b);
    na.intersection(&nb).count() as u64
}

/// Adamic-Adar: `Σ_{w ∈ N(a) ∩ N(b)} 1 / log(|N(w)|)`.
/// Weights rare shared neighbors higher.
pub fn adamic_adar(graph: &CoGraph, a: usize, b: usize) -> f64 {
    let na = graph.neighbor_set(a);
    let nb = graph.neighbor_set(b);
    na.intersection(&nb)
        .map(|&w| {
            let deg = graph.adj[w].len();
            if deg > 1 {
                1.0 / (deg as f64).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Preferential attachment: `|N(a)| × |N(b)|`. O(1).
pub fn preferential_attachment(graph: &CoGraph, a: usize, b: usize) -> u64 {
    (graph.adj[a].len() as u64) * (graph.adj[b].len() as u64)
}

/// Resource allocation: `Σ_{w ∈ N(a) ∩ N(b)} 1 / |N(w)|`.
pub fn resource_allocation(graph: &CoGraph, a: usize, b: usize) -> f64 {
    let na = graph.neighbor_set(a);
    let nb = graph.neighbor_set(b);
    na.intersection(&nb)
        .map(|&w| {
            let deg = graph.adj[w].len();
            if deg > 0 {
                1.0 / deg as f64
            } else {
                0.0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_common_neighbors() {
        // Triangle: each pair shares 1 common neighbor
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);
        let graph = crate::analysis::graph_projection::build_co_graph(&hg, "n1").unwrap();
        let ai = graph.entities.iter().position(|&e| e == a).unwrap();
        let bi = graph.entities.iter().position(|&e| e == b).unwrap();
        assert_eq!(common_neighbors(&graph, ai, bi), 1);
    }

    #[test]
    fn test_adamic_adar() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);
        let graph = crate::analysis::graph_projection::build_co_graph(&hg, "n1").unwrap();
        let ai = graph.entities.iter().position(|&e| e == a).unwrap();
        let bi = graph.entities.iter().position(|&e| e == b).unwrap();
        let score = adamic_adar(&graph, ai, bi);
        assert!(score > 0.0, "AA should be positive for triangle");
    }

    #[test]
    fn test_preferential_attachment() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);
        let graph = crate::analysis::graph_projection::build_co_graph(&hg, "n1").unwrap();
        let ai = graph.entities.iter().position(|&e| e == a).unwrap();
        let bi = graph.entities.iter().position(|&e| e == b).unwrap();
        // Both have degree 2
        assert_eq!(preferential_attachment(&graph, ai, bi), 4);
    }

    #[test]
    fn test_resource_allocation() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);
        let graph = crate::analysis::graph_projection::build_co_graph(&hg, "n1").unwrap();
        let ai = graph.entities.iter().position(|&e| e == a).unwrap();
        let bi = graph.entities.iter().position(|&e| e == b).unwrap();
        let score = resource_allocation(&graph, ai, bi);
        // Shared neighbor C has degree 2: 1/2 = 0.5
        assert!((score - 0.5).abs() < 0.01);
    }
}
