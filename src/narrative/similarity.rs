//! Graph similarity scoring using kernel methods.
//!
//! Provides Weisfeiler-Leman subtree kernel and random walk kernel
//! for computing similarity between narrative graphs.

use std::collections::HashMap;

use super::subgraph::NarrativeGraph;

// ─── Weisfeiler-Leman Kernel ─────────────────────────────────

/// Compute the Weisfeiler-Leman subtree kernel between two graphs.
///
/// Iteratively refines node labels by incorporating neighbor labels,
/// then computes the inner product of label histograms. Returns a
/// similarity score in [0, 1].
pub fn wl_kernel(g1: &NarrativeGraph, g2: &NarrativeGraph, iterations: usize) -> f64 {
    if g1.nodes.is_empty() && g2.nodes.is_empty() {
        return 1.0;
    }
    if g1.nodes.is_empty() || g2.nodes.is_empty() {
        return 0.0;
    }

    let mut labels1: Vec<String> = g1.nodes.iter().map(|n| n.label.clone()).collect();
    let mut labels2: Vec<String> = g2.nodes.iter().map(|n| n.label.clone()).collect();

    let mut total_similarity = 0.0;
    let num_rounds = iterations + 1;

    for _ in 0..num_rounds {
        // Build histograms
        let hist1 = build_histogram(&labels1);
        let hist2 = build_histogram(&labels2);
        total_similarity += histogram_similarity(&hist1, &hist2);

        // Refine labels
        labels1 = refine_labels(&labels1, &g1.adjacency);
        labels2 = refine_labels(&labels2, &g2.adjacency);
    }

    total_similarity / num_rounds as f64
}

fn build_histogram(labels: &[String]) -> HashMap<String, usize> {
    let mut hist = HashMap::new();
    for label in labels {
        *hist.entry(label.clone()).or_insert(0) += 1;
    }
    hist
}

fn histogram_similarity(h1: &HashMap<String, usize>, h2: &HashMap<String, usize>) -> f64 {
    // Cosine similarity of histograms
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    let all_keys: std::collections::HashSet<&String> = h1.keys().chain(h2.keys()).collect();

    for key in all_keys {
        let v1 = *h1.get(key).unwrap_or(&0) as f64;
        let v2 = *h2.get(key).unwrap_or(&0) as f64;
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot / (norm1.sqrt() * norm2.sqrt())
}

fn refine_labels(
    labels: &[String],
    adjacency: &[Vec<(usize, super::subgraph::EdgeLabel)>],
) -> Vec<String> {
    labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let mut neighbor_labels: Vec<String> = adjacency[i]
                .iter()
                .map(|(neighbor, edge_label)| format!("{}:{:?}", labels[*neighbor], edge_label))
                .collect();
            neighbor_labels.sort();
            format!("{}|{}", label, neighbor_labels.join(","))
        })
        .collect()
}

// ─── Random Walk Kernel ──────────────────────────────────────

/// Compute random walk similarity between two graphs.
///
/// Samples random walks from both graphs and compares the
/// label sequences. Returns similarity in [0, 1].
pub fn random_walk_similarity(
    g1: &NarrativeGraph,
    g2: &NarrativeGraph,
    walk_length: usize,
    n_walks: usize,
) -> f64 {
    if g1.nodes.is_empty() || g2.nodes.is_empty() {
        return 0.0;
    }

    let walks1 = sample_walks(g1, walk_length, n_walks);
    let walks2 = sample_walks(g2, walk_length, n_walks);

    if walks1.is_empty() || walks2.is_empty() {
        return 0.0;
    }

    // Compare walk label sequences using set overlap
    let set1: std::collections::HashSet<String> = walks1.into_iter().collect();
    let set2: std::collections::HashSet<String> = walks2.into_iter().collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

/// Sample random walks from a graph. Uses a deterministic pseudo-random
/// sequence based on node indices for reproducibility.
fn sample_walks(graph: &NarrativeGraph, walk_length: usize, n_walks: usize) -> Vec<String> {
    let mut walks = Vec::new();
    let n = graph.nodes.len();
    if n == 0 {
        return walks;
    }

    for walk_idx in 0..n_walks {
        let start = walk_idx % n;
        let mut current = start;
        let mut path = vec![graph.nodes[current].label.clone()];

        for step in 0..walk_length {
            let neighbors = &graph.adjacency[current];
            if neighbors.is_empty() {
                break;
            }
            // Deterministic "random" selection
            let next_idx = (walk_idx * 7 + step * 13 + current * 3) % neighbors.len();
            current = neighbors[next_idx].0;
            path.push(graph.nodes[current].label.clone());
        }

        walks.push(path.join("->"));
    }
    walks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrative::subgraph::*;
    use crate::types::*;
    use uuid::Uuid;

    fn make_graph(labels: Vec<&str>, edges: Vec<(usize, usize, EdgeLabel)>) -> NarrativeGraph {
        let nodes: Vec<GraphNode> = labels
            .iter()
            .map(|l| GraphNode {
                id: Uuid::now_v7(),
                node_type: GraphNodeType::Entity(EntityType::Actor),
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
            narrative_id: "test".to_string(),
            nodes,
            adjacency,
        }
    }

    #[test]
    fn test_wl_identical_graphs() {
        let g = make_graph(
            vec!["A", "B", "C"],
            vec![(0, 1, EdgeLabel::Causal), (1, 2, EdgeLabel::Causal)],
        );
        let score = wl_kernel(&g, &g, 3);
        assert!((score - 1.0).abs() < 0.01, "score={}", score);
    }

    #[test]
    fn test_wl_disjoint_graphs() {
        let g1 = make_graph(vec!["A", "B"], vec![(0, 1, EdgeLabel::Causal)]);
        let g2 = make_graph(vec!["X", "Y"], vec![(0, 1, EdgeLabel::Causal)]);
        let score = wl_kernel(&g1, &g2, 3);
        // Different labels → low similarity
        assert!(score < 0.5, "score={}", score);
    }

    #[test]
    fn test_wl_empty_graphs() {
        let g1 = NarrativeGraph {
            narrative_id: "a".to_string(),
            nodes: vec![],
            adjacency: vec![],
        };
        let g2 = NarrativeGraph {
            narrative_id: "b".to_string(),
            nodes: vec![],
            adjacency: vec![],
        };
        assert_eq!(wl_kernel(&g1, &g2, 3), 1.0);
    }

    #[test]
    fn test_wl_one_empty() {
        let g1 = make_graph(vec!["A"], vec![]);
        let g2 = NarrativeGraph {
            narrative_id: "b".to_string(),
            nodes: vec![],
            adjacency: vec![],
        };
        assert_eq!(wl_kernel(&g1, &g2, 3), 0.0);
    }

    #[test]
    fn test_wl_same_labels_different_structure() {
        let g1 = make_graph(
            vec!["A", "B", "C"],
            vec![(0, 1, EdgeLabel::Causal), (1, 2, EdgeLabel::Causal)],
        );
        let g2 = make_graph(vec!["A", "B", "C"], vec![(0, 2, EdgeLabel::Causal)]);
        let score = wl_kernel(&g1, &g2, 3);
        // Same labels, different edges → intermediate similarity
        assert!(score > 0.0 && score < 1.0, "score={}", score);
    }

    #[test]
    fn test_random_walk_empty() {
        let g1 = NarrativeGraph {
            narrative_id: "a".to_string(),
            nodes: vec![],
            adjacency: vec![],
        };
        let g2 = make_graph(vec!["A"], vec![]);
        assert_eq!(random_walk_similarity(&g1, &g2, 3, 5), 0.0);
    }

    #[test]
    fn test_random_walk_identical() {
        let g = make_graph(vec!["A", "B"], vec![(0, 1, EdgeLabel::Causal)]);
        let score = random_walk_similarity(&g, &g, 3, 5);
        assert!(score > 0.5, "score={}", score);
    }

    #[test]
    fn test_histogram_similarity_identical() {
        let h = HashMap::from([("A".to_string(), 2), ("B".to_string(), 1)]);
        let score = histogram_similarity(&h, &h);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_similarity_disjoint() {
        let h1 = HashMap::from([("A".to_string(), 1)]);
        let h2 = HashMap::from([("B".to_string(), 1)]);
        let score = histogram_similarity(&h1, &h2);
        assert_eq!(score, 0.0);
    }
}
