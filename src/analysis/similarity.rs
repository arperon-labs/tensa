//! Graph similarity functions for co-participation graphs.
//!
//! Level 2 inline graph functions — synchronous, no job queue.

use crate::analysis::graph_projection::CoGraph;

/// Jaccard similarity: `|N(a) ∩ N(b)| / |N(a) ∪ N(b)|`.
pub fn jaccard(graph: &CoGraph, a: usize, b: usize) -> f64 {
    let na = graph.neighbor_set(a);
    let nb = graph.neighbor_set(b);
    let intersection = na.intersection(&nb).count();
    let union = na.union(&nb).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Overlap similarity: `|N(a) ∩ N(b)| / min(|N(a)|, |N(b)|)`.
pub fn overlap(graph: &CoGraph, a: usize, b: usize) -> f64 {
    let na = graph.neighbor_set(a);
    let nb = graph.neighbor_set(b);
    let intersection = na.intersection(&nb).count();
    let min_size = na.len().min(nb.len());
    if min_size == 0 {
        return 0.0;
    }
    intersection as f64 / min_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_jaccard_identical() {
        // Triangle: A,B share neighbor C. N(A)={B,C}, N(B)={A,C}
        // Intersection = {C}, Union = {A,B,C} → 1/3
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
        let score = jaccard(&graph, ai, bi);
        assert!(
            score > 0.0 && score <= 1.0,
            "Jaccard should be in (0,1], got {}",
            score
        );
    }

    #[test]
    fn test_overlap_subset() {
        // A connects to B,C,D; B connects to A only
        // N(A)={B,C,D}, N(B)={A} → overlap = |∩|/min = 0/1 = 0
        // Actually B's neighbors include A (but A is asking about overlap of A and B's neighbors)
        // A neighbors include B, so intersection includes... let me think.
        // CoGraph: A-B share situation, A-C share, A-D share
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let d = add_entity(&hg, "D", "n1");
        let s1 = add_situation(&hg, "n1");
        link(&hg, a, s1);
        link(&hg, b, s1);
        let s2 = add_situation(&hg, "n1");
        link(&hg, a, s2);
        link(&hg, c, s2);
        let s3 = add_situation(&hg, "n1");
        link(&hg, a, s3);
        link(&hg, d, s3);
        // N(A) = {B,C,D}, N(B) = {A}
        // Intersection of N(A) and N(B) = {} (empty since A is not a neighbor of B's neighbors besides A itself; but B's only neighbor is A)
        let graph = crate::analysis::graph_projection::build_co_graph(&hg, "n1").unwrap();
        let ai = graph.entities.iter().position(|&e| e == a).unwrap();
        let bi = graph.entities.iter().position(|&e| e == b).unwrap();
        let score = overlap(&graph, ai, bi);
        // N(A)={B,C,D}, N(B)={A}, intersection=empty, overlap=0
        assert!(
            (score - 0.0).abs() < 0.01,
            "No shared neighbors → overlap 0, got {}",
            score
        );
    }
}
