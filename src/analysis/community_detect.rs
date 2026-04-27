//! Community detection algorithms beyond Leiden.
//!
//! - **Label Propagation**: Fast, parameter-free community detection.
//!   O(m) per iteration. Store community label at `an/lp/{narrative_id}/{entity_id}`.

use crate::analysis::graph_projection::{self, CoGraph};
use crate::analysis::{extract_narrative_id, make_engine_result, store_entity_scores};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult};

// ─── Label Propagation ─────────────────────────────────────────

const LABEL_PROP_MAX_ITER: usize = 100;

/// Run label propagation community detection on the co-graph.
///
/// Each node starts with a unique label. In each iteration, every node adopts
/// the most common label among its weighted neighbors. Ties are broken using
/// a seeded deterministic rule (lowest label wins).
///
/// Returns community label per node (contiguous 0..k).
pub fn label_propagation(graph: &CoGraph, seed: u64) -> Vec<usize> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let mut labels: Vec<usize> = (0..n).collect();

    // Deterministic iteration order from seed (simple LCG shuffle)
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng_state = seed.wrapping_add(1);

    for _ in 0..LABEL_PROP_MAX_ITER {
        // Shuffle order deterministically
        for i in (1..n).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        let mut changed = false;
        for &i in &order {
            if graph.adj[i].is_empty() {
                continue;
            }

            // Count weighted votes for each neighbor label
            let mut votes: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &(j, w) in &graph.adj[i] {
                *votes.entry(labels[j]).or_insert(0) += w;
            }

            // Find the label with maximum weight (break ties: lowest label)
            let best_label = votes
                .into_iter()
                .max_by(|(la, wa), (lb, wb)| wa.cmp(wb).then(lb.cmp(la)))
                .map(|(label, _)| label)
                .unwrap_or(labels[i]);

            if best_label != labels[i] {
                labels[i] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Renumber labels to contiguous 0..k
    let mut label_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next_id = 0;
    for label in &mut labels {
        let new_id = label_map.entry(*label).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *label = *new_id;
    }

    labels
}

/// Label Propagation inference engine.
pub struct LabelPropagationEngine;

impl InferenceEngine for LabelPropagationEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::LabelPropagation
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(2000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;

        // Use a fixed seed for deterministic results (can be overridden via params)
        let seed = job
            .parameters
            .get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(42);

        let labels = label_propagation(&graph, seed);

        let scores: Vec<f64> = labels.iter().map(|&l| l as f64).collect();
        store_entity_scores(
            hypergraph,
            &graph.entities,
            &scores,
            b"an/lp/",
            narrative_id,
        )?;

        let result_map: Vec<serde_json::Value> = graph
            .entities
            .iter()
            .enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "label": labels[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::LabelPropagation,
            narrative_id,
            result_map,
            "Label propagation community detection complete",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_label_propagation_two_clusters() {
        // Two disconnected pairs should get different labels
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let d = add_entity(&hg, "D", "n1");
        let s1 = add_situation(&hg, "n1");
        let s2 = add_situation(&hg, "n1");
        link(&hg, a, s1);
        link(&hg, b, s1);
        link(&hg, c, s2);
        link(&hg, d, s2);

        let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
        let labels = label_propagation(&graph, 42);
        let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
        let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();
        let c_idx = graph.entities.iter().position(|&e| e == c).unwrap();
        assert_eq!(
            labels[a_idx], labels[b_idx],
            "A and B should be in same community"
        );
        assert_ne!(
            labels[a_idx], labels[c_idx],
            "A and C should be in different communities"
        );
    }

    #[test]
    fn test_label_propagation_single_node() {
        let hg = make_hg();
        let _a = add_entity(&hg, "A", "n1");
        let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
        let labels = label_propagation(&graph, 42);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], 0);
    }

    #[test]
    fn test_label_propagation_deterministic_seed() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let c = add_entity(&hg, "C", "n1");
        let s1 = add_situation(&hg, "n1");
        let s2 = add_situation(&hg, "n1");
        link(&hg, a, s1);
        link(&hg, b, s1);
        link(&hg, b, s2);
        link(&hg, c, s2);

        let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
        let labels1 = label_propagation(&graph, 42);
        let labels2 = label_propagation(&graph, 42);
        assert_eq!(labels1, labels2, "Same seed should produce same labels");
    }
}
