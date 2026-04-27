//! Graph embedding algorithms: FastRP and Node2Vec.
//!
//! Produce entity embeddings capturing graph structure (co-participation topology).
//! FastRP: sparse random projection (fast, no learning). Node2Vec: biased random
//! walks → PMI matrix → truncated SVD.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::graph_projection::{build_co_graph, CoGraph};
use crate::analysis::{analysis_key, extract_narrative_id, make_engine_result};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

// ─── Key prefixes ──────────────────────────────────────────

const FASTRP_PREFIX: &[u8] = b"an/frp/";
const NODE2VEC_PREFIX: &[u8] = b"an/n2v/";

// ─── FastRP ────────────────────────────────────────────────

/// FastRP embedding result for one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastRPResult {
    pub entity_id: Uuid,
    pub embedding: Vec<f32>,
}

/// Compute FastRP embeddings for all entities in a narrative.
///
/// Sparse random projection: initialize random vectors, then iteratively
/// average neighbor embeddings. `dim` = embedding dimension, `iterations` = number
/// of averaging passes. Uses deterministic LCG-based random for reproducibility.
pub fn fastrp(graph: &CoGraph, dim: usize, iterations: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    // Initialize sparse random vectors using LCG.
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut rng = seed.wrapping_add(1);
    for _ in 0..n {
        let mut vec = vec![0.0f32; dim];
        for d in 0..dim {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((rng >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            // Sparse: zero out ~2/3 of entries (sqrt(3) sparse).
            let bucket = (rng >> 61) % 3;
            vec[d] = if bucket == 0 { r * 1.732 } else { 0.0 };
        }
        embeddings.push(vec);
    }

    // Iterative neighbor averaging.
    for _ in 0..iterations {
        let prev = embeddings.clone();
        for i in 0..n {
            if graph.adj[i].is_empty() {
                continue;
            }
            let mut avg = vec![0.0f32; dim];
            let mut total_weight = 0.0f32;
            for &(j, w) in &graph.adj[i] {
                let wf = w as f32;
                total_weight += wf;
                for d in 0..dim {
                    avg[d] += prev[j][d] * wf;
                }
            }
            if total_weight > 0.0 {
                for d in 0..dim {
                    embeddings[i][d] = (embeddings[i][d] + avg[d] / total_weight) / 2.0;
                }
            }
        }
    }

    // L2 normalize.
    for emb in &mut embeddings {
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in emb.iter_mut() {
                *x /= norm;
            }
        }
    }

    embeddings
}

/// Run FastRP and store results.
pub fn run_fastrp(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    dim: usize,
    iterations: usize,
    seed: u64,
) -> Result<Vec<FastRPResult>> {
    let graph = build_co_graph(hypergraph, narrative_id)?;
    let embeddings = fastrp(&graph, dim, iterations, seed);

    let mut results = Vec::with_capacity(graph.entities.len());
    for (i, eid) in graph.entities.iter().enumerate() {
        let key = analysis_key(FASTRP_PREFIX, &[narrative_id, &eid.to_string()]);
        let bytes = serde_json::to_vec(&embeddings[i])?;
        hypergraph.store().put(&key, &bytes)?;
        results.push(FastRPResult {
            entity_id: *eid,
            embedding: embeddings[i].clone(),
        });
    }
    Ok(results)
}

// ─── Node2Vec ──────────────────────────────────────────────

/// Node2Vec embedding result for one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node2VecResult {
    pub entity_id: Uuid,
    pub embedding: Vec<f32>,
}

/// Perform a biased random walk from `start` node.
///
/// `p` controls return to previous node, `q` controls exploration vs exploitation.
/// Low p = more backtracking (local). Low q = more exploration (global).
fn biased_walk(
    graph: &CoGraph,
    start: usize,
    walk_length: usize,
    p: f64,
    q: f64,
    rng: &mut u64,
) -> Vec<usize> {
    let mut walk = vec![start];
    if graph.adj[start].is_empty() {
        return walk;
    }

    // First step: uniform random neighbor.
    let neighbors = &graph.adj[start];
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let idx = (*rng >> 33) as usize % neighbors.len();
    walk.push(neighbors[idx].0);

    for _ in 2..walk_length {
        let curr = *walk.last().unwrap();
        let prev = walk[walk.len() - 2];
        let curr_neighbors = &graph.adj[curr];
        if curr_neighbors.is_empty() {
            break;
        }

        // Build unnormalized transition weights.
        let prev_neighbors: std::collections::HashSet<usize> =
            graph.adj[prev].iter().map(|&(v, _)| v).collect();

        let mut weights: Vec<f64> = Vec::with_capacity(curr_neighbors.len());
        for &(next, _w) in curr_neighbors {
            let alpha = if next == prev {
                1.0 / p
            } else if prev_neighbors.contains(&next) {
                1.0
            } else {
                1.0 / q
            };
            weights.push(alpha);
        }

        // Weighted random selection.
        let total: f64 = weights.iter().sum();
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((*rng >> 33) as f64 / u32::MAX as f64) * total;
        let mut cumulative = 0.0;
        let mut chosen = 0;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                chosen = i;
                break;
            }
        }
        walk.push(curr_neighbors[chosen].0);
    }
    walk
}

/// Compute Node2Vec embeddings via biased random walks → PMI → truncated SVD.
pub fn node2vec(
    graph: &CoGraph,
    dim: usize,
    walk_length: usize,
    walks_per_node: usize,
    p: f64,
    q: f64,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let window = 5;
    let mut rng = seed.wrapping_add(42);

    // Generate walks and count co-occurrences within window.
    let mut cooccur: HashMap<(usize, usize), f64> = HashMap::new();
    let mut node_count: Vec<f64> = vec![0.0; n];

    for node in 0..n {
        for _ in 0..walks_per_node {
            let walk = biased_walk(graph, node, walk_length, p, q, &mut rng);
            for (i, &u) in walk.iter().enumerate() {
                node_count[u] += 1.0;
                let end = (i + window + 1).min(walk.len());
                for &v in &walk[i + 1..end] {
                    if u != v {
                        let (lo, hi) = if u < v { (u, v) } else { (v, u) };
                        *cooccur.entry((lo, hi)).or_default() += 1.0;
                    }
                }
            }
        }
    }

    // Build sparse PMI-like matrix and do truncated SVD via power iteration.
    // Simplified: use co-occurrence directly as affinity matrix for SVD.
    let total: f64 = cooccur.values().sum::<f64>().max(1.0);

    // Power iteration for top-dim singular vectors.
    // Initialize random matrix V (n × dim).
    let mut v_mat: Vec<Vec<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            row.push((rng >> 33) as f64 / u32::MAX as f64 - 0.5);
        }
        v_mat.push(row);
    }

    // 10 iterations of power method: V = normalize(A · V).
    for _ in 0..10 {
        let mut new_v: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
        // Sparse matrix-vector product using co-occurrence entries.
        for (&(i, j), &val) in &cooccur {
            let pmi = (val * total / (node_count[i] * node_count[j]).max(1.0))
                .ln()
                .max(0.0);
            for d in 0..dim {
                new_v[i][d] += pmi * v_mat[j][d];
                new_v[j][d] += pmi * v_mat[i][d];
            }
        }
        // QR-like normalization (simplified: just normalize each column).
        for d in 0..dim {
            let col_norm: f64 = new_v.iter().map(|row| row[d] * row[d]).sum::<f64>().sqrt();
            if col_norm > 1e-10 {
                for row in &mut new_v {
                    row[d] /= col_norm;
                }
            }
        }
        v_mat = new_v;
    }

    // Convert to f32 and L2-normalize each row.
    v_mat
        .into_iter()
        .map(|row| {
            let mut emb: Vec<f32> = row.into_iter().map(|x| x as f32).collect();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut emb {
                    *x /= norm;
                }
            }
            emb
        })
        .collect()
}

/// Run Node2Vec and store results.
pub fn run_node2vec(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    dim: usize,
    walk_length: usize,
    walks_per_node: usize,
    p: f64,
    q: f64,
    seed: u64,
) -> Result<Vec<Node2VecResult>> {
    let graph = build_co_graph(hypergraph, narrative_id)?;
    let embeddings = node2vec(&graph, dim, walk_length, walks_per_node, p, q, seed);

    let mut results = Vec::with_capacity(graph.entities.len());
    for (i, eid) in graph.entities.iter().enumerate() {
        let key = analysis_key(NODE2VEC_PREFIX, &[narrative_id, &eid.to_string()]);
        let bytes = serde_json::to_vec(&embeddings[i])?;
        hypergraph.store().put(&key, &bytes)?;
        results.push(Node2VecResult {
            entity_id: *eid,
            embedding: embeddings[i].clone(),
        });
    }
    Ok(results)
}

// ─── Inference Engines ─────────────────────────────────────

/// FastRP inference engine.
pub struct FastRPEngine;

impl InferenceEngine for FastRPEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::FastRP
    }
    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }
    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let dim = job
            .parameters
            .get("dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let iters = job
            .parameters
            .get("iterations")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let seed = job
            .parameters
            .get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(42);

        let results = run_fastrp(hg, narrative_id, dim, iters, seed)?;
        let scores: Vec<serde_json::Value> = results
            .iter()
            .map(|r| serde_json::json!({"entity_id": r.entity_id, "dim": r.embedding.len()}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::FastRP,
            narrative_id,
            scores,
            &format!("FastRP: {} entities, dim={}", results.len(), dim),
        ))
    }
}

/// Node2Vec inference engine.
pub struct Node2VecEngine;

impl InferenceEngine for Node2VecEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::Node2Vec
    }
    fn estimate_cost(&self, _job: &InferenceJob, _hg: &Hypergraph) -> Result<u64> {
        Ok(10000)
    }
    fn execute(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let dim = job
            .parameters
            .get("dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let walk_len = job
            .parameters
            .get("walk_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as usize;
        let walks = job
            .parameters
            .get("walks_per_node")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;
        let p = job
            .parameters
            .get("p")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let q = job
            .parameters
            .get("q")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let seed = job
            .parameters
            .get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(42);

        let results = run_node2vec(hg, narrative_id, dim, walk_len, walks, p, q, seed)?;
        let scores: Vec<serde_json::Value> = results
            .iter()
            .map(|r| serde_json::json!({"entity_id": r.entity_id, "dim": r.embedding.len()}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::Node2Vec,
            narrative_id,
            scores,
            &format!("Node2Vec: {} entities, dim={}", results.len(), dim),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_fastrp_dimension() {
        let hg = make_hg();
        let n = "frp_dim";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let s = add_situation(&hg, n);
        link(&hg, a, s);
        link(&hg, b, s);

        let results = run_fastrp(&hg, n, 32, 3, 42).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].embedding.len(), 32);
        assert_eq!(results[1].embedding.len(), 32);
    }

    #[test]
    fn test_fastrp_deterministic() {
        let hg = make_hg();
        let n = "frp_det";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let c = add_entity(&hg, "C", n);
        let s1 = add_situation(&hg, n);
        link(&hg, a, s1);
        link(&hg, b, s1);
        let s2 = add_situation(&hg, n);
        link(&hg, b, s2);
        link(&hg, c, s2);

        let _r1 = run_fastrp(&hg, n, 16, 2, 123).unwrap();
        // Re-run with same seed should give same embeddings.
        let graph = build_co_graph(&hg, n).unwrap();
        let e1 = fastrp(&graph, 16, 2, 123);
        let e2 = fastrp(&graph, 16, 2, 123);
        assert_eq!(e1, e2, "Same seed should give identical embeddings");
    }

    #[test]
    fn test_fastrp_similar_neighbors() {
        let hg = make_hg();
        let n = "frp_sim";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let c = add_entity(&hg, "C", n);
        // A and B co-occur a lot, C is isolated.
        for _ in 0..5 {
            let s = add_situation(&hg, n);
            link(&hg, a, s);
            link(&hg, b, s);
        }
        let s_alone = add_situation(&hg, n);
        link(&hg, c, s_alone);

        let results = run_fastrp(&hg, n, 16, 3, 42).unwrap();
        // A and B should have more similar embeddings than A and C.
        let sim_ab = cosine(&results[0].embedding, &results[1].embedding);
        let sim_ac = cosine(&results[0].embedding, &results[2].embedding);
        assert!(
            sim_ab > sim_ac,
            "A-B similarity ({}) should exceed A-C similarity ({})",
            sim_ab,
            sim_ac
        );
    }

    #[test]
    fn test_node2vec_dimension() {
        let hg = make_hg();
        let n = "n2v_dim";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let s = add_situation(&hg, n);
        link(&hg, a, s);
        link(&hg, b, s);

        let results = run_node2vec(&hg, n, 16, 10, 5, 1.0, 1.0, 42).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].embedding.len(), 16);
    }

    #[test]
    fn test_node2vec_walk_length() {
        let hg = make_hg();
        let n = "n2v_walk";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let c = add_entity(&hg, "C", n);
        let s1 = add_situation(&hg, n);
        link(&hg, a, s1);
        link(&hg, b, s1);
        let s2 = add_situation(&hg, n);
        link(&hg, b, s2);
        link(&hg, c, s2);

        // Should complete without error even with long walks.
        let results = run_node2vec(&hg, n, 8, 50, 3, 1.0, 1.0, 42).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_node2vec_pq_params() {
        let hg = make_hg();
        let n = "n2v_pq";
        let a = add_entity(&hg, "A", n);
        let b = add_entity(&hg, "B", n);
        let c = add_entity(&hg, "C", n);
        let s = add_situation(&hg, n);
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);

        // Different p,q should produce different embeddings.
        let graph = build_co_graph(&hg, n).unwrap();
        let e_local = node2vec(&graph, 8, 10, 5, 0.5, 2.0, 42); // local-biased
        let e_global = node2vec(&graph, 8, 10, 5, 2.0, 0.5, 42); // global-biased
        assert_ne!(e_local, e_global);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}
