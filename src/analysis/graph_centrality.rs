//! Level 1 graph centrality algorithms: PageRank, Eigenvector, Harmonic, HITS.
//!
//! All algorithms operate on shared graph projections from `graph_projection.rs`.
//! Results are stored in KV and queryable as `e.an.*` virtual properties.

use serde::{Deserialize, Serialize};

use crate::analysis::graph_projection::{self, BipartiteGraph, CoGraph};
use crate::analysis::{
    analysis_key, extract_narrative_id, make_engine_result, store_entity_scores,
};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult};

// ─── PageRank ──────────────────────────────────────────────────

const PAGERANK_DAMPING: f64 = 0.85;
const PAGERANK_EPSILON: f64 = 1e-6;
const PAGERANK_MAX_ITER: usize = 100;

/// Compute PageRank scores on the co-participation graph.
///
/// Uses power iteration with damping factor 0.85. Dangling nodes (no outgoing
/// edges) distribute their mass uniformly across all nodes.
pub fn compute_pagerank(graph: &CoGraph) -> Vec<f64> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let uniform = 1.0 / n as f64;
    let mut rank = vec![uniform; n];

    let out_weight: Vec<f64> = graph
        .adj
        .iter()
        .map(|neighbors| neighbors.iter().map(|&(_, w)| w as f64).sum::<f64>())
        .collect();

    for _ in 0..PAGERANK_MAX_ITER {
        let mut new_rank = vec![0.0_f64; n];

        let dangling_sum: f64 = (0..n)
            .filter(|&i| out_weight[i] == 0.0)
            .map(|i| rank[i])
            .sum();

        for i in 0..n {
            if out_weight[i] == 0.0 {
                continue;
            }
            let contribution = rank[i] / out_weight[i];
            for &(j, w) in &graph.adj[i] {
                new_rank[j] += contribution * w as f64;
            }
        }

        let teleport = (1.0 - PAGERANK_DAMPING + PAGERANK_DAMPING * dangling_sum) * uniform;
        for val in &mut new_rank {
            *val = PAGERANK_DAMPING * *val + teleport;
        }

        let diff: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        rank = new_rank;
        if diff < PAGERANK_EPSILON {
            break;
        }
    }

    rank
}

/// Personalized PageRank: seed from specific nodes, run with restart α=0.15.
///
/// `seed_weights`: per-node probability mass for teleport. Must sum to 1.0.
/// Nodes not in seed_weights have 0 teleport probability.
pub fn personalized_pagerank(graph: &CoGraph, seed_weights: &[f64], alpha: f64) -> Vec<f64> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let mut rank = seed_weights.to_vec();
    if rank.len() != n {
        rank.resize(n, 0.0);
    }

    let out_weight: Vec<f64> = graph
        .adj
        .iter()
        .map(|nb| nb.iter().map(|&(_, w)| w as f64).sum::<f64>())
        .collect();

    for _ in 0..PAGERANK_MAX_ITER {
        let mut new_rank = vec![0.0_f64; n];
        let dangling_sum: f64 = (0..n)
            .filter(|&i| out_weight[i] == 0.0)
            .map(|i| rank[i])
            .sum();

        for i in 0..n {
            if out_weight[i] == 0.0 {
                continue;
            }
            let contribution = rank[i] / out_weight[i];
            for &(j, w) in &graph.adj[i] {
                new_rank[j] += contribution * w as f64;
            }
        }

        // Personalized teleport: restart to seed distribution
        for i in 0..n {
            new_rank[i] = (1.0 - alpha) * new_rank[i]
                + alpha
                    * (seed_weights.get(i).copied().unwrap_or(0.0)
                        + dangling_sum * seed_weights.get(i).copied().unwrap_or(0.0));
        }

        let diff: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        rank = new_rank;
        if diff < PAGERANK_EPSILON {
            break;
        }
    }

    rank
}

/// PageRank inference engine.
pub struct PageRankEngine;

impl InferenceEngine for PageRankEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::PageRank
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
        let scores = compute_pagerank(&graph);
        store_entity_scores(
            hypergraph,
            &graph.entities,
            &scores,
            b"an/pr/",
            narrative_id,
        )?;

        let result_map: Vec<serde_json::Value> = graph
            .entities
            .iter()
            .enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "pagerank": scores[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::PageRank,
            narrative_id,
            result_map,
            "PageRank analysis complete",
        ))
    }
}

// ─── Eigenvector Centrality ────────────────────────────────────

const EIGENVECTOR_EPSILON: f64 = 1e-6;
const EIGENVECTOR_MAX_ITER: usize = 100;

/// Compute eigenvector centrality via power iteration on the adjacency matrix.
///
/// For disconnected graphs, computes per-component using WCC from Sprint 0,
/// then normalizes globally.
pub fn compute_eigenvector(graph: &CoGraph) -> Vec<f64> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let mut scores = vec![0.0_f64; n];
    let components = graph_projection::wcc(graph);

    for component in &components {
        if component.len() == 1 {
            continue;
        }

        let comp_n = component.len();
        let init = 1.0 / (comp_n as f64).sqrt();
        let mut x: Vec<f64> = vec![init; comp_n];

        let mut global_to_local: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (local, &global) in component.iter().enumerate() {
            global_to_local.insert(global, local);
        }

        for _ in 0..EIGENVECTOR_MAX_ITER {
            let mut new_x = vec![0.0_f64; comp_n];
            for (local_i, &global_i) in component.iter().enumerate() {
                for &(global_j, w) in &graph.adj[global_i] {
                    if let Some(&local_j) = global_to_local.get(&global_j) {
                        new_x[local_i] += w as f64 * x[local_j];
                    }
                }
            }

            let norm: f64 = new_x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in &mut new_x {
                    *v /= norm;
                }
            }

            let diff: f64 = x.iter().zip(new_x.iter()).map(|(a, b)| (a - b).abs()).sum();
            x = new_x;
            if diff < EIGENVECTOR_EPSILON {
                break;
            }
        }

        for (local, &global) in component.iter().enumerate() {
            scores[global] = x[local];
        }
    }

    scores
}

/// Eigenvector centrality inference engine.
pub struct EigenvectorEngine;

impl InferenceEngine for EigenvectorEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::EigenvectorCentrality
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
        let scores = compute_eigenvector(&graph);
        store_entity_scores(
            hypergraph,
            &graph.entities,
            &scores,
            b"an/ev_c/",
            narrative_id,
        )?;

        let result_map: Vec<serde_json::Value> = graph
            .entities
            .iter()
            .enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "eigenvector": scores[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::EigenvectorCentrality,
            narrative_id,
            result_map,
            "Eigenvector centrality analysis complete",
        ))
    }
}

// ─── Harmonic Centrality ───────────────────────────────────────

/// Compute harmonic centrality: H(v) = sum(1/d(v,u)) for all u != v.
///
/// Naturally handles disconnected graphs (infinite distance contributes 0).
/// Uses BFS from each node — O(V*(V+E)).
pub fn compute_harmonic(graph: &CoGraph) -> Vec<f64> {
    let n = graph.entities.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    let mut harmonic = vec![0.0_f64; n];
    let norm = (n - 1) as f64;

    for s in 0..n {
        let mut dist: Vec<i64> = vec![-1; n];
        dist[s] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            for &(w, _) in &graph.adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
            }
        }

        let h: f64 = dist
            .iter()
            .enumerate()
            .filter(|&(i, &d)| i != s && d > 0)
            .map(|(_, &d)| 1.0 / d as f64)
            .sum();

        harmonic[s] = h / norm;
    }

    harmonic
}

/// Harmonic centrality inference engine.
pub struct HarmonicEngine;

impl InferenceEngine for HarmonicEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::HarmonicCentrality
    }

    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
        // O(V*(V+E)) — scale cost with entity count
        let narrative_id = extract_narrative_id(job).unwrap_or("unknown");
        let n = hypergraph
            .list_entities_by_narrative(narrative_id)
            .map(|e| e.len())
            .unwrap_or(0);
        Ok(1000 + (n as u64).saturating_mul(n as u64 / 10).min(60000))
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
        let scores = compute_harmonic(&graph);
        store_entity_scores(
            hypergraph,
            &graph.entities,
            &scores,
            b"an/hc/",
            narrative_id,
        )?;

        let result_map: Vec<serde_json::Value> = graph
            .entities
            .iter()
            .enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "harmonic": scores[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::HarmonicCentrality,
            narrative_id,
            result_map,
            "Harmonic centrality analysis complete",
        ))
    }
}

// ─── HITS (Hubs & Authorities) ─────────────────────────────────

const HITS_EPSILON: f64 = 1e-6;
const HITS_MAX_ITER: usize = 100;

/// HITS scores for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitsScore {
    pub hub_score: f64,
    pub authority_score: f64,
}

/// Compute HITS on the bipartite graph.
///
/// Situations are natural hubs (connecting many actors), entities are authorities.
/// Returns (hub_scores_per_entity, authority_scores_per_entity).
pub fn compute_hits(bipartite: &BipartiteGraph) -> (Vec<f64>, Vec<f64>) {
    let n_ent = bipartite.entities.len();
    let n_sit = bipartite.situations.len();

    if n_ent == 0 || n_sit == 0 || bipartite.edges.is_empty() {
        return (vec![0.0; n_ent], vec![0.0; n_ent]);
    }

    let mut ent_to_sit: Vec<Vec<usize>> = vec![vec![]; n_ent];
    let mut sit_to_ent: Vec<Vec<usize>> = vec![vec![]; n_sit];
    for &(ei, si) in &bipartite.edges {
        ent_to_sit[ei].push(si);
        sit_to_ent[si].push(ei);
    }

    let init = 1.0 / (n_ent as f64).sqrt();
    let mut authority = vec![init; n_ent];
    let mut hub = vec![0.0_f64; n_ent];

    for _ in 0..HITS_MAX_ITER {
        let mut sit_auth = vec![0.0_f64; n_sit];
        for (si, ents) in sit_to_ent.iter().enumerate() {
            for &ei in ents {
                sit_auth[si] += authority[ei];
            }
        }

        let mut new_hub = vec![0.0_f64; n_ent];
        for (ei, sits) in ent_to_sit.iter().enumerate() {
            for &si in sits {
                new_hub[ei] += sit_auth[si];
            }
        }

        let hub_norm: f64 = new_hub.iter().map(|v| v * v).sum::<f64>().sqrt();
        if hub_norm > 0.0 {
            for v in &mut new_hub {
                *v /= hub_norm;
            }
        }

        let mut sit_hub = vec![0.0_f64; n_sit];
        for (si, ents) in sit_to_ent.iter().enumerate() {
            for &ei in ents {
                sit_hub[si] += new_hub[ei];
            }
        }

        let mut new_auth = vec![0.0_f64; n_ent];
        for (ei, sits) in ent_to_sit.iter().enumerate() {
            for &si in sits {
                new_auth[ei] += sit_hub[si];
            }
        }

        let auth_norm: f64 = new_auth.iter().map(|v| v * v).sum::<f64>().sqrt();
        if auth_norm > 0.0 {
            for v in &mut new_auth {
                *v /= auth_norm;
            }
        }

        let diff: f64 = authority
            .iter()
            .zip(new_auth.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            + hub
                .iter()
                .zip(new_hub.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();

        hub = new_hub;
        authority = new_auth;
        if diff < HITS_EPSILON {
            break;
        }
    }

    (hub, authority)
}

/// HITS inference engine.
pub struct HitsEngine;

impl InferenceEngine for HitsEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::HITS
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let bipartite = graph_projection::build_bipartite(hypergraph, narrative_id)?;
        let (hub, authority) = compute_hits(&bipartite);

        for (i, &eid) in bipartite.entities.iter().enumerate() {
            let score = HitsScore {
                hub_score: hub[i],
                authority_score: authority[i],
            };
            let key = analysis_key(b"an/hits/", &[narrative_id, &eid.to_string()]);
            let bytes = serde_json::to_vec(&score)?;
            hypergraph.store().put(&key, &bytes)?;
        }

        let result_map: Vec<serde_json::Value> = bipartite.entities.iter().enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "hub_score": hub[i], "authority_score": authority[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::HITS,
            narrative_id,
            result_map,
            "HITS hub/authority analysis complete",
        ))
    }
}

#[cfg(test)]
#[path = "graph_centrality_tests.rs"]
mod tests;
