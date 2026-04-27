//! Narrative-native INFER algorithms that exploit TENSA's temporal hypergraph.
//!
//! - **Temporal PageRank**: time-decayed PageRank — "who is influential NOW?"
//! - **Causal Influence Centrality**: betweenness on the causal DAG
//! - **Information Bottleneck**: epistemic chokepoints via belief analysis
//! - **Assortativity**: degree correlation at edge endpoints (narrative-level scalar)

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::analysis::graph_projection::{self, CausalDag, CoGraph};
use crate::analysis::{
    analysis_key, extract_narrative_id, make_engine_result, store_entity_scores,
};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult};

/// Earliest→latest situation-start span in days for a narrative. Returns 0.0
/// when fewer than two dated situations exist.
pub fn narrative_span_days(hypergraph: &Hypergraph, narrative_id: &str) -> Result<f64> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let starts: Vec<DateTime<Utc>> = situations.iter().filter_map(|s| s.temporal.start).collect();
    let (Some(&earliest), Some(&latest)) = (starts.iter().min(), starts.iter().max()) else {
        return Ok(0.0);
    };
    Ok((latest - earliest).num_seconds().max(0) as f64 / 86400.0)
}

// ─── Temporal PageRank ─────────────────────────────────────────

/// Decay scaling mode for temporal PageRank.
///
/// `Fixed(λ)` uses the given constant. `Auto` derives λ from the narrative's
/// temporal span so mid-narrative situations decay by half regardless of
/// wall-clock age — the natural choice when the narrative sits far in the
/// past (e.g. benchmark archives) or when you want recency *within* the
/// story rather than relative to today.
#[derive(Debug, Clone, Copy)]
pub enum TemporalPageRankDecay {
    Fixed(f64),
    Auto,
}

impl TemporalPageRankDecay {
    /// Resolve to a concrete λ. For `Auto`, `span_days` is the range between
    /// the earliest and latest situation timestamp. Falls back to `0.1` when
    /// the span is unknown or degenerate.
    pub fn resolve(self, span_days: f64) -> f64 {
        match self {
            Self::Fixed(l) => l,
            Self::Auto if span_days > 0.0 => std::f64::consts::LN_2 / (span_days * 0.5).max(1.0),
            Self::Auto => 0.1,
        }
    }
}

/// Compute Temporal PageRank: PageRank with exponentially time-decayed edge weights.
///
/// `decay_lambda`: decay rate (higher = more recent bias). Default 0.1.
/// Age is measured relative to the **latest situation in the narrative**, not
/// wall-clock time — so a narrative spanning Jan–Jun 2025 still produces a
/// recency gradient even when queried in 2026.
///
/// `window`: optional time window to restrict situations.
pub fn temporal_pagerank(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    decay_lambda: f64,
    window: Option<(DateTime<Utc>, DateTime<Utc>)>,
) -> Result<(Vec<Uuid>, Vec<f64>)> {
    // Build time-filtered co-graph
    let graph = graph_projection::build_temporal_graph(hypergraph, narrative_id, window)?;
    if graph.entities.is_empty() {
        return Ok((vec![], vec![]));
    }

    // For time-decay, we need situation timestamps. Build a mapping of
    // entity pairs → most recent co-participation time.
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    // Reference time = latest situation start. Ages in this narrative are
    // measured against this, so a narrative that closed two years ago still
    // differentiates early vs late actors.
    let reference_time = situations
        .iter()
        .filter_map(|s| s.temporal.start)
        .max()
        .unwrap_or_else(Utc::now);

    // Build entity index
    let entity_idx: std::collections::HashMap<Uuid, usize> = graph
        .entities
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Compute time-decayed weights
    let n = graph.entities.len();
    let mut decayed_adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

    for sit in &situations {
        let t = sit.temporal.start.unwrap_or(reference_time);
        let age_days = (reference_time - t).num_seconds().max(0) as f64 / 86400.0;
        let decay = (-decay_lambda * age_days).exp();

        let participants = hypergraph.get_participants_for_situation(&sit.id)?;
        let idxs: Vec<usize> = participants
            .iter()
            .filter_map(|p| entity_idx.get(&p.entity_id).copied())
            .collect();

        for a in 0..idxs.len() {
            for b in (a + 1)..idxs.len() {
                // Add decayed weight to both directions
                decayed_adj[idxs[a]].push((idxs[b], decay));
                decayed_adj[idxs[b]].push((idxs[a], decay));
            }
        }
    }

    // Merge duplicate edges (sum weights)
    let mut merged: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
    for i in 0..n {
        let mut edge_map: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for &(j, w) in &decayed_adj[i] {
            *edge_map.entry(j).or_insert(0.0) += w;
        }
        merged[i] = edge_map
            .into_iter()
            .map(|(j, w)| (j, ((w * 1000.0) as usize).max(1)))
            .collect();
    }

    let decayed_graph = CoGraph {
        entities: graph.entities.clone(),
        adj: merged,
    };
    let scores = crate::analysis::graph_centrality::compute_pagerank(&decayed_graph);

    Ok((graph.entities, scores))
}

/// Temporal PageRank engine.
pub struct TemporalPageRankEngine;

impl InferenceEngine for TemporalPageRankEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::TemporalPageRank
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(6000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        // `decay_lambda` accepts either a positive number or the string
        // `"auto"`. Auto derives λ from the narrative's temporal span so
        // mid-narrative situations decay by half regardless of wall-clock age.
        let raw = job.parameters.get("decay_lambda");
        let mode = match raw {
            Some(v) if v.as_str().is_some_and(|s| s.eq_ignore_ascii_case("auto")) => {
                TemporalPageRankDecay::Auto
            }
            Some(v) => TemporalPageRankDecay::Fixed(v.as_f64().unwrap_or(0.1)),
            None => TemporalPageRankDecay::Fixed(0.1),
        };
        let span_days = narrative_span_days(hypergraph, narrative_id)?;
        let decay = mode.resolve(span_days);

        let (entities, scores) = temporal_pagerank(hypergraph, narrative_id, decay, None)?;
        store_entity_scores(hypergraph, &entities, &scores, b"an/tpr/", narrative_id)?;

        let result_map: Vec<serde_json::Value> = entities
            .iter()
            .enumerate()
            .map(|(i, eid)| serde_json::json!({"entity_id": eid, "temporal_pagerank": scores[i]}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::TemporalPageRank,
            narrative_id,
            result_map,
            "Temporal PageRank complete",
        ))
    }
}

// ─── Causal Influence Centrality ───────────────────────────────

/// Compute causal influence centrality: betweenness on the CausalDag.
///
/// Measures who *causes* the most downstream effects by computing
/// shortest-path betweenness on the directed causal graph.
pub fn causal_influence(
    dag: &CausalDag,
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<(Uuid, f64)>> {
    let n = dag.situations.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Compute betweenness on the causal DAG (directed shortest paths)
    let mut cb = vec![0.0_f64; n];
    for s in 0..n {
        // BFS from s on the directed DAG
        let mut stack = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0_f64; n];
        sigma[s] = 1.0;
        let mut dist: Vec<i64> = vec![-1; n];
        dist[s] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &(w, _) in &dag.adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0_f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // Map situation betweenness to entity influence via participation
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let sit_idx: std::collections::HashMap<Uuid, usize> = dag
        .situations
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut entity_influence: Vec<(Uuid, f64)> = Vec::new();
    for entity in &entities {
        let participations = hypergraph.get_situations_for_entity(&entity.id)?;
        let total: f64 = participations
            .iter()
            .filter_map(|p| sit_idx.get(&p.situation_id).map(|&i| cb[i]))
            .sum();
        entity_influence.push((entity.id, total));
    }

    Ok(entity_influence)
}

/// Causal Influence engine.
pub struct CausalInfluenceEngine;

impl InferenceEngine for CausalInfluenceEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CausalInfluence
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let dag = graph_projection::build_causal_dag(hypergraph, narrative_id)?;
        let influence = causal_influence(&dag, hypergraph, narrative_id)?;

        let entity_ids: Vec<Uuid> = influence.iter().map(|&(id, _)| id).collect();
        let scores: Vec<f64> = influence.iter().map(|&(_, s)| s).collect();
        store_entity_scores(hypergraph, &entity_ids, &scores, b"an/ci/", narrative_id)?;

        let result_map: Vec<serde_json::Value> = influence
            .iter()
            .map(|(eid, score)| serde_json::json!({"entity_id": eid, "causal_influence": score}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::CausalInfluence,
            narrative_id,
            result_map,
            "Causal influence centrality complete",
        ))
    }
}

// ─── Information Bottleneck ────────────────────────────────────

/// Compute information bottleneck scores: entities that are sole knowers of facts.
///
/// For each fact in the belief network, count how many entities know it.
/// An entity's bottleneck score = number of facts it is the SOLE knower of.
pub fn information_bottleneck(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<(Uuid, f64)>> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    if entities.is_empty() {
        return Ok(vec![]);
    }

    // Scan belief snapshots to build fact → knower set mapping
    let prefix = format!("an/b/{}/", narrative_id);
    let entries = hypergraph.store().prefix_scan(prefix.as_bytes())?;

    let mut fact_knowers: std::collections::HashMap<String, std::collections::HashSet<Uuid>> =
        std::collections::HashMap::new();

    for (_key, value) in &entries {
        if let Ok(snapshot) = serde_json::from_slice::<serde_json::Value>(value) {
            let entity_a = snapshot
                .get("entity_a")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<Uuid>().ok());
            let believed = snapshot
                .get("believed_knowledge")
                .and_then(|v| v.as_array());

            if let (Some(eid), Some(facts)) = (entity_a, believed) {
                for fact in facts {
                    let fact_key = fact
                        .get("fact")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !fact_key.is_empty() {
                        fact_knowers.entry(fact_key).or_default().insert(eid);
                    }
                }
            }
        }
    }

    // Score: number of facts where entity is the sole knower
    let mut scores: std::collections::HashMap<Uuid, f64> = std::collections::HashMap::new();
    for (_fact, knowers) in &fact_knowers {
        if knowers.len() == 1 {
            let sole_knower = knowers.iter().next().unwrap();
            *scores.entry(*sole_knower).or_insert(0.0) += 1.0;
        }
    }

    Ok(entities
        .iter()
        .map(|e| (e.id, scores.get(&e.id).copied().unwrap_or(0.0)))
        .collect())
}

/// Information Bottleneck engine.
pub struct InfoBottleneckEngine;

impl InferenceEngine for InfoBottleneckEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::InfoBottleneck
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(4000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let bottlenecks = information_bottleneck(hypergraph, narrative_id)?;

        let entity_ids: Vec<Uuid> = bottlenecks.iter().map(|&(id, _)| id).collect();
        let scores: Vec<f64> = bottlenecks.iter().map(|&(_, s)| s).collect();
        store_entity_scores(hypergraph, &entity_ids, &scores, b"an/ib/", narrative_id)?;

        let result_map: Vec<serde_json::Value> = bottlenecks
            .iter()
            .map(|(eid, score)| serde_json::json!({"entity_id": eid, "bottleneck_score": score}))
            .collect();

        Ok(make_engine_result(
            job,
            InferenceJobType::InfoBottleneck,
            narrative_id,
            result_map,
            "Information bottleneck analysis complete",
        ))
    }
}

// ─── Assortativity ─────────────────────────────────────────────

/// Compute degree assortativity: Pearson correlation of degrees at edge endpoints.
///
/// Positive = assortative (hubs connect to hubs).
/// Negative = disassortative (hubs connect to leaves).
/// Returns a single scalar for the narrative.
pub fn degree_assortativity(graph: &CoGraph) -> f64 {
    let n = graph.entities.len();
    if n < 2 {
        return 0.0;
    }

    let degree: Vec<f64> = graph.adj.iter().map(|nb| nb.len() as f64).collect();
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut sum_x2 = 0.0_f64;
    let mut sum_y2 = 0.0_f64;
    let mut m = 0u64;

    for i in 0..n {
        for &(j, _) in &graph.adj[i] {
            if i < j {
                // Count each undirected edge once
                let di = degree[i];
                let dj = degree[j];
                sum_x += di;
                sum_y += dj;
                sum_xy += di * dj;
                sum_x2 += di * di;
                sum_y2 += dj * dj;
                m += 1;
            }
        }
    }

    if m == 0 {
        return 0.0;
    }

    let mf = m as f64;
    let num = sum_xy / mf - (sum_x / mf) * (sum_y / mf);
    let denom_x = (sum_x2 / mf - (sum_x / mf).powi(2)).sqrt();
    let denom_y = (sum_y2 / mf - (sum_y / mf).powi(2)).sqrt();

    if denom_x * denom_y == 0.0 {
        return 0.0;
    }

    num / (denom_x * denom_y)
}

/// Assortativity engine.
pub struct AssortativityEngine;

impl InferenceEngine for AssortativityEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::Assortativity
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(2000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
        let r = degree_assortativity(&graph);

        // Store as narrative-level scalar at an/as/{narrative_id}
        let key = analysis_key(b"an/as/", &[narrative_id]);
        let bytes = serde_json::to_vec(&r)?;
        hypergraph.store().put(&key, &bytes)?;

        Ok(make_engine_result(
            job,
            InferenceJobType::Assortativity,
            narrative_id,
            vec![serde_json::json!({"assortativity": r})],
            "Degree assortativity computed",
        ))
    }
}

#[cfg(test)]
#[path = "narrative_centrality_tests.rs"]
mod tests;
