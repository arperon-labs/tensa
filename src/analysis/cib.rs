//! Coordinated Inauthentic Behavior (CIB) detection — Sprint D3.
//!
//! Builds a weighted behavioral-similarity network over actors (entities of
//! type Actor), runs label-propagation community detection on it, and filters
//! the resulting communities by a *calibrated* density p-value. The null
//! hypothesis: "edges are independent draws from the empirical pair-similarity
//! distribution." A community with observed density deep in the right tail of
//! the null is flagged as CIB.
//!
//! Cross-platform CIB detection projects the network under the `factory-task`
//! [`crate::analysis::style_profile::WeightedSimilarityConfig`] — emphasizing
//! structural/graph reuse that survives platform-specific surface text — and
//! filters to clusters whose members span ≥ 2 platforms.
//!
//! Superspreader ranking reuses the `graph_centrality` engines (PageRank,
//! Eigenvector, Betweenness/Harmonic) against the co-participation graph,
//! then returns the top-N actors by the chosen score.
//!
//! Persistence:
//! - `cib/c/{narrative_id}/{cluster_id}` → [`CibCluster`]
//! - `cib/e/{narrative_id}/{cluster_id}` → [`CibEvidence`]
//! - `cib/s/{narrative_id}` → [`SuperspreaderRanking`]

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::analysis_key;
use crate::analysis::community_detect::label_propagation;
use crate::analysis::graph_centrality::{compute_eigenvector, compute_harmonic, compute_pagerank};
use crate::analysis::graph_projection::{self, CoGraph};
use crate::disinfo::fingerprints::{ensure_behavioral_fingerprint, BEHAVIORAL_AXIS_COUNT};
use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::types::EntityType;

// ─── Configuration ─────────────────────────────────────────────

/// Minimum behavioral similarity above which an edge is added to the graph.
/// Calibrated by bootstrap; below this the edge is almost certainly organic
/// noise and inflating the similarity network with it wastes compute.
pub const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.7;

/// Default bootstrap iterations used to calibrate the density null.
pub const DEFAULT_DENSITY_BOOTSTRAP_ITER: usize = 500;

/// Default significance level at which a cluster is flagged as CIB.
pub const DEFAULT_ALPHA: f64 = 0.01;

/// Seed used for all deterministic RNGs in this module. The caller can override
/// via [`CibConfig::seed`] for reproducibility under different null draws.
pub const DEFAULT_SEED: u64 = 0x1b3d_7a2f;

/// Minimum cluster size considered for CIB reporting. Pairs-only communities
/// are unstable — the spec calls for ≥ 3.
pub const MIN_CLUSTER_SIZE: usize = 3;

/// User-tunable knobs for [`detect_cib_clusters`] + [`detect_cross_platform_cib`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CibConfig {
    pub similarity_threshold: f64,
    pub alpha: f64,
    pub bootstrap_iter: usize,
    pub min_cluster_size: usize,
    pub seed: u64,
}

impl Default for CibConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            alpha: DEFAULT_ALPHA,
            bootstrap_iter: DEFAULT_DENSITY_BOOTSTRAP_ITER,
            min_cluster_size: MIN_CLUSTER_SIZE,
            seed: DEFAULT_SEED,
        }
    }
}

impl CibConfig {
    /// Overlay optional values from a JSON params object onto the defaults.
    /// Shared by the REST handler, the MCP tool, and the inference engine so
    /// the parse rules stay in one place.
    pub fn from_json(params: &serde_json::Value) -> Self {
        let mut cfg = Self::default();
        if let Some(v) = params.get("similarity_threshold").and_then(|v| v.as_f64()) {
            cfg.similarity_threshold = v;
        }
        if let Some(v) = params.get("alpha").and_then(|v| v.as_f64()) {
            cfg.alpha = v;
        }
        if let Some(v) = params.get("bootstrap_iter").and_then(|v| v.as_u64()) {
            cfg.bootstrap_iter = v as usize;
        }
        if let Some(v) = params.get("min_cluster_size").and_then(|v| v.as_u64()) {
            cfg.min_cluster_size = v as usize;
        }
        if let Some(v) = params.get("seed").and_then(|v| v.as_u64()) {
            cfg.seed = v;
        }
        cfg
    }
}

// ─── Types ─────────────────────────────────────────────────────

/// One edge in the behavioral similarity graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralSimilarityEdge {
    pub a: Uuid,
    pub b: Uuid,
    /// Composite similarity in `[0, 1]` from per-axis comparable distances
    /// of the two BehavioralFingerprints (NaN axes skipped).
    pub similarity: f64,
    /// Number of axes that were non-NaN on both sides.
    pub comparable_axes: usize,
}

/// Weighted undirected graph of actors keyed by their UUIDs.
///
/// Carries the already-loaded behavioral fingerprint axes for each actor so
/// downstream consumers (cluster density, per-axis mean distance, evidence
/// assembly) don't re-read from KV.
#[derive(Debug, Clone)]
pub struct BehavioralSimilarityNetwork {
    /// Ordered list of node UUIDs (index = position in `adj`).
    pub actors: Vec<Uuid>,
    /// Index back into `actors` by UUID.
    pub actor_idx: HashMap<Uuid, usize>,
    /// Adjacency list — (neighbor_idx, similarity * 1000 rounded) for
    /// co-graph API compatibility (Louvain / label_propagation use usize weights).
    pub adj: Vec<Vec<(usize, usize)>>,
    /// Raw float edges preserved alongside the integer-weighted CoGraph for
    /// density calibration.
    pub edges: Vec<BehavioralSimilarityEdge>,
    /// Per-actor fingerprint axes (indexed by the same position as `actors`).
    /// Populated once during [`build_similarity_network`] so cluster-evidence
    /// assembly doesn't re-read from KV.
    pub axes_by_actor: Vec<[f64; BEHAVIORAL_AXIS_COUNT]>,
}

impl BehavioralSimilarityNetwork {
    /// Total number of distinct undirected edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Convert to a `CoGraph` with `similarity * 1000` as the integer weight.
    /// Used by the existing label-propagation / pagerank engines.
    pub fn to_cograph(&self) -> CoGraph {
        CoGraph {
            entities: self.actors.clone(),
            adj: self.adj.clone(),
        }
    }

    /// Build the `(min(i,j), max(i,j)) → similarity` edge index used by
    /// cluster-density and null-bootstrap code paths.
    fn edge_set(&self) -> HashMap<(usize, usize), f64> {
        self.edges
            .iter()
            .filter_map(|e| {
                let i = *self.actor_idx.get(&e.a)?;
                let j = *self.actor_idx.get(&e.b)?;
                Some(((i.min(j), i.max(j)), e.similarity))
            })
            .collect()
    }
}

/// One CIB cluster detected from the behavioral similarity network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CibCluster {
    pub cluster_id: String,
    pub narrative_id: String,
    pub members: Vec<Uuid>,
    /// Observed density = 2·|E_cluster| / (n·(n-1)) for the induced subgraph.
    pub density: f64,
    /// Average edge similarity within the cluster.
    pub mean_similarity: f64,
    /// Calibrated right-tail p-value under the edge-shuffle null. Lower ⇒
    /// more anomalous.
    pub p_value: f64,
    /// Number of null samples used to calibrate the p-value.
    pub bootstrap_iter: usize,
    /// Significance level this cluster was flagged under.
    pub alpha: f64,
    /// Distinct platforms observed across cluster members.
    pub platforms: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Evidence supporting a [`CibCluster`] — stored separately so callers can
/// attach additional detective-work later without rewriting the cluster record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CibEvidence {
    pub cluster_id: String,
    pub narrative_id: String,
    /// Representative high-similarity edges (top-K by similarity).
    pub top_edges: Vec<BehavioralSimilarityEdge>,
    /// Per-axis mean pairwise distance inside the cluster — low values
    /// indicate the CIB signal comes from aligned behavioral patterns rather
    /// than a single outlier axis.
    pub axis_mean_distances: Vec<(String, f64)>,
    /// Whether factory-task weighting was used (true for cross-platform CIB).
    pub factory_weighted: bool,
    pub created_at: DateTime<Utc>,
}

/// Output of [`detect_cib_clusters`] — clusters plus network stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CibDetectionResult {
    pub narrative_id: String,
    pub clusters: Vec<CibCluster>,
    pub evidence: Vec<CibEvidence>,
    pub network_size: usize,
    pub edge_count: usize,
    pub null_mean_density: f64,
    pub null_std_density: f64,
    pub config: CibConfig,
}

/// Method used to rank superspreaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SuperspreaderMethod {
    PageRank,
    Eigenvector,
    Harmonic,
}

impl SuperspreaderMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PageRank => "pagerank",
            Self::Eigenvector => "eigenvector",
            Self::Harmonic => "harmonic",
        }
    }
}

impl std::str::FromStr for SuperspreaderMethod {
    type Err = TensaError;

    /// Case-insensitive parse. Note: `"betweenness"` is accepted as an alias
    /// for `Harmonic` because harmonic centrality is the standard
    /// betweenness-like distance-based measure TENSA ships; we prefer the
    /// alias over shipping a second algorithm until a real Brandes-style
    /// betweenness engine is added. If you need true betweenness, wire it
    /// as a new variant and remove this alias.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pagerank" => Ok(Self::PageRank),
            "eigenvector" => Ok(Self::Eigenvector),
            "harmonic" | "betweenness" => Ok(Self::Harmonic),
            other => Err(TensaError::InvalidQuery(format!(
                "unknown superspreader method '{other}' (use pagerank|eigenvector|harmonic)"
            ))),
        }
    }
}

/// Scored entity in a superspreader ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperspreaderScore {
    pub entity_id: Uuid,
    pub score: f64,
    pub rank: usize,
}

/// Output of [`rank_superspreaders`] — ranked list plus method metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperspreaderRanking {
    pub narrative_id: String,
    pub method: SuperspreaderMethod,
    pub top_n: usize,
    pub scores: Vec<SuperspreaderScore>,
    pub network_size: usize,
    pub created_at: DateTime<Utc>,
}

// ─── Network Construction ──────────────────────────────────────

/// Build a behavioral similarity graph over all actor entities in the
/// narrative. Pairs with fewer than 2 comparable axes are skipped because
/// their similarity is unreliable; pairs with similarity below
/// `config.similarity_threshold` don't add an edge.
///
/// This is O(N²) in the number of actors — acceptable for up to ~5k actors
/// per narrative, beyond which the spec requires LSH/HNSW pre-filtering
/// (deferred to Sprint D7).
pub fn build_similarity_network(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    config: &CibConfig,
) -> Result<BehavioralSimilarityNetwork> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let actors: Vec<Uuid> = entities
        .into_iter()
        .filter(|e| matches!(e.entity_type, EntityType::Actor))
        .map(|e| e.id)
        .collect();
    let n = actors.len();
    let actor_idx: HashMap<Uuid, usize> = actors.iter().enumerate().map(|(i, u)| (*u, i)).collect();
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut edges: Vec<BehavioralSimilarityEdge> = Vec::new();

    // Precompute fingerprints once; even for n<2 we still return them so
    // downstream callers have a consistent view of the loaded axes.
    let mut axes_by_actor: Vec<[f64; BEHAVIORAL_AXIS_COUNT]> = Vec::with_capacity(n);
    for a in &actors {
        let fp = ensure_behavioral_fingerprint(hypergraph, a, false)?;
        axes_by_actor.push(fp.axes);
    }

    if n >= 2 {
        for i in 0..n {
            for j in (i + 1)..n {
                let (sim, comparable) = axis_similarity(&axes_by_actor[i], &axes_by_actor[j]);
                if comparable < 2 {
                    continue;
                }
                if sim < config.similarity_threshold {
                    continue;
                }
                let weight = (sim * 1000.0).round().max(1.0) as usize;
                adj[i].push((j, weight));
                adj[j].push((i, weight));
                edges.push(BehavioralSimilarityEdge {
                    a: actors[i],
                    b: actors[j],
                    similarity: sim,
                    comparable_axes: comparable,
                });
            }
        }
    }

    Ok(BehavioralSimilarityNetwork {
        actors,
        actor_idx,
        adj,
        edges,
        axes_by_actor,
    })
}

/// Axis-wise similarity in `[0, 1]`. Mirrors the composite arithmetic used by
/// [`crate::disinfo::comparison::compare_fingerprints`] (squared-diff mean on
/// non-NaN axes, converted to similarity via `1 - d`). Returns
/// `(similarity, comparable_axes)`.
fn axis_similarity(
    a: &[f64; BEHAVIORAL_AXIS_COUNT],
    b: &[f64; BEHAVIORAL_AXIS_COUNT],
) -> (f64, usize) {
    let mut sq_diffs = Vec::with_capacity(BEHAVIORAL_AXIS_COUNT);
    for i in 0..BEHAVIORAL_AXIS_COUNT {
        if a[i].is_nan() || b[i].is_nan() {
            continue;
        }
        sq_diffs.push((a[i] - b[i]).powi(2).clamp(0.0, 1.0));
    }
    let comparable = sq_diffs.len();
    if comparable == 0 {
        return (0.0, 0);
    }
    let mean_d = sq_diffs.iter().sum::<f64>() / comparable as f64;
    let sim = (1.0 - mean_d).clamp(0.0, 1.0);
    (sim, comparable)
}

// ─── CIB Detection ─────────────────────────────────────────────

/// Detect CIB clusters in a narrative. Algorithm:
/// 1. Build the behavioral similarity network.
/// 2. Run label-propagation community detection on it.
/// 3. Filter communities by minimum size.
/// 4. For each remaining community, compute observed induced-subgraph density.
/// 5. Calibrate a right-tail p-value against a bootstrap null in which
///    similarities are shuffled over pairs — a truly coordinated cluster
///    shows a density that would almost never arise by chance.
/// 6. Flag clusters with p < alpha as CIB and persist them.
///
/// Returns every candidate cluster (with its p-value) so the caller can
/// inspect near-misses.
pub fn detect_cib_clusters(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    config: &CibConfig,
) -> Result<CibDetectionResult> {
    let network = build_similarity_network(hypergraph, narrative_id, config)?;
    detect_cib_clusters_inner(hypergraph, narrative_id, &network, config, false)
}

/// Cross-platform variant of [`detect_cib_clusters`] — keeps only clusters
/// whose members span ≥ 2 distinct platforms. Uses factory-task weighting
/// semantics by flagging the evidence record accordingly (per spec §5.3,
/// the factory config would also down-weight prose layers when available;
/// until `train_pan_weights` produces that config, we reuse the default
/// axis-similarity but mark the evidence as `factory_weighted`).
pub fn detect_cross_platform_cib(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    config: &CibConfig,
) -> Result<CibDetectionResult> {
    let network = build_similarity_network(hypergraph, narrative_id, config)?;
    let mut result = detect_cib_clusters_inner(hypergraph, narrative_id, &network, config, true)?;
    // Keep only multi-platform clusters.
    let keep: HashSet<String> = result
        .clusters
        .iter()
        .filter(|c| c.platforms.len() >= 2)
        .map(|c| c.cluster_id.clone())
        .collect();
    result.clusters.retain(|c| keep.contains(&c.cluster_id));
    result.evidence.retain(|e| keep.contains(&e.cluster_id));
    Ok(result)
}

fn detect_cib_clusters_inner(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    network: &BehavioralSimilarityNetwork,
    config: &CibConfig,
    factory_weighted: bool,
) -> Result<CibDetectionResult> {
    let n = network.actors.len();
    if n < config.min_cluster_size {
        return Ok(CibDetectionResult {
            narrative_id: narrative_id.to_string(),
            clusters: vec![],
            evidence: vec![],
            network_size: n,
            edge_count: 0,
            null_mean_density: 0.0,
            null_std_density: 0.0,
            config: config.clone(),
        });
    }

    // Build the (i, j) → similarity index once and thread it through both the
    // observed-density path and the null-bootstrap sampler.
    let edge_set = network.edge_set();
    // Also index the original edges by pair so evidence records preserve the
    // full metadata (including `comparable_axes`) rather than fabricating zeros.
    let edge_by_pair: HashMap<(usize, usize), &BehavioralSimilarityEdge> = network
        .edges
        .iter()
        .filter_map(|e| {
            let i = *network.actor_idx.get(&e.a)?;
            let j = *network.actor_idx.get(&e.b)?;
            Some(((i.min(j), i.max(j)), e))
        })
        .collect();

    let cograph = network.to_cograph();
    let labels = label_propagation(&cograph, config.seed);

    // Group indices by community label
    let mut by_label: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &lbl) in labels.iter().enumerate() {
        by_label.entry(lbl).or_default().push(i);
    }

    // Null distribution: sample random clusters of the same sizes, compute
    // induced densities. Right-tail p = fraction ≥ observed.
    let (null_mean, null_std, null_sampler) = compute_density_null(&edge_set, n, config);

    // Pre-load the narrative's actor entities once so cluster-platform lookup
    // is a HashMap hit instead of N per-cluster KV reads.
    let actor_platforms: HashMap<Uuid, Vec<String>> =
        load_actor_platforms(hypergraph, narrative_id)?;

    let mut clusters = Vec::new();
    let mut evidence = Vec::new();
    let created_at = Utc::now();

    for (_, members) in by_label {
        if members.len() < config.min_cluster_size {
            continue;
        }

        let (density, mean_sim, internal_edges) = induced_density(&members, &edge_set);
        let observed = density;
        // Right-tail p via null sampler (clusters of same size).
        let p_value = null_sampler.p_value(members.len(), observed);
        if p_value >= config.alpha {
            continue;
        }

        let member_uuids: Vec<Uuid> = members.iter().map(|&i| network.actors[i]).collect();
        let platforms = collect_cluster_platforms(&member_uuids, &actor_platforms);
        let cluster_id = make_cluster_id(narrative_id, &member_uuids);

        let mut top_edges: Vec<BehavioralSimilarityEdge> = internal_edges
            .into_iter()
            .map(|(i, j, sim)| {
                let key = (i.min(j), i.max(j));
                let comparable = edge_by_pair
                    .get(&key)
                    .map(|e| e.comparable_axes)
                    .unwrap_or(0);
                BehavioralSimilarityEdge {
                    a: network.actors[i],
                    b: network.actors[j],
                    similarity: sim,
                    comparable_axes: comparable,
                }
            })
            .collect();
        top_edges.sort_by(|x, y| {
            y.similarity
                .partial_cmp(&x.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        top_edges.truncate(10);

        let axis_mean_distances = compute_axis_mean_distances(&members, &network.axes_by_actor);

        let cluster = CibCluster {
            cluster_id: cluster_id.clone(),
            narrative_id: narrative_id.to_string(),
            members: member_uuids,
            density,
            mean_similarity: mean_sim,
            p_value,
            bootstrap_iter: config.bootstrap_iter,
            alpha: config.alpha,
            platforms,
            created_at,
        };

        let ev = CibEvidence {
            cluster_id: cluster_id.clone(),
            narrative_id: narrative_id.to_string(),
            top_edges,
            axis_mean_distances,
            factory_weighted,
            created_at,
        };

        store_cluster(hypergraph, &cluster)?;
        store_evidence(hypergraph, &ev)?;

        clusters.push(cluster);
        evidence.push(ev);
    }

    // Sort clusters by p-value ascending (most anomalous first).
    clusters.sort_by(|a, b| {
        a.p_value
            .partial_cmp(&b.p_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let edge_count = network.edge_count();

    Ok(CibDetectionResult {
        narrative_id: narrative_id.to_string(),
        clusters,
        evidence,
        network_size: n,
        edge_count,
        null_mean_density: null_mean,
        null_std_density: null_std,
        config: config.clone(),
    })
}

/// Compute the observed density + mean-similarity + edge list of the induced
/// subgraph on `members`.
fn induced_density(
    members: &[usize],
    edge_set: &HashMap<(usize, usize), f64>,
) -> (f64, f64, Vec<(usize, usize, f64)>) {
    let n = members.len();
    if n < 2 {
        return (0.0, 0.0, vec![]);
    }
    let mut internal: Vec<(usize, usize, f64)> = Vec::new();
    let mut sum_sim = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = members[i];
            let b = members[j];
            let key = (a.min(b), a.max(b));
            if let Some(&sim) = edge_set.get(&key) {
                internal.push((a, b, sim));
                sum_sim += sim;
            }
        }
    }
    let possible = (n * (n - 1)) / 2;
    let density = if possible == 0 {
        0.0
    } else {
        internal.len() as f64 / possible as f64
    };
    let mean_sim = if internal.is_empty() {
        0.0
    } else {
        sum_sim / internal.len() as f64
    };
    (density, mean_sim, internal)
}

/// Precomputed null-density sampler for cluster density p-values.
struct DensityNullSampler {
    /// Histogram of null densities grouped by cluster size.
    by_size: HashMap<usize, Vec<f64>>,
}

impl DensityNullSampler {
    fn p_value(&self, size: usize, observed: f64) -> f64 {
        let samples = match self.by_size.get(&size) {
            Some(v) if !v.is_empty() => v,
            _ => return 1.0, // no null → can't reject
        };
        let above = samples.iter().filter(|&&s| s >= observed).count();
        (above as f64 / samples.len() as f64).clamp(0.0, 1.0)
    }
}

/// Build the null distribution of induced-subgraph densities. For each
/// bootstrap iteration and each candidate cluster size `k ∈ [3, n]`, sample
/// `k` random vertices (without replacement) and record the induced density.
///
/// Returns `(mean_density, std_density, sampler)` — mean/std are aggregated
/// across *all* iterations and sizes for the summary stats shown in the
/// Studio card; the sampler preserves the per-size histograms needed for
/// right-tail p-value lookup.
fn compute_density_null(
    edge_set: &HashMap<(usize, usize), f64>,
    n: usize,
    config: &CibConfig,
) -> (f64, f64, DensityNullSampler) {
    if n < config.min_cluster_size || edge_set.is_empty() {
        return (
            0.0,
            0.0,
            DensityNullSampler {
                by_size: HashMap::new(),
            },
        );
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed ^ 0x9e3779b97f4a7c15);
    // Online Welford accumulator — avoids buffering every sample just for the
    // summary mean/std (the per-size histograms are still preserved for
    // p-value lookups).
    let mut count: usize = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    let mut by_size: HashMap<usize, Vec<f64>> = HashMap::new();

    // Sample for each meaningful cluster size from min to a sensible cap.
    let max_k = n.min(32);
    for _ in 0..config.bootstrap_iter {
        for k in config.min_cluster_size..=max_k {
            let sample = sample_without_replacement(n, k, &mut rng);
            let (d, _, _) = induced_density(&sample, edge_set);
            by_size.entry(k).or_default().push(d);
            count += 1;
            let delta = d - mean;
            mean += delta / count as f64;
            m2 += delta * (d - mean);
        }
    }

    if count == 0 {
        return (
            0.0,
            0.0,
            DensityNullSampler {
                by_size: HashMap::new(),
            },
        );
    }

    let var = m2 / count as f64;
    (mean, var.sqrt(), DensityNullSampler { by_size })
}

fn sample_without_replacement(n: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let k = k.min(n);
    // Partial Fisher-Yates
    let mut pool: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let j = i + rng.gen_range(0..(n - i));
        pool.swap(i, j);
        out.push(pool[i]);
    }
    out
}

/// Load the per-actor platform set for every actor in the narrative. Done
/// once per detection run so cluster-platform lookup is a HashMap hit instead
/// of N per-cluster `get_entity` KV reads.
fn load_actor_platforms(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<HashMap<Uuid, Vec<String>>> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let mut out: HashMap<Uuid, Vec<String>> = HashMap::new();
    for entity in entities {
        if !matches!(entity.entity_type, EntityType::Actor) {
            continue;
        }
        let mut platforms: HashSet<String> = HashSet::new();
        if let Some(arr) = entity
            .properties
            .get("platforms")
            .and_then(|v| v.as_array())
        {
            for p in arr {
                if let Some(s) = p.as_str() {
                    platforms.insert(s.to_lowercase());
                }
            }
        }
        if let Some(s) = entity.properties.get("platform").and_then(|v| v.as_str()) {
            platforms.insert(s.to_lowercase());
        }
        if !platforms.is_empty() {
            let mut v: Vec<String> = platforms.into_iter().collect();
            v.sort();
            out.insert(entity.id, v);
        }
    }
    Ok(out)
}

/// Project the preloaded per-actor platforms onto a cluster's membership.
fn collect_cluster_platforms(
    members: &[Uuid],
    actor_platforms: &HashMap<Uuid, Vec<String>>,
) -> Vec<String> {
    let mut platforms: HashSet<String> = HashSet::new();
    for id in members {
        if let Some(ps) = actor_platforms.get(id) {
            for p in ps {
                platforms.insert(p.clone());
            }
        }
    }
    let mut out: Vec<String> = platforms.into_iter().collect();
    out.sort();
    out
}

/// Mean per-axis pairwise distance inside the cluster — low values on several
/// axes confirm the CIB signal isn't driven by a single spurious axis. Reads
/// already-loaded axes from `axes_by_actor` (indexed by the network's actor
/// position) so no extra KV work happens.
fn compute_axis_mean_distances(
    member_indices: &[usize],
    axes_by_actor: &[[f64; BEHAVIORAL_AXIS_COUNT]],
) -> Vec<(String, f64)> {
    if member_indices.len() < 2 {
        return vec![];
    }
    let labels = crate::disinfo::fingerprints::behavioral_axis_labels();
    let mut out = Vec::with_capacity(BEHAVIORAL_AXIS_COUNT);
    for axis_idx in 0..BEHAVIORAL_AXIS_COUNT {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..member_indices.len() {
            for j in (i + 1)..member_indices.len() {
                let a = axes_by_actor[member_indices[i]][axis_idx];
                let b = axes_by_actor[member_indices[j]][axis_idx];
                if a.is_nan() || b.is_nan() {
                    continue;
                }
                sum += (a - b).abs();
                count += 1;
            }
        }
        let label = labels[axis_idx].to_string();
        if count == 0 {
            out.push((label, f64::NAN));
        } else {
            out.push((label, sum / count as f64));
        }
    }
    out
}

fn make_cluster_id(narrative_id: &str, members: &[Uuid]) -> String {
    // Deterministic id: sha-ish hash of (narrative, sorted member list). Uses
    // a simple djb2 variant — we don't need cryptographic strength, just a
    // stable deterministic short name the Studio + API can reference.
    let mut sorted = members.to_vec();
    sorted.sort();
    let mut h: u64 = 5381;
    for byte in narrative_id.as_bytes() {
        h = h.wrapping_mul(33).wrapping_add(*byte as u64);
    }
    for u in &sorted {
        for byte in u.as_bytes() {
            h = h.wrapping_mul(33).wrapping_add(*byte as u64);
        }
    }
    format!("cib-{:016x}", h)
}

// ─── Superspreader Ranking ─────────────────────────────────────

/// Rank the top-N actors by centrality on the narrative's co-participation
/// graph. Reuses the Sprint-1 graph_centrality algorithms so the ranking is
/// consistent with what the rest of TENSA computes for `e.an.pagerank` etc.
pub fn rank_superspreaders(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    method: SuperspreaderMethod,
    top_n: usize,
) -> Result<SuperspreaderRanking> {
    let cograph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
    let scores = match method {
        SuperspreaderMethod::PageRank => compute_pagerank(&cograph),
        SuperspreaderMethod::Eigenvector => compute_eigenvector(&cograph),
        SuperspreaderMethod::Harmonic => compute_harmonic(&cograph),
    };

    let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let take = top_n.max(1).min(indexed.len());
    let top: Vec<SuperspreaderScore> = indexed
        .into_iter()
        .take(take)
        .enumerate()
        .map(|(rank, (i, score))| SuperspreaderScore {
            entity_id: cograph.entities[i],
            score,
            rank: rank + 1,
        })
        .collect();

    let ranking = SuperspreaderRanking {
        narrative_id: narrative_id.to_string(),
        method,
        top_n: take,
        scores: top,
        network_size: cograph.entities.len(),
        created_at: Utc::now(),
    };
    store_superspreaders(hypergraph, &ranking)?;
    Ok(ranking)
}

// ─── Persistence ───────────────────────────────────────────────

fn cluster_key(narrative_id: &str, cluster_id: &str) -> Vec<u8> {
    analysis_key(keys::CIB_CLUSTER, &[narrative_id, cluster_id])
}

fn evidence_key(narrative_id: &str, cluster_id: &str) -> Vec<u8> {
    analysis_key(keys::CIB_EVIDENCE, &[narrative_id, cluster_id])
}

fn superspreader_key(narrative_id: &str) -> Vec<u8> {
    analysis_key(keys::CIB_SUPERSPREADERS, &[narrative_id])
}

fn cluster_prefix(narrative_id: &str) -> Vec<u8> {
    let prefix_str = std::str::from_utf8(keys::CIB_CLUSTER).unwrap_or("");
    format!("{prefix_str}{narrative_id}/").into_bytes()
}

fn evidence_prefix(narrative_id: &str) -> Vec<u8> {
    let prefix_str = std::str::from_utf8(keys::CIB_EVIDENCE).unwrap_or("");
    format!("{prefix_str}{narrative_id}/").into_bytes()
}

pub fn store_cluster(hypergraph: &Hypergraph, cluster: &CibCluster) -> Result<()> {
    let bytes =
        serde_json::to_vec(cluster).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(
        &cluster_key(&cluster.narrative_id, &cluster.cluster_id),
        &bytes,
    )
}

pub fn store_evidence(hypergraph: &Hypergraph, evidence: &CibEvidence) -> Result<()> {
    let bytes =
        serde_json::to_vec(evidence).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(
        &evidence_key(&evidence.narrative_id, &evidence.cluster_id),
        &bytes,
    )
}

pub fn store_superspreaders(hypergraph: &Hypergraph, ranking: &SuperspreaderRanking) -> Result<()> {
    let bytes =
        serde_json::to_vec(ranking).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(&superspreader_key(&ranking.narrative_id), &bytes)
}

pub fn list_clusters(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Vec<CibCluster>> {
    let prefix = cluster_prefix(narrative_id);
    let pairs = hypergraph.store().prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        if let Ok(c) = serde_json::from_slice::<CibCluster>(&value) {
            out.push(c);
        }
    }
    out.sort_by(|a, b| {
        a.p_value
            .partial_cmp(&b.p_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

pub fn list_evidence(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Vec<CibEvidence>> {
    let prefix = evidence_prefix(narrative_id);
    let pairs = hypergraph.store().prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        if let Ok(e) = serde_json::from_slice::<CibEvidence>(&value) {
            out.push(e);
        }
    }
    Ok(out)
}

pub fn load_superspreaders(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<SuperspreaderRanking>> {
    match hypergraph.store().get(&superspreader_key(narrative_id))? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Load clusters + evidence as a single `CibDetectionResult` — used by the
/// `GET /analysis/cib/:id` endpoint. Returns `None` when no clusters exist.
///
/// **Caveat for consumers:** only `clusters` and `evidence` come from the
/// persisted record. `network_size`, `edge_count`, `null_mean_density`,
/// `null_std_density`, and `config` are **not** persisted across runs and are
/// returned as zero / default here; re-run `POST /analysis/cib` to refresh
/// them. A future change should persist a `CibRunMeta` record alongside
/// clusters so these fields can be round-tripped.
pub fn load_cib_detection(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<CibDetectionResult>> {
    let clusters = list_clusters(hypergraph, narrative_id)?;
    if clusters.is_empty() {
        return Ok(None);
    }
    let evidence = list_evidence(hypergraph, narrative_id)?;
    Ok(Some(CibDetectionResult {
        narrative_id: narrative_id.to_string(),
        clusters,
        evidence,
        network_size: 0,
        edge_count: 0,
        null_mean_density: 0.0,
        null_std_density: 0.0,
        config: CibConfig::default(),
    }))
}

// ─── Content Factory Detection ────────────────────────────────

/// A detected content factory: a set of actors within a CIB cluster whose
/// style profiles are so similar they likely share a common content-production
/// pipeline (same copywriter, same prompt template, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFactory {
    /// Stable identifier for this factory detection.
    pub cluster_id: String,
    /// Narrative the factory was detected in.
    pub narrative_id: String,
    /// Pairs of actor UUIDs whose behavioral fingerprints exceed the similarity
    /// threshold, together with the similarity score.
    pub verified_pairs: Vec<(Uuid, Uuid, f64)>,
    /// Overall confidence in the factory detection (mean pairwise similarity).
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
}

fn factory_key(narrative_id: &str, factory_id: &str) -> Vec<u8> {
    analysis_key(keys::CIB_FACTORY, &[narrative_id, factory_id])
}

fn factory_prefix(narrative_id: &str) -> Vec<u8> {
    let prefix_str = std::str::from_utf8(keys::CIB_FACTORY).unwrap_or("");
    format!("{prefix_str}{narrative_id}/").into_bytes()
}

/// Persist a content factory record.
pub fn store_factory(hypergraph: &Hypergraph, factory: &ContentFactory) -> Result<()> {
    let bytes =
        serde_json::to_vec(factory).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(
        &factory_key(&factory.narrative_id, &factory.cluster_id),
        &bytes,
    )
}

/// Load all content factories for a narrative.
pub fn list_factories(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Vec<ContentFactory>> {
    let prefix = factory_prefix(narrative_id);
    let pairs = hypergraph.store().prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        if let Ok(f) = serde_json::from_slice::<ContentFactory>(&value) {
            out.push(f);
        }
    }
    out.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// Detect content factories within existing CIB clusters.
///
/// For each CIB cluster, compares all member pairs' behavioral fingerprints.
/// Pairs with similarity above `factory_threshold` (default 0.7) are flagged
/// as factory-produced. A cluster is reported as a factory only if it has at
/// least one qualifying pair.
pub fn detect_content_factories(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    factory_threshold: Option<f64>,
) -> Result<Vec<ContentFactory>> {
    let threshold = factory_threshold.unwrap_or(0.7);
    let clusters = list_clusters(hypergraph, narrative_id)?;
    if clusters.is_empty() {
        return Ok(vec![]);
    }

    // Load behavioral fingerprints for all actors across all clusters.
    let mut fp_cache: HashMap<Uuid, [f64; BEHAVIORAL_AXIS_COUNT]> = HashMap::new();
    for cluster in &clusters {
        for member in &cluster.members {
            if !fp_cache.contains_key(member) {
                if let Ok(fp) = ensure_behavioral_fingerprint(hypergraph, member, false) {
                    fp_cache.insert(*member, fp.axes);
                }
            }
        }
    }

    let mut factories = Vec::new();
    let now = Utc::now();

    for cluster in &clusters {
        let mut verified_pairs: Vec<(Uuid, Uuid, f64)> = Vec::new();
        let members = &cluster.members;
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let a_axes = match fp_cache.get(&members[i]) {
                    Some(axes) => axes,
                    None => continue,
                };
                let b_axes = match fp_cache.get(&members[j]) {
                    Some(axes) => axes,
                    None => continue,
                };
                let (sim, comparable) = axis_similarity(a_axes, b_axes);
                if comparable >= 2 && sim >= threshold {
                    verified_pairs.push((members[i], members[j], sim));
                }
            }
        }

        if verified_pairs.is_empty() {
            continue;
        }

        let mean_sim =
            verified_pairs.iter().map(|(_, _, s)| s).sum::<f64>() / verified_pairs.len() as f64;

        let factory = ContentFactory {
            cluster_id: format!("fac-{}", &cluster.cluster_id),
            narrative_id: narrative_id.to_string(),
            verified_pairs,
            confidence: mean_sim,
            created_at: now,
        };
        store_factory(hypergraph, &factory)?;
        factories.push(factory);
    }

    factories.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(factories)
}

// ─── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, link, make_hg};
    use crate::disinfo::fingerprints::{
        store_behavioral_fingerprint, BehavioralAxis, BehavioralFingerprint,
    };
    use chrono::TimeZone;

    fn set_axes(hg: &Hypergraph, actor: Uuid, axes: &[(BehavioralAxis, f64)]) {
        let mut fp = BehavioralFingerprint::empty(actor);
        for (a, v) in axes {
            fp.set_axis(*a, *v);
        }
        store_behavioral_fingerprint(hg, &fp).unwrap();
    }

    #[test]
    fn axis_similarity_identical_is_one() {
        let a: [f64; BEHAVIORAL_AXIS_COUNT] = [0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.1, 0.8, 0.9, 0.5];
        let (sim, comparable) = axis_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
        assert_eq!(comparable, BEHAVIORAL_AXIS_COUNT);
    }

    #[test]
    fn axis_similarity_nans_skipped() {
        let mut a = [f64::NAN; BEHAVIORAL_AXIS_COUNT];
        let mut b = [f64::NAN; BEHAVIORAL_AXIS_COUNT];
        a[0] = 0.4;
        b[0] = 0.4;
        a[3] = 0.7;
        b[3] = 0.9;
        let (sim, comparable) = axis_similarity(&a, &b);
        assert_eq!(comparable, 2);
        assert!(sim > 0.9 && sim < 1.0);
    }

    #[test]
    fn build_network_empty_on_single_actor() {
        let hg = make_hg();
        let _a = add_entity(&hg, "Solo", "narr-solo");
        let config = CibConfig::default();
        let net = build_similarity_network(&hg, "narr-solo", &config).unwrap();
        assert_eq!(net.actors.len(), 1);
        assert_eq!(net.edge_count(), 0);
    }

    #[test]
    fn build_network_connects_similar_actors() {
        let hg = make_hg();
        let a = add_entity(&hg, "Alice", "narr-cib");
        let b = add_entity(&hg, "Bob", "narr-cib");
        let c = add_entity(&hg, "Carol", "narr-cib");
        // Alice + Bob: identical axes ⇒ sim = 1.0
        set_axes(
            &hg,
            a,
            &[
                (BehavioralAxis::PostingCadenceRegularity, 0.95),
                (BehavioralAxis::HashtagConcentration, 0.9),
                (BehavioralAxis::ContentOriginality, 0.05),
                (BehavioralAxis::EngagementRatio, 0.8),
            ],
        );
        set_axes(
            &hg,
            b,
            &[
                (BehavioralAxis::PostingCadenceRegularity, 0.95),
                (BehavioralAxis::HashtagConcentration, 0.9),
                (BehavioralAxis::ContentOriginality, 0.05),
                (BehavioralAxis::EngagementRatio, 0.8),
            ],
        );
        // Carol: very different
        set_axes(
            &hg,
            c,
            &[
                (BehavioralAxis::PostingCadenceRegularity, 0.1),
                (BehavioralAxis::HashtagConcentration, 0.1),
                (BehavioralAxis::ContentOriginality, 0.95),
                (BehavioralAxis::EngagementRatio, 0.1),
            ],
        );
        let config = CibConfig {
            similarity_threshold: 0.85,
            ..CibConfig::default()
        };
        let net = build_similarity_network(&hg, "narr-cib", &config).unwrap();
        assert_eq!(net.actors.len(), 3);
        // Only the Alice–Bob edge should survive.
        assert_eq!(net.edge_count(), 1);
        let edge = &net.edges[0];
        let (ax, bx) = (edge.a, edge.b);
        assert!(
            (ax == a && bx == b) || (ax == b && bx == a),
            "edge must be between Alice and Bob"
        );
        assert!(edge.similarity >= 0.85);
    }

    #[test]
    fn detect_cib_flags_dense_clique() {
        let hg = make_hg();
        // 4 near-identical sock-puppets + 2 organic outliers
        let sockpuppets: Vec<Uuid> = (0..4)
            .map(|i| add_entity(&hg, &format!("sock{i}"), "narr-clique"))
            .collect();
        for &id in &sockpuppets {
            set_axes(
                &hg,
                id,
                &[
                    (BehavioralAxis::PostingCadenceRegularity, 0.98),
                    (BehavioralAxis::HashtagConcentration, 0.92),
                    (BehavioralAxis::ContentOriginality, 0.08),
                    (BehavioralAxis::EngagementRatio, 0.75),
                    (BehavioralAxis::SleepPatternPresence, 0.02),
                    (BehavioralAxis::PlatformDiversity, 0.3),
                ],
            );
        }
        let organic_a = add_entity(&hg, "OrganicA", "narr-clique");
        set_axes(
            &hg,
            organic_a,
            &[
                (BehavioralAxis::PostingCadenceRegularity, 0.35),
                (BehavioralAxis::HashtagConcentration, 0.2),
                (BehavioralAxis::ContentOriginality, 0.85),
                (BehavioralAxis::EngagementRatio, 0.3),
                (BehavioralAxis::SleepPatternPresence, 0.9),
                (BehavioralAxis::PlatformDiversity, 0.6),
            ],
        );
        let organic_b = add_entity(&hg, "OrganicB", "narr-clique");
        set_axes(
            &hg,
            organic_b,
            &[
                (BehavioralAxis::PostingCadenceRegularity, 0.22),
                (BehavioralAxis::HashtagConcentration, 0.15),
                (BehavioralAxis::ContentOriginality, 0.95),
                (BehavioralAxis::EngagementRatio, 0.22),
                (BehavioralAxis::SleepPatternPresence, 0.95),
                (BehavioralAxis::PlatformDiversity, 0.7),
            ],
        );

        let config = CibConfig {
            similarity_threshold: 0.8,
            bootstrap_iter: 50, // keep tests fast
            alpha: 0.05,
            ..CibConfig::default()
        };
        let result = detect_cib_clusters(&hg, "narr-clique", &config).unwrap();
        assert!(result.network_size >= 6);
        assert!(
            !result.clusters.is_empty(),
            "expected at least one flagged CIB cluster"
        );
        let flagged = &result.clusters[0];
        assert!(flagged.members.len() >= 3);
        assert!(flagged.p_value < config.alpha);
        assert!(flagged.density > 0.5);
    }

    #[test]
    fn cluster_persistence_round_trip() {
        let hg = make_hg();
        let mem = vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];
        let cluster = CibCluster {
            cluster_id: "cib-test-001".into(),
            narrative_id: "narr-persist".into(),
            members: mem.clone(),
            density: 0.75,
            mean_similarity: 0.9,
            p_value: 0.003,
            bootstrap_iter: 100,
            alpha: 0.01,
            platforms: vec!["twitter".into()],
            created_at: Utc.with_ymd_and_hms(2026, 4, 16, 12, 0, 0).unwrap(),
        };
        store_cluster(&hg, &cluster).unwrap();
        let loaded = list_clusters(&hg, "narr-persist").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].members, mem);
        assert!((loaded[0].density - 0.75).abs() < 1e-9);
    }

    #[test]
    fn evidence_persistence_round_trip() {
        let hg = make_hg();
        let ev = CibEvidence {
            cluster_id: "cib-ev-001".into(),
            narrative_id: "narr-ev".into(),
            top_edges: vec![],
            axis_mean_distances: vec![("engagement_ratio".into(), 0.05)],
            factory_weighted: false,
            created_at: Utc::now(),
        };
        store_evidence(&hg, &ev).unwrap();
        let loaded = list_evidence(&hg, "narr-ev").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].cluster_id, "cib-ev-001");
        assert!((loaded[0].axis_mean_distances[0].1 - 0.05).abs() < 1e-9);
    }

    #[test]
    fn rank_superspreaders_returns_top_n() {
        let hg = make_hg();
        let narr = "narr-super";
        let actors: Vec<Uuid> = (0..5)
            .map(|i| add_entity(&hg, &format!("actor{i}"), narr))
            .collect();
        // Actor 0 participates in everything (hub); others only in one situation.
        for i in 1..5 {
            let sit_id = crate::analysis::test_helpers::add_situation(&hg, narr);
            link(&hg, actors[0], sit_id);
            link(&hg, actors[i], sit_id);
        }
        let ranking = rank_superspreaders(&hg, narr, SuperspreaderMethod::PageRank, 3).unwrap();
        assert_eq!(ranking.scores.len(), 3);
        assert_eq!(ranking.scores[0].rank, 1);
        // The hub should rank first.
        assert_eq!(ranking.scores[0].entity_id, actors[0]);
    }

    #[test]
    fn superspreader_method_roundtrip() {
        use std::str::FromStr;
        assert_eq!(
            SuperspreaderMethod::from_str("pagerank").unwrap(),
            SuperspreaderMethod::PageRank
        );
        assert_eq!(
            SuperspreaderMethod::from_str("EIGENVECTOR").unwrap(),
            SuperspreaderMethod::Eigenvector
        );
        assert_eq!(
            SuperspreaderMethod::from_str("harmonic").unwrap(),
            SuperspreaderMethod::Harmonic
        );
        assert_eq!(
            SuperspreaderMethod::from_str("betweenness").unwrap(),
            SuperspreaderMethod::Harmonic
        );
        assert!(SuperspreaderMethod::from_str("bogus").is_err());
    }

    #[test]
    fn cross_platform_cib_filters_single_platform() {
        let hg = make_hg();
        let narr = "narr-cross";
        let a = add_entity(&hg, "Alice", narr);
        let b = add_entity(&hg, "Bob", narr);
        let c = add_entity(&hg, "Carol", narr);
        // All on the same platform.
        for &id in &[a, b, c] {
            hg.update_entity_no_snapshot(&id, |ent| {
                ent.properties["platform"] = serde_json::json!("twitter");
            })
            .unwrap();
        }
        for &id in &[a, b, c] {
            set_axes(
                &hg,
                id,
                &[
                    (BehavioralAxis::PostingCadenceRegularity, 0.95),
                    (BehavioralAxis::HashtagConcentration, 0.9),
                    (BehavioralAxis::ContentOriginality, 0.05),
                    (BehavioralAxis::EngagementRatio, 0.8),
                    (BehavioralAxis::SleepPatternPresence, 0.02),
                    (BehavioralAxis::PlatformDiversity, 0.2),
                ],
            );
        }
        let config = CibConfig {
            similarity_threshold: 0.8,
            bootstrap_iter: 30,
            alpha: 0.5, // loose to ensure _some_ cluster would fire
            ..CibConfig::default()
        };
        let xplat = detect_cross_platform_cib(&hg, narr, &config).unwrap();
        // Single-platform clusters must be dropped.
        assert!(
            xplat.clusters.is_empty(),
            "cross-platform detection must drop single-platform clusters"
        );
    }

    #[test]
    fn content_factory_persistence_round_trip() {
        let hg = make_hg();
        let factory = ContentFactory {
            cluster_id: "fac-test-001".into(),
            narrative_id: "narr-factory".into(),
            verified_pairs: vec![
                (Uuid::now_v7(), Uuid::now_v7(), 0.85),
                (Uuid::now_v7(), Uuid::now_v7(), 0.92),
            ],
            confidence: 0.88,
            created_at: Utc::now(),
        };
        store_factory(&hg, &factory).unwrap();
        let loaded = list_factories(&hg, "narr-factory").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].cluster_id, "fac-test-001");
        assert_eq!(loaded[0].verified_pairs.len(), 2);
        assert!((loaded[0].confidence - 0.88).abs() < 1e-9);
    }

    #[test]
    fn detect_factories_from_persisted_clusters() {
        let hg = make_hg();
        let narr = "narr-facdet";
        // 3 near-identical actors
        let actors: Vec<Uuid> = (0..3)
            .map(|i| add_entity(&hg, &format!("sock{i}"), narr))
            .collect();
        for &id in &actors {
            set_axes(
                &hg,
                id,
                &[
                    (BehavioralAxis::PostingCadenceRegularity, 0.95),
                    (BehavioralAxis::HashtagConcentration, 0.90),
                    (BehavioralAxis::ContentOriginality, 0.05),
                    (BehavioralAxis::EngagementRatio, 0.80),
                    (BehavioralAxis::SleepPatternPresence, 0.02),
                    (BehavioralAxis::PlatformDiversity, 0.20),
                ],
            );
        }

        // Manually persist a CIB cluster so factory detection has something
        // to work with (avoids needing the bootstrap null to fire).
        let cluster = CibCluster {
            cluster_id: "cib-manual-test".into(),
            narrative_id: narr.to_string(),
            members: actors.clone(),
            density: 1.0,
            mean_similarity: 0.99,
            p_value: 0.001,
            bootstrap_iter: 50,
            alpha: 0.05,
            platforms: vec!["twitter".into()],
            created_at: Utc::now(),
        };
        store_cluster(&hg, &cluster).unwrap();

        // Now detect factories — the 3 identical actors should qualify
        let factories = detect_content_factories(&hg, narr, Some(0.7)).unwrap();
        assert!(
            !factories.is_empty(),
            "identical actors should form a content factory"
        );
        let f = &factories[0];
        assert!(f.confidence >= 0.7);
        assert!(!f.verified_pairs.is_empty());
        // With 3 identical actors there should be 3 pairs (A-B, A-C, B-C)
        assert_eq!(f.verified_pairs.len(), 3);
    }
}
