//! Metric adapters + K-loop helper for [`super::SurrogateSignificanceEngine`].
//!
//! Each adapter knows how to:
//! 1. Compute the metric on an arbitrary [`Hypergraph`] scoped to a
//!    `narrative_id`, returning element keys + values.
//! 2. Serialize the source observation as JSON for archival in the report.
//!
//! The K-loop helper [`run_significance_pipeline`] runs the metric on the
//! source narrative ONCE, then dispatches K parallel synthetic generations
//! (each using its own ephemeral `MemoryStore` for isolation), and aligns
//! the per-element distributions across the union of keys seen.
//!
//! Determinism: per-sample seed = `base_seed XOR (k_idx * SAMPLE_SEED_MIX)`,
//! reusing the constant from `fidelity.rs`. Same algebra means single-thread
//! and multi-thread runs produce identical reports.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::json;

use crate::analysis::community_detect::label_propagation;
use crate::analysis::graph_projection::{build_co_graph, CoGraph};
use crate::analysis::temporal_motifs::{temporal_motif_census, MotifCensus, TemporalMotif};
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::narrative::pattern::{mine_patterns_with_config, NarrativePattern, PatternMiningConfig};
use crate::narrative::subgraph::NarrativeGraph;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;

use super::super::eath::EathSurrogate;
use super::super::fidelity::{ParallelismMode, SAMPLE_SEED_MIX};
use super::super::surrogate::SurrogateModel;
use super::super::types::{EathParams, SurrogateParams};
use super::stats::build_distribution;
use super::SyntheticDistribution;

// ── Public dispatch ─────────────────────────────────────────────────────────

/// Selects which adapter the engine should use. One enum variant per shipped
/// metric — no string dispatch in the hot path.
///
/// `Contagion` is Phase 7b. Unlike the other three (which are scalar,
/// stateless-after-construction observers), the contagion adapter carries
/// per-call simulation parameters in its `&'a HigherOrderSirParams` payload.
/// Lifetime keeps the params alive for the duration of the K-loop.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AdapterChoice<'a> {
    TemporalMotifs,
    Communities,
    Patterns,
    /// Phase 7b — higher-order contagion observed over K surrogates.
    /// Carries the simulation params verbatim so each sample uses the
    /// same dynamics on its synthetic substrate.
    Contagion(&'a crate::analysis::higher_order_contagion::HigherOrderSirParams),
}

/// Output of the K-loop pipeline.
pub(crate) struct PipelineResult {
    pub(crate) source_observation: serde_json::Value,
    pub(crate) distribution: SyntheticDistribution,
    /// True when the metric returned no observations on the SOURCE narrative.
    /// Doesn't suppress the report — just toggles the engine's `note` field.
    pub(crate) source_was_empty: bool,
}

/// The single source of truth for the K-sample loop. Computes the source
/// observation ONCE (avoiding the classic N+1 inside the loop), then runs
/// K parallel synthetic samples, aligns element keys, and computes
/// per-element z-scores + p-values.
pub(crate) fn run_significance_pipeline(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &EathParams,
    adapter: AdapterChoice,
    metric_params: &serde_json::Value,
    k: usize,
    parallelism: ParallelismMode,
) -> Result<PipelineResult> {
    let base_seed = params_seed(params);

    // 1. Source observation — ONCE, outside the loop.
    let source = compute_for(adapter, hypergraph, narrative_id, metric_params)?;
    let source_observation = serialize_source(adapter, &source);
    let source_was_empty = source.is_empty();

    // 2. K-sample synthetic loop. Each sample writes into its OWN ephemeral
    //    MemoryStore so no inter-sample contention is possible.
    let synthetic_results: Vec<MetricSnapshot> =
        run_k_samples(params, base_seed, k, parallelism, adapter, metric_params)?;

    // 3. Build the aligned distribution.
    let distribution = build_distribution(&source, &synthetic_results, adapter);

    Ok(PipelineResult {
        source_observation,
        distribution,
        source_was_empty,
    })
}

// ── Per-sample metric snapshot ───────────────────────────────────────────────

/// One adapter call's output: aligned canonical-key → scalar map.
/// `HashMap` for cheap union assembly across K samples. The `values` field
/// is `pub(crate)` so `super::stats::build_distribution` can read it
/// without duplicating accessors.
#[derive(Debug, Clone, Default)]
pub(crate) struct MetricSnapshot {
    pub(crate) values: HashMap<String, f64>,
}

impl MetricSnapshot {
    fn is_empty(&self) -> bool {
        // "Empty" means every entry is zero / no entries — the metric saw
        // nothing to count. Communities always emits 2 keys even on an empty
        // graph (`num_communities=0`, `modularity=0`), so we use sum>0.
        self.values.values().all(|&v| v == 0.0)
    }
}

// ── Adapter dispatcher ───────────────────────────────────────────────────────

/// Compute `metric` on `hypergraph` scoped to `narrative_id`. Visible to
/// sibling engines (Phase 13c's dual-significance engine reuses it so both
/// the source-observation path and NuDHy K-sample path share one dispatcher).
pub(crate) fn compute_for(
    adapter: AdapterChoice,
    hypergraph: &Hypergraph,
    narrative_id: &str,
    metric_params: &serde_json::Value,
) -> Result<MetricSnapshot> {
    match adapter {
        AdapterChoice::TemporalMotifs => compute_temporal_motifs(hypergraph, narrative_id, metric_params),
        AdapterChoice::Communities => compute_communities(hypergraph, narrative_id),
        AdapterChoice::Patterns => compute_patterns(hypergraph, narrative_id),
        AdapterChoice::Contagion(params) => compute_contagion(hypergraph, narrative_id, params),
    }
}

fn serialize_source(adapter: AdapterChoice, snapshot: &MetricSnapshot) -> serde_json::Value {
    match adapter {
        AdapterChoice::TemporalMotifs => json!({
            "metric": "temporal_motifs",
            "counts_by_key": snapshot.values,
        }),
        AdapterChoice::Communities => json!({
            "metric": "communities",
            "num_communities": snapshot.values.get("num_communities").copied().unwrap_or(0.0),
            "modularity": snapshot.values.get("modularity").copied().unwrap_or(0.0),
        }),
        AdapterChoice::Patterns => json!({
            "metric": "patterns",
            "presence_by_key": snapshot.values,
        }),
        AdapterChoice::Contagion(_) => json!({
            "metric": "contagion",
            "scalars": snapshot.values,
        }),
    }
}

// ── Temporal Motifs adapter ──────────────────────────────────────────────────

const MOTIF_DEFAULT_MAX_SIZE: usize = 3;
const MOTIF_TOTAL_KEY: &str = "_total_motifs";

fn compute_temporal_motifs(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    metric_params: &serde_json::Value,
) -> Result<MetricSnapshot> {
    let max_size = metric_params
        .get("max_motif_size")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(MOTIF_DEFAULT_MAX_SIZE);
    let census = temporal_motif_census(hypergraph, narrative_id, max_size)?;
    Ok(motif_census_to_snapshot(&census))
}

fn motif_census_to_snapshot(census: &MotifCensus) -> MetricSnapshot {
    let mut values: HashMap<String, f64> =
        HashMap::with_capacity(census.motifs.len() + 1);
    for (motif, count) in &census.motifs {
        values.insert(motif_to_key(motif), *count as f64);
    }
    // Always include the summary row so it exists across all K samples.
    values.insert(MOTIF_TOTAL_KEY.into(), census.total_motifs_found as f64);
    MetricSnapshot { values }
}

/// Canonical motif key. Stable as long as `AllenRelation` and
/// `NarrativeLevel` Debug formats stay variant-name-only — both are C-like
/// enums so this is currently true.
fn motif_to_key(m: &TemporalMotif) -> String {
    let levels: String = m
        .situation_levels
        .iter()
        .map(|l| format!("{l:?}"))
        .collect::<Vec<_>>()
        .join("");
    let rels: String = m
        .temporal_relations
        .iter()
        .map(|r| format!("{r:?}"))
        .collect::<Vec<_>>()
        .join("");
    format!("motif_sz{}_lev{levels}_rel{rels}", m.size)
}

// ── Communities adapter ──────────────────────────────────────────────────────

const COMMUNITIES_KEY_NUM: &str = "num_communities";
const COMMUNITIES_KEY_MOD: &str = "modularity";
const COMMUNITIES_LABEL_PROP_SEED: u64 = 42;

fn compute_communities(hypergraph: &Hypergraph, narrative_id: &str) -> Result<MetricSnapshot> {
    let graph = build_co_graph(hypergraph, narrative_id)?;
    let labels = label_propagation(&graph, COMMUNITIES_LABEL_PROP_SEED);
    let num = if labels.is_empty() {
        0.0
    } else {
        labels.iter().collect::<HashSet<_>>().len() as f64
    };
    let q = compute_modularity(&graph, &labels);
    let mut values = HashMap::with_capacity(2);
    values.insert(COMMUNITIES_KEY_NUM.to_string(), num);
    values.insert(COMMUNITIES_KEY_MOD.to_string(), q);
    Ok(MetricSnapshot { values })
}

/// Weighted modularity on the CoGraph adjacency. Formula:
/// `Q = Σ_c [W_c / (2W) - (D_c / (2W))²]`, where `W` is total weight,
/// `W_c` is intra-community weight, `D_c` is community degree-sum.
///
/// Uses weighted edges to stay consistent with `label_propagation`'s
/// weighted-vote semantics (Q4 default per design doc §11).
fn compute_modularity(graph: &CoGraph, labels: &[usize]) -> f64 {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }
    // Total weight (each undirected edge appears twice in adjacency lists).
    let total_double: usize = graph
        .adj
        .iter()
        .map(|row| row.iter().map(|&(_, w)| w).sum::<usize>())
        .sum();
    if total_double == 0 {
        return 0.0;
    }
    let two_w = total_double as f64;

    let mut intra_double_per_c: HashMap<usize, f64> = HashMap::new();
    let mut deg_per_c: HashMap<usize, f64> = HashMap::new();
    for i in 0..n {
        let ci = labels[i];
        for &(j, w) in &graph.adj[i] {
            let wf = w as f64;
            *deg_per_c.entry(ci).or_insert(0.0) += wf;
            if labels[j] == ci {
                // Counted twice across (i,j) and (j,i); we'll halve when
                // dividing by 2W.
                *intra_double_per_c.entry(ci).or_insert(0.0) += wf;
            }
        }
    }

    let mut q = 0.0;
    for (&c, &intra_double) in &intra_double_per_c {
        let dc = deg_per_c.get(&c).copied().unwrap_or(0.0);
        // intra_double already counts each intra-edge twice, so dividing by 2W
        // gives the correct W_c / (2W) ratio without halving.
        let term1 = intra_double / two_w;
        let term2 = (dc / two_w).powi(2);
        q += term1 - term2;
    }
    q
}

// ── Patterns adapter ─────────────────────────────────────────────────────────

const PATTERN_MAX_CHAIN_LENGTH: usize = 4;
const PATTERN_MAX_STAR_SIZE: usize = 3;
const PATTERN_MAX_PATTERNS: usize = 500;

fn compute_patterns(hypergraph: &Hypergraph, narrative_id: &str) -> Result<MetricSnapshot> {
    let graph = NarrativeGraph::extract(narrative_id, hypergraph)?;
    let config = PatternMiningConfig {
        min_support: 1,
        max_chain_length: PATTERN_MAX_CHAIN_LENGTH,
        max_star_size: PATTERN_MAX_STAR_SIZE,
        enable_star_motifs: true,
        max_patterns: PATTERN_MAX_PATTERNS,
    };
    let patterns = mine_patterns_with_config(&[graph], &config);

    // Per design doc §9 + Q1: `mine_patterns_with_config` on a single graph
    // sets `frequency = num_narratives_containing = 1` for every discovered
    // pattern. We convert to PRESENCE/ABSENCE (binary 1.0/0.0) — the only
    // statistically meaningful interpretation in a single-graph null test.
    let mut values: HashMap<String, f64> = HashMap::with_capacity(patterns.len());
    for p in &patterns {
        values.insert(pattern_to_key(p), 1.0);
    }
    Ok(MetricSnapshot { values })
}

/// Canonical pattern key from the subgraph structure (NOT the UUID id).
/// Sorted node + edge type lists give a stable canonical form.
fn pattern_to_key(p: &NarrativePattern) -> String {
    let mut node_types: Vec<String> = p
        .subgraph
        .nodes
        .iter()
        .map(|n| {
            n.node_type
                .as_ref()
                .map(|t| format!("{t:?}"))
                .unwrap_or_else(|| "Unknown".into())
        })
        .collect();
    node_types.sort();
    let mut edge_types: Vec<String> = p
        .subgraph
        .edges
        .iter()
        .map(|e| format!("{:?}", e.edge_type))
        .collect();
    edge_types.sort();
    format!("nodes={};edges={}", node_types.join(","), edge_types.join(","))
}

// ── Contagion adapter (Phase 7b) ─────────────────────────────────────────────

const CONTAGION_KEY_PEAK_PREVALENCE: &str = "peak_prevalence";
const CONTAGION_KEY_R0_ESTIMATE: &str = "r0_estimate";
const CONTAGION_KEY_TIME_TO_PEAK: &str = "time_to_peak";
const CONTAGION_KEY_TOTAL_INFECTED: &str = "total_infected";

/// Compute the higher-order SIR scalar metrics on the substrate. Element keys
/// are scalars chosen so they're meaningful in BOTH the source narrative and
/// the K synthetic surrogates (no per-Uuid keys — those wouldn't align across
/// the synthetic Vec).
pub(crate) fn compute_contagion(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &crate::analysis::higher_order_contagion::HigherOrderSirParams,
) -> Result<MetricSnapshot> {
    let result = crate::analysis::higher_order_contagion::simulate_higher_order_sir(
        hypergraph,
        narrative_id,
        params,
    )?;
    let mut values: HashMap<String, f64> = HashMap::with_capacity(4 + result.size_attribution.len());
    values.insert(CONTAGION_KEY_PEAK_PREVALENCE.into(), result.peak_prevalence as f64);
    values.insert(CONTAGION_KEY_R0_ESTIMATE.into(), result.r0_estimate as f64);
    values.insert(CONTAGION_KEY_TIME_TO_PEAK.into(), result.time_to_peak as f64);
    let total_infected: u64 = result.size_attribution.iter().sum();
    values.insert(CONTAGION_KEY_TOTAL_INFECTED.into(), total_infected as f64);
    for (i, count) in result.size_attribution.iter().enumerate() {
        // i=0 → size 2, i=1 → size 3, etc.
        let key = format!("size_attribution_d{}", i + 2);
        values.insert(key, *count as f64);
    }
    Ok(MetricSnapshot { values })
}

// ── K-sample loop ────────────────────────────────────────────────────────────

/// Per-sample seed mix — identical algebra to `fidelity.rs` so the first
/// `min(K_fidelity, K_significance)` synthetic narratives match bit-for-bit
/// across the two systems. Cheap optimization opportunity for callers that
/// want to share generated narratives between fidelity and significance runs.
#[inline]
fn per_sample_seed(base_seed: u64, sample_idx: usize) -> u64 {
    base_seed ^ (sample_idx as u64).wrapping_mul(SAMPLE_SEED_MIX)
}

/// `params.seed` isn't a field — derive a deterministic base seed from the
/// EathParams shape so reruns of the same calibration produce reproducible
/// significance reports. Uses `num_entities + max_group_size + group_size_dist`
/// length as a lightweight digest.
fn params_seed(params: &EathParams) -> u64 {
    let mut seed: u64 = 0xA5A5_A5A5_5A5A_5A5A;
    seed ^= params.num_entities as u64;
    seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    seed ^= params.max_group_size as u64;
    seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    seed ^= params.group_size_distribution.len() as u64;
    seed
}

/// Generate one synthetic narrative into an ephemeral MemoryStore and return
/// its metric snapshot. Hypergraph + store dropped at function exit.
fn generate_one_sample(
    params: &EathParams,
    base_seed: u64,
    sample_idx: usize,
    adapter: AdapterChoice,
    metric_params: &serde_json::Value,
) -> Result<MetricSnapshot> {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let target = Hypergraph::new(store);
    let envelope = SurrogateParams {
        model: "eath".into(),
        params_json: serde_json::to_value(params)?,
        seed: per_sample_seed(base_seed, sample_idx),
        num_steps: 200,
        label_prefix: "sig-sample".into(),
    };
    // Per Q5 default — every K-sample has its OWN MemoryStore so a single
    // stable narrative_id is fine; the per-sample suffix keeps the records
    // unique anyway.
    let output_narrative_id = format!("sig-sample-{sample_idx}");
    let summary = EathSurrogate.generate(&envelope, &target, &output_narrative_id)?;
    compute_for(adapter, &target, &summary.output_narrative_id, metric_params)
}

/// Resolve `Auto` parallelism into a thread count, capped at K. Mirrors the
/// `fidelity_pipeline.rs::resolve_thread_count` helper.
fn resolve_thread_count(mode: ParallelismMode, k: usize) -> usize {
    let raw = match mode {
        ParallelismMode::Single => 1,
        ParallelismMode::Threads(n) => n.max(1),
        ParallelismMode::Auto => std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(1),
    };
    raw.min(k).max(1)
}

/// Run K samples and collect their snapshots. Single-threaded fallback when
/// only one core is available; otherwise spawns a fixed thread pool via
/// `std::thread::scope` (same approach as `fidelity_pipeline.rs::run_k_samples`).
fn run_k_samples(
    params: &EathParams,
    base_seed: u64,
    k: usize,
    parallelism: ParallelismMode,
    adapter: AdapterChoice,
    metric_params: &serde_json::Value,
) -> Result<Vec<MetricSnapshot>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let threads = resolve_thread_count(parallelism, k);
    if threads == 1 {
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            out.push(generate_one_sample(
                params,
                base_seed,
                i,
                adapter,
                metric_params,
            )?);
        }
        return Ok(out);
    }

    let mut results: Vec<Option<Result<MetricSnapshot>>> = (0..k).map(|_| None).collect();
    std::thread::scope(|scope| {
        let chunk_size = k.div_ceil(threads);
        let mut handles = Vec::with_capacity(threads);
        for chunk_start in (0..k).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(k);
            let params_ref = params;
            let metric_params_ref = metric_params;
            let handle = scope.spawn(move || {
                let mut local: Vec<(usize, Result<MetricSnapshot>)> =
                    Vec::with_capacity(chunk_end - chunk_start);
                for i in chunk_start..chunk_end {
                    local.push((
                        i,
                        generate_one_sample(
                            params_ref,
                            base_seed,
                            i,
                            adapter,
                            metric_params_ref,
                        ),
                    ));
                }
                local
            });
            handles.push(handle);
        }
        for handle in handles {
            if let Ok(local) = handle.join() {
                for (idx, res) in local {
                    results[idx] = Some(res);
                }
            }
        }
    });

    let mut out = Vec::with_capacity(k);
    for (i, slot) in results.into_iter().enumerate() {
        match slot {
            Some(Ok(s)) => out.push(s),
            Some(Err(e)) => return Err(e),
            None => {
                return Err(TensaError::SynthFailure(format!(
                    "significance: sample {i} thread panicked"
                )));
            }
        }
    }
    Ok(out)
}

// Distribution alignment + per-element statistics live in `super::stats` to
// keep this file under the 500-line cap. The pipeline calls into them via
// `super::stats::build_distribution`.
