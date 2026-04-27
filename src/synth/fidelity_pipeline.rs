//! Internal pipeline for [`super::fidelity`]: source-narrative stat
//! extraction, K-sample generation, K-sample collapse, and the per-metric
//! comparison body.
//!
//! Kept separate from `fidelity.rs` (public surface — types, entry point,
//! persistence, renderer) so each file stays under the 500-line cap. Every
//! item here is `pub(super)` — only the fidelity entry point calls in.
//!
//! Determinism contract for K-sample generation:
//! - Per-sample seed = `base_seed XOR (sample_idx * SAMPLE_SEED_MIX)`.
//! - Single- and multi-threaded execution paths walk the same sample
//!   indices, so the gathered `Vec<NarrativeStats>` is identical across
//!   threading modes.

use std::collections::HashMap;
use std::sync::Arc;

use uuid::Uuid;

use crate::analysis::graph_projection::collect_participation_index;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::types::Situation;

use super::calibrate_fitters::sort_situations_chronologically;
use super::eath::EathSurrogate;
use super::fidelity::{
    metric_passed, FidelityMetric, FidelityThresholds, ParallelismMode, DEFAULT_GEN_STEPS,
    SAMPLE_SEED_MIX,
};
use super::fidelity_metrics::{burstiness, ks_two_sample, lag1_autocorr, mae, spearman_rho};
use super::surrogate::SurrogateModel;
use super::types::{EathParams, SurrogateParams};

// ── Per-narrative stats ─────────────────────────────────────────────────────

/// Per-narrative aggregate stats consumed by the fidelity comparison. Same
/// shape for source AND synthetic samples — that's what makes the metric
/// computations one-liners.
#[derive(Debug, Clone, Default)]
pub(super) struct NarrativeStats {
    /// Per-entity participation count (≡ hyperdegree by construction).
    pub(super) activity_per_entity: Vec<f64>,
    /// Per-entity mean group size when participating (order-propensity proxy).
    pub(super) propensity_per_entity: Vec<f64>,
    /// Per-entity inter-event times (in tick units). Flattened across all
    /// entities for the inter_event_ks metric; per-entity vectors retained
    /// in `inter_event_per_entity` for burstiness.
    pub(super) inter_event_flat: Vec<f64>,
    /// Per-entity inter-event time vectors (one Vec per entity). Aligned
    /// positionally with `activity_per_entity`.
    pub(super) inter_event_per_entity: Vec<Vec<f64>>,
    /// Per-entity binary participation series (1 = active at tick t).
    pub(super) binary_series_per_entity: Vec<Vec<f64>>,
    /// Per-situation group sizes.
    pub(super) group_sizes: Vec<f64>,
}

/// Compute [`NarrativeStats`] from the sorted situation list + per-situation
/// participants. Shared by source extraction and per-K-sample extraction.
fn compute_stats(
    sorted_situations: &[&Situation],
    participants_by_sit: &HashMap<Uuid, Vec<usize>>,
    n_entities: usize,
) -> NarrativeStats {
    let num_situations = sorted_situations.len();
    let mut activity = vec![0.0_f64; n_entities];
    let mut propensity_sum = vec![0.0_f64; n_entities];
    let mut binary = vec![vec![0.0_f64; num_situations]; n_entities];
    let mut group_sizes = Vec::with_capacity(num_situations);

    for (t, sit) in sorted_situations.iter().enumerate() {
        let members = match participants_by_sit.get(&sit.id) {
            Some(v) => v,
            None => continue,
        };
        let group_size = members.len();
        if group_size >= 2 {
            group_sizes.push(group_size as f64);
        }
        for &idx in members {
            if idx < n_entities {
                activity[idx] += 1.0;
                propensity_sum[idx] += group_size as f64;
                binary[idx][t] = 1.0;
            }
        }
    }

    let propensity: Vec<f64> = activity
        .iter()
        .zip(propensity_sum.iter())
        .map(|(&count, &sum)| if count > 0.0 { sum / count } else { 0.0 })
        .collect();

    let mut inter_event_per_entity: Vec<Vec<f64>> = Vec::with_capacity(n_entities);
    let mut inter_event_flat: Vec<f64> = Vec::new();
    for series in &binary {
        let mut deltas = Vec::new();
        let mut last_active: Option<usize> = None;
        for (t, &v) in series.iter().enumerate() {
            if v > 0.0 {
                if let Some(prev) = last_active {
                    deltas.push((t - prev) as f64);
                }
                last_active = Some(t);
            }
        }
        inter_event_flat.extend_from_slice(&deltas);
        inter_event_per_entity.push(deltas);
    }

    NarrativeStats {
        activity_per_entity: activity,
        propensity_per_entity: propensity,
        inter_event_flat,
        inter_event_per_entity,
        binary_series_per_entity: binary,
        group_sizes,
    }
}

/// Walk a narrative once: list entities, list situations, sort
/// chronologically, build the participation index, run [`compute_stats`].
/// Used for both source extraction and per-K-sample synthetic extraction.
fn extract_stats_for_narrative(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    error_on_empty: bool,
) -> Result<NarrativeStats> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let n = entities.len();
    if n == 0 {
        if error_on_empty {
            return Err(TensaError::SynthFailure(format!(
                "fidelity: narrative '{narrative_id}' has 0 entities"
            )));
        }
        return Ok(NarrativeStats::default());
    }
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let entity_idx: HashMap<Uuid, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id, i))
        .collect();
    let order = sort_situations_chronologically(&situations);
    let sorted: Vec<&Situation> = order.iter().map(|&i| &situations[i]).collect();
    let situation_ids: Vec<Uuid> = sorted.iter().map(|s| s.id).collect();
    let participants_by_sit =
        collect_participation_index(hypergraph, &entity_idx, &situation_ids, None)?;
    Ok(compute_stats(&sorted, &participants_by_sit, n))
}

/// One-shot source-narrative extraction. Errors when the narrative has zero
/// entities — calibration would already have failed in that case, but we
/// double-check here for the case where the user runs fidelity against a
/// different narrative than they calibrated on.
pub(super) fn extract_source_stats(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<NarrativeStats> {
    extract_stats_for_narrative(hypergraph, narrative_id, true)
}

/// Per-sample synthetic stat extraction. Empty synthetic narrative is OK —
/// returns a zero-stats blob; metrics module degrades gracefully.
pub(super) fn extract_synthetic_stats(
    hypergraph: &Hypergraph,
    output_narrative_id: &str,
) -> Result<NarrativeStats> {
    extract_stats_for_narrative(hypergraph, output_narrative_id, false)
}

// ── K-sample generation ─────────────────────────────────────────────────────

#[inline]
fn per_sample_seed(base_seed: u64, sample_idx: usize) -> u64 {
    base_seed ^ (sample_idx as u64).wrapping_mul(SAMPLE_SEED_MIX)
}

/// Generate a single synthetic sample into an ephemeral [`MemoryStore`]
/// hypergraph and extract its stats. The hypergraph is dropped at function
/// exit; nothing persists.
fn generate_one_sample(
    params: &EathParams,
    base_seed: u64,
    sample_idx: usize,
) -> Result<NarrativeStats> {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let target = Hypergraph::new(store);
    let envelope = SurrogateParams {
        model: "eath".into(),
        params_json: serde_json::to_value(params)?,
        seed: per_sample_seed(base_seed, sample_idx),
        num_steps: DEFAULT_GEN_STEPS,
        label_prefix: "fidelity-sample".into(),
    };
    let summary = EathSurrogate.generate(&envelope, &target, "fidelity-out")?;
    extract_synthetic_stats(&target, &summary.output_narrative_id)
}

/// Resolve the parallelism mode into an actual thread count. `Auto` queries
/// `std::thread::available_parallelism`; on single-core / unknown returns 1.
fn resolve_thread_count(mode: ParallelismMode, k_samples: usize) -> usize {
    let raw = match mode {
        ParallelismMode::Single => 1,
        ParallelismMode::Threads(n) => n.max(1),
        ParallelismMode::Auto => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    };
    raw.min(k_samples).max(1)
}

/// Run K samples and collect their stats. Single-threaded when `Single` or
/// when `available_parallelism` reports 1 core; otherwise spawns a fixed
/// thread pool via `std::thread::scope`.
pub(super) fn run_k_samples(
    params: &EathParams,
    base_seed: u64,
    k: usize,
    parallelism: ParallelismMode,
) -> Result<Vec<NarrativeStats>> {
    let threads = resolve_thread_count(parallelism, k);
    if threads == 1 {
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            out.push(generate_one_sample(params, base_seed, i)?);
        }
        return Ok(out);
    }

    // Multi-threaded: split sample indices into `threads` chunks. Each chunk
    // runs sequentially in its own thread; results are gathered in original
    // order so the K=20 vec is identical to single-threaded output.
    let mut results: Vec<Option<Result<NarrativeStats>>> = (0..k).map(|_| None).collect();
    std::thread::scope(|scope| {
        let chunk_size = k.div_ceil(threads);
        let mut handles = Vec::with_capacity(threads);
        for chunk_start in (0..k).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(k);
            let params_ref = params;
            let handle = scope.spawn(move || {
                let mut local: Vec<(usize, Result<NarrativeStats>)> =
                    Vec::with_capacity(chunk_end - chunk_start);
                for i in chunk_start..chunk_end {
                    local.push((i, generate_one_sample(params_ref, base_seed, i)));
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
                    "fidelity: sample {i} thread panicked"
                )))
            }
        }
    }
    Ok(out)
}

// ── K-sample collapse + metric computation ─────────────────────────────────

/// Collapsed K-sample stats. Per-entity scalars are mean-aggregated; per-entity
/// vector members are concatenated across K (KS divergence and burstiness
/// MAE consume the larger sample naturally).
pub(super) struct CollapsedSynth {
    pub(super) activity_per_entity: Vec<f64>,
    pub(super) propensity_per_entity: Vec<f64>,
    pub(super) inter_event_flat: Vec<f64>,
    pub(super) inter_event_per_entity: Vec<Vec<f64>>,
    pub(super) binary_series_per_entity: Vec<Vec<f64>>,
    pub(super) group_sizes: Vec<f64>,
}

pub(super) fn collapse_k_samples(samples: &[NarrativeStats]) -> CollapsedSynth {
    if samples.is_empty() {
        return CollapsedSynth {
            activity_per_entity: Vec::new(),
            propensity_per_entity: Vec::new(),
            inter_event_flat: Vec::new(),
            inter_event_per_entity: Vec::new(),
            binary_series_per_entity: Vec::new(),
            group_sizes: Vec::new(),
        };
    }
    // Per-entity vectors might differ in length across samples (different K
    // generated different numbers of synthetic entities). Take the max
    // length and zero-pad shorter samples.
    let max_n = samples
        .iter()
        .map(|s| s.activity_per_entity.len())
        .max()
        .unwrap_or(0);

    let mut activity = vec![0.0_f64; max_n];
    let mut propensity = vec![0.0_f64; max_n];
    for s in samples {
        for i in 0..max_n {
            activity[i] += s.activity_per_entity.get(i).copied().unwrap_or(0.0);
            propensity[i] += s.propensity_per_entity.get(i).copied().unwrap_or(0.0);
        }
    }
    let k = samples.len() as f64;
    for v in &mut activity {
        *v /= k;
    }
    for v in &mut propensity {
        *v /= k;
    }

    // Per-entity inter-event lists: concatenate across K.
    let mut inter_per: Vec<Vec<f64>> = vec![Vec::new(); max_n];
    let mut binary_per: Vec<Vec<f64>> = vec![Vec::new(); max_n];
    let mut inter_flat = Vec::new();
    let mut group_sizes = Vec::new();
    for s in samples {
        for i in 0..max_n {
            if let Some(v) = s.inter_event_per_entity.get(i) {
                inter_per[i].extend_from_slice(v);
            }
            if let Some(v) = s.binary_series_per_entity.get(i) {
                binary_per[i].extend_from_slice(v);
            }
        }
        inter_flat.extend_from_slice(&s.inter_event_flat);
        group_sizes.extend_from_slice(&s.group_sizes);
    }

    CollapsedSynth {
        activity_per_entity: activity,
        propensity_per_entity: propensity,
        inter_event_flat: inter_flat,
        inter_event_per_entity: inter_per,
        binary_series_per_entity: binary_per,
        group_sizes,
    }
}

/// Compute the 7-metric vector. Each metric is one of: KS divergence on a
/// flat sample (lower is better), Spearman rank correlation on per-entity
/// vectors (higher is better), or MAE on per-entity scalar derivations
/// (lower is better).
pub(super) fn compute_metrics(
    source: &NarrativeStats,
    synth: &CollapsedSynth,
    thresholds: &FidelityThresholds,
) -> Vec<FidelityMetric> {
    let mut metrics = Vec::with_capacity(7);

    let inter_event = ks_two_sample(&source.inter_event_flat, &synth.inter_event_flat) as f32;
    metrics.push(make_metric(
        "inter_event_time_distribution",
        "ks_divergence",
        inter_event,
        thresholds.inter_event_ks,
    ));

    let group_size = ks_two_sample(&source.group_sizes, &synth.group_sizes) as f32;
    metrics.push(make_metric(
        "group_size_distribution",
        "ks_divergence",
        group_size,
        thresholds.group_size_ks,
    ));

    let activity = align_and_spearman(&source.activity_per_entity, &synth.activity_per_entity);
    metrics.push(make_metric(
        "activity_match",
        "spearman_rho",
        activity as f32,
        thresholds.activity_spearman,
    ));

    let propensity = align_and_spearman(
        &source.propensity_per_entity,
        &synth.propensity_per_entity,
    );
    metrics.push(make_metric(
        "order_propensity_match",
        "spearman_rho",
        propensity as f32,
        thresholds.order_propensity_spearman,
    ));

    let burstiness_value = paired_burstiness_mae(
        &source.inter_event_per_entity,
        &synth.inter_event_per_entity,
    ) as f32;
    metrics.push(make_metric(
        "burstiness_parity",
        "mae",
        burstiness_value,
        thresholds.burstiness_mae,
    ));

    let autocorr_value = paired_lag1_autocorr_mae(
        &source.binary_series_per_entity,
        &synth.binary_series_per_entity,
    ) as f32;
    metrics.push(make_metric(
        "memory_autocorrelation",
        "mae",
        autocorr_value,
        thresholds.memory_autocorr_mae,
    ));

    // Hyperdegree distribution — KS on raw per-entity activity counts.
    let hyperdegree =
        ks_two_sample(&source.activity_per_entity, &synth.activity_per_entity) as f32;
    metrics.push(make_metric(
        "hyperdegree_distribution",
        "ks_divergence",
        hyperdegree,
        thresholds.hyperdegree_ks,
    ));

    metrics
}

fn make_metric(name: &str, statistic: &str, value: f32, threshold: f32) -> FidelityMetric {
    FidelityMetric {
        name: name.to_string(),
        statistic: statistic.to_string(),
        value,
        threshold,
        passed: metric_passed(statistic, value, threshold),
    }
}

/// Length-aligned Spearman: pad the shorter vector with zeros so per-position
/// ranks compare cleanly. Both per-entity vectors come from independent
/// narratives that may have different N; padding with zeros (entities that
/// "don't exist") keeps the rank correlation interpretable.
fn align_and_spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut a_pad = a.to_vec();
    a_pad.resize(n, 0.0);
    let mut b_pad = b.to_vec();
    b_pad.resize(n, 0.0);
    spearman_rho(&a_pad, &b_pad)
}

/// Per-entity burstiness vector → MAE. Aligned by entity position; missing
/// rows on either side are treated as `B = 0` (Poisson default).
fn paired_burstiness_mae(source: &[Vec<f64>], synth: &[Vec<f64>]) -> f64 {
    let n = source.len().max(synth.len());
    if n == 0 {
        return 0.0;
    }
    let src: Vec<f64> = (0..n)
        .map(|i| source.get(i).map(|v| burstiness(v)).unwrap_or(0.0))
        .collect();
    let syn: Vec<f64> = (0..n)
        .map(|i| synth.get(i).map(|v| burstiness(v)).unwrap_or(0.0))
        .collect();
    mae(&src, &syn)
}

/// Per-entity lag-1 autocorr vector → MAE. Same alignment as burstiness.
fn paired_lag1_autocorr_mae(source: &[Vec<f64>], synth: &[Vec<f64>]) -> f64 {
    let n = source.len().max(synth.len());
    if n == 0 {
        return 0.0;
    }
    let src: Vec<f64> = (0..n)
        .map(|i| source.get(i).map(|v| lag1_autocorr(v)).unwrap_or(0.0))
        .collect();
    let syn: Vec<f64> = (0..n)
        .map(|i| synth.get(i).map(|v| lag1_autocorr(v)).unwrap_or(0.0))
        .collect();
    mae(&src, &syn)
}

/// Weighted aggregate score. Per-metric `passed` contributes 1.0 (full
/// weight) or 0.0 (no weight); unspecified-weight metrics default to 1.0.
///
/// Bit-identical to the pre-sprint numerics via the underlying weighted
/// fraction. See [`aggregate_score_with_tnorm`] for the fuzzy-wired
/// variant that honours a t-norm selection on the per-metric pass/fail
/// reduction (default Gödel reproduces this function exactly when all
/// weights are equal).
pub(super) fn aggregate_score(
    metrics: &[FidelityMetric],
    weights: &[(String, f32)],
) -> f32 {
    let weight_for = |name: &str| -> f32 {
        weights
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, w)| *w)
            .unwrap_or(1.0)
    };
    let mut numerator = 0.0_f32;
    let mut denominator = 0.0_f32;
    for m in metrics {
        let w = weight_for(&m.name);
        denominator += w;
        if m.passed {
            numerator += w;
        }
    }
    if denominator <= f32::EPSILON {
        return 0.0;
    }
    numerator / denominator
}

/// Variant of [`aggregate_score`] that folds per-metric pass/fail
/// indicators under a t-norm. Gödel corresponds to "all must pass"
/// (min), Goguen to "product of passes", and Łukasiewicz to a stricter
/// saturating AND. With equal weights and `TNormKind::Godel` the result
/// equals `1.0` iff every metric passes and `0.0` otherwise — matching
/// the classical overall pass/fail reduction used by
/// [`crate::synth::fidelity::OVERALL_PASS_SCORE`] thresholding when the
/// weighted-fraction path is unwanted.
///
/// Callers can still use [`aggregate_score`] for weighted fraction
/// scoring; this function is explicitly for the AND-style reduction.
pub fn aggregate_score_with_tnorm(
    metrics: &[FidelityMetric],
    tnorm: crate::fuzzy::tnorm::TNormKind,
) -> f32 {
    if metrics.is_empty() {
        return 0.0;
    }
    let xs: Vec<f64> = metrics
        .iter()
        .map(|m| if m.passed { 1.0 } else { 0.0 })
        .collect();
    crate::fuzzy::tnorm::reduce_tnorm(tnorm, &xs) as f32
}

/// Phase 2 full-aggregator variant of [`aggregate_score_with_tnorm`].
///
/// Phase 1 shipped a `TNormKind` selector for the pass/fail fold; Phase 2
/// widens the API to the full `AggregatorKind` family (Mean / Median /
/// OWA / Choquet / TNormReduce / TConormReduce). This lets callers pick
/// e.g. `AggregatorKind::Owa(linguistic_weights(Quantifier::Most, n))`
/// to "grade the fidelity like a reviewer would — tolerate a few
/// failures if most tests pass". Input metrics are reduced to a
/// 0.0 / 1.0 pass indicator; custom graded-pass indicators live in
/// downstream analytics.
///
/// Returns `0.0` on empty input (consistent with
/// [`aggregate_score_with_tnorm`]).
///
/// Cites: [yager1988owa] [grabisch1996choquet].
pub fn aggregate_metrics_with_aggregator(
    metrics: &[FidelityMetric],
    agg: &crate::fuzzy::aggregation::AggregatorKind,
) -> crate::error::Result<f32> {
    aggregate_metrics_with_aggregator_tracked(metrics, agg, None, None)
        .map(|(score, _, _)| score)
}

/// Provenance-tracking sibling of [`aggregate_metrics_with_aggregator`].
///
/// Returns `(score, measure_id, measure_version)`. The slot rules match
/// [`crate::fuzzy::aggregation_learn::resolve_measure_provenance`]:
/// caller-supplied IDs take priority, then `Choquet(measure)` identity,
/// otherwise `None`/`None`. Persistence callers stash the IDs into the
/// new `FidelityReport.fuzzy_measure_id` / `fuzzy_measure_version`
/// fields (serde-defaulted, so old blobs load unchanged).
///
/// Cites: [grabisch1996choquet], [bustince2016choquet].
pub fn aggregate_metrics_with_aggregator_tracked(
    metrics: &[FidelityMetric],
    agg: &crate::fuzzy::aggregation::AggregatorKind,
    measure_id: Option<String>,
    measure_version: Option<u32>,
) -> crate::error::Result<(f32, Option<String>, Option<u32>)> {
    if metrics.is_empty() {
        let (id_out, ver_out) =
            crate::fuzzy::aggregation_learn::resolve_measure_provenance(
                agg,
                measure_id,
                measure_version,
            );
        return Ok((0.0, id_out, ver_out));
    }
    let xs: Vec<f64> = metrics
        .iter()
        .map(|m| if m.passed { 1.0 } else { 0.0 })
        .collect();
    let aggregator = crate::fuzzy::aggregation::aggregator_for(agg.clone());
    let score = aggregator.aggregate(&xs)? as f32;
    let (id_out, ver_out) = crate::fuzzy::aggregation_learn::resolve_measure_provenance(
        agg,
        measure_id,
        measure_version,
    );
    Ok((score, id_out, ver_out))
}
