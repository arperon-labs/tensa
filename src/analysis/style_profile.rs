//! Multi-layer narrative style profile and fingerprint comparison.
//!
//! Computes structural style features from the hypergraph for a given narrative
//! across six layers: structural rhythm, character dynamics, information management,
//! causal architecture, temporal texture, and graph topology. Provides fingerprint
//! comparison, anomaly detection, and radar chart normalization.

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::load_sorted_situations;
use crate::analysis::stylometry::ProseStyleFeatures;
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::*;

// ─── Data Structures ────────────────────────────────────────

/// Full six-layer narrative style profile.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativeStyleProfile {
    pub narrative_id: String,
    pub computed_at: DateTime<Utc>,

    // Layer 1: Structural Rhythm
    pub situation_density_curve: Vec<f32>,
    pub avg_participants_per_situation: f32,
    pub participation_count_variance: f32,
    pub arc_type: Option<String>,
    pub arc_confidence: f32,
    pub situation_entity_ratio: f32,

    // Layer 2: Character Dynamics
    pub game_type_distribution: Vec<f32>,
    pub role_entropy: f32,
    pub power_asymmetry_gini: f32,
    pub co_participation_density: f32,
    pub protagonist_concentration: f32,

    // Layer 3: Information Management
    pub avg_info_r0: f32,
    pub deception_index: f32,
    pub knowledge_asymmetry_gini: f32,
    pub revelation_timing: f32,
    pub secret_survival_rate: f32,

    // Layer 4: Causal Architecture
    pub causal_density: f32,
    pub avg_causal_chain_length: f32,
    pub max_causal_chain_length: usize,
    pub unexplained_event_ratio: f32,
    pub causal_branching_factor: f32,
    pub causal_convergence_factor: f32,

    // Layer 5: Temporal Texture
    pub allen_relation_distribution: Vec<f32>,
    pub flashback_frequency: f32,
    pub temporal_span_variance: f32,
    pub temporal_gap_ratio: f32,

    // Layer 6: Graph Topology
    pub wl_hash_histogram: Vec<f32>,
    /// 256-bit SimHash of the full bag of Weisfeiler–Lehman refinement labels,
    /// suitable for Hamming-distance topological comparison. Defaults to all-zero
    /// for profiles written before this field existed (v0.28 addition).
    #[serde(default)]
    pub wl_simhash: [u64; 4],
    pub community_count: usize,
    pub avg_shortest_path: f32,
    pub graph_diameter: usize,
    pub edge_density: f32,

    // Narrative surprise: mean self-information over situation feature signatures,
    // normalized by log2(N) to land in [0, 1]. Defaults to 0 for profiles written
    // before this field existed.
    #[serde(default)]
    pub narrative_surprise: f32,

    // ── Layer 7: Generative Architecture (Sprint D9.8) ──────
    // These 8 axes capture the structural design patterns that distinguish
    // narrative architectures. All bounded [0, 1]. Default to 0 for
    // profiles computed before D9.
    /// Fulfilled / (fulfilled + abandoned) commitments. D9.1.
    #[serde(default)]
    pub promise_fulfillment_ratio: f32,
    /// Mean chapters between setup and payoff. Normalized to [0, 1]. D9.1.
    #[serde(default)]
    pub average_payoff_distance: f32,
    /// Normalized Kendall tau distance between fabula and sjužet. D9.2.
    #[serde(default)]
    pub fabula_sjuzet_divergence: f32,
    /// Fraction of situations with active dramatic irony. D9.3.
    #[serde(default)]
    pub dramatic_irony_density: f32,
    /// Number of unique focalizers normalized by total chapters. D9.3.
    #[serde(default)]
    pub focalization_diversity: f32,
    /// Average arc completeness across main characters. D9.4.
    #[serde(default)]
    pub character_arc_completeness: f32,
    /// Fraction of subplots that converge into the main climax. D9.4.
    #[serde(default)]
    pub subplot_convergence_ratio: f32,
    /// Composite pacing: scene-sequel rhythm + narration mode + density. D9.4.
    #[serde(default)]
    pub scene_sequel_rhythm_score: f32,
}

/// Combined fingerprint: prose style + structural profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeFingerprint {
    pub narrative_id: String,
    pub computed_at: DateTime<Utc>,
    pub prose: ProseStyleFeatures,
    pub structure: NarrativeStyleProfile,
}

/// Per-layer similarity scores between two narratives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleSimilarity {
    pub overall: f32,
    pub prose_similarity: f32,
    pub rhythm_similarity: f32,
    pub character_dynamics_similarity: f32,
    pub information_flow_similarity: f32,
    pub causal_similarity: f32,
    pub temporal_similarity: f32,
    pub topology_similarity: f32,
    /// Per-metric breakdown of each layer's component (JS / Mahalanobis / Burrows-Cosine /
    /// Hamming). Populated by weighted similarity computation. Empty for legacy callers
    /// of `fingerprint_similarity` — `#[serde(default)]` keeps JSON compatibility.
    #[serde(default)]
    pub layer_details: Vec<LayerDetail>,
}

/// One component of the weighted similarity breakdown (layer + metric + value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDetail {
    pub layer: String,
    pub metric: String,
    pub value: f32,
}

/// Tunable weights and corpus statistics for weighted fingerprint comparison.
///
/// Each `w_*` field scales the contribution of one of the 8 atomic layers in the
/// weighted overall similarity. Weights are renormalized internally so they do
/// not need to sum to 1.
///
/// `corpus_stats` provides the function-word means and standard deviations needed
/// by Burrows-Cosine; `scalar_stds` provides the diagonal variance vector used by
/// Mahalanobis similarity on scalar feature concatenations. When absent, the
/// affected kernel falls back to raw cosine / unit-variance Euclidean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedSimilarityConfig {
    pub w_prose_fw: f32,
    pub w_prose_scalars: f32,
    pub w_rhythm: f32,
    pub w_character: f32,
    pub w_info: f32,
    pub w_causal: f32,
    pub w_temporal: f32,
    pub w_topology: f32,
    /// Weight for generative architecture layer (D9.8). Defaults to 0.0
    /// to maintain backward compatibility with pre-D9 weight configs.
    #[serde(default)]
    pub w_generative: f32,
    #[serde(default)]
    pub corpus_stats: Option<crate::analysis::stylometry::CorpusStats>,
    #[serde(default)]
    pub scalar_stds: Option<ScalarStds>,
}

/// Diagonal standard deviations for scalar feature vectors, grouped by layer.
/// All fields default to empty vectors — kernels substitute unit variance when
/// a layer's stds are missing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScalarStds {
    #[serde(default)]
    pub prose_scalars: Vec<f32>,
    #[serde(default)]
    pub rhythm_scalars: Vec<f32>,
    #[serde(default)]
    pub character_scalars: Vec<f32>,
    #[serde(default)]
    pub info_scalars: Vec<f32>,
    #[serde(default)]
    pub causal_scalars: Vec<f32>,
    #[serde(default)]
    pub temporal_scalars: Vec<f32>,
    #[serde(default)]
    pub topology_scalars: Vec<f32>,
    /// D9.8: generative architecture layer standard deviations (8 axes).
    #[serde(default)]
    pub generative_scalars: Vec<f32>,
}

impl Default for WeightedSimilarityConfig {
    /// Uniform weights (1.0 across all 8 atomic layers) with no corpus statistics —
    /// the neutral baseline that reproduces legacy averaging behavior, absent any
    /// PAN-learned tuning.
    fn default() -> Self {
        Self {
            w_prose_fw: 1.0,
            w_prose_scalars: 1.0,
            w_rhythm: 1.0,
            w_character: 1.0,
            w_info: 1.0,
            w_causal: 1.0,
            w_temporal: 1.0,
            w_topology: 1.0,
            w_generative: 0.0, // Off by default for backward compatibility
            corpus_stats: None,
            scalar_stds: None,
        }
    }
}

impl WeightedSimilarityConfig {
    /// Default-but-corpus-aware config. Requires passing corpus stats computed
    /// from the text being compared (or the full reference corpus).
    pub fn with_corpus(corpus_stats: crate::analysis::stylometry::CorpusStats) -> Self {
        Self {
            corpus_stats: Some(corpus_stats),
            ..Self::default()
        }
    }

    /// Load the user-tuned config persisted by `PUT /settings/style-weights`,
    /// falling back to `Default` when the key is absent or unreadable.
    ///
    /// Callers that consume similarity (anomaly detection, pair comparison)
    /// should route through this so weights uploaded via `train_pan_weights`
    /// actually take effect.
    /// Weights optimized for generation fitness evaluation (D9.8).
    /// High weight on structural/generative axes, lower on stylometric axes
    /// (which LoRA/embedding handles).
    pub fn generative_weights() -> Self {
        Self {
            w_prose_fw: 0.3,
            w_prose_scalars: 0.3,
            w_rhythm: 0.8,
            w_character: 0.8,
            w_info: 0.7,
            w_causal: 0.8,
            w_temporal: 0.7,
            w_topology: 0.5,
            w_generative: 2.0, // High weight on generative architecture axes
            corpus_stats: None,
            scalar_stds: None,
        }
    }

    /// Load the user-tuned config persisted by `PUT /settings/style-weights`,
    /// falling back to `Default` when the key is absent or unreadable.
    ///
    /// Callers that consume similarity (anomaly detection, pair comparison)
    /// should route through this so weights uploaded via `train_pan_weights`
    /// actually take effect.
    pub fn load_or_default(store: &dyn crate::store::KVStore) -> Self {
        let key = crate::analysis::analysis_key(
            crate::hypergraph::keys::ANALYSIS_STYLE_WEIGHTS,
            &["default"],
        );
        match store.get(&key) {
            Ok(Some(bytes)) => serde_json::from_slice(&bytes).unwrap_or_default(),
            _ => Self::default(),
        }
    }
}

/// A detected style anomaly in a narrative chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAnomaly {
    pub chunk_index: usize,
    pub similarity_to_global: f32,
    pub most_deviant_features: Vec<(String, f32)>,
}

/// Which profile layers have meaningful data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCompleteness {
    pub prose: bool,
    pub rhythm: bool,
    pub character_dynamics: bool,
    pub information_flow: bool,
    pub causal: bool,
    pub temporal: bool,
    pub topology: bool,
}

/// Radar chart axes, all normalized to [0, 1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintRadar {
    pub pacing: f32,
    pub ensemble: f32,
    pub causal_density: f32,
    pub info_r0: f32,
    pub deception: f32,
    pub temporal_complexity: f32,
    pub strategic_variety: f32,
    pub power_asymmetry: f32,
    pub protagonist_focus: f32,
    pub late_revelation: f32,
    pub subplot_richness: f32,
    pub surprise: f32,
}

// ─── Helpers ────────────────────────────────────────────────

fn shannon_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}

fn gini(values: &[f64]) -> f32 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = sorted.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut numerator = 0.0_f64;
    for (i, &v) in sorted.iter().enumerate() {
        numerator += (2.0 * (i as f64) + 1.0 - n as f64) * v;
    }
    (numerator / (n as f64 * total)) as f32
}

/// Cosine similarity between two f32 vectors.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        // Both vectors zero → identical (both have no data) → 1.0
        // One zero, one not → completely dissimilar → 0.0
        return if norm_a < 1e-12 && norm_b < 1e-12 {
            1.0
        } else {
            0.0
        };
    }
    (dot / denom).clamp(0.0, 1.0)
}

fn scalar_sim(a: f32, b: f32) -> f32 {
    let denom = a.abs().max(b.abs()).max(1.0);
    (1.0 - (a - b).abs() / denom).clamp(0.0, 1.0)
}

fn normalize(counts: &[f32], bins: usize) -> Vec<f32> {
    let total: f32 = counts.iter().sum();
    if total > 0.0 {
        counts.iter().map(|&c| c / total).collect()
    } else {
        // Return zeros instead of fake uniform — downstream computes honest 0%
        vec![0.0; bins]
    }
}

fn avg_of(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f32>() / vals.len() as f32
    }
}

// ─── Narrative Surprise ─────────────────────────────────────

/// Discretize a count into a small bucket (shared shape with `entropy.rs`).
fn surprise_bucket(count: usize) -> usize {
    match count {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=6 => 3,
        _ => 4,
    }
}

/// Compute mean self-information across all situations based on coarse
/// feature signatures (participant count, roles, narrative level, game
/// presence, causal in/out degree). Normalizes by log2(N) so the result
/// lands in [0, 1] regardless of corpus size.
fn compute_narrative_surprise(
    situations: &[Situation],
    all_parts: &[Vec<Participation>],
    hypergraph: &Hypergraph,
) -> f32 {
    let n = situations.len();
    if n <= 1 {
        return 0.0;
    }

    let sit_ids: HashSet<Uuid> = situations.iter().map(|s| s.id).collect();
    let mut in_deg: HashMap<Uuid, usize> = HashMap::new();
    let mut out_deg: HashMap<Uuid, usize> = HashMap::new();
    for sit in situations {
        if let Ok(antecedents) = hypergraph.get_antecedents(&sit.id) {
            for link in antecedents {
                if sit_ids.contains(&link.from_situation) {
                    *in_deg.entry(sit.id).or_insert(0) += 1;
                    *out_deg.entry(link.from_situation).or_insert(0) += 1;
                }
            }
        }
    }

    let mut signatures: Vec<String> = Vec::with_capacity(n);
    for (i, sit) in situations.iter().enumerate() {
        let parts = all_parts.get(i).map(|p| p.as_slice()).unwrap_or(&[]);
        let pc = parts.len();
        let rc: HashSet<String> = parts.iter().map(|p| format!("{:?}", p.role)).collect();
        let sig = format!(
            "pc{}:rc{}:nl{}:gs{}:ci{}:co{}",
            surprise_bucket(pc),
            surprise_bucket(rc.len()),
            sit.narrative_level.ordinal(),
            sit.game_structure.is_some() as u8,
            surprise_bucket(in_deg.get(&sit.id).copied().unwrap_or(0)),
            surprise_bucket(out_deg.get(&sit.id).copied().unwrap_or(0)),
        );
        signatures.push(sig);
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in &signatures {
        *counts.entry(s.as_str()).or_insert(0) += 1;
    }

    let total = n as f32;
    let mut sum = 0.0_f32;
    for s in &signatures {
        let p = counts[s.as_str()] as f32 / total;
        if p > 0.0 {
            sum += -p.log2();
        }
    }
    let mean = sum / total;
    let max = (n as f32).log2();
    if max > 0.0 {
        (mean / max).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ─── Layer 1: Structural Rhythm ─────────────────────────────

fn compute_situation_density(situations: &[Situation]) -> Vec<f32> {
    const BINS: usize = 20;
    let n = situations.len();
    if n == 0 {
        return vec![1.0 / BINS as f32; BINS];
    }
    let valid_starts: Vec<i64> = situations
        .iter()
        .filter_map(|s| s.temporal.start.map(|t| t.timestamp_millis()))
        .collect();
    if valid_starts.len() >= 2 {
        let min_t = *valid_starts.iter().min().unwrap_or(&0);
        let max_t = *valid_starts.iter().max().unwrap_or(&1);
        let span = (max_t - min_t).max(1) as f64;
        let mut bins = vec![0.0_f32; BINS];
        for &t in &valid_starts {
            let idx = (((t - min_t) as f64 / span) * BINS as f64) as usize;
            bins[idx.min(BINS - 1)] += 1.0;
        }
        return normalize(&bins, BINS);
    }
    let mut bins = vec![0.0_f32; BINS];
    for i in 0..n {
        let frac = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.5
        };
        bins[((frac * BINS as f64) as usize).min(BINS - 1)] += 1.0;
    }
    normalize(&bins, BINS)
}

fn compute_participation_stats(
    situations: &[Situation],
    hypergraph: &Hypergraph,
) -> Result<(f32, f32, Vec<Vec<Participation>>)> {
    let mut counts = Vec::with_capacity(situations.len());
    let mut all_parts = Vec::with_capacity(situations.len());
    for sit in situations {
        let parts = hypergraph.get_participants_for_situation(&sit.id)?;
        counts.push(parts.len() as f32);
        all_parts.push(parts);
    }
    if counts.is_empty() {
        return Ok((0.0, 0.0, all_parts));
    }
    let avg = counts.iter().sum::<f32>() / counts.len() as f32;
    let var = if counts.len() > 1 {
        counts.iter().map(|&c| (c - avg).powi(2)).sum::<f32>() / counts.len() as f32
    } else {
        0.0
    };
    Ok((avg, var, all_parts))
}

// ─── Layer 2: Character Dynamics ────────────────────────────

const NUM_GAME_TYPES: usize = 7;

fn game_type_index(gc: &GameClassification) -> Option<usize> {
    match gc {
        GameClassification::PrisonersDilemma => Some(0),
        GameClassification::Coordination => Some(1),
        GameClassification::Signaling => Some(2),
        GameClassification::Auction => Some(3),
        GameClassification::Bargaining => Some(4),
        GameClassification::ZeroSum => Some(5),
        GameClassification::AsymmetricInformation => Some(6),
        GameClassification::Custom(_) => None,
    }
}

fn compute_game_type_dist(situations: &[Situation]) -> Vec<f32> {
    let mut counts = vec![0.0_f32; NUM_GAME_TYPES];
    for sit in situations {
        if let Some(gs) = &sit.game_structure {
            if let Some(idx) = game_type_index(&gs.game_type) {
                counts[idx] += 1.0;
            }
        }
    }
    normalize(&counts, NUM_GAME_TYPES)
}

fn compute_role_entropy(all_parts: &[Vec<Participation>]) -> f32 {
    let mut role_counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    for parts in all_parts {
        for p in parts {
            *role_counts.entry(format!("{:?}", p.role)).or_insert(0) += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let probs: Vec<f32> = role_counts
        .values()
        .map(|&c| c as f32 / total as f32)
        .collect();
    shannon_entropy(&probs)
}

fn compute_power_asymmetry(all_parts: &[Vec<Participation>]) -> f32 {
    let payoffs: Vec<f64> = all_parts
        .iter()
        .flatten()
        .filter_map(|p| p.payoff.as_ref().and_then(|v| v.as_f64()))
        .collect();
    if payoffs.is_empty() {
        0.0
    } else {
        gini(&payoffs)
    }
}

fn compute_co_participation_density(entity_ids: &[Uuid], all_parts: &[Vec<Participation>]) -> f32 {
    let n = entity_ids.len();
    if n < 2 {
        return 0.0;
    }
    let id_map: HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut shared_pairs: HashSet<(usize, usize)> = HashSet::new();
    for parts in all_parts {
        let v: Vec<usize> = parts
            .iter()
            .filter_map(|p| id_map.get(&p.entity_id).copied())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                let (lo, hi) = if v[i] < v[j] {
                    (v[i], v[j])
                } else {
                    (v[j], v[i])
                };
                shared_pairs.insert((lo, hi));
            }
        }
    }
    shared_pairs.len() as f32 / (n * (n - 1) / 2) as f32
}

fn compute_protagonist_concentration(entity_ids: &[Uuid], all_parts: &[Vec<Participation>]) -> f32 {
    let mut counts: HashMap<Uuid, f64> = HashMap::new();
    for parts in all_parts {
        for p in parts {
            *counts.entry(p.entity_id).or_insert(0.0) += 1.0;
        }
    }
    let vals: Vec<f64> = entity_ids
        .iter()
        .map(|id| counts.get(id).copied().unwrap_or(0.0))
        .collect();
    gini(&vals)
}

// ─── Layer 3: Information Management ────────────────────────

fn compute_info_layer(
    situations: &[Situation],
    all_parts: &[Vec<Participation>],
) -> (f32, f32, f32, f32, f32) {
    let mut all_reveals: Vec<(usize, String)> = Vec::new();
    let mut all_learns: Vec<(usize, String, Uuid)> = Vec::new();
    let mut entity_knowledge: HashMap<Uuid, HashSet<String>> = HashMap::new();
    let mut total_facts: HashSet<String> = HashSet::new();
    let n_sit = situations.len();

    for (i, parts) in all_parts.iter().enumerate() {
        for p in parts {
            if let Some(ref info) = p.info_set {
                for fact in &info.knows_before {
                    let key = format!("{}:{}", fact.about_entity, fact.fact);
                    entity_knowledge
                        .entry(p.entity_id)
                        .or_default()
                        .insert(key.clone());
                    total_facts.insert(key);
                }
                for fact in &info.learns {
                    let key = format!("{}:{}", fact.about_entity, fact.fact);
                    entity_knowledge
                        .entry(p.entity_id)
                        .or_default()
                        .insert(key.clone());
                    all_learns.push((i, key.clone(), p.entity_id));
                    total_facts.insert(key);
                }
                for fact in &info.reveals {
                    let key = format!("{}:{}", fact.about_entity, fact.fact);
                    all_reveals.push((i, key.clone()));
                    total_facts.insert(key);
                }
            }
        }
    }

    let avg_info_r0 = if all_reveals.is_empty() {
        0.0
    } else {
        // Group learns by fact key for O(reveals * learns_per_fact) instead of O(reveals * learns)
        let mut learns_by_fact: HashMap<&str, Vec<(usize, Uuid)>> = HashMap::new();
        for (li, lk, eid) in &all_learns {
            learns_by_fact
                .entry(lk.as_str())
                .or_default()
                .push((*li, *eid));
        }
        let mut spread = 0.0_f32;
        for (ri, rk) in &all_reveals {
            let learners = learns_by_fact
                .get(rk.as_str())
                .map(|ls| {
                    ls.iter()
                        .filter(|(li, _)| li > ri)
                        .map(|(_, e)| e)
                        .collect::<HashSet<_>>()
                        .len()
                })
                .unwrap_or(0);
            spread += learners as f32;
        }
        spread / all_reveals.len() as f32
    };

    let mut deceptive = 0usize;
    let mut eligible = 0usize;
    for parts in all_parts {
        for p in parts {
            if matches!(p.role, Role::Informant | Role::Confidant) {
                eligible += 1;
                let has_reveals = p
                    .info_set
                    .as_ref()
                    .map(|i| !i.reveals.is_empty())
                    .unwrap_or(false);
                if !has_reveals {
                    deceptive += 1;
                }
            }
        }
    }
    let deception_index = if eligible > 0 {
        deceptive as f32 / eligible as f32
    } else {
        0.0
    };

    let mid = n_sit / 2;
    let mut mid_k: HashMap<Uuid, usize> = HashMap::new();
    for (i, parts) in all_parts.iter().enumerate() {
        if i > mid {
            break;
        }
        for p in parts {
            if let Some(ref info) = p.info_set {
                *mid_k.entry(p.entity_id).or_insert(0) +=
                    info.knows_before.len() + info.learns.len();
            }
        }
    }
    let kv: Vec<f64> = mid_k.values().map(|&c| c as f64).collect();
    let knowledge_asymmetry_gini = gini(&kv);

    let revelation_timing = if all_reveals.is_empty() || n_sit <= 1 {
        0.5
    } else {
        all_reveals
            .iter()
            .map(|(i, _)| *i as f32 / (n_sit - 1) as f32)
            .sum::<f32>()
            / all_reveals.len() as f32
    };

    let secret_survival_rate = if total_facts.is_empty() {
        0.0
    } else {
        // Count how many entities know each fact, then filter for secrets (known by exactly 1)
        let mut fact_knower_count: HashMap<&str, usize> = HashMap::new();
        for ks in entity_knowledge.values() {
            for fk in ks {
                *fact_knower_count.entry(fk.as_str()).or_insert(0) += 1;
            }
        }
        let single = fact_knower_count.values().filter(|&&c| c == 1).count();
        single as f32 / total_facts.len() as f32
    };

    (
        avg_info_r0,
        deception_index,
        knowledge_asymmetry_gini,
        revelation_timing,
        secret_survival_rate,
    )
}

// ─── Layer 4: Causal Architecture ───────────────────────────

fn compute_causal_layer(
    situations: &[Situation],
    hypergraph: &Hypergraph,
) -> Result<(f32, f32, usize, f32, f32, f32)> {
    let n = situations.len();
    if n == 0 {
        return Ok((0.0, 0.0, 0, 0.0, 0.0, 0.0));
    }

    let sit_ids: HashSet<Uuid> = situations.iter().map(|s| s.id).collect();
    let mut out_edges: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    let mut in_deg: HashMap<Uuid, usize> = HashMap::new();
    let mut out_deg: HashMap<Uuid, usize> = HashMap::new();
    let mut total_links = 0usize;

    for sit in situations {
        let cons = hypergraph.get_consequences(&sit.id)?;
        for link in &cons {
            if sit_ids.contains(&link.to_situation) {
                out_edges.entry(sit.id).or_default().push(link.to_situation);
                *in_deg.entry(link.to_situation).or_insert(0) += 1;
                *out_deg.entry(sit.id).or_insert(0) += 1;
                total_links += 1;
            }
        }
    }

    let causal_density = total_links as f32 / n as f32;

    let roots: Vec<Uuid> = sit_ids
        .iter()
        .filter(|id| in_deg.get(id).copied().unwrap_or(0) == 0)
        .copied()
        .collect();

    let mut max_depth = 0usize;
    let mut depth_sum = 0usize;
    let mut depth_count = 0usize;
    for root in &roots {
        let mut queue: VecDeque<(Uuid, usize)> = VecDeque::new();
        let mut visited: HashSet<Uuid> = HashSet::new();
        queue.push_back((*root, 0));
        visited.insert(*root);
        while let Some((node, depth)) = queue.pop_front() {
            if depth > max_depth {
                max_depth = depth;
            }
            depth_sum += depth;
            depth_count += 1;
            if let Some(children) = out_edges.get(&node) {
                for &child in children {
                    if visited.insert(child) {
                        queue.push_back((child, depth + 1));
                    }
                }
            }
        }
    }

    let avg_chain = if depth_count > 0 {
        depth_sum as f32 / depth_count as f32
    } else {
        0.0
    };

    let unexplained = if n <= 1 {
        0.0
    } else {
        let root_count = roots.len().saturating_sub(1);
        root_count as f32 / (n - 1) as f32
    };

    let branching = if n > 0 {
        out_deg.values().sum::<usize>() as f32 / n as f32
    } else {
        0.0
    };
    let convergence = if n > 0 {
        in_deg.values().sum::<usize>() as f32 / n as f32
    } else {
        0.0
    };

    Ok((
        causal_density,
        avg_chain,
        max_depth,
        unexplained,
        branching,
        convergence,
    ))
}

// ─── Layer 5: Temporal Texture ──────────────────────────────

const NUM_ALLEN_RELATIONS: usize = 13;

fn allen_relation_index(rel: &AllenRelation) -> usize {
    match rel {
        AllenRelation::Before => 0,
        AllenRelation::After => 1,
        AllenRelation::Meets => 2,
        AllenRelation::MetBy => 3,
        AllenRelation::Overlaps => 4,
        AllenRelation::OverlappedBy => 5,
        AllenRelation::During => 6,
        AllenRelation::Contains => 7,
        AllenRelation::Starts => 8,
        AllenRelation::StartedBy => 9,
        AllenRelation::Finishes => 10,
        AllenRelation::FinishedBy => 11,
        AllenRelation::Equals => 12,
    }
}

fn compute_temporal_layer(situations: &[Situation]) -> (Vec<f32>, f32, f32, f32) {
    let mut rel_counts = vec![0.0_f32; NUM_ALLEN_RELATIONS];
    for sit in situations {
        for rt in &sit.temporal.relations {
            rel_counts[allen_relation_index(&rt.relation)] += 1.0;
        }
    }
    let allen_dist = normalize(&rel_counts, NUM_ALLEN_RELATIONS);

    let mut flashback_count = 0usize;
    for sit in situations {
        if let Some(ref disc) = sit.discourse {
            if let Some(ref order) = disc.order {
                if order.to_lowercase().contains("analepsis") {
                    flashback_count += 1;
                }
            }
        }
    }
    let flashback_freq = if situations.is_empty() {
        0.0
    } else {
        flashback_count as f32 / situations.len() as f32
    };

    let mut durations = Vec::new();
    for sit in situations {
        if let (Some(s), Some(e)) = (sit.temporal.start, sit.temporal.end) {
            durations.push((e - s).num_milliseconds() as f64);
        }
    }
    let span_var = if durations.len() > 1 {
        let mean = durations.iter().sum::<f64>() / durations.len() as f64;
        (durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64).sqrt()
            as f32
    } else {
        0.0
    };

    let gap_ratio = compute_temporal_gap_ratio(situations);
    (allen_dist, flashback_freq, span_var, gap_ratio)
}

fn compute_temporal_gap_ratio(situations: &[Situation]) -> f32 {
    let mut intervals: Vec<(i64, i64)> = situations
        .iter()
        .filter_map(|s| {
            let s_ms = s.temporal.start?.timestamp_millis();
            let e_ms = s.temporal.end?.timestamp_millis();
            if e_ms > s_ms {
                Some((s_ms, e_ms))
            } else {
                None
            }
        })
        .collect();
    if intervals.is_empty() {
        return 0.0;
    }
    intervals.sort_by_key(|&(s, _)| s);
    let g_start = intervals[0].0;
    let g_end = intervals.iter().map(|&(_, e)| e).max().unwrap_or(g_start);
    let total_span = (g_end - g_start) as f64;
    if total_span <= 0.0 {
        return 0.0;
    }
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for &(s, e) in &intervals {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }
    let covered: f64 = merged.iter().map(|&(s, e)| (e - s) as f64).sum();
    (1.0 - covered / total_span).max(0.0) as f32
}

// ─── Layer 6: Graph Topology ────────────────────────────────

struct CoParticipationGraph {
    n: usize,
    adj: Vec<HashSet<usize>>,
}

fn build_co_participation_graph(
    entity_ids: &[Uuid],
    all_parts: &[Vec<Participation>],
    min_shared: usize,
) -> CoParticipationGraph {
    let n = entity_ids.len();
    let id_map: HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut pair_count: HashMap<(usize, usize), usize> = HashMap::new();
    for parts in all_parts {
        let indices: Vec<usize> = parts
            .iter()
            .filter_map(|p| id_map.get(&p.entity_id).copied())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let (lo, hi) = if indices[i] < indices[j] {
                    (indices[i], indices[j])
                } else {
                    (indices[j], indices[i])
                };
                *pair_count.entry((lo, hi)).or_insert(0) += 1;
            }
        }
    }
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (&(a, b), &count) in &pair_count {
        if count >= min_shared {
            adj[a].insert(b);
            adj[b].insert(a);
        }
    }
    CoParticipationGraph { n, adj }
}

/// Signature of the Weisfeiler–Lehman refinement over a graph:
/// a top-K histogram of label frequencies plus a 256-bit SimHash of the full bag.
pub(crate) struct WlSignature {
    pub histogram: Vec<f32>,
    pub simhash: [u64; 4],
}

/// Splat a 64-bit WL label into 256 independently-hashed bits using four
/// salted mixing constants. Deterministic, no external RNG required.
fn splat_wl_label_to_256(label: u64) -> [u64; 4] {
    // Four salts chosen to be large primes with well-distributed bits.
    // Each produces one 64-bit slice of the 256-bit expansion.
    const SALTS: [u64; 4] = [
        0x9E3779B97F4A7C15, // golden ratio
        0xBF58476D1CE4E5B9, // splitmix64 mix 1
        0x94D049BB133111EB, // splitmix64 mix 2
        0xC6BC279692B5C323, // arbitrary large prime
    ];
    let mut out = [0u64; 4];
    for (i, &salt) in SALTS.iter().enumerate() {
        // splitmix64-style finalization on label XOR salt
        let mut z = label.wrapping_add(salt);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        out[i] = z;
    }
    out
}

fn compute_wl_signature(graph: &CoParticipationGraph) -> WlSignature {
    const TOP_K: usize = 50;
    let n = graph.n;
    if n == 0 {
        return WlSignature {
            histogram: vec![0.0; TOP_K],
            simhash: [0u64; 4],
        };
    }

    let mut labels: Vec<u64> = graph.adj.iter().map(|a| a.len() as u64).collect();
    let mut all_labels: HashMap<u64, usize> = HashMap::new();
    for &l in &labels {
        *all_labels.entry(l).or_insert(0) += 1;
    }

    for _ in 0..3 {
        let mut new_labels = Vec::with_capacity(n);
        for i in 0..n {
            let mut nls: Vec<u64> = graph.adj[i].iter().map(|&j| labels[j]).collect();
            nls.sort();
            let mut hash = labels[i].wrapping_mul(17);
            for nl in &nls {
                hash = hash.wrapping_mul(31).wrapping_add(*nl);
            }
            new_labels.push(hash);
            *all_labels.entry(hash).or_insert(0) += 1;
        }
        labels = new_labels;
    }

    // --- SimHash over the full bag-of-labels (before truncation) ---
    // For each of 256 bits, accumulate +count when the bit is 1 in the
    // splatted representation of the label, -count when 0. Collapse to 1
    // iff the accumulator is positive. Identical bags → identical SimHash;
    // perturbations flip proportionally few bits → Hamming ≈ structural distance.
    let mut accum = [0i64; 256];
    for (&lbl, &count) in &all_labels {
        let bits = splat_wl_label_to_256(lbl);
        let c = count as i64;
        for bit_idx in 0..256 {
            let word = bits[bit_idx >> 6];
            let mask = 1u64 << (bit_idx & 63);
            if word & mask != 0 {
                accum[bit_idx] += c;
            } else {
                accum[bit_idx] -= c;
            }
        }
    }
    let mut simhash = [0u64; 4];
    for bit_idx in 0..256 {
        if accum[bit_idx] > 0 {
            simhash[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
        }
    }

    // --- Top-K normalized histogram (unchanged) ---
    let mut freq_vec: Vec<(u64, usize)> = all_labels.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
    freq_vec.truncate(TOP_K);
    let total: f32 = freq_vec.iter().map(|(_, c)| *c as f32).sum();
    let mut hist: Vec<f32> = freq_vec.iter().map(|(_, c)| *c as f32).collect();
    if total > 0.0 {
        for v in &mut hist {
            *v /= total;
        }
    }
    hist.resize(TOP_K, 0.0);

    WlSignature {
        histogram: hist,
        simhash,
    }
}

fn compute_graph_topology(graph: &CoParticipationGraph) -> (usize, f32, usize, f32) {
    let n = graph.n;
    if n == 0 {
        return (0, 0.0, 0, 0.0);
    }

    let mut visited = vec![false; n];
    let mut components = 0usize;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(v) = queue.pop_front() {
            for &w in &graph.adj[v] {
                if !visited[w] {
                    visited[w] = true;
                    queue.push_back(w);
                }
            }
        }
    }

    let mut total_dist = 0u64;
    let mut pair_count = 0u64;
    let mut diameter = 0usize;
    for s in 0..n {
        let mut dist = vec![u32::MAX; n];
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);
        while let Some(v) = queue.pop_front() {
            for &w in &graph.adj[v] {
                if dist[w] == u32::MAX {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
            }
        }
        for t in (s + 1)..n {
            if dist[t] != u32::MAX {
                total_dist += dist[t] as u64;
                pair_count += 1;
                if dist[t] as usize > diameter {
                    diameter = dist[t] as usize;
                }
            }
        }
    }

    let avg_path = if pair_count > 0 {
        total_dist as f32 / pair_count as f32
    } else {
        0.0
    };
    let total_edges: usize = graph.adj.iter().map(|a| a.len()).sum::<usize>() / 2;
    let possible = if n >= 2 { n * (n - 1) / 2 } else { 1 };
    let density = total_edges as f32 / possible as f32;

    (components, avg_path, diameter, density)
}

// ─── Public API: Profile Computation ────────────────────────

/// Compute the full narrative style profile from the hypergraph.
pub fn compute_style_profile(
    narrative_id: &str,
    hypergraph: &Hypergraph,
) -> Result<NarrativeStyleProfile> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    let situations = load_sorted_situations(hypergraph, narrative_id)?;
    let n_sit = situations.len();
    let n_ent = entity_ids.len();

    let situation_density_curve = compute_situation_density(&situations);
    let (avg_parts, part_var, all_parts) = compute_participation_stats(&situations, hypergraph)?;
    let sit_ent_ratio = if n_ent > 0 {
        n_sit as f32 / n_ent as f32
    } else {
        0.0
    };

    let game_type_distribution = compute_game_type_dist(&situations);
    let role_entropy = compute_role_entropy(&all_parts);
    let power_asymmetry_gini = compute_power_asymmetry(&all_parts);
    let co_participation_density = compute_co_participation_density(&entity_ids, &all_parts);
    let protagonist_concentration = compute_protagonist_concentration(&entity_ids, &all_parts);

    let (
        avg_info_r0,
        deception_index,
        knowledge_asymmetry_gini,
        revelation_timing,
        secret_survival_rate,
    ) = compute_info_layer(&situations, &all_parts);

    let (
        causal_density,
        avg_causal_chain_length,
        max_causal_chain_length,
        unexplained_event_ratio,
        causal_branching_factor,
        causal_convergence_factor,
    ) = compute_causal_layer(&situations, hypergraph)?;

    let (
        allen_relation_distribution,
        flashback_frequency,
        temporal_span_variance,
        temporal_gap_ratio,
    ) = compute_temporal_layer(&situations);

    let co_graph = build_co_participation_graph(&entity_ids, &all_parts, 2);
    let wl_sig = compute_wl_signature(&co_graph);
    let wl_hash_histogram = wl_sig.histogram;
    let wl_simhash = wl_sig.simhash;
    let (community_count, avg_shortest_path, graph_diameter, edge_density) =
        compute_graph_topology(&co_graph);

    let narrative_surprise = compute_narrative_surprise(&situations, &all_parts, hypergraph);

    let (arc_type, arc_confidence) = crate::narrative::arc::classify_arc(narrative_id, hypergraph)
        .map(|c| (Some(format!("{:?}", c.arc_type)), c.confidence))
        .unwrap_or((None, 0.0));

    Ok(NarrativeStyleProfile {
        narrative_id: narrative_id.to_string(),
        computed_at: Utc::now(),
        situation_density_curve,
        avg_participants_per_situation: avg_parts,
        participation_count_variance: part_var,
        arc_type,
        arc_confidence,
        situation_entity_ratio: sit_ent_ratio,
        game_type_distribution,
        role_entropy,
        power_asymmetry_gini,
        co_participation_density,
        protagonist_concentration,
        avg_info_r0,
        deception_index,
        knowledge_asymmetry_gini,
        revelation_timing,
        secret_survival_rate,
        causal_density,
        avg_causal_chain_length,
        max_causal_chain_length,
        unexplained_event_ratio,
        causal_branching_factor,
        causal_convergence_factor,
        allen_relation_distribution,
        flashback_frequency,
        temporal_span_variance,
        temporal_gap_ratio,
        wl_hash_histogram,
        wl_simhash,
        community_count,
        avg_shortest_path,
        graph_diameter,
        edge_density,
        narrative_surprise,
        // D9.8: generative architecture axes (computed separately by D9 modules)
        promise_fulfillment_ratio: 0.0,
        average_payoff_distance: 0.0,
        fabula_sjuzet_divergence: 0.0,
        dramatic_irony_density: 0.0,
        focalization_diversity: 0.0,
        character_arc_completeness: 0.0,
        subplot_convergence_ratio: 0.0,
        scene_sequel_rhythm_score: 0.0,
    })
}

// ─── Public API: Similarity ─────────────────────────────────

/// Compare two style profiles structurally (legacy cosine + scalar_sim baseline).
///
/// This wrapper calls the weighted variant with a default (uniform-weight, no-corpus)
/// config and discards the per-metric breakdown — preserves the exact public signature
/// that existed before v0.28.
pub fn profile_similarity(a: &NarrativeStyleProfile, b: &NarrativeStyleProfile) -> StyleSimilarity {
    profile_similarity_weighted(a, b, &WeightedSimilarityConfig::default())
}

/// Compute prose-layer similarity (legacy cosine on function words + scalar comparison).
pub fn prose_similarity(a: &ProseStyleFeatures, b: &ProseStyleFeatures) -> f32 {
    prose_similarity_weighted(a, b, &WeightedSimilarityConfig::default())
}

/// Compare two narrative fingerprints using default uniform weighting.
pub fn fingerprint_similarity(
    a: &NarrativeFingerprint,
    b: &NarrativeFingerprint,
) -> StyleSimilarity {
    fingerprint_similarity_weighted(a, b, &WeightedSimilarityConfig::default())
}

/// Weighted prose-layer similarity.
///
/// Combines a vector-comparison kernel on function-word frequencies with scalar
/// similarity on 7 prose scalars (sentence length, std, word length, TTR,
/// dialogue ratio, Flesch-Kincaid, passive voice).
///
/// - When `corpus_stats` is available, uses Burrows-Cosine (z-scored cosine).
/// - Otherwise falls back to legacy raw cosine.
/// - When `scalar_stds.prose_scalars` is available, uses Mahalanobis on the scalar block.
pub fn prose_similarity_weighted(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    cfg: &WeightedSimilarityConfig,
) -> f32 {
    let fw_sim = match &cfg.corpus_stats {
        Some(stats) => crate::analysis::similarity_metrics::burrows_cosine(a, b, stats),
        None => cosine_sim(&a.function_word_frequencies, &b.function_word_frequencies),
    };
    let scalars_a = prose_scalar_vec(a);
    let scalars_b = prose_scalar_vec(b);
    let scalar_sim_value = match cfg
        .scalar_stds
        .as_ref()
        .filter(|s| !s.prose_scalars.is_empty())
    {
        Some(s) => crate::analysis::similarity_metrics::mahalanobis_sim_diag(
            &scalars_a,
            &scalars_b,
            &s.prose_scalars,
        ),
        None => avg_of(
            &scalars_a
                .iter()
                .zip(scalars_b.iter())
                .map(|(a, b)| scalar_sim(*a, *b))
                .collect::<Vec<_>>(),
        ),
    };
    // Blend fw-kernel and scalar-kernel by the configured per-sub-layer weights.
    let w_fw = cfg.w_prose_fw.max(0.0);
    let w_sc = cfg.w_prose_scalars.max(0.0);
    let denom = w_fw + w_sc;
    if denom <= 1e-6 {
        0.5 * fw_sim + 0.5 * scalar_sim_value
    } else {
        (w_fw * fw_sim + w_sc * scalar_sim_value) / denom
    }
}

/// Weighted structural-profile similarity.
///
/// Each of the 6 structural layers (rhythm, character, info, causal, temporal,
/// topology) is compared with its appropriate kernel:
/// - Distributions (situation density, Allen relations, game-type distribution)
///   use Jensen–Shannon similarity.
/// - Scalar blocks (rhythm scalars, character scalars, …) use Mahalanobis
///   when `scalar_stds` is populated for that block, otherwise fall back to
///   averaged `scalar_sim`.
/// - The topology layer blends JS on the WL histogram with Hamming similarity
///   on the WL SimHash (50/50 mix), combined with Mahalanobis on topology scalars.
pub fn profile_similarity_weighted(
    a: &NarrativeStyleProfile,
    b: &NarrativeStyleProfile,
    cfg: &WeightedSimilarityConfig,
) -> StyleSimilarity {
    use crate::analysis::similarity_metrics::{hamming_sim_u256, jensen_shannon_sim};

    let stds = cfg.scalar_stds.as_ref();
    let mut details = Vec::<LayerDetail>::new();

    // Rhythm
    let rhythm_js = jensen_shannon_sim(&a.situation_density_curve, &b.situation_density_curve);
    let rhythm_scalars_a = vec![
        a.avg_participants_per_situation,
        a.participation_count_variance,
        a.situation_entity_ratio,
    ];
    let rhythm_scalars_b = vec![
        b.avg_participants_per_situation,
        b.participation_count_variance,
        b.situation_entity_ratio,
    ];
    let rhythm_scalar = scalar_block_sim(
        &rhythm_scalars_a,
        &rhythm_scalars_b,
        stds.map(|s| &s.rhythm_scalars[..]),
    );
    details.push(LayerDetail {
        layer: "rhythm".into(),
        metric: "jensen_shannon".into(),
        value: rhythm_js,
    });
    details.push(LayerDetail {
        layer: "rhythm".into(),
        metric: "scalar".into(),
        value: rhythm_scalar,
    });
    let rhythm = 0.5 * rhythm_js + 0.5 * rhythm_scalar;

    // Character dynamics
    let char_js = jensen_shannon_sim(&a.game_type_distribution, &b.game_type_distribution);
    let char_scalars_a = vec![
        a.role_entropy,
        a.power_asymmetry_gini,
        a.co_participation_density,
        a.protagonist_concentration,
    ];
    let char_scalars_b = vec![
        b.role_entropy,
        b.power_asymmetry_gini,
        b.co_participation_density,
        b.protagonist_concentration,
    ];
    let char_scalar = scalar_block_sim(
        &char_scalars_a,
        &char_scalars_b,
        stds.map(|s| &s.character_scalars[..]),
    );
    details.push(LayerDetail {
        layer: "character".into(),
        metric: "jensen_shannon".into(),
        value: char_js,
    });
    details.push(LayerDetail {
        layer: "character".into(),
        metric: "scalar".into(),
        value: char_scalar,
    });
    let character = 0.5 * char_js + 0.5 * char_scalar;

    // Information flow — all scalar
    let info_a = vec![
        a.avg_info_r0,
        a.deception_index,
        a.knowledge_asymmetry_gini,
        a.revelation_timing,
        a.secret_survival_rate,
    ];
    let info_b = vec![
        b.avg_info_r0,
        b.deception_index,
        b.knowledge_asymmetry_gini,
        b.revelation_timing,
        b.secret_survival_rate,
    ];
    let info = scalar_block_sim(&info_a, &info_b, stds.map(|s| &s.info_scalars[..]));
    details.push(LayerDetail {
        layer: "info".into(),
        metric: "scalar".into(),
        value: info,
    });

    // Causal — all scalar
    let causal_a = vec![
        a.causal_density,
        a.avg_causal_chain_length,
        a.max_causal_chain_length as f32,
        a.unexplained_event_ratio,
        a.causal_branching_factor,
        a.causal_convergence_factor,
    ];
    let causal_b = vec![
        b.causal_density,
        b.avg_causal_chain_length,
        b.max_causal_chain_length as f32,
        b.unexplained_event_ratio,
        b.causal_branching_factor,
        b.causal_convergence_factor,
    ];
    let causal = scalar_block_sim(&causal_a, &causal_b, stds.map(|s| &s.causal_scalars[..]));
    details.push(LayerDetail {
        layer: "causal".into(),
        metric: "scalar".into(),
        value: causal,
    });

    // Temporal
    let temp_js = jensen_shannon_sim(
        &a.allen_relation_distribution,
        &b.allen_relation_distribution,
    );
    let temp_scalars_a = vec![
        a.flashback_frequency,
        a.temporal_span_variance,
        a.temporal_gap_ratio,
    ];
    let temp_scalars_b = vec![
        b.flashback_frequency,
        b.temporal_span_variance,
        b.temporal_gap_ratio,
    ];
    let temp_scalar = scalar_block_sim(
        &temp_scalars_a,
        &temp_scalars_b,
        stds.map(|s| &s.temporal_scalars[..]),
    );
    details.push(LayerDetail {
        layer: "temporal".into(),
        metric: "jensen_shannon".into(),
        value: temp_js,
    });
    details.push(LayerDetail {
        layer: "temporal".into(),
        metric: "scalar".into(),
        value: temp_scalar,
    });
    let temporal = 0.5 * temp_js + 0.5 * temp_scalar;

    // Topology — JS on histogram + Hamming on simhash + Mahalanobis on scalars
    let topo_hist_js = jensen_shannon_sim(&a.wl_hash_histogram, &b.wl_hash_histogram);
    let topo_hamming = hamming_sim_u256(a.wl_simhash, b.wl_simhash);
    let topo_scalars_a = vec![
        a.community_count as f32,
        a.avg_shortest_path,
        a.graph_diameter as f32,
        a.edge_density,
    ];
    let topo_scalars_b = vec![
        b.community_count as f32,
        b.avg_shortest_path,
        b.graph_diameter as f32,
        b.edge_density,
    ];
    let topo_scalar = scalar_block_sim(
        &topo_scalars_a,
        &topo_scalars_b,
        stds.map(|s| &s.topology_scalars[..]),
    );
    details.push(LayerDetail {
        layer: "topology".into(),
        metric: "jensen_shannon".into(),
        value: topo_hist_js,
    });
    details.push(LayerDetail {
        layer: "topology".into(),
        metric: "hamming_u256".into(),
        value: topo_hamming,
    });
    details.push(LayerDetail {
        layer: "topology".into(),
        metric: "scalar".into(),
        value: topo_scalar,
    });
    // Blend graph layer: 30% hist JS, 30% SimHash Hamming, 40% topology scalars.
    let topology = 0.3 * topo_hist_js + 0.3 * topo_hamming + 0.4 * topo_scalar;

    // Weighted overall (excludes prose — that is added by `fingerprint_similarity_weighted`).
    let weights = [
        (cfg.w_rhythm.max(0.0), rhythm),
        (cfg.w_character.max(0.0), character),
        (cfg.w_info.max(0.0), info),
        (cfg.w_causal.max(0.0), causal),
        (cfg.w_temporal.max(0.0), temporal),
        (cfg.w_topology.max(0.0), topology),
    ];
    let overall = weighted_mean(&weights);

    StyleSimilarity {
        overall,
        prose_similarity: 0.0,
        rhythm_similarity: rhythm,
        character_dynamics_similarity: character,
        information_flow_similarity: info,
        causal_similarity: causal,
        temporal_similarity: temporal,
        topology_similarity: topology,
        layer_details: details,
    }
}

/// Weighted fingerprint similarity — combines structural layers with prose.
pub fn fingerprint_similarity_weighted(
    a: &NarrativeFingerprint,
    b: &NarrativeFingerprint,
    cfg: &WeightedSimilarityConfig,
) -> StyleSimilarity {
    let mut sim = profile_similarity_weighted(&a.structure, &b.structure, cfg);
    sim.prose_similarity = prose_similarity_weighted(&a.prose, &b.prose, cfg);
    sim.layer_details.insert(
        0,
        LayerDetail {
            layer: "prose".into(),
            metric: if cfg.corpus_stats.is_some() {
                "burrows_cosine".into()
            } else {
                "raw_cosine".into()
            },
            value: sim.prose_similarity,
        },
    );

    // Recompute overall to include prose at the configured weight (prose weight is
    // max of the two prose sub-weights so the combined prose layer contributes
    // comparably to a single structural layer).
    let w_prose = cfg.w_prose_fw.max(cfg.w_prose_scalars).max(0.0);
    let weights = [
        (w_prose, sim.prose_similarity),
        (cfg.w_rhythm.max(0.0), sim.rhythm_similarity),
        (cfg.w_character.max(0.0), sim.character_dynamics_similarity),
        (cfg.w_info.max(0.0), sim.information_flow_similarity),
        (cfg.w_causal.max(0.0), sim.causal_similarity),
        (cfg.w_temporal.max(0.0), sim.temporal_similarity),
        (cfg.w_topology.max(0.0), sim.topology_similarity),
    ];
    sim.overall = weighted_mean(&weights);
    sim
}

// ─── Weighted-similarity helpers ────────────────────────────

fn prose_scalar_vec(p: &ProseStyleFeatures) -> Vec<f32> {
    vec![
        p.avg_sentence_length,
        p.sentence_length_std,
        p.avg_word_length,
        p.type_token_ratio,
        p.dialogue_ratio,
        p.flesch_kincaid_grade,
        p.passive_voice_ratio,
    ]
}

/// Scalar-block similarity: Mahalanobis when stds are provided (non-empty),
/// else average `scalar_sim`. Skips Mahalanobis if stds slice does not match
/// the length of `a` / `b`.
fn scalar_block_sim(a: &[f32], b: &[f32], stds: Option<&[f32]>) -> f32 {
    match stds {
        Some(s) if !s.is_empty() && s.len() >= a.len().min(b.len()) => {
            crate::analysis::similarity_metrics::mahalanobis_sim_diag(a, b, s)
        }
        _ => {
            if a.is_empty() || b.is_empty() {
                return 1.0;
            }
            let len = a.len().min(b.len());
            let mut sum = 0.0_f32;
            for i in 0..len {
                sum += scalar_sim(a[i], b[i]);
            }
            sum / len as f32
        }
    }
}

fn weighted_mean(pairs: &[(f32, f32)]) -> f32 {
    let denom: f32 = pairs.iter().map(|(w, _)| *w).sum();
    if denom <= 1e-6 {
        // Fall back to uniform mean of available values when all weights are zero.
        let n = pairs.len().max(1) as f32;
        return pairs.iter().map(|(_, v)| *v).sum::<f32>() / n;
    }
    pairs.iter().map(|(w, v)| w * v).sum::<f32>() / denom
}

// ─── Public API: Anomaly Detection ──────────────────────────

/// Top-K absolute deviations between two prose feature vectors. Shared by
/// `detect_style_anomalies` and `detect_per_source_type_anomalies` so the
/// feature list stays in one place.
pub fn ranked_prose_deviations(
    baseline: &ProseStyleFeatures,
    sample: &ProseStyleFeatures,
    top_k: usize,
) -> Vec<(String, f32)> {
    let mut deviations = vec![
        (
            "avg_sentence_length",
            sample.avg_sentence_length - baseline.avg_sentence_length,
        ),
        (
            "sentence_length_std",
            sample.sentence_length_std - baseline.sentence_length_std,
        ),
        (
            "avg_word_length",
            sample.avg_word_length - baseline.avg_word_length,
        ),
        (
            "type_token_ratio",
            sample.type_token_ratio - baseline.type_token_ratio,
        ),
        (
            "dialogue_ratio",
            sample.dialogue_ratio - baseline.dialogue_ratio,
        ),
        (
            "flesch_kincaid_grade",
            sample.flesch_kincaid_grade - baseline.flesch_kincaid_grade,
        ),
        (
            "passive_voice_ratio",
            sample.passive_voice_ratio - baseline.passive_voice_ratio,
        ),
    ]
    .into_iter()
    .map(|(name, diff)| (name.to_string(), diff.abs()))
    .collect::<Vec<_>>();
    deviations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    deviations.truncate(top_k);
    deviations
}

/// Detect style anomalies by comparing each chunk to the global profile.
pub fn detect_style_anomalies(
    global: &ProseStyleFeatures,
    chunks: &[ProseStyleFeatures],
    threshold: f32,
) -> Vec<StyleAnomaly> {
    chunks
        .iter()
        .enumerate()
        .filter_map(|(i, chunk)| {
            let sim = prose_similarity(global, chunk);
            (sim < threshold).then(|| StyleAnomaly {
                chunk_index: i,
                similarity_to_global: sim,
                most_deviant_features: ranked_prose_deviations(global, chunk, 5),
            })
        })
        .collect()
}

/// Render a `SourceType` enum as a stable string key for cohorting.
fn source_type_label(st: &crate::source::SourceType) -> String {
    match st {
        crate::source::SourceType::Custom(s) => s.clone(),
        other => format!("{:?}", other),
    }
}

/// Gather prose text attributed to `source`: every situation that carries a
/// SourceAttribution from this source, flattened into one string. `sit_index`
/// should map every situation UUID in the narrative to its full record so we
/// avoid per-attribution KV fetches (typical cohort: 500 reports × ~8
/// attributions = 4000 lookups otherwise).
fn build_source_prose(
    hypergraph: &Hypergraph,
    source: &crate::source::Source,
    sit_index: &std::collections::HashMap<uuid::Uuid, &crate::types::Situation>,
) -> crate::error::Result<String> {
    let mut buf = String::new();
    for attr in hypergraph.get_attributions_for_source(&source.id)? {
        if let Some(sit) = sit_index.get(&attr.target_id) {
            for cb in &sit.raw_content {
                if !buf.is_empty() {
                    buf.push(' ');
                }
                buf.push_str(cb.content.as_str());
            }
        }
    }
    Ok(buf)
}

/// A per-source style anomaly — the report's prose features deviate from the
/// baseline formed by **other reports of the same `source_type`**. Use when
/// asking "is this surveillance log stylistically inconsistent with other
/// surveillance logs?" rather than "… with this narrative's global prose
/// style?".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerSourceTypeAnomaly {
    pub source_id: uuid::Uuid,
    pub source_name: String,
    pub source_type: String,
    pub similarity_to_baseline: f32,
    pub baseline_cohort_size: usize,
    pub most_deviant_features: Vec<(String, f32)>,
}

/// Detect reports that diverge from their **own source-type** baseline.
///
/// Groups `(source, text)` pairs by their `source_type` label. For each group
/// with ≥ `min_cohort` members, computes an aggregate baseline from the other
/// members of the cohort (leave-one-out) and flags reports whose similarity
/// to that baseline is below `threshold`.
pub fn detect_per_source_type_anomalies(
    sources: &[(uuid::Uuid, String, String, ProseStyleFeatures)], // (id, name, source_type, features)
    threshold: f32,
    min_cohort: usize,
) -> Vec<PerSourceTypeAnomaly> {
    use std::collections::HashMap;

    let mut by_type: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, entry) in sources.iter().enumerate() {
        by_type.entry(entry.2.as_str()).or_default().push(i);
    }

    let mut anomalies = Vec::new();
    for (st, idxs) in &by_type {
        if idxs.len() < min_cohort {
            continue;
        }
        for &i in idxs {
            // Leave-one-out baseline over the rest of the cohort.
            let rest: Vec<ProseStyleFeatures> = idxs
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| sources[j].3.clone())
                .collect();
            let baseline = aggregate_features(&rest);
            let (id, name, _st, feat) = &sources[i];
            let sim = prose_similarity(&baseline, feat);
            if sim < threshold {
                anomalies.push(PerSourceTypeAnomaly {
                    source_id: *id,
                    source_name: name.clone(),
                    source_type: (*st).to_string(),
                    similarity_to_baseline: sim,
                    baseline_cohort_size: idxs.len() - 1,
                    most_deviant_features: ranked_prose_deviations(&baseline, feat, 5),
                });
            }
        }
    }
    anomalies
}

// ─── Public API: Radar Chart ────────────────────────────────

/// Compute radar chart axes from a style profile, all normalized to [0, 1].
pub fn compute_radar(profile: &NarrativeStyleProfile) -> FingerprintRadar {
    let mean: f32 = profile.situation_density_curve.iter().sum::<f32>()
        / profile.situation_density_curve.len().max(1) as f32;
    let pacing = (profile
        .situation_density_curve
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f32>()
        / profile.situation_density_curve.len().max(1) as f32)
        .sqrt()
        * 10.0;

    let tc_max = (NUM_ALLEN_RELATIONS as f32).log2();
    let temporal_complexity = if tc_max > 0.0 {
        shannon_entropy(&profile.allen_relation_distribution) / tc_max
    } else {
        0.0
    };

    let sv_max = (NUM_GAME_TYPES as f32).log2();
    let strategic_variety = if sv_max > 0.0 {
        shannon_entropy(&profile.game_type_distribution) / sv_max
    } else {
        0.0
    };

    FingerprintRadar {
        pacing: pacing.clamp(0.0, 1.0),
        ensemble: (1.0 - profile.protagonist_concentration).clamp(0.0, 1.0),
        causal_density: (profile.causal_density / 3.0).clamp(0.0, 1.0),
        info_r0: (profile.avg_info_r0 / 5.0).clamp(0.0, 1.0),
        deception: profile.deception_index.clamp(0.0, 1.0),
        temporal_complexity: temporal_complexity.clamp(0.0, 1.0),
        strategic_variety: strategic_variety.clamp(0.0, 1.0),
        power_asymmetry: profile.power_asymmetry_gini.clamp(0.0, 1.0),
        protagonist_focus: profile.protagonist_concentration.clamp(0.0, 1.0),
        late_revelation: profile.revelation_timing.clamp(0.0, 1.0),
        subplot_richness: (profile.community_count as f32 / 10.0).clamp(0.0, 1.0),
        surprise: profile.narrative_surprise.clamp(0.0, 1.0),
    }
}

// ─── Public API: Data Completeness ──────────────────────────

/// Assess which profile layers have meaningful (non-default) data.
/// Note: `prose` requires a `NarrativeFingerprint`; use `assess_fingerprint_completeness` for that.
pub fn assess_completeness(profile: &NarrativeStyleProfile) -> DataCompleteness {
    DataCompleteness {
        prose: false, // prose completeness requires fingerprint; see assess_fingerprint_completeness
        rhythm: profile.avg_participants_per_situation > 0.0
            || profile.situation_entity_ratio > 0.0,
        character_dynamics: profile.role_entropy > 0.0 || profile.co_participation_density > 0.0,
        information_flow: profile.avg_info_r0 > 0.0
            || profile.deception_index > 0.0
            || profile.secret_survival_rate > 0.0,
        causal: profile.causal_density > 0.0 || profile.max_causal_chain_length > 0,
        temporal: profile.flashback_frequency > 0.0
            || profile.temporal_span_variance > 0.0
            || profile.temporal_gap_ratio > 0.0,
        topology: profile.community_count > 0 || profile.edge_density > 0.0,
    }
}

/// Assess completeness including prose layer (requires the full fingerprint).
pub fn assess_fingerprint_completeness(fp: &NarrativeFingerprint) -> DataCompleteness {
    let mut c = assess_completeness(&fp.structure);
    c.prose = fp.prose.total_words > 0;
    c
}

// ─── InferenceEngine Implementations ───────────────────────

use crate::analysis::extract_narrative_id;
use crate::analysis::stylometry::{aggregate_features, compute_prose_features};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Collect raw text from a narrative's situations for prose analysis.
fn collect_narrative_text(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> crate::error::Result<String> {
    // Prefer chunk text if available, fall back to situation raw_content
    let chunks = hypergraph.list_chunks_by_narrative(narrative_id)?;
    if !chunks.is_empty() {
        let mut text = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            text.push_str(chunk.text_without_overlap(i));
        }
        return Ok(text);
    }
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    Ok(situations
        .iter()
        .flat_map(|s| s.raw_content.iter())
        .map(|cb| cb.content.as_str())
        .collect::<Vec<_>>()
        .join(" "))
}

/// Build a full NarrativeFingerprint (prose + structure) for a narrative.
pub fn build_fingerprint(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> crate::error::Result<NarrativeFingerprint> {
    let text = collect_narrative_text(hypergraph, narrative_id)?;
    let prose = compute_prose_features(&text);
    let structure = compute_style_profile(narrative_id, hypergraph)?;
    Ok(NarrativeFingerprint {
        narrative_id: narrative_id.to_string(),
        computed_at: Utc::now(),
        prose,
        structure,
    })
}

/// Compute narrative style profile via the inference job queue.
pub struct StyleProfileEngine;

impl InferenceEngine for StyleProfileEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::StyleProfile
    }

    fn estimate_cost(
        &self,
        _job: &InferenceJob,
        _hypergraph: &Hypergraph,
    ) -> crate::error::Result<u64> {
        Ok(8000) // 8 seconds estimate
    }

    fn execute(
        &self,
        job: &InferenceJob,
        hypergraph: &Hypergraph,
    ) -> crate::error::Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let fingerprint = build_fingerprint(hypergraph, narrative_id)?;

        // Persist to the KV cache keys used by the style/fingerprint/radar
        // HTTP endpoints so a stale cached profile doesn't mask the new one.
        let profile_key = crate::analysis::analysis_key(
            crate::hypergraph::keys::ANALYSIS_STYLE_PROFILE,
            &[narrative_id],
        );
        let profile_bytes = serde_json::to_vec(&fingerprint.structure)?;
        hypergraph.store().put(&profile_key, &profile_bytes)?;

        let fp_key = crate::analysis::analysis_key(
            crate::hypergraph::keys::ANALYSIS_FINGERPRINT,
            &[narrative_id],
        );
        let fp_bytes = serde_json::to_vec(&fingerprint)?;
        hypergraph.store().put(&fp_key, &fp_bytes)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::StyleProfile,
            target_id: job.target_id,
            result: serde_json::to_value(&fingerprint)?,
            confidence: 1.0,
            explanation: Some("Narrative style profile computed".into()),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Compare the style fingerprints of two narratives via the inference job queue.
pub struct StyleComparisonEngine;

impl InferenceEngine for StyleComparisonEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::StyleComparison
    }

    fn estimate_cost(
        &self,
        _job: &InferenceJob,
        _hypergraph: &Hypergraph,
    ) -> crate::error::Result<u64> {
        Ok(12000) // 12 seconds (two profiles)
    }

    fn execute(
        &self,
        job: &InferenceJob,
        hypergraph: &Hypergraph,
    ) -> crate::error::Result<InferenceResult> {
        let nid_a = job
            .parameters
            .get("narrative_id_a")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::error::TensaError::InferenceError("missing narrative_id_a".into())
            })?;
        let nid_b = job
            .parameters
            .get("narrative_id_b")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::error::TensaError::InferenceError("missing narrative_id_b".into())
            })?;

        let fp_a = build_fingerprint(hypergraph, nid_a)?;
        let fp_b = build_fingerprint(hypergraph, nid_b)?;
        let similarity = fingerprint_similarity(&fp_a, &fp_b);

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::StyleComparison,
            target_id: job.target_id,
            result: serde_json::to_value(&similarity)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Style comparison between '{}' and '{}': overall={:.3}",
                nid_a, nid_b, similarity.overall
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

/// Detect style anomalies within a narrative via the inference job queue.
pub struct StyleAnomalyEngine;

impl InferenceEngine for StyleAnomalyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::StyleAnomaly
    }

    fn estimate_cost(
        &self,
        _job: &InferenceJob,
        _hypergraph: &Hypergraph,
    ) -> crate::error::Result<u64> {
        Ok(10000) // 10 seconds
    }

    fn execute(
        &self,
        job: &InferenceJob,
        hypergraph: &Hypergraph,
    ) -> crate::error::Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let threshold = job
            .parameters
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32;

        // Calibrated mode is opted into by `mode == "calibrated"` or by
        // passing `alpha`/`n_iter` without an explicit `threshold`.
        let mode = job
            .parameters
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("threshold");
        let alpha = job
            .parameters
            .get("alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(crate::analysis::stylometry_stats::DEFAULT_ALPHA as f64)
            as f32;
        let n_iter = job
            .parameters
            .get("n_iter")
            .and_then(|v| v.as_u64())
            .unwrap_or(crate::analysis::stylometry_stats::DEFAULT_BOOTSTRAP_ITER as u64)
            as usize;
        let seed = job
            .parameters
            .get("seed")
            .and_then(|v| v.as_u64())
            .unwrap_or(0xC0FFEE);
        let calibrated = matches!(mode, "calibrated" | "pvalue" | "bootstrap");
        let per_source_type = matches!(mode, "per_source_type" | "source_type_baseline");

        // ── Per-source-type mode: build cohort baselines by source_type ──
        if per_source_type {
            let min_cohort = job
                .parameters
                .get("min_cohort")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize;
            // Load situations once and index by id — avoids N+1 KV fetches
            // when `build_source_prose` resolves every attribution per source.
            let all_situations = hypergraph.list_situations_by_narrative(narrative_id)?;
            let sit_index: std::collections::HashMap<uuid::Uuid, &crate::types::Situation> =
                all_situations.iter().map(|s| (s.id, s)).collect();
            let sources = hypergraph.list_sources_for_narrative(narrative_id)?;
            let mut cohort: Vec<(uuid::Uuid, String, String, ProseStyleFeatures)> = Vec::new();
            for src in &sources {
                let text = build_source_prose(hypergraph, src, &sit_index)?;
                if text.trim().is_empty() {
                    continue;
                }
                let st_label = source_type_label(&src.source_type);
                cohort.push((
                    src.id,
                    src.name.clone(),
                    st_label,
                    compute_prose_features(&text),
                ));
            }
            let anomalies = detect_per_source_type_anomalies(&cohort, threshold, min_cohort.max(2));
            return Ok(InferenceResult {
                job_id: job.id.clone(),
                job_type: InferenceJobType::StyleAnomaly,
                target_id: job.target_id,
                result: serde_json::to_value(&anomalies)?,
                confidence: 1.0,
                explanation: Some(format!(
                    "Per-source-type style anomalies: {} flagged across {} sources",
                    anomalies.len(),
                    cohort.len()
                )),
                status: JobStatus::Completed,
                created_at: job.created_at,
                completed_at: Some(Utc::now()),
            });
        }

        // Build per-chunk prose features and aggregate for global baseline
        let chunks = hypergraph.list_chunks_by_narrative(narrative_id)?;
        let (global, chunk_features) = if !chunks.is_empty() {
            let per_chunk: Vec<_> = chunks
                .iter()
                .map(|c| compute_prose_features(&c.text))
                .collect();
            (aggregate_features(&per_chunk), per_chunk)
        } else {
            let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
            let per_sit: Vec<_> = situations
                .iter()
                .map(|s| {
                    let text: String = s
                        .raw_content
                        .iter()
                        .map(|cb| cb.content.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    compute_prose_features(&text)
                })
                .collect();
            (aggregate_features(&per_sit), per_sit)
        };

        if calibrated {
            let cfg = WeightedSimilarityConfig::load_or_default(hypergraph.store());
            let results = crate::analysis::stylometry_stats::detect_style_anomalies_calibrated(
                &chunk_features,
                alpha,
                n_iter,
                &cfg,
                seed,
            );
            let flagged = results.iter().filter(|a| a.flagged).count();
            return Ok(InferenceResult {
                job_id: job.id.clone(),
                job_type: InferenceJobType::StyleAnomaly,
                target_id: job.target_id,
                result: serde_json::to_value(&results)?,
                confidence: 1.0,
                explanation: Some(format!(
                    "Calibrated style anomaly detection: {} flagged (alpha={}, n_iter={})",
                    flagged, alpha, n_iter
                )),
                status: JobStatus::Completed,
                created_at: job.created_at,
                completed_at: Some(Utc::now()),
            });
        }

        let anomalies = detect_style_anomalies(&global, &chunk_features, threshold);

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::StyleAnomaly,
            target_id: job.target_id,
            result: serde_json::to_value(&anomalies)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Style anomaly detection: {} anomalies found (threshold={:.2})",
                anomalies.len(),
                threshold
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod per_source_type_tests {
    use super::*;

    fn feat(sent_len: f32, dialogue: f32) -> ProseStyleFeatures {
        ProseStyleFeatures {
            avg_sentence_length: sent_len,
            sentence_length_std: 2.0,
            avg_word_length: 5.0,
            type_token_ratio: 0.5,
            dialogue_ratio: dialogue,
            flesch_kincaid_grade: 10.0,
            passive_voice_ratio: 0.1,
            ..Default::default()
        }
    }

    #[test]
    fn per_source_type_flags_outlier_within_cohort() {
        // 6 surveillance logs — 5 consistent + 1 clear outlier. With leave-one-out
        // baselines the outlier can't drag the baseline when it's the one being
        // compared; conversely, the normal ones' baselines stay tight even with
        // the outlier included (5 normals dominate 1 anomaly).
        let id_outlier = uuid::Uuid::now_v7();
        let sources = vec![
            (
                uuid::Uuid::now_v7(),
                "Surv 1".into(),
                "surveillance_log".into(),
                feat(12.0, 0.0),
            ),
            (
                uuid::Uuid::now_v7(),
                "Surv 2".into(),
                "surveillance_log".into(),
                feat(12.5, 0.02),
            ),
            (
                uuid::Uuid::now_v7(),
                "Surv 3".into(),
                "surveillance_log".into(),
                feat(13.0, 0.01),
            ),
            (
                uuid::Uuid::now_v7(),
                "Surv 4".into(),
                "surveillance_log".into(),
                feat(12.3, 0.0),
            ),
            (
                uuid::Uuid::now_v7(),
                "Surv 5".into(),
                "surveillance_log".into(),
                feat(12.8, 0.03),
            ),
            (
                id_outlier,
                "Surv 6 (odd)".into(),
                "surveillance_log".into(),
                feat(60.0, 0.95),
            ),
        ];
        let anomalies = detect_per_source_type_anomalies(&sources, 0.80, 3);
        assert!(
            anomalies.iter().any(|a| a.source_id == id_outlier),
            "the clear outlier must be flagged"
        );
        assert!(
            anomalies
                .iter()
                .all(|a| a.source_type == "surveillance_log"),
            "only the surveillance cohort is large enough to evaluate"
        );
    }

    #[test]
    fn per_source_type_skips_small_cohorts() {
        let sources = vec![
            (
                uuid::Uuid::now_v7(),
                "A".into(),
                "anonymous_tip".into(),
                feat(12.0, 0.0),
            ),
            (
                uuid::Uuid::now_v7(),
                "B".into(),
                "anonymous_tip".into(),
                feat(40.0, 0.9),
            ),
        ];
        let anomalies = detect_per_source_type_anomalies(&sources, 0.9, 3);
        assert!(
            anomalies.is_empty(),
            "cohort of 2 is below min_cohort=3 — nothing should be flagged"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};
    use crate::types::*;
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    fn add_situation_at(hg: &Hypergraph, narrative: &str, offset_hours: i64) -> Uuid {
        let start = Utc::now() + Duration::hours(offset_hours);
        let end = start + Duration::hours(1);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(end),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    fn add_situation_with_game(hg: &Hypergraph, narrative: &str, gc: GameClassification) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: Some(GameStructure {
                game_type: gc,
                info_structure: InfoStructureType::Complete,
                description: None,
                maturity: MaturityLevel::Candidate,
            }),
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("game")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    fn add_situation_with_allen(
        hg: &Hypergraph,
        narrative: &str,
        relations: Vec<AllenRelationTo>,
    ) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations,
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("temporal")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    fn link_with_role(hg: &Hypergraph, entity: Uuid, situation: Uuid, role: Role) {
        hg.add_participant(Participation {
            entity_id: entity,
            situation_id: situation,
            role,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    fn zeroed_prose() -> ProseStyleFeatures {
        compute_prose_features("")
    }

    fn make_prose(
        avg_sl: f32,
        sl_std: f32,
        awl: f32,
        ttr: f32,
        dr: f32,
        fkg: f32,
        pvr: f32,
    ) -> ProseStyleFeatures {
        let mut p = zeroed_prose();
        p.avg_sentence_length = avg_sl;
        p.sentence_length_std = sl_std;
        p.avg_word_length = awl;
        p.type_token_ratio = ttr;
        p.dialogue_ratio = dr;
        p.flesch_kincaid_grade = fkg;
        p.passive_voice_ratio = pvr;
        // Give non-zero function word freqs so cosine_sim works.
        for (i, v) in p.function_word_frequencies.iter_mut().enumerate() {
            *v = 1.0 / (i as f32 + 1.0);
        }
        p
    }

    // 1
    #[test]
    fn test_empty_narrative() {
        let hg = make_hg();
        let p = compute_style_profile("empty", &hg).unwrap();
        assert_eq!(p.avg_participants_per_situation, 0.0);
        assert_eq!(p.causal_density, 0.0);
        assert_eq!(p.max_causal_chain_length, 0);
        assert_eq!(p.community_count, 0);
    }

    // 2
    #[test]
    fn test_situation_density_uniform() {
        let hg = make_hg();
        for i in 0..20 {
            add_situation_at(&hg, "uni", i * 10);
        }
        let p = compute_style_profile("uni", &hg).unwrap();
        for &v in &p.situation_density_curve {
            assert!(v > 0.0);
        }
        assert!((p.situation_density_curve.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    // 3
    #[test]
    fn test_situation_density_front_loaded() {
        let hg = make_hg();
        for i in 0..10 {
            add_situation_at(&hg, "front", i);
        }
        add_situation_at(&hg, "front", 200);
        let p = compute_style_profile("front", &hg).unwrap();
        let first: f32 = p.situation_density_curve[..10].iter().sum();
        let second: f32 = p.situation_density_curve[10..].iter().sum();
        assert!(first > second);
    }

    // 4
    #[test]
    fn test_avg_participants() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "avg");
        let e2 = add_entity(&hg, "B", "avg");
        let s1 = add_situation(&hg, "avg");
        let s2 = add_situation(&hg, "avg");
        link(&hg, e1, s1);
        link(&hg, e2, s1);
        link(&hg, e1, s2);
        let p = compute_style_profile("avg", &hg).unwrap();
        assert!((p.avg_participants_per_situation - 1.5).abs() < 0.01);
    }

    // 5
    #[test]
    fn test_game_type_distribution_single() {
        let hg = make_hg();
        for _ in 0..3 {
            add_situation_with_game(&hg, "pd", GameClassification::PrisonersDilemma);
        }
        let p = compute_style_profile("pd", &hg).unwrap();
        assert!((p.game_type_distribution[0] - 1.0).abs() < 0.01);
        for &v in &p.game_type_distribution[1..] {
            assert!(v.abs() < 0.01);
        }
    }

    // 6
    #[test]
    fn test_game_type_distribution_empty() {
        let hg = make_hg();
        add_situation(&hg, "ng");
        let p = compute_style_profile("ng", &hg).unwrap();
        // No game_structure data → all zeros (honest empty)
        for &v in &p.game_type_distribution {
            assert!((v - 0.0).abs() < 0.01);
        }
    }

    // 7
    #[test]
    fn test_role_entropy_single() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "rs");
        let e2 = add_entity(&hg, "B", "rs");
        let s1 = add_situation(&hg, "rs");
        let s2 = add_situation(&hg, "rs");
        link(&hg, e1, s1);
        link(&hg, e2, s1);
        link(&hg, e1, s2);
        let p = compute_style_profile("rs", &hg).unwrap();
        assert!(p.role_entropy.abs() < 0.01);
    }

    // 8
    #[test]
    fn test_role_entropy_diverse() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "rd");
        let e2 = add_entity(&hg, "B", "rd");
        let e3 = add_entity(&hg, "C", "rd");
        let e4 = add_entity(&hg, "D", "rd");
        let s = add_situation(&hg, "rd");
        link_with_role(&hg, e1, s, Role::Protagonist);
        link_with_role(&hg, e2, s, Role::Antagonist);
        link_with_role(&hg, e3, s, Role::Witness);
        link_with_role(&hg, e4, s, Role::Bystander);
        let p = compute_style_profile("rd", &hg).unwrap();
        assert!((p.role_entropy - 2.0).abs() < 0.01);
    }

    // 9
    #[test]
    fn test_co_participation_density_complete() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "cpd");
        let e2 = add_entity(&hg, "B", "cpd");
        let e3 = add_entity(&hg, "C", "cpd");
        let s = add_situation(&hg, "cpd");
        link(&hg, e1, s);
        link(&hg, e2, s);
        link(&hg, e3, s);
        let p = compute_style_profile("cpd", &hg).unwrap();
        assert!((p.co_participation_density - 1.0).abs() < 0.01);
    }

    // 10
    #[test]
    fn test_co_participation_density_none() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "cpn");
        let e2 = add_entity(&hg, "B", "cpn");
        let s1 = add_situation(&hg, "cpn");
        let s2 = add_situation(&hg, "cpn");
        link(&hg, e1, s1);
        link(&hg, e2, s2);
        let p = compute_style_profile("cpn", &hg).unwrap();
        assert!(p.co_participation_density.abs() < 0.01);
    }

    // 11
    #[test]
    fn test_causal_density_none() {
        let hg = make_hg();
        add_situation(&hg, "cdn");
        add_situation(&hg, "cdn");
        let p = compute_style_profile("cdn", &hg).unwrap();
        assert!(p.causal_density.abs() < 0.01);
    }

    // 12
    #[test]
    fn test_causal_chain_linear() {
        let hg = make_hg();
        let s1 = add_situation_at(&hg, "ccl", 0);
        let s2 = add_situation_at(&hg, "ccl", 1);
        let s3 = add_situation_at(&hg, "ccl", 2);
        let s4 = add_situation_at(&hg, "ccl", 3);
        for (a, b) in [(s1, s2), (s2, s3), (s3, s4)] {
            hg.add_causal_link(CausalLink {
                from_situation: a,
                to_situation: b,
                mechanism: None,
                strength: 1.0,
                causal_type: CausalType::Sufficient,
                maturity: MaturityLevel::Candidate,
            })
            .unwrap();
        }
        let p = compute_style_profile("ccl", &hg).unwrap();
        assert_eq!(p.max_causal_chain_length, 3);
    }

    // 13
    #[test]
    fn test_unexplained_ratio() {
        let hg = make_hg();
        let s1 = add_situation_at(&hg, "unex", 0);
        let s2 = add_situation_at(&hg, "unex", 1);
        let _s3 = add_situation_at(&hg, "unex", 2);
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            mechanism: None,
            strength: 1.0,
            causal_type: CausalType::Sufficient,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        let p = compute_style_profile("unex", &hg).unwrap();
        assert!((p.unexplained_event_ratio - 0.5).abs() < 0.01);
    }

    // 14
    #[test]
    fn test_allen_distribution_all_before() {
        let hg = make_hg();
        let target = Uuid::now_v7();
        for _ in 0..5 {
            add_situation_with_allen(
                &hg,
                "ab",
                vec![AllenRelationTo {
                    target_situation: target,
                    relation: AllenRelation::Before,
                }],
            );
        }
        let p = compute_style_profile("ab", &hg).unwrap();
        assert!((p.allen_relation_distribution[0] - 1.0).abs() < 0.01);
    }

    // 15
    #[test]
    fn test_graph_density_complete() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "gdc");
        let e2 = add_entity(&hg, "B", "gdc");
        let e3 = add_entity(&hg, "C", "gdc");
        let s1 = add_situation(&hg, "gdc");
        let s2 = add_situation(&hg, "gdc");
        link(&hg, e1, s1);
        link(&hg, e2, s1);
        link(&hg, e3, s1);
        link(&hg, e1, s2);
        link(&hg, e2, s2);
        link(&hg, e3, s2);
        let p = compute_style_profile("gdc", &hg).unwrap();
        assert!((p.edge_density - 1.0).abs() < 0.01);
    }

    // 16
    #[test]
    fn test_graph_density_empty() {
        let hg = make_hg();
        add_entity(&hg, "A", "gde");
        add_entity(&hg, "B", "gde");
        let p = compute_style_profile("gde", &hg).unwrap();
        assert!(p.edge_density.abs() < 0.01);
    }

    // 17
    #[test]
    fn test_profile_similarity_identical() {
        let hg = make_hg();
        let e = add_entity(&hg, "A", "si");
        let s = add_situation(&hg, "si");
        link(&hg, e, s);
        let p = compute_style_profile("si", &hg).unwrap();
        let sim = profile_similarity(&p, &p);
        assert!((sim.overall - 1.0).abs() < 0.01);
        assert!((sim.rhythm_similarity - 1.0).abs() < 0.01);
    }

    // 18
    #[test]
    fn test_profile_similarity_different() {
        let hg = make_hg();
        let ea = add_entity(&hg, "A", "da");
        let sa = add_situation(&hg, "da");
        link(&hg, ea, sa);
        let pa = compute_style_profile("da", &hg).unwrap();
        for i in 0..5 {
            let e = add_entity(&hg, &format!("B{}", i), "db");
            let s = add_situation_at(&hg, "db", i as i64);
            link(&hg, e, s);
        }
        let pb = compute_style_profile("db", &hg).unwrap();
        let sim = profile_similarity(&pa, &pb);
        assert!(sim.overall < 1.0);
    }

    // 19
    #[test]
    fn test_prose_similarity_identical() {
        let p = make_prose(15.0, 5.0, 4.5, 0.6, 0.3, 8.0, 0.1);
        assert!((prose_similarity(&p, &p) - 1.0).abs() < 0.01);
    }

    // 20
    #[test]
    fn test_prose_similarity_orthogonal() {
        let a = make_prose(5.0, 1.0, 3.0, 0.9, 0.0, 2.0, 0.0);
        let mut b = make_prose(50.0, 20.0, 8.0, 0.2, 0.9, 15.0, 0.5);
        // Make function word vectors orthogonal.
        b.function_word_frequencies = vec![0.0; 100];
        if b.function_word_frequencies.len() > 50 {
            b.function_word_frequencies[50] = 1.0;
        }
        let sim = prose_similarity(&a, &b);
        assert!(sim < 0.5);
    }

    // 21
    #[test]
    fn test_fingerprint_similarity_self() {
        let hg = make_hg();
        let e = add_entity(&hg, "A", "fs");
        let s = add_situation(&hg, "fs");
        link(&hg, e, s);
        let fp = NarrativeFingerprint {
            narrative_id: "fs".into(),
            computed_at: Utc::now(),
            prose: make_prose(15.0, 5.0, 4.5, 0.6, 0.3, 8.0, 0.1),
            structure: compute_style_profile("fs", &hg).unwrap(),
        };
        let sim = fingerprint_similarity(&fp, &fp);
        assert!((sim.overall - 1.0).abs() < 0.02);
    }

    // 22
    #[test]
    fn test_anomaly_detection_all_consistent() {
        let g = make_prose(15.0, 5.0, 4.5, 0.6, 0.3, 8.0, 0.1);
        let chunks = vec![g.clone(), g.clone(), g.clone()];
        assert!(detect_style_anomalies(&g, &chunks, 0.5).is_empty());
    }

    // 23
    #[test]
    fn test_anomaly_detection_one_outlier() {
        let g = make_prose(15.0, 5.0, 4.5, 0.6, 0.3, 8.0, 0.1);
        let mut outlier = make_prose(50.0, 30.0, 9.0, 0.1, 0.9, 18.0, 0.6);
        // Make function word frequencies very different (reversed pattern).
        for (i, v) in outlier.function_word_frequencies.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) / 100.0;
        }
        let chunks = vec![g.clone(), outlier, g.clone()];
        let anomalies = detect_style_anomalies(&g, &chunks, 0.9);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].chunk_index, 1);
    }

    // 24
    #[test]
    fn test_radar_chart_bounds() {
        let hg = make_hg();
        let e = add_entity(&hg, "A", "rc");
        let s = add_situation(&hg, "rc");
        link(&hg, e, s);
        let r = compute_radar(&compute_style_profile("rc", &hg).unwrap());
        for v in [
            r.pacing,
            r.ensemble,
            r.causal_density,
            r.info_r0,
            r.deception,
            r.temporal_complexity,
            r.strategic_variety,
            r.power_asymmetry,
            r.protagonist_focus,
            r.late_revelation,
            r.subplot_richness,
            r.surprise,
        ] {
            assert!((0.0..=1.0).contains(&v), "Value {} out of [0,1]", v);
        }
    }

    // 25
    #[test]
    fn test_data_completeness() {
        let hg = make_hg();
        let dc = assess_completeness(&compute_style_profile("empty_dc", &hg).unwrap());
        assert!(!dc.prose);
        assert!(!dc.rhythm);
        assert!(!dc.causal);
    }

    // 26
    #[test]
    fn test_cosine_similarity_known() {
        assert!(cosine_sim(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.01);
        assert!((cosine_sim(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.01);
        assert!((cosine_sim(&[1.0, 1.0], &[1.0, 1.0]) - 1.0).abs() < 0.01);
    }

    // 27
    #[test]
    fn test_gini_coefficient() {
        assert!(gini(&[1.0, 1.0, 1.0, 1.0]).abs() < 0.01);
        assert!(gini(&[0.0, 0.0, 0.0, 100.0]) > 0.7);
    }

    // 28
    #[test]
    fn test_situation_entity_ratio() {
        let hg = make_hg();
        add_entity(&hg, "A", "ser");
        add_entity(&hg, "B", "ser");
        for _ in 0..4 {
            add_situation(&hg, "ser");
        }
        let p = compute_style_profile("ser", &hg).unwrap();
        assert!((p.situation_entity_ratio - 2.0).abs() < 0.01);
    }

    // 29
    #[test]
    fn test_participation_count_variance() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "A", "pcv");
        let e2 = add_entity(&hg, "B", "pcv");
        let s1 = add_situation(&hg, "pcv");
        let _s2 = add_situation(&hg, "pcv");
        link(&hg, e1, s1);
        link(&hg, e2, s1);
        let p = compute_style_profile("pcv", &hg).unwrap();
        assert!((p.participation_count_variance - 1.0).abs() < 0.01);
    }

    // 30
    #[test]
    fn test_scalar_similarity_self() {
        assert!((scalar_sim(5.0, 5.0) - 1.0).abs() < 0.01);
        assert!((scalar_sim(0.0, 0.0) - 1.0).abs() < 0.01);
    }

    // ─── Engine Tests ──────────────────────────────────────

    fn make_style_job(job_type: InferenceJobType, params: serde_json::Value) -> InferenceJob {
        InferenceJob {
            id: "style-test".to_string(),
            job_type,
            target_id: Uuid::now_v7(),
            parameters: params,
            priority: crate::types::JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    // 31
    #[test]
    fn test_style_profile_engine_execute() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "Alice", "sp-eng");
        let s1 = add_situation(&hg, "sp-eng");
        link(&hg, e1, s1);

        let engine = StyleProfileEngine;
        assert_eq!(engine.job_type(), InferenceJobType::StyleProfile);

        let job = make_style_job(
            InferenceJobType::StyleProfile,
            serde_json::json!({"narrative_id": "sp-eng"}),
        );
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());
        assert!(result.result.get("narrative_id").is_some());
    }

    // 32
    #[test]
    fn test_style_comparison_engine_execute() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "Alice", "cmp-a");
        let s1 = add_situation(&hg, "cmp-a");
        link(&hg, e1, s1);
        let e2 = add_entity(&hg, "Bob", "cmp-b");
        let s2 = add_situation(&hg, "cmp-b");
        link(&hg, e2, s2);

        let engine = StyleComparisonEngine;
        assert_eq!(engine.job_type(), InferenceJobType::StyleComparison);

        let job = make_style_job(
            InferenceJobType::StyleComparison,
            serde_json::json!({"narrative_id_a": "cmp-a", "narrative_id_b": "cmp-b"}),
        );
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.result.get("overall").is_some());
    }

    // 33
    #[test]
    fn test_style_anomaly_engine_execute() {
        let hg = make_hg();
        let e1 = add_entity(&hg, "Alice", "anom-eng");
        let s1 = add_situation(&hg, "anom-eng");
        link(&hg, e1, s1);

        let engine = StyleAnomalyEngine;
        assert_eq!(engine.job_type(), InferenceJobType::StyleAnomaly);

        let job = make_style_job(
            InferenceJobType::StyleAnomaly,
            serde_json::json!({"narrative_id": "anom-eng", "threshold": 0.5}),
        );
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.result.is_array());
    }
}
