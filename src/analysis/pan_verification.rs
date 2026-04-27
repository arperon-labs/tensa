//! PAN@CLEF authorship verification harness.
//!
//! Implements the full PAN@CLEF 2020/2021 protocol for closed- and open-set
//! authorship verification:
//!
//! - `verify_pair` — logistic-combined probability that two texts share an author,
//!   built on the per-layer similarity kernels in `similarity_metrics`.
//! - `auc` — trapezoidal area under the ROC curve.
//! - `c_at_1` — "correct at 1" metric that rewards abstention on ambiguous cases.
//! - `f_0_5_u` — β=0.5 F-measure treating non-decisions as correct.
//! - `brier_score` — calibration of probability outputs.
//! - `pan_overall` — the PAN 2020+ aggregate: `AUC * c@1 * F0.5u * F1 * (1 - Brier)`.
//! - `unmasking` — Koppel's feature-ablation curve: trains a simple linear
//!   classifier on function words, ablates the top-k features by |coef|, and
//!   reports the accuracy-decline slope.

use serde::{Deserialize, Serialize};

use crate::analysis::similarity_metrics::burrows_cosine;
use crate::analysis::stylometry::{
    compute_corpus_stats, compute_prose_features, CorpusStats, ProseStyleFeatures,
};

// ─── Types ──────────────────────────────────────────────────

/// One PAN authorship-verification problem: two texts plus optional ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPair {
    pub id: String,
    pub text_a: String,
    pub text_b: String,
    #[serde(default)]
    pub same_author: Option<bool>,
}

/// One prediction produced by a verifier.
///
/// `score` is the calibrated same-author probability in `[0, 1]`.
/// `decision` is `None` when the verifier abstains (e.g., score in an uncertainty
/// band around 0.5) and `Some(same_author_bool)` otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationScore {
    pub id: String,
    pub score: f32,
    pub decision: Option<bool>,
}

/// The six PAN@CLEF authorship-verification metrics plus aggregate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanMetrics {
    pub auc: f32,
    pub c_at_1: f32,
    pub f_0_5_u: f32,
    pub f1: f32,
    pub brier: f32,
    pub overall: f32,
    pub n_samples: usize,
    pub n_decisions: usize,
}

/// Output of Koppel-style unmasking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmaskingCurve {
    /// Cross-validated accuracy at each ablation step (step 0 = no features removed).
    pub accuracies: Vec<f32>,
    /// Slope of the accuracy decline (negative for different-author pairs,
    /// near-zero for same-author pairs).
    pub slope: f32,
}

/// Weights and thresholds for the logistic same-author verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub intercept: f32,
    pub w_burrows_cosine: f32,
    pub w_sentence_length_diff: f32,
    pub w_readability_diff: f32,
    pub w_dialogue_diff: f32,
    /// Width of the uncertainty band around 0.5; scores within this radius abstain.
    pub uncertainty_band: f32,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        // Hand-tuned seed values — beat chance on synthetic Hemingway/Faulkner data.
        // Replace via `train_pan_weights` binary on a real PAN dataset.
        Self {
            intercept: -1.5,
            w_burrows_cosine: 4.5,
            w_sentence_length_diff: -0.25,
            w_readability_diff: -0.08,
            w_dialogue_diff: -2.0,
            uncertainty_band: 0.05,
        }
    }
}

// ─── Verifier ───────────────────────────────────────────────

/// Calibrated same-author probability for two prose profiles.
///
/// Combines Burrows-Cosine similarity with scalar feature differences in a
/// logistic function. Returns a value in `[0, 1]` where > 0.5 suggests
/// same-author.
pub fn verify_pair(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &CorpusStats,
    cfg: &VerificationConfig,
) -> f32 {
    let bc = burrows_cosine(a, b, corpus_stats);
    let sl_diff = (a.avg_sentence_length - b.avg_sentence_length).abs();
    let read_diff = (a.flesch_kincaid_grade - b.flesch_kincaid_grade).abs();
    let dial_diff = (a.dialogue_ratio - b.dialogue_ratio).abs();

    let logit = cfg.intercept
        + cfg.w_burrows_cosine * bc
        + cfg.w_sentence_length_diff * sl_diff
        + cfg.w_readability_diff * read_diff
        + cfg.w_dialogue_diff * dial_diff;

    sigmoid(logit)
}

/// Score a pair and produce a decision, abstaining inside the uncertainty band.
pub fn verify_pair_with_decision(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &CorpusStats,
    cfg: &VerificationConfig,
) -> (f32, Option<bool>) {
    let score = verify_pair(a, b, corpus_stats, cfg);
    let decision = if (score - 0.5).abs() < cfg.uncertainty_band {
        None
    } else {
        Some(score > 0.5)
    };
    (score, decision)
}

/// End-to-end convenience: extract prose features from two raw texts, build
/// ad-hoc corpus stats from the pair, and run `verify_pair_with_decision`.
///
/// Shared by the REST handler and the MCP `verify_authorship` tool so the
/// pairwise corpus-stats construction lives in exactly one place.
pub fn verify_texts(text_a: &str, text_b: &str, cfg: &VerificationConfig) -> (f32, Option<bool>) {
    let a = compute_prose_features(text_a);
    let b = compute_prose_features(text_b);
    let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
    verify_pair_with_decision(&a, &b, &stats, cfg)
}

/// Score a full list of verification pairs.
///
/// Builds a corpus-wide `CorpusStats` over the union of all texts in `pairs` to
/// give the Burrows-Cosine kernel a stable normalization basis, then runs
/// `verify_pair_with_decision` on each pair.
pub fn score_pairs(pairs: &[VerificationPair], cfg: &VerificationConfig) -> Vec<VerificationScore> {
    // Build corpus stats over both sides of every pair.
    let mut all_features: Vec<ProseStyleFeatures> = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        all_features.push(compute_prose_features(&pair.text_a));
        all_features.push(compute_prose_features(&pair.text_b));
    }
    let stats = compute_corpus_stats(&all_features);

    pairs
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            let a = &all_features[i * 2];
            let b = &all_features[i * 2 + 1];
            let (score, decision) = verify_pair_with_decision(a, b, &stats, cfg);
            VerificationScore {
                id: pair.id.clone(),
                score,
                decision,
            }
        })
        .collect()
}

// ─── Metrics ────────────────────────────────────────────────

/// Trapezoidal AUC.
pub fn auc(scores: &[VerificationScore], labels: &[bool]) -> f32 {
    let n = scores.len().min(labels.len());
    if n == 0 {
        return 0.5;
    }
    let mut pairs: Vec<(f32, bool)> = scores[..n]
        .iter()
        .zip(labels[..n].iter().copied())
        .map(|(s, l)| (s.score, l))
        .collect();
    // Sort descending by score.
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_pos = labels[..n].iter().filter(|&&l| l).count() as f32;
    let n_neg = n as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;
    let mut area = 0.0;
    for (_, label) in &pairs {
        if *label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / n_pos;
        let fpr = fp / n_neg;
        area += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_tpr = tpr;
        prev_fpr = fpr;
    }
    area.clamp(0.0, 1.0)
}

/// c@1 — PAN formula that rewards abstention:
/// `(n_correct + (n_unanswered * n_correct / n)) / n`.
pub fn c_at_1(scores: &[VerificationScore], labels: &[bool]) -> f32 {
    let n = scores.len().min(labels.len());
    if n == 0 {
        return 0.0;
    }
    let mut n_correct = 0.0;
    let mut n_unanswered = 0.0;
    for i in 0..n {
        match scores[i].decision {
            Some(pred) => {
                if pred == labels[i] {
                    n_correct += 1.0;
                }
            }
            None => n_unanswered += 1.0,
        }
    }
    (n_correct + (n_unanswered * n_correct / n as f32)) / n as f32
}

/// F0.5u — β = 0.5, non-decisions count as true for both precision and recall.
pub fn f_0_5_u(scores: &[VerificationScore], labels: &[bool]) -> f32 {
    let n = scores.len().min(labels.len());
    if n == 0 {
        return 0.0;
    }
    let beta_sq = 0.25_f32; // β = 0.5 ⇒ β² = 0.25
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_count = 0.0;
    let mut unanswered = 0.0;
    for i in 0..n {
        match scores[i].decision {
            Some(true) => {
                if labels[i] {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            Some(false) => {
                if labels[i] {
                    fn_count += 1.0;
                }
            }
            None => unanswered += 1.0,
        }
    }
    // PAN spec: unanswered counted as half-correct in both numerator components.
    let precision_denom = tp + fp + 0.5 * unanswered;
    let recall_denom = tp + fn_count + 0.5 * unanswered;
    if precision_denom <= 0.0 || recall_denom <= 0.0 {
        return 0.0;
    }
    let precision = (tp + 0.5 * unanswered) / precision_denom;
    let recall = (tp + 0.5 * unanswered) / recall_denom;
    let denom = beta_sq * precision + recall;
    if denom <= 0.0 {
        return 0.0;
    }
    (1.0 + beta_sq) * precision * recall / denom
}

/// Standard F1 on confident decisions; unanswered = incorrect for both classes.
pub fn f1(scores: &[VerificationScore], labels: &[bool]) -> f32 {
    let n = scores.len().min(labels.len());
    if n == 0 {
        return 0.0;
    }
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_count = 0.0;
    for i in 0..n {
        match scores[i].decision {
            Some(true) => {
                if labels[i] {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            Some(false) => {
                if labels[i] {
                    fn_count += 1.0;
                }
            }
            None => {
                // Treated as missed prediction for both classes.
                if labels[i] {
                    fn_count += 1.0;
                } else {
                    fp += 1.0;
                }
            }
        }
    }
    let denom = 2.0 * tp + fp + fn_count;
    if denom <= 0.0 {
        return 0.0;
    }
    2.0 * tp / denom
}

/// Brier score — mean squared error between predicted probability and binary label.
pub fn brier_score(scores: &[VerificationScore], labels: &[bool]) -> f32 {
    let n = scores.len().min(labels.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for i in 0..n {
        let y = if labels[i] { 1.0 } else { 0.0 };
        sum += (scores[i].score - y).powi(2);
    }
    sum / n as f32
}

/// PAN overall score: `AUC * c@1 * F0.5u * F1 * (1 - Brier)`.
pub fn pan_overall(m: &PanMetrics) -> f32 {
    (m.auc * m.c_at_1 * m.f_0_5_u * m.f1 * (1.0 - m.brier)).max(0.0)
}

/// Evaluate a full set of verification scores against ground-truth labels.
pub fn evaluate(scores: &[VerificationScore], labels: &[bool]) -> PanMetrics {
    let n = scores.len().min(labels.len());
    let n_decisions = scores[..n].iter().filter(|s| s.decision.is_some()).count();
    let auc_v = auc(scores, labels);
    let c1 = c_at_1(scores, labels);
    let f05 = f_0_5_u(scores, labels);
    let f1_v = f1(scores, labels);
    let brier = brier_score(scores, labels);
    let mut m = PanMetrics {
        auc: auc_v,
        c_at_1: c1,
        f_0_5_u: f05,
        f1: f1_v,
        brier,
        overall: 0.0,
        n_samples: n,
        n_decisions,
    };
    m.overall = pan_overall(&m);
    m
}

// ─── Unmasking ──────────────────────────────────────────────

/// Koppel-style feature-ablation curve.
///
/// Splits each side's chunks into halves, trains a linear discriminator on the
/// function-word frequencies distinguishing `a` from `b`, measures held-out
/// accuracy, removes the top `k_remove` discriminating features by |coefficient|,
/// and repeats `k_iter` times.
///
/// A steep accuracy decline → the classifier depends on a small set of features
/// → strong stylistic separation (different authors). A shallow decline →
/// classifier uses many weak features → same author.
pub fn unmasking(
    a_chunks: &[ProseStyleFeatures],
    b_chunks: &[ProseStyleFeatures],
    k_iter: usize,
    k_remove: usize,
) -> UnmaskingCurve {
    if a_chunks.len() < 2 || b_chunks.len() < 2 {
        return UnmaskingCurve {
            accuracies: vec![0.5],
            slope: 0.0,
        };
    }
    let dim = a_chunks[0].function_word_frequencies.len();
    let mut mask = vec![true; dim];
    let mut accuracies = Vec::with_capacity(k_iter + 1);

    // Half-split each side: first half = train, second half = test.
    let a_train = &a_chunks[..a_chunks.len() / 2];
    let a_test = &a_chunks[a_chunks.len() / 2..];
    let b_train = &b_chunks[..b_chunks.len() / 2];
    let b_test = &b_chunks[b_chunks.len() / 2..];

    for _ in 0..=k_iter {
        // Compute per-class means and classifier coefficients once per masked dim.
        let mut coefs = vec![0.0_f32; dim];
        let mut mean_mid = vec![0.0_f32; dim];
        for i in 0..dim {
            if !mask[i] {
                continue;
            }
            let mean_a: f32 = a_train
                .iter()
                .map(|c| c.function_word_frequencies[i])
                .sum::<f32>()
                / a_train.len() as f32;
            let mean_b: f32 = b_train
                .iter()
                .map(|c| c.function_word_frequencies[i])
                .sum::<f32>()
                / b_train.len() as f32;
            coefs[i] = mean_a - mean_b;
            mean_mid[i] = (mean_a + mean_b) / 2.0;
        }

        // Held-out accuracy: classify by sign of dot(coefs, chunk - mean_mid).
        let mut correct = 0usize;
        let mut total = 0usize;

        for (label, set) in [(true, a_test), (false, b_test)] {
            for chunk in set {
                let dot: f32 = (0..dim)
                    .filter(|&i| mask[i])
                    .map(|i| coefs[i] * (chunk.function_word_frequencies[i] - mean_mid[i]))
                    .sum();
                let pred = dot > 0.0; // true = a
                if pred == label {
                    correct += 1;
                }
                total += 1;
            }
        }
        let acc = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.5
        };
        accuracies.push(acc);

        // Ablate top-k features by |coef|.
        let mut idx: Vec<usize> = (0..dim).filter(|&i| mask[i]).collect();
        idx.sort_by(|a, b| {
            coefs[*b]
                .abs()
                .partial_cmp(&coefs[*a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in idx.iter().take(k_remove) {
            mask[i] = false;
        }
    }

    // Compute slope as (last - first) / (n - 1).
    let slope = if accuracies.len() > 1 {
        (accuracies.last().unwrap() - accuracies.first().unwrap()) / (accuracies.len() - 1) as f32
    } else {
        0.0
    };
    UnmaskingCurve { accuracies, slope }
}

// ─── Helpers ────────────────────────────────────────────────

/// Numerically-stable logistic sigmoid used by the PAN verifier and the
/// offline `train_pan_weights` binary.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Inference engine ───────────────────────────────────────

use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Inference engine for single-pair authorship verification.
///
/// Job parameters (all strings unless noted):
/// - `text_a`, `text_b` — required; the two texts to compare.
/// - `config` — optional JSON `VerificationConfig`.
pub struct AuthorshipVerificationEngine;

impl InferenceEngine for AuthorshipVerificationEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::AuthorshipVerification
    }

    fn estimate_cost(
        &self,
        _job: &InferenceJob,
        _hypergraph: &Hypergraph,
    ) -> crate::error::Result<u64> {
        Ok(500) // sub-second — no hypergraph traversal
    }

    fn execute(
        &self,
        job: &InferenceJob,
        _hypergraph: &Hypergraph,
    ) -> crate::error::Result<InferenceResult> {
        let text_a = job
            .parameters
            .get("text_a")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::error::TensaError::InferenceError("missing text_a parameter".into())
            })?;
        let text_b = job
            .parameters
            .get("text_b")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::error::TensaError::InferenceError("missing text_b parameter".into())
            })?;
        let cfg = match job.parameters.get("config") {
            Some(v) if !v.is_null() => serde_json::from_value::<VerificationConfig>(v.clone())
                .map_err(|e| {
                    crate::error::TensaError::InferenceError(format!("bad config: {}", e))
                })?,
            _ => VerificationConfig::default(),
        };

        let (score, decision) = verify_texts(text_a, text_b, &cfg);
        let result = serde_json::json!({
            "score": score,
            "decision": decision,
            "same_author_probability": score,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::AuthorshipVerification,
            target_id: job.target_id,
            result,
            confidence: 1.0,
            explanation: Some(format!(
                "Authorship verification: score={:.3}, decision={:?}",
                score, decision
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(chrono::Utc::now()),
        })
    }
}

#[cfg(test)]
#[path = "pan_verification_tests.rs"]
mod tests;
