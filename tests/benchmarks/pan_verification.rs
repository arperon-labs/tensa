//! PAN@CLEF authorship-verification benchmark.
//!
//! Builds a synthetic same/different-author corpus from the Hemingway/Faulkner
//! samples in `stylometry.rs` (plus a third author) and runs the full PAN 2020+
//! metric suite — AUC, c@1, F0.5u, F1, Brier, and the aggregate overall score —
//! against TENSA's `verify_pair` verifier.
//!
//! This benchmark is synthetic by design: published PAN datasets are large and
//! not redistributed. When a real dataset is available, call
//! `POST /style/pan/evaluate` with `dataset_path` or run the `train_pan_weights`
//! binary to produce tuned weights.

use super::*;
use tensa::analysis::pan_verification::{
    evaluate as pan_evaluate, score_pairs, VerificationConfig, VerificationPair,
};

const HEMINGWAY_CHUNKS: [&str; 6] = [
    "The old man sat alone. He drank. The sun set over the sea.",
    "He walked to the dock. The boat was small. The line held in his hands.",
    "The fish was big. He fought it. The line held. His arms ached all night.",
    "She was gone. He drank. He did not speak. The rain was cold.",
    "They fought the bulls. The crowd was loud. He was tired by noon.",
    "The room was dark. He lay still. He could hear the waves outside.",
];

const FAULKNER_CHUNKS: [&str; 6] = [
    "The long and winding sentences that stretched across the page, each one folding into the next with a kind of breathless urgency that seemed to suggest the very fabric of time was unraveling.",
    "She thought about the old house, the one that had stood for generations upon generations, its walls thick with the accumulated weight of memory and regret and something else entirely.",
    "And so it was that the afternoon, that particular afternoon which would later come to be remembered as the beginning of everything, passed slowly in the humid stillness of the delta.",
    "Across the porch where the generations had sat and watched the cotton rise and fall, the light fell in that particular gold of an afternoon that had not yet consented to become evening.",
    "He remembered, or thought he remembered, the summer of the great flood, though perhaps it was not that summer at all but another, further back, when the river had first turned.",
    "In the dim and unremembered light of the afternoon when the wind had only barely begun to stir the curtains in the room where she had been, he thought of everything.",
];

/// Third author — short, punchy, first-person, contemporary (distinct from both).
const AUSTEN_CHUNKS: [&str; 6] = [
    "It is a truth universally acknowledged that a young man in possession of a good fortune must be in want of a wife.",
    "Miss Bingley, however, declared her preference for tea, and the matter was settled with a curtsey and a knowing smile.",
    "Elizabeth observed her sister's agitation with considerable concern, though she was careful to betray no sign of it to the gentleman in question.",
    "The ball at Netherfield would not be postponed, Mr. Bingley assured them, for he held society in the highest regard, especially when it involved dancing.",
    "Mr. Darcy, for his part, was sensible of the impropriety of the whole affair, yet he could not bring himself to depart.",
    "Jane received the letter with perfect composure, though her heart, as she later confessed, was in a state of some disorder.",
];

/// Pair two chunks under a deterministic id. `same` indicates ground truth.
fn make_pair(id: &str, a: &str, b: &str, same: bool) -> VerificationPair {
    VerificationPair {
        id: id.into(),
        text_a: a.into(),
        text_b: b.into(),
        same_author: Some(same),
    }
}

/// Build a balanced same-author / different-author verification corpus.
///
/// - Same-author pairs: random disjoint chunks within each author.
/// - Different-author pairs: one chunk each from two distinct authors.
fn build_pairs() -> Vec<VerificationPair> {
    let authors: [(&str, &[&str]); 3] = [
        ("hem", &HEMINGWAY_CHUNKS),
        ("falk", &FAULKNER_CHUNKS),
        ("aus", &AUSTEN_CHUNKS),
    ];
    let mut pairs = Vec::new();
    // Same-author pairs: (chunk[2i], chunk[2i+1]) for each author.
    for (name, chunks) in &authors {
        for i in 0..chunks.len() / 2 {
            let a = chunks[2 * i];
            let b = chunks[2 * i + 1];
            pairs.push(make_pair(&format!("{}-same-{}", name, i), a, b, true));
        }
    }
    // Different-author pairs: one from each distinct author combination.
    for i in 0..authors.len() {
        for j in (i + 1)..authors.len() {
            let (name_i, chunks_i) = authors[i];
            let (name_j, chunks_j) = authors[j];
            for k in 0..3 {
                pairs.push(make_pair(
                    &format!("{}-{}-diff-{}", name_i, name_j, k),
                    chunks_i[k],
                    chunks_j[k],
                    false,
                ));
            }
        }
    }
    pairs
}

/// Run the PAN benchmark and report a BenchmarkReport.
pub fn run() -> BenchmarkReport {
    let pairs = build_pairs();
    let labels: Vec<bool> = pairs
        .iter()
        .map(|p| p.same_author.unwrap_or(false))
        .collect();
    let cfg = VerificationConfig::default();

    let (metrics, ms) = timed(|| {
        let scores = score_pairs(&pairs, &cfg);
        pan_evaluate(&scores, &labels)
    });

    let bench_metrics = BenchmarkMetrics {
        precision: metrics.f1 as f64,
        recall: metrics.c_at_1 as f64,
        f1: metrics.f1 as f64,
        accuracy: Some(metrics.c_at_1 as f64),
        latency_ms: Some(ms as f64),
        extra: serde_json::json!({
            "auc": metrics.auc,
            "c_at_1": metrics.c_at_1,
            "f_0_5_u": metrics.f_0_5_u,
            "f1": metrics.f1,
            "brier": metrics.brier,
            "overall": metrics.overall,
            "n_samples": metrics.n_samples,
            "n_decisions": metrics.n_decisions,
        }),
    };

    let baselines = vec![BaselineComparison {
        method: "chance_baseline".into(),
        metric: "auc".into(),
        baseline_value: 0.5,
        tensa_value: metrics.auc as f64,
        delta: (metrics.auc as f64) - 0.5,
    }];

    BenchmarkReport {
        benchmark: "pan_verification".into(),
        dataset: "synthetic_hemingway_faulkner_austen".into(),
        metrics: bench_metrics,
        baseline_comparison: baselines,
        duration_ms: ms,
    }
}
