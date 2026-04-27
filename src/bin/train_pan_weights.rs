//! Offline training binary for PAN@CLEF authorship-verification weights.
//!
//! Given a labeled PAN verification dataset (JSONL with `{id, pair, same_author}`
//! lines or split pairs+truth files), fits a 4-parameter logistic regression
//! that maps per-layer prose features to the same-author probability:
//!
//! ```text
//!   logit = intercept
//!         + w_burrows_cosine  * burrows_cosine(a, b)
//!         + w_sentence_length * |avg_sentence_length(a) - avg_sentence_length(b)|
//!         + w_readability     * |flesch_kincaid(a)      - flesch_kincaid(b)|
//!         + w_dialogue        * |dialogue_ratio(a)      - dialogue_ratio(b)|
//! ```
//!
//! Prints the learned `VerificationConfig` as JSON on stdout (optionally writes
//! to `--output`). Upload the resulting JSON to the running server via
//! `PUT /settings/style-weights` (wrapped inside a `WeightedSimilarityConfig`
//! body) to have `StyleAnomalyEngine` and `/style/compare?ci=true` pick it up
//! automatically via `WeightedSimilarityConfig::load_or_default`.
//!
//! This binary has *no* dependency on the running TENSA server — it reads the
//! dataset from disk and writes weights to disk.
//!
//! Build with: `cargo build --features cli --bin train_pan_weights`
//! Run:        `cargo run --features cli --bin train_pan_weights -- --dataset PATH [--output PATH]`

use std::path::PathBuf;

use clap::Parser;

use tensa::analysis::pan_loader::{apply_truth, load_pan_jsonl, load_pan_truth};
use tensa::analysis::pan_verification::{sigmoid, VerificationConfig};
use tensa::analysis::similarity_metrics::burrows_cosine;
use tensa::analysis::stylometry::{
    compute_corpus_stats, compute_prose_features, ProseStyleFeatures,
};

#[derive(Parser, Debug)]
#[command(name = "train_pan_weights")]
#[command(about = "Train PAN@CLEF authorship-verification weights from a labeled dataset")]
struct Args {
    /// Path to PAN pairs JSONL (each line: {id, pair: [text_a, text_b], same_author})
    #[arg(long)]
    dataset: PathBuf,

    /// Optional separate truth JSONL (each line: {id, value})
    #[arg(long)]
    truth: Option<PathBuf>,

    /// Output path for the learned VerificationConfig JSON. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Logistic-regression learning rate.
    #[arg(long, default_value_t = 0.05)]
    lr: f32,

    /// Training iterations (full-batch gradient descent).
    #[arg(long, default_value_t = 500)]
    iters: usize,

    /// L2 regularization coefficient.
    #[arg(long, default_value_t = 0.01)]
    l2: f32,

    /// Optional seeded uncertainty-band width in the final config.
    #[arg(long, default_value_t = 0.05)]
    uncertainty_band: f32,
}

fn feature_vector(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &tensa::analysis::stylometry::CorpusStats,
) -> [f32; 4] {
    [
        burrows_cosine(a, b, corpus_stats),
        (a.avg_sentence_length - b.avg_sentence_length).abs(),
        (a.flesch_kincaid_grade - b.flesch_kincaid_grade).abs(),
        (a.dialogue_ratio - b.dialogue_ratio).abs(),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    eprintln!("Loading dataset {} …", args.dataset.display());
    let mut pairs = load_pan_jsonl(&args.dataset)?;
    if let Some(truth_path) = &args.truth {
        eprintln!("Merging truth from {} …", truth_path.display());
        let truth = load_pan_truth(truth_path)?;
        apply_truth(&mut pairs, &truth);
    }
    eprintln!("Loaded {} pairs", pairs.len());
    let labels: Vec<bool> = pairs
        .iter()
        .map(|p| {
            p.same_author
                .ok_or_else(|| format!("pair {} missing label", p.id))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Precompute prose features over union of all texts.
    eprintln!("Computing prose features …");
    let mut features: Vec<ProseStyleFeatures> = Vec::with_capacity(pairs.len() * 2);
    for pair in &pairs {
        features.push(compute_prose_features(&pair.text_a));
        features.push(compute_prose_features(&pair.text_b));
    }
    let corpus_stats = compute_corpus_stats(&features);

    // Build feature matrix X (N x 4) and label vector y.
    let n = pairs.len();
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let a = &features[i * 2];
        let b = &features[i * 2 + 1];
        x.push(feature_vector(a, b, &corpus_stats));
    }
    let y: Vec<f32> = labels.iter().map(|&l| if l { 1.0 } else { 0.0 }).collect();

    // Full-batch gradient descent on logistic loss + L2.
    eprintln!(
        "Training (lr={}, iters={}, l2={}) …",
        args.lr, args.iters, args.l2
    );
    let mut intercept = 0.0_f32;
    let mut w = [0.0_f32; 4];

    for iter in 0..args.iters {
        let mut grad_intercept = 0.0_f32;
        let mut grad_w = [0.0_f32; 4];
        let mut loss = 0.0_f32;
        for i in 0..n {
            let logit = intercept + w.iter().zip(x[i].iter()).map(|(w, xi)| w * xi).sum::<f32>();
            let pred = sigmoid(logit);
            let err = pred - y[i];
            grad_intercept += err;
            for k in 0..4 {
                grad_w[k] += err * x[i][k];
            }
            // Binary cross-entropy (clipped).
            let p = pred.clamp(1e-7, 1.0 - 1e-7);
            loss += -(y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln());
        }
        grad_intercept /= n as f32;
        for k in 0..4 {
            grad_w[k] = grad_w[k] / n as f32 + args.l2 * w[k];
        }
        loss = loss / n as f32 + 0.5 * args.l2 * w.iter().map(|v| v * v).sum::<f32>();

        intercept -= args.lr * grad_intercept;
        for k in 0..4 {
            w[k] -= args.lr * grad_w[k];
        }

        if iter % 50 == 0 || iter == args.iters - 1 {
            eprintln!("  iter {:>4}: loss={:.4}", iter, loss);
        }
    }

    let cfg = VerificationConfig {
        intercept,
        w_burrows_cosine: w[0],
        w_sentence_length_diff: w[1],
        w_readability_diff: w[2],
        w_dialogue_diff: w[3],
        uncertainty_band: args.uncertainty_band,
    };
    let json = serde_json::to_string_pretty(&cfg)?;

    if let Some(out) = args.output {
        std::fs::write(&out, &json)?;
        eprintln!("Wrote {}", out.display());
    } else {
        println!("{}", json);
    }

    Ok(())
}
