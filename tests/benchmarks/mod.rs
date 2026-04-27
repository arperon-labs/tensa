//! Benchmark suite for academic credibility (Sprint 12).
//!
//! All benchmarks are `#[ignore]` — they require external datasets at
//! `TENSA_BENCHMARK_DATA` env var path. Run with: `cargo test --ignored`
//!
//! Each benchmark outputs a JSON report with: dataset, metrics, baseline_comparison.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Standard benchmark report format.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub benchmark: String,
    pub dataset: String,
    pub metrics: BenchmarkMetrics,
    pub baseline_comparison: Vec<BaselineComparison>,
    pub duration_ms: u64,
}

/// Core metrics for classification/retrieval benchmarks.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub accuracy: Option<f64>,
    pub latency_ms: Option<f64>,
    pub extra: serde_json::Value,
}

/// Comparison against a published baseline.
#[derive(Debug, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub method: String,
    pub metric: String,
    pub baseline_value: f64,
    pub tensa_value: f64,
    pub delta: f64,
}

impl BenchmarkMetrics {
    /// Compute P/R/F1 from confusion matrix counts.
    pub fn from_counts(tp: usize, fp: usize, fneg: usize) -> Self {
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fneg > 0 {
            tp as f64 / (tp + fneg) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self {
            precision,
            recall,
            f1,
            accuracy: None,
            latency_ms: None,
            extra: serde_json::Value::Null,
        }
    }
}

impl BenchmarkReport {
    /// Print a Markdown summary to stderr (visible with --nocapture).
    pub fn print_markdown(&self) {
        eprintln!("## {}", self.benchmark);
        eprintln!("Dataset: {}", self.dataset);
        eprintln!(
            "P={:.3} R={:.3} F1={:.3} ({}ms)",
            self.metrics.precision, self.metrics.recall, self.metrics.f1, self.duration_ms
        );
        for b in &self.baseline_comparison {
            let arrow = if b.delta >= 0.0 { "+" } else { "" };
            eprintln!(
                "  vs {}: {} {:.3} → {:.3} ({}{:.3})",
                b.method, b.metric, b.baseline_value, b.tensa_value, arrow, b.delta
            );
        }
        eprintln!();
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Get the benchmark data directory from `TENSA_BENCHMARK_DATA` env var.
#[allow(dead_code)]
pub fn benchmark_data_dir() -> Option<std::path::PathBuf> {
    std::env::var("TENSA_BENCHMARK_DATA")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

/// Time a closure, returning (result, duration_ms).
#[allow(dead_code)]
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let start = Instant::now();
    let result = f();
    let ms = start.elapsed().as_millis() as u64;
    (result, ms)
}
