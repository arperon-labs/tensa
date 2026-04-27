//! `tensa-bench` — Academic benchmark suite for TENSA.
//!
//! Evaluates TENSA against standard academic datasets in three domains:
//! 1. **TKG** — Temporal knowledge graph link prediction (ICEWS14/18, GDELT)
//! 2. **Narrative** — Narrative understanding (ROCStories, NarrativeQA, MAVEN-ERE)
//! 3. **Multi-hop** — Multi-hop GraphRAG (HotpotQA, 2WikiMultiHopQA)
//!
//! All heavy benchmarks are `#[ignore]` — run with:
//! ```bash
//! TENSA_BENCHMARK_DATA=/path/to/data cargo test --no-default-features --ignored bench_
//! ```

pub mod adapters;
pub mod baselines;
pub mod datasets;
pub mod metrics;
pub mod report;

use serde::{Deserialize, Serialize};

// Re-export shared infra from the Sprint 12 benchmarks.
#[path = "../benchmarks/mod.rs"]
pub mod benchmarks_base;
pub use benchmarks_base::{benchmark_data_dir, timed, BaselineComparison};

/// Top-level report aggregating all benchmark domains.
#[derive(Debug, Serialize, Deserialize)]
pub struct TensaBenchReport {
    pub suite: String,
    pub version: String,
    pub timestamp: String,
    pub domains: Vec<DomainReport>,
    pub total_duration_sec: f64,
}

/// Results for one benchmark domain (TKG, Narrative, or Multi-hop).
#[derive(Debug, Serialize, Deserialize)]
pub struct DomainReport {
    pub domain: String,
    pub datasets: Vec<DatasetReport>,
}

/// Results for a single dataset within a domain.
#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetReport {
    pub name: String,
    pub task: String,
    pub metrics: serde_json::Value,
    pub baselines: Vec<BaselineComparison>,
    pub num_items: usize,
    pub duration_ms: u64,
}

impl TensaBenchReport {
    /// Print a one-line-per-dataset headline summary to stderr.
    pub fn print_summary(&self) {
        eprintln!("=== TENSA-BENCH {} ({}) ===", self.version, self.timestamp);
        for domain in &self.domains {
            for ds in &domain.datasets {
                let best_delta = ds
                    .baselines
                    .iter()
                    .map(|b| b.delta)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                let best_method = ds
                    .baselines
                    .iter()
                    .max_by(|a, b| {
                        a.delta
                            .partial_cmp(&b.delta)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|b| b.method.as_str())
                    .unwrap_or("n/a");
                let arrow = if best_delta >= 0.0 { "+" } else { "" };
                eprintln!(
                    "  {} ({}): {} items, {:.1}s  (vs {}: {}{:.3})",
                    domain.domain,
                    ds.name,
                    ds.num_items,
                    ds.duration_ms as f64 / 1000.0,
                    best_method,
                    arrow,
                    best_delta,
                );
            }
        }
        eprintln!("Total: {:.1}s", self.total_duration_sec);
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

impl DatasetReport {
    /// Print a Markdown table row for this dataset.
    pub fn print_markdown(&self) {
        eprintln!("### {}", self.name);
        eprintln!("Task: {}", self.task);
        eprintln!(
            "Items: {}, Duration: {}ms",
            self.num_items, self.duration_ms
        );
        if let Some(obj) = self.metrics.as_object() {
            for (k, v) in obj {
                eprintln!("  {}: {}", k, v);
            }
        }
        for b in &self.baselines {
            let arrow = if b.delta >= 0.0 { "+" } else { "" };
            eprintln!(
                "  vs {}: {} {:.3} → {:.3} ({}{:.3})",
                b.method, b.metric, b.baseline_value, b.tensa_value, arrow, b.delta
            );
        }
        eprintln!();
    }
}
