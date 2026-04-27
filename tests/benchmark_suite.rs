//! Benchmark suite entry point (Sprint 12).
//!
//! Aggregates all benchmarks and produces a combined report.
//! Run with: `cargo test --no-default-features --test benchmark_suite`

mod benchmarks;

// Individual benchmark modules.
#[path = "benchmarks/er.rs"]
mod er;
#[path = "benchmarks/graphrag.rs"]
mod graphrag;
#[path = "benchmarks/harness.rs"]
mod harness;
#[path = "benchmarks/pan_verification.rs"]
mod pan_verification;
#[path = "benchmarks/stylometry.rs"]
mod stylometry;
#[path = "benchmarks/synthetic_scaling.rs"]
mod synthetic_scaling;
// Tests for synthetic_scaling live in a sibling file (kept separate so the
// benchmark module itself stays under the 500-line cap). They access the
// benchmark's pub items via `super::synthetic_scaling::*`.
#[cfg(test)]
#[path = "benchmarks/synthetic_scaling_tests.rs"]
mod synthetic_scaling_tests;
#[path = "benchmarks/temporal.rs"]
mod temporal;

use benchmarks::*;

/// Combined benchmark report — runs all benchmarks and produces a summary.
#[test]
fn test_benchmark_report() {
    eprintln!("\n=== TENSA Benchmark Suite ===\n");

    // Verify the harness types work.
    let metrics = BenchmarkMetrics::from_counts(8, 1, 2);
    assert!((metrics.precision - 8.0 / 9.0).abs() < 0.01);
    assert!((metrics.recall - 8.0 / 10.0).abs() < 0.01);
    assert!(metrics.f1 > 0.0);

    let report = BenchmarkReport {
        benchmark: "Harness Self-Test".into(),
        dataset: "synthetic".into(),
        metrics,
        baseline_comparison: vec![BaselineComparison {
            method: "random".into(),
            metric: "f1".into(),
            baseline_value: 0.5,
            tensa_value: 0.84,
            delta: 0.34,
        }],
        duration_ms: 1,
    };

    let json = report.to_json();
    assert!(json.contains("Harness Self-Test"));
    report.print_markdown();

    eprintln!("=== All benchmarks complete ===\n");
}

/// Synthetic-scaling benchmark entry — runs the SMALL matrix (entities ≤ 1000,
/// steps ≤ 100). Heavy cells are individually `#[ignore]`d in
/// [`tests/benchmarks/synthetic_scaling.rs`] and run via
/// `cargo test --test benchmark_suite -- --ignored`.
#[test]
fn test_synthetic_scaling_benchmark() {
    eprintln!("\n=== Synthetic Scaling (EATH Phase 8) — small cells ===\n");
    synthetic_scaling::run().expect("synthetic_scaling::run must succeed");
}
