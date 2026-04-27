//! Integration test: run the PAN@CLEF authorship-verification benchmark and
//! assert the overall score clears a non-trivial above-chance threshold.

#[path = "benchmarks/mod.rs"]
mod benchmarks;

use benchmarks::*;

#[path = "benchmarks/pan_verification.rs"]
mod pan_verification;

#[test]
fn pan_verification_beats_chance_on_synthetic_corpus() {
    let report = pan_verification::run();
    report.print_markdown();

    // Print JSON for CI consumption.
    eprintln!("{}", report.to_json());

    // AUC must beat chance on our synthetic three-author corpus.
    let auc = report.metrics.extra["auc"].as_f64().unwrap_or(0.0);
    assert!(
        auc > 0.55,
        "AUC {} did not beat chance (>0.55) on synthetic corpus",
        auc
    );

    // Overall score must be strictly positive (no PAN metric collapsed).
    let overall = report.metrics.extra["overall"].as_f64().unwrap_or(0.0);
    assert!(overall > 0.0, "PAN overall score = {}", overall);
}
