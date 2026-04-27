//! Tests for [`crate::synthetic_scaling`] (EATH Phase 8).
//!
//! Lives in a sibling file (rather than `mod tests {}` inline) so the
//! benchmark module proper stays under the 500-line cap. Brought back into the
//! main module via `#[cfg(test)] #[path = "synthetic_scaling_tests.rs"] mod
//! synthetic_scaling_tests;` at the bottom of the benchmark file.
//!
//! Three categories of tests, in declaration order:
//! 1. Cell-level smoke tests for SMALL cells (run by default).
//! 2. Heavy cells, each `#[ignore]`-gated (run via `cargo test -- --ignored`).
//! 3. The full-matrix entry + Markdown / JSON shape unit tests.

use super::synthetic_scaling::{
    build_report, emit_reports, render_markdown, run_cell, AlgoTiming, CellResult, MatrixSpec,
    ScalingReport, ALGO_LABELS, EATH_PAPER_DOI,
};

// ── 1. Small cells (default-run) ─────────────────────────────────────────────

#[test]
fn test_cell_100_entities_100_steps() {
    let cell = run_cell(100, 100).expect("100×100 cell must complete");
    assert_eq!(cell.entities, 100);
    assert_eq!(cell.steps, 100);
    assert_eq!(cell.algos.len(), ALGO_LABELS.len());
    for (a, expected_label) in cell.algos.iter().zip(ALGO_LABELS.iter()) {
        assert_eq!(&a.algo, expected_label);
        assert!(a.wall_ms.is_finite() && a.wall_ms >= 0.0);
    }
    assert_eq!(cell.params_hash.len(), 64, "params_hash should be sha256 hex");
}

#[test]
fn test_cell_500_entities_100_steps() {
    let cell = run_cell(500, 100).expect("500×100 cell must complete");
    assert_eq!(cell.entities, 500);
    assert_eq!(cell.algos.len(), ALGO_LABELS.len());
}

#[test]
fn test_cell_1000_entities_100_steps() {
    let cell = run_cell(1000, 100).expect("1000×100 cell must complete");
    assert_eq!(cell.entities, 1000);
    assert_eq!(cell.algos.len(), ALGO_LABELS.len());
}

// ── 2. Heavy cells — gated behind --ignored ─────────────────────────────────

#[test]
#[ignore = "heavy: 100×1000 — run via cargo test -- --ignored"]
fn test_cell_100_entities_1000_steps() {
    let cell = run_cell(100, 1000).expect("100×1000 cell must complete");
    assert_eq!(cell.steps, 1000);
}

#[test]
#[ignore = "heavy: 5000×100 — run via cargo test -- --ignored"]
fn test_cell_5000_entities_100_steps() {
    let cell = run_cell(5000, 100).expect("5000×100 cell must complete");
    assert_eq!(cell.entities, 5000);
}

#[test]
#[ignore = "heavy: 1000×1000 — run via cargo test -- --ignored"]
fn test_cell_1000_entities_1000_steps() {
    let cell = run_cell(1000, 1000).expect("1000×1000 cell must complete");
    assert_eq!(cell.entities, 1000);
    assert_eq!(cell.steps, 1000);
}

#[test]
#[ignore = "heavy: 5000×1000 — run via cargo test -- --ignored"]
fn test_cell_5000_entities_1000_steps() {
    let cell = run_cell(5000, 1000).expect("5000×1000 cell must complete");
    assert_eq!(cell.entities, 5000);
    assert_eq!(cell.steps, 1000);
}

// ── 3. Full-matrix entry + shape tests ──────────────────────────────────────

#[test]
#[ignore = "full matrix — run via cargo test -- --ignored"]
fn test_full_synthetic_scaling_matrix() {
    let entity_counts = [100_usize, 500, 1000, 5000];
    let steps_counts = [100_usize, 1000];

    let mut cells = Vec::new();
    for &n_ent in &entity_counts {
        for &n_steps in &steps_counts {
            let cell = run_cell(n_ent, n_steps).expect("cell must complete");
            cells.push(cell);
        }
    }

    let matrix = MatrixSpec {
        entity_counts: entity_counts.to_vec(),
        steps: steps_counts.to_vec(),
    };
    let report = build_report(matrix, cells);
    emit_reports(&report);
}

#[test]
fn test_markdown_table_includes_every_algorithm_column() {
    let report = build_report(
        MatrixSpec {
            entity_counts: vec![10],
            steps: vec![10],
        },
        vec![CellResult {
            entities: 10,
            steps: 10,
            params_hash: "deadbeef".into(),
            generate_ms: 1.0,
            generate_rss_delta_kb: 100,
            algos: ALGO_LABELS
                .iter()
                .map(|l| AlgoTiming {
                    algo: (*l).to_string(),
                    wall_ms: 1.0,
                    rss_delta_kb: 1,
                })
                .collect(),
        }],
    );

    let md = render_markdown(&report);
    for label in ALGO_LABELS {
        assert!(md.contains(label), "Markdown must include column for {label}");
    }
    assert!(md.contains(EATH_PAPER_DOI), "Markdown must cite the EATH paper DOI");
}

#[test]
fn test_sidecar_json_round_trip_preserves_shape() {
    let original = build_report(
        MatrixSpec {
            entity_counts: vec![100, 500],
            steps: vec![100],
        },
        vec![CellResult {
            entities: 100,
            steps: 100,
            params_hash: "abc123".into(),
            generate_ms: 2.5,
            generate_rss_delta_kb: 1024,
            algos: vec![AlgoTiming {
                algo: "pagerank".into(),
                wall_ms: 0.5,
                rss_delta_kb: 32,
            }],
        }],
    );

    let json = serde_json::to_string_pretty(&original).unwrap();
    let parsed: ScalingReport = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.tensa_version, original.tensa_version);
    assert_eq!(parsed.eath_paper_doi, EATH_PAPER_DOI);
    assert_eq!(parsed.matrix.entity_counts, vec![100, 500]);
    assert_eq!(parsed.cells.len(), 1);
    assert_eq!(parsed.cells[0].params_hash, "abc123");
    assert_eq!(parsed.algorithms.len(), ALGO_LABELS.len());
}
