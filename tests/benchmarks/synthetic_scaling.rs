//! Synthetic scaling benchmark — wall-clock + RSS across (entities × steps)
//! cells (EATH Phase 8).
//!
//! Runs every analysis-stack algorithm on EATH-generated synthetic narratives
//! at four entity counts (100, 500, 1000, 5000) crossed with two step counts
//! (100, 1000). For each cell the benchmark:
//!
//! 1. Builds an EathSurrogate with hand-crafted, deterministic params (no
//!    source-narrative calibration — that costs O(P) and would dominate small
//!    cells, distorting the reported wall-clock).
//! 2. Samples RSS BEFORE generation (true baseline for the process).
//! 3. Generates the synthetic narrative.
//! 4. Samples RSS AFTER generation — this is the post-generate baseline that
//!    every algorithm RSS sample is reported relative to.
//! 5. Times each algorithm and samples RSS after each. The reported
//!    "rss_delta_kb" is `post_algo_kb - post_generate_kb` — the algorithm's
//!    working-set overhead.
//! 6. Emits a row to a Markdown table.
//!
//! Cells with `entities > 1000 OR steps > 1000` are marked `#[ignore]` so the
//! default `cargo test` run pays only the cheap-cell cost. Run the heavy
//! cells with `cargo test -- --ignored`.
//!
//! ## Memory measurement
//!
//! `memory-stats` is the only cross-platform crate that reports physical RSS
//! without requiring a custom allocator. jemalloc-ctl would only populate on
//! Linux; reports would be non-comparable between Windows-dev and Linux-CI
//! runs. memory-stats is less precise (process-level granularity) but
//! consistent across platforms.
//!
//! ## Zenodo metadata sidecar
//!
//! `synthetic_scaling_report.json` is emitted alongside the Markdown report
//! when `TENSA_BENCH_REPORT_DIR` is set. The sidecar carries the EATH paper
//! DOI, TENSA version, git SHA (best-effort), platform, rustc version,
//! benchmark matrix, and per-cell results with the canonical `params_hash`
//! so any cell can be reproduced bit-for-bit.

use std::sync::Arc;
use std::time::Instant;

use memory_stats::memory_stats;
use serde::{Deserialize, Serialize};

use tensa::analysis::community_detect::label_propagation;
use tensa::analysis::contagion;
use tensa::analysis::graph_centrality::compute_pagerank;
use tensa::analysis::graph_projection::build_co_graph;
use tensa::analysis::higher_order_contagion::{
    simulate_higher_order_sir, HigherOrderSirParams, SeedStrategy, ThresholdRule,
};
use tensa::analysis::temporal_motifs::temporal_motif_census;
use tensa::analysis::topology::compute_kcore;
use tensa::error::Result;
use tensa::hypergraph::Hypergraph;
use tensa::narrative::pattern::mine_patterns;
use tensa::narrative::subgraph::NarrativeGraph;
use tensa::store::memory::MemoryStore;
use tensa::store::KVStore;
use tensa::synth::eath::EathSurrogate;
use tensa::synth::hashing::canonical_params_hash;
use tensa::synth::{EathParams, SurrogateModel, SurrogateParams};

// ── Constants ────────────────────────────────────────────────────────────────

/// Paper DOI for the EATH model — embedded in the Zenodo sidecar so the
/// dataset is independently reproducible from the cited paper alone. `pub`
/// because the shape tests in `synthetic_scaling_tests.rs` assert it appears
/// in the rendered Markdown.
pub const EATH_PAPER_DOI: &str = "10.48550/arXiv.2507.01124";

/// Fixed seed mixing constant. Applied per-cell as `seed_for_cell(entities,
/// steps) = entities as u64 * MIX + steps as u64`. Same mixing every run ⇒
/// every cell is bit-for-bit reproducible.
const CELL_SEED_MIX: u64 = 0x9e37_79b9_7f4a_7c15;

/// Algorithm column labels in the Markdown table. Same order as
/// `time_all_algorithms` populates them — keep in sync. `pub` because the
/// smoke tests in the sibling tests file assert one row per label.
pub const ALGO_LABELS: &[&str] = &[
    "pagerank",
    "leiden_or_lp",
    "kcore",
    "label_propagation",
    "temporal_motifs_3node",
    "pattern_mining_top5",
    "pairwise_sir",
    "higher_order_sir",
];

// ── Per-algorithm timing struct ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgoTiming {
    pub algo: String,
    pub wall_ms: f64,
    /// post_algo_kb - post_generate_kb. Negative deltas are clamped to 0
    /// (process-level RSS sampling can race with the OS reclaim).
    pub rss_delta_kb: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellResult {
    pub entities: usize,
    pub steps: usize,
    /// Canonical params hash — embed in the sidecar so the cell is
    /// reproducible bit-for-bit from the published metadata alone.
    pub params_hash: String,
    /// Wall-clock for the EATH generation step (separate from algorithm timings).
    pub generate_ms: f64,
    /// post_generate_kb - pre_generate_kb. The "loaded narrative" overhead.
    pub generate_rss_delta_kb: i64,
    pub algos: Vec<AlgoTiming>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingReport {
    pub tensa_version: String,
    pub git_sha: Option<String>,
    pub platform: String,
    pub rustc_version: String,
    pub eath_paper_doi: String,
    pub matrix: MatrixSpec,
    pub algorithms: Vec<String>,
    pub cells: Vec<CellResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixSpec {
    pub entity_counts: Vec<usize>,
    pub steps: Vec<usize>,
}

// ── EATH params construction ─────────────────────────────────────────────────

/// Hand-crafted, deterministic EATH params for benchmark cells. Avoids the
/// O(P) calibration cost from a source narrative — every cell at the same
/// (entities, steps) gets identical params, guaranteeing reproducibility.
fn bench_eath_params(entities: usize) -> EathParams {
    EathParams {
        // Uniform 0.5 activation rate per actor.
        a_t_distribution: vec![0.5_f32; entities],
        // Uniform hyperactivity weights.
        a_h_distribution: vec![1.0_f32; entities],
        // Empty schedule → engine uses uniform Λ_t internally.
        lambda_schedule: vec![],
        p_from_scratch: 0.5,
        omega_decay: 0.95,
        // Mostly dyads, some triads, occasional tetrads.
        group_size_distribution: vec![6, 3, 1],
        rho_low: 0.1,
        rho_high: 0.5,
        xi: 1.0,
        order_propensity: vec![],
        max_group_size: 4,
        stm_capacity: 7,
        num_entities: entities,
    }
}

fn cell_seed(entities: usize, steps: usize) -> u64 {
    (entities as u64).wrapping_mul(CELL_SEED_MIX) ^ (steps as u64)
}

fn build_surrogate_params(entities: usize, steps: usize) -> SurrogateParams {
    SurrogateParams {
        model: "eath".into(),
        params_json: serde_json::to_value(bench_eath_params(entities)).unwrap(),
        seed: cell_seed(entities, steps),
        num_steps: steps,
        label_prefix: "bench".into(),
    }
}

fn fresh_hypergraph() -> Arc<Hypergraph> {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Arc::new(Hypergraph::new(store))
}

// ── RSS sampling ─────────────────────────────────────────────────────────────

/// Sample physical RSS in KB. Returns 0 if the platform refuses to report
/// (memory-stats is best-effort) — downstream delta computation handles the
/// zero case gracefully.
fn sample_rss_kb() -> i64 {
    memory_stats()
        .map(|s| (s.physical_mem / 1024) as i64)
        .unwrap_or(0)
}

/// Compute working-set delta `post - pre` in KB. Negative values clamp to 0
/// (the OS may have reclaimed pages between samples — not a real "savings").
fn rss_delta(pre_kb: i64, post_kb: i64) -> i64 {
    (post_kb - pre_kb).max(0)
}

// ── Per-cell execution ───────────────────────────────────────────────────────

/// Generate one cell's worth of synthetic narrative + time every algorithm
/// against it. Returns the populated `CellResult`.
pub fn run_cell(entities: usize, steps: usize) -> Result<CellResult> {
    let params = build_surrogate_params(entities, steps);
    let params_hash = canonical_params_hash(&params);
    let nid = format!("bench-{}-{}", entities, steps);

    let hg = fresh_hypergraph();
    let surrogate = EathSurrogate;

    // 1. RSS baseline BEFORE generation.
    let pre_kb = sample_rss_kb();

    // 2. Generation timing.
    let gen_start = Instant::now();
    surrogate.generate(&params, &hg, &nid)?;
    let generate_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

    // 3. RSS baseline AFTER generation — every algorithm's RSS delta is
    //    relative to this point, so the "loaded narrative" cost is excluded
    //    from each algorithm's working-set overhead.
    let post_gen_kb = sample_rss_kb();
    let generate_rss_delta_kb = rss_delta(pre_kb, post_gen_kb);

    // 4. Time every algorithm.
    let algos = time_all_algorithms(&hg, &nid, post_gen_kb)?;

    Ok(CellResult {
        entities,
        steps,
        params_hash,
        generate_ms,
        generate_rss_delta_kb,
        algos,
    })
}

/// Run every benchmarked algorithm on the populated narrative once. The
/// co-graph is built once on the first row (PageRank) so that build cost is
/// reported in the table — never silently amortized across rows.
///
/// Note on the `leiden_or_lp` slot: Sprint 2's
/// `analysis::community_detect::label_propagation` is the parameter-free O(m)
/// community algorithm; the heavyweight `centrality::leiden` lives behind a
/// separate engine. We run LP here under the legacy "leiden_or_lp" label so
/// the original spec's algorithm slot stays self-documenting.
fn time_all_algorithms(
    hg: &Hypergraph,
    narrative_id: &str,
    baseline_rss_kb: i64,
) -> Result<Vec<AlgoTiming>> {
    let mut timings: Vec<AlgoTiming> = Vec::with_capacity(ALGO_LABELS.len());
    let mut record = |label: &str, start: Instant| {
        timings.push(AlgoTiming {
            algo: label.into(),
            wall_ms: start.elapsed().as_secs_f64() * 1000.0,
            rss_delta_kb: rss_delta(baseline_rss_kb, sample_rss_kb()),
        });
    };

    // 1. PageRank — includes co-graph build cost.
    let start = Instant::now();
    let cograph = build_co_graph(hg, narrative_id)?;
    let _ = compute_pagerank(&cograph);
    record("pagerank", start);

    // 2. Leiden-slot (label_propagation under the legacy column label).
    let start = Instant::now();
    let _ = label_propagation(&cograph, 0xC0FFEE);
    record("leiden_or_lp", start);

    // 3. K-Core decomposition.
    let start = Instant::now();
    let _ = compute_kcore(&cograph);
    record("kcore", start);

    // 4. Label propagation (second run, different seed — exercises the LP path
    //    independently of the leiden-slot row above).
    let start = Instant::now();
    let _ = label_propagation(&cograph, 0xBADCAFE);
    record("label_propagation", start);

    // 5. Temporal motif census (3-node).
    let start = Instant::now();
    let _ = temporal_motif_census(hg, narrative_id, 3)?;
    record("temporal_motifs_3node", start);

    // 6. Pattern mining (top-5). min_support=1 since the benchmark passes a
    //    single narrative; sort + truncate is charged to this row.
    let start = Instant::now();
    let narr_graph = NarrativeGraph::extract(narrative_id, hg)?;
    let mut patterns = mine_patterns(std::slice::from_ref(&narr_graph), 1);
    patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
    patterns.truncate(5);
    record("pattern_mining_top5", start);

    // 7. Pairwise SIR — seed on first entity by canonical UUID order.
    let start = Instant::now();
    if let Some(first) = hg.list_entities_by_narrative(narrative_id)?.first() {
        let _ = contagion::run_contagion(hg, narrative_id, "bench-fact", first.id)?;
    }
    record("pairwise_sir", start);

    // 8. Higher-order SIR (Phase 7b). ThresholdRule::Absolute(1) reduces to
    //    pairwise; sized betas + 5% RandomFraction seed keep the parameters
    //    inside the regime where the higher-order pathway actually fires.
    let ho_params = HigherOrderSirParams {
        beta_per_size: vec![0.3_f32, 0.5_f32, 0.7_f32],
        gamma: 0.1_f32,
        threshold: ThresholdRule::Absolute(1),
        seed_strategy: SeedStrategy::RandomFraction { fraction: 0.05 },
        max_steps: 50,
        rng_seed: 0xDEADBEEF,
    };
    let start = Instant::now();
    let _ = simulate_higher_order_sir(hg, narrative_id, &ho_params)?;
    record("higher_order_sir", start);

    Ok(timings)
}

// ── Markdown rendering ───────────────────────────────────────────────────────

pub fn render_markdown(report: &ScalingReport) -> String {
    let mut out = String::new();
    out.push_str("# Synthetic Scaling Benchmark — EATH Phase 8\n\n");
    out.push_str(&format!("**TENSA version:** `{}`\n", report.tensa_version));
    if let Some(sha) = &report.git_sha {
        out.push_str(&format!("**Git SHA:** `{}`\n", sha));
    }
    out.push_str(&format!("**Platform:** `{}`\n", report.platform));
    out.push_str(&format!("**rustc:** `{}`\n", report.rustc_version));
    out.push_str(&format!(
        "**EATH paper:** [{}](https://doi.org/{})\n\n",
        report.eath_paper_doi, report.eath_paper_doi
    ));

    // Header row: entities × steps + per-algorithm wall_ms columns.
    out.push_str("| entities | steps | gen_ms | gen_rss_kb |");
    for algo in &report.algorithms {
        out.push_str(&format!(" {algo} (ms / Δkb) |"));
    }
    out.push('\n');
    out.push_str("|---|---|---|---|");
    for _ in &report.algorithms {
        out.push_str("---|");
    }
    out.push('\n');

    for cell in &report.cells {
        out.push_str(&format!(
            "| {} | {} | {:.1} | {} |",
            cell.entities, cell.steps, cell.generate_ms, cell.generate_rss_delta_kb
        ));
        // Match algo timings to header order. If a row's algorithm vector
        // doesn't carry a label that's in the header, leave the column blank
        // (the table stays aligned even when a future algorithm gets retired).
        for algo_label in &report.algorithms {
            let cell_algo = cell.algos.iter().find(|a| &a.algo == algo_label);
            match cell_algo {
                Some(a) => {
                    out.push_str(&format!(" {:.1} / {} |", a.wall_ms, a.rss_delta_kb));
                }
                None => out.push_str(" — |"),
            }
        }
        out.push('\n');
    }

    out
}

// ── Sidecar JSON construction ────────────────────────────────────────────────

pub fn build_report(matrix: MatrixSpec, cells: Vec<CellResult>) -> ScalingReport {
    ScalingReport {
        tensa_version: env!("CARGO_PKG_VERSION").to_string(),
        git_sha: option_env!("TENSA_GIT_HASH").map(str::to_string),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        rustc_version: option_env!("RUSTC_VERSION")
            .unwrap_or("unknown")
            .to_string(),
        eath_paper_doi: EATH_PAPER_DOI.to_string(),
        matrix,
        algorithms: ALGO_LABELS.iter().map(|s| s.to_string()).collect(),
        cells,
    }
}

/// If `TENSA_BENCH_REPORT_DIR` is set and writable, write both the Markdown
/// and JSON report files to it. Always echoes the Markdown to stderr (visible
/// via `cargo test -- --nocapture`).
pub fn emit_reports(report: &ScalingReport) {
    let md = render_markdown(report);
    eprintln!("\n{md}");

    if let Ok(dir) = std::env::var("TENSA_BENCH_REPORT_DIR") {
        let path_md = std::path::Path::new(&dir).join("synthetic_scaling_report.md");
        let path_json = std::path::Path::new(&dir).join("synthetic_scaling_report.json");
        if let Err(e) = std::fs::write(&path_md, &md) {
            eprintln!("warning: failed to write Markdown report to {path_md:?}: {e}");
        }
        match serde_json::to_string_pretty(report) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path_json, json) {
                    eprintln!("warning: failed to write JSON sidecar to {path_json:?}: {e}");
                }
            }
            Err(e) => eprintln!("warning: failed to serialize JSON sidecar: {e}"),
        }
    }
}

// ── Public benchmark entry point (called from benchmark_suite.rs) ────────────

/// Run the SMALL benchmark matrix (cells where entities ≤ 1000 AND steps ≤ 100).
/// Larger cells live behind `#[ignore]`-marked tests and are run via
/// `cargo test -- --ignored`.
pub fn run() -> Result<()> {
    let entity_counts = [100_usize, 500, 1000];
    let steps_counts = [100_usize];

    let mut cells = Vec::new();
    for &n_ent in &entity_counts {
        for &n_steps in &steps_counts {
            let cell = run_cell(n_ent, n_steps)?;
            cells.push(cell);
        }
    }

    let matrix = MatrixSpec {
        entity_counts: entity_counts.to_vec(),
        steps: steps_counts.to_vec(),
    };
    let report = build_report(matrix, cells);
    emit_reports(&report);
    Ok(())
}

// Tests live in the sibling `synthetic_scaling_tests.rs` file (kept separate
// to keep this module under the 500-line cap). `benchmark_suite.rs` declares
// it as a sibling module with `#[path = "benchmarks/synthetic_scaling_tests.rs"]`.
