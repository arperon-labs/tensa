//! NIGHTFALL benchmark evaluation entry point.
//!
//! ```bash
//! # Validate ground truth (fast, no external deps)
//! cargo test --no-default-features --test nightfall_eval test_ground_truth_consistency
//!
//! # Run full benchmark (requires running TENSA server + Python deps + API keys)
//! TENSA_ADDR=http://localhost:3000 \
//! OPENAI_API_KEY=sk-... \
//! ANTHROPIC_API_KEY=sk-ant-... \
//! cargo test --no-default-features --test nightfall_eval test_full_benchmark -- --ignored --nocapture
//!
//! # Run single system
//! TENSA_ADDR=http://localhost:3000 \
//! cargo test --no-default-features --test nightfall_eval test_tensa_only -- --ignored --nocapture
//! ```

mod nightfall;

use nightfall::harness::{runner, systems};
use std::path::PathBuf;

fn nightfall_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("nightfall")
}

fn ground_truth_dir() -> PathBuf {
    nightfall_dir().join("ground_truth")
}

// ─── Validation ─────────────────────────────────────────────────

#[test]
fn test_ground_truth_consistency() {
    let dir = ground_truth_dir();
    let report = nightfall::validation::consistency_checker::validate_ground_truth(&dir);
    report.print_summary();
    assert!(
        report.is_valid(),
        "Ground truth has {} consistency errors",
        report.errors.len()
    );
}

// ─── Full Benchmark ─────────────────────────────────────────────

#[test]
#[ignore] // Requires running systems + API keys
fn test_full_benchmark() {
    let base = nightfall_dir();
    let questions = runner::load_questions(&base.join("questions").join("question_templates.json"));
    eprintln!("Loaded {} question templates", questions.len());

    let output_dir = base.join("results");
    std::fs::create_dir_all(&output_dir).ok();

    // ── System 1: TENSA Full ──

    let tensa_url =
        std::env::var("TENSA_ADDR").unwrap_or_else(|_| "http://localhost:4350".to_string());
    let archives_dir = base.join("archives");
    let mut tensa = systems::TensaFull::new(&tensa_url, archives_dir);
    let tensa_output = runner::run_benchmark(&mut tensa, &base, &questions);
    let tensa_json = serde_json::to_string_pretty(&tensa_output).unwrap();
    std::fs::write(output_dir.join("tensa_full.json"), &tensa_json).unwrap();
    eprintln!(
        "[tensa_full] Done: {:.0}s, ${:.2}",
        tensa_output.total_wall_sec, tensa_output.total_cost_usd
    );

    // ── System 2: Vanilla RAG ──

    let vanilla_script = base.join("baselines").join("vanilla_rag.py");
    let vanilla_work = output_dir.join("vanilla_rag_work");
    std::fs::create_dir_all(&vanilla_work).ok();
    let mut vanilla = systems::PythonRagSystem::new("vanilla_rag", vanilla_script, vanilla_work);
    if let Err(e) = vanilla.start() {
        eprintln!("[vanilla_rag] Failed to start: {}", e);
    } else {
        let vanilla_output = runner::run_benchmark(&mut vanilla, &base, &questions);
        let vanilla_json = serde_json::to_string_pretty(&vanilla_output).unwrap();
        std::fs::write(output_dir.join("vanilla_rag.json"), &vanilla_json).unwrap();
        eprintln!(
            "[vanilla_rag] Done: {:.0}s, ${:.2}",
            vanilla_output.total_wall_sec, vanilla_output.total_cost_usd
        );
    }

    // ── System 3: GraphRAG ──

    let graphrag_script = base.join("baselines").join("graphrag_adapter.py");
    let graphrag_work = output_dir.join("graphrag_work");
    std::fs::create_dir_all(&graphrag_work).ok();
    let mut graphrag = systems::PythonRagSystem::new("graphrag", graphrag_script, graphrag_work);
    if let Err(e) = graphrag.start() {
        eprintln!("[graphrag] Failed to start: {}", e);
    } else {
        let graphrag_output = runner::run_benchmark(&mut graphrag, &base, &questions);
        let graphrag_json = serde_json::to_string_pretty(&graphrag_output).unwrap();
        std::fs::write(output_dir.join("graphrag.json"), &graphrag_json).unwrap();
        eprintln!(
            "[graphrag] Done: {:.0}s, ${:.2}",
            graphrag_output.total_wall_sec, graphrag_output.total_cost_usd
        );
    }

    // ── System 4: LightRAG ──

    let lightrag_script = base.join("baselines").join("lightrag_adapter.py");
    let lightrag_work = output_dir.join("lightrag_work");
    std::fs::create_dir_all(&lightrag_work).ok();
    let mut lightrag = systems::PythonRagSystem::new("lightrag", lightrag_script, lightrag_work);
    if let Err(e) = lightrag.start() {
        eprintln!("[lightrag] Failed to start: {}", e);
    } else {
        let lightrag_output = runner::run_benchmark(&mut lightrag, &base, &questions);
        let lightrag_json = serde_json::to_string_pretty(&lightrag_output).unwrap();
        std::fs::write(output_dir.join("lightrag.json"), &lightrag_json).unwrap();
        eprintln!(
            "[lightrag] Done: {:.0}s, ${:.2}",
            lightrag_output.total_wall_sec, lightrag_output.total_cost_usd
        );
    }

    eprintln!(
        "\n=== All systems complete. Results in {:?} ===",
        output_dir
    );
}

// ─── Single-System Tests ────────────────────────────────────────

#[test]
#[ignore]
fn test_tensa_only() {
    let base = nightfall_dir();
    let questions = runner::load_questions(&base.join("questions").join("question_templates.json"));

    let output_dir = base.join("results");
    std::fs::create_dir_all(&output_dir).ok();

    // Timestamped run ID so each run is preserved and not overwritten.
    // Include stage range if set (e.g. for resumed runs).
    let stage_start = std::env::var("NIGHTFALL_STAGE_START").unwrap_or_else(|_| "1".to_string());
    let stage_end = std::env::var("NIGHTFALL_STAGE_END").unwrap_or_else(|_| "12".to_string());
    let range_tag = if stage_start == "1" && stage_end == "12" {
        String::new()
    } else {
        format!("_s{}-{}", stage_start, stage_end)
    };
    let run_id = format!(
        "{}{}",
        chrono::Utc::now().format("%Y%m%d_%H%M%S"),
        range_tag
    );
    eprintln!("[tensa_full] Run ID: {}", run_id);

    let tensa_url =
        std::env::var("TENSA_ADDR").unwrap_or_else(|_| "http://localhost:4350".to_string());
    let archives_dir = base.join("archives");
    let mut tensa = systems::TensaFull::new(&tensa_url, archives_dir);
    let output = runner::run_benchmark(&mut tensa, &base, &questions);
    let json = serde_json::to_string_pretty(&output).unwrap();

    // Write two files: timestamped snapshot + convenience symlink "latest"
    let timestamped = output_dir.join(format!("tensa_full_{}.json", run_id));
    std::fs::write(&timestamped, &json).unwrap();
    eprintln!("[tensa_full] Results saved to: {}", timestamped.display());

    // Also write to tensa_full_latest.json for convenience
    let latest = output_dir.join("tensa_full_latest.json");
    std::fs::write(&latest, &json).unwrap();

    nightfall::harness::report::print_markdown(
        &nightfall::harness::metrics::NightfallMetrics::compute(
            "tensa_full",
            vec![], // placeholder — real scoring needs answer key comparison
            vec![],
        ),
    );
}

#[test]
#[ignore]
fn test_vanilla_rag_only() {
    let base = nightfall_dir();
    let questions = runner::load_questions(&base.join("questions").join("question_templates.json"));

    let output_dir = base.join("results");
    std::fs::create_dir_all(&output_dir).ok();

    let script = base.join("baselines").join("vanilla_rag.py");
    let work = output_dir.join("vanilla_rag_work");
    std::fs::create_dir_all(&work).ok();

    let mut system = systems::PythonRagSystem::new("vanilla_rag", script, work);
    system.start().expect("Failed to start vanilla_rag.py");
    let output = runner::run_benchmark(&mut system, &base, &questions);
    let json = serde_json::to_string_pretty(&output).unwrap();
    std::fs::write(output_dir.join("vanilla_rag.json"), &json).unwrap();
    eprintln!(
        "[vanilla_rag] Done: {:.0}s, ${:.2}",
        output.total_wall_sec, output.total_cost_usd
    );
}
