//! TENSA × ChronoSense Evaluation Protocol (v0.1)
//!
//! Implements the TimeBlind-style metric hierarchy for Allen temporal reasoning.
//! Tests TENSA's structural Allen-relation extractor against baselines.
//!
//! Run with: `cargo test --no-default-features --test chronosense_eval -- --nocapture`
//! Run all (including full evaluation): `cargo test --no-default-features --test chronosense_eval -- --nocapture --ignored`
//!
//! LLM baseline (requires API key):
//!   OPENROUTER_API_KEY=sk-or-... cargo test --no-default-features --test chronosense_eval test_llm_baseline -- --ignored --nocapture
//!   ANTHROPIC_API_KEY=sk-ant-... TENSA_MODEL=claude-sonnet-4-20250514 cargo test --no-default-features --test chronosense_eval test_llm_baseline -- --ignored --nocapture
//!
//! Export prompts for offline evaluation:
//!   cargo test --no-default-features --test chronosense_eval test_export_prompts -- --ignored --nocapture

mod chronosense;

use chronosense::*;

// ─── Unit tests (fast, always run) ───────────────────────────

#[test]
fn test_dataset_generation_valid() {
    let config = EvalConfig {
        items_per_relation: 10,
        ..Default::default()
    };
    let items = dataset::generate_abstract_split(&config);
    assert_eq!(items.len(), 13 * 10);

    // Every generated item must realize its ground-truth relation
    for item in &items {
        let computed =
            tensa::temporal::interval::relation_between(&item.interval_a, &item.interval_b)
                .unwrap();
        assert_eq!(
            computed, item.ground_truth,
            "Item {} expected {:?}, got {:?}",
            item.id, item.ground_truth, computed
        );
    }
}

#[test]
fn test_minimal_pairs_valid() {
    let config = EvalConfig {
        items_per_relation: 10,
        ..Default::default()
    };
    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);

    assert!(!pairs.is_empty());
    for pair in &pairs {
        // c1 realizes R, c2 realizes R'
        let r1 = tensa::temporal::interval::relation_between(
            &pair.context_1.interval_a,
            &pair.context_1.interval_b,
        )
        .unwrap();
        let r2 = tensa::temporal::interval::relation_between(
            &pair.context_2.interval_a,
            &pair.context_2.interval_b,
        )
        .unwrap();

        assert_eq!(r1, pair.hypothesis_1.relation);
        assert_eq!(r2, pair.hypothesis_2.relation);
        assert_ne!(
            r1, r2,
            "Minimal pair contexts must realize different relations"
        );
    }
}

#[test]
fn test_tensa_analyzer_perfect() {
    let config = EvalConfig {
        items_per_relation: 15,
        ..Default::default()
    };
    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);
    let results = analyzer::tensa_evaluate_all(&pairs);
    let m = metrics::compute_metrics(&results);

    assert!(
        (m.i_acc - 1.0).abs() < 1e-9,
        "TENSA I-Acc on structured input should be 1.0, got {:.4}",
        m.i_acc
    );
}

#[test]
fn test_baselines_below_tensa() {
    let config = EvalConfig {
        items_per_relation: 10,
        ..Default::default()
    };
    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);

    let tensa_m = metrics::compute_metrics(&analyzer::tensa_evaluate_all(&pairs));
    let random_m = metrics::compute_metrics(&baselines::random_evaluate(&pairs, 99));
    let true_m = metrics::compute_metrics(&baselines::all_true_evaluate(&pairs));
    let tbias_m = metrics::compute_metrics(&baselines::t_bias_evaluate(&pairs));

    assert!(tensa_m.i_acc > random_m.i_acc);
    assert!(tensa_m.i_acc > true_m.i_acc);
    assert!(tensa_m.i_acc > tbias_m.i_acc);
}

#[test]
fn test_composition_table_valid() {
    let (verified, failures, total) = analyzer::validate_composition_table();
    assert_eq!(
        failures, 0,
        "Composition table: {verified}/{total} verified, {failures} failures"
    );
}

#[test]
fn test_shortcut_controls() {
    let config = EvalConfig {
        items_per_relation: 10,
        ..Default::default()
    };
    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);
    let analysis = controls::analyze_shortcuts(&pairs, 1.0);

    // TENSA: no name reliance, high interval reliance
    assert!(
        analysis.name_reliance.abs() < 1e-9,
        "Name reliance should be 0"
    );
    assert!(
        analysis.interval_reliance > 0.9,
        "Interval reliance should be high"
    );
}

// ─── Full evaluation (slower, use --ignored) ─────────────────

#[test]
#[ignore]
fn test_full_chronosense_evaluation() {
    let config = EvalConfig {
        items_per_relation: 50,
        ..Default::default()
    };

    eprintln!("\n=== TENSA × ChronoSense Full Evaluation ===\n");

    // Generate datasets
    let abstract_items = dataset::generate_abstract_split(&config);
    let real_items = dataset::generate_real_split(&config);
    eprintln!(
        "Generated: {} abstract + {} real items",
        abstract_items.len(),
        real_items.len()
    );

    // Generate minimal pairs
    let abstract_pairs = neighbors::generate_minimal_pairs(&abstract_items, config.delta_seconds);
    let real_pairs = neighbors::generate_minimal_pairs(&real_items, config.delta_seconds);
    eprintln!(
        "Minimal pairs: {} abstract ({:.1}% coverage), {} real ({:.1}% coverage)",
        abstract_pairs.len(),
        abstract_pairs.len() as f64 / abstract_items.len() as f64 * 100.0,
        real_pairs.len(),
        real_pairs.len() as f64 / real_items.len() as f64 * 100.0,
    );

    // Evaluate TENSA analyzer on both splits
    let tensa_abstract = metrics::compute_metrics(&analyzer::tensa_evaluate_all(&abstract_pairs));
    let tensa_real = metrics::compute_metrics(&analyzer::tensa_evaluate_all(&real_pairs));
    let real_abstract_gap = metrics::memorization_gap(&tensa_real, &tensa_abstract);

    // Evaluate baselines on abstract split
    let random_m = metrics::compute_metrics(&baselines::random_evaluate(&abstract_pairs, 42));
    let true_m = metrics::compute_metrics(&baselines::all_true_evaluate(&abstract_pairs));
    let false_m = metrics::compute_metrics(&baselines::all_false_evaluate(&abstract_pairs));
    let tbias_m = metrics::compute_metrics(&baselines::t_bias_evaluate(&abstract_pairs));

    // Shortcut analysis
    let shortcut = controls::analyze_shortcuts(&abstract_pairs, tensa_abstract.i_acc);

    // Composition table validation
    let (comp_verified, comp_failures, comp_total) = analyzer::validate_composition_table();

    // Build report rows
    let rows = vec![
        report::ModelRow::from_metrics(
            "TENSA-analyzer (abstract)",
            &tensa_abstract,
            Some(real_abstract_gap),
        ),
        report::ModelRow::from_metrics("TENSA-analyzer (real)", &tensa_real, None),
        report::ModelRow::from_metrics("Random (p=0.5)", &random_m, None),
        report::ModelRow::from_metrics("All-True", &true_m, None),
        report::ModelRow::from_metrics("All-False", &false_m, None),
        report::ModelRow::from_metrics("T-bias", &tbias_m, None),
    ];

    // Build and print full report
    let full_report = report::build_report(
        &config,
        abstract_items.len(),
        abstract_pairs.len(),
        rows,
        &tensa_abstract,
        Some(&shortcut),
        Some((comp_verified, comp_failures, comp_total)),
    );

    report::print_full_report(&full_report, &tensa_abstract);

    // Output JSON report
    let json = serde_json::to_string_pretty(&full_report).unwrap();
    eprintln!("--- JSON Report ---");
    eprintln!("{json}");
    eprintln!("--- End Report ---\n");

    // Assertions
    assert!(
        (tensa_abstract.i_acc - 1.0).abs() < 1e-9,
        "TENSA abstract I-Acc should be 1.0"
    );
    assert!(
        (tensa_real.i_acc - 1.0).abs() < 1e-9,
        "TENSA real I-Acc should be 1.0"
    );
    assert!(
        real_abstract_gap.abs() < 1e-9,
        "Real-Abstract gap should be 0 for TENSA"
    );
    assert_eq!(comp_failures, 0, "No composition failures");
}

// ─── LLM baseline evaluation (requires API key) ─────────────

#[test]
#[ignore]
fn test_llm_baseline() {
    let config = EvalConfig {
        items_per_relation: 10, // Small for API cost control; increase for paper
        ..Default::default()
    };

    let llm_config = match llm_baseline::detect_llm_auto(false) {
        Some(c) => c,
        None => {
            eprintln!("Skipping LLM baseline: no API key found.");
            eprintln!("Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or LOCAL_LLM_URL.");
            return;
        }
    };

    eprintln!("\n=== ChronoSense LLM Baseline ===");
    eprintln!("Model: {} (CoT: {})", llm_config.model, llm_config.use_cot);

    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);
    eprintln!("Evaluating {} minimal pairs...", pairs.len());

    // Run LLM evaluation (async via tokio)
    let rt = tokio::runtime::Runtime::new().unwrap();
    let llm_results = rt.block_on(llm_baseline::llm_evaluate_all(&pairs, &llm_config));
    let llm_m = metrics::compute_metrics(&llm_results);

    // Compare against TENSA
    let tensa_results = analyzer::tensa_evaluate_all(&pairs);
    let tensa_m = metrics::compute_metrics(&tensa_results);

    let rows = vec![
        report::ModelRow::from_metrics("TENSA-analyzer", &tensa_m, None),
        report::ModelRow::from_metrics(&format!("{} (zero-shot)", llm_config.model), &llm_m, None),
    ];

    eprintln!();
    report::print_summary_table(&rows);
    report::print_per_relation_table(&llm_config.model, &llm_m);

    // TENSA should beat the LLM
    eprintln!(
        "TENSA I-Acc: {:.3} vs LLM I-Acc: {:.3} (delta: {:.3})",
        tensa_m.i_acc,
        llm_m.i_acc,
        tensa_m.i_acc - llm_m.i_acc
    );
}

#[test]
#[ignore]
fn test_llm_baseline_cot() {
    let config = EvalConfig {
        items_per_relation: 10,
        ..Default::default()
    };

    let llm_config = match llm_baseline::detect_llm_auto(true) {
        Some(c) => c,
        None => {
            eprintln!("Skipping LLM CoT baseline: no API key found.");
            return;
        }
    };

    eprintln!("\n=== ChronoSense LLM Baseline (Chain-of-Thought) ===");
    eprintln!("Model: {} + CoT", llm_config.model);

    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);
    eprintln!("Evaluating {} minimal pairs...", pairs.len());

    let rt = tokio::runtime::Runtime::new().unwrap();
    let llm_results = rt.block_on(llm_baseline::llm_evaluate_all(&pairs, &llm_config));
    let llm_m = metrics::compute_metrics(&llm_results);

    let tensa_m = metrics::compute_metrics(&analyzer::tensa_evaluate_all(&pairs));

    let rows = vec![
        report::ModelRow::from_metrics("TENSA-analyzer", &tensa_m, None),
        report::ModelRow::from_metrics(&format!("{} (CoT)", llm_config.model), &llm_m, None),
    ];

    eprintln!();
    report::print_summary_table(&rows);
    report::print_per_relation_table(&format!("{} CoT", llm_config.model), &llm_m);
}

// ─── Real ChronoSense dataset evaluation ─────────────────────

#[test]
fn test_real_chronosense_dataset() {
    let data_dir = std::path::Path::new("tests/chronosense/data");
    if !data_dir.join("Before_real.json").exists() {
        eprintln!("Skipping: ChronoSense data not downloaded to tests/chronosense/data/");
        return;
    }

    eprintln!("\n=== Real ChronoSense Dataset (Islakoglu & Kalo, ACL 2025) ===\n");

    let mut total_correct = 0usize;
    let mut total_incorrect = 0usize;
    let mut total_unparseable = 0usize;
    let mut total_items = 0usize;

    eprintln!(
        "| {:<16} | {:>7} | {:>7} | {:>7} | {:>6} |",
        "Relation", "Correct", "Wrong", "Unparse", "Acc"
    );
    eprintln!("|{:-<18}|{:-<9}|{:-<9}|{:-<9}|{:-<8}|", "", "", "", "", "");

    // Real-Allen split
    let relations = chronosense::chronosense_data::load_all_relations(data_dir, false);
    for (rel_name, items) in &relations {
        if items.is_empty() {
            continue;
        }
        let (correct, incorrect, unparseable) =
            chronosense::chronosense_data::evaluate_tensa(items);
        let total = correct + incorrect;
        let acc = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        eprintln!(
            "| {:<16} | {:>7} | {:>7} | {:>7} | {:.3}  |",
            rel_name, correct, incorrect, unparseable, acc
        );
        total_correct += correct;
        total_incorrect += incorrect;
        total_unparseable += unparseable;
        total_items += items.len();
    }

    let grand_total = total_correct + total_incorrect;
    let grand_acc = if grand_total > 0 {
        total_correct as f64 / grand_total as f64
    } else {
        0.0
    };
    eprintln!("|{:-<18}|{:-<9}|{:-<9}|{:-<9}|{:-<8}|", "", "", "", "", "");
    eprintln!(
        "| {:<16} | {:>7} | {:>7} | {:>7} | {:.3}  |",
        "TOTAL (real)", total_correct, total_incorrect, total_unparseable, grand_acc
    );
    eprintln!(
        "\nTotal items: {total_items}, Parsed: {grand_total}, Unparseable: {total_unparseable}"
    );

    // Abstract-Allen split
    eprintln!("\n--- Abstract-Allen split ---");
    let abstract_relations = chronosense::chronosense_data::load_all_relations(data_dir, true);
    let mut abs_correct = 0usize;
    let mut abs_incorrect = 0usize;
    let mut abs_unparseable = 0usize;
    for (rel_name, items) in &abstract_relations {
        if items.is_empty() {
            continue;
        }
        let (c, inc, unp) = chronosense::chronosense_data::evaluate_tensa(items);
        let t = c + inc;
        let acc = if t > 0 { c as f64 / t as f64 } else { 0.0 };
        eprintln!("  {:<16}: {c}/{t} ({acc:.3}), {unp} unparseable", rel_name);
        abs_correct += c;
        abs_incorrect += inc;
        abs_unparseable += unp;
    }
    let abs_total = abs_correct + abs_incorrect;
    let abs_acc = if abs_total > 0 {
        abs_correct as f64 / abs_total as f64
    } else {
        0.0
    };
    eprintln!(
        "  TOTAL (abstract): {abs_correct}/{abs_total} ({abs_acc:.3}), {abs_unparseable} unparseable"
    );

    // TENSA should achieve very high accuracy on structured year data
    assert!(
        grand_acc > 0.90,
        "TENSA should score >90% on real ChronoSense, got {grand_acc:.3}"
    );
}

// ─── Real ChronoSense: LLM vs TENSA (paper-comparable) ──────

/// Runs both TENSA (structured) and an LLM on the REAL ChronoSense dataset.
/// Uses the paper's exact scoring methodology (interpret_response).
/// Same data, same prompts, same metrics — no cheating.
#[test]
#[ignore]
fn test_real_chronosense_llm_vs_tensa() {
    use chronosense::real_eval;

    let data_dir = std::path::Path::new("tests/chronosense/data");
    if !data_dir.join("Before_real.json").exists() {
        eprintln!("Skipping: ChronoSense data not downloaded");
        return;
    }

    let llm_config = match llm_baseline::detect_llm_auto(false) {
        Some(c) => c,
        None => {
            eprintln!("No LLM config found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY.");
            return;
        }
    };

    eprintln!("\n{}", "=".repeat(60));
    eprintln!(
        "=== ChronoSense: TENSA vs {} (paper-comparable) ===",
        llm_config.model
    );
    eprintln!("Dataset: Real-Allen (Wikidata events), 13 relations × 500 items");
    eprintln!("Scoring: paper-exact interpret_response (true/false keyword)");
    eprintln!("{}\n", "=".repeat(60));

    let relations = chronosense::chronosense_data::load_all_relations(data_dir, false);

    // TENSA structured (algebraic upper bound)
    let tensa_results: Vec<_> = relations
        .iter()
        .map(|(name, items)| real_eval::tensa_structured_evaluate(items, name, "Wikidata"))
        .collect();
    let tensa_eval = real_eval::aggregate_results("TENSA (structured)", None, tensa_results);
    real_eval::print_eval_result(&tensa_eval);

    // LLM baseline on real ChronoSense prompts
    eprintln!(
        "\nRunning LLM baseline on {} items...",
        relations.iter().map(|(_, v)| v.len()).sum::<usize>()
    );
    let rt = tokio::runtime::Runtime::new().unwrap();
    let llm_results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for (name, items) in &relations {
            eprintln!("  Evaluating {} ({} items)...", name, items.len());
            let r = real_eval::llm_evaluate_relation(items, &llm_config, name, "Wikidata").await;
            results.push(r);
        }
        results
    });
    let llm_eval =
        real_eval::aggregate_results("LLM baseline", Some(&llm_config.model), llm_results);
    real_eval::print_eval_result(&llm_eval);

    // Side-by-side comparison
    eprintln!("\n### Side-by-side comparison");
    eprintln!(
        "| {:<16} | {:>12} | {:>12} |",
        "Relation", "TENSA", &llm_config.model
    );
    eprintln!("|{:-<18}|{:-<14}|{:-<14}|", "", "", "");
    for (t, l) in tensa_eval.relations.iter().zip(llm_eval.relations.iter()) {
        eprintln!(
            "| {:<16} | {:>12.3} | {:>12.3} |",
            t.relation, t.accuracy, l.accuracy
        );
    }
    eprintln!("|{:-<18}|{:-<14}|{:-<14}|", "", "", "");
    eprintln!(
        "| {:<16} | {:>12.3} | {:>12.3} |",
        "OVERALL", tensa_eval.overall_accuracy, llm_eval.overall_accuracy
    );
    eprintln!(
        "\nTENSA: {:.1}% | LLM: {:.1}% | Delta: {:.1}pp",
        tensa_eval.overall_accuracy * 100.0,
        llm_eval.overall_accuracy * 100.0,
        (tensa_eval.overall_accuracy - llm_eval.overall_accuracy) * 100.0
    );

    // Output JSON for paper
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "benchmark": "ChronoSense (Islakoglu & Kalo, ACL 2025)",
        "dataset": "Real-Allen (Wikidata)",
        "tensa": tensa_eval,
        "llm": llm_eval,
    }))
    .unwrap();
    eprintln!("\n--- JSON ---\n{json}\n--- End ---");
}

// ─── Prompt export for offline evaluation ────────────────────

#[test]
#[ignore]
fn test_export_prompts() {
    let config = EvalConfig {
        items_per_relation: 50,
        ..Default::default()
    };

    let items = dataset::generate_abstract_split(&config);
    let pairs = neighbors::generate_minimal_pairs(&items, config.delta_seconds);
    let exports = llm_baseline::export_prompts(&pairs);

    let json = serde_json::to_string_pretty(&exports).unwrap();

    // Write to file
    let path = std::env::var("CHRONOSENSE_PROMPT_FILE")
        .unwrap_or_else(|_| "chronosense_prompts.json".into());
    std::fs::write(&path, &json).unwrap();
    eprintln!("Exported {} prompt sets to {}", exports.len(), path);
    eprintln!("Each set has 3 paraphrase variants.");
    eprintln!(
        "To evaluate: fill in 'response' field for each prompt, save as chronosense_responses.json"
    );
}
