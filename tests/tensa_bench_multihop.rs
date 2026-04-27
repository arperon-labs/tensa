//! Multi-hop GraphRAG benchmark suite.
//!
//! Evaluates TENSA's RAG modes on HotpotQA and 2WikiMultiHopQA against
//! GraphRAG, LightRAG, HippoRAG, and vanilla RAG.
//!
//! Run: TENSA_BENCHMARK_DATA=/path OPENROUTER_API_KEY=... cargo test --no-default-features --ignored bench_multihop

mod tensa_bench;

use tensa_bench::adapters::multihop_adapter::{
    self, MultihopConfig, MultihopItemResult, MultihopModeResult,
};
use tensa_bench::baselines::multihop_baselines;
use tensa_bench::datasets::hotpotqa::{HotpotQA, HotpotQAItem};
use tensa_bench::datasets::wikimultihop::WikiMultiHop;
use tensa_bench::datasets::{self, DatasetLoader, Split};
use tensa_bench::metrics::qa;
use tensa_bench::{benchmark_data_dir, timed, DatasetReport};

// ─── Unit Tests (no data or LLM required) ───

#[test]
fn test_qa_metrics_exact_match() {
    assert!(qa::exact_match("New York City", "new york city"));
    assert!(qa::exact_match("The answer is 42.", "answer is 42"));
    assert!(!qa::exact_match("yes", "no"));
}

#[test]
fn test_qa_metrics_token_f1() {
    let f1 = qa::token_f1("the quick brown fox", "quick brown fox the");
    assert!(
        (f1 - 1.0).abs() < 1e-6,
        "Same tokens (reordered) should give F1~1.0, got {}",
        f1
    );

    let f1 = qa::token_f1("quick brown", "quick brown fox");
    assert!(
        (f1 - 0.8).abs() < 0.02,
        "2/3 overlap should give F1~0.8, got {}",
        f1
    );

    let f1 = qa::token_f1("hello world", "foo bar");
    assert!(f1 < 0.01, "No overlap should give F1~0, got {}", f1);
}

#[test]
fn test_qa_normalize() {
    assert_eq!(qa::normalize_answer("The cat's hat."), "cats hat");
    assert_eq!(qa::normalize_answer("  A   big   house  "), "big house");
    assert_eq!(qa::normalize_answer("AN APPLE"), "apple");
}

#[test]
fn test_multihop_config_default() {
    let config = MultihopConfig::default();
    assert_eq!(config.sample_size, 100);
    assert_eq!(config.modes.len(), 4);
}

#[test]
fn test_build_ask_query() {
    let q = multihop_adapter::build_ask_query("Who was the president?", "story-1", "hybrid");
    assert!(q.contains("ASK"));
    assert!(q.contains("OVER"));
    assert!(q.contains("story-1"));
    assert!(q.contains("hybrid"));
}

#[test]
fn test_score_answer() {
    let (em, f1) = multihop_adapter::score_answer("New York City", "new york city");
    assert!(em);
    assert!((f1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_mode_result_aggregation() {
    let items = vec![
        MultihopItemResult {
            item_id: "1".to_string(),
            question: "Q1".to_string(),
            gold_answer: "A1".to_string(),
            predicted_answer: "A1".to_string(),
            mode: "local".to_string(),
            exact_match: true,
            token_f1: 1.0,
            latency_ms: 100,
        },
        MultihopItemResult {
            item_id: "2".to_string(),
            question: "Q2".to_string(),
            gold_answer: "A2".to_string(),
            predicted_answer: "wrong".to_string(),
            mode: "local".to_string(),
            exact_match: false,
            token_f1: 0.0,
            latency_ms: 200,
        },
    ];
    let result = MultihopModeResult::from_items("local", &items);
    assert_eq!(result.num_items, 2);
    assert!((result.avg_em - 0.5).abs() < 1e-6);
    assert!((result.avg_f1 - 0.5).abs() < 1e-6);
}

#[test]
fn test_baseline_constants() {
    let baselines = multihop_baselines::hotpotqa_baselines();
    assert!(baselines.len() >= 4);
    for (name, em, f1) in &baselines {
        assert!(*em >= 0.0 && *em <= 1.0, "{}: EM out of range", name);
        assert!(*f1 >= 0.0 && *f1 <= 1.0, "{}: F1 out of range", name);
        assert!(
            f1 >= em,
            "{}: F1 should be >= EM (F1={}, EM={})",
            name,
            f1,
            em
        );
    }
}

// ─── HotpotQA Loader Test (requires data) ───

#[test]
#[ignore]
fn bench_hotpotqa_loader() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("hotpotqa"));
            return;
        }
    };

    if !HotpotQA::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("hotpotqa"));
        return;
    }

    let (items, ms) = timed(|| HotpotQA::load(&data_dir, Split::Valid));
    let items = items.expect("Failed to load HotpotQA");
    eprintln!("Loaded {} HotpotQA items in {}ms", items.len(), ms);
    assert!(!items.is_empty());

    // Verify structure
    let item = &items[0];
    assert!(!item.question.is_empty());
    assert!(!item.answer.is_empty());
    assert!(!item.context.is_empty());
    assert!(
        item.question_type == "bridge" || item.question_type == "comparison",
        "Unexpected question type: {}",
        item.question_type
    );
    eprintln!(
        "Sample: Q='{}' A='{}' type={} contexts={}",
        &item.question[..item.question.len().min(60)],
        &item.answer,
        item.question_type,
        item.context.len()
    );

    // Count by type
    let bridge_count = items.iter().filter(|i| i.question_type == "bridge").count();
    let comparison_count = items
        .iter()
        .filter(|i| i.question_type == "comparison")
        .count();
    eprintln!(
        "Types: {} bridge, {} comparison",
        bridge_count, comparison_count
    );
}

// ─── Full Benchmark (requires data + LLM) ───

#[test]
#[ignore]
fn bench_hotpotqa_smoke() {
    // Smoke test: 5 items, verify end-to-end pipeline works
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("hotpotqa"));
            return;
        }
    };

    if !HotpotQA::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("hotpotqa"));
        return;
    }

    // Check for LLM API key
    if std::env::var("OPENROUTER_API_KEY").is_err()
        && std::env::var("ANTHROPIC_API_KEY").is_err()
        && std::env::var("LOCAL_LLM_URL").is_err()
    {
        eprintln!("Smoke test requires an LLM provider. Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or LOCAL_LLM_URL.");
        return;
    }

    let items = HotpotQA::load(&data_dir, Split::Valid).expect("Failed to load HotpotQA");
    let sample: Vec<&HotpotQAItem> = items.iter().take(5).collect();

    eprintln!("=== HotpotQA Smoke Test ({} items) ===", sample.len());
    for (i, item) in sample.iter().enumerate() {
        eprintln!(
            "  [{}] Q: {} | A: {}",
            i + 1,
            &item.question[..item.question.len().min(50)],
            &item.answer
        );
    }

    // At this point the full evaluation would:
    // 1. For each item: create MemoryStore + Hypergraph
    // 2. Ingest context via IngestionPipeline
    // 3. Call execute_ask() with the question
    // 4. Score answer with EM + F1
    //
    // This requires the server feature for full RAG. For now, verify the
    // adapter functions work correctly.
    for item in &sample {
        let text = multihop_adapter::build_ingestion_text(item);
        assert!(!text.is_empty(), "Ingestion text should not be empty");
        assert!(
            text.len() > 100,
            "Ingestion text should be substantial (got {} chars)",
            text.len()
        );

        let query = multihop_adapter::build_ask_query(&item.question, "test", "hybrid");
        assert!(query.contains("ASK"));
    }

    eprintln!("Smoke test passed: adapter functions work correctly.");
}

#[test]
#[ignore]
fn bench_hotpotqa_full() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("hotpotqa"));
            return;
        }
    };

    if !HotpotQA::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("hotpotqa"));
        return;
    }

    if std::env::var("OPENROUTER_API_KEY").is_err()
        && std::env::var("ANTHROPIC_API_KEY").is_err()
        && std::env::var("LOCAL_LLM_URL").is_err()
    {
        eprintln!("Full benchmark requires an LLM provider.");
        return;
    }

    let config = MultihopConfig::from_env();
    let items = HotpotQA::load(&data_dir, Split::Valid).expect("Failed to load HotpotQA");
    let sample_size = if config.sample_size == 0 {
        items.len()
    } else {
        config.sample_size.min(items.len())
    };

    eprintln!("=== HotpotQA Full Benchmark ({} items) ===", sample_size);
    eprintln!("Modes: {:?}", config.modes);

    // TODO: Full evaluation loop using TENSA's RAG pipeline.
    // For each mode:
    //   For each item (up to sample_size):
    //     1. Create fresh MemoryStore + Hypergraph
    //     2. Ingest all context paragraphs
    //     3. execute_ask(question, narrative_id, mode)
    //     4. Score with EM + F1
    //     5. Record MultihopItemResult
    //   Aggregate into MultihopModeResult
    //
    // Compare best mode against baselines.
    // Write results to tensa_bench_results.json.

    eprintln!(
        "Full evaluation requires integration with TENSA's RAG pipeline. \
         Use the adapter functions in tensa_bench::adapters::multihop_adapter."
    );

    // Output baseline reference
    let baselines = multihop_baselines::hotpotqa_baselines();
    eprintln!("\nReference baselines (HotpotQA distractor):");
    for (name, em, f1) in &baselines {
        eprintln!("  {}: EM={:.3}, F1={:.3}", name, em, f1);
    }
}

#[test]
#[ignore]
fn bench_2wikimultihop_loader() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("wikimultihop"));
            return;
        }
    };

    if !WikiMultiHop::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("wikimultihop"));
        return;
    }

    let items =
        WikiMultiHop::load(&data_dir, Split::Valid).expect("Failed to load 2WikiMultiHopQA");
    eprintln!("Loaded {} 2WikiMultiHopQA items", items.len());
    assert!(!items.is_empty());

    let item = &items[0];
    eprintln!(
        "Sample: Q='{}' A='{}' type={} evidences={}",
        &item.question[..item.question.len().min(60)],
        &item.answer,
        item.question_type,
        item.evidences.len()
    );
}
