//! Narrative understanding benchmark suite.
//!
//! Evaluates TENSA on ROCStories (story completion), NarrativeQA (reading comprehension),
//! and MAVEN-ERE (event relation extraction).
//!
//! Run: TENSA_BENCHMARK_DATA=/path OPENROUTER_API_KEY=... cargo test --no-default-features --ignored bench_narrative

mod tensa_bench;

use tensa_bench::adapters::narrative_adapter;
use tensa_bench::baselines::narrative_baselines;
use tensa_bench::datasets::maven_ere::MavenEre;
use tensa_bench::datasets::narrativeqa::NarrativeQA;
use tensa_bench::datasets::rocstories::RocStories;
use tensa_bench::datasets::{self, DatasetLoader, Split};
use tensa_bench::metrics::classification::ConfusionMatrix;
use tensa_bench::metrics::nlg;
use tensa_bench::{benchmark_data_dir, timed, DatasetReport};

// ─── Unit Tests (no data required) ───

#[test]
fn test_nlg_bleu_identical() {
    let score = nlg::bleu_1("the cat sat on the mat", &["the cat sat on the mat"]);
    assert!(
        (score - 1.0).abs() < 1e-6,
        "Identical should give BLEU-1=1.0, got {}",
        score
    );
}

#[test]
fn test_nlg_bleu_no_overlap() {
    let score = nlg::bleu_1("hello world", &["foo bar baz"]);
    assert!(score < 0.01, "No overlap should give ~0, got {}", score);
}

#[test]
fn test_nlg_rouge_l() {
    let score = nlg::rouge_l("the cat sat on the mat", "the cat sat on the mat");
    assert!(
        (score - 1.0).abs() < 1e-6,
        "Identical should give ROUGE-L=1.0, got {}",
        score
    );

    let score = nlg::rouge_l("the cat sat", "the cat sat on the mat");
    assert!(
        score > 0.5 && score < 1.0,
        "Partial overlap should give moderate ROUGE-L, got {}",
        score
    );
}

#[test]
fn test_nlg_meteor() {
    let score = nlg::meteor("the cat sat on the mat", "the cat sat on the mat");
    assert!(
        score > 0.9,
        "Identical should give high METEOR, got {}",
        score
    );

    let score = nlg::meteor("hello world", "foo bar baz");
    assert!(score < 0.01, "No overlap should give ~0, got {}", score);
}

#[test]
fn test_classification_metrics() {
    let mut cm = ConfusionMatrix::new();
    cm.add_pair("CAUSE", "CAUSE");
    cm.add_pair("CAUSE", "CAUSE");
    cm.add_pair("BEFORE", "BEFORE");
    cm.add_pair("CAUSE", "BEFORE"); // FP for CAUSE, FN for BEFORE

    let macro_f1 = cm.macro_f1();
    let micro_f1 = cm.micro_f1();
    assert!(macro_f1 > 0.0 && macro_f1 < 1.0);
    assert!(micro_f1 > 0.0 && micro_f1 < 1.0);
}

#[test]
fn test_tensa_relation_mapping() {
    assert_eq!(
        narrative_adapter::tensa_causal_to_maven("Necessary"),
        Some("CAUSE")
    );
    assert_eq!(
        narrative_adapter::tensa_causal_to_maven("Contributing"),
        Some("PRECONDITION")
    );
    assert_eq!(
        narrative_adapter::tensa_temporal_to_maven("Before"),
        Some("BEFORE")
    );
    assert_eq!(
        narrative_adapter::tensa_temporal_to_maven("Equals"),
        Some("SIMULTANEOUS")
    );
}

#[test]
fn test_parse_ending_choice() {
    assert_eq!(narrative_adapter::parse_ending_choice("A"), Some(1));
    assert_eq!(
        narrative_adapter::parse_ending_choice("B. The second"),
        Some(2)
    );
    assert_eq!(
        narrative_adapter::parse_ending_choice("I pick ending A"),
        Some(1)
    );
    assert_eq!(narrative_adapter::parse_ending_choice("unclear"), None);
}

#[test]
fn test_baseline_constants() {
    let roc = narrative_baselines::rocstories_baselines();
    assert!(roc.len() >= 3);
    for (name, acc) in &roc {
        assert!(
            *acc >= 0.0 && *acc <= 1.0,
            "{}: accuracy out of range",
            name
        );
    }

    let maven = narrative_baselines::maven_ere_baselines();
    assert!(maven.len() >= 3);

    let nqa = narrative_baselines::narrativeqa_baselines();
    assert!(nqa.len() >= 3);
}

// ─── Dataset Loader Tests (require data) ───

#[test]
#[ignore]
fn bench_rocstories_loader() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("rocstories"));
            return;
        }
    };

    if !RocStories::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("rocstories"));
        return;
    }

    let (items, ms) = timed(|| RocStories::load(&data_dir, Split::Valid));
    let items = items.expect("Failed to load ROCStories");
    eprintln!("Loaded {} ROCStories items in {}ms", items.len(), ms);
    assert!(!items.is_empty());

    let item = &items[0];
    eprintln!("Sample prefix: {}", item.prefix());
    eprintln!("  Ending A: {}", item.ending1);
    eprintln!("  Ending B: {}", item.ending2);
    eprintln!("  Correct: {}", item.correct_ending);
}

#[test]
#[ignore]
fn bench_maven_ere_loader() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("maven_ere"));
            return;
        }
    };

    if !MavenEre::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("maven_ere"));
        return;
    }

    let (docs, ms) = timed(|| MavenEre::load(&data_dir, Split::Valid));
    let docs = docs.expect("Failed to load MAVEN-ERE");
    eprintln!("Loaded {} MAVEN-ERE documents in {}ms", docs.len(), ms);
    assert!(!docs.is_empty());

    let doc = &docs[0];
    eprintln!("Sample doc: id={}, title={}", doc.id, doc.title);
    eprintln!("  Events: {}", doc.events.len());
    eprintln!("  Temporal relations: {}", doc.temporal_relations.len());
    eprintln!("  Causal relations: {}", doc.causal_relations.len());
    eprintln!("  Subevent relations: {}", doc.subevent_relations.len());

    // Aggregate stats
    let total_events: usize = docs.iter().map(|d| d.events.len()).sum();
    let total_relations: usize = docs.iter().map(|d| d.all_relations().len()).sum();
    eprintln!(
        "Total: {} events, {} relations across {} docs",
        total_events,
        total_relations,
        docs.len()
    );
}

#[test]
#[ignore]
fn bench_narrativeqa_loader() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("narrativeqa"));
            return;
        }
    };

    if !NarrativeQA::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("narrativeqa"));
        return;
    }

    let (items, ms) = timed(|| NarrativeQA::load(&data_dir, Split::Valid));
    let items = items.expect("Failed to load NarrativeQA");
    eprintln!("Loaded {} NarrativeQA items in {}ms", items.len(), ms);
    assert!(!items.is_empty());

    let item = &items[0];
    eprintln!("Sample: doc_id={}", item.document_id);
    eprintln!("  Q: {}", &item.question[..item.question.len().min(80)]);
    eprintln!("  Answers: {:?}", item.answers);
    if let Some(ref summary) = item.summary {
        eprintln!("  Summary: {}...", &summary[..summary.len().min(100)]);
    }
}

// ─── Full Benchmarks (require data + LLM) ───

#[test]
#[ignore]
fn bench_maven_ere_full() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("maven_ere"));
            return;
        }
    };

    if !MavenEre::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("maven_ere"));
        return;
    }

    let docs = MavenEre::load(&data_dir, Split::Valid).expect("Failed to load MAVEN-ERE");
    let sample_size = std::env::var("TENSA_BENCH_SAMPLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50usize)
        .min(docs.len());

    eprintln!("=== MAVEN-ERE Evaluation ({} documents) ===", sample_size);

    // TODO: Full evaluation loop:
    // For each document:
    //   1. Ingest document text via IngestionPipeline
    //   2. Extract causal links and temporal relations from hypergraph
    //   3. Map to MAVEN relation types via tensa_causal_to_maven / tensa_temporal_to_maven
    //   4. Compare against ground truth using ConfusionMatrix
    //   5. Accumulate results

    let baselines = narrative_baselines::maven_ere_baselines();
    eprintln!("\nReference baselines (MAVEN-ERE micro-F1):");
    for (name, f1) in &baselines {
        eprintln!("  {}: {:.3}", name, f1);
    }
}

#[test]
#[ignore]
fn bench_rocstories_full() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("rocstories"));
            return;
        }
    };

    if !RocStories::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("rocstories"));
        return;
    }

    if std::env::var("OPENROUTER_API_KEY").is_err()
        && std::env::var("ANTHROPIC_API_KEY").is_err()
        && std::env::var("LOCAL_LLM_URL").is_err()
    {
        eprintln!("ROCStories benchmark requires an LLM provider.");
        return;
    }

    let items = RocStories::load(&data_dir, Split::Valid).expect("Failed to load ROCStories");
    let sample_size = std::env::var("TENSA_BENCH_SAMPLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100usize)
        .min(items.len());

    eprintln!(
        "=== ROCStories Story Cloze Test ({} items) ===",
        sample_size
    );

    // TODO: Full evaluation loop:
    // For each item:
    //   1. Ingest 4-sentence prefix
    //   2. execute_ask() with story completion prompt
    //   3. Parse A/B response
    //   4. Score accuracy

    let baselines = narrative_baselines::rocstories_baselines();
    eprintln!("\nReference baselines (Story Cloze accuracy):");
    for (name, acc) in &baselines {
        eprintln!("  {}: {:.3}", name, acc);
    }
}

#[test]
#[ignore]
fn bench_narrativeqa_full() {
    let data_dir = match benchmark_data_dir() {
        Some(d) => d,
        None => {
            eprintln!("{}", datasets::download_instructions("narrativeqa"));
            return;
        }
    };

    if !NarrativeQA::is_available(&data_dir) {
        eprintln!("{}", datasets::download_instructions("narrativeqa"));
        return;
    }

    if std::env::var("OPENROUTER_API_KEY").is_err()
        && std::env::var("ANTHROPIC_API_KEY").is_err()
        && std::env::var("LOCAL_LLM_URL").is_err()
    {
        eprintln!("NarrativeQA benchmark requires an LLM provider.");
        return;
    }

    let items = NarrativeQA::load(&data_dir, Split::Valid).expect("Failed to load NarrativeQA");
    let sample_size = std::env::var("TENSA_BENCH_SAMPLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100usize)
        .min(items.len());

    eprintln!(
        "=== NarrativeQA Reading Comprehension ({} items) ===",
        sample_size
    );

    // TODO: Full evaluation loop:
    // For each item:
    //   1. Ingest document summary (or full text)
    //   2. execute_ask() with the question
    //   3. Score with BLEU-1/4, ROUGE-L, METEOR

    let baselines = narrative_baselines::narrativeqa_baselines();
    eprintln!("\nReference baselines (NarrativeQA ROUGE-L):");
    for (name, rl) in &baselines {
        eprintln!("  {}: {:.3}", name, rl);
    }
}
