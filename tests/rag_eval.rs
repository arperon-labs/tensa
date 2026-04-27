//! RAG Baselines × ChronoSense Evaluation
//!
//! Tests GraphRAG, LightRAG, and HippoRAG 2 on the same 6,500-item
//! Real-Allen ChronoSense dataset used for TENSA and Claude baselines.
//!
//! Run smoke test (10 items, GraphRAG only):
//!   cargo test --test rag_eval test_graphrag_mode_a_smoke -- --ignored --nocapture
//!
//! Run full GraphRAG Mode A:
//!   cargo test --test rag_eval test_graphrag_mode_a_full -- --ignored --nocapture
//!
//! Run all systems comparison:
//!   cargo test --test rag_eval test_rag_comparison -- --ignored --nocapture
//!
//! Requires: Python 3.10+, OPENROUTER_API_KEY env var, ChronoSense data

mod chronosense;

use chronosense::chronosense_data;
use chronosense::llm_baseline;
use chronosense::rag_baselines::graphrag_runner::{self, GraphRagRetrievalMode};
use chronosense::rag_baselines::hipporag_runner;
use chronosense::rag_baselines::lightrag_runner;
use chronosense::rag_baselines::scoring;
use chronosense::rag_baselines::shared::RagSubprocess;
use chronosense::rag_baselines::{RagMode, RagSystem};
use chronosense::real_eval;

fn data_dir() -> &'static std::path::Path {
    std::path::Path::new("tests/chronosense/data")
}

fn check_prereqs() -> Option<chronosense::llm_baseline::LlmConfig> {
    if !data_dir().join("Before_real.json").exists() {
        eprintln!("Skipping: ChronoSense data not in tests/chronosense/data/");
        return None;
    }
    match llm_baseline::detect_llm_auto(false) {
        Some(c) => {
            eprintln!(
                "LLM: {} (via {:?})",
                c.model,
                std::mem::discriminant(&c.provider)
            );
            Some(c)
        }
        None => {
            eprintln!("No LLM config. Set OPENROUTER_API_KEY.");
            None
        }
    }
}

// ─── GraphRAG ────────────────────────────────────────────────

#[test]
#[ignore]
fn test_graphrag_mode_a_smoke() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== GraphRAG Mode A — Smoke Test (10 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let (name, items) = &relations[0]; // Before
    let items_10: Vec<_> = items.iter().take(10).cloned().collect();

    let mut proc = match RagSubprocess::spawn(&RagSystem::GraphRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn GraphRAG: {e}");
            eprintln!("Install: pip install -r tests/chronosense/rag_baselines/python/requirements_graphrag.txt");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(graphrag_runner::evaluate_relation_mode_a(
        &items_10,
        &config,
        name,
        GraphRagRetrievalMode::Local,
        &mut proc,
    ));

    proc.shutdown();

    eprintln!(
        "\nGraphRAG smoke: {}/{} correct ({:.3})",
        result.correct,
        result.correct + result.incorrect,
        result.accuracy
    );
}

#[test]
#[ignore]
fn test_graphrag_mode_a_full() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== GraphRAG Mode A — Full (6,500 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let start = std::time::Instant::now();

    let mut proc = match RagSubprocess::spawn(&RagSystem::GraphRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn GraphRAG: {e}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for (name, items) in &relations {
            eprintln!("  Evaluating {name} ({} items)...", items.len());
            let r = graphrag_runner::evaluate_relation_mode_a(
                items,
                &config,
                name,
                GraphRagRetrievalMode::Local,
                &mut proc,
            )
            .await;
            results.push(r);
        }
        results
    });

    proc.shutdown();

    let output = scoring::build_output(
        &RagSystem::GraphRAG,
        &RagMode::PerItem,
        "local",
        "unknown",
        &results,
        start.elapsed(),
        scoring::TokenUsage::default(),
        0.0,
    );
    let out_path = std::path::Path::new("tests/chronosense/rag_cache/graphrag_mode_a_results.json");
    if let Err(e) = scoring::write_output(&output, out_path) {
        eprintln!("Failed to write output: {e}");
    }

    let eval = real_eval::aggregate_results("GraphRAG (Mode A)", Some(&config.model), results);
    real_eval::print_eval_result(&eval);

    let json = serde_json::to_string_pretty(&output).unwrap_or_default();
    eprintln!("\n--- JSON ---\n{json}\n--- End ---");
}

// ─── LightRAG ────────────────────────────────────────────────

#[test]
#[ignore]
fn test_lightrag_mode_a_smoke() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== LightRAG Mode A — Smoke Test (10 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let (name, items) = &relations[0];
    let items_10: Vec<_> = items.iter().take(10).cloned().collect();

    let mut proc = match RagSubprocess::spawn(&RagSystem::LightRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn LightRAG: {e}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(lightrag_runner::evaluate_relation_mode_a(
        &items_10, &config, name, &mut proc,
    ));

    proc.shutdown();
    eprintln!(
        "\nLightRAG smoke: {}/{} correct ({:.3})",
        result.correct,
        result.correct + result.incorrect,
        result.accuracy
    );
}

/// 50 items per relation (650 total, ~2.7h). Statistically meaningful, practical to run.
#[test]
#[ignore]
fn test_lightrag_mode_a_50() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== LightRAG Mode A — 50/relation (650 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let start = std::time::Instant::now();

    let mut proc = match RagSubprocess::spawn(&RagSystem::LightRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn LightRAG: {e}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for (name, items) in &relations {
            let items_50: Vec<_> = items.iter().take(50).cloned().collect();
            eprintln!("  Evaluating {name} ({} items)...", items_50.len());
            let r = lightrag_runner::evaluate_relation_mode_a(&items_50, &config, name, &mut proc)
                .await;
            results.push(r);
        }
        results
    });

    proc.shutdown();

    let output = scoring::build_output(
        &RagSystem::LightRAG,
        &RagMode::PerItem,
        "hybrid",
        "unknown",
        &results,
        start.elapsed(),
        scoring::TokenUsage::default(),
        0.0,
    );
    let out_path =
        std::path::Path::new("tests/chronosense/rag_cache/lightrag_mode_a_50_results.json");
    let _ = scoring::write_output(&output, out_path);

    let eval =
        real_eval::aggregate_results("LightRAG (Mode A, 50/rel)", Some(&config.model), results);
    real_eval::print_eval_result(&eval);
    eprintln!("\nResults written to {}", out_path.display());
    eprintln!("Wall time: {:.0}s", start.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn test_lightrag_mode_a_full() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== LightRAG Mode A — Full (6,500 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let start = std::time::Instant::now();

    let mut proc = match RagSubprocess::spawn(&RagSystem::LightRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn LightRAG: {e}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for (name, items) in &relations {
            eprintln!("  Evaluating {name} ({} items)...", items.len());
            let r =
                lightrag_runner::evaluate_relation_mode_a(items, &config, name, &mut proc).await;
            results.push(r);
        }
        results
    });

    proc.shutdown();

    let output = scoring::build_output(
        &RagSystem::LightRAG,
        &RagMode::PerItem,
        "hybrid",
        "unknown",
        &results,
        start.elapsed(),
        scoring::TokenUsage::default(),
        0.0,
    );
    let out_path = std::path::Path::new("tests/chronosense/rag_cache/lightrag_mode_a_results.json");
    let _ = scoring::write_output(&output, out_path);

    let eval = real_eval::aggregate_results("LightRAG (Mode A)", Some(&config.model), results);
    real_eval::print_eval_result(&eval);
    eprintln!("\nResults written to {}", out_path.display());
}

// ─── HippoRAG ────────────────────────────────────────────────

#[test]
#[ignore]
fn test_hipporag_mode_a_smoke() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== HippoRAG Mode A — Smoke Test (10 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let (name, items) = &relations[0];
    let items_10: Vec<_> = items.iter().take(10).cloned().collect();

    let mut proc = match RagSubprocess::spawn(&RagSystem::HippoRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn HippoRAG: {e}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(hipporag_runner::evaluate_relation_mode_a(
        &items_10, &config, name, &mut proc,
    ));

    proc.shutdown();
    eprintln!(
        "\nHippoRAG smoke: {}/{} correct ({:.3})",
        result.correct,
        result.correct + result.incorrect,
        result.accuracy
    );
}

#[test]
#[ignore]
fn test_hipporag_mode_a_full() {
    let config = match check_prereqs() {
        Some(c) => c,
        None => return,
    };

    eprintln!("\n=== HippoRAG Mode A — Full (6,500 items) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let start = std::time::Instant::now();

    let mut proc = match RagSubprocess::spawn(&RagSystem::HippoRAG, &RagMode::PerItem) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to spawn HippoRAG: {e}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for (name, items) in &relations {
            eprintln!("  Evaluating {name} ({} items)...", items.len());
            let r =
                hipporag_runner::evaluate_relation_mode_a(items, &config, name, &mut proc).await;
            results.push(r);
        }
        results
    });

    proc.shutdown();

    let output = scoring::build_output(
        &RagSystem::HippoRAG,
        &RagMode::PerItem,
        "ppr",
        "unknown",
        &results,
        start.elapsed(),
        scoring::TokenUsage::default(),
        0.0,
    );
    let out_path = std::path::Path::new("tests/chronosense/rag_cache/hipporag_mode_a_results.json");
    let _ = scoring::write_output(&output, out_path);

    let eval = real_eval::aggregate_results("HippoRAG (Mode A)", Some(&config.model), results);
    real_eval::print_eval_result(&eval);
    eprintln!("\nResults written to {}", out_path.display());
}

// ─── TENSA Ingestion Mode ────────────────────────────────────

/// System prompt for ChronoSense year-interval extraction.
/// Tuned for year-resolution temporal data, not narrative prose.
const CHRONOSENSE_EXTRACT_PROMPT: &str = r#"You are a temporal interval parser. Given a text describing two historical events with year ranges, extract the start and end years for each event.

Return ONLY a JSON object:
{"event_a": {"name": "...", "start_year": NNNN, "end_year": NNNN}, "event_b": {"name": "...", "start_year": NNNN, "end_year": NNNN}}

Rules:
- start_year and end_year are integers (years)
- Extract EXACTLY the years stated in the text
- "between year X and year Y" means start_year=X, end_year=Y
- Return ONLY JSON, no explanation"#;

/// Direct LLM call with custom system prompt (bypasses llm_baseline's system prompt).
async fn call_llm_with_system(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system: &str,
    user: &str,
) -> Option<String> {
    #[derive(serde::Serialize)]
    struct Req {
        model: String,
        messages: Vec<Msg>,
        max_tokens: u32,
        temperature: f64,
    }
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Msg {
        role: String,
        content: String,
    }
    #[derive(serde::Deserialize)]
    struct Resp {
        choices: Vec<Choice>,
    }
    #[derive(serde::Deserialize)]
    struct Choice {
        message: Msg,
    }

    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&Req {
            model: model.into(),
            messages: vec![
                Msg {
                    role: "system".into(),
                    content: system.into(),
                },
                Msg {
                    role: "user".into(),
                    content: user.into(),
                },
            ],
            max_tokens: 200,
            temperature: 0.0,
        })
        .send()
        .await
        .ok()?;
    let body: Resp = resp.json().await.ok()?;
    body.choices.first().map(|c| c.message.content.clone())
}

/// Extract year intervals from a ChronoSense item using an LLM,
/// then compute the Allen relation algebraically.
///
/// This uses a ChronoSense-tuned prompt (not TENSA's narrative extraction prompt)
/// to isolate the extraction-vs-reasoning question: can an LLM reliably
/// extract year intervals that feed into correct algebraic reasoning?
async fn llm_extract_and_reason(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    item: &chronosense::chronosense_data::ParsedItem,
) -> Option<tensa::types::AllenRelation> {
    use tensa::temporal::interval::relation_between;

    let user_prompt = format!("Text: {}", item.input);

    let response = call_llm_with_system(
        client,
        api_key,
        model,
        CHRONOSENSE_EXTRACT_PROMPT,
        &user_prompt,
    )
    .await?;
    let response = response.trim();

    // Strip markdown code fences if present
    let json_str = if response.starts_with("```") {
        response
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        response
    };

    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let a_start = val["event_a"]["start_year"].as_i64()? as i32;
    let a_end = val["event_a"]["end_year"].as_i64()? as i32;
    let b_start = val["event_b"]["start_year"].as_i64()? as i32;
    let b_end = val["event_b"]["end_year"].as_i64()? as i32;

    let ia = chronosense::chronosense_data::interval_from_years_pub(a_start, a_end);
    let ib = chronosense::chronosense_data::interval_from_years_pub(b_start, b_end);

    relation_between(&ia, &ib).ok()
}

/// TENSA ingestion: ingest each ChronoSense item as text via the real
/// extraction pipeline (OpenRouter gpt-4o-mini, same as LightRAG internal),
/// then answer using `relation_between()` on extracted intervals.
///
/// This tests the full claim: TENSA extraction + algebraic reasoning vs
/// RAG extraction + LLM reasoning, with identical extraction LLMs.
fn tensa_ingest_evaluate_relation(
    items: &[chronosense::chronosense_data::ParsedItem],
    relation_name: &str,
    api_key: &str,
    model: &str,
) -> real_eval::RelationResult {
    use std::sync::Arc;
    use tensa::hypergraph::Hypergraph;
    use tensa::ingestion::embed::HashEmbedding;
    use tensa::ingestion::llm::OpenRouterClient;
    use tensa::ingestion::pipeline::{IngestionPipeline, PipelineConfig};
    use tensa::ingestion::queue::ValidationQueue;
    use tensa::ingestion::vector::VectorIndex;
    use tensa::store::memory::MemoryStore;
    use tensa::temporal::interval::relation_between;

    // OpenRouterClient needs a tokio runtime handle on the current thread.
    // It calls Handle::try_current() internally in its sync extract method.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let _guard = rt.enter();

    let mut correct = 0usize;
    let mut incorrect = 0usize;
    let mut no_extraction = 0usize;

    for (idx, item) in items.iter().enumerate() {
        if idx % 10 == 0 {
            eprintln!("    [{idx}/{}]", items.len());
        }

        let asked = match item.asked_relation {
            Some(r) => r,
            None => {
                no_extraction += 1;
                continue;
            }
        };

        // Fresh store per item (isolated)
        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store.clone()));
        let queue = Arc::new(ValidationQueue::new(store.clone()));
        let extractor: Arc<dyn tensa::ingestion::llm::NarrativeExtractor> = Arc::new(
            OpenRouterClient::new(api_key.to_string(), model.to_string()),
        );
        let embedder = Arc::new(HashEmbedding::new(64));
        let vector_index = Arc::new(std::sync::RwLock::new(VectorIndex::new(64)));

        let narrative_id = format!("cs_{}_{:04}", relation_name, idx);
        let config = PipelineConfig {
            narrative_id: Some(narrative_id.clone()),
            source_id: "chronosense".into(),
            source_type: "benchmark".into(),
            enrich: true,               // Need enrichment for temporal chain extraction
            auto_commit_threshold: 0.1, // Accept everything (don't reject low-confidence)
            review_threshold: 0.0,
            ..Default::default()
        };

        let pipeline = IngestionPipeline::new(
            hg.clone(),
            extractor,
            Some(embedder),
            Some(vector_index),
            queue,
            config,
        );

        // Ingest the ChronoSense item text
        let report = match pipeline.ingest_text(&item.input, "chronosense") {
            Ok(r) => r,
            Err(e) => {
                eprintln!("    [ERROR] {relation_name}_{idx:04}: ingest failed: {e}");
                no_extraction += 1;
                continue;
            }
        };

        if report.situations_created < 2 {
            // Need at least 2 situations to compute a relation
            no_extraction += 1;
            continue;
        }

        // Get extracted situations and compute temporal relation
        let situations = match hg.list_situations_by_narrative(&narrative_id) {
            Ok(s) => s,
            Err(_) => {
                no_extraction += 1;
                continue;
            }
        };

        // Find two situations with valid temporal intervals
        let with_intervals: Vec<_> = situations
            .iter()
            .filter(|s| s.temporal.start.is_some() && s.temporal.end.is_some())
            .collect();

        if with_intervals.len() < 2 {
            no_extraction += 1;
            continue;
        }

        // Compute relation between first two situations with intervals
        let computed =
            match relation_between(&with_intervals[0].temporal, &with_intervals[1].temporal) {
                Ok(r) => r,
                Err(_) => {
                    no_extraction += 1;
                    continue;
                }
            };

        let predicted = computed == asked;
        if predicted == item.ground_truth {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }

    let total = correct + incorrect;
    eprintln!("    {relation_name}: {correct}/{total} correct, {no_extraction} no-extraction");

    real_eval::RelationResult {
        relation: relation_name.into(),
        dataset: "Wikidata".into(),
        total: total + no_extraction,
        correct,
        incorrect: incorrect + no_extraction,
        unclear: no_extraction,
        accuracy: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
    }
}

/// Smoke test: 10 items, TENSA ingestion with gpt-4o-mini extraction.
#[test]
#[ignore]
fn test_tensa_ingestion_smoke() {
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No OPENROUTER_API_KEY");
            return;
        }
    };
    // Use same extraction model as LightRAG for fairness
    let model = std::env::var("RAG_INTERNAL_MODEL").unwrap_or_else(|_| "openai/gpt-4o-mini".into());

    if !data_dir().join("Before_real.json").exists() {
        eprintln!("No ChronoSense data");
        return;
    }

    eprintln!("\n=== TENSA Ingestion — Smoke Test (10 items, {model}) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let (name, items) = &relations[0];
    let items_10: Vec<_> = items.iter().take(10).cloned().collect();

    let result = tensa_ingest_evaluate_relation(&items_10, name, &api_key, &model);
    eprintln!(
        "\nTENSA ingestion smoke: {}/{} correct ({:.3}), {} no-extraction",
        result.correct,
        result.correct + result.incorrect - result.unclear,
        result.accuracy,
        result.unclear
    );
}

/// 50 items per relation, TENSA ingestion with gpt-4o-mini extraction.
#[test]
#[ignore]
fn test_tensa_ingestion_50() {
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No OPENROUTER_API_KEY");
            return;
        }
    };
    let model = std::env::var("RAG_INTERNAL_MODEL").unwrap_or_else(|_| "openai/gpt-4o-mini".into());

    if !data_dir().join("Before_real.json").exists() {
        eprintln!("No ChronoSense data");
        return;
    }

    eprintln!("\n=== TENSA Ingestion — 50/relation (650 items, {model}) ===\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let start = std::time::Instant::now();

    let mut results = Vec::new();
    for (name, items) in &relations {
        let items_50: Vec<_> = items.iter().take(50).cloned().collect();
        eprintln!("  Evaluating {name} ({} items)...", items_50.len());
        let r = tensa_ingest_evaluate_relation(&items_50, name, &api_key, &model);
        results.push(r);
    }

    let eval = real_eval::aggregate_results("TENSA (ingestion)", Some(&model), results);
    real_eval::print_eval_result(&eval);
    eprintln!("\nWall time: {:.0}s", start.elapsed().as_secs_f64());
}

// ─── TENSA Tuned Extraction (LLM → algebra) ─────────────────

/// Evaluate one relation using tuned extraction: LLM parses years → relation_between().
async fn tensa_tuned_evaluate_relation(
    items: &[chronosense::chronosense_data::ParsedItem],
    api_key: &str,
    model: &str,
    relation_name: &str,
) -> real_eval::RelationResult {
    let client = reqwest::Client::new();
    let mut correct = 0usize;
    let mut incorrect = 0usize;
    let mut unclear = 0usize;

    for (idx, item) in items.iter().enumerate() {
        if idx % 10 == 0 && idx > 0 {
            eprintln!("    [{idx}/{}]", items.len());
        }

        let asked = match item.asked_relation {
            Some(r) => r,
            None => {
                unclear += 1;
                continue;
            }
        };

        let computed = llm_extract_and_reason(&client, api_key, model, item).await;

        match computed {
            Some(rel) => {
                let predicted = rel == asked;
                if predicted == item.ground_truth {
                    correct += 1;
                } else {
                    incorrect += 1;
                }
            }
            None => {
                // Extraction failed — count as wrong
                unclear += 1;
                incorrect += 1;
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let total = correct + incorrect;
    real_eval::RelationResult {
        relation: relation_name.into(),
        dataset: "Wikidata".into(),
        total: total + (unclear.min(1) * 0),
        correct,
        incorrect,
        unclear,
        accuracy: if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        },
    }
}

/// Smoke test: tuned extraction on 10 items from problematic relations.
#[test]
#[ignore]
fn test_tensa_tuned_smoke() {
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No OPENROUTER_API_KEY");
            return;
        }
    };
    let model = std::env::var("RAG_INTERNAL_MODEL").unwrap_or_else(|_| "openai/gpt-4o-mini".into());

    if !data_dir().join("Before_real.json").exists() {
        eprintln!("No ChronoSense data");
        return;
    }

    eprintln!("\n=== TENSA Tuned Extraction — Smoke (problematic relations) ===");
    eprintln!("Model: {model}\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let targets = ["Equals", "Meets", "During", "Overlaps", "Before"];

    let rt = tokio::runtime::Runtime::new().unwrap();
    for (name, items) in &relations {
        if !targets.contains(&name.as_str()) {
            continue;
        }
        let items_10: Vec<_> = items.iter().take(10).cloned().collect();
        eprintln!("  {name} (10 items)...");
        let result = rt.block_on(tensa_tuned_evaluate_relation(
            &items_10, &api_key, &model, name,
        ));
        eprintln!(
            "    -> {}/{} correct ({:.3}), {} unclear",
            result.correct,
            result.correct + result.incorrect - result.unclear,
            result.accuracy,
            result.unclear
        );
    }
}

/// 50 items per relation, tuned extraction.
#[test]
#[ignore]
fn test_tensa_tuned_50() {
    let api_key = match std::env::var("OPENROUTER_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            eprintln!("No OPENROUTER_API_KEY");
            return;
        }
    };
    let model = std::env::var("RAG_INTERNAL_MODEL").unwrap_or_else(|_| "openai/gpt-4o-mini".into());

    if !data_dir().join("Before_real.json").exists() {
        eprintln!("No ChronoSense data");
        return;
    }

    eprintln!("\n=== TENSA Tuned Extraction — 50/relation (650 items) ===");
    eprintln!("Model: {model}\n");

    let relations = chronosense_data::load_all_relations(data_dir(), false);
    let start = std::time::Instant::now();

    let rt = tokio::runtime::Runtime::new().unwrap();
    let results: Vec<_> = rt.block_on(async {
        let mut results = Vec::new();
        for (name, items) in &relations {
            let items_50: Vec<_> = items.iter().take(50).cloned().collect();
            eprintln!("  Evaluating {name} ({} items)...", items_50.len());
            let r = tensa_tuned_evaluate_relation(&items_50, &api_key, &model, name).await;
            results.push(r);
        }
        results
    });

    let eval = real_eval::aggregate_results("TENSA (tuned extract)", Some(&model), results);
    real_eval::print_eval_result(&eval);
    eprintln!("\nWall time: {:.0}s", start.elapsed().as_secs_f64());
}

// ─── Comparison ──────────────────────────────────────────────

#[test]
#[ignore]
fn test_rag_comparison() {
    eprintln!("\n=== ChronoSense RAG Baselines — Comparison Table ===\n");

    // Load cached results
    let systems = [
        ("graphrag_mode_a_results.json", "GraphRAG (Mode A)"),
        ("lightrag_mode_a_results.json", "LightRAG (Mode A)"),
        ("hipporag_mode_a_results.json", "HippoRAG (Mode A)"),
    ];

    eprintln!(
        "| {:<24} | {:>8} | {:>7} | {:>7} |",
        "System", "Accuracy", "Unclear", "Items"
    );
    eprintln!("|{:-<26}|{:-<10}|{:-<9}|{:-<9}|", "", "", "", "");

    // Known results (hardcoded from prior runs)
    eprintln!(
        "| {:<24} | {:>8.3} | {:>7} | {:>7} |",
        "TENSA (structured)", 1.000, 0, 6500
    );
    eprintln!(
        "| {:<24} | {:>8.3} | {:>7} | {:>7} |",
        "Claude Sonnet 4.6", 0.983, 12, 6500
    );

    for (filename, label) in &systems {
        let path = std::path::PathBuf::from("tests/chronosense/rag_cache").join(filename);
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(output) = serde_json::from_str::<scoring::RagEvalOutput>(&data) {
                eprintln!(
                    "| {:<24} | {:>8.3} | {:>7} | {:>7} |",
                    label, output.overall_accuracy, output.unclear_total, output.total_items
                );
            } else {
                eprintln!("| {:<24} | {:>8} | {:>7} | {:>7} |", label, "—", "—", "—");
            }
        } else {
            eprintln!(
                "| {:<24} | {:>8} | {:>7} | {:>7} |",
                label, "not run", "—", "—"
            );
        }
    }
    eprintln!();
}
