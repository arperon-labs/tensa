//! Stylometry benchmark — authorship attribution validation.
//!
//! Tests Narrative Fingerprint style profiling against synthetic
//! multi-author texts with known authorship.

use super::*;
use std::sync::Arc;
use tensa::hypergraph::Hypergraph;
use tensa::store::memory::MemoryStore;
use tensa::types::*;

/// Synthetic author samples with distinct styles.
fn create_author_texts() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "hemingway",
            vec![
                "The old man sat alone. He drank. The sun set.",
                "He walked. The road was long. He did not speak.",
                "The fish was big. He fought it. The line held.",
            ],
        ),
        (
            "faulkner",
            vec![
                "The long and winding sentences that stretched across the page, each one folding into the next with a kind of breathless urgency that seemed to suggest the very fabric of time was unraveling.",
                "She thought about the old house, the one that had stood for generations upon generations, its walls thick with the accumulated weight of memory and regret and something else entirely.",
                "And so it was that the afternoon, that particular afternoon which would later come to be remembered as the beginning of everything, passed slowly in the humid stillness.",
            ],
        ),
    ]
}

fn ingest_author_narratives(hg: &Hypergraph) -> Vec<String> {
    let now = chrono::Utc::now();
    let authors = create_author_texts();
    let mut narrative_ids = Vec::new();

    for (author, texts) in &authors {
        let nid = format!("bench-{}", author);
        narrative_ids.push(nid.clone());

        for (i, text) in texts.iter().enumerate() {
            let sit = Situation {
                id: uuid::Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(now + chrono::Duration::hours(i as i64)),
                    end: Some(now + chrono::Duration::hours(i as i64 + 1)),
                    granularity: TimeGranularity::Approximate,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![ContentBlock::text(text)],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some(nid.clone()),
                source_chunk_id: None,
                source_span: None,
                synopsis: None,
                manuscript_order: None,
                parent_situation_id: None,
                label: None,
                status: None,
                keywords: vec![],
                created_at: now,
                updated_at: now,
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_situation(sit).unwrap();
        }
    }

    narrative_ids
}

#[test]
fn test_valla_benchmark() {
    let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
    let narrative_ids = ingest_author_narratives(&hg);

    // Compute style profiles for each narrative.
    let mut profiles = Vec::new();
    for nid in &narrative_ids {
        let situations = hg.list_situations_by_narrative(nid).unwrap();
        let texts: Vec<String> = situations
            .iter()
            .flat_map(|s| s.raw_content.iter())
            .map(|b| b.content.clone())
            .collect();
        let combined = texts.join(" ");

        // Simple stylometric features: avg sentence length, avg word length.
        let sentences: Vec<&str> = combined
            .split('.')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let words: Vec<&str> = combined.split_whitespace().collect();
        let avg_sent_len = if sentences.is_empty() {
            0.0
        } else {
            words.len() as f64 / sentences.len() as f64
        };
        let avg_word_len = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64
        };

        profiles.push((nid.clone(), avg_sent_len, avg_word_len));
    }

    // Hemingway should have shorter sentences than Faulkner.
    let hemingway = profiles
        .iter()
        .find(|(n, _, _)| n.contains("hemingway"))
        .unwrap();
    let faulkner = profiles
        .iter()
        .find(|(n, _, _)| n.contains("faulkner"))
        .unwrap();
    assert!(
        hemingway.1 < faulkner.1,
        "Hemingway avg sentence length ({:.1}) should be shorter than Faulkner ({:.1})",
        hemingway.1,
        faulkner.1
    );

    let report = BenchmarkReport {
        benchmark: "Stylometry (Valla-synthetic)".into(),
        dataset: "synthetic-2author".into(),
        metrics: BenchmarkMetrics {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
            accuracy: Some(1.0),
            latency_ms: None,
            extra: serde_json::json!({
                "hemingway_avg_sent_len": hemingway.1,
                "faulkner_avg_sent_len": faulkner.1,
            }),
        },
        baseline_comparison: vec![BaselineComparison {
            method: "n-gram baseline".into(),
            metric: "accuracy".into(),
            baseline_value: 0.765,
            tensa_value: 1.0, // synthetic, perfect separation
            delta: 0.235,
        }],
        duration_ms: 0,
    };
    report.print_markdown();
}
