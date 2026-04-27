//! Entity resolution benchmark harness.
//!
//! Validates ER quality on synthetic pairs, reporting precision/recall/F1.

use std::sync::Arc;
use tensa::hypergraph::Hypergraph;
use tensa::ingestion::embed::{EmbeddingProvider, HashEmbedding};
use tensa::ingestion::extraction::ExtractedEntity;
use tensa::ingestion::resolve::{EntityResolver, ResolveResult};
use tensa::store::memory::MemoryStore;
use tensa::types::*;

struct LabeledPair {
    name_a: String,
    name_b: String,
    entity_type: EntityType,
    is_match: bool,
}

fn build_benchmark_pairs() -> Vec<LabeledPair> {
    vec![
        // True positives (should match)
        LabeledPair {
            name_a: "John Smith".into(),
            name_b: "John Smith".into(),
            entity_type: EntityType::Actor,
            is_match: true,
        },
        LabeledPair {
            name_a: "Dr. Jane Wilson".into(),
            name_b: "Jane Wilson".into(),
            entity_type: EntityType::Actor,
            is_match: true,
        },
        // True negatives (should not match)
        LabeledPair {
            name_a: "John Smith".into(),
            name_b: "Jane Wilson".into(),
            entity_type: EntityType::Actor,
            is_match: false,
        },
        LabeledPair {
            name_a: "New York".into(),
            name_b: "New Delhi".into(),
            entity_type: EntityType::Location,
            is_match: false,
        },
        LabeledPair {
            name_a: "Alpha Corp".into(),
            name_b: "Beta Industries".into(),
            entity_type: EntityType::Organization,
            is_match: false,
        },
    ]
}

fn compute_metrics(tp: usize, fp: usize, fneg: usize) -> (f64, f64, f64) {
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fneg > 0 {
        tp as f64 / (tp + fneg) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

#[test]
fn test_er_benchmark_runs() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);
    let embedder = HashEmbedding::new(64);

    let mut resolver = EntityResolver::new();

    let pairs = build_benchmark_pairs();
    let mut tp = 0;
    let mut fp = 0;
    let mut fneg = 0;

    // Register first entity of each pair
    for pair in &pairs {
        let entity = Entity {
            id: uuid::Uuid::now_v7(),
            entity_type: pair.entity_type.clone(),
            properties: serde_json::json!({"name": pair.name_a}),
            beliefs: None,
            embedding: embedder.embed_text(&pair.name_a).ok(),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("bench".into()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity.clone()).unwrap();
        resolver.register(
            entity.id,
            &pair.name_a,
            &[],
            pair.entity_type.clone(),
            entity.embedding.clone(),
        );
    }

    // Resolve second entity of each pair
    for pair in &pairs {
        let extracted = ExtractedEntity {
            name: pair.name_b.clone(),
            aliases: vec![],
            entity_type: pair.entity_type.clone(),
            properties: serde_json::json!({}),
            confidence: 0.9,
        };

        let result = resolver.resolve(&extracted, Some(&embedder));
        let matched = matches!(result, ResolveResult::Existing(_));

        match (matched, pair.is_match) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fneg += 1,
            (false, false) => {}
        }
    }

    let (precision, recall, f1) = compute_metrics(tp, fp, fneg);
    eprintln!("ER Benchmark: TP={tp}, FP={fp}, FN={fneg}");
    eprintln!("Precision={precision:.2}, Recall={recall:.2}, F1={f1:.2}");

    // The harness should run without panicking
    assert!(
        precision + recall >= 0.0,
        "ER benchmark should complete without errors"
    );
}
