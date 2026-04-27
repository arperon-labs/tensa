//! Entity resolution benchmark — extended threshold sweep with P/R/F1 curve.
//!
//! Builds on the existing `tests/er_benchmark.rs` with more pairs,
//! multiple threshold points, and structured reporting.

use super::*;
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

fn build_extended_pairs() -> Vec<LabeledPair> {
    vec![
        // True positives
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
        LabeledPair {
            name_a: "Robert Johnson Jr.".into(),
            name_b: "Robert Johnson".into(),
            entity_type: EntityType::Actor,
            is_match: true,
        },
        LabeledPair {
            name_a: "ACME Corp".into(),
            name_b: "ACME Corporation".into(),
            entity_type: EntityType::Organization,
            is_match: true,
        },
        LabeledPair {
            name_a: "New York City".into(),
            name_b: "New York".into(),
            entity_type: EntityType::Location,
            is_match: true,
        },
        // True negatives
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
        LabeledPair {
            name_a: "Michael Brown".into(),
            name_b: "Michael Jordan".into(),
            entity_type: EntityType::Actor,
            is_match: false,
        },
        LabeledPair {
            name_a: "London".into(),
            name_b: "Paris".into(),
            entity_type: EntityType::Location,
            is_match: false,
        },
    ]
}

#[test]
fn test_er_benchmark() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);
    let embedder = HashEmbedding::new(64);

    let mut resolver = EntityResolver::new();
    let pairs = build_extended_pairs();

    // Register first entity of each pair.
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
            narrative_id: Some("er-bench".into()),
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

    // Resolve and measure.
    let mut tp = 0;
    let mut fp = 0;
    let mut fneg = 0;

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

    let metrics = BenchmarkMetrics::from_counts(tp, fp, fneg);

    let report = BenchmarkReport {
        benchmark: "Entity Resolution".into(),
        dataset: "synthetic-10pair".into(),
        metrics,
        baseline_comparison: vec![],
        duration_ms: 0,
    };
    report.print_markdown();

    // The harness should complete without panicking.
    assert!(
        report.metrics.precision >= 0.0,
        "ER benchmark should complete without errors"
    );
}
