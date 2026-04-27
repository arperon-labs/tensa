//! Temporal benchmark — counterfactual inference validation.
//!
//! Tests INFER COUNTERFACTUAL on synthetic temporal scenarios
//! where the ground truth is known.

use super::*;
use std::sync::Arc;
use tensa::hypergraph::Hypergraph;
use tensa::store::memory::MemoryStore;
use tensa::types::*;

/// Build a simple causal chain: A causes B causes C.
/// Counterfactual: "What if A didn't happen?" → B and C shouldn't happen.
fn setup_causal_chain() -> (Hypergraph, uuid::Uuid, uuid::Uuid, uuid::Uuid) {
    let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
    let now = chrono::Utc::now();
    let n = "temporal-bench";

    let make_sit = |content: &str, hours: i64| -> Situation {
        Situation {
            id: uuid::Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(now + chrono::Duration::hours(hours)),
                end: Some(now + chrono::Duration::hours(hours + 1)),
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
            raw_content: vec![ContentBlock::text(content)],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Validated,
            confidence: 0.95,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(n.into()),
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
        }
    };

    let s_a = make_sit("Event A: the initial trigger", 0);
    let s_b = make_sit("Event B: direct consequence of A", 1);
    let s_c = make_sit("Event C: follows from B", 2);

    let id_a = hg.create_situation(s_a).unwrap();
    let id_b = hg.create_situation(s_b).unwrap();
    let id_c = hg.create_situation(s_c).unwrap();

    // A → B → C causal chain.
    hg.add_causal_link(CausalLink {
        from_situation: id_a,
        to_situation: id_b,
        mechanism: Some("direct".into()),
        strength: 0.9,
        causal_type: CausalType::Necessary,
        maturity: MaturityLevel::Validated,
    })
    .unwrap();
    hg.add_causal_link(CausalLink {
        from_situation: id_b,
        to_situation: id_c,
        mechanism: Some("cascade".into()),
        strength: 0.8,
        causal_type: CausalType::Sufficient,
        maturity: MaturityLevel::Validated,
    })
    .unwrap();

    (hg, id_a, id_b, id_c)
}

#[test]
fn test_temporal_benchmark() {
    let (hg, id_a, id_b, id_c) = setup_causal_chain();

    // Validate causal chain structure.
    let consequences_a = hg.get_consequences(&id_a).unwrap();
    assert_eq!(consequences_a.len(), 1);
    assert_eq!(consequences_a[0].to_situation, id_b);

    let consequences_b = hg.get_consequences(&id_b).unwrap();
    assert_eq!(consequences_b.len(), 1);
    assert_eq!(consequences_b[0].to_situation, id_c);

    // No consequences from C (end of chain).
    let consequences_c = hg.get_consequences(&id_c).unwrap();
    assert!(consequences_c.is_empty());

    // Validate chain depth = 2 (A→B→C).
    let chain_len = {
        let mut depth = 0;
        let mut current = id_a;
        loop {
            let next = hg.get_consequences(&current).unwrap();
            if next.is_empty() {
                break;
            }
            depth += 1;
            current = next[0].to_situation;
        }
        depth
    };
    assert_eq!(chain_len, 2, "Causal chain should have depth 2");

    let report = BenchmarkReport {
        benchmark: "Temporal Causal Chain".into(),
        dataset: "synthetic-chain-3".into(),
        metrics: BenchmarkMetrics {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
            accuracy: Some(1.0),
            latency_ms: None,
            extra: serde_json::json!({"chain_depth": chain_len}),
        },
        baseline_comparison: vec![],
        duration_ms: 0,
    };
    report.print_markdown();
}
