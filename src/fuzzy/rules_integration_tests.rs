//! Fuzzy Sprint Phase 9 — Mamdani rule system integration tests.
//!
//! Pulled out of [`super::rules_tests`] so that file stays under the
//! 500-line cap. Covers the hypergraph + TensaQL + post-ingest helper
//! paths, which bring heavier fixtures than the pure-numerics tests.
//!
//! Cites: [mamdani1975mamdani].

use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::fuzzy::rules::{
    build_rule, evaluate_rules_against_entity, evaluate_rules_over_narrative, save_rule,
    FuzzyCondition, FuzzyOutput, MembershipFunction, Rule,
};
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::types::{Entity, EntityType, MaturityLevel};

// ── Shared fixtures (same shape as rules_tests.rs — deliberate ────────────────
// duplication to keep the two files independently readable) ─────────────────

fn sample_entity(props: serde_json::Value) -> Entity {
    Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: props,
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some("phase9-test".into()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    }
}

fn elevated_risk_rule() -> Rule {
    build_rule(
        "elevated-disinfo-risk",
        "phase9-test",
        vec![
            FuzzyCondition {
                variable_path: "entity.properties.political_alignment".into(),
                membership: MembershipFunction::Triangular {
                    a: 0.6,
                    b: 0.9,
                    c: 1.0,
                },
                linguistic_term: "highly-partisan".into(),
            },
            FuzzyCondition {
                variable_path: "entity.properties.inflammatory_score".into(),
                membership: MembershipFunction::Trapezoidal {
                    a: 0.5,
                    b: 0.7,
                    c: 0.9,
                    d: 1.0,
                },
                linguistic_term: "high".into(),
            },
        ],
        FuzzyOutput {
            variable: "disinfo_risk".into(),
            membership: MembershipFunction::Gaussian {
                mean: 0.8,
                sigma: 0.1,
            },
            linguistic_term: "elevated".into(),
        },
    )
}

// ── Integration tests ──────────────────────────────────────────────────────

#[test]
fn test_evaluate_rules_against_entity_hypergraph_path() {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let rule = elevated_risk_rule();
    let rule_id = rule.id;
    save_rule(store.as_ref(), "phase9-test", &rule).unwrap();

    let ent = sample_entity(serde_json::json!({
        "political_alignment": 0.85,
        "inflammatory_score": 0.8,
    }));
    let eid = hg.create_entity(ent).unwrap();
    let eval = evaluate_rules_against_entity(
        &hg,
        "phase9-test",
        Some(&[rule_id.to_string()]),
        &eid,
    )
    .unwrap();
    assert_eq!(eval.fired_rules.len(), 1);
    assert!(eval.fired_rules[0].firing_strength > 0.0);
    assert!(eval.defuzzified_output.is_some());
}

#[test]
fn test_tensaql_evaluate_rules_parses_and_plans() {
    use crate::query::parser::{parse_statement, TensaStatement};
    use crate::query::planner::{plan_statement, PlanStep, QueryClass};

    let stmt = parse_statement(
        r#"EVALUATE RULES FOR "phase9-test" AGAINST (e:Actor) RULES ['01930000-0000-7000-8000-000000000001']"#,
    )
    .expect("EVALUATE RULES must parse");
    match &stmt {
        TensaStatement::EvaluateRules {
            narrative_id,
            entity_type,
            rule_ids,
            ..
        } => {
            assert_eq!(narrative_id, "phase9-test");
            assert_eq!(entity_type, "Actor");
            assert_eq!(rule_ids.as_ref().map(|v| v.len()), Some(1));
        }
        other => panic!("unexpected AST variant: {:?}", other),
    }
    let plan = plan_statement(&stmt).expect("must plan");
    assert!(matches!(plan.class, QueryClass::Instant));
    assert!(matches!(
        plan.steps.first().unwrap(),
        PlanStep::RunEvaluateRules { .. }
    ));
}

#[test]
fn test_post_ingest_mamdani_helper_tags_matching_entities() {
    use crate::ingestion::pipeline::{IngestionPipeline, PipelineConfig};
    use crate::ingestion::queue::ValidationQueue;

    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let rule = elevated_risk_rule();
    let rule_id = rule.id;
    save_rule(store.as_ref(), "phase9-test", &rule).unwrap();

    let mut ent = sample_entity(serde_json::json!({
        "political_alignment": 0.85,
        "inflammatory_score": 0.8,
    }));
    ent.narrative_id = Some("phase9-test".into());
    let eid = hg.create_entity(ent).unwrap();

    let hg_arc = Arc::new(hg);
    let config = PipelineConfig {
        narrative_id: Some("phase9-test".into()),
        post_ingest_mamdani_rule_id: Some(rule_id.to_string()),
        ..Default::default()
    };
    let pipeline = IngestionPipeline::new(
        hg_arc.clone(),
        Arc::new(DummyExtractor),
        None,
        None,
        Arc::new(ValidationQueue::new(store.clone())),
        config,
    );
    let tagged = pipeline
        .apply_post_ingest_mamdani("phase9-test", &[eid])
        .expect("apply");
    assert_eq!(tagged, 1);
    let reloaded = hg_arc.get_entity(&eid).unwrap();
    let mamdani = reloaded.properties.get("mamdani").expect("mamdani tag");
    let strength = mamdani
        .get("firing_strength")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(strength > 0.0, "firing strength should be positive");
}

/// Minimal no-op extractor used by the post-ingest helper test — we
/// never actually run the ingest loop, just construct the pipeline so
/// we can call the Phase 9 helper.
struct DummyExtractor;
impl crate::ingestion::llm::NarrativeExtractor for DummyExtractor {
    fn extract_narrative(
        &self,
        _chunk: &crate::ingestion::chunker::TextChunk,
    ) -> crate::error::Result<crate::ingestion::extraction::NarrativeExtraction> {
        Err(crate::error::TensaError::LlmError(
            "dummy — not used".into(),
        ))
    }
}

#[test]
fn test_evaluate_rules_over_narrative_filters_entity_type() {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let rule = elevated_risk_rule();
    save_rule(store.as_ref(), "phase9-test", &rule).unwrap();
    for _ in 0..2 {
        let e = sample_entity(serde_json::json!({
            "political_alignment": 0.85,
            "inflammatory_score": 0.8,
        }));
        hg.create_entity(e).unwrap();
    }
    let mut loc = sample_entity(serde_json::json!({
        "political_alignment": 0.85,
        "inflammatory_score": 0.8,
    }));
    loc.entity_type = EntityType::Location;
    hg.create_entity(loc).unwrap();

    let evs = evaluate_rules_over_narrative(
        &hg,
        "phase9-test",
        None,
        Some(EntityType::Actor),
    )
    .unwrap();
    assert_eq!(evs.len(), 2);
    for ev in &evs {
        assert!(ev.fired_rules.iter().any(|f| f.firing_strength > 0.0));
    }
}
