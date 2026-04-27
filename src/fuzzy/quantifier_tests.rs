//! Fuzzy Sprint Phase 6 — tests for the intermediate-quantifier surface.
//!
//! Coverage:
//! * 4 × 7 value-point assertions (28 total) for `Q_most / Q_many /
//!   Q_almost_all / Q_few`.
//! * Domain restriction (Actor-only vs unrestricted).
//! * Empty domain → `Q(0) = 0` for all four.
//! * TensaQL grammar: QUANTIFY parses + plans + executes.
//! * Planner rejects unknown quantifier names.
//! * KV cache round-trip.
//! * AlertRule quantifier_condition serde round-trip.
//!
//! Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].

use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::fuzzy::quantifier::{
    delete_quantifier_result, evaluate, evaluate_over_entities, evaluate_over_situations,
    list_quantifier_results_for_narrative, load_quantifier_result, predicate_hash,
    quantifier_from_name, save_quantifier_result, QuantifierCondition, QuantifierResult,
    Quantifier,
};
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::types::*;

// ── Tolerance for ramp interior points (exact-rational → f64 drift). ───────
const EPS: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

// ── Test fixture: narrative with Actor + Location entities ─────────────────

fn make_hg() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn mk_entity(nid: &str, et: EntityType, confidence: f32, name: &str) -> Entity {
    Entity {
        id: Uuid::now_v7(),
        entity_type: et,
        properties: serde_json::json!({"name": name}),
        beliefs: None,
        embedding: None,
        narrative_id: Some(nid.into()),
        maturity: MaturityLevel::Candidate,
        confidence,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        deleted_at: None,
        transaction_time: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

// ── Value-point tests: Q_most ───────────────────────────────────────────────

#[test]
fn test_q_most_value_points() {
    // Ramp: 0 at r ≤ 0.3, linear to 1.0 at r ≥ 0.8.
    assert!(close(evaluate(Quantifier::Most, 0.0), 0.0));
    assert!(close(evaluate(Quantifier::Most, 0.2), 0.0));
    assert!(close(evaluate(Quantifier::Most, 0.3), 0.0));
    // At r = 0.55 (midpoint), Q_most = (0.55 - 0.3) / 0.5 = 0.5.
    assert!(close(evaluate(Quantifier::Most, 0.55), 0.5));
    assert!(close(evaluate(Quantifier::Most, 0.8), 1.0));
    assert!(close(evaluate(Quantifier::Most, 0.9), 1.0));
    assert!(close(evaluate(Quantifier::Most, 1.0), 1.0));
}

// ── Q_many ─────────────────────────────────────────────────────────────────

#[test]
fn test_q_many_value_points() {
    // Ramp: 0 at r ≤ 0.1, linear to 1.0 at r ≥ 0.5.
    assert!(close(evaluate(Quantifier::Many, 0.0), 0.0));
    assert!(close(evaluate(Quantifier::Many, 0.05), 0.0));
    assert!(close(evaluate(Quantifier::Many, 0.1), 0.0));
    // r = 0.3 (midpoint) -> (0.3 - 0.1) / 0.4 = 0.5.
    assert!(close(evaluate(Quantifier::Many, 0.3), 0.5));
    assert!(close(evaluate(Quantifier::Many, 0.5), 1.0));
    assert!(close(evaluate(Quantifier::Many, 0.75), 1.0));
    assert!(close(evaluate(Quantifier::Many, 1.0), 1.0));
}

// ── Q_almost_all ───────────────────────────────────────────────────────────

#[test]
fn test_q_almost_all_value_points() {
    // Ramp: 0 at r ≤ 0.7, linear to 1.0 at r ≥ 0.95.
    assert!(close(evaluate(Quantifier::AlmostAll, 0.0), 0.0));
    assert!(close(evaluate(Quantifier::AlmostAll, 0.5), 0.0));
    assert!(close(evaluate(Quantifier::AlmostAll, 0.7), 0.0));
    // r = 0.825 -> (0.825 - 0.7) / 0.25 = 0.5.
    assert!(close(evaluate(Quantifier::AlmostAll, 0.825), 0.5));
    assert!(close(evaluate(Quantifier::AlmostAll, 0.95), 1.0));
    assert!(close(evaluate(Quantifier::AlmostAll, 0.99), 1.0));
    assert!(close(evaluate(Quantifier::AlmostAll, 1.0), 1.0));
}

// ── Q_few = 1 - Q_many ─────────────────────────────────────────────────────

#[test]
fn test_q_few_value_points() {
    // Q_few(r) = 1 - Q_many(r): highest at low r, 0 at r ≥ 0.5.
    assert!(close(evaluate(Quantifier::Few, 0.0), 1.0));
    assert!(close(evaluate(Quantifier::Few, 0.05), 1.0));
    assert!(close(evaluate(Quantifier::Few, 0.1), 1.0));
    // r = 0.3 (midpoint) -> 1 - 0.5 = 0.5.
    assert!(close(evaluate(Quantifier::Few, 0.3), 0.5));
    assert!(close(evaluate(Quantifier::Few, 0.5), 0.0));
    assert!(close(evaluate(Quantifier::Few, 0.75), 0.0));
    assert!(close(evaluate(Quantifier::Few, 1.0), 0.0));
}

// ── Domain restriction: Actor-only vs unrestricted ────────────────────────

#[test]
fn test_domain_restriction_actors_only() {
    let hg = make_hg();
    let nid = "phase6-domain";
    // 5 Actors, 5 Locations — predicate fires on all Actors.
    for i in 0..5 {
        hg.create_entity(mk_entity(nid, EntityType::Actor, 0.9, &format!("actor-{i}")))
            .unwrap();
        hg.create_entity(mk_entity(
            nid,
            EntityType::Location,
            0.4,
            &format!("loc-{i}"),
        ))
        .unwrap();
    }

    // Actor-restricted: all 5 fire → r = 1.0 → Q(1.0) = 1.0 for Most.
    let actor_restricted = evaluate_over_entities(
        &hg,
        nid,
        Some(EntityType::Actor),
        |e| {
            if e.confidence > 0.8 {
                1.0
            } else {
                0.0
            }
        },
        Quantifier::Most,
    )
    .unwrap();
    assert!(close(actor_restricted, 1.0));

    // Unrestricted: 5 of 10 fire → r = 0.5 → Q_most(0.5) = (0.5-0.3)/0.5 = 0.4.
    let unrestricted = evaluate_over_entities(
        &hg,
        nid,
        None,
        |e| {
            if e.confidence > 0.8 {
                1.0
            } else {
                0.0
            }
        },
        Quantifier::Most,
    )
    .unwrap();
    assert!(close(unrestricted, 0.4));
}

// ── Empty domain: r = 0 → Q(0) = 0 for all four ───────────────────────────

#[test]
fn test_empty_domain() {
    let hg = make_hg();
    let nid = "phase6-empty";
    // No entities created.
    for q in [
        Quantifier::Most,
        Quantifier::Many,
        Quantifier::AlmostAll,
    ] {
        let v = evaluate_over_entities(&hg, nid, None, |_| 1.0, q).unwrap();
        assert!(close(v, 0.0), "{:?} at r=0 should be 0", q);
    }
    // Q_few(0) = 1 - 0 = 1.0. Keep this contract explicit.
    let v_few = evaluate_over_entities(&hg, nid, None, |_| 1.0, Quantifier::Few).unwrap();
    assert!(close(v_few, 1.0));

    // Same for situation domain (no situations).
    let v_sit =
        evaluate_over_situations(&hg, nid, |_| 1.0, Quantifier::Most).unwrap();
    assert!(close(v_sit, 0.0));
}

// ── TensaQL grammar: parse, plan, execute ─────────────────────────────────

#[test]
fn test_grammar_quantify_parses() {
    let stmt = crate::query::parser::parse_statement(
        r#"QUANTIFY MOST (e:Actor) WHERE e.confidence > 0.7 FOR "n1" AS "high_conf""#,
    )
    .expect("parse");
    match stmt {
        crate::query::parser::TensaStatement::Quantify {
            quantifier,
            binding,
            type_name,
            narrative_id,
            label,
            ..
        } => {
            assert_eq!(quantifier, "most");
            assert_eq!(binding.as_deref(), Some("e"));
            assert_eq!(type_name, "Actor");
            assert_eq!(narrative_id.as_deref(), Some("n1"));
            assert_eq!(label.as_deref(), Some("high_conf"));
        }
        other => panic!("Expected Quantify, got: {other:?}"),
    }
}

#[test]
fn test_planner_rejects_unknown_quantifier() {
    // Parse a fake quantifier name (grammar gates this, so drive through
    // the parser-planner API directly).
    let stmt = crate::query::parser::TensaStatement::Quantify {
        quantifier: "plurality".into(), // unknown
        binding: Some("e".into()),
        type_name: "Actor".into(),
        where_clause: None,
        narrative_id: Some("n1".into()),
        label: None,
    };
    let err = crate::query::planner::plan_statement(&stmt);
    assert!(err.is_err(), "expected plan_statement to reject 'plurality'");
}

#[test]
fn test_executor_e2e() {
    let hg = make_hg();
    let nid = "phase6-e2e";
    // 5 Actors, 3 above the WHERE threshold (confidence > 0.7).
    hg.create_entity(mk_entity(nid, EntityType::Actor, 0.95, "a1"))
        .unwrap();
    hg.create_entity(mk_entity(nid, EntityType::Actor, 0.9, "a2"))
        .unwrap();
    hg.create_entity(mk_entity(nid, EntityType::Actor, 0.85, "a3"))
        .unwrap();
    hg.create_entity(mk_entity(nid, EntityType::Actor, 0.4, "a4"))
        .unwrap();
    hg.create_entity(mk_entity(nid, EntityType::Actor, 0.3, "a5"))
        .unwrap();

    let stmt = crate::query::parser::parse_statement(
        r#"QUANTIFY MOST (e:Actor) WHERE e.confidence > 0.7 FOR "phase6-e2e""#,
    )
    .unwrap();
    let plan = crate::query::planner::plan_statement(&stmt).unwrap();
    let interval_tree = crate::temporal::index::IntervalTree::new();
    let rows = crate::query::executor::execute(&plan, &hg, &interval_tree).unwrap();
    assert_eq!(rows.len(), 1);

    // r = 3/5 = 0.6 → Q_most(0.6) = (0.6 - 0.3) / 0.5 = 0.6.
    let value = rows[0]
        .get("quantifier_result")
        .and_then(|v| v.as_f64())
        .expect("quantifier_result column missing");
    assert!((value - 0.6).abs() < 1e-9, "expected 0.6, got {value}");

    let r_diag = rows[0]
        .get("_cardinality_ratio")
        .and_then(|v| v.as_f64())
        .expect("_cardinality_ratio missing");
    assert!((r_diag - 0.6).abs() < 1e-9);
}

// ── KV cache round-trip ───────────────────────────────────────────────────

#[test]
fn test_kv_roundtrip() {
    let hg = make_hg();
    let nid = "phase6-kv";
    let hash = predicate_hash("confidence>0.7", Quantifier::Most, Some("Actor"));
    let result = QuantifierResult {
        quantifier: "most".into(),
        value: 0.6,
        label: Some("high-conf".into()),
    };
    save_quantifier_result(hg.store(), nid, &hash, &result).unwrap();

    let loaded = load_quantifier_result(hg.store(), nid, &hash).unwrap();
    assert_eq!(loaded, Some(result.clone()));

    let listed = list_quantifier_results_for_narrative(hg.store(), nid).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].0, hash);
    assert_eq!(listed[0].1.value, 0.6);

    delete_quantifier_result(hg.store(), nid, &hash).unwrap();
    let after = load_quantifier_result(hg.store(), nid, &hash).unwrap();
    assert!(after.is_none());
}

// ── AlertRule quantifier_condition serde round-trip ────────────────────────

#[test]
fn test_alerts_rule_with_quantifier_condition_serde_roundtrip() {
    use crate::analysis::alerts::{AlertRule, AlertRuleType};
    let qc = QuantifierCondition {
        predicate: "confidence>0.7".into(),
        quantifier: "most".into(),
        threshold: 0.5,
    };
    let rule = AlertRule {
        id: "r42".into(),
        narrative_id: Some("n1".into()),
        rule_type: AlertRuleType::ConfidenceDrop { entity_type: None },
        threshold: 0.3,
        enabled: true,
        created_at: Utc::now(),
        aggregator: None,
        quantifier_condition: Some(qc.clone()),
        mamdani_rule_id: None,
    };
    let json = serde_json::to_string(&rule).unwrap();
    let parsed: AlertRule = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.quantifier_condition, Some(qc));

    // Legacy payload without the field should still deserialize (serde default).
    let legacy = r#"{
        "id": "r43",
        "narrative_id": null,
        "rule_type": {"TrustDrop": {"source_id": null}},
        "threshold": 0.5,
        "enabled": true,
        "created_at": "2026-01-01T00:00:00Z"
    }"#;
    let legacy_parsed: AlertRule = serde_json::from_str(legacy).unwrap();
    assert!(legacy_parsed.quantifier_condition.is_none());
    assert!(legacy_parsed.aggregator.is_none());
}

// ── quantifier_from_name resolver ─────────────────────────────────────────

#[test]
fn test_quantifier_from_name_resolves_all_variants() {
    assert_eq!(
        quantifier_from_name("MOST").unwrap(),
        Quantifier::Most
    );
    assert_eq!(
        quantifier_from_name("most").unwrap(),
        Quantifier::Most
    );
    assert_eq!(
        quantifier_from_name("Many").unwrap(),
        Quantifier::Many
    );
    assert_eq!(
        quantifier_from_name("almost_all").unwrap(),
        Quantifier::AlmostAll
    );
    assert_eq!(
        quantifier_from_name("almost-all").unwrap(),
        Quantifier::AlmostAll
    );
    assert_eq!(
        quantifier_from_name("few").unwrap(),
        Quantifier::Few
    );
    assert!(quantifier_from_name("plurality").is_err());
    assert!(quantifier_from_name("").is_err());
}
