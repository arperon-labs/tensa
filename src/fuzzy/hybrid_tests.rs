//! Fuzzy Sprint Phase 10 — hybrid fuzzy-probability tests.
//!
//! Coverage (six tests minimum):
//! 1. Quantifier-predicate path against a 5-entity fixture under a
//!    uniform distribution (hand-verified).
//! 2. MamdaniRule-predicate path against a 3-entity fixture under a
//!    skewed distribution (hand-verified).
//! 3. Custom-predicate path with pre-computed μ (exact equality).
//! 4. Invalid distribution: sum ≠ 1 rejects with `InvalidInput`.
//! 5. Empty distribution short-circuits to 0.
//! 6. KV round-trip: save → load → delete → list.
//!
//! Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::fuzzy::hybrid::{
    build_hybrid_report, delete_hybrid_result, fuzzy_probability, list_hybrid_results_for_narrative,
    load_hybrid_result, save_hybrid_result, FuzzyEvent, FuzzyEventPredicate, ProbDist,
};
use crate::fuzzy::rules::{build_rule, save_rule, FuzzyCondition, FuzzyOutput, MembershipFunction};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::types::{Entity, EntityType, MaturityLevel};

// ── Test fixtures ───────────────────────────────────────────────────────────

fn make_hg() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn mk_entity(nid: &str, et: EntityType, confidence: f32, props: serde_json::Value) -> Entity {
    Entity {
        id: Uuid::now_v7(),
        entity_type: et,
        properties: props,
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

/// 5-actor fixture — confidences and partisan scores per §3 worked
/// example in `docs/fuzzy_hybrid_algorithm.md`.
fn fixture_5_actors(hg: &Hypergraph, nid: &str) -> Vec<Uuid> {
    let rows: &[(f32, f64)] = &[
        (0.80, 0.95),
        (0.70, 0.90),
        (0.85, 0.60),
        (0.90, 0.20),
        (0.95, 0.10),
    ];
    let mut ids = Vec::with_capacity(rows.len());
    for (conf, partisan) in rows {
        let mut e = mk_entity(
            nid,
            EntityType::Actor,
            *conf,
            serde_json::json!({"partisan_score": partisan}),
        );
        let id = hg.create_entity(e.clone()).unwrap();
        e.id = id;
        ids.push(id);
    }
    ids
}

// ── T1 — Quantifier predicate ─────────────────────────────────────────────

#[test]
fn test_fuzzy_probability_with_quantifier_predicate() {
    let hg = make_hg();
    let nid = "phase10-quantifier";
    let ids = fixture_5_actors(&hg, nid);
    // Uniform distribution over all 5 actors.
    let outcomes: Vec<(Uuid, f64)> = ids.iter().map(|id| (*id, 0.2)).collect();
    let dist = ProbDist::Discrete { outcomes };
    // WHERE = "partisan_score > 0.5" — 3/5 actors pass (Alice, Bob, Carol).
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Quantifier,
        predicate_payload: serde_json::json!({
            "quantifier": "most",
            "where": "partisan_score>0.5",
            "entity_type": "Actor",
        }),
    };
    let p = fuzzy_probability(&hg, nid, &event, &dist, TNormKind::Godel).unwrap();
    // 3 actors × 0.2 = 0.6.
    assert!((p - 0.6).abs() < 1e-9, "expected 0.6, got {p}");

    // And with "confidence > 0.5" every actor passes → P_fuzzy = 1.0.
    let event_all = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Quantifier,
        predicate_payload: serde_json::json!({
            "quantifier": "most",
            "where": "confidence>0.5",
            "entity_type": "Actor",
        }),
    };
    let p_all = fuzzy_probability(&hg, nid, &event_all, &dist, TNormKind::Godel).unwrap();
    assert!((p_all - 1.0).abs() < 1e-9, "expected 1.0, got {p_all}");
}

// ── T2 — MamdaniRule predicate ────────────────────────────────────────────

#[test]
fn test_fuzzy_probability_with_mamdani_predicate() {
    let hg = make_hg();
    let nid = "phase10-mamdani";

    // 3 entities with varied inflammatory_score so the Trapezoidal
    // membership gives distinct firing strengths.
    let e_hi = mk_entity(
        nid,
        EntityType::Actor,
        0.9,
        serde_json::json!({"inflammatory_score": 0.95}),
    );
    let e_mid = mk_entity(
        nid,
        EntityType::Actor,
        0.9,
        serde_json::json!({"inflammatory_score": 0.6}),
    );
    let e_lo = mk_entity(
        nid,
        EntityType::Actor,
        0.9,
        serde_json::json!({"inflammatory_score": 0.0}),
    );
    let id_hi = hg.create_entity(e_hi).unwrap();
    let id_mid = hg.create_entity(e_mid).unwrap();
    let id_lo = hg.create_entity(e_lo).unwrap();

    // Rule: IF inflammatory_score IS Trapezoidal(0.5, 0.7, 0.9, 1.0) high
    //       THEN disinfo_risk IS Gaussian(0.8, 0.1) elevated.
    let rule = build_rule(
        "elevated-disinfo-risk",
        nid,
        vec![FuzzyCondition {
            variable_path: "entity.properties.inflammatory_score".into(),
            membership: MembershipFunction::Trapezoidal {
                a: 0.5,
                b: 0.7,
                c: 0.9,
                d: 1.0,
            },
            linguistic_term: "high".into(),
        }],
        FuzzyOutput {
            variable: "disinfo_risk".into(),
            membership: MembershipFunction::Gaussian {
                mean: 0.8,
                sigma: 0.1,
            },
            linguistic_term: "elevated".into(),
        },
    );
    let rule_id = rule.id;
    save_rule(hg.store(), nid, &rule).unwrap();

    // Hand-verified firing strengths:
    //   inflammatory_score = 0.95 → trapezoid plateau region [0.7, 0.9] is
    //     to the left; 0.95 is on the falling leg [0.9, 1.0]:
    //     (1.0 - 0.95) / (1.0 - 0.9) = 0.5.
    //   inflammatory_score = 0.6 → rising leg [0.5, 0.7]: (0.6 - 0.5) / 0.2 = 0.5.
    //   inflammatory_score = 0.0 → below a, μ = 0.
    //
    // Distribution skewed to the high entity: P(hi)=0.5, P(mid)=0.3, P(lo)=0.2.
    //
    // P_fuzzy = 0.5·0.5 + 0.5·0.3 + 0.0·0.2 = 0.25 + 0.15 = 0.40.
    let dist = ProbDist::Discrete {
        outcomes: vec![(id_hi, 0.5), (id_mid, 0.3), (id_lo, 0.2)],
    };
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::MamdaniRule,
        predicate_payload: serde_json::json!({
            "rule_id": rule_id.to_string(),
            "narrative_id": nid,
        }),
    };
    let p = fuzzy_probability(&hg, nid, &event, &dist, TNormKind::Godel).unwrap();
    assert!(
        (p - 0.40).abs() < 1e-9,
        "expected 0.40 from hand-verified Σ μ·P, got {p}"
    );
}

// ── T3 — Custom predicate ─────────────────────────────────────────────────

#[test]
fn test_fuzzy_probability_custom_predicate() {
    let hg = make_hg();
    let nid = "phase10-custom";
    let id_a = Uuid::now_v7();
    let id_b = Uuid::now_v7();
    let id_c = Uuid::now_v7();

    let mut mu = HashMap::new();
    mu.insert(id_a.to_string(), 0.7_f64);
    mu.insert(id_b.to_string(), 0.3_f64);
    mu.insert(id_c.to_string(), 1.0_f64);
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Custom,
        predicate_payload: serde_json::json!({ "memberships": mu }),
    };
    // P weights chosen so the expected = exact float.
    let dist = ProbDist::Discrete {
        outcomes: vec![(id_a, 0.5), (id_b, 0.25), (id_c, 0.25)],
    };
    // Σ μ·P = 0.7·0.5 + 0.3·0.25 + 1.0·0.25 = 0.35 + 0.075 + 0.25 = 0.675.
    let p = fuzzy_probability(&hg, nid, &event, &dist, TNormKind::Godel).unwrap();
    assert!((p - 0.675).abs() < 1e-9, "expected 0.675, got {p}");
}

// ── T4 — Distribution validation ──────────────────────────────────────────

#[test]
fn test_distribution_sum_not_one_rejects() {
    let hg = make_hg();
    let id = Uuid::now_v7();
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Custom,
        predicate_payload: serde_json::json!({ "memberships": { id.to_string(): 0.5 } }),
    };
    // Σ P = 0.8, not 1.0 → rejects with InvalidInput.
    let dist = ProbDist::Discrete {
        outcomes: vec![(id, 0.8)],
    };
    let err = fuzzy_probability(&hg, "phase10-reject", &event, &dist, TNormKind::Godel)
        .expect_err("expected InvalidInput for Σ P ≠ 1");
    let msg = format!("{err}");
    assert!(
        msg.contains("sum to 1.0"),
        "error should mention sum-to-one constraint, got: {msg}"
    );

    // Negative probability also rejects.
    let dist_neg = ProbDist::Discrete {
        outcomes: vec![(id, -0.1), (Uuid::now_v7(), 1.1)],
    };
    assert!(fuzzy_probability(&hg, "phase10-reject", &event, &dist_neg, TNormKind::Godel).is_err());
}

// ── T5 — Empty distribution ───────────────────────────────────────────────

#[test]
fn test_empty_distribution_returns_zero() {
    let hg = make_hg();
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Custom,
        predicate_payload: serde_json::json!({ "memberships": {} }),
    };
    let dist = ProbDist::Discrete { outcomes: vec![] };
    let p =
        fuzzy_probability(&hg, "phase10-empty", &event, &dist, TNormKind::Godel).unwrap();
    assert_eq!(p, 0.0);
}

// ── T6 — KV round-trip ────────────────────────────────────────────────────

#[test]
fn test_kv_roundtrip() {
    let store = MemoryStore::new();
    let nid = "phase10-kv";

    // Build a minimal report we can round-trip without running
    // fuzzy_probability — the semantics are exercised in T1..T3.
    let id = Uuid::now_v7();
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Custom,
        predicate_payload: serde_json::json!({ "memberships": { id.to_string(): 0.5 } }),
    };
    let dist = ProbDist::Discrete {
        outcomes: vec![(id, 1.0)],
    };
    let report = build_hybrid_report(nid, &event, &dist, TNormKind::Godel, 0.5);
    let query_id = report.query_id;
    save_hybrid_result(&store, nid, &report).unwrap();

    let loaded = load_hybrid_result(&store, nid, &query_id).unwrap();
    assert!(loaded.is_some());
    let loaded = loaded.unwrap();
    assert_eq!(loaded.query_id, query_id);
    assert_eq!(loaded.narrative_id, nid);
    assert_eq!(loaded.event_kind, "custom");
    assert_eq!(loaded.distribution_summary, "discrete:1");
    assert!((loaded.value - 0.5).abs() < 1e-12);

    let listed = list_hybrid_results_for_narrative(&store, nid).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].query_id, query_id);

    delete_hybrid_result(&store, nid, &query_id).unwrap();
    assert!(load_hybrid_result(&store, nid, &query_id)
        .unwrap()
        .is_none());
}

// ── Bonus coverage: duplicate UUID rejected ───────────────────────────────

#[test]
fn test_duplicate_uuid_rejects() {
    let hg = make_hg();
    let id = Uuid::now_v7();
    let event = FuzzyEvent {
        predicate_kind: FuzzyEventPredicate::Custom,
        predicate_payload: serde_json::json!({ "memberships": { id.to_string(): 1.0 } }),
    };
    let dist = ProbDist::Discrete {
        outcomes: vec![(id, 0.5), (id, 0.5)],
    };
    let err = fuzzy_probability(&hg, "phase10-dup", &event, &dist, TNormKind::Godel)
        .expect_err("expected duplicate UUID rejection");
    assert!(format!("{err}").contains("duplicate"));
}
