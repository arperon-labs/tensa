//! Tests for the Phase 7 graded-syllogism verifier.
//!
//! Every test uses a small in-memory fixture: a handful of entities of a
//! given type under a narrative_id. The default [`TypePredicateResolver`]
//! handles `"type:Actor"` / `"type:Location"` / ... predicate ids; a
//! custom resolver is exercised in
//! [`test_non_canonical_figure_fallback`] to prove the trait plugs.
//!
//! Acceptance criteria per Phase 7 docs:
//! * ≥ 10 test functions (we ship 11).
//! * 5 Peterson figures cover the spec (I / I* / II / III / IV each
//!   have a dedicated test, plus a non-canonical fallback).
//! * Prototype status documented — tests rely on the Gödel t-norm as
//!   default + threshold 0.5 unless overridden.

// Peterson figure names (I, I*, II, III, IV) read better in CamelCase tests.
#![allow(non_snake_case)]

use uuid::Uuid;

use crate::fuzzy::quantifier::Quantifier;
use crate::fuzzy::syllogism::{
    classify_figure, delete_syllogism_proof, list_syllogism_proofs_for_narrative,
    load_syllogism_proof, parse_statement, save_syllogism_proof, verify, GradedValidity,
    PredicateResolver, Syllogism, SyllogismProof, SyllogismStatement,
    TypePredicateResolver,
};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::types::{Entity, EntityType, MaturityLevel};

// ── Fixture helpers ──────────────────────────────────────────────────────────

fn make_hg() -> Hypergraph {
    use std::sync::Arc;
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn mint_entity(hg: &Hypergraph, nid: &str, et: EntityType, name: &str) -> Uuid {
    let id = Uuid::now_v7();
    let e = Entity {
        id,
        entity_type: et,
        properties: serde_json::json!({"name": name}),
        beliefs: None,
        embedding: None,
        confidence_breakdown: None,
        confidence: 1.0,
        maturity: MaturityLevel::Candidate,
        narrative_id: Some(nid.into()),
        provenance: vec![],
        extraction_method: None,
        deleted_at: None,
        transaction_time: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    hg.create_entity(e).expect("create_entity");
    id
}

/// Seed a narrative with N entities of one type. Used as the simplest
/// positive fixture — every entity satisfies `type:<et>`, so every
/// quantifier ramp saturates at r=1.0.
fn seed_uniform(hg: &Hypergraph, nid: &str, et: EntityType, n: usize) {
    for i in 0..n {
        mint_entity(hg, nid, et.clone(), &format!("{:?}-{i}", et));
    }
}

/// Build a `SyllogismStatement` with `subject_pred_id == object_pred_id
/// == "type:Actor"`. Reuses the same predicate in every slot to keep
/// fixtures compact; callers override the object slot when they want
/// a failing conclusion.
fn stmt_actor(q: Quantifier) -> SyllogismStatement {
    SyllogismStatement {
        quantifier: q,
        subject_pred_id: "type:Actor".into(),
        object_pred_id: "type:Actor".into(),
    }
}

// ── T1: Figure I — classical All/All/All, all true ──────────────────────────

#[test]
fn test_figure_I_all_valid() {
    let hg = make_hg();
    let nid = "fig-I-valid";
    seed_uniform(&hg, nid, EntityType::Actor, 5);

    let s = Syllogism {
        major: stmt_actor(Quantifier::AlmostAll),
        minor: stmt_actor(Quantifier::AlmostAll),
        conclusion: stmt_actor(Quantifier::AlmostAll),
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    assert!(
        (gv.degree - 1.0).abs() < 1e-9,
        "expected degree ≈ 1.0 got {}",
        gv.degree
    );
    assert_eq!(gv.figure, "I");
    assert!(gv.valid, "Figure I with all-true premises must be valid");
}

// ── T2: Figure I — conclusion fails, degree < threshold ─────────────────────

#[test]
fn test_figure_I_all_invalid() {
    // Conclusion object = type:Location but narrative has no Location
    // entities → Q_almost_all(0) = 0. Gödel min-fold returns 0.
    let hg = make_hg();
    let nid = "fig-I-invalid";
    seed_uniform(&hg, nid, EntityType::Actor, 5);
    let s = Syllogism {
        major: stmt_actor(Quantifier::AlmostAll),
        minor: stmt_actor(Quantifier::AlmostAll),
        conclusion: SyllogismStatement {
            quantifier: Quantifier::AlmostAll,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "type:Location".into(),
        },
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    assert!(gv.degree < 0.5, "expected degree < 0.5 got {}", gv.degree);
    assert!(!gv.valid, "expected valid=false");
}

// ── T3: Figure I* — Most/All/Most, graded-valid ─────────────────────────────

#[test]
fn test_figure_I_star_most_all_most() {
    // subject=type:Actor restricts to 10 Actors; object=type:Actor over
    // that domain yields r=1.0 → Q_most(1.0) = 1.0 for all three.
    let hg = make_hg();
    let nid = "fig-I-star";
    seed_uniform(&hg, nid, EntityType::Actor, 10);
    seed_uniform(&hg, nid, EntityType::Location, 10);

    let s = Syllogism {
        major: stmt_actor(Quantifier::Most),
        minor: stmt_actor(Quantifier::AlmostAll),
        conclusion: stmt_actor(Quantifier::Most),
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    assert!(gv.degree > 0.5, "expected degree > 0.5, got {}", gv.degree);
    assert_eq!(gv.figure, "I*");
    assert!(gv.valid);
}

// ── T4: Figure II — INVALID in Peterson ─────────────────────────────────────

#[test]
fn test_figure_II_invalid_in_peterson() {
    // Figure II is invalid by Peterson taxonomy: even when every
    // premise evaluates to 1.0 (degree under Gödel is 1.0), the
    // verifier flags `valid = false`. The degree computation and the
    // validity verdict are distinct — this test pins that contract.
    let hg = make_hg();
    let nid = "fig-II";
    seed_uniform(&hg, nid, EntityType::Actor, 10);

    let s = Syllogism {
        major: stmt_actor(Quantifier::AlmostAll),
        minor: stmt_actor(Quantifier::Most),
        conclusion: stmt_actor(Quantifier::Most),
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    assert_eq!(gv.figure, "II", "quantifier triple (All,Most,*) → II");
    assert!(
        !gv.valid,
        "Figure II must be invalid by Peterson taxonomy regardless of degree"
    );
}

// ── T5: Figure III — Almost-all + All via hint ──────────────────────────────

#[test]
fn test_figure_III_almost_all() {
    // Figure III shares its quantifier triple with Figure I; callers
    // that mean III disambiguate via `figure_hint`.
    let hg = make_hg();
    let nid = "fig-III";
    seed_uniform(&hg, nid, EntityType::Actor, 5);
    let s = Syllogism {
        major: stmt_actor(Quantifier::AlmostAll),
        minor: stmt_actor(Quantifier::AlmostAll),
        conclusion: stmt_actor(Quantifier::AlmostAll),
        figure_hint: Some("III".into()),
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    assert_eq!(gv.figure, "III");
    assert!(gv.valid);
    assert!((gv.degree - 1.0).abs() < 1e-9);
}

// ── T6: Figure IV — Most/Most/Many, graded-valid ─────────────────────────────

#[test]
fn test_figure_IV_graded_valid() {
    let hg = make_hg();
    let nid = "fig-IV";
    seed_uniform(&hg, nid, EntityType::Actor, 10);

    let s = Syllogism {
        major: stmt_actor(Quantifier::Most),
        minor: stmt_actor(Quantifier::Most),
        conclusion: stmt_actor(Quantifier::Many),
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    assert_eq!(gv.figure, "IV", "quantifier triple (Most,Most,Many) → IV");
    assert!(gv.degree > 0.5, "degree should exceed threshold");
    assert!(gv.valid);
}

// ── T7: Non-canonical figure fallback (custom resolver) ─────────────────────

/// Custom resolver that maps `"even-id-name"` to "Entity name ends in an
/// even digit." Used to demonstrate the trait plugs with something that
/// isn't a type filter.
struct NameSuffixResolver;
impl PredicateResolver for NameSuffixResolver {
    fn resolve(
        &self,
        predicate_id: &str,
    ) -> Result<Box<dyn Fn(&Entity) -> f64 + Send + Sync>, crate::error::TensaError> {
        // Route unknown ids through the default to keep the fixture
        // small — only intercept our one custom id.
        if predicate_id == "even-name-suffix" {
            return Ok(Box::new(|e: &Entity| -> f64 {
                let name = e
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let last = name.chars().last().unwrap_or(' ');
                match last.to_digit(10) {
                    Some(d) if d % 2 == 0 => 1.0,
                    _ => 0.0,
                }
            }));
        }
        TypePredicateResolver.resolve(predicate_id)
    }
}

#[test]
fn test_non_canonical_figure_fallback() {
    // (Few, Many, Many) matches no Peterson triple → "non-canonical".
    let hg = make_hg();
    let nid = "noncanon";
    seed_uniform(&hg, nid, EntityType::Actor, 10);

    let s = Syllogism {
        major: SyllogismStatement {
            quantifier: Quantifier::Few,
            subject_pred_id: "type:Actor".into(),
            object_pred_id: "even-name-suffix".into(),
        },
        minor: stmt_actor(Quantifier::Many),
        conclusion: stmt_actor(Quantifier::Many),
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &NameSuffixResolver)
        .expect("verify");
    assert_eq!(gv.figure, "non-canonical");
    // Non-canonical figure must flag valid=false regardless of degree.
    assert!(!gv.valid);
}

// ── T8: T-norm switch produces different degrees ────────────────────────────

#[test]
fn test_verify_with_tnorm_switch() {
    let hg = make_hg();
    let nid = "tnorm-switch";
    // 10 Actors + 10 Locations. subject=type:Actor restricts to 10
    // Actors; object=type:Actor over that domain → r=1.0 → all three
    // premise evaluations saturate, Gödel min returns 1.0. To get
    // different degrees between Gödel and Łukasiewicz we need a
    // non-trivial intermediate ratio.
    seed_uniform(&hg, nid, EntityType::Actor, 5);
    seed_uniform(&hg, nid, EntityType::Location, 5);

    // d_major = d_minor = d_conclusion = Q_most(0.5) = 0.4 (ramp
    // interior). Gödel min-fold yields 0.4; Łukasiewicz T_Luk(0.4,
    // T_Luk(0.4, 0.4)) = T_Luk(0.4, 0) = 0. Three-way Łukasiewicz
    // only diverges from Gödel when at least two args are strictly
    // below 1.0 — hence this fixture.
    let s_over_entity = |q| SyllogismStatement {
        quantifier: q,
        subject_pred_id: "entity".into(),
        object_pred_id: "type:Actor".into(),
    };
    let s = Syllogism {
        major: s_over_entity(Quantifier::Most),
        minor: s_over_entity(Quantifier::Most),
        conclusion: s_over_entity(Quantifier::Most),
        figure_hint: None,
    };
    let g_godel = verify(&hg, nid, &s, TNormKind::Godel, 0.3, &TypePredicateResolver)
        .expect("verify");
    let g_luk = verify(
        &hg,
        nid,
        &s,
        TNormKind::Lukasiewicz,
        0.3,
        &TypePredicateResolver,
    )
    .expect("verify");
    assert!(
        (g_godel.degree - g_luk.degree).abs() > 1e-3,
        "expected Gödel ({}) ≠ Łukasiewicz ({}) on non-trivial fixture",
        g_godel.degree,
        g_luk.degree
    );
}

// ── T9: KV round-trip ────────────────────────────────────────────────────────

#[test]
fn test_kv_roundtrip() {
    let hg = make_hg();
    let nid = "kv";
    seed_uniform(&hg, nid, EntityType::Actor, 3);

    let s = Syllogism {
        major: stmt_actor(Quantifier::AlmostAll),
        minor: stmt_actor(Quantifier::AlmostAll),
        conclusion: stmt_actor(Quantifier::AlmostAll),
        figure_hint: None,
    };
    let gv = verify(&hg, nid, &s, TNormKind::Godel, 0.5, &TypePredicateResolver)
        .expect("verify");
    let proof = SyllogismProof::new(s, gv.clone());
    save_syllogism_proof(hg.store(), nid, &proof).expect("save");

    let loaded = load_syllogism_proof(hg.store(), nid, &proof.id)
        .expect("load")
        .expect("proof present");
    assert_eq!(loaded.id, proof.id);
    assert_eq!(loaded.graded_validity, proof.graded_validity);

    let listed =
        list_syllogism_proofs_for_narrative(hg.store(), nid).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, proof.id);

    delete_syllogism_proof(hg.store(), nid, &proof.id).expect("delete");
    let reloaded =
        load_syllogism_proof(hg.store(), nid, &proof.id).expect("load-after-delete");
    assert!(reloaded.is_none());
}

// ── T10: TensaQL VERIFY SYLLOGISM end-to-end ────────────────────────────────

#[test]
fn test_grammar_and_executor_e2e() {
    use crate::hypergraph::Hypergraph;
    use crate::query::{executor, parser, planner};
    use crate::temporal::index::IntervalTree;

    let hg = make_hg();
    let nid = "e2e";
    seed_uniform(&hg, nid, EntityType::Actor, 5);

    let q = format!(
        "VERIFY SYLLOGISM {{ major: 'ALL type:Actor IS type:Actor', \
         minor: 'ALL type:Actor IS type:Actor', \
         conclusion: 'ALL type:Actor IS type:Actor' }} FOR \"{nid}\""
    );
    let stmt = parser::parse_statement(&q).expect("parse");
    let plan = planner::plan_statement(&stmt).expect("plan");
    let interval_tree = IntervalTree::default();
    let rows = executor::execute(&plan, &hg, &interval_tree).expect("execute");
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    let deg = row.get("degree").and_then(|v| v.as_f64()).expect("degree");
    let fig = row.get("figure").and_then(|v| v.as_str()).expect("figure");
    let valid = row.get("valid").and_then(|v| v.as_bool()).expect("valid");
    assert!((deg - 1.0).abs() < 1e-9, "degree ≈ 1.0 got {deg}");
    assert_eq!(fig, "I");
    assert!(valid);
}

// ── T11: parse_statement + classify_figure unit coverage ────────────────────

#[test]
fn test_parse_statement_and_classify_figure() {
    let maj = parse_statement("MOST type:Actor IS type:Actor").expect("maj");
    assert_eq!(maj.quantifier, Quantifier::Most);
    assert_eq!(maj.subject_pred_id, "type:Actor");
    assert_eq!(maj.object_pred_id, "type:Actor");

    let _all = parse_statement("ALL type:Actor IS type:Actor").expect("all");
    let almost =
        parse_statement("almost_all type:Actor IS type:Actor").expect("almost");
    assert_eq!(almost.quantifier, Quantifier::AlmostAll);

    let missing = parse_statement("TRUTHINESS whatever");
    assert!(missing.is_err(), "missing IS must error");
    let unknown = parse_statement("NOBODY type:Actor IS type:Actor");
    assert!(unknown.is_err(), "unknown quantifier token must error");

    let syl = Syllogism {
        major: parse_statement("MOST type:Actor IS type:Actor").unwrap(),
        minor: parse_statement("MOST type:Actor IS type:Actor").unwrap(),
        conclusion: parse_statement("MANY type:Actor IS type:Actor").unwrap(),
        figure_hint: None,
    };
    assert_eq!(classify_figure(&syl), "IV");

    let noncanon = Syllogism {
        major: parse_statement("FEW type:Actor IS type:Actor").unwrap(),
        minor: parse_statement("FEW type:Actor IS type:Actor").unwrap(),
        conclusion: parse_statement("FEW type:Actor IS type:Actor").unwrap(),
        figure_hint: None,
    };
    assert_eq!(classify_figure(&noncanon), "non-canonical");
}

// Dummy check: GradedValidity shape used by callers is stable.
#[test]
fn test_graded_validity_serde_shape() {
    let gv = GradedValidity {
        degree: 0.7,
        figure: "I".into(),
        valid: true,
        threshold: 0.5,
    };
    let json = serde_json::to_value(&gv).expect("to_value");
    assert_eq!(json["degree"], 0.7);
    assert_eq!(json["figure"], "I");
    assert_eq!(json["valid"], true);
    assert_eq!(json["threshold"], 0.5);
    let back: GradedValidity = serde_json::from_value(json).expect("round-trip");
    assert_eq!(back, gv);
}
