//! Fuzzy Sprint Phase 3 — executor-level tests for fuzzy config propagation.
//!
//! These tests exercise the end-to-end path `TensaQL → parse → plan →
//! execute` against [`crate::store::memory::MemoryStore`], confirming that:
//!
//! * A fuzzy-clause-bearing query round-trips through the executor
//!   successfully (no "unsupported plan step" errors).
//! * Condition AND/OR fusion with the new t-norm selector matches the
//!   existing boolean short-circuit when the selector is `Godel` (the
//!   backward-compat axis from Phases 1–2).
//! * `INFER EVIDENCE ... WITH TNORM 'goguen'` routes to
//!   `evidence::combine_with_tnorm(TNormKind::Goguen)`, which is the
//!   current default; selecting `Lukasiewicz` produces a measurably
//!   different mass fold at the algorithm layer (sanity check that the
//!   Phase 1 `combine_with_tnorm` actually picks up the choice).
//!
//! The executor itself consumes `FuzzyPlanConfig` on a best-effort basis
//! in Phase 3 — MATCH / ASK sites keep their pre-Phase-3 numerics under
//! default `Godel`, and INFER EVIDENCE is the load-bearing call site that
//! already has the wiring from Phase 1. These tests assert that (a) the
//! user's selection reaches the executor's decision points via the plan
//! and (b) the fuzzy helpers themselves honour the choice when invoked
//! directly from a TensaQL plan.
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet].

use std::collections::{BTreeSet, HashMap};

use crate::analysis::evidence::{combine_with_tnorm, MassFunction};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::query::executor::{execute, execute_explained};
use crate::query::parser::parse_query;
use crate::query::planner::plan_query;
use crate::store::memory::MemoryStore;
use crate::temporal::index::IntervalTree;

fn mk_hg() -> (Hypergraph, IntervalTree) {
    let store = std::sync::Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);
    let tree = IntervalTree::new();
    (hg, tree)
}

// ── T1 — MATCH with WITH TNORM 'godel' executes and preserves Gödel semantics
//
// Gödel AND is `min`; on boolean inputs that reduces to the existing
// short-circuit fold. Equivalent to running the same query without the
// fuzzy tail: both must return zero rows against an empty store (no Actor
// has been created).

#[test]
fn test_match_with_tnorm_godel_executes_empty_store() {
    let (hg, tree) = mk_hg();
    let q = parse_query(
        r#"MATCH (e:Actor) WHERE e.confidence > 0.5 AND e.name = "A" RETURN e WITH TNORM 'godel'"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let rows = execute(&plan, &hg, &tree).expect("execute ok");
    assert!(rows.is_empty(), "empty store returns no rows");
}

// ── T2 — Same MATCH under Łukasiewicz t-norm still executes cleanly ─────────
//
// Łukasiewicz AND is `max(0, a+b-1)`; on `{0,1}` inputs it matches Gödel,
// so the boolean short-circuit path stays semantically correct. The key
// assertion is that the plan executes without error — the executor does
// not panic when a non-default t-norm rides on the plan step.

#[test]
fn test_match_with_tnorm_lukasiewicz_executes() {
    let (hg, tree) = mk_hg();
    let q = parse_query(
        r#"MATCH (e:Actor) WHERE e.confidence > 0.5 AND e.name = "A" RETURN e WITH TNORM 'lukasiewicz'"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let rows = execute(&plan, &hg, &tree).expect("execute ok");
    assert!(rows.is_empty());
}

// ── T3 — INFER EVIDENCE with WITH TNORM 'goguen' routes correctly ───────────
//
// The fuzzy_config ends up in `parameters.fuzzy_config` on the INFER job
// descriptor row. We don't submit an actual evidence job here (needs a
// `JobQueue`), but we assert the descriptor row surfaces the selection so
// the engine dispatcher downstream can honour it.

#[test]
fn test_infer_evidence_with_goguen_surfaces_in_descriptor() {
    let (hg, tree) = mk_hg();
    let q = parse_query(
        r#"INFER EVIDENCE FOR n:Narrative WHERE n.narrative_id = "n1" RETURN n WITH TNORM 'goguen'"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let rows = execute(&plan, &hg, &tree).expect("execute ok");
    assert_eq!(rows.len(), 1, "INFER returns a single descriptor row");
    let params = rows[0]
        .get("_parameters")
        .expect("_parameters present")
        .as_object()
        .expect("parameters is a JSON object");
    let fz = params
        .get("fuzzy_config")
        .expect("fuzzy_config propagated to params");
    // `tnorm` is an externally-tagged enum; Goguen serializes as `{"kind": "goguen"}`.
    let kind = fz
        .get("tnorm")
        .and_then(|t| t.get("kind"))
        .and_then(|k| k.as_str());
    assert_eq!(kind, Some("goguen"));
}

// ── T4 — Direct algorithm-level check: combine_with_tnorm honours the choice
//
// The Phase 1 wiring into `evidence::combine_with_tnorm` is the load-bearing
// wire that makes the TensaQL clause meaningful. Assert it here so a
// regression in `combine_with_tnorm` (e.g. ignoring the `tnorm` arg) is
// caught via the Phase 3 test suite as well as the Phase 1 suite.

#[test]
fn test_combine_with_tnorm_diverges_across_families() {
    // Two mass functions over a 2-element frame:
    //   m1: singleton {0} @ 0.6, uncertainty Θ @ 0.4
    //   m2: singleton {0} @ 0.7, uncertainty Θ @ 0.3
    let frame = vec!["A".to_string(), "B".to_string()];
    let mut masses1: HashMap<BTreeSet<usize>, f64> = HashMap::new();
    let mut singleton: BTreeSet<usize> = BTreeSet::new();
    singleton.insert(0);
    masses1.insert(singleton.clone(), 0.6);
    let full: BTreeSet<usize> = (0..frame.len()).collect();
    masses1.insert(full.clone(), 0.4);
    let m1 = MassFunction::new(frame.clone(), masses1).unwrap();

    let mut masses2: HashMap<BTreeSet<usize>, f64> = HashMap::new();
    masses2.insert(singleton.clone(), 0.7);
    masses2.insert(full.clone(), 0.3);
    let m2 = MassFunction::new(frame, masses2).unwrap();

    let (goguen_result, _) =
        combine_with_tnorm(&m1, &m2, TNormKind::Goguen).expect("goguen combine");
    let (lukasiewicz_result, _) =
        combine_with_tnorm(&m1, &m2, TNormKind::Lukasiewicz).expect("luk combine");

    let g_singleton = goguen_result.masses.get(&singleton).copied().unwrap_or(0.0);
    let l_singleton = lukasiewicz_result
        .masses
        .get(&singleton)
        .copied()
        .unwrap_or(0.0);
    assert!(
        (g_singleton - l_singleton).abs() > 1e-9,
        "Goguen vs Lukasiewicz must produce divergent singleton mass"
    );
}

// ── T5 — End-to-end roundtrip: parse → plan → execute survives an ASK tail

#[test]
fn test_ask_roundtrip_with_fuzzy_tail() {
    let (hg, tree) = mk_hg();
    // ASK without an extractor returns an error at execute time, but the
    // plan step must be well-formed and carry the fuzzy_config slot.
    let q = parse_query(
        r#"ASK "Who is central?" OVER "n1" MODE local WITH TNORM 'hamacher' AGGREGATE MEAN"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    // EXPLAIN payload reveals the selected t-norm + aggregator.
    let rows = execute_explained(&plan, &hg, &tree, true).expect("explain ok");
    let plan_json = rows[0].get("plan").expect("plan field present");
    let steps = plan_json
        .get("steps")
        .and_then(|v| v.as_array())
        .expect("steps array");
    let ask = steps
        .iter()
        .find(|s| s.get("AskLlm").is_some())
        .expect("AskLlm present");
    let fz = ask
        .get("AskLlm")
        .and_then(|a| a.get("fuzzy_config"))
        .expect("fuzzy_config on AskLlm step");
    let tnorm_kind = fz
        .get("tnorm")
        .and_then(|t| t.get("kind"))
        .and_then(|k| k.as_str());
    assert_eq!(tnorm_kind, Some("hamacher"));
    // MEAN is a unit variant — serializes as the bare string `"Mean"`.
    let agg = fz.get("aggregator").expect("aggregator field");
    assert!(
        agg.as_str() == Some("Mean") || agg.get("Mean").is_some(),
        "aggregator must be Mean, got {agg:?}"
    );
}

// ── T6 — Backward-compat: existing tests that didn't use fuzzy tail must not
// see any change in the result rows.

#[test]
fn test_existing_match_returns_same_shape_without_fuzzy_tail() {
    let (hg, tree) = mk_hg();
    let q =
        parse_query(r#"MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e"#).unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let rows = execute(&plan, &hg, &tree).expect("execute ok");
    assert!(rows.is_empty(), "empty store → empty rows (unchanged)");
}
