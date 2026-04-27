//! Fuzzy Sprint Phase 3 — parser tests for the TensaQL `WITH TNORM` and
//! `AGGREGATE` tail clauses on MATCH, INFER, and ASK statements.
//!
//! Scope: pure AST-shape assertions. Semantic validation (unknown t-norm,
//! OWA sum-to-1, Choquet measure resolution) is tested in
//! [`crate::query::planner_fuzzy_tests`]; end-to-end behaviour under the
//! executor lives in [`crate::query::executor_fuzzy_tests`].
//!
//! The backward-compat invariant is load-bearing: every query without the
//! new clauses must parse to `FuzzyConfig::default()` (`tnorm = None,
//! aggregator = None`) so that Phases 1–2 numerics stay bit-identical at
//! call sites that have not been migrated to consume the override yet.
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet].

use crate::fuzzy::tnorm::TNormKind;
use crate::query::parser::{
    parse_query, parse_statement, AggregatorSpec, FuzzyConfig, TensaStatement,
};

// ── T1 — `WITH TNORM '<kind>'` attaches to a MATCH query ─────────────────────

#[test]
fn test_match_with_tnorm_lukasiewicz_parses() {
    let q = parse_query(
        r#"MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e WITH TNORM 'lukasiewicz'"#,
    )
    .expect("MATCH ... WITH TNORM 'lukasiewicz' must parse");
    assert_eq!(
        q.fuzzy_config.tnorm.as_deref(),
        Some("lukasiewicz"),
        "raw tnorm name preserved — planner resolves into TNormKind"
    );
    assert!(q.fuzzy_config.aggregator.is_none());
}

#[test]
fn test_match_with_tnorm_godel_parses() {
    let q = parse_query(r#"MATCH (e:Actor) RETURN e WITH TNORM 'godel'"#)
        .expect("WITH TNORM 'godel' must parse");
    assert_eq!(q.fuzzy_config.tnorm.as_deref(), Some("godel"));
}

// ── T2 — `AGGREGATE OWA [w...]` attaches and carries weights ─────────────────

#[test]
fn test_match_aggregate_owa_weights_parses() {
    let q = parse_query(r#"MATCH (e:Actor) RETURN e AGGREGATE OWA [0.4, 0.3, 0.2, 0.1]"#)
        .expect("AGGREGATE OWA must parse");
    match q.fuzzy_config.aggregator {
        Some(AggregatorSpec::Owa(ref weights)) => {
            assert_eq!(weights.len(), 4);
            assert!((weights[0] - 0.4).abs() < 1e-12);
            assert!((weights[1] - 0.3).abs() < 1e-12);
            assert!((weights[2] - 0.2).abs() < 1e-12);
            assert!((weights[3] - 0.1).abs() < 1e-12);
        }
        other => panic!("expected AggregatorSpec::Owa, got {:?}", other),
    }
    assert!(q.fuzzy_config.tnorm.is_none());
}

#[test]
fn test_match_aggregate_owa_single_weight_parses() {
    let q = parse_query(r#"MATCH (e:Actor) RETURN e AGGREGATE OWA [1.0]"#)
        .expect("single-element OWA must parse");
    match q.fuzzy_config.aggregator {
        Some(AggregatorSpec::Owa(w)) => assert_eq!(w, vec![1.0]),
        other => panic!("expected OWA, got {:?}", other),
    }
}

// ── T3 — `AGGREGATE CHOQUET '<name>'` carries the opaque measure reference ──

#[test]
fn test_infer_evidence_aggregate_choquet_by_ref_parses() {
    let q = parse_query(
        r#"INFER EVIDENCE FOR s:Situation RETURN s AGGREGATE CHOQUET 'my-measure'"#,
    )
    .expect("INFER EVIDENCE ... AGGREGATE CHOQUET 'my-measure' must parse");
    match q.fuzzy_config.aggregator {
        Some(AggregatorSpec::ChoquetByRef(ref name)) => assert_eq!(name, "my-measure"),
        other => panic!("expected ChoquetByRef, got {:?}", other),
    }
}

// ── T4 — INFER HIGHER_ORDER_CONTAGION accepts the tail ──────────────────────

#[test]
fn test_infer_higher_order_contagion_with_tnorm_godel_parses() {
    let q = parse_query(
        r#"INFER HIGHER_ORDER_CONTAGION FOR n:Narrative WHERE n.narrative_id = "disinfo-1" RETURN n WITH TNORM 'godel'"#,
    )
    .expect("INFER HIGHER_ORDER_CONTAGION ... WITH TNORM 'godel' must parse");
    assert_eq!(q.fuzzy_config.tnorm.as_deref(), Some("godel"));
}

// ── T5 — ASK with both WITH TNORM and AGGREGATE tail clauses ────────────────

#[test]
fn test_ask_with_tnorm_and_aggregate_mean_parses() {
    let q = parse_query(
        r#"ASK "who are the suspects?" OVER "case-1" WITH TNORM 'goguen' AGGREGATE MEAN"#,
    )
    .expect("ASK ... WITH TNORM 'goguen' AGGREGATE MEAN must parse");
    let ask = q.ask_clause.as_ref().expect("ask clause");
    assert_eq!(ask.question, "who are the suspects?");
    assert_eq!(ask.narrative_id.as_deref(), Some("case-1"));
    assert_eq!(q.fuzzy_config.tnorm.as_deref(), Some("goguen"));
    assert!(matches!(
        q.fuzzy_config.aggregator,
        Some(AggregatorSpec::Mean)
    ));
}

// ── T6 — Empty tail → FuzzyConfig::default() (backward-compat) ──────────────

#[test]
fn test_empty_tail_yields_default_fuzzy_config() {
    let q = parse_query(r#"MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e"#)
        .expect("match without fuzzy clauses must parse");
    assert!(q.fuzzy_config.is_empty());
    assert!(q.fuzzy_config.tnorm.is_none());
    assert!(q.fuzzy_config.aggregator.is_none());
    assert_eq!(q.fuzzy_config, FuzzyConfig::default());
}

// ── T7 — Unknown t-norm name parses cleanly; planner rejects ────────────────

#[test]
fn test_unknown_tnorm_name_parses_as_string() {
    let q =
        parse_query(r#"MATCH (e:Actor) RETURN e WITH TNORM 'einstein'"#).expect(
            "unknown t-norm name parses — rejection belongs to the planner",
        );
    assert_eq!(q.fuzzy_config.tnorm.as_deref(), Some("einstein"));
}

// ── T8 — Weight vector with 0 elements: parser rejects ──────────────────────
//
// The grammar rule for `weight_vector` requires at least one entry
// (`weight_entry ~ ("," ~ weight_entry)*`), so `[]` is a genuine parse
// error. `parse_query` returns `TensaError::ParseError` in this case.

#[test]
fn test_empty_owa_weight_vector_is_parse_error() {
    let err = parse_query(r#"MATCH (e:Actor) RETURN e AGGREGATE OWA []"#).unwrap_err();
    let msg = format!("{err}");
    // Pest surfaces a syntax error; either the word "ParseError" or "OWA"
    // should appear — assert the error path was taken, not OK.
    assert!(!msg.is_empty(), "empty OWA must surface a parse error");
}

// ── T9 — Round-trip (via serialisation) preserves fuzzy clauses ─────────────

#[test]
fn test_fuzzy_config_serde_roundtrip() {
    let cases: Vec<FuzzyConfig> = vec![
        FuzzyConfig::default(),
        FuzzyConfig {
            tnorm: Some("lukasiewicz".into()),
            aggregator: None,
            ..Default::default()
        },
        FuzzyConfig {
            tnorm: None,
            aggregator: Some(AggregatorSpec::Owa(vec![0.4, 0.3, 0.2, 0.1])),
            ..Default::default()
        },
        FuzzyConfig {
            tnorm: Some("godel".into()),
            aggregator: Some(AggregatorSpec::Mean),
            ..Default::default()
        },
        FuzzyConfig {
            tnorm: None,
            aggregator: Some(AggregatorSpec::ChoquetByRef("m1".into())),
            ..Default::default()
        },
        FuzzyConfig {
            tnorm: None,
            aggregator: Some(AggregatorSpec::TNormReduce(TNormKind::Goguen)),
            ..Default::default()
        },
        FuzzyConfig {
            tnorm: None,
            aggregator: Some(AggregatorSpec::TConormReduce(TNormKind::Hamacher(0.5))),
            ..Default::default()
        },
    ];
    for cfg in cases {
        let ser = serde_json::to_string(&cfg).expect("serialize");
        let back: FuzzyConfig = serde_json::from_str(&ser).expect("deserialize");
        assert_eq!(cfg, back, "round-trip failed for {:?}, json={}", cfg, ser);
    }
}

// ── INFER tail round-trip — one test per listed INFER verb ──────────────────
//
// The spec requires at least 1 round-trip test per INFER verb; the listed
// set is the 16 verbs that must accept the `WITH TNORM / AGGREGATE` tail.
// We parametrize here instead of 16 hand-written test fns, then assert the
// tail is preserved on every variant.

#[test]
fn test_infer_verbs_accept_fuzzy_tail() {
    let verbs: &[&str] = &[
        "CENTRALITY",
        "ENTROPY",
        "BELIEFS",
        "EVIDENCE",
        "ARGUMENTS",
        "CONTAGION",
        "TEMPORAL_RULES",
        "MEAN_FIELD",
        "PSL",
        "TRAJECTORY",
        "SIMULATE",
        "COMMUNITIES",
        "HIGHER_ORDER_CONTAGION",
    ];
    for verb in verbs {
        let query = format!(
            r#"INFER {verb} FOR n:Narrative WHERE n.narrative_id = "x" RETURN n WITH TNORM 'godel' AGGREGATE MEAN"#
        );
        let q = parse_query(&query)
            .unwrap_or_else(|e| panic!("INFER {verb} tail must parse: {e}"));
        assert_eq!(q.fuzzy_config.tnorm.as_deref(), Some("godel"), "{verb}");
        assert!(
            matches!(q.fuzzy_config.aggregator, Some(AggregatorSpec::Mean)),
            "{verb}"
        );
    }
}

// OPINION_DYNAMICS, OPINION_PHASE_TRANSITION, and HYPERGRAPH FROM DYNAMICS
// are carried on TensaStatement variants outside `TensaQuery`. Assert the
// tail lands on their struct's `fuzzy_config` field.

#[test]
fn test_opinion_dynamics_accepts_fuzzy_tail() {
    let stmt = parse_statement(
        r#"INFER OPINION_DYNAMICS( confidence_bound := 0.3, variant := 'pairwise' ) FOR "n1" WITH TNORM 'goguen' AGGREGATE MEAN"#,
    )
    .expect("opinion dynamics with fuzzy tail must parse");
    match stmt {
        TensaStatement::InferOpinionDynamics { fuzzy_config, .. } => {
            assert_eq!(fuzzy_config.tnorm.as_deref(), Some("goguen"));
            assert!(matches!(
                fuzzy_config.aggregator,
                Some(AggregatorSpec::Mean)
            ));
        }
        other => panic!("expected InferOpinionDynamics, got {other:?}"),
    }
}

#[test]
fn test_opinion_phase_transition_accepts_fuzzy_tail() {
    let stmt = parse_statement(
        r#"INFER OPINION_PHASE_TRANSITION( c_start := 0.05, c_end := 0.5, c_steps := 10 ) FOR "n1" WITH TNORM 'lukasiewicz'"#,
    )
    .expect("opinion phase transition with fuzzy tail must parse");
    match stmt {
        TensaStatement::InferOpinionPhaseTransition { fuzzy_config, .. } => {
            assert_eq!(fuzzy_config.tnorm.as_deref(), Some("lukasiewicz"));
        }
        other => panic!("expected InferOpinionPhaseTransition, got {other:?}"),
    }
}

#[test]
fn test_hypergraph_reconstruction_accepts_fuzzy_tail() {
    let stmt = parse_statement(
        r#"INFER HYPERGRAPH FROM DYNAMICS FOR "corpus-1" AGGREGATE TNORM_REDUCE 'godel'"#,
    )
    .expect("hypergraph reconstruction with fuzzy tail must parse");
    match stmt {
        TensaStatement::InferHypergraphReconstruction { fuzzy_config, .. } => {
            assert!(fuzzy_config.tnorm.is_none());
            match fuzzy_config.aggregator {
                Some(AggregatorSpec::TNormReduce(TNormKind::Godel)) => {}
                other => panic!("expected TNormReduce(Godel), got {other:?}"),
            }
        }
        other => panic!("expected InferHypergraphReconstruction, got {other:?}"),
    }
}

#[test]
fn test_tconorm_reduce_resolves_at_parse_time() {
    let q = parse_query(
        r#"MATCH (e:Actor) RETURN e AGGREGATE TCONORM_REDUCE 'lukasiewicz'"#,
    )
    .expect("tconorm_reduce with lukasiewicz must parse");
    match q.fuzzy_config.aggregator {
        Some(AggregatorSpec::TConormReduce(TNormKind::Lukasiewicz)) => {}
        other => panic!("expected TConormReduce(Lukasiewicz), got {other:?}"),
    }
}

// ── Backward-compat guarantee: existing tests still represent the state we
// ship. Non-fuzzy queries must expose `FuzzyConfig::default()` — same
// contract as Phase 1 / Phase 2's backward_compat_tests.

#[test]
fn test_default_preserved_on_existing_shapes() {
    let queries = &[
        "MATCH (e:Actor) RETURN e",
        "MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e",
        r#"INFER CAUSES FOR s:Situation RETURN s"#,
        r#"ASK "What happened?""#,
        r#"ASK "Who is the villain?" OVER "story-1" MODE local"#,
    ];
    for q_str in queries {
        let q = parse_query(q_str).unwrap_or_else(|e| panic!("{q_str} must parse: {e}"));
        assert!(q.fuzzy_config.is_empty(), "{q_str} must have empty config");
    }
}
