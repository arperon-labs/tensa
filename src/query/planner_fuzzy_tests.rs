//! Fuzzy Sprint Phase 3 — planner tests for the resolved `FuzzyPlanConfig`
//! that the TensaQL `WITH TNORM` / `AGGREGATE` tail clauses produce.
//!
//! Planner responsibilities enforced here:
//! * Unknown t-norm names surface as [`crate::TensaError::InvalidInput`] with
//!   the registry's "known: [...]" hint — NOT a parse error.
//! * Unknown aggregator-kind strings surface as the same error with a hint
//!   naming the unresolved clause (Choquet-by-ref → Phase-4 pointer).
//! * OWA weights are validated to be non-empty and sum to 1.0 ± 1e-9.
//! * Valid configs propagate to the appropriate plan step's `fuzzy_config`
//!   slot with the resolved [`TNormKind`] / [`AggregatorKind`].
//! * `EXPLAIN` output JSON includes the fuzzy_config on steps that consume it.
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet].

use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::tnorm::TNormKind;
use crate::query::parser::{parse_query, parse_statement};
use crate::query::planner::{plan_query, plan_statement, PlanStep};
use crate::TensaError;

// ── T1 — Unknown t-norm name → planner error with registry hint ─────────────

#[test]
fn test_unknown_tnorm_name_rejected_at_plan_time() {
    let q = parse_query(r#"MATCH (e:Actor) RETURN e WITH TNORM 'einstein'"#)
        .expect("unknown name parses");
    let err = plan_query(&q).expect_err("unknown t-norm must fail at plan time");
    match err {
        TensaError::InvalidInput(msg) => {
            assert!(
                msg.contains("einstein"),
                "message should name the unknown kind, got: {msg}"
            );
            assert!(
                msg.contains("known:"),
                "message should list known kinds, got: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

// ── T2 — Choquet measure-ref → planner error with Phase-4 hint ──────────────

#[test]
fn test_choquet_by_ref_rejected_with_phase_4_hint() {
    let q = parse_query(
        r#"INFER EVIDENCE FOR s:Situation RETURN s AGGREGATE CHOQUET 'my-measure'"#,
    )
    .expect("choquet ref parses");
    let err = plan_query(&q).expect_err("choquet ref must fail at plan time");
    match err {
        TensaError::InvalidInput(msg) => {
            assert!(msg.contains("my-measure"), "message should name the ref");
            assert!(
                msg.contains("Phase 4") || msg.contains("/fuzzy/measures"),
                "should point at Phase 4 / measures registry, got: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

// ── T3 — OWA weights not summing to 1 → plan-time error ─────────────────────

#[test]
fn test_owa_weights_must_sum_to_one() {
    let q = parse_query(r#"MATCH (e:Actor) RETURN e AGGREGATE OWA [0.5, 0.3, 0.1]"#)
        .expect("weights parse");
    let err = plan_query(&q).expect_err("non-unit-sum OWA must fail at plan time");
    match err {
        TensaError::InvalidInput(msg) => {
            assert!(
                msg.contains("sum to 1.0"),
                "message should mention the unit-sum requirement, got: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn test_owa_uniform_weights_accepted() {
    // Four 0.25 weights sum to exactly 1.0.
    let q = parse_query(
        r#"MATCH (e:Actor) RETURN e AGGREGATE OWA [0.25, 0.25, 0.25, 0.25]"#,
    )
    .expect("parse");
    plan_query(&q).expect("unit-sum weights plan cleanly");
}

// ── T4 — Valid config propagates to MATCH plan steps ────────────────────────

#[test]
fn test_valid_fuzzy_config_reaches_filter_properties() {
    let q = parse_query(
        r#"MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e WITH TNORM 'lukasiewicz'"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let filter_step = plan
        .steps
        .iter()
        .find(|s| matches!(s, PlanStep::FilterProperties { .. }))
        .expect("filter step present");
    match filter_step {
        PlanStep::FilterProperties { fuzzy_config, .. } => {
            assert!(
                matches!(fuzzy_config.tnorm, Some(TNormKind::Lukasiewicz)),
                "tnorm must resolve to Lukasiewicz"
            );
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_valid_config_reaches_submit_inference_job_params() {
    let q = parse_query(
        r#"INFER EVIDENCE FOR n:Narrative WHERE n.narrative_id = "n1" RETURN n WITH TNORM 'goguen' AGGREGATE MEAN"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    match &plan.steps[0] {
        PlanStep::SubmitInferenceJob {
            parameters,
            fuzzy_config,
            ..
        } => {
            // Plan-step slot is populated with resolved kinds.
            assert!(matches!(fuzzy_config.tnorm, Some(TNormKind::Goguen)));
            assert!(matches!(fuzzy_config.aggregator, Some(AggregatorKind::Mean)));
            // Engine-visible parameters JSON carries the same resolved config.
            let fz = parameters
                .get("fuzzy_config")
                .expect("parameters.fuzzy_config slot populated");
            assert!(fz.is_object(), "fuzzy_config must round-trip via JSON");
        }
        other => panic!("expected SubmitInferenceJob, got {other:?}"),
    }
}

// ── T5 — Plan JSON (EXPLAIN payload) includes fuzzy_config ──────────────────

#[test]
fn test_explain_plan_json_includes_fuzzy_config() {
    let q = parse_query(
        r#"EXPLAIN MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e WITH TNORM 'lukasiewicz'"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let json = serde_json::to_value(&plan).expect("plan serialize");
    let steps = json
        .get("steps")
        .and_then(|v| v.as_array())
        .expect("steps array");
    // Find the FilterProperties step.
    let filter = steps
        .iter()
        .find(|s| s.get("FilterProperties").is_some())
        .expect("FilterProperties present in EXPLAIN");
    let inner = filter.get("FilterProperties").unwrap();
    let fuzzy = inner
        .get("fuzzy_config")
        .expect("fuzzy_config serialised into EXPLAIN plan");
    assert_eq!(
        fuzzy
            .get("tnorm")
            .and_then(|t| t.get("kind"))
            .and_then(|k| k.as_str()),
        Some("lukasiewicz"),
        "EXPLAIN must reveal the chosen t-norm"
    );
}

// ── T6 — Mean / Median / TNormReduce / TConormReduce all plan cleanly ───────

#[test]
fn test_all_scalar_aggregators_resolve_cleanly() {
    let cases: &[(&str, fn(&AggregatorKind) -> bool)] = &[
        ("MEAN", |k| matches!(k, AggregatorKind::Mean)),
        ("MEDIAN", |k| matches!(k, AggregatorKind::Median)),
        ("TNORM_REDUCE 'godel'", |k| {
            matches!(k, AggregatorKind::TNormReduce(TNormKind::Godel))
        }),
        ("TCONORM_REDUCE 'goguen'", |k| {
            matches!(k, AggregatorKind::TConormReduce(TNormKind::Goguen))
        }),
    ];
    for (clause, pred) in cases {
        let q = parse_query(&format!(
            r#"MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e AGGREGATE {clause}"#
        ))
        .unwrap_or_else(|e| panic!("{clause} must parse: {e}"));
        let plan = plan_query(&q).unwrap_or_else(|e| panic!("{clause} must plan: {e}"));
        let step = plan
            .steps
            .iter()
            .find(|s| matches!(s, PlanStep::FilterProperties { .. }))
            .expect("filter step present");
        match step {
            PlanStep::FilterProperties { fuzzy_config, .. } => {
                let agg = fuzzy_config.aggregator.as_ref().expect("aggregator present");
                assert!(pred(agg), "{clause} → unexpected aggregator {agg:?}");
            }
            _ => unreachable!(),
        }
    }
}

// ── T7 — Statement-carrier variants route fuzzy config to plan step ────────

#[test]
fn test_opinion_dynamics_plan_carries_fuzzy_config() {
    let stmt = parse_statement(
        r#"INFER OPINION_DYNAMICS( confidence_bound := 0.3, variant := 'pairwise' ) FOR "n1" WITH TNORM 'lukasiewicz'"#,
    )
    .unwrap();
    let plan = plan_statement(&stmt).expect("plan ok");
    match &plan.steps[0] {
        PlanStep::RunOpinionDynamics { fuzzy_config, .. } => {
            assert!(matches!(fuzzy_config.tnorm, Some(TNormKind::Lukasiewicz)));
        }
        other => panic!("expected RunOpinionDynamics, got {other:?}"),
    }
}

#[test]
fn test_opinion_phase_transition_plan_rejects_unknown_tnorm() {
    let stmt = parse_statement(
        r#"INFER OPINION_PHASE_TRANSITION( c_start := 0.05, c_end := 0.5, c_steps := 10 ) FOR "n1" WITH TNORM 'bogus'"#,
    )
    .unwrap();
    let err = plan_statement(&stmt).expect_err("unknown tnorm must fail");
    assert!(matches!(err, TensaError::InvalidInput(_)));
}

#[test]
fn test_hypergraph_reconstruction_plan_carries_fuzzy_config() {
    let stmt = parse_statement(
        r#"INFER HYPERGRAPH FROM DYNAMICS FOR "corpus-1" AGGREGATE TNORM_REDUCE 'godel'"#,
    )
    .unwrap();
    let plan = plan_statement(&stmt).expect("plan ok");
    match &plan.steps[0] {
        PlanStep::SubmitHypergraphReconstructionJob {
            fuzzy_config,
            params_json,
            ..
        } => {
            assert!(matches!(
                fuzzy_config.aggregator,
                Some(AggregatorKind::TNormReduce(TNormKind::Godel))
            ));
            // Reconstruction does not leak the config into params_json; it
            // travels only on the plan-step slot (the engine doesn't consume
            // it in Phase 3 but the slot is load-bearing for EXPLAIN).
            assert!(!params_json
                .as_object()
                .map(|o| o.contains_key("fuzzy_config"))
                .unwrap_or(false));
        }
        other => panic!("expected SubmitHypergraphReconstructionJob, got {other:?}"),
    }
}

// ── T8 — Backward-compat: no tail → empty FuzzyPlanConfig on all steps ──────

#[test]
fn test_no_tail_yields_empty_plan_config() {
    let q =
        parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e").unwrap();
    let plan = plan_query(&q).unwrap();
    for step in &plan.steps {
        match step {
            PlanStep::FilterProperties { fuzzy_config, .. } => {
                assert!(
                    fuzzy_config.is_empty(),
                    "default MATCH must not populate fuzzy_config"
                );
            }
            PlanStep::VectorNear { fuzzy_config, .. } => {
                assert!(fuzzy_config.is_empty());
            }
            _ => {}
        }
    }
}

#[test]
fn test_explain_infer_plan_includes_fuzzy_config_param() {
    let q = parse_query(
        r#"EXPLAIN INFER CENTRALITY FOR n:Narrative WHERE n.narrative_id = "n1" RETURN n WITH TNORM 'hamacher'"#,
    )
    .unwrap();
    let plan = plan_query(&q).expect("plan ok");
    let json = serde_json::to_value(&plan).expect("serialize");
    let steps = json.get("steps").and_then(|v| v.as_array()).unwrap();
    let submit = steps
        .iter()
        .find(|s| s.get("SubmitInferenceJob").is_some())
        .expect("submit step present");
    let params = submit
        .get("SubmitInferenceJob")
        .and_then(|s| s.get("parameters"))
        .expect("parameters field");
    assert!(
        params.get("fuzzy_config").is_some(),
        "parameters must carry fuzzy_config in EXPLAIN JSON"
    );
}
