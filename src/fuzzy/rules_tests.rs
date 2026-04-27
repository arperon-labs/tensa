//! Fuzzy Sprint Phase 9 — Mamdani rule system tests.
//!
//! Hand-verified membership boundaries + firing-strength t-norm
//! reductions + Centroid / MeanOfMaxima / end-to-end evaluation + KV
//! round-trip + aggregator wire.
//!
//! Cites: [mamdani1975mamdani].

use chrono::Utc;
use uuid::Uuid;

use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::rules::{
    aggregate_consequents, build_rule, centroid_defuzz, defuzzify, evaluate_rule_set,
    find_rule_any_narrative, firing_strength, fuzzify_condition, mean_of_maxima_defuzz,
    resolve_variable, save_rule, validate_rule, Defuzzification, FuzzyCondition, FuzzyOutput,
    MembershipFunction, Rule, RuleSet, DEFAULT_DEFUZZ_BINS,
};
use crate::fuzzy::rules_store::{list_rules_for_narrative, load_rule};
use crate::fuzzy::tnorm::TNormKind;
use crate::store::memory::MemoryStore;
use crate::types::{Entity, EntityType, MaturityLevel};

// ── Fixture builders ────────────────────────────────────────────────────────

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

// ── T1 / T2 / T3 — membership correctness ───────────────────────────────────

#[test]
fn test_triangular_membership_boundary_and_center() {
    let mf = MembershipFunction::Triangular {
        a: 0.0,
        b: 0.5,
        c: 1.0,
    };
    // Boundaries are strictly zero.
    assert_eq!(mf.membership(0.0), 0.0);
    assert_eq!(mf.membership(1.0), 0.0);
    // Peak at b.
    assert!((mf.membership(0.5) - 1.0).abs() < 1e-12);
    // Midpoint of rising leg = 0.5 (half of 1.0).
    assert!((mf.membership(0.25) - 0.5).abs() < 1e-12);
    // Midpoint of falling leg = 0.5.
    assert!((mf.membership(0.75) - 0.5).abs() < 1e-12);
    // Outside support.
    assert_eq!(mf.membership(-0.1), 0.0);
    assert_eq!(mf.membership(1.5), 0.0);
}

#[test]
fn test_trapezoidal_membership_shoulders() {
    let mf = MembershipFunction::Trapezoidal {
        a: 0.0,
        b: 0.3,
        c: 0.7,
        d: 1.0,
    };
    assert_eq!(mf.membership(0.0), 0.0);
    assert!((mf.membership(0.3) - 1.0).abs() < 1e-12);
    assert!((mf.membership(0.5) - 1.0).abs() < 1e-12);
    assert!((mf.membership(0.7) - 1.0).abs() < 1e-12);
    assert_eq!(mf.membership(1.0), 0.0);
    // Rising leg midpoint — (x - a) / (b - a) = 0.15 / 0.3 = 0.5.
    assert!((mf.membership(0.15) - 0.5).abs() < 1e-12);
    // Falling leg midpoint — (d - x) / (d - c) = 0.15 / 0.3 = 0.5.
    assert!((mf.membership(0.85) - 0.5).abs() < 1e-12);
}

#[test]
fn test_gaussian_membership_mean_is_one() {
    let mf = MembershipFunction::Gaussian {
        mean: 0.5,
        sigma: 0.1,
    };
    assert!((mf.membership(0.5) - 1.0).abs() < 1e-12);
    // Tail decay.
    let mu_at_1sigma = mf.membership(0.6);
    let mu_at_2sigma = mf.membership(0.7);
    assert!(mu_at_1sigma < 1.0 && mu_at_1sigma > 0.5);
    assert!(mu_at_2sigma < mu_at_1sigma);
    // Symmetric around mean.
    assert!((mf.membership(0.4) - mf.membership(0.6)).abs() < 1e-12);
    // Degenerate sigma collapses to 0.0 (not NaN).
    let bad = MembershipFunction::Gaussian {
        mean: 0.0,
        sigma: 0.0,
    };
    assert_eq!(bad.membership(0.0), 0.0);
}

// ── T4 — firing strength t-norm reduce ──────────────────────────────────────

#[test]
fn test_rule_firing_strength_tnorm_reduce() {
    let ent = sample_entity(serde_json::json!({
        "a": 0.5,
        "b": 0.7,
        "c": 0.9,
    }));
    let mk_cond = |path: &str, mean: f64| FuzzyCondition {
        variable_path: path.into(),
        membership: MembershipFunction::Triangular {
            a: mean - 0.25,
            b: mean,
            c: mean + 0.25,
        },
        linguistic_term: "_".into(),
    };
    // Each μ ≈ 1.0 at the peak; offset the three peaks so we get three
    // distinct μ values we can combine under different t-norms.
    let ant = vec![
        mk_cond("a", 0.4),  // μ(0.5) on triangle peaking at 0.4 → (0.65-0.5)/(0.65-0.4) = 0.6
        mk_cond("b", 0.65), // μ(0.7) on triangle peaking at 0.65 → (0.9-0.7)/(0.9-0.65) = 0.8
        mk_cond("c", 0.8),  // μ(0.9) on triangle peaking at 0.8 → (1.05-0.9)/(1.05-0.8) = 0.6
    ];

    let mu_list: Vec<f64> = ant.iter().map(|c| fuzzify_condition(&ent, c)).collect();
    assert_eq!(mu_list.len(), 3);
    for m in &mu_list {
        assert!((0.0..=1.0).contains(m), "μ out of range: {}", m);
    }

    // Gödel = min.
    let mut rule = build_rule("t4-godel", "phase9-test", ant.clone(), dummy_output());
    rule.tnorm = TNormKind::Godel;
    let (godel_strength, _) = firing_strength(&ent, &rule);
    let expected_min = mu_list.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (godel_strength - expected_min).abs() < 1e-12,
        "Gödel strength {godel_strength} vs min {expected_min}"
    );

    // Goguen = product.
    let mut rule = build_rule("t4-goguen", "phase9-test", ant, dummy_output());
    rule.tnorm = TNormKind::Goguen;
    let (goguen_strength, _) = firing_strength(&ent, &rule);
    let expected_prod = mu_list.iter().product::<f64>();
    assert!(
        (goguen_strength - expected_prod).abs() < 1e-12,
        "Goguen strength {goguen_strength} vs product {expected_prod}"
    );
    // Sanity — product ≤ min for values in [0, 1].
    assert!(goguen_strength <= godel_strength + 1e-12);
}

fn dummy_output() -> FuzzyOutput {
    FuzzyOutput {
        variable: "out".into(),
        membership: MembershipFunction::Triangular {
            a: 0.0,
            b: 0.5,
            c: 1.0,
        },
        linguistic_term: "mid".into(),
    }
}

// ── T5 — Centroid defuzzification ──────────────────────────────────────────

#[test]
fn test_centroid_defuzzification() {
    // Single rule firing at strength 1.0 over a Gaussian bell at mean=0.5.
    // Centroid should be ≈ 0.5 (symmetric mass).
    let cons = FuzzyOutput {
        variable: "out".into(),
        membership: MembershipFunction::Gaussian {
            mean: 0.5,
            sigma: 0.1,
        },
        linguistic_term: "mid".into(),
    };
    let fired = vec![(1.0, &cons)];
    let (xs, mus) = aggregate_consequents(&fired, DEFAULT_DEFUZZ_BINS);
    let centroid = centroid_defuzz(&xs, &mus).expect("centroid");
    assert!(
        (centroid - 0.5).abs() < 1e-3,
        "centroid {centroid} far from 0.5"
    );

    // Hand-computed 3-bin case: μ=[0.2, 0.8, 0.5], xs=[0.1, 0.3, 0.5]
    // expected centroid = (0.1*0.2 + 0.3*0.8 + 0.5*0.5)/(0.2+0.8+0.5) = 0.51 / 1.5 = 0.34
    let centroid = centroid_defuzz(&[0.1, 0.3, 0.5], &[0.2, 0.8, 0.5]).unwrap();
    assert!((centroid - 0.34).abs() < 1e-12, "centroid = {centroid}");
}

// ── T6 — Mean of maxima ────────────────────────────────────────────────────

#[test]
fn test_mean_of_maxima_defuzzification() {
    // Two bins with the same max μ → mean of their x values.
    let xs = [0.1, 0.2, 0.3, 0.4, 0.5];
    let mus = [0.2, 0.8, 0.8, 0.3, 0.1];
    let mom = mean_of_maxima_defuzz(&xs, &mus).unwrap();
    assert!((mom - 0.25).abs() < 1e-12, "mean-of-maxima = {mom}");

    // Single max bin — mom should equal that bin's x.
    let mom = mean_of_maxima_defuzz(&[0.1, 0.9, 0.3], &[0.1, 1.0, 0.5]).unwrap();
    assert!((mom - 0.9).abs() < 1e-12);

    // All zero — None.
    assert!(mean_of_maxima_defuzz(&[0.1, 0.2], &[0.0, 0.0]).is_none());
}

// ── T7 — End-to-end evaluation ──────────────────────────────────────────────

#[test]
fn test_evaluate_rule_set_end_to_end() {
    let rule_a = elevated_risk_rule();
    // Build a complementary low-risk rule firing on low inflammatory score.
    let rule_b = build_rule(
        "low-disinfo-risk",
        "phase9-test",
        vec![FuzzyCondition {
            variable_path: "entity.properties.inflammatory_score".into(),
            membership: MembershipFunction::Triangular {
                a: 0.0,
                b: 0.1,
                c: 0.3,
            },
            linguistic_term: "low".into(),
        }],
        FuzzyOutput {
            variable: "disinfo_risk".into(),
            membership: MembershipFunction::Gaussian {
                mean: 0.2,
                sigma: 0.1,
            },
            linguistic_term: "low".into(),
        },
    );
    let set = RuleSet::new(vec![rule_a.clone(), rule_b.clone()]);
    // Entity that activates rule A only (high partisanship, high
    // inflammatory score).
    let ent_a = sample_entity(serde_json::json!({
        "political_alignment": 0.85,
        "inflammatory_score": 0.8,
    }));
    let eval_a = evaluate_rule_set(&set, &ent_a).unwrap();
    assert_eq!(eval_a.fired_rules.len(), 2);
    let fire_a = eval_a
        .fired_rules
        .iter()
        .find(|f| f.rule_name == "elevated-disinfo-risk")
        .unwrap();
    assert!(fire_a.firing_strength > 0.0, "rule A should fire");
    let fire_b = eval_a
        .fired_rules
        .iter()
        .find(|f| f.rule_name == "low-disinfo-risk")
        .unwrap();
    assert_eq!(fire_b.firing_strength, 0.0);
    let out = eval_a.defuzzified_output.expect("defuzzified output");
    assert!(
        (0.5..=1.0).contains(&out),
        "high-partisan entity defuzzifies to {out} (expected toward 0.8)"
    );

    // Entity that activates rule B only.
    let ent_b = sample_entity(serde_json::json!({
        "political_alignment": 0.0,
        "inflammatory_score": 0.1,
    }));
    let eval_b = evaluate_rule_set(&set, &ent_b).unwrap();
    let fire_b = eval_b
        .fired_rules
        .iter()
        .find(|f| f.rule_name == "low-disinfo-risk")
        .unwrap();
    assert!(fire_b.firing_strength > 0.0);
    let out = eval_b.defuzzified_output.expect("defuzzified output");
    assert!(
        out < 0.5,
        "low-inflammatory entity defuzzifies to {out} (expected toward 0.2)"
    );
}

// ── T8 — KV round-trip ─────────────────────────────────────────────────────

#[test]
fn test_kv_roundtrip() {
    let store = MemoryStore::new();
    let rule = elevated_risk_rule();
    let id = rule.id;
    save_rule(&store, "phase9-test", &rule).unwrap();
    let loaded = load_rule(&store, "phase9-test", &id).unwrap().unwrap();
    assert_eq!(loaded.id, id);
    assert_eq!(loaded.name, "elevated-disinfo-risk");
    assert_eq!(loaded.antecedent.len(), 2);
    assert_eq!(loaded.consequent.variable, "disinfo_risk");
    // List helper.
    let all = list_rules_for_narrative(&store, "phase9-test").unwrap();
    assert_eq!(all.len(), 1);
    // Cross-narrative find.
    let found = find_rule_any_narrative(&store, &id).unwrap();
    assert!(found.is_some());
}

// ── T9 — Aggregator over rules ──────────────────────────────────────────────

#[test]
fn test_aggregator_over_rules() {
    // Two rules firing at strengths we can predict; assert the
    // optional firing_aggregator wire returns the mean.
    let r1 = {
        let mut r = build_rule(
            "rule-1",
            "phase9-test",
            vec![FuzzyCondition {
                variable_path: "x".into(),
                membership: MembershipFunction::Triangular {
                    a: 0.0,
                    b: 0.5,
                    c: 1.0,
                },
                linguistic_term: "_".into(),
            }],
            dummy_output(),
        );
        r.tnorm = TNormKind::Godel;
        r
    };
    let r2 = build_rule(
        "rule-2",
        "phase9-test",
        vec![FuzzyCondition {
            variable_path: "x".into(),
            membership: MembershipFunction::Trapezoidal {
                a: 0.0,
                b: 0.2,
                c: 0.4,
                d: 0.6,
            },
            linguistic_term: "_".into(),
        }],
        dummy_output(),
    );
    let ent = sample_entity(serde_json::json!({"x": 0.3}));
    let mut set = RuleSet::new(vec![r1, r2]);
    set.firing_aggregator = Some(AggregatorKind::Mean);
    let eval = evaluate_rule_set(&set, &ent).unwrap();
    assert_eq!(eval.fired_rules.len(), 2);
    let mean = eval.firing_aggregate.expect("firing aggregate");
    let recomputed: f64 = eval
        .fired_rules
        .iter()
        .map(|f| f.firing_strength)
        .sum::<f64>()
        / 2.0;
    assert!((mean - recomputed).abs() < 1e-12);
}

// ── Extra coverage — variable resolution + default-defuzzification path ────

#[test]
fn test_variable_path_resolution() {
    let ent = sample_entity(serde_json::json!({
        "nested": {"score": 0.42},
        "flag": true,
        "label": "partisan",
    }));
    assert_eq!(
        resolve_variable(&ent, "entity.properties.nested.score"),
        Some(0.42)
    );
    assert_eq!(resolve_variable(&ent, "nested.score"), Some(0.42));
    // entity.confidence is f32-widened to f64 → compare with tolerance.
    let c = resolve_variable(&ent, "entity.confidence").unwrap();
    assert!((c - 0.9).abs() < 1e-6, "confidence path = {c}");
    assert_eq!(resolve_variable(&ent, "flag"), Some(1.0));
    assert_eq!(resolve_variable(&ent, "label"), Some(1.0));
    assert!(resolve_variable(&ent, "missing").is_none());
    // Reserved-name path always resolves.
    assert!(resolve_variable(&ent, "entity.entity_type").is_some());
}

#[test]
fn test_defuzzification_strategy_dispatch() {
    let xs = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mus = [0.2, 0.6, 1.0, 0.6, 0.2];
    let centroid = defuzzify(Defuzzification::Centroid, &xs, &mus).unwrap();
    assert!((centroid - 0.5).abs() < 1e-12, "centroid {centroid}");
    let mom = defuzzify(Defuzzification::MeanOfMaxima, &xs, &mus).unwrap();
    assert!((mom - 0.5).abs() < 1e-12, "mom {mom}");
}

#[test]
fn test_validate_rule_rejects_empty_antecedent() {
    let mut rule = elevated_risk_rule();
    rule.antecedent.clear();
    let err = validate_rule(&rule).unwrap_err();
    assert!(
        err.to_string().contains("antecedent is empty"),
        "unexpected error: {err}"
    );
}

// Integration tests (hypergraph + TensaQL + post-ingest helper) live in
// [`super::rules_integration_tests`] so this file stays under the
// 500-line cap.
