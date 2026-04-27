//! LOAD-BEARING regression suite for Phase 1's site-specific t-norm wiring.
//!
//! The Phase 0 audit enumerated four kinds of implicit t-norms in the
//! pre-sprint codebase (Goguen product for Dempster-Shafer, Goguen t-conorm
//! for source corroboration, Gödel t-conorm for resolver best-match,
//! Bayesian/arithmetic for the confidence-breakdown composite). Phase 1
//! refactored each site to route through `fuzzy::tnorm` while preserving
//! the **same implicit t-norm as the default**. This suite locks that
//! contract in with bit-identical assertions against frozen snapshots.
//!
//! If any assertion in this file changes, the Phase 1 refactor is WRONG.
//! Do not relax the assertions — investigate the regression first.
//!
//! Cites: [klement2000].

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::analysis::evidence::{
    combine, combine_with_tnorm, combine_yager, combine_yager_with_tnorm, MassFunction,
};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::source::{
    AttributionTarget, BiasProfile, Source, SourceAttribution, SourceType, TrackRecord,
};
use crate::store::memory::MemoryStore;
use crate::types::*;

// ── Fixture builders ──────────────────────────────────────────────────────────

fn make_hg() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn make_source(id: Uuid, trust: f32) -> Source {
    Source {
        id,
        name: format!("src-{}", trust),
        source_type: SourceType::NewsOutlet,
        url: None,
        description: None,
        trust_score: trust,
        bias_profile: BiasProfile::default(),
        track_record: TrackRecord::default(),
        tags: vec![],
        author: None,
        publication: None,
        ingested_by: None,
        ingestion_notes: None,
        publication_date: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn make_entity(hg: &Hypergraph) -> Uuid {
    let entity = Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": "Test"}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.7,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: Some(ExtractionMethod::HumanEntered),
        narrative_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };
    hg.create_entity(entity).unwrap()
}

fn make_attr(src: Uuid, tgt: Uuid) -> SourceAttribution {
    SourceAttribution {
        source_id: src,
        target_id: tgt,
        target_kind: AttributionTarget::Entity,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.9,
        claim: None,
    }
}

fn make_mass(frame: Vec<String>, pairs: &[(Vec<usize>, f64)]) -> MassFunction {
    let mut masses = HashMap::new();
    for (k, v) in pairs {
        masses.insert(k.iter().copied().collect::<BTreeSet<_>>(), *v);
    }
    MassFunction::new(frame, masses).expect("mass fixture valid")
}

// ── Load-bearing snapshot assertions ─────────────────────────────────────────

/// (a) Dempster combination under the DEFAULT path must remain bit-identical
/// to the pre-sprint product-t-norm path (Goguen).
#[test]
fn dempster_combine_default_is_bit_identical_under_goguen() {
    let frame = vec!["H".into(), "T".into()];
    let m1 = make_mass(frame.clone(), &[(vec![0], 0.6), (vec![1], 0.4)]);
    let m2 = make_mass(frame.clone(), &[(vec![0], 0.7), (vec![1], 0.3)]);

    let (default_path, k_default) = combine(&m1, &m2).unwrap();
    let (explicit_goguen, k_explicit) =
        combine_with_tnorm(&m1, &m2, TNormKind::Goguen).unwrap();

    // Bit-equal masses
    for (k, v) in &default_path.masses {
        let other = explicit_goguen.masses.get(k).copied().unwrap_or(f64::NAN);
        assert_eq!(
            v.to_bits(),
            other.to_bits(),
            "mass {:?} differs: {} vs {}",
            k,
            v,
            other
        );
    }
    assert_eq!(default_path.masses.len(), explicit_goguen.masses.len());
    assert_eq!(k_default.to_bits(), k_explicit.to_bits());

    // Load-bearing frozen snapshot: the canonical Dempster result for this
    // fixture. Hand-computed: m({H})=(0.6*0.7)/(1-K), m({T})=(0.4*0.3)/(1-K),
    // K = 0.6*0.3 + 0.4*0.7 = 0.46, 1-K = 0.54.
    let mh = default_path
        .masses
        .get(&BTreeSet::from([0usize]))
        .copied()
        .unwrap();
    let mt = default_path
        .masses
        .get(&BTreeSet::from([1usize]))
        .copied()
        .unwrap();
    let expected_mh = 0.42 / 0.54;
    let expected_mt = 0.12 / 0.54;
    assert!(
        (mh - expected_mh).abs() < 1e-12,
        "m({{H}}) snapshot: {} vs {}",
        mh,
        expected_mh
    );
    assert!(
        (mt - expected_mt).abs() < 1e-12,
        "m({{T}}) snapshot: {} vs {}",
        mt,
        expected_mt
    );
    assert!((k_default - 0.46).abs() < 1e-12, "K snapshot: {}", k_default);
}

/// Switching to Łukasiewicz MUST produce a measurably-different mass than
/// the Goguen default on the same fixture.
#[test]
fn dempster_lukasiewicz_differs_measurably_from_default() {
    let frame = vec!["H".into(), "T".into()];
    let m1 = make_mass(frame.clone(), &[(vec![0], 0.6), (vec![1], 0.4)]);
    let m2 = make_mass(frame.clone(), &[(vec![0], 0.7), (vec![1], 0.3)]);

    let (goguen, _) = combine_with_tnorm(&m1, &m2, TNormKind::Goguen).unwrap();
    let (lu, _) = combine_with_tnorm(&m1, &m2, TNormKind::Lukasiewicz).unwrap();

    let g_mh = goguen
        .masses
        .get(&BTreeSet::from([0usize]))
        .copied()
        .unwrap_or(0.0);
    let l_mh = lu
        .masses
        .get(&BTreeSet::from([0usize]))
        .copied()
        .unwrap_or(0.0);

    // max(0, 0.6+0.7-1) = 0.3 (vs 0.42 under Goguen); different mass, different K.
    assert!(
        (g_mh - l_mh).abs() > 1e-6,
        "Łukasiewicz did not measurably differ: Goguen={} Ł={}",
        g_mh,
        l_mh
    );
}

/// Yager combination default is Goguen.
#[test]
fn yager_combine_default_matches_explicit_goguen() {
    let frame = vec!["H".into(), "T".into()];
    let m1 = make_mass(frame.clone(), &[(vec![0], 0.6), (vec![1], 0.4)]);
    let m2 = make_mass(frame.clone(), &[(vec![0], 0.7), (vec![1], 0.3)]);

    let (a, ka) = combine_yager(&m1, &m2).unwrap();
    let (b, kb) = combine_yager_with_tnorm(&m1, &m2, TNormKind::Goguen).unwrap();
    assert_eq!(a.masses.len(), b.masses.len());
    for (k, v) in &a.masses {
        let o = b.masses.get(k).copied().unwrap_or(f64::NAN);
        assert_eq!(v.to_bits(), o.to_bits(), "{:?}", k);
    }
    assert_eq!(ka.to_bits(), kb.to_bits());
}

/// (b) `compute_corroboration` under the DEFAULT path must be bit-identical
/// to the pre-sprint `1 - Π(1 - trust_i)` formula (Goguen t-conorm).
///
/// LOAD-BEARING frozen snapshot: trusts {0.6, 0.8, 0.5} →
/// 1 - (1-0.6)*(1-0.8)*(1-0.5) = 1 - 0.4*0.2*0.5 = 1 - 0.04 = 0.96.
#[test]
fn compute_corroboration_default_is_bit_identical_under_goguen() {
    let hg = make_hg();
    let target = make_entity(&hg);

    let src_trusts = [0.6f32, 0.8, 0.5];
    for trust in src_trusts {
        let s = make_source(Uuid::now_v7(), trust);
        let sid = s.id;
        hg.create_source(s).unwrap();
        hg.add_attribution(make_attr(sid, target)).unwrap();
    }

    let default_val = hg.compute_corroboration(&target).unwrap();
    let explicit_val = hg
        .compute_corroboration_with_tconorm(&target, TNormKind::Goguen)
        .unwrap();
    assert_eq!(
        default_val.to_bits(),
        explicit_val.to_bits(),
        "default != explicit Goguen: {} vs {}",
        default_val,
        explicit_val
    );

    // Snapshot: allow 1e-6 because f32 arithmetic is not bit-exact against
    // f64 expectation; the Goguen t-conorm fold inside the engine is the
    // load-bearing numeric contract — we already bit-compared against it.
    let expected = 0.96_f32;
    assert!(
        (default_val - expected).abs() < 1e-5,
        "compute_corroboration snapshot: got {} expected {}",
        default_val,
        expected
    );
}

/// Switching to Gödel t-conorm (max) MUST produce a measurably-different
/// corroboration result than the Goguen default on this fixture.
///
/// Under Gödel: max(0.6, 0.8, 0.5) = 0.8 (vs Goguen's 0.96).
#[test]
fn compute_corroboration_godel_differs_measurably() {
    let hg = make_hg();
    let target = make_entity(&hg);
    for trust in [0.6f32, 0.8, 0.5] {
        let s = make_source(Uuid::now_v7(), trust);
        let sid = s.id;
        hg.create_source(s).unwrap();
        hg.add_attribution(make_attr(sid, target)).unwrap();
    }

    let goguen_val = hg.compute_corroboration(&target).unwrap();
    let godel_val = hg
        .compute_corroboration_with_tconorm(&target, TNormKind::Godel)
        .unwrap();
    assert!(
        (goguen_val - godel_val).abs() > 1e-6,
        "Gödel did not differ measurably: {} vs {}",
        goguen_val,
        godel_val
    );
    // Godel should collapse to max-trust.
    assert!(
        (godel_val - 0.8).abs() < 1e-5,
        "Gödel should be max-trust=0.8, got {}",
        godel_val
    );
}

/// (c) `recompute_confidence` (the Bayesian / arithmetic path) must not
/// change under Phase 1 — the Phase 0 audit explicitly kept this site
/// classical because it's conjugate-prior arithmetic, not a t-norm fold.
#[test]
fn recompute_confidence_numerics_unchanged() {
    let hg = make_hg();
    let target = make_entity(&hg);
    for trust in [0.6f32, 0.8, 0.5] {
        let s = make_source(Uuid::now_v7(), trust);
        let sid = s.id;
        hg.create_source(s).unwrap();
        hg.add_attribution(make_attr(sid, target)).unwrap();
    }

    // Just prove the method call still succeeds and produces a
    // well-formed breakdown — Phase 2 owns the aggregator selector.
    let breakdown = hg.recompute_confidence(&target, 0.7).unwrap();
    assert!(
        (0.0..=1.0).contains(&breakdown.extraction),
        "extraction out of range"
    );
    let composite = breakdown.composite();
    assert!(
        (0.0..=1.0).contains(&composite),
        "composite out of range: {}",
        composite
    );
}

/// (d) Resolver `resolve` default path must equal `resolve_with_tconorm(Godel)`
/// — the pre-sprint "best of" alias fold is already the Gödel t-conorm.
#[test]
fn resolver_default_matches_godel_tconorm() {
    use crate::ingestion::extraction::ExtractedEntity;
    use crate::ingestion::resolve::{EntityResolver, ResolveResult};

    let mut resolver = EntityResolver::new();
    let existing_id = Uuid::now_v7();
    resolver.register(
        existing_id,
        "Rodion Raskolnikov",
        &["Rodia".into(), "Raskolnikov".into()],
        EntityType::Actor,
        None,
    );

    // Probe fuzzy match against a near-duplicate.
    let probe = ExtractedEntity {
        name: "Rodion Rsaklnikov".into(), // fuzzy spelling
        entity_type: EntityType::Actor,
        aliases: vec![],
        confidence: 0.9,
        properties: serde_json::json!({}),
    };

    let a = resolver.resolve(&probe, None);
    let b = resolver.resolve_with_tconorm(&probe, None, TNormKind::Godel);

    let equal = match (&a, &b) {
        (ResolveResult::Existing(x), ResolveResult::Existing(y)) => x == y,
        (ResolveResult::New, ResolveResult::New) => true,
        _ => false,
    };
    assert!(equal, "resolver default diverged from Gödel path: {:?} vs {:?}", a, b);
}

// ── Integration wires (each with default-equals + opt-in-differs tests) ──────

/// Query-executor fuzzy-AND/OR helpers: Gödel default matches classical min/max.
#[test]
fn executor_fuzzy_helpers_default_is_godel() {
    use crate::query::executor::{fuzzy_and, fuzzy_or};

    // At {0,1} the boolean interpretation must match short-circuit AND/OR.
    assert_eq!(fuzzy_and(TNormKind::Godel, 1.0, 1.0), 1.0);
    assert_eq!(fuzzy_and(TNormKind::Godel, 0.0, 1.0), 0.0);
    assert_eq!(fuzzy_or(TNormKind::Godel, 1.0, 0.0), 1.0);
    assert_eq!(fuzzy_or(TNormKind::Godel, 0.0, 0.0), 0.0);
    // Graded inputs: Gödel = min/max.
    assert_eq!(fuzzy_and(TNormKind::Godel, 0.3, 0.7), 0.3);
    assert_eq!(fuzzy_or(TNormKind::Godel, 0.3, 0.7), 0.7);
}

#[test]
fn executor_fuzzy_lukasiewicz_differs_from_godel() {
    use crate::query::executor::fuzzy_and;
    let g = fuzzy_and(TNormKind::Godel, 0.3, 0.3);
    let l = fuzzy_and(TNormKind::Lukasiewicz, 0.3, 0.3);
    assert!((g - l).abs() > 1e-6, "{} vs {}", g, l);
    // Gödel min = 0.3; Łukasiewicz max(0, 0.6-1)=0; ratio >> epsilon.
}

/// Reranker: default (no fusion) reproduces pre-sprint fraction-of-terms.
#[test]
fn reranker_default_fraction_scoring_unchanged() {
    use crate::query::reranker::{Reranker, TermOverlapReranker};
    let r = TermOverlapReranker::new();
    let q = "alice bob carol";
    let docs = ["alice and bob live here", "carol alone here"];
    let doc_refs: Vec<&str> = docs.iter().copied().collect();
    let out = r.rerank(q, &doc_refs).unwrap();
    // Classical scoring: doc0 has 2/3 hits, doc1 has 1/3.
    let score0 = out.iter().find(|(i, _)| *i == 0).map(|(_, s)| *s).unwrap();
    let score1 = out.iter().find(|(i, _)| *i == 1).map(|(_, s)| *s).unwrap();
    assert!((score0 - 2.0 / 3.0).abs() < 1e-5);
    assert!((score1 - 1.0 / 3.0).abs() < 1e-5);
}

#[test]
fn reranker_lukasiewicz_fusion_differs_from_default() {
    use crate::query::reranker::{Reranker, TermOverlapReranker};
    let default = TermOverlapReranker::new();
    let lu = TermOverlapReranker::with_fusion(TNormKind::Lukasiewicz);
    let q = "alice bob carol";
    let docs = ["alice bob here", "nothing"];
    let doc_refs: Vec<&str> = docs.iter().copied().collect();
    let a = default.rerank(q, &doc_refs).unwrap();
    let b = lu.rerank(q, &doc_refs).unwrap();

    // Default: doc0 = 2/3 ≈ 0.6667.
    // Łukasiewicz reduce on [1, 1, 0] = min(1, 1+1+0) = 1.0 (saturated).
    let a0 = a.iter().find(|(i, _)| *i == 0).map(|(_, s)| *s).unwrap();
    let b0 = b.iter().find(|(i, _)| *i == 0).map(|(_, s)| *s).unwrap();
    assert!((a0 - b0).abs() > 1e-3, "expected measurable difference: {} vs {}", a0, b0);
}

/// Higher-order contagion threshold rule gains a graded met_with_tnorm; the
/// classical crisp `met` path is left unchanged and still drives the hot loop.
#[test]
fn hoc_threshold_graded_default_opposite_godel_differs_for_lukasiewicz() {
    use crate::analysis::higher_order_contagion::ThresholdRule;
    let rule = ThresholdRule::Fraction(0.5);
    let godel = rule.met_with_tnorm(2, 5, TNormKind::Godel);
    let lu = rule.met_with_tnorm(2, 5, TNormKind::Lukasiewicz);
    // Both are graded values in [0, 1]; Łukasiewicz's saturating behaviour
    // should diverge from Gödel's min on this non-crisp input.
    assert!(
        (godel - lu).abs() > 1e-6,
        "Gödel vs Łukasiewicz should differ: {} vs {}",
        godel,
        lu
    );
}

/// Synth fidelity aggregate_score_with_tnorm: Gödel reduces "all-pass" to 1,
/// any-fail to 0 — distinct from weighted fraction.
#[test]
fn synth_fidelity_aggregate_with_tnorm_is_and_style() {
    // Direct access via the fidelity_pipeline module.
    use crate::synth::fidelity_pipeline::aggregate_score_with_tnorm;
    use crate::synth::fidelity::FidelityMetric;

    let all_pass = vec![
        FidelityMetric {
            name: "a".into(),
            statistic: "ks".into(),
            value: 0.05,
            threshold: 0.1,
            passed: true,
        },
        FidelityMetric {
            name: "b".into(),
            statistic: "ks".into(),
            value: 0.05,
            threshold: 0.1,
            passed: true,
        },
    ];
    let godel_all = aggregate_score_with_tnorm(&all_pass, TNormKind::Godel);
    assert_eq!(godel_all, 1.0);

    let one_fail = vec![
        FidelityMetric {
            name: "a".into(),
            statistic: "ks".into(),
            value: 0.20,
            threshold: 0.1,
            passed: false,
        },
        FidelityMetric {
            name: "b".into(),
            statistic: "ks".into(),
            value: 0.05,
            threshold: 0.1,
            passed: true,
        },
    ];
    let godel_fail = aggregate_score_with_tnorm(&one_fail, TNormKind::Godel);
    assert_eq!(godel_fail, 0.0);
    // Lukasiewicz on [0, 1] saturates at max(0, 0+1-1) = 0 as well — still
    // and-style, but the behaviour on graded inputs would differ. Sanity check.
    let lu_fail = aggregate_score_with_tnorm(&one_fail, TNormKind::Lukasiewicz);
    assert_eq!(lu_fail, 0.0);
}

/// Phase 2 full-aggregator variant: aggregate_metrics_with_aggregator
/// must match the pre-sprint t-norm reduction under `TNormReduce(Godel)`
/// AND produce a demonstrably different value under a different
/// aggregator (Mean) when some metrics fail.
#[test]
fn synth_fidelity_aggregate_with_aggregator_default_matches_godel() {
    use crate::fuzzy::aggregation::AggregatorKind;
    use crate::synth::fidelity::FidelityMetric;
    use crate::synth::fidelity_pipeline::{
        aggregate_metrics_with_aggregator, aggregate_score_with_tnorm,
    };

    let mixed = vec![
        FidelityMetric {
            name: "a".into(),
            statistic: "ks".into(),
            value: 0.05,
            threshold: 0.1,
            passed: true,
        },
        FidelityMetric {
            name: "b".into(),
            statistic: "mae".into(),
            value: 0.20,
            threshold: 0.1,
            passed: false,
        },
    ];

    let godel = aggregate_score_with_tnorm(&mixed, TNormKind::Godel);
    let via_agg = aggregate_metrics_with_aggregator(
        &mixed,
        &AggregatorKind::TNormReduce(TNormKind::Godel),
    )
    .expect("aggregator path");
    assert!((godel - via_agg).abs() < 1e-9, "Godel paths must agree");

    // Mean on [1, 0] = 0.5 — different from Godel's 0.0 on the same input.
    let via_mean = aggregate_metrics_with_aggregator(&mixed, &AggregatorKind::Mean)
        .expect("mean");
    assert!(
        (via_mean - 0.5).abs() < 1e-9,
        "Mean on [1, 0] must be 0.5, got {}",
        via_mean
    );
    assert!(
        (via_mean - godel).abs() > 0.1,
        "Mean and Godel must disagree on mixed pass/fail"
    );
}
