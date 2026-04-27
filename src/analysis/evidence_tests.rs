use super::*;

fn frame3() -> Vec<String> {
    vec!["A".into(), "B".into(), "C".into()]
}

// ─── Mass Function Tests ────────────────────────────────

#[test]
fn test_mass_function_creation() {
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.6);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.4);
    let mf = MassFunction::new(frame3(), masses).unwrap();
    assert_eq!(mf.frame.len(), 3);
}

#[test]
fn test_mass_function_invalid_sum() {
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.3);
    masses.insert(BTreeSet::from([1]), 0.3);
    // Sum = 0.6, not 1.0.
    let result = MassFunction::new(frame3(), masses);
    assert!(result.is_err());
}

#[test]
fn test_vacuous_mass_function() {
    let mf = MassFunction::vacuous(frame3());
    let full: BTreeSet<usize> = (0..3).collect();
    assert_eq!(mf.masses[&full], 1.0);
}

#[test]
fn test_categorical_mass_function() {
    let mf = MassFunction::categorical(frame3(), 1).unwrap();
    assert_eq!(mf.masses[&BTreeSet::from([1])], 1.0);
}

// ─── Combination Tests ──────────────────────────────────

#[test]
fn test_combine_agreeing_sources() {
    // Both sources strongly support hypothesis A.
    let mut m1 = HashMap::new();
    m1.insert(BTreeSet::from([0]), 0.8);
    m1.insert(BTreeSet::from([0, 1, 2]), 0.2);
    let mf1 = MassFunction {
        frame: frame3(),
        masses: m1,
    };

    let mut m2 = HashMap::new();
    m2.insert(BTreeSet::from([0]), 0.7);
    m2.insert(BTreeSet::from([0, 1, 2]), 0.3);
    let mf2 = MassFunction {
        frame: frame3(),
        masses: m2,
    };

    let (combined, k) = combine(&mf1, &mf2).unwrap();
    let bel_a = belief(&combined, &BTreeSet::from([0]));
    assert!(bel_a > 0.9, "Bel(A) = {}", bel_a);
    assert!(k < 0.1, "Conflict = {}", k);
}

#[test]
fn test_combine_conflicting_sources() {
    // Source 1 supports A, source 2 supports B.
    let mut m1 = HashMap::new();
    m1.insert(BTreeSet::from([0]), 0.9);
    m1.insert(BTreeSet::from([0, 1, 2]), 0.1);
    let mf1 = MassFunction {
        frame: frame3(),
        masses: m1,
    };

    let mut m2 = HashMap::new();
    m2.insert(BTreeSet::from([1]), 0.9);
    m2.insert(BTreeSet::from([0, 1, 2]), 0.1);
    let mf2 = MassFunction {
        frame: frame3(),
        masses: m2,
    };

    let (_, k) = combine(&mf1, &mf2).unwrap();
    assert!(k > 0.7, "Conflict should be high: K = {}", k);
}

#[test]
fn test_combine_three_sources_associative() {
    // Verify combining [m1, m2, m3] gives same result as ((m1⊕m2)⊕m3).
    let make_mf = |idx: usize, conf: f64| -> MassFunction {
        let mut m = HashMap::new();
        m.insert(BTreeSet::from([idx]), conf);
        m.insert(BTreeSet::from([0, 1, 2]), 1.0 - conf);
        MassFunction {
            frame: frame3(),
            masses: m,
        }
    };

    let mf1 = make_mf(0, 0.6);
    let mf2 = make_mf(0, 0.5);
    let mf3 = make_mf(1, 0.3);

    let (result_multi, _) = combine_multiple(&[mf1.clone(), mf2.clone(), mf3.clone()]).unwrap();
    let (step1, _) = combine(&mf1, &mf2).unwrap();
    let (result_seq, _) = combine(&step1, &mf3).unwrap();

    // Check belief values are the same.
    let bel_multi = belief(&result_multi, &BTreeSet::from([0]));
    let bel_seq = belief(&result_seq, &BTreeSet::from([0]));
    assert!(
        (bel_multi - bel_seq).abs() < 0.01,
        "multi={}, seq={}",
        bel_multi,
        bel_seq
    );
}

// ─── Belief / Plausibility Tests ────────────────────────

#[test]
fn test_belief_plausibility_singleton() {
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.3);
    masses.insert(BTreeSet::from([0, 1]), 0.4);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.3);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };

    let bel_a = belief(&mf, &BTreeSet::from([0]));
    let pl_a = plausibility(&mf, &BTreeSet::from([0]));

    // Bel(A) = m({A}) = 0.3
    assert!((bel_a - 0.3).abs() < 0.001, "Bel(A) = {}", bel_a);
    // Pl(A) = m({A}) + m({A,B}) + m({A,B,C}) = 1.0
    assert!((pl_a - 1.0).abs() < 0.001, "Pl(A) = {}", pl_a);
    // Uncertainty = Pl - Bel
    assert!(pl_a >= bel_a);
}

#[test]
fn test_belief_plausibility_composite() {
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.3);
    masses.insert(BTreeSet::from([1]), 0.2);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.5);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };

    let hyp_ab = BTreeSet::from([0, 1]);
    let bel_ab = belief(&mf, &hyp_ab);
    let pl_ab = plausibility(&mf, &hyp_ab);

    // Bel({A,B}) = m({A}) + m({B}) = 0.5
    assert!((bel_ab - 0.5).abs() < 0.001, "Bel(AB) = {}", bel_ab);
    // Pl({A,B}) = m({A}) + m({B}) + m(Θ) = 1.0
    assert!((pl_ab - 1.0).abs() < 0.001, "Pl(AB) = {}", pl_ab);
}

#[test]
fn test_uncertainty_interval() {
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.4);
    masses.insert(BTreeSet::from([1]), 0.3);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.3);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };

    let intervals = compute_intervals(&mf);
    for bp in &intervals {
        assert!(bp.belief <= bp.plausibility);
        assert!((bp.uncertainty - (bp.plausibility - bp.belief)).abs() < 0.001);
    }
}

// ─── Edge Cases ─────────────────────────────────────────

#[test]
fn test_vacuous_belief_plausibility() {
    let mf = MassFunction::vacuous(frame3());
    let intervals = compute_intervals(&mf);
    for bp in &intervals {
        assert!(bp.belief < 0.001, "Bel = {}", bp.belief);
        assert!(
            (bp.plausibility - 1.0).abs() < 0.001,
            "Pl = {}",
            bp.plausibility
        );
    }
}

#[test]
fn test_categorical_belief_plausibility() {
    let mf = MassFunction::categorical(frame3(), 0).unwrap();
    let intervals = compute_intervals(&mf);
    assert!((intervals[0].belief - 1.0).abs() < 0.001);
    assert!((intervals[0].plausibility - 1.0).abs() < 0.001);
    assert!(intervals[1].belief < 0.001);
    assert!(intervals[1].plausibility < 0.001);
}

#[test]
fn test_combine_with_vacuous() {
    // Combining with vacuous should return the other mass function.
    let mut m1 = HashMap::new();
    m1.insert(BTreeSet::from([0]), 0.7);
    m1.insert(BTreeSet::from([0, 1, 2]), 0.3);
    let mf1 = MassFunction {
        frame: frame3(),
        masses: m1,
    };

    let vacuous = MassFunction::vacuous(frame3());
    let (combined, k) = combine(&mf1, &vacuous).unwrap();

    let bel_orig = belief(&mf1, &BTreeSet::from([0]));
    let bel_combined = belief(&combined, &BTreeSet::from([0]));
    assert!(
        (bel_orig - bel_combined).abs() < 0.01,
        "orig={}, combined={}",
        bel_orig,
        bel_combined
    );
    assert!(k < 0.001);
}

// ─── Integration ────────────────────────────────────────

#[test]
fn test_integration_run_evidence() {
    use crate::source::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
    let n = "evidence_test";

    let sit_id = {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
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
            raw_content: vec![ContentBlock::text("claim")],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(n.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    };

    // Create two sources with attributions.
    let src1 = Source {
        id: Uuid::now_v7(),
        name: "Reuters".into(),
        source_type: SourceType::NewsOutlet,
        url: None,
        description: None,
        trust_score: 0.85,
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
    };
    let src2 = Source {
        id: Uuid::now_v7(),
        name: "AP".into(),
        source_type: SourceType::NewsOutlet,
        url: None,
        description: None,
        trust_score: 0.80,
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
    };

    let s1_id = hg.create_source(src1).unwrap();
    let s2_id = hg.create_source(src2).unwrap();

    for sid in [s1_id, s2_id] {
        hg.add_attribution(SourceAttribution {
            source_id: sid,
            target_id: sit_id,
            target_kind: AttributionTarget::Situation,
            retrieved_at: Utc::now(),
            original_url: None,
            excerpt: None,
            extraction_confidence: 0.9,
            claim: None,
        })
        .unwrap();
    }

    let result =
        run_evidence(&hg, n, vec!["suspect_A".into(), "suspect_B".into()], sit_id).unwrap();
    assert_eq!(result.frame.len(), 2);
    assert!(result.conflict < 0.5); // agreeing sources
    assert_eq!(result.belief_plausibility.len(), 2);
}

// ─── Claim-Aware Mass Assignment Tests ──────────────────

/// Helper: create a Hypergraph with MemoryStore, a situation, and sources with attributions.
/// Returns (hypergraph, situation_id, narrative_id).
fn setup_evidence_scenario(
    sources: &[(&str, f32, Option<&str>)], // (name, trust, claim)
) -> (Hypergraph, Uuid, String) {
    use crate::source::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
    let n = "claim_test".to_string();

    let sit_id = {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
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
            raw_content: vec![ContentBlock::text("evidence claim")],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(n.clone()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    };

    for (name, trust, claim) in sources {
        let src = Source {
            id: Uuid::now_v7(),
            name: name.to_string(),
            source_type: SourceType::NewsOutlet,
            url: None,
            description: None,
            trust_score: *trust,
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
        };
        let src_id = hg.create_source(src).unwrap();
        hg.add_attribution(SourceAttribution {
            source_id: src_id,
            target_id: sit_id,
            target_kind: AttributionTarget::Situation,
            retrieved_at: Utc::now(),
            original_url: None,
            excerpt: None,
            extraction_confidence: 0.9,
            claim: claim.map(|s| s.to_string()),
        })
        .unwrap();
    }

    (hg, sit_id, n)
}

#[test]
fn test_run_evidence_with_claim_concentrates_mass() {
    let (hg, sit_id, n) = setup_evidence_scenario(&[("Reuters", 0.9, Some("suspect_A"))]);
    let result = run_evidence(
        &hg,
        &n,
        vec!["suspect_A".into(), "suspect_B".into()],
        sit_id,
    )
    .unwrap();

    // Source with trust=0.9 claiming suspect_A should give high belief to suspect_A
    let bel_a = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([0]))
        .unwrap();
    let bel_b = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([1]))
        .unwrap();

    assert!(
        bel_a.belief > bel_b.belief,
        "Bel(A)={} should be > Bel(B)={}",
        bel_a.belief,
        bel_b.belief
    );
    assert!(
        bel_a.belief > 0.8,
        "Bel(A)={} should be > 0.8 with trust=0.9 claim",
        bel_a.belief
    );
}

#[test]
fn test_run_evidence_without_claim_distributes_uniformly() {
    let (hg, sit_id, n) = setup_evidence_scenario(&[("Reuters", 0.9, None)]);
    let result = run_evidence(
        &hg,
        &n,
        vec!["suspect_A".into(), "suspect_B".into()],
        sit_id,
    )
    .unwrap();

    // No claim: trust distributed uniformly, both hypotheses should have similar belief
    let bel_a = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([0]))
        .unwrap();
    let bel_b = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([1]))
        .unwrap();

    let diff = (bel_a.belief - bel_b.belief).abs();
    assert!(
        diff < 0.05,
        "Without claim, beliefs should be roughly equal: Bel(A)={}, Bel(B)={}, diff={}",
        bel_a.belief,
        bel_b.belief,
        diff
    );
}

#[test]
fn test_run_evidence_mixed_claims() {
    let (hg, sit_id, n) = setup_evidence_scenario(&[
        ("Reuters", 0.8, Some("suspect_A")),
        ("Blog", 0.7, Some("suspect_B")),
    ]);
    let result = run_evidence(
        &hg,
        &n,
        vec!["suspect_A".into(), "suspect_B".into()],
        sit_id,
    )
    .unwrap();

    // Conflicting claims should produce high conflict
    assert!(
        result.conflict > 0.3,
        "Conflict={} should be significant with opposing claims",
        result.conflict
    );

    // Both hypotheses should have meaningful belief
    let bel_a = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([0]))
        .unwrap();
    let bel_b = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([1]))
        .unwrap();

    assert!(
        bel_a.belief > 0.1,
        "Bel(A)={} should be > 0.1",
        bel_a.belief
    );
    assert!(
        bel_b.belief > 0.1,
        "Bel(B)={} should be > 0.1",
        bel_b.belief
    );
}

#[test]
fn test_run_evidence_invalid_claim_falls_back() {
    // Claim "unknown_person" not in the frame — should fall back to uniform distribution
    let (hg, sit_id, n) = setup_evidence_scenario(&[("Reuters", 0.9, Some("unknown_person"))]);
    let result = run_evidence(
        &hg,
        &n,
        vec!["suspect_A".into(), "suspect_B".into()],
        sit_id,
    )
    .unwrap();

    let bel_a = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([0]))
        .unwrap();
    let bel_b = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([1]))
        .unwrap();

    let diff = (bel_a.belief - bel_b.belief).abs();
    assert!(
        diff < 0.05,
        "Invalid claim should fall back to uniform: Bel(A)={}, Bel(B)={}, diff={}",
        bel_a.belief,
        bel_b.belief,
        diff
    );
}

#[test]
fn test_run_evidence_agreeing_claims() {
    let (hg, sit_id, n) = setup_evidence_scenario(&[
        ("Reuters", 0.8, Some("suspect_A")),
        ("AP", 0.7, Some("suspect_A")),
    ]);
    let result = run_evidence(
        &hg,
        &n,
        vec!["suspect_A".into(), "suspect_B".into()],
        sit_id,
    )
    .unwrap();

    // Two sources both claiming suspect_A should give very high belief
    let bel_a = result
        .belief_plausibility
        .iter()
        .find(|bp| bp.hypothesis == BTreeSet::from([0]))
        .unwrap();

    assert!(
        bel_a.belief > 0.9,
        "Bel(A)={} should be very high with two agreeing claims",
        bel_a.belief
    );

    // Low conflict since they agree
    assert!(
        result.conflict < 0.1,
        "Conflict={} should be low with agreeing claims",
        result.conflict
    );
}

// ─── Dirichlet / Evidential DL Tests ───────────────────

#[test]
fn test_dirichlet_from_ds() {
    // Single source strongly supporting hypothesis A.
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.8);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.2);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };
    let dc = dirichlet_from_mass(&mf, 1);
    assert_eq!(dc.alpha.len(), 3);
    // alpha_0 = 0.8*1 + 1 = 1.8
    assert!((dc.alpha[0] - 1.8).abs() < 0.01, "alpha_0={}", dc.alpha[0]);
    // alpha_1 = 0.0*1 + 1 = 1.0
    assert!((dc.alpha[1] - 1.0).abs() < 0.01, "alpha_1={}", dc.alpha[1]);
    assert!((dc.total_evidence - 0.8).abs() < 0.01);
}

#[test]
fn test_epistemic_uncertainty() {
    // Vacuous mass → maximum epistemic uncertainty.
    let mf = MassFunction::vacuous(frame3());
    let dc = dirichlet_from_mass(&mf, 1);
    // Zero evidence → alpha all 1.0 → S = K → epistemic = 1.0
    assert!(
        (dc.epistemic_uncertainty - 1.0).abs() < 0.01,
        "epistemic={}",
        dc.epistemic_uncertainty
    );

    // Strong evidence → low epistemic uncertainty.
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.9);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.1);
    let mf_strong = MassFunction {
        frame: frame3(),
        masses,
    };
    let dc_strong = dirichlet_from_mass(&mf_strong, 10);
    assert!(
        dc_strong.epistemic_uncertainty < 0.3,
        "epistemic={}",
        dc_strong.epistemic_uncertainty
    );
}

#[test]
fn test_aleatoric_uncertainty() {
    // Even split across hypotheses → high aleatoric.
    let mut masses = HashMap::new();
    let k = 3;
    for i in 0..k {
        masses.insert(BTreeSet::from([i]), 1.0 / k as f64);
    }
    let mf_even = MassFunction {
        frame: frame3(),
        masses,
    };
    let dc_even = dirichlet_from_mass(&mf_even, 10);
    assert!(
        dc_even.aleatoric_uncertainty > 0.9,
        "aleatoric={}",
        dc_even.aleatoric_uncertainty
    );

    // All mass on one hypothesis → low aleatoric.
    let mf_certain = MassFunction::categorical(frame3(), 0).unwrap();
    let dc_certain = dirichlet_from_mass(&mf_certain, 10);
    assert!(
        dc_certain.aleatoric_uncertainty < 0.5,
        "aleatoric={}",
        dc_certain.aleatoric_uncertainty
    );
}

#[test]
fn test_dirichlet_with_aggregator_mean_matches_default_path() {
    // Phase 2 backward-compat: when every hypothesis has a single mass
    // contribution, `dirichlet_from_mass_with_aggregator(&_, &Mean)` is
    // bit-identical to `dirichlet_from_mass` on singleton contributions.
    use crate::fuzzy::aggregation::AggregatorKind;
    let mut masses = std::collections::HashMap::new();
    masses.insert(std::collections::BTreeSet::from([0]), 0.8);
    masses.insert(std::collections::BTreeSet::from([1]), 0.1);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };
    let default = dirichlet_from_mass(&mf, 3);
    let via_agg = dirichlet_from_mass_with_aggregator(&mf, 3, &AggregatorKind::Mean)
        .expect("aggregator path");
    // alpha_0: 0.8·3 + 1 = 3.4 (singleton → mean of [0.8] = 0.8, identical).
    for (a, b) in default.alpha.iter().zip(via_agg.alpha.iter()) {
        assert!((a - b).abs() < 1e-10, "mean aggregator mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_dirichlet_with_aggregator_tnorm_reduce_differs() {
    use crate::fuzzy::aggregation::AggregatorKind;
    use crate::fuzzy::tnorm::TNormKind;
    // When a singleton belief has multiple mass contributions, the
    // Godel t-norm reduction (recovers min) should demonstrably differ
    // from the default additive accumulation. TNormReduce has no fixed
    // arity (unlike Choquet), so it composes cleanly with variable-size
    // contribution vectors across hypotheses.
    let mut masses = std::collections::HashMap::new();
    masses.insert(std::collections::BTreeSet::from([0]), 0.6);
    // Add the empty set as a degenerate "B ⊆ any hypothesis" contribution.
    masses.insert(std::collections::BTreeSet::new(), 0.2);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };
    let via_godel = dirichlet_from_mass_with_aggregator(
        &mf,
        1,
        &AggregatorKind::TNormReduce(TNormKind::Godel),
    )
    .expect("aggregator path");
    let default = dirichlet_from_mass(&mf, 1);
    // Default: alpha_0 = (0.6 + 0.2)·1 + 1 = 1.8.
    // Godel min on [0.6, 0.2]: 0.2 → alpha_0 = 0.2·1 + 1 = 1.2.
    assert!((default.alpha[0] - 1.8).abs() < 0.01);
    assert!(
        (via_godel.alpha[0] - 1.2).abs() < 0.01,
        "expected Godel min α_0 = 1.2, got {}",
        via_godel.alpha[0]
    );
}

#[test]
fn test_uncertainty_decomposition() {
    // With moderate evidence spread across 2 of 3 hypotheses:
    // both epistemic and aleatoric should be non-trivial.
    let mut masses = HashMap::new();
    masses.insert(BTreeSet::from([0]), 0.4);
    masses.insert(BTreeSet::from([1]), 0.3);
    masses.insert(BTreeSet::from([0, 1, 2]), 0.3);
    let mf = MassFunction {
        frame: frame3(),
        masses,
    };
    let dc = dirichlet_from_mass(&mf, 5);

    // Should have meaningful values in (0, 1) for both.
    assert!(
        dc.epistemic_uncertainty > 0.1 && dc.epistemic_uncertainty < 1.0,
        "epistemic={}",
        dc.epistemic_uncertainty
    );
    assert!(
        dc.aleatoric_uncertainty > 0.1 && dc.aleatoric_uncertainty < 1.0,
        "aleatoric={}",
        dc.aleatoric_uncertainty
    );
    // Serialization round-trip.
    let json = serde_json::to_vec(&dc).unwrap();
    let decoded: DirichletConfidence = serde_json::from_slice(&json).unwrap();
    assert_eq!(decoded.alpha.len(), 3);
    assert!((decoded.total_evidence - dc.total_evidence).abs() < 1e-10);
}

#[test]
fn test_inference_engine_trait() {
    let engine = EvidenceEngine;
    assert_eq!(engine.job_type(), InferenceJobType::EvidenceCombination);
}

#[test]
fn test_yager_combination() {
    let frame = vec!["guilty".to_string(), "innocent".to_string()];

    // Source 1: 70% guilty
    let mut m1_masses = HashMap::new();
    m1_masses.insert(BTreeSet::from([0]), 0.7); // guilty
    m1_masses.insert(BTreeSet::from([0, 1]), 0.3); // uncertain
    let m1 = MassFunction {
        frame: frame.clone(),
        masses: m1_masses,
    };

    // Source 2: 80% innocent (conflicting)
    let mut m2_masses = HashMap::new();
    m2_masses.insert(BTreeSet::from([1]), 0.8); // innocent
    m2_masses.insert(BTreeSet::from([0, 1]), 0.2); // uncertain
    let m2 = MassFunction {
        frame: frame.clone(),
        masses: m2_masses,
    };

    // Dempster's rule: normalizes conflict away
    let (dempster_result, dempster_conflict) = combine(&m1, &m2).unwrap();
    assert!(
        dempster_conflict > 0.5,
        "Expected high conflict, got {}",
        dempster_conflict
    );

    // Yager's rule: assigns conflict to Θ (universal set)
    let (yager_result, yager_conflict) = combine_yager(&m1, &m2).unwrap();
    assert_eq!(yager_conflict, dempster_conflict); // same K value

    // Yager should have mass on the full frame from conflict reallocation
    let full_set: BTreeSet<usize> = BTreeSet::from([0, 1]);
    let yager_theta_mass = yager_result.masses.get(&full_set).copied().unwrap_or(0.0);
    assert!(
        yager_theta_mass > 0.5,
        "Expected conflict mass reallocated to Θ, got {}",
        yager_theta_mass
    );

    // Dempster should NOT have that extra mass on Θ (conflict is normalized away)
    let dempster_theta_mass = dempster_result
        .masses
        .get(&full_set)
        .copied()
        .unwrap_or(0.0);
    assert!(yager_theta_mass > dempster_theta_mass);

    // Test combine_with_rule dispatches correctly
    let (via_rule, _) = combine_with_rule(&m1, &m2, CombinationRule::Yager).unwrap();
    let via_rule_theta = via_rule.masses.get(&full_set).copied().unwrap_or(0.0);
    assert!((via_rule_theta - yager_theta_mass).abs() < 1e-10);
}
