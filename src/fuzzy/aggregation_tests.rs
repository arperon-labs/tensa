//! Tests for Phase 2 aggregators — OWA + Choquet + dispatch.
//!
//! Cites: [yager1988owa] [grabisch1996choquet] [grabisch2000fuzzymeasure].

use super::aggregation::{aggregator_for, AggregatorKind, FuzzyMeasure};
use super::aggregation_choquet::{
    choquet, choquet_exact, choquet_monte_carlo, ChoquetResult, EXACT_N_CAP,
};
use super::aggregation_measure::{
    check_monotone, measure_from_mobius, mobius_from_measure, new_monotone, symmetric_additive,
    symmetric_optimistic, symmetric_pessimistic,
};
use super::aggregation_owa::{linguistic_weights, owa, owa_normalize, Quantifier};
use super::tnorm::TNormKind;

// ── OWA tests ────────────────────────────────────────────────────────────────

#[test]
fn test_owa_max_weights() {
    // weights = [1, 0, 0, 0] puts all mass on the largest element.
    let xs = [0.2, 0.9, 0.5, 0.1];
    let w = [1.0, 0.0, 0.0, 0.0];
    let got = owa(&xs, &w).expect("owa ok");
    assert!((got - 0.9).abs() < 1e-12, "expected max, got {}", got);
}

#[test]
fn test_owa_min_weights() {
    // weights = [0, 0, 0, 1] puts all mass on the smallest element.
    let xs = [0.2, 0.9, 0.5, 0.1];
    let w = [0.0, 0.0, 0.0, 1.0];
    let got = owa(&xs, &w).expect("owa ok");
    assert!((got - 0.1).abs() < 1e-12, "expected min, got {}", got);
}

#[test]
fn test_owa_uniform_weights() {
    // weights = [1/n; n] recovers the arithmetic mean.
    let xs = [0.2, 0.4, 0.6, 0.8];
    let w = vec![0.25_f64; 4];
    let got = owa(&xs, &w).expect("owa ok");
    let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!((got - mean).abs() < 1e-12, "expected {}, got {}", mean, got);
}

#[test]
fn test_owa_linguistic_most() {
    // xs sorted descending already: [0.9, 0.8, 0.7, 0.6, 0.5].
    // Q_most(r) = clamp((r - 0.3)/0.5, 0, 1), so with n=5 the weight
    // vector is [Q(0.2)-Q(0), Q(0.4)-Q(0.2), Q(0.6)-Q(0.4), Q(0.8)-Q(0.6), Q(1.0)-Q(0.8)].
    //  Q(0)=0, Q(0.2)=0 (below 0.3), Q(0.4)=0.2, Q(0.6)=0.6, Q(0.8)=1.0, Q(1.0)=1.0.
    //  w = [0, 0.2, 0.4, 0.4, 0].
    let xs = [0.9, 0.8, 0.7, 0.6, 0.5];
    let w = linguistic_weights(Quantifier::Most, 5).expect("weights");
    // Within 1e-12 after internal renormalisation (Σw already = 1 exactly
    // by telescoping, so no drift should appear).
    let sum: f64 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "Σw must be 1, got {}", sum);
    let expected_w = [0.0, 0.2, 0.4, 0.4, 0.0];
    for (got, expect) in w.iter().zip(expected_w.iter()) {
        assert!(
            (got - expect).abs() < 1e-12,
            "weight mismatch: got {}, expected {}",
            got,
            expect
        );
    }
    let got_val = owa(&xs, &w).expect("owa ok");
    // = 0.9·0 + 0.8·0.2 + 0.7·0.4 + 0.6·0.4 + 0.5·0 = 0.16 + 0.28 + 0.24 = 0.68
    assert!(
        (got_val - 0.68).abs() < 1e-12,
        "expected 0.68, got {}",
        got_val
    );
}

#[test]
fn test_owa_rejects_length_mismatch() {
    let xs = [0.5, 0.3];
    let w = [0.5, 0.3, 0.2];
    assert!(owa(&xs, &w).is_err());
}

#[test]
fn test_owa_rejects_non_unit_sum() {
    let xs = [0.5, 0.3];
    let w = [0.5, 0.3];
    let err = owa(&xs, &w).expect_err("non-unit sum must error");
    assert!(
        format!("{:?}", err).contains("sum to 1.0"),
        "error should mention unit sum; got {:?}",
        err
    );
}

#[test]
fn test_owa_normalize() {
    let mut w = [0.5_f64, 0.3, 0.2];
    // already sums to 1 — should remain unchanged within FP noise.
    owa_normalize(&mut w).expect("normalize ok");
    let sum: f64 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);

    let mut w2 = [2.0_f64, 4.0, 4.0];
    owa_normalize(&mut w2).expect("normalize ok");
    assert!((w2.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((w2[0] - 0.2).abs() < 1e-12);
    assert!((w2[1] - 0.4).abs() < 1e-12);
    assert!((w2[2] - 0.4).abs() < 1e-12);

    let mut w3 = [0.0_f64, 0.0];
    assert!(owa_normalize(&mut w3).is_err());
}

// ── Choquet tests ─────────────────────────────────────────────────────────────

#[test]
fn test_choquet_symmetric_additive_equals_mean() {
    // For n = 8 inputs, symmetric_additive should recover the arithmetic
    // mean within 1e-12.
    let xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let m = symmetric_additive(8).expect("measure");
    let got = choquet_exact(&xs, &m).expect("choquet ok");
    let mean: f64 = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(
        (got - mean).abs() < 1e-12,
        "expected mean {}, got {}",
        mean,
        got
    );
}

#[test]
fn test_choquet_symmetric_pessimistic_equals_min() {
    let xs = [0.1, 0.2, 0.3, 0.4];
    let m = symmetric_pessimistic(4).expect("measure");
    let got = choquet_exact(&xs, &m).expect("choquet ok");
    let min_x = *xs
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(
        (got - min_x).abs() < 1e-12,
        "expected min {}, got {}",
        min_x,
        got
    );
}

#[test]
fn test_choquet_symmetric_optimistic_equals_max() {
    let xs = [0.1, 0.2, 0.3, 0.4];
    let m = symmetric_optimistic(4).expect("measure");
    let got = choquet_exact(&xs, &m).expect("choquet ok");
    let max_x = *xs
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    assert!(
        (got - max_x).abs() < 1e-12,
        "expected max {}, got {}",
        max_x,
        got
    );
}

#[test]
fn test_choquet_monotonicity_enforced_at_construction() {
    // n = 3 → 8 entries indexed by bitmask:
    //   ∅=0, {0}=1, {1}=2, {0,1}=3, {2}=4, {0,2}=5, {1,2}=6, full=7.
    // Valid endpoints (∅=0, full=1), but μ({0})=0.5 > μ({0,1})=0.3 breaks
    // monotonicity (subset has larger mass than superset).
    let bad = vec![0.0_f64, 0.5, 0.1, 0.3, 0.1, 0.4, 0.3, 1.0];

    // Base FuzzyMeasure::new accepts this (endpoint + range only).
    let m = FuzzyMeasure::new(3, bad.clone()).expect("length+endpoint valid");
    let err = check_monotone(&m).expect_err("non-monotone must error");
    assert!(
        format!("{:?}", err).contains("monotonicity"),
        "error should mention monotonicity; got {:?}",
        err
    );

    // The wrapper constructor surfaces the same error up front.
    let err2 = new_monotone(3, bad).expect_err("non-monotone must error");
    assert!(
        format!("{:?}", err2).contains("monotonicity"),
        "wrapped error should mention monotonicity; got {:?}",
        err2
    );

    // A valid monotone measure passes both paths.
    let ok = vec![0.0_f64, 0.1, 0.2, 0.4, 0.3, 0.5, 0.6, 1.0];
    assert!(new_monotone(3, ok).is_ok());
}

#[test]
fn test_choquet_n8_exact() {
    // Hand-computed non-symmetric measure for n = 3. Indices by bitmask:
    // ∅=0, {0}=1, {1}=2, {0,1}=3, {2}=4, {0,2}=5, {1,2}=6, full=7.
    // Pick a monotone measure:
    //   μ(∅)=0, μ({0})=0.1, μ({1})=0.2, μ({0,1})=0.4,
    //   μ({2})=0.3, μ({0,2})=0.5, μ({1,2})=0.6, μ({0,1,2})=1.
    let vals = vec![0.0, 0.1, 0.2, 0.4, 0.3, 0.5, 0.6, 1.0];
    let m = new_monotone(3, vals).expect("monotone measure");

    // xs = [0.5, 0.3, 0.9]. Sorted ascending: 0.3 (idx 1), 0.5 (idx 0),
    // 0.9 (idx 2).
    //   x_(1) = 0.3, A_1 = {0, 1, 2} → μ = 1,      step contribution = 0.3 · 1 = 0.3
    //   x_(2) = 0.5, A_2 = {0, 2}    → μ = 0.5,    step = (0.5 - 0.3) · 0.5 = 0.1
    //   x_(3) = 0.9, A_3 = {2}       → μ = 0.3,    step = (0.9 - 0.5) · 0.3 = 0.12
    // Total = 0.52.
    let xs = [0.5, 0.3, 0.9];
    let got = choquet_exact(&xs, &m).expect("choquet ok");
    assert!(
        (got - 0.52).abs() < 1e-12,
        "hand-computed expected 0.52, got {}",
        got
    );
}

#[test]
fn test_choquet_n12_monte_carlo_determinism() {
    // n = 12 → above EXACT_N_CAP, forces MC path. Same seed → same result.
    assert!(12 > EXACT_N_CAP);
    let n = 12u8;
    let xs: Vec<f64> = (0..n as usize).map(|i| (i as f64) * 0.07).collect();
    let m = symmetric_additive(n).expect("measure");
    let r1 = choquet_monte_carlo(&xs, &m, 256, 42).expect("mc ok");
    let r2 = choquet_monte_carlo(&xs, &m, 256, 42).expect("mc ok");
    assert_eq!(r1, r2, "same seed must produce same MC result");
    assert!(r1.std_err.is_some());
}

#[test]
fn test_choquet_dispatch_picks_exact_or_mc() {
    // n = 3 → exact (std_err = None).
    let xs_small = [0.1, 0.5, 0.9];
    let m_small = symmetric_additive(3).expect("m");
    let r = choquet(&xs_small, &m_small, 0).expect("ok");
    assert_eq!(r.std_err, None);

    // n = 11 → MC (std_err = Some).
    let n = 11u8;
    let xs_large: Vec<f64> = (0..n as usize).map(|i| i as f64 * 0.08).collect();
    let m_large = symmetric_additive(n).expect("m");
    let r2 = choquet(&xs_large, &m_large, 7).expect("ok");
    assert!(r2.std_err.is_some());
}

// ── Möbius roundtrip ─────────────────────────────────────────────────────────

#[test]
fn test_mobius_recovers_measure() {
    // Arbitrary valid monotone measure for n = 3.
    let vals = vec![0.0, 0.1, 0.2, 0.4, 0.3, 0.5, 0.6, 1.0];
    let m = new_monotone(3, vals.clone()).expect("m");
    let mob = mobius_from_measure(&m);
    let recovered = measure_from_mobius(3, &mob).expect("recovered");
    for (a, b) in m.values.iter().zip(recovered.values.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "Möbius round-trip mismatch: orig={}, got={}",
            a,
            b
        );
    }
}

// ── AggregatorKind dispatch ──────────────────────────────────────────────────

#[test]
fn test_aggregator_kind_dispatch() {
    let xs = [0.2_f64, 0.4, 0.6, 0.8];

    // Mean.
    let mean = aggregator_for(AggregatorKind::Mean);
    assert!((mean.aggregate(&xs).unwrap() - 0.5).abs() < 1e-12);

    // Median.
    let med = aggregator_for(AggregatorKind::Median);
    assert!((med.aggregate(&xs).unwrap() - 0.5).abs() < 1e-12);

    // OWA with descending "most" weights — linguistic_weights(Most, 4):
    //   Q(0)=0, Q(0.25)=0, Q(0.5)=0.4, Q(0.75)=0.9, Q(1)=1.
    //   w = [0, 0.4, 0.5, 0.1].
    // Sorted descending xs = [0.8, 0.6, 0.4, 0.2]. Aggregate:
    //   0.8·0 + 0.6·0.4 + 0.4·0.5 + 0.2·0.1 = 0.24 + 0.20 + 0.02 = 0.46.
    let w = linguistic_weights(Quantifier::Most, 4).unwrap();
    let owa_agg = aggregator_for(AggregatorKind::Owa(w));
    let owa_val = owa_agg.aggregate(&xs).unwrap();
    assert!(
        (owa_val - 0.46).abs() < 1e-12,
        "linguistic Most owa expected 0.46, got {}",
        owa_val
    );

    // Choquet with symmetric_additive recovers arithmetic mean.
    let measure = symmetric_additive(4).unwrap();
    let ch = aggregator_for(AggregatorKind::Choquet(measure));
    let ch_val = ch.aggregate(&xs).unwrap();
    assert!((ch_val - 0.5).abs() < 1e-12);

    // TNormReduce(Godel) = min.
    let tnorm = aggregator_for(AggregatorKind::TNormReduce(TNormKind::Godel));
    let tnorm_val = tnorm.aggregate(&xs).unwrap();
    assert!(
        (tnorm_val - 0.2).abs() < 1e-12,
        "Godel reduce expected 0.2, got {}",
        tnorm_val
    );

    // TConormReduce(Godel) = max.
    let tcon = aggregator_for(AggregatorKind::TConormReduce(TNormKind::Godel));
    let tcon_val = tcon.aggregate(&xs).unwrap();
    assert!((tcon_val - 0.8).abs() < 1e-12);
}

// ── ChoquetResult serialisability sanity ─────────────────────────────────────

#[test]
fn test_choquet_result_equality() {
    let a = ChoquetResult {
        value: 0.5,
        std_err: None,
    };
    let b = ChoquetResult {
        value: 0.5,
        std_err: None,
    };
    assert_eq!(a, b);
}
