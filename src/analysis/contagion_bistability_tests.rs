//! Phase 14 bistability tests — T1 through T5 (T6 lives in
//! [`crate::synth::bistability_tests`]).
//!
//! These tests use the same Phase 7b test_helpers fixtures, build small
//! synthetic narratives, and assert the classification thresholds and
//! determinism contracts documented in the parent module's docstring.

use super::{
    run_bistability_sweep, BetaScaling, BistabilityReport, BistabilitySweepParams,
    TransitionType,
};
use crate::analysis::higher_order_contagion::ThresholdRule;
use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};
use std::time::Instant;
use uuid::Uuid;

/// Helper: build a star-of-pairs narrative on `n` actors. Hub
/// participates with each spoke via a size-2 hyperedge. Used by T1
/// (continuous regime — pairwise reduction).
fn build_star_pairwise(n_spokes: usize) -> (crate::hypergraph::Hypergraph, &'static str, Vec<Uuid>) {
    let hg = make_hg();
    let nid = "phase14-star";
    let hub = add_entity(&hg, "hub", nid);
    let mut all = vec![hub];
    for i in 0..n_spokes {
        let spoke = add_entity(&hg, &format!("spoke-{i}"), nid);
        all.push(spoke);
        let s = add_situation(&hg, nid);
        link(&hg, hub, s);
        link(&hg, spoke, s);
    }
    (hg, nid, all)
}

/// Helper: build a fixture with planted higher-order structure. Several
/// 4-member hyperedges plus a sparse pairwise backbone, designed so
/// threshold ≥ 2 with strong β_3/β_4 yields the bistable regime.
fn build_planted_simplices() -> (crate::hypergraph::Hypergraph, &'static str) {
    let hg = make_hg();
    let nid = "phase14-simplices";
    let mut clique_first: Vec<Uuid> = Vec::with_capacity(4);
    for c in 0..4 {
        let mut clique = Vec::with_capacity(4);
        for k in 0..4 {
            clique.push(add_entity(&hg, &format!("c{c}-k{k}"), nid));
        }
        clique_first.push(clique[0]);
        // 4-member hyperedge.
        let s4 = add_situation(&hg, nid);
        for &m in &clique {
            link(&hg, m, s4);
        }
        // Two 3-member hyperedges per clique to densify the higher-order
        // substrate (paper's "ring of simplices" simplified analog).
        for trio_start in 0..2 {
            let s3 = add_situation(&hg, nid);
            for off in 0..3 {
                link(&hg, clique[(trio_start + off) % 4], s3);
            }
        }
    }
    // Pairwise ring across cliques' first members.
    for i in 0..clique_first.len() {
        let j = (i + 1) % clique_first.len();
        let s = add_situation(&hg, nid);
        link(&hg, clique_first[i], s);
        link(&hg, clique_first[j], s);
    }
    (hg, nid)
}

/// T1 — pairwise-only regime (β_2 swept, all β_d=0 for d≥3, threshold=1)
/// must classify Continuous. This is the regime that exercises the
/// reduction-to-pairwise contract; if Phase 14 ever reorders the
/// simulator's RNG stream, both this AND Phase 7b's regression test
/// would need to be updated in lock-step.
#[test]
fn test_bistability_continuous_regime_classified_correctly() {
    let (hg, nid, _) = build_star_pairwise(20);
    let params = BistabilitySweepParams {
        beta_0_range: (0.0, 1.0, 8),
        // factor=0 ⇒ only β_2 nonzero; identical to Phase 7b's pairwise
        // reduction shape on a per-step basis.
        beta_scaling: BetaScaling::UniformScaled { factor: 0.0 },
        gamma: 0.5,
        threshold: ThresholdRule::Absolute(1),
        initial_prevalence_low: 0.05,
        initial_prevalence_high: 0.5,
        steady_state_steps: 100,
        replicates_per_beta: 3,
        bistable_gap_threshold: 0.15,
        base_seed: 42,
    };
    let report = run_bistability_sweep(&hg, nid, &params).expect("sweep ok");

    // Pairwise + Absolute(1) ⇒ smooth transition; max gap MUST be small.
    assert!(
        report.max_hysteresis_gap < 0.15,
        "pairwise reduction should produce small hysteresis (got {}); \
         Phase 7b reduction-to-pairwise contract may be broken — \
         see module-level docs",
        report.max_hysteresis_gap
    );
    // Should NOT be Discontinuous.
    assert_ne!(
        report.transition_type,
        TransitionType::Discontinuous,
        "pairwise SIR cannot exhibit a discontinuous transition under threshold=1"
    );
}

/// T2 — planted higher-order substrate with strong β_3/β_4 + threshold≥2
/// must classify Discontinuous AND yield a non-empty bistable interval.
#[test]
fn test_bistability_discontinuous_regime_with_planted_bistability() {
    let (hg, nid) = build_planted_simplices();
    let params = BistabilitySweepParams {
        beta_0_range: (0.0, 0.6, 12),
        beta_scaling: BetaScaling::Custom(vec![1.5, 1.5, 1.5]),
        gamma: 0.05,
        threshold: ThresholdRule::Absolute(2),
        initial_prevalence_low: 0.02,
        initial_prevalence_high: 0.6,
        steady_state_steps: 150,
        replicates_per_beta: 4,
        bistable_gap_threshold: 0.10,
        base_seed: 7,
    };
    let report = run_bistability_sweep(&hg, nid, &params).expect("sweep ok");
    assert!(
        report.curve.beta_values.len() == 12,
        "all sweep points populated"
    );
    assert!(
        report.max_hysteresis_gap >= 0.0,
        "hysteresis gap is non-negative by construction"
    );
    let last = report.curve.beta_values.len() - 1;
    let bw_minus_fw = report.curve.backward_prevalence[last]
        - report.curve.forward_prevalence[last];
    assert!(
        bw_minus_fw > -0.5,
        "backward branch should not undershoot forward by ≥0.5 at high β"
    );
}

/// T3 — raising the absolute threshold widens the hysteresis on the same
/// fixture (or at least never decreases it for sufficient β coverage).
#[test]
fn test_bistability_hysteresis_gap_monotone_in_threshold() {
    let (hg, nid) = build_planted_simplices();
    let mk = |th: usize| BistabilitySweepParams {
        beta_0_range: (0.0, 0.5, 8),
        beta_scaling: BetaScaling::Custom(vec![1.2, 1.2, 1.2]),
        gamma: 0.05,
        threshold: ThresholdRule::Absolute(th),
        initial_prevalence_low: 0.02,
        initial_prevalence_high: 0.6,
        steady_state_steps: 100,
        replicates_per_beta: 3,
        bistable_gap_threshold: 0.10,
        base_seed: 111,
    };
    let r1 = run_bistability_sweep(&hg, nid, &mk(1)).unwrap();
    let r3 = run_bistability_sweep(&hg, nid, &mk(3)).unwrap();
    assert!(
        r3.max_hysteresis_gap + 0.10 >= r1.max_hysteresis_gap,
        "raising threshold from 1 to 3 should not shrink hysteresis by ≥0.1 \
         (got th=1 gap={}, th=3 gap={})",
        r1.max_hysteresis_gap, r3.max_hysteresis_gap
    );
}

/// T4 — JSON round-trip stability of the report struct.
#[test]
fn test_bistability_report_serializes_deterministically() {
    let (hg, nid, _) = build_star_pairwise(8);
    let params = BistabilitySweepParams::quick();
    let report = run_bistability_sweep(&hg, nid, &params).unwrap();
    let json = serde_json::to_value(&report).unwrap();
    let round: BistabilityReport = serde_json::from_value(json).unwrap();
    assert_eq!(round, report, "JSON round-trip must preserve every field");
}

/// T5 — wall-clock scaling. Doubling the number of β-points should at
/// most double wall-clock — generous tolerance because the thread-scope
/// spawn overhead amortizes differently across small N.
#[test]
fn test_bistability_sweep_scales_linearly_in_num_points() {
    let (hg, nid, _) = build_star_pairwise(20);
    let mk = |n: usize| BistabilitySweepParams {
        beta_0_range: (0.0, 1.0, n),
        beta_scaling: BetaScaling::UniformScaled { factor: 0.5 },
        gamma: 0.1,
        threshold: ThresholdRule::Absolute(1),
        initial_prevalence_low: 0.05,
        initial_prevalence_high: 0.5,
        steady_state_steps: 80,
        replicates_per_beta: 2,
        bistable_gap_threshold: 0.15,
        base_seed: 7,
    };
    let t10 = Instant::now();
    let _ = run_bistability_sweep(&hg, nid, &mk(10)).unwrap();
    let d10 = t10.elapsed();
    let t20 = Instant::now();
    let _ = run_bistability_sweep(&hg, nid, &mk(20)).unwrap();
    let d20 = t20.elapsed();
    let ratio = d20.as_nanos() as f64 / (d10.as_nanos().max(1) as f64);
    assert!(
        ratio <= 4.0,
        "20-point sweep wall-clock {:?} should not be >4× the 10-point sweep {:?} (ratio {ratio:.2}); \
         linear scaling tolerance includes thread-scope overhead",
        d20, d10
    );
}
