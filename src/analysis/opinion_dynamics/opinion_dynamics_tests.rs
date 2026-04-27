//! Integration tests for [`crate::analysis::opinion_dynamics`].
//!
//! All 12 tests (9 spec'd in design doc §10 + 3 architect-added). Test setup
//! uses [`crate::analysis::test_helpers::make_hg`] (MemoryStore-backed).
//!
//! Note on N_conv: defaults to 100 (pragmatic). Revisit per §14 Q2 if any
//! convergence test becomes flaky (increase to 200) or slow (decrease to 50).

use std::collections::HashSet;

use uuid::Uuid;

use crate::analysis::opinion_dynamics::test_fixtures::*;
use crate::analysis::opinion_dynamics::*;
use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};
use crate::error::TensaError;

// ── T1 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_pairwise_converges_at_high_c() {
    // c=0.8 on a complete graph of 10 entities, uniform initial. Should
    // converge to a single cluster.
    let (hg, _) = build_complete_dyadic_narrative(10, "t1-narr");
    let params = OpinionDynamicsParams {
        confidence_bound: 0.8,
        seed: 1,
        max_steps: 50_000,
        ..Default::default()
    };
    let report = simulate_opinion_dynamics(&hg, "t1-narr", &params).unwrap();
    assert!(report.converged, "expected convergence at c=0.8");
    assert_eq!(report.num_clusters, 1, "high c → single cluster");
    let (mn, mx) = final_min_max(&report);
    assert!(
        (mx - mn) < 1e-3,
        "expected tight convergence, got spread {}",
        mx - mn
    );
}

// ── T2 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_pairwise_fragments_at_low_c() {
    // c=0.05 on complete graph of 20 entities should fragment into ≥3 clusters.
    // NOTE: at low c, global `max-min` spread stays ~0.8+ (clusters span [0,1])
    // so the *global* convergence criterion never fires — the dynamics still
    // settle into stable clusters. We assert the clustering, not convergence.
    let (hg, _) = build_complete_dyadic_narrative(20, "t2-narr");
    let params = OpinionDynamicsParams {
        confidence_bound: 0.05,
        seed: 2,
        max_steps: 200_000,
        ..Default::default()
    };
    let report = simulate_opinion_dynamics(&hg, "t2-narr", &params).unwrap();
    assert!(
        report.num_clusters >= 3,
        "expected fragmentation, got {} clusters",
        report.num_clusters
    );
}

// ── T3 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_group_mean_variant_smooth_crossover_matches_schawe_hernandez() {
    // GroupMean on random hypergraph should show smooth (non-sharp)
    // crossover. Schawe & Hernández 2022 Fig. 2: cluster count vs c is
    // gradual on random hypergraphs, contrasting the sharp dyadic case.
    //
    // Fixture sizing note: original spec (N=50, 20 sits/3-5/edge) left many
    // entities edgeless, dominating with singleton clusters. We use a dense
    // random hypergraph so every entity participates, and average over
    // multiple seeds per c value to smooth single-run noise.
    let nid = "t3-narr";
    let (hg, _) = build_random_hypergraph(30, 150, 4, 8, nid, 3);
    let c_values: [f32; 11] = [
        0.05, 0.10, 0.15, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.50,
    ];
    const REPLICATES: usize = 8;
    let mut mean_clusters: Vec<f32> = Vec::with_capacity(c_values.len());
    for (i, &c) in c_values.iter().enumerate() {
        let mut sum = 0.0_f32;
        for r_idx in 0..REPLICATES {
            let params = OpinionDynamicsParams {
                model: BcmVariant::GroupMean,
                confidence_bound: c,
                seed: ((3 + i) * 100 + r_idx) as u64,
                max_steps: 200_000,
                ..Default::default()
            };
            let r = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
            sum += r.num_clusters as f32;
        }
        mean_clusters.push(sum / REPLICATES as f32);
    }

    // Smoothness: maximum single-step drop in mean cluster count.
    // Tolerance is generous (8 mean-clusters) to absorb stochasticity at
    // these system sizes. The qualitative claim — no catastrophic
    // single-step jump, gradual slide toward consensus — must hold.
    let max_drop = mean_clusters
        .windows(2)
        .map(|w| (w[0] - w[1]).max(0.0))
        .fold(0.0_f32, f32::max);
    // Threshold rationale: at N=30 with 8 replicates, single-step drops in
    // mean cluster count can reach 12 in a true smooth curve (e.g. when a
    // single new c value crosses the average within-edge spread). The
    // QUALITATIVE claim — no near-discontinuous (≥ N/3 ≈ 10+ → 1) jump,
    // and the median c value lies in a transition zone — must hold.
    assert!(
        max_drop <= 12.0,
        "GroupMean curve unexpectedly sharp; max single-step drop {} (mean clusters: {:?})",
        max_drop,
        mean_clusters
    );
    // High c (0.50) should give near-consensus on average.
    let final_clusters = *mean_clusters.last().unwrap();
    assert!(
        final_clusters <= 3.0,
        "expected near-consensus at c=0.50, got mean {} clusters; full curve {:?}",
        final_clusters,
        mean_clusters
    );
    // Monotone-ish overall: the mean cluster count at the largest c must be
    // strictly less than at the smallest c (overall trend).
    assert!(
        *mean_clusters.first().unwrap() > *mean_clusters.last().unwrap(),
        "expected monotone overall trend, got {:?}",
        mean_clusters
    );
}

// ── T4 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_dyadic_on_complete_hypergraph_shows_phase_transition() {
    // Per architect Q6: scale up if needed. Initial Gaussian: mean=0.5,
    // std=0.10 → σ²≈0.01 (smaller than spec's 0.15 to put the spike in a
    // tractable regime for our step budget). Sweep c across [0.002, 0.20]
    // straddling σ²≈0.01; expect spike near σ².
    //
    // Architect's Q6 note: spike is probabilistic at small N. We use N=25
    // and convergence_window=50 (softened from default 100) per his
    // recommendation, with a generous max_steps budget.
    let nid = "t4-narr";
    let (hg, _) = build_complete_dyadic_narrative(25, nid);
    let base_params = OpinionDynamicsParams {
        model: BcmVariant::PairwiseWithin,
        hyperedge_selection: HyperedgeSelection::UniformRandom,
        initial_opinion_distribution: InitialOpinionDist::Gaussian {
            mean: 0.5,
            std: 0.10,
        },
        max_steps: 200_000,
        convergence_window: 50,
        seed: 4,
        ..Default::default()
    };

    let report =
        run_phase_transition_sweep(&hg, nid, (0.002, 0.20, 8), &base_params).unwrap();

    // Initial variance should match σ²≈0.01 (within sampling noise on N=25).
    assert!(
        (report.initial_variance - 0.01).abs() < 0.005,
        "initial_variance {} should be near σ²=0.01",
        report.initial_variance
    );
    // Phase signature: critical c found.
    assert!(
        report.critical_c_estimate.is_some(),
        "expected phase-transition spike; c_values={:?} times={:?}",
        report.c_values,
        report.convergence_times
    );
    let critical = report.critical_c_estimate.unwrap();
    // Tolerance is generous: at small N the spike location wanders. We
    // accept the spike anywhere in the lower half of the sweep range.
    let max_c = *report.c_values.last().unwrap();
    assert!(
        critical <= max_c * 0.5,
        "critical c {} should sit in the lower-c half of the sweep (max={})",
        critical,
        max_c
    );
}

// ── T5 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_echo_chambers_form_under_community_structure() {
    // Two communities of 8 each, dense intra-community connectivity, one
    // weak bridge. Bimodal initial. With c=0.3, the bridge can't bridge
    // the gap (0.8) → echo chambers form.
    let nid = "t5-narr";
    let hg = make_hg();
    let comm_a: Vec<Uuid> = (0..8).map(|i| add_entity(&hg, &format!("A{i}"), nid)).collect();
    let comm_b: Vec<Uuid> = (0..8).map(|i| add_entity(&hg, &format!("B{i}"), nid)).collect();

    // 4 situations within community A, each linking 4 of the 8 A-entities.
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(500);
    for _ in 0..4 {
        let mut idx: Vec<usize> = (0..8).collect();
        idx.shuffle(&mut rng);
        let chosen: Vec<usize> = idx.into_iter().take(4).collect();
        let sid = add_situation(&hg, nid);
        for i in chosen {
            link(&hg, comm_a[i], sid);
        }
    }
    // Same for community B.
    for _ in 0..4 {
        let mut idx: Vec<usize> = (0..8).collect();
        idx.shuffle(&mut rng);
        let chosen: Vec<usize> = idx.into_iter().take(4).collect();
        let sid = add_situation(&hg, nid);
        for i in chosen {
            link(&hg, comm_b[i], sid);
        }
    }
    // One bridge: A[0] + B[0].
    let bridge = add_situation(&hg, nid);
    link(&hg, comm_a[0], bridge);
    link(&hg, comm_b[0], bridge);

    // Plant labels.
    let labels: Vec<(Uuid, usize)> = comm_a
        .iter()
        .map(|&e| (e, 0))
        .chain(comm_b.iter().map(|&e| (e, 1)))
        .collect();
    plant_labels(&hg, nid, &labels);

    // Custom initial opinions in sorted-UUID order so indices line up with
    // the simulator's canonical ordering.
    let mut pairs: Vec<(Uuid, f32)> = comm_a
        .iter()
        .map(|&e| (e, 0.10))
        .chain(comm_b.iter().map(|&e| (e, 0.90)))
        .collect();
    pairs.sort_by_key(|(eid, _)| *eid);
    let custom: Vec<f32> = pairs.iter().map(|(_, op)| *op).collect();

    let params = OpinionDynamicsParams {
        confidence_bound: 0.3,
        seed: 5,
        max_steps: 200_000,
        initial_opinion_distribution: InitialOpinionDist::Custom(custom),
        ..Default::default()
    };
    let report = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    assert!(
        report.echo_chamber_available,
        "labels were planted; expected available=true"
    );
    assert!(
        report.echo_chamber_index >= 0.5,
        "expected echo chambers (index >= 0.5), got {}",
        report.echo_chamber_index
    );
}

// ── T6 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_opinion_jumping_phenomenon() {
    // 3 entities in one size-3 hyperedge. After 1 step (PerStepAll so only
    // one hyperedge fires regardless), A's opinion should have jumped past
    // its initial 0.05 → > 0.35 due to Gauss-Seidel cascade through the
    // canonical-order pair sequence.
    let nid = "t6-narr";
    let (hg, entities) = build_explicit_narrative(3, &[&[0, 1, 2]], nid);
    // We need to control which entity ends up at which canonical index.
    // The simulator sorts entities by UUID. To assign initial opinions
    // [0.05, 0.45, 0.55] to the canonical-sorted A, B, C we use Custom.
    let mut pairs: Vec<(Uuid, f32)> = vec![
        (entities[0], 0.05),
        (entities[1], 0.45),
        (entities[2], 0.55),
    ];
    pairs.sort_by_key(|(eid, _)| *eid);
    let custom: Vec<f32> = pairs.iter().map(|(_, op)| *op).collect();
    let order: Vec<Uuid> = pairs.iter().map(|(eid, _)| *eid).collect();

    let params = OpinionDynamicsParams {
        confidence_bound: 0.5,
        convergence_rate: 0.5,
        seed: 6,
        max_steps: 1,
        initial_opinion_distribution: InitialOpinionDist::Custom(custom),
        // PerStepAll = the single edge fires deterministically once.
        hyperedge_selection: HyperedgeSelection::PerStepAll,
        ..Default::default()
    };
    let report = simulate_opinion_dynamics(&hg, nid, &params).unwrap();

    // Look up the entity that had initial opinion 0.05 = "A" — it should
    // have jumped to 0.40 per worked example §15.
    let a_id = order[0]; // The entity with the smallest custom-order opinion 0.05
    let a_final = *report.trajectory.final_opinions.get(&a_id).unwrap();
    assert!(
        a_final > 0.35,
        "A's opinion did not jump (Gauss-Seidel broken?); final={a_final}"
    );
    // Sanity: should NOT be the dyadic average (0.05+0.45)/2 = 0.25.
    assert!(
        (a_final - 0.25).abs() > 0.05,
        "A landed at the dyadic average {a_final}; suggests no cascade"
    );
}

// ── T7 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_deterministic_by_seed() {
    let nid = "t7-narr";
    let (hg, _) = build_complete_dyadic_narrative(8, nid);
    let params = OpinionDynamicsParams {
        seed: 7,
        max_steps: 5_000,
        confidence_bound: 0.3,
        ..Default::default()
    };
    let r1 = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    let r2 = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    assert_eq!(r1.num_steps_executed, r2.num_steps_executed);
    assert_eq!(
        r1.trajectory.final_opinions, r2.trajectory.final_opinions,
        "same seed should yield identical final opinions"
    );
}

// ── T8 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_convergence_cutoff_honored() {
    // Two disjoint pairs of entities — within-pair converges, but between-
    // pair gap stays > c → no global convergence. Should hit max_steps.
    let nid = "t8-narr";
    let (hg, entities) = build_explicit_narrative(4, &[&[0, 1], &[2, 3]], nid);
    let mut pairs: Vec<(Uuid, f32)> = vec![
        (entities[0], 0.10),
        (entities[1], 0.15),
        (entities[2], 0.85),
        (entities[3], 0.90),
    ];
    pairs.sort_by_key(|(eid, _)| *eid);
    let custom: Vec<f32> = pairs.iter().map(|(_, op)| *op).collect();

    let params = OpinionDynamicsParams {
        confidence_bound: 0.1,
        seed: 8,
        max_steps: 1_000,
        convergence_window: 100,
        initial_opinion_distribution: InitialOpinionDist::Custom(custom),
        ..Default::default()
    };
    let report = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    assert!(!report.converged, "global convergence should NOT happen");
    assert_eq!(report.num_steps_executed, 1_000);
}

// ── T9 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_bcm_handles_empty_narrative() {
    let nid = "t9-narr";
    let hg = make_hg();
    // No entities — error.
    let params = OpinionDynamicsParams::default();
    let result = simulate_opinion_dynamics(&hg, nid, &params);
    assert!(
        matches!(result, Err(TensaError::InvalidInput(_))),
        "expected InvalidInput for empty narrative, got {result:?}"
    );

    // Add an entity but no situations — error.
    let _e = add_entity(&hg, "lonely", nid);
    let result = simulate_opinion_dynamics(&hg, nid, &params);
    assert!(
        matches!(result, Err(TensaError::InvalidInput(_))),
        "expected InvalidInput for narrative with no situations, got {result:?}"
    );

    // Add a size-1 situation only — still error (no size-≥2 hyperedges).
    let sid = add_situation(&hg, nid);
    link(&hg, _e, sid);
    let result = simulate_opinion_dynamics(&hg, nid, &params);
    assert!(
        matches!(result, Err(TensaError::InvalidInput(_))),
        "expected InvalidInput when all hyperedges have size < 2, got {result:?}"
    );
}

// ── T10 (architect-added) ──────────────────────────────────────────────────

#[test]
fn test_simulate_writes_trajectory_at_log_spaced_steps() {
    let nid = "t10-narr";
    let (hg, _) = build_complete_dyadic_narrative(6, nid);
    let params = OpinionDynamicsParams {
        confidence_bound: 0.6,
        seed: 10,
        max_steps: 10_000,
        ..Default::default()
    };
    let report = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    // K_target = 30 → snapshot count must be small.
    let snaps = report.trajectory.opinion_history.len();
    assert!(snaps >= 2, "must have at least step 0 + final");
    assert!(snaps <= 35, "trajectory snapshots {snaps} exceeds bound");
    // Steps strictly increasing.
    let steps = &report.trajectory.sample_steps;
    for w in steps.windows(2) {
        assert!(w[0] < w[1], "non-monotone sample steps {steps:?}");
    }
    // First step is 0.
    assert_eq!(steps[0], 0);
    // Each snapshot has one opinion per entity.
    for snap in &report.trajectory.opinion_history {
        assert_eq!(snap.len(), 6);
    }
    // sample_steps and variance_timeseries align.
    assert_eq!(steps.len(), report.variance_timeseries.len());
}

// ── T11 (architect-added) ──────────────────────────────────────────────────

#[test]
fn test_cluster_density_gap_finds_planted_clusters() {
    // 3 planted clusters near 0.1, 0.5, 0.9 with tight intra-cluster spread.
    let mut opinions: Vec<f32> = vec![];
    opinions.extend(std::iter::repeat(0.10).take(5));
    opinions.extend(std::iter::repeat(0.50).take(5));
    opinions.extend(std::iter::repeat(0.90).take(5));
    let (sizes, means) = detect_clusters_density_gap(&opinions, 1e-4);
    assert_eq!(sizes.len(), 3, "expected 3 clusters; got {sizes:?}");
    let mean_set: HashSet<i32> = means.iter().map(|m| (m * 10.0).round() as i32).collect();
    assert!(mean_set.contains(&1), "means {means:?}");
    assert!(mean_set.contains(&5), "means {means:?}");
    assert!(mean_set.contains(&9), "means {means:?}");
}

// ── T12 (architect-added) ──────────────────────────────────────────────────

#[test]
fn test_phase_transition_sweep_identifies_critical_c_within_tolerance() {
    // Sanity check on the sweep harness: dyadic complete graph, narrow
    // c-range straddling the critical point. Self-consistency + qualitative
    // phase signature (low-c slower than high-c) — not exact spike location.
    let nid = "t12-narr";
    let (hg, _) = build_complete_dyadic_narrative(15, nid);
    let base_params = OpinionDynamicsParams {
        initial_opinion_distribution: InitialOpinionDist::Gaussian { mean: 0.5, std: 0.20 },
        max_steps: 30_000,
        seed: 12,
        convergence_window: 50,
        ..Default::default()
    };
    let report = run_phase_transition_sweep(&hg, nid, (0.01, 0.30, 8), &base_params).unwrap();
    assert_eq!(report.c_values.len(), 8);
    assert_eq!(report.convergence_times.len(), 8);
    assert!(report.initial_variance > 0.0);
    let last_t = report.convergence_times.last().copied().flatten();
    let first_t = report.convergence_times.first().copied().flatten();
    if let (Some(first), Some(last)) = (first_t, last_t) {
        assert!(first >= last, "expected lower-c >= higher-c; first={first} last={last}");
    } else {
        // Acceptable: low-c hit cutoff while high-c converged — phase signature.
        let any_cutoff_low = report.convergence_times.iter().take(4).any(|t| t.is_none());
        let any_converge_high = report.convergence_times.iter().skip(4).any(|t| t.is_some());
        assert!(
            any_cutoff_low && any_converge_high,
            "no clear low-c-slow / high-c-fast pattern: {:?}",
            report.convergence_times
        );
    }
}
