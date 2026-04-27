//! Tests for the hybrid generator (EATH Phase 9).
//!
//! Two tests covering the design-doc worked example + the synthetic-provenance
//! invariant. Adversarial-feature-gated wargame substrate tests live in
//! [`crate::adversarial::wargame_substrate_tests`] (cfg `adversarial`).

use std::sync::Arc;

use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::calibrate::save_params;
use crate::synth::emit::{is_synthetic_entity, is_synthetic_situation};
use crate::synth::registry::SurrogateRegistry;
use crate::synth::types::EathParams;

use super::{generate_hybrid_hypergraph, HybridComponent, HYBRID_WEIGHT_TOLERANCE};

fn fresh_hg() -> crate::Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    crate::Hypergraph::new(store)
}

/// Build calibrated EathParams with controlled mean group size and persist
/// them under `(narrative_id, "eath")`. `mean_group_size` is set by writing a
/// single-bin group_size_distribution at `mean_group_size - 2` so
/// `sample_group_size` returns exactly that value every draw.
fn seed_params(
    hg: &crate::Hypergraph,
    narrative_id: &str,
    n_entities: usize,
    mean_group_size: usize,
) {
    let max_g = mean_group_size.max(2) + 2;
    // group_size_distribution[k] = count for size k+2. Spike at the desired bin.
    let mut hist = vec![0_u32; max_g - 1];
    let bin = mean_group_size.saturating_sub(2);
    let last = hist.len().saturating_sub(1);
    if bin < hist.len() {
        hist[bin] = 100;
    } else {
        hist[last] = 100;
    }
    let params = EathParams {
        a_t_distribution: vec![0.5; n_entities],
        a_h_distribution: vec![1.0; n_entities],
        lambda_schedule: vec![1.0],
        p_from_scratch: 0.9, // mostly fresh recruits → group sizes track histogram
        omega_decay: 0.95,
        group_size_distribution: hist,
        rho_low: 0.5,
        rho_high: 0.3,
        xi: 1.0,
        order_propensity: vec![],
        max_group_size: max_g,
        stm_capacity: 7,
        num_entities: n_entities,
    };
    save_params(hg.store(), narrative_id, "eath", &params).unwrap();
}

#[test]
fn test_hybrid_mean_group_size_matches_weighted_components() {
    // Worked example from docs/synth_hybrid_semantics.md: source A with
    // mean group size 2, source B with mean group size 5, weights 0.7/0.3.
    // Predicted output mean: 0.7 * 2 + 0.3 * 5 = 2.9.
    let hg = fresh_hg();
    seed_params(&hg, "src-a", 12, 2);
    seed_params(&hg, "src-b", 12, 5);

    let registry = SurrogateRegistry::default();
    let components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: 0.7,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 0.3,
        },
    ];

    let summary = generate_hybrid_hypergraph(
        &components,
        "synth-hybrid-out",
        42,
        1000,
        &hg,
        &registry,
    )
    .expect("hybrid generation succeeds");

    assert!(
        summary.num_situations > 0,
        "hybrid run should emit at least one situation"
    );
    assert!(
        summary.num_participations >= summary.num_situations * 2,
        "every emitted situation has at least 2 members"
    );

    // Empirical mean group size = participations / situations.
    let mean = summary.num_participations as f64 / summary.num_situations as f64;
    let predicted = 0.7 * 2.0 + 0.3 * 5.0; // = 2.9
    // Initial 1000-step run produced empirical_mean ≈ 2.9129 against
    // predicted 2.9 — well under the ±0.6 tolerance below. The tolerance
    // is loose because real EATH recruitment can shrink groups when
    // activity is low and small-N statistical noise dominates at smaller
    // step counts; bumping num_steps tightens the bound at a linear cost.
    assert!(
        (mean - predicted).abs() < 0.6,
        "empirical mean group size {mean} should be within 0.6 of predicted {predicted}"
    );
}

#[test]
fn test_hybrid_emits_synthetic_provenance_on_all_records() {
    // Every entity + situation produced by hybrid generation MUST carry
    // `synthetic = true` in its properties JSON sidecar. Phase 3's
    // is_synthetic_* predicates are the load-bearing test surface.
    let hg = fresh_hg();
    seed_params(&hg, "src-a", 8, 3);
    seed_params(&hg, "src-b", 8, 3);

    let registry = SurrogateRegistry::default();
    let components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: 0.5,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 0.5,
        },
    ];

    let summary = generate_hybrid_hypergraph(
        &components,
        "synth-hybrid-prov",
        7,
        100,
        &hg,
        &registry,
    )
    .expect("hybrid generation succeeds");

    let entities = hg
        .list_entities_by_narrative("synth-hybrid-prov")
        .expect("list entities");
    assert!(!entities.is_empty(), "hybrid emits entities");
    assert!(
        entities.iter().all(is_synthetic_entity),
        "every entity must carry synthetic=true"
    );

    let situations = hg
        .list_situations_by_narrative("synth-hybrid-prov")
        .expect("list situations");
    assert!(!situations.is_empty(), "hybrid emits situations");
    assert!(
        situations.iter().all(is_synthetic_situation),
        "every situation must carry synthetic=true"
    );

    assert_eq!(summary.model, "hybrid");
    assert!(
        matches!(summary.kind, crate::synth::types::RunKind::Hybrid),
        "hybrid run summary kind = Hybrid (not Generation)"
    );
}

#[test]
fn test_hybrid_weights_must_sum_to_one() {
    // Weight vector {0.5, 0.3} sums to 0.8 ⇒ rejected at validation.
    let hg = fresh_hg();
    seed_params(&hg, "src-a", 4, 2);
    seed_params(&hg, "src-b", 4, 2);
    let registry = SurrogateRegistry::default();
    let components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: 0.5,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 0.3,
        },
    ];

    let err = generate_hybrid_hypergraph(&components, "out", 1, 10, &hg, &registry)
        .expect_err("non-summing weights should reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("sum to 1.0"),
        "error message should mention weight sum: {msg}"
    );

    // Edge: weights exactly inside tolerance are accepted.
    let edge_components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: 0.5 + HYBRID_WEIGHT_TOLERANCE / 2.0,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 0.5 - HYBRID_WEIGHT_TOLERANCE / 2.0,
        },
    ];
    generate_hybrid_hypergraph(&edge_components, "out-edge", 1, 5, &hg, &registry)
        .expect("within-tolerance weights should be accepted");
}

#[test]
fn test_hybrid_generation_is_deterministic_with_fixed_seed() {
    // Same seed + same params → bit-for-bit identical output (entity
    // count, situation count, participation count). Run on two FRESH
    // hypergraphs since UUIDs are minted from seed and would collide on
    // KV writes if reused.
    let hg_a = fresh_hg();
    seed_params(&hg_a, "src-a", 10, 3);
    seed_params(&hg_a, "src-b", 10, 3);

    let hg_b = fresh_hg();
    seed_params(&hg_b, "src-a", 10, 3);
    seed_params(&hg_b, "src-b", 10, 3);

    let registry = SurrogateRegistry::default();
    let components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: 0.6,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 0.4,
        },
    ];

    let s1 =
        generate_hybrid_hypergraph(&components, "out-det", 12345, 200, &hg_a, &registry).unwrap();
    let s2 =
        generate_hybrid_hypergraph(&components, "out-det", 12345, 200, &hg_b, &registry).unwrap();

    assert_eq!(s1.num_entities, s2.num_entities);
    assert_eq!(s1.num_situations, s2.num_situations);
    assert_eq!(s1.num_participations, s2.num_participations);
    assert_eq!(s1.run_id, s2.run_id, "same seed → same run_id");
    assert_eq!(s1.params_hash, s2.params_hash);
}

#[test]
fn test_hybrid_rejects_uncalibrated_source() {
    let hg = fresh_hg();
    // Only seed src-a; src-b is missing.
    seed_params(&hg, "src-a", 4, 2);
    let registry = SurrogateRegistry::default();
    let components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: 0.5,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 0.5,
        },
    ];

    let err = generate_hybrid_hypergraph(&components, "out", 1, 10, &hg, &registry)
        .expect_err("missing calibrated params should reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("calibrate the source first"),
        "error should hint at calibration: {msg}"
    );
}

#[test]
fn test_hybrid_rejects_negative_weight() {
    let hg = fresh_hg();
    seed_params(&hg, "src-a", 4, 2);
    seed_params(&hg, "src-b", 4, 2);
    let registry = SurrogateRegistry::default();
    let components = vec![
        HybridComponent {
            narrative_id: "src-a".into(),
            model: "eath".into(),
            weight: -0.1,
        },
        HybridComponent {
            narrative_id: "src-b".into(),
            model: "eath".into(),
            weight: 1.1,
        },
    ];

    let err = generate_hybrid_hypergraph(&components, "out", 1, 10, &hg, &registry)
        .expect_err("negative weight should reject");
    let msg = format!("{err}");
    assert!(msg.contains("< 0"), "error should mention negative: {msg}");
}
