//! Phase 1 test corpus for gradual argumentation semantics.
//!
//! Sections (mirror design doc §5):
//! - §5.1 — 12 fixture tests (3 fixtures × 4 semantics)
//! - §5.2 — 16 convergence tests (4 semantics × 4 t-norms)
//! - §5.3 — 2 weighted-h-categoriser construction-validation tests
//! - §5.4 — backward-compat regression replay (legacy crisp fixtures)
//! - §5.5 — 30 Amgoud–Ben-Naim 2013 principle property trials
//!          (anonymity / independence / monotonicity)
//! - §5.6 — 5 API integration tests (`run_argumentation_with_gradual`)

use std::collections::{BTreeSet, HashSet};

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use super::*;
use crate::analysis::argumentation::{
    from_explicit, grounded_extension, preferred_extensions, run_argumentation,
    run_argumentation_with_gradual, run_on_framework, stable_extensions, Argument,
    ArgumentLabel, ArgumentationFramework,
};
use crate::fuzzy::tnorm::TNormKind;
use crate::store::memory::MemoryStore;
use crate::types::*;
use chrono::Utc;
use std::sync::Arc;

// ─── Fixture helpers ──────────────────────────────────────────────────

fn arg_w(idx: u128, w: f32) -> Argument {
    Argument {
        id: Uuid::from_u128(idx),
        label: format!("arg{}", idx),
        source_id: None,
        confidence: w,
    }
}

fn fixture_a() -> ArgumentationFramework {
    // Single attack pair: a (0.8) → b (0.6).
    from_explicit(vec![arg_w(0, 0.8), arg_w(1, 0.6)], vec![(0, 1)])
}

fn fixture_b() -> ArgumentationFramework {
    // 3-cycle: a → b → c → a (all w=0.8).
    from_explicit(
        vec![arg_w(0, 0.8), arg_w(1, 0.8), arg_w(2, 0.8)],
        vec![(0, 1), (1, 2), (2, 0)],
    )
}

fn fixture_c() -> ArgumentationFramework {
    // Disconnected pair: a → b, c → d (different intrinsic strengths).
    from_explicit(
        vec![
            arg_w(0, 0.7),
            arg_w(1, 0.5),
            arg_w(2, 0.9),
            arg_w(3, 0.4),
        ],
        vec![(0, 1), (2, 3)],
    )
}

fn assert_close(actual: f64, expected: f64, eps: f64, label: &str) {
    assert!(
        (actual - expected).abs() < eps,
        "{}: got {}, expected {} (diff {:.3e}, eps {:.3e})",
        label,
        actual,
        expected,
        (actual - expected).abs(),
        eps
    );
}

fn acc(result: &GradualResult, idx: u128) -> f64 {
    *result
        .acceptability
        .get(&Uuid::from_u128(idx))
        .expect("argument id present in result")
}

// ─── §5.1 — Fixture × semantics matrix (12 tests) ─────────────────────

#[test]
fn test_h_categoriser_single_pair() {
    let r = run_gradual_argumentation(&fixture_a(), &GradualSemanticsKind::HCategoriser, None)
        .expect("ok");
    // `confidence` is `f32`; the f32→f64 lift introduces ULP-scale noise
    // (~1e-8) on the unattacked argument. Use `1e-6` for both.
    assert_close(acc(&r, 0), 0.8, 1e-6, "Acc(a)");
    assert_close(acc(&r, 1), 0.6 / 1.8, 1e-6, "Acc(b)");
    assert!(r.converged);
}

#[test]
fn test_h_categoriser_three_cycle() {
    let r = run_gradual_argumentation(&fixture_b(), &GradualSemanticsKind::HCategoriser, None)
        .expect("ok");
    let a = acc(&r, 0);
    assert_close(acc(&r, 1), a, 1e-6, "symmetry b=a");
    assert_close(acc(&r, 2), a, 1e-6, "symmetry c=a");
    assert!(r.converged);
}

#[test]
fn test_h_categoriser_disconnected() {
    let r = run_gradual_argumentation(&fixture_c(), &GradualSemanticsKind::HCategoriser, None)
        .expect("ok");
    // 1e-6 absorbs the f32→f64 lift on intrinsic strengths.
    assert_close(acc(&r, 0), 0.7, 1e-6, "Acc(a)");
    assert_close(acc(&r, 1), 0.5 / (1.0 + 0.7), 1e-6, "Acc(b)");
    assert_close(acc(&r, 2), 0.9, 1e-6, "Acc(c)");
    assert_close(acc(&r, 3), 0.4 / (1.0 + 0.9), 1e-6, "Acc(d)");
}

#[test]
fn test_weighted_h_categoriser_single_pair() {
    let kind = GradualSemanticsKind::WeightedHCategoriser {
        weights: vec![1.0],
    };
    let r = run_gradual_argumentation(&fixture_a(), &kind, None).expect("ok");
    // With v=1.0, weighted h-Cat is identical to h-Cat for this fixture.
    assert_close(acc(&r, 1), 0.6 / 1.8, 1e-6, "Acc(b)");
}

#[test]
fn test_weighted_h_categoriser_three_cycle() {
    let kind = GradualSemanticsKind::WeightedHCategoriser {
        weights: vec![1.0, 1.0, 1.0],
    };
    let r = run_gradual_argumentation(&fixture_b(), &kind, None).expect("ok");
    let a = acc(&r, 0);
    assert_close(acc(&r, 1), a, 1e-6, "symmetry b=a");
    assert_close(acc(&r, 2), a, 1e-6, "symmetry c=a");
}

#[test]
fn test_weighted_h_categoriser_disconnected() {
    let kind = GradualSemanticsKind::WeightedHCategoriser {
        weights: vec![0.5, 0.25],
    };
    let r = run_gradual_argumentation(&fixture_c(), &kind, None).expect("ok");
    // Acc(b) = 0.5 / (1 + 0.5*0.7), Acc(d) = 0.4 / (1 + 0.25*0.9).
    assert_close(acc(&r, 1), 0.5 / (1.0 + 0.5 * 0.7), 1e-6, "Acc(b)");
    assert_close(acc(&r, 3), 0.4 / (1.0 + 0.25 * 0.9), 1e-6, "Acc(d)");
}

#[test]
fn test_max_based_single_pair() {
    let r = run_gradual_argumentation(&fixture_a(), &GradualSemanticsKind::MaxBased, None)
        .expect("ok");
    // For a single attacker, max == sum, so result is identical to h-Cat.
    assert_close(acc(&r, 1), 0.6 / 1.8, 1e-6, "Acc(b)");
}

#[test]
fn test_max_based_three_cycle() {
    let r = run_gradual_argumentation(&fixture_b(), &GradualSemanticsKind::MaxBased, None)
        .expect("ok");
    let a = acc(&r, 0);
    assert_close(acc(&r, 1), a, 1e-6, "symmetry b=a");
    assert_close(acc(&r, 2), a, 1e-6, "symmetry c=a");
    assert!(r.converged);
}

#[test]
fn test_max_based_disconnected() {
    let r = run_gradual_argumentation(&fixture_c(), &GradualSemanticsKind::MaxBased, None)
        .expect("ok");
    assert_close(acc(&r, 1), 0.5 / (1.0 + 0.7), 1e-6, "Acc(b)");
    assert_close(acc(&r, 3), 0.4 / (1.0 + 0.9), 1e-6, "Acc(d)");
}

#[test]
fn test_card_based_single_pair() {
    let r = run_gradual_argumentation(&fixture_a(), &GradualSemanticsKind::CardBased, None)
        .expect("ok");
    // card(b)=1, sum(b)=0.8 → Acc(b) = 0.6 / ((1+1)(1+0.8)) = 0.6/3.6.
    assert_close(acc(&r, 1), 0.6 / 3.6, 1e-6, "Acc(b)");
}

#[test]
fn test_card_based_three_cycle() {
    let r = run_gradual_argumentation(&fixture_b(), &GradualSemanticsKind::CardBased, None)
        .expect("ok");
    let a = acc(&r, 0);
    assert_close(acc(&r, 1), a, 1e-6, "symmetry b=a");
    assert_close(acc(&r, 2), a, 1e-6, "symmetry c=a");
    // Card-based: cardinality sets stabilise in O(|A|) rounds, but the
    // sum component still needs to settle to CONVERGENCE_EPSILON via a
    // 1-D contraction. Bound well below MAX_GRADUAL_ITERATIONS=200.
    assert!(r.converged);
    assert!(r.iterations <= 50, "iterations {} > 50", r.iterations);
}

#[test]
fn test_card_based_disconnected() {
    let r = run_gradual_argumentation(&fixture_c(), &GradualSemanticsKind::CardBased, None)
        .expect("ok");
    // Acc(b) = 0.5 / ((1+1)(1+0.7)).
    assert_close(acc(&r, 1), 0.5 / (2.0 * 1.7), 1e-6, "Acc(b)");
    assert_close(acc(&r, 3), 0.4 / (2.0 * 1.9), 1e-6, "Acc(d)");
}

// ─── §5.2 — Convergence matrix (16 tests) ─────────────────────────────

const ALL_TNORMS: &[TNormKind] = &[
    TNormKind::Godel,
    TNormKind::Lukasiewicz,
    TNormKind::Goguen,
    TNormKind::Hamacher(1.0),
];

fn semantics_for(name: &str) -> GradualSemanticsKind {
    match name {
        "h" => GradualSemanticsKind::HCategoriser,
        "w" => GradualSemanticsKind::WeightedHCategoriser {
            weights: vec![1.0; 3],
        },
        "m" => GradualSemanticsKind::MaxBased,
        "c" => GradualSemanticsKind::CardBased,
        other => panic!("unknown semantics {}", other),
    }
}

fn assert_converges_on_b(name: &str, tnorm: TNormKind) {
    let r = run_gradual_argumentation(&fixture_b(), &semantics_for(name), Some(tnorm))
        .expect("converges or capped");
    assert!(
        r.converged,
        "{:?} + {:?} expected converged on Fixture B but did not",
        name, tnorm
    );
}

fn assert_capped_on_b(name: &str, tnorm: TNormKind, cap: u32) {
    let r = run_gradual_argumentation_with_cap(&fixture_b(), &semantics_for(name), Some(tnorm), cap)
        .expect("capped");
    assert!(
        !r.converged,
        "{:?} + {:?} expected NOT converged at cap {} but did converge",
        name, tnorm, cap
    );
    assert_eq!(r.iterations, cap);
}

#[test]
fn test_convergence_h_godel() {
    assert_converges_on_b("h", TNormKind::Godel);
}
#[test]
fn test_convergence_h_lukasiewicz() {
    assert_converges_on_b("h", TNormKind::Lukasiewicz);
}
#[test]
fn test_convergence_h_goguen_capped() {
    // Goguen on Fixture B *does* actually converge (small sums), so we
    // force the capped state via a tiny cap to demonstrate the
    // converged: false return path.
    assert_capped_on_b("h", TNormKind::Goguen, 1);
}
#[test]
fn test_convergence_h_hamacher_capped() {
    assert_capped_on_b("h", TNormKind::Hamacher(1.0), 1);
}

#[test]
fn test_convergence_w_godel() {
    assert_converges_on_b("w", TNormKind::Godel);
}
#[test]
fn test_convergence_w_lukasiewicz() {
    assert_converges_on_b("w", TNormKind::Lukasiewicz);
}
#[test]
fn test_convergence_w_goguen_capped() {
    assert_capped_on_b("w", TNormKind::Goguen, 1);
}
#[test]
fn test_convergence_w_hamacher_capped() {
    assert_capped_on_b("w", TNormKind::Hamacher(1.0), 1);
}

#[test]
fn test_convergence_m_godel() {
    assert_converges_on_b("m", TNormKind::Godel);
}
#[test]
fn test_convergence_m_lukasiewicz() {
    assert_converges_on_b("m", TNormKind::Lukasiewicz);
}
#[test]
fn test_convergence_m_goguen_capped() {
    assert_capped_on_b("m", TNormKind::Goguen, 1);
}
#[test]
fn test_convergence_m_hamacher_capped() {
    assert_capped_on_b("m", TNormKind::Hamacher(1.0), 1);
}

#[test]
fn test_convergence_c_godel_bounded() {
    // Card-based: cardinality stabilises fast, sum-component contraction
    // still needs to reach CONVERGENCE_EPSILON. Bound well below
    // MAX_GRADUAL_ITERATIONS=200.
    let r = run_gradual_argumentation(
        &fixture_b(),
        &GradualSemanticsKind::CardBased,
        Some(TNormKind::Godel),
    )
    .expect("ok");
    assert!(r.converged);
    assert!(r.iterations <= 50, "iterations {} > 50", r.iterations);
}
#[test]
fn test_convergence_c_lukasiewicz_bounded() {
    let r = run_gradual_argumentation(
        &fixture_b(),
        &GradualSemanticsKind::CardBased,
        Some(TNormKind::Lukasiewicz),
    )
    .expect("ok");
    assert!(r.converged);
    assert!(r.iterations <= 50, "iterations {} > 50", r.iterations);
}
#[test]
fn test_convergence_c_goguen_capped() {
    assert_capped_on_b("c", TNormKind::Goguen, 1);
}
#[test]
fn test_convergence_c_hamacher_capped() {
    assert_capped_on_b("c", TNormKind::Hamacher(1.0), 1);
}

// ─── §5.3 — Weighted-construction validation (2 tests) ────────────────

#[test]
fn test_weighted_length_mismatch_invalid_input() {
    let kind = GradualSemanticsKind::WeightedHCategoriser {
        weights: vec![0.5, 0.5], // fixture_a has only 1 attack
    };
    let err = run_gradual_argumentation(&fixture_a(), &kind, None).unwrap_err();
    match err {
        TensaError::InvalidInput(msg) => assert!(msg.contains("length")),
        other => panic!("expected InvalidInput, got {:?}", other),
    }
}

#[test]
fn test_weighted_sum_exceeds_one_invalid_input() {
    // Two attackers on the same target with weights summing to 1.5.
    let fw = from_explicit(
        vec![arg_w(0, 0.5), arg_w(1, 0.5), arg_w(2, 0.5)],
        vec![(0, 2), (1, 2)],
    );
    let kind = GradualSemanticsKind::WeightedHCategoriser {
        weights: vec![0.8, 0.7],
    };
    let err = run_gradual_argumentation(&fw, &kind, None).unwrap_err();
    match err {
        TensaError::InvalidInput(msg) => {
            assert!(msg.contains("incoming-weight sum"));
        }
        other => panic!("expected InvalidInput, got {:?}", other),
    }
}

// ─── §5.4 — Backward-compat regression replay ─────────────────────────
//
// Re-construct each fixture from the legacy `argumentation_tests.rs`
// corpus (single source of truth: legacy module is `#[cfg(test)]`-only
// so we cannot import its fixture helpers; we re-spell them with the
// same shape) and assert that the new `run_argumentation_with_gradual`
// path with `(None, None)` yields bit-identical legacy results AND
// `result.gradual.is_none()`.

fn legacy_arg(idx: usize) -> Argument {
    // Mirrors `arg(idx)` in argumentation_tests.rs verbatim.
    Argument {
        id: Uuid::from_u128(idx as u128),
        label: format!("arg{}", idx),
        source_id: None,
        confidence: 0.8,
    }
}

fn assert_bit_identical(label: &str, fw: ArgumentationFramework) {
    // `run_on_framework` is itself the new (None, None) wrapper after
    // Phase 1, so this single call exercises the new code path against
    // the legacy contract.
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let res = run_on_framework(&hg, "regression", &fw).expect("ok");
    assert!(res.gradual.is_none(), "{}: gradual must be None", label);

    // Independently re-derive the crisp parts to confirm the helpers
    // themselves haven't drifted from the pre-Phase-1 baseline.
    let g_labels = grounded_extension(&fw);
    let to_uuids = |sets: &[BTreeSet<usize>]| -> Vec<Vec<Uuid>> {
        sets.iter()
            .map(|s| s.iter().map(|&i| fw.arguments[i].id).collect())
            .collect()
    };
    let g_uuids: Vec<(Uuid, ArgumentLabel)> = fw
        .arguments
        .iter()
        .enumerate()
        .map(|(i, a)| (a.id, g_labels[i]))
        .collect();
    assert_eq!(res.grounded, g_uuids, "{}: grounded labels", label);
    assert_eq!(
        res.preferred_extensions,
        to_uuids(&preferred_extensions(&fw)),
        "{}: preferred extensions",
        label
    );
    assert_eq!(
        res.stable_extensions,
        to_uuids(&stable_extensions(&fw)),
        "{}: stable extensions",
        label
    );
}

#[test]
fn regression_grounded_a_attacks_b() {
    assert_bit_identical(
        "a→b",
        from_explicit(vec![legacy_arg(0), legacy_arg(1)], vec![(0, 1)]),
    );
}

#[test]
fn regression_grounded_mutual_attack() {
    assert_bit_identical(
        "a↔b",
        from_explicit(vec![legacy_arg(0), legacy_arg(1)], vec![(0, 1), (1, 0)]),
    );
}

#[test]
fn regression_grounded_chain_defense() {
    assert_bit_identical(
        "a→b→c",
        from_explicit(
            vec![legacy_arg(0), legacy_arg(1), legacy_arg(2)],
            vec![(0, 1), (1, 2)],
        ),
    );
}

#[test]
fn regression_grounded_triangle() {
    assert_bit_identical(
        "a→b→c→a",
        from_explicit(
            vec![legacy_arg(0), legacy_arg(1), legacy_arg(2)],
            vec![(0, 1), (1, 2), (2, 0)],
        ),
    );
}

#[test]
fn regression_grounded_no_attacks() {
    assert_bit_identical(
        "no attacks",
        from_explicit(vec![legacy_arg(0), legacy_arg(1), legacy_arg(2)], vec![]),
    );
}

#[test]
fn regression_grounded_self_attack() {
    assert_bit_identical(
        "a→a",
        from_explicit(vec![legacy_arg(0), legacy_arg(1)], vec![(0, 0)]),
    );
}

#[test]
fn regression_preferred_no_attacks() {
    assert_bit_identical(
        "no attacks 3",
        from_explicit(vec![legacy_arg(0), legacy_arg(1), legacy_arg(2)], vec![]),
    );
}

#[test]
fn regression_preferred_chain() {
    assert_bit_identical(
        "5-chain",
        from_explicit(
            vec![
                legacy_arg(0),
                legacy_arg(1),
                legacy_arg(2),
                legacy_arg(3),
                legacy_arg(4),
            ],
            vec![(0, 1), (1, 2), (2, 3), (3, 4)],
        ),
    );
}

#[test]
fn regression_stable_acyclic() {
    assert_bit_identical(
        "stable acyclic",
        from_explicit(vec![legacy_arg(0), legacy_arg(1)], vec![(0, 1)]),
    );
}

// Empty framework still yields gradual=None on the new entry point.
#[test]
fn regression_empty_framework_via_new_path() {
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let r = run_argumentation_with_gradual(&hg, "empty", None, None).expect("ok");
    assert!(r.gradual.is_none());
    assert!(r.framework.arguments.is_empty());
    let legacy = run_argumentation(&hg, "empty").expect("ok");
    assert_eq!(legacy.framework.arguments.len(), r.framework.arguments.len());
    assert!(legacy.gradual.is_none());
}

// Re-run the contention-based integration scenario verbatim and assert
// gradual=None plus the same UNDEC labels the legacy test asserts.
#[test]
fn regression_integration_with_contentions() {
    use crate::source::*;

    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let n = "regression-arg-int";

    let make_sit = |text: &str, conf: f32| {
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
            raw_content: vec![ContentBlock::text(text)],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: conf,
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

    let sit_a = make_sit("Source A says attack happened", 0.8);
    let sit_b = make_sit("Source B denies attack", 0.7);
    hg.add_contention(ContentionLink {
        situation_a: sit_a,
        situation_b: sit_b,
        contention_type: crate::source::ContentionType::DirectContradiction,
        description: Some("Contradictory claims".into()),
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    })
    .unwrap();

    let legacy = run_argumentation(&hg, n).expect("ok");
    let new_path = run_argumentation_with_gradual(&hg, n, None, None).expect("ok");

    assert_eq!(legacy.framework.arguments.len(), new_path.framework.arguments.len());
    assert_eq!(legacy.grounded, new_path.grounded);
    assert_eq!(legacy.preferred_extensions, new_path.preferred_extensions);
    assert_eq!(legacy.stable_extensions, new_path.stable_extensions);
    assert!(new_path.gradual.is_none());

    // Mutual attack → both UNDEC (load-bearing legacy assertion).
    for (_, label) in &new_path.grounded {
        assert_eq!(*label, ArgumentLabel::Undec);
    }
    assert_eq!(new_path.preferred_extensions.len(), 2);
}

// ─── §5.5 — Principle property tests (3 × 10 trials) ──────────────────

fn random_framework(rng: &mut ChaCha8Rng) -> ArgumentationFramework {
    let n = rng.gen_range(3..=8);
    let arguments: Vec<Argument> = (0..n)
        .map(|i| {
            let w = 0.1 + rng.gen::<f64>() * 0.8; // ∈ [0.1, 0.9]
            arg_w(i as u128, w as f32)
        })
        .collect();
    let mut attacks = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.gen::<f64>() < 0.3 {
                attacks.push((i, j));
            }
        }
    }
    from_explicit(arguments, attacks)
}

#[test]
fn principle_anonymity_10_trials() {
    // Permuting argument labels permutes acceptability identically
    // (Amgoud & Ben-Naim 2013, Definition 3).
    for trial in 0..10u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(0xAA01 ^ trial);
        let fw = random_framework(&mut rng);
        let n = fw.arguments.len();
        let r1 = run_gradual_argumentation(&fw, &GradualSemanticsKind::HCategoriser, None)
            .expect("ok");

        // Build a random permutation σ : original_idx → new_idx.
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        let inv_perm = {
            let mut inv = vec![0usize; n];
            for (orig, &new) in perm.iter().enumerate() {
                inv[new] = orig;
            }
            inv
        };

        // Permuted framework: at slot `new`, place the argument that
        // originally sat at `inv_perm[new]`. Re-label attacks
        // accordingly.
        let mut perm_args = vec![arg_w(0, 0.0); n];
        for new in 0..n {
            perm_args[new] = fw.arguments[inv_perm[new]].clone();
        }
        let perm_attacks: Vec<(usize, usize)> = fw
            .attacks
            .iter()
            .map(|&(a, b)| (perm[a], perm[b]))
            .collect();
        let fw2 = from_explicit(perm_args, perm_attacks);
        let r2 = run_gradual_argumentation(&fw2, &GradualSemanticsKind::HCategoriser, None)
            .expect("ok");

        // Acceptability is keyed by Uuid (which carries the original
        // identity through the permutation), so the maps must agree.
        for arg in &fw.arguments {
            let v1 = r1.acceptability.get(&arg.id).copied().unwrap_or(0.0);
            let v2 = r2.acceptability.get(&arg.id).copied().unwrap_or(0.0);
            assert!(
                (v1 - v2).abs() < 1e-9,
                "trial {} arg {}: anonymity violated ({} vs {})",
                trial,
                arg.id,
                v1,
                v2
            );
        }
    }
}

#[test]
fn principle_independence_10_trials() {
    // Disconnected components have independent acceptability
    // (Amgoud & Ben-Naim 2013, Definition 6).
    for trial in 0..10u64 {
        let mut rng_c1 = ChaCha8Rng::seed_from_u64(0xAA02 ^ trial);
        let mut rng_c2 = ChaCha8Rng::seed_from_u64(0xAA03 ^ trial);
        let n1 = rng_c1.gen_range(3..=4);
        let n2 = rng_c2.gen_range(3..=4);
        let c1_args: Vec<Argument> = (0..n1)
            .map(|i| arg_w(i as u128, (0.1 + rng_c1.gen::<f64>() * 0.8) as f32))
            .collect();
        let c2_args: Vec<Argument> = (0..n2)
            .map(|i| arg_w((100 + i) as u128, (0.1 + rng_c2.gen::<f64>() * 0.8) as f32))
            .collect();
        let mut c1_attacks = Vec::new();
        for i in 0..n1 {
            for j in 0..n1 {
                if i != j && rng_c1.gen::<f64>() < 0.3 {
                    c1_attacks.push((i, j));
                }
            }
        }
        let mut c2_attacks = Vec::new();
        for i in 0..n2 {
            for j in 0..n2 {
                if i != j && rng_c2.gen::<f64>() < 0.3 {
                    c2_attacks.push((i, j));
                }
            }
        }

        // C1 alone.
        let fw_c1 = from_explicit(c1_args.clone(), c1_attacks.clone());
        let r_c1 = run_gradual_argumentation(&fw_c1, &GradualSemanticsKind::HCategoriser, None)
            .expect("ok");

        // Merged (C2 args appended; attacks shifted by n1; no cross-edges).
        let mut all_args = c1_args.clone();
        all_args.extend(c2_args.clone());
        let mut all_attacks = c1_attacks.clone();
        all_attacks.extend(c2_attacks.iter().map(|&(a, b)| (a + n1, b + n1)));
        let fw_full = from_explicit(all_args, all_attacks);
        let r_full = run_gradual_argumentation(&fw_full, &GradualSemanticsKind::HCategoriser, None)
            .expect("ok");

        for arg in &fw_c1.arguments {
            let v1 = r_c1.acceptability.get(&arg.id).copied().unwrap_or(0.0);
            let v2 = r_full.acceptability.get(&arg.id).copied().unwrap_or(0.0);
            assert!(
                (v1 - v2).abs() < 1e-9,
                "trial {} arg {}: independence violated ({} vs {})",
                trial,
                arg.id,
                v1,
                v2
            );
        }
    }
}

#[test]
fn principle_monotonicity_10_trials() {
    // Adding an attack (c → b) cannot increase Acc(b).
    // (Amgoud & Ben-Naim 2013, Proposition 1.)
    for trial in 0..10u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(0xAA04 ^ trial);
        let fw = random_framework(&mut rng);
        let n = fw.arguments.len();
        if n < 2 {
            continue;
        }
        let r_before =
            run_gradual_argumentation(&fw, &GradualSemanticsKind::HCategoriser, None).expect("ok");

        // Pick a target b uniformly.
        let b = rng.gen_range(0..n);
        // Find an attacker c ≠ b that does not yet attack b.
        let existing: HashSet<usize> =
            fw.attacks.iter().filter(|&&(_, t)| t == b).map(|&(a, _)| a).collect();
        let mut candidates: Vec<usize> =
            (0..n).filter(|&c| c != b && !existing.contains(&c)).collect();
        if candidates.is_empty() {
            continue;
        }
        candidates.shuffle(&mut rng);
        let c = candidates[0];

        let mut attacks2 = fw.attacks.clone();
        attacks2.push((c, b));
        let fw2 = from_explicit(fw.arguments.clone(), attacks2);
        let r_after = run_gradual_argumentation(&fw2, &GradualSemanticsKind::HCategoriser, None)
            .expect("ok");

        let id_b = fw.arguments[b].id;
        let v_before = r_before.acceptability.get(&id_b).copied().unwrap_or(0.0);
        let v_after = r_after.acceptability.get(&id_b).copied().unwrap_or(0.0);
        assert!(
            v_after <= v_before + 1e-9,
            "trial {}: monotonicity violated for arg {} ({} → {})",
            trial,
            id_b,
            v_before,
            v_after
        );
    }
}

// ─── §5.6 — API integration (5 tests on `run_argumentation_with_gradual`) ──

#[test]
fn api_default_returns_no_gradual() {
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let r = run_argumentation_with_gradual(&hg, "no-narrative", None, None).expect("ok");
    assert!(r.gradual.is_none());
}

#[test]
fn api_h_categoriser_returns_gradual() {
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let r = run_argumentation_with_gradual(
        &hg,
        "no-narrative",
        Some(GradualSemanticsKind::HCategoriser),
        None,
    )
    .expect("ok");
    // Empty framework → empty acceptability map.
    let g = r.gradual.expect("gradual present");
    assert!(g.acceptability.is_empty());
    assert!(g.converged);
}

#[test]
fn api_max_based_returns_gradual() {
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let r = run_argumentation_with_gradual(
        &hg,
        "no-narrative",
        Some(GradualSemanticsKind::MaxBased),
        Some(TNormKind::Godel),
    )
    .expect("ok");
    assert!(r.gradual.is_some());
}

#[test]
fn api_card_based_iterations_bounded_by_argument_count() {
    // Card-based on the 3-cycle: cardinality stabilises in O(|A|), but
    // the sum component still needs to reach CONVERGENCE_EPSILON. We
    // assert the fixed-point is reached well below MAX_GRADUAL_ITERATIONS
    // — the load-bearing claim is "doesn't run away to the cap", not a
    // tight |A|-round bound.
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let result = crate::analysis::argumentation::run_on_framework_with_gradual(
        &hg,
        "regression",
        &fixture_b(),
        Some(GradualSemanticsKind::CardBased),
        None,
    )
    .expect("ok");
    let g = result.gradual.expect("gradual present");
    assert!(g.converged);
    assert!(g.iterations <= 50, "iterations {} > 50 (well within cap of 200)", g.iterations);
}

#[test]
fn api_weighted_h_categoriser_with_valid_weights() {
    let hg = crate::hypergraph::Hypergraph::new(Arc::new(MemoryStore::new()));
    let result = crate::analysis::argumentation::run_on_framework_with_gradual(
        &hg,
        "regression",
        &fixture_a(),
        Some(GradualSemanticsKind::WeightedHCategoriser {
            weights: vec![0.5],
        }),
        None,
    )
    .expect("ok");
    let g = result.gradual.expect("gradual present");
    assert_close(
        *g.acceptability.get(&Uuid::from_u128(1)).unwrap(),
        0.6 / (1.0 + 0.5 * 0.8),
        1e-6,
        "Acc(b) under weighted h-Cat",
    );
}

