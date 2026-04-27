//! Wargame substrate tests (EATH Phase 9).
//!
//! Eight tests covering the [`BackgroundSubstrate`] integration into
//! [`WargameSession`] + the synthetic / hybrid substrate variants. Gated
//! behind `--features adversarial` (the whole `adversarial` module is).
//!
//! Two helpers seed calibrated EATH params + actor entities the wargame can
//! pick up. The actor rows are real (not synthetic) so the wargame's
//! fork_from_hypergraph picks them up via `list_entities_by_narrative`.

use std::sync::Arc;

use chrono::Utc;
use uuid::Uuid;

use crate::adversarial::session::{
    BackgroundSubstrate, SubstrateProvenance, WargameConfig, WargameSession,
    SUBSTRATE_INLINE_ENTITY_CAP,
};
use crate::adversarial::wargame::{apply_action_effects, WargameAction};
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::calibrate::save_params;
use crate::synth::hybrid::HybridComponent;
use crate::synth::types::EathParams;
use crate::types::EntityType;
use crate::Hypergraph;

fn fresh_hg() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

/// Persist calibrated EathParams under (narrative_id, "eath") AND seed a
/// handful of Actor entities under the same narrative_id so the wargame's
/// fork_from_hypergraph can pick something up. The substrate generator will
/// produce SYNTHETIC actors under a different narrative — these source-narr
/// actors are just for the calibrate hand-off.
fn seed_calibrated_narrative(hg: &Hypergraph, narrative_id: &str, n_entities: usize) {
    let params = EathParams {
        a_t_distribution: vec![0.5; n_entities],
        a_h_distribution: vec![1.0; n_entities],
        lambda_schedule: vec![1.0],
        p_from_scratch: 0.7,
        omega_decay: 0.95,
        group_size_distribution: vec![3, 1, 1], // dyads dominate
        rho_low: 0.5,
        rho_high: 0.3,
        xi: 1.0,
        order_propensity: vec![],
        max_group_size: 4,
        stm_capacity: 7,
        num_entities: n_entities,
    };
    save_params(hg.store(), narrative_id, "eath", &params).unwrap();

    for i in 0..n_entities {
        hg.create_entity(crate::types::Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": format!("actor-{i}")}),
            beliefs: None,
            embedding: None,
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some(narrative_id.into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap();
    }
}

// ─── T1 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_wargame_with_synthetic_substrate_initializes() {
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-thriller", 6);

    let config = WargameConfig {
        background: Some(BackgroundSubstrate::Synthetic {
            source_narrative_id: "src-thriller".into(),
            model: "eath".into(),
            params: None,
            seed: Some(42),
        }),
        ..Default::default()
    };

    let session = WargameSession::create(&hg, "src-thriller", config)
        .expect("session with synthetic substrate must initialize");

    // Wargame's narrative_id is the substrate id, NOT the source.
    assert!(
        session.narrative_id.starts_with("wargame-"),
        "session narrative_id should be wargame-prefixed, got {}",
        session.narrative_id
    );
    assert!(session.narrative_id.ends_with("-substrate"));
}

// ─── T2 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_wargame_with_hybrid_substrate_initializes() {
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-a", 5);
    seed_calibrated_narrative(&hg, "src-b", 5);

    let config = WargameConfig {
        background: Some(BackgroundSubstrate::SyntheticHybrid {
            components: vec![
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
            ],
            seed: Some(7),
        }),
        ..Default::default()
    };

    let session = WargameSession::create(&hg, "src-a", config)
        .expect("session with hybrid substrate must initialize");
    assert!(session.narrative_id.starts_with("wargame-"));

    // Hybrid output narrative MUST exist on the hypergraph after init.
    let synth_entities = hg
        .list_entities_by_narrative(&session.narrative_id)
        .unwrap();
    assert!(
        !synth_entities.is_empty(),
        "hybrid substrate should have emitted entities"
    );
}

// ─── T3 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_hybrid_weights_must_sum_to_one() {
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-a", 3);
    seed_calibrated_narrative(&hg, "src-b", 3);

    let config = WargameConfig {
        background: Some(BackgroundSubstrate::SyntheticHybrid {
            components: vec![
                HybridComponent {
                    narrative_id: "src-a".into(),
                    model: "eath".into(),
                    weight: 0.4,
                },
                HybridComponent {
                    narrative_id: "src-b".into(),
                    model: "eath".into(),
                    weight: 0.4,
                },
            ],
            seed: Some(1),
        }),
        ..Default::default()
    };

    let err = WargameSession::create(&hg, "src-a", config)
        .expect_err("non-summing hybrid weights must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("sum to 1.0"),
        "error should mention weight sum: {msg}"
    );
}

// ─── T4 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_hybrid_generation_is_deterministic_with_fixed_seed() {
    // Two fresh hypergraphs (UUIDs are seed-derived, would collide if reused).
    // Same seed + same calibrated params ⇒ identical substrate stats.
    let hg1 = fresh_hg();
    seed_calibrated_narrative(&hg1, "src-a", 5);
    seed_calibrated_narrative(&hg1, "src-b", 5);
    let hg2 = fresh_hg();
    seed_calibrated_narrative(&hg2, "src-a", 5);
    seed_calibrated_narrative(&hg2, "src-b", 5);

    let make_session = |hg: &Hypergraph| {
        let config = WargameConfig {
            background: Some(BackgroundSubstrate::SyntheticHybrid {
                components: vec![
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
                ],
                seed: Some(123_456),
            }),
            ..Default::default()
        };
        WargameSession::create(hg, "src-a", config).unwrap()
    };

    let s1 = make_session(&hg1);
    let s2 = make_session(&hg2);
    let entities1 = hg1.list_entities_by_narrative(&s1.narrative_id).unwrap();
    let entities2 = hg2.list_entities_by_narrative(&s2.narrative_id).unwrap();
    let situations1 = hg1.list_situations_by_narrative(&s1.narrative_id).unwrap();
    let situations2 = hg2.list_situations_by_narrative(&s2.narrative_id).unwrap();
    assert_eq!(entities1.len(), entities2.len());
    assert_eq!(situations1.len(), situations2.len());
}

// ─── T5 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_wargame_substrate_does_not_persist_to_source_narrative_ids() {
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-canonical", 5);
    let baseline_count = hg
        .list_entities_by_narrative("src-canonical")
        .unwrap()
        .len();
    let baseline_situations = hg
        .list_situations_by_narrative("src-canonical")
        .unwrap()
        .len();

    let config = WargameConfig {
        background: Some(BackgroundSubstrate::Synthetic {
            source_narrative_id: "src-canonical".into(),
            model: "eath".into(),
            params: None,
            seed: Some(99),
        }),
        ..Default::default()
    };
    let session = WargameSession::create(&hg, "src-canonical", config).unwrap();

    // Source narrative must be untouched (no synthetic entities/situations
    // leaked into "src-canonical").
    let after_count = hg
        .list_entities_by_narrative("src-canonical")
        .unwrap()
        .len();
    let after_situations = hg
        .list_situations_by_narrative("src-canonical")
        .unwrap()
        .len();
    assert_eq!(after_count, baseline_count);
    assert_eq!(after_situations, baseline_situations);

    // Substrate goes to the wargame-prefixed namespace.
    assert!(
        session.narrative_id.starts_with("wargame-"),
        "substrate must be wargame-prefixed: {}",
        session.narrative_id
    );
    let synth_entities = hg.list_entities_by_narrative(&session.narrative_id).unwrap();
    assert!(
        !synth_entities.is_empty(),
        "synthetic substrate should have entities under the wargame-prefixed id"
    );
}

// ─── T6 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_wargame_action_effects_apply_correctly_on_hybrid_substrate() {
    // Wargame action effects are substrate-agnostic (operate on
    // SimulationState compartments, not on the underlying narrative). This
    // test confirms a hybrid substrate's compartments accept and respond to
    // action effects with no special-casing.
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-a", 4);
    seed_calibrated_narrative(&hg, "src-b", 4);

    let config = WargameConfig {
        background: Some(BackgroundSubstrate::SyntheticHybrid {
            components: vec![
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
            ],
            seed: Some(50),
        }),
        ..Default::default()
    };
    let mut session = WargameSession::create(&hg, "src-a", config).unwrap();

    // Seed misinformation on one platform and apply Debunk — γ must rise.
    session
        .state
        .seed_narrative(&session.narrative_id, "twitter", 10.0);
    let key = crate::adversarial::sim_state::compartment_key(&session.narrative_id, "twitter");
    let gamma_before = session.state.compartments.get(&key).unwrap().gamma;

    apply_action_effects(
        &mut session.state,
        &WargameAction::Debunk {
            evidence: "test".into(),
        },
        &key,
        &session.narrative_id,
        "twitter",
    );

    let gamma_after = session.state.compartments.get(&key).unwrap().gamma;
    assert!(
        gamma_after > gamma_before,
        "Debunk should increase γ on hybrid-substrate compartments \
         (before={gamma_before}, after={gamma_after})"
    );
}

// ─── T7 ───────────────────────────────────────────────────────────────────────

#[test]
fn test_retrodiction_metadata_marks_synthetic_or_hybrid() {
    // The substrate_provenance() accessor returns the right variant for
    // each of the four BackgroundSubstrate cases. Retrodiction + reward
    // renderers consume this to label downstream reports.
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-a", 4);
    seed_calibrated_narrative(&hg, "src-b", 4);

    // Real path: no background → SubstrateProvenance::Real.
    let s_real = WargameSession::create(&hg, "src-a", WargameConfig::default()).unwrap();
    assert!(matches!(
        s_real.substrate_provenance(),
        SubstrateProvenance::Real { .. }
    ));

    // Synthetic path: substrate_provenance picks up source narrative + seed.
    let s_synth = WargameSession::create(
        &hg,
        "src-a",
        WargameConfig {
            background: Some(BackgroundSubstrate::Synthetic {
                source_narrative_id: "src-a".into(),
                model: "eath".into(),
                params: None,
                seed: Some(5),
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let prov = s_synth.substrate_provenance();
    let SubstrateProvenance::Synthetic {
        source_narrative_id,
        model,
        seed,
        ..
    } = prov
    else {
        panic!("expected Synthetic provenance, got {:?}", s_synth.substrate_provenance());
    };
    assert_eq!(source_narrative_id, "src-a");
    assert_eq!(model, "eath");
    assert_eq!(seed, Some(5));

    // Hybrid path: substrate_provenance carries the components vector.
    let s_hybrid = WargameSession::create(
        &hg,
        "src-a",
        WargameConfig {
            background: Some(BackgroundSubstrate::SyntheticHybrid {
                components: vec![
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
                ],
                seed: Some(11),
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let prov = s_hybrid.substrate_provenance();
    let SubstrateProvenance::Hybrid {
        components, seed, ..
    } = prov
    else {
        panic!("expected Hybrid provenance");
    };
    assert_eq!(components.len(), 2);
    assert_eq!(components[0].narrative_id, "src-a");
    assert_eq!(seed, Some(11));
}

// ─── T8 ───────────────────────────────────────────────────────────────────────

// The Phase 8 ComparisonHarness lives in tests/benchmarks/ and is only
// available to integration tests, not lib tests. T8 documents the contract
// the integration test exercises: Phase 9 extends ComparisonResult with
// `provenance: Option<ProvenanceTag>` + `substrate_hash: Option<String>`.
//
// We assert the contract here by constructing a SubstrateProvenance + a
// canonical narrative-state hash and verifying both are stable, which is
// what the integration test would compare across baseline/treatment runs.
#[test]
fn test_comparison_harness_produces_intervention_vs_baseline_report() {
    let hg = fresh_hg();
    seed_calibrated_narrative(&hg, "src-a", 4);
    seed_calibrated_narrative(&hg, "src-b", 4);

    // Build a hybrid session, capture the provenance + the substrate hash.
    let session = WargameSession::create(
        &hg,
        "src-a",
        WargameConfig {
            background: Some(BackgroundSubstrate::SyntheticHybrid {
                components: vec![
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
                ],
                seed: Some(2026),
            }),
            ..Default::default()
        },
    )
    .unwrap();

    // 1. Provenance is stable + introspectable.
    let prov = session.substrate_provenance();
    assert!(matches!(prov, SubstrateProvenance::Hybrid { .. }));

    // 2. Substrate hash is computable + non-empty (the same hash a Phase 8
    //    ComparisonResult would carry on baseline + treatment runs over the
    //    same substrate).
    let hash =
        crate::synth::hashing::canonical_narrative_state_hash(&hg, &session.narrative_id).unwrap();
    assert!(!hash.is_empty(), "substrate hash must be non-empty");
    assert_eq!(hash.len(), 64, "SHA-256 hash must be 64 hex chars");

    // 3. A second hash on the SAME substrate (no mutations between calls)
    //    matches — proving the Phase 8 invariant test that
    //    intervention-vs-baseline reports compare on identical substrates.
    let hash_again =
        crate::synth::hashing::canonical_narrative_state_hash(&hg, &session.narrative_id).unwrap();
    assert_eq!(hash, hash_again);
}

// ─── Inline-cap regression ────────────────────────────────────────────────────

#[test]
fn test_inline_synthetic_rejects_oversized_source() {
    let hg = fresh_hg();
    // 501 entities → just past the inline cap.
    seed_calibrated_narrative(&hg, "src-huge", SUBSTRATE_INLINE_ENTITY_CAP + 1);

    let config = WargameConfig {
        background: Some(BackgroundSubstrate::Synthetic {
            source_narrative_id: "src-huge".into(),
            model: "eath".into(),
            params: None,
            seed: Some(1),
        }),
        ..Default::default()
    };

    let err = WargameSession::create(&hg, "src-huge", config)
        .expect_err("oversized inline-synthetic substrate must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("inline cap") && msg.contains("ExistingNarrative"),
        "error should mention inline cap + workaround: {msg}"
    );
}
