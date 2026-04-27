//! Phase 2 tests for `crate::synth::calibrate::*` — fitter on real
//! narratives, persistence round-trip, and a calibrate→generate consistency
//! check.
//!
//! All tests use `MemoryStore` and never depend on wall-clock outside of the
//! seed (which is acceptable — calibration is offline setup, not the
//! deterministic generation hot path).

use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use uuid::Uuid;

use super::*; // brings fit_params_from_narrative / save_params / load_params /
              // delete_params / EATH_MODEL_NAME / Hypergraph / TensaError into scope.
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::eath::EathSurrogate;
use crate::synth::surrogate::SurrogateModel;
use crate::synth::types::SurrogateParams;
use crate::types::{
    AllenInterval, ContentBlock, Entity, EntityType, ExtractionMethod, MaturityLevel,
    NarrativeLevel, Participation, Role, Situation, TimeGranularity,
};

// ── Fixtures ────────────────────────────────────────────────────────────────

fn fresh_hypergraph() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

fn make_entity(narrative_id: &str, label: &str) -> Entity {
    let now = Utc::now();
    Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": label}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 1.0,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some(narrative_id.into()),
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    }
}

fn make_situation(
    narrative_id: &str,
    granularity: TimeGranularity,
    start_offset_secs: i64,
) -> Situation {
    let now = Utc::now();
    let epoch = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let start = if matches!(granularity, TimeGranularity::Unknown) {
        None
    } else {
        Some(epoch + Duration::seconds(start_offset_secs))
    };
    Situation {
        id: Uuid::now_v7(),
        name: None,
        description: None,
        properties: serde_json::Value::Null,
        temporal: AllenInterval {
            start,
            end: start,
            granularity,
            relations: vec![],
            fuzzy_endpoints: None,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text("calibration fixture")],
        narrative_level: NarrativeLevel::Event,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 1.0,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(narrative_id.into()),
        source_chunk_id: None,
        source_span: None,
        synopsis: None,
        manuscript_order: None,
        parent_situation_id: None,
        label: None,
        status: None,
        keywords: vec![],
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    }
}

/// Build a 5-entity / 10-situation narrative with a deterministic
/// participation pattern. Used by the basic calibration tests.
fn build_basic_narrative(hg: &Hypergraph, narrative_id: &str) -> (Vec<Uuid>, Vec<Uuid>) {
    let mut entity_ids = Vec::new();
    for i in 0..5 {
        let e = make_entity(narrative_id, &format!("actor-{i}"));
        let id = e.id;
        hg.create_entity(e).unwrap();
        entity_ids.push(id);
    }
    let mut situation_ids = Vec::new();
    for t in 0..10 {
        let s = make_situation(narrative_id, TimeGranularity::Exact, t * 60);
        let id = s.id;
        hg.create_situation(s).unwrap();
        situation_ids.push(id);

        // Group composition: rotate through entity windows of size 3 so we
        // get varied group sizes and entities have differing aT.
        let group_size = 2 + (t % 3) as usize; // 2, 3, 4, 2, 3, 4, ...
        for k in 0..group_size {
            let eid = entity_ids[(t as usize + k) % entity_ids.len()];
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: id,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }
    (entity_ids, situation_ids)
}

// ── T1 ─────────────────────────────────────────────────────────────────────

/// All 7 Phase-1 extension fields plus the original 6 must populate within
/// expected ranges on a real (small) narrative.
#[test]
fn test_calibrate_returns_valid_params_on_real_narrative() {
    let hg = fresh_hypergraph();
    let nid = "calib-basic";
    let (entity_ids, situation_ids) = build_basic_narrative(&hg, nid);

    let params = fit_params_from_narrative(&hg, nid).unwrap();

    assert_eq!(params.a_t_distribution.len(), entity_ids.len());
    assert_eq!(params.a_h_distribution.len(), entity_ids.len());
    assert_eq!(params.num_entities, entity_ids.len());

    for (i, &v) in params.a_t_distribution.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "a_t[{i}] = {v} out of [0, 1]"
        );
    }
    for (i, &v) in params.a_h_distribution.iter().enumerate() {
        assert!(v >= 0.0_f32, "a_h[{i}] = {v} negative");
        assert!(f32::is_finite(v));
    }

    // lambda_schedule populated, all entries within clamp range.
    assert!(!params.lambda_schedule.is_empty());
    for &v in &params.lambda_schedule {
        assert!((0.01..=100.0).contains(&v), "lambda out of clamp: {v}");
    }

    assert!(
        (0.0..=1.0).contains(&params.p_from_scratch),
        "p_from_scratch out of [0, 1]: {}",
        params.p_from_scratch
    );

    // Group-size histogram sums to the number of situations with >= 2
    // participants. Our fixture builds exactly `situation_ids.len()` such
    // groups (smallest group_size is 2).
    let hist_sum: u32 = params.group_size_distribution.iter().sum();
    assert_eq!(hist_sum as usize, situation_ids.len());

    // 7 Phase-1 extension fields populated.
    assert!((0.01..=1.0).contains(&params.rho_low));
    assert!((0.01..=1.0).contains(&params.rho_high));
    assert!((0.1..=50.0).contains(&params.xi));
    assert!(params.order_propensity.is_empty()); // Phase 2 leaves empty
    assert!((2..=50).contains(&params.max_group_size));
    assert_eq!(params.stm_capacity, 7);
    assert_eq!(params.num_entities, entity_ids.len());
}

// ── T2 ─────────────────────────────────────────────────────────────────────

/// Calibrating an unknown narrative should not panic. The narrative-by-id
/// list returns an empty entity vec, which the calibrator surfaces as
/// `SynthFailure("0 entities; cannot calibrate")`.
#[test]
fn test_calibrate_errors_on_unknown_narrative() {
    let hg = fresh_hypergraph();
    let err = fit_params_from_narrative(&hg, "no-such-narrative").unwrap_err();
    match err {
        TensaError::SynthFailure(msg) => assert!(
            msg.contains("0 entities"),
            "unexpected SynthFailure message: {msg}"
        ),
        other => panic!("expected SynthFailure, got {other:?}"),
    }
}

// ── T3 ─────────────────────────────────────────────────────────────────────

/// Same outcome as T2 but with a narrative that "exists" (other narratives
/// in the same hypergraph) and just happens to have zero entities. The
/// per-narrative entity list returns empty either way.
#[test]
fn test_calibrate_errors_on_narrative_with_zero_entities() {
    let hg = fresh_hypergraph();
    // Plant unrelated entities under a different narrative_id.
    for i in 0..3 {
        hg.create_entity(make_entity("other-narrative", &format!("e{i}"))).unwrap();
    }
    let err = fit_params_from_narrative(&hg, "empty-narr").unwrap_err();
    assert!(
        matches!(err, TensaError::SynthFailure(_)),
        "expected SynthFailure on zero-entity narrative"
    );
}

// ── T4 ─────────────────────────────────────────────────────────────────────

/// fit → save → load → assert equality (full PartialEq via the derived impl).
/// delete should make a subsequent load return None.
#[test]
fn test_calibrate_persistence_roundtrip() {
    let hg = fresh_hypergraph();
    let nid = "calib-rt";
    build_basic_narrative(&hg, nid);

    let fitted = fit_params_from_narrative(&hg, nid).unwrap();
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());

    save_params(store.as_ref(), nid, EATH_MODEL_NAME, &fitted).unwrap();
    let loaded = load_params(store.as_ref(), nid, EATH_MODEL_NAME)
        .unwrap()
        .expect("params round-trip");
    assert_eq!(fitted, loaded);

    delete_params(store.as_ref(), nid, EATH_MODEL_NAME).unwrap();
    let after_delete =
        load_params(store.as_ref(), nid, EATH_MODEL_NAME).unwrap();
    assert!(after_delete.is_none(), "delete should remove the entry");
}

// ── T5 ─────────────────────────────────────────────────────────────────────

/// Calibrate from a narrative whose group sizes are concentrated around 3,
/// generate via the surrogate, and assert the synthetic distribution stays
/// close (mean within ±0.5, KS-style divergence < 0.20).
#[test]
fn test_calibrate_then_generate_produces_similar_size_distribution() {
    let hg = fresh_hypergraph();
    let nid = "calib-ks";

    // 8 entities, 30 situations all of size 3 ± 1 (mostly 3).
    let mut entity_ids = Vec::new();
    for i in 0..8 {
        let e = make_entity(nid, &format!("e{i}"));
        let id = e.id;
        hg.create_entity(e).unwrap();
        entity_ids.push(id);
    }
    let group_sizes = [3usize, 3, 3, 2, 3, 4, 3, 3, 2, 3];
    for t in 0..30 {
        let s = make_situation(nid, TimeGranularity::Exact, t * 60);
        let id = s.id;
        hg.create_situation(s).unwrap();
        let size = group_sizes[(t as usize) % group_sizes.len()];
        for k in 0..size {
            hg.add_participant(Participation {
                entity_id: entity_ids[(t as usize + k) % entity_ids.len()],
                situation_id: id,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }

    let params = fit_params_from_narrative(&hg, nid).unwrap();
    let source_mean = source_mean_group_size(&group_sizes);
    let source_dist = empirical_dist_from_sizes(&group_sizes, params.max_group_size);

    // Generate via the surrogate using a fixed seed.
    let target = fresh_hypergraph();
    let envelope = SurrogateParams {
        model: "eath".into(),
        params_json: serde_json::to_value(&params).unwrap(),
        seed: 12345,
        num_steps: 200,
        label_prefix: "synth".into(),
    };
    let summary = EathSurrogate
        .generate(&envelope, &target, "synth-out")
        .unwrap();
    assert!(summary.num_situations > 0, "no synthetic situations produced");

    // Walk synthetic situations to compute their group-size distribution.
    let synth_situations = target
        .list_situations_by_narrative("synth-out")
        .unwrap();
    let mut synth_sizes = Vec::with_capacity(synth_situations.len());
    for s in &synth_situations {
        let m = target.get_participants_for_situation(&s.id).unwrap();
        if m.len() >= 2 {
            synth_sizes.push(m.len());
        }
    }
    let synth_mean = synth_sizes.iter().copied().map(|n| n as f64).sum::<f64>()
        / synth_sizes.len() as f64;

    // T5 spec: |Δ_mean| < 0.5.
    assert!(
        (synth_mean - source_mean as f64).abs() < 0.5,
        "synth_mean = {synth_mean}, source_mean = {source_mean}"
    );

    // T5 spec: KS divergence < 0.20.
    let synth_dist = empirical_dist_from_sizes(&synth_sizes, params.max_group_size);
    let ks = ks_divergence(&source_dist, &synth_dist);
    assert!(
        ks < 0.20,
        "KS divergence too large: {ks}; source = {source_dist:?}, synth = {synth_dist:?}"
    );

    // Telemetry — the orchestrator's report can grep for this.
    eprintln!(
        "[T5] source_mean={source_mean:.3}, synth_mean={synth_mean:.3}, ks={ks:.4}",
    );
}

fn source_mean_group_size(sizes: &[usize]) -> f32 {
    sizes.iter().copied().map(|n| n as f32).sum::<f32>() / sizes.len() as f32
}

/// Empirical PMF over sizes 2..=max_group_size as Vec<f64> indexed by
/// `size - 2`. Empty input ⇒ uniform.
fn empirical_dist_from_sizes(sizes: &[usize], max_group_size: usize) -> Vec<f64> {
    let bins = max_group_size.saturating_sub(1).max(1);
    let mut counts = vec![0u64; bins];
    for &s in sizes {
        if s >= 2 {
            let bin = (s - 2).min(bins - 1);
            counts[bin] += 1;
        }
    }
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return vec![1.0 / bins as f64; bins];
    }
    counts.into_iter().map(|c| c as f64 / total as f64).collect()
}

/// KS-style divergence: max |CDF_a(k) - CDF_b(k)| over the shared support.
fn ks_divergence(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().max(b.len());
    let mut cdf_a = 0.0_f64;
    let mut cdf_b = 0.0_f64;
    let mut ks = 0.0_f64;
    for k in 0..len {
        cdf_a += a.get(k).copied().unwrap_or(0.0);
        cdf_b += b.get(k).copied().unwrap_or(0.0);
        ks = ks.max((cdf_a - cdf_b).abs());
    }
    ks
}

// ── T6 ─────────────────────────────────────────────────────────────────────

/// `TimeGranularity::Unknown` → `temporal.start` is None → calibration must
/// fall back to creation-order sorting and still produce a non-empty
/// `lambda_schedule`.
#[test]
fn test_calibrate_handles_unknown_temporal_granularity() {
    let hg = fresh_hypergraph();
    let nid = "calib-unknown";
    let mut entity_ids = Vec::new();
    for i in 0..3 {
        let e = make_entity(nid, &format!("e{i}"));
        let id = e.id;
        hg.create_entity(e).unwrap();
        entity_ids.push(id);
    }
    for t in 0..6 {
        let s = make_situation(nid, TimeGranularity::Unknown, t * 60);
        let id = s.id;
        hg.create_situation(s).unwrap();
        for k in 0..2 {
            hg.add_participant(Participation {
                entity_id: entity_ids[(t as usize + k) % entity_ids.len()],
                situation_id: id,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }
    let params = fit_params_from_narrative(&hg, nid).unwrap();
    assert!(!params.lambda_schedule.is_empty());
    assert_eq!(params.num_entities, 3);
}

// ── T7 ─────────────────────────────────────────────────────────────────────

/// A degenerate narrative with one entity and zero situations is allowed by
/// the n-entity check (n=1 ≥ 1) and must produce clean defaults instead of
/// returning NaN/inf in any field. Confirms the NaN guard is effective on
/// the divide-by-zero edge cases.
#[test]
fn test_calibrate_rejects_nan_after_fit() {
    let hg = fresh_hypergraph();
    let nid = "calib-degen";
    hg.create_entity(make_entity(nid, "lone-actor")).unwrap();

    // Zero situations → aT[0] = 0, aH[0] = 0, lambda empty, p_from_scratch=1.
    let params = fit_params_from_narrative(&hg, nid).unwrap();
    for v in params.a_t_distribution.iter().copied() {
        assert!(f32::is_finite(v), "aT contains non-finite: {v}");
    }
    for v in params.a_h_distribution.iter().copied() {
        assert!(f32::is_finite(v), "aH contains non-finite: {v}");
    }
    for v in params.lambda_schedule.iter().copied() {
        assert!(f32::is_finite(v), "lambda contains non-finite: {v}");
    }
    assert!(params.p_from_scratch.is_finite());
    assert!(params.rho_low.is_finite());
    assert!(params.rho_high.is_finite());
    assert!(params.xi.is_finite());
    assert!(params.omega_decay.is_finite());

    // With zero situations, `p_from_scratch` should default to 1.0 (no
    // continuation evidence).
    assert!((params.p_from_scratch - 1.0).abs() < 1e-6);
}

// ── T8 ─────────────────────────────────────────────────────────────────────
//
// `test_long_term_memory_from_observed_pairs` lives in `synth::memory` —
// see `src/synth/memory.rs`. Phase 2 owns the API but the test naturally
// belongs alongside the type it exercises.
