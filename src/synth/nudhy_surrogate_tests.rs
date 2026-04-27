//! NuDHy surrogate-impl unit tests (T6, T8).
//!
//! T7 (registry lookup) lives alongside the registry test in
//! `src/synth/registry.rs`.

use super::NudhySurrogate;
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::emit::{is_synthetic_participation, is_synthetic_situation};
use crate::synth::surrogate::SurrogateModel;
use crate::synth::types::SurrogateParams;
use crate::types::*;
use chrono::Utc;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use uuid::Uuid;

const TEST_NARRATIVE: &str = "nudhy-test-narrative";
const SYNTH_NARRATIVE: &str = "nudhy-synth-out";

fn fresh_hg() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

/// Seed a deterministic `n_entities × n_situations × group_size` narrative
/// of REAL records (no synthetic flag). Returns the entity UUIDs in
/// creation order.
fn seed_narrative(
    hg: &Hypergraph,
    narrative_id: &str,
    n_entities: usize,
    n_situations: usize,
    group_size: usize,
) -> Vec<Uuid> {
    let now = Utc::now();
    let mut entity_ids = Vec::with_capacity(n_entities);
    for i in 0..n_entities {
        let e = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": format!("real-actor-{i}")}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::HumanEntered),
            narrative_id: Some(narrative_id.into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        entity_ids.push(hg.create_entity(e).unwrap());
    }
    for s_idx in 0..n_situations {
        let s = Situation {
            id: Uuid::now_v7(),
            name: Some(format!("Real meeting {s_idx}")),
            description: None,
            properties: serde_json::Value::Null,
            temporal: AllenInterval {
                start: Some(now),
                end: Some(now + chrono::Duration::seconds(60)),
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
            raw_content: vec![ContentBlock::text(&format!("real event {s_idx}"))],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
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
        };
        let sid = hg.create_situation(s).unwrap();
        for k in 0..group_size {
            let eid = entity_ids[(s_idx + k) % n_entities];
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: sid,
                role: Role::Witness,
                info_set: Some(InfoSet {
                    knows_before: vec![],
                    learns: vec![],
                    reveals: vec![],
                    beliefs_about_others: vec![],
                }),
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }
    entity_ids
}

// ── T6: calibrate → generate roundtrip ───────────────────────────────────────

#[test]
fn test_nudhy_calibrate_generate_roundtrip_produces_valid_records() {
    let hg = fresh_hg();
    let source_entities = seed_narrative(&hg, TEST_NARRATIVE, 8, 20, 4);

    // Calibrate.
    let m = NudhySurrogate;
    let params_json = m.calibrate(&hg, TEST_NARRATIVE).unwrap();

    // Snapshot original edge sizes.
    let source_situations = hg.list_situations_by_narrative(TEST_NARRATIVE).unwrap();
    let mut source_sizes: Vec<usize> = Vec::new();
    for s in &source_situations {
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        source_sizes.push(parts.len());
    }
    source_sizes.sort_unstable();

    // Generate.
    let surr_params = SurrogateParams {
        model: "nudhy".into(),
        params_json,
        seed: 0xC0DE_F00D,
        num_steps: 0, // unused by NuDHy (steps come from NudhyParams.burn_in_steps)
        label_prefix: "nudhy-actor".into(),
    };
    let summary = m.generate(&surr_params, &hg, SYNTH_NARRATIVE).unwrap();
    assert_eq!(summary.model, "nudhy");
    assert_eq!(summary.output_narrative_id, SYNTH_NARRATIVE);
    assert_eq!(
        summary.num_situations,
        source_situations.len(),
        "NuDHy must emit one situation per source situation (chain + fixed)"
    );
    // num_entities is 0 — NuDHy reuses source entities, mints none.
    assert_eq!(summary.num_entities, 0);

    // Edge-size sequence preservation.
    let synth_situations = hg.list_situations_by_narrative(SYNTH_NARRATIVE).unwrap();
    assert_eq!(synth_situations.len(), source_situations.len());
    let mut synth_sizes: Vec<usize> = Vec::new();
    for s in &synth_situations {
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        synth_sizes.push(parts.len());
    }
    synth_sizes.sort_unstable();
    assert_eq!(
        source_sizes, synth_sizes,
        "edge-size multiset must be preserved end-to-end"
    );

    // Synthetic flag on every emitted situation.
    for s in &synth_situations {
        assert!(
            is_synthetic_situation(s),
            "every emitted situation must carry synthetic=true"
        );
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        for p in parts {
            assert!(
                is_synthetic_participation(&p),
                "every emitted participation must carry the synthetic sentinel"
            );
            // Sentinel must mention model=nudhy.
            let info = p.info_set.as_ref().unwrap();
            let fact = info
                .knows_before
                .iter()
                .find(|f| f.fact.starts_with("synthetic|run_id="))
                .expect("synthetic sentinel fact present");
            assert!(
                fact.fact.contains("model=nudhy"),
                "sentinel must record model=nudhy, got: {}",
                fact.fact
            );
        }
    }

    // No new Entity records under SYNTH_NARRATIVE — every participation
    // entity_id MUST point at a source-narrative entity.
    let source_set: HashSet<Uuid> = source_entities.into_iter().collect();
    let synth_entities = hg.list_entities_by_narrative(SYNTH_NARRATIVE).unwrap();
    assert!(
        synth_entities.is_empty(),
        "NuDHy must NOT mint entities under the output narrative; got {} entities",
        synth_entities.len()
    );
    for s in &synth_situations {
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        for p in parts {
            assert!(
                source_set.contains(&p.entity_id),
                "participation entity_id {} not in source entity set — entity reuse broken",
                p.entity_id
            );
        }
    }
}

// ── T8: determinism — same seed ⇒ same output ────────────────────────────────

#[test]
fn test_nudhy_determinism_same_seed_same_output() {
    // Build identical source narratives in two fresh hypergraphs so the
    // generated outputs can be compared without prior-run interference.
    fn make_seeded() -> (Hypergraph, serde_json::Value) {
        let hg = fresh_hg();
        seed_narrative(&hg, TEST_NARRATIVE, 6, 12, 3);
        let m = NudhySurrogate;
        let params_json = m.calibrate(&hg, TEST_NARRATIVE).unwrap();
        (hg, params_json)
    }

    let (hg_a, params_a) = make_seeded();
    let (hg_b, params_b) = make_seeded();

    let m = NudhySurrogate;
    let surr_a = SurrogateParams {
        model: "nudhy".into(),
        params_json: params_a,
        seed: 0x123_4567,
        num_steps: 0,
        label_prefix: "nudhy-actor".into(),
    };
    let surr_b = SurrogateParams {
        model: "nudhy".into(),
        params_json: params_b,
        seed: 0x123_4567,
        num_steps: 0,
        label_prefix: "nudhy-actor".into(),
    };

    let summary_a = m.generate(&surr_a, &hg_a, SYNTH_NARRATIVE).unwrap();
    let summary_b = m.generate(&surr_b, &hg_b, SYNTH_NARRATIVE).unwrap();

    // Same seed → same run_id by construction (run_id_from_seed is pure).
    assert_eq!(
        summary_a.run_id, summary_b.run_id,
        "same seed must produce same run_id"
    );

    // Collect (synth_step → sorted participant UUID set) tuples for each output.
    fn collect_step_signatures(hg: &Hypergraph) -> Vec<(u64, BTreeSet<Uuid>)> {
        let situations = hg.list_situations_by_narrative(SYNTH_NARRATIVE).unwrap();
        let mut sigs = Vec::with_capacity(situations.len());
        for s in situations {
            let step = s
                .properties
                .get("synth_step")
                .and_then(|v| v.as_u64())
                .expect("synth_step missing");
            let parts = hg.get_participants_for_situation(&s.id).unwrap();
            let members: BTreeSet<Uuid> = parts.iter().map(|p| p.entity_id).collect();
            sigs.push((step, members));
        }
        sigs.sort_by_key(|(step, _)| *step);
        sigs
    }
    let sigs_a = collect_step_signatures(&hg_a);
    let sigs_b = collect_step_signatures(&hg_b);

    // Note: source entity UUIDs differ across runs (Uuid::now_v7 in seed),
    // so we compare SHAPE (per-step membership set sizes + step count)
    // rather than identity. The MCMC kernel itself is verified deterministic
    // by `test_nudhy_state_seed_replay_is_bit_identical` in nudhy_tests.rs;
    // here we verify the generate path is deterministic relative to its
    // own input (params_json).
    assert_eq!(sigs_a.len(), sigs_b.len(), "same step count expected");
    let sizes_a: Vec<usize> = sigs_a.iter().map(|(_, m)| m.len()).collect();
    let sizes_b: Vec<usize> = sigs_b.iter().map(|(_, m)| m.len()).collect();
    assert_eq!(sizes_a, sizes_b, "same per-step membership sizes expected");
}

// ── Bonus: same hypergraph, two generates with same seed produce same trace ──

#[test]
fn test_nudhy_two_generates_on_same_hg_same_seed_same_membership_per_step() {
    let hg = fresh_hg();
    seed_narrative(&hg, TEST_NARRATIVE, 6, 12, 3);
    let m = NudhySurrogate;
    let params_json = m.calibrate(&hg, TEST_NARRATIVE).unwrap();

    let surr = SurrogateParams {
        model: "nudhy".into(),
        params_json,
        seed: 0xAABB_CCDD,
        num_steps: 0,
        label_prefix: "nudhy-actor".into(),
    };

    // First run.
    m.generate(&surr, &hg, "nudhy-out-1").unwrap();
    // Second run into a different output narrative.
    m.generate(&surr, &hg, "nudhy-out-2").unwrap();

    fn collect(hg: &Hypergraph, nid: &str) -> Vec<(u64, BTreeSet<Uuid>)> {
        let situations = hg.list_situations_by_narrative(nid).unwrap();
        let mut out = Vec::with_capacity(situations.len());
        for s in situations {
            let step = s
                .properties
                .get("synth_step")
                .and_then(|v| v.as_u64())
                .unwrap();
            let parts = hg.get_participants_for_situation(&s.id).unwrap();
            let members: BTreeSet<Uuid> = parts.iter().map(|p| p.entity_id).collect();
            out.push((step, members));
        }
        out.sort_by_key(|(s, _)| *s);
        out
    }
    let a = collect(&hg, "nudhy-out-1");
    let b = collect(&hg, "nudhy-out-2");
    assert_eq!(
        a, b,
        "two generates with the same seed against the same source hypergraph \
         must produce identical (step, members) signatures"
    );
}
