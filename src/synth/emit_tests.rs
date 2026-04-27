//! Unit tests for `src/synth/emit.rs`.
//!
//! Extracted from the parent module so `emit.rs` itself stays under the
//! 500-line file-size cap (Phase 13b added the
//! `EmitContext::reuse_entities` field + a regression test for it).

use super::*;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use rand::SeedableRng;
use std::sync::Arc;

fn fresh_hg() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

fn ctx() -> EmitContext {
    EmitContext::new(
        Uuid::now_v7(),
        "synth-narr".to_string(),
        "synth-actor-".to_string(),
        Utc::now(),
        60,
        "eath".to_string(),
    )
}

#[test]
fn write_synthetic_entities_round_trips_with_provenance() {
    let hg = fresh_hg();
    let c = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let ids = write_synthetic_entities(&c, 3, &mut rng, &hg).unwrap();
    assert_eq!(ids.len(), 3);
    for (i, id) in ids.iter().enumerate() {
        let e = hg.get_entity(id).unwrap();
        assert!(is_synthetic_entity(&e));
        assert_eq!(entity_run_id(&e), Some(c.run_id));
        assert_eq!(
            e.properties.get("name").and_then(|v| v.as_str()),
            Some(format!("synth-actor-{i}").as_str())
        );
        assert!(matches!(
            e.extraction_method,
            Some(ExtractionMethod::Synthetic { .. })
        ));
        assert_eq!(e.confidence, DEFAULT_SYNTHETIC_CONFIDENCE);
        assert_eq!(e.maturity, MaturityLevel::Candidate);
        assert_eq!(e.narrative_id.as_deref(), Some("synth-narr"));
    }
}

#[test]
fn write_synthetic_situation_carries_run_id_and_step() {
    let hg = fresh_hg();
    let c = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let members = write_synthetic_entities(&c, 2, &mut rng, &hg).unwrap();
    let sid = write_synthetic_situation(&c, 5, &members, &mut rng, &hg).unwrap();
    let s = hg.get_situation(&sid).unwrap();
    assert!(is_synthetic_situation(&s));
    assert_eq!(situation_run_id(&s), Some(c.run_id));
    assert_eq!(s.properties.get("synth_step").and_then(|v| v.as_u64()), Some(5));
    assert!(matches!(s.temporal.granularity, TimeGranularity::Synthetic));
    assert_eq!(s.narrative_level, NarrativeLevel::Scene);
    let parts = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(parts.len(), 2);
    for p in parts {
        assert!(is_synthetic_participation(&p));
        assert_eq!(participation_run_id(&p), Some(c.run_id));
        assert_eq!(p.role, Role::Bystander);
    }
}

#[test]
fn filter_helpers_respect_include_flag() {
    let mut a = Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": "real"}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 1.0,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };
    let b = Entity {
        properties: serde_json::json!({"name": "synth", "synthetic": true}),
        ..a.clone()
    };
    a.id = Uuid::now_v7();
    let merged = vec![a.clone(), b.clone()];
    let real = filter_synthetic_entities(merged.clone(), false);
    assert_eq!(real.len(), 1);
    let all = filter_synthetic_entities(merged, true);
    assert_eq!(all.len(), 2);
}

/// Phase 13b T9: `EmitContext::reuse_entities` defaults to `None`
/// (preserving the EATH path) and, when set to `Some`, the field is
/// honoured by callers that consult it (verified via direct field check
/// — the writer helpers themselves don't branch on the flag, the
/// surrogate decides whether to call `write_synthetic_entities` or not).
#[test]
fn test_emit_context_reuse_entities_skips_minting() {
    let hg = fresh_hg();
    let c_default = ctx();
    // Default constructor preserves EATH path: reuse_entities is None.
    assert!(
        c_default.reuse_entities.is_none(),
        "EmitContext::new must default reuse_entities to None (EATH-path preservation)"
    );

    // EATH path: write_synthetic_entities still works against a default ctx.
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let ids = write_synthetic_entities(&c_default, 2, &mut rng, &hg).unwrap();
    assert_eq!(ids.len(), 2);
    // And the situation references those freshly-minted UUIDs.
    let sid = write_synthetic_situation(&c_default, 0, &ids, &mut rng, &hg).unwrap();
    let parts = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(parts.len(), 2);

    // NuDHy path: build a ctx with reuse_entities = Some(pre-existing UUIDs).
    // No fresh entity minting — write_synthetic_situation references the
    // supplied UUIDs directly. We pre-create two entities outside emit.rs
    // (simulating the source narrative) and verify the situation links to
    // them with no extra Entity rows created.
    let pre_existing: Vec<Uuid> = (0..2).map(|_| Uuid::now_v7()).collect();
    let now = Utc::now();
    for &eid in &pre_existing {
        hg.create_entity(Entity {
            id: eid,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "pre-existing"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::HumanEntered),
            narrative_id: Some("source-narr".into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap();
    }
    let entity_count_before_synth = hg
        .list_entities_by_narrative("synth-narr")
        .unwrap()
        .len();

    let c_reuse = EmitContext {
        run_id: Uuid::now_v7(),
        narrative_id: "synth-narr".to_string(),
        maturity: MaturityLevel::Candidate,
        confidence: DEFAULT_SYNTHETIC_CONFIDENCE,
        label_prefix: "synth-actor-".to_string(),
        time_anchor: now,
        step_duration_seconds: 60,
        model: "nudhy".to_string(),
        reuse_entities: Some(pre_existing.clone()),
    };
    assert_eq!(c_reuse.reuse_entities.as_ref().unwrap().len(), 2);

    // The NuDHy generation path NEVER calls write_synthetic_entities — it
    // calls write_synthetic_situation directly with pre-existing UUIDs.
    let sid2 = write_synthetic_situation(&c_reuse, 0, &pre_existing, &mut rng, &hg).unwrap();
    let parts2 = hg.get_participants_for_situation(&sid2).unwrap();
    assert_eq!(parts2.len(), 2);
    for p in &parts2 {
        assert!(
            pre_existing.contains(&p.entity_id),
            "participation must reference a pre-existing entity (no minting)"
        );
        assert!(
            is_synthetic_participation(p),
            "participation still carries the synthetic sentinel"
        );
    }
    // No new entities under the synth narrative — count is unchanged.
    let entity_count_after_synth = hg
        .list_entities_by_narrative("synth-narr")
        .unwrap()
        .len();
    assert_eq!(
        entity_count_before_synth, entity_count_after_synth,
        "NuDHy path must NOT mint new Entity records under the output narrative"
    );
}

#[test]
fn participation_sentinel_round_trips() {
    let hg = fresh_hg();
    let c = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let ids = write_synthetic_entities(&c, 1, &mut rng, &hg).unwrap();
    let sid = write_synthetic_situation(&c, 0, &ids, &mut rng, &hg).unwrap();
    let parts = hg.get_participants_for_situation(&sid).unwrap();
    assert!(parts.iter().all(is_synthetic_participation));
    assert!(parts.iter().all(|p| participation_run_id(p) == Some(c.run_id)));
}
