use super::*;
use crate::store::memory::MemoryStore;
use std::sync::Arc;

fn test_store() -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new())
}

fn test_entity(entity_type: EntityType) -> Entity {
    Entity {
        id: Uuid::now_v7(),
        entity_type,
        properties: serde_json::json!({"name": "Test"}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    }
}

#[test]
fn test_create_entity_actor() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = hg.create_entity(entity).unwrap();
    let retrieved = hg.get_entity(&id).unwrap();
    assert_eq!(retrieved.entity_type, EntityType::Actor);
}

#[test]
fn test_create_entity_location() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Location);
    let id = hg.create_entity(entity).unwrap();
    let retrieved = hg.get_entity(&id).unwrap();
    assert_eq!(retrieved.entity_type, EntityType::Location);
}

#[test]
fn test_create_entity_all_types() {
    let hg = Hypergraph::new(test_store());
    let types = vec![
        EntityType::Actor,
        EntityType::Location,
        EntityType::Artifact,
        EntityType::Concept,
        EntityType::Organization,
    ];
    for t in types {
        let entity = test_entity(t.clone());
        let id = hg.create_entity(entity).unwrap();
        let retrieved = hg.get_entity(&id).unwrap();
        assert_eq!(retrieved.entity_type, t);
    }
}

#[test]
fn test_get_entity_exists() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = entity.id;
    hg.create_entity(entity).unwrap();
    let retrieved = hg.get_entity(&id).unwrap();
    assert_eq!(retrieved.id, id);
    assert_eq!(retrieved.confidence, 0.9);
}

#[test]
fn test_get_entity_not_found() {
    let hg = Hypergraph::new(test_store());
    let id = Uuid::now_v7();
    let result = hg.get_entity(&id);
    assert!(matches!(result, Err(TensaError::EntityNotFound(_))));
}

#[test]
fn test_update_entity_properties() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = hg.create_entity(entity).unwrap();
    let updated = hg
        .update_entity(&id, |e| {
            e.properties = serde_json::json!({"name": "Updated"});
        })
        .unwrap();
    assert_eq!(updated.properties["name"], "Updated");
    // Verify persisted
    let retrieved = hg.get_entity(&id).unwrap();
    assert_eq!(retrieved.properties["name"], "Updated");
}

#[test]
fn test_update_entity_embedding() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = hg.create_entity(entity).unwrap();
    let updated = hg
        .update_entity(&id, |e| {
            e.embedding = Some(vec![0.1, 0.2, 0.3]);
        })
        .unwrap();
    assert_eq!(updated.embedding, Some(vec![0.1, 0.2, 0.3]));
}

#[test]
fn test_delete_entity() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = hg.create_entity(entity).unwrap();
    hg.delete_entity(&id).unwrap();
    assert!(matches!(
        hg.get_entity(&id),
        Err(TensaError::EntityNotFound(_))
    ));
}

#[test]
fn test_delete_entity_not_found() {
    let hg = Hypergraph::new(test_store());
    let id = Uuid::now_v7();
    assert!(matches!(
        hg.delete_entity(&id),
        Err(TensaError::EntityNotFound(_))
    ));
}

#[test]
fn test_list_entities_by_type() {
    let hg = Hypergraph::new(test_store());
    hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.create_entity(test_entity(EntityType::Location)).unwrap();

    let actors = hg.list_entities_by_type(&EntityType::Actor).unwrap();
    assert_eq!(actors.len(), 2);
    let locations = hg.list_entities_by_type(&EntityType::Location).unwrap();
    assert_eq!(locations.len(), 1);
}

#[test]
fn test_list_entities_by_maturity() {
    let hg = Hypergraph::new(test_store());
    let mut e1 = test_entity(EntityType::Actor);
    e1.maturity = MaturityLevel::Candidate;
    let mut e2 = test_entity(EntityType::Actor);
    e2.maturity = MaturityLevel::Validated;
    let mut e3 = test_entity(EntityType::Actor);
    e3.maturity = MaturityLevel::GroundTruth;

    hg.create_entity(e1).unwrap();
    hg.create_entity(e2).unwrap();
    hg.create_entity(e3).unwrap();

    let validated_plus = hg
        .list_entities_by_maturity(MaturityLevel::Validated)
        .unwrap();
    assert_eq!(validated_plus.len(), 2);
    let all = hg
        .list_entities_by_maturity(MaturityLevel::Candidate)
        .unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_entity_serialization_roundtrip() {
    let hg = Hypergraph::new(test_store());
    let mut entity = test_entity(EntityType::Actor);
    entity.properties = serde_json::json!({
        "name": "Raskolnikov",
        "age": 23,
        "tags": ["student", "murderer"]
    });
    entity.confidence = 0.75;
    let id = hg.create_entity(entity.clone()).unwrap();
    let retrieved = hg.get_entity(&id).unwrap();
    assert_eq!(retrieved.properties, entity.properties);
    assert_eq!(retrieved.confidence, 0.75);
}

#[test]
fn test_entity_with_beliefs() {
    let hg = Hypergraph::new(test_store());
    let mut entity = test_entity(EntityType::Actor);
    entity.beliefs = Some(serde_json::json!({
        "guilt": 0.8,
        "freedom": 0.3
    }));
    let id = hg.create_entity(entity).unwrap();
    let retrieved = hg.get_entity(&id).unwrap();
    assert!(retrieved.beliefs.is_some());
    assert_eq!(retrieved.beliefs.unwrap()["guilt"], 0.8);
}

#[test]
fn test_entity_with_provenance() {
    let hg = Hypergraph::new(test_store());
    let mut entity = test_entity(EntityType::Actor);
    entity.provenance = vec![SourceReference {
        source_type: "novel".to_string(),
        source_id: Some("crime-and-punishment".to_string()),
        description: Some("Chapter 1".to_string()),
        timestamp: Utc::now(),
        registered_source: None,
    }];
    let id = hg.create_entity(entity).unwrap();
    let retrieved = hg.get_entity(&id).unwrap();
    assert_eq!(retrieved.provenance.len(), 1);
    assert_eq!(retrieved.provenance[0].source_type, "novel");
}

#[test]
fn test_list_entities_empty() {
    let hg = Hypergraph::new(test_store());
    let actors = hg.list_entities_by_type(&EntityType::Actor).unwrap();
    assert!(actors.is_empty());
}

#[test]
fn test_update_entity_updates_timestamp() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let original_time = entity.updated_at;
    let id = hg.create_entity(entity).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let updated = hg
        .update_entity(&id, |e| {
            e.confidence = 0.5;
        })
        .unwrap();
    assert!(updated.updated_at > original_time);
}

// ─── Auto-Snapshot Tests ───────────────────────────────────

#[test]
fn test_update_entity_creates_snapshot() {
    let hg = Hypergraph::new(test_store());
    let mut entity = test_entity(EntityType::Actor);
    entity.properties = serde_json::json!({"name": "Original"});
    let id = hg.create_entity(entity).unwrap();

    hg.update_entity(&id, |e| {
        e.properties = serde_json::json!({"name": "Updated"});
    })
    .unwrap();

    let history = hg.get_state_history(&id).unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].properties["name"], "Original");
}

#[test]
fn test_snapshot_preserves_pre_update_properties() {
    let hg = Hypergraph::new(test_store());
    let mut entity = test_entity(EntityType::Actor);
    entity.properties = serde_json::json!({"score": 42, "level": "beginner"});
    entity.beliefs = Some(serde_json::json!({"trust": 0.7}));
    entity.embedding = Some(vec![1.0, 2.0, 3.0]);
    let id = hg.create_entity(entity).unwrap();

    hg.update_entity(&id, |e| {
        e.properties = serde_json::json!({"score": 99, "level": "expert"});
        e.beliefs = Some(serde_json::json!({"trust": 0.1}));
        e.embedding = Some(vec![4.0, 5.0, 6.0]);
    })
    .unwrap();

    let history = hg.get_state_history(&id).unwrap();
    assert_eq!(history.len(), 1);
    let snap = &history[0];
    // Snapshot should have the BEFORE state
    assert_eq!(snap.properties["score"], 42);
    assert_eq!(snap.properties["level"], "beginner");
    assert_eq!(snap.beliefs.as_ref().unwrap()["trust"], 0.7);
    assert_eq!(snap.embedding.as_ref().unwrap(), &vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_snapshot_situation_id_is_v7() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = hg.create_entity(entity).unwrap();

    hg.update_entity(&id, |e| {
        e.confidence = 0.5;
    })
    .unwrap();

    let history = hg.get_state_history(&id).unwrap();
    assert_eq!(history.len(), 1);
    let snap_sit_id = history[0].situation_id;
    // UUID v7 has version nibble = 7 (bits 48-51)
    assert_eq!(snap_sit_id.get_version_num(), 7);
}

#[test]
fn test_no_snapshot_variant_skips() {
    let hg = Hypergraph::new(test_store());
    let entity = test_entity(EntityType::Actor);
    let id = hg.create_entity(entity).unwrap();

    hg.update_entity_no_snapshot(&id, |e| {
        e.confidence = 0.5;
    })
    .unwrap();

    let history = hg.get_state_history(&id).unwrap();
    assert!(history.is_empty());
}

#[test]
fn test_multiple_updates_create_history() {
    let hg = Hypergraph::new(test_store());
    let mut entity = test_entity(EntityType::Actor);
    entity.properties = serde_json::json!({"step": 0});
    let id = hg.create_entity(entity).unwrap();

    hg.update_entity(&id, |e| {
        e.properties = serde_json::json!({"step": 1});
    })
    .unwrap();
    hg.update_entity(&id, |e| {
        e.properties = serde_json::json!({"step": 2});
    })
    .unwrap();
    hg.update_entity(&id, |e| {
        e.properties = serde_json::json!({"step": 3});
    })
    .unwrap();

    let history = hg.get_state_history(&id).unwrap();
    assert_eq!(history.len(), 3);
    // Snapshots are ordered chronologically (v7 UUIDs sort by time)
    assert_eq!(history[0].properties["step"], 0);
    assert_eq!(history[1].properties["step"], 1);
    assert_eq!(history[2].properties["step"], 2);
}

// ─── Merge Tests ────────────────────────────────────────────

fn create_test_situation(hg: &Hypergraph) -> Uuid {
    let sit = Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: None,
            end: None,
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
        raw_content: vec![ContentBlock::text("Test situation")],
        narrative_level: NarrativeLevel::Scene,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.8,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: None,
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
}

fn test_participation(entity_id: Uuid, situation_id: Uuid, role: Role) -> Participation {
    Participation {
        entity_id,
        situation_id,
        role,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    }
}

#[test]
fn test_merge_reassigns_participations() {
    let hg = Hypergraph::new(test_store());
    let mut e1 = test_entity(EntityType::Actor);
    e1.properties = serde_json::json!({"name": "Alice"});
    let id1 = hg.create_entity(e1).unwrap();
    let mut e2 = test_entity(EntityType::Actor);
    e2.properties = serde_json::json!({"name": "Alicia"});
    let id2 = hg.create_entity(e2).unwrap();

    let sid = create_test_situation(&hg);
    hg.add_participant(test_participation(id2, sid, Role::Witness))
        .unwrap();

    hg.merge_entities(&id1, &id2).unwrap();

    // id2 participation should now be on id1
    let parts = hg.get_situations_for_entity(&id1).unwrap();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].situation_id, sid);

    // id2 should be gone
    assert!(hg.get_entity(&id2).is_err());
}

#[test]
fn test_merge_reassigns_state_versions() {
    let hg = Hypergraph::new(test_store());
    let id1 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let id2 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let sid = create_test_situation(&hg);

    hg.create_state_version(StateVersion {
        entity_id: id2,
        situation_id: sid,
        properties: serde_json::json!({"hp": 100}),
        beliefs: None,
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();

    hg.merge_entities(&id1, &id2).unwrap();

    // State version should now be under id1
    let sv = hg.get_state_at_situation(&id1, &sid).unwrap();
    assert_eq!(sv.properties["hp"], 100);
}

#[test]
fn test_merge_deletes_absorbed_entity() {
    let hg = Hypergraph::new(test_store());
    let id1 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let id2 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();

    hg.merge_entities(&id1, &id2).unwrap();
    assert!(hg.get_entity(&id2).is_err());
}

#[test]
fn test_merge_preserves_keep_properties() {
    let hg = Hypergraph::new(test_store());
    let mut e1 = test_entity(EntityType::Actor);
    e1.properties = serde_json::json!({"name": "Alice", "role": "hero"});
    let id1 = hg.create_entity(e1).unwrap();
    let mut e2 = test_entity(EntityType::Actor);
    e2.properties = serde_json::json!({"name": "Alicia", "age": 25});
    let id2 = hg.create_entity(e2).unwrap();

    let merged = hg.merge_entities(&id1, &id2).unwrap();
    // Keep's "name" should win, absorbed's "age" should be added
    assert_eq!(merged.properties["name"], "Alice");
    assert_eq!(merged.properties["role"], "hero");
    assert_eq!(merged.properties["age"], 25);
}

#[test]
fn test_merge_combines_provenance() {
    let hg = Hypergraph::new(test_store());
    let mut e1 = test_entity(EntityType::Actor);
    e1.provenance = vec![SourceReference {
        source_type: "manual".into(),
        source_id: None,
        description: Some("Source A".into()),
        timestamp: Utc::now(),
        registered_source: None,
    }];
    let id1 = hg.create_entity(e1).unwrap();
    let mut e2 = test_entity(EntityType::Actor);
    e2.provenance = vec![SourceReference {
        source_type: "llm".into(),
        source_id: None,
        description: Some("Source B".into()),
        timestamp: Utc::now(),
        registered_source: None,
    }];
    let id2 = hg.create_entity(e2).unwrap();

    let merged = hg.merge_entities(&id1, &id2).unwrap();
    assert_eq!(merged.provenance.len(), 2);
}

#[test]
fn test_merge_takes_higher_confidence() {
    let hg = Hypergraph::new(test_store());
    let mut e1 = test_entity(EntityType::Actor);
    e1.confidence = 0.5;
    let id1 = hg.create_entity(e1).unwrap();
    let mut e2 = test_entity(EntityType::Actor);
    e2.confidence = 0.9;
    let id2 = hg.create_entity(e2).unwrap();

    let merged = hg.merge_entities(&id1, &id2).unwrap();
    assert!((merged.confidence - 0.9).abs() < 0.01);
}

#[test]
fn test_merge_handles_duplicate_participation() {
    let hg = Hypergraph::new(test_store());
    let id1 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let id2 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let sid = create_test_situation(&hg);

    // Both participate in the same situation
    let mut p1 = test_participation(id1, sid, Role::Protagonist);
    p1.action = Some("leads".into());
    hg.add_participant(p1).unwrap();
    hg.add_participant(test_participation(id2, sid, Role::Witness))
        .unwrap();

    hg.merge_entities(&id1, &id2).unwrap();

    let parts = hg.get_situations_for_entity(&id1).unwrap();
    // keep's original + absorbed's re-pointed
    assert!(parts.len() >= 1);
}

#[test]
fn test_merge_nonexistent_entity_errors() {
    let hg = Hypergraph::new(test_store());
    let id1 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let fake_id = Uuid::now_v7();

    assert!(hg.merge_entities(&id1, &fake_id).is_err());
    assert!(hg.merge_entities(&fake_id, &id1).is_err());
}

#[test]
fn test_merge_same_entity_errors() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    assert!(hg.merge_entities(&id, &id).is_err());
}

// ─── Split Tests ────────────────────────────────────────────

#[test]
fn test_split_creates_new_entity() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let sid = create_test_situation(&hg);
    hg.add_participant(test_participation(id, sid, Role::Protagonist))
        .unwrap();

    let (source, clone) = hg.split_entity(&id, vec![sid]).unwrap();
    assert_ne!(source.id, clone.id);
    assert_eq!(clone.entity_type, source.entity_type);
}

#[test]
fn test_split_moves_specified_participations() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let sid1 = create_test_situation(&hg);
    let sid2 = create_test_situation(&hg);

    for sid in [sid1, sid2] {
        hg.add_participant(test_participation(id, sid, Role::Protagonist))
            .unwrap();
    }

    let (source, clone) = hg.split_entity(&id, vec![sid2]).unwrap();

    // Source keeps sid1, clone gets sid2
    let source_parts = hg.get_situations_for_entity(&source.id).unwrap();
    let clone_parts = hg.get_situations_for_entity(&clone.id).unwrap();
    assert_eq!(source_parts.len(), 1);
    assert_eq!(source_parts[0].situation_id, sid1);
    assert_eq!(clone_parts.len(), 1);
    assert_eq!(clone_parts[0].situation_id, sid2);
}

#[test]
fn test_split_moves_state_versions() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let sid1 = create_test_situation(&hg);
    let sid2 = create_test_situation(&hg);

    for (sid, hp) in [(sid1, 100), (sid2, 50)] {
        hg.add_participant(test_participation(id, sid, Role::Protagonist))
            .unwrap();
        hg.create_state_version(StateVersion {
            entity_id: id,
            situation_id: sid,
            properties: serde_json::json!({"hp": hp}),
            beliefs: None,
            embedding: None,
            timestamp: Utc::now(),
        })
        .unwrap();
    }

    let (_source, clone) = hg.split_entity(&id, vec![sid2]).unwrap();

    // Source keeps sid1 state, clone gets sid2 state
    let sv1 = hg.get_state_at_situation(&id, &sid1).unwrap();
    assert_eq!(sv1.properties["hp"], 100);
    let sv2 = hg.get_state_at_situation(&clone.id, &sid2).unwrap();
    assert_eq!(sv2.properties["hp"], 50);
}

#[test]
fn test_split_preserves_remaining() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let sid1 = create_test_situation(&hg);
    let sid2 = create_test_situation(&hg);
    let sid3 = create_test_situation(&hg);

    for sid in [sid1, sid2, sid3] {
        hg.add_participant(test_participation(id, sid, Role::Protagonist))
            .unwrap();
    }

    let (source, _clone) = hg.split_entity(&id, vec![sid3]).unwrap();
    let remaining = hg.get_situations_for_entity(&source.id).unwrap();
    assert_eq!(remaining.len(), 2); // sid1 and sid2 remain
}

#[test]
fn test_split_empty_situations_errors() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    assert!(hg.split_entity(&id, vec![]).is_err());
}

// ─── Soft Delete Tests ─────────────────────────────────────

#[test]
fn test_soft_delete_entity_hides_from_get() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.delete_entity(&id).unwrap();
    assert!(matches!(
        hg.get_entity(&id),
        Err(TensaError::EntityNotFound(_))
    ));
}

#[test]
fn test_soft_delete_entity_visible_with_include_deleted() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.delete_entity(&id).unwrap();
    let entity = hg.get_entity_include_deleted(&id).unwrap();
    assert!(entity.deleted_at.is_some());
}

#[test]
fn test_restore_entity() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.delete_entity(&id).unwrap();
    let restored = hg.restore_entity(&id).unwrap();
    assert!(restored.deleted_at.is_none());
    // Should be visible via get_entity again
    assert!(hg.get_entity(&id).is_ok());
}

#[test]
fn test_restore_non_deleted_entity_errors() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    assert!(hg.restore_entity(&id).is_err());
}

#[test]
fn test_soft_delete_excludes_from_list_by_type() {
    let hg = Hypergraph::new(test_store());
    let id1 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    let _id2 = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.delete_entity(&id1).unwrap();
    let actors = hg.list_entities_by_type(&EntityType::Actor).unwrap();
    assert_eq!(actors.len(), 1);
}

#[test]
fn test_soft_delete_excludes_from_list_by_maturity() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.delete_entity(&id).unwrap();
    let all = hg
        .list_entities_by_maturity(MaturityLevel::Candidate)
        .unwrap();
    assert_eq!(all.len(), 1);
}

#[test]
fn test_hard_delete_entity() {
    let hg = Hypergraph::new(test_store());
    let id = hg.create_entity(test_entity(EntityType::Actor)).unwrap();
    hg.delete_entity(&id).unwrap();
    hg.hard_delete_entity(&id).unwrap();
    assert!(hg.get_entity_include_deleted(&id).is_err());
}

#[test]
fn test_backward_compat_no_deleted_at_field() {
    // Simulate an entity serialized before deleted_at was added
    let entity_json = serde_json::json!({
        "id": Uuid::now_v7(),
        "entity_type": "Actor",
        "properties": {"name": "Old"},
        "beliefs": null,
        "embedding": null,
        "maturity": "Candidate",
        "confidence": 0.5,
        "provenance": [],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z"
    });
    let entity: Entity = serde_json::from_value(entity_json).unwrap();
    assert!(entity.deleted_at.is_none());
}

#[test]
fn test_find_entity_by_name_matches_name_and_folds_diacritics() {
    let hg = Hypergraph::new(test_store());
    let mut e = test_entity(EntityType::Actor);
    e.properties = serde_json::json!({"name": "Drago Milošević"});
    e.narrative_id = Some("nf".into());
    let id = hg.create_entity(e).unwrap();

    // Exact name
    let hit = hg.find_entity_by_name("nf", "Drago Milošević").unwrap();
    assert_eq!(hit.map(|e| e.id), Some(id));

    // ASCII-folded (no diacritic)
    let hit = hg.find_entity_by_name("nf", "drago milosevic").unwrap();
    assert_eq!(hit.map(|e| e.id), Some(id));

    // Hyphen/space equivalence
    let hit = hg.find_entity_by_name("nf", "drago-milosevic").unwrap();
    assert_eq!(hit.map(|e| e.id), Some(id));

    // Wrong narrative → no match
    let hit = hg.find_entity_by_name("other", "Drago Milošević").unwrap();
    assert!(hit.is_none());
}

#[test]
fn test_find_entity_by_name_matches_alias() {
    let hg = Hypergraph::new(test_store());
    let mut e = test_entity(EntityType::Actor);
    e.properties = serde_json::json!({
        "name": "Nadia Osman",
        "aliases": ["Nightingale", "Source WREN"]
    });
    e.narrative_id = Some("nf".into());
    let id = hg.create_entity(e).unwrap();

    assert_eq!(
        hg.find_entity_by_name("nf", "nightingale")
            .unwrap()
            .map(|e| e.id),
        Some(id)
    );
    assert_eq!(
        hg.find_entity_by_name("nf", "source wren")
            .unwrap()
            .map(|e| e.id),
        Some(id)
    );
}
