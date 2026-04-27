use super::*;
use crate::store::memory::MemoryStore;
use chrono::Utc;
use std::sync::Arc;

fn test_store() -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new())
}

fn setup_entity(hg: &Hypergraph, entity_type: EntityType) -> Uuid {
    let entity = Entity {
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
    };
    hg.create_entity(entity).unwrap()
}

fn setup_situation(hg: &Hypergraph) -> Uuid {
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
        raw_content: vec![ContentBlock::text("Test")],
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

fn make_participation(eid: Uuid, sid: Uuid, role: Role) -> Participation {
    Participation {
        entity_id: eid,
        situation_id: sid,
        role,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    }
}

#[test]
fn test_add_participant() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(participants.len(), 1);
    assert_eq!(participants[0].entity_id, eid);
    assert_eq!(participants[0].seq, 0);
}

#[test]
fn test_add_multiple_participants() {
    let hg = Hypergraph::new(test_store());
    let e1 = setup_entity(&hg, EntityType::Actor);
    let e2 = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(e1, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(e2, sid, Role::Antagonist))
        .unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(participants.len(), 2);
}

#[test]
fn test_get_participants_for_situation() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let s1 = setup_situation(&hg);
    let s2 = setup_situation(&hg);

    hg.add_participant(make_participation(eid, s1, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, s2, Role::Witness))
        .unwrap();

    let p1 = hg.get_participants_for_situation(&s1).unwrap();
    assert_eq!(p1.len(), 1);
    let p2 = hg.get_participants_for_situation(&s2).unwrap();
    assert_eq!(p2.len(), 1);
}

#[test]
fn test_get_situations_for_entity() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let s1 = setup_situation(&hg);
    let s2 = setup_situation(&hg);

    hg.add_participant(make_participation(eid, s1, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, s2, Role::Witness))
        .unwrap();

    let situations = hg.get_situations_for_entity(&eid).unwrap();
    assert_eq!(situations.len(), 2);
}

#[test]
fn test_remove_participant() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    hg.remove_participant(&eid, &sid, None).unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    assert!(participants.is_empty());
    let situations = hg.get_situations_for_entity(&eid).unwrap();
    assert!(situations.is_empty());
}

#[test]
fn test_remove_participant_not_found() {
    let hg = Hypergraph::new(test_store());
    let eid = Uuid::now_v7();
    let sid = Uuid::now_v7();
    let result = hg.remove_participant(&eid, &sid, None);
    assert!(matches!(
        result,
        Err(TensaError::ParticipationNotFound { .. })
    ));
}

#[test]
fn test_participation_multi_role() {
    // Formerly test_participation_duplicate_rejected — now multi-role is allowed.
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    // Same entity, same situation, different role — should succeed
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(participants.len(), 2);
    assert_eq!(participants[0].seq, 0);
    assert_eq!(participants[1].seq, 1);
}

#[test]
fn test_participation_with_role() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(Participation {
        entity_id: eid,
        situation_id: sid,
        role: Role::Custom("Narrator".to_string()),
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(participants[0].role, Role::Custom("Narrator".to_string()));
}

#[test]
fn test_participation_with_info_set() {
    let hg = Hypergraph::new(test_store());
    let e1 = setup_entity(&hg, EntityType::Actor);
    let e2 = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(Participation {
        entity_id: e1,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![KnowledgeFact {
                about_entity: e2,
                fact: "is a detective".to_string(),
                confidence: 0.95,
            }],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    let info = participants[0].info_set.as_ref().unwrap();
    assert_eq!(info.knows_before.len(), 1);
    assert_eq!(info.knows_before[0].about_entity, e2);
}

#[test]
fn test_participation_with_action_and_payoff() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(Participation {
        entity_id: eid,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: None,
        action: Some("confesses".to_string()),
        payoff: Some(serde_json::json!({"freedom": -1, "redemption": 1})),
        seq: 0,
    })
    .unwrap();

    let participants = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(participants[0].action.as_deref(), Some("confesses"));
    assert!(participants[0].payoff.is_some());
}

#[test]
fn test_filter_by_role() {
    let hg = Hypergraph::new(test_store());
    let e1 = setup_entity(&hg, EntityType::Actor);
    let e2 = setup_entity(&hg, EntityType::Actor);
    let e3 = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    for (eid, role) in [
        (e1, Role::Protagonist),
        (e2, Role::Antagonist),
        (e3, Role::Witness),
    ] {
        hg.add_participant(make_participation(eid, sid, role))
            .unwrap();
    }

    let all = hg.get_participants_for_situation(&sid).unwrap();
    let protagonists = Hypergraph::filter_by_role(&all, &Role::Protagonist);
    assert_eq!(protagonists.len(), 1);
    assert_eq!(protagonists[0].entity_id, e1);
}

#[test]
fn test_add_participant_entity_not_found() {
    let hg = Hypergraph::new(test_store());
    let sid = setup_situation(&hg);
    let result = hg.add_participant(make_participation(Uuid::now_v7(), sid, Role::Protagonist));
    assert!(matches!(result, Err(TensaError::EntityNotFound(_))));
}

#[test]
fn test_add_participant_situation_not_found() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let result = hg.add_participant(make_participation(eid, Uuid::now_v7(), Role::Protagonist));
    assert!(matches!(result, Err(TensaError::SituationNotFound(_))));
}

#[test]
fn test_bidirectional_index_consistency() {
    let hg = Hypergraph::new(test_store());
    let e1 = setup_entity(&hg, EntityType::Actor);
    let e2 = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(e1, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(e2, sid, Role::Antagonist))
        .unwrap();

    // Forward: each entity should see 1 situation
    assert_eq!(hg.get_situations_for_entity(&e1).unwrap().len(), 1);
    assert_eq!(hg.get_situations_for_entity(&e2).unwrap().len(), 1);
    // Reverse: situation should see 2 entities
    assert_eq!(hg.get_participants_for_situation(&sid).unwrap().len(), 2);
}

#[test]
fn test_knowledge_tracking() {
    let hg = Hypergraph::new(test_store());
    let e1 = setup_entity(&hg, EntityType::Actor);
    let e2 = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(Participation {
        entity_id: e1,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![KnowledgeFact {
                about_entity: e2,
                fact: "is alive".to_string(),
                confidence: 1.0,
            }],
            learns: vec![KnowledgeFact {
                about_entity: e2,
                fact: "is a spy".to_string(),
                confidence: 0.7,
            }],
            reveals: vec![KnowledgeFact {
                about_entity: e1,
                fact: "own identity".to_string(),
                confidence: 1.0,
            }],
            beliefs_about_others: vec![],
        }),
        action: Some("interrogates".to_string()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    let p = &hg.get_participants_for_situation(&sid).unwrap()[0];
    let info = p.info_set.as_ref().unwrap();
    assert_eq!(info.knows_before.len(), 1);
    assert_eq!(info.learns.len(), 1);
    assert_eq!(info.reveals.len(), 1);
    assert_eq!(info.learns[0].fact, "is a spy");
}

// ─── New multi-role tests ──────────────────────────────────────

#[test]
fn test_add_participant_multi_role() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap();

    let parts = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].role, Role::Protagonist);
    assert_eq!(parts[0].seq, 0);
    assert_eq!(parts[1].role, Role::Witness);
    assert_eq!(parts[1].seq, 1);
}

#[test]
fn test_seq_auto_increment() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Antagonist))
        .unwrap();

    let parts = hg.get_participations_for_pair(&eid, &sid).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].seq, 0);
    assert_eq!(parts[1].seq, 1);
    assert_eq!(parts[2].seq, 2);
}

#[test]
fn test_get_participations_for_pair() {
    let hg = Hypergraph::new(test_store());
    let e1 = setup_entity(&hg, EntityType::Actor);
    let e2 = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(e1, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(e1, sid, Role::Witness))
        .unwrap();
    hg.add_participant(make_participation(e2, sid, Role::Antagonist))
        .unwrap();

    let pair_parts = hg.get_participations_for_pair(&e1, &sid).unwrap();
    assert_eq!(pair_parts.len(), 2);
    let pair_parts2 = hg.get_participations_for_pair(&e2, &sid).unwrap();
    assert_eq!(pair_parts2.len(), 1);
}

#[test]
fn test_remove_participant_specific_seq() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Antagonist))
        .unwrap();

    // Remove seq=1 (Witness)
    hg.remove_participant(&eid, &sid, Some(1)).unwrap();

    let parts = hg.get_participations_for_pair(&eid, &sid).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].role, Role::Protagonist);
    assert_eq!(parts[0].seq, 0);
    assert_eq!(parts[1].role, Role::Antagonist);
    assert_eq!(parts[1].seq, 2);
}

#[test]
fn test_remove_participant_all() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap();

    // Remove all with seq=None
    hg.remove_participant(&eid, &sid, None).unwrap();

    let parts = hg.get_participations_for_pair(&eid, &sid).unwrap();
    assert!(parts.is_empty());
    let sit_parts = hg.get_participants_for_situation(&sid).unwrap();
    assert!(sit_parts.is_empty());
}

#[test]
fn test_bidirectional_index_multi_role() {
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap();
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap();

    // Forward scan should return both
    let forward = hg.get_situations_for_entity(&eid).unwrap();
    assert_eq!(forward.len(), 2);

    // Reverse scan should return both
    let reverse = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(reverse.len(), 2);

    // Both should have matching roles
    let roles: Vec<_> = forward.iter().map(|p| p.role.clone()).collect();
    assert!(roles.contains(&Role::Protagonist));
    assert!(roles.contains(&Role::Witness));
}

#[test]
fn test_remove_all_after_gap() {
    // Regression: bulk delete must use actual stored keys, not contiguous indices
    let hg = Hypergraph::new(test_store());
    let eid = setup_entity(&hg, EntityType::Actor);
    let sid = setup_situation(&hg);

    hg.add_participant(make_participation(eid, sid, Role::Protagonist))
        .unwrap(); // seq 0
    hg.add_participant(make_participation(eid, sid, Role::Witness))
        .unwrap(); // seq 1
    hg.add_participant(make_participation(eid, sid, Role::Antagonist))
        .unwrap(); // seq 2

    // Delete middle entry, creating a gap: seq 0, 2 remain
    hg.remove_participant(&eid, &sid, Some(1)).unwrap();
    assert_eq!(hg.get_participations_for_pair(&eid, &sid).unwrap().len(), 2);

    // Bulk delete must remove all remaining entries despite gap
    hg.remove_participant(&eid, &sid, None).unwrap();
    assert!(hg
        .get_participations_for_pair(&eid, &sid)
        .unwrap()
        .is_empty());
    // Reverse index must also be clean
    assert!(hg.get_participants_for_situation(&sid).unwrap().is_empty());
}
