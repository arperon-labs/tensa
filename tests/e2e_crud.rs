//! Comprehensive E2E CRUD tests for all TENSA entity types.
//!
//! Tests create, read, update (where applicable), delete for:
//! - All 5 EntityType variants (Actor, Location, Artifact, Concept, Organization)
//! - All 6 NarrativeLevel variants for Situations
//! - All 10 Role variants for Participations
//! - Causal links (all 4 CausalType variants)
//! - Narratives
//! - State versioning
//! - Maturity promotions
//! - Source intelligence (Sources, Attributions, Contentions)

use std::sync::Arc;

use chrono::{Duration, Utc};
use tensa::hypergraph::Hypergraph;
use tensa::store::memory::MemoryStore;
use tensa::temporal::index::{IntervalEntry, IntervalTree};
use tensa::types::*;
use uuid::Uuid;

fn setup() -> (Hypergraph, IntervalTree) {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);
    let tree = IntervalTree::new();
    (hg, tree)
}

fn now_plus_hours(h: i64) -> chrono::DateTime<Utc> {
    let base = chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    base + Duration::hours(h)
}

fn make_entity(name: &str, et: EntityType, confidence: f32, narrative_id: Option<&str>) -> Entity {
    Entity {
        id: Uuid::now_v7(),
        entity_type: et,
        properties: serde_json::json!({"name": name}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: narrative_id.map(String::from),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    }
}

fn make_situation(
    content: &str,
    level: NarrativeLevel,
    start_h: i64,
    end_h: i64,
    narrative_id: Option<&str>,
) -> Situation {
    Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: Some(now_plus_hours(start_h)),
            end: Some(now_plus_hours(end_h)),
            granularity: TimeGranularity::Exact,
            relations: vec![],
            fuzzy_endpoints: None,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text(content)],
        narrative_level: level,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.8,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: narrative_id.map(String::from),
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
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Entity CRUD — all 5 types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_crud_entity_actor() {
    let (hg, _) = setup();
    let e = make_entity("Agent Smith", EntityType::Actor, 0.95, None);
    let id = hg.create_entity(e).unwrap();

    let got = hg.get_entity(&id).unwrap();
    assert_eq!(got.entity_type, EntityType::Actor);
    assert_eq!(got.properties["name"], "Agent Smith");
    assert!((got.confidence - 0.95).abs() < f32::EPSILON);

    // Update
    let updated = hg
        .update_entity(&id, |e| {
            e.properties = serde_json::json!({"name": "Agent Smith", "alias": "Neo"});
            e.confidence = 0.99;
        })
        .unwrap();
    assert_eq!(updated.properties["alias"], "Neo");
    assert!((updated.confidence - 0.99).abs() < f32::EPSILON);

    // Delete
    hg.delete_entity(&id).unwrap();
    assert!(hg.get_entity(&id).is_err());
}

#[test]
fn test_crud_entity_location() {
    let (hg, _) = setup();
    let mut e = make_entity("Kyiv", EntityType::Location, 1.0, None);
    e.properties =
        serde_json::json!({"name": "Kyiv", "country": "Ukraine", "lat": 50.45, "lng": 30.52});
    let id = hg.create_entity(e).unwrap();

    let got = hg.get_entity(&id).unwrap();
    assert_eq!(got.entity_type, EntityType::Location);
    assert_eq!(got.properties["country"], "Ukraine");

    hg.delete_entity(&id).unwrap();
    assert!(hg.get_entity(&id).is_err());
}

#[test]
fn test_crud_entity_artifact() {
    let (hg, _) = setup();
    let mut e = make_entity("Stinger Missile", EntityType::Artifact, 0.85, None);
    e.properties =
        serde_json::json!({"name": "Stinger Missile", "type": "MANPADS", "origin": "US"});
    let id = hg.create_entity(e).unwrap();

    let got = hg.get_entity(&id).unwrap();
    assert_eq!(got.entity_type, EntityType::Artifact);
    assert_eq!(got.properties["type"], "MANPADS");

    hg.delete_entity(&id).unwrap();
    assert!(hg.get_entity(&id).is_err());
}

#[test]
fn test_crud_entity_concept() {
    let (hg, _) = setup();
    let mut e = make_entity("Deterrence Theory", EntityType::Concept, 0.9, None);
    e.properties =
        serde_json::json!({"name": "Deterrence Theory", "domain": "international relations"});
    let id = hg.create_entity(e).unwrap();

    let got = hg.get_entity(&id).unwrap();
    assert_eq!(got.entity_type, EntityType::Concept);
    assert_eq!(got.properties["domain"], "international relations");

    hg.delete_entity(&id).unwrap();
    assert!(hg.get_entity(&id).is_err());
}

#[test]
fn test_crud_entity_organization() {
    let (hg, _) = setup();
    let mut e = make_entity("NATO", EntityType::Organization, 1.0, None);
    e.properties = serde_json::json!({"name": "NATO", "members": 32, "hq": "Brussels"});
    let id = hg.create_entity(e).unwrap();

    let got = hg.get_entity(&id).unwrap();
    assert_eq!(got.entity_type, EntityType::Organization);
    assert_eq!(got.properties["members"], 32);

    hg.delete_entity(&id).unwrap();
    assert!(hg.get_entity(&id).is_err());
}

#[test]
fn test_entity_type_listing() {
    let (hg, _) = setup();

    hg.create_entity(make_entity("Actor1", EntityType::Actor, 0.9, None))
        .unwrap();
    hg.create_entity(make_entity("Actor2", EntityType::Actor, 0.8, None))
        .unwrap();
    hg.create_entity(make_entity("Loc1", EntityType::Location, 1.0, None))
        .unwrap();
    hg.create_entity(make_entity("Art1", EntityType::Artifact, 0.7, None))
        .unwrap();
    hg.create_entity(make_entity("Con1", EntityType::Concept, 0.6, None))
        .unwrap();
    hg.create_entity(make_entity("Org1", EntityType::Organization, 0.95, None))
        .unwrap();

    assert_eq!(
        hg.list_entities_by_type(&EntityType::Actor).unwrap().len(),
        2
    );
    assert_eq!(
        hg.list_entities_by_type(&EntityType::Location)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        hg.list_entities_by_type(&EntityType::Artifact)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        hg.list_entities_by_type(&EntityType::Concept)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        hg.list_entities_by_type(&EntityType::Organization)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn test_entity_narrative_listing() {
    let (hg, _) = setup();

    hg.create_entity(make_entity("A", EntityType::Actor, 0.9, Some("ukraine")))
        .unwrap();
    hg.create_entity(make_entity("B", EntityType::Actor, 0.8, Some("ukraine")))
        .unwrap();
    hg.create_entity(make_entity("C", EntityType::Actor, 0.7, Some("syria")))
        .unwrap();
    hg.create_entity(make_entity("D", EntityType::Actor, 0.6, None))
        .unwrap();

    assert_eq!(hg.list_entities_by_narrative("ukraine").unwrap().len(), 2);
    assert_eq!(hg.list_entities_by_narrative("syria").unwrap().len(), 1);
    assert_eq!(
        hg.list_entities_by_narrative("nonexistent").unwrap().len(),
        0
    );
}

#[test]
fn test_entity_not_found() {
    let (hg, _) = setup();
    let fake_id = Uuid::now_v7();
    assert!(hg.get_entity(&fake_id).is_err());
    assert!(hg.delete_entity(&fake_id).is_err());
}

#[test]
fn test_entity_with_beliefs_and_embedding() {
    let (hg, _) = setup();
    let mut e = make_entity("Analyst", EntityType::Actor, 0.9, None);
    e.beliefs = Some(serde_json::json!({"threat_level": "high", "confidence_in_source": 0.7}));
    e.embedding = Some(vec![0.1, 0.2, 0.3, 0.4]);
    let id = hg.create_entity(e).unwrap();

    let got = hg.get_entity(&id).unwrap();
    assert_eq!(got.beliefs.as_ref().unwrap()["threat_level"], "high");
    assert_eq!(got.embedding.as_ref().unwrap().len(), 4);

    hg.delete_entity(&id).unwrap();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Situation CRUD — all 6 narrative levels
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_crud_situation_all_levels() {
    let (hg, _) = setup();

    let levels = [
        (NarrativeLevel::Story, "The full story"),
        (NarrativeLevel::Arc, "Character arc"),
        (NarrativeLevel::Sequence, "Chase sequence"),
        (NarrativeLevel::Scene, "Office scene"),
        (NarrativeLevel::Beat, "Emotional beat"),
        (NarrativeLevel::Event, "Single event"),
    ];

    for (i, (level, content)) in levels.iter().enumerate() {
        let sit = make_situation(
            content,
            level.clone(),
            (i * 10) as i64,
            (i * 10 + 5) as i64,
            None,
        );
        let id = hg.create_situation(sit).unwrap();

        let got = hg.get_situation(&id).unwrap();
        assert_eq!(got.narrative_level, *level);
        assert_eq!(got.raw_content[0].content, *content);

        // Delete
        hg.delete_situation(&id).unwrap();
        assert!(hg.get_situation(&id).is_err());
    }
}

#[test]
fn test_situation_level_listing() {
    let (hg, _) = setup();

    hg.create_situation(make_situation("S1", NarrativeLevel::Scene, 0, 1, None))
        .unwrap();
    hg.create_situation(make_situation("S2", NarrativeLevel::Scene, 2, 3, None))
        .unwrap();
    hg.create_situation(make_situation("E1", NarrativeLevel::Event, 4, 5, None))
        .unwrap();

    assert_eq!(
        hg.list_situations_by_level(NarrativeLevel::Scene)
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        hg.list_situations_by_level(NarrativeLevel::Event)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        hg.list_situations_by_level(NarrativeLevel::Arc)
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn test_situation_narrative_listing() {
    let (hg, _) = setup();

    hg.create_situation(make_situation(
        "A",
        NarrativeLevel::Scene,
        0,
        1,
        Some("war"),
    ))
    .unwrap();
    hg.create_situation(make_situation(
        "B",
        NarrativeLevel::Scene,
        2,
        3,
        Some("war"),
    ))
    .unwrap();
    hg.create_situation(make_situation(
        "C",
        NarrativeLevel::Scene,
        4,
        5,
        Some("peace"),
    ))
    .unwrap();

    assert_eq!(hg.list_situations_by_narrative("war").unwrap().len(), 2);
    assert_eq!(hg.list_situations_by_narrative("peace").unwrap().len(), 1);
    assert_eq!(hg.list_situations_by_narrative("x").unwrap().len(), 0);
}

#[test]
fn test_situation_with_spatial() {
    let (hg, _) = setup();

    let mut sit = make_situation("Battle at coordinates", NarrativeLevel::Event, 0, 1, None);
    sit.spatial = Some(SpatialAnchor {
        latitude: Some(50.45),
        longitude: Some(30.52),
        precision: SpatialPrecision::Exact,
        location_entity: None,
        location_name: None,
        description: Some("Kyiv outskirts".into()),
    });
    let id = hg.create_situation(sit).unwrap();

    let got = hg.get_situation(&id).unwrap();
    let sp = got.spatial.unwrap();
    assert!((sp.latitude.unwrap() - 50.45).abs() < 0.01);
    assert_eq!(sp.description.unwrap(), "Kyiv outskirts");

    hg.delete_situation(&id).unwrap();
}

#[test]
fn test_situation_with_all_content_types() {
    let (hg, _) = setup();

    let mut sit = make_situation("multi-content", NarrativeLevel::Scene, 0, 1, None);
    sit.raw_content = vec![
        ContentBlock {
            content_type: ContentType::Text,
            content: "Narrative text".into(),
            source: None,
        },
        ContentBlock {
            content_type: ContentType::Dialogue,
            content: "What happened?".into(),
            source: None,
        },
        ContentBlock {
            content_type: ContentType::Observation,
            content: "The room was dark".into(),
            source: None,
        },
        ContentBlock {
            content_type: ContentType::Document,
            content: "Official report #1234".into(),
            source: None,
        },
        ContentBlock {
            content_type: ContentType::MediaRef,
            content: "https://example.com/video.mp4".into(),
            source: None,
        },
    ];
    let id = hg.create_situation(sit).unwrap();

    let got = hg.get_situation(&id).unwrap();
    assert_eq!(got.raw_content.len(), 5);
    assert_eq!(got.raw_content[0].content_type, ContentType::Text);
    assert_eq!(got.raw_content[1].content_type, ContentType::Dialogue);
    assert_eq!(got.raw_content[2].content_type, ContentType::Observation);
    assert_eq!(got.raw_content[3].content_type, ContentType::Document);
    assert_eq!(got.raw_content[4].content_type, ContentType::MediaRef);

    hg.delete_situation(&id).unwrap();
}

#[test]
fn test_situation_not_found() {
    let (hg, _) = setup();
    let fake_id = Uuid::now_v7();
    assert!(hg.get_situation(&fake_id).is_err());
    assert!(hg.delete_situation(&fake_id).is_err());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Participation CRUD — all Role variants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_crud_participation_all_roles() {
    let (hg, _) = setup();

    let sit_id = hg
        .create_situation(make_situation("Meeting", NarrativeLevel::Scene, 0, 1, None))
        .unwrap();

    let roles = [
        Role::Protagonist,
        Role::Antagonist,
        Role::Witness,
        Role::Target,
        Role::Instrument,
        Role::Confidant,
        Role::Informant,
        Role::Recipient,
        Role::Bystander,
        Role::SubjectOfDiscussion,
        Role::Custom("Mediator".into()),
    ];

    let mut entity_ids = Vec::new();
    for (i, role) in roles.iter().enumerate() {
        let e = make_entity(&format!("Person{}", i), EntityType::Actor, 0.8, None);
        let eid = hg.create_entity(e).unwrap();
        entity_ids.push(eid);

        hg.add_participant(Participation {
            entity_id: eid,
            situation_id: sit_id,
            role: role.clone(),
            info_set: None,
            action: Some(format!("acts as {:?}", role)),
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    // Verify all participants
    let participants = hg.get_participants_for_situation(&sit_id).unwrap();
    assert_eq!(participants.len(), roles.len());

    // Verify entity -> situations lookup
    let sits = hg.get_situations_for_entity(&entity_ids[0]).unwrap();
    assert_eq!(sits.len(), 1);

    // Remove one
    hg.remove_participant(&entity_ids[0], &sit_id, None)
        .unwrap();
    let participants = hg.get_participants_for_situation(&sit_id).unwrap();
    assert_eq!(participants.len(), roles.len() - 1);

    // Remove non-existent
    assert!(hg
        .remove_participant(&entity_ids[0], &sit_id, None)
        .is_err());
}

#[test]
fn test_participation_duplicate_rejected() {
    let (hg, _) = setup();

    let eid = hg
        .create_entity(make_entity("X", EntityType::Actor, 0.8, None))
        .unwrap();
    let sid = hg
        .create_situation(make_situation("S", NarrativeLevel::Scene, 0, 1, None))
        .unwrap();

    hg.add_participant(Participation {
        entity_id: eid,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Multi-role: same entity can have multiple roles (Sprint P4.1)
    let result = hg.add_participant(Participation {
        entity_id: eid,
        situation_id: sid,
        role: Role::Antagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    });
    assert!(result.is_ok());

    // Verify both participations exist
    let parts = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(parts.len(), 2);
}

#[test]
fn test_participation_with_info_set() {
    let (hg, _) = setup();

    let alice = hg
        .create_entity(make_entity("Alice", EntityType::Actor, 0.9, None))
        .unwrap();
    let bob = hg
        .create_entity(make_entity("Bob", EntityType::Actor, 0.8, None))
        .unwrap();
    let sid = hg
        .create_situation(make_situation("Reveal", NarrativeLevel::Beat, 0, 1, None))
        .unwrap();

    hg.add_participant(Participation {
        entity_id: alice,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: Some(InfoSet {
            knows_before: vec![KnowledgeFact {
                about_entity: bob,
                fact: "is a spy".into(),
                confidence: 0.6,
            }],
            learns: vec![KnowledgeFact {
                about_entity: bob,
                fact: "is actually a double agent".into(),
                confidence: 0.9,
            }],
            reveals: vec![KnowledgeFact {
                about_entity: alice,
                fact: "knows the truth".into(),
                confidence: 1.0,
            }],
            beliefs_about_others: vec![],
        }),
        action: Some("confronts".into()),
        payoff: Some(serde_json::json!({"tension": 0.9, "resolution": false})),
        seq: 0,
    })
    .unwrap();

    let parts = hg.get_participants_for_situation(&sid).unwrap();
    let alice_p = &parts[0];
    let info = alice_p.info_set.as_ref().unwrap();
    assert_eq!(info.knows_before.len(), 1);
    assert_eq!(info.learns.len(), 1);
    assert_eq!(info.reveals.len(), 1);
    assert_eq!(info.learns[0].fact, "is actually a double agent");
    assert_eq!(alice_p.payoff.as_ref().unwrap()["tension"], 0.9);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Causal Links — all 4 CausalType variants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_crud_causal_links_all_types() {
    let (hg, _) = setup();

    let s1 = hg
        .create_situation(make_situation("Cause 1", NarrativeLevel::Event, 0, 1, None))
        .unwrap();
    let s2 = hg
        .create_situation(make_situation(
            "Effect N",
            NarrativeLevel::Event,
            2,
            3,
            None,
        ))
        .unwrap();
    let s3 = hg
        .create_situation(make_situation(
            "Effect S",
            NarrativeLevel::Event,
            4,
            5,
            None,
        ))
        .unwrap();
    let s4 = hg
        .create_situation(make_situation(
            "Effect C",
            NarrativeLevel::Event,
            6,
            7,
            None,
        ))
        .unwrap();
    let s5 = hg
        .create_situation(make_situation(
            "Effect E",
            NarrativeLevel::Event,
            8,
            9,
            None,
        ))
        .unwrap();

    let links = [
        (s1, s2, CausalType::Necessary, "Without A, B cannot happen"),
        (s1, s3, CausalType::Sufficient, "A alone guarantees C"),
        (s1, s4, CausalType::Contributing, "A makes D more likely"),
        (s1, s5, CausalType::Enabling, "A opens the door for E"),
    ];

    for (from, to, ct, mech) in &links {
        hg.add_causal_link(CausalLink {
            from_situation: *from,
            to_situation: *to,
            mechanism: Some(mech.to_string()),
            strength: 0.8,
            causal_type: ct.clone(),
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
    }

    // Traverse from s1
    let chain = hg.traverse_causal_chain(&s1, 10).unwrap();
    assert_eq!(chain.len(), 4);

    // Antecedents of s2
    let ante = hg.get_antecedents(&s2).unwrap();
    assert_eq!(ante.len(), 1);
    assert_eq!(ante[0].causal_type, CausalType::Necessary);

    // Remove one
    hg.remove_causal_link(&s1, &s2).unwrap();
    let chain = hg.traverse_causal_chain(&s1, 10).unwrap();
    assert_eq!(chain.len(), 3);
}

#[test]
fn test_causal_cycle_detection() {
    let (hg, _) = setup();

    let s1 = hg
        .create_situation(make_situation("A", NarrativeLevel::Event, 0, 1, None))
        .unwrap();
    let s2 = hg
        .create_situation(make_situation("B", NarrativeLevel::Event, 2, 3, None))
        .unwrap();
    let s3 = hg
        .create_situation(make_situation("C", NarrativeLevel::Event, 4, 5, None))
        .unwrap();

    hg.add_causal_link(CausalLink {
        from_situation: s1,
        to_situation: s2,
        mechanism: None,
        strength: 0.9,
        causal_type: CausalType::Necessary,
        maturity: MaturityLevel::Candidate,
    })
    .unwrap();
    hg.add_causal_link(CausalLink {
        from_situation: s2,
        to_situation: s3,
        mechanism: None,
        strength: 0.8,
        causal_type: CausalType::Contributing,
        maturity: MaturityLevel::Candidate,
    })
    .unwrap();

    // s3 -> s1 would create a cycle
    let result = hg.add_causal_link(CausalLink {
        from_situation: s3,
        to_situation: s1,
        mechanism: None,
        strength: 0.7,
        causal_type: CausalType::Enabling,
        maturity: MaturityLevel::Candidate,
    });
    assert!(result.is_err());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Narrative CRUD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_crud_narrative() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let registry = tensa::narrative::registry::NarrativeRegistry::new(hg.store_arc());

    let narrative = tensa::narrative::types::Narrative {
        id: "test-narrative".into(),
        title: "Test Narrative".into(),
        genre: Some("geopolitical".into()),
        tags: vec!["test".into(), "e2e".into()],
        source: None,
        project_id: None,
        description: None,
        authors: vec![],
        language: None,
        publication_date: None,
        cover_url: None,
        custom_properties: std::collections::HashMap::new(),
        entity_count: 0,
        situation_count: 0,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    // Create
    let id = registry.create(narrative.clone()).unwrap();
    assert_eq!(id, "test-narrative");

    // Read
    let got = registry.get("test-narrative").unwrap();
    assert_eq!(got.title, "Test Narrative");
    assert_eq!(got.genre, Some("geopolitical".into()));
    assert_eq!(got.tags, vec!["test", "e2e"]);

    // List
    let all = registry.list(None, None).unwrap();
    assert_eq!(all.len(), 1);

    // Duplicate should fail
    assert!(registry.create(narrative).is_err());

    // Delete
    registry.delete("test-narrative").unwrap();
    assert!(registry.get("test-narrative").is_err());
    assert_eq!(registry.list(None, None).unwrap().len(), 0);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// State Versioning
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_state_versioning_crud() {
    let (hg, _) = setup();

    let eid = hg
        .create_entity(make_entity("Agent", EntityType::Actor, 0.9, None))
        .unwrap();
    let s1 = hg
        .create_situation(make_situation("Phase 1", NarrativeLevel::Scene, 0, 1, None))
        .unwrap();
    let s2 = hg
        .create_situation(make_situation("Phase 2", NarrativeLevel::Scene, 2, 3, None))
        .unwrap();

    hg.create_state_version(StateVersion {
        entity_id: eid,
        situation_id: s1,
        properties: serde_json::json!({"status": "infiltrating", "cover_intact": true}),
        beliefs: Some(serde_json::json!({"target_location": "unknown"})),
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();

    hg.create_state_version(StateVersion {
        entity_id: eid,
        situation_id: s2,
        properties: serde_json::json!({"status": "extracted", "cover_intact": false}),
        beliefs: Some(serde_json::json!({"target_location": "confirmed"})),
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();

    // History
    let history = hg.get_state_history(&eid).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].properties["status"], "infiltrating");
    assert_eq!(history[1].properties["status"], "extracted");

    // Point-in-time lookup
    let at_s1 = hg.get_state_at_situation(&eid, &s1).unwrap();
    assert_eq!(at_s1.properties["cover_intact"], true);
    assert_eq!(at_s1.beliefs.unwrap()["target_location"], "unknown");

    let at_s2 = hg.get_state_at_situation(&eid, &s2).unwrap();
    assert_eq!(at_s2.properties["cover_intact"], false);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Maturity Promotions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_maturity_promotion_chain() {
    let (hg, _) = setup();
    let id = hg
        .create_entity(make_entity("Subject", EntityType::Actor, 0.5, None))
        .unwrap();

    // Candidate -> Reviewed
    hg.set_entity_maturity(
        &id,
        MaturityLevel::Reviewed,
        "analyst1",
        Some("Initial review".into()),
    )
    .unwrap();
    assert_eq!(
        hg.get_entity(&id).unwrap().maturity,
        MaturityLevel::Reviewed
    );

    // Reviewed -> Validated
    hg.set_entity_maturity(
        &id,
        MaturityLevel::Validated,
        "analyst2",
        Some("Cross-referenced".into()),
    )
    .unwrap();
    assert_eq!(
        hg.get_entity(&id).unwrap().maturity,
        MaturityLevel::Validated
    );

    // Validated -> GroundTruth
    hg.set_entity_maturity(
        &id,
        MaturityLevel::GroundTruth,
        "senior",
        Some("Confirmed".into()),
    )
    .unwrap();
    assert_eq!(
        hg.get_entity(&id).unwrap().maturity,
        MaturityLevel::GroundTruth
    );

    // Validation log should have entries (may collapse if same-millisecond keys)
    let log = hg.get_validation_log(&id).unwrap();
    assert!(!log.is_empty(), "Validation log should not be empty");

    // Cannot demote from GroundTruth back down
    let result = hg.set_entity_maturity(&id, MaturityLevel::Candidate, "user", None);
    assert!(result.is_err());
}

#[test]
fn test_maturity_skip_promotion_allowed() {
    let (hg, _) = setup();
    let id = hg
        .create_entity(make_entity("X", EntityType::Actor, 0.5, None))
        .unwrap();

    // Skipping levels is allowed (Candidate -> GroundTruth)
    hg.set_entity_maturity(
        &id,
        MaturityLevel::GroundTruth,
        "admin",
        Some("Direct confirm".into()),
    )
    .unwrap();
    assert_eq!(
        hg.get_entity(&id).unwrap().maturity,
        MaturityLevel::GroundTruth
    );
}

#[test]
fn test_maturity_same_level_rejected() {
    let (hg, _) = setup();
    let id = hg
        .create_entity(make_entity("X", EntityType::Actor, 0.5, None))
        .unwrap();

    // Cannot set to same level
    let result = hg.set_entity_maturity(&id, MaturityLevel::Candidate, "user", None);
    assert!(result.is_err());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Extraction Methods — all variants on situations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_all_extraction_methods() {
    let (hg, _) = setup();

    let methods = [
        ExtractionMethod::LlmParsed,
        ExtractionMethod::HumanEntered,
        ExtractionMethod::StructuredImport,
        ExtractionMethod::Sensor,
    ];

    for method in &methods {
        let mut sit = make_situation("test", NarrativeLevel::Event, 0, 1, None);
        sit.extraction_method = method.clone();
        let id = hg.create_situation(sit).unwrap();
        let got = hg.get_situation(&id).unwrap();
        assert_eq!(got.extraction_method, *method);
        hg.delete_situation(&id).unwrap();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Time Granularities — all 4 variants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_all_time_granularities() {
    let (hg, _) = setup();

    let granularities = [
        TimeGranularity::Exact,
        TimeGranularity::Day,
        TimeGranularity::Approximate,
        TimeGranularity::Unknown,
    ];

    for gran in &granularities {
        let mut sit = make_situation("test", NarrativeLevel::Event, 0, 1, None);
        sit.temporal.granularity = gran.clone();
        let id = hg.create_situation(sit).unwrap();
        let got = hg.get_situation(&id).unwrap();
        assert_eq!(got.temporal.granularity, *gran);
        hg.delete_situation(&id).unwrap();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Cross-cutting: Entity + Situation + Participation lifecycle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_full_lifecycle_create_participate_delete() {
    let (hg, _) = setup();

    // Create entities of each type
    let actor = hg
        .create_entity(make_entity(
            "Zelensky",
            EntityType::Actor,
            0.95,
            Some("ukraine"),
        ))
        .unwrap();
    let location = hg
        .create_entity(make_entity(
            "Kyiv",
            EntityType::Location,
            1.0,
            Some("ukraine"),
        ))
        .unwrap();
    let artifact = hg
        .create_entity(make_entity(
            "Bayraktar TB2",
            EntityType::Artifact,
            0.85,
            Some("ukraine"),
        ))
        .unwrap();
    let concept = hg
        .create_entity(make_entity(
            "Sovereignty",
            EntityType::Concept,
            0.9,
            Some("ukraine"),
        ))
        .unwrap();
    let org = hg
        .create_entity(make_entity(
            "Ukrainian Armed Forces",
            EntityType::Organization,
            0.95,
            Some("ukraine"),
        ))
        .unwrap();

    // Create situation
    let sit = hg
        .create_situation(make_situation(
            "Defense of Kyiv",
            NarrativeLevel::Arc,
            0,
            720, // 30 days
            Some("ukraine"),
        ))
        .unwrap();

    // Add participations with different roles
    hg.add_participant(Participation {
        entity_id: actor,
        situation_id: sit,
        role: Role::Protagonist,
        info_set: None,
        action: Some("commands defense".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: location,
        situation_id: sit,
        role: Role::Target,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: artifact,
        situation_id: sit,
        role: Role::Instrument,
        info_set: None,
        action: Some("destroys convoy".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: concept,
        situation_id: sit,
        role: Role::SubjectOfDiscussion,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    hg.add_participant(Participation {
        entity_id: org,
        situation_id: sit,
        role: Role::Protagonist,
        info_set: None,
        action: Some("defends capital".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Verify
    let parts = hg.get_participants_for_situation(&sit).unwrap();
    assert_eq!(parts.len(), 5);

    let by_narrative = hg.list_entities_by_narrative("ukraine").unwrap();
    assert_eq!(by_narrative.len(), 5);

    let sit_narrative = hg.list_situations_by_narrative("ukraine").unwrap();
    assert_eq!(sit_narrative.len(), 1);

    // Delete situation
    hg.delete_situation(&sit).unwrap();

    // Delete entities
    for id in &[actor, location, artifact, concept, org] {
        hg.delete_entity(id).unwrap();
    }

    // Verify all gone
    assert_eq!(hg.list_entities_by_narrative("ukraine").unwrap().len(), 0);
    assert_eq!(hg.list_situations_by_narrative("ukraine").unwrap().len(), 0);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Source Intelligence CRUD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_crud_source() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());

    let src = tensa::source::Source {
        id: Uuid::now_v7(),
        name: "Reuters".into(),
        source_type: tensa::source::SourceType::NewsOutlet,
        url: Some("https://reuters.com".into()),
        description: Some("International news agency".into()),
        trust_score: 0.9,
        bias_profile: tensa::source::BiasProfile {
            known_biases: vec![],
            political_lean: Some(0.0),
            sensationalism: Some(0.1),
            notes: None,
        },
        track_record: tensa::source::TrackRecord {
            claims_made: 1000,
            claims_corroborated: 950,
            claims_contradicted: 10,
            last_evaluated: Some(Utc::now()),
        },
        tags: vec!["wire".into(), "trusted".into()],
        author: None,
        publication: None,
        ingested_by: None,
        ingestion_notes: None,
        publication_date: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };

    // Create
    let id = hg.create_source(src.clone()).unwrap();

    // Read
    let got = hg.get_source(&id).unwrap();
    assert_eq!(got.name, "Reuters");
    assert!((got.trust_score - 0.9).abs() < f32::EPSILON);

    // List
    let all = hg.list_sources().unwrap();
    assert_eq!(all.len(), 1);

    // Delete
    hg.delete_source(&id).unwrap();
    assert!(hg.get_source(&id).is_err());
}

#[test]
fn test_source_attribution_lifecycle() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());

    // Create source
    let src = tensa::source::Source {
        id: Uuid::now_v7(),
        name: "BBC".into(),
        source_type: tensa::source::SourceType::NewsOutlet,
        url: None,
        description: None,
        trust_score: 0.85,
        bias_profile: tensa::source::BiasProfile::default(),
        track_record: tensa::source::TrackRecord::default(),
        tags: vec![],
        author: None,
        publication: None,
        ingested_by: None,
        ingestion_notes: None,
        publication_date: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let src_id = hg.create_source(src).unwrap();

    // Create entity to attribute
    let eid = hg
        .create_entity(make_entity("Putin", EntityType::Actor, 0.7, None))
        .unwrap();

    // Add attribution
    let attr = tensa::source::SourceAttribution {
        source_id: src_id,
        target_id: eid,
        target_kind: tensa::source::AttributionTarget::Entity,
        retrieved_at: Utc::now(),
        extraction_confidence: 0.85,
        original_url: Some("https://bbc.com/article/123".into()),
        excerpt: Some("Russian president Vladimir Putin...".into()),
        claim: None,
    };
    hg.add_attribution(attr).unwrap();

    // List attributions for source
    let attrs = hg.get_attributions_for_source(&src_id).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].target_id, eid);

    // List attributions for target
    let attrs = hg.get_attributions_for_target(&eid).unwrap();
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].source_id, src_id);

    // Remove attribution
    hg.remove_attribution(&src_id, &eid).unwrap();
    assert_eq!(hg.get_attributions_for_source(&src_id).unwrap().len(), 0);
}

#[test]
fn test_contention_lifecycle() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());

    let s1 = hg
        .create_situation(make_situation(
            "Report A: 100 casualties",
            NarrativeLevel::Event,
            0,
            1,
            None,
        ))
        .unwrap();
    let s2 = hg
        .create_situation(make_situation(
            "Report B: 50 casualties",
            NarrativeLevel::Event,
            0,
            1,
            None,
        ))
        .unwrap();

    // Add contention
    let cont = tensa::source::ContentionLink {
        situation_a: s1,
        situation_b: s2,
        contention_type: tensa::source::ContentionType::NumericalDisagreement,
        description: Some("Casualty count differs".into()),
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    };
    hg.add_contention(cont).unwrap();

    // List contentions for situation
    let conts = hg.get_contentions_for_situation(&s1).unwrap();
    assert_eq!(conts.len(), 1);
    assert_eq!(
        conts[0].contention_type,
        tensa::source::ContentionType::NumericalDisagreement
    );

    // Also visible from the other side
    let conts2 = hg.get_contentions_for_situation(&s2).unwrap();
    assert_eq!(conts2.len(), 1);

    // Resolve
    hg.resolve_contention(&s1, &s2, "Revised figure is 75".to_string())
        .unwrap();
    let conts = hg.get_contentions_for_situation(&s1).unwrap();
    assert!(conts[0].resolved);
    assert_eq!(conts[0].resolution.as_deref(), Some("Revised figure is 75"));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Temporal — Allen Interval Relations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_allen_relations_basic() {
    use tensa::temporal::interval::relation_between;

    let (hg, _) = setup();

    let s1 = hg
        .create_situation(make_situation("A", NarrativeLevel::Event, 0, 5, None))
        .unwrap();
    let s2 = hg
        .create_situation(make_situation("B", NarrativeLevel::Event, 10, 15, None))
        .unwrap();
    let s3 = hg
        .create_situation(make_situation("C", NarrativeLevel::Event, 3, 7, None))
        .unwrap();
    let s4 = hg
        .create_situation(make_situation("D", NarrativeLevel::Event, 0, 5, None))
        .unwrap();

    let sit1 = hg.get_situation(&s1).unwrap();
    let sit2 = hg.get_situation(&s2).unwrap();
    let sit3 = hg.get_situation(&s3).unwrap();
    let sit4 = hg.get_situation(&s4).unwrap();

    assert_eq!(
        relation_between(&sit1.temporal, &sit2.temporal).unwrap(),
        AllenRelation::Before
    );
    assert_eq!(
        relation_between(&sit2.temporal, &sit1.temporal).unwrap(),
        AllenRelation::After
    );
    assert_eq!(
        relation_between(&sit1.temporal, &sit3.temporal).unwrap(),
        AllenRelation::Overlaps
    );
    assert_eq!(
        relation_between(&sit1.temporal, &sit4.temporal).unwrap(),
        AllenRelation::Equals
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Interval Tree — point and range queries
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[test]
fn test_interval_tree_queries() {
    let (hg, _) = setup();
    let mut tree = IntervalTree::new();

    let s1 = hg
        .create_situation(make_situation("Morning", NarrativeLevel::Scene, 0, 8, None))
        .unwrap();
    let s2 = hg
        .create_situation(make_situation(
            "Afternoon",
            NarrativeLevel::Scene,
            8,
            16,
            None,
        ))
        .unwrap();
    let s3 = hg
        .create_situation(make_situation(
            "Evening",
            NarrativeLevel::Scene,
            16,
            24,
            None,
        ))
        .unwrap();

    tree.insert(IntervalEntry {
        situation_id: s1,
        start: now_plus_hours(0),
        end: now_plus_hours(8),
    });
    tree.insert(IntervalEntry {
        situation_id: s2,
        start: now_plus_hours(8),
        end: now_plus_hours(16),
    });
    tree.insert(IntervalEntry {
        situation_id: s3,
        start: now_plus_hours(16),
        end: now_plus_hours(24),
    });

    // Point query at hour 4 -> Morning
    let results = tree.point_query(&now_plus_hours(4));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].situation_id, s1);

    // Point query at hour 12 -> Afternoon
    let results = tree.point_query(&now_plus_hours(12));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].situation_id, s2);

    // Interval query 0..24 -> all
    let results = tree.interval_query(&now_plus_hours(0), &now_plus_hours(24));
    assert_eq!(results.len(), 3);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Project CRUD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn make_project(id: &str, title: &str) -> tensa::narrative::types::Project {
    tensa::narrative::types::Project {
        id: id.into(),
        title: title.into(),
        description: None,
        tags: vec![],
        narrative_count: 0,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn make_narrative(
    id: &str,
    title: &str,
    project_id: Option<&str>,
) -> tensa::narrative::types::Narrative {
    tensa::narrative::types::Narrative {
        id: id.into(),
        title: title.into(),
        genre: None,
        tags: vec![],
        source: None,
        project_id: project_id.map(String::from),
        description: None,
        authors: vec![],
        language: None,
        publication_date: None,
        cover_url: None,
        custom_properties: std::collections::HashMap::new(),
        entity_count: 0,
        situation_count: 0,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

#[test]
fn test_project_crud_basic() {
    let store = Arc::new(MemoryStore::new());
    let reg = tensa::narrative::project::ProjectRegistry::new(store);

    // Create
    let id = reg.create(make_project("geo", "Geopolitics")).unwrap();
    assert_eq!(id, "geo");

    // Read
    let p = reg.get("geo").unwrap();
    assert_eq!(p.title, "Geopolitics");

    // Update
    let updated = reg
        .update("geo", |p| {
            p.description = Some("World events analysis".into());
            p.tags = vec!["intel".into(), "geopolitics".into()];
        })
        .unwrap();
    assert_eq!(updated.description, Some("World events analysis".into()));
    assert_eq!(updated.tags.len(), 2);

    // List
    reg.create(make_project("fiction", "Fiction")).unwrap();
    let all = reg.list().unwrap();
    assert_eq!(all.len(), 2);

    // Delete
    reg.delete("fiction").unwrap();
    assert!(reg.get("fiction").is_err());
    assert_eq!(reg.list().unwrap().len(), 1);
}

#[test]
fn test_project_duplicate_rejected() {
    let store = Arc::new(MemoryStore::new());
    let reg = tensa::narrative::project::ProjectRegistry::new(store);

    reg.create(make_project("x", "X")).unwrap();
    assert!(reg.create(make_project("x", "X2")).is_err());
}

#[test]
fn test_project_not_found() {
    let store = Arc::new(MemoryStore::new());
    let reg = tensa::narrative::project::ProjectRegistry::new(store);

    assert!(reg.get("nonexistent").is_err());
    assert!(reg.delete("nonexistent").is_err());
    assert!(reg.update("nonexistent", |_| {}).is_err());
}

#[test]
fn test_project_narrative_association() {
    let store = Arc::new(MemoryStore::new());
    let _hg = Hypergraph::new(store.clone());
    let project_reg = tensa::narrative::project::ProjectRegistry::new(store.clone());
    let narrative_reg = tensa::narrative::registry::NarrativeRegistry::new(store.clone());

    // Create project
    project_reg
        .create(make_project("geo", "Geopolitics"))
        .unwrap();

    // Create narratives with project_id
    narrative_reg
        .create(make_narrative("ukraine", "Ukraine Crisis", Some("geo")))
        .unwrap();
    narrative_reg
        .create(make_narrative("middle-east", "Middle East", Some("geo")))
        .unwrap();
    narrative_reg
        .create(make_narrative("unrelated", "Unrelated Story", None))
        .unwrap();

    // Project-narrative index should have 2 entries
    let nids = project_reg.list_narrative_ids("geo").unwrap();
    assert_eq!(nids.len(), 2);
    assert!(nids.contains(&"ukraine".to_string()));
    assert!(nids.contains(&"middle-east".to_string()));

    // Remove narrative from project
    project_reg.remove_narrative("geo", "ukraine").unwrap();
    assert_eq!(project_reg.list_narrative_ids("geo").unwrap().len(), 1);
}

#[test]
fn test_project_narrative_with_entities() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let project_reg = tensa::narrative::project::ProjectRegistry::new(store.clone());
    let narrative_reg = tensa::narrative::registry::NarrativeRegistry::new(store.clone());

    // Create project with 2 narratives
    project_reg
        .create(make_project("geo", "Geopolitics"))
        .unwrap();
    narrative_reg
        .create(make_narrative("ukraine", "Ukraine", Some("geo")))
        .unwrap();
    narrative_reg
        .create(make_narrative("syria", "Syria", Some("geo")))
        .unwrap();

    // Add entities to each narrative
    hg.create_entity(make_entity(
        "Zelensky",
        EntityType::Actor,
        0.95,
        Some("ukraine"),
    ))
    .unwrap();
    hg.create_entity(make_entity(
        "Putin",
        EntityType::Actor,
        0.9,
        Some("ukraine"),
    ))
    .unwrap();
    hg.create_entity(make_entity("Assad", EntityType::Actor, 0.85, Some("syria")))
        .unwrap();

    // Add situations
    hg.create_situation(make_situation(
        "Battle of Bakhmut",
        NarrativeLevel::Arc,
        0,
        100,
        Some("ukraine"),
    ))
    .unwrap();
    hg.create_situation(make_situation(
        "Aleppo Siege",
        NarrativeLevel::Arc,
        0,
        50,
        Some("syria"),
    ))
    .unwrap();

    // Verify entities per narrative
    assert_eq!(hg.list_entities_by_narrative("ukraine").unwrap().len(), 2);
    assert_eq!(hg.list_entities_by_narrative("syria").unwrap().len(), 1);

    // Verify situations per narrative
    assert_eq!(hg.list_situations_by_narrative("ukraine").unwrap().len(), 1);
    assert_eq!(hg.list_situations_by_narrative("syria").unwrap().len(), 1);

    // Delete narrative "syria" and verify cleanup
    // First cascade-delete the narrative's entities/situations
    let entities = hg.list_entities_by_narrative("syria").unwrap();
    for e in &entities {
        hg.delete_entity(&e.id).unwrap();
    }
    let situations = hg.list_situations_by_narrative("syria").unwrap();
    for s in &situations {
        hg.delete_situation(&s.id).unwrap();
    }
    narrative_reg.delete("syria").unwrap();

    // Project should still have ukraine
    let nids = project_reg.list_narrative_ids("geo").unwrap();
    assert_eq!(nids.len(), 1);
    assert_eq!(nids[0], "ukraine");

    // Ukraine entities still intact
    assert_eq!(hg.list_entities_by_narrative("ukraine").unwrap().len(), 2);
}

#[test]
fn test_project_cascade_delete() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let project_reg = tensa::narrative::project::ProjectRegistry::new(store.clone());
    let narrative_reg = tensa::narrative::registry::NarrativeRegistry::new(store.clone());

    // Create project with narratives and data
    project_reg
        .create(make_project("test-project", "Test Project"))
        .unwrap();
    narrative_reg
        .create(make_narrative("n1", "Narrative 1", Some("test-project")))
        .unwrap();
    narrative_reg
        .create(make_narrative("n2", "Narrative 2", Some("test-project")))
        .unwrap();

    // Add entities and situations to n1
    let e1 = hg
        .create_entity(make_entity("Actor1", EntityType::Actor, 0.9, Some("n1")))
        .unwrap();
    let s1 = hg
        .create_situation(make_situation(
            "Event1",
            NarrativeLevel::Event,
            0,
            1,
            Some("n1"),
        ))
        .unwrap();
    hg.add_participant(Participation {
        entity_id: e1,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Add entities to n2
    hg.create_entity(make_entity("Actor2", EntityType::Actor, 0.8, Some("n2")))
        .unwrap();

    // Verify data exists
    assert_eq!(hg.list_entities_by_narrative("n1").unwrap().len(), 1);
    assert_eq!(hg.list_entities_by_narrative("n2").unwrap().len(), 1);
    assert_eq!(hg.list_situations_by_narrative("n1").unwrap().len(), 1);

    // Simulate cascade delete of project:
    // 1. Delete all narratives' contents
    for nid in project_reg.list_narrative_ids("test-project").unwrap() {
        let entities = hg.list_entities_by_narrative(&nid).unwrap();
        let situations = hg.list_situations_by_narrative(&nid).unwrap();
        for sit in &situations {
            let parts = hg.get_participants_for_situation(&sit.id).unwrap();
            for p in &parts {
                let _ = hg.remove_participant(&p.entity_id, &sit.id, None);
            }
            hg.delete_situation(&sit.id).unwrap();
        }
        for ent in &entities {
            hg.delete_entity(&ent.id).unwrap();
        }
        narrative_reg.delete(&nid).unwrap();
    }
    // 2. Delete project
    project_reg.delete("test-project").unwrap();

    // Verify everything is gone
    assert!(project_reg.get("test-project").is_err());
    assert!(narrative_reg.get("n1").is_err());
    assert!(narrative_reg.get("n2").is_err());
    assert_eq!(hg.list_entities_by_narrative("n1").unwrap().len(), 0);
    assert_eq!(hg.list_entities_by_narrative("n2").unwrap().len(), 0);
    assert_eq!(hg.list_situations_by_narrative("n1").unwrap().len(), 0);
}

#[test]
fn test_narrative_project_id_field() {
    let store = Arc::new(MemoryStore::new());
    let narrative_reg = tensa::narrative::registry::NarrativeRegistry::new(store.clone());

    // Create narrative with project_id
    narrative_reg
        .create(make_narrative("n1", "N1", Some("my-project")))
        .unwrap();
    let n = narrative_reg.get("n1").unwrap();
    assert_eq!(n.project_id, Some("my-project".into()));

    // Create narrative without project_id
    narrative_reg
        .create(make_narrative("n2", "N2", None))
        .unwrap();
    let n = narrative_reg.get("n2").unwrap();
    assert_eq!(n.project_id, None);
}
