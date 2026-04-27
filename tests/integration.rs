use std::sync::Arc;

use chrono::{Duration, Utc};
use tensa::hypergraph::Hypergraph;
use tensa::query::{executor, parser, planner};
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

fn create_entity(hg: &Hypergraph, name: &str, et: EntityType, confidence: f32) -> Uuid {
    let entity = Entity {
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
        narrative_id: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };
    hg.create_entity(entity).unwrap()
}

fn create_situation_at(
    hg: &Hypergraph,
    tree: &mut IntervalTree,
    start_hours: i64,
    end_hours: i64,
    level: NarrativeLevel,
    content: &str,
) -> Uuid {
    let base = chrono::DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let start = base + Duration::hours(start_hours);
    let end = base + Duration::hours(end_hours);

    let sit = Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: Some(start),
            end: Some(end),
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
    let id = hg.create_situation(sit).unwrap();
    tree.insert(IntervalEntry {
        situation_id: id,
        start,
        end,
    });
    id
}

// ─── Full Pipeline Tests ─────────────────────────────────────

#[test]
fn test_full_pipeline_create_and_query_narrative() {
    let (hg, mut tree) = setup();

    // Create characters
    let rask = create_entity(&hg, "Raskolnikov", EntityType::Actor, 0.95);
    let _sonya = create_entity(&hg, "Sonya", EntityType::Actor, 0.90);
    let porfiry = create_entity(&hg, "Porfiry", EntityType::Actor, 0.85);

    // Create locations
    let apartment = create_entity(&hg, "Raskolnikov's Apartment", EntityType::Location, 1.0);
    let _station = create_entity(&hg, "Police Station", EntityType::Location, 1.0);

    // Create situations
    let s1 = create_situation_at(
        &hg,
        &mut tree,
        0,
        2,
        NarrativeLevel::Scene,
        "Raskolnikov plans the murder in his apartment",
    );
    let s2 = create_situation_at(
        &hg,
        &mut tree,
        3,
        5,
        NarrativeLevel::Scene,
        "The murder is committed",
    );
    let s3 = create_situation_at(
        &hg,
        &mut tree,
        10,
        12,
        NarrativeLevel::Scene,
        "Porfiry interrogates Raskolnikov",
    );

    // Add participations
    hg.add_participant(Participation {
        entity_id: rask,
        situation_id: s1,
        role: Role::Protagonist,
        info_set: None,
        action: Some("plans murder".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    hg.add_participant(Participation {
        entity_id: apartment,
        situation_id: s1,
        role: Role::Instrument,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    hg.add_participant(Participation {
        entity_id: rask,
        situation_id: s2,
        role: Role::Protagonist,
        info_set: None,
        action: Some("commits murder".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    hg.add_participant(Participation {
        entity_id: rask,
        situation_id: s3,
        role: Role::Target,
        info_set: Some(InfoSet {
            knows_before: vec![KnowledgeFact {
                about_entity: porfiry,
                fact: "is an investigator".into(),
                confidence: 0.9,
            }],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: Some("denies everything".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    hg.add_participant(Participation {
        entity_id: porfiry,
        situation_id: s3,
        role: Role::Antagonist,
        info_set: Some(InfoSet {
            knows_before: vec![],
            learns: vec![KnowledgeFact {
                about_entity: rask,
                fact: "is lying".into(),
                confidence: 0.7,
            }],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: Some("interrogates".into()),
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Add causal links
    hg.add_causal_link(CausalLink {
        from_situation: s1,
        to_situation: s2,
        mechanism: Some("Planning leads to execution".into()),
        strength: 0.9,
        causal_type: CausalType::Enabling,
        maturity: MaturityLevel::Candidate,
    })
    .unwrap();

    hg.add_causal_link(CausalLink {
        from_situation: s2,
        to_situation: s3,
        mechanism: Some("Murder triggers investigation".into()),
        strength: 0.95,
        causal_type: CausalType::Necessary,
        maturity: MaturityLevel::Candidate,
    })
    .unwrap();

    // Query: Find all actors
    let q = parser::parse_query("MATCH (e:Actor) RETURN e").unwrap();
    let plan = planner::plan_query(&q).unwrap();
    let results = executor::execute(&plan, &hg, &tree).unwrap();
    assert_eq!(results.len(), 3); // Raskolnikov, Sonya, Porfiry

    // Query: Find actors with high confidence
    let q = parser::parse_query("MATCH (e:Actor) WHERE e.confidence > 0.9 RETURN e").unwrap();
    let plan = planner::plan_query(&q).unwrap();
    let results = executor::execute(&plan, &hg, &tree).unwrap();
    assert_eq!(results.len(), 1); // Only Raskolnikov (0.95)

    // Query: Find all situations
    let q = parser::parse_query("MATCH (s:Scene) RETURN s").unwrap();
    let plan = planner::plan_query(&q).unwrap();
    let results = executor::execute(&plan, &hg, &tree).unwrap();
    assert_eq!(results.len(), 3);

    // Verify causal chain traversal
    let chain = hg.traverse_causal_chain(&s1, 10).unwrap();
    assert_eq!(chain.len(), 2); // s1->s2 and s2->s3

    // Verify participation queries
    let rask_situations = hg.get_situations_for_entity(&rask).unwrap();
    assert_eq!(rask_situations.len(), 3); // s1, s2, s3

    let s3_participants = hg.get_participants_for_situation(&s3).unwrap();
    assert_eq!(s3_participants.len(), 2); // Raskolnikov and Porfiry

    // Verify knowledge tracking
    let rask_in_s3 = s3_participants
        .iter()
        .find(|p| p.entity_id == rask)
        .unwrap();
    let info = rask_in_s3.info_set.as_ref().unwrap();
    assert_eq!(info.knows_before.len(), 1);
    assert_eq!(info.knows_before[0].fact, "is an investigator");

    // Verify state versioning
    hg.create_state_version(StateVersion {
        entity_id: rask,
        situation_id: s1,
        properties: serde_json::json!({"mental_state": "anxious", "guilt": 0.0}),
        beliefs: None,
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();
    hg.create_state_version(StateVersion {
        entity_id: rask,
        situation_id: s2,
        properties: serde_json::json!({"mental_state": "terrified", "guilt": 0.8}),
        beliefs: None,
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();

    let history = hg.get_state_history(&rask).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].properties["guilt"], 0.0);
    assert_eq!(history[1].properties["guilt"], 0.8);

    // Verify maturity promotions
    hg.set_entity_maturity(
        &rask,
        MaturityLevel::Validated,
        "editor",
        Some("Confirmed from text".into()),
    )
    .unwrap();
    let updated = hg.get_entity(&rask).unwrap();
    assert_eq!(updated.maturity, MaturityLevel::Validated);
    let log = hg.get_validation_log(&rask).unwrap();
    assert_eq!(log.len(), 1);
}

#[test]
fn test_temporal_chain_three_situations() {
    let (hg, mut tree) = setup();

    let s1 = create_situation_at(&hg, &mut tree, 0, 5, NarrativeLevel::Scene, "Act 1");
    let s2 = create_situation_at(&hg, &mut tree, 10, 15, NarrativeLevel::Scene, "Act 2");
    let s3 = create_situation_at(&hg, &mut tree, 20, 25, NarrativeLevel::Scene, "Act 3");

    // Verify temporal ordering via interval tree
    use tensa::temporal::interval::relation_between;
    let sit1 = hg.get_situation(&s1).unwrap();
    let sit2 = hg.get_situation(&s2).unwrap();
    let sit3 = hg.get_situation(&s3).unwrap();

    assert_eq!(
        relation_between(&sit1.temporal, &sit2.temporal).unwrap(),
        AllenRelation::Before
    );
    assert_eq!(
        relation_between(&sit2.temporal, &sit3.temporal).unwrap(),
        AllenRelation::Before
    );

    // Point query at hour 12 should find s2
    let base = chrono::DateTime::parse_from_rfc3339("2025-01-01T12:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let results = tree.point_query(&base);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].situation_id, s2);
}

#[test]
fn test_causal_chain_traversal() {
    let (hg, mut tree) = setup();

    let s1 = create_situation_at(&hg, &mut tree, 0, 1, NarrativeLevel::Event, "Cause");
    let s2 = create_situation_at(&hg, &mut tree, 2, 3, NarrativeLevel::Event, "Effect 1");
    let s3 = create_situation_at(&hg, &mut tree, 4, 5, NarrativeLevel::Event, "Effect 2");
    let s4 = create_situation_at(&hg, &mut tree, 6, 7, NarrativeLevel::Event, "Final");

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
    hg.add_causal_link(CausalLink {
        from_situation: s3,
        to_situation: s4,
        mechanism: None,
        strength: 0.7,
        causal_type: CausalType::Enabling,
        maturity: MaturityLevel::Candidate,
    })
    .unwrap();

    // Full chain from s1
    let chain = hg.traverse_causal_chain(&s1, 10).unwrap();
    assert_eq!(chain.len(), 3);

    // Depth-limited
    let chain_2 = hg.traverse_causal_chain(&s1, 2).unwrap();
    assert_eq!(chain_2.len(), 2); // s1->s2 and s2->s3

    // Antecedents of s4
    let antecedents = hg.get_antecedents(&s4).unwrap();
    assert_eq!(antecedents.len(), 1);
    assert_eq!(antecedents[0].from_situation, s3);

    // Cycle detection
    let result = hg.add_causal_link(CausalLink {
        from_situation: s4,
        to_situation: s1,
        mechanism: None,
        strength: 0.5,
        causal_type: CausalType::Contributing,
        maturity: MaturityLevel::Candidate,
    });
    assert!(result.is_err()); // Would create cycle
}

#[test]
fn test_maturity_filter_hides_candidates() {
    let (hg, _tree) = setup();

    let id1 = create_entity(&hg, "Confirmed", EntityType::Actor, 0.9);
    let _id2 = create_entity(&hg, "Unconfirmed", EntityType::Actor, 0.3);

    hg.set_entity_maturity(&id1, MaturityLevel::Validated, "admin", None)
        .unwrap();

    let validated = hg
        .list_entities_by_maturity(MaturityLevel::Validated)
        .unwrap();
    assert_eq!(validated.len(), 1);
    assert_eq!(validated[0].properties["name"], "Confirmed");
}

#[test]
fn test_state_versioning_through_situations() {
    let (hg, mut tree) = setup();

    let character = create_entity(&hg, "Hero", EntityType::Actor, 0.9);
    let s1 = create_situation_at(&hg, &mut tree, 0, 5, NarrativeLevel::Scene, "Beginning");
    let s2 = create_situation_at(&hg, &mut tree, 10, 15, NarrativeLevel::Scene, "Middle");
    let s3 = create_situation_at(&hg, &mut tree, 20, 25, NarrativeLevel::Scene, "End");

    hg.create_state_version(StateVersion {
        entity_id: character,
        situation_id: s1,
        properties: serde_json::json!({"health": 100, "gold": 0}),
        beliefs: None,
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();
    hg.create_state_version(StateVersion {
        entity_id: character,
        situation_id: s2,
        properties: serde_json::json!({"health": 50, "gold": 100}),
        beliefs: None,
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();
    hg.create_state_version(StateVersion {
        entity_id: character,
        situation_id: s3,
        properties: serde_json::json!({"health": 100, "gold": 500}),
        beliefs: Some(serde_json::json!({"victory": true})),
        embedding: None,
        timestamp: Utc::now(),
    })
    .unwrap();

    let history = hg.get_state_history(&character).unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].properties["health"], 100);
    assert_eq!(history[1].properties["health"], 50);
    assert_eq!(history[2].properties["gold"], 500);

    let mid_state = hg.get_state_at_situation(&character, &s2).unwrap();
    assert_eq!(mid_state.properties["gold"], 100);
}

#[test]
fn test_tensaql_query_with_participation() {
    let (hg, tree) = setup();

    let alice = create_entity(&hg, "Alice", EntityType::Actor, 0.9);
    let bob = create_entity(&hg, "Bob", EntityType::Actor, 0.8);

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
        raw_content: vec![ContentBlock::text("Meeting")],
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
    let sid = hg.create_situation(sit).unwrap();

    hg.add_participant(Participation {
        entity_id: alice,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    hg.add_participant(Participation {
        entity_id: bob,
        situation_id: sid,
        role: Role::Antagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();

    // Query entity-situation patterns
    let q =
        parser::parse_query("MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s").unwrap();
    let plan = planner::plan_query(&q).unwrap();
    let results = executor::execute(&plan, &hg, &tree).unwrap();
    assert_eq!(results.len(), 2); // Alice-Meeting, Bob-Meeting

    // Query with limit
    let q =
        parser::parse_query("MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s LIMIT 1")
            .unwrap();
    let plan = planner::plan_query(&q).unwrap();
    let results = executor::execute(&plan, &hg, &tree).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_interval_tree_persistence_and_rebuild() {
    let store = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store.clone());
    let mut tree = IntervalTree::new();

    // Create situations and add to tree
    for i in 0..5 {
        create_situation_at(
            &hg,
            &mut tree,
            i * 10,
            i * 10 + 5,
            NarrativeLevel::Scene,
            &format!("Scene {}", i),
        );
    }

    // Persist
    tree.save(store.as_ref()).unwrap();

    // Load
    let loaded = IntervalTree::load(store.as_ref()).unwrap();
    assert_eq!(loaded.len(), 5);

    // Rebuild from situations
    let rebuilt = IntervalTree::rebuild(store.as_ref()).unwrap();
    assert_eq!(rebuilt.len(), 5);
}

// ─── TensaQL parser tests for new INFER types ─────────────

#[test]
fn test_parse_infer_temporal_rules() {
    let q = parser::parse_query("INFER TEMPORAL_RULES FOR s:Situation RETURN s").unwrap();
    assert!(q.infer_clause.is_some());
    assert_eq!(
        q.infer_clause.unwrap().infer_type,
        parser::InferType::TemporalRules
    );
}

#[test]
fn test_parse_infer_mean_field() {
    let q = parser::parse_query("INFER MEAN_FIELD FOR s:Situation RETURN s").unwrap();
    assert!(q.infer_clause.is_some());
    assert_eq!(
        q.infer_clause.unwrap().infer_type,
        parser::InferType::MeanField
    );
}

#[test]
fn test_parse_infer_psl() {
    let q = parser::parse_query("INFER PSL FOR e:Actor RETURN e").unwrap();
    assert!(q.infer_clause.is_some());
    assert_eq!(q.infer_clause.unwrap().infer_type, parser::InferType::Psl);
}

// ─── Entity CRUD lifecycle tests ──────────────────────────

#[test]
fn test_entity_update_and_delete() {
    let (hg, _tree) = setup();
    let eid = create_entity(&hg, "Alice", EntityType::Actor, 0.9);

    // Update via closure
    hg.update_entity(&eid, |e| {
        e.properties = serde_json::json!({"name": "Alice Updated", "age": 30});
        e.confidence = 0.95;
    })
    .unwrap();

    let updated = hg.get_entity(&eid).unwrap();
    assert_eq!(updated.properties["name"], "Alice Updated");
    assert!((updated.confidence - 0.95).abs() < 0.01);

    // Delete
    hg.delete_entity(&eid).unwrap();
    assert!(hg.get_entity(&eid).is_err());
}

#[test]
fn test_situation_delete() {
    let (hg, mut tree) = setup();
    let sid = create_situation_at(&hg, &mut tree, 0, 1, NarrativeLevel::Scene, "Deletable");
    assert!(hg.get_situation(&sid).is_ok());

    hg.delete_situation(&sid).unwrap();
    assert!(hg.get_situation(&sid).is_err());
}

#[test]
fn test_entity_list_by_type_and_narrative() {
    let (hg, _tree) = setup();

    // Create entities in different narratives
    let _a1 = create_entity(&hg, "Actor1", EntityType::Actor, 0.9);
    let _a2 = create_entity(&hg, "Actor2", EntityType::Actor, 0.8);
    let _l1 = create_entity(&hg, "Location1", EntityType::Location, 0.7);

    // List by type
    let actors = hg.list_entities_by_type(&EntityType::Actor).unwrap();
    assert_eq!(actors.len(), 2);

    let locations = hg.list_entities_by_type(&EntityType::Location).unwrap();
    assert_eq!(locations.len(), 1);
}

#[test]
fn test_entity_not_found_returns_error() {
    let (hg, _tree) = setup();
    let result = hg.get_entity(&Uuid::now_v7());
    assert!(result.is_err());
}

#[test]
fn test_situation_not_found_returns_error() {
    let (hg, _tree) = setup();
    let result = hg.get_situation(&Uuid::now_v7());
    assert!(result.is_err());
}

#[test]
fn test_participation_removal() {
    let (hg, mut tree) = setup();
    let eid = create_entity(&hg, "Removable", EntityType::Actor, 0.9);
    let sid = create_situation_at(&hg, &mut tree, 0, 1, NarrativeLevel::Scene, "Scene");

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

    let parts = hg.get_participants_for_situation(&sid).unwrap();
    assert_eq!(parts.len(), 1);

    hg.remove_participant(&eid, &sid, None).unwrap();
    let parts_after = hg.get_participants_for_situation(&sid).unwrap();
    assert!(parts_after.is_empty());
}

#[test]
fn test_narrative_registry_crud() {
    let store: Arc<dyn tensa::store::KVStore> = Arc::new(MemoryStore::new());
    let registry = tensa::narrative::registry::NarrativeRegistry::new(store.clone());

    let narrative = tensa::narrative::types::Narrative {
        id: "test-story".into(),
        title: "Test Story".into(),
        genre: Some("fiction".into()),
        tags: vec!["test".into()],
        source: None,
        project_id: None,
        description: Some("A test narrative".into()),
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
    registry.create(narrative).unwrap();

    let loaded = registry.get("test-story").unwrap();
    assert_eq!(loaded.title, "Test Story");

    let all = registry.list(None, None).unwrap();
    assert!(!all.is_empty());

    registry.delete("test-story").unwrap();
    assert!(registry.get("test-story").is_err());
}

#[test]
fn test_query_aggregate_count() {
    let (hg, tree) = setup();
    for i in 0..5 {
        create_entity(&hg, &format!("Actor{}", i), EntityType::Actor, 0.8);
    }

    let q = parser::parse_query("MATCH (e:Actor) RETURN COUNT(*)").unwrap();
    let plan = planner::plan_query(&q).unwrap();
    let results = executor::execute(&plan, &hg, &tree).unwrap();
    assert_eq!(results.len(), 1);

    let count = results[0].get("COUNT(*)").and_then(|v| v.as_u64()).unwrap();
    assert_eq!(count, 5);
}
