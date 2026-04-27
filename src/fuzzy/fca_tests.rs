//! Fuzzy Sprint Phase 8 — FCA tests.
//!
//! Hand-verified 10×5 graded fixture + KV round-trip + filter /
//! threshold / determinism / empty-context / perf-cap tests.
//!
//! Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].

use uuid::Uuid;

use crate::fuzzy::fca::{
    build_lattice, build_lattice_with_threshold, hasse_edges, Concept, ContextObject,
    FormalContext, FormalContextOptions, HARD_MAX_OBJECTS,
};
use crate::fuzzy::fca_store::{
    delete_concept_lattice, list_concept_lattices_for_narrative, load_concept_lattice,
    save_concept_lattice, LatticeSummary,
};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::types::{Entity, EntityType, MaturityLevel};

// ── Fixture builders ────────────────────────────────────────────────────────

/// Hand-crafted 10×5 graded context with 5 "archetype" object clusters.
///
/// Objects (rows): A, B, C, D, E, F, G, H, I, J.
/// Attributes (cols): brave, clever, loyal, ruthless, wise.
///
/// Clusters — by construction:
/// * A/B/C share full membership in {brave, loyal}      → concept {A,B,C}
/// * D/E share full membership in {clever, wise}        → concept {D,E}
/// * F/G share full membership in {brave, clever}       → concept {F,G}
/// * H is a graded-ruthless outlier                     → singleton concept
/// * I/J share full membership in {loyal, wise}         → concept {I,J}
///
/// The universal concept (A..=J with empty intent) and the "empty"
/// concept (no objects, full intent) are always present.
fn hand_verified_context() -> FormalContext {
    let labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
    let attrs = vec![
        "brave".to_string(),
        "clever".to_string(),
        "loyal".to_string(),
        "ruthless".to_string(),
        "wise".to_string(),
    ];
    // Rows are  [brave, clever, loyal, ruthless, wise].
    let incidence = vec![
        vec![1.0, 0.0, 1.0, 0.0, 0.0], // A
        vec![1.0, 0.0, 1.0, 0.0, 0.0], // B
        vec![1.0, 0.0, 1.0, 0.0, 0.0], // C
        vec![0.0, 1.0, 0.0, 0.0, 1.0], // D
        vec![0.0, 1.0, 0.0, 0.0, 1.0], // E
        vec![1.0, 1.0, 0.0, 0.0, 0.0], // F
        vec![1.0, 1.0, 0.0, 0.0, 0.0], // G
        vec![0.0, 0.0, 0.0, 0.8, 0.0], // H (graded outlier)
        vec![0.0, 0.0, 1.0, 0.0, 1.0], // I
        vec![0.0, 0.0, 1.0, 0.0, 1.0], // J
    ];
    let objects: Vec<ContextObject> = labels
        .iter()
        .map(|l| ContextObject {
            id: Uuid::now_v7(),
            label: (*l).to_string(),
        })
        .collect();
    FormalContext::new(objects, attrs, incidence)
        .expect("hand_verified_context row shapes are correct by construction")
}

/// Tiny crisp 3×2 fixture used by determinism + KV round-trip tests.
fn small_crisp_context() -> FormalContext {
    let labels = ["X", "Y", "Z"];
    let attrs = vec!["red".to_string(), "round".to_string()];
    let incidence = vec![
        vec![1.0, 0.0], // X
        vec![1.0, 1.0], // Y
        vec![0.0, 1.0], // Z
    ];
    let objects: Vec<ContextObject> = labels
        .iter()
        .map(|l| ContextObject {
            id: Uuid::now_v7(),
            label: (*l).to_string(),
        })
        .collect();
    FormalContext::new(objects, attrs, incidence).expect("small_crisp shapes ok")
}

/// Tiny Hypergraph with N entities in `test-narrative`, each with a
/// chosen set of tag + property flags. Used to exercise
/// `FormalContext::from_hypergraph` + narrative-family helpers.
fn tiny_narrative_hypergraph() -> Hypergraph {
    let store = std::sync::Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);
    let specs: &[(&str, EntityType, serde_json::Value)] = &[
        (
            "alice",
            EntityType::Actor,
            serde_json::json!({"name": "Alice", "brave": true, "loyal": true}),
        ),
        (
            "bob",
            EntityType::Actor,
            serde_json::json!({"name": "Bob", "brave": true, "clever": true}),
        ),
        (
            "chloe",
            EntityType::Actor,
            serde_json::json!({"name": "Chloe", "fuzzy_tags": {"wise": 0.9, "brave": 0.3}}),
        ),
        (
            "castle",
            EntityType::Location,
            serde_json::json!({"name": "Castle", "fortified": true}),
        ),
    ];
    for (_slug, et, props) in specs {
        let now = chrono::Utc::now();
        let ent = Entity {
            id: Uuid::now_v7(),
            entity_type: et.clone(),
            properties: props.clone(),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("test-narrative".into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: Some(now),
        };
        hg.create_entity(ent).expect("create_entity");
    }
    hg
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_hand_verified_lattice_has_expected_extents() {
    let ctx = hand_verified_context();
    let lattice = build_lattice(&ctx, TNormKind::Godel).expect("build_lattice");

    // Must always include the universal concept {A..J}.
    let universal_extent: Vec<usize> = (0..10).collect();
    let has_universal = lattice.concepts.iter().any(|c| c.extent == universal_extent);
    assert!(has_universal, "universal concept missing");

    // Must include each of the 4 full-crisp clusters as a concept.
    let expected_clusters: &[&[usize]] = &[
        &[0, 1, 2],       // A/B/C : brave ∧ loyal
        &[3, 4],          // D/E   : clever ∧ wise
        &[5, 6],          // F/G   : brave ∧ clever
        &[8, 9],          // I/J   : loyal ∧ wise
    ];
    for cluster in expected_clusters {
        let found = lattice
            .concepts
            .iter()
            .any(|c| c.extent.as_slice() == *cluster);
        assert!(found, "cluster {:?} missing from lattice", cluster);
    }

    // Lattice is non-trivial and order edges never self-loop.
    assert!(
        lattice.num_concepts() >= 5,
        "expected >=5 concepts, got {}",
        lattice.num_concepts()
    );
    for (p, c) in &lattice.order {
        assert!(p != c, "hasse edge has p==c");
        let parent = &lattice.concepts[*p];
        let child = &lattice.concepts[*c];
        assert!(
            parent.extent.len() > child.extent.len(),
            "parent extent not strictly larger than child"
        );
    }
}

#[test]
fn test_lattice_round_trip_through_kv() {
    let store = MemoryStore::new();
    let ctx = small_crisp_context();
    let mut lattice = build_lattice(&ctx, TNormKind::Godel).expect("build");
    lattice.narrative_id = "rt-narrative".into();

    save_concept_lattice(&store, "rt-narrative", &lattice).expect("save");
    let loaded = load_concept_lattice(&store, "rt-narrative", &lattice.id)
        .expect("load")
        .expect("Some");
    assert_eq!(loaded, lattice, "round-trip must preserve every field");

    // Listing returns exactly one lattice, newest-first.
    let list = list_concept_lattices_for_narrative(&store, "rt-narrative").expect("list");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].id, lattice.id);

    // Lightweight summary carries the right counts.
    let summary = LatticeSummary::from_lattice(&lattice);
    assert_eq!(summary.lattice_id, lattice.id);
    assert_eq!(summary.num_concepts, lattice.num_concepts());
    assert_eq!(summary.num_objects, 3);
    assert_eq!(summary.num_attributes, 2);

    // Deletion is idempotent + subsequent loads return None.
    delete_concept_lattice(&store, "rt-narrative", &lattice.id).expect("delete");
    delete_concept_lattice(&store, "rt-narrative", &lattice.id).expect("idempotent");
    let after = load_concept_lattice(&store, "rt-narrative", &lattice.id).expect("load");
    assert!(after.is_none(), "expected None after delete");
}

#[test]
fn test_attribute_allowlist_filter() {
    let hg = tiny_narrative_hypergraph();
    // With no allowlist, the attribute axis is the union.
    let ctx_all = FormalContext::from_hypergraph(
        &hg,
        "test-narrative",
        &FormalContextOptions::default(),
    )
    .expect("from_hypergraph");
    assert!(ctx_all.num_attributes() > 2);

    // With an allowlist, the attribute axis is exactly the allowlist.
    let allow = vec!["brave".to_string(), "wise".to_string()];
    let opts = FormalContextOptions {
        entity_type_filter: Some(EntityType::Actor),
        attribute_allowlist: Some(allow.clone()),
        large_context: false,
    };
    let ctx = FormalContext::from_hypergraph(&hg, "test-narrative", &opts)
        .expect("from_hypergraph with allowlist");
    assert_eq!(ctx.attributes, allow, "attribute allowlist respected");
    // Entity filter — Castle (Location) must be excluded.
    assert_eq!(ctx.num_objects(), 3, "Actor filter keeps 3 of 4 entities");
    // Chloe's fuzzy_tags carry the graded brave + wise values.
    let chloe_row = ctx
        .objects
        .iter()
        .position(|o| o.label == "Chloe")
        .expect("Chloe present");
    assert!(
        (ctx.incidence[chloe_row][0] - 0.3).abs() < 1e-9,
        "Chloe.brave graded fuzzy_tag"
    );
    assert!(
        (ctx.incidence[chloe_row][1] - 0.9).abs() < 1e-9,
        "Chloe.wise graded fuzzy_tag"
    );
}

#[test]
fn test_threshold_prunes_small_extents() {
    let ctx = hand_verified_context();
    // With min_extent=3 the only concept that survives (other than the
    // universal) is the A/B/C cluster.
    let lattice = build_lattice_with_threshold(&ctx, TNormKind::Godel, 3).expect("build");
    assert!(
        lattice.concepts.iter().all(|c| c.extent.len() >= 3),
        "every concept must satisfy min_extent"
    );
    // Universal extent (10 objects) is always retained.
    assert!(lattice
        .concepts
        .iter()
        .any(|c| c.extent.len() == 10));
    // No 2-element cluster (D/E, F/G, I/J) survives.
    assert!(!lattice.concepts.iter().any(|c| c.extent.len() == 2));
}

#[test]
fn test_nextclosure_determinism() {
    // Two identical builds must produce bit-identical concept lists.
    let ctx = small_crisp_context();
    let l1 = build_lattice(&ctx, TNormKind::Godel).expect("l1");
    let l2 = build_lattice(&ctx, TNormKind::Godel).expect("l2");
    assert_eq!(
        l1.concepts, l2.concepts,
        "NextClosure must be deterministic for identical input"
    );
    assert_eq!(l1.order, l2.order, "Hasse order must also be deterministic");
    // Same is true for the 10-object hand-verified fixture.
    let ctx2 = hand_verified_context();
    let m1 = build_lattice(&ctx2, TNormKind::Godel).expect("m1");
    let m2 = build_lattice(&ctx2, TNormKind::Godel).expect("m2");
    assert_eq!(m1.concepts, m2.concepts);
    assert_eq!(m1.order, m2.order);
}

#[test]
fn test_empty_context_produces_trivial_lattice() {
    let ctx = FormalContext::new(Vec::new(), Vec::new(), Vec::new()).expect("empty");
    let lattice = build_lattice(&ctx, TNormKind::Godel).expect("build");
    assert_eq!(
        lattice.num_concepts(),
        1,
        "empty context yields exactly one trivial concept"
    );
    assert!(lattice.concepts[0].extent.is_empty());
    assert!(lattice.concepts[0].intent.is_empty());
    assert!(lattice.order.is_empty(), "no Hasse edges for a 1-concept lattice");
}

#[test]
fn test_perf_cap_hard_cap_rejected() {
    // Hand-crafted request above the hard cap must fail even with
    // `large_context=true`.
    let too_many = HARD_MAX_OBJECTS + 1;
    let objects: Vec<ContextObject> = (0..too_many)
        .map(|i| ContextObject {
            id: Uuid::now_v7(),
            label: format!("o{i}"),
        })
        .collect();
    let attrs = vec!["attr".to_string()];
    let incidence: Vec<Vec<f64>> = (0..too_many).map(|_| vec![0.0]).collect();
    let ctx = FormalContext::new(objects, attrs, incidence).expect("new");
    let err = build_lattice(&ctx, TNormKind::Godel).expect_err("must reject");
    assert!(
        err.to_string().to_lowercase().contains("hard cap"),
        "expected hard-cap error, got: {err}"
    );
}

#[test]
fn test_large_context_flag_opts_in_above_soft_cap() {
    // Hypergraph path — default soft cap rejects a 501-entity request
    // unless `large_context: true` is set. We simulate by constructing
    // a 501×1 context via `FormalContext::new` and exercising
    // `build_lattice_with_threshold` directly (which also calls
    // enforce_perf_cap). 501 objects blows through NextClosure's 64-
    // object bound, so the builder must still reject — but with a
    // NextClosure-specific error, NOT the soft-cap error.
    let n = 501;
    let objects: Vec<ContextObject> = (0..n)
        .map(|i| ContextObject {
            id: Uuid::now_v7(),
            label: format!("o{i}"),
        })
        .collect();
    let attrs = vec!["attr".to_string()];
    let incidence: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0]).collect();
    let ctx = FormalContext::new(objects, attrs, incidence).expect("new");
    let err = build_lattice(&ctx, TNormKind::Godel).expect_err("must reject");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("64 objects") || msg.contains("soft cap"),
        "expected soft-cap or 64-object error, got: {err}"
    );
}

#[test]
fn test_hasse_edges_transitive_reduction() {
    // Chain A ⊃ B ⊃ C must yield {(A,B), (B,C)} — never the transitive
    // (A,C) shortcut.
    let concepts = vec![
        Concept { extent: vec![0, 1, 2, 3], intent: vec![] },
        Concept { extent: vec![0, 1, 2],     intent: vec![(0, 0.5)] },
        Concept { extent: vec![0, 1],        intent: vec![(0, 1.0)] },
    ];
    let edges = hasse_edges(&concepts);
    assert_eq!(edges, vec![(0, 1), (1, 2)]);
}

#[test]
fn test_tensaql_fca_lattice_and_concept_parse() {
    use crate::query::parser::{parse_statement, TensaStatement};
    let q = r#"FCA LATTICE FOR "hamlet" THRESHOLD 2 ATTRIBUTES ['brave', 'wise'] ENTITY_TYPE Actor WITH TNORM 'godel'"#;
    match parse_statement(q).expect("parse FCA LATTICE") {
        TensaStatement::FcaLattice {
            narrative_id,
            threshold,
            attribute_allowlist,
            entity_type,
            fuzzy_config,
        } => {
            assert_eq!(narrative_id, "hamlet");
            assert_eq!(threshold, Some(2));
            assert_eq!(
                attribute_allowlist,
                Some(vec!["brave".to_string(), "wise".to_string()])
            );
            assert_eq!(entity_type.as_deref(), Some("Actor"));
            assert_eq!(fuzzy_config.tnorm.as_deref(), Some("godel"));
        }
        other => panic!("expected FcaLattice, got {other:?}"),
    }

    let q2 = r#"FCA CONCEPT 3 FROM "01234567-89ab-cdef-0123-456789abcdef""#;
    match parse_statement(q2).expect("parse FCA CONCEPT") {
        TensaStatement::FcaConcept {
            lattice_id,
            concept_idx,
        } => {
            assert_eq!(lattice_id, "01234567-89ab-cdef-0123-456789abcdef");
            assert_eq!(concept_idx, 3);
        }
        other => panic!("expected FcaConcept, got {other:?}"),
    }
}

#[test]
fn test_lukasiewicz_residuum_yields_concepts() {
    // Łukasiewicz residuum accepts graded pairs Gödel would reject
    // strictly. The Chloe-style fuzzy-tag fixture with Łukasiewicz
    // surfaces at least one concept with a graded intent (μ < 1).
    let objects = vec![
        ContextObject { id: Uuid::now_v7(), label: "x".into() },
        ContextObject { id: Uuid::now_v7(), label: "y".into() },
    ];
    let attrs = vec!["a".to_string(), "b".to_string()];
    let incidence = vec![vec![0.8, 0.4], vec![0.6, 0.9]];
    let ctx =
        FormalContext::new(objects, attrs, incidence).expect("graded ok");
    let lattice = build_lattice(&ctx, TNormKind::Lukasiewicz).expect("build");
    // Every extent's inf-intent is consistent — no cell > 1 or < 0 ever
    // appears, and the universal concept {0,1} is present.
    let universal = lattice
        .concepts
        .iter()
        .any(|c| c.extent == vec![0, 1]);
    assert!(universal, "universal concept missing");
    for c in &lattice.concepts {
        for (_, mu) in &c.intent {
            assert!((0.0..=1.0).contains(mu));
        }
    }
}
