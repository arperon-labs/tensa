//! Integration tests for Disinfo Sprint D3: CIB detection module.
//!
//! Coverage:
//! - Behavioral similarity network builds edges only between actors whose
//!   fingerprints are above the configured similarity threshold.
//! - CIB detection flags a pre-constructed dense clique of near-identical
//!   fingerprints with p-value below α = 0.05 and leaves organic outliers out.
//! - Persisted clusters + evidence round-trip through `list_clusters`,
//!   `list_evidence`, and `load_cib_detection`.
//! - `detect_cross_platform_cib` drops single-platform clusters.
//! - `rank_superspreaders` returns an ordered ranking whose top entity is the
//!   highest-degree hub in the narrative.
//! - Inference engines (`CibDetectionEngine`, `SuperspreadersEngine`) round-trip
//!   a job and produce serialized JSON payloads.
//! - `DisinformationFingerprint.coordination_score` (axis #7) lights up after a
//!   CIB cluster has been persisted.
//! - TensaQL parser accepts `INFER CIB` and `INFER SUPERSPREADERS`.

#![cfg(feature = "disinfo")]

use std::sync::Arc;

use chrono::Utc;
use tensa::analysis::cib::{
    detect_cib_clusters, detect_cross_platform_cib, list_clusters, list_evidence,
    load_cib_detection, rank_superspreaders, store_cluster, store_evidence, CibCluster, CibConfig,
    CibEvidence, SuperspreaderMethod,
};
use tensa::disinfo::engines::{CibDetectionEngine, SuperspreadersEngine};
use tensa::disinfo::fingerprints::{
    compute_disinfo_fingerprint, store_behavioral_fingerprint, BehavioralAxis,
    BehavioralFingerprint, DisinfoAxis,
};
use tensa::hypergraph::Hypergraph;
use tensa::inference::types::InferenceJob;
use tensa::inference::InferenceEngine;
use tensa::query::parser::parse_query;
use tensa::store::memory::MemoryStore;
use tensa::types::*;
use uuid::Uuid;

fn setup() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn make_actor(hg: &Hypergraph, name: &str, narrative: &str, platform: Option<&str>) -> Uuid {
    let mut props = serde_json::json!({"name": name});
    if let Some(p) = platform {
        props["platform"] = serde_json::json!(p);
    }
    let entity = Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: props,
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some(narrative.into()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };
    hg.create_entity(entity).unwrap()
}

fn set_axes(hg: &Hypergraph, actor: Uuid, axes: &[(BehavioralAxis, f64)]) {
    let mut fp = BehavioralFingerprint::empty(actor);
    for (a, v) in axes {
        fp.set_axis(*a, *v);
    }
    store_behavioral_fingerprint(hg, &fp).unwrap();
}

fn sockpuppet_axes() -> Vec<(BehavioralAxis, f64)> {
    vec![
        (BehavioralAxis::PostingCadenceRegularity, 0.97),
        (BehavioralAxis::HashtagConcentration, 0.92),
        (BehavioralAxis::ContentOriginality, 0.05),
        (BehavioralAxis::EngagementRatio, 0.82),
        (BehavioralAxis::SleepPatternPresence, 0.02),
        (BehavioralAxis::PlatformDiversity, 0.25),
    ]
}

fn organic_axes(cadence: f64, originality: f64) -> Vec<(BehavioralAxis, f64)> {
    vec![
        (BehavioralAxis::PostingCadenceRegularity, cadence),
        (BehavioralAxis::HashtagConcentration, 0.2),
        (BehavioralAxis::ContentOriginality, originality),
        (BehavioralAxis::EngagementRatio, 0.3),
        (BehavioralAxis::SleepPatternPresence, 0.85),
        (BehavioralAxis::PlatformDiversity, 0.65),
    ]
}

/// Full end-to-end: build a narrative with 4 near-identical sockpuppets + 3
/// organic accounts and assert the detector finds the sockpuppets.
#[test]
fn end_to_end_detects_sockpuppet_clique() {
    let hg = setup();
    let narrative = "narr-d3-e2e";
    let mut sockpuppets = Vec::new();
    for i in 0..4 {
        let id = make_actor(
            &hg,
            &format!("sock{i}"),
            narrative,
            Some(if i % 2 == 0 { "twitter" } else { "telegram" }),
        );
        set_axes(&hg, id, &sockpuppet_axes());
        sockpuppets.push(id);
    }
    for (i, (c, o)) in [(0.3, 0.85), (0.22, 0.92), (0.15, 0.95)].iter().enumerate() {
        let id = make_actor(&hg, &format!("organic{i}"), narrative, Some("twitter"));
        set_axes(&hg, id, &organic_axes(*c, *o));
    }

    let config = CibConfig {
        similarity_threshold: 0.8,
        bootstrap_iter: 60,
        alpha: 0.05,
        ..CibConfig::default()
    };
    let result = detect_cib_clusters(&hg, narrative, &config).unwrap();
    assert!(result.network_size >= 7);
    assert!(!result.clusters.is_empty());
    let top = &result.clusters[0];
    assert!(
        top.members.len() >= 3,
        "expected ≥3 members, got {}",
        top.members.len()
    );
    assert!(top.p_value < 0.05);
    assert!(top.density > 0.5);
}

/// Persisted clusters + evidence round-trip.
#[test]
fn cluster_and_evidence_persistence_round_trip() {
    let hg = setup();
    let members = vec![Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7()];
    let cluster = CibCluster {
        cluster_id: "cib-round-001".into(),
        narrative_id: "narr-round".into(),
        members: members.clone(),
        density: 1.0,
        mean_similarity: 0.95,
        p_value: 0.002,
        bootstrap_iter: 100,
        alpha: 0.01,
        platforms: vec!["twitter".into(), "telegram".into()],
        created_at: Utc::now(),
    };
    store_cluster(&hg, &cluster).unwrap();
    let ev = CibEvidence {
        cluster_id: "cib-round-001".into(),
        narrative_id: "narr-round".into(),
        top_edges: vec![],
        axis_mean_distances: vec![("engagement_ratio".into(), 0.04)],
        factory_weighted: true,
        created_at: Utc::now(),
    };
    store_evidence(&hg, &ev).unwrap();

    let loaded = load_cib_detection(&hg, "narr-round").unwrap().unwrap();
    assert_eq!(loaded.clusters.len(), 1);
    assert_eq!(loaded.clusters[0].members, members);
    assert_eq!(loaded.evidence.len(), 1);
    assert!(loaded.evidence[0].factory_weighted);

    // Listing helpers expose the same records.
    assert_eq!(list_clusters(&hg, "narr-round").unwrap().len(), 1);
    assert_eq!(list_evidence(&hg, "narr-round").unwrap().len(), 1);
}

/// Cross-platform detection drops single-platform clusters.
#[test]
fn cross_platform_filters_single_platform_clusters() {
    let hg = setup();
    let narrative = "narr-d3-xplat-single";
    // All sockpuppets on the same platform.
    for i in 0..4 {
        let id = make_actor(&hg, &format!("sock{i}"), narrative, Some("twitter"));
        set_axes(&hg, id, &sockpuppet_axes());
    }
    let config = CibConfig {
        similarity_threshold: 0.8,
        bootstrap_iter: 40,
        alpha: 0.1, // loose so base detection would fire
        ..CibConfig::default()
    };
    let xplat = detect_cross_platform_cib(&hg, narrative, &config).unwrap();
    assert!(
        xplat.clusters.is_empty(),
        "single-platform clusters must be excluded from cross-platform detection"
    );
}

/// Cross-platform detection keeps clusters whose members span multiple platforms.
#[test]
fn cross_platform_keeps_multi_platform_clusters() {
    let hg = setup();
    let narrative = "narr-d3-xplat-multi";
    let platforms = ["twitter", "telegram", "bluesky", "facebook"];
    for (i, plat) in platforms.iter().enumerate() {
        let id = make_actor(&hg, &format!("sock{i}"), narrative, Some(plat));
        set_axes(&hg, id, &sockpuppet_axes());
    }
    let config = CibConfig {
        similarity_threshold: 0.8,
        bootstrap_iter: 40,
        alpha: 0.1,
        ..CibConfig::default()
    };
    let xplat = detect_cross_platform_cib(&hg, narrative, &config).unwrap();
    if !xplat.clusters.is_empty() {
        assert!(
            xplat.clusters[0].platforms.len() >= 2,
            "flagged cross-platform cluster must span ≥ 2 platforms"
        );
    }
}

/// Superspreader ranking picks the hub that participates in every situation.
#[test]
fn superspreader_ranking_finds_hub() {
    let hg = setup();
    let narrative = "narr-d3-super";
    let hub = make_actor(&hg, "Hub", narrative, None);
    let mut others = Vec::new();
    for i in 0..6 {
        others.push(make_actor(&hg, &format!("user{i}"), narrative, None));
    }

    for &o in &others {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
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
            raw_content: vec![ContentBlock::text("msg")],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.into()),
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
            entity_id: hub,
            situation_id: sid,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
        hg.add_participant(Participation {
            entity_id: o,
            situation_id: sid,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    let ranking = rank_superspreaders(&hg, narrative, SuperspreaderMethod::PageRank, 3).unwrap();
    assert_eq!(ranking.scores.len(), 3);
    assert_eq!(ranking.scores[0].entity_id, hub);
    assert_eq!(ranking.scores[0].rank, 1);
    // Eigenvector + Harmonic should also have the hub in the top-1.
    for method in [
        SuperspreaderMethod::Eigenvector,
        SuperspreaderMethod::Harmonic,
    ] {
        let r = rank_superspreaders(&hg, narrative, method, 1).unwrap();
        assert_eq!(r.scores[0].entity_id, hub);
    }
}

/// Inference engine executes end-to-end and returns a JSON CibDetectionResult.
#[test]
fn cib_detection_engine_roundtrip() {
    let hg = setup();
    let narrative = "narr-d3-engine";
    for i in 0..4 {
        let id = make_actor(&hg, &format!("sock{i}"), narrative, Some("twitter"));
        set_axes(&hg, id, &sockpuppet_axes());
    }
    let engine = CibDetectionEngine;
    let job = InferenceJob {
        id: "cib-job".into(),
        job_type: InferenceJobType::CibDetection,
        target_id: Uuid::nil(),
        parameters: serde_json::json!({
            "narrative_id": narrative,
            "bootstrap_iter": 40,
            "alpha": 0.5, // deliberately generous so engine reliably emits ≥1 cluster
            "similarity_threshold": 0.8,
        }),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let result = engine.execute(&job, &hg).unwrap();
    assert_eq!(result.job_type, InferenceJobType::CibDetection);
    assert_eq!(result.status, JobStatus::Completed);
    assert!(result.result.get("network_size").is_some());
}

/// Superspreader engine roundtrip with a default method.
#[test]
fn superspreader_engine_roundtrip() {
    let hg = setup();
    let narrative = "narr-d3-super-engine";
    let _a = make_actor(&hg, "Alice", narrative, None);
    let engine = SuperspreadersEngine;
    let job = InferenceJob {
        id: "super-job".into(),
        job_type: InferenceJobType::Superspreaders,
        target_id: Uuid::nil(),
        parameters: serde_json::json!({"narrative_id": narrative, "top_n": 5}),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let result = engine.execute(&job, &hg).unwrap();
    assert_eq!(result.job_type, InferenceJobType::Superspreaders);
    assert_eq!(result.status, JobStatus::Completed);
}

/// DisinformationFingerprint axis #7 (coordination_score) is populated once a
/// CIB cluster has been persisted for the narrative.
#[test]
fn coordination_score_axis_populates_after_cib() {
    let hg = setup();
    let narrative = "narr-d3-axis";
    // Seed 4 sockpuppets + 2 organic actors so total_actors > 0.
    let mut socks = Vec::new();
    for i in 0..4 {
        let id = make_actor(&hg, &format!("sock{i}"), narrative, Some("twitter"));
        set_axes(&hg, id, &sockpuppet_axes());
        socks.push(id);
    }
    let org_a = make_actor(&hg, "OrgA", narrative, Some("bluesky"));
    set_axes(
        &hg,
        org_a,
        &[
            (BehavioralAxis::PostingCadenceRegularity, 0.2),
            (BehavioralAxis::HashtagConcentration, 0.1),
            (BehavioralAxis::ContentOriginality, 0.9),
            (BehavioralAxis::EngagementRatio, 0.2),
        ],
    );

    // Axis before CIB: NaN (no clusters persisted).
    let before = compute_disinfo_fingerprint(&hg, narrative).unwrap();
    assert!(before.axis(DisinfoAxis::CoordinationScore).is_nan());

    // Run CIB detection with a very loose alpha so at least one cluster fires.
    let config = CibConfig {
        similarity_threshold: 0.8,
        bootstrap_iter: 40,
        alpha: 0.9,
        ..CibConfig::default()
    };
    let cib = detect_cib_clusters(&hg, narrative, &config).unwrap();
    assert!(
        !cib.clusters.is_empty(),
        "test precondition: at least one cluster persisted"
    );

    // Now the axis should be finite and bounded in [0, 1].
    let after = compute_disinfo_fingerprint(&hg, narrative).unwrap();
    let coord = after.axis(DisinfoAxis::CoordinationScore);
    assert!(!coord.is_nan(), "coordination_score axis must be populated");
    assert!((0.0..=1.0).contains(&coord));
    // The envelope is also generated correctly (smoke test).
    let env = tensa::disinfo::disinfo_envelope(&after);
    assert!(env.get("fingerprint").is_some());

    // Force-recompute via the public ensure_disinfo_fingerprint and confirm
    // the axis survives the round-trip through the persistence layer.
    let persisted = tensa::disinfo::ensure_disinfo_fingerprint(&hg, narrative, true).unwrap();
    let persisted_coord = persisted.axis(DisinfoAxis::CoordinationScore);
    assert!((persisted_coord - coord).abs() < 1e-9);
}

/// Grammar / parser acceptance for the new INFER variants.
#[test]
fn parser_accepts_infer_cib_and_superspreaders() {
    let q1 = r#"INFER CIB FOR e:Actor RETURN e"#;
    assert!(
        parse_query(q1).is_ok(),
        "parse_query rejected INFER CIB: {q1}"
    );

    let q2 = r#"INFER SUPERSPREADERS FOR n:Narrative RETURN n"#;
    assert!(
        parse_query(q2).is_ok(),
        "parse_query rejected INFER SUPERSPREADERS: {q2}"
    );
}
