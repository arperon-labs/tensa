//! Integration tests for Disinfo Sprint D1: dual fingerprint foundation.
//!
//! End-to-end coverage:
//! - Create an actor + several situations with realistic posting cadence,
//!   hashtags, and source attributions.
//! - Compute behavioral fingerprint via the public API surface
//!   (`compute_behavioral_fingerprint` + persistence) and assert non-NaN axes
//!   match expectations.
//! - Compute disinfo fingerprint for the narrative and verify
//!   source_diversity wires through.
//! - Compare two behavioral fingerprints (identical + different) and assert
//!   the same-source verdict + p-value behavior.
//! - Verify the InferenceJobType engines round-trip through job execution.

#![cfg(feature = "disinfo")]

use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use tensa::disinfo::comparison::{compare_fingerprints, ComparisonKind, ComparisonTask};
use tensa::disinfo::engines::{BehavioralFingerprintEngine, DisinfoFingerprintEngine};
use tensa::disinfo::fingerprints::{
    compute_behavioral_fingerprint, compute_disinfo_fingerprint, load_behavioral_fingerprint,
    load_disinfo_fingerprint, store_behavioral_fingerprint, store_disinfo_fingerprint,
    BehavioralAxis, DisinfoAxis,
};
use tensa::hypergraph::Hypergraph;
use tensa::inference::types::InferenceJob;
use tensa::inference::InferenceEngine;
use tensa::store::memory::MemoryStore;
use tensa::types::*;
use uuid::Uuid;

fn setup() -> Hypergraph {
    let store = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

fn make_actor(hg: &Hypergraph, name: &str, narrative: &str, platform: &str) -> Uuid {
    let entity = Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({
            "name": name,
            "platform": platform,
            "platforms": [platform],
        }),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some(narrative.to_string()),
        created_at: Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };
    hg.create_entity(entity).unwrap()
}

fn add_post(hg: &Hypergraph, actor: Uuid, narrative: &str, hour_offset: i64, text: &str) -> Uuid {
    let base = Utc.with_ymd_and_hms(2026, 4, 1, 12, 0, 0).unwrap();
    let sit = Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: Some(base + Duration::hours(hour_offset)),
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
        raw_content: vec![ContentBlock::text(text)],
        narrative_level: NarrativeLevel::Event,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(narrative.to_string()),
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
        entity_id: actor,
        situation_id: sid,
        role: Role::Protagonist,
        info_set: None,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
    sid
}

#[test]
fn end_to_end_behavioral_fingerprint() {
    let hg = setup();
    let actor = make_actor(&hg, "Alice", "campaign-a", "twitter");

    // Hourly posts with consistent #news hashtag — should yield high cadence
    // regularity and high hashtag concentration.
    for i in 0..12i64 {
        add_post(
            &hg,
            actor,
            "campaign-a",
            i * 4, // every 4 hours
            &format!("Post {i} #news"),
        );
    }

    let fp = compute_behavioral_fingerprint(&hg, &actor).expect("compute should succeed");
    store_behavioral_fingerprint(&hg, &fp).expect("persist should succeed");

    assert_eq!(fp.actor_id, actor);
    assert_eq!(fp.sample_size, 12);

    // Cadence regularity: 4-hour intervals are perfectly regular.
    let cadence = fp.axis(BehavioralAxis::PostingCadenceRegularity);
    assert!(cadence > 0.5, "regularity should be high, got {cadence}");

    // Single hashtag #news → Herfindahl = 1.
    let hash_h = fp.axis(BehavioralAxis::HashtagConcentration);
    assert!(
        (hash_h - 1.0).abs() < 1e-6,
        "single-hashtag → 1.0, got {hash_h}"
    );

    // No reposts → originality 1.
    let originality = fp.axis(BehavioralAxis::ContentOriginality);
    assert!((originality - 1.0).abs() < 1e-6);

    // Platform diversity: only one platform → > 0 but small (tanh(1/4)).
    let div = fp.axis(BehavioralAxis::PlatformDiversity);
    assert!(div > 0.0 && div < 0.5);

    // Future-sprint axes remain NaN.
    assert!(fp.axis(BehavioralAxis::TemporalCoordination).is_nan());
    assert!(fp.axis(BehavioralAxis::NetworkInsularity).is_nan());

    // Round-trip through KV.
    let loaded = load_behavioral_fingerprint(&hg, &actor)
        .expect("load")
        .expect("present");
    assert_eq!(loaded.actor_id, actor);
    assert!((loaded.axis(BehavioralAxis::HashtagConcentration) - 1.0).abs() < 1e-6);
    assert!(loaded.axis(BehavioralAxis::TemporalCoordination).is_nan());
}

#[test]
fn end_to_end_disinfo_fingerprint_partial() {
    let hg = setup();
    let actor = make_actor(&hg, "Alice", "campaign-a", "twitter");
    for i in 0..3 {
        add_post(&hg, actor, "campaign-a", i * 6, &format!("Post {i}"));
    }

    let fp = compute_disinfo_fingerprint(&hg, "campaign-a").expect("compute");
    store_disinfo_fingerprint(&hg, &fp).expect("persist");
    assert_eq!(fp.narrative_id, "campaign-a");
    assert_eq!(fp.sample_size, 3);

    // Sprint D1 doesn't yet wire SourceDiversity unless attributions exist;
    // with no attributions the axis stays NaN.
    assert!(fp.axis(DisinfoAxis::SourceDiversity).is_nan());

    // All other axes are NaN until later sprints. Sample-size > 0 should
    // round-trip though.
    let loaded = load_disinfo_fingerprint(&hg, "campaign-a")
        .unwrap()
        .unwrap();
    assert_eq!(loaded.sample_size, 3);
}

#[test]
fn behavioral_comparison_identical_then_different() {
    let hg = setup();
    let alice = make_actor(&hg, "Alice", "campaign-a", "twitter");
    let bob = make_actor(&hg, "Bob", "campaign-a", "telegram");

    // Alice: 3 posts with regular cadence + #news.
    for i in 0..6 {
        add_post(&hg, alice, "campaign-a", i * 4, &format!("Post {i} #news"));
    }
    // Bob: 6 posts with chaotic cadence + #different hashtags.
    let bob_hours = [0, 1, 100, 105, 200, 210];
    for (i, h) in bob_hours.iter().enumerate() {
        add_post(&hg, bob, "campaign-a", *h, &format!("Post {i} #other{i}"));
    }

    let alice_fp = compute_behavioral_fingerprint(&hg, &alice).unwrap();
    let bob_fp = compute_behavioral_fingerprint(&hg, &bob).unwrap();

    let alice_value = serde_json::to_value(&alice_fp).unwrap();
    let bob_value = serde_json::to_value(&bob_fp).unwrap();

    // Identical comparison: composite distance ~ 0, same-source verdict true.
    let identical = compare_fingerprints(
        ComparisonKind::Behavioral,
        ComparisonTask::Cib,
        &alice_value,
        &alice_value,
    )
    .unwrap();
    assert!(identical.composite_distance < 1e-9);
    assert!(identical.same_source_verdict);

    // Different actors: composite > 0, comparable_axes > 0.
    let different = compare_fingerprints(
        ComparisonKind::Behavioral,
        ComparisonTask::Cib,
        &alice_value,
        &bob_value,
    )
    .unwrap();
    assert!(different.composite_distance > 0.0);
    assert!(different.comparable_axes >= 3);
    // Confidence interval brackets the composite.
    let (lo, hi) = different.confidence_interval;
    assert!(lo <= different.composite_distance + 1e-9);
    assert!(hi + 1e-9 >= different.composite_distance);
}

#[test]
fn engine_round_trip_for_behavioral_fingerprint_job() {
    let hg = setup();
    let actor = make_actor(&hg, "Alice", "campaign-a", "twitter");
    for i in 0..4 {
        add_post(&hg, actor, "campaign-a", i * 6, &format!("Post {i} #news"));
    }

    let job = InferenceJob {
        id: "test-job-1".into(),
        job_type: InferenceJobType::BehavioralFingerprint,
        target_id: actor,
        parameters: serde_json::json!({"actor_id": actor.to_string()}),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let result = BehavioralFingerprintEngine.execute(&job, &hg).unwrap();
    assert_eq!(result.job_type, InferenceJobType::BehavioralFingerprint);
    assert_eq!(result.target_id, actor);
    assert_eq!(result.status, JobStatus::Completed);
    assert!(result.result.get("axes").is_some());
}

#[test]
fn engine_round_trip_for_disinfo_fingerprint_job() {
    let hg = setup();
    let actor = make_actor(&hg, "Alice", "campaign-a", "twitter");
    for i in 0..3 {
        add_post(&hg, actor, "campaign-a", i * 4, &format!("Post {i}"));
    }
    let job = InferenceJob {
        id: "test-job-2".into(),
        job_type: InferenceJobType::DisinfoFingerprint,
        target_id: Uuid::nil(),
        parameters: serde_json::json!({"narrative_id": "campaign-a"}),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let result = DisinfoFingerprintEngine.execute(&job, &hg).unwrap();
    assert_eq!(result.job_type, InferenceJobType::DisinfoFingerprint);
    assert_eq!(result.status, JobStatus::Completed);
}

#[test]
fn tensaql_grammar_parses_disinfo_infer_variants() {
    use tensa::query::parser::{parse_query, InferType};

    let q = "INFER BEHAVIORAL_FINGERPRINT FOR e:Actor RETURN e";
    let parsed = parse_query(q).expect("BEHAVIORAL_FINGERPRINT must parse");
    let infer_clause = parsed.infer_clause.expect("infer clause");
    assert_eq!(infer_clause.infer_type, InferType::BehavioralFingerprint);

    let q = "INFER DISINFO_FINGERPRINT FOR n:Narrative RETURN n";
    let parsed = parse_query(q).expect("DISINFO_FINGERPRINT must parse");
    let infer_clause = parsed.infer_clause.expect("infer clause");
    assert_eq!(infer_clause.infer_type, InferType::DisinfoFingerprint);
}
