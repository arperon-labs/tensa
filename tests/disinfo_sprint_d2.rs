//! Integration tests for Disinfo Sprint D2: spread dynamics extension.
//!
//! Coverage:
//! - SMIR contagion runs end-to-end on a multi-platform narrative and emits
//!   per-platform R₀ + jump detections + velocity-monitor alerts.
//! - Counterfactual `RemoveTopAmplifiers` projection lowers R₀ (or holds it
//!   when nothing can be removed).
//! - DisinformationFingerprint axes #1, #2, #11 light up after SMIR + jumps
//!   + an alert have been recorded.
//! - TensaQL parser accepts the new INFER variants.
//! - Both inference engines round-trip a job through the worker layer.

#![cfg(feature = "disinfo")]

use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use tensa::analysis::contagion::{
    detect_cross_platform_jumps, list_cross_platform_jumps, load_smir_result, run_smir_contagion,
    situation_platform, PlatformBeta,
};
use tensa::analysis::spread_intervention::{simulate_intervention, Intervention};
use tensa::analysis::velocity_monitor::{BaselineSource, VelocityMonitor, ANOMALY_Z_THRESHOLD};
use tensa::disinfo::engines::{SpreadInterventionEngine, SpreadVelocityEngine};
use tensa::disinfo::fingerprints::{
    compute_disinfo_fingerprint, store_disinfo_fingerprint, DisinfoAxis,
};
use tensa::hypergraph::Hypergraph;
use tensa::inference::types::InferenceJob;
use tensa::inference::InferenceEngine;
use tensa::store::memory::MemoryStore;
use tensa::types::*;
use uuid::Uuid;

fn setup() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn make_actor(hg: &Hypergraph, name: &str, narrative: &str) -> Uuid {
    let entity = Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": name}),
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

/// Add a situation with a `SourceReference` whose `source_type` matches a
/// `Platform::from_str` token — that's what `situation_platform()` reads.
fn add_situation_on(
    hg: &Hypergraph,
    narrative: &str,
    platform_str: &str,
    hour_offset: i64,
    text: &str,
) -> Uuid {
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
        raw_content: vec![ContentBlock {
            content_type: ContentType::Text,
            content: text.into(),
            source: Some(SourceReference {
                source_type: platform_str.into(),
                source_id: None,
                description: None,
                timestamp: Utc::now(),
                registered_source: None,
            }),
        }],
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
    hg.create_situation(sit).unwrap()
}

fn make_info(about: Uuid, fact: &str, knows: bool, learns: bool, reveals: bool) -> Option<InfoSet> {
    let kf = KnowledgeFact {
        about_entity: about,
        fact: fact.into(),
        confidence: 1.0,
    };
    Some(InfoSet {
        knows_before: if knows { vec![kf.clone()] } else { vec![] },
        learns: if learns { vec![kf.clone()] } else { vec![] },
        reveals: if reveals { vec![kf] } else { vec![] },
        beliefs_about_others: vec![],
    })
}

fn link_with_info(hg: &Hypergraph, entity: Uuid, sit: Uuid, info: Option<InfoSet>) {
    hg.add_participant(Participation {
        entity_id: entity,
        situation_id: sit,
        role: Role::Protagonist,
        info_set: info,
        action: None,
        payoff: None,
        seq: 0,
    })
    .unwrap();
}

#[test]
fn smir_runs_end_to_end_with_per_platform_r0() {
    let hg = setup();
    let n = "campaign-d2";
    let about = Uuid::now_v7();
    let a = make_actor(&hg, "Alice", n);
    let b = make_actor(&hg, "Bob", n);
    let c = make_actor(&hg, "Carol", n);

    // Twitter post: Alice reveals, Bob learns.
    let s1 = add_situation_on(&hg, n, "twitter", 0, "fact reveal");
    link_with_info(&hg, a, s1, make_info(about, "secret", true, false, true));
    link_with_info(&hg, b, s1, make_info(about, "secret", false, true, false));

    // Telegram post: Bob reveals, Carol learns. Cross-platform jump.
    let s2 = add_situation_on(&hg, n, "telegram", 4, "rebroadcast");
    link_with_info(&hg, b, s2, make_info(about, "secret", true, false, true));
    link_with_info(&hg, c, s2, make_info(about, "secret", false, true, false));

    let result = run_smir_contagion(&hg, n, "secret", about, &[]).unwrap();
    assert!(
        result.r0_overall >= 0.0,
        "R₀ should be a real number, got {}",
        result.r0_overall
    );
    assert!(
        result.r0_by_platform.contains_key("twitter")
            || result.r0_by_platform.contains_key("telegram"),
        "expected at least one of twitter/telegram in r0_by_platform: {:?}",
        result.r0_by_platform
    );
    // SMIR persisted — load_smir_result returns it.
    let loaded = load_smir_result(&hg, n).unwrap().expect("persisted");
    assert_eq!(loaded.fact, "secret");

    // Cross-platform jump detected.
    let jumps = detect_cross_platform_jumps(&hg, n, "secret", about).unwrap();
    assert!(
        !jumps.is_empty(),
        "expected at least one twitter→telegram jump"
    );
    let stored_jumps = list_cross_platform_jumps(&hg, n).unwrap();
    assert_eq!(stored_jumps.len(), jumps.len());
}

#[test]
fn situation_platform_reads_from_source_type() {
    let hg = setup();
    let n = "x";
    let s = add_situation_on(&hg, n, "twitter", 0, "tweet");
    let situation = hg.get_situation(&s).unwrap();
    let p = situation_platform(&situation);
    assert_eq!(p, Platform::Twitter);
}

#[test]
fn platform_beta_defaults_align_with_spec() {
    // Platform-specific β tuning per spec §2.1: Twitter highest, Web/RSS lowest.
    let twitter = PlatformBeta::default_for(&Platform::Twitter);
    let web = PlatformBeta::default_for(&Platform::Web);
    let unknown = PlatformBeta::default_for(&Platform::Other("threads".into()));
    assert!(twitter > web);
    assert!(unknown > 0.0);
}

#[test]
fn velocity_monitor_fires_on_synthetic_baseline() {
    let hg = setup();
    let monitor = VelocityMonitor::new(&hg);
    // Twitter synthetic mean ≈ 1.6, std ≈ 0.4 → 5.0 well past 2σ.
    let alert = monitor
        .check_anomaly("narr-z", &Platform::Twitter, "political", 5.0)
        .unwrap();
    let alert = alert.expect("expected 2σ alert");
    assert!(alert.z_score.abs() >= ANOMALY_Z_THRESHOLD);
    assert_eq!(alert.baseline_source, BaselineSource::Synthetic);
}

#[test]
fn intervention_remove_top_amplifiers_reduces_r0() {
    let hg = setup();
    let n = "intervention-d2";
    let about = Uuid::now_v7();
    let a = make_actor(&hg, "Alice", n);
    let b = make_actor(&hg, "Bob", n);
    let c = make_actor(&hg, "Carol", n);

    let s1 = add_situation_on(&hg, n, "twitter", 0, "fact reveal");
    link_with_info(&hg, a, s1, make_info(about, "secret", true, false, true));
    link_with_info(&hg, b, s1, make_info(about, "secret", false, true, false));
    link_with_info(&hg, c, s1, make_info(about, "secret", false, true, false));

    // Force baseline persistence by running SMIR first.
    let baseline = run_smir_contagion(&hg, n, "secret", about, &[]).unwrap();

    let projection = simulate_intervention(
        &hg,
        n,
        "secret",
        about,
        Intervention::RemoveTopAmplifiers { n: 1 },
        &[],
    )
    .unwrap();

    assert_eq!(projection.baseline_r0, baseline.r0_overall);
    assert!(projection.projected_r0 <= projection.baseline_r0);
}

#[test]
fn disinfo_fingerprint_wires_d2_axes_after_smir_and_alert() {
    let hg = setup();
    let n = "fp-d2";
    let about = Uuid::now_v7();
    let a = make_actor(&hg, "Alice", n);
    let b = make_actor(&hg, "Bob", n);

    let s1 = add_situation_on(&hg, n, "twitter", 0, "post");
    link_with_info(&hg, a, s1, make_info(about, "claim", true, false, true));
    link_with_info(&hg, b, s1, make_info(about, "claim", false, true, false));
    let s2 = add_situation_on(&hg, n, "telegram", 6, "rebroadcast");
    link_with_info(&hg, b, s2, make_info(about, "claim", true, false, true));

    // Generate the upstream signals.
    let _ = run_smir_contagion(&hg, n, "claim", about, &[]).unwrap();
    let _ = detect_cross_platform_jumps(&hg, n, "claim", about).unwrap();
    // Force a velocity alert so axis #11 lights up.
    let monitor = VelocityMonitor::new(&hg);
    let _ = monitor
        .check_anomaly(n, &Platform::Twitter, "default", 5.5)
        .unwrap();

    let fp = compute_disinfo_fingerprint(&hg, n).unwrap();
    store_disinfo_fingerprint(&hg, &fp).unwrap();

    assert!(
        !fp.axis(DisinfoAxis::ViralityVelocity).is_nan(),
        "virality_velocity should be wired after SMIR runs"
    );
    assert!(
        !fp.axis(DisinfoAxis::CrossPlatformJumpRate).is_nan(),
        "cross_platform_jump_rate should be wired after detect_cross_platform_jumps"
    );
    assert!(
        !fp.axis(DisinfoAxis::TemporalAnomaly).is_nan(),
        "temporal_anomaly should be wired after a velocity alert"
    );
}

#[test]
fn tensaql_parses_d2_infer_variants() {
    use tensa::query::parser::{parse_query, InferType};

    let q = "INFER SPREAD_VELOCITY FOR n:Narrative RETURN n";
    let parsed = parse_query(q).unwrap();
    let infer = parsed.infer_clause.unwrap();
    assert_eq!(infer.infer_type, InferType::SpreadVelocity);

    let q = "INFER SPREAD_INTERVENTION FOR n:Narrative RETURN n";
    let parsed = parse_query(q).unwrap();
    let infer = parsed.infer_clause.unwrap();
    assert_eq!(infer.infer_type, InferType::SpreadIntervention);
}

#[test]
fn spread_velocity_engine_round_trips() {
    let hg = setup();
    let n = "engine-d2";
    let about = Uuid::now_v7();
    let a = make_actor(&hg, "Alice", n);
    let b = make_actor(&hg, "Bob", n);
    let s1 = add_situation_on(&hg, n, "twitter", 0, "post");
    link_with_info(&hg, a, s1, make_info(about, "fact", true, false, true));
    link_with_info(&hg, b, s1, make_info(about, "fact", false, true, false));

    let job = InferenceJob {
        id: "j1".into(),
        job_type: InferenceJobType::SpreadVelocity,
        target_id: Uuid::nil(),
        parameters: serde_json::json!({
            "narrative_id": n,
            "fact": "fact",
            "about_entity": about.to_string(),
            "narrative_kind": "default",
        }),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let result = SpreadVelocityEngine.execute(&job, &hg).unwrap();
    assert_eq!(result.job_type, InferenceJobType::SpreadVelocity);
    assert_eq!(result.status, JobStatus::Completed);
    assert!(result.result.get("smir").is_some());
    assert!(result.result.get("alerts").is_some());
}

#[test]
fn spread_intervention_engine_round_trips() {
    let hg = setup();
    let n = "engine-d2-int";
    let about = Uuid::now_v7();
    let a = make_actor(&hg, "Alice", n);
    let b = make_actor(&hg, "Bob", n);
    let s = add_situation_on(&hg, n, "twitter", 0, "post");
    link_with_info(&hg, a, s, make_info(about, "fact", true, false, true));
    link_with_info(&hg, b, s, make_info(about, "fact", false, true, false));

    let job = InferenceJob {
        id: "j2".into(),
        job_type: InferenceJobType::SpreadIntervention,
        target_id: Uuid::nil(),
        parameters: serde_json::json!({
            "narrative_id": n,
            "fact": "fact",
            "about_entity": about.to_string(),
            "intervention": {"type": "RemoveTopAmplifiers", "n": 1},
        }),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let result = SpreadInterventionEngine.execute(&job, &hg).unwrap();
    assert_eq!(result.job_type, InferenceJobType::SpreadIntervention);
    assert_eq!(result.status, JobStatus::Completed);
    assert!(result.result.get("baseline_r0").is_some());
}
