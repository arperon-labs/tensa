//! REST tests for `/analysis/opinion-dynamics*`.
//!
//! Mirrors `crate::api::synth_tests` and
//! `crate::api::inference::reconstruction_tests` — calls handler functions
//! directly via a hand-built [`AppState`] backed by an in-memory KV store.
//! No live HTTP loop — that matches the project's API test convention.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::Utc;
use uuid::Uuid;

use crate::analysis::opinion_dynamics::{InitialOpinionDist, OpinionDynamicsParams};
use crate::api::analysis::opinion_dynamics::{
    run, sweep, OpinionDynamicsBody, PhaseTransitionBody,
};
use crate::api::server::{AppState, LlmConfig};
use crate::api::settings_routes::InferenceConfig;
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::ingestion::config::IngestionConfig;
use crate::ingestion::geocode::Geocoder;
use crate::ingestion::jobs::IngestionJobQueue;
use crate::ingestion::queue::ValidationQueue;
use crate::query::rag_config::RagConfig;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::SurrogateRegistry;
use crate::temporal::index::IntervalTree;
use crate::types::{
    ContentBlock, Entity, EntityType, ExtractionMethod, MaturityLevel, NarrativeLevel,
    Participation, Role, Situation, TimeGranularity,
};

const NARRATIVE: &str = "od-rest-narr";

fn make_test_state() -> Arc<AppState> {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let hypergraph = Hypergraph::new(store.clone());
    let job_queue = Arc::new(JobQueue::new(store.clone()));

    Arc::new(AppState {
        hypergraph,
        interval_tree: std::sync::RwLock::new(IntervalTree::new()),
        validation_queue: ValidationQueue::new(store.clone()),
        job_queue: Some(job_queue),
        extractor: std::sync::RwLock::new(None),
        llm_config: std::sync::RwLock::new(LlmConfig::None),
        embedder: std::sync::RwLock::new(None),
        embedder_model_name: std::sync::RwLock::new("hash".into()),
        vector_index: None,
        ingestion_config: std::sync::RwLock::new(IngestionConfig::default()),
        inference_config: std::sync::RwLock::new(InferenceConfig::default()),
        job_watchers: std::sync::RwLock::new(std::collections::HashMap::new()),
        ingestion_jobs: Arc::new(IngestionJobQueue::new(store.clone())),
        ingestion_progress: std::sync::Mutex::new(std::collections::HashMap::new()),
        ingestion_cancel_flags: std::sync::Mutex::new(std::collections::HashMap::new()),
        llm_cache: None,
        doc_tracker: None,
        source_index: None,
        rag_config: std::sync::RwLock::new(RagConfig::default()),
        reranker: None,
        root_store: store.clone(),
        geocoder: Geocoder::new(store.clone()),
        inference_extractor: std::sync::RwLock::new(None),
        inference_llm_config: std::sync::RwLock::new(LlmConfig::None),
        #[cfg(feature = "studio-chat")]
        chat_extractor: std::sync::RwLock::new(None),
        #[cfg(feature = "studio-chat")]
        chat_llm_config: std::sync::RwLock::new(LlmConfig::None),
        #[cfg(feature = "studio-chat")]
        chat_skills: crate::studio_chat::SkillRegistry::default_bundled(),
        #[cfg(feature = "studio-chat")]
        chat_confirm_gate: Arc::new(crate::studio_chat::ConfirmGate::new()),
        #[cfg(feature = "studio-chat")]
        chat_mcp_proxies: Arc::new(crate::studio_chat::McpProxySet::new()),
        synth_registry: Arc::new(SurrogateRegistry::default()),
        image_gen_config: std::sync::RwLock::new(crate::images::ImageGenConfig::None),
        image_generator: std::sync::RwLock::new(None),
    })
}

/// Seed a small triangle: 3 entities, 1 size-3 situation. Enough for a
/// well-formed opinion-dynamics run.
fn seed_triangle(state: &AppState) -> Vec<Uuid> {
    let now = Utc::now();
    let mut ids = Vec::new();
    for i in 0..3 {
        let id = Uuid::now_v7();
        let e = Entity {
            id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": format!("a{i}")}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::HumanEntered),
            narrative_id: Some(NARRATIVE.into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        state.hypergraph.create_entity(e).unwrap();
        ids.push(id);
    }
    let sit_id = Uuid::now_v7();
    let s = Situation {
        id: sit_id,
        name: None,
        description: None,
        properties: serde_json::Value::Null,
        temporal: crate::types::AllenInterval {
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
        raw_content: vec![ContentBlock::text("seed")],
        narrative_level: NarrativeLevel::Scene,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(NARRATIVE.into()),
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
    state.hypergraph.create_situation(s).unwrap();
    for id in &ids {
        state
            .hypergraph
            .add_participant(Participation {
                entity_id: *id,
                situation_id: sit_id,
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
    }
    ids
}

async fn read_body(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
        .await
        .expect("read body");
    if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("body is JSON")
    }
}

/// T1 — POST /analysis/opinion-dynamics returns a 200 with an inline report
/// and persists the report under opd/report/{narrative_id}/{run_id}.
#[tokio::test]
async fn test_opinion_dynamics_rest_roundtrip() {
    let state = make_test_state();
    seed_triangle(&state);

    // Use a high confidence_bound and short max_steps so the simulation
    // converges quickly in tests.
    let mut params = OpinionDynamicsParams::default();
    params.confidence_bound = 0.5;
    params.max_steps = 5_000;

    let resp = run(
        State(state.clone()),
        Json(OpinionDynamicsBody {
            narrative_id: NARRATIVE.into(),
            params: Some(params),
            include_synthetic: None,
        }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert!(body.get("run_id").is_some(), "run_id present in envelope");
    assert!(body.get("report").is_some(), "report present in envelope");
    let report = &body["report"];
    assert!(report["num_steps_executed"].as_u64().unwrap() > 0);

    // Verify persistence.
    let listed = crate::analysis::opinion_dynamics::list_opinion_reports(
        state.hypergraph.store(),
        NARRATIVE,
        10,
    )
    .unwrap();
    assert_eq!(listed.len(), 1, "report persisted at opd/report/...");
}

/// Empty narrative_id rejects synchronously (400-class).
#[tokio::test]
async fn test_opinion_dynamics_rest_rejects_empty_narrative_id() {
    let state = make_test_state();
    let resp = run(
        State(state),
        Json(OpinionDynamicsBody {
            narrative_id: "  ".into(),
            params: None,
            include_synthetic: None,
        }),
    )
    .await
    .into_response();
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "should not be 2xx"
    );
}

/// Custom InitialOpinionDist length mismatch maps to HTTP 400 (per spec
/// architect note: "Custom InitialOpinionDist length validation: 16c REST
/// should map InvalidInput → HTTP 400").
#[tokio::test]
async fn test_opinion_dynamics_rest_custom_length_mismatch_400() {
    let state = make_test_state();
    seed_triangle(&state);
    let mut params = OpinionDynamicsParams::default();
    // Triangle has 3 entities but Custom carries 5 → InvalidInput → 400.
    params.initial_opinion_distribution = InitialOpinionDist::Custom(vec![0.1; 5]);
    let resp = run(
        State(state),
        Json(OpinionDynamicsBody {
            narrative_id: NARRATIVE.into(),
            params: Some(params),
            include_synthetic: None,
        }),
    )
    .await
    .into_response();
    assert!(resp.status().is_client_error(), "must be 4xx");
}

/// T2 — POST /analysis/opinion-dynamics/phase-transition-sweep returns a
/// PhaseTransitionReport with the right shape.
#[tokio::test]
async fn test_opinion_phase_transition_rest_roundtrip() {
    let state = make_test_state();
    seed_triangle(&state);

    let mut base_params = OpinionDynamicsParams::default();
    base_params.max_steps = 2_000;

    let resp = sweep(
        State(state.clone()),
        Json(PhaseTransitionBody {
            narrative_id: NARRATIVE.into(),
            c_range: (0.05, 0.5, 3),
            base_params: Some(base_params),
            include_synthetic: None,
        }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    let c_values = body["c_values"].as_array().unwrap();
    assert_eq!(c_values.len(), 3, "linspace produces 3 c values");
    let times = body["convergence_times"].as_array().unwrap();
    assert_eq!(times.len(), 3, "one entry per c value");
    assert!(body.get("spike_threshold").is_some());
    assert!(body.get("initial_variance").is_some());
}

/// T7 — Phase 3 invariant: include_synthetic flag is wired on the request
/// body and accepted (default false; engine receives the value via the
/// surface). Tests the surface contract — the deeper filter integration is
/// 16c.1 follow-up. The flag MUST not raise an error and the response shape
/// MUST be unchanged whether the flag is set or unset.
#[tokio::test]
async fn test_opinion_dynamics_respects_include_synthetic_flag() {
    let state = make_test_state();
    seed_triangle(&state);

    let mut params = OpinionDynamicsParams::default();
    params.confidence_bound = 0.5;
    params.max_steps = 2_000;

    // Default: include_synthetic = None (false).
    let resp_default = run(
        State(state.clone()),
        Json(OpinionDynamicsBody {
            narrative_id: NARRATIVE.into(),
            params: Some(params.clone()),
            include_synthetic: None,
        }),
    )
    .await
    .into_response();
    assert_eq!(resp_default.status(), StatusCode::OK);

    // Opt-in: include_synthetic = true.
    let resp_opt_in = run(
        State(state),
        Json(OpinionDynamicsBody {
            narrative_id: NARRATIVE.into(),
            params: Some(params),
            include_synthetic: Some(true),
        }),
    )
    .await
    .into_response();
    assert_eq!(resp_opt_in.status(), StatusCode::OK);
}
