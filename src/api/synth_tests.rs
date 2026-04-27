//! REST tests for `/synth/*`.
//!
//! Phase 6 shipped six tests for the calibration / generation / params /
//! runs surface; Phase 7 adds three more under `mod significance_tests`
//! covering the new `/synth/significance` endpoints.
//!
//! Tests call the handler functions directly with an `AppState` built via
//! [`make_test_state`] — no live HTTP loop required. This mirrors how every
//! other API test in the codebase exercises route handlers (the codebase
//! convention; `tower::ServiceExt::oneshot` is not used elsewhere). Going
//! through the handlers means we exercise the same extractor + body decode
//! + storage paths the real server uses.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use uuid::Uuid;

use crate::api::server::{AppState, LlmConfig};
use crate::api::settings_routes::InferenceConfig;
use crate::api::synth::calibration::{calibrate, delete_params_route, get_params, put_params};
use crate::api::synth::generation::{generate, get_run, list_runs, GenerateBody, RunsListQuery};
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::ingestion::config::IngestionConfig;
use crate::ingestion::geocode::Geocoder;
use crate::ingestion::jobs::IngestionJobQueue;
use crate::ingestion::queue::ValidationQueue;
use crate::query::rag_config::RagConfig;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::types::EathParams;
use crate::synth::{key_synth_run, RunKind, SurrogateRegistry, SurrogateRunSummary};
use crate::temporal::index::IntervalTree;

// ── Test fixtures ───────────────────────────────────────────────────────────

/// Build a minimal `AppState` backed by an in-memory KV store and an
/// inference job queue (so synth-job submission paths actually queue work
/// instead of erroring with "Inference not enabled").
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
    })
}

/// Decode an `axum::response::Response` body into a JSON `Value`. Lets tests
/// assert on response shape without re-deserializing into a typed struct.
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

/// Persist a `SurrogateRunSummary` directly into the KV store so the runs
/// endpoints have something to read back. Mirrors what the generation engine
/// would write at the end of a successful run, minus the lineage marker.
fn seed_run(store: &dyn KVStore, narrative_id: &str) -> Uuid {
    let run_id = Uuid::now_v7();
    let summary = SurrogateRunSummary {
        run_id,
        model: "eath".into(),
        params_hash: "deadbeef".into(),
        source_narrative_id: Some(narrative_id.into()),
        source_state_hash: None,
        output_narrative_id: format!("synth-{run_id}"),
        num_entities: 5,
        num_situations: 10,
        num_participations: 25,
        started_at: chrono::Utc::now(),
        finished_at: chrono::Utc::now(),
        duration_ms: 100,
        kind: RunKind::Generation,
    };
    let key = key_synth_run(narrative_id, &run_id);
    store
        .put(&key, &serde_json::to_vec(&summary).unwrap())
        .expect("seed run");
    run_id
}

// ── Phase 6 tests ───────────────────────────────────────────────────────────

/// T1 — POST /synth/calibrate/{narrative_id} queues a job and returns its id.
#[tokio::test]
async fn test_route_calibrate_returns_job_id() {
    let state = make_test_state();
    // Empty-body case — clients that want the EATH default POST no body.
    let resp = calibrate(State(state.clone()), Path("hamlet".to_string()), None)
        .await
        .into_response();

    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = read_body(resp).await;
    let job_id = body
        .get("job_id")
        .and_then(|v| v.as_str())
        .expect("body must carry job_id");
    assert!(!job_id.is_empty(), "job_id must be non-empty");

    // Job actually landed in the queue (not just a fake response).
    let queue = state.job_queue.as_ref().expect("queue");
    let job = queue.get_job(job_id).expect("queued job");
    match &job.job_type {
        crate::types::InferenceJobType::SurrogateCalibration { narrative_id, model } => {
            assert_eq!(narrative_id, "hamlet");
            assert_eq!(model, "eath", "default model is 'eath'");
        }
        other => panic!("unexpected job type: {other:?}"),
    }
}

/// T2 — POST /synth/generate queues a job and returns its id.
#[tokio::test]
async fn test_route_generate_returns_job_id() {
    let state = make_test_state();
    let body = GenerateBody {
        source_narrative_id: Some("hamlet".into()),
        output_narrative_id: "synth-hamlet".into(),
        model: None,
        params: None,
        seed: Some(42),
        num_steps: Some(50),
        label_prefix: Some("synth".into()),
    };

    let resp = generate(State(state.clone()), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body_json = read_body(resp).await;
    let job_id = body_json
        .get("job_id")
        .and_then(|v| v.as_str())
        .expect("body must carry job_id");

    let queue = state.job_queue.as_ref().expect("queue");
    let job = queue.get_job(job_id).expect("queued job");
    match &job.job_type {
        crate::types::InferenceJobType::SurrogateGeneration {
            source_narrative_id,
            output_narrative_id,
            model,
            seed_override,
            ..
        } => {
            assert_eq!(source_narrative_id.as_deref(), Some("hamlet"));
            assert_eq!(output_narrative_id, "synth-hamlet");
            assert_eq!(model, "eath", "default model is 'eath'");
            assert_eq!(*seed_override, Some(42));
        }
        other => panic!("unexpected job type: {other:?}"),
    }

    // num_steps + label_prefix must land in job.parameters (engine reads them
    // from there because the variant doesn't carry the fields directly).
    assert_eq!(job.parameters.get("num_steps"), Some(&serde_json::json!(50)));
    assert_eq!(
        job.parameters.get("label_prefix"),
        Some(&serde_json::json!("synth"))
    );
}

/// T3 — params round-trip: PUT writes, GET reads, DELETE removes, second GET
/// returns 404. Exercises the entire CRUD surface against one (narrative, model)
/// pair.
#[tokio::test]
async fn test_route_params_round_trip() {
    let state = make_test_state();
    let nid = "macbeth";
    let model = "eath";

    // 1. GET first → 404 (nothing persisted yet).
    let resp = get_params(
        State(state.clone()),
        Path((nid.to_string(), model.to_string())),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // 2. PUT → ok.
    let mut params = EathParams::default();
    params.num_entities = 7;
    params.max_group_size = 5;
    params.p_from_scratch = 0.3;
    let body = serde_json::to_value(&params).unwrap();

    let resp = put_params(
        State(state.clone()),
        Path((nid.to_string(), model.to_string())),
        Json(body),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body, serde_json::json!({"ok": true}));

    // 3. GET → params present + fields preserved.
    let resp = get_params(
        State(state.clone()),
        Path((nid.to_string(), model.to_string())),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body.get("num_entities"), Some(&serde_json::json!(7)));
    assert_eq!(body.get("max_group_size"), Some(&serde_json::json!(5)));

    // 4. DELETE → ok.
    let resp = delete_params_route(
        State(state.clone()),
        Path((nid.to_string(), model.to_string())),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body, serde_json::json!({"ok": true}));

    // 5. GET again → 404 (the round-trip closes here).
    let resp = get_params(
        State(state.clone()),
        Path((nid.to_string(), model.to_string())),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// T4 — POST /synth/generate without source_narrative_id AND without inline
/// params → 400. The engine would error on this same case at runtime; the
/// route catches it synchronously.
#[tokio::test]
async fn test_route_generate_rejects_missing_source() {
    let state = make_test_state();
    let body = GenerateBody {
        source_narrative_id: None,
        output_narrative_id: "synth-orphan".into(),
        model: None,
        params: None,
        seed: None,
        num_steps: None,
        label_prefix: None,
    };
    let resp = generate(State(state), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = read_body(resp).await;
    let msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(
        msg.contains("source_narrative_id"),
        "error must mention the missing field, got: {msg}"
    );
}

/// T5 — runs listing respects `?limit=` and returns newest-first.
#[tokio::test]
async fn test_route_runs_pagination() {
    let state = make_test_state();
    let nid = "lear";

    // Seed 3 runs in chronological order.
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.push(seed_run(state.hypergraph.store(), nid));
        // 2ms separation guarantees v7 UUIDs sort distinctly.
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // No limit → all 3, newest first.
    let resp = list_runs(
        State(state.clone()),
        Path(nid.to_string()),
        Query(RunsListQuery { limit: None }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    let runs = body.as_array().expect("body is an array");
    assert_eq!(runs.len(), 3, "all 3 seeded runs must come back");

    // Newest first: latest seeded id comes first.
    let first_id = runs[0]
        .get("run_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .expect("run_id parses");
    assert_eq!(first_id, *ids.last().unwrap(), "newest first");

    // limit=2 → top 2 only, still newest first.
    let resp = list_runs(
        State(state.clone()),
        Path(nid.to_string()),
        Query(RunsListQuery { limit: Some(2) }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    let runs = body.as_array().expect("body is an array");
    assert_eq!(runs.len(), 2, "limit=2 returns 2 rows");
    let first_id = runs[0]
        .get("run_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .expect("run_id parses");
    assert_eq!(first_id, *ids.last().unwrap(), "still newest first");
}

/// T6 — GET /synth/runs/{narrative_id}/{run_id} for an unknown run → 404.
#[tokio::test]
async fn test_route_runs_get_404_on_unknown_job() {
    let state = make_test_state();
    let nid = "tempest";
    let unknown_run = Uuid::now_v7();

    let resp = get_run(State(state), Path((nid.to_string(), unknown_run)))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = read_body(resp).await;
    let msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("");
    assert!(msg.contains("no run for"), "error mentions missing run: {msg}");
}

// ── Phase 7 significance tests ──────────────────────────────────────────────

mod significance_tests {
    use super::*;

    use crate::api::synth::significance::{
        get_significance_result, list_significance_results, post_significance,
        SignificanceBody, SignificanceListQuery,
    };
    use crate::synth::significance::{
        save_significance_report, SignificanceMetric, SignificanceReport, SyntheticDistribution,
    };

    fn seed_significance_report(
        store: &dyn KVStore,
        narrative_id: &str,
        metric: SignificanceMetric,
    ) -> Uuid {
        let now = chrono::Utc::now();
        let report = SignificanceReport {
            run_id: Uuid::now_v7(),
            narrative_id: narrative_id.into(),
            model: "eath".into(),
            metric,
            k_samples_used: 20,
            source_observation: serde_json::json!({"metric": metric.as_kv_str()}),
            synthetic_distribution: SyntheticDistribution {
                element_keys: vec!["k1".into(), "k2".into()],
                source_values: vec![1.0, 2.0],
                means: vec![0.5, 1.5],
                stddevs: vec![0.1, 0.2],
                z_scores: vec![5.0, 2.5],
                p_values: vec![0.0, 0.05],
                direction: "more_is_significant".into(),
            },
            auto_calibrated: false,
            calibration_fidelity: None,
            note: None,
            started_at: now,
            finished_at: now,
            duration_ms: 1234,
        };
        let id = report.run_id;
        save_significance_report(store, &report).expect("save sig report");
        id
    }

    /// T7 — POST /synth/significance queues a job and returns its id.
    #[tokio::test]
    async fn test_route_significance_returns_job_id() {
        let state = make_test_state();
        let body = SignificanceBody {
            narrative_id: "hamlet".into(),
            metric: "temporal_motifs".into(),
            k: Some(50),
            model: None,
            params_override: None,
            metric_params: Some(serde_json::json!({"max_motif_size": 4})),
        };

        let resp = post_significance(State(state.clone()), Json(body))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let body_json = read_body(resp).await;
        let job_id = body_json
            .get("job_id")
            .and_then(|v| v.as_str())
            .expect("body must carry job_id");

        let queue = state.job_queue.as_ref().expect("queue");
        let job = queue.get_job(job_id).expect("queued job");
        match &job.job_type {
            crate::types::InferenceJobType::SurrogateSignificance {
                narrative_id,
                metric_kind,
                k,
                model,
            } => {
                assert_eq!(narrative_id, "hamlet");
                assert_eq!(metric_kind, "temporal_motifs");
                assert_eq!(*k, 50);
                assert_eq!(model, "eath");
            }
            other => panic!("unexpected job type: {other:?}"),
        }

        // metric_params landed in job.parameters.
        assert_eq!(
            job.parameters.get("metric_params"),
            Some(&serde_json::json!({"max_motif_size": 4}))
        );

        // Sub-test: invalid metric → 400.
        let bad = SignificanceBody {
            narrative_id: "hamlet".into(),
            metric: "not_a_metric".into(),
            k: None,
            model: None,
            params_override: None,
            metric_params: None,
        };
        let resp = post_significance(State(state.clone()), Json(bad))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// T8 — GET on an unknown (narrative, metric, run_id) → 404.
    #[tokio::test]
    async fn test_route_significance_get_404_on_unknown() {
        let state = make_test_state();
        let resp = get_significance_result(
            State(state),
            Path((
                "nope".to_string(),
                "communities".to_string(),
                Uuid::now_v7(),
            )),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// T9 — listing returns newest-first and respects ?limit.
    #[tokio::test]
    async fn test_route_significance_list_returns_paginated() {
        let state = make_test_state();
        let nid = "macbeth";
        let metric = SignificanceMetric::Communities;

        let mut ids = Vec::new();
        for _ in 0..3 {
            ids.push(seed_significance_report(state.hypergraph.store(), nid, metric));
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        // No limit → all 3, newest first.
        let resp = list_significance_results(
            State(state.clone()),
            Path((nid.to_string(), "communities".to_string())),
            Query(SignificanceListQuery { limit: None }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = read_body(resp).await;
        let reports = body.as_array().expect("body is an array");
        assert_eq!(reports.len(), 3);
        let first_id = reports[0]
            .get("run_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .expect("run_id parses");
        assert_eq!(first_id, *ids.last().unwrap(), "newest first");

        // limit=2 → top 2 only.
        let resp = list_significance_results(
            State(state.clone()),
            Path((nid.to_string(), "communities".to_string())),
            Query(SignificanceListQuery { limit: Some(2) }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = read_body(resp).await;
        let reports = body.as_array().expect("body is an array");
        assert_eq!(reports.len(), 2);
    }
}

// ── Phase 7b — contagion REST tests ─────────────────────────────────────────

mod contagion_tests {
    use super::*;

    use crate::api::routes::compute_higher_order_contagion;
    use crate::api::synth::contagion::{
        post_contagion_significance, ContagionSignificanceBody,
    };
    use crate::types::{Entity, EntityType, MaturityLevel};

    fn seed_simple_narrative(state: &Arc<AppState>) -> String {
        let nid = "ho-rest-test".to_string();
        let now = chrono::Utc::now();
        let ids: Vec<Uuid> = (0..3)
            .map(|i| {
                let e = Entity {
                    id: Uuid::now_v7(),
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"name": format!("a{i}")}),
                    beliefs: None,
                    embedding: None,
                    maturity: MaturityLevel::Candidate,
                    confidence: 1.0,
                    confidence_breakdown: None,
                    provenance: vec![],
                    extraction_method: None,
                    narrative_id: Some(nid.clone()),
                    created_at: now,
                    updated_at: now,
                    deleted_at: None,
                    transaction_time: None,
                };
                let id = e.id;
                state.hypergraph.create_entity(e).unwrap();
                id
            })
            .collect();
        // One 3-actor hyperedge.
        let sit = crate::types::Situation {
            id: Uuid::now_v7(),
            name: None,
            description: None,
            properties: serde_json::Value::Null,
            temporal: crate::types::AllenInterval {
                start: Some(now),
                end: Some(now),
                granularity: crate::types::TimeGranularity::Exact,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![crate::types::ContentBlock::text("rest fixture")],
            narrative_level: crate::types::NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            extraction_method: crate::types::ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.clone()),
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
        let sid = sit.id;
        state.hypergraph.create_situation(sit).unwrap();
        for &eid in &ids {
            state
                .hypergraph
                .add_participant(crate::types::Participation {
                    entity_id: eid,
                    situation_id: sid,
                    role: crate::types::Role::Bystander,
                    info_set: None,
                    action: None,
                    payoff: None,
                    seq: 0,
                })
                .unwrap();
        }
        nid
    }

    /// REST T1 — POST /analysis/higher-order-contagion runs the simulation
    /// inline and returns the result blob (no job id).
    #[tokio::test]
    async fn test_route_higher_order_contagion_returns_inline() {
        let state = make_test_state();
        let nid = seed_simple_narrative(&state);

        let body = serde_json::json!({
            "narrative_id": nid,
            "params": {
                "beta_per_size": [1.0, 1.0],
                "gamma": 0.0,
                "threshold": {"kind": "absolute", "value": 1},
                "seed_strategy": {"kind": "random_fraction", "fraction": 0.34},
                "max_steps": 5,
                "rng_seed": 42
            }
        });
        let resp = compute_higher_order_contagion(State(state), Json(body))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = read_body(resp).await;
        // Result shape carries per_step_infected + size_attribution + total_entities.
        assert!(body.get("per_step_infected").is_some());
        assert!(body.get("size_attribution").is_some());
        assert_eq!(body.get("total_entities").and_then(|v| v.as_u64()), Some(3));
    }

    /// REST T2 — POST /synth/contagion-significance queues a
    /// SurrogateContagionSignificance job and returns its id.
    #[tokio::test]
    async fn test_route_contagion_significance_returns_job_id() {
        let state = make_test_state();

        let body = ContagionSignificanceBody {
            narrative_id: "macbeth".into(),
            params: serde_json::json!({
                "beta_per_size": [0.4, 0.5],
                "gamma": 0.1,
                "threshold": {"kind": "absolute", "value": 1},
                "seed_strategy": {"kind": "random_fraction", "fraction": 0.1},
                "max_steps": 50,
                "rng_seed": 9
            }),
            k: Some(20),
            model: None,
            params_override: Some(serde_json::to_value(EathParams {
                a_t_distribution: vec![0.5; 4],
                a_h_distribution: vec![1.0; 4],
                lambda_schedule: vec![],
                p_from_scratch: 0.5,
                omega_decay: 0.95,
                group_size_distribution: vec![1, 1, 1],
                rho_low: 0.5,
                rho_high: 0.3,
                xi: 1.0,
                order_propensity: vec![],
                max_group_size: 4,
                stm_capacity: 7,
                num_entities: 4,
            }).unwrap()),
        };

        let resp = post_contagion_significance(State(state.clone()), Json(body))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let body_json = read_body(resp).await;
        let job_id = body_json
            .get("job_id")
            .and_then(|v| v.as_str())
            .expect("body must carry job_id");

        let queue = state.job_queue.as_ref().expect("queue");
        let job = queue.get_job(job_id).expect("queued job");
        match &job.job_type {
            crate::types::InferenceJobType::SurrogateContagionSignificance {
                narrative_id,
                k,
                model,
                ..
            } => {
                assert_eq!(narrative_id, "macbeth");
                assert_eq!(*k, 20);
                assert_eq!(model, "eath");
            }
            other => panic!("unexpected job type: {other:?}"),
        }

        // params_override propagated into job.parameters.
        assert!(job.parameters.get("params_override").is_some());
        // contagion_params duplicated into job.parameters too (engine reads here).
        assert!(job.parameters.get("contagion_params").is_some());

        // Sub-test: empty narrative_id → 400.
        let bad = ContagionSignificanceBody {
            narrative_id: "".into(),
            params: serde_json::json!({
                "beta_per_size": [0.1],
                "gamma": 0.0,
                "threshold": {"kind": "absolute", "value": 1},
                "seed_strategy": {"kind": "random_fraction", "fraction": 0.1},
                "max_steps": 5,
                "rng_seed": 0
            }),
            k: None,
            model: None,
            params_override: None,
        };
        let resp = post_contagion_significance(State(state), Json(bad))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

// ── Phase 9 — hybrid REST tests ─────────────────────────────────────────────

mod hybrid_tests {
    use super::*;

    use crate::api::synth::generation::{generate_hybrid, GenerateHybridBody};
    use crate::synth::hybrid::HybridComponent;

    /// Phase 9 REST T1 — POST /synth/generate-hybrid queues a job and
    /// returns its id. The queued job's `job_type` is
    /// `SurrogateHybridGeneration`.
    #[tokio::test]
    async fn test_route_generate_hybrid_returns_job_id() {
        let state = make_test_state();
        let body = GenerateHybridBody {
            components: vec![
                HybridComponent {
                    narrative_id: "thriller".into(),
                    model: "eath".into(),
                    weight: 0.7,
                },
                HybridComponent {
                    narrative_id: "drama".into(),
                    model: "eath".into(),
                    weight: 0.3,
                },
            ],
            output_narrative_id: "synth-hybrid-out".into(),
            seed: Some(42),
            num_steps: Some(50),
        };

        let resp = generate_hybrid(State(state.clone()), Json(body))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let body_json = read_body(resp).await;
        let job_id = body_json
            .get("job_id")
            .and_then(|v| v.as_str())
            .expect("body must carry job_id");

        let queue = state.job_queue.as_ref().expect("queue");
        let job = queue.get_job(job_id).expect("queued job");
        match &job.job_type {
            crate::types::InferenceJobType::SurrogateHybridGeneration {
                output_narrative_id,
                seed_override,
                num_steps,
                components,
            } => {
                assert_eq!(output_narrative_id, "synth-hybrid-out");
                assert_eq!(*seed_override, Some(42));
                assert_eq!(*num_steps, Some(50));
                // Components round-tripped through JSON; just confirm the
                // length so the test isn't fragile to field-renames.
                let arr = components.as_array().expect("components is JSON array");
                assert_eq!(arr.len(), 2);
            }
            other => panic!("unexpected job type: {other:?}"),
        }
    }

    /// Phase 9 REST T2 — POST /synth/generate-hybrid rejects weight vectors
    /// that don't sum to 1.0 within tolerance with a synchronous 400.
    #[tokio::test]
    async fn test_route_generate_hybrid_rejects_invalid_weights() {
        let state = make_test_state();
        let body = GenerateHybridBody {
            components: vec![
                HybridComponent {
                    narrative_id: "src-a".into(),
                    model: "eath".into(),
                    weight: 0.4,
                },
                HybridComponent {
                    narrative_id: "src-b".into(),
                    model: "eath".into(),
                    weight: 0.4,
                },
            ],
            output_narrative_id: "synth-hybrid-bad".into(),
            seed: None,
            num_steps: None,
        };

        let resp = generate_hybrid(State(state), Json(body)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = read_body(resp).await;
        let err = body
            .get("error")
            .and_then(|v| v.as_str())
            .expect("error string");
        assert!(err.contains("sum to 1.0"), "error should mention sum: {err}");
    }
}
