//! REST tests for `/inference/hypergraph-reconstruction*`.
//!
//! Mirrors `crate::api::synth_tests` — calls handler functions directly via
//! a hand-built [`AppState`] backed by an in-memory KV store + a real
//! `JobQueue`. No live HTTP loop — that matches the project's API test
//! convention.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use uuid::Uuid;

use crate::api::inference::reconstruction::{
    get_result, materialize, submit, MaterializeBody, SubmitReconstructionBody,
};
use crate::api::server::{AppState, LlmConfig};
use crate::api::settings_routes::InferenceConfig;
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::inference::types::InferenceJob;
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
    Entity, EntityType, ExtractionMethod, InferenceJobType, InferenceResult, JobPriority,
    JobStatus, MaturityLevel,
};

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

/// T1 — POST /inference/hypergraph-reconstruction queues a job, GET on the
/// returned job id (after a synthetic completion writeback to the queue)
/// returns the InferenceResult envelope.
#[tokio::test]
async fn test_reconstruction_rest_roundtrip() {
    let state = make_test_state();
    let resp = submit(
        State(state.clone()),
        Json(SubmitReconstructionBody {
            narrative_id: "n1".into(),
            params: Some(serde_json::json!({ "max_order": 3 })),
        }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = read_body(resp).await;
    let job_id = body["job_id"].as_str().expect("job_id").to_string();
    assert_eq!(body["status"], "Pending");

    // Confirm the job landed on the queue with the right variant.
    let job = state.job_queue.as_ref().unwrap().get_job(&job_id).unwrap();
    match &job.job_type {
        InferenceJobType::HypergraphReconstruction { narrative_id, .. } => {
            assert_eq!(narrative_id, "n1");
        }
        other => panic!("unexpected variant: {other:?}"),
    }

    // Without a worker pool to execute the job, simulate completion by
    // calling JobQueue.complete directly (the engine would do this from
    // the worker thread). This exercises the GET path's success branch.
    let dummy_result_payload = serde_json::json!({
        "kind": "reconstruction_done",
        "result": {
            "inferred_edges": [],
            "coefficient_matrix_stats": {
                "n_entities": 0,
                "n_library_terms": 0,
                "n_timesteps": 0,
                "sparsity": 1.0,
                "condition_number_approx": 1.0,
                "lambda_used": 0.05,
                "pearson_filtered_pairs": 0,
            },
            "goodness_of_fit": 0.9,
            "observation_source": "ParticipationRate",
            "params_used": {
                "observation": "ParticipationRate",
                "window_seconds": 60,
                "time_resolution_seconds": 60,
                "max_order": 3,
                "lambda_l1": 0.0,
                "derivative_estimator": {"SavitzkyGolay": {"window": 5, "order": 2}},
                "symmetrize": true,
                "pearson_filter_threshold": 0.1,
                "bootstrap_k": 10,
                "entity_cap": 200,
                "lambda_cv": false,
                "bootstrap_seed": 3405691597u64,
            },
            "time_range": ["2026-04-22T00:00:00Z", "2026-04-22T00:01:00Z"],
            "bootstrap_resamples_completed": 10,
            "warnings": [],
        }
    });
    let inference_result = InferenceResult {
        job_id: job_id.clone(),
        job_type: job.job_type.clone(),
        target_id: job.target_id,
        result: dummy_result_payload,
        confidence: 0.9,
        explanation: Some("test".into()),
        status: JobStatus::Completed,
        created_at: job.created_at,
        completed_at: Some(chrono::Utc::now()),
    };
    state
        .job_queue
        .as_ref()
        .unwrap()
        .store_result(inference_result)
        .unwrap();

    // GET the result — should now succeed with 200.
    let resp = get_result(State(state.clone()), Path(job_id.clone()))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body["job_id"].as_str().unwrap(), job_id);
    assert_eq!(body["status"], "Completed");
}

/// Submitting with an empty narrative_id must reject synchronously (400).
#[tokio::test]
async fn test_reconstruction_rest_rejects_empty_narrative_id() {
    let state = make_test_state();
    let resp = submit(
        State(state),
        Json(SubmitReconstructionBody {
            narrative_id: "  ".into(),
            params: None,
        }),
    )
    .await
    .into_response();
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "should not be 2xx"
    );
}

/// Materialization without `opt_in: true` must reject synchronously (400).
#[tokio::test]
async fn test_reconstruction_rest_materialize_requires_opt_in() {
    let state = make_test_state();
    // Use a random uuid string for job_id — handler should refuse before
    // even hitting the queue because opt_in is false.
    let resp = materialize(
        State(state),
        Path("anything".into()),
        Json(MaterializeBody {
            output_narrative_id: "out".into(),
            opt_in: false,
            confidence_threshold: None,
        }),
    )
    .await
    .into_response();
    assert!(resp.status().is_client_error() || resp.status().is_server_error());
}

/// End-to-end materialization roundtrip: submit → simulate completion →
/// materialize → assert the new Situations exist under output_narrative_id
/// with ExtractionMethod::Reconstructed.
#[tokio::test]
async fn test_reconstruction_rest_materialize_roundtrip() {
    let state = make_test_state();
    let hg = &state.hypergraph;

    // Seed two entities under the source narrative so members can be
    // resolved during materialization.
    let mut member_uuids = Vec::new();
    for _ in 0..2 {
        let id = Uuid::now_v7();
        let entity = Entity {
            id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({}),
            beliefs: None,
            embedding: None,
            narrative_id: Some("src".into()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::Sensor),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap();
        member_uuids.push(id);
    }

    // Submit reconstruction job and stamp a synthetic InferenceResult on
    // the queue so the materialize handler has something to read.
    let resp = submit(
        State(state.clone()),
        Json(SubmitReconstructionBody {
            narrative_id: "src".into(),
            params: None,
        }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let job_id = read_body(resp).await["job_id"]
        .as_str()
        .unwrap()
        .to_string();

    let job = state.job_queue.as_ref().unwrap().get_job(&job_id).unwrap();
    let payload = serde_json::json!({
        "kind": "reconstruction_done",
        "result": {
            "inferred_edges": [
                {
                    "members": member_uuids,
                    "order": 2,
                    "weight": 0.42,
                    "confidence": 0.95,
                    "possible_masking_artifact": false,
                }
            ],
            "coefficient_matrix_stats": {
                "n_entities": 2,
                "n_library_terms": 1,
                "n_timesteps": 10,
                "sparsity": 0.5,
                "condition_number_approx": 1.0,
                "lambda_used": 0.05,
                "pearson_filtered_pairs": 0,
            },
            "goodness_of_fit": 0.9,
            "observation_source": "ParticipationRate",
            "params_used": {
                "observation": "ParticipationRate",
                "window_seconds": 60,
                "time_resolution_seconds": 60,
                "max_order": 3,
                "lambda_l1": 0.0,
                "derivative_estimator": {"SavitzkyGolay": {"window": 5, "order": 2}},
                "symmetrize": true,
                "pearson_filter_threshold": 0.1,
                "bootstrap_k": 10,
                "entity_cap": 200,
                "lambda_cv": false,
                "bootstrap_seed": 3405691597u64,
            },
            "time_range": ["2026-04-22T00:00:00Z", "2026-04-22T00:01:00Z"],
            "bootstrap_resamples_completed": 10,
            "warnings": [],
        }
    });
    let result = InferenceResult {
        job_id: job_id.clone(),
        job_type: job.job_type.clone(),
        target_id: job.target_id,
        result: payload,
        confidence: 0.95,
        explanation: None,
        status: JobStatus::Completed,
        created_at: job.created_at,
        completed_at: Some(chrono::Utc::now()),
    };
    state
        .job_queue
        .as_ref()
        .unwrap()
        .store_result(result)
        .unwrap();

    // Materialize — opt_in must be true for the handler to proceed.
    let resp = materialize(
        State(state.clone()),
        Path(job_id.clone()),
        Json(MaterializeBody {
            output_narrative_id: "out".into(),
            opt_in: true,
            confidence_threshold: None,
        }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(body["situations_created"].as_u64(), Some(1));
    assert_eq!(body["output_narrative_id"], "out");

    // Verify the Situation was created with ExtractionMethod::Reconstructed.
    let sits = state.hypergraph.list_situations_by_narrative("out").unwrap();
    assert_eq!(sits.len(), 1);
    match &sits[0].extraction_method {
        ExtractionMethod::Reconstructed {
            source_narrative_id,
            job_id: jid,
        } => {
            assert_eq!(source_narrative_id, "src");
            assert_eq!(jid, &job_id);
        }
        other => panic!("unexpected extraction_method: {other:?}"),
    }
}

// Suppress unused-import warnings for helpers added for symmetry with the
// synth_tests module — keep them around for future fixtures.
#[allow(dead_code)]
fn _unused_imports() {
    let _ = (InferenceJob {
        id: String::new(),
        job_type: InferenceJobType::HypergraphReconstruction {
            narrative_id: String::new(),
            params: serde_json::Value::Null,
        },
        target_id: Uuid::nil(),
        parameters: serde_json::Value::Null,
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    },);
}
