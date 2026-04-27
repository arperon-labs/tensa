//! REST tests for Graded Acceptability Sprint Phase 3.
//!
//! Covers:
//! * Version-aware measure CRUD: full round-trip via
//!   `POST /fuzzy/measures/learn` → `GET /fuzzy/measures/{name}/versions`
//!   → `GET /fuzzy/measures/{name}?version=N` → `DELETE …?version=N`.
//! * Backward-compat for the unversioned list path.
//! * Synchronous gradual-argumentation endpoint
//!   (`POST /analysis/argumentation/gradual`).
//!
//! Mirrors the test harness in `src/api/fuzzy_tests.rs`: handlers are
//! called directly with extractor + JSON wrapping; no live HTTP loop.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::Utc;
use uuid::Uuid;

use crate::api::argumentation_gradual::{run_gradual, GradualArgumentationRequest};
use crate::api::fuzzy::learn::{learn_measure_handler, LearnMeasureRequest};
use crate::api::fuzzy::measure::{
    delete_measure, get_measure, list_measures, list_versions, MeasureVersionQuery,
};
use crate::api::server::{AppState, LlmConfig};
use crate::api::settings_routes::InferenceConfig;
use crate::analysis::argumentation_gradual::GradualSemanticsKind;
use crate::fuzzy::synthetic_cib_dataset::generate_synthetic_cib;
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::ingestion::config::IngestionConfig;
use crate::ingestion::geocode::Geocoder;
use crate::ingestion::jobs::IngestionJobQueue;
use crate::ingestion::queue::ValidationQueue;
use crate::query::rag_config::RagConfig;
use crate::source::{ContentionLink, ContentionType};
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::SurrogateRegistry;
use crate::temporal::index::IntervalTree;
use crate::types::{
    AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Situation,
    TimeGranularity,
};

// ── Test fixtures ───────────────────────────────────────────────────────────

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

/// Small, deterministic Phase-2 learn payload — used everywhere we
/// need a learned-measure record without burning compute. Cluster
/// count of 100 matches the §3.3 paper-figure run, but the test
/// asserts behaviour (versioning + 404), not AUC.
fn cib_learn_request(name: &str, dataset_id: &str) -> LearnMeasureRequest {
    LearnMeasureRequest {
        name: name.into(),
        n: 4,
        dataset: generate_synthetic_cib(42, 100),
        dataset_id: dataset_id.into(),
    }
}

/// Seed a 2-situation contention so [`build_framework`] returns a
/// non-empty argumentation framework.
fn seed_contention_pair(state: &AppState, narrative_id: &str) -> (Uuid, Uuid) {
    let a = make_situation(&state.hypergraph, narrative_id);
    let b = make_situation(&state.hypergraph, narrative_id);
    state
        .hypergraph
        .add_contention(ContentionLink {
            situation_a: a,
            situation_b: b,
            contention_type: ContentionType::DirectContradiction,
            description: Some("test contention".into()),
            resolved: false,
            resolution: None,
            created_at: Utc::now(),
        })
        .expect("add contention");
    (a, b)
}

fn make_situation(hg: &Hypergraph, narrative_id: &str) -> Uuid {
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
        raw_content: vec![ContentBlock::text("test scene")],
        narrative_level: NarrativeLevel::Event,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.7,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(narrative_id.into()),
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
    hg.create_situation(sit).expect("create situation")
}

// ── Phase 3 — Version-aware CRUD ───────────────────────────────────────────

/// T1 — full CRUD round-trip via REST handlers: learn → versions list
/// → versioned get → versioned delete → versions list empty.
#[tokio::test]
async fn t1_full_crud_round_trip_via_rest_handlers() {
    let state = make_test_state();

    // 1. POST /fuzzy/measures/learn
    let req = cib_learn_request("learned-cib", "phase3-t1");
    let resp = learn_measure_handler(State(state.clone()), Json(req))
        .await
        .into_response();
    assert_eq!(
        resp.status(),
        StatusCode::CREATED,
        "learn must return 201"
    );
    let summary = read_body(resp).await;
    assert_eq!(summary.get("version").and_then(|v| v.as_u64()), Some(1));

    // 2. GET /fuzzy/measures/{name}/versions
    let resp = list_versions(State(state.clone()), Path("learned-cib".into()))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    let versions = body
        .get("versions")
        .and_then(|v| v.as_array())
        .expect("versions array");
    assert_eq!(
        versions
            .iter()
            .filter_map(|v| v.as_u64())
            .collect::<Vec<_>>(),
        vec![1u64],
        "single learned version after one POST"
    );

    // 3. GET /fuzzy/measures/{name}?version=1
    let resp = get_measure(
        State(state.clone()),
        Path("learned-cib".into()),
        Query(MeasureVersionQuery { version: Some(1) }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let stored = read_body(resp).await;
    assert_eq!(stored.get("version").and_then(|v| v.as_u64()), Some(1));
    assert_eq!(stored.get("name").and_then(|n| n.as_str()), Some("learned-cib"));

    // 4. DELETE /fuzzy/measures/{name}?version=1
    let resp = delete_measure(
        State(state.clone()),
        Path("learned-cib".into()),
        Query(MeasureVersionQuery { version: Some(1) }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    // 5. Versions list now empty.
    let resp = list_versions(State(state.clone()), Path("learned-cib".into()))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    let versions = body
        .get("versions")
        .and_then(|v| v.as_array())
        .expect("versions array");
    assert!(
        versions.is_empty(),
        "version 1 deleted, list must be empty: {versions:?}"
    );
}

/// T2 — re-training under the same name increments the version number;
/// the latest pointer (unversioned read) sees v2 after the second POST.
#[tokio::test]
async fn t2_retraining_increments_version_and_updates_latest_pointer() {
    let state = make_test_state();

    // First training → v1.
    let req = cib_learn_request("learned-cib", "phase3-t2-a");
    let _ = learn_measure_handler(State(state.clone()), Json(req))
        .await
        .into_response();

    // Second training → v2 (different dataset_id so the train/test
    // split is independent — content of the measure isn't asserted,
    // only the version stamp).
    let req = cib_learn_request("learned-cib", "phase3-t2-b");
    let resp = learn_measure_handler(State(state.clone()), Json(req))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let summary = read_body(resp).await;
    assert_eq!(
        summary.get("version").and_then(|v| v.as_u64()),
        Some(2),
        "second POST must bump version to 2"
    );

    // Versions list returns [1, 2] (sorted ascending).
    let resp = list_versions(State(state.clone()), Path("learned-cib".into()))
        .await
        .into_response();
    let body = read_body(resp).await;
    let versions: Vec<u64> = body
        .get("versions")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        .unwrap_or_default();
    assert_eq!(versions, vec![1, 2], "two versions after two trainings");

    // Unversioned GET returns the latest pointer = v2.
    let resp = get_measure(
        State(state.clone()),
        Path("learned-cib".into()),
        Query(MeasureVersionQuery::default()),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let latest = read_body(resp).await;
    assert_eq!(
        latest.get("version").and_then(|v| v.as_u64()),
        Some(2),
        "latest pointer reflects most recent training"
    );
}

/// T3 — `GET /fuzzy/measures/{name}?version=999` returns HTTP 404 with
/// a body that mentions both the name and the requested version.
#[tokio::test]
async fn t3_get_unknown_version_returns_404_with_actionable_error_body() {
    let state = make_test_state();

    // Seed at least one real version so we know the 404 distinguishes
    // "version 999 missing" from "name unknown".
    let req = cib_learn_request("learned-cib", "phase3-t3");
    let _ = learn_measure_handler(State(state.clone()), Json(req))
        .await
        .into_response();

    let resp = get_measure(
        State(state.clone()),
        Path("learned-cib".into()),
        Query(MeasureVersionQuery { version: Some(999) }),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let err = read_body(resp).await;
    let msg = err
        .get("error")
        .and_then(|v| v.as_str())
        .expect("error body has 'error' field");
    assert!(
        msg.contains("learned-cib"),
        "error must mention name; got: {msg}"
    );
    assert!(
        msg.contains("999"),
        "error must mention requested version; got: {msg}"
    );
    assert!(
        msg.to_lowercase().contains("not found"),
        "error must say 'not found'; got: {msg}"
    );
}

/// T4 — `GET /fuzzy/measures` (no query) returns one row per unique
/// name (latest pointer only). Versioned-history slices MUST be
/// filtered out so the list view stays backward-compatible.
#[tokio::test]
async fn t4_list_measures_returns_latest_pointer_only_not_history() {
    let state = make_test_state();

    // Two trainings under the same name → 1 latest pointer + 2
    // version slices in KV. The list view must show 1 row.
    for dataset_id in &["phase3-t4-a", "phase3-t4-b"] {
        let req = cib_learn_request("learned-cib", dataset_id);
        let _ = learn_measure_handler(State(state.clone()), Json(req))
            .await
            .into_response();
    }

    let resp = list_measures(State(state.clone())).await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    let arr = body
        .get("measures")
        .and_then(|v| v.as_array())
        .expect("measures array");
    assert_eq!(
        arr.len(),
        1,
        "exactly one row per unique name; got {} rows: {body:?}",
        arr.len()
    );
    assert_eq!(
        arr[0].get("name").and_then(|n| n.as_str()),
        Some("learned-cib")
    );
    // The single row carries the LATEST version stamp.
    assert_eq!(arr[0].get("version").and_then(|v| v.as_u64()), Some(2));
}

// ── Phase 3 — Synchronous gradual-argumentation endpoint ──────────────────

/// T5 — happy path for the new synchronous endpoint with the
/// h-categoriser semantics. Returns a populated `gradual` field +
/// telemetry.
#[tokio::test]
async fn t5_post_gradual_argumentation_h_categoriser_returns_acceptability() {
    let state = make_test_state();
    let (a, b) = seed_contention_pair(&state, "case-graded-t5");

    let req = GradualArgumentationRequest {
        narrative_id: "case-graded-t5".into(),
        gradual_semantics: GradualSemanticsKind::HCategoriser,
        tnorm: None,
    };
    let resp = run_gradual(State(state.clone()), Json(req))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;
    assert_eq!(
        body.get("narrative_id").and_then(|v| v.as_str()),
        Some("case-graded-t5")
    );

    // Per-argument acceptability map populated for both situations.
    let acc = body
        .get("gradual")
        .and_then(|g| g.get("acceptability"))
        .and_then(|m| m.as_object())
        .expect("gradual.acceptability map");
    assert_eq!(
        acc.len(),
        2,
        "framework has 2 arguments (one per situation in the contention pair)"
    );
    assert!(acc.contains_key(&a.to_string()));
    assert!(acc.contains_key(&b.to_string()));

    // Telemetry surfaced at envelope level.
    let iters = body
        .get("iterations")
        .and_then(|v| v.as_u64())
        .expect("iterations field");
    assert!(iters >= 1, "h-categoriser runs at least one iteration");
    assert_eq!(
        body.get("converged").and_then(|v| v.as_bool()),
        Some(true),
        "h-categoriser is contracting under Gödel — must converge"
    );
}

/// T6 — card-based semantics under the canonical Gödel default
/// reaches a fixed point on a tiny framework. The Amgoud & Ben-Naim
/// 2013 §4.3 Proposition 3 bound is `O(|A|)` *distinct ranks*, not
/// `O(|A|)` numerical iteration steps — the in-tree implementation
/// uses an `L_∞ < 1e-9` convergence test, which on small frameworks
/// may take a handful of refinement steps. We assert convergence and
/// an iteration cap well below `MAX_GRADUAL_ITERATIONS` so the test
/// fails loudly if the loop stops contracting.
#[tokio::test]
async fn t6_post_gradual_argumentation_card_based_converges_within_loop_cap() {
    use crate::analysis::argumentation_gradual::MAX_GRADUAL_ITERATIONS;

    let state = make_test_state();
    let _ = seed_contention_pair(&state, "case-graded-t6");

    let req = GradualArgumentationRequest {
        narrative_id: "case-graded-t6".into(),
        gradual_semantics: GradualSemanticsKind::CardBased,
        tnorm: None,
    };
    let resp = run_gradual(State(state.clone()), Json(req))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;

    let iters = body
        .get("iterations")
        .and_then(|v| v.as_u64())
        .expect("iterations") as u32;
    let argument_count = body
        .get("gradual")
        .and_then(|g| g.get("acceptability"))
        .and_then(|m| m.as_object())
        .map(|m| m.len())
        .unwrap_or(0);
    assert!(
        iters < MAX_GRADUAL_ITERATIONS,
        "card-based must converge well below the safety cap: iters={iters} cap={MAX_GRADUAL_ITERATIONS}"
    );
    assert!(
        argument_count == 2,
        "framework size sanity check: |A|={argument_count}"
    );
    assert_eq!(
        body.get("converged").and_then(|v| v.as_bool()),
        Some(true),
        "card-based reaches a fixed point"
    );
}

/// T7 — invalid `gradual_semantics` body rejects with an HTTP 400.
/// We exercise the axum extractor directly: an unknown variant in the
/// JSON triggers a serde decode error that surfaces as 400 from the
/// `Json<…>` extractor (Phase 0 + Phase 1 wired this; we only verify
/// the contract holds end-to-end).
#[tokio::test]
async fn t7_post_gradual_argumentation_unknown_semantics_rejects_400() {
    use axum::body::Body;
    use axum::http::Request;
    use axum::Router;
    use tower::util::ServiceExt;

    let state = make_test_state();
    let app: Router = Router::new()
        .route(
            "/analysis/argumentation/gradual",
            axum::routing::post(run_gradual),
        )
        .with_state(state);

    let body_json = serde_json::json!({
        "narrative_id": "case-graded-t7",
        "gradual_semantics": "TotallyUnknownVariant",
        "tnorm": null
    });
    let bytes = serde_json::to_vec(&body_json).expect("serialise");
    let req = Request::builder()
        .method("POST")
        .uri("/analysis/argumentation/gradual")
        .header("content-type", "application/json")
        .body(Body::from(bytes))
        .expect("request");

    let resp = app.oneshot(req).await.expect("oneshot");
    // Axum's JSON extractor returns 422 for malformed bodies + 400 for
    // unknown enum variants depending on serde strictness; both belong
    // to the 4xx client-error class. We accept either.
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::UNPROCESSABLE_ENTITY,
        "unknown variant must yield 4xx; got {}",
        resp.status()
    );
}
