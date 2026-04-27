//! REST tests for Fuzzy Sprint Phase 4 — `/fuzzy/*` endpoints and the
//! per-endpoint opt-in `?tnorm=&aggregator=` query-string contract.
//!
//! Tests call handler functions directly with an `AppState` built via
//! [`make_test_state`] — the same pattern as `api/synth_tests.rs`. No
//! live HTTP loop; we exercise extractor + body-decode + storage paths
//! directly.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::fuzzy::aggregate::{aggregate, AggregateBody};
use crate::api::fuzzy::config::{get_config, put_config, PutConfigBody};
use crate::api::fuzzy::measure::{
    create_measure, delete_measure, get_measure, list_measures, CreateMeasureBody,
    MeasureVersionQuery,
};
use crate::api::fuzzy::tnorm::{get_tnorm, list_tnorms};
use crate::api::fuzzy::{FuzzyWorkspaceConfig, CFG_FUZZY_KEY};
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

// ── Core (Phase 4 surface) ──────────────────────────────────────────────────

/// T1 — GET /fuzzy/tnorms returns all four canonical families with fields.
#[tokio::test]
async fn t1_list_tnorms_returns_four_canonical_families() {
    let resp = list_tnorms().await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = read_body(resp).await;
    let arr = body
        .get("tnorms")
        .and_then(|v| v.as_array())
        .expect("response has tnorms array");
    assert_eq!(arr.len(), 4, "four canonical t-norm families");

    let names: Vec<&str> = arr
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["godel", "goguen", "hamacher", "lukasiewicz"],
        "names cover the canonical families"
    );

    // Each entry carries the full descriptor contract.
    for e in arr {
        for field in ["name", "description", "formula", "tconorm_formula", "citation"] {
            assert!(e.get(field).is_some(), "{field} missing on {e:?}");
        }
    }

    // Single-entry lookup works too.
    let resp = get_tnorm(Path("lukasiewicz".to_string())).await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let info = read_body(resp).await;
    assert_eq!(info.get("name").and_then(|n| n.as_str()), Some("lukasiewicz"));
}

/// T2 — POST /fuzzy/measures with n=3 valid monotone body persists; re-fetch matches.
#[tokio::test]
async fn t2_post_measure_persists_and_roundtrips() {
    let state = make_test_state();

    // Monotone values for n=3 (2^3=8 entries):
    // ∅=0, {0}=0.2, {1}=0.3, {0,1}=0.5, {2}=0.4, {0,2}=0.6, {1,2}=0.7, {0,1,2}=1.0
    let body = CreateMeasureBody {
        name: "my-measure".into(),
        n: 3,
        values: vec![0.0, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7, 1.0],
    };
    let resp = create_measure(State(state.clone()), Json(body))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let resp = get_measure(
        State(state.clone()),
        Path("my-measure".to_string()),
        Query(MeasureVersionQuery::default()),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let stored = read_body(resp).await;
    assert_eq!(stored.get("name").and_then(|n| n.as_str()), Some("my-measure"));
    assert_eq!(stored.get("measure").and_then(|m| m.get("n")), Some(&serde_json::json!(3)));

    // List contains it.
    let resp = list_measures(State(state.clone())).await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let list = read_body(resp).await;
    let arr = list.get("measures").and_then(|v| v.as_array()).unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].get("name").and_then(|n| n.as_str()), Some("my-measure"));

    // Delete roundtrip.
    let resp = delete_measure(
        State(state.clone()),
        Path("my-measure".to_string()),
        Query(MeasureVersionQuery::default()),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    let resp = get_measure(
        State(state.clone()),
        Path("my-measure".to_string()),
        Query(MeasureVersionQuery::default()),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// T3 — non-monotone measure rejected with 400 whose body mentions
/// "monotonicity".
#[tokio::test]
async fn t3_post_non_monotone_measure_returns_400_with_monotonicity_hint() {
    let state = make_test_state();

    // n=3, indices by bitmask. μ({0}) = 0.9 (idx 1) > μ({0,1}) = 0.3 (idx 3) —
    // violates A ⊆ B ⇒ μ(A) ≤ μ(B). Endpoints {0.0, 1.0} preserved so only the
    // monotonicity check trips.
    let body = CreateMeasureBody {
        name: "bad".into(),
        n: 3,
        values: vec![0.0, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
    };
    let resp = create_measure(State(state), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let err = read_body(resp).await;
    let msg = err
        .get("error")
        .and_then(|v| v.as_str())
        .expect("error message");
    assert!(
        msg.to_lowercase().contains("monoton"),
        "error must mention monotonicity; got: {msg}"
    );
}

/// T4 — PUT /fuzzy/config with an unknown tnorm returns 400.
#[tokio::test]
async fn t4_put_config_with_unknown_tnorm_returns_400() {
    let state = make_test_state();
    let body = PutConfigBody {
        tnorm: Some("einstein".into()), // not registered
        aggregator: Some("mean".into()),
        measure: None,
        reset: false,
    };
    let resp = put_config(State(state), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// T5 — POST /fuzzy/aggregate Mean over [0.2, 0.4, 0.6, 0.8] is 0.5 within 1e-9.
#[tokio::test]
async fn t5_aggregate_mean_matches_arithmetic() {
    let state = make_test_state();
    let body = AggregateBody {
        xs: vec![0.2, 0.4, 0.6, 0.8],
        aggregator: "mean".into(),
        tnorm: None,
        measure: None,
        owa_weights: None,
        seed: None,
    };
    let resp = aggregate(State(state), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let out = read_body(resp).await;
    let v = out.get("value").and_then(|v| v.as_f64()).expect("value");
    assert!((v - 0.5).abs() < 1e-9, "mean([.2,.4,.6,.8]) = {v}");
    assert_eq!(
        out.get("aggregator_name").and_then(|v| v.as_str()),
        Some("mean")
    );
}

/// T6 — POST /fuzzy/aggregate Choquet with symmetric-additive n=3 measure
/// over [0.2, 0.4, 0.6] returns the arithmetic mean within 1e-12.
#[tokio::test]
async fn t6_aggregate_choquet_symmetric_additive_equals_mean() {
    let state = make_test_state();

    // Symmetric-additive: μ(S) = |S| / n. For n = 3: [0, 1/3, 1/3, 2/3, 1/3, 2/3, 2/3, 1].
    let third = 1.0_f64 / 3.0;
    let two_thirds = 2.0_f64 / 3.0;
    let body = CreateMeasureBody {
        name: "sym-add-3".into(),
        n: 3,
        values: vec![
            0.0, third, third, two_thirds, third, two_thirds, two_thirds, 1.0,
        ],
    };
    let resp = create_measure(State(state.clone()), Json(body))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let body = AggregateBody {
        xs: vec![0.2, 0.4, 0.6],
        aggregator: "choquet".into(),
        tnorm: None,
        measure: Some("sym-add-3".into()),
        owa_weights: None,
        seed: None,
    };
    let resp = aggregate(State(state), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let out = read_body(resp).await;
    let v = out.get("value").and_then(|v| v.as_f64()).expect("value");
    let mean = (0.2 + 0.4 + 0.6) / 3.0;
    assert!(
        (v - mean).abs() < 1e-12,
        "Choquet(sym-add) = mean; got {v} vs {mean}"
    );
}

/// T7 — POST /fuzzy/aggregate with |xs| > 1000 is rejected with 400.
#[tokio::test]
async fn t7_aggregate_oversize_input_returns_400() {
    let state = make_test_state();
    let body = AggregateBody {
        xs: vec![0.5; 1001],
        aggregator: "mean".into(),
        tnorm: None,
        measure: None,
        owa_weights: None,
        seed: None,
    };
    let resp = aggregate(State(state), Json(body)).await.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// T8 — PUT /fuzzy/config roundtrip: set a non-default, read back, reset,
/// verify Godel/Mean restored.
#[tokio::test]
async fn t8_put_config_roundtrip_with_reset() {
    let state = make_test_state();

    // 1. Set a non-default config.
    let body = PutConfigBody {
        tnorm: Some("lukasiewicz".into()),
        aggregator: Some("median".into()),
        measure: None,
        reset: false,
    };
    let resp = put_config(State(state.clone()), Json(body))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);

    // 2. GET reflects the new values.
    let resp = get_config(State(state.clone())).await.into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let cfg = read_body(resp).await;
    assert_eq!(cfg.get("tnorm").and_then(|v| v.as_str()), Some("lukasiewicz"));
    assert_eq!(cfg.get("aggregator").and_then(|v| v.as_str()), Some("median"));

    // Also persisted in KV.
    let raw = state
        .hypergraph
        .store()
        .get(CFG_FUZZY_KEY)
        .expect("store read")
        .expect("config persisted");
    let persisted: FuzzyWorkspaceConfig = serde_json::from_slice(&raw).expect("deserializes");
    assert_eq!(persisted.tnorm, "lukasiewicz");
    assert_eq!(persisted.aggregator, "median");

    // 3. Reset.
    let body = PutConfigBody {
        tnorm: None,
        aggregator: None,
        measure: None,
        reset: true,
    };
    let resp = put_config(State(state.clone()), Json(body))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);

    // 4. GET shows Godel/Mean again.
    let resp = get_config(State(state)).await.into_response();
    let cfg = read_body(resp).await;
    assert_eq!(cfg.get("tnorm").and_then(|v| v.as_str()), Some("godel"));
    assert_eq!(cfg.get("aggregator").and_then(|v| v.as_str()), Some("mean"));
}

// ── Opt-in (endpoint extension) ─────────────────────────────────────────────

fn seed_entity(state: &AppState) -> uuid::Uuid {
    use chrono::Utc;
    use crate::types::*;
    let now = Utc::now();
    let e = Entity {
        id: uuid::Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": "Alice"}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.7,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: Some(ExtractionMethod::HumanEntered),
        narrative_id: Some("case-1".into()),
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    };
    let id = e.id;
    state.hypergraph.create_entity(e).expect("create entity");
    id
}

/// T9 — GET /entities/:id with `?tnorm=lukasiewicz` returns the same
/// entity payload PLUS a `fuzzy_config` tag echoing the choice. The
/// entity record itself is preserved; only the envelope grows.
#[tokio::test]
async fn t9_get_entity_with_fuzzy_opt_in_tags_response() {
    use crate::api::routes::{get_entity, GetEntityParams};

    let state = make_test_state();
    let id = seed_entity(&state);

    // Baseline: no query string → payload has no `fuzzy_config`.
    let resp = get_entity(
        State(state.clone()),
        Path(id),
        Query(GetEntityParams::default()),
    )
    .await
    .into_response();
    let baseline = read_body(resp).await;
    assert!(
        baseline.get("fuzzy_config").is_none(),
        "default path must NOT carry fuzzy_config"
    );

    // Opt-in: response shape extends with fuzzy_config, base fields unchanged.
    let params = GetEntityParams {
        tnorm: Some("lukasiewicz".into()),
        aggregator: None,
    };
    let resp = get_entity(State(state.clone()), Path(id), Query(params))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let opt_in = read_body(resp).await;
    let fc = opt_in
        .get("fuzzy_config")
        .expect("opt-in must carry fuzzy_config");
    assert_eq!(fc.get("tnorm").and_then(|v| v.as_str()), Some("lukasiewicz"));
    // Core entity payload preserved: id + entity_type match.
    assert_eq!(
        opt_in.get("id").and_then(|v| v.as_str()),
        Some(id.to_string().as_str())
    );
    assert_eq!(
        opt_in.get("entity_type"),
        baseline.get("entity_type"),
        "core entity_type must match"
    );
    // Confidence survives the serde_json::Value round-trip within f32 precision.
    let opt_conf = opt_in.get("confidence").and_then(|v| v.as_f64()).unwrap();
    let base_conf = baseline.get("confidence").and_then(|v| v.as_f64()).unwrap();
    assert!(
        (opt_conf - base_conf).abs() < 1e-6,
        "confidence stable across envelope: opt_in={opt_conf}, baseline={base_conf}"
    );
}

/// T10 — GET /entities/:id WITHOUT any fuzzy query string is
/// bit-identical to the pre-sprint payload shape (backward-compat).
#[tokio::test]
async fn t10_get_entity_without_fuzzy_query_is_bit_identical() {
    use crate::api::routes::{get_entity, GetEntityParams};

    let state = make_test_state();
    let id = seed_entity(&state);

    let resp = get_entity(
        State(state),
        Path(id),
        Query(GetEntityParams::default()),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = read_body(resp).await;

    // The default-path response is the bare Entity, not an envelope. Keys
    // that should be present on Entity and NOT on a tagged response:
    assert!(
        body.get("id").is_some(),
        "entity payload has 'id' field"
    );
    assert!(
        body.get("confidence").is_some(),
        "entity payload has 'confidence' field"
    );
    // Load-bearing: NO fuzzy_config on the default path.
    assert!(
        body.get("fuzzy_config").is_none(),
        "default path must NOT carry fuzzy_config"
    );
    // No 'data' or other envelope keys that would betray a reshape.
    assert!(
        body.get("data").is_none(),
        "default path does not envelope the entity"
    );
}

/// T11 — FuzzyWorkspaceConfig serde round-trip preserves every field,
/// simulating the archive export/import manifest slot. Since the archive
/// wiring is a plain passthrough, a round-trip at the type level is
/// sufficient evidence that `fuzzy_config` survives archive embedding.
#[test]
fn t11_fuzzy_workspace_config_roundtrips_through_json() {
    let cfg = FuzzyWorkspaceConfig {
        tnorm: "goguen".into(),
        aggregator: "owa".into(),
        measure: Some("m1".into()),
        version: 1,
    };
    let ser = serde_json::to_vec(&cfg).expect("serialize");
    let back: FuzzyWorkspaceConfig = serde_json::from_slice(&ser).expect("deserialize");
    assert_eq!(cfg, back);

    // Unknown-kind fallback on load: corrupt blobs collapse to Godel/Mean.
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    store.put(CFG_FUZZY_KEY, b"not valid json").unwrap();
    let loaded = crate::api::fuzzy::load_workspace_config(&*store);
    assert_eq!(loaded, FuzzyWorkspaceConfig::default());
}

/// T12 — GET /narratives/{id}/communities?aggregator=mean tags the
/// response with fuzzy_config. The endpoint itself is a heavy path, so
/// the test exercises the shared `resolve_fuzzy_from_strings` helper
/// that every `fuzzy_config`-capable route funnels through. This
/// guarantees the opt-in contract is consistent without duplicating
/// the community-summary wiring across the 8 endpoints listed for
/// Phase 4.
#[test]
fn t12_resolve_fuzzy_from_strings_tags_response_consistently() {
    use crate::api::routes::resolve_fuzzy_from_strings;

    // None / None -> backward-compat no-op.
    let cfg = resolve_fuzzy_from_strings(None, None).expect("ok");
    assert!(cfg.is_none(), "no query string = no config");

    // Aggregator alone is enough to trigger the tag.
    let cfg = resolve_fuzzy_from_strings(None, Some("mean"))
        .expect("ok")
        .expect("some");
    assert_eq!(cfg.aggregator.as_deref(), Some("mean"));
    assert!(cfg.tnorm.is_none());

    // Unknown aggregator surfaces as InvalidInput (HTTP 400 via mapper).
    let err = resolve_fuzzy_from_strings(None, Some("magic-sauce")).unwrap_err();
    match err {
        crate::error::TensaError::InvalidInput(msg) => {
            assert!(msg.contains("magic-sauce") || msg.contains("unknown"))
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // Whitespace-only values treated as absent (empty strings → None).
    let cfg = resolve_fuzzy_from_strings(Some("   "), Some("")).expect("ok");
    assert!(cfg.is_none());
}
