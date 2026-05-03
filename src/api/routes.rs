use std::sync::Arc;

use tracing;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::server::AppState;
use crate::error::TensaError;
use crate::inference::types::InferenceJob;
use crate::query::{executor, parser, planner};
use crate::types::*;

/// Query parameters for paginated list endpoints.
#[derive(Deserialize)]
pub struct PaginationParams {
    pub limit: Option<usize>,
    pub after: Option<String>,
}

/// Paginated response wrapper.
#[derive(Serialize)]
pub struct PaginatedResponse<T: Serialize> {
    pub data: Vec<T>,
    pub next_cursor: Option<String>,
}

/// Clamp pagination limit to valid range (1..=1000, default 50).
pub fn clamp_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(50).clamp(1, 1000)
}

/// Serialize a value to JSON and return 200 OK, or 500 on serialization failure.
/// Uses single-pass serialization (to_vec) to avoid the double-traversal of to_value + Json.
pub fn json_ok(val: &impl Serialize) -> axum::response::Response {
    match serde_json::to_vec(val) {
        Ok(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Err(e) => error_response(TensaError::Serialization(e.to_string())).into_response(),
    }
}

/// Map TensaError to HTTP status code + JSON body.
pub fn error_response(err: TensaError) -> (StatusCode, Json<serde_json::Value>) {
    let (status, msg) = match &err {
        TensaError::EntityNotFound(_) | TensaError::SituationNotFound(_) => {
            (StatusCode::NOT_FOUND, err.to_string())
        }
        TensaError::ParticipationNotFound { .. }
        | TensaError::NotFound(_)
        | TensaError::NarrativeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        TensaError::ParticipationExists { .. } | TensaError::NarrativeExists(_) => {
            (StatusCode::CONFLICT, err.to_string())
        }
        TensaError::InvalidMaturityTransition { .. }
        | TensaError::ParseError(_)
        | TensaError::InvalidQuery(_)
        | TensaError::InvalidInterval(_)
        | TensaError::CausalCycle { .. }
        | TensaError::ExtractionError(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        TensaError::ValidationQueueError(ref msg) if msg.contains("not found") => {
            (StatusCode::NOT_FOUND, err.to_string())
        }
        TensaError::ValidationQueueError(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        TensaError::JobNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        TensaError::InferenceError(_) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        TensaError::SourceNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        TensaError::SourceExists(_) => (StatusCode::CONFLICT, err.to_string()),
        TensaError::AttributionExists { .. } | TensaError::ContentionExists { .. } => {
            (StatusCode::CONFLICT, err.to_string())
        }
        TensaError::ContentionNotFound { .. } => (StatusCode::NOT_FOUND, err.to_string()),
        TensaError::LlmRateLimit { .. } => (StatusCode::TOO_MANY_REQUESTS, err.to_string()),
        TensaError::ImportError(_) | TensaError::DocParseError(_) => {
            (StatusCode::BAD_REQUEST, err.to_string())
        }
        TensaError::TaxonomyEntryExists(_, _) => (StatusCode::CONFLICT, err.to_string()),
        TensaError::TaxonomyBuiltinRemoval(_, _) => (StatusCode::BAD_REQUEST, err.to_string()),
        TensaError::NarrativeMergeError(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        TensaError::ExportError(_) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        TensaError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    };
    (status, Json(serde_json::json!({"error": msg})))
}

/// GET /health
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "git": env!("TENSA_GIT_HASH"),
        "build": crate::build_label(),
    }))
}

/// POST /reset — Delete all data while preserving settings (cfg/ prefix).
///
/// Wipes every KV prefix except `cfg/`. Clears the in-memory interval tree
/// and vector index as well.
pub async fn reset_data(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use crate::hypergraph::keys;

    // Every data prefix in the KV store. `cfg/` is intentionally excluded.
    let data_prefixes: &[&[u8]] = &[
        keys::ENTITY,
        keys::SITUATION,
        keys::PARTICIPATION,
        keys::PARTICIPATION_REVERSE,
        keys::CAUSAL,
        keys::CAUSAL_REVERSE,
        keys::STATE_VERSION,
        keys::INFERENCE_RESULT,
        keys::VALIDATION_LOG,
        keys::META,
        keys::VALIDATION_QUEUE,
        keys::ENTITY_ALIAS,
        keys::CHUNK,
        keys::INFERENCE_JOB,
        keys::NARRATIVE,
        keys::CORPUS,
        keys::PATTERN,
        keys::USER_ARC,
        keys::REVISION,
        keys::REVISION_NARRATIVE_IDX,
        keys::NARRATIVE_PLAN,
        keys::WORKSHOP_REPORT,
        keys::WORKSHOP_REPORT_NARRATIVE_IDX,
        keys::PINNED_FACT,
        keys::COST_LEDGER,
        keys::PROJECT,
        keys::TAXONOMY,
        keys::PROJECT_NARRATIVE_IDX,
        keys::SOURCE,
        keys::SOURCE_ATTRIBUTION,
        keys::SOURCE_ATTRIBUTION_REV,
        keys::CONTENTION,
        keys::CONTENTION_REVERSE,
        keys::ENTITY_TYPE_IDX,
        keys::ENTITY_NARRATIVE_IDX,
        keys::SITUATION_LEVEL_IDX,
        keys::SITUATION_NARRATIVE_IDX,
        // Analysis sub-prefixes (all under an/)
        keys::ANALYSIS_CENTRALITY,
        keys::ANALYSIS_ENTROPY,
        keys::ANALYSIS_MUTUAL_INFO,
        keys::ANALYSIS_BELIEF,
        keys::ANALYSIS_EVIDENCE,
        keys::ANALYSIS_ARGUMENTATION,
        keys::ANALYSIS_CONTAGION,
        keys::ANALYSIS_STYLE_PROFILE,
        keys::ANALYSIS_FINGERPRINT,
        // Ingestion jobs (ig/ prefix)
        b"ig/",
    ];

    let store = state.hypergraph.store();
    let mut deleted = 0usize;
    for prefix in data_prefixes {
        match store.prefix_scan(prefix) {
            Ok(pairs) => {
                let ops: Vec<_> = pairs
                    .iter()
                    .map(|(k, _)| crate::store::TxnOp::Delete(k.clone()))
                    .collect();
                deleted += ops.len();
                if !ops.is_empty() {
                    if let Err(e) = store.transaction(ops) {
                        return error_response(e).into_response();
                    }
                }
            }
            Err(e) => return error_response(e).into_response(),
        }
    }

    // Clear in-memory interval tree
    if let Ok(mut tree) = state.interval_tree.write() {
        *tree = crate::temporal::index::IntervalTree::new();
    }

    // Clear in-memory vector index
    if let Some(ref vi) = state.vector_index {
        if let Ok(mut idx) = vi.write() {
            let dim = idx.dimension();
            *idx = crate::ingestion::vector::VectorIndex::new(dim);
        }
    }

    // Clear ephemeral ingestion progress/cancel maps
    if let Ok(mut progress) = state.ingestion_progress.lock() {
        progress.clear();
    }
    if let Ok(mut cancel) = state.ingestion_cancel_flags.lock() {
        cancel.clear();
    }

    Json(serde_json::json!({
        "status": "ok",
        "deleted_keys": deleted
    }))
    .into_response()
}

/// POST /entities
pub async fn create_entity(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let entity: Entity = match serde_json::from_value(body) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    match state.hypergraph.create_entity(entity) {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Query parameters for GET /entities.
#[derive(Deserialize)]
pub struct ListEntitiesParams {
    pub narrative_id: Option<String>,
    pub entity_type: Option<String>,
    pub limit: Option<usize>,
    /// EATH Phase 3 — when true, include synthetic records in the response.
    /// Default false: synthetic records are filtered out at the boundary so
    /// no aggregation endpoint silently mixes empirical and surrogate data.
    /// Accepts the legacy `includeSynthetic` spelling too — if a Studio build
    /// shipped before the canonical spelling was settled, a `tracing::warn!`
    /// fires and the legacy value wins.
    #[serde(default)]
    pub include_synthetic: Option<bool>,
    #[serde(default, rename = "includeSynthetic")]
    pub include_synthetic_camel: Option<bool>,
    /// Fuzzy Sprint Phase 4 — per-request t-norm override.
    /// `None` = pre-sprint bit-identical behaviour.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Fuzzy Sprint Phase 4 — per-request aggregator override.
    #[serde(default)]
    pub aggregator: Option<String>,
}

impl ListEntitiesParams {
    fn include_synthetic(&self) -> bool {
        if let Some(v) = self.include_synthetic_camel {
            tracing::warn!(
                "deprecated query param `includeSynthetic` — use `include_synthetic` instead"
            );
            return v;
        }
        self.include_synthetic.unwrap_or(false)
    }
}

/// GET /entities — list entities with optional server-side filtering.
pub async fn list_entities(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<ListEntitiesParams>,
) -> impl IntoResponse {
    // Fuzzy Sprint Phase 4 — resolve the optional ?tnorm=&aggregator= opt-in
    // BEFORE any data work so an invalid kind returns 400 without touching
    // the store. When neither param is supplied we take the pre-sprint path
    // bit-identically (Phase 4 T10 regression).
    let fuzzy_cfg = match resolve_fuzzy_from_strings(
        params.tnorm.as_deref(),
        params.aggregator.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => return error_response(e).into_response(),
    };

    let limit = params.limit.unwrap_or(2000).clamp(1, 10000);
    let include_synth = params.include_synthetic();
    let hg = &state.hypergraph;

    // Parse entity_type filter if provided
    let type_filter: Option<EntityType> = params.entity_type.as_deref().and_then(|t| match t {
        "Actor" => Some(EntityType::Actor),
        "Location" => Some(EntityType::Location),
        "Artifact" => Some(EntityType::Artifact),
        "Concept" => Some(EntityType::Concept),
        "Organization" => Some(EntityType::Organization),
        _ => None,
    });

    // Apply synthetic filter BEFORE the limit so a real-records-only query
    // doesn't get silently truncated by a flood of synthetic noise.
    let entities: Vec<Entity> = match (&params.narrative_id, &type_filter) {
        // Both filters: fetch by narrative, then filter by type in Rust
        (Some(nid), Some(et)) => crate::synth::emit::filter_synthetic_entities(
            hg.list_entities_by_narrative(nid).unwrap_or_default(),
            include_synth,
        )
        .into_iter()
        .filter(|e| &e.entity_type == et)
        .take(limit)
        .collect(),
        // Narrative only
        (Some(nid), None) => {
            let mut v = crate::synth::emit::filter_synthetic_entities(
                hg.list_entities_by_narrative(nid).unwrap_or_default(),
                include_synth,
            );
            v.truncate(limit);
            v
        }
        // Type only
        (None, Some(et)) => {
            let mut v = crate::synth::emit::filter_synthetic_entities(
                hg.list_entities_by_type(et).unwrap_or_default(),
                include_synth,
            );
            v.truncate(limit);
            v
        }
        // No filters: scan all (capped)
        (None, None) => crate::synth::emit::filter_synthetic_entities(
            hg.list_entities_by_maturity(MaturityLevel::Candidate)
                .unwrap_or_default(),
            include_synth,
        )
        .into_iter()
        .take(limit)
        .collect(),
    };

    if let Some(ref cfg) = fuzzy_cfg {
        json_ok(&serde_json::json!({
            "data": entities,
            "total": entities.len(),
            "fuzzy_config": cfg,
        }))
    } else {
        json_ok(&serde_json::json!({ "data": entities, "total": entities.len() }))
    }
}

/// Query parameters for GET /entities/:id — currently only the Fuzzy
/// Sprint Phase 4 opt-in `?tnorm=&aggregator=` pair. All fields optional.
#[derive(Deserialize, Default)]
pub struct GetEntityParams {
    #[serde(default)]
    pub tnorm: Option<String>,
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// GET /entities/:id
pub async fn get_entity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    axum::extract::Query(params): axum::extract::Query<GetEntityParams>,
) -> impl IntoResponse {
    let fuzzy_cfg = match resolve_fuzzy_from_strings(
        params.tnorm.as_deref(),
        params.aggregator.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => return error_response(e).into_response(),
    };

    match state.hypergraph.get_entity(&id) {
        Ok(entity) => {
            if let Some(ref cfg) = fuzzy_cfg {
                // Opt-in response envelope: tag the fuzzy config back so the
                // caller can audit which semantics were used. Pre-sprint
                // (no-query-string) path returns the raw Entity unchanged.
                let mut val = match serde_json::to_value(&entity) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_response(TensaError::Serialization(e.to_string()))
                            .into_response()
                    }
                };
                if let serde_json::Value::Object(ref mut map) = val {
                    map.insert(
                        "fuzzy_config".to_string(),
                        serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
                    );
                }
                json_ok(&val)
            } else {
                json_ok(&entity)
            }
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// Resolve a `?tnorm=&aggregator=` pair via
/// [`crate::api::fuzzy::parse_fuzzy_config`]. Shared across every endpoint
/// that respects the Phase 4 opt-in so the response shape is uniform.
pub fn resolve_fuzzy_from_strings(
    tnorm: Option<&str>,
    aggregator: Option<&str>,
) -> Result<Option<crate::api::fuzzy::FuzzyConfig>, TensaError> {
    let mut params: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    if let Some(t) = tnorm {
        if !t.trim().is_empty() {
            params.insert("tnorm".to_string(), t.to_string());
        }
    }
    if let Some(a) = aggregator {
        if !a.trim().is_empty() {
            params.insert("aggregator".to_string(), a.to_string());
        }
    }
    crate::api::fuzzy::parse_fuzzy_config(&params)
}

/// PUT /entities/:id — Update entity properties, confidence, or narrative.
pub async fn update_entity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let updates = match body.as_object() {
        Some(_) => body,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Request body must be a JSON object"})),
            )
                .into_response()
        }
    };
    match state
        .hypergraph
        .update_entity(&id, |e| e.apply_patch(&updates))
    {
        Ok(entity) => json_ok(&entity),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /entities/:id
pub async fn delete_entity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.delete_entity(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /entities/:id/restore — Restore a soft-deleted entity.
pub async fn restore_entity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.restore_entity(&id) {
        Ok(entity) => json_ok(&entity),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /entities/merge — Merge two entities.
pub async fn merge_entities(
    State(state): State<Arc<AppState>>,
    Json(body): Json<MergeRequest>,
) -> impl IntoResponse {
    match state
        .hypergraph
        .merge_entities(&body.keep_id, &body.absorb_id)
    {
        Ok(entity) => json_ok(&entity),
        Err(e) => error_response(e).into_response(),
    }
}

/// Request body for POST /entities/merge.
#[derive(Deserialize)]
pub struct MergeRequest {
    pub keep_id: Uuid,
    pub absorb_id: Uuid,
}

/// POST /entities/:id/split — Split an entity.
pub async fn split_entity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(body): Json<SplitRequest>,
) -> impl IntoResponse {
    match state.hypergraph.split_entity(&id, body.situation_ids) {
        Ok((source, clone)) => {
            let result = match (serde_json::to_value(&source), serde_json::to_value(&clone)) {
                (Ok(s), Ok(c)) => serde_json::json!({ "source": s, "clone": c }),
                _ => {
                    return error_response(TensaError::Serialization(
                        "Failed to serialize split result".into(),
                    ))
                    .into_response()
                }
            };
            (StatusCode::OK, Json(result)).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// Request body for POST /entities/:id/split.
#[derive(Deserialize)]
pub struct SplitRequest {
    pub situation_ids: Vec<Uuid>,
}

/// Fill missing required fields on a `Situation` JSON body so that a caller
/// can POST a permissive partial DTO (matching what the MCP `create_situation`
/// shim and most naive HTTP clients send) and still get a valid `Situation`
/// after deserialization. A body that already has every field stays
/// unchanged — full-struct callers continue to work.
///
/// Permissive conveniences applied (idempotent):
/// - `id` defaults to a fresh v7 UUID
/// - `maturity` defaults to `"Candidate"`, `confidence` to `0.5`
/// - `extraction_method` defaults to `"HumanEntered"`
/// - `narrative_level` defaults to `"Scene"`
/// - `created_at` / `updated_at` default to now
/// - `temporal` is assembled from flat `start` / `end` / `granularity` keys
///   when absent (granularity defaults to `"Approximate"`)
/// - `raw_content` accepts a bare string and wraps it into
///   `[{"content_type":"Text","content":s,"source":null}]`
/// - `causes` / `provenance` default to `[]`
pub fn fill_situation_defaults(body: &mut serde_json::Value) {
    use serde_json::{json, Value};
    let obj = match body.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    let now = chrono::Utc::now();

    obj.entry("id")
        .or_insert_with(|| json!(uuid::Uuid::now_v7()));
    obj.entry("maturity").or_insert_with(|| json!("Candidate"));
    obj.entry("confidence").or_insert_with(|| json!(0.5));
    obj.entry("extraction_method")
        .or_insert_with(|| json!("HumanEntered"));
    obj.entry("narrative_level")
        .or_insert_with(|| json!("Scene"));
    obj.entry("created_at").or_insert_with(|| json!(now));
    obj.entry("updated_at").or_insert_with(|| json!(now));
    obj.entry("causes").or_insert_with(|| json!([]));
    obj.entry("provenance").or_insert_with(|| json!([]));

    if !obj.contains_key("temporal") {
        let start = obj.remove("start").unwrap_or(Value::Null);
        let end = obj.remove("end").unwrap_or(Value::Null);
        let granularity = obj
            .remove("granularity")
            .unwrap_or_else(|| json!("Approximate"));
        obj.insert(
            "temporal".into(),
            json!({
                "start": start,
                "end": end,
                "granularity": granularity,
                "relations": [],
            }),
        );
    }

    if let Some(rc) = obj.get_mut("raw_content") {
        if rc.is_string() {
            let s = rc.as_str().unwrap().to_string();
            *rc = json!([{
                "content_type": "Text",
                "content": s,
                "source": null,
            }]);
        }
    }
}

/// POST /situations
///
/// Accepts either a full `Situation` JSON or a permissive partial body —
/// see [`fill_situation_defaults`] for the fields that are auto-supplied.
pub async fn create_situation(
    State(state): State<Arc<AppState>>,
    Json(mut body): Json<serde_json::Value>,
) -> impl IntoResponse {
    fill_situation_defaults(&mut body);
    let situation: Situation = match serde_json::from_value(body) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    match state.hypergraph.create_situation(situation) {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Query parameters for GET /situations.
#[derive(Deserialize)]
pub struct ListSituationsParams {
    pub narrative_id: Option<String>,
    pub narrative_level: Option<String>,
    pub limit: Option<usize>,
    /// EATH Phase 3 — when true, include synthetic records in the response.
    /// Default false. Same legacy-spelling fallback as `ListEntitiesParams`.
    #[serde(default)]
    pub include_synthetic: Option<bool>,
    #[serde(default, rename = "includeSynthetic")]
    pub include_synthetic_camel: Option<bool>,
    /// Fuzzy Sprint Phase 4 — optional per-request t-norm override.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Fuzzy Sprint Phase 4 — optional per-request aggregator override.
    #[serde(default)]
    pub aggregator: Option<String>,
}

impl ListSituationsParams {
    fn include_synthetic(&self) -> bool {
        if let Some(v) = self.include_synthetic_camel {
            tracing::warn!(
                "deprecated query param `includeSynthetic` — use `include_synthetic` instead"
            );
            return v;
        }
        self.include_synthetic.unwrap_or(false)
    }
}

/// GET /situations — list situations with optional server-side filtering.
pub async fn list_situations(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<ListSituationsParams>,
) -> impl IntoResponse {
    let fuzzy_cfg = match resolve_fuzzy_from_strings(
        params.tnorm.as_deref(),
        params.aggregator.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => return error_response(e).into_response(),
    };

    let limit = params.limit.unwrap_or(2000).clamp(1, 10000);
    let include_synth = params.include_synthetic();
    let hg = &state.hypergraph;

    let level_filter: Option<NarrativeLevel> =
        params.narrative_level.as_deref().and_then(|l| match l {
            "Story" => Some(NarrativeLevel::Story),
            "Arc" => Some(NarrativeLevel::Arc),
            "Sequence" => Some(NarrativeLevel::Sequence),
            "Scene" => Some(NarrativeLevel::Scene),
            "Beat" => Some(NarrativeLevel::Beat),
            "Event" => Some(NarrativeLevel::Event),
            _ => None,
        });

    let situations: Vec<Situation> = match (&params.narrative_id, &level_filter) {
        (Some(nid), Some(lv)) => crate::synth::emit::filter_synthetic_situations(
            hg.list_situations_by_narrative(nid).unwrap_or_default(),
            include_synth,
        )
        .into_iter()
        .filter(|s| s.narrative_level == *lv)
        .take(limit)
        .collect(),
        (Some(nid), None) => {
            let mut v = crate::synth::emit::filter_synthetic_situations(
                hg.list_situations_by_narrative(nid).unwrap_or_default(),
                include_synth,
            );
            v.truncate(limit);
            v
        }
        (None, Some(lv)) => {
            let mut v = crate::synth::emit::filter_synthetic_situations(
                hg.list_situations_by_level(*lv).unwrap_or_default(),
                include_synth,
            );
            v.truncate(limit);
            v
        }
        (None, None) => crate::synth::emit::filter_synthetic_situations(
            hg.list_situations_by_maturity(MaturityLevel::Candidate)
                .unwrap_or_default(),
            include_synth,
        )
        .into_iter()
        .take(limit)
        .collect(),
    };

    if let Some(ref cfg) = fuzzy_cfg {
        json_ok(&serde_json::json!({
            "data": situations,
            "total": situations.len(),
            "fuzzy_config": cfg,
        }))
    } else {
        json_ok(&serde_json::json!({ "data": situations, "total": situations.len() }))
    }
}

/// Query params for GET /situations/:id — Fuzzy Sprint Phase 4 opt-in.
#[derive(Deserialize, Default)]
pub struct GetSituationParams {
    #[serde(default)]
    pub tnorm: Option<String>,
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// GET /situations/:id
pub async fn get_situation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    axum::extract::Query(params): axum::extract::Query<GetSituationParams>,
) -> impl IntoResponse {
    let fuzzy_cfg = match resolve_fuzzy_from_strings(
        params.tnorm.as_deref(),
        params.aggregator.as_deref(),
    ) {
        Ok(c) => c,
        Err(e) => return error_response(e).into_response(),
    };

    match state.hypergraph.get_situation(&id) {
        Ok(sit) => {
            if let Some(ref cfg) = fuzzy_cfg {
                let mut val = match serde_json::to_value(&sit) {
                    Ok(v) => v,
                    Err(e) => {
                        return error_response(TensaError::Serialization(e.to_string()))
                            .into_response()
                    }
                };
                if let serde_json::Value::Object(ref mut map) = val {
                    map.insert(
                        "fuzzy_config".to_string(),
                        serde_json::to_value(cfg).unwrap_or(serde_json::Value::Null),
                    );
                }
                json_ok(&val)
            } else {
                json_ok(&sit)
            }
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// Patch body for `PUT /situations/:id`.
///
/// Uses `Option<Option<T>>` for nullable fields so that `{}` (omitted), `{"field": null}`
/// (cleared), and `{"field": "value"}` (set) are all representable.
#[derive(Deserialize, Default)]
pub struct SituationPatch {
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub name: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub narrative_level: Option<NarrativeLevel>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub narrative_id: Option<Option<String>>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub raw_content: Option<Vec<ContentBlock>>,
    #[serde(default)]
    pub temporal: Option<AllenInterval>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub spatial: Option<Option<SpatialAnchor>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub discourse: Option<Option<DiscourseAnnotation>>,
    /// Sprint P4.2 enrichment field: game-theoretic frame for the scene.
    /// `Some(Some(_))` sets, `Some(None)` clears, `None` preserves.
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub game_structure: Option<Option<GameStructure>>,
    /// Sprint P4.2 enrichment: factual / known-deterministic content.
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub deterministic: Option<Option<serde_json::Value>>,
    /// Sprint P4.2 enrichment: uncertain / probabilistic outcomes.
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub probabilistic: Option<Option<serde_json::Value>>,
    // Writer scene-schema fields (Sprint W7)
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub synopsis: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub manuscript_order: Option<Option<u32>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub parent_situation_id: Option<Option<Uuid>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub label: Option<Option<String>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub status: Option<Option<String>>,
    #[serde(default)]
    pub keywords: Option<Vec<String>>,
    /// Free-form properties patch (object). Merged into the existing
    /// `Situation.properties` JSON Value (existing keys overwritten, others
    /// preserved). Other Situation primitive fields use the typed slots above.
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
}

/// Distinguishes "field omitted" from "field set to null". Needed because serde's
/// `#[serde(default)]` alone collapses `Option<Option<T>>` so you can't clear a field.
pub(crate) fn deserialize_optional_option<'de, T, D>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error>
where
    T: serde::Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// Maximum number of content blocks accepted in a single situation patch.
pub const MAX_RAW_CONTENT_BLOCKS: usize = 500;

/// Validate a `raw_content` payload before it hits the hypergraph. Rejects
/// empty arrays (callers should `delete_situation` instead) and blank-string
/// Text/Dialogue/Observation blocks (almost always a caller prompt-formatting
/// bug). Called from both the typed PUT handler and the MCP backend so writes
/// that arrive via either route fail the same way.
pub(crate) fn validate_raw_content(blocks: &[ContentBlock]) -> Result<(), TensaError> {
    if blocks.is_empty() {
        return Err(TensaError::InvalidInput(
            "raw_content: empty array — use delete_situation to remove, or omit the field to preserve existing prose".into(),
        ));
    }
    if blocks.len() > MAX_RAW_CONTENT_BLOCKS {
        return Err(TensaError::InvalidQuery(format!(
            "raw_content exceeds maximum {} blocks",
            MAX_RAW_CONTENT_BLOCKS
        )));
    }
    for (i, b) in blocks.iter().enumerate() {
        let requires_prose = matches!(
            b.content_type,
            crate::types::ContentType::Text
                | crate::types::ContentType::Dialogue
                | crate::types::ContentType::Observation
        );
        if requires_prose && b.content.trim().is_empty() {
            return Err(TensaError::InvalidInput(format!(
                "raw_content[{i}] ({:?}) has empty content",
                b.content_type
            )));
        }
    }
    Ok(())
}

/// PUT /situations/:id — Update situation fields (name, description, narrative_level,
/// narrative_id, confidence, raw_content, temporal, spatial).
pub async fn update_situation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(patch): Json<SituationPatch>,
) -> impl IntoResponse {
    if let Some(ref blocks) = patch.raw_content {
        if let Err(e) = validate_raw_content(blocks) {
            return error_response(e).into_response();
        }
    }
    if let Some(c) = patch.confidence {
        if !(0.0..=1.0).contains(&c) {
            return error_response(TensaError::InvalidQuery(
                "confidence must be between 0.0 and 1.0".into(),
            ))
            .into_response();
        }
    }
    if let Some(Some(parent_id)) = patch.parent_situation_id {
        if parent_id == id {
            return error_response(TensaError::InvalidQuery(
                "parent_situation_id cannot equal the situation's own id".into(),
            ))
            .into_response();
        }
        if let Err(e) = crate::writer::scene::check_parent_cycle(&state.hypergraph, &id, &parent_id)
        {
            return error_response(e).into_response();
        }
    }

    match state.hypergraph.update_situation(&id, |s| {
        if let Some(name) = patch.name {
            s.name = name;
        }
        if let Some(description) = patch.description {
            s.description = description;
        }
        if let Some(level) = patch.narrative_level {
            s.narrative_level = level;
        }
        if let Some(narrative_id) = patch.narrative_id {
            s.narrative_id = narrative_id;
        }
        if let Some(c) = patch.confidence {
            s.confidence = c;
        }
        if let Some(raw_content) = patch.raw_content {
            s.raw_content = raw_content;
        }
        if let Some(temporal) = patch.temporal {
            s.temporal = temporal;
        }
        if let Some(spatial) = patch.spatial {
            s.spatial = spatial;
        }
        if let Some(discourse) = patch.discourse {
            s.discourse = discourse;
        }
        if let Some(game_structure) = patch.game_structure {
            s.game_structure = game_structure;
        }
        if let Some(deterministic) = patch.deterministic {
            s.deterministic = deterministic;
        }
        if let Some(probabilistic) = patch.probabilistic {
            s.probabilistic = probabilistic;
        }
        if let Some(synopsis) = patch.synopsis {
            s.synopsis = synopsis;
        }
        if let Some(order) = patch.manuscript_order {
            s.manuscript_order = order;
        }
        if let Some(parent) = patch.parent_situation_id {
            s.parent_situation_id = parent;
        }
        if let Some(label) = patch.label {
            s.label = label;
        }
        if let Some(status) = patch.status {
            s.status = status;
        }
        if let Some(keywords) = patch.keywords {
            s.keywords = keywords;
        }
        if let Some(props_patch) = patch.properties {
            if let Some(obj) = props_patch.as_object() {
                if !s.properties.is_object() {
                    s.properties = serde_json::Value::Object(serde_json::Map::new());
                }
                if let Some(existing) = s.properties.as_object_mut() {
                    for (k, v) in obj {
                        existing.insert(k.clone(), v.clone());
                    }
                }
            }
        }
    }) {
        Ok(sit) => json_ok(&sit),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /situations/:id
pub async fn delete_situation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.delete_situation(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /situations/:id/restore — Restore a soft-deleted situation.
pub async fn restore_situation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.restore_situation(&id) {
        Ok(situation) => json_ok(&situation),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /participations
pub async fn create_participation(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let participation: Participation = match serde_json::from_value(body) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response()
        }
    };
    match state.hypergraph.add_participant(participation) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"status": "ok"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /situations/:id/participants
pub async fn get_participants(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.get_participants_for_situation(&id) {
        Ok(participants) => json_ok(&participants),
        Err(e) => error_response(e).into_response(),
    }
}

/// Patch shape for PUT /situations/:sid/participants/:eid/:seq.
/// Sprint P4.2 retro-enrichment: lets a caller fill in `info_set`, `payoff`,
/// `action`, or `role` on an existing participation without re-creating it.
/// Each field uses the `Some(Some(_))` set / `Some(None)` clear / `None` preserve
/// pattern documented on `SituationPatch`.
#[derive(Deserialize, Default)]
pub struct ParticipationPatch {
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub info_set: Option<Option<InfoSet>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub payoff: Option<Option<serde_json::Value>>,
    #[serde(default, deserialize_with = "deserialize_optional_option")]
    pub action: Option<Option<String>>,
    #[serde(default)]
    pub role: Option<Role>,
}

/// PUT /situations/:sid/participants/:eid/:seq — Update a participation in place.
///
/// Sprint P4.2 retro-enrichment endpoint: lets a script (or the
/// `tensa-narrative-llm` skill's `enrichment-backfill` analysis-type) populate
/// `info_set` (knows_before / learns / reveals / beliefs_about_others) and
/// `payoff` on participations from archive imports that skipped the enrichment
/// pass. Without this surface, `Participation.info_set` and `payoff` could only
/// be set at create-time, leaving the StyleProfile axes (Info R₀, Power
/// Asymmetry, Deception, Late Revelation) permanently zeroed for those imports.
pub async fn update_participation(
    State(state): State<Arc<AppState>>,
    Path((situation_id, entity_id, seq)): Path<(Uuid, Uuid, u16)>,
    Json(patch): Json<ParticipationPatch>,
) -> impl IntoResponse {
    let pairs = match state
        .hypergraph
        .get_participations_for_pair(&entity_id, &situation_id)
    {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    let mut existing = match pairs.into_iter().find(|p| p.seq == seq) {
        Some(p) => p,
        None => {
            return error_response(TensaError::NotFound(format!(
                "participation (situation={situation_id}, entity={entity_id}, seq={seq}) not found"
            )))
            .into_response()
        }
    };
    if let Some(info_set) = patch.info_set {
        existing.info_set = info_set;
    }
    if let Some(payoff) = patch.payoff {
        existing.payoff = payoff;
    }
    if let Some(action) = patch.action {
        existing.action = action;
    }
    if let Some(role) = patch.role {
        existing.role = role;
    }
    match state.hypergraph.update_participation(&existing) {
        Ok(()) => json_ok(&existing),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /entities/:id/situations
///
/// Returns the full situations an entity participates in. The underlying
/// `Hypergraph::get_situations_for_entity` is participation-typed for historical
/// reasons, so this handler dereferences each participation into its situation
/// record, deduplicating when an entity holds multiple roles in the same
/// situation (multi-role participation, Sprint P4.1).
pub async fn get_entity_situations(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.hypergraph.get_situations_for_entity(&id) {
        Ok(participations) => {
            let mut seen = std::collections::HashSet::new();
            let mut situations = Vec::with_capacity(participations.len());
            for p in participations {
                if !seen.insert(p.situation_id) {
                    continue;
                }
                if let Ok(sit) = state.hypergraph.get_situation(&p.situation_id) {
                    situations.push(sit);
                }
            }
            json_ok(&situations)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// Query parameters for listing validation queue items.
#[derive(Deserialize)]
pub struct QueueListParams {
    pub limit: Option<usize>,
    /// Optional narrative scope. When provided, only items whose
    /// `narrative_id` matches the filter are returned. Items with no
    /// `narrative_id` (legacy entries from before the field was added)
    /// are excluded when this filter is set.
    pub narrative_id: Option<String>,
}

/// GET /validation-queue
pub async fn list_validation_queue(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<QueueListParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50);
    match state.validation_queue.list_pending(limit) {
        Ok(items) => {
            let filtered = match params.narrative_id {
                Some(nid) => items
                    .into_iter()
                    .filter(|i| i.narrative_id.as_deref() == Some(nid.as_str()))
                    .collect(),
                None => items,
            };
            json_ok(&filtered)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /validation-queue/:id
pub async fn get_queue_item(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.validation_queue.get(&id) {
        Ok(item) => json_ok(&item),
        Err(e) => error_response(e).into_response(),
    }
}

/// Request body for approval/rejection.
#[derive(Deserialize)]
pub struct ReviewRequest {
    pub reviewer: String,
    pub notes: Option<String>,
}

/// POST /validation-queue/:id/approve
pub async fn approve_queue_item(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(body): Json<ReviewRequest>,
) -> impl IntoResponse {
    match state.validation_queue.approve(&id, &body.reviewer) {
        Ok(item) => json_ok(&item),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /validation-queue/:id/reject
pub async fn reject_queue_item(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(body): Json<ReviewRequest>,
) -> impl IntoResponse {
    match state
        .validation_queue
        .reject(&id, &body.reviewer, body.notes)
    {
        Ok(item) => json_ok(&item),
        Err(e) => error_response(e).into_response(),
    }
}

/// Request body for edit-and-approve.
#[derive(Deserialize)]
pub struct EditRequest {
    pub reviewer: String,
    pub edited_data: serde_json::Value,
}

/// POST /validation-queue/:id/edit
pub async fn edit_queue_item(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(body): Json<EditRequest>,
) -> impl IntoResponse {
    match state
        .validation_queue
        .edit_and_approve(&id, &body.reviewer, body.edited_data)
    {
        Ok(item) => json_ok(&item),
        Err(e) => error_response(e).into_response(),
    }
}

/// Request body for the /query endpoint.
#[derive(Deserialize)]
pub struct QueryRequest {
    pub query: String,
}

/// POST /infer — Parse + plan + execute a TensaQL INFER / DISCOVER query and
/// submit the resulting job to the inference queue. Returns `{job_id, status}`.
///
/// This endpoint is what the HTTP MCP backend's `submit_inference_query` calls.
/// `POST /query` alone only runs the executor and returns the descriptor row;
/// this route adds the submission step so HTTP and embedded backends behave
/// identically.
pub async fn submit_inference(
    State(state): State<Arc<AppState>>,
    Json(body): Json<QueryRequest>,
) -> impl IntoResponse {
    use crate::inference::dispatch;

    // Parse + plan + execute — same path as /query but we need the raw rows.
    let stmt = match parser::parse_statement(&body.query) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let explain = match &stmt {
        parser::TensaStatement::Query(q) => q.explain,
        _ => false,
    };
    let plan = match planner::plan_statement(&stmt) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    let tree = match state
        .interval_tree
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()))
    {
        Ok(t) => t,
        Err(e) => return error_response(e).into_response(),
    };
    let emb_guard = state.embedder.read().unwrap();
    let results = match executor::execute_full(
        &plan,
        &state.hypergraph,
        &*tree,
        explain,
        None,
        emb_guard.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => return error_response(e).into_response(),
    };

    // If the first row is a job descriptor, submit it. Otherwise return raw
    // results — callers pointed at /infer with a plain MATCH query still get
    // something useful back.
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    if let Some(row) = results.first() {
        if let Ok(job_type) = dispatch::infer_type_from_row(row) {
            let target_id = dispatch::extract_target_id(row).unwrap_or_else(Uuid::now_v7);
            let parameters = dispatch::extract_parameters(row).unwrap_or(serde_json::json!({}));

            let job = InferenceJob {
                id: Uuid::now_v7().to_string(),
                job_type,
                target_id,
                parameters,
                priority: JobPriority::Normal,
                status: JobStatus::Pending,
                estimated_cost_ms: 0,
                created_at: chrono::Utc::now(),
                started_at: None,
                completed_at: None,
                error: None,
            };

            let job_id = job.id.clone();
            return match job_queue.submit(job) {
                Ok(returned_id) => (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({
                        "job_id": returned_id,
                        "status": "Pending",
                        "message": if returned_id == job_id {
                            "Job submitted to inference queue"
                        } else {
                            "Duplicate job \u{2014} returning existing in-flight job"
                        }
                    })),
                )
                    .into_response(),
                Err(e) => error_response(e).into_response(),
            };
        }
    }

    // Not a job descriptor — behave like /query.
    json_ok(&results)
}

/// POST /query — Execute a TensaQL query or DML mutation.
pub async fn execute_query(
    State(state): State<Arc<AppState>>,
    Json(body): Json<QueryRequest>,
) -> impl IntoResponse {
    let stmt = match parser::parse_statement(&body.query) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let explain = match &stmt {
        parser::TensaStatement::Query(q) => q.explain,
        _ => false,
    };
    let plan = match planner::plan_statement(&stmt) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    let tree = state
        .interval_tree
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    let tree = match tree {
        Ok(t) => t,
        Err(e) => return error_response(e).into_response(),
    };
    // Detect if the plan needs vector index or extractor
    let needs_near = plan
        .steps
        .iter()
        .any(|s| matches!(s, planner::PlanStep::VectorNear { .. }));
    let needs_ask = plan
        .steps
        .iter()
        .any(|s| matches!(s, planner::PlanStep::AskLlm { .. }));
    let needs_vector = needs_near || needs_ask;
    let vi_guard = if needs_vector {
        state.vector_index.as_ref().and_then(|vi| vi.read().ok())
    } else {
        None
    };
    let vi_ref = vi_guard.as_deref();
    let emb_guard = state.embedder.read().unwrap();
    let emb_ref = if needs_vector {
        emb_guard.as_deref()
    } else {
        None
    };

    if needs_ask {
        let inf_guard = state.inference_extractor.read().unwrap();
        let ext_guard = state.extractor.read().unwrap();
        let ext_ref = inf_guard.as_deref().or(ext_guard.as_deref());
        match executor::execute_full_with_extractor(
            &plan,
            &state.hypergraph,
            &*tree,
            explain,
            vi_ref,
            emb_ref,
            ext_ref,
        ) {
            Ok(results) => json_ok(&results),
            Err(e) => error_response(e).into_response(),
        }
    } else {
        match executor::execute_full(&plan, &state.hypergraph, &*tree, explain, vi_ref, emb_ref) {
            Ok(results) => json_ok(&results),
            Err(e) => error_response(e).into_response(),
        }
    }
}

// ─── Prompt Tuning Endpoints (GraphRAG Sprint 1) ────────────

/// POST /prompts/tune — Trigger prompt auto-tuning for a narrative.
pub async fn tune_prompts(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let narrative_id = match body.get("narrative_id").and_then(|v| v.as_str()) {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing or empty 'narrative_id' field"})),
            )
                .into_response()
        }
    };
    let ext_guard = state.extractor.read().unwrap();
    let extractor = match ext_guard.as_ref() {
        Some(e) => e.as_ref(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "No LLM configured"})),
            )
                .into_response()
        }
    };
    match crate::ingestion::prompt_tuning::tune_prompts(
        state.hypergraph.store(),
        extractor,
        &state.hypergraph,
        &narrative_id,
    ) {
        Ok(tuned) => json_ok(&tuned),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /prompts — List all tuned prompts.
pub async fn list_prompts(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match crate::ingestion::prompt_tuning::list_tuned_prompts(state.hypergraph.store()) {
        Ok(prompts) => json_ok(&prompts),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /prompts/:narrative_id — Get tuned prompt for a narrative.
pub async fn get_prompt(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(narrative_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    match crate::ingestion::prompt_tuning::get_tuned_prompt(state.hypergraph.store(), &narrative_id)
    {
        Ok(Some(prompt)) => json_ok(&prompt),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "No tuned prompt for this narrative"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /prompts/:narrative_id — Manually update a tuned prompt.
pub async fn update_prompt(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(narrative_id): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let prompt_text = body
        .get("prompt_text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let domain_description = body
        .get("domain_description")
        .and_then(|v| v.as_str())
        .unwrap_or("Custom domain")
        .to_string();
    let entity_types: Vec<String> = body
        .get("entity_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let tuned = crate::ingestion::prompt_tuning::TunedPrompt {
        narrative_id: narrative_id.clone(),
        prompt_text,
        domain_description,
        entity_types,
        generated_at: chrono::Utc::now(),
        model: Some("manual".to_string()),
    };
    match crate::ingestion::prompt_tuning::store_tuned_prompt(state.hypergraph.store(), &tuned) {
        Ok(()) => json_ok(&tuned),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /prompts/:narrative_id — Delete a tuned prompt.
pub async fn delete_prompt(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(narrative_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    match crate::ingestion::prompt_tuning::delete_tuned_prompt(
        state.hypergraph.store(),
        &narrative_id,
    ) {
        Ok(true) => json_ok(&serde_json::json!({"deleted": true})),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "No tuned prompt for this narrative"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Vector Store Settings (GraphRAG Sprint 6) ─────────────

/// GET /settings/vector-store — Get current vector store configuration.
pub async fn get_vector_store_config(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    // Currently always in-memory; return the default config
    let config = crate::ingestion::vector_store::VectorStoreConfig::default();
    json_ok(&config)
}

// ─── RAG Ask Endpoint (Sprint RAG-3) ────────────────────────

/// POST /ask — Execute a RAG question-answering query.
pub async fn ask_question(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let question = match body.get("question").and_then(|v| v.as_str()) {
        Some(q) if !q.is_empty() => q.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing or empty 'question' field"})),
            )
                .into_response()
        }
    };
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let mode_str = body
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("hybrid");
    let mode = crate::query::rag_config::RetrievalMode::from_str_or_default(mode_str);
    let response_type = body
        .get("response_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let suggest = body
        .get("suggest")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let debug = body.get("debug").and_then(|v| v.as_bool()).unwrap_or(false);
    // EATH Phase 3 — opt-in to surface synthetic records in RAG context.
    // Default false: the RAG context assembly downstream will filter
    // synthetic entities/situations via `crate::synth::emit::filter_*`
    // when this flag is unset. (See `RagConfig.include_synthetic`.)
    let include_synthetic = body
        .get("include_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let _ = include_synthetic; // wired for surface; underlying RAG filter at Phase 12.5.
    let budget = state.rag_config.read().unwrap().budget.clone();
    // Prefer dedicated inference LLM, fall back to ingestion extractor
    let inf_guard = state.inference_extractor.read().unwrap();
    let ext_guard = state.extractor.read().unwrap();
    let extractor = match inf_guard.as_deref().or(ext_guard.as_deref()) {
        Some(e) => e,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "No LLM configured. Set a Query LLM in Settings, or configure an Ingestion LLM."})),
            )
                .into_response()
        }
    };
    let vi_guard = state.vector_index.as_ref().and_then(|vi| vi.read().ok());
    let vi_ref = vi_guard.as_deref();
    let emb_guard = state.embedder.read().unwrap();
    let emb_ref = emb_guard.as_deref();

    let rr_ref = state.reranker.as_deref();
    let ask_fn = if debug {
        crate::query::rag::execute_ask_debug
    } else {
        crate::query::rag::execute_ask
    };
    match ask_fn(
        &question,
        narrative_id.as_deref(),
        &mode,
        &budget,
        &state.hypergraph,
        vi_ref,
        emb_ref,
        extractor,
        rr_ref,
        response_type.as_deref(),
        suggest,
        None,
    ) {
        Ok(answer) => json_ok(&answer),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Inference Job Endpoints (Phase 2) ───────────────────────

/// Request body for POST /jobs.
///
/// `target_id` is optional. Narrative-scoped jobs (`ArcClassification`,
/// `PatternMining`, `CentralityAnalysis`, `StyleProfile`, ...) key on
/// `parameters.narrative_id` and ignore `target_id`, so the route does not
/// require it. When omitted, a throwaway `Uuid::now_v7()` is generated —
/// matching the embedded MCP path at `src/mcp/embedded.rs:submit_inference_query`.
#[derive(Deserialize)]
pub struct SubmitJobRequest {
    pub job_type: InferenceJobType,
    #[serde(default)]
    pub target_id: Option<Uuid>,
    pub parameters: Option<serde_json::Value>,
    pub priority: Option<JobPriority>,
}

/// POST /jobs — Submit an inference job.
pub async fn submit_job(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SubmitJobRequest>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    let job = InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type: body.job_type,
        target_id: body.target_id.unwrap_or_else(Uuid::now_v7),
        parameters: body.parameters.unwrap_or(serde_json::json!({})),
        priority: body.priority.unwrap_or(JobPriority::Normal),
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };

    match job_queue.submit(job) {
        Ok(id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"job_id": id, "status": "Pending"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /jobs/:id — Get job status.
pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    match job_queue.get_job(&id) {
        Ok(job) => json_ok(&job),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /jobs/:id/result — Get job result.
pub async fn get_job_result(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    match job_queue.get_result(&id) {
        Ok(result) => json_ok(&result),
        Err(e) => error_response(e).into_response(),
    }
}

/// Query parameters for GET /jobs
#[derive(Deserialize)]
pub struct JobListParams {
    pub target_id: Option<Uuid>,
    pub narrative_id: Option<String>,
    pub limit: Option<usize>,
}

/// GET /jobs — List jobs (optionally filtered by target or narrative).
pub async fn list_jobs(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<JobListParams>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    let limit = params.limit.unwrap_or(200);

    let result = if let Some(target_id) = params.target_id {
        job_queue.list_by_target(&target_id)
    } else {
        job_queue.list_recent(limit)
    };

    match result {
        Ok(jobs) => {
            // Server-side narrative_id filter (checks parameters.narrative_id)
            let filtered = if let Some(ref nid) = params.narrative_id {
                jobs.into_iter()
                    .filter(|j| {
                        j.parameters.get("narrative_id").and_then(|v| v.as_str())
                            == Some(nid.as_str())
                    })
                    .collect()
            } else {
                jobs
            };
            json_ok(&filtered)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// Query params for DELETE /jobs/:id.
#[derive(serde::Deserialize, Default)]
pub struct CancelJobParams {
    /// When `true`, also abort jobs stuck in `Running` — useful for reaping
    /// zombies left over from a server crash / binary swap. The worker pool
    /// owns `Running` status normally, but if the owning worker is gone the
    /// job will never transition without intervention.
    #[serde(default)]
    pub force: bool,
}

/// DELETE /jobs/:id?force=bool — Cancel a pending (or force-abort a running) job.
pub async fn cancel_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<CancelJobParams>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    let result = if params.force {
        job_queue.force_abort(&id)
    } else {
        job_queue.cancel(&id)
    };
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Ingestion Endpoint ─────────────────────────────────────

/// Request body for POST /ingest.
#[derive(Deserialize)]
pub struct IngestRequest {
    pub text: String,
    pub narrative_id: Option<String>,
    pub source_name: Option<String>,
    pub source_type: Option<String>,
    pub auto_commit_threshold: Option<f32>,
    pub review_threshold: Option<f32>,
    /// Enable step-2 enrichment (beliefs, game structures, discourse, info sets).
    #[serde(default)]
    pub enrich: bool,
    /// Use SingleSession mode (entire text in one LLM context window).
    #[serde(default)]
    pub single_session: bool,
    /// Enable verbose debug logging for this ingestion job.
    #[serde(default)]
    pub debug: bool,
    /// Override the ingestion mode for this request (selects domain-specific extraction prompt).
    /// If not set, uses the global `ingestion_mode` from IngestionConfig.
    #[serde(default)]
    pub ingestion_mode: Option<crate::ingestion::config::IngestionMode>,
}

/// POST /ingest — Submit an async ingestion job.
///
/// Returns immediately with `{ job_id, status }`. The pipeline runs in the background.
/// Poll `GET /ingest/jobs/:id` for progress and results.
pub async fn ingest_text(
    State(state): State<Arc<AppState>>,
    Json(body): Json<IngestRequest>,
) -> impl IntoResponse {
    use crate::ingestion::jobs::{IngestionJob, IngestionProgress, PassMode};

    // Build extractors from ingestion config (pass1 / pass2), falling back to
    // the global extractor set from .env / Settings > LLM.
    let (
        extractor,
        pass2_extractor,
        pass_mode,
        pass1_concurrency,
        chunker_config,
        strip_boilerplate,
        ic_single_session,
        ic_session_max_ctx,
        ic_enrich,
    ) = {
        let ic = state
            .ingestion_config
            .read()
            .unwrap_or_else(|e| e.into_inner());

        // Per-request override takes priority over global config.
        let ingest_mode = body.ingestion_mode.as_ref().unwrap_or(&ic.ingestion_mode);
        let mode_prompt = crate::ingestion::llm::system_prompt_for_mode(ingest_mode);

        let pass1: Option<Arc<dyn crate::ingestion::llm::NarrativeExtractor>> =
            ic.pass1.build_extractor_with_prompt(mode_prompt);
        let pass2: Option<Arc<dyn crate::ingestion::llm::NarrativeExtractor>> =
            ic.pass2.build_extractor();

        let extractor = match pass1 {
            Some(e) => e,
            None => {
                // Fall back to the global extractor
                let guard = state
                    .extractor
                    .read()
                    .map_err(|_| TensaError::Internal("Lock poisoned".into()));
                match guard {
                    Ok(ref opt) => match opt.as_ref() {
                        Some(e) => Arc::clone(e),
                        None => {
                            return error_response(TensaError::LlmError(
                                "No LLM provider configured. Set Pass 1 in Settings > Ingestion, or configure a global LLM.".into(),
                            ))
                            .into_response()
                        }
                    },
                    Err(e) => return error_response(e).into_response(),
                }
            }
        };

        let mode = if ic.mode == crate::ingestion::config::PipelineMode::SingleSession {
            PassMode::Session
        } else if ic.mode == crate::ingestion::config::PipelineMode::Multi && pass2.is_some() {
            PassMode::Dual
        } else {
            PassMode::Single
        };

        let is_session = ic.mode == crate::ingestion::config::PipelineMode::SingleSession;
        let max_ctx = ic.session_max_context_tokens;
        (
            extractor,
            pass2,
            mode,
            ic.pass1_concurrency.max(1),
            ic.chunker_config(),
            ic.strip_boilerplate,
            is_session,
            max_ctx,
            ic.enrich,
        )
    };

    let narrative_id = body.narrative_id.clone();
    let source_name = body
        .source_name
        .clone()
        .unwrap_or_else(|| "api-upload".into());
    let text = body.text.clone();
    let text_preview = text.chars().take(200).collect::<String>();

    // Determine model name and session capability for debugging
    let extractor_model = extractor.model_name().unwrap_or_else(|| "unknown".into());
    let extractor_session_capable = extractor.as_session().is_some();
    let is_single_session = body.single_session || ic_single_session;

    tracing::info!(
        model = %extractor_model,
        session_capable = extractor_session_capable,
        single_session_requested = is_single_session,
        pass_mode = ?pass_mode,
        text_len = text.len(),
        "Ingestion job starting"
    );

    // Create job record
    let job = IngestionJob {
        id: uuid::Uuid::now_v7().to_string(),
        status: crate::types::JobStatus::Pending,
        text_preview,
        text_length: text.len(),
        narrative_id: narrative_id.clone(),
        source_name: source_name.clone(),
        pass_mode,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
        report: None,
        model: Some(extractor_model.clone()),
        single_session_requested: is_single_session,
        session_capable: extractor_session_capable,
        effective_mode: None,
        enrich: body.enrich || ic_enrich,
        parent_job_id: None,
    };
    let job_id = match state.ingestion_jobs.submit(job) {
        Ok(id) => id,
        Err(e) => return error_response(e).into_response(),
    };

    // Persist the full source text for retry support
    let _ = state.ingestion_jobs.store_source_text(&job_id, &text);

    // Set up ephemeral progress tracker and cancel flag
    let progress = Arc::new(std::sync::Mutex::new(IngestionProgress::new(pass_mode)));
    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        if let Ok(mut map) = state.ingestion_progress.lock() {
            map.insert(job_id.clone(), Arc::clone(&progress));
        }
        if let Ok(mut map) = state.ingestion_cancel_flags.lock() {
            map.insert(job_id.clone(), Arc::clone(&cancel_flag));
        }
    }

    let config = crate::ingestion::pipeline::PipelineConfig {
        chunker: chunker_config,
        auto_commit_threshold: body.auto_commit_threshold.unwrap_or(0.8),
        review_threshold: body.review_threshold.unwrap_or(0.3),
        source_id: narrative_id.clone().unwrap_or_else(|| "unknown".into()),
        source_type: body.source_type.clone().unwrap_or_else(|| "text".into()),
        narrative_id: narrative_id.clone(),
        job_id: Some(job_id.clone()),
        concurrency: pass1_concurrency,
        strip_boilerplate,
        enrich: body.enrich || ic_enrich,
        single_session: body.single_session || ic_single_session,
        session_max_context_tokens: ic_session_max_ctx,
        debug: body.debug,
        cascade_mode: Default::default(),
        post_ingest_mamdani_rule_id: None,
    };

    let hg = Arc::new(crate::hypergraph::Hypergraph::new(
        state.hypergraph.store_arc(),
    ));
    let queue = Arc::new(crate::ingestion::queue::ValidationQueue::new(
        state.hypergraph.store_arc(),
    ));
    let embedder = state.embedder.read().unwrap().clone();
    let vector_index = state.vector_index.clone();
    let ingestion_jobs = Arc::clone(&state.ingestion_jobs);
    let progress_map_ref = {
        // We need the whole state for cleanup — use a weak-like pattern
        let state = Arc::clone(&state);
        state
    };
    let jid = job_id.clone();
    let vi_for_save = vector_index.clone();

    // Mark running and spawn background task
    let _ = state.ingestion_jobs.mark_running(&job_id);

    tokio::task::spawn_blocking(move || {
        let mut pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
            hg.clone(),
            extractor,
            embedder,
            vector_index,
            queue,
            config,
        )
        .with_job_queue(Arc::clone(&ingestion_jobs))
        .with_progress(progress)
        .with_cancel_flag(cancel_flag);

        if let Some(p2) = pass2_extractor {
            pipeline = pipeline.with_pass2_extractor(p2);
        }

        match pipeline.ingest_text(&text, &source_name) {
            Ok(report) => {
                if report.cancelled {
                    let _ = ingestion_jobs.mark_failed(&jid, "Cancelled by user");
                    // Still store the partial report so rollback can see created IDs
                    let _ = ingestion_jobs.store_partial_report(&jid, report);
                } else {
                    // Update narrative counts
                    if let Some(ref nid) = narrative_id {
                        let registry =
                            crate::narrative::registry::NarrativeRegistry::new(hg.store_arc());
                        let _ = registry.update(nid, |n| {
                            n.entity_count += report.entities_created;
                            n.situation_count += report.situations_created;
                        });
                    }
                    let _ = ingestion_jobs.mark_completed(&jid, report);
                }
            }
            Err(e) => {
                let _ = ingestion_jobs.mark_failed(&jid, &e.to_string());
            }
        }

        // Persist vector index to KV after ingestion
        if let Some(vi) = &vi_for_save {
            if let Ok(idx) = vi.read() {
                if let Err(e) = idx.save(hg.store()) {
                    tracing::warn!("Failed to save vector index: {}", e);
                }
            }
        }

        // Clean up ephemeral progress and cancel flag
        if let Ok(mut map) = progress_map_ref.ingestion_progress.lock() {
            map.remove(&jid);
        }
        if let Ok(mut map) = progress_map_ref.ingestion_cancel_flags.lock() {
            map.remove(&jid);
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "job_id": job_id, "status": "Running" })),
    )
        .into_response()
}

/// GET /ingest/jobs — List all ingestion jobs (newest first).
pub async fn list_ingestion_jobs(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<QueueListParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50);
    match state.ingestion_jobs.list_all(limit) {
        Ok(jobs) => json_ok(&jobs),
        Err(e) => error_response(e).into_response(),
    }
}

/// Query parameters for GET /ingest/jobs/:id.
#[derive(Deserialize)]
pub struct JobDetailParams {
    /// Include LLM call log summary (expensive KV scan). Default: false.
    #[serde(default)]
    pub debug: bool,
}

/// GET /ingest/jobs/:id — Get ingestion job status + live progress + debug info.
pub async fn get_ingestion_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<JobDetailParams>,
) -> impl IntoResponse {
    match state.ingestion_jobs.get_job(&job_id) {
        Ok(job) => {
            // Merge in live progress if job is running
            let progress = if job.status == crate::types::JobStatus::Running {
                state
                    .ingestion_progress
                    .lock()
                    .ok()
                    .and_then(|map| map.get(&job_id).cloned())
                    .and_then(|p| p.lock().ok().map(|g| g.clone()))
            } else {
                None
            };

            // Include LLM call log summary only when ?debug=true (avoids expensive KV scan on every poll)
            let llm_log_summary = if params.debug && job.status != crate::types::JobStatus::Running
            {
                state
                    .ingestion_jobs
                    .get_logs_for_job(&job_id)
                    .ok()
                    .map(|logs| {
                        let total_calls = logs.len();
                        let total_duration_ms: u64 = logs.iter().map(|l| l.duration_ms).sum();
                        let errors: Vec<String> = logs
                            .iter()
                            .filter_map(|l| {
                                l.parse_error
                                    .as_ref()
                                    .map(|e| format!("chunk {}: {}", l.chunk_index, e))
                            })
                            .collect();
                        let models_used: Vec<String> = logs
                            .iter()
                            .filter_map(|l| l.model.clone())
                            .collect::<std::collections::HashSet<_>>()
                            .into_iter()
                            .collect();
                        let endpoints_used: Vec<String> = logs
                            .iter()
                            .filter_map(|l| l.endpoint.clone())
                            .collect::<std::collections::HashSet<_>>()
                            .into_iter()
                            .collect();
                        serde_json::json!({
                            "total_calls": total_calls,
                            "total_duration_ms": total_duration_ms,
                            "parse_errors": errors,
                            "models_used": models_used,
                            "endpoints_used": endpoints_used,
                        })
                    })
            } else {
                None
            };

            json_ok(&serde_json::json!({
                "id": job.id,
                "status": job.status,
                "text_preview": job.text_preview,
                "text_length": job.text_length,
                "narrative_id": job.narrative_id,
                "source_name": job.source_name,
                "pass_mode": job.pass_mode,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error": job.error,
                "report": job.report,
                "progress": progress,
                "model": job.model,
                "single_session_requested": job.single_session_requested,
                "session_capable": job.session_capable,
                "effective_mode": job.effective_mode,
                "llm_log_summary": llm_log_summary,
            }))
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /ingest/jobs/:id — Cancel a running ingestion job.
pub async fn cancel_ingestion_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    // Set the cancel flag (pipeline checks this in its loops)
    let flagged = state
        .ingestion_cancel_flags
        .lock()
        .ok()
        .and_then(|map| map.get(&job_id).cloned())
        .map(|flag| {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
            true
        })
        .unwrap_or(false);

    if !flagged {
        // Job is not currently running — check status
        match state.ingestion_jobs.get_job(&job_id) {
            Ok(job) if job.status == crate::types::JobStatus::Pending => {
                let _ = state
                    .ingestion_jobs
                    .mark_failed(&job_id, "Cancelled by user");
            }
            Ok(job)
                if job.status == crate::types::JobStatus::Completed
                    || job.status == crate::types::JobStatus::Failed =>
            {
                // Already finished — return success (idempotent)
                return Json(serde_json::json!({
                    "cancelled": false,
                    "job_id": job_id,
                    "already_finished": true,
                    "status": job.status,
                }))
                .into_response();
            }
            Ok(_) => {
                // Running but no cancel flag — orphaned after server restart.
                // Mark as failed directly.
                let _ = state
                    .ingestion_jobs
                    .mark_failed(&job_id, "Cancelled by user");
            }
            Err(e) => return error_response(e).into_response(),
        }
    }

    Json(serde_json::json!({ "cancelled": true, "job_id": job_id })).into_response()
}

/// POST /ingest/jobs/:id/rollback — Delete entities and situations created by this job.
pub async fn rollback_ingestion_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let job = match state.ingestion_jobs.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };

    let report = match &job.report {
        Some(r) => r,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "No report available for rollback" })),
            )
                .into_response()
        }
    };

    let mut entities_deleted = 0usize;
    let mut situations_deleted = 0usize;

    for id in &report.created_entity_ids {
        if state.hypergraph.delete_entity(id).is_ok() {
            entities_deleted += 1;
        }
    }
    for id in &report.created_situation_ids {
        if state.hypergraph.delete_situation(id).is_ok() {
            situations_deleted += 1;
        }
    }

    // Update narrative counts
    if let Some(ref nid) = job.narrative_id {
        let registry =
            crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());
        let _ = registry.update(nid, |n| {
            n.entity_count = n.entity_count.saturating_sub(entities_deleted);
            n.situation_count = n.situation_count.saturating_sub(situations_deleted);
        });
    }

    json_ok(&serde_json::json!({
        "rolled_back": true,
        "entities_deleted": entities_deleted,
        "situations_deleted": situations_deleted,
    }))
}

/// POST /ingest/jobs/:id/retry — Re-run only the failed chunks from a completed/failed job.
///
/// Retrieves the stored source text, re-chunks it, and runs the pipeline with the same
/// config. Successfully ingested chunks are skipped via content-hash dedup, so only
/// failed/missing chunks get re-extracted. Returns a new job ID for the retry run.
pub async fn retry_ingestion_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    use crate::ingestion::jobs::{IngestionJob, IngestionProgress, PassMode};

    // Validate the original job exists and is not still running
    let orig_job = match state.ingestion_jobs.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };
    match orig_job.status {
        crate::types::JobStatus::Pending | crate::types::JobStatus::Running => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "error": "Job is still running — cannot retry yet" })),
            )
                .into_response();
        }
        _ => {}
    }

    // Retrieve the stored source text
    let text = match state.ingestion_jobs.get_source_text(&job_id) {
        Ok(Some(t)) => t,
        Ok(None) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Source text not available for this job (pre-retry feature). Re-submit the text instead."
                })),
            )
                .into_response();
        }
        Err(e) => return error_response(e).into_response(),
    };

    // Build extractors using same config as a fresh ingest
    let (
        extractor,
        pass2_extractor,
        pass_mode,
        pass1_concurrency,
        chunker_config,
        strip_boilerplate,
        ic_single_session,
        ic_session_max_ctx,
        ic_enrich,
    ) = {
        let ic = state
            .ingestion_config
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let pass1: Option<Arc<dyn crate::ingestion::llm::NarrativeExtractor>> =
            ic.pass1.build_extractor();
        let pass2: Option<Arc<dyn crate::ingestion::llm::NarrativeExtractor>> =
            ic.pass2.build_extractor();
        let extractor = match pass1 {
            Some(e) => e,
            None => {
                let guard = state
                    .extractor
                    .read()
                    .map_err(|_| TensaError::Internal("Lock poisoned".into()));
                match guard {
                    Ok(ref opt) => match opt.as_ref() {
                        Some(e) => Arc::clone(e),
                        None => {
                            return error_response(TensaError::LlmError(
                                "No LLM provider configured.".into(),
                            ))
                            .into_response()
                        }
                    },
                    Err(e) => return error_response(e).into_response(),
                }
            }
        };
        let mode = if ic.mode == crate::ingestion::config::PipelineMode::SingleSession {
            PassMode::Session
        } else if ic.mode == crate::ingestion::config::PipelineMode::Multi && pass2.is_some() {
            PassMode::Dual
        } else {
            PassMode::Single
        };
        let is_session = ic.mode == crate::ingestion::config::PipelineMode::SingleSession;
        let max_ctx = ic.session_max_context_tokens;
        (
            extractor,
            pass2,
            mode,
            ic.pass1_concurrency.max(1),
            ic.chunker_config(),
            ic.strip_boilerplate,
            is_session,
            max_ctx,
            ic.enrich,
        )
    };

    let narrative_id = orig_job.narrative_id.clone();
    let source_name = format!("{} (retry)", orig_job.source_name);

    // Determine model name and session capability for debugging
    let extractor_model = extractor.model_name().unwrap_or_else(|| "unknown".into());
    let extractor_session_capable = extractor.as_session().is_some();

    tracing::info!(
        retry_of = %job_id,
        model = %extractor_model,
        session_capable = extractor_session_capable,
        single_session = ic_single_session,
        pass_mode = ?pass_mode,
        "Retry job starting"
    );

    // Create a new job record for the retry
    let retry_job = IngestionJob {
        id: uuid::Uuid::now_v7().to_string(),
        status: crate::types::JobStatus::Pending,
        text_preview: text.chars().take(200).collect::<String>(),
        text_length: text.len(),
        narrative_id: narrative_id.clone(),
        source_name: source_name.clone(),
        pass_mode,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
        report: None,
        model: Some(extractor_model),
        single_session_requested: ic_single_session,
        session_capable: extractor_session_capable,
        effective_mode: None,
        enrich: ic_enrich,
        parent_job_id: Some(job_id.clone()),
    };
    let retry_job_id = match state.ingestion_jobs.submit(retry_job) {
        Ok(id) => id,
        Err(e) => return error_response(e).into_response(),
    };

    // Store the source text for the retry job too (enables chained retries)
    let _ = state.ingestion_jobs.store_source_text(&retry_job_id, &text);

    // Set up progress and cancel flag
    let progress = Arc::new(std::sync::Mutex::new(IngestionProgress::new(pass_mode)));
    let cancel_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        if let Ok(mut map) = state.ingestion_progress.lock() {
            map.insert(retry_job_id.clone(), Arc::clone(&progress));
        }
        if let Ok(mut map) = state.ingestion_cancel_flags.lock() {
            map.insert(retry_job_id.clone(), Arc::clone(&cancel_flag));
        }
    }

    let config = crate::ingestion::pipeline::PipelineConfig {
        chunker: chunker_config,
        auto_commit_threshold: 0.8,
        review_threshold: 0.3,
        source_id: narrative_id.clone().unwrap_or_else(|| "unknown".into()),
        source_type: "text".into(),
        narrative_id: narrative_id.clone(),
        job_id: Some(retry_job_id.clone()),
        concurrency: pass1_concurrency,
        strip_boilerplate,
        enrich: ic_enrich,
        single_session: ic_single_session,
        session_max_context_tokens: ic_session_max_ctx,
        debug: false,
        cascade_mode: Default::default(),
        post_ingest_mamdani_rule_id: None,
    };

    let hg = Arc::new(crate::hypergraph::Hypergraph::new(
        state.hypergraph.store_arc(),
    ));
    let queue = Arc::new(crate::ingestion::queue::ValidationQueue::new(
        state.hypergraph.store_arc(),
    ));
    let embedder = state.embedder.read().unwrap().clone();
    let vector_index = state.vector_index.clone();
    let vi_for_save = vector_index.clone();
    let ingestion_jobs = Arc::clone(&state.ingestion_jobs);
    let progress_map_ref = Arc::clone(&state);
    let jid = retry_job_id.clone();

    let _ = state.ingestion_jobs.mark_running(&retry_job_id);

    tokio::task::spawn_blocking(move || {
        let mut pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
            hg.clone(),
            extractor,
            embedder,
            vector_index,
            queue,
            config,
        )
        .with_job_queue(Arc::clone(&ingestion_jobs))
        .with_progress(progress)
        .with_cancel_flag(cancel_flag);

        if let Some(p2) = pass2_extractor {
            pipeline = pipeline.with_pass2_extractor(p2);
        }

        // The pipeline's chunk-hash dedup automatically skips already-ingested chunks
        match pipeline.ingest_text(&text, &source_name) {
            Ok(report) => {
                if report.cancelled {
                    let _ = ingestion_jobs.mark_failed(&jid, "Cancelled by user");
                    let _ = ingestion_jobs.store_partial_report(&jid, report);
                } else {
                    if let Some(ref nid) = narrative_id {
                        let registry =
                            crate::narrative::registry::NarrativeRegistry::new(hg.store_arc());
                        let _ = registry.update(nid, |n| {
                            n.entity_count += report.entities_created;
                            n.situation_count += report.situations_created;
                        });
                    }
                    let _ = ingestion_jobs.mark_completed(&jid, report);
                }
            }
            Err(e) => {
                let _ = ingestion_jobs.mark_failed(&jid, &e.to_string());
            }
        }

        // Persist vector index to KV after ingestion
        if let Some(vi) = &vi_for_save {
            if let Ok(idx) = vi.read() {
                if let Err(e) = idx.save(hg.store()) {
                    tracing::warn!("Failed to save vector index: {}", e);
                }
            }
        }

        if let Ok(mut map) = progress_map_ref.ingestion_progress.lock() {
            map.remove(&jid);
        }
        if let Ok(mut map) = progress_map_ref.ingestion_cancel_flags.lock() {
            map.remove(&jid);
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "job_id": retry_job_id,
            "retry_of": job_id,
            "status": "Running",
        })),
    )
        .into_response()
}

// ─── Ingestion Templates ───────────────────────────────────────

/// GET /templates — List all ingestion templates.
pub async fn list_templates(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.ingestion_jobs.list_templates() {
        Ok(templates) => {
            let redacted: Vec<_> = templates.iter().map(|t| t.redacted()).collect();
            json_ok(&redacted)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /templates/:id — Get a template by ID.
pub async fn get_template(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.ingestion_jobs.get_template(&id) {
        Ok(Some(t)) => json_ok(&t.redacted()),
        Ok(None) => error_response(TensaError::Internal(format!("Template '{}' not found", id)))
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /templates — Create a new template.
pub async fn create_template(
    State(state): State<Arc<AppState>>,
    Json(template): Json<crate::ingestion::config::IngestionTemplate>,
) -> impl IntoResponse {
    match state.ingestion_jobs.store_template(&template) {
        Ok(()) => json_ok(&serde_json::json!({ "id": template.id, "status": "created" })),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /templates/:id — Update a template.
pub async fn update_template(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(mut template): Json<crate::ingestion::config::IngestionTemplate>,
) -> impl IntoResponse {
    template.id = id;
    match state.ingestion_jobs.store_template(&template) {
        Ok(()) => json_ok(&serde_json::json!({ "id": template.id, "status": "updated" })),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /templates/:id — Delete a template (not builtin).
pub async fn delete_template(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.ingestion_jobs.delete_template(&id) {
        Ok(()) => json_ok(&serde_json::json!({ "status": "deleted" })),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── LLM Call Logs & Chunk Extractions ─────────────────────────

/// GET /ingest/jobs/:id/logs — Retrieve all LLM call logs for an ingestion job.
pub async fn get_job_logs(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    match state.ingestion_jobs.get_logs_for_job(&job_id) {
        Ok(logs) => json_ok(&logs),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /ingest/jobs/:id/chunks/:index/logs — Retrieve LLM call logs for a specific chunk.
pub async fn get_chunk_logs(
    State(state): State<Arc<AppState>>,
    Path((job_id, index)): Path<(String, usize)>,
) -> impl IntoResponse {
    match state.ingestion_jobs.get_logs_for_chunk(&job_id, index) {
        Ok(logs) => json_ok(&logs),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /ingest/jobs/:id/extractions — Retrieve all chunk extraction records for a job.
pub async fn get_job_extractions(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    match state.ingestion_jobs.get_all_chunk_extractions(&job_id) {
        Ok(records) => json_ok(&records),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /ingest/jobs/:id/chunks/:index/extraction — Retrieve extraction record for a specific chunk.
pub async fn get_chunk_extraction(
    State(state): State<Arc<AppState>>,
    Path((job_id, index)): Path<(String, usize)>,
) -> impl IntoResponse {
    match state.ingestion_jobs.get_chunk_extraction(&job_id, index) {
        Ok(Some(record)) => json_ok(&record),
        Ok(None) => error_response(TensaError::Internal(
            "No extraction record found for this chunk".into(),
        ))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /ingest/jobs/:id/chunks/:index/resend — Re-extract a single chunk with optional custom prompts.
pub async fn resend_chunk(
    State(state): State<Arc<AppState>>,
    Path((job_id, index)): Path<(String, usize)>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Get source text and manifest
    let text = match state.ingestion_jobs.get_source_text(&job_id) {
        Ok(Some(t)) => t,
        Ok(None) => {
            return error_response(TensaError::Internal(
                "Source text not found for this job".into(),
            ))
            .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    let manifest = match state.ingestion_jobs.get_chunk_manifest(&job_id) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return error_response(TensaError::Internal(
                "Chunk manifest not found for this job".into(),
            ))
            .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };

    // Find the chunk in the manifest
    let entry = match manifest.iter().find(|e| e.chunk_index == index) {
        Some(e) => e,
        None => {
            return error_response(TensaError::Internal(format!(
                "Chunk index {} not found in manifest",
                index
            )))
            .into_response()
        }
    };

    // Reconstruct the chunk text
    let chunk_text = if entry.end <= text.len() {
        &text[entry.start..entry.end]
    } else {
        return error_response(TensaError::Internal(
            "Chunk byte range out of bounds".into(),
        ))
        .into_response();
    };

    // Build extractor from current config
    let ic = state
        .ingestion_config
        .read()
        .unwrap_or_else(|e| e.into_inner());
    let extractor: Arc<dyn crate::ingestion::llm::NarrativeExtractor> = match ic
        .pass1
        .build_extractor()
    {
        Some(e) => e,
        None => match state
            .extractor
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(Arc::clone))
        {
            Some(e) => e,
            None => {
                return error_response(TensaError::LlmError("No LLM provider configured".into()))
                    .into_response()
            }
        },
    };

    // Apply custom system prompt if provided
    let custom_system = body.get("system_prompt").and_then(|v| v.as_str());
    let custom_user = body.get("user_prompt").and_then(|v| v.as_str());

    // Build the TextChunk
    let overlap_start = if entry.overlap_bytes > 0 && entry.start >= entry.overlap_bytes {
        entry.start - entry.overlap_bytes
    } else {
        entry.start
    };
    let overlap_prefix = if overlap_start < entry.start {
        text[overlap_start..entry.start].to_string()
    } else {
        String::new()
    };

    let chunk = crate::ingestion::chunker::TextChunk {
        chunk_id: index,
        text: chunk_text.to_string(),
        chapter: entry.chapter.clone(),
        start_offset: entry.start,
        end_offset: entry.end,
        overlap_prefix,
    };

    // If custom prompts provided, we need a custom extractor; otherwise use extract_with_logging
    let result = if custom_system.is_some() || custom_user.is_some() {
        // Build custom prompt and use a one-off local LLM client with custom system prompt
        // For simplicity, use the default extraction but with custom prompts
        // The user_prompt override means we call the extractor directly
        let _user_prompt = custom_user
            .map(|s| s.to_string())
            .unwrap_or_else(|| crate::ingestion::llm::build_extraction_prompt(&chunk));

        // The NarrativeExtractor trait doesn't support custom system prompts yet.
        // Log a warning so callers know their custom_system was not applied.
        if custom_system.is_some() {
            tracing::warn!("Custom system prompts not yet supported for chunk re-extraction; parameter ignored");
        }
        extractor.extract_with_logging(&chunk, &[])
    } else {
        extractor.extract_with_logging(&chunk, &[])
    };

    match result {
        Ok((extraction, exchange)) => {
            // Determine next attempt number from existing logs
            let attempt = state
                .ingestion_jobs
                .get_logs_for_chunk(&job_id, index)
                .map(|logs| logs.len() as u8)
                .unwrap_or(0);

            // Persist the log
            if let Some(ref ex) = exchange {
                let log = crate::ingestion::jobs::LlmCallLog {
                    job_id: job_id.clone(),
                    chunk_index: index,
                    pass: 1,
                    attempt,
                    system_prompt: ex.system_prompt.clone(),
                    user_prompt: ex.user_prompt.clone(),
                    raw_response: ex.raw_response.clone(),
                    parsed_extraction: Some(extraction.clone()),
                    parse_error: ex.parse_error.clone(),
                    retry_prompt: ex.retry_prompt.clone(),
                    retry_response: ex.retry_response.clone(),
                    duration_ms: ex.duration_ms,
                    model: ex.model.clone(),
                    endpoint: ex.endpoint.clone(),
                    timestamp: chrono::Utc::now(),
                };
                let _ = state.ingestion_jobs.store_llm_log(&log);
            }

            json_ok(&serde_json::json!({
                "extraction": extraction,
                "exchange": exchange,
                "chunk_index": index,
                "job_id": job_id,
            }))
        }
        Err(e) => error_response(e).into_response(),
    }
}

// ─── WebSocket Job Status (Sprint P3.9) ──────────────────────

/// GET /ws/jobs/:id — WebSocket for real-time job status updates.
pub async fn ws_job_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    ws: axum::extract::WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_job_ws(socket, state, job_id))
}

async fn handle_job_ws(
    mut socket: axum::extract::ws::WebSocket,
    state: Arc<AppState>,
    job_id: String,
) {
    use axum::extract::ws::Message;

    // Create or get a watch channel for this job (drop lock before await)
    let mut rx = {
        let Ok(mut watchers) = state.job_watchers.write() else {
            let _ = socket
                .send(Message::Text(r#"{"error":"internal lock failure"}"#.into()))
                .await;
            return;
        };
        if let Some(tx) = watchers.get(&job_id) {
            tx.subscribe()
        } else {
            let initial = crate::api::server::JobStatusEvent {
                job_id: job_id.clone(),
                status: "watching".into(),
                timestamp: chrono::Utc::now(),
            };
            let (tx, rx) = tokio::sync::watch::channel(initial);
            watchers.insert(job_id.clone(), tx);
            rx
        }
    }; // lock dropped here
       // Send current status immediately
    {
        let current = rx.borrow().clone();
        if let Ok(json) = serde_json::to_string(&current) {
            let _ = socket.send(Message::Text(json.into())).await;
        }
    }

    // Wait for status changes and forward to client
    loop {
        tokio::select! {
            changed = rx.changed() => {
                match changed {
                    Ok(()) => {
                        let event = { rx.borrow().clone() };
                        let is_terminal = event.status == "Completed" || event.status == "Failed";
                        if let Ok(json) = serde_json::to_string(&event) {
                            if socket.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        if is_terminal {
                            break;
                        }
                    }
                    Err(_) => break, // sender dropped
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // ignore other messages
                }
            }
        }
    }

    // Drop receiver before cleanup so receiver_count reflects actual state
    drop(rx);

    // Clean up watcher if no remaining subscribers
    if let Ok(mut watchers) = state.job_watchers.write() {
        if let Some(tx) = watchers.get(&job_id) {
            if tx.receiver_count() == 0 {
                watchers.remove(&job_id);
            }
        }
    }
}

// ─── URL/RSS Ingestion (Sprint P3.9) ──────────────────────────

/// Request body for POST /ingest/url.
#[derive(Deserialize)]
pub struct IngestUrlRequest {
    pub url: String,
    pub narrative_id: Option<String>,
    pub source_name: Option<String>,
}

/// POST /ingest/url — Fetch a URL, strip HTML, and ingest the text.
pub async fn ingest_url(
    State(state): State<Arc<AppState>>,
    Json(body): Json<IngestUrlRequest>,
) -> impl IntoResponse {
    let text = match crate::ingestion::web::fetch_and_extract_text(&body.url).await {
        Ok(t) => t,
        Err(e) => return error_response(e).into_response(),
    };

    if text.trim().is_empty() {
        return error_response(TensaError::ExtractionError(
            "No text extracted from URL".into(),
        ))
        .into_response();
    }

    // Reuse the ingest_text pipeline
    let extractor = state.extractor.read().unwrap().clone();
    let extractor = match extractor {
        Some(e) => e,
        None => {
            return error_response(TensaError::LlmError("No LLM configured".into())).into_response()
        }
    };

    let hg = Arc::new(crate::hypergraph::Hypergraph::new(
        state.hypergraph.store_arc(),
    ));
    let embedder = state.embedder.read().unwrap().clone();
    let vi = state.vector_index.clone();
    let queue = Arc::new(crate::ingestion::queue::ValidationQueue::new(
        state.hypergraph.store_arc(),
    ));
    let source_name = body.source_name.unwrap_or_else(|| body.url.clone());
    let narrative_id = body.narrative_id.clone();

    let ic_enrich = state
        .ingestion_config
        .read()
        .map(|ic| ic.enrich)
        .unwrap_or(true);
    let config = crate::ingestion::pipeline::PipelineConfig {
        chunker: crate::ingestion::chunker::ChunkerConfig::default(),
        auto_commit_threshold: 0.8,
        review_threshold: 0.3,
        source_id: narrative_id.clone().unwrap_or_else(|| "url-ingest".into()),
        source_type: "url".to_string(),
        narrative_id,
        job_id: None,
        concurrency: 1,
        strip_boilerplate: true,
        enrich: ic_enrich,
        single_session: false,
        session_max_context_tokens: 0,
        debug: false,
        cascade_mode: Default::default(),
        post_ingest_mamdani_rule_id: None,
    };

    let result = tokio::task::spawn_blocking(move || {
        let pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
            hg, extractor, embedder, vi, queue, config,
        );
        pipeline.ingest_text(&text, &source_name)
    })
    .await;

    match result {
        Ok(Ok(report)) => json_ok(&report),
        Ok(Err(e)) => error_response(e).into_response(),
        Err(e) => {
            error_response(TensaError::Internal(format!("Task panicked: {}", e))).into_response()
        }
    }
}

/// Request body for POST /ingest/rss.
#[derive(Deserialize)]
pub struct IngestRssRequest {
    pub feed_url: String,
    pub narrative_id: Option<String>,
    pub max_items: Option<usize>,
}

/// POST /ingest/rss — Fetch an RSS/Atom feed and ingest each item.
#[cfg(feature = "web-ingest")]
pub async fn ingest_rss(
    State(state): State<Arc<AppState>>,
    Json(body): Json<IngestRssRequest>,
) -> impl IntoResponse {
    let max_items = body.max_items.unwrap_or(10);
    let items = match crate::ingestion::web::fetch_rss_items(&body.feed_url, max_items).await {
        Ok(items) => items,
        Err(e) => return error_response(e).into_response(),
    };

    let mut results = Vec::new();
    for item in &items {
        if item.content.trim().is_empty() {
            continue;
        }
        results.push(serde_json::json!({
            "title": item.title,
            "link": item.link,
            "content_length": item.content.len(),
        }));
    }

    json_ok(&serde_json::json!({
        "items_found": items.len(),
        "items": results,
    }))
}

// ─── Narrative Endpoints (Phase 3) ──────────────────────────

/// POST /narratives
pub async fn create_narrative(
    State(state): State<Arc<AppState>>,
    Json(body): Json<crate::narrative::types::Narrative>,
) -> impl IntoResponse {
    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());

    match registry.create(body) {
        Ok(id) => (StatusCode::CREATED, Json(serde_json::json!({"id": id}))).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives
pub async fn list_narratives(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<PaginationParams>,
) -> impl IntoResponse {
    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());
    let limit = clamp_limit(params.limit);

    match registry.list_paginated(limit, params.after.as_deref()) {
        Ok((narratives, next_cursor)) => {
            let resp = PaginatedResponse {
                data: narratives,
                next_cursor,
            };
            json_ok(&resp)
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/{id}
pub async fn get_narrative(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());

    match registry.get(&id) {
        Ok(narrative) => Json(serde_json::json!(narrative)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /narratives/:id?cascade=true
///
/// Without `cascade=true`, deletes only the narrative metadata.
/// With `cascade=true`, deletes all entities and situations belonging to the
/// narrative (plus their participations and causal links) before removing the
/// narrative record.
pub async fn delete_narrative(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let cascade = params.get("cascade").map(|v| v == "true").unwrap_or(false);
    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());

    // Verify narrative exists first
    if let Err(e) = registry.get(&id) {
        return error_response(e).into_response();
    }

    if cascade {
        match cascade_delete_narrative(&state.hypergraph, &id) {
            Ok(stats) => {
                if let Err(e) = registry.delete(&id) {
                    return error_response(e).into_response();
                }
                Json(serde_json::json!({
                    "deleted": true,
                    "cascade": stats,
                }))
                .into_response()
            }
            Err(e) => error_response(e).into_response(),
        }
    } else {
        match registry.delete(&id) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => error_response(e).into_response(),
        }
    }
}

/// Cascade-delete all entities and situations belonging to a narrative.
/// Returns counts of deleted items.
pub fn cascade_delete_narrative(
    hg: &crate::hypergraph::Hypergraph,
    narrative_id: &str,
) -> std::result::Result<serde_json::Value, TensaError> {
    let mut entities_deleted = 0u64;
    let mut situations_deleted = 0u64;
    let mut participations_removed = 0u64;
    let mut causal_links_removed = 0u64;

    // Delete situations first (removes participations/causal links)
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    for sit in &situations {
        // Remove participations for this situation
        let participants = hg.get_participants_for_situation(&sit.id)?;
        for p in &participants {
            let _ = hg.remove_participant(&p.entity_id, &sit.id, None);
            participations_removed += 1;
        }
        // Remove causal links originating from this situation
        let effects = hg.traverse_causal_chain(&sit.id, 1)?;
        for link in &effects {
            let _ = hg.remove_causal_link(&link.from_situation, &link.to_situation);
            causal_links_removed += 1;
        }
        // Remove causal links pointing to this situation
        let antecedents = hg.get_antecedents(&sit.id)?;
        for link in &antecedents {
            let _ = hg.remove_causal_link(&link.from_situation, &link.to_situation);
            causal_links_removed += 1;
        }
        hg.delete_situation(&sit.id)?;
        situations_deleted += 1;
    }

    // Delete entities
    let entities = hg.list_entities_by_narrative(narrative_id)?;
    for ent in &entities {
        // Remove any remaining participations
        let sits = hg.get_situations_for_entity(&ent.id)?;
        for p in &sits {
            let _ = hg.remove_participant(&ent.id, &p.situation_id, None);
            participations_removed += 1;
        }
        hg.delete_entity(&ent.id)?;
        entities_deleted += 1;
    }

    // Delete ingestion jobs associated with this narrative
    let ingestion_queue = crate::ingestion::jobs::IngestionJobQueue::new(hg.store_arc());
    let ingestion_jobs_deleted = ingestion_queue
        .delete_jobs_for_narrative(narrative_id)
        .unwrap_or(0);

    Ok(serde_json::json!({
        "entities_deleted": entities_deleted,
        "situations_deleted": situations_deleted,
        "participations_removed": participations_removed,
        "causal_links_removed": causal_links_removed,
        "ingestion_jobs_deleted": ingestion_jobs_deleted,
    }))
}

/// Query params accepting `include_synthetic=true` (with legacy camelCase
/// fallback). EATH Phase 3 — keeps the synthetic-leak invariant test
/// passing for endpoints that don't already have a typed query struct.
#[derive(Deserialize, Default)]
pub struct IncludeSyntheticQuery {
    #[serde(default)]
    pub include_synthetic: Option<bool>,
    #[serde(default, rename = "includeSynthetic")]
    pub include_synthetic_camel: Option<bool>,
}

impl IncludeSyntheticQuery {
    pub(crate) fn flag(&self) -> bool {
        if let Some(v) = self.include_synthetic_camel {
            tracing::warn!(
                "deprecated query param `includeSynthetic` — use `include_synthetic` instead"
            );
            return v;
        }
        self.include_synthetic.unwrap_or(false)
    }
}

/// GET /narratives/{id}/stats
pub async fn get_narrative_stats(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<IncludeSyntheticQuery>,
) -> impl IntoResponse {
    let corpus = crate::narrative::corpus::CorpusManager::new(state.hypergraph.store_arc());
    match corpus.compute_stats_with_synthetic(&id, &state.hypergraph, params.flag()) {
        Ok(stats) => Json(serde_json::json!(stats)).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/anomalies — Get persisted anomaly report.
pub async fn get_narrative_anomalies(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match crate::analysis::anomaly::get_anomaly_report(state.hypergraph.store(), &id) {
        Ok(Some(report)) => Json(serde_json::json!(report)).into_response(),
        Ok(None) => Json(serde_json::json!({
            "error": "No anomaly report found. Run INFER ANOMALY first."
        }))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /narratives/:id — Partial update of narrative metadata.
pub async fn update_narrative(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());

    match registry.update(&id, |n| {
        if let Some(v) = body.get("title").and_then(|v| v.as_str()) {
            n.title = v.to_string();
        }
        if let Some(v) = body.get("genre") {
            n.genre = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("description") {
            n.description = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("tags").and_then(|v| v.as_array()) {
            n.tags = v
                .iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect();
        }
        if let Some(v) = body.get("authors").and_then(|v| v.as_array()) {
            n.authors = v
                .iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect();
        }
        if let Some(v) = body.get("language") {
            n.language = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("publication_date") {
            n.publication_date = v.as_str().and_then(|s| s.parse().ok());
        }
        if let Some(v) = body.get("cover_url") {
            n.cover_url = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
        if let Some(v) = body.get("custom_properties").and_then(|v| v.as_object()) {
            n.custom_properties = v.iter().map(|(k, val)| (k.clone(), val.clone())).collect();
        }
        if let Some(v) = body.get("project_id") {
            n.project_id = if v.is_null() {
                None
            } else {
                v.as_str().map(String::from)
            };
        }
    }) {
        Ok(narrative) => json_ok(&narrative),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/merge — Merge source narrative into target.
pub async fn merge_narratives(
    State(state): State<Arc<AppState>>,
    Path(target_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let source_id = match body.get("source_id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "source_id is required"})),
            )
                .into_response()
        }
    };

    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());
    match registry.merge_narratives(&target_id, &source_id, &state.hypergraph) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/reorder — Batch-reorder scenes inside a narrative (Sprint W8).
///
/// Body:
/// ```json
/// { "entries": [ { "situation_id": "...", "parent_id": "..."? }, ... ] }
/// ```
///
/// Writes `manuscript_order` + `parent_situation_id` atomically (per-situation),
/// densifying positions to `1000, 2000, 3000, …` so drag-inserts don't cascade.
#[derive(serde::Deserialize)]
pub struct ReorderBody {
    pub entries: Vec<crate::writer::reorder::ReorderEntry>,
}

pub async fn reorder_narrative_scenes(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<ReorderBody>,
) -> impl IntoResponse {
    if body.entries.is_empty() {
        return error_response(TensaError::InvalidQuery(
            "reorder payload must contain at least one entry".into(),
        ))
        .into_response();
    }
    const MAX_REORDER_ENTRIES: usize = 10_000;
    if body.entries.len() > MAX_REORDER_ENTRIES {
        return error_response(TensaError::InvalidQuery(format!(
            "reorder payload exceeds maximum {MAX_REORDER_ENTRIES} entries"
        )))
        .into_response();
    }
    match crate::writer::reorder::apply_reorder(&state.hypergraph, &narrative_id, &body.entries) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/causal-links/backfill-adjacent — Synthesise weak
/// sequential `Enabling` causal edges between adjacent same-level situations
/// that don't already have one. Idempotent. Used to repair legacy or
/// writer-generated narratives that landed with an empty causal graph.
pub async fn backfill_adjacent_causal_links_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match crate::narrative::backfill::backfill_adjacent_causal_links(
        &state.hypergraph,
        &narrative_id,
    ) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/names/backfill-from-content — Derive a short title
/// from the first line of prose for every situation whose `name` is empty.
/// Idempotent; situations that already have a non-empty name are left alone.
/// Useful for legacy `HumanEntered` situations where the chapter title was
/// baked into `raw_content[0]` but never promoted to `Situation.name`.
pub async fn backfill_names_from_content_handler(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match crate::narrative::name_backfill::backfill_names_from_content(
        &state.hypergraph,
        &narrative_id,
    ) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/:id/dedup-entities — Propose entity merge candidates.
///
/// Returns a list of candidate merges for human review. Does NOT actually
/// merge anything — the Studio UI or an analyst calls POST /entities/merge
/// for each accepted candidate.
///
/// Request body (all fields optional, camelCase also accepted via serde):
/// ```json
/// {
///   "threshold": 0.7,
///   "max_candidates": 200,
///   "entity_types": ["Actor", "Location"]
/// }
/// ```
pub async fn dedup_entities(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Accept an empty body as "use all defaults"
    let opts: crate::narrative::dedup::DedupOptions =
        if body.is_null() || matches!(&body, serde_json::Value::Object(m) if m.is_empty()) {
            crate::narrative::dedup::DedupOptions::default()
        } else {
            match serde_json::from_value(body) {
                Ok(o) => o,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": format!("invalid options: {}", e)})),
                    )
                        .into_response()
                }
            }
        };

    match crate::narrative::dedup::find_duplicate_candidates(
        &state.hypergraph,
        &narrative_id,
        &opts,
    ) {
        Ok(candidates) => {
            let response = serde_json::json!({
                "narrative_id": narrative_id,
                "threshold": opts.threshold,
                "candidate_count": candidates.len(),
                "candidates": candidates,
            });
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Taxonomy Endpoints ─────────────────────────────────────

/// GET /taxonomy/:category — List builtin + custom entries for a category.
pub async fn list_taxonomy(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
) -> impl IntoResponse {
    let registry = crate::narrative::taxonomy::TaxonomyRegistry::new(state.hypergraph.store_arc());
    match registry.list(&category) {
        Ok(entries) => json_ok(&entries),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /taxonomy/:category — Add a custom taxonomy entry.
pub async fn add_taxonomy(
    State(state): State<Arc<AppState>>,
    Path(category): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let value = match body.get("value").and_then(|v| v.as_str()) {
        Some(v) => v.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "value is required"})),
            )
                .into_response()
        }
    };
    let label = body
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or(&value)
        .to_string();
    let description = body
        .get("description")
        .and_then(|v| v.as_str())
        .map(String::from);

    let entry = crate::narrative::types::TaxonomyEntry {
        category: category.clone(),
        value,
        label,
        description,
        is_builtin: false,
    };

    let registry = crate::narrative::taxonomy::TaxonomyRegistry::new(state.hypergraph.store_arc());
    match registry.add(entry) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"status": "created"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /taxonomy/:category/:value — Remove a custom taxonomy entry.
pub async fn remove_taxonomy(
    State(state): State<Arc<AppState>>,
    Path((category, value)): Path<(String, String)>,
) -> impl IntoResponse {
    let registry = crate::narrative::taxonomy::TaxonomyRegistry::new(state.hypergraph.store_arc());
    match registry.remove(&category, &value) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /analysis/by-tag — Submit analysis across all narratives matching a tag.
pub async fn analyze_by_tag(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let tag = match body.get("tag").and_then(|v| v.as_str()) {
        Some(t) => t.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "tag is required"})),
            )
                .into_response()
        }
    };
    let job_type_str = body
        .get("job_type")
        .and_then(|v| v.as_str())
        .unwrap_or("StyleProfile");

    let registry = crate::narrative::registry::NarrativeRegistry::new(state.hypergraph.store_arc());
    let narratives = match registry.list(None, Some(&tag)) {
        Ok(n) => n,
        Err(e) => return error_response(e).into_response(),
    };

    if narratives.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("No narratives found with tag '{}'", tag)})),
        )
            .into_response();
    }

    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    let job_type = match job_type_str {
        "StyleProfile" => InferenceJobType::StyleProfile,
        "StyleComparison" => InferenceJobType::StyleComparison,
        "StyleAnomaly" => InferenceJobType::StyleAnomaly,
        "ArcClassification" => InferenceJobType::ArcClassification,
        "ActorArcClassification" | "ActorArcs" => InferenceJobType::ActorArcClassification,
        "Centrality" | "CentralityAnalysis" => InferenceJobType::CentralityAnalysis,
        "Entropy" | "EntropyAnalysis" => InferenceJobType::EntropyAnalysis,
        "Contagion" | "ContagionAnalysis" => InferenceJobType::ContagionAnalysis,
        "PageRank" => InferenceJobType::PageRank,
        "Eigenvector" | "EigenvectorCentrality" => InferenceJobType::EigenvectorCentrality,
        "Harmonic" | "HarmonicCentrality" => InferenceJobType::HarmonicCentrality,
        "HITS" => InferenceJobType::HITS,
        "Topology" => InferenceJobType::Topology,
        "LabelPropagation" => InferenceJobType::LabelPropagation,
        "KCore" => InferenceJobType::KCore,
        "TemporalPageRank" => InferenceJobType::TemporalPageRank,
        "CausalInfluence" => InferenceJobType::CausalInfluence,
        "InfoBottleneck" => InferenceJobType::InfoBottleneck,
        "Assortativity" => InferenceJobType::Assortativity,
        "TemporalMotifs" => InferenceJobType::TemporalMotifs,
        "FactionEvolution" => InferenceJobType::FactionEvolution,
        "FastRP" => InferenceJobType::FastRP,
        "Node2Vec" => InferenceJobType::Node2Vec,
        "NetworkInference" => InferenceJobType::NetworkInference,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("Unknown job type: {}", other)})),
            )
                .into_response()
        }
    };

    let force = body
        .get("force")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let scope = body
        .get("scope")
        .and_then(|v| v.as_str())
        .unwrap_or("story")
        .to_string();
    let status_store = crate::analysis_status::AnalysisStatusStore::new(
        state.hypergraph.store_arc(),
    );

    let mut submitted = Vec::new();
    let mut skipped = Vec::new();
    for narrative in &narratives {
        // One KV read tells us both whether to skip AND the row to surface.
        if !force {
            match status_store.get(&narrative.id, &job_type, &scope) {
                Ok(Some(existing)) if existing.locked => {
                    skipped.push(serde_json::json!({
                        "narrative_id": &narrative.id,
                        "reason": "locked",
                        "existing": existing,
                    }));
                    continue;
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(
                    target: "tensa::analysis_status",
                    error = %e,
                    "status lookup failed"
                ),
            }
        }

        let first_entity = state
            .hypergraph
            .list_entities_by_narrative(&narrative.id)
            .ok()
            .and_then(|e| e.first().map(|ent| ent.id));
        let target_id = first_entity.unwrap_or_default();

        let job = InferenceJob {
            id: uuid::Uuid::now_v7().to_string(),
            job_type: job_type.clone(),
            target_id,
            parameters: serde_json::json!({
                "narrative_id": &narrative.id,
                "tag": &tag,
                "scope": &scope,
            }),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let job_id = job.id.clone();
        if let Ok(_) = job_queue.submit(job) {
            submitted.push(serde_json::json!({
                "job_id": job_id,
                "narrative_id": &narrative.id,
            }));
        }
    }

    json_ok(&serde_json::json!({
        "tag": tag,
        "job_type": job_type_str,
        "scope": scope,
        "force": force,
        "narratives_matched": narratives.len(),
        "jobs_submitted": submitted,
        "skipped_locked": skipped,
    }))
}

// ─── Sprint 3: Pathfinding Endpoints ────────────────────────

/// POST /analysis/shortest-path — Compute weighted shortest path(s) between entities.
pub async fn compute_shortest_path(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let source = body
        .get("source")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<uuid::Uuid>().ok());
    let target = body
        .get("target")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<uuid::Uuid>().ok());
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let k = body.get("k").and_then(|v| v.as_u64()).map(|v| v as usize);

    let (Some(src), Some(tgt)) = (source, target) else {
        return error_response(TensaError::QueryError(
            "source and target UUIDs required".into(),
        ))
        .into_response();
    };

    let graph =
        match crate::analysis::graph_projection::build_co_graph(&state.hypergraph, narrative_id) {
            Ok(g) => g,
            Err(e) => return error_response(e).into_response(),
        };
    let src_idx = graph.entities.iter().position(|&e| e == src);
    let tgt_idx = graph.entities.iter().position(|&e| e == tgt);
    let (Some(si), Some(ti)) = (src_idx, tgt_idx) else {
        return error_response(TensaError::QueryError(
            "source or target not in narrative graph".into(),
        ))
        .into_response();
    };

    if let Some(k) = k {
        let paths = crate::analysis::pathfinding::yen_k_shortest(&graph, si, ti, k);
        let results: Vec<serde_json::Value> = paths.iter().map(|sp| {
            let ids: Vec<String> = sp.path.iter().map(|&i| graph.entities[i].to_string()).collect();
            serde_json::json!({"path": ids, "length": sp.path.len() - 1, "total_weight": sp.total_weight})
        }).collect();
        json_ok(&serde_json::json!({"paths": results}))
    } else {
        match crate::analysis::pathfinding::dijkstra(&graph, si, ti) {
            Some(sp) => {
                let ids: Vec<String> = sp
                    .path
                    .iter()
                    .map(|&i| graph.entities[i].to_string())
                    .collect();
                json_ok(
                    &serde_json::json!({"path": ids, "length": sp.path.len() - 1, "total_weight": sp.total_weight}),
                )
            }
            None => json_ok(&serde_json::json!({"path": null, "message": "No path found"})),
        }
    }
}

/// POST /analysis/narrative-diameter — Compute longest causal chain.
pub async fn compute_narrative_diameter(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let dag = match crate::analysis::graph_projection::build_causal_dag(
        &state.hypergraph,
        narrative_id,
    ) {
        Ok(d) => d,
        Err(e) => return error_response(e).into_response(),
    };
    match crate::analysis::pathfinding::narrative_diameter(&dag) {
        Ok(Some(sp)) => {
            let ids: Vec<String> = sp
                .path
                .iter()
                .map(|&i| dag.situations[i].to_string())
                .collect();
            json_ok(
                &serde_json::json!({"path": ids, "length": sp.path.len() - 1, "total_weight": sp.total_weight}),
            )
        }
        Ok(None) => json_ok(
            &serde_json::json!({"path": null, "length": 0, "message": "No causal chains found"}),
        ),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /analysis/max-flow — Compute max-flow/min-cut between entities.
pub async fn compute_max_flow(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let source = body
        .get("source")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<uuid::Uuid>().ok());
    let target = body
        .get("target")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<uuid::Uuid>().ok());
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let (Some(src), Some(tgt)) = (source, target) else {
        return error_response(TensaError::QueryError(
            "source and target UUIDs required".into(),
        ))
        .into_response();
    };

    let graph =
        match crate::analysis::graph_projection::build_co_graph(&state.hypergraph, narrative_id) {
            Ok(g) => g,
            Err(e) => return error_response(e).into_response(),
        };
    let src_idx = graph.entities.iter().position(|&e| e == src);
    let tgt_idx = graph.entities.iter().position(|&e| e == tgt);
    let (Some(si), Some(ti)) = (src_idx, tgt_idx) else {
        return error_response(TensaError::QueryError(
            "source or target not in narrative graph".into(),
        ))
        .into_response();
    };

    let (flow, cut_edges) = crate::analysis::pathfinding::max_flow(&graph, si, ti);
    let cuts: Vec<serde_json::Value> = cut_edges
        .iter()
        .map(|&(a, b)| {
            serde_json::json!([graph.entities[a].to_string(), graph.entities[b].to_string()])
        })
        .collect();
    json_ok(&serde_json::json!({"flow": flow, "cut_edges": cuts}))
}

/// POST /analysis/contagion-bistability — Run a forward-backward β-sweep on a
/// real narrative and return a bistability classification.
///
/// Synchronous (no job queue) — Phase 14 "real-narrative analysis" endpoint
/// mirroring `compute_higher_order_contagion`. Replicates × num_betas stays
/// bounded by the sweep design, so an inline endpoint is acceptable. For
/// significance-against-surrogates, see `POST /synth/bistability-significance`.
pub async fn compute_contagion_bistability(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if narrative_id.is_empty() {
        return error_response(TensaError::InvalidQuery(
            "narrative_id is required".into(),
        ))
        .into_response();
    }
    let params_json = match body.get("params") {
        Some(p) if !p.is_null() => p.clone(),
        _ => {
            return error_response(TensaError::InvalidQuery(
                "params blob (BistabilitySweepParams) is required".into(),
            ))
            .into_response();
        }
    };
    let params = match crate::analysis::contagion_bistability::parse_params(&params_json) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    match crate::analysis::contagion_bistability::run_bistability_sweep(
        &state.hypergraph,
        narrative_id,
        &params,
    ) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /analysis/higher-order-contagion — Run higher-order SIR on a real narrative.
///
/// Synchronous (no job queue) — Phase 7b "real-narrative analysis" endpoint.
/// For null-model significance versus K EATH surrogates, see
/// `POST /synth/contagion-significance` instead.
pub async fn compute_higher_order_contagion(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if narrative_id.is_empty() {
        return error_response(TensaError::InvalidQuery(
            "narrative_id is required".into(),
        ))
        .into_response();
    }
    let params_json = match body.get("params") {
        Some(p) if !p.is_null() => p.clone(),
        _ => {
            return error_response(TensaError::InvalidQuery(
                "params blob (HigherOrderSirParams) is required".into(),
            ))
            .into_response();
        }
    };
    let params = match crate::analysis::higher_order_contagion::parse_params(&params_json) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    match crate::analysis::higher_order_contagion::simulate_higher_order_sir(
        &state.hypergraph,
        narrative_id,
        &params,
    ) {
        Ok(result) => json_ok(&result),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /narratives/{id}/analyze — Submit a batch of inference/analysis jobs for a narrative.
///
/// Submits the right jobs in priority order so foundational analyses run first.
/// Safe to call multiple times — the job queue deduplicates by (target_id, job_type).
pub async fn analyze_narrative(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    body: Option<Json<serde_json::Value>>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    // Parse optional tier selection + force flag from body (default: all tiers, force off)
    let all_tiers = vec![
        "foundational",
        "structural",
        "per_actor",
        "temporal",
        "advanced",
    ];
    let body_value = body.map(|Json(v)| v).unwrap_or_else(|| serde_json::json!({}));
    let selected_tiers: Vec<String> = body_value
        .get("tiers")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| all_tiers.iter().map(|s| s.to_string()).collect());
    let force = body_value
        .get("force")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let tier = |name: &str| selected_tiers.iter().any(|t| t == name);

    // Collect entities and situations for this narrative
    let entities = match state.hypergraph.list_entities_by_narrative(&narrative_id) {
        Ok(e) => e,
        Err(e) => return error_response(e).into_response(),
    };
    let situations = match state.hypergraph.list_situations_by_narrative(&narrative_id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };

    if entities.is_empty() && situations.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Narrative has no entities or situations" })),
        )
            .into_response();
    }

    let actors: Vec<_> = entities
        .iter()
        .filter(|e| e.entity_type == EntityType::Actor)
        .collect();
    let first_entity_id = entities.first().map(|e| e.id).unwrap_or_default();
    let first_situation_id = situations.first().map(|s| s.id).unwrap_or_default();
    let narrative_params = serde_json::json!({ "narrative_id": narrative_id });

    let mut submitted = Vec::new();
    let mut skipped: Vec<serde_json::Value> = Vec::new();
    let status_store = crate::analysis_status::AnalysisStatusStore::new(
        state.hypergraph.store_arc(),
    );

    // Helper to submit a job and record it. Skips locked Skill/Manual entries
    // unless `force` was passed.
    let narrative_id_for_check = narrative_id.clone();
    let mut submit = |job_type: InferenceJobType,
                      target_id: uuid::Uuid,
                      priority: JobPriority,
                      params: serde_json::Value| {
        if !force {
            if let Ok(Some(existing)) =
                status_store.get(&narrative_id_for_check, &job_type, "story")
            {
                if existing.locked {
                    skipped.push(serde_json::json!({
                        "job_type": job_type.variant_name(),
                        "reason": "locked",
                        "existing": existing,
                    }));
                    return;
                }
            }
        }
        let job = crate::inference::types::InferenceJob {
            id: uuid::Uuid::now_v7().to_string(),
            job_type,
            target_id,
            parameters: params,
            priority,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let jt = format!("{:?}", job.job_type);
        match job_queue.submit(job) {
            Ok(id) => submitted.push(serde_json::json!({ "job_id": id, "job_type": jt })),
            Err(_) => {} // Dedup rejection is fine
        }
    };

    // ── Foundational: fast narrative-wide analysis (High priority) ──
    if tier("foundational") {
        submit(
            InferenceJobType::CentralityAnalysis,
            first_entity_id,
            JobPriority::High,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::EntropyAnalysis,
            first_situation_id,
            JobPriority::High,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::ArcClassification,
            first_situation_id,
            JobPriority::High,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::AnomalyDetection,
            first_situation_id,
            JobPriority::High,
            narrative_params.clone(),
        );
    }

    // ── Structural: causal, patterns, style (Normal priority) ──
    if tier("structural") && !situations.is_empty() {
        submit(
            InferenceJobType::CausalDiscovery,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::StyleProfile,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::PatternMining,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::MissingEventPrediction,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::ContagionAnalysis,
            first_entity_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::EvidenceCombination,
            first_entity_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::ArgumentationAnalysis,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::StyleComparison,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::StyleAnomaly,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::TCGAnomaly,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::NextEvent,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::MissingLinks,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::Counterfactual,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::GameClassification,
            first_situation_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
    }

    // ── Per-actor: motivation, beliefs (Normal priority, scales with actors) ──
    if tier("per_actor") {
        for actor in &actors {
            submit(
                InferenceJobType::MotivationInference,
                actor.id,
                JobPriority::Normal,
                narrative_params.clone(),
            );
        }
        submit(
            InferenceJobType::BeliefModeling,
            first_entity_id,
            JobPriority::Normal,
            narrative_params.clone(),
        );
    }

    // ── Temporal: ILP, mean-field, PSL (Low priority) ──
    if tier("temporal") && !situations.is_empty() {
        submit(
            InferenceJobType::TemporalILP,
            first_situation_id,
            JobPriority::Low,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::MeanFieldGame,
            first_situation_id,
            JobPriority::Low,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::ProbabilisticSoftLogic,
            first_entity_id,
            JobPriority::Low,
            narrative_params.clone(),
        );
        submit(
            InferenceJobType::TrajectoryEmbedding,
            first_entity_id,
            JobPriority::Low,
            narrative_params.clone(),
        );
    }

    // ── NarrativeSimulation: NOT submitted from bulk-analyze ──
    // The engine materialises forward-play "Simulation turn N" Scene situations
    // into the hypergraph (extraction_method = Simulated, confidence = 0.5),
    // which pollutes the canonical narrative graph and re-runs of bulk-analyze
    // duplicate the synthetic rows. Studio filters them at render-time, but
    // the cleaner fix is to keep them out of the bulk path entirely. Callers
    // who actually want a forward simulation submit the job explicitly:
    //   POST /jobs {"job_type":"NarrativeSimulation","target_id":"<sit-uuid>"}

    let count = submitted.len();
    Json(serde_json::json!({
        "narrative_id": narrative_id,
        "submitted": count,
        "entities": entities.len(),
        "situations": situations.len(),
        "actors": actors.len(),
        "jobs": submitted,
        "skipped_locked": skipped,
        "force": force,
    }))
    .into_response()
}

/// POST /narratives/{id}/batch-infer — Submit specific inference job types for a narrative.
///
/// Body: `{ "job_types": ["CausalDiscovery", "MotivationInference"], "priority": "normal" }`
/// Returns list of submitted job IDs.
pub async fn batch_infer_narrative(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let job_queue = match &state.job_queue {
        Some(q) => q,
        None => {
            return error_response(TensaError::InferenceError("Inference not enabled".into()))
                .into_response()
        }
    };

    let job_types: Vec<String> = body
        .get("job_types")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    if job_types.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "job_types array is required" })),
        )
            .into_response();
    }

    let priority: JobPriority = body
        .get("priority")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(JobPriority::Normal);

    // Get a target ID from the narrative (first entity or situation)
    let entities = state
        .hypergraph
        .list_entities_by_narrative(&narrative_id)
        .unwrap_or_default();
    let situations = state
        .hypergraph
        .list_situations_by_narrative(&narrative_id)
        .unwrap_or_default();
    let target_id = entities
        .first()
        .map(|e| e.id)
        .or_else(|| situations.first().map(|s| s.id))
        .unwrap_or_default();

    let narrative_params = serde_json::json!({ "narrative_id": narrative_id });
    let mut submitted = Vec::new();

    for jt_str in &job_types {
        let job_type: InferenceJobType =
            match serde_json::from_value(serde_json::Value::String(jt_str.clone())) {
                Ok(jt) => jt,
                Err(_) => {
                    submitted.push(serde_json::json!({
                        "job_type": jt_str,
                        "error": "Unknown job type"
                    }));
                    continue;
                }
            };

        // For per-entity jobs (Motivation), submit one per actor
        let target_ids = if matches!(job_type, InferenceJobType::MotivationInference) {
            entities
                .iter()
                .filter(|e| e.entity_type == EntityType::Actor)
                .map(|e| e.id)
                .collect::<Vec<_>>()
        } else {
            vec![target_id]
        };

        for tid in target_ids {
            let job = crate::inference::types::InferenceJob {
                id: uuid::Uuid::now_v7().to_string(),
                job_type: job_type.clone(),
                target_id: tid,
                parameters: narrative_params.clone(),
                priority: priority.clone(),
                status: JobStatus::Pending,
                estimated_cost_ms: 0,
                created_at: chrono::Utc::now(),
                started_at: None,
                completed_at: None,
                error: None,
            };
            let jt = format!("{:?}", job.job_type);
            match job_queue.submit(job) {
                Ok(id) => submitted.push(serde_json::json!({ "job_id": id, "job_type": jt })),
                Err(_) => {} // Dedup rejection
            }
        }
    }

    Json(serde_json::json!({
        "narrative_id": narrative_id,
        "submitted": submitted,
    }))
    .into_response()
}

// ─── Settings Endpoints ────────────────────────────────────

/// GET /settings/llm — Return the current LLM provider config (keys redacted).
pub async fn get_llm_settings(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state
        .llm_config
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    match config {
        Ok(c) => Json(serde_json::json!({ "llm": c.redacted() })).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/llm — Update the active LLM provider at runtime.
pub async fn set_llm_settings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<crate::api::server::LlmConfig>,
) -> impl IntoResponse {
    // Build extractor from new config to validate it
    let new_extractor = body.build_extractor();
    let redacted = body.redacted();

    // Swap extractor
    {
        let mut ext = state
            .extractor
            .write()
            .map_err(|_| TensaError::Internal("Lock poisoned".into()));
        match ext {
            Ok(ref mut e) => **e = new_extractor,
            Err(e) => return error_response(e).into_response(),
        }
    }
    // Persist to KV store
    crate::api::settings_routes::persist_config(
        state.hypergraph.store(),
        crate::api::settings_routes::CFG_LLM_KEY,
        &body,
        "LLM",
    );

    // Swap config in memory
    {
        let mut cfg = state
            .llm_config
            .write()
            .map_err(|_| TensaError::Internal("Lock poisoned".into()));
        match cfg {
            Ok(ref mut c) => **c = body,
            Err(e) => return error_response(e).into_response(),
        }
    }

    tracing::info!("LLM provider updated (persisted): {:?}", redacted);
    Json(serde_json::json!({ "llm": redacted, "status": "ok" })).into_response()
}

// ─── Export Endpoints (Sprint P3.6) ────────────────────────

/// Query parameters for the export endpoint.
#[derive(Deserialize)]
pub struct ExportParams {
    #[serde(default = "default_export_format")]
    pub format: String,
    /// When true, manuscript export uses original chunk text instead of extracted content.
    #[serde(default)]
    pub source: bool,
    /// Archive-only: layer preset. `"default"` (current default — skips
    /// inference + embeddings), `"full"` (everything including embeddings,
    /// inference results, synthetic records), `"minimal"` (core graph only).
    /// Ignored for non-archive formats. For per-flag control use `POST
    /// /export/archive` with an `ArchiveExportOptions` body.
    #[serde(default)]
    pub preset: Option<String>,
}

fn default_export_format() -> String {
    "json".into()
}

fn archive_options_for_preset(preset: Option<&str>) -> crate::export::archive_types::ArchiveExportOptions {
    use crate::export::archive_types::ArchiveExportOptions;
    let default = ArchiveExportOptions::default();
    match preset.map(|s| s.to_lowercase()).as_deref() {
        // Default already has every v1.1.0 toggle ON; "full" only flips the
        // expensive opt-ins (inference results, embeddings, synthetic records).
        Some("full") => ArchiveExportOptions {
            include_inference: true,
            include_embeddings: true,
            include_synthetic: true,
            ..default
        },
        // Strip everything except core graph data.
        Some("minimal") => ArchiveExportOptions {
            include_sources: false,
            include_chunks: false,
            include_state_versions: false,
            include_analysis: false,
            include_tuning: false,
            include_taxonomy: false,
            include_projects: false,
            include_annotations: false,
            include_pinned_facts: false,
            include_revisions: false,
            include_workshop_reports: false,
            include_narrative_plan: false,
            include_analysis_status: false,
            ..default
        },
        _ => default,
    }
}

/// GET /narratives/:id/export?format=csv|graphml|json|manuscript|report|archive|stix[&preset=full|minimal]
pub async fn export_narrative(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<ExportParams>,
) -> impl IntoResponse {
    let format = match params.format.to_lowercase().as_str() {
        "csv" => crate::export::ExportFormat::Csv,
        "graphml" => crate::export::ExportFormat::GraphML,
        "json" => crate::export::ExportFormat::Json,
        "manuscript" => crate::export::ExportFormat::Manuscript,
        "report" => crate::export::ExportFormat::Report,
        "archive" | "tensa" => crate::export::ExportFormat::Archive,
        "stix" => crate::export::ExportFormat::Stix,
        other => {
            return error_response(TensaError::ExportError(format!(
                "Unknown export format: '{other}'. Use csv, graphml, json, manuscript, report, archive, or stix"
            )))
            .into_response();
        }
    };

    // Archive format with a preset bypasses the format-dispatcher so the
    // caller can opt into the v1.1.0 round-trip layers (annotations, pinned
    // facts, revisions, workshop reports, plan, analysis-status) plus
    // inference + embeddings + synthetic records, all via a single query
    // param. Non-archive formats and archive-without-preset use the default.
    if matches!(format, crate::export::ExportFormat::Archive) && params.preset.is_some() {
        let opts = archive_options_for_preset(params.preset.as_deref());
        return match crate::export::archive::export_archive(&[&id], &state.hypergraph, &opts) {
            Ok(bytes) => (
                StatusCode::OK,
                [(
                    axum::http::header::CONTENT_TYPE,
                    crate::export::archive_types::ARCHIVE_CONTENT_TYPE,
                )],
                bytes,
            )
                .into_response(),
            Err(e) => error_response(e).into_response(),
        };
    }

    match crate::export::export_narrative(&id, format, &state.hypergraph, params.source) {
        Ok(output) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, output.content_type)],
            output.body,
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Archive Export/Import Endpoints ─────────────────────────

/// Request body for POST /export/archive.
#[derive(Deserialize)]
pub struct ExportArchiveBody {
    pub narrative_ids: Vec<String>,
    #[serde(default)]
    pub options: crate::export::archive_types::ArchiveExportOptions,
}

/// POST /export/archive — Export multiple narratives as a .tensa archive.
pub async fn export_archive(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ExportArchiveBody>,
) -> impl IntoResponse {
    if body.narrative_ids.is_empty() {
        return error_response(TensaError::InvalidQuery(
            "narrative_ids must not be empty".into(),
        ))
        .into_response();
    }

    let refs: Vec<&str> = body.narrative_ids.iter().map(|s| s.as_str()).collect();
    match crate::export::archive::export_archive(&refs, &state.hypergraph, &body.options) {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (
                    axum::http::header::CONTENT_TYPE,
                    crate::export::archive_types::ARCHIVE_CONTENT_TYPE,
                ),
                (
                    axum::http::header::CONTENT_DISPOSITION,
                    "attachment; filename=\"export.tensa\"",
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Query params for POST /import/archive.
#[derive(Deserialize, Default)]
pub struct ImportArchiveParams {
    /// When `true`, merge into the existing narrative instead of auto-suffixing
    /// a colliding ID (e.g. `nightfall` → `nightfall-2`). Required by the
    /// NIGHTFALL benchmark harness so stage re-imports don't accumulate
    /// `nightfall-N` clones on every run.
    #[serde(default)]
    pub merge_mode: Option<bool>,
    /// Override default strict-reference checking.
    #[serde(default)]
    pub strict_references: Option<bool>,
    /// Re-target the archive into a different narrative ID.
    #[serde(default)]
    pub target_narrative_id: Option<String>,
}

/// POST /import/archive — Import a .tensa archive.
///
/// Accepts raw archive bytes as the request body. Options come from query
/// parameters: `?merge_mode=true`, `?strict_references=false`,
/// `?target_narrative_id=xyz` (all optional). Absent query params fall back
/// to `ArchiveImportOptions::default()` so callers that don't pass any see
/// identical behavior to pre-v0.73.4.
pub async fn import_archive(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<ImportArchiveParams>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let mut opts = crate::export::archive_types::ArchiveImportOptions::default();
    if let Some(m) = params.merge_mode {
        opts.merge_mode = m;
    }
    if let Some(s) = params.strict_references {
        opts.strict_references = s;
    }
    if let Some(t) = params.target_narrative_id {
        opts.target_narrative_id = Some(t);
    }
    match crate::ingestion::archive::import_archive(&body, &state.hypergraph, &opts) {
        Ok(report) => json_ok(&report).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Community Summary Endpoints (Sprint RAG-5) ────────────

/// POST /narratives/:id/communities/summarize — Generate community summaries for a narrative.
pub async fn summarize_communities(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    let inf_guard = state.inference_extractor.read().unwrap();
    let extractor_guard = state.extractor.read().unwrap();
    let extractor = match inf_guard.as_deref().or(extractor_guard.as_deref()) {
        Some(e) => e,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "No LLM configured"})),
            )
                .into_response()
        }
    };
    match crate::analysis::community::generate_summaries(
        &narrative_id,
        &state.hypergraph,
        extractor,
        state.hypergraph.store(),
    ) {
        Ok(summaries) => json_ok(&summaries),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/communities — List community summaries for a narrative.
pub async fn list_communities(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    // Optional ?level=N filter
    if let Some(level_str) = params.get("level") {
        if let Ok(level) = level_str.parse::<usize>() {
            return match crate::analysis::community::list_summaries_at_level(
                state.hypergraph.store(),
                &narrative_id,
                level,
            ) {
                Ok(summaries) => json_ok(&summaries),
                Err(e) => error_response(e).into_response(),
            };
        }
    }
    match crate::analysis::community::list_summaries(state.hypergraph.store(), &narrative_id) {
        Ok(summaries) => json_ok(&summaries),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/communities/hierarchy — Get the full community hierarchy.
pub async fn get_community_hierarchy(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> impl IntoResponse {
    match crate::analysis::community::get_hierarchy(state.hypergraph.store(), &narrative_id) {
        Ok(hierarchy) => json_ok(&hierarchy),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/communities/:cid — Get a single community summary.
pub async fn get_community(
    State(state): State<Arc<AppState>>,
    Path((narrative_id, cid)): Path<(String, usize)>,
) -> impl IntoResponse {
    match crate::analysis::community::get_summary(state.hypergraph.store(), &narrative_id, cid) {
        Ok(Some(s)) => json_ok(&s),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Community not found"})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /ingest/document — Ingest a PDF or DOCX document.
#[cfg(feature = "docparse")]
pub async fn ingest_document(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    use crate::ingestion::docparse;

    if let Err(e) = docparse::validate_doc_size(&body) {
        return error_response(e).into_response();
    }

    let text = match docparse::extract_text(&body) {
        Ok(t) => t,
        Err(e) => return error_response(e).into_response(),
    };

    // Delegate to the existing ingest_text pipeline
    let source_name = params
        .get("source_name")
        .cloned()
        .unwrap_or_else(|| "document_upload".into());
    let narrative_id = params.get("narrative_id").cloned();

    let ingest_req = serde_json::json!({
        "text": text,
        "source_name": source_name,
        "narrative_id": narrative_id,
    });

    // Reuse the existing ingest_text handler logic
    let req = match serde_json::from_value(ingest_req) {
        Ok(r) => r,
        Err(e) => return error_response(TensaError::Serialization(e.to_string())).into_response(),
    };
    ingest_text(State(state), Json(req)).await.into_response()
}

// ─── Reconcile & Reprocess Endpoints ──────────────────────────────────

/// POST /ingest/jobs/:id/reconcile — Run Pass 2 reconciliation on stored extractions.
///
/// Loads all chunk extraction records for the job, runs reconciliation using the
/// current pass2 LLM config, and updates the stored extraction records in-place
/// (confidence boosts). Does NOT re-commit to hypergraph.
pub async fn reconcile_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    // Load existing extractions
    let records = match state.ingestion_jobs.get_all_chunk_extractions(&job_id) {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "No extraction records found for this job" })),
            )
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };

    // Get pass2 extractor from current config
    let pass2_config = match state.ingestion_config.read() {
        Ok(cfg) => cfg.pass2.clone(),
        Err(_) => {
            return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
        }
    };
    let pass2 = match pass2_config.build_extractor() {
        Some(e) => e,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "No Pass 2 LLM configured in settings" })),
            )
                .into_response()
        }
    };

    // Collect extractions in order
    let mut extractions: Vec<crate::ingestion::extraction::NarrativeExtraction> =
        records.iter().map(|r| r.extraction.clone()).collect();

    // Run reconciliation using sliding windows (same logic as pipeline)
    let ingestion_config = match state.ingestion_config.read() {
        Ok(cfg) => cfg.clone(),
        Err(_) => {
            return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
        }
    };
    let window_size = ingestion_config.pass2_window_size;
    let overlap = ingestion_config.pass2_window_overlap;
    let step = window_size.saturating_sub(overlap).max(1);
    let mut errors: Vec<String> = Vec::new();
    let mut windows_processed = 0usize;

    let mut window_start = 0;
    while window_start < extractions.len() {
        let window_end = (window_start + window_size).min(extractions.len());
        let window = &extractions[window_start..window_end];

        let mut context = String::from(
            "Review these narrative extractions for consistency. \
             Identify duplicate entities that should be merged, \
             fix inconsistent names, and adjust confidence scores.\n\n",
        );
        for (i, ext) in window.iter().enumerate() {
            context.push_str(&format!("--- Chunk {} ---\n", window_start + i));
            if let Ok(json) = serde_json::to_string_pretty(ext) {
                context.push_str(&json);
            }
            context.push('\n');
        }

        let reconcile_chunk = crate::ingestion::chunker::TextChunk {
            chunk_id: window_start,
            text: context,
            chapter: Some("reconciliation".to_string()),
            start_offset: 0,
            end_offset: 0,
            overlap_prefix: String::new(),
        };

        match pass2.extract_narrative(&reconcile_chunk) {
            Ok(reconciled) => {
                let reconciled_names: std::collections::HashSet<String> = reconciled
                    .entities
                    .iter()
                    .map(|e| e.name.to_lowercase())
                    .collect();

                for ext in &mut extractions[window_start..window_end] {
                    for ent in &mut ext.entities {
                        if reconciled_names.contains(&ent.name.to_lowercase()) {
                            ent.confidence = (ent.confidence + 0.1).min(1.0);
                        }
                    }
                }
                windows_processed += 1;
            }
            Err(e) => {
                errors.push(format!("Window {}: {}", window_start, e));
            }
        }

        window_start += step;
    }

    // Store updated extractions back
    for (i, record) in records.iter().enumerate() {
        if let Some(ext) = extractions.get(i) {
            let updated = crate::ingestion::jobs::ChunkExtractionRecord {
                extraction: ext.clone(),
                ..record.clone()
            };
            let _ = state.ingestion_jobs.store_chunk_extraction(&updated);
        }
    }

    json_ok(&serde_json::json!({
        "status": "ok",
        "windows_processed": windows_processed,
        "errors": errors,
    }))
}

/// POST /ingest/jobs/:id/reprocess — Rollback then re-commit all extractions.
///
/// Deletes previously committed entities/situations, then re-runs the processing
/// pipeline (resolve → gate → commit) on stored extraction records.
pub async fn reprocess_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    // 1. Rollback existing committed data
    let job = match state.ingestion_jobs.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };

    let mut entities_deleted = 0usize;
    let mut situations_deleted = 0usize;
    if let Some(ref report) = job.report {
        for id in &report.created_entity_ids {
            if state.hypergraph.delete_entity(id).is_ok() {
                entities_deleted += 1;
            }
        }
        for id in &report.created_situation_ids {
            if state.hypergraph.delete_situation(id).is_ok() {
                situations_deleted += 1;
            }
        }
    }

    // 2. Load stored extractions
    let records = match state.ingestion_jobs.get_all_chunk_extractions(&job_id) {
        Ok(r) if !r.is_empty() => r,
        Ok(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "No extraction records found for this job" })),
            )
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };

    // 3. Build a pipeline for processing (no LLM needed — just resolve/gate/commit)
    let ingestion_config = match state.ingestion_config.read() {
        Ok(cfg) => cfg.clone(),
        Err(_) => {
            return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
        }
    };

    let hg = Arc::new(crate::hypergraph::Hypergraph::new(
        state.hypergraph.store_arc(),
    ));
    let queue = Arc::new(crate::ingestion::queue::ValidationQueue::new(
        state.hypergraph.store_arc(),
    ));

    // Dummy extractor — not used, processing doesn't call LLM
    let extractor: Arc<dyn crate::ingestion::llm::NarrativeExtractor> =
        Arc::new(crate::ingestion::llm::MockExtractor::empty());

    let config = crate::ingestion::pipeline::PipelineConfig {
        chunker: ingestion_config.chunker_config(),
        auto_commit_threshold: ingestion_config.auto_commit_threshold,
        review_threshold: ingestion_config.review_threshold,
        source_id: job.narrative_id.clone().unwrap_or_else(|| "unknown".into()),
        source_type: "text".into(),
        narrative_id: job.narrative_id.clone(),
        job_id: Some(job_id.clone()),
        concurrency: 1,
        strip_boilerplate: false,
        enrich: ingestion_config.enrich,
        single_session: false,
        session_max_context_tokens: 0,
        debug: false,
        cascade_mode: Default::default(),
        post_ingest_mamdani_rule_id: None,
    };

    let pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
        hg.clone(),
        extractor,
        state.embedder.read().unwrap().clone(),
        state.vector_index.clone(),
        queue,
        config,
    )
    .with_job_queue(Arc::clone(&state.ingestion_jobs));

    // 4. Re-process each extraction
    let mut resolver = crate::ingestion::resolve::EntityResolver::new();
    let provenance = crate::types::SourceReference {
        source_type: "text".into(),
        source_id: Some(job.narrative_id.clone().unwrap_or_default()),
        description: Some(job.source_name.clone()),
        timestamp: chrono::Utc::now(),
        registered_source: None,
    };

    let mut new_entity_ids = Vec::new();
    let mut new_situation_ids = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut entities_created = 0usize;
    let mut situations_created = 0usize;
    let mut auto_committed = 0usize;
    let mut queued = 0usize;
    let mut rejected = 0usize;

    for record in &records {
        match pipeline.process_extraction_standalone(
            &record.extraction,
            record.chunk_index,
            &mut resolver,
            &provenance,
        ) {
            Ok(gate_report) => {
                new_entity_ids.extend_from_slice(&gate_report.entity_ids);
                new_situation_ids.extend_from_slice(&gate_report.situation_ids);
                entities_created += gate_report.entity_ids.len();
                situations_created += gate_report.situation_ids.len();
                auto_committed += gate_report.auto_committed;
                queued += gate_report.queued;
                rejected += gate_report.rejected;
            }
            Err(e) => {
                errors.push(format!("Chunk {}: {}", record.chunk_index, e));
            }
        }
    }

    // 5. Update the job report with new IDs
    if let Ok(orig_job) = state.ingestion_jobs.get_job(&job_id) {
        let mut report = orig_job.report.unwrap_or_default();
        report.created_entity_ids = new_entity_ids;
        report.created_situation_ids = new_situation_ids;
        report.entities_created = entities_created;
        report.situations_created = situations_created;
        report.items_auto_committed = auto_committed;
        report.items_queued = queued;
        report.items_rejected = rejected;
        report.entity_resolutions = resolver.len();
        let _ = state.ingestion_jobs.store_partial_report(&job_id, report);
    }

    json_ok(&serde_json::json!({
        "status": "ok",
        "rolled_back": { "entities": entities_deleted, "situations": situations_deleted },
        "reprocessed": {
            "entities_created": entities_created,
            "situations_created": situations_created,
            "auto_committed": auto_committed,
            "queued": queued,
            "rejected": rejected,
            "errors": errors,
        },
    }))
}

/// POST /ingest/jobs/:id/chunks/:index/reprocess — Rollback and reprocess a single chunk.
pub async fn reprocess_chunk(
    State(state): State<Arc<AppState>>,
    Path((job_id, index)): Path<(String, usize)>,
) -> impl IntoResponse {
    // Load the chunk extraction record
    let record = match state.ingestion_jobs.get_chunk_extraction(&job_id, index) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "No extraction record for this chunk" })),
            )
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };

    // Rollback entities/situations from this chunk
    let mut entities_deleted = 0usize;
    let mut situations_deleted = 0usize;
    for (_name, id) in &record.entity_map {
        if state.hypergraph.delete_entity(id).is_ok() {
            entities_deleted += 1;
        }
    }
    for id in &record.situation_ids {
        if state.hypergraph.delete_situation(id).is_ok() {
            situations_deleted += 1;
        }
    }

    // Build pipeline for processing
    let ingestion_config = match state.ingestion_config.read() {
        Ok(cfg) => cfg.clone(),
        Err(_) => {
            return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
        }
    };
    let job = match state.ingestion_jobs.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };

    let hg = Arc::new(crate::hypergraph::Hypergraph::new(
        state.hypergraph.store_arc(),
    ));
    let queue = Arc::new(crate::ingestion::queue::ValidationQueue::new(
        state.hypergraph.store_arc(),
    ));
    let extractor: Arc<dyn crate::ingestion::llm::NarrativeExtractor> =
        Arc::new(crate::ingestion::llm::MockExtractor::empty());

    let config = crate::ingestion::pipeline::PipelineConfig {
        chunker: ingestion_config.chunker_config(),
        auto_commit_threshold: ingestion_config.auto_commit_threshold,
        review_threshold: ingestion_config.review_threshold,
        source_id: job.narrative_id.clone().unwrap_or_else(|| "unknown".into()),
        source_type: "text".into(),
        narrative_id: job.narrative_id.clone(),
        job_id: Some(job_id.clone()),
        concurrency: 1,
        strip_boilerplate: false,
        enrich: ingestion_config.enrich,
        single_session: false,
        session_max_context_tokens: 0,
        debug: false,
        cascade_mode: Default::default(),
        post_ingest_mamdani_rule_id: None,
    };

    let pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
        hg,
        extractor,
        state.embedder.read().unwrap().clone(),
        state.vector_index.clone(),
        queue,
        config,
    )
    .with_job_queue(Arc::clone(&state.ingestion_jobs));

    let mut resolver = crate::ingestion::resolve::EntityResolver::new();
    let provenance = crate::types::SourceReference {
        source_type: "text".into(),
        source_id: Some(job.narrative_id.clone().unwrap_or_default()),
        description: Some(job.source_name.clone()),
        timestamp: chrono::Utc::now(),
        registered_source: None,
    };

    match pipeline.process_extraction_standalone(
        &record.extraction,
        index,
        &mut resolver,
        &provenance,
    ) {
        Ok(gate_report) => json_ok(&serde_json::json!({
            "status": "ok",
            "rolled_back": { "entities": entities_deleted, "situations": situations_deleted },
            "reprocessed": {
                "entities_created": gate_report.entity_ids.len(),
                "situations_created": gate_report.situation_ids.len(),
                "auto_committed": gate_report.auto_committed,
                "queued": gate_report.queued,
                "rejected": gate_report.rejected,
            },
        })),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Cache & Document Status (Sprint RAG-1) ───────────────────

/// Get LLM response cache statistics.
pub async fn cache_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.llm_cache.as_ref() {
        Some(cache) => match cache.stats() {
            Ok(stats) => json_ok(&stats),
            Err(e) => error_response(e).into_response(),
        },
        None => json_ok(&serde_json::json!({"entries": 0, "total_bytes": 0})),
    }
}

/// Clear all cached LLM responses.
pub async fn clear_cache(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.llm_cache.as_ref() {
        Some(cache) => match cache.clear() {
            Ok(count) => json_ok(&serde_json::json!({"cleared": count})),
            Err(e) => error_response(e).into_response(),
        },
        None => json_ok(&serde_json::json!({"cleared": 0})),
    }
}

/// List all ingested document statuses.
pub async fn list_ingest_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.doc_tracker.as_ref() {
        Some(tracker) => match tracker.list_all() {
            Ok(statuses) => json_ok(&statuses),
            Err(e) => error_response(e).into_response(),
        },
        None => json_ok(&serde_json::json!([])),
    }
}

/// Get a specific ingested document status by content hash.
pub async fn get_ingest_status(
    State(state): State<Arc<AppState>>,
    Path(hash): Path<String>,
) -> impl IntoResponse {
    match state.doc_tracker.as_ref() {
        Some(tracker) => match tracker.get_by_hash(&hash) {
            Ok(Some(status)) => json_ok(&status),
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "Document not found"})),
            )
                .into_response(),
            Err(e) => error_response(e).into_response(),
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Document tracker not configured"})),
        )
            .into_response(),
    }
}

/// Cascade-delete a source and all its entities, situations, and participations.
pub async fn delete_ingest_source(
    State(state): State<Arc<AppState>>,
    Path(source_id): Path<String>,
) -> impl IntoResponse {
    let source_index = match &state.source_index {
        Some(si) => si,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "Source index not available"})),
            )
                .into_response()
        }
    };
    match crate::ingestion::deletion::cascade_delete_source(
        &source_id,
        &state.hypergraph,
        source_index,
        state.doc_tracker.as_ref(),
    ) {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Geocoding ─────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct GeocodeRequest {
    pub place: String,
}

#[derive(Deserialize)]
pub struct GeocodeBackfillRequest {
    pub narrative_id: Option<String>,
    /// Free-form setting hint for LLM canonicalization
    /// (e.g. "Early-19th-century France and Italy"). Falls back to the narrative's
    /// description + genre if omitted.
    #[serde(default)]
    pub setting: Option<String>,
    /// Default ISO 3166-1 alpha-2 country code if the narrative is single-country.
    #[serde(default)]
    pub country_hint: Option<String>,
    /// Skip LLM canonicalization entirely. Use when an extractor is configured but
    /// you specifically want a direct Nominatim lookup (provenance: `geocoded`).
    #[serde(default)]
    pub skip_canonicalization: bool,
}

/// POST /geocode — Geocode a single place name to coordinates.
pub async fn geocode_place(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GeocodeRequest>,
) -> impl IntoResponse {
    match state.geocoder.geocode(&body.place).await {
        Ok(Some(geo)) => json_ok(&geo),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "place not found", "place": body.place})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// Body for `POST /geocode/canonicalize-debug`.
#[derive(Deserialize)]
pub struct CanonicalizeDebugRequest {
    pub setting: String,
    #[serde(default)]
    pub country_hint: Option<String>,
    pub places: Vec<String>,
}

/// POST /geocode/canonicalize-batch-debug — Diagnostic: run the full
/// `Geocoder::canonicalize_places_batch` pipeline against the actual Location
/// entities of a narrative. Returns the resulting (raw → canonicalization) map
/// + a list of entities whose normalized raw name was NOT in the map. The
/// "missing" list is the exact set that would silently fall through to the
/// `Geocoded` provenance path during a real backfill.
pub async fn geocode_canonicalize_batch_debug(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GeocodeBackfillRequest>,
) -> impl IntoResponse {
    let Some(nid) = body.narrative_id.as_deref() else {
        return error_response(crate::error::TensaError::InvalidInput(
            "narrative_id is required".into(),
        ))
        .into_response();
    };

    let entities = state
        .hypergraph
        .list_entities_by_type(&crate::types::EntityType::Location)
        .unwrap_or_default()
        .into_iter()
        .filter(|e| e.narrative_id.as_deref() == Some(nid))
        .collect::<Vec<_>>();

    let to_resolve: Vec<(uuid::Uuid, String)> = entities
        .iter()
        .filter_map(|e| {
            let name = e.properties.get("name").and_then(|v| v.as_str())?;
            if name.is_empty() {
                return None;
            }
            Some((e.id, name.to_string()))
        })
        .collect();

    let extractor = match state.inference_extractor.read().ok().and_then(|g| g.clone()) {
        Some(e) => e,
        None => match state.extractor.read().ok().and_then(|g| g.clone()) {
            Some(e) => e,
            None => {
                return json_ok(&serde_json::json!({
                    "extractor_configured": false,
                    "candidates": to_resolve.len(),
                }))
            }
        },
    };

    let setting = crate::ingestion::extraction::NarrativeSettingHint {
        setting: body.setting.clone().unwrap_or_default(),
        country_hint: body.country_hint.clone(),
    };

    let canon_map = state
        .geocoder
        .debug_canonicalize_places_batch(nid, &setting, extractor.as_ref(), &to_resolve)
        .await;

    let mut missing = Vec::new();
    for (eid, raw) in &to_resolve {
        let normalized_raw = raw.trim().to_lowercase();
        if !canon_map.contains_key(&normalized_raw) {
            missing.push(serde_json::json!({
                "entity_id": eid.to_string(),
                "raw_name": raw,
                "normalized_raw": normalized_raw,
            }));
        }
    }

    let canon_list: Vec<serde_json::Value> = canon_map
        .iter()
        .map(|(k, c)| {
            serde_json::json!({
                "lookup_key": k,
                "raw_name": c.raw_name,
                "canonical_name": c.canonical_name,
                "country_code": c.country_code,
                "confidence": c.confidence,
            })
        })
        .collect();

    json_ok(&serde_json::json!({
        "extractor_configured": true,
        "model": extractor.model_name(),
        "candidates": to_resolve.len(),
        "canon_map_size": canon_map.len(),
        "missing_count": missing.len(),
        "canon_rows": canon_list,
        "missing": missing,
    }))
}

/// POST /geocode/canonicalize-debug — Diagnostic: run `canonicalize_places` for
/// a handful of raw place strings and return the LLM-resolved rows directly.
///
/// Bypasses the geocoder entirely; useful to confirm the extractor is configured
/// and the LLM is producing parseable output before running a full backfill.
pub async fn geocode_canonicalize_debug(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CanonicalizeDebugRequest>,
) -> impl IntoResponse {
    if body.places.is_empty() {
        return json_ok(&serde_json::json!({"rows": [], "extractor_configured": false, "note": "empty places"}));
    }
    let extractor: Option<Arc<dyn crate::ingestion::llm::NarrativeExtractor>> = {
        let inf = state.inference_extractor.read().ok().and_then(|g| g.clone());
        match inf {
            Some(e) => Some(e),
            None => state.extractor.read().ok().and_then(|g| g.clone()),
        }
    };
    let extractor_configured = extractor.is_some();
    let Some(extractor) = extractor else {
        return json_ok(&serde_json::json!({
            "rows": [],
            "extractor_configured": false,
            "note": "no LLM extractor configured (set /settings/inference-llm or /settings/llm)",
        }));
    };
    let setting = crate::ingestion::extraction::NarrativeSettingHint {
        setting: body.setting.clone(),
        country_hint: body.country_hint.clone(),
    };
    let pairs: Vec<(String, String)> = body
        .places
        .iter()
        .enumerate()
        .map(|(i, p)| (format!("dbg-{}", i), p.clone()))
        .collect();
    let rows_result = extractor.canonicalize_places(&setting, &pairs);
    match rows_result {
        Ok(rows) => json_ok(&serde_json::json!({
            "rows": rows,
            "requested": body.places.len(),
            "returned": rows.len(),
            "extractor_configured": extractor_configured,
            "model": extractor.model_name(),
        })),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /geocode/backfill — Batch-geocode situations and entities missing coordinates.
///
/// When a narrative_id and a configured LLM extractor are both available (and
/// `skip_canonicalization` is false), runs a one-shot batch canonicalization
/// pass to disambiguate ambiguous place names ("Marseilles" → Marseille, FR)
/// before Nominatim lookups. Each result records `geo_provenance` so the caller
/// can tell hard-fact from inferred coordinates.
pub async fn geocode_backfill(
    State(state): State<Arc<AppState>>,
    Json(body): Json<GeocodeBackfillRequest>,
) -> impl IntoResponse {
    // Collect all situations (geocode_situations handles filtering internally)
    let situations: Vec<crate::types::Situation> = if let Some(ref nid) = body.narrative_id {
        state
            .hypergraph
            .list_situations_by_narrative(nid)
            .unwrap_or_default()
    } else {
        // Candidate is the lowest maturity — includes all non-deleted situations
        state
            .hypergraph
            .list_situations_by_maturity(crate::types::MaturityLevel::Candidate)
            .unwrap_or_default()
    };

    // Collect Location entities (geocode_location_entities handles filtering internally)
    let entities = state
        .hypergraph
        .list_entities_by_type(&crate::types::EntityType::Location)
        .unwrap_or_default();

    // Build the optional NarrativeSettingHint by combining body overrides with
    // the narrative's description/genre, when available.
    let setting_hint = if body.skip_canonicalization {
        None
    } else if let Some(ref nid) = body.narrative_id {
        let setting_text = if let Some(ref s) = body.setting {
            s.clone()
        } else {
            let registry = crate::narrative::registry::NarrativeRegistry::new(
                state.hypergraph.store_arc(),
            );
            match registry.get(nid) {
                Ok(narr) => {
                    let mut parts: Vec<String> = Vec::new();
                    if let Some(d) = narr.description.as_ref() {
                        if !d.trim().is_empty() {
                            parts.push(d.clone());
                        }
                    }
                    if let Some(g) = narr.genre.as_ref() {
                        if !g.trim().is_empty() {
                            parts.push(format!("Genre: {}", g));
                        }
                    }
                    parts.join(" — ")
                }
                _ => String::new(),
            }
        };
        Some(crate::ingestion::extraction::NarrativeSettingHint {
            setting: setting_text,
            country_hint: body.country_hint.clone(),
        })
    } else {
        None
    };

    // Resolve the LLM extractor preferring inference over ingestion
    // (canonicalization is a query-shaped task, cheaper LLM is fine).
    let extractor: Option<Arc<dyn crate::ingestion::llm::NarrativeExtractor>> =
        if body.skip_canonicalization {
            None
        } else {
            let inf = state.inference_extractor.read().ok().and_then(|g| g.clone());
            match inf {
                Some(e) => Some(e),
                None => state.extractor.read().ok().and_then(|g| g.clone()),
            }
        };
    let extractor_ref: Option<&dyn crate::ingestion::llm::NarrativeExtractor> =
        extractor.as_deref().map(|e| e as _);

    let sit_count = match state
        .geocoder
        .geocode_situations_with_canon(
            &state.hypergraph,
            situations,
            body.narrative_id.as_deref(),
            setting_hint.as_ref(),
            extractor_ref,
        )
        .await
    {
        Ok(n) => n,
        Err(e) => return error_response(e).into_response(),
    };
    let ent_count = match state
        .geocoder
        .geocode_location_entities_with_canon(
            &state.hypergraph,
            entities,
            body.narrative_id.as_deref(),
            setting_hint.as_ref(),
            extractor_ref,
        )
        .await
    {
        Ok(n) => n,
        Err(e) => return error_response(e).into_response(),
    };

    json_ok(&serde_json::json!({
        "situations_geocoded": sit_count,
        "entities_geocoded": ent_count,
        "total_updated": sit_count + ent_count,
        "canonicalization_used": setting_hint.is_some() && extractor.is_some(),
    }))
}

// ─── Embedding Backfill ─────────────────────────────────────────

#[derive(Deserialize)]
pub struct EmbeddingBackfillRequest {
    pub narrative_id: Option<String>,
    #[serde(default)]
    pub force: bool,
}

/// POST /embeddings/backfill — Generate embeddings for entities and situations that lack them.
pub async fn embedding_backfill(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EmbeddingBackfillRequest>,
) -> impl IntoResponse {
    let embedder = match state.embedder.read().unwrap().clone() {
        Some(e) => e,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "No embedding provider configured"})),
            )
                .into_response()
        }
    };

    // Load entities
    let entities: Vec<crate::types::Entity> = if let Some(ref nid) = body.narrative_id {
        state
            .hypergraph
            .list_entities_by_narrative(nid)
            .unwrap_or_default()
    } else {
        state
            .hypergraph
            .list_entities_by_maturity(MaturityLevel::Candidate)
            .unwrap_or_default()
    };

    // Load situations
    let situations: Vec<crate::types::Situation> = if let Some(ref nid) = body.narrative_id {
        state
            .hypergraph
            .list_situations_by_narrative(nid)
            .unwrap_or_default()
    } else {
        state
            .hypergraph
            .list_situations_by_maturity(MaturityLevel::Candidate)
            .unwrap_or_default()
    };

    // Per-row error counters and error log (capped to keep response small).
    let mut skipped: usize = 0;
    let mut empty_skipped: usize = 0;
    let mut failed: usize = 0;
    let mut errors: Vec<serde_json::Value> = Vec::new();
    const MAX_ERRORS: usize = 50;

    // Build (id, text) pairs for entities that need embedding. Trim and drop empties.
    let mut ent_jobs: Vec<(uuid::Uuid, String)> = Vec::new();
    for e in &entities {
        if !body.force && e.embedding.is_some() {
            skipped += 1;
            continue;
        }
        let text = e
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| e.entity_type.as_index_str())
            .trim()
            .to_string();
        if text.is_empty() {
            empty_skipped += 1;
            continue;
        }
        ent_jobs.push((e.id, text));
    }

    // Build (id, text) pairs for situations. Trim and drop empties.
    let mut sit_jobs: Vec<(uuid::Uuid, String)> = Vec::new();
    for s in &situations {
        if !body.force && s.embedding.is_some() {
            skipped += 1;
            continue;
        }
        // Build a richer text composed of name + description + first raw_content block.
        let mut parts: Vec<&str> = Vec::new();
        if let Some(n) = s.name.as_deref() {
            if !n.trim().is_empty() {
                parts.push(n);
            }
        }
        if let Some(d) = s.description.as_deref() {
            if !d.trim().is_empty() {
                parts.push(d);
            }
        }
        if let Some(rc) = s.raw_content.first() {
            if !rc.content.trim().is_empty() {
                parts.push(&rc.content);
            }
        }
        let text = parts.join(" ").trim().to_string();
        if text.is_empty() {
            empty_skipped += 1;
            continue;
        }
        sit_jobs.push((s.id, text));
    }

    // Per-row embed: skip empties, catch errors, log via tracing, accumulate at most MAX_ERRORS.
    let mut entities_embedded: usize = 0;
    for (id, text) in &ent_jobs {
        match embedder.embed_text(text) {
            Ok(emb) => {
                if state
                    .hypergraph
                    .update_entity_no_snapshot(id, |e| {
                        e.embedding = Some(emb.clone());
                    })
                    .is_ok()
                {
                    entities_embedded += 1;
                    if let Some(vi) = &state.vector_index {
                        if let Ok(mut idx) = vi.write() {
                            let _ = idx.add(*id, &emb);
                        }
                    }
                }
            }
            Err(e) => {
                failed += 1;
                tracing::warn!("embedding_backfill: entity {} failed: {}", id, e);
                if errors.len() < MAX_ERRORS {
                    errors.push(serde_json::json!({
                        "kind": "entity",
                        "id": id.to_string(),
                        "error": e.to_string(),
                    }));
                }
            }
        }
    }

    let mut situations_embedded: usize = 0;
    for (id, text) in &sit_jobs {
        match embedder.embed_text(text) {
            Ok(emb) => {
                if state
                    .hypergraph
                    .update_situation(id, |s| {
                        s.embedding = Some(emb.clone());
                    })
                    .is_ok()
                {
                    situations_embedded += 1;
                    if let Some(vi) = &state.vector_index {
                        if let Ok(mut idx) = vi.write() {
                            let _ = idx.add(*id, &emb);
                        }
                    }
                }
            }
            Err(e) => {
                failed += 1;
                tracing::warn!("embedding_backfill: situation {} failed: {}", id, e);
                if errors.len() < MAX_ERRORS {
                    errors.push(serde_json::json!({
                        "kind": "situation",
                        "id": id.to_string(),
                        "error": e.to_string(),
                    }));
                }
            }
        }
    }

    // Persist the updated vector index.
    if let Some(vi) = &state.vector_index {
        if let Ok(idx) = vi.read() {
            if let Err(e) = idx.save(state.hypergraph.store()) {
                tracing::warn!("Failed to save vector index after backfill: {}", e);
            }
        }
    }

    json_ok(&serde_json::json!({
        "entities_embedded": entities_embedded,
        "situations_embedded": situations_embedded,
        "total_updated": entities_embedded + situations_embedded,
        "skipped": skipped,
        "empty_skipped": empty_skipped,
        "failed": failed,
        "errors": errors,
    }))
}

// ─── Location → Setting Backfill ────────────────────────────────

#[derive(Deserialize)]
pub struct BackfillSettingsRequest {
    /// When set, restricts the scan to one narrative; otherwise all
    /// narratives in the workspace are scanned.
    pub narrative_id: Option<String>,
}

/// POST /entities/backfill-settings — Add missing `Setting` participations.
///
/// Some narratives were ingested before the LLM extraction prompt enforced
/// the rule "every Location appearing in a situation MUST be a participant
/// with role Setting". For those narratives the Locations exist as entities
/// but no Participation row ties them to the situations where they appear,
/// so the Inspector's Relations tab on a Location shows nothing.
///
/// This deterministic backfill scans every situation's prose for any of
/// each Location's name + alias terms (case-insensitive, ASCII-word-bounded
/// match, terms < 3 chars skipped to avoid noise). When a match is found
/// and no participation already exists for the (location, situation) pair,
/// it inserts a fresh `Participation { role: Custom("Setting"), … }`.
///
/// Idempotent — re-running it after a partial pass only adds the
/// still-missing rows. Pure data backfill, no LLM calls.
pub async fn backfill_location_settings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BackfillSettingsRequest>,
) -> impl IntoResponse {
    let locations: Vec<Entity> = match &body.narrative_id {
        Some(nid) => state
            .hypergraph
            .list_entities_by_narrative(nid)
            .unwrap_or_default()
            .into_iter()
            .filter(|e| matches!(e.entity_type, EntityType::Location))
            .collect(),
        None => state
            .hypergraph
            .list_entities_by_type(&EntityType::Location)
            .unwrap_or_default(),
    };
    let situations: Vec<Situation> = match &body.narrative_id {
        Some(nid) => state
            .hypergraph
            .list_situations_by_narrative(nid)
            .unwrap_or_default(),
        None => state
            .hypergraph
            .list_situations_by_maturity(MaturityLevel::Candidate)
            .unwrap_or_default(),
    };

    // Per-location term list: name + any string in `properties.aliases`.
    // Terms shorter than 3 bytes are dropped — they trigger too many
    // false-positive matches against ordinary words.
    let candidates: Vec<(Uuid, Vec<String>)> = locations
        .iter()
        .map(|loc| {
            let mut terms: Vec<String> = Vec::new();
            if let Some(name) = loc.properties.get("name").and_then(|v| v.as_str()) {
                terms.push(name.to_lowercase());
            }
            if let Some(arr) = loc.properties.get("aliases").and_then(|v| v.as_array()) {
                for a in arr {
                    if let Some(s) = a.as_str() {
                        terms.push(s.to_lowercase());
                    }
                }
            }
            terms.retain(|t| t.len() >= 3);
            (loc.id, terms)
        })
        .filter(|(_, terms)| !terms.is_empty())
        .collect();

    let mut links_created: usize = 0;
    let mut skipped_existing: usize = 0;
    let mut errors: Vec<String> = Vec::new();

    for sit in &situations {
        let mut hay = String::new();
        if let Some(n) = &sit.name {
            hay.push_str(n);
            hay.push(' ');
        }
        if let Some(d) = &sit.description {
            hay.push_str(d);
            hay.push(' ');
        }
        if let Some(spat) = &sit.spatial {
            if let Some(d) = &spat.description {
                hay.push_str(d);
                hay.push(' ');
            }
            if let Some(n) = &spat.location_name {
                hay.push_str(n);
                hay.push(' ');
            }
        }
        for cb in &sit.raw_content {
            hay.push_str(&cb.content);
            hay.push(' ');
        }
        if hay.trim().is_empty() {
            continue;
        }
        let hay_lower = hay.to_lowercase();

        for (loc_id, terms) in &candidates {
            // Already participates in any role → skip; the Inspector picks
            // up the existing link without us creating a duplicate.
            match state.hypergraph.get_participations_for_pair(loc_id, &sit.id) {
                Ok(existing) if !existing.is_empty() => {
                    skipped_existing += 1;
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    errors.push(format!("pair {}↔{}: {}", loc_id, sit.id, e));
                    continue;
                }
            }
            let matched = terms.iter().any(|t| word_bounded_contains(&hay_lower, t));
            if !matched {
                continue;
            }
            let part = Participation {
                entity_id: *loc_id,
                situation_id: sit.id,
                role: Role::Custom("Setting".into()),
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            };
            match state.hypergraph.add_participant(part) {
                Ok(()) => links_created += 1,
                Err(e) => errors.push(format!("add {}↔{}: {}", loc_id, sit.id, e)),
            }
        }
    }

    json_ok(&serde_json::json!({
        "locations_scanned": candidates.len(),
        "situations_scanned": situations.len(),
        "links_created": links_created,
        "skipped_existing_pairs": skipped_existing,
        "errors": errors,
    }))
}

/// Case-insensitive substring match with ASCII word boundaries — the needle
/// must be flanked by a non-alphanumeric ASCII byte (or be at the haystack
/// edge). Both inputs are expected pre-lowercased. Multi-byte UTF-8 chars
/// are treated as boundaries, which is fine for matching `paris` inside
/// `château de paris,` — French diacritics are ≥0x80 bytes that
/// `is_ascii_alphanumeric` rejects, keeping the boundary check sound.
fn word_bounded_contains(hay: &str, needle: &str) -> bool {
    if needle.is_empty() || hay.len() < needle.len() {
        return false;
    }
    let bytes = hay.as_bytes();
    let nlen = needle.len();
    let mut start = 0usize;
    while let Some(rel) = hay[start..].find(needle) {
        let abs = start + rel;
        let prev_ok = abs == 0 || !bytes[abs - 1].is_ascii_alphanumeric();
        let after = abs + nlen;
        let next_ok = after >= bytes.len() || !bytes[after].is_ascii_alphanumeric();
        if prev_ok && next_ok {
            return true;
        }
        start = abs + 1;
        if start >= hay.len() {
            break;
        }
    }
    false
}

// ─── Job Lineage & Batch Chunk Operations ──────────────────────

/// GET /ingest/jobs/:id/lineage — Return the full job lineage chain (parent + children).
pub async fn get_job_lineage(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    // Get the target job
    let target = match state.ingestion_jobs.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };

    // Walk up parent chain to find root
    let mut root_id = job_id.clone();
    let mut seen = std::collections::HashSet::new();
    seen.insert(root_id.clone());
    {
        let mut current = target.clone();
        while let Some(ref pid) = current.parent_job_id {
            if !seen.insert(pid.clone()) {
                break; // cycle guard
            }
            root_id = pid.clone();
            match state.ingestion_jobs.get_job(pid) {
                Ok(j) => current = j,
                Err(_) => break,
            }
        }
    }

    // Collect all jobs — scan recent jobs and filter those in the lineage tree
    let all_jobs = match state.ingestion_jobs.list_all(500) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };

    // Build a set of IDs in this lineage: start from root, find all children recursively
    let mut lineage_ids = std::collections::HashSet::new();
    lineage_ids.insert(root_id.clone());
    // Multiple passes to catch deep chains (max depth 10)
    for _ in 0..10 {
        let before = lineage_ids.len();
        for j in &all_jobs {
            if let Some(ref pid) = j.parent_job_id {
                if lineage_ids.contains(pid) {
                    lineage_ids.insert(j.id.clone());
                }
            }
        }
        if lineage_ids.len() == before {
            break;
        }
    }

    let lineage_jobs: Vec<serde_json::Value> = all_jobs
        .iter()
        .filter(|j| lineage_ids.contains(&j.id))
        .map(|j| {
            serde_json::json!({
                "id": j.id,
                "status": j.status,
                "created_at": j.created_at,
                "parent_job_id": j.parent_job_id,
                "source_name": j.source_name,
                "report": j.report,
                "pass_mode": j.pass_mode,
                "enrich": j.enrich,
            })
        })
        .collect();

    json_ok(&serde_json::json!({
        "root_job_id": root_id,
        "target_job_id": job_id,
        "jobs": lineage_jobs,
    }))
}

/// POST /ingest/jobs/:id/chunks/batch — Batch operations on selected chunks.
pub async fn batch_chunk_action(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let chunk_indices: Vec<usize> = match body.get("chunk_indices").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "chunk_indices array required" })),
            )
                .into_response()
        }
    };

    if chunk_indices.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "chunk_indices must not be empty" })),
        )
            .into_response();
    }

    if chunk_indices.len() > 200 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Maximum 200 chunks per batch" })),
        )
            .into_response();
    }

    let action = body
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("reextract");
    let context_mode = body
        .get("context_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("selected");

    // Validate the job exists
    let job = match state.ingestion_jobs.get_job(&job_id) {
        Ok(j) => j,
        Err(e) => return error_response(e).into_response(),
    };

    match action {
        "reextract" => {
            batch_reextract(state, &job_id, &job, &chunk_indices, context_mode).await
        }
        "reprocess" => batch_reprocess(state, &job_id, &job, &chunk_indices).await,
        "enrich" => batch_enrich(state, &job_id, &job, &chunk_indices).await,
        "reconcile" => batch_reconcile(state, &job_id, &job, &chunk_indices).await,
        _ => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("Unknown action: {}. Expected: reextract, reprocess, enrich, reconcile", action)
            })),
        )
            .into_response(),
    }
}

/// Resolve the LLM extractor from config or global state.
fn resolve_extractor(
    state: &AppState,
) -> Result<Arc<dyn crate::ingestion::llm::NarrativeExtractor>, axum::response::Response> {
    let ic = state
        .ingestion_config
        .read()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(e) = ic.pass1.build_extractor() {
        return Ok(e);
    }
    state
        .extractor
        .read()
        .ok()
        .and_then(|g| g.as_ref().map(Arc::clone))
        .ok_or_else(|| {
            error_response(TensaError::LlmError("No LLM provider configured".into()))
                .into_response()
        })
}

/// Load source text and chunk manifest for a job, or return error response.
fn load_text_and_manifest(
    state: &AppState,
    job_id: &str,
) -> Result<(String, Vec<crate::ingestion::jobs::ChunkManifestEntry>), axum::response::Response> {
    let text = match state.ingestion_jobs.get_source_text(job_id) {
        Ok(Some(t)) => t,
        Ok(None) => {
            return Err(
                error_response(TensaError::Internal("Source text not found".into()))
                    .into_response(),
            )
        }
        Err(e) => return Err(error_response(e).into_response()),
    };
    let manifest = match state.ingestion_jobs.get_chunk_manifest(job_id) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return Err(
                error_response(TensaError::Internal("Chunk manifest not found".into()))
                    .into_response(),
            )
        }
        Err(e) => return Err(error_response(e).into_response()),
    };
    Ok((text, manifest))
}

/// Reconstruct a TextChunk from source text and manifest entry.
fn reconstruct_chunk(
    text: &str,
    entry: &crate::ingestion::jobs::ChunkManifestEntry,
) -> crate::ingestion::chunker::TextChunk {
    let overlap_start = if entry.overlap_bytes > 0 && entry.start >= entry.overlap_bytes {
        entry.start - entry.overlap_bytes
    } else {
        entry.start
    };
    let overlap_prefix = if overlap_start < entry.start {
        text[overlap_start..entry.start].to_string()
    } else {
        String::new()
    };
    crate::ingestion::chunker::TextChunk {
        chunk_id: entry.chunk_index,
        text: text[entry.start..entry.end].to_string(),
        chapter: entry.chapter.clone(),
        start_offset: entry.start,
        end_offset: entry.end,
        overlap_prefix,
    }
}

/// Build a HashMap index from manifest entries for O(1) lookup by chunk_index.
fn index_manifest(
    manifest: &[crate::ingestion::jobs::ChunkManifestEntry],
) -> std::collections::HashMap<usize, &crate::ingestion::jobs::ChunkManifestEntry> {
    manifest.iter().map(|e| (e.chunk_index, e)).collect()
}

/// Batch re-extract: re-run LLM extraction on selected chunks.
async fn batch_reextract(
    state: Arc<AppState>,
    job_id: &str,
    _job: &crate::ingestion::jobs::IngestionJob,
    indices: &[usize],
    context_mode: &str,
) -> axum::response::Response {
    let (text, manifest) = match load_text_and_manifest(&state, job_id) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let extractor = match resolve_extractor(&state) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let manifest_map = index_manifest(&manifest);

    // Build context chunk indices based on context_mode
    let context_indices: Vec<usize> = match context_mode {
        "all" => (0..manifest.len()).collect(),
        "neighbors" => {
            let mut set = std::collections::BTreeSet::new();
            for &idx in indices {
                if idx > 0 {
                    set.insert(idx - 1);
                }
                set.insert(idx);
                if idx + 1 < manifest.len() {
                    set.insert(idx + 1);
                }
            }
            set.into_iter().collect()
        }
        _ => indices.to_vec(),
    };

    // Build known entity names from context chunks (for cross-chunk coherence)
    let known_names: Vec<String> = context_indices
        .iter()
        .filter_map(|&i| {
            state
                .ingestion_jobs
                .get_chunk_extraction(job_id, i)
                .ok()
                .flatten()
        })
        .flat_map(|r| r.extraction.entities.into_iter().map(|e| e.name))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut results = Vec::new();

    for &idx in indices {
        let entry = match manifest_map.get(&idx) {
            Some(e) => *e,
            None => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": "Chunk not found in manifest"
                }));
                continue;
            }
        };
        if entry.end > text.len() {
            results.push(serde_json::json!({
                "chunk_index": idx, "status": "error", "error": "Chunk byte range out of bounds"
            }));
            continue;
        }

        let chunk = reconstruct_chunk(&text, entry);

        // "selected" mode: no cross-chunk entity context; others: full context
        let names_for_extraction = if context_mode == "selected" {
            &[][..]
        } else {
            &known_names[..]
        };

        match extractor.extract_with_logging(&chunk, names_for_extraction) {
            Ok((extraction, exchange)) => {
                let attempt = state
                    .ingestion_jobs
                    .get_logs_for_chunk(job_id, idx)
                    .map(|logs| logs.len() as u8)
                    .unwrap_or(0);

                if let Some(ref ex) = exchange {
                    let log = crate::ingestion::jobs::LlmCallLog {
                        job_id: job_id.to_string(),
                        chunk_index: idx,
                        pass: 1,
                        attempt,
                        system_prompt: ex.system_prompt.clone(),
                        user_prompt: ex.user_prompt.clone(),
                        raw_response: ex.raw_response.clone(),
                        parsed_extraction: Some(extraction.clone()),
                        parse_error: ex.parse_error.clone(),
                        retry_prompt: ex.retry_prompt.clone(),
                        retry_response: ex.retry_response.clone(),
                        duration_ms: ex.duration_ms,
                        model: ex.model.clone(),
                        endpoint: ex.endpoint.clone(),
                        timestamp: chrono::Utc::now(),
                    };
                    let _ = state.ingestion_jobs.store_llm_log(&log);
                }

                let record = crate::ingestion::jobs::ChunkExtractionRecord {
                    job_id: job_id.to_string(),
                    chunk_index: idx,
                    extraction: extraction.clone(),
                    gate_decisions: vec![],
                    entity_map: std::collections::HashMap::new(),
                    situation_ids: vec![],
                };
                let _ = state.ingestion_jobs.store_chunk_extraction(&record);

                results.push(serde_json::json!({
                    "chunk_index": idx,
                    "status": "ok",
                    "entities": extraction.entities.len(),
                    "situations": extraction.situations.len(),
                }));
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "chunk_index": idx,
                    "status": "error",
                    "error": e.to_string(),
                }));
            }
        }
    }

    json_ok(&serde_json::json!({
        "action": "reextract",
        "context_mode": context_mode,
        "results": results,
    }))
}

/// Batch reprocess: rollback + re-gate selected chunks (no LLM call).
async fn batch_reprocess(
    state: Arc<AppState>,
    job_id: &str,
    job: &crate::ingestion::jobs::IngestionJob,
    indices: &[usize],
) -> axum::response::Response {
    let ingestion_config = match state.ingestion_config.read() {
        Ok(cfg) => cfg.clone(),
        Err(_) => {
            return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
        }
    };

    let hg = Arc::new(crate::hypergraph::Hypergraph::new(
        state.hypergraph.store_arc(),
    ));
    let queue = Arc::new(crate::ingestion::queue::ValidationQueue::new(
        state.hypergraph.store_arc(),
    ));
    let extractor: Arc<dyn crate::ingestion::llm::NarrativeExtractor> =
        Arc::new(crate::ingestion::llm::MockExtractor::empty());

    let config = crate::ingestion::pipeline::PipelineConfig {
        chunker: ingestion_config.chunker_config(),
        auto_commit_threshold: ingestion_config.auto_commit_threshold,
        review_threshold: ingestion_config.review_threshold,
        source_id: job.narrative_id.clone().unwrap_or_else(|| "unknown".into()),
        source_type: "text".into(),
        narrative_id: job.narrative_id.clone(),
        job_id: Some(job_id.to_string()),
        concurrency: 1,
        strip_boilerplate: false,
        enrich: ingestion_config.enrich,
        single_session: false,
        session_max_context_tokens: 0,
        debug: false,
        cascade_mode: Default::default(),
        post_ingest_mamdani_rule_id: None,
    };

    let pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
        hg,
        extractor,
        state.embedder.read().unwrap().clone(),
        state.vector_index.clone(),
        queue,
        config,
    )
    .with_job_queue(Arc::clone(&state.ingestion_jobs));

    let mut resolver = crate::ingestion::resolve::EntityResolver::new();
    let provenance = crate::types::SourceReference {
        source_type: "text".into(),
        source_id: Some(job.narrative_id.clone().unwrap_or_default()),
        description: Some(job.source_name.clone()),
        timestamp: chrono::Utc::now(),
        registered_source: None,
    };

    let mut results = Vec::new();
    for &idx in indices {
        let record = match state.ingestion_jobs.get_chunk_extraction(job_id, idx) {
            Ok(Some(r)) => r,
            Ok(None) => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": "No extraction record"
                }));
                continue;
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": e.to_string()
                }));
                continue;
            }
        };

        // Rollback
        let mut ent_del = 0usize;
        let mut sit_del = 0usize;
        for (_name, id) in &record.entity_map {
            if state.hypergraph.delete_entity(id).is_ok() {
                ent_del += 1;
            }
        }
        for id in &record.situation_ids {
            if state.hypergraph.delete_situation(id).is_ok() {
                sit_del += 1;
            }
        }

        match pipeline.process_extraction_standalone(
            &record.extraction,
            idx,
            &mut resolver,
            &provenance,
        ) {
            Ok(gate_report) => {
                results.push(serde_json::json!({
                    "chunk_index": idx,
                    "status": "ok",
                    "rolled_back": { "entities": ent_del, "situations": sit_del },
                    "reprocessed": {
                        "entities_created": gate_report.entity_ids.len(),
                        "situations_created": gate_report.situation_ids.len(),
                        "auto_committed": gate_report.auto_committed,
                        "queued": gate_report.queued,
                        "rejected": gate_report.rejected,
                    },
                }));
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": e.to_string()
                }));
            }
        }
    }

    json_ok(&serde_json::json!({
        "action": "reprocess",
        "results": results,
    }))
}

/// Batch enrich: run enrichment step 2 on selected chunks.
async fn batch_enrich(
    state: Arc<AppState>,
    job_id: &str,
    _job: &crate::ingestion::jobs::IngestionJob,
    indices: &[usize],
) -> axum::response::Response {
    let extractor = match resolve_extractor(&state) {
        Ok(e) => e,
        Err(r) => return r,
    };
    let (text, manifest) = match load_text_and_manifest(&state, job_id) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let manifest_map = index_manifest(&manifest);

    let mut results = Vec::new();

    for &idx in indices {
        let record = match state.ingestion_jobs.get_chunk_extraction(job_id, idx) {
            Ok(Some(r)) => r,
            Ok(None) => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": "No extraction record"
                }));
                continue;
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": e.to_string()
                }));
                continue;
            }
        };

        let entry = match manifest_map.get(&idx) {
            Some(e) => *e,
            None => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": "Chunk not in manifest"
                }));
                continue;
            }
        };
        if entry.end > text.len() {
            results.push(serde_json::json!({
                "chunk_index": idx, "status": "error", "error": "Out of bounds"
            }));
            continue;
        }

        let chunk = reconstruct_chunk(&text, entry);

        match extractor.enrich_extraction(&chunk, &record.extraction) {
            Ok(enrichment) => {
                results.push(serde_json::json!({
                    "chunk_index": idx,
                    "status": "ok",
                    "enrichment_fields": {
                        "entity_beliefs": enrichment.entity_beliefs.len(),
                        "game_structures": enrichment.game_structures.len(),
                        "discourse": enrichment.discourse.len(),
                        "extra_causal_links": enrichment.extra_causal_links.len(),
                        "temporal_chain": enrichment.temporal_chain.len(),
                    },
                }));
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": e.to_string()
                }));
            }
        }
    }

    json_ok(&serde_json::json!({
        "action": "enrich",
        "results": results,
    }))
}

/// Batch reconcile: run temporal reconciliation on selected chunks.
async fn batch_reconcile(
    state: Arc<AppState>,
    job_id: &str,
    _job: &crate::ingestion::jobs::IngestionJob,
    indices: &[usize],
) -> axum::response::Response {
    let extractor = match resolve_extractor(&state) {
        Ok(e) => e,
        Err(r) => return r,
    };

    // Collect chunk summaries; skip missing chunks with per-chunk errors
    let mut chunk_summaries: Vec<(usize, Vec<(String, Option<String>, Option<String>)>)> =
        Vec::new();
    let mut skipped = Vec::new();
    for &idx in indices {
        match state.ingestion_jobs.get_chunk_extraction(job_id, idx) {
            Ok(Some(r)) => {
                let sit_summaries: Vec<(String, Option<String>, Option<String>)> = r
                    .extraction
                    .situations
                    .iter()
                    .map(|s| (s.description.clone(), s.temporal_marker.clone(), None))
                    .collect();
                chunk_summaries.push((idx, sit_summaries));
            }
            Ok(None) => {
                skipped.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": "No extraction record"
                }));
            }
            Err(e) => {
                skipped.push(serde_json::json!({
                    "chunk_index": idx, "status": "error", "error": e.to_string()
                }));
            }
        }
    }

    if chunk_summaries.is_empty() {
        return json_ok(&serde_json::json!({
            "action": "reconcile",
            "chunks_reconciled": 0,
            "skipped": skipped,
        }));
    }

    match extractor.reconcile_temporal(&chunk_summaries) {
        Ok(reconciliation) => json_ok(&serde_json::json!({
            "action": "reconcile",
            "chunks_reconciled": chunk_summaries.len(),
            "relations": reconciliation.relations.len(),
            "timeline_events": reconciliation.timeline.len(),
            "skipped": skipped,
        })),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /style-embeddings — List all stored style embeddings.
///
/// Returns 200 with `[]` when none exist. Workspace scoping follows the same
/// pattern as the other list handlers (no per-request workspace filter at the
/// route layer; `state.hypergraph` is the shared store). Feature-gated behind
/// `generation` because the underlying `style::embedding` module is gated too.
#[cfg(feature = "generation")]
pub async fn list_style_embeddings(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match crate::style::embedding::list_embeddings(&state.hypergraph) {
        Ok(items) => json_ok(&items),
        Err(e) => error_response(e).into_response(),
    }
}

/// Body for `POST /resolve/labels`.
#[derive(Deserialize)]
pub struct ResolveLabelsRequest {
    pub ids: Vec<Uuid>,
}

/// Per-id label entry returned by `POST /resolve/labels`.
#[derive(Serialize)]
pub struct ResolvedLabel {
    pub kind: ResolvedKind,
    pub label: String,
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ResolvedKind {
    Entity,
    Situation,
    Unknown,
}

/// `POST /resolve/labels` — bulk-resolve UUIDs to display labels.
///
/// Replaces the InferenceLab pattern of probing GET /entities/:id then
/// GET /situations/:id per row, which produced 404 noise for synth /
/// sentinel target_ids. Unknown ids return `kind = "unknown"` with a
/// short uuid prefix label — never a 404.
pub async fn resolve_labels(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResolveLabelsRequest>,
) -> impl IntoResponse {
    let mut out: std::collections::HashMap<String, ResolvedLabel> =
        std::collections::HashMap::with_capacity(req.ids.len());
    for id in req.ids {
        let key = id.to_string();
        let short = format!("{}...", &key[..key.len().min(12)]);
        let label = if let Ok(entity) = state.hypergraph.get_entity(&id) {
            let name = entity
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from);
            let label = match name {
                Some(n) => format!("{n} ({})", &key[..8]),
                None => short.clone(),
            };
            ResolvedLabel {
                kind: ResolvedKind::Entity,
                label,
            }
        } else if let Ok(situation) = state.hypergraph.get_situation(&id) {
            let label = situation_label(&situation, &key);
            ResolvedLabel {
                kind: ResolvedKind::Situation,
                label,
            }
        } else {
            ResolvedLabel {
                kind: ResolvedKind::Unknown,
                label: short,
            }
        };
        out.insert(key, label);
    }
    json_ok(&out)
}

fn situation_label(s: &Situation, uuid: &str) -> String {
    let prefix = &uuid[..8.min(uuid.len())];
    if let Some(name) = &s.name {
        return format!("{name} ({prefix})");
    }
    if let Some(desc) = &s.description {
        let truncated = if desc.len() > 60 {
            format!("{}...", &desc[..57])
        } else {
            desc.clone()
        };
        return format!("{truncated} ({prefix})");
    }
    if let Some(text) = s.raw_content.first().map(|cb| &cb.content) {
        let truncated = if text.len() > 60 {
            format!("{}...", &text[..57])
        } else {
            text.clone()
        };
        return format!("{truncated} ({prefix})");
    }
    format!("[{:?}] ({prefix})", s.narrative_level)
}

#[cfg(test)]
mod situation_patch_tests {
    use super::*;

    #[test]
    fn patch_deserializes_omitted_fields_as_none() {
        let json = serde_json::json!({});
        let p: SituationPatch = serde_json::from_value(json).unwrap();
        assert!(p.name.is_none());
        assert!(p.description.is_none());
        assert!(p.narrative_level.is_none());
        assert!(p.narrative_id.is_none());
        assert!(p.confidence.is_none());
        assert!(p.raw_content.is_none());
        assert!(p.temporal.is_none());
        assert!(p.spatial.is_none());
    }

    #[test]
    fn patch_distinguishes_null_from_omitted_for_name() {
        let cleared: SituationPatch =
            serde_json::from_value(serde_json::json!({ "name": null })).unwrap();
        assert_eq!(cleared.name, Some(None));

        let set: SituationPatch =
            serde_json::from_value(serde_json::json!({ "name": "Chapter 1" })).unwrap();
        assert_eq!(set.name, Some(Some("Chapter 1".into())));
    }

    #[test]
    fn patch_distinguishes_null_from_omitted_for_narrative_id() {
        let cleared: SituationPatch =
            serde_json::from_value(serde_json::json!({ "narrative_id": null })).unwrap();
        assert_eq!(cleared.narrative_id, Some(None));

        let set: SituationPatch =
            serde_json::from_value(serde_json::json!({ "narrative_id": "hamlet" })).unwrap();
        assert_eq!(set.narrative_id, Some(Some("hamlet".into())));
    }

    #[test]
    fn patch_distinguishes_null_from_omitted_for_spatial() {
        let cleared: SituationPatch =
            serde_json::from_value(serde_json::json!({ "spatial": null })).unwrap();
        assert!(matches!(cleared.spatial, Some(None)));
    }

    #[test]
    fn patch_accepts_full_raw_content_array() {
        let json = serde_json::json!({
            "raw_content": [
                { "content_type": "Text", "content": "She stood at the window." },
                { "content_type": "Dialogue", "content": "\"Hello,\" said Alice." },
                { "content_type": "Observation", "content": "Her voice wavered." },
            ]
        });
        let p: SituationPatch = serde_json::from_value(json).unwrap();
        let blocks = p.raw_content.expect("raw_content should be Some");
        assert_eq!(blocks.len(), 3);
        assert!(matches!(blocks[0].content_type, ContentType::Text));
        assert!(matches!(blocks[1].content_type, ContentType::Dialogue));
        assert!(matches!(blocks[2].content_type, ContentType::Observation));
    }

    #[test]
    fn patch_accepts_narrative_level_tag() {
        let json = serde_json::json!({ "narrative_level": "Arc" });
        let p: SituationPatch = serde_json::from_value(json).unwrap();
        assert!(matches!(p.narrative_level, Some(NarrativeLevel::Arc)));
    }

    #[test]
    fn validate_raw_content_rejects_empty_array() {
        assert!(matches!(
            validate_raw_content(&[]),
            Err(TensaError::InvalidInput(_))
        ));
    }

    #[test]
    fn validate_raw_content_rejects_whitespace_only_prose() {
        let blocks = vec![ContentBlock {
            content_type: ContentType::Text,
            content: "   \n\t".into(),
            source: None,
        }];
        assert!(matches!(
            validate_raw_content(&blocks),
            Err(TensaError::InvalidInput(_))
        ));
    }

    #[test]
    fn validate_raw_content_allows_media_ref_with_empty_content() {
        // MediaRef / Document blocks legitimately carry no prose.
        let blocks = vec![ContentBlock {
            content_type: ContentType::MediaRef,
            content: String::new(),
            source: None,
        }];
        assert!(validate_raw_content(&blocks).is_ok());
    }

    #[test]
    fn validate_raw_content_accepts_real_prose() {
        let blocks = vec![
            ContentBlock::text("She hesitated."),
            ContentBlock {
                content_type: ContentType::Dialogue,
                content: "\"Not yet.\"".into(),
                source: None,
            },
        ];
        assert!(validate_raw_content(&blocks).is_ok());
    }

    #[test]
    fn patch_preserves_temporal_round_trip() {
        use chrono::TimeZone;
        let start = chrono::Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap();
        let json = serde_json::json!({
            "temporal": {
                "start": start.to_rfc3339(),
                "end": null,
                "granularity": "Approximate",
                "relations": []
            }
        });
        let p: SituationPatch = serde_json::from_value(json).unwrap();
        let temporal = p.temporal.expect("temporal should be Some");
        assert_eq!(temporal.start, Some(start));
        assert!(temporal.end.is_none());
    }

    #[test]
    fn patch_applies_each_field_via_update_situation_closure() {
        use crate::store::memory::MemoryStore;
        use crate::Hypergraph;
        use chrono::Utc;

        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some("old".into()),
            description: Some("old desc".into()),
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: Some(SpatialAnchor {
                latitude: None,
                longitude: None,
                precision: SpatialPrecision::Unknown,
                location_entity: None,
                location_name: Some("old location".into()),
                description: None,
                geo_provenance: None,
            }),
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.5,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("old-id".into()),
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

        // Apply the same closure the handler uses.
        let patch = SituationPatch {
            name: Some(Some("new name".into())),
            description: Some(None), // clear
            narrative_level: Some(NarrativeLevel::Arc),
            narrative_id: Some(Some("new-id".into())),
            confidence: Some(0.9),
            raw_content: Some(vec![ContentBlock {
                content_type: ContentType::Text,
                content: "prose".into(),
                source: None,
            }]),
            temporal: None,
            spatial: Some(None), // clear
            discourse: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: None,
            game_structure: None,
            deterministic: None,
            probabilistic: None,
            properties: None,
        };
        let updated = hg
            .update_situation(&id, |s| {
                if let Some(name) = patch.name.clone() {
                    s.name = name;
                }
                if let Some(description) = patch.description.clone() {
                    s.description = description;
                }
                if let Some(level) = patch.narrative_level {
                    s.narrative_level = level;
                }
                if let Some(narrative_id) = patch.narrative_id.clone() {
                    s.narrative_id = narrative_id;
                }
                if let Some(c) = patch.confidence {
                    s.confidence = c;
                }
                if let Some(raw_content) = patch.raw_content.clone() {
                    s.raw_content = raw_content;
                }
                if let Some(spatial) = patch.spatial.clone() {
                    s.spatial = spatial;
                }
            })
            .unwrap();

        assert_eq!(updated.name.as_deref(), Some("new name"));
        assert_eq!(updated.description, None);
        assert!(matches!(updated.narrative_level, NarrativeLevel::Arc));
        assert_eq!(updated.narrative_id.as_deref(), Some("new-id"));
        assert!((updated.confidence - 0.9).abs() < 1e-6);
        assert_eq!(updated.raw_content.len(), 1);
        assert!(updated.spatial.is_none());
    }
}
