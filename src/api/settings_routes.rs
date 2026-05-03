//! Settings API routes — model discovery and ingestion pipeline configuration.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::error_response;
use crate::api::server::AppState;
use crate::error::TensaError;
#[cfg(feature = "embedding")]
use crate::ingestion::embed::EmbeddingProvider;

// ─── KV Keys for Persistent Settings ────────────────────────

/// KV key for persisted ingestion config.
const CFG_INGESTION_KEY: &[u8] = b"cfg/ingestion";
/// KV key for persisted LLM config.
pub const CFG_LLM_KEY: &[u8] = b"cfg/llm";
/// KV key for persisted inference engine config.
const CFG_INFERENCE_KEY: &[u8] = b"cfg/inference";
/// KV key for persisted RAG config.
const CFG_RAG_KEY: &[u8] = b"cfg/rag";
/// KV key for persisted inference/RAG LLM config.
pub const CFG_INFERENCE_LLM_KEY: &[u8] = b"cfg/inference_llm";
/// KV key for persisted image-generation provider config.
pub const CFG_IMAGE_GEN_KEY: &[u8] = b"cfg/image_gen";

/// Load persisted image-gen config from KV store (if any).
pub fn load_persisted_image_gen_config(
    store: &dyn crate::store::KVStore,
) -> Option<crate::images::ImageGenConfig> {
    store
        .get(CFG_IMAGE_GEN_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// Load persisted ingestion config from KV store (if any).
pub fn load_persisted_ingestion_config(
    store: &dyn crate::store::KVStore,
) -> Option<crate::ingestion::config::IngestionConfig> {
    store
        .get(CFG_INGESTION_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// Load persisted LLM config from KV store (if any).
pub fn load_persisted_llm_config(
    store: &dyn crate::store::KVStore,
) -> Option<crate::api::server::LlmConfig> {
    store
        .get(CFG_LLM_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// Persist a serializable config value to the KV store, logging on failure.
pub fn persist_config(
    store: &dyn crate::store::KVStore,
    key: &[u8],
    value: &impl serde::Serialize,
    label: &str,
) {
    match serde_json::to_vec(value) {
        Ok(bytes) => {
            if let Err(e) = store.put(key, &bytes) {
                tracing::warn!("Failed to persist {} config: {}", label, e);
            }
        }
        Err(e) => tracing::warn!(
            "Failed to serialize {} config for persistence: {}",
            label,
            e
        ),
    }
}

// ─── Model Discovery ─────────────────────────────────────────

/// Query parameters for model discovery.
#[derive(Debug, Deserialize)]
pub struct ModelDiscoveryQuery {
    /// Base URL of the inference server to probe.
    pub url: String,
}

/// A discovered model from an inference server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    /// Model identifier (e.g. "qwen3.5:4b" for Ollama, "Qwen/Qwen3.5-27B" for vLLM).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Size in bytes (if reported).
    pub size_bytes: Option<u64>,
    /// Additional details from the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Response from model discovery.
#[derive(Debug, Serialize)]
pub struct ModelDiscoveryResponse {
    /// Detected provider type.
    pub provider: String,
    /// Available models.
    pub models: Vec<DiscoveredModel>,
    /// Whether the server responded successfully.
    pub online: bool,
}

/// GET /settings/models?url=http://localhost:11434
///
/// Probes an inference server and returns available models.
/// Tries Ollama's `/api/tags` first, then vLLM/OpenAI's `/v1/models`.
pub async fn discover_models(Query(query): Query<ModelDiscoveryQuery>) -> impl IntoResponse {
    let base_url = query.url.trim_end_matches('/').to_string();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    // Try Ollama first
    match probe_ollama(&client, &base_url).await {
        Ok(models) => {
            return Json(ModelDiscoveryResponse {
                provider: "ollama".into(),
                models,
                online: true,
            })
            .into_response();
        }
        Err(_) => { /* fall through */ }
    }

    // Try OpenAI-compatible endpoints: vLLM, LiteLLM, and similar proxies
    for (path, provider) in &[
        ("/v1/models", "vllm"),
        ("/api/models", "litellm"),
        ("/models", "litellm"),
    ] {
        match probe_openai_models(&client, &base_url, path).await {
            Ok(models) => {
                return Json(ModelDiscoveryResponse {
                    provider: provider.to_string(),
                    models,
                    online: true,
                })
                .into_response();
            }
            Err(_) => { /* try next */ }
        }
    }

    // Server unreachable or unrecognized
    Json(ModelDiscoveryResponse {
        provider: "unknown".into(),
        models: vec![],
        online: false,
    })
    .into_response()
}

/// Probe Ollama's `/api/tags` endpoint.
async fn probe_ollama(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<DiscoveredModel>, TensaError> {
    let url = format!("{}/api/tags", base_url);

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| TensaError::Internal(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(TensaError::Internal("Ollama API returned non-200".into()));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| TensaError::Internal(e.to_string()))?;

    let models = body["models"]
        .as_array()
        .ok_or_else(|| TensaError::Internal("No models array in Ollama response".into()))?;

    let mut result = Vec::new();
    for m in models {
        let name = m["name"].as_str().unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        result.push(DiscoveredModel {
            id: name.clone(),
            name: name.clone(),
            size_bytes: m["size"].as_u64(),
            details: m.get("details").cloned(),
        });
    }

    Ok(result)
}

/// Probe an OpenAI-compatible models endpoint at an arbitrary path.
/// Works for LiteLLM (`/api/models`, `/models`) and similar proxies.
async fn probe_openai_models(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
) -> Result<Vec<DiscoveredModel>, TensaError> {
    let url = format!("{}{}", base_url, path);

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| TensaError::Internal(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(TensaError::Internal(format!(
            "{} returned {}",
            path,
            resp.status()
        )));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| TensaError::Internal(e.to_string()))?;

    // OpenAI format: { "data": [ { "id": "..." }, ... ] }
    if let Some(data) = body["data"].as_array() {
        let mut result = Vec::new();
        for m in data {
            let id = m["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            result.push(DiscoveredModel {
                id: id.clone(),
                name: m["model_name"]
                    .as_str()
                    .or_else(|| m["name"].as_str())
                    .unwrap_or(&id)
                    .to_string(),
                size_bytes: None,
                details: m
                    .get("model_info")
                    .cloned()
                    .or_else(|| m.get("litellm_params").cloned()),
            });
        }
        return Ok(result);
    }

    // LiteLLM simple format: { "models": ["model-a", "model-b", ...] }
    if let Some(models) = body["models"].as_array() {
        let mut result = Vec::new();
        for m in models {
            if let Some(name) = m.as_str() {
                if !name.is_empty() {
                    result.push(DiscoveredModel {
                        id: name.to_string(),
                        name: name.to_string(),
                        size_bytes: None,
                        details: None,
                    });
                }
            }
        }
        return Ok(result);
    }

    Err(TensaError::Internal(format!(
        "No data or models array in {} response",
        path
    )))
}

// ─── Ingestion Config ─────────────────────────────────────────

/// GET /settings/ingestion — Return the current ingestion pipeline config.
pub async fn get_ingestion_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state
        .ingestion_config
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    match config {
        Ok(c) => Json(serde_json::json!({ "ingestion": c.redacted_hint() })).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/ingestion — Update the ingestion pipeline config at runtime.
pub async fn set_ingestion_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<crate::ingestion::config::IngestionConfig>,
) -> impl IntoResponse {
    // Merge API keys: if incoming key is a redacted hint or empty, keep the existing real key.
    // This prevents the Settings UI from accidentally wiping keys it never saw.
    let mut body = body;
    {
        if let Ok(current) = state.ingestion_config.read() {
            body.pass1 = body.pass1.merge_keys(&current.pass1);
            body.pass2 = body.pass2.merge_keys(&current.pass2);
        }
    }

    // Validate: try to build extractors from both pass configs
    if body.pass1 != crate::api::server::LlmConfig::None {
        if body.pass1.build_extractor().is_none() {
            return error_response(TensaError::Internal(
                "Failed to build Pass 1 extractor from config".into(),
            ))
            .into_response();
        }
    }
    if body.mode == crate::ingestion::config::PipelineMode::Multi
        && body.pass2 != crate::api::server::LlmConfig::None
    {
        if body.pass2.build_extractor().is_none() {
            return error_response(TensaError::Internal(
                "Failed to build Pass 2 extractor from config".into(),
            ))
            .into_response();
        }
    }

    // Validate numeric ranges
    if body.pass1_concurrency == 0 || body.pass1_concurrency > 64 {
        return error_response(TensaError::Internal(
            "pass1_concurrency must be between 1 and 64".into(),
        ))
        .into_response();
    }
    if body.pass2_window_size < 2 || body.pass2_window_size > 100 {
        return error_response(TensaError::Internal(
            "pass2_window_size must be between 2 and 100".into(),
        ))
        .into_response();
    }
    if body.pass2_window_overlap >= body.pass2_window_size {
        return error_response(TensaError::Internal(
            "pass2_window_overlap must be less than pass2_window_size".into(),
        ))
        .into_response();
    }
    if body.auto_commit_threshold <= body.review_threshold {
        return error_response(TensaError::Internal(
            "auto_commit_threshold must be greater than review_threshold".into(),
        ))
        .into_response();
    }

    let redacted = body.redacted_hint();

    // Also update the legacy extractor for single-pass backward compat
    {
        let new_extractor = body.pass1.build_extractor();
        let mut ext = match state.extractor.write() {
            Ok(e) => e,
            Err(_) => {
                return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
            }
        };
        *ext = new_extractor;
    }
    {
        let mut cfg = match state.llm_config.write() {
            Ok(c) => c,
            Err(_) => {
                return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
            }
        };
        *cfg = body.pass1.clone();
    }

    // Persist to KV store before updating in-memory config
    persist_config(
        state.hypergraph.store(),
        CFG_INGESTION_KEY,
        &body,
        "ingestion",
    );

    // Store new ingestion config in memory
    {
        let mut cfg = match state.ingestion_config.write() {
            Ok(c) => c,
            Err(_) => {
                return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
            }
        };
        *cfg = body;
    }

    tracing::info!("Ingestion config updated (persisted): {:?}", redacted.mode);
    Json(serde_json::json!({ "ingestion": redacted, "status": "ok" })).into_response()
}

/// GET /settings/presets — list available extraction prompt presets.
///
/// Returns a JSON array of preset descriptors. These are compiled-in, not stored in KV.
pub async fn list_presets() -> impl IntoResponse {
    use crate::ingestion::config::IngestionMode;

    let presets: Vec<_> = IngestionMode::PRESETS
        .iter()
        .map(|m| serde_json::json!({ "id": m.id_str(), "description": m.description() }))
        .collect();

    Json(presets).into_response()
}

/// GET /settings/embedding — report current embedding provider status.
pub async fn get_embedding_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let guard = state.embedder.read().unwrap();
    let model_name = state.embedder_model_name.read().unwrap().clone();
    let info = match guard.as_ref() {
        Some(e) => serde_json::json!({
            "enabled": true,
            "dimension": e.dimension(),
            "provider": e.provider_name(),
            "model": model_name,
        }),
        None => serde_json::json!({ "enabled": false, "model": "" }),
    };
    Json(info).into_response()
}

/// GET /settings/embedding/models — list available embedding models in models/embeddings/.
pub async fn list_embedding_models() -> impl IntoResponse {
    let models_dir = std::path::Path::new("models/embeddings");
    let mut models = Vec::new();

    if models_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let has_model = path.join("model.onnx").exists();
                    let has_tokenizer = path.join("tokenizer.json").exists();
                    let name = entry.file_name().to_string_lossy().to_string();
                    models.push(serde_json::json!({
                        "name": name,
                        "path": path.to_string_lossy(),
                        "ready": has_model && has_tokenizer,
                        "has_model": has_model,
                        "has_tokenizer": has_tokenizer,
                    }));
                }
            }
        }
    }

    Json(serde_json::json!({ "models": models, "models_dir": models_dir.to_string_lossy() }))
        .into_response()
}

/// PUT /settings/embedding — switch the active embedding model.
pub async fn set_embedding_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_name = body.get("model").and_then(|v| v.as_str());

    // "none" / null → disable embedder
    match model_name {
        None | Some("none") | Some("") => {
            *state.embedder.write().unwrap() = None;
            *state.embedder_model_name.write().unwrap() = String::new();
            return Json(serde_json::json!({
                "status": "ok",
                "provider": "none",
                "enabled": false,
            }))
            .into_response();
        }
        Some("hash") => {
            let hash_emb = crate::ingestion::embed::HashEmbedding::new(64);
            *state.embedder.write().unwrap() = Some(std::sync::Arc::new(hash_emb));
            *state.embedder_model_name.write().unwrap() = "hash".to_string();
            return Json(serde_json::json!({
                "status": "ok",
                "provider": "hash",
                "dimension": 64,
                "enabled": true,
            }))
            .into_response();
        }
        Some(name) => {
            let model_dir = format!("models/embeddings/{name}");
            let model_path = std::path::Path::new(&model_dir);

            if !model_path.is_dir() {
                return error_response(TensaError::Internal(format!(
                    "Model directory not found: {model_dir}"
                )))
                .into_response();
            }

            #[cfg(feature = "embedding")]
            {
                match crate::ingestion::onnx_embedder::OnnxEmbedder::from_directory(&model_dir) {
                    Ok(embedder) => {
                        let dim = embedder.dimension();
                        let provider = embedder.provider_name().to_string();
                        *state.embedder.write().unwrap() = Some(std::sync::Arc::new(embedder));
                        *state.embedder_model_name.write().unwrap() = name.to_string();
                        tracing::info!(
                            "Switched embedding model to: {name} ({provider}, dim={dim})"
                        );
                        return Json(serde_json::json!({
                            "status": "ok",
                            "provider": provider,
                            "dimension": dim,
                            "model": name,
                            "enabled": true,
                        }))
                        .into_response();
                    }
                    Err(e) => {
                        return error_response(TensaError::Internal(format!(
                            "Failed to load model '{name}': {e}"
                        )))
                        .into_response();
                    }
                }
            }

            #[cfg(not(feature = "embedding"))]
            {
                let _ = model_path;
                return error_response(TensaError::Internal(
                    "ONNX embedding support not compiled in. Build with --features embedding"
                        .into(),
                ))
                .into_response();
            }
        }
    }
}

/// POST /settings/embedding/download — download an embedding model from HuggingFace.
pub async fn download_embedding_model(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let repo_id = match body.get("repo_id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            return error_response(TensaError::Internal(
                "Missing 'repo_id' field (e.g. 'sentence-transformers/all-MiniLM-L6-v2')".into(),
            ))
            .into_response();
        }
    };

    // Derive model name from repo_id (e.g. "sentence-transformers/all-MiniLM-L6-v2" → "all-MiniLM-L6-v2")
    let model_name = body
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| repo_id.split('/').last().unwrap_or(&repo_id).to_string());

    let models_dir = std::path::PathBuf::from("models/embeddings");
    let target_dir = models_dir.join(&model_name);

    // Create directories
    if let Err(e) = std::fs::create_dir_all(&target_dir) {
        return error_response(TensaError::Internal(format!(
            "Failed to create model directory: {e}"
        )))
        .into_response();
    }

    let model_path = target_dir.join("model.onnx");
    let tokenizer_path = target_dir.join("tokenizer.json");

    // Download files from HuggingFace
    let base_url = format!("https://huggingface.co/{repo_id}/resolve/main");
    let client = reqwest::Client::new();

    let files_to_download = vec![
        ("onnx/model.onnx", model_path.clone()),
        ("tokenizer.json", tokenizer_path.clone()),
    ];

    for (remote_path, local_path) in &files_to_download {
        if local_path.exists() {
            tracing::info!("Skipping {} (already exists)", local_path.display());
            continue;
        }
        let url = format!("{base_url}/{remote_path}");
        tracing::info!("Downloading {} → {}", url, local_path.display());

        match client.get(&url).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    // Try alternative path for model.onnx (some repos put it at root)
                    if *remote_path == "onnx/model.onnx" {
                        let alt_url = format!("{base_url}/model.onnx");
                        match client.get(&alt_url).send().await {
                            Ok(alt_resp) if alt_resp.status().is_success() => {
                                let bytes = alt_resp.bytes().await.unwrap_or_default();
                                if let Err(e) = std::fs::write(local_path, &bytes) {
                                    return error_response(TensaError::Internal(format!(
                                        "Failed to write {}: {e}",
                                        local_path.display()
                                    )))
                                    .into_response();
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }
                    return error_response(TensaError::Internal(format!(
                        "Failed to download {remote_path}: HTTP {}",
                        resp.status()
                    )))
                    .into_response();
                }
                let bytes = resp.bytes().await.unwrap_or_default();
                if let Err(e) = std::fs::write(local_path, &bytes) {
                    return error_response(TensaError::Internal(format!(
                        "Failed to write {}: {e}",
                        local_path.display()
                    )))
                    .into_response();
                }
            }
            Err(e) => {
                return error_response(TensaError::Internal(format!(
                    "Download error for {remote_path}: {e}"
                )))
                .into_response();
            }
        }
    }

    let ready = model_path.exists() && tokenizer_path.exists();
    Json(serde_json::json!({
        "status": "ok",
        "model": model_name,
        "path": target_dir.to_string_lossy(),
        "ready": ready,
    }))
    .into_response()
}

// ─── Inference Engine Config ─────────────────────────────────

/// Combined configuration for all inference engines, exposed via the settings API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Causal engine (NOTEARS + do-calculus) parameters.
    pub causal: CausalConfigDto,
    /// Game-theoretic engine (QRE) parameters.
    pub game: GameConfigDto,
    /// Motivation engine (MaxEnt IRL) parameters.
    pub motivation: MotivationConfigDto,
}

/// Causal engine settings DTO (mirrors `CausalConfig` fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalConfigDto {
    pub decomposition_threshold: usize,
    pub learning_rate: f64,
    pub convergence_threshold: f64,
    pub max_iterations: usize,
    pub min_edge_weight: f64,
    pub beam_width: usize,
    pub beam_depth: usize,
    pub prune_threshold: f64,
}

/// Game engine settings DTO (mirrors `GameConfig` fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfigDto {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub initial_lambda: f64,
    pub max_lambda: f64,
    pub lambda_step: f64,
    pub sub_game_threshold: usize,
}

/// Motivation engine settings DTO (mirrors `MotivationConfig` fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationConfigDto {
    pub learning_rate: f64,
    pub convergence_threshold: f64,
    pub max_iterations: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            causal: CausalConfigDto {
                decomposition_threshold: 50,
                learning_rate: 0.01,
                convergence_threshold: 1e-6,
                max_iterations: 500,
                min_edge_weight: 0.1,
                beam_width: 5,
                beam_depth: 20,
                prune_threshold: 0.05,
            },
            game: GameConfigDto {
                max_iterations: 1000,
                convergence_threshold: 1e-6,
                initial_lambda: 1.0,
                max_lambda: 10.0,
                lambda_step: 0.5,
                sub_game_threshold: 4,
            },
            motivation: MotivationConfigDto {
                learning_rate: 0.01,
                convergence_threshold: 1e-6,
                max_iterations: 200,
            },
        }
    }
}

/// Load persisted inference config from KV store (if any).
pub fn load_persisted_inference_config(
    store: &dyn crate::store::KVStore,
) -> Option<InferenceConfig> {
    store
        .get(CFG_INFERENCE_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// GET /settings/inference — Return the current inference engine config.
pub async fn get_inference_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state
        .inference_config
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    match config {
        Ok(c) => Json(serde_json::json!({ "inference": *c })).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/inference — Update inference engine config at runtime.
pub async fn set_inference_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<InferenceConfig>,
) -> impl IntoResponse {
    // Basic validation
    if body.causal.max_iterations == 0 || body.causal.max_iterations > 10000 {
        return error_response(TensaError::Internal(
            "causal.max_iterations must be between 1 and 10000".into(),
        ))
        .into_response();
    }
    if body.game.max_iterations == 0 || body.game.max_iterations > 10000 {
        return error_response(TensaError::Internal(
            "game.max_iterations must be between 1 and 10000".into(),
        ))
        .into_response();
    }
    if body.motivation.max_iterations == 0 || body.motivation.max_iterations > 10000 {
        return error_response(TensaError::Internal(
            "motivation.max_iterations must be between 1 and 10000".into(),
        ))
        .into_response();
    }

    // Persist to KV store
    persist_config(
        state.hypergraph.store(),
        CFG_INFERENCE_KEY,
        &body,
        "inference",
    );

    // Update in-memory config
    {
        let mut cfg = match state.inference_config.write() {
            Ok(c) => c,
            Err(_) => {
                return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
            }
        };
        *cfg = body.clone();
    }

    tracing::info!("Inference config updated (persisted)");
    Json(serde_json::json!({ "inference": body, "status": "ok" })).into_response()
}

// ─── RAG Config ─────────────────────────────────────────────

/// Load persisted RAG config from KV store (if any).
pub fn load_persisted_rag_config(
    store: &dyn crate::store::KVStore,
) -> Option<crate::query::rag_config::RagConfig> {
    store
        .get(CFG_RAG_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// GET /settings/rag — Return the current RAG config.
pub async fn get_rag_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state
        .rag_config
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    match config {
        Ok(c) => Json(serde_json::json!({ "rag": *c })).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/rag — Update RAG config at runtime.
pub async fn set_rag_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<crate::query::rag_config::RagConfig>,
) -> impl IntoResponse {
    // Basic validation
    if body.budget.total_tokens == 0 || body.budget.total_tokens > 200_000 {
        return error_response(TensaError::Internal(
            "total_tokens must be between 1 and 200000".into(),
        ))
        .into_response();
    }

    // Persist to KV store
    persist_config(state.hypergraph.store(), CFG_RAG_KEY, &body, "rag");

    // Update in-memory config
    {
        let mut cfg = match state.rag_config.write() {
            Ok(c) => c,
            Err(_) => {
                return error_response(TensaError::Internal("Lock poisoned".into())).into_response()
            }
        };
        *cfg = body.clone();
    }

    tracing::info!("RAG config updated (persisted)");
    Json(serde_json::json!({ "rag": body, "status": "ok" })).into_response()
}

// ─── Inference/RAG LLM Settings ────────────────────────────────

/// Load persisted inference LLM config from KV store (if any).
pub fn load_persisted_inference_llm_config(
    store: &dyn crate::store::KVStore,
) -> Option<crate::api::server::LlmConfig> {
    store
        .get(CFG_INFERENCE_LLM_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// GET /settings/inference-llm — Get the dedicated inference/RAG LLM config.
pub async fn get_inference_llm(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state
        .inference_llm_config
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    match config {
        Ok(c) => Json(serde_json::json!({ "llm": c.redacted_hint() })).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/inference-llm — Update the dedicated inference/RAG LLM provider.
pub async fn set_inference_llm(
    State(state): State<Arc<AppState>>,
    Json(body): Json<crate::api::server::LlmConfig>,
) -> impl IntoResponse {
    // Merge keys: if incoming key is redacted/empty, preserve existing real key
    let merged = {
        let current = state.inference_llm_config.read().unwrap();
        body.merge_keys(&current)
    };

    // Validate: try to build extractor (unless None)
    if !matches!(merged, crate::api::server::LlmConfig::None) {
        if merged.build_extractor().is_none() {
            return error_response(TensaError::Internal(
                "Failed to build extractor from inference LLM config".into(),
            ))
            .into_response();
        }
    }

    let new_extractor = merged.build_extractor();
    let redacted = merged.redacted_hint();

    // Swap extractor
    {
        let mut ext = state.inference_extractor.write().unwrap();
        *ext = new_extractor;
    }
    // Swap config
    {
        let mut cfg = state.inference_llm_config.write().unwrap();
        *cfg = merged.clone();
    }
    // Persist to KV store
    persist_config(
        state.hypergraph.store(),
        CFG_INFERENCE_LLM_KEY,
        &merged,
        "inference LLM",
    );

    tracing::info!("Inference LLM updated (persisted): {:?}", redacted);
    Json(serde_json::json!({ "llm": redacted, "status": "ok" })).into_response()
}

// ─── Studio Chat LLM (v0.61, feature: studio-chat) ─────────────

/// GET /settings/chat-llm — read the chat-only LLM config with keys hinted.
#[cfg(feature = "studio-chat")]
pub async fn get_chat_llm(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = state
        .chat_llm_config
        .read()
        .map_err(|_| TensaError::Internal("Lock poisoned".into()));
    match config {
        Ok(c) => Json(serde_json::json!({ "llm": c.redacted_hint() })).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/chat-llm — swap the Studio chat LLM provider independently
/// from the ingestion / inference pipelines.
#[cfg(feature = "studio-chat")]
pub async fn set_chat_llm(
    State(state): State<Arc<AppState>>,
    Json(body): Json<crate::api::server::LlmConfig>,
) -> impl IntoResponse {
    let merged = {
        let current = state.chat_llm_config.read().unwrap();
        body.merge_keys(&current)
    };

    if !matches!(merged, crate::api::server::LlmConfig::None) && merged.build_extractor().is_none()
    {
        return error_response(TensaError::Internal(
            "Failed to build extractor from chat LLM config".into(),
        ))
        .into_response();
    }

    let new_extractor = merged.build_extractor();
    let redacted = merged.redacted_hint();

    {
        let mut ext = state.chat_extractor.write().unwrap();
        *ext = new_extractor;
    }
    {
        let mut cfg = state.chat_llm_config.write().unwrap();
        *cfg = merged.clone();
    }
    persist_config(
        state.hypergraph.store(),
        crate::studio_chat::CFG_CHAT_LLM_KEY,
        &merged,
        "chat LLM",
    );

    tracing::info!("Chat LLM updated (persisted): {:?}", redacted);
    Json(serde_json::json!({ "llm": redacted, "status": "ok" })).into_response()
}
