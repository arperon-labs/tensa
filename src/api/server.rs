use std::sync::{Arc, RwLock};

use axum::routing::{delete, get, post, put};
use axum::Router;
use serde::{Deserialize, Serialize};

use crate::api::{
    adaptation_routes, analysis, analytics_readback_routes, annotation_routes,
    architecture_routes, argumentation_gradual, bulk_routes, chunk_routes, collection_routes,
    compile_routes, continuity_routes, cost_ledger_routes, debug_routes, editing_routes, fuzzy,
    generation_routes, import_routes, inference, openai_compat, openapi, plan_routes,
    project_routes, research_routes, revision_routes, routes, settings_routes, source_routes,
    storywriting_routes, style_routes, synth, template_routes, temporal_ordhorn,
    workshop_routes, workspace_routes,
};
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::ingestion::config::IngestionConfig;
use crate::ingestion::embed::EmbeddingProvider;
use crate::ingestion::jobs::{IngestionJobQueue, IngestionProgress};
use crate::ingestion::llm::NarrativeExtractor;
use crate::ingestion::queue::ValidationQueue;
use crate::ingestion::vector::VectorIndex;
use crate::temporal::index::IntervalTree;

/// Which LLM provider is active.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "provider")]
pub enum LlmConfig {
    /// Anthropic direct API.
    #[serde(rename = "anthropic")]
    Anthropic { api_key: String, model: String },
    /// OpenRouter (OpenAI-compatible, cloud).
    #[serde(rename = "openrouter")]
    OpenRouter { api_key: String, model: String },
    /// Local / self-hosted endpoint (vLLM, Ollama, LiteLLM, etc.).
    #[serde(rename = "local")]
    Local {
        base_url: String,
        model: String,
        api_key: Option<String>,
    },
    /// Google Gemini API.
    #[serde(rename = "gemini")]
    Gemini { api_key: String, model: String },
    /// AWS Bedrock (Claude models).
    #[serde(rename = "bedrock")]
    Bedrock {
        region: String,
        model_id: String,
        access_key_id: String,
        secret_access_key: String,
    },
    /// No LLM configured.
    #[serde(rename = "none")]
    None,
}

impl LlmConfig {
    /// Build a `NarrativeExtractor` from this config.
    pub fn build_extractor(&self) -> Option<Arc<dyn NarrativeExtractor>> {
        match self {
            LlmConfig::Anthropic { api_key, model } => Some(Arc::new(
                crate::ingestion::llm::ClaudeClient::new(api_key.clone(), model.clone()),
            )),
            LlmConfig::OpenRouter { api_key, model } => Some(Arc::new(
                crate::ingestion::llm::OpenRouterClient::new(api_key.clone(), model.clone()),
            )),
            LlmConfig::Local {
                base_url,
                model,
                api_key,
            } => {
                let mut client =
                    crate::ingestion::llm::LocalLLMClient::new(base_url.clone(), model.clone());
                if let Some(key) = api_key {
                    client = client.with_api_key(key.clone());
                }
                Some(Arc::new(client))
            }
            #[cfg(feature = "gemini")]
            LlmConfig::Gemini { api_key, model } => Some(Arc::new(
                crate::ingestion::gemini::GeminiClient::new(api_key.clone(), model.clone()),
            )),
            #[cfg(not(feature = "gemini"))]
            LlmConfig::Gemini { .. } => {
                tracing::warn!("Gemini configured but `gemini` feature not enabled");
                None
            }
            #[cfg(feature = "bedrock")]
            LlmConfig::Bedrock {
                region,
                model_id,
                access_key_id,
                secret_access_key,
            } => Some(Arc::new(crate::ingestion::bedrock::BedrockClient::new(
                region.clone(),
                model_id.clone(),
                access_key_id.clone(),
                secret_access_key.clone(),
            ))),
            #[cfg(not(feature = "bedrock"))]
            LlmConfig::Bedrock { .. } => {
                tracing::warn!("Bedrock configured but `bedrock` feature not enabled");
                None
            }
            LlmConfig::None => None,
        }
    }

    /// Build a `NarrativeExtractor` with a custom system prompt override.
    /// Used for domain-specific extraction modes (temporal events, legal, etc.).
    pub fn build_extractor_with_prompt(&self, prompt: &str) -> Option<Arc<dyn NarrativeExtractor>> {
        match self {
            LlmConfig::Anthropic { api_key, model } => Some(Arc::new(
                crate::ingestion::llm::ClaudeClient::new(api_key.clone(), model.clone())
                    .with_system_prompt(prompt.to_string()),
            )),
            LlmConfig::OpenRouter { api_key, model } => Some(Arc::new(
                crate::ingestion::llm::OpenRouterClient::new(api_key.clone(), model.clone())
                    .with_system_prompt(prompt.to_string()),
            )),
            LlmConfig::Local {
                base_url,
                model,
                api_key,
            } => {
                let mut client =
                    crate::ingestion::llm::LocalLLMClient::new(base_url.clone(), model.clone())
                        .with_system_prompt(prompt.to_string());
                if let Some(key) = api_key {
                    client = client.with_api_key(key.clone());
                }
                Some(Arc::new(client))
            }
            #[cfg(feature = "gemini")]
            LlmConfig::Gemini { api_key, model } => Some(Arc::new(
                crate::ingestion::gemini::GeminiClient::new(api_key.clone(), model.clone()),
                // Gemini doesn't have with_system_prompt yet — uses default
            )),
            #[cfg(not(feature = "gemini"))]
            LlmConfig::Gemini { .. } => None,
            #[cfg(feature = "bedrock")]
            LlmConfig::Bedrock {
                region,
                model_id,
                access_key_id,
                secret_access_key,
            } => Some(Arc::new(crate::ingestion::bedrock::BedrockClient::new(
                region.clone(),
                model_id.clone(),
                access_key_id.clone(),
                secret_access_key.clone(),
            ))),
            #[cfg(not(feature = "bedrock"))]
            LlmConfig::Bedrock { .. } => None,
            LlmConfig::None => None,
        }
    }

    /// Return a redacted copy (API keys masked) for safe serialization to the frontend.
    pub fn redacted(&self) -> Self {
        match self {
            LlmConfig::Anthropic { model, .. } => LlmConfig::Anthropic {
                api_key: "***".into(),
                model: model.clone(),
            },
            LlmConfig::OpenRouter { model, .. } => LlmConfig::OpenRouter {
                api_key: "***".into(),
                model: model.clone(),
            },
            LlmConfig::Local {
                base_url,
                model,
                api_key,
            } => LlmConfig::Local {
                base_url: base_url.clone(),
                model: model.clone(),
                api_key: api_key.as_ref().map(|_| "***".into()),
            },
            LlmConfig::Gemini { model, .. } => LlmConfig::Gemini {
                api_key: "***".into(),
                model: model.clone(),
            },
            LlmConfig::Bedrock {
                region, model_id, ..
            } => LlmConfig::Bedrock {
                region: region.clone(),
                model_id: model_id.clone(),
                access_key_id: "***".into(),
                secret_access_key: "***".into(),
            },
            LlmConfig::None => LlmConfig::None,
        }
    }

    /// Return a copy with API keys replaced by a hint (first 4 chars + "...").
    /// Unlike `redacted()` which shows `"***"`, this lets the frontend display
    /// that a key exists without revealing it.
    pub fn redacted_hint(&self) -> Self {
        fn hint(key: &str) -> String {
            if key.is_empty() {
                return String::new();
            }
            let n = key.len().min(4);
            format!("{}...", &key[..n])
        }
        match self {
            LlmConfig::Anthropic { api_key, model } => LlmConfig::Anthropic {
                api_key: hint(api_key),
                model: model.clone(),
            },
            LlmConfig::OpenRouter { api_key, model } => LlmConfig::OpenRouter {
                api_key: hint(api_key),
                model: model.clone(),
            },
            LlmConfig::Local {
                base_url,
                model,
                api_key,
            } => LlmConfig::Local {
                base_url: base_url.clone(),
                model: model.clone(),
                api_key: api_key.as_ref().map(|k| hint(k)),
            },
            LlmConfig::Gemini { api_key, model } => LlmConfig::Gemini {
                api_key: hint(api_key),
                model: model.clone(),
            },
            LlmConfig::Bedrock {
                region,
                model_id,
                access_key_id,
                secret_access_key,
            } => LlmConfig::Bedrock {
                region: region.clone(),
                model_id: model_id.clone(),
                access_key_id: hint(access_key_id),
                secret_access_key: hint(secret_access_key),
            },
            LlmConfig::None => LlmConfig::None,
        }
    }

    /// Check if a key value is a redacted placeholder or empty (should not overwrite real key).
    pub fn is_redacted_or_empty(key: &str) -> bool {
        key.is_empty() || key == "***" || (key.ends_with("...") && key.len() <= 8)
    }

    /// Merge: if self's API key looks redacted/empty, keep the real key from `existing`.
    /// If the provider type changed, use self as-is (new provider needs a new key).
    pub fn merge_keys(&self, existing: &LlmConfig) -> Self {
        match (self, existing) {
            (
                LlmConfig::Anthropic { api_key, model },
                LlmConfig::Anthropic {
                    api_key: real_key, ..
                },
            ) => LlmConfig::Anthropic {
                api_key: if Self::is_redacted_or_empty(api_key) {
                    real_key.clone()
                } else {
                    api_key.clone()
                },
                model: model.clone(),
            },
            (
                LlmConfig::OpenRouter { api_key, model },
                LlmConfig::OpenRouter {
                    api_key: real_key, ..
                },
            ) => LlmConfig::OpenRouter {
                api_key: if Self::is_redacted_or_empty(api_key) {
                    real_key.clone()
                } else {
                    api_key.clone()
                },
                model: model.clone(),
            },
            (
                LlmConfig::Local {
                    base_url,
                    model,
                    api_key,
                },
                LlmConfig::Local {
                    api_key: real_key, ..
                },
            ) => LlmConfig::Local {
                base_url: base_url.clone(),
                model: model.clone(),
                api_key: match api_key {
                    Some(k) if Self::is_redacted_or_empty(k) => real_key.clone(),
                    other => other.clone(),
                },
            },
            (
                LlmConfig::Gemini { api_key, model },
                LlmConfig::Gemini {
                    api_key: real_key, ..
                },
            ) => LlmConfig::Gemini {
                api_key: if Self::is_redacted_or_empty(api_key) {
                    real_key.clone()
                } else {
                    api_key.clone()
                },
                model: model.clone(),
            },
            (
                LlmConfig::Bedrock {
                    region,
                    model_id,
                    access_key_id,
                    secret_access_key,
                },
                LlmConfig::Bedrock {
                    access_key_id: real_ak,
                    secret_access_key: real_sk,
                    ..
                },
            ) => LlmConfig::Bedrock {
                region: region.clone(),
                model_id: model_id.clone(),
                access_key_id: if Self::is_redacted_or_empty(access_key_id) {
                    real_ak.clone()
                } else {
                    access_key_id.clone()
                },
                secret_access_key: if Self::is_redacted_or_empty(secret_access_key) {
                    real_sk.clone()
                } else {
                    secret_access_key.clone()
                },
            },
            // Provider changed or no existing — use incoming as-is
            _ => self.clone(),
        }
    }
}

/// Event sent to WebSocket clients when job status changes.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatusEvent {
    pub job_id: String,
    pub status: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Shared application state.
pub struct AppState {
    pub hypergraph: Hypergraph,
    pub interval_tree: RwLock<IntervalTree>,
    pub validation_queue: ValidationQueue,
    pub job_queue: Option<Arc<JobQueue>>,
    pub extractor: RwLock<Option<Arc<dyn NarrativeExtractor>>>,
    pub llm_config: RwLock<LlmConfig>,
    pub embedder: RwLock<Option<Arc<dyn EmbeddingProvider>>>,
    /// Name of the active embedding model (e.g. "all-MiniLM-L6-v2", "hash", or empty).
    pub embedder_model_name: RwLock<String>,
    pub vector_index: Option<Arc<RwLock<VectorIndex>>>,
    pub ingestion_config: RwLock<IngestionConfig>,
    pub inference_config: RwLock<settings_routes::InferenceConfig>,
    /// Per-job status watchers for WebSocket push.
    pub job_watchers:
        RwLock<std::collections::HashMap<String, tokio::sync::watch::Sender<JobStatusEvent>>>,
    /// KV-backed ingestion job queue.
    pub ingestion_jobs: Arc<IngestionJobQueue>,
    /// Ephemeral progress map for running ingestion jobs (job_id → progress).
    pub ingestion_progress: std::sync::Mutex<
        std::collections::HashMap<String, Arc<std::sync::Mutex<IngestionProgress>>>,
    >,
    /// Cancellation flags for running ingestion jobs.
    pub ingestion_cancel_flags:
        std::sync::Mutex<std::collections::HashMap<String, Arc<std::sync::atomic::AtomicBool>>>,
    /// LLM response cache (Sprint RAG-1).
    pub llm_cache: Option<crate::ingestion::llm_cache::LlmCache>,
    /// Document status tracker (Sprint RAG-1).
    pub doc_tracker: Option<crate::ingestion::doc_status::DocStatusTracker>,
    /// Source-to-entity/situation index for cascade deletion (Sprint RAG-4).
    pub source_index: Option<crate::ingestion::deletion::SourceIndex>,
    /// RAG query configuration (Sprint RAG-2).
    pub rag_config: RwLock<crate::query::rag_config::RagConfig>,
    /// Optional reranker for RAG context reordering (Sprint RAG-6).
    pub reranker: Option<Arc<dyn crate::query::reranker::Reranker>>,
    /// Root KV store (not workspace-scoped) for workspace management metadata.
    pub root_store: Arc<dyn crate::store::KVStore>,
    /// Geocoder for place name → coordinate resolution (Nominatim + KV cache).
    pub geocoder: crate::ingestion::geocode::Geocoder,
    /// Dedicated LLM for inference/RAG queries (falls back to extractor if None).
    pub inference_extractor: RwLock<Option<Arc<dyn NarrativeExtractor>>>,
    /// Config for inference/RAG LLM (persisted at cfg/inference_llm).
    pub inference_llm_config: RwLock<LlmConfig>,
    /// Dedicated LLM for Studio chat (v0.61; studio-chat feature).
    /// When None, chat falls back to `inference_extractor`, then `extractor`.
    #[cfg(feature = "studio-chat")]
    pub chat_extractor: RwLock<Option<Arc<dyn NarrativeExtractor>>>,
    /// Config for the Studio chat LLM, persisted at `cfg/studio_chat_llm`.
    #[cfg(feature = "studio-chat")]
    pub chat_llm_config: RwLock<LlmConfig>,
    /// Bundled skill registry for the chat agent (Phase 2: studio-ui only).
    #[cfg(feature = "studio-chat")]
    pub chat_skills: crate::studio_chat::SkillRegistry,
    /// Pending-confirmation gate (Phase 3) — parks mutating tool calls
    /// until the user responds via `POST /studio/chat/sessions/:id/confirm`.
    #[cfg(feature = "studio-chat")]
    pub chat_confirm_gate: Arc<crate::studio_chat::ConfirmGate>,
    /// Third-party MCP server proxies (Phase 4). Each configured server is
    /// spawned lazily on first tool request and its tool list aggregated
    /// into the chat catalog.
    #[cfg(feature = "studio-chat")]
    pub chat_mcp_proxies: Arc<crate::studio_chat::McpProxySet>,
    /// Process-wide surrogate-model registry (EATH Phase 6). Constructed once
    /// here and shared by every `/synth/*` handler — handlers MUST NOT
    /// re-instantiate via `SurrogateRegistry::default()` because that would
    /// drift from any registry the engines were built against.
    pub synth_registry: Arc<crate::synth::SurrogateRegistry>,
}

/// Build the Axum router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    let router = Router::new()
        .route("/openapi.json", get(openapi::openapi_json))
        .route("/health", get(routes::health))
        .route("/reset", post(routes::reset_data))
        .route(
            "/entities",
            get(routes::list_entities).post(routes::create_entity),
        )
        .route("/entities/merge", post(routes::merge_entities))
        .route(
            "/entities/:id",
            get(routes::get_entity)
                .put(routes::update_entity)
                .delete(routes::delete_entity),
        )
        .route("/entities/:id/split", post(routes::split_entity))
        .route("/entities/:id/restore", post(routes::restore_entity))
        .route(
            "/situations",
            get(routes::list_situations).post(routes::create_situation),
        )
        .route(
            "/situations/:id",
            get(routes::get_situation)
                .put(routes::update_situation)
                .delete(routes::delete_situation),
        )
        .route("/situations/:id/restore", post(routes::restore_situation))
        .route(
            "/situations/:id/participants",
            get(routes::get_participants),
        )
        .route(
            "/entities/:id/situations",
            get(routes::get_entity_situations),
        )
        .route("/participations", post(routes::create_participation))
        // Storywriting: bulk + delete participations (v0.48.0)
        .route(
            "/participations/bulk",
            post(storywriting_routes::bulk_create_participations),
        )
        .route(
            "/participations/:entity_id/:situation_id",
            delete(storywriting_routes::remove_participation),
        )
        .route("/query", post(routes::execute_query))
        .route("/infer", post(routes::submit_inference))
        .route("/ask", post(routes::ask_question))
        // Prompt tuning endpoints (GraphRAG Sprint 1)
        .route("/prompts/tune", post(routes::tune_prompts))
        .route("/prompts", get(routes::list_prompts))
        .route("/prompts/:narrative_id", get(routes::get_prompt))
        .route("/prompts/:narrative_id", put(routes::update_prompt))
        .route("/prompts/:narrative_id", delete(routes::delete_prompt))
        // Validation queue endpoints (Phase 1)
        .route("/validation-queue", get(routes::list_validation_queue))
        .route("/validation-queue/:id", get(routes::get_queue_item))
        .route(
            "/validation-queue/:id/approve",
            post(routes::approve_queue_item),
        )
        .route(
            "/validation-queue/:id/reject",
            post(routes::reject_queue_item),
        )
        .route("/validation-queue/:id/edit", post(routes::edit_queue_item))
        // Inference job endpoints (Phase 2)
        .route("/jobs", post(routes::submit_job).get(routes::list_jobs))
        .route("/jobs/:id", get(routes::get_job).delete(routes::cancel_job))
        .route("/jobs/:id/result", get(routes::get_job_result))
        // Ingestion endpoint
        .route("/ingest", post(routes::ingest_text))
        .route("/ingest/jobs", get(routes::list_ingestion_jobs))
        .route(
            "/ingest/jobs/:id",
            get(routes::get_ingestion_job).delete(routes::cancel_ingestion_job),
        )
        .route(
            "/ingest/jobs/:id/rollback",
            post(routes::rollback_ingestion_job),
        )
        .route("/ingest/jobs/:id/retry", post(routes::retry_ingestion_job))
        .route("/ingest/jobs/:id/lineage", get(routes::get_job_lineage))
        .route(
            "/ingest/jobs/:id/chunks/batch",
            post(routes::batch_chunk_action),
        )
        // Ingestion templates
        .route(
            "/templates",
            get(routes::list_templates).post(routes::create_template),
        )
        .route(
            "/templates/:id",
            get(routes::get_template)
                .put(routes::update_template)
                .delete(routes::delete_template),
        )
        // LLM call logs & chunk extraction records
        .route("/ingest/jobs/:id/logs", get(routes::get_job_logs))
        .route(
            "/ingest/jobs/:id/chunks/:index/logs",
            get(routes::get_chunk_logs),
        )
        .route(
            "/ingest/jobs/:id/extractions",
            get(routes::get_job_extractions),
        )
        .route(
            "/ingest/jobs/:id/chunks/:index/extraction",
            get(routes::get_chunk_extraction),
        )
        .route(
            "/ingest/jobs/:id/chunks/:index/resend",
            post(routes::resend_chunk),
        )
        .route("/ingest/jobs/:id/reconcile", post(routes::reconcile_job))
        .route("/ingest/jobs/:id/reprocess", post(routes::reprocess_job))
        .route(
            "/ingest/jobs/:id/chunks/:index/reprocess",
            post(routes::reprocess_chunk),
        )
        // Narrative endpoints (Phase 3)
        .route(
            "/narratives",
            post(routes::create_narrative).get(routes::list_narratives),
        )
        .route(
            "/narratives/:id",
            get(routes::get_narrative)
                .put(routes::update_narrative)
                .delete(routes::delete_narrative),
        )
        .route("/narratives/:id/stats", get(routes::get_narrative_stats))
        .route(
            "/narratives/:id/anomalies",
            get(routes::get_narrative_anomalies),
        )
        .route("/narratives/:id/merge", post(routes::merge_narratives))
        .route(
            "/narratives/:id/reorder",
            post(routes::reorder_narrative_scenes),
        )
        .route(
            "/narratives/:id/causal-links/backfill-adjacent",
            post(routes::backfill_adjacent_causal_links_handler),
        )
        .route(
            "/narratives/:id/names/backfill-from-content",
            post(routes::backfill_names_from_content_handler),
        )
        .route(
            "/narratives/:id/dedup-entities",
            post(routes::dedup_entities),
        )
        .route("/narratives/:id/analyze", post(routes::analyze_narrative))
        .route(
            "/narratives/:id/batch-infer",
            post(routes::batch_infer_narrative),
        )
        .route("/narratives/:id/export", get(routes::export_narrative))
        // Narrative Debugger (Sprint D10.4)
        .route("/narratives/:id/diagnose", post(debug_routes::diagnose))
        .route(
            "/narratives/:id/diagnostics",
            get(debug_routes::get_diagnostics),
        )
        .route(
            "/narratives/:id/health-score",
            get(debug_routes::get_health_score),
        )
        .route(
            "/narratives/:id/diagnose-chapter/:n",
            post(debug_routes::diagnose_single_chapter),
        )
        .route(
            "/narratives/:id/suggest-fixes",
            post(debug_routes::suggest_narrative_fixes),
        )
        .route(
            "/narratives/:id/apply-fix",
            post(debug_routes::apply_narrative_fix),
        )
        .route(
            "/narratives/:id/auto-repair",
            post(debug_routes::auto_repair_narrative),
        )
        // Narrative Adaptation (Sprint D11)
        .route(
            "/narratives/:id/essentiality",
            post(adaptation_routes::essentiality),
        )
        .route(
            "/narratives/:id/compress",
            post(adaptation_routes::compress),
        )
        .route(
            "/narratives/:id/compress/preview",
            post(adaptation_routes::compress_preview),
        )
        .route("/narratives/:id/expand", post(adaptation_routes::expand))
        .route(
            "/narratives/:id/expand/preview",
            post(adaptation_routes::expand_preview),
        )
        .route(
            "/narratives/:id/expand/add-subplot",
            post(adaptation_routes::add_subplot),
        )
        .route("/narratives/diff", post(adaptation_routes::diff))
        // Narrative architecture (Sprint W14 / D9)
        .route(
            "/narratives/:id/commitments",
            get(architecture_routes::commitments_handler),
        )
        .route(
            "/narratives/:id/commitment-rhythm",
            get(architecture_routes::commitment_rhythm_handler),
        )
        .route(
            "/narratives/:id/fabula",
            get(architecture_routes::fabula_handler),
        )
        .route(
            "/narratives/:id/sjuzet",
            get(architecture_routes::sjuzet_handler),
        )
        .route(
            "/narratives/:id/sjuzet/reorderings",
            get(architecture_routes::sjuzet_reorderings_handler),
        )
        .route(
            "/narratives/:id/dramatic-irony",
            get(architecture_routes::dramatic_irony_handler),
        )
        .route(
            "/narratives/:id/focalization",
            get(architecture_routes::focalization_handler),
        )
        .route(
            "/narratives/:id/character-arc",
            get(architecture_routes::character_arc_handler),
        )
        .route(
            "/narratives/:id/subplots",
            get(architecture_routes::subplots_handler),
        )
        .route(
            "/narratives/:id/scene-sequel",
            get(architecture_routes::scene_sequel_handler),
        )
        // Community summary endpoints (Sprint RAG-5)
        .route(
            "/narratives/:id/communities/summarize",
            post(routes::summarize_communities),
        )
        .route("/narratives/:id/communities", get(routes::list_communities))
        .route(
            "/narratives/:id/communities/hierarchy",
            get(routes::get_community_hierarchy),
        )
        .route(
            "/narratives/:id/communities/:cid",
            get(routes::get_community),
        )
        // Cost ledger — AI operation cost records (Sprint W5, v0.49.4)
        .route(
            "/narratives/:id/cost-ledger",
            get(cost_ledger_routes::list_handler),
        )
        .route(
            "/narratives/:id/cost-ledger/summary",
            get(cost_ledger_routes::summary_handler),
        )
        // Continuity — pinned facts, continuity check, workspace summary (Sprint W4, v0.49.3)
        .route(
            "/narratives/:id/pinned-facts",
            get(continuity_routes::list_facts_handler).post(continuity_routes::create_fact_handler),
        )
        .route(
            "/narratives/:id/pinned-facts/:fact_id",
            put(continuity_routes::update_fact_handler)
                .delete(continuity_routes::delete_fact_handler),
        )
        .route(
            "/narratives/:id/continuity-check",
            post(continuity_routes::check_handler),
        )
        .route(
            "/narratives/:id/continuity/check",
            post(continuity_routes::check_prose_handler),
        )
        .route(
            "/narratives/:id/workspace",
            get(continuity_routes::workspace_handler),
        )
        // Workshop — tiered analysis (Sprint W3, v0.49.2)
        .route(
            "/narratives/:id/workshop/estimate",
            post(workshop_routes::estimate_handler),
        )
        .route(
            "/narratives/:id/workshop/run",
            post(workshop_routes::run_handler),
        )
        .route(
            "/narratives/:id/workshop/reports",
            get(workshop_routes::list_reports_handler),
        )
        .route(
            "/workshop/reports/:report_id",
            get(workshop_routes::get_report_handler),
        )
        // Edit engine — Sprint W2 (v0.49.1)
        .route("/situations/:id/edit", post(editing_routes::edit_handler))
        .route(
            "/situations/:id/edit/apply",
            post(editing_routes::apply_handler),
        )
        // Generation engine — Sprint W1 (v0.49.0)
        .route(
            "/narratives/:id/generate",
            post(generation_routes::generate_handler),
        )
        .route(
            "/narratives/:id/generate/apply",
            post(generation_routes::apply_handler),
        )
        .route(
            "/narratives/:id/generate/estimate",
            post(generation_routes::estimate_handler),
        )
        // Narrative plan — writer's canonical plot/style/length doc (v0.48.2)
        .route(
            "/narratives/:id/plan",
            get(plan_routes::get_handler)
                .post(plan_routes::upsert_handler)
                .put(plan_routes::patch_handler)
                .delete(plan_routes::delete_handler),
        )
        // Narrative revisions — version control (v0.48.1)
        .route(
            "/narratives/:id/revisions",
            get(revision_routes::list_handler).post(revision_routes::commit_handler),
        )
        .route(
            "/narratives/:id/revisions/head",
            get(revision_routes::head_handler),
        )
        .route("/revisions/:rev_id", get(revision_routes::get_handler))
        .route(
            "/revisions/:rev_id/restore",
            post(revision_routes::restore_handler),
        )
        .route(
            "/narratives/:id/diff-revisions",
            get(revision_routes::diff_handler),
        )
        // Storywriting user arc scaffolding (v0.48.0)
        .route(
            "/narratives/:id/arcs",
            get(storywriting_routes::list_arcs).post(storywriting_routes::create_arc),
        )
        .route(
            "/narratives/:id/arcs/:arc_id",
            put(storywriting_routes::update_arc).delete(storywriting_routes::delete_arc),
        )
        .route(
            "/narratives/:id/situations/reorder",
            post(storywriting_routes::reorder_situations),
        )
        // Taxonomy endpoints
        .route(
            "/taxonomy/:category",
            get(routes::list_taxonomy).post(routes::add_taxonomy),
        )
        .route(
            "/taxonomy/:category/:value",
            delete(routes::remove_taxonomy),
        )
        // Cross-narrative tag analysis
        .route("/analysis/by-tag", post(routes::analyze_by_tag))
        .route(
            "/analysis/shortest-path",
            post(routes::compute_shortest_path),
        )
        .route(
            "/analysis/narrative-diameter",
            post(routes::compute_narrative_diameter),
        )
        .route("/analysis/max-flow", post(routes::compute_max_flow))
        // Name-keyed path/flow wrappers (v0.73.0)
        .route(
            "/analysis/shortest-path-by-name",
            post(analytics_readback_routes::shortest_path_by_name),
        )
        .route(
            "/analysis/max-flow-by-name",
            post(analytics_readback_routes::max_flow_by_name),
        )
        // Narrative-scoped analytics read-back (v0.73.0)
        .route(
            "/narratives/:id/contentions",
            get(analytics_readback_routes::get_narrative_contentions),
        )
        .route(
            "/narratives/:id/contagion",
            get(analytics_readback_routes::get_contagion),
        )
        .route(
            "/narratives/:id/netinf",
            get(analytics_readback_routes::get_netinf),
        )
        .route(
            "/narratives/:id/temporal-motifs",
            get(analytics_readback_routes::get_temporal_motifs),
        )
        .route(
            "/narratives/:id/faction-evolution",
            get(analytics_readback_routes::get_faction_evolution),
        )
        .route(
            "/narratives/:id/temporal-rules",
            get(analytics_readback_routes::get_temporal_rules),
        )
        .route(
            "/narratives/:id/mean-field",
            get(analytics_readback_routes::get_mean_field),
        )
        .route(
            "/narratives/:id/psl",
            get(analytics_readback_routes::get_psl),
        )
        .route(
            "/narratives/:id/arguments",
            get(analytics_readback_routes::get_arguments),
        )
        .route(
            "/narratives/:id/evidence",
            get(analytics_readback_routes::get_evidence),
        )
        // Chunk endpoints
        .route("/chunks", get(chunk_routes::list_chunks))
        .route("/chunks/:id", get(chunk_routes::get_chunk))
        .route(
            "/narratives/:id/chunks",
            get(chunk_routes::list_narrative_chunks),
        )
        .route(
            "/ingest/jobs/:id/chunks",
            get(chunk_routes::list_job_chunks),
        )
        // Bulk operations (Sprint P3.6)
        .route("/entities/bulk", post(bulk_routes::bulk_create_entities))
        .route(
            "/situations/bulk",
            post(bulk_routes::bulk_create_situations),
        )
        // Structured import (Sprint P3.6)
        .route("/import/json", post(import_routes::import_json))
        .route("/import/csv", post(import_routes::import_csv))
        .route("/import/stix", post(import_routes::import_stix))
        // Archive export/import
        .route("/export/archive", post(routes::export_archive))
        .route("/import/archive", post(routes::import_archive))
        // WebSocket job status (Sprint P3.9)
        .route("/ws/jobs/:id", get(routes::ws_job_status))
        // URL ingestion (Sprint P3.9)
        .route("/ingest/url", post(routes::ingest_url))
        // Settings endpoints
        .route(
            "/settings/llm",
            get(routes::get_llm_settings).put(routes::set_llm_settings),
        )
        .route("/settings/models", get(settings_routes::discover_models))
        .route(
            "/settings/ingestion",
            get(settings_routes::get_ingestion_config).put(settings_routes::set_ingestion_config),
        )
        .route("/settings/presets", get(settings_routes::list_presets))
        .route(
            "/settings/embedding",
            get(settings_routes::get_embedding_info).put(settings_routes::set_embedding_model),
        )
        .route(
            "/settings/embedding/models",
            get(settings_routes::list_embedding_models),
        )
        .route(
            "/settings/embedding/download",
            post(settings_routes::download_embedding_model),
        )
        .route(
            "/settings/vector-store",
            get(routes::get_vector_store_config),
        )
        .route(
            "/settings/inference",
            get(settings_routes::get_inference_config).put(settings_routes::set_inference_config),
        )
        .route(
            "/settings/rag",
            get(settings_routes::get_rag_config).put(settings_routes::set_rag_config),
        )
        .route(
            "/settings/inference-llm",
            get(settings_routes::get_inference_llm).put(settings_routes::set_inference_llm),
        )
        // Project endpoints
        .route(
            "/projects",
            post(project_routes::create_project).get(project_routes::list_projects),
        )
        .route(
            "/projects/:id",
            get(project_routes::get_project)
                .put(project_routes::update_project)
                .delete(project_routes::delete_project),
        )
        .route(
            "/projects/:id/narratives",
            get(project_routes::list_project_narratives),
        )
        // Source intelligence endpoints (Phase 4)
        .route(
            "/sources",
            post(source_routes::create_source).get(source_routes::list_sources),
        )
        .route(
            "/sources/:id",
            get(source_routes::get_source)
                .put(source_routes::update_source)
                .delete(source_routes::delete_source),
        )
        .route(
            "/sources/:source_id/attributions",
            post(source_routes::add_attribution).get(source_routes::list_attributions_for_source),
        )
        .route(
            "/sources/:source_id/attributions/:target_id",
            delete(source_routes::remove_attribution),
        )
        .route(
            "/entities/:id/attributions",
            get(source_routes::list_attributions_for_target),
        )
        .route(
            "/situations/:id/attributions",
            get(source_routes::list_attributions_for_target),
        )
        .route("/contentions", post(source_routes::add_contention))
        .route(
            "/contentions/resolve",
            post(source_routes::resolve_contention),
        )
        .route(
            "/situations/:id/contentions",
            get(source_routes::list_contentions),
        )
        // Sprint W9 — scene research context + research notes.
        .route(
            "/situations/:id/research-context",
            get(research_routes::get_scene_research_context),
        )
        .route(
            "/situations/:id/research-notes",
            get(research_routes::list_notes_for_situation)
                .post(research_routes::create_note_for_situation),
        )
        .route(
            "/situations/:id/research-notes/from-chunk",
            post(research_routes::promote_chunk),
        )
        .route(
            "/situations/:id/factcheck",
            post(research_routes::factcheck_scene),
        )
        // Sprint W11 — cited generation helpers.
        .route(
            "/situations/:id/generation-prompt",
            post(research_routes::build_generation_prompt),
        )
        .route(
            "/situations/:id/hallucination-guard",
            post(research_routes::hallucination_guard_route),
        )
        .route(
            "/generation/parse-citations",
            post(research_routes::parse_citations_route),
        )
        // Sprint W12 — structured annotations.
        .route(
            "/situations/:id/annotations",
            get(annotation_routes::list_for_situation)
                .post(annotation_routes::create_for_situation),
        )
        .route(
            "/narratives/:id/annotations",
            get(annotation_routes::list_for_narrative),
        )
        .route(
            "/situations/:id/annotations/reconcile",
            post(annotation_routes::reconcile),
        )
        .route(
            "/annotations/:id",
            get(annotation_routes::get_one)
                .put(annotation_routes::update_one)
                .delete(annotation_routes::delete_one),
        )
        // Sprint W13 — compile profiles + compile endpoint.
        .route(
            "/narratives/:id/compile-profiles",
            get(compile_routes::list_profiles).post(compile_routes::create_profile_route),
        )
        .route(
            "/narratives/:id/compile",
            post(compile_routes::compile_route),
        )
        .route(
            "/compile-profiles/:id",
            get(compile_routes::get_profile_route)
                .put(compile_routes::update_profile_route)
                .delete(compile_routes::delete_profile_route),
        )
        // Sprint W15 — collections (saved searches).
        .route(
            "/narratives/:id/collections",
            get(collection_routes::list_route).post(collection_routes::create_route),
        )
        .route(
            "/collections/:id",
            get(collection_routes::get_route)
                .put(collection_routes::update_route)
                .delete(collection_routes::delete_route),
        )
        .route(
            "/collections/:id/resolve",
            get(collection_routes::resolve_route),
        )
        // Sprint W15 — narrative templates (builtin scaffolds + stored).
        .route("/narrative-templates", get(template_routes::list_handler))
        .route(
            "/narrative-templates/:id/instantiate",
            post(template_routes::instantiate_handler),
        )
        .route(
            "/narratives/:id/research-notes",
            get(research_routes::list_notes_in_narrative),
        )
        .route(
            "/research-notes/:id",
            get(research_routes::get_note)
                .put(research_routes::update_note)
                .delete(research_routes::delete_note),
        )
        .route(
            "/entities/:id/recompute-confidence",
            post(source_routes::recompute_entity_confidence),
        )
        .route(
            "/situations/:id/recompute-confidence",
            post(source_routes::recompute_situation_confidence),
        )
        // Bulk UUID → label resolver. Used by InferenceLab + similar views
        // that display lists of UUIDs and need to label each one without
        // generating a 404 per row for synth/sentinel ids.
        .route("/resolve/labels", post(routes::resolve_labels))
        // Cache & document status endpoints (Sprint RAG-1)
        .route("/cache/stats", get(routes::cache_stats))
        .route("/cache/clear", post(routes::clear_cache))
        .route("/ingest/status", get(routes::list_ingest_status))
        .route("/ingest/status/:hash", get(routes::get_ingest_status))
        // Source cascade deletion (Sprint RAG-4)
        .route(
            "/ingest/source/:source_id",
            delete(routes::delete_ingest_source),
        )
        // Style / Fingerprint endpoints
        .route(
            "/narratives/:id/style",
            post(style_routes::compute_narrative_style).get(style_routes::get_narrative_style),
        )
        .route(
            "/narratives/:id/fingerprint",
            get(style_routes::get_narrative_fingerprint),
        )
        .route("/style/compare", post(style_routes::compare_styles))
        .route(
            "/narratives/:id/style/anomalies",
            get(style_routes::get_style_anomalies),
        )
        .route(
            "/narratives/:id/style/radar",
            get(style_routes::get_style_radar),
        )
        // PAN@CLEF authorship verification (v0.28)
        .route("/style/verify", post(style_routes::verify_authorship))
        .route(
            "/style/pan/evaluate",
            post(style_routes::pan_evaluate_handler),
        )
        .route(
            "/settings/style-weights",
            get(style_routes::get_style_weights).put(style_routes::put_style_weights),
        )
        // Geocoding endpoints
        .route("/geocode", post(routes::geocode_place))
        .route("/geocode/backfill", post(routes::geocode_backfill))
        // Embedding backfill
        .route("/embeddings/backfill", post(routes::embedding_backfill))
        // OpenAI-compatible endpoints (Sprint RAG-6)
        .route(
            "/v1/chat/completions",
            post(openai_compat::chat_completions),
        )
        .route("/v1/models", get(openai_compat::list_models))
        // Workspace isolation (Sprint 7)
        .route(
            "/workspaces",
            post(workspace_routes::create_workspace).get(workspace_routes::list_workspaces),
        )
        .route(
            "/workspaces/:id",
            get(workspace_routes::get_workspace).delete(workspace_routes::delete_workspace),
        )
        // Synthetic surrogate generation (EATH Phase 6 + Phase 7 significance).
        // Routes split into per-submodule files under src/api/synth/ in Phase 7;
        // paths and verbs unchanged to preserve the frozen API contract.
        .route(
            "/synth/calibrate/:narrative_id",
            post(synth::calibration::calibrate),
        )
        .route(
            "/synth/params/:narrative_id/:model",
            get(synth::calibration::get_params)
                .put(synth::calibration::put_params)
                .delete(synth::calibration::delete_params_route),
        )
        .route("/synth/generate", post(synth::generation::generate))
        // EATH Phase 9 — hybrid (mixture-distribution) generation.
        .route(
            "/synth/generate-hybrid",
            post(synth::generation::generate_hybrid),
        )
        .route(
            "/synth/runs/:narrative_id",
            get(synth::generation::list_runs),
        )
        .route(
            "/synth/runs/:narrative_id/:run_id",
            get(synth::generation::get_run),
        )
        .route("/synth/seed/:run_id", get(synth::generation::get_seed))
        .route(
            "/synth/fidelity/:narrative_id/:run_id",
            get(synth::fidelity::get_fidelity),
        )
        .route(
            "/synth/fidelity-thresholds/:narrative_id",
            get(synth::fidelity::get_fidelity_thresholds)
                .put(synth::fidelity::put_fidelity_thresholds),
        )
        .route("/synth/models", get(synth::models::list_models))
        // Phase 7 — significance endpoints.
        .route(
            "/synth/significance",
            post(synth::significance::post_significance),
        )
        .route(
            "/synth/significance/:narrative_id/:metric/:run_id",
            get(synth::significance::get_significance_result),
        )
        .route(
            "/synth/significance/:narrative_id/:metric",
            get(synth::significance::list_significance_results),
        )
        // Phase 7b — higher-order contagion significance endpoints. Reuses the
        // K-loop infrastructure from Phase 7 (run_significance_pipeline) but
        // dispatches via a dedicated engine + URL pair so the contagion
        // request/response shape stays separate from the metric-string-based
        // /synth/significance triple.
        .route(
            "/synth/contagion-significance",
            post(synth::contagion::post_contagion_significance),
        )
        .route(
            "/synth/contagion-significance/:narrative_id/:run_id",
            get(synth::contagion::get_contagion_significance_result),
        )
        .route(
            "/synth/contagion-significance/:narrative_id",
            get(synth::contagion::list_contagion_significance_results),
        )
        // EATH Extension Phase 13c — dual-null-model significance. Compares
        // the source narrative against TWO independent null models (default
        // EATH + NuDHy) and returns per-model + combined verdicts.
        .route(
            "/synth/dual-significance",
            post(synth::dual_significance::post_dual_significance),
        )
        .route(
            "/synth/dual-significance/:narrative_id/:metric/:run_id",
            get(synth::dual_significance::get_dual_significance_result),
        )
        .route(
            "/synth/dual-significance/:narrative_id/:metric",
            get(synth::dual_significance::list_dual_significance_results),
        )
        // Phase 7b — synchronous higher-order SIR on the real narrative
        // (no job queue, no surrogates). For null-model significance,
        // see /synth/contagion-significance.
        .route(
            "/analysis/higher-order-contagion",
            post(routes::compute_higher_order_contagion),
        )
        // EATH Extension Phase 14 — synchronous bistability β-sweep on the
        // real narrative. Returns the hysteresis curve + classification.
        // For null-model significance vs K surrogates, see
        // /synth/bistability-significance.
        .route(
            "/analysis/contagion-bistability",
            post(routes::compute_contagion_bistability),
        )
        // Graded Acceptability Sprint Phase 3 — synchronous gradual /
        // ranking-based argumentation. Mirrors the
        // /analysis/higher-order-contagion sync precedent so the read-back
        // GET /narratives/:id/arguments stays cacheable + idempotent.
        // Cites: [amgoud2013ranking], [besnard2001hcategoriser],
        //        [amgoud2017weighted].
        .route(
            "/analysis/argumentation/gradual",
            post(argumentation_gradual::run_gradual),
        )
        // Graded Acceptability Sprint Phase 4 — synchronous van Beek
        // path-consistency closure on an Allen interval-algebra
        // network. Cites: [nebel1995ordhorn]. Sound for any network;
        // complete only for ORD-Horn (see module docs).
        .route(
            "/temporal/ordhorn/closure",
            post(temporal_ordhorn::closure_handler),
        )
        // EATH Extension Phase 14 — bistability significance against
        // surrogate models (default EATH + NuDHy).
        .route(
            "/synth/bistability-significance",
            post(synth::bistability_significance::post_bistability_significance),
        )
        .route(
            "/synth/bistability-significance/:narrative_id/:run_id",
            get(synth::bistability_significance::get_bistability_significance_result),
        )
        .route(
            "/synth/bistability-significance/:narrative_id",
            get(synth::bistability_significance::list_bistability_significance_results),
        )
        // EATH Extension Phase 16c — opinion-dynamics-significance against
        // surrogate models (default EATH + NuDHy). Engine returns a helpful
        // error when a requested model isn't registered (Phase 13b NuDHy
        // dependency); no panic.
        .route(
            "/synth/opinion-significance",
            post(synth::opinion_significance::post_opinion_significance),
        )
        .route(
            "/synth/opinion-significance/:narrative_id/:run_id",
            get(synth::opinion_significance::get_opinion_significance_result),
        )
        .route(
            "/synth/opinion-significance/:narrative_id",
            get(synth::opinion_significance::list_opinion_significance_results),
        )
        // EATH Extension Phase 15c — SINDy hypergraph reconstruction.
        // Submission queues a HypergraphReconstruction job; result polling
        // mirrors the standard /jobs/{id}/result envelope; opt-in
        // materialization commits inferred hyperedges as Situations under
        // ExtractionMethod::Reconstructed.
        .route(
            "/inference/hypergraph-reconstruction",
            post(inference::reconstruction::submit),
        )
        .route(
            "/inference/hypergraph-reconstruction/:job_id",
            get(inference::reconstruction::get_result),
        )
        .route(
            "/inference/hypergraph-reconstruction/:job_id/materialize",
            post(inference::reconstruction::materialize),
        )
        // EATH Extension Phase 16c — opinion dynamics on hypergraphs (BCM +
        // Deffuant). Synchronous (Phase 16b benchmarks: 100×10k = 21 ms),
        // mirroring `/analysis/contagion-bistability`'s sync pattern. Each
        // run persists at `opd/report/{narrative_id}/{run_id_v7}`.
        .route(
            "/analysis/opinion-dynamics",
            post(analysis::opinion_dynamics::run),
        )
        .route(
            "/analysis/opinion-dynamics/phase-transition-sweep",
            post(analysis::opinion_dynamics::sweep),
        )
        // Fuzzy Logic Sprint Phase 5 — graded Allen 13-vector for a pair
        // of situations. POST computes + caches; GET reads the cache (or
        // recomputes on miss). Cached at `fz/allen/{nid}/{a_id}/{b_id}`.
        .route(
            "/analysis/fuzzy-allen",
            post(analysis::fuzzy_allen::compute),
        )
        .route(
            "/analysis/fuzzy-allen/:nid/:a_id/:b_id",
            get(analysis::fuzzy_allen::get),
        )
        // Fuzzy Logic Sprint Phase 4 — REST API for fuzzy configuration
        // and one-shot aggregation. See `src/api/fuzzy/mod.rs` for the
        // full route surface + the `?tnorm=&aggregator=` query-string
        // opt-in contract threaded through confidence-returning
        // endpoints elsewhere in this router.
        .route("/fuzzy/tnorms", get(fuzzy::tnorm::list_tnorms))
        .route("/fuzzy/tnorms/:kind", get(fuzzy::tnorm::get_tnorm))
        .route(
            "/fuzzy/aggregators",
            get(fuzzy::aggregator::list_aggregators),
        )
        .route(
            "/fuzzy/aggregators/:kind",
            get(fuzzy::aggregator::get_aggregator),
        )
        .route(
            "/fuzzy/measures",
            post(fuzzy::measure::create_measure).get(fuzzy::measure::list_measures),
        )
        // Graded Acceptability Sprint Phase 2 — POST /fuzzy/measures/learn.
        // Fits a Choquet capacity from a ranking-supervised dataset via
        // pure-Rust PGD; persists with `MeasureProvenance::Learned`.
        // Cites: [grabisch1996choquet], [bustince2016choquet].
        .route(
            "/fuzzy/measures/learn",
            post(fuzzy::learn::learn_measure_handler),
        )
        // Graded Acceptability Sprint Phase 3 — versions endpoint MUST
        // be registered BEFORE the `/:name` catch-all so axum routes
        // `/fuzzy/measures/{name}/versions` here rather than treating
        // `versions` as a measure name.
        .route(
            "/fuzzy/measures/:name/versions",
            get(fuzzy::measure::list_versions),
        )
        .route(
            "/fuzzy/measures/:name",
            get(fuzzy::measure::get_measure).delete(fuzzy::measure::delete_measure),
        )
        .route(
            "/fuzzy/config",
            get(fuzzy::config::get_config).put(fuzzy::config::put_config),
        )
        .route("/fuzzy/aggregate", post(fuzzy::aggregate::aggregate))
        // Fuzzy Sprint Phase 6 — intermediate quantifier evaluation.
        // POST computes + caches; GET returns cached scalar or 404.
        // Cached at `fz/quant/{nid}/{predicate_hash}`.
        .route("/fuzzy/quantify", post(fuzzy::quantify::quantify))
        .route(
            "/fuzzy/quantify/:nid/:predicate_hash",
            get(fuzzy::quantify::get_quantify),
        )
        // Fuzzy Sprint Phase 7 — graded syllogism verification.
        // Cites: [murinovanovak2014peterson].
        .route(
            "/fuzzy/syllogism/verify",
            post(fuzzy::syllogism::verify_syllogism),
        )
        .route(
            "/fuzzy/syllogism/:nid/:proof_id",
            get(fuzzy::syllogism::get_syllogism_proof),
        )
        // Fuzzy Sprint Phase 8 — FCA concept lattice surface.
        // Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].
        .route(
            "/fuzzy/fca/lattice",
            post(fuzzy::fca::build_lattice_endpoint),
        )
        .route(
            "/fuzzy/fca/lattice/:lattice_id",
            get(fuzzy::fca::get_lattice_endpoint).delete(fuzzy::fca::delete_lattice_endpoint),
        )
        .route(
            "/fuzzy/fca/lattices/:nid",
            get(fuzzy::fca::list_lattices_endpoint),
        )
        // Fuzzy Sprint Phase 9 — Mamdani fuzzy rule systems.
        // Cites: [mamdani1975mamdani].
        .route(
            "/fuzzy/rules",
            post(fuzzy::rules::create_rule),
        )
        .route(
            "/fuzzy/rules/:nid",
            get(fuzzy::rules::list_rules_for_narrative),
        )
        .route(
            "/fuzzy/rules/:nid/:rule_id",
            get(fuzzy::rules::get_rule).delete(fuzzy::rules::delete_rule_endpoint),
        )
        .route(
            "/fuzzy/rules/:nid/evaluate",
            post(fuzzy::rules::evaluate_rules_endpoint),
        )
        // Fuzzy Sprint Phase 10 — hybrid fuzzy-probability surface
        // (Cao–Holčapek base case). Synchronous; persists at
        // `fz/hybrid/{nid}/{query_id_BE_16}`. Cites:
        // [flaminio2026fsta] [faginhalpern1994fuzzyprob].
        .route(
            "/fuzzy/hybrid/probability",
            post(fuzzy::hybrid::compute_probability),
        )
        .route(
            "/fuzzy/hybrid/probability/:nid",
            get(fuzzy::hybrid::list_probability),
        )
        .route(
            "/fuzzy/hybrid/probability/:nid/:query_id",
            get(fuzzy::hybrid::get_probability).delete(fuzzy::hybrid::delete_probability),
        );

    // Studio integrated agent chat (v0.60 Phase 1, v0.61 Phase 2: real LLM loop)
    #[cfg(feature = "studio-chat")]
    let router = {
        use crate::studio_chat::routes as chat_routes;
        router
            .route("/studio/chat", post(chat_routes::chat_turn))
            .route(
                "/studio/chat/sessions",
                post(chat_routes::create_session).get(chat_routes::list_sessions),
            )
            .route(
                "/studio/chat/sessions/:id",
                get(chat_routes::get_session)
                    .patch(chat_routes::patch_session)
                    .delete(chat_routes::delete_session),
            )
            .route(
                "/studio/chat/sessions/:id/confirm",
                post(chat_routes::confirm_tool_call),
            )
            .route(
                "/studio/chat/sessions/:id/stop",
                post(chat_routes::stop_turn),
            )
            .route("/studio/chat/skills", get(chat_routes::list_skills))
            .route(
                "/studio/chat/mcp-servers",
                get(chat_routes::list_mcp_servers).post(chat_routes::upsert_mcp_server),
            )
            .route(
                "/studio/chat/mcp-servers/:name",
                delete(chat_routes::delete_mcp_server),
            )
            .route(
                "/settings/chat-llm",
                get(settings_routes::get_chat_llm).put(settings_routes::set_chat_llm),
            )
    };

    // Style embeddings catalog (read-only list) — requires `generation` feature
    // because the underlying `style::embedding` module is gated behind it.
    #[cfg(feature = "generation")]
    let router = router.route("/style-embeddings", get(routes::list_style_embeddings));

    // Narrative architecture: plan / materialize / validate / chapter prep
    // (D9.6 + D9.7). Gated behind `generation` because the backend module is.
    #[cfg(feature = "generation")]
    let router = router
        .route(
            "/narratives/plan",
            post(architecture_routes::generate_plan_handler),
        )
        .route(
            "/plans/:plan_id/materialize",
            post(architecture_routes::materialize_plan_handler),
        )
        .route(
            "/narratives/:id/validate-materialized",
            get(architecture_routes::validate_materialized_handler),
        )
        .route(
            "/narratives/:id/generate-chapter",
            post(architecture_routes::generate_chapter_prep_handler),
        )
        .route(
            "/narratives/:id/generate-narrative",
            post(architecture_routes::generate_narrative_prep_handler),
        );

    // Document ingestion (Sprint P3.6, requires docparse feature)
    #[cfg(feature = "docparse")]
    let router = router.route("/ingest/document", post(routes::ingest_document));

    // RSS ingestion (requires web-ingest feature for feed parsing)
    #[cfg(feature = "web-ingest")]
    let router = router.route("/ingest/rss", post(routes::ingest_rss));

    // Disinfo extension (Sprint D1: dual fingerprint API + Sprint D2: spread +
    // Sprint D3: CIB + Sprint D7: monitor)
    #[cfg(feature = "disinfo")]
    let router = {
        use crate::api::{
            archetype_routes, cib_routes, claims_routes, disinfo_routes, monitor_routes,
            multilingual_routes, scheduler_routes, spread_routes,
        };
        router
            .route(
                "/entities/:id/behavioral-fingerprint",
                get(disinfo_routes::get_behavioral_fingerprint),
            )
            .route(
                "/entities/:id/behavioral-fingerprint/compute",
                post(disinfo_routes::compute_behavioral_fingerprint_route),
            )
            .route(
                "/narratives/:id/disinfo-fingerprint",
                get(disinfo_routes::get_disinfo_fingerprint),
            )
            .route(
                "/narratives/:id/disinfo-fingerprint/compute",
                post(disinfo_routes::compute_disinfo_fingerprint_route),
            )
            .route(
                "/fingerprints/compare",
                post(disinfo_routes::compare_fingerprints_route),
            )
            // Sprint D2: spread dynamics
            .route("/spread/r0", post(spread_routes::run_spread_r0))
            .route("/spread/r0/:id", get(spread_routes::get_spread_r0))
            .route(
                "/spread/velocity/:id",
                get(spread_routes::list_velocity_alerts),
            )
            .route(
                "/spread/jumps/:id",
                get(spread_routes::get_cross_platform_jumps),
            )
            .route(
                "/spread/intervention",
                post(spread_routes::run_intervention),
            )
            // Sprint D3: CIB detection + superspreaders
            .route("/analysis/cib", post(cib_routes::run_cib_detection))
            .route("/analysis/cib/:id", get(cib_routes::get_cib_detection))
            .route(
                "/analysis/superspreaders",
                post(cib_routes::run_superspreaders),
            )
            // Sprint D4: Claims & fact-check pipeline
            .route("/claims", post(claims_routes::create_claims))
            .route("/claims/:id", get(claims_routes::get_claim))
            .route("/claims/match", post(claims_routes::match_claim_route))
            .route("/fact-checks", post(claims_routes::create_fact_check))
            .route("/claims/:id/origin", get(claims_routes::trace_claim_origin))
            .route("/claims/:id/mutations", get(claims_routes::list_mutations))
            // Sprint D5: Archetypes + Disinfo Assessment
            .route(
                "/actors/:id/archetype",
                post(archetype_routes::classify_archetype).get(archetype_routes::get_archetype),
            )
            .route(
                "/analysis/disinfo-assess",
                post(archetype_routes::assess_disinfo),
            )
            .route(
                "/analysis/disinfo-assess/:id",
                get(archetype_routes::get_assessment),
            )
            // Sprint D6: Multilingual + MCP sources
            .route("/lang/detect", post(multilingual_routes::detect_language))
            .route(
                "/ingest/mcp-sources",
                get(multilingual_routes::list_mcp_sources),
            )
            // Sprint D7: Monitor subscriptions + alerts
            .route(
                "/monitor/subscriptions",
                post(monitor_routes::create_subscription).get(monitor_routes::list_subscriptions),
            )
            .route(
                "/monitor/subscriptions/:id",
                delete(monitor_routes::delete_subscription),
            )
            .route("/monitor/alerts", get(monitor_routes::list_alerts))
            // Sprint D8: Scheduler + Discovery + Sync + Reports + Health
            .route(
                "/scheduler/tasks",
                get(scheduler_routes::list_tasks).post(scheduler_routes::create_task),
            )
            .route(
                "/scheduler/tasks/:id",
                delete(scheduler_routes::delete_task),
            )
            .route(
                "/scheduler/tasks/:id/history",
                get(scheduler_routes::get_task_history),
            )
            .route(
                "/scheduler/tasks/:id/run-now",
                post(scheduler_routes::run_task_now),
            )
            .route(
                "/discovery/candidates",
                get(scheduler_routes::list_discovery_candidates),
            )
            .route(
                "/discovery/candidates/:id/approve",
                post(scheduler_routes::approve_discovery),
            )
            .route(
                "/discovery/candidates/:id/reject",
                post(scheduler_routes::reject_discovery),
            )
            .route(
                "/discovery/policy",
                get(scheduler_routes::get_discovery_policy)
                    .put(scheduler_routes::set_discovery_policy),
            )
            .route("/claims/sync", post(scheduler_routes::trigger_sync))
            .route("/claims/sync/history", get(scheduler_routes::sync_history))
            .route("/claims/sync/sources", get(scheduler_routes::sync_sources))
            .route(
                "/reports/situation",
                post(scheduler_routes::generate_report),
            )
            .route("/reports", get(scheduler_routes::list_reports))
            .route("/reports/:id", get(scheduler_routes::get_report))
            .route(
                "/ingest/mcp-sources/health",
                get(scheduler_routes::list_source_health),
            )
    };

    // Sprint D12: adversarial wargame REST surface.
    // Mirrors the existing MCP wargame tools — see src/api/wargame_routes.rs.
    #[cfg(feature = "adversarial")]
    let router = {
        use crate::api::wargame_routes;
        router
            .route(
                "/wargame/sessions",
                post(wargame_routes::create_wargame_session)
                    .get(wargame_routes::list_wargame_sessions),
            )
            .route(
                "/wargame/sessions/:id",
                delete(wargame_routes::delete_wargame_session),
            )
            .route(
                "/wargame/sessions/:id/state",
                get(wargame_routes::get_wargame_state),
            )
            .route(
                "/wargame/sessions/:id/auto-play",
                post(wargame_routes::auto_play_wargame),
            )
            .route(
                "/wargame/sessions/:id/moves",
                post(wargame_routes::submit_wargame_moves),
            )
    };

    router.with_state(state)
}

/// Start the API server on the given address.
pub async fn run(state: Arc<AppState>, addr: &str) -> anyhow::Result<()> {
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("TENSA API listening on {}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
