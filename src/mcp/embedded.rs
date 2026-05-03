//! Embedded backend — direct library access to the hypergraph.
//!
//! This backend creates its own Hypergraph, IntervalTree, and supporting
//! services. It does NOT depend on the `server` feature or AppState.

use std::sync::{Arc, RwLock};

use serde_json::Value;
use uuid::Uuid;

use std::collections::HashMap;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::inference::types::InferenceJob;
use crate::ingestion::embed::EmbeddingProvider;
use crate::ingestion::llm::NarrativeExtractor;
use crate::ingestion::queue::ValidationQueue;
use crate::ingestion::vector::VectorIndex;
use crate::narrative::registry::NarrativeRegistry;
use crate::query::{executor, parser, planner};
use crate::store::KVStore;
use crate::temporal::index::IntervalTree;
use crate::types::{InferenceJobType, JobPriority, JobStatus};

use super::backend::McpBackend;
use super::embedded_ext::{parse_uuid, to_json};

/// Embedded backend that operates directly on the TENSA library.
///
/// Wraps inner state in `Arc` so the struct is cheaply `Clone`-able
/// (required by rmcp's `ServerHandler`).
#[derive(Clone)]
pub struct EmbeddedBackend {
    inner: Arc<EmbeddedInner>,
}

struct EmbeddedInner {
    hypergraph: Hypergraph,
    interval_tree: RwLock<IntervalTree>,
    validation_queue: ValidationQueue,
    job_queue: Arc<JobQueue>,
    extractor: Option<Arc<dyn NarrativeExtractor>>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    vector_index: Option<Arc<RwLock<VectorIndex>>>,
}

impl EmbeddedBackend {
    /// Create from environment variables (production constructor).
    ///
    /// Reads:
    /// - `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` for LLM extraction
    /// - `TENSA_MODEL` for model selection
    pub fn from_env(store: Arc<dyn KVStore>) -> Self {
        let hypergraph = Hypergraph::new(store.clone());
        let interval_tree =
            IntervalTree::load(store.as_ref()).unwrap_or_else(|_| IntervalTree::new());
        let validation_queue = ValidationQueue::new(store.clone());
        let job_queue = Arc::new(JobQueue::new(store.clone()));

        let extractor: Option<Arc<dyn NarrativeExtractor>> =
            if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
                let model = std::env::var("TENSA_MODEL")
                    .unwrap_or_else(|_| "anthropic/claude-sonnet-4-20250514".to_string());
                tracing::info!("LLM extraction enabled via OpenRouter (model: {})", model);
                Some(Arc::new(crate::ingestion::llm::OpenRouterClient::new(
                    key, model,
                )))
            } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                let model = std::env::var("TENSA_MODEL")
                    .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
                tracing::info!("LLM extraction enabled via Anthropic (model: {})", model);
                Some(Arc::new(crate::ingestion::llm::ClaudeClient::new(
                    key, model,
                )))
            } else {
                tracing::warn!("No API key set — LLM ingestion disabled");
                None
            };

        let embedder: Option<Arc<dyn EmbeddingProvider>> =
            Some(Arc::new(crate::ingestion::embed::HashEmbedding::new(64)));
        let vector_index = Some(Arc::new(RwLock::new(VectorIndex::new(64))));

        Self {
            inner: Arc::new(EmbeddedInner {
                hypergraph,
                interval_tree: RwLock::new(interval_tree),
                validation_queue,
                job_queue,
                extractor,
                embedder,
                vector_index,
            }),
        }
    }

    /// Create from an existing store (test constructor).
    pub fn from_store(store: Arc<dyn KVStore>) -> Self {
        let hypergraph = Hypergraph::new(store.clone());
        let validation_queue = ValidationQueue::new(store.clone());
        let job_queue = Arc::new(JobQueue::new(store.clone()));

        Self {
            inner: Arc::new(EmbeddedInner {
                hypergraph,
                interval_tree: RwLock::new(IntervalTree::new()),
                validation_queue,
                job_queue,
                extractor: None,
                embedder: None,
                vector_index: None,
            }),
        }
    }

    pub(crate) fn store_arc(&self) -> Arc<dyn KVStore> {
        self.inner.hypergraph.store_arc()
    }

    /// Access the inner hypergraph.
    pub(crate) fn hypergraph(&self) -> &Hypergraph {
        &self.inner.hypergraph
    }

    /// Access the configured LLM extractor (if any).
    pub(crate) fn extractor_opt(&self) -> Option<Arc<dyn NarrativeExtractor>> {
        self.inner.extractor.clone()
    }

    /// Access the inner job queue (for synth tool submissions etc.).
    pub(crate) fn job_queue_arc(&self) -> Arc<JobQueue> {
        self.inner.job_queue.clone()
    }
}

impl McpBackend for EmbeddedBackend {
    async fn execute_query(&self, tensaql: &str) -> Result<Value> {
        let stmt = parser::parse_statement(tensaql)?;
        let plan = planner::plan_statement(&stmt)?;
        let tree = self
            .inner
            .interval_tree
            .read()
            .map_err(|_| TensaError::Internal("Lock poisoned".into()))?;
        let results = executor::execute(&plan, &self.inner.hypergraph, &*tree)?;
        to_json(results)
    }

    async fn submit_inference_query(&self, tensaql: &str) -> Result<Value> {
        let parsed = parser::parse_query(tensaql)?;
        let plan = planner::plan_query(&parsed)?;
        let tree = self
            .inner
            .interval_tree
            .read()
            .map_err(|_| TensaError::Internal("Lock poisoned".into()))?;
        let results = executor::execute(&plan, &self.inner.hypergraph, &*tree)?;

        // Try to extract a job descriptor from the executor results and submit
        // it to the inference job queue.
        if let Some(row) = results.first() {
            if let Ok(job_type) = infer_type_from_row(row) {
                let target_id = extract_target_id(row);
                let parameters = extract_parameters(row);

                let job = InferenceJob {
                    id: Uuid::now_v7().to_string(),
                    job_type,
                    target_id: target_id.unwrap_or_else(Uuid::now_v7),
                    parameters: parameters.unwrap_or(serde_json::json!({})),
                    priority: JobPriority::Normal,
                    status: JobStatus::Pending,
                    estimated_cost_ms: 0,
                    created_at: chrono::Utc::now(),
                    started_at: None,
                    completed_at: None,
                    error: None,
                };

                let job_id = job.id.clone();
                let returned_id = self.inner.job_queue.submit(job)?;

                return Ok(serde_json::json!({
                    "job_id": returned_id,
                    "status": "Pending",
                    "message": if returned_id == job_id {
                        "Job submitted to inference queue"
                    } else {
                        "Duplicate job — returning existing in-flight job"
                    }
                }));
            }
        }

        // Fallback: return raw results if no job descriptor found
        to_json(results)
    }

    async fn create_entity(&self, data: Value) -> Result<Value> {
        let entity: crate::types::Entity = super::embedded_ext::parse_as(data, "entity")?;
        let id = self.inner.hypergraph.create_entity(entity)?;
        Ok(serde_json::json!({"id": id}))
    }

    async fn get_entity(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        let entity = self.inner.hypergraph.get_entity(&uuid)?;
        to_json(entity)
    }

    async fn create_situation(&self, data: Value) -> Result<Value> {
        let situation: crate::types::Situation = super::embedded_ext::parse_as(data, "situation")?;
        let id = self.inner.hypergraph.create_situation(situation)?;
        Ok(serde_json::json!({"id": id}))
    }

    async fn get_situation(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        let situation = self.inner.hypergraph.get_situation(&uuid)?;
        to_json(situation)
    }

    async fn add_participant(&self, data: Value) -> Result<Value> {
        let participation: crate::types::Participation =
            super::embedded_ext::parse_as(data, "participation")?;
        self.inner.hypergraph.add_participant(participation)?;
        Ok(serde_json::json!({"status": "ok"}))
    }

    async fn ingest_text(
        &self,
        text: &str,
        narrative_id: &str,
        source: &str,
        auto_commit_threshold: Option<f32>,
        review_threshold: Option<f32>,
    ) -> Result<Value> {
        let extractor = self
            .inner
            .extractor
            .as_ref()
            .ok_or_else(|| {
                TensaError::LlmError(
                    "No LLM provider configured. Go to Settings to configure one.".into(),
                )
            })?
            .clone();

        let hg = Arc::new(Hypergraph::new(self.store_arc()));
        let queue = Arc::new(ValidationQueue::new(self.store_arc()));
        let embedder = self.inner.embedder.clone();
        let vector_index = self.inner.vector_index.clone();
        let narrative_id_owned = narrative_id.to_string();
        let source_owned = source.to_string();
        let text_owned = text.to_string();

        let config = crate::ingestion::pipeline::PipelineConfig {
            chunker: crate::ingestion::chunker::ChunkerConfig::default(),
            auto_commit_threshold: auto_commit_threshold.unwrap_or(0.8),
            review_threshold: review_threshold.unwrap_or(0.3),
            source_id: narrative_id_owned.clone(),
            source_type: "text".into(),
            narrative_id: Some(narrative_id_owned.clone()),
            job_id: None,
            concurrency: 1,
            strip_boilerplate: true,
            enrich: true,
            single_session: false,
            session_max_context_tokens: 0,
            debug: false,
            cascade_mode: crate::ingestion::pipeline::CascadeMode::default(),
            post_ingest_mamdani_rule_id: None,
        };

        let store = self.store_arc();
        let result = tokio::task::spawn_blocking(move || {
            let pipeline = crate::ingestion::pipeline::IngestionPipeline::new(
                hg.clone(),
                extractor,
                embedder,
                vector_index,
                queue,
                config,
            );
            let report = pipeline.ingest_text(&text_owned, &source_owned)?;

            let registry = NarrativeRegistry::new(store);
            let _ = registry.update(&narrative_id_owned, |n| {
                n.entity_count += report.entities_created;
                n.situation_count += report.situations_created;
            });

            Ok::<_, TensaError>(report)
        })
        .await
        .map_err(|e| TensaError::Internal(format!("Task panicked: {}", e)))?;

        let report = result?;
        to_json(report)
    }

    async fn list_narratives(&self) -> Result<Value> {
        let registry = NarrativeRegistry::new(self.store_arc());
        let narratives = registry.list(None, None)?;
        to_json(narratives)
    }

    async fn create_narrative(&self, data: Value) -> Result<Value> {
        let narrative: crate::narrative::types::Narrative =
            super::embedded_ext::parse_as(data, "narrative")?;
        let registry = NarrativeRegistry::new(self.store_arc());
        let id = registry.create(narrative)?;
        Ok(serde_json::json!({"id": id}))
    }

    async fn get_job_status(&self, job_id: &str) -> Result<Value> {
        let job = self.inner.job_queue.get_job(job_id)?;
        to_json(job)
    }

    async fn get_job_result(&self, job_id: &str) -> Result<Value> {
        let result = self.inner.job_queue.get_result(job_id)?;
        to_json(result)
    }

    async fn create_source(&self, data: Value) -> Result<Value> {
        let source: crate::source::Source = super::embedded_ext::parse_as(data, "source")?;
        let id = self.inner.hypergraph.create_source(source)?;
        Ok(serde_json::json!({"id": id}))
    }

    async fn get_source(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        let source = self.inner.hypergraph.get_source(&uuid)?;
        to_json(source)
    }

    async fn list_sources(&self) -> Result<Value> {
        let sources = self.inner.hypergraph.list_sources()?;
        to_json(sources)
    }

    async fn add_attribution(&self, data: Value) -> Result<Value> {
        let attr: crate::source::SourceAttribution =
            super::embedded_ext::parse_as(data, "source attribution")?;
        self.inner.hypergraph.add_attribution(attr)?;
        Ok(serde_json::json!({"status": "ok"}))
    }

    async fn list_contentions(&self, situation_id: &str) -> Result<Value> {
        let uuid = parse_uuid(situation_id)?;
        let links = self.inner.hypergraph.get_contentions_for_situation(&uuid)?;
        to_json(links)
    }

    async fn recompute_confidence(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        // Try entity first, fall back to situation
        let current_conf = if let Ok(entity) = self.inner.hypergraph.get_entity(&uuid) {
            entity.confidence
        } else if let Ok(sit) = self.inner.hypergraph.get_situation(&uuid) {
            sit.confidence
        } else {
            return Err(TensaError::NotFound(format!(
                "No entity or situation with id {}",
                uuid
            )));
        };
        let breakdown = self
            .inner
            .hypergraph
            .recompute_confidence(&uuid, current_conf)?;
        Ok(serde_json::json!({
            "id": uuid,
            "breakdown": {
                "extraction": breakdown.extraction,
                "source_credibility": breakdown.source_credibility,
                "corroboration": breakdown.corroboration,
                "recency": breakdown.recency,
            },
            "composite": breakdown.composite(),
        }))
    }

    async fn review_queue(
        &self,
        action: &str,
        item_id: Option<&str>,
        reviewer: Option<&str>,
        notes: Option<&str>,
        edited_data: Option<Value>,
        limit: Option<usize>,
    ) -> Result<Value> {
        match action {
            "list" => {
                let items = self
                    .inner
                    .validation_queue
                    .list_pending(limit.unwrap_or(50))?;
                to_json(items)
            }
            "get" => {
                let id = parse_uuid(item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'get'".into())
                })?)?;
                let item = self.inner.validation_queue.get(&id)?;
                to_json(item)
            }
            "approve" => {
                let id = parse_uuid(item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'approve'".into())
                })?)?;
                let reviewer = reviewer.ok_or_else(|| {
                    TensaError::InvalidQuery("reviewer required for 'approve'".into())
                })?;
                let item = self.inner.validation_queue.approve(&id, reviewer)?;
                to_json(item)
            }
            "reject" => {
                let id = parse_uuid(item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'reject'".into())
                })?)?;
                let reviewer = reviewer.ok_or_else(|| {
                    TensaError::InvalidQuery("reviewer required for 'reject'".into())
                })?;
                let item =
                    self.inner
                        .validation_queue
                        .reject(&id, reviewer, notes.map(String::from))?;
                to_json(item)
            }
            "edit" => {
                let id = parse_uuid(item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'edit'".into())
                })?)?;
                let reviewer = reviewer.ok_or_else(|| {
                    TensaError::InvalidQuery("reviewer required for 'edit'".into())
                })?;
                let data = edited_data.ok_or_else(|| {
                    TensaError::InvalidQuery("edited_data required for 'edit'".into())
                })?;
                let item = self
                    .inner
                    .validation_queue
                    .edit_and_approve(&id, reviewer, data)?;
                to_json(item)
            }
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown review action: '{}'. Use list/get/approve/reject/edit.",
                other
            ))),
        }
    }

    async fn delete_entity(&self, id: &str) -> Result<Value> {
        self.delete_entity_impl(id).await
    }

    async fn delete_situation(&self, id: &str) -> Result<Value> {
        self.delete_situation_impl(id).await
    }

    async fn update_entity(&self, id: &str, updates: Value) -> Result<Value> {
        self.update_entity_impl(id, updates).await
    }

    async fn update_situation(&self, id: &str, updates: Value) -> Result<Value> {
        self.update_situation_impl(id, updates).await
    }

    async fn update_participation(
        &self,
        situation_id: &str,
        entity_id: &str,
        seq: u16,
        updates: Value,
    ) -> Result<Value> {
        self.update_participation_impl(situation_id, entity_id, seq, updates)
            .await
    }

    async fn list_entities(
        &self,
        entity_type: Option<&str>,
        narrative_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Value> {
        self.list_entities_impl(entity_type, narrative_id, limit)
            .await
    }

    async fn merge_entities(&self, keep_id: &str, absorb_id: &str) -> Result<Value> {
        self.merge_entities_impl(keep_id, absorb_id).await
    }

    async fn export_narrative(&self, narrative_id: &str, format: &str) -> Result<Value> {
        self.export_narrative_impl(narrative_id, format).await
    }

    async fn get_narrative_stats(
        &self,
        narrative_id: &str,
        _tnorm: Option<&str>,
        _aggregator: Option<&str>,
    ) -> Result<Value> {
        // Embedded path ignores fuzzy knobs — the REST layer mirrors the
        // stats unchanged when ?tnorm=&aggregator= are absent; HTTP-
        // forwarder tests cover the query-string threading.
        self.get_narrative_stats_impl(narrative_id).await
    }

    async fn search_entities(
        &self,
        query: &str,
        limit: Option<usize>,
        _tnorm: Option<&str>,
        _aggregator: Option<&str>,
    ) -> Result<Value> {
        self.search_entities_impl(query, limit).await
    }

    async fn ingest_url(
        &self,
        url: &str,
        narrative_id: &str,
        source_name: Option<&str>,
    ) -> Result<Value> {
        let text = crate::ingestion::web::fetch_and_extract_text(url).await?;
        if text.trim().is_empty() {
            return Err(TensaError::ExtractionError(
                "No text extracted from URL".into(),
            ));
        }
        let source = source_name.unwrap_or(url);
        self.ingest_text(&text, narrative_id, source, None, None)
            .await
    }

    async fn ingest_rss(
        &self,
        feed_url: &str,
        narrative_id: &str,
        max_items: Option<usize>,
    ) -> Result<Value> {
        #[cfg(feature = "web-ingest")]
        {
            let max = max_items.unwrap_or(10);
            let items = crate::ingestion::web::fetch_rss_items(feed_url, max).await?;
            let mut ingested = Vec::new();
            let mut errors = Vec::new();
            for item in &items {
                if item.content.trim().is_empty() {
                    continue;
                }
                let source = item.link.as_deref().unwrap_or(&item.title);
                match self
                    .ingest_text(&item.content, narrative_id, source, None, None)
                    .await
                {
                    Ok(report) => ingested.push(serde_json::json!({
                        "title": item.title,
                        "link": item.link,
                        "report": report,
                    })),
                    Err(e) => errors.push(serde_json::json!({
                        "title": item.title,
                        "error": e.to_string(),
                    })),
                }
            }
            Ok(serde_json::json!({
                "items_found": items.len(),
                "items_ingested": ingested.len(),
                "items": ingested,
                "errors": errors,
            }))
        }
        #[cfg(not(feature = "web-ingest"))]
        {
            let _ = (feed_url, narrative_id, max_items);
            Err(TensaError::Internal(
                "RSS ingestion requires the 'web-ingest' feature to be enabled".into(),
            ))
        }
    }

    async fn get_actor_profile(&self, actor_id: &str) -> Result<Value> {
        self.get_actor_profile_impl(actor_id).await
    }

    async fn split_entity(&self, entity_id: &str, situation_ids: &[String]) -> Result<Value> {
        self.split_entity_impl(entity_id, situation_ids).await
    }

    async fn restore_entity(&self, id: &str) -> Result<Value> {
        self.restore_entity_impl(id).await
    }

    async fn restore_situation(&self, id: &str) -> Result<Value> {
        self.restore_situation_impl(id).await
    }

    async fn create_project(&self, data: Value) -> Result<Value> {
        self.create_project_impl(data).await
    }

    async fn get_project(&self, id: &str) -> Result<Value> {
        self.get_project_impl(id).await
    }

    async fn list_projects(&self, limit: Option<usize>) -> Result<Value> {
        self.list_projects_impl(limit).await
    }

    async fn update_project(&self, id: &str, updates: Value) -> Result<Value> {
        self.update_project_impl(id, updates).await
    }

    async fn delete_project(&self, id: &str, cascade: bool) -> Result<Value> {
        self.delete_project_impl(id, cascade).await
    }

    async fn cache_stats(&self) -> Result<Value> {
        let cache = crate::ingestion::llm_cache::LlmCache::new(self.store_arc());
        let stats = cache.stats()?;
        serde_json::to_value(stats).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn run_full_analysis(
        &self,
        narrative_id: &str,
        _tiers: Option<Vec<String>>,
        force: bool,
    ) -> Result<Value> {
        // Embedded backend ships a flat curated set of algorithmic jobs (the
        // common foundational + structural + temporal slice). Callers wanting
        // Studio's full 5-tier dispatch with priorities should use the HTTP
        // backend, which delegates to /narratives/:id/analyze. The `_tiers`
        // arg is accepted for API symmetry but ignored here.
        let entities = self.inner.hypergraph.list_entities_by_narrative(narrative_id)?;
        let situations = self
            .inner
            .hypergraph
            .list_situations_by_narrative(narrative_id)?;
        if entities.is_empty() && situations.is_empty() {
            return Err(TensaError::InvalidInput(format!(
                "Narrative '{}' has no entities or situations",
                narrative_id
            )));
        }
        let first_entity_id = entities.first().map(|e| e.id).unwrap_or_default();
        let first_situation_id = situations.first().map(|s| s.id).unwrap_or_default();

        let status_store =
            crate::analysis_status::AnalysisStatusStore::new(self.store_arc());
        let curated: &[(InferenceJobType, JobPriority, Uuid)] = &[
            (InferenceJobType::CentralityAnalysis, JobPriority::High, first_entity_id),
            (InferenceJobType::EntropyAnalysis, JobPriority::High, first_situation_id),
            (InferenceJobType::AnomalyDetection, JobPriority::High, first_situation_id),
            (InferenceJobType::PageRank, JobPriority::Normal, first_entity_id),
            (InferenceJobType::EigenvectorCentrality, JobPriority::Normal, first_entity_id),
            (InferenceJobType::HarmonicCentrality, JobPriority::Normal, first_entity_id),
            (InferenceJobType::HITS, JobPriority::Normal, first_entity_id),
            (InferenceJobType::Topology, JobPriority::Normal, first_entity_id),
            (InferenceJobType::KCore, JobPriority::Normal, first_entity_id),
            (InferenceJobType::LabelPropagation, JobPriority::Normal, first_entity_id),
            (InferenceJobType::TemporalPageRank, JobPriority::Normal, first_entity_id),
            (InferenceJobType::CausalInfluence, JobPriority::Normal, first_entity_id),
            (InferenceJobType::Assortativity, JobPriority::Normal, first_entity_id),
            (InferenceJobType::TemporalMotifs, JobPriority::Normal, first_situation_id),
            (InferenceJobType::FactionEvolution, JobPriority::Normal, first_entity_id),
            (InferenceJobType::FastRP, JobPriority::Low, first_entity_id),
            (InferenceJobType::Node2Vec, JobPriority::Low, first_entity_id),
            (InferenceJobType::ContagionAnalysis, JobPriority::Low, first_situation_id),
        ];

        let params = serde_json::json!({"narrative_id": narrative_id, "scope": "story"});
        let mut submitted = Vec::new();
        let mut skipped = Vec::new();
        for (jt, prio, target) in curated.iter().cloned() {
            if !force {
                if let Ok(Some(existing)) = status_store.get(narrative_id, &jt, "story") {
                    if existing.locked {
                        skipped.push(serde_json::json!({
                            "job_type": jt.variant_name(),
                            "reason": "locked",
                            "existing": existing,
                        }));
                        continue;
                    }
                }
            }
            let job = InferenceJob {
                id: Uuid::now_v7().to_string(),
                job_type: jt.clone(),
                target_id: target,
                parameters: params.clone(),
                priority: prio,
                status: JobStatus::Pending,
                estimated_cost_ms: 0,
                created_at: chrono::Utc::now(),
                started_at: None,
                completed_at: None,
                error: None,
            };
            let job_id = job.id.clone();
            if self.inner.job_queue.submit(job).is_ok() {
                submitted.push(serde_json::json!({
                    "job_id": job_id,
                    "job_type": jt.variant_name(),
                }));
            }
        }

        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "submitted": submitted.len(),
            "jobs": submitted,
            "skipped_locked": skipped,
            "force": force,
            "entities": entities.len(),
            "situations": situations.len(),
        }))
    }

    async fn backfill_embeddings(
        &self,
        narrative_id: Option<&str>,
        force: bool,
    ) -> Result<Value> {
        let embedder = self
            .inner
            .embedder
            .as_ref()
            .ok_or_else(|| {
                TensaError::InvalidInput("No embedding provider configured".into())
            })?
            .clone();

        let entities = match narrative_id {
            Some(nid) => self.inner.hypergraph.list_entities_by_narrative(nid)?,
            None => self
                .inner
                .hypergraph
                .list_entities_by_maturity(crate::types::MaturityLevel::Candidate)?,
        };
        let situations = match narrative_id {
            Some(nid) => self.inner.hypergraph.list_situations_by_narrative(nid)?,
            None => self
                .inner
                .hypergraph
                .list_situations_by_maturity(crate::types::MaturityLevel::Candidate)?,
        };

        let mut skipped: usize = 0;
        let mut ent_ids = Vec::new();
        let mut ent_texts = Vec::new();
        for e in &entities {
            if !force && e.embedding.is_some() {
                skipped += 1;
                continue;
            }
            let text = e
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| e.entity_type.as_index_str())
                .to_string();
            ent_ids.push(e.id);
            ent_texts.push(text);
        }
        let mut sit_ids = Vec::new();
        let mut sit_texts = Vec::new();
        for s in &situations {
            if !force && s.embedding.is_some() {
                skipped += 1;
                continue;
            }
            let text = s
                .description
                .as_deref()
                .or_else(|| s.raw_content.first().map(|c| c.content.as_str()))
                .or_else(|| s.name.as_deref())
                .unwrap_or("unnamed")
                .to_string();
            sit_ids.push(s.id);
            sit_texts.push(text);
        }

        let ent_refs: Vec<&str> = ent_texts.iter().map(|s| s.as_str()).collect();
        let ent_embeddings = embedder.embed_batch(&ent_refs)?;
        let sit_refs: Vec<&str> = sit_texts.iter().map(|s| s.as_str()).collect();
        let sit_embeddings = embedder.embed_batch(&sit_refs)?;

        let mut entities_embedded: usize = 0;
        for (id, emb) in ent_ids.iter().zip(ent_embeddings.iter()) {
            if self
                .inner
                .hypergraph
                .update_entity_no_snapshot(id, |e| {
                    e.embedding = Some(emb.clone());
                })
                .is_ok()
            {
                entities_embedded += 1;
            }
        }
        let mut situations_embedded: usize = 0;
        for (id, emb) in sit_ids.iter().zip(sit_embeddings.iter()) {
            if self
                .inner
                .hypergraph
                .update_situation(id, |s| {
                    s.embedding = Some(emb.clone());
                })
                .is_ok()
            {
                situations_embedded += 1;
            }
        }
        if let Some(vi) = &self.inner.vector_index {
            if let Ok(mut idx) = vi.write() {
                for (id, emb) in ent_ids.iter().zip(ent_embeddings.iter()) {
                    let _ = idx.add(*id, emb);
                }
                for (id, emb) in sit_ids.iter().zip(sit_embeddings.iter()) {
                    let _ = idx.add(*id, emb);
                }
                let _ = idx.save(self.inner.hypergraph.store());
            }
        }

        Ok(serde_json::json!({
            "entities_embedded": entities_embedded,
            "situations_embedded": situations_embedded,
            "total_updated": entities_embedded + situations_embedded,
            "skipped": skipped,
        }))
    }

    async fn ask(
        &self,
        question: &str,
        narrative_id: Option<&str>,
        mode: Option<&str>,
    ) -> Result<Value> {
        let extractor = self
            .inner
            .extractor
            .as_ref()
            .ok_or_else(|| TensaError::LlmError("No LLM extractor configured".into()))?;
        let rag_mode =
            crate::query::rag_config::RetrievalMode::from_str_or_default(mode.unwrap_or("hybrid"));
        let budget = crate::query::token_budget::TokenBudget::default();
        let vi_guard = self
            .inner
            .vector_index
            .as_ref()
            .and_then(|vi| vi.read().ok());
        let vi_ref = vi_guard.as_deref();
        let emb_ref = self.inner.embedder.as_deref();
        let answer = crate::query::rag::execute_ask(
            question,
            narrative_id,
            &rag_mode,
            &budget,
            &self.inner.hypergraph,
            vi_ref,
            emb_ref,
            extractor.as_ref(),
            None,  // no reranker in MCP path
            None,  // no response_type in MCP path
            false, // no suggest in MCP path
            None,  // no session in MCP path
        )?;
        serde_json::to_value(answer).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn tune_prompts(&self, narrative_id: &str) -> Result<Value> {
        let extractor = self
            .inner
            .extractor
            .as_ref()
            .ok_or_else(|| TensaError::LlmError("No LLM extractor configured".into()))?;
        let tuned = crate::ingestion::prompt_tuning::tune_prompts(
            self.inner.hypergraph.store(),
            extractor.as_ref(),
            &self.inner.hypergraph,
            narrative_id,
        )?;
        serde_json::to_value(tuned).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn community_hierarchy(&self, narrative_id: &str, level: Option<usize>) -> Result<Value> {
        let store = self.inner.hypergraph.store();
        let result = if let Some(lvl) = level {
            crate::analysis::community::list_summaries_at_level(store, narrative_id, lvl)?
        } else {
            crate::analysis::community::list_summaries(store, narrative_id)?
        };
        serde_json::to_value(result).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn export_archive(&self, narrative_ids: Vec<String>) -> Result<Value> {
        use base64::Engine;
        let refs: Vec<&str> = narrative_ids.iter().map(|s| s.as_str()).collect();
        let opts = crate::export::archive_types::ArchiveExportOptions::default();
        let bytes = crate::export::archive::export_archive(&refs, &self.inner.hypergraph, &opts)?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        Ok(serde_json::json!({
            "format": "base64",
            "size_bytes": bytes.len(),
            "data": encoded,
        }))
    }

    async fn import_archive(&self, data_base64: &str) -> Result<Value> {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_base64)
            .map_err(|e| TensaError::Internal(format!("Invalid base64: {e}")))?;
        let opts = crate::export::archive_types::ArchiveImportOptions::default();
        let report =
            crate::ingestion::archive::import_archive(&bytes, &self.inner.hypergraph, &opts)?;
        serde_json::to_value(report).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn verify_authorship(&self, text_a: &str, text_b: &str) -> Result<Value> {
        use crate::analysis::pan_verification::{verify_texts, VerificationConfig};
        let (score, decision) = verify_texts(text_a, text_b, &VerificationConfig::default());
        Ok(serde_json::json!({
            "score": score,
            "decision": decision,
            "same_author_probability": score,
        }))
    }

    // ─── Narrative Debugger (Sprint D10) ─────────────────────

    async fn diagnose_narrative(&self, narrative_id: &str, genre: Option<&str>) -> Result<Value> {
        use crate::narrative::debug::{
            diagnose_narrative_with, store_diagnosis, DiagnosticConfig, GenrePreset,
        };
        use std::str::FromStr;
        let g = genre
            .and_then(|s| GenrePreset::from_str(s).ok())
            .unwrap_or_default();
        let cfg = DiagnosticConfig::for_genre(g);
        let diag = diagnose_narrative_with(&self.inner.hypergraph, narrative_id, &cfg)?;
        let _ = store_diagnosis(&self.inner.hypergraph, &diag);
        serde_json::to_value(diag).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn get_health_score(&self, narrative_id: &str) -> Result<Value> {
        let diag =
            crate::narrative::debug::diagnose_narrative(&self.inner.hypergraph, narrative_id)?;
        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "health_score": diag.health_score,
            "errors": diag.error_count,
            "warnings": diag.warning_count,
            "infos": diag.info_count,
            "worst_chapter": diag.summary.worst_chapter,
        }))
    }

    async fn auto_repair(
        &self,
        narrative_id: &str,
        max_severity: Option<&str>,
        max_iterations: Option<usize>,
    ) -> Result<Value> {
        use crate::narrative::debug::PathologySeverity;
        use std::str::FromStr;
        let sev = max_severity
            .and_then(|s| PathologySeverity::from_str(s).ok())
            .unwrap_or(PathologySeverity::Warning);
        let iters = max_iterations.unwrap_or(5).clamp(1, 20);
        let report = crate::narrative::debug_fixes::auto_repair(
            &self.inner.hypergraph,
            narrative_id,
            sev,
            iters,
        )?;
        serde_json::to_value(report).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    // ─── Narrative Adaptation (Sprint D11) ────────────────

    async fn compute_essentiality(&self, narrative_id: &str) -> Result<Value> {
        let report = crate::narrative::essentiality::compute_essentiality(
            &self.inner.hypergraph,
            narrative_id,
        )?;
        let _ = crate::narrative::essentiality::store_essentiality(&self.inner.hypergraph, &report);
        serde_json::to_value(report).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn compress_narrative(
        &self,
        narrative_id: &str,
        preset: Option<&str>,
        target_chapters: Option<usize>,
        target_ratio: Option<f64>,
    ) -> Result<Value> {
        use crate::narrative::compression::{
            compress_narrative, compress_to_novella, compress_to_screenplay_outline,
            compress_to_short_story, CompressionConfig,
        };
        let plan = match preset {
            Some("novella") => compress_to_novella(&self.inner.hypergraph, narrative_id)?,
            Some("short_story") => compress_to_short_story(&self.inner.hypergraph, narrative_id)?,
            Some("screenplay_outline") => {
                compress_to_screenplay_outline(&self.inner.hypergraph, narrative_id)?
            }
            _ => {
                let cfg = CompressionConfig {
                    target_chapters: target_chapters.unwrap_or(12),
                    target_ratio,
                    ..Default::default()
                };
                compress_narrative(&self.inner.hypergraph, narrative_id, &cfg)?
            }
        };
        serde_json::to_value(plan).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn expand_narrative(&self, narrative_id: &str, target_chapters: usize) -> Result<Value> {
        let plan = crate::narrative::expansion::expand_to_novel(
            &self.inner.hypergraph,
            narrative_id,
            target_chapters,
        )?;
        serde_json::to_value(plan).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    async fn diff_narratives(&self, narrative_a: &str, narrative_b: &str) -> Result<Value> {
        let diff = crate::narrative::diff::diff_narratives(
            &self.inner.hypergraph,
            narrative_a,
            narrative_b,
        )?;
        serde_json::to_value(diff).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    // ─── Disinfo Sprint D1: dual fingerprint tools ─────────────

    #[cfg(feature = "disinfo")]
    async fn get_behavioral_fingerprint(&self, actor_id: &str, recompute: bool) -> Result<Value> {
        use crate::disinfo::{behavioral_envelope, ensure_behavioral_fingerprint};
        let id = parse_uuid(actor_id)?;
        let fp = ensure_behavioral_fingerprint(&self.inner.hypergraph, &id, recompute)?;
        Ok(behavioral_envelope(&fp))
    }

    #[cfg(feature = "disinfo")]
    async fn get_disinfo_fingerprint(&self, narrative_id: &str, recompute: bool) -> Result<Value> {
        use crate::disinfo::{disinfo_envelope, ensure_disinfo_fingerprint};
        let fp = ensure_disinfo_fingerprint(&self.inner.hypergraph, narrative_id, recompute)?;
        Ok(disinfo_envelope(&fp))
    }

    #[cfg(feature = "disinfo")]
    async fn compare_fingerprints(
        &self,
        kind: &str,
        task: Option<&str>,
        a_id: &str,
        b_id: &str,
    ) -> Result<Value> {
        use crate::disinfo::{
            compare_fingerprints, ensure_behavioral_fingerprint, ensure_disinfo_fingerprint,
            ComparisonKind,
        };

        let comp_kind = parse_comparison_kind(kind)?;
        let comp_task = parse_comparison_task(task.unwrap_or("literary"))?;
        let (a_value, b_value) = match comp_kind {
            ComparisonKind::Behavioral => {
                let a_uuid = parse_uuid(a_id)?;
                let b_uuid = parse_uuid(b_id)?;
                let a = ensure_behavioral_fingerprint(&self.inner.hypergraph, &a_uuid, false)?;
                let b = ensure_behavioral_fingerprint(&self.inner.hypergraph, &b_uuid, false)?;
                (to_value(&a)?, to_value(&b)?)
            }
            ComparisonKind::Disinfo => {
                let a = ensure_disinfo_fingerprint(&self.inner.hypergraph, a_id, false)?;
                let b = ensure_disinfo_fingerprint(&self.inner.hypergraph, b_id, false)?;
                (to_value(&a)?, to_value(&b)?)
            }
            ComparisonKind::Narrative => {
                return Err(TensaError::InvalidQuery(
                    "narrative-content fingerprint comparison: use the verify_authorship tool"
                        .into(),
                ));
            }
        };
        let comparison = compare_fingerprints(comp_kind, comp_task, &a_value, &b_value)?;
        to_value(&comparison)
    }

    // ─── Spread Dynamics (Sprint D2) ───────────────────────────

    #[cfg(feature = "disinfo")]
    async fn estimate_r0_by_platform(
        &self,
        narrative_id: &str,
        fact: &str,
        about_entity: &str,
        narrative_kind: Option<&str>,
        beta_overrides: Option<std::collections::HashMap<String, f64>>,
    ) -> Result<Value> {
        let about = parse_uuid(about_entity)?;
        let beta = parse_beta_overrides_map(beta_overrides);
        crate::analysis::contagion::compute_spread_r0_payload(
            &self.inner.hypergraph,
            narrative_id,
            fact,
            about,
            narrative_kind.unwrap_or("default"),
            &beta,
        )
    }

    #[cfg(feature = "disinfo")]
    async fn simulate_intervention(
        &self,
        narrative_id: &str,
        fact: &str,
        about_entity: &str,
        intervention: Value,
        beta_overrides: Option<std::collections::HashMap<String, f64>>,
    ) -> Result<Value> {
        use crate::analysis::spread_intervention::{parse_intervention, simulate_intervention};

        let about = parse_uuid(about_entity)?;
        let beta = parse_beta_overrides_map(beta_overrides);
        let parsed = parse_intervention(&intervention)?;
        let projection = simulate_intervention(
            &self.inner.hypergraph,
            narrative_id,
            fact,
            about,
            parsed,
            &beta,
        )?;
        to_value(&projection)
    }

    // ─── CIB Detection (Sprint D3) ───────────────────────────

    #[cfg(feature = "disinfo")]
    async fn detect_cib_cluster(
        &self,
        narrative_id: &str,
        cross_platform: bool,
        similarity_threshold: Option<f64>,
        alpha: Option<f64>,
        bootstrap_iter: Option<usize>,
        min_cluster_size: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Value> {
        use crate::analysis::cib::{detect_cib_clusters, detect_cross_platform_cib, CibConfig};
        let config = CibConfig::from_json(&serde_json::json!({
            "similarity_threshold": similarity_threshold,
            "alpha": alpha,
            "bootstrap_iter": bootstrap_iter,
            "min_cluster_size": min_cluster_size,
            "seed": seed,
        }));
        let result = if cross_platform {
            detect_cross_platform_cib(&self.inner.hypergraph, narrative_id, &config)?
        } else {
            detect_cib_clusters(&self.inner.hypergraph, narrative_id, &config)?
        };
        to_value(&result)
    }

    #[cfg(feature = "disinfo")]
    async fn rank_superspreaders(
        &self,
        narrative_id: &str,
        method: Option<&str>,
        top_n: Option<usize>,
    ) -> Result<Value> {
        use crate::analysis::cib::{rank_superspreaders, SuperspreaderMethod};
        use std::str::FromStr;
        let method = match method {
            Some(s) => SuperspreaderMethod::from_str(s)?,
            None => SuperspreaderMethod::PageRank,
        };
        let top_n = top_n.unwrap_or(10);
        let ranking = rank_superspreaders(&self.inner.hypergraph, narrative_id, method, top_n)?;
        to_value(&ranking)
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_claim(&self, text: &str, narrative_id: Option<&str>) -> Result<Value> {
        let claims = crate::claims::detect_claims(text, narrative_id, None, None);
        for claim in &claims {
            crate::claims::detection::store_claim(&self.inner.hypergraph, claim)?;
        }
        to_value(&serde_json::json!({
            "claims_detected": claims.len(),
            "claims": claims,
        }))
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_fact_check(
        &self,
        claim_id: &str,
        verdict: &str,
        source: &str,
        url: Option<&str>,
        language: &str,
    ) -> Result<Value> {
        let uuid: uuid::Uuid = claim_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid claim UUID: {}", e)))?;
        let v: crate::claims::FactCheckVerdict = verdict.parse()?;
        let fc = crate::claims::ingest_fact_check(
            &self.inner.hypergraph,
            uuid,
            v,
            source,
            url,
            language,
            None,
            0.9,
        )?;
        to_value(&fc)
    }

    #[cfg(feature = "disinfo")]
    async fn fetch_fact_checks(&self, claim_id: &str, min_similarity: f64) -> Result<Value> {
        let uuid: uuid::Uuid = claim_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid claim UUID: {}", e)))?;
        let claim = crate::claims::detection::load_claim(&self.inner.hypergraph, &uuid)?
            .ok_or_else(|| TensaError::NotFound(format!("Claim {} not found", claim_id)))?;
        let matches = crate::claims::match_claim(&self.inner.hypergraph, &claim, min_similarity)?;
        to_value(&matches)
    }

    #[cfg(feature = "disinfo")]
    async fn trace_claim_origin(&self, claim_id: &str) -> Result<Value> {
        let uuid: uuid::Uuid = claim_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid claim UUID: {}", e)))?;
        let claim = crate::claims::detection::load_claim(&self.inner.hypergraph, &uuid)?
            .ok_or_else(|| TensaError::NotFound(format!("Claim {} not found", claim_id)))?;
        let mutations = crate::claims::track_mutations(&self.inner.hypergraph, &claim)?;
        let mut chain = vec![crate::claims::ClaimAppearance {
            claim_id: claim.id,
            situation_id: claim.source_situation_id,
            entity_id: claim.source_entity_id,
            timestamp: Some(claim.created_at),
            similarity_to_original: 1.0,
        }];
        for m in &mutations {
            if m.original_claim_id != claim.id {
                if let Ok(Some(orig)) = crate::claims::detection::load_claim(
                    &self.inner.hypergraph,
                    &m.original_claim_id,
                ) {
                    chain.push(crate::claims::ClaimAppearance {
                        claim_id: orig.id,
                        situation_id: orig.source_situation_id,
                        entity_id: orig.source_entity_id,
                        timestamp: Some(orig.created_at),
                        similarity_to_original: 1.0 - m.embedding_drift,
                    });
                }
            }
        }
        chain.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        let earliest = chain.first().cloned();
        let trace = crate::claims::ClaimOriginTrace {
            claim_id: claim.id,
            chain,
            earliest,
        };
        to_value(&trace)
    }

    #[cfg(feature = "disinfo")]
    async fn classify_archetype(&self, actor_id: &str, force: bool) -> Result<Value> {
        let uuid: uuid::Uuid = actor_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid actor UUID: {}", e)))?;
        if !force {
            if let Some(cached) =
                crate::disinfo::archetypes::load_archetype(&self.inner.hypergraph, &uuid)?
            {
                return to_value(&cached);
            }
        }
        let dist =
            crate::disinfo::archetypes::classify_actor_archetype(&self.inner.hypergraph, uuid)?;
        to_value(&dist)
    }

    #[cfg(feature = "disinfo")]
    async fn assess_disinfo(&self, target_id: &str, signals: Value) -> Result<Value> {
        let sigs: Vec<crate::disinfo::fusion::DisinfoSignal> = serde_json::from_value(signals)
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid signals: {}", e)))?;
        let assessment = crate::disinfo::fusion::fuse_disinfo_signals(target_id, &sigs)?;
        let _ = crate::disinfo::fusion::store_assessment(&self.inner.hypergraph, &assessment);
        to_value(&assessment)
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_post(
        &self,
        text: &str,
        actor_id: &str,
        narrative_id: &str,
        platform: Option<&str>,
    ) -> Result<Value> {
        use chrono::Utc;
        let actor_uuid: uuid::Uuid = actor_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid actor UUID: {}", e)))?;
        let mut props = serde_json::json!({ "text": text });
        if let Some(p) = platform {
            props["platform"] = serde_json::Value::String(p.to_string());
        }
        let sit = crate::types::Situation {
            id: uuid::Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: crate::types::AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
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
            raw_content: vec![crate::types::ContentBlock::text(text)],
            narrative_level: crate::types::NarrativeLevel::Event,
            narrative_id: Some(narrative_id.to_string()),
            discourse: None,
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: crate::types::ExtractionMethod::StructuredImport,
            provenance: vec![],
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            source_chunk_id: None,
            source_span: None,
            deleted_at: None,
            transaction_time: None,
        };
        let sid = self.inner.hypergraph.create_situation(sit)?;
        let participation = crate::types::Participation {
            entity_id: actor_uuid,
            situation_id: sid,
            role: crate::types::Role::Custom("author".into()),
            info_set: None,
            action: Some("posted".to_string()),
            payoff: None,
            seq: 0,
        };
        self.inner.hypergraph.add_participant(participation)?;
        Ok(serde_json::json!({ "situation_id": sid.to_string(), "actor_id": actor_id }))
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_actor(
        &self,
        name: &str,
        narrative_id: &str,
        platform: Option<&str>,
        properties: Option<Value>,
    ) -> Result<Value> {
        use chrono::Utc;
        let mut props = properties.unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = props.as_object_mut() {
            obj.insert(
                "name".to_string(),
                serde_json::Value::String(name.to_string()),
            );
            if let Some(p) = platform {
                obj.insert(
                    "platform".to_string(),
                    serde_json::Value::String(p.to_string()),
                );
            }
        }
        let entity = crate::types::Entity {
            id: uuid::Uuid::now_v7(),
            entity_type: crate::types::EntityType::Actor,
            properties: props,
            beliefs: None,
            embedding: None,
            narrative_id: Some(narrative_id.to_string()),
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: None,
            provenance: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let id = self.inner.hypergraph.create_entity(entity)?;
        Ok(serde_json::json!({ "entity_id": id.to_string(), "name": name }))
    }

    #[cfg(feature = "disinfo")]
    async fn link_narrative(&self, entity_id: &str, narrative_id: &str) -> Result<Value> {
        let uuid: uuid::Uuid = entity_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid entity UUID: {}", e)))?;
        let nar_id = narrative_id.to_string();
        self.inner
            .hypergraph
            .update_entity_no_snapshot(&uuid, |e| {
                e.narrative_id = Some(nar_id.clone());
                e.updated_at = chrono::Utc::now();
            })?;
        Ok(
            serde_json::json!({ "entity_id": entity_id, "narrative_id": narrative_id, "linked": true }),
        )
    }

    // ─── Multilingual & Export (Sprint D6) ───────────────────────

    #[cfg(feature = "disinfo")]
    async fn detect_language(&self, text: &str) -> Result<Value> {
        let result = crate::disinfo::multilingual::detect_language(text);
        Ok(serde_json::json!({
            "language": result.language,
            "confidence": result.confidence,
        }))
    }

    #[cfg(feature = "disinfo")]
    async fn export_misp_event(&self, narrative_id: &str) -> Result<Value> {
        let event = crate::export::misp::export_misp(&self.inner.hypergraph, narrative_id)?;
        serde_json::to_value(&event).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    #[cfg(feature = "disinfo")]
    async fn export_maltego(&self, narrative_id: &str) -> Result<Value> {
        let transform =
            crate::export::maltego::export_maltego(&self.inner.hypergraph, narrative_id)?;
        serde_json::to_value(&transform).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    #[cfg(feature = "disinfo")]
    async fn generate_disinfo_report(&self, narrative_id: &str) -> Result<Value> {
        let report = crate::export::disinfo_report::export_disinfo_report(
            &self.inner.hypergraph,
            narrative_id,
        )?;
        Ok(serde_json::json!({ "narrative_id": narrative_id, "report": report }))
    }

    // ─── Sprint D8: Scheduler + Reports + Health ─────────────────

    #[cfg(feature = "disinfo")]
    async fn list_scheduled_tasks(&self) -> Result<Value> {
        let tasks = crate::scheduler::engine::list_tasks(&self.inner.hypergraph)?;
        to_value(&tasks)
    }

    #[cfg(feature = "disinfo")]
    async fn create_scheduled_task(
        &self,
        task_type: &str,
        schedule: &str,
        config: Option<Value>,
    ) -> Result<Value> {
        let tt: crate::scheduler::types::ScheduledTaskType = task_type.parse()?;
        let sched = crate::scheduler::types::parse_schedule(schedule)?;
        let task = crate::scheduler::types::ScheduledTask {
            id: uuid::Uuid::now_v7(),
            task_type: tt,
            schedule: sched,
            enabled: true,
            last_run: None,
            last_result: None,
            config: config.unwrap_or(serde_json::Value::Null),
            created_at: chrono::Utc::now(),
        };
        let id = crate::scheduler::engine::create_task(&self.inner.hypergraph, &task)?;
        Ok(serde_json::json!({ "id": id.to_string(), "status": "created" }))
    }

    #[cfg(feature = "disinfo")]
    async fn run_task_now(&self, task_id: &str) -> Result<Value> {
        let uuid: uuid::Uuid = task_id
            .parse()
            .map_err(|e| TensaError::InvalidQuery(format!("Invalid task UUID: {}", e)))?;
        let _task = crate::scheduler::engine::get_task(&self.inner.hypergraph, &uuid)?
            .ok_or_else(|| TensaError::NotFound(format!("Task {} not found", task_id)))?;
        let result = crate::scheduler::types::TaskResult::Success {
            duration_ms: 0,
            summary: "Manual trigger via MCP".to_string(),
        };
        crate::scheduler::engine::mark_task_run(&self.inner.hypergraph, &uuid, result)?;
        Ok(serde_json::json!({ "status": "triggered", "task_id": task_id }))
    }

    #[cfg(feature = "disinfo")]
    async fn list_discovery_candidates(&self) -> Result<Value> {
        let candidates = crate::ingestion::discovery::list_candidates(&self.inner.hypergraph)?;
        to_value(&candidates)
    }

    #[cfg(feature = "disinfo")]
    async fn sync_fact_checks(&self) -> Result<Value> {
        let sources = crate::claims::sync::default_sources();
        let results = crate::claims::sync::execute_sync(
            &self.inner.hypergraph,
            &sources,
            &["en".to_string()],
        )?;
        to_value(&results)
    }

    #[cfg(feature = "disinfo")]
    async fn generate_situation_report(&self, hours: u64) -> Result<Value> {
        let end = chrono::Utc::now();
        let start = end - chrono::Duration::hours(hours as i64);
        let report = crate::export::situation_report::generate_situation_report(
            &self.inner.hypergraph,
            start,
            end,
        )?;
        to_value(&report)
    }

    // ─── Adversarial Wargaming (Sprint D12) ──────────────────────

    async fn generate_adversary_policy(
        &self,
        narrative_id: &str,
        actor_id: Option<&str>,
        archetype: Option<&str>,
        lambda: Option<f64>,
        lambda_cap: Option<f64>,
        reward_weights: Option<Vec<f64>>,
    ) -> Result<Value> {
        #[cfg(feature = "adversarial")]
        {
            use crate::adversarial::policy_gen::{
                store_policy, AdversaryPolicyGenerator, CachedPolicy,
            };
            use crate::adversarial::suqr::SubjectiveUtilityQR;
            use crate::adversarial::types::OperationalConstraints;

            let weights = reward_weights.unwrap_or_else(|| vec![1.0; 5]);
            let lam = lambda.unwrap_or(1.0);
            let cap = lambda_cap.unwrap_or(4.6);

            let (entity_id, has_real_id) = match actor_id {
                Some(id) => (parse_uuid(id)?, true),
                None => (Uuid::now_v7(), false),
            };

            let generator = AdversaryPolicyGenerator {
                rationality: SubjectiveUtilityQR::new(lam, weights, cap),
                constraints: OperationalConstraints::default(),
                archetype: archetype.map(String::from),
                narrative_id: Some(narrative_id.to_string()),
            };

            let actions = generator.generate_turn(chrono::Utc::now());

            let cached = CachedPolicy {
                policy: generator,
                entity_id,
                created_at: chrono::Utc::now(),
                last_actions: actions,
            };

            // Only persist when we have a real entity ID (ephemeral UUIDs are unreachable)
            if has_real_id {
                if let Err(e) =
                    store_policy(&self.inner.hypergraph, narrative_id, &entity_id, &cached)
                {
                    tracing::warn!("Failed to cache adversary policy: {}", e);
                }
            }
            return to_value(&cached);
        }
        #[cfg(not(feature = "adversarial"))]
        {
            let _ = (
                narrative_id,
                actor_id,
                archetype,
                lambda,
                lambda_cap,
                reward_weights,
            );
            Err(TensaError::Internal(
                "adversarial feature not enabled".into(),
            ))
        }
    }

    async fn configure_rationality(
        &self,
        model: &str,
        lambda: Option<f64>,
        lambda_cap: Option<f64>,
        tau: Option<f64>,
        feature_weights: Option<Vec<f64>>,
    ) -> Result<Value> {
        #[cfg(feature = "adversarial")]
        {
            use crate::adversarial::cognitive_hierarchy::CognitiveHierarchy;
            use crate::adversarial::suqr::SubjectiveUtilityQR;

            match model.to_lowercase().as_str() {
                "qre" => {
                    let lam = lambda.unwrap_or(1.0);
                    Ok(serde_json::json!({
                        "model": "qre",
                        "lambda": lam,
                        "description": "Standard Quantal Response Equilibrium"
                    }))
                }
                "suqr" => {
                    let weights = feature_weights.unwrap_or_else(|| vec![1.0]);
                    let lam = lambda.unwrap_or(1.0);
                    let cap = lambda_cap.unwrap_or(4.6);
                    let suqr = SubjectiveUtilityQR::new(lam, weights, cap);
                    to_value(&suqr)
                }
                "cognitive_hierarchy" | "ch" => {
                    let t = tau.unwrap_or(1.5);
                    let ch = CognitiveHierarchy {
                        tau: t,
                        max_level: 5,
                    };
                    to_value(&ch)
                }
                other => Err(TensaError::InvalidQuery(format!(
                    "Unknown rationality model '{}' (use 'qre', 'suqr', or 'cognitive_hierarchy')",
                    other
                ))),
            }
        }
        #[cfg(not(feature = "adversarial"))]
        {
            let _ = (model, lambda, lambda_cap, tau, feature_weights);
            Err(TensaError::Internal(
                "adversarial feature not enabled".into(),
            ))
        }
    }

    async fn create_wargame(
        &self,
        narrative_id: &str,
        max_turns: Option<usize>,
        time_step_minutes: Option<u64>,
        auto_red: Option<bool>,
        auto_blue: Option<bool>,
    ) -> Result<Value> {
        #[cfg(feature = "adversarial")]
        {
            use crate::adversarial::session::*;
            let config = WargameConfig {
                max_turns: max_turns.unwrap_or(20),
                time_step_minutes: time_step_minutes.unwrap_or(60),
                auto_red: auto_red.unwrap_or(true),
                auto_blue: auto_blue.unwrap_or(false),
                ..Default::default()
            };
            let session = WargameSession::create(&self.inner.hypergraph, narrative_id, config)?;
            let sid = session.session_id.clone();
            store_session(&self.inner.hypergraph, &session)?;
            return Ok(serde_json::json!({"session_id": sid, "status": "Created"}));
        }
        #[cfg(not(feature = "adversarial"))]
        {
            let _ = (
                narrative_id,
                max_turns,
                time_step_minutes,
                auto_red,
                auto_blue,
            );
            Err(TensaError::Internal(
                "adversarial feature not enabled".into(),
            ))
        }
    }

    async fn submit_wargame_move(
        &self,
        session_id: &str,
        red_moves: Option<Value>,
        blue_moves: Option<Value>,
    ) -> Result<Value> {
        #[cfg(feature = "adversarial")]
        {
            use crate::adversarial::session::*;
            use crate::adversarial::wargame::WargameMove;
            let mut session =
                load_session(&self.inner.hypergraph, session_id)?.ok_or_else(|| {
                    TensaError::Internal(format!("Session not found: {}", session_id))
                })?;

            let red: Vec<WargameMove> = match red_moves {
                Some(v) => serde_json::from_value(v)
                    .map_err(|e| TensaError::Internal(format!("red_moves parse error: {e}")))?,
                None => vec![],
            };
            let blue: Vec<WargameMove> = match blue_moves {
                Some(v) => serde_json::from_value(v)
                    .map_err(|e| TensaError::Internal(format!("blue_moves parse error: {e}")))?,
                None => vec![],
            };

            let result = session.play_turn(&red, &blue)?;
            store_session(&self.inner.hypergraph, &session)?;
            return to_value(&result);
        }
        #[cfg(not(feature = "adversarial"))]
        {
            let _ = (session_id, red_moves, blue_moves);
            Err(TensaError::Internal(
                "adversarial feature not enabled".into(),
            ))
        }
    }

    async fn get_wargame_state(&self, session_id: &str) -> Result<Value> {
        #[cfg(feature = "adversarial")]
        {
            use crate::adversarial::session::*;
            let session = load_session(&self.inner.hypergraph, session_id)?.ok_or_else(|| {
                TensaError::Internal(format!("Session not found: {}", session_id))
            })?;
            // EATH Phase 10 — surface SubstrateProvenance so callers see whether
            // the session is running on real, synthetic, or hybrid data without
            // round-tripping into the wargame KV manually. Additive change:
            // existing summary fields are preserved; provenance is grafted on
            // as a new top-level `substrate_provenance` key. Callers reading
            // `result.turn`, `result.r0`, etc., are unaffected.
            let mut value = serde_json::to_value(session.summary())
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            let provenance = serde_json::to_value(session.substrate_provenance())
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            if let Some(obj) = value.as_object_mut() {
                obj.insert("substrate_provenance".into(), provenance);
            }
            return Ok(value);
        }
        #[cfg(not(feature = "adversarial"))]
        {
            let _ = session_id;
            Err(TensaError::Internal(
                "adversarial feature not enabled".into(),
            ))
        }
    }

    // ─── Writer Workflows (Sprint W6) ────────────────────────────

    async fn get_narrative_plan(&self, narrative_id: &str) -> Result<Value> {
        self.get_narrative_plan_impl(narrative_id).await
    }

    async fn upsert_narrative_plan(&self, narrative_id: &str, patch: Value) -> Result<Value> {
        self.upsert_narrative_plan_impl(narrative_id, patch).await
    }

    async fn get_writer_workspace(&self, narrative_id: &str) -> Result<Value> {
        self.get_writer_workspace_impl(narrative_id).await
    }

    async fn run_workshop(
        &self,
        narrative_id: &str,
        tier: &str,
        focuses: Option<Vec<String>>,
        max_llm_calls: Option<u32>,
    ) -> Result<Value> {
        self.run_workshop_impl(narrative_id, tier, focuses, max_llm_calls)
            .await
    }

    async fn list_pinned_facts(&self, narrative_id: &str) -> Result<Value> {
        self.list_pinned_facts_impl(narrative_id).await
    }

    async fn create_pinned_fact(&self, narrative_id: &str, fact: Value) -> Result<Value> {
        self.create_pinned_fact_impl(narrative_id, fact).await
    }

    async fn check_continuity(&self, narrative_id: &str, prose: &str) -> Result<Value> {
        self.check_continuity_impl(narrative_id, prose).await
    }

    async fn list_narrative_revisions(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        self.list_narrative_revisions_impl(narrative_id, limit)
            .await
    }

    async fn restore_narrative_revision(
        &self,
        narrative_id: &str,
        revision_id: &str,
        author: &str,
    ) -> Result<Value> {
        self.restore_narrative_revision_impl(narrative_id, revision_id, author)
            .await
    }

    async fn get_writer_cost_summary(
        &self,
        narrative_id: &str,
        window: Option<&str>,
    ) -> Result<Value> {
        self.get_writer_cost_summary_impl(narrative_id, window)
            .await
    }

    async fn set_situation_content(
        &self,
        situation_id: &str,
        content: Value,
        status: Option<&str>,
    ) -> Result<Value> {
        self.set_situation_content_impl(situation_id, content, status)
            .await
    }

    async fn get_scene_context(
        &self,
        narrative_id: &str,
        situation_id: Option<&str>,
        pov_entity_id: Option<&str>,
        lookback_scenes: Option<usize>,
    ) -> Result<Value> {
        self.get_scene_context_impl(narrative_id, situation_id, pov_entity_id, lookback_scenes)
            .await
    }

    async fn auto_play_wargame(&self, session_id: &str, num_turns: usize) -> Result<Value> {
        #[cfg(feature = "adversarial")]
        {
            use crate::adversarial::session::*;
            let mut session =
                load_session(&self.inner.hypergraph, session_id)?.ok_or_else(|| {
                    TensaError::Internal(format!("Session not found: {}", session_id))
                })?;
            let results = session.auto_play(num_turns)?;
            store_session(&self.inner.hypergraph, &session)?;
            return to_value(&results);
        }
        #[cfg(not(feature = "adversarial"))]
        {
            let _ = (session_id, num_turns);
            Err(TensaError::Internal(
                "adversarial feature not enabled".into(),
            ))
        }
    }

    // ─── Sprint W14: narrative architecture ───────────────────────

    async fn detect_commitments(&self, narrative_id: &str) -> Result<Value> {
        self.detect_commitments_impl(narrative_id).await
    }

    async fn get_commitment_rhythm(&self, narrative_id: &str) -> Result<Value> {
        self.get_commitment_rhythm_impl(narrative_id).await
    }

    async fn extract_fabula(&self, narrative_id: &str) -> Result<Value> {
        self.extract_fabula_impl(narrative_id).await
    }

    async fn extract_sjuzet(&self, narrative_id: &str) -> Result<Value> {
        self.extract_sjuzet_impl(narrative_id).await
    }

    async fn suggest_reordering(&self, narrative_id: &str) -> Result<Value> {
        self.suggest_reordering_impl(narrative_id).await
    }

    async fn compute_dramatic_irony(&self, narrative_id: &str) -> Result<Value> {
        self.compute_dramatic_irony_impl(narrative_id).await
    }

    async fn detect_focalization(&self, narrative_id: &str) -> Result<Value> {
        self.detect_focalization_impl(narrative_id).await
    }

    async fn detect_character_arc(
        &self,
        narrative_id: &str,
        character_id: Option<&str>,
    ) -> Result<Value> {
        self.detect_character_arc_impl(narrative_id, character_id)
            .await
    }

    async fn detect_subplots(&self, narrative_id: &str) -> Result<Value> {
        self.detect_subplots_impl(narrative_id).await
    }

    async fn classify_scene_sequel(&self, narrative_id: &str) -> Result<Value> {
        self.classify_scene_sequel_impl(narrative_id).await
    }

    #[cfg(feature = "generation")]
    async fn generate_narrative_plan(
        &self,
        premise: &str,
        genre: &str,
        chapter_count: usize,
        subplot_count: usize,
    ) -> Result<Value> {
        self.generate_narrative_plan_impl(premise, genre, chapter_count, subplot_count)
            .await
    }

    #[cfg(feature = "generation")]
    async fn materialize_plan(&self, plan_id: &str) -> Result<Value> {
        self.materialize_plan_impl(plan_id).await
    }

    #[cfg(feature = "generation")]
    async fn validate_materialized_narrative(&self, narrative_id: &str) -> Result<Value> {
        self.validate_materialized_impl(narrative_id).await
    }

    #[cfg(feature = "generation")]
    async fn generate_chapter(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
    ) -> Result<Value> {
        self.generate_chapter_prep_impl(narrative_id, chapter, voice_description)
            .await
    }

    #[cfg(feature = "generation")]
    async fn generate_narrative(
        &self,
        narrative_id: &str,
        chapter_count: usize,
        voice_description: Option<&str>,
    ) -> Result<Value> {
        self.generate_narrative_prep_impl(narrative_id, chapter_count, voice_description)
            .await
    }

    #[cfg(feature = "generation")]
    async fn generate_chapter_with_fitness(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
        style_embedding_id: Option<&str>,
        target_fingerprint_source: Option<&str>,
        fitness_threshold: Option<f64>,
        max_retries: Option<usize>,
        temperature: Option<f64>,
    ) -> Result<Value> {
        use crate::generation::types::{StyleTarget, Threshold};

        // Resolve optional target fingerprint by sourcing it from another narrative.
        let target_fingerprint = match target_fingerprint_source {
            Some(src) => Some(crate::analysis::style_profile::build_fingerprint(
                &self.inner.hypergraph,
                src,
            )?),
            None => None,
        };

        // Parse optional style embedding ID.
        let style_embedding_uuid = match style_embedding_id {
            Some(s) => Some(parse_uuid(s)?),
            None => None,
        };

        // Build StyleTarget with sensible defaults; only override fields the caller specified.
        let mut style = StyleTarget {
            voice_description: voice_description.map(String::from),
            style_embedding_id: style_embedding_uuid,
            target_fingerprint,
            ..StyleTarget::default()
        };
        if let Some(t) = temperature {
            style.temperature = t;
        }
        if let Some(r) = max_retries {
            style.max_retries_per_chapter = r;
        }
        if let Some(thr) = fitness_threshold {
            style.fitness_threshold = Threshold::new(thr)?;
        }

        let parameters = serde_json::json!({
            "narrative_id": narrative_id,
            "chapter": chapter,
            "style": style,
        });

        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::ChapterGenerationFitness,
            target_id: Uuid::now_v7(),
            parameters,
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let job_id = self.inner.job_queue.submit(job)?;
        Ok(serde_json::json!({
            "job_id": job_id,
            "status": "Pending",
        }))
    }

    // ─── Sprint W15: Writer MCP bridge ────────────────────────

    async fn create_annotation(
        &self,
        situation_id: &str,
        kind: &str,
        body: &str,
        span_start: usize,
        span_end: usize,
        source_id: Option<&str>,
        chunk_id: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        self.create_annotation_impl(
            situation_id,
            kind,
            body,
            span_start,
            span_end,
            source_id,
            chunk_id,
            author,
        )
        .await
    }

    async fn list_annotations(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> Result<Value> {
        self.list_annotations_impl(situation_id, narrative_id).await
    }

    async fn update_annotation(&self, annotation_id: &str, patch: Value) -> Result<Value> {
        self.update_annotation_impl(annotation_id, patch).await
    }

    async fn delete_annotation(&self, annotation_id: &str) -> Result<Value> {
        self.delete_annotation_impl(annotation_id).await
    }

    async fn create_collection(
        &self,
        narrative_id: &str,
        name: &str,
        description: Option<&str>,
        query: Value,
    ) -> Result<Value> {
        self.create_collection_impl(narrative_id, name, description, query)
            .await
    }

    async fn list_collections(&self, narrative_id: &str) -> Result<Value> {
        self.list_collections_impl(narrative_id).await
    }

    async fn get_collection(&self, collection_id: &str, resolve: bool) -> Result<Value> {
        self.get_collection_impl(collection_id, resolve).await
    }

    async fn update_collection(&self, collection_id: &str, patch: Value) -> Result<Value> {
        self.update_collection_impl(collection_id, patch).await
    }

    async fn delete_collection(&self, collection_id: &str) -> Result<Value> {
        self.delete_collection_impl(collection_id).await
    }

    async fn create_research_note(
        &self,
        narrative_id: &str,
        situation_id: &str,
        kind: &str,
        body: &str,
        source_id: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        self.create_research_note_impl(narrative_id, situation_id, kind, body, source_id, author)
            .await
    }

    async fn list_research_notes(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> Result<Value> {
        self.list_research_notes_impl(situation_id, narrative_id)
            .await
    }

    async fn get_research_note(&self, note_id: &str) -> Result<Value> {
        self.get_research_note_impl(note_id).await
    }

    async fn update_research_note(&self, note_id: &str, patch: Value) -> Result<Value> {
        self.update_research_note_impl(note_id, patch).await
    }

    async fn delete_research_note(&self, note_id: &str) -> Result<Value> {
        self.delete_research_note_impl(note_id).await
    }

    async fn promote_chunk_to_note(
        &self,
        narrative_id: &str,
        situation_id: &str,
        chunk_id: &str,
        body: &str,
        source_id: Option<&str>,
        kind: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        self.promote_chunk_to_note_impl(
            narrative_id,
            situation_id,
            chunk_id,
            body,
            source_id,
            kind,
            author,
        )
        .await
    }

    async fn propose_edit(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> Result<Value> {
        self.propose_edit_impl(situation_id, instruction, style_preset)
            .await
    }

    async fn apply_edit(&self, proposal: Value, author: Option<&str>) -> Result<Value> {
        self.apply_edit_impl(proposal, author).await
    }

    async fn estimate_edit_tokens(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> Result<Value> {
        self.estimate_edit_tokens_impl(situation_id, instruction, style_preset)
            .await
    }

    async fn commit_narrative_revision(
        &self,
        narrative_id: &str,
        message: &str,
        author: Option<&str>,
    ) -> Result<Value> {
        self.commit_narrative_revision_impl(narrative_id, message, author)
            .await
    }

    async fn diff_narrative_revisions(
        &self,
        narrative_id: &str,
        from_rev: &str,
        to_rev: &str,
    ) -> Result<Value> {
        self.diff_narrative_revisions_impl(narrative_id, from_rev, to_rev)
            .await
    }

    async fn list_workshop_reports(&self, narrative_id: &str) -> Result<Value> {
        self.list_workshop_reports_impl(narrative_id).await
    }

    async fn get_workshop_report(&self, report_id: &str) -> Result<Value> {
        self.get_workshop_report_impl(report_id).await
    }

    async fn list_cost_ledger_entries(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        self.list_cost_ledger_entries_impl(narrative_id, limit)
            .await
    }

    async fn list_compile_profiles(&self, narrative_id: &str) -> Result<Value> {
        self.list_compile_profiles_impl(narrative_id).await
    }

    async fn compile_narrative(
        &self,
        narrative_id: &str,
        format: &str,
        profile_id: Option<&str>,
    ) -> Result<Value> {
        self.compile_narrative_impl(narrative_id, format, profile_id)
            .await
    }

    async fn upsert_compile_profile(
        &self,
        narrative_id: &str,
        profile_id: Option<&str>,
        patch: Value,
    ) -> Result<Value> {
        self.upsert_compile_profile_impl(narrative_id, profile_id, patch)
            .await
    }

    async fn list_narrative_templates(&self) -> Result<Value> {
        self.list_narrative_templates_impl().await
    }

    async fn instantiate_template(
        &self,
        template_id: &str,
        bindings: std::collections::HashMap<String, String>,
    ) -> Result<Value> {
        self.instantiate_template_impl(template_id, bindings).await
    }

    async fn extract_narrative_skeleton(&self, narrative_id: &str) -> Result<Value> {
        self.extract_narrative_skeleton_impl(narrative_id).await
    }

    async fn find_duplicate_candidates(
        &self,
        narrative_id: &str,
        threshold: Option<f64>,
        max_candidates: Option<usize>,
    ) -> Result<Value> {
        self.find_duplicate_candidates_impl(narrative_id, threshold, max_candidates)
            .await
    }

    async fn suggest_narrative_fixes(&self, narrative_id: &str) -> Result<Value> {
        self.suggest_narrative_fixes_impl(narrative_id).await
    }

    async fn apply_narrative_fix(&self, narrative_id: &str, fix: Value) -> Result<Value> {
        self.apply_narrative_fix_impl(narrative_id, fix).await
    }

    async fn apply_reorder(&self, narrative_id: &str, entries: Value) -> Result<Value> {
        self.apply_reorder_impl(narrative_id, entries).await
    }

    // ─── EATH Phase 10 — Synthetic Hypergraph MCP tools ──────────

    async fn calibrate_surrogate(
        &self,
        narrative_id: &str,
        model: Option<&str>,
    ) -> Result<Value> {
        self.calibrate_surrogate_impl(narrative_id, model).await
    }

    async fn generate_synthetic_narrative(
        &self,
        source_narrative_id: &str,
        output_narrative_id: &str,
        model: Option<&str>,
        params: Option<Value>,
        seed: Option<u64>,
        num_steps: Option<usize>,
        label_prefix: Option<&str>,
    ) -> Result<Value> {
        self.generate_synthetic_narrative_impl(
            source_narrative_id,
            output_narrative_id,
            model,
            params,
            seed,
            num_steps,
            label_prefix,
        )
        .await
    }

    async fn generate_hybrid_narrative(
        &self,
        components: Value,
        output_narrative_id: &str,
        seed: Option<u64>,
        num_steps: Option<usize>,
    ) -> Result<Value> {
        self.generate_hybrid_narrative_impl(components, output_narrative_id, seed, num_steps)
            .await
    }

    async fn list_synthetic_runs(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        self.list_synthetic_runs_impl(narrative_id, limit).await
    }

    async fn get_fidelity_report(
        &self,
        narrative_id: &str,
        run_id: &str,
    ) -> Result<Value> {
        self.get_fidelity_report_impl(narrative_id, run_id).await
    }

    async fn compute_pattern_significance(
        &self,
        narrative_id: &str,
        metric: &str,
        k: Option<u16>,
        model: Option<&str>,
        params_override: Option<Value>,
    ) -> Result<Value> {
        self.compute_pattern_significance_impl(narrative_id, metric, k, model, params_override)
            .await
    }

    async fn simulate_higher_order_contagion(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        model: Option<&str>,
    ) -> Result<Value> {
        self.simulate_higher_order_contagion_impl(narrative_id, params, k, model)
            .await
    }

    async fn compute_dual_significance(
        &self,
        narrative_id: &str,
        metric: &str,
        k_per_model: Option<u16>,
        models: Option<Vec<String>>,
    ) -> Result<Value> {
        self.compute_dual_significance_impl(narrative_id, metric, k_per_model, models)
            .await
    }

    async fn compute_bistability_significance(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        models: Option<Vec<String>>,
    ) -> Result<Value> {
        self.compute_bistability_significance_impl(narrative_id, params, k, models)
            .await
    }

    async fn reconstruct_hypergraph(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> Result<Value> {
        self.reconstruct_hypergraph_impl(narrative_id, params).await
    }

    async fn simulate_opinion_dynamics(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> Result<Value> {
        self.simulate_opinion_dynamics_impl(narrative_id, params)
            .await
    }

    async fn simulate_opinion_phase_transition(
        &self,
        narrative_id: &str,
        c_range: [Value; 3],
        base_params: Option<Value>,
    ) -> Result<Value> {
        self.simulate_opinion_phase_transition_impl(narrative_id, c_range, base_params)
            .await
    }

    async fn fuzzy_probability(
        &self,
        narrative_id: &str,
        event: Value,
        distribution: Value,
        tnorm: Option<&str>,
    ) -> Result<Value> {
        self.fuzzy_probability_impl(narrative_id, event, distribution, tnorm)
            .await
    }

    // ─── Fuzzy Sprint Phase 11 — 13 new fuzzy MCP tool delegates ──

    async fn fuzzy_list_tnorms(&self) -> Result<Value> {
        self.fuzzy_list_tnorms_impl().await
    }

    async fn fuzzy_list_aggregators(&self) -> Result<Value> {
        self.fuzzy_list_aggregators_impl().await
    }

    async fn fuzzy_get_config(&self) -> Result<Value> {
        self.fuzzy_get_config_impl().await
    }

    async fn fuzzy_set_config(
        &self,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
        measure: Option<Option<&str>>,
        reset: bool,
    ) -> Result<Value> {
        self.fuzzy_set_config_impl(tnorm, aggregator, measure, reset)
            .await
    }

    async fn fuzzy_create_measure(
        &self,
        name: &str,
        n: u8,
        values: Vec<f64>,
    ) -> Result<Value> {
        self.fuzzy_create_measure_impl(name, n, values).await
    }

    async fn fuzzy_list_measures(&self) -> Result<Value> {
        self.fuzzy_list_measures_impl().await
    }

    async fn fuzzy_aggregate(
        &self,
        xs: Vec<f64>,
        aggregator: &str,
        tnorm: Option<&str>,
        measure: Option<&str>,
        owa_weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> Result<Value> {
        self.fuzzy_aggregate_impl(xs, aggregator, tnorm, measure, owa_weights, seed)
            .await
    }

    async fn fuzzy_allen_gradation(
        &self,
        narrative_id: &str,
        a_id: &str,
        b_id: &str,
    ) -> Result<Value> {
        self.fuzzy_allen_gradation_impl(narrative_id, a_id, b_id)
            .await
    }

    async fn fuzzy_quantify(
        &self,
        narrative_id: &str,
        quantifier: &str,
        entity_type: Option<&str>,
        where_spec: Option<&str>,
        label: Option<&str>,
    ) -> Result<Value> {
        self.fuzzy_quantify_impl(narrative_id, quantifier, entity_type, where_spec, label)
            .await
    }

    async fn fuzzy_verify_syllogism(
        &self,
        narrative_id: &str,
        major: &str,
        minor: &str,
        conclusion: &str,
        threshold: Option<f64>,
        tnorm: Option<&str>,
        figure_hint: Option<&str>,
    ) -> Result<Value> {
        self.fuzzy_verify_syllogism_impl(
            narrative_id,
            major,
            minor,
            conclusion,
            threshold,
            tnorm,
            figure_hint,
        )
        .await
    }

    async fn fuzzy_build_lattice(
        &self,
        narrative_id: &str,
        entity_type: Option<&str>,
        attribute_allowlist: Option<Vec<String>>,
        threshold: Option<usize>,
        tnorm: Option<&str>,
        large_context: bool,
    ) -> Result<Value> {
        self.fuzzy_build_lattice_impl(
            narrative_id,
            entity_type,
            attribute_allowlist,
            threshold,
            tnorm,
            large_context,
        )
        .await
    }

    async fn fuzzy_create_rule(
        &self,
        name: &str,
        narrative_id: &str,
        antecedent: Value,
        consequent: Value,
        tnorm: Option<&str>,
        enabled: Option<bool>,
    ) -> Result<Value> {
        self.fuzzy_create_rule_impl(name, narrative_id, antecedent, consequent, tnorm, enabled)
            .await
    }

    async fn fuzzy_evaluate_rules(
        &self,
        narrative_id: &str,
        entity_id: &str,
        rule_ids: Option<Vec<String>>,
        firing_aggregator: Option<crate::fuzzy::aggregation::AggregatorKind>,
    ) -> Result<Value> {
        self.fuzzy_evaluate_rules_impl(narrative_id, entity_id, rule_ids, firing_aggregator)
            .await
    }

    // ─── Graded Acceptability Sprint Phase 5 — 5 new MCP tools ───

    async fn argumentation_gradual(
        &self,
        narrative_id: &str,
        gradual_semantics: Value,
        tnorm: Option<Value>,
    ) -> Result<Value> {
        self.argumentation_gradual_impl(narrative_id, gradual_semantics, tnorm)
            .await
    }

    async fn fuzzy_learn_measure(
        &self,
        name: &str,
        n: u8,
        dataset: Vec<(Vec<f64>, u32)>,
        dataset_id: &str,
    ) -> Result<Value> {
        self.fuzzy_learn_measure_impl(name, n, dataset, dataset_id)
            .await
    }

    async fn fuzzy_get_measure_version(
        &self,
        name: &str,
        version: Option<u32>,
    ) -> Result<Value> {
        self.fuzzy_get_measure_version_impl(name, version).await
    }

    async fn fuzzy_list_measure_versions(&self, name: &str) -> Result<Value> {
        self.fuzzy_list_measure_versions_impl(name).await
    }

    async fn temporal_ordhorn_closure(&self, network: Value) -> Result<Value> {
        self.temporal_ordhorn_closure_impl(network).await
    }
}

#[cfg(feature = "disinfo")]
fn parse_comparison_kind(s: &str) -> Result<crate::disinfo::ComparisonKind> {
    use crate::disinfo::ComparisonKind;
    match s.to_lowercase().as_str() {
        "behavioral" => Ok(ComparisonKind::Behavioral),
        "disinfo" => Ok(ComparisonKind::Disinfo),
        "narrative" => Ok(ComparisonKind::Narrative),
        other => Err(TensaError::InvalidQuery(format!(
            "unknown comparison kind '{other}' (use 'behavioral' or 'disinfo')"
        ))),
    }
}

#[cfg(feature = "disinfo")]
fn parse_comparison_task(s: &str) -> Result<crate::disinfo::ComparisonTask> {
    use crate::disinfo::ComparisonTask;
    match s.to_lowercase().as_str() {
        "literary" => Ok(ComparisonTask::Literary),
        "cib" => Ok(ComparisonTask::Cib),
        "factory" => Ok(ComparisonTask::Factory),
        other => Err(TensaError::InvalidQuery(format!(
            "unknown comparison task '{other}' (use 'literary', 'cib', or 'factory')"
        ))),
    }
}

#[cfg(feature = "disinfo")]
fn to_value<T: serde::Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value).map_err(|e| TensaError::Serialization(e.to_string()))
}

#[cfg(feature = "disinfo")]
fn parse_beta_overrides_map(
    overrides: Option<std::collections::HashMap<String, f64>>,
) -> Vec<(crate::types::Platform, f64)> {
    crate::analysis::contagion::parse_beta_overrides(overrides.unwrap_or_default())
}

// ─── Helper Functions: Descriptor Row → InferenceJob ────────
//
// These delegate to `crate::inference::dispatch` so the HTTP `POST /infer`
// route and the embedded MCP path use the same mapping table and can't drift.

type ResultRow = HashMap<String, Value>;

fn infer_type_from_row(row: &ResultRow) -> Result<InferenceJobType> {
    crate::inference::dispatch::infer_type_from_row(row)
}

fn extract_target_id(row: &ResultRow) -> Option<Uuid> {
    crate::inference::dispatch::extract_target_id(row)
}

fn extract_parameters(row: &ResultRow) -> Option<Value> {
    crate::inference::dispatch::extract_parameters(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn test_backend() -> EmbeddedBackend {
        let store = Arc::new(MemoryStore::new());
        EmbeddedBackend::from_store(store)
    }

    #[tokio::test]
    async fn test_execute_query_basic() {
        let backend = test_backend();
        let result = backend
            .execute_query("MATCH (e:Actor) RETURN e LIMIT 5")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_and_get_entity() {
        let backend = test_backend();

        let data = serde_json::json!({
            "id": Uuid::now_v7(),
            "entity_type": "Actor",
            "properties": {"name": "Test Actor"},
            "beliefs": null,
            "embedding": null,
            "narrative_id": null,
            "maturity": "Candidate",
            "confidence": 0.9,
            "provenance": [],
            "created_at": chrono::Utc::now(),
            "updated_at": chrono::Utc::now()
        });

        let result = backend.create_entity(data.clone()).await.unwrap();
        let id = result["id"].as_str().unwrap();

        let entity = backend.get_entity(id).await.unwrap();
        assert_eq!(entity["properties"]["name"], "Test Actor");
    }

    #[tokio::test]
    async fn test_list_narratives_empty() {
        let backend = test_backend();
        let result = backend.list_narratives().await.unwrap();
        assert!(result.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_review_queue_list_empty() {
        let backend = test_backend();
        let result = backend
            .review_queue("list", None, None, None, None, None)
            .await
            .unwrap();
        assert!(result.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_review_queue_invalid_action() {
        let backend = test_backend();
        let result = backend
            .review_queue("invalid", None, None, None, None, None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_actor_profile() {
        let backend = test_backend();

        // Create an entity
        let entity_id = Uuid::now_v7();
        let now = chrono::Utc::now();
        let entity_data = serde_json::json!({
            "id": entity_id,
            "entity_type": "Actor",
            "properties": {"name": "Test Actor"},
            "beliefs": null,
            "embedding": null,
            "narrative_id": null,
            "maturity": "Candidate",
            "confidence": 0.9,
            "provenance": [],
            "created_at": now,
            "updated_at": now
        });
        backend.create_entity(entity_data).await.unwrap();

        // Create a situation
        let sit_id = Uuid::now_v7();
        let sit_data = serde_json::json!({
            "id": sit_id,
            "temporal": {
                "start": null,
                "end": null,
                "granularity": "Approximate",
                "relations": [],
            },
            "spatial": null,
            "game_structure": null,
            "causes": [],
            "deterministic": null,
            "probabilistic": null,
            "embedding": null,
            "raw_content": [{"content_type": "Text", "content": "A test situation", "source": null}],
            "narrative_level": "Scene",
            "narrative_id": null,
            "discourse": null,
            "maturity": "Candidate",
            "confidence": 0.7,
            "extraction_method": "HumanEntered",
            "created_at": now,
            "updated_at": now,
        });
        backend.create_situation(sit_data).await.unwrap();

        // Add participant
        let part_data = serde_json::json!({
            "entity_id": entity_id,
            "situation_id": sit_id,
            "role": "Protagonist",
            "info_set": null,
            "action": "investigates",
            "payoff": null,
        });
        backend.add_participant(part_data).await.unwrap();

        // Get actor profile
        let profile = backend
            .get_actor_profile(&entity_id.to_string())
            .await
            .unwrap();
        assert_eq!(profile["entity"]["properties"]["name"], "Test Actor");
        assert_eq!(profile["participation_count"], 1);
        assert!(profile["participations"].as_array().unwrap().len() == 1);
        assert_eq!(profile["participations"][0]["action"], "investigates");
        assert!(profile["state_history"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_actor_profile_not_found() {
        let backend = test_backend();
        let random_id = Uuid::now_v7();
        let result = backend.get_actor_profile(&random_id.to_string()).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_counterfactual_query_parses() {
        use crate::query::parser;
        let q = r#"INFER COUNTERFACTUAL FOR s:Situation ASSUMING s.action = "cooperate" RETURN s"#;
        assert!(parser::parse_query(q).is_ok());
    }

    #[tokio::test]
    async fn test_ingest_url_no_extractor() {
        let backend = test_backend();
        // Without an LLM extractor configured, ingest_url should still fetch but
        // fail at the pipeline stage (no extractor). We use a data URI to avoid network.
        let result = backend
            .ingest_url("http://127.0.0.1:1/nonexistent", "test-narrative", None)
            .await;
        // Should fail — either network error or no extractor
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ingest_rss_no_feature_or_no_extractor() {
        let backend = test_backend();
        let result = backend
            .ingest_rss(
                "http://127.0.0.1:1/nonexistent.rss",
                "test-narrative",
                Some(5),
            )
            .await;
        // Should fail — either feature not enabled or network error
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_narrative_query_parses() {
        use crate::query::parser;
        let q = r#"DISCOVER PATTERNS ACROSS NARRATIVES ("hamlet", "macbeth") RETURN *"#;
        assert!(parser::parse_query(q).is_ok());
    }

    #[tokio::test]
    async fn test_submit_inference_query_creates_job() {
        let backend = test_backend();
        let result = backend
            .submit_inference_query("INFER CAUSES FOR s:Situation RETURN s")
            .await
            .unwrap();
        assert_eq!(result["status"], "Pending");
        assert!(result["job_id"].as_str().is_some());
        assert!(result["message"].as_str().unwrap().contains("submitted"));
    }

    #[tokio::test]
    async fn test_submit_inference_query_centrality() {
        let backend = test_backend();
        let result = backend
            .submit_inference_query("INFER CENTRALITY FOR n:Narrative RETURN n")
            .await
            .unwrap();
        assert_eq!(result["status"], "Pending");
        assert!(result["job_id"].as_str().is_some());

        // Verify the job actually exists in the queue
        let job_id = result["job_id"].as_str().unwrap();
        let status = backend.get_job_status(job_id).await.unwrap();
        assert_eq!(status["status"], "Pending");
        assert_eq!(status["job_type"], "CentralityAnalysis");
    }

    #[tokio::test]
    async fn test_submit_inference_query_style() {
        let backend = test_backend();
        let result = backend
            .submit_inference_query("INFER STYLE FOR n:Narrative RETURN n")
            .await
            .unwrap();
        assert_eq!(result["status"], "Pending");

        let job_id = result["job_id"].as_str().unwrap();
        let status = backend.get_job_status(job_id).await.unwrap();
        assert_eq!(status["job_type"], "StyleProfile");
    }

    #[tokio::test]
    async fn test_submit_discover_query_creates_job() {
        let backend = test_backend();
        let result = backend
            .submit_inference_query("DISCOVER PATTERNS RETURN *")
            .await
            .unwrap();
        assert_eq!(result["status"], "Pending");
        assert!(result["job_id"].as_str().is_some());

        let job_id = result["job_id"].as_str().unwrap();
        let status = backend.get_job_status(job_id).await.unwrap();
        assert_eq!(status["job_type"], "PatternMining");
    }

    #[tokio::test]
    async fn test_submit_inference_query_different_types() {
        let backend = test_backend();
        let r1 = backend
            .submit_inference_query("INFER CAUSES FOR s:Situation RETURN s")
            .await
            .unwrap();
        let r2 = backend
            .submit_inference_query("INFER MOTIVATION FOR e:Actor RETURN e")
            .await
            .unwrap();
        // Different infer types produce different jobs
        assert_ne!(r1["job_id"], r2["job_id"]);
    }

    #[test]
    fn test_infer_type_from_row_all_variants() {
        let cases = vec![
            ("Causes", InferenceJobType::CausalDiscovery),
            ("Motivation", InferenceJobType::MotivationInference),
            ("Game", InferenceJobType::GameClassification),
            ("Counterfactual", InferenceJobType::Counterfactual),
            ("Missing", InferenceJobType::MissingLinks),
            ("Anomalies", InferenceJobType::AnomalyDetection),
            ("Centrality", InferenceJobType::CentralityAnalysis),
            ("Entropy", InferenceJobType::EntropyAnalysis),
            ("Beliefs", InferenceJobType::BeliefModeling),
            ("Evidence", InferenceJobType::EvidenceCombination),
            ("Arguments", InferenceJobType::ArgumentationAnalysis),
            ("Contagion", InferenceJobType::ContagionAnalysis),
            ("Style", InferenceJobType::StyleProfile),
            ("StyleCompare", InferenceJobType::StyleComparison),
            ("StyleAnomalies", InferenceJobType::StyleAnomaly),
            ("VerifyAuthorship", InferenceJobType::AuthorshipVerification),
        ];
        for (name, expected) in cases {
            let mut row = HashMap::new();
            row.insert("_infer_type".into(), serde_json::json!(name));
            assert_eq!(
                infer_type_from_row(&row).unwrap(),
                expected,
                "Failed for infer type: {}",
                name
            );
        }
    }

    #[test]
    fn test_discover_type_from_row_all_variants() {
        let cases = vec![
            ("Patterns", InferenceJobType::PatternMining),
            ("Arcs", InferenceJobType::ArcClassification),
            ("Missing", InferenceJobType::MissingEventPrediction),
        ];
        for (name, expected) in cases {
            let mut row = HashMap::new();
            row.insert("_discover_type".into(), serde_json::json!(name));
            assert_eq!(
                infer_type_from_row(&row).unwrap(),
                expected,
                "Failed for discover type: {}",
                name
            );
        }
    }

    #[test]
    fn test_infer_type_from_row_unknown() {
        let mut row = HashMap::new();
        row.insert("_infer_type".into(), serde_json::json!("Nonexistent"));
        assert!(infer_type_from_row(&row).is_err());
    }

    #[test]
    fn test_infer_type_from_row_empty() {
        let row = HashMap::new();
        assert!(infer_type_from_row(&row).is_err());
    }

    #[test]
    fn test_infer_type_from_row_new_types() {
        let new_cases = vec![
            ("TemporalRules", InferenceJobType::TemporalILP),
            ("MeanField", InferenceJobType::MeanFieldGame),
            ("Psl", InferenceJobType::ProbabilisticSoftLogic),
        ];
        for (name, expected) in new_cases {
            let mut row = HashMap::new();
            row.insert("_infer_type".into(), serde_json::json!(name));
            assert_eq!(
                infer_type_from_row(&row).unwrap(),
                expected,
                "Failed for new infer type: {}",
                name
            );
        }
    }

    #[test]
    fn test_parse_new_infer_types() {
        use crate::query::parser::{self, InferType};
        let cases = vec![
            (
                "INFER CENTRALITY FOR n:Narrative RETURN n",
                InferType::Centrality,
            ),
            ("INFER ENTROPY FOR n:Narrative RETURN n", InferType::Entropy),
            ("INFER BELIEFS FOR n:Narrative RETURN n", InferType::Beliefs),
            (
                "INFER EVIDENCE FOR n:Narrative RETURN n",
                InferType::Evidence,
            ),
            (
                "INFER ARGUMENTS FOR n:Narrative RETURN n",
                InferType::Arguments,
            ),
            (
                "INFER CONTAGION FOR n:Narrative RETURN n",
                InferType::Contagion,
            ),
        ];
        for (query, expected) in cases {
            let q = parser::parse_query(query).unwrap();
            assert_eq!(
                q.infer_clause.unwrap().infer_type,
                expected,
                "Failed for query: {}",
                query
            );
        }
    }
}
