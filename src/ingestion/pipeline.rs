//! End-to-end ingestion pipeline orchestration.
//!
//! Coordinates chunking, LLM extraction, entity resolution,
//! confidence gating, and embedding in a streaming pipeline.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ingestion::jobs::{IngestionPhase, IngestionProgress};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::chunker::{chunk_hash, chunk_text, ChunkerConfig};
use crate::ingestion::embed::EmbeddingProvider;
use crate::ingestion::extraction::{repair_extraction, validate_extraction, NarrativeExtraction};
use crate::ingestion::gate::{ConfidenceGate, GateDecision, GateReport};
use crate::ingestion::llm::{NarrativeExtractor, RawLlmExchange};
use crate::ingestion::queue::{
    QueueItemStatus, QueueItemType, ValidationQueue, ValidationQueueItem,
};
use crate::ingestion::resolve::{EntityResolver, ResolveResult};
use crate::ingestion::vector::VectorIndex;
use crate::types::*;

/// Extraction cascade mode controlling whether NLP pre-pass runs before LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CascadeMode {
    /// LLM-only extraction (current default behavior).
    #[default]
    LlmOnly,
    /// Run NLP fast extraction first; skip LLM if NLP yields enough entities.
    NlpFirst,
    /// NLP-only extraction (no LLM calls at all).
    NlpOnly,
}

/// Minimum entities from NLP to skip the LLM call in `NlpFirst` mode.
const NLP_SKIP_LLM_ENTITY_THRESHOLD: usize = 3;

/// Known biographical / hard-fact property keys that the pipeline merges
/// from an `ExtractedEntity.properties` payload into the stored entity.
/// First-write-wins: existing values are never overwritten.
///
/// Explicit whitelist (not a blanket merge) so the LLM cannot silently
/// rewrite arbitrary fields when the same entity is resolved on a later chunk.
pub const BIOGRAPHICAL_KEYS: &[&str] = &[
    "date_of_birth",
    "place_of_birth",
    "date_of_death",
    "place_of_death",
    "nationality",
    "occupation",
    "gender",
    "description",
    "title",
];

/// Merge `name`, `aliases`, and known biographical keys into a properties
/// object. Existing scalar values win; aliases are set-unioned.
fn merge_into_properties(
    props: &mut serde_json::Value,
    name: &str,
    aliases: &[String],
    incoming: &serde_json::Value,
) {
    use crate::ingestion::resolve::normalize_name;

    if !props.is_object() {
        *props = serde_json::json!({});
    }
    let obj = props.as_object_mut().expect("just ensured object");

    obj.entry("name".to_string())
        .or_insert_with(|| serde_json::Value::String(name.to_string()));

    // Filter against the stored canonical name, not `name`, so a later
    // mention's name (e.g. "Count Dracula" on a stored "Dracula") can still
    // become an alias.
    let canonical_key = obj
        .get("name")
        .and_then(|v| v.as_str())
        .map(normalize_name)
        .unwrap_or_default();

    let existing_arr = obj.get("aliases").and_then(|v| v.as_array());
    if existing_arr.is_some() || !aliases.is_empty() {
        let capacity = existing_arr.map_or(0, |a| a.len()) + aliases.len();
        let mut seen: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(capacity);
        let mut merged: Vec<serde_json::Value> = Vec::with_capacity(capacity);
        if let Some(existing) = existing_arr {
            for v in existing {
                if let Some(s) = v.as_str() {
                    let key = normalize_name(s);
                    if !key.is_empty() && key != canonical_key && seen.insert(key) {
                        merged.push(serde_json::Value::String(s.to_string()));
                    }
                }
            }
        }
        for a in aliases {
            let key = normalize_name(a);
            if key.is_empty() || key == canonical_key {
                continue;
            }
            if seen.insert(key) {
                merged.push(serde_json::Value::String(a.clone()));
            }
        }
        if !merged.is_empty() {
            obj.insert("aliases".to_string(), serde_json::Value::Array(merged));
        }
    }

    if let Some(incoming_obj) = incoming.as_object() {
        for key in BIOGRAPHICAL_KEYS {
            if obj.contains_key(*key) {
                continue;
            }
            if let Some(v) = incoming_obj.get(*key) {
                if !v.is_null() {
                    obj.insert((*key).to_string(), v.clone());
                }
            }
        }
    }
}

/// True if `incoming` carries any non-null biographical key the pipeline
/// would merge. Used to short-circuit a KV round-trip when a resolve-hit has
/// nothing new to contribute.
fn has_mergeable_bio_keys(incoming: &serde_json::Value) -> bool {
    match incoming.as_object() {
        Some(obj) => BIOGRAPHICAL_KEYS
            .iter()
            .any(|k| obj.get(*k).map(|v| !v.is_null()).unwrap_or(false)),
        None => false,
    }
}

/// Strip publisher boilerplate from text (Standard Ebooks colophon, Project Gutenberg
/// headers, imprint/uncopyright sections). Keeps only the actual narrative content.
///
/// Strategy: detect the first chapter boundary using the chunker's detect_chapters,
/// strip everything before it that looks like front-matter, and strip known
/// back-matter patterns (Colophon, Uncopyright, Endnotes, etc.) from the end.
pub fn strip_boilerplate(text: &str) -> String {
    use crate::ingestion::chunker::detect_chapters;
    use regex::Regex;

    let mut start = 0;
    let mut end = text.len();

    // Try to find the first chapter marker
    let chapters = detect_chapters(text, None);
    if let Some((first_offset, _)) = chapters.first() {
        start = *first_offset;
    }

    // Strip trailing boilerplate: Colophon, Uncopyright, Endnotes, etc.
    // These are common in Standard Ebooks and Project Gutenberg texts.
    if let Ok(re) = Regex::new(
        r"(?mi)^(?:Colophon|Uncopyright|Imprint|End of (?:the )?Project Gutenberg|Endnotes?|About the Author)\s*$",
    ) {
        if let Some(m) = re.find(&text[start..]) {
            let candidate = start + m.start();
            // Only strip if it's in the last 10% of the text (avoid false matches)
            if candidate > start + (end - start) * 9 / 10 {
                end = candidate;
            }
        }
    }

    text[start..end].to_string()
}

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub chunker: ChunkerConfig,
    pub auto_commit_threshold: f32,
    pub review_threshold: f32,
    pub source_id: String,
    pub source_type: String,
    /// Narrative ID for incremental re-ingestion (chunk hash tracking).
    #[serde(default)]
    pub narrative_id: Option<String>,
    /// Ingestion job ID (set by the API handler for chunk provenance).
    #[serde(default)]
    pub job_id: Option<String>,
    /// Number of chunks to extract concurrently (1 = sequential).
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Strip publisher boilerplate (Standard Ebooks, Gutenberg headers) before chunking.
    #[serde(default = "default_strip_boilerplate")]
    pub strip_boilerplate: bool,
    /// Enable step-2 enrichment pass (beliefs, game structures, discourse, info sets).
    /// Doubles LLM calls per chunk but produces much richer data.
    #[serde(default)]
    pub enrich: bool,
    /// Use SingleSession mode (entire text in one LLM context, per-chunk extraction turns).
    #[serde(default)]
    pub single_session: bool,
    /// Maximum context window tokens for SingleSession mode (0 = 1M default).
    #[serde(default)]
    pub session_max_context_tokens: usize,
    /// Enable verbose debug logging for this pipeline run.
    #[serde(default)]
    pub debug: bool,
    /// Extraction cascade mode: LlmOnly (default), NlpFirst, or NlpOnly.
    #[serde(default)]
    pub cascade_mode: CascadeMode,
    /// Fuzzy Sprint Phase 9 — persisted Mamdani rule id to apply to
    /// freshly-ingested entities after commit. `None` (default)
    /// preserves the pre-Phase-9 pipeline bit-identically.
    ///
    /// Cites: [mamdani1975mamdani].
    #[serde(default)]
    pub post_ingest_mamdani_rule_id: Option<String>,
}

fn default_strip_boilerplate() -> bool {
    true
}

fn default_concurrency() -> usize {
    1
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunker: ChunkerConfig::default(),
            auto_commit_threshold: 0.8,
            review_threshold: 0.3,
            source_id: "unknown".to_string(),
            source_type: "text".to_string(),
            narrative_id: None,
            job_id: None,
            concurrency: 1,
            strip_boilerplate: true,
            enrich: true,
            single_session: false,
            session_max_context_tokens: 0,
            debug: false,
            cascade_mode: CascadeMode::default(),
            post_ingest_mamdani_rule_id: None,
        }
    }
}

/// Report of an ingestion run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestionReport {
    pub chunks_processed: usize,
    pub chunks_skipped: usize,
    pub entities_created: usize,
    pub situations_created: usize,
    pub participations_created: usize,
    pub causal_links_created: usize,
    pub items_auto_committed: usize,
    pub items_queued: usize,
    pub items_rejected: usize,
    pub entity_resolutions: usize,
    pub errors: Vec<String>,
    pub duration_secs: f64,
    /// IDs of entities created during this run (for rollback).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub created_entity_ids: Vec<Uuid>,
    /// IDs of situations created during this run (for rollback).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub created_situation_ids: Vec<Uuid>,
    /// IDs of chunks stored during this run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub created_chunk_ids: Vec<Uuid>,
    /// Whether the pipeline was cancelled mid-run.
    #[serde(default)]
    pub cancelled: bool,
    /// Session reconciliation result (SingleSession mode only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_reconciliation: Option<Box<crate::ingestion::extraction::SessionReconciliation>>,
}

/// Run cascading extraction: NLP first (if configured), then LLM if needed.
///
/// Returns `(extraction, exchange)` — exchange is `None` for NLP-only results.
fn cascade_extract(
    extractor: &dyn NarrativeExtractor,
    chunk: &crate::ingestion::chunker::TextChunk,
    known_entities: &[String],
    mode: &CascadeMode,
) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
    match mode {
        CascadeMode::LlmOnly => extractor.extract_with_logging(chunk, known_entities),
        CascadeMode::NlpOnly => {
            let nlp = crate::ingestion::nlp_extract::NlpFastExtractor::new();
            let extraction = nlp.extract_with_context(chunk, known_entities)?;
            Ok((extraction, None))
        }
        CascadeMode::NlpFirst => {
            // Phase 1: Run NLP fast extraction
            let nlp = crate::ingestion::nlp_extract::NlpFastExtractor::new();
            let nlp_result = nlp.extract_with_context(chunk, known_entities)?;

            // If NLP found enough entities, use its result and skip the LLM call
            if nlp_result.entities.len() >= NLP_SKIP_LLM_ENTITY_THRESHOLD {
                tracing::debug!(
                    "NlpFirst: NLP found {} entities (>= {}), skipping LLM for chunk {}",
                    nlp_result.entities.len(),
                    NLP_SKIP_LLM_ENTITY_THRESHOLD,
                    chunk.chunk_id,
                );
                return Ok((nlp_result, None));
            }

            // Phase 2: NLP was sparse, fall through to LLM
            tracing::debug!(
                "NlpFirst: NLP found only {} entities, falling through to LLM for chunk {}",
                nlp_result.entities.len(),
                chunk.chunk_id,
            );
            let (llm_result, exchange) = extractor.extract_with_logging(chunk, known_entities)?;

            // Merge: add any NLP-only entities that the LLM missed
            let merged = merge_extractions(nlp_result, llm_result);
            Ok((merged, exchange))
        }
    }
}

/// Merge NLP and LLM extractions, deduplicating entities by normalized name.
/// LLM results take priority (higher confidence). NLP-only entities are appended.
fn merge_extractions(nlp: NarrativeExtraction, llm: NarrativeExtraction) -> NarrativeExtraction {
    use std::collections::HashSet;

    let llm_names: HashSet<String> = llm
        .entities
        .iter()
        .map(|e| e.name.trim().to_lowercase())
        .collect();

    let mut merged_entities = llm.entities;
    for nlp_entity in nlp.entities {
        let normalized = nlp_entity.name.trim().to_lowercase();
        if !llm_names.contains(&normalized) {
            merged_entities.push(nlp_entity);
        }
    }

    NarrativeExtraction {
        entities: merged_entities,
        situations: llm.situations,
        participations: llm.participations,
        causal_links: llm.causal_links,
        temporal_relations: llm.temporal_relations,
    }
}

/// The ingestion pipeline, wiring together all Phase 1 components.
pub struct IngestionPipeline {
    hypergraph: Arc<Hypergraph>,
    extractor: Arc<dyn NarrativeExtractor>,
    pass2_extractor: Option<Arc<dyn NarrativeExtractor>>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    vector_index: Option<Arc<std::sync::RwLock<VectorIndex>>>,
    gate: ConfidenceGate,
    queue: Arc<ValidationQueue>,
    config: PipelineConfig,
    progress: Option<Arc<Mutex<IngestionProgress>>>,
    cancelled: Arc<AtomicBool>,
    job_queue: Option<Arc<crate::ingestion::jobs::IngestionJobQueue>>,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline.
    pub fn new(
        hypergraph: Arc<Hypergraph>,
        extractor: Arc<dyn NarrativeExtractor>,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        vector_index: Option<Arc<std::sync::RwLock<VectorIndex>>>,
        queue: Arc<ValidationQueue>,
        config: PipelineConfig,
    ) -> Self {
        let gate = ConfidenceGate::new(config.auto_commit_threshold, config.review_threshold);
        Self {
            hypergraph,
            extractor,
            pass2_extractor: None,
            embedder,
            vector_index,
            gate,
            queue,
            config,
            progress: None,
            cancelled: Arc::new(AtomicBool::new(false)),
            job_queue: None,
        }
    }

    /// Attach a job queue for storing chunk manifests.
    pub fn with_job_queue(mut self, jq: Arc<crate::ingestion::jobs::IngestionJobQueue>) -> Self {
        self.job_queue = Some(jq);
        self
    }

    /// Set the pass 2 (reconciliation) extractor for multi-pass ingestion.
    pub fn with_pass2_extractor(mut self, extractor: Arc<dyn NarrativeExtractor>) -> Self {
        self.pass2_extractor = Some(extractor);
        self
    }

    /// Attach a progress tracker for live status updates.
    pub fn with_progress(mut self, progress: Arc<Mutex<IngestionProgress>>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Attach a cancellation flag. Set to `true` to stop the pipeline.
    pub fn with_cancel_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancelled = flag;
        self
    }

    /// Check if cancellation was requested.
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Persist an LLM call log entry if a job queue is available.
    /// Return job queue + job ID if both are available, None otherwise.
    fn job_context(&self) -> Option<(&crate::ingestion::jobs::IngestionJobQueue, String)> {
        let jq = self.job_queue.as_ref()?;
        let job_id = self.config.job_id.clone()?;
        Some((jq, job_id))
    }

    fn store_llm_log(
        &self,
        chunk_index: usize,
        pass: u8,
        attempt: u8,
        exchange: &Option<RawLlmExchange>,
        extraction: Option<&NarrativeExtraction>,
        error: Option<&str>,
    ) {
        let (jq, job_id) = match self.job_context() {
            Some(ctx) => ctx,
            None => return,
        };
        if let Some(ex) = exchange {
            let log = crate::ingestion::jobs::LlmCallLog {
                job_id: job_id.clone(),
                chunk_index,
                pass,
                attempt,
                system_prompt: ex.system_prompt.clone(),
                user_prompt: ex.user_prompt.clone(),
                raw_response: ex.raw_response.clone(),
                parsed_extraction: extraction.cloned(),
                parse_error: ex
                    .parse_error
                    .clone()
                    .or_else(|| error.map(|s| s.to_string())),
                retry_prompt: ex.retry_prompt.clone(),
                retry_response: ex.retry_response.clone(),
                duration_ms: ex.duration_ms,
                model: ex.model.clone(),
                endpoint: ex.endpoint.clone(),
                timestamp: Utc::now(),
            };
            if let Err(e) = jq.store_llm_log(&log) {
                tracing::warn!(
                    "Failed to store LLM log for job={} chunk={} pass={} attempt={}: {}",
                    job_id,
                    chunk_index,
                    pass,
                    attempt,
                    e
                );
            }
        } else {
            tracing::debug!("store_llm_log: exchange is None for chunk={} pass={} attempt={} — extractor did not return logging data", chunk_index, pass, attempt);
        }
    }

    /// Persist a chunk extraction record if a job queue is available.
    fn store_chunk_extraction_record(
        &self,
        chunk_index: usize,
        extraction: &NarrativeExtraction,
        gate_decisions: Vec<crate::ingestion::jobs::GateDecisionEntry>,
        entity_map: &std::collections::HashMap<String, Uuid>,
        situation_ids: &[Uuid],
    ) {
        let (jq, job_id) = match self.job_context() {
            Some(ctx) => ctx,
            None => return,
        };
        let record = crate::ingestion::jobs::ChunkExtractionRecord {
            job_id,
            chunk_index,
            extraction: extraction.clone(),
            gate_decisions,
            entity_map: entity_map.clone(),
            situation_ids: situation_ids.to_vec(),
        };
        let _ = jq.store_chunk_extraction(&record);
    }

    /// Update the progress tracker (no-op if none attached).
    fn update_progress(&self, f: impl FnOnce(&mut IngestionProgress)) {
        if let Some(ref p) = self.progress {
            if let Ok(mut guard) = p.lock() {
                f(&mut guard);
            }
        }
    }

    /// Seed `resolver` with entities already in the hypergraph that belong to the
    /// configured narrative (and its project siblings if the narrative belongs to
    /// a project). Prevents cross-run duplicates such as re-ingesting the same
    /// source producing a new copy of "Ana Stojanović" on every run.
    ///
    /// Best-effort: no `narrative_id` → no-op. KV / registry errors are swallowed
    /// via `if let Ok(...)` and will only cost dedup coverage, never abort the
    /// ingestion run.
    fn bootstrap_resolver(&self, resolver: &mut EntityResolver) -> usize {
        let Some(ref nid) = self.config.narrative_id else {
            return 0;
        };
        let store = self.hypergraph.store_arc();
        let mut narrative_ids: Vec<String> = vec![nid.clone()];

        let registry = crate::narrative::registry::NarrativeRegistry::new(store.clone());
        if let Ok(current) = registry.get(nid) {
            if let Some(ref pid) = current.project_id {
                let projects = crate::narrative::project::ProjectRegistry::new(store);
                if let Ok(sibling_ids) = projects.list_narrative_ids(pid) {
                    for sid in sibling_ids {
                        if sid != *nid {
                            narrative_ids.push(sid);
                        }
                    }
                }
            }
        }

        let mut total = 0;
        for nar_id in &narrative_ids {
            if let Ok(ents) = self.hypergraph.list_entities_by_narrative(nar_id) {
                total += resolver.bootstrap_from_entities(&ents);
            }
        }
        if total > 0 {
            tracing::info!(
                narrative_id = %nid,
                narratives = narrative_ids.len(),
                bootstrapped = total,
                "Bootstrapped entity resolver from existing store",
            );
        }
        total
    }

    /// Fuzzy Sprint Phase 9 — apply the configured post-ingest Mamdani
    /// rule (if any) to a set of freshly-ingested entities. Called
    /// after a pipeline run commits its entities when
    /// `post_ingest_mamdani_rule_id` is set. Best-effort: rule-load
    /// failures log a warning and skip; the ingestion run itself is
    /// never aborted. Attaches `properties.mamdani` =
    /// `{rule_id, firing_strength, linguistic_term, defuzzified_output}`
    /// on success.
    ///
    /// Cites: [mamdani1975mamdani].
    pub fn apply_post_ingest_mamdani(
        &self,
        narrative_id: &str,
        entity_ids: &[Uuid],
    ) -> Result<usize> {
        let Some(ref rule_id) = self.config.post_ingest_mamdani_rule_id else {
            return Ok(0);
        };
        let uuid: Uuid = match rule_id.parse() {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(
                    rule_id = %rule_id,
                    "post_ingest_mamdani_rule_id is not a UUID ({e}); skipping"
                );
                return Ok(0);
            }
        };
        let store = self.hypergraph.store();
        let rule = match crate::fuzzy::rules::load_rule_fn(store, narrative_id, &uuid)? {
            Some(r) => r,
            None => match crate::fuzzy::rules::find_rule_any_narrative(store, &uuid)? {
                Some(r) => r,
                None => {
                    tracing::warn!(
                        rule_id = %uuid,
                        narrative_id = %narrative_id,
                        "post_ingest Mamdani rule not persisted; skipping"
                    );
                    return Ok(0);
                }
            },
        };
        let mut tagged = 0usize;
        for eid in entity_ids {
            let entity_snapshot = match self.hypergraph.get_entity(eid) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let ev = crate::fuzzy::rules::evaluate_single_rule(&rule, &entity_snapshot);
            let strength = ev
                .fired_rules
                .first()
                .map(|f| f.firing_strength)
                .unwrap_or(0.0);
            let rule_name = rule.name.clone();
            let linguistic_term = rule.consequent.linguistic_term.clone();
            let defuzzified = ev.defuzzified_output;
            // Best-effort write-back — use the no-snapshot variant so
            // we don't accumulate a StateVersion per ingested entity for
            // a derived-property annotation.
            let res = self.hypergraph.update_entity_no_snapshot(eid, |entity| {
                if let Some(obj) = entity.properties.as_object_mut() {
                    obj.insert(
                        "mamdani".into(),
                        serde_json::json!({
                            "rule_id": uuid,
                            "rule_name": rule_name,
                            "firing_strength": strength,
                            "linguistic_term": linguistic_term,
                            "defuzzified_output": defuzzified,
                        }),
                    );
                }
            });
            if res.is_ok() {
                tagged += 1;
            }
        }
        Ok(tagged)
    }

    /// Ingest raw text through the full pipeline.
    pub fn ingest_text(&self, text: &str, source_name: &str) -> Result<IngestionReport> {
        let start = std::time::Instant::now();
        let mut report = IngestionReport::default();
        let mut resolver = EntityResolver::new();
        self.bootstrap_resolver(&mut resolver);

        // Stage 0: Strip publisher boilerplate if enabled
        let cleaned;
        let input = if self.config.strip_boilerplate {
            cleaned = strip_boilerplate(text);
            cleaned.as_str()
        } else {
            text
        };

        // Stage 1: Chunk
        self.update_progress(|p| p.phase = IngestionPhase::Chunking);
        let chunks = chunk_text(input, &self.config.chunker);
        report.chunks_processed = chunks.len();
        self.update_progress(|p| {
            p.total_chunks = chunks.len();
            p.phase = IngestionPhase::Extracting;
        });

        if chunks.is_empty() {
            report.duration_secs = start.elapsed().as_secs_f64();
            return Ok(report);
        }

        // Dispatch to SingleSession mode if configured and extractor supports it
        let extractor_model = self
            .extractor
            .model_name()
            .unwrap_or_else(|| "unknown".into());
        let extractor_supports_session = self.extractor.as_session().is_some();
        tracing::info!(
            single_session = self.config.single_session,
            extractor_model = %extractor_model,
            extractor_supports_session = extractor_supports_session,
            session_max_ctx = self.config.session_max_context_tokens,
            chunks = chunks.len(),
            "Pipeline dispatch: evaluating mode"
        );
        self.update_progress(|p| {
            p.model = Some(extractor_model.clone());
        });

        if self.config.single_session {
            if let Some(session_ext) = self.extractor.as_session() {
                let ctx_limit = if self.config.session_max_context_tokens > 0 {
                    self.config.session_max_context_tokens
                } else {
                    1_000_000 // default 1M
                };
                let text_tokens = crate::ingestion::chunker::estimate_tokens(input);
                let estimated_total = text_tokens + chunks.len() * 3000;
                tracing::info!(
                    text_tokens = text_tokens,
                    estimated_total = estimated_total,
                    ctx_limit = ctx_limit,
                    "SingleSession: token budget check"
                );
                if estimated_total <= ctx_limit {
                    tracing::info!(
                        "SingleSession: dispatching to session pipeline (model={})",
                        extractor_model
                    );
                    self.update_progress(|p| {
                        p.session_status = Some("Starting SingleSession...".into());
                    });
                    return self.ingest_text_single_session(
                        input,
                        source_name,
                        &chunks,
                        session_ext,
                        ctx_limit,
                        start,
                    );
                }
                tracing::warn!(
                    "SingleSession: estimated {} tokens exceeds context limit {}. Falling back to Single mode.",
                    estimated_total, ctx_limit
                );
                self.update_progress(|p| {
                    p.session_status = Some("Fallback: token limit exceeded".into());
                });
            } else {
                tracing::warn!(
                    "SingleSession requested but extractor ({}) does not support sessions. Falling back to Single mode.",
                    extractor_model
                );
                self.update_progress(|p| {
                    p.session_status = Some("Fallback: extractor not session-capable".into());
                });
            }
        }

        // Propagate cancel flag to LLM extractor so in-flight HTTP requests abort
        self.extractor.set_cancel_flag(self.cancelled.clone());
        if let Some(ref p2) = self.pass2_extractor {
            p2.set_cancel_flag(self.cancelled.clone());
        }

        // Stage 2: Extract from each chunk (with incremental skip via hash)
        //
        // Phase A: Compute hashes, skip already-ingested, init per-chunk progress.
        // Phase B: Extract pending chunks (sequential or concurrent via LLM).
        // Phase C: Process results in order — validate, store chunks, collect extractions.

        use crate::ingestion::jobs::{ChunkProgress, ChunkStatus};

        // Phase A: Hash + skip check + initialize chunk progress
        struct PendingChunk {
            chunk_i: usize,
            hash: String,
        }
        let mut pending: Vec<PendingChunk> = Vec::new();
        let mut chunk_progress: Vec<ChunkProgress> = Vec::new();
        for (chunk_i, chunk) in chunks.iter().enumerate() {
            let hash = chunk_hash(&chunk.text);
            if let Some(ref nid) = self.config.narrative_id {
                if self.hypergraph.chunk_exists_by_hash(nid, &hash)?.is_some() {
                    report.chunks_skipped += 1;
                    chunk_progress.push(ChunkProgress {
                        index: chunk_i,
                        status: ChunkStatus::Skipped,
                        attempts: 0,
                        started_ms: None,
                        finished_ms: None,
                        error: None,
                        entities: 0,
                        situations: 0,
                    });
                    continue;
                }
            }
            chunk_progress.push(ChunkProgress {
                index: chunk_i,
                status: ChunkStatus::Pending,
                attempts: 0,
                started_ms: None,
                finished_ms: None,
                error: None,
                entities: 0,
                situations: 0,
            });
            pending.push(PendingChunk { chunk_i, hash });
        }

        // Store chunk manifest (byte positions) before extraction so retries
        // can reconstruct chunks from source text without re-chunking.
        if let (Some(ref jq), Some(ref job_id)) = (&self.job_queue, &self.config.job_id) {
            use crate::ingestion::jobs::ChunkManifestEntry;
            let manifest: Vec<ChunkManifestEntry> = chunks
                .iter()
                .enumerate()
                .map(|(i, c)| ChunkManifestEntry {
                    chunk_index: i,
                    start: c.start_offset,
                    end: c.end_offset,
                    overlap_bytes: c.overlap_prefix.len(),
                    chapter: c.chapter.clone(),
                    content_hash: chunk_hash(&c.text),
                })
                .collect();
            let _ = jq.store_chunk_manifest(job_id, &manifest);
        }

        // Populate initial progress with concurrency info, model, and chunk grid
        let concurrency = self.config.concurrency.max(1);
        self.update_progress(|p| {
            p.concurrency = concurrency;
            p.model = self.extractor.model_name();
            p.chunks = chunk_progress.clone();
        });

        // Helper: update a single chunk's status in the shared progress
        fn update_chunk_progress(
            progress: &Option<Arc<Mutex<IngestionProgress>>>,
            chunk_i: usize,
            f: impl FnOnce(&mut ChunkProgress),
        ) {
            if let Some(ref p) = progress {
                if let Ok(mut guard) = p.lock() {
                    if let Some(cp) = guard.chunks.get_mut(chunk_i) {
                        f(cp);
                    }
                }
            }
        }

        // Phase B: LLM extraction (sequential with context accumulation, or concurrent)
        let active_workers = Arc::new(AtomicUsize::new(0));
        let use_context =
            self.config.chunker.strategy == crate::ingestion::chunker::ChunkStrategy::Chapter;
        // Results slot per pending chunk: (chunk_i, hash, Ok(extraction) | Err(msg))
        let extraction_results: Vec<(
            usize,
            String,
            std::result::Result<NarrativeExtraction, String>,
        )> = if concurrency <= 1 {
            // Sequential path — accumulates known entities across chunks for coherence.
            // Each chunk gets up to MAX_CHUNK_RETRIES attempts with backoff on rate limits.
            const MAX_CHUNK_RETRIES: u32 = 3;
            let mut results = Vec::with_capacity(pending.len());
            let mut known_entities: Vec<String> = Vec::new();
            for pc in &pending {
                if self.is_cancelled() {
                    report.cancelled = true;
                    report.duration_secs = start.elapsed().as_secs_f64();
                    return Ok(report);
                }

                self.update_progress(|p| {
                    p.current_chunk = pc.chunk_i + 1;
                    p.active_workers = 1;
                });

                // Try extraction with inline retry on rate-limit errors
                let mut last_result = None;
                for attempt in 0..=MAX_CHUNK_RETRIES {
                    if self.is_cancelled() {
                        break;
                    }

                    let now_ms = Utc::now().timestamp_millis();
                    active_workers.store(1, Ordering::Relaxed);
                    update_chunk_progress(&self.progress, pc.chunk_i, |cp| {
                        cp.status = ChunkStatus::Extracting;
                        cp.started_ms = Some(now_ms);
                        cp.attempts += 1;
                        cp.error = None;
                    });

                    let entities_ref = if use_context {
                        &known_entities[..]
                    } else {
                        &[]
                    };
                    match self
                        .extractor
                        .extract_with_logging(&chunks[pc.chunk_i], entities_ref)
                    {
                        Ok((ext, exchange)) => {
                            // Persist LLM call log
                            self.store_llm_log(
                                pc.chunk_i,
                                1,
                                attempt as u8,
                                &exchange,
                                Some(&ext),
                                None,
                            );
                            let ext = ext; // rebind
                            let ent = ext.entities.len();
                            let sit = ext.situations.len();
                            for e in &ext.entities {
                                if !known_entities.contains(&e.name) {
                                    known_entities.push(e.name.clone());
                                }
                            }
                            update_chunk_progress(&self.progress, pc.chunk_i, |cp| {
                                cp.status = ChunkStatus::Done;
                                cp.finished_ms = Some(Utc::now().timestamp_millis());
                                cp.entities = ent;
                                cp.situations = sit;
                            });
                            last_result = Some(Ok(ext));
                            break; // success — move to next chunk
                        }
                        Err(e) => {
                            let err_str = format!("{}", e);
                            let is_rate_limit = matches!(&e, TensaError::LlmRateLimit { .. })
                                || err_str.to_lowercase().contains("rate limit");
                            let is_transient = is_rate_limit
                                || err_str.to_lowercase().contains("decoding response body")
                                || err_str.to_lowercase().contains("failed to parse")
                                || err_str.to_lowercase().contains("connection")
                                || err_str.to_lowercase().contains("timeout");

                            if is_transient && attempt < MAX_CHUNK_RETRIES {
                                // Rate limits get longer waits; parse/network errors get shorter
                                let wait_secs = if is_rate_limit {
                                    match &e {
                                        TensaError::LlmRateLimit { retry_after_secs } => {
                                            (*retry_after_secs).max(5)
                                        }
                                        _ => 10u64 * 2u64.pow(attempt), // 10s, 20s, 40s
                                    }
                                } else {
                                    3u64 * 2u64.pow(attempt) // 3s, 6s, 12s for parse/network errors
                                };
                                tracing::warn!(
                                    "Chunk {} transient error (attempt {}/{}), waiting {}s: {}",
                                    chunks[pc.chunk_i].chunk_id,
                                    attempt + 1,
                                    MAX_CHUNK_RETRIES + 1,
                                    wait_secs,
                                    err_str
                                );
                                update_chunk_progress(&self.progress, pc.chunk_i, |cp| {
                                    cp.status = ChunkStatus::Failed;
                                    cp.error =
                                        Some(format!("Retrying in {}s: {}", wait_secs, err_str));
                                });
                                std::thread::sleep(std::time::Duration::from_secs(wait_secs));
                                continue; // retry this chunk
                            }

                            // Non-transient error or retries exhausted
                            let msg = if attempt > 0 {
                                format!(
                                    "Chunk {} extraction failed after {} attempts: {}",
                                    chunks[pc.chunk_i].chunk_id,
                                    attempt + 1,
                                    e
                                )
                            } else {
                                format!(
                                    "Chunk {} extraction failed: {}",
                                    chunks[pc.chunk_i].chunk_id, e
                                )
                            };
                            update_chunk_progress(&self.progress, pc.chunk_i, |cp| {
                                cp.status = ChunkStatus::Failed;
                                cp.finished_ms = Some(Utc::now().timestamp_millis());
                                cp.error = Some(msg.clone());
                            });
                            last_result = Some(Err(msg));
                            break;
                        }
                    }
                }
                results.push((
                    pc.chunk_i,
                    pc.hash.clone(),
                    last_result.unwrap_or_else(|| Err("Cancelled".into())),
                ));
            }
            active_workers.store(0, Ordering::Relaxed);
            self.update_progress(|p| p.active_workers = 0);
            results
        } else {
            // Concurrent path: work queue dispatched to a fixed thread pool.
            // Only `concurrency` OS threads exist at any time (not one per chunk).
            let extractor = Arc::clone(&self.extractor);
            let cancel = self.cancelled.clone();
            let progress = self.progress.clone();
            let workers = Arc::clone(&active_workers);
            let completed = Arc::new(AtomicUsize::new(0));
            let jq_for_workers = self.job_queue.clone();
            let job_id_for_workers = self.config.job_id.clone();

            // Capture tokio runtime handle so worker threads can call block_on
            let rt_handle = tokio::runtime::Handle::try_current().ok();

            // Work channel: dispatcher sends chunks, workers pull from it
            type WorkItem = (usize, String, crate::ingestion::chunker::TextChunk);
            let (work_tx, work_rx) = std::sync::mpsc::channel::<WorkItem>();
            let work_rx = Arc::new(Mutex::new(work_rx));

            // Result channel: workers send results back
            type ResultItem = (
                usize,
                String,
                std::result::Result<NarrativeExtraction, String>,
            );
            let (result_tx, result_rx) = std::sync::mpsc::channel::<ResultItem>();

            // Spawn exactly `concurrency` worker threads
            let mut thread_handles = Vec::with_capacity(concurrency);
            for _ in 0..concurrency {
                let rx = Arc::clone(&work_rx);
                let tx = result_tx.clone();
                let ext = Arc::clone(&extractor);
                let flag = cancel.clone();
                let rt = rt_handle.clone();
                let prog = progress.clone();
                let done = Arc::clone(&completed);
                let wkrs = Arc::clone(&workers);
                let jq = jq_for_workers.clone();
                let jid = job_id_for_workers.clone();
                let cascade = self.config.cascade_mode.clone();

                thread_handles.push(std::thread::spawn(move || {
                    // Enter tokio runtime context so extract_narrative can call block_on
                    let _guard = rt.as_ref().map(|h| h.enter());
                    loop {
                        let item = match rx.lock().ok().and_then(|rx| rx.recv().ok()) {
                            Some(item) => item,
                            None => break, // channel closed
                        };
                        let (chunk_i, hash, chunk) = item;

                        if flag.load(Ordering::Relaxed) {
                            let _ = tx.send((chunk_i, hash, Err("Cancelled".into())));
                            continue;
                        }

                        let now_ms = Utc::now().timestamp_millis();
                        let w = wkrs.fetch_add(1, Ordering::Relaxed) + 1;
                        update_chunk_progress(&prog, chunk_i, |cp| {
                            cp.status = ChunkStatus::Extracting;
                            cp.started_ms = Some(now_ms);
                            cp.attempts += 1;
                        });
                        if let Some(ref p) = prog {
                            if let Ok(mut guard) = p.lock() {
                                guard.active_workers = w;
                            }
                        }

                        let r = match cascade_extract(ext.as_ref(), &chunk, &[], &cascade) {
                            Ok((extraction, exchange)) => {
                                // Persist LLM call log
                                if let (Some(ref jq), Some(ref jid)) = (&jq, &jid) {
                                    if let Some(ref ex) = exchange {
                                        let log = crate::ingestion::jobs::LlmCallLog {
                                            job_id: jid.clone(),
                                            chunk_index: chunk_i,
                                            pass: 1,
                                            attempt: 0,
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
                                            timestamp: Utc::now(),
                                        };
                                        if let Err(e) = jq.store_llm_log(&log) {
                                            tracing::warn!(
                                                "Failed to store LLM log for job={} chunk={}: {}",
                                                jid,
                                                chunk_i,
                                                e
                                            );
                                        }
                                    } else {
                                        tracing::debug!(
                                            "Concurrent worker: exchange is None for chunk={}",
                                            chunk_i
                                        );
                                    }
                                }
                                let ent = extraction.entities.len();
                                let sit = extraction.situations.len();
                                update_chunk_progress(&prog, chunk_i, |cp| {
                                    cp.status = ChunkStatus::Done;
                                    cp.finished_ms = Some(Utc::now().timestamp_millis());
                                    cp.entities = ent;
                                    cp.situations = sit;
                                });
                                Ok(extraction)
                            }
                            Err(e) => {
                                let msg =
                                    format!("Chunk {} extraction failed: {}", chunk.chunk_id, e);
                                update_chunk_progress(&prog, chunk_i, |cp| {
                                    cp.status = ChunkStatus::Failed;
                                    cp.finished_ms = Some(Utc::now().timestamp_millis());
                                    cp.error = Some(msg.clone());
                                });
                                Err(msg)
                            }
                        };

                        let w = wkrs.fetch_sub(1, Ordering::Relaxed) - 1;
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(ref p) = prog {
                            if let Ok(mut guard) = p.lock() {
                                guard.current_chunk = n;
                                guard.active_workers = w;
                            }
                        }
                        let _ = tx.send((chunk_i, hash, r));
                    }
                }));
            }
            // Drop our copy so workers see channel close when dispatcher is done
            drop(result_tx);

            // Dispatch all work items
            for pc in &pending {
                let _ = work_tx.send((pc.chunk_i, pc.hash.clone(), chunks[pc.chunk_i].clone()));
            }
            drop(work_tx); // signal workers to exit after draining

            // Collect results
            let mut results: Vec<ResultItem> = result_rx.iter().collect();
            for handle in thread_handles {
                let _ = handle.join();
            }
            results.sort_by_key(|(i, _, _)| *i);

            // Retry failed chunks sequentially with backoff (up to 3 rounds)
            for retry_round in 1..=3u32 {
                let failed_indices: Vec<usize> = results
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, _, r))| r.is_err())
                    .map(|(i, _)| i)
                    .collect();
                if failed_indices.is_empty() || cancel.load(Ordering::Relaxed) {
                    break;
                }
                let wait_secs = 10u64 * 2u64.pow(retry_round - 1);
                tracing::info!(
                    "Retrying {} failed chunks (round {}/3) after {}s backoff",
                    failed_indices.len(),
                    retry_round,
                    wait_secs
                );
                std::thread::sleep(std::time::Duration::from_secs(wait_secs));
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                for &fi in &failed_indices {
                    let (chunk_i, ref _hash, _) = results[fi];
                    update_chunk_progress(&progress, chunk_i, |cp| {
                        cp.status = ChunkStatus::Extracting;
                        cp.started_ms = Some(Utc::now().timestamp_millis());
                        cp.attempts += 1;
                        cp.error = None;
                    });
                    let retry_r = match extractor.extract_with_logging(&chunks[chunk_i], &[]) {
                        Ok((ext, exchange)) => {
                            // Persist retry log
                            if let (Some(ref jq), Some(ref jid)) =
                                (&jq_for_workers, &job_id_for_workers)
                            {
                                if let Some(ref ex) = exchange {
                                    let log = crate::ingestion::jobs::LlmCallLog {
                                        job_id: jid.clone(),
                                        chunk_index: chunk_i,
                                        pass: 1,
                                        attempt: retry_round as u8,
                                        system_prompt: ex.system_prompt.clone(),
                                        user_prompt: ex.user_prompt.clone(),
                                        raw_response: ex.raw_response.clone(),
                                        parsed_extraction: Some(ext.clone()),
                                        parse_error: ex.parse_error.clone(),
                                        retry_prompt: ex.retry_prompt.clone(),
                                        retry_response: ex.retry_response.clone(),
                                        duration_ms: ex.duration_ms,
                                        model: ex.model.clone(),
                                        endpoint: ex.endpoint.clone(),
                                        timestamp: Utc::now(),
                                    };
                                    let _ = jq.store_llm_log(&log);
                                }
                            }
                            let ent = ext.entities.len();
                            let sit = ext.situations.len();
                            update_chunk_progress(&progress, chunk_i, |cp| {
                                cp.status = ChunkStatus::Done;
                                cp.finished_ms = Some(Utc::now().timestamp_millis());
                                cp.entities = ent;
                                cp.situations = sit;
                            });
                            Ok(ext)
                        }
                        Err(e) => {
                            let msg = format!(
                                "Chunk {} extraction failed (retry {}): {}",
                                chunks[chunk_i].chunk_id, retry_round, e
                            );
                            update_chunk_progress(&progress, chunk_i, |cp| {
                                cp.status = ChunkStatus::Failed;
                                cp.finished_ms = Some(Utc::now().timestamp_millis());
                                cp.error = Some(msg.clone());
                            });
                            Err(msg)
                        }
                    };
                    results[fi].2 = retry_r;
                }
            }
            results
        };

        // Phase C: Process results in order — store chunks, validate, collect extractions
        let mut all_extractions: Vec<(
            NarrativeExtraction,
            Option<crate::ingestion::extraction::ExtractionEnrichment>,
        )> = Vec::new();
        let mut chunk_ids: Vec<Option<Uuid>> = Vec::new();
        for (chunk_i, hash, result) in extraction_results {
            if self.is_cancelled() {
                report.cancelled = true;
                report.duration_secs = start.elapsed().as_secs_f64();
                return Ok(report);
            }
            match result {
                Ok(extraction) => {
                    // Repair common LLM errors before validation
                    let mut extraction = extraction;
                    let repair_warnings = repair_extraction(&mut extraction);
                    for w in &repair_warnings {
                        report.errors.push(format!(
                            "Chunk {} [repaired]: {}",
                            chunks[chunk_i].chunk_id, w.message
                        ));
                    }
                    let warnings = validate_extraction(&extraction);
                    for w in &warnings {
                        report
                            .errors
                            .push(format!("Chunk {}: {}", chunks[chunk_i].chunk_id, w.message));
                    }
                    let chunk_record = ChunkRecord {
                        id: Uuid::now_v7(),
                        job_id: self.config.job_id.clone().unwrap_or_default(),
                        narrative_id: self.config.narrative_id.clone(),
                        chunk_index: chunk_i as u32,
                        text: chunks[chunk_i].text.clone(),
                        byte_range: (chunks[chunk_i].start_offset, chunks[chunk_i].end_offset),
                        overlap_bytes: chunks[chunk_i].overlap_prefix.len(),
                        chapter: chunks[chunk_i].chapter.clone(),
                        content_hash: hash,
                        embedding: self
                            .embedder
                            .as_ref()
                            .and_then(|e| e.embed_text(&chunks[chunk_i].text).ok()),
                        created_at: Utc::now(),
                    };
                    let chunk_id = self.hypergraph.store_chunk(&chunk_record)?;
                    report.created_chunk_ids.push(chunk_id);
                    chunk_ids.push(Some(chunk_id));

                    let enrichment = if self.config.enrich {
                        match self
                            .extractor
                            .enrich_extraction(&chunks[chunk_i], &extraction)
                        {
                            Ok(enrichment) => Some(enrichment),
                            Err(e) => {
                                report.errors.push(format!(
                                    "Chunk {} [enrich]: {}",
                                    chunks[chunk_i].chunk_id, e
                                ));
                                None
                            }
                        }
                    } else {
                        None
                    };

                    all_extractions.push((extraction, enrichment));
                }
                Err(msg) => {
                    report.errors.push(msg);
                }
            }
        }

        // Stage 2b: Multi-pass reconciliation (if configured)
        if self.pass2_extractor.is_some() && !all_extractions.is_empty() {
            self.update_progress(|p| {
                p.phase = IngestionPhase::Reconciling;
                p.current_pass = 2;
            });
            let mut extractions_only: Vec<NarrativeExtraction> =
                all_extractions.iter().map(|(e, _)| e.clone()).collect();
            self.run_pass2_reconciliation(&chunks, &mut extractions_only, &mut report);
            // Write back reconciled extractions
            for (i, ext) in extractions_only.into_iter().enumerate() {
                if let Some(entry) = all_extractions.get_mut(i) {
                    entry.0 = ext;
                }
            }
        }

        // Stage 3-5: Process each extraction
        self.update_progress(|p| p.phase = IngestionPhase::Processing);
        let provenance = SourceReference {
            source_type: self.config.source_type.clone(),
            source_id: Some(self.config.source_id.clone()),
            description: Some(source_name.to_string()),
            timestamp: Utc::now(),
            registered_source: None,
        };

        for (chunk_idx, (extraction, enrichment)) in all_extractions.iter().enumerate() {
            if self.is_cancelled() {
                report.cancelled = true;
                report.duration_secs = start.elapsed().as_secs_f64();
                return Ok(report);
            }
            let chunk_id = chunk_ids.get(chunk_idx).copied().flatten();
            let byte_range = chunks
                .get(chunk_idx)
                .map(|c| (c.start_offset, c.end_offset));
            let chunk_text = chunks.get(chunk_idx).map(|c| c.text.as_str());
            let chunk_report = self.process_extraction(
                extraction,
                chunk_idx,
                chunk_id,
                byte_range,
                chunk_text,
                enrichment.as_ref(),
                &mut resolver,
                &provenance,
            );
            match chunk_report {
                Ok(gate_report) => {
                    let entity_map: std::collections::HashMap<String, Uuid> = extraction
                        .entities
                        .iter()
                        .zip(gate_report.entity_ids.iter())
                        .map(|(e, id)| (e.name.clone(), *id))
                        .collect();
                    self.store_chunk_extraction_record(
                        chunk_idx,
                        extraction,
                        Vec::new(), // gate decisions collected at finer grain later
                        &entity_map,
                        &gate_report.situation_ids,
                    );

                    report
                        .created_entity_ids
                        .extend_from_slice(&gate_report.entity_ids);
                    report
                        .created_situation_ids
                        .extend_from_slice(&gate_report.situation_ids);
                    report.entities_created += gate_report.entity_ids.len();
                    report.situations_created += gate_report.situation_ids.len();
                    report.items_auto_committed += gate_report.auto_committed;
                    report.items_queued += gate_report.queued;
                    report.items_rejected += gate_report.rejected;
                    self.update_progress(|p| {
                        p.entities_so_far = report.entities_created;
                        p.situations_so_far = report.situations_created;
                    });
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("Chunk {} processing failed: {}", chunk_idx, e));
                }
            }
        }

        report.entity_resolutions = resolver.len();

        // Cross-chunk temporal reconciliation (if enrichment enabled and multiple chunks)
        if self.config.enrich && all_extractions.len() > 1 {
            self.update_progress(|p| p.phase = IngestionPhase::Reconciling);
            let chunk_summaries: Vec<(usize, Vec<(String, Option<String>, Option<String>)>)> =
                all_extractions
                    .iter()
                    .enumerate()
                    .map(|(i, (ext, _))| {
                        let sums = ext
                            .situations
                            .iter()
                            .map(|s| {
                                let name = s.name.clone().unwrap_or_else(|| s.description.clone());
                                (name, s.temporal_marker.clone(), s.location.clone())
                            })
                            .collect();
                        (i, sums)
                    })
                    .collect();
            match self.extractor.reconcile_temporal(&chunk_summaries) {
                Ok(reconciliation) => {
                    // Pre-load situation name index to avoid O(n*m) KV reads
                    let sit_name_index: Vec<(Uuid, String, bool)> = report
                        .created_situation_ids
                        .iter()
                        .filter_map(|&sid| {
                            self.hypergraph.get_situation(&sid).ok().map(|sit| {
                                let name = sit
                                    .name
                                    .as_ref()
                                    .or(sit.description.as_ref())
                                    .cloned()
                                    .unwrap_or_default()
                                    .to_lowercase();
                                let has_time = sit.temporal.start.is_some();
                                (sid, name, has_time)
                            })
                        })
                        .collect();

                    for te in &reconciliation.timeline {
                        let parsed = chrono::NaiveDate::parse_from_str(&te.date, "%Y-%m-%d")
                            .ok()
                            .and_then(|d| d.and_hms_opt(0, 0, 0))
                            .map(|dt| chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
                        if let Some(dt) = parsed {
                            let needle = te.situation.to_lowercase();
                            if let Some((sid, _, _)) = sit_name_index
                                .iter()
                                .find(|(_, name, has_time)| !has_time && name.contains(&needle))
                            {
                                if let Err(e) = self.hypergraph.update_situation(sid, |s| {
                                    s.temporal.start = Some(dt);
                                    s.temporal.end = Some(dt);
                                    s.temporal.granularity = TimeGranularity::Day;
                                }) {
                                    tracing::warn!(situation_id = %sid, error = %e, "Failed to persist reconciled temporal date");
                                }
                            }
                        }
                    }

                    for tr in &reconciliation.relations {
                        let relation = match tr.relation.parse::<AllenRelation>() {
                            Ok(r) => r,
                            Err(_) => continue,
                        };
                        let a_lower = tr.situation_a.to_lowercase();
                        let b_lower = tr.situation_b.to_lowercase();
                        let sit_a_id = sit_name_index
                            .iter()
                            .find(|(_, n, _)| n.contains(&a_lower))
                            .map(|(id, _, _)| *id);
                        let sit_b_id = sit_name_index
                            .iter()
                            .find(|(_, n, _)| n.contains(&b_lower))
                            .map(|(id, _, _)| *id);
                        if let (Some(a_id), Some(b_id)) = (sit_a_id, sit_b_id) {
                            let rel = AllenRelationTo {
                                target_situation: b_id,
                                relation,
                            };
                            if let Err(e) = self.hypergraph.update_situation(&a_id, |sit| {
                                if !sit.temporal.relations.iter().any(|r| {
                                    r.target_situation == rel.target_situation
                                        && r.relation == rel.relation
                                }) {
                                    sit.temporal.relations.push(rel.clone());
                                }
                            }) {
                                tracing::warn!(situation_id = %a_id, error = %e, "Failed to persist reconciled Allen relation");
                            }
                        }
                    }
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("Temporal reconciliation failed: {}", e));
                }
            }
        }

        report.duration_secs = start.elapsed().as_secs_f64();
        self.update_progress(|p| p.phase = IngestionPhase::Complete);
        Ok(report)
    }

    /// Run pass 2 reconciliation over overlapping windows of pass 1 extractions.
    ///
    /// For each window, sends a reconciliation prompt to the pass 2 LLM containing
    /// the pass 1 extractions. The LLM identifies entity merges and confidence
    /// adjustments. These are applied to the extractions before stage 3-5 processing.
    fn run_pass2_reconciliation(
        &self,
        _chunks: &[crate::ingestion::chunker::TextChunk],
        extractions: &mut Vec<NarrativeExtraction>,
        report: &mut IngestionReport,
    ) {
        let pass2 = match &self.pass2_extractor {
            Some(e) => e,
            None => return,
        };

        // Build overlapping windows (default: 20 chunks, 3 overlap)
        let window_size = 20usize.max(1);
        let window_overlap = 3usize.min(window_size.saturating_sub(1));
        let step = window_size.saturating_sub(window_overlap).max(1);

        let mut window_start = 0;
        while window_start < extractions.len() {
            let window_end = (window_start + window_size).min(extractions.len());
            let window = &extractions[window_start..window_end];

            // Build reconciliation context from pass 1 extractions
            let mut context = String::from(
                "Review these narrative extractions for consistency. \
                 Identify duplicate entities that should be merged, \
                 and flag any confidence adjustments needed.\n\n",
            );
            for (i, ext) in window.iter().enumerate() {
                context.push_str(&format!("--- Chunk {} ---\n", window_start + i));
                for ent in &ext.entities {
                    context.push_str(&format!(
                        "Entity: {} ({:?}, confidence: {:.2})\n",
                        ent.name, ent.entity_type, ent.confidence
                    ));
                }
                for sit in &ext.situations {
                    context.push_str(&format!(
                        "Situation: {} (confidence: {:.2})\n",
                        sit.description, sit.confidence
                    ));
                }
                context.push('\n');
            }

            // Use a dummy TextChunk to invoke the pass2 extractor
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
                    // Apply confidence boosts: if pass 2 mentions the same entity,
                    // bump confidence by 0.1 (capped at 1.0)
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
                }
                Err(e) => {
                    report.errors.push(format!(
                        "Pass 2 reconciliation window {}: {}",
                        window_start, e
                    ));
                }
            }

            window_start += step;
        }
    }

    /// Public entry point for reprocessing a stored extraction (used by API endpoints).
    /// Delegates to `process_extraction` but takes an external resolver and provenance.
    pub fn process_extraction_standalone(
        &self,
        extraction: &NarrativeExtraction,
        chunk_idx: usize,
        resolver: &mut EntityResolver,
        provenance: &SourceReference,
    ) -> Result<GateReport> {
        self.process_extraction(
            extraction, chunk_idx, None, None, None, None, resolver, provenance,
        )
    }

    /// Process a single extraction through resolve → gate → commit/queue.
    #[allow(clippy::too_many_arguments)]
    fn process_extraction(
        &self,
        extraction: &NarrativeExtraction,
        chunk_idx: usize,
        chunk_id: Option<Uuid>,
        chunk_byte_range: Option<(usize, usize)>,
        chunk_text: Option<&str>,
        enrichment: Option<&crate::ingestion::extraction::ExtractionEnrichment>,
        resolver: &mut EntityResolver,
        provenance: &SourceReference,
    ) -> Result<GateReport> {
        let mut report = GateReport::default();

        // Maps from extraction index to hypergraph UUID
        let mut entity_map: std::collections::HashMap<String, Uuid> =
            std::collections::HashMap::new();
        let mut situation_ids: Vec<Uuid> = Vec::new();

        // Process entities
        for extracted in &extraction.entities {
            let decision = self.gate.decide(extracted.confidence);
            match decision {
                GateDecision::AutoCommit => {
                    let (id, is_new) = self.commit_entity(extracted, resolver, provenance)?;
                    entity_map.insert(extracted.name.clone(), id);
                    if is_new {
                        report.entity_ids.push(id);
                    }
                    report.auto_committed += 1;
                }
                GateDecision::QueueForReview => {
                    self.queue_item(
                        QueueItemType::Entity,
                        serde_json::to_value(extracted)?,
                        chunk_idx,
                        chunk_id,
                        extracted.confidence,
                    )?;
                    report.queued += 1;
                }
                GateDecision::Reject => {
                    report.rejected += 1;
                }
            }
        }

        // Process situations
        for (local_idx, extracted) in extraction.situations.iter().enumerate() {
            let decision = self.gate.decide(extracted.confidence);
            let span = chunk_byte_range.map(|(chunk_start, chunk_end)| {
                // Narrow to per-situation span when the LLM returned verbatim fingerprints.
                // Fallback to the whole-chunk range when fingerprints are missing or unmatched.
                let (byte_start, byte_end) = chunk_text
                    .and_then(|ct| {
                        crate::ingestion::span_resolve::resolve_span(
                            ct,
                            chunk_start,
                            extracted.text_start.as_deref(),
                            extracted.text_end.as_deref(),
                        )
                    })
                    .unwrap_or_else(|| {
                        if extracted.text_start.is_some() || extracted.text_end.is_some() {
                            tracing::debug!(
                                situation = extracted.name.as_deref().unwrap_or("(unnamed)"),
                                "span fingerprint miss; falling back to chunk-wide offsets"
                            );
                        }
                        (chunk_start, chunk_end)
                    });
                SourceSpan {
                    chunk_index: chunk_idx as u32,
                    byte_offset_start: byte_start,
                    byte_offset_end: byte_end,
                    local_index: local_idx as u16,
                }
            });
            match decision {
                GateDecision::AutoCommit => {
                    let id = self.commit_situation(extracted, provenance, chunk_id, span)?;
                    situation_ids.push(id);
                    report.situation_ids.push(id);
                    report.auto_committed += 1;
                }
                GateDecision::QueueForReview => {
                    self.queue_item(
                        QueueItemType::Situation,
                        serde_json::to_value(extracted)?,
                        chunk_idx,
                        chunk_id,
                        extracted.confidence,
                    )?;
                    report.queued += 1;
                }
                GateDecision::Reject => {
                    report.rejected += 1;
                }
            }
        }

        // Process participations — auto-create missing entities on the fly.
        // LLMs frequently reference locations/artifacts in participations without
        // adding them to the entities array. Instead of dropping these, we upsert
        // the missing entity through the normal gate + resolver path.
        for extracted in &extraction.participations {
            // Case-insensitive lookup in committed entity map
            let entity_id_opt = entity_map.get(&extracted.entity_name).copied().or_else(|| {
                let lower = extracted.entity_name.to_lowercase();
                entity_map
                    .iter()
                    .find(|(k, _)| k.to_lowercase() == lower)
                    .map(|(_, &v)| v)
            });
            let entity_id = match entity_id_opt {
                Some(id) => id,
                None if extracted.entity_name.trim().is_empty() => continue,
                None => {
                    // Auto-create the missing entity — guess type from participation role
                    let entity_type =
                        crate::ingestion::extraction::guess_entity_type_from_role(&extracted.role);
                    let synthetic = crate::ingestion::extraction::ExtractedEntity {
                        name: extracted.entity_name.clone(),
                        aliases: vec![],
                        entity_type,
                        properties: serde_json::json!({}),
                        confidence: (extracted.confidence * 0.8).max(0.3),
                    };
                    let decision = self.gate.decide(synthetic.confidence);
                    match decision {
                        GateDecision::AutoCommit => {
                            let (id, is_new) =
                                self.commit_entity(&synthetic, resolver, provenance)?;
                            entity_map.insert(extracted.entity_name.clone(), id);
                            if is_new {
                                report.entity_ids.push(id);
                            }
                            report.auto_committed += 1;
                            id
                        }
                        GateDecision::QueueForReview => {
                            self.queue_item(
                                QueueItemType::Entity,
                                serde_json::to_value(&synthetic)?,
                                chunk_idx,
                                chunk_id,
                                synthetic.confidence,
                            )?;
                            report.queued += 1;
                            continue;
                        }
                        GateDecision::Reject => {
                            report.rejected += 1;
                            continue;
                        }
                    }
                }
            };
            if let Some(&situation_id) = situation_ids.get(extracted.situation_index) {
                let decision = self.gate.decide(extracted.confidence);
                if decision == GateDecision::AutoCommit {
                    let participation = Participation {
                        entity_id,
                        situation_id,
                        role: extracted.role.clone(),
                        info_set: None,
                        action: extracted.action.clone(),
                        payoff: None,
                        seq: 0,
                    };
                    match self.hypergraph.add_participant(participation) {
                        Ok(()) => report.auto_committed += 1,
                        Err(e) => return Err(e),
                    }
                } else if decision == GateDecision::QueueForReview {
                    self.queue_item(
                        QueueItemType::Participation,
                        serde_json::to_value(extracted)?,
                        chunk_idx,
                        chunk_id,
                        extracted.confidence,
                    )?;
                    report.queued += 1;
                } else {
                    report.rejected += 1;
                }
            }
        }

        // Auto-link Location entities to situations via their `location` field.
        // The LLM extracts locations as entities but doesn't include them in the
        // participations array. We bridge the gap by matching each situation's
        // location string to a Location entity and creating a Setting participation.
        for (sit_idx, extracted_sit) in extraction.situations.iter().enumerate() {
            if let Some(ref loc_name) = extracted_sit.location {
                if loc_name.is_empty() {
                    continue;
                }
                // Find matching Location entity (case-insensitive)
                let loc_lower = loc_name.to_lowercase();
                let loc_entity_id = entity_map
                    .iter()
                    .find(|(k, _)| k.to_lowercase() == loc_lower)
                    .map(|(_, &v)| v);

                if let Some(entity_id) = loc_entity_id {
                    if let Some(&situation_id) = situation_ids.get(sit_idx) {
                        let participation = Participation {
                            entity_id,
                            situation_id,
                            role: Role::Custom("Setting".into()),
                            info_set: None,
                            action: None,
                            payoff: None,
                            seq: 0,
                        };
                        match self.hypergraph.add_participant(participation) {
                            Ok(()) => report.auto_committed += 1,
                            Err(e) => return Err(e),
                        }
                    }
                }
            }
        }

        // Process causal links (from step 1 + step 2 enrichment extras)
        let all_causal = if let Some(enr) = enrichment {
            let mut combined = extraction.causal_links.clone();
            combined.extend(enr.extra_causal_links.iter().cloned());
            combined
        } else {
            extraction.causal_links.clone()
        };
        let mut causal_committed_in_chunk = 0usize;
        for extracted in &all_causal {
            if let (Some(&from_id), Some(&to_id)) = (
                situation_ids.get(extracted.from_situation_index),
                situation_ids.get(extracted.to_situation_index),
            ) {
                let decision = self.gate.decide(extracted.confidence);
                if decision == GateDecision::AutoCommit {
                    let link = CausalLink {
                        from_situation: from_id,
                        to_situation: to_id,
                        mechanism: extracted.mechanism.clone(),
                        strength: extracted.strength,
                        causal_type: extracted.causal_type,
                        maturity: MaturityLevel::Candidate,
                    };
                    match self.hypergraph.add_causal_link(link) {
                        Ok(()) => {
                            report.auto_committed += 1;
                            causal_committed_in_chunk += 1;
                        }
                        Err(TensaError::CausalCycle { .. }) => {} // Skip cycles
                        Err(e) => return Err(e),
                    }
                }
            }
        }

        // Adjacency fallback: when the LLM commits zero causal edges for a
        // chunk with multiple situations, chain them sequentially so
        // downstream analysis (Workshop orphan detector, causal influence,
        // narrative diameter) doesn't degenerate. Mechanism string marks
        // these as synthesised.
        if causal_committed_in_chunk == 0 && situation_ids.len() >= 2 {
            for pair in situation_ids.windows(2) {
                if crate::narrative::causal_helpers::add_sequential_link(
                    &self.hypergraph,
                    pair[0],
                    pair[1],
                    crate::narrative::causal_helpers::MECHANISM_FALLBACK,
                    0.3,
                    MaturityLevel::Candidate,
                )? {
                    report.auto_committed += 1;
                }
            }
        }

        // Process temporal relations (from step 1 + step 2 enrichment temporal_chain)
        let mut all_temporal = extraction.temporal_relations.clone();
        if let Some(enr) = enrichment {
            all_temporal.extend(enr.temporal_chain.iter().cloned());
        }
        for extracted in &all_temporal {
            if let (Some(&sit_a), Some(&sit_b)) = (
                situation_ids.get(extracted.situation_a_index),
                situation_ids.get(extracted.situation_b_index),
            ) {
                // Add the relation to situation A's temporal.relations
                let relation_to = AllenRelationTo {
                    target_situation: sit_b,
                    relation: extracted.relation,
                };
                if let Err(e) = self.hypergraph.update_situation(&sit_a, |sit| {
                    // Avoid duplicates
                    if !sit.temporal.relations.iter().any(|r| {
                        r.target_situation == relation_to.target_situation
                            && r.relation == relation_to.relation
                    }) {
                        sit.temporal.relations.push(relation_to.clone());
                    }
                }) {
                    tracing::warn!(situation_id = %sit_a, error = %e, "Failed to persist temporal relation");
                }
            }
        }

        // Apply step-2 enrichment to committed entities and situations
        if let Some(enr) = enrichment {
            self.apply_enrichment(enr, &entity_map, &situation_ids)?;
        }

        Ok(report)
    }

    // ─── SingleSession ingestion ─────────────────────────────────

    /// Insert chunk boundary markers into the original text.
    #[allow(dead_code)]
    fn insert_chunk_markers(text: &str, chunks: &[crate::ingestion::chunker::TextChunk]) -> String {
        let mut marked = String::with_capacity(text.len() + chunks.len() * 80);
        let mut last_end = 0;

        for chunk in chunks {
            // Any text gap before this chunk
            if chunk.start_offset > last_end {
                marked.push_str(&text[last_end..chunk.start_offset]);
            }

            let title = chunk.chapter.as_deref().unwrap_or("");
            if title.is_empty() {
                marked.push_str(&format!("\n[=== CHUNK {} ===]\n", chunk.chunk_id + 1));
            } else {
                marked.push_str(&format!(
                    "\n[=== CHUNK {}: \"{}\" ===]\n",
                    chunk.chunk_id + 1,
                    title
                ));
            }
            marked.push_str(&chunk.text);
            marked.push_str(&format!("\n[=== END CHUNK {} ===]\n", chunk.chunk_id + 1));

            last_end = chunk.end_offset;
        }

        if last_end < text.len() {
            marked.push_str(&text[last_end..]);
        }
        marked
    }

    /// Run the SingleSession ingestion flow.
    fn ingest_text_single_session(
        &self,
        _text: &str,
        source_name: &str,
        chunks: &[crate::ingestion::chunker::TextChunk],
        session: &dyn crate::ingestion::llm::SessionCapableExtractor,
        ctx_limit: usize,
        start: std::time::Instant,
    ) -> Result<IngestionReport> {
        use crate::ingestion::chunker::estimate_tokens;
        use crate::ingestion::extraction::{
            parse_enrichment_response, parse_llm_response, parse_session_reconciliation_response,
            repair_enrichment, repair_extraction, ExtractionEnrichment, NarrativeExtraction,
        };
        use crate::ingestion::llm::*;

        let mut report = IngestionReport::default();
        report.chunks_processed = chunks.len();

        let model_name = self
            .extractor
            .model_name()
            .unwrap_or_else(|| "unknown".into());
        tracing::info!(
            chunks = chunks.len(),
            ctx_limit = ctx_limit,
            model = %model_name,
            job_id = ?self.config.job_id,
            "SingleSession: starting session pipeline"
        );

        self.extractor.set_cancel_flag(self.cancelled.clone());

        let base_system = crate::ingestion::llm::EXTRACTION_SYSTEM_PROMPT.to_string();
        let system_prompt = build_session_system_prompt(&base_system);

        // Session accumulator
        let mut entity_names: Vec<String> = Vec::new();
        let mut situation_summaries: Vec<(usize, String)> = Vec::new();
        let mut all_extractions: Vec<Option<(NarrativeExtraction, Option<ExtractionEnrichment>)>> =
            vec![None; chunks.len()];
        let mut estimated_tokens: usize = estimate_tokens(&system_prompt);

        let mut messages: Vec<ApiMessage> = vec![ApiMessage {
            role: "system".to_string(),
            content: system_prompt,
        }];

        // Store chunk manifest for session mode (same as regular mode)
        if let (Some(jq), Some(job_id)) = (&self.job_queue, &self.config.job_id) {
            let manifest: Vec<crate::ingestion::jobs::ChunkManifestEntry> = chunks
                .iter()
                .enumerate()
                .map(|(i, c)| crate::ingestion::jobs::ChunkManifestEntry {
                    chunk_index: i,
                    start: c.start_offset,
                    end: c.end_offset,
                    overlap_bytes: c.overlap_prefix.len(),
                    chapter: c.chapter.clone(),
                    content_hash: crate::ingestion::chunker::chunk_hash(&c.text),
                })
                .collect();
            let _ = jq.store_chunk_manifest(job_id, &manifest);
        }

        // Store ALL chunk records upfront (before LLM contact — ensures text is always available)
        let mut chunk_id_map: Vec<Option<Uuid>> = vec![None; chunks.len()];
        for (chunk_i, chunk) in chunks.iter().enumerate() {
            let hash = crate::ingestion::chunker::chunk_hash(&chunk.text);
            let chunk_record = ChunkRecord {
                id: Uuid::now_v7(),
                job_id: self.config.job_id.clone().unwrap_or_default(),
                narrative_id: self.config.narrative_id.clone(),
                chunk_index: chunk_i as u32,
                text: chunk.text.clone(),
                byte_range: (chunk.start_offset, chunk.end_offset),
                overlap_bytes: chunk.overlap_prefix.len(),
                chapter: chunk.chapter.clone(),
                content_hash: hash,
                embedding: self
                    .embedder
                    .as_ref()
                    .and_then(|e| e.embed_text(&chunk.text).ok()),
                created_at: Utc::now(),
            };
            if let Ok(chunk_id) = self.hypergraph.store_chunk(&chunk_record) {
                report.created_chunk_ids.push(chunk_id);
                chunk_id_map[chunk_i] = Some(chunk_id);
            }
        }

        // Step 2: Load context chunk by chunk (avoids single massive payload)
        self.update_progress(|p| {
            p.phase = IngestionPhase::ContextLoading;
            p.pass_mode = crate::ingestion::jobs::PassMode::Session;
            p.model = Some(model_name.clone());
            p.total_chunks = chunks.len();
            p.session_status = Some(format!(
                "Loading document into LLM context (model: {})...",
                model_name
            ));
        });

        // Send text in segments — one per chunk with markers, asking LLM to acknowledge each.
        // This keeps individual payloads small and lets the LLM cache incrementally.
        let batch_size = 3; // send 3 chunks per context loading message
        let mut chunk_batches: Vec<String> = Vec::new();
        let mut current_batch = String::new();
        let mut batch_count = 0;

        for (i, chunk) in chunks.iter().enumerate() {
            let title = chunk.chapter.as_deref().unwrap_or("");
            if title.is_empty() {
                current_batch.push_str(&format!("\n[=== CHUNK {} ===]\n", i + 1));
            } else {
                current_batch.push_str(&format!("\n[=== CHUNK {}: \"{}\" ===]\n", i + 1, title));
            }
            // Use chunk.text directly — safe UTF-8, includes overlap prefix
            current_batch.push_str(&chunk.text);
            current_batch.push_str(&format!("\n[=== END CHUNK {} ===]\n", i + 1));
            batch_count += 1;

            if batch_count >= batch_size || i == chunks.len() - 1 {
                chunk_batches.push(std::mem::take(&mut current_batch));
                batch_count = 0;
            }
        }

        for (batch_idx, batch_text) in chunk_batches.iter().enumerate() {
            if self.is_cancelled() {
                report.cancelled = true;
                report.duration_secs = start.elapsed().as_secs_f64();
                return Ok(report);
            }

            let is_last = batch_idx == chunk_batches.len() - 1;
            let msg = if batch_idx == 0 {
                format!(
                    "I will send you a document in {} parts with chunk markers. Read and acknowledge each part. Part {}/{}:\n\n{}",
                    chunk_batches.len(), batch_idx + 1, chunk_batches.len(), batch_text
                )
            } else if is_last {
                format!(
                    "Part {}/{} (final):\n\n{}\n\nAll {} chunks loaded. Ready for extraction.",
                    batch_idx + 1,
                    chunk_batches.len(),
                    batch_text,
                    chunks.len()
                )
            } else {
                format!(
                    "Part {}/{}:\n\n{}",
                    batch_idx + 1,
                    chunk_batches.len(),
                    batch_text
                )
            };

            estimated_tokens += estimate_tokens(&msg);
            messages.push(ApiMessage {
                role: "user".to_string(),
                content: msg,
            });

            self.update_progress(|p| {
                p.session_status = Some(format!(
                    "Loading context part {}/{} (~{} tokens)",
                    batch_idx + 1,
                    chunk_batches.len(),
                    estimated_tokens
                ));
                p.session_turn = messages.len();
                p.session_tokens_used = estimated_tokens;
            });

            tracing::debug!(
                batch = batch_idx + 1,
                total_batches = chunk_batches.len(),
                est_tokens = estimated_tokens,
                messages_len = messages.len(),
                "SingleSession: sending context batch to LLM"
            );
            let batch_start = std::time::Instant::now();
            let _user_msg = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            // Retry context batch up to 3 times with backoff
            let mut batch_ok = false;
            for attempt in 0..3u32 {
                if attempt > 0 {
                    let wait = 3u64 * 2u64.pow(attempt - 1); // 3s, 6s
                    tracing::warn!(
                        batch = batch_idx + 1,
                        attempt = attempt + 1,
                        wait_secs = wait,
                        "SingleSession: retrying context batch"
                    );
                    std::thread::sleep(std::time::Duration::from_secs(wait));
                }
                match session.send_session_messages(&messages) {
                    Ok(ack) => {
                        let batch_ms = batch_start.elapsed().as_millis() as u64;
                        tracing::info!(
                            batch = batch_idx + 1,
                            duration_ms = batch_ms,
                            ack_len = ack.len(),
                            attempt = attempt + 1,
                            "SingleSession: context batch acknowledged"
                        );
                        self.store_llm_log(
                            batch_idx,
                            0,
                            0,
                            &Some(crate::ingestion::llm::RawLlmExchange {
                                system_prompt: "[session context load]".into(),
                                user_prompt: format!(
                                    "[context batch {}/{}]",
                                    batch_idx + 1,
                                    chunk_batches.len()
                                ),
                                raw_response: ack.chars().take(500).collect(),
                                retry_prompt: None,
                                retry_response: None,
                                parse_error: None,
                                duration_ms: batch_ms,
                                model: Some(model_name.clone()),
                                endpoint: Some("session".into()),
                            }),
                            None,
                            None,
                        );
                        estimated_tokens += estimate_tokens(&ack);
                        messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: ack,
                        });
                        batch_ok = true;
                        break;
                    }
                    Err(e) => {
                        tracing::error!(
                            batch = batch_idx + 1,
                            attempt = attempt + 1,
                            error = %e,
                            elapsed_ms = batch_start.elapsed().as_millis(),
                            "SingleSession: context loading failed"
                        );
                        if attempt == 2 {
                            self.store_llm_log(
                                batch_idx,
                                0,
                                0,
                                &Some(crate::ingestion::llm::RawLlmExchange {
                                    system_prompt: "[session context load]".into(),
                                    user_prompt: format!(
                                        "[context batch {}/{}]",
                                        batch_idx + 1,
                                        chunk_batches.len()
                                    ),
                                    raw_response: String::new(),
                                    retry_prompt: None,
                                    retry_response: None,
                                    parse_error: Some(e.to_string()),
                                    duration_ms: batch_start.elapsed().as_millis() as u64,
                                    model: Some(model_name.clone()),
                                    endpoint: Some("session".into()),
                                }),
                                None,
                                None,
                            );
                            report.errors.push(format!(
                                "Context loading part {} failed after 3 attempts: {}",
                                batch_idx + 1,
                                e
                            ));
                        }
                    }
                }
            }
            if !batch_ok {
                // Context loading failed — chunks are already stored, skip to report
                report.duration_secs = start.elapsed().as_secs_f64();
                return Ok(report);
            }
        }

        self.update_progress(|p| {
            p.phase = IngestionPhase::Extracting;
            p.session_tokens_used = estimated_tokens;
            p.session_status = Some(format!(
                "Context loaded (~{} tokens). Starting extraction...",
                estimated_tokens
            ));
        });

        // Step 3: Per-chunk extraction
        for (i, chunk) in chunks.iter().enumerate() {
            if self.is_cancelled() {
                report.cancelled = true;
                break;
            }

            // Check token budget (85% safety margin)
            if estimated_tokens > ctx_limit * 85 / 100 {
                report.errors.push(format!(
                    "Session token budget exceeded at chunk {} ({} est. tokens / {} limit). Stopping extraction.",
                    i, estimated_tokens, ctx_limit
                ));
                break;
            }

            let chunk_title = chunk.chapter.as_deref().unwrap_or("untitled");
            self.update_progress(|p| {
                p.current_chunk = i;
                p.session_status = Some(format!(
                    "Extracting chunk {}/{}: \"{}\"...",
                    i + 1,
                    chunks.len(),
                    chunk_title
                ));
            });

            let summary = Self::build_accumulator_summary(&entity_names, &situation_summaries);
            let prompt = build_session_chunk_prompt(i, chunk.chapter.as_deref(), &summary);

            messages.push(ApiMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            });
            estimated_tokens += estimate_tokens(&prompt);

            if self.config.debug {
                tracing::info!(
                    chunk = i + 1,
                    total = chunks.len(),
                    est_tokens = estimated_tokens,
                    messages_len = messages.len(),
                    prompt_len = prompt.len(),
                    "DEBUG: sending extraction request for chunk"
                );
            } else {
                tracing::debug!(
                    chunk = i + 1,
                    total = chunks.len(),
                    est_tokens = estimated_tokens,
                    messages_len = messages.len(),
                    "SingleSession: sending extraction request for chunk"
                );
            }
            let chunk_start = std::time::Instant::now();
            let response = match session.send_session_messages(&messages) {
                Ok(r) => {
                    if self.config.debug {
                        let preview: String = r.chars().take(500).collect();
                        tracing::info!(
                            chunk = i + 1,
                            duration_ms = chunk_start.elapsed().as_millis() as u64,
                            response_len = r.len(),
                            response_preview = %preview,
                            "DEBUG: chunk extraction response"
                        );
                    } else {
                        tracing::info!(
                            chunk = i + 1,
                            duration_ms = chunk_start.elapsed().as_millis(),
                            response_len = r.len(),
                            "SingleSession: chunk extraction response received"
                        );
                    }
                    r
                }
                Err(e) => {
                    let fail_ms = chunk_start.elapsed().as_millis() as u64;
                    tracing::error!(
                        chunk = i + 1,
                        error = %e,
                        elapsed_ms = fail_ms,
                        "SingleSession: chunk extraction FAILED"
                    );
                    self.store_llm_log(
                        i,
                        1,
                        0,
                        &Some(crate::ingestion::llm::RawLlmExchange {
                            system_prompt: "[session extraction]".into(),
                            user_prompt: prompt.clone(),
                            raw_response: String::new(),
                            retry_prompt: None,
                            retry_response: None,
                            parse_error: Some(e.to_string()),
                            duration_ms: fail_ms,
                            model: Some(model_name.clone()),
                            endpoint: Some("session".into()),
                        }),
                        None,
                        None,
                    );
                    report
                        .errors
                        .push(format!("Chunk {} session call failed: {}", i, e));
                    // Remove the failed user message to keep conversation clean
                    messages.pop();
                    continue;
                }
            };
            estimated_tokens += estimate_tokens(&response);
            messages.push(ApiMessage {
                role: "assistant".to_string(),
                content: response.clone(),
            });

            // Parse extraction
            let extraction = match parse_llm_response(&response) {
                Ok(mut ext) => {
                    repair_extraction(&mut ext);
                    ext
                }
                Err(e) => {
                    // In-session repair
                    let repair_msg = format!(
                        "JSON parse error for CHUNK {}: {}. Please return the corrected JSON.",
                        i + 1,
                        e
                    );
                    messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: repair_msg,
                    });
                    match session.send_session_messages(&messages) {
                        Ok(retry) => {
                            estimated_tokens += estimate_tokens(&retry);
                            messages.push(ApiMessage {
                                role: "assistant".to_string(),
                                content: retry.clone(),
                            });
                            match parse_llm_response(&retry) {
                                Ok(mut ext) => {
                                    repair_extraction(&mut ext);
                                    ext
                                }
                                Err(e2) => {
                                    report.errors.push(format!(
                                        "Chunk {} extraction failed after retry: {}",
                                        i, e2
                                    ));
                                    continue;
                                }
                            }
                        }
                        Err(e2) => {
                            report
                                .errors
                                .push(format!("Chunk {} repair call failed: {}", i, e2));
                            continue;
                        }
                    }
                }
            };

            let chunk_duration_ms = chunk_start.elapsed().as_millis() as u64;
            // Log the session extraction call
            self.store_llm_log(
                i,
                1,
                0,
                &Some(crate::ingestion::llm::RawLlmExchange {
                    system_prompt: "[session extraction]".into(),
                    user_prompt: prompt,
                    raw_response: response.chars().take(2000).collect(),
                    retry_prompt: None,
                    retry_response: None,
                    parse_error: None,
                    duration_ms: chunk_duration_ms,
                    model: Some(model_name.clone()),
                    endpoint: Some("session".into()),
                }),
                Some(&extraction),
                None,
            );

            let n_ent = extraction.entities.len();
            let n_sit = extraction.situations.len();
            for e in &extraction.entities {
                if !entity_names.iter().any(|n| n.eq_ignore_ascii_case(&e.name)) {
                    entity_names.push(e.name.clone());
                }
            }
            for s in &extraction.situations {
                let name = s.name.clone().unwrap_or_else(|| s.description.clone());
                situation_summaries.push((i, name));
            }
            all_extractions[i] = Some((extraction, None));

            self.update_progress(|p| {
                p.current_chunk = i + 1;
                p.session_turn = messages.len();
                p.session_tokens_used = estimated_tokens;
                p.entities_so_far += n_ent;
                p.situations_so_far += n_sit;
                p.session_status = Some(format!(
                    "Extracted chunk {}/{} — {} entities, {} situations total (~{} tokens)",
                    i + 1,
                    chunks.len(),
                    entity_names.len(),
                    situation_summaries.len(),
                    estimated_tokens
                ));
            });
        }

        // Step 4: Enrichment (if enabled)
        let has_extractions = all_extractions.iter().any(|e| e.is_some());
        if self.config.enrich && has_extractions {
            self.update_progress(|p| p.phase = IngestionPhase::Enriching);

            // Switch to enrichment system context
            let enrich_switch = "Now switch to enrichment mode. For each chunk I specify, produce deep annotations: \
                entity_beliefs, game_structures, discourse, info_sets, extra_causal_links, outcomes, temporal_chain, temporal_normalizations. \
                Return ONLY the enrichment JSON.";
            messages.push(ApiMessage {
                role: "user".to_string(),
                content: enrich_switch.to_string(),
            });
            estimated_tokens += estimate_tokens(enrich_switch);

            if let Ok(ack) = session.send_session_messages(&messages) {
                estimated_tokens += estimate_tokens(&ack);
                messages.push(ApiMessage {
                    role: "assistant".to_string(),
                    content: ack,
                });
            }

            for (i, chunk) in chunks.iter().enumerate() {
                if self.is_cancelled() || estimated_tokens > ctx_limit * 85 / 100 {
                    break;
                }
                // Skip chunks that had no successful extraction
                if all_extractions[i].is_none() {
                    continue;
                }

                self.update_progress(|p| {
                    p.current_chunk = i + 1;
                    p.session_status =
                        Some(format!("Enriching chunk {}/{}...", i + 1, chunks.len()));
                });

                let prompt = build_session_enrichment_prompt(i, chunk.chapter.as_deref());
                messages.push(ApiMessage {
                    role: "user".to_string(),
                    content: prompt.clone(),
                });
                estimated_tokens += estimate_tokens(&prompt);

                match session.send_session_messages(&messages) {
                    Ok(response) => {
                        estimated_tokens += estimate_tokens(&response);
                        messages.push(ApiMessage {
                            role: "assistant".to_string(),
                            content: response.clone(),
                        });
                        if let Ok(mut enrichment) = parse_enrichment_response(&response) {
                            repair_enrichment(&mut enrichment);
                            if let Some(ref mut entry) = all_extractions[i] {
                                entry.1 = Some(enrichment);
                            }
                        }
                    }
                    Err(e) => {
                        report
                            .errors
                            .push(format!("Chunk {} enrichment failed: {}", i, e));
                        messages.pop(); // remove failed user message
                    }
                }
            }
        }

        let successful_count = all_extractions.iter().filter(|e| e.is_some()).count();
        if successful_count > 1 {
            self.update_progress(|p| {
                p.phase = IngestionPhase::Reconciling;
                p.session_status = Some(
                    "Running final reconciliation (entity merges, timeline, causal links)..."
                        .into(),
                );
            });

            let recon_prompt = build_session_reconciliation_prompt(successful_count);
            messages.push(ApiMessage {
                role: "user".to_string(),
                content: recon_prompt,
            });

            if let Ok(response) = session.send_session_messages(&messages) {
                messages.push(ApiMessage {
                    role: "assistant".to_string(),
                    content: response.clone(),
                });
                if let Ok(recon) = parse_session_reconciliation_response(&response) {
                    // Apply entity merges: rename duplicates in extractions before committing
                    for merge in &recon.entity_merges {
                        for entry in all_extractions.iter_mut().flatten() {
                            for e in &mut entry.0.entities {
                                if merge
                                    .duplicate_names
                                    .iter()
                                    .any(|d| d.eq_ignore_ascii_case(&e.name))
                                {
                                    e.aliases.push(e.name.clone());
                                    e.name = merge.canonical_name.clone();
                                }
                            }
                            for p in &mut entry.0.participations {
                                if merge
                                    .duplicate_names
                                    .iter()
                                    .any(|d| d.eq_ignore_ascii_case(&p.entity_name))
                                {
                                    p.entity_name = merge.canonical_name.clone();
                                }
                            }
                        }
                    }
                    // Apply confidence adjustments
                    for adj in &recon.confidence_adjustments {
                        let lower = adj.name.to_lowercase();
                        for entry in all_extractions.iter_mut().flatten() {
                            for e in &mut entry.0.entities {
                                if e.name.to_lowercase() == lower {
                                    e.confidence = adj.adjusted_confidence;
                                }
                            }
                        }
                    }
                    // Apply anaphora resolutions: add Participations that the
                    // per-chunk extraction couldn't infer because the referent
                    // was only identifiable by looking across chunks (e.g. "a
                    // figure" in chunk 3 revealed to be the Electric Monk in
                    // chunk 7). For each resolution, find the situation by
                    // substring match on its name/description, and append a
                    // new ExtractedParticipation pointing at that situation.
                    // The entity must already exist in the extraction; if not,
                    // we add it as a low-confidence stub so the normal pipeline
                    // creates it.
                    let mut anaphora_applied: usize = 0;
                    for res in &recon.anaphora_resolutions {
                        let target_lower = res.situation_name.to_lowercase();
                        if target_lower.is_empty() || res.resolved_entity.is_empty() {
                            continue;
                        }
                        let role = res
                            .role
                            .as_deref()
                            .map(crate::ingestion::extraction::parse_role_lenient)
                            .unwrap_or(crate::types::Role::Witness);
                        // Search all extractions for a matching situation.
                        for entry in all_extractions.iter_mut().flatten() {
                            // Ensure the resolved entity exists in this chunk;
                            // if not, add a minimal stub.
                            let entity_exists = entry.0.entities.iter().any(|e| {
                                e.name.eq_ignore_ascii_case(&res.resolved_entity)
                                    || e.aliases
                                        .iter()
                                        .any(|a| a.eq_ignore_ascii_case(&res.resolved_entity))
                            });
                            // Find matching situations in this chunk by name or description.
                            let matching_indices: Vec<usize> = entry
                                .0
                                .situations
                                .iter()
                                .enumerate()
                                .filter_map(|(i, s)| {
                                    let name_match = s
                                        .name
                                        .as_deref()
                                        .map(|n| n.to_lowercase().contains(&target_lower))
                                        .unwrap_or(false);
                                    let desc_match =
                                        s.description.to_lowercase().contains(&target_lower);
                                    if name_match || desc_match {
                                        Some(i)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if matching_indices.is_empty() {
                                continue;
                            }
                            if !entity_exists {
                                entry.0.entities.push(
                                    crate::ingestion::extraction::ExtractedEntity {
                                        name: res.resolved_entity.clone(),
                                        aliases: Vec::new(),
                                        entity_type: crate::types::EntityType::Actor,
                                        properties: serde_json::json!({}),
                                        confidence: res.confidence * 0.8,
                                    },
                                );
                            }
                            for sit_idx in matching_indices {
                                // Avoid duplicating an existing participation
                                // for the same (entity, situation) pair.
                                let already = entry.0.participations.iter().any(|p| {
                                    p.entity_name.eq_ignore_ascii_case(&res.resolved_entity)
                                        && p.situation_index == sit_idx
                                });
                                if already {
                                    continue;
                                }
                                entry.0.participations.push(
                                    crate::ingestion::extraction::ExtractedParticipation {
                                        entity_name: res.resolved_entity.clone(),
                                        situation_index: sit_idx,
                                        role: role.clone(),
                                        action: res.action.clone(),
                                        confidence: res.confidence,
                                    },
                                );
                                anaphora_applied += 1;
                            }
                        }
                    }
                    if anaphora_applied > 0 {
                        self.update_progress(|p| {
                            p.session_status = Some(format!(
                                "Applied {} anaphora resolutions from reconciliation",
                                anaphora_applied
                            ));
                        });
                    }
                    // Store timeline and causal links for post-commit processing
                    report.session_reconciliation = Some(Box::new(recon));
                }
            }
        }

        // Step 6: Process extractions through existing pipeline
        self.update_progress(|p| p.phase = IngestionPhase::Processing);
        let mut resolver = crate::ingestion::resolve::EntityResolver::new();
        self.bootstrap_resolver(&mut resolver);
        let provenance = SourceReference {
            source_type: self.config.source_type.clone(),
            source_id: Some(self.config.source_id.clone()),
            description: Some(source_name.to_string()),
            timestamp: Utc::now(),
            registered_source: None,
        };

        // Process each extraction (skip chunks that had no successful extraction)
        for (chunk_idx, entry) in all_extractions.iter().enumerate() {
            let (extraction, enrichment) = match entry {
                Some(e) => e,
                None => continue, // extraction failed for this chunk
            };
            if self.is_cancelled() {
                report.cancelled = true;
                break;
            }
            let chunk_id = chunk_id_map[chunk_idx];
            let byte_range = chunks
                .get(chunk_idx)
                .map(|c| (c.start_offset, c.end_offset));
            let chunk_text = chunks.get(chunk_idx).map(|c| c.text.as_str());
            let chunk_report = self.process_extraction(
                extraction,
                chunk_idx,
                chunk_id,
                byte_range,
                chunk_text,
                enrichment.as_ref(),
                &mut resolver,
                &provenance,
            );
            match chunk_report {
                Ok(gate_report) => {
                    // Store chunk extraction record for session mode (same as regular mode)
                    let entity_map: std::collections::HashMap<String, Uuid> = extraction
                        .entities
                        .iter()
                        .zip(gate_report.entity_ids.iter())
                        .map(|(e, id)| (e.name.clone(), *id))
                        .collect();
                    self.store_chunk_extraction_record(
                        chunk_idx,
                        extraction,
                        Vec::new(),
                        &entity_map,
                        &gate_report.situation_ids,
                    );
                    report
                        .created_entity_ids
                        .extend_from_slice(&gate_report.entity_ids);
                    report
                        .created_situation_ids
                        .extend_from_slice(&gate_report.situation_ids);
                    report.entities_created += gate_report.entity_ids.len();
                    report.situations_created += gate_report.situation_ids.len();
                    report.items_auto_committed += gate_report.auto_committed;
                    report.items_queued += gate_report.queued;
                    report.items_rejected += gate_report.rejected;
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("Chunk {} processing failed: {}", chunk_idx, e));
                }
            }
        }

        report.entity_resolutions = resolver.len();
        report.duration_secs = start.elapsed().as_secs_f64();
        self.update_progress(|p| p.phase = IngestionPhase::Complete);
        Ok(report)
    }

    /// Build a compact accumulator summary for injection into session prompts.
    fn build_accumulator_summary(
        entity_names: &[String],
        situation_summaries: &[(usize, String)],
    ) -> String {
        if entity_names.is_empty() && situation_summaries.is_empty() {
            return String::new();
        }
        let mut summary = String::from("[ACCUMULATED SO FAR]\n");
        if !entity_names.is_empty() {
            summary.push_str("Entities: ");
            summary.push_str(&entity_names.join(", "));
            summary.push('\n');
        }
        if !situation_summaries.is_empty() {
            // Show last 20 situations to keep it compact
            let recent: Vec<_> = situation_summaries.iter().rev().take(20).rev().collect();
            summary.push_str("Recent situations:\n");
            for (chunk_idx, name) in recent {
                let short = if name.len() > 60 { &name[..60] } else { name };
                summary.push_str(&format!("  chunk {}: {}\n", chunk_idx + 1, short));
            }
        }
        summary.push_str("[END ACCUMULATED]\n");
        summary
    }

    /// Case-insensitive entity name lookup in the entity map.
    fn resolve_entity_name(
        entity_map: &std::collections::HashMap<String, Uuid>,
        name: &str,
    ) -> Option<Uuid> {
        entity_map.get(name).copied().or_else(|| {
            let lower = name.to_lowercase();
            entity_map
                .iter()
                .find(|(k, _)| k.to_lowercase() == lower)
                .map(|(_, &v)| v)
        })
    }

    /// Apply step-2 enrichment data to already-committed entities and situations.
    fn apply_enrichment(
        &self,
        enrichment: &crate::ingestion::extraction::ExtractionEnrichment,
        entity_map: &std::collections::HashMap<String, Uuid>,
        situation_ids: &[Uuid],
    ) -> Result<()> {
        use crate::types::*;

        for eb in &enrichment.entity_beliefs {
            let entity_id = Self::resolve_entity_name(entity_map, &eb.entity_name);
            if let Some(id) = entity_id {
                let beliefs = serde_json::json!({
                    "beliefs": eb.beliefs,
                    "goals": eb.goals,
                    "misconceptions": eb.misconceptions,
                });
                if let Err(e) = self.hypergraph.update_entity_no_snapshot(&id, |entity| {
                    entity.beliefs = Some(beliefs);
                }) {
                    tracing::warn!(entity_id = %id, error = %e, "Failed to persist entity beliefs enrichment");
                }
            }
        }

        for gs in &enrichment.game_structures {
            if let Some(&sit_id) = situation_ids.get(gs.situation_index) {
                let game_type: GameClassification = gs
                    .game_type
                    .parse()
                    .unwrap_or(GameClassification::Custom(gs.game_type.clone()));
                let info_structure: InfoStructureType = gs
                    .info_structure
                    .parse()
                    .unwrap_or(InfoStructureType::Custom(gs.info_structure.clone()));
                let description = gs.description.clone();
                if let Err(e) = self.hypergraph.update_situation(&sit_id, |sit| {
                    sit.game_structure = Some(GameStructure {
                        game_type,
                        info_structure,
                        description,
                        maturity: MaturityLevel::Candidate,
                    });
                }) {
                    tracing::warn!(situation_id = %sit_id, error = %e, "Failed to persist game_structure enrichment");
                }
            }
        }

        for da in &enrichment.discourse {
            if let Some(&sit_id) = situation_ids.get(da.situation_index) {
                // Resolve focalization entity name → UUID
                let focalization_id = da
                    .focalization
                    .as_ref()
                    .and_then(|name| Self::resolve_entity_name(entity_map, name));
                let order = da.order.clone();
                let duration = da.duration.clone();
                let voice = da.voice.clone();
                if let Err(e) = self.hypergraph.update_situation(&sit_id, |sit| {
                    sit.discourse = Some(DiscourseAnnotation {
                        order,
                        duration,
                        focalization: focalization_id,
                        voice,
                    });
                }) {
                    tracing::warn!(situation_id = %sit_id, error = %e, "Failed to persist discourse enrichment");
                }
            }
        }

        for oe in &enrichment.outcomes {
            if let Some(&sit_id) = situation_ids.get(oe.situation_index) {
                let det = oe.deterministic.clone();
                let alts: Vec<_> = oe.alternatives.iter().map(|a| {
                    serde_json::json!({"description": a.description, "probability": a.probability})
                }).collect();
                if let Err(e) = self.hypergraph.update_situation(&sit_id, |sit| {
                    if let Some(ref d) = det {
                        sit.deterministic = Some(serde_json::json!({"outcome": d}));
                    }
                    if !alts.is_empty() {
                        sit.probabilistic = Some(serde_json::json!({"alternatives": alts}));
                    }
                }) {
                    tracing::warn!(situation_id = %sit_id, error = %e, "Failed to persist outcomes enrichment");
                }
            }
        }

        for is in &enrichment.info_sets {
            let entity_id = Self::resolve_entity_name(entity_map, &is.entity_name);
            if let (Some(eid), Some(&sid)) = (entity_id, situation_ids.get(is.situation_index)) {
                // Get existing participations and update the first matching one
                if let Ok(parts) = self.hypergraph.get_participations_for_pair(&eid, &sid) {
                    if let Some(mut part) = parts.into_iter().next() {
                        let null_uuid = Uuid::nil();
                        let info_set = InfoSet {
                            knows_before: is
                                .knows_before
                                .iter()
                                .map(|f| KnowledgeFact {
                                    about_entity: null_uuid,
                                    fact: f.clone(),
                                    confidence: 0.7,
                                })
                                .collect(),
                            learns: is
                                .learns
                                .iter()
                                .map(|f| KnowledgeFact {
                                    about_entity: null_uuid,
                                    fact: f.clone(),
                                    confidence: 0.7,
                                })
                                .collect(),
                            reveals: is
                                .reveals
                                .iter()
                                .map(|f| KnowledgeFact {
                                    about_entity: null_uuid,
                                    fact: f.clone(),
                                    confidence: 0.7,
                                })
                                .collect(),
                            beliefs_about_others: vec![],
                        };
                        part.info_set = Some(info_set);
                        // Re-add with updated info_set (remove + add)
                        if let Err(e) =
                            self.hypergraph
                                .remove_participant(&eid, &sid, Some(part.seq))
                        {
                            tracing::warn!(entity_id = %eid, situation_id = %sid, error = %e, "Failed to remove participant for info_set update");
                        }
                        if let Err(e) = self.hypergraph.add_participant(part) {
                            tracing::warn!(entity_id = %eid, situation_id = %sid, error = %e, "Failed to persist info_set enrichment");
                        }
                    }
                }
            }
        }

        for tn in &enrichment.temporal_normalizations {
            if let Some(&sit_id) = situation_ids.get(tn.situation_index) {
                if let Some(ref date_str) = tn.normalized_date {
                    // Try parsing the normalized date
                    let parsed = chrono::DateTime::parse_from_rfc3339(date_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                        .or_else(|| {
                            chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                                .ok()
                                .and_then(|d| d.and_hms_opt(0, 0, 0))
                                .map(|dt| {
                                    chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc)
                                })
                        });
                    if let Some(dt) = parsed {
                        // Fuzzy Sprint Phase 5 — detect fuzziness cues in
                        // the relative_description and widen the crisp
                        // point into a trapezoidal window. Falls through
                        // harmlessly (None) when no cue fires.
                        let fuzzy_cue = tn.relative_description.as_deref().unwrap_or("");
                        let fuzzy_endpoints = if !fuzzy_cue.is_empty() {
                            crate::fuzzy::allen::fuzzy_from_marker(dt, dt, fuzzy_cue)
                        } else {
                            None
                        };
                        let granularity_new = if date_str.contains('T') {
                            TimeGranularity::Exact
                        } else {
                            TimeGranularity::Day
                        };
                        if let Err(e) = self.hypergraph.update_situation(&sit_id, |sit| {
                            sit.temporal.start = Some(dt);
                            sit.temporal.end = Some(dt);
                            sit.temporal.granularity = granularity_new;
                            if let Some(fe) = fuzzy_endpoints {
                                sit.temporal.fuzzy_endpoints = Some(fe);
                            }
                        }) {
                            tracing::warn!(situation_id = %sit_id, error = %e, "Failed to persist temporal normalization");
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Commit an extracted entity to the hypergraph.
    /// Returns (id, is_new) — is_new is false if entity was resolved to existing.
    fn commit_entity(
        &self,
        extracted: &crate::ingestion::extraction::ExtractedEntity,
        resolver: &mut EntityResolver,
        provenance: &SourceReference,
    ) -> Result<(Uuid, bool)> {
        // Check for existing entity via resolver
        let resolve_result = resolver.resolve(extracted, self.embedder.as_deref());

        match resolve_result {
            ResolveResult::Existing(id) => {
                // The in-memory resolver already tracks the alias set, so
                // extending it tells us whether this mention contributes any
                // alias the entity doesn't already have. Combined with a
                // bio-key check on the incoming payload, this avoids the
                // round-trip through `update_entity_no_snapshot` (read →
                // deserialize → re-serialize → write → reindex) for the
                // overwhelming common case where a later chunk just re-mentions
                // a known character with nothing new.
                let mut mention_aliases: Vec<String> =
                    Vec::with_capacity(extracted.aliases.len() + 1);
                if !extracted.name.trim().is_empty() {
                    mention_aliases.push(extracted.name.clone());
                }
                mention_aliases.extend(extracted.aliases.iter().cloned());
                let added = resolver.extend_known_aliases(id, &mention_aliases);
                if added.is_empty() && !has_mergeable_bio_keys(&extracted.properties) {
                    return Ok((id, false));
                }
                let updated =
                    self.hypergraph
                        .update_entity_no_snapshot(&id, |entity: &mut Entity| {
                            merge_into_properties(
                                &mut entity.properties,
                                &extracted.name,
                                &mention_aliases,
                                &extracted.properties,
                            );
                        });
                match updated {
                    Ok(_) => {}
                    Err(crate::error::TensaError::EntityNotFound(_)) => {
                        tracing::warn!(
                            entity_id = %id,
                            "resolver hit a deleted entity; alias/bio merge skipped"
                        );
                    }
                    Err(e) => return Err(e),
                }
                Ok((id, false))
            }
            ResolveResult::New => {
                let now = Utc::now();
                let embedding = self
                    .embedder
                    .as_ref()
                    .and_then(|e| e.embed_text(&extracted.name).ok());

                let mut props = extracted.properties.clone();
                merge_into_properties(
                    &mut props,
                    &extracted.name,
                    &extracted.aliases,
                    &serde_json::Value::Null,
                );

                let entity = Entity {
                    id: Uuid::now_v7(),
                    entity_type: extracted.entity_type.clone(),
                    properties: props,
                    beliefs: None,
                    embedding: embedding.clone(),
                    maturity: MaturityLevel::Candidate,
                    confidence: extracted.confidence,
                    confidence_breakdown: None,
                    provenance: vec![provenance.clone()],
                    extraction_method: None,
                    narrative_id: self.config.narrative_id.clone(),
                    created_at: now,
                    updated_at: now,
                    deleted_at: None,
                    transaction_time: None,
                };

                let id = self.hypergraph.create_entity(entity)?;

                // Register in resolver
                resolver.register(
                    id,
                    &extracted.name,
                    &extracted.aliases,
                    extracted.entity_type.clone(),
                    embedding.clone(),
                );

                // Add to vector index
                if let (Some(vi), Some(emb)) = (&self.vector_index, &embedding) {
                    if let Ok(mut idx) = vi.write() {
                        let _ = idx.add(id, emb);
                    }
                }

                Ok((id, true))
            }
        }
    }

    /// Commit an extracted situation to the hypergraph.
    fn commit_situation(
        &self,
        extracted: &crate::ingestion::extraction::ExtractedSituation,
        _provenance: &SourceReference,
        chunk_id: Option<Uuid>,
        source_span: Option<SourceSpan>,
    ) -> Result<Uuid> {
        let now = Utc::now();

        let mut raw_content = if extracted.content_blocks.is_empty() {
            vec![ContentBlock::text(&extracted.description)]
        } else {
            extracted.content_blocks.clone()
        };

        let embedding = self
            .embedder
            .as_ref()
            .and_then(|e| e.embed_text(&extracted.description).ok());

        // Try to parse temporal_marker into a timestamp; otherwise store as description
        let (temporal_start, temporal_granularity, temporal_description) =
            match &extracted.temporal_marker {
                Some(marker) if !marker.is_empty() => {
                    // Try RFC 3339 or date-only parsing
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(marker) {
                        (Some(dt.with_timezone(&Utc)), TimeGranularity::Exact, None)
                    } else if let Ok(date) = chrono::NaiveDate::parse_from_str(marker, "%Y-%m-%d") {
                        let naive_dt = date.and_hms_opt(0, 0, 0).unwrap();
                        let dt = chrono::DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc);
                        (Some(dt), TimeGranularity::Day, None)
                    } else {
                        // Store as description for relative markers ("Chapter 3", "evening", etc.)
                        (None, TimeGranularity::Unknown, Some(marker.clone()))
                    }
                }
                _ => (None, TimeGranularity::Unknown, None),
            };

        // If we have a temporal description but couldn't parse a date, prepend it to content
        if let Some(ref marker) = temporal_description {
            raw_content.insert(
                0,
                ContentBlock {
                    content_type: crate::types::ContentType::Observation,
                    content: format!("[Time: {}]", marker),
                    source: None,
                },
            );
        }

        let situation = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: extracted.name.clone(),
            description: Some(extracted.description.clone()),
            temporal: AllenInterval {
                start: temporal_start,
                end: temporal_start, // same as start for point-in-time events
                granularity: temporal_granularity,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: extracted.location.as_ref().map(|loc| SpatialAnchor {
                latitude: None,
                longitude: None,
                precision: SpatialPrecision::Unknown,
                location_entity: None,
                location_name: Some(loc.clone()),
                description: Some(loc.clone()),
                geo_provenance: None,
            }),
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: embedding.clone(),
            raw_content,
            narrative_level: extracted.narrative_level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: extracted.confidence,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: self.config.narrative_id.clone(),
            source_chunk_id: chunk_id,
            source_span,
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

        let id = self.hypergraph.create_situation(situation)?;

        // Add to vector index
        if let (Some(vi), Some(emb)) = (&self.vector_index, &embedding) {
            if let Ok(mut idx) = vi.write() {
                let _ = idx.add(id, emb);
            }
        }

        Ok(id)
    }

    /// Queue an item for human review.
    fn queue_item(
        &self,
        item_type: QueueItemType,
        data: serde_json::Value,
        chunk_idx: usize,
        chunk_id: Option<Uuid>,
        confidence: f32,
    ) -> Result<Uuid> {
        let item = ValidationQueueItem {
            id: Uuid::now_v7(),
            item_type,
            extracted_data: data,
            source_text: String::new(),
            chunk_id: chunk_idx,
            source_chunk_id: chunk_id,
            narrative_id: self.config.narrative_id.clone(),
            confidence,
            status: QueueItemStatus::Pending,
            reviewer_notes: None,
            created_at: Utc::now(),
            reviewed_at: None,
        };
        self.queue.enqueue(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::chunker::TextChunk;
    use crate::ingestion::embed::HashEmbedding;
    use crate::ingestion::extraction::*;
    use crate::store::memory::MemoryStore;

    /// Mock extractor that returns canned responses.
    struct MockExtractor {
        response: NarrativeExtraction,
    }

    impl NarrativeExtractor for MockExtractor {
        fn extract_narrative(&self, _chunk: &TextChunk) -> Result<NarrativeExtraction> {
            Ok(self.response.clone())
        }
    }

    fn sample_extraction() -> NarrativeExtraction {
        NarrativeExtraction {
            entities: vec![
                ExtractedEntity {
                    name: "Raskolnikov".to_string(),
                    aliases: vec!["Rodion".to_string(), "Rodya".to_string()],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"age": 23}),
                    confidence: 0.95,
                },
                ExtractedEntity {
                    name: "Sonya".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.9,
                },
                ExtractedEntity {
                    name: "Unclear Person".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.5, // medium — goes to queue
                },
                ExtractedEntity {
                    name: "Ghost".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.1, // low — rejected
                },
            ],
            situations: vec![ExtractedSituation {
                name: Some("Raskolnikov confesses to Sonya".to_string()),
                description: "Raskolnikov confesses to Sonya".to_string(),
                temporal_marker: Some("evening".to_string()),
                location: Some("Sonya's room".to_string()),
                narrative_level: NarrativeLevel::Scene,
                content_blocks: vec![],
                confidence: 0.85,
                text_start: None,
                text_end: None,
            }],
            participations: vec![
                ExtractedParticipation {
                    entity_name: "Raskolnikov".to_string(),
                    situation_index: 0,
                    role: Role::Protagonist,
                    action: Some("confesses".to_string()),
                    confidence: 0.9,
                },
                ExtractedParticipation {
                    entity_name: "Sonya".to_string(),
                    situation_index: 0,
                    role: Role::Witness,
                    action: Some("listens".to_string()),
                    confidence: 0.88,
                },
            ],
            causal_links: vec![],
            temporal_relations: vec![],
        }
    }

    fn setup_pipeline(
        extraction: NarrativeExtraction,
    ) -> (IngestionPipeline, Arc<Hypergraph>, Arc<ValidationQueue>) {
        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store.clone()));
        let queue = Arc::new(ValidationQueue::new(store.clone()));
        let extractor = Arc::new(MockExtractor {
            response: extraction,
        });
        let embedder: Option<Arc<dyn EmbeddingProvider>> = Some(Arc::new(HashEmbedding::new(64)));
        let vector_index = Some(Arc::new(std::sync::RwLock::new(VectorIndex::new(64))));

        let config = PipelineConfig {
            chunker: ChunkerConfig {
                max_tokens: 5000,
                overlap_tokens: 0,
                chapter_regex: None,
                ..Default::default()
            },
            auto_commit_threshold: 0.8,
            review_threshold: 0.3,
            source_id: "test".to_string(),
            source_type: "book".to_string(),
            ..Default::default()
        };

        let pipeline = IngestionPipeline::new(
            hg.clone(),
            extractor,
            embedder,
            vector_index,
            queue.clone(),
            config,
        );
        (pipeline, hg, queue)
    }

    #[test]
    fn test_pipeline_ingest_text() {
        let (pipeline, _hg, _queue) = setup_pipeline(sample_extraction());
        let report = pipeline
            .ingest_text("He confessed everything to her that evening.", "test")
            .unwrap();

        assert_eq!(report.chunks_processed, 1);
        // 2 high-confidence entities auto-committed
        assert!(report.entities_created >= 2);
        // 1 situation auto-committed
        assert!(report.situations_created >= 1);
        // medium-confidence entity queued
        assert!(report.items_queued >= 1);
        // low-confidence entity rejected
        assert!(report.items_rejected >= 1);
    }

    #[test]
    fn test_pipeline_empty_text() {
        let (pipeline, _, _) = setup_pipeline(sample_extraction());
        let report = pipeline.ingest_text("", "test").unwrap();
        assert_eq!(report.chunks_processed, 0);
        assert_eq!(report.entities_created, 0);
    }

    #[test]
    fn test_pipeline_entity_resolution() {
        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store.clone()));
        let queue = Arc::new(ValidationQueue::new(store.clone()));

        // Extraction where "Rodya" appears (alias of Raskolnikov)
        let extraction = NarrativeExtraction {
            entities: vec![
                ExtractedEntity {
                    name: "Raskolnikov".to_string(),
                    aliases: vec!["Rodya".to_string()],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.9,
                },
                ExtractedEntity {
                    name: "Rodya".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.9,
                },
            ],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            temporal_relations: vec![],
        };

        let extractor = Arc::new(MockExtractor {
            response: extraction,
        });

        let config = PipelineConfig {
            chunker: ChunkerConfig {
                max_tokens: 5000,
                overlap_tokens: 0,
                chapter_regex: None,
                ..Default::default()
            },
            auto_commit_threshold: 0.8,
            review_threshold: 0.3,
            source_id: "test".to_string(),
            source_type: "book".to_string(),
            ..Default::default()
        };

        let pipeline = IngestionPipeline::new(hg.clone(), extractor, None, None, queue, config);

        let report = pipeline.ingest_text("Rodya went out.", "test").unwrap();

        // Only 1 entity created (Rodya resolved to Raskolnikov)
        assert_eq!(report.entities_created, 1);
    }

    #[test]
    fn test_pipeline_bootstrap_dedupes_across_runs() {
        // Ingesting the same actor twice in two separate runs must not create
        // duplicate entities for the same narrative_id.
        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store.clone()));
        let queue = Arc::new(ValidationQueue::new(store.clone()));

        let extraction = NarrativeExtraction {
            entities: vec![ExtractedEntity {
                name: "Ana Stojanović".to_string(),
                aliases: vec![],
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.99,
            }],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            temporal_relations: vec![],
        };

        let config = PipelineConfig {
            chunker: ChunkerConfig {
                max_tokens: 5000,
                overlap_tokens: 0,
                chapter_regex: None,
                ..Default::default()
            },
            auto_commit_threshold: 0.8,
            review_threshold: 0.3,
            source_id: "src-1".to_string(),
            source_type: "book".to_string(),
            narrative_id: Some("campaign-a".into()),
            ..Default::default()
        };

        let make_pipeline = || {
            IngestionPipeline::new(
                hg.clone(),
                Arc::new(MockExtractor {
                    response: extraction.clone(),
                }),
                None,
                None,
                queue.clone(),
                config.clone(),
            )
        };

        let r1 = make_pipeline().ingest_text("Run 1.", "test").unwrap();
        assert_eq!(r1.entities_created, 1);

        let r2 = make_pipeline().ingest_text("Run 2.", "test").unwrap();
        assert_eq!(r2.entities_created, 0, "second run must resolve to first");

        let all = hg.list_entities_by_narrative("campaign-a").unwrap();
        assert_eq!(
            all.len(),
            1,
            "hypergraph should hold a single canonical entity"
        );
    }

    #[test]
    fn test_pipeline_bootstrap_dedupes_across_project_siblings() {
        // Two narratives in the same project; an entity ingested into narrative A
        // should be resolved when re-ingested into narrative B.
        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store.clone()));
        let queue = Arc::new(ValidationQueue::new(store.clone()));

        let registry = crate::narrative::registry::NarrativeRegistry::new(store.clone());
        let mut a = crate::narrative::types::Narrative {
            id: "nar-a".into(),
            title: "A".into(),
            genre: None,
            tags: vec![],
            source: None,
            project_id: Some("shared-proj".into()),
            description: None,
            authors: vec![],
            language: None,
            publication_date: None,
            cover_url: None,
            custom_properties: std::collections::HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let mut b = a.clone();
        b.id = "nar-b".into();
        b.title = "B".into();
        registry.create(a.clone()).unwrap();
        registry.create(b.clone()).unwrap();

        // Pre-existing entity in narrative A
        let seeded = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Ana Stojanović"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.99,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("nar-a".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        a.entity_count = 1;
        let seeded_id = hg.create_entity(seeded).unwrap();

        let extraction = NarrativeExtraction {
            entities: vec![ExtractedEntity {
                name: "Ana Stojanović".to_string(),
                aliases: vec![],
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.99,
            }],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            temporal_relations: vec![],
        };

        let config = PipelineConfig {
            chunker: ChunkerConfig {
                max_tokens: 5000,
                overlap_tokens: 0,
                chapter_regex: None,
                ..Default::default()
            },
            auto_commit_threshold: 0.8,
            review_threshold: 0.3,
            source_id: "src-b".to_string(),
            source_type: "book".to_string(),
            narrative_id: Some("nar-b".into()),
            ..Default::default()
        };

        let pipeline = IngestionPipeline::new(
            hg.clone(),
            Arc::new(MockExtractor {
                response: extraction,
            }),
            None,
            None,
            queue,
            config,
        );

        let report = pipeline.ingest_text("Story B.", "test").unwrap();
        assert_eq!(
            report.entities_created, 0,
            "sibling-narrative dedup via project_id should resolve to existing id"
        );
        assert!(
            hg.get_entity(&seeded_id).is_ok(),
            "original entity should still exist"
        );
    }

    #[test]
    fn test_pipeline_validation_queue_populated() {
        let (pipeline, _, queue) = setup_pipeline(sample_extraction());
        pipeline.ingest_text("Test text.", "test").unwrap();

        let pending = queue.list_pending(10).unwrap();
        assert!(!pending.is_empty(), "Queue should have pending items");
    }

    #[test]
    fn test_pipeline_auto_links_locations_to_situations() {
        let mut extraction = sample_extraction();
        // Add a Location entity matching the situation's location field
        extraction.entities.push(ExtractedEntity {
            name: "Sonya's room".to_string(),
            aliases: vec![],
            entity_type: EntityType::Location,
            properties: serde_json::json!({}),
            confidence: 0.9,
        });

        let (pipeline, hg, _) = setup_pipeline(extraction);
        pipeline.ingest_text("Test text.", "test").unwrap();

        // Find the Location entity
        let all_keys = hg.store().prefix_scan(b"e/").unwrap();
        let mut location_id = None;
        for (_key, value) in &all_keys {
            let entity: Entity = serde_json::from_slice(value).unwrap();
            if entity.entity_type == EntityType::Location {
                location_id = Some(entity.id);
                break;
            }
        }
        let location_id = location_id.expect("Location entity should exist");

        // The location should be linked to the situation as a participant
        let participations = hg.get_situations_for_entity(&location_id).unwrap();
        assert!(
            !participations.is_empty(),
            "Location should be auto-linked to situation via Setting role"
        );
        assert_eq!(
            participations[0].role,
            Role::Custom("Setting".into()),
            "Location participation should have Setting role"
        );
    }

    #[test]
    fn test_insert_chunk_markers() {
        let text = "Chapter One\nHello world.\n\nChapter Two\nGoodbye world.";
        let chunks = vec![
            TextChunk {
                chunk_id: 0,
                text: "Chapter One\nHello world.".to_string(),
                chapter: Some("Chapter One".to_string()),
                start_offset: 0,
                end_offset: 25,
                overlap_prefix: String::new(),
            },
            TextChunk {
                chunk_id: 1,
                text: "Chapter Two\nGoodbye world.".to_string(),
                chapter: Some("Chapter Two".to_string()),
                start_offset: 26,
                end_offset: 52,
                overlap_prefix: String::new(),
            },
        ];
        let marked = IngestionPipeline::insert_chunk_markers(text, &chunks);
        assert!(marked.contains("[=== CHUNK 1: \"Chapter One\" ===]"));
        assert!(marked.contains("[=== END CHUNK 1 ===]"));
        assert!(marked.contains("[=== CHUNK 2: \"Chapter Two\" ===]"));
        assert!(marked.contains("[=== END CHUNK 2 ===]"));
        assert!(marked.contains("Hello world."));
        assert!(marked.contains("Goodbye world."));
    }

    #[test]
    fn test_build_accumulator_summary_empty() {
        let summary = IngestionPipeline::build_accumulator_summary(&[], &[]);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_build_accumulator_summary_with_data() {
        let entities = vec!["Alice".to_string(), "Bob".to_string()];
        let situations = vec![
            (0, "Alice meets Bob".to_string()),
            (1, "Bob leaves".to_string()),
        ];
        let summary = IngestionPipeline::build_accumulator_summary(&entities, &situations);
        assert!(summary.contains("[ACCUMULATED SO FAR]"));
        assert!(summary.contains("Alice, Bob"));
        assert!(summary.contains("Alice meets Bob"));
        assert!(summary.contains("[END ACCUMULATED]"));
    }

    /// Mock session extractor that returns canned JSON responses.
    struct MockSessionExtractor {
        response: NarrativeExtraction,
    }

    impl NarrativeExtractor for MockSessionExtractor {
        fn extract_narrative(&self, _chunk: &TextChunk) -> Result<NarrativeExtraction> {
            Ok(self.response.clone())
        }

        fn as_session(&self) -> Option<&dyn crate::ingestion::llm::SessionCapableExtractor> {
            Some(self)
        }
    }

    impl crate::ingestion::llm::SessionCapableExtractor for MockSessionExtractor {
        fn send_session_messages(
            &self,
            messages: &[crate::ingestion::llm::ApiMessage],
        ) -> Result<String> {
            // Check what kind of request this is based on the last user message
            let last_user = messages.iter().rev().find(|m| m.role == "user");
            if let Some(msg) = last_user {
                if msg.content.contains("Part ") || msg.content.contains("I will send") {
                    // Context loading — acknowledge
                    return Ok("Acknowledged. Ready for next part.".to_string());
                }
                if msg.content.contains("Extract all entities") {
                    // Extraction request — return canned JSON
                    return Ok(serde_json::to_string(&self.response).unwrap());
                }
                if msg.content.contains("enrichment") || msg.content.contains("Enriching") {
                    return Ok(r#"{"entity_beliefs":[],"game_structures":[],"discourse":[],"info_sets":[],"extra_causal_links":[],"outcomes":[],"temporal_chain":[],"temporal_normalizations":[]}"#.to_string());
                }
                if msg.content.contains("reconciliation") {
                    return Ok(r#"{"entity_merges":[],"timeline":[],"confidence_adjustments":[],"cross_chunk_causal_links":[]}"#.to_string());
                }
            }
            Ok("OK".to_string())
        }
    }

    #[test]
    fn test_single_session_ingestion() {
        let response = sample_extraction();
        let mock = MockSessionExtractor { response };
        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store.clone()));
        let queue = Arc::new(ValidationQueue::new(store.clone()));

        let config = PipelineConfig {
            single_session: true,
            session_max_context_tokens: 1_000_000,
            chunker: crate::ingestion::chunker::ChunkerConfig {
                max_tokens: 30, // very small chunks to force splitting
                overlap_tokens: 0,
                chapter_regex: None,
                strategy: crate::ingestion::chunker::ChunkStrategy::FixedSize,
            },
            ..Default::default()
        };

        let pipeline =
            IngestionPipeline::new(hg.clone(), Arc::new(mock), None, None, queue, config);

        let text = "Alice met Bob in the garden on a sunny Tuesday morning. They discussed their plans for the evening in great detail, covering every possible scenario and contingency.\n\n\
                     Bob revealed the truth to Alice about the hidden treasure buried beneath the old oak tree in the north field. She was completely shocked by the unexpected revelation.\n\n\
                     They decided to investigate together and set off toward the old mansion on the hill. The journey took several hours through winding forest paths and overgrown trails.\n\n\
                     At the mansion they found a locked door. Inside they discovered ancient manuscripts describing a lost civilization and maps to further buried treasures across the countryside.";
        let report = pipeline.ingest_text(text, "test-session").unwrap();

        // Should have processed multiple chunks via session path
        assert!(
            report.chunks_processed >= 2,
            "Should have >=2 chunks, got {}",
            report.chunks_processed
        );
        assert!(
            report.entities_created > 0,
            "Should create entities, got {}",
            report.entities_created
        );
        assert!(
            report.situations_created > 0,
            "Should create situations, got {}",
            report.situations_created
        );
        assert!(!report.cancelled, "Should not be cancelled");
    }

    // ─── Cascade mode tests ────────────────────────────────────

    #[test]
    fn test_cascade_mode_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.cascade_mode, CascadeMode::LlmOnly);
    }

    #[test]
    fn test_cascade_extract_llm_only() {
        let mock = MockExtractor {
            response: sample_extraction(),
        };
        let chunk = TextChunk {
            chunk_id: 0,
            text: "Robert Parker arrived at the Grand Hotel.".to_string(),
            chapter: None,
            start_offset: 0,
            end_offset: 41,
            overlap_prefix: String::new(),
        };
        let (result, _) = cascade_extract(&mock, &chunk, &[], &CascadeMode::LlmOnly).unwrap();
        assert_eq!(result.entities.len(), sample_extraction().entities.len());
    }

    #[test]
    fn test_cascade_extract_nlp_only() {
        let mock = MockExtractor {
            response: sample_extraction(),
        };
        let chunk = TextChunk {
            chunk_id: 0,
            text: "Robert Parker arrived at the Grand Hotel in London. Sarah Mitchell was already waiting.".to_string(),
            chapter: None,
            start_offset: 0,
            end_offset: 87,
            overlap_prefix: String::new(),
        };
        let (result, exchange) =
            cascade_extract(&mock, &chunk, &[], &CascadeMode::NlpOnly).unwrap();
        // NLP-only: no LLM exchange
        assert!(exchange.is_none());
        // NLP should find entities from capitalized names
        assert!(!result.entities.is_empty());
    }

    #[test]
    fn test_cascade_extract_nlp_first_sparse_falls_through() {
        let mock = MockExtractor {
            response: sample_extraction(),
        };
        // Text with only 1 capitalized name — NLP won't meet threshold
        let chunk = TextChunk {
            chunk_id: 0,
            text: "A mysterious figure walked through the rain.".to_string(),
            chapter: None,
            start_offset: 0,
            end_offset: 45,
            overlap_prefix: String::new(),
        };
        let (result, _) = cascade_extract(&mock, &chunk, &[], &CascadeMode::NlpFirst).unwrap();
        // Falls through to LLM, gets mock response
        assert_eq!(result.entities.len(), sample_extraction().entities.len());
    }

    #[test]
    fn test_cascade_extract_nlp_first_rich_skips_llm() {
        let mock = MockExtractor {
            response: sample_extraction(),
        };
        // Text with many capitalized names — NLP should meet threshold
        let chunk = TextChunk {
            chunk_id: 0,
            text: "Robert Parker met Sarah Mitchell and David Wilson at the Grand Hotel. James Brown joined them later.".to_string(),
            chapter: None,
            start_offset: 0,
            end_offset: 100,
            overlap_prefix: String::new(),
        };
        let (result, exchange) =
            cascade_extract(&mock, &chunk, &[], &CascadeMode::NlpFirst).unwrap();
        // NLP found enough entities, should skip LLM
        assert!(exchange.is_none());
        assert!(result.entities.len() >= NLP_SKIP_LLM_ENTITY_THRESHOLD);
    }

    #[test]
    fn test_merge_extractions_dedup() {
        let nlp = NarrativeExtraction {
            entities: vec![
                ExtractedEntity {
                    name: "Alice".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.5,
                },
                ExtractedEntity {
                    name: "Bob".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.5,
                },
            ],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            temporal_relations: vec![],
        };
        let llm = NarrativeExtraction {
            entities: vec![ExtractedEntity {
                name: "Alice".to_string(),
                aliases: vec![],
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"role": "protagonist"}),
                confidence: 0.95,
            }],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            temporal_relations: vec![],
        };

        let merged = merge_extractions(nlp, llm);
        // Alice from LLM (higher confidence) + Bob from NLP (unique)
        assert_eq!(merged.entities.len(), 2);
        let names: Vec<&str> = merged.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
        // Alice should be the LLM version (high confidence)
        let alice = merged.entities.iter().find(|e| e.name == "Alice").unwrap();
        assert!((alice.confidence - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_merge_into_properties_new_entity() {
        let mut props = serde_json::json!({});
        let incoming = serde_json::json!({
            "date_of_birth": "1431",
            "place_of_birth": "Transylvania",
            "unrecognized_key": "should be dropped"
        });
        merge_into_properties(
            &mut props,
            "Count Dracula",
            &["Dracula".into(), "Nosferatu".into(), "Vlad Dracula".into()],
            &incoming,
        );
        let obj = props.as_object().unwrap();
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("Count Dracula")
        );
        let aliases: Vec<&str> = obj
            .get("aliases")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(aliases, vec!["Dracula", "Nosferatu", "Vlad Dracula"]);
        assert_eq!(
            obj.get("date_of_birth").and_then(|v| v.as_str()),
            Some("1431")
        );
        assert_eq!(
            obj.get("place_of_birth").and_then(|v| v.as_str()),
            Some("Transylvania")
        );
        // Unrecognized key must not be merged — keeps the LLM from silently
        // rewriting arbitrary fields on resolve-merge.
        assert!(obj.get("unrecognized_key").is_none());
    }

    #[test]
    fn test_merge_into_properties_first_write_wins_and_union() {
        let mut props = serde_json::json!({
            "name": "Dracula",
            "aliases": ["Drac"],
            "date_of_birth": "1431",
        });
        // Later mention: adds a new alias, tries to overwrite DOB, drops a duplicate alias.
        let incoming = serde_json::json!({
            "date_of_birth": "1500",
            "place_of_birth": "Transylvania",
        });
        merge_into_properties(
            &mut props,
            "Count Dracula",
            &["Drac".into(), "Nosferatu".into(), "Count Dracula".into()],
            &incoming,
        );
        let obj = props.as_object().unwrap();
        // Canonical name preserved (first-write-wins).
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("Dracula"));
        // Alias list is set-unioned; self-name filtered out; dedup is case-insensitive.
        let aliases: Vec<&str> = obj
            .get("aliases")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(aliases.contains(&"Drac"));
        assert!(aliases.contains(&"Nosferatu"));
        assert!(aliases.contains(&"Count Dracula"));
        assert_eq!(aliases.len(), 3);
        // DOB NOT overwritten — first-write-wins.
        assert_eq!(
            obj.get("date_of_birth").and_then(|v| v.as_str()),
            Some("1431")
        );
        // POB newly merged.
        assert_eq!(
            obj.get("place_of_birth").and_then(|v| v.as_str()),
            Some("Transylvania")
        );
    }
}
