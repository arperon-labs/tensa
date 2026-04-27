//! KV-backed ingestion job queue with progress tracking.
//!
//! Jobs are persisted in the KV store so they survive restarts.
//! Live progress is tracked via `Arc<Mutex<IngestionProgress>>` for
//! fast polling during active ingestion runs.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::ingestion::extraction::NarrativeExtraction;
use crate::ingestion::pipeline::IngestionReport;
use crate::store::KVStore;
use crate::types::JobStatus;

/// Whether the pipeline uses single-pass or dual-pass (reconciliation) extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassMode {
    Single,
    Dual,
    Session,
}

/// Current phase of an active ingestion run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestionPhase {
    Queued,
    Chunking,
    ContextLoading,
    Extracting,
    Enriching,
    Reconciling,
    Processing,
    Complete,
}

impl std::fmt::Display for IngestionPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Chunking => write!(f, "Chunking"),
            Self::ContextLoading => write!(f, "Loading context"),
            Self::Extracting => write!(f, "Extracting"),
            Self::Enriching => write!(f, "Enriching"),
            Self::Reconciling => write!(f, "Reconciling"),
            Self::Processing => write!(f, "Processing"),
            Self::Complete => write!(f, "Complete"),
        }
    }
}

/// Per-chunk extraction status for live progress tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProgress {
    /// Chunk index in the original text.
    pub index: usize,
    /// Current status.
    pub status: ChunkStatus,
    /// Number of extraction attempts so far.
    #[serde(default)]
    pub attempts: u8,
    /// When extraction started (epoch millis for compact serialization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_ms: Option<i64>,
    /// When extraction finished (epoch millis).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_ms: Option<i64>,
    /// Error message if failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Number of entities extracted from this chunk.
    #[serde(default)]
    pub entities: usize,
    /// Number of situations extracted from this chunk.
    #[serde(default)]
    pub situations: usize,
}

/// Status of a single chunk during extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStatus {
    Pending,
    Extracting,
    Done,
    Failed,
    Skipped,
}

/// Ephemeral progress snapshot for a running ingestion job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionProgress {
    pub phase: IngestionPhase,
    pub current_chunk: usize,
    pub total_chunks: usize,
    pub current_pass: u8,
    pub pass_mode: PassMode,
    pub entities_so_far: usize,
    pub situations_so_far: usize,
    /// Number of worker threads currently extracting.
    #[serde(default)]
    pub active_workers: usize,
    /// Configured concurrency level.
    #[serde(default)]
    pub concurrency: usize,
    /// Model name being used for extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Per-chunk status (populated during extraction phase).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<ChunkProgress>,
    /// Current session turn (SingleSession mode).
    #[serde(default)]
    pub session_turn: usize,
    /// Estimated token usage in session (SingleSession mode).
    #[serde(default)]
    pub session_tokens_used: usize,
    /// Human-readable session status message for real-time monitoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_status: Option<String>,
}

impl IngestionProgress {
    /// Create initial progress for a new job.
    pub fn new(pass_mode: PassMode) -> Self {
        Self {
            phase: IngestionPhase::Queued,
            current_chunk: 0,
            total_chunks: 0,
            current_pass: 1,
            pass_mode,
            entities_so_far: 0,
            situations_so_far: 0,
            active_workers: 0,
            concurrency: 1,
            model: None,
            chunks: Vec::new(),
            session_turn: 0,
            session_tokens_used: 0,
            session_status: None,
        }
    }
}

/// A single chunk's position in the source text, stored before LLM extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkManifestEntry {
    pub chunk_index: usize,
    pub start: usize,
    pub end: usize,
    pub overlap_bytes: usize,
    pub chapter: Option<String>,
    pub content_hash: String,
}

/// Persisted ingestion job record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionJob {
    pub id: String,
    pub status: JobStatus,
    pub text_preview: String,
    pub text_length: usize,
    pub narrative_id: Option<String>,
    pub source_name: String,
    pub pass_mode: PassMode,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub report: Option<IngestionReport>,
    /// Model name used for extraction (for debugging).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether single_session was requested (from config + request).
    #[serde(default)]
    pub single_session_requested: bool,
    /// Whether the extractor actually supports sessions.
    #[serde(default)]
    pub session_capable: bool,
    /// Effective mode after fallback resolution (may differ from pass_mode if fallback occurred).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_mode: Option<String>,
    /// Whether enrichment (step 2) is enabled for this job.
    #[serde(default)]
    pub enrich: bool,
    /// Parent job ID if this is a retry/repair of another job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_job_id: Option<String>,
}

/// Record of a single LLM API call during ingestion, persisted for debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallLog {
    /// Ingestion job this call belongs to.
    pub job_id: String,
    /// Chunk index within the job.
    pub chunk_index: usize,
    /// Pass number (1 = extraction, 2 = reconciliation).
    pub pass: u8,
    /// Attempt number (0-based; >0 means retry).
    pub attempt: u8,
    /// System prompt sent to the LLM.
    pub system_prompt: String,
    /// User prompt sent to the LLM.
    pub user_prompt: String,
    /// Raw text response from the LLM.
    pub raw_response: String,
    /// Successfully parsed extraction (None if parse failed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parsed_extraction: Option<NarrativeExtraction>,
    /// Parse error message (None if parse succeeded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,
    /// Retry prompt (if parse failed and a repair was attempted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_prompt: Option<String>,
    /// Retry response (if a repair call was made).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_response: Option<String>,
    /// Wall-clock duration of the LLM call(s) in milliseconds.
    pub duration_ms: u64,
    /// Model name used for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Endpoint URL called.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Timestamp of the call.
    pub timestamp: DateTime<Utc>,
}

/// Gating decision for a single extracted item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecisionEntry {
    /// "entity", "situation", "participation", or "causal_link".
    pub item_type: String,
    /// Index within its extraction array.
    pub index: usize,
    /// The gating decision.
    pub decision: String, // "committed", "queued", "rejected"
    /// Entity name or situation description (for display).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Committed UUID (if committed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_id: Option<Uuid>,
}

/// Per-chunk extraction record: the raw parsed JSON plus gating outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkExtractionRecord {
    pub job_id: String,
    pub chunk_index: usize,
    /// The raw parsed extraction from the LLM (before gating/resolution).
    pub extraction: NarrativeExtraction,
    /// Per-item gating decisions.
    #[serde(default)]
    pub gate_decisions: Vec<GateDecisionEntry>,
    /// Entity name → committed UUID mapping.
    #[serde(default)]
    pub entity_map: HashMap<String, Uuid>,
    /// Committed situation UUIDs (in extraction order).
    #[serde(default)]
    pub situation_ids: Vec<Uuid>,
}

/// KV-backed ingestion job queue.
pub struct IngestionJobQueue {
    store: Arc<dyn KVStore>,
}

impl IngestionJobQueue {
    /// Create a new ingestion job queue backed by the given store.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Submit a new ingestion job. Returns the job ID.
    pub fn submit(&self, job: IngestionJob) -> Result<String> {
        let id = job.id.clone();
        let key = job_key(&id);
        let data = serde_json::to_vec(&job)?;
        self.store.put(&key, &data)?;

        // Queue index for ordering
        let queue_key = queue_index_key(&job.created_at, &id);
        self.store.put(&queue_key, id.as_bytes())?;

        Ok(id)
    }

    /// Get a job by ID.
    pub fn get_job(&self, job_id: &str) -> Result<IngestionJob> {
        let key = job_key(job_id);
        match self.store.get(&key)? {
            Some(data) => Ok(serde_json::from_slice(&data)?),
            None => Err(TensaError::JobNotFound(job_id.to_string())),
        }
    }

    /// List all ingestion jobs (most recent first), up to `limit`.
    pub fn list_all(&self, limit: usize) -> Result<Vec<IngestionJob>> {
        let prefix = b"ig/q/".to_vec();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut jobs = Vec::new();

        // Pairs are in ascending order; collect all then reverse for newest-first
        let mut all_ids: Vec<String> = Vec::new();
        for (_key, id_bytes) in pairs {
            all_ids.push(String::from_utf8_lossy(&id_bytes).to_string());
        }
        all_ids.reverse();

        for id in all_ids {
            if jobs.len() >= limit {
                break;
            }
            if let Ok(job) = self.get_job(&id) {
                jobs.push(job);
            }
        }

        Ok(jobs)
    }

    /// Mark a job as running.
    pub fn mark_running(&self, job_id: &str) -> Result<()> {
        let mut job = self.get_job(job_id)?;
        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());
        self.update_job(&job)
    }

    /// Mark a job as completed with its report.
    pub fn mark_completed(&self, job_id: &str, report: IngestionReport) -> Result<()> {
        let mut job = self.get_job(job_id)?;
        job.status = JobStatus::Completed;
        job.completed_at = Some(Utc::now());
        job.report = Some(report);
        self.update_job(&job)
    }

    /// Mark a job as failed with an error message.
    pub fn mark_failed(&self, job_id: &str, error: &str) -> Result<()> {
        let mut job = self.get_job(job_id)?;
        job.status = JobStatus::Failed;
        job.completed_at = Some(Utc::now());
        job.error = Some(error.to_string());
        self.update_job(&job)
    }

    /// Attach a partial report to a failed/cancelled job (for rollback data).
    pub fn store_partial_report(
        &self,
        job_id: &str,
        report: crate::ingestion::pipeline::IngestionReport,
    ) -> Result<()> {
        let mut job = self.get_job(job_id)?;
        job.report = Some(report);
        self.update_job(&job)
    }

    /// Store the full source text for a job (separate key to avoid bloating job listings).
    pub fn store_source_text(&self, job_id: &str, text: &str) -> Result<()> {
        let key = source_text_key(job_id);
        self.store.put(&key, text.as_bytes())?;
        Ok(())
    }

    /// Retrieve the full source text for a job (returns None if not stored).
    pub fn get_source_text(&self, job_id: &str) -> Result<Option<String>> {
        let key = source_text_key(job_id);
        match self.store.get(&key)? {
            Some(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).to_string())),
            None => Ok(None),
        }
    }

    /// Store the chunk manifest (byte positions) before LLM extraction begins.
    pub fn store_chunk_manifest(&self, job_id: &str, entries: &[ChunkManifestEntry]) -> Result<()> {
        let key = manifest_key(job_id);
        let data = serde_json::to_vec(entries)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    /// Load the chunk manifest for a job.
    pub fn get_chunk_manifest(&self, job_id: &str) -> Result<Option<Vec<ChunkManifestEntry>>> {
        let key = manifest_key(job_id);
        match self.store.get(&key)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    fn update_job(&self, job: &IngestionJob) -> Result<()> {
        let key = job_key(&job.id);
        let data = serde_json::to_vec(job)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    // ─── LLM Call Log Methods ───────────────────────────────────

    /// Persist an LLM call log entry.
    pub fn store_llm_log(&self, log: &LlmCallLog) -> Result<()> {
        let key = llm_log_key(&log.job_id, log.chunk_index, log.pass, log.attempt);
        let data = serde_json::to_vec(log)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    /// Retrieve all LLM call logs for a job (ordered by chunk/pass/attempt).
    pub fn get_logs_for_job(&self, job_id: &str) -> Result<Vec<LlmCallLog>> {
        let prefix = llm_log_job_prefix(job_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut logs = Vec::new();
        for (_key, data) in pairs {
            if let Ok(log) = serde_json::from_slice::<LlmCallLog>(&data) {
                logs.push(log);
            }
        }
        Ok(logs)
    }

    /// Retrieve LLM call logs for a specific chunk within a job.
    pub fn get_logs_for_chunk(&self, job_id: &str, chunk_index: usize) -> Result<Vec<LlmCallLog>> {
        let prefix = llm_log_chunk_prefix(job_id, chunk_index);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut logs = Vec::new();
        for (_key, data) in pairs {
            if let Ok(log) = serde_json::from_slice::<LlmCallLog>(&data) {
                logs.push(log);
            }
        }
        Ok(logs)
    }

    /// Store a chunk extraction record (parsed extraction + gating decisions).
    pub fn store_chunk_extraction(&self, record: &ChunkExtractionRecord) -> Result<()> {
        let key = chunk_extraction_key(&record.job_id, record.chunk_index);
        let data = serde_json::to_vec(record)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    /// Retrieve the extraction record for a specific chunk.
    pub fn get_chunk_extraction(
        &self,
        job_id: &str,
        chunk_index: usize,
    ) -> Result<Option<ChunkExtractionRecord>> {
        let key = chunk_extraction_key(job_id, chunk_index);
        match self.store.get(&key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    /// Retrieve all chunk extraction records for a job.
    pub fn get_all_chunk_extractions(&self, job_id: &str) -> Result<Vec<ChunkExtractionRecord>> {
        let prefix = chunk_extraction_job_prefix(job_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut records = Vec::new();
        for (_key, data) in pairs {
            if let Ok(rec) = serde_json::from_slice::<ChunkExtractionRecord>(&data) {
                records.push(rec);
            }
        }
        Ok(records)
    }

    // ─── Template Methods (server feature only) ──────────────────

    /// Store or update an ingestion template.
    #[cfg(feature = "server")]
    pub fn store_template(
        &self,
        template: &crate::ingestion::config::IngestionTemplate,
    ) -> Result<()> {
        let key = template_key(&template.id);
        let data = serde_json::to_vec(template)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    /// Get a template by ID.
    #[cfg(feature = "server")]
    pub fn get_template(
        &self,
        id: &str,
    ) -> Result<Option<crate::ingestion::config::IngestionTemplate>> {
        let key = template_key(id);
        match self.store.get(&key)? {
            Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
            None => Ok(None),
        }
    }

    /// List all templates.
    #[cfg(feature = "server")]
    pub fn list_templates(&self) -> Result<Vec<crate::ingestion::config::IngestionTemplate>> {
        let prefix = b"tpl/".to_vec();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut templates = Vec::new();
        for (_key, data) in pairs {
            if let Ok(t) = serde_json::from_slice(&data) {
                templates.push(t);
            }
        }
        Ok(templates)
    }

    /// Delete a template by ID. Refuses to delete builtin templates.
    #[cfg(feature = "server")]
    pub fn delete_template(&self, id: &str) -> Result<()> {
        if let Some(t) = self.get_template(id)? {
            if t.builtin {
                return Err(TensaError::Internal(
                    "Cannot delete builtin template".into(),
                ));
            }
        }
        let key = template_key(id);
        self.store.delete(&key)?;
        Ok(())
    }

    /// Seed builtin templates if none exist yet.
    #[cfg(feature = "server")]
    pub fn init_builtin_templates(&self) -> Result<()> {
        let existing = self.list_templates()?;
        if existing.is_empty() {
            for t in crate::ingestion::config::builtin_templates() {
                self.store_template(&t)?;
            }
        }
        Ok(())
    }

    /// Delete all ingestion jobs (and their logs, extractions, source text, manifests)
    /// that belong to the given narrative. Returns the number of jobs deleted.
    pub fn delete_jobs_for_narrative(&self, narrative_id: &str) -> Result<usize> {
        let all_jobs = self.list_all(10_000)?;
        let mut deleted = 0usize;
        for job in &all_jobs {
            if job.narrative_id.as_deref() == Some(narrative_id) {
                // Delete associated data
                self.delete_logs_for_job(&job.id)?;
                let _ = self.store.delete(&source_text_key(&job.id));
                let _ = self.store.delete(&manifest_key(&job.id));
                // Delete job record
                let _ = self.store.delete(&job_key(&job.id));
                // Delete queue index entry
                let queue_key = queue_index_key(&job.created_at, &job.id);
                let _ = self.store.delete(&queue_key);
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Delete all LLM logs and extraction records for a job (used during rollback).
    pub fn delete_logs_for_job(&self, job_id: &str) -> Result<()> {
        // Delete LLM call logs
        let prefix = llm_log_job_prefix(job_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        for (key, _) in pairs {
            let _ = self.store.delete(&key);
        }
        // Delete chunk extraction records
        let prefix = chunk_extraction_job_prefix(job_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        for (key, _) in pairs {
            let _ = self.store.delete(&key);
        }
        Ok(())
    }
}

// ─── Key Construction ────────────────────────────────────────

/// Primary job key: ig/{job_id}
fn job_key(job_id: &str) -> Vec<u8> {
    let mut key = b"ig/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// Source text key: ig/t/{job_id}
fn source_text_key(job_id: &str) -> Vec<u8> {
    let mut key = b"ig/t/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// Chunk manifest key: ig/m/{job_id}
fn manifest_key(job_id: &str) -> Vec<u8> {
    let mut key = b"ig/m/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// Queue index key: ig/q/{created_be}/{job_id}
fn queue_index_key(created_at: &DateTime<Utc>, job_id: &str) -> Vec<u8> {
    let mut key = b"ig/q/".to_vec();
    key.extend_from_slice(&created_at.timestamp_millis().to_be_bytes());
    key.push(b'/');
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// LLM log key: lg/{job_id}/{chunk_index:04BE}/{pass:01}/{attempt:01}
fn llm_log_key(job_id: &str, chunk_index: usize, pass: u8, attempt: u8) -> Vec<u8> {
    let mut key = b"lg/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key.push(b'/');
    key.extend_from_slice(&(chunk_index as u32).to_be_bytes());
    key.push(b'/');
    key.push(pass);
    key.push(b'/');
    key.push(attempt);
    key
}

/// LLM log prefix for all logs in a job: lg/{job_id}/
fn llm_log_job_prefix(job_id: &str) -> Vec<u8> {
    let mut key = b"lg/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key.push(b'/');
    key
}

/// LLM log prefix for a specific chunk: lg/{job_id}/{chunk_index:04BE}/
fn llm_log_chunk_prefix(job_id: &str, chunk_index: usize) -> Vec<u8> {
    let mut key = b"lg/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key.push(b'/');
    key.extend_from_slice(&(chunk_index as u32).to_be_bytes());
    key.push(b'/');
    key
}

/// Chunk extraction record key: lg/x/{job_id}/{chunk_index:04BE}
fn chunk_extraction_key(job_id: &str, chunk_index: usize) -> Vec<u8> {
    let mut key = b"lg/x/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key.push(b'/');
    key.extend_from_slice(&(chunk_index as u32).to_be_bytes());
    key
}

/// Chunk extraction prefix for all records in a job: lg/x/{job_id}/
fn chunk_extraction_job_prefix(job_id: &str) -> Vec<u8> {
    let mut key = b"lg/x/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key.push(b'/');
    key
}

/// Build a KV key for a user-defined ingestion template.
#[cfg(feature = "server")]
fn template_key(id: &str) -> Vec<u8> {
    format!("tpl/{id}").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn setup() -> IngestionJobQueue {
        let store = Arc::new(MemoryStore::new());
        IngestionJobQueue::new(store)
    }

    fn make_job(name: &str) -> IngestionJob {
        IngestionJob {
            id: uuid::Uuid::now_v7().to_string(),
            status: JobStatus::Pending,
            text_preview: format!("Preview of {}", name),
            text_length: 1000,
            narrative_id: Some("test-narrative".into()),
            source_name: name.to_string(),
            pass_mode: PassMode::Single,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
            report: None,
            model: None,
            single_session_requested: false,
            session_capable: false,
            effective_mode: None,
            enrich: true,
            parent_job_id: None,
        }
    }

    #[test]
    fn test_submit_and_get() {
        let queue = setup();
        let job = make_job("chapter-1.txt");
        let id = job.id.clone();

        let returned = queue.submit(job).unwrap();
        assert_eq!(returned, id);

        let retrieved = queue.get_job(&id).unwrap();
        assert_eq!(retrieved.source_name, "chapter-1.txt");
        assert_eq!(retrieved.status, JobStatus::Pending);
    }

    #[test]
    fn test_get_nonexistent_fails() {
        let queue = setup();
        assert!(queue.get_job("nope").is_err());
    }

    #[test]
    fn test_lifecycle() {
        let queue = setup();
        let job = make_job("test.txt");
        let id = job.id.clone();
        queue.submit(job).unwrap();

        // Pending -> Running
        queue.mark_running(&id).unwrap();
        let j = queue.get_job(&id).unwrap();
        assert_eq!(j.status, JobStatus::Running);
        assert!(j.started_at.is_some());

        // Running -> Completed
        let report = IngestionReport {
            chunks_processed: 5,
            entities_created: 3,
            ..Default::default()
        };
        queue.mark_completed(&id, report).unwrap();
        let j = queue.get_job(&id).unwrap();
        assert_eq!(j.status, JobStatus::Completed);
        assert!(j.completed_at.is_some());
        assert_eq!(j.report.unwrap().chunks_processed, 5);
    }

    #[test]
    fn test_mark_failed() {
        let queue = setup();
        let job = make_job("bad.txt");
        let id = job.id.clone();
        queue.submit(job).unwrap();

        queue.mark_failed(&id, "LLM timeout").unwrap();
        let j = queue.get_job(&id).unwrap();
        assert_eq!(j.status, JobStatus::Failed);
        assert_eq!(j.error.as_deref(), Some("LLM timeout"));
    }

    #[test]
    fn test_list_all() {
        let queue = setup();
        queue.submit(make_job("a.txt")).unwrap();
        queue.submit(make_job("b.txt")).unwrap();
        queue.submit(make_job("c.txt")).unwrap();

        let all = queue.list_all(10).unwrap();
        assert_eq!(all.len(), 3);
        // Newest first
        assert_eq!(all[0].source_name, "c.txt");
    }

    #[test]
    fn test_list_all_with_limit() {
        let queue = setup();
        for i in 0..5 {
            queue.submit(make_job(&format!("{}.txt", i))).unwrap();
        }
        let limited = queue.list_all(2).unwrap();
        assert_eq!(limited.len(), 2);
    }

    fn make_log(job_id: &str, chunk: usize, pass: u8, attempt: u8) -> LlmCallLog {
        LlmCallLog {
            job_id: job_id.to_string(),
            chunk_index: chunk,
            pass,
            attempt,
            system_prompt: "You are a test extractor.".into(),
            user_prompt: format!("Extract chunk {}", chunk),
            raw_response: r#"{"entities":[],"situations":[]}"#.into(),
            parsed_extraction: None,
            parse_error: None,
            retry_prompt: None,
            retry_response: None,
            duration_ms: 1200,
            model: Some("test-model".into()),
            endpoint: None,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_store_and_get_llm_logs() {
        let queue = setup();
        let job_id = "test-job-123";

        queue.store_llm_log(&make_log(job_id, 0, 1, 0)).unwrap();
        queue.store_llm_log(&make_log(job_id, 0, 1, 1)).unwrap();
        queue.store_llm_log(&make_log(job_id, 1, 1, 0)).unwrap();
        queue.store_llm_log(&make_log(job_id, 2, 1, 0)).unwrap();

        // All logs for job
        let all = queue.get_logs_for_job(job_id).unwrap();
        assert_eq!(all.len(), 4);

        // Logs for chunk 0 only (2 attempts)
        let chunk0 = queue.get_logs_for_chunk(job_id, 0).unwrap();
        assert_eq!(chunk0.len(), 2);

        // Logs for chunk 1 only
        let chunk1 = queue.get_logs_for_chunk(job_id, 1).unwrap();
        assert_eq!(chunk1.len(), 1);
    }

    #[test]
    fn test_store_and_get_chunk_extraction() {
        use crate::ingestion::extraction::NarrativeExtraction;

        let queue = setup();
        let job_id = "test-job-456";

        let record = ChunkExtractionRecord {
            job_id: job_id.to_string(),
            chunk_index: 0,
            extraction: NarrativeExtraction {
                entities: vec![],
                situations: vec![],
                participations: vec![],
                causal_links: vec![],
                temporal_relations: vec![],
            },
            gate_decisions: vec![GateDecisionEntry {
                item_type: "entity".into(),
                index: 0,
                decision: "committed".into(),
                label: Some("Alice".into()),
                committed_id: Some(Uuid::now_v7()),
            }],
            entity_map: HashMap::new(),
            situation_ids: vec![],
        };

        queue.store_chunk_extraction(&record).unwrap();

        let retrieved = queue.get_chunk_extraction(job_id, 0).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().gate_decisions.len(), 1);

        let none = queue.get_chunk_extraction(job_id, 99).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn test_delete_logs_for_job() {
        let queue = setup();
        let job_id = "test-job-789";

        queue.store_llm_log(&make_log(job_id, 0, 1, 0)).unwrap();
        queue.store_llm_log(&make_log(job_id, 1, 1, 0)).unwrap();

        let record = ChunkExtractionRecord {
            job_id: job_id.to_string(),
            chunk_index: 0,
            extraction: NarrativeExtraction {
                entities: vec![],
                situations: vec![],
                participations: vec![],
                causal_links: vec![],
                temporal_relations: vec![],
            },
            gate_decisions: vec![],
            entity_map: HashMap::new(),
            situation_ids: vec![],
        };
        queue.store_chunk_extraction(&record).unwrap();

        // Verify they exist
        assert_eq!(queue.get_logs_for_job(job_id).unwrap().len(), 2);
        assert!(queue.get_chunk_extraction(job_id, 0).unwrap().is_some());

        // Delete
        queue.delete_logs_for_job(job_id).unwrap();

        assert_eq!(queue.get_logs_for_job(job_id).unwrap().len(), 0);
        assert!(queue.get_chunk_extraction(job_id, 0).unwrap().is_none());
    }
}
