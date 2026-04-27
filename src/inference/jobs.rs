//! KV-backed job queue for async inference.
//!
//! Jobs are persisted in the KV store so they survive process restarts.
//! The queue supports priority-based dequeuing, deduplication by
//! target+type, and status tracking through the job lifecycle.

use std::sync::Arc;

use chrono::Utc;
use serde_json;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;
use crate::types::{InferenceJobType, InferenceResult, JobPriority, JobStatus};

use super::types::InferenceJob;

/// KV-backed inference job queue.
pub struct JobQueue {
    store: Arc<dyn KVStore>,
}

impl JobQueue {
    /// Create a new job queue backed by the given store.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Submit a new inference job. Returns the job ID.
    ///
    /// If an in-flight job (Pending or Running) already exists for the
    /// same target and job type, returns the existing job's ID instead
    /// of creating a duplicate.
    pub fn submit(&self, job: InferenceJob) -> Result<String> {
        // Check for deduplication
        if let Some(existing_id) = self.find_in_flight(&job.target_id, &job.job_type)? {
            return Ok(existing_id);
        }

        let id = job.id.clone();

        // Store the job
        let job_key = job_key(&id);
        let job_data = serde_json::to_vec(&job)?;
        self.store.put(&job_key, &job_data)?;

        // Add to priority queue index
        let queue_key = queue_index_key(&job.priority, &job.created_at, &id);
        self.store.put(&queue_key, id.as_bytes())?;

        // Add to target index
        let target_key = target_index_key(&job.target_id, &job.job_type, &id);
        self.store.put(&target_key, id.as_bytes())?;

        Ok(id)
    }

    /// Get a job by ID.
    pub fn get_job(&self, job_id: &str) -> Result<InferenceJob> {
        let key = job_key(job_id);
        match self.store.get(&key)? {
            Some(data) => Ok(serde_json::from_slice(&data)?),
            None => Err(TensaError::JobNotFound(job_id.to_string())),
        }
    }

    /// Get an inference result by job ID.
    pub fn get_result(&self, job_id: &str) -> Result<InferenceResult> {
        let key = result_key(job_id);
        match self.store.get(&key)? {
            Some(data) => Ok(serde_json::from_slice(&data)?),
            None => Err(TensaError::JobNotFound(format!(
                "No result for job: {}",
                job_id
            ))),
        }
    }

    /// Store an inference result.
    pub fn store_result(&self, result: InferenceResult) -> Result<()> {
        let key = result_key(&result.job_id);
        let data = serde_json::to_vec(&result)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    /// List pending jobs, ordered by priority then creation time.
    pub fn list_pending(&self, limit: usize) -> Result<Vec<InferenceJob>> {
        let prefix = b"ij/q/".to_vec();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut jobs = Vec::new();

        for (_key, id_bytes) in pairs {
            if jobs.len() >= limit {
                break;
            }
            let id = String::from_utf8_lossy(&id_bytes).to_string();
            match self.get_job(&id) {
                Ok(job) if job.status == JobStatus::Pending => {
                    jobs.push(job);
                }
                _ => continue,
            }
        }

        Ok(jobs)
    }

    /// List recent jobs across all statuses (Pending, Running, Completed, Failed).
    /// Returns up to `limit` jobs, most recent first.
    pub fn list_recent(&self, limit: usize) -> Result<Vec<InferenceJob>> {
        let prefix = b"ij/".to_vec();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut jobs = Vec::new();

        for (key, data) in pairs {
            // Only match primary job keys (ij/{uuid}), skip index keys (ij/q/, ij/t/)
            let key_str = std::str::from_utf8(&key).unwrap_or("");
            if key_str.starts_with("ij/q/") || key_str.starts_with("ij/t/") {
                continue;
            }
            if let Ok(job) = serde_json::from_slice::<InferenceJob>(&data) {
                jobs.push(job);
            }
        }

        // Sort by created_at descending (most recent first)
        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        jobs.truncate(limit);
        Ok(jobs)
    }

    /// List all jobs for a given target entity/situation.
    pub fn list_by_target(&self, target_id: &Uuid) -> Result<Vec<InferenceJob>> {
        let mut prefix = b"ij/t/".to_vec();
        prefix.extend_from_slice(target_id.as_bytes());
        prefix.push(b'/');

        let pairs = self.store.prefix_scan(&prefix)?;
        let mut jobs = Vec::new();

        for (_key, id_bytes) in pairs {
            let id = String::from_utf8_lossy(&id_bytes).to_string();
            if let Ok(job) = self.get_job(&id) {
                jobs.push(job);
            }
        }

        Ok(jobs)
    }

    /// Dequeue the next pending job (highest priority, oldest first).
    pub fn dequeue_next(&self) -> Result<Option<InferenceJob>> {
        let prefix = b"ij/q/".to_vec();
        let pairs = self.store.prefix_scan(&prefix)?;

        for (queue_key, id_bytes) in pairs {
            let id = String::from_utf8_lossy(&id_bytes).to_string();
            match self.get_job(&id) {
                Ok(job) if job.status == JobStatus::Pending => {
                    // Remove from queue index (claimed)
                    self.store.delete(&queue_key)?;
                    return Ok(Some(job));
                }
                _ => {
                    // Stale entry, clean up
                    self.store.delete(&queue_key)?;
                    continue;
                }
            }
        }

        Ok(None)
    }

    /// Mark a job as running.
    pub fn mark_running(&self, job_id: &str) -> Result<()> {
        let mut job = self.get_job(job_id)?;
        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());
        self.update_job(&job)
    }

    /// Mark a job as completed.
    pub fn mark_completed(&self, job_id: &str) -> Result<()> {
        let mut job = self.get_job(job_id)?;
        job.status = JobStatus::Completed;
        job.completed_at = Some(Utc::now());
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

    /// Cancel a pending job. Returns error if job is not pending.
    pub fn cancel(&self, job_id: &str) -> Result<()> {
        let job = self.get_job(job_id)?;
        if job.status != JobStatus::Pending {
            return Err(TensaError::InferenceError(format!(
                "Cannot cancel job {} with status {:?}",
                job_id, job.status
            )));
        }
        // Remove from queue index
        let queue_key = queue_index_key(&job.priority, &job.created_at, job_id);
        self.store.delete(&queue_key)?;
        // Mark as failed
        self.mark_failed(job_id, "Cancelled by user")
    }

    /// Force-abort a job regardless of status — including `Running` zombies
    /// whose worker is gone (crashed, binary swapped, power loss). Does not
    /// attempt to signal the worker; if one is genuinely still running the
    /// job, its completion will clobber this. Safe to call after a restart.
    pub fn force_abort(&self, job_id: &str) -> Result<()> {
        let job = self.get_job(job_id)?;
        // If pending, also remove from the queue index.
        if job.status == JobStatus::Pending {
            let queue_key = queue_index_key(&job.priority, &job.created_at, job_id);
            let _ = self.store.delete(&queue_key);
        }
        self.mark_failed(job_id, "Force-aborted")
    }

    /// Sweep every job currently stamped `Running` to `Failed`. Intended for
    /// server startup: the worker pool is empty, so any `Running` status is
    /// stale. Returns the count swept. Caps the scan at one million jobs,
    /// which is far beyond any realistic queue backlog.
    pub fn reap_stale_running(&self) -> Result<usize> {
        let jobs = self.list_recent(1_000_000)?;
        let mut swept = 0;
        for job in jobs {
            if job.status == JobStatus::Running {
                if self
                    .mark_failed(&job.id, "Reaped on server startup (worker gone)")
                    .is_ok()
                {
                    // Best-effort: also drop the pending-queue index if somehow present.
                    let queue_key = queue_index_key(&job.priority, &job.created_at, &job.id);
                    let _ = self.store.delete(&queue_key);
                    swept += 1;
                }
            }
        }
        Ok(swept)
    }

    fn update_job(&self, job: &InferenceJob) -> Result<()> {
        let key = job_key(&job.id);
        let data = serde_json::to_vec(job)?;
        self.store.put(&key, &data)?;
        Ok(())
    }

    /// Find an in-flight (Pending or Running) job for the given target and type.
    fn find_in_flight(
        &self,
        target_id: &Uuid,
        job_type: &InferenceJobType,
    ) -> Result<Option<String>> {
        let mut prefix = b"ij/t/".to_vec();
        prefix.extend_from_slice(target_id.as_bytes());
        prefix.push(b'/');
        let type_str = serde_json::to_string(job_type)?;
        prefix.extend_from_slice(type_str.as_bytes());
        prefix.push(b'/');

        let pairs = self.store.prefix_scan(&prefix)?;
        for (_key, id_bytes) in pairs {
            let id = String::from_utf8_lossy(&id_bytes).to_string();
            if let Ok(job) = self.get_job(&id) {
                if job.status == JobStatus::Pending || job.status == JobStatus::Running {
                    return Ok(Some(id));
                }
            }
        }

        Ok(None)
    }
}

// ─── Key Construction ────────────────────────────────────────

/// Primary job key: ij/{job_id}
fn job_key(job_id: &str) -> Vec<u8> {
    let mut key = b"ij/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// Result key: ir/{job_id}
fn result_key(job_id: &str) -> Vec<u8> {
    let mut key = b"ir/".to_vec();
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// Priority queue index key: ij/q/{priority_byte}/{created_be}/{job_id}
fn queue_index_key(
    priority: &JobPriority,
    created_at: &chrono::DateTime<chrono::Utc>,
    job_id: &str,
) -> Vec<u8> {
    let mut key = b"ij/q/".to_vec();
    key.push(*priority as u8);
    key.push(b'/');
    key.extend_from_slice(&created_at.timestamp_millis().to_be_bytes());
    key.push(b'/');
    key.extend_from_slice(job_id.as_bytes());
    key
}

/// Target index key: ij/t/{target_uuid}/{job_type_json}/{job_id}
fn target_index_key(target_id: &Uuid, job_type: &InferenceJobType, job_id: &str) -> Vec<u8> {
    let mut key = b"ij/t/".to_vec();
    key.extend_from_slice(target_id.as_bytes());
    key.push(b'/');
    // Use JSON serialization for consistent type encoding
    let type_str = serde_json::to_string(job_type).unwrap_or_default();
    key.extend_from_slice(type_str.as_bytes());
    key.push(b'/');
    key.extend_from_slice(job_id.as_bytes());
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn setup() -> JobQueue {
        let store = Arc::new(MemoryStore::new());
        JobQueue::new(store)
    }

    fn make_job(job_type: InferenceJobType, priority: JobPriority) -> InferenceJob {
        InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({}),
            priority,
            status: JobStatus::Pending,
            estimated_cost_ms: 1000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    fn make_job_for_target(
        target_id: Uuid,
        job_type: InferenceJobType,
        priority: JobPriority,
    ) -> InferenceJob {
        InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type,
            target_id,
            parameters: serde_json::json!({}),
            priority,
            status: JobStatus::Pending,
            estimated_cost_ms: 1000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    #[test]
    fn test_submit_and_get_job() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();

        let returned_id = queue.submit(job).unwrap();
        assert_eq!(returned_id, id);

        let retrieved = queue.get_job(&id).unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.status, JobStatus::Pending);
    }

    #[test]
    fn test_get_nonexistent_job_fails() {
        let queue = setup();
        assert!(queue.get_job("nonexistent").is_err());
    }

    #[test]
    fn test_dequeue_by_priority() {
        let queue = setup();

        let low = make_job(InferenceJobType::CausalDiscovery, JobPriority::Low);
        let high = make_job(InferenceJobType::GameClassification, JobPriority::High);
        let normal = make_job(InferenceJobType::MotivationInference, JobPriority::Normal);

        let low_id = low.id.clone();
        let high_id = high.id.clone();
        let normal_id = normal.id.clone();

        queue.submit(low).unwrap();
        queue.submit(high).unwrap();
        queue.submit(normal).unwrap();

        // Should dequeue in priority order: High, Normal, Low
        let first = queue.dequeue_next().unwrap().unwrap();
        assert_eq!(first.id, high_id);

        let second = queue.dequeue_next().unwrap().unwrap();
        assert_eq!(second.id, normal_id);

        let third = queue.dequeue_next().unwrap().unwrap();
        assert_eq!(third.id, low_id);

        // Queue empty
        assert!(queue.dequeue_next().unwrap().is_none());
    }

    #[test]
    fn test_mark_running() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();

        queue.mark_running(&id).unwrap();
        let updated = queue.get_job(&id).unwrap();
        assert_eq!(updated.status, JobStatus::Running);
        assert!(updated.started_at.is_some());
    }

    #[test]
    fn test_mark_completed() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();

        queue.mark_running(&id).unwrap();
        queue.mark_completed(&id).unwrap();
        let updated = queue.get_job(&id).unwrap();
        assert_eq!(updated.status, JobStatus::Completed);
        assert!(updated.completed_at.is_some());
    }

    #[test]
    fn test_mark_failed() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();

        queue.mark_failed(&id, "Algorithm diverged").unwrap();
        let updated = queue.get_job(&id).unwrap();
        assert_eq!(updated.status, JobStatus::Failed);
        assert_eq!(updated.error.as_deref(), Some("Algorithm diverged"));
    }

    #[test]
    fn test_store_and_get_result() {
        let queue = setup();
        let result = InferenceResult {
            job_id: "job-001".to_string(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: Uuid::now_v7(),
            result: serde_json::json!({"links": []}),
            confidence: 0.85,
            explanation: None,
            status: JobStatus::Completed,
            created_at: Utc::now(),
            completed_at: Some(Utc::now()),
        };

        queue.store_result(result.clone()).unwrap();
        let retrieved = queue.get_result("job-001").unwrap();
        assert_eq!(retrieved.confidence, 0.85);
    }

    #[test]
    fn test_get_nonexistent_result_fails() {
        let queue = setup();
        assert!(queue.get_result("nonexistent").is_err());
    }

    #[test]
    fn test_list_pending() {
        let queue = setup();
        queue
            .submit(make_job(
                InferenceJobType::CausalDiscovery,
                JobPriority::Normal,
            ))
            .unwrap();
        queue
            .submit(make_job(
                InferenceJobType::GameClassification,
                JobPriority::High,
            ))
            .unwrap();
        queue
            .submit(make_job(
                InferenceJobType::MotivationInference,
                JobPriority::Low,
            ))
            .unwrap();

        let pending = queue.list_pending(10).unwrap();
        assert_eq!(pending.len(), 3);
    }

    #[test]
    fn test_list_pending_limit() {
        let queue = setup();
        for _ in 0..5 {
            queue
                .submit(make_job(
                    InferenceJobType::CausalDiscovery,
                    JobPriority::Normal,
                ))
                .unwrap();
        }
        let pending = queue.list_pending(2).unwrap();
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_list_pending_excludes_completed() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();
        queue
            .submit(make_job(
                InferenceJobType::GameClassification,
                JobPriority::Normal,
            ))
            .unwrap();

        queue.mark_completed(&id).unwrap();

        let pending = queue.list_pending(10).unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_list_by_target() {
        let queue = setup();
        let target = Uuid::now_v7();

        queue
            .submit(make_job_for_target(
                target,
                InferenceJobType::CausalDiscovery,
                JobPriority::Normal,
            ))
            .unwrap();
        queue
            .submit(make_job_for_target(
                target,
                InferenceJobType::GameClassification,
                JobPriority::Normal,
            ))
            .unwrap();
        // Different target
        queue
            .submit(make_job(
                InferenceJobType::MotivationInference,
                JobPriority::Normal,
            ))
            .unwrap();

        let jobs = queue.list_by_target(&target).unwrap();
        assert_eq!(jobs.len(), 2);
    }

    #[test]
    fn test_deduplication() {
        let queue = setup();
        let target = Uuid::now_v7();

        let job1 = make_job_for_target(
            target,
            InferenceJobType::CausalDiscovery,
            JobPriority::Normal,
        );
        let id1 = job1.id.clone();
        queue.submit(job1).unwrap();

        // Submit another job for same target+type — should return existing id
        let job2 = make_job_for_target(
            target,
            InferenceJobType::CausalDiscovery,
            JobPriority::Normal,
        );
        let returned_id = queue.submit(job2).unwrap();
        assert_eq!(returned_id, id1);
    }

    #[test]
    fn test_deduplication_allows_different_types() {
        let queue = setup();
        let target = Uuid::now_v7();

        let job1 = make_job_for_target(
            target,
            InferenceJobType::CausalDiscovery,
            JobPriority::Normal,
        );
        let id1 = job1.id.clone();
        queue.submit(job1).unwrap();

        let job2 = make_job_for_target(
            target,
            InferenceJobType::GameClassification,
            JobPriority::Normal,
        );
        let id2 = job2.id.clone();
        let returned_id = queue.submit(job2).unwrap();
        assert_eq!(returned_id, id2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_deduplication_allows_after_completion() {
        let queue = setup();
        let target = Uuid::now_v7();

        let job1 = make_job_for_target(
            target,
            InferenceJobType::CausalDiscovery,
            JobPriority::Normal,
        );
        let id1 = job1.id.clone();
        queue.submit(job1).unwrap();
        queue.mark_completed(&id1).unwrap();

        // Now a new job for same target+type should be accepted
        let job2 = make_job_for_target(
            target,
            InferenceJobType::CausalDiscovery,
            JobPriority::Normal,
        );
        let id2 = job2.id.clone();
        let returned_id = queue.submit(job2).unwrap();
        assert_eq!(returned_id, id2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_cancel_pending_job() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();

        queue.cancel(&id).unwrap();
        let updated = queue.get_job(&id).unwrap();
        assert_eq!(updated.status, JobStatus::Failed);
        assert!(updated.error.as_deref().unwrap().contains("Cancelled"));
    }

    #[test]
    fn test_cancel_running_job_fails() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();
        queue.mark_running(&id).unwrap();

        assert!(queue.cancel(&id).is_err());
    }

    #[test]
    fn test_state_transition_lifecycle() {
        let queue = setup();
        let job = make_job(InferenceJobType::CausalDiscovery, JobPriority::Normal);
        let id = job.id.clone();
        queue.submit(job).unwrap();

        // Pending -> Running -> Completed
        assert_eq!(queue.get_job(&id).unwrap().status, JobStatus::Pending);
        queue.mark_running(&id).unwrap();
        assert_eq!(queue.get_job(&id).unwrap().status, JobStatus::Running);
        queue.mark_completed(&id).unwrap();
        assert_eq!(queue.get_job(&id).unwrap().status, JobStatus::Completed);
    }
}
