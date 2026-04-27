//! Scheduler engine — persistent task management + execution dispatch.
//!
//! CRUD operations for [`ScheduledTask`] records stored in KV at the
//! `sched/` prefix, plus due-check logic and distributed-safe locking.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::Hypergraph;

use super::types::*;

// ─── KV Prefixes ─────────────────────────────────────────────

/// KV prefix for scheduled tasks: `sched/`.
const TASK_PREFIX: &str = "sched/";
/// KV prefix for task execution history: `sched/hist/`.
const HISTORY_PREFIX: &str = "sched/hist/";
/// KV prefix for task execution locks: `sched/lock/`.
const LOCK_PREFIX: &str = "sched/lock/";

/// Lock TTL in seconds (30 minutes).
const LOCK_TTL_SECS: i64 = 1800;

// ─── CRUD ────────────────────────────────────────────────────

/// Create a new scheduled task and persist it to the KV store.
pub fn create_task(hypergraph: &Hypergraph, task: &ScheduledTask) -> Result<Uuid> {
    let key = format!("{}{}", TASK_PREFIX, task.id);
    let value = serde_json::to_vec(task)?;
    hypergraph.store().put(key.as_bytes(), &value)?;
    Ok(task.id)
}

/// List all scheduled tasks.
pub fn list_tasks(hypergraph: &Hypergraph) -> Result<Vec<ScheduledTask>> {
    let pairs = hypergraph.store().prefix_scan(TASK_PREFIX.as_bytes())?;
    let mut tasks = Vec::new();
    for (key, value) in pairs {
        // Skip history and lock sub-prefixes which share the `sched/` root.
        let key_str = String::from_utf8_lossy(&key);
        if key_str.starts_with(HISTORY_PREFIX) || key_str.starts_with(LOCK_PREFIX) {
            continue;
        }
        if let Ok(task) = serde_json::from_slice::<ScheduledTask>(&value) {
            tasks.push(task);
        }
    }
    Ok(tasks)
}

/// Get a single task by ID.
pub fn get_task(hypergraph: &Hypergraph, task_id: &Uuid) -> Result<Option<ScheduledTask>> {
    let key = format!("{}{}", TASK_PREFIX, task_id);
    match hypergraph.store().get(key.as_bytes())? {
        Some(bytes) => {
            let task: ScheduledTask = serde_json::from_slice(&bytes)?;
            Ok(Some(task))
        }
        None => Ok(None),
    }
}

/// Update a scheduled task (full replace). Fails if the task does not exist.
pub fn update_task(hypergraph: &Hypergraph, task: &ScheduledTask) -> Result<()> {
    let key = format!("{}{}", TASK_PREFIX, task.id);
    if hypergraph.store().get(key.as_bytes())?.is_none() {
        return Err(TensaError::NotFound(format!("Task {} not found", task.id)));
    }
    let value = serde_json::to_vec(task)?;
    hypergraph.store().put(key.as_bytes(), &value)
}

/// Delete a scheduled task by ID.
pub fn delete_task(hypergraph: &Hypergraph, task_id: &Uuid) -> Result<()> {
    let key = format!("{}{}", TASK_PREFIX, task_id);
    hypergraph.store().delete(key.as_bytes())
}

// ─── History ─────────────────────────────────────────────────

/// Record a task execution result in the history log.
pub fn record_history(hypergraph: &Hypergraph, task_id: &Uuid, result: &TaskResult) -> Result<()> {
    let now = Utc::now();
    let entry = TaskHistory {
        task_id: *task_id,
        result: result.clone(),
        ran_at: now,
    };
    // Use a v7 UUID suffix to guarantee uniqueness even when two records
    // are created within the same millisecond.
    let unique = Uuid::now_v7();
    let key = format!(
        "{}{}/{}/{}",
        HISTORY_PREFIX,
        task_id,
        now.timestamp_millis(),
        unique
    );
    let value = serde_json::to_vec(&entry)?;
    hypergraph.store().put(key.as_bytes(), &value)
}

/// Retrieve task execution history (most recent first, up to `limit`).
pub fn get_history(
    hypergraph: &Hypergraph,
    task_id: &Uuid,
    limit: usize,
) -> Result<Vec<TaskHistory>> {
    let prefix = format!("{}{}/", HISTORY_PREFIX, task_id);
    let pairs = hypergraph.store().prefix_scan(prefix.as_bytes())?;
    let mut entries: Vec<TaskHistory> = pairs
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
        .collect();
    entries.sort_by(|a, b| b.ran_at.cmp(&a.ran_at));
    entries.truncate(limit);
    Ok(entries)
}

// ─── Scheduling Logic ────────────────────────────────────────

/// Check whether a task is due to run based on its schedule and last execution.
pub fn is_task_due(task: &ScheduledTask) -> bool {
    if !task.enabled {
        return false;
    }
    match &task.schedule {
        TaskSchedule::Interval { seconds } => match task.last_run {
            None => true,
            Some(last) => {
                let elapsed = (Utc::now() - last).num_seconds();
                elapsed >= *seconds as i64
            }
        },
        TaskSchedule::Once { at } => task.last_run.is_none() && Utc::now() >= *at,
    }
}

/// Return all tasks that are currently due to run.
pub fn due_tasks(hypergraph: &Hypergraph) -> Result<Vec<ScheduledTask>> {
    let all = list_tasks(hypergraph)?;
    Ok(all.into_iter().filter(is_task_due).collect())
}

// ─── Locking ─────────────────────────────────────────────────

/// Try to acquire a lock for a task.
///
/// Returns `true` if the lock was acquired, `false` if another execution
/// holds a valid (non-expired) lock. Locks expire after 30 minutes.
pub fn try_acquire_lock(hypergraph: &Hypergraph, task_id: &Uuid) -> Result<bool> {
    let key = format!("{}{}", LOCK_PREFIX, task_id);
    if let Some(bytes) = hypergraph.store().get(key.as_bytes())? {
        if let Ok(locked_at) = String::from_utf8(bytes) {
            if let Ok(ts) = locked_at.parse::<i64>() {
                let elapsed = Utc::now().timestamp() - ts;
                if elapsed < LOCK_TTL_SECS {
                    return Ok(false); // Lock still valid
                }
            }
        }
        // Lock expired or corrupt — overwrite.
    }
    let value = Utc::now().timestamp().to_string();
    hypergraph.store().put(key.as_bytes(), value.as_bytes())?;
    Ok(true)
}

/// Release a task execution lock.
pub fn release_lock(hypergraph: &Hypergraph, task_id: &Uuid) -> Result<()> {
    let key = format!("{}{}", LOCK_PREFIX, task_id);
    hypergraph.store().delete(key.as_bytes())
}

// ─── Composite Operations ────────────────────────────────────

/// Mark a task as having just run with the given result.
///
/// Updates the task's `last_run` and `last_result` fields, then records
/// the result in the history log.
pub fn mark_task_run(hypergraph: &Hypergraph, task_id: &Uuid, result: TaskResult) -> Result<()> {
    let mut task = get_task(hypergraph, task_id)?
        .ok_or_else(|| TensaError::NotFound(format!("Task {} not found", task_id)))?;
    task.last_run = Some(Utc::now());
    task.last_result = Some(result.clone());
    // Write the updated task — bypass the existence check since we just loaded it.
    let key = format!("{}{}", TASK_PREFIX, task.id);
    let value = serde_json::to_vec(&task)?;
    hypergraph.store().put(key.as_bytes(), &value)?;
    record_history(hypergraph, task_id, &result)?;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn sample_task() -> ScheduledTask {
        ScheduledTask::new(
            ScheduledTaskType::CibScan,
            TaskSchedule::Interval { seconds: 3600 },
        )
    }

    #[test]
    fn test_create_and_get_task() {
        let hg = test_hg();
        let task = sample_task();
        let id = create_task(&hg, &task).unwrap();
        assert_eq!(id, task.id);

        let loaded = get_task(&hg, &id).unwrap().unwrap();
        assert_eq!(loaded.id, task.id);
        assert_eq!(loaded.task_type, ScheduledTaskType::CibScan);
        assert!(loaded.enabled);
    }

    #[test]
    fn test_list_tasks() {
        let hg = test_hg();
        let t1 = sample_task();
        let mut t2 = sample_task();
        t2.task_type = ScheduledTaskType::McpPoll;
        create_task(&hg, &t1).unwrap();
        create_task(&hg, &t2).unwrap();

        let all = list_tasks(&hg).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_delete_task() {
        let hg = test_hg();
        let task = sample_task();
        let id = create_task(&hg, &task).unwrap();
        delete_task(&hg, &id).unwrap();
        assert!(get_task(&hg, &id).unwrap().is_none());
    }

    #[test]
    fn test_update_task() {
        let hg = test_hg();
        let mut task = sample_task();
        create_task(&hg, &task).unwrap();

        task.enabled = false;
        update_task(&hg, &task).unwrap();

        let loaded = get_task(&hg, &task.id).unwrap().unwrap();
        assert!(!loaded.enabled);
    }

    #[test]
    fn test_update_nonexistent_fails() {
        let hg = test_hg();
        let task = sample_task();
        let res = update_task(&hg, &task);
        assert!(res.is_err());
    }

    #[test]
    fn test_is_task_due_never_run() {
        let task = sample_task();
        assert!(is_task_due(&task));
    }

    #[test]
    fn test_is_task_due_recently_run() {
        let mut task = sample_task();
        task.last_run = Some(Utc::now());
        assert!(!is_task_due(&task));
    }

    #[test]
    fn test_is_task_due_disabled() {
        let mut task = sample_task();
        task.enabled = false;
        assert!(!is_task_due(&task));
    }

    #[test]
    fn test_is_task_due_once_in_future() {
        let task = ScheduledTask {
            id: Uuid::now_v7(),
            task_type: ScheduledTaskType::ReportGeneration,
            schedule: TaskSchedule::Once {
                at: Utc::now() + chrono::Duration::hours(1),
            },
            enabled: true,
            last_run: None,
            last_result: None,
            config: serde_json::Value::Null,
            created_at: Utc::now(),
        };
        assert!(!is_task_due(&task));
    }

    #[test]
    fn test_is_task_due_once_in_past() {
        let task = ScheduledTask {
            id: Uuid::now_v7(),
            task_type: ScheduledTaskType::ReportGeneration,
            schedule: TaskSchedule::Once {
                at: Utc::now() - chrono::Duration::hours(1),
            },
            enabled: true,
            last_run: None,
            last_result: None,
            config: serde_json::Value::Null,
            created_at: Utc::now(),
        };
        assert!(is_task_due(&task));
    }

    #[test]
    fn test_is_task_due_once_already_run() {
        let task = ScheduledTask {
            id: Uuid::now_v7(),
            task_type: ScheduledTaskType::ReportGeneration,
            schedule: TaskSchedule::Once {
                at: Utc::now() - chrono::Duration::hours(1),
            },
            enabled: true,
            last_run: Some(Utc::now()),
            last_result: None,
            config: serde_json::Value::Null,
            created_at: Utc::now(),
        };
        assert!(!is_task_due(&task));
    }

    #[test]
    fn test_lock_acquire_release() {
        let hg = test_hg();
        let id = Uuid::now_v7();

        // First acquire succeeds.
        assert!(try_acquire_lock(&hg, &id).unwrap());

        // Second acquire fails (lock held).
        assert!(!try_acquire_lock(&hg, &id).unwrap());

        // Release, then re-acquire succeeds.
        release_lock(&hg, &id).unwrap();
        assert!(try_acquire_lock(&hg, &id).unwrap());
    }

    #[test]
    fn test_record_and_get_history() {
        let hg = test_hg();
        let task = sample_task();
        create_task(&hg, &task).unwrap();

        let r1 = TaskResult::Success {
            duration_ms: 100,
            summary: "run 1".into(),
        };
        let r2 = TaskResult::Failed {
            error: "boom".into(),
            duration_ms: 50,
        };
        record_history(&hg, &task.id, &r1).unwrap();
        record_history(&hg, &task.id, &r2).unwrap();

        let hist = get_history(&hg, &task.id, 10).unwrap();
        assert_eq!(hist.len(), 2);
        // Most recent first.
        assert_eq!(hist[0].task_id, task.id);
    }

    #[test]
    fn test_get_history_respects_limit() {
        let hg = test_hg();
        let task = sample_task();
        create_task(&hg, &task).unwrap();

        for i in 0..5 {
            let r = TaskResult::Success {
                duration_ms: i,
                summary: format!("run {}", i),
            };
            record_history(&hg, &task.id, &r).unwrap();
        }

        let hist = get_history(&hg, &task.id, 3).unwrap();
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_mark_task_run() {
        let hg = test_hg();
        let task = sample_task();
        create_task(&hg, &task).unwrap();

        let result = TaskResult::Success {
            duration_ms: 42,
            summary: "all good".into(),
        };
        mark_task_run(&hg, &task.id, result.clone()).unwrap();

        let loaded = get_task(&hg, &task.id).unwrap().unwrap();
        assert!(loaded.last_run.is_some());
        assert_eq!(loaded.last_result, Some(result));

        let hist = get_history(&hg, &task.id, 10).unwrap();
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn test_mark_task_run_nonexistent() {
        let hg = test_hg();
        let id = Uuid::now_v7();
        let result = TaskResult::Skipped {
            reason: "test".into(),
        };
        assert!(mark_task_run(&hg, &id, result).is_err());
    }

    #[test]
    fn test_due_tasks() {
        let hg = test_hg();

        let t1 = sample_task(); // never run, enabled → due
        let mut t2 = sample_task();
        t2.enabled = false; // disabled → not due
        let mut t3 = sample_task();
        t3.last_run = Some(Utc::now()); // just ran → not due

        create_task(&hg, &t1).unwrap();
        create_task(&hg, &t2).unwrap();
        create_task(&hg, &t3).unwrap();

        let due = due_tasks(&hg).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, t1.id);
    }

    #[test]
    fn test_list_tasks_excludes_history_and_locks() {
        let hg = test_hg();
        let task = sample_task();
        create_task(&hg, &task).unwrap();

        // Record some history and acquire a lock.
        let r = TaskResult::Success {
            duration_ms: 1,
            summary: "ok".into(),
        };
        record_history(&hg, &task.id, &r).unwrap();
        try_acquire_lock(&hg, &task.id).unwrap();

        // list_tasks should still return exactly 1 task.
        let all = list_tasks(&hg).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, task.id);
    }
}
