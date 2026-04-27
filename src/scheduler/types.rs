//! Scheduler types — task definitions, schedules, and results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};

// ─── Schedule ────────────────────────────────────────────────

/// When a task should run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskSchedule {
    /// Repeat every N seconds.
    Interval { seconds: u64 },
    /// Run exactly once at the given time.
    Once { at: DateTime<Utc> },
}

/// Parse a human-friendly duration string into a [`TaskSchedule::Interval`].
///
/// Accepted suffixes: `s` (seconds), `m` (minutes), `h` (hours), `d` (days).
///
/// # Examples
/// ```
/// # use tensa::scheduler::parse_schedule;
/// let s = parse_schedule("30m").unwrap();
/// let s = parse_schedule("6h").unwrap();
/// let s = parse_schedule("7d").unwrap();
/// let s = parse_schedule("3600s").unwrap();
/// ```
pub fn parse_schedule(s: &str) -> Result<TaskSchedule> {
    let s = s.trim();
    if s.is_empty() {
        return Err(TensaError::InvalidQuery(
            "Empty schedule string".to_string(),
        ));
    }
    let (num_part, suffix) = s.split_at(s.len() - 1);
    let multiplier: u64 = match suffix {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        "d" => 86400,
        _ => {
            return Err(TensaError::InvalidQuery(format!(
                "Unknown schedule suffix '{}'; expected s/m/h/d",
                suffix
            )))
        }
    };
    let count: u64 = num_part.parse::<u64>().map_err(|_| {
        TensaError::InvalidQuery(format!("Invalid number in schedule string: '{}'", num_part))
    })?;
    if count == 0 {
        return Err(TensaError::InvalidQuery(
            "Schedule interval must be > 0".to_string(),
        ));
    }
    Ok(TaskSchedule::Interval {
        seconds: count * multiplier,
    })
}

// ─── Task Types ──────────────────────────────────────────────

/// The kind of analysis job a scheduled task executes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledTaskType {
    /// Coordinated Inauthentic Behavior scan.
    CibScan,
    /// Discover new information sources.
    SourceDiscovery,
    /// Sync external fact-check databases.
    FactCheckSync,
    /// Generate or refresh analytical reports.
    ReportGeneration,
    /// Poll MCP sources for new posts.
    McpPoll,
    /// Recompute stylometric fingerprints.
    FingerprintRefresh,
    /// Update spread-velocity baselines.
    VelocityBaselineUpdate,
}

impl ScheduledTaskType {
    /// Stable string representation for serialization and display.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CibScan => "cib_scan",
            Self::SourceDiscovery => "source_discovery",
            Self::FactCheckSync => "fact_check_sync",
            Self::ReportGeneration => "report_generation",
            Self::McpPoll => "mcp_poll",
            Self::FingerprintRefresh => "fingerprint_refresh",
            Self::VelocityBaselineUpdate => "velocity_baseline_update",
        }
    }
}

impl std::str::FromStr for ScheduledTaskType {
    type Err = TensaError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cib_scan" => Ok(Self::CibScan),
            "source_discovery" => Ok(Self::SourceDiscovery),
            "fact_check_sync" => Ok(Self::FactCheckSync),
            "report_generation" => Ok(Self::ReportGeneration),
            "mcp_poll" => Ok(Self::McpPoll),
            "fingerprint_refresh" => Ok(Self::FingerprintRefresh),
            "velocity_baseline_update" => Ok(Self::VelocityBaselineUpdate),
            _ => Err(TensaError::InvalidQuery(format!(
                "Unknown scheduled task type: '{}'",
                s
            ))),
        }
    }
}

impl std::fmt::Display for ScheduledTaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── Task Result ─────────────────────────────────────────────

/// Outcome of a single task execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TaskResult {
    Success { duration_ms: u64, summary: String },
    Failed { error: String, duration_ms: u64 },
    Skipped { reason: String },
}

// ─── Scheduled Task ──────────────────────────────────────────

/// A persistently-stored scheduled task definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: Uuid,
    pub task_type: ScheduledTaskType,
    pub schedule: TaskSchedule,
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_result: Option<TaskResult>,
    /// Opaque configuration blob for the specific task type.
    #[serde(default)]
    pub config: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

impl ScheduledTask {
    /// Create a new enabled task with the given type and schedule.
    pub fn new(task_type: ScheduledTaskType, schedule: TaskSchedule) -> Self {
        Self {
            id: Uuid::now_v7(),
            task_type,
            schedule,
            enabled: true,
            last_run: None,
            last_result: None,
            config: serde_json::Value::Null,
            created_at: Utc::now(),
        }
    }
}

// ─── Task History ────────────────────────────────────────────

/// A single entry in a task's execution history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskHistory {
    pub task_id: Uuid,
    pub result: TaskResult,
    pub ran_at: DateTime<Utc>,
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schedule_seconds() {
        let s = parse_schedule("3600s").unwrap();
        assert_eq!(s, TaskSchedule::Interval { seconds: 3600 });
    }

    #[test]
    fn test_parse_schedule_minutes() {
        let s = parse_schedule("30m").unwrap();
        assert_eq!(s, TaskSchedule::Interval { seconds: 1800 });
    }

    #[test]
    fn test_parse_schedule_hours() {
        let s = parse_schedule("6h").unwrap();
        assert_eq!(s, TaskSchedule::Interval { seconds: 21600 });
    }

    #[test]
    fn test_parse_schedule_days() {
        let s = parse_schedule("7d").unwrap();
        assert_eq!(s, TaskSchedule::Interval { seconds: 604800 });
    }

    #[test]
    fn test_parse_schedule_invalid_suffix() {
        assert!(parse_schedule("10x").is_err());
    }

    #[test]
    fn test_parse_schedule_zero_rejected() {
        assert!(parse_schedule("0h").is_err());
    }

    #[test]
    fn test_parse_schedule_empty_rejected() {
        assert!(parse_schedule("").is_err());
    }

    #[test]
    fn test_parse_schedule_non_numeric() {
        assert!(parse_schedule("abch").is_err());
    }

    #[test]
    fn test_scheduled_task_type_roundtrip() {
        let variants = [
            ScheduledTaskType::CibScan,
            ScheduledTaskType::SourceDiscovery,
            ScheduledTaskType::FactCheckSync,
            ScheduledTaskType::ReportGeneration,
            ScheduledTaskType::McpPoll,
            ScheduledTaskType::FingerprintRefresh,
            ScheduledTaskType::VelocityBaselineUpdate,
        ];
        for v in &variants {
            let s = v.as_str();
            let parsed: ScheduledTaskType = s.parse().unwrap();
            assert_eq!(&parsed, v);
        }
    }

    #[test]
    fn test_scheduled_task_type_unknown() {
        let res = "banana".parse::<ScheduledTaskType>();
        assert!(res.is_err());
    }

    #[test]
    fn test_task_schedule_serde_interval() {
        let s = TaskSchedule::Interval { seconds: 3600 };
        let json = serde_json::to_string(&s).unwrap();
        let deser: TaskSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(s, deser);
    }

    #[test]
    fn test_task_schedule_serde_once() {
        let now = Utc::now();
        let s = TaskSchedule::Once { at: now };
        let json = serde_json::to_string(&s).unwrap();
        let deser: TaskSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(s, deser);
    }

    #[test]
    fn test_task_result_serde() {
        let r = TaskResult::Success {
            duration_ms: 42,
            summary: "ok".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, deser);
    }

    #[test]
    fn test_scheduled_task_new() {
        let task = ScheduledTask::new(
            ScheduledTaskType::CibScan,
            TaskSchedule::Interval { seconds: 3600 },
        );
        assert!(task.enabled);
        assert!(task.last_run.is_none());
        assert!(task.last_result.is_none());
        assert_eq!(task.task_type, ScheduledTaskType::CibScan);
    }
}
