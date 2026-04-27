//! Task scheduler engine (Sprint D8.1).
//!
//! Persistent, configurable scheduler that runs analysis jobs on intervals.
//! Foundation for all autonomous operations in D8.
//!
//! KV layout:
//! - `sched/{task_uuid}`                  → [`ScheduledTask`]
//! - `sched/hist/{task_uuid}/{ts_millis}` → [`TaskHistory`]
//! - `sched/lock/{task_uuid}`             → lock timestamp (ASCII)

pub mod cib_scan;
pub mod engine;
pub mod types;

pub use cib_scan::{
    execute_cib_scan, execute_fingerprint_refresh, CibScanConfig, CibScanDelta,
    FingerprintRefreshConfig,
};
pub use engine::*;
pub use types::*;
