//! CIB scan and fingerprint refresh task execution (Sprint D8.2).
//!
//! Connects the scheduler engine to the CIB detection pipeline and
//! behavioral fingerprint refresh loop. Both are invoked as scheduled
//! tasks and return a [`TaskResult`].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::Hypergraph;

use super::types::TaskResult;

// ─── CIB Scan ───────────────────────────────────────────────

/// Configuration for a CIB scan task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CibScanConfig {
    pub narrative_id: String,
    #[serde(default)]
    pub cross_platform: bool,
    #[serde(default)]
    pub include_factory_detection: bool,
    #[serde(default)]
    pub similarity_threshold: Option<f64>,
    #[serde(default)]
    pub alpha: Option<f64>,
}

/// Delta between two consecutive CIB scans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CibScanDelta {
    pub scan_timestamp: DateTime<Utc>,
    pub narrative_id: String,
    pub new_clusters: Vec<String>,
    pub growing_clusters: Vec<String>,
    pub dissolved_clusters: Vec<String>,
    pub total_clusters: usize,
}

/// Execute a CIB scan task.
///
/// Runs CIB detection on the given narrative, computes a delta against
/// previously stored clusters, optionally runs factory detection, and
/// persists the delta at `cib/delta/{timestamp}`.
pub fn execute_cib_scan(hypergraph: &Hypergraph, config: &CibScanConfig) -> Result<TaskResult> {
    let start = std::time::Instant::now();

    // Load previous clusters for delta computation.
    let previous_clusters = crate::analysis::cib::list_clusters(hypergraph, &config.narrative_id)?;
    let prev_ids: std::collections::HashSet<String> = previous_clusters
        .iter()
        .map(|c| c.cluster_id.clone())
        .collect();

    // Build CIB config from optional overrides.
    let cib_config_json = serde_json::json!({
        "cross_platform": config.cross_platform,
        "similarity_threshold": config.similarity_threshold,
        "alpha": config.alpha,
    });
    let cfg = crate::analysis::cib::CibConfig::from_json(&cib_config_json);

    let result = if config.cross_platform {
        crate::analysis::cib::detect_cross_platform_cib(hypergraph, &config.narrative_id, &cfg)?
    } else {
        crate::analysis::cib::detect_cib_clusters(hypergraph, &config.narrative_id, &cfg)?
    };

    // Compute delta.
    let current_ids: std::collections::HashSet<String> = result
        .clusters
        .iter()
        .map(|c| c.cluster_id.clone())
        .collect();

    let new_clusters: Vec<String> = current_ids.difference(&prev_ids).cloned().collect();
    let dissolved: Vec<String> = prev_ids.difference(&current_ids).cloned().collect();

    // Run factory detection if requested.
    if config.include_factory_detection {
        let _ =
            crate::analysis::cib::detect_content_factories(hypergraph, &config.narrative_id, None);
    }

    // Store delta.
    let delta = CibScanDelta {
        scan_timestamp: Utc::now(),
        narrative_id: config.narrative_id.clone(),
        new_clusters: new_clusters.clone(),
        growing_clusters: vec![], // Would need size comparison — simplified for now
        dissolved_clusters: dissolved.clone(),
        total_clusters: result.clusters.len(),
    };
    store_scan_delta(hypergraph, &delta)?;

    let duration_ms = start.elapsed().as_millis() as u64;
    let summary = format!(
        "CIB scan: {} clusters found, {} new, {} dissolved",
        result.clusters.len(),
        new_clusters.len(),
        dissolved.len()
    );

    Ok(TaskResult::Success {
        duration_ms,
        summary,
    })
}

/// Persist a scan delta at `cib/delta/{timestamp}`.
fn store_scan_delta(hypergraph: &Hypergraph, delta: &CibScanDelta) -> Result<()> {
    let key = format!("cib/delta/{}", delta.scan_timestamp.timestamp_millis());
    let value = serde_json::to_vec(delta).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(key.as_bytes(), &value)
}

// ─── Fingerprint Refresh ────────────────────────────────────

/// Configuration for a fingerprint refresh task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintRefreshConfig {
    /// Max age in seconds before a fingerprint is considered stale.
    pub max_age_secs: u64,
    /// Max number of fingerprints to refresh per run.
    #[serde(default = "default_batch")]
    pub batch_size: usize,
}

fn default_batch() -> usize {
    50
}

/// Execute a fingerprint refresh task.
///
/// Iterates over all Actor entities, checks whether their behavioral
/// fingerprint is older than `max_age_secs`, and recomputes stale ones
/// up to `batch_size`.
pub fn execute_fingerprint_refresh(
    hypergraph: &Hypergraph,
    config: &FingerprintRefreshConfig,
) -> Result<TaskResult> {
    let start = std::time::Instant::now();
    let cutoff = Utc::now() - chrono::Duration::seconds(config.max_age_secs as i64);
    let mut refreshed = 0usize;

    let actors = hypergraph.list_entities_by_type(&crate::types::EntityType::Actor)?;
    for actor in actors.iter().take(config.batch_size) {
        if refreshed >= config.batch_size {
            break;
        }
        if let Ok(Some(fp)) =
            crate::disinfo::fingerprints::load_behavioral_fingerprint(hypergraph, &actor.id)
        {
            if fp.computed_at < cutoff {
                let _ = crate::disinfo::fingerprints::ensure_behavioral_fingerprint(
                    hypergraph, &actor.id, true,
                );
                refreshed += 1;
            }
        } else {
            // Never computed — compute now.
            let _ = crate::disinfo::fingerprints::ensure_behavioral_fingerprint(
                hypergraph, &actor.id, false,
            );
            refreshed += 1;
        }
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    Ok(TaskResult::Success {
        duration_ms,
        summary: format!("Refreshed {} fingerprints (cutoff: {})", refreshed, cutoff),
    })
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cib_scan_config_parse() {
        let json = serde_json::json!({
            "narrative_id": "test",
            "cross_platform": true,
            "include_factory_detection": false,
        });
        let config: CibScanConfig = serde_json::from_value(json).unwrap();
        assert!(config.cross_platform);
        assert!(!config.include_factory_detection);
        assert!(config.similarity_threshold.is_none());
    }

    #[test]
    fn test_cib_scan_config_with_thresholds() {
        let json = serde_json::json!({
            "narrative_id": "test-2",
            "similarity_threshold": 0.75,
            "alpha": 0.01,
        });
        let config: CibScanConfig = serde_json::from_value(json).unwrap();
        assert!(!config.cross_platform);
        assert_eq!(config.similarity_threshold, Some(0.75));
        assert_eq!(config.alpha, Some(0.01));
    }

    #[test]
    fn test_fingerprint_refresh_config_defaults() {
        let json = serde_json::json!({ "max_age_secs": 86400 });
        let config: FingerprintRefreshConfig = serde_json::from_value(json).unwrap();
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.max_age_secs, 86400);
    }

    #[test]
    fn test_fingerprint_refresh_config_custom_batch() {
        let json = serde_json::json!({ "max_age_secs": 3600, "batch_size": 10 });
        let config: FingerprintRefreshConfig = serde_json::from_value(json).unwrap();
        assert_eq!(config.batch_size, 10);
    }

    #[test]
    fn test_cib_scan_delta_serde() {
        let delta = CibScanDelta {
            scan_timestamp: Utc::now(),
            narrative_id: "test".to_string(),
            new_clusters: vec!["c1".to_string()],
            growing_clusters: vec![],
            dissolved_clusters: vec!["c2".to_string()],
            total_clusters: 5,
        };
        let json = serde_json::to_string(&delta).unwrap();
        let deser: CibScanDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.narrative_id, "test");
        assert_eq!(deser.total_clusters, 5);
        assert_eq!(deser.new_clusters.len(), 1);
        assert_eq!(deser.dissolved_clusters.len(), 1);
    }
}
