//! Fact-check database sync engine (Sprint D8.4).
//!
//! Syncs fact-checks from external sources (Google Fact Check Tools API,
//! ClaimsKG, RSS feeds) into TENSA's claim-matching database. Sync results
//! are persisted at `fc/sync/{timestamp}` for audit trail.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// Configured fact-check sync source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckSource {
    pub name: String,
    pub source_type: FactCheckSourceType,
    pub enabled: bool,
    pub last_sync: Option<DateTime<Utc>>,
}

/// Supported fact-check data providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactCheckSourceType {
    GoogleFactCheck,
    ClaimsKG,
    Rss,
}

/// Result of a single sync run against one source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    pub source: String,
    pub new_claims: usize,
    pub duplicates_skipped: usize,
    pub errors: usize,
    pub synced_at: DateTime<Utc>,
}

// ─── Public API ─────────────────────────────────────────────

/// Execute a fact-check sync for all enabled sources.
///
/// Currently a stub that returns sync metadata — actual HTTP calls to
/// external APIs would be added when network features are available.
pub fn execute_sync(
    hypergraph: &Hypergraph,
    sources: &[FactCheckSource],
    languages: &[String],
) -> Result<Vec<SyncResult>> {
    let mut results = Vec::new();

    for source in sources {
        if !source.enabled {
            continue;
        }

        let result = match source.source_type {
            FactCheckSourceType::GoogleFactCheck => {
                sync_google_fact_check(hypergraph, &source.name, languages)?
            }
            FactCheckSourceType::ClaimsKG => sync_claimskg(hypergraph, &source.name)?,
            FactCheckSourceType::Rss => sync_rss_feeds(hypergraph, &source.name)?,
        };

        // Record sync run.
        store_sync_result(hypergraph, &result)?;
        results.push(result);
    }

    Ok(results)
}

/// Persist a sync result at `fc/sync/{timestamp}`.
pub fn store_sync_result(hypergraph: &Hypergraph, result: &SyncResult) -> Result<()> {
    let key = format!("fc/sync/{}", result.synced_at.timestamp_millis());
    let value = serde_json::to_vec(result).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(key.as_bytes(), &value)
}

/// Load sync history (most recent first, up to `limit`).
pub fn list_sync_history(hypergraph: &Hypergraph, limit: usize) -> Result<Vec<SyncResult>> {
    let prefix = b"fc/sync/";
    let pairs = hypergraph.store().prefix_scan(prefix)?;
    let mut results: Vec<SyncResult> = pairs
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
        .collect();
    results.sort_by(|a, b| b.synced_at.cmp(&a.synced_at));
    results.truncate(limit);
    Ok(results)
}

/// Return the default set of fact-check sources.
pub fn default_sources() -> Vec<FactCheckSource> {
    vec![
        FactCheckSource {
            name: "Google Fact Check Tools".to_string(),
            source_type: FactCheckSourceType::GoogleFactCheck,
            enabled: true,
            last_sync: None,
        },
        FactCheckSource {
            name: "ClaimsKG".to_string(),
            source_type: FactCheckSourceType::ClaimsKG,
            enabled: true,
            last_sync: None,
        },
        FactCheckSource {
            name: "RSS Fact-Check Feeds".to_string(),
            source_type: FactCheckSourceType::Rss,
            enabled: true,
            last_sync: None,
        },
    ]
}

// ─── Stub Sync Implementations ──────────────────────────────

/// Stub: sync from Google Fact Check Tools API.
/// In production, this would make HTTP requests to the API.
fn sync_google_fact_check(
    _hypergraph: &Hypergraph,
    source_name: &str,
    _languages: &[String],
) -> Result<SyncResult> {
    Ok(SyncResult {
        source: source_name.to_string(),
        new_claims: 0,
        duplicates_skipped: 0,
        errors: 0,
        synced_at: Utc::now(),
    })
}

/// Stub: sync from ClaimsKG SPARQL endpoint.
fn sync_claimskg(_hypergraph: &Hypergraph, source_name: &str) -> Result<SyncResult> {
    Ok(SyncResult {
        source: source_name.to_string(),
        new_claims: 0,
        duplicates_skipped: 0,
        errors: 0,
        synced_at: Utc::now(),
    })
}

/// Stub: sync from RSS feeds.
fn sync_rss_feeds(_hypergraph: &Hypergraph, source_name: &str) -> Result<SyncResult> {
    Ok(SyncResult {
        source: source_name.to_string(),
        new_claims: 0,
        duplicates_skipped: 0,
        errors: 0,
        synced_at: Utc::now(),
    })
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_execute_sync_all_enabled() {
        let hg = setup();
        let sources = default_sources();
        let results = execute_sync(&hg, &sources, &["en".to_string()]).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.errors, 0);
        }
    }

    #[test]
    fn test_store_and_list_sync_history() {
        let hg = setup();
        let result = SyncResult {
            source: "test".to_string(),
            new_claims: 5,
            duplicates_skipped: 2,
            errors: 0,
            synced_at: Utc::now(),
        };
        store_sync_result(&hg, &result).unwrap();
        let history = list_sync_history(&hg, 10).unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].new_claims, 5);
    }

    #[test]
    fn test_disabled_source_skipped() {
        let hg = setup();
        let sources = vec![FactCheckSource {
            name: "disabled".to_string(),
            source_type: FactCheckSourceType::Rss,
            enabled: false,
            last_sync: None,
        }];
        let results = execute_sync(&hg, &sources, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_default_sources() {
        let sources = default_sources();
        assert_eq!(sources.len(), 3);
        assert!(sources.iter().all(|s| s.enabled));
        assert!(sources.iter().all(|s| s.last_sync.is_none()));
    }

    #[test]
    fn test_sync_result_serde() {
        let result = SyncResult {
            source: "test-src".to_string(),
            new_claims: 3,
            duplicates_skipped: 1,
            errors: 0,
            synced_at: Utc::now(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: SyncResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.source, "test-src");
        assert_eq!(deser.new_claims, 3);
    }

    #[test]
    fn test_list_sync_history_respects_limit() {
        let hg = setup();
        for i in 0..5 {
            let result = SyncResult {
                source: format!("src-{}", i),
                new_claims: i,
                duplicates_skipped: 0,
                errors: 0,
                synced_at: Utc::now() + chrono::Duration::milliseconds(i as i64),
            };
            store_sync_result(&hg, &result).unwrap();
        }
        let history = list_sync_history(&hg, 3).unwrap();
        assert_eq!(history.len(), 3);
    }
}
