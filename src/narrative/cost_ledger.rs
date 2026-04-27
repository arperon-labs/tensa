//! Cost ledger — per-narrative record of AI operation costs (Sprint W5, v0.49.4).
//!
//! Every generation / edit / workshop LLM call writes one [`CostLedgerEntry`]
//! via [`record`]. Entries are append-only at `cl/{narrative_id}/{entry_uuid_v7}`
//! so prefix-scan yields chronological order. No secondary index — list-recent
//! is a bounded tail scan.
//!
//! # Simplify
//! - Token counts use the same chars/4 heuristic as generation/editing/workshop
//!   (`writer_common::approx_tokens`). No provider-side token reconciliation
//!   in v1 — the ledger is advisory, not billing-grade.
//! - Failed calls still record so the writer sees wasted prompt tokens.
//! - Cache hits record `cache_hit = true` with `response_tokens = 0` so the
//!   rollup shows "would-have-cost" vs "actually-cost".

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::keys;
use crate::store::KVStore;
use crate::types::*;

/// Append a new cost record. Returns the stored entry (with fresh id + timestamp).
///
/// Never fails the calling operation: `record` is best-effort — on KV error the
/// error is swallowed and logged. This keeps a failing ledger from breaking the
/// actual AI operation.
#[allow(clippy::too_many_arguments)]
pub fn record(
    store: &dyn KVStore,
    narrative_id: &str,
    operation: CostOperation,
    kind: impl Into<String>,
    prompt_tokens: u32,
    response_tokens: u32,
    model: Option<String>,
    cache_hit: bool,
    success: bool,
    duration_ms: u64,
    metadata: Option<serde_json::Value>,
) -> CostLedgerEntry {
    let entry = CostLedgerEntry {
        id: Uuid::now_v7(),
        narrative_id: narrative_id.to_string(),
        operation,
        kind: kind.into(),
        prompt_tokens,
        response_tokens,
        model,
        cache_hit,
        success,
        duration_ms,
        created_at: Utc::now(),
        metadata,
    };
    let key = keys::cost_ledger_key(narrative_id, &entry.id);
    match serde_json::to_vec(&entry) {
        Ok(bytes) => {
            if let Err(e) = store.put(&key, &bytes) {
                tracing::warn!("cost ledger write failed: {}", e);
            }
        }
        Err(e) => tracing::warn!("cost ledger serialize failed: {}", e),
    }
    entry
}

/// Paged / capped list of recent entries, newest first.
pub fn list(store: &dyn KVStore, narrative_id: &str, limit: usize) -> Result<Vec<CostLedgerEntry>> {
    let prefix = keys::cost_ledger_narrative_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    let mut out: Vec<CostLedgerEntry> = Vec::new();
    // Prefix scan yields ascending; take the tail (most recent).
    let start = pairs.len().saturating_sub(limit);
    for (_, value) in pairs.into_iter().skip(start) {
        if let Ok(entry) = serde_json::from_slice::<CostLedgerEntry>(&value) {
            out.push(entry);
        }
    }
    out.reverse();
    Ok(out)
}

/// Aggregate entries inside a time window (None = all-time).
pub fn summary(
    store: &dyn KVStore,
    narrative_id: &str,
    window: Option<Duration>,
) -> Result<CostSummary> {
    let prefix = keys::cost_ledger_narrative_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    let cutoff: Option<DateTime<Utc>> = window.map(|w| Utc::now() - w);

    let mut totals = CostSummary {
        narrative_id: narrative_id.to_string(),
        window: window_label(window),
        ..Default::default()
    };
    let mut by_op: std::collections::HashMap<CostOperation, CostOperationSummary> =
        std::collections::HashMap::new();

    for (_, value) in pairs {
        let entry: CostLedgerEntry = match serde_json::from_slice(&value) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if let Some(cut) = cutoff {
            if entry.created_at < cut {
                continue;
            }
        }
        totals.total_calls += 1;
        if entry.cache_hit {
            totals.cache_hits += 1;
        }
        totals.total_prompt_tokens += entry.prompt_tokens as u64;
        totals.total_response_tokens += entry.response_tokens as u64;
        totals.total_duration_ms += entry.duration_ms;

        let e = by_op
            .entry(entry.operation)
            .or_insert(CostOperationSummary {
                operation: entry.operation,
                calls: 0,
                cache_hits: 0,
                prompt_tokens: 0,
                response_tokens: 0,
            });
        e.calls += 1;
        if entry.cache_hit {
            e.cache_hits += 1;
        }
        e.prompt_tokens += entry.prompt_tokens as u64;
        e.response_tokens += entry.response_tokens as u64;
    }

    totals.by_operation = by_op.into_values().collect();
    totals
        .by_operation
        .sort_by_key(|o| format!("{:?}", o.operation));
    Ok(totals)
}

fn window_label(window: Option<Duration>) -> String {
    match window {
        None => "all".into(),
        Some(w) => {
            let days = w.num_days();
            if days >= 1 {
                format!("{}d", days)
            } else {
                format!("{}h", w.num_hours())
            }
        }
    }
}

/// Parse a "30d" / "7d" / "24h" / "all" label to a Duration.
pub fn parse_window(label: &str) -> Option<Duration> {
    let s = label.trim().to_lowercase();
    if s == "all" {
        return None;
    }
    if let Some(num) = s.strip_suffix('d') {
        num.parse::<i64>().ok().map(Duration::days)
    } else if let Some(num) = s.strip_suffix('h') {
        num.parse::<i64>().ok().map(Duration::hours)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn fresh_store() -> MemoryStore {
        MemoryStore::new()
    }

    #[test]
    fn record_and_list() {
        let s = fresh_store();
        let entry = record(
            &s,
            "draft",
            CostOperation::Generation,
            "outline",
            500,
            300,
            Some("claude".into()),
            false,
            true,
            1234,
            None,
        );
        assert_eq!(entry.prompt_tokens, 500);
        let list = list(&s, "draft", 10).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, entry.id);
    }

    #[test]
    fn list_is_newest_first_and_bounded() {
        let s = fresh_store();
        for i in 0..5 {
            record(
                &s,
                "draft",
                CostOperation::Generation,
                "outline",
                100 * i as u32,
                50,
                None,
                false,
                true,
                100,
                None,
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let list = list(&s, "draft", 3).unwrap();
        assert_eq!(list.len(), 3);
        // Newest first — i=4 was the last recorded.
        assert_eq!(list[0].prompt_tokens, 400);
    }

    #[test]
    fn summary_aggregates_by_operation() {
        let s = fresh_store();
        record(
            &s,
            "draft",
            CostOperation::Generation,
            "outline",
            500,
            300,
            None,
            false,
            true,
            100,
            None,
        );
        record(
            &s,
            "draft",
            CostOperation::Generation,
            "character",
            400,
            200,
            None,
            true,
            true,
            10,
            None,
        );
        record(
            &s,
            "draft",
            CostOperation::Edit,
            "tighten",
            200,
            150,
            None,
            false,
            true,
            50,
            None,
        );
        record(
            &s,
            "draft",
            CostOperation::Workshop,
            "standard",
            1200,
            400,
            None,
            false,
            true,
            1000,
            None,
        );

        let sum = summary(&s, "draft", None).unwrap();
        assert_eq!(sum.total_calls, 4);
        assert_eq!(sum.cache_hits, 1);
        assert_eq!(sum.total_prompt_tokens, 500 + 400 + 200 + 1200);
        assert_eq!(sum.by_operation.len(), 3);
    }

    #[test]
    fn summary_honors_window_cutoff() {
        let s = fresh_store();
        // Fake an old entry by writing it directly with a past timestamp.
        let mut old = CostLedgerEntry {
            id: Uuid::now_v7(),
            narrative_id: "draft".into(),
            operation: CostOperation::Generation,
            kind: "outline".into(),
            prompt_tokens: 100,
            response_tokens: 50,
            model: None,
            cache_hit: false,
            success: true,
            duration_ms: 10,
            created_at: Utc::now() - Duration::days(40),
            metadata: None,
        };
        let key = keys::cost_ledger_key("draft", &old.id);
        s.put(&key, &serde_json::to_vec(&old).unwrap()).unwrap();
        old.id = Uuid::now_v7();
        old.created_at = Utc::now();
        let key = keys::cost_ledger_key("draft", &old.id);
        s.put(&key, &serde_json::to_vec(&old).unwrap()).unwrap();

        let sum = summary(&s, "draft", Some(Duration::days(30))).unwrap();
        assert_eq!(
            sum.total_calls, 1,
            "entry older than 30 days should be excluded"
        );
    }

    #[test]
    fn parse_window_understands_units() {
        assert_eq!(parse_window("30d").map(|d| d.num_days()), Some(30));
        assert_eq!(parse_window("24h").map(|d| d.num_hours()), Some(24));
        assert_eq!(parse_window("all"), None);
        assert!(parse_window("xyz").is_none());
    }
}
