//! Monitor subscriptions and alerts for disinfo narratives (Sprint D7.3).
//!
//! A subscription watches for new posts matching a narrative's behavioral
//! profile. When a post arrives (via [`check_post`]), it is compared against
//! all active subscriptions whose platform filter matches. If the post content
//! yields a similarity score above the subscription's threshold, a
//! [`MonitorAlert`] is fired and persisted.
//!
//! Persistence:
//! - `mon/{subscription_uuid}` → [`MonitorSubscription`]
//! - `mon/alert/{alert_uuid}` → [`MonitorAlert`]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::types::Platform;

// ─── Types ────────────────────────────────────────────────────

/// How to deliver an alert when a subscription fires.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MonitorCallback {
    /// Write to the TENSA log (default).
    Log,
    /// POST the alert JSON to a webhook URL.
    Webhook(String),
    /// Enqueue on an internal processing queue (future use).
    InternalQueue,
}

impl Default for MonitorCallback {
    fn default() -> Self {
        Self::Log
    }
}

/// A subscription that watches for posts matching a narrative's profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorSubscription {
    pub id: Uuid,
    /// Narrative this subscription is watching for.
    pub narrative_id: String,
    /// Only match posts from these platforms (empty = all platforms).
    #[serde(default)]
    pub platforms: Vec<Platform>,
    /// Minimum similarity score (0.0–1.0) to fire an alert.
    pub threshold: f64,
    /// How to deliver the alert.
    #[serde(default)]
    pub callback: MonitorCallback,
    /// Whether this subscription is currently active.
    #[serde(default = "default_active")]
    pub active: bool,
    pub created_at: DateTime<Utc>,
}

fn default_active() -> bool {
    true
}

/// An alert fired when a post matches a subscription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorAlert {
    pub id: Uuid,
    pub subscription_id: Uuid,
    /// The post that triggered this alert (if identifiable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_id: Option<Uuid>,
    /// Similarity distance that caused the match.
    pub distance: f64,
    pub matched_at: DateTime<Utc>,
}

// ─── KV Keys ──────────────────────────────────────────────────

fn subscription_key(id: &Uuid) -> Vec<u8> {
    let mut key = keys::MONITOR_SUBSCRIPTION.to_vec();
    key.extend_from_slice(id.as_bytes());
    key
}

fn alert_key(id: &Uuid) -> Vec<u8> {
    let mut key = keys::MONITOR_ALERT.to_vec();
    key.extend_from_slice(id.as_bytes());
    key
}

// ─── CRUD ─────────────────────────────────────────────────────

/// Create a new monitor subscription and persist it.
pub fn create_subscription(
    hypergraph: &Hypergraph,
    mut sub: MonitorSubscription,
) -> Result<MonitorSubscription> {
    if sub.id.is_nil() {
        sub.id = Uuid::now_v7();
    }
    let bytes = serde_json::to_vec(&sub).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(&subscription_key(&sub.id), &bytes)?;
    Ok(sub)
}

/// List all subscriptions (optionally filtered to active only).
///
/// Uses a length check to skip `mon/alert/` sub-prefix keys efficiently
/// (subscription keys are `mon/` + 16 UUID bytes = 20 bytes, while alert
/// keys start with `mon/alert/` which is 10 bytes before the UUID).
pub fn list_subscriptions(
    hypergraph: &Hypergraph,
    active_only: bool,
) -> Result<Vec<MonitorSubscription>> {
    let pairs = hypergraph.store().prefix_scan(keys::MONITOR_SUBSCRIPTION)?;
    let alert_prefix_len = keys::MONITOR_ALERT.len();
    let mut out = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        // Skip alert keys which share the `mon/` prefix.
        if key.len() >= alert_prefix_len && key[..alert_prefix_len] == *keys::MONITOR_ALERT {
            continue;
        }
        if let Ok(sub) = serde_json::from_slice::<MonitorSubscription>(&value) {
            if !active_only || sub.active {
                out.push(sub);
            }
        }
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

/// Delete a subscription by ID.
pub fn delete_subscription(hypergraph: &Hypergraph, id: &Uuid) -> Result<()> {
    hypergraph.store().delete(&subscription_key(id))
}

/// Persist an alert.
fn store_alert(hypergraph: &Hypergraph, alert: &MonitorAlert) -> Result<()> {
    let bytes = serde_json::to_vec(alert).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(&alert_key(&alert.id), &bytes)
}

/// List recent alerts (most recent first, up to `limit`).
pub fn list_alerts(hypergraph: &Hypergraph, limit: usize) -> Result<Vec<MonitorAlert>> {
    let pairs = hypergraph.store().prefix_scan(keys::MONITOR_ALERT)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        if let Ok(alert) = serde_json::from_slice::<MonitorAlert>(&value) {
            out.push(alert);
        }
    }
    out.sort_by(|a, b| b.matched_at.cmp(&a.matched_at));
    out.truncate(limit);
    Ok(out)
}

// ─── Post Checking ────────────────────────────────────────────

/// Check a post against all active subscriptions.
///
/// For each active subscription whose platform filter matches, a simple
/// text-overlap similarity between the post content and the narrative's
/// situations is computed. If the score exceeds the subscription threshold,
/// a [`MonitorAlert`] is fired.
///
/// The similarity heuristic uses Jaccard overlap on word-level unigrams.
/// This is intentionally lightweight (no LLM, no embedding) so it can run
/// inline on every incoming post.
pub fn check_post(
    hypergraph: &Hypergraph,
    post_content: &str,
    platform: &Platform,
    post_id: Option<Uuid>,
) -> Result<Vec<MonitorAlert>> {
    let subs = list_subscriptions(hypergraph, true)?;
    if subs.is_empty() || post_content.is_empty() {
        return Ok(vec![]);
    }

    let post_words = tokenize(post_content);
    let mut alerts = Vec::new();
    let now = Utc::now();

    for sub in &subs {
        // Platform filter
        if !sub.platforms.is_empty() && !sub.platforms.contains(platform) {
            continue;
        }

        // Collect narrative content words
        let narrative_words = narrative_word_set(hypergraph, &sub.narrative_id)?;
        if narrative_words.is_empty() {
            continue;
        }

        let sim = jaccard_similarity(&post_words, &narrative_words);
        if sim >= sub.threshold {
            let alert = MonitorAlert {
                id: Uuid::now_v7(),
                subscription_id: sub.id,
                post_id,
                distance: sim,
                matched_at: now,
            };
            store_alert(hypergraph, &alert)?;
            alerts.push(alert);
        }
    }

    Ok(alerts)
}

/// Tokenize text into a set of lowercase word-level unigrams.
fn tokenize(text: &str) -> std::collections::HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() >= 2)
        .collect()
}

/// Build a word set from all situation content in a narrative.
fn narrative_word_set(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<std::collections::HashSet<String>> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let mut words = std::collections::HashSet::new();
    for sit in &situations {
        for block in &sit.raw_content {
            for w in tokenize(&block.content) {
                words.insert(w);
            }
        }
        if let Some(ref name) = sit.name {
            for w in tokenize(name) {
                words.insert(w);
            }
        }
    }
    Ok(words)
}

/// Jaccard similarity between two word sets.
fn jaccard_similarity(
    a: &std::collections::HashSet<String>,
    b: &std::collections::HashSet<String>,
) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ─── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use std::sync::Arc;

    fn make_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_sub(narrative_id: &str, threshold: f64) -> MonitorSubscription {
        MonitorSubscription {
            id: Uuid::now_v7(),
            narrative_id: narrative_id.to_string(),
            platforms: vec![],
            threshold,
            callback: MonitorCallback::Log,
            active: true,
            created_at: Utc::now(),
        }
    }

    fn add_situation_with_text(hg: &Hypergraph, narrative: &str, text: &str) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(text)],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_subscription_crud() {
        let hg = make_hg();
        let sub = make_sub("narr-mon", 0.3);
        let id = sub.id;

        let created = create_subscription(&hg, sub).unwrap();
        assert_eq!(created.id, id);

        let subs = list_subscriptions(&hg, false).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].narrative_id, "narr-mon");

        delete_subscription(&hg, &id).unwrap();
        let subs = list_subscriptions(&hg, false).unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn test_list_active_only() {
        let hg = make_hg();
        create_subscription(&hg, make_sub("narr-a", 0.5)).unwrap();
        let mut inactive = make_sub("narr-b", 0.5);
        inactive.active = false;
        create_subscription(&hg, inactive).unwrap();

        let all = list_subscriptions(&hg, false).unwrap();
        assert_eq!(all.len(), 2);
        let active = list_subscriptions(&hg, true).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].narrative_id, "narr-a");
    }

    #[test]
    fn test_check_post_fires_alert() {
        let hg = make_hg();
        // Create narrative content
        add_situation_with_text(
            &hg,
            "narr-watch",
            "disinformation campaign detected in ukraine conflict",
        );

        // Create subscription with low threshold
        create_subscription(&hg, make_sub("narr-watch", 0.1)).unwrap();

        // Check a matching post
        let alerts = check_post(
            &hg,
            "disinformation campaign detected in ukraine",
            &Platform::Twitter,
            Some(Uuid::now_v7()),
        )
        .unwrap();
        assert!(
            !alerts.is_empty(),
            "should fire at least one alert for matching content"
        );
        assert!(alerts[0].distance >= 0.1);

        // Verify alert is persisted
        let loaded = list_alerts(&hg, 10).unwrap();
        assert_eq!(loaded.len(), alerts.len());
    }

    #[test]
    fn test_check_post_no_match() {
        let hg = make_hg();
        add_situation_with_text(&hg, "narr-specific", "climate change policy reform");
        create_subscription(&hg, make_sub("narr-specific", 0.8)).unwrap();

        // Post about completely different topic
        let alerts = check_post(
            &hg,
            "kittens playing in the garden",
            &Platform::Twitter,
            None,
        )
        .unwrap();
        assert!(alerts.is_empty(), "unrelated content should not fire");
    }

    #[test]
    fn test_platform_filter() {
        let hg = make_hg();
        add_situation_with_text(&hg, "narr-tw-only", "important keyword match test");

        let mut sub = make_sub("narr-tw-only", 0.1);
        sub.platforms = vec![Platform::Telegram]; // only watch Telegram
        create_subscription(&hg, sub).unwrap();

        // Post on Twitter — should be filtered out
        let alerts = check_post(
            &hg,
            "important keyword match test here",
            &Platform::Twitter,
            None,
        )
        .unwrap();
        assert!(alerts.is_empty(), "wrong platform should be filtered");

        // Post on Telegram — should match
        let alerts = check_post(
            &hg,
            "important keyword match test here",
            &Platform::Telegram,
            None,
        )
        .unwrap();
        assert!(!alerts.is_empty(), "correct platform should match");
    }

    #[test]
    fn test_alert_round_trip() {
        let hg = make_hg();
        let alert = MonitorAlert {
            id: Uuid::now_v7(),
            subscription_id: Uuid::now_v7(),
            post_id: Some(Uuid::now_v7()),
            distance: 0.75,
            matched_at: Utc::now(),
        };
        store_alert(&hg, &alert).unwrap();
        let loaded = list_alerts(&hg, 10).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!((loaded[0].distance - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: std::collections::HashSet<String> = ["hello", "world", "test"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let b: std::collections::HashSet<String> = ["hello", "world", "other"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let sim = jaccard_similarity(&a, &b);
        // intersection = {hello, world} = 2, union = {hello, world, test, other} = 4
        assert!((sim - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_empty_sets_zero_similarity() {
        let empty: std::collections::HashSet<String> = std::collections::HashSet::new();
        let non_empty: std::collections::HashSet<String> =
            ["hello"].iter().map(|s| s.to_string()).collect();
        assert_eq!(jaccard_similarity(&empty, &non_empty), 0.0);
        assert_eq!(jaccard_similarity(&empty, &empty), 0.0);
    }
}
