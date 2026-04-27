//! MCP client orchestrator for multi-platform source ingestion (Sprint D6).
//!
//! The orchestrator manages a registry of MCP sources (social media connectors,
//! RSS feeds, web scrapers) and normalizes their heterogeneous outputs into a
//! common [`NormalizedPost`] representation suitable for entity extraction,
//! spread analysis, and CIB detection.
//!
//! Audit entries are persisted at the KV prefix
//! [`crate::hypergraph::keys::MCP_AUDIT`] (`mcp/audit/`) for provenance
//! tracking and rate-limit compliance monitoring.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::{EngagementMetrics, Platform};

// ─── NormalizedPost ──────────────────────────────────────────

/// Platform-agnostic representation of a social media post / content item.
///
/// All platform-specific MCP tool outputs are mapped into this structure
/// before entering the TENSA ingestion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedPost {
    /// TENSA-internal unique identifier.
    pub id: Uuid,
    /// Originating platform.
    pub platform: Platform,
    /// Platform-native post identifier (e.g. tweet ID, Telegram message ID).
    pub platform_post_id: String,
    /// Author's platform-native identifier.
    pub author_id: String,
    /// Post textual content (may be empty for media-only posts).
    pub content: String,
    /// Detected language code (ISO 639-1).
    pub language: String,
    /// Publication timestamp.
    pub timestamp: DateTime<Utc>,
    /// Engagement metrics snapshot, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engagement: Option<EngagementMetrics>,
    /// Extracted hashtags (without the `#` prefix).
    #[serde(default)]
    pub hashtags: Vec<String>,
    /// Mentioned user handles.
    #[serde(default)]
    pub mentions: Vec<String>,
    /// URLs found in the post body.
    #[serde(default)]
    pub urls: Vec<String>,
    /// Platform-native ID of the parent post if this is a reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
    /// Platform-native ID of the original post if this is a repost/retweet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repost_of: Option<String>,
    /// Full raw JSON from the MCP tool response (for provenance).
    #[serde(default)]
    pub raw_json: serde_json::Value,
    /// Name of the MCP source that produced this post.
    pub ingestion_source: String,
    /// Extraction confidence (0.0–1.0).
    pub confidence: f64,
}

// ─── McpSource ───────────────────────────────────────────────

/// Registration entry for an MCP data source (social media connector).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSource {
    /// Human-readable name (e.g. "twitter-search", "telegram-monitor").
    pub name: String,
    /// Target platform.
    pub platform: Platform,
    /// MCP tool names this source exposes.
    #[serde(default)]
    pub tools_used: Vec<String>,
    /// Requests-per-minute rate limit.
    #[serde(default = "default_rate_limit")]
    pub rate_limit_rpm: u32,
    /// Priority for scheduling (lower = higher priority).
    #[serde(default)]
    pub priority: u8,
    /// Whether this source is currently active.
    #[serde(default = "default_active")]
    pub active: bool,
}

fn default_rate_limit() -> u32 {
    60
}

fn default_active() -> bool {
    true
}

// ─── SourceRegistry ──────────────────────────────────────────

/// Registry of all configured MCP data sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRegistry {
    pub sources: Vec<McpSource>,
}

impl SourceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Load a registry from a JSON value (typically from a config file or KV).
    pub fn load_from_json(value: serde_json::Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    /// Return all sources configured for a specific platform.
    pub fn sources_for_platform(&self, platform: &Platform) -> Vec<&McpSource> {
        self.sources
            .iter()
            .filter(|s| &s.platform == platform && s.active)
            .collect()
    }

    /// Return all active sources, sorted by priority (ascending = highest first).
    pub fn active_sources(&self) -> Vec<&McpSource> {
        let mut sources: Vec<&McpSource> = self.sources.iter().filter(|s| s.active).collect();
        sources.sort_by_key(|s| s.priority);
        sources
    }

    /// Add a source to the registry.
    pub fn register(&mut self, source: McpSource) {
        self.sources.push(source);
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Post Normalization ──────────────────────────────────────

/// Normalize a raw JSON post from an MCP tool response into a [`NormalizedPost`].
///
/// Attempts to extract common fields (`text`, `content`, `created_at`, etc.)
/// from the raw JSON and falls back to sensible defaults. The `source` parameter
/// provides platform context for field mapping.
pub fn normalize_post(raw: &serde_json::Value, source: &McpSource) -> Result<NormalizedPost> {
    let content = raw
        .get("text")
        .or_else(|| raw.get("content"))
        .or_else(|| raw.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let platform_post_id = raw
        .get("id")
        .or_else(|| raw.get("post_id"))
        .or_else(|| raw.get("message_id"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .unwrap_or_default();

    let author_id = raw
        .get("author_id")
        .or_else(|| raw.get("user_id"))
        .or_else(|| raw.get("from_id"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .unwrap_or_default();

    let timestamp = raw
        .get("created_at")
        .or_else(|| raw.get("timestamp"))
        .or_else(|| raw.get("date"))
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let language = {
        let lang_hint = raw
            .get("lang")
            .or_else(|| raw.get("language"))
            .and_then(|v| v.as_str());
        match lang_hint {
            Some(l) if !l.is_empty() => l.to_string(),
            _ => {
                let detected = super::multilingual::detect_language(&content);
                detected.language
            }
        }
    };

    let hashtags = extract_string_array(raw, "hashtags");
    let mentions = extract_string_array(raw, "mentions");
    let urls = extract_string_array(raw, "urls");

    let reply_to = raw
        .get("reply_to")
        .or_else(|| raw.get("in_reply_to_id"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        });

    let repost_of = raw
        .get("repost_of")
        .or_else(|| raw.get("retweeted_id"))
        .and_then(|v| {
            v.as_str()
                .map(|s| s.to_string())
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        });

    let engagement = raw
        .get("engagement")
        .and_then(|e| serde_json::from_value::<EngagementMetrics>(e.clone()).ok());

    Ok(NormalizedPost {
        id: Uuid::now_v7(),
        platform: source.platform.clone(),
        platform_post_id,
        author_id,
        content,
        language,
        timestamp,
        engagement,
        hashtags,
        mentions,
        urls,
        reply_to,
        repost_of,
        raw_json: raw.clone(),
        ingestion_source: source.name.clone(),
        confidence: 0.8, // default — downstream enrichment may adjust
    })
}

/// Extract a `Vec<String>` from a JSON array field, tolerating missing/non-array.
fn extract_string_array(obj: &serde_json::Value, field: &str) -> Vec<String> {
    obj.get(field)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

// ─── NormalizedPost → Hypergraph Conversion ─────────────────

/// Convert a [`NormalizedPost`] into an Entity + Situation pair suitable for
/// the hypergraph. The entity represents the post's author (Actor); the
/// situation represents the post content as an Event-level occurrence.
///
/// The caller is responsible for entity-resolution deduplication (the same
/// `author_id` appearing in multiple posts should be merged after insertion).
pub fn post_to_hypergraph_items(
    post: &NormalizedPost,
    narrative_id: Option<&str>,
) -> (crate::types::Entity, crate::types::Situation) {
    let now = Utc::now();

    // ── Author entity (Actor) ────────────────────────────────
    let mut props = serde_json::json!({
        "name": &post.author_id,
        "platform": post.platform.as_index_str(),
        "platform_user_id": &post.author_id,
        "ingestion_source": &post.ingestion_source,
    });
    if !post.language.is_empty() {
        props["language"] = serde_json::json!(&post.language);
    }

    let entity = crate::types::Entity {
        id: Uuid::now_v7(),
        entity_type: crate::types::EntityType::Actor,
        properties: props,
        beliefs: None,
        embedding: None,
        maturity: crate::types::MaturityLevel::Candidate,
        confidence: (post.confidence as f32).clamp(0.0, 1.0),
        confidence_breakdown: None,
        provenance: vec![crate::types::SourceReference {
            source_type: "mcp_ingest".to_string(),
            source_id: Some(post.ingestion_source.clone()),
            description: Some(format!(
                "auto-created from {} post {}",
                post.platform.as_index_str(),
                post.platform_post_id
            )),
            timestamp: post.timestamp,
            registered_source: None,
        }],
        extraction_method: Some(crate::types::ExtractionMethod::StructuredImport),
        narrative_id: narrative_id.map(String::from),
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    };

    // ── Post situation (Event) ───────────────────────────────
    let mut content_parts: Vec<String> = Vec::new();
    if !post.content.is_empty() {
        content_parts.push(post.content.clone());
    }
    if !post.hashtags.is_empty() {
        content_parts.push(format!("[hashtags: {}]", post.hashtags.join(", ")));
    }

    let situation = crate::types::Situation {
        id: post.id, // reuse the NormalizedPost id for traceability
        properties: serde_json::Value::Null,
        name: Some(format!(
            "{} post by {}",
            post.platform.as_index_str(),
            post.author_id
        )),
        description: if post.content.len() > 120 {
            Some(format!("{}...", &post.content[..117]))
        } else if !post.content.is_empty() {
            Some(post.content.clone())
        } else {
            None
        },
        temporal: crate::types::AllenInterval {
            start: Some(post.timestamp),
            end: Some(post.timestamp),
            granularity: crate::types::TimeGranularity::Exact,
            relations: vec![],
            fuzzy_endpoints: None,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![crate::types::ContentBlock::text(&content_parts.join("\n"))],
        narrative_level: crate::types::NarrativeLevel::Event,
        discourse: None,
        maturity: crate::types::MaturityLevel::Candidate,
        confidence: (post.confidence as f32).clamp(0.0, 1.0),
        confidence_breakdown: None,
        extraction_method: crate::types::ExtractionMethod::StructuredImport,
        provenance: vec![crate::types::SourceReference {
            source_type: "mcp_ingest".to_string(),
            source_id: Some(post.platform_post_id.clone()),
            description: Some(format!(
                "{} post id={}",
                post.platform.as_index_str(),
                post.platform_post_id
            )),
            timestamp: post.timestamp,
            registered_source: None,
        }],
        narrative_id: narrative_id.map(String::from),
        source_chunk_id: None,
        source_span: None,
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

    (entity, situation)
}

// ─── Audit Log ───────────────────────────────────────────────

/// Audit trail entry for an MCP tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// MCP source name that was invoked.
    pub source_name: String,
    /// Specific MCP tool that was called.
    pub tool: String,
    /// When the call was made.
    pub timestamp: DateTime<Utc>,
    /// Size of the response payload in bytes.
    pub response_size: usize,
    /// Whether the call succeeded.
    pub success: bool,
}

/// Persist an audit entry to KV at `mcp/audit/{timestamp_be}`.
pub fn store_audit_entry(hypergraph: &Hypergraph, entry: &AuditEntry) -> Result<()> {
    let ts_bytes = entry.timestamp.timestamp_millis().to_be_bytes();
    let mut key = crate::hypergraph::keys::MCP_AUDIT.to_vec();
    key.extend_from_slice(&ts_bytes);
    let value = serde_json::to_vec(entry).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(&key, &value)
}

/// Load recent audit entries (most recent first, up to `limit`).
pub fn list_audit_entries(hypergraph: &Hypergraph, limit: usize) -> Result<Vec<AuditEntry>> {
    let pairs = hypergraph
        .store()
        .prefix_scan(crate::hypergraph::keys::MCP_AUDIT)?;
    let mut entries: Vec<AuditEntry> = pairs
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
        .collect();
    // Sort by timestamp descending (most recent first).
    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    entries.truncate(limit);
    Ok(entries)
}

// ─── MCP Source Health (Sprint D8.6) ───────────────────────

/// Health status for an MCP source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceHealth {
    pub source_name: String,
    pub healthy: bool,
    pub consecutive_failures: u32,
    pub last_success: Option<DateTime<Utc>>,
    pub last_failure: Option<DateTime<Utc>>,
    pub posts_ingested_24h: usize,
    pub paused: bool,
}

/// Persist source health at `mcp/health/{source_name}`.
pub fn store_source_health(hypergraph: &Hypergraph, health: &SourceHealth) -> Result<()> {
    let key = format!("mcp/health/{}", health.source_name);
    let value = serde_json::to_vec(health).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

/// Load source health.
pub fn load_source_health(hypergraph: &Hypergraph, name: &str) -> Result<Option<SourceHealth>> {
    let key = format!("mcp/health/{}", name);
    match hypergraph
        .store()
        .get(key.as_bytes())
        .map_err(|e| TensaError::Internal(e.to_string()))?
    {
        Some(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).map_err(|e| TensaError::Serialization(e.to_string()))?,
        )),
        None => Ok(None),
    }
}

/// List all source health records.
pub fn list_source_health(hypergraph: &Hypergraph) -> Result<Vec<SourceHealth>> {
    let prefix = b"mcp/health/";
    let pairs = hypergraph
        .store()
        .prefix_scan(prefix)
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    Ok(pairs
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
        .collect())
}

/// Record a poll failure. After 3 consecutive failures, mark as unhealthy.
pub fn record_poll_failure(hypergraph: &Hypergraph, source_name: &str) -> Result<SourceHealth> {
    let mut health = load_source_health(hypergraph, source_name)?.unwrap_or_else(|| SourceHealth {
        source_name: source_name.to_string(),
        healthy: true,
        consecutive_failures: 0,
        last_success: None,
        last_failure: None,
        posts_ingested_24h: 0,
        paused: false,
    });
    health.consecutive_failures += 1;
    health.last_failure = Some(Utc::now());
    if health.consecutive_failures >= 3 {
        health.healthy = false;
    }
    store_source_health(hypergraph, &health)?;
    Ok(health)
}

/// Record a successful poll. Resets failure count and restores health.
pub fn record_poll_success(
    hypergraph: &Hypergraph,
    source_name: &str,
    posts_ingested: usize,
) -> Result<SourceHealth> {
    let mut health = load_source_health(hypergraph, source_name)?.unwrap_or_else(|| SourceHealth {
        source_name: source_name.to_string(),
        healthy: true,
        consecutive_failures: 0,
        last_success: None,
        last_failure: None,
        posts_ingested_24h: 0,
        paused: false,
    });
    health.consecutive_failures = 0;
    health.healthy = true;
    let now = Utc::now();
    health.last_success = Some(now);
    // Reset 24h counter if last success was >24h ago (or first success)
    let should_reset = health
        .last_success
        .map(|ls| (now - ls).num_hours() >= 24)
        .unwrap_or(true);
    if should_reset {
        health.posts_ingested_24h = posts_ingested;
    } else {
        health.posts_ingested_24h += posts_ingested;
    }
    store_source_health(hypergraph, &health)?;
    Ok(health)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn make_source() -> McpSource {
        McpSource {
            name: "test-twitter".to_string(),
            platform: Platform::Twitter,
            tools_used: vec!["search_tweets".to_string()],
            rate_limit_rpm: 60,
            priority: 1,
            active: true,
        }
    }

    #[test]
    fn test_source_registry_empty() {
        let reg = SourceRegistry::new();
        assert!(reg.sources.is_empty());
        assert!(reg.active_sources().is_empty());
    }

    #[test]
    fn test_source_registry_load_json() {
        let json = serde_json::json!({
            "sources": [
                {
                    "name": "tw-search",
                    "platform": "Twitter",
                    "tools_used": ["search"],
                    "rate_limit_rpm": 30,
                    "priority": 0,
                    "active": true
                }
            ]
        });
        let reg = SourceRegistry::load_from_json(json).unwrap();
        assert_eq!(reg.sources.len(), 1);
        assert_eq!(reg.sources[0].name, "tw-search");
    }

    #[test]
    fn test_sources_for_platform() {
        let mut reg = SourceRegistry::new();
        reg.register(make_source());
        reg.register(McpSource {
            name: "telegram-mon".to_string(),
            platform: Platform::Telegram,
            tools_used: vec![],
            rate_limit_rpm: 20,
            priority: 2,
            active: true,
        });
        let tw = reg.sources_for_platform(&Platform::Twitter);
        assert_eq!(tw.len(), 1);
        assert_eq!(tw[0].name, "test-twitter");
    }

    #[test]
    fn test_normalize_post_basic() {
        let source = make_source();
        let raw = serde_json::json!({
            "id": "123456",
            "text": "Breaking news about disinformation #disinfo",
            "author_id": "user_789",
            "created_at": "2026-04-17T12:00:00Z",
            "hashtags": ["disinfo"],
            "lang": "en"
        });
        let post = normalize_post(&raw, &source).unwrap();
        assert_eq!(post.platform_post_id, "123456");
        assert_eq!(post.content, "Breaking news about disinformation #disinfo");
        assert_eq!(post.author_id, "user_789");
        assert_eq!(post.language, "en");
        assert_eq!(post.hashtags, vec!["disinfo"]);
        assert_eq!(post.ingestion_source, "test-twitter");
    }

    #[test]
    fn test_normalize_post_missing_fields() {
        let source = make_source();
        let raw = serde_json::json!({});
        let post = normalize_post(&raw, &source).unwrap();
        assert!(post.content.is_empty());
        assert!(post.platform_post_id.is_empty());
        assert!(post.hashtags.is_empty());
    }

    #[test]
    fn test_normalize_post_auto_detects_language() {
        let source = make_source();
        let raw = serde_json::json!({
            "text": "Привет мир, это сообщение на русском языке"
        });
        let post = normalize_post(&raw, &source).unwrap();
        assert_eq!(post.language, "ru");
    }

    #[test]
    fn test_audit_entry_round_trip() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let entry = AuditEntry {
            source_name: "test-source".to_string(),
            tool: "search_tweets".to_string(),
            timestamp: Utc::now(),
            response_size: 4096,
            success: true,
        };
        store_audit_entry(&hg, &entry).unwrap();

        let entries = list_audit_entries(&hg, 10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source_name, "test-source");
        assert_eq!(entries[0].response_size, 4096);
        assert!(entries[0].success);
    }

    #[test]
    fn test_active_sources_sorted_by_priority() {
        let mut reg = SourceRegistry::new();
        reg.register(McpSource {
            name: "low-priority".to_string(),
            platform: Platform::Web,
            tools_used: vec![],
            rate_limit_rpm: 10,
            priority: 5,
            active: true,
        });
        reg.register(McpSource {
            name: "high-priority".to_string(),
            platform: Platform::Twitter,
            tools_used: vec![],
            rate_limit_rpm: 60,
            priority: 1,
            active: true,
        });
        reg.register(McpSource {
            name: "inactive".to_string(),
            platform: Platform::Telegram,
            tools_used: vec![],
            rate_limit_rpm: 30,
            priority: 0,
            active: false,
        });
        let active = reg.active_sources();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].name, "high-priority");
        assert_eq!(active[1].name, "low-priority");
    }

    #[test]
    fn test_post_to_hypergraph_items() {
        let source = make_source();
        let raw = serde_json::json!({
            "id": "post-001",
            "text": "Disinformation campaign detected",
            "author_id": "user_42",
            "created_at": "2026-04-17T12:00:00Z",
            "hashtags": ["disinfo", "osint"],
            "lang": "en"
        });
        let post = normalize_post(&raw, &source).unwrap();
        let (entity, situation) = post_to_hypergraph_items(&post, Some("narr-d7"));

        // Entity checks
        assert_eq!(entity.entity_type, crate::types::EntityType::Actor);
        assert_eq!(entity.properties["name"], "user_42");
        assert_eq!(entity.properties["platform"], "twitter");
        assert_eq!(entity.narrative_id.as_deref(), Some("narr-d7"));
        assert!(entity.confidence > 0.0 && entity.confidence <= 1.0);
        assert_eq!(
            entity.extraction_method,
            Some(crate::types::ExtractionMethod::StructuredImport)
        );

        // Situation checks
        assert_eq!(situation.id, post.id); // traceability
        assert!(situation.name.as_ref().unwrap().contains("twitter"));
        assert_eq!(situation.narrative_id.as_deref(), Some("narr-d7"));
        assert_eq!(
            situation.narrative_level,
            crate::types::NarrativeLevel::Event
        );
        assert!(situation.temporal.start.is_some());
        assert!(!situation.raw_content.is_empty());
        let text = &situation.raw_content[0].content;
        assert!(text.contains("Disinformation campaign detected"));
        assert!(text.contains("disinfo"));
    }

    #[test]
    fn test_health_degradation_and_recovery() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        record_poll_failure(&hg, "test-source").unwrap();
        record_poll_failure(&hg, "test-source").unwrap();
        let h = record_poll_failure(&hg, "test-source").unwrap();
        assert!(!h.healthy);
        assert_eq!(h.consecutive_failures, 3);
        let h2 = record_poll_success(&hg, "test-source", 5).unwrap();
        assert!(h2.healthy);
        assert_eq!(h2.consecutive_failures, 0);
    }

    #[test]
    fn test_list_source_health_empty() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let health = list_source_health(&hg).unwrap();
        assert!(health.is_empty());
    }

    #[test]
    fn test_list_source_health_after_records() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        record_poll_success(&hg, "src-a", 10).unwrap();
        record_poll_failure(&hg, "src-b").unwrap();
        let health = list_source_health(&hg).unwrap();
        assert_eq!(health.len(), 2);
    }

    #[test]
    fn test_post_to_hypergraph_no_narrative() {
        let source = make_source();
        let raw = serde_json::json!({"text": "hello"});
        let post = normalize_post(&raw, &source).unwrap();
        let (entity, situation) = post_to_hypergraph_items(&post, None);
        assert!(entity.narrative_id.is_none());
        assert!(situation.narrative_id.is_none());
    }
}
