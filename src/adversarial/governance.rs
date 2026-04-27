//! Ethics governance layer — access control + audit logging.
//!
//! Enforces tier-based access control on all adversarial endpoints and
//! logs every action for accountability. Designed for EU AI Act Article 5
//! and Regulation 2021/821 Article 5 compliance.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

// ─── Access Tiers ────────────────────────────────────────────

/// Access tier for adversarial capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AccessTier {
    /// Retrodiction, analysis, academic use.
    Research = 0,
    /// Wargame sessions, counter-narrative generation.
    Training = 1,
    /// Real-time ingestion, intervention optimization.
    Operational = 2,
    /// Integration with national CERT systems (future).
    Classified = 3,
}

impl AccessTier {
    /// Minimum tier required for each operation category.
    pub fn required_for(operation: &str) -> Self {
        match operation {
            "retrodiction"
            | "reward_fingerprint"
            | "export_stix"
            | "import_stix"
            | "configure_rationality" => Self::Research,
            "create_wargame" | "submit_move" | "auto_play" | "generate_policy"
            | "counter_narrative" | "get_wargame_state" => Self::Training,
            "optimize_intervention"
            | "stream_ingest"
            | "fast_counterfactual"
            | "calibrate_beta" => Self::Operational,
            _ => Self::Research,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Research => "Research",
            Self::Training => "Training",
            Self::Operational => "Operational",
            Self::Classified => "Classified",
        }
    }
}

/// Check whether a user's tier is sufficient for an operation.
pub fn check_access(user_tier: AccessTier, operation: &str) -> Result<()> {
    let required = AccessTier::required_for(operation);
    if user_tier >= required {
        Ok(())
    } else {
        Err(TensaError::Internal(format!(
            "Access denied: operation '{}' requires {} tier, user has {} tier",
            operation,
            required.label(),
            user_tier.label()
        )))
    }
}

// ─── Data Scope ──────────────────────────────────────────────

/// Scope of data being accessed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataScope {
    /// Training/retrodiction on historical data only.
    SyntheticOnly,
    /// Publicly available social media posts.
    PublicData,
    /// Requires Operational tier.
    IdentifiableActors,
}

// ─── Audit Logging ───────────────────────────────────────────

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID.
    pub id: String,
    /// User or system identity.
    pub user_id: String,
    /// Access tier at time of action.
    pub access_tier: AccessTier,
    /// What action was performed.
    pub action: String,
    /// When the action occurred.
    pub timestamp: DateTime<Utc>,
    /// Why the action was performed (user-provided justification).
    pub justification: Option<String>,
    /// Scope of data accessed.
    pub data_scope: DataScope,
    /// Additional context (e.g., narrative_id, session_id).
    pub context: serde_json::Value,
}

/// Log an adversarial action to the audit trail.
pub fn log_audit(
    hypergraph: &Hypergraph,
    user_id: &str,
    tier: AccessTier,
    action: &str,
    data_scope: DataScope,
    justification: Option<&str>,
    context: serde_json::Value,
) -> Result<String> {
    let entry = AuditEntry {
        id: Uuid::now_v7().to_string(),
        user_id: user_id.to_string(),
        access_tier: tier,
        action: action.to_string(),
        timestamp: Utc::now(),
        justification: justification.map(String::from),
        data_scope,
        context,
    };

    store_audit_entry(hypergraph, &entry)?;
    Ok(entry.id)
}

/// Query recent audit entries.
pub fn list_audit_entries(hypergraph: &Hypergraph, limit: usize) -> Result<Vec<AuditEntry>> {
    let pairs = hypergraph.store().prefix_scan(AUDIT_PREFIX)?;
    // UUID v7 keys are lexicographically time-sorted; reverse for most-recent-first.
    // Deserialize only up to `limit` entries to avoid loading the full audit log.
    let entries: Vec<AuditEntry> = pairs
        .into_iter()
        .rev()
        .take(limit)
        .map(|(_, v)| serde_json::from_slice(&v))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(entries)
}

// ─── Content Watermarking ────────────────────────────────────

/// Marker appended to LLM-generated counter-narratives.
pub const WATERMARK_MARKER: &str = "[TENSA-GENERATED]";

/// Apply watermark metadata to generated content.
pub fn watermark_content(content: &str) -> String {
    format!(
        "{}\n\n<!-- {} Generated by TENSA counter-narrative engine. \
         C2PA provenance: machine-generated content per EU AI Act Article 50. -->",
        content, WATERMARK_MARKER
    )
}

/// Check if content contains the TENSA watermark.
pub fn has_watermark(content: &str) -> bool {
    content.contains(WATERMARK_MARKER)
}

// ─── KV Storage ──────────────────────────────────────────────

const AUDIT_PREFIX: &[u8] = b"adv/audit/";

fn store_audit_entry(hypergraph: &Hypergraph, entry: &AuditEntry) -> Result<()> {
    let mut key = AUDIT_PREFIX.to_vec();
    key.extend_from_slice(entry.id.as_bytes());
    let value = serde_json::to_vec(entry)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

pub fn load_audit_entry(hypergraph: &Hypergraph, entry_id: &str) -> Result<Option<AuditEntry>> {
    let mut key = AUDIT_PREFIX.to_vec();
    key.extend_from_slice(entry_id.as_bytes());
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
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

    #[test]
    fn test_research_tier_can_access_retrodiction() {
        assert!(check_access(AccessTier::Research, "retrodiction").is_ok());
    }

    #[test]
    fn test_research_tier_cannot_access_operational() {
        assert!(check_access(AccessTier::Research, "optimize_intervention").is_err());
    }

    #[test]
    fn test_training_tier_can_access_wargame() {
        assert!(check_access(AccessTier::Training, "create_wargame").is_ok());
        assert!(check_access(AccessTier::Training, "counter_narrative").is_ok());
    }

    #[test]
    fn test_operational_tier_can_access_everything() {
        assert!(check_access(AccessTier::Operational, "retrodiction").is_ok());
        assert!(check_access(AccessTier::Operational, "create_wargame").is_ok());
        assert!(check_access(AccessTier::Operational, "optimize_intervention").is_ok());
    }

    #[test]
    fn test_audit_logging() {
        let hg = test_hg();
        let entry_id = log_audit(
            &hg,
            "analyst-001",
            AccessTier::Training,
            "create_wargame",
            DataScope::SyntheticOnly,
            Some("Testing defense strategy for Q2 campaign"),
            serde_json::json!({"narrative_id": "test-nar"}),
        )
        .unwrap();

        let loaded = load_audit_entry(&hg, &entry_id).unwrap();
        assert!(loaded.is_some());
        let entry = loaded.unwrap();
        assert_eq!(entry.user_id, "analyst-001");
        assert_eq!(entry.action, "create_wargame");
        assert_eq!(entry.access_tier, AccessTier::Training);
    }

    #[test]
    fn test_list_audit_entries() {
        let hg = test_hg();

        for i in 0..5 {
            log_audit(
                &hg,
                &format!("user-{}", i),
                AccessTier::Research,
                "retrodiction",
                DataScope::SyntheticOnly,
                None,
                serde_json::json!({}),
            )
            .unwrap();
        }

        let entries = list_audit_entries(&hg, 3).unwrap();
        assert_eq!(entries.len(), 3, "should respect limit");
    }

    #[test]
    fn test_watermark() {
        let original = "This is a counter-narrative about vaccines.";
        let watermarked = watermark_content(original);
        assert!(has_watermark(&watermarked));
        assert!(!has_watermark(original));
        assert!(watermarked.contains("C2PA provenance"));
    }

    #[test]
    fn test_access_tier_ordering() {
        assert!(AccessTier::Research < AccessTier::Training);
        assert!(AccessTier::Training < AccessTier::Operational);
        assert!(AccessTier::Operational < AccessTier::Classified);
    }
}
