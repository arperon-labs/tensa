//! Source discovery engine (Sprint D8.3).
//!
//! Automatically discovers new sources by following links, co-amplification
//! patterns, and narrative trails. Candidates are stored at `disc/{uuid}`
//! and promoted to approved sources at `disc/approved/{uuid}` based on the
//! configured [`DiscoveryPolicy`].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::types::Platform;
use crate::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// How a source was discovered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryType {
    /// Telegram channel forward or link in a post.
    ChannelForward,
    /// URL shared in a monitored post.
    LinkShare,
    /// Repost/retweet of unknown content.
    Repost,
    /// Mentioned account/channel.
    Mention,
    /// Found in a CIB cluster.
    CoAmplification,
}

/// Policy for auto-approving discovered sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryPolicy {
    /// All candidates need analyst approval.
    Manual,
    /// Auto-add CIB-detected sources, manual for rest.
    AutoApproveCoAmplification,
    /// Add everything, analyst reviews retroactively.
    AutoApproveAll,
}

impl Default for DiscoveryPolicy {
    fn default() -> Self {
        Self::Manual
    }
}

/// A discovered source candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryCandidate {
    pub id: Uuid,
    pub platform: Platform,
    /// Platform-specific source identifier (channel name, account handle, etc.)
    pub source_id: String,
    /// Post or cluster that led to discovery.
    pub discovered_via: String,
    pub discovery_type: DiscoveryType,
    /// How many times this source has been seen.
    pub times_seen: usize,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    /// Whether automatically approved by policy.
    pub auto_approved: bool,
    /// Whether manually approved by analyst.
    pub approved: bool,
    /// Whether rejected by analyst.
    pub rejected: bool,
}

// ─── KV Prefixes ────────────────────────────────────────────

/// KV prefix for discovery candidates.
const DISC_PREFIX: &str = "disc/";
/// KV prefix for approved sources.
const DISC_APPROVED_PREFIX: &str = "disc/approved/";
/// KV key for discovery policy.
const DISC_POLICY_KEY: &str = "disc/policy";

// ─── Public API ─────────────────────────────────────────────

/// Create or update a discovery candidate. If the source was already seen,
/// increment `times_seen` and update `last_seen`.
pub fn report_discovery(
    hypergraph: &Hypergraph,
    platform: Platform,
    source_id: &str,
    discovered_via: &str,
    discovery_type: DiscoveryType,
) -> Result<DiscoveryCandidate> {
    let policy = load_policy(hypergraph)?;

    // Check if candidate already exists.
    if let Some(mut existing) = find_candidate(hypergraph, &platform, source_id)? {
        existing.times_seen += 1;
        existing.last_seen = Utc::now();
        store_candidate(hypergraph, &existing)?;
        return Ok(existing);
    }

    let auto_approved = match (&policy, &discovery_type) {
        (DiscoveryPolicy::AutoApproveAll, _) => true,
        (DiscoveryPolicy::AutoApproveCoAmplification, DiscoveryType::CoAmplification) => true,
        _ => false,
    };

    let candidate = DiscoveryCandidate {
        id: Uuid::now_v7(),
        platform,
        source_id: source_id.to_string(),
        discovered_via: discovered_via.to_string(),
        discovery_type,
        times_seen: 1,
        first_seen: Utc::now(),
        last_seen: Utc::now(),
        auto_approved,
        approved: auto_approved,
        rejected: false,
    };

    store_candidate(hypergraph, &candidate)?;

    if auto_approved {
        store_approved(hypergraph, &candidate)?;
    }

    Ok(candidate)
}

/// Discover sources from URLs in a post's content.
pub fn discover_from_urls(
    hypergraph: &Hypergraph,
    post_id: &str,
    urls: &[String],
    platform: &Platform,
) -> Result<Vec<DiscoveryCandidate>> {
    let mut candidates = Vec::new();
    for url in urls {
        let candidate = report_discovery(
            hypergraph,
            platform.clone(),
            url,
            post_id,
            DiscoveryType::LinkShare,
        )?;
        candidates.push(candidate);
    }
    Ok(candidates)
}

/// Discover sources from CIB cluster members.
pub fn discover_from_cib_cluster(
    hypergraph: &Hypergraph,
    cluster_id: &str,
    member_ids: &[String],
) -> Result<Vec<DiscoveryCandidate>> {
    let mut candidates = Vec::new();
    for member in member_ids {
        let candidate = report_discovery(
            hypergraph,
            Platform::Other("unknown".to_string()),
            member,
            cluster_id,
            DiscoveryType::CoAmplification,
        )?;
        candidates.push(candidate);
    }
    Ok(candidates)
}

/// Approve a discovery candidate.
pub fn approve_candidate(hypergraph: &Hypergraph, candidate_id: &Uuid) -> Result<()> {
    let mut candidate = load_candidate(hypergraph, candidate_id)?
        .ok_or_else(|| TensaError::NotFound(format!("Candidate {} not found", candidate_id)))?;
    candidate.approved = true;
    candidate.rejected = false;
    store_candidate(hypergraph, &candidate)?;
    store_approved(hypergraph, &candidate)?;
    Ok(())
}

/// Reject a discovery candidate.
pub fn reject_candidate(hypergraph: &Hypergraph, candidate_id: &Uuid) -> Result<()> {
    let mut candidate = load_candidate(hypergraph, candidate_id)?
        .ok_or_else(|| TensaError::NotFound(format!("Candidate {} not found", candidate_id)))?;
    candidate.rejected = true;
    candidate.approved = false;
    store_candidate(hypergraph, &candidate)?;
    Ok(())
}

/// List all discovery candidates (excludes approved-index and policy entries).
pub fn list_candidates(hypergraph: &Hypergraph) -> Result<Vec<DiscoveryCandidate>> {
    let pairs = hypergraph.store().prefix_scan(DISC_PREFIX.as_bytes())?;
    let mut candidates = Vec::new();
    for (key, value) in pairs {
        let key_str = String::from_utf8_lossy(&key);
        if key_str.starts_with(DISC_APPROVED_PREFIX) || key_str == DISC_POLICY_KEY {
            continue;
        }
        if let Ok(c) = serde_json::from_slice::<DiscoveryCandidate>(&value) {
            candidates.push(c);
        }
    }
    Ok(candidates)
}

/// Load the current discovery policy (defaults to [`DiscoveryPolicy::Manual`]).
pub fn load_policy(hypergraph: &Hypergraph) -> Result<DiscoveryPolicy> {
    match hypergraph.store().get(DISC_POLICY_KEY.as_bytes())? {
        Some(bytes) => {
            serde_json::from_slice(&bytes).map_err(|e| TensaError::Serialization(e.to_string()))
        }
        None => Ok(DiscoveryPolicy::default()),
    }
}

/// Persist a discovery policy.
pub fn save_policy(hypergraph: &Hypergraph, policy: &DiscoveryPolicy) -> Result<()> {
    let value = serde_json::to_vec(policy).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(DISC_POLICY_KEY.as_bytes(), &value)
}

// ─── Internal Helpers ───────────────────────────────────────

fn store_candidate(hypergraph: &Hypergraph, c: &DiscoveryCandidate) -> Result<()> {
    let key = format!("{}{}", DISC_PREFIX, c.id);
    let value = serde_json::to_vec(c).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(key.as_bytes(), &value)
}

fn load_candidate(hypergraph: &Hypergraph, id: &Uuid) -> Result<Option<DiscoveryCandidate>> {
    let key = format!("{}{}", DISC_PREFIX, id);
    match hypergraph.store().get(key.as_bytes())? {
        Some(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).map_err(|e| TensaError::Serialization(e.to_string()))?,
        )),
        None => Ok(None),
    }
}

fn find_candidate(
    hypergraph: &Hypergraph,
    platform: &Platform,
    source_id: &str,
) -> Result<Option<DiscoveryCandidate>> {
    let candidates = list_candidates(hypergraph)?;
    Ok(candidates
        .into_iter()
        .find(|c| c.platform == *platform && c.source_id == source_id))
}

fn store_approved(hypergraph: &Hypergraph, c: &DiscoveryCandidate) -> Result<()> {
    let key = format!("{}{}", DISC_APPROVED_PREFIX, c.id);
    let value = serde_json::to_vec(c).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph.store().put(key.as_bytes(), &value)
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
    fn test_report_and_list_discovery() {
        let hg = setup();
        let c = report_discovery(
            &hg,
            Platform::Twitter,
            "unknown_account",
            "post-1",
            DiscoveryType::LinkShare,
        )
        .unwrap();
        assert!(!c.approved);
        assert_eq!(c.times_seen, 1);
        let all = list_candidates(&hg).unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_duplicate_increments_times_seen() {
        let hg = setup();
        report_discovery(
            &hg,
            Platform::Telegram,
            "channel_x",
            "post-1",
            DiscoveryType::ChannelForward,
        )
        .unwrap();
        let c2 = report_discovery(
            &hg,
            Platform::Telegram,
            "channel_x",
            "post-2",
            DiscoveryType::ChannelForward,
        )
        .unwrap();
        assert_eq!(c2.times_seen, 2);
    }

    #[test]
    fn test_approve_and_reject() {
        let hg = setup();
        let c = report_discovery(
            &hg,
            Platform::Twitter,
            "acct1",
            "post-1",
            DiscoveryType::Mention,
        )
        .unwrap();
        approve_candidate(&hg, &c.id).unwrap();
        let loaded = load_candidate(&hg, &c.id).unwrap().unwrap();
        assert!(loaded.approved);

        reject_candidate(&hg, &c.id).unwrap();
        let loaded2 = load_candidate(&hg, &c.id).unwrap().unwrap();
        assert!(loaded2.rejected);
        assert!(!loaded2.approved);
    }

    #[test]
    fn test_auto_approve_co_amplification() {
        let hg = setup();
        save_policy(&hg, &DiscoveryPolicy::AutoApproveCoAmplification).unwrap();
        let c = report_discovery(
            &hg,
            Platform::Telegram,
            "bot_acct",
            "cluster-1",
            DiscoveryType::CoAmplification,
        )
        .unwrap();
        assert!(c.auto_approved);
        assert!(c.approved);
    }

    #[test]
    fn test_manual_policy_no_auto_approve() {
        let hg = setup();
        save_policy(&hg, &DiscoveryPolicy::Manual).unwrap();
        let c = report_discovery(
            &hg,
            Platform::Twitter,
            "acct2",
            "post-5",
            DiscoveryType::LinkShare,
        )
        .unwrap();
        assert!(!c.auto_approved);
        assert!(!c.approved);
    }

    #[test]
    fn test_discover_from_urls() {
        let hg = setup();
        let urls = vec![
            "https://t.me/unknownchannel".to_string(),
            "https://twitter.com/bot123".to_string(),
        ];
        let candidates = discover_from_urls(&hg, "post-1", &urls, &Platform::Twitter).unwrap();
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_discover_from_cib_cluster() {
        let hg = setup();
        let members = vec!["member-a".to_string(), "member-b".to_string()];
        let candidates = discover_from_cib_cluster(&hg, "cluster-42", &members).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].discovery_type, DiscoveryType::CoAmplification);
    }

    #[test]
    fn test_auto_approve_all_policy() {
        let hg = setup();
        save_policy(&hg, &DiscoveryPolicy::AutoApproveAll).unwrap();
        let c = report_discovery(
            &hg,
            Platform::Twitter,
            "random_acct",
            "post-10",
            DiscoveryType::Mention,
        )
        .unwrap();
        assert!(c.auto_approved);
        assert!(c.approved);
    }

    #[test]
    fn test_default_policy_is_manual() {
        let hg = setup();
        let policy = load_policy(&hg).unwrap();
        assert_eq!(policy, DiscoveryPolicy::Manual);
    }
}
