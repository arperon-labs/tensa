//! Community summary generation for narrative entity clusters.
//!
//! Uses Louvain community assignments from centrality analysis to group
//! entities, then generates LLM-powered summaries for each community.
//! Summaries are cached in KV at `cs/{narrative_id}/{community_id}`.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::centrality::CentralityResult;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::llm::NarrativeExtractor;
use crate::store::KVStore;

/// Summary of a community (cluster of related entities).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunitySummary {
    /// Narrative this community belongs to.
    pub narrative_id: String,
    /// Community index from Leiden detection.
    pub community_id: usize,
    /// LLM-generated or fallback summary text.
    pub summary: String,
    /// UUIDs of entities in this community.
    pub entity_ids: Vec<Uuid>,
    /// Human-readable names of entities.
    pub entity_names: Vec<String>,
    /// Key themes extracted from the summary.
    pub key_themes: Vec<String>,
    /// Number of entities in this community.
    pub entity_count: usize,
    /// When the summary was generated.
    pub generated_at: DateTime<Utc>,
    /// Model used for generation (if LLM was used).
    pub model: Option<String>,
    /// Hierarchy level (0 = leaf/most granular, higher = coarser).
    #[serde(default)]
    pub level: usize,
    /// Parent community ID at the next coarser level (None for top-level).
    #[serde(default)]
    pub parent_community_id: Option<usize>,
    /// Child community IDs at the next finer level (empty for leaf).
    #[serde(default)]
    pub child_community_ids: Vec<usize>,
}

/// KV prefix for community summaries.
const COMMUNITY_PREFIX: &str = "cs/";

/// Store a community summary in KV.
pub fn store_summary(store: &dyn KVStore, summary: &CommunitySummary) -> Result<()> {
    let key = format!(
        "{}{}/{}",
        COMMUNITY_PREFIX, summary.narrative_id, summary.community_id
    );
    let value = serde_json::to_vec(summary)?;
    store.put(key.as_bytes(), &value)
}

/// Get a single community summary from KV.
pub fn get_summary(
    store: &dyn KVStore,
    narrative_id: &str,
    community_id: usize,
) -> Result<Option<CommunitySummary>> {
    let key = format!("{}{}/{}", COMMUNITY_PREFIX, narrative_id, community_id);
    match store.get(key.as_bytes())? {
        Some(data) => Ok(Some(serde_json::from_slice(&data)?)),
        None => Ok(None),
    }
}

/// List all community summaries for a narrative, sorted by community_id.
pub fn list_summaries(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<CommunitySummary>> {
    let prefix = format!("{}{}/", COMMUNITY_PREFIX, narrative_id);
    let entries = store.prefix_scan(prefix.as_bytes())?;
    let mut summaries = Vec::new();
    for (_, value) in entries {
        if let Ok(s) = serde_json::from_slice::<CommunitySummary>(&value) {
            summaries.push(s);
        }
    }
    summaries.sort_by_key(|s| s.community_id);
    Ok(summaries)
}

/// List community summaries at a specific hierarchy level.
pub fn list_summaries_at_level(
    store: &dyn KVStore,
    narrative_id: &str,
    level: usize,
) -> Result<Vec<CommunitySummary>> {
    let all = list_summaries(store, narrative_id)?;
    Ok(all.into_iter().filter(|s| s.level == level).collect())
}

/// Get the full community hierarchy for a narrative (all levels).
pub fn get_hierarchy(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<HashMap<usize, Vec<CommunitySummary>>> {
    let all = list_summaries(store, narrative_id)?;
    let mut by_level: HashMap<usize, Vec<CommunitySummary>> = HashMap::new();
    for s in all {
        by_level.entry(s.level).or_default().push(s);
    }
    Ok(by_level)
}

/// Generate community summaries for a narrative using LLM.
///
/// Requires centrality analysis to have been run first (to get community
/// assignments from Louvain detection). Steps:
///
/// 1. Load centrality results from KV (`an/c/{narrative_id}/`)
/// 2. Group entities by `community_id`
/// 3. For each community: load entity details, collect key properties
/// 4. Call LLM to generate a 2-3 sentence summary + key themes
/// 5. Cache result at `cs/{narrative_id}/{community_id}`
pub fn generate_summaries(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    extractor: &dyn NarrativeExtractor,
    store: &dyn KVStore,
) -> Result<Vec<CommunitySummary>> {
    // Step 1: Load centrality results
    let prefix = format!(
        "{}{}",
        std::str::from_utf8(crate::hypergraph::keys::ANALYSIS_CENTRALITY).unwrap_or("an/c/"),
        narrative_id
    );
    let scan_prefix = format!("{}/", prefix);
    let entries = store.prefix_scan(scan_prefix.as_bytes())?;

    if entries.is_empty() {
        return Err(TensaError::InferenceError(format!(
            "No centrality analysis found for narrative '{}'. Run INFER CENTRALITY first.",
            narrative_id
        )));
    }

    // Parse centrality results and group by community.
    // Cache entity properties alongside the name to avoid re-fetching in the prompt loop.
    let mut communities: HashMap<usize, Vec<(Uuid, String, Option<serde_json::Value>)>> =
        HashMap::new();

    for (_, value) in &entries {
        if let Ok(cr) = serde_json::from_slice::<CentralityResult>(value) {
            let (entity_name, entity_props) = match hypergraph.get_entity(&cr.entity_id) {
                Ok(e) => {
                    let name = e
                        .properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unnamed")
                        .to_string();
                    (name, Some(e.properties.clone()))
                }
                Err(_) => ("unknown".to_string(), None),
            };
            communities.entry(cr.community_id).or_default().push((
                cr.entity_id,
                entity_name,
                entity_props,
            ));
        }
    }

    // Step 2: For each community, generate summary
    let mut summaries = Vec::new();
    let model = extractor.model_name();

    // Collect and sort keys for deterministic ordering
    let mut community_ids: Vec<usize> = communities.keys().copied().collect();
    community_ids.sort();

    for community_id in community_ids {
        let members = match communities.get(&community_id) {
            Some(m) => m,
            None => continue,
        };

        let entity_ids: Vec<Uuid> = members.iter().map(|(id, _, _)| *id).collect();
        let entity_names: Vec<String> = members.iter().map(|(_, name, _)| name.clone()).collect();

        // Build context for LLM (cap at 20 entities for prompt length)
        let mut context = format!(
            "Community {} contains {} entities:\n",
            community_id,
            members.len()
        );
        for (id, name, props) in members.iter().take(20) {
            context.push_str(&format!("- {} ({})\n", name, id));
            // Add entity properties from cached data
            if let Some(props_val) = props {
                if let Some(obj) = props_val.as_object() {
                    for (k, v) in obj.iter().take(5) {
                        if k != "name" {
                            context.push_str(&format!("  {}: {}\n", k, v));
                        }
                    }
                }
            }
        }

        let system =
            "You are a narrative analyst. Given a community of entities from a narrative, \
            write a concise 2-3 sentence summary describing:\n\
            1. Who/what this group represents\n\
            2. Key themes or roles\n\
            3. Their significance in the narrative\n\
            Also extract 3-5 key themes as a JSON array.\n\
            Format: {\"summary\": \"...\", \"themes\": [\"theme1\", \"theme2\", ...]}";

        let (summary_text, themes) = match extractor.answer_question(system, &context) {
            Ok(response) => parse_summary_response(&response),
            Err(_) => {
                // Fallback: simple description
                let desc = format!(
                    "Group of {} entities including {}.",
                    members.len(),
                    entity_names
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                (desc, vec![])
            }
        };

        let summary = CommunitySummary {
            narrative_id: narrative_id.to_string(),
            community_id,
            summary: summary_text,
            entity_ids,
            entity_names,
            key_themes: themes,
            entity_count: members.len(),
            generated_at: Utc::now(),
            model: model.clone(),
            level: 0,
            parent_community_id: None,
            child_community_ids: vec![],
        };

        store_summary(store, &summary)?;
        summaries.push(summary);
    }

    summaries.sort_by_key(|s| s.community_id);
    Ok(summaries)
}

/// Parse an LLM response into summary text and themes.
fn parse_summary_response(response: &str) -> (String, Vec<String>) {
    #[derive(Deserialize)]
    struct SummaryResponse {
        summary: String,
        #[serde(default)]
        themes: Vec<String>,
    }

    let cleaned = crate::ingestion::extraction::extract_json_from_response(response);
    if let Ok(parsed) = serde_json::from_str::<SummaryResponse>(&cleaned) {
        return (parsed.summary, parsed.themes);
    }
    // Fallback: use raw response as summary
    (response.trim().to_string(), vec![])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::centrality::run_centrality;
    use crate::analysis::test_helpers::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_store_get_summary() {
        let store = MemoryStore::new();
        let summary = CommunitySummary {
            narrative_id: "test-nar".to_string(),
            community_id: 0,
            summary: "A group of protagonists.".to_string(),
            entity_ids: vec![Uuid::nil()],
            entity_names: vec!["Alice".to_string()],
            key_themes: vec!["leadership".to_string()],
            entity_count: 1,
            generated_at: Utc::now(),
            model: Some("test-model".to_string()),
            level: 0,
            parent_community_id: None,
            child_community_ids: vec![],
        };

        store_summary(&store, &summary).unwrap();
        let loaded = get_summary(&store, "test-nar", 0).unwrap().unwrap();
        assert_eq!(loaded.narrative_id, "test-nar");
        assert_eq!(loaded.community_id, 0);
        assert_eq!(loaded.summary, "A group of protagonists.");
        assert_eq!(loaded.entity_count, 1);
    }

    #[test]
    fn test_list_summaries_sorted() {
        let store = MemoryStore::new();
        for i in [2, 0, 1] {
            let summary = CommunitySummary {
                narrative_id: "nar-a".to_string(),
                community_id: i,
                summary: format!("Community {}", i),
                entity_ids: vec![],
                entity_names: vec![],
                key_themes: vec![],
                entity_count: 0,
                generated_at: Utc::now(),
                model: None,
                level: 0,
                parent_community_id: None,
                child_community_ids: vec![],
            };
            store_summary(&store, &summary).unwrap();
        }

        let summaries = list_summaries(&store, "nar-a").unwrap();
        assert_eq!(summaries.len(), 3);
        assert_eq!(summaries[0].community_id, 0);
        assert_eq!(summaries[1].community_id, 1);
        assert_eq!(summaries[2].community_id, 2);
    }

    #[test]
    fn test_list_summaries_empty_narrative() {
        let store = MemoryStore::new();
        let summaries = list_summaries(&store, "nonexistent").unwrap();
        assert!(summaries.is_empty());
    }

    #[test]
    fn test_parse_summary_json() {
        let json = r#"{"summary": "A tightly knit group.", "themes": ["unity", "power"]}"#;
        let (summary, themes) = parse_summary_response(json);
        assert_eq!(summary, "A tightly knit group.");
        assert_eq!(themes, vec!["unity", "power"]);
    }

    #[test]
    fn test_parse_summary_raw() {
        let raw = "This is just some raw text without JSON.";
        let (summary, themes) = parse_summary_response(raw);
        assert_eq!(summary, "This is just some raw text without JSON.");
        assert!(themes.is_empty());
    }

    #[test]
    fn test_parse_summary_embedded_json() {
        let text = r#"Here is the analysis: {"summary": "Leaders group.", "themes": ["leadership"]} Done."#;
        let (summary, themes) = parse_summary_response(text);
        assert_eq!(summary, "Leaders group.");
        assert_eq!(themes, vec!["leadership"]);
    }

    #[test]
    fn test_generate_no_centrality() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let mock = crate::ingestion::llm::MockExtractor;

        let result = generate_summaries("nonexistent", &hg, &mock, store.as_ref());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No centrality analysis found"));
    }

    #[test]
    fn test_generate_summaries_with_centrality() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());

        // Create entities and link them
        let e1 = add_entity(&hg, "Alice", "test-gen");
        let e2 = add_entity(&hg, "Bob", "test-gen");
        let s1 = add_situation(&hg, "test-gen");
        link(&hg, e1, s1);
        link(&hg, e2, s1);

        // Run centrality to populate an/c/ prefix
        run_centrality(&hg, "test-gen").unwrap();

        // Use mock extractor (will fallback since MockExtractor returns error for answer_question)
        let mock = crate::ingestion::llm::MockExtractor;
        let summaries = generate_summaries("test-gen", &hg, &mock, store.as_ref()).unwrap();

        assert!(!summaries.is_empty());
        // All summaries should have entity_count > 0
        for s in &summaries {
            assert!(s.entity_count > 0);
            assert_eq!(s.narrative_id, "test-gen");
        }

        // Verify they were cached
        let listed = list_summaries(store.as_ref(), "test-gen").unwrap();
        assert_eq!(listed.len(), summaries.len());
    }

    #[test]
    fn test_get_summary_not_found() {
        let store = MemoryStore::new();
        let result = get_summary(&store, "nar", 999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_summary_serialization_roundtrip() {
        let summary = CommunitySummary {
            narrative_id: "n".to_string(),
            community_id: 42,
            summary: "Test".to_string(),
            entity_ids: vec![Uuid::nil()],
            entity_names: vec!["X".to_string()],
            key_themes: vec!["theme".to_string()],
            entity_count: 1,
            generated_at: Utc::now(),
            model: None,
            level: 0,
            parent_community_id: None,
            child_community_ids: vec![],
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: CommunitySummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.community_id, 42);
        assert_eq!(parsed.summary, "Test");
    }
}
