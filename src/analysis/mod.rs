//! Analysis module — advanced analytical capabilities for narrative intelligence.
//!
//! Provides six analysis engines that operate on the hypergraph:
//! - **Centrality**: Network centrality measures and community detection
//! - **Entropy**: Information-theoretic situation surprise and mutual information
//! - **Beliefs**: Recursive belief modeling (depth 2) for epistemic reasoning
//! - **Evidence**: Dempster-Shafer belief functions for uncertainty quantification
//! - **Argumentation**: Dung's abstract argumentation frameworks for claim resolution
//! - **Contagion**: SIR-based information spread modeling with R₀ computation

pub mod ach;
pub mod alerts;
pub mod anomaly;
pub mod argumentation;
pub mod argumentation_gradual;
pub mod beliefs;
pub mod centrality;
#[cfg(feature = "disinfo")]
pub mod cib;
pub mod community;
pub mod community_detect;
pub mod contagion;
pub mod contagion_bistability;
pub mod embeddings;
pub mod entropy;
pub mod evidence;
pub mod evolution;
pub mod graph_centrality;
pub mod graph_projection;
pub mod higher_order_contagion;
pub mod investigation;
pub mod link_prediction;
pub mod narrative_centrality;
pub mod opinion_dynamics;
pub mod pan_loader;
pub mod pan_verification;
pub mod pathfinding;
pub mod psl;
pub mod similarity;
pub mod similarity_metrics;
#[cfg(feature = "disinfo")]
pub mod spread_intervention;
pub mod style_profile;
pub mod stylometry;
pub mod stylometry_stats;
pub mod tcg;
pub mod temporal_motifs;
pub mod topology;
#[cfg(feature = "disinfo")]
pub mod velocity_monitor;

use crate::error::{Result, TensaError};
use crate::inference::types::InferenceJob;

/// Extract `narrative_id` from inference job parameters.
/// Shared by all analysis engine `execute()` implementations.
pub(crate) fn extract_narrative_id(job: &InferenceJob) -> Result<&str> {
    job.parameters
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| TensaError::InferenceError("missing narrative_id".into()))
}

/// Load situations for a narrative, sorted chronologically by temporal start.
pub(crate) fn load_sorted_situations(
    hypergraph: &crate::hypergraph::Hypergraph,
    narrative_id: &str,
) -> Result<Vec<crate::types::Situation>> {
    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    situations.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));
    Ok(situations)
}

/// Build an analysis KV key from a prefix constant and path segments.
pub(crate) fn analysis_key(prefix: &[u8], segments: &[&str]) -> Vec<u8> {
    let prefix_str = std::str::from_utf8(prefix).unwrap_or("");
    let full = format!("{}{}", prefix_str, segments.join("/"));
    full.into_bytes()
}

/// Store per-entity scalar scores in KV at the given prefix.
/// Shared by all Level 1 graph analysis engines.
pub(crate) fn store_entity_scores(
    hypergraph: &crate::hypergraph::Hypergraph,
    entities: &[uuid::Uuid],
    scores: &[f64],
    prefix: &[u8],
    narrative_id: &str,
) -> Result<()> {
    for (i, eid) in entities.iter().enumerate() {
        let key = analysis_key(prefix, &[narrative_id, &eid.to_string()]);
        let bytes = serde_json::to_vec(&scores[i])?;
        hypergraph.store().put(&key, &bytes)?;
    }
    Ok(())
}

/// Build the standard `InferenceResult` returned by Level 1 graph analysis engines.
pub(crate) fn make_engine_result(
    job: &InferenceJob,
    job_type: crate::types::InferenceJobType,
    narrative_id: &str,
    scores_json: Vec<serde_json::Value>,
    explanation: &str,
) -> crate::types::InferenceResult {
    crate::types::InferenceResult {
        job_id: job.id.clone(),
        job_type,
        target_id: job.target_id,
        result: serde_json::json!({"narrative_id": narrative_id, "scores": scores_json}),
        confidence: 1.0,
        explanation: Some(explanation.into()),
        status: crate::types::JobStatus::Completed,
        created_at: job.created_at,
        completed_at: Some(chrono::Utc::now()),
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use crate::hypergraph::Hypergraph;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    pub fn make_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    pub fn add_entity(hg: &Hypergraph, name: &str, narrative: &str) -> Uuid {
        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some(narrative.to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap()
    }

    pub fn add_situation(hg: &Hypergraph, narrative: &str) -> Uuid {
        add_situation_with_level(hg, narrative, NarrativeLevel::Scene)
    }

    pub fn add_situation_with_level(
        hg: &Hypergraph,
        narrative: &str,
        level: NarrativeLevel,
    ) -> Uuid {
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
            raw_content: vec![ContentBlock::text("test")],
            narrative_level: level,
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

    pub fn link(hg: &Hypergraph, entity: Uuid, situation: Uuid) {
        hg.add_participant(Participation {
            entity_id: entity,
            situation_id: situation,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    pub fn link_with_action(hg: &Hypergraph, entity: Uuid, situation: Uuid, action: Option<&str>) {
        hg.add_participant(Participation {
            entity_id: entity,
            situation_id: situation,
            role: Role::Protagonist,
            info_set: None,
            action: action.map(String::from),
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }
}
