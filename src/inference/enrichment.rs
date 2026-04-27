//! Pre-computed enrichment cache for inference results.
//!
//! Checks for existing completed results before submitting new jobs,
//! providing a fast path for repeated queries on the same target.
//!
//! Also provides post-execution enrichment: writing analysis results
//! back into entity properties and situation fields so downstream
//! analyses and queries can build on upstream results.

use std::sync::Arc;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

use super::jobs::JobQueue;
use super::types::InferenceJob;

/// Pre-computed result cache backed by the job queue's target index.
pub struct EnrichmentCache {
    job_queue: Arc<JobQueue>,
}

impl EnrichmentCache {
    /// Create a new enrichment cache.
    pub fn new(job_queue: Arc<JobQueue>) -> Self {
        Self { job_queue }
    }

    /// Check for a cached result for the given target and job type.
    /// Returns the most recent completed result, if any.
    pub fn get_cached(
        &self,
        target_id: &uuid::Uuid,
        job_type: &InferenceJobType,
    ) -> Result<Option<InferenceResult>> {
        let jobs = self.job_queue.list_by_target(target_id)?;

        // Find the most recent completed job of the requested type
        let completed = jobs
            .iter()
            .filter(|j| j.job_type == *job_type && j.status == JobStatus::Completed)
            .max_by_key(|j| j.completed_at);

        match completed {
            Some(job) => match self.job_queue.get_result(&job.id) {
                Ok(result) => Ok(Some(result)),
                Err(_) => Ok(None),
            },
            None => Ok(None),
        }
    }

    /// Check if a result exists and return it, or return None for cache miss.
    pub fn try_fast_path(
        &self,
        target_id: &uuid::Uuid,
        job_type: &InferenceJobType,
    ) -> Result<Option<InferenceResult>> {
        self.get_cached(target_id, job_type)
    }
}

/// Enrich entities and situations with analysis results after a job completes.
///
/// This is called by the worker pool after storing the job result. It writes
/// key findings back into entity.properties / situation fields so they're
/// visible in queries and available for downstream analyses.
///
/// Errors are logged but never propagated — enrichment failure should not
/// cause the job to be marked as failed.
pub fn enrich_from_result(job: &InferenceJob, result: &InferenceResult, hypergraph: &Hypergraph) {
    if let Err(e) = try_enrich(job, result, hypergraph) {
        tracing::warn!(
            "Enrichment failed for job {} ({:?}): {}",
            job.id,
            job.job_type,
            e
        );
    }
}

fn try_enrich(job: &InferenceJob, result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    match job.job_type {
        InferenceJobType::CentralityAnalysis => enrich_centrality(result, hypergraph),
        InferenceJobType::MotivationInference => enrich_motivation(job, result, hypergraph),
        InferenceJobType::CausalDiscovery => enrich_causal(result, hypergraph),
        InferenceJobType::GameClassification => enrich_game(job, result, hypergraph),
        InferenceJobType::ArcClassification => enrich_arc(result, hypergraph),
        InferenceJobType::BeliefModeling => enrich_beliefs(result, hypergraph),
        InferenceJobType::ContagionAnalysis => enrich_contagion(result, hypergraph),
        InferenceJobType::EntropyAnalysis => enrich_entropy(result, hypergraph),
        InferenceJobType::EvidenceCombination => enrich_evidence(job, result, hypergraph),
        InferenceJobType::ArgumentationAnalysis => enrich_argumentation(result, hypergraph),
        _ => Ok(()),
    }
}

/// Store centrality scores (betweenness, closeness, degree, community) in entity.properties.
fn enrich_centrality(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    let results = result.result.get("results").and_then(|v| v.as_array());
    let results = match results {
        Some(r) => r,
        None => return Ok(()),
    };

    let mut enriched = 0;
    for entry in results {
        let entity_id = match entry.get("entity_id").and_then(|v| v.as_str()) {
            Some(id) => match uuid::Uuid::parse_str(id) {
                Ok(u) => u,
                Err(_) => continue,
            },
            None => continue,
        };

        let centrality = serde_json::json!({
            "betweenness": entry.get("betweenness"),
            "closeness": entry.get("closeness"),
            "degree": entry.get("degree"),
            "community_id": entry.get("community_id"),
        });

        if let Ok(_entity) = hypergraph.get_entity(&entity_id) {
            let centrality_clone = centrality.clone();
            let _ = hypergraph.update_entity(&entity_id, |e| {
                if let Some(props) = e.properties.as_object_mut() {
                    props.insert("centrality".to_string(), centrality_clone);
                    e.updated_at = chrono::Utc::now();
                }
            });
            enriched += 1;
        }
    }
    if enriched > 0 {
        tracing::info!("Enriched {} entities with centrality scores", enriched);
    }
    Ok(())
}

/// Store motivation archetype and top features in entity.properties.
fn enrich_motivation(
    job: &InferenceJob,
    result: &InferenceResult,
    hypergraph: &Hypergraph,
) -> Result<()> {
    let archetype = result.result.get("archetype").and_then(|v| v.as_str());
    let features = result.result.get("feature_weights");

    if archetype.is_none() && features.is_none() {
        return Ok(());
    }

    let motivation = serde_json::json!({
        "archetype": archetype,
        "feature_weights": features,
        "confidence": result.confidence,
    });

    if let Ok(_entity) = hypergraph.get_entity(&job.target_id) {
        let motivation_clone = motivation.clone();
        let tid = job.target_id;
        let _ = hypergraph.update_entity(&job.target_id, |e| {
            if let Some(props) = e.properties.as_object_mut() {
                props.insert("motivation".to_string(), motivation_clone);
                e.updated_at = chrono::Utc::now();
            }
        });
        tracing::info!("Enriched entity {} with motivation profile", tid);
    }
    Ok(())
}

/// Add inferred causal links to the hypergraph as actual causal edges.
fn enrich_causal(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    let links = result.result.get("links").and_then(|v| v.as_array());
    let links = match links {
        Some(l) => l,
        None => return Ok(()),
    };

    let mut added = 0;
    for link in links {
        let from_id = link
            .get("from")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok());
        let to_id = link
            .get("to")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok());
        let confidence = link
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        if let (Some(from), Some(to)) = (from_id, to_id) {
            let causal_link = crate::types::CausalLink {
                from_situation: from,
                to_situation: to,
                causal_type: crate::types::CausalType::Contributing,
                mechanism: link
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                strength: confidence as f32,
                maturity: crate::types::MaturityLevel::Candidate,
            };
            match hypergraph.add_causal_link(causal_link) {
                Ok(_) => added += 1,
                Err(_) => {} // May already exist
            }
        }
    }
    if added > 0 {
        tracing::info!("Enriched hypergraph with {} inferred causal links", added);
    }
    Ok(())
}

/// Store game classification in situation.game_structure.
fn enrich_game(
    job: &InferenceJob,
    result: &InferenceResult,
    hypergraph: &Hypergraph,
) -> Result<()> {
    let game_type = result.result.get("game_type").and_then(|v| v.as_str());
    if game_type.is_none() {
        return Ok(());
    }

    let game_structure =
        serde_json::from_value::<crate::types::GameStructure>(result.result.clone());

    if let Ok(gs) = game_structure {
        if let Ok(_situation) = hypergraph.get_situation(&job.target_id) {
            let tid = job.target_id;
            let _ = hypergraph.update_situation(&job.target_id, |s| {
                s.game_structure = Some(gs.clone());
                s.updated_at = chrono::Utc::now();
            });
            tracing::info!("Enriched situation {} with game structure", tid);
        }
    }
    // Also write payoffs to participations from equilibria
    enrich_game_payoffs(job, result, hypergraph);
    Ok(())
}

/// Store game classification in situation + write payoffs to participations.
fn enrich_game_payoffs(job: &InferenceJob, result: &InferenceResult, hypergraph: &Hypergraph) {
    // Extract equilibria payoffs and write to participation.payoff
    if let Some(equilibria) = result.result.get("equilibria").and_then(|v| v.as_array()) {
        if let Some(first_eq) = equilibria.first() {
            if let Some(profile) = first_eq.get("strategy_profile").and_then(|v| v.as_array()) {
                for strategy in profile {
                    let entity_id = strategy
                        .get("entity_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| uuid::Uuid::parse_str(s).ok());
                    let payoff = strategy.get("expected_utility");
                    if let (Some(eid), Some(pay)) = (entity_id, payoff) {
                        if let Ok(parts) =
                            hypergraph.get_participations_for_pair(&eid, &job.target_id)
                        {
                            for mut p in parts {
                                p.payoff = Some(pay.clone());
                                let _ = hypergraph.update_participation(&p);
                            }
                        }
                    }
                }
                tracing::info!(
                    "Enriched participations with payoffs for situation {}",
                    job.target_id
                );
            }
        }
    }
}

/// Write belief snapshots back to participation InfoSet fields.
fn enrich_beliefs(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    let snapshots = result.result.get("snapshots").and_then(|v| v.as_array());
    let snapshots = match snapshots {
        Some(s) => s,
        None => return Ok(()),
    };

    let mut enriched = 0;
    for snap in snapshots {
        let entity_a = snap
            .get("entity_a")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok());
        let situation_id = snap
            .get("situation_id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok());
        let believed = snap
            .get("believed_knowledge")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let actual = snap
            .get("actual_knowledge")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if let (Some(eid), Some(sid)) = (entity_a, situation_id) {
            if let Ok(parts) = hypergraph.get_participations_for_pair(&eid, &sid) {
                for mut p in parts {
                    let entity_b = snap
                        .get("entity_b")
                        .and_then(|v| v.as_str())
                        .and_then(|s| uuid::Uuid::parse_str(s).ok())
                        .unwrap_or(uuid::Uuid::nil());
                    // Build InfoSet from belief data
                    let knows = believed
                        .iter()
                        .map(|f| crate::types::KnowledgeFact {
                            about_entity: entity_b,
                            fact: f.clone(),
                            confidence: 0.8,
                        })
                        .collect();
                    let learns = actual
                        .iter()
                        .filter(|f| !believed.contains(f))
                        .map(|f| crate::types::KnowledgeFact {
                            about_entity: entity_b,
                            fact: f.clone(),
                            confidence: 0.7,
                        })
                        .collect();
                    p.info_set = Some(crate::types::InfoSet {
                        knows_before: knows,
                        learns,
                        reveals: vec![],
                        beliefs_about_others: vec![],
                    });
                    let _ = hypergraph.update_participation(&p);
                    enriched += 1;
                }
            }
        }
    }
    if enriched > 0 {
        tracing::info!("Enriched {} participations with belief InfoSets", enriched);
    }
    Ok(())
}

/// Tag entities with contagion roles and R₀ data.
fn enrich_contagion(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    let r0 = result
        .result
        .get("r0")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let entity_states = result.result.get("entity_states");
    let critical_spreaders = result
        .result
        .get("critical_spreaders")
        .and_then(|v| v.as_array());

    // Build a set of critical spreader IDs
    let mut critical_ids = std::collections::HashSet::new();
    if let Some(spreaders) = critical_spreaders {
        for s in spreaders {
            if let Some(eid) = s
                .get("entity_id")
                .and_then(|v| v.as_str())
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
            {
                critical_ids.insert(eid);
            }
        }
    }

    let mut enriched = 0;
    if let Some(states) = entity_states.and_then(|v| v.as_object()) {
        for (eid_str, state) in states {
            let eid = match uuid::Uuid::parse_str(eid_str) {
                Ok(u) => u,
                Err(_) => continue,
            };
            let role = if critical_ids.contains(&eid) {
                "critical_spreader"
            } else {
                state.as_str().unwrap_or("unknown")
            };
            let contagion = serde_json::json!({
                "r0": r0,
                "role": role,
                "state": state,
            });
            let contagion_clone = contagion.clone();
            if hypergraph.get_entity(&eid).is_ok() {
                let _ = hypergraph.update_entity_no_snapshot(&eid, |e| {
                    if let Some(props) = e.properties.as_object_mut() {
                        props.insert("contagion".to_string(), contagion_clone);
                    }
                });
                enriched += 1;
            }
        }
    }
    if enriched > 0 {
        tracing::info!(
            "Enriched {} entities with contagion data (R₀={:.2})",
            enriched,
            r0
        );
    }
    Ok(())
}

/// Write entropy scores to entity properties.
fn enrich_entropy(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    // Enrich mutual information into entity properties
    let mi_results = result
        .result
        .get("mutual_information")
        .and_then(|v| v.as_array());
    let mut enriched = 0;
    if let Some(mi_arr) = mi_results {
        // Aggregate max MI per entity
        let mut entity_mi: std::collections::HashMap<uuid::Uuid, f64> =
            std::collections::HashMap::new();
        for mi in mi_arr {
            let val = mi
                .get("mutual_information")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            for field in &["entity_a", "entity_b"] {
                if let Some(eid) = mi
                    .get(*field)
                    .and_then(|v| v.as_str())
                    .and_then(|s| uuid::Uuid::parse_str(s).ok())
                {
                    let entry = entity_mi.entry(eid).or_insert(0.0);
                    if val > *entry {
                        *entry = val;
                    }
                }
            }
        }
        for (eid, max_mi) in &entity_mi {
            if hypergraph.get_entity(eid).is_ok() {
                let mi_val = *max_mi;
                let _ = hypergraph.update_entity_no_snapshot(eid, |e| {
                    if let Some(props) = e.properties.as_object_mut() {
                        props.insert(
                            "entropy".to_string(),
                            serde_json::json!({ "max_mutual_information": mi_val }),
                        );
                    }
                });
                enriched += 1;
            }
        }
    }

    // Enrich KL divergence into entity properties
    let kl_results = result
        .result
        .get("kl_divergences")
        .and_then(|v| v.as_array());
    if let Some(kl_arr) = kl_results {
        for kl in kl_arr {
            let eid = kl
                .get("entity_id")
                .and_then(|v| v.as_str())
                .and_then(|s| uuid::Uuid::parse_str(s).ok());
            let kl_val = kl.get("kl_divergence").and_then(|v| v.as_f64());
            if let (Some(eid), Some(val)) = (eid, kl_val) {
                if hypergraph.get_entity(&eid).is_ok() {
                    let _ = hypergraph.update_entity_no_snapshot(&eid, |e| {
                        if let Some(props) = e.properties.as_object_mut() {
                            let ent = props
                                .entry("entropy".to_string())
                                .or_insert(serde_json::json!({}));
                            if let Some(obj) = ent.as_object_mut() {
                                obj.insert("kl_divergence".to_string(), serde_json::json!(val));
                            }
                        }
                    });
                }
            }
        }
    }

    if enriched > 0 {
        tracing::info!("Enriched {} entities with entropy data", enriched);
    }
    Ok(())
}

/// Update confidence from Dempster-Shafer belief/plausibility intervals.
fn enrich_evidence(
    job: &InferenceJob,
    result: &InferenceResult,
    hypergraph: &Hypergraph,
) -> Result<()> {
    let bp = result
        .result
        .get("belief_plausibility")
        .and_then(|v| v.as_array());
    let bp = match bp {
        Some(b) => b,
        None => return Ok(()),
    };

    // Store evidence summary on the target entity/situation
    let target = job.target_id;
    let evidence = serde_json::json!({
        "belief_plausibility": bp,
        "conflict": result.result.get("conflict"),
        "frame": result.result.get("frame"),
    });

    if let Ok(_e) = hypergraph.get_entity(&target) {
        let evidence_clone = evidence.clone();
        let _ = hypergraph.update_entity_no_snapshot(&target, |e| {
            if let Some(props) = e.properties.as_object_mut() {
                props.insert("evidence".to_string(), evidence_clone);
            }
        });
        tracing::info!("Enriched entity {} with Dempster-Shafer evidence", target);
    }
    Ok(())
}

/// Tag entities with argumentation status (In/Out/Undecided).
fn enrich_argumentation(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    let grounded = result.result.get("grounded").and_then(|v| v.as_array());
    let grounded = match grounded {
        Some(g) => g,
        None => return Ok(()),
    };

    let mut enriched = 0;
    for entry in grounded {
        let arr = match entry.as_array() {
            Some(a) if a.len() == 2 => a,
            _ => continue,
        };
        let eid = arr[0].as_str().and_then(|s| uuid::Uuid::parse_str(s).ok());
        let label = arr[1].as_str().unwrap_or("Undec");

        if let Some(eid) = eid {
            if hypergraph.get_entity(&eid).is_ok() {
                let label_str = label.to_string();
                let _ = hypergraph.update_entity_no_snapshot(&eid, |e| {
                    if let Some(props) = e.properties.as_object_mut() {
                        props.insert(
                            "argumentation".to_string(),
                            serde_json::json!({ "status": label_str }),
                        );
                    }
                });
                enriched += 1;
            }
        }
    }
    if enriched > 0 {
        tracing::info!("Enriched {} entities with argumentation labels", enriched);
    }
    Ok(())
}

/// Store arc classification in narrative metadata.
fn enrich_arc(result: &InferenceResult, hypergraph: &Hypergraph) -> Result<()> {
    let arc_type = result.result.get("arc_type").and_then(|v| v.as_str());
    let narrative_id = result.result.get("narrative_id").and_then(|v| v.as_str());

    if let (Some(arc), Some(nid)) = (arc_type, narrative_id) {
        let registry = crate::narrative::registry::NarrativeRegistry::new(hypergraph.store_arc());
        let tag = format!("arc:{}", arc.to_lowercase().replace(' ', "-"));
        let _ = registry.update(nid, |meta| {
            if meta.tags.iter().all(|t| !t.starts_with("arc:")) {
                meta.tags.push(tag.clone());
            }
        });
        tracing::info!("Enriched narrative {} with arc type: {}", nid, arc);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::InferenceJob;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn setup() -> (Arc<JobQueue>, EnrichmentCache) {
        let store = Arc::new(MemoryStore::new());
        let queue = Arc::new(super::super::jobs::JobQueue::new(store));
        let cache = EnrichmentCache::new(queue.clone());
        (queue, cache)
    }

    #[test]
    fn test_cache_miss() {
        let (_queue, cache) = setup();
        let result = cache
            .get_cached(&Uuid::now_v7(), &InferenceJobType::CausalDiscovery)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_hit() {
        let (queue, cache) = setup();
        let target = Uuid::now_v7();

        let job = InferenceJob {
            id: "cached-001".to_string(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: target,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 1000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        queue.submit(job).unwrap();
        queue.mark_completed("cached-001").unwrap();

        let result = InferenceResult {
            job_id: "cached-001".to_string(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: target,
            result: serde_json::json!({"links": []}),
            confidence: 0.8,
            explanation: None,
            status: JobStatus::Completed,
            created_at: Utc::now(),
            completed_at: Some(Utc::now()),
        };
        queue.store_result(result).unwrap();

        let cached = cache
            .get_cached(&target, &InferenceJobType::CausalDiscovery)
            .unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().confidence, 0.8);
    }

    #[test]
    fn test_cache_ignores_pending_jobs() {
        let (queue, cache) = setup();
        let target = Uuid::now_v7();

        let job = InferenceJob {
            id: "pending-001".to_string(),
            job_type: InferenceJobType::CausalDiscovery,
            target_id: target,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 1000,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        queue.submit(job).unwrap();

        let cached = cache
            .get_cached(&target, &InferenceJobType::CausalDiscovery)
            .unwrap();
        assert!(cached.is_none());
    }

    #[test]
    fn test_cache_filters_by_job_type() {
        let (queue, cache) = setup();
        let target = Uuid::now_v7();

        let job = InferenceJob {
            id: "game-001".to_string(),
            job_type: InferenceJobType::GameClassification,
            target_id: target,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 500,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        queue.submit(job).unwrap();
        queue.mark_completed("game-001").unwrap();
        queue
            .store_result(InferenceResult {
                job_id: "game-001".to_string(),
                job_type: InferenceJobType::GameClassification,
                target_id: target,
                result: serde_json::json!({}),
                confidence: 0.7,
                explanation: None,
                status: JobStatus::Completed,
                created_at: Utc::now(),
                completed_at: Some(Utc::now()),
            })
            .unwrap();

        // Query for a different type
        let cached = cache
            .get_cached(&target, &InferenceJobType::CausalDiscovery)
            .unwrap();
        assert!(cached.is_none());

        // Query for the correct type
        let cached = cache
            .get_cached(&target, &InferenceJobType::GameClassification)
            .unwrap();
        assert!(cached.is_some());
    }

    #[test]
    fn test_fast_path() {
        let (_queue, cache) = setup();
        let result = cache
            .try_fast_path(&Uuid::now_v7(), &InferenceJobType::CausalDiscovery)
            .unwrap();
        assert!(result.is_none());
    }
}
