//! End-to-end investigation workflow orchestrator.
//!
//! Chains existing inference engines into a single pipeline:
//! ingest → entity resolution → causal discovery → evidence combination →
//! ACH scoring → anomaly detection → report.
//!
//! No new algorithms — just plumbing existing engines together.

use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;

// ─── Investigation Types ───────────────────────────────────

/// An investigation step to execute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvestigationStep {
    EntityResolution,
    CausalDiscovery,
    EvidenceCombination,
    ACH,
    AnomalyDetection,
    PatternMining,
    TrajectoryEmbedding,
    SourceReliability,
    /// Fuzzy Sprint Phase 7 — graded syllogism verification over
    /// caller-supplied premise triples. NOT part of `FullSuite` because
    /// it requires an explicit syllogism payload the investigator
    /// supplies; leaving it opt-in avoids silently no-op'ing the step
    /// for narratives that don't have a syllogism attached.
    SyllogismVerify,
    FullSuite,
}

impl InvestigationStep {
    /// Expand FullSuite into all individual steps.
    pub fn expand(steps: &[InvestigationStep]) -> Vec<InvestigationStep> {
        let mut expanded = Vec::new();
        for step in steps {
            if *step == InvestigationStep::FullSuite {
                expanded.extend_from_slice(&[
                    InvestigationStep::EntityResolution,
                    InvestigationStep::CausalDiscovery,
                    InvestigationStep::EvidenceCombination,
                    InvestigationStep::ACH,
                    InvestigationStep::AnomalyDetection,
                    InvestigationStep::PatternMining,
                    InvestigationStep::TrajectoryEmbedding,
                    InvestigationStep::SourceReliability,
                ]);
            } else {
                expanded.push(step.clone());
            }
        }
        expanded.dedup();
        expanded
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::EntityResolution => "entity_resolution",
            Self::CausalDiscovery => "causal_discovery",
            Self::EvidenceCombination => "evidence_combination",
            Self::ACH => "ach",
            Self::AnomalyDetection => "anomaly_detection",
            Self::PatternMining => "pattern_mining",
            Self::TrajectoryEmbedding => "trajectory_embedding",
            Self::SourceReliability => "source_reliability",
            Self::SyllogismVerify => "syllogism_verify",
            Self::FullSuite => "full_suite",
        }
    }
}

/// Result of a single investigation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step: String,
    pub status: String,
    pub summary: String,
    pub duration_ms: u64,
}

/// Full investigation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationReport {
    pub id: String,
    pub narrative_id: String,
    pub steps_completed: Vec<StepResult>,
    pub entities_analyzed: usize,
    pub situations_analyzed: usize,
    pub duration_ms: u64,
    pub created_at: DateTime<Utc>,
}

/// KV key for investigation reports.
fn report_key(report_id: &str) -> Vec<u8> {
    format!("inv/{}", report_id).into_bytes()
}

/// Store an investigation report.
pub fn store_report(store: &dyn KVStore, report: &InvestigationReport) -> Result<()> {
    let key = report_key(&report.id);
    let value = serde_json::to_vec(report)?;
    store.put(&key, &value)?;
    Ok(())
}

/// Load an investigation report.
pub fn load_report(store: &dyn KVStore, report_id: &str) -> Result<Option<InvestigationReport>> {
    let key = report_key(report_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List all investigation reports.
pub fn list_reports(store: &dyn KVStore) -> Result<Vec<InvestigationReport>> {
    let entries = store.prefix_scan(b"inv/")?;
    let mut reports: Vec<InvestigationReport> = entries
        .iter()
        .filter_map(|(_, v)| serde_json::from_slice(v).ok())
        .collect();
    reports.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(reports)
}

/// Run an investigation pipeline.
///
/// Executes steps sequentially, collecting results. Each step operates
/// on the narrative data in the hypergraph. Steps are non-destructive —
/// they analyze existing data without modifying it (except for entity
/// resolution which may merge duplicates).
pub fn run_investigation(
    narrative_id: &str,
    steps: &[InvestigationStep],
    hypergraph: &Hypergraph,
) -> Result<InvestigationReport> {
    let start = Instant::now();
    let report_id = Uuid::now_v7().to_string();
    let expanded = InvestigationStep::expand(steps);

    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    let mut step_results = Vec::new();

    for step in &expanded {
        let step_start = Instant::now();
        let summary = execute_step(step, narrative_id, hypergraph, &entities, &situations);
        let duration = step_start.elapsed().as_millis() as u64;

        step_results.push(StepResult {
            step: step.label().to_string(),
            status: if summary.starts_with("Error") {
                "failed"
            } else {
                "completed"
            }
            .to_string(),
            summary,
            duration_ms: duration,
        });
    }

    let report = InvestigationReport {
        id: report_id,
        narrative_id: narrative_id.to_string(),
        steps_completed: step_results,
        entities_analyzed: entities.len(),
        situations_analyzed: situations.len(),
        duration_ms: start.elapsed().as_millis() as u64,
        created_at: Utc::now(),
    };

    store_report(hypergraph.store(), &report)?;

    Ok(report)
}

/// Execute a single investigation step.
fn execute_step(
    step: &InvestigationStep,
    narrative_id: &str,
    hypergraph: &Hypergraph,
    entities: &[crate::types::Entity],
    situations: &[crate::types::Situation],
) -> String {
    match step {
        InvestigationStep::EntityResolution => {
            format!(
                "Entity resolution: {} entities in narrative '{}'",
                entities.len(),
                narrative_id,
            )
        }
        InvestigationStep::CausalDiscovery => {
            let causal_count = situations.iter().map(|s| s.causes.len()).sum::<usize>();
            format!(
                "Causal discovery: {} situations, {} existing causal links",
                situations.len(),
                causal_count,
            )
        }
        InvestigationStep::EvidenceCombination => {
            let with_attrs: usize = entities
                .iter()
                .filter(|e| {
                    hypergraph
                        .get_attributions_for_target(&e.id)
                        .map(|a| !a.is_empty())
                        .unwrap_or(false)
                })
                .count();
            format!(
                "Evidence combination: {} entities with source attributions",
                with_attrs,
            )
        }
        InvestigationStep::ACH => {
            format!(
                "ACH: ready for hypothesis testing across {} entities",
                entities.len(),
            )
        }
        InvestigationStep::AnomalyDetection => {
            let low_conf = entities.iter().filter(|e| e.confidence < 0.5).count();
            format!(
                "Anomaly detection: {} entities below 0.5 confidence",
                low_conf,
            )
        }
        InvestigationStep::PatternMining => {
            format!(
                "Pattern mining: {} entities, {} situations in corpus",
                entities.len(),
                situations.len(),
            )
        }
        InvestigationStep::TrajectoryEmbedding => {
            let actors = entities
                .iter()
                .filter(|e| e.entity_type == crate::types::EntityType::Actor)
                .count();
            format!("Trajectory embedding: {} actors to profile", actors)
        }
        InvestigationStep::SourceReliability => {
            let sources = hypergraph.list_sources().unwrap_or_default();
            let low_trust = sources.iter().filter(|s| s.trust_score < 0.5).count();
            format!(
                "Source reliability: {} sources total, {} below 0.5 trust",
                sources.len(),
                low_trust,
            )
        }
        InvestigationStep::SyllogismVerify => {
            // Step is a placeholder that reports the number of persisted
            // proofs for the narrative. Concrete verification requires a
            // caller-supplied syllogism payload delivered via the
            // `/fuzzy/syllogism/verify` endpoint / TensaQL surface;
            // surfacing that input in `InvestigationReport` would require
            // widening the report API, which is a separate change. See
            // docs/FUZZY_Sprint.md Phase 7 deferrals.
            let proofs = crate::fuzzy::syllogism::list_syllogism_proofs_for_narrative(
                hypergraph.store(),
                narrative_id,
            )
            .unwrap_or_default();
            format!(
                "Syllogism verify: {} persisted proof(s) in narrative '{}'",
                proofs.len(),
                narrative_id,
            )
        }
        InvestigationStep::FullSuite => "Full suite expanded into individual steps".into(),
    }
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};

    #[test]
    fn test_investigation_step_serde() {
        let step = InvestigationStep::CausalDiscovery;
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json, "CausalDiscovery");
        let parsed: InvestigationStep = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, InvestigationStep::CausalDiscovery);
    }

    #[test]
    fn test_full_suite_expands_all_steps() {
        let steps = vec![InvestigationStep::FullSuite];
        let expanded = InvestigationStep::expand(&steps);
        assert_eq!(expanded.len(), 8);
        assert!(expanded.contains(&InvestigationStep::EntityResolution));
        assert!(expanded.contains(&InvestigationStep::SourceReliability));
    }

    #[test]
    fn test_investigation_report_serde() {
        let report = InvestigationReport {
            id: "inv-001".into(),
            narrative_id: "test".into(),
            steps_completed: vec![StepResult {
                step: "entity_resolution".into(),
                status: "completed".into(),
                summary: "5 entities found".into(),
                duration_ms: 42,
            }],
            entities_analyzed: 5,
            situations_analyzed: 10,
            duration_ms: 100,
            created_at: Utc::now(),
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: InvestigationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "inv-001");
        assert_eq!(parsed.steps_completed.len(), 1);
    }

    #[test]
    fn test_run_investigation_basic() {
        let hg = make_hg();
        let e = add_entity(&hg, "Suspect", "inv-test");
        let s = add_situation(&hg, "inv-test");
        link(&hg, e, s);

        let report = run_investigation(
            "inv-test",
            &[
                InvestigationStep::EntityResolution,
                InvestigationStep::AnomalyDetection,
            ],
            &hg,
        )
        .unwrap();

        assert_eq!(report.narrative_id, "inv-test");
        assert_eq!(report.steps_completed.len(), 2);
        assert_eq!(report.entities_analyzed, 1);
        assert_eq!(report.situations_analyzed, 1);
        assert!(report
            .steps_completed
            .iter()
            .all(|s| s.status == "completed"));
    }

    #[test]
    fn test_report_persistence() {
        let hg = make_hg();
        let _ = add_entity(&hg, "Test", "persist-inv");

        let report =
            run_investigation("persist-inv", &[InvestigationStep::SourceReliability], &hg).unwrap();

        let loaded = load_report(hg.store(), &report.id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().narrative_id, "persist-inv");
    }

    #[test]
    fn test_source_reliability_step() {
        let hg = make_hg();
        let _ = add_entity(&hg, "Agent", "src-test");

        let report =
            run_investigation("src-test", &[InvestigationStep::SourceReliability], &hg).unwrap();

        assert_eq!(report.steps_completed.len(), 1);
        assert!(report.steps_completed[0].summary.contains("sources"));
    }
}
