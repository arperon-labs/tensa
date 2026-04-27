//! Situation report generation (Sprint D8.5).
//!
//! Generates periodic disinformation situation reports aggregating data
//! from all analysis modules for a given time window.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::Hypergraph;

/// A periodic situation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationReport {
    pub id: uuid::Uuid,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub narratives_tracked: usize,
    pub new_narratives: Vec<NarrativeSummary>,
    pub active_cib_clusters: usize,
    pub top_claims: Vec<ClaimSummary>,
    pub velocity_alerts: usize,
    pub new_sources_discovered: usize,
    pub fact_checks_ingested: usize,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSummary {
    pub narrative_id: String,
    pub entity_count: usize,
    pub situation_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimSummary {
    pub claim_text: String,
    pub category: String,
    pub confidence: f64,
}

/// Generate a situation report for the given time window.
pub fn generate_situation_report(
    hypergraph: &Hypergraph,
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
) -> Result<SituationReport> {
    // Count narratives
    let narratives = crate::narrative::registry::NarrativeRegistry::new(hypergraph.store_arc())
        .list(None, None)?;
    let narratives_tracked = narratives.len();

    // Build narrative summaries
    let new_narratives: Vec<NarrativeSummary> = narratives
        .iter()
        .take(10)
        .map(|n| {
            let entities = hypergraph
                .list_entities_by_narrative(&n.id)
                .unwrap_or_default();
            let situations = hypergraph
                .list_situations_by_narrative(&n.id)
                .unwrap_or_default();
            NarrativeSummary {
                narrative_id: n.id.clone(),
                entity_count: entities.len(),
                situation_count: situations.len(),
            }
        })
        .collect();

    // Count CIB clusters across all narratives
    let mut active_cib = 0usize;
    for n in &narratives {
        if let Ok(clusters) = crate::analysis::cib::list_clusters(hypergraph, &n.id) {
            active_cib += clusters.len();
        }
    }

    // Count top claims
    let mut top_claims = Vec::new();
    for n in &narratives {
        if let Ok(claims) = crate::claims::detection::list_claims_for_narrative(hypergraph, &n.id) {
            for claim in claims.iter().take(5) {
                top_claims.push(ClaimSummary {
                    claim_text: if claim.text.len() > 100 {
                        format!("{}...", &claim.text[..100])
                    } else {
                        claim.text.clone()
                    },
                    category: format!("{:?}", claim.category),
                    confidence: claim.confidence,
                });
            }
        }
    }
    top_claims.truncate(20);

    // Count velocity alerts (all narratives)
    let mut velocity_alerts = 0usize;
    let monitor = crate::analysis::velocity_monitor::VelocityMonitor::new(hypergraph);
    for n in &narratives {
        if let Ok(alerts) = monitor.recent_alerts(&n.id, 100) {
            velocity_alerts += alerts
                .iter()
                .filter(|a| a.fired_at >= period_start && a.fired_at <= period_end)
                .count();
        }
    }

    // Count discovery candidates in time window
    let new_sources = crate::ingestion::discovery::list_candidates(hypergraph)
        .map(|c| {
            c.iter()
                .filter(|c| c.first_seen >= period_start && c.first_seen <= period_end)
                .count()
        })
        .unwrap_or(0);

    let report = SituationReport {
        id: uuid::Uuid::now_v7(),
        period_start,
        period_end,
        narratives_tracked,
        new_narratives,
        active_cib_clusters: active_cib,
        top_claims,
        velocity_alerts,
        new_sources_discovered: new_sources,
        fact_checks_ingested: 0, // Would count from fc/sync/ entries
        generated_at: Utc::now(),
    };

    // Persist
    store_report(hypergraph, &report)?;

    Ok(report)
}

/// Store a report at `reports/{id}`.
pub fn store_report(hypergraph: &Hypergraph, report: &SituationReport) -> Result<()> {
    let key = format!("reports/{}", report.id);
    let value = serde_json::to_vec(report).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

/// Load a report by ID.
pub fn load_report(hypergraph: &Hypergraph, id: &uuid::Uuid) -> Result<Option<SituationReport>> {
    let key = format!("reports/{}", id);
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

/// List all reports, most recent first.
pub fn list_reports(hypergraph: &Hypergraph, limit: usize) -> Result<Vec<SituationReport>> {
    let prefix = b"reports/";
    let pairs = hypergraph
        .store()
        .prefix_scan(prefix)
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    let mut reports: Vec<SituationReport> = pairs
        .into_iter()
        .filter_map(|(_, v)| serde_json::from_slice(&v).ok())
        .collect();
    reports.sort_by(|a, b| b.generated_at.cmp(&a.generated_at));
    reports.truncate(limit);
    Ok(reports)
}

/// Render report as Markdown.
pub fn render_markdown(report: &SituationReport) -> String {
    let mut md = String::new();
    md.push_str("# Situation Report\n\n");
    md.push_str(&format!(
        "**Period:** {} -- {}\n\n",
        report.period_start.format("%Y-%m-%d %H:%M"),
        report.period_end.format("%Y-%m-%d %H:%M")
    ));
    md.push_str(&format!(
        "**Generated:** {}\n\n",
        report.generated_at.format("%Y-%m-%d %H:%M")
    ));
    md.push_str("## Overview\n\n");
    md.push_str("| Metric | Value |\n|--------|-------|\n");
    md.push_str(&format!(
        "| Narratives tracked | {} |\n",
        report.narratives_tracked
    ));
    md.push_str(&format!(
        "| Active CIB clusters | {} |\n",
        report.active_cib_clusters
    ));
    md.push_str(&format!(
        "| Velocity alerts | {} |\n",
        report.velocity_alerts
    ));
    md.push_str(&format!(
        "| New sources discovered | {} |\n",
        report.new_sources_discovered
    ));
    md.push_str(&format!(
        "| Fact-checks ingested | {} |\n",
        report.fact_checks_ingested
    ));
    if !report.top_claims.is_empty() {
        md.push_str("\n## Top Claims\n\n");
        for claim in &report.top_claims {
            md.push_str(&format!(
                "- **[{}]** (conf: {:.2}) {}\n",
                claim.category, claim.confidence, claim.claim_text
            ));
        }
    }
    if !report.new_narratives.is_empty() {
        md.push_str("\n## Narratives\n\n");
        md.push_str(
            "| Narrative | Entities | Situations |\n|-----------|----------|------------|\n",
        );
        for n in &report.new_narratives {
            md.push_str(&format!(
                "| {} | {} | {} |\n",
                n.narrative_id, n.entity_count, n.situation_count
            ));
        }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    #[test]
    fn test_store_and_load_report() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let report = SituationReport {
            id: uuid::Uuid::now_v7(),
            period_start: Utc::now() - chrono::Duration::hours(24),
            period_end: Utc::now(),
            narratives_tracked: 3,
            new_narratives: vec![],
            active_cib_clusters: 1,
            top_claims: vec![],
            velocity_alerts: 2,
            new_sources_discovered: 5,
            fact_checks_ingested: 10,
            generated_at: Utc::now(),
        };
        store_report(&hg, &report).unwrap();
        let loaded = load_report(&hg, &report.id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().narratives_tracked, 3);
    }

    #[test]
    fn test_render_markdown() {
        let report = SituationReport {
            id: uuid::Uuid::now_v7(),
            period_start: Utc::now() - chrono::Duration::hours(24),
            period_end: Utc::now(),
            narratives_tracked: 5,
            new_narratives: vec![],
            active_cib_clusters: 2,
            top_claims: vec![ClaimSummary {
                claim_text: "Test claim".to_string(),
                category: "Numerical".to_string(),
                confidence: 0.85,
            }],
            velocity_alerts: 1,
            new_sources_discovered: 3,
            fact_checks_ingested: 7,
            generated_at: Utc::now(),
        };
        let md = render_markdown(&report);
        assert!(md.contains("Situation Report"));
        assert!(md.contains("5"));
        assert!(md.contains("Test claim"));
    }

    #[test]
    fn test_list_reports_empty() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let reports = list_reports(&hg, 10).unwrap();
        assert!(reports.is_empty());
    }
}
