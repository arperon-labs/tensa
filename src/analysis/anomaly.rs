//! Anomaly detection engine for narrative data.
//!
//! Uses z-score based anomaly detection on entity confidence values,
//! situation confidence values, and temporal gap sizes between
//! consecutive situations. Items with |z| > threshold are flagged.
//! Results are persisted to KV at `an/ad/{narrative_id}`.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::KVStore;
use crate::types::*;

use super::extract_narrative_id;

/// KV prefix for persisted anomaly reports.
pub const ANOMALY_PREFIX: &str = "an/ad/";

/// Configuration for anomaly detection thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Z-score threshold for flagging anomalies. Default: 2.0.
    #[serde(default = "default_z_threshold")]
    pub z_threshold: f64,
}

fn default_z_threshold() -> f64 {
    2.0
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            z_threshold: default_z_threshold(),
        }
    }
}

/// Z-score based anomaly detection engine.
pub struct AnomalyDetectionEngine {
    pub config: AnomalyConfig,
}

impl Default for AnomalyDetectionEngine {
    fn default() -> Self {
        Self {
            config: AnomalyConfig::default(),
        }
    }
}

impl InferenceEngine for AnomalyDetectionEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::AnomalyDetection
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(4000) // 4 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let report = detect_anomalies(narrative_id, hypergraph, &self.config)?;

        // Persist report to KV
        persist_anomaly_report(hypergraph.store(), &report)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::AnomalyDetection,
            target_id: job.target_id,
            result: serde_json::to_value(&report)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Anomaly detection: {} entity, {} situation, {} temporal gap anomalies",
                report.entity_anomalies.len(),
                report.situation_anomalies.len(),
                report.temporal_gap_anomalies.len(),
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Types ──────────────────────────────────────────────────

/// Full anomaly report for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyReport {
    pub narrative_id: String,
    pub entity_anomalies: Vec<EntityAnomaly>,
    pub situation_anomalies: Vec<SituationAnomaly>,
    pub temporal_gap_anomalies: Vec<TemporalGapAnomaly>,
}

/// An entity with anomalous confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAnomaly {
    pub entity_id: Uuid,
    pub confidence: f32,
    pub z_score: f64,
}

/// A situation with anomalous confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationAnomaly {
    pub situation_id: Uuid,
    pub confidence: f32,
    pub z_score: f64,
}

/// An anomalously large or small temporal gap between consecutive situations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalGapAnomaly {
    pub from_situation: Uuid,
    pub to_situation: Uuid,
    pub gap_hours: f64,
    pub z_score: f64,
}

// ─── Core Algorithm ─────────────────────────────────────────

/// Persist an anomaly report to KV.
pub fn persist_anomaly_report(store: &dyn KVStore, report: &AnomalyReport) -> Result<()> {
    let key = format!("{}{}", ANOMALY_PREFIX, report.narrative_id);
    let value = serde_json::to_vec(report)?;
    store.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Load a persisted anomaly report from KV.
pub fn get_anomaly_report(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Option<AnomalyReport>> {
    let key = format!("{}{}", ANOMALY_PREFIX, narrative_id);
    match store.get(key.as_bytes())? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Detect anomalies in a narrative using z-scores on confidence and temporal gaps.
fn detect_anomalies(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    config: &AnomalyConfig,
) -> Result<AnomalyReport> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    situations.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    // Entity confidence anomalies
    let entity_confs: Vec<f64> = entities.iter().map(|e| e.confidence as f64).collect();
    let (e_mean, e_std) = mean_std(&entity_confs);
    let entity_anomalies: Vec<EntityAnomaly> = entities
        .iter()
        .filter_map(|e| {
            let z = z_score(e.confidence as f64, e_mean, e_std);
            if z.abs() > config.z_threshold {
                Some(EntityAnomaly {
                    entity_id: e.id,
                    confidence: e.confidence,
                    z_score: z,
                })
            } else {
                None
            }
        })
        .collect();

    // Situation confidence anomalies
    let sit_confs: Vec<f64> = situations.iter().map(|s| s.confidence as f64).collect();
    let (s_mean, s_std) = mean_std(&sit_confs);
    let situation_anomalies: Vec<SituationAnomaly> = situations
        .iter()
        .filter_map(|s| {
            let z = z_score(s.confidence as f64, s_mean, s_std);
            if z.abs() > config.z_threshold {
                Some(SituationAnomaly {
                    situation_id: s.id,
                    confidence: s.confidence,
                    z_score: z,
                })
            } else {
                None
            }
        })
        .collect();

    // Temporal gap anomalies
    let mut gaps: Vec<(Uuid, Uuid, f64)> = Vec::new();
    for pair in situations.windows(2) {
        let from = &pair[0];
        let to = &pair[1];
        let from_end = from
            .temporal
            .end
            .or(from.temporal.start)
            .unwrap_or_default();
        let to_start = to.temporal.start.or(to.temporal.end).unwrap_or_default();
        let gap_hours = to_start.signed_duration_since(from_end).num_minutes().abs() as f64 / 60.0;
        gaps.push((from.id, to.id, gap_hours));
    }

    let gap_values: Vec<f64> = gaps.iter().map(|(_, _, g)| *g).collect();
    let (g_mean, g_std) = mean_std(&gap_values);
    let temporal_gap_anomalies: Vec<TemporalGapAnomaly> = gaps
        .iter()
        .filter_map(|(from, to, gap)| {
            let z = z_score(*gap, g_mean, g_std);
            if z.abs() > config.z_threshold {
                Some(TemporalGapAnomaly {
                    from_situation: *from,
                    to_situation: *to,
                    gap_hours: *gap,
                    z_score: z,
                })
            } else {
                None
            }
        })
        .collect();

    Ok(AnomalyReport {
        narrative_id: narrative_id.to_string(),
        entity_anomalies,
        situation_anomalies,
        temporal_gap_anomalies,
    })
}

/// Compute mean and sample standard deviation (Bessel's correction: n-1).
///
/// Uses `n-1` denominator for unbiased sample variance estimation.
/// Falls back to population stddev when n < 2.
fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let denom = if values.len() > 1 { n - 1.0 } else { n };
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / denom;
    (mean, variance.sqrt())
}

/// Compute z-score for a value given mean and standard deviation.
fn z_score(value: f64, mean: f64, std: f64) -> f64 {
    if std == 0.0 {
        return 0.0;
    }
    (value - mean) / std
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Duration;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_anomaly_engine_execute() {
        let hg = test_hg();
        let nid = "anomaly-test";

        // Create entities with varied confidence — one outlier
        for conf in &[0.8f32, 0.85, 0.82, 0.79, 0.15] {
            let entity = Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "test"}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: *conf,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some(nid.to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(entity).unwrap();
        }

        // Create situations with one temporal gap outlier
        let base = Utc::now();
        for (i, conf) in [0.7f32, 0.75, 0.72, 0.74, 0.1].iter().enumerate() {
            let offset = if i == 4 { 1000 } else { i as i64 * 2 }; // big gap before last
            let sit = Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(base + Duration::hours(offset)),
                    end: Some(base + Duration::hours(offset + 1)),
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
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: *conf,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some(nid.to_string()),
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
            hg.create_situation(sit).unwrap();
        }

        let engine = AnomalyDetectionEngine::default();
        assert_eq!(engine.job_type(), InferenceJobType::AnomalyDetection);

        let job = InferenceJob {
            id: "anomaly-001".to_string(),
            job_type: InferenceJobType::AnomalyDetection,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": nid}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());

        // Should detect the confidence outlier entity and situation
        let report: serde_json::Value = result.result;
        assert!(report.get("entity_anomalies").is_some());
        assert!(report.get("situation_anomalies").is_some());
        assert!(report.get("temporal_gap_anomalies").is_some());
    }

    #[test]
    fn test_mean_std_computation() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_std(&vals);
        assert!((mean - 5.0).abs() < 0.01);
        assert!(std > 0.0);
    }

    #[test]
    fn test_stddev_bessel_correction() {
        // Known values: [2, 4, 6, 8, 10] → mean=6, population var=8, sample var=10
        let vals = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (mean, std) = mean_std(&vals);
        assert!((mean - 6.0).abs() < 1e-10);
        // Sample std = sqrt(10) ≈ 3.162
        let expected_sample_std = (10.0_f64).sqrt();
        assert!(
            (std - expected_sample_std).abs() < 0.01,
            "Expected sample std {}, got {} (Bessel's correction should use n-1)",
            expected_sample_std,
            std
        );
        // Population std would be sqrt(8) ≈ 2.828, which should NOT match
        let population_std = (8.0_f64).sqrt();
        assert!((std - population_std).abs() > 0.1);
    }

    #[test]
    fn test_z_score_zero_std() {
        assert_eq!(z_score(5.0, 5.0, 0.0), 0.0);
    }

    #[test]
    fn test_anomaly_config_default() {
        let config = AnomalyConfig::default();
        assert_eq!(config.z_threshold, 2.0);
    }

    #[test]
    fn test_anomaly_config_custom_threshold() {
        let config = AnomalyConfig { z_threshold: 1.5 };
        let hg = test_hg();
        let nid = "config-test";
        for conf in &[0.8f32, 0.85, 0.82, 0.79, 0.15] {
            let entity = Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "test"}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: *conf,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some(nid.to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(entity).unwrap();
        }
        // Lower threshold should flag more anomalies
        let report = detect_anomalies(nid, &hg, &config).unwrap();
        assert!(report.entity_anomalies.len() >= 1);
    }

    #[test]
    fn test_anomaly_persist_and_read() {
        let store = Arc::new(MemoryStore::new());
        let report = AnomalyReport {
            narrative_id: "persist-test".to_string(),
            entity_anomalies: vec![EntityAnomaly {
                entity_id: Uuid::now_v7(),
                confidence: 0.1,
                z_score: 3.5,
            }],
            situation_anomalies: vec![],
            temporal_gap_anomalies: vec![],
        };
        persist_anomaly_report(store.as_ref(), &report).unwrap();
        let loaded = get_anomaly_report(store.as_ref(), "persist-test")
            .unwrap()
            .expect("report should exist");
        assert_eq!(loaded.entity_anomalies.len(), 1);
        assert_eq!(loaded.narrative_id, "persist-test");
    }

    #[test]
    fn test_anomaly_report_not_found() {
        let store = Arc::new(MemoryStore::new());
        let loaded = get_anomaly_report(store.as_ref(), "nonexistent").unwrap();
        assert!(loaded.is_none());
    }
}
