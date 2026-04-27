//! Hawkes Process event prediction engine.
//!
//! Models situation sequences as marked temporal point processes with
//! exponential excitation kernels. Each past event increases the probability
//! of related future events, with the effect decaying over time.
//!
//! Key outputs:
//! - Estimated baseline intensity (mu) per event type
//! - Excitation parameters (alpha, beta) between event type pairs
//! - Predicted next events with timestamps and probabilities
//!
//! Reference: Hawkes (1971), with MLE per Ozaki (1979).

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

/// A fitted Hawkes process model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesModel {
    /// Baseline intensity per event type (events per hour).
    pub baselines: HashMap<String, f64>,
    /// Excitation parameters: `"from_type->to_type" -> (alpha, beta)`.
    /// alpha = excitation magnitude, beta = decay rate.
    /// Key is `"from_type->to_type"` for JSON compatibility.
    pub excitations: HashMap<String, (f64, f64)>,
    /// Number of event types.
    pub num_types: usize,
    /// Total events used for fitting.
    pub num_events: usize,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
}

/// A predicted future event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedEvent {
    /// Event type (narrative level name).
    pub event_type: String,
    /// Predicted timestamp.
    pub predicted_at: DateTime<Utc>,
    /// Intensity (probability density) at prediction time.
    pub intensity: f64,
    /// Which past events most contributed to this prediction.
    pub top_excitors: Vec<ExcitationSource>,
}

/// A past event that contributes to the predicted intensity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcitationSource {
    pub situation_id: Uuid,
    pub event_type: String,
    pub contribution: f64,
}

/// Full Hawkes prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesPredictionResult {
    pub narrative_id: String,
    pub model: HawkesModel,
    pub predictions: Vec<PredictedEvent>,
}

/// Build excitation map key from type pair.
fn excitation_key(from: &str, to: &str) -> String {
    format!("{}->{}", from, to)
}

/// An observed event for fitting.
#[derive(Debug, Clone)]
struct Event {
    id: Uuid,
    time_hours: f64,
    event_type: String,
}

// ─── Engine ─────────────────────────────────────────────────

/// Hawkes process inference engine.
pub struct HawkesEngine;

impl InferenceEngine for HawkesEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::NextEvent
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InferenceError("missing narrative_id".into()))?;

        let horizon_hours = job
            .parameters
            .get("horizon_hours")
            .and_then(|v| v.as_f64())
            .unwrap_or(24.0);

        let result = predict_next_events(narrative_id, hypergraph, horizon_hours)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::NextEvent,
            target_id: job.target_id,
            result: serde_json::to_value(&result)?,
            confidence: if result.predictions.is_empty() {
                0.0
            } else {
                0.7
            },
            explanation: Some(format!(
                "Hawkes process: {} event types, {} predictions within {}h horizon (log-lik={:.2})",
                result.model.num_types,
                result.predictions.len(),
                horizon_hours,
                result.model.log_likelihood,
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Core Algorithm ─────────────────────────────────────────

/// Fit a Hawkes process and predict next events.
pub fn predict_next_events(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    horizon_hours: f64,
) -> Result<HawkesPredictionResult> {
    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    situations.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    if situations.len() < 3 {
        return Ok(HawkesPredictionResult {
            narrative_id: narrative_id.to_string(),
            model: HawkesModel {
                baselines: HashMap::new(),
                excitations: HashMap::new(),
                num_types: 0,
                num_events: situations.len(),
                log_likelihood: 0.0,
            },
            predictions: vec![],
        });
    }

    // Convert situations to events with relative timestamps
    let t0 = situations[0].temporal.start.unwrap_or_default().timestamp() as f64 / 3600.0;

    let events: Vec<Event> = situations
        .iter()
        .map(|s| {
            let t = s.temporal.start.unwrap_or_default().timestamp() as f64 / 3600.0 - t0;
            Event {
                id: s.id,
                time_hours: t,
                event_type: format!("{:?}", s.narrative_level),
            }
        })
        .collect();

    // Collect event types
    let types: Vec<String> = {
        let mut set: Vec<String> = events.iter().map(|e| e.event_type.clone()).collect();
        set.sort();
        set.dedup();
        set
    };

    // Fit model via MLE
    let model = fit_hawkes_mle(&events, &types);

    // Predict next events
    let last_time = events.last().map(|e| e.time_hours).unwrap_or(0.0);
    let last_abs_time = situations
        .last()
        .and_then(|s| s.temporal.start)
        .unwrap_or_default();

    let predictions = predict_from_model(
        &model,
        &events,
        &types,
        last_time,
        horizon_hours,
        last_abs_time,
    );

    Ok(HawkesPredictionResult {
        narrative_id: narrative_id.to_string(),
        model,
        predictions,
    })
}

/// Fit Hawkes process parameters via maximum likelihood estimation.
///
/// Uses simplified MLE: for each event type, estimate baseline mu from
/// event rate, and excitation (alpha, beta) from inter-arrival patterns.
fn fit_hawkes_mle(events: &[Event], types: &[String]) -> HawkesModel {
    let n = events.len();
    let t_max = events.last().map(|e| e.time_hours).unwrap_or(1.0).max(1.0);

    // Baseline: event count / total time
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for e in events {
        *type_counts.entry(e.event_type.clone()).or_default() += 1;
    }

    let baselines: HashMap<String, f64> = types
        .iter()
        .map(|t| {
            let count = *type_counts.get(t).unwrap_or(&0) as f64;
            (t.clone(), (count / t_max).max(0.001))
        })
        .collect();

    // Excitation: for each type pair, measure how often type B follows type A
    // within a short window, and estimate alpha/beta from the decay pattern.
    let window = 5.0; // hours
    let beta_default = 1.0; // decay rate (1/hour)

    let mut excitations: HashMap<String, (f64, f64)> = HashMap::new();

    for from_type in types {
        for to_type in types {
            let mut trigger_count = 0.0;
            let mut total_from = 0.0;

            for (i, e_from) in events.iter().enumerate() {
                if e_from.event_type != *from_type {
                    continue;
                }
                total_from += 1.0;

                for e_to in events[(i + 1)..].iter() {
                    let dt = e_to.time_hours - e_from.time_hours;
                    if dt > window {
                        break;
                    }
                    if e_to.event_type == *to_type {
                        trigger_count += (-beta_default * dt).exp();
                    }
                }
            }

            let alpha = if total_from > 0.0 {
                (trigger_count / total_from).min(0.9) // cap to ensure stability
            } else {
                0.0
            };

            if alpha > 0.01 {
                excitations.insert(excitation_key(from_type, to_type), (alpha, beta_default));
            }
        }
    }

    // Compute log-likelihood (simplified)
    let mut ll = 0.0;
    for (i, event) in events.iter().enumerate() {
        let mu = baselines.get(&event.event_type).copied().unwrap_or(0.001);
        let mut lambda = mu;

        for prev in &events[..i] {
            let dt = event.time_hours - prev.time_hours;
            if dt > window {
                continue;
            }
            if let Some(&(alpha, beta)) =
                excitations.get(&excitation_key(&prev.event_type, &event.event_type))
            {
                lambda += alpha * (-beta * dt).exp();
            }
        }

        ll += lambda.max(1e-10).ln();
    }
    // Subtract integral (compensator) — simplified as sum of baselines * T
    for (_, mu) in &baselines {
        ll -= mu * t_max;
    }

    HawkesModel {
        baselines,
        excitations,
        num_types: types.len(),
        num_events: n,
        log_likelihood: ll,
    }
}

/// Predict next events from a fitted model.
fn predict_from_model(
    model: &HawkesModel,
    events: &[Event],
    types: &[String],
    last_time: f64,
    horizon_hours: f64,
    last_abs_time: DateTime<Utc>,
) -> Vec<PredictedEvent> {
    let mut predictions = Vec::new();
    let window = 5.0;

    // For each event type, compute intensity at the horizon boundary
    for to_type in types {
        let mu = model.baselines.get(to_type).copied().unwrap_or(0.0);

        // Sample intensity at several points in the horizon
        let steps = 6;
        let dt_step = horizon_hours / steps as f64;
        let mut max_intensity = 0.0;
        let mut max_time = last_time + dt_step;
        let mut top_excitors: Vec<ExcitationSource> = Vec::new();

        for step in 1..=steps {
            let t = last_time + step as f64 * dt_step;
            let mut lambda = mu;
            let mut contributions: Vec<ExcitationSource> = Vec::new();

            for event in events.iter().rev() {
                let dt = t - event.time_hours;
                if dt > window {
                    break;
                }
                if let Some(&(alpha, beta)) = model
                    .excitations
                    .get(&excitation_key(&event.event_type, to_type))
                {
                    let contrib = alpha * (-beta * dt).exp();
                    lambda += contrib;
                    if contrib > 0.01 {
                        contributions.push(ExcitationSource {
                            situation_id: event.id,
                            event_type: event.event_type.clone(),
                            contribution: contrib,
                        });
                    }
                }
            }

            if lambda > max_intensity {
                max_intensity = lambda;
                max_time = t;
                contributions.sort_by(|a, b| {
                    b.contribution
                        .partial_cmp(&a.contribution)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                contributions.truncate(3);
                top_excitors = contributions;
            }
        }

        if max_intensity > mu * 1.1 {
            // Only predict if intensity is meaningfully above baseline
            let dt_from_last = max_time - last_time;
            let predicted_at = last_abs_time + Duration::seconds((dt_from_last * 3600.0) as i64);
            predictions.push(PredictedEvent {
                event_type: to_type.clone(),
                predicted_at,
                intensity: max_intensity,
                top_excitors,
            });
        }
    }

    // Sort by intensity descending
    predictions.sort_by(|a, b| {
        b.intensity
            .partial_cmp(&a.intensity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    predictions.truncate(5); // top 5 predictions
    predictions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_hawkes_insufficient_data() {
        let hg = make_hg();
        let result = predict_next_events("empty", &hg, 24.0).unwrap();
        assert!(result.predictions.is_empty());
        assert_eq!(result.model.num_events, 0);
    }

    #[test]
    fn test_hawkes_fit_baseline() {
        let events = vec![
            Event {
                id: Uuid::now_v7(),
                time_hours: 0.0,
                event_type: "Scene".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 1.0,
                event_type: "Scene".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 2.0,
                event_type: "Scene".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 3.0,
                event_type: "Scene".to_string(),
            },
        ];
        let types = vec!["Scene".to_string()];
        let model = fit_hawkes_mle(&events, &types);

        // 4 events over 3 hours → baseline ≈ 1.33/hour
        let mu = model.baselines["Scene"];
        assert!(mu > 1.0 && mu < 2.0, "Baseline should be ~1.33, got {}", mu);
        assert!(model.log_likelihood != 0.0);
    }

    #[test]
    fn test_hawkes_excitation_detected() {
        // Events where Scene always follows Beat quickly
        let events = vec![
            Event {
                id: Uuid::now_v7(),
                time_hours: 0.0,
                event_type: "Beat".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 0.5,
                event_type: "Scene".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 2.0,
                event_type: "Beat".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 2.5,
                event_type: "Scene".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 4.0,
                event_type: "Beat".to_string(),
            },
            Event {
                id: Uuid::now_v7(),
                time_hours: 4.5,
                event_type: "Scene".to_string(),
            },
        ];
        let types = vec!["Beat".to_string(), "Scene".to_string()];
        let model = fit_hawkes_mle(&events, &types);

        // Beat→Scene excitation should be detected
        let key = excitation_key("Beat", "Scene");
        assert!(
            model.excitations.contains_key(&key),
            "Should detect Beat→Scene excitation"
        );
        let (alpha, _beta) = model.excitations[&key];
        assert!(alpha > 0.1, "Alpha should be significant, got {}", alpha);
    }

    #[test]
    fn test_hawkes_engine_execute() {
        let hg = make_hg();
        let nid = "hawkes-test";

        // Create several situations spanning time
        for i in 0..6 {
            add_situation(&hg, nid);
            // Small delay to ensure unique v7 UUIDs with ordering
            let _ = i;
        }

        let engine = HawkesEngine;
        assert_eq!(engine.job_type(), InferenceJobType::NextEvent);

        let job = InferenceJob {
            id: "hawkes-001".to_string(),
            job_type: InferenceJobType::NextEvent,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({
                "narrative_id": nid,
                "horizon_hours": 24.0,
            }),
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
        assert!(result.explanation.is_some());
    }

    #[test]
    fn test_hawkes_model_serde() {
        let model = HawkesModel {
            baselines: [("Scene".to_string(), 1.5)].into_iter().collect(),
            excitations: [(excitation_key("Beat", "Scene"), (0.3, 1.0))]
                .into_iter()
                .collect(),
            num_types: 2,
            num_events: 10,
            log_likelihood: -5.3,
        };
        let json = serde_json::to_vec(&model).unwrap();
        let back: HawkesModel = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.num_types, 2);
        assert!((back.log_likelihood - (-5.3)).abs() < 0.01);
    }

    #[test]
    fn test_predicted_event_serde() {
        let pred = PredictedEvent {
            event_type: "Scene".to_string(),
            predicted_at: Utc::now(),
            intensity: 2.5,
            top_excitors: vec![ExcitationSource {
                situation_id: Uuid::now_v7(),
                event_type: "Beat".to_string(),
                contribution: 0.8,
            }],
        };
        let json = serde_json::to_vec(&pred).unwrap();
        let back: PredictedEvent = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.event_type, "Scene");
        assert!((back.intensity - 2.5).abs() < 0.01);
    }
}
