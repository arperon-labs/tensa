//! Faction Evolution: track community assignments across temporal windows.
//!
//! Divides a narrative's time span into sliding windows, runs community detection
//! on each, and detects formation, merges, splits, and dissolution events.
//! Results stored at `an/fe/{narrative_id}`.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::community_detect::label_propagation;
use crate::analysis::graph_projection;
use crate::analysis::{analysis_key, extract_narrative_id, make_engine_result};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult};

/// A community snapshot at a specific time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSnapshot {
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    /// Entity ID → community label at this window.
    pub assignments: Vec<(Uuid, usize)>,
    pub num_communities: usize,
}

/// A faction evolution event (formation, merge, split, dissolution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionEvent {
    pub event_type: FactionEventType,
    pub window_index: usize,
    pub entities: Vec<Uuid>,
    pub from_community: Option<usize>,
    pub to_community: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FactionEventType {
    Formation,
    Dissolution,
    Merge,
    Split,
}

/// Full faction evolution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionEvolution {
    pub narrative_id: String,
    pub windows: Vec<WindowSnapshot>,
    pub events: Vec<FactionEvent>,
    pub num_windows: usize,
}

/// Run faction evolution analysis with the given number of windows.
pub fn faction_evolution(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    num_windows: usize,
) -> Result<FactionEvolution> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(FactionEvolution {
            narrative_id: narrative_id.to_string(),
            windows: vec![],
            events: vec![],
            num_windows: 0,
        });
    }

    // Find time span
    let mut min_t = DateTime::<Utc>::MAX_UTC;
    let mut max_t = DateTime::<Utc>::MIN_UTC;
    for s in &situations {
        if let Some(t) = s.temporal.start {
            if t < min_t {
                min_t = t;
            }
            if t > max_t {
                max_t = t;
            }
        }
    }

    if min_t >= max_t {
        // Single point in time — one window
        let graph = graph_projection::build_co_graph(hypergraph, narrative_id)?;
        let labels = label_propagation(&graph, 42);
        let num_c = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let assignments: Vec<(Uuid, usize)> = graph
            .entities
            .iter()
            .zip(labels.iter())
            .map(|(&eid, &l)| (eid, l))
            .collect();
        return Ok(FactionEvolution {
            narrative_id: narrative_id.to_string(),
            windows: vec![WindowSnapshot {
                window_start: min_t,
                window_end: max_t,
                assignments,
                num_communities: num_c,
            }],
            events: vec![],
            num_windows: 1,
        });
    }

    let total_duration = (max_t - min_t).num_seconds().max(1) as f64;
    let window_secs = (total_duration / num_windows as f64) as i64;
    let window_dur = Duration::seconds(window_secs.max(1));

    let mut windows = Vec::new();
    let mut prev_labels: Option<(Vec<Uuid>, Vec<usize>)> = None;
    let mut events = Vec::new();

    for i in 0..num_windows {
        let w_start = min_t + window_dur * i as i32;
        let w_end = if i == num_windows - 1 {
            max_t + Duration::seconds(1)
        } else {
            w_start + window_dur
        };

        let graph = graph_projection::build_temporal_graph(
            hypergraph,
            narrative_id,
            Some((w_start, w_end)),
        )?;

        if graph.entities.is_empty() {
            continue;
        }

        let labels = label_propagation(&graph, 42 + i as u64);
        let num_c = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        let assignments: Vec<(Uuid, usize)> = graph
            .entities
            .iter()
            .zip(labels.iter())
            .map(|(&eid, &l)| (eid, l))
            .collect();

        // Detect events by comparing with previous window
        if let Some((prev_ents, prev_labels_vec)) = &prev_labels {
            let prev_map: std::collections::HashMap<Uuid, usize> = prev_ents
                .iter()
                .zip(prev_labels_vec.iter())
                .map(|(&e, &l)| (e, l))
                .collect();
            let cur_map: std::collections::HashMap<Uuid, usize> = graph
                .entities
                .iter()
                .zip(labels.iter())
                .map(|(&e, &l)| (e, l))
                .collect();

            // New entities not in previous window → formation
            for (&eid, &label) in &cur_map {
                if !prev_map.contains_key(&eid) {
                    events.push(FactionEvent {
                        event_type: FactionEventType::Formation,
                        window_index: i,
                        entities: vec![eid],
                        from_community: None,
                        to_community: Some(label),
                    });
                }
            }

            // Entities in previous but not current → dissolution
            for (&eid, &label) in &prev_map {
                if !cur_map.contains_key(&eid) {
                    events.push(FactionEvent {
                        event_type: FactionEventType::Dissolution,
                        window_index: i,
                        entities: vec![eid],
                        from_community: Some(label),
                        to_community: None,
                    });
                }
            }

            // Detect merges/splits: entities that change community
            let mut community_transitions: std::collections::HashMap<(usize, usize), Vec<Uuid>> =
                std::collections::HashMap::new();
            for (&eid, &cur_label) in &cur_map {
                if let Some(&prev_label) = prev_map.get(&eid) {
                    if prev_label != cur_label {
                        community_transitions
                            .entry((prev_label, cur_label))
                            .or_default()
                            .push(eid);
                    }
                }
            }
            for ((from, to), ents) in community_transitions {
                if ents.len() > 1 {
                    events.push(FactionEvent {
                        event_type: FactionEventType::Merge,
                        window_index: i,
                        entities: ents,
                        from_community: Some(from),
                        to_community: Some(to),
                    });
                } else {
                    events.push(FactionEvent {
                        event_type: FactionEventType::Split,
                        window_index: i,
                        entities: ents,
                        from_community: Some(from),
                        to_community: Some(to),
                    });
                }
            }
        }

        prev_labels = Some((graph.entities.clone(), labels));
        windows.push(WindowSnapshot {
            window_start: w_start,
            window_end: w_end,
            assignments,
            num_communities: num_c,
        });
    }

    Ok(FactionEvolution {
        narrative_id: narrative_id.to_string(),
        num_windows: windows.len(),
        windows,
        events,
    })
}

/// Faction Evolution engine.
pub struct FactionEvolutionEngine;

impl InferenceEngine for FactionEvolutionEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::FactionEvolution
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(8000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let num_windows = job
            .parameters
            .get("num_windows")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let result = faction_evolution(hypergraph, narrative_id, num_windows)?;

        let key = analysis_key(b"an/fe/", &[narrative_id]);
        let bytes = serde_json::to_vec(&result)?;
        hypergraph.store().put(&key, &bytes)?;

        Ok(make_engine_result(
            job,
            InferenceJobType::FactionEvolution,
            narrative_id,
            vec![serde_json::json!({
                "num_windows": result.num_windows,
                "num_events": result.events.len(),
            })],
            "Faction evolution analysis complete",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_faction_stable() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
        let result = faction_evolution(&hg, "n1", 3).unwrap();
        assert!(result.windows.len() >= 1);
    }

    #[test]
    fn test_faction_empty() {
        let hg = make_hg();
        let result = faction_evolution(&hg, "n1", 5).unwrap();
        assert!(result.windows.is_empty());
        assert!(result.events.is_empty());
    }

    #[test]
    fn test_faction_evolution_events() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
        let result = faction_evolution(&hg, "n1", 2).unwrap();
        // With 2 windows on a single-timepoint narrative, should produce 1 window
        assert!(result.num_windows >= 1);
    }
}
