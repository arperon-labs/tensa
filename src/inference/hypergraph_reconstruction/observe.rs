// Loop variables that index time bins read more clearly explicitly.
#![allow(clippy::needless_range_loop)]

//! Observation functions: turn a (hypergraph, narrative) into a state matrix.
//!
//! Returns `X[T × N]` as a row-major `Vec<Vec<f32>>` plus the time-bin
//! anchors. One function per [`ObservationSource`] variant; the dispatcher
//! lives at the bottom.

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use crate::analysis::graph_projection::collect_participation_index;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

use super::types::ObservationSource;

/// Output of [`build_state_matrix`].
#[derive(Debug, Clone)]
pub struct ObservationMatrix {
    /// `X[t][i]` — entity `i`'s observation at bin `t`. Row-major.
    pub x: Vec<Vec<f32>>,
    /// Anchor for bin `t`: the right-edge timestamp of the window covering
    /// `[anchor - window, anchor]`. `len(time_axis) == x.len()`.
    pub time_axis: Vec<DateTime<Utc>>,
    /// Entity UUIDs in column order (matches `entity_idx`).
    pub entities: Vec<Uuid>,
}

/// Build the observation matrix for a narrative.
///
/// `entity_idx` maps each entity UUID to its column index. `entities` is
/// the matching column-ordered UUID list.
///
/// Returns an `InferenceError` when the time range is too short, the
/// observation source is unimplemented, or all entries would be zero.
pub fn build_state_matrix(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entities: Vec<Uuid>,
    entity_idx: &HashMap<Uuid, usize>,
    window_seconds: i64,
    time_resolution_seconds: i64,
    source: &ObservationSource,
) -> Result<ObservationMatrix> {
    if window_seconds <= 0 {
        return Err(TensaError::InvalidInput(
            "window_seconds must be > 0".into(),
        ));
    }
    if time_resolution_seconds <= 0 {
        return Err(TensaError::InvalidInput(
            "time_resolution_seconds must be > 0".into(),
        ));
    }

    match source {
        ObservationSource::ParticipationRate => build_participation_rate(
            hypergraph,
            narrative_id,
            entities,
            entity_idx,
            window_seconds,
            time_resolution_seconds,
        ),
        ObservationSource::SentimentMean => Err(TensaError::InferenceError(
            "ObservationSource::SentimentMean: PrerequisiteMissing — \
             requires `Situation.properties[\"sentiment\"]` populated. \
             Use ParticipationRate or run the sentiment-extraction pass first."
                .into(),
        )),
        ObservationSource::BeliefMass { proposition } => Err(TensaError::InferenceError(format!(
            "ObservationSource::BeliefMass{{proposition: {proposition:?}}}: \
             PrerequisiteMissing — requires `an/ev/` evidence keys for the \
             proposition. Run INFER EVIDENCE first."
        ))),
        ObservationSource::Engagement => Err(TensaError::InferenceError(
            "ObservationSource::Engagement is not yet implemented in MVP — \
             multi-dimensional state requires pipeline changes deferred to \
             Phase 15c. Use ParticipationRate."
                .into(),
        )),
    }
}

/// Build the participation-rate state matrix.
///
/// For each time bin `t` (right-edge anchor) and each entity `i`, count the
/// situations entity `i` participated in whose `temporal.start` falls in
/// `(anchor - window, anchor]`. Normalize by the window length so the value
/// is "situations per second."
///
/// Pipeline N+1 guard: `collect_participation_index` is called ONCE here;
/// neither bootstrap nor any downstream stage re-invokes it.
fn build_participation_rate(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entities: Vec<Uuid>,
    entity_idx: &HashMap<Uuid, usize>,
    window_seconds: i64,
    time_resolution_seconds: i64,
) -> Result<ObservationMatrix> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Err(TensaError::InferenceError(format!(
            "build_state_matrix: narrative '{narrative_id}' has no situations"
        )));
    }

    // Resolve start timestamps once (skip situations missing a start).
    let mut sit_starts: Vec<(Uuid, DateTime<Utc>)> = situations
        .iter()
        .filter_map(|s| s.temporal.start.map(|ts| (s.id, ts)))
        .collect();
    if sit_starts.is_empty() {
        return Err(TensaError::InferenceError(format!(
            "build_state_matrix: narrative '{narrative_id}' has no situations \
             with a populated temporal.start"
        )));
    }
    sit_starts.sort_by_key(|(_, ts)| *ts);
    let t_start = sit_starts.first().map(|(_, ts)| *ts).unwrap();
    let t_end = sit_starts.last().map(|(_, ts)| *ts).unwrap();
    let span = (t_end - t_start).num_seconds();
    if span < time_resolution_seconds {
        return Err(TensaError::InferenceError(format!(
            "build_state_matrix: time range ({span}s) is shorter than \
             time_resolution_seconds ({time_resolution_seconds}s)"
        )));
    }

    let n_bins = ((span / time_resolution_seconds) + 1).max(2) as usize;
    let n = entities.len();

    // Build the participation index ONCE — the architectural N+1 guard.
    let sit_ids: Vec<Uuid> = sit_starts.iter().map(|(id, _)| *id).collect();
    let sit_to_entities = collect_participation_index(hypergraph, entity_idx, &sit_ids, None)?;

    // Pre-resolve a per-situation (entity_columns, start) tuple for the binning loop.
    let sit_meta: Vec<(Vec<usize>, DateTime<Utc>)> = sit_starts
        .into_iter()
        .filter_map(|(sid, start)| sit_to_entities.get(&sid).map(|ents| (ents.clone(), start)))
        .collect();

    let window = Duration::seconds(window_seconds);
    let mut x: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; n_bins];
    let mut time_axis: Vec<DateTime<Utc>> = Vec::with_capacity(n_bins);
    for bin in 0..n_bins {
        let anchor = t_start + Duration::seconds(time_resolution_seconds * bin as i64);
        time_axis.push(anchor);
        let lo = anchor - window;
        for (ents, start) in &sit_meta {
            if *start <= anchor && *start > lo {
                for &col in ents {
                    if col < n {
                        x[bin][col] += 1.0;
                    }
                }
            }
        }
    }

    // Normalize by window length (situations per second).
    let inv_window = 1.0_f32 / window_seconds as f32;
    for row in &mut x {
        for cell in row.iter_mut() {
            *cell *= inv_window;
        }
    }

    // Guard: all-zero matrix indicates no observable activity.
    let any_nonzero = x.iter().any(|row| row.iter().any(|&v| v > 0.0));
    if !any_nonzero {
        return Err(TensaError::InferenceError(
            "build_state_matrix: all-zero state matrix — no entities had activity \
             in any time bin. Try a larger window_seconds or verify ingestion."
                .into(),
        ));
    }

    Ok(ObservationMatrix {
        x,
        time_axis,
        entities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use std::sync::Arc;

    fn make_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn add_actor(hg: &Hypergraph, narrative: &str, name: &str) -> Uuid {
        let e = Entity {
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
        hg.create_entity(e).unwrap()
    }

    fn add_situation_with_participants(
        hg: &Hypergraph,
        narrative: &str,
        start: DateTime<Utc>,
        members: &[Uuid],
    ) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            name: None,
            description: None,
            properties: serde_json::Value::Null,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(start + Duration::seconds(1)),
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
        let sid = hg.create_situation(sit).unwrap();
        for &m in members {
            hg.add_participant(Participation {
                entity_id: m,
                situation_id: sid,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
        sid
    }

    #[test]
    fn test_participation_rate_basic_shape() {
        let hg = make_hg();
        let narrative = "obs-test-1";
        let a = add_actor(&hg, narrative, "Alice");
        let b = add_actor(&hg, narrative, "Bob");
        let entities = vec![a, b];
        let mut entity_idx = HashMap::new();
        entity_idx.insert(a, 0);
        entity_idx.insert(b, 1);

        let base = Utc::now() - Duration::seconds(120);
        for i in 0..5 {
            let when = base + Duration::seconds(i * 30);
            add_situation_with_participants(&hg, narrative, when, &[a, b]);
        }

        let mat = build_state_matrix(
            &hg,
            narrative,
            entities.clone(),
            &entity_idx,
            60,
            30,
            &ObservationSource::ParticipationRate,
        )
        .expect("build_state_matrix should succeed");

        assert_eq!(mat.entities, entities);
        assert!(mat.x.len() >= 2, "must have at least 2 time bins");
        assert!(mat.time_axis.len() == mat.x.len());
        assert!(mat.x.iter().any(|row| row.iter().any(|&v| v > 0.0)));
    }

    #[test]
    fn test_engagement_returns_inference_error() {
        let hg = make_hg();
        let narrative = "obs-stub";
        let a = add_actor(&hg, narrative, "A");
        let entities = vec![a];
        let mut entity_idx = HashMap::new();
        entity_idx.insert(a, 0);
        // Need at least one situation so we don't bail at the empty-narrative guard.
        let base = Utc::now();
        add_situation_with_participants(&hg, narrative, base, &[a]);
        add_situation_with_participants(&hg, narrative, base + Duration::seconds(60), &[a]);

        let err = build_state_matrix(
            &hg,
            narrative,
            entities,
            &entity_idx,
            60,
            60,
            &ObservationSource::Engagement,
        )
        .expect_err("Engagement must error in MVP");
        match err {
            TensaError::InferenceError(msg) => assert!(msg.contains("Engagement")),
            other => panic!("expected InferenceError, got {other:?}"),
        }
    }

    #[test]
    fn test_belief_mass_includes_proposition_in_message() {
        let hg = make_hg();
        let narrative = "obs-bm";
        let a = add_actor(&hg, narrative, "A");
        let mut entity_idx = HashMap::new();
        entity_idx.insert(a, 0);
        let base = Utc::now();
        add_situation_with_participants(&hg, narrative, base, &[a]);
        add_situation_with_participants(&hg, narrative, base + Duration::seconds(120), &[a]);

        let err = build_state_matrix(
            &hg,
            narrative,
            vec![a],
            &entity_idx,
            60,
            60,
            &ObservationSource::BeliefMass {
                proposition: "raven_is_dead".into(),
            },
        )
        .expect_err("BeliefMass needs prerequisite data");
        match err {
            TensaError::InferenceError(msg) => assert!(msg.contains("raven_is_dead")),
            other => panic!("expected InferenceError, got {other:?}"),
        }
    }
}
