//! Temporal Motif Census: detect recurring temporal patterns with Allen relation constraints.
//!
//! Enumerates 3-node and 4-node motifs where edges carry Allen temporal relations.
//! A motif is a (node_types, edge_label_sequence, temporal_relation_sequence) triple.
//! Store census results at `an/tm/{narrative_id}`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::analysis::{analysis_key, extract_narrative_id, make_engine_result};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::temporal::interval::relation_between;
use crate::types::*;

/// A temporal motif: a recurring pattern of entity co-participations with Allen relations.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct TemporalMotif {
    /// Number of entities in the motif.
    pub size: usize,
    /// Sequence of Allen relations between consecutive situations.
    pub temporal_relations: Vec<AllenRelation>,
    /// Narrative levels of the situations involved.
    pub situation_levels: Vec<NarrativeLevel>,
}

/// Census result: motif → occurrence count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifCensus {
    pub narrative_id: String,
    pub motifs: Vec<(TemporalMotif, usize)>,
    pub total_motifs_found: usize,
}

/// Run temporal motif census on a narrative.
///
/// Enumerates 3-node motifs: for each entity, take consecutive situation triples
/// from their participation sequence and record the Allen relations between them.
pub fn temporal_motif_census(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    max_motif_size: usize,
) -> Result<MotifCensus> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let mut motif_counts: HashMap<TemporalMotif, usize> = HashMap::new();

    for entity in &entities {
        let participations = hypergraph.get_situations_for_entity(&entity.id)?;
        if participations.len() < 2 {
            continue;
        }

        // Get situations with their temporal data, sorted by start time
        let mut sit_data: Vec<(Situation, AllenInterval)> = Vec::new();
        for p in &participations {
            if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
                let interval = sit.temporal.clone();
                sit_data.push((sit, interval));
            }
        }
        sit_data.sort_by(|a, b| a.1.start.cmp(&b.1.start));

        // 3-node motifs: sliding window of 3 consecutive situations
        let motif_size = max_motif_size.min(4).max(3);
        for window_size in 3..=motif_size {
            if sit_data.len() < window_size {
                continue;
            }
            for w in sit_data.windows(window_size) {
                let mut relations = Vec::new();
                let mut levels = Vec::new();
                let mut valid = true;

                for i in 0..w.len() - 1 {
                    match relation_between(&w[i].1, &w[i + 1].1) {
                        Ok(rel) => relations.push(rel),
                        Err(_) => {
                            valid = false;
                            break;
                        }
                    }
                }

                if !valid {
                    continue;
                }

                for (sit, _) in w {
                    levels.push(sit.narrative_level);
                }

                let motif = TemporalMotif {
                    size: window_size,
                    temporal_relations: relations,
                    situation_levels: levels,
                };
                *motif_counts.entry(motif).or_insert(0) += 1;
            }
        }
    }

    let total = motif_counts.values().sum();
    let mut motifs: Vec<(TemporalMotif, usize)> = motif_counts.into_iter().collect();
    motifs.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by frequency descending

    Ok(MotifCensus {
        narrative_id: narrative_id.to_string(),
        motifs,
        total_motifs_found: total,
    })
}

/// Temporal Motif Census engine.
pub struct TemporalMotifEngine;

impl InferenceEngine for TemporalMotifEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::TemporalMotifs
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(6000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let max_size = job
            .parameters
            .get("max_motif_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let census = temporal_motif_census(hypergraph, narrative_id, max_size)?;

        let key = analysis_key(b"an/tm/", &[narrative_id]);
        let bytes = serde_json::to_vec(&census)?;
        hypergraph.store().put(&key, &bytes)?;

        Ok(make_engine_result(
            job,
            InferenceJobType::TemporalMotifs,
            narrative_id,
            vec![serde_json::json!({
                "total_motifs": census.total_motifs_found,
                "unique_motifs": census.motifs.len(),
            })],
            "Temporal motif census complete",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_temporal_motif_3node() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let s1 = add_situation(&hg, "n1");
        let s2 = add_situation(&hg, "n1");
        let s3 = add_situation(&hg, "n1");
        link(&hg, a, s1);
        link(&hg, a, s2);
        link(&hg, a, s3);
        let census = temporal_motif_census(&hg, "n1", 3).unwrap();
        // Entity A participates in 3 situations → at least 1 motif
        assert!(
            census.total_motifs_found >= 1,
            "Should find at least 1 motif"
        );
    }

    #[test]
    fn test_motif_census_counts() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let b = add_entity(&hg, "B", "n1");
        for _ in 0..4 {
            let s = add_situation(&hg, "n1");
            link(&hg, a, s);
            link(&hg, b, s);
        }
        let census = temporal_motif_census(&hg, "n1", 3).unwrap();
        assert!(
            census.total_motifs_found >= 2,
            "4 situations → multiple motifs"
        );
    }

    #[test]
    fn test_motif_empty() {
        let hg = make_hg();
        let census = temporal_motif_census(&hg, "n1", 3).unwrap();
        assert_eq!(census.total_motifs_found, 0);
    }

    #[test]
    fn test_motif_single_entity() {
        let hg = make_hg();
        let a = add_entity(&hg, "A", "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        let census = temporal_motif_census(&hg, "n1", 3).unwrap();
        // Only 1 situation → no motifs (need ≥3)
        assert_eq!(census.total_motifs_found, 0);
    }
}
