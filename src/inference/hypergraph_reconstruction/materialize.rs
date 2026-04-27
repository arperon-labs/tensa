//! Phase 15c — opt-in materialization of inferred hyperedges as
//! [`Situation`] records.
//!
//! Inferred hyperedges live in the [`ReconstructionResult`] blob produced by
//! the reconstruction engine. They are NOT automatically committed to the
//! hypergraph — that would conflate "observed" structure with "inferred"
//! structure, breaking provenance. Phase 15c adds an explicit, opt-in
//! materialization step:
//!
//! ```text
//! POST /inference/hypergraph-reconstruction/{job_id}/materialize
//!     body: { output_narrative_id, opt_in: true }
//! ```
//!
//! For each [`InferredHyperedge`] with `confidence > threshold` (default
//! `0.7` per architect §13.7 of `docs/synth_reconstruction_algorithm.md`) we:
//!
//! 1. Create a [`Situation`] under `output_narrative_id` with
//!    `extraction_method = ExtractionMethod::Reconstructed { source_narrative_id, job_id }`
//!    (variant introduced in Phase 15b).
//! 2. Add every member entity of the hyperedge as a [`Participation`] with
//!    `Role::SubjectOfDiscussion` (neutral catch-all — coordination
//!    membership has no narrative role).
//! 3. Persist a `ReconstructedSituationRef` under
//!    `syn/recon/{output_narrative_id}/{job_id}/{situation_id}` so consumers
//!    can list every situation produced by a given reconstruction job
//!    without scanning the entire situations table.
//!
//! The Situation's `confidence` field carries the bootstrap confidence from
//! the inferred edge directly — analysts can re-filter the materialized
//! corpus on the same axis they filtered the inference on.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::synth::key_synth_recon_situation;
use crate::types::{
    AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Participation,
    Role, Situation, TimeGranularity,
};

use super::types::{InferredHyperedge, ReconstructionResult};

/// Default minimum confidence for materialization. Per architect §13.7 — the
/// analyst workflow is to filter by confidence > 0.7 because Taylor-expansion
/// masking artifacts can clear the weight threshold but rarely clear the
/// bootstrap-stability threshold.
pub const DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Persisted KV record at `syn/recon/{output_narrative_id}/{job_id}/{situation_id}`.
/// One row per materialized situation; lets consumers enumerate everything a
/// given reconstruction job produced without filtering the global situations
/// table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReconstructedSituationRef {
    pub situation_id: Uuid,
    pub source_narrative_id: String,
    pub output_narrative_id: String,
    pub job_id: String,
    pub members: Vec<Uuid>,
    pub order: u8,
    pub weight: f32,
    pub confidence: f32,
    pub possible_masking_artifact: bool,
}

/// Outcome of a materialization request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterializationReport {
    pub output_narrative_id: String,
    pub job_id: String,
    pub situations_created: usize,
    pub situations_skipped: usize,
    pub confidence_threshold: f32,
}

/// Materialize every [`InferredHyperedge`] whose `confidence > threshold` as
/// a [`Situation`] under `output_narrative_id`. Returns a
/// [`MaterializationReport`] summarizing the outcome.
///
/// `confidence_threshold` defaults to [`DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD`]
/// when caller passes `None`.
pub fn materialize_reconstruction(
    hypergraph: &Hypergraph,
    result: &ReconstructionResult,
    source_narrative_id: &str,
    output_narrative_id: &str,
    job_id: &str,
    confidence_threshold: Option<f32>,
) -> Result<MaterializationReport> {
    if output_narrative_id.is_empty() {
        return Err(TensaError::InvalidInput(
            "materialize_reconstruction: output_narrative_id is empty".into(),
        ));
    }
    if job_id.is_empty() {
        return Err(TensaError::InvalidInput(
            "materialize_reconstruction: job_id is empty".into(),
        ));
    }
    let threshold = confidence_threshold
        .unwrap_or(DEFAULT_MATERIALIZE_CONFIDENCE_THRESHOLD)
        .clamp(0.0, 1.0);

    let mut created = 0usize;
    let mut skipped = 0usize;
    let store = hypergraph.store();

    for edge in &result.inferred_edges {
        if edge.confidence <= threshold {
            skipped += 1;
            continue;
        }
        if edge.members.len() < 2 {
            skipped += 1;
            continue;
        }
        let situation_id = create_situation_for_edge(
            hypergraph,
            edge,
            source_narrative_id,
            output_narrative_id,
            job_id,
        )?;
        // Persist per-job ref so callers can enumerate without scanning the
        // global situations table.
        let key = key_synth_recon_situation(output_narrative_id, job_id, &situation_id);
        let ref_record = ReconstructedSituationRef {
            situation_id,
            source_narrative_id: source_narrative_id.to_string(),
            output_narrative_id: output_narrative_id.to_string(),
            job_id: job_id.to_string(),
            members: edge.members.clone(),
            order: edge.order,
            weight: edge.weight,
            confidence: edge.confidence,
            possible_masking_artifact: edge.possible_masking_artifact,
        };
        let bytes = serde_json::to_vec(&ref_record)?;
        store.put(&key, &bytes)?;
        created += 1;
    }

    Ok(MaterializationReport {
        output_narrative_id: output_narrative_id.to_string(),
        job_id: job_id.to_string(),
        situations_created: created,
        situations_skipped: skipped,
        confidence_threshold: threshold,
    })
}

/// Build + commit one situation + member participations for a single inferred
/// edge. Members must already exist as [`Entity`] records in the hypergraph
/// (they came from the source narrative the engine read).
fn create_situation_for_edge(
    hypergraph: &Hypergraph,
    edge: &InferredHyperedge,
    source_narrative_id: &str,
    output_narrative_id: &str,
    job_id: &str,
) -> Result<Uuid> {
    let now = Utc::now();
    let summary = format!(
        "Inferred hyperedge order={} weight={:.4} confidence={:.4}",
        edge.order, edge.weight, edge.confidence
    );
    let situation = Situation {
        id: Uuid::now_v7(),
        name: Some(format!(
            "Reconstructed coordination edge ({} members)",
            edge.members.len()
        )),
        description: Some(summary.clone()),
        properties: serde_json::json!({
            "reconstructed": true,
            "source_narrative_id": source_narrative_id,
            "job_id": job_id,
            "weight": edge.weight,
            "bootstrap_confidence": edge.confidence,
            "possible_masking_artifact": edge.possible_masking_artifact,
        }),
        temporal: AllenInterval {
            start: None,
            end: None,
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
        raw_content: vec![ContentBlock::text(&summary)],
        narrative_level: NarrativeLevel::Event,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: edge.confidence,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::Reconstructed {
            source_narrative_id: source_narrative_id.to_string(),
            job_id: job_id.to_string(),
        },
        provenance: vec![],
        narrative_id: Some(output_narrative_id.to_string()),
        source_chunk_id: None,
        source_span: None,
        synopsis: None,
        manuscript_order: None,
        parent_situation_id: None,
        label: None,
        status: None,
        keywords: vec![],
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    };
    let situation_id = hypergraph.create_situation(situation)?;

    for member in &edge.members {
        // Tolerate "entity not found" — the source narrative may have
        // pruned an entity between reconstruction and materialization. Skip
        // missing members rather than abort the whole materialization.
        if hypergraph.get_entity(member).is_err() {
            continue;
        }
        let participation = Participation {
            entity_id: *member,
            situation_id,
            role: Role::SubjectOfDiscussion,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        };
        // add_participant ignores caller-supplied `seq`, so the value is fine.
        let _ = hypergraph.add_participant(participation);
    }
    Ok(situation_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::synth::SYN_RECON_PREFIX;
    use crate::types::{Entity, EntityType};
    use std::sync::Arc;

    fn fresh_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_entity(narrative_id: &str) -> Uuid {
        let hg = fresh_hg();
        let id = Uuid::now_v7();
        let entity = Entity {
            id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({}),
            beliefs: None,
            embedding: None,
            narrative_id: Some(narrative_id.into()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::Sensor),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap()
    }

    /// Helper for tests in the same module: create N entities in the given
    /// hypergraph + narrative, return their UUIDs.
    fn seed_entities(hg: &Hypergraph, narrative_id: &str, n: usize) -> Vec<Uuid> {
        (0..n)
            .map(|_| {
                let id = Uuid::now_v7();
                let entity = Entity {
                    id,
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    beliefs: None,
                    embedding: None,
                    narrative_id: Some(narrative_id.into()),
                    maturity: MaturityLevel::Candidate,
                    confidence: 0.9,
                    confidence_breakdown: None,
                    provenance: vec![],
                    extraction_method: Some(ExtractionMethod::Sensor),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    deleted_at: None,
                    transaction_time: None,
                };
                hg.create_entity(entity).unwrap()
            })
            .collect()
    }

    fn dummy_result(members: Vec<Uuid>, confidence: f32) -> ReconstructionResult {
        use super::super::types::{MatrixStats, ObservationSource, ReconstructionParams};
        ReconstructionResult {
            inferred_edges: vec![InferredHyperedge {
                members: members.clone(),
                order: members.len() as u8,
                weight: 0.42,
                confidence,
                possible_masking_artifact: false,
            }],
            coefficient_matrix_stats: MatrixStats {
                n_entities: members.len(),
                n_library_terms: 1,
                n_timesteps: 10,
                sparsity: 0.5,
                condition_number_approx: 1.0,
                lambda_used: 0.05,
                pearson_filtered_pairs: 0,
            },
            goodness_of_fit: 0.9,
            observation_source: ObservationSource::ParticipationRate,
            params_used: ReconstructionParams::default(),
            time_range: (Utc::now(), Utc::now()),
            bootstrap_resamples_completed: 10,
            warnings: vec![],
        }
    }

    /// T4 — Materialize endpoint creates Situations with
    /// ExtractionMethod::Reconstructed { source_narrative_id, job_id }.
    #[test]
    fn test_reconstruction_extraction_method_variant_threaded_through() {
        let hg = fresh_hg();
        let members = seed_entities(&hg, "src", 3);
        let result = dummy_result(members.clone(), 0.95);
        let report = materialize_reconstruction(
            &hg,
            &result,
            "src",
            "out",
            "job-abc",
            None,
        )
        .unwrap();
        assert_eq!(report.situations_created, 1);

        // Find the materialized situation and check its extraction method.
        let sits = hg.list_situations_by_narrative("out").unwrap();
        assert_eq!(sits.len(), 1, "expected exactly one materialized situation");
        match &sits[0].extraction_method {
            ExtractionMethod::Reconstructed {
                source_narrative_id,
                job_id,
            } => {
                assert_eq!(source_narrative_id, "src");
                assert_eq!(job_id, "job-abc");
            }
            other => panic!(
                "expected ExtractionMethod::Reconstructed, got {other:?}"
            ),
        }
    }

    /// T6 — Materialized situations land at keys with SYN_RECON_PREFIX.
    #[test]
    fn test_reconstruction_materialize_writes_under_syn_recon_prefix() {
        let hg = fresh_hg();
        let members = seed_entities(&hg, "src", 3);
        let result = dummy_result(members, 0.95);
        let _ = materialize_reconstruction(&hg, &result, "src", "out", "job-xyz", None).unwrap();

        // Scan the syn/recon/ prefix and check at least one key shows up.
        let scan = hg.store().prefix_scan(SYN_RECON_PREFIX).unwrap();
        assert_eq!(
            scan.len(),
            1,
            "expected exactly one key under syn/recon/, got {}",
            scan.len()
        );
        // The key must contain the output narrative id and job id.
        let (k, _v) = &scan[0];
        let key_str = String::from_utf8_lossy(k);
        assert!(
            key_str.starts_with("syn/recon/out/job-xyz/"),
            "key prefix mismatch: got '{key_str}'"
        );
        // The persisted record deserializes as ReconstructedSituationRef.
        let rec: ReconstructedSituationRef = serde_json::from_slice(&scan[0].1).unwrap();
        assert_eq!(rec.output_narrative_id, "out");
        assert_eq!(rec.job_id, "job-xyz");
        assert_eq!(rec.source_narrative_id, "src");
    }

    #[test]
    fn test_materialize_skips_low_confidence_edges() {
        let hg = fresh_hg();
        let members = seed_entities(&hg, "src", 3);
        let result = dummy_result(members, 0.5); // below 0.7 threshold
        let report = materialize_reconstruction(&hg, &result, "src", "out", "job", None).unwrap();
        assert_eq!(report.situations_created, 0);
        assert_eq!(report.situations_skipped, 1);
    }

    #[test]
    fn test_materialize_rejects_empty_output_narrative_id() {
        let hg = fresh_hg();
        let result = dummy_result(vec![Uuid::now_v7(), Uuid::now_v7()], 0.95);
        assert!(materialize_reconstruction(&hg, &result, "src", "", "job", None).is_err());
    }

    #[test]
    fn test_materialize_default_threshold_is_0_7() {
        // Edge confidence == 0.7 is NOT > 0.7 so should be skipped.
        let hg = fresh_hg();
        let members = seed_entities(&hg, "src", 2);
        let result = dummy_result(members, 0.7);
        let report = materialize_reconstruction(&hg, &result, "src", "out", "job", None).unwrap();
        assert_eq!(report.situations_created, 0);
        assert_eq!(report.confidence_threshold, 0.7);
    }

    // Suppress unused-helper warning; `make_entity` is kept for future tests.
    #[test]
    fn test_make_entity_helper_does_not_panic() {
        let _ = make_entity("noop");
    }
}
