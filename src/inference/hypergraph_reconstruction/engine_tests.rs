//! T6: ReconstructionEngine fits inference-layer conventions.

use std::sync::Arc;

use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::hypergraph::Hypergraph;
use crate::inference::hypergraph_reconstruction::{
    DerivativeEstimator, ObservationSource, ReconstructionEngine, ReconstructionParams,
    ReconstructionResult,
};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::memory::MemoryStore;
use crate::types::*;

fn make_hg() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn add_actor(hg: &Hypergraph, narrative: &str, name: &str) -> Uuid {
    hg.create_entity(Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": name}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 1.0,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some(narrative.to_string()),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    })
    .unwrap()
}

fn add_situation(
    hg: &Hypergraph,
    narrative: &str,
    start: chrono::DateTime<Utc>,
    members: &[Uuid],
) {
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
        raw_content: vec![ContentBlock::text("synth")],
        narrative_level: NarrativeLevel::Scene,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.9,
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
}

fn small_planted_narrative(hg: &Hypergraph, narrative: &str) -> Vec<Uuid> {
    let entities: Vec<Uuid> = (0..5).map(|i| add_actor(hg, narrative, &format!("e{i}"))).collect();
    let base = Utc::now() - Duration::seconds(60 * 200);
    // Plant pair {0, 1} firing every 60s.
    for tick in 0..60 {
        let ts = base + Duration::seconds(60 * tick);
        if tick % 2 == 0 {
            add_situation(hg, narrative, ts, &[entities[0], entities[1]]);
        }
        if tick % 3 == 0 {
            add_situation(hg, narrative, ts, &[entities[2], entities[3]]);
        }
        // Independent chatter on isolated entity 4.
        if tick % 7 == 0 {
            add_situation(hg, narrative, ts, &[entities[4]]);
        }
    }
    entities
}

#[test]
fn test_reconstruction_module_fits_inference_layer_conventions() {
    let hg = make_hg();
    let narrative = "engine-smoke-1";
    let _entities = small_planted_narrative(&hg, narrative);

    let engine = ReconstructionEngine;

    // Trait compliance: discriminant matches the variant.
    let job_type = engine.job_type();
    assert!(matches!(
        job_type,
        InferenceJobType::HypergraphReconstruction { .. }
    ));

    let params = ReconstructionParams {
        observation: ObservationSource::ParticipationRate,
        window_seconds: 60,
        time_resolution_seconds: 60,
        max_order: 2,
        lambda_l1: 0.0,
        derivative_estimator: DerivativeEstimator::SavitzkyGolay {
            window: 5,
            order: 2,
        },
        symmetrize: true,
        pearson_filter_threshold: 0.1,
        bootstrap_k: 3,
        entity_cap: 200,
        lambda_cv: false,
        bootstrap_seed: 42,
    };

    let job = InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type: InferenceJobType::HypergraphReconstruction {
            narrative_id: narrative.to_string(),
            params: serde_json::to_value(&params).unwrap(),
        },
        target_id: Uuid::now_v7(),
        parameters: serde_json::json!({}),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };

    // Cost estimate is positive and Result-shaped.
    let cost = engine.estimate_cost(&job, &hg).expect("estimate_cost must succeed");
    assert!(cost > 0, "cost estimate must be positive");

    // Execute round-trips through serde_json.
    let result = engine.execute(&job, &hg).expect("execute must succeed");
    assert_eq!(result.job_id, job.id);
    assert_eq!(result.status, JobStatus::Completed);
    assert_eq!(
        result.result.get("kind").and_then(|v| v.as_str()),
        Some("reconstruction_done")
    );

    // ReconstructionResult deserializes from the JSON envelope without loss.
    let inner = result.result.get("result").expect("result.result must exist");
    let reconstructed: ReconstructionResult = serde_json::from_value(inner.clone())
        .expect("ReconstructionResult must deserialize");
    assert_eq!(
        reconstructed.observation_source,
        ObservationSource::ParticipationRate
    );
    assert!(reconstructed.coefficient_matrix_stats.n_entities >= 5);
}
