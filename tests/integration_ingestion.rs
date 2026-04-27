//! Integration tests for the Phase 1 ingestion pipeline.

use std::sync::Arc;

use tensa::error::Result;
use tensa::hypergraph::Hypergraph;
use tensa::ingestion::chunker::TextChunk;
use tensa::ingestion::embed::EmbeddingProvider;
use tensa::ingestion::embed::HashEmbedding;
use tensa::ingestion::extraction::*;
use tensa::ingestion::gate::ConfidenceGate;
use tensa::ingestion::llm::NarrativeExtractor;
use tensa::ingestion::pipeline::{IngestionPipeline, PipelineConfig};
use tensa::ingestion::queue::ValidationQueue;
use tensa::ingestion::resolve::EntityResolver;
use tensa::ingestion::vector::VectorIndex;
use tensa::store::memory::MemoryStore;
use tensa::types::*;
use tensa::KVStore;

/// Mock extractor that returns a Crime and Punishment-inspired extraction.
struct CrimeAndPunishmentExtractor;

impl NarrativeExtractor for CrimeAndPunishmentExtractor {
    fn extract_narrative(&self, _chunk: &TextChunk) -> Result<NarrativeExtraction> {
        Ok(NarrativeExtraction {
            entities: vec![
                ExtractedEntity {
                    name: "Raskolnikov".to_string(),
                    aliases: vec!["Rodion".to_string(), "Rodya".to_string()],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"age": 23, "occupation": "student"}),
                    confidence: 0.95,
                },
                ExtractedEntity {
                    name: "Sonya Marmeladova".to_string(),
                    aliases: vec!["Sonya".to_string(), "Sonechka".to_string()],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.92,
                },
                ExtractedEntity {
                    name: "Porfiry Petrovich".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({"role": "detective"}),
                    confidence: 0.88,
                },
                ExtractedEntity {
                    name: "Sonya's Room".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Location,
                    properties: serde_json::json!({}),
                    confidence: 0.85,
                },
                ExtractedEntity {
                    name: "Mysterious Figure".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.4, // Medium — will go to queue
                },
                ExtractedEntity {
                    name: "Maybe a Cat".to_string(),
                    aliases: vec![],
                    entity_type: EntityType::Actor,
                    properties: serde_json::json!({}),
                    confidence: 0.15, // Low — will be rejected
                },
            ],
            situations: vec![
                ExtractedSituation {
                    name: None,
                    description: "Raskolnikov confesses the murder to Sonya".to_string(),
                    temporal_marker: Some("evening".to_string()),
                    location: Some("Sonya's room".to_string()),
                    narrative_level: NarrativeLevel::Scene,
                    content_blocks: vec![ContentBlock::text(
                        "He told her everything. The whole truth.",
                    )],
                    confidence: 0.9,
                    text_start: None,
                    text_end: None,
                },
                ExtractedSituation {
                    name: None,
                    description: "Porfiry interrogates Raskolnikov".to_string(),
                    temporal_marker: Some("the next morning".to_string()),
                    location: None,
                    narrative_level: NarrativeLevel::Scene,
                    content_blocks: vec![],
                    confidence: 0.85,
                    text_start: None,
                    text_end: None,
                },
            ],
            participations: vec![
                ExtractedParticipation {
                    entity_name: "Raskolnikov".to_string(),
                    situation_index: 0,
                    role: Role::Protagonist,
                    action: Some("confesses".to_string()),
                    confidence: 0.9,
                },
                ExtractedParticipation {
                    entity_name: "Sonya Marmeladova".to_string(),
                    situation_index: 0,
                    role: Role::Witness,
                    action: Some("listens and weeps".to_string()),
                    confidence: 0.88,
                },
                ExtractedParticipation {
                    entity_name: "Raskolnikov".to_string(),
                    situation_index: 1,
                    role: Role::Target,
                    action: Some("answers questions".to_string()),
                    confidence: 0.85,
                },
                ExtractedParticipation {
                    entity_name: "Porfiry Petrovich".to_string(),
                    situation_index: 1,
                    role: Role::Protagonist,
                    action: Some("interrogates".to_string()),
                    confidence: 0.9,
                },
            ],
            causal_links: vec![ExtractedCausalLink {
                from_situation_index: 0,
                to_situation_index: 1,
                mechanism: Some("Confession leads to investigation".to_string()),
                causal_type: CausalType::Contributing,
                strength: 0.7,
                confidence: 0.82,
            }],
            temporal_relations: vec![ExtractedTemporalRelation {
                situation_a_index: 0,
                situation_b_index: 1,
                relation: AllenRelation::Before,
                confidence: 0.95,
            }],
        })
    }
}

fn setup_pipeline() -> (
    IngestionPipeline,
    Arc<Hypergraph>,
    Arc<ValidationQueue>,
    Arc<dyn KVStore>,
) {
    let store = Arc::new(MemoryStore::new());
    let hg = Arc::new(Hypergraph::new(store.clone()));
    let queue = Arc::new(ValidationQueue::new(store.clone()));
    let extractor = Arc::new(CrimeAndPunishmentExtractor);
    let embedder: Option<Arc<dyn EmbeddingProvider>> = Some(Arc::new(HashEmbedding::new(64)));
    let vector_index = Some(Arc::new(std::sync::RwLock::new(VectorIndex::new(64))));

    let config = PipelineConfig {
        chunker: tensa::ingestion::chunker::ChunkerConfig {
            max_tokens: 5000,
            overlap_tokens: 0,
            chapter_regex: None,
            ..Default::default()
        },
        auto_commit_threshold: 0.8,
        review_threshold: 0.3,
        source_id: "crime-and-punishment".to_string(),
        source_type: "novel".to_string(),
        ..Default::default()
    };

    let pipeline = IngestionPipeline::new(
        hg.clone(),
        extractor,
        embedder,
        vector_index,
        queue.clone(),
        config,
    );
    (pipeline, hg, queue, store as Arc<dyn KVStore>)
}

#[test]
fn test_end_to_end_ingest_and_query() {
    let (pipeline, hg, _queue, _store) = setup_pipeline();

    let text = r#"
        Raskolnikov walked slowly up the stairs to Sonya's room. He had to tell her.
        The weight of what he had done was crushing him. He entered and found her alone.

        "I have come to tell you something," he said. She looked at him with those eyes
        full of suffering and understanding. He told her everything. The whole truth.
        She wept, but she did not turn away.
    "#;

    let report = pipeline
        .ingest_text(text, "Crime and Punishment - Part 5")
        .unwrap();

    // Verify chunks were processed
    assert_eq!(report.chunks_processed, 1);

    // Verify entities created (4 high-confidence: Raskolnikov, Sonya, Porfiry, Room)
    assert!(
        report.entities_created >= 4,
        "Expected >=4 entities, got {}",
        report.entities_created
    );

    // Verify situations created (2 scenes)
    assert!(
        report.situations_created >= 2,
        "Expected >=2 situations, got {}",
        report.situations_created
    );

    // Verify gating: medium-confidence item queued, low rejected
    assert!(report.items_queued >= 1, "Expected >=1 queued item");
    assert!(report.items_rejected >= 1, "Expected >=1 rejected item");

    // Verify we can query the ingested data via TensaQL
    let tree = tensa::temporal::index::IntervalTree::new();
    let q = tensa::query::parser::parse_query("MATCH (e:Actor) RETURN e").unwrap();
    let plan = tensa::query::planner::plan_query(&q).unwrap();
    let results = tensa::query::executor::execute(&plan, &hg, &tree).unwrap();
    assert!(
        results.len() >= 3,
        "Expected >=3 actors, got {}",
        results.len()
    );
}

#[test]
fn test_ingest_with_validation_queue() {
    let (pipeline, _hg, queue, _store) = setup_pipeline();

    let report = pipeline
        .ingest_text("Some narrative text about characters.", "test")
        .unwrap();

    // Medium-confidence items should be in the queue
    let pending = queue.list_pending(50).unwrap();
    assert!(
        !pending.is_empty(),
        "Queue should have pending items for medium-confidence extractions"
    );
    assert_eq!(pending.len(), report.items_queued);

    // Approve one item
    let first_id = pending[0].id;
    let approved = queue.approve(&first_id, "test_reviewer").unwrap();
    assert_eq!(
        approved.status,
        tensa::ingestion::queue::QueueItemStatus::Approved
    );

    // Verify pending count decreased
    let remaining = queue.list_pending(50).unwrap();
    assert_eq!(remaining.len(), pending.len() - 1);
}

#[test]
fn test_ingest_entity_resolution_across_chunks() {
    let store = Arc::new(MemoryStore::new());
    let hg = Arc::new(Hypergraph::new(store.clone()));
    let queue = Arc::new(ValidationQueue::new(store.clone()));

    // Second extraction also mentions "Rodya" (alias of Raskolnikov)
    struct TwoChunkExtractor;
    impl NarrativeExtractor for TwoChunkExtractor {
        fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction> {
            if chunk.chunk_id == 0 {
                Ok(NarrativeExtraction {
                    entities: vec![ExtractedEntity {
                        name: "Raskolnikov".to_string(),
                        aliases: vec!["Rodya".to_string()],
                        entity_type: EntityType::Actor,
                        properties: serde_json::json!({"age": 23}),
                        confidence: 0.9,
                    }],
                    situations: vec![],
                    participations: vec![],
                    causal_links: vec![],
                    temporal_relations: vec![],
                })
            } else {
                Ok(NarrativeExtraction {
                    entities: vec![ExtractedEntity {
                        name: "Rodya".to_string(),
                        aliases: vec![],
                        entity_type: EntityType::Actor,
                        properties: serde_json::json!({}),
                        confidence: 0.9,
                    }],
                    situations: vec![],
                    participations: vec![],
                    causal_links: vec![],
                    temporal_relations: vec![],
                })
            }
        }
    }

    let config = PipelineConfig {
        chunker: tensa::ingestion::chunker::ChunkerConfig {
            max_tokens: 20,
            overlap_tokens: 5,
            chapter_regex: None,
            ..Default::default()
        },
        auto_commit_threshold: 0.8,
        review_threshold: 0.3,
        source_id: "test".to_string(),
        source_type: "novel".to_string(),
        ..Default::default()
    };

    let pipeline = IngestionPipeline::new(
        hg.clone(),
        Arc::new(TwoChunkExtractor),
        None,
        None,
        queue,
        config,
    );

    // Text long enough to produce 2 chunks
    let text = "Raskolnikov went out into the cold morning air.\n\nRodya walked through the streets of Petersburg thinking about everything.";
    let report = pipeline.ingest_text(text, "test").unwrap();

    // Only 1 entity should be created (Rodya resolved to Raskolnikov)
    assert_eq!(
        report.entities_created, 1,
        "Expected 1 entity (Rodya should resolve to Raskolnikov), got {}",
        report.entities_created
    );
}

#[test]
fn test_confidence_gating_thresholds() {
    let gate = ConfidenceGate::default();
    use tensa::ingestion::gate::GateDecision;

    // Auto-commit
    assert_eq!(gate.decide(0.95), GateDecision::AutoCommit);
    assert_eq!(gate.decide(0.80), GateDecision::AutoCommit);

    // Queue for review
    assert_eq!(gate.decide(0.79), GateDecision::QueueForReview);
    assert_eq!(gate.decide(0.50), GateDecision::QueueForReview);
    assert_eq!(gate.decide(0.30), GateDecision::QueueForReview);

    // Reject
    assert_eq!(gate.decide(0.29), GateDecision::Reject);
    assert_eq!(gate.decide(0.10), GateDecision::Reject);
    assert_eq!(gate.decide(0.00), GateDecision::Reject);
}

#[test]
fn test_entity_resolver_coreference() {
    let embedder = HashEmbedding::new(64);
    let mut resolver = EntityResolver::new();

    // Register Raskolnikov with aliases
    let id = uuid::Uuid::now_v7();
    let emb = embedder.embed_text("Raskolnikov").unwrap();
    resolver.register(
        id,
        "Raskolnikov",
        &["Rodion".to_string(), "Rodya".to_string()],
        EntityType::Actor,
        Some(emb),
    );

    // "Rodya" should resolve to the same entity
    let extracted = ExtractedEntity {
        name: "Rodya".to_string(),
        aliases: vec![],
        entity_type: EntityType::Actor,
        properties: serde_json::json!({}),
        confidence: 0.9,
    };

    match resolver.resolve(&extracted, Some(&embedder)) {
        tensa::ingestion::resolve::ResolveResult::Existing(found_id) => {
            assert_eq!(found_id, id, "Rodya should resolve to Raskolnikov");
        }
        tensa::ingestion::resolve::ResolveResult::New => {
            panic!("Expected Rodya to resolve to existing Raskolnikov");
        }
    }

    // Completely new character should not resolve
    let new_entity = ExtractedEntity {
        name: "Svidrigailov".to_string(),
        aliases: vec![],
        entity_type: EntityType::Actor,
        properties: serde_json::json!({}),
        confidence: 0.9,
    };
    match resolver.resolve(&new_entity, None) {
        tensa::ingestion::resolve::ResolveResult::New => {} // Expected
        tensa::ingestion::resolve::ResolveResult::Existing(_) => {
            panic!("Svidrigailov should not resolve to any existing entity");
        }
    }
}

#[test]
fn test_vector_index_integration() {
    let store = Arc::new(MemoryStore::new());
    let embedder = HashEmbedding::new(64);
    let mut index = VectorIndex::new(64);

    // Add some entities
    let id1 = uuid::Uuid::now_v7();
    let id2 = uuid::Uuid::now_v7();
    let id3 = uuid::Uuid::now_v7();

    let emb1 = embedder.embed_text("Raskolnikov the murderer").unwrap();
    let emb2 = embedder.embed_text("Sonya the compassionate").unwrap();
    let emb3 = embedder.embed_text("Porfiry the detective").unwrap();

    index.add(id1, &emb1).unwrap();
    index.add(id2, &emb2).unwrap();
    index.add(id3, &emb3).unwrap();

    // Search for something similar to Raskolnikov
    let query = embedder.embed_text("Raskolnikov the murderer").unwrap();
    let results = index.search(&query, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, id1, "Exact match should be first result");
    assert!(
        (results[0].score - 1.0).abs() < 1e-5,
        "Exact match should have score ~1.0"
    );

    // Persist and reload
    index.save(store.as_ref()).unwrap();
    let loaded = VectorIndex::load(store.as_ref(), 64).unwrap();
    assert_eq!(loaded.len(), 3);
    let loaded_results = loaded.search(&query, 1).unwrap();
    assert_eq!(loaded_results[0].id, id1);
}

#[test]
fn test_extraction_parse_and_validate() {
    let json = r#"{
        "entities": [
            {"name": "Alice", "entity_type": "Actor", "confidence": 0.9},
            {"name": "Bob", "entity_type": "Actor", "confidence": 0.85}
        ],
        "situations": [
            {
                "description": "Alice meets Bob",
                "narrative_level": "Event",
                "confidence": 0.8
            }
        ],
        "participations": [
            {"entity_name": "Alice", "situation_index": 0, "role": "Protagonist", "confidence": 0.9},
            {"entity_name": "Bob", "situation_index": 0, "role": "Witness", "confidence": 0.85}
        ],
        "causal_links": [],
        "temporal_relations": []
    }"#;

    let extraction = tensa::ingestion::extraction::parse_llm_response(json).unwrap();
    assert_eq!(extraction.entities.len(), 2);
    assert_eq!(extraction.situations.len(), 1);
    assert_eq!(extraction.participations.len(), 2);

    let warnings = tensa::ingestion::extraction::validate_extraction(&extraction);
    assert!(
        warnings.is_empty(),
        "Valid extraction should have no warnings"
    );
}
