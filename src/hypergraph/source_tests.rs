use super::*;
use crate::store::memory::MemoryStore;
use crate::types::*;
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

fn make_hg() -> Hypergraph {
    Hypergraph::new(Arc::new(MemoryStore::new()))
}

fn make_source(name: &str, trust: f32) -> Source {
    Source {
        id: Uuid::now_v7(),
        name: name.to_string(),
        source_type: SourceType::NewsOutlet,
        url: None,
        description: None,
        trust_score: trust,
        bias_profile: BiasProfile::default(),
        track_record: TrackRecord::default(),
        tags: vec![],
        author: None,
        publication: None,
        ingested_by: None,
        ingestion_notes: None,
        publication_date: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn make_situation_with(hg: &Hypergraph, narrative_id: Option<&str>) -> Uuid {
    let sit = Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: Some(Utc::now()),
            end: Some(Utc::now()),
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
        narrative_level: NarrativeLevel::Event,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.7,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: narrative_id.map(String::from),
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
    hg.create_situation(sit).unwrap()
}

fn make_situation(hg: &Hypergraph) -> Uuid {
    make_situation_with(hg, None)
}

fn make_entity_with(hg: &Hypergraph, confidence: f32, narrative_id: Option<&str>) -> Uuid {
    let entity = Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": "Test Actor"}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: Some(ExtractionMethod::HumanEntered),
        narrative_id: narrative_id.map(String::from),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        deleted_at: None,
        transaction_time: None,
    };
    hg.create_entity(entity).unwrap()
}

fn make_entity(hg: &Hypergraph, confidence: f32) -> Uuid {
    make_entity_with(hg, confidence, None)
}

fn make_attribution(
    source_id: Uuid,
    target_id: Uuid,
    target_kind: AttributionTarget,
) -> SourceAttribution {
    SourceAttribution {
        source_id,
        target_id,
        target_kind,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.9,
        claim: None,
    }
}

// ─── Source CRUD ─────────────────────────────────────────────

#[test]
fn test_create_and_get_source() {
    let hg = make_hg();
    let src = make_source("Reuters", 0.85);
    let id = hg.create_source(src.clone()).unwrap();
    let fetched = hg.get_source(&id).unwrap();
    assert_eq!(fetched.name, "Reuters");
    assert_eq!(fetched.trust_score, 0.85);
}

#[test]
fn test_get_source_not_found() {
    let hg = make_hg();
    let result = hg.get_source(&Uuid::now_v7());
    assert!(matches!(result, Err(TensaError::SourceNotFound(_))));
}

#[test]
fn test_update_source() {
    let hg = make_hg();
    let src = make_source("RT", 0.3);
    let id = hg.create_source(src).unwrap();
    let updated = hg
        .update_source(&id, |s| {
            s.trust_score = 0.2;
            s.bias_profile.known_biases.push("state-owned".to_string());
        })
        .unwrap();
    assert_eq!(updated.trust_score, 0.2);
    assert_eq!(updated.bias_profile.known_biases, vec!["state-owned"]);
}

#[test]
fn test_delete_source() {
    let hg = make_hg();
    let src = make_source("Test", 0.5);
    let id = hg.create_source(src).unwrap();
    hg.delete_source(&id).unwrap();
    assert!(matches!(
        hg.get_source(&id),
        Err(TensaError::SourceNotFound(_))
    ));
}

#[test]
fn test_list_sources() {
    let hg = make_hg();
    hg.create_source(make_source("A", 0.5)).unwrap();
    hg.create_source(make_source("B", 0.7)).unwrap();
    let sources = hg.list_sources().unwrap();
    assert_eq!(sources.len(), 2);
}

#[test]
fn test_list_sources_for_narrative_filters_by_attribution() {
    let hg = make_hg();
    let src_a = hg.create_source(make_source("A", 0.8)).unwrap();
    let src_b = hg.create_source(make_source("B", 0.8)).unwrap();
    let src_c = hg.create_source(make_source("C", 0.8)).unwrap();
    let _orphan = hg.create_source(make_source("Orphan", 0.5)).unwrap();

    let sit_x = make_situation_with(&hg, Some("nar-x"));
    let ent_x = make_entity_with(&hg, 0.9, Some("nar-x"));
    let sit_y = make_situation_with(&hg, Some("nar-y"));

    // A → only nar-x, B → only nar-y, C → both narratives
    hg.add_attribution(make_attribution(src_a, sit_x, AttributionTarget::Situation))
        .unwrap();
    hg.add_attribution(make_attribution(src_a, ent_x, AttributionTarget::Entity))
        .unwrap();
    hg.add_attribution(make_attribution(src_b, sit_y, AttributionTarget::Situation))
        .unwrap();
    hg.add_attribution(make_attribution(src_c, sit_x, AttributionTarget::Situation))
        .unwrap();
    hg.add_attribution(make_attribution(src_c, sit_y, AttributionTarget::Situation))
        .unwrap();

    let mut names_x: Vec<String> = hg
        .list_sources_for_narrative("nar-x")
        .unwrap()
        .into_iter()
        .map(|s| s.name)
        .collect();
    names_x.sort();
    assert_eq!(names_x, vec!["A".to_string(), "C".to_string()]);

    let mut names_y: Vec<String> = hg
        .list_sources_for_narrative("nar-y")
        .unwrap()
        .into_iter()
        .map(|s| s.name)
        .collect();
    names_y.sort();
    assert_eq!(names_y, vec!["B".to_string(), "C".to_string()]);

    assert!(hg
        .list_sources_for_narrative("nonexistent")
        .unwrap()
        .is_empty());
}

// ─── Attribution ─────────────────────────────────────────────

#[test]
fn test_add_and_query_attribution() {
    let hg = make_hg();
    let src = make_source("BBC", 0.9);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    let attr = SourceAttribution {
        source_id: src_id,
        target_id: sit_id,
        target_kind: AttributionTarget::Situation,
        retrieved_at: Utc::now(),
        original_url: Some("https://bbc.com/article".to_string()),
        excerpt: Some("Event happened".to_string()),
        extraction_confidence: 0.95,
        claim: None,
    };
    hg.add_attribution(attr).unwrap();

    let by_source = hg.get_attributions_for_source(&src_id).unwrap();
    assert_eq!(by_source.len(), 1);
    assert_eq!(by_source[0].target_id, sit_id);

    let by_target = hg.get_attributions_for_target(&sit_id).unwrap();
    assert_eq!(by_target.len(), 1);
    assert_eq!(by_target[0].source_id, src_id);
}

#[test]
fn test_attribution_duplicate_rejected() {
    let hg = make_hg();
    let src = make_source("AP", 0.85);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    let attr = SourceAttribution {
        source_id: src_id,
        target_id: sit_id,
        target_kind: AttributionTarget::Situation,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.8,
        claim: None,
    };
    hg.add_attribution(attr.clone()).unwrap();
    assert!(matches!(
        hg.add_attribution(attr),
        Err(TensaError::AttributionExists { .. })
    ));
}

#[test]
fn test_remove_attribution() {
    let hg = make_hg();
    let src = make_source("Test", 0.5);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    let attr = SourceAttribution {
        source_id: src_id,
        target_id: sit_id,
        target_kind: AttributionTarget::Situation,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.8,
        claim: None,
    };
    hg.add_attribution(attr).unwrap();
    hg.remove_attribution(&src_id, &sit_id).unwrap();
    assert!(hg.get_attributions_for_source(&src_id).unwrap().is_empty());
    assert!(hg.get_attributions_for_target(&sit_id).unwrap().is_empty());
}

#[test]
fn test_delete_source_cascades_attributions() {
    let hg = make_hg();
    let src = make_source("Test", 0.5);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    let attr = SourceAttribution {
        source_id: src_id,
        target_id: sit_id,
        target_kind: AttributionTarget::Situation,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.8,
        claim: None,
    };
    hg.add_attribution(attr).unwrap();
    hg.delete_source(&src_id).unwrap();
    // Reverse index should also be cleaned up
    assert!(hg.get_attributions_for_target(&sit_id).unwrap().is_empty());
}

// ─── Contention ──────────────────────────────────────────────

#[test]
fn test_add_and_query_contention() {
    let hg = make_hg();
    let sit_a = make_situation(&hg);
    let sit_b = make_situation(&hg);

    let link = ContentionLink {
        situation_a: sit_a,
        situation_b: sit_b,
        contention_type: ContentionType::DirectContradiction,
        description: Some("Source A says yes, source B says no".to_string()),
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    };
    hg.add_contention(link).unwrap();

    let from_a = hg.get_contentions_for_situation(&sit_a).unwrap();
    assert_eq!(from_a.len(), 1);
    let from_b = hg.get_contentions_for_situation(&sit_b).unwrap();
    assert_eq!(from_b.len(), 1);
}

#[test]
fn test_contention_duplicate_rejected() {
    let hg = make_hg();
    let sit_a = make_situation(&hg);
    let sit_b = make_situation(&hg);

    let link = ContentionLink {
        situation_a: sit_a,
        situation_b: sit_b,
        contention_type: ContentionType::NumericalDisagreement,
        description: None,
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    };
    hg.add_contention(link.clone()).unwrap();
    assert!(matches!(
        hg.add_contention(link),
        Err(TensaError::ContentionExists { .. })
    ));
}

#[test]
fn test_remove_contention() {
    let hg = make_hg();
    let sit_a = make_situation(&hg);
    let sit_b = make_situation(&hg);

    let link = ContentionLink {
        situation_a: sit_a,
        situation_b: sit_b,
        contention_type: ContentionType::TemporalDisagreement,
        description: None,
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    };
    hg.add_contention(link).unwrap();
    hg.remove_contention(&sit_a, &sit_b).unwrap();
    assert!(hg.get_contentions_for_situation(&sit_a).unwrap().is_empty());
}

#[test]
fn test_resolve_contention() {
    let hg = make_hg();
    let sit_a = make_situation(&hg);
    let sit_b = make_situation(&hg);

    let link = ContentionLink {
        situation_a: sit_a,
        situation_b: sit_b,
        contention_type: ContentionType::AttributionDisagreement,
        description: None,
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    };
    hg.add_contention(link).unwrap();
    let resolved = hg
        .resolve_contention(&sit_a, &sit_b, "Source A confirmed correct".to_string())
        .unwrap();
    assert!(resolved.resolved);
    assert_eq!(
        resolved.resolution.as_deref(),
        Some("Source A confirmed correct")
    );
}

// ─── Corroboration ───────────────────────────────────────────

#[test]
fn test_corroboration_no_sources() {
    let hg = make_hg();
    let target_id = Uuid::now_v7();
    let score = hg.compute_corroboration(&target_id).unwrap();
    assert_eq!(score, 0.0);
}

#[test]
fn test_corroboration_single_source() {
    let hg = make_hg();
    let src = make_source("Reuters", 0.8);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    let attr = SourceAttribution {
        source_id: src_id,
        target_id: sit_id,
        target_kind: AttributionTarget::Situation,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.9,
        claim: None,
    };
    hg.add_attribution(attr).unwrap();

    let score = hg.compute_corroboration(&sit_id).unwrap();
    // 1 - (1 - 0.8) = 0.8
    assert!((score - 0.8).abs() < 0.001);
}

#[test]
fn test_corroboration_multiple_sources() {
    let hg = make_hg();
    let src1 = make_source("Reuters", 0.8);
    let src2 = make_source("AP", 0.7);
    let src1_id = hg.create_source(src1).unwrap();
    let src2_id = hg.create_source(src2).unwrap();
    let sit_id = make_situation(&hg);

    for src_id in [src1_id, src2_id] {
        let attr = SourceAttribution {
            source_id: src_id,
            target_id: sit_id,
            target_kind: AttributionTarget::Situation,
            retrieved_at: Utc::now(),
            original_url: None,
            excerpt: None,
            extraction_confidence: 0.9,
            claim: None,
        };
        hg.add_attribution(attr).unwrap();
    }

    let score = hg.compute_corroboration(&sit_id).unwrap();
    // 1 - (1-0.8)(1-0.7) = 1 - 0.2*0.3 = 1 - 0.06 = 0.94
    assert!((score - 0.94).abs() < 0.001);
}

#[test]
fn test_recompute_confidence() {
    let hg = make_hg();
    let src = make_source("BBC", 0.9);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    let attr = SourceAttribution {
        source_id: src_id,
        target_id: sit_id,
        target_kind: AttributionTarget::Situation,
        retrieved_at: Utc::now(),
        original_url: None,
        excerpt: None,
        extraction_confidence: 0.95,
        claim: None,
    };
    hg.add_attribution(attr).unwrap();

    let breakdown = hg.recompute_confidence(&sit_id, 0.7).unwrap();
    assert_eq!(breakdown.extraction, 0.7);
    assert!((breakdown.source_credibility - 0.9).abs() < 0.001);
    assert!(breakdown.corroboration > 0.0); // derived from Beta concentration
    assert!(breakdown.recency > 0.9); // just created, should be very recent
    assert!(breakdown.composite() > 0.0);
    assert!(breakdown.composite() <= 1.0);
    // Bayesian fields should be populated
    assert!(breakdown.posterior_alpha.is_some());
    assert!(breakdown.posterior_beta.is_some());
}

// ─── Reactive Bayesian Confidence ────────────────────────────

#[test]
fn test_add_attribution_triggers_recompute() {
    let hg = make_hg();
    let src = make_source("Reuters", 0.9);
    let src_id = hg.create_source(src).unwrap();
    let ent_id = make_entity(&hg, 0.5);

    // Before attribution, confidence is 0.5
    let before = hg.get_entity(&ent_id).unwrap();
    assert_eq!(before.confidence, 0.5);
    assert!(before.confidence_breakdown.is_none());

    // Add attribution — should trigger reactive recompute
    let attr = make_attribution(src_id, ent_id, AttributionTarget::Entity);
    hg.add_attribution(attr).unwrap();

    let after = hg.get_entity(&ent_id).unwrap();
    // Confidence should have changed from 0.5
    assert_ne!(after.confidence, 0.5);
    // Bayesian: prior Beta(1.0, 1.0), then trust=0.9 evidence pushes alpha up
    // Posterior mean should be above the prior mean of 0.5
    assert!(
        after.confidence > 0.5,
        "Trusted source should increase confidence above 0.5, got {}",
        after.confidence
    );
    assert!(after.confidence_breakdown.is_some());
}

#[test]
fn test_remove_attribution_triggers_recompute() {
    let hg = make_hg();
    let src = make_source("Reuters", 0.9);
    let src_id = hg.create_source(src).unwrap();
    let ent_id = make_entity(&hg, 0.5);

    let attr = make_attribution(src_id, ent_id, AttributionTarget::Entity);
    hg.add_attribution(attr).unwrap();

    let with_source = hg.get_entity(&ent_id).unwrap();
    assert!(
        with_source.confidence > 0.5,
        "Trusted source should push confidence above 0.5, got {}",
        with_source.confidence
    );

    // Remove attribution — should trigger recompute back toward no-source state
    hg.remove_attribution(&src_id, &ent_id).unwrap();

    let after = hg.get_entity(&ent_id).unwrap();
    // With no sources: source_credibility = current_confidence, corroboration = 0
    // The confidence should drop from the boosted value
    assert!(after.confidence < with_source.confidence);
}

#[test]
fn test_propagate_source_trust_change() {
    let hg = make_hg();
    let src = make_source("Sketchy News", 0.8);
    let src_id = hg.create_source(src).unwrap();
    let ent1_id = make_entity(&hg, 0.6);
    let ent2_id = make_entity(&hg, 0.7);

    // Attribute source to both entities
    hg.add_attribution(make_attribution(src_id, ent1_id, AttributionTarget::Entity))
        .unwrap();
    hg.add_attribution(make_attribution(src_id, ent2_id, AttributionTarget::Entity))
        .unwrap();

    let before1 = hg.get_entity(&ent1_id).unwrap().confidence;
    let before2 = hg.get_entity(&ent2_id).unwrap().confidence;

    // Downgrade source trust
    hg.update_source(&src_id, |s| {
        s.trust_score = 0.2;
    })
    .unwrap();

    // Propagate the trust change
    hg.propagate_source_trust_change(&src_id).unwrap();

    let after1 = hg.get_entity(&ent1_id).unwrap().confidence;
    let after2 = hg.get_entity(&ent2_id).unwrap().confidence;

    // Both entities should have decreased confidence
    assert!(
        after1 < before1,
        "Entity 1 confidence should decrease: {} -> {}",
        before1,
        after1
    );
    assert!(
        after2 < before2,
        "Entity 2 confidence should decrease: {} -> {}",
        before2,
        after2
    );
}

#[test]
fn test_propagate_no_attributions_noop() {
    let hg = make_hg();
    let src = make_source("Lonely Source", 0.5);
    let src_id = hg.create_source(src).unwrap();

    // Should not error even with zero attributions
    hg.propagate_source_trust_change(&src_id).unwrap();
}

#[test]
fn test_reactive_confidence_entity_target() {
    let hg = make_hg();
    let src = make_source("AP", 0.85);
    let src_id = hg.create_source(src).unwrap();
    let ent_id = make_entity(&hg, 0.6);

    hg.add_attribution(make_attribution(src_id, ent_id, AttributionTarget::Entity))
        .unwrap();

    let entity = hg.get_entity(&ent_id).unwrap();
    let bd = entity
        .confidence_breakdown
        .expect("breakdown should be set");
    assert_eq!(bd.extraction, 0.6);
    assert!((bd.source_credibility - 0.85).abs() < 0.001);
    assert!(bd.corroboration > 0.0); // derived from Beta concentration
    assert!(bd.recency > 0.9);
    // Verify composite matches stored confidence (Bayesian posterior mean)
    assert!((entity.confidence - bd.composite()).abs() < 0.001);
    // Bayesian fields populated
    assert!(bd.posterior_alpha.is_some());
}

#[test]
fn test_reactive_confidence_situation_target() {
    let hg = make_hg();
    let src = make_source("BBC", 0.9);
    let src_id = hg.create_source(src).unwrap();
    let sit_id = make_situation(&hg);

    hg.add_attribution(make_attribution(
        src_id,
        sit_id,
        AttributionTarget::Situation,
    ))
    .unwrap();

    let sit = hg.get_situation(&sit_id).unwrap();
    let bd = sit
        .confidence_breakdown
        .expect("breakdown should be set on situation");
    assert_eq!(bd.extraction, 0.7); // make_situation uses 0.7
    assert!((bd.source_credibility - 0.9).abs() < 0.001);
    assert!(bd.recency > 0.9);
    // Verify composite matches stored confidence (Bayesian posterior mean)
    assert!((sit.confidence - bd.composite()).abs() < 0.001);
    assert!(bd.posterior_alpha.is_some());
}

// ─── Bayesian confidence tests ─────────────────────────────

#[test]
fn test_bayesian_no_attributions() {
    let hg = make_hg();
    let extraction_conf = 0.8;
    let entity_id = make_entity(&hg, extraction_conf);

    let bd = hg
        .recompute_confidence(&entity_id, extraction_conf)
        .unwrap();

    assert!(bd.prior_alpha.is_some());
    assert!(bd.prior_beta.is_some());
    assert!(bd.posterior_alpha.is_some());
    assert!(bd.posterior_beta.is_some());

    // With no evidence, posterior = prior
    let pa = bd.posterior_alpha.unwrap();
    let pb = bd.posterior_beta.unwrap();
    assert!((pa - bd.prior_alpha.unwrap()).abs() < 1e-6);
    assert!((pb - bd.prior_beta.unwrap()).abs() < 1e-6);

    // Posterior mean should equal extraction confidence
    let posterior_mean = pa / (pa + pb);
    assert!((posterior_mean - extraction_conf).abs() < 0.01);
}

#[test]
fn test_bayesian_trusted_source_increases() {
    let hg = make_hg();
    let extraction_conf = 0.8;
    let entity_id = make_entity(&hg, extraction_conf);

    let source = make_source("Trusted", 0.95);
    let source_id = source.id;
    hg.create_source(source).unwrap();

    hg.add_attribution(make_attribution(
        source_id,
        entity_id,
        AttributionTarget::Entity,
    ))
    .unwrap();

    let bd = hg
        .recompute_confidence(&entity_id, extraction_conf)
        .unwrap();
    let posterior_mean = bd.composite();

    // High-trust source should maintain or increase confidence
    assert!(
        posterior_mean > extraction_conf - 0.05,
        "Trusted source should maintain or increase confidence, got {} vs extraction {}",
        posterior_mean,
        extraction_conf
    );
    assert!(bd.posterior_alpha.unwrap() > bd.prior_alpha.unwrap());
}

#[test]
fn test_bayesian_untrusted_source_decreases() {
    let hg = make_hg();
    let extraction_conf = 0.8;
    let entity_id = make_entity(&hg, extraction_conf);

    let source = make_source("Unreliable", 0.1);
    let source_id = source.id;
    hg.create_source(source).unwrap();

    hg.add_attribution(make_attribution(
        source_id,
        entity_id,
        AttributionTarget::Entity,
    ))
    .unwrap();

    let bd = hg
        .recompute_confidence(&entity_id, extraction_conf)
        .unwrap();
    let posterior_mean = bd.composite();

    // Low-trust source should decrease confidence
    assert!(
        posterior_mean < extraction_conf,
        "Untrusted source should decrease confidence, got {} vs extraction {}",
        posterior_mean,
        extraction_conf
    );
    let alpha_growth = bd.posterior_alpha.unwrap() - bd.prior_alpha.unwrap();
    let beta_growth = bd.posterior_beta.unwrap() - bd.prior_beta.unwrap();
    assert!(
        beta_growth > alpha_growth,
        "Beta should grow more for untrusted source: alpha={}, beta={}",
        alpha_growth,
        beta_growth
    );
}

#[test]
fn test_bayesian_multiple_sources_converge() {
    let hg = make_hg();
    let entity_id = make_entity(&hg, 0.5);

    for i in 0..5 {
        let source = make_source(&format!("Source-{}", i), 0.85);
        let source_id = source.id;
        hg.create_source(source).unwrap();
        hg.add_attribution(make_attribution(
            source_id,
            entity_id,
            AttributionTarget::Entity,
        ))
        .unwrap();
    }

    let bd = hg.recompute_confidence(&entity_id, 0.5).unwrap();
    let posterior_mean = bd.composite();

    assert!(
        posterior_mean > 0.6,
        "5 trusted sources should pull posterior above 0.6, got {}",
        posterior_mean
    );

    let prior_conc = bd.prior_alpha.unwrap() + bd.prior_beta.unwrap();
    let post_conc = bd.posterior_alpha.unwrap() + bd.posterior_beta.unwrap();
    assert!(
        post_conc > prior_conc * 2.0,
        "Concentration should increase with evidence: prior={}, posterior={}",
        prior_conc,
        post_conc
    );
}

#[test]
fn test_bayesian_backward_compat_composite() {
    let bd = ConfidenceBreakdown {
        extraction: 0.8,
        source_credibility: 0.7,
        corroboration: 0.9,
        recency: 0.6,
        prior_alpha: None,
        prior_beta: None,
        posterior_alpha: None,
        posterior_beta: None,
    };
    let expected = 0.8 * 0.2 + 0.7 * 0.35 + 0.9 * 0.35 + 0.6 * 0.1;
    assert!((bd.composite() - expected).abs() < 0.001);
}

#[test]
fn test_list_contentions_for_narrative_dedupes_and_filters() {
    let hg = make_hg();

    // Two situations in narrative "nf" + one in "other".
    let sit_nf_a = make_situation(&hg);
    let sit_nf_b = make_situation(&hg);
    let sit_other = make_situation(&hg);
    hg.update_situation(&sit_nf_a, |s| s.narrative_id = Some("nf".into()))
        .unwrap();
    hg.update_situation(&sit_nf_b, |s| s.narrative_id = Some("nf".into()))
        .unwrap();
    hg.update_situation(&sit_other, |s| s.narrative_id = Some("other".into()))
        .unwrap();

    // Two contentions: one inside nf (should surface once, not twice),
    // one straddling narratives (should be filtered out).
    hg.add_contention(ContentionLink {
        situation_a: sit_nf_a,
        situation_b: sit_nf_b,
        contention_type: ContentionType::DirectContradiction,
        description: Some("conflict".into()),
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    })
    .unwrap();
    hg.add_contention(ContentionLink {
        situation_a: sit_nf_a,
        situation_b: sit_other,
        contention_type: ContentionType::AttributionDisagreement,
        description: Some("cross-narrative".into()),
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    })
    .unwrap();

    let nf_contentions = hg.list_contentions_for_narrative("nf").unwrap();
    assert_eq!(
        nf_contentions.len(),
        1,
        "dedupe fwd/rev + drop cross-narrative"
    );
    assert_eq!(
        nf_contentions[0].contention_type,
        ContentionType::DirectContradiction
    );

    let empty = hg.list_contentions_for_narrative("nobody").unwrap();
    assert!(empty.is_empty());
}

// ── Phase 2 aggregator wiring ──────────────────────────────────────────────────

#[test]
fn test_recompute_confidence_none_aggregator_is_bit_identical() {
    use crate::fuzzy::aggregation::AggregatorKind;
    let hg = make_hg();
    let extraction_conf = 0.7;
    let entity_id = make_entity(&hg, extraction_conf);

    let s = make_source("S1", 0.9);
    let sid = s.id;
    hg.create_source(s).unwrap();
    hg.add_attribution(make_attribution(sid, entity_id, AttributionTarget::Entity))
        .unwrap();

    let default = hg.recompute_confidence(&entity_id, extraction_conf).unwrap();
    let via_none = hg
        .recompute_confidence_with_aggregator(&entity_id, extraction_conf, None)
        .unwrap();
    assert_eq!(default.posterior_alpha, via_none.posterior_alpha);
    assert_eq!(default.posterior_beta, via_none.posterior_beta);
    // The two paths must produce the exact same composite.
    let _ = AggregatorKind::Mean;
    assert!((default.composite() - via_none.composite()).abs() < 1e-9);
}

#[test]
fn test_recompute_confidence_mean_aggregator_differs() {
    use crate::fuzzy::aggregation::AggregatorKind;
    let hg = make_hg();
    let extraction_conf = 0.2;
    let entity_id = make_entity(&hg, extraction_conf);

    // Two sources with contrasting trust.
    let s1 = make_source("Trusted", 0.9);
    let s2 = make_source("Doubtful", 0.2);
    let s1_id = s1.id;
    let s2_id = s2.id;
    hg.create_source(s1).unwrap();
    hg.create_source(s2).unwrap();
    hg.add_attribution(make_attribution(s1_id, entity_id, AttributionTarget::Entity))
        .unwrap();
    hg.add_attribution(make_attribution(s2_id, entity_id, AttributionTarget::Entity))
        .unwrap();

    let bayesian = hg.recompute_confidence(&entity_id, extraction_conf).unwrap();
    let aggregator = hg
        .recompute_confidence_with_aggregator(
            &entity_id,
            extraction_conf,
            Some(AggregatorKind::Mean),
        )
        .unwrap();

    // Aggregator path must be well-formed (no NaN, in [0, 1]).
    let agg_composite = aggregator.composite();
    assert!(agg_composite.is_finite());
    assert!((0.0..=1.0).contains(&agg_composite));
    // Demonstrably different from the pure-Bayesian path for this fixture.
    assert!(
        (bayesian.composite() - aggregator.composite()).abs() > 1e-6,
        "aggregator path should diverge from Bayesian default"
    );
}
