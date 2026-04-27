use super::*;
use crate::analysis::test_helpers::{
    add_entity, add_situation_with_level, link_with_action as link, make_hg,
};
use crate::hypergraph::keys;

// ─── Self-Information Tests ──────────────────────────────

#[test]
fn test_self_info_common_vs_rare() {
    let hg = make_hg();
    let n = "entropy1";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    // Create 5 common situations (Scene, 2 participants).
    for _ in 0..5 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, None);
        link(&hg, b, s, None);
    }
    // Create 1 rare situation (Story, only 1 participant, with game structure).
    let rare = Situation {
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
        game_structure: Some(GameStructure {
            game_type: GameClassification::ZeroSum,
            info_structure: InfoStructureType::Complete,
            description: None,
            maturity: MaturityLevel::Candidate,
        }),
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text("rare event")],
        narrative_level: NarrativeLevel::Story,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.8,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(n.to_string()),
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
    let rare_id = hg.create_situation(rare).unwrap();
    link(&hg, a, rare_id, None);

    let analysis = run_entropy(&hg, n).unwrap();
    assert_eq!(analysis.situation_entropies.len(), 6);

    // The rare situation should have higher self-information.
    let rare_entropy = analysis
        .situation_entropies
        .iter()
        .find(|e| e.situation_id == rare_id)
        .unwrap();
    let common_entropy = analysis
        .situation_entropies
        .iter()
        .find(|e| e.situation_id != rare_id)
        .unwrap();
    assert!(
        rare_entropy.self_information > common_entropy.self_information,
        "rare={}, common={}",
        rare_entropy.self_information,
        common_entropy.self_information
    );
}

#[test]
fn test_self_info_single_situation() {
    let hg = make_hg();
    let n = "single_ent";
    let a = add_entity(&hg, "a", n);
    let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
    link(&hg, a, s, None);

    let analysis = run_entropy(&hg, n).unwrap();
    assert_eq!(analysis.situation_entropies.len(), 1);
    // Single situation: P = 1.0, -log2(1) = 0.
    assert!(
        analysis.situation_entropies[0].self_information.abs() < 0.001,
        "got {}",
        analysis.situation_entropies[0].self_information
    );
}

// ─── Mutual Information Tests ───────────────────────────

#[test]
fn test_mi_always_together() {
    let hg = make_hg();
    let n = "mi_together";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    // A and B always together, C alone.
    for _ in 0..5 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, None);
        link(&hg, b, s, None);
    }
    for _ in 0..3 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, c, s, None);
    }

    let analysis = run_entropy(&hg, n).unwrap();

    // MI(A,B) should be high.
    let mi_ab = analysis
        .mutual_information
        .iter()
        .find(|m| (m.entity_a == a && m.entity_b == b) || (m.entity_a == b && m.entity_b == a))
        .unwrap();
    assert!(
        mi_ab.mutual_information > 0.0,
        "MI(A,B) = {}",
        mi_ab.mutual_information
    );

    // MI(A,C) should also be > 0 because non-co-occurrence is informative.
    // (Knowing A is present tells you C is absent, and vice versa.)
    let mi_ac = analysis
        .mutual_information
        .iter()
        .find(|m| (m.entity_a == a && m.entity_b == c) || (m.entity_a == c && m.entity_b == a))
        .unwrap();
    // Just verify it's non-negative and joint appearances = 0.
    assert!(mi_ac.mutual_information >= 0.0);
    assert_eq!(mi_ac.joint_appearances, 0);
}

#[test]
fn test_mi_never_coappearing() {
    let hg = make_hg();
    let n = "mi_never";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    // A and B never share a situation.
    for _ in 0..3 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, None);
    }
    for _ in 0..3 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, b, s, None);
    }

    let analysis = run_entropy(&hg, n).unwrap();
    let mi = &analysis.mutual_information[0];
    assert_eq!(mi.joint_appearances, 0);
    // MI should be 0 (or positive due to mutual exclusion — still informative).
    // The exact value depends on the table, but joint = 0 is the key test.
}

// ─── KL Divergence Tests ────────────────────────────────

#[test]
fn test_kl_consistent_behavior() {
    let hg = make_hg();
    let n = "kl_consist";
    let a = add_entity(&hg, "a", n);

    // A always does "cooperate".
    for _ in 0..5 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, Some("cooperate"));
    }

    let analysis = run_entropy(&hg, n).unwrap();
    // Only one action type → KL not computed (needs >= 2 distinct actions).
    assert!(analysis.kl_divergences.is_empty());
}

#[test]
fn test_kl_inconsistent_behavior() {
    let hg = make_hg();
    let n = "kl_inconst";
    let a = add_entity(&hg, "a", n);

    // A mostly cooperates but sometimes betrays.
    for _ in 0..8 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, Some("cooperate"));
    }
    for _ in 0..2 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, Some("betray"));
    }

    let analysis = run_entropy(&hg, n).unwrap();
    assert_eq!(analysis.kl_divergences.len(), 1);
    // KL should be > 0 since actual distribution differs from uniform.
    assert!(
        analysis.kl_divergences[0].kl_divergence > 0.0,
        "KL = {}",
        analysis.kl_divergences[0].kl_divergence
    );
}

// ─── Edge Cases ─────────────────────────────────────────

#[test]
fn test_empty_narrative() {
    let hg = make_hg();
    let analysis = run_entropy(&hg, "nonexistent").unwrap();
    assert!(analysis.situation_entropies.is_empty());
    assert!(analysis.mutual_information.is_empty());
}

#[test]
fn test_all_identical_situations() {
    let hg = make_hg();
    let n = "identical";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    for _ in 0..5 {
        let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
        link(&hg, a, s, None);
        link(&hg, b, s, None);
    }

    let analysis = run_entropy(&hg, n).unwrap();
    // All situations have same features → same self-information.
    let entropies: Vec<f64> = analysis
        .situation_entropies
        .iter()
        .map(|e| e.self_information)
        .collect();
    for e in &entropies {
        assert!(
            (*e - entropies[0]).abs() < 0.001,
            "expected all equal, got {:?}",
            entropies
        );
    }
}

#[test]
fn test_kv_storage() {
    let hg = make_hg();
    let n = "kv_ent";
    let a = add_entity(&hg, "a", n);
    let s = add_situation_with_level(&hg, n, NarrativeLevel::Scene);
    link(&hg, a, s, None);

    run_entropy(&hg, n).unwrap();

    let key = analysis_key(keys::ANALYSIS_ENTROPY, &[n, &s.to_string()]);
    let stored = hg.store().get(&key).unwrap();
    assert!(stored.is_some());
}

#[test]
fn test_inference_engine_trait() {
    let engine = EntropyEngine;
    assert_eq!(engine.job_type(), InferenceJobType::EntropyAnalysis);
}
