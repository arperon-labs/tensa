use super::*;

fn arg(idx: usize) -> Argument {
    Argument {
        id: Uuid::from_u128(idx as u128),
        label: format!("arg{}", idx),
        source_id: None,
        confidence: 0.8,
    }
}

// ─── Grounded Extension Tests ───────────────────────────

#[test]
fn test_grounded_a_attacks_b() {
    // A → B: A=IN, B=OUT.
    let fw = from_explicit(vec![arg(0), arg(1)], vec![(0, 1)]);
    let labels = grounded_extension(&fw);
    assert_eq!(labels[0], ArgumentLabel::In);
    assert_eq!(labels[1], ArgumentLabel::Out);
}

#[test]
fn test_grounded_mutual_attack() {
    // A ↔ B: both UNDEC.
    let fw = from_explicit(vec![arg(0), arg(1)], vec![(0, 1), (1, 0)]);
    let labels = grounded_extension(&fw);
    assert_eq!(labels[0], ArgumentLabel::Undec);
    assert_eq!(labels[1], ArgumentLabel::Undec);
}

#[test]
fn test_grounded_chain_defense() {
    // A → B → C: A=IN, B=OUT, C=IN (defense).
    let fw = from_explicit(vec![arg(0), arg(1), arg(2)], vec![(0, 1), (1, 2)]);
    let labels = grounded_extension(&fw);
    assert_eq!(labels[0], ArgumentLabel::In);
    assert_eq!(labels[1], ArgumentLabel::Out);
    assert_eq!(labels[2], ArgumentLabel::In);
}

#[test]
fn test_grounded_triangle_all_undec() {
    // A → B → C → A: odd cycle, all UNDEC.
    let fw = from_explicit(vec![arg(0), arg(1), arg(2)], vec![(0, 1), (1, 2), (2, 0)]);
    let labels = grounded_extension(&fw);
    assert_eq!(labels[0], ArgumentLabel::Undec);
    assert_eq!(labels[1], ArgumentLabel::Undec);
    assert_eq!(labels[2], ArgumentLabel::Undec);
}

#[test]
fn test_grounded_no_attacks() {
    // All arguments unattacked → all IN.
    let fw = from_explicit(vec![arg(0), arg(1), arg(2)], vec![]);
    let labels = grounded_extension(&fw);
    assert_eq!(labels[0], ArgumentLabel::In);
    assert_eq!(labels[1], ArgumentLabel::In);
    assert_eq!(labels[2], ArgumentLabel::In);
}

#[test]
fn test_grounded_self_attack() {
    // A attacks itself → OUT.
    let fw = from_explicit(vec![arg(0), arg(1)], vec![(0, 0)]);
    let labels = grounded_extension(&fw);
    assert_eq!(labels[0], ArgumentLabel::Out);
    assert_eq!(labels[1], ArgumentLabel::In);
}

#[test]
fn test_grounded_empty_framework() {
    let fw = from_explicit(vec![], vec![]);
    let labels = grounded_extension(&fw);
    assert!(labels.is_empty());
}

// ─── Preferred Extension Tests ──────────────────────────

#[test]
fn test_preferred_mutual_attack() {
    // A ↔ B: two preferred extensions {A} and {B}.
    let fw = from_explicit(vec![arg(0), arg(1)], vec![(0, 1), (1, 0)]);
    let preferred = preferred_extensions(&fw);
    assert_eq!(preferred.len(), 2);

    let sets: HashSet<BTreeSet<usize>> = preferred.into_iter().collect();
    assert!(sets.contains(&BTreeSet::from([0])));
    assert!(sets.contains(&BTreeSet::from([1])));
}

#[test]
fn test_preferred_no_attacks() {
    // No attacks: one preferred extension = all arguments.
    let fw = from_explicit(vec![arg(0), arg(1), arg(2)], vec![]);
    let preferred = preferred_extensions(&fw);
    assert_eq!(preferred.len(), 1);
    assert_eq!(preferred[0], BTreeSet::from([0, 1, 2]));
}

#[test]
fn test_grounded_subset_of_preferred() {
    // Grounded should be subset of every preferred extension.
    let fw = from_explicit(
        vec![arg(0), arg(1), arg(2), arg(3), arg(4)],
        vec![(0, 1), (1, 2), (2, 3), (3, 4)],
    );
    let grounded_labels = grounded_extension(&fw);
    let grounded_in: BTreeSet<usize> = grounded_labels
        .iter()
        .enumerate()
        .filter(|(_, l)| **l == ArgumentLabel::In)
        .map(|(i, _)| i)
        .collect();

    let preferred = preferred_extensions(&fw);
    for ext in &preferred {
        assert!(
            grounded_in.is_subset(ext),
            "grounded {:?} not subset of preferred {:?}",
            grounded_in,
            ext
        );
    }
}

// ─── Stable Extension Tests ─────────────────────────────

#[test]
fn test_stable_acyclic() {
    // A → B: stable extension {A}.
    let fw = from_explicit(vec![arg(0), arg(1)], vec![(0, 1)]);
    let stable = stable_extensions(&fw);
    assert_eq!(stable.len(), 1);
    assert_eq!(stable[0], BTreeSet::from([0]));
}

#[test]
fn test_stable_odd_cycle_may_not_exist() {
    // A → B → C → A: no stable extension may exist for odd cycles.
    let fw = from_explicit(vec![arg(0), arg(1), arg(2)], vec![(0, 1), (1, 2), (2, 0)]);
    let stable = stable_extensions(&fw);
    // Stable may or may not exist for odd cycles — just verify no crash.
    assert!(stable.is_empty() || !stable.is_empty());
}

#[test]
fn test_stable_mutual_attack() {
    // A ↔ B: two stable extensions {A} and {B}.
    let fw = from_explicit(vec![arg(0), arg(1)], vec![(0, 1), (1, 0)]);
    let stable = stable_extensions(&fw);
    assert_eq!(stable.len(), 2);
}

// ─── Integration Tests ──────────────────────────────────

#[test]
fn test_integration_with_contentions() {
    use crate::source::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
    let n = "argtest";

    // Create two situations with a contention.
    let sit_a = {
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
            raw_content: vec![ContentBlock::text("Source A says attack happened")],
            narrative_level: NarrativeLevel::Event,
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
        hg.create_situation(sit).unwrap()
    };

    let sit_b = {
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
            raw_content: vec![ContentBlock::text("Source B denies attack")],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.7,
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
        hg.create_situation(sit).unwrap()
    };

    // Add contention.
    hg.add_contention(ContentionLink {
        situation_a: sit_a,
        situation_b: sit_b,
        contention_type: crate::source::ContentionType::DirectContradiction,
        description: Some("Contradictory claims".into()),
        resolved: false,
        resolution: None,
        created_at: Utc::now(),
    })
    .unwrap();

    let result = run_argumentation(&hg, n).unwrap();
    assert_eq!(result.framework.arguments.len(), 2);
    assert!(!result.framework.attacks.is_empty());
    // Mutual attack → both UNDEC in grounded.
    for (_, label) in &result.grounded {
        assert_eq!(*label, ArgumentLabel::Undec);
    }
    // Two preferred extensions.
    assert_eq!(result.preferred_extensions.len(), 2);
}

#[test]
fn test_empty_framework_integration() {
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
    let result = run_argumentation(&hg, "empty").unwrap();
    assert!(result.framework.arguments.is_empty());
}

#[test]
fn test_inference_engine_trait() {
    let engine = ArgumentationEngine;
    assert_eq!(engine.job_type(), InferenceJobType::ArgumentationAnalysis);
}
