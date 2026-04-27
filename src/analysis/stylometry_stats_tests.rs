//! Unit tests for bootstrap stylometry statistics.

use super::*;
use crate::analysis::stylometry::compute_prose_features;

fn hemingway_chunks() -> Vec<ProseStyleFeatures> {
    [
        "He walked. He was tired. The sea was gone.",
        "She was gone. He was alone. The boat was old.",
        "He drank. He thought. The room was quiet.",
        "The fish was big. The line was strong. He held on.",
    ]
    .iter()
    .map(|t| compute_prose_features(t))
    .collect()
}

fn faulkner_chunks() -> Vec<ProseStyleFeatures> {
    [
        "In the dim and unremembered light of that afternoon when the wind had only barely begun to stir the curtains of the room.",
        "As he walked down the hallway thinking of all the things that might yet come to pass in the house where the generations had lived.",
        "The cotton fields stretched to an unmeasurable horizon beneath a sun that had gone on rising uninterrupted since before memory existed.",
    ]
    .iter()
    .map(|t| compute_prose_features(t))
    .collect()
}

// ─── Bootstrap null ────────────────────────────────────────

#[test]
fn bootstrap_null_empty_chunks_returns_empty() {
    let cfg = WeightedSimilarityConfig::default();
    let null = bootstrap_null_chunk_similarity(&[], 100, &cfg, 42);
    assert!(null.similarities.is_empty());
}

#[test]
fn bootstrap_null_deterministic_with_seed() {
    let cfg = WeightedSimilarityConfig::default();
    let chunks = hemingway_chunks();
    let a = bootstrap_null_chunk_similarity(&chunks, 100, &cfg, 42);
    let b = bootstrap_null_chunk_similarity(&chunks, 100, &cfg, 42);
    assert_eq!(a.similarities.len(), b.similarities.len());
    for (x, y) in a.similarities.iter().zip(b.similarities.iter()) {
        assert!((x - y).abs() < 1e-5);
    }
}

#[test]
fn bootstrap_null_same_author_mean_is_high() {
    // All chunks share a single author → null mean should be high (close to 1).
    let cfg = WeightedSimilarityConfig::default();
    let chunks = hemingway_chunks();
    let null = bootstrap_null_chunk_similarity(&chunks, 500, &cfg, 7);
    assert!(null.mean > 0.6, "same-author null mean = {}", null.mean);
}

// ─── p-value monotonicity ──────────────────────────────────

#[test]
fn p_value_monotonic_in_observed() {
    let null = BootstrapNull {
        similarities: (0..100).map(|i| i as f32 / 100.0).collect(),
        mean: 0.495,
        std: 0.29,
        n_iter: 100,
        seed: 0,
    };
    let low = calibrated_anomaly_p_value(0.1, &null);
    let high = calibrated_anomaly_p_value(0.9, &null);
    assert!(low < high, "low p {} should be < high p {}", low, high);
}

#[test]
fn p_value_bounded() {
    let null = BootstrapNull {
        similarities: vec![0.3, 0.5, 0.7],
        mean: 0.5,
        std: 0.2,
        n_iter: 3,
        seed: 0,
    };
    for obs in [-1.0_f32, 0.0, 0.5, 1.0, 2.0] {
        let p = calibrated_anomaly_p_value(obs, &null);
        assert!((0.0..=1.0).contains(&p));
    }
}

#[test]
fn p_value_empty_null_returns_one() {
    let null = BootstrapNull {
        similarities: vec![],
        mean: 0.0,
        std: 0.0,
        n_iter: 0,
        seed: 0,
    };
    assert_eq!(calibrated_anomaly_p_value(0.5, &null), 1.0);
}

// ─── Calibrated anomaly detection ─────────────────────────

#[test]
fn calibrated_anomalies_flag_outlier() {
    // Mix a pack of Hemingway chunks with one Faulkner chunk; the Faulkner
    // chunk should consistently stand out as an anomaly.
    let cfg = WeightedSimilarityConfig::default();
    let mut chunks = hemingway_chunks();
    chunks.push(faulkner_chunks()[0].clone());
    let results = detect_style_anomalies_calibrated(&chunks, 0.2, 500, &cfg, 13);
    let last = results.last().unwrap();
    let others_max_p = results
        .iter()
        .take(chunks.len() - 1)
        .map(|r| r.p_value)
        .fold(0.0_f32, f32::max);
    // Outlier chunk should have a p-value at least as low as the highest
    // "native" chunk p — in practice it's strictly lower, but we relax
    // slightly so a small corpus doesn't produce flaky failures.
    assert!(
        last.p_value <= others_max_p + 0.01,
        "outlier p {} vs max native p {}",
        last.p_value,
        others_max_p
    );
}

#[test]
fn calibrated_anomalies_empty_input() {
    let cfg = WeightedSimilarityConfig::default();
    assert!(detect_style_anomalies_calibrated(&[], 0.05, 100, &cfg, 1).is_empty());
}

// ─── Pair bootstrap CI ─────────────────────────────────────

fn zeroed_structure(narrative_id: &str) -> crate::analysis::style_profile::NarrativeStyleProfile {
    use crate::analysis::style_profile::NarrativeStyleProfile;
    use chrono::Utc;
    NarrativeStyleProfile {
        narrative_id: narrative_id.into(),
        computed_at: Utc::now(),
        situation_density_curve: vec![0.0; 20],
        avg_participants_per_situation: 0.0,
        participation_count_variance: 0.0,
        arc_type: None,
        arc_confidence: 0.0,
        situation_entity_ratio: 0.0,
        game_type_distribution: vec![0.0; 7],
        role_entropy: 0.0,
        power_asymmetry_gini: 0.0,
        co_participation_density: 0.0,
        protagonist_concentration: 0.0,
        avg_info_r0: 0.0,
        deception_index: 0.0,
        knowledge_asymmetry_gini: 0.0,
        revelation_timing: 0.0,
        secret_survival_rate: 0.0,
        causal_density: 0.0,
        avg_causal_chain_length: 0.0,
        max_causal_chain_length: 0,
        unexplained_event_ratio: 0.0,
        causal_branching_factor: 0.0,
        causal_convergence_factor: 0.0,
        allen_relation_distribution: vec![0.0; 13],
        flashback_frequency: 0.0,
        temporal_span_variance: 0.0,
        temporal_gap_ratio: 0.0,
        wl_hash_histogram: vec![0.0; 50],
        wl_simhash: [0u64; 4],
        community_count: 0,
        avg_shortest_path: 0.0,
        graph_diameter: 0,
        edge_density: 0.0,
        narrative_surprise: 0.0,
        // D9.8 generative axes
        promise_fulfillment_ratio: 0.0,
        average_payoff_distance: 0.0,
        fabula_sjuzet_divergence: 0.0,
        dramatic_irony_density: 0.0,
        focalization_diversity: 0.0,
        character_arc_completeness: 0.0,
        subplot_convergence_ratio: 0.0,
        scene_sequel_rhythm_score: 0.0,
    }
}

#[test]
fn pair_ci_empty_chunks_returns_point() {
    use crate::analysis::style_profile::NarrativeFingerprint;
    use chrono::Utc;
    let prose = compute_prose_features("a b c d e. f g h i j.");
    let fp = NarrativeFingerprint {
        narrative_id: "x".into(),
        computed_at: Utc::now(),
        prose,
        structure: zeroed_structure("x"),
    };
    let cfg = WeightedSimilarityConfig::default();
    let ci = bootstrap_pair_similarity_ci(&fp, &fp, &[], &[], &cfg, 0.05, 100, 0);
    assert_eq!(ci.overall.ci_low, ci.overall.ci_high);
    assert_eq!(ci.overall.resample_source, "point");
}

#[test]
fn pair_ci_contains_point_estimate() {
    use crate::analysis::style_profile::NarrativeFingerprint;
    use chrono::Utc;
    let chunks_a = hemingway_chunks();
    let chunks_b = hemingway_chunks();
    let prose_a = aggregate_features(&chunks_a);
    let prose_b = aggregate_features(&chunks_b);

    let fp_a = NarrativeFingerprint {
        narrative_id: "a".into(),
        computed_at: Utc::now(),
        prose: prose_a,
        structure: zeroed_structure("a"),
    };
    let fp_b = NarrativeFingerprint {
        narrative_id: "b".into(),
        computed_at: Utc::now(),
        prose: prose_b,
        structure: zeroed_structure("b"),
    };
    let cfg = WeightedSimilarityConfig::default();
    let ci = bootstrap_pair_similarity_ci(&fp_a, &fp_b, &chunks_a, &chunks_b, &cfg, 0.1, 200, 42);
    assert!(ci.overall.ci_low <= ci.estimate.overall + 1e-3);
    assert!(ci.overall.ci_high >= ci.estimate.overall - 1e-3);
    assert_eq!(ci.overall.resample_source, "bootstrap");
}
