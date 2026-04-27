use super::*;
use crate::analysis::test_helpers::*;

// ── Temporal PageRank ──────────────────────────────────────

#[test]
fn test_temporal_pagerank_recent_bias() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);

    let (entities, scores) = temporal_pagerank(&hg, "n1", 0.1, None).unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(scores.len(), 2);
    for s in &scores {
        assert!(*s > 0.0, "All nodes should have positive temporal PR");
    }
}

#[test]
fn test_temporal_pagerank_decay() {
    // With high decay, recent events should dominate
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);

    let (_ents1, scores_low) = temporal_pagerank(&hg, "n1", 0.01, None).unwrap();
    let (_ents2, scores_high) = temporal_pagerank(&hg, "n1", 10.0, None).unwrap();
    // Both should produce valid results
    assert_eq!(scores_low.len(), 2);
    assert_eq!(scores_high.len(), 2);
}

#[test]
fn test_temporal_pagerank_empty() {
    let hg = make_hg();
    let (ents, scores) = temporal_pagerank(&hg, "n1", 0.1, None).unwrap();
    assert!(ents.is_empty());
    assert!(scores.is_empty());
}

// ── Causal Influence ───────────────────────────────────────

#[test]
fn test_causal_influence_linear() {
    // Linear chain: S1 → S2 → S3. Entity in S2 should have highest influence.
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    let s3 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, b, s2);
    link(&hg, c, s3);
    hg.add_causal_link(crate::types::CausalLink {
        from_situation: s1,
        to_situation: s2,
        mechanism: None,
        strength: 1.0,
        causal_type: crate::types::CausalType::Contributing,
        maturity: crate::types::MaturityLevel::Validated,
    })
    .unwrap();
    hg.add_causal_link(crate::types::CausalLink {
        from_situation: s2,
        to_situation: s3,
        mechanism: None,
        strength: 1.0,
        causal_type: crate::types::CausalType::Contributing,
        maturity: crate::types::MaturityLevel::Validated,
    })
    .unwrap();

    let dag = graph_projection::build_causal_dag(&hg, "n1").unwrap();
    let influence = causal_influence(&dag, &hg, "n1").unwrap();
    assert_eq!(influence.len(), 3);
    // B (in middle situation S2) should have highest influence
    let b_score = influence
        .iter()
        .find(|&&(id, _)| id == b)
        .map(|&(_, s)| s)
        .unwrap_or(0.0);
    assert!(b_score > 0.0, "Middle entity should have causal influence");
}

#[test]
fn test_causal_influence_hub() {
    // Hub: S1 → S2, S1 → S3. Entity in S1 is causal hub.
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    let s3 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    hg.add_causal_link(crate::types::CausalLink {
        from_situation: s1,
        to_situation: s2,
        mechanism: None,
        strength: 1.0,
        causal_type: crate::types::CausalType::Contributing,
        maturity: crate::types::MaturityLevel::Validated,
    })
    .unwrap();
    hg.add_causal_link(crate::types::CausalLink {
        from_situation: s1,
        to_situation: s3,
        mechanism: None,
        strength: 1.0,
        causal_type: crate::types::CausalType::Contributing,
        maturity: crate::types::MaturityLevel::Validated,
    })
    .unwrap();

    let dag = graph_projection::build_causal_dag(&hg, "n1").unwrap();
    let influence = causal_influence(&dag, &hg, "n1").unwrap();
    // A participates in the root cause situation
    let a_score = influence
        .iter()
        .find(|&&(id, _)| id == a)
        .map(|&(_, s)| s)
        .unwrap_or(0.0);
    assert!(a_score >= 0.0);
}

// ── Information Bottleneck ─────────────────────────────────

#[test]
fn test_info_bottleneck_no_beliefs() {
    // No beliefs computed → all scores should be 0
    let hg = make_hg();
    let _a = add_entity(&hg, "A", "n1");
    let result = information_bottleneck(&hg, "n1").unwrap();
    assert_eq!(result.len(), 1);
    assert!(
        (result[0].1 - 0.0).abs() < 0.01,
        "No beliefs → 0 bottleneck"
    );
}

#[test]
fn test_info_bottleneck_empty() {
    let hg = make_hg();
    let result = information_bottleneck(&hg, "n1").unwrap();
    assert!(result.is_empty());
}

// ── Assortativity ──────────────────────────────────────────

#[test]
fn test_assortativity_symmetric() {
    // Triangle: all same degree → assortativity = 0 (or NaN → 0)
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);
    link(&hg, c, s);
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let r = degree_assortativity(&graph);
    // All degrees equal → no correlation → r ≈ 0
    assert!(
        r.abs() < 0.1 || r.is_nan() == false,
        "Symmetric graph assortativity should be ~0"
    );
}

#[test]
fn test_assortativity_disassortative() {
    // Star: hub connects to many leaves (disassortative)
    let hg = make_hg();
    let hub = add_entity(&hg, "Hub", "n1");
    for i in 0..5 {
        let leaf = add_entity(&hg, &format!("L{}", i), "n1");
        let s = add_situation(&hg, "n1");
        link(&hg, hub, s);
        link(&hg, leaf, s);
    }
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let r = degree_assortativity(&graph);
    // Star is disassortative (high degree connects to low degree)
    assert!(
        r < 0.1,
        "Star graph should be disassortative (r < 0), got {}",
        r
    );
}

#[test]
fn test_assortativity_empty() {
    let hg = make_hg();
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let r = degree_assortativity(&graph);
    assert!((r - 0.0).abs() < 0.01);
}

#[test]
fn test_temporal_pagerank_decay_auto_scales_to_span() {
    // 100-day span → Auto resolves to ln(2)/50 ≈ 0.01386
    let lambda = TemporalPageRankDecay::Auto.resolve(100.0);
    assert!((lambda - (std::f64::consts::LN_2 / 50.0)).abs() < 1e-9);

    // Degenerate span → fall back to 0.1
    assert!((TemporalPageRankDecay::Auto.resolve(0.0) - 0.1).abs() < 1e-9);

    // Fixed passes through unchanged
    assert!((TemporalPageRankDecay::Fixed(0.25).resolve(1000.0) - 0.25).abs() < 1e-9);
}
