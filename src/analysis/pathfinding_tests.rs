use super::*;
use crate::analysis::test_helpers::*;

// ── Dijkstra ───────────────────────────────────────────────

#[test]
fn test_dijkstra_simple() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, b, s1);
    link(&hg, b, s2);
    link(&hg, c, s2);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let c_idx = graph.entities.iter().position(|&e| e == c).unwrap();

    let result = dijkstra(&graph, a_idx, c_idx);
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.path.len(), 3); // A -> B -> C
    assert!(path.total_weight > 0.0);
}

#[test]
fn test_dijkstra_unreachable() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    // No connection
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();

    let result = dijkstra(&graph, a_idx, b_idx);
    assert!(result.is_none());
}

#[test]
fn test_dijkstra_weighted() {
    // Prefer path through high-weight edge (more co-participations = shorter distance)
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    // A-B share 3 situations (weight 3), B-C share 1 situation (weight 1)
    for _ in 0..3 {
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
    }
    let s2 = add_situation(&hg, "n1");
    link(&hg, b, s2);
    link(&hg, c, s2);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let c_idx = graph.entities.iter().position(|&e| e == c).unwrap();

    let result = dijkstra(&graph, a_idx, c_idx).unwrap();
    // A->B edge weight = 1/3, B->C edge weight = 1/1, total ~1.33
    assert!(result.total_weight > 1.0 && result.total_weight < 2.0);
}

// ── Yen's K-Shortest ───────────────────────────────────────

#[test]
fn test_yen_k_shortest() {
    // Diamond: A-B, A-C, B-D, C-D
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let d = add_entity(&hg, "D", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    let s3 = add_situation(&hg, "n1");
    let s4 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, b, s1); // A-B
    link(&hg, a, s2);
    link(&hg, c, s2); // A-C
    link(&hg, b, s3);
    link(&hg, d, s3); // B-D
    link(&hg, c, s4);
    link(&hg, d, s4); // C-D

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let d_idx = graph.entities.iter().position(|&e| e == d).unwrap();

    let paths = yen_k_shortest(&graph, a_idx, d_idx, 3);
    assert!(paths.len() >= 2, "Diamond should have at least 2 paths A→D");
    // First path should be shortest
    assert!(paths[0].total_weight <= paths[1].total_weight);
}

#[test]
fn test_yen_fewer_than_k() {
    // Chain: only 1 path
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();

    let paths = yen_k_shortest(&graph, a_idx, b_idx, 5);
    assert_eq!(paths.len(), 1, "Chain has exactly 1 path");
}

#[test]
fn test_yen_weighted() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let d = add_entity(&hg, "D", "n1");
    // A-B heavy (3 situations), A-C light (1), B-D light, C-D heavy
    for _ in 0..3 {
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
    }
    let s1 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, c, s1);
    let s2 = add_situation(&hg, "n1");
    link(&hg, b, s2);
    link(&hg, d, s2);
    for _ in 0..3 {
        let s = add_situation(&hg, "n1");
        link(&hg, c, s);
        link(&hg, d, s);
    }

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let d_idx = graph.entities.iter().position(|&e| e == d).unwrap();

    let paths = yen_k_shortest(&graph, a_idx, d_idx, 2);
    assert_eq!(paths.len(), 2);
    // Both paths should have positive weight
    for p in &paths {
        assert!(p.total_weight > 0.0);
    }
}

// ── Narrative Diameter ─────────────────────────────────────

#[test]
fn test_narrative_diameter_linear() {
    // Linear chain: S1 → S2 → S3
    let hg = make_hg();
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    let s3 = add_situation(&hg, "n1");
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
    let result = narrative_diameter(&dag).unwrap();
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.path.len(), 3);
    assert!((path.total_weight - 2.0).abs() < 0.01);
}

#[test]
fn test_narrative_diameter_diamond() {
    // Diamond: S1 → S2, S1 → S3, S2 → S4, S3 → S4
    let hg = make_hg();
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    let s3 = add_situation(&hg, "n1");
    let s4 = add_situation(&hg, "n1");
    for (from, to) in [(s1, s2), (s1, s3), (s2, s4), (s3, s4)] {
        hg.add_causal_link(crate::types::CausalLink {
            from_situation: from,
            to_situation: to,
            mechanism: None,
            strength: 1.0,
            causal_type: crate::types::CausalType::Contributing,
            maturity: crate::types::MaturityLevel::Validated,
        })
        .unwrap();
    }

    let dag = graph_projection::build_causal_dag(&hg, "n1").unwrap();
    let result = narrative_diameter(&dag).unwrap();
    assert!(result.is_some());
    let path = result.unwrap();
    // Longest path: S1→S2→S4 or S1→S3→S4, length = 2
    assert!((path.total_weight - 2.0).abs() < 0.01);
}

#[test]
fn test_narrative_diameter_single_event() {
    let hg = make_hg();
    let _s = add_situation(&hg, "n1");
    let dag = graph_projection::build_causal_dag(&hg, "n1").unwrap();
    let result = narrative_diameter(&dag).unwrap();
    assert!(result.is_none(), "Single event has no causal path");
}

// ── Max-Flow ───────────────────────────────────────────────

#[test]
fn test_max_flow_simple() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();

    let (flow, _) = max_flow(&graph, a_idx, b_idx);
    assert!(flow > 0.0, "Direct connection should have positive flow");
}

#[test]
fn test_max_flow_bottleneck() {
    // A-B (weight 3), B-C (weight 1): bottleneck is B-C
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    for _ in 0..3 {
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
    }
    let s2 = add_situation(&hg, "n1");
    link(&hg, b, s2);
    link(&hg, c, s2);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let c_idx = graph.entities.iter().position(|&e| e == c).unwrap();

    let (flow, _cuts) = max_flow(&graph, a_idx, c_idx);
    // Bottleneck at B-C (capacity 1)
    assert!(
        (flow - 1.0).abs() < 0.01,
        "Flow should be 1.0 (B-C bottleneck), got {}",
        flow
    );
}

#[test]
fn test_min_cut() {
    // Same as bottleneck test — min-cut should include the B-C edge
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    for _ in 0..3 {
        let s = add_situation(&hg, "n1");
        link(&hg, a, s);
        link(&hg, b, s);
    }
    let s2 = add_situation(&hg, "n1");
    link(&hg, b, s2);
    link(&hg, c, s2);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let c_idx = graph.entities.iter().position(|&e| e == c).unwrap();

    let (_, cut_edges) = max_flow(&graph, a_idx, c_idx);
    assert!(
        !cut_edges.is_empty(),
        "Min-cut should have at least one edge"
    );
}

// ── PCST ───────────────────────────────────────────────────

#[test]
fn test_pcst_connects_prizes() {
    // Chain: A-B-C-D. Prize nodes = {A, D}. PCST should include B, C.
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let d = add_entity(&hg, "D", "n1");
    let s1 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, b, s1);
    let s2 = add_situation(&hg, "n1");
    link(&hg, b, s2);
    link(&hg, c, s2);
    let s3 = add_situation(&hg, "n1");
    link(&hg, c, s3);
    link(&hg, d, s3);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let d_idx = graph.entities.iter().position(|&e| e == d).unwrap();

    let tree = pcst_approximation(&graph, &[a_idx, d_idx]);
    assert!(tree.contains(&a_idx), "Prize A should be in tree");
    assert!(tree.contains(&d_idx), "Prize D should be in tree");
    assert!(tree.len() >= 3, "Tree should include intermediate nodes");
}

#[test]
fn test_pcst_minimal_cost() {
    // Single prize node → just that node
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let tree = pcst_approximation(&graph, &[a_idx]);
    assert_eq!(tree.len(), 1);
}
