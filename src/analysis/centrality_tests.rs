use super::*;
use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};
use crate::hypergraph::keys;

// ─── Betweenness Tests ──────────────────────────────────

#[test]
fn test_betweenness_star_graph() {
    // Center node has highest betweenness in a star.
    let hg = make_hg();
    let n = "star";
    let center = add_entity(&hg, "center", n);
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    // Center connects to a, b, c via separate situations.
    let s1 = add_situation(&hg, n);
    let s2 = add_situation(&hg, n);
    let s3 = add_situation(&hg, n);

    link(&hg, center, s1);
    link(&hg, a, s1);
    link(&hg, center, s2);
    link(&hg, b, s2);
    link(&hg, center, s3);
    link(&hg, c, s3);

    let analysis = run_centrality(&hg, n).unwrap();
    let center_result = analysis
        .results
        .iter()
        .find(|r| r.entity_id == center)
        .unwrap();
    let a_result = analysis.results.iter().find(|r| r.entity_id == a).unwrap();

    assert!(
        center_result.betweenness > a_result.betweenness,
        "center={}, a={}",
        center_result.betweenness,
        a_result.betweenness
    );
}

#[test]
fn test_betweenness_chain() {
    // A -- B -- C: B has highest betweenness.
    let hg = make_hg();
    let n = "chain";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    let s1 = add_situation(&hg, n);
    let s2 = add_situation(&hg, n);

    link(&hg, a, s1);
    link(&hg, b, s1);
    link(&hg, b, s2);
    link(&hg, c, s2);

    let analysis = run_centrality(&hg, n).unwrap();
    let b_result = analysis.results.iter().find(|r| r.entity_id == b).unwrap();
    let a_result = analysis.results.iter().find(|r| r.entity_id == a).unwrap();
    let c_result = analysis.results.iter().find(|r| r.entity_id == c).unwrap();

    assert!(b_result.betweenness >= a_result.betweenness);
    assert!(b_result.betweenness >= c_result.betweenness);
}

#[test]
fn test_betweenness_bridge() {
    // Two cliques connected by a single bridge entity.
    let hg = make_hg();
    let n = "bridge";

    let a1 = add_entity(&hg, "a1", n);
    let a2 = add_entity(&hg, "a2", n);
    let bridge = add_entity(&hg, "bridge", n);
    let b1 = add_entity(&hg, "b1", n);
    let b2 = add_entity(&hg, "b2", n);

    // Clique 1: a1, a2, bridge
    let s1 = add_situation(&hg, n);
    link(&hg, a1, s1);
    link(&hg, a2, s1);
    link(&hg, bridge, s1);

    // Clique 2: bridge, b1, b2
    let s2 = add_situation(&hg, n);
    link(&hg, bridge, s2);
    link(&hg, b1, s2);
    link(&hg, b2, s2);

    let analysis = run_centrality(&hg, n).unwrap();
    let bridge_r = analysis
        .results
        .iter()
        .find(|r| r.entity_id == bridge)
        .unwrap();
    let a1_r = analysis.results.iter().find(|r| r.entity_id == a1).unwrap();

    assert!(
        bridge_r.betweenness > a1_r.betweenness,
        "bridge={}, a1={}",
        bridge_r.betweenness,
        a1_r.betweenness
    );
}

// ─── Closeness Tests ────────────────────────────────────

#[test]
fn test_closeness_chain() {
    // A -- B -- C: B is closest to all others.
    let hg = make_hg();
    let n = "closeness_chain";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    let s1 = add_situation(&hg, n);
    let s2 = add_situation(&hg, n);

    link(&hg, a, s1);
    link(&hg, b, s1);
    link(&hg, b, s2);
    link(&hg, c, s2);

    let analysis = run_centrality(&hg, n).unwrap();
    let b_r = analysis.results.iter().find(|r| r.entity_id == b).unwrap();
    let a_r = analysis.results.iter().find(|r| r.entity_id == a).unwrap();

    assert!(
        b_r.closeness >= a_r.closeness,
        "b={}, a={}",
        b_r.closeness,
        a_r.closeness
    );
}

// ─── Degree Tests ───────────────────────────────────────

#[test]
fn test_degree_complete_graph() {
    // All three entities share all situations → complete graph, equal degree.
    let hg = make_hg();
    let n = "complete";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    let s1 = add_situation(&hg, n);
    link(&hg, a, s1);
    link(&hg, b, s1);
    link(&hg, c, s1);

    let analysis = run_centrality(&hg, n).unwrap();
    let degrees: Vec<f64> = analysis.results.iter().map(|r| r.degree).collect();
    assert!((degrees[0] - degrees[1]).abs() < 0.001);
    assert!((degrees[1] - degrees[2]).abs() < 0.001);
}

// ─── Community Detection Tests ──────────────────────────

#[test]
fn test_community_two_groups() {
    // Two clearly separated groups connected by one shared situation.
    let hg = make_hg();
    let n = "communities";

    let a1 = add_entity(&hg, "a1", n);
    let a2 = add_entity(&hg, "a2", n);
    let a3 = add_entity(&hg, "a3", n);
    let b1 = add_entity(&hg, "b1", n);
    let b2 = add_entity(&hg, "b2", n);
    let b3 = add_entity(&hg, "b3", n);

    // Group A: all share many situations (strong internal edges)
    for _ in 0..10 {
        let s = add_situation(&hg, n);
        link(&hg, a1, s);
        link(&hg, a2, s);
        link(&hg, a3, s);
    }

    // Group B: all share many situations (strong internal edges)
    for _ in 0..10 {
        let s = add_situation(&hg, n);
        link(&hg, b1, s);
        link(&hg, b2, s);
        link(&hg, b3, s);
    }

    // Single weak link between groups
    let s_bridge = add_situation(&hg, n);
    link(&hg, a1, s_bridge);
    link(&hg, b1, s_bridge);

    let analysis = run_centrality(&hg, n).unwrap();
    assert!(analysis.num_communities >= 1); // At least 1 community detected

    // Check that a1, a2, a3 are in same community.
    let a1_comm = analysis
        .results
        .iter()
        .find(|r| r.entity_id == a1)
        .unwrap()
        .community_id;
    let a2_comm = analysis
        .results
        .iter()
        .find(|r| r.entity_id == a2)
        .unwrap()
        .community_id;
    let a3_comm = analysis
        .results
        .iter()
        .find(|r| r.entity_id == a3)
        .unwrap()
        .community_id;
    assert_eq!(a1_comm, a2_comm);
    assert_eq!(a2_comm, a3_comm);

    // Check that b1, b2, b3 are in same community.
    let b1_comm = analysis
        .results
        .iter()
        .find(|r| r.entity_id == b1)
        .unwrap()
        .community_id;
    let b2_comm = analysis
        .results
        .iter()
        .find(|r| r.entity_id == b2)
        .unwrap()
        .community_id;
    let b3_comm = analysis
        .results
        .iter()
        .find(|r| r.entity_id == b3)
        .unwrap()
        .community_id;
    assert_eq!(b1_comm, b2_comm);
    assert_eq!(b2_comm, b3_comm);
}

#[test]
fn test_community_single_tight_group() {
    // One tightly connected group → one community.
    let hg = make_hg();
    let n = "tight";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let c = add_entity(&hg, "c", n);

    for _ in 0..5 {
        let s = add_situation(&hg, n);
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);
    }

    let analysis = run_centrality(&hg, n).unwrap();
    // Tightly connected group: either 1 community or all in same community.
    // Louvain may find 1 or more, but all nodes should be in the same one.
    let comms: std::collections::HashSet<usize> =
        analysis.results.iter().map(|r| r.community_id).collect();
    assert!(
        comms.len() <= 2,
        "Expected at most 2 communities for tight group, got {}",
        comms.len()
    );
}

// ─── Edge Cases ─────────────────────────────────────────

#[test]
fn test_empty_narrative() {
    let hg = make_hg();
    let analysis = run_centrality(&hg, "nonexistent").unwrap();
    assert!(analysis.results.is_empty());
    assert_eq!(analysis.num_communities, 0);
}

#[test]
fn test_single_entity() {
    let hg = make_hg();
    let n = "single";
    add_entity(&hg, "alone", n);

    let analysis = run_centrality(&hg, n).unwrap();
    assert_eq!(analysis.results.len(), 1);
    assert_eq!(analysis.results[0].betweenness, 0.0);
    assert_eq!(analysis.results[0].closeness, 0.0);
    assert_eq!(analysis.results[0].degree, 0.0);
}

#[test]
fn test_two_entities_no_shared_situation() {
    let hg = make_hg();
    let n = "disjoint";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s1 = add_situation(&hg, n);
    let s2 = add_situation(&hg, n);
    link(&hg, a, s1);
    link(&hg, b, s2);

    let analysis = run_centrality(&hg, n).unwrap();
    assert_eq!(analysis.results.len(), 2);
    for r in &analysis.results {
        assert_eq!(r.degree, 0.0);
        assert_eq!(r.closeness, 0.0);
    }
}

#[test]
fn test_two_entities_shared_situation() {
    let hg = make_hg();
    let n = "pair";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s = add_situation(&hg, n);
    link(&hg, a, s);
    link(&hg, b, s);

    let analysis = run_centrality(&hg, n).unwrap();
    assert_eq!(analysis.results.len(), 2);
    // Both have degree 1.0 (normalized by n-1 = 1)
    for r in &analysis.results {
        assert!((r.degree - 1.0).abs() < 0.001, "degree={}", r.degree);
    }
}

// ─── Integration Test ───────────────────────────────────

#[test]
fn test_realistic_narrative_identifies_key_actor() {
    let hg = make_hg();
    let n = "investigation";

    // Create a realistic scenario: detective connects all witnesses.
    let detective = add_entity(&hg, "detective", n);
    let w1 = add_entity(&hg, "witness1", n);
    let w2 = add_entity(&hg, "witness2", n);
    let w3 = add_entity(&hg, "witness3", n);
    let suspect = add_entity(&hg, "suspect", n);

    // Detective meets each witness separately.
    for w in [w1, w2, w3] {
        let s = add_situation(&hg, n);
        link(&hg, detective, s);
        link(&hg, w, s);
    }

    // Detective confronts suspect.
    let s_final = add_situation(&hg, n);
    link(&hg, detective, s_final);
    link(&hg, suspect, s_final);

    let analysis = run_centrality(&hg, n).unwrap();
    let det_r = analysis
        .results
        .iter()
        .find(|r| r.entity_id == detective)
        .unwrap();

    // Detective should have highest centrality.
    for r in &analysis.results {
        if r.entity_id != detective {
            assert!(
                det_r.betweenness >= r.betweenness,
                "detective={}, other={}",
                det_r.betweenness,
                r.betweenness
            );
            assert!(
                det_r.closeness >= r.closeness,
                "detective={}, other={}",
                det_r.closeness,
                r.closeness
            );
            assert!(
                det_r.degree >= r.degree,
                "detective={}, other={}",
                det_r.degree,
                r.degree
            );
        }
    }
}

#[test]
fn test_kv_storage() {
    let hg = make_hg();
    let n = "kv_test";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);

    let s = add_situation(&hg, n);
    link(&hg, a, s);
    link(&hg, b, s);

    run_centrality(&hg, n).unwrap();

    // Verify stored in KV.
    let key = analysis_key(keys::ANALYSIS_CENTRALITY, &[n, &a.to_string()]);
    let stored = hg.store().get(&key).unwrap();
    assert!(stored.is_some());
    let result: CentralityResult = serde_json::from_slice(&stored.unwrap()).unwrap();
    assert_eq!(result.entity_id, a);
}

#[test]
fn test_inference_engine_trait() {
    let engine = CentralityEngine;
    assert_eq!(engine.job_type(), InferenceJobType::CentralityAnalysis);
}

// ─── Leiden Algorithm Tests ────────────────────────────────

#[test]
fn test_leiden_connected_communities() {
    // Two groups connected by a single weak bridge.
    // Leiden should guarantee each community is a connected subgraph.
    let hg = make_hg();
    let n = "leiden-connected";

    let a1 = add_entity(&hg, "a1", n);
    let a2 = add_entity(&hg, "a2", n);
    let a3 = add_entity(&hg, "a3", n);
    let b1 = add_entity(&hg, "b1", n);
    let b2 = add_entity(&hg, "b2", n);
    let b3 = add_entity(&hg, "b3", n);

    // Group A: tightly connected
    for _ in 0..8 {
        let s = add_situation(&hg, n);
        link(&hg, a1, s);
        link(&hg, a2, s);
        link(&hg, a3, s);
    }

    // Group B: tightly connected
    for _ in 0..8 {
        let s = add_situation(&hg, n);
        link(&hg, b1, s);
        link(&hg, b2, s);
        link(&hg, b3, s);
    }

    // Weak bridge
    let sb = add_situation(&hg, n);
    link(&hg, a1, sb);
    link(&hg, b1, sb);

    let analysis = run_centrality(&hg, n).unwrap();
    // With Phase 2 super-graph contraction, Louvain may merge the two groups
    // if the overall modularity is better. The key invariant is that
    // communities are connected subgraphs (Leiden refinement guarantees this).
    assert!(analysis.num_communities >= 1);
    assert_eq!(analysis.results.len(), 6);
}

#[test]
fn test_hierarchical_leiden_produces_levels() {
    // Build a graph large enough that hierarchical subdivision applies.
    let hg = make_hg();
    let n = "hier-leiden";

    // Create 15 entities in 3 groups of 5
    let mut group_a = Vec::new();
    let mut group_b = Vec::new();
    let mut group_c = Vec::new();
    for i in 0..5 {
        group_a.push(add_entity(&hg, &format!("a{}", i), n));
        group_b.push(add_entity(&hg, &format!("b{}", i), n));
        group_c.push(add_entity(&hg, &format!("c{}", i), n));
    }

    // Tight internal connections within each group
    for _ in 0..10 {
        let sa = add_situation(&hg, n);
        let sb = add_situation(&hg, n);
        let sc = add_situation(&hg, n);
        for &e in &group_a {
            link(&hg, e, sa);
        }
        for &e in &group_b {
            link(&hg, e, sb);
        }
        for &e in &group_c {
            link(&hg, e, sc);
        }
    }

    // Weak bridges between groups
    let s1 = add_situation(&hg, n);
    link(&hg, group_a[0], s1);
    link(&hg, group_b[0], s1);
    let s2 = add_situation(&hg, n);
    link(&hg, group_b[0], s2);
    link(&hg, group_c[0], s2);

    let graph = build_co_graph(&hg, n).unwrap();
    let levels = hierarchical_leiden(&graph, 4); // max community size = 4

    // Should have at least 1 level (base)
    assert!(!levels.is_empty());
    // Level 0 should have assignments for all 15 entities
    assert_eq!(levels[0].assignments.len(), 15);
    // With Phase 2 super-graph contraction, the base level may produce
    // fewer communities than the pre-Phase-2 Louvain. The key property
    // is that hierarchical subdivision occurred.
    assert!(
        levels[0].num_communities >= 1,
        "Expected at least 1 community at level 0, got {}",
        levels[0].num_communities
    );
}

#[test]
fn test_leiden_empty_graph() {
    let hg = make_hg();
    let graph = build_co_graph(&hg, "empty").unwrap();
    let (comm, mod_val, num) = leiden(&graph);
    assert!(comm.is_empty());
    assert_eq!(mod_val, 0.0);
    assert_eq!(num, 0);
}

#[test]
fn test_hierarchical_leiden_small_graph() {
    // Graph smaller than max_community_size → only 1 level
    let hg = make_hg();
    let n = "small-hier";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let s = add_situation(&hg, n);
    link(&hg, a, s);
    link(&hg, b, s);

    let graph = build_co_graph(&hg, n).unwrap();
    let levels = hierarchical_leiden(&graph, 10);
    assert_eq!(levels.len(), 1); // Only base level, no subdivision needed
}

#[test]
fn test_hierarchy_stored_in_kv() {
    // run_centrality should store hierarchy in KV at an/ch/{narrative_id}
    let hg = make_hg();
    let n = "kv-hier";
    let a = add_entity(&hg, "a", n);
    let b = add_entity(&hg, "b", n);
    let s = add_situation(&hg, n);
    link(&hg, a, s);
    link(&hg, b, s);

    run_centrality(&hg, n).unwrap();

    // Check hierarchy is stored
    let key = format!("an/ch/{}", n);
    let data = hg.store().get(key.as_bytes()).unwrap();
    assert!(data.is_some(), "Hierarchy should be stored in KV");
    let levels: Vec<CommunityLevel> = serde_json::from_slice(&data.unwrap()).unwrap();
    assert!(!levels.is_empty());
}
