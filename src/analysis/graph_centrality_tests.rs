use super::*;
use crate::analysis::test_helpers::*;
use crate::types::JobStatus;
use uuid::Uuid;

// ── PageRank ───────────────────────────────────────────────

#[test]
fn test_pagerank_simple() {
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
    let pr = compute_pagerank(&graph);
    assert_eq!(pr.len(), 3);
    let total: f64 = pr.iter().sum();
    assert!(
        (total - 1.0).abs() < 0.01,
        "PageRank should sum to ~1, got {}",
        total
    );
    // B is the hub connecting A and C, should have highest rank
    let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    assert!(
        pr[b_idx] > pr[a_idx],
        "Hub B should rank higher than leaf A"
    );
}

#[test]
fn test_pagerank_convergence() {
    let hg = make_hg();
    let entities: Vec<Uuid> = (0..10)
        .map(|i| add_entity(&hg, &format!("E{}", i), "n1"))
        .collect();
    let s = add_situation(&hg, "n1");
    for &e in &entities {
        link(&hg, e, s);
    }
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let pr = compute_pagerank(&graph);
    let expected = 1.0 / 10.0;
    for score in &pr {
        assert!(
            (score - expected).abs() < 0.01,
            "Expected uniform ~{}, got {}",
            expected,
            score
        );
    }
}

#[test]
fn test_pagerank_damping() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let _b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let pr = compute_pagerank(&graph);
    assert_eq!(pr.len(), 2);
    for score in &pr {
        assert!(*score > 0.0, "All nodes should have positive PageRank");
    }
}

#[test]
fn test_pagerank_dangling_node() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let pr = compute_pagerank(&graph);
    let total: f64 = pr.iter().sum();
    assert!((total - 1.0).abs() < 0.01);
}

// ── Eigenvector Centrality ─────────────────────────────────

#[test]
fn test_eigenvector_star() {
    let hg = make_hg();
    let center = add_entity(&hg, "Center", "n1");
    let leaves: Vec<Uuid> = (0..4)
        .map(|i| add_entity(&hg, &format!("Leaf{}", i), "n1"))
        .collect();
    for &leaf in &leaves {
        let s = add_situation(&hg, "n1");
        link(&hg, center, s);
        link(&hg, leaf, s);
    }
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let ev = compute_eigenvector(&graph);
    let center_idx = graph.entities.iter().position(|&e| e == center).unwrap();
    for (i, score) in ev.iter().enumerate() {
        if i != center_idx {
            assert!(
                ev[center_idx] >= *score,
                "Center should have highest eigenvector centrality"
            );
        }
    }
}

#[test]
fn test_eigenvector_chain() {
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
    let ev = compute_eigenvector(&graph);
    assert_eq!(ev.len(), 3);
    let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    assert!(ev[b_idx] >= ev[a_idx]);
}

#[test]
fn test_eigenvector_disconnected() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let d = add_entity(&hg, "D", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, b, s1);
    link(&hg, c, s2);
    link(&hg, d, s2);
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let ev = compute_eigenvector(&graph);
    assert_eq!(ev.len(), 4);
    for score in &ev {
        assert!(!score.is_nan());
        assert!(*score >= 0.0);
    }
}

// ── Harmonic Centrality ────────────────────────────────────

#[test]
fn test_harmonic_connected() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);
    link(&hg, c, s);
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let h = compute_harmonic(&graph);
    assert_eq!(h.len(), 3);
    for score in &h {
        assert!(
            (score - 1.0).abs() < 0.01,
            "Triangle harmonic should be ~1.0, got {}",
            score
        );
    }
}

#[test]
fn test_harmonic_disconnected() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let d = add_entity(&hg, "D", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, b, s1);
    link(&hg, c, s2);
    link(&hg, d, s2);
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let h = compute_harmonic(&graph);
    assert_eq!(h.len(), 4);
    for score in &h {
        assert!(
            (score - 1.0 / 3.0).abs() < 0.01,
            "Disconnected pair harmonic should be ~0.333, got {}",
            score
        );
    }
}

// ── HITS ───────────────────────────────────────────────────

#[test]
fn test_hits_bipartite() {
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s1 = add_situation(&hg, "n1");
    let s2 = add_situation(&hg, "n1");
    link(&hg, a, s1);
    link(&hg, a, s2);
    link(&hg, b, s1);
    link(&hg, c, s2);
    let bipartite = graph_projection::build_bipartite(&hg, "n1").unwrap();
    let (hub, auth) = compute_hits(&bipartite);
    assert_eq!(hub.len(), 3);
    assert_eq!(auth.len(), 3);
    let a_idx = bipartite.entities.iter().position(|&e| e == a).unwrap();
    let b_idx = bipartite.entities.iter().position(|&e| e == b).unwrap();
    assert!(
        hub[a_idx] > hub[b_idx],
        "A (2 situations) should have higher hub than B (1 situation)"
    );
}

#[test]
fn test_hits_convergence() {
    let hg = make_hg();
    let entities: Vec<Uuid> = (0..4)
        .map(|i| add_entity(&hg, &format!("E{}", i), "n1"))
        .collect();
    let s = add_situation(&hg, "n1");
    for &e in &entities {
        link(&hg, e, s);
    }
    let bipartite = graph_projection::build_bipartite(&hg, "n1").unwrap();
    let (hub, auth) = compute_hits(&bipartite);
    let expected = hub[0];
    for score in &hub {
        assert!((score - expected).abs() < 0.01);
    }
    let expected_auth = auth[0];
    for score in &auth {
        assert!((score - expected_auth).abs() < 0.01);
    }
}

#[test]
fn test_hits_queryable_virtual_prop() {
    use crate::inference::types::InferenceJob;

    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);

    let job = InferenceJob {
        id: "test-hits".into(),
        job_type: InferenceJobType::HITS,
        target_id: Uuid::nil(),
        parameters: serde_json::json!({"narrative_id": "n1"}),
        priority: crate::types::JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let engine = HitsEngine;
    let result = engine.execute(&job, &hg).unwrap();
    assert_eq!(result.status, JobStatus::Completed);

    let key = analysis_key(b"an/hits/", &["n1", &a.to_string()]);
    let bytes = hg
        .store()
        .get(&key)
        .unwrap()
        .expect("HITS score should be stored");
    let score: HitsScore = serde_json::from_slice(&bytes).unwrap();
    assert!(score.hub_score > 0.0);
    assert!(score.authority_score > 0.0);
}

// ── Personalized PageRank ──────────────────────────────────

#[test]
fn test_ppr_seed_bias() {
    // Triangle: A-B-C. Seed from A → A should have highest PPR.
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);
    link(&hg, c, s);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let mut seed = vec![0.0; 3];
    seed[a_idx] = 1.0;

    let ppr = personalized_pagerank(&graph, &seed, 0.15);
    assert_eq!(ppr.len(), 3);
    assert!(
        ppr[a_idx] > ppr[(a_idx + 1) % 3],
        "Seed node should have highest PPR"
    );
}
