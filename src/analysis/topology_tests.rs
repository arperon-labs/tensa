use super::*;
use crate::analysis::test_helpers::*;
use crate::types::JobStatus;

// ── Articulation Points ────────────────────────────────────

#[test]
fn test_articulation_point_simple() {
    // A-B-C chain: B is articulation point
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
    let (is_ap, _bridges) = find_articulation_points_and_bridges(&graph);
    let b_idx = graph.entities.iter().position(|&e| e == b).unwrap();
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    assert!(is_ap[b_idx], "B should be an articulation point");
    assert!(!is_ap[a_idx], "A should NOT be an articulation point");
}

#[test]
fn test_articulation_point_none() {
    // Triangle: no articulation points
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);
    link(&hg, c, s);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let (is_ap, _) = find_articulation_points_and_bridges(&graph);
    for ap in &is_ap {
        assert!(!ap, "Triangle has no articulation points");
    }
}

// ── Bridges ────────────────────────────────────────────────

#[test]
fn test_bridges_simple() {
    // A-B-C chain: edge B-C (or A-B) is a bridge
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
    let (_, bridges) = find_articulation_points_and_bridges(&graph);
    assert_eq!(bridges.len(), 2, "Chain of 3 has 2 bridges");
}

#[test]
fn test_bridges_none() {
    // Triangle: no bridges
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let s = add_situation(&hg, "n1");
    link(&hg, a, s);
    link(&hg, b, s);
    link(&hg, c, s);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let (_, bridges) = find_articulation_points_and_bridges(&graph);
    assert!(bridges.is_empty(), "Triangle has no bridges");
}

// ── K-Core ─────────────────────────────────────────────────

#[test]
fn test_kcore_simple() {
    // Triangle (3-clique) + pendant: triangle nodes = 2-core, pendant = 1-core
    let hg = make_hg();
    let a = add_entity(&hg, "A", "n1");
    let b = add_entity(&hg, "B", "n1");
    let c = add_entity(&hg, "C", "n1");
    let d = add_entity(&hg, "D", "n1");
    let s = add_situation(&hg, "n1"); // triangle: a,b,c
    link(&hg, a, s);
    link(&hg, b, s);
    link(&hg, c, s);
    let s2 = add_situation(&hg, "n1"); // pendant: c-d
    link(&hg, c, s2);
    link(&hg, d, s2);

    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let cores = compute_kcore(&graph);
    let a_idx = graph.entities.iter().position(|&e| e == a).unwrap();
    let d_idx = graph.entities.iter().position(|&e| e == d).unwrap();
    assert_eq!(cores[a_idx], 2, "Triangle node should be in 2-core");
    assert_eq!(cores[d_idx], 1, "Pendant node should be in 1-core");
}

#[test]
fn test_kcore_empty() {
    let hg = make_hg();
    let graph = graph_projection::build_co_graph(&hg, "n1").unwrap();
    let cores = compute_kcore(&graph);
    assert!(cores.is_empty());
}

// ── INFER TOPOLOGY integration ─────────────────────────────

#[test]
fn test_topology_queryable_virtual_props() {
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

    let job = crate::inference::types::InferenceJob {
        id: "test-topo".into(),
        job_type: InferenceJobType::Topology,
        target_id: uuid::Uuid::nil(),
        parameters: serde_json::json!({"narrative_id": "n1"}),
        priority: crate::types::JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    let engine = TopologyEngine;
    let result = engine.execute(&job, &hg).unwrap();
    assert_eq!(result.status, JobStatus::Completed);

    // Verify B is stored as articulation point
    let key = analysis_key(b"an/tp/", &["n1", &b.to_string()]);
    let bytes = hg
        .store()
        .get(&key)
        .unwrap()
        .expect("topology should be stored");
    let topo: TopologyResult = serde_json::from_slice(&bytes).unwrap();
    assert!(topo.is_articulation_point, "B should be articulation point");
    assert!(topo.is_bridge_endpoint, "B should be bridge endpoint");
}

#[test]
fn test_topology_projection_causal_finds_bridge_not_in_cooccurrence() {
    // Setup: three actors that all happen to co-participate in one big meeting
    // situation s_meet (creates A-B-C triangle in the co-graph → no articulation),
    // but on the causal DAG side a chain s_a → s_b → s_c connects them linearly
    // through Bob, making Bob an articulation point in the causal projection.
    let hg = make_hg();
    let nid = "n_proj";
    let a = add_entity(&hg, "A", nid);
    let b = add_entity(&hg, "B", nid);
    let c = add_entity(&hg, "C", nid);

    let s_meet = add_situation(&hg, nid);
    link(&hg, a, s_meet);
    link(&hg, b, s_meet);
    link(&hg, c, s_meet);

    let s_a = add_situation(&hg, nid);
    let s_b = add_situation(&hg, nid);
    let s_c = add_situation(&hg, nid);
    link(&hg, a, s_a);
    link(&hg, b, s_b);
    link(&hg, c, s_c);

    hg.add_causal_link(crate::types::CausalLink {
        from_situation: s_a,
        to_situation: s_b,
        strength: 0.9,
        mechanism: None,
        causal_type: crate::types::CausalType::Contributing,
        maturity: crate::types::MaturityLevel::Candidate,
    })
    .unwrap();
    hg.add_causal_link(crate::types::CausalLink {
        from_situation: s_b,
        to_situation: s_c,
        strength: 0.9,
        mechanism: None,
        causal_type: crate::types::CausalType::Contributing,
        maturity: crate::types::MaturityLevel::Candidate,
    })
    .unwrap();

    let run_with = |projection: &str| {
        let job = crate::inference::types::InferenceJob {
            id: format!("test-topo-{projection}"),
            job_type: InferenceJobType::Topology,
            target_id: uuid::Uuid::nil(),
            parameters: serde_json::json!({
                "narrative_id": nid,
                "projection": projection,
            }),
            priority: crate::types::JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        TopologyEngine.execute(&job, &hg).unwrap();
        let key = analysis_key(b"an/tp/", &[nid, &b.to_string()]);
        let bytes = hg.store().get(&key).unwrap().unwrap();
        serde_json::from_slice::<TopologyResult>(&bytes).unwrap()
    };

    let cooc = run_with("cooccurrence");
    assert!(
        !cooc.is_articulation_point,
        "triangle co-graph has no articulation points"
    );

    let causal = run_with("causal");
    assert!(
        causal.is_articulation_point,
        "B bridges A and C in the causal projection (s_a → s_b → s_c)"
    );
}

#[test]
fn test_topology_projection_defaults_to_cooccurrence() {
    let job = crate::inference::types::InferenceJob {
        id: "x".into(),
        job_type: InferenceJobType::Topology,
        target_id: uuid::Uuid::nil(),
        parameters: serde_json::json!({"narrative_id": "n"}),
        priority: crate::types::JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    assert_eq!(
        TopologyProjection::from_job(&job),
        TopologyProjection::Cooccurrence
    );

    let mut job_causal = job.clone();
    job_causal.parameters = serde_json::json!({"narrative_id": "n", "projection": "CAUSAL"});
    assert_eq!(
        TopologyProjection::from_job(&job_causal),
        TopologyProjection::Causal
    );

    let mut job_junk = job.clone();
    job_junk.parameters = serde_json::json!({"narrative_id": "n", "projection": "nonsense"});
    assert_eq!(
        TopologyProjection::from_job(&job_junk),
        TopologyProjection::Cooccurrence
    );
}
