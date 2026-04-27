//! Graded Acceptability Sprint Phase 5 — embedded MCP round-trip tests
//! for the five graded-surface tools.
//!
//! Each test exercises the `EmbeddedBackend` impl path end-to-end against
//! an `Arc<MemoryStore>` so the harness mirrors the
//! [`crate::mcp::embedded_synth::tests`] pattern (no live HTTP loop, no
//! axum router). The tests assert wire-format invariants — keys present,
//! shape correct — rather than re-validating the algorithms (those live
//! in their respective sprint test suites).
//!
//! Cites: [amgoud2013ranking] [grabisch1996choquet] [nebel1995ordhorn].

use std::sync::Arc;

use uuid::Uuid;

use crate::analysis::argumentation_gradual::GradualSemanticsKind;
use crate::fuzzy::synthetic_cib_dataset::generate_synthetic_cib;
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::mcp::backend::McpBackend;
use crate::mcp::embedded::EmbeddedBackend;
use crate::source::{ContentionLink, ContentionType};
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::temporal::ordhorn::{OrdHornConstraint, OrdHornNetwork};
use crate::types::{
    AllenInterval, AllenRelation, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel,
    Situation, TimeGranularity,
};

// ── Test fixtures ──────────────────────────────────────────────────────

fn backend() -> EmbeddedBackend {
    EmbeddedBackend::from_store(Arc::new(MemoryStore::new()))
}

fn make_situation(hg: &Hypergraph, narrative_id: &str) -> Uuid {
    let now = chrono::Utc::now();
    let sit = Situation {
        id: Uuid::now_v7(),
        properties: serde_json::Value::Null,
        name: None,
        description: None,
        temporal: AllenInterval {
            start: Some(now),
            end: Some(now),
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
        raw_content: vec![ContentBlock::text("test scene")],
        narrative_level: NarrativeLevel::Event,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 0.7,
        confidence_breakdown: None,
        extraction_method: ExtractionMethod::HumanEntered,
        provenance: vec![],
        narrative_id: Some(narrative_id.into()),
        source_chunk_id: None,
        source_span: None,
        synopsis: None,
        manuscript_order: None,
        parent_situation_id: None,
        label: None,
        status: None,
        keywords: vec![],
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    };
    hg.create_situation(sit).expect("create situation")
}

/// Seed a 2-situation contention so `build_framework` returns a non-empty
/// argumentation framework.
fn seed_contention_pair(b: &EmbeddedBackend, narrative_id: &str) -> (Uuid, Uuid) {
    let hg = b.hypergraph();
    let a = make_situation(hg, narrative_id);
    let b_id = make_situation(hg, narrative_id);
    hg.add_contention(ContentionLink {
        situation_a: a,
        situation_b: b_id,
        contention_type: ContentionType::DirectContradiction,
        description: Some("test contention".into()),
        resolved: false,
        resolution: None,
        created_at: chrono::Utc::now(),
    })
    .expect("add contention");
    (a, b_id)
}

// ── T1: argumentation_gradual ─────────────────────────────────────────

/// T1 — submitting `argumentation_gradual` over a 2-situation contention
/// framework returns a `gradual.acceptability` map keyed by argument
/// UUIDs.
#[tokio::test]
async fn t1_argumentation_gradual_returns_acceptability_map() {
    let b = backend();
    let nid = "case-graded-mcp-t1";
    let (a, c) = seed_contention_pair(&b, nid);

    let kind = serde_json::to_value(&GradualSemanticsKind::HCategoriser).unwrap();
    let result = b
        .argumentation_gradual(nid, kind, None)
        .await
        .expect("argumentation_gradual ok");

    assert_eq!(
        result["narrative_id"].as_str(),
        Some(nid),
        "envelope must echo narrative_id"
    );
    let acc = result["gradual"]["acceptability"]
        .as_object()
        .expect("gradual.acceptability map");
    assert_eq!(
        acc.len(),
        2,
        "framework has 2 arguments (one per situation in the contention pair)"
    );
    assert!(
        acc.contains_key(&a.to_string()),
        "acceptability map must include situation a"
    );
    assert!(
        acc.contains_key(&c.to_string()),
        "acceptability map must include situation b"
    );

    // Telemetry surfaced at envelope level.
    let iters = result["iterations"].as_u64().expect("iterations field");
    assert!(iters >= 1, "h-categoriser runs at least one iteration");
    assert_eq!(
        result["converged"].as_bool(),
        Some(true),
        "h-categoriser is contracting under Gödel — must converge"
    );

    // Empty narrative_id → InvalidInput.
    let bad = b
        .argumentation_gradual("", serde_json::json!("HCategoriser"), None)
        .await;
    assert!(bad.is_err(), "empty narrative_id must error");

    // Optional `tnorm` blob round-trips through the JSON dispatcher.
    let kind2 = serde_json::to_value(&GradualSemanticsKind::HCategoriser).unwrap();
    let tnorm_blob = serde_json::to_value(&TNormKind::Godel).unwrap();
    let with_tnorm = b
        .argumentation_gradual(nid, kind2, Some(tnorm_blob))
        .await
        .expect("argumentation_gradual w/ tnorm ok");
    assert_eq!(with_tnorm["narrative_id"].as_str(), Some(nid));
}

// ── T2: fuzzy_learn_measure ───────────────────────────────────────────

/// T2 — fitting on the synthetic-CIB dataset returns version=1 and a
/// non-trivial train AUC (> 0.5 ⇒ better than random; the §3.3 paper
/// figure asserts > 0.85 but we keep the threshold loose so the test
/// stays tolerant of stochastic train/test splits).
#[tokio::test]
async fn t2_fuzzy_learn_measure_persists_with_version_one() {
    let b = backend();
    let dataset = generate_synthetic_cib(42, 100);

    let result = b
        .fuzzy_learn_measure("learned-cib-mcp-t2", 4, dataset, "phase5-mcp-t2")
        .await
        .expect("fuzzy_learn_measure ok");

    assert_eq!(result["name"].as_str(), Some("learned-cib-mcp-t2"));
    assert_eq!(result["version"].as_u64(), Some(1));
    assert_eq!(result["n"].as_u64(), Some(4));
    let train_auc = result["train_auc"].as_f64().expect("train_auc");
    assert!(
        train_auc > 0.5,
        "train_auc must be > 0.5 (random baseline); got {train_auc}"
    );
    let test_auc = result["test_auc"].as_f64().expect("test_auc");
    assert!(
        (0.0..=1.0).contains(&test_auc),
        "test_auc must be in [0, 1]; got {test_auc}"
    );

    // Empty / invalid name surfaces InvalidInput.
    let bad = b
        .fuzzy_learn_measure("", 4, generate_synthetic_cib(42, 100), "ds")
        .await;
    assert!(bad.is_err(), "empty name must error");

    let bad2 = b
        .fuzzy_learn_measure("bad/name", 4, generate_synthetic_cib(42, 100), "ds")
        .await;
    assert!(bad2.is_err(), "name with '/' must error");
}

// ── T3: fuzzy_get_measure_version ─────────────────────────────────────

/// T3 — round-trip a learn → get(version=1) → unversioned get sequence.
/// Covers both the versioned slice path and the latest-pointer path
/// (legacy unversioned read).
#[tokio::test]
async fn t3_fuzzy_get_measure_version_round_trip() {
    let b = backend();
    let dataset = generate_synthetic_cib(42, 100);
    let _ = b
        .fuzzy_learn_measure("learned-cib-mcp-t3", 4, dataset, "phase5-mcp-t3")
        .await
        .expect("learn ok");

    // Versioned get.
    let v1 = b
        .fuzzy_get_measure_version("learned-cib-mcp-t3", Some(1))
        .await
        .expect("versioned get ok");
    assert_eq!(v1["name"].as_str(), Some("learned-cib-mcp-t3"));
    assert_eq!(v1["version"].as_u64(), Some(1));

    // Unversioned (latest pointer) — same record after a single learn.
    let latest = b
        .fuzzy_get_measure_version("learned-cib-mcp-t3", None)
        .await
        .expect("latest get ok");
    assert_eq!(latest["version"].as_u64(), Some(1));

    // Missing version surfaces a clear error.
    let missing = b
        .fuzzy_get_measure_version("learned-cib-mcp-t3", Some(999))
        .await;
    assert!(missing.is_err(), "missing version must error");

    // Missing name (unversioned path) also errors.
    let no_name = b
        .fuzzy_get_measure_version("nonexistent-mcp-t3", None)
        .await;
    assert!(no_name.is_err(), "missing name must error");
}

// ── T4: fuzzy_list_measure_versions ───────────────────────────────────

/// T4 — after a single learn, the versions list contains exactly `[1]`;
/// after a re-train, `[1, 2]`. Asserts the prefix-scan + version-suffix
/// parser stay in lockstep with `fuzzy_learn_measure`'s dual-write
/// contract.
#[tokio::test]
async fn t4_fuzzy_list_measure_versions_round_trip() {
    let b = backend();

    // Empty list before any learn.
    let empty = b
        .fuzzy_list_measure_versions("learned-cib-mcp-t4")
        .await
        .expect("list ok (empty)");
    let versions: Vec<u64> = empty["versions"]
        .as_array()
        .expect("versions array")
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    assert!(versions.is_empty(), "no versions before any learn");

    // First learn → [1].
    let _ = b
        .fuzzy_learn_measure(
            "learned-cib-mcp-t4",
            4,
            generate_synthetic_cib(42, 100),
            "phase5-mcp-t4-a",
        )
        .await
        .expect("first learn");
    let one = b
        .fuzzy_list_measure_versions("learned-cib-mcp-t4")
        .await
        .expect("list ok (one)");
    let versions: Vec<u64> = one["versions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    assert_eq!(versions, vec![1u64], "single version after one learn");
    assert_eq!(one["name"].as_str(), Some("learned-cib-mcp-t4"));

    // Re-train → [1, 2].
    let _ = b
        .fuzzy_learn_measure(
            "learned-cib-mcp-t4",
            4,
            generate_synthetic_cib(42, 100),
            "phase5-mcp-t4-b",
        )
        .await
        .expect("re-learn");
    let two = b
        .fuzzy_list_measure_versions("learned-cib-mcp-t4")
        .await
        .expect("list ok (two)");
    let versions: Vec<u64> = two["versions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_u64())
        .collect();
    assert_eq!(
        versions,
        vec![1u64, 2u64],
        "two versions after re-train, sorted ascending"
    );
}

// ── T5: temporal_ordhorn_closure ──────────────────────────────────────

/// T5 — a satisfiable 2-node Allen network round-trips through closure,
/// returning `satisfiable: true` and a non-empty `closed_network`. Uses
/// the same `0 Before 1` fixture as the REST handler test
/// (`t_rest_1_satisfiable_round_trip`).
#[tokio::test]
async fn t5_temporal_ordhorn_closure_satisfiable_round_trip() {
    let b = backend();
    let net = OrdHornNetwork {
        n: 2,
        constraints: vec![OrdHornConstraint {
            a: 0,
            b: 1,
            relations: vec![AllenRelation::Before],
        }],
    };
    let payload = serde_json::to_value(&net).expect("serialise network");

    let result = b
        .temporal_ordhorn_closure(payload)
        .await
        .expect("closure ok");

    assert_eq!(
        result["satisfiable"].as_bool(),
        Some(true),
        "2-node Before is satisfiable"
    );
    let closed = result["closed_network"]
        .as_object()
        .expect("closed_network object");
    assert_eq!(
        closed.get("n").and_then(|v| v.as_u64()),
        Some(2u64),
        "closed network preserves interval count"
    );
    let constraints = closed
        .get("constraints")
        .and_then(|v| v.as_array())
        .expect("constraints array");
    assert!(
        !constraints.is_empty(),
        "satisfiable closure preserves the input constraint"
    );

    // Unsatisfiable network round-trips with `satisfiable: false`.
    let unsat = OrdHornNetwork {
        n: 3,
        constraints: vec![
            OrdHornConstraint {
                a: 0,
                b: 1,
                relations: vec![AllenRelation::Before],
            },
            OrdHornConstraint {
                a: 1,
                b: 2,
                relations: vec![AllenRelation::Before],
            },
            OrdHornConstraint {
                a: 0,
                b: 2,
                relations: vec![AllenRelation::After],
            },
        ],
    };
    let unsat_payload = serde_json::to_value(&unsat).expect("serialise unsat network");
    let unsat_result = b
        .temporal_ordhorn_closure(unsat_payload)
        .await
        .expect("unsat closure ok (200, satisfiable=false)");
    assert_eq!(
        unsat_result["satisfiable"].as_bool(),
        Some(false),
        "Before-Before-After cycle is unsatisfiable"
    );

    // Garbage payload → InvalidInput.
    let bad = b
        .temporal_ordhorn_closure(serde_json::json!({"missing": "fields"}))
        .await;
    assert!(bad.is_err(), "garbage payload must error");
}
