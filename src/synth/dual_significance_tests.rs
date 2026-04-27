//! EATH Extension Phase 13c — dual-null-model significance tests.
//!
//! T1: dual run reports both EATH and NuDHy entries.
//! T2: combined verdict is `false` when only one model flags significance.
//! T3: same inputs → bit-identical output (parallel-by-model is deterministic).
//! T4: k_per_model is clamped at K_MAX with a tracing::warn.
//!
//! T5 lives in `src/query/parser.rs` tests (TensaQL grammar coverage).
//! T6 lives in `src/mcp/...` tests (round-trip through the MCP backend).
//!
//! All tests use `MemoryStore`. NuDHy K-loop builds its own ephemeral
//! hypergraphs; nothing here writes outside the test's own store.

use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use uuid::Uuid;

use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::registry::SurrogateRegistry;
use crate::synth::SurrogateDualSignificanceEngine;
use crate::types::{
    AllenInterval, ContentBlock, Entity, EntityType, ExtractionMethod, InferenceJobType,
    JobPriority, JobStatus, MaturityLevel, NarrativeLevel, Participation, Role, Situation,
    TimeGranularity,
};
use crate::Hypergraph;

// ── Fixtures ────────────────────────────────────────────────────────────────

fn fresh_hg() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

fn make_entity(narrative_id: &str, label: &str) -> Entity {
    let now = Utc::now();
    Entity {
        id: Uuid::now_v7(),
        entity_type: EntityType::Actor,
        properties: serde_json::json!({"name": label}),
        beliefs: None,
        embedding: None,
        maturity: MaturityLevel::Candidate,
        confidence: 1.0,
        confidence_breakdown: None,
        provenance: vec![],
        extraction_method: None,
        narrative_id: Some(narrative_id.into()),
        created_at: now,
        updated_at: now,
        deleted_at: None,
        transaction_time: None,
    }
}

fn make_situation(narrative_id: &str, t_offset_seconds: i64) -> Situation {
    let now = Utc::now();
    let epoch = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let start = Some(epoch + Duration::seconds(t_offset_seconds));
    Situation {
        id: Uuid::now_v7(),
        name: None,
        description: None,
        properties: serde_json::Value::Null,
        temporal: AllenInterval {
            start,
            end: start,
            granularity: TimeGranularity::Exact,
            relations: vec![],
            fuzzy_endpoints: None,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text("dual-sig fixture")],
        narrative_level: NarrativeLevel::Scene,
        discourse: None,
        maturity: MaturityLevel::Candidate,
        confidence: 1.0,
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
    }
}

/// Build a small narrative with `n_entities` actors and `n_situations`
/// hyperedges of size `group_size`. Each situation rotates through entities
/// so degree counts are balanced and NuDHy MCMC has many valid swaps.
fn seed_narrative(
    hg: &Hypergraph,
    narrative_id: &str,
    n_entities: usize,
    n_situations: usize,
    group_size: usize,
) -> Vec<Uuid> {
    let mut entity_ids = Vec::with_capacity(n_entities);
    for i in 0..n_entities {
        let id = hg
            .create_entity(make_entity(narrative_id, &format!("a{i}")))
            .unwrap();
        entity_ids.push(id);
    }
    for s_idx in 0..n_situations {
        let sid = hg
            .create_situation(make_situation(narrative_id, (s_idx as i64) * 60))
            .unwrap();
        for k in 0..group_size {
            let eid = entity_ids[(s_idx + k) % n_entities];
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: sid,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }
    entity_ids
}

fn engine() -> SurrogateDualSignificanceEngine {
    SurrogateDualSignificanceEngine::new(Arc::new(SurrogateRegistry::default()))
}

fn make_dual_job(
    narrative_id: &str,
    metric: &str,
    k_per_model: u16,
    models: Vec<String>,
) -> InferenceJob {
    InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type: InferenceJobType::SurrogateDualSignificance {
            narrative_id: narrative_id.into(),
            metric: metric.into(),
            k_per_model,
            models,
        },
        target_id: Uuid::now_v7(),
        parameters: serde_json::Value::Object(serde_json::Map::new()),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 1_000,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    }
}

// ── T1 — both models produce a per-model row ───────────────────────────────

#[test]
fn test_dual_sig_eath_and_nudhy_both_report() {
    let hg = fresh_hg();
    let nid = "t1-dual";
    seed_narrative(&hg, nid, 6, 12, 3);

    // Tiny K so the test is fast (NuDHy MCMC default burn-in is at least
    // 10_000 steps; with K=2 chains × 12-edge-source × ~360 burn-in steps
    // this completes in well under 10s on CI hardware).
    let job = make_dual_job(
        nid,
        "patterns",
        2,
        vec!["eath".into(), "nudhy".into()],
    );
    let result = engine().execute(&job, &hg).expect("T1 must succeed");
    assert_eq!(
        result.result["kind"].as_str(),
        Some("dual_significance_done")
    );
    let report = &result.result["report"];
    let per_model = report["per_model"].as_array().expect("per_model array");
    assert_eq!(per_model.len(), 2, "T1 expected exactly 2 per-model rows");
    let model_names: Vec<&str> = per_model
        .iter()
        .map(|r| r["model"].as_str().unwrap())
        .collect();
    assert!(model_names.contains(&"eath"), "T1 missing eath row");
    assert!(model_names.contains(&"nudhy"), "T1 missing nudhy row");
    let combined = &report["combined"];
    assert!(combined["significant_vs_all_at_p05"].is_boolean());
    assert!(combined["significant_vs_all_at_p01"].is_boolean());
}

// ── T2 — combined verdict is AND across models ─────────────────────────────

#[test]
fn test_dual_sig_pattern_significant_vs_only_one_model_flagged() {
    // We can't easily contrive a real-world data fixture where one model
    // flags significance and the other doesn't (deterministic engineering
    // of one-sided rejection across two MCMC distributions is paper-level
    // work). What we CAN test deterministically is the AND-reduction logic:
    // construct two SingleModelSignificance rows by hand, run the combiner,
    // and assert the verdict matches.
    use crate::synth::{CombinedSignificance, SingleModelSignificance};

    // Helper to call into the (private) combiner via the engine module.
    // The combiner is `pub(super)` so we go through a thin wrapper test.
    // Manual reduction matching the engine's algebra:
    fn combine(rows: &[SingleModelSignificance]) -> CombinedSignificance {
        super::combine_per_model(rows)
    }

    // Case A: both models flag (z > 1.96, p < 0.05) → all-pass.
    let rows_a = vec![
        SingleModelSignificance {
            model: "eath".into(),
            observed_value: 5.0,
            mean_null: 1.0,
            std_null: 1.0,
            z_score: 4.0,
            p_value: 0.001,
            samples_used: 50,
            starvations: 0,
        },
        SingleModelSignificance {
            model: "nudhy".into(),
            observed_value: 5.0,
            mean_null: 2.0,
            std_null: 0.5,
            z_score: 6.0,
            p_value: 0.001,
            samples_used: 50,
            starvations: 0,
        },
    ];
    let c_a = combine(&rows_a);
    assert!(c_a.significant_vs_all_at_p05);
    assert!(c_a.significant_vs_all_at_p01);

    // Case B: only EATH flags; NuDHy z is sub-threshold → AND fails.
    let rows_b = vec![
        SingleModelSignificance {
            model: "eath".into(),
            observed_value: 5.0,
            mean_null: 1.0,
            std_null: 1.0,
            z_score: 4.0,
            p_value: 0.001,
            samples_used: 50,
            starvations: 0,
        },
        SingleModelSignificance {
            model: "nudhy".into(),
            observed_value: 5.0,
            mean_null: 4.5,
            std_null: 1.0,
            z_score: 0.5,
            p_value: 0.4,
            samples_used: 50,
            starvations: 0,
        },
    ];
    let c_b = combine(&rows_b);
    assert!(
        !c_b.significant_vs_all_at_p05,
        "T2 only-one-model significance must NOT trigger combined p05 verdict"
    );
    assert!(!c_b.significant_vs_all_at_p01);
    // min p across models reflects EATH's row (0.001).
    assert!((c_b.min_p_across_models - 0.001f32).abs() < 1e-6);
    // max |z| across models reflects EATH's |4.0|.
    assert!((c_b.max_abs_z_across_models - 4.0f32).abs() < 1e-6);
}

// ── T3 — determinism: same inputs → bit-identical output ───────────────────

#[test]
fn test_dual_sig_parallelism_no_determinism_break() {
    let hg = fresh_hg();
    let nid = "t3-dual-det";
    seed_narrative(&hg, nid, 5, 8, 3);

    let run = || {
        let job = make_dual_job(
            nid,
            "patterns",
            2,
            vec!["eath".into(), "nudhy".into()],
        );
        let result = engine()
            .execute(&job, &hg)
            .expect("T3 dual run must succeed");
        let r = &result.result["report"];
        let per_model = r["per_model"]
            .as_array()
            .expect("per_model array")
            .clone();
        let mut summaries: Vec<(String, f64, f64, u16, u16)> = per_model
            .iter()
            .map(|m| {
                (
                    m["model"].as_str().unwrap().to_string(),
                    m["z_score"].as_f64().unwrap_or(f64::NAN),
                    m["mean_null"].as_f64().unwrap_or(f64::NAN),
                    m["samples_used"].as_u64().unwrap_or(0) as u16,
                    m["starvations"].as_u64().unwrap_or(0) as u16,
                )
            })
            .collect();
            // Sort by model name so per-thread join order doesn't matter for
            // the comparison itself (the engine does already sort by index;
            // this is belt-and-braces).
        summaries.sort_by(|a, b| a.0.cmp(&b.0));
        summaries
    };

    let a = run();
    let b = run();
    assert_eq!(
        a.len(),
        b.len(),
        "T3 dual runs must yield same number of per-model rows"
    );
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.0, rb.0, "T3 model names must match");
        // z and mean: NaN-aware bitwise compare.
        let bitwise_eq_or_nan = |x: f64, y: f64| {
            (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits()
        };
        assert!(
            bitwise_eq_or_nan(ra.1, rb.1),
            "T3 z_score drift for {}: {} vs {}",
            ra.0,
            ra.1,
            rb.1
        );
        assert!(
            bitwise_eq_or_nan(ra.2, rb.2),
            "T3 mean_null drift for {}: {} vs {}",
            ra.0,
            ra.2,
            rb.2
        );
        assert_eq!(ra.3, rb.3, "T3 samples_used drift for {}", ra.0);
        assert_eq!(ra.4, rb.4, "T3 starvations drift for {}", ra.0);
    }
}

// ── T4 — k_per_model is clamped at K_MAX ───────────────────────────────────

#[test]
fn test_dual_sig_caps_k_per_model() {
    use crate::synth::significance::K_MAX;

    let hg = fresh_hg();
    let nid = "t4-dual-cap";
    seed_narrative(&hg, nid, 4, 6, 3);

    // Request way over K_MAX. The engine clamps and the report's
    // k_per_model field reflects the clamp.
    let job = make_dual_job(
        nid,
        "patterns",
        u16::MAX,
        // Single model for speed — clamp logic is identical regardless of
        // model count, and EATH is the cheapest of the two.
        vec!["eath".into()],
    );
    let result = engine().execute(&job, &hg).expect("T4 must succeed");
    let report = &result.result["report"];
    let reported_k = report["k_per_model"].as_u64().unwrap();
    assert_eq!(
        reported_k, K_MAX as u64,
        "T4 k_per_model must be clamped to K_MAX"
    );
    // samples_used in the per-model row reflects the clamped k (no
    // starvation expected for EATH on this fixture).
    let per_model = report["per_model"].as_array().unwrap();
    assert_eq!(per_model.len(), 1);
    let samples_used = per_model[0]["samples_used"].as_u64().unwrap();
    assert!(
        samples_used <= K_MAX as u64,
        "T4 per-model samples_used must not exceed K_MAX"
    );
}
