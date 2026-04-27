//! EATH Extension Phase 14 — bistability-significance tests.
//!
//! T6: bistability-significance against surrogates returns sensible quantiles
//! when the source narrative has planted higher-order structure.
//!
//! T1-T5 live in [`crate::analysis::contagion_bistability`] (the algorithmic
//! tests). This file covers the engine + KV persistence + surrogate-comparison
//! end-to-end.
//!
//! All tests use `MemoryStore`. The engine builds its own ephemeral
//! hypergraphs for surrogate samples.

use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use uuid::Uuid;

use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::registry::SurrogateRegistry;
use crate::synth::SurrogateBistabilitySignificanceEngine;
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
        raw_content: vec![ContentBlock::text("bistab fixture")],
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

/// Build a narrative with planted higher-order structure: `n_cliques`
/// disjoint groups of `clique_size` actors, each connected via two
/// hyperedges (one full clique + one trio). Designed so the bistability
/// sweep with threshold ≥ 2 produces a non-trivial gap on the source.
fn seed_planted_simplices(hg: &Hypergraph, narrative_id: &str) {
    let n_cliques = 3;
    let clique_size = 4;
    for c in 0..n_cliques {
        let mut clique = Vec::with_capacity(clique_size);
        for k in 0..clique_size {
            let id = hg
                .create_entity(make_entity(narrative_id, &format!("c{c}-{k}")))
                .unwrap();
            clique.push(id);
        }
        // Full clique hyperedge.
        let s_full = hg
            .create_situation(make_situation(narrative_id, (c as i64) * 60))
            .unwrap();
        for &m in &clique {
            hg.add_participant(Participation {
                entity_id: m,
                situation_id: s_full,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
        // Trio hyperedge.
        let s_trio = hg
            .create_situation(make_situation(narrative_id, (c as i64) * 60 + 30))
            .unwrap();
        for &m in clique.iter().take(3) {
            hg.add_participant(Participation {
                entity_id: m,
                situation_id: s_trio,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }
}

fn engine() -> SurrogateBistabilitySignificanceEngine {
    SurrogateBistabilitySignificanceEngine::new(Arc::new(SurrogateRegistry::default()))
}

fn make_bistability_job(
    narrative_id: &str,
    params: serde_json::Value,
    k: u16,
    models: Vec<String>,
) -> InferenceJob {
    InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type: InferenceJobType::SurrogateBistabilitySignificance {
            narrative_id: narrative_id.into(),
            params,
            k,
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

// ── T6 ──────────────────────────────────────────────────────────────────────

/// T6 — Construct a narrative with planted higher-order structure, run
/// bistability-significance with K=10 against EATH only (NuDHy MCMC needs
/// more variance in the source than this fixture provides), and assert that
/// the per-model report is well-formed: at least one row, observed
/// quantiles in [0, 1], and the report round-trips through KV.
#[test]
fn test_bistability_significance_against_surrogates_returns_sensible_quantiles() {
    let hg = fresh_hg();
    let nid = "t6-bistab";
    seed_planted_simplices(&hg, nid);

    let params = serde_json::json!({
        "beta_0_range": [0.0, 0.5, 6],
        "beta_scaling": {"kind": "custom", "value": [1.2, 1.2, 1.2]},
        "gamma": 0.05,
        "threshold": {"kind": "absolute", "value": 2},
        "initial_prevalence_low": 0.05,
        "initial_prevalence_high": 0.6,
        "steady_state_steps": 80,
        "replicates_per_beta": 2
    });

    // K=10 EATH-only — keeps wall-clock manageable; NuDHy is exercised by
    // Phase 13c's dual-sig tests already.
    let job = make_bistability_job(nid, params, 10, vec!["eath".into()]);
    let result = engine()
        .execute(&job, &hg)
        .expect("T6 bistability-significance must succeed");

    assert_eq!(
        result.result["kind"].as_str(),
        Some("bistability_significance_done"),
        "T6 result envelope kind"
    );
    let report = &result.result["report"];
    let per_model = report["per_model"]
        .as_array()
        .expect("per_model array present");
    assert!(
        !per_model.is_empty(),
        "T6 expected at least one per-model row"
    );

    // Per-model row hygiene: quantiles in [0, 1], samples_used > 0.
    for row in per_model {
        let q_w = row["bistable_interval_width_quantile"]
            .as_f64()
            .unwrap_or(-1.0);
        let q_g = row["max_hysteresis_gap_quantile"].as_f64().unwrap_or(-1.0);
        let n = row["samples_used"].as_u64().unwrap_or(0);
        assert!(
            (0.0..=1.0).contains(&q_w),
            "bistable_interval_width_quantile must be in [0, 1] (got {q_w})"
        );
        assert!(
            (0.0..=1.0).contains(&q_g),
            "max_hysteresis_gap_quantile must be in [0, 1] (got {q_g})"
        );
        assert!(n > 0, "samples_used must be > 0 (got {n})");
    }

    // The combined verdict must echo a finite min_quantile_across_models.
    let min_q = report["combined"]["min_quantile_across_models"]
        .as_f64()
        .unwrap_or(-1.0);
    assert!(
        (0.0..=1.0).contains(&min_q),
        "combined min_quantile_across_models must be in [0, 1] (got {min_q})"
    );

    // Round-trip via KV — confirms persistence path is wired and JSON-stable.
    let run_id_str = report["run_id"].as_str().expect("run_id present");
    let run_id = Uuid::parse_str(run_id_str).expect("valid UUID");
    let loaded = crate::synth::load_bistability_significance_report(
        hg.store(),
        nid,
        &run_id,
    )
    .unwrap()
    .expect("report must round-trip through KV");
    assert_eq!(loaded.run_id, run_id);
    assert_eq!(loaded.narrative_id, nid);
    assert_eq!(loaded.k, 10);
}
