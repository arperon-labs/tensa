//! Phase 4 tests for `crate::synth::engines::*` — exercise both new engines
//! end-to-end through the `WorkerPool`, plus a couple of standalone "engine
//! invariants" tests that don't need the queue to be live.
//!
//! These tests never depend on wall-clock — every seed is fixed.

use std::sync::Arc;

use chrono::{Duration, TimeZone, Utc};
use uuid::Uuid;

use crate::inference::cost::estimate_cost;
use crate::inference::types::InferenceJob;
use crate::inference::worker::WorkerPool;
use crate::inference::{jobs::JobQueue, InferenceEngine};
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::calibrate;
use crate::synth::engines::{
    make_synth_engines, SurrogateCalibrationEngine, SurrogateGenerationEngine,
};
use crate::synth::registry::SurrogateRegistry;
use crate::synth::types::EathParams;
use crate::synth::load_reproducibility_blob;
use crate::types::{
    AllenInterval, ContentBlock, Entity, EntityType, ExtractionMethod, InferenceJobType,
    JobPriority, JobStatus, MaturityLevel, NarrativeLevel, Participation, Role, Situation,
    TimeGranularity,
};
use crate::Hypergraph;

// ── Fixtures ────────────────────────────────────────────────────────────────

/// Build a tiny narrative the calibrator can fit. Mirrors the
/// `build_basic_narrative` helper from `calibrate_tests.rs` but lives here
/// to avoid a cross-test-module import.
fn build_basic_narrative(hg: &Hypergraph, narrative_id: &str) {
    let mut entity_ids = Vec::new();
    for i in 0..5 {
        let now = Utc::now();
        let e = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": format!("actor-{i}")}),
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
        };
        let id = e.id;
        hg.create_entity(e).unwrap();
        entity_ids.push(id);
    }
    let epoch = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    for t in 0..10 {
        let now = Utc::now();
        let start = epoch + Duration::seconds(t * 60);
        let s = Situation {
            id: Uuid::now_v7(),
            name: None,
            description: None,
            properties: serde_json::Value::Null,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(start),
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
            raw_content: vec![ContentBlock::text("phase 4 fixture")],
            narrative_level: NarrativeLevel::Event,
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
        };
        let sid = s.id;
        hg.create_situation(s).unwrap();
        let group_size = 2 + (t as usize % 3); // 2, 3, 4, ...
        for k in 0..group_size {
            let eid = entity_ids[(t as usize + k) % entity_ids.len()];
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
}

fn fresh_store_and_hypergraph() -> (Arc<dyn KVStore>, Arc<Hypergraph>) {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let hg = Arc::new(Hypergraph::new(store.clone()));
    (store, hg)
}

fn fresh_hypergraph() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

fn make_calibration_job(narrative_id: &str, model: &str) -> InferenceJob {
    InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type: InferenceJobType::SurrogateCalibration {
            narrative_id: narrative_id.into(),
            model: model.into(),
        },
        target_id: Uuid::now_v7(),
        parameters: serde_json::json!({}),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 1_000,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    }
}

fn make_generation_job(
    source_narrative_id: Option<&str>,
    output_narrative_id: &str,
    model: &str,
    params_override: Option<serde_json::Value>,
    seed_override: Option<u64>,
    num_steps: usize,
) -> InferenceJob {
    InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type: InferenceJobType::SurrogateGeneration {
            source_narrative_id: source_narrative_id.map(str::to_owned),
            output_narrative_id: output_narrative_id.into(),
            model: model.into(),
            params_override,
            seed_override,
        },
        target_id: Uuid::now_v7(),
        parameters: serde_json::json!({"num_steps": num_steps}),
        priority: JobPriority::Normal,
        status: JobStatus::Pending,
        estimated_cost_ms: 1_000,
        created_at: Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    }
}

/// Shared fixture: a registered, default WorkerPool with synth engines wired.
fn pool_with_synth_engines(
    job_queue: Arc<JobQueue>,
    hg: Arc<Hypergraph>,
) -> WorkerPool {
    let mut pool = WorkerPool::new(job_queue, hg, 1);
    let registry = Arc::new(SurrogateRegistry::default());
    let (cal, gen, hybrid) = make_synth_engines(registry);
    pool.register_engine(cal);
    pool.register_engine(gen);
    pool.register_engine(hybrid);
    pool
}

// ── T1 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_calibration_engine_registered_in_default_pool() {
    let (store, hg) = fresh_store_and_hypergraph();
    let queue = Arc::new(JobQueue::new(store));
    let pool = pool_with_synth_engines(queue, hg);

    // Field values are sentinel — discriminant-keyed lookup matches anyway.
    let probe = InferenceJobType::SurrogateCalibration {
        narrative_id: "anything".into(),
        model: "anything".into(),
    };
    assert!(
        pool.has_engine(&probe),
        "SurrogateCalibrationEngine must be registered (discriminant lookup)"
    );
}

// ── T2 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_generation_engine_registered_in_default_pool() {
    let (store, hg) = fresh_store_and_hypergraph();
    let queue = Arc::new(JobQueue::new(store));
    let pool = pool_with_synth_engines(queue, hg);

    let probe = InferenceJobType::SurrogateGeneration {
        source_narrative_id: Some("foo".into()),
        output_narrative_id: "bar".into(),
        model: "eath".into(),
        params_override: None,
        seed_override: None,
    };
    assert!(
        pool.has_engine(&probe),
        "SurrogateGenerationEngine must be registered (discriminant lookup)"
    );
}

// ── T3 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_calibration_job_round_trip() {
    let hg = fresh_hypergraph();
    let nid = "calib-rt";
    build_basic_narrative(&hg, nid);

    let registry = Arc::new(SurrogateRegistry::default());
    let cal = SurrogateCalibrationEngine::new(registry);
    let job = make_calibration_job(nid, "eath");

    let result = cal.execute(&job, &hg).expect("calibration must succeed");
    assert_eq!(result.status, JobStatus::Completed);

    let kind = result.result["kind"].as_str().unwrap_or("");
    assert_eq!(kind, "calibration_done");

    // Fidelity report MUST be present in the result envelope.
    assert!(
        result.result["fidelity_report"].is_object(),
        "fidelity_report must be in the InferenceResult JSON"
    );
    assert!(result.result["params_summary"].is_object());
    assert_eq!(result.result["narrative_id"].as_str(), Some(nid));
    assert_eq!(result.result["model"].as_str(), Some("eath"));

    // Params + fidelity report must have been persisted.
    let stored_params = calibrate::load_params(hg.store(), nid, "eath").unwrap();
    assert!(
        stored_params.is_some(),
        "params must be persisted at syn/p/{{nid}}/eath"
    );
}

// ── T4 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_generation_job_round_trip() {
    let hg = fresh_hypergraph();
    let nid = "gen-rt";
    build_basic_narrative(&hg, nid);

    let registry = Arc::new(SurrogateRegistry::default());

    // Calibrate first so generation can load params from storage.
    let cal = SurrogateCalibrationEngine::new(registry.clone());
    let cal_job = make_calibration_job(nid, "eath");
    cal.execute(&cal_job, &hg).unwrap();

    let gen_engine = SurrogateGenerationEngine::new(registry);
    let output = "synthetic-rt";
    let gen_job = make_generation_job(Some(nid), output, "eath", None, Some(42), 50);

    let result = gen_engine
        .execute(&gen_job, &hg)
        .expect("generation must succeed");
    assert_eq!(result.status, JobStatus::Completed);
    assert_eq!(result.result["kind"].as_str(), Some("generation_done"));
    assert_eq!(result.result["output_narrative_id"].as_str(), Some(output));

    // run_summary is the source of truth for "what got created".
    let summary = &result.result["run_summary"];
    assert!(summary.is_object(), "run_summary must be an object");
    let num_entities = summary["num_entities"].as_u64().unwrap_or(0) as usize;
    // We calibrated against a 5-entity narrative, so the generated narrative
    // should also have 5 entities (calibrated num_entities round-trips).
    assert_eq!(
        num_entities, 5,
        "generated num_entities should match calibrated source size"
    );

    // Output narrative actually has the entities + situations in the hypergraph.
    let entities = hg.list_entities_by_narrative(output).unwrap();
    assert_eq!(entities.len(), 5);
}

// ── T5 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_generation_job_uses_seed_override_when_provided() {
    let hg = fresh_hypergraph();
    let nid = "seed-test";
    build_basic_narrative(&hg, nid);

    let registry = Arc::new(SurrogateRegistry::default());
    let cal = SurrogateCalibrationEngine::new(registry.clone());
    cal.execute(&make_calibration_job(nid, "eath"), &hg).unwrap();

    let gen_engine = SurrogateGenerationEngine::new(registry);

    // Two runs with DIFFERENT seeds — outputs to different narratives so they
    // don't collide. The run_id is derived from the seed (see
    // `eath::run_id_from_seed`), so different seeds ⇒ different run_id.
    let job_a = make_generation_job(Some(nid), "out-a", "eath", None, Some(123), 30);
    let result_a = gen_engine.execute(&job_a, &hg).unwrap();
    let run_id_a = result_a.result["run_summary"]["run_id"]
        .as_str()
        .unwrap()
        .to_string();

    let job_b = make_generation_job(Some(nid), "out-b", "eath", None, Some(456), 30);
    let result_b = gen_engine.execute(&job_b, &hg).unwrap();
    let run_id_b = result_b.result["run_summary"]["run_id"]
        .as_str()
        .unwrap()
        .to_string();

    assert_ne!(
        run_id_a, run_id_b,
        "different seeds must produce different run_ids"
    );

    // Two runs with the SAME seed — outputs to different narratives. Same
    // seed ⇒ identical num_situations + num_participations (the
    // determinism contract from Phase 1 — same seed produces same trace).
    let job_c = make_generation_job(Some(nid), "out-c", "eath", None, Some(789), 30);
    let result_c = gen_engine.execute(&job_c, &hg).unwrap();
    let job_d = make_generation_job(Some(nid), "out-d", "eath", None, Some(789), 30);
    let result_d = gen_engine.execute(&job_d, &hg).unwrap();

    let n_sit_c = result_c.result["run_summary"]["num_situations"]
        .as_u64()
        .unwrap();
    let n_sit_d = result_d.result["run_summary"]["num_situations"]
        .as_u64()
        .unwrap();
    let n_part_c = result_c.result["run_summary"]["num_participations"]
        .as_u64()
        .unwrap();
    let n_part_d = result_d.result["run_summary"]["num_participations"]
        .as_u64()
        .unwrap();
    assert_eq!(
        n_sit_c, n_sit_d,
        "same seed must produce identical num_situations across runs"
    );
    assert_eq!(
        n_part_c, n_part_d,
        "same seed must produce identical num_participations across runs"
    );
}

// ── T6 ──────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "TENSA's WorkerPool does not yet expose per-job cancellation tokens. \
            Logged as a Phase 12.5 follow-up — see docs/EATH_sprint.md Notes."]
fn test_generation_job_cancellation_returns_clean_error() {
    // When cancellation lands, this test will:
    //   1. Submit a long-running generation job (e.g. num_steps = 1_000_000).
    //   2. Mid-run, call queue.cancel(job_id) or pool's cancellation API.
    //   3. Wait for the job to finish.
    //   4. Assert the resulting InferenceResult / job error string indicates
    //      cancellation (e.g. "cancelled by user"), not panic.
    //
    // The current `WorkerPool` runs each engine to completion under
    // `tokio::task::spawn_blocking` with no per-task cancel signal. Adding it
    // requires plumbing an `Arc<AtomicBool>` (or `CancellationToken`) into
    // the `InferenceEngine::execute` signature — out of scope for Phase 4.
}

// ── T7 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_generation_writes_seed_blob_before_starting() {
    // We can't observe "before starting" mid-execution from a unit test (the
    // engine call is synchronous), but we CAN verify post-completion that
    // the ReproducibilityBlob exists at `syn/seed/{run_id}` — and the
    // `eath::run_generate_with_source` body writes it BEFORE the main loop
    // (see step 5 in src/synth/eath.rs::run_generate_with_source). That code
    // contract is tested via the invariant suite
    // (`test_reproducibility_blob_written_for_run`). This test verifies the
    // engine path preserves the contract.
    let hg = fresh_hypergraph();
    let nid = "blob-test";
    build_basic_narrative(&hg, nid);

    let registry = Arc::new(SurrogateRegistry::default());
    let cal = SurrogateCalibrationEngine::new(registry.clone());
    cal.execute(&make_calibration_job(nid, "eath"), &hg).unwrap();

    let gen_engine = SurrogateGenerationEngine::new(registry);
    let result = gen_engine
        .execute(
            &make_generation_job(Some(nid), "blob-out", "eath", None, Some(0xCAFE), 20),
            &hg,
        )
        .unwrap();
    let run_id_str = result.result["run_summary"]["run_id"].as_str().unwrap();
    let run_id = Uuid::parse_str(run_id_str).unwrap();

    let blob = load_reproducibility_blob(hg.store(), &run_id)
        .unwrap()
        .expect("ReproducibilityBlob must be persisted at syn/seed/{run_id}");
    assert_eq!(blob.run_id, run_id);
    assert_eq!(blob.seed, 0xCAFE);
    // Source narrative was provided → source_state_hash MUST be populated.
    assert!(
        blob.source_state_hash.is_some(),
        "source_state_hash must be set when source_narrative_id is provided"
    );
}

// ── T8 ──────────────────────────────────────────────────────────────────────

#[test]
fn test_calibration_and_generation_use_different_cost_classes() {
    // Build a small narrative so the cost helpers have something to size against.
    let hg = fresh_hypergraph();
    let nid = "cost-cmp";
    build_basic_narrative(&hg, nid);

    let cal_job = make_calibration_job(nid, "eath");
    let cal_cost = estimate_cost(&cal_job, &hg).unwrap();

    // Generation cost scales linearly with num_entities × num_steps (see
    // `estimate_generation_cost` — ms per entity-step is 1/GEN_MS_PER_ENTITY_STEP_DENOM).
    // Calibration is bounded by the small source narrative, so a large-enough
    // generation job MUST exceed it. We use a params_override carrying a
    // large num_entities so the formula has something to scale against
    // independent of the source size.
    let big_params = EathParams {
        a_t_distribution: vec![0.5; 1_000],
        a_h_distribution: vec![1.0; 1_000],
        lambda_schedule: vec![],
        p_from_scratch: 0.5,
        omega_decay: 0.95,
        group_size_distribution: vec![1, 1, 1],
        rho_low: 0.5,
        rho_high: 0.3,
        xi: 1.0,
        order_propensity: vec![],
        max_group_size: 4,
        stm_capacity: 7,
        num_entities: 1_000,
    };
    let mut gen_job = make_generation_job(
        None,
        "cost-out",
        "eath",
        Some(serde_json::to_value(&big_params).unwrap()),
        None,
        1_000_000,
    );
    gen_job.parameters = serde_json::json!({"num_steps": 1_000_000});
    let gen_cost = estimate_cost(&gen_job, &hg).unwrap();

    assert!(
        gen_cost > cal_cost,
        "generation cost ({gen_cost}) should exceed calibration cost ({cal_cost}) at \
         num_entities = 1000, num_steps = 1M; calibration is bounded by source \
         size, generation by num_entities × num_steps"
    );
}

// ── Bonus: standalone engine-level tests ────────────────────────────────────
//
// Not part of the named T1-T8 but cheap to add — make sure error paths return
// the right TensaError variants so future grammar/API work has a stable
// contract to test against.

#[test]
fn test_calibration_engine_rejects_unknown_model() {
    let hg = fresh_hypergraph();
    build_basic_narrative(&hg, "n");
    let registry = Arc::new(SurrogateRegistry::default());
    let cal = SurrogateCalibrationEngine::new(registry);
    let mut job = make_calibration_job("n", "not-a-real-model");
    job.target_id = Uuid::now_v7();
    let err = cal.execute(&job, &hg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown surrogate model"),
        "expected unknown-model error, got: {msg}"
    );
}

#[test]
fn test_generation_engine_errors_when_no_params_and_no_source() {
    let hg = fresh_hypergraph();
    let registry = Arc::new(SurrogateRegistry::default());
    let gen = SurrogateGenerationEngine::new(registry);
    let job = make_generation_job(None, "out", "eath", None, Some(1), 10);
    let err = gen.execute(&job, &hg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("no params_override")
            || msg.contains("source_narrative_id")
            || msg.contains("calibrate first"),
        "expected missing-source-or-override error, got: {msg}"
    );
}

#[test]
fn test_generation_engine_accepts_inline_params_override() {
    // Build a hand-tuned EathParams blob so the engine doesn't need to
    // load anything from KV. Verifies the params_override branch works
    // without a calibrated source.
    let hg = fresh_hypergraph();
    let registry = Arc::new(SurrogateRegistry::default());
    let gen = SurrogateGenerationEngine::new(registry);

    let inline = EathParams {
        a_t_distribution: vec![0.5; 4],
        a_h_distribution: vec![1.0; 4],
        lambda_schedule: vec![],
        p_from_scratch: 0.5,
        omega_decay: 0.95,
        group_size_distribution: vec![1, 1, 1],
        rho_low: 0.5,
        rho_high: 0.3,
        xi: 1.0,
        order_propensity: vec![],
        max_group_size: 4,
        stm_capacity: 7,
        num_entities: 4,
    };
    let params_json = serde_json::to_value(&inline).unwrap();
    let job = make_generation_job(None, "inline-out", "eath", Some(params_json), Some(7), 10);
    let result = gen.execute(&job, &hg).unwrap();
    assert_eq!(result.result["kind"].as_str(), Some("generation_done"));
    let n_entities = result.result["run_summary"]["num_entities"]
        .as_u64()
        .unwrap();
    assert_eq!(n_entities, 4, "inline params produced 4 entities as requested");
}
