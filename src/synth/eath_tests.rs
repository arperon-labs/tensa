//! Phase 1 tests for `EathSurrogate::generate()` — eight named tests per
//! `docs/synth_eath_algorithm.md` §10.
//!
//! All tests use `MemoryStore`. None depend on wall-clock — the run timestamps
//! we DO assert on are taken from `Utc::now()` *during* the run only as a
//! sanity check that the summary's `started_at` exists; we never compare them
//! across runs.

use super::*;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::synth::registry::SurrogateRegistry;
use crate::types::{Entity, EntityType, MaturityLevel};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// Helper: assemble an `EathParams` blob with sensible defaults that the
/// individual tests override field-by-field. Keeps each test focused on the
/// one or two values it actually cares about.
fn base_params(n: usize) -> EathParams {
    EathParams {
        a_t_distribution: vec![0.5; n],
        a_h_distribution: vec![1.0; n],
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
        num_entities: n,
    }
}

fn surrogate_params(seed: u64, num_steps: usize, eath: EathParams) -> SurrogateParams {
    SurrogateParams {
        model: "eath".into(),
        params_json: serde_json::to_value(eath).unwrap(),
        seed,
        num_steps,
        label_prefix: "synth".into(),
    }
}

fn fresh_hypergraph() -> Hypergraph {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    Hypergraph::new(store)
}

/// Read every situation under a narrative_id and return their UUIDs sorted +
/// deterministic-key-sorted member-UUID lists. Used by T1 to verify the
/// (situation_id, sorted_member_uuids) shape matches across two runs.
fn collect_situations_with_members(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Vec<(Uuid, Vec<Uuid>)> {
    let situations = hg.list_situations_by_narrative(narrative_id).unwrap();
    let mut out: Vec<(Uuid, Vec<Uuid>)> = situations
        .iter()
        .map(|s| {
            let parts = hg.get_participants_for_situation(&s.id).unwrap();
            let mut member_ids: Vec<Uuid> = parts.iter().map(|p| p.entity_id).collect();
            member_ids.sort();
            (s.id, member_ids)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

// ── T1: deterministic by seed ──────────────────────────────────────────────

#[test]
fn test_eath_deterministic_by_seed() {
    let p = base_params(10);
    let sp = surrogate_params(12345, 50, p);

    let hg_a = fresh_hypergraph();
    let hg_b = fresh_hypergraph();

    let s_a = EathSurrogate.generate(&sp, &hg_a, "narr-A").unwrap();
    let s_b = EathSurrogate.generate(&sp, &hg_b, "narr-B").unwrap();

    assert_eq!(s_a.num_situations, s_b.num_situations);
    assert_eq!(s_a.num_participations, s_b.num_participations);
    assert_eq!(s_a.num_entities, s_b.num_entities);

    // Entity UUID vectors come out of the deterministic builder — must match.
    let entities_a = hg_a.list_entities_by_narrative("narr-A").unwrap();
    let entities_b = hg_b.list_entities_by_narrative("narr-B").unwrap();
    let mut uuids_a: Vec<Uuid> = entities_a.iter().map(|e| e.id).collect();
    let mut uuids_b: Vec<Uuid> = entities_b.iter().map(|e| e.id).collect();
    uuids_a.sort();
    uuids_b.sort();
    assert_eq!(uuids_a, uuids_b);

    // Sorted (situation_id, sorted_member_uuids) pairs must match exactly.
    let groups_a = collect_situations_with_members(&hg_a, "narr-A");
    let groups_b = collect_situations_with_members(&hg_b, "narr-B");
    assert_eq!(groups_a, groups_b);
}

// ── T2: situations in proportion to activity ───────────────────────────────

#[test]
fn test_eath_produces_situations_in_proportion_to_activity() {
    let n = 20;
    let mut p = base_params(n);
    // First 10 are high-activity, last 10 are low-activity.
    for i in 0..n {
        p.a_t_distribution[i] = if i < 10 { 0.9 } else { 0.1 };
    }
    p.lambda_schedule = vec![1.0]; // keep transitions hot
    p.rho_low = 0.5;
    p.rho_high = 0.1; // longer high-phase dwell
    let sp = surrogate_params(42, 200, p);
    let hg = fresh_hypergraph();
    let _summary = EathSurrogate.generate(&sp, &hg, "T2").unwrap();

    let entities = hg.list_entities_by_narrative("T2").unwrap();
    let mut by_index: std::collections::HashMap<Uuid, usize> = std::collections::HashMap::new();
    for e in &entities {
        if let Some(idx) = e
            .properties
            .get("synth_index")
            .and_then(|v| v.as_u64())
        {
            by_index.insert(e.id, idx as usize);
        }
    }

    let situations = hg.list_situations_by_narrative("T2").unwrap();
    let mut counts = vec![0_usize; n];
    for s in &situations {
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        for p in &parts {
            if let Some(&idx) = by_index.get(&p.entity_id) {
                counts[idx] += 1;
            }
        }
    }

    let high_mean: f32 =
        counts.iter().take(10).map(|&c| c as f32).sum::<f32>() / 10.0;
    let low_mean: f32 =
        counts.iter().skip(10).take(10).map(|&c| c as f32).sum::<f32>() / 10.0;
    let ratio = if low_mean > 0.0 {
        high_mean / low_mean
    } else {
        f32::INFINITY
    };
    assert!(
        ratio >= 2.5,
        "expected high/low participation ratio >= 2.5, got {ratio} (high_mean={high_mean}, low_mean={low_mean})"
    );
}

// ── T3: short-term memory continuation rate matches p ──────────────────────

#[test]
fn test_eath_short_term_memory_continuation_rate_matches_p() {
    // With only 10 entities + group size 2-4, baseline overlap rate from
    // pure-random recruitment is already ~50%, so the test compares the two
    // p_from_scratch settings against each other (relative differential)
    // rather than against an absolute ratio of 0.5/0.3. This still proves
    // continuation strengthens recall vs pure freshness, which is the
    // actual EATH-vs-null contract.
    fn run(p_from_scratch: f32) -> f32 {
        let mut p = base_params(30);
        p.p_from_scratch = p_from_scratch;
        p.stm_capacity = 5;
        p.lambda_schedule = vec![1.0];
        // Force entities into high phase so groups actually form.
        p.rho_low = 1.0;
        p.rho_high = 0.05;
        // Damp LTM influence so from-scratch is genuinely "fresh" picks
        // weighted only by current activity (not by accumulated co-mems).
        p.omega_decay = 0.1;
        let sp = surrogate_params(99, 500, p);
        let hg = fresh_hypergraph();
        EathSurrogate.generate(&sp, &hg, "T3").unwrap();

        let mut situations = hg.list_situations_by_narrative("T3").unwrap();
        situations.sort_by_key(|s| s.temporal.start);

        let mut prev_members: Option<std::collections::HashSet<Uuid>> = None;
        let mut continued = 0_usize;
        let mut total = 0_usize;
        for s in &situations {
            let parts = hg.get_participants_for_situation(&s.id).unwrap();
            let here: std::collections::HashSet<Uuid> =
                parts.iter().map(|p| p.entity_id).collect();
            if let Some(prev) = &prev_members {
                let overlap = here.intersection(prev).count();
                if !here.is_empty() && (overlap as f32 / here.len() as f32) >= 0.5 {
                    continued += 1;
                }
                total += 1;
            }
            prev_members = Some(here);
        }
        if total == 0 {
            0.0
        } else {
            continued as f32 / total as f32
        }
    }

    let frac_low_p = run(0.1); // mostly continue ⇒ high continuation rate
    let frac_high_p = run(0.9); // mostly fresh ⇒ low continuation rate

    assert!(
        frac_low_p > frac_high_p + 0.10,
        "expected continuation fraction(p=0.1) to exceed continuation fraction(p=0.9) by > 0.10, got {frac_low_p} vs {frac_high_p}"
    );
    assert!(
        frac_low_p >= 0.4,
        "expected continuation fraction >= 0.4 at p_from_scratch=0.1, got {frac_low_p}"
    );
}

// ── T4: handles all-zero activity ──────────────────────────────────────────

#[test]
fn test_eath_handles_all_zero_activity() {
    let mut p = base_params(5);
    p.a_t_distribution = vec![0.0; 5];
    p.a_h_distribution = vec![0.0; 5];
    let sp = surrogate_params(0, 100, p);

    let hg = fresh_hypergraph();
    let s = EathSurrogate.generate(&sp, &hg, "T4").unwrap();
    assert_eq!(s.num_situations, 0);
    assert_eq!(s.num_participations, 0);
    let entities = hg.list_entities_by_narrative("T4").unwrap();
    assert_eq!(entities.len(), 5);
}

// ── T5: errors on zero (and on degenerate input) ───────────────────────────

#[test]
fn test_eath_errors_on_zero_entities() {
    // Single-entity case.
    let mut p = base_params(1);
    p.a_t_distribution = vec![1.0];
    p.a_h_distribution = vec![1.0];
    p.num_entities = 1;
    let sp = surrogate_params(0, 10, p);
    let hg = fresh_hypergraph();
    let r = EathSurrogate.generate(&sp, &hg, "T5a");
    assert!(matches!(r, Err(TensaError::SynthFailure(_))));
    assert!(hg.list_entities_by_narrative("T5a").unwrap().is_empty());

    // Empty case.
    let mut p2 = base_params(0);
    p2.a_t_distribution = vec![];
    p2.a_h_distribution = vec![];
    let sp2 = surrogate_params(0, 10, p2);
    let hg2 = fresh_hypergraph();
    let r2 = EathSurrogate.generate(&sp2, &hg2, "T5b");
    assert!(matches!(r2, Err(TensaError::SynthFailure(_))));

    // NaN case.
    let mut p3 = base_params(3);
    p3.a_t_distribution = vec![0.5, f32::NAN, 0.3];
    let sp3 = surrogate_params(0, 10, p3);
    let hg3 = fresh_hypergraph();
    let r3 = EathSurrogate.generate(&sp3, &hg3, "T5c");
    assert!(matches!(r3, Err(TensaError::SynthFailure(_))));
}

// ── T6: phase distribution matches a_T (Spearman ρ ≥ threshold) ────────────

#[test]
#[allow(non_snake_case)] // intentional: matches the design doc §10 test name
fn test_eath_phase_distribution_matches_aT() {
    let n = 20;
    let mut p = base_params(n);
    // Linearly increasing a_T from 0.0 (i=0) to 0.95 (i=19).
    for i in 0..n {
        p.a_t_distribution[i] = i as f32 / 20.0;
    }
    p.a_h_distribution = vec![1.0; n];
    p.lambda_schedule = vec![0.5; 200];
    p.rho_low = 0.2;
    p.rho_high = 0.3;
    let sp = surrogate_params(7, 500, p);
    let hg = fresh_hypergraph();
    EathSurrogate.generate(&sp, &hg, "T6").unwrap();

    // Reconstruct per-entity participation counts as a proxy for high-phase
    // dwell time (more activity ⇒ more participations).
    let entities = hg.list_entities_by_narrative("T6").unwrap();
    let mut by_index: std::collections::HashMap<Uuid, usize> = std::collections::HashMap::new();
    for e in &entities {
        if let Some(idx) = e
            .properties
            .get("synth_index")
            .and_then(|v| v.as_u64())
        {
            by_index.insert(e.id, idx as usize);
        }
    }

    let situations = hg.list_situations_by_narrative("T6").unwrap();
    let mut counts = vec![0_usize; n];
    for s in &situations {
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        for p in &parts {
            if let Some(&idx) = by_index.get(&p.entity_id) {
                counts[idx] += 1;
            }
        }
    }

    // Spearman ρ between a_T (index i / 20) and counts[i].
    let a_t_ranks = ranks(&(0..n).map(|i| i as f32 / 20.0).collect::<Vec<_>>());
    let counts_f: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
    let count_ranks = ranks(&counts_f);
    let rho = pearson(&a_t_ranks, &count_ranks);
    assert!(
        rho >= 0.5,
        "expected Spearman ρ >= 0.5 between a_T and participation counts, got {rho} (counts: {:?})",
        counts
    );
}

// Tied-rank Spearman: convert raw values to mid-ranks then Pearson.
fn ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> =
        values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0 + 1.0;
        for k in i..=j {
            out[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    out
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mx: f32 = x.iter().sum::<f32>() / n;
    let my: f32 = y.iter().sum::<f32>() / n;
    let mut num = 0.0_f32;
    let mut dx = 0.0_f32;
    let mut dy = 0.0_f32;
    for (a, b) in x.iter().zip(y.iter()) {
        let xa = a - mx;
        let yb = b - my;
        num += xa * yb;
        dx += xa * xa;
        dy += yb * yb;
    }
    let denom = (dx * dy).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        num / denom
    }
}

// ── T7: group size distribution matches target ─────────────────────────────

#[test]
fn test_eath_group_size_distribution_matches_target() {
    let n = 15;
    let mut p = base_params(n);
    p.group_size_distribution = vec![10, 5, 2]; // dyads:triads:quads ≈ 59/29/12
    p.a_t_distribution = vec![0.5; n];
    p.a_h_distribution = vec![1.0; n];
    p.p_from_scratch = 1.0; // pure from-scratch — STM bias doesn't pollute sizes
    p.max_group_size = 4;
    // Force entities high so groups actually form.
    p.lambda_schedule = vec![1.0];
    p.rho_low = 1.0;
    p.rho_high = 0.05;
    let sp = surrogate_params(55, 300, p);
    let hg = fresh_hypergraph();
    EathSurrogate.generate(&sp, &hg, "T7").unwrap();

    let situations = hg.list_situations_by_narrative("T7").unwrap();
    assert!(
        situations.len() > 50,
        "T7 needs enough samples to test the distribution (got {})",
        situations.len()
    );
    let mut bins = vec![0_usize; 3];
    let mut total = 0_usize;
    for s in &situations {
        let parts = hg.get_participants_for_situation(&s.id).unwrap();
        let m = parts.len();
        if (2..=4).contains(&m) {
            bins[m - 2] += 1;
            total += 1;
        }
    }
    let target = [10.0 / 17.0, 5.0 / 17.0, 2.0 / 17.0];
    for k in 0..3 {
        let observed = bins[k] as f32 / total.max(1) as f32;
        assert!(
            (observed - target[k]).abs() <= 0.10,
            "size {} fraction off target: observed {} vs target {} (bins={:?}, total={})",
            k + 2,
            observed,
            target[k],
            bins,
            total
        );
    }
}

// ── T8: perf smoke (1000 entities × 100 steps) ─────────────────────────────

#[test]
fn test_eath_perf_smoke() {
    let n = 1000;
    let mut p = base_params(n);
    p.a_t_distribution = vec![1.0 / n as f32; n];
    p.a_h_distribution = vec![1.0; n];
    p.group_size_distribution = vec![1, 1, 1];
    p.p_from_scratch = 0.5;
    p.stm_capacity = 7;
    let sp = surrogate_params(0, 100, p);
    let hg = fresh_hypergraph();
    let t0 = Instant::now();
    let _summary = EathSurrogate.generate(&sp, &hg, "T8").unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_secs() <= 5,
        "perf smoke exceeded 5s budget: {:?}",
        elapsed
    );
}

// ── Internal unit tests (migrated from eath.rs::tests for line-cap reasons) ──

#[test]
fn test_eath_registers_under_canonical_name() {
    let r = SurrogateRegistry::default();
    let m = r.get("eath").unwrap();
    assert_eq!(m.name(), "eath");
    assert_eq!(m.version(), "v0.2");
    assert!(!m.fidelity_metrics().is_empty());
}

#[test]
fn test_validate_params_rejects_short_a_t() {
    let p = EathParams {
        a_t_distribution: vec![1.0],
        ..Default::default()
    };
    assert!(matches!(
        validate_params(&p),
        Err(TensaError::SynthFailure(_))
    ));
}

#[test]
fn test_validate_params_rejects_nan() {
    let p = EathParams {
        a_t_distribution: vec![0.5, f32::NAN, 0.3],
        a_h_distribution: vec![1.0; 3],
        max_group_size: 4,
        ..Default::default()
    };
    assert!(matches!(
        validate_params(&p),
        Err(TensaError::SynthFailure(_))
    ));
}

#[test]
fn test_calibrate_returns_eath_params_for_planted_narrative() {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let hg = Hypergraph::new(store);
    let nid = "calib-test";
    for i in 0..3 {
        let now = chrono::Utc::now();
        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": format!("e{i}")}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some(nid.into()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap();
    }
    let blob = EathSurrogate.calibrate(&hg, nid).unwrap();
    let parsed: EathParams = serde_json::from_value(blob).unwrap();
    assert_eq!(parsed.a_t_distribution.len(), 3);
    assert_eq!(parsed.a_h_distribution.len(), 3);
}

#[test]
fn test_build_entity_ids_is_deterministic() {
    let a = build_entity_ids(5, 12345);
    let b = build_entity_ids(5, 12345);
    assert_eq!(a, b);
    let c = build_entity_ids(5, 12346);
    assert_ne!(a, c);
}
