//! NuDHy state + MCMC unit tests (T1-T5, T9). T6, T7, T8 live alongside
//! the surrogate impl — see `nudhy_surrogate_tests.rs` and the registry test.

// `#[path = "nudhy_tests.rs"] mod nudhy_tests;` lives inside `nudhy.rs`, so
// `super::*` refers to `synth::nudhy` directly — no extra path segment.
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, HashMap};
use uuid::Uuid;

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Build a deterministic hyperedge list with `num_edges` edges over
/// `num_entities` entities. Each edge is constructed with a rotation pattern
/// so the sample is dense enough for many valid double-swaps.
fn build_dense_hypergraph(num_entities: usize, num_edges: usize, edge_size: usize) -> Vec<Vec<Uuid>> {
    let entities: Vec<Uuid> = (0..num_entities)
        .map(|i| {
            // Build deterministic UUIDs for reproducible degree counts.
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&(i as u64).to_be_bytes());
            bytes[6] = (bytes[6] & 0x0f) | 0x40;
            bytes[8] = (bytes[8] & 0x3f) | 0x80;
            Uuid::from_bytes(bytes)
        })
        .collect();
    let mut edges = Vec::with_capacity(num_edges);
    for i in 0..num_edges {
        let mut e = Vec::with_capacity(edge_size);
        for k in 0..edge_size {
            e.push(entities[(i + k) % num_entities]);
        }
        edges.push(e);
    }
    edges
}

fn count_edges_containing(state: &NudhyState, v: Uuid) -> u32 {
    state
        .hyperedges
        .iter()
        .filter(|e: &&Vec<Uuid>| e.binary_search(&v).is_ok())
        .count() as u32
}

fn sorted_size_multiset(state: &NudhyState) -> Vec<usize> {
    let mut sizes: Vec<usize> = state
        .hyperedges
        .iter()
        .map(|e: &Vec<Uuid>| e.len())
        .collect();
    sizes.sort_unstable();
    sizes
}

// ── T1: degree-sequence preservation ─────────────────────────────────────────

#[test]
fn test_nudhy_mcmc_preserves_entity_degree() {
    let raw = build_dense_hypergraph(5, 10, 3);
    let mut state = NudhyState::from_hyperedges(raw);

    // Snapshot original degrees.
    let original: HashMap<Uuid, u32> = state.entity_degree.clone();

    let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
    for _ in 0..10_000 {
        nudhy_mcmc_step(&mut state, &mut rng);
    }

    // Recompute degrees from scratch and assert equality with both the
    // running counter and the original snapshot.
    for (&v, &original_deg) in &original {
        let recount = count_edges_containing(&state, v);
        assert_eq!(
            recount, original_deg,
            "entity {} degree drifted: original={}, recount={}",
            v, original_deg, recount
        );
        assert_eq!(
            state.entity_degree[&v], original_deg,
            "running entity_degree drifted for {}", v
        );
    }
}

// ── T2: edge-size-sequence preservation ──────────────────────────────────────

#[test]
fn test_nudhy_mcmc_preserves_edge_size_sequence() {
    let raw = build_dense_hypergraph(7, 12, 4);
    let mut state = NudhyState::from_hyperedges(raw);
    let original_sizes = sorted_size_multiset(&state);

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for _ in 0..10_000 {
        nudhy_mcmc_step(&mut state, &mut rng);
    }

    let final_sizes = sorted_size_multiset(&state);
    assert_eq!(
        original_sizes, final_sizes,
        "sorted edge-size multiset must be byte-identical after MCMC"
    );
}

// ── T3: chain actually randomises (not isomorphic to input) ──────────────────
//
// The Phase 13b spec lists T3 as "two seeds diverge after burn-in"; we keep
// that as T4 and add this stricter "not isomorphic" check from §12 #T3 of the
// design doc — it catches a chain that runs but never accepts.

#[test]
fn test_nudhy_output_not_isomorphic_to_input_after_burn_in() {
    let raw = build_dense_hypergraph(10, 20, 4);
    let mut state = NudhyState::from_hyperedges(raw.clone());
    let burn_in = std::cmp::max(10_000, 10 * state.sum_of_edge_sizes());

    let mut rng = ChaCha8Rng::seed_from_u64(0x1234_5678);
    for _ in 0..burn_in {
        nudhy_mcmc_step(&mut state, &mut rng);
    }

    // Compare multiset of sorted-tuple edges (frozenset semantics).
    let mut before: Vec<BTreeSet<Uuid>> = raw
        .iter()
        .map(|e: &Vec<Uuid>| e.iter().copied().collect::<BTreeSet<Uuid>>())
        .collect();
    let mut after: Vec<BTreeSet<Uuid>> = state
        .hyperedges
        .iter()
        .map(|e: &Vec<Uuid>| e.iter().copied().collect::<BTreeSet<Uuid>>())
        .collect();
    before.sort_by_key(|s: &BTreeSet<Uuid>| s.iter().copied().collect::<Vec<Uuid>>());
    after.sort_by_key(|s: &BTreeSet<Uuid>| s.iter().copied().collect::<Vec<Uuid>>());
    assert_ne!(
        before, after,
        "post-burn-in hyperedge multiset must differ from source — chain failed to randomise"
    );
}

// ── T4: two seeds diverge after burn-in ──────────────────────────────────────

#[test]
fn test_nudhy_two_seeds_diverge_after_burn_in() {
    let raw = build_dense_hypergraph(12, 20, 4);
    let mut state_a = NudhyState::from_hyperedges(raw.clone());
    let mut state_b = NudhyState::from_hyperedges(raw);
    let burn_in = std::cmp::max(10_000, 10 * state_a.sum_of_edge_sizes());

    let mut rng_a = ChaCha8Rng::seed_from_u64(11);
    let mut rng_b = ChaCha8Rng::seed_from_u64(22);
    for _ in 0..burn_in {
        nudhy_mcmc_step(&mut state_a, &mut rng_a);
        nudhy_mcmc_step(&mut state_b, &mut rng_b);
    }

    let mut diff = false;
    for (a, b) in state_a.hyperedges.iter().zip(state_b.hyperedges.iter()) {
        if a != b {
            diff = true;
            break;
        }
    }
    assert!(
        diff,
        "two chains with different seeds must produce at least one differing hyperedge"
    );
}

// ── T5: starvation detection on rigid input ──────────────────────────────────

#[test]
fn test_nudhy_detects_mcmc_starvation_on_rigid_input() {
    // All 6 hyperedges contain the same 4 entities — every proposed swap
    // creates a duplicate (v1 is already in e2 because e1 == e2 set-wise).
    let entities: Vec<Uuid> = (0..4)
        .map(|i| {
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&(i as u64).to_be_bytes());
            bytes[6] = (bytes[6] & 0x0f) | 0x40;
            bytes[8] = (bytes[8] & 0x3f) | 0x80;
            Uuid::from_bytes(bytes)
        })
        .collect();
    let raw: Vec<Vec<Uuid>> = (0..6).map(|_| entities.clone()).collect();
    let state = NudhyState::from_hyperedges(raw);

    let initial_state_json = serde_json::to_value(&state.hyperedges).unwrap();
    let params = NudhyParams {
        burn_in_steps: 5_000,
        sample_gap_steps: 0,
        accept_rejection_rate_min: 0.99, // artificially high to force the trip
        initial_state_json,
        fixed_edges_json: serde_json::Value::Null,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    let result = run_nudhy_chain(state, &params, &mut rng);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected starvation error on fully-rigid input"),
    };
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("starvation"),
        "error message must mention starvation: {msg}"
    );
}

// ── Determinism: same seed → same swap trace ─────────────────────────────────

#[test]
fn test_nudhy_state_seed_replay_is_bit_identical() {
    let raw = build_dense_hypergraph(8, 15, 3);
    let mut state_a = NudhyState::from_hyperedges(raw.clone());
    let mut state_b = NudhyState::from_hyperedges(raw);
    let mut rng_a = ChaCha8Rng::seed_from_u64(0x600D_5EED);
    let mut rng_b = ChaCha8Rng::seed_from_u64(0x600D_5EED);
    for _ in 0..2_000 {
        nudhy_mcmc_step(&mut state_a, &mut rng_a);
        nudhy_mcmc_step(&mut state_b, &mut rng_b);
    }
    assert_eq!(state_a.hyperedges, state_b.hyperedges);
}

// ── from_source_state defaults sanity ────────────────────────────────────────

#[test]
fn test_params_from_source_state_uses_design_doc_defaults() {
    // 10 edges × size 4 → sum_sizes = 40
    let raw = build_dense_hypergraph(8, 10, 4);
    let state = NudhyState::from_hyperedges(raw);
    let p = NudhyParams::from_source_state(&state, vec![]).unwrap();
    assert_eq!(p.burn_in_steps, std::cmp::max(10_000, 10 * 40));
    assert_eq!(p.sample_gap_steps, std::cmp::max(1_000, 40));
    assert!((p.accept_rejection_rate_min - 0.01).abs() < f32::EPSILON);
}

// ── T9: perf smoke (1000 situations × 100k steps < 30s) ──────────────────────

#[test]
#[ignore = "perf smoke — run manually with --ignored"]
fn test_nudhy_perf_smoke() {
    let raw = build_dense_hypergraph(200, 1_000, 5);
    let mut state = NudhyState::from_hyperedges(raw);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        nudhy_mcmc_step(&mut state, &mut rng);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 30,
        "100k steps on 1000 situations took {:?} (>30s budget)",
        elapsed
    );
}
