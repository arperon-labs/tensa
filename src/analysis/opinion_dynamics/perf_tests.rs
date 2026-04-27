//! Performance smoke tests for [`super::simulate::simulate_opinion_dynamics`].
//!
//! All tests are `#[ignore]` so CI doesn't run them by default. Invoke
//! manually with:
//!
//! ```text
//! cargo test --release --no-default-features \
//!     --lib analysis::opinion_dynamics::perf_tests \
//!     -- --ignored --nocapture
//! ```
//!
//! Targets (per `docs/opinion_dynamics_algorithm.md` §5):
//! - 100 entities × 10k steps: < 1s
//! - 1000 entities × 100k steps: < 30s

use super::test_fixtures::*;
use super::types::OpinionDynamicsParams;
use super::simulate::simulate_opinion_dynamics;

/// 100 entities × 10k steps target: < 1s.
#[test]
#[ignore]
fn perf_smoke_100_x_10k() {
    let nid = "perf-100";
    let (hg, _) = build_complete_dyadic_narrative(100, nid);
    let params = OpinionDynamicsParams {
        confidence_bound: 0.3,
        seed: 100,
        max_steps: 10_000,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let _ = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    let elapsed = t0.elapsed();
    eprintln!("100 entities × 10k steps: {:?}", elapsed);
    assert!(elapsed.as_secs() < 5, "perf regression: {:?}", elapsed);
}

/// 1000 entities × 100k steps target: < 30s (release mode).
#[test]
#[ignore]
fn perf_target_1000_x_100k() {
    let nid = "perf-1000";
    // Use a sparser fixture — complete dyadic on 1000 would be ~500k
    // situations, prohibitive setup time. Random hyperedges of size 3-8.
    let (hg, _) = build_random_hypergraph(1000, 2000, 3, 8, nid, 1000);
    let params = OpinionDynamicsParams {
        confidence_bound: 0.3,
        seed: 1000,
        max_steps: 100_000,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let _ = simulate_opinion_dynamics(&hg, nid, &params).unwrap();
    let elapsed = t0.elapsed();
    eprintln!("1000 entities × 100k steps: {:?}", elapsed);
    assert!(elapsed.as_secs() < 60, "perf regression: {:?}", elapsed);
}
