//! Shared test fixtures for the opinion-dynamics test suite.
//!
//! Kept in a sibling `cfg(test)` module so [`super::opinion_dynamics_tests`]
//! stays focused on assertions and stays under the 500-line cap.

use uuid::Uuid;

use crate::analysis::test_helpers::*;
use crate::hypergraph::Hypergraph;

use super::types::OpinionDynamicsReport;

/// Build a complete hypergraph: every pair of entities forms a size-2
/// situation. `n` entities → `n*(n-1)/2` situations. All entities and
/// situations belong to `narrative_id`.
pub(super) fn build_complete_dyadic_narrative(
    n: usize,
    narrative_id: &str,
) -> (Hypergraph, Vec<Uuid>) {
    let hg = make_hg();
    let entities: Vec<Uuid> = (0..n)
        .map(|i| add_entity(&hg, &format!("E{i}"), narrative_id))
        .collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let sid = add_situation(&hg, narrative_id);
            link(&hg, entities[i], sid);
            link(&hg, entities[j], sid);
        }
    }
    (hg, entities)
}

/// Random-hypergraph fixture: `n` entities, `n_situations` hyperedges, each
/// with `min_size`..`max_size` entities. Uses ChaCha8Rng seeded from `seed`
/// so calls are reproducible across platforms.
pub(super) fn build_random_hypergraph(
    n: usize,
    n_situations: usize,
    min_size: usize,
    max_size: usize,
    narrative_id: &str,
    seed: u64,
) -> (Hypergraph, Vec<Uuid>) {
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let hg = make_hg();
    let entities: Vec<Uuid> = (0..n)
        .map(|i| add_entity(&hg, &format!("E{i}"), narrative_id))
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..n_situations {
        let size = rng.gen_range(min_size..=max_size).min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let chosen: Vec<usize> = indices.into_iter().take(size).collect();
        let sid = add_situation(&hg, narrative_id);
        for idx in chosen {
            link(&hg, entities[idx], sid);
        }
    }
    (hg, entities)
}

/// Build narrative with exactly the supplied list of hyperedges
/// (specified as entity-index slices into a fresh entity set of size `n`).
pub(super) fn build_explicit_narrative(
    n: usize,
    hyperedges: &[&[usize]],
    narrative_id: &str,
) -> (Hypergraph, Vec<Uuid>) {
    let hg = make_hg();
    let entities: Vec<Uuid> = (0..n)
        .map(|i| add_entity(&hg, &format!("E{i}"), narrative_id))
        .collect();
    for edge in hyperedges {
        let sid = add_situation(&hg, narrative_id);
        for &idx in *edge {
            link(&hg, entities[idx], sid);
        }
    }
    (hg, entities)
}

/// Plant label-propagation labels in KV without actually running the
/// algorithm. Used by the echo-chamber test (T5).
pub(super) fn plant_labels(hg: &Hypergraph, narrative_id: &str, labels: &[(Uuid, usize)]) {
    for (eid, label) in labels {
        let key = format!("an/lp/{narrative_id}/{eid}");
        let val = serde_json::to_vec(&(*label as f64)).unwrap();
        hg.store().put(key.as_bytes(), &val).unwrap();
    }
}

/// Convenience: extract (min, max) of `final_opinions` values.
pub(super) fn final_min_max(report: &OpinionDynamicsReport) -> (f32, f32) {
    let vals: Vec<f32> = report.trajectory.final_opinions.values().copied().collect();
    let mn = vals.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mn, mx)
}
