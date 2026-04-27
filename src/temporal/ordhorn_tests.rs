//! Tests for [`crate::temporal::ordhorn`] — Phase 4.
//!
//! Cites: [nebel1995ordhorn].

use super::*;
use crate::types::AllenRelation;

/// Helper: build a constraint with a single basic relation.
fn single(a: usize, b: usize, r: AllenRelation) -> OrdHornConstraint {
    OrdHornConstraint {
        a,
        b,
        relations: vec![r],
    }
}

/// Find the constraint covering pair `(a, b)` in either direction.
/// Returns the relations exactly as stored if `(a, b)` matches; the
/// inverse if `(b, a)` matches.
fn find_pair(net: &OrdHornNetwork, a: usize, b: usize) -> Option<Vec<AllenRelation>> {
    for c in &net.constraints {
        if c.a == a && c.b == b {
            return Some(c.relations.clone());
        }
        if c.a == b && c.b == a {
            return Some(inverse_set(&c.relations));
        }
    }
    None
}

#[test]
fn t1_three_node_satisfiable_propagates_before() {
    // 0 Before 1, 1 Before 2 → 0 Before 2 (via the canonical
    // composition table: Before ∘ Before = {Before}).
    let net = OrdHornNetwork {
        n: 3,
        constraints: vec![
            single(0, 1, AllenRelation::Before),
            single(1, 2, AllenRelation::Before),
        ],
    };
    let closed = closure(&net).expect("closure ok");
    assert!(is_satisfiable(&net), "t1 network should be satisfiable");

    let r02 = find_pair(&closed, 0, 2).expect("(0, 2) constraint missing after closure");
    assert_eq!(
        r02,
        vec![AllenRelation::Before],
        "expected (0, 2) tightened to {{Before}}, got {:?}",
        r02
    );
}

#[test]
fn t2_three_node_unsatisfiable_yields_empty_constraint() {
    // 0 Before 1, 1 Before 2, 0 After 2 → 0 ? 2 must intersect
    // {Before} and {After} → empty.
    let net = OrdHornNetwork {
        n: 3,
        constraints: vec![
            single(0, 1, AllenRelation::Before),
            single(1, 2, AllenRelation::Before),
            single(0, 2, AllenRelation::After),
        ],
    };
    assert!(
        !is_satisfiable(&net),
        "t2 network must be flagged unsatisfiable"
    );

    let closed = closure(&net).expect("closure ok");
    let has_empty = closed.constraints.iter().any(|c| c.relations.is_empty());
    assert!(
        has_empty,
        "expected at least one empty constraint, got {:?}",
        closed.constraints
    );
}

#[test]
fn t3_closure_is_idempotent_on_satisfiable_input() {
    let net = OrdHornNetwork {
        n: 3,
        constraints: vec![
            single(0, 1, AllenRelation::Before),
            single(1, 2, AllenRelation::Before),
        ],
    };
    let once = closure(&net).expect("first closure");
    let twice = closure(&once).expect("second closure");
    assert_eq!(once, twice, "closure must be idempotent");
}

#[test]
fn t4_two_node_crisp_inverse_recoverable() {
    // 0 Before 1 — closure should leave (0, 1) at {Before}; the
    // (1, 0) direction is recoverable as inverse_set({Before}) =
    // {After}.
    let net = OrdHornNetwork {
        n: 2,
        constraints: vec![single(0, 1, AllenRelation::Before)],
    };
    let closed = closure(&net).expect("closure ok");

    let r01 = find_pair(&closed, 0, 1).expect("(0, 1) missing");
    assert_eq!(r01, vec![AllenRelation::Before]);

    let r10 = find_pair(&closed, 1, 0).expect("(1, 0) missing — inverse should be reachable");
    assert_eq!(r10, vec![AllenRelation::After]);

    assert!(is_satisfiable(&net));
}

#[test]
fn t5_eight_node_chain_with_disjunctions_is_satisfiable() {
    // Build a chain of 8 intervals where consecutive ones are
    // explicitly Before, with one weakened constraint expressed as a
    // disjunction. Path-consistency must propagate enough to keep
    // every cell non-empty. This exercises the closure on a network
    // strictly larger than the trivial 3-node fixtures of T1-T2.
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(0x4ad08e9_a4_u64);
    let n: usize = 8;

    let mut constraints: Vec<OrdHornConstraint> = Vec::new();
    // Chain Before constraints.
    for i in 0..(n - 1) {
        constraints.push(single(i, i + 1, AllenRelation::Before));
    }
    // Inject a wider disjunction on one non-adjacent pair to force
    // intersection-based tightening rather than pure crisp transit.
    let weakened = vec![
        AllenRelation::Before,
        AllenRelation::Meets,
        AllenRelation::Overlaps,
    ];
    constraints.push(OrdHornConstraint {
        a: 0,
        b: 5,
        relations: weakened,
    });

    // Add a few random extra "Before" constraints between random
    // (i, j) pairs with i < j to ensure the closure is iterated, not
    // just one-shot evaluated.
    for _ in 0..3 {
        let i = rng.gen_range(0..(n - 2));
        let j = rng.gen_range((i + 1)..n);
        constraints.push(single(i, j, AllenRelation::Before));
    }

    let net = OrdHornNetwork { n, constraints };
    assert!(
        is_satisfiable(&net),
        "Before-chain network must be satisfiable"
    );

    let closed = closure(&net).expect("closure ok");
    for c in &closed.constraints {
        assert!(
            !c.relations.is_empty(),
            "closure of satisfiable Before-chain should not produce empty cells, got {:?}",
            c
        );
    }

    // (0, 5) was supplied as {Before, Meets, Overlaps}. After
    // composing the explicit Before chain 0 → 1 → … → 5, the
    // intersection should collapse to {Before}.
    let r05 = find_pair(&closed, 0, 5).expect("(0, 5) constraint missing after closure");
    assert_eq!(
        r05,
        vec![AllenRelation::Before],
        "expected (0, 5) tightened to {{Before}} via chain propagation, got {:?}",
        r05
    );
}

#[test]
fn t6_inverse_set_round_trip_preserves_disjunctions() {
    // Defensive helper test: inverse_set ∘ inverse_set = identity for
    // any subset.
    let sample = vec![
        AllenRelation::Before,
        AllenRelation::Meets,
        AllenRelation::Equals,
        AllenRelation::Contains,
        AllenRelation::After,
    ];
    let twice = inverse_set(&inverse_set(&sample));
    let mut expected = sample.clone();
    expected.sort_by_key(|r| match r {
        AllenRelation::Before => 0,
        AllenRelation::Meets => 1,
        AllenRelation::Overlaps => 2,
        AllenRelation::Starts => 3,
        AllenRelation::During => 4,
        AllenRelation::Finishes => 5,
        AllenRelation::Equals => 6,
        AllenRelation::FinishedBy => 7,
        AllenRelation::Contains => 8,
        AllenRelation::StartedBy => 9,
        AllenRelation::OverlappedBy => 10,
        AllenRelation::MetBy => 11,
        AllenRelation::After => 12,
    });
    assert_eq!(twice, expected);

    // Also confirm canonical pairs.
    assert_eq!(inverse(AllenRelation::Before), AllenRelation::After);
    assert_eq!(inverse(AllenRelation::Meets), AllenRelation::MetBy);
    assert_eq!(inverse(AllenRelation::Equals), AllenRelation::Equals);
}

#[test]
fn constraint_round_trips_through_json() {
    let c = OrdHornConstraint {
        a: 0,
        b: 1,
        relations: vec![AllenRelation::Before, AllenRelation::Meets],
    };
    let bytes = serde_json::to_vec(&c).expect("serialise");
    let back: OrdHornConstraint = serde_json::from_slice(&bytes).expect("deserialise");
    assert_eq!(c, back);
}

#[test]
fn empty_network_round_trips_unchanged() {
    let net = OrdHornNetwork {
        n: 0,
        constraints: vec![],
    };
    let closed = closure(&net).expect("closure on empty network");
    assert_eq!(closed.n, 0);
    assert!(closed.constraints.is_empty());
    assert!(is_satisfiable(&net));
}

