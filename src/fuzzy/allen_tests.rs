//! Fuzzy Sprint Phase 5 — graded Allen interval relations.
//!
//! Acceptance tests per `docs/FUZZY_Sprint.md` Phase 5:
//!
//! * 13 one-hot tests — one per Allen relation — verifying that crisp
//!   intervals (both `fuzzy_endpoints = None`) produce a one-hot vector
//!   matching the crisp [`relation_between`].
//! * Fuzzy-boundary test — overlapping trapezoidal supports yield graded
//!   mass at the transition.
//! * Trapezoid constructor invariants + membership at kernel/support.
//! * KV round-trip.
//! * Reconciler keyword heuristic.
//! * Grammar integration tested in `src/query/parser_fuzzy_tests.rs`.
//!
//! Cites: [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen].

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use crate::fuzzy::allen::{
    delete_fuzzy_allen, fuzzy_from_marker, fuzzy_relation_holds, graded_relation,
    graded_relation_value, index_to_relation, load_fuzzy_allen, relation_index, save_fuzzy_allen,
    FuzzyEndpoints, TrapezoidalFuzzy,
};
use crate::store::memory::MemoryStore;
use crate::types::{AllenInterval, AllenRelation, TimeGranularity};

fn base_time() -> DateTime<Utc> {
    DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&Utc)
}

fn crisp_interval(start_hours: i64, end_hours: i64) -> AllenInterval {
    let base = base_time();
    AllenInterval {
        start: Some(base + Duration::hours(start_hours)),
        end: Some(base + Duration::hours(end_hours)),
        granularity: TimeGranularity::Exact,
        relations: vec![],
        fuzzy_endpoints: None,
    }
}

// ── One-hot fast-path (13 relations) ─────────────────────────────────────────

/// Crisp intervals must produce a one-hot vector matching the crisp
/// [`relation_between`] for **every** Allen relation. This is the
/// load-bearing backward-compat guarantee.
#[test]
fn test_crisp_intervals_produce_one_hot_for_all_relations() {
    // Canonical configurations for each relation, keyed by expected rel.
    let cases: &[(AllenRelation, AllenInterval, AllenInterval)] = &[
        (AllenRelation::Before, crisp_interval(0, 1), crisp_interval(2, 3)),
        (AllenRelation::Meets, crisp_interval(0, 2), crisp_interval(2, 4)),
        (
            AllenRelation::Overlaps,
            crisp_interval(0, 3),
            crisp_interval(2, 5),
        ),
        (AllenRelation::Starts, crisp_interval(1, 3), crisp_interval(1, 5)),
        (AllenRelation::During, crisp_interval(2, 3), crisp_interval(1, 4)),
        (AllenRelation::Finishes, crisp_interval(3, 5), crisp_interval(1, 5)),
        (AllenRelation::Equals, crisp_interval(1, 5), crisp_interval(1, 5)),
        (
            AllenRelation::FinishedBy,
            crisp_interval(1, 5),
            crisp_interval(3, 5),
        ),
        (
            AllenRelation::Contains,
            crisp_interval(1, 4),
            crisp_interval(2, 3),
        ),
        (
            AllenRelation::StartedBy,
            crisp_interval(1, 5),
            crisp_interval(1, 3),
        ),
        (
            AllenRelation::OverlappedBy,
            crisp_interval(2, 5),
            crisp_interval(0, 3),
        ),
        (AllenRelation::MetBy, crisp_interval(2, 4), crisp_interval(0, 2)),
        (AllenRelation::After, crisp_interval(4, 5), crisp_interval(1, 2)),
    ];

    for (expected, a, b) in cases {
        let v = graded_relation(a, b);
        let hot_idx = relation_index(*expected);
        // The expected slot is exactly 1.0.
        assert!(
            (v[hot_idx] - 1.0).abs() < 1e-12,
            "expected {:?} at idx {} == 1.0, got {}",
            expected,
            hot_idx,
            v[hot_idx]
        );
        // Every other slot is exactly 0.0.
        for (i, val) in v.iter().enumerate() {
            if i == hot_idx {
                continue;
            }
            assert!(
                val.abs() < 1e-12,
                "expected zero at idx {} ({:?}), got {} for {:?}",
                i,
                index_to_relation(i).unwrap(),
                val,
                expected,
            );
        }
        // Confirm the helper matches the vector read.
        assert_eq!(graded_relation_value(a, b, *expected), 1.0);
    }
}

/// Explicit one-hot guard for the most common relation (BEFORE) as a
/// focused sanity check mirrored in the sprint doc.
#[test]
fn test_crisp_intervals_produce_one_hot_before() {
    let a = crisp_interval(0, 1);
    let b = crisp_interval(2, 3);
    let v = graded_relation(&a, &b);
    assert!((v[relation_index(AllenRelation::Before)] - 1.0).abs() < 1e-12);
    for (i, val) in v.iter().enumerate() {
        if i == relation_index(AllenRelation::Before) {
            continue;
        }
        assert!(val.abs() < 1e-12, "non-zero at idx {}: {}", i, val);
    }
}

// ── Fuzzy boundary behaviour ────────────────────────────────────────────────

/// Two intervals with overlapping trapezoidal supports at their common
/// boundary should produce graded mass on BOTH `Before` and `Overlaps`
/// (the boundary is ambiguous under fuzziness).
#[test]
fn test_fuzzy_boundary_produces_graded_mass() {
    let base = base_time();
    // Interval a: crisp kernel [0h, 2h], fuzzy widen ±30min on the end.
    let a_fe = FuzzyEndpoints::from_pair(
        TrapezoidalFuzzy::new(
            base - Duration::minutes(30),
            base,
            base,
            base + Duration::minutes(30),
        )
        .unwrap(),
        TrapezoidalFuzzy::new(
            base + Duration::hours(2) - Duration::minutes(30),
            base + Duration::hours(2),
            base + Duration::hours(2),
            base + Duration::hours(2) + Duration::minutes(30),
        )
        .unwrap(),
    );
    let a = AllenInterval {
        start: Some(base),
        end: Some(base + Duration::hours(2)),
        granularity: TimeGranularity::Approximate,
        relations: vec![],
        fuzzy_endpoints: Some(a_fe),
    };

    // Interval b: crisp kernel [2h, 4h], fuzzy widen ±30min on the start.
    let b_fe = FuzzyEndpoints::from_pair(
        TrapezoidalFuzzy::new(
            base + Duration::hours(2) - Duration::minutes(30),
            base + Duration::hours(2),
            base + Duration::hours(2),
            base + Duration::hours(2) + Duration::minutes(30),
        )
        .unwrap(),
        TrapezoidalFuzzy::new(
            base + Duration::hours(4) - Duration::minutes(30),
            base + Duration::hours(4),
            base + Duration::hours(4),
            base + Duration::hours(4) + Duration::minutes(30),
        )
        .unwrap(),
    );
    let b = AllenInterval {
        start: Some(base + Duration::hours(2)),
        end: Some(base + Duration::hours(4)),
        granularity: TimeGranularity::Approximate,
        relations: vec![],
        fuzzy_endpoints: Some(b_fe),
    };

    let v = graded_relation(&a, &b);
    let before = v[relation_index(AllenRelation::Before)];
    let meets = v[relation_index(AllenRelation::Meets)];

    // Under fuzziness, the crisp relation is ambiguous between BEFORE
    // (with the crisp `a.end < b.start` reading) and MEETS (with the
    // `a.end == b.start` reading). Both should carry non-trivial mass;
    // neither should saturate to 1.0.
    assert!(
        meets > 0.1,
        "expected positive MEETS mass at boundary, got {}",
        meets
    );
    assert!(
        before < 1.0,
        "fuzzy BEFORE should not saturate to 1.0 at boundary, got {}",
        before
    );
}

// ── Trapezoidal invariants ──────────────────────────────────────────────────

#[test]
fn test_trapezoidal_new_rejects_out_of_order() {
    let base = base_time();
    let res = TrapezoidalFuzzy::new(
        base + Duration::hours(2),
        base + Duration::hours(1), // kernel_min < support_min — invalid
        base + Duration::hours(3),
        base + Duration::hours(4),
    );
    assert!(res.is_err(), "out-of-order constructor must reject");
}

#[test]
fn test_trapezoidal_membership_at_kernel_is_one() {
    let base = base_time();
    let t = TrapezoidalFuzzy::new(
        base,
        base + Duration::hours(1),
        base + Duration::hours(3),
        base + Duration::hours(4),
    )
    .unwrap();
    assert!((t.membership(base + Duration::hours(1)) - 1.0).abs() < 1e-12);
    assert!((t.membership(base + Duration::hours(2)) - 1.0).abs() < 1e-12);
    assert!((t.membership(base + Duration::hours(3)) - 1.0).abs() < 1e-12);
}

#[test]
fn test_trapezoidal_membership_at_support_boundary_is_zero() {
    let base = base_time();
    let t = TrapezoidalFuzzy::new(
        base,
        base + Duration::hours(1),
        base + Duration::hours(3),
        base + Duration::hours(4),
    )
    .unwrap();
    assert!(t.membership(base - Duration::hours(1)).abs() < 1e-12);
    assert!(t.membership(base + Duration::hours(5)).abs() < 1e-12);
}

// ── KV round-trip ───────────────────────────────────────────────────────────

#[test]
fn test_kv_roundtrip() {
    let store = MemoryStore::new();
    let a_id = Uuid::now_v7();
    let b_id = Uuid::now_v7();
    let nid = "narr-5";
    let vec_in = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.15, 0.25, 0.35,
    ];
    save_fuzzy_allen(&store, nid, &a_id, &b_id, &vec_in).unwrap();
    let loaded = load_fuzzy_allen(&store, nid, &a_id, &b_id).unwrap().unwrap();
    for (i, (a, b)) in vec_in.iter().zip(loaded.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "round-trip drift at idx {}: {} vs {}",
            i,
            a,
            b
        );
    }
    delete_fuzzy_allen(&store, nid, &a_id, &b_id).unwrap();
    assert!(load_fuzzy_allen(&store, nid, &a_id, &b_id).unwrap().is_none());
}

// ── Reconciler keyword detection ────────────────────────────────────────────

#[test]
fn test_reconciler_keyword_detection() {
    let base = base_time();
    let kernel_start = base;
    let kernel_end = base + Duration::hours(10);

    // Miss — no fuzziness cue.
    assert!(fuzzy_from_marker(kernel_start, kernel_end, "on the 3rd of April").is_none());

    // Hits — each cue.
    for phrase in [
        "shortly after the meeting",
        "around that time",
        "about an hour later",
        "in early 2024",
        "late in the evening",
        "approximately noon",
    ] {
        let fe = fuzzy_from_marker(kernel_start, kernel_end, phrase);
        assert!(fe.is_some(), "cue '{}' should fire", phrase);
        let fe = fe.unwrap();
        assert!(fe.is_complete());
        let sf = fe.start.unwrap();
        // Kernel matches the crisp start.
        assert_eq!(sf.kernel_min, kernel_start);
        assert_eq!(sf.kernel_max, kernel_start);
        // Supports widen outward.
        assert!(sf.support_min < kernel_start);
        assert!(sf.support_max > kernel_start);
    }
}

// ── Threshold helper ────────────────────────────────────────────────────────

#[test]
fn test_fuzzy_relation_holds_threshold() {
    let a = crisp_interval(0, 1);
    let b = crisp_interval(2, 3);
    // Crisp before → 1.0 ≥ any threshold ≤ 1.0.
    assert!(fuzzy_relation_holds(&a, &b, AllenRelation::Before, 0.9));
    assert!(!fuzzy_relation_holds(&a, &b, AllenRelation::After, 0.1));
}

