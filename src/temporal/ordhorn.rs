//! ORD-Horn tractable subfragment of Allen interval algebra — Phase 4.
//!
//! ORD-Horn is the maximal tractable subclass of Allen's interval
//! algebra (Nebel & Bürckert 1995). Path-consistency on an ORD-Horn
//! constraint network is sound + complete for satisfiability and runs in
//! `O(n^3)`. This module ships the canonical van Beek path-consistency
//! closure built on top of TENSA's existing 13×13 composition table
//! ([`crate::temporal::interval::compose`]) — the algorithm is a 30-year-
//! old port, not a reinvention.
//!
//! ## Soundness vs completeness
//!
//! Path-consistency is:
//!   * **sound** for any Allen constraint network — if the closure
//!     produces an empty constraint, the network is provably
//!     unsatisfiable;
//!   * **complete** *only* for ORD-Horn networks — if every constraint's
//!     disjunction lies inside the 868-element ORD-Horn class, then a
//!     non-empty closure proves satisfiability (Nebel & Bürckert 1995,
//!     Theorem 1).
//!
//! For general Allen networks the closure may report "satisfiable" when
//! the network is actually unsatisfiable — additional backtracking
//! search is required. This module ships the closure proper; it does NOT
//! ship the 868-element ORD-Horn membership oracle. Callers that need
//! decidability guarantees must restrict their inputs to ORD-Horn by
//! construction (e.g. only Pointisable Allen relations, or only the
//! "convex" subset).
//!
//! Cites: [nebel1995ordhorn].

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::temporal::interval::compose;
use crate::types::AllenRelation;

/// All 13 basic Allen relations, in the canonical Allen 1983 order.
///
/// Used as the "no constraint" sentinel inside [`build_matrix`] — every
/// pair starts unconstrained and is tightened only when an explicit
/// [`OrdHornConstraint`] supplies a smaller disjunction.
pub const ALL_RELATIONS: [AllenRelation; 13] = [
    AllenRelation::Before,
    AllenRelation::Meets,
    AllenRelation::Overlaps,
    AllenRelation::Starts,
    AllenRelation::During,
    AllenRelation::Finishes,
    AllenRelation::Equals,
    AllenRelation::FinishedBy,
    AllenRelation::Contains,
    AllenRelation::StartedBy,
    AllenRelation::OverlappedBy,
    AllenRelation::MetBy,
    AllenRelation::After,
];

/// One binary constraint: interval `a` stands in some basic relation
/// from the disjunction `relations` to interval `b`.
///
/// `relations` is sorted in canonical Allen order (matching
/// [`ALL_RELATIONS`]) with no duplicates. An empty `relations` slot
/// signals an unsatisfiable constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrdHornConstraint {
    pub a: usize,
    pub b: usize,
    pub relations: Vec<AllenRelation>,
}

/// A constraint network over `n` intervals.
///
/// Pairs that do not appear in `constraints` are treated as
/// unconstrained (the full 13-relation disjunction). After
/// [`closure`] runs, the returned network omits any pair whose
/// constraint is the full set — only tightened constraints are listed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrdHornNetwork {
    pub n: usize,
    pub constraints: Vec<OrdHornConstraint>,
}

// ---------- canonical relation ordering ----------

fn relation_rank(r: AllenRelation) -> usize {
    use AllenRelation::*;
    match r {
        Before => 0,
        Meets => 1,
        Overlaps => 2,
        Starts => 3,
        During => 4,
        Finishes => 5,
        Equals => 6,
        FinishedBy => 7,
        Contains => 8,
        StartedBy => 9,
        OverlappedBy => 10,
        MetBy => 11,
        After => 12,
    }
}

fn sort_dedup(mut rs: Vec<AllenRelation>) -> Vec<AllenRelation> {
    rs.sort_by_key(|r| relation_rank(*r));
    rs.dedup();
    rs
}

// ---------- helpers ----------

/// Compose two disjunctive relation sets: the union of all pairwise
/// compositions, sorted + deduplicated. Empty input → empty output.
pub fn compose_sets(r1: &[AllenRelation], r2: &[AllenRelation]) -> Vec<AllenRelation> {
    let mut out: Vec<AllenRelation> = Vec::new();
    for a in r1 {
        for b in r2 {
            for c in compose(*a, *b) {
                out.push(c);
            }
        }
    }
    sort_dedup(out)
}

/// Intersect two sorted disjunctive relation sets. Output is sorted +
/// deduplicated.
pub fn intersect_sets(a: &[AllenRelation], b: &[AllenRelation]) -> Vec<AllenRelation> {
    // The inputs are typically small (≤ 13), so the explicit `O(n*m)`
    // pass is faster than building a hash set.
    let mut out: Vec<AllenRelation> = Vec::new();
    for r in a {
        if b.contains(r) {
            out.push(*r);
        }
    }
    sort_dedup(out)
}

/// Inverse of an Allen relation: the relation that holds when interval
/// roles swap. Re-exported wrapper around [`AllenRelation::inverse`] so
/// the public surface of this module is self-contained.
pub fn inverse(r: AllenRelation) -> AllenRelation {
    r.inverse()
}

/// Inverse of a disjunctive relation set.
pub fn inverse_set(rs: &[AllenRelation]) -> Vec<AllenRelation> {
    sort_dedup(rs.iter().map(|r| inverse(*r)).collect())
}

// ---------- matrix form ----------

/// Build the dense `n×n` matrix where `m[i][j]` is the disjunctive
/// constraint from interval `i` to interval `j`. Diagonals are
/// `[Equals]`. Pairs not listed in `network.constraints` start as
/// [`ALL_RELATIONS`].
fn build_matrix(net: &OrdHornNetwork) -> Vec<Vec<Vec<AllenRelation>>> {
    let n = net.n;
    let all = ALL_RELATIONS.to_vec();
    let mut mat: Vec<Vec<Vec<AllenRelation>>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { vec![AllenRelation::Equals] } else { all.clone() })
                .collect()
        })
        .collect();

    for c in &net.constraints {
        if c.a >= n || c.b >= n || c.a == c.b {
            continue; // skip out-of-range and self-loops; no error — caller can't break network shape
        }
        let rs = sort_dedup(c.relations.clone());
        // Intersect the existing slot with the supplied disjunction so
        // multiple constraints on the same pair AND together rather
        // than the last write winning.
        mat[c.a][c.b] = intersect_sets(&mat[c.a][c.b], &rs);
        mat[c.b][c.a] = intersect_sets(&mat[c.b][c.a], &inverse_set(&rs));
    }

    mat
}

/// Convert the dense matrix back into a `Vec<OrdHornConstraint>`,
/// emitting only pairs `(i, j)` with `i < j` whose constraint differs
/// from [`ALL_RELATIONS`]. The reverse direction `(j, i)` is recoverable
/// via [`inverse_set`] so omitting it keeps the wire format compact.
fn matrix_to_constraints(mat: &[Vec<Vec<AllenRelation>>]) -> Vec<OrdHornConstraint> {
    let n = mat.len();
    let all = ALL_RELATIONS.to_vec();
    let mut out: Vec<OrdHornConstraint> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let cell = &mat[i][j];
            if cell.as_slice() != all.as_slice() {
                out.push(OrdHornConstraint {
                    a: i,
                    b: j,
                    relations: cell.clone(),
                });
            }
        }
    }
    out
}

// ---------- closure ----------

/// Run path-consistency closure on the network. Returns the tightened
/// network in the same shape (`Vec<OrdHornConstraint>`).
///
/// Soundness is guaranteed for any Allen network. Completeness (i.e.
/// `is_satisfiable` decides satisfiability) is guaranteed only when the
/// input is in the ORD-Horn fragment — see the module docs.
///
/// Cites: [nebel1995ordhorn] — Theorem 1 (path-consistency is complete
/// for ORD-Horn).
pub fn closure(network: &OrdHornNetwork) -> Result<OrdHornNetwork> {
    let n = network.n;
    if n == 0 {
        return Ok(network.clone());
    }

    let mut mat = build_matrix(network);

    // Safety cap. A single triple-update can shrink one cell by at most
    // 13 relations, so the total monotone-shrink budget is bounded by
    // `n^2 * 13`. Multiplying by another `n` to absorb the iteration of
    // the outer changed-loop yields the cap below — 4-5 orders of
    // magnitude above any realistic running time. Hitting it almost
    // certainly indicates a bug, not a slow input.
    let cap = n.saturating_mul(n).saturating_mul(n).saturating_mul(13);
    let mut iters: usize = 0;

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                for k in 0..n {
                    if k == i || k == j {
                        continue;
                    }
                    iters = iters.saturating_add(1);
                    if iters > cap {
                        return Err(TensaError::InvalidInput(
                            "ORD-Horn closure exceeded iteration cap; likely a bug".into(),
                        ));
                    }
                    let composed = compose_sets(&mat[i][j], &mat[j][k]);
                    let new_ik = intersect_sets(&mat[i][k], &composed);
                    if new_ik != mat[i][k] {
                        let became_empty = new_ik.is_empty();
                        let inv = inverse_set(&new_ik);
                        mat[i][k] = new_ik;
                        mat[k][i] = inv;
                        changed = true;
                        if became_empty {
                            tracing::debug!(
                                "ORD-Horn closure reached empty constraint at ({}, {})",
                                i,
                                k
                            );
                            // Preserve the empty constraint in the
                            // output so callers can locate the
                            // contradiction.
                            let mut out = matrix_to_constraints(&mat);
                            out.push(OrdHornConstraint {
                                a: i.min(k),
                                b: i.max(k),
                                relations: Vec::new(),
                            });
                            // Dedup in case the (min, max) pair already
                            // surfaced from `matrix_to_constraints`.
                            out.sort_by_key(|c| (c.a, c.b));
                            out.dedup_by(|x, y| x.a == y.a && x.b == y.b);
                            return Ok(OrdHornNetwork { n, constraints: out });
                        }
                    }
                }
            }
        }
    }

    Ok(OrdHornNetwork {
        n,
        constraints: matrix_to_constraints(&mat),
    })
}

/// Returns `true` iff path-consistency closure reaches a fixed point
/// with no empty constraint. For ORD-Horn networks this decides
/// satisfiability; for general networks it is sound but incomplete.
pub fn is_satisfiable(network: &OrdHornNetwork) -> bool {
    match closure(network) {
        Ok(closed) => closed.constraints.iter().all(|c| !c.relations.is_empty()),
        Err(_) => false,
    }
}

#[cfg(test)]
#[path = "ordhorn_tests.rs"]
mod tests;
