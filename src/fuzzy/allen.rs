//! Fuzzy Allen interval algebra — graded versions of the 13 Allen
//! relations over trapezoidal fuzzy interval endpoints.
//!
//! # Design
//!
//! Each situation's temporal interval carries an optional [`FuzzyEndpoints`]
//! struct on top of the crisp `AllenInterval { start, end, … }` tuple. When
//! the field is `None` (the common case), [`graded_relation`] returns a
//! **one-hot** 13-vector for the crisp Allen relation — bit-identical to the
//! pre-Phase-5 semantics. This is the fast path that Phase 1-4 backward-
//! compatibility tests depend on.
//!
//! When both sides carry trapezoidal fuzzy endpoints, the graded relation
//! follows Dubois-Prade's construction:
//!
//! 1. Each of the 13 Allen relations is a conjunction of point-order
//!    constraints on `(a.start, a.end, b.start, b.end)` (e.g. BEFORE ≡
//!    `a.end < b.start`; OVERLAPS ≡ `a.start < b.start ∧ a.end > b.start
//!    ∧ a.end < b.end`).
//! 2. Each point-order constraint `x̃ ≤ ỹ` on trapezoidal fuzzy numbers is
//!    scored with a graded truth value. We use the mean of the possibility
//!    and necessity measures (Schockaert & De Cock 2008 convention):
//!
//!    ```text
//!      possibility(x̃ ≤ ỹ) = sup_{a ≤ b} min(μ_x̃(a), μ_ỹ(b))
//!      necessity(x̃ ≤ ỹ)   = 1 - sup_{a > b} min(μ_x̃(a), μ_ỹ(b))
//!      point_order_degree = (possibility + necessity) / 2
//!    ```
//!
//! 3. The per-relation degree is the t-norm combination of its point-order
//!    constraints. Default = Gödel (`min`); configurable via
//!    [`GradedAllenConfig`].
//!
//! # Persistence
//!
//! Graded 13-vectors are cached at `fz/allen/{narrative_id}/{a_id}/{b_id}`
//! (see [`crate::fuzzy::FUZZY_ALLEN_PREFIX`]). Recompute-on-demand if
//! missing. Phase 5 exposes [`invalidate_pair`] but does NOT wire auto-
//! invalidation into `Hypergraph::update_situation` — logged as a Phase 5
//! deferral.
//!
//! Cites: [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen].

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::TNormKind;
use crate::temporal::interval::relation_between;
use crate::types::{AllenInterval, AllenRelation};

// ── Trapezoidal fuzzy number ─────────────────────────────────────────────────

/// Trapezoidal fuzzy number over `DateTime<Utc>`. Represents a fuzzy
/// endpoint: the membership function ramps up from `support_min` to
/// `kernel_min`, stays at `1` on `[kernel_min, kernel_max]`, and ramps
/// down to `0` at `support_max`.
///
/// Invariant (enforced by [`TrapezoidalFuzzy::new`]):
/// `support_min <= kernel_min <= kernel_max <= support_max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrapezoidalFuzzy {
    pub support_min: DateTime<Utc>,
    pub kernel_min: DateTime<Utc>,
    pub kernel_max: DateTime<Utc>,
    pub support_max: DateTime<Utc>,
}

impl TrapezoidalFuzzy {
    /// Construct a trapezoidal fuzzy number, validating the monotone-
    /// endpoint invariant. Out-of-order bounds collapse to a crisp
    /// interval via [`TensaError::InvalidInput`].
    pub fn new(
        support_min: DateTime<Utc>,
        kernel_min: DateTime<Utc>,
        kernel_max: DateTime<Utc>,
        support_max: DateTime<Utc>,
    ) -> Result<Self> {
        if !(support_min <= kernel_min && kernel_min <= kernel_max && kernel_max <= support_max) {
            return Err(TensaError::InvalidInput(format!(
                "TrapezoidalFuzzy requires support_min ({}) <= kernel_min ({}) <= kernel_max ({}) <= support_max ({})",
                support_min, kernel_min, kernel_max, support_max
            )));
        }
        Ok(Self {
            support_min,
            kernel_min,
            kernel_max,
            support_max,
        })
    }

    /// Construct a crisp (degenerate) trapezoid from a single timestamp:
    /// `support_min == kernel_min == kernel_max == support_max == t`.
    /// Useful for comparing a fuzzy interval against a crisp reference.
    pub fn crisp(t: DateTime<Utc>) -> Self {
        Self {
            support_min: t,
            kernel_min: t,
            kernel_max: t,
            support_max: t,
        }
    }

    /// Membership value `μ(t) ∈ [0, 1]` at an arbitrary timestamp.
    pub fn membership(&self, t: DateTime<Utc>) -> f64 {
        if t < self.support_min || t > self.support_max {
            return 0.0;
        }
        if t >= self.kernel_min && t <= self.kernel_max {
            return 1.0;
        }
        if t < self.kernel_min {
            // left shoulder ramp
            let total = (self.kernel_min - self.support_min).num_nanoseconds();
            let partial = (t - self.support_min).num_nanoseconds();
            match (total, partial) {
                (Some(total), Some(partial)) if total > 0 => {
                    (partial as f64 / total as f64).clamp(0.0, 1.0)
                }
                _ => 1.0,
            }
        } else {
            // right shoulder ramp
            let total = (self.support_max - self.kernel_max).num_nanoseconds();
            let partial = (self.support_max - t).num_nanoseconds();
            match (total, partial) {
                (Some(total), Some(partial)) if total > 0 => {
                    (partial as f64 / total as f64).clamp(0.0, 1.0)
                }
                _ => 1.0,
            }
        }
    }
}

// ── Fuzzy endpoints (the two trapezoids of a fuzzy interval) ─────────────────

/// Trapezoidal fuzzy endpoints for a situation's temporal interval. The
/// interval's logical start is a trapezoid, and so is its logical end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FuzzyEndpoints {
    /// Fuzzy interval over the start-time.
    pub start: Option<TrapezoidalFuzzy>,
    /// Fuzzy interval over the end-time.
    pub end: Option<TrapezoidalFuzzy>,
}

impl FuzzyEndpoints {
    /// Build from a pair of trapezoids.
    pub fn from_pair(start: TrapezoidalFuzzy, end: TrapezoidalFuzzy) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
        }
    }

    /// True iff both start and end trapezoids are set.
    pub fn is_complete(&self) -> bool {
        self.start.is_some() && self.end.is_some()
    }
}

// ── Graded Allen configuration ───────────────────────────────────────────────

/// Per-call configuration for the graded Allen computation. Thread the
/// t-norm through the public API so the fuzzy-surface opt-in query
/// parameter (Phase 4) can override the default Gödel semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradedAllenConfig {
    pub tnorm: TNormKind,
}

impl Default for GradedAllenConfig {
    fn default() -> Self {
        Self {
            tnorm: TNormKind::Godel,
        }
    }
}

// ── Relation indexing ────────────────────────────────────────────────────────

/// Stable index of a relation into the 13-vector returned by
/// [`graded_relation`]. Order matches Allen's canonical ordering.
pub fn relation_index(r: AllenRelation) -> usize {
    match r {
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
    }
}

/// Inverse of [`relation_index`].
pub fn index_to_relation(idx: usize) -> Option<AllenRelation> {
    match idx {
        0 => Some(AllenRelation::Before),
        1 => Some(AllenRelation::Meets),
        2 => Some(AllenRelation::Overlaps),
        3 => Some(AllenRelation::Starts),
        4 => Some(AllenRelation::During),
        5 => Some(AllenRelation::Finishes),
        6 => Some(AllenRelation::Equals),
        7 => Some(AllenRelation::FinishedBy),
        8 => Some(AllenRelation::Contains),
        9 => Some(AllenRelation::StartedBy),
        10 => Some(AllenRelation::OverlappedBy),
        11 => Some(AllenRelation::MetBy),
        12 => Some(AllenRelation::After),
        _ => None,
    }
}

// ── Core: graded relation ────────────────────────────────────────────────────

/// Compute the 13-vector of graded Allen relation degrees between two
/// intervals. Indexed by [`relation_index`].
///
/// * If **both** intervals have `fuzzy_endpoints = None`, returns a one-hot
///   vector matching the crisp [`relation_between`]. This is the fast path
///   and must stay bit-identical to the pre-Phase-5 semantics.
/// * Otherwise, builds trapezoidal fuzzy numbers for each endpoint (crisp
///   timestamps collapse to a degenerate trapezoid via
///   [`TrapezoidalFuzzy::crisp`]) and computes each relation as the
///   t-norm conjunction of its point-order constraints.
///
/// Uses the default [`GradedAllenConfig`] (Gödel t-norm). Call
/// [`graded_relation_with`] for a custom configuration.
pub fn graded_relation(a: &AllenInterval, b: &AllenInterval) -> [f64; 13] {
    graded_relation_with(a, b, &GradedAllenConfig::default())
}

/// Same as [`graded_relation`] but with an explicit configuration.
pub fn graded_relation_with(
    a: &AllenInterval,
    b: &AllenInterval,
    cfg: &GradedAllenConfig,
) -> [f64; 13] {
    // Fast path — both sides crisp. One-hot at the existing AllenRelation.
    if a.fuzzy_endpoints.is_none() && b.fuzzy_endpoints.is_none() {
        let mut out = [0.0f64; 13];
        if let Ok(rel) = relation_between(a, b) {
            out[relation_index(rel)] = 1.0;
        }
        return out;
    }

    // Build trapezoids for every endpoint, defaulting crisp endpoints to
    // degenerate trapezoids.
    let (a_start_tf, a_end_tf) = endpoints_as_trapezoids(a);
    let (b_start_tf, b_end_tf) = endpoints_as_trapezoids(b);

    // Undefined on either side → all zeros.
    let (a_start_tf, a_end_tf, b_start_tf, b_end_tf) =
        match (a_start_tf, a_end_tf, b_start_tf, b_end_tf) {
            (Some(a1), Some(a2), Some(b1), Some(b2)) => (a1, a2, b1, b2),
            _ => return [0.0f64; 13],
        };

    // Point-order degrees between the four endpoints that span the 13
    // relations. `leq(x, y)` = point-order degree of `x̃ ≤ ỹ`.
    let leq_aend_bstart = point_order_leq(&a_end_tf, &b_start_tf);
    let leq_bstart_aend = point_order_leq(&b_start_tf, &a_end_tf);
    let leq_astart_bstart = point_order_leq(&a_start_tf, &b_start_tf);
    let leq_bstart_astart = point_order_leq(&b_start_tf, &a_start_tf);
    let leq_aend_bend = point_order_leq(&a_end_tf, &b_end_tf);
    let leq_bend_aend = point_order_leq(&b_end_tf, &a_end_tf);
    let leq_astart_bend = point_order_leq(&a_start_tf, &b_end_tf);
    let leq_bend_astart = point_order_leq(&b_end_tf, &a_start_tf);

    // Strict (<) degrees derived from the `≤` duality:
    //   μ_<(x̃, ỹ) = 1 - μ_≤(ỹ, x̃)
    let lt_aend_bstart = 1.0 - leq_bstart_aend;
    let lt_bstart_aend = 1.0 - leq_aend_bstart;
    let lt_astart_bstart = 1.0 - leq_bstart_astart;
    let lt_bstart_astart = 1.0 - leq_astart_bstart;
    let lt_aend_bend = 1.0 - leq_bend_aend;
    let lt_bend_aend = 1.0 - leq_aend_bend;
    let lt_astart_bend = 1.0 - leq_bend_astart;

    // Equality ≈ both directions of ≤. We take `eq = min(leq, geq)` for
    // a Gödel-style conjunction of the two weak inequalities.
    let eq_astart_bstart = leq_astart_bstart.min(leq_bstart_astart);
    let eq_aend_bend = leq_aend_bend.min(leq_bend_aend);
    let eq_aend_bstart = leq_aend_bstart.min(leq_bstart_aend);
    let eq_astart_bend = leq_astart_bend.min(leq_bend_astart);

    // T-norm conjunction of point-order constraints. Gödel uses `min`;
    // Goguen uses product; Łukasiewicz uses bounded diff. We synthesise
    // through the existing TNormKind enum so a future planner opt-in can
    // thread it all the way through.
    let and = |xs: &[f64]| combine_tnorm(cfg.tnorm, xs);

    let mut out = [0.0f64; 13];
    //   BEFORE  : a.end < b.start
    out[relation_index(AllenRelation::Before)] = lt_aend_bstart;
    //   MEETS   : a.end = b.start
    out[relation_index(AllenRelation::Meets)] = eq_aend_bstart;
    //   OVERLAPS: a.start < b.start ∧ a.end > b.start ∧ a.end < b.end
    out[relation_index(AllenRelation::Overlaps)] =
        and(&[lt_astart_bstart, lt_bstart_aend, lt_aend_bend]);
    //   STARTS  : a.start = b.start ∧ a.end < b.end
    out[relation_index(AllenRelation::Starts)] = and(&[eq_astart_bstart, lt_aend_bend]);
    //   DURING  : a.start > b.start ∧ a.end < b.end
    out[relation_index(AllenRelation::During)] = and(&[lt_bstart_astart, lt_aend_bend]);
    //   FINISHES: a.start > b.start ∧ a.end = b.end
    out[relation_index(AllenRelation::Finishes)] = and(&[lt_bstart_astart, eq_aend_bend]);
    //   EQUALS  : a.start = b.start ∧ a.end = b.end
    out[relation_index(AllenRelation::Equals)] = and(&[eq_astart_bstart, eq_aend_bend]);
    //   FINISHED_BY: a.start < b.start ∧ a.end = b.end
    out[relation_index(AllenRelation::FinishedBy)] = and(&[lt_astart_bstart, eq_aend_bend]);
    //   CONTAINS: a.start < b.start ∧ a.end > b.end
    out[relation_index(AllenRelation::Contains)] = and(&[lt_astart_bstart, lt_bend_aend]);
    //   STARTED_BY: a.start = b.start ∧ a.end > b.end
    out[relation_index(AllenRelation::StartedBy)] = and(&[eq_astart_bstart, lt_bend_aend]);
    //   OVERLAPPED_BY: a.start > b.start ∧ a.start < b.end ∧ a.end > b.end
    out[relation_index(AllenRelation::OverlappedBy)] =
        and(&[lt_bstart_astart, lt_astart_bend, lt_bend_aend]);
    //   MET_BY  : a.start = b.end
    out[relation_index(AllenRelation::MetBy)] = eq_astart_bend;
    //   AFTER   : a.start > b.end
    out[relation_index(AllenRelation::After)] = 1.0 - leq_astart_bend;

    // Clamp for numerical safety.
    for v in out.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    out
}

/// Read a single relation's graded degree.
pub fn graded_relation_value(a: &AllenInterval, b: &AllenInterval, rel: AllenRelation) -> f64 {
    graded_relation(a, b)[relation_index(rel)]
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build trapezoidal fuzzy numbers for an interval's start and end. If
/// `fuzzy_endpoints` is set we use those; otherwise we collapse the crisp
/// `start` / `end` timestamps to a degenerate (crisp) trapezoid.
fn endpoints_as_trapezoids(
    i: &AllenInterval,
) -> (Option<TrapezoidalFuzzy>, Option<TrapezoidalFuzzy>) {
    let crisp_start = i.start.map(TrapezoidalFuzzy::crisp);
    let crisp_end = i.end.map(TrapezoidalFuzzy::crisp);
    match &i.fuzzy_endpoints {
        Some(fe) => (fe.start.or(crisp_start), fe.end.or(crisp_end)),
        None => (crisp_start, crisp_end),
    }
}

/// Possibility that `x̃ ≤ ỹ` for two trapezoidal fuzzy numbers.
///
/// `Poss(x̃ ≤ ỹ) = sup_{a ≤ b} min(μ_x̃(a), μ_ỹ(b))`.
///
/// For trapezoids this simplifies to a closed form: the possibility is
/// `1` when `x.support_min ≤ y.support_max`, and ramps down linearly
/// otherwise. Full derivation: see Dubois & Prade 1988 §5.
fn possibility_leq(x: &TrapezoidalFuzzy, y: &TrapezoidalFuzzy) -> f64 {
    // Fully possible: the kernel of x reaches below (or up to) the kernel of y.
    if x.kernel_min <= y.kernel_max {
        return 1.0;
    }
    // Ramp region: x.kernel_min is past y.kernel_max, but y's right shoulder
    // may still cross x's left shoulder. Find the intersection of the
    // ramps.
    // x's left shoulder (μ_x rises from 0 at x.support_min to 1 at x.kernel_min).
    // y's right shoulder (μ_y falls from 1 at y.kernel_max to 0 at y.support_max).
    // At the intersection, μ_x(t) = μ_y(t) and t ≤ x.kernel_min, t ≥ y.kernel_max.
    //
    // If the shoulders don't overlap at all, possibility = 0.
    if x.support_min > y.support_max {
        return 0.0;
    }
    // Linear solve: let t ∈ [y.kernel_max, x.kernel_min].
    //   μ_x(t) = (t - x.support_min) / (x.kernel_min - x.support_min)
    //   μ_y(t) = (y.support_max - t) / (y.support_max - y.kernel_max)
    // Solve for the equal point and return that value. Defensive on zero
    // denominators (degenerate crisp cases).
    let x_left = (x.kernel_min - x.support_min).num_nanoseconds().unwrap_or(0);
    let y_right = (y.support_max - y.kernel_max).num_nanoseconds().unwrap_or(0);
    if x_left == 0 && y_right == 0 {
        // Both crisp at their respective boundaries. Already handled by
        // the `x.kernel_min <= y.kernel_max` check; falling through means
        // `x.kernel_min > y.kernel_max`, so possibility = 0.
        return 0.0;
    }
    // Standard trapezoidal-intersection formula. At the crossing,
    //   μ = (y.support_max - x.support_min) / (x_left + y_right)
    // Defensive zero-denominator path returns 0 (degenerate geometry).
    let d = (y.support_max - x.support_min).num_nanoseconds().unwrap_or(0);
    let l = x_left + y_right;
    if l == 0 {
        return 0.0;
    }
    (d as f64 / l as f64).clamp(0.0, 1.0)
}

/// Necessity that `x̃ ≤ ỹ`: `1 - Poss(y ≤ x)`.
///
/// For trapezoids:
/// * `x.support_max ≤ y.support_min` → necessity = 1 (x entirely below y).
/// * `x.kernel_min  >  y.kernel_max` → necessity = 0 (x kernel past y).
/// * otherwise, `1 - Poss(y ≤ x)` via the dual formula. The strict-vs-weak
///   `<` / `≤` distinction only matters on measure-zero boundary points —
///   Dubois & Prade 1988 §5 note.
fn necessity_leq(x: &TrapezoidalFuzzy, y: &TrapezoidalFuzzy) -> f64 {
    if x.support_max <= y.support_min {
        return 1.0;
    }
    if x.kernel_min > y.kernel_max {
        return 0.0;
    }
    1.0 - possibility_leq(y, x)
}

/// Schockaert-De Cock point-order degree: mean of possibility and necessity.
fn point_order_leq(x: &TrapezoidalFuzzy, y: &TrapezoidalFuzzy) -> f64 {
    let p = possibility_leq(x, y);
    let n = necessity_leq(x, y);
    ((p + n) / 2.0).clamp(0.0, 1.0)
}

/// Fold a slice of point-order degrees through a t-norm.
fn combine_tnorm(kind: TNormKind, xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 1.0;
    }
    let mut acc = xs[0].clamp(0.0, 1.0);
    for x in &xs[1..] {
        let y = x.clamp(0.0, 1.0);
        acc = match kind {
            TNormKind::Godel => acc.min(y),
            TNormKind::Goguen => acc * y,
            TNormKind::Lukasiewicz => (acc + y - 1.0).max(0.0),
            TNormKind::Hamacher(lambda) => {
                let lam = lambda.max(0.0);
                let denom = lam + (1.0 - lam) * (acc + y - acc * y);
                if denom <= 0.0 {
                    0.0
                } else {
                    (acc * y / denom).clamp(0.0, 1.0)
                }
            }
        };
    }
    acc.clamp(0.0, 1.0)
}

// ── Convenience: fuzzy relation filter ──────────────────────────────────────

/// Check whether the graded degree of `rel` between `a` and `b` is at
/// least `threshold`. Used by the TensaQL `AS FUZZY <rel> THRESHOLD <t>`
/// tail and by downstream consumers that want a boolean filter.
pub fn fuzzy_relation_holds(
    a: &AllenInterval,
    b: &AllenInterval,
    rel: AllenRelation,
    threshold: f64,
) -> bool {
    graded_relation_value(a, b, rel) >= threshold
}

// Re-exports from the split `allen_store.rs` sibling module. Keeps the
// public surface stable across the split.
pub use crate::fuzzy::allen_store::{
    crisp_reference_interval, delete_fuzzy_allen, fuzzy_from_marker, fuzzy_relation_degree,
    fuzzy_relation_query_situations, invalidate_pair, load_fuzzy_allen, save_fuzzy_allen,
};
