//! Gradual / ranking-based argumentation semantics — Phase 1 implementation.
//!
//! This module hosts four canonical gradual semantics from the Amgoud–
//! Ben-Naim lineage, all sharing a single iteration loop with a
//! parametric *influence step* and a hard convergence cap.
//!
//! ## Variants
//!
//! - **h-Categoriser** — `Acc_{i+1}(a) = w(a) / (1 + Σ_{b ∈ Att(a)} Acc_i(b))`.
//!   Converges by contraction on `[0,1]^n`.
//! - **Weighted h-Categoriser** — `Acc_{i+1}(a) = w(a) / (1 + Σ_b v_{ba} ·
//!   Acc_i(b))` with `v_{ba} ∈ [0,1]`. Converges under `Σ_b v_{ba} ≤ 1`,
//!   asserted at construction.
//! - **Max-based** — replaces sum aggregation with max. Contraction
//!   preserved.
//! - **Card-based** — `Acc_{i+1}(a) = w(a) / ((1 + card^+(a)) · (1 + sum(a)))`
//!   where `card^+` counts attackers with strictly positive acceptability
//!   and `sum` ranges over ALL attackers. Reaches a fixed point in
//!   `O(|A|)` rounds (Amgoud & Ben-Naim 2013 §4.3, Proposition 3).
//!
//! ## Influence-step t-norm coupling — option B from feedback
//!
//! The standard h-Categoriser influence function is `infl(s) = s / (1 + s)`,
//! i.e. denominator `1 + s`. To make it t-norm-aware coherently we KEEP
//! the canonical aggregation step verbatim (sum / max / card) and expose
//! a parametric influence-step family. The default `TNormKind::Godel`
//! reproduces the legacy `s / (1 + s)` behaviour bit-identically — so the
//! §10.2 backward-compat regression corpus stays intact when no t-norm
//! is supplied.
//!
//! Mapping (see [`infl_denom`]):
//!
//! | `TNormKind` | denominator term |
//! |---|---|
//! | `Godel` (default) | `s` |
//! | `Lukasiewicz` | `s.min(1.0)` |
//! | `Goguen` | `1 - exp(-s)` (Poisson-sum limit of probabilistic OR) |
//! | `Hamacher(λ)` | `S_Hamacher(s.clamp(0,1), s.clamp(0,1))` |
//!
//! For card-based the cardinality factor `(1 + card)` stays raw — the
//! t-norm only modulates the sum-component denominator term. See the
//! design doc §1.4 for the rationale.
//!
//! ## Convergence guard
//!
//! Generalising the influence step to arbitrary t-conorms preserves
//! monotonicity but contraction can break for non-strict t-conorms
//! (Goguen/Hamacher in particular). [`MAX_GRADUAL_ITERATIONS`] caps the
//! loop at 200 rounds; [`GradualResult::converged`] reports whether the
//! `||Acc_{i+1} - Acc_i||_∞ < CONVERGENCE_EPSILON` test ever held. We
//! emit `tracing::warn!` on cap-hit so callers see the trade-off
//! explicitly.
//!
//! Cites: [amgoud2013ranking], [besnard2001hcategoriser],
//!        [amgoud2017weighted].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::analysis::argumentation::ArgumentationFramework;
use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::{tconorm_for, TNormKind};

/// Hard cap on the gradual-semantics iteration loop. Prevents non-
/// contracting (semantics, t-norm) pairs from spinning indefinitely.
pub const MAX_GRADUAL_ITERATIONS: u32 = 200;

/// Convergence threshold on the `L_∞` distance between consecutive
/// acceptability vectors.
pub const CONVERGENCE_EPSILON: f64 = 1e-9;

/// One of four canonical gradual semantics. Variant payload only present
/// for [`GradualSemanticsKind::WeightedHCategoriser`], which carries the
/// per-attack `v_{ba}` weight vector flattened in attack-list order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradualSemanticsKind {
    /// h-Categoriser (Besnard & Hunter 2001).
    HCategoriser,
    /// Weighted h-Categoriser (Amgoud & Ben-Naim 2017). `weights[k]`
    /// applies to attack `attacks[k]` of the framework.
    WeightedHCategoriser {
        /// Per-attack weight in `[0, 1]`. Construction asserts
        /// `Σ_{b ∈ Att(a)} weights ≤ 1` for every target `a`.
        weights: Vec<f64>,
    },
    /// Max-based ranking semantics (Amgoud & Ben-Naim 2013).
    MaxBased,
    /// Card-based ranking semantics with lexicographic tie-break
    /// (Amgoud & Ben-Naim 2013).
    CardBased,
}

/// Result of a gradual-semantics evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradualResult {
    /// Acceptability degree per argument. Maps each `Argument.id` to a
    /// degree in `[0, 1]`.
    pub acceptability: HashMap<Uuid, f64>,
    /// Number of iteration rounds executed (`≤ MAX_GRADUAL_ITERATIONS`).
    pub iterations: u32,
    /// `true` iff `||Acc_{i+1} - Acc_i||_∞ < CONVERGENCE_EPSILON`
    /// before the cap. `false` indicates the cap was hit and the result
    /// reflects the last computed iterate.
    pub converged: bool,
}

/// T-norm-parametric influence-step denominator. Default `Godel` returns
/// `s` unchanged → bit-identical to the canonical paper formulas.
///
/// `Hamacher` applies a lossy clamp to `[0, 1]` because t-conorms are
/// only defined on the unit interval; for large attack sums this
/// projects all `s ≥ 1` to `S(1, 1) = 1`. Documented as a known
/// trade-off in the module doc.
#[inline]
pub(crate) fn infl_denom(s: f64, tnorm: Option<TNormKind>) -> f64 {
    match tnorm.unwrap_or(TNormKind::Godel) {
        TNormKind::Godel => s,
        TNormKind::Lukasiewicz => s.min(1.0),
        TNormKind::Goguen => 1.0 - (-s).exp(),
        TNormKind::Hamacher(lambda) => {
            let sc = s.clamp(0.0, 1.0);
            tconorm_for(TNormKind::Hamacher(lambda)).combine(sc, sc)
        }
    }
}

/// Run gradual / ranking-based argumentation on an explicit framework.
///
/// `tnorm = None` corresponds to the legacy default that reproduces the
/// canonical formulas of the cited papers bit-identically.
pub fn run_gradual_argumentation(
    framework: &ArgumentationFramework,
    kind: &GradualSemanticsKind,
    tnorm: Option<TNormKind>,
) -> Result<GradualResult> {
    run_gradual_inner(framework, kind, tnorm, MAX_GRADUAL_ITERATIONS)
}

/// Test-only entry point that accepts a custom iteration cap. Used by
/// the convergence-table tests (§5.2) to deliberately force `capped`
/// behaviour on cells without a contraction proof.
#[cfg(test)]
pub(crate) fn run_gradual_argumentation_with_cap(
    framework: &ArgumentationFramework,
    kind: &GradualSemanticsKind,
    tnorm: Option<TNormKind>,
    cap: u32,
) -> Result<GradualResult> {
    run_gradual_inner(framework, kind, tnorm, cap)
}

fn run_gradual_inner(
    framework: &ArgumentationFramework,
    kind: &GradualSemanticsKind,
    tnorm: Option<TNormKind>,
    cap: u32,
) -> Result<GradualResult> {
    let n = framework.arguments.len();

    // Validate weighted-h-categoriser construction invariants up front.
    // Surfaces the most actionable error message before we waste work
    // building the index structures.
    if let GradualSemanticsKind::WeightedHCategoriser { weights } = kind {
        if weights.len() != framework.attacks.len() {
            return Err(TensaError::InvalidInput(format!(
                "weighted h-categoriser: weights length {} does not match attacks length {}",
                weights.len(),
                framework.attacks.len()
            )));
        }
        // Per-target weight-sum constraint Σ_b v_{ba} ≤ 1 (Amgoud &
        // Ben-Naim 2017 §3 Remark 1). 1e-9 tolerance for FP rounding.
        let mut per_target = vec![0.0_f64; n];
        for (k, &(_, t)) in framework.attacks.iter().enumerate() {
            per_target[t] += weights[k];
        }
        for (a, &sum) in per_target.iter().enumerate() {
            if sum > 1.0 + 1e-9 {
                return Err(TensaError::InvalidInput(format!(
                    "weighted h-categoriser: target argument {} has incoming-weight sum {} > 1.0",
                    a, sum
                )));
            }
        }
    }

    if n == 0 {
        return Ok(GradualResult {
            acceptability: HashMap::new(),
            iterations: 0,
            converged: true,
        });
    }

    // Per-argument intrinsic strength `w(a)`. Clamped to [0, 1] so a
    // stale negative confidence cannot drive the recurrence outside the
    // unit interval.
    let w: Vec<f64> = framework
        .arguments
        .iter()
        .map(|a| (a.confidence as f64).clamp(0.0, 1.0))
        .collect();

    // Per-target attacker indices. For weighted, also collect the
    // parallel weight slice.
    let mut attackers_of: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut weights_of: Vec<Vec<f64>> = vec![Vec::new(); n];
    let weights_slice: Option<&[f64]> = match kind {
        GradualSemanticsKind::WeightedHCategoriser { weights } => Some(weights.as_slice()),
        _ => None,
    };
    for (k, &(attacker, target)) in framework.attacks.iter().enumerate() {
        if attacker >= n || target >= n {
            return Err(TensaError::InvalidInput(format!(
                "attack ({}, {}) references out-of-range argument index (n = {})",
                attacker, target, n
            )));
        }
        attackers_of[target].push(attacker);
        if let Some(ws) = weights_slice {
            weights_of[target].push(ws[k]);
        }
    }

    let mut acc_prev = w.clone();
    let mut acc_next = vec![0.0_f64; n];
    let mut iterations: u32 = 0;
    let mut converged = false;

    for _ in 0..cap {
        iterations += 1;
        // Index loops here are clearer than zip-iterators because each
        // iteration reads from multiple parallel slices.
        #[allow(clippy::needless_range_loop)]
        for a in 0..n {
            acc_next[a] = step_one(a, kind, &w, &acc_prev, &attackers_of, &weights_of, tnorm);
        }
        let delta = acc_next
            .iter()
            .zip(acc_prev.iter())
            .map(|(n, p)| (n - p).abs())
            .fold(0.0_f64, f64::max);
        std::mem::swap(&mut acc_prev, &mut acc_next);
        if delta < CONVERGENCE_EPSILON {
            converged = true;
            break;
        }
    }

    if !converged {
        warn!(
            "gradual semantics did not converge in {} iterations",
            iterations
        );
    }

    let mut acceptability = HashMap::with_capacity(n);
    for (i, arg) in framework.arguments.iter().enumerate() {
        acceptability.insert(arg.id, acc_prev[i]);
    }

    Ok(GradualResult {
        acceptability,
        iterations,
        converged,
    })
}

/// Single-argument update body. Branches on the semantics-specific
/// aggregator and delegates the t-norm influence to [`infl_denom`].
#[inline]
fn step_one(
    a: usize,
    kind: &GradualSemanticsKind,
    w: &[f64],
    acc_prev: &[f64],
    attackers_of: &[Vec<usize>],
    weights_of: &[Vec<f64>],
    tnorm: Option<TNormKind>,
) -> f64 {
    let attackers = &attackers_of[a];
    match kind {
        GradualSemanticsKind::HCategoriser => {
            let s: f64 = attackers.iter().map(|&b| acc_prev[b]).sum();
            w[a] / (1.0 + infl_denom(s, tnorm))
        }
        GradualSemanticsKind::WeightedHCategoriser { .. } => {
            let s: f64 = weights_of[a]
                .iter()
                .zip(attackers.iter())
                .map(|(&v, &b)| v * acc_prev[b])
                .sum();
            w[a] / (1.0 + infl_denom(s, tnorm))
        }
        GradualSemanticsKind::MaxBased => {
            let s = attackers
                .iter()
                .map(|&b| acc_prev[b])
                .fold(0.0_f64, f64::max);
            w[a] / (1.0 + infl_denom(s, tnorm))
        }
        GradualSemanticsKind::CardBased => {
            // Cardinality of strictly-positive attackers + sum over ALL
            // attackers (Amgoud & Ben-Naim 2013 §4.3 Definition 6).
            // T-norm modulates the sum component only; the cardinality
            // factor stays raw — see module doc.
            let mut card: u32 = 0;
            let mut sum: f64 = 0.0;
            for &b in attackers {
                let v = acc_prev[b];
                if v > 0.0 {
                    card += 1;
                }
                sum += v;
            }
            w[a] / ((1.0 + card as f64) * (1.0 + infl_denom(sum, tnorm)))
        }
    }
}

#[cfg(test)]
#[path = "argumentation_gradual_tests.rs"]
mod tests;
