//! Supervised learning of Choquet fuzzy measures via projected gradient
//! descent on the pairwise-hinge ranking loss.
//!
//! ## What this module solves
//!
//! Given a dataset `[(input_vec, rank)]` where lower rank means
//! "more strongly coordinated", fit a fuzzy measure `μ` on the universe of
//! signals so the resulting Choquet integral preserves the ranking on a
//! held-out half. The full design lives in
//! [`docs/choquet_learning_algorithm.md`](../../docs/choquet_learning_algorithm.md);
//! this module is its in-tree implementation.
//!
//! ## Solver
//!
//! Pure-Rust projected gradient descent in μ-space (full `2^n` capacity
//! table). Per iteration:
//!
//! 1. Compute the analytic gradient of the pairwise hinge loss
//!    `L(μ) = Σ_{rank_i < rank_j} max(0, C_μ(x_j) - C_μ(x_i) + ε)` —
//!    closed-form `O(n)` per active pair via the Choquet integral's
//!    piecewise-linear structure (sort `x` ascending; tail subsets `A_k`
//!    contribute `x_(k) - x_(k-1)`).
//! 2. Take a step `μ' = μ - η · ∇L`.
//! 3. Project onto the feasible set: clip to `[0, 1]`, monotonicity
//!    sweep `μ(S ∪ {i}) ≥ μ(S)` smaller-subsets-first, then re-pin
//!    the boundaries `μ(∅) = 0` and `μ(N) = 1`.
//!
//! Termination on `max_A |Δμ[A]| < 1e-6` or 5 000 iterations, whichever
//! comes first. `n` is hard-capped at 6 — k-additive specialisation
//! (Phase 12 / future sprint) extends to `n > 6` per
//! [grabisch1997kadditive].
//!
//! ## Provenance contract (LOAD-BEARING)
//!
//! Every aggregation result that uses a learned measure carries
//! `fuzzy_config.measure_id` + `fuzzy_config.measure_version` in the
//! emitted envelope. Symmetric defaults emit `None`/`None` —
//! bit-identical to pre-Phase-0 envelopes. Three workflow surfaces
//! threaded the slot through their `_tracked` siblings in Phase 2:
//! `ConfidenceBreakdown::composite_with_aggregator_tracked`,
//! `aggregate_metrics_with_aggregator_tracked`, and
//! `RewardProfile::score_with_aggregator_tracked`.
//!
//! Cites: [grabisch1996choquet], [grabisch1997kadditive],
//!        [grabisch2000fuzzymeasure], [bustince2016choquet].

use std::time::Instant;

use chrono::{DateTime, Utc};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::FuzzyMeasure;
use crate::fuzzy::aggregation_measure::symmetric_additive;

// ── Numerical-stability constants ───────────────────────────────────────────

/// Maximum number of outer PGD iterations. Far above the ≤ 1 000 it
/// usually takes for `n ≤ 4`; warns when reached.
pub const MAX_PGD_ITERATIONS: usize = 5_000;

/// Iteration terminates when no entry of `μ` moved by more than this.
pub const PGD_TOLERANCE: f64 = 1e-6;

/// Default learning rate `η`. Halved on stall (no loss decrease in 100
/// iterations); halved again on divergence (loss increase in 200 iters).
///
/// Tuned for the **mean**-normalised pairwise hinge loss — when the
/// gradient was a `Σ` rather than a `mean` this needed to be `1e-4`
/// to avoid divergence on `m = 50`. Now it can be ~ unit scale.
pub const DEFAULT_LEARNING_RATE: f64 = 1.0;

/// Default pairwise-hinge margin `ε`.
pub const DEFAULT_MARGIN: f64 = 0.05;

/// Hard upper bound on `n` for the in-tree PGD path. `n = 6` ⇒ 64 capacity
/// entries, 192 monotonicity inequalities — fits the 60-second budget.
pub const N_CAP: u8 = 6;

// ── Default constructors ────────────────────────────────────────────────────

/// Default version stamp for `StoredMeasure` records that pre-date the
/// Graded Acceptability sprint. New measures created from Phase 2 onward
/// increment this when re-trained on a fresh dataset.
pub fn default_measure_version() -> u32 {
    1
}

// ── Provenance types ────────────────────────────────────────────────────────

/// Detail block describing how a learned Choquet measure was fit.
/// Persisted alongside the measure so paper-figure traces can recover
/// every relevant hyperparameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedMeasureProvenance {
    /// Caller-supplied dataset identifier (e.g. `"synthetic-cib-v1"`).
    pub dataset_id: String,
    /// Number of `(input_vec, rank)` pairs the fit consumed.
    pub n_samples: u32,
    /// Final pairwise-hinge loss after the PGD outer loop.
    pub fit_loss: f64,
    /// Solver tag — `"pgd"` for the in-tree pure-Rust baseline. If a QP
    /// crate ever lands, add `"osqp"` / `"clarabel"` variants.
    pub fit_method: String,
    /// Wall-clock seconds spent inside the fit routine.
    pub fit_seconds: f64,
    /// Timestamp the fit completed.
    pub trained_at: DateTime<Utc>,
}

/// Tag describing how a `StoredMeasure` came to exist. Lets the FUZZY
/// surface tell paper-figure callers whether they're looking at a
/// hand-rolled measure, a built-in symmetric default, or a learned
/// measure with full provenance.
///
/// `Default::default() = Manual` keeps `#[serde(default)]` cheap on
/// pre-sprint records that never wrote a provenance field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeasureProvenance {
    /// One of the `symmetric_*` built-ins. `kind` is one of
    /// `"additive"`, `"pessimistic"`, `"optimistic"`.
    Symmetric { kind: String },
    /// User-supplied via the existing `POST /fuzzy/measures` route or
    /// the `fuzzy_create_measure` MCP tool. Backward-compat default.
    Manual,
    /// Result of a Phase 2 ranking-supervised fit.
    Learned(LearnedMeasureProvenance),
}

impl Default for MeasureProvenance {
    fn default() -> Self {
        MeasureProvenance::Manual
    }
}

// ── Public entry point ──────────────────────────────────────────────────────

/// Output of a successful PGD fit.
///
/// `measure.measure_id` is left `None` — the REST handler that knows the
/// caller's chosen name stamps it. The library function does not assume
/// a name on its own.
#[derive(Debug, Clone)]
pub struct LearnedChoquetMeasureOutput {
    pub measure: FuzzyMeasure,
    pub provenance: LearnedMeasureProvenance,
    pub train_auc: f64,
    pub test_auc: f64,
}

/// Learn a Choquet fuzzy measure from a labelled `(input_vec, rank)`
/// dataset.
///
/// Lower rank = more strongly coordinated. The fit minimises pairwise
/// hinge loss on a deterministically split 50 / 50 train / test partition
/// (split seed derived from `dataset_id`).
///
/// Returns the learned measure together with provenance metadata and
/// ranking-AUC scores on both halves. The library does **not** assign a
/// `measure_id` — the calling REST handler does, since only it knows the
/// user-chosen name. Inspect the returned `measure` via
/// [`FuzzyMeasure::with_id`] to stamp it post-hoc.
///
/// # Errors
///
/// * `n > 6` — `InvalidInput` with a pointer to [grabisch1997kadditive].
/// * `dataset.len() < 4` — too few samples to split + train.
/// * Any `input_vec.len() != n` — `InvalidInput`.
pub fn learn_choquet_measure(
    n: u8,
    dataset: &[(Vec<f64>, u32)],
    dataset_id: &str,
) -> Result<LearnedChoquetMeasureOutput> {
    if n > N_CAP {
        return Err(TensaError::InvalidInput(
            "n > 6 requires k-additive specialisation; cite [grabisch1997kadditive]; \
             tracked in §12.2 of the architecture paper"
                .into(),
        ));
    }
    if n == 0 {
        return Err(TensaError::InvalidInput(
            "learn_choquet_measure: n must be ≥ 1".into(),
        ));
    }
    if dataset.len() < 4 {
        return Err(TensaError::InvalidInput(format!(
            "learn_choquet_measure: need at least 4 samples to split + train (got {})",
            dataset.len()
        )));
    }

    let n_us = n as usize;
    // Validate + clamp inputs.
    let mut sanitized: Vec<(Vec<f64>, u32)> = Vec::with_capacity(dataset.len());
    for (idx, (xs, rank)) in dataset.iter().enumerate() {
        if xs.len() != n_us {
            return Err(TensaError::InvalidInput(format!(
                "learn_choquet_measure: sample {idx} has |xs|={}, expected {n_us}",
                xs.len()
            )));
        }
        let clamped: Vec<f64> = xs.iter().map(|v| v.clamp(0.0, 1.0)).collect();
        sanitized.push((clamped, *rank));
    }

    // Deterministic split on a SHA-256 derived seed.
    let split_seed = sha256_u64(dataset_id.as_bytes());
    let (train, test) = split_dataset(&sanitized, split_seed);
    if train.is_empty() || test.is_empty() {
        return Err(TensaError::InvalidInput(format!(
            "learn_choquet_measure: degenerate split (train={}, test={}) — increase dataset size",
            train.len(),
            test.len()
        )));
    }

    // Initial μ = symmetric additive (a strictly feasible point).
    let mut mu = symmetric_additive(n)?.values;

    let started = Instant::now();
    let final_loss = run_pgd(&mut mu, &train, n_us);
    let elapsed = started.elapsed();

    let train_auc = ranking_auc(&mu, &train, n_us);
    let test_auc = ranking_auc(&mu, &test, n_us);

    let measure = FuzzyMeasure::with_id(n, mu, None, None)?;
    let provenance = LearnedMeasureProvenance {
        dataset_id: dataset_id.to_string(),
        n_samples: dataset.len() as u32,
        fit_loss: final_loss,
        fit_method: "pgd".to_string(),
        fit_seconds: elapsed.as_secs_f64(),
        trained_at: Utc::now(),
    };

    Ok(LearnedChoquetMeasureOutput {
        measure,
        provenance,
        train_auc,
        test_auc,
    })
}

// ── Provenance resolver shared by every `_tracked` workflow wire ───────────

/// Resolve which `(measure_id, measure_version)` pair a `_tracked`
/// workflow site should echo back to its caller.
///
/// Slot rules:
/// 1. Caller-supplied slots take priority.
/// 2. If `agg` is [`crate::fuzzy::aggregation::AggregatorKind::Choquet`]
///    and the embedded `FuzzyMeasure` carries identity, those are used.
/// 3. Otherwise both slots are `None` (bit-identical to pre-Phase-2).
pub fn resolve_measure_provenance(
    agg: &crate::fuzzy::aggregation::AggregatorKind,
    measure_id: Option<String>,
    measure_version: Option<u32>,
) -> (Option<String>, Option<u32>) {
    if measure_id.is_some() {
        return (measure_id, measure_version);
    }
    if let crate::fuzzy::aggregation::AggregatorKind::Choquet(measure) = agg {
        if measure.measure_id.is_some() {
            return (measure.measure_id.clone(), measure.measure_version);
        }
    }
    (None, None)
}

// ── PGD core loop ───────────────────────────────────────────────────────────

/// Per-input pre-computed sort permutation. Computed once per
/// μ-evaluation and shared by both the score read and the gradient
/// accumulation, so we never sort the same input twice in one iteration.
type SortIndex = Vec<Vec<usize>>;

/// Build the per-input ascending-sort permutation. Allocated once at
/// the top of [`run_pgd`] and re-used across all iterations — the
/// permutations depend only on the input vectors, not on `μ`.
fn build_sort_index(dataset: &[(Vec<f64>, u32)], n: usize) -> SortIndex {
    dataset
        .iter()
        .map(|(xs, _)| {
            let mut sorted: Vec<usize> = (0..n).collect();
            sorted.sort_by(|&a, &b| {
                xs[a]
                    .partial_cmp(&xs[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted
        })
        .collect()
}

/// Compute mean pairwise hinge loss + analytic gradient in a single
/// pass. The gradient buffer is reused across iterations (caller
/// supplies it; routine zero-fills before accumulating).
fn loss_and_grad(
    mu: &[f64],
    dataset: &[(Vec<f64>, u32)],
    sort_index: &SortIndex,
    n: usize,
    grad_out: &mut [f64],
) -> f64 {
    grad_out.fill(0.0);
    let m = dataset.len();
    if m < 2 {
        return 0.0;
    }
    // Score per input under the current μ, reusing the cached sort
    // permutations.
    let scores: Vec<f64> = dataset
        .iter()
        .zip(sort_index.iter())
        .map(|((xs, _), sorted)| score_from_sorted(mu, xs, sorted, n))
        .collect();

    let mut loss = 0.0_f64;
    let mut pair_count: usize = 0;
    for i in 0..m {
        for j in 0..m {
            if dataset[i].1 >= dataset[j].1 {
                continue;
            }
            pair_count += 1;
            let h = scores[j] - scores[i] + DEFAULT_MARGIN;
            if h > 0.0 {
                loss += h;
                accumulate_choquet_gradient(grad_out, &dataset[j].0, &sort_index[j], n, 1.0);
                accumulate_choquet_gradient(grad_out, &dataset[i].0, &sort_index[i], n, -1.0);
            }
        }
    }
    if pair_count == 0 {
        return 0.0;
    }
    let inv = 1.0 / pair_count as f64;
    for g in grad_out.iter_mut() {
        *g *= inv;
    }
    loss * inv
}

/// Run PGD in place on `mu`. Returns the final pairwise-hinge loss.
///
/// The loss is mean-normalised so the LR is unit-scale. Adaptive
/// schedule: rollback + halve on divergence, halve on stall.
fn run_pgd(mu: &mut [f64], train: &[(Vec<f64>, u32)], n: usize) -> f64 {
    let size = 1usize << n;
    let sort_index = build_sort_index(train, n);
    let mut grad = vec![0.0_f64; size];
    let mut snapshot = vec![0.0_f64; size];
    let mut prev_mu = mu.to_vec();

    let mut lr = DEFAULT_LEARNING_RATE;
    // Compute the initial loss — gradient is unused this round.
    let mut last_loss = loss_and_grad(mu, train, &sort_index, n, &mut grad);
    let mut stall_counter: usize = 0;

    for iter in 0..MAX_PGD_ITERATIONS {
        // Re-compute gradient at the current μ (loss returned here is
        // not load-bearing — we re-derive `cur_loss` after the step).
        let _ = loss_and_grad(mu, train, &sort_index, n, &mut grad);

        snapshot.copy_from_slice(mu);
        for (m, g) in mu.iter_mut().zip(grad.iter()) {
            *m -= lr * g;
        }
        project_to_feasible(mu, n);

        // Recompute loss at the new μ. Pass an unused gradient buffer to
        // avoid an allocation; the gradient itself is overwritten next
        // iteration anyway.
        let cur_loss = loss_and_grad(mu, train, &sort_index, n, &mut grad);
        if cur_loss > last_loss + 1e-9 {
            // Divergence — roll back, halve LR, retry next iteration.
            mu.copy_from_slice(&snapshot);
            lr *= 0.5;
            tracing::debug!(
                target: "tensa::fuzzy",
                "pgd diverging at iter {iter}; rolling back, lr → {lr}"
            );
            if lr < 1e-9 {
                tracing::debug!(
                    target: "tensa::fuzzy",
                    "pgd lr collapsed below 1e-9 at iter {iter}; halting"
                );
                return last_loss;
            }
            continue;
        }

        let max_delta = mu
            .iter()
            .zip(prev_mu.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        prev_mu.copy_from_slice(mu);

        if cur_loss + 1e-12 < last_loss {
            stall_counter = 0;
        } else {
            stall_counter += 1;
            if stall_counter >= 50 {
                lr *= 0.5;
                stall_counter = 0;
                tracing::debug!(
                    target: "tensa::fuzzy",
                    "pgd stalled at iter {iter}; halving lr → {lr}"
                );
            }
        }
        last_loss = cur_loss;

        if max_delta < PGD_TOLERANCE {
            tracing::debug!(
                target: "tensa::fuzzy",
                "pgd converged in {iter} iterations (loss={last_loss:.6e})"
            );
            return last_loss;
        }
    }

    tracing::warn!(
        target: "tensa::fuzzy",
        "pgd hit MAX_PGD_ITERATIONS ({MAX_PGD_ITERATIONS}) without convergence (loss={last_loss:.6e})"
    );
    last_loss
}

// ── Loss + gradient + AUC ───────────────────────────────────────────────────

/// Mean pairwise hinge loss `mean_{rank_i < rank_j} max(0, C_μ(x_j) - C_μ(x_i) + ε)`.
///
/// Normalised by pair count so the gradient magnitude does not scale
/// quadratically with the dataset size — keeps the LR schedule stable
/// as `m` grows.
///
/// The PGD inner loop fuses loss + gradient via [`loss_and_grad`] so it
/// never calls this directly; this standalone helper exists so the
/// design §5 unit-test suite can hand-verify the loss value separately
/// from the gradient.
#[cfg(test)]
pub(crate) fn pairwise_hinge_loss(
    mu: &[f64],
    dataset: &[(Vec<f64>, u32)],
    n: usize,
    epsilon: f64,
) -> f64 {
    let m = dataset.len();
    if m < 2 {
        return 0.0;
    }
    // Pre-compute Choquet of every input under the current μ.
    let scores: Vec<f64> = dataset
        .iter()
        .map(|(xs, _)| choquet_via_mu(mu, xs, n))
        .collect();

    let mut loss = 0.0;
    let mut pair_count: usize = 0;
    for i in 0..m {
        for j in 0..m {
            if dataset[i].1 < dataset[j].1 {
                pair_count += 1;
                let h = scores[j] - scores[i] + epsilon;
                if h > 0.0 {
                    loss += h;
                }
            }
        }
    }
    if pair_count == 0 {
        0.0
    } else {
        loss / pair_count as f64
    }
}


/// Add `sign · ∂C_μ(x) / ∂μ[A_k]` for every tail subset `A_k` to `grad`.
fn accumulate_choquet_gradient(
    grad: &mut [f64],
    xs: &[f64],
    sorted: &[usize],
    n: usize,
    sign: f64,
) {
    let mut prev = 0.0_f64;
    let mut mask: usize = (1usize << n) - 1; // start = full universe (A_1)
    for rank in 0..n {
        let cur = xs[sorted[rank]];
        let delta = cur - prev;
        if delta != 0.0 {
            grad[mask] += sign * delta;
        }
        // Drop the rank-th index from the tail mask before next iteration.
        mask &= !(1usize << sorted[rank]);
        prev = cur;
    }
}

/// Ranking-AUC: fraction of `(i, j)` pairs with `rank_i < rank_j` for
/// which `score_i > score_j`. Ties at score boundary count 0.5.
pub(crate) fn ranking_auc(mu: &[f64], dataset: &[(Vec<f64>, u32)], n: usize) -> f64 {
    let m = dataset.len();
    if m < 2 {
        return 0.0;
    }
    let scores: Vec<f64> = dataset
        .iter()
        .map(|(xs, _)| choquet_via_mu(mu, xs, n))
        .collect();
    let mut correct = 0.0_f64;
    let mut total = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            if dataset[i].1 < dataset[j].1 {
                total += 1.0;
                if scores[i] > scores[j] {
                    correct += 1.0;
                } else if (scores[i] - scores[j]).abs() <= f64::EPSILON {
                    correct += 0.5;
                }
            }
        }
    }
    if total == 0.0 {
        0.0
    } else {
        correct / total
    }
}

// ── Choquet evaluation (no clamping during learning) ────────────────────────

/// Evaluate the Choquet integral of `xs` under capacity `mu` directly.
/// Does NOT clamp the output — clamping is correct for downstream
/// inference but masks gradients near the boundary during learning.
pub(crate) fn choquet_via_mu(mu: &[f64], xs: &[f64], n: usize) -> f64 {
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    score_from_sorted(mu, xs, &sorted, n)
}

/// Same as `choquet_via_mu` but reuses a caller-supplied permutation —
/// the gradient routine pre-sorts once per input and amortises the cost
/// over both the score read and the gradient accumulation.
fn score_from_sorted(mu: &[f64], xs: &[f64], sorted: &[usize], n: usize) -> f64 {
    let mut score = 0.0_f64;
    let mut prev = 0.0_f64;
    let mut mask: usize = (1usize << n) - 1;
    for rank in 0..n {
        let cur = xs[sorted[rank]];
        score += (cur - prev) * mu[mask];
        mask &= !(1usize << sorted[rank]);
        prev = cur;
    }
    score
}

// ── Projection onto the feasible set ────────────────────────────────────────

/// Clip + monotonicity sweep + boundary renormalisation.
pub(crate) fn project_to_feasible(mu: &mut [f64], n: usize) {
    let size = 1usize << n;
    // 1. Clip to [0, 1].
    for v in mu.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    // 2. Monotonicity sweep (smaller subsets first by bitmask order).
    for s in 0..size {
        for i in 0..n {
            let bit = 1usize << i;
            if s & bit == 0 {
                let t = s | bit;
                if mu[t] < mu[s] {
                    mu[t] = mu[s];
                }
            }
        }
    }
    // 3. Re-pin boundaries.
    mu[0] = 0.0;
    mu[size - 1] = 1.0;
}

// ── Train/test split ────────────────────────────────────────────────────────

fn sha256_u64(bytes: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(out)
}

fn split_dataset(
    dataset: &[(Vec<f64>, u32)],
    seed: u64,
) -> (Vec<(Vec<f64>, u32)>, Vec<(Vec<f64>, u32)>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    indices.shuffle(&mut rng);
    let split_point = dataset.len() / 2;
    let mut train = Vec::with_capacity(split_point);
    let mut test = Vec::with_capacity(dataset.len() - split_point);
    for (k, &idx) in indices.iter().enumerate() {
        if k < split_point {
            train.push(dataset[idx].clone());
        } else {
            test.push(dataset[idx].clone());
        }
    }
    (train, test)
}

#[cfg(test)]
#[path = "aggregation_learn_tests.rs"]
mod tests;
