//! EATH Extension Phase 14 — Bistability / hysteresis in higher-order contagion.
//!
//! Forward-backward β-sweep method of Ferraz de Arruda et al.
//! (Nat. Commun. 14:1375 2023). Forward branch starts from
//! `initial_prevalence_low`, backward from `initial_prevalence_high`;
//! hysteresis is the gap between the two at the same β. Transition-type
//! classification follows the Nat. Rev. Phys. 2024 review taxonomy:
//! `gap < 0.05` → Continuous; `gap >= 0.30 OR bistable_interval is Some`
//! → Discontinuous; else Hybrid.
//!
//! ## Reduction-to-pairwise contract — preserved
//!
//! The sweep dispatches to
//! [`super::higher_order_contagion::simulate_higher_order_sir`] for every
//! (β, branch, replicate) triple. Since we never modify the simulator,
//! Phase 7b's load-bearing reduction contract (`beta_per_size = [β, 0, 0,
//! ...]` + `threshold = Absolute(1)` ⇒ pairwise SIR) is automatically
//! preserved. T1 in tests/contagion_bistability_tests.rs uses the same
//! pairwise-equivalent params Phase 7b's regression test does.
//!
//! ## Determinism + parallelism
//!
//! Per-(β, replicate, branch) seeds are XOR-mixes of a base seed with
//! `(beta_idx, replicate_idx, branch_tag)`. Bit-identical reports across
//! thread-scope reorderings. Outer parallelism is per-(β, branch) via
//! `std::thread::scope` (Phase 13c pattern); replicates are sequential
//! within each spawned task.

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

use super::higher_order_contagion::{
    simulate_higher_order_sir, HigherOrderSirParams, HigherOrderSirResult, SeedStrategy,
    ThresholdRule,
};

/// How β scales with hyperedge size when the sweep varies the size-2 β.
/// `UniformScaled { factor }` ⇒ β_d = factor × β_2 for d ≥ 3.
/// `Custom(rest)` overrides higher-d rates explicitly (β_2 is still the
/// swept variable, `rest[i]` is the size-(i+3) rate).
/// Wire formats: `{"kind": "uniform_scaled", "value": {"factor": 1.0}}` |
/// `{"kind": "custom", "value": [1.2, 1.2, 1.2]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum BetaScaling {
    UniformScaled { factor: f32 },
    Custom(Vec<f32>),
}

impl Default for BetaScaling {
    fn default() -> Self {
        BetaScaling::UniformScaled { factor: 1.0 }
    }
}

/// Parameters for one bistability sweep. See module-level doc for the
/// forward/backward branch convention + transition-type taxonomy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BistabilitySweepParams {
    /// Linspace `(start, end, num_points)` of size-2 β values to sweep.
    pub beta_0_range: (f32, f32, usize),
    /// How β scales with hyperedge size. See [`BetaScaling`].
    pub beta_scaling: BetaScaling,
    /// Recovery rate γ. 0.0 disables recovery.
    pub gamma: f32,
    /// Threshold rule gating transmission (Phase 7b type reused).
    pub threshold: ThresholdRule,
    /// Forward-branch initial infected fraction (default 0.01).
    pub initial_prevalence_low: f32,
    /// Backward-branch initial infected fraction (default 0.5).
    pub initial_prevalence_high: f32,
    /// Per-β simulation length in steps.
    pub steady_state_steps: usize,
    /// Replicates per (β, branch).
    pub replicates_per_beta: usize,
    /// Contiguous β-span where `(backward - forward) > threshold` counts as
    /// bistable (default 0.15).
    #[serde(default = "default_bistable_gap_threshold")]
    pub bistable_gap_threshold: f32,
    /// Base RNG seed; per-(β, replicate, branch) seeds are XOR-mixed off it.
    #[serde(default = "default_base_seed")]
    pub base_seed: u64,
}

fn default_bistable_gap_threshold() -> f32 {
    0.15
}

fn default_base_seed() -> u64 {
    0xBADC_AFE5_5BAD_C0DE
}

impl BistabilitySweepParams {
    /// Quick smoke sweep on small narratives: 10 β-points × 5 replicates ×
    /// 200 steps. For tests; not statistically tight for paper results.
    pub fn quick() -> Self {
        Self {
            beta_0_range: (0.0, 1.0, 10),
            beta_scaling: BetaScaling::UniformScaled { factor: 1.0 },
            gamma: 0.1,
            threshold: ThresholdRule::Absolute(1),
            initial_prevalence_low: 0.01,
            initial_prevalence_high: 0.5,
            steady_state_steps: 200,
            replicates_per_beta: 5,
            bistable_gap_threshold: 0.15,
            base_seed: default_base_seed(),
        }
    }
}

/// One sweep curve — parallel `Vec`s indexed together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HysteresisCurve {
    pub beta_values: Vec<f32>,
    pub forward_prevalence: Vec<f32>,
    pub backward_prevalence: Vec<f32>,
    pub forward_std: Vec<f32>,
    pub backward_std: Vec<f32>,
}

/// Transition-regime classification (see module-doc table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionType {
    Continuous,
    Discontinuous,
    Hybrid,
}

/// Bistability sweep output: curve + derived diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BistabilityReport {
    pub curve: HysteresisCurve,
    pub bistable_interval: Option<(f32, f32)>,
    pub transition_type: TransitionType,
    pub max_hysteresis_gap: f32,
    pub critical_beta_estimate: Option<f32>,
}

/// Forward-backward β-sweep on `narrative_id`. For each β in linspace, runs
/// `replicates_per_beta` forward sims (from `initial_prevalence_low`) and
/// `replicates_per_beta` backward sims (from `initial_prevalence_high`).
/// Steady-state prevalence = mean of last 10% of `per_step_infected` /
/// `total_entities`. Empty narrative → degenerate report; simulator handles
/// 0-entity gracefully and we propagate.
pub fn run_bistability_sweep(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &BistabilitySweepParams,
) -> Result<BistabilityReport> {
    let (beta_start, beta_end, num_points) = params.beta_0_range;
    let bad = |msg: &str| TensaError::InvalidInput(format!("bistability sweep: {msg}"));
    if narrative_id.is_empty() {
        return Err(bad("narrative_id is empty"));
    }
    if num_points == 0 {
        return Err(bad("num_points must be >= 1"));
    }
    if !beta_start.is_finite() || !beta_end.is_finite() {
        return Err(bad("beta_start/beta_end must be finite"));
    }
    if params.replicates_per_beta == 0 {
        return Err(bad("replicates_per_beta must be >= 1"));
    }
    if params.steady_state_steps == 0 {
        return Err(bad("steady_state_steps must be >= 1"));
    }
    if !(0.0..=1.0).contains(&params.initial_prevalence_low)
        || !(0.0..=1.0).contains(&params.initial_prevalence_high)
    {
        return Err(bad("initial prevalences must be in [0, 1]"));
    }

    let betas: Vec<f32> = linspace(beta_start, beta_end, num_points);

    // 2. One thread per (β_idx, branch). Slot index = 2*beta_idx + branch_tag
    //    (0=forward, 1=backward). Mirrors Phase 13c's per-model scope pattern.
    let n_slots = betas.len() * 2;
    let mut slots: Vec<Option<Result<BranchOutcome>>> = (0..n_slots).map(|_| None).collect();

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_slots);
        for (beta_idx, &beta) in betas.iter().enumerate() {
            for branch_tag in 0u8..2 {
                let init_prevalence = if branch_tag == 0 {
                    params.initial_prevalence_low
                } else {
                    params.initial_prevalence_high
                };
                let handle = scope.spawn(move || {
                    let outcome = run_branch(
                        hypergraph,
                        narrative_id,
                        beta,
                        init_prevalence,
                        params,
                        beta_idx,
                        branch_tag,
                    );
                    (beta_idx, branch_tag, outcome)
                });
                handles.push(handle);
            }
        }
        for handle in handles {
            if let Ok((beta_idx, branch_tag, res)) = handle.join() {
                let slot = 2 * beta_idx + branch_tag as usize;
                slots[slot] = Some(res);
            }
        }
    });

    // 3. Reassemble into the curve. Any single failure aborts the sweep.
    let mut forward_prevalence = vec![0.0_f32; betas.len()];
    let mut backward_prevalence = vec![0.0_f32; betas.len()];
    let mut forward_std = vec![0.0_f32; betas.len()];
    let mut backward_std = vec![0.0_f32; betas.len()];
    for (idx, slot) in slots.into_iter().enumerate() {
        let beta_idx = idx / 2;
        let branch_tag = idx % 2;
        let outcome = match slot {
            Some(Ok(o)) => o,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(TensaError::InferenceError(format!(
                    "bistability sweep: slot {idx} thread panicked"
                )));
            }
        };
        if branch_tag == 0 {
            forward_prevalence[beta_idx] = outcome.mean;
            forward_std[beta_idx] = outcome.std;
        } else {
            backward_prevalence[beta_idx] = outcome.mean;
            backward_std[beta_idx] = outcome.std;
        }
    }

    let curve = HysteresisCurve {
        beta_values: betas,
        forward_prevalence,
        backward_prevalence,
        forward_std,
        backward_std,
    };

    // 4. Diagnostics.
    let max_hysteresis_gap = compute_max_gap(&curve);
    let bistable_interval = detect_bistable_interval(&curve, params.bistable_gap_threshold);
    let transition_type = classify_transition(max_hysteresis_gap, bistable_interval);
    let critical_beta_estimate = estimate_critical_beta(&curve, bistable_interval);

    Ok(BistabilityReport {
        curve,
        bistable_interval,
        transition_type,
        max_hysteresis_gap,
        critical_beta_estimate,
    })
}

// ── Per-branch dispatch ─────────────────────────────────────────────────────

/// Aggregate of replicates for one (β, branch) tuple.
struct BranchOutcome {
    mean: f32,
    std: f32,
}

/// Run `replicates_per_beta` forward (or backward) simulations for one β
/// and return mean + population stddev of steady-state prevalence.
fn run_branch(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    beta: f32,
    init_prevalence: f32,
    params: &BistabilitySweepParams,
    beta_idx: usize,
    branch_tag: u8,
) -> Result<BranchOutcome> {
    let beta_per_size = build_beta_per_size(beta, &params.beta_scaling);

    let mut prevalences: Vec<f32> = Vec::with_capacity(params.replicates_per_beta);
    for replicate_idx in 0..params.replicates_per_beta {
        let rng_seed = mix_seed(params.base_seed, beta_idx, replicate_idx, branch_tag);
        let sir_params = HigherOrderSirParams {
            beta_per_size: beta_per_size.clone(),
            gamma: params.gamma,
            threshold: params.threshold,
            seed_strategy: SeedStrategy::RandomFraction {
                fraction: init_prevalence,
            },
            max_steps: params.steady_state_steps,
            rng_seed,
        };
        let result = simulate_higher_order_sir(hypergraph, narrative_id, &sir_params)?;
        prevalences.push(steady_state_prevalence(&result));
    }
    let (mean, std) = mean_std(&prevalences);
    Ok(BranchOutcome { mean, std })
}

/// Build per-size β vector from swept β_2 + scaling rule. Sized for sizes
/// 2..10; larger hyperedges contribute zero (silently skipped by simulator).
fn build_beta_per_size(beta: f32, scaling: &BetaScaling) -> Vec<f32> {
    const DEFAULT_LEN: usize = 9;
    match scaling {
        BetaScaling::UniformScaled { factor } => {
            let f = factor.max(0.0);
            let mut v = Vec::with_capacity(DEFAULT_LEN);
            v.push(beta);
            for _ in 1..DEFAULT_LEN {
                v.push(beta * f);
            }
            v
        }
        BetaScaling::Custom(rest) => {
            let mut v = Vec::with_capacity(1 + rest.len());
            v.push(beta);
            v.extend(rest.iter().copied());
            v
        }
    }
}

/// Steady-state prevalence: mean of last 10% of `per_step_infected` /
/// `total_entities`. Falls back to the last step on early burnout.
fn steady_state_prevalence(result: &HigherOrderSirResult) -> f32 {
    if result.total_entities == 0 || result.per_step_infected.is_empty() {
        return 0.0;
    }
    let total = result.total_entities as f32;
    let n = result.per_step_infected.len();
    let window = (n / 10).max(1);
    let start = n.saturating_sub(window);
    let sum: usize = result.per_step_infected[start..].iter().sum();
    let cnt = (n - start) as f32;
    (sum as f32 / cnt) / total
}

/// Population mean + stddev; returns `(0, 0)` on empty input.
fn mean_std(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() { return (0.0, 0.0); }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Linspace inclusive of both endpoints; n=0 → empty, n=1 → `[start]`.
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![start]; }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| start + step * i as f32).collect()
}

// ── Diagnostics ──────────────────────────────────────────────────────────────

/// Maximum |backward - forward| across the curve.
fn compute_max_gap(curve: &HysteresisCurve) -> f32 {
    let mut max_gap = 0.0_f32;
    for (f, b) in curve
        .forward_prevalence
        .iter()
        .zip(curve.backward_prevalence.iter())
    {
        let gap = (b - f).abs();
        if gap > max_gap {
            max_gap = gap;
        }
    }
    max_gap
}

/// Detect contiguous β-range where `(backward - forward) > threshold`.
/// Returns the longest such run (left-most on ties).
fn detect_bistable_interval(
    curve: &HysteresisCurve,
    threshold: f32,
) -> Option<(f32, f32)> {
    let mut best: Option<(usize, usize)> = None;
    let mut cur_start: Option<usize> = None;
    for i in 0..curve.beta_values.len() {
        let gap = curve.backward_prevalence[i] - curve.forward_prevalence[i];
        if gap > threshold {
            if cur_start.is_none() {
                cur_start = Some(i);
            }
        } else if let Some(start) = cur_start.take() {
            let end = i - 1;
            if best.map(|(s, e)| (e - s) < (end - start)).unwrap_or(true) {
                best = Some((start, end));
            }
        }
    }
    if let Some(start) = cur_start {
        let end = curve.beta_values.len() - 1;
        if best.map(|(s, e)| (e - s) < (end - start)).unwrap_or(true) {
            best = Some((start, end));
        }
    }
    best.map(|(s, e)| (curve.beta_values[s], curve.beta_values[e]))
}

/// Apply the module-doc classification table.
fn classify_transition(gap: f32, bistable: Option<(f32, f32)>) -> TransitionType {
    if gap < 0.05 {
        return TransitionType::Continuous;
    }
    if gap >= 0.30 || bistable.is_some() {
        return TransitionType::Discontinuous;
    }
    TransitionType::Hybrid
}

/// Critical β: midpoint of detected interval, else β with steepest positive
/// slope on the forward branch.
fn estimate_critical_beta(
    curve: &HysteresisCurve,
    bistable_interval: Option<(f32, f32)>,
) -> Option<f32> {
    if let Some((lo, hi)) = bistable_interval {
        return Some((lo + hi) * 0.5);
    }
    if curve.beta_values.len() < 2 {
        return None;
    }
    let mut best_idx: Option<usize> = None;
    let mut best_slope = 0.0_f32;
    for i in 1..curve.forward_prevalence.len() {
        let dp = curve.forward_prevalence[i] - curve.forward_prevalence[i - 1];
        let dbeta = curve.beta_values[i] - curve.beta_values[i - 1];
        if dbeta.abs() < f32::EPSILON {
            continue;
        }
        let slope = dp / dbeta;
        if slope > best_slope {
            best_slope = slope;
            best_idx = Some(i);
        }
    }
    if best_slope <= 0.0 {
        return None;
    }
    best_idx.map(|i| (curve.beta_values[i] + curve.beta_values[i - 1]) * 0.5)
}

/// Per-(β, replicate, branch) seed via XOR mixing — deterministic across
/// thread-scope reorderings.
#[inline]
fn mix_seed(base: u64, beta_idx: usize, replicate_idx: usize, branch_tag: u8) -> u64 {
    const BETA_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
    const REPLICATE_MIX: u64 = 0xBF58_476D_1CE4_E5B9;
    const BRANCH_MIX: u64 = 0x94D0_49BB_1331_11EB;
    base ^ (beta_idx as u64).wrapping_mul(BETA_MIX)
        ^ (replicate_idx as u64).wrapping_mul(REPLICATE_MIX)
        ^ (branch_tag as u64).wrapping_mul(BRANCH_MIX)
}

// ── Conversion helpers ──────────────────────────────────────────────────────

/// Parse [`BistabilitySweepParams`] from a JSON blob — REST handler + engine
/// share one source of truth.
pub fn parse_params(value: &serde_json::Value) -> Result<BistabilitySweepParams> {
    if value.is_null() {
        return Err(TensaError::InvalidInput(
            "bistability sweep: params blob is null".into(),
        ));
    }
    serde_json::from_value(value.clone()).map_err(|e| {
        TensaError::InvalidInput(format!(
            "bistability sweep: invalid BistabilitySweepParams: {e}"
        ))
    })
}

/// Aggregate (interval_width, max_gap) from a report — used by the
/// surrogate-significance engine to build its per-K observation series.
#[inline]
pub fn summary_scalars(report: &BistabilityReport) -> (f32, f32) {
    let width = report
        .bistable_interval
        .map(|(lo, hi)| (hi - lo).max(0.0))
        .unwrap_or(0.0);
    (width, report.max_hysteresis_gap)
}


// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "contagion_bistability_tests.rs"]
mod tests;
