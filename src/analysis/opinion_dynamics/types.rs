//! Public types for [`super`] (EATH Phase 16b — opinion dynamics on hypergraphs).
//!
//! All types are `serde`-derivable for KV / REST round-tripping. JsonSchema
//! derives are deliberately *not* added in Phase 16b — they belong in 16c
//! when these types start flowing through the MCP tool interface.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Core algorithmic enums ──────────────────────────────────────────────────

/// Algorithm variant. Both share the same hyperedge-selection / convergence
/// machinery; they differ only in the per-edge update rule (see [`super::update`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BcmVariant {
    /// Hickok et al. 2022 — generalises Deffuant 2000 dyadic BCM by applying
    /// the pairwise update to every ordered pair within the selected
    /// hyperedge in canonical order (Gauss-Seidel).
    #[default]
    PairwiseWithin,
    /// Schawe & Hernández 2022 — when the spread within the selected edge is
    /// below `c_e`, every member moves toward the group mean by `μ`.
    /// Otherwise, no update.
    GroupMean,
}

/// How to pick which hyperedge(s) to update at each step.
///
/// **Default is [`Self::UniformRandom`]** — paper-fidelity for the Hickok 2022
/// experimental setup. `ActivityWeighted` is a production-realistic
/// alternative for TENSA narratives where high-confidence situations are
/// more salient. `PerStepAll` changes the dynamics fundamentally and is
/// reserved for deterministic-convergence analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HyperedgeSelection {
    #[default]
    UniformRandom,
    ActivityWeighted,
    PerStepAll,
}

/// Initial opinion distribution.
///
/// `Gaussian` clamps draws into `[0, 1]` (no resampling) — this preserves the
/// boundary-accumulation effect at 0 and 1 used in the opinion-dynamics
/// literature.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InitialOpinionDist {
    /// Uniform over `[0, 1]`.
    #[default]
    Uniform,
    /// `N(mean, std²)` clamped to `[0, 1]`.
    Gaussian { mean: f32, std: f32 },
    /// Coin-flip between `mode_a` and `mode_b`, plus `N(0, spread²)` noise,
    /// clamped to `[0, 1]`.
    Bimodal { mode_a: f32, mode_b: f32, spread: f32 },
    /// Caller-supplied vector; length must equal the number of entities in
    /// the narrative or [`super::simulate_opinion_dynamics`] returns
    /// `TensaError::InvalidInput`.
    Custom(Vec<f32>),
}

/// Optional size-dependent scaling for the effective confidence bound.
///
/// `None`/`Flat` (the default) means `c_e = c` for every hyperedge. The
/// inverse variants reduce `c_e` for larger groups, requiring stronger
/// cohesion for them to interact (Hickok §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceScaling {
    /// `c_e = c / sqrt(|e| - 1)`.
    InverseSqrtSize,
    /// `c_e = c / (|e| - 1)`.
    InverseSize,
    /// `c_e = c` (default).
    #[default]
    Flat,
}

// ── Params ─────────────────────────────────────────────────────────────────

/// Parameters for one opinion-dynamics simulation.
///
/// Defaults match design doc §2.1: PairwiseWithin / UniformRandom / Uniform
/// initial dist / `c = 0.3` / `μ = 0.5` / `N_conv = 100` / `max_steps = 100k`
/// / `seed = 42`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpinionDynamicsParams {
    pub model: BcmVariant,
    /// Global confidence bound `c ∈ (0, 1)`. Strictly validated.
    pub confidence_bound: f32,
    /// Optional size-dependent scaling. `None` ≡ `Flat`.
    pub confidence_size_scaling: Option<ConfidenceScaling>,
    /// Convergence rate `μ ∈ (0, 1]`. Deffuant canonical: `0.5`.
    pub convergence_rate: f32,
    pub hyperedge_selection: HyperedgeSelection,
    pub initial_opinion_distribution: InitialOpinionDist,
    /// `ε_conv` — convergence tolerance on global spread.
    pub convergence_tol: f32,
    /// `N_conv` — required consecutive sub-tolerance steps. Pragmatic default
    /// 100; see §1.3 / §14 Q2 for rationale.
    pub convergence_window: usize,
    pub max_steps: usize,
    /// RNG seed for the single ChaCha8Rng driving the run.
    pub seed: u64,
}

impl Default for OpinionDynamicsParams {
    fn default() -> Self {
        Self {
            model: BcmVariant::default(),
            confidence_bound: 0.3,
            confidence_size_scaling: None,
            convergence_rate: 0.5,
            hyperedge_selection: HyperedgeSelection::default(),
            initial_opinion_distribution: InitialOpinionDist::default(),
            convergence_tol: 1e-4,
            convergence_window: 100,
            max_steps: 100_000,
            seed: 42,
        }
    }
}

// ── Outputs ────────────────────────────────────────────────────────────────

/// Opinion snapshots taken at log-spaced intervals over the simulation.
///
/// `opinion_history[k][i]` is the opinion of entity `entity_order[i]` at
/// step `sample_steps[k]`. Step 0 is always the first sample; the final
/// executed step is always the last sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpinionTrajectory {
    pub opinion_history: Vec<Vec<f32>>,
    pub sample_steps: Vec<usize>,
    pub final_opinions: HashMap<Uuid, f32>,
    pub entity_order: Vec<Uuid>,
}

/// Top-level report from [`super::simulate_opinion_dynamics`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpinionDynamicsReport {
    pub num_steps_executed: usize,
    pub converged: bool,
    pub convergence_step: Option<usize>,
    pub num_clusters: usize,
    pub cluster_sizes: Vec<usize>,
    pub cluster_means: Vec<f32>,
    /// Opinion variance at each `trajectory.sample_steps` point.
    pub variance_timeseries: Vec<f32>,
    /// Normalised-variance polarisation index, range `[0, 1]`. See §9.3.
    pub polarization_index: f32,
    /// Echo-chamber index, range `[0, 1]`. Requires pre-computed
    /// label-propagation labels at `an/lp/{narrative_id}/{entity_id}`. When
    /// labels are missing, this is `0.0` and `echo_chamber_available = false`.
    pub echo_chamber_index: f32,
    pub echo_chamber_available: bool,
    pub trajectory: OpinionTrajectory,
    /// Snapshot of the params used for this run, for reproducibility.
    pub params_used: OpinionDynamicsParams,
}

/// Output of [`super::run_phase_transition_sweep`].
///
/// `convergence_times[i]` is `None` when the sweep at `c_values[i]` hit the
/// `max_steps` cutoff without converging. `critical_c_estimate` is the
/// smallest `c_i` whose convergence-time exceeds
/// `median(convergence_times) * spike_threshold` — `None` when no such spike
/// is observed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseTransitionReport {
    pub c_values: Vec<f32>,
    pub convergence_times: Vec<Option<usize>>,
    pub critical_c_estimate: Option<f32>,
    /// Variance of the initial opinion vector (sampled with `base_params.seed`).
    pub initial_variance: f32,
    /// Multiplicative threshold for spike detection (default 3.0×).
    pub spike_threshold: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_match_design_doc() {
        let p = OpinionDynamicsParams::default();
        assert_eq!(p.model, BcmVariant::PairwiseWithin);
        assert_eq!(p.confidence_bound, 0.3);
        assert!(p.confidence_size_scaling.is_none());
        assert_eq!(p.convergence_rate, 0.5);
        assert_eq!(p.hyperedge_selection, HyperedgeSelection::UniformRandom);
        assert_eq!(p.initial_opinion_distribution, InitialOpinionDist::Uniform);
        assert_eq!(p.convergence_tol, 1e-4);
        assert_eq!(p.convergence_window, 100);
        assert_eq!(p.max_steps, 100_000);
        assert_eq!(p.seed, 42);
    }

    #[test]
    fn test_params_roundtrip_serde_json() {
        let p = OpinionDynamicsParams {
            model: BcmVariant::GroupMean,
            confidence_size_scaling: Some(ConfidenceScaling::InverseSqrtSize),
            initial_opinion_distribution: InitialOpinionDist::Bimodal {
                mode_a: 0.1,
                mode_b: 0.9,
                spread: 0.05,
            },
            ..Default::default()
        };
        let s = serde_json::to_string(&p).unwrap();
        let q: OpinionDynamicsParams = serde_json::from_str(&s).unwrap();
        assert_eq!(p, q);
    }
}
