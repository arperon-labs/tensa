//! Subjective Utility Quantal Response (SUQR) bounded rationality model.
//!
//! Extends QRE with subjective weights on payoff features rather than raw
//! expected utility. Includes λ-curriculum annealing for progressive
//! adversary strength escalation.
//!
//! ## References
//!
//! - Nguyen, Yang, Azaria, Kraus & Tambe (2013). "Analyzing the
//!   Effectiveness of Adversary Modeling in Security Games." AAAI-13.
//! - McKelvey & Palfrey (1995). "Quantal Response Equilibria for Normal
//!   Form Games." Games & Economic Behavior 10(1), 6-38.

use serde::{Deserialize, Serialize};

// ─── Lambda Curriculum ───────────────────────────────────────

/// Annealing schedule for the rationality parameter λ.
///
/// Controls how adversary rationality increases over the course of
/// a wargame, from near-random (low λ) to near-optimal (high λ).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnealSchedule {
    /// Linear interpolation over `steps` iterations.
    Linear { steps: usize },
    /// Cosine annealing over `steps` iterations.
    Cosine { steps: usize },
    /// Multiply λ by `factor` every `every` steps.
    StepDecay { factor: f64, every: usize },
}

/// Lambda curriculum: controls λ annealing from weak to strong adversary.
///
/// Default range: 0.25 (near-random) to 4.6 (observed IRA-level rationality,
/// calibrated from Tambe PROTECT deployment data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaCurriculum {
    /// Starting λ (typically 0.25 for near-random).
    pub start_lambda: f64,
    /// Ending λ (typically 2.0-4.6 for observed adversary range).
    pub end_lambda: f64,
    /// Annealing schedule type.
    pub schedule: AnnealSchedule,
    /// Current step in the curriculum.
    pub current_step: usize,
}

impl Default for LambdaCurriculum {
    fn default() -> Self {
        Self {
            start_lambda: 0.25,
            end_lambda: 4.6,
            schedule: AnnealSchedule::Linear { steps: 20 },
            current_step: 0,
        }
    }
}

impl LambdaCurriculum {
    /// Get the current λ value based on the annealing schedule.
    pub fn current_lambda(&self) -> f64 {
        let t = self.current_step;
        match &self.schedule {
            AnnealSchedule::Linear { steps } => {
                if *steps == 0 {
                    return self.end_lambda;
                }
                let frac = (t as f64 / *steps as f64).min(1.0);
                self.start_lambda + frac * (self.end_lambda - self.start_lambda)
            }
            AnnealSchedule::Cosine { steps } => {
                if *steps == 0 {
                    return self.end_lambda;
                }
                let frac = (t as f64 / *steps as f64).min(1.0);
                let cos_val = (1.0 - (std::f64::consts::PI * frac).cos()) / 2.0;
                self.start_lambda + cos_val * (self.end_lambda - self.start_lambda)
            }
            AnnealSchedule::StepDecay { factor, every } => {
                if *every == 0 {
                    return self.start_lambda;
                }
                // Step decay: lambda grows by factor at each step boundary
                let n_decays = t / every;
                let lambda = self.start_lambda * factor.powi(n_decays as i32);
                lambda.min(self.end_lambda)
            }
        }
    }

    /// Advance the curriculum by one step.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Reset the curriculum to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ─── SUQR Model ──────────────────────────────────────────────

/// Subjective Utility Quantal Response model.
///
/// `P(action_i) = exp(λ × Σ_k w_k × f_k(action_i)) / Z`
///
/// where `w_k` are subjective weights on payoff features (not raw EU),
/// and λ is the rationality parameter (capped to prevent superhuman play).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveUtilityQR {
    /// Rationality parameter (higher = more rational).
    pub lambda: f64,
    /// Subjective weights per payoff feature.
    pub feature_weights: Vec<f64>,
    /// Maximum λ to prevent superhuman adversaries (default 4.6).
    pub lambda_cap: f64,
}

impl Default for SubjectiveUtilityQR {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            feature_weights: vec![1.0], // uniform weight = standard QRE
            lambda_cap: 4.6,
        }
    }
}

/// Input for SUQR: a player's actions with feature vectors.
#[derive(Debug, Clone)]
pub struct SuqrPlayer {
    /// Entity UUID.
    pub entity_id: uuid::Uuid,
    /// Available actions.
    pub actions: Vec<String>,
    /// Feature vectors per action: `features[action_idx][feature_idx]`.
    pub features: Vec<Vec<f64>>,
}

/// Output: per-action probabilities from SUQR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuqrStrategy {
    /// Entity UUID.
    pub entity_id: uuid::Uuid,
    /// Action name.
    pub action: String,
    /// Probability of selecting this action.
    pub probability: f64,
    /// Subjective utility for this action.
    pub subjective_utility: f64,
}

impl SubjectiveUtilityQR {
    /// Create with specific lambda and weights.
    pub fn new(lambda: f64, feature_weights: Vec<f64>, lambda_cap: f64) -> Self {
        Self {
            lambda: lambda.min(lambda_cap),
            feature_weights,
            lambda_cap,
        }
    }

    /// Compute the effective (capped) lambda.
    pub fn effective_lambda(&self) -> f64 {
        self.lambda.min(self.lambda_cap)
    }

    /// Compute subjective utility for an action given its feature vector.
    ///
    /// `SU(action) = Σ_k w_k × f_k(action)`
    pub fn subjective_utility(&self, features: &[f64]) -> f64 {
        self.feature_weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// Solve SUQR for a single player: returns action probabilities.
    ///
    /// `P(a_i) = exp(λ_eff × SU(a_i)) / Σ_j exp(λ_eff × SU(a_j))`
    pub fn solve_single(&self, player: &SuqrPlayer) -> Vec<SuqrStrategy> {
        if player.actions.is_empty() {
            return vec![];
        }

        let lambda_eff = self.effective_lambda();

        // Compute subjective utilities
        let sus: Vec<f64> = player
            .features
            .iter()
            .map(|feat| self.subjective_utility(feat))
            .collect();

        // Softmax with numerical stability
        let max_su = sus.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = sus
            .iter()
            .map(|su| (lambda_eff * (su - max_su)).exp())
            .collect();
        let sum_exp: f64 = exp_vals.iter().sum();

        player
            .actions
            .iter()
            .enumerate()
            .map(|(i, action)| {
                let prob = if sum_exp > 0.0 {
                    exp_vals[i] / sum_exp
                } else {
                    1.0 / player.actions.len() as f64
                };
                SuqrStrategy {
                    entity_id: player.entity_id,
                    action: action.clone(),
                    probability: prob,
                    subjective_utility: sus[i],
                }
            })
            .collect()
    }

    /// Solve SUQR for multiple interacting players.
    ///
    /// Features are static in SUQR (subjective utilities don't depend on
    /// opponents' strategies), so probabilities are computed once per player.
    /// The `max_iterations` and `convergence_eps` parameters are retained for
    /// future extensions where features may depend on opponent distributions.
    pub fn solve_multi(
        &self,
        players: &[SuqrPlayer],
        _max_iterations: usize,
        _convergence_eps: f64,
    ) -> Vec<Vec<SuqrStrategy>> {
        if players.is_empty() {
            return vec![];
        }

        // Features are static in SUQR — each player's probabilities are
        // independent of opponents, so we compute once (no iteration needed).
        players
            .iter()
            .map(|player| self.solve_single(player))
            .collect()
    }

    /// Check if this SUQR instance is equivalent to standard QRE
    /// (all weights are equal).
    pub fn is_standard_qre(&self) -> bool {
        if self.feature_weights.is_empty() {
            return true;
        }
        let first = self.feature_weights[0];
        self.feature_weights
            .iter()
            .all(|w| (*w - first).abs() < 1e-10)
    }
}

// ─── Rationality Model Selection ─────────────────────────────

/// Which rationality model to use for game-theoretic analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RationalityModel {
    /// Standard Quantal Response Equilibrium.
    Qre,
    /// Subjective Utility Quantal Response.
    Suqr,
    /// Cognitive Hierarchy (Camerer, Ho & Chong 2004).
    CognitiveHierarchy,
}

impl Default for RationalityModel {
    fn default() -> Self {
        Self::Qre
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_player(actions: &[&str], features: Vec<Vec<f64>>) -> SuqrPlayer {
        SuqrPlayer {
            entity_id: Uuid::now_v7(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
            features,
        }
    }

    #[test]
    fn test_suqr_converges_to_qre_with_uniform_weights() {
        // When all feature weights are 1.0 and each action has a single
        // feature equal to its raw utility, SUQR should produce the same
        // probabilities as standard QRE softmax.
        let suqr = SubjectiveUtilityQR::new(2.0, vec![1.0], 10.0);
        assert!(suqr.is_standard_qre());

        let player = make_player(&["cooperate", "defect"], vec![vec![3.0], vec![5.0]]);

        let strategies = suqr.solve_single(&player);
        assert_eq!(strategies.len(), 2);

        // Defect (utility 5) should have higher probability than cooperate (utility 3)
        let p_coop = strategies[0].probability;
        let p_defect = strategies[1].probability;
        assert!(
            p_defect > p_coop,
            "defect should be more likely: {} vs {}",
            p_defect,
            p_coop
        );

        // Probabilities sum to 1
        assert!((p_coop + p_defect - 1.0).abs() < 1e-10);

        // Verify matches manual QRE calculation
        let exp_coop = (2.0 * 3.0_f64).exp();
        let exp_defect = (2.0 * 5.0_f64).exp();
        let z = exp_coop + exp_defect;
        let expected_p_coop = exp_coop / z;
        assert!(
            (p_coop - expected_p_coop).abs() < 1e-6,
            "SUQR should match QRE: {} vs {}",
            p_coop,
            expected_p_coop
        );
    }

    #[test]
    fn test_lambda_cap_prevents_superhuman_play() {
        let suqr = SubjectiveUtilityQR::new(100.0, vec![1.0], 4.6);
        assert_eq!(suqr.effective_lambda(), 4.6);

        let player = make_player(&["good", "bad"], vec![vec![10.0], vec![1.0]]);

        let strategies = suqr.solve_single(&player);
        let p_good = strategies[0].probability;
        let p_bad = strategies[1].probability;

        // Even with very high utility difference, lambda cap prevents
        // probability from being exactly 1.0
        assert!(p_good > 0.99, "good should dominate: {}", p_good);
        assert!(
            p_bad > 0.0,
            "bad should still have some probability: {}",
            p_bad
        );
        assert!(p_bad < 0.01);
    }

    #[test]
    fn test_lambda_curriculum_linear() {
        let mut curriculum = LambdaCurriculum {
            start_lambda: 0.25,
            end_lambda: 4.0,
            schedule: AnnealSchedule::Linear { steps: 10 },
            current_step: 0,
        };

        assert!((curriculum.current_lambda() - 0.25).abs() < 1e-10);

        // Step to midpoint
        curriculum.current_step = 5;
        let mid = curriculum.current_lambda();
        assert!(
            (mid - 2.125).abs() < 1e-10,
            "midpoint should be 2.125, got {}",
            mid
        );

        // Step to end
        curriculum.current_step = 10;
        assert!((curriculum.current_lambda() - 4.0).abs() < 1e-10);

        // Beyond end should clamp
        curriculum.current_step = 20;
        assert!((curriculum.current_lambda() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_lambda_curriculum_cosine() {
        let curriculum = LambdaCurriculum {
            start_lambda: 0.0,
            end_lambda: 4.0,
            schedule: AnnealSchedule::Cosine { steps: 10 },
            current_step: 5,
        };

        // At midpoint of cosine, should be exactly half range
        let mid = curriculum.current_lambda();
        assert!(
            (mid - 2.0).abs() < 1e-10,
            "cosine midpoint should be 2.0, got {}",
            mid
        );
    }

    #[test]
    fn test_lambda_curriculum_step_decay() {
        let curriculum = LambdaCurriculum {
            start_lambda: 1.0,
            end_lambda: 100.0,
            schedule: AnnealSchedule::StepDecay {
                factor: 2.0,
                every: 5,
            },
            current_step: 10,
        };

        // After 10 steps with factor 2.0 every 5: 1.0 * 2^2 = 4.0
        assert!(
            (curriculum.current_lambda() - 4.0).abs() < 1e-10,
            "step decay should give 4.0, got {}",
            curriculum.current_lambda()
        );
    }

    #[test]
    fn test_curriculum_step_and_reset() {
        let mut curriculum = LambdaCurriculum::default();
        assert_eq!(curriculum.current_step, 0);

        curriculum.step();
        curriculum.step();
        assert_eq!(curriculum.current_step, 2);

        curriculum.reset();
        assert_eq!(curriculum.current_step, 0);
    }

    #[test]
    fn test_suqr_multi_feature_weights() {
        // Test with multiple features and non-uniform weights.
        // Feature 0: raw payoff (weight 0.3)
        // Feature 1: moral outrage (weight 0.7)
        let suqr = SubjectiveUtilityQR::new(1.0, vec![0.3, 0.7], 10.0);

        assert!(!suqr.is_standard_qre());

        // Action A: payoff=5, outrage=1 -> SU = 0.3*5 + 0.7*1 = 2.2
        // Action B: payoff=2, outrage=8 -> SU = 0.3*2 + 0.7*8 = 6.2
        let player = make_player(
            &["rational_choice", "outrage_choice"],
            vec![vec![5.0, 1.0], vec![2.0, 8.0]],
        );

        let strategies = suqr.solve_single(&player);
        assert_eq!(strategies.len(), 2);

        // Outrage choice has higher SU despite lower raw payoff
        assert!(
            strategies[1].probability > strategies[0].probability,
            "outrage-weighted action should dominate: {} vs {}",
            strategies[1].probability,
            strategies[0].probability
        );

        assert!(
            (strategies[0].subjective_utility - 2.2).abs() < 1e-10,
            "SU should be 2.2, got {}",
            strategies[0].subjective_utility
        );
        assert!(
            (strategies[1].subjective_utility - 6.2).abs() < 1e-10,
            "SU should be 6.2, got {}",
            strategies[1].subjective_utility
        );
    }

    #[test]
    fn test_suqr_multi_player() {
        let suqr = SubjectiveUtilityQR::new(1.0, vec![1.0], 10.0);

        let players = vec![
            make_player(&["attack", "wait"], vec![vec![3.0], vec![1.0]]),
            make_player(&["defend", "ignore"], vec![vec![2.0], vec![4.0]]),
        ];

        let result = suqr.solve_multi(&players, 100, 1e-6);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].len(), 2);

        // Each player's probabilities should sum to 1
        let sum_0: f64 = result[0].iter().map(|s| s.probability).sum();
        let sum_1: f64 = result[1].iter().map(|s| s.probability).sum();
        assert!((sum_0 - 1.0).abs() < 1e-10);
        assert!((sum_1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_suqr_empty_player() {
        let suqr = SubjectiveUtilityQR::default();
        let player = SuqrPlayer {
            entity_id: Uuid::now_v7(),
            actions: vec![],
            features: vec![],
        };
        let result = suqr.solve_single(&player);
        assert!(result.is_empty());
    }

    #[test]
    fn test_low_lambda_produces_near_uniform() {
        let suqr = SubjectiveUtilityQR::new(0.01, vec![1.0], 10.0);

        let player = make_player(&["a", "b", "c"], vec![vec![10.0], vec![1.0], vec![5.0]]);

        let strategies = suqr.solve_single(&player);

        // With very low lambda, all probabilities should be near 1/3
        for s in &strategies {
            assert!(
                (s.probability - 1.0 / 3.0).abs() < 0.05,
                "low lambda should give near-uniform: {}",
                s.probability
            );
        }
    }

    #[test]
    fn test_high_lambda_produces_near_deterministic() {
        let suqr = SubjectiveUtilityQR::new(4.5, vec![1.0], 4.6);

        let player = make_player(&["best", "worst"], vec![vec![10.0], vec![0.0]]);

        let strategies = suqr.solve_single(&player);

        assert!(
            strategies[0].probability > 0.99,
            "high lambda should make best action near-certain: {}",
            strategies[0].probability
        );
    }
}
