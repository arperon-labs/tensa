//! Cognitive Hierarchy (CH) bounded rationality model.
//!
//! Implements Camerer, Ho & Chong (2004) Poisson CH model as an alternative
//! to QRE for first-move scenarios where the adversary hasn't been observed.
//!
//! ## Model
//!
//! - Level-0: uniform random over actions
//! - Level-k: best-responds to mixture of levels 0..k-1
//! - Population follows Poisson(τ) with τ ≈ 1.5
//!
//! ## Reference
//!
//! Camerer, Ho & Chong (2004). "A Cognitive Hierarchy Model of Games."
//! Quarterly Journal of Economics 119(3), 861-898.

use serde::{Deserialize, Serialize};

/// Cognitive Hierarchy model parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveHierarchy {
    /// Mean thinking steps (Poisson parameter). Default 1.5.
    pub tau: f64,
    /// Maximum level of reasoning to consider. Default 5.
    pub max_level: usize,
}

impl Default for CognitiveHierarchy {
    fn default() -> Self {
        Self {
            tau: 1.5,
            max_level: 5,
        }
    }
}

/// Input: a player's available actions with payoff vectors.
#[derive(Debug, Clone)]
pub struct ChPlayer {
    /// Entity UUID.
    pub entity_id: uuid::Uuid,
    /// Available actions.
    pub actions: Vec<String>,
    /// Payoff for each action against each opponent action:
    /// `payoffs[my_action][opponent_action]`.
    /// For single-player or non-interactive: `payoffs[my_action][0]`.
    pub payoffs: Vec<Vec<f64>>,
}

/// Output: per-action probability from the CH model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChStrategy {
    pub entity_id: uuid::Uuid,
    pub action: String,
    pub probability: f64,
    /// The dominant thinking level that chose this action.
    pub dominant_level: usize,
}

impl CognitiveHierarchy {
    /// Compute Poisson probability for level k: P(k) = e^{-τ} × τ^k / k!
    fn poisson_prob(&self, k: usize) -> f64 {
        let log_p = -(self.tau) + (k as f64) * self.tau.ln() - log_factorial(k);
        log_p.exp()
    }

    /// Compute the normalized level weights for levels 0..max_level.
    fn level_weights(&self) -> Vec<f64> {
        let raw: Vec<f64> = (0..=self.max_level).map(|k| self.poisson_prob(k)).collect();
        let sum: f64 = raw.iter().sum();
        if sum > 0.0 {
            raw.iter().map(|p| p / sum).collect()
        } else {
            let n = raw.len();
            vec![1.0 / n as f64; n]
        }
    }

    /// Solve for a single player's CH strategy distribution.
    ///
    /// Returns the weighted mixture across all thinking levels.
    pub fn solve(&self, player: &ChPlayer) -> Vec<ChStrategy> {
        let n_actions = player.actions.len();
        if n_actions == 0 {
            return vec![];
        }

        let level_weights = self.level_weights();

        // Level-0: uniform random
        let mut level_strategies: Vec<Vec<f64>> = Vec::with_capacity(self.max_level + 1);
        level_strategies.push(vec![1.0 / n_actions as f64; n_actions]);

        // Level-k: best-responds to the weighted mixture of levels 0..k-1
        for k in 1..=self.max_level {
            // Build the belief about opponents: weighted average of lower levels
            let mut belief = vec![0.0_f64; n_actions];
            let mut belief_weight_sum = 0.0_f64;

            for j in 0..k {
                let w = level_weights[j];
                for a in 0..n_actions {
                    belief[a] += w * level_strategies[j][a];
                }
                belief_weight_sum += w;
            }

            // Normalize belief
            if belief_weight_sum > 0.0 {
                for b in &mut belief {
                    *b /= belief_weight_sum;
                }
            }

            // Compute expected payoff for each action against this belief
            let expected_payoffs: Vec<f64> = (0..n_actions)
                .map(|my_a| {
                    if player.payoffs[my_a].len() == 1 {
                        // Non-interactive: payoff doesn't depend on opponent
                        player.payoffs[my_a][0]
                    } else {
                        // Interactive: expected payoff = Σ_j belief[j] * payoff[my_a][j]
                        belief
                            .iter()
                            .enumerate()
                            .map(|(opp_a, &b)| {
                                if opp_a < player.payoffs[my_a].len() {
                                    b * player.payoffs[my_a][opp_a]
                                } else {
                                    0.0
                                }
                            })
                            .sum()
                    }
                })
                .collect();

            // Best-respond: put all probability on the max-payoff action(s)
            let max_payoff = expected_payoffs
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let best_actions: Vec<usize> = expected_payoffs
                .iter()
                .enumerate()
                .filter(|(_, &p)| (p - max_payoff).abs() < 1e-10)
                .map(|(i, _)| i)
                .collect();

            let mut level_k_strategy = vec![0.0_f64; n_actions];
            let share = 1.0 / best_actions.len() as f64;
            for &a in &best_actions {
                level_k_strategy[a] = share;
            }
            level_strategies.push(level_k_strategy);
        }

        // Final strategy: weighted mixture across all levels
        let mut final_probs = vec![0.0_f64; n_actions];
        for (k, strat) in level_strategies.iter().enumerate() {
            let w = level_weights[k];
            for (a, &p) in strat.iter().enumerate() {
                final_probs[a] += w * p;
            }
        }

        // Find dominant level for each action (highest weighted contributor)
        let dominant_levels: Vec<usize> = (0..n_actions)
            .map(|a| {
                (0..=self.max_level)
                    .max_by(|&k1, &k2| {
                        let c1 = level_weights[k1] * level_strategies[k1][a];
                        let c2 = level_weights[k2] * level_strategies[k2][a];
                        c1.partial_cmp(&c2).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect();

        player
            .actions
            .iter()
            .enumerate()
            .map(|(i, action)| ChStrategy {
                entity_id: player.entity_id,
                action: action.clone(),
                probability: final_probs[i],
                dominant_level: dominant_levels[i],
            })
            .collect()
    }
}

/// Log-factorial using Stirling's approximation for large n,
/// exact for small n.
fn log_factorial(n: usize) -> f64 {
    if n <= 20 {
        // Exact for small values
        (1..=n).map(|i| (i as f64).ln()).sum()
    } else {
        // Stirling's approximation
        let n_f = n as f64;
        n_f * n_f.ln() - n_f + 0.5 * (2.0 * std::f64::consts::PI * n_f).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_player(actions: &[&str], payoffs: Vec<Vec<f64>>) -> ChPlayer {
        ChPlayer {
            entity_id: Uuid::now_v7(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
            payoffs,
        }
    }

    #[test]
    fn test_level_0_is_uniform() {
        let ch = CognitiveHierarchy {
            tau: 1.5,
            max_level: 0,
        };

        let player = make_player(&["a", "b", "c"], vec![vec![10.0], vec![1.0], vec![5.0]]);

        let strategies = ch.solve(&player);
        assert_eq!(strategies.len(), 3);

        // With max_level=0, only level-0 (uniform) contributes
        for s in &strategies {
            assert!(
                (s.probability - 1.0 / 3.0).abs() < 1e-10,
                "level-0 only should be uniform: {}",
                s.probability
            );
        }
    }

    #[test]
    fn test_high_tau_converges_toward_best_response() {
        // With very high tau, almost all weight is on high-level thinkers
        // who best-respond, so the result approaches a pure best response.
        let ch = CognitiveHierarchy {
            tau: 50.0,
            max_level: 10,
        };

        let player = make_player(&["best", "worst"], vec![vec![10.0], vec![1.0]]);

        let strategies = ch.solve(&player);

        // "best" should dominate
        assert!(
            strategies[0].probability > 0.95,
            "high tau should converge toward best response: {}",
            strategies[0].probability
        );
    }

    #[test]
    fn test_poisson_weights_sum_to_one() {
        let ch = CognitiveHierarchy::default();
        let weights = ch.level_weights();
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "level weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_interactive_payoffs() {
        let ch = CognitiveHierarchy {
            tau: 1.5,
            max_level: 5,
        };

        // Prisoner's dilemma payoff matrix:
        // (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
        // Player's payoffs: cooperate=[3, 0], defect=[5, 1]
        let player = make_player(
            &["cooperate", "defect"],
            vec![vec![3.0, 0.0], vec![5.0, 1.0]],
        );

        let strategies = ch.solve(&player);
        assert_eq!(strategies.len(), 2);

        // Defect should have higher probability (dominant strategy in PD)
        assert!(
            strategies[1].probability > strategies[0].probability,
            "defect should dominate: {} vs {}",
            strategies[1].probability,
            strategies[0].probability
        );
    }

    #[test]
    fn test_empty_player() {
        let ch = CognitiveHierarchy::default();
        let player = ChPlayer {
            entity_id: Uuid::now_v7(),
            actions: vec![],
            payoffs: vec![],
        };
        assert!(ch.solve(&player).is_empty());
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let ch = CognitiveHierarchy::default();

        let player = make_player(
            &["a", "b", "c", "d"],
            vec![vec![3.0], vec![7.0], vec![1.0], vec![5.0]],
        );

        let strategies = ch.solve(&player);
        let sum: f64 = strategies.iter().map(|s| s.probability).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1.0, got {}",
            sum
        );
    }
}
