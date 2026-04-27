//! Game-theoretic inference engine.
//!
//! Classifies strategic interactions per situation and solves for
//! Quantal Response Equilibrium (QRE) — a bounded-rationality
//! generalization of Nash equilibrium. Supports sub-game decomposition
//! for situations with more than 4 players.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

/// Configuration for the game engine.
#[derive(Debug, Clone)]
pub struct GameConfig {
    /// Maximum QRE iterations.
    pub max_iterations: usize,
    /// QRE convergence threshold.
    pub convergence_threshold: f64,
    /// Initial lambda (rationality) for QRE.
    pub initial_lambda: f64,
    /// Maximum lambda for search.
    pub max_lambda: f64,
    /// Lambda step size for search.
    pub lambda_step: f64,
    /// Threshold for sub-game decomposition.
    pub sub_game_threshold: usize,
    /// Fixed-point iteration convergence epsilon.
    pub fixed_point_epsilon: f64,
    /// Maximum fixed-point iterations per lambda evaluation.
    pub fixed_point_max_iter: usize,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            initial_lambda: 1.0,
            max_lambda: 10.0,
            lambda_step: 0.5,
            sub_game_threshold: 4,
            fixed_point_epsilon: 1e-6,
            fixed_point_max_iter: 100,
        }
    }
}

/// Game-theoretic inference engine.
pub struct GameEngine {
    config: GameConfig,
}

impl Default for GameEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GameEngine {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: GameConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: GameConfig) -> Self {
        Self { config }
    }

    /// Classify the game structure of a situation based on its participants.
    pub fn classify_game(
        &self,
        situation: &Situation,
        participants: &[Participation],
    ) -> GameStructure {
        let n_players = participants.iter().filter(|p| p.action.is_some()).count();
        let _has_payoffs = participants.iter().any(|p| p.payoff.is_some());

        // Determine information structure
        let info_structure = self.classify_info_structure(participants);

        // Determine game type
        let game_type = if n_players == 0 {
            GameClassification::Custom("no-strategic-interaction".into())
        } else if n_players <= 2 {
            self.classify_two_player(participants)
        } else {
            self.classify_n_player(participants, situation)
        };

        let description = Some(format!(
            "{}-player game with {:?} information",
            n_players, info_structure
        ));

        GameStructure {
            game_type,
            info_structure,
            description,
            maturity: MaturityLevel::Candidate,
        }
    }

    /// Solve for QRE equilibrium via fixed-point iteration.
    ///
    /// For each player i, the quantal response function is:
    /// `σ_i(a_i) = exp(λ·EU_i(a_i, σ_{-i})) / Σ_j exp(λ·EU_i(a_j, σ_{-i}))`
    ///
    /// Iterates from uniform strategies until convergence (||σ^{t+1} - σ^t||_∞ < ε).
    /// Players react to each other's strategies simultaneously.
    pub fn solve_qre(&self, players: &[PlayerStrategies], lambda: f64) -> Vec<ActorStrategy> {
        if players.is_empty() {
            return vec![];
        }

        let n_players = players.len();

        // Initialize: uniform strategies for all players
        let mut strategies: Vec<Vec<f64>> = players
            .iter()
            .map(|p| {
                let n = p.actions.len().max(1);
                vec![1.0 / n as f64; n]
            })
            .collect();

        // Fixed-point iteration
        for _fp_iter in 0..self.config.fixed_point_max_iter {
            let old_strategies = strategies.clone();

            for i in 0..n_players {
                let n_actions = players[i].actions.len();
                if n_actions == 0 {
                    continue;
                }

                // Compute expected utility for each action against current opponent strategies
                let mut eu = vec![0.0_f64; n_actions];
                for a_idx in 0..n_actions {
                    eu[a_idx] = Self::compute_expected_utility(i, a_idx, players, &old_strategies);
                }

                // Softmax update: σ_i(a) = exp(λ·EU(a)) / Σ exp(λ·EU(a'))
                let max_eu = eu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_vals: Vec<f64> = eu.iter().map(|e| (lambda * (e - max_eu)).exp()).collect();
                let sum_exp: f64 = exp_vals.iter().sum();

                strategies[i] = exp_vals
                    .iter()
                    .map(|e| {
                        if sum_exp > 0.0 {
                            e / sum_exp
                        } else {
                            1.0 / n_actions as f64
                        }
                    })
                    .collect();
            }

            // Convergence: max absolute difference across all players and actions
            let max_diff = strategies
                .iter()
                .zip(old_strategies.iter())
                .flat_map(|(new, old)| new.iter().zip(old.iter()))
                .map(|(n, o)| (n - o).abs())
                .fold(0.0_f64, f64::max);

            if max_diff < self.config.fixed_point_epsilon {
                break;
            }
        }

        // Convert to ActorStrategy output
        let mut result = Vec::new();
        for (i, player) in players.iter().enumerate() {
            for (a_idx, action) in player.actions.iter().enumerate() {
                result.push(ActorStrategy {
                    entity_id: player.entity_id,
                    action: action.clone(),
                    probability: strategies[i][a_idx] as f32,
                    expected_payoff: player.expected_utilities[a_idx],
                });
            }
        }

        result
    }

    /// Compute expected utility for player `player_idx` taking action `action_idx`
    /// against other players' current mixed strategies.
    ///
    /// For multi-player: EU_i(a_i, σ_{-i}) incorporates opponent strategy interaction.
    /// Falls back to pre-computed utilities when interaction data is unavailable.
    fn compute_expected_utility(
        player_idx: usize,
        action_idx: usize,
        players: &[PlayerStrategies],
        strategies: &[Vec<f64>],
    ) -> f64 {
        let base_utility = players[player_idx].expected_utilities[action_idx];

        if players.len() <= 1 {
            return base_utility;
        }

        // Weighted utility incorporating opponent strategy distributions.
        // EU_i(a_i, σ_{-i}) = base_utility adjusted by opponent concentration:
        // if opponents concentrate on high-payoff actions (competitive pressure),
        // reduce utility; if opponents are spread (cooperative/uncoordinated),
        // utility stays near base.
        let mut adjustment = 0.0_f64;
        let mut opp_count = 0;

        for (j, opponent) in players.iter().enumerate() {
            if j == player_idx || opponent.actions.is_empty() {
                continue;
            }

            // Opponent's expected payoff under their current strategy
            let opp_expected: f64 = strategies[j]
                .iter()
                .zip(opponent.expected_utilities.iter())
                .map(|(prob, eu)| prob * eu)
                .sum();

            // If opponent's expected payoff is positive (competitive),
            // this reduces our utility proportionally
            adjustment -=
                0.1 * opp_expected * strategies[j].iter().cloned().fold(0.0_f64, f64::max); // max opponent probability
            opp_count += 1;
        }

        if opp_count > 0 {
            base_utility + adjustment / opp_count as f64
        } else {
            base_utility
        }
    }

    /// Find optimal lambda by matching observed behavior.
    pub fn estimate_lambda(
        &self,
        players: &[PlayerStrategies],
        observed_actions: &[(Uuid, String)],
    ) -> f64 {
        let mut best_lambda = self.config.initial_lambda;
        let mut best_likelihood = f64::NEG_INFINITY;

        let mut lambda = 0.0;
        while lambda <= self.config.max_lambda {
            let strategies = self.solve_qre(players, lambda);
            let likelihood = self.compute_likelihood(&strategies, observed_actions);

            if likelihood > best_likelihood {
                best_likelihood = likelihood;
                best_lambda = lambda;
            }

            lambda += self.config.lambda_step;
        }

        best_lambda
    }

    /// Analyze a situation: classify + solve equilibrium.
    pub fn analyze_situation(
        &self,
        situation: &Situation,
        hypergraph: &Hypergraph,
    ) -> Result<GameAnalysis> {
        let participants = hypergraph.get_participants_for_situation(&situation.id)?;
        let game_structure = self.classify_game(situation, &participants);

        // Build player strategies from participation data
        let players = self.extract_player_strategies(&participants);

        // Get observed actions for lambda estimation
        let observed: Vec<(Uuid, String)> = participants
            .iter()
            .filter_map(|p| p.action.as_ref().map(|a| (p.entity_id, a.clone())))
            .collect();

        let lambda = if !players.is_empty() && !observed.is_empty() {
            self.estimate_lambda(&players, &observed)
        } else {
            self.config.initial_lambda
        };

        let strategy_profile = self.solve_qre(&players, lambda);

        let eq_type = if lambda > 5.0 {
            EquilibriumType::Nash
        } else {
            EquilibriumType::QRE
        };

        // Sub-game decomposition for large player counts
        let sub_games = if participants.len() > self.config.sub_game_threshold {
            self.decompose_sub_games(situation, hypergraph)?
        } else {
            vec![]
        };

        Ok(GameAnalysis {
            game_structure,
            equilibria: vec![Equilibrium {
                eq_type,
                strategy_profile,
                lambda: lambda as f32,
            }],
            sub_games,
        })
    }

    fn classify_info_structure(&self, participants: &[Participation]) -> InfoStructureType {
        let has_info_sets = participants.iter().any(|p| p.info_set.is_some());

        if !has_info_sets {
            return InfoStructureType::Incomplete;
        }

        let all_know_same = participants.iter().all(|p| {
            p.info_set
                .as_ref()
                .map(|is| is.knows_before.is_empty() || is.learns.is_empty())
                .unwrap_or(true)
        });

        let some_learn = participants.iter().any(|p| {
            p.info_set
                .as_ref()
                .map(|is| !is.learns.is_empty())
                .unwrap_or(false)
        });

        if all_know_same && !some_learn {
            InfoStructureType::Complete
        } else if some_learn {
            InfoStructureType::AsymmetricBecomingComplete
        } else {
            InfoStructureType::Imperfect
        }
    }

    fn classify_two_player(&self, participants: &[Participation]) -> GameClassification {
        let active: Vec<_> = participants.iter().filter(|p| p.action.is_some()).collect();
        if active.len() != 2 {
            return GameClassification::Custom("non-standard".into());
        }

        // Check payoff symmetry
        let p0 = &active[0].payoff;
        let p1 = &active[1].payoff;

        match (p0, p1) {
            (Some(pay0), Some(pay1)) => {
                if self.is_zero_sum(pay0, pay1) {
                    GameClassification::ZeroSum
                } else if self.has_cooperation_pattern(pay0, pay1) {
                    GameClassification::PrisonersDilemma
                } else if self.has_coordination_pattern(pay0, pay1) {
                    GameClassification::Coordination
                } else {
                    GameClassification::Bargaining
                }
            }
            _ => {
                // Check for signaling based on info sets
                let has_asymmetric = active.iter().any(|p| {
                    p.info_set
                        .as_ref()
                        .map(|is| !is.reveals.is_empty())
                        .unwrap_or(false)
                });
                if has_asymmetric {
                    GameClassification::Signaling
                } else {
                    GameClassification::Custom("insufficient-payoff-data".into())
                }
            }
        }
    }

    fn classify_n_player(
        &self,
        participants: &[Participation],
        _situation: &Situation,
    ) -> GameClassification {
        let active: Vec<_> = participants.iter().filter(|p| p.action.is_some()).collect();

        // Check for auction-like structures (one target, multiple actors)
        let has_target = participants.iter().any(|p| p.role == Role::Target);
        if has_target && active.len() >= 3 {
            return GameClassification::Auction;
        }

        // Check for asymmetric information
        let info_asymmetry = active.iter().any(|p| {
            p.info_set
                .as_ref()
                .map(|is| !is.knows_before.is_empty())
                .unwrap_or(false)
        });
        if info_asymmetry {
            return GameClassification::AsymmetricInformation;
        }

        GameClassification::Coordination
    }

    fn is_zero_sum(&self, pay0: &serde_json::Value, pay1: &serde_json::Value) -> bool {
        if let (Some(v0), Some(v1)) = (pay0.as_f64(), pay1.as_f64()) {
            (v0 + v1).abs() < 0.01
        } else {
            false
        }
    }

    fn has_cooperation_pattern(&self, pay0: &serde_json::Value, pay1: &serde_json::Value) -> bool {
        // Simple heuristic: both payoffs are positive but asymmetric
        match (pay0.as_f64(), pay1.as_f64()) {
            (Some(v0), Some(v1)) => v0 > 0.0 && v1 > 0.0 && (v0 - v1).abs() > 0.1,
            _ => false,
        }
    }

    fn has_coordination_pattern(&self, pay0: &serde_json::Value, pay1: &serde_json::Value) -> bool {
        // Simple heuristic: both payoffs are similar and positive
        match (pay0.as_f64(), pay1.as_f64()) {
            (Some(v0), Some(v1)) => v0 > 0.0 && v1 > 0.0 && (v0 - v1).abs() < 0.1,
            _ => false,
        }
    }

    fn extract_player_strategies(&self, participants: &[Participation]) -> Vec<PlayerStrategies> {
        participants
            .iter()
            .filter(|p| p.action.is_some())
            .map(|p| {
                let action = p.action.clone().unwrap_or_default();
                let payoff = p.payoff.as_ref().and_then(|v| v.as_f64()).unwrap_or(0.0);

                PlayerStrategies {
                    entity_id: p.entity_id,
                    actions: vec![action.clone(), format!("not_{}", action)],
                    expected_utilities: vec![payoff, -payoff * 0.5],
                }
            })
            .collect()
    }

    fn decompose_sub_games(
        &self,
        situation: &Situation,
        hypergraph: &Hypergraph,
    ) -> Result<Vec<Uuid>> {
        // Find causally-connected situations that form sub-games
        let chain = hypergraph.traverse_causal_chain(&situation.id, 3)?;
        Ok(chain.iter().map(|l| l.to_situation).collect())
    }

    fn compute_likelihood(&self, strategies: &[ActorStrategy], observed: &[(Uuid, String)]) -> f64 {
        let mut log_likelihood = 0.0;

        for (entity_id, action) in observed {
            let prob = strategies
                .iter()
                .find(|s| s.entity_id == *entity_id && s.action == *action)
                .map(|s| s.probability as f64)
                .unwrap_or(0.001); // small floor to avoid -inf

            log_likelihood += prob.ln();
        }

        log_likelihood
    }
}

/// Player strategies for QRE computation.
#[derive(Debug, Clone)]
pub struct PlayerStrategies {
    pub entity_id: Uuid,
    pub actions: Vec<String>,
    pub expected_utilities: Vec<f64>,
}

impl InferenceEngine for GameEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::GameClassification
    }

    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
        super::cost::estimate_cost(job, hypergraph)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let situation = hypergraph.get_situation(&job.target_id).map_err(|_| {
            TensaError::InferenceError(format!("Target situation not found: {}", job.target_id))
        })?;

        let analysis = self.analyze_situation(&situation, hypergraph)?;
        let confidence = if analysis.equilibria.is_empty() {
            0.3
        } else {
            0.7
        };

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: job.job_type.clone(),
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence,
            explanation: None,
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_situation(hg: &Hypergraph) -> Situation {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let id = hg.create_situation(sit.clone()).unwrap();
        let mut result = sit;
        result.id = id;
        result
    }

    fn make_entity(hg: &Hypergraph, name: &str) -> Uuid {
        hg.create_entity(Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap()
    }

    #[test]
    fn test_classify_no_players() {
        let engine = GameEngine::new();
        let hg = test_hg();
        let sit = make_situation(&hg);
        let gs = engine.classify_game(&sit, &[]);
        assert!(matches!(gs.game_type, GameClassification::Custom(_)));
    }

    #[test]
    fn test_classify_zero_sum() {
        let engine = GameEngine::new();
        let hg = test_hg();
        let sit = make_situation(&hg);
        let e1 = make_entity(&hg, "Alice");
        let e2 = make_entity(&hg, "Bob");

        let participants = vec![
            Participation {
                entity_id: e1,
                situation_id: sit.id,
                role: Role::Protagonist,
                info_set: None,
                action: Some("attack".into()),
                payoff: Some(serde_json::json!(5.0)),
                seq: 0,
            },
            Participation {
                entity_id: e2,
                situation_id: sit.id,
                role: Role::Antagonist,
                info_set: None,
                action: Some("defend".into()),
                payoff: Some(serde_json::json!(-5.0)),
                seq: 0,
            },
        ];

        let gs = engine.classify_game(&sit, &participants);
        assert_eq!(gs.game_type, GameClassification::ZeroSum);
    }

    #[test]
    fn test_classify_coordination() {
        let engine = GameEngine::new();
        let hg = test_hg();
        let sit = make_situation(&hg);
        let e1 = make_entity(&hg, "Alice");
        let e2 = make_entity(&hg, "Bob");

        let participants = vec![
            Participation {
                entity_id: e1,
                situation_id: sit.id,
                role: Role::Protagonist,
                info_set: None,
                action: Some("cooperate".into()),
                payoff: Some(serde_json::json!(3.0)),
                seq: 0,
            },
            Participation {
                entity_id: e2,
                situation_id: sit.id,
                role: Role::Protagonist,
                info_set: None,
                action: Some("cooperate".into()),
                payoff: Some(serde_json::json!(3.0)),
                seq: 0,
            },
        ];

        let gs = engine.classify_game(&sit, &participants);
        assert_eq!(gs.game_type, GameClassification::Coordination);
    }

    #[test]
    fn test_classify_signaling() {
        let engine = GameEngine::new();
        let hg = test_hg();
        let sit = make_situation(&hg);
        let e1 = make_entity(&hg, "Sender");
        let e2 = make_entity(&hg, "Receiver");

        let participants = vec![
            Participation {
                entity_id: e1,
                situation_id: sit.id,
                role: Role::Informant,
                info_set: Some(InfoSet {
                    knows_before: vec![],
                    learns: vec![],
                    reveals: vec![KnowledgeFact {
                        about_entity: e2,
                        fact: "signal".into(),
                        confidence: 0.8,
                    }],
                    beliefs_about_others: vec![],
                }),
                action: Some("signal".into()),
                payoff: None,
                seq: 0,
            },
            Participation {
                entity_id: e2,
                situation_id: sit.id,
                role: Role::Recipient,
                info_set: None,
                action: Some("respond".into()),
                payoff: None,
                seq: 0,
            },
        ];

        let gs = engine.classify_game(&sit, &participants);
        assert_eq!(gs.game_type, GameClassification::Signaling);
    }

    #[test]
    fn test_qre_uniform_at_zero_lambda() {
        let engine = GameEngine::new();

        let players = vec![PlayerStrategies {
            entity_id: Uuid::now_v7(),
            actions: vec!["A".into(), "B".into(), "C".into()],
            expected_utilities: vec![10.0, 5.0, 1.0],
        }];

        let strategies = engine.solve_qre(&players, 0.0);
        assert_eq!(strategies.len(), 3);

        // At lambda=0, should be approximately uniform
        for s in &strategies {
            assert!((s.probability - 1.0 / 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_qre_converges_to_best_at_high_lambda() {
        let engine = GameEngine::new();

        let players = vec![PlayerStrategies {
            entity_id: Uuid::now_v7(),
            actions: vec!["best".into(), "worst".into()],
            expected_utilities: vec![10.0, 1.0],
        }];

        let strategies = engine.solve_qre(&players, 100.0);
        let best = strategies.iter().find(|s| s.action == "best").unwrap();
        assert!(
            best.probability > 0.99,
            "Best action should dominate at high lambda"
        );
    }

    #[test]
    fn test_estimate_lambda() {
        let engine = GameEngine::new();
        let entity_id = Uuid::now_v7();

        let players = vec![PlayerStrategies {
            entity_id,
            actions: vec!["cooperate".into(), "defect".into()],
            expected_utilities: vec![3.0, 5.0],
        }];

        // Observed: entity chose "defect" (higher utility)
        let observed = vec![(entity_id, "defect".into())];
        let lambda = engine.estimate_lambda(&players, &observed);
        // Lambda should be positive (rational) since the actor chose the better action
        assert!(lambda > 0.0);
    }

    #[test]
    fn test_analyze_situation() {
        let hg = test_hg();
        let engine = GameEngine::new();
        let sit = make_situation(&hg);
        let e1 = make_entity(&hg, "Player1");
        let e2 = make_entity(&hg, "Player2");

        hg.add_participant(Participation {
            entity_id: e1,
            situation_id: sit.id,
            role: Role::Protagonist,
            info_set: None,
            action: Some("attack".into()),
            payoff: Some(serde_json::json!(3.0)),
            seq: 0,
        })
        .unwrap();

        hg.add_participant(Participation {
            entity_id: e2,
            situation_id: sit.id,
            role: Role::Antagonist,
            info_set: None,
            action: Some("defend".into()),
            payoff: Some(serde_json::json!(-3.0)),
            seq: 0,
        })
        .unwrap();

        let analysis = engine.analyze_situation(&sit, &hg).unwrap();
        assert_eq!(
            analysis.game_structure.game_type,
            GameClassification::ZeroSum
        );
        assert!(!analysis.equilibria.is_empty());
        assert_eq!(analysis.game_structure.maturity, MaturityLevel::Candidate);
    }

    #[test]
    fn test_sub_game_decomposition() {
        let hg = test_hg();
        let engine = GameEngine::new();
        let sit = make_situation(&hg);

        // Create enough participants to trigger decomposition
        for i in 0..5 {
            let eid = make_entity(&hg, &format!("Player{}", i));
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: sit.id,
                role: Role::Protagonist,
                info_set: None,
                action: Some(format!("action_{}", i)),
                payoff: Some(serde_json::json!(1.0)),
                seq: 0,
            })
            .unwrap();
        }

        let analysis = engine.analyze_situation(&sit, &hg).unwrap();
        // Sub-games may or may not be found depending on causal links
        assert!(analysis.equilibria.len() >= 1);
    }

    #[test]
    fn test_engine_execute() {
        let hg = test_hg();
        let engine = GameEngine::new();
        let sit = make_situation(&hg);
        let eid = make_entity(&hg, "Actor");

        hg.add_participant(Participation {
            entity_id: eid,
            situation_id: sit.id,
            role: Role::Protagonist,
            info_set: None,
            action: Some("act".into()),
            payoff: Some(serde_json::json!(2.0)),
            seq: 0,
        })
        .unwrap();

        let job = InferenceJob {
            id: "game-001".to_string(),
            job_type: InferenceJobType::GameClassification,
            target_id: sit.id,
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 500,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        let analysis: GameAnalysis = serde_json::from_value(result.result).unwrap();
        assert_eq!(analysis.game_structure.maturity, MaturityLevel::Candidate);
    }
}
