//! Wargame move system — actions, validation, and effects.
//!
//! Defines the complete action taxonomy for red/blue teams,
//! validates moves against resource constraints, and applies
//! effects to the SimulationState.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};

use super::sim_state::*;
use super::types::*;

// ─── Wargame Moves ───────────────────────────────────────────

/// A single move in the wargame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WargameMove {
    pub turn: usize,
    pub team: Team,
    pub actor_id: Uuid,
    pub action: WargameAction,
    pub target_narrative: String,
    pub target_platform: String,
}

/// Complete action taxonomy for wargame moves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WargameAction {
    // ─── Red team actions ────────────────────────
    /// Publish original content on a platform.
    PublishContent { content: String, amplify: bool },
    /// Create new fake accounts/personas.
    CreateAccounts {
        count: usize,
        persona_type: PersonaType,
    },
    /// Coordinate amplification across multiple accounts.
    CoordinateAmplification { account_count: usize },
    /// Mutate an existing narrative with a new spin.
    MutateNarrative { mutation: String },
    /// Bridge a narrative from one platform to another.
    CrossPlatformBridge {
        from_platform: String,
        to_platform: String,
    },

    // ─── Blue team actions ───────────────────────
    /// Prebunk: inoculate audience before misinformation arrives.
    Prebunk {
        content: String,
        technique: InoculationTechnique,
    },
    /// Debunk: counter existing misinformation with evidence.
    Debunk { evidence: String },
    /// Request platform takedown of accounts.
    TakeDown { account_count: usize },
    /// Reduce algorithmic amplification.
    ReduceAmplification { method: AmplReductionMethod },
    /// Launch an inoculation campaign targeting a segment.
    InoculationCampaign {
        target_segment: String,
        technique: InoculationTechnique,
    },
    /// Deploy a counter-narrative.
    CounterNarrative { counter_content: String },

    // ─── Both teams ──────────────────────────────
    /// Wait and observe without acting.
    WaitAndObserve,
}

/// Result of advancing one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnResult {
    pub turn: usize,
    pub red_moves_applied: usize,
    pub blue_moves_applied: usize,
    pub r0_after: f64,
    pub misinformed_total: f64,
    pub susceptible_total: f64,
    pub objectives_met: Vec<String>,
}

// ─── Move Validation ─────────────────────────────────────────

/// Validate a move against the current simulation state.
pub fn validate_move(state: &SimulationState, mv: &WargameMove) -> Result<()> {
    // Actor must exist
    let actor = state
        .actors
        .get(&mv.actor_id)
        .ok_or_else(|| TensaError::Internal(format!("Actor not found: {}", mv.actor_id)))?;

    // Team must match
    if actor.team != mv.team {
        return Err(TensaError::Internal(format!(
            "Actor {} is on team {:?}, not {:?}",
            mv.actor_id, actor.team, mv.team
        )));
    }

    // Check resources
    if actor.resources.posts_remaining == 0 {
        match &mv.action {
            WargameAction::WaitAndObserve => {} // Always allowed
            _ => {
                return Err(TensaError::Internal(
                    "Actor has no posts remaining this turn".into(),
                ));
            }
        }
    }

    // Red-only actions
    match &mv.action {
        WargameAction::PublishContent { .. }
        | WargameAction::CreateAccounts { .. }
        | WargameAction::CoordinateAmplification { .. }
        | WargameAction::MutateNarrative { .. }
        | WargameAction::CrossPlatformBridge { .. } => {
            if mv.team != Team::Red {
                return Err(TensaError::Internal(
                    "Only Red team can perform offensive actions".into(),
                ));
            }
        }
        // Blue-only actions
        WargameAction::Prebunk { .. }
        | WargameAction::Debunk { .. }
        | WargameAction::TakeDown { .. }
        | WargameAction::ReduceAmplification { .. }
        | WargameAction::InoculationCampaign { .. }
        | WargameAction::CounterNarrative { .. } => {
            if mv.team != Team::Blue {
                return Err(TensaError::Internal(
                    "Only Blue team can perform defensive actions".into(),
                ));
            }
        }
        WargameAction::WaitAndObserve => {}
    }

    Ok(())
}

// ─── Move Effects ────────────────────────────────────────────

/// Apply action effects to SMIR compartments and platform state.
///
/// Shared between `apply_move` (wargame turns) and the intervention
/// optimizer (counterfactual testing). Extracted to prevent coefficient drift.
pub(super) fn apply_action_effects(
    state: &mut SimulationState,
    action: &WargameAction,
    compartment_key: &str,
    target_narrative: &str,
    target_platform: &str,
) {
    match action {
        WargameAction::PublishContent { amplify, .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                let seed = if *amplify { 20.0 } else { 10.0 };
                comp.seed_misinformed(seed);
            }
        }
        WargameAction::CreateAccounts { count, .. } => {
            if let Some(ps) = state.platforms.get_mut(target_platform) {
                ps.active_accounts += count;
            }
        }
        WargameAction::CoordinateAmplification { account_count } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.beta *= 1.0 + (*account_count as f64 * 0.02);
            }
        }
        WargameAction::MutateNarrative { .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                let reconvert = comp.recovered * 0.1;
                comp.recovered -= reconvert;
                comp.susceptible += reconvert;
            }
        }
        WargameAction::CrossPlatformBridge { to_platform, .. } => {
            let to_key = super::sim_state::compartment_key(target_narrative, to_platform);
            if let Some(comp) = state.compartments.get_mut(&to_key) {
                comp.seed_misinformed(5.0);
            }
        }
        WargameAction::Prebunk { .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.delta += 0.02;
            }
        }
        WargameAction::Debunk { .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.gamma += 0.03;
            }
        }
        WargameAction::TakeDown { account_count } => {
            if let Some(ps) = state.platforms.get_mut(target_platform) {
                ps.active_accounts = ps.active_accounts.saturating_sub(*account_count);
            }
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.beta *= (1.0 - *account_count as f64 * 0.01).max(0.05);
            }
        }
        WargameAction::ReduceAmplification { .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.beta *= 0.8;
            }
        }
        WargameAction::InoculationCampaign { .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.delta += 0.05;
            }
        }
        WargameAction::CounterNarrative { .. } => {
            if let Some(comp) = state.compartments.get_mut(compartment_key) {
                comp.gamma += 0.02;
                comp.beta *= 0.9;
            }
        }
        WargameAction::WaitAndObserve => {}
    }
}

/// Apply a single move's effects to the simulation state.
pub fn apply_move(state: &mut SimulationState, mv: &WargameMove) -> Result<()> {
    let key = super::sim_state::compartment_key(&mv.target_narrative, &mv.target_platform);

    apply_action_effects(
        state,
        &mv.action,
        &key,
        &mv.target_narrative,
        &mv.target_platform,
    );

    // Deduct resources
    if !matches!(mv.action, WargameAction::WaitAndObserve) {
        if let Some(actor) = state.actors.get_mut(&mv.actor_id) {
            actor.resources.posts_remaining = actor.resources.posts_remaining.saturating_sub(1);
            actor.actions_this_turn += 1;
        }
    }

    // Log the move
    state.move_log.push(MoveLogEntry {
        turn: mv.turn,
        team: mv.team,
        actor_id: mv.actor_id,
        action_type: wargame_action_to_type(&mv.action),
        platform: mv.target_platform.clone(),
        narrative_id: mv.target_narrative.clone(),
        timestamp: state.current_time,
    });

    Ok(())
}

/// Convert WargameAction to the simpler ActionType for logging.
fn wargame_action_to_type(action: &WargameAction) -> ActionType {
    match action {
        WargameAction::PublishContent { .. } => ActionType::Post,
        WargameAction::CreateAccounts { .. } => ActionType::CreateAccount,
        WargameAction::CoordinateAmplification { .. } => ActionType::Coordinate,
        WargameAction::MutateNarrative { .. } => ActionType::Post,
        WargameAction::CrossPlatformBridge { .. } => ActionType::Amplify,
        WargameAction::Prebunk { .. } => ActionType::Prebunk,
        WargameAction::Debunk { .. } => ActionType::Debunk,
        WargameAction::TakeDown { .. } => ActionType::TakeDown,
        WargameAction::ReduceAmplification { .. } => ActionType::TakeDown,
        WargameAction::InoculationCampaign { .. } => ActionType::Prebunk,
        WargameAction::CounterNarrative { .. } => ActionType::Debunk,
        WargameAction::WaitAndObserve => ActionType::Observe,
    }
}

/// Advance one turn: validate + apply red moves, then blue moves, then evolve SMIR.
///
/// Validates and applies each move sequentially so resource constraints
/// are correctly enforced across multiple moves from the same actor.
pub fn advance_turn(
    state: &mut SimulationState,
    red_moves: &[WargameMove],
    blue_moves: &[WargameMove],
) -> Result<TurnResult> {
    let turn = state.turn_number;

    // Validate and apply each move sequentially (resource checks are stateful)
    for mv in red_moves.iter().chain(blue_moves.iter()) {
        validate_move(state, mv)?;
        apply_move(state, mv)?;
    }

    // Advance time + evolve SMIR
    state.advance_turn();

    // Compute aggregate stats
    let (r0, misinformed, susceptible) = aggregate_compartments(state);

    Ok(TurnResult {
        turn,
        red_moves_applied: red_moves.len(),
        blue_moves_applied: blue_moves.len(),
        r0_after: r0,
        misinformed_total: misinformed,
        susceptible_total: susceptible,
        objectives_met: vec![],
    })
}

fn aggregate_compartments(state: &SimulationState) -> (f64, f64, f64) {
    let mut total_r0 = 0.0;
    let mut total_m = 0.0;
    let mut total_s = 0.0;
    let mut count = 0;

    for comp in state.compartments.values() {
        total_r0 += comp.r0();
        total_m += comp.misinformed;
        total_s += comp.susceptible;
        count += 1;
    }

    let avg_r0 = if count > 0 {
        total_r0 / count as f64
    } else {
        0.0
    };
    (avg_r0, total_m, total_s)
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::EntityType;
    use std::sync::Arc;

    fn setup_state() -> (SimulationState, Uuid, Uuid) {
        let store = Arc::new(MemoryStore::new());
        let hg = crate::Hypergraph::new(store);

        let red_id = hg
            .create_entity(crate::types::Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "RedAgent"}),
                beliefs: None,
                embedding: None,
                maturity: crate::types::MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("test-war".to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();

        let blue_id = hg
            .create_entity(crate::types::Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "BlueAgent"}),
                beliefs: None,
                embedding: None,
                maturity: crate::types::MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("test-war".to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();

        let mut state = SimulationState::fork_from_hypergraph(&hg, "test-war", 60).unwrap();
        state.assign_team(&red_id, Team::Red).unwrap();
        state.assign_team(&blue_id, Team::Blue).unwrap();
        state.seed_narrative("test-war", "twitter", 10.0);

        (state, red_id, blue_id)
    }

    #[test]
    fn test_red_publish_increases_misinformed() {
        let (mut state, red_id, _) = setup_state();

        let key = compartment_key("test-war", "twitter");
        let m_before = state.compartments[&key].misinformed;

        let mv = WargameMove {
            turn: 0,
            team: Team::Red,
            actor_id: red_id,
            action: WargameAction::PublishContent {
                content: "fake news".into(),
                amplify: false,
            },
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        };

        apply_move(&mut state, &mv).unwrap();
        assert!(
            state.compartments[&key].misinformed > m_before,
            "misinformed should increase after red publish"
        );
    }

    #[test]
    fn test_blue_debunk_increases_gamma() {
        let (mut state, _, blue_id) = setup_state();

        let key = compartment_key("test-war", "twitter");
        let gamma_before = state.compartments[&key].gamma;

        let mv = WargameMove {
            turn: 0,
            team: Team::Blue,
            actor_id: blue_id,
            action: WargameAction::Debunk {
                evidence: "fact check".into(),
            },
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        };

        apply_move(&mut state, &mv).unwrap();
        assert!(
            state.compartments[&key].gamma > gamma_before,
            "gamma should increase after debunk"
        );
    }

    #[test]
    fn test_team_mismatch_rejected() {
        let (state, red_id, _) = setup_state();

        let mv = WargameMove {
            turn: 0,
            team: Team::Blue, // Red actor claiming Blue team
            actor_id: red_id,
            action: WargameAction::Debunk {
                evidence: "fake".into(),
            },
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        };

        assert!(validate_move(&state, &mv).is_err());
    }

    #[test]
    fn test_red_action_rejected_for_blue() {
        let (state, _, blue_id) = setup_state();

        let mv = WargameMove {
            turn: 0,
            team: Team::Blue,
            actor_id: blue_id,
            action: WargameAction::PublishContent {
                content: "disinfo".into(),
                amplify: true,
            },
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        };

        assert!(validate_move(&state, &mv).is_err());
    }

    #[test]
    fn test_advance_turn_full_cycle() {
        let (mut state, red_id, blue_id) = setup_state();

        let red_moves = vec![WargameMove {
            turn: 0,
            team: Team::Red,
            actor_id: red_id,
            action: WargameAction::PublishContent {
                content: "propaganda".into(),
                amplify: true,
            },
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        }];

        let blue_moves = vec![WargameMove {
            turn: 0,
            team: Team::Blue,
            actor_id: blue_id,
            action: WargameAction::Debunk {
                evidence: "correction".into(),
            },
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        }];

        let result = advance_turn(&mut state, &red_moves, &blue_moves).unwrap();
        assert_eq!(result.turn, 0);
        assert_eq!(result.red_moves_applied, 1);
        assert_eq!(result.blue_moves_applied, 1);
        assert_eq!(state.turn_number, 1);
        assert!(result.r0_after >= 0.0);
    }

    #[test]
    fn test_wait_and_observe_costs_nothing() {
        let (mut state, red_id, _) = setup_state();

        let posts_before = state.actors[&red_id].resources.posts_remaining;

        let mv = WargameMove {
            turn: 0,
            team: Team::Red,
            actor_id: red_id,
            action: WargameAction::WaitAndObserve,
            target_narrative: "test-war".into(),
            target_platform: "twitter".into(),
        };

        apply_move(&mut state, &mv).unwrap();
        assert_eq!(
            state.actors[&red_id].resources.posts_remaining, posts_before,
            "observe should not cost resources"
        );
    }

    #[test]
    fn test_five_turn_autoplay() {
        let (mut state, red_id, blue_id) = setup_state();

        for turn in 0..5 {
            let red = vec![WargameMove {
                turn,
                team: Team::Red,
                actor_id: red_id,
                action: WargameAction::PublishContent {
                    content: format!("turn {} propaganda", turn),
                    amplify: turn % 2 == 0,
                },
                target_narrative: "test-war".into(),
                target_platform: "twitter".into(),
            }];

            let blue = vec![WargameMove {
                turn,
                team: Team::Blue,
                actor_id: blue_id,
                action: WargameAction::Debunk {
                    evidence: format!("turn {} fact check", turn),
                },
                target_narrative: "test-war".into(),
                target_platform: "twitter".into(),
            }];

            let result = advance_turn(&mut state, &red, &blue).unwrap();
            assert_eq!(result.turn, turn);
        }

        assert_eq!(state.turn_number, 5);
        assert_eq!(state.move_log.len(), 10); // 5 red + 5 blue
    }
}
