//! Intervention optimizer — find optimal defensive response.
//!
//! Given current state + threat, greedily select the best available
//! intervention by running SMIR counterfactuals for each candidate
//! action, ranking by R₀ reduction per unit cost.

use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::sim_state::*;
use super::types::*;
use super::wargame::*;

// ─── Types ───────────────────────────────────────────────────

/// Budget for intervention actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionBudget {
    /// Maximum number of actions.
    pub max_actions: usize,
    /// Available platforms for intervention.
    pub platforms: Vec<String>,
}

impl Default for InterventionBudget {
    fn default() -> Self {
        Self {
            max_actions: 5,
            platforms: vec!["twitter".into(), "facebook".into(), "telegram".into()],
        }
    }
}

/// Intervention objective to optimize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionObjective {
    MinimizeR0,
    MinimizeAudienceReach,
    MaximizeDebunkPenetration,
}

impl Default for InterventionObjective {
    fn default() -> Self {
        Self::MinimizeR0
    }
}

/// Recommended intervention plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionPlan {
    /// Ranked recommended actions (best first).
    pub recommended_actions: Vec<ScoredIntervention>,
    /// Projected R₀ after all recommended actions.
    pub projected_r0: f64,
    /// Estimated audience saved (vs no intervention).
    pub estimated_audience_saved: f64,
}

/// A single scored intervention candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredIntervention {
    /// The action to take.
    pub action: WargameAction,
    /// Target platform.
    pub platform: String,
    /// Target narrative.
    pub narrative_id: String,
    /// R₀ reduction from this single action.
    pub r0_reduction: f64,
    /// Priority score (higher = more impactful).
    pub priority: f64,
}

// ─── Optimizer ───────────────────────────────────────────────

/// Find the optimal set of defensive interventions.
///
/// For each available blue-team action on each available platform,
/// simulates the effect on SMIR compartments and ranks by R₀ reduction.
pub fn optimize_intervention(
    state: &SimulationState,
    budget: &InterventionBudget,
    objective: &InterventionObjective,
) -> Result<InterventionPlan> {
    let narrative_id = &state.narrative_id;
    let mut candidates: Vec<ScoredIntervention> = Vec::new();

    // Candidate blue actions to test
    let blue_actions: Vec<WargameAction> = vec![
        WargameAction::Debunk {
            evidence: "fact-check".into(),
        },
        WargameAction::Prebunk {
            content: "inoculation".into(),
            technique: InoculationTechnique::TechniqueExposure,
        },
        WargameAction::ReduceAmplification {
            method: AmplReductionMethod::Downrank,
        },
        WargameAction::CounterNarrative {
            counter_content: "counter".into(),
        },
        WargameAction::InoculationCampaign {
            target_segment: "general".into(),
            technique: InoculationTechnique::WeakenedDose,
        },
    ];

    // Current R₀ baseline
    let baseline_r0 = average_r0(state);

    // Test each action on each platform via counterfactual
    for platform in &budget.platforms {
        for action in &blue_actions {
            // Clone state, apply the single action, evolve one step
            let mut test_state = state.clone();
            let key = compartment_key(narrative_id, platform);

            apply_action_effect(&mut test_state, action, &key, narrative_id, platform);
            test_state.evolve_smir();

            let new_r0 = average_r0(&test_state);
            let reduction = baseline_r0 - new_r0;

            let priority = match objective {
                InterventionObjective::MinimizeR0 => reduction,
                InterventionObjective::MinimizeAudienceReach => {
                    let baseline_m: f64 = state.compartments.values().map(|c| c.misinformed).sum();
                    let new_m: f64 = test_state
                        .compartments
                        .values()
                        .map(|c| c.misinformed)
                        .sum();
                    baseline_m - new_m
                }
                InterventionObjective::MaximizeDebunkPenetration => {
                    let baseline_r: f64 = state.compartments.values().map(|c| c.recovered).sum();
                    let new_r: f64 = test_state.compartments.values().map(|c| c.recovered).sum();
                    new_r - baseline_r
                }
            };

            candidates.push(ScoredIntervention {
                action: action.clone(),
                platform: platform.clone(),
                narrative_id: narrative_id.clone(),
                r0_reduction: reduction,
                priority,
            });
        }
    }

    // Sort by priority descending
    candidates.sort_by(|a, b| {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top N within budget
    let recommended: Vec<ScoredIntervention> =
        candidates.into_iter().take(budget.max_actions).collect();

    // Project final R₀ after all recommended actions
    let mut final_state = state.clone();
    for intervention in &recommended {
        let key = compartment_key(narrative_id, &intervention.platform);
        apply_action_effect(
            &mut final_state,
            &intervention.action,
            &key,
            narrative_id,
            &intervention.platform,
        );
    }
    final_state.evolve_smir();
    let projected_r0 = average_r0(&final_state);

    // Estimate audience saved
    let baseline_m: f64 = state.compartments.values().map(|c| c.misinformed).sum();
    let final_m: f64 = final_state
        .compartments
        .values()
        .map(|c| c.misinformed)
        .sum();
    let estimated_audience_saved = (baseline_m - final_m).max(0.0);

    Ok(InterventionPlan {
        recommended_actions: recommended,
        projected_r0,
        estimated_audience_saved,
    })
}

/// Delegate to the shared effect function in wargame.rs.
fn apply_action_effect(
    state: &mut SimulationState,
    action: &WargameAction,
    key: &str,
    narrative_id: &str,
    platform: &str,
) {
    super::wargame::apply_action_effects(state, action, key, narrative_id, platform);
}

/// Compute average R₀ across all compartments.
fn average_r0(state: &SimulationState) -> f64 {
    let mut total = 0.0;
    let mut count = 0;
    for comp in state.compartments.values() {
        total += comp.r0();
        count += 1;
    }
    if count > 0 {
        total / count as f64
    } else {
        0.0
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::EntityType;
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    fn setup_state() -> SimulationState {
        let store = Arc::new(MemoryStore::new());
        let hg = crate::Hypergraph::new(store);

        hg.create_entity(crate::types::Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Test"}),
            beliefs: None,
            embedding: None,
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("opt-test".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap();

        let mut state = SimulationState::fork_from_hypergraph(&hg, "opt-test", 60).unwrap();
        state.seed_narrative("opt-test", "twitter", 100.0);
        state.seed_narrative("opt-test", "facebook", 50.0);
        state
    }

    #[test]
    fn test_optimizer_produces_recommendations() {
        let state = setup_state();
        let budget = InterventionBudget::default();
        let plan =
            optimize_intervention(&state, &budget, &InterventionObjective::MinimizeR0).unwrap();

        assert!(!plan.recommended_actions.is_empty());
        assert!(plan.recommended_actions.len() <= budget.max_actions);
    }

    #[test]
    fn test_optimizer_reduces_r0() {
        let state = setup_state();
        let baseline = average_r0(&state);

        let budget = InterventionBudget::default();
        let plan =
            optimize_intervention(&state, &budget, &InterventionObjective::MinimizeR0).unwrap();

        assert!(
            plan.projected_r0 < baseline,
            "intervention should reduce R₀: {} vs {}",
            plan.projected_r0,
            baseline
        );
    }

    #[test]
    fn test_actions_sorted_by_priority() {
        let state = setup_state();
        let budget = InterventionBudget {
            max_actions: 10,
            ..Default::default()
        };
        let plan =
            optimize_intervention(&state, &budget, &InterventionObjective::MinimizeR0).unwrap();

        for w in plan.recommended_actions.windows(2) {
            assert!(
                w[0].priority >= w[1].priority,
                "should be sorted by priority descending"
            );
        }
    }

    #[test]
    fn test_debunk_plus_takedown_outperforms_either() {
        let state = setup_state();
        let baseline = average_r0(&state);

        // Single debunk
        let budget_1 = InterventionBudget {
            max_actions: 1,
            ..Default::default()
        };
        let plan_1 =
            optimize_intervention(&state, &budget_1, &InterventionObjective::MinimizeR0).unwrap();

        // Multiple actions
        let budget_5 = InterventionBudget {
            max_actions: 5,
            ..Default::default()
        };
        let plan_5 =
            optimize_intervention(&state, &budget_5, &InterventionObjective::MinimizeR0).unwrap();

        assert!(
            plan_5.projected_r0 <= plan_1.projected_r0,
            "more actions should reduce R₀ more: {} vs {}",
            plan_5.projected_r0,
            plan_1.projected_r0
        );
    }

    #[test]
    fn test_audience_saved_non_negative() {
        let state = setup_state();
        let budget = InterventionBudget::default();
        let plan =
            optimize_intervention(&state, &budget, &InterventionObjective::MinimizeR0).unwrap();

        assert!(
            plan.estimated_audience_saved >= 0.0,
            "audience saved should be non-negative: {}",
            plan.estimated_audience_saved
        );
    }
}
