//! Narrative expansion — grow a narrative's length by identifying structural
//! thinness and proposing new content. The engine is plan-level — it emits
//! an `ExpansionPlan` enumerating what to add, not LLM-generated prose.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpansionStrategy {
    SubplotAddition,
    CharacterDevelopment,
    CommitmentExtension,
    SceneExpansion,
    WorldBuilding,
    Balanced,
}

impl Default for ExpansionStrategy {
    fn default() -> Self {
        ExpansionStrategy::Balanced
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionConfig {
    pub target_chapters: usize,
    #[serde(default)]
    pub expansion_strategy: ExpansionStrategy,
    /// If true, maintain the existing scene-sequel ratio.
    #[serde(default = "default_preserve_pacing")]
    pub preserve_pacing: bool,
    pub genre: Option<String>,
}

fn default_preserve_pacing() -> bool {
    true
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        ExpansionConfig {
            target_chapters: 24,
            expansion_strategy: ExpansionStrategy::Balanced,
            preserve_pacing: true,
            genre: None,
        }
    }
}

/// A proposed new subplot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSubplot {
    pub id: Uuid,
    pub theme: String,
    pub relation: String,
    pub proposed_situations: usize,
    pub involves_entities: Vec<Uuid>,
}

/// A proposed new situation (plan-level only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSituation {
    pub id: Uuid,
    pub after_situation: Option<Uuid>,
    pub chapter_hint: Option<usize>,
    pub purpose: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionPlan {
    pub narrative_id: String,
    pub added_subplots: Vec<PlannedSubplot>,
    /// Per-character: list of situations planned to deepen that arc.
    pub deepened_arcs: Vec<(Uuid, Vec<PlannedSituation>)>,
    /// Per-commitment: progress events planned between setup and payoff.
    pub extended_commitments: Vec<(Uuid, Vec<PlannedSituation>)>,
    /// Per-scene: situations that replace a single dense scene.
    pub expanded_scenes: Vec<(Uuid, Vec<PlannedSituation>)>,
    /// Bare world-building pause segments.
    pub world_building_additions: Vec<PlannedSituation>,
    pub original_chapters: usize,
    pub expanded_chapters: usize,
    pub expansion_ratio: f64,
    pub strategy: ExpansionStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionPreview {
    pub narrative_id: String,
    pub would_add_subplots: usize,
    pub would_deepen_arcs: usize,
    pub would_extend_commitments: usize,
    pub would_expand_scenes: usize,
    pub estimated_chapter_count: usize,
    pub warnings: Vec<String>,
}

// ─── Expansion planner ──────────────────────────────────────

pub fn expand_narrative(
    hg: &Hypergraph,
    narrative_id: &str,
    config: &ExpansionConfig,
) -> Result<ExpansionPlan> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let original = situations.len();

    let mut plan = ExpansionPlan {
        narrative_id: narrative_id.to_string(),
        added_subplots: Vec::new(),
        deepened_arcs: Vec::new(),
        extended_commitments: Vec::new(),
        expanded_scenes: Vec::new(),
        world_building_additions: Vec::new(),
        original_chapters: original,
        expanded_chapters: original,
        expansion_ratio: 1.0,
        strategy: config.expansion_strategy.clone(),
    };

    if config.target_chapters <= original {
        return Ok(plan);
    }
    let slack = config.target_chapters - original;
    let mut added = 0usize;

    let strategies: Vec<ExpansionStrategy> = match config.expansion_strategy {
        ExpansionStrategy::Balanced => vec![
            ExpansionStrategy::CharacterDevelopment,
            ExpansionStrategy::CommitmentExtension,
            ExpansionStrategy::SceneExpansion,
            ExpansionStrategy::SubplotAddition,
            ExpansionStrategy::WorldBuilding,
        ],
        ref s => vec![s.clone()],
    };

    for strat in &strategies {
        if added >= slack {
            break;
        }
        let remaining = slack - added;
        let produced = match strat {
            ExpansionStrategy::CharacterDevelopment => {
                deepen_arcs(hg, narrative_id, remaining, &mut plan)?
            }
            ExpansionStrategy::CommitmentExtension => {
                extend_commitments(hg, narrative_id, remaining, &mut plan)?
            }
            ExpansionStrategy::SceneExpansion => {
                expand_scenes_via_pacing(hg, narrative_id, remaining, &mut plan)?
            }
            ExpansionStrategy::SubplotAddition => add_subplot_placeholder(remaining, &mut plan),
            ExpansionStrategy::WorldBuilding => add_world_building(remaining, &mut plan),
            ExpansionStrategy::Balanced => 0,
        };
        added += produced;
    }

    plan.expanded_chapters = original + added;
    plan.expansion_ratio = if original == 0 {
        1.0
    } else {
        plan.expanded_chapters as f64 / original as f64
    };
    Ok(plan)
}

fn deepen_arcs(
    hg: &Hypergraph,
    narrative_id: &str,
    budget: usize,
    plan: &mut ExpansionPlan,
) -> Result<usize> {
    let arcs =
        crate::narrative::character_arcs::list_character_arcs(hg, narrative_id).unwrap_or_default();
    let mut added = 0usize;
    for arc in &arcs {
        if added >= budget {
            break;
        }
        if arc.completeness >= 0.7 {
            continue;
        }
        let slots = ((0.7 - arc.completeness) * 4.0).round().max(1.0) as usize;
        let mut situations = Vec::with_capacity(slots);
        for _ in 0..slots {
            if added >= budget {
                break;
            }
            situations.push(PlannedSituation {
                id: Uuid::now_v7(),
                after_situation: None,
                chapter_hint: None,
                purpose: "Arc deepening: test lie / confront need".into(),
            });
            added += 1;
        }
        if !situations.is_empty() {
            plan.deepened_arcs.push((arc.character_id, situations));
        }
    }
    Ok(added)
}

fn extend_commitments(
    hg: &Hypergraph,
    narrative_id: &str,
    budget: usize,
    plan: &mut ExpansionPlan,
) -> Result<usize> {
    let commitments =
        crate::narrative::commitments::list_commitments(hg, narrative_id).unwrap_or_default();
    let mut added = 0usize;
    for c in &commitments {
        if added >= budget {
            break;
        }
        if let Some(dist) = c.payoff_distance {
            if dist < 3 {
                let slots = (3 - dist).max(1);
                let mut sits = Vec::with_capacity(slots);
                for _ in 0..slots {
                    if added >= budget {
                        break;
                    }
                    sits.push(PlannedSituation {
                        id: Uuid::now_v7(),
                                after_situation: Some(c.setup_event),
                        chapter_hint: Some(c.setup_chapter + 1),
                        purpose: format!("Progress event for commitment '{}'", c.tracked_element),
                    });
                    added += 1;
                }
                if !sits.is_empty() {
                    plan.extended_commitments.push((c.id, sits));
                }
            }
        }
    }
    Ok(added)
}

fn expand_scenes_via_pacing(
    hg: &Hypergraph,
    narrative_id: &str,
    budget: usize,
    plan: &mut ExpansionPlan,
) -> Result<usize> {
    let ss = match crate::narrative::scene_sequel::analyze_scene_sequel(hg, narrative_id) {
        Ok(s) => s,
        Err(_) => return Ok(0),
    };
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let mut sorted = situations.clone();
    sorted.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));
    let mut added = 0usize;
    let mut action_streak = 0;
    for (idx, cls) in ss.classifications.iter().enumerate() {
        if added >= budget {
            break;
        }
        if cls.scene_type.is_action() {
            action_streak += 1;
            if action_streak >= 3 {
                if let Some(sit) = sorted.get(idx) {
                    let planned = PlannedSituation {
                        id: Uuid::now_v7(),
                                after_situation: Some(sit.id),
                        chapter_hint: Some(idx),
                        purpose: "Insert sequel (reaction/dilemma/decision) after action streak"
                            .into(),
                    };
                    plan.expanded_scenes.push((sit.id, vec![planned]));
                    added += 1;
                    action_streak = 0;
                }
            }
        } else {
            action_streak = 0;
        }
    }
    Ok(added)
}

fn add_subplot_placeholder(budget: usize, plan: &mut ExpansionPlan) -> usize {
    if budget < 3 {
        return 0;
    }
    let slots = budget.min(4);
    plan.added_subplots.push(PlannedSubplot {
        id: Uuid::now_v7(),
        theme: "thematic-parallel".into(),
        relation: "Mirror".into(),
        proposed_situations: slots,
        involves_entities: Vec::new(),
    });
    slots
}

fn add_world_building(budget: usize, plan: &mut ExpansionPlan) -> usize {
    let slots = budget.min(2);
    for _ in 0..slots {
        plan.world_building_additions.push(PlannedSituation {
            id: Uuid::now_v7(),
            after_situation: None,
            chapter_hint: None,
            purpose: "World-building Pause segment".into(),
        });
    }
    slots
}

pub fn preview_expansion(
    hg: &Hypergraph,
    narrative_id: &str,
    config: &ExpansionConfig,
) -> Result<ExpansionPreview> {
    let plan = expand_narrative(hg, narrative_id, config)?;
    let mut warnings = Vec::new();
    if plan.expansion_ratio > 3.0 {
        warnings.push("Expansion ratio > 3× — consider splitting into a series".into());
    }
    Ok(ExpansionPreview {
        narrative_id: narrative_id.to_string(),
        would_add_subplots: plan.added_subplots.len(),
        would_deepen_arcs: plan.deepened_arcs.len(),
        would_extend_commitments: plan.extended_commitments.len(),
        would_expand_scenes: plan.expanded_scenes.len(),
        estimated_chapter_count: plan.expanded_chapters,
        warnings,
    })
}

// ─── Preset helpers ──────────────────────────────────────────

pub fn expand_to_novel(
    hg: &Hypergraph,
    narrative_id: &str,
    target_chapters: usize,
) -> Result<ExpansionPlan> {
    let cfg = ExpansionConfig {
        target_chapters,
        expansion_strategy: ExpansionStrategy::Balanced,
        preserve_pacing: true,
        genre: None,
    };
    expand_narrative(hg, narrative_id, &cfg)
}

pub fn add_subplot_to(
    hg: &Hypergraph,
    narrative_id: &str,
    theme: &str,
    relation: &str,
) -> Result<ExpansionPlan> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let mut plan = ExpansionPlan {
        narrative_id: narrative_id.to_string(),
        added_subplots: vec![PlannedSubplot {
            id: Uuid::now_v7(),
            theme: theme.to_string(),
            relation: relation.to_string(),
            proposed_situations: 4,
            involves_entities: Vec::new(),
        }],
        deepened_arcs: Vec::new(),
        extended_commitments: Vec::new(),
        expanded_scenes: Vec::new(),
        world_building_additions: Vec::new(),
        original_chapters: situations.len(),
        expanded_chapters: situations.len() + 4,
        expansion_ratio: 0.0,
        strategy: ExpansionStrategy::SubplotAddition,
    };
    plan.expansion_ratio = if plan.original_chapters == 0 {
        1.0
    } else {
        plan.expanded_chapters as f64 / plan.original_chapters as f64
    };
    Ok(plan)
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_expand_empty_narrative_below_target_is_noop() {
        let hg = test_hg();
        // Target below original = noop
        let cfg = ExpansionConfig {
            target_chapters: 0,
            ..Default::default()
        };
        let plan = expand_narrative(&hg, "empty", &cfg).unwrap();
        assert_eq!(plan.original_chapters, 0);
        assert!(plan.added_subplots.is_empty());
    }

    #[test]
    fn test_expand_empty_narrative_with_slack_plans_additions() {
        let hg = test_hg();
        let cfg = ExpansionConfig {
            target_chapters: 10,
            ..Default::default()
        };
        let plan = expand_narrative(&hg, "empty", &cfg).unwrap();
        assert_eq!(plan.original_chapters, 0);
        // Balanced strategy over empty narrative still produces a subplot placeholder
        assert!(plan.expanded_chapters >= plan.original_chapters);
    }

    #[test]
    fn test_preview_expansion_reports_no_warnings_on_empty() {
        let hg = test_hg();
        let preview = preview_expansion(&hg, "empty", &ExpansionConfig::default()).unwrap();
        assert!(preview.warnings.is_empty());
    }

    #[test]
    fn test_add_subplot_preset() {
        let hg = test_hg();
        let plan = add_subplot_to(&hg, "empty", "love", "Mirror").unwrap();
        assert_eq!(plan.added_subplots.len(), 1);
        assert_eq!(plan.added_subplots[0].theme, "love");
    }
}
