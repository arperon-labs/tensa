//! Narrative plan generation (Sprint D9.6, Stage 1).
//!
//! Generates a formal `NarrativePlan` from a `PlanConfig` (premise, genre,
//! chapter count, constraints). The plan is a hypergraph skeleton with entities,
//! situations, Allen relations, causal chains, commitments, subplots, and
//! character arcs — but no prose yet.

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::narrative::character_arcs::ArcType;
use crate::narrative::commitments::CommitmentType;
use crate::narrative::fabula_sjuzet::NarrationMode;
use crate::narrative::scene_sequel::SceneType;
use crate::narrative::subplots::SubplotRelation;

use super::types::*;

// ─── KV Operations ──────────────────────────────────────────

pub fn store_plan(hg: &Hypergraph, plan: &NarrativePlan) -> Result<()> {
    let key = plan_key(&plan.id);
    let val = serde_json::to_vec(plan)?;
    hg.store().put(&key, &val)
}

pub fn load_plan(hg: &Hypergraph, plan_id: &Uuid) -> Result<Option<NarrativePlan>> {
    let key = plan_key(plan_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn delete_plan(hg: &Hypergraph, plan_id: &Uuid) -> Result<()> {
    let key = plan_key(plan_id);
    hg.store().delete(&key)
}

// ─── Plan Generation ────────────────────────────────────────

/// Generate a narrative plan from configuration.
///
/// Hierarchical top-down approach:
/// 1. Macro structure from emotional arc shape
/// 2. Character design with arc types
/// 3. Fact universe design
/// 4. Subplot design
/// 5. Commitment planning
/// 6. Situation generation per chapter
/// 7. Temporal structure
/// 8. Dramatic irony placement
pub fn generate_plan(config: PlanConfig) -> Result<NarrativePlan> {
    if config.premise.is_empty() {
        return Err(TensaError::InvalidQuery("premise is required".into()));
    }
    if config.chapter_count == 0 {
        return Err(TensaError::InvalidQuery("chapter_count must be > 0".into()));
    }

    let plan_id = Uuid::now_v7();
    let narrative_id = format!("gen-{}", &plan_id.to_string()[..8]);

    // 1. Design characters
    let entities = design_characters(&config);

    // 2. Design fact universe
    let facts = design_facts(&config, &entities);

    // 3. Design character arcs
    let character_arcs = design_arcs(&config, &entities);

    // 4. Design subplots
    let subplots = design_subplots(&config, &entities);

    // 5. Design commitments
    let commitments = design_commitments(&config);

    // 6. Generate situations per chapter
    let fabula = generate_situations(&config, &entities, &commitments, &facts);

    Ok(NarrativePlan {
        id: plan_id,
        narrative_id,
        config,
        entities,
        facts,
        fabula,
        commitments,
        subplots,
        character_arcs,
        created_at: Utc::now(),
        updated_at: Utc::now(),
    })
}

/// Design characters based on configuration.
fn design_characters(config: &PlanConfig) -> Vec<PlannedEntity> {
    let mut entities = Vec::new();

    // Protagonist(s)
    for i in 0..config.protagonist_count {
        let id = Uuid::now_v7();
        entities.push(PlannedEntity {
            id,
            name: format!("Protagonist {}", i + 1),
            entity_type: "Actor".into(),
            role: crate::types::Role::Protagonist,
            arc_type: if i == 0 {
                ArcType::PositiveChange
            } else {
                ArcType::Flat
            },
            want: format!("(TBD: conscious desire for protagonist {})", i + 1),
            need: format!("(TBD: unconscious need for protagonist {})", i + 1),
            lie: Some("(TBD: false belief)".into()),
            truth: Some("(TBD: truth to discover)".into()),
            initial_knowledge: Default::default(),
            initial_motivation: vec![0.5; 5],
            relationships: Vec::new(),
        });
    }

    // Antagonist
    let antag_id = Uuid::now_v7();
    entities.push(PlannedEntity {
        id: antag_id,
        name: "Antagonist".into(),
        entity_type: "Actor".into(),
        role: crate::types::Role::Antagonist,
        arc_type: ArcType::NegativeCorruption,
        want: "(TBD: antagonist desire)".into(),
        need: "(TBD: antagonist need)".into(),
        lie: None,
        truth: None,
        initial_knowledge: Default::default(),
        initial_motivation: vec![0.5; 5],
        relationships: Vec::new(),
    });

    // Supporting characters for subplots
    for i in 0..config.subplot_count {
        entities.push(PlannedEntity {
            id: Uuid::now_v7(),
            name: format!("Supporting {}", i + 1),
            entity_type: "Actor".into(),
            role: crate::types::Role::Bystander,
            arc_type: ArcType::Flat,
            want: format!("(TBD: supporting {} desire)", i + 1),
            need: format!("(TBD: supporting {} need)", i + 1),
            lie: None,
            truth: None,
            initial_knowledge: Default::default(),
            initial_motivation: vec![0.5; 5],
            relationships: Vec::new(),
        });
    }

    // Set up protagonist-antagonist relationships
    if entities.len() >= 2 {
        let protag_id = entities[0].id;
        entities[0]
            .relationships
            .push((antag_id, "adversary".into()));
        if let Some(antag) = entities.iter_mut().find(|e| e.id == antag_id) {
            antag.relationships.push((protag_id, "adversary".into()));
        }
    }

    entities
}

/// Design the fact universe.
fn design_facts(config: &PlanConfig, entities: &[PlannedEntity]) -> Vec<PlannedFact> {
    let mut facts = Vec::new();

    // Core premise fact
    facts.push(PlannedFact {
        id: "premise".into(),
        description: config.premise.clone(),
        known_by: Vec::new(), // Reader discovers through story
        revealed_in: None,
        is_true: true,
    });

    // One secret per protagonist (creates dramatic irony potential)
    for (i, entity) in entities
        .iter()
        .filter(|e| e.role == crate::types::Role::Protagonist)
        .enumerate()
    {
        facts.push(PlannedFact {
            id: format!("secret-{}", i),
            description: format!("(TBD: {}'s secret)", entity.name),
            known_by: vec![entity.id],
            revealed_in: None,
            is_true: true,
        });
    }

    // Antagonist's deception
    if let Some(antag) = entities
        .iter()
        .find(|e| e.role == crate::types::Role::Antagonist)
    {
        facts.push(PlannedFact {
            id: "deception-1".into(),
            description: "(TBD: false fact the antagonist propagates)".into(),
            known_by: vec![antag.id],
            revealed_in: None,
            is_true: false,
        });
    }

    facts
}

/// Design character arcs with waypoints.
fn design_arcs(config: &PlanConfig, entities: &[PlannedEntity]) -> Vec<PlannedCharacterArc> {
    let n = config.chapter_count;
    let midpoint = n / 2;
    let dark_night = (n * 3) / 4;
    let resolution = n.saturating_sub(1);

    entities
        .iter()
        .filter(|e| {
            e.role == crate::types::Role::Protagonist || e.role == crate::types::Role::Antagonist
        })
        .map(|e| {
            PlannedCharacterArc {
                character_id: e.id,
                arc_type: e.arc_type.clone(),
                midpoint_chapter: midpoint,
                dark_night_chapter: dark_night,
                resolution_chapter: resolution,
                motivation_waypoints: vec![
                    (0, e.initial_motivation.clone()),
                    (midpoint, vec![0.3; 5]),   // Midpoint disruption
                    (dark_night, vec![0.1; 5]), // Dark night
                    (resolution, vec![0.8; 5]), // Resolution
                ],
            }
        })
        .collect()
}

/// Design subplots.
fn design_subplots(config: &PlanConfig, entities: &[PlannedEntity]) -> Vec<PlannedSubplot> {
    let n = config.chapter_count;
    let supporting: Vec<_> = entities
        .iter()
        .filter(|e| e.role == crate::types::Role::Bystander)
        .collect();

    supporting
        .iter()
        .take(config.subplot_count)
        .enumerate()
        .map(|(i, e)| {
            let start = (i * n / config.subplot_count.max(1)).max(1);
            let end = ((i + 1) * n / config.subplot_count.max(1)).min(n);
            let convergence = if i == 0 { Some(n - 2) } else { None };

            PlannedSubplot {
                label: format!("Subplot {}: {}", i + 1, e.name),
                chapters_active: (start..end).collect(),
                relation_to_main: if convergence.is_some() {
                    SubplotRelation::Convergence
                } else {
                    SubplotRelation::Mirror
                },
                convergence_chapter: convergence,
                characters: vec![e.id],
            }
        })
        .collect()
}

/// Design commitments (setup-payoff pairs).
fn design_commitments(config: &PlanConfig) -> Vec<PlannedCommitment> {
    let n = config.chapter_count;
    let count = (n as f64 * config.commitment_density).round() as usize;
    let count = count.max(1).min(n / 2);

    (0..count)
        .map(|i| {
            let setup = (i * n / count).max(0).min(n / 3); // Setups in act 1
            let payoff = (n - 1 - (count - 1 - i) * n / count)
                .max(setup + 2)
                .min(n - 1);
            let progress = if payoff - setup > 3 {
                vec![(setup + payoff) / 2]
            } else {
                Vec::new()
            };

            PlannedCommitment {
                id: format!("commitment-{}", i),
                commitment_type: if i == 0 {
                    CommitmentType::ChekhovsGun
                } else if i % 3 == 1 {
                    CommitmentType::Foreshadowing
                } else {
                    CommitmentType::DramaticQuestion
                },
                element: format!("(TBD: commitment element {})", i + 1),
                setup_chapter: setup,
                payoff_chapter: payoff,
                intermediate_progress_chapters: progress,
            }
        })
        .collect()
}

/// Generate situations for each chapter.
fn generate_situations(
    config: &PlanConfig,
    entities: &[PlannedEntity],
    commitments: &[PlannedCommitment],
    facts: &[PlannedFact],
) -> Vec<PlannedSituation> {
    let n = config.chapter_count;
    let protagonist_ids: Vec<Uuid> = entities
        .iter()
        .filter(|e| e.role == crate::types::Role::Protagonist)
        .map(|e| e.id)
        .collect();
    let antagonist_ids: Vec<Uuid> = entities
        .iter()
        .filter(|e| e.role == crate::types::Role::Antagonist)
        .map(|e| e.id)
        .collect();

    (0..n)
        .map(|chapter| {
            // Determine participants for this chapter
            let mut participants = protagonist_ids.clone();
            // Antagonist appears in key chapters
            if chapter == 0 || chapter == n / 2 || chapter >= n - 2 {
                participants.extend(antagonist_ids.iter());
            }

            // Scene type: alternate scene/sequel
            let scene_type = if chapter % 2 == 0 {
                SceneType::ActionScene {
                    goal: None,
                    conflict: None,
                    disaster: None,
                }
            } else {
                SceneType::Sequel {
                    reaction: None,
                    dilemma: None,
                    decision: None,
                }
            };

            // Narration mode varies by story position
            let narration_mode = if chapter == 0 {
                NarrationMode::Scene // Strong opening
            } else if chapter == n - 1 {
                NarrationMode::Scene // Strong ending
            } else if chapter % 5 == 0 {
                NarrationMode::Summary // Occasional summary
            } else {
                NarrationMode::Scene
            };

            // Emotional valence: dips in middle, rises at end (rags-to-riches shape)
            let progress = chapter as f64 / n as f64;
            let valence = if progress < 0.25 {
                0.5 - progress * 0.5 // Initial decline
            } else if progress < 0.75 {
                0.25 + (progress - 0.25) * 0.3 // Gradual recovery
            } else {
                0.4 + (progress - 0.75) * 2.0 // Final rise
            };

            // Commitment instructions
            let planted: Vec<String> = commitments
                .iter()
                .filter(|c| c.setup_chapter == chapter)
                .map(|c| c.id.clone())
                .collect();
            let fulfilled: Vec<String> = commitments
                .iter()
                .filter(|c| c.payoff_chapter == chapter)
                .map(|c| c.id.clone())
                .collect();

            // Knowledge transitions
            let knowledge_transitions: Vec<KnowledgeTransition> = facts
                .iter()
                .filter(|f| f.revealed_in.is_some())
                .filter_map(|_f| {
                    // Placeholder: no transitions by default
                    None
                })
                .collect();

            let causal_predecessors = vec![]; // Linked during materialization

            PlannedSituation {
                id: Uuid::now_v7(),
                chapter,
                summary: format!(
                    "(TBD: chapter {} summary for '{}')",
                    chapter + 1,
                    config.premise
                ),
                participants,
                scene_type,
                causal_predecessors,
                commitments_planted: planted,
                commitments_fulfilled: fulfilled,
                narration_mode,
                emotional_valence: valence,
                game_type: if chapter == n / 2 {
                    Some(GameType::Adversarial)
                } else {
                    None
                },
                knowledge_transitions,
            }
        })
        .collect()
}

/// Validate a plan for internal consistency.
pub fn validate_plan(plan: &NarrativePlan) -> Vec<ConsistencyIssue> {
    let mut issues = Vec::new();

    // Check: every commitment has a payoff chapter within bounds
    for c in &plan.commitments {
        if c.payoff_chapter >= plan.config.chapter_count {
            issues.push(ConsistencyIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::CommitmentOrphaned,
                description: format!(
                    "Commitment '{}' payoff chapter {} exceeds chapter count {}",
                    c.id, c.payoff_chapter, plan.config.chapter_count
                ),
                target_id: None,
                chapter: Some(c.payoff_chapter),
            });
        }
        if c.setup_chapter >= c.payoff_chapter {
            issues.push(ConsistencyIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::CausalInconsistency,
                description: format!(
                    "Commitment '{}' setup (ch {}) must precede payoff (ch {})",
                    c.id, c.setup_chapter, c.payoff_chapter
                ),
                target_id: None,
                chapter: Some(c.setup_chapter),
            });
        }
    }

    // Check: character arcs have valid chapter references
    for arc in &plan.character_arcs {
        if arc.resolution_chapter >= plan.config.chapter_count {
            issues.push(ConsistencyIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::ArcIncomplete,
                description: format!(
                    "Character arc resolution chapter {} exceeds chapter count",
                    arc.resolution_chapter
                ),
                target_id: Some(arc.character_id),
                chapter: Some(arc.resolution_chapter),
            });
        }
        if arc.midpoint_chapter >= arc.resolution_chapter {
            issues.push(ConsistencyIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::ArcIncomplete,
                description: "Midpoint should precede resolution".into(),
                target_id: Some(arc.character_id),
                chapter: Some(arc.midpoint_chapter),
            });
        }
    }

    // Check: no participant references non-existent entity
    let entity_ids: std::collections::HashSet<Uuid> = plan.entities.iter().map(|e| e.id).collect();
    for sit in &plan.fabula {
        for pid in &sit.participants {
            if !entity_ids.contains(pid) {
                issues.push(ConsistencyIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::CausalInconsistency,
                    description: format!(
                        "Situation in chapter {} references non-existent entity {}",
                        sit.chapter, pid
                    ),
                    target_id: Some(*pid),
                    chapter: Some(sit.chapter),
                });
            }
        }
    }

    // Check: facts known_by reference existing entities
    for fact in &plan.facts {
        for eid in &fact.known_by {
            if !entity_ids.contains(eid) {
                issues.push(ConsistencyIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::FactContradiction,
                    description: format!(
                        "Fact '{}' known_by references non-existent entity {}",
                        fact.id, eid
                    ),
                    target_id: Some(*eid),
                    chapter: None,
                });
            }
        }
    }

    issues
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
    fn test_generate_plan_basic() {
        let config = PlanConfig {
            genre: "mystery".into(),
            chapter_count: 12,
            protagonist_count: 1,
            subplot_count: 2,
            commitment_density: 0.5,
            premise: "A detective investigates a disappearance in a small town".into(),
            constraints: Vec::new(),
        };

        let plan = generate_plan(config).unwrap();
        assert_eq!(plan.fabula.len(), 12);
        assert!(!plan.entities.is_empty());
        assert!(!plan.commitments.is_empty());
        // Should have protagonist + antagonist + 2 supporting = 4 entities
        assert_eq!(plan.entities.len(), 4);
    }

    #[test]
    fn test_generate_plan_rejects_empty_premise() {
        let config = PlanConfig {
            premise: String::new(),
            ..PlanConfig::default()
        };
        assert!(generate_plan(config).is_err());
    }

    #[test]
    fn test_validate_plan_clean() {
        let config = PlanConfig {
            premise: "A story about something".into(),
            chapter_count: 10,
            ..PlanConfig::default()
        };
        let plan = generate_plan(config).unwrap();
        let issues = validate_plan(&plan);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Clean plan should have no errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_plan_detects_bad_commitment() {
        let config = PlanConfig {
            premise: "A story".into(),
            chapter_count: 5,
            ..PlanConfig::default()
        };
        let mut plan = generate_plan(config).unwrap();

        // Inject a bad commitment
        plan.commitments.push(PlannedCommitment {
            id: "bad".into(),
            commitment_type: CommitmentType::ChekhovsGun,
            element: "broken gun".into(),
            setup_chapter: 10,  // Out of bounds
            payoff_chapter: 20, // Way out of bounds
            intermediate_progress_chapters: Vec::new(),
        });

        let issues = validate_plan(&plan);
        assert!(issues
            .iter()
            .any(|i| i.category == IssueCategory::CommitmentOrphaned));
    }

    #[test]
    fn test_plan_kv_persistence() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "Test story".into(),
            ..PlanConfig::default()
        };
        let plan = generate_plan(config).unwrap();
        let plan_id = plan.id;

        store_plan(&hg, &plan).unwrap();
        let loaded = load_plan(&hg, &plan_id).unwrap().unwrap();
        assert_eq!(loaded.narrative_id, plan.narrative_id);
        assert_eq!(loaded.fabula.len(), plan.fabula.len());

        delete_plan(&hg, &plan_id).unwrap();
        assert!(load_plan(&hg, &plan_id).unwrap().is_none());
    }

    #[test]
    fn test_commitment_distribution() {
        let config = PlanConfig {
            premise: "Test".into(),
            chapter_count: 20,
            commitment_density: 0.5,
            ..PlanConfig::default()
        };
        let plan = generate_plan(config).unwrap();

        // Verify commitments have setup < payoff
        for c in &plan.commitments {
            assert!(
                c.setup_chapter < c.payoff_chapter,
                "Setup {} should be before payoff {}",
                c.setup_chapter,
                c.payoff_chapter
            );
        }
    }
}
