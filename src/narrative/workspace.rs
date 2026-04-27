//! Workspace summary — the writer's dashboard feed (Sprint W4, v0.49.3).
//!
//! Pure orchestration: gathers recent revisions, recent workshop reports, and
//! computes a handful of "next step" suggestions so the StorywritingHub has a
//! coherent landing page. Zero new state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::continuity::list_pinned_facts;
use crate::narrative::plan as plan_store;
use crate::narrative::revision::list_revisions_tail;
use crate::narrative::workshop::{list_reports as list_workshop_reports, ReportSummary};
use crate::narrative::writer_common::count_words_blocks;
use crate::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceSummary {
    pub narrative_id: String,
    pub counts: WorkspaceCounts,
    /// Up to 5 most recent revisions, newest first.
    pub recent_revisions: Vec<RevisionSummary>,
    /// Up to 3 most recent workshop reports, newest first.
    pub recent_workshop_reports: Vec<ReportSummary>,
    /// Suggested next steps for the writer.
    pub suggestions: Vec<NextStepSuggestion>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceCounts {
    pub situations: usize,
    pub entities: usize,
    /// Number of user-defined `NarrativeArc` records (stored at `ua/`).
    /// Replaces the old ambiguous `arcs` field which conflated this with
    /// Arc-level situations. Deserialization still accepts the old
    /// `arcs` key so older clients keep parsing.
    #[serde(alias = "arcs")]
    pub narrative_arcs: usize,
    /// Count of situations whose `narrative_level == Arc` — distinct from
    /// `narrative_arcs`. Writers who lay out chapters as Arc-level
    /// situations will see this match their chapter count.
    pub arc_situations: usize,
    pub pinned_facts: usize,
    pub total_words: usize,
    pub has_plan: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStepSuggestion {
    pub kind: SuggestionKind,
    /// Short label for a button (~3 words).
    pub label: String,
    /// Longer blurb explaining why.
    pub blurb: String,
    /// Where the Studio should navigate. Relative to the narrative.
    pub href: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionKind {
    DefinePlan,
    GenerateOutline,
    GenerateCharacters,
    DraftChapter,
    RunWorkshop,
    PinFacts,
    ReviewFindings,
    CommitRevision,
}

pub fn get_workspace_summary(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<WorkspaceSummary> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let plan = plan_store::get_plan(hypergraph.store(), narrative_id)?;
    let pinned = list_pinned_facts(hypergraph.store(), narrative_id)?;

    let arc_prefix = crate::hypergraph::keys::user_arc_prefix(narrative_id);
    let arc_pairs = hypergraph.store().prefix_scan(&arc_prefix)?;

    let total_words: usize = situations
        .iter()
        .map(|s| count_words_blocks(&s.raw_content))
        .sum();

    let arc_situations = situations
        .iter()
        .filter(|s| s.narrative_level == NarrativeLevel::Arc)
        .count();

    let counts = WorkspaceCounts {
        situations: situations.len(),
        entities: entities.len(),
        narrative_arcs: arc_pairs.len(),
        arc_situations,
        pinned_facts: pinned.len(),
        total_words,
        has_plan: plan.is_some(),
    };

    // Only the last 5 revisions get their snapshots deserialized — list_revisions_tail
    // skips the wasted per-entry KV reads for older revisions we'd discard anyway.
    let mut recent_revisions = list_revisions_tail(hypergraph.store(), narrative_id, Some(5))?;
    recent_revisions.reverse();

    let mut reports = list_workshop_reports(hypergraph.store(), narrative_id)?;
    reports.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let recent_workshop_reports: Vec<ReportSummary> = reports.into_iter().take(3).collect();

    let suggestions = compute_suggestions(&counts, plan.as_ref(), &recent_workshop_reports);

    Ok(WorkspaceSummary {
        narrative_id: narrative_id.to_string(),
        counts,
        recent_revisions,
        recent_workshop_reports,
        suggestions,
        generated_at: Utc::now(),
    })
}

fn compute_suggestions(
    counts: &WorkspaceCounts,
    plan: Option<&NarrativePlan>,
    recent_reports: &[ReportSummary],
) -> Vec<NextStepSuggestion> {
    let mut out: Vec<NextStepSuggestion> = Vec::new();

    // Cold-start path.
    if !counts.has_plan {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::DefinePlan,
            label: "Define the plan".into(),
            blurb:
                "Set logline, style, and length targets. Generation and edits will respect them."
                    .into(),
            href: "plan".into(),
        });
    }
    if counts.situations == 0 {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::GenerateOutline,
            label: "Generate an outline".into(),
            blurb: "Turn your premise into chapter-by-chapter scaffolding.".into(),
            href: "generate".into(),
        });
    } else if counts.entities == 0 {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::GenerateCharacters,
            label: "Generate characters".into(),
            blurb:
                "You have chapters but no cast yet. Generate a protagonist + supporting characters."
                    .into(),
            href: "generate".into(),
        });
    }

    // Drafting paths.
    if counts.situations > 0 && counts.total_words < 500 {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::DraftChapter,
            label: "Draft Chapter 1".into(),
            blurb:
                "Outline is in place. Start writing or ask the Generate panel for scene sketches."
                    .into(),
            href: "manuscript".into(),
        });
    }

    // Plan-drift / length-drift.
    if let Some(plan) = plan {
        if let Some(target) = plan.length.target_words {
            let ratio = counts.total_words as f64 / target as f64;
            if counts.total_words > 0 && (ratio < 0.6 || ratio > 1.4) {
                out.push(NextStepSuggestion {
                    kind: SuggestionKind::RunWorkshop,
                    label: "Run a pacing workshop".into(),
                    blurb: format!(
                        "Manuscript is {:.0}% of plan target ({}w). Workshop → Pacing will flag the outliers.",
                        ratio * 100.0,
                        target,
                    ),
                    href: "workshop".into(),
                });
            }
        }
    }

    // Pinning facts.
    if counts.pinned_facts == 0 && counts.entities >= 3 {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::PinFacts,
            label: "Pin canonical facts".into(),
            blurb: "Mark character ages, world rules, and other canon so AI edits don't drift."
                .into(),
            href: "cast".into(),
        });
    }

    // Workshop findings waiting.
    if let Some(latest) = recent_reports.first() {
        let total = latest.finding_counts.high + latest.finding_counts.warning;
        if total > 0 {
            out.push(NextStepSuggestion {
                kind: SuggestionKind::ReviewFindings,
                label: "Review workshop findings".into(),
                blurb: format!(
                    "Your last workshop surfaced {} serious finding{}. Apply suggestions or mark resolved.",
                    total,
                    if total == 1 { "" } else { "s" },
                ),
                href: "workshop".into(),
            });
        }
    } else if counts.situations >= 5 {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::RunWorkshop,
            label: "Run your first workshop".into(),
            blurb: "Cheap tier is free and runs in milliseconds. See where pacing or characterization drifts.".into(),
            href: "workshop".into(),
        });
    }

    // Commit reminder.
    if counts.total_words > 1000 {
        out.push(NextStepSuggestion {
            kind: SuggestionKind::CommitRevision,
            label: "Commit a revision".into(),
            blurb: "Snapshot your current state so you can diff and roll back.".into(),
            href: "history".into(),
        });
    }

    // Cap the list so the UI stays tidy.
    out.truncate(4);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrative::registry::NarrativeRegistry;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;
    use uuid::Uuid;

    fn setup() -> Hypergraph {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let reg = NarrativeRegistry::new(store);
        reg.create(Narrative {
            id: "draft".into(),
            title: "Draft".into(),
            genre: None,
            tags: vec![],
            description: None,
            authors: vec![],
            language: None,
            publication_date: None,
            cover_url: None,
            custom_properties: std::collections::HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            source: None,
            project_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
        .unwrap();
        hg
    }

    #[test]
    fn empty_narrative_suggests_plan_and_outline() {
        let hg = setup();
        let s = get_workspace_summary(&hg, "draft").unwrap();
        assert_eq!(s.counts.situations, 0);
        assert!(!s.counts.has_plan);
        let kinds: Vec<SuggestionKind> = s.suggestions.iter().map(|x| x.kind).collect();
        assert!(kinds.contains(&SuggestionKind::DefinePlan));
        assert!(kinds.contains(&SuggestionKind::GenerateOutline));
    }

    #[test]
    fn cap_at_four_suggestions() {
        let hg = setup();
        let s = get_workspace_summary(&hg, "draft").unwrap();
        assert!(s.suggestions.len() <= 4);
    }

    #[test]
    fn pin_suggestion_appears_when_cast_exists_but_no_facts() {
        let hg = setup();
        for i in 0..3 {
            hg.create_entity(Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("Character {}", i)}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 1.0,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: Some(ExtractionMethod::HumanEntered),
                narrative_id: Some("draft".into()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();
        }
        let s = get_workspace_summary(&hg, "draft").unwrap();
        let kinds: Vec<SuggestionKind> = s.suggestions.iter().map(|x| x.kind).collect();
        assert!(kinds.contains(&SuggestionKind::PinFacts));
    }
}
