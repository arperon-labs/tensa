//! Fix suggestions and auto-repair for narrative pathologies.
//!
//! `suggest_fixes` enumerates template fixes for each detected pathology.
//! `apply_fix` mutates the hypergraph to implement a fix (conservative — only
//! lossless edits, no LLM content generation). `auto_repair` runs diagnose →
//! fix → re-diagnose loop until stable or max iterations.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::debug::{
    diagnose_narrative, NarrativePathology, PathologyKind, PathologySeverity,
};

// ─── Types ──────────────────────────────────────────────────

/// Categories of structural fixes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixType {
    AddSituation,
    RemoveSituation,
    MoveSituation,
    AddCommitment,
    FulfillCommitment,
    AbandonCommitment,
    MarkRedHerring,
    AddKnowledgeTransfer,
    ModifyArc,
    AddCausalLink,
    AdjustTemporal,
    ChangeFocalization,
    SplitScene,
}

/// A proposed fix for a single pathology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedFix {
    pub id: Uuid,
    pub pathology_kind: PathologyKind,
    pub fix_type: FixType,
    pub description: String,
    #[serde(default)]
    pub affected_chapters: Vec<usize>,
    #[serde(default)]
    pub affected_entities: Vec<Uuid>,
    #[serde(default)]
    pub affected_situations: Vec<Uuid>,
    pub confidence: f64,
    #[serde(default)]
    pub side_effects: Vec<String>,
    /// Target commitment (for commitment fixes).
    pub target_commitment: Option<Uuid>,
}

/// Result of applying a single fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixApplicationResult {
    pub fix_id: Uuid,
    pub applied: bool,
    pub message: String,
}

/// Full auto-repair report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RepairReport {
    pub narrative_id: String,
    pub iterations: usize,
    pub fixes_applied: Vec<FixApplicationResult>,
    pub starting_health: f64,
    pub ending_health: f64,
    pub remaining_pathologies: usize,
}

// ─── Suggest fixes ──────────────────────────────────────────

/// Generate fix suggestions for a list of detected pathologies.
///
/// Returns one or zero SuggestedFix per pathology, populated with template fixes.
/// Only pathologies with structural (non-LLM) fixes get suggestions.
pub fn suggest_fixes(
    _hg: &Hypergraph,
    pathologies: &[NarrativePathology],
) -> Result<Vec<SuggestedFix>> {
    let mut fixes = Vec::new();
    for p in pathologies {
        if let Some(fix) = suggest_fix_for(p) {
            fixes.push(fix);
        }
    }
    Ok(fixes)
}

fn suggest_fix_for(p: &NarrativePathology) -> Option<SuggestedFix> {
    let chapters: Vec<usize> = p.chapter.into_iter().collect();
    let situations: Vec<Uuid> = p.related_situations.clone();
    let entities: Vec<Uuid> = p.related_entities.clone();
    let commitment = p.location.commitment;

    let (fix_type, desc, confidence) = match p.kind {
        PathologyKind::OrphanedSetup => (
            FixType::MarkRedHerring,
            "Mark commitment as RedHerringResolved to explicitly absorb the dangling setup".to_string(),
            0.6,
        ),
        PathologyKind::UnseededPayoff => (
            FixType::AddCommitment,
            "Plant a setup situation earlier that introduces the resolving element".to_string(),
            0.4,
        ),
        PathologyKind::PrematurePayoff => (
            FixType::MoveSituation,
            "Delay the payoff by reordering downstream situations, or add progress events in between".to_string(),
            0.5,
        ),
        PathologyKind::PromiseOverload => (
            FixType::FulfillCommitment,
            "Resolve or mark-abandoned at least one outstanding commitment before this chapter".to_string(),
            0.5,
        ),
        PathologyKind::PromiseDesert => (
            FixType::AddCommitment,
            "Plant or resolve a commitment in this chapter range to maintain rhythm".to_string(),
            0.4,
        ),
        PathologyKind::ImpossibleKnowledge => (
            FixType::AddKnowledgeTransfer,
            "Insert a situation where the character learns the referenced fact before acting on it".to_string(),
            0.7,
        ),
        PathologyKind::ForgottenKnowledge => (
            FixType::AddSituation,
            "Add a scene where the character acts on or acknowledges the known fact".to_string(),
            0.5,
        ),
        PathologyKind::CausalOrphan => (
            FixType::AddCausalLink,
            "Link this situation to a preceding event that motivates it".to_string(),
            0.6,
        ),
        PathologyKind::CausalContradiction => (
            FixType::AdjustTemporal,
            "Reorder temporal constraints so cause precedes effect".to_string(),
            0.9,
        ),
        PathologyKind::CausalIsland => (
            FixType::AddCausalLink,
            "Add a causal edge connecting this cluster to the main narrative, or remove the cluster".to_string(),
            0.4,
        ),
        PathologyKind::MotivationDiscontinuity => (
            FixType::AddSituation,
            "Add a catalytic event explaining the motivation shift".to_string(),
            0.4,
        ),
        PathologyKind::ArcAbandonment => (
            FixType::ModifyArc,
            "Add a midpoint-turn or dark-night situation to complete the arc".to_string(),
            0.4,
        ),
        PathologyKind::FlatProtagonist => (
            FixType::ModifyArc,
            "Either deepen the transformation, or classify as Flat arc explicitly".to_string(),
            0.5,
        ),
        PathologyKind::MotivationImplausibility => (
            FixType::ModifyArc,
            "Adjust motivation profile leading to this action, or change the action".to_string(),
            0.3,
        ),
        PathologyKind::PacingArrhythmia => (
            FixType::AddSituation,
            "Insert the opposite rhythm beat (scene or sequel) to restore alternation".to_string(),
            0.6,
        ),
        PathologyKind::NarrationModeMonotony => (
            FixType::ChangeFocalization,
            "Switch narration mode in at least one situation within the span".to_string(),
            0.5,
        ),
        PathologyKind::SubplotStarvation => (
            FixType::AddSituation,
            "Add at least one progress scene for this subplot in the gap".to_string(),
            0.5,
        ),
        PathologyKind::SubplotOrphan => (
            FixType::AddSituation,
            "Add a convergence point linking this subplot to the main plot".to_string(),
            0.4,
        ),
        PathologyKind::IronyCollapse => (
            FixType::AddSituation,
            "Add a dedicated revelation scene with dramatic weight".to_string(),
            0.5,
        ),
        PathologyKind::LeakyFocalization => (
            FixType::ChangeFocalization,
            "Either switch to omniscient focalization, or restrict to the focalizer's perception".to_string(),
            0.4,
        ),
        PathologyKind::TemporalImpossibility => (
            FixType::AdjustTemporal,
            "Relax one of the conflicting Allen constraints".to_string(),
            0.8,
        ),
        PathologyKind::AnachronismRisk => (
            FixType::MoveSituation,
            "Move entity introduction earlier, or remove the reference".to_string(),
            0.5,
        ),
    };

    Some(SuggestedFix {
        id: Uuid::now_v7(),
        pathology_kind: p.kind.clone(),
        fix_type,
        description: desc,
        affected_chapters: chapters,
        affected_entities: entities,
        affected_situations: situations,
        confidence,
        side_effects: Vec::new(),
        target_commitment: commitment,
    })
}

// ─── Apply fixes ────────────────────────────────────────────

/// Apply a single suggested fix. Conservative: only performs safe,
/// reversible mutations on the hypergraph. Returns Ok(true) on success.
fn set_commitment_status(
    hg: &Hypergraph,
    narrative_id: &str,
    fix: &SuggestedFix,
    new_status: crate::narrative::commitments::CommitmentStatus,
    label: &str,
) -> Result<FixApplicationResult> {
    if let Some(cid) = fix.target_commitment {
        if let Some(mut c) = crate::narrative::commitments::load_commitment(hg, narrative_id, &cid)?
        {
            c.status = new_status;
            c.updated_at = chrono::Utc::now();
            crate::narrative::commitments::store_commitment(hg, &c)?;
            return Ok(FixApplicationResult {
                fix_id: fix.id,
                applied: true,
                message: format!("Commitment {} marked {}", cid, label),
            });
        }
    }
    Ok(FixApplicationResult {
        fix_id: fix.id,
        applied: false,
        message: "No target commitment found".into(),
    })
}

pub fn apply_fix(
    hg: &Hypergraph,
    narrative_id: &str,
    fix: &SuggestedFix,
) -> Result<FixApplicationResult> {
    use crate::narrative::commitments::CommitmentStatus;
    match fix.fix_type {
        FixType::MarkRedHerring => set_commitment_status(
            hg,
            narrative_id,
            fix,
            CommitmentStatus::RedHerringResolved,
            "RedHerringResolved",
        ),
        FixType::AbandonCommitment => set_commitment_status(
            hg,
            narrative_id,
            fix,
            CommitmentStatus::Abandoned,
            "Abandoned",
        ),
        _ => {
            // Most fixes require LLM-assisted content generation; we don't apply them here.
            Ok(FixApplicationResult {
                fix_id: fix.id,
                applied: false,
                message: format!(
                    "Fix type {:?} requires manual or LLM-assisted application",
                    fix.fix_type
                ),
            })
        }
    }
}

// ─── Auto repair ────────────────────────────────────────────

/// Iteratively diagnose → apply safe fixes → re-diagnose, up to `max_iterations`.
///
/// Only applies fixes with confidence ≥ 0.5 and severity ≤ `max_severity`.
pub fn auto_repair(
    hg: &Hypergraph,
    narrative_id: &str,
    max_severity: PathologySeverity,
    max_iterations: usize,
) -> Result<RepairReport> {
    let mut diag = diagnose_narrative(hg, narrative_id)?;
    let starting_health = diag.health_score;
    let mut report = RepairReport {
        narrative_id: narrative_id.to_string(),
        iterations: 0,
        fixes_applied: Vec::new(),
        starting_health,
        ending_health: starting_health,
        remaining_pathologies: diag.pathologies.len(),
    };
    let mut last_count = diag.pathologies.len();
    for i in 0..max_iterations {
        if diag.pathologies.is_empty() {
            break;
        }
        let fixes: Vec<SuggestedFix> = diag
            .pathologies
            .iter()
            .filter(|p| p.severity <= max_severity)
            .filter_map(suggest_fix_for)
            .filter(|f| f.confidence >= 0.5)
            .collect();

        let mut progressed = false;
        for fix in &fixes {
            let res = apply_fix(hg, narrative_id, fix)?;
            if res.applied {
                progressed = true;
            }
            report.fixes_applied.push(res);
        }

        report.iterations = i + 1;
        if !progressed {
            break;
        }
        diag = diagnose_narrative(hg, narrative_id)?;
        if diag.pathologies.len() >= last_count {
            break;
        }
        last_count = diag.pathologies.len();
    }
    report.ending_health = diag.health_score;
    report.remaining_pathologies = diag.pathologies.len();
    Ok(report)
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
    fn test_suggest_fixes_produces_one_per_pathology() {
        let hg = test_hg();
        let pathologies = vec![NarrativePathology::new(
            PathologyKind::OrphanedSetup,
            PathologySeverity::Warning,
            "test",
        )];
        let fixes = suggest_fixes(&hg, &pathologies).unwrap();
        assert_eq!(fixes.len(), 1);
        assert_eq!(fixes[0].pathology_kind, PathologyKind::OrphanedSetup);
    }

    #[test]
    fn test_apply_mark_red_herring() {
        let hg = test_hg();
        let nid = "auto-repair-test";
        let commitment = crate::narrative::commitments::NarrativeCommitment {
            id: Uuid::now_v7(),
            narrative_id: nid.into(),
            commitment_type: crate::narrative::commitments::CommitmentType::ChekhovsGun,
            setup_event: Uuid::nil(),
            setup_chapter: 1,
            setup_salience: 2.0,
            status: crate::narrative::commitments::CommitmentStatus::Planted,
            payoff_event: None,
            payoff_chapter: None,
            payoff_distance: None,
            causal_chain: vec![],
            description: "".into(),
            tracked_element: "x".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        crate::narrative::commitments::store_commitment(&hg, &commitment).unwrap();

        let fix = SuggestedFix {
            id: Uuid::now_v7(),
            pathology_kind: PathologyKind::OrphanedSetup,
            fix_type: FixType::MarkRedHerring,
            description: "mark".into(),
            affected_chapters: vec![1],
            affected_entities: vec![],
            affected_situations: vec![],
            confidence: 0.6,
            side_effects: vec![],
            target_commitment: Some(commitment.id),
        };
        let result = apply_fix(&hg, "auto-repair-test", &fix).unwrap();
        assert!(result.applied);

        let loaded =
            crate::narrative::commitments::load_commitment(&hg, "auto-repair-test", &commitment.id)
                .unwrap()
                .unwrap();
        assert_eq!(
            loaded.status,
            crate::narrative::commitments::CommitmentStatus::RedHerringResolved
        );
    }

    #[test]
    fn test_auto_repair_resolves_orphaned_setups() {
        let hg = test_hg();
        let nid = "auto-fix";
        for _ in 0..3 {
            let commitment = crate::narrative::commitments::NarrativeCommitment {
                id: Uuid::now_v7(),
                narrative_id: nid.into(),
                commitment_type: crate::narrative::commitments::CommitmentType::ChekhovsGun,
                setup_event: Uuid::nil(),
                setup_chapter: 1,
                setup_salience: 2.0,
                status: crate::narrative::commitments::CommitmentStatus::Planted,
                payoff_event: None,
                payoff_chapter: None,
                payoff_distance: None,
                causal_chain: vec![],
                description: "".into(),
                tracked_element: "x".into(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            };
            crate::narrative::commitments::store_commitment(&hg, &commitment).unwrap();
        }
        let report = auto_repair(&hg, nid, PathologySeverity::Warning, 5).unwrap();
        // Should apply at least 3 MarkRedHerring fixes
        let applied = report.fixes_applied.iter().filter(|r| r.applied).count();
        assert!(applied >= 3, "expected ≥3 applied fixes, got {}", applied);
        assert!(report.ending_health > report.starting_health - 0.001);
    }
}
