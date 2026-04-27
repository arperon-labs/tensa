//! Narrative debugging — structural pathology detection ("compiler warnings for stories").
//!
//! `diagnose_narrative(narrative_id)` runs all narrative analysis engines and
//! reports structural failures across five families: commitment, knowledge,
//! causal, motivation/arc, and pacing/rhythm/temporal.
//!
//! Companion module [`debug_fixes`](crate::narrative::debug_fixes) provides
//! `suggest_fixes`, `apply_fix`, and `auto_repair`.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// Severity of a narrative pathology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PathologySeverity {
    /// Structural error — breaks narrative logic. Must fix.
    Error,
    /// Structural warning — weakens narrative. Should fix.
    Warning,
    /// Stylistic concern — consider fixing.
    Info,
    /// Observation — awareness only.
    Note,
}

/// Category of narrative pathology (18 detector types, D10.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathologyKind {
    // Commitment pathologies
    OrphanedSetup,
    UnseededPayoff,
    PrematurePayoff,
    PromiseOverload,
    PromiseDesert,
    // Knowledge pathologies
    ImpossibleKnowledge,
    ForgottenKnowledge,
    IronyCollapse,
    LeakyFocalization,
    // Causal pathologies
    CausalOrphan,
    CausalContradiction,
    CausalIsland,
    // Motivation / arc
    MotivationDiscontinuity,
    ArcAbandonment,
    FlatProtagonist,
    MotivationImplausibility,
    // Pacing / rhythm
    PacingArrhythmia,
    NarrationModeMonotony,
    SubplotStarvation,
    SubplotOrphan,
    // Temporal
    TemporalImpossibility,
    AnachronismRisk,
}

/// Structured location for a pathology.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathologyLocation {
    pub chapter: Option<usize>,
    pub situation: Option<Uuid>,
    pub entity: Option<Uuid>,
    pub commitment: Option<Uuid>,
    /// Chapter range for multi-chapter pathologies (inclusive).
    pub span: Option<(usize, usize)>,
}

/// Where the evidence for a pathology came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceSource {
    Commitments,
    DramaticIrony,
    CharacterArcs,
    SceneSequel,
    Causal,
    Subplots,
    Temporal,
    Motivation,
    Focalization,
}

/// A single piece of evidence supporting the pathology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub label: String,
    pub source: EvidenceSource,
    pub data: serde_json::Value,
}

/// A single narrative pathology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativePathology {
    pub kind: PathologyKind,
    pub severity: PathologySeverity,
    pub description: String,
    /// Entity or situation primarily involved (legacy; prefer `location`).
    pub target_id: Option<Uuid>,
    /// Chapter where the problem manifests (legacy; prefer `location`).
    pub chapter: Option<usize>,
    /// Suggested fix description.
    pub suggestion: Option<String>,
    /// Structured location (new in D10).
    #[serde(default)]
    pub location: PathologyLocation,
    /// Evidence items linking to source analysis modules.
    #[serde(default)]
    pub evidence: Vec<EvidenceItem>,
    /// Related entities (for fix planning).
    #[serde(default)]
    pub related_entities: Vec<Uuid>,
    /// Related situations (for fix planning).
    #[serde(default)]
    pub related_situations: Vec<Uuid>,
}

impl NarrativePathology {
    pub fn new(
        kind: PathologyKind,
        severity: PathologySeverity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            severity,
            description: description.into(),
            target_id: None,
            chapter: None,
            suggestion: None,
            location: PathologyLocation::default(),
            evidence: Vec::new(),
            related_entities: Vec::new(),
            related_situations: Vec::new(),
        }
    }
    pub fn at_chapter(mut self, chapter: usize) -> Self {
        self.chapter = Some(chapter);
        self.location.chapter = Some(chapter);
        self
    }
    pub fn with_situation(mut self, id: Uuid) -> Self {
        self.target_id = Some(id);
        self.location.situation = Some(id);
        self
    }
    pub fn with_entity(mut self, id: Uuid) -> Self {
        self.location.entity = Some(id);
        self
    }
    pub fn with_commitment(mut self, id: Uuid) -> Self {
        self.location.commitment = Some(id);
        self
    }
    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }
    pub fn with_evidence(mut self, e: EvidenceItem) -> Self {
        self.evidence.push(e);
        self
    }
}

/// Genre preset for tuning thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenrePreset {
    Generic,
    Thriller,
    LiteraryFiction,
    EpicFantasy,
    Mystery,
}

impl Default for GenrePreset {
    fn default() -> Self {
        GenrePreset::Generic
    }
}

impl std::str::FromStr for GenrePreset {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "thriller" => GenrePreset::Thriller,
            "literary" | "literary_fiction" => GenrePreset::LiteraryFiction,
            "epic" | "epic_fantasy" => GenrePreset::EpicFantasy,
            "mystery" => GenrePreset::Mystery,
            _ => GenrePreset::Generic,
        })
    }
}

impl std::str::FromStr for PathologySeverity {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "error" => PathologySeverity::Error,
            "warning" => PathologySeverity::Warning,
            "info" => PathologySeverity::Info,
            "note" => PathologySeverity::Note,
            _ => return Err(()),
        })
    }
}

/// Diagnostic configuration (thresholds + genre + suppression).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticConfig {
    pub genre: GenrePreset,
    pub max_simultaneous_commitments: usize,
    pub min_payoff_distance: usize,
    pub max_action_streak: usize,
    pub max_sequel_streak: usize,
    pub min_promise_desert_len: usize,
    pub narration_monotony_ratio: f64,
    pub narration_monotony_min_span: usize,
    pub subplot_starvation_gap: usize,
    pub motivation_jump_sigmas: f64,
    pub arc_abandonment_completeness: f64,
    pub flat_protagonist_cosine: f64,
    pub suppress: Vec<PathologyKind>,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        DiagnosticConfig {
            genre: GenrePreset::Generic,
            max_simultaneous_commitments: 10,
            min_payoff_distance: 2,
            max_action_streak: 4,
            max_sequel_streak: 3,
            min_promise_desert_len: 3,
            narration_monotony_ratio: 0.8,
            narration_monotony_min_span: 5,
            subplot_starvation_gap: 5,
            motivation_jump_sigmas: 2.0,
            arc_abandonment_completeness: 0.5,
            flat_protagonist_cosine: 0.95,
            suppress: Vec::new(),
        }
    }
}

impl DiagnosticConfig {
    /// Apply a genre preset, overriding thresholds.
    pub fn for_genre(genre: GenrePreset) -> Self {
        let mut c = DiagnosticConfig {
            genre,
            ..Default::default()
        };
        match genre {
            GenrePreset::Generic => {}
            GenrePreset::Thriller => {
                c.max_simultaneous_commitments = 8;
                c.max_action_streak = 3;
                c.max_sequel_streak = 2;
                c.min_payoff_distance = 2;
                c.suppress.push(PathologyKind::FlatProtagonist);
            }
            GenrePreset::LiteraryFiction => {
                c.max_action_streak = 6;
                c.max_sequel_streak = 6;
                c.min_promise_desert_len = 6;
                c.motivation_jump_sigmas = 1.5;
                c.arc_abandonment_completeness = 0.6;
                c.suppress.push(PathologyKind::SubplotOrphan);
            }
            GenrePreset::EpicFantasy => {
                c.max_simultaneous_commitments = 14;
                c.subplot_starvation_gap = 4;
            }
            GenrePreset::Mystery => {
                c.max_simultaneous_commitments = 12;
                c.suppress.push(PathologyKind::CausalIsland);
            }
        }
        c
    }
}

/// Summary counts for a diagnosis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticSummary {
    pub errors: usize,
    pub warnings: usize,
    pub infos: usize,
    pub notes: usize,
    pub health_score: f64,
    pub worst_chapter: Option<usize>,
    pub cleanest_chapter: Option<usize>,
}

/// Full diagnosis result (aka `DiagnosticReport`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisResult {
    pub narrative_id: String,
    pub pathologies: Vec<NarrativePathology>,
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    /// Overall structural health score (1.0 = no issues, 0.0 = many).
    pub health_score: f64,
    #[serde(default)]
    pub note_count: usize,
    #[serde(default)]
    pub summary: DiagnosticSummary,
}

// ─── Diagnosis ──────────────────────────────────────────────

/// Run all structural checks using the default config.
pub fn diagnose_narrative(hg: &Hypergraph, narrative_id: &str) -> Result<DiagnosisResult> {
    diagnose_narrative_with(hg, narrative_id, &DiagnosticConfig::default())
}

/// Run all structural checks using a specific config.
pub fn diagnose_narrative_with(
    hg: &Hypergraph,
    narrative_id: &str,
    cfg: &DiagnosticConfig,
) -> Result<DiagnosisResult> {
    let mut pathologies = Vec::new();

    // Commitment pathologies
    check_orphaned_setups(hg, narrative_id, cfg, &mut pathologies)?;
    check_premature_payoff(hg, narrative_id, cfg, &mut pathologies)?;
    check_promise_density(hg, narrative_id, cfg, &mut pathologies)?;

    // Knowledge pathologies
    check_impossible_knowledge(hg, narrative_id, &mut pathologies)?;
    check_irony_collapse(hg, narrative_id, &mut pathologies)?;

    // Causal pathologies — share one fetch + index across all three detectors
    // instead of each re-scanning `cr/` per situation.
    let causal_situations = hg.list_situations_by_narrative(narrative_id)?;
    let causal_index =
        crate::narrative::causal_helpers::CausalIndex::build(hg, &causal_situations)?;
    check_causal_orphans(&causal_situations, &causal_index, &mut pathologies);
    check_causal_contradiction(&causal_situations, &causal_index, &mut pathologies);
    check_causal_islands(&causal_situations, &causal_index, &mut pathologies);

    // Motivation / arc pathologies
    check_arc_abandonment(hg, narrative_id, cfg, &mut pathologies)?;
    check_flat_protagonist(hg, narrative_id, cfg, &mut pathologies)?;

    // Pacing / rhythm / subplot pathologies
    check_pacing_arrhythmia(hg, narrative_id, cfg, &mut pathologies)?;
    check_narration_monotony(hg, narrative_id, cfg, &mut pathologies)?;
    check_subplots(hg, narrative_id, cfg, &mut pathologies)?;

    // Filter suppressed kinds
    if !cfg.suppress.is_empty() {
        pathologies.retain(|p| !cfg.suppress.contains(&p.kind));
    }

    // Sort by severity then chapter
    pathologies.sort_by(|a, b| {
        a.severity.cmp(&b.severity).then(
            a.chapter
                .unwrap_or(usize::MAX)
                .cmp(&b.chapter.unwrap_or(usize::MAX)),
        )
    });

    let total_situations = hg.list_situations_by_narrative(narrative_id)?.len().max(1);
    Ok(finalize(narrative_id, pathologies, total_situations))
}

/// Diagnose a single chapter (incremental).
pub fn diagnose_chapter(
    hg: &Hypergraph,
    narrative_id: &str,
    chapter: usize,
) -> Result<Vec<NarrativePathology>> {
    let full = diagnose_narrative(hg, narrative_id)?;
    Ok(full
        .pathologies
        .into_iter()
        .filter(|p| {
            p.chapter == Some(chapter)
                || matches!(p.location.span, Some((s, e)) if chapter >= s && chapter <= e)
        })
        .collect())
}

fn finalize(
    narrative_id: &str,
    pathologies: Vec<NarrativePathology>,
    total_situations: usize,
) -> DiagnosisResult {
    let mut errors = 0;
    let mut warnings = 0;
    let mut infos = 0;
    let mut notes = 0;
    let mut chapter_counts: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::new();
    for p in &pathologies {
        match p.severity {
            PathologySeverity::Error => errors += 1,
            PathologySeverity::Warning => warnings += 1,
            PathologySeverity::Info => infos += 1,
            PathologySeverity::Note => notes += 1,
        }
        if let Some(c) = p.chapter {
            let w = match p.severity {
                PathologySeverity::Error => 5,
                PathologySeverity::Warning => 2,
                PathologySeverity::Info => 1,
                PathologySeverity::Note => 0,
            };
            *chapter_counts.entry(c).or_insert(0) += w;
        }
    }
    let penalty = (errors as f64 * 0.15 + warnings as f64 * 0.05 + infos as f64 * 0.01)
        / total_situations as f64;
    let health_score = (1.0 - penalty).clamp(0.0, 1.0);

    let worst_chapter = chapter_counts
        .iter()
        .max_by_key(|(_, w)| **w)
        .map(|(c, _)| *c);
    let cleanest_chapter = chapter_counts
        .iter()
        .filter(|(_, w)| **w == 0)
        .map(|(c, _)| *c)
        .min();

    let summary = DiagnosticSummary {
        errors,
        warnings,
        infos,
        notes,
        health_score,
        worst_chapter,
        cleanest_chapter,
    };

    DiagnosisResult {
        narrative_id: narrative_id.to_string(),
        pathologies,
        error_count: errors,
        warning_count: warnings,
        info_count: infos,
        note_count: notes,
        health_score,
        summary,
    }
}

// ─── Commitment detectors ───────────────────────────────────

fn check_orphaned_setups(
    hg: &Hypergraph,
    nid: &str,
    _cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let commitments = crate::narrative::commitments::list_commitments(hg, nid)?;
    for c in &commitments {
        use crate::narrative::commitments::CommitmentStatus::*;
        let (severity, kind) = match c.status {
            Planted | InProgress => (PathologySeverity::Warning, PathologyKind::OrphanedSetup),
            Abandoned => (PathologySeverity::Error, PathologyKind::OrphanedSetup),
            _ => continue,
        };
        let p = NarrativePathology::new(
            kind,
            severity,
            format!(
                "Commitment '{}' (type: {:?}) planted in chapter {} has status {:?}",
                c.tracked_element, c.commitment_type, c.setup_chapter, c.status
            ),
        )
        .at_chapter(c.setup_chapter)
        .with_situation(c.setup_event)
        .with_commitment(c.id)
        .with_suggestion(format!(
            "Add a payoff for '{}' or explicitly mark as RedHerring",
            c.tracked_element
        ))
        .with_evidence(EvidenceItem {
            label: "commitment".into(),
            source: EvidenceSource::Commitments,
            data: serde_json::json!({
                "element": c.tracked_element,
                "setup_salience": c.setup_salience,
                "status": format!("{:?}", c.status),
            }),
        });
        out.push(p);
    }
    Ok(())
}

fn check_premature_payoff(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let commitments = crate::narrative::commitments::list_commitments(hg, nid)?;
    for c in &commitments {
        if !matches!(
            c.status,
            crate::narrative::commitments::CommitmentStatus::Fulfilled
                | crate::narrative::commitments::CommitmentStatus::Subverted
        ) {
            continue;
        }
        if let Some(dist) = c.payoff_distance {
            if dist < cfg.min_payoff_distance && c.setup_salience >= 2.0 {
                let p = NarrativePathology::new(
                    PathologyKind::PrematurePayoff,
                    PathologySeverity::Info,
                    format!(
                        "Commitment '{}' set up in chapter {} pays off in {} chapter(s)",
                        c.tracked_element, c.setup_chapter, dist
                    ),
                )
                .at_chapter(c.setup_chapter)
                .with_commitment(c.id)
                .with_suggestion("Consider delaying the payoff to build anticipation.");
                out.push(p);
            }
        }
    }
    Ok(())
}

fn check_promise_density(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let rhythm = match crate::narrative::commitments::compute_promise_rhythm(hg, nid) {
        Ok(r) => r,
        Err(_) => return Ok(()),
    };
    // PromiseOverload
    if let Some(peak) = rhythm
        .chapters
        .iter()
        .max_by_key(|c| c.promises_outstanding)
    {
        if peak.promises_outstanding > cfg.max_simultaneous_commitments {
            let p = NarrativePathology::new(
                PathologyKind::PromiseOverload,
                PathologySeverity::Warning,
                format!(
                    "Chapter {} has {} simultaneous unresolved commitments (threshold: {})",
                    peak.chapter, peak.promises_outstanding, cfg.max_simultaneous_commitments
                ),
            )
            .at_chapter(peak.chapter)
            .with_suggestion("Resolve some outstanding setups before introducing new ones.");
            out.push(p);
        }
    }
    // PromiseDesert: 3+ consecutive chapters with zero new and zero fulfilled
    let mut streak_start: Option<usize> = None;
    let mut streak_len = 0;
    for r in &rhythm.chapters {
        if r.new_promises == 0 && r.promises_fulfilled == 0 {
            if streak_start.is_none() {
                streak_start = Some(r.chapter);
            }
            streak_len += 1;
        } else {
            if streak_len >= cfg.min_promise_desert_len {
                let start = streak_start.unwrap_or(0);
                let mut p = NarrativePathology::new(
                    PathologyKind::PromiseDesert,
                    PathologySeverity::Info,
                    format!(
                        "Chapters {}–{} plant no new promises and resolve none",
                        start,
                        r.chapter - 1
                    ),
                )
                .at_chapter(start)
                .with_suggestion("Add a setup or payoff to maintain momentum.");
                p.location.span = Some((start, r.chapter - 1));
                out.push(p);
            }
            streak_start = None;
            streak_len = 0;
        }
    }
    Ok(())
}

// ─── Knowledge detectors ────────────────────────────────────

fn check_impossible_knowledge(
    hg: &Hypergraph,
    nid: &str,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let irony_map = match crate::narrative::dramatic_irony::compute_dramatic_irony_map(hg, nid) {
        Ok(m) => m,
        Err(_) => return Ok(()),
    };
    for event in &irony_map.events {
        if event.irony_intensity > 5.0 {
            let p = NarrativePathology::new(
                PathologyKind::ImpossibleKnowledge,
                PathologySeverity::Info,
                format!(
                    "High knowledge gap ({:.0} facts) at situation {} for character {} — verify character doesn't act on unknown info",
                    event.irony_intensity, event.situation_id, event.character_id
                ),
            )
            .with_situation(event.situation_id)
            .with_entity(event.character_id)
            .with_suggestion("Check character actions only reference information available to them.")
            .with_evidence(EvidenceItem {
                label: "irony_event".into(),
                source: EvidenceSource::DramaticIrony,
                data: serde_json::json!({ "intensity": event.irony_intensity }),
            });
            out.push(p);
        }
    }
    Ok(())
}

fn check_irony_collapse(
    hg: &Hypergraph,
    nid: &str,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let irony_map = match crate::narrative::dramatic_irony::load_irony_map(hg, nid)? {
        Some(m) => m,
        None => return Ok(()),
    };
    if irony_map.irony_curve.len() < 3 {
        return Ok(());
    }
    for i in 2..irony_map.irony_curve.len() {
        let prev = irony_map.irony_curve[i - 1];
        let curr = irony_map.irony_curve[i];
        let two_back = irony_map.irony_curve[i - 2];
        if two_back < prev && prev > 0.0 && curr < prev * 0.5 {
            out.push(
                NarrativePathology::new(
                    PathologyKind::IronyCollapse,
                    PathologySeverity::Warning,
                    format!(
                        "Dramatic irony built to {:.1} then dropped to {:.1} at chapter {}",
                        prev, curr, i
                    ),
                )
                .at_chapter(i)
                .with_suggestion("Ensure the irony payoff scene has sufficient dramatic impact."),
            );
        }
    }
    Ok(())
}

// ─── Causal detectors ───────────────────────────────────────

fn check_causal_orphans(
    situations: &[crate::types::Situation],
    index: &crate::narrative::causal_helpers::CausalIndex,
    out: &mut Vec<NarrativePathology>,
) {
    let mut sorted: Vec<&crate::types::Situation> = situations.iter().collect();
    sorted.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    let significant_count = sorted
        .iter()
        .filter(|s| is_significant_level(s.narrative_level))
        .count();
    // Total edges across the narrative — indexed scan plus legacy inline vec.
    let total_edges: usize =
        index.total_edges() + sorted.iter().map(|s| s.causes.len()).sum::<usize>();

    // Writer-generated narratives legitimately have zero causal edges;
    // flagging every scene floods Workshop, so summarise once instead.
    if total_edges == 0 && significant_count > 2 {
        out.push(
            NarrativePathology::new(
                PathologyKind::CausalOrphan,
                PathologySeverity::Info,
                format!(
                    "Narrative has no causal graph ({} significant situations, 0 causal links)",
                    significant_count
                ),
            )
            .with_suggestion(
                "Add causal links between scenes (add_causal_link / Studio → Relations), or ignore if causal structure is tracked elsewhere.",
            ),
        );
        return;
    }

    for (idx, sit) in sorted.iter().enumerate().skip(1) {
        if !index.antecedents_of(&sit.id).is_empty() || !sit.causes.is_empty() {
            continue;
        }
        if !is_significant_level(sit.narrative_level) {
            continue;
        }
        out.push(
            NarrativePathology::new(
                PathologyKind::CausalOrphan,
                PathologySeverity::Warning,
                format!(
                    "Situation '{}' at position {} has no causal predecessor",
                    sit.name.as_deref().unwrap_or("(unnamed)"),
                    idx
                ),
            )
            .at_chapter(idx)
            .with_situation(sit.id)
            .with_suggestion("Add a causal link explaining what led to this event."),
        );
    }
}

fn is_significant_level(level: crate::types::NarrativeLevel) -> bool {
    matches!(
        level,
        crate::types::NarrativeLevel::Scene
            | crate::types::NarrativeLevel::Sequence
            | crate::types::NarrativeLevel::Arc
    )
}

fn check_causal_contradiction(
    situations: &[crate::types::Situation],
    index: &crate::narrative::causal_helpers::CausalIndex,
    out: &mut Vec<NarrativePathology>,
) {
    use std::collections::HashMap;
    let by_id: HashMap<Uuid, &crate::types::Situation> =
        situations.iter().map(|s| (s.id, s)).collect();
    for sit in situations {
        let edges = index
            .antecedents_of(&sit.id)
            .iter()
            .cloned()
            .chain(sit.causes.iter().cloned());
        for cause in edges {
            let from = match by_id.get(&cause.from_situation) {
                Some(s) => s,
                None => continue,
            };
            // Effect's start precedes cause's end (strict)
            if let (Some(cause_start), Some(effect_start)) =
                (from.temporal.start, sit.temporal.start)
            {
                if effect_start < cause_start {
                    out.push(
                        NarrativePathology::new(
                            PathologyKind::CausalContradiction,
                            PathologySeverity::Error,
                            format!(
                                "Causal edge {} → {} but effect starts before cause",
                                from.name.as_deref().unwrap_or("(unnamed)"),
                                sit.name.as_deref().unwrap_or("(unnamed)"),
                            ),
                        )
                        .with_situation(sit.id)
                        .with_suggestion("Reorder events or relax temporal constraints.")
                        .with_evidence(EvidenceItem {
                            label: "temporal_order".into(),
                            source: EvidenceSource::Temporal,
                            data: serde_json::json!({
                                "cause_start": cause_start,
                                "effect_start": effect_start,
                            }),
                        }),
                    );
                }
            }
        }
    }
}

fn check_causal_islands(
    situations: &[crate::types::Situation],
    index: &crate::narrative::causal_helpers::CausalIndex,
    out: &mut Vec<NarrativePathology>,
) {
    use std::collections::{HashMap, HashSet};
    if situations.len() < 3 {
        return;
    }
    let mut adj: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for s in situations {
        adj.entry(s.id).or_default();
        for c in index.antecedents_of(&s.id).iter().chain(s.causes.iter()) {
            adj.entry(c.from_situation).or_default().push(s.id);
            adj.entry(s.id).or_default().push(c.from_situation);
        }
    }
    let mut visited: HashSet<Uuid> = HashSet::new();
    let mut components: Vec<Vec<Uuid>> = Vec::new();
    for s in situations {
        if visited.contains(&s.id) {
            continue;
        }
        let mut comp = Vec::new();
        let mut stack = vec![s.id];
        while let Some(v) = stack.pop() {
            if !visited.insert(v) {
                continue;
            }
            comp.push(v);
            if let Some(ns) = adj.get(&v) {
                stack.extend(ns);
            }
        }
        components.push(comp);
    }
    if components.len() > 1 {
        // Largest component is "main plot"; others are islands
        components.sort_by_key(|c| std::cmp::Reverse(c.len()));
        for island in components.iter().skip(1) {
            if island.len() >= 2 {
                let first_id = island[0];
                out.push(
                    NarrativePathology::new(
                        PathologyKind::CausalIsland,
                        PathologySeverity::Info,
                        format!(
                            "Causal island of {} situations disconnected from the main graph",
                            island.len()
                        ),
                    )
                    .with_situation(first_id)
                    .with_suggestion("Connect this sub-sequence to the main causal chain, or evaluate whether it's necessary."),
                );
            }
        }
    }
}

// ─── Arc / motivation detectors ─────────────────────────────

fn check_arc_abandonment(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let arcs = crate::narrative::character_arcs::list_character_arcs(hg, nid)?;
    for arc in &arcs {
        let is_flat = matches!(
            arc.arc_type,
            crate::narrative::character_arcs::ArcType::Flat
        );
        if is_flat {
            continue;
        }
        if arc.completeness < cfg.arc_abandonment_completeness
            && !arc.motivation_trajectory.is_empty()
        {
            out.push(
                NarrativePathology::new(
                    PathologyKind::ArcAbandonment,
                    PathologySeverity::Warning,
                    format!(
                        "Character {} has arc type {:?} but completeness is only {:.0}%",
                        arc.character_id,
                        arc.arc_type,
                        arc.completeness * 100.0
                    ),
                )
                .with_entity(arc.character_id)
                .with_suggestion("Add a midpoint turn and/or dark night moment."),
            );
        }
    }
    Ok(())
}

fn check_flat_protagonist(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let arcs = crate::narrative::character_arcs::list_character_arcs(hg, nid)?;
    for arc in &arcs {
        if matches!(
            arc.arc_type,
            crate::narrative::character_arcs::ArcType::Flat
        ) {
            continue;
        }
        if arc.motivation_trajectory.len() < 2 {
            continue;
        }
        let first = arc.motivation_trajectory[0].1;
        let last = arc.motivation_trajectory[arc.motivation_trajectory.len() - 1].1;
        let delta = (last - first).abs();
        if delta < (1.0 - cfg.flat_protagonist_cosine) {
            out.push(
                NarrativePathology::new(
                    PathologyKind::FlatProtagonist,
                    PathologySeverity::Info,
                    format!(
                        "Character {} undergoes minimal change (|Δmotivation| = {:.3})",
                        arc.character_id, delta
                    ),
                )
                .with_entity(arc.character_id)
                .with_suggestion(
                    "Deepen the transformation or clarify intent by switching to ArcType::Flat.",
                ),
            );
        }
    }
    Ok(())
}

// ─── Pacing detectors ───────────────────────────────────────

fn check_pacing_arrhythmia(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let analysis = crate::narrative::scene_sequel::analyze_scene_sequel(hg, nid)?;
    if analysis.max_action_streak >= cfg.max_action_streak {
        out.push(
            NarrativePathology::new(
                PathologyKind::PacingArrhythmia,
                PathologySeverity::Warning,
                format!(
                    "{} consecutive action scenes without a reflective sequel",
                    analysis.max_action_streak
                ),
            )
            .with_suggestion("Insert a reflective sequel (reaction/dilemma/decision)."),
        );
    }
    if analysis.max_sequel_streak >= cfg.max_sequel_streak {
        out.push(
            NarrativePathology::new(
                PathologyKind::PacingArrhythmia,
                PathologySeverity::Warning,
                format!(
                    "{} consecutive sequel scenes without action",
                    analysis.max_sequel_streak
                ),
            )
            .with_suggestion("Insert an action scene (goal/conflict/disaster)."),
        );
    }
    Ok(())
}

fn check_narration_monotony(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let sjuzet = match crate::narrative::fabula_sjuzet::extract_sjuzet(hg, nid) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    if sjuzet.segments.len() < cfg.narration_monotony_min_span {
        return Ok(());
    }
    use std::collections::HashMap;
    let mut mode_count: HashMap<String, usize> = HashMap::new();
    for seg in &sjuzet.segments {
        *mode_count
            .entry(format!("{:?}", seg.narration_mode))
            .or_insert(0) += 1;
    }
    let total = sjuzet.segments.len();
    if let Some((mode, count)) = mode_count.iter().max_by_key(|(_, c)| **c) {
        let ratio = *count as f64 / total as f64;
        if ratio >= cfg.narration_monotony_ratio && total >= cfg.narration_monotony_min_span {
            out.push(
                NarrativePathology::new(
                    PathologyKind::NarrationModeMonotony,
                    PathologySeverity::Info,
                    format!(
                        "Narration mode {} dominates {:.0}% of {} situations",
                        mode,
                        ratio * 100.0,
                        total
                    ),
                )
                .with_suggestion("Vary narration modes for texture."),
            );
        }
    }
    Ok(())
}

fn check_subplots(
    hg: &Hypergraph,
    nid: &str,
    cfg: &DiagnosticConfig,
    out: &mut Vec<NarrativePathology>,
) -> Result<()> {
    let subplots = match crate::narrative::subplots::load_subplot_analysis(hg, nid)? {
        Some(s) => s,
        None => return Ok(()),
    };
    for sp in &subplots.subplots {
        // SubplotStarvation: density < 1 situation per gap-sized window
        let span = sp.end_chapter.saturating_sub(sp.start_chapter);
        let starved = span >= cfg.subplot_starvation_gap
            && sp.situations.len() * cfg.subplot_starvation_gap < span;
        if starved {
            out.push(
                NarrativePathology::new(
                    PathologyKind::SubplotStarvation,
                    PathologySeverity::Warning,
                    format!(
                        "Subplot '{}' active chapters {}–{} has only {} situations",
                        sp.label,
                        sp.start_chapter,
                        sp.end_chapter,
                        sp.situations.len()
                    ),
                )
                .at_chapter(sp.start_chapter)
                .with_suggestion("Add at least a progress scene so the thread isn't forgotten."),
            );
        }
        // SubplotOrphan: has relation that should converge but no convergence point
        use crate::narrative::subplots::SubplotRelation::*;
        let expects_convergence = matches!(
            sp.relation_to_main,
            Complication | Convergence | Mirror | Setup
        );
        if expects_convergence && sp.convergence_point.is_none() {
            out.push(
                NarrativePathology::new(
                    PathologyKind::SubplotOrphan,
                    PathologySeverity::Warning,
                    format!(
                        "Subplot '{}' ({:?}) never converges with the main plot",
                        sp.label, sp.relation_to_main
                    ),
                )
                .at_chapter(sp.start_chapter)
                .with_suggestion("Add a convergence point or reclassify as Independent."),
            );
        }
    }
    Ok(())
}

// ─── KV ─────────────────────────────────────────────────────

fn diagnosis_key(narrative_id: &str) -> Vec<u8> {
    format!("nd/{}", narrative_id).into_bytes()
}

pub fn store_diagnosis(hg: &Hypergraph, result: &DiagnosisResult) -> Result<()> {
    let key = diagnosis_key(&result.narrative_id);
    let val = serde_json::to_vec(result)?;
    hg.store().put(&key, &val)
}

pub fn load_diagnosis(hg: &Hypergraph, narrative_id: &str) -> Result<Option<DiagnosisResult>> {
    let key = diagnosis_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
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
    fn test_diagnose_empty_narrative() {
        let hg = test_hg();
        let result = diagnose_narrative(&hg, "nonexistent").unwrap();
        assert!(result.pathologies.is_empty());
        assert_eq!(result.health_score, 1.0);
    }

    #[test]
    fn test_diagnosis_kv_persistence() {
        let hg = test_hg();
        let result = DiagnosisResult {
            narrative_id: "test".into(),
            pathologies: vec![NarrativePathology::new(
                PathologyKind::OrphanedSetup,
                PathologySeverity::Warning,
                "Test pathology",
            )
            .at_chapter(3)],
            error_count: 0,
            warning_count: 1,
            info_count: 0,
            note_count: 0,
            health_score: 0.95,
            summary: DiagnosticSummary {
                warnings: 1,
                health_score: 0.95,
                ..Default::default()
            },
        };
        store_diagnosis(&hg, &result).unwrap();
        let loaded = load_diagnosis(&hg, "test").unwrap().unwrap();
        assert_eq!(loaded.pathologies.len(), 1);
        assert_eq!(loaded.health_score, 0.95);
    }

    #[test]
    fn test_health_score_calculation() {
        let hg = test_hg();
        let commitment = crate::narrative::commitments::NarrativeCommitment {
            id: uuid::Uuid::now_v7(),
            narrative_id: "health-test".into(),
            commitment_type: crate::narrative::commitments::CommitmentType::ChekhovsGun,
            setup_event: uuid::Uuid::nil(),
            setup_chapter: 1,
            setup_salience: 2.0,
            status: crate::narrative::commitments::CommitmentStatus::Abandoned,
            payoff_event: None,
            payoff_chapter: None,
            payoff_distance: None,
            causal_chain: vec![],
            description: "The gun on the mantel".into(),
            tracked_element: "gun".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        crate::narrative::commitments::store_commitment(&hg, &commitment).unwrap();

        let result = diagnose_narrative(&hg, "health-test").unwrap();
        assert!(result.error_count > 0);
        assert!(result.health_score < 1.0);
        assert_eq!(result.summary.errors, result.error_count);
    }

    #[test]
    fn test_genre_preset_thriller_strict() {
        let cfg = DiagnosticConfig::for_genre(GenrePreset::Thriller);
        assert_eq!(cfg.max_simultaneous_commitments, 8);
        assert_eq!(cfg.max_action_streak, 3);
        assert!(cfg.suppress.contains(&PathologyKind::FlatProtagonist));
    }

    #[test]
    fn test_genre_preset_literary_lenient() {
        let cfg = DiagnosticConfig::for_genre(GenrePreset::LiteraryFiction);
        assert_eq!(cfg.max_action_streak, 6);
        assert!(cfg.suppress.contains(&PathologyKind::SubplotOrphan));
    }

    #[test]
    fn test_suppression_filters_pathologies() {
        let hg = test_hg();
        let commitment = crate::narrative::commitments::NarrativeCommitment {
            id: uuid::Uuid::now_v7(),
            narrative_id: "suppress-test".into(),
            commitment_type: crate::narrative::commitments::CommitmentType::ChekhovsGun,
            setup_event: uuid::Uuid::nil(),
            setup_chapter: 1,
            setup_salience: 2.0,
            status: crate::narrative::commitments::CommitmentStatus::Abandoned,
            payoff_event: None,
            payoff_chapter: None,
            payoff_distance: None,
            causal_chain: vec![],
            description: "Thing".into(),
            tracked_element: "thing".into(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        crate::narrative::commitments::store_commitment(&hg, &commitment).unwrap();

        let cfg = DiagnosticConfig {
            suppress: vec![PathologyKind::OrphanedSetup],
            ..Default::default()
        };
        let result = diagnose_narrative_with(&hg, "suppress-test", &cfg).unwrap();
        assert_eq!(
            result
                .pathologies
                .iter()
                .filter(|p| p.kind == PathologyKind::OrphanedSetup)
                .count(),
            0
        );
    }

    #[test]
    fn test_diagnose_chapter_filters_by_chapter() {
        let hg = test_hg();
        // Plant a commitment in chapter 5 with orphan status
        let commitment = crate::narrative::commitments::NarrativeCommitment {
            id: uuid::Uuid::now_v7(),
            narrative_id: "chapter-test".into(),
            commitment_type: crate::narrative::commitments::CommitmentType::ChekhovsGun,
            setup_event: uuid::Uuid::nil(),
            setup_chapter: 5,
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
        let ch5 = diagnose_chapter(&hg, "chapter-test", 5).unwrap();
        // Chapter 5 should contain the orphaned setup
        assert!(ch5.iter().any(|p| p.kind == PathologyKind::OrphanedSetup));
        // Chapter 99 should contain nothing
        let ch99 = diagnose_chapter(&hg, "chapter-test", 99).unwrap();
        assert!(ch99.is_empty());
    }
}
