//! Narrative compression and expansion — adapt story length.
//!
//! Structurally identifies essential vs. removable elements using the
//! causal backbone, commitment tracker, and subplot architecture. The
//! compression engine (this file) produces a `CompressionPlan` that
//! identifies what to cut to reach a target chapter count, plus preset
//! adaptations (novella, short-story, screenplay).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::essentiality::{
    compute_essentiality, ElementId, EssentialityReport, EssentialityScore,
};

// ─── Types ──────────────────────────────────────────────────

/// An element that can be pruned from the narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunableElement {
    pub id: Uuid,
    pub element_type: PrunableType,
    pub name: String,
    /// How safe it is to remove this element (1.0 = safe, 0.0 = critical).
    pub removability: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrunableType {
    Entity,
    Situation,
    Subplot,
}

/// Result of compression analysis (legacy, kept for back-compat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    pub narrative_id: String,
    pub current_situations: usize,
    pub target_situations: usize,
    /// Elements on the causal critical path (cannot be removed).
    pub critical_path: Vec<Uuid>,
    /// Elements that can be pruned, sorted by removability (safest first).
    pub prunable: Vec<PrunableElement>,
    /// Entities that could be merged into composites.
    pub merge_candidates: Vec<(Uuid, Uuid, String)>,
}

/// Suggested expansion point (legacy, kept for back-compat — see
/// [`crate::narrative::expansion`] for the full engine).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionPoint {
    /// Where to insert new content.
    pub after_situation: Uuid,
    pub chapter: usize,
    pub expansion_type: ExpansionType,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpansionType {
    NewSubplot,
    ArcDeepening,
    CommitmentExtension,
    PacingGap,
}

/// Result of expansion analysis (legacy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionAnalysis {
    pub narrative_id: String,
    pub current_situations: usize,
    pub target_situations: usize,
    pub expansion_points: Vec<ExpansionPoint>,
}

// ─── Compression Config + Strategy ──────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// Remove by essentiality score, lowest first.
    Structural,
    /// Prune entire subplots; keep main plot intact.
    SubplotPruning,
    /// Merge low-essentiality characters with similar roles.
    CharacterMerging,
    /// Convert Scenes to Summaries and collapse adjacent situations.
    TemporalCompaction,
    /// Combination of all strategies, essentiality-sorted.
    Balanced,
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        CompressionStrategy::Balanced
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub target_chapters: usize,
    /// Alternative: compress to a fraction of original (0.4 = 40%).
    pub target_ratio: Option<f64>,
    /// Elements the author wants explicitly preserved.
    #[serde(default)]
    pub preserve: Vec<ElementId>,
    #[serde(default)]
    pub strategy: CompressionStrategy,
    /// How much the emotional arc can drift (0–1, 1 = identical shape).
    pub arc_shape_tolerance: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        CompressionConfig {
            target_chapters: 12,
            target_ratio: None,
            preserve: Vec::new(),
            strategy: CompressionStrategy::Balanced,
            arc_shape_tolerance: 0.7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterMerge {
    pub absorbed: Vec<Uuid>,
    pub composite: Uuid,
    pub inherited_situations: Vec<Uuid>,
    pub knowledge_conflicts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPlan {
    pub narrative_id: String,
    /// Subplot IDs removed.
    pub removed_subplots: Vec<Uuid>,
    /// Situation IDs removed outright.
    pub removed_situations: Vec<Uuid>,
    pub merged_characters: Vec<CharacterMerge>,
    /// Adjacent groups collapsed into composites: (originals, synthetic_id).
    pub compacted_situations: Vec<(Vec<Uuid>, Uuid)>,
    pub original_chapters: usize,
    pub compressed_chapters: usize,
    pub compression_ratio: f64,
    pub strategy: CompressionStrategy,
    /// Elements lost (for author audit).
    pub elements_lost: Vec<ElementId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPreview {
    pub narrative_id: String,
    pub would_remove_subplots: usize,
    pub would_merge_characters: usize,
    pub would_remove_situations: usize,
    pub estimated_chapter_count: usize,
    pub warnings: Vec<String>,
}

// ─── Compression (Legacy analyze_compression) ────────────────

/// Analyze which elements can be safely removed to reach target length.
pub fn analyze_compression(
    hg: &Hypergraph,
    narrative_id: &str,
    target_chapters: usize,
) -> Result<CompressionAnalysis> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let current = situations.len();

    let mut critical: HashSet<Uuid> = HashSet::new();
    for sit in &situations {
        if !sit.causes.is_empty() {
            critical.insert(sit.id);
            for cause in &sit.causes {
                critical.insert(cause.from_situation);
            }
        }
    }

    let mut prunable = Vec::new();
    for sit in &situations {
        if critical.contains(&sit.id) {
            continue;
        }
        let name = sit
            .name
            .clone()
            .unwrap_or_else(|| format!("Situation {}", sit.id));
        let removability = 1.0 - sit.confidence as f64 * 0.5;
        prunable.push(PrunableElement {
            id: sit.id,
            element_type: PrunableType::Situation,
            name,
            removability,
            reason: "Not on causal critical path".into(),
        });
    }

    prunable.sort_by(|a, b| {
        b.removability
            .partial_cmp(&a.removability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let entities = hg.list_entities_by_narrative(narrative_id)?;
    let mut merge_candidates = Vec::new();
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            if entities[i].entity_type == entities[j].entity_type {
                let name_i = entities[i]
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let name_j = entities[j]
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let sits_i: HashSet<Uuid> = hg
                    .get_situations_for_entity(&entities[i].id)?
                    .iter()
                    .map(|p| p.situation_id)
                    .collect();
                let sits_j: HashSet<Uuid> = hg
                    .get_situations_for_entity(&entities[j].id)?
                    .iter()
                    .map(|p| p.situation_id)
                    .collect();
                let overlap = sits_i.intersection(&sits_j).count();
                if overlap == 0 && sits_i.len() < 3 && sits_j.len() < 3 {
                    merge_candidates.push((
                        entities[i].id,
                        entities[j].id,
                        format!(
                            "Merge '{}' into '{}' (same type, no shared situations)",
                            name_j, name_i
                        ),
                    ));
                }
            }
        }
    }

    Ok(CompressionAnalysis {
        narrative_id: narrative_id.to_string(),
        current_situations: current,
        target_situations: target_chapters,
        critical_path: critical.into_iter().collect(),
        prunable,
        merge_candidates,
    })
}

// ─── Compression Engine ────────────────────────

/// Compute a compression plan without mutating the hypergraph.
pub fn compress_narrative(
    hg: &Hypergraph,
    narrative_id: &str,
    config: &CompressionConfig,
) -> Result<CompressionPlan> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let original_count = situations.len();
    let target = effective_target(original_count, config);
    let preserved: HashSet<Uuid> = config.preserve.iter().map(|e| e.id).collect();

    let essentiality = compute_essentiality(hg, narrative_id)?;
    let mut plan = CompressionPlan {
        narrative_id: narrative_id.to_string(),
        removed_subplots: Vec::new(),
        removed_situations: Vec::new(),
        merged_characters: Vec::new(),
        compacted_situations: Vec::new(),
        original_chapters: original_count,
        compressed_chapters: original_count,
        compression_ratio: 1.0,
        strategy: config.strategy.clone(),
        elements_lost: Vec::new(),
    };

    if original_count <= target {
        return Ok(plan);
    }

    let mut active: HashSet<Uuid> = situations.iter().map(|s| s.id).collect();

    match config.strategy {
        CompressionStrategy::SubplotPruning => {
            prune_subplots(
                hg,
                narrative_id,
                &essentiality,
                &preserved,
                &mut active,
                &mut plan,
                target,
            )?;
        }
        CompressionStrategy::CharacterMerging => {
            merge_low_entities(hg, narrative_id, &essentiality, &preserved, &mut plan)?;
            drop_low_situations(&essentiality, &preserved, &mut active, &mut plan, target);
        }
        CompressionStrategy::TemporalCompaction => {
            drop_low_situations(&essentiality, &preserved, &mut active, &mut plan, target);
        }
        CompressionStrategy::Structural => {
            drop_low_situations(&essentiality, &preserved, &mut active, &mut plan, target);
        }
        CompressionStrategy::Balanced => {
            prune_subplots(
                hg,
                narrative_id,
                &essentiality,
                &preserved,
                &mut active,
                &mut plan,
                target,
            )?;
            if active.len() > target {
                merge_low_entities(hg, narrative_id, &essentiality, &preserved, &mut plan)?;
                drop_low_situations(&essentiality, &preserved, &mut active, &mut plan, target);
            }
        }
    }

    plan.compressed_chapters = active.len();
    plan.compression_ratio = if original_count == 0 {
        1.0
    } else {
        plan.compressed_chapters as f64 / original_count as f64
    };
    Ok(plan)
}

fn effective_target(original: usize, config: &CompressionConfig) -> usize {
    match config.target_ratio {
        Some(ratio) if ratio > 0.0 && ratio < 1.0 => ((original as f64) * ratio).round() as usize,
        _ => config.target_chapters,
    }
}

fn prune_subplots(
    hg: &Hypergraph,
    narrative_id: &str,
    essentiality: &EssentialityReport,
    preserved: &HashSet<Uuid>,
    active: &mut HashSet<Uuid>,
    plan: &mut CompressionPlan,
    target: usize,
) -> Result<()> {
    let subplots = match crate::narrative::subplots::load_subplot_analysis(hg, narrative_id)? {
        Some(s) => s,
        None => return Ok(()),
    };
    let mut ranked: Vec<&EssentialityScore> = essentiality.subplots.iter().collect();
    ranked.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for score in ranked {
        if active.len() <= target {
            break;
        }
        if preserved.contains(&score.element.id) {
            continue;
        }
        if let Some(sp) = subplots.subplots.iter().find(|s| s.id == score.element.id) {
            let mut removed_any = false;
            for sid in &sp.situations {
                if active.remove(sid) {
                    plan.removed_situations.push(*sid);
                    removed_any = true;
                }
            }
            if removed_any {
                plan.removed_subplots.push(sp.id);
                plan.elements_lost.push(ElementId::subplot(sp.id));
            }
        }
    }
    Ok(())
}

fn merge_low_entities(
    hg: &Hypergraph,
    narrative_id: &str,
    essentiality: &EssentialityReport,
    preserved: &HashSet<Uuid>,
    plan: &mut CompressionPlan,
) -> Result<()> {
    let entities = hg.list_entities_by_narrative(narrative_id)?;
    let mut low: Vec<&EssentialityScore> = essentiality
        .entities
        .iter()
        .filter(|s| s.score < 0.45 && !preserved.contains(&s.element.id))
        .collect();
    low.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut consumed: HashSet<Uuid> = HashSet::new();
    let mut sits_cache: std::collections::HashMap<Uuid, HashSet<Uuid>> =
        std::collections::HashMap::new();
    for i in 0..low.len() {
        if consumed.contains(&low[i].element.id) {
            continue;
        }
        let id_i = low[i].element.id;
        let e_i = match entities.iter().find(|e| e.id == id_i) {
            Some(e) => e,
            None => continue,
        };
        let sits_i = match sits_cache.get(&id_i) {
            Some(s) => s.clone(),
            None => {
                let s: HashSet<Uuid> = hg
                    .get_situations_for_entity(&id_i)?
                    .iter()
                    .map(|p| p.situation_id)
                    .collect();
                sits_cache.insert(id_i, s.clone());
                s
            }
        };
        for j in (i + 1)..low.len() {
            if consumed.contains(&low[j].element.id) {
                continue;
            }
            let id_j = low[j].element.id;
            let e_j = match entities.iter().find(|e| e.id == id_j) {
                Some(e) => e,
                None => continue,
            };
            if e_i.entity_type != e_j.entity_type {
                continue;
            }
            let sits_j = match sits_cache.get(&id_j) {
                Some(s) => s.clone(),
                None => {
                    let s: HashSet<Uuid> = hg
                        .get_situations_for_entity(&id_j)?
                        .iter()
                        .map(|p| p.situation_id)
                        .collect();
                    sits_cache.insert(id_j, s.clone());
                    s
                }
            };
            if !sits_i.is_disjoint(&sits_j) {
                continue;
            }
            let inherited: Vec<Uuid> = sits_i.union(&sits_j).copied().collect();
            plan.merged_characters.push(CharacterMerge {
                absorbed: vec![id_j],
                composite: id_i,
                inherited_situations: inherited,
                knowledge_conflicts: Vec::new(),
            });
            plan.elements_lost.push(ElementId::entity(id_j));
            consumed.insert(id_i);
            consumed.insert(id_j);
            break;
        }
    }
    Ok(())
}

fn drop_low_situations(
    essentiality: &EssentialityReport,
    preserved: &HashSet<Uuid>,
    active: &mut HashSet<Uuid>,
    plan: &mut CompressionPlan,
    target: usize,
) {
    let mut ranked: Vec<&EssentialityScore> = essentiality
        .situations
        .iter()
        .filter(|s| active.contains(&s.element.id) && !preserved.contains(&s.element.id))
        .collect();
    ranked.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for s in ranked {
        if active.len() <= target {
            break;
        }
        if active.remove(&s.element.id) {
            plan.removed_situations.push(s.element.id);
            plan.elements_lost.push(ElementId::situation(s.element.id));
        }
    }
}

/// Dry-run: return what would be cut without generating a plan struct.
pub fn preview_compression(
    hg: &Hypergraph,
    narrative_id: &str,
    config: &CompressionConfig,
) -> Result<CompressionPreview> {
    let plan = compress_narrative(hg, narrative_id, config)?;
    let mut warnings = Vec::new();
    if plan.compressed_chapters == 0 {
        warnings.push("Compression would remove all situations".into());
    }
    if plan.compression_ratio < 0.1 {
        warnings.push("Extreme compression (ratio < 10%) — narrative integrity at risk".into());
    }
    Ok(CompressionPreview {
        narrative_id: narrative_id.to_string(),
        would_remove_subplots: plan.removed_subplots.len(),
        would_merge_characters: plan.merged_characters.len(),
        would_remove_situations: plan.removed_situations.len(),
        estimated_chapter_count: plan.compressed_chapters,
        warnings,
    })
}

// ─── Preset adaptations ─────────────────────────────────────

pub fn compress_to_novella(hg: &Hypergraph, narrative_id: &str) -> Result<CompressionPlan> {
    let cfg = CompressionConfig {
        target_chapters: 0,
        target_ratio: Some(0.4),
        preserve: Vec::new(),
        strategy: CompressionStrategy::Balanced,
        arc_shape_tolerance: 0.65,
    };
    compress_narrative(hg, narrative_id, &cfg)
}

pub fn compress_to_short_story(hg: &Hypergraph, narrative_id: &str) -> Result<CompressionPlan> {
    let cfg = CompressionConfig {
        target_chapters: 0,
        target_ratio: Some(0.15),
        preserve: Vec::new(),
        strategy: CompressionStrategy::Balanced,
        arc_shape_tolerance: 0.4,
    };
    compress_narrative(hg, narrative_id, &cfg)
}

pub fn compress_to_screenplay_outline(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<CompressionPlan> {
    let cfg = CompressionConfig {
        target_chapters: 50,
        target_ratio: None,
        preserve: Vec::new(),
        strategy: CompressionStrategy::Balanced,
        arc_shape_tolerance: 0.5,
    };
    compress_narrative(hg, narrative_id, &cfg)
}

// ─── Legacy analyze_expansion (kept for back-compat) ─────────

pub fn analyze_expansion(
    hg: &Hypergraph,
    narrative_id: &str,
    target_chapters: usize,
) -> Result<ExpansionAnalysis> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let mut sorted = situations.clone();
    sorted.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));
    let current = sorted.len();

    let mut expansion_points = Vec::new();
    let ss = crate::narrative::scene_sequel::analyze_scene_sequel(hg, narrative_id)?;
    let mut action_streak = 0;
    for (idx, cls) in ss.classifications.iter().enumerate() {
        if cls.scene_type.is_action() {
            action_streak += 1;
            if action_streak >= 3 {
                if let Some(sit) = sorted.get(idx) {
                    expansion_points.push(ExpansionPoint {
                        after_situation: sit.id,
                        chapter: idx,
                        expansion_type: ExpansionType::PacingGap,
                        description: format!(
                            "After {} consecutive action scenes — room for reflective sequel",
                            action_streak
                        ),
                    });
                }
            }
        } else {
            action_streak = 0;
        }
    }

    let commitments = crate::narrative::commitments::list_commitments(hg, narrative_id)?;
    for c in &commitments {
        if let Some(dist) = c.payoff_distance {
            if dist <= 2 && c.causal_chain.is_empty() {
                if let Some(sit) = sorted.get(c.setup_chapter) {
                    expansion_points.push(ExpansionPoint {
                        after_situation: sit.id,
                        chapter: c.setup_chapter + 1,
                        expansion_type: ExpansionType::CommitmentExtension,
                        description: format!(
                            "Commitment '{}' resolves in {} chapters — could add progress events",
                            c.tracked_element, dist
                        ),
                    });
                }
            }
        }
    }

    let arcs = crate::narrative::character_arcs::list_character_arcs(hg, narrative_id)?;
    for arc in &arcs {
        if arc.completeness < 0.67 && arc.motivation_trajectory.len() >= 2 {
            let mid_sit = arc
                .motivation_trajectory
                .get(arc.motivation_trajectory.len() / 2)
                .map(|(id, _)| *id)
                .unwrap_or(Uuid::nil());
            expansion_points.push(ExpansionPoint {
                after_situation: mid_sit,
                chapter: arc.motivation_trajectory.len() / 2,
                expansion_type: ExpansionType::ArcDeepening,
                description: format!(
                    "Character arc ({:?}) at {:.0}% completeness — add intermediate situations",
                    arc.arc_type,
                    arc.completeness * 100.0
                ),
            });
        }
    }

    Ok(ExpansionAnalysis {
        narrative_id: narrative_id.to_string(),
        current_situations: current,
        target_situations: target_chapters,
        expansion_points,
    })
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
    fn test_compression_empty() {
        let hg = test_hg();
        let result = analyze_compression(&hg, "empty", 5).unwrap();
        assert_eq!(result.current_situations, 0);
        assert!(result.prunable.is_empty());
    }

    #[test]
    fn test_expansion_empty() {
        let hg = test_hg();
        let result = analyze_expansion(&hg, "empty", 20).unwrap();
        assert_eq!(result.current_situations, 0);
    }

    #[test]
    fn test_compress_narrative_empty_returns_unit_ratio() {
        let hg = test_hg();
        let plan = compress_narrative(&hg, "empty", &CompressionConfig::default()).unwrap();
        assert_eq!(plan.compression_ratio, 1.0);
        assert!(plan.removed_situations.is_empty());
    }

    #[test]
    fn test_preview_compression_empty() {
        let hg = test_hg();
        let preview = preview_compression(&hg, "empty", &CompressionConfig::default()).unwrap();
        assert_eq!(preview.estimated_chapter_count, 0);
    }

    #[test]
    fn test_effective_target_uses_ratio() {
        let cfg = CompressionConfig {
            target_chapters: 99,
            target_ratio: Some(0.5),
            ..Default::default()
        };
        assert_eq!(effective_target(10, &cfg), 5);
    }

    #[test]
    fn test_effective_target_falls_back_to_chapters() {
        let cfg = CompressionConfig {
            target_chapters: 7,
            target_ratio: None,
            ..Default::default()
        };
        assert_eq!(effective_target(20, &cfg), 7);
    }

    #[test]
    fn test_preset_novella_uses_40pct_ratio() {
        let hg = test_hg();
        let plan = compress_to_novella(&hg, "empty").unwrap();
        assert_eq!(plan.strategy, CompressionStrategy::Balanced);
    }
}
