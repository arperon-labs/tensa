//! Structural essentiality analysis — score elements 0 (removable) to 1 (essential).
//!
//! Prerequisite for compression and expansion. Each narrative element (situation,
//! entity, or subplot) gets a weighted score that combines causal criticality,
//! commitment load, knowledge-gate role, arc anchoring, and dramatic irony role.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    Entity,
    Situation,
    Subplot,
    Commitment,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ElementId {
    pub kind: ElementType,
    pub id: Uuid,
}

impl ElementId {
    pub fn situation(id: Uuid) -> Self {
        Self {
            kind: ElementType::Situation,
            id,
        }
    }
    pub fn entity(id: Uuid) -> Self {
        Self {
            kind: ElementType::Entity,
            id,
        }
    }
    pub fn subplot(id: Uuid) -> Self {
        Self {
            kind: ElementType::Subplot,
            id,
        }
    }
    pub fn commitment(id: Uuid) -> Self {
        Self {
            kind: ElementType::Commitment,
            id,
        }
    }
}

/// Why a particular score was assigned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssentialityReason {
    pub label: String,
    pub weight: f64,
    pub contribution: f64,
}

/// What breaks if this element is removed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RemovalConsequence {
    pub broken_commitments: Vec<Uuid>,
    pub knowledge_violations: Vec<(Uuid, String)>,
    pub broken_arcs: Vec<Uuid>,
    pub causal_gaps: Vec<(Uuid, Uuid)>,
    pub orphaned_subplots: Vec<Uuid>,
}

/// An essentiality score for a single element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssentialityScore {
    pub element: ElementId,
    /// 0.0 = completely removable, 1.0 = absolutely essential.
    pub score: f64,
    pub reasons: Vec<EssentialityReason>,
    pub removal_consequences: RemovalConsequence,
}

impl EssentialityScore {
    fn new(element: ElementId) -> Self {
        Self {
            element,
            score: 0.0,
            reasons: Vec::new(),
            removal_consequences: RemovalConsequence::default(),
        }
    }
    fn add(&mut self, label: &str, weight: f64, raw: f64) {
        let contribution = (weight * raw).clamp(0.0, weight);
        self.score = (self.score + contribution).min(1.0);
        self.reasons.push(EssentialityReason {
            label: label.to_string(),
            weight,
            contribution,
        });
    }
}

// ─── Situation essentiality ──────────────────────────────────

const CAUSAL_WEIGHT: f64 = 0.3;
const COMMITMENT_WEIGHT: f64 = 0.25;
const KNOWLEDGE_WEIGHT: f64 = 0.2;
const ARC_WEIGHT: f64 = 0.15;
const IRONY_WEIGHT: f64 = 0.1;

pub fn compute_situation_essentiality(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<EssentialityScore>> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(Vec::new());
    }

    // Build in-degree / out-degree on causal graph for betweenness proxy
    let mut in_deg: HashMap<Uuid, usize> = HashMap::new();
    let mut out_deg: HashMap<Uuid, usize> = HashMap::new();
    for s in &situations {
        in_deg.entry(s.id).or_insert(0);
        out_deg.entry(s.id).or_insert(0);
        for c in &s.causes {
            *in_deg.entry(s.id).or_insert(0) += 1;
            *out_deg.entry(c.from_situation).or_insert(0) += 1;
        }
    }
    let max_deg = in_deg
        .values()
        .chain(out_deg.values())
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as f64;

    // Commitment load: which situations are setup/payoff events
    let commitments =
        crate::narrative::commitments::list_commitments(hg, narrative_id).unwrap_or_default();
    let mut commitment_load: HashMap<Uuid, f64> = HashMap::new();
    let mut sole_payoff: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for c in &commitments {
        *commitment_load.entry(c.setup_event).or_insert(0.0) += c.setup_salience.min(3.0) / 3.0;
        if let Some(pid) = c.payoff_event {
            *commitment_load.entry(pid).or_insert(0.0) += 1.0;
            sole_payoff.entry(pid).or_default().push(c.id);
        }
    }

    // Arc anchors: situations at the start, midpoint, end of motivation_trajectory
    let arcs =
        crate::narrative::character_arcs::list_character_arcs(hg, narrative_id).unwrap_or_default();
    let mut arc_anchors: HashMap<Uuid, f64> = HashMap::new();
    for arc in &arcs {
        let traj = &arc.motivation_trajectory;
        if traj.is_empty() {
            continue;
        }
        let first = traj.first().map(|(id, _)| *id);
        let mid = traj.get(traj.len() / 2).map(|(id, _)| *id);
        let last = traj.last().map(|(id, _)| *id);
        for id in [first, mid, last].iter().flatten() {
            *arc_anchors.entry(*id).or_insert(0.0) += 1.0;
        }
    }

    // Dramatic irony: situations that generate irony events
    let irony_map =
        crate::narrative::dramatic_irony::compute_dramatic_irony_map(hg, narrative_id).ok();
    let mut irony_load: HashMap<Uuid, f64> = HashMap::new();
    if let Some(map) = &irony_map {
        for ev in &map.events {
            *irony_load.entry(ev.situation_id).or_insert(0.0) += ev.irony_intensity / 10.0;
        }
    }

    let mut results = Vec::with_capacity(situations.len());
    for s in &situations {
        let mut score = EssentialityScore::new(ElementId::situation(s.id));
        let deg_score = ((in_deg.get(&s.id).copied().unwrap_or(0)
            + out_deg.get(&s.id).copied().unwrap_or(0)) as f64)
            / (2.0 * max_deg);
        score.add("causal_criticality", CAUSAL_WEIGHT, deg_score);
        let cl = commitment_load.get(&s.id).copied().unwrap_or(0.0).min(1.0);
        score.add("commitment_load", COMMITMENT_WEIGHT, cl);
        let parts = hg.get_participants_for_situation(&s.id)?;
        let knowledge = if parts.is_empty() {
            0.0
        } else {
            (parts.len() as f64 / 5.0).min(1.0)
        };
        score.add("knowledge_gate", KNOWLEDGE_WEIGHT, knowledge);
        let anchor = arc_anchors.get(&s.id).copied().unwrap_or(0.0).min(1.0);
        score.add("arc_anchor", ARC_WEIGHT, anchor);
        let ir = irony_load.get(&s.id).copied().unwrap_or(0.0).min(1.0);
        score.add("dramatic_irony", IRONY_WEIGHT, ir);

        // Record removal consequences
        if let Some(broken) = sole_payoff.get(&s.id) {
            score.removal_consequences.broken_commitments = broken.clone();
        }
        results.push(score);
    }

    // Sort by score descending
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

// ─── Entity essentiality ────────────────────────────────────

pub fn compute_entity_essentiality(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<EssentialityScore>> {
    let entities = hg.list_entities_by_narrative(narrative_id)?;
    if entities.is_empty() {
        return Ok(Vec::new());
    }
    let mut part_counts: HashMap<Uuid, usize> = HashMap::new();
    for e in &entities {
        let sits = hg.get_situations_for_entity(&e.id)?;
        part_counts.insert(e.id, sits.len());
    }
    let max_parts = part_counts.values().copied().max().unwrap_or(1).max(1) as f64;

    // Arc importance
    let arcs =
        crate::narrative::character_arcs::list_character_arcs(hg, narrative_id).unwrap_or_default();
    let mut arc_importance: HashMap<Uuid, f64> = HashMap::new();
    for a in &arcs {
        arc_importance.insert(a.character_id, a.completeness.max(0.5));
    }

    let mut results = Vec::with_capacity(entities.len());
    for e in &entities {
        let mut score = EssentialityScore::new(ElementId::entity(e.id));
        let centrality = part_counts.get(&e.id).copied().unwrap_or(0) as f64 / max_parts;
        score.add("participation_centrality", 0.6, centrality);
        let arc = arc_importance.get(&e.id).copied().unwrap_or(0.0);
        score.add("arc_importance", 0.4, arc);
        results.push(score);
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

// ─── Subplot essentiality ───────────────────────────────────

pub fn compute_subplot_essentiality(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<EssentialityScore>> {
    let analysis = match crate::narrative::subplots::load_subplot_analysis(hg, narrative_id)? {
        Some(a) => a,
        None => return Ok(Vec::new()),
    };
    let mut results = Vec::with_capacity(analysis.subplots.len());
    for sp in &analysis.subplots {
        let mut score = EssentialityScore::new(ElementId::subplot(sp.id));
        use crate::narrative::subplots::SubplotRelation::*;
        let relation_weight = match sp.relation_to_main {
            Complication => 1.0,
            Convergence => 0.9,
            Mirror => 0.6,
            Contrast => 0.5,
            Setup => 0.75,
            Independent => 0.1,
        };
        score.add("subplot_relation", 0.5, relation_weight);
        let has_convergence = if sp.convergence_point.is_some() {
            1.0
        } else {
            0.0
        };
        score.add("convergence_present", 0.3, has_convergence);
        let size = (sp.situations.len() as f64 / 10.0).min(1.0);
        score.add("subplot_size", 0.2, size);
        results.push(score);
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results)
}

// ─── Aggregate ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssentialityReport {
    pub narrative_id: String,
    pub situations: Vec<EssentialityScore>,
    pub entities: Vec<EssentialityScore>,
    pub subplots: Vec<EssentialityScore>,
}

pub fn compute_essentiality(hg: &Hypergraph, narrative_id: &str) -> Result<EssentialityReport> {
    Ok(EssentialityReport {
        narrative_id: narrative_id.to_string(),
        situations: compute_situation_essentiality(hg, narrative_id)?,
        entities: compute_entity_essentiality(hg, narrative_id)?,
        subplots: compute_subplot_essentiality(hg, narrative_id)?,
    })
}

fn essentiality_key(narrative_id: &str) -> Vec<u8> {
    format!("ess/{}", narrative_id).into_bytes()
}

pub fn store_essentiality(hg: &Hypergraph, report: &EssentialityReport) -> Result<()> {
    let key = essentiality_key(&report.narrative_id);
    let val = serde_json::to_vec(report)?;
    hg.store().put(&key, &val)
}

pub fn load_essentiality(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<EssentialityReport>> {
    let key = essentiality_key(narrative_id);
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
    fn test_empty_narrative_yields_empty_scores() {
        let hg = test_hg();
        let report = compute_essentiality(&hg, "empty").unwrap();
        assert!(report.situations.is_empty());
        assert!(report.entities.is_empty());
        assert!(report.subplots.is_empty());
    }

    #[test]
    fn test_element_id_constructors() {
        let u = Uuid::now_v7();
        assert_eq!(ElementId::situation(u).kind, ElementType::Situation);
        assert_eq!(ElementId::entity(u).kind, ElementType::Entity);
        assert_eq!(ElementId::subplot(u).kind, ElementType::Subplot);
    }

    #[test]
    fn test_essentiality_kv_persistence() {
        let hg = test_hg();
        let report = EssentialityReport {
            narrative_id: "kv-test".into(),
            situations: vec![],
            entities: vec![],
            subplots: vec![],
        };
        store_essentiality(&hg, &report).unwrap();
        let loaded = load_essentiality(&hg, "kv-test").unwrap().unwrap();
        assert_eq!(loaded.narrative_id, "kv-test");
    }

    #[test]
    fn test_score_bounded_in_unit_interval() {
        let mut s = EssentialityScore::new(ElementId::situation(Uuid::now_v7()));
        s.add("a", 0.5, 2.0); // raw > 1 should be clamped
        s.add("b", 0.5, 1.5);
        assert!(s.score <= 1.0);
        assert!(s.score >= 0.0);
    }
}
