//! Temporal Inductive Logic Programming (ILP) engine.
//!
//! Learns interpretable temporal Horn clauses from entity-situation
//! participation patterns in the hypergraph. Rules combine entity types,
//! roles, narrative levels, and Allen temporal relations.
//!
//! Example learned rule:
//!   "If Actor/Protagonist in Scene BEFORE Scene, then Actor/Protagonist
//!    in Scene follows with confidence 0.85 (support=12, lift=2.3)"
//!
//! Algorithm (bottom-up mining):
//! 1. Extract per-entity chronological situation sequences
//! 2. Mine bigram/trigram candidate rules from sliding windows
//! 3. Score by confidence and lift, filter by thresholds
//! 4. Prune subsumed (more-general) rules
//!
//! Reference: "Temporal Inductive Logic Reasoning over Hypergraphs" (IJCAI 2024)

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id, load_sorted_situations};
use crate::error::Result;
use crate::hypergraph::keys::ANALYSIS_TEMPORAL_ILP;
use crate::hypergraph::Hypergraph;
use crate::temporal::interval::relation_between;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

// ─── Configuration ─────────────────────────────────────────

/// Configuration for the Temporal ILP rule mining algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalILPConfig {
    /// Minimum number of pattern occurrences to form a rule.
    pub min_support: usize,
    /// Minimum rule confidence (support / (support + violations)).
    pub min_confidence: f64,
    /// Minimum lift over baseline probability.
    pub min_lift: f64,
    /// Maximum number of atoms in the rule body (1 = bigram only, 2 = bigram + trigram).
    pub max_body_size: usize,
    /// Maximum temporal gap in hours between body and head events.
    pub max_temporal_gap_hours: Option<f64>,
    /// Whether to learn rules across multiple narratives.
    pub cross_narrative: bool,
}

impl Default for TemporalILPConfig {
    fn default() -> Self {
        Self {
            min_support: 3,
            min_confidence: 0.5,
            min_lift: 1.2,
            max_body_size: 2,
            max_temporal_gap_hours: None,
            cross_narrative: true,
        }
    }
}

// ─── Rule Types ────────────────────────────────────────────

/// A single atom in a temporal Horn clause (describes an entity-situation step).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleAtom {
    pub entity_type: Option<EntityType>,
    pub role: Option<Role>,
    pub narrative_level: Option<NarrativeLevel>,
    /// Allen relation to the next atom in the body (TILR extension).
    /// If Some, this atom's situation must have this relation to the next atom's situation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation_to_next: Option<AllenRelation>,
}

impl std::fmt::Display for RuleAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let et = self
            .entity_type
            .as_ref()
            .map(|t| format!("{:?}", t))
            .unwrap_or_else(|| "*".into());
        let role = self
            .role
            .as_ref()
            .map(|r| format!("{:?}", r))
            .unwrap_or_else(|| "*".into());
        let level = self
            .narrative_level
            .as_ref()
            .map(|l| format!("{:?}", l))
            .unwrap_or_else(|| "*".into());
        write!(f, "{}/{} in {}", et, role, level)
    }
}

/// A learned temporal Horn clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRule {
    pub id: String,
    pub head: RuleAtom,
    pub body: Vec<RuleAtom>,
    pub temporal_constraint: AllenRelation,
    pub temporal_gap_hours: Option<f64>,
    pub support: usize,
    pub confidence: f64,
    pub lift: f64,
    pub narrative_ids: Vec<String>,
}

/// Full result of a Temporal ILP mining job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalILPResult {
    pub narrative_id: String,
    pub rules: Vec<TemporalRule>,
    pub config: TemporalILPConfig,
    pub total_sequences_scanned: usize,
}

// ─── Internal types ────────────────────────────────────────

/// A single step in an entity's chronological situation sequence.
#[derive(Debug, Clone)]
pub(crate) struct SequenceStep {
    role: Role,
    narrative_level: NarrativeLevel,
    temporal: AllenInterval,
}

/// An entity's full chronological participation sequence.
#[derive(Debug)]
pub(crate) struct EntitySequence {
    entity_type: EntityType,
    steps: Vec<SequenceStep>,
    narrative_id: String,
}

/// Hashable signature for counting candidate rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RuleSignature {
    body: Vec<RuleAtom>,
    head: RuleAtom,
    temporal_constraint: AllenRelation,
}

/// Accumulator for counting support and tracking provenance.
#[derive(Debug, Default)]
struct RuleAccumulator {
    support: usize,
    narrative_ids: Vec<String>,
    gap_hours_sum: f64,
    gap_count: usize,
}

// ─── Engine ────────────────────────────────────────────────

/// Temporal ILP inference engine (unit struct — stateless).
pub struct TemporalILPEngine;

impl InferenceEngine for TemporalILPEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::TemporalILP
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(5000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;

        // Parse optional config overrides from job parameters
        let mut config = TemporalILPConfig::default();
        if let Some(v) = job.parameters.get("min_support").and_then(|v| v.as_u64()) {
            config.min_support = v as usize;
        }
        if let Some(v) = job
            .parameters
            .get("min_confidence")
            .and_then(|v| v.as_f64())
        {
            config.min_confidence = v;
        }
        if let Some(v) = job.parameters.get("min_lift").and_then(|v| v.as_f64()) {
            config.min_lift = v;
        }
        if let Some(v) = job.parameters.get("max_body_size").and_then(|v| v.as_u64()) {
            config.max_body_size = v as usize;
        }
        if let Some(v) = job
            .parameters
            .get("max_temporal_gap_hours")
            .and_then(|v| v.as_f64())
        {
            config.max_temporal_gap_hours = Some(v);
        }

        let sequences = extract_entity_sequences(hypergraph, narrative_id)?;
        let total_sequences = sequences.len();
        let rules = mine_rules(&sequences, &config);

        // Persist rules to KV
        let result_data = TemporalILPResult {
            narrative_id: narrative_id.to_string(),
            rules,
            config: config.clone(),
            total_sequences_scanned: total_sequences,
        };

        let result_value = serde_json::to_value(&result_data)?;
        let key = analysis_key(ANALYSIS_TEMPORAL_ILP, &[narrative_id]);
        hypergraph
            .store()
            .put(&key, result_value.to_string().as_bytes())?;

        let num_rules = result_data.rules.len();
        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::TemporalILP,
            target_id: job.target_id,
            result: result_value,
            confidence: if num_rules == 0 { 0.0 } else { 0.8 },
            explanation: Some(format!(
                "Temporal ILP: {} rules mined from {} entity sequences \
                 (min_support={}, min_confidence={:.2}, min_lift={:.2})",
                num_rules,
                total_sequences,
                config.min_support,
                config.min_confidence,
                config.min_lift,
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Core Algorithm ────────────────────────────────────────

/// Extract per-entity chronological situation sequences from a narrative.
pub(crate) fn extract_entity_sequences(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<EntitySequence>> {
    let situations = load_sorted_situations(hypergraph, narrative_id)?;
    if situations.is_empty() {
        return Ok(vec![]);
    }

    // Cache entity lookups
    let mut entity_cache: HashMap<Uuid, EntityType> = HashMap::new();
    // entity_id -> Vec<(step, narrative_id)>
    let mut entity_steps: HashMap<Uuid, Vec<SequenceStep>> = HashMap::new();

    for sit in &situations {
        let participants = hypergraph.get_participants_for_situation(&sit.id)?;
        for p in participants {
            // Populate entity type cache on first encounter
            if !entity_cache.contains_key(&p.entity_id) {
                let entity = hypergraph.get_entity(&p.entity_id)?;
                entity_cache.insert(p.entity_id, entity.entity_type.clone());
            }

            entity_steps
                .entry(p.entity_id)
                .or_default()
                .push(SequenceStep {
                    role: p.role.clone(),
                    narrative_level: sit.narrative_level,
                    temporal: sit.temporal.clone(),
                });
        }
    }

    let sequences = entity_steps
        .into_iter()
        .map(|(eid, steps)| {
            let entity_type = entity_cache.get(&eid).cloned().unwrap_or(EntityType::Actor);
            EntitySequence {
                entity_type,
                steps,
                narrative_id: narrative_id.to_string(),
            }
        })
        .collect();

    Ok(sequences)
}

/// Mine temporal rules from entity sequences.
pub(crate) fn mine_rules(
    sequences: &[EntitySequence],
    config: &TemporalILPConfig,
) -> Vec<TemporalRule> {
    if sequences.is_empty() {
        return vec![];
    }

    // Phase 1: Count candidate rules via sliding window
    let mut candidates: HashMap<RuleSignature, RuleAccumulator> = HashMap::new();
    // Also count head occurrences for baseline probability
    let mut head_counts: HashMap<RuleAtom, usize> = HashMap::new();
    // Pre-compute body occurrence counts to avoid O(C*S*L) re-scanning in Phase 2
    let mut body_counts: HashMap<Vec<RuleAtom>, usize> = HashMap::new();
    let mut total_steps: usize = 0;

    for seq in sequences {
        total_steps += seq.steps.len();

        // Count all atoms for baseline
        for step in &seq.steps {
            let atom = atom_from_step(step, &seq.entity_type);
            *head_counts.entry(atom).or_default() += 1;
        }

        // Bigram candidates
        for i in 0..seq.steps.len().saturating_sub(1) {
            let body_atom = atom_from_step(&seq.steps[i], &seq.entity_type);
            let head_atom = atom_from_step(&seq.steps[i + 1], &seq.entity_type);

            // Count body occurrences for confidence calculation
            *body_counts.entry(vec![body_atom.clone()]).or_default() += 1;

            let allen = match relation_between(&seq.steps[i].temporal, &seq.steps[i + 1].temporal) {
                Ok(r) => r,
                Err(_) => continue, // skip pairs with missing timestamps
            };

            let gap_hours = compute_gap_hours(&seq.steps[i].temporal, &seq.steps[i + 1].temporal);

            // Skip if exceeds max temporal gap
            if let (Some(max_gap), Some(gap)) = (config.max_temporal_gap_hours, gap_hours) {
                if gap > max_gap {
                    continue;
                }
            }

            let sig = RuleSignature {
                body: vec![body_atom],
                head: head_atom,
                temporal_constraint: allen,
            };

            let acc = candidates.entry(sig).or_default();
            acc.support += 1;
            if !acc.narrative_ids.contains(&seq.narrative_id) {
                acc.narrative_ids.push(seq.narrative_id.clone());
            }
            if let Some(g) = gap_hours {
                acc.gap_hours_sum += g;
                acc.gap_count += 1;
            }
        }

        // Trigram candidates (body size = 2)
        if config.max_body_size >= 2 {
            for i in 0..seq.steps.len().saturating_sub(2) {
                let body1 = atom_from_step(&seq.steps[i], &seq.entity_type);
                let body2 = atom_from_step(&seq.steps[i + 1], &seq.entity_type);
                let head_atom = atom_from_step(&seq.steps[i + 2], &seq.entity_type);

                // Count body occurrences for confidence calculation
                *body_counts
                    .entry(vec![body1.clone(), body2.clone()])
                    .or_default() += 1;

                let allen = match relation_between(
                    &seq.steps[i + 1].temporal,
                    &seq.steps[i + 2].temporal,
                ) {
                    Ok(r) => r,
                    Err(_) => continue, // skip pairs with missing timestamps
                };

                let gap_hours =
                    compute_gap_hours(&seq.steps[i + 1].temporal, &seq.steps[i + 2].temporal);

                if let (Some(max_gap), Some(gap)) = (config.max_temporal_gap_hours, gap_hours) {
                    if gap > max_gap {
                        continue;
                    }
                }

                let sig = RuleSignature {
                    body: vec![body1, body2],
                    head: head_atom,
                    temporal_constraint: allen,
                };

                let acc = candidates.entry(sig).or_default();
                acc.support += 1;
                if !acc.narrative_ids.contains(&seq.narrative_id) {
                    acc.narrative_ids.push(seq.narrative_id.clone());
                }
                if let Some(g) = gap_hours {
                    acc.gap_hours_sum += g;
                    acc.gap_count += 1;
                }
            }
        }
    }

    if total_steps == 0 {
        return vec![];
    }

    // Phase 2: Score and filter
    let mut rules: Vec<TemporalRule> = Vec::new();

    for (sig, acc) in &candidates {
        if acc.support < config.min_support {
            continue;
        }

        // Confidence = support / body_occurrences (pre-computed in Phase 1)
        let body_occurrences = body_counts.get(&sig.body).copied().unwrap_or(0);
        let confidence = if body_occurrences > 0 {
            acc.support as f64 / body_occurrences as f64
        } else {
            0.0
        };

        if confidence < config.min_confidence {
            continue;
        }

        // Compute lift
        let head_baseline =
            head_counts.get(&sig.head).copied().unwrap_or(0) as f64 / total_steps.max(1) as f64;
        let lift = if head_baseline > 0.001 {
            confidence / head_baseline
        } else {
            confidence / 0.001
        };

        if lift < config.min_lift {
            continue;
        }

        let avg_gap = if acc.gap_count > 0 {
            Some(acc.gap_hours_sum / acc.gap_count as f64)
        } else {
            None
        };

        rules.push(TemporalRule {
            id: Uuid::now_v7().to_string(),
            head: sig.head.clone(),
            body: sig.body.clone(),
            temporal_constraint: sig.temporal_constraint,
            temporal_gap_hours: avg_gap,
            support: acc.support,
            confidence,
            lift,
            narrative_ids: acc.narrative_ids.clone(),
        });
    }

    // Phase 3: Sort by confidence descending, then prune subsumed
    rules.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    prune_subsumed(&mut rules);

    rules
}

// ─── Helpers ───────────────────────────────────────────────

/// Build a RuleAtom from a sequence step.
fn atom_from_step(step: &SequenceStep, entity_type: &EntityType) -> RuleAtom {
    RuleAtom {
        entity_type: Some(entity_type.clone()),
        role: Some(step.role.clone()),
        narrative_level: Some(step.narrative_level),
        relation_to_next: None,
    }
}

/// Create a RuleAtom with an Allen relation to the next body atom (TILR extension).
#[allow(dead_code)]
fn atom_from_step_with_relation(
    step: &SequenceStep,
    entity_type: &EntityType,
    next_step: &SequenceStep,
) -> RuleAtom {
    let relation =
        crate::temporal::interval::relation_between(&step.temporal, &next_step.temporal).ok();
    RuleAtom {
        entity_type: Some(entity_type.clone()),
        role: Some(step.role.clone()),
        narrative_level: Some(step.narrative_level),
        relation_to_next: relation,
    }
}

/// Compute temporal gap in hours between two intervals.
fn compute_gap_hours(a: &AllenInterval, b: &AllenInterval) -> Option<f64> {
    let a_end = a.end?;
    let b_start = b.start?;
    let diff = b_start.signed_duration_since(a_end);
    Some(diff.num_milliseconds() as f64 / 3_600_000.0)
}

/// Remove rules subsumed by a more specific rule with similar confidence.
fn prune_subsumed(rules: &mut Vec<TemporalRule>) {
    let mut keep = vec![true; rules.len()];

    for i in 0..rules.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..rules.len() {
            if !keep[j] {
                continue;
            }
            // Rule i subsumes rule j if:
            // - Same head
            // - i's body is a superset of j's body (more specific)
            // - i's confidence >= j's confidence - 0.05
            if rules[i].head == rules[j].head
                && rules[i].temporal_constraint == rules[j].temporal_constraint
                && rules[i].body.len() > rules[j].body.len()
                && rules[j]
                    .body
                    .iter()
                    .all(|atom| rules[i].body.contains(atom))
                && rules[i].confidence >= rules[j].confidence - 0.05
            {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    rules.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, make_hg};
    use chrono::{DateTime, Duration};

    /// Create a situation with explicit temporal offset (minutes from base time).
    fn add_situation_at(
        hg: &Hypergraph,
        narrative: &str,
        level: NarrativeLevel,
        base: DateTime<Utc>,
        offset_min: i64,
        name: Option<&str>,
    ) -> Uuid {
        let start = base + Duration::minutes(offset_min);
        let end = start + Duration::minutes(5);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: name.map(String::from),
            description: None,
            temporal: AllenInterval {
                start: Some(start),
                end: Some(end),
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
            raw_content: vec![ContentBlock::text("test")],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.to_string()),
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
        hg.create_situation(sit).unwrap()
    }

    fn link_with_role(hg: &Hypergraph, entity: Uuid, situation: Uuid, role: Role) {
        hg.add_participant(Participation {
            entity_id: entity,
            situation_id: situation,
            role,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();
    }

    #[test]
    fn test_empty_narrative() {
        let hg = make_hg();
        let seqs = extract_entity_sequences(&hg, "empty").unwrap();
        assert!(seqs.is_empty());
        let rules = mine_rules(&seqs, &TemporalILPConfig::default());
        assert!(rules.is_empty());
    }

    #[test]
    fn test_insufficient_data() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Alice", "n1");
        let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 0, None);
        link_with_role(&hg, e, s1, Role::Protagonist);

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        let config = TemporalILPConfig {
            min_support: 3,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &config);
        assert!(rules.is_empty());
    }

    #[test]
    fn test_single_entity_bigram() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Bob", "n1");

        // Create 4 situations for one entity
        for i in 0..4 {
            let s = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, i * 10, None);
            link_with_role(&hg, e, s, Role::Protagonist);
        }

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0].steps.len(), 4);

        // With min_support=1, should find bigram rules
        let config = TemporalILPConfig {
            min_support: 1,
            min_confidence: 0.0,
            min_lift: 0.0,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &config);
        assert!(!rules.is_empty(), "Should produce at least one bigram rule");
    }

    #[test]
    fn test_support_counting() {
        let hg = make_hg();
        let base = Utc::now();

        // 3 entities all showing same pattern: Protagonist in Scene -> Protagonist in Scene
        for i in 0..3 {
            let e = add_entity(&hg, &format!("Entity{}", i), "n1");
            let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, i as i64 * 100, None);
            let s2 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Scene,
                base,
                i as i64 * 100 + 10,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Protagonist);
        }

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        let config = TemporalILPConfig {
            min_support: 3,
            min_confidence: 0.0,
            min_lift: 0.0,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &config);
        assert!(
            rules.iter().any(|r| r.support >= 3),
            "At least one rule should have support >= 3"
        );
    }

    #[test]
    fn test_confidence_filtering() {
        let hg = make_hg();
        let base = Utc::now();

        // 2 entities with pattern, 4 entities without pattern completion
        // Pattern: Protagonist/Scene -> Antagonist/Scene
        for i in 0..2 {
            let e = add_entity(&hg, &format!("Match{}", i), "n1");
            let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, i as i64 * 100, None);
            let s2 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Scene,
                base,
                i as i64 * 100 + 10,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Antagonist);
        }
        // Add 4 entities that start with Protagonist but go to Witness (violations)
        for i in 0..4 {
            let e = add_entity(&hg, &format!("Violate{}", i), "n1");
            let s1 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Scene,
                base,
                200 + i as i64 * 100,
                None,
            );
            let s2 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Scene,
                base,
                200 + i as i64 * 100 + 10,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Witness);
        }

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        let high_conf = TemporalILPConfig {
            min_support: 1,
            min_confidence: 0.8,
            min_lift: 0.0,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &high_conf);
        // Protagonist -> Antagonist has 2/6 = 0.33 confidence, should be filtered
        assert!(
            !rules
                .iter()
                .any(|r| r.head.role == Some(Role::Antagonist) && r.confidence < 0.8),
            "Low-confidence Protagonist->Antagonist rule should be filtered"
        );
    }

    #[test]
    fn test_lift_filtering() {
        let hg = make_hg();
        let base = Utc::now();

        // Create a very common pattern (Protagonist/Scene -> Protagonist/Scene)
        // with many examples so baseline is high, making lift close to 1.0
        for i in 0..10 {
            let e = add_entity(&hg, &format!("E{}", i), "n1");
            let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, i as i64 * 20, None);
            let s2 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Scene,
                base,
                i as i64 * 20 + 10,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Protagonist);
        }

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        let high_lift = TemporalILPConfig {
            min_support: 1,
            min_confidence: 0.0,
            min_lift: 5.0, // very high lift requirement
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &high_lift);
        // When everything is the same pattern, lift should be ~1.0, filtering everything
        for r in &rules {
            assert!(
                r.lift >= 5.0,
                "Rule with lift {} should have been filtered",
                r.lift
            );
        }
    }

    #[test]
    fn test_allen_relation_detected() {
        let hg = make_hg();
        let base = Utc::now();
        let e = add_entity(&hg, "Carol", "n1");

        // Non-overlapping (Before): s1 ends before s2 starts
        let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 0, None);
        let s2 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, 30, None);
        link_with_role(&hg, e, s1, Role::Protagonist);
        link_with_role(&hg, e, s2, Role::Protagonist);

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        let config = TemporalILPConfig {
            min_support: 1,
            min_confidence: 0.0,
            min_lift: 0.0,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &config);

        // Should detect Before relation (s1 ends at base+5min, s2 starts at base+30min)
        assert!(
            rules
                .iter()
                .any(|r| r.temporal_constraint == AllenRelation::Before),
            "Should detect Before relation; got {:?}",
            rules
                .iter()
                .map(|r| r.temporal_constraint)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_trigram_rules() {
        let hg = make_hg();
        let base = Utc::now();

        // 3 entities with 3-step sequences
        for i in 0..3 {
            let e = add_entity(&hg, &format!("Tri{}", i), "n1");
            let s1 = add_situation_at(&hg, "n1", NarrativeLevel::Scene, base, i as i64 * 100, None);
            let s2 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Event,
                base,
                i as i64 * 100 + 10,
                None,
            );
            let s3 = add_situation_at(
                &hg,
                "n1",
                NarrativeLevel::Beat,
                base,
                i as i64 * 100 + 20,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Witness);
            link_with_role(&hg, e, s3, Role::Target);
        }

        let seqs = extract_entity_sequences(&hg, "n1").unwrap();
        let config = TemporalILPConfig {
            min_support: 3,
            min_confidence: 0.0,
            min_lift: 0.0,
            max_body_size: 2,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &config);

        // Should find trigram rule with 2-atom body
        assert!(
            rules.iter().any(|r| r.body.len() == 2),
            "Should have at least one trigram rule (body size 2)"
        );
    }

    #[test]
    fn test_pruning_removes_redundant() {
        // Create rules manually and test pruning
        let atom_a = RuleAtom {
            entity_type: Some(EntityType::Actor),
            role: Some(Role::Protagonist),
            narrative_level: Some(NarrativeLevel::Scene),
            relation_to_next: None,
        };
        let atom_b = RuleAtom {
            entity_type: Some(EntityType::Actor),
            role: Some(Role::Witness),
            narrative_level: Some(NarrativeLevel::Event),
            relation_to_next: None,
        };
        let head = RuleAtom {
            entity_type: Some(EntityType::Actor),
            role: Some(Role::Target),
            narrative_level: Some(NarrativeLevel::Beat),
            relation_to_next: None,
        };

        let mut rules = vec![
            // More specific (body = [a, b]) with higher confidence
            TemporalRule {
                id: "specific".into(),
                head: head.clone(),
                body: vec![atom_a.clone(), atom_b.clone()],
                temporal_constraint: AllenRelation::Before,
                temporal_gap_hours: None,
                support: 5,
                confidence: 0.9,
                lift: 2.0,
                narrative_ids: vec!["n1".into()],
            },
            // More general (body = [a]) with similar confidence — should be pruned
            TemporalRule {
                id: "general".into(),
                head: head.clone(),
                body: vec![atom_a.clone()],
                temporal_constraint: AllenRelation::Before,
                temporal_gap_hours: None,
                support: 8,
                confidence: 0.88,
                lift: 1.8,
                narrative_ids: vec!["n1".into()],
            },
        ];

        prune_subsumed(&mut rules);
        assert_eq!(rules.len(), 1, "General rule should be pruned");
        assert_eq!(rules[0].id, "specific");
    }

    #[test]
    fn test_engine_execute() {
        let hg = make_hg();
        let base = Utc::now();

        // Create a pattern with sufficient support
        for i in 0..3 {
            let e = add_entity(&hg, &format!("E{}", i), "test-ilp");
            let s1 = add_situation_at(
                &hg,
                "test-ilp",
                NarrativeLevel::Scene,
                base,
                i as i64 * 100,
                None,
            );
            let s2 = add_situation_at(
                &hg,
                "test-ilp",
                NarrativeLevel::Scene,
                base,
                i as i64 * 100 + 10,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Protagonist);
        }

        let engine = TemporalILPEngine;
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::TemporalILP,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": "test-ilp"}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.explanation.is_some());

        let parsed: TemporalILPResult = serde_json::from_value(result.result).unwrap();
        assert_eq!(parsed.narrative_id, "test-ilp");
        assert_eq!(parsed.total_sequences_scanned, 3);
    }

    #[test]
    fn test_result_serde() {
        let result = TemporalILPResult {
            narrative_id: "test".to_string(),
            rules: vec![TemporalRule {
                id: "r1".into(),
                head: RuleAtom {
                    entity_type: Some(EntityType::Actor),
                    role: Some(Role::Protagonist),
                    narrative_level: Some(NarrativeLevel::Scene),
                    relation_to_next: None,
                },
                body: vec![RuleAtom {
                    entity_type: Some(EntityType::Actor),
                    role: Some(Role::Witness),
                    narrative_level: Some(NarrativeLevel::Event),
                    relation_to_next: None,
                }],
                temporal_constraint: AllenRelation::Before,
                temporal_gap_hours: Some(2.5),
                support: 10,
                confidence: 0.85,
                lift: 2.3,
                narrative_ids: vec!["n1".into(), "n2".into()],
            }],
            config: TemporalILPConfig::default(),
            total_sequences_scanned: 42,
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: TemporalILPResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rules.len(), 1);
        assert_eq!(parsed.rules[0].support, 10);
        assert_eq!(parsed.rules[0].temporal_constraint, AllenRelation::Before);
    }

    #[test]
    fn test_cross_narrative() {
        let hg = make_hg();
        let base = Utc::now();

        // Same pattern in two different narratives
        for (nid, offset) in [("alpha", 0i64), ("beta", 1000)] {
            for i in 0..2 {
                let e = add_entity(&hg, &format!("{}-e{}", nid, i), nid);
                let s1 = add_situation_at(
                    &hg,
                    nid,
                    NarrativeLevel::Scene,
                    base,
                    offset + i as i64 * 100,
                    None,
                );
                let s2 = add_situation_at(
                    &hg,
                    nid,
                    NarrativeLevel::Scene,
                    base,
                    offset + i as i64 * 100 + 10,
                    None,
                );
                link_with_role(&hg, e, s1, Role::Protagonist);
                link_with_role(&hg, e, s2, Role::Protagonist);
            }
        }

        // Combine sequences from both narratives
        let mut seqs = extract_entity_sequences(&hg, "alpha").unwrap();
        seqs.extend(extract_entity_sequences(&hg, "beta").unwrap());

        let config = TemporalILPConfig {
            min_support: 4,
            min_confidence: 0.0,
            min_lift: 0.0,
            ..Default::default()
        };
        let rules = mine_rules(&seqs, &config);

        // Should find a rule with support=4 (2 from each narrative)
        assert!(
            rules.iter().any(|r| r.support >= 4),
            "Cross-narrative rule should have support >= 4"
        );
        // Should have both narrative IDs
        if let Some(r) = rules.iter().find(|r| r.support >= 4) {
            assert!(
                r.narrative_ids.len() >= 2,
                "Should reference both narratives"
            );
        }
    }

    #[test]
    fn test_kv_persistence() {
        let hg = make_hg();
        let base = Utc::now();

        for i in 0..3 {
            let e = add_entity(&hg, &format!("P{}", i), "persist-test");
            let s1 = add_situation_at(
                &hg,
                "persist-test",
                NarrativeLevel::Scene,
                base,
                i as i64 * 100,
                None,
            );
            let s2 = add_situation_at(
                &hg,
                "persist-test",
                NarrativeLevel::Scene,
                base,
                i as i64 * 100 + 10,
                None,
            );
            link_with_role(&hg, e, s1, Role::Protagonist);
            link_with_role(&hg, e, s2, Role::Protagonist);
        }

        let engine = TemporalILPEngine;
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::TemporalILP,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": "persist-test"}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        engine.execute(&job, &hg).unwrap();

        // Verify KV persistence
        let key = analysis_key(ANALYSIS_TEMPORAL_ILP, &["persist-test"]);
        let stored = hg.store().get(&key).unwrap();
        assert!(stored.is_some(), "Result should be persisted in KV");

        let parsed: TemporalILPResult = serde_json::from_slice(&stored.unwrap()).unwrap();
        assert_eq!(parsed.narrative_id, "persist-test");
    }
}
