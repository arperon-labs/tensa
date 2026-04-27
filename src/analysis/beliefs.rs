//! Recursive belief modeling (depth 2).
//!
//! Extends the InfoSet model from "what does A know" to "what does A
//! think B knows." Enables epistemic queries like "does A realize
//! their cover is blown?"
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! SymbolicToM fusion here is **set-theoretic**, not arithmetic — belief
//! sets are combined via `difference` / `union` on `HashSet<Uuid>`. There
//! are no confidence-combination call sites for Phase 1 to rewire. If
//! future depth-3+ extensions introduce graded belief strengths, thread
//! `fuzzy::tnorm::TNormKind` with a default-Gödel hook.
//!
//! Cites: [klement2000].

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id, load_sorted_situations};
use crate::error::Result;
use crate::hypergraph::{keys, Hypergraph};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

// ─── Data Structures ────────────────────────────────────────

/// A single knowledge fact (simplified key for set operations).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FactKey {
    pub about_entity: Uuid,
    pub fact: String,
}

impl From<&KnowledgeFact> for FactKey {
    fn from(kf: &KnowledgeFact) -> Self {
        FactKey {
            about_entity: kf.about_entity,
            fact: kf.fact.clone(),
        }
    }
}

/// The belief state of entity A about entity B at a specific situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefSnapshot {
    pub entity_a: Uuid,
    pub entity_b: Uuid,
    pub situation_id: Uuid,
    /// What A thinks B knows at this point.
    pub believed_knowledge: HashSet<FactKey>,
    /// What B actually knows at this point.
    pub actual_knowledge: HashSet<FactKey>,
}

/// A belief gap: where A's model of B differs from reality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefGap {
    pub entity_a: Uuid,
    pub entity_b: Uuid,
    pub at_situation: Uuid,
    /// Facts B knows but A doesn't think B knows.
    pub unknown_to_a: Vec<FactKey>,
    /// Facts A thinks B knows but B actually doesn't.
    pub false_beliefs: Vec<FactKey>,
}

/// Full belief analysis for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefAnalysis {
    pub narrative_id: String,
    pub snapshots: Vec<BeliefSnapshot>,
    pub gaps: Vec<BeliefGap>,
}

// ─── SymbolicToM — text-based initial belief seeding ────────

/// Parsed initial belief state for seeding the belief pipeline.
/// Maps entity → set of facts they initially know (from `beliefs_about_others`
/// in InfoSet or text-heuristic extraction).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolicToMState {
    /// Initial actual knowledge per entity: entity_id → known facts.
    pub initial_knowledge: HashMap<Uuid, HashSet<FactKey>>,
    /// Initial belief models: (entity_a, entity_b) → facts A thinks B knows.
    pub initial_beliefs: HashMap<(Uuid, Uuid), HashSet<FactKey>>,
}

/// Parse initial belief states from participation data.
///
/// Reads `beliefs_about_others` from InfoSets in the first situation each entity
/// appears in, seeding the belief model before the 4-phase update pipeline.
/// This bridges LLM text understanding with formal belief modeling: an LLM
/// populates `beliefs_about_others` during ingestion, and this function
/// converts those into the initial `SymbolicToMState`.
pub fn parse_symbolic_tom(hypergraph: &Hypergraph, narrative_id: &str) -> Result<SymbolicToMState> {
    let situations = load_sorted_situations(hypergraph, narrative_id)?;
    parse_symbolic_tom_from_situations(hypergraph, &situations)
}

/// Inner implementation that works on pre-loaded situations to avoid
/// redundant KV scans when called from `run_beliefs`.
fn parse_symbolic_tom_from_situations(
    hypergraph: &Hypergraph,
    situations: &[crate::types::Situation],
) -> Result<SymbolicToMState> {
    let mut state = SymbolicToMState::default();
    let mut seen_entities: HashSet<Uuid> = HashSet::new();

    for situation in situations {
        let participants = hypergraph.get_participants_for_situation(&situation.id)?;
        for p in &participants {
            // Only seed from the first situation each entity appears in.
            if seen_entities.contains(&p.entity_id) {
                continue;
            }
            seen_entities.insert(p.entity_id);

            if let Some(info) = &p.info_set {
                // Seed actual knowledge from knows_before.
                let knowledge = state.initial_knowledge.entry(p.entity_id).or_default();
                for kf in &info.knows_before {
                    knowledge.insert(FactKey::from(kf));
                }

                // Seed beliefs from beliefs_about_others (the ToM layer).
                for rb in &info.beliefs_about_others {
                    let beliefs = state
                        .initial_beliefs
                        .entry((p.entity_id, rb.about_entity))
                        .or_default();
                    for kf in &rb.believed_knowledge {
                        beliefs.insert(FactKey::from(kf));
                    }
                }
            }
        }
    }

    Ok(state)
}

// ─── Core Algorithm ─────────────────────────────────────────

/// Run belief state computation for a narrative.
/// Traverses situations chronologically, tracking what each entity
/// actually knows and what each entity thinks others know.
pub fn run_beliefs(hypergraph: &Hypergraph, narrative_id: &str) -> Result<BeliefAnalysis> {
    // Load situations once, shared between ToM parsing and the main pipeline.
    let situations = load_sorted_situations(hypergraph, narrative_id)?;
    let tom = parse_symbolic_tom_from_situations(hypergraph, &situations)?;
    run_beliefs_with_tom_inner(hypergraph, narrative_id, &tom, situations)
}

/// Run belief computation seeded with a pre-parsed SymbolicToM state.
///
/// The ToM state provides initial knowledge and belief models that are
/// populated before the 4-phase update pipeline begins processing situations.
pub fn run_beliefs_with_tom(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    tom: &SymbolicToMState,
) -> Result<BeliefAnalysis> {
    let situations = load_sorted_situations(hypergraph, narrative_id)?;
    run_beliefs_with_tom_inner(hypergraph, narrative_id, tom, situations)
}

fn run_beliefs_with_tom_inner(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    tom: &SymbolicToMState,
    situations: Vec<crate::types::Situation>,
) -> Result<BeliefAnalysis> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();

    if entity_ids.is_empty() || situations.is_empty() {
        return Ok(BeliefAnalysis {
            narrative_id: narrative_id.to_string(),
            snapshots: vec![],
            gaps: vec![],
        });
    }

    // Actual knowledge state per entity, seeded from ToM.
    let mut actual_knowledge: HashMap<Uuid, HashSet<FactKey>> = HashMap::new();
    for &eid in &entity_ids {
        let initial = tom.initial_knowledge.get(&eid).cloned().unwrap_or_default();
        actual_knowledge.insert(eid, initial);
    }

    // A's model of B's knowledge, seeded from ToM beliefs_about_others.
    let mut beliefs: HashMap<Uuid, HashMap<Uuid, HashSet<FactKey>>> = HashMap::new();
    for &a in &entity_ids {
        let mut inner = HashMap::new();
        for &b in &entity_ids {
            if a != b {
                let initial = tom
                    .initial_beliefs
                    .get(&(a, b))
                    .cloned()
                    .unwrap_or_default();
                inner.insert(b, initial);
            }
        }
        beliefs.insert(a, inner);
    }

    let mut all_snapshots: Vec<BeliefSnapshot> = Vec::new();
    let mut all_gaps: Vec<BeliefGap> = Vec::new();

    for situation in &situations {
        let participants = hypergraph.get_participants_for_situation(&situation.id)?;
        let participant_ids: HashSet<Uuid> = participants.iter().map(|p| p.entity_id).collect();

        // Phase 1: Update actual knowledge from InfoSet.
        for p in &participants {
            if let Some(info) = &p.info_set {
                let knowledge = actual_knowledge.entry(p.entity_id).or_default();
                for kf in &info.knows_before {
                    knowledge.insert(FactKey::from(kf));
                }
                for kf in &info.learns {
                    knowledge.insert(FactKey::from(kf));
                }
            }
        }

        // Phase 2: Collect revealed facts in this situation.
        let mut revealed_facts: HashSet<FactKey> = HashSet::new();
        let mut revealers: HashSet<Uuid> = HashSet::new();
        for p in &participants {
            if let Some(info) = &p.info_set {
                for kf in &info.reveals {
                    revealed_facts.insert(FactKey::from(kf));
                    revealers.insert(p.entity_id);
                }
            }
        }

        // Build per-entity info set lookup (avoids O(P) scan per pair).
        let participant_info: HashMap<Uuid, &InfoSet> = participants
            .iter()
            .filter_map(|p| p.info_set.as_ref().map(|info| (p.entity_id, info)))
            .collect();

        // Phase 3: Update belief models for co-present entities.
        // If A and B are both present and facts are revealed,
        // A knows B now knows those revealed facts.
        for &a in &participant_ids {
            for &b in &participant_ids {
                if a == b {
                    continue;
                }
                if let Some(a_beliefs) = beliefs.get_mut(&a) {
                    if let Some(a_model_of_b) = a_beliefs.get_mut(&b) {
                        // A knows B heard whatever was revealed in this situation.
                        for fact in &revealed_facts {
                            a_model_of_b.insert(fact.clone());
                        }
                        // A also knows B learned whatever B's InfoSet says they learned
                        // (since A is present and can observe).
                        if let Some(info) = participant_info.get(&b) {
                            for kf in &info.learns {
                                a_model_of_b.insert(FactKey::from(kf));
                            }
                        }
                    }
                }
            }
        }

        // Phase 4: Record snapshots and compute gaps for all entity pairs present.
        for &a in &participant_ids {
            for &b in &participant_ids {
                if a == b {
                    continue;
                }
                let a_model = beliefs
                    .get(&a)
                    .and_then(|m| m.get(&b))
                    .cloned()
                    .unwrap_or_default();
                let b_actual = actual_knowledge.get(&b).cloned().unwrap_or_default();

                let snapshot = BeliefSnapshot {
                    entity_a: a,
                    entity_b: b,
                    situation_id: situation.id,
                    believed_knowledge: a_model.clone(),
                    actual_knowledge: b_actual.clone(),
                };

                // Store in KV.
                let key = analysis_key(
                    keys::ANALYSIS_BELIEF,
                    &[
                        narrative_id,
                        &a.to_string(),
                        &b.to_string(),
                        &situation.id.to_string(),
                    ],
                );
                let bytes = serde_json::to_vec(&snapshot)?;
                hypergraph.store().put(&key, &bytes)?;

                // Compute gap.
                let unknown_to_a: Vec<FactKey> = b_actual.difference(&a_model).cloned().collect();
                let false_beliefs: Vec<FactKey> = a_model.difference(&b_actual).cloned().collect();

                if !unknown_to_a.is_empty() || !false_beliefs.is_empty() {
                    all_gaps.push(BeliefGap {
                        entity_a: a,
                        entity_b: b,
                        at_situation: situation.id,
                        unknown_to_a,
                        false_beliefs,
                    });
                }

                all_snapshots.push(snapshot);
            }
        }
    }

    Ok(BeliefAnalysis {
        narrative_id: narrative_id.to_string(),
        snapshots: all_snapshots,
        gaps: all_gaps,
    })
}

/// Query what A thinks B knows at a specific situation.
pub fn query_belief_at(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entity_a: Uuid,
    entity_b: Uuid,
    at_situation: Uuid,
) -> Result<Option<BeliefSnapshot>> {
    let key = analysis_key(
        keys::ANALYSIS_BELIEF,
        &[
            narrative_id,
            &entity_a.to_string(),
            &entity_b.to_string(),
            &at_situation.to_string(),
        ],
    );
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Query belief gaps between A and B at a specific situation.
pub fn query_belief_gap(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entity_a: Uuid,
    entity_b: Uuid,
    at_situation: Uuid,
) -> Result<Option<BeliefGap>> {
    let snapshot = query_belief_at(hypergraph, narrative_id, entity_a, entity_b, at_situation)?;
    match snapshot {
        Some(snap) => {
            let unknown: Vec<FactKey> = snap
                .actual_knowledge
                .difference(&snap.believed_knowledge)
                .cloned()
                .collect();
            let false_beliefs: Vec<FactKey> = snap
                .believed_knowledge
                .difference(&snap.actual_knowledge)
                .cloned()
                .collect();

            if unknown.is_empty() && false_beliefs.is_empty() {
                Ok(None)
            } else {
                Ok(Some(BeliefGap {
                    entity_a,
                    entity_b,
                    at_situation,
                    unknown_to_a: unknown,
                    false_beliefs,
                }))
            }
        }
        None => Ok(None),
    }
}

// ─── InferenceEngine ────────────────────────────────────────

pub struct BeliefEngine;

impl InferenceEngine for BeliefEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::BeliefModeling
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(4000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let analysis = run_beliefs(hypergraph, narrative_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::BeliefModeling,
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Belief analysis: {} snapshots, {} gaps",
                analysis.snapshots.len(),
                analysis.gaps.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
#[path = "beliefs_tests.rs"]
mod tests;
