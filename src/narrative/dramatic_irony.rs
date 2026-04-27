//! Dramatic irony mapping and focalization tracking (Sprint D9.3).
//!
//! Tracks the knowledge gap between reader and characters at every point
//! in the narrative. Formalizes focalization (whose perspective) and its
//! relationship to information revelation.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::InferenceEngine;
use crate::types::InferenceResult;
use crate::types::{InferenceJobType, JobStatus, Situation};

// ─── Types ──────────────────────────────────────────────────

/// A narrative fact that can be known or unknown.
pub type FactId = String;

/// Knowledge state for an entity at a point in the narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeState {
    pub entity_id: Uuid,
    pub situation_id: Uuid,
    /// Facts this entity knows to be true.
    pub knows: HashSet<FactId>,
    /// Facts this entity believes true (may or may not be correct).
    pub believes_true: HashSet<FactId>,
    /// Facts this entity believes false (wrong beliefs).
    pub believes_false: HashSet<FactId>,
}

/// Reader knowledge state at a point in the sjužet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaderKnowledgeState {
    pub discourse_position: usize,
    /// Everything the reader knows at this point in the telling.
    pub knows: HashSet<FactId>,
}

/// Type of dramatic irony.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IronyType {
    /// Reader knows danger the character doesn't — "don't go in there!"
    Suspense,
    /// Reader knows good outcome character doesn't.
    Anticipation,
    /// Reader knows tragic outcome is inevitable.
    TragedyForeknowledge,
    /// Reader sees the mix-up characters can't.
    ComedicMisunderstanding,
}

/// A moment of dramatic irony.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DramaticIronyEvent {
    pub situation_id: Uuid,
    pub character_id: Uuid,
    /// Facts the reader knows at this point.
    pub reader_knows: HashSet<FactId>,
    /// Facts the reader knows but the character doesn't.
    pub character_doesnt_know: HashSet<FactId>,
    /// Irony intensity: count of unknown facts weighted by consequence.
    pub irony_intensity: f64,
    pub irony_type: IronyType,
}

/// Full dramatic irony map across a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DramaticIronyMap {
    pub narrative_id: String,
    pub events: Vec<DramaticIronyEvent>,
    /// Fraction of situations with active dramatic irony.
    pub irony_density: f64,
    /// Time series of irony intensity across the narrative.
    pub irony_curve: Vec<f64>,
}

/// Focalization type (Genette).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Focalization {
    /// Omniscient — narrator knows more than any character.
    Zero,
    /// Narrator knows what one character knows.
    Internal { focalizer: Uuid },
    /// Narrator knows less than characters — behaviorist observation.
    External,
}

/// A contiguous segment of consistent focalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalizationSegment {
    /// Situation range for this segment.
    pub situation_ids: Vec<Uuid>,
    pub start_chapter: usize,
    pub end_chapter: usize,
    pub focalization: Focalization,
}

/// Focalization analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalizationAnalysis {
    pub narrative_id: String,
    pub segments: Vec<FocalizationSegment>,
    /// Number of unique focalizers.
    pub unique_focalizers: usize,
    /// Average segments per focalizer switch.
    pub switch_rate: f64,
    /// Focalization type distribution.
    pub zero_fraction: f64,
    pub internal_fraction: f64,
    pub external_fraction: f64,
}

/// Moments where focalization shifts cause dramatic irony.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocalizationIronyInteraction {
    /// The segment boundary where the shift happens.
    pub shift_at_chapter: usize,
    /// Who we switch from.
    pub from_focalizer: Option<Uuid>,
    /// Who we switch to.
    pub to_focalizer: Option<Uuid>,
    /// Facts revealed by the new perspective that the previous focalizer doesn't know.
    pub revealed_facts: HashSet<FactId>,
    /// Resulting irony intensity.
    pub irony_intensity: f64,
}

// ─── KV ─────────────────────────────────────────────────────

fn irony_key(narrative_id: &str) -> Vec<u8> {
    format!("di/i/{}", narrative_id).into_bytes()
}

fn focalization_key(narrative_id: &str) -> Vec<u8> {
    format!("di/f/{}", narrative_id).into_bytes()
}

pub fn store_irony_map(hg: &Hypergraph, map: &DramaticIronyMap) -> Result<()> {
    let key = irony_key(&map.narrative_id);
    let val = serde_json::to_vec(map)?;
    hg.store().put(&key, &val)
}

pub fn load_irony_map(hg: &Hypergraph, narrative_id: &str) -> Result<Option<DramaticIronyMap>> {
    let key = irony_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn store_focalization(hg: &Hypergraph, analysis: &FocalizationAnalysis) -> Result<()> {
    let key = focalization_key(&analysis.narrative_id);
    let val = serde_json::to_vec(analysis)?;
    hg.store().put(&key, &val)
}

pub fn load_focalization(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<FocalizationAnalysis>> {
    let key = focalization_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

// ─── Analysis ───────────────────────────────────────────────

/// Extract facts from a situation's info_sets and content.
fn extract_facts_from_situation(hg: &Hypergraph, sit: &Situation) -> HashSet<FactId> {
    let mut facts = HashSet::new();

    // Facts from raw content
    if let Some(name) = &sit.name {
        facts.insert(format!("event:{}", name));
    }

    // Facts from causal links
    for cause in &sit.causes {
        facts.insert(format!("cause:{}→{}", cause.from_situation, sit.id));
    }

    // Facts from participations
    if let Ok(participants) = hg.get_participants_for_situation(&sit.id) {
        for p in &participants {
            if let Some(action) = &p.action {
                facts.insert(format!("action:{}:{}", p.entity_id, action));
            }
            if let Some(info) = &p.info_set {
                for learn in &info.learns {
                    facts.insert(format!("learns:{}:{}", p.entity_id, learn.fact));
                }
                for reveal in &info.reveals {
                    facts.insert(format!("reveals:{}:{}", p.entity_id, reveal.fact));
                }
            }
        }
    }

    facts
}

/// Compute the dramatic irony map for a narrative.
pub fn compute_dramatic_irony_map(hg: &Hypergraph, narrative_id: &str) -> Result<DramaticIronyMap> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(DramaticIronyMap {
            narrative_id: narrative_id.to_string(),
            events: Vec::new(),
            irony_density: 0.0,
            irony_curve: Vec::new(),
        });
    }

    // Sort by discourse order (created_at)
    let mut discourse_order: Vec<&Situation> = situations.iter().collect();
    discourse_order.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    let mut reader_knowledge = HashSet::new();
    let mut character_knowledge: HashMap<Uuid, HashSet<FactId>> = HashMap::new();
    let mut events = Vec::new();
    let mut irony_curve = Vec::new();

    for (_disc_pos, sit) in discourse_order.iter().enumerate() {
        // Reader learns all facts revealed in this situation
        let sit_facts = extract_facts_from_situation(hg, sit);
        reader_knowledge.extend(sit_facts.clone());

        // Update character knowledge for participants
        if let Ok(participants) = hg.get_participants_for_situation(&sit.id) {
            for p in &participants {
                let char_know = character_knowledge.entry(p.entity_id).or_default();

                // Character learns facts from participation
                if let Some(name) = &sit.name {
                    char_know.insert(format!("event:{}", name));
                }
                if let Some(info) = &p.info_set {
                    for l in &info.learns {
                        char_know.insert(l.fact.clone());
                    }
                }

                // Check for dramatic irony: reader knows things character doesn't
                let gap: HashSet<FactId> =
                    reader_knowledge.difference(char_know).cloned().collect();

                if !gap.is_empty() {
                    let intensity = gap.len() as f64;
                    let irony_type = classify_irony_type(&gap);

                    events.push(DramaticIronyEvent {
                        situation_id: sit.id,
                        character_id: p.entity_id,
                        reader_knows: reader_knowledge.clone(),
                        character_doesnt_know: gap,
                        irony_intensity: intensity,
                        irony_type,
                    });
                }
            }
        }

        // Record max irony at this discourse position
        let max_irony = events
            .iter()
            .filter(|e| e.situation_id == sit.id)
            .map(|e| e.irony_intensity)
            .fold(0.0f64, f64::max);
        irony_curve.push(max_irony);
    }

    let situations_with_irony = irony_curve.iter().filter(|&&v| v > 0.0).count();
    let irony_density = if discourse_order.is_empty() {
        0.0
    } else {
        situations_with_irony as f64 / discourse_order.len() as f64
    };

    let map = DramaticIronyMap {
        narrative_id: narrative_id.to_string(),
        events,
        irony_density,
        irony_curve,
    };

    store_irony_map(hg, &map)?;
    Ok(map)
}

/// Classify irony type from the knowledge gap facts.
fn classify_irony_type(gap: &HashSet<FactId>) -> IronyType {
    let gap_text: String = gap.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(" ");
    let lower = gap_text.to_lowercase();

    if lower.contains("danger")
        || lower.contains("kill")
        || lower.contains("trap")
        || lower.contains("attack")
        || lower.contains("death")
    {
        IronyType::Suspense
    } else if lower.contains("love")
        || lower.contains("rescue")
        || lower.contains("gift")
        || lower.contains("reward")
        || lower.contains("save")
    {
        IronyType::Anticipation
    } else if lower.contains("doom")
        || lower.contains("fate")
        || lower.contains("inevitable")
        || lower.contains("tragic")
    {
        IronyType::TragedyForeknowledge
    } else {
        // Default to suspense for unknown patterns
        IronyType::Suspense
    }
}

/// Detect focalization segments across a narrative.
pub fn detect_focalization(hg: &Hypergraph, narrative_id: &str) -> Result<FocalizationAnalysis> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(FocalizationAnalysis {
            narrative_id: narrative_id.to_string(),
            segments: Vec::new(),
            unique_focalizers: 0,
            switch_rate: 0.0,
            zero_fraction: 0.0,
            internal_fraction: 0.0,
            external_fraction: 0.0,
        });
    }

    // Sort by discourse order
    let mut sorted: Vec<&Situation> = situations.iter().collect();
    sorted.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    let mut segments: Vec<FocalizationSegment> = Vec::new();
    let mut current_focalizer: Option<Uuid> = None;
    let mut current_situations: Vec<Uuid> = Vec::new();
    let mut current_start = 0;

    for (chapter, sit) in sorted.iter().enumerate() {
        // Determine focalizer for this situation
        let focalizer = detect_situation_focalizer(hg, sit)?;

        let new_focus = match (&current_focalizer, &focalizer) {
            (Some(curr), Focalization::Internal { focalizer: f }) => curr != f,
            (Some(_), Focalization::Zero) => true,
            (Some(_), Focalization::External) => true,
            (None, _) => false, // First situation
        };

        if new_focus && !current_situations.is_empty() {
            segments.push(FocalizationSegment {
                situation_ids: current_situations.clone(),
                start_chapter: current_start,
                end_chapter: chapter - 1,
                focalization: match current_focalizer {
                    Some(id) => Focalization::Internal { focalizer: id },
                    None => Focalization::Zero,
                },
            });
            current_situations.clear();
            current_start = chapter;
        }

        current_focalizer = match &focalizer {
            Focalization::Internal { focalizer: f } => Some(*f),
            _ => None,
        };
        current_situations.push(sit.id);
    }

    // Final segment
    if !current_situations.is_empty() {
        segments.push(FocalizationSegment {
            situation_ids: current_situations,
            start_chapter: current_start,
            end_chapter: sorted.len() - 1,
            focalization: match current_focalizer {
                Some(id) => Focalization::Internal { focalizer: id },
                None => Focalization::Zero,
            },
        });
    }

    // Compute statistics
    let unique_focalizers: HashSet<_> = segments
        .iter()
        .filter_map(|s| match &s.focalization {
            Focalization::Internal { focalizer } => Some(*focalizer),
            _ => None,
        })
        .collect();

    let total = segments.len() as f64;
    let zero_count = segments
        .iter()
        .filter(|s| matches!(s.focalization, Focalization::Zero))
        .count();
    let internal_count = segments
        .iter()
        .filter(|s| matches!(s.focalization, Focalization::Internal { .. }))
        .count();
    let external_count = segments
        .iter()
        .filter(|s| matches!(s.focalization, Focalization::External))
        .count();

    let switch_rate = if sorted.len() > 1 {
        (segments.len() as f64 - 1.0) / sorted.len() as f64
    } else {
        0.0
    };

    let analysis = FocalizationAnalysis {
        narrative_id: narrative_id.to_string(),
        segments,
        unique_focalizers: unique_focalizers.len(),
        switch_rate,
        zero_fraction: if total > 0.0 {
            zero_count as f64 / total
        } else {
            0.0
        },
        internal_fraction: if total > 0.0 {
            internal_count as f64 / total
        } else {
            0.0
        },
        external_fraction: if total > 0.0 {
            external_count as f64 / total
        } else {
            0.0
        },
    };

    store_focalization(hg, &analysis)?;
    Ok(analysis)
}

/// Detect the focalizer for a single situation.
///
/// Heuristic: the character whose internal thoughts/perceptions are described is
/// the focalizer. We approximate this by looking at info_sets — the character
/// who "learns" or has "knows_before" entries is likely the POV character.
fn detect_situation_focalizer(hg: &Hypergraph, sit: &Situation) -> Result<Focalization> {
    let participants = hg.get_participants_for_situation(&sit.id)?;

    if participants.is_empty() {
        return Ok(Focalization::External);
    }

    // Score each participant as potential focalizer
    let mut scores: Vec<(Uuid, f64)> = Vec::new();

    for p in &participants {
        let mut score = 0.0;

        // Characters with info_sets (internal knowledge) are likely focalizers
        if let Some(info) = &p.info_set {
            score += info.knows_before.len() as f64 * 2.0;
            score += info.learns.len() as f64 * 3.0; // Learning = internal process
            score += info.reveals.len() as f64 * 1.0;
        }

        // Protagonists are more likely focalizers
        if matches!(p.role, crate::types::Role::Protagonist) {
            score += 5.0;
        }

        scores.push((p.entity_id, score));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((top_id, top_score)) = scores.first() {
        if *top_score > 0.0 {
            // Check if multiple participants have similar scores → omniscient
            let second_score = scores.get(1).map(|(_, s)| *s).unwrap_or(0.0);
            if second_score > 0.0 && (top_score - second_score) / top_score < 0.3 {
                Ok(Focalization::Zero)
            } else {
                Ok(Focalization::Internal { focalizer: *top_id })
            }
        } else {
            Ok(Focalization::External)
        }
    } else {
        Ok(Focalization::External)
    }
}

/// Find moments where focalization shifts create dramatic irony.
pub fn compute_focalization_irony_interaction(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<FocalizationIronyInteraction>> {
    let focalization = detect_focalization(hg, narrative_id)?;
    let irony_map = compute_dramatic_irony_map(hg, narrative_id)?;

    let mut interactions = Vec::new();

    for i in 1..focalization.segments.len() {
        let prev = &focalization.segments[i - 1];
        let curr = &focalization.segments[i];

        let from_focalizer = match &prev.focalization {
            Focalization::Internal { focalizer } => Some(*focalizer),
            _ => None,
        };
        let to_focalizer = match &curr.focalization {
            Focalization::Internal { focalizer } => Some(*focalizer),
            _ => None,
        };

        // Find irony events at the boundary
        let boundary_irony: Vec<_> = irony_map
            .events
            .iter()
            .filter(|e| curr.situation_ids.contains(&e.situation_id))
            .collect();

        if !boundary_irony.is_empty() {
            let revealed_facts: HashSet<_> = boundary_irony
                .iter()
                .flat_map(|e| e.character_doesnt_know.iter().cloned())
                .collect();
            let max_intensity = boundary_irony
                .iter()
                .map(|e| e.irony_intensity)
                .fold(0.0f64, f64::max);

            if !revealed_facts.is_empty() {
                interactions.push(FocalizationIronyInteraction {
                    shift_at_chapter: curr.start_chapter,
                    from_focalizer,
                    to_focalizer,
                    revealed_facts,
                    irony_intensity: max_intensity,
                });
            }
        }
    }

    Ok(interactions)
}

// ─── Inference Engines ──────────────────────────────────────

pub struct DramaticIronyEngine;

impl InferenceEngine for DramaticIronyEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::DramaticIrony
    }

    fn estimate_cost(
        &self,
        _job: &crate::inference::types::InferenceJob,
        _hg: &Hypergraph,
    ) -> Result<u64> {
        Ok(5000)
    }

    fn execute(
        &self,
        job: &crate::inference::types::InferenceJob,
        hg: &Hypergraph,
    ) -> Result<InferenceResult> {
        let narrative_id = crate::analysis::extract_narrative_id(job)?;

        let irony_map = compute_dramatic_irony_map(hg, narrative_id)?;
        let focalization = detect_focalization(hg, narrative_id)?;
        let interactions = compute_focalization_irony_interaction(hg, narrative_id)?;

        let result = serde_json::json!({
            "irony_map": irony_map,
            "focalization": focalization,
            "focalization_irony_interactions": interactions,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::DramaticIrony,
            target_id: job.target_id,
            result,
            confidence: 0.7,
            explanation: Some(format!(
                "Irony density: {:.0}%, {} focalizers, {} shifts causing irony",
                irony_map.irony_density * 100.0,
                focalization.unique_focalizers,
                interactions.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

pub struct FocalizationEngine;

impl InferenceEngine for FocalizationEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::Focalization
    }

    fn estimate_cost(
        &self,
        _job: &crate::inference::types::InferenceJob,
        _hg: &Hypergraph,
    ) -> Result<u64> {
        Ok(3000)
    }

    fn execute(
        &self,
        job: &crate::inference::types::InferenceJob,
        hg: &Hypergraph,
    ) -> Result<InferenceResult> {
        let narrative_id = crate::analysis::extract_narrative_id(job)?;

        let analysis = detect_focalization(hg, narrative_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::Focalization,
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence: 0.7,
            explanation: Some(format!(
                "{} focalization segments, {} unique focalizers, switch rate {:.2}",
                analysis.segments.len(),
                analysis.unique_focalizers,
                analysis.switch_rate
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::DateTime;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn setup_narrative(hg: &Hypergraph) -> (String, Vec<Uuid>, Vec<Uuid>) {
        let nid = "irony-test";

        // Create characters
        let alice = hg
            .create_entity(Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "Alice"}),
                beliefs: None,
                embedding: None,
                narrative_id: Some(nid.into()),
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();

        let bob = hg
            .create_entity(Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "Bob"}),
                beliefs: None,
                embedding: None,
                narrative_id: Some(nid.into()),
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();

        // Create situations
        let mut sit_ids = Vec::new();
        for i in 0..5 {
            let start = DateTime::from_timestamp(1700000000 + i * 3600, 0).unwrap();
            let sid = hg
                .create_situation(Situation {
                    id: Uuid::now_v7(),
                    properties: serde_json::Value::Null,
                    name: Some(format!("Scene {}", i)),
                    description: None,
                    temporal: crate::types::AllenInterval {
                        start: Some(start),
                        end: Some(start + chrono::Duration::hours(1)),
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
                    raw_content: vec![ContentBlock::text(&format!("Content of scene {}", i))],
                    narrative_level: NarrativeLevel::Scene,
                    narrative_id: Some(nid.into()),
                    discourse: None,
                    maturity: MaturityLevel::Candidate,
                    confidence: 0.8,
                    confidence_breakdown: None,
                    extraction_method: ExtractionMethod::LlmParsed,
                    provenance: vec![],
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
                })
                .unwrap();
            sit_ids.push(sid);
        }

        // Alice is in scenes 0, 1, 2 with rich info_set
        for &idx in &[0, 1, 2] {
            hg.add_participant(Participation {
                entity_id: alice,
                situation_id: sit_ids[idx],
                role: Role::Protagonist,
                info_set: Some(InfoSet {
                    knows_before: vec![KnowledgeFact {
                        about_entity: alice,
                        fact: format!("alice_knows_{}", idx),
                        confidence: 1.0,
                    }],
                    learns: vec![KnowledgeFact {
                        about_entity: alice,
                        fact: format!("alice_learns_{}", idx),
                        confidence: 1.0,
                    }],
                    reveals: vec![],
                    beliefs_about_others: vec![],
                }),
                action: Some(format!("alice_acts_{}", idx)),
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        // Bob is in scenes 2, 3, 4
        for &idx in &[2, 3, 4] {
            hg.add_participant(Participation {
                entity_id: bob,
                situation_id: sit_ids[idx],
                role: Role::Antagonist,
                info_set: Some(InfoSet {
                    knows_before: vec![],
                    learns: vec![KnowledgeFact {
                        about_entity: bob,
                        fact: format!("bob_learns_{}", idx),
                        confidence: 1.0,
                    }],
                    reveals: vec![],
                    beliefs_about_others: vec![],
                }),
                action: Some(format!("bob_acts_{}", idx)),
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        (nid.to_string(), vec![alice, bob], sit_ids)
    }

    #[test]
    fn test_dramatic_irony_map() {
        let hg = test_hg();
        let (nid, chars, _sits) = setup_narrative(&hg);

        let map = compute_dramatic_irony_map(&hg, &nid).unwrap();
        // There should be some irony events (reader knows things characters don't)
        assert!(!map.irony_curve.is_empty());
        assert_eq!(map.irony_curve.len(), 5);
    }

    #[test]
    fn test_focalization_detection() {
        let hg = test_hg();
        let (nid, chars, _sits) = setup_narrative(&hg);

        let analysis = detect_focalization(&hg, &nid).unwrap();
        assert!(!analysis.segments.is_empty());
        // Alice has protagonist role + rich info_set → should be primary focalizer
        assert!(analysis.unique_focalizers >= 1);
    }

    #[test]
    fn test_irony_kv_persistence() {
        let hg = test_hg();
        let (nid, _, _) = setup_narrative(&hg);

        let map = compute_dramatic_irony_map(&hg, &nid).unwrap();
        let loaded = load_irony_map(&hg, &nid).unwrap().unwrap();
        assert_eq!(map.events.len(), loaded.events.len());
        assert!((map.irony_density - loaded.irony_density).abs() < 0.001);
    }

    #[test]
    fn test_classify_irony_type() {
        let mut gap = HashSet::new();
        gap.insert("action:villain:sets trap".to_string());
        assert_eq!(classify_irony_type(&gap), IronyType::Suspense);

        let mut gap2 = HashSet::new();
        gap2.insert("action:hero:secret love".to_string());
        assert_eq!(classify_irony_type(&gap2), IronyType::Anticipation);
    }

    #[test]
    fn test_empty_narrative() {
        let hg = test_hg();
        let map = compute_dramatic_irony_map(&hg, "nonexistent").unwrap();
        assert!(map.events.is_empty());
        assert_eq!(map.irony_density, 0.0);

        let foc = detect_focalization(&hg, "nonexistent").unwrap();
        assert!(foc.segments.is_empty());
    }
}
