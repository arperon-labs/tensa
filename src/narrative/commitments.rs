//! Narrative commitment tracking (Sprint D9.1).
//!
//! Formalizes Chekhov's Gun as a computable system. Tracks narrative promises,
//! their status (pending/fulfilled/abandoned/subverted), and the promise/progress/payoff
//! rhythm across the full narrative.
//!
//! KV persistence at `nc/` prefix.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

// ─── KV prefix ──────────────────────────────────────────────
/// Narrative commitment prefix: `nc/{narrative_id}/{commitment_id}`
pub const COMMITMENT_PREFIX: &str = "nc/";

// ─── Types ──────────────────────────────────────────────────

/// The type of narrative commitment (promise to the reader).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentType {
    /// Concrete object/skill/trait introduced with emphasis.
    ChekhovsGun,
    /// Atmospheric hint at future event.
    Foreshadowing,
    /// Deliberate misdirection — setup with intentional non-payoff.
    RedHerring,
    /// Will they succeed? Will they find love?
    DramaticQuestion,
    /// Character trait/flaw that must be tested.
    CharacterPromise,
    /// Thematic idea planted for later exploration.
    ThematicSeed,
    /// Question raised that demands answer.
    MysterySetup,
}

/// Lifecycle status of a narrative commitment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentStatus {
    /// Setup introduced but not yet progressed.
    Planted,
    /// Partial progress toward payoff.
    InProgress,
    /// Commitment fulfilled — payoff delivered.
    Fulfilled,
    /// Setup with no payoff — a flaw unless intentional.
    Abandoned,
    /// Payoff contradicts expectation — twist.
    Subverted,
    /// Red herring revealed as misdirection.
    RedHerringResolved,
}

/// A single narrative commitment (setup-payoff pair).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeCommitment {
    pub id: Uuid,
    pub narrative_id: String,
    pub commitment_type: CommitmentType,
    /// The situation where this commitment was introduced.
    pub setup_event: Uuid,
    /// Chapter index of the setup.
    pub setup_chapter: usize,
    /// How prominently the element was introduced (narrative position, repetition, detail density).
    pub setup_salience: f64,
    /// Current lifecycle status.
    pub status: CommitmentStatus,
    /// The situation where this commitment was fulfilled/resolved.
    pub payoff_event: Option<Uuid>,
    /// Chapter index of the payoff.
    pub payoff_chapter: Option<usize>,
    /// Chapters between setup and payoff.
    pub payoff_distance: Option<usize>,
    /// Intermediate progress events (causal chain).
    pub causal_chain: Vec<Uuid>,
    /// Description of what was promised.
    pub description: String,
    /// The element being tracked (entity name, object, concept).
    pub tracked_element: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Per-chapter promise rhythm data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterRhythm {
    pub chapter: usize,
    /// Number of promises still outstanding at end of chapter.
    pub promises_outstanding: usize,
    /// Promises fulfilled in this chapter.
    pub promises_fulfilled: usize,
    /// New promises planted in this chapter.
    pub new_promises: usize,
    /// Net tension: outstanding - fulfilled.
    pub net_tension: i64,
}

/// Promise rhythm across the full narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromiseRhythm {
    pub narrative_id: String,
    pub chapters: Vec<ChapterRhythm>,
    /// Ratio of fulfilled to (fulfilled + abandoned). Higher = tighter craft.
    pub fulfillment_ratio: f64,
    /// Mean chapters between setup and payoff.
    pub average_payoff_distance: f64,
    /// Total commitments detected.
    pub total_commitments: usize,
    /// Unfired guns (Planted status commitments past the expected payoff window).
    pub unfired_guns: Vec<Uuid>,
}

// ─── KV Operations ──────────────────────────────────────────

fn commitment_key(narrative_id: &str, commitment_id: &Uuid) -> Vec<u8> {
    format!("nc/{}/{}", narrative_id, commitment_id).into_bytes()
}

fn rhythm_key(narrative_id: &str) -> Vec<u8> {
    format!("nc/{}/rhythm", narrative_id).into_bytes()
}

/// Store a commitment.
pub fn store_commitment(hg: &Hypergraph, c: &NarrativeCommitment) -> Result<()> {
    let key = commitment_key(&c.narrative_id, &c.id);
    let val = serde_json::to_vec(c)?;
    hg.store().put(&key, &val)
}

/// Load a commitment.
pub fn load_commitment(
    hg: &Hypergraph,
    narrative_id: &str,
    id: &Uuid,
) -> Result<Option<NarrativeCommitment>> {
    let key = commitment_key(narrative_id, id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

/// List all commitments for a narrative.
pub fn list_commitments(hg: &Hypergraph, narrative_id: &str) -> Result<Vec<NarrativeCommitment>> {
    let prefix = format!("nc/{}/", narrative_id).into_bytes();
    let items = hg.store().prefix_scan(&prefix)?;
    let mut out = Vec::new();
    for (k, v) in items {
        let key_str = String::from_utf8_lossy(&k);
        // Skip the rhythm key
        if key_str.ends_with("/rhythm") {
            continue;
        }
        match serde_json::from_slice::<NarrativeCommitment>(&v) {
            Ok(c) => out.push(c),
            Err(_) => continue,
        }
    }
    out.sort_by_key(|c| c.setup_chapter);
    Ok(out)
}

/// Update a commitment's status.
pub fn update_commitment(
    hg: &Hypergraph,
    narrative_id: &str,
    id: &Uuid,
    status: CommitmentStatus,
    payoff_event: Option<Uuid>,
    payoff_chapter: Option<usize>,
) -> Result<()> {
    let mut c = load_commitment(hg, narrative_id, id)?
        .ok_or_else(|| TensaError::NotFound(format!("commitment {}", id)))?;
    c.status = status;
    c.payoff_event = payoff_event;
    c.payoff_chapter = payoff_chapter;
    if let Some(payoff) = c.payoff_chapter {
        c.payoff_distance = Some(payoff.saturating_sub(c.setup_chapter));
    }
    c.updated_at = Utc::now();
    store_commitment(hg, &c)
}

// ─── Detection ──────────────────────────────────────────────

/// Detect commitments in a narrative using heuristic analysis.
///
/// Heuristic: entities mentioned in high detail (long raw_content or detailed
/// properties) that are then not referenced for N+ chapters are potential
/// unfired guns. Situations with causal gaps are potential mystery setups.
pub fn detect_commitments(hg: &Hypergraph, narrative_id: &str) -> Result<Vec<NarrativeCommitment>> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(Vec::new());
    }

    // Sort situations by temporal start for chapter assignment
    let mut sorted_sits: Vec<_> = situations.iter().collect();
    sorted_sits.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    // Assign chapter indices based on temporal ordering
    let chapter_map: std::collections::HashMap<Uuid, usize> = sorted_sits
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i))
        .collect();

    // Track entity appearances per chapter
    let entities = hg.list_entities_by_narrative(narrative_id)?;
    let mut entity_chapters: std::collections::HashMap<Uuid, Vec<usize>> =
        std::collections::HashMap::new();
    let mut entity_detail_scores: std::collections::HashMap<Uuid, f64> =
        std::collections::HashMap::new();

    for entity in &entities {
        // Compute detail score from properties
        let props_str = entity.properties.to_string();
        let detail_score = props_str.len() as f64 / 100.0;
        entity_detail_scores.insert(entity.id, detail_score.min(5.0));

        // Find all situations this entity participates in
        let participations = hg.get_situations_for_entity(&entity.id)?;
        let mut chapters: Vec<usize> = participations
            .iter()
            .filter_map(|p| chapter_map.get(&p.situation_id).copied())
            .collect();
        chapters.sort();
        chapters.dedup();
        entity_chapters.insert(entity.id, chapters);
    }

    let total_chapters = sorted_sits.len();
    let gap_threshold = (total_chapters / 4).max(3); // At least 3 chapters gap

    let mut commitments = Vec::new();

    // Detect Chekhov's Guns: entities introduced with detail then absent
    for entity in &entities {
        let detail = entity_detail_scores.get(&entity.id).copied().unwrap_or(0.0);
        let chapters = entity_chapters.get(&entity.id).cloned().unwrap_or_default();

        if detail > 1.0 && !chapters.is_empty() {
            let first_chapter = chapters[0];
            let last_chapter = chapters.last().copied().unwrap_or(0);

            // Check for gaps in appearances
            for i in 0..chapters.len().saturating_sub(1) {
                let gap = chapters[i + 1] - chapters[i];
                if gap >= gap_threshold {
                    let name = entity
                        .properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let setup_sit = sorted_sits
                        .get(chapters[i])
                        .map(|s| s.id)
                        .unwrap_or(Uuid::nil());

                    let commitment = NarrativeCommitment {
                        id: Uuid::now_v7(),
                        narrative_id: narrative_id.to_string(),
                        commitment_type: CommitmentType::ChekhovsGun,
                        setup_event: setup_sit,
                        setup_chapter: chapters[i],
                        setup_salience: detail,
                        status: if chapters.get(i + 1).is_some() {
                            CommitmentStatus::Fulfilled
                        } else {
                            CommitmentStatus::Planted
                        },
                        payoff_event: sorted_sits.get(chapters[i + 1]).map(|s| s.id),
                        payoff_chapter: Some(chapters[i + 1]),
                        payoff_distance: Some(gap),
                        causal_chain: Vec::new(),
                        description: format!(
                            "Entity '{}' introduced with detail score {:.1}, gap of {} chapters",
                            name, detail, gap
                        ),
                        tracked_element: name,
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    };
                    commitments.push(commitment);
                }
            }

            // Check if entity disappears entirely after introduction
            if last_chapter < total_chapters.saturating_sub(gap_threshold) && detail > 2.0 {
                let name = entity
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let setup_sit = sorted_sits
                    .get(first_chapter)
                    .map(|s| s.id)
                    .unwrap_or(Uuid::nil());

                let commitment = NarrativeCommitment {
                    id: Uuid::now_v7(),
                    narrative_id: narrative_id.to_string(),
                    commitment_type: CommitmentType::ChekhovsGun,
                    setup_event: setup_sit,
                    setup_chapter: first_chapter,
                    setup_salience: detail,
                    status: CommitmentStatus::Abandoned,
                    payoff_event: None,
                    payoff_chapter: None,
                    payoff_distance: None,
                    causal_chain: Vec::new(),
                    description: format!("Entity '{}' introduced with detail score {:.1} then abandoned after chapter {}", name, detail, last_chapter),
                    tracked_element: name,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                commitments.push(commitment);
            }
        }
    }

    // Detect Dramatic Questions from causal gaps
    for sit in &sorted_sits {
        if !sit.causes.is_empty() {
            // Situations with explicit causal links that have low confidence
            for cause in &sit.causes {
                if cause.strength < 0.5 {
                    let chapter = chapter_map.get(&sit.id).copied().unwrap_or(0);
                    let commitment = NarrativeCommitment {
                        id: Uuid::now_v7(),
                        narrative_id: narrative_id.to_string(),
                        commitment_type: CommitmentType::MysterySetup,
                        setup_event: sit.id,
                        setup_chapter: chapter,
                        setup_salience: 1.0 - cause.strength as f64,
                        status: CommitmentStatus::Planted,
                        payoff_event: None,
                        payoff_chapter: None,
                        payoff_distance: None,
                        causal_chain: Vec::new(),
                        description: format!(
                            "Weak causal link (strength {:.2}) suggests unresolved mystery",
                            cause.strength
                        ),
                        tracked_element: sit.name.clone().unwrap_or_default(),
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    };
                    commitments.push(commitment);
                }
            }
        }
    }

    // Persist all detected commitments
    for c in &commitments {
        store_commitment(hg, c)?;
    }

    Ok(commitments)
}

/// Compute promise rhythm across chapters.
pub fn compute_promise_rhythm(hg: &Hypergraph, narrative_id: &str) -> Result<PromiseRhythm> {
    let commitments = list_commitments(hg, narrative_id)?;
    if commitments.is_empty() {
        return Ok(PromiseRhythm {
            narrative_id: narrative_id.to_string(),
            chapters: Vec::new(),
            fulfillment_ratio: 0.0,
            average_payoff_distance: 0.0,
            total_commitments: 0,
            unfired_guns: Vec::new(),
        });
    }

    let max_chapter = commitments
        .iter()
        .map(|c| c.payoff_chapter.unwrap_or(c.setup_chapter))
        .max()
        .unwrap_or(0);

    let mut chapters = Vec::new();
    let mut outstanding = 0usize;

    for ch in 0..=max_chapter {
        let new_promises = commitments.iter().filter(|c| c.setup_chapter == ch).count();
        let fulfilled = commitments
            .iter()
            .filter(|c| {
                c.payoff_chapter == Some(ch)
                    && matches!(
                        c.status,
                        CommitmentStatus::Fulfilled
                            | CommitmentStatus::Subverted
                            | CommitmentStatus::RedHerringResolved
                    )
            })
            .count();

        outstanding = outstanding
            .saturating_add(new_promises)
            .saturating_sub(fulfilled);

        chapters.push(ChapterRhythm {
            chapter: ch,
            promises_outstanding: outstanding,
            promises_fulfilled: fulfilled,
            new_promises,
            net_tension: new_promises as i64 - fulfilled as i64,
        });
    }

    let fulfilled_count = commitments
        .iter()
        .filter(|c| {
            matches!(
                c.status,
                CommitmentStatus::Fulfilled
                    | CommitmentStatus::Subverted
                    | CommitmentStatus::RedHerringResolved
            )
        })
        .count();
    let abandoned_count = commitments
        .iter()
        .filter(|c| matches!(c.status, CommitmentStatus::Abandoned))
        .count();

    let fulfillment_ratio = if fulfilled_count + abandoned_count > 0 {
        fulfilled_count as f64 / (fulfilled_count + abandoned_count) as f64
    } else {
        1.0
    };

    let distances: Vec<f64> = commitments
        .iter()
        .filter_map(|c| c.payoff_distance.map(|d| d as f64))
        .collect();
    let average_payoff_distance = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f64>() / distances.len() as f64
    };

    let unfired_guns: Vec<Uuid> = commitments
        .iter()
        .filter(|c| matches!(c.status, CommitmentStatus::Planted))
        .map(|c| c.id)
        .collect();

    let rhythm = PromiseRhythm {
        narrative_id: narrative_id.to_string(),
        chapters,
        fulfillment_ratio,
        average_payoff_distance,
        total_commitments: commitments.len(),
        unfired_guns,
    };

    // Persist
    let key = rhythm_key(narrative_id);
    let val = serde_json::to_vec(&rhythm)?;
    hg.store().put(&key, &val)?;

    Ok(rhythm)
}

// ─── Inference Engine ───────────────────────────────────────

/// Inference engine for commitment detection.
pub struct CommitmentEngine;

impl InferenceEngine for CommitmentEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::CommitmentDetection
    }

    fn estimate_cost(
        &self,
        _job: &crate::inference::types::InferenceJob,
        _hg: &Hypergraph,
    ) -> Result<u64> {
        Ok(5000) // 5 seconds
    }

    fn execute(
        &self,
        job: &crate::inference::types::InferenceJob,
        hg: &Hypergraph,
    ) -> Result<InferenceResult> {
        let narrative_id = crate::analysis::extract_narrative_id(job)?;

        let commitments = detect_commitments(hg, narrative_id)?;
        let rhythm = compute_promise_rhythm(hg, narrative_id)?;

        let result = serde_json::json!({
            "commitments": commitments,
            "rhythm": rhythm,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::CommitmentDetection,
            target_id: job.target_id,
            result,
            confidence: 0.7,
            explanation: Some(format!(
                "Detected {} commitments with {:.0}% fulfillment ratio",
                commitments.len(),
                rhythm.fulfillment_ratio * 100.0
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
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_entity(hg: &Hypergraph, name: &str, narrative_id: &str) -> Uuid {
        let e = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name, "description": "A character with detailed backstory and motivations", "traits": ["brave", "curious"]}),
            beliefs: None,
            embedding: None,
            narrative_id: Some(narrative_id.to_string()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(e).unwrap()
    }

    fn make_situation(hg: &Hypergraph, narrative_id: &str, start_offset: i64) -> Uuid {
        let start = Utc::now() + chrono::Duration::hours(start_offset);
        let end = start + chrono::Duration::hours(1);
        let s = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(format!("Situation at hour {}", start_offset)),
            description: None,
            temporal: crate::types::AllenInterval {
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
            raw_content: vec![ContentBlock::text("A scene")],
            narrative_level: NarrativeLevel::Scene,
            narrative_id: Some(narrative_id.to_string()),
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
        };
        hg.create_situation(s).unwrap()
    }

    #[test]
    fn test_commitment_crud() {
        let hg = test_hg();
        let nid = "test-story";

        let c = NarrativeCommitment {
            id: Uuid::now_v7(),
            narrative_id: nid.to_string(),
            commitment_type: CommitmentType::ChekhovsGun,
            setup_event: Uuid::now_v7(),
            setup_chapter: 2,
            setup_salience: 3.5,
            status: CommitmentStatus::Planted,
            payoff_event: None,
            payoff_chapter: None,
            payoff_distance: None,
            causal_chain: Vec::new(),
            description: "The revolver on the mantelpiece".into(),
            tracked_element: "revolver".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        store_commitment(&hg, &c).unwrap();
        let loaded = load_commitment(&hg, nid, &c.id).unwrap().unwrap();
        assert_eq!(loaded.commitment_type, CommitmentType::ChekhovsGun);
        assert_eq!(loaded.status, CommitmentStatus::Planted);

        // Update status
        update_commitment(
            &hg,
            nid,
            &c.id,
            CommitmentStatus::Fulfilled,
            Some(Uuid::now_v7()),
            Some(8),
        )
        .unwrap();
        let updated = load_commitment(&hg, nid, &c.id).unwrap().unwrap();
        assert_eq!(updated.status, CommitmentStatus::Fulfilled);
        assert_eq!(updated.payoff_chapter, Some(8));
        assert_eq!(updated.payoff_distance, Some(6));
    }

    #[test]
    fn test_commitment_list() {
        let hg = test_hg();
        let nid = "list-test";

        for i in 0..5 {
            let c = NarrativeCommitment {
                id: Uuid::now_v7(),
                narrative_id: nid.to_string(),
                commitment_type: CommitmentType::Foreshadowing,
                setup_event: Uuid::now_v7(),
                setup_chapter: i,
                setup_salience: 1.0,
                status: if i < 3 {
                    CommitmentStatus::Fulfilled
                } else {
                    CommitmentStatus::Planted
                },
                payoff_event: None,
                payoff_chapter: if i < 3 { Some(i + 2) } else { None },
                payoff_distance: if i < 3 { Some(2) } else { None },
                causal_chain: Vec::new(),
                description: format!("Commitment {}", i),
                tracked_element: format!("element-{}", i),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            store_commitment(&hg, &c).unwrap();
        }

        let list = list_commitments(&hg, nid).unwrap();
        assert_eq!(list.len(), 5);
        // Should be sorted by setup_chapter
        for (i, c) in list.iter().enumerate() {
            assert_eq!(c.setup_chapter, i);
        }
    }

    #[test]
    fn test_promise_rhythm_computation() {
        let hg = test_hg();
        let nid = "rhythm-test";

        // Create 3 fulfilled + 2 planted commitments
        let types = [
            (0, Some(3), CommitmentStatus::Fulfilled),
            (1, Some(4), CommitmentStatus::Fulfilled),
            (2, Some(5), CommitmentStatus::Fulfilled),
            (3, None, CommitmentStatus::Planted),
            (4, None, CommitmentStatus::Abandoned),
        ];

        for (setup, payoff, status) in &types {
            let c = NarrativeCommitment {
                id: Uuid::now_v7(),
                narrative_id: nid.to_string(),
                commitment_type: CommitmentType::DramaticQuestion,
                setup_event: Uuid::now_v7(),
                setup_chapter: *setup,
                setup_salience: 1.0,
                status: status.clone(),
                payoff_event: payoff.map(|_| Uuid::now_v7()),
                payoff_chapter: *payoff,
                payoff_distance: payoff.map(|p| p - setup),
                causal_chain: Vec::new(),
                description: format!("Q at ch {}", setup),
                tracked_element: format!("q-{}", setup),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            store_commitment(&hg, &c).unwrap();
        }

        let rhythm = compute_promise_rhythm(&hg, nid).unwrap();
        assert_eq!(rhythm.total_commitments, 5);
        assert!((rhythm.fulfillment_ratio - 0.75).abs() < 0.01); // 3 / (3+1)
        assert!((rhythm.average_payoff_distance - 3.0).abs() < 0.01); // (3+3+3)/3
        assert_eq!(rhythm.unfired_guns.len(), 1); // Only 1 Planted
    }

    #[test]
    fn test_detect_commitments_basic() {
        let hg = test_hg();
        let nid = "detect-test";

        // Create entities and situations with gaps
        let e1 = make_entity(&hg, "Detective Holmes", nid);
        let mut sit_ids = Vec::new();
        for i in 0..10 {
            sit_ids.push(make_situation(&hg, nid, i * 24));
        }

        // Holmes appears in situations 0, 1, then gap, then 8, 9
        for &idx in &[0, 1, 8, 9] {
            hg.add_participant(crate::types::Participation {
                entity_id: e1,
                situation_id: sit_ids[idx],
                role: crate::types::Role::Protagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let commitments = detect_commitments(&hg, nid).unwrap();
        // Should detect the gap (chapters 1→8 = gap of 7, threshold = max(10/4, 3) = 3)
        assert!(!commitments.is_empty());
    }
}
