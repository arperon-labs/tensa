//! Fabula/Sjužet separation and telling-order optimization (Sprint D9.2).
//!
//! Formally separates "what happened" (fabula/story time) from "how it's told"
//! (sjužet/discourse time). Enables reordering the telling for maximum dramatic effect.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus, Situation};

// ─── Types ──────────────────────────────────────────────────

/// Duration category (Genette): how narrated time relates to discourse time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NarrationMode {
    /// Narrated time ≈ discourse time (dialogue, real-time action).
    Scene,
    /// Narrated time >> discourse time (covering days/weeks in a paragraph).
    Summary,
    /// Narrated time = 0, discourse continues (description, reflection).
    Pause,
    /// Discourse time = 0, narrated time passes (time skip).
    Ellipsis,
    /// Narrated time < discourse time (slow motion, microscopic detail).
    Stretch,
}

/// Temporal relationship to the previous segment in discourse order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalShift {
    /// Events proceed in chronological order.
    Chronological,
    /// Flashback: how far back (reach chapters) and how long (extent chapters).
    Analepsis { reach: usize, extent: usize },
    /// Flash-forward.
    Prolepsis { reach: usize, extent: usize },
    /// Opening mid-action.
    InMediasRes,
    /// Story within a story.
    FrameNarrative,
}

/// Chronological ordering of all situations in a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fabula {
    pub narrative_id: String,
    /// Situations sorted by temporal start (story-world time).
    pub situations: Vec<Uuid>,
    /// Temporal span of the fabula.
    pub earliest: Option<DateTime<Utc>>,
    pub latest: Option<DateTime<Utc>>,
}

/// A single segment in the discourse ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SjuzetSegment {
    pub situation_id: Uuid,
    /// Position in discourse order (0-indexed).
    pub discourse_position: usize,
    /// Corresponding position in fabula (chronological) order.
    pub fabula_position: usize,
    /// Duration category for this segment.
    pub narration_mode: NarrationMode,
    /// Temporal relationship to the preceding segment.
    pub temporal_shift: TemporalShift,
}

/// The discourse ordering — sequence in which situations are presented to the reader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sjuzet {
    pub narrative_id: String,
    pub segments: Vec<SjuzetSegment>,
}

/// A candidate reordering of the sjužet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SjuzetCandidate {
    pub label: String,
    pub segments: Vec<SjuzetSegment>,
    /// Estimated dramatic irony score improvement.
    pub irony_score: f64,
    /// Estimated commitment tension improvement.
    pub tension_score: f64,
    /// Normalized Kendall tau distance from fabula.
    pub divergence: f64,
}

/// Distribution of narration modes across a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrationModeDistribution {
    pub scene_fraction: f64,
    pub summary_fraction: f64,
    pub pause_fraction: f64,
    pub ellipsis_fraction: f64,
    pub stretch_fraction: f64,
}

// ─── KV ─────────────────────────────────────────────────────

fn fabula_key(narrative_id: &str) -> Vec<u8> {
    format!("fs/f/{}", narrative_id).into_bytes()
}

fn sjuzet_key(narrative_id: &str) -> Vec<u8> {
    format!("fs/s/{}", narrative_id).into_bytes()
}

pub fn store_fabula(hg: &Hypergraph, f: &Fabula) -> Result<()> {
    let key = fabula_key(&f.narrative_id);
    let val = serde_json::to_vec(f)?;
    hg.store().put(&key, &val)
}

pub fn load_fabula(hg: &Hypergraph, narrative_id: &str) -> Result<Option<Fabula>> {
    let key = fabula_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn store_sjuzet(hg: &Hypergraph, s: &Sjuzet) -> Result<()> {
    let key = sjuzet_key(&s.narrative_id);
    let val = serde_json::to_vec(s)?;
    hg.store().put(&key, &val)
}

pub fn load_sjuzet(hg: &Hypergraph, narrative_id: &str) -> Result<Option<Sjuzet>> {
    let key = sjuzet_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

// ─── Extraction ─────────────────────────────────────────────

/// Derive chronological order from temporal constraints.
pub fn extract_fabula(hg: &Hypergraph, narrative_id: &str) -> Result<Fabula> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(Fabula {
            narrative_id: narrative_id.to_string(),
            situations: Vec::new(),
            earliest: None,
            latest: None,
        });
    }

    // Sort by temporal start (chronological order = fabula)
    let mut sorted: Vec<&Situation> = situations.iter().collect();
    sorted.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    let earliest = sorted.first().and_then(|s| s.temporal.start);
    let latest = sorted
        .last()
        .and_then(|s| s.temporal.end.or(s.temporal.start));

    let fab = Fabula {
        narrative_id: narrative_id.to_string(),
        situations: sorted.iter().map(|s| s.id).collect(),
        earliest,
        latest,
    };

    store_fabula(hg, &fab)?;
    Ok(fab)
}

/// Derive discourse order from ingestion/chapter structure.
///
/// Uses `created_at` timestamps as proxy for discourse order (the order in
/// which situations were ingested = the order they appear in the source text).
pub fn extract_sjuzet(hg: &Hypergraph, narrative_id: &str) -> Result<Sjuzet> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(Sjuzet {
            narrative_id: narrative_id.to_string(),
            segments: Vec::new(),
        });
    }

    // Discourse order: by created_at (ingestion order = text order)
    let mut discourse_order: Vec<&Situation> = situations.iter().collect();
    discourse_order.sort_by(|a, b| a.created_at.cmp(&b.created_at));

    // Build fabula index for cross-referencing
    let mut chrono_order: Vec<&Situation> = situations.iter().collect();
    chrono_order.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));
    let fabula_index: std::collections::HashMap<Uuid, usize> = chrono_order
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i))
        .collect();

    let mut segments = Vec::new();
    let mut prev_fabula_pos: Option<usize> = None;

    for (discourse_pos, sit) in discourse_order.iter().enumerate() {
        let fabula_pos = fabula_index.get(&sit.id).copied().unwrap_or(discourse_pos);

        // Classify narration mode from content characteristics
        let narration_mode = classify_narration_mode(sit);

        // Classify temporal shift relative to previous
        let temporal_shift = if discourse_pos == 0 {
            // Check if opening in medias res (not starting at fabula position 0)
            if fabula_pos > 0 {
                TemporalShift::InMediasRes
            } else {
                TemporalShift::Chronological
            }
        } else if let Some(prev_fp) = prev_fabula_pos {
            if fabula_pos == prev_fp + 1 || fabula_pos == prev_fp {
                TemporalShift::Chronological
            } else if fabula_pos < prev_fp {
                TemporalShift::Analepsis {
                    reach: prev_fp - fabula_pos,
                    extent: 1,
                }
            } else {
                // Jump forward
                let gap = fabula_pos - prev_fp;
                if gap > 2 {
                    TemporalShift::Prolepsis {
                        reach: gap,
                        extent: 1,
                    }
                } else {
                    TemporalShift::Chronological
                }
            }
        } else {
            TemporalShift::Chronological
        };

        segments.push(SjuzetSegment {
            situation_id: sit.id,
            discourse_position: discourse_pos,
            fabula_position: fabula_pos,
            narration_mode,
            temporal_shift,
        });

        prev_fabula_pos = Some(fabula_pos);
    }

    let sjuzet = Sjuzet {
        narrative_id: narrative_id.to_string(),
        segments,
    };

    store_sjuzet(hg, &sjuzet)?;
    Ok(sjuzet)
}

/// Classify a situation's narration mode from content characteristics.
fn classify_narration_mode(sit: &Situation) -> NarrationMode {
    let text = sit
        .raw_content
        .iter()
        .map(|cb| cb.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let word_count = text.split_whitespace().count();

    // Check for dialogue markers
    let dialogue_markers = text.matches('"').count() + text.matches('\u{201c}').count();
    let has_dialogue = dialogue_markers >= 4;

    // Check temporal span
    let temporal_span_hours = match (sit.temporal.start, sit.temporal.end) {
        (Some(s), Some(e)) => (e - s).num_hours() as f64,
        _ => 1.0,
    };

    if word_count < 20 && temporal_span_hours > 24.0 {
        NarrationMode::Summary
    } else if has_dialogue && word_count > 50 {
        NarrationMode::Scene
    } else if temporal_span_hours < 0.01 && word_count > 100 {
        NarrationMode::Stretch
    } else if word_count > 80 && dialogue_markers < 2 {
        // Long descriptive passage with no dialogue
        NarrationMode::Pause
    } else {
        NarrationMode::Scene
    }
}

/// Compute fabula-sjužet divergence as normalized Kendall tau distance.
///
/// 0.0 = perfectly chronological telling.
/// 1.0 = maximum reordering.
pub fn compute_divergence(_fabula: &Fabula, sjuzet: &Sjuzet) -> f64 {
    if sjuzet.segments.len() < 2 {
        return 0.0;
    }

    let n = sjuzet.segments.len();
    let fabula_positions: Vec<usize> = sjuzet.segments.iter().map(|s| s.fabula_position).collect();

    // Count inversions (Kendall tau distance)
    let mut inversions = 0usize;
    let total_pairs = n * (n - 1) / 2;

    for i in 0..n {
        for j in (i + 1)..n {
            if fabula_positions[i] > fabula_positions[j] {
                inversions += 1;
            }
        }
    }

    if total_pairs == 0 {
        0.0
    } else {
        inversions as f64 / total_pairs as f64
    }
}

/// Compute narration mode distribution.
pub fn narration_mode_distribution(sjuzet: &Sjuzet) -> NarrationModeDistribution {
    if sjuzet.segments.is_empty() {
        return NarrationModeDistribution {
            scene_fraction: 0.0,
            summary_fraction: 0.0,
            pause_fraction: 0.0,
            ellipsis_fraction: 0.0,
            stretch_fraction: 0.0,
        };
    }

    let n = sjuzet.segments.len() as f64;
    let (mut scene, mut summary, mut pause, mut ellipsis, mut stretch) = (0usize, 0, 0, 0, 0);
    for seg in &sjuzet.segments {
        match seg.narration_mode {
            NarrationMode::Scene => scene += 1,
            NarrationMode::Summary => summary += 1,
            NarrationMode::Pause => pause += 1,
            NarrationMode::Ellipsis => ellipsis += 1,
            NarrationMode::Stretch => stretch += 1,
        }
    }

    NarrationModeDistribution {
        scene_fraction: scene as f64 / n,
        summary_fraction: summary as f64 / n,
        pause_fraction: pause as f64 / n,
        ellipsis_fraction: ellipsis as f64 / n,
        stretch_fraction: stretch as f64 / n,
    }
}

/// Suggest alternative telling orders that maximize dramatic effect.
pub fn suggest_reordering(
    fabula: &Fabula,
    sjuzet: &Sjuzet,
    _hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<SjuzetCandidate>> {
    if fabula.situations.len() < 3 {
        return Ok(Vec::new());
    }

    let mut candidates = Vec::new();

    // Candidate 1: In medias res — start at the highest-tension situation (midpoint)
    {
        let mid = fabula.situations.len() / 2;
        let mut reordered = sjuzet.segments.clone();
        if mid < reordered.len() {
            let midpoint = reordered.remove(mid);
            reordered.insert(0, midpoint);
            // Re-number discourse positions
            for (i, seg) in reordered.iter_mut().enumerate() {
                seg.discourse_position = i;
                if i == 0 {
                    seg.temporal_shift = TemporalShift::InMediasRes;
                } else if i == 1 {
                    seg.temporal_shift = TemporalShift::Analepsis {
                        reach: mid,
                        extent: mid,
                    };
                }
            }
            let candidate_sjuzet = Sjuzet {
                narrative_id: narrative_id.to_string(),
                segments: reordered.clone(),
            };
            candidates.push(SjuzetCandidate {
                label: "In Medias Res".to_string(),
                segments: reordered,
                irony_score: 0.3,
                tension_score: 0.7,
                divergence: compute_divergence(fabula, &candidate_sjuzet),
            });
        }
    }

    // Candidate 2: Reverse chronological (memento-style)
    {
        let mut reordered = sjuzet.segments.clone();
        reordered.reverse();
        for (i, seg) in reordered.iter_mut().enumerate() {
            seg.discourse_position = i;
            if i > 0 {
                seg.temporal_shift = TemporalShift::Analepsis {
                    reach: 1,
                    extent: 1,
                };
            }
        }
        let candidate_sjuzet = Sjuzet {
            narrative_id: narrative_id.to_string(),
            segments: reordered.clone(),
        };
        candidates.push(SjuzetCandidate {
            label: "Reverse Chronological".to_string(),
            segments: reordered,
            irony_score: 0.8,
            tension_score: 0.5,
            divergence: compute_divergence(fabula, &candidate_sjuzet),
        });
    }

    // Candidate 3: Frame narrative — bookend structure
    if fabula.situations.len() >= 4 {
        let mut reordered = sjuzet.segments.clone();
        if let Some(last) = reordered.pop() {
            reordered.insert(1, last);
            for (i, seg) in reordered.iter_mut().enumerate() {
                seg.discourse_position = i;
            }
            let candidate_sjuzet = Sjuzet {
                narrative_id: narrative_id.to_string(),
                segments: reordered.clone(),
            };
            candidates.push(SjuzetCandidate {
                label: "Frame Narrative".to_string(),
                segments: reordered,
                irony_score: 0.5,
                tension_score: 0.4,
                divergence: compute_divergence(fabula, &candidate_sjuzet),
            });
        }
    }

    Ok(candidates)
}

// ─── Inference Engine ───────────────────────────────────────

pub struct FabulaSjuzetEngine;

impl InferenceEngine for FabulaSjuzetEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::FabulaExtraction
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

        let fabula = extract_fabula(hg, narrative_id)?;
        let sjuzet = extract_sjuzet(hg, narrative_id)?;
        let divergence = compute_divergence(&fabula, &sjuzet);
        let mode_dist = narration_mode_distribution(&sjuzet);

        let result = serde_json::json!({
            "fabula": fabula,
            "sjuzet": sjuzet,
            "divergence": divergence,
            "narration_mode_distribution": mode_dist,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::FabulaExtraction,
            target_id: job.target_id,
            result,
            confidence: 0.8,
            explanation: Some(format!(
                "Fabula-sjužet divergence: {:.2} ({} situations)",
                divergence,
                fabula.situations.len()
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

    fn make_situation(hg: &Hypergraph, nid: &str, hour: i64, text: &str) -> Uuid {
        let start = DateTime::from_timestamp(1700000000 + hour * 3600, 0).unwrap();
        let end = start + chrono::Duration::hours(1);
        let s = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(format!("Scene at hour {}", hour)),
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
            raw_content: vec![ContentBlock::text(text)],
            narrative_level: NarrativeLevel::Scene,
            narrative_id: Some(nid.to_string()),
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
    fn test_fabula_extraction_chronological() {
        let hg = test_hg();
        let nid = "chrono-story";

        let s1 = make_situation(&hg, nid, 0, "The beginning");
        let s2 = make_situation(&hg, nid, 24, "The middle");
        let s3 = make_situation(&hg, nid, 48, "The end");

        let fabula = extract_fabula(&hg, nid).unwrap();
        assert_eq!(fabula.situations.len(), 3);
        assert_eq!(fabula.situations[0], s1);
        assert_eq!(fabula.situations[1], s2);
        assert_eq!(fabula.situations[2], s3);
    }

    #[test]
    fn test_sjuzet_extraction() {
        let hg = test_hg();
        let nid = "sjuzet-test";

        make_situation(&hg, nid, 0, "First event");
        make_situation(&hg, nid, 24, "Second event");
        make_situation(&hg, nid, 48, "Third event");

        let sjuzet = extract_sjuzet(&hg, nid).unwrap();
        assert_eq!(sjuzet.segments.len(), 3);
    }

    #[test]
    fn test_divergence_chronological_is_zero() {
        let hg = test_hg();
        let nid = "div-zero";

        make_situation(&hg, nid, 0, "A");
        make_situation(&hg, nid, 24, "B");
        make_situation(&hg, nid, 48, "C");

        let fabula = extract_fabula(&hg, nid).unwrap();
        let sjuzet = extract_sjuzet(&hg, nid).unwrap();
        let div = compute_divergence(&fabula, &sjuzet);
        assert!(
            div < 0.01,
            "Chronological narrative should have ~0 divergence, got {}",
            div
        );
    }

    #[test]
    fn test_divergence_reversed_is_one() {
        // A fully reversed sequence has maximum inversions
        let fabula = Fabula {
            narrative_id: "test".into(),
            situations: vec![Uuid::nil(); 4],
            earliest: None,
            latest: None,
        };
        let sjuzet = Sjuzet {
            narrative_id: "test".into(),
            segments: (0..4)
                .rev()
                .map(|i| SjuzetSegment {
                    situation_id: Uuid::nil(),
                    discourse_position: 3 - i,
                    fabula_position: i,
                    narration_mode: NarrationMode::Scene,
                    temporal_shift: TemporalShift::Chronological,
                })
                .collect(),
        };
        let div = compute_divergence(&fabula, &sjuzet);
        assert!(
            (div - 1.0).abs() < 0.01,
            "Fully reversed should be ~1.0, got {}",
            div
        );
    }

    #[test]
    fn test_narration_mode_classification() {
        let sit_dialogue = Situation {
            id: Uuid::nil(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: crate::types::AllenInterval {
                start: None,
                end: None,
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
            raw_content: vec![ContentBlock::text(
                r#""Hello," said Alice. "How are you?" asked Bob. "Fine," she replied. "Good," he said."#,
            )],
            narrative_level: NarrativeLevel::Scene,
            narrative_id: None,
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
        assert_eq!(classify_narration_mode(&sit_dialogue), NarrationMode::Scene);
    }

    #[test]
    fn test_suggest_reordering() {
        let hg = test_hg();
        let nid = "reorder-test";

        for i in 0..6 {
            make_situation(&hg, nid, i * 24, &format!("Event {}", i));
        }

        let fabula = extract_fabula(&hg, nid).unwrap();
        let sjuzet = extract_sjuzet(&hg, nid).unwrap();
        let candidates = suggest_reordering(&fabula, &sjuzet, &hg, nid).unwrap();

        assert!(candidates.len() >= 2);
        assert_eq!(candidates[0].label, "In Medias Res");
        assert_eq!(candidates[1].label, "Reverse Chronological");
        // Reverse should have divergence ~1.0
        assert!(candidates[1].divergence > 0.8);
    }

    #[test]
    fn test_fabula_sjuzet_kv_persistence() {
        let hg = test_hg();
        let nid = "persist-test";

        make_situation(&hg, nid, 0, "A");
        make_situation(&hg, nid, 24, "B");

        let fabula = extract_fabula(&hg, nid).unwrap();
        let loaded = load_fabula(&hg, nid).unwrap().unwrap();
        assert_eq!(fabula.situations, loaded.situations);

        let sjuzet = extract_sjuzet(&hg, nid).unwrap();
        let loaded_s = load_sjuzet(&hg, nid).unwrap().unwrap();
        assert_eq!(sjuzet.segments.len(), loaded_s.segments.len());
    }
}
