//! Three Primary Narrative Process model.
//!
//! Complements Reagan 6-arc taxonomy with structural phase analysis.
//! Based on "The Narrative Arc" (Science Advances, 2020) validated on ~60,000 narratives.
//!
//! Three processes measured as intensity curves across situations:
//! 1. **Staging**: setting establishment, character introductions
//! 2. **Plot Progression**: action, conflict, causal density
//! 3. **Cognitive Tension**: mystery, suspense, uncertainty

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;

/// Result of three-process analysis for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeProcessResult {
    pub narrative_id: String,
    /// Per-situation process intensities, in temporal order.
    pub points: Vec<ProcessPoint>,
    /// Aggregate phase boundaries (fraction of narrative).
    pub staging_peak: f64,
    pub progression_peak: f64,
    pub tension_peak: f64,
}

/// Process intensities for a single situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessPoint {
    pub situation_id: Uuid,
    /// Normalized position in narrative (0.0 = start, 1.0 = end).
    pub position: f64,
    /// Staging intensity: character introductions, setting descriptions.
    pub staging: f64,
    /// Plot progression intensity: action density, causal link count.
    pub progression: f64,
    /// Cognitive tension intensity: questions, uncertainty, suspense.
    pub tension: f64,
}

// ─── Keyword sets for process detection ─────────────────────

const STAGING_KEYWORDS: &[&str] = &[
    "arrived",
    "entered",
    "introduced",
    "appeared",
    "described",
    "setting",
    "background",
    "born",
    "grew up",
    "lived",
    "was a",
    "name was",
    "known as",
    "called",
    "first",
    "begin",
    "once upon",
];

const PROGRESSION_KEYWORDS: &[&str] = &[
    "attacked",
    "fought",
    "killed",
    "escaped",
    "chased",
    "discovered",
    "revealed",
    "betrayed",
    "confronted",
    "decided",
    "planned",
    "executed",
    "struggled",
    "defeated",
    "won",
    "lost",
    "changed",
    "transformed",
];

const TENSION_KEYWORDS: &[&str] = &[
    "mystery",
    "secret",
    "hidden",
    "unknown",
    "suspicious",
    "wondered",
    "feared",
    "worried",
    "uncertain",
    "confused",
    "strange",
    "dangerous",
    "threatened",
    "risk",
    "doubt",
    "question",
    "puzzle",
    "clue",
];

/// Analyze the three narrative processes for a narrative.
pub fn analyze_three_processes(
    narrative_id: &str,
    hypergraph: &Hypergraph,
) -> Result<ThreeProcessResult> {
    let mut situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    situations.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    let n = situations.len();
    if n == 0 {
        return Ok(ThreeProcessResult {
            narrative_id: narrative_id.to_string(),
            points: vec![],
            staging_peak: 0.0,
            progression_peak: 0.0,
            tension_peak: 0.0,
        });
    }

    let mut points = Vec::with_capacity(n);

    for (i, sit) in situations.iter().enumerate() {
        let position = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.5
        };

        // Collect text from content blocks
        let text: String = sit
            .raw_content
            .iter()
            .map(|b| b.content.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Staging: keyword count + new entity introductions signal
        let staging_keywords = count_keywords(&text, STAGING_KEYWORDS);
        // Bonus for early position (staging front-loads)
        let staging_position_bonus = (1.0 - position).powi(2) * 0.5;
        let staging = staging_keywords + staging_position_bonus;

        // Progression: keyword count + causal link density
        let progression_keywords = count_keywords(&text, PROGRESSION_KEYWORDS);
        let causal_links = hypergraph
            .get_consequences(&sit.id)
            .unwrap_or_default()
            .len() as f64;
        let progression = progression_keywords + causal_links * 0.5;

        // Tension: keyword count + question marks + uncertainty markers
        let tension_keywords = count_keywords(&text, TENSION_KEYWORDS);
        let question_marks = text.matches('?').count() as f64 * 0.3;
        let tension = tension_keywords + question_marks;

        points.push(ProcessPoint {
            situation_id: sit.id,
            position,
            staging,
            progression,
            tension,
        });
    }

    // Find peaks (position of maximum intensity for each process)
    let staging_peak = points
        .iter()
        .max_by(|a, b| {
            a.staging
                .partial_cmp(&b.staging)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.position)
        .unwrap_or(0.0);
    let progression_peak = points
        .iter()
        .max_by(|a, b| {
            a.progression
                .partial_cmp(&b.progression)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.position)
        .unwrap_or(0.5);
    let tension_peak = points
        .iter()
        .max_by(|a, b| {
            a.tension
                .partial_cmp(&b.tension)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.position)
        .unwrap_or(0.5);

    Ok(ThreeProcessResult {
        narrative_id: narrative_id.to_string(),
        points,
        staging_peak,
        progression_peak,
        tension_peak,
    })
}

/// Count how many keywords from the set appear in the text.
fn count_keywords(text: &str, keywords: &[&str]) -> f64 {
    keywords.iter().filter(|&&kw| text.contains(kw)).count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;

    #[test]
    fn test_three_process_empty() {
        let hg = make_hg();
        let result = analyze_three_processes("empty", &hg).unwrap();
        assert!(result.points.is_empty());
    }

    #[test]
    fn test_three_process_staging_front_loads() {
        let hg = make_hg();
        let nid = "tp-test";

        // Create situations with staging content early, action content late
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);

        // Overwrite content — we can't modify after creation, but staging
        // detection also uses position bonus (front-loads), so the first
        // situation gets staging bonus regardless.
        let _ = (s1, s2, s3); // used for creation

        let result = analyze_three_processes(nid, &hg).unwrap();
        assert_eq!(result.points.len(), 3);

        // First point should have highest staging (position bonus)
        assert!(
            result.points[0].staging >= result.points[2].staging,
            "First situation should have more staging than last"
        );
    }

    #[test]
    fn test_three_process_peaks() {
        let hg = make_hg();
        let nid = "tp-peaks";
        add_situation(&hg, nid);
        add_situation(&hg, nid);
        add_situation(&hg, nid);

        let result = analyze_three_processes(nid, &hg).unwrap();
        // Staging peak should be early (due to position bonus)
        assert!(result.staging_peak <= 0.5, "Staging should peak early");
    }

    #[test]
    fn test_count_keywords() {
        // "mystery" and "feared" both match
        assert_eq!(
            count_keywords("the mystery deepened as she feared", TENSION_KEYWORDS),
            2.0
        );
        assert_eq!(
            count_keywords("nothing special here", TENSION_KEYWORDS),
            0.0
        );
        // "arrived" and "introduced" both match
        assert_eq!(
            count_keywords("he arrived and introduced himself", STAGING_KEYWORDS),
            2.0
        );
    }

    #[test]
    fn test_three_process_serde() {
        let result = ThreeProcessResult {
            narrative_id: "test".to_string(),
            points: vec![ProcessPoint {
                situation_id: uuid::Uuid::now_v7(),
                position: 0.5,
                staging: 1.0,
                progression: 2.0,
                tension: 0.5,
            }],
            staging_peak: 0.1,
            progression_peak: 0.5,
            tension_peak: 0.7,
        };
        let json = serde_json::to_vec(&result).unwrap();
        let back: ThreeProcessResult = serde_json::from_slice(&json).unwrap();
        assert_eq!(back.points.len(), 1);
        assert!((back.tension_peak - 0.7).abs() < f64::EPSILON);
    }
}
