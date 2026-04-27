//! Analysis of Competing Hypotheses (ACH) workflow.
//!
//! Implements the Heuer & Pherson structured analytic technique:
//! 1. Define a set of competing hypotheses
//! 2. List evidence items
//! 3. Score each evidence-hypothesis pair for consistency
//! 4. Rank hypotheses by weighted inconsistency minimization
//!
//! Builds on Dempster-Shafer evidence framework for mass-function scoring.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::store::KVStore;

/// KV prefix for ACH hypothesis sets.
pub const ACH_PREFIX: &str = "ach/";

/// A hypothesis in the ACH framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

/// An evidence item to evaluate against hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    /// Source credibility weight (0.0-1.0). Default: 1.0.
    #[serde(default = "default_credibility")]
    pub credibility: f64,
    /// UUID of the entity or situation this evidence refers to.
    pub source_id: Option<Uuid>,
}

fn default_credibility() -> f64 {
    1.0
}

/// Consistency rating between evidence and hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Consistency {
    /// Evidence strongly supports hypothesis.
    StronglyConsistent,
    /// Evidence is consistent with hypothesis.
    Consistent,
    /// Evidence is neutral / not applicable.
    Neutral,
    /// Evidence is inconsistent with hypothesis.
    Inconsistent,
    /// Evidence strongly contradicts hypothesis.
    StronglyInconsistent,
}

impl Consistency {
    /// Numeric score: positive = consistent, negative = inconsistent.
    pub fn score(&self) -> f64 {
        match self {
            Consistency::StronglyConsistent => 2.0,
            Consistency::Consistent => 1.0,
            Consistency::Neutral => 0.0,
            Consistency::Inconsistent => -1.0,
            Consistency::StronglyInconsistent => -2.0,
        }
    }
}

/// A full ACH hypothesis set with evidence and scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisSet {
    pub id: String,
    pub narrative_id: String,
    pub hypotheses: Vec<Hypothesis>,
    pub evidence: Vec<EvidenceItem>,
    /// Consistency matrix: `scores[evidence_idx][hypothesis_idx]`.
    #[serde(default)]
    pub scores: Vec<Vec<Consistency>>,
    /// Ranked hypotheses (best first) after scoring.
    #[serde(default)]
    pub ranking: Vec<HypothesisRank>,
}

/// Ranked hypothesis with aggregated scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisRank {
    pub hypothesis_id: String,
    pub label: String,
    /// Weighted consistency score (higher = more supported).
    pub weighted_score: f64,
    /// Count of inconsistent evidence items.
    pub inconsistency_count: usize,
    /// Rank position (1 = best).
    pub rank: usize,
}

/// Score the consistency matrix and rank hypotheses.
///
/// ACH methodology: rank by minimizing weighted inconsistency.
/// Hypotheses with the fewest strong inconsistencies rank highest.
pub fn score_and_rank(set: &mut HypothesisSet) {
    let nh = set.hypotheses.len();
    let ne = set.evidence.len();

    if nh == 0 || ne == 0 || set.scores.len() != ne {
        return;
    }

    let mut rankings: Vec<HypothesisRank> = Vec::with_capacity(nh);

    for (h_idx, hyp) in set.hypotheses.iter().enumerate() {
        let mut weighted_score = 0.0;
        let mut inconsistency_count = 0;

        for (e_idx, ev) in set.evidence.iter().enumerate() {
            if h_idx >= set.scores[e_idx].len() {
                continue;
            }
            let consistency = &set.scores[e_idx][h_idx];
            let score = consistency.score() * ev.credibility;
            weighted_score += score;

            if matches!(
                consistency,
                Consistency::Inconsistent | Consistency::StronglyInconsistent
            ) {
                inconsistency_count += 1;
            }
        }

        rankings.push(HypothesisRank {
            hypothesis_id: hyp.id.clone(),
            label: hyp.label.clone(),
            weighted_score,
            inconsistency_count,
            rank: 0, // will be set after sorting
        });
    }

    // Sort: fewest inconsistencies first, then highest weighted score
    rankings.sort_by(|a, b| {
        a.inconsistency_count
            .cmp(&b.inconsistency_count)
            .then_with(|| {
                b.weighted_score
                    .partial_cmp(&a.weighted_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Assign ranks
    for (i, r) in rankings.iter_mut().enumerate() {
        r.rank = i + 1;
    }

    set.ranking = rankings;
}

/// Store an ACH hypothesis set to KV.
pub fn store_hypothesis_set(store: &dyn KVStore, set: &HypothesisSet) -> Result<()> {
    let key = format!("{}{}/{}", ACH_PREFIX, set.narrative_id, set.id);
    let value = serde_json::to_vec(set)?;
    store.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Load an ACH hypothesis set from KV.
pub fn get_hypothesis_set(
    store: &dyn KVStore,
    narrative_id: &str,
    set_id: &str,
) -> Result<Option<HypothesisSet>> {
    let key = format!("{}{}/{}", ACH_PREFIX, narrative_id, set_id);
    match store.get(key.as_bytes())? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List all ACH hypothesis sets for a narrative.
pub fn list_hypothesis_sets(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<HypothesisSet>> {
    let prefix = format!("{}{}/", ACH_PREFIX, narrative_id);
    let pairs = store.prefix_scan(prefix.as_bytes())?;
    let mut sets = Vec::new();
    for (_key, value) in pairs {
        if let Ok(set) = serde_json::from_slice(&value) {
            sets.push(set);
        }
    }
    Ok(sets)
}

/// Delete an ACH hypothesis set.
pub fn delete_hypothesis_set(
    store: &dyn KVStore,
    narrative_id: &str,
    set_id: &str,
) -> Result<bool> {
    let key = format!("{}{}/{}", ACH_PREFIX, narrative_id, set_id);
    let existed = store.get(key.as_bytes())?.is_some();
    if existed {
        store.transaction(vec![crate::store::TxnOp::Delete(key.into_bytes())])?;
    }
    Ok(existed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn make_set() -> HypothesisSet {
        HypothesisSet {
            id: "ach-001".to_string(),
            narrative_id: "case-alpha".to_string(),
            hypotheses: vec![
                Hypothesis {
                    id: "h1".to_string(),
                    label: "Suspect A did it".to_string(),
                    description: None,
                },
                Hypothesis {
                    id: "h2".to_string(),
                    label: "Suspect B did it".to_string(),
                    description: None,
                },
                Hypothesis {
                    id: "h3".to_string(),
                    label: "Accident (no suspect)".to_string(),
                    description: None,
                },
            ],
            evidence: vec![
                EvidenceItem {
                    id: "e1".to_string(),
                    label: "Fingerprints at scene".to_string(),
                    description: None,
                    credibility: 0.9,
                    source_id: None,
                },
                EvidenceItem {
                    id: "e2".to_string(),
                    label: "Witness saw Suspect A".to_string(),
                    description: None,
                    credibility: 0.7,
                    source_id: None,
                },
                EvidenceItem {
                    id: "e3".to_string(),
                    label: "Suspect B has alibi".to_string(),
                    description: None,
                    credibility: 0.8,
                    source_id: None,
                },
            ],
            scores: vec![
                // e1: fingerprints → consistent with A, neutral for B, inconsistent with accident
                vec![
                    Consistency::Consistent,
                    Consistency::Neutral,
                    Consistency::Inconsistent,
                ],
                // e2: witness saw A → strongly consistent with A, inconsistent with B, neutral for accident
                vec![
                    Consistency::StronglyConsistent,
                    Consistency::Inconsistent,
                    Consistency::Neutral,
                ],
                // e3: B has alibi → neutral for A, strongly inconsistent with B, neutral for accident
                vec![
                    Consistency::Neutral,
                    Consistency::StronglyInconsistent,
                    Consistency::Neutral,
                ],
            ],
            ranking: vec![],
        }
    }

    #[test]
    fn test_ach_score_and_rank() {
        let mut set = make_set();
        score_and_rank(&mut set);

        assert_eq!(set.ranking.len(), 3);
        // Suspect A should rank highest (most consistent, fewest inconsistencies)
        assert_eq!(set.ranking[0].hypothesis_id, "h1");
        assert_eq!(set.ranking[0].rank, 1);
        assert_eq!(set.ranking[0].inconsistency_count, 0);
        // Suspect B should rank last (most inconsistencies)
        assert_eq!(set.ranking[2].hypothesis_id, "h2");
        assert!(set.ranking[2].inconsistency_count >= 2);
    }

    #[test]
    fn test_ach_weighted_score() {
        let mut set = make_set();
        score_and_rank(&mut set);

        let h1_rank = set
            .ranking
            .iter()
            .find(|r| r.hypothesis_id == "h1")
            .unwrap();
        // e1: 1.0 * 0.9 + e2: 2.0 * 0.7 + e3: 0.0 * 0.8 = 0.9 + 1.4 = 2.3
        assert!((h1_rank.weighted_score - 2.3).abs() < 0.01);
    }

    #[test]
    fn test_ach_empty_evidence() {
        let mut set = HypothesisSet {
            id: "empty".to_string(),
            narrative_id: "test".to_string(),
            hypotheses: vec![Hypothesis {
                id: "h1".to_string(),
                label: "Only hypothesis".to_string(),
                description: None,
            }],
            evidence: vec![],
            scores: vec![],
            ranking: vec![],
        };
        score_and_rank(&mut set);
        assert!(set.ranking.is_empty()); // no evidence → no ranking
    }

    #[test]
    fn test_ach_persistence() {
        let store = Arc::new(MemoryStore::new());
        let set = make_set();
        store_hypothesis_set(store.as_ref(), &set).unwrap();

        let loaded = get_hypothesis_set(store.as_ref(), "case-alpha", "ach-001")
            .unwrap()
            .expect("should exist");
        assert_eq!(loaded.hypotheses.len(), 3);
        assert_eq!(loaded.evidence.len(), 3);
    }

    #[test]
    fn test_ach_list_and_delete() {
        let store = Arc::new(MemoryStore::new());
        let set = make_set();
        store_hypothesis_set(store.as_ref(), &set).unwrap();

        let list = list_hypothesis_sets(store.as_ref(), "case-alpha").unwrap();
        assert_eq!(list.len(), 1);

        let deleted = delete_hypothesis_set(store.as_ref(), "case-alpha", "ach-001").unwrap();
        assert!(deleted);

        let list2 = list_hypothesis_sets(store.as_ref(), "case-alpha").unwrap();
        assert!(list2.is_empty());
    }

    #[test]
    fn test_consistency_scores() {
        assert_eq!(Consistency::StronglyConsistent.score(), 2.0);
        assert_eq!(Consistency::Consistent.score(), 1.0);
        assert_eq!(Consistency::Neutral.score(), 0.0);
        assert_eq!(Consistency::Inconsistent.score(), -1.0);
        assert_eq!(Consistency::StronglyInconsistent.score(), -2.0);
    }
}
