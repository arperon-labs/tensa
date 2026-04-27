//! Narrative adapter: maps ROCStories, NarrativeQA, and MAVEN-ERE to TENSA.
//!
//! MAVEN-ERE: compares TENSA's extracted causal/temporal relations against ground truth.
//! ROCStories: uses RAG to pick the correct story ending.
//! NarrativeQA: uses RAG to answer reading comprehension questions.

use crate::tensa_bench::datasets::maven_ere::{MavenDocument, MavenEvent, MavenRelation};
use crate::tensa_bench::metrics::classification::ConfusionMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mapping from TENSA causal types to MAVEN-ERE relation types.
///
/// TENSA: Necessary, Sufficient, Contributing, Preventive
/// MAVEN: CAUSE, PRECONDITION
/// TENSA temporal: Before, After, Overlaps, During, etc.
/// MAVEN temporal: BEFORE, AFTER, OVERLAP, SIMULTANEOUS, etc.
pub fn tensa_causal_to_maven(tensa_type: &str) -> Option<&'static str> {
    match tensa_type.to_lowercase().as_str() {
        "necessary" | "sufficient" => Some("CAUSE"),
        "contributing" => Some("PRECONDITION"),
        "preventive" => None, // No MAVEN equivalent
        _ => None,
    }
}

/// Map TENSA Allen relations to MAVEN-ERE temporal relation types.
pub fn tensa_temporal_to_maven(allen: &str) -> Option<&'static str> {
    match allen.to_lowercase().as_str() {
        "before" => Some("BEFORE"),
        "after" => Some("AFTER"),
        "overlaps" | "overlappedby" => Some("OVERLAP"),
        "during" | "contains" => Some("CONTAINS"),
        "equals" => Some("SIMULTANEOUS"),
        "starts" | "startedby" => Some("BEGINS_ON"),
        "finishes" | "finishedby" => Some("ENDS_ON"),
        _ => None,
    }
}

/// Result from evaluating MAVEN-ERE on a single document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MavenDocResult {
    pub doc_id: String,
    pub num_gold_relations: usize,
    pub num_predicted_relations: usize,
    pub num_matched: usize,
}

/// Extracted relation from TENSA (after ingestion).
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
}

/// Evaluate TENSA's extracted relations against MAVEN-ERE ground truth.
///
/// Uses fuzzy matching: if the extracted relation mentions match event trigger words
/// and the relation type maps correctly, it's counted as a match.
pub fn evaluate_maven_document(
    gold: &MavenDocument,
    predicted: &[ExtractedRelation],
) -> (ConfusionMatrix, MavenDocResult) {
    let mut cm = ConfusionMatrix::new();
    let mut matched = 0usize;

    let event_by_id: HashMap<&str, &MavenEvent> =
        gold.events.iter().map(|e| (e.id.as_str(), e)).collect();

    let gold_relations: Vec<(&MavenRelation, &str)> = gold
        .all_relations()
        .into_iter()
        .map(|r| (r, r.relation_type.as_str()))
        .collect();

    let mut used_predictions = vec![false; predicted.len()];

    for (gold_rel, gold_type) in &gold_relations {
        let mut found = false;
        for (pi, pred) in predicted.iter().enumerate() {
            if used_predictions[pi] {
                continue;
            }
            if relations_match(gold_rel, pred, &event_by_id) {
                cm.add_pair(&pred.relation_type, gold_type);
                used_predictions[pi] = true;
                found = true;
                matched += 1;
                break;
            }
        }
        if !found {
            cm.add(None, gold_type);
        }
    }

    for (pi, pred) in predicted.iter().enumerate() {
        if !used_predictions[pi] {
            // Unmatched prediction → FP against the predicted class
            cm.add(Some(&pred.relation_type), "__NONE__");
        }
    }

    let result = MavenDocResult {
        doc_id: gold.id.clone(),
        num_gold_relations: gold_relations.len(),
        num_predicted_relations: predicted.len(),
        num_matched: matched,
    };

    (cm, result)
}

/// Check if a predicted relation matches a gold relation.
///
/// Matching criteria:
/// 1. Predicted from_entity name contains or matches gold from_event trigger word
/// 2. Predicted to_entity name contains or matches gold to_event trigger word
/// 3. Relation types are compatible (via mapping)
fn relations_match(
    gold: &MavenRelation,
    predicted: &ExtractedRelation,
    event_by_id: &HashMap<&str, &MavenEvent>,
) -> bool {
    let from_trigger = event_by_id
        .get(gold.from_event.as_str())
        .map(|e| e.trigger_word.to_lowercase())
        .filter(|t| !t.is_empty());
    let to_trigger = event_by_id
        .get(gold.to_event.as_str())
        .map(|e| e.trigger_word.to_lowercase())
        .filter(|t| !t.is_empty());

    let (from_trigger, to_trigger) = match (from_trigger, to_trigger) {
        (Some(f), Some(t)) => (f, t),
        _ => return false,
    };

    let pred_from = predicted.from_entity.to_lowercase();
    let pred_to = predicted.to_entity.to_lowercase();

    let from_match = pred_from.contains(&from_trigger) || from_trigger.contains(&pred_from);
    let to_match = pred_to.contains(&to_trigger) || to_trigger.contains(&pred_to);
    if !from_match || !to_match {
        return false;
    }

    let pred_type_upper = predicted.relation_type.to_uppercase();
    let gold_type_upper = gold.relation_type.to_uppercase();
    if pred_type_upper == gold_type_upper {
        return true;
    }

    if let Some(mapped) = tensa_causal_to_maven(&predicted.relation_type) {
        if mapped == gold_type_upper {
            return true;
        }
    }
    if let Some(mapped) = tensa_temporal_to_maven(&predicted.relation_type) {
        if mapped == gold_type_upper {
            return true;
        }
    }

    false
}

/// Result for ROCStories story completion task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryClozeResult {
    pub item_id: String,
    pub correct: bool,
    pub predicted_ending: u8,
    pub gold_ending: u8,
}

/// Aggregate ROCStories results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryClozeAggResult {
    pub num_items: usize,
    pub accuracy: f64,
}

impl StoryClozeAggResult {
    pub fn from_results(results: &[StoryClozeResult]) -> Self {
        if results.is_empty() {
            return Self {
                num_items: 0,
                accuracy: 0.0,
            };
        }
        let correct = results.iter().filter(|r| r.correct).count();
        Self {
            num_items: results.len(),
            accuracy: correct as f64 / results.len() as f64,
        }
    }

    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "num_items": self.num_items,
            "accuracy": self.accuracy,
        })
    }
}

/// Parse an LLM response to determine which ending (A or B) was chosen.
pub fn parse_ending_choice(response: &str) -> Option<u8> {
    let response_lower = response.to_lowercase();

    // Look for explicit "A" or "B" indicators
    if response_lower.starts_with("a")
        || response_lower.contains("ending a")
        || response_lower.contains("option a")
        || response_lower.contains("first ending")
    {
        return Some(1);
    }
    if response_lower.starts_with("b")
        || response_lower.contains("ending b")
        || response_lower.contains("option b")
        || response_lower.contains("second ending")
    {
        return Some(2);
    }

    // Look for "1" or "2"
    if response_lower.contains("ending 1") || response_lower.starts_with("1") {
        return Some(1);
    }
    if response_lower.contains("ending 2") || response_lower.starts_with("2") {
        return Some(2);
    }

    None
}

/// Build a prompt for story completion evaluation.
pub fn build_story_cloze_prompt(prefix: &str, ending_a: &str, ending_b: &str) -> String {
    format!(
        "Given this story:\n\n{}\n\n\
         Which ending is more coherent and likely?\n\
         A: {}\n\
         B: {}\n\n\
         Answer with just the letter A or B.",
        prefix, ending_a, ending_b
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensa_causal_mapping() {
        assert_eq!(tensa_causal_to_maven("Necessary"), Some("CAUSE"));
        assert_eq!(tensa_causal_to_maven("Sufficient"), Some("CAUSE"));
        assert_eq!(tensa_causal_to_maven("Contributing"), Some("PRECONDITION"));
        assert_eq!(tensa_causal_to_maven("Preventive"), None);
    }

    #[test]
    fn test_tensa_temporal_mapping() {
        assert_eq!(tensa_temporal_to_maven("Before"), Some("BEFORE"));
        assert_eq!(tensa_temporal_to_maven("After"), Some("AFTER"));
        assert_eq!(tensa_temporal_to_maven("Overlaps"), Some("OVERLAP"));
        assert_eq!(tensa_temporal_to_maven("Equals"), Some("SIMULTANEOUS"));
    }

    #[test]
    fn test_parse_ending_choice() {
        assert_eq!(parse_ending_choice("A"), Some(1));
        assert_eq!(parse_ending_choice("B. The second ending"), Some(2));
        assert_eq!(parse_ending_choice("I think ending A is correct"), Some(1));
        assert_eq!(parse_ending_choice("Option B makes more sense"), Some(2));
        assert_eq!(parse_ending_choice("not sure"), None);
    }

    #[test]
    fn test_story_cloze_agg() {
        let results = vec![
            StoryClozeResult {
                item_id: "1".into(),
                correct: true,
                predicted_ending: 1,
                gold_ending: 1,
            },
            StoryClozeResult {
                item_id: "2".into(),
                correct: false,
                predicted_ending: 1,
                gold_ending: 2,
            },
            StoryClozeResult {
                item_id: "3".into(),
                correct: true,
                predicted_ending: 2,
                gold_ending: 2,
            },
        ];
        let agg = StoryClozeAggResult::from_results(&results);
        assert_eq!(agg.num_items, 3);
        assert!((agg.accuracy - 2.0 / 3.0).abs() < 0.01);
    }
}
