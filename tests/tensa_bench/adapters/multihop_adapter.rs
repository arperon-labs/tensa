//! Multi-hop QA adapter: ingests HotpotQA/2WikiMultiHop passages into TENSA,
//! queries via RAG, and scores answers.

use crate::tensa_bench::datasets::hotpotqa::HotpotQAItem;
use crate::tensa_bench::metrics::qa;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for multi-hop evaluation.
pub struct MultihopConfig {
    /// Number of items to evaluate (0 = all).
    pub sample_size: usize,
    /// Retrieval modes to test.
    pub modes: Vec<String>,
    /// Narrative ID prefix.
    pub narrative_prefix: String,
}

impl Default for MultihopConfig {
    fn default() -> Self {
        Self {
            sample_size: 100,
            modes: vec![
                "local".to_string(),
                "hybrid".to_string(),
                "drift".to_string(),
                "ppr".to_string(),
            ],
            narrative_prefix: "hotpotqa".to_string(),
        }
    }
}

impl MultihopConfig {
    /// Create from environment variables.
    pub fn from_env() -> Self {
        let sample_size = std::env::var("TENSA_BENCH_SAMPLE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        Self {
            sample_size,
            ..Default::default()
        }
    }
}

/// Result for a single HotpotQA item evaluated with one retrieval mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultihopItemResult {
    pub item_id: String,
    pub question: String,
    pub gold_answer: String,
    pub predicted_answer: String,
    pub mode: String,
    pub exact_match: bool,
    pub token_f1: f64,
    pub latency_ms: u64,
}

/// Aggregated results for one retrieval mode across all items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultihopModeResult {
    pub mode: String,
    pub num_items: usize,
    pub avg_em: f64,
    pub avg_f1: f64,
    pub avg_latency_ms: f64,
    pub em_by_type: HashMap<String, f64>,
    pub f1_by_type: HashMap<String, f64>,
}

impl MultihopModeResult {
    /// Aggregate from individual item results.
    pub fn from_items(mode: &str, items: &[MultihopItemResult]) -> Self {
        if items.is_empty() {
            return Self {
                mode: mode.to_string(),
                num_items: 0,
                avg_em: 0.0,
                avg_f1: 0.0,
                avg_latency_ms: 0.0,
                em_by_type: HashMap::new(),
                f1_by_type: HashMap::new(),
            };
        }

        let n = items.len() as f64;
        let avg_em = items.iter().filter(|i| i.exact_match).count() as f64 / n;
        let avg_f1 = items.iter().map(|i| i.token_f1).sum::<f64>() / n;
        let avg_latency = items.iter().map(|i| i.latency_ms as f64).sum::<f64>() / n;

        Self {
            mode: mode.to_string(),
            num_items: items.len(),
            avg_em,
            avg_f1,
            avg_latency_ms: avg_latency,
            em_by_type: HashMap::new(), // Filled by caller if needed
            f1_by_type: HashMap::new(),
        }
    }

    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": self.mode,
            "num_items": self.num_items,
            "exact_match": self.avg_em,
            "token_f1": self.avg_f1,
            "avg_latency_ms": self.avg_latency_ms,
        })
    }
}

/// Build the ingestion text for a single HotpotQA item.
///
/// Concatenates all context paragraphs with titles as section headers.
pub fn build_ingestion_text(item: &HotpotQAItem) -> String {
    item.context
        .iter()
        .map(|(title, sentences)| format!("## {}\n\n{}", title, sentences.join(" ")))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Score a predicted answer against the gold answer.
pub fn score_answer(predicted: &str, gold: &str) -> (bool, f64) {
    let em = qa::exact_match(predicted, gold);
    let f1 = qa::token_f1(predicted, gold);
    (em, f1)
}

/// Build a TensaQL ASK query for a HotpotQA item.
pub fn build_ask_query(question: &str, narrative_id: &str, mode: &str) -> String {
    format!(
        r#"ASK "{}" OVER "{}" MODE {}"#,
        question.replace('"', r#"\""#),
        narrative_id,
        mode
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ask_query() {
        let q = build_ask_query("Who was the president?", "story-1", "hybrid");
        assert!(q.contains("ASK"));
        assert!(q.contains("OVER"));
        assert!(q.contains("hybrid"));
    }

    #[test]
    fn test_score_answer() {
        let (em, f1) = score_answer("New York City", "new york city");
        assert!(em);
        assert!((f1 - 1.0).abs() < 1e-6);

        let (em, f1) = score_answer("the big apple", "New York City");
        assert!(!em);
        assert!(f1 < 0.01); // No token overlap
    }

    #[test]
    fn test_mode_result_aggregation() {
        let items = vec![
            MultihopItemResult {
                item_id: "1".to_string(),
                question: "Q1".to_string(),
                gold_answer: "A1".to_string(),
                predicted_answer: "A1".to_string(),
                mode: "local".to_string(),
                exact_match: true,
                token_f1: 1.0,
                latency_ms: 100,
            },
            MultihopItemResult {
                item_id: "2".to_string(),
                question: "Q2".to_string(),
                gold_answer: "A2".to_string(),
                predicted_answer: "wrong".to_string(),
                mode: "local".to_string(),
                exact_match: false,
                token_f1: 0.0,
                latency_ms: 200,
            },
        ];
        let result = MultihopModeResult::from_items("local", &items);
        assert_eq!(result.num_items, 2);
        assert!((result.avg_em - 0.5).abs() < 1e-6);
        assert!((result.avg_f1 - 0.5).abs() < 1e-6);
        assert!((result.avg_latency_ms - 150.0).abs() < 1e-6);
    }
}
