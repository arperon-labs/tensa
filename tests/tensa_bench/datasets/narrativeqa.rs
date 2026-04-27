//! NarrativeQA dataset loader.
//!
//! Format: CSV/JSON with document IDs, questions, and reference answers.
//! Each question has 2 reference answers for generative evaluation.
//!
//! Source: https://github.com/google-deepmind/narrativeqa
//! License: Apache 2.0

use super::{DatasetLoader, Split};
use serde::Deserialize;
use std::path::Path;

/// A single NarrativeQA item.
#[derive(Debug, Clone)]
pub struct NarrativeQAItem {
    pub document_id: String,
    pub question: String,
    pub answers: Vec<String>,
    /// Summary text (shorter than full document, used for summary-based eval).
    pub summary: Option<String>,
    /// Full document text (if available).
    pub document_text: Option<String>,
}

/// Raw JSON structure for NarrativeQA.
#[derive(Deserialize)]
struct RawNarrativeQAItem {
    #[serde(alias = "document_id", alias = "documentId")]
    document_id: String,
    question: String,
    #[serde(alias = "answer1")]
    answer_1: Option<String>,
    #[serde(alias = "answer2")]
    answer_2: Option<String>,
    #[serde(default)]
    answers: Vec<String>,
    #[serde(default)]
    summary: Option<String>,
}

pub struct NarrativeQA;

impl DatasetLoader for NarrativeQA {
    type Item = NarrativeQAItem;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        let dataset_dir = data_dir.join(Self::dir_name());

        // Try JSON format first, then CSV
        let json_path = dataset_dir.join(format!("{}.json", split.filename_suffix()));
        if json_path.exists() {
            return load_json(&json_path, &dataset_dir);
        }

        // Try JSONL
        let jsonl_path = dataset_dir.join(format!("{}.jsonl", split.filename_suffix()));
        if jsonl_path.exists() {
            return load_jsonl(&jsonl_path, &dataset_dir);
        }

        // Try CSV (the original format uses qaps.csv with a split column)
        let csv_path = dataset_dir.join("qaps.csv");
        if csv_path.exists() {
            return load_csv(&csv_path, &dataset_dir, split);
        }

        Err(format!(
            "NarrativeQA data not found in {}. Expected {}.json, {}.jsonl, or qaps.csv",
            dataset_dir.display(),
            split.filename_suffix(),
            split.filename_suffix(),
        ))
    }

    fn dir_name() -> &'static str {
        "narrativeqa"
    }
}

fn load_json(path: &Path, dataset_dir: &Path) -> Result<Vec<NarrativeQAItem>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let raw: Vec<RawNarrativeQAItem> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse NarrativeQA JSON: {}", e))?;
    Ok(raw
        .into_iter()
        .map(|r| convert_raw(r, dataset_dir))
        .collect())
}

fn load_jsonl(path: &Path, dataset_dir: &Path) -> Result<Vec<NarrativeQAItem>, String> {
    let raw_items: Vec<RawNarrativeQAItem> = super::load_jsonl(path)?;
    Ok(raw_items
        .into_iter()
        .map(|r| convert_raw(r, dataset_dir))
        .collect())
}

fn load_csv(path: &Path, dataset_dir: &Path, split: Split) -> Result<Vec<NarrativeQAItem>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    // Pre-load summaries from summaries.csv for efficient lookup
    let summaries = load_summaries_csv(dataset_dir);

    let split_name = split.filename_suffix();
    let mut items = Vec::new();
    let mut lines = content.lines();
    let _header = lines.next(); // Skip header
                                // qaps.csv: document_id,set,question,answer1,answer2,question_tokenized,answer1_tokenized,answer2_tokenized
                                // Fields may contain commas inside quotes

    for line in lines {
        let fields = super::parse_csv_row(line, ',');
        if fields.len() < 5 {
            continue;
        }
        let set = fields[1].trim();
        if set != split_name {
            continue;
        }

        let doc_id = fields[0].trim().to_string();
        let question = fields[2].trim().to_string();
        let answer1 = fields[3].trim().to_string();
        let answer2 = fields[4].trim().to_string();

        let summary = summaries
            .get(&doc_id)
            .cloned()
            .or_else(|| load_summary(dataset_dir, &doc_id));

        items.push(NarrativeQAItem {
            document_id: doc_id,
            question,
            answers: vec![answer1, answer2],
            summary,
            document_text: None,
        });
    }

    Ok(items)
}

/// Load all summaries from summaries.csv into a HashMap.
fn load_summaries_csv(dataset_dir: &Path) -> std::collections::HashMap<String, String> {
    let mut summaries = std::collections::HashMap::new();

    let path = dataset_dir.join("summaries.csv");
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return summaries,
    };

    let mut lines = content.lines();
    lines.next(); // header: document_id,set,summary,summary_tokenized

    for line in lines {
        let fields = super::parse_csv_row(line, ',');
        if fields.len() >= 3 {
            let doc_id = fields[0].trim().to_string();
            let summary = fields[2].trim().to_string();
            if !summary.is_empty() {
                summaries.insert(doc_id, summary);
            }
        }
    }

    summaries
}

fn convert_raw(raw: RawNarrativeQAItem, dataset_dir: &Path) -> NarrativeQAItem {
    let mut answers = raw.answers;
    if let Some(a1) = raw.answer_1 {
        if !answers.contains(&a1) {
            answers.push(a1);
        }
    }
    if let Some(a2) = raw.answer_2 {
        if !answers.contains(&a2) {
            answers.push(a2);
        }
    }

    let summary = raw
        .summary
        .or_else(|| load_summary(dataset_dir, &raw.document_id));

    NarrativeQAItem {
        document_id: raw.document_id,
        question: raw.question,
        answers,
        summary,
        document_text: None,
    }
}

/// Try to load a document summary from the summaries directory.
fn load_summary(dataset_dir: &Path, doc_id: &str) -> Option<String> {
    let summary_path = dataset_dir
        .join("summaries")
        .join(format!("{}.txt", doc_id));
    std::fs::read_to_string(summary_path).ok()
}
