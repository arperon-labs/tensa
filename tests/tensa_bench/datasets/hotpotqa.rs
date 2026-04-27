//! HotpotQA dataset loader (distractor setting).
//!
//! Format: JSON array of objects, each with:
//! - `_id`: question ID
//! - `question`: the question string
//! - `answer`: gold answer string
//! - `type`: "comparison" or "bridge"
//! - `supporting_facts`: [[title, sent_idx], ...]
//! - `context`: [[title, [sent1, sent2, ...]], ...]
//!
//! Source: https://hotpotqa.github.io/
//! License: CC-BY-SA 4.0

use super::{DatasetLoader, Split};
use serde::Deserialize;
use std::path::Path;

/// A single HotpotQA item.
#[derive(Debug, Clone)]
pub struct HotpotQAItem {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub question_type: String,
    pub level: String,
    pub supporting_facts: Vec<(String, usize)>,
    pub context: Vec<(String, Vec<String>)>,
}

/// Raw JSON structure for deserialization.
#[derive(Deserialize)]
struct RawHotpotItem {
    _id: String,
    question: String,
    answer: String,
    #[serde(rename = "type")]
    question_type: String,
    #[serde(default)]
    level: String,
    supporting_facts: Vec<(String, usize)>,
    context: Vec<(String, Vec<String>)>,
}

pub struct HotpotQA;

impl DatasetLoader for HotpotQA {
    type Item = HotpotQAItem;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        let dataset_dir = data_dir.join(Self::dir_name());

        let filename = match split {
            Split::Train => "hotpot_train_v1.1.json",
            Split::Valid | Split::Test => "hotpot_dev_distractor_v1.json",
        };

        let path = dataset_dir.join(filename);
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let raw_items: Vec<RawHotpotItem> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse HotpotQA JSON: {}", e))?;

        let items = raw_items
            .into_iter()
            .map(|r| HotpotQAItem {
                id: r._id,
                question: r.question,
                answer: r.answer,
                question_type: r.question_type,
                level: r.level,
                supporting_facts: r.supporting_facts,
                context: r.context,
            })
            .collect();

        Ok(items)
    }

    fn dir_name() -> &'static str {
        "hotpotqa"
    }
}

impl HotpotQAItem {
    /// Get all context paragraphs as (title, full_text) pairs.
    pub fn context_paragraphs(&self) -> Vec<(String, String)> {
        self.context
            .iter()
            .map(|(title, sentences)| (title.clone(), sentences.join(" ")))
            .collect()
    }

    /// Get the gold supporting paragraph titles.
    pub fn gold_titles(&self) -> Vec<String> {
        self.supporting_facts
            .iter()
            .map(|(title, _)| title.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Get the full text of all context paragraphs concatenated.
    pub fn full_context_text(&self) -> String {
        self.context
            .iter()
            .map(|(title, sentences)| format!("{}: {}", title, sentences.join(" ")))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
