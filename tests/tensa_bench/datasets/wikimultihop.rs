//! 2WikiMultiHopQA dataset loader.
//!
//! Format: JSON array of objects with fields similar to HotpotQA plus
//! evidence chains and question decomposition.
//!
//! Source: https://github.com/Alab-NII/2wikimultihop
//! License: MIT

use super::{DatasetLoader, Split};
use serde::Deserialize;
use std::path::Path;

/// A single 2WikiMultiHopQA item.
#[derive(Debug, Clone)]
pub struct WikiMultiHopItem {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub question_type: String,
    pub supporting_facts: Vec<(String, usize)>,
    pub context: Vec<(String, Vec<String>)>,
    pub evidences: Vec<Evidence>,
}

/// An evidence triple linking two entities through a relation.
#[derive(Debug, Clone, Deserialize)]
pub struct Evidence {
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub relation: String,
    #[serde(default)]
    pub object: String,
}

/// Raw JSON structure.
#[derive(Deserialize)]
struct RawWikiItem {
    _id: String,
    question: String,
    answer: String,
    #[serde(rename = "type")]
    question_type: String,
    #[serde(default)]
    supporting_facts: Vec<(String, usize)>,
    context: Vec<(String, Vec<String>)>,
    #[serde(default)]
    evidences: Vec<Evidence>,
}

pub struct WikiMultiHop;

impl DatasetLoader for WikiMultiHop {
    type Item = WikiMultiHopItem;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        let dataset_dir = data_dir.join(Self::dir_name());

        let filename = match split {
            Split::Train => "train.json",
            Split::Valid => "dev.json",
            Split::Test => "test.json",
        };

        let path = dataset_dir.join(filename);
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let raw_items: Vec<RawWikiItem> = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse 2WikiMultiHopQA JSON: {}", e))?;

        let items = raw_items
            .into_iter()
            .map(|r| WikiMultiHopItem {
                id: r._id,
                question: r.question,
                answer: r.answer,
                question_type: r.question_type,
                supporting_facts: r.supporting_facts,
                context: r.context,
                evidences: r.evidences,
            })
            .collect();

        Ok(items)
    }

    fn dir_name() -> &'static str {
        "2wikimultihop"
    }
}

impl WikiMultiHopItem {
    /// Get all context paragraphs as concatenated text.
    pub fn full_context_text(&self) -> String {
        self.context
            .iter()
            .map(|(title, sentences)| format!("{}: {}", title, sentences.join(" ")))
            .collect::<Vec<_>>()
            .join("\n\n")
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
}
