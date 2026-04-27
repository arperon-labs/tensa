//! ROCStories / Story Cloze Test dataset loader.
//!
//! Format: CSV with columns:
//!   InputStoryid, InputSentence1, ..., InputSentence4,
//!   RandomFifthSentenceQuiz1, RandomFifthSentenceQuiz2, AnswerRightEnding
//!
//! Source: https://cs.rochester.edu/nlp/rocstories/
//! License: Requires signing academic agreement.

use super::{DatasetLoader, Split};
use std::path::Path;

/// A single Story Cloze Test item.
#[derive(Debug, Clone)]
pub struct StoryClozeItem {
    pub id: String,
    pub sentence1: String,
    pub sentence2: String,
    pub sentence3: String,
    pub sentence4: String,
    pub ending1: String,
    pub ending2: String,
    /// 1 or 2 — which ending is correct.
    pub correct_ending: u8,
}

impl StoryClozeItem {
    /// Get the 4-sentence story prefix as a single string.
    pub fn prefix(&self) -> String {
        format!(
            "{} {} {} {}",
            self.sentence1, self.sentence2, self.sentence3, self.sentence4
        )
    }

    /// Get the correct ending.
    pub fn correct(&self) -> &str {
        if self.correct_ending == 1 {
            &self.ending1
        } else {
            &self.ending2
        }
    }

    /// Get the incorrect ending.
    pub fn incorrect(&self) -> &str {
        if self.correct_ending == 1 {
            &self.ending2
        } else {
            &self.ending1
        }
    }
}

pub struct RocStories;

impl DatasetLoader for RocStories {
    type Item = StoryClozeItem;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        let dataset_dir = data_dir.join(Self::dir_name());

        let filename = match split {
            Split::Valid | Split::Test => "cloze_test_val__spring2016.csv",
            Split::Train => "cloze_test_val__spring2016.csv", // No separate train split
        };

        let path = dataset_dir.join(filename);
        // Try alternative filenames
        let path = if path.exists() {
            path
        } else {
            let alt = dataset_dir.join("cloze_test_val.csv");
            if alt.exists() {
                alt
            } else {
                let alt2 = dataset_dir.join("val.csv");
                if alt2.exists() {
                    alt2
                } else {
                    return Err(format!(
                        "Story Cloze Test CSV not found in {}. Expected cloze_test_val__spring2016.csv",
                        dataset_dir.display()
                    ));
                }
            }
        };

        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let mut items = Vec::new();
        let mut lines = content.lines();

        // Skip header
        let header = lines.next().unwrap_or("");
        if !header.contains("Sentence") && !header.contains("sentence") {
            // No header — treat as data
            if let Some(item) = parse_csv_line(header) {
                items.push(item);
            }
        }

        for line in lines {
            if let Some(item) = parse_csv_line(line) {
                items.push(item);
            }
        }

        Ok(items)
    }

    fn dir_name() -> &'static str {
        "rocstories"
    }
}

/// Parse a single CSV line into a StoryClozeItem.
///
/// The published file is comma-separated; legacy redistributions use tabs,
/// so we try both delimiters and pick whichever yields the expected column count.
fn parse_csv_line(line: &str) -> Option<StoryClozeItem> {
    let comma = super::parse_csv_row(line, ',');
    let fields = if comma.len() >= 8 {
        comma
    } else {
        let tab = super::parse_csv_row(line, '\t');
        if tab.len() >= 8 {
            tab
        } else {
            return None;
        }
    };

    let correct_ending: u8 = fields[7].trim().parse().ok()?;
    if correct_ending != 1 && correct_ending != 2 {
        return None;
    }

    Some(StoryClozeItem {
        id: fields[0].trim().to_string(),
        sentence1: fields[1].trim().to_string(),
        sentence2: fields[2].trim().to_string(),
        sentence3: fields[3].trim().to_string(),
        sentence4: fields[4].trim().to_string(),
        ending1: fields[5].trim().to_string(),
        ending2: fields[6].trim().to_string(),
        correct_ending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_line() {
        let line = "1,First sentence.,Second.,Third.,Fourth.,Ending A.,Ending B.,1";
        let item = parse_csv_line(line).unwrap();
        assert_eq!(item.id, "1");
        assert_eq!(item.sentence1, "First sentence.");
        assert_eq!(item.correct_ending, 1);
        assert_eq!(item.correct(), "Ending A.");
        assert_eq!(item.incorrect(), "Ending B.");
    }
}
