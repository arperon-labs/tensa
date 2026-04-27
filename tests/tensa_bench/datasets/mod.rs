//! Dataset loaders for academic benchmarks.
//!
//! Each dataset is expected at `$TENSA_BENCHMARK_DATA/<dataset_name>/`.
//! Loaders parse standard formats and return typed Rust structs.

pub mod gdelt;
pub mod hotpotqa;
pub mod icews;
pub mod maven_ere;
pub mod narrativeqa;
pub mod rocstories;
pub mod wikimultihop;

use serde::de::DeserializeOwned;
use std::path::Path;

/// Parse a CSV row, handling quoted fields with embedded commas/tabs.
///
/// `delimiter` should be `,` for CSV or `\t` for TSV.
pub fn parse_csv_row(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == delimiter {
            fields.push(current.clone());
            current.clear();
        } else {
            current.push(c);
        }
    }
    fields.push(current);
    fields
}

/// Load a JSONL (newline-delimited JSON) file into `Vec<T>`.
///
/// Skips blank lines; returns `Err` with file:line context on malformed records.
pub fn load_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut items = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let item: T = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {}", path.display(), i + 1, e))?;
        items.push(item);
    }
    Ok(items)
}

/// Dataset split (train/validation/test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Valid,
    Test,
}

impl Split {
    pub fn filename_suffix(&self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Valid => "valid",
            Split::Test => "test",
        }
    }
}

/// Trait for dataset loaders.
pub trait DatasetLoader {
    type Item;

    /// Load a specific split of the dataset.
    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String>;

    /// Expected subdirectory name under TENSA_BENCHMARK_DATA.
    fn dir_name() -> &'static str;

    /// Check if the dataset directory exists.
    fn is_available(data_dir: &Path) -> bool {
        data_dir.join(Self::dir_name()).exists()
    }

    /// Path to this dataset's directory.
    fn dataset_path(data_dir: &Path) -> std::path::PathBuf {
        data_dir.join(Self::dir_name())
    }
}

/// Print download instructions for a missing dataset.
pub fn download_instructions(dataset: &str) -> String {
    match dataset {
        "icews14" => concat!(
            "ICEWS14 dataset not found.\n",
            "Download from: https://github.com/INK-USC/RE-Net/tree/master/data/ICEWS14\n",
            "Place files (train.txt, valid.txt, test.txt, entity2id.txt, relation2id.txt) ",
            "in $TENSA_BENCHMARK_DATA/icews14/"
        )
        .to_string(),
        "icews18" => concat!(
            "ICEWS18 dataset not found.\n",
            "Download from: https://github.com/INK-USC/RE-Net/tree/master/data/ICEWS18\n",
            "Place files in $TENSA_BENCHMARK_DATA/icews18/"
        )
        .to_string(),
        "gdelt" => concat!(
            "GDELT dataset not found.\n",
            "Download from: https://github.com/INK-USC/RE-Net/tree/master/data/GDELT\n",
            "Place files in $TENSA_BENCHMARK_DATA/gdelt/"
        )
        .to_string(),
        "hotpotqa" => concat!(
            "HotpotQA dataset not found.\n",
            "Download hotpot_dev_distractor_v1.json from: https://hotpotqa.github.io/\n",
            "Place in $TENSA_BENCHMARK_DATA/hotpotqa/\n",
            "License: CC-BY-SA 4.0"
        )
        .to_string(),
        "2wikimultihop" | "wikimultihop" => concat!(
            "2WikiMultiHopQA dataset not found.\n",
            "Download from: https://github.com/Alab-NII/2wikimultihop\n",
            "Place dev.json in $TENSA_BENCHMARK_DATA/2wikimultihop/\n",
            "License: MIT"
        )
        .to_string(),
        "rocstories" => concat!(
            "ROCStories / Story Cloze Test not found.\n",
            "Requires signing agreement at: https://cs.rochester.edu/nlp/rocstories/\n",
            "Place cloze_test_val.csv in $TENSA_BENCHMARK_DATA/rocstories/"
        )
        .to_string(),
        "narrativeqa" => concat!(
            "NarrativeQA dataset not found.\n",
            "Download from: https://github.com/google-deepmind/narrativeqa\n",
            "Place files in $TENSA_BENCHMARK_DATA/narrativeqa/\n",
            "License: Apache 2.0"
        )
        .to_string(),
        "maven_ere" => concat!(
            "MAVEN-ERE dataset not found.\n",
            "Download from: https://github.com/THU-KEG/MAVEN-ERE\n",
            "Place valid.jsonl (or test.jsonl) in $TENSA_BENCHMARK_DATA/maven_ere/\n",
            "License: CC-BY-SA 4.0"
        )
        .to_string(),
        _ => format!(
            "Unknown dataset: {}. Set TENSA_BENCHMARK_DATA to the data root.",
            dataset
        ),
    }
}
