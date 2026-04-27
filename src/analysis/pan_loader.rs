//! Loader for PAN@CLEF authorship verification datasets.
//!
//! Supports two common PAN distribution formats:
//!
//! 1. Combined JSONL — one JSON object per line containing both the pair and
//!    the truth label:
//!
//!    ```json
//!    {"id": "0001", "pair": ["text1", "text2"], "same_author": true}
//!    ```
//!
//! 2. Split: one JSONL with the pairs (no `same_author` field) and a separate
//!    truth JSONL keyed by ID:
//!
//!    ```json
//!    {"id": "0001", "value": true}
//!    ```
//!
//! The loader reads both formats and produces `Vec<VerificationPair>`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::analysis::pan_verification::VerificationPair;
use crate::error::{Result, TensaError};

/// A raw PAN pairs record (matches the canonical PAN JSONL layout).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawPair {
    pub id: String,
    #[serde(default)]
    pub pair: Option<Vec<String>>,
    #[serde(default)]
    pub text_a: Option<String>,
    #[serde(default)]
    pub text_b: Option<String>,
    #[serde(default)]
    pub same_author: Option<bool>,
}

/// A raw PAN truth record.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawTruth {
    pub id: String,
    #[serde(default)]
    pub value: Option<bool>,
    #[serde(default)]
    pub same_author: Option<bool>,
}

fn raw_to_pair(raw: RawPair) -> Result<VerificationPair> {
    let (text_a, text_b) = match (raw.pair, raw.text_a, raw.text_b) {
        (Some(v), _, _) if v.len() >= 2 => (v[0].clone(), v[1].clone()),
        (_, Some(a), Some(b)) => (a, b),
        _ => {
            return Err(TensaError::ParseError(format!(
                "PAN pair {}: missing text_a/text_b (and no `pair` array)",
                raw.id
            )));
        }
    };
    Ok(VerificationPair {
        id: raw.id,
        text_a,
        text_b,
        same_author: raw.same_author,
    })
}

/// Load a JSONL file of PAN verification pairs.
///
/// Each line must be a JSON object with at minimum `id` and either a `pair`
/// array of two strings, or both `text_a` and `text_b` string fields. An
/// optional `same_author` boolean may be present (the combined format).
pub fn load_pan_jsonl<P: AsRef<Path>>(path: P) -> Result<Vec<VerificationPair>> {
    let file = File::open(&path).map_err(|e| {
        TensaError::Store(format!(
            "open PAN dataset {}: {}",
            path.as_ref().display(),
            e
        ))
    })?;
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            TensaError::Store(format!(
                "read PAN dataset {} line {}: {}",
                path.as_ref().display(),
                lineno + 1,
                e
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawPair = serde_json::from_str(&line).map_err(|e| {
            TensaError::Serialization(format!(
                "parse PAN line {} ({}): {}",
                lineno + 1,
                line.chars().take(80).collect::<String>(),
                e
            ))
        })?;
        pairs.push(raw_to_pair(raw)?);
    }
    Ok(pairs)
}

/// Load a separate PAN truth JSONL keyed by ID → same_author boolean.
pub fn load_pan_truth<P: AsRef<Path>>(path: P) -> Result<HashMap<String, bool>> {
    let file = File::open(&path).map_err(|e| {
        TensaError::Store(format!("open PAN truth {}: {}", path.as_ref().display(), e))
    })?;
    let reader = BufReader::new(file);
    let mut truth = HashMap::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            TensaError::Store(format!(
                "read PAN truth {} line {}: {}",
                path.as_ref().display(),
                lineno + 1,
                e
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawTruth = serde_json::from_str(&line).map_err(|e| {
            TensaError::Serialization(format!("parse PAN truth line {}: {}", lineno + 1, e))
        })?;
        let value = raw.value.or(raw.same_author).ok_or_else(|| {
            TensaError::ParseError(format!(
                "PAN truth {}: neither `value` nor `same_author` present",
                raw.id
            ))
        })?;
        truth.insert(raw.id, value);
    }
    Ok(truth)
}

/// Merge a truth map into a set of pairs, setting each pair's `same_author`
/// from the truth map. Pairs without a matching truth entry are left unchanged.
pub fn apply_truth(pairs: &mut [VerificationPair], truth: &HashMap<String, bool>) {
    for pair in pairs {
        if let Some(&v) = truth.get(&pair.id) {
            pair.same_author = Some(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_combined_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"id":"p1","pair":["alpha","beta"],"same_author":true}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"id":"p2","pair":["gamma","delta"],"same_author":false}}"#
        )
        .unwrap();
        let pairs = load_pan_jsonl(file.path()).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].id, "p1");
        assert_eq!(pairs[0].text_a, "alpha");
        assert_eq!(pairs[0].same_author, Some(true));
        assert_eq!(pairs[1].same_author, Some(false));
    }

    #[test]
    fn loads_text_ab_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"id":"x","text_a":"hello","text_b":"world"}}"#).unwrap();
        let pairs = load_pan_jsonl(file.path()).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].text_a, "hello");
        assert_eq!(pairs[0].text_b, "world");
        assert_eq!(pairs[0].same_author, None);
    }

    #[test]
    fn loads_truth_and_merges() {
        let mut pairs_file = NamedTempFile::new().unwrap();
        writeln!(pairs_file, r#"{{"id":"p1","pair":["a","b"]}}"#).unwrap();
        writeln!(pairs_file, r#"{{"id":"p2","pair":["c","d"]}}"#).unwrap();

        let mut truth_file = NamedTempFile::new().unwrap();
        writeln!(truth_file, r#"{{"id":"p1","value":true}}"#).unwrap();
        writeln!(truth_file, r#"{{"id":"p2","same_author":false}}"#).unwrap();

        let mut pairs = load_pan_jsonl(pairs_file.path()).unwrap();
        let truth = load_pan_truth(truth_file.path()).unwrap();
        apply_truth(&mut pairs, &truth);
        assert_eq!(pairs[0].same_author, Some(true));
        assert_eq!(pairs[1].same_author, Some(false));
    }

    #[test]
    fn rejects_malformed_lines() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "not json").unwrap();
        assert!(load_pan_jsonl(file.path()).is_err());
    }

    #[test]
    fn skips_blank_lines() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"id":"p1","pair":["a","b"],"same_author":true}}"#).unwrap();
        writeln!(file).unwrap();
        writeln!(
            file,
            r#"{{"id":"p2","pair":["c","d"],"same_author":false}}"#
        )
        .unwrap();
        let pairs = load_pan_jsonl(file.path()).unwrap();
        assert_eq!(pairs.len(), 2);
    }
}
