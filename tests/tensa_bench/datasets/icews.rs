//! ICEWS14 / ICEWS18 temporal knowledge graph dataset loader.
//!
//! Supports two formats:
//! 1. RE-Net format: whitespace-separated `subject relation object timestamp_int [extra]`
//!    where timestamp_int is an integer index. ICEWS14: 0-based day offset from 2014-01-01.
//!    ICEWS18: 0-based day offset from 2018-01-01. May have a 5th unused column (always 0).
//! 2. Date format: tab-separated `subject\trelation\tobject\tYYYY-MM-DD`.
//!
//! Mapping files: entity2id.txt, relation2id.txt (tab-separated name\tid).
//! If entity2id.txt is missing, synthetic names ("entity_N") are generated from stat.txt.

use super::{DatasetLoader, Split};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::path::Path;

/// A single temporal triple (s, r, o, t).
#[derive(Debug, Clone)]
pub struct TemporalTriple {
    pub subject: u32,
    pub relation: u32,
    pub object: u32,
    pub timestamp: NaiveDate,
}

/// Loaded ICEWS dataset with entity/relation name mappings.
pub struct IcewsDataset {
    pub triples: Vec<TemporalTriple>,
    pub entity_names: HashMap<u32, String>,
    pub relation_names: HashMap<u32, String>,
}

impl IcewsDataset {
    pub fn num_entities(&self) -> usize {
        self.entity_names.len()
    }

    pub fn num_relations(&self) -> usize {
        self.relation_names.len()
    }
}

/// Base date for converting integer timestamps to NaiveDate.
#[derive(Debug, Clone, Copy)]
pub struct TimestampConfig {
    /// Base date: timestamp 0 maps to this date.
    pub base_date: NaiveDate,
    /// Granularity in days per timestamp unit (1 = daily).
    pub days_per_unit: i64,
}

impl TimestampConfig {
    pub fn icews14() -> Self {
        Self {
            base_date: NaiveDate::from_ymd_opt(2014, 1, 1).unwrap(),
            days_per_unit: 1,
        }
    }

    pub fn icews18() -> Self {
        Self {
            base_date: NaiveDate::from_ymd_opt(2018, 1, 1).unwrap(),
            days_per_unit: 1,
        }
    }

    pub fn gdelt() -> Self {
        Self {
            base_date: NaiveDate::from_ymd_opt(2018, 1, 1).unwrap(),
            days_per_unit: 1,
        }
    }

    /// Convert integer timestamp to NaiveDate.
    pub fn to_date(&self, ts: i64) -> NaiveDate {
        self.base_date + chrono::Duration::days(ts * self.days_per_unit)
    }
}

/// ICEWS14 loader.
pub struct Icews14;

/// ICEWS18 loader.
pub struct Icews18;

impl DatasetLoader for Icews14 {
    type Item = TemporalTriple;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        load_icews_split(
            &data_dir.join(Self::dir_name()),
            split,
            TimestampConfig::icews14(),
        )
    }

    fn dir_name() -> &'static str {
        "icews14"
    }
}

impl DatasetLoader for Icews18 {
    type Item = TemporalTriple;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        load_icews_split(
            &data_dir.join(Self::dir_name()),
            split,
            TimestampConfig::icews18(),
        )
    }

    fn dir_name() -> &'static str {
        "icews18"
    }
}

/// Load a full ICEWS dataset (all splits + mappings).
pub fn load_icews_full(dataset_dir: &Path) -> Result<IcewsDataset, String> {
    // Detect which ICEWS variant by directory name
    let dir_name = dataset_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let ts_config = if dir_name.contains("18") {
        TimestampConfig::icews18()
    } else if dir_name.contains("gdelt") {
        TimestampConfig::gdelt()
    } else {
        TimestampConfig::icews14()
    };

    let entity_names = load_id_mapping_or_generate(dataset_dir, "entity2id.txt", "stat.txt", 0)?;
    let relation_names =
        load_id_mapping_or_generate(dataset_dir, "relation2id.txt", "stat.txt", 1)?;

    let mut all_triples = Vec::new();
    for split in &[Split::Train, Split::Valid, Split::Test] {
        // ICEWS14 has no valid.txt; tolerate missing splits.
        if let Ok(triples) = load_icews_split(dataset_dir, *split, ts_config) {
            all_triples.extend(triples);
        }
    }

    Ok(IcewsDataset {
        triples: all_triples,
        entity_names,
        relation_names,
    })
}

/// Load triples from a single split file.
///
/// Handles both formats:
/// - RE-Net: `subject relation object timestamp_int [extra]` (whitespace-separated)
/// - Date: `subject\trelation\tobject\tYYYY-MM-DD`
fn load_icews_split(
    dataset_dir: &Path,
    split: Split,
    ts_config: TimestampConfig,
) -> Result<Vec<TemporalTriple>, String> {
    let filename = format!("{}.txt", split.filename_suffix());
    let path = dataset_dir.join(&filename);

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut triples = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Split on any whitespace (handles both tab and space-separated)
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!(
                "{}:{}: expected at least 4 fields, got {}",
                filename,
                line_no + 1,
                parts.len()
            ));
        }

        let subject: u32 = parts[0]
            .parse()
            .map_err(|e| format!("{}:{}: bad subject: {}", filename, line_no + 1, e))?;
        let relation: u32 = parts[1]
            .parse()
            .map_err(|e| format!("{}:{}: bad relation: {}", filename, line_no + 1, e))?;
        let object: u32 = parts[2]
            .parse()
            .map_err(|e| format!("{}:{}: bad object: {}", filename, line_no + 1, e))?;

        // Try parsing as date first, fall back to integer timestamp
        let timestamp = if let Ok(date) = NaiveDate::parse_from_str(parts[3], "%Y-%m-%d") {
            date
        } else if let Ok(ts_int) = parts[3].parse::<i64>() {
            ts_config.to_date(ts_int)
        } else {
            return Err(format!(
                "{}:{}: bad timestamp '{}': expected YYYY-MM-DD or integer",
                filename,
                line_no + 1,
                parts[3]
            ));
        };

        triples.push(TemporalTriple {
            subject,
            relation,
            object,
            timestamp,
        });
    }

    Ok(triples)
}

/// Load an entity2id.txt or relation2id.txt mapping file.
///
/// Format: first line is count (optional), remaining lines are `name\tid`.
fn load_id_mapping(path: &Path) -> Result<HashMap<u32, String>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut mapping = HashMap::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();

        // First line might be just a count (single integer)
        if i == 0 && parts.len() == 1 {
            if parts[0].parse::<usize>().is_ok() {
                continue; // Skip count header
            }
        }

        if parts.len() < 2 {
            continue;
        }

        let name = parts[0].to_string();
        let id: u32 = parts[1]
            .parse()
            .map_err(|e| format!("{}:{}: bad id: {}", path.display(), i + 1, e))?;
        mapping.insert(id, name);
    }

    Ok(mapping)
}

/// Load an ID mapping file, or generate synthetic names from stat.txt if missing.
///
/// `stat_field_idx`: 0 = entity count, 1 = relation count (in stat.txt).
fn load_id_mapping_or_generate(
    dataset_dir: &Path,
    mapping_file: &str,
    stat_file: &str,
    stat_field_idx: usize,
) -> Result<HashMap<u32, String>, String> {
    let mapping_path = dataset_dir.join(mapping_file);
    if mapping_path.exists() {
        let mapping = load_id_mapping(&mapping_path)?;
        if !mapping.is_empty() {
            return Ok(mapping);
        }
    }

    // Fall back to stat.txt
    let stat_path = dataset_dir.join(stat_file);
    if stat_path.exists() {
        let content = std::fs::read_to_string(&stat_path)
            .map_err(|e| format!("Failed to read {}: {}", stat_path.display(), e))?;
        let parts: Vec<&str> = content.trim().split_whitespace().collect();
        if let Some(count_str) = parts.get(stat_field_idx) {
            if let Ok(count) = count_str.parse::<u32>() {
                let prefix = if stat_field_idx == 0 {
                    "entity"
                } else {
                    "relation"
                };
                let mapping: HashMap<u32, String> = (0..count)
                    .map(|i| (i, format!("{}_{}", prefix, i)))
                    .collect();
                return Ok(mapping);
            }
        }
    }

    // Last resort: scan the data files to find max entity/relation ID
    let mut max_id: u32 = 0;
    for split_name in &["train", "valid", "test"] {
        let path = dataset_dir.join(format!("{}.txt", split_name));
        if let Ok(content) = std::fs::read_to_string(&path) {
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let idx = if stat_field_idx == 0 {
                        // Scan subject and object columns for entity IDs
                        let s: u32 = parts[0].parse().unwrap_or(0);
                        let o: u32 = parts[2].parse().unwrap_or(0);
                        s.max(o)
                    } else {
                        // Scan relation column
                        parts[1].parse().unwrap_or(0)
                    };
                    max_id = max_id.max(idx);
                }
            }
        }
    }

    let prefix = if stat_field_idx == 0 {
        "entity"
    } else {
        "relation"
    };
    let mapping: HashMap<u32, String> = (0..=max_id)
        .map(|i| (i, format!("{}_{}", prefix, i)))
        .collect();
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_triple_date_format() {
        let line = "42\t7\t100\t2014-01-15";
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts.len(), 4);
        let subject: u32 = parts[0].parse().unwrap();
        let relation: u32 = parts[1].parse().unwrap();
        let object: u32 = parts[2].parse().unwrap();
        let timestamp = NaiveDate::parse_from_str(parts[3], "%Y-%m-%d").unwrap();
        assert_eq!(subject, 42);
        assert_eq!(relation, 7);
        assert_eq!(object, 100);
        assert_eq!(timestamp, NaiveDate::from_ymd_opt(2014, 1, 15).unwrap());
    }

    #[test]
    fn test_parse_triple_int_timestamp_format() {
        // RE-Net format: 5 whitespace-separated columns, 4th is integer timestamp
        let line = "2\t8\t26\t0\t0";
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
        let subject: u32 = parts[0].parse().unwrap();
        let relation: u32 = parts[1].parse().unwrap();
        let object: u32 = parts[2].parse().unwrap();
        let ts_int: i64 = parts[3].parse().unwrap();
        let config = TimestampConfig::icews14();
        let date = config.to_date(ts_int);
        assert_eq!(subject, 2);
        assert_eq!(relation, 8);
        assert_eq!(object, 26);
        assert_eq!(date, NaiveDate::from_ymd_opt(2014, 1, 1).unwrap()); // ts=0 → base date
    }

    #[test]
    fn test_timestamp_config_offset() {
        let config = TimestampConfig::icews14();
        // Day 14 → 2014-01-15
        assert_eq!(
            config.to_date(14),
            NaiveDate::from_ymd_opt(2014, 1, 15).unwrap()
        );
        // Day 365 → 2015-01-01
        assert_eq!(
            config.to_date(365),
            NaiveDate::from_ymd_opt(2015, 1, 1).unwrap()
        );
    }
}
