//! GDELT temporal knowledge graph dataset loader.
//!
//! Same format as ICEWS: whitespace-separated `subject relation object timestamp_int [extra]`.
//! Typically a larger dataset (~500K triples, benchmarks use subsets).

use super::{DatasetLoader, Split};
use crate::tensa_bench::datasets::icews::{TemporalTriple, TimestampConfig};
use std::path::Path;

/// GDELT dataset loader. Reuses the ICEWS triple format.
pub struct Gdelt;

impl DatasetLoader for Gdelt {
    type Item = TemporalTriple;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        let dataset_dir = data_dir.join(Self::dir_name());
        let filename = format!("{}.txt", split.filename_suffix());
        let path = dataset_dir.join(&filename);

        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let ts_config = TimestampConfig::gdelt();
        let mut triples = Vec::new();

        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                continue;
            }
            if let (Ok(s), Ok(r), Ok(o)) = (
                parts[0].parse::<u32>(),
                parts[1].parse::<u32>(),
                parts[2].parse::<u32>(),
            ) {
                let timestamp =
                    if let Ok(date) = chrono::NaiveDate::parse_from_str(parts[3], "%Y-%m-%d") {
                        date
                    } else if let Ok(ts_int) = parts[3].parse::<i64>() {
                        ts_config.to_date(ts_int)
                    } else {
                        return Err(format!(
                            "{}:{}: bad timestamp '{}'",
                            filename,
                            line_no + 1,
                            parts[3]
                        ));
                    };
                triples.push(TemporalTriple {
                    subject: s,
                    relation: r,
                    object: o,
                    timestamp,
                });
            }
        }

        Ok(triples)
    }

    fn dir_name() -> &'static str {
        "gdelt"
    }
}
