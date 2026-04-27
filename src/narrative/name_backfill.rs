//! One-shot backfill for empty `Situation.name` fields.
//!
//! Several creation paths (legacy `POST /situations` bodies, chapter-prose
//! generators that populate `raw_content` but not `name`, pre-v0.73.x
//! ingestion) leave situations with `name = None` even when the first line
//! of prose is obviously a chapter title like
//! `"Chapter 1 — The Inventory. Monday..."`. Studio renders these as
//! `(untitled)` in every outline view.
//!
//! [`backfill_names_from_content`] walks the narrative's situations and, for
//! each one with a missing name, derives a short title from the first line
//! of `raw_content` using [`derive_title_from_content`]. Idempotent: named
//! situations are left alone.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{ContentType, Situation};

/// Upper bound on synthesised titles. Keeps the outline panel legible.
const MAX_TITLE_LEN: usize = 80;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameBackfillReport {
    pub narrative_id: String,
    pub situations_total: usize,
    /// Situations that received a new title.
    pub names_set: usize,
    /// Situations that already had a non-empty name.
    pub skipped_already_named: usize,
    /// Situations we couldn't derive a title for (no prose content).
    pub skipped_no_content: usize,
}

/// Walk all situations for `narrative_id` and populate empty names from the
/// first line of prose content. Uses `update_situation_no_snapshot` so the
/// state-version log doesn't balloon with one entry per scene for a purely
/// derived cosmetic change.
pub fn backfill_names_from_content(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<NameBackfillReport> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let mut report = NameBackfillReport {
        narrative_id: narrative_id.to_string(),
        situations_total: situations.len(),
        names_set: 0,
        skipped_already_named: 0,
        skipped_no_content: 0,
    };

    for sit in &situations {
        if sit.name.as_deref().is_some_and(|n| !n.trim().is_empty()) {
            report.skipped_already_named += 1;
            continue;
        }
        let Some(title) = derive_title_from_content(sit) else {
            report.skipped_no_content += 1;
            continue;
        };
        hg.update_situation(&sit.id, |s| {
            s.name = Some(title);
        })?;
        report.names_set += 1;
    }
    Ok(report)
}

/// Heuristic: the first "title-shaped" line of prose.
///
/// Strategy:
/// - Only look at `Text` / `Dialogue` blocks.
/// - Split the first block on newline; keep the first non-empty line.
/// - Drop a leading `"Chapter N — "`, `"Chapter N: "`, or `"Chapter N. "`
///   prefix if one is present, since the `narrative_level` + outline position
///   already convey chapter numbers.
/// - Cut at the first sentence terminator (`. ` / `!` / `?` / `—` / `:`).
/// - Truncate to [`MAX_TITLE_LEN`] chars on a word boundary.
pub fn derive_title_from_content(sit: &Situation) -> Option<String> {
    let text = sit.raw_content.iter().find_map(|b| match b.content_type {
        ContentType::Text | ContentType::Dialogue => Some(b.content.as_str()),
        _ => None,
    })?;
    let first_line = text.lines().find(|l| !l.trim().is_empty())?.trim();
    let stripped = strip_chapter_prefix(first_line);
    let clipped = clip_to_sentence(stripped);
    let truncated = truncate_to_word_boundary(clipped, MAX_TITLE_LEN);
    let trimmed = truncated
        .trim()
        .trim_end_matches(&['.', ',', ':', ';', '-', '—'][..]);
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn strip_chapter_prefix(line: &str) -> &str {
    // Accept `Chapter 1 — X`, `Chapter 1: X`, `Chapter 1. X`, `Chapter 1 X`.
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("chapter ") {
        return line;
    }
    let rest = &line["chapter ".len()..];
    // Consume the chapter number (digits and/or roman-ish chars).
    let num_end = rest
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || c.is_alphabetic())
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    if num_end == 0 {
        return line;
    }
    let tail = rest[num_end..].trim_start();
    for sep in ["—", "-", ":", "."] {
        if let Some(after) = tail.strip_prefix(sep) {
            return after.trim_start();
        }
    }
    tail
}

fn clip_to_sentence(s: &str) -> &str {
    // Break on the first sentence terminator. ". ", "! ", "? " advance the
    // cut past the punctuation (keeping it out of the title). "\n" and em-dash
    // cut at the marker so the break stays visible.
    let mut best: Option<usize> = None;
    let candidates = [(". ", 2), ("! ", 2), ("? ", 2), ("\n", 0), ("\u{2014}", 0)];
    for (pat, offset) in candidates {
        if let Some(idx) = s.find(pat) {
            let candidate = idx + offset;
            if best.map_or(true, |b| candidate < b) {
                best = Some(candidate);
            }
        }
    }
    match best {
        Some(idx) => &s[..idx],
        None => s,
    }
}

fn truncate_to_word_boundary(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let mut boundary = 0usize;
    let mut chars = 0usize;
    for (i, c) in s.char_indices() {
        if chars >= max_chars {
            break;
        }
        if c.is_whitespace() {
            boundary = i;
        }
        chars += 1;
        if chars == max_chars {
            // Rewind to the last whitespace if one existed; otherwise cut mid-word.
            if boundary > 0 {
                return &s[..boundary];
            }
            return &s[..i + c.len_utf8()];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ContentType, ExtractionMethod, MaturityLevel, NarrativeLevel,
        Situation, TimeGranularity,
    };
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    fn make_sit(nid: &str, name: Option<&str>, content: &str) -> Situation {
        let now = Utc::now();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: name.map(|s| s.to_string()),
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock {
                content_type: ContentType::Text,
                content: content.into(),
                source: None,
            }],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[test]
    fn derives_title_from_chapter_em_dash_prefix() {
        let s = make_sit(
            "n",
            None,
            "Chapter 1 — The Inventory. Monday, mid-July. Fergus opens...",
        );
        assert_eq!(
            derive_title_from_content(&s).as_deref(),
            Some("The Inventory")
        );
    }

    #[test]
    fn derives_title_from_chapter_colon_prefix() {
        let s = make_sit("n", None, "Chapter 2: The Regulars. Tuesday. Ewan appears.");
        assert_eq!(
            derive_title_from_content(&s).as_deref(),
            Some("The Regulars")
        );
    }

    #[test]
    fn derives_title_from_bare_first_sentence() {
        let s = make_sit("n", None, "A grey morning on the esplanade. She hesitated.");
        assert_eq!(
            derive_title_from_content(&s).as_deref(),
            Some("A grey morning on the esplanade"),
        );
    }

    #[test]
    fn truncates_long_first_sentence_on_word_boundary() {
        let long =
            "A long opening paragraph without punctuation that just keeps rolling past eighty characters for sure";
        let s = make_sit("n", None, long);
        let title = derive_title_from_content(&s).unwrap();
        assert!(title.len() <= MAX_TITLE_LEN);
        assert!(!title.ends_with(' '));
    }

    #[test]
    fn returns_none_when_no_prose_content() {
        let mut s = make_sit("n", None, "irrelevant");
        s.raw_content = vec![];
        assert!(derive_title_from_content(&s).is_none());
    }

    #[test]
    fn backfill_skips_already_named_and_sets_rest() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        hg.create_situation(make_sit("n1", None, "Chapter 1 — The Inventory. Monday."))
            .unwrap();
        hg.create_situation(make_sit("n1", Some("Existing"), "Chapter 2 — Later."))
            .unwrap();
        hg.create_situation(make_sit("n1", None, "A quiet road."))
            .unwrap();

        let report = backfill_names_from_content(&hg, "n1").unwrap();
        assert_eq!(report.situations_total, 3);
        assert_eq!(report.names_set, 2);
        assert_eq!(report.skipped_already_named, 1);
        assert_eq!(report.skipped_no_content, 0);

        let all = hg.list_situations_by_narrative("n1").unwrap();
        for s in &all {
            assert!(s.name.as_deref().is_some_and(|n| !n.is_empty()));
        }
    }

    #[test]
    fn backfill_is_idempotent() {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        hg.create_situation(make_sit("n1", None, "Chapter 1 — Opening."))
            .unwrap();
        let first = backfill_names_from_content(&hg, "n1").unwrap();
        let second = backfill_names_from_content(&hg, "n1").unwrap();
        assert_eq!(first.names_set, 1);
        assert_eq!(second.names_set, 0);
        assert_eq!(second.skipped_already_named, 1);
    }
}
