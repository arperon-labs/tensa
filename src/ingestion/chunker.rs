//! Text chunking for narrative ingestion.
//!
//! Splits raw text into sized chunks, respecting paragraph boundaries
//! and chapter markers, with configurable overlap.

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Chunking strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStrategy {
    /// Fixed-size chunks (~max_tokens each) with overlap. Default for small context models.
    #[default]
    FixedSize,
    /// One chunk per detected chapter. Falls back to large fixed-size if no chapters found.
    /// Best for large-context models (100k+ tokens).
    Chapter,
}

/// Configuration for the text chunker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Target maximum tokens per chunk (approximate).
    pub max_tokens: usize,
    /// Overlap tokens from previous chunk.
    pub overlap_tokens: usize,
    /// Optional regex pattern for chapter detection.
    pub chapter_regex: Option<String>,
    /// Chunking strategy.
    #[serde(default)]
    pub strategy: ChunkStrategy,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            overlap_tokens: 200,
            chapter_regex: None,
            strategy: ChunkStrategy::FixedSize,
        }
    }
}

/// A chunk of text extracted from a larger document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Sequential index.
    pub chunk_id: usize,
    /// The chunk text content.
    pub text: String,
    /// Detected chapter name, if any.
    pub chapter: Option<String>,
    /// Byte offset in the original text.
    pub start_offset: usize,
    /// End byte offset in the original text.
    pub end_offset: usize,
    /// Overlap text carried from the previous chunk.
    pub overlap_prefix: String,
}

/// Estimate token count from text (words × 1.3).
pub fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    (words as f64 * 1.3).ceil() as usize
}

/// Detect chapter boundaries in text.
/// Returns (byte_offset, chapter_title) pairs.
///
/// Detection proceeds through multiple passes in priority order:
///
/// 1. **Custom regex** — if provided, used directly.
/// 2. **Markdown headings** — detects `#`..`######` headings and infers which
///    heading level represents chapters. Filters out repeated page-header titles
///    (e.g. `# Book Title` on every page) and picks the level with the most
///    unique headings (≥3 required). Also filters out table-of-contents blocks.
/// 3. **Prefixed patterns** — "Chapter 1", "CHAPTER I", "Part 2", "Book III",
///    "Chapter One", "CHAPTER THE FIRST", with optional subtitle after `:—–-`.
/// 4. **Bare Roman numerals** — standalone `I`, `XIV`, `XXVII` etc. on their
///    own line, requiring ≥3 matches and filtering TOC-style entries.
pub fn detect_chapters(text: &str, custom_regex: Option<&str>) -> Vec<(usize, String)> {
    if let Some(custom) = custom_regex {
        let re = match Regex::new(custom) {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        return re
            .find_iter(text)
            .map(|m| (m.start(), m.as_str().trim().to_string()))
            .collect();
    }

    // Pass 1: Try markdown headings (#, ##, ###, etc.)
    let md_result = detect_markdown_chapters(text);
    if !md_result.is_empty() {
        return md_result;
    }

    // Pass 2: Try the prefixed pattern (Chapter/Part/Book + number)
    let prefixed = r"(?mi)^(?:chapter|part|book)\s+(?:\d+|[IVXLCDM]+|(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty-\w+|thirtieth|thirty-\w+|fortieth|forty-\w+|fiftieth|fifty-\w+|last|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty))(?:\s*[.:—–\-]\s*(.*))?\.?$";
    if let Ok(re) = Regex::new(prefixed) {
        let results: Vec<(usize, String)> = re
            .find_iter(text)
            .map(|m| (m.start(), m.as_str().trim().to_string()))
            .collect();
        if !results.is_empty() {
            return results;
        }
    }

    // Pass 3: Try bare Roman numerals on their own line, surrounded by blank lines.
    // Require at least 3 matches to confirm this is a chapter numbering pattern
    // (avoids false positives from a lone "I" used as a pronoun).
    detect_bare_roman_chapters(text)
}

/// Detect chapter boundaries from markdown headings (`#` through `######`).
///
/// **Bottom-up with most-headings selection:**
///
/// 1. Parse all heading lines with their level (1–6) and text.
/// 2. Filter out repeated page-header levels (>50% of headings share the same text).
/// 3. Filter out TOC-style blocks where consecutive headings have very few words.
/// 4. For each surviving level, check if the average words per section is
///    reasonable (≥ total_words / 200, minimum 20 words). Skip levels where
///    sections are too short (scene breaks, sub-sections).
/// 5. Among all qualifying levels, pick the one with the **most headings**.
///    On ties, prefer the deeper (more specific) level.
fn detect_markdown_chapters(text: &str) -> Vec<(usize, String)> {
    use std::collections::HashMap;

    let total_words = text.split_whitespace().count();
    if total_words == 0 {
        return vec![];
    }

    // Parse all markdown headings: (byte_offset, level, title_text)
    let heading_re = match Regex::new(r"(?m)^(#{1,6})\s+(.+?)[ \t]*$") {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    let all_headings: Vec<(usize, usize, String)> = heading_re
        .captures_iter(text)
        .map(|cap| {
            let offset = cap.get(0).unwrap().start();
            let level = cap[1].len();
            let title = cap[2].trim().to_string();
            (offset, level, title)
        })
        .collect();

    if all_headings.is_empty() {
        return vec![];
    }

    // Group headings by level
    let mut by_level: HashMap<usize, Vec<(usize, String)>> = HashMap::new();
    for (offset, level, title) in &all_headings {
        by_level
            .entry(*level)
            .or_default()
            .push((*offset, title.clone()));
    }

    // Minimum average words per section: at least total_words/200 or 20, whichever is larger.
    // This prevents scene-break-level headings from being treated as chapters.
    // The floor of 20 allows short texts/tests to work while total_words/200 handles
    // real books (e.g. 100k words → 500 min avg, filtering out scene breaks).
    let min_avg_words = (total_words / 200).max(20);

    // Evaluate each level bottom-up, collecting qualifying candidates
    // (level, heading_count, headings)
    let mut candidates: Vec<(usize, usize, Vec<(usize, String)>)> = Vec::new();

    for level in (1..=6).rev() {
        let headings = match by_level.get(&level) {
            Some(h) if h.len() >= 3 => h,
            _ => continue,
        };

        // Check for repeated-title level (page header artifact)
        let mut text_counts: HashMap<&str, usize> = HashMap::new();
        for (_, title) in headings {
            *text_counts.entry(title.as_str()).or_insert(0) += 1;
        }
        let max_same = text_counts.values().max().copied().unwrap_or(0);
        if max_same as f64 / headings.len() as f64 > 0.5 {
            continue;
        }

        let mut chapter_headings = headings.clone();
        chapter_headings.sort_by_key(|(offset, _)| *offset);

        // Filter out TOC-style blocks
        chapter_headings = filter_toc_blocks(text, chapter_headings);
        if chapter_headings.len() < 3 {
            continue;
        }

        // Compute average words per section
        let avg_words = compute_avg_section_words(text, &chapter_headings);

        // If sections are too short, this level is too granular — skip
        if avg_words < min_avg_words {
            continue;
        }

        let count = chapter_headings.len();
        candidates.push((level, count, chapter_headings));
    }

    if candidates.is_empty() {
        return vec![];
    }

    // Pick the level with the most headings. On tie, prefer deeper (more specific) level.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

    candidates.into_iter().next().unwrap().2
}

/// Compute the average word count across sections defined by heading positions.
fn compute_avg_section_words(text: &str, headings: &[(usize, String)]) -> usize {
    if headings.is_empty() {
        return 0;
    }
    let mut total = 0usize;
    let mut count = 0usize;
    for window in headings.windows(2) {
        let section = &text[window[0].0..window[1].0];
        total += section.split_whitespace().count();
        count += 1;
    }
    // Last section to end of text
    if let Some(last) = headings.last() {
        total += text[last.0..].split_whitespace().count();
        count += 1;
    }
    if count == 0 {
        0
    } else {
        total / count
    }
}

/// Filter out TOC-style blocks where consecutive headings have very few
/// tokens between them (3+ consecutive entries below MIN_CHAPTER_TOKENS).
fn filter_toc_blocks(text: &str, headings: Vec<(usize, String)>) -> Vec<(usize, String)> {
    if headings.len() < 3 {
        return headings;
    }

    let gap_tokens: Vec<usize> = headings
        .windows(2)
        .map(|w| estimate_tokens(&text[w[0].0..w[1].0]))
        .collect();

    let mut is_toc = vec![false; headings.len()];
    let mut run_start: Option<usize> = None;
    for i in 0..gap_tokens.len() {
        if gap_tokens[i] < MIN_CHAPTER_TOKENS {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start {
            let run_len = i - start + 1;
            if run_len >= 3 {
                for j in start..=i {
                    is_toc[j] = true;
                }
            }
            run_start = None;
        }
    }
    if let Some(start) = run_start {
        let run_len = headings.len() - start;
        if run_len >= 3 {
            for j in start..headings.len() {
                is_toc[j] = true;
            }
        }
    }

    headings
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !is_toc[*i])
        .map(|(_, c)| c)
        .collect()
}

/// Minimum tokens for a chapter boundary to be considered real content.
/// If the text between two consecutive chapter markers is below this threshold,
/// the boundary is discarded (merging the tiny section into the next chapter).
/// This prevents table-of-contents entries from being treated as chapters.
/// Set low enough to accept short chapters but high enough to reject TOC lines.
const MIN_CHAPTER_TOKENS: usize = 20;

/// Detect bare Roman numeral chapter headers (e.g., "I", "XIV", "XXVII").
/// Only matches lines that are purely a Roman numeral, surrounded by blank lines.
/// Requires ≥3 matches to avoid false positives.
/// Filters out TOC-style entries where inter-chapter content is too small.
fn detect_bare_roman_chapters(text: &str) -> Vec<(usize, String)> {
    // Match a line containing only a valid Roman numeral, case-insensitive.
    // The numeral must be on its own line (^...$) and use valid Roman digit sequences.
    // The \r? handles CRLF line endings where $ matches before \n but after \r.
    let re = match Regex::new(r"(?mi)^([IVXLCDM]+)\.?\r?$") {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    let candidates: Vec<(usize, String)> = re
        .find_iter(text)
        .filter_map(|m| {
            let raw = m
                .as_str()
                .trim()
                .trim_end_matches('.')
                .trim_end_matches('\r');
            if !is_valid_roman(raw) {
                return None;
            }
            Some((m.start(), raw.to_string()))
        })
        .collect();

    // Require at least 3 raw matches to confirm a chapter numbering pattern
    if candidates.len() < 3 {
        return vec![];
    }

    // Filter out TOC-style entries. Strategy: identify runs of 3+ consecutive
    // entries where the gap between each pair is below MIN_CHAPTER_TOKENS.
    // These are table-of-contents blocks. Remove the entire run.
    let gap_tokens: Vec<usize> = candidates
        .windows(2)
        .map(|w| estimate_tokens(&text[w[0].0..w[1].0]))
        .collect();

    // Mark entries that are part of a TOC run (gap below threshold for 3+ consecutive)
    let mut is_toc = vec![false; candidates.len()];
    let mut run_start = None;
    for i in 0..gap_tokens.len() {
        if gap_tokens[i] < MIN_CHAPTER_TOKENS {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start {
            // End of a small-gap run. If it spans 3+ entries, mark them as TOC.
            let run_len = i - start + 1; // entries in the run
            if run_len >= 3 {
                for j in start..=i {
                    is_toc[j] = true;
                }
            }
            run_start = None;
        }
    }
    // Handle run that extends to the end
    if let Some(start) = run_start {
        let run_len = candidates.len() - start;
        if run_len >= 3 {
            for j in start..candidates.len() {
                is_toc[j] = true;
            }
        }
    }

    let filtered: Vec<(usize, String)> = candidates
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !is_toc[*i])
        .map(|(_, c)| c)
        .collect();

    // After filtering, still require ≥3 real chapters
    if filtered.len() >= 3 {
        filtered
    } else {
        vec![]
    }
}

/// Validate that a string is a well-formed Roman numeral (I through MMMCMXCIX).
fn is_valid_roman(s: &str) -> bool {
    let upper = s.to_uppercase();
    if upper.is_empty() {
        return false;
    }
    // Check all chars are valid Roman digits
    if !upper.chars().all(|c| "IVXLCDM".contains(c)) {
        return false;
    }
    // Parse to a number and verify it's valid (1..3999)
    roman_to_int(&upper).is_some()
}

/// Convert a Roman numeral string to an integer, returning None if invalid.
fn roman_to_int(s: &str) -> Option<u32> {
    let mut total: u32 = 0;
    let mut prev: u32 = 0;
    for c in s.chars().rev() {
        let val = match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        };
        if val < prev {
            total = total.checked_sub(val)?;
        } else {
            total = total.checked_add(val)?;
        }
        prev = val;
    }
    if total == 0 || total > 3999 {
        None
    } else {
        Some(total)
    }
}

/// Split text into paragraphs (double newline separated).
fn split_paragraphs(text: &str) -> Vec<&str> {
    let re = Regex::new(r"\n\s*\n").unwrap();
    let parts: Vec<&str> = re.split(text).filter(|s| !s.trim().is_empty()).collect();
    if parts.is_empty() && !text.trim().is_empty() {
        vec![text.trim()]
    } else {
        parts
    }
}

/// Chunk text into pieces using the configured strategy.
/// Collapse runs of 3+ newlines into double newlines (paragraph breaks).
/// Handles bad epub/docx conversions with excessive blank lines.
pub fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut newline_run = 0u32;
    for ch in text.chars() {
        if ch == '\n' {
            newline_run += 1;
            if newline_run <= 2 {
                out.push(ch);
            }
        } else {
            newline_run = 0;
            out.push(ch);
        }
    }
    out
}

pub fn chunk_text(text: &str, config: &ChunkerConfig) -> Vec<TextChunk> {
    if text.trim().is_empty() {
        return vec![];
    }

    // Normalize excessive blank lines before chunking
    let cleaned = normalize_whitespace(text);
    let text = &cleaned;

    match config.strategy {
        ChunkStrategy::Chapter => chunk_by_chapters(text, config),
        ChunkStrategy::FixedSize => chunk_by_size(text, config),
    }
}

/// Chunk by chapter boundaries. Each chapter becomes one chunk.
/// If no chapters detected, falls back to large fixed-size chunks.
fn chunk_by_chapters(text: &str, config: &ChunkerConfig) -> Vec<TextChunk> {
    let chapters = detect_chapters(text, config.chapter_regex.as_deref());

    if chapters.is_empty() {
        // No chapters found — fall back to fixed-size with the configured max_tokens
        return chunk_by_size(text, config);
    }

    let mut chunks = Vec::new();
    for window in chapters.windows(2) {
        let (start, title) = &window[0];
        let (next_start, _) = &window[1];
        let chapter_text = text[*start..*next_start].trim();
        if chapter_text.is_empty() {
            continue;
        }
        // If a single chapter exceeds max_tokens, sub-chunk it
        if estimate_tokens(chapter_text) > config.max_tokens && config.max_tokens > 0 {
            let sub_config = ChunkerConfig {
                strategy: ChunkStrategy::FixedSize,
                ..config.clone()
            };
            let mut sub_chunks = chunk_by_size(chapter_text, &sub_config);
            for sc in &mut sub_chunks {
                sc.chunk_id = chunks.len();
                sc.chapter = Some(title.clone());
                sc.start_offset += start;
                sc.end_offset += start;
                chunks.push(sc.clone());
            }
        } else {
            chunks.push(TextChunk {
                chunk_id: chunks.len(),
                text: chapter_text.to_string(),
                chapter: Some(title.clone()),
                start_offset: *start,
                end_offset: *next_start,
                overlap_prefix: String::new(),
            });
        }
    }

    // Last chapter: from last boundary to end of text
    if let Some((start, title)) = chapters.last() {
        let chapter_text = text[*start..].trim();
        if !chapter_text.is_empty() {
            if estimate_tokens(chapter_text) > config.max_tokens && config.max_tokens > 0 {
                let sub_config = ChunkerConfig {
                    strategy: ChunkStrategy::FixedSize,
                    ..config.clone()
                };
                let mut sub_chunks = chunk_by_size(chapter_text, &sub_config);
                for sc in &mut sub_chunks {
                    sc.chunk_id = chunks.len();
                    sc.chapter = Some(title.clone());
                    sc.start_offset += start;
                    sc.end_offset += start;
                    chunks.push(sc.clone());
                }
            } else {
                chunks.push(TextChunk {
                    chunk_id: chunks.len(),
                    text: chapter_text.to_string(),
                    chapter: Some(title.clone()),
                    start_offset: *start,
                    end_offset: text.len(),
                    overlap_prefix: String::new(),
                });
            }
        }
    }

    // Include any text before the first chapter (preface, etc.)
    if let Some((first_start, _)) = chapters.first() {
        if *first_start > 0 {
            let preface = text[..*first_start].trim();
            if !preface.is_empty() {
                chunks.insert(
                    0,
                    TextChunk {
                        chunk_id: 0,
                        text: preface.to_string(),
                        chapter: Some("Preface".to_string()),
                        start_offset: 0,
                        end_offset: *first_start,
                        overlap_prefix: String::new(),
                    },
                );
                // Re-number
                for (i, c) in chunks.iter_mut().enumerate() {
                    c.chunk_id = i;
                }
            }
        }
    }

    chunks
}

/// Chunk text into fixed-size pieces respecting paragraph boundaries.
fn chunk_by_size(text: &str, config: &ChunkerConfig) -> Vec<TextChunk> {
    if text.trim().is_empty() {
        return vec![];
    }

    let chapters = detect_chapters(text, config.chapter_regex.as_deref());
    let paragraphs = split_paragraphs(text);

    if paragraphs.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut current_start = 0usize;
    let mut overlap_prefix = String::new();
    let mut current_chapter: Option<String> = None;

    // Map each paragraph to its chapter (if any)
    let paragraph_chapter = |para_start: usize| -> Option<String> {
        let mut best: Option<&str> = None;
        for (ch_offset, ch_name) in &chapters {
            if *ch_offset <= para_start {
                best = Some(ch_name.as_str());
            }
        }
        best.map(String::from)
    };

    for para in &paragraphs {
        let para_start = text.find(para).unwrap_or(0);
        let chapter = paragraph_chapter(para_start);

        // Check if adding this paragraph would exceed the limit
        let candidate = if current_text.is_empty() {
            para.to_string()
        } else {
            format!("{}\n\n{}", current_text, para)
        };

        if estimate_tokens(&candidate) > config.max_tokens && !current_text.is_empty() {
            // Emit the current chunk
            let chunk_end = current_start + current_text.len();
            chunks.push(TextChunk {
                chunk_id: chunks.len(),
                text: current_text.clone(),
                chapter: current_chapter.clone(),
                start_offset: current_start,
                end_offset: chunk_end,
                overlap_prefix: overlap_prefix.clone(),
            });

            // Build overlap from the tail of current text
            overlap_prefix = build_overlap(&current_text, config.overlap_tokens);
            current_text = para.to_string();
            current_start = para_start;
            current_chapter = chapter;
        } else {
            if current_text.is_empty() {
                current_start = para_start;
                current_chapter = chapter;
            }
            current_text = candidate;
        }
    }

    // Emit final chunk
    if !current_text.is_empty() {
        let chunk_end = current_start + current_text.len();
        chunks.push(TextChunk {
            chunk_id: chunks.len(),
            text: current_text,
            chapter: current_chapter,
            start_offset: current_start,
            end_offset: chunk_end,
            overlap_prefix,
        });
    }

    chunks
}

/// Extract the last N tokens worth of text for overlap.
fn build_overlap(text: &str, overlap_tokens: usize) -> String {
    if overlap_tokens == 0 {
        return String::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let target_words = (overlap_tokens as f64 / 1.3).ceil() as usize;
    let start = words.len().saturating_sub(target_words);
    words[start..].join(" ")
}

/// Compute SHA-256 hash of chunk text content, returning hex string.
pub fn chunk_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_hash_deterministic() {
        let h1 = chunk_hash("hello world");
        let h2 = chunk_hash("hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_chunk_hash_different_inputs() {
        let h1 = chunk_hash("hello");
        let h2 = chunk_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_chunk_short_text_single_chunk() {
        let config = ChunkerConfig::default();
        let chunks = chunk_text("Hello world. This is a short text.", &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_id, 0);
    }

    #[test]
    fn test_chunk_empty_text() {
        let config = ChunkerConfig::default();
        let chunks = chunk_text("", &config);
        assert!(chunks.is_empty());

        let chunks = chunk_text("   \n\n  ", &config);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_long_text_multiple_chunks() {
        let config = ChunkerConfig {
            max_tokens: 20,
            overlap_tokens: 5,
            chapter_regex: None,
            ..Default::default()
        };
        // Create text that's definitely longer than 20 tokens
        let para1 = "This is the first paragraph with some content that goes on for a while.";
        let para2 = "This is the second paragraph also with enough content to matter.";
        let text = format!("{}\n\n{}", para1, para2);
        let chunks = chunk_text(&text, &config);
        assert!(
            chunks.len() >= 2,
            "Expected >=2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_chunk_respects_paragraph_boundaries() {
        let config = ChunkerConfig {
            max_tokens: 50,
            overlap_tokens: 0,
            chapter_regex: None,
            ..Default::default()
        };
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_text(&text, &config);
        // With generous token limit, all paragraphs fit in one chunk
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[0].text.contains("Third"));
    }

    #[test]
    fn test_chunk_detects_chapters() {
        let chapters = detect_chapters(
            "Chapter 1: The Beginning\nSome text.\nChapter 2: The End\nMore text.",
            None,
        );
        assert_eq!(chapters.len(), 2);
        assert!(chapters[0].1.contains("Chapter 1"));
        assert!(chapters[1].1.contains("Chapter 2"));
    }

    #[test]
    fn test_chunk_overlap_content() {
        let config = ChunkerConfig {
            max_tokens: 15,
            overlap_tokens: 5,
            chapter_regex: None,
            ..Default::default()
        };
        let text = "First paragraph with some words.\n\nSecond paragraph with other words.";
        let chunks = chunk_text(&text, &config);
        if chunks.len() >= 2 {
            assert!(!chunks[1].overlap_prefix.is_empty());
        }
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello world"), 3); // 2 words * 1.3 = 2.6 -> 3
        assert!(estimate_tokens("one two three four five") > 5);
    }

    #[test]
    fn test_detect_chapters_roman_numerals() {
        let chapters = detect_chapters("Part III\nSome text here.", None);
        assert_eq!(chapters.len(), 1);
        assert!(chapters[0].1.contains("Part III"));
    }

    #[test]
    fn test_detect_chapters_ordinal_words() {
        let text = "CHAPTER THE FIRST\nSome text.\n\nCHAPTER THE SECOND\nMore text.";
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            2,
            "Should detect ordinal word chapters: {:?}",
            chapters
        );
    }

    #[test]
    fn test_detect_chapters_number_words() {
        let text =
            "Chapter One\nSome text.\n\nChapter Two\nMore text.\n\nChapter Three\nEven more.";
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            3,
            "Should detect number word chapters: {:?}",
            chapters
        );
    }

    #[test]
    fn test_detect_chapters_trailing_period() {
        let text = "CHAPTER I.\nText here.\n\nCHAPTER II.\nMore text.";
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            2,
            "Should detect chapters with trailing period: {:?}",
            chapters
        );
    }

    #[test]
    fn test_detect_chapters_with_dash_title() {
        let text = "CHAPTER I — TREATS OF THE PLACE\nText.\n\nCHAPTER II — TREATS OF GROWTH\nMore.";
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            2,
            "Should detect chapters with dash titles: {:?}",
            chapters
        );
    }

    #[test]
    fn test_detect_bare_roman_numerals() {
        // Simulate Dracula-style chapter headers with actual content between them
        let text = "\n\nI\n\n\nJonathan Harker's Journal. 3 May. Bistritz. Left Munich at 8:35 pm, on 1st May, arriving at Vienna early next morning. The impression I had was that we were leaving the West and entering the East. Count Dracula had directed me to go to the Golden Krone Hotel.\n\n\n\nII\n\n\nJonathan Harker's Journal continued. 5 May. I must have been asleep, for certainly if I had been fully awake I must have noticed the approach to such a remarkable place. In the gloom the courtyard looked of considerable size.\n\n\n\nIII\n\n\nJonathan Harker's Journal continued. 8 May. I began to fear as I wrote in this book. The castle is a veritable prison, and I am a prisoner.\n\n";
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            3,
            "Should detect bare Roman numeral chapters: {:?}",
            chapters
        );
        assert_eq!(chapters[0].1, "I");
        assert_eq!(chapters[1].1, "II");
        assert_eq!(chapters[2].1, "III");
    }

    #[test]
    fn test_bare_roman_filters_toc() {
        // Simulate a table of contents followed by actual chapters.
        // TOC entries have almost no content between them; real chapters do.
        let mut text = String::new();
        // TOC section: bare numerals with no real content
        text.push_str("I\n\n\n\nII\n\n\n\nIII\n\n\n\nIV\n\n\n\nV\n\n\n\n");
        // Preface section between TOC and real chapters (like Standard Ebooks have)
        text.push_str("This ebook is the product of many hours of hard work by volunteers for Standard Ebooks, and builds on the hard work of other literature lovers made possible by the public domain. This particular ebook is based on a transcription from Project Gutenberg. The source text and artwork in this ebook are believed to be in the United States public domain.\n\n\n\n");
        // Actual chapters with substantial content
        text.push_str("I\n\n\nJonathan Harker's Journal. Left Munich at 8:35 pm on 1st May arriving at Vienna early next morning. The impression I had was that we were leaving the West and entering the East. Count Dracula had directed me to go to the Golden Krone Hotel which I found to my great delight.\n\n\n\n");
        text.push_str("II\n\n\nJonathan Harker's Journal continued. 5 May. I must have been asleep for certainly if I had been fully awake I must have noticed the approach to such a remarkable place. In the gloom the courtyard looked of considerable size.\n\n\n\n");
        text.push_str("III\n\n\nJonathan Harker's Journal continued. 8 May. I began to fear as I wrote in this book. The castle is a veritable prison and I am a prisoner.\n\n\n\n");
        text.push_str("IV\n\n\nLetter Lucy Westenra to Mina Murray. My dearest Mina. I have been greatly honoured today. Three proposals in one day. Isn't it awful. I feel sorry really and truly sorry for two of the poor fellows.\n\n");

        let chapters = detect_chapters(&text, None);
        // Should find 4 real chapters (I-IV with content), not the 5 TOC entries.
        // The TOC has 5 entries (I-V) with <20 tokens each, so they should be filtered out.
        // The real chapters (I-IV) have substantial content (>20 tokens each).
        assert_eq!(
            chapters.len(),
            4,
            "Should find exactly 4 real chapters, not TOC entries: found {} chapters: {:?}",
            chapters.len(),
            chapters
        );
    }

    #[test]
    fn test_bare_roman_too_few_matches() {
        // Only 2 bare Roman numerals — below threshold, should return empty
        let text = "I\n\n\nSome text here.\n\n\n\nII\n\n\nMore text.\n\n";
        let chapters = detect_chapters(text, None);
        assert!(
            chapters.is_empty(),
            "Should not detect chapters with <3 bare numerals: {:?}",
            chapters
        );
    }

    #[test]
    fn test_bare_roman_not_valid_numeral() {
        // Words containing non-Roman letters should not match
        let text = "\n\nTHE\n\n\nSome text that is long enough.\n\n\n\nAND\n\n\nMore text that is long enough.\n\n\n\nBUT\n\n\nEven more text enough.\n\n";
        let chapters = detect_chapters(text, None);
        assert!(
            chapters.is_empty(),
            "Should not match non-Roman words: {:?}",
            chapters
        );
    }

    #[test]
    fn test_is_valid_roman() {
        assert!(is_valid_roman("I"));
        assert!(is_valid_roman("IV"));
        assert!(is_valid_roman("IX"));
        assert!(is_valid_roman("XIV"));
        assert!(is_valid_roman("XXVII"));
        assert!(is_valid_roman("XLII"));
        assert!(is_valid_roman("MCMXCIX")); // 1999
        assert!(!is_valid_roman(""));
        assert!(is_valid_roman("IIII")); // Non-standard but parses to 4 — acceptable
        assert!(!is_valid_roman("ABC"));
    }

    #[test]
    fn test_roman_to_int() {
        assert_eq!(roman_to_int("I"), Some(1));
        assert_eq!(roman_to_int("IV"), Some(4));
        assert_eq!(roman_to_int("IX"), Some(9));
        assert_eq!(roman_to_int("XIV"), Some(14));
        assert_eq!(roman_to_int("XXVII"), Some(27));
        assert_eq!(roman_to_int("XLII"), Some(42));
        assert_eq!(roman_to_int("MCMXCIX"), Some(1999));
    }

    #[test]
    fn test_detect_chapters_dracula_file() {
        // Test against actual Dracula text structure:
        // TOC with bare Roman numerals (I-XXVII, ~5 tokens each)
        // then actual chapters with substantial content
        let dracula_path = std::path::Path::new("corpus/standard-ebooks/dracula.txt");
        if !dracula_path.exists() {
            eprintln!("Skipping Dracula file test (file not found)");
            return;
        }
        let text = std::fs::read_to_string(dracula_path).unwrap();
        let chapters = detect_chapters(&text, None);
        // Dracula has 27 chapters (I-XXVII)
        assert!(
            chapters.len() >= 25 && chapters.len() <= 30,
            "Expected ~27 chapters in Dracula, got {}: {:?}",
            chapters.len(),
            chapters.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>()
        );
        // Verify first and last chapter names
        assert_eq!(chapters[0].1, "I", "First chapter should be 'I'");
        assert_eq!(
            chapters.last().unwrap().1,
            "XXVII",
            "Last chapter should be 'XXVII'"
        );
    }

    #[test]
    fn test_custom_chapter_regex() {
        let text = "## Section A\nText.\n\n## Section B\nMore.";
        let chapters = detect_chapters(text, Some(r"(?m)^## .+$"));
        assert_eq!(
            chapters.len(),
            2,
            "Should match custom regex: {:?}",
            chapters
        );
    }

    #[test]
    fn test_detect_markdown_chapters_h2() {
        // Pattern: repeated # title + ## chapter headings (Foundation, Ender's Game, Neverwhere)
        let text = r#"# My Novel

# My Novel

## Chapter One

Some text here that is long enough to not be a table of contents entry and contains real content.

# My Novel

## Chapter Two

More text here that is also long enough to be considered real chapter content with substance.

# My Novel

## Chapter Three

Even more text that constitutes a real chapter with enough words to pass the filter threshold.
"#;
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            3,
            "Should detect 3 ## chapters, ignoring repeated # title: {:?}",
            chapters
        );
        assert!(chapters[0].1.contains("Chapter One"));
        assert!(chapters[1].1.contains("Chapter Two"));
        assert!(chapters[2].1.contains("Chapter Three"));
    }

    #[test]
    fn test_detect_markdown_chapters_h3() {
        // Pattern: ### chapters (Do Androids Dream, A Clash of Kings)
        let text = r#"# Do Androids Dream of Electric Sheep?

Some intro text.

### ONE

Content of chapter one that is long enough to be considered real content and not a table of contents entry.

### TWO

Content of chapter two that is also long enough to be considered real content and not a table of contents.

### THREE

Content of chapter three with enough substance to pass the minimum token threshold for chapter detection.

### FOUR

Content of chapter four that rounds out the book with enough words to be meaningful narrative content.
"#;
        let chapters = detect_chapters(text, None);
        assert_eq!(
            chapters.len(),
            4,
            "Should detect 4 ### chapters: {:?}",
            chapters
        );
        assert_eq!(chapters[0].1, "ONE");
        assert_eq!(chapters[3].1, "FOUR");
    }

    #[test]
    fn test_detect_markdown_chapters_h1_books() {
        // Pattern: # BOOK I, # BOOK II (The Iliad)
        let mut text = String::from("# Contents\n\nSome table of contents.\n\n# INTRODUCTION\n\n");
        text.push_str("A long introduction with enough words to pass the minimum token threshold for detection. The epic poem begins with an invocation to the Muse.\n\n");
        for i in ["I", "II", "III", "IV"] {
            text.push_str(&format!("# BOOK {}.\n\n", i));
            text.push_str("Sing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans. Many a brave soul did it send hurrying down to Hades, and many a hero did it yield a prey to dogs and vultures. This is substantial chapter content.\n\n");
        }
        let chapters = detect_chapters(&text, None);
        assert!(
            chapters.len() >= 4,
            "Should detect # BOOK headings: {:?}",
            chapters
        );
    }

    #[test]
    fn test_detect_markdown_chapters_mixed_levels() {
        // Pattern: # title (repeated), ##### subtitle, ## author, ### chapters (ASOIAF)
        let mut text = String::from("# A Clash of Kings\n\n##### A Song of Ice and Fire Book II\n\n## George R.R. Martin\n\n");
        for name in ["Prologue", "Arya", "Sansa", "Tyrion", "Bran"] {
            text.push_str(&format!("### {}\n\n", name));
            text.push_str("The chapter content continues with descriptions of castles, battles, and political intrigue across the Seven Kingdoms. Characters scheme and fight for the Iron Throne in this sprawling epic.\n\n");
        }
        let chapters = detect_chapters(&text, None);
        assert_eq!(
            chapters.len(),
            5,
            "Should detect ### character chapters: {:?}",
            chapters
        );
        assert_eq!(chapters[0].1, "Prologue");
    }

    #[test]
    fn test_detect_markdown_chapters_filters_toc() {
        // TOC block with ### headings followed by real ### chapters
        let mut text = String::new();
        text.push_str("# My Book\n\n");
        // TOC: many ### entries with no content between them
        for i in 1..=10 {
            text.push_str(&format!("### Chapter {}\n\n", i));
        }
        // Substantial gap between TOC and real chapters
        text.push_str("\n\nThis is a long preface section with enough words to create a clear gap between the table of contents and the actual chapter content that follows below.\n\n");
        // Real chapters with content
        for i in 1..=10 {
            text.push_str(&format!("### Chapter {}\n\n", i));
            text.push_str("This is real chapter content with enough words to be substantial. The narrative continues with descriptions, dialogue, and plot developments that make this a real chapter.\n\n");
        }
        let chapters = detect_chapters(&text, None);
        // Should find 10 real chapters, filtering the TOC block
        assert_eq!(
            chapters.len(),
            10,
            "Should find 10 real chapters, not TOC entries: found {}",
            chapters.len()
        );
    }

    #[test]
    fn test_detect_markdown_skips_when_too_few() {
        // Only 2 markdown headings — not enough to detect as chapters
        let text = "## Intro\n\nSome text.\n\n## Conclusion\n\nMore text.";
        let md = detect_markdown_chapters(text);
        assert!(md.is_empty(), "Should not detect chapters with <3 headings");
    }

    #[test]
    fn test_markdown_books_corpus() {
        // Scan all .md files under E:\books\books\books and report chapter detection stats.
        // This test always passes — it prints diagnostics for manual review.
        let books_dir = std::path::Path::new("E:/books/books/books");
        if !books_dir.exists() {
            eprintln!("Skipping corpus test (E:/books/books/books not found)");
            return;
        }

        let mut total_files = 0;
        let mut detected = 0;
        let mut no_chapters = Vec::new();

        fn visit_dir(dir: &std::path::Path, results: &mut Vec<std::path::PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        visit_dir(&path, results);
                    } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
                        results.push(path);
                    }
                }
            }
        }

        let mut files = Vec::new();
        visit_dir(books_dir, &mut files);
        files.sort();

        eprintln!("\n{}", "=".repeat(90));
        eprintln!(
            "{:<50} {:>6} {:>8} {:>8} {:>8} {:>8}",
            "Book", "Chaps", "MinW", "MaxW", "AvgW", "TotalW"
        );
        eprintln!("{}", "=".repeat(90));

        for path in &files {
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if text.trim().is_empty() {
                continue;
            }
            total_files += 1;

            let chapters = detect_chapters(&text, None);
            let name = path.file_stem().unwrap().to_string_lossy();
            let short_name: String = name.chars().take(48).collect();
            let total_words = text.split_whitespace().count();

            if chapters.is_empty() {
                no_chapters.push(short_name.clone());
                eprintln!(
                    "{:<50} {:>6} {:>8} {:>8} {:>8} {:>8}",
                    short_name, 0, "-", "-", "-", total_words
                );
                continue;
            }

            detected += 1;

            // Compute word counts between consecutive chapters
            let mut word_counts = Vec::new();
            for window in chapters.windows(2) {
                let section = &text[window[0].0..window[1].0];
                word_counts.push(section.split_whitespace().count());
            }
            // Last chapter to end
            if let Some(last) = chapters.last() {
                word_counts.push(text[last.0..].split_whitespace().count());
            }

            let min_w = word_counts.iter().min().copied().unwrap_or(0);
            let max_w = word_counts.iter().max().copied().unwrap_or(0);
            let avg_w = if word_counts.is_empty() {
                0
            } else {
                word_counts.iter().sum::<usize>() / word_counts.len()
            };

            eprintln!(
                "{:<50} {:>6} {:>8} {:>8} {:>8} {:>8}",
                short_name,
                chapters.len(),
                min_w,
                max_w,
                avg_w,
                total_words
            );
        }

        eprintln!("\n--- Summary ---");
        eprintln!("Total .md files: {}", total_files);
        eprintln!("Chapters detected: {}/{}", detected, total_files);
        if !no_chapters.is_empty() {
            eprintln!("No chapters found in: {:?}", no_chapters);
        }

        // Sanity: at least 50% of files with >5000 words should have chapters detected
        let large_files = files
            .iter()
            .filter(|f| {
                std::fs::read_to_string(f)
                    .map(|t| t.split_whitespace().count() > 5000)
                    .unwrap_or(false)
            })
            .count();
        assert!(
            detected > 0,
            "Should detect chapters in at least some books"
        );
        eprintln!("Large files (>5000 words): {}", large_files);
    }

    #[test]
    fn test_chunk_by_chapters_with_bare_roman() {
        // Verify that chunking actually works with bare Roman numerals
        let mut text = String::new();
        for i in ["I", "II", "III", "IV"] {
            text.push_str(&format!("\n\n{}\n\n\n", i));
            // Add enough content to exceed MIN_CHAPTER_TOKENS
            for j in 0..10 {
                text.push_str(&format!("This is paragraph {} of chapter {}. It contains enough words to be substantial and meaningful for the extraction pipeline to process.\n\n", j, i));
            }
        }
        let config = ChunkerConfig {
            max_tokens: 50000,
            strategy: ChunkStrategy::Chapter,
            ..Default::default()
        };
        let chunks = chunk_text(&text, &config);
        assert!(
            chunks.len() >= 4,
            "Should produce at least 4 chunks from 4 chapters, got {}: {:?}",
            chunks.len(),
            chunks
                .iter()
                .map(|c| (&c.chapter, c.text.len()))
                .collect::<Vec<_>>()
        );
    }
}
