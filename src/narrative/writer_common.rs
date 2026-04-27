//! Shared primitives for the writer-workflow modules (W1 generation, W2 edit,
//! W3 workshop). Everything here is prompt-assembly / cost-heuristic /
//! block-serialization boilerplate that would otherwise triplicate across the
//! three modules.
//!
//! Kept deliberately small: any time a second module needs the same helper
//! with even slightly different semantics, that's the sign to widen the
//! signature here instead of forking a copy.

use crate::types::{ContentBlock, ContentType, NarrativeSnapshot};

/// chars/4 is the widely-used rule-of-thumb for English LLM token count.
/// Good enough for a UI-visible estimate; the real count lives on the
/// provider side (and will be written to the cost ledger once W5 lands).
pub(crate) fn approx_tokens(s: &str) -> u32 {
    (s.len() as u32 + 3) / 4
}

/// Total word count across a slice of content blocks.
pub(crate) fn count_words_blocks(blocks: &[ContentBlock]) -> usize {
    blocks
        .iter()
        .flat_map(|b| b.content.split_whitespace())
        .count()
}

/// Split text on sentence terminators (`.`, `!`, `?`, `\n`), preserving each
/// delimiter at the end of its sentence. UTF-8 safe — returns empty slices only
/// if the input is empty. Used by `writer::factcheck` (claim extraction) and
/// `writer::cited_generation` (span grouping for citation markers).
pub(crate) fn split_sentences(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut last = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if matches!(b, b'.' | b'!' | b'?' | b'\n') {
            let end = (i + 1).min(bytes.len());
            if end > last {
                if let Ok(s) = std::str::from_utf8(&bytes[last..end]) {
                    out.push(s);
                }
                last = end;
            }
        }
    }
    if last < bytes.len() {
        if let Ok(s) = std::str::from_utf8(&bytes[last..]) {
            out.push(s);
        }
    }
    out
}

/// Serialize a situation's content blocks as a tagged plaintext dump suitable
/// for LLM prompts (`[TEXT]`, `[DIALOGUE]`, `[OBSERVATION]`, `[DOCUMENT]`,
/// `[MEDIA]`). Distinct from `export::manuscript` which emits Markdown prose.
pub(crate) fn blocks_to_labeled(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        let tag = match b.content_type {
            ContentType::Text => "TEXT",
            ContentType::Dialogue => "DIALOGUE",
            ContentType::Observation => "OBSERVATION",
            ContentType::Document => "DOCUMENT",
            ContentType::MediaRef => "MEDIA",
        };
        out.push_str(&format!("[{}]\n{}\n\n", tag, b.content));
    }
    out
}

/// Write the narrative's plan + metadata into an LLM prompt buffer.
///
/// `include_setting` controls whether time-period / locations / world_notes
/// / research_notes are written. Generation (W1) wants everything;
/// editing (W2) and workshop (W3) want the cheaper subset.
pub(crate) fn write_plan_section(
    out: &mut String,
    snapshot: &NarrativeSnapshot,
    include_setting: bool,
) {
    if let Some(plan) = snapshot.plan.as_ref() {
        out.push_str("[narrative plan]\n");
        if let Some(l) = &plan.logline {
            out.push_str(&format!("Logline: {}\n", l));
        }
        if include_setting {
            if let Some(s) = &plan.synopsis {
                out.push_str(&format!("Synopsis: {}\n", s));
            }
            if let Some(p) = &plan.premise {
                out.push_str(&format!("Premise: {}\n", p));
            }
            if !plan.themes.is_empty() {
                out.push_str(&format!("Themes: {}\n", plan.themes.join(", ")));
            }
            if let Some(c) = &plan.central_conflict {
                out.push_str(&format!("Central conflict: {}\n", c));
            }
        }

        let s = &plan.style;
        let mut style_parts: Vec<String> = Vec::new();
        if let Some(pov) = &s.pov {
            style_parts.push(format!("POV={}", pov));
        }
        if let Some(t) = &s.tense {
            style_parts.push(format!("tense={}", t));
        }
        if !s.tone.is_empty() {
            style_parts.push(format!("tone={}", s.tone.join("/")));
        }
        if let Some(v) = &s.voice {
            style_parts.push(format!("voice={}", v));
        }
        if !s.influences.is_empty() {
            style_parts.push(format!("influences={}", s.influences.join(", ")));
        }
        if !s.avoid.is_empty() {
            style_parts.push(format!("avoid={}", s.avoid.join(", ")));
        }
        if !style_parts.is_empty() {
            out.push_str(&format!("Style: {}\n", style_parts.join(" \u{00b7} ")));
        }

        if include_setting {
            let l = &plan.length;
            let mut length_parts: Vec<String> = Vec::new();
            if let Some(k) = &l.kind {
                length_parts.push(k.clone());
            }
            if let Some(tw) = l.target_words {
                length_parts.push(format!("target {}w", tw));
            }
            if let Some(tc) = l.target_chapters {
                length_parts.push(format!("target {} chapters", tc));
            }
            if !length_parts.is_empty() {
                out.push_str(&format!("Length: {}\n", length_parts.join(" \u{00b7} ")));
            }

            let st = &plan.setting;
            if let Some(tp) = &st.time_period {
                out.push_str(&format!("Time period: {}\n", tp));
            }
            if !st.locations.is_empty() {
                out.push_str(&format!("Locations: {}\n", st.locations.join(", ")));
            }
        }
        out.push('\n');
    }

    // Fallback metadata (title/genre) — useful when no plan is set.
    if include_setting {
        if let Some(meta) = snapshot.narrative_metadata.as_ref() {
            if let Some(title) = meta.get("title").and_then(|v| v.as_str()) {
                out.push_str(&format!("Title: {}\n", title));
            }
            if let Some(genre) = meta.get("genre").and_then(|v| v.as_str()) {
                out.push_str(&format!("Genre: {}\n", genre));
            }
        }
    }
}

/// Safe UTF-8 truncation that won't panic on char boundaries.
/// Returns the input unchanged when it's already under the limit.
pub(crate) fn truncate_utf8(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    // Walk backward from `n` to the nearest char boundary. `floor_char_boundary`
    // is nightly-only; this portable version does the same thing.
    let mut boundary = n;
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    format!("{}\u{2026}", &s[..boundary])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_tokens_rounds_up() {
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("abcd"), 1);
        assert_eq!(approx_tokens("abcde"), 2);
    }

    #[test]
    fn count_words_counts_whitespace_separated() {
        let blocks = vec![
            ContentBlock {
                content_type: ContentType::Text,
                content: "one two three".into(),
                source: None,
            },
            ContentBlock {
                content_type: ContentType::Dialogue,
                content: "\"four\"".into(),
                source: None,
            },
        ];
        assert_eq!(count_words_blocks(&blocks), 4);
    }

    #[test]
    fn blocks_to_labeled_tags_each_kind() {
        let blocks = vec![
            ContentBlock {
                content_type: ContentType::Text,
                content: "t".into(),
                source: None,
            },
            ContentBlock {
                content_type: ContentType::Dialogue,
                content: "d".into(),
                source: None,
            },
            ContentBlock {
                content_type: ContentType::Observation,
                content: "o".into(),
                source: None,
            },
        ];
        let out = blocks_to_labeled(&blocks);
        assert!(out.contains("[TEXT]\nt"));
        assert!(out.contains("[DIALOGUE]\nd"));
        assert!(out.contains("[OBSERVATION]\no"));
    }

    #[test]
    fn truncate_utf8_does_not_panic_on_multibyte() {
        // "\u{2192}" is a 3-byte char. Truncating at byte 2 falls inside it;
        // the result should fall back to an earlier char boundary.
        let s = "ab\u{2192}cd";
        // len = 2 + 3 + 2 = 7 bytes
        let out = truncate_utf8(s, 3);
        // Should have cut at byte 2 (the "ab" boundary), not mid-arrow.
        assert!(out.starts_with("ab"));
        assert!(out.ends_with('\u{2026}'));
    }

    #[test]
    fn truncate_utf8_passes_through_short_strings() {
        assert_eq!(truncate_utf8("short", 100), "short");
    }

    #[test]
    fn write_plan_section_skips_missing_plan() {
        let mut out = String::new();
        write_plan_section(&mut out, &NarrativeSnapshot::default(), true);
        assert!(out.is_empty());
    }
}
