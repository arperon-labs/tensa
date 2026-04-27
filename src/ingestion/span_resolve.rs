//! Resolve LLM-returned verbatim fragments (`text_start`, `text_end`) back to
//! byte offsets within a source chunk.
//!
//! The goal is to narrow `SourceSpan.byte_offset_*` from whole-chunk ranges
//! to the exact prose a situation was extracted from. Matching is tolerant
//! of whitespace differences (common when models normalize line wrapping),
//! but does not attempt edit-distance recovery — if the LLM paraphrases,
//! we return `None` and the pipeline keeps the chunk-wide fallback.
//!
//! Offsets are absolute: the caller passes `chunk_byte_start`, which is the
//! chunk's position in the original ingested source, and we return
//! `(abs_start, abs_end)` in that same frame.
//!
//! Key callers:
//! - `src/ingestion/pipeline.rs` (both the regular and SingleSession paths).

/// Resolve a pair of verbatim text fingerprints to absolute byte offsets.
///
/// Returns `(abs_start, abs_end)` in the original source coordinate system,
/// or `None` when neither fingerprint is present or no match can be made.
///
/// Matching rules:
/// - Whitespace is collapsed (runs of any Unicode whitespace → single space,
///   leading/trailing trimmed) for comparison only; returned offsets are
///   relative to the *original* haystack bytes.
/// - `text_start` is matched leftmost.
/// - `text_end` is matched rightmost at-or-after the `text_start` match.
/// - If only one fingerprint is supplied, the missing side is treated as the
///   chunk boundary (degraded but still narrower than chunk-wide).
pub fn resolve_span(
    chunk_text: &str,
    chunk_byte_start: usize,
    text_start: Option<&str>,
    text_end: Option<&str>,
) -> Option<(usize, usize)> {
    let start_fp = text_start.map(str::trim).filter(|s| !s.is_empty());
    let end_fp = text_end.map(str::trim).filter(|s| !s.is_empty());
    if start_fp.is_none() && end_fp.is_none() {
        return None;
    }

    let (hay_norm, hay_map) = normalize(chunk_text);
    if hay_norm.is_empty() {
        return None;
    }

    // Position in the normalized haystack where the situation starts.
    let start_norm = match start_fp {
        Some(fp) => {
            let (needle, _) = normalize(fp);
            if needle.is_empty() {
                0
            } else {
                hay_norm.find(&needle)?
            }
        }
        None => 0,
    };

    // End of the text_start match (or start of body when no text_start).
    let start_search_end = start_fp
        .map(|fp| start_norm + normalize(fp).0.len())
        .unwrap_or(start_norm);

    // Rightmost occurrence of text_end at-or-after start_search_end.
    let end_norm = match end_fp {
        Some(fp) => {
            let (needle, _) = normalize(fp);
            if needle.is_empty() {
                hay_norm.len()
            } else if start_search_end > hay_norm.len() {
                return None;
            } else {
                let rel = hay_norm[start_search_end..].rfind(&needle)?;
                start_search_end + rel + needle.len()
            }
        }
        None => hay_norm.len(),
    };

    if end_norm <= start_norm {
        return None;
    }

    // Map normalized byte positions back to original.
    let abs_start = hay_map.get(start_norm).copied()?;
    let abs_end = hay_map.get(end_norm).copied()?;
    if abs_end <= abs_start || abs_end > chunk_text.len() {
        return None;
    }
    Some((chunk_byte_start + abs_start, chunk_byte_start + abs_end))
}

/// Collapse Unicode whitespace runs to single ASCII spaces and trim.
/// Returns the normalized string plus a byte-offset map: for each byte `i`
/// in the normalized output, `map[i]` is the byte offset of the corresponding
/// position in the original. `map.len() == normalized.len() + 1` — the last
/// entry is a sentinel equal to `original.len()` so callers can query the
/// end-exclusive position of any match.
fn normalize(s: &str) -> (String, Vec<usize>) {
    let mut out = String::with_capacity(s.len());
    let mut map: Vec<usize> = Vec::with_capacity(s.len() + 1);
    let mut prev_was_space = true; // start in "trim leading ws" mode

    for (i, ch) in s.char_indices() {
        if ch.is_whitespace() {
            if !prev_was_space {
                map.push(i);
                out.push(' ');
                prev_was_space = true;
            }
        } else {
            let char_len = ch.len_utf8();
            let before = out.len();
            out.push(ch);
            let added = out.len() - before;
            // Each normalized byte maps back to the corresponding original byte
            // inside this character (UTF-8 bytes are consecutive in both).
            for k in 0..added {
                map.push(i + k.min(char_len - 1));
            }
            prev_was_space = false;
        }
    }

    // Trim a trailing space if we emitted one for a trailing-whitespace run.
    if out.ends_with(' ') {
        out.pop();
        map.pop();
    }

    map.push(s.len()); // sentinel for end-exclusive lookups
    (out, map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let chunk = "Hello world, this is a test sentence for span resolution.";
        let span = resolve_span(chunk, 1000, Some("Hello world"), Some("span resolution."));
        let (s, e) = span.expect("should match");
        assert_eq!(s, 1000);
        assert_eq!(&chunk[..e - 1000], chunk);
    }

    #[test]
    fn narrower_subspan() {
        let chunk = "Intro line. Jonathan left Munich at 8:35pm and arrived at Vienna. Outro line.";
        let (s, e) = resolve_span(
            chunk,
            0,
            Some("Jonathan left Munich"),
            Some("arrived at Vienna."),
        )
        .expect("match");
        assert!(&chunk[s..e].contains("Jonathan"));
        assert!(&chunk[s..e].ends_with("Vienna."));
        assert!(s > 0);
        assert!(e < chunk.len());
    }

    #[test]
    fn whitespace_tolerant_match() {
        // Original has line breaks; fingerprint is a single-spaced fragment.
        let chunk = "Alpha beta\ngamma\tdelta\n\nepsilon zeta.";
        let (s, e) = resolve_span(chunk, 0, Some("beta gamma delta"), Some("epsilon zeta."))
            .expect("whitespace-tolerant");
        assert!(&chunk[s..e].contains("gamma"));
        assert_eq!(&chunk[e - "zeta.".len()..e], "zeta.");
    }

    #[test]
    fn missing_fingerprints_returns_none() {
        let chunk = "Some text here.";
        assert!(resolve_span(chunk, 0, None, None).is_none());
        assert!(resolve_span(chunk, 0, Some(""), Some("")).is_none());
        assert!(resolve_span(chunk, 0, Some("   "), None).is_none());
    }

    #[test]
    fn unmatched_start_returns_none() {
        let chunk = "The quick brown fox.";
        assert!(resolve_span(chunk, 0, Some("lazy dog"), Some("fox.")).is_none());
    }

    #[test]
    fn unmatched_end_returns_none() {
        let chunk = "The quick brown fox.";
        assert!(resolve_span(chunk, 0, Some("The quick"), Some("jumped over")).is_none());
    }

    #[test]
    fn end_before_start_returns_none() {
        let chunk = "Foo bar. Baz qux.";
        // text_end can't appear before text_start in the haystack.
        assert!(resolve_span(chunk, 0, Some("Baz qux."), Some("Foo bar.")).is_none());
    }

    #[test]
    fn duplicate_start_picks_first() {
        // If text_start appears twice, the leftmost wins.
        let chunk = "Alpha. Alpha and then beta. End.";
        let (s, e) = resolve_span(chunk, 0, Some("Alpha"), Some("beta.")).expect("match");
        // First "Alpha" is at offset 0.
        assert_eq!(s, 0);
        assert!(&chunk[s..e].ends_with("beta."));
    }

    #[test]
    fn duplicate_end_picks_rightmost() {
        // If text_end appears twice after text_start, the rightmost wins.
        let chunk = "Start here. foo bar. middle. foo bar. finish.";
        let (s, e) = resolve_span(chunk, 0, Some("Start here."), Some("foo bar.")).expect("match");
        assert_eq!(s, 0);
        // Rightmost "foo bar." is the second one; e should reach past it.
        let rightmost = chunk.rfind("foo bar.").unwrap() + "foo bar.".len();
        assert_eq!(e, rightmost);
    }

    #[test]
    fn only_text_start_extends_to_end() {
        let chunk = "Prefix noise. The real beginning is here and continues.";
        let (s, e) =
            resolve_span(chunk, 0, Some("The real beginning"), None).expect("partial match");
        assert_eq!(&chunk[s..s + 18], "The real beginning");
        assert_eq!(e, chunk.len());
    }

    #[test]
    fn only_text_end_extends_from_start() {
        let chunk = "Here is the body and here is the terminator.";
        let (s, e) = resolve_span(chunk, 0, None, Some("terminator.")).expect("partial match");
        assert_eq!(s, 0);
        assert!(&chunk[s..e].ends_with("terminator."));
    }

    #[test]
    fn absolute_offset_respects_chunk_start() {
        let chunk = "Short chunk text.";
        let (s, e) = resolve_span(chunk, 42, Some("Short"), Some("text.")).expect("match");
        assert_eq!(s, 42);
        assert_eq!(e, 42 + chunk.len());
    }

    #[test]
    fn unicode_text_preserves_byte_offsets() {
        // "naïve café — résumé" — mixed multi-byte characters.
        let chunk = "Prefix. naïve café — résumé. Suffix.";
        let (s, e) = resolve_span(chunk, 0, Some("naïve café"), Some("résumé.")).expect("match");
        let slice = &chunk[s..e];
        assert!(slice.starts_with("naïve"));
        assert!(slice.ends_with("résumé."));
    }
}
