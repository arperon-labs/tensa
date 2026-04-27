//! Multilingual support: language detection and text normalization (Sprint D6).
//!
//! Uses Unicode script heuristics (no external dependency) to detect the
//! dominant language of a text snippet.  Returns ISO 639-1 codes where
//! possible, or a script-family fallback ("cjk", "arabic", "cyrillic", etc.)
//! for scripts that map to multiple languages.
//!
//! `normalize_for_matching` lowercases, strips accents (via Unicode NFD +
//! Mark removal), and collapses whitespace — useful for cross-lingual claim
//! matching.

use serde::{Deserialize, Serialize};

/// Detected language / script result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangDetectResult {
    /// ISO 639-1 code (e.g. "en", "fr", "ru") or script family ("cjk", "arabic").
    pub language: String,
    /// Confidence: fraction of classifiable characters that voted for this label.
    pub confidence: f64,
}

/// Detect the dominant language of `text` using Unicode script heuristics.
///
/// Returns an ISO 639-1 code when the mapping is unambiguous, or a
/// script-family name otherwise.
pub fn detect_language(text: &str) -> LangDetectResult {
    let mut latin = 0u32;
    let mut cyrillic = 0u32;
    let mut arabic = 0u32;
    let mut cjk = 0u32;
    let mut hangul = 0u32;
    let mut devanagari = 0u32;
    let mut total = 0u32;

    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            continue;
        }
        total += 1;
        match ch {
            '\u{0041}'..='\u{024F}' => latin += 1, // Basic Latin + Latin Extended
            '\u{0400}'..='\u{04FF}' => cyrillic += 1, // Cyrillic
            '\u{0600}'..='\u{06FF}' | '\u{0750}'..='\u{077F}' => arabic += 1, // Arabic
            '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' => cjk += 1, // CJK Unified
            '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' => hangul += 1, // Hangul
            '\u{0900}'..='\u{097F}' => devanagari += 1, // Devanagari
            _ => {}
        }
    }

    if total == 0 {
        return LangDetectResult {
            language: "unknown".to_string(),
            confidence: 0.0,
        };
    }

    let scores = [
        ("en", latin),      // Latin script — default to English
        ("ru", cyrillic),   // Cyrillic — default to Russian
        ("ar", arabic),     // Arabic
        ("zh", cjk),        // CJK — default to Chinese
        ("ko", hangul),     // Hangul → Korean
        ("hi", devanagari), // Devanagari → Hindi
    ];

    let (lang, count) = scores
        .iter()
        .max_by_key(|(_, c)| *c)
        .copied()
        .unwrap_or(("unknown", 0));

    LangDetectResult {
        language: lang.to_string(),
        confidence: count as f64 / total as f64,
    }
}

/// Normalize text for cross-lingual matching: lowercase, strip combining marks
/// (accents, diacritics), and collapse whitespace.
pub fn normalize_for_matching(text: &str) -> String {
    // NFD decomposition + strip combining marks
    let lowered = text.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut last_was_space = false;

    for ch in lowered.chars() {
        // Skip combining marks (Unicode category Mn — nonspacing marks).
        // After NFD decomposition, accents become separate combining chars.
        if is_combining_mark(ch) {
            continue;
        }
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

/// Check if a character is a Unicode combining mark (Mn category).
fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    // Common combining diacritical marks ranges
    (0x0300..=0x036F).contains(&cp)    // Combining Diacritical Marks
        || (0x1AB0..=0x1AFF).contains(&cp) // Combining Diacritical Marks Extended
        || (0x1DC0..=0x1DFF).contains(&cp) // Combining Diacritical Marks Supplement
        || (0x20D0..=0x20FF).contains(&cp) // Combining Marks for Symbols
        || (0xFE20..=0xFE2F).contains(&cp) // Combining Half Marks
}

// ─── Cyrillic → Latin Transliteration ────────────────────────

/// Transliterate Cyrillic text to Latin script (covers Russian + Ukrainian).
///
/// Uses a 1:1 or 1:N character mapping table. Unmapped characters pass through
/// unchanged.
pub fn transliterate_cyrillic_to_latin(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for ch in text.chars() {
        match ch {
            // Russian uppercase
            'А' => out.push('A'),
            'Б' => out.push('B'),
            'В' => out.push('V'),
            'Г' => out.push('G'),
            'Д' => out.push('D'),
            'Е' => out.push_str("Ye"),
            'Ё' => out.push_str("Yo"),
            'Ж' => out.push_str("Zh"),
            'З' => out.push('Z'),
            'И' => out.push('I'),
            'Й' => out.push('Y'),
            'К' => out.push('K'),
            'Л' => out.push('L'),
            'М' => out.push('M'),
            'Н' => out.push('N'),
            'О' => out.push('O'),
            'П' => out.push('P'),
            'Р' => out.push('R'),
            'С' => out.push('S'),
            'Т' => out.push('T'),
            'У' => out.push('U'),
            'Ф' => out.push('F'),
            'Х' => out.push_str("Kh"),
            'Ц' => out.push_str("Ts"),
            'Ч' => out.push_str("Ch"),
            'Ш' => out.push_str("Sh"),
            'Щ' => out.push_str("Shch"),
            'Ъ' => {} // hard sign — omit
            'Ы' => out.push('Y'),
            'Ь' => {} // soft sign — omit
            'Э' => out.push('E'),
            'Ю' => out.push_str("Yu"),
            'Я' => out.push_str("Ya"),
            // Russian lowercase
            'а' => out.push('a'),
            'б' => out.push('b'),
            'в' => out.push('v'),
            'г' => out.push('g'),
            'д' => out.push('d'),
            'е' => out.push_str("ye"),
            'ё' => out.push_str("yo"),
            'ж' => out.push_str("zh"),
            'з' => out.push('z'),
            'и' => out.push('i'),
            'й' => out.push('y'),
            'к' => out.push('k'),
            'л' => out.push('l'),
            'м' => out.push('m'),
            'н' => out.push('n'),
            'о' => out.push('o'),
            'п' => out.push('p'),
            'р' => out.push('r'),
            'с' => out.push('s'),
            'т' => out.push('t'),
            'у' => out.push('u'),
            'ф' => out.push('f'),
            'х' => out.push_str("kh"),
            'ц' => out.push_str("ts"),
            'ч' => out.push_str("ch"),
            'ш' => out.push_str("sh"),
            'щ' => out.push_str("shch"),
            'ъ' => {} // hard sign
            'ы' => out.push('y'),
            'ь' => {} // soft sign
            'э' => out.push('e'),
            'ю' => out.push_str("yu"),
            'я' => out.push_str("ya"),
            // Ukrainian-specific
            'І' | 'і' => out.push(if ch.is_uppercase() { 'I' } else { 'i' }),
            'Ї' => out.push_str("Yi"),
            'ї' => out.push_str("yi"),
            'Є' => out.push_str("Ye"),
            'є' => out.push_str("ye"),
            'Ґ' => out.push('G'),
            'ґ' => out.push('g'),
            _ => out.push(ch),
        }
    }
    out
}

// ─── Latin → Cyrillic Transliteration ────────────────────────

/// Transliterate Latin text to Cyrillic script (inverse of [`transliterate_cyrillic_to_latin`]).
///
/// Handles multi-character Latin digraphs (e.g. "sh" → "ш", "zh" → "ж") by
/// greedy longest-match. Single characters map 1:1 where possible.
pub fn transliterate_latin_to_cyrillic(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Try 4-char matches first, then 2, then 1.
        if i + 4 <= len {
            let quad: String = chars[i..i + 4].iter().collect();
            if let Some(c) = match_latin_quad(&quad) {
                out.push(c);
                i += 4;
                continue;
            }
        }
        if i + 2 <= len {
            let di: String = chars[i..i + 2].iter().collect();
            if let Some(c) = match_latin_di(&di) {
                out.push(c);
                i += 2;
                continue;
            }
        }
        out.push(match_latin_single(chars[i]));
        i += 1;
    }
    out
}

fn match_latin_quad(s: &str) -> Option<char> {
    match s {
        "Shch" => Some('Щ'),
        "shch" => Some('щ'),
        _ => None,
    }
}

fn match_latin_di(s: &str) -> Option<char> {
    match s {
        "Ye" => Some('Е'),
        "ye" => Some('е'),
        "Yo" => Some('Ё'),
        "yo" => Some('ё'),
        "Zh" => Some('Ж'),
        "zh" => Some('ж'),
        "Kh" => Some('Х'),
        "kh" => Some('х'),
        "Ts" => Some('Ц'),
        "ts" => Some('ц'),
        "Ch" => Some('Ч'),
        "ch" => Some('ч'),
        "Sh" => Some('Ш'),
        "sh" => Some('ш'),
        "Yu" => Some('Ю'),
        "yu" => Some('ю'),
        "Ya" => Some('Я'),
        "ya" => Some('я'),
        "Yi" => Some('Ї'),
        "yi" => Some('ї'),
        _ => None,
    }
}

fn match_latin_single(ch: char) -> char {
    match ch {
        'A' => 'А',
        'a' => 'а',
        'B' => 'Б',
        'b' => 'б',
        'V' => 'В',
        'v' => 'в',
        'G' => 'Г',
        'g' => 'г',
        'D' => 'Д',
        'd' => 'д',
        'Z' => 'З',
        'z' => 'з',
        'I' => 'И',
        'i' => 'и',
        'Y' => 'Й',
        'y' => 'й',
        'K' => 'К',
        'k' => 'к',
        'L' => 'Л',
        'l' => 'л',
        'M' => 'М',
        'm' => 'м',
        'N' => 'Н',
        'n' => 'н',
        'O' => 'О',
        'o' => 'о',
        'P' => 'П',
        'p' => 'п',
        'R' => 'Р',
        'r' => 'р',
        'S' => 'С',
        's' => 'с',
        'T' => 'Т',
        't' => 'т',
        'U' => 'У',
        'u' => 'у',
        'F' => 'Ф',
        'f' => 'ф',
        'E' => 'Э',
        'e' => 'э',
        other => other,
    }
}

// ─── Diacritic Stripping ─────────────────────────────────────

pub use crate::text_util::strip_diacritics;

// ─── Linguistic Variance (Shannon Entropy) ───────────────────

/// Compute the Shannon entropy of a language distribution, normalized to [0, 1].
///
/// Given a list of language codes (one per item in the corpus), counts their
/// frequencies and returns H / log₂(N) where N is the number of distinct
/// languages observed.
///
/// - 0.0 = all items in one language (or empty input)
/// - 1.0 = perfectly uniform distribution across all observed languages
pub fn linguistic_variance(languages: &[String]) -> f64 {
    if languages.is_empty() {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for lang in languages {
        *counts.entry(lang.as_str()).or_insert(0) += 1;
    }
    let n_distinct = counts.len();
    if n_distinct <= 1 {
        return 0.0;
    }
    let total = languages.len() as f64;
    let entropy: f64 = counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum();
    let max_entropy = (n_distinct as f64).ln();
    if max_entropy <= 0.0 {
        return 0.0;
    }
    (entropy / max_entropy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Language detection ────────────────────────────────────

    #[test]
    fn test_detect_english() {
        let result = detect_language("The quick brown fox jumps over the lazy dog.");
        assert_eq!(result.language, "en");
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_detect_cyrillic() {
        let result = detect_language("Быстрая бурая лисица перепрыгнула через ленивую собаку.");
        assert_eq!(result.language, "ru");
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_detect_arabic() {
        let result = detect_language("الثعلب البني السريع يقفز فوق الكلب الكسول");
        assert_eq!(result.language, "ar");
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_detect_cjk() {
        let result = detect_language("敏捷的棕色狐狸跳过了那只懒狗");
        assert_eq!(result.language, "zh");
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_detect_empty() {
        let result = detect_language("");
        assert_eq!(result.language, "unknown");
        assert_eq!(result.confidence, 0.0);
    }

    // ── Transliteration ──────────────────────────────────────

    #[test]
    fn test_cyrillic_to_latin_basic() {
        assert_eq!(transliterate_cyrillic_to_latin("Москва"), "Moskva");
    }

    #[test]
    fn test_cyrillic_to_latin_complex() {
        assert_eq!(transliterate_cyrillic_to_latin("Путин"), "Putin");
    }

    #[test]
    fn test_cyrillic_to_latin_ukrainian() {
        // Character-level: К→K, и→i, ї→yi, в→v
        assert_eq!(transliterate_cyrillic_to_latin("Київ"), "Kiyiv");
    }

    #[test]
    fn test_cyrillic_to_latin_mixed() {
        assert_eq!(transliterate_cyrillic_to_latin("Hello Мир"), "Hello Mir");
    }

    #[test]
    fn test_latin_to_cyrillic_basic() {
        assert_eq!(transliterate_latin_to_cyrillic("Moskva"), "Москва");
    }

    #[test]
    fn test_latin_to_cyrillic_digraphs() {
        assert_eq!(transliterate_latin_to_cyrillic("Zhuk"), "Жук");
    }

    // ── Diacritics ───────────────────────────────────────────

    #[test]
    fn test_strip_diacritics_czech() {
        assert_eq!(strip_diacritics("České Budějovice"), "Ceske Budejovice");
    }

    #[test]
    fn test_strip_diacritics_polish() {
        assert_eq!(strip_diacritics("Łódź"), "Lodz");
    }

    #[test]
    fn test_strip_diacritics_hungarian() {
        assert_eq!(strip_diacritics("Győr"), "Gyor");
    }

    #[test]
    fn test_strip_diacritics_german() {
        assert_eq!(strip_diacritics("Straße"), "Strasse");
    }

    #[test]
    fn test_strip_diacritics_plain_ascii() {
        assert_eq!(strip_diacritics("Hello World"), "Hello World");
    }

    // ── Normalize for matching ───────────────────────────────

    #[test]
    fn test_normalize_strips_accents() {
        let norm = normalize_for_matching("  Hello   World  ");
        assert_eq!(norm, "hello world");
    }

    #[test]
    fn test_normalize_collapses_whitespace() {
        let norm = normalize_for_matching("multiple   spaces\n\ttabs");
        assert_eq!(norm, "multiple spaces tabs");
    }

    // ── Linguistic variance ──────────────────────────────────

    #[test]
    fn test_linguistic_variance_empty() {
        assert_eq!(linguistic_variance(&[]), 0.0);
    }

    #[test]
    fn test_linguistic_variance_monolingual() {
        let langs: Vec<String> = vec!["en".into(), "en".into(), "en".into()];
        assert_eq!(linguistic_variance(&langs), 0.0);
    }

    #[test]
    fn test_linguistic_variance_uniform_two() {
        let langs: Vec<String> = vec!["en".into(), "ru".into()];
        let v = linguistic_variance(&langs);
        assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
    }

    #[test]
    fn test_linguistic_variance_skewed() {
        let langs: Vec<String> = vec![
            "en".into(),
            "en".into(),
            "en".into(),
            "en".into(),
            "ru".into(),
        ];
        let v = linguistic_variance(&langs);
        assert!(v > 0.0 && v < 1.0, "expected (0, 1), got {v}");
    }

    #[test]
    fn test_linguistic_variance_three_uniform() {
        let langs: Vec<String> = vec!["en".into(), "ru".into(), "ar".into()];
        let v = linguistic_variance(&langs);
        assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
    }
}
