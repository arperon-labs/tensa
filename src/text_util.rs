//! Small text utilities shared across feature-gated and non-gated modules.
//!
//! Kept non-gated so `hypergraph`, `api`, and ingestion can all consume the
//! same diacritic-stripping logic. The `disinfo::multilingual` module
//! re-exports `strip_diacritics` from here for backward compatibility.

/// Strip diacritical marks from text for Central/Eastern European name matching.
///
/// Covers common accented characters in Czech, Slovak, Polish, Hungarian,
/// Romanian, Croatian, Serbian Latin, Nordic, German, Turkish.
pub fn strip_diacritics(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            // Acute, grave, circumflex, tilde, umlaut — vowels
            'á' | 'à' | 'â' | 'ã' | 'ä' | 'å' => out.push('a'),
            'Á' | 'À' | 'Â' | 'Ã' | 'Ä' | 'Å' => out.push('A'),
            'é' | 'è' | 'ê' | 'ë' | 'ě' => out.push('e'),
            'É' | 'È' | 'Ê' | 'Ë' | 'Ě' => out.push('E'),
            'í' | 'ì' | 'î' | 'ï' | 'ı' => out.push('i'),
            'Í' | 'Ì' | 'Î' | 'Ï' | 'İ' => out.push('I'),
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' | 'ő' => out.push('o'),
            'Ó' | 'Ò' | 'Ô' | 'Õ' | 'Ö' | 'Ő' => out.push('O'),
            'ú' | 'ù' | 'û' | 'ü' | 'ű' | 'ů' => out.push('u'),
            'Ú' | 'Ù' | 'Û' | 'Ü' | 'Ű' | 'Ů' => out.push('U'),
            'ý' | 'ÿ' => out.push('y'),
            'Ý' | 'Ÿ' => out.push('Y'),
            // Consonants with háčky, cedilla, ogonek, etc.
            'č' | 'ć' | 'ç' => out.push('c'),
            'Č' | 'Ć' | 'Ç' => out.push('C'),
            'ď' | 'đ' => out.push('d'),
            'Ď' | 'Đ' => out.push('D'),
            'ğ' => out.push('g'),
            'Ğ' => out.push('G'),
            'ł' | 'ľ' => out.push('l'),
            'Ł' | 'Ľ' => out.push('L'),
            'ň' | 'ń' | 'ñ' => out.push('n'),
            'Ň' | 'Ń' | 'Ñ' => out.push('N'),
            'ř' => out.push('r'),
            'Ř' => out.push('R'),
            'š' | 'ś' | 'ş' => out.push('s'),
            'Š' | 'Ś' | 'Ş' => out.push('S'),
            'ť' | 'ţ' => out.push('t'),
            'Ť' | 'Ţ' => out.push('T'),
            'ž' | 'ź' | 'ż' => out.push('z'),
            'Ž' | 'Ź' | 'Ż' => out.push('Z'),
            'ą' | 'ă' => out.push('a'),
            'Ą' | 'Ă' => out.push('A'),
            'ę' => out.push('e'),
            'Ę' => out.push('E'),
            'ð' => out.push('d'),
            'Ð' => out.push('D'),
            'þ' => out.push_str("th"),
            'Þ' => out.push_str("Th"),
            'æ' => out.push_str("ae"),
            'Æ' => out.push_str("AE"),
            'ø' => out.push('o'),
            'Ø' => out.push('O'),
            'ß' => out.push_str("ss"),
            _ => out.push(ch),
        }
    }
    out
}

/// Lowercase + diacritic-fold + separator-normalize (`_`/` ` → `-`) —
/// idempotent slug form used for name-based entity lookup.
pub fn normalize_slug(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in strip_diacritics(&s.to_lowercase()).chars() {
        out.push(match ch {
            '_' | ' ' => '-',
            c => c,
        });
    }
    out
}
