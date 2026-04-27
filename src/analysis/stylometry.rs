//! Prose-level stylometry analysis for narrative text.
//!
//! Computes linguistic features from raw text using pure Rust algorithms:
//! lexical richness (Yule's K, Simpson's D, hapax legomena), sentence rhythm
//! (length distribution, autocorrelation), readability (Flesch-Kincaid, ARI),
//! formality markers (passive voice, adjective/adverb density, dialogue ratio),
//! and function word frequencies for Burrows' Delta authorship comparison.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ─── Canonical Function Word List (top 100 by English frequency) ───────────

const FUNCTION_WORDS: [&str; 100] = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "it", "for", "as", "with", "his",
    "he", "on", "be", "at", "by", "i", "this", "had", "not", "are", "but", "from", "or", "have",
    "an", "they", "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
    "we", "him", "been", "has", "when", "who", "will", "more", "no", "if", "out", "so", "what",
    "up", "its", "about", "into", "than", "them", "can", "only", "other", "new", "some", "could",
    "time", "these", "two", "may", "then", "do", "first", "any", "my", "now", "such", "like",
    "our", "over", "man", "me", "even", "most", "after", "also", "did", "many", "before", "must",
    "through", "back", "years", "where", "much", "your", "way", "well", "down", "should",
    "because",
];

const CONJUNCTIONS: [&str; 7] = ["and", "but", "or", "so", "yet", "for", "nor"];

const BE_FORMS: [&str; 8] = ["was", "were", "is", "are", "been", "be", "being", "am"];

const ABBREVIATIONS: [&str; 10] = [
    "mr.", "mrs.", "dr.", "ms.", "jr.", "sr.", "st.", "ave.", "blvd.", "etc.",
];

// ─── Data Structures ───────────────────────────────────────────

/// Prose-level stylometric features extracted from raw text.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProseStyleFeatures {
    // Lexical (Burrows' Delta inputs)
    pub function_word_frequencies: Vec<f32>,
    pub type_token_ratio: f32,
    pub hapax_legomena_ratio: f32,
    pub yule_k: f32,
    pub simpsons_d: f32,

    // Sentence Rhythm
    pub avg_sentence_length: f32,
    pub sentence_length_std: f32,
    pub sentence_length_cv: f32,
    pub sentence_length_autocorrelation: f32,
    pub short_sentence_ratio: f32,
    pub medium_sentence_ratio: f32,
    pub long_sentence_ratio: f32,
    pub very_long_sentence_ratio: f32,

    // Readability
    pub flesch_kincaid_grade: f32,
    pub flesch_reading_ease: f32,
    pub automated_readability_index: f32,

    // Formality & Craft
    pub passive_voice_ratio: f32,
    pub adjective_density: f32,
    pub adverb_density: f32,
    pub dialogue_ratio: f32,
    pub question_ratio: f32,
    pub exclamation_ratio: f32,
    pub avg_word_length: f32,
    pub conjunction_density: f32,

    // Metadata
    pub total_words: usize,
    pub total_sentences: usize,
    pub total_unique_words: usize,
}

/// Corpus-level statistics for Burrows' Delta normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub function_word_means: Vec<f32>,
    pub function_word_stds: Vec<f32>,
}

// ─── Main Entry Point ──────────────────────────────────────────

/// Compute all prose stylometric features from raw text.
pub fn compute_prose_features(text: &str) -> ProseStyleFeatures {
    let tokens = tokenize(text);
    let sentences = split_sentences(text);
    let n = tokens.len();
    let n_sent = sentences.len();

    if n == 0 {
        return zeroed_features();
    }

    // Word frequency map
    let mut freq_map: HashMap<&str, usize> = HashMap::new();
    for t in &tokens {
        *freq_map.entry(t.as_str()).or_insert(0) += 1;
    }
    let unique = freq_map.len();

    // Function word frequencies
    let fw_freqs = function_word_frequencies(&tokens, n);

    // TTR on 1000-word windows
    let ttr = compute_ttr(&tokens);

    // Hapax legomena
    let hapax_count = freq_map.values().filter(|&&c| c == 1).count();
    let hapax_ratio = if unique > 0 {
        hapax_count as f32 / unique as f32
    } else {
        0.0
    };

    // Yule's K
    let yule_k = compute_yule_k(&freq_map, n);

    // Simpson's D
    let simpsons_d = compute_simpsons_d(&freq_map, n);

    // Sentence lengths (use whitespace splitting to avoid re-tokenizing every sentence)
    let sent_lengths: Vec<usize> = sentences
        .iter()
        .map(|s| s.split_whitespace().count())
        .collect();
    let (avg_sl, std_sl, cv_sl) = sentence_length_stats(&sent_lengths);
    let autocorr = lag1_autocorrelation(&sent_lengths);
    let (short, medium, long, very_long) = sentence_bins(&sent_lengths);

    // Syllable count
    let total_syllables: usize = tokens.iter().map(|w| count_syllables(w)).sum();
    let total_chars: usize = tokens
        .iter()
        .map(|w| w.chars().filter(|c| c.is_alphabetic()).count())
        .sum();
    let nf = n as f32;
    let sf = n_sent.max(1) as f32;

    // Readability
    let wps = nf / sf;
    let spw = total_syllables as f32 / nf;
    let cpw = total_chars as f32 / nf;
    let fk_grade = 0.39 * wps + 11.8 * spw - 15.59;
    let fre = 206.835 - 1.015 * wps - 84.6 * spw;
    let ari = 4.71 * cpw + 0.5 * wps - 21.43;

    // Formality markers
    let passive = passive_voice_ratio(&tokens);
    let adj_den = adjective_density(&tokens);
    let adv_den = adverb_density(&tokens);
    let dial = dialogue_ratio(text);
    let (q_ratio, exc_ratio) = question_exclamation_ratios(&sentences);
    let avg_wl = cpw;
    let conj_den = conjunction_density(&tokens);

    ProseStyleFeatures {
        function_word_frequencies: fw_freqs,
        type_token_ratio: ttr,
        hapax_legomena_ratio: hapax_ratio,
        yule_k,
        simpsons_d,
        avg_sentence_length: avg_sl,
        sentence_length_std: std_sl,
        sentence_length_cv: cv_sl,
        sentence_length_autocorrelation: autocorr,
        short_sentence_ratio: short,
        medium_sentence_ratio: medium,
        long_sentence_ratio: long,
        very_long_sentence_ratio: very_long,
        flesch_kincaid_grade: fk_grade,
        flesch_reading_ease: fre,
        automated_readability_index: ari,
        passive_voice_ratio: passive,
        adjective_density: adj_den,
        adverb_density: adv_den,
        dialogue_ratio: dial,
        question_ratio: q_ratio,
        exclamation_ratio: exc_ratio,
        avg_word_length: avg_wl,
        conjunction_density: conj_den,
        total_words: n,
        total_sentences: n_sent,
        total_unique_words: unique,
    }
}

/// Compute z-scored function word frequency vectors for two profiles.
/// Returns paired z-score vectors, skipping slots with zero corpus std.
fn zscore_pair(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &CorpusStats,
) -> (Vec<f32>, Vec<f32>) {
    let k = a
        .function_word_frequencies
        .len()
        .min(b.function_word_frequencies.len());
    let mut za_vec = Vec::with_capacity(k);
    let mut zb_vec = Vec::with_capacity(k);
    for i in 0..k {
        let std = if i < corpus_stats.function_word_stds.len() {
            corpus_stats.function_word_stds[i]
        } else {
            1.0
        };
        if std > 1e-12 {
            let mean = corpus_stats
                .function_word_means
                .get(i)
                .copied()
                .unwrap_or(0.0);
            za_vec.push((a.function_word_frequencies[i] - mean) / std);
            zb_vec.push((b.function_word_frequencies[i] - mean) / std);
        }
    }
    (za_vec, zb_vec)
}

/// Burrows' Delta distance between two feature profiles, normalized by corpus stats.
pub fn burrows_delta(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &CorpusStats,
) -> f32 {
    let (za, zb) = zscore_pair(a, b, corpus_stats);
    if za.is_empty() {
        return 0.0;
    }
    let sum: f32 = za.iter().zip(zb.iter()).map(|(a, b)| (a - b).abs()).sum();
    sum / za.len() as f32
}

/// Compute corpus-level mean and standard deviation for each function word slot.
pub fn compute_corpus_stats(profiles: &[ProseStyleFeatures]) -> CorpusStats {
    if profiles.is_empty() {
        return CorpusStats {
            function_word_means: vec![0.0; 100],
            function_word_stds: vec![1.0; 100],
        };
    }
    let k = 100;
    let n = profiles.len() as f32;
    let mut means = vec![0.0f32; k];
    let mut stds = vec![0.0f32; k];

    for p in profiles {
        for i in 0..k.min(p.function_word_frequencies.len()) {
            means[i] += p.function_word_frequencies[i];
        }
    }
    for m in means.iter_mut() {
        *m /= n;
    }
    for p in profiles {
        for i in 0..k.min(p.function_word_frequencies.len()) {
            let diff = p.function_word_frequencies[i] - means[i];
            stds[i] += diff * diff;
        }
    }
    for s in stds.iter_mut() {
        *s = (*s / n).sqrt();
    }
    CorpusStats {
        function_word_means: means,
        function_word_stds: stds,
    }
}

/// Aggregate multiple feature sets into one, weighted by word count.
pub fn aggregate_features(features: &[ProseStyleFeatures]) -> ProseStyleFeatures {
    if features.is_empty() {
        return zeroed_features();
    }
    if features.len() == 1 {
        return features[0].clone();
    }

    let total_w: usize = features.iter().map(|f| f.total_words).sum();
    if total_w == 0 {
        return zeroed_features();
    }
    let tw = total_w as f32;

    let weighted = |getter: fn(&ProseStyleFeatures) -> f32| -> f32 {
        features
            .iter()
            .map(|f| getter(f) * f.total_words as f32)
            .sum::<f32>()
            / tw
    };

    let mut fw = vec![0.0f32; 100];
    for f in features {
        let w = f.total_words as f32 / tw;
        for (i, v) in f.function_word_frequencies.iter().enumerate().take(100) {
            fw[i] += v * w;
        }
    }

    let total_sent: usize = features.iter().map(|f| f.total_sentences).sum();
    let total_unique: usize = features.iter().map(|f| f.total_unique_words).sum();

    ProseStyleFeatures {
        function_word_frequencies: fw,
        type_token_ratio: weighted(|f| f.type_token_ratio),
        hapax_legomena_ratio: weighted(|f| f.hapax_legomena_ratio),
        yule_k: weighted(|f| f.yule_k),
        simpsons_d: weighted(|f| f.simpsons_d),
        avg_sentence_length: weighted(|f| f.avg_sentence_length),
        sentence_length_std: weighted(|f| f.sentence_length_std),
        sentence_length_cv: weighted(|f| f.sentence_length_cv),
        sentence_length_autocorrelation: weighted(|f| f.sentence_length_autocorrelation),
        short_sentence_ratio: weighted(|f| f.short_sentence_ratio),
        medium_sentence_ratio: weighted(|f| f.medium_sentence_ratio),
        long_sentence_ratio: weighted(|f| f.long_sentence_ratio),
        very_long_sentence_ratio: weighted(|f| f.very_long_sentence_ratio),
        flesch_kincaid_grade: weighted(|f| f.flesch_kincaid_grade),
        flesch_reading_ease: weighted(|f| f.flesch_reading_ease),
        automated_readability_index: weighted(|f| f.automated_readability_index),
        passive_voice_ratio: weighted(|f| f.passive_voice_ratio),
        adjective_density: weighted(|f| f.adjective_density),
        adverb_density: weighted(|f| f.adverb_density),
        dialogue_ratio: weighted(|f| f.dialogue_ratio),
        question_ratio: weighted(|f| f.question_ratio),
        exclamation_ratio: weighted(|f| f.exclamation_ratio),
        avg_word_length: weighted(|f| f.avg_word_length),
        conjunction_density: weighted(|f| f.conjunction_density),
        total_words: total_w,
        total_sentences: total_sent,
        total_unique_words: total_unique,
    }
}

// ─── Private Helpers ───────────────────────────────────────────

fn zeroed_features() -> ProseStyleFeatures {
    ProseStyleFeatures {
        function_word_frequencies: vec![0.0; 100],
        type_token_ratio: 0.0,
        hapax_legomena_ratio: 0.0,
        yule_k: 0.0,
        simpsons_d: 0.0,
        avg_sentence_length: 0.0,
        sentence_length_std: 0.0,
        sentence_length_cv: 0.0,
        sentence_length_autocorrelation: 0.0,
        short_sentence_ratio: 0.0,
        medium_sentence_ratio: 0.0,
        long_sentence_ratio: 0.0,
        very_long_sentence_ratio: 0.0,
        flesch_kincaid_grade: 0.0,
        flesch_reading_ease: 0.0,
        automated_readability_index: 0.0,
        passive_voice_ratio: 0.0,
        adjective_density: 0.0,
        adverb_density: 0.0,
        dialogue_ratio: 0.0,
        question_ratio: 0.0,
        exclamation_ratio: 0.0,
        avg_word_length: 0.0,
        conjunction_density: 0.0,
        total_words: 0,
        total_sentences: 0,
        total_unique_words: 0,
    }
}

/// Tokenize text: lowercase, strip leading/trailing punctuation, keep internal hyphens/apostrophes.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let lower = raw.to_lowercase();
            let trimmed =
                lower.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'' && c != '-');
            // Also strip leading/trailing apostrophes and hyphens that are standalone
            let trimmed = trimmed.trim_matches(|c: char| c == '\'' || c == '-');
            if trimmed.is_empty() {
                None
            } else {
                // Re-strip trailing punctuation that may remain after the word
                let t = trimmed
                    .trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '\'' && c != '-');
                let t = t.trim_start_matches(|c: char| {
                    c.is_ascii_punctuation() && c != '\'' && c != '-'
                });
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                }
            }
        })
        .collect()
}

/// Split text into sentences, handling abbreviations and ellipsis.
fn split_sentences(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut sentences = Vec::new();
    let mut start = 0;

    let mut i = 0;
    while i < len {
        let ch = chars[i];

        // Handle ellipsis: skip consecutive dots
        if ch == '.' && i + 1 < len && chars[i + 1] == '.' {
            while i < len && chars[i] == '.' {
                i += 1;
            }
            continue;
        }

        if ch == '.' || ch == '!' || ch == '?' {
            // Check for abbreviation (only for '.')
            if ch == '.' {
                // Look back to find the word before this period
                let word_end = i;
                let mut word_start = i;
                if word_start > 0 {
                    word_start -= 1;
                    while word_start > 0 && chars[word_start].is_alphabetic() {
                        word_start -= 1;
                    }
                    if !chars[word_start].is_alphabetic() {
                        word_start += 1;
                    }
                }
                let before_word: String = chars[word_start..=word_end]
                    .iter()
                    .collect::<String>()
                    .to_lowercase();
                if ABBREVIATIONS.iter().any(|&abbr| before_word == abbr) {
                    i += 1;
                    continue;
                }
            }

            // Check if followed by whitespace + uppercase (or end of text) = sentence boundary
            let mut j = i + 1;
            // Skip consecutive sentence-ending punctuation (e.g., ?! or !!)
            while j < len && (chars[j] == '.' || chars[j] == '!' || chars[j] == '?') {
                j += 1;
            }
            // Check for closing quote after punctuation
            while j < len && (chars[j] == '"' || chars[j] == '\'' || chars[j] == '\u{201D}') {
                j += 1;
            }

            let is_boundary = if j >= len {
                true
            } else if chars[j].is_whitespace() {
                // Look ahead past whitespace for uppercase
                let mut k = j;
                while k < len && chars[k].is_whitespace() {
                    k += 1;
                }
                k >= len || chars[k].is_uppercase() || chars[k] == '"' || chars[k] == '\u{201C}'
            } else {
                false
            };

            if is_boundary {
                let sent: String = chars[start..j].iter().collect();
                let trimmed = sent.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                start = j;
            }
        }
        i += 1;
    }

    // Remaining text
    if start < len {
        let sent: String = chars[start..].iter().collect();
        let trimmed = sent.trim();
        if !trimmed.is_empty() {
            sentences.push(trimmed.to_string());
        }
    }

    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }

    sentences
}

/// Approximate English syllable count.
fn count_syllables(word: &str) -> usize {
    let lower = word.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    if chars.is_empty() {
        return 1;
    }

    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let mut count = 0usize;
    let mut prev_vowel = false;

    for &ch in &chars {
        let is_v = vowels.contains(&ch);
        if is_v && !prev_vowel {
            count += 1;
        }
        prev_vowel = is_v;
    }

    // Silent 'e' at end: only subtract if we have more than 1 vowel group
    // and the word doesn't end in a vowel cluster before 'e' (like "create" = cre-ate)
    if count > 1
        && chars.len() > 2
        && chars[chars.len() - 1] == 'e'
        && !vowels.contains(&chars[chars.len() - 2])
    {
        // Don't subtract for -le endings (e.g., "table", "little") or -te after a vowel
        let second_last = chars[chars.len() - 2];
        if second_last != 'l'
            && !(vowels.contains(
                &chars
                    .get(chars.len().wrapping_sub(3))
                    .copied()
                    .unwrap_or(' '),
            ))
        {
            count = count.saturating_sub(1);
        }
    }

    count.max(1)
}

/// Compute function word frequencies (length 100) from tokens.
fn function_word_frequencies(tokens: &[String], total: usize) -> Vec<f32> {
    use std::sync::OnceLock;
    static FW_INDEX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    let index = FW_INDEX.get_or_init(|| {
        FUNCTION_WORDS
            .iter()
            .enumerate()
            .map(|(i, &w)| (w, i))
            .collect()
    });

    if total == 0 {
        return vec![0.0; 100];
    }
    let mut counts = vec![0usize; 100];
    for t in tokens {
        if let Some(&i) = index.get(t.as_str()) {
            counts[i] += 1;
        }
    }
    counts.iter().map(|&c| c as f32 / total as f32).collect()
}

/// TTR on 1000-word windows, averaged.
fn compute_ttr(tokens: &[String]) -> f32 {
    let n = tokens.len();
    if n == 0 {
        return 0.0;
    }
    let window = 1000.min(n);
    if n <= window {
        let unique: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        return unique.len() as f32 / n as f32;
    }
    let num_windows = n - window + 1;
    // Sample up to 10 evenly spaced windows for efficiency
    let step = (num_windows / 10).max(1);
    let mut total_ttr = 0.0f32;
    let mut count = 0;
    let mut i = 0;
    while i < num_windows {
        let unique: std::collections::HashSet<&str> =
            tokens[i..i + window].iter().map(|s| s.as_str()).collect();
        total_ttr += unique.len() as f32 / window as f32;
        count += 1;
        i += step;
    }
    total_ttr / count as f32
}

/// Yule's K: K = 10000 * (M2 - N) / N^2
fn compute_yule_k(freq_map: &HashMap<&str, usize>, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    // freq_spectrum[i] = number of words that appear exactly i times
    let mut spectrum: HashMap<usize, usize> = HashMap::new();
    for &count in freq_map.values() {
        *spectrum.entry(count).or_insert(0) += 1;
    }
    let m2: f64 = spectrum
        .iter()
        .map(|(&i, &fi)| (i as f64) * (i as f64) * (fi as f64))
        .sum();
    let nf = n as f64;
    let k = 10000.0 * (m2 - nf) / (nf * nf);
    k as f32
}

/// Simpson's D: D = 1 - sum(n_i * (n_i - 1)) / (N * (N - 1))
fn compute_simpsons_d(freq_map: &HashMap<&str, usize>, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let numer: f64 = freq_map
        .values()
        .map(|&c| (c as f64) * (c as f64 - 1.0))
        .sum();
    let denom = (n as f64) * (n as f64 - 1.0);
    (1.0 - numer / denom) as f32
}

fn sentence_length_stats(lengths: &[usize]) -> (f32, f32, f32) {
    if lengths.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = lengths.len() as f32;
    let mean = lengths.iter().sum::<usize>() as f32 / n;
    if lengths.len() == 1 {
        return (mean, 0.0, 0.0);
    }
    let var = lengths
        .iter()
        .map(|&l| (l as f32 - mean).powi(2))
        .sum::<f32>()
        / n;
    let std = var.sqrt();
    let cv = if mean > 0.0 { std / mean } else { 0.0 };
    (mean, std, cv)
}

/// Lag-1 autocorrelation of sentence lengths.
fn lag1_autocorrelation(lengths: &[usize]) -> f32 {
    if lengths.len() < 3 {
        return 0.0;
    }
    let n = lengths.len();
    let mean = lengths.iter().sum::<usize>() as f32 / n as f32;
    let denom: f32 = lengths.iter().map(|&l| (l as f32 - mean).powi(2)).sum();
    if denom < 1e-12 {
        return 0.0;
    }
    let numer: f32 = (0..n - 1)
        .map(|i| (lengths[i] as f32 - mean) * (lengths[i + 1] as f32 - mean))
        .sum();
    numer / denom
}

fn sentence_bins(lengths: &[usize]) -> (f32, f32, f32, f32) {
    if lengths.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = lengths.len() as f32;
    let short = lengths.iter().filter(|&&l| l <= 5).count() as f32 / n;
    let medium = lengths.iter().filter(|&&l| (6..=20).contains(&l)).count() as f32 / n;
    let long = lengths.iter().filter(|&&l| (21..=40).contains(&l)).count() as f32 / n;
    let very_long = lengths.iter().filter(|&&l| l > 40).count() as f32 / n;
    (short, medium, long, very_long)
}

/// Approximate passive voice ratio: "to be" form followed within 3 words by -ed/-en/-t ending.
fn passive_voice_ratio(tokens: &[String]) -> f32 {
    let n = tokens.len();
    if n == 0 {
        return 0.0;
    }
    let mut passive_count = 0usize;
    // Count approximate passive constructions
    for i in 0..n {
        if BE_FORMS.contains(&tokens[i].as_str()) {
            let end = (i + 4).min(n);
            for j in (i + 1)..end {
                let w = &tokens[j];
                if w.ends_with("ed")
                    || w.ends_with("en")
                    || w.ends_with("wn")
                    || w.ends_with("nt")
                    || w.ends_with("lt")
                    || w.ends_with("pt")
                {
                    passive_count += 1;
                    break;
                }
            }
        }
    }
    // Rough sentence count for ratio
    let sentence_count = (n / 15).max(1);
    passive_count as f32 / sentence_count as f32
}

/// Adjective density: words with -ful/-less/-ous/-ive/-al/-ent/-ant/-able/-ible/-ic endings.
fn adjective_density(tokens: &[String]) -> f32 {
    let n = tokens.len();
    if n == 0 {
        return 0.0;
    }
    let suffixes = [
        "ful", "less", "ous", "ive", "al", "ent", "ant", "able", "ible", "ic",
    ];
    let count = tokens
        .iter()
        .filter(|w| w.len() > 3 && suffixes.iter().any(|s| w.ends_with(s)))
        .count();
    count as f32 / n as f32
}

/// Adverb density: words ending in -ly.
fn adverb_density(tokens: &[String]) -> f32 {
    let n = tokens.len();
    if n == 0 {
        return 0.0;
    }
    let count = tokens
        .iter()
        .filter(|w| w.len() > 2 && w.ends_with("ly"))
        .count();
    count as f32 / n as f32
}

/// Dialogue ratio: fraction of characters inside quotation marks.
fn dialogue_ratio(text: &str) -> f32 {
    let total = text.len();
    if total == 0 {
        return 0.0;
    }
    let mut in_quote = false;
    let mut dialogue_chars = 0usize;
    for ch in text.chars() {
        match ch {
            '"' | '\u{201C}' | '\u{201D}' => {
                in_quote = !in_quote;
            }
            _ => {
                if in_quote {
                    dialogue_chars += ch.len_utf8();
                }
            }
        }
    }
    dialogue_chars as f32 / total as f32
}

/// Question and exclamation ratios from sentences.
fn question_exclamation_ratios(sentences: &[String]) -> (f32, f32) {
    let n = sentences.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let nf = n as f32;
    let q = sentences
        .iter()
        .filter(|s| s.trim_end().ends_with('?'))
        .count() as f32
        / nf;
    let e = sentences
        .iter()
        .filter(|s| s.trim_end().ends_with('!'))
        .count() as f32
        / nf;
    (q, e)
}

/// Conjunction density.
fn conjunction_density(tokens: &[String]) -> f32 {
    let n = tokens.len();
    if n == 0 {
        return 0.0;
    }
    let count = tokens
        .iter()
        .filter(|t| CONJUNCTIONS.contains(&t.as_str()))
        .count();
    count as f32 / n as f32
}

// ─── Cosine Delta (Würzburg Delta) ────────────────────────────

/// Cosine Delta distance between two feature profiles, normalized by corpus stats.
///
/// Unlike Burrows' Delta (which uses Manhattan distance on z-scores),
/// Cosine Delta uses cosine distance on z-scored function word frequencies.
/// Research shows it outperforms Burrows' Delta on most authorship attribution tasks.
pub fn cosine_delta(
    a: &ProseStyleFeatures,
    b: &ProseStyleFeatures,
    corpus_stats: &CorpusStats,
) -> f32 {
    let (za, zb) = zscore_pair(a, b, corpus_stats);
    if za.is_empty() {
        return 0.0;
    }

    let dot: f32 = za.iter().zip(zb.iter()).map(|(a, b)| a * b).sum();
    let mag_a: f32 = za.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mag_b: f32 = zb.iter().map(|v| v * v).sum::<f32>().sqrt();

    if mag_a < 1e-12 || mag_b < 1e-12 {
        return 1.0;
    }

    1.0 - (dot / (mag_a * mag_b))
}

// ─── Rolling Stylometry ──────────────────────────────────────

/// Sliding-window prose analysis for authorship change detection.
///
/// Computes `ProseStyleFeatures` for overlapping windows of text,
/// enabling detection of stylistic shifts (potential authorship changes,
/// genre transitions, or tonal shifts) across a document.
///
/// `window_size`: number of words per window.
/// `step`: number of words to advance between windows.
pub fn rolling_style_analysis(
    text: &str,
    window_size: usize,
    step: usize,
) -> Vec<ProseStyleFeatures> {
    if window_size == 0 || step == 0 {
        return vec![];
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < window_size {
        // Entire text is smaller than one window — return single analysis
        return vec![compute_prose_features(text)];
    }

    let mut results = Vec::new();
    let mut offset = 0;

    while offset + window_size <= words.len() {
        let window_text = words[offset..offset + window_size].join(" ");
        results.push(compute_prose_features(&window_text));
        offset += step;
    }

    results
}

// ─── Fitness Loop Helpers ──────────────────────────────────────
//
// Two helpers consumed by the SE→fitness chapter generation loop:
// rendering a target into LLM-readable directives, and rendering
// per-axis deltas as actionable corrections. Only the prompt-actionable
// subset of axes is surfaced to the LLM — function-word frequencies,
// hapax ratios, etc. influence the *score* via fingerprint similarity
// but not the *constraint list* (you cannot tell a model "use 'the'
// 4.2% of the time").

const FITNESS_ACTIONABLE_DELTA_AVG_SENTENCE: f32 = 3.0;
const FITNESS_ACTIONABLE_DELTA_SENTENCE_STD: f32 = 2.0;
const FITNESS_ACTIONABLE_DELTA_DIALOGUE: f32 = 0.10;
const FITNESS_ACTIONABLE_DELTA_PASSIVE: f32 = 0.05;
const FITNESS_ACTIONABLE_DELTA_FK: f32 = 2.0;
const FITNESS_ACTIONABLE_DELTA_TTR: f32 = 0.05;

/// Render a target `ProseStyleFeatures` into LLM-readable directives.
///
/// Used at iteration 0 of the fitness loop to seed the prompt with
/// numeric/qualitative style targets. Skips axes whose target is
/// effectively zero (typical for empty/zeroed fingerprints).
pub fn prose_features_to_directives(target: &ProseStyleFeatures) -> Vec<String> {
    let mut out = Vec::new();
    if target.avg_sentence_length > 0.0 {
        out.push(format!(
            "Target average sentence length: ~{:.0} words.",
            target.avg_sentence_length
        ));
    }
    if target.sentence_length_std > 0.0 {
        out.push(format!(
            "Target sentence-length variation (std): ~{:.1}.",
            target.sentence_length_std
        ));
    }
    if target.dialogue_ratio > 0.0 {
        out.push(format!(
            "Target dialogue ratio: ~{:.0}% of lines.",
            target.dialogue_ratio * 100.0
        ));
    }
    if target.passive_voice_ratio >= 0.0 {
        let qualitative = if target.passive_voice_ratio < 0.05 {
            "minimize passive voice"
        } else if target.passive_voice_ratio < 0.15 {
            "use passive voice sparingly"
        } else {
            "passive voice is acceptable"
        };
        out.push(format!(
            "Passive voice: {} (~{:.0}% of clauses).",
            qualitative,
            target.passive_voice_ratio * 100.0
        ));
    }
    if target.flesch_kincaid_grade > 0.0 {
        out.push(format!(
            "Target reading grade level (Flesch-Kincaid): ~{:.0}.",
            target.flesch_kincaid_grade
        ));
    }
    if target.type_token_ratio > 0.0 {
        out.push(format!(
            "Target lexical variety (type-token ratio): ~{:.2}.",
            target.type_token_ratio
        ));
    }
    out
}

/// Render the delta between target and actual prose features as
/// directional corrections.
///
/// Only emits constraints for axes the LLM can plausibly act on. Axes
/// like function-word frequencies or Yule's K are intentionally omitted
/// — they shape the discrimination score but not the feedback prompt.
pub fn prose_delta_to_constraints(
    target: &ProseStyleFeatures,
    actual: &ProseStyleFeatures,
) -> Vec<String> {
    let mut out = Vec::new();

    let d = actual.avg_sentence_length - target.avg_sentence_length;
    if d.abs() > FITNESS_ACTIONABLE_DELTA_AVG_SENTENCE {
        if d > 0.0 {
            out.push(format!(
                "Sentences are too long (avg {:.0}, target {:.0}) — break them into shorter ones.",
                actual.avg_sentence_length, target.avg_sentence_length
            ));
        } else {
            out.push(format!(
                "Sentences are too short (avg {:.0}, target {:.0}) — develop them with more clauses.",
                actual.avg_sentence_length, target.avg_sentence_length
            ));
        }
    }

    let d = actual.sentence_length_std - target.sentence_length_std;
    if d.abs() > FITNESS_ACTIONABLE_DELTA_SENTENCE_STD {
        if d > 0.0 {
            out.push(
                "Sentence-length variation is too high — settle into a steadier rhythm.".into(),
            );
        } else {
            out.push(
                "Sentence-length variation is too low — vary your rhythm with mixed lengths."
                    .into(),
            );
        }
    }

    let d = actual.dialogue_ratio - target.dialogue_ratio;
    if d.abs() > FITNESS_ACTIONABLE_DELTA_DIALOGUE {
        if d > 0.0 {
            out.push(format!(
                "Reduce dialogue (currently {:.0}%, target ~{:.0}%) — favor narration and description.",
                actual.dialogue_ratio * 100.0,
                target.dialogue_ratio * 100.0
            ));
        } else {
            out.push(format!(
                "Add more dialogue (currently {:.0}%, target ~{:.0}%) — let characters speak.",
                actual.dialogue_ratio * 100.0,
                target.dialogue_ratio * 100.0
            ));
        }
    }

    let d = actual.passive_voice_ratio - target.passive_voice_ratio;
    if d.abs() > FITNESS_ACTIONABLE_DELTA_PASSIVE {
        if d > 0.0 {
            out.push(format!(
                "Reduce passive voice (currently {:.0}%, target <{:.0}%) — prefer active constructions.",
                actual.passive_voice_ratio * 100.0,
                target.passive_voice_ratio * 100.0
            ));
        } else {
            out.push(format!(
                "Passive voice is below target (currently {:.0}%, target ~{:.0}%) — acceptable to use it where it suits the action.",
                actual.passive_voice_ratio * 100.0,
                target.passive_voice_ratio * 100.0
            ));
        }
    }

    let d = actual.flesch_kincaid_grade - target.flesch_kincaid_grade;
    if d.abs() > FITNESS_ACTIONABLE_DELTA_FK {
        if d > 0.0 {
            out.push(format!(
                "Use simpler language (current FK grade {:.0}, target ~{:.0}) — shorter words, clearer syntax.",
                actual.flesch_kincaid_grade, target.flesch_kincaid_grade
            ));
        } else {
            out.push(format!(
                "Elevate the prose register (current FK grade {:.0}, target ~{:.0}) — richer vocabulary, more layered syntax.",
                actual.flesch_kincaid_grade, target.flesch_kincaid_grade
            ));
        }
    }

    let d = actual.type_token_ratio - target.type_token_ratio;
    if d.abs() > FITNESS_ACTIONABLE_DELTA_TTR {
        if d > 0.0 {
            out.push(format!(
                "Lexical variety is higher than target (TTR {:.2}, target {:.2}) — repeat key vocabulary for cohesion.",
                actual.type_token_ratio, target.type_token_ratio
            ));
        } else {
            out.push(format!(
                "Lexical variety is too low (TTR {:.2}, target {:.2}) — use a wider vocabulary.",
                actual.type_token_ratio, target.type_token_ratio
            ));
        }
    }

    out
}

// ─── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_contractions() {
        let tokens = tokenize("don't won't");
        assert_eq!(tokens, vec!["don't", "won't"]);
    }

    #[test]
    fn test_tokenize_hyphens() {
        let tokens = tokenize("well-known fact");
        assert_eq!(tokens, vec!["well-known", "fact"]);
    }

    #[test]
    fn test_function_word_counting() {
        let tokens = tokenize("the the the a a");
        let freqs = function_word_frequencies(&tokens, tokens.len());
        // "the" is index 0, "a" is index 4
        assert!((freqs[0] - 0.6).abs() < 0.01); // 3/5
        assert!((freqs[4] - 0.4).abs() < 0.01); // 2/5
    }

    #[test]
    fn test_ttr_all_unique() {
        let tokens = tokenize("alpha bravo charlie delta echo foxtrot");
        let ttr = compute_ttr(&tokens);
        assert!((ttr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ttr_single_repeated() {
        let tokens = tokenize("the the the the");
        let ttr = compute_ttr(&tokens);
        assert!((ttr - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_hapax_all_unique() {
        let f = compute_prose_features("alpha bravo charlie delta echo");
        assert!((f.hapax_legomena_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_yule_k_repetitive() {
        // Very repetitive text should have high K
        let text = "the the the the the the the the the the";
        let f = compute_prose_features(text);
        // All same word: spectrum = {10: 1}, M2 = 100, K = 10000*(100-10)/100 = 9000
        assert!(f.yule_k > 100.0);
    }

    #[test]
    fn test_yule_k_diverse() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet";
        let f = compute_prose_features(text);
        // All unique: spectrum = {1: 10}, M2 = 10, K = 10000*(10-10)/100 = 0
        assert!(f.yule_k.abs() < 0.01);
    }

    #[test]
    fn test_sentence_split_basic() {
        let sents = split_sentences("Hello world. Goodbye world.");
        assert_eq!(sents.len(), 2);
    }

    #[test]
    fn test_sentence_split_abbreviations() {
        let sents = split_sentences("Mr. Smith went to Dr. Jones.");
        assert_eq!(sents.len(), 1);
    }

    #[test]
    fn test_sentence_split_dialogue() {
        let sents = split_sentences("He said \"Hello there.\" She replied \"Goodbye.\"");
        // The inner periods are followed by closing quotes, then space + uppercase
        assert!(sents.len() >= 1);
    }

    #[test]
    fn test_sentence_length_stats() {
        // "One two. Three four five." => lengths [2, 3]
        let sents = split_sentences("One two. Three four five.");
        let lengths: Vec<usize> = sents.iter().map(|s| tokenize(s).len()).collect();
        let (avg, std, _cv) = sentence_length_stats(&lengths);
        assert!((avg - 2.5).abs() < 0.01);
        assert!(std > 0.0);
    }

    #[test]
    fn test_autocorrelation_alternating() {
        // Alternating short/long sentences should give negative autocorrelation
        let lengths = vec![5usize, 20, 5, 20, 5, 20];
        let ac = lag1_autocorrelation(&lengths);
        assert!(ac < -0.3, "Expected negative autocorrelation, got {}", ac);
    }

    #[test]
    fn test_autocorrelation_constant() {
        let lengths = vec![10usize, 10, 10, 10];
        let ac = lag1_autocorrelation(&lengths);
        assert!(ac.abs() < 0.01, "Expected ~0 autocorrelation, got {}", ac);
    }

    #[test]
    fn test_syllable_counter() {
        assert_eq!(count_syllables("beautiful"), 3);
        assert_eq!(count_syllables("the"), 1);
        assert_eq!(count_syllables("create"), 2);
        assert_eq!(count_syllables("a"), 1);
    }

    #[test]
    fn test_flesch_kincaid() {
        // Simple short sentences should get a low grade level
        let f = compute_prose_features("The cat sat. The dog ran. The bird flew. A fish swam.");
        assert!(
            f.flesch_kincaid_grade < 5.0,
            "Expected low grade for simple text, got {}",
            f.flesch_kincaid_grade
        );
    }

    #[test]
    fn test_passive_voice() {
        let tokens = tokenize("The ball was thrown by the boy");
        let ratio = passive_voice_ratio(&tokens);
        assert!(
            ratio > 0.0,
            "Expected passive voice detection, got {}",
            ratio
        );
    }

    #[test]
    fn test_dialogue_ratio() {
        let text = r#""Hello there" said the man. No dialogue here."#;
        let ratio = dialogue_ratio(text);
        assert!(
            ratio > 0.1 && ratio < 0.9,
            "Expected partial dialogue ratio, got {}",
            ratio
        );
    }

    #[test]
    fn test_empty_text() {
        let f = compute_prose_features("");
        assert_eq!(f.total_words, 0);
        assert_eq!(f.total_sentences, 0);
        assert_eq!(f.type_token_ratio, 0.0);
        // No panics
    }

    #[test]
    fn test_single_word() {
        let f = compute_prose_features("hello");
        assert_eq!(f.total_words, 1);
        assert!(f.total_sentences >= 1);
        // No panics
    }

    #[test]
    fn test_burrows_delta_identical() {
        let f = compute_prose_features("The quick brown fox jumps over the lazy dog.");
        let stats = compute_corpus_stats(&[f.clone()]);
        let delta = burrows_delta(&f, &f, &stats);
        assert!(
            delta.abs() < 0.01,
            "Identical texts should have delta ~0, got {}",
            delta
        );
    }

    #[test]
    fn test_burrows_delta_different() {
        let a = compute_prose_features(
            "The the the the the the the the the the. Of of of of of of of of.",
        );
        let b =
            compute_prose_features("I I I I I I I. You you you you you you you. We we we we we.");
        let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
        let delta = burrows_delta(&a, &b, &stats);
        assert!(
            delta > 0.1,
            "Different texts should have positive delta, got {}",
            delta
        );
    }

    #[test]
    fn test_aggregate_features() {
        let a = compute_prose_features("Short text here.");
        let b = compute_prose_features(
            "A much longer piece of text with many more words in it than the first one.",
        );
        let agg = aggregate_features(&[a.clone(), b.clone()]);
        assert_eq!(agg.total_words, a.total_words + b.total_words);
        // Weighted avg should be closer to b (more words)
        let w_a = a.total_words as f32 / agg.total_words as f32;
        let w_b = b.total_words as f32 / agg.total_words as f32;
        let expected_ttr = a.type_token_ratio * w_a + b.type_token_ratio * w_b;
        assert!(
            (agg.type_token_ratio - expected_ttr).abs() < 0.01,
            "Aggregate TTR mismatch: {} vs {}",
            agg.type_token_ratio,
            expected_ttr
        );
    }

    #[test]
    fn test_compute_full_paragraph() {
        let text = "The old man sat by the river. He watched the boats drift slowly past. \
                     Some were large and beautiful, carrying passengers to distant shores. \
                     Others were small and weathered, their paint peeling in the sun. \
                     He thought about his youth, when he had sailed across the ocean. \
                     Those were wonderful days, full of adventure and possibility.";
        let f = compute_prose_features(text);
        assert!(f.total_words > 40);
        assert!(f.total_sentences >= 5);
        assert!(f.total_unique_words > 20);
        assert!(f.type_token_ratio > 0.0 && f.type_token_ratio <= 1.0);
        assert!(f.flesch_reading_ease > 0.0);
        assert!(f.avg_sentence_length > 5.0);
        assert!(f.function_word_frequencies.len() == 100);
        // "the" should appear multiple times
        assert!(f.function_word_frequencies[0] > 0.0);
    }

    #[test]
    fn test_cosine_delta() {
        let text_a = "The old man walked slowly along the dusty road. He paused and looked at the sky. The clouds were dark and threatening. He pulled his coat tighter and continued walking into the wind.";
        let text_b = "She ran quickly through the bright garden. The flowers were beautiful and the sun shone warmly. She laughed with joy as the butterflies danced around her. It was a perfect morning.";
        let text_c = "The old man walked slowly along the dusty road. He paused and looked at the sky. The clouds were dark and threatening. He pulled his coat tighter.";

        let fa = compute_prose_features(text_a);
        let fb = compute_prose_features(text_b);
        let fc = compute_prose_features(text_c);

        let corpus_stats = compute_corpus_stats(&[fa.clone(), fb.clone(), fc.clone()]);

        let delta_ab = cosine_delta(&fa, &fb, &corpus_stats);
        let delta_ac = cosine_delta(&fa, &fc, &corpus_stats);

        // a and c are more similar (same author style) than a and b
        assert!(
            delta_ac < delta_ab,
            "Expected a-c distance ({}) < a-b distance ({})",
            delta_ac,
            delta_ab
        );
        // Delta should be non-negative
        assert!(delta_ab >= 0.0);
        assert!(delta_ac >= 0.0);
    }

    #[test]
    fn test_rolling_style_window() {
        let text = "One two three four five six seven eight nine ten. \
                     Alpha beta gamma delta epsilon zeta eta theta iota kappa. \
                     Red blue green yellow orange purple pink brown black white.";

        let results = rolling_style_analysis(text, 10, 5);
        // With ~30 words, window=10, step=5: should get several windows
        assert!(
            results.len() >= 2,
            "Expected at least 2 windows, got {}",
            results.len()
        );

        // Each window should have ~10 words
        for r in &results {
            assert!(r.total_words <= 12); // may include a bit more due to whitespace splitting
            assert!(r.total_words >= 8);
        }

        // Empty/small text
        let small = rolling_style_analysis("Hello world.", 100, 50);
        assert_eq!(small.len(), 1); // entire text as one window

        // Zero params
        assert!(rolling_style_analysis("text", 0, 1).is_empty());
        assert!(rolling_style_analysis("text", 1, 0).is_empty());
    }

    fn fp(
        avg_sent: f32,
        sent_std: f32,
        dialogue: f32,
        passive: f32,
        fk: f32,
        ttr: f32,
    ) -> ProseStyleFeatures {
        let mut f = zeroed_features();
        f.avg_sentence_length = avg_sent;
        f.sentence_length_std = sent_std;
        f.dialogue_ratio = dialogue;
        f.passive_voice_ratio = passive;
        f.flesch_kincaid_grade = fk;
        f.type_token_ratio = ttr;
        f
    }

    #[test]
    fn directives_cover_actionable_axes() {
        let target = fp(12.0, 5.0, 0.40, 0.05, 8.0, 0.55);
        let out = prose_features_to_directives(&target);
        let joined = out.join(" | ");
        assert!(joined.contains("sentence length"));
        assert!(joined.contains("dialogue"));
        assert!(joined.contains("Passive voice"));
        assert!(joined.contains("Flesch-Kincaid"));
        assert!(joined.contains("type-token"));
        assert!(out.len() >= 5);
    }

    #[test]
    fn delta_emits_directional_constraints_long_to_short() {
        let target = fp(10.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        let actual = fp(25.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        let out = prose_delta_to_constraints(&target, &actual);
        let joined = out.join(" | ");
        assert!(joined.contains("too long"));
        assert!(joined.contains("shorter"));
        assert_eq!(out.len(), 1, "only one axis differs beyond threshold");
    }

    #[test]
    fn delta_emits_directional_constraints_short_to_long() {
        let target = fp(25.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        let actual = fp(10.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        let out = prose_delta_to_constraints(&target, &actual);
        let joined = out.join(" | ");
        assert!(joined.contains("too short"));
        assert!(joined.contains("develop"));
    }

    #[test]
    fn delta_omits_burrows_axes() {
        // Function-word frequencies differ wildly, but no constraints are emitted
        // because they're intentionally not in the actionable subset.
        let mut target = fp(12.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        let mut actual = fp(12.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        target.function_word_frequencies = vec![0.05, 0.04, 0.03];
        actual.function_word_frequencies = vec![0.20, 0.18, 0.15];
        target.yule_k = 100.0;
        actual.yule_k = 200.0;
        target.hapax_legomena_ratio = 0.10;
        actual.hapax_legomena_ratio = 0.50;
        let out = prose_delta_to_constraints(&target, &actual);
        assert!(out.is_empty(), "unexpected constraints: {:?}", out);
    }

    #[test]
    fn delta_within_thresholds_emits_nothing() {
        let target = fp(12.0, 4.0, 0.40, 0.05, 8.0, 0.55);
        let actual = fp(13.0, 4.5, 0.42, 0.06, 9.0, 0.57); // all within deltas
        let out = prose_delta_to_constraints(&target, &actual);
        assert!(out.is_empty(), "unexpected constraints: {:?}", out);
    }
}
