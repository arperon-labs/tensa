//! Natural language generation metrics: BLEU-1/4, ROUGE-L, METEOR.
//!
//! Used for NarrativeQA and other generative QA benchmarks.
//! Pure Rust implementations following standard definitions.

use std::collections::HashMap;

/// Tokenize a string into lowercase words.
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Count n-grams of a given order in a token list.
fn ngram_counts(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut counts = HashMap::new();
    if tokens.len() < n {
        return counts;
    }
    for window in tokens.windows(n) {
        *counts.entry(window.to_vec()).or_insert(0) += 1;
    }
    counts
}

/// Clipped n-gram precision: count of candidate n-grams found in reference,
/// clipped by reference count, divided by total candidate n-grams.
fn clipped_precision(candidate: &[String], reference: &[String], n: usize) -> f64 {
    if candidate.len() < n {
        return 0.0;
    }
    let cand_counts = ngram_counts(candidate, n);
    let ref_counts = ngram_counts(reference, n);

    let mut clipped_sum = 0usize;
    let mut total = 0usize;

    for (ngram, &cand_count) in &cand_counts {
        let ref_count = ref_counts.get(ngram).copied().unwrap_or(0);
        clipped_sum += cand_count.min(ref_count);
        total += cand_count;
    }

    if total == 0 {
        0.0
    } else {
        clipped_sum as f64 / total as f64
    }
}

/// Brevity penalty for BLEU.
fn brevity_penalty(candidate_len: usize, reference_len: usize) -> f64 {
    if candidate_len == 0 {
        return 0.0;
    }
    if candidate_len >= reference_len {
        1.0
    } else {
        (1.0 - reference_len as f64 / candidate_len as f64).exp()
    }
}

/// BLEU-1: unigram precision with brevity penalty.
///
/// When multiple references are available, use the one that gives the best score.
pub fn bleu_1(candidate: &str, references: &[&str]) -> f64 {
    let cand_tokens = tokenize(candidate);
    if cand_tokens.is_empty() {
        return 0.0;
    }

    references
        .iter()
        .map(|r| {
            let ref_tokens = tokenize(r);
            let p1 = clipped_precision(&cand_tokens, &ref_tokens, 1);
            let bp = brevity_penalty(cand_tokens.len(), ref_tokens.len());
            bp * p1
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// BLEU-4: geometric mean of 1-4 gram precisions with brevity penalty.
///
/// Standard BLEU score used in machine translation and QA evaluation.
pub fn bleu_4(candidate: &str, references: &[&str]) -> f64 {
    let cand_tokens = tokenize(candidate);
    if cand_tokens.len() < 4 {
        // 4-gram precision undefined on shorter candidates — fall back to unigram.
        return bleu_1(candidate, references);
    }

    references
        .iter()
        .map(|r| {
            let ref_tokens = tokenize(r);
            let mut log_sum = 0.0;
            let mut valid_n = 0;

            for n in 1..=4 {
                let p = clipped_precision(&cand_tokens, &ref_tokens, n);
                if p > 0.0 {
                    log_sum += p.ln();
                    valid_n += 1;
                } else {
                    // Smoothing: add epsilon to avoid log(0)
                    log_sum += (1e-10_f64).ln();
                    valid_n += 1;
                }
            }

            let bp = brevity_penalty(cand_tokens.len(), ref_tokens.len());
            bp * (log_sum / valid_n as f64).exp()
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// ROUGE-L: F1 based on longest common subsequence.
pub fn rouge_l(candidate: &str, reference: &str) -> f64 {
    let cand_tokens = tokenize(candidate);
    let ref_tokens = tokenize(reference);

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let lcs_len = lcs_length(&cand_tokens, &ref_tokens);
    if lcs_len == 0 {
        return 0.0;
    }

    let precision = lcs_len as f64 / cand_tokens.len() as f64;
    let recall = lcs_len as f64 / ref_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

/// Longest common subsequence length (O(n*m) DP).
fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    // Space-optimized: two rows instead of full matrix
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.iter_mut().for_each(|x| *x = 0);
    }

    *prev.iter().max().unwrap_or(&0)
}

/// METEOR: unigram matching with stemming approximation.
///
/// Simplified METEOR: exact unigram match with harmonic mean weighting.
/// Full METEOR would require a stemmer and synonym tables; this is a
/// reasonable approximation for benchmark reporting.
pub fn meteor(candidate: &str, reference: &str) -> f64 {
    let cand_tokens = tokenize(candidate);
    let ref_tokens = tokenize(reference);

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    // Count unigram matches (multiset intersection)
    let mut ref_counts: HashMap<&str, usize> = HashMap::new();
    for t in &ref_tokens {
        *ref_counts.entry(t.as_str()).or_insert(0) += 1;
    }

    let mut matches = 0usize;
    for t in &cand_tokens {
        if let Some(count) = ref_counts.get_mut(t.as_str()) {
            if *count > 0 {
                matches += 1;
                *count -= 1;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let precision = matches as f64 / cand_tokens.len() as f64;
    let recall = matches as f64 / ref_tokens.len() as f64;

    // METEOR uses a harmonic mean weighted toward recall (alpha=0.9)
    let alpha = 0.9;
    let f_mean = 1.0 / (alpha / precision + (1.0 - alpha) / recall);

    // Penalty for fragmentation (simplified: based on chunk count)
    let chunks = count_chunks(&cand_tokens, &ref_tokens);
    let frag = if matches > 0 {
        chunks as f64 / matches as f64
    } else {
        0.0
    };
    let penalty = 0.5 * frag.powi(3); // gamma=0.5, beta=3

    f_mean * (1.0 - penalty).max(0.0)
}

/// Count contiguous chunks of matched tokens (for METEOR fragmentation penalty).
fn count_chunks(candidate: &[String], reference: &[String]) -> usize {
    if candidate.is_empty() || reference.is_empty() {
        return 0;
    }

    // Build alignment: for each candidate token, find its position in reference
    let mut ref_available: Vec<Option<usize>> =
        reference.iter().enumerate().map(|(i, _)| Some(i)).collect();

    let mut alignment: Vec<Option<usize>> = Vec::new();
    for ct in candidate {
        let mut matched = None;
        for (ri, ra) in ref_available.iter_mut().enumerate() {
            if ra.is_some() && reference[ri] == *ct {
                matched = Some(ri);
                *ra = None;
                break;
            }
        }
        alignment.push(matched);
    }

    // Count chunks: sequences of consecutive aligned positions
    let mut chunks = 0;
    let mut prev_ref_pos: Option<usize> = None;
    for aligned_pos in &alignment {
        if let Some(pos) = aligned_pos {
            match prev_ref_pos {
                Some(prev) if *pos == prev + 1 => {
                    // Continuation of current chunk
                }
                _ => {
                    // New chunk
                    chunks += 1;
                }
            }
            prev_ref_pos = Some(*pos);
        } else {
            prev_ref_pos = None;
        }
    }

    chunks
}

/// Aggregate NLG metrics over multiple (candidate, references) pairs.
pub fn aggregate_nlg_metrics(pairs: &[(&str, &[&str])]) -> NlgMetrics {
    if pairs.is_empty() {
        return NlgMetrics::default();
    }
    let n = pairs.len() as f64;
    let b1 = pairs.iter().map(|(c, rs)| bleu_1(c, rs)).sum::<f64>() / n;
    let b4 = pairs.iter().map(|(c, rs)| bleu_4(c, rs)).sum::<f64>() / n;
    // ROUGE-L and METEOR use first reference
    let rl = pairs
        .iter()
        .map(|(c, rs)| {
            if rs.is_empty() {
                0.0
            } else {
                rouge_l(c, rs[0])
            }
        })
        .sum::<f64>()
        / n;
    let met = pairs
        .iter()
        .map(|(c, rs)| if rs.is_empty() { 0.0 } else { meteor(c, rs[0]) })
        .sum::<f64>()
        / n;

    NlgMetrics {
        bleu_1: b1,
        bleu_4: b4,
        rouge_l: rl,
        meteor: met,
    }
}

/// Aggregated NLG metrics.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct NlgMetrics {
    pub bleu_1: f64,
    pub bleu_4: f64,
    pub rouge_l: f64,
    pub meteor: f64,
}

impl NlgMetrics {
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "bleu_1": self.bleu_1,
            "bleu_4": self.bleu_4,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bleu_1_identical() {
        let score = bleu_1("the cat sat on the mat", &["the cat sat on the mat"]);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Identical should give BLEU-1=1.0, got {}",
            score
        );
    }

    #[test]
    fn test_bleu_1_no_overlap() {
        let score = bleu_1("hello world", &["foo bar baz"]);
        assert!(score < 0.01, "No overlap should give ~0, got {}", score);
    }

    #[test]
    fn test_bleu_4_short_candidate() {
        // Candidate shorter than 4 tokens falls back to BLEU-1
        let score = bleu_4("cat sat", &["the cat sat on the mat"]);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_rouge_l_identical() {
        let score = rouge_l("the cat sat on the mat", "the cat sat on the mat");
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Identical should give ROUGE-L=1.0, got {}",
            score
        );
    }

    #[test]
    fn test_rouge_l_partial() {
        // LCS of "the cat sat" and "the cat sat on the mat" = "the cat sat" (3 tokens)
        // P=3/3=1.0, R=3/6=0.5, F1=2*1.0*0.5/1.5 = 0.667
        let score = rouge_l("the cat sat", "the cat sat on the mat");
        assert!(
            (score - 0.667).abs() < 0.01,
            "Expected ~0.667, got {}",
            score
        );
    }

    #[test]
    fn test_rouge_l_no_overlap() {
        let score = rouge_l("hello world", "foo bar baz");
        assert!(score < 0.01);
    }

    #[test]
    fn test_meteor_identical() {
        let score = meteor("the cat sat on the mat", "the cat sat on the mat");
        assert!(
            score > 0.9,
            "Identical should give high METEOR, got {}",
            score
        );
    }

    #[test]
    fn test_meteor_no_overlap() {
        let score = meteor("hello world", "foo bar baz");
        assert!(score < 0.01);
    }

    #[test]
    fn test_aggregate_nlg() {
        let pairs: Vec<(&str, &[&str])> = vec![
            ("the cat sat", &["the cat sat on the mat"]),
            ("hello world", &["hello world"]),
        ];
        let metrics = aggregate_nlg_metrics(&pairs);
        assert!(metrics.bleu_1 > 0.0);
        assert!(metrics.rouge_l > 0.0);
    }
}
