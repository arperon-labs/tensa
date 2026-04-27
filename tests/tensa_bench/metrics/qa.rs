//! QA metrics: Exact Match (EM) and token-level F1.
//!
//! Standard HotpotQA / SQuAD evaluation metrics.
//! Normalization follows the official HotpotQA eval script.

/// Normalize an answer string for comparison.
///
/// Steps: lowercase, strip articles (a/an/the), strip punctuation, collapse whitespace.
pub fn normalize_answer(s: &str) -> String {
    let lowered = s.to_lowercase();

    // Strip punctuation
    let no_punct: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();

    // Strip articles
    let tokens: Vec<&str> = no_punct
        .split_whitespace()
        .filter(|t| !matches!(*t, "a" | "an" | "the"))
        .collect();

    tokens.join(" ")
}

/// Exact match after normalization.
pub fn exact_match(prediction: &str, gold: &str) -> bool {
    normalize_answer(prediction) == normalize_answer(gold)
}

/// Token-level F1 score (bag-of-words intersection).
pub fn token_f1(prediction: &str, gold: &str) -> f64 {
    let pred_tokens: Vec<String> = normalize_answer(prediction)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let gold_tokens: Vec<String> = normalize_answer(gold)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    if pred_tokens.is_empty() && gold_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || gold_tokens.is_empty() {
        return 0.0;
    }

    // Count shared tokens (multiset intersection)
    let mut gold_counts = std::collections::HashMap::new();
    for t in &gold_tokens {
        *gold_counts.entry(t.as_str()).or_insert(0usize) += 1;
    }

    let mut common = 0usize;
    for t in &pred_tokens {
        if let Some(count) = gold_counts.get_mut(t.as_str()) {
            if *count > 0 {
                common += 1;
                *count -= 1;
            }
        }
    }

    if common == 0 {
        return 0.0;
    }

    let precision = common as f64 / pred_tokens.len() as f64;
    let recall = common as f64 / gold_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

/// Compute EM and F1 averaged over multiple (prediction, gold) pairs.
pub fn aggregate_qa_metrics(pairs: &[(&str, &str)]) -> (f64, f64) {
    if pairs.is_empty() {
        return (0.0, 0.0);
    }
    let n = pairs.len() as f64;
    let em = pairs.iter().filter(|(p, g)| exact_match(p, g)).count() as f64 / n;
    let f1 = pairs.iter().map(|(p, g)| token_f1(p, g)).sum::<f64>() / n;
    (em, f1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_strips_articles_and_punct() {
        assert_eq!(normalize_answer("The cat's hat."), "cats hat");
    }

    #[test]
    fn test_exact_match_basic() {
        assert!(exact_match("New York City", "new york city"));
        assert!(exact_match("The answer is 42.", "answer is 42"));
        assert!(!exact_match("yes", "no"));
    }

    #[test]
    fn test_exact_match_empty() {
        assert!(exact_match("", ""));
        assert!(!exact_match("hello", ""));
    }

    #[test]
    fn test_token_f1_perfect() {
        let f1 = token_f1("the quick brown fox", "quick brown fox the");
        assert!(
            (f1 - 1.0).abs() < 1e-9,
            "Same tokens should give F1=1.0, got {}",
            f1
        );
    }

    #[test]
    fn test_token_f1_partial() {
        // prediction: "quick brown" (2 tokens), gold: "quick brown fox" (3 tokens)
        // common=2, precision=2/2=1.0, recall=2/3=0.667
        // F1 = 2*1.0*0.667/(1.0+0.667) = 0.8
        let f1 = token_f1("quick brown", "quick brown fox");
        assert!((f1 - 0.8).abs() < 0.01, "Expected ~0.8, got {}", f1);
    }

    #[test]
    fn test_token_f1_no_overlap() {
        let f1 = token_f1("hello world", "foo bar");
        assert!((f1).abs() < 1e-9);
    }

    #[test]
    fn test_token_f1_empty() {
        assert!((token_f1("", "")).abs() < 1e-9 || (token_f1("", "") - 1.0).abs() < 1e-9);
        assert!((token_f1("hello", "")).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_metrics() {
        let pairs = vec![
            ("New York", "New York"),      // EM=true, F1=1.0
            ("the big apple", "new york"), // EM=false, F1=0.0
        ];
        let (em, f1) = aggregate_qa_metrics(&pairs);
        assert!((em - 0.5).abs() < 1e-9);
        assert!(f1 >= 0.0 && f1 <= 1.0);
    }
}
