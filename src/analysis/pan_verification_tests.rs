//! Unit tests for PAN authorship verification metrics.

use super::*;

fn score(id: &str, score: f32, decision: Option<bool>) -> VerificationScore {
    VerificationScore {
        id: id.into(),
        score,
        decision,
    }
}

// ─── AUC ────────────────────────────────────────────────────

#[test]
fn auc_perfect_separation_is_one() {
    // All positives have higher scores than negatives → AUC = 1.
    let scores = vec![
        score("1", 0.9, Some(true)),
        score("2", 0.8, Some(true)),
        score("3", 0.3, Some(false)),
        score("4", 0.1, Some(false)),
    ];
    let labels = vec![true, true, false, false];
    let a = auc(&scores, &labels);
    assert!((a - 1.0).abs() < 1e-3, "auc = {}", a);
}

#[test]
fn auc_inverse_separation_is_zero() {
    let scores = vec![
        score("1", 0.1, Some(false)),
        score("2", 0.2, Some(false)),
        score("3", 0.8, Some(true)),
        score("4", 0.9, Some(true)),
    ];
    let labels = vec![true, true, false, false]; // inverted
    let a = auc(&scores, &labels);
    assert!(a < 0.01, "auc = {}", a);
}

#[test]
fn auc_tied_scores_bounded() {
    // Tied scores: any AUC is valid as long as it's in [0, 1].
    let scores = vec![score("1", 0.5, Some(true)), score("2", 0.5, Some(false))];
    let labels = vec![true, false];
    let a = auc(&scores, &labels);
    assert!(
        (0.0..=1.0).contains(&a),
        "auc should be in [0,1], got {}",
        a
    );
}

#[test]
fn auc_all_one_class_returns_half() {
    let scores = vec![score("1", 0.7, Some(true)), score("2", 0.8, Some(true))];
    let labels = vec![true, true];
    assert_eq!(auc(&scores, &labels), 0.5);
}

// ─── c@1 ────────────────────────────────────────────────────

#[test]
fn c_at_1_all_correct_is_one() {
    let scores = vec![score("1", 0.9, Some(true)), score("2", 0.1, Some(false))];
    let labels = vec![true, false];
    assert!((c_at_1(&scores, &labels) - 1.0).abs() < 1e-5);
}

#[test]
fn c_at_1_all_wrong_is_zero() {
    let scores = vec![score("1", 0.9, Some(false)), score("2", 0.1, Some(true))];
    let labels = vec![true, false];
    assert!(c_at_1(&scores, &labels).abs() < 1e-5);
}

#[test]
fn c_at_1_abstention_partial_credit() {
    // 1 correct, 1 abstention, 1 wrong → (1 + 1 * 1/3) / 3 = 4/9 ≈ 0.444
    let scores = vec![
        score("1", 0.9, Some(true)),
        score("2", 0.5, None),
        score("3", 0.9, Some(false)),
    ];
    let labels = vec![true, true, true];
    let c = c_at_1(&scores, &labels);
    let expected = (1.0 + 1.0 / 3.0) / 3.0;
    assert!((c - expected).abs() < 1e-4, "c@1 = {} vs {}", c, expected);
}

// ─── F0.5u ──────────────────────────────────────────────────

#[test]
fn f_0_5_u_perfect_is_one() {
    let scores = vec![score("1", 0.9, Some(true)), score("2", 0.1, Some(false))];
    let labels = vec![true, false];
    let f = f_0_5_u(&scores, &labels);
    assert!((f - 1.0).abs() < 1e-4);
}

#[test]
fn f_0_5_u_bounded() {
    let scores = vec![
        score("1", 0.9, Some(true)),
        score("2", 0.5, None),
        score("3", 0.1, Some(false)),
    ];
    let labels = vec![false, true, true];
    let f = f_0_5_u(&scores, &labels);
    assert!((0.0..=1.0).contains(&f));
}

// ─── F1 ─────────────────────────────────────────────────────

#[test]
fn f1_perfect_is_one() {
    let scores = vec![score("1", 0.9, Some(true)), score("2", 0.1, Some(false))];
    let labels = vec![true, false];
    assert!((f1(&scores, &labels) - 1.0).abs() < 1e-4);
}

// ─── Brier ──────────────────────────────────────────────────

#[test]
fn brier_perfect_is_zero() {
    let scores = vec![score("1", 1.0, Some(true)), score("2", 0.0, Some(false))];
    let labels = vec![true, false];
    assert!(brier_score(&scores, &labels).abs() < 1e-5);
}

#[test]
fn brier_worst_is_one() {
    let scores = vec![score("1", 0.0, Some(false)), score("2", 1.0, Some(true))];
    let labels = vec![true, false];
    assert!((brier_score(&scores, &labels) - 1.0).abs() < 1e-5);
}

// ─── Overall evaluate ──────────────────────────────────────

#[test]
fn evaluate_returns_bounded_metrics() {
    let scores = vec![
        score("1", 0.9, Some(true)),
        score("2", 0.2, Some(false)),
        score("3", 0.5, None),
        score("4", 0.8, Some(true)),
    ];
    let labels = vec![true, false, true, true];
    let m = evaluate(&scores, &labels);
    assert!((0.0..=1.0).contains(&m.auc));
    assert!((0.0..=1.0).contains(&m.c_at_1));
    assert!((0.0..=1.0).contains(&m.f_0_5_u));
    assert!((0.0..=1.0).contains(&m.f1));
    assert!((0.0..=1.0).contains(&m.brier));
    assert!((0.0..=1.0).contains(&m.overall));
    assert_eq!(m.n_samples, 4);
    assert_eq!(m.n_decisions, 3);
}

// ─── verify_pair integration ────────────────────────────────

#[test]
fn verify_pair_identical_text_high_score() {
    let text = "he walked down the hallway. she was not there. the door closed.";
    let a = compute_prose_features(text);
    let b = compute_prose_features(text);
    let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
    let cfg = VerificationConfig::default();
    let s = verify_pair(&a, &b, &stats, &cfg);
    assert!(s > 0.7, "identical text verifier score = {}", s);
}

#[test]
fn verify_pair_decision_abstains_near_half() {
    // Pick a score that lands near 0.5 by using a config with intercept roughly
    // centered. Use max-uncertainty-band config.
    let a = compute_prose_features("a b c d e");
    let b = compute_prose_features("f g h i j");
    let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
    let cfg = VerificationConfig {
        intercept: 0.0,
        w_burrows_cosine: 0.0,
        w_sentence_length_diff: 0.0,
        w_readability_diff: 0.0,
        w_dialogue_diff: 0.0,
        uncertainty_band: 0.2,
    };
    let (score, decision) = verify_pair_with_decision(&a, &b, &stats, &cfg);
    assert!((score - 0.5).abs() < 1e-4);
    assert!(decision.is_none());
}

// ─── Unmasking ──────────────────────────────────────────────

#[test]
fn unmasking_same_author_shallow_decline() {
    // Give both sides identical corpus → classifier has nothing to latch onto.
    let base = compute_prose_features(
        "the man walked. the woman was there. he looked at her. she was waiting.",
    );
    let chunks = vec![base.clone(), base.clone(), base.clone(), base.clone()];
    let curve = unmasking(&chunks, &chunks, 3, 5);
    // Same-author: slope should not be strongly negative.
    assert!(curve.slope > -0.2, "same-author slope = {}", curve.slope);
}

#[test]
fn unmasking_insufficient_chunks_returns_flat() {
    let single = compute_prose_features("hello world");
    let curve = unmasking(&[single.clone()], &[single], 3, 5);
    assert_eq!(curve.accuracies.len(), 1);
    assert_eq!(curve.slope, 0.0);
}
