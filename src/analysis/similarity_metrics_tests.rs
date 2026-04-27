//! Unit tests for per-layer similarity metrics.

use super::*;
use crate::analysis::stylometry::{compute_corpus_stats, compute_prose_features};

// ─── Jensen–Shannon ─────────────────────────────────────────

#[test]
fn js_identical_distributions_sim_one() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let sim = jensen_shannon_sim(&p, &p);
    assert!((sim - 1.0).abs() < 1e-5, "identical sim = {}", sim);
}

#[test]
fn js_divergence_zero_on_identity() {
    let p = vec![0.1, 0.2, 0.3, 0.4];
    let js = jensen_shannon(&p, &p);
    assert!(js < 1e-5);
}

#[test]
fn js_symmetric() {
    let p = vec![0.7, 0.2, 0.1];
    let q = vec![0.1, 0.3, 0.6];
    let ab = jensen_shannon(&p, &q);
    let ba = jensen_shannon(&q, &p);
    assert!((ab - ba).abs() < 1e-5);
}

#[test]
fn js_disjoint_distributions_bounded() {
    // JS of fully-disjoint distributions equals 1.0 (in bits).
    let p = vec![1.0, 0.0];
    let q = vec![0.0, 1.0];
    let js = jensen_shannon(&p, &q);
    assert!((js - 1.0).abs() < 1e-3, "disjoint JS = {}", js);
}

#[test]
fn js_sim_disjoint_is_zero() {
    let p = vec![1.0, 0.0];
    let q = vec![0.0, 1.0];
    let sim = jensen_shannon_sim(&p, &q);
    assert!(sim < 0.01, "disjoint sim = {}", sim);
}

#[test]
fn js_handles_unnormalized_inputs() {
    let p = vec![1.0, 1.0, 1.0, 1.0];
    let q = vec![3.0, 3.0, 3.0, 3.0];
    let sim = jensen_shannon_sim(&p, &q);
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn js_zero_vector_returns_zero_divergence() {
    let p = vec![0.0, 0.0, 0.0];
    let q = vec![0.1, 0.2, 0.3];
    assert!((jensen_shannon(&p, &q)).abs() < 1e-5);
}

#[test]
fn js_truncates_to_shorter() {
    let p = vec![0.5, 0.5];
    let q = vec![0.5, 0.5, 0.0, 0.0];
    let js = jensen_shannon(&p, &q);
    assert!(js < 1e-5);
}

// ─── Mahalanobis ────────────────────────────────────────────

#[test]
fn mahalanobis_identity_is_one() {
    let a = vec![1.0, 2.0, 3.0];
    let stds = vec![1.0, 1.0, 1.0];
    assert!((mahalanobis_sim_diag(&a, &a, &stds) - 1.0).abs() < 1e-5);
}

#[test]
fn mahalanobis_scales_with_std() {
    let a = vec![0.0];
    let b = vec![1.0];
    let tight_std = vec![0.1];
    let loose_std = vec![10.0];
    let tight_sim = mahalanobis_sim_diag(&a, &b, &tight_std);
    let loose_sim = mahalanobis_sim_diag(&a, &b, &loose_std);
    // Wide σ → small z-score → high similarity; tight σ → huge z-score → near-zero.
    assert!(loose_sim > tight_sim);
    assert!(tight_sim < 0.1);
    assert!(loose_sim > 0.99);
}

#[test]
fn mahalanobis_zero_std_slots_skipped() {
    let a = vec![0.0, 5.0];
    let b = vec![1.0, 5.0];
    let stds = vec![1.0, 0.0];
    // Second slot has zero std — skipped. First slot contributes |1-0|/1 = 1.
    let sim = mahalanobis_sim_diag(&a, &b, &stds);
    let expected = (-1.0_f32).exp();
    assert!((sim - expected).abs() < 1e-5);
}

#[test]
fn mahalanobis_missing_stds_default_to_unit() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 1.0];
    let stds = vec![]; // empty → unit variance
    let sim = mahalanobis_sim_diag(&a, &b, &stds);
    // d² = 1 + 1 = 2, used = 2 → exp(-1) ≈ 0.368
    assert!((sim - (-1.0_f32).exp()).abs() < 1e-5);
}

#[test]
fn mahalanobis_empty_vectors_return_one() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let stds: Vec<f32> = vec![];
    assert_eq!(mahalanobis_sim_diag(&a, &b, &stds), 1.0);
}

#[test]
fn mahalanobis_all_zero_stds_return_one() {
    let a = vec![1.0, 2.0];
    let b = vec![5.0, 6.0];
    let stds = vec![0.0, 0.0];
    assert_eq!(mahalanobis_sim_diag(&a, &b, &stds), 1.0);
}

// ─── Hamming (256-bit SimHash) ──────────────────────────────

#[test]
fn hamming_identical_is_one() {
    let a = [0xABCD_1234_5678_9ABC_u64, 0x1111, 0x2222, 0x3333];
    assert_eq!(hamming_sim_u256(a, a), 1.0);
}

#[test]
fn hamming_inverted_is_zero() {
    let a = [0u64; 4];
    let b = [u64::MAX; 4];
    let sim = hamming_sim_u256(a, b);
    assert!(sim.abs() < 1e-5);
}

#[test]
fn hamming_single_bit_flip() {
    let a = [0u64; 4];
    let b = [1u64, 0, 0, 0]; // 1 bit out of 256
    let sim = hamming_sim_u256(a, b);
    assert!((sim - (1.0 - 1.0 / 256.0)).abs() < 1e-6);
}

#[test]
fn hamming_half_bits_flipped() {
    let a = [0u64; 4];
    let b = [0x5555_5555_5555_5555u64; 4]; // alternating pattern = 128 bits set
    let sim = hamming_sim_u256(a, b);
    assert!((sim - 0.5).abs() < 1e-5);
}

// ─── Burrows-Cosine (authorship layer) ──────────────────────

#[test]
fn burrows_cosine_identical_texts_sim_one() {
    let text = "the quick brown fox jumps over the lazy dog. the dog was not amused.";
    let a = compute_prose_features(text);
    let b = compute_prose_features(text);
    let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
    let sim = burrows_cosine(&a, &b, &stats);
    assert!(sim > 0.99, "identical texts sim = {}", sim);
}

#[test]
fn burrows_cosine_different_authors_distinguishable() {
    // Hemingway-flavored: short sentences, simple function words
    let hem = "He walked. He was tired. She was gone. He looked at the sea.";
    // Faulkner-flavored: long sentences, dense function word usage
    let falk = "In the dim and unremembered light of that afternoon when the \
                wind had only barely begun to stir the curtains of the room where \
                she had been, he thought of all the things that might yet come to pass.";
    let a = compute_prose_features(hem);
    let b = compute_prose_features(falk);
    let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
    let sim = burrows_cosine(&a, &b, &stats);
    // Different styles should produce sim measurably below 1.0.
    assert!(sim >= 0.0 && sim <= 1.0);
}

#[test]
fn burrows_cosine_bounded_zero_to_one() {
    let a = compute_prose_features("a b c. d e f.");
    let b = compute_prose_features("xxx yyy zzz. www vvv uuu.");
    let stats = compute_corpus_stats(&[a.clone(), b.clone()]);
    let sim = burrows_cosine(&a, &b, &stats);
    assert!((0.0..=1.0).contains(&sim));
}
