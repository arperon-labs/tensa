//! Comprehensive correctness + axiom tests for the canonical t-norm /
//! t-conorm families. Runs under `cargo test --lib fuzzy::tnorm_tests`.
//!
//! Cites: [klement2000].

use super::tnorm::{
    combine_tconorm, combine_tnorm, reduce_tconorm, reduce_tnorm, tconorm_for, tnorm_for,
    TNormKind,
};

const ALL: [TNormKind; 4] = [
    TNormKind::Godel,
    TNormKind::Goguen,
    TNormKind::Lukasiewicz,
    TNormKind::Hamacher(0.5),
];

// ── Per-family correctness tests ──────────────────────────────────────────────

#[test]
fn test_godel_combine() {
    let t = tnorm_for(TNormKind::Godel);
    let s = tconorm_for(TNormKind::Godel);
    // T = min
    assert_eq!(t.combine(0.0, 1.0), 0.0);
    assert_eq!(t.combine(1.0, 0.0), 0.0);
    assert_eq!(t.combine(0.3, 0.7), 0.3);
    assert_eq!(t.combine(0.5, 0.5), 0.5);
    assert_eq!(t.combine(1.0, 1.0), 1.0);
    // S = max
    assert_eq!(s.combine(0.0, 1.0), 1.0);
    assert_eq!(s.combine(0.3, 0.7), 0.7);
}

#[test]
fn test_goguen_combine() {
    let t = tnorm_for(TNormKind::Goguen);
    let s = tconorm_for(TNormKind::Goguen);
    // T = a*b
    assert_eq!(t.combine(0.0, 1.0), 0.0);
    assert_eq!(t.combine(0.5, 0.5), 0.25);
    assert!((t.combine(0.3, 0.7) - 0.21).abs() < 1e-12);
    assert_eq!(t.combine(1.0, 1.0), 1.0);
    // S = a+b-ab
    assert!((s.combine(0.3, 0.7) - (0.3 + 0.7 - 0.21)).abs() < 1e-12);
    assert_eq!(s.combine(0.0, 0.0), 0.0);
    assert_eq!(s.combine(1.0, 0.5), 1.0);
}

#[test]
fn test_lukasiewicz_combine() {
    let t = tnorm_for(TNormKind::Lukasiewicz);
    let s = tconorm_for(TNormKind::Lukasiewicz);
    // T = max(0, a+b-1) — saturates to 0
    assert_eq!(t.combine(0.0, 1.0), 0.0);
    assert_eq!(t.combine(0.3, 0.3), 0.0); // saturated
    assert!((t.combine(0.7, 0.7) - 0.4).abs() < 1e-12);
    assert_eq!(t.combine(1.0, 1.0), 1.0);
    // S = min(1, a+b) — saturates at 1
    assert_eq!(s.combine(0.7, 0.7), 1.0); // saturated
    assert!((s.combine(0.3, 0.4) - 0.7).abs() < 1e-12);
    assert_eq!(s.combine(0.0, 0.0), 0.0);
}

#[test]
fn test_hamacher_combine() {
    // λ=0 → Hamacher product: ab / (a + b - ab)
    let t0 = tnorm_for(TNormKind::Hamacher(0.0));
    let v = t0.combine(0.3, 0.7);
    let expected = 0.21 / (0.3 + 0.7 - 0.21);
    assert!((v - expected).abs() < 1e-12, "got {}, expected {}", v, expected);

    // λ=1 → Goguen (product) within 1e-12
    let t1 = tnorm_for(TNormKind::Hamacher(1.0));
    let goguen = tnorm_for(TNormKind::Goguen);
    for &(a, b) in &[(0.2, 0.8), (0.5, 0.5), (0.3, 0.7), (0.9, 0.1)] {
        let h = t1.combine(a, b);
        let g = goguen.combine(a, b);
        assert!((h - g).abs() < 1e-12, "λ=1 must equal Goguen: {} vs {}", h, g);
    }

    // λ=0.5 mid-way
    let th = tnorm_for(TNormKind::Hamacher(0.5));
    let a = 0.3;
    let b = 0.7;
    let expected = (a * b) / (0.5 + 0.5 * (a + b - a * b));
    assert!((th.combine(a, b) - expected).abs() < 1e-12);
}

#[test]
fn test_hamacher_recovers_goguen_at_lambda_one() {
    let goguen = tnorm_for(TNormKind::Goguen);
    let ham = tnorm_for(TNormKind::Hamacher(1.0));
    // 11×11 grid
    for i in 0..=10u32 {
        for j in 0..=10u32 {
            let a = i as f64 / 10.0;
            let b = j as f64 / 10.0;
            let g = goguen.combine(a, b);
            let h = ham.combine(a, b);
            assert!(
                (g - h).abs() < 1e-12,
                "Hamacher(λ=1) ≠ Goguen at ({},{}): {} vs {}",
                a,
                b,
                h,
                g
            );
        }
    }
}

// ── Boundary identity axioms ─────────────────────────────────────────────────

#[test]
fn test_boundary_identities() {
    let pts = [0.0, 0.25, 0.5, 0.75, 1.0];
    for k in ALL {
        let t = tnorm_for(k);
        let s = tconorm_for(k);
        for &a in &pts {
            // T(a, 0) = 0
            assert!(
                t.combine(a, 0.0).abs() < 1e-12,
                "{}: T({},0) != 0",
                t.name(),
                a
            );
            // T(a, 1) = a
            assert!(
                (t.combine(a, 1.0) - a).abs() < 1e-12,
                "{}: T({},1) != {}",
                t.name(),
                a,
                a
            );
            // S(a, 0) = a
            assert!(
                (s.combine(a, 0.0) - a).abs() < 1e-12,
                "{}: S({},0) != {}",
                s.name(),
                a,
                a
            );
            // S(a, 1) = 1
            assert!(
                (s.combine(a, 1.0) - 1.0).abs() < 1e-12,
                "{}: S({},1) != 1",
                s.name(),
                a
            );
        }
    }
}

#[test]
fn test_commutativity_monotonicity() {
    let pts = [0.1, 0.3, 0.5, 0.7, 0.9];
    for k in ALL {
        let t = tnorm_for(k);
        // Commutativity
        for &a in &pts {
            for &b in &pts {
                let ab = t.combine(a, b);
                let ba = t.combine(b, a);
                assert!((ab - ba).abs() < 1e-12, "{}: not commutative", t.name());
            }
        }
        // Monotonicity: a1 <= a2 ⇒ T(a1, b) <= T(a2, b)
        for &b in &pts {
            for i in 0..pts.len() {
                for j in i..pts.len() {
                    let a1 = pts[i];
                    let a2 = pts[j];
                    let lo = t.combine(a1, b);
                    let hi = t.combine(a2, b);
                    assert!(
                        lo <= hi + 1e-12,
                        "{}: non-monotone: T({},{})={} > T({},{})={}",
                        t.name(),
                        a1,
                        b,
                        lo,
                        a2,
                        b,
                        hi
                    );
                }
            }
        }
    }
}

#[test]
fn test_de_morgan_duality() {
    // S(a, b) == 1 - T(1-a, 1-b) within 1e-9.
    let pts = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0];
    for k in ALL {
        let t = tnorm_for(k);
        let s = tconorm_for(k);
        for &a in &pts {
            for &b in &pts {
                let lhs = s.combine(a, b);
                let rhs = 1.0 - t.combine(1.0 - a, 1.0 - b);
                assert!(
                    (lhs - rhs).abs() < 1e-9,
                    "{}: De Morgan failed at ({},{}): S={} vs 1-T(1-a,1-b)={}",
                    s.name(),
                    a,
                    b,
                    lhs,
                    rhs
                );
            }
        }
    }
}

#[test]
fn test_ordering() {
    // Łukasiewicz ≤ Goguen ≤ Gödel at 36 points.
    let pts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let lu = tnorm_for(TNormKind::Lukasiewicz);
    let gu = tnorm_for(TNormKind::Goguen);
    let go = tnorm_for(TNormKind::Godel);
    for &a in &pts {
        for &b in &pts {
            let l = lu.combine(a, b);
            let g = gu.combine(a, b);
            let d = go.combine(a, b);
            assert!(
                l <= g + 1e-12,
                "Łukasiewicz > Goguen at ({},{}): {} > {}",
                a,
                b,
                l,
                g
            );
            assert!(
                g <= d + 1e-12,
                "Goguen > Gödel at ({},{}): {} > {}",
                a,
                b,
                g,
                d
            );
        }
    }
}

// ── Reduction helpers ─────────────────────────────────────────────────────────

#[test]
fn test_reduce_neutral_element() {
    for k in ALL {
        assert_eq!(reduce_tnorm(k, &[]), 1.0, "{}: empty t-norm reduce != 1", k.name());
        assert_eq!(
            reduce_tconorm(k, &[]),
            0.0,
            "{}: empty t-conorm reduce != 0",
            k.name()
        );
    }
}

#[test]
fn test_reduce_single_element() {
    // T(neutral, x) == x — the fold collapses to the element itself.
    for k in ALL {
        let x = 0.42;
        assert!((reduce_tnorm(k, &[x]) - x).abs() < 1e-12);
        assert!((reduce_tconorm(k, &[x]) - x).abs() < 1e-12);
    }
}

#[test]
fn test_reduce_three_elements() {
    // Associativity: T(T(a, b), c) == T(a, T(b, c)).
    let a = 0.3;
    let b = 0.5;
    let c = 0.8;
    for k in ALL {
        let left = combine_tnorm(k, combine_tnorm(k, a, b), c);
        let right = combine_tnorm(k, a, combine_tnorm(k, b, c));
        assert!(
            (left - right).abs() < 1e-9,
            "{}: t-norm not associative",
            k.name()
        );
        let reduced = reduce_tnorm(k, &[a, b, c]);
        assert!(
            (reduced - left).abs() < 1e-9,
            "{}: reduce != repeated combine",
            k.name()
        );
    }
}

#[test]
fn test_clamps_out_of_range_inputs() {
    // Defensive clamp to [0, 1] for every family.
    for k in ALL {
        let t = tnorm_for(k);
        let s = tconorm_for(k);
        // Values outside [0, 1] should not produce NaN/Inf; outputs stay bounded.
        let v = t.combine(-0.5, 0.5);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        let v = s.combine(1.5, 0.2);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        // NaN collapses to 0.
        let v = t.combine(f64::NAN, 0.5);
        assert!(v.is_finite());
    }
}
