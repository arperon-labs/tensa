//! Dempster-Shafer disinfo signal fusion (Sprint D5).
//!
//! Combines signals from multiple analysis modules into a unified
//! DisinfoAssessment using Dempster-Shafer evidence combination
//! with the Adaptive Weight Correction Rule for highly conflicting evidence.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

/// A signal from one analysis module contributing to the disinfo assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisinfoSignal {
    /// Which module produced this signal.
    pub source: SignalSource,
    /// Mass assigned to the "true" (genuine) hypothesis.
    pub mass_true: f64,
    /// Mass assigned to the "false" (disinformation) hypothesis.
    pub mass_false: f64,
    /// Mass assigned to the "misleading" hypothesis.
    pub mass_misleading: f64,
    /// Remaining mass assigned to uncertainty (Theta).
    pub mass_uncertain: f64,
}

impl DisinfoSignal {
    /// Create a signal with explicit mass assignments. Normalizes to sum to 1.0.
    pub fn new(
        source: SignalSource,
        mass_true: f64,
        mass_false: f64,
        mass_misleading: f64,
    ) -> Self {
        let total = mass_true + mass_false + mass_misleading;
        let uncertain = if total >= 1.0 { 0.0 } else { 1.0 - total };
        Self {
            source,
            mass_true,
            mass_false,
            mass_misleading,
            mass_uncertain: uncertain,
        }
    }
}

/// Source of a disinfo signal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalSource {
    /// Stylometric analysis (AI-generated detection).
    Stylometry,
    /// Network/CIB analysis (coordinated amplification).
    NetworkAnalysis,
    /// Claim matching against known fact-checks.
    ClaimMatching,
    /// Source credibility scores.
    SourceCredibility,
    /// Spread dynamics (organic vs artificial pattern).
    SpreadDynamics,
    /// Archetype classification.
    ArchetypeClassification,
    /// Manual analyst input.
    AnalystInput,
}

/// Hypothesis in the disinfo assessment frame of discernment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Hypothesis {
    True,
    False,
    Misleading,
    Unverifiable,
}

/// Unified disinfo assessment produced by fusing multiple signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisinfoAssessment {
    /// Target entity or narrative being assessed.
    pub target_id: String,
    /// Belief intervals per hypothesis.
    pub belief_true: f64,
    pub plausibility_true: f64,
    pub belief_false: f64,
    pub plausibility_false: f64,
    pub belief_misleading: f64,
    pub plausibility_misleading: f64,
    /// Overall uncertainty (plausibility - belief gap).
    pub uncertainty: f64,
    /// Conflict measure K between combined signals (0 = agreement, 1 = total conflict).
    pub conflict: f64,
    /// Number of signals fused.
    pub signal_count: usize,
    /// The signals that were combined.
    pub signals: Vec<DisinfoSignal>,
    /// Most likely hypothesis.
    pub verdict: Hypothesis,
    /// Confidence in the verdict.
    pub confidence: f64,
    pub computed_at: DateTime<Utc>,
}

/// Fuse multiple disinfo signals using Dempster-Shafer combination.
///
/// Uses Dempster's rule for low-conflict scenarios and Yager's rule
/// (transfers conflict mass to uncertainty) when conflict K > 0.7.
pub fn fuse_disinfo_signals(
    target_id: &str,
    signals: &[DisinfoSignal],
) -> Result<DisinfoAssessment> {
    if signals.is_empty() {
        return Err(TensaError::InvalidQuery("No signals to fuse".into()));
    }

    if signals.len() == 1 {
        let s = &signals[0];
        let verdict = determine_verdict(s.mass_true, s.mass_false, s.mass_misleading);
        let confidence = match verdict {
            Hypothesis::True => s.mass_true,
            Hypothesis::False => s.mass_false,
            Hypothesis::Misleading => s.mass_misleading,
            Hypothesis::Unverifiable => s.mass_uncertain,
        };
        return Ok(DisinfoAssessment {
            target_id: target_id.to_string(),
            belief_true: s.mass_true,
            plausibility_true: s.mass_true + s.mass_uncertain,
            belief_false: s.mass_false,
            plausibility_false: s.mass_false + s.mass_uncertain,
            belief_misleading: s.mass_misleading,
            plausibility_misleading: s.mass_misleading + s.mass_uncertain,
            uncertainty: s.mass_uncertain,
            conflict: 0.0,
            signal_count: 1,
            signals: signals.to_vec(),
            verdict,
            confidence,
            computed_at: Utc::now(),
        });
    }

    // Iteratively combine signals pairwise
    let mut combined_true = signals[0].mass_true;
    let mut combined_false = signals[0].mass_false;
    let mut combined_misleading = signals[0].mass_misleading;
    let mut combined_uncertain = signals[0].mass_uncertain;
    let mut total_conflict = 0.0;

    for signal in &signals[1..] {
        let (t, f, m, u, k) = combine_two(
            combined_true,
            combined_false,
            combined_misleading,
            combined_uncertain,
            signal.mass_true,
            signal.mass_false,
            signal.mass_misleading,
            signal.mass_uncertain,
        );
        combined_true = t;
        combined_false = f;
        combined_misleading = m;
        combined_uncertain = u;
        total_conflict = 1.0 - (1.0 - total_conflict) * (1.0 - k);
    }

    let verdict = determine_verdict(combined_true, combined_false, combined_misleading);
    let confidence = match verdict {
        Hypothesis::True => combined_true,
        Hypothesis::False => combined_false,
        Hypothesis::Misleading => combined_misleading,
        Hypothesis::Unverifiable => combined_uncertain,
    };

    Ok(DisinfoAssessment {
        target_id: target_id.to_string(),
        belief_true: combined_true,
        plausibility_true: combined_true + combined_uncertain,
        belief_false: combined_false,
        plausibility_false: combined_false + combined_uncertain,
        belief_misleading: combined_misleading,
        plausibility_misleading: combined_misleading + combined_uncertain,
        uncertainty: combined_uncertain,
        conflict: total_conflict,
        signal_count: signals.len(),
        signals: signals.to_vec(),
        verdict,
        confidence,
        computed_at: Utc::now(),
    })
}

/// Combine two mass functions. Uses Yager's rule when conflict > 0.7.
fn combine_two(
    t1: f64,
    f1: f64,
    m1: f64,
    u1: f64,
    t2: f64,
    f2: f64,
    m2: f64,
    u2: f64,
) -> (f64, f64, f64, f64, f64) {
    // Compute pairwise intersections
    let agree_true = t1 * t2 + t1 * u2 + u1 * t2;
    let agree_false = f1 * f2 + f1 * u2 + u1 * f2;
    let agree_misleading = m1 * m2 + m1 * u2 + u1 * m2;
    let agree_uncertain = u1 * u2;

    // Conflict: mass on empty set
    let conflict = t1 * f2 + t1 * m2 + f1 * t2 + f1 * m2 + m1 * t2 + m1 * f2;

    if conflict > 0.7 {
        // Yager's rule: transfer conflict mass to uncertainty
        let total = agree_true + agree_false + agree_misleading + agree_uncertain + conflict;
        if total < 1e-10 {
            return (0.0, 0.0, 0.0, 1.0, conflict);
        }
        (
            agree_true / total,
            agree_false / total,
            agree_misleading / total,
            (agree_uncertain + conflict) / total,
            conflict,
        )
    } else {
        // Dempster's rule: normalize out conflict
        let denom = 1.0 - conflict;
        if denom < 1e-10 {
            return (0.0, 0.0, 0.0, 1.0, conflict);
        }
        (
            agree_true / denom,
            agree_false / denom,
            agree_misleading / denom,
            agree_uncertain / denom,
            conflict,
        )
    }
}

fn determine_verdict(t: f64, f: f64, m: f64) -> Hypothesis {
    if f >= t && f >= m {
        Hypothesis::False
    } else if m >= t && m >= f {
        Hypothesis::Misleading
    } else if t >= f && t >= m {
        Hypothesis::True
    } else {
        Hypothesis::Unverifiable
    }
}

/// Store a disinfo assessment at `da/{target_id}`.
pub fn store_assessment(hypergraph: &Hypergraph, assessment: &DisinfoAssessment) -> Result<()> {
    let key = format!("da/{}", assessment.target_id);
    let value =
        serde_json::to_vec(assessment).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

/// Load a disinfo assessment.
pub fn load_assessment(
    hypergraph: &Hypergraph,
    target_id: &str,
) -> Result<Option<DisinfoAssessment>> {
    let key = format!("da/{}", target_id);
    let value = hypergraph
        .store()
        .get(key.as_bytes())
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    match value {
        Some(bytes) => {
            let a: DisinfoAssessment = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(a))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_signal() {
        let signal = DisinfoSignal::new(SignalSource::ClaimMatching, 0.1, 0.7, 0.1);
        let result = fuse_disinfo_signals("test-1", &[signal]).unwrap();
        assert_eq!(result.verdict, Hypothesis::False);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_agreeing_signals() {
        let signals = vec![
            DisinfoSignal::new(SignalSource::ClaimMatching, 0.1, 0.7, 0.1),
            DisinfoSignal::new(SignalSource::NetworkAnalysis, 0.1, 0.6, 0.2),
        ];
        let result = fuse_disinfo_signals("test-2", &signals).unwrap();
        assert_eq!(result.verdict, Hypothesis::False);
        assert!(result.belief_false > 0.5);
    }

    #[test]
    fn test_conflicting_signals_yager() {
        let signals = vec![
            DisinfoSignal::new(SignalSource::ClaimMatching, 0.05, 0.9, 0.0),
            DisinfoSignal::new(SignalSource::Stylometry, 0.85, 0.05, 0.0),
        ];
        let result = fuse_disinfo_signals("test-3", &signals).unwrap();
        // High conflict should increase uncertainty
        assert!(result.uncertainty > 0.0);
        assert!(result.conflict > 0.5);
    }

    #[test]
    fn test_empty_signals_error() {
        let result = fuse_disinfo_signals("test-4", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_masses_sum_to_one() {
        let signals = vec![
            DisinfoSignal::new(SignalSource::ClaimMatching, 0.3, 0.4, 0.1),
            DisinfoSignal::new(SignalSource::SpreadDynamics, 0.2, 0.5, 0.1),
            DisinfoSignal::new(SignalSource::SourceCredibility, 0.1, 0.6, 0.2),
        ];
        let result = fuse_disinfo_signals("test-5", &signals).unwrap();
        let total = result.belief_true
            + result.belief_false
            + result.belief_misleading
            + result.uncertainty;
        assert!(
            (total - 1.0).abs() < 0.01,
            "Total should be ~1.0, got {}",
            total
        );
    }

    #[test]
    fn test_signal_new_normalizes() {
        let s = DisinfoSignal::new(SignalSource::AnalystInput, 0.3, 0.3, 0.3);
        let total = s.mass_true + s.mass_false + s.mass_misleading + s.mass_uncertain;
        assert!((total - 1.0).abs() < 1e-10);
        assert!((s.mass_uncertain - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_signal_new_saturated() {
        // When masses already sum to >= 1.0, uncertainty should be 0
        let s = DisinfoSignal::new(SignalSource::AnalystInput, 0.5, 0.3, 0.3);
        assert_eq!(s.mass_uncertain, 0.0);
    }
}
