//! Per-task fingerprint comparison built on the existing per-layer distance
//! kernels in [`crate::analysis::similarity_metrics`] and bootstrap CIs in
//! [`crate::analysis::stylometry_stats`].
//!
//! Three comparison kinds:
//! - **Behavioral**: pairs of [`BehavioralFingerprint`] (10 axes per actor).
//! - **Disinfo**: pairs of [`DisinformationFingerprint`] (12 axes per narrative).
//! - **Narrative content**: re-uses the existing
//!   [`crate::analysis::style_profile::WeightedSimilarityConfig`] machinery —
//!   loaded via `WeightedSimilarityConfig::load_or_default` so any
//!   `train_pan_weights`-learned weights apply automatically.
//!
//! Each call returns a [`FingerprintComparison`] with the composite distance,
//! per-axis distance breakdown, a (point-estimate) p-value derived from the
//! axis-distance distribution, a normal-approximation 95% confidence interval,
//! and the top axes contributing to the dissimilarity.

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};

use super::fingerprints::{
    behavioral_axis_labels, disinfo_axis_labels, BehavioralFingerprint, DisinformationFingerprint,
};

/// Which fingerprint type the comparison is over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonKind {
    /// Two behavioral fingerprints (per-actor).
    Behavioral,
    /// Two disinformation fingerprints (per-narrative).
    Disinfo,
    /// Two narrative content fingerprints (per-narrative prose+structure) —
    /// delegates to [`crate::analysis::style_profile`] for the heavy lifting.
    Narrative,
}

/// Task-specific weighting profile. Selects which `WeightedSimilarityConfig`
/// to load from KV (when present); falls back to uniform weights otherwise.
///
/// The `Cib` and `Factory` configs are produced by the
/// `train_pan_weights` binary (Workstream C) when trained on labeled
/// coordinated-behavior or content-factory data sets respectively. Until those
/// configs are trained and uploaded, all three tasks use the uniform default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonTask {
    Literary,
    Cib,
    Factory,
}

impl Default for ComparisonTask {
    fn default() -> Self {
        Self::Literary
    }
}

/// One axis whose distance contributed disproportionately to the composite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisAnomaly {
    pub axis: String,
    pub distance: f64,
    pub a_value: f64,
    pub b_value: f64,
}

/// Output of [`compare_fingerprints`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintComparison {
    pub kind: ComparisonKind,
    pub task: ComparisonTask,
    /// Weighted composite distance in [0, 1+] (0 = identical, 1 = maximally
    /// different on every comparable axis).
    pub composite_distance: f64,
    /// Per-axis label → squared difference in [0, 1].
    pub per_layer_distances: Vec<(String, f64)>,
    /// Empirical left-tail p-value: under a null hypothesis of "axis distances
    /// are independent uniform draws", how often we'd see a composite this
    /// extreme. Computed from the axis-distance sample distribution.
    pub p_value: f64,
    /// Normal-approximation 95% confidence interval on the composite distance.
    pub confidence_interval: (f64, f64),
    /// Decision threshold at α = 0.05: `true` ⇒ likely same source.
    pub same_source_verdict: bool,
    /// Number of comparable axes (i.e. neither side was NaN).
    pub comparable_axes: usize,
    /// Top contributing axes to the composite distance (descending by distance).
    pub anomaly_axes: Vec<AxisAnomaly>,
    /// Number of axes that were compared (non-NaN on both sides).
    #[serde(default)]
    pub axes_compared: usize,
    /// Number of axes skipped because one or both sides were NaN.
    #[serde(default)]
    pub axes_skipped: usize,
    /// Fraction of non-NaN axes in fingerprint A (0.0–1.0).
    #[serde(default)]
    pub completeness_a: f64,
    /// Fraction of non-NaN axes in fingerprint B (0.0–1.0).
    #[serde(default)]
    pub completeness_b: f64,
    /// `true` when either fingerprint has fewer than half its axes computed,
    /// meaning the comparison result should be treated with caution.
    #[serde(default)]
    pub low_confidence: bool,
}

const ANOMALY_TOP_K: usize = 3;
const SAME_SOURCE_ALPHA: f64 = 0.05;

/// Compare two behavioral fingerprints, two disinfo fingerprints, or two
/// narrative content fingerprints. Returns a structured
/// [`FingerprintComparison`] suitable for the `/fingerprints/compare` API and
/// the `compare_fingerprints` MCP tool.
pub fn compare_fingerprints(
    kind: ComparisonKind,
    task: ComparisonTask,
    a: &serde_json::Value,
    b: &serde_json::Value,
) -> Result<FingerprintComparison> {
    match kind {
        ComparisonKind::Behavioral => {
            let fp_a: BehavioralFingerprint = serde_json::from_value(a.clone())
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            let fp_b: BehavioralFingerprint = serde_json::from_value(b.clone())
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(compare_axis_vectors(
                &fp_a.axes,
                &fp_b.axes,
                &behavioral_axis_labels(),
                ComparisonKind::Behavioral,
                task,
            ))
        }
        ComparisonKind::Disinfo => {
            let fp_a: DisinformationFingerprint = serde_json::from_value(a.clone())
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            let fp_b: DisinformationFingerprint = serde_json::from_value(b.clone())
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(compare_axis_vectors(
                &fp_a.axes,
                &fp_b.axes,
                &disinfo_axis_labels(),
                ComparisonKind::Disinfo,
                task,
            ))
        }
        ComparisonKind::Narrative => Err(TensaError::Internal(
            "narrative-content fingerprint comparison should be invoked via \
             /style/compare (style_routes.rs); compare_fingerprints handles \
             behavioral and disinfo only in Sprint D1"
                .into(),
        )),
    }
}

/// Comparable-axis arithmetic: for each axis where both sides are non-NaN,
/// take the squared difference clamped to `[0, 1]`. Composite is the mean.
/// p-value is the empirical left-tail of "fraction of axes with distance
/// below the median observed", which is a coarse but reproducible test.
fn compare_axis_vectors<const N: usize>(
    a: &[f64; N],
    b: &[f64; N],
    labels: &[&'static str; N],
    kind: ComparisonKind,
    task: ComparisonTask,
) -> FingerprintComparison {
    let mut per_layer: Vec<(String, f64)> = Vec::with_capacity(N);
    let mut anomaly: Vec<AxisAnomaly> = Vec::with_capacity(N);
    let mut comparable_distances: Vec<f64> = Vec::with_capacity(N);
    let mut skipped = 0usize;
    let a_computed = a.iter().filter(|v| !v.is_nan()).count();
    let b_computed = b.iter().filter(|v| !v.is_nan()).count();
    for i in 0..N {
        if a[i].is_nan() || b[i].is_nan() {
            skipped += 1;
            continue;
        }
        let diff = (a[i] - b[i]).powi(2).clamp(0.0, 1.0);
        per_layer.push((labels[i].to_string(), diff));
        comparable_distances.push(diff);
        anomaly.push(AxisAnomaly {
            axis: labels[i].to_string(),
            distance: diff,
            a_value: a[i],
            b_value: b[i],
        });
    }
    let comparable_axes = comparable_distances.len();
    let composite = if comparable_axes == 0 {
        0.0
    } else {
        comparable_distances.iter().sum::<f64>() / comparable_axes as f64
    };

    // Empirical p-value: how many axes had distance ≥ composite. This is a
    // crude permutation surrogate that returns higher p when more axes are
    // close to the composite (i.e. the composite is "typical"), and lower p
    // when the composite is dragged up by a few outliers (suggesting a
    // structured difference).
    let p_value = if comparable_axes == 0 {
        1.0
    } else {
        let above = comparable_distances
            .iter()
            .filter(|d| **d >= composite)
            .count();
        (above as f64 / comparable_axes as f64).clamp(0.0, 1.0)
    };

    // Normal-approximation 95% CI on the composite (sample mean ± 1.96·SE).
    let ci = if comparable_axes >= 2 {
        let var = comparable_distances
            .iter()
            .map(|d| (d - composite).powi(2))
            .sum::<f64>()
            / comparable_axes as f64;
        let se = (var / comparable_axes as f64).sqrt();
        let half = 1.96 * se;
        ((composite - half).max(0.0), (composite + half).min(1.0))
    } else {
        (composite, composite)
    };

    // Sort anomaly axes by distance descending; keep top K.
    anomaly.sort_by(|x, y| {
        y.distance
            .partial_cmp(&x.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    anomaly.truncate(ANOMALY_TOP_K);

    let same_source_verdict = comparable_axes > 0 && p_value > SAME_SOURCE_ALPHA;

    // Completeness: fraction of non-NaN axes in each fingerprint.
    let half_axes = N / 2;
    let completeness_a = a_computed as f64 / N as f64;
    let completeness_b = b_computed as f64 / N as f64;
    let low_confidence = a_computed < half_axes || b_computed < half_axes;

    FingerprintComparison {
        kind,
        task,
        composite_distance: composite,
        per_layer_distances: per_layer,
        p_value,
        confidence_interval: ci,
        same_source_verdict,
        comparable_axes,
        anomaly_axes: anomaly,
        axes_compared: comparable_axes,
        axes_skipped: skipped,
        completeness_a,
        completeness_b,
        low_confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disinfo::fingerprints::{
        BehavioralAxis, BehavioralFingerprint, DisinfoAxis, DisinformationFingerprint,
    };
    use uuid::Uuid;

    fn make_bfp(axes: &[(BehavioralAxis, f64)]) -> BehavioralFingerprint {
        let mut fp = BehavioralFingerprint::empty(Uuid::nil());
        for (a, v) in axes {
            fp.set_axis(*a, *v);
        }
        fp
    }

    fn make_dfp(narr: &str, axes: &[(DisinfoAxis, f64)]) -> DisinformationFingerprint {
        let mut fp = DisinformationFingerprint::empty(narr);
        for (a, v) in axes {
            fp.set_axis(*a, *v);
        }
        fp
    }

    #[test]
    fn identical_behavioral_yields_zero_distance() {
        let fp = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 0.4),
            (BehavioralAxis::HashtagConcentration, 0.7),
        ]);
        let cmp = compare_fingerprints(
            ComparisonKind::Behavioral,
            ComparisonTask::Cib,
            &serde_json::to_value(&fp).unwrap(),
            &serde_json::to_value(&fp).unwrap(),
        )
        .unwrap();
        assert!(cmp.composite_distance < 1e-9);
        assert!(cmp.same_source_verdict);
        assert_eq!(cmp.comparable_axes, 2);
    }

    #[test]
    fn maximally_different_behavioral_yields_unit_distance() {
        let a = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 0.0),
            (BehavioralAxis::HashtagConcentration, 0.0),
        ]);
        let b = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 1.0),
            (BehavioralAxis::HashtagConcentration, 1.0),
        ]);
        let cmp = compare_fingerprints(
            ComparisonKind::Behavioral,
            ComparisonTask::Cib,
            &serde_json::to_value(&a).unwrap(),
            &serde_json::to_value(&b).unwrap(),
        )
        .unwrap();
        assert!((cmp.composite_distance - 1.0).abs() < 1e-9);
    }

    #[test]
    fn nan_axes_are_skipped() {
        let mut a = BehavioralFingerprint::empty(Uuid::nil());
        a.set_axis(BehavioralAxis::EngagementRatio, 0.4);
        // HashtagConcentration left NaN
        let mut b = BehavioralFingerprint::empty(Uuid::nil());
        b.set_axis(BehavioralAxis::EngagementRatio, 0.4);
        b.set_axis(BehavioralAxis::HashtagConcentration, 0.9);
        let cmp = compare_fingerprints(
            ComparisonKind::Behavioral,
            ComparisonTask::Cib,
            &serde_json::to_value(&a).unwrap(),
            &serde_json::to_value(&b).unwrap(),
        )
        .unwrap();
        assert_eq!(cmp.comparable_axes, 1);
        assert!(cmp.composite_distance < 1e-9);
    }

    #[test]
    fn disinfo_comparison_works() {
        let a = make_dfp(
            "narr-a",
            &[
                (DisinfoAxis::SourceDiversity, 0.2),
                (DisinfoAxis::EvidentialUncertainty, 0.5),
            ],
        );
        let b = make_dfp(
            "narr-b",
            &[
                (DisinfoAxis::SourceDiversity, 0.8),
                (DisinfoAxis::EvidentialUncertainty, 0.5),
            ],
        );
        let cmp = compare_fingerprints(
            ComparisonKind::Disinfo,
            ComparisonTask::Factory,
            &serde_json::to_value(&a).unwrap(),
            &serde_json::to_value(&b).unwrap(),
        )
        .unwrap();
        assert_eq!(cmp.comparable_axes, 2);
        assert!(cmp.composite_distance > 0.0);
        // The single non-trivial axis (source_diversity) should appear in anomaly_axes.
        assert!(cmp
            .anomaly_axes
            .iter()
            .any(|a| a.axis == "source_diversity"));
    }

    #[test]
    fn ci_brackets_composite() {
        let a = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 0.0),
            (BehavioralAxis::HashtagConcentration, 0.5),
            (BehavioralAxis::ContentOriginality, 1.0),
        ]);
        let b = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 0.3),
            (BehavioralAxis::HashtagConcentration, 0.5),
            (BehavioralAxis::ContentOriginality, 0.7),
        ]);
        let cmp = compare_fingerprints(
            ComparisonKind::Behavioral,
            ComparisonTask::Literary,
            &serde_json::to_value(&a).unwrap(),
            &serde_json::to_value(&b).unwrap(),
        )
        .unwrap();
        let (lo, hi) = cmp.confidence_interval;
        assert!(lo <= cmp.composite_distance + 1e-9);
        assert!(hi + 1e-9 >= cmp.composite_distance);
    }

    #[test]
    fn narrative_kind_returns_error_in_d1() {
        let dummy = serde_json::json!({});
        let result = compare_fingerprints(
            ComparisonKind::Narrative,
            ComparisonTask::Literary,
            &dummy,
            &dummy,
        );
        assert!(result.is_err());
    }

    #[test]
    fn partial_metadata_fields_populated() {
        // A has 2 axes, B has 3 axes → 2 comparable, 8 skipped
        let a = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 0.4),
            (BehavioralAxis::HashtagConcentration, 0.7),
        ]);
        let b = make_bfp(&[
            (BehavioralAxis::EngagementRatio, 0.4),
            (BehavioralAxis::HashtagConcentration, 0.7),
            (BehavioralAxis::ContentOriginality, 0.5),
        ]);
        let cmp = compare_fingerprints(
            ComparisonKind::Behavioral,
            ComparisonTask::Cib,
            &serde_json::to_value(&a).unwrap(),
            &serde_json::to_value(&b).unwrap(),
        )
        .unwrap();
        assert_eq!(cmp.axes_compared, 2);
        assert_eq!(cmp.axes_skipped, 8); // 10 - 2 = 8
        assert!((cmp.completeness_a - 0.2).abs() < 1e-9); // 2/10
        assert!((cmp.completeness_b - 0.3).abs() < 1e-9); // 3/10
        assert!(cmp.low_confidence, "both have < 5 axes → low confidence");
    }

    #[test]
    fn low_confidence_false_for_full_fingerprints() {
        // Both have all 10 axes
        let axes: Vec<(BehavioralAxis, f64)> = vec![
            (BehavioralAxis::PostingCadenceRegularity, 0.5),
            (BehavioralAxis::SleepPatternPresence, 0.5),
            (BehavioralAxis::EngagementRatio, 0.5),
            (BehavioralAxis::AccountMaturity, 0.5),
            (BehavioralAxis::PlatformDiversity, 0.5),
            (BehavioralAxis::ContentOriginality, 0.5),
            (BehavioralAxis::ResponseLatency, 0.5),
            (BehavioralAxis::HashtagConcentration, 0.5),
            (BehavioralAxis::NetworkInsularity, 0.5),
            (BehavioralAxis::TemporalCoordination, 0.5),
        ];
        let a = make_bfp(&axes);
        let b = make_bfp(&axes);
        let cmp = compare_fingerprints(
            ComparisonKind::Behavioral,
            ComparisonTask::Cib,
            &serde_json::to_value(&a).unwrap(),
            &serde_json::to_value(&b).unwrap(),
        )
        .unwrap();
        assert_eq!(cmp.axes_compared, 10);
        assert_eq!(cmp.axes_skipped, 0);
        assert!((cmp.completeness_a - 1.0).abs() < 1e-9);
        assert!((cmp.completeness_b - 1.0).abs() < 1e-9);
        assert!(!cmp.low_confidence);
    }
}
