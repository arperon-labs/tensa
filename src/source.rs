//! Source intelligence types for multi-source data ingestion.
//!
//! Defines first-class Source objects with trust scores and bias profiles,
//! SourceAttribution links connecting sources to claims, ConfidenceBreakdown
//! for decomposed confidence scoring, and ContentionLinks for conflicting claims.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─── Source ──────────────────────────────────────────────────

/// A registered information source with trust and bias metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: Uuid,
    pub name: String,
    pub source_type: SourceType,
    pub url: Option<String>,
    pub description: Option<String>,
    /// Overall trust score, 0.0 (untrusted) to 1.0 (fully trusted).
    pub trust_score: f32,
    pub bias_profile: BiasProfile,
    pub track_record: TrackRecord,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Author or byline (e.g. "Jane Smith").
    #[serde(default)]
    pub author: Option<String>,
    /// Parent publication (e.g. "New York Times").
    #[serde(default)]
    pub publication: Option<String>,
    /// Who ingested/registered this source.
    #[serde(default)]
    pub ingested_by: Option<String>,
    /// Analyst notes on ingestion context.
    #[serde(default)]
    pub ingestion_notes: Option<String>,
    /// Original publication date.
    #[serde(default)]
    pub publication_date: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Classification of information source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    NewsOutlet,
    GovernmentAgency,
    AcademicInstitution,
    SocialMedia,
    Sensor,
    StructuredApi,
    HumanAnalyst,
    OsintTool,
    Custom(String),
}

/// Known biases and editorial tendencies of a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BiasProfile {
    /// Free-text labels, e.g. "pro-government", "sensationalist".
    #[serde(default)]
    pub known_biases: Vec<String>,
    /// Political lean: -1.0 (far left) to 1.0 (far right), None if unknown.
    #[serde(default)]
    pub political_lean: Option<f32>,
    /// Sensationalism: 0.0 (dry factual) to 1.0 (tabloid).
    #[serde(default)]
    pub sensationalism: Option<f32>,
    /// Free-form analyst notes on bias.
    #[serde(default)]
    pub notes: Option<String>,
}

/// Cumulative accuracy track record for a source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackRecord {
    pub claims_made: u64,
    pub claims_corroborated: u64,
    pub claims_contradicted: u64,
    pub last_evaluated: Option<DateTime<Utc>>,
}

impl TrackRecord {
    /// Suggest a trust score based on corroboration ratio.
    /// Returns 0.5 (neutral prior) when no claims have been evaluated.
    pub fn suggested_trust(&self) -> f32 {
        if self.claims_made == 0 {
            return 0.5;
        }
        self.claims_corroborated as f32 / self.claims_made as f32
    }
}

// ─── Source Attribution ──────────────────────────────────────

/// Links a registered Source to a specific Entity or Situation claim.
/// Stored with dual-index (sa/ forward, sar/ reverse) for efficient
/// queries in both directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub target_kind: AttributionTarget,
    /// When this claim was retrieved from the source.
    pub retrieved_at: DateTime<Utc>,
    /// URL of the specific article/page/endpoint.
    pub original_url: Option<String>,
    /// Relevant excerpt from the source material.
    pub excerpt: Option<String>,
    /// How well the extraction process parsed this source (0.0–1.0).
    pub extraction_confidence: f32,
    /// Which hypothesis this source supports (matches a frame element name).
    /// When None, trust is distributed uniformly across all hypotheses.
    #[serde(default)]
    pub claim: Option<String>,
}

/// Whether an attribution points to an Entity or Situation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionTarget {
    Entity,
    Situation,
}

// ─── Confidence Breakdown ────────────────────────────────────

/// Decomposed confidence replacing the flat f32 score.
/// Separates extraction quality from source reliability and corroboration.
/// When Bayesian posterior fields are present, `composite()` returns the
/// posterior mean `alpha / (alpha + beta)` from a Beta distribution prior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceBreakdown {
    /// How well the LLM/parser extracted this claim (0.0–1.0).
    pub extraction: f32,
    /// Weighted trust of attributing sources (0.0–1.0).
    pub source_credibility: f32,
    /// Independent source agreement (0.0–1.0).
    pub corroboration: f32,
    /// Time decay factor — fresher data scores higher (0.0–1.0).
    pub recency: f32,
    /// Beta prior alpha (shape parameter), initialized from extraction confidence.
    #[serde(default)]
    pub prior_alpha: Option<f32>,
    /// Beta prior beta (shape parameter).
    #[serde(default)]
    pub prior_beta: Option<f32>,
    /// Beta posterior alpha after incorporating source evidence.
    #[serde(default)]
    pub posterior_alpha: Option<f32>,
    /// Beta posterior beta after incorporating source evidence.
    #[serde(default)]
    pub posterior_beta: Option<f32>,
}

impl ConfidenceBreakdown {
    /// Legacy weighted-average coefficients `[extraction, source_credibility,
    /// corroboration, recency]`. Exposed so Phase 2's `composite_with_aggregator`
    /// can reuse them inside the OWA path without drifting.
    pub const LEGACY_WEIGHTS: [f32; 4] = [0.2, 0.35, 0.35, 0.1];

    /// Bayesian composite score.
    ///
    /// If posterior parameters are available, returns the Beta posterior mean
    /// `alpha / (alpha + beta)`. Otherwise falls back to the weighted average
    /// of the four component scores for backward compatibility.
    ///
    /// Cites: [yager1988owa] — Phase 2 exposes an aggregator-selectable
    /// variant via [`Self::composite_with_aggregator`]; the default path
    /// stays bit-identical.
    pub fn composite(&self) -> f32 {
        if let (Some(pa), Some(pb)) = (self.posterior_alpha, self.posterior_beta) {
            let sum = pa + pb;
            if sum > 0.0 {
                return (pa / sum).clamp(0.0, 1.0);
            }
        }
        // Fallback: weighted average (backward compat for pre-Bayesian data).
        let w = Self::LEGACY_WEIGHTS;
        (self.extraction * w[0]
            + self.source_credibility * w[1]
            + self.corroboration * w[2]
            + self.recency * w[3])
            .clamp(0.0, 1.0)
    }

    /// Component vector `[extraction, source_credibility, corroboration,
    /// recency]` — handy for the Phase 2 aggregator wiring.
    #[inline]
    pub fn components(&self) -> [f64; 4] {
        [
            self.extraction as f64,
            self.source_credibility as f64,
            self.corroboration as f64,
            self.recency as f64,
        ]
    }

    /// Aggregator-selectable composite score (Phase 2 wiring).
    ///
    /// Under `AggregatorKind::Owa(LEGACY_WEIGHTS)` this is bit-identical
    /// to the fallback branch of [`Self::composite`] when posterior
    /// parameters are unavailable, modulo OWA's sort-by-descending step —
    /// which is a no-op for the weighted-mean case as long as the
    /// weights are paired with their respective component values (OWA
    /// permutes both into sorted order, so the dot product is preserved).
    /// For bit-identical backward-compat callers should continue using
    /// [`Self::composite`]; this variant is the opt-in knob.
    ///
    /// Cites: [yager1988owa] [grabisch1996choquet].
    pub fn composite_with_aggregator(
        &self,
        agg: &crate::fuzzy::aggregation::AggregatorKind,
    ) -> crate::error::Result<f32> {
        self.composite_with_aggregator_tracked(agg, None, None)
            .map(|(score, _, _)| score)
    }

    /// Provenance-tracking sibling of [`Self::composite_with_aggregator`].
    ///
    /// Returns `(score, measure_id, measure_version)`. The slot rules
    /// (Phase 2 of the Graded Acceptability sprint):
    ///
    /// 1. If the caller passes `Some(measure_id)`, that takes priority.
    /// 2. Otherwise, if `agg` is `Choquet(measure)` and `measure.measure_id`
    ///    is `Some`, those slots are returned.
    /// 3. Otherwise both slots are `None` — wire-bit-identical to pre-Phase-2.
    ///
    /// Cites: [grabisch1996choquet], [bustince2016choquet].
    pub fn composite_with_aggregator_tracked(
        &self,
        agg: &crate::fuzzy::aggregation::AggregatorKind,
        measure_id: Option<String>,
        measure_version: Option<u32>,
    ) -> crate::error::Result<(f32, Option<String>, Option<u32>)> {
        let xs = self.components();
        let aggregator = crate::fuzzy::aggregation::aggregator_for(agg.clone());
        let v = aggregator.aggregate(&xs)?;
        let score = v.clamp(0.0, 1.0) as f32;
        let (id_out, ver_out) =
            crate::fuzzy::aggregation_learn::resolve_measure_provenance(
                agg,
                measure_id,
                measure_version,
            );
        Ok((score, id_out, ver_out))
    }
}

// ─── Contention ──────────────────────────────────────────────

/// Connects two Situations that make contradictory claims.
/// Stored with dual-index (ct/ forward, ctr/ reverse).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionLink {
    pub situation_a: Uuid,
    pub situation_b: Uuid,
    pub contention_type: ContentionType,
    pub description: Option<String>,
    /// Whether this contention has been adjudicated.
    pub resolved: bool,
    /// Analyst notes on the resolution.
    pub resolution: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Classification of how two claims disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentionType {
    /// A says X, B says not-X.
    DirectContradiction,
    /// A says 100, B says 500.
    NumericalDisagreement,
    /// A says Tuesday, B says Wednesday.
    TemporalDisagreement,
    /// A says actor X did it, B says actor Y.
    AttributionDisagreement,
    /// A covers a fact, B omits it entirely.
    OmissionBias,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn test_source_serialization_roundtrip() {
        let source = Source {
            id: Uuid::now_v7(),
            name: "Reuters".to_string(),
            source_type: SourceType::NewsOutlet,
            url: Some("https://reuters.com".to_string()),
            description: Some("International news agency".to_string()),
            trust_score: 0.85,
            bias_profile: BiasProfile::default(),
            track_record: TrackRecord::default(),
            tags: vec!["wire-service".to_string()],
            author: Some("John Doe".to_string()),
            publication: Some("Reuters Agency".to_string()),
            ingested_by: None,
            ingestion_notes: None,
            publication_date: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let json = serde_json::to_vec(&source).unwrap();
        let decoded: Source = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.name, "Reuters");
        assert_eq!(decoded.trust_score, 0.85);
        assert_eq!(decoded.source_type, SourceType::NewsOutlet);
    }

    #[test]
    fn test_source_attribution_roundtrip() {
        let attr = SourceAttribution {
            source_id: Uuid::now_v7(),
            target_id: Uuid::now_v7(),
            target_kind: AttributionTarget::Situation,
            retrieved_at: Utc::now(),
            original_url: Some("https://example.com/article".to_string()),
            excerpt: Some("The event occurred on Tuesday.".to_string()),
            extraction_confidence: 0.92,
            claim: None,
        };
        let json = serde_json::to_vec(&attr).unwrap();
        let decoded: SourceAttribution = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.target_kind, AttributionTarget::Situation);
        assert_eq!(decoded.extraction_confidence, 0.92);
    }

    #[test]
    fn test_confidence_breakdown_composite() {
        // Without Bayesian fields: weighted average fallback
        let bd = ConfidenceBreakdown {
            extraction: 0.9,
            source_credibility: 0.8,
            corroboration: 0.7,
            recency: 1.0,
            prior_alpha: None,
            prior_beta: None,
            posterior_alpha: None,
            posterior_beta: None,
        };
        // 0.9*0.2 + 0.8*0.35 + 0.7*0.35 + 1.0*0.1 = 0.18 + 0.28 + 0.245 + 0.1 = 0.805
        let composite = bd.composite();
        assert!((composite - 0.805).abs() < 0.001);
    }

    #[test]
    fn test_confidence_breakdown_bayesian_composite() {
        // With Bayesian fields: posterior mean
        let bd = ConfidenceBreakdown {
            extraction: 0.5,
            source_credibility: 0.5,
            corroboration: 0.5,
            recency: 0.5,
            prior_alpha: Some(1.0),
            prior_beta: Some(1.0),
            posterior_alpha: Some(3.0),
            posterior_beta: Some(1.0),
        };
        // Posterior mean = 3.0 / (3.0 + 1.0) = 0.75
        assert!((bd.composite() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_confidence_breakdown_clamp() {
        let bd = ConfidenceBreakdown {
            extraction: 1.0,
            source_credibility: 1.0,
            corroboration: 1.0,
            recency: 1.0,
            prior_alpha: None,
            prior_beta: None,
            posterior_alpha: None,
            posterior_beta: None,
        };
        assert!(bd.composite() <= 1.0);

        let bd_zero = ConfidenceBreakdown {
            extraction: 0.0,
            source_credibility: 0.0,
            corroboration: 0.0,
            recency: 0.0,
            prior_alpha: None,
            prior_beta: None,
            posterior_alpha: None,
            posterior_beta: None,
        };
        assert!(bd_zero.composite() >= 0.0);
    }

    #[test]
    fn test_composite_with_aggregator_mean_equals_arithmetic_mean() {
        // Phase 2 aggregator wiring — AggregatorKind::Mean produces the
        // unweighted arithmetic mean of the four component scores. This
        // demonstrably differs from the legacy weighted average path that
        // `composite()` falls back on.
        use crate::fuzzy::aggregation::AggregatorKind;
        let bd = ConfidenceBreakdown {
            extraction: 0.9,
            source_credibility: 0.8,
            corroboration: 0.7,
            recency: 1.0,
            prior_alpha: None,
            prior_beta: None,
            posterior_alpha: None,
            posterior_beta: None,
        };
        // Weighted default (0.2/0.35/0.35/0.1) → 0.805 as in the legacy test.
        let legacy = bd.composite();
        let mean = bd.composite_with_aggregator(&AggregatorKind::Mean).unwrap();
        let expect_mean = (0.9 + 0.8 + 0.7 + 1.0) / 4.0;
        assert!((mean - expect_mean).abs() < 1e-6);
        // The two paths differ (weighted vs uniform) — confirms the knob works.
        assert!((legacy - mean).abs() > 0.01);
    }

    #[test]
    fn test_composite_with_aggregator_choquet_additive_matches_mean() {
        use crate::fuzzy::aggregation::AggregatorKind;
        use crate::fuzzy::aggregation_measure::symmetric_additive;
        let bd = ConfidenceBreakdown {
            extraction: 0.2,
            source_credibility: 0.4,
            corroboration: 0.6,
            recency: 0.8,
            prior_alpha: None,
            prior_beta: None,
            posterior_alpha: None,
            posterior_beta: None,
        };
        let measure = symmetric_additive(4).unwrap();
        let choquet_val = bd
            .composite_with_aggregator(&AggregatorKind::Choquet(measure))
            .unwrap();
        let mean = bd.composite_with_aggregator(&AggregatorKind::Mean).unwrap();
        // Choquet with symmetric_additive = arithmetic mean (within FP noise).
        assert!((choquet_val - mean).abs() < 1e-6);
    }

    #[test]
    fn test_contention_link_roundtrip() {
        let link = ContentionLink {
            situation_a: Uuid::now_v7(),
            situation_b: Uuid::now_v7(),
            contention_type: ContentionType::DirectContradiction,
            description: Some("Source A says attack, Source B denies".to_string()),
            resolved: false,
            resolution: None,
            created_at: Utc::now(),
        };
        let json = serde_json::to_vec(&link).unwrap();
        let decoded: ContentionLink = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.contention_type, ContentionType::DirectContradiction);
        assert!(!decoded.resolved);
    }

    #[test]
    fn test_track_record_suggested_trust() {
        let empty = TrackRecord::default();
        assert_eq!(empty.suggested_trust(), 0.5);

        let good = TrackRecord {
            claims_made: 100,
            claims_corroborated: 80,
            claims_contradicted: 10,
            last_evaluated: Some(Utc::now()),
        };
        assert!((good.suggested_trust() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_bias_profile_default() {
        let bp = BiasProfile::default();
        assert!(bp.known_biases.is_empty());
        assert!(bp.political_lean.is_none());
        assert!(bp.sensationalism.is_none());
        assert!(bp.notes.is_none());
    }

    #[test]
    fn test_source_type_custom() {
        let st = SourceType::Custom("satellite-imagery".to_string());
        let json = serde_json::to_string(&st).unwrap();
        let decoded: SourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, st);
    }

    #[test]
    fn test_contention_type_variants() {
        let types = vec![
            ContentionType::DirectContradiction,
            ContentionType::NumericalDisagreement,
            ContentionType::TemporalDisagreement,
            ContentionType::AttributionDisagreement,
            ContentionType::OmissionBias,
        ];
        for ct in types {
            let json = serde_json::to_string(&ct).unwrap();
            let decoded: ContentionType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, ct);
        }
    }
}
