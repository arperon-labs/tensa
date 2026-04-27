//! Actor archetype classification (Sprint D5).
//!
//! Classifies actors into adversarial archetypes based on behavioral
//! fingerprints and action patterns. Reuses MaxEnt IRL concepts from
//! `crate::inference::motivation`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

/// Adversarial archetype categories.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Archetype {
    /// State-sponsored actor (e.g., IRA, Ghostwriter).
    StateActor,
    /// Organic conspiracy theorist spreading without coordination.
    OrganicConspiracist,
    /// Commercial troll farm operating for profit.
    CommercialTrollFarm,
    /// Hacktivist with ideological motivation.
    Hacktivist,
    /// Unwitting amplifier — not malicious but spreads disinfo.
    UsefulIdiot,
    /// State-funded but presenting as organic.
    HybridActor,
}

impl std::fmt::Display for Archetype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StateActor => write!(f, "state_actor"),
            Self::OrganicConspiracist => write!(f, "organic_conspiracist"),
            Self::CommercialTrollFarm => write!(f, "commercial_troll_farm"),
            Self::Hacktivist => write!(f, "hacktivist"),
            Self::UsefulIdiot => write!(f, "useful_idiot"),
            Self::HybridActor => write!(f, "hybrid_actor"),
        }
    }
}

impl std::str::FromStr for Archetype {
    type Err = TensaError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "state_actor" | "stateactor" => Ok(Self::StateActor),
            "organic_conspiracist" | "organicconspiracist" | "organic" => {
                Ok(Self::OrganicConspiracist)
            }
            "commercial_troll_farm" | "commercialtrollfarm" | "troll_farm" => {
                Ok(Self::CommercialTrollFarm)
            }
            "hacktivist" => Ok(Self::Hacktivist),
            "useful_idiot" | "usefulidiot" => Ok(Self::UsefulIdiot),
            "hybrid_actor" | "hybridactor" | "hybrid" => Ok(Self::HybridActor),
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown archetype: {}",
                other
            ))),
        }
    }
}

/// Probability distribution over archetypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeDistribution {
    pub actor_id: Uuid,
    /// Probability for each archetype (sums to 1.0).
    pub scores: Vec<(Archetype, f64)>,
    /// Most likely archetype.
    pub primary: Archetype,
    /// Confidence in the primary classification.
    pub confidence: f64,
    pub computed_at: DateTime<Utc>,
}

/// Archetype feature template — defines expected behavioral patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeTemplate {
    pub archetype: Archetype,
    /// Expected behavioral fingerprint axis ranges [min, max] per axis.
    /// Index matches BehavioralAxis enum.
    pub expected_ranges: Vec<(f64, f64)>,
    /// Weight for each axis in the classification (higher = more important).
    pub axis_weights: Vec<f64>,
}

/// Default archetype templates based on known campaign characteristics.
pub fn default_templates() -> Vec<ArchetypeTemplate> {
    vec![
        ArchetypeTemplate {
            archetype: Archetype::StateActor,
            expected_ranges: vec![
                (0.7, 1.0), // cadence: very regular (automated scheduling)
                (0.0, 0.3), // sleep: minimal (24/7 operation)
                (0.0, 0.3), // engagement: low original engagement
                (0.0, 0.4), // maturity: newer accounts
                (0.3, 0.8), // platform_diversity: multi-platform
                (0.0, 0.3), // originality: low (copy-paste campaigns)
                (0.0, 0.2), // response_latency: fast (automated)
                (0.6, 1.0), // hashtag: concentrated
                (0.7, 1.0), // insularity: very insular
                (0.7, 1.0), // coordination: highly coordinated
            ],
            axis_weights: vec![1.0, 1.5, 0.8, 0.5, 1.0, 1.2, 0.8, 0.6, 1.5, 2.0],
        },
        ArchetypeTemplate {
            archetype: Archetype::OrganicConspiracist,
            expected_ranges: vec![
                (0.2, 0.6), // cadence: irregular bursts
                (0.5, 1.0), // sleep: has sleep patterns
                (0.3, 0.7), // engagement: moderate
                (0.5, 1.0), // maturity: established accounts
                (0.0, 0.3), // platform_diversity: few platforms
                (0.4, 0.8), // originality: moderate (adds own spin)
                (0.3, 0.7), // response_latency: variable
                (0.4, 0.8), // hashtag: moderate concentration
                (0.4, 0.7), // insularity: somewhat insular
                (0.0, 0.3), // coordination: low
            ],
            axis_weights: vec![0.8, 1.2, 0.6, 1.0, 0.5, 1.0, 0.5, 0.8, 1.0, 1.5],
        },
        ArchetypeTemplate {
            archetype: Archetype::CommercialTrollFarm,
            expected_ranges: vec![
                (0.6, 0.9), // cadence: regular but with variance
                (0.1, 0.4), // sleep: minimal
                (0.0, 0.2), // engagement: very low (disposable accounts)
                (0.0, 0.3), // maturity: new accounts
                (0.1, 0.5), // platform_diversity: moderate
                (0.0, 0.2), // originality: very low (templates)
                (0.0, 0.3), // response_latency: fast
                (0.5, 0.9), // hashtag: concentrated (campaign-specific)
                (0.5, 0.8), // insularity: insular
                (0.5, 0.9), // coordination: high
            ],
            axis_weights: vec![0.8, 1.0, 0.5, 0.8, 0.6, 1.5, 0.7, 0.8, 1.0, 1.8],
        },
        ArchetypeTemplate {
            archetype: Archetype::Hacktivist,
            expected_ranges: vec![
                (0.3, 0.7), // cadence: event-driven bursts
                (0.4, 0.8), // sleep: some pattern
                (0.4, 0.8), // engagement: moderate-high
                (0.3, 0.8), // maturity: variable
                (0.2, 0.6), // platform_diversity: moderate
                (0.5, 0.9), // originality: high (manifestos, leaks)
                (0.2, 0.5), // response_latency: engaged
                (0.3, 0.7), // hashtag: campaign-focused
                (0.3, 0.6), // insularity: moderate
                (0.2, 0.6), // coordination: moderate
            ],
            axis_weights: vec![0.6, 0.7, 1.0, 0.5, 0.6, 1.5, 0.8, 1.0, 0.8, 1.0],
        },
        ArchetypeTemplate {
            archetype: Archetype::UsefulIdiot,
            expected_ranges: vec![
                (0.1, 0.5), // cadence: irregular
                (0.6, 1.0), // sleep: normal
                (0.3, 0.6), // engagement: moderate
                (0.5, 1.0), // maturity: established
                (0.0, 0.3), // platform_diversity: low (one main platform)
                (0.2, 0.5), // originality: shares more than creates
                (0.3, 0.8), // response_latency: normal human speed
                (0.1, 0.4), // hashtag: low (not campaign-focused)
                (0.2, 0.5), // insularity: somewhat open
                (0.0, 0.2), // coordination: none
            ],
            axis_weights: vec![0.5, 1.5, 0.8, 1.2, 0.5, 1.0, 1.0, 0.5, 0.8, 2.0],
        },
        ArchetypeTemplate {
            archetype: Archetype::HybridActor,
            expected_ranges: vec![
                (0.4, 0.8), // cadence: mixed
                (0.3, 0.7), // sleep: some pattern
                (0.3, 0.6), // engagement: moderate
                (0.3, 0.7), // maturity: variable
                (0.2, 0.6), // platform_diversity: moderate
                (0.3, 0.6), // originality: moderate
                (0.2, 0.5), // response_latency: moderate
                (0.3, 0.7), // hashtag: moderate
                (0.4, 0.7), // insularity: somewhat insular
                (0.3, 0.7), // coordination: moderate
            ],
            axis_weights: vec![0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        },
    ]
}

/// Classify an actor into archetype distribution based on their behavioral fingerprint.
pub fn classify_actor_archetype(
    hypergraph: &Hypergraph,
    actor_id: Uuid,
) -> Result<ArchetypeDistribution> {
    let fp =
        crate::disinfo::fingerprints::ensure_behavioral_fingerprint(hypergraph, &actor_id, false)?;
    let templates = default_templates();

    let mut raw_scores: Vec<(Archetype, f64)> = Vec::new();

    for template in &templates {
        let score =
            compute_template_score(&fp.axes, &template.expected_ranges, &template.axis_weights);
        raw_scores.push((template.archetype.clone(), score));
    }

    // Softmax normalization
    let max_score = raw_scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = raw_scores.iter().map(|(_, s)| (s - max_score).exp()).sum();

    let mut scores: Vec<(Archetype, f64)> = raw_scores
        .iter()
        .map(|(a, s)| (a.clone(), (s - max_score).exp() / exp_sum))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let primary = scores[0].0.clone();
    let confidence = scores[0].1;

    let dist = ArchetypeDistribution {
        actor_id,
        scores,
        primary,
        confidence,
        computed_at: Utc::now(),
    };

    // Persist
    store_archetype(hypergraph, &dist)?;

    Ok(dist)
}

/// Compute the template match score for a behavioral fingerprint.
/// Higher score = better match to the template.
fn compute_template_score(axes: &[f64], expected_ranges: &[(f64, f64)], weights: &[f64]) -> f64 {
    let mut total_score = 0.0;
    let mut total_weight = 0.0;

    for (i, &value) in axes.iter().enumerate() {
        if value.is_nan() || i >= expected_ranges.len() || i >= weights.len() {
            continue;
        }
        let (lo, hi) = expected_ranges[i];
        let w = weights[i];

        // Score: 1.0 if within range, decays with Gaussian outside range
        let score = if value >= lo && value <= hi {
            1.0
        } else {
            let dist = if value < lo { lo - value } else { value - hi };
            (-dist * dist / 0.1).exp() // Gaussian decay with sigma^2 = 0.1
        };

        total_score += score * w;
        total_weight += w;
    }

    if total_weight < 1e-10 {
        return 0.0;
    }
    total_score / total_weight
}

/// Persist archetype distribution at `arch/{actor_uuid}`.
pub fn store_archetype(hypergraph: &Hypergraph, dist: &ArchetypeDistribution) -> Result<()> {
    let key = format!("arch/{}", dist.actor_id);
    let value = serde_json::to_vec(dist).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hypergraph
        .store()
        .put(key.as_bytes(), &value)
        .map_err(|e| TensaError::Internal(e.to_string()))
}

/// Load archetype distribution for an actor.
pub fn load_archetype(
    hypergraph: &Hypergraph,
    actor_id: &Uuid,
) -> Result<Option<ArchetypeDistribution>> {
    let key = format!("arch/{}", actor_id);
    let value = hypergraph
        .store()
        .get(key.as_bytes())
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    match value {
        Some(bytes) => {
            let dist: ArchetypeDistribution = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(dist))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_template_score_perfect_match() {
        let axes = vec![0.8, 0.1, 0.2, 0.2, 0.5, 0.1, 0.1, 0.8, 0.8, 0.9];
        let template = &default_templates()[0]; // StateActor
        let score =
            compute_template_score(&axes, &template.expected_ranges, &template.axis_weights);
        assert!(score > 0.7, "State actor should score high: {}", score);
    }

    #[test]
    fn test_compute_template_score_mismatch() {
        // Useful idiot profile tested against state actor template
        let axes = vec![0.3, 0.8, 0.5, 0.8, 0.1, 0.3, 0.5, 0.2, 0.3, 0.1];
        let template = &default_templates()[0]; // StateActor
        let score =
            compute_template_score(&axes, &template.expected_ranges, &template.axis_weights);
        assert!(
            score < 0.6,
            "Should score low against state actor: {}",
            score
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![
            (Archetype::StateActor, 0.5),
            (Archetype::OrganicConspiracist, 0.3),
            (Archetype::CommercialTrollFarm, 0.2),
        ];
        let max_score = 0.5_f64;
        let exp_sum: f64 = scores.iter().map(|(_, s)| (s - max_score).exp()).sum();
        let normalized: Vec<f64> = scores
            .iter()
            .map(|(_, s)| (s - max_score).exp() / exp_sum)
            .collect();
        let total: f64 = normalized.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_nan_axes() {
        let axes = vec![f64::NAN; 10];
        let template = &default_templates()[0];
        let score =
            compute_template_score(&axes, &template.expected_ranges, &template.axis_weights);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_archetype_from_str() {
        assert_eq!(
            "state_actor".parse::<Archetype>().unwrap(),
            Archetype::StateActor
        );
        assert_eq!(
            "hacktivist".parse::<Archetype>().unwrap(),
            Archetype::Hacktivist
        );
        assert_eq!(
            "troll_farm".parse::<Archetype>().unwrap(),
            Archetype::CommercialTrollFarm
        );
        assert!("unknown_type".parse::<Archetype>().is_err());
    }

    #[test]
    fn test_archetype_display() {
        assert_eq!(Archetype::StateActor.to_string(), "state_actor");
        assert_eq!(
            Archetype::CommercialTrollFarm.to_string(),
            "commercial_troll_farm"
        );
        assert_eq!(Archetype::UsefulIdiot.to_string(), "useful_idiot");
    }
}
