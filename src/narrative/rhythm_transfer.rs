//! Rhythm transfer and fingerprint layer composition.
//!
//! Decompose narrative fingerprints into independent layers and recombine
//! layers from different source narratives. Take pacing from a thriller,
//! character arcs from Tolstoy, commitment density from Christie.

use serde::{Deserialize, Serialize};

use crate::analysis::style_profile::{NarrativeStyleProfile, WeightedSimilarityConfig};
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// A named fingerprint layer that can be extracted and transplanted.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FingerprintLayer {
    /// Situation density, participants, arc type.
    StructuralRhythm,
    /// Game types, role entropy, power asymmetry.
    CharacterDynamics,
    /// Info R0, deception, knowledge asymmetry.
    InformationFlow,
    /// Causal density, chain length, branching.
    CausalArchitecture,
    /// Allen relations, flashbacks, temporal spans.
    TemporalTexture,
    /// WL hash, community count, graph topology.
    GraphTopology,
    /// Promise fulfillment, payoff distance, irony, arcs, pacing.
    GenerativeArchitecture,
}

/// A recipe for composing a target fingerprint from multiple sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRecipe {
    pub label: String,
    /// (layer, source narrative ID) pairs.
    pub layers: Vec<(FingerprintLayer, String)>,
}

/// Result of composing a fingerprint from multiple sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedFingerprint {
    pub recipe: CompositionRecipe,
    pub profile: NarrativeStyleProfile,
    /// Which layers were successfully sourced.
    pub sourced_layers: Vec<FingerprintLayer>,
    /// Which layers had to use defaults (source not found).
    pub defaulted_layers: Vec<FingerprintLayer>,
}

// ─── Composition ────────────────────────────────────────────

/// Compose a target fingerprint by picking layers from different source narratives.
pub fn compose_fingerprint(
    hg: &Hypergraph,
    recipe: &CompositionRecipe,
) -> Result<ComposedFingerprint> {
    use crate::analysis::style_profile::compute_style_profile;

    // Start with a blank profile
    let mut profiles: std::collections::HashMap<String, NarrativeStyleProfile> =
        std::collections::HashMap::new();

    // Load profiles for each unique source narrative
    let source_ids: std::collections::HashSet<&str> =
        recipe.layers.iter().map(|(_, nid)| nid.as_str()).collect();

    for nid in &source_ids {
        match compute_style_profile(nid, hg) {
            Ok(profile) => {
                profiles.insert(nid.to_string(), profile);
            }
            Err(_) => {} // Will be reported as defaulted
        }
    }

    if profiles.is_empty() {
        return Err(TensaError::InvalidQuery(
            "no source narratives could be profiled".into(),
        ));
    }

    // Start with the first available profile as baseline
    let first_nid = recipe
        .layers
        .first()
        .map(|(_, nid)| nid.as_str())
        .unwrap_or("");
    let mut composed = profiles
        .get(first_nid)
        .or_else(|| profiles.values().next())
        .cloned()
        .ok_or_else(|| TensaError::InvalidQuery("no profiles available".into()))?;

    let mut sourced = Vec::new();
    let mut defaulted = Vec::new();

    for (layer, source_nid) in &recipe.layers {
        if let Some(source) = profiles.get(source_nid) {
            transfer_layer(&mut composed, source, layer);
            sourced.push(layer.clone());
        } else {
            defaulted.push(layer.clone());
        }
    }

    composed.narrative_id = format!("composed:{}", recipe.label);

    Ok(ComposedFingerprint {
        recipe: recipe.clone(),
        profile: composed,
        sourced_layers: sourced,
        defaulted_layers: defaulted,
    })
}

/// Transfer a specific layer from source profile to target profile.
fn transfer_layer(
    target: &mut NarrativeStyleProfile,
    source: &NarrativeStyleProfile,
    layer: &FingerprintLayer,
) {
    match layer {
        FingerprintLayer::StructuralRhythm => {
            target.situation_density_curve = source.situation_density_curve.clone();
            target.avg_participants_per_situation = source.avg_participants_per_situation;
            target.participation_count_variance = source.participation_count_variance;
            target.arc_type = source.arc_type.clone();
            target.arc_confidence = source.arc_confidence;
            target.situation_entity_ratio = source.situation_entity_ratio;
        }
        FingerprintLayer::CharacterDynamics => {
            target.game_type_distribution = source.game_type_distribution.clone();
            target.role_entropy = source.role_entropy;
            target.power_asymmetry_gini = source.power_asymmetry_gini;
            target.co_participation_density = source.co_participation_density;
            target.protagonist_concentration = source.protagonist_concentration;
        }
        FingerprintLayer::InformationFlow => {
            target.avg_info_r0 = source.avg_info_r0;
            target.deception_index = source.deception_index;
            target.knowledge_asymmetry_gini = source.knowledge_asymmetry_gini;
            target.revelation_timing = source.revelation_timing;
            target.secret_survival_rate = source.secret_survival_rate;
        }
        FingerprintLayer::CausalArchitecture => {
            target.causal_density = source.causal_density;
            target.avg_causal_chain_length = source.avg_causal_chain_length;
            target.max_causal_chain_length = source.max_causal_chain_length;
            target.unexplained_event_ratio = source.unexplained_event_ratio;
            target.causal_branching_factor = source.causal_branching_factor;
            target.causal_convergence_factor = source.causal_convergence_factor;
        }
        FingerprintLayer::TemporalTexture => {
            target.allen_relation_distribution = source.allen_relation_distribution.clone();
            target.flashback_frequency = source.flashback_frequency;
            target.temporal_span_variance = source.temporal_span_variance;
            target.temporal_gap_ratio = source.temporal_gap_ratio;
        }
        FingerprintLayer::GraphTopology => {
            target.wl_hash_histogram = source.wl_hash_histogram.clone();
            target.wl_simhash = source.wl_simhash;
            target.community_count = source.community_count;
            target.avg_shortest_path = source.avg_shortest_path;
            target.graph_diameter = source.graph_diameter;
            target.edge_density = source.edge_density;
        }
        FingerprintLayer::GenerativeArchitecture => {
            target.promise_fulfillment_ratio = source.promise_fulfillment_ratio;
            target.average_payoff_distance = source.average_payoff_distance;
            target.fabula_sjuzet_divergence = source.fabula_sjuzet_divergence;
            target.dramatic_irony_density = source.dramatic_irony_density;
            target.focalization_diversity = source.focalization_diversity;
            target.character_arc_completeness = source.character_arc_completeness;
            target.subplot_convergence_ratio = source.subplot_convergence_ratio;
            target.scene_sequel_rhythm_score = source.scene_sequel_rhythm_score;
        }
    }
}

/// Build a WeightedSimilarityConfig that emphasizes specific layers.
pub fn weights_for_layers(layers: &[FingerprintLayer]) -> WeightedSimilarityConfig {
    let mut config = WeightedSimilarityConfig::default();
    // Start with low weights for all
    config.w_prose_fw = 0.1;
    config.w_prose_scalars = 0.1;
    config.w_rhythm = 0.1;
    config.w_character = 0.1;
    config.w_info = 0.1;
    config.w_causal = 0.1;
    config.w_temporal = 0.1;
    config.w_topology = 0.1;
    config.w_generative = 0.1;

    // Boost selected layers
    for layer in layers {
        match layer {
            FingerprintLayer::StructuralRhythm => config.w_rhythm = 2.0,
            FingerprintLayer::CharacterDynamics => config.w_character = 2.0,
            FingerprintLayer::InformationFlow => config.w_info = 2.0,
            FingerprintLayer::CausalArchitecture => config.w_causal = 2.0,
            FingerprintLayer::TemporalTexture => config.w_temporal = 2.0,
            FingerprintLayer::GraphTopology => config.w_topology = 2.0,
            FingerprintLayer::GenerativeArchitecture => config.w_generative = 2.0,
        }
    }

    config
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_for_layers() {
        let weights = weights_for_layers(&[
            FingerprintLayer::CausalArchitecture,
            FingerprintLayer::GenerativeArchitecture,
        ]);
        assert!(weights.w_causal > weights.w_rhythm);
        assert!(weights.w_generative > weights.w_topology);
        assert!((weights.w_causal - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_fingerprint_layer_serialization() {
        let layer = FingerprintLayer::GenerativeArchitecture;
        let json = serde_json::to_string(&layer).unwrap();
        let back: FingerprintLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(layer, back);
    }
}
