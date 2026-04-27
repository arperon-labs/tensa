//! Descriptor-row → InferenceJob dispatch helpers.
//!
//! Both the embedded MCP backend and the `POST /infer` REST endpoint turn a
//! parsed-and-executed descriptor row from [`crate::query::executor`] into a
//! concrete [`InferenceJob`] and submit it to the job queue. These helpers
//! are the single source of truth for that conversion so the HTTP and
//! embedded paths can't drift again.
//!
//! The executor emits one descriptor row for every JOB or DISCOVERY plan.
//! `_infer_type` / `_discover_type` names the variant; `_parameters` carries
//! any extracted narrative_id / target_id / assumption / rationality fields.

use std::collections::HashMap;

use serde_json::Value;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::types::InferenceJobType;

/// A descriptor row is a flat map the executor hands back for job queries.
pub type DescriptorRow = HashMap<String, Value>;

/// Resolve `_infer_type` / `_discover_type` to a concrete [`InferenceJobType`].
///
/// Returns `Err(InvalidQuery)` if the row carries neither key or if the value
/// is unknown — callers should treat that as "not a job descriptor, fall
/// through to plain result return."
pub fn infer_type_from_row(row: &DescriptorRow) -> Result<InferenceJobType> {
    // Target-type overrides: `INFER ARCS FOR e:Actor` dispatches to the
    // per-actor engine even though the keyword is plain `ARCS`. This keeps
    // the grammar symmetric with the way PageRank etc. work — same keyword,
    // scope inferred from the binding type.
    let target_type = row.get("_target_type").and_then(|v| v.as_str());

    if let Some(val) = row.get("_infer_type").and_then(|v| v.as_str()) {
        return match val {
            "Causes" => Ok(InferenceJobType::CausalDiscovery),
            "Motivation" => Ok(InferenceJobType::MotivationInference),
            "Game" => Ok(InferenceJobType::GameClassification),
            "Counterfactual" => Ok(InferenceJobType::Counterfactual),
            "Missing" => Ok(InferenceJobType::MissingLinks),
            "MissingEvents" => Ok(InferenceJobType::MissingEventPrediction),
            "Anomalies" => Ok(InferenceJobType::AnomalyDetection),
            "Centrality" | "Communities" => Ok(InferenceJobType::CentralityAnalysis),
            "Entropy" => Ok(InferenceJobType::EntropyAnalysis),
            "Beliefs" => Ok(InferenceJobType::BeliefModeling),
            "Evidence" => Ok(InferenceJobType::EvidenceCombination),
            "Arguments" => Ok(InferenceJobType::ArgumentationAnalysis),
            "Contagion" => Ok(InferenceJobType::ContagionAnalysis),
            // Phase 7b — INFER HIGHER_ORDER_CONTAGION(<json>) FOR n:Narrative.
            // Routes to a SurrogateContagionSignificance job with default
            // model = "eath" and a sentinel k = 0 (the engine clamps to ≥ 1).
            // Callers wanting K > 1 must use POST /synth/contagion-significance
            // directly — the TensaQL form is for the inline real-narrative
            // simulation reachable via the engine's K=1 single-pass mode.
            "HigherOrderContagion" => Ok(InferenceJobType::SurrogateContagionSignificance {
                narrative_id: String::new(),
                k: 1,
                model: "eath".into(),
                contagion_params: serde_json::Value::Null,
            }),
            // Phase 14 — INFER CONTAGION_BISTABILITY(...) FOR n:Narrative.
            // Routes to a SurrogateBistabilitySignificance job. The grammar
            // doesn't carry the BistabilitySweepParams blob (executor wraps
            // the function args separately); the engine accepts a Null
            // params blob and falls back to defaults via parse_params, so
            // callers wanting full control should use POST
            // /synth/bistability-significance directly.
            "ContagionBistability" => Ok(InferenceJobType::SurrogateBistabilitySignificance {
                narrative_id: String::new(),
                params: serde_json::Value::Null,
                k: 1,
                models: Vec::new(),
            }),
            "Style" => Ok(InferenceJobType::StyleProfile),
            "StyleCompare" => Ok(InferenceJobType::StyleComparison),
            "StyleAnomalies" => Ok(InferenceJobType::StyleAnomaly),
            "VerifyAuthorship" => Ok(InferenceJobType::AuthorshipVerification),
            "TemporalRules" => Ok(InferenceJobType::TemporalILP),
            "MeanField" => Ok(InferenceJobType::MeanFieldGame),
            "Psl" => Ok(InferenceJobType::ProbabilisticSoftLogic),
            "Trajectory" => Ok(InferenceJobType::TrajectoryEmbedding),
            "Simulate" => Ok(InferenceJobType::NarrativeSimulation),
            // EATH Phase 15b: SINDy hypergraph reconstruction. Phase 15c
            // adds the TensaQL grammar (`INFER HYPERGRAPH FROM DYNAMICS`);
            // for now this entry exists so callers that hand-build a
            // descriptor row can route through the same machinery as other
            // INFER verbs. Sentinel narrative_id + null params — the
            // engine resolves both from `parameters` at execution time.
            "HypergraphReconstruction" => Ok(InferenceJobType::HypergraphReconstruction {
                narrative_id: String::new(),
                params: serde_json::Value::Null,
            }),
            // `INFER ARCS FOR e:Actor` → per-actor classification.
            // `INFER ARCS FOR n:Narrative` → narrative-level (default).
            "Arcs" => match target_type {
                Some("Actor") => Ok(InferenceJobType::ActorArcClassification),
                _ => Ok(InferenceJobType::ArcClassification),
            },
            "ActorArcs" => Ok(InferenceJobType::ActorArcClassification),
            "Patterns" => Ok(InferenceJobType::PatternMining),
            #[cfg(feature = "disinfo")]
            "BehavioralFingerprint" => Ok(InferenceJobType::BehavioralFingerprint),
            #[cfg(feature = "disinfo")]
            "DisinfoFingerprint" => Ok(InferenceJobType::DisinfoFingerprint),
            #[cfg(feature = "disinfo")]
            "SpreadVelocity" => Ok(InferenceJobType::SpreadVelocity),
            #[cfg(feature = "disinfo")]
            "SpreadIntervention" => Ok(InferenceJobType::SpreadIntervention),
            #[cfg(feature = "disinfo")]
            "Cib" => Ok(InferenceJobType::CibDetection),
            #[cfg(feature = "disinfo")]
            "Superspreaders" => Ok(InferenceJobType::Superspreaders),
            #[cfg(feature = "adversarial")]
            "AdversaryPolicy" => Ok(InferenceJobType::AdversaryPolicy),
            #[cfg(feature = "adversarial")]
            "CognitiveHierarchy" => Ok(InferenceJobType::CognitiveHierarchy),
            #[cfg(feature = "adversarial")]
            "Wargame" => Ok(InferenceJobType::WargameSimulation),
            #[cfg(feature = "adversarial")]
            "CounterNarrative" => Ok(InferenceJobType::CounterNarrative),
            #[cfg(feature = "adversarial")]
            "RewardFingerprint" => Ok(InferenceJobType::RewardFingerprint),
            #[cfg(feature = "adversarial")]
            "Retrodiction" => Ok(InferenceJobType::Retrodiction),
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown infer type: {}",
                other
            ))),
        };
    }
    if let Some(val) = row.get("_discover_type").and_then(|v| v.as_str()) {
        return match val {
            "Patterns" => Ok(InferenceJobType::PatternMining),
            "Arcs" => Ok(InferenceJobType::ArcClassification),
            "Missing" => Ok(InferenceJobType::MissingEventPrediction),
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown discover type: {}",
                other
            ))),
        };
    }
    // EATH Phase 5 — surrogate descriptors. Carries `_synth_kind` plus
    // variant-specific fields (`_synth_model`, `_synth_narrative_id`, …).
    // Rebuild the typed `InferenceJobType` here so HTTP `/infer` and
    // embedded MCP submission routes pick the new TensaQL verbs up
    // automatically — same single-source-of-truth mapping table as
    // INFER / DISCOVER.
    if let Some(kind) = row.get("_synth_kind").and_then(|v| v.as_str()) {
        return match kind {
            "SurrogateCalibration" => {
                let narrative_id = row
                    .get("_synth_narrative_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let model = row
                    .get("_synth_model")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                Ok(InferenceJobType::SurrogateCalibration {
                    narrative_id,
                    model,
                })
            }
            "SurrogateGeneration" => {
                let source_narrative_id = row
                    .get("_synth_source_narrative_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
                let output_narrative_id = row
                    .get("_synth_output_narrative_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let model = row
                    .get("_synth_model")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let params_override = row.get("_synth_params_override").cloned();
                let seed_override = row.get("_synth_seed").and_then(|v| v.as_u64());
                Ok(InferenceJobType::SurrogateGeneration {
                    source_narrative_id,
                    output_narrative_id,
                    model,
                    params_override,
                    seed_override,
                })
            }
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown synth kind: {}",
                other
            ))),
        };
    }
    Err(TensaError::InvalidQuery(
        "No _infer_type, _discover_type, or _synth_kind in descriptor row".into(),
    ))
}

/// Try to pull a real UUID `target_id` out of the descriptor's `_parameters`.
///
/// Narrative-scoped jobs rarely carry a UUID — the engine reads
/// `parameters.narrative_id` instead. Callers should fall back to
/// `Uuid::now_v7()` when this returns `None`.
pub fn extract_target_id(row: &DescriptorRow) -> Option<Uuid> {
    if let Some(params) = row.get("_parameters") {
        if let Some(nid) = params.get("narrative_id").and_then(|v| v.as_str()) {
            if let Ok(uuid) = Uuid::try_parse(nid) {
                return Some(uuid);
            }
        }
        if let Some(tid) = params.get("target_id").and_then(|v| v.as_str()) {
            if let Ok(uuid) = Uuid::try_parse(tid) {
                return Some(uuid);
            }
        }
    }
    None
}

/// Extract the `_parameters` object from the descriptor row.
///
/// For surrogate (EATH Phase 5) descriptors that don't carry a
/// `_parameters` blob, assemble one from the optional `_synth_num_steps`
/// + `_synth_label_prefix` fields so the engine reads them from
/// `job.parameters` as documented. Returning `None` here would force
/// the engine to fall back to its hard-coded defaults (100 steps,
/// "synth" prefix), silently dropping the user's STEPS / LABEL_PREFIX
/// clauses.
pub fn extract_parameters(row: &DescriptorRow) -> Option<Value> {
    if let Some(p) = row.get("_parameters") {
        return Some(p.clone());
    }
    // Synth descriptor path. Build a parameters blob if the descriptor
    // carried any synth-specific knobs the engine needs.
    if row.get("_synth_kind").is_some() {
        let mut map = serde_json::Map::new();
        if let Some(n) = row.get("_synth_num_steps") {
            map.insert("num_steps".into(), n.clone());
        }
        if let Some(lp) = row.get("_synth_label_prefix") {
            map.insert("label_prefix".into(), lp.clone());
        }
        if !map.is_empty() {
            return Some(Value::Object(map));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn infer_type_arcs_maps_to_arc_classification() {
        let mut row = DescriptorRow::new();
        row.insert("_infer_type".into(), json!("Arcs"));
        assert_eq!(
            infer_type_from_row(&row).unwrap(),
            InferenceJobType::ArcClassification
        );
    }

    #[test]
    fn infer_type_patterns_maps_to_pattern_mining() {
        let mut row = DescriptorRow::new();
        row.insert("_infer_type".into(), json!("Patterns"));
        assert_eq!(
            infer_type_from_row(&row).unwrap(),
            InferenceJobType::PatternMining
        );
    }

    #[test]
    fn discover_type_arcs_still_works() {
        let mut row = DescriptorRow::new();
        row.insert("_discover_type".into(), json!("Arcs"));
        assert_eq!(
            infer_type_from_row(&row).unwrap(),
            InferenceJobType::ArcClassification
        );
    }

    #[test]
    fn missing_type_returns_err() {
        let row = DescriptorRow::new();
        assert!(infer_type_from_row(&row).is_err());
    }

    #[test]
    fn extract_target_id_from_narrative_uuid() {
        let mut row = DescriptorRow::new();
        let uuid = Uuid::now_v7();
        row.insert(
            "_parameters".into(),
            json!({"narrative_id": uuid.to_string()}),
        );
        assert_eq!(extract_target_id(&row), Some(uuid));
    }

    #[test]
    fn extract_target_id_from_string_narrative_returns_none() {
        let mut row = DescriptorRow::new();
        row.insert("_parameters".into(), json!({"narrative_id": "dracula"}));
        assert_eq!(extract_target_id(&row), None);
    }
}
