use serde::Serialize;

use crate::error::Result;
use crate::query::parser::*;

/// Classification of a query.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum QueryClass {
    Instant,
    Job,
    Discovery,
    Mutation,
    Export,
    /// RAG question-answering query (ASK).
    Ask,
}

/// A step in the execution plan.
#[derive(Debug, Clone, Serialize)]
pub enum PlanStep {
    /// Scan entities or situations by type name.
    ScanByType { type_name: String, binding: String },
    /// Follow participation edges between entities and situations.
    FollowEdge {
        from_binding: String,
        rel_type: String,
        to_binding: String,
        directed: bool,
    },
    /// Filter items by property conditions (supports AND, OR, parenthesized grouping).
    FilterProperties {
        expr: ConditionExpr,
        /// Fuzzy Sprint Phase 3 — optional per-query fuzzy override that
        /// drives the AND/OR fusion inside the condition tree. `None` =
        /// engine default (Gödel short-circuit boolean fold from Phase 1).
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Filter by temporal relation.
    FilterTemporal {
        at_clause: AtClause,
        /// Phase 3 — fuzzy config slot; threshold-based interval merging
        /// will consume this once the fuzzy-Allen surface phase lands.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Vector similarity search.
    VectorNear {
        binding: String,
        query_text: String,
        k: usize,
        /// Phase 3 — fuzzy config slot used when fusing vector similarity
        /// scores with keyword / PPR scores downstream.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Submit an inference job.
    SubmitInferenceJob {
        infer_type: InferType,
        target_binding: String,
        target_type: String,
        parameters: serde_json::Value,
        /// Phase 3 — per-job fuzzy override. Consumed by the engine
        /// dispatcher via `parameters.fuzzy_config` when present; INFER
        /// EVIDENCE + ARGUMENTS route t-norm through `combine_with_tnorm`,
        /// and aggregator-consuming engines pick it up through the same
        /// JSON slot.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Filter bindings to specific narratives.
    FilterNarrative { narrative_ids: Option<Vec<String>> },
    /// Submit a cross-narrative discovery job.
    SubmitDiscoverJob {
        discover_type: DiscoverType,
        target_binding: Option<String>,
        target_type: Option<String>,
        narrative_ids: Option<Vec<String>>,
        parameters: serde_json::Value,
    },
    /// Filter by geospatial proximity (Haversine distance).
    FilterSpatial {
        field: String,
        radius_km: f64,
        center_lat: f64,
        center_lon: f64,
    },
    /// Aggregate results with GROUP BY and aggregate functions.
    Aggregate {
        group_by: Vec<String>,
        aggregates: Vec<(AggregateFunc, String)>,
    },
    /// Project final results.
    Project { return_clause: ReturnClause },
    /// Find paths between two nodes via causal or participation edges.
    FindPath {
        start_binding: String,
        end_binding: String,
        rel_type: String,
        mode: PathMode,
        restrictor: PathRestrictor,
        min_depth: usize,
        max_depth: usize,
        top_k: Option<usize>,
        weight_field: Option<String>,
    },
    /// Execute max-flow or min-cut between two nodes.
    FindFlow {
        start_binding: String,
        end_binding: String,
        rel_type: String,
        flow_type: FlowType,
    },
    /// Execute a DML mutation.
    Mutate { mutation: MutationStatement },
    /// Export a narrative in a given format.
    Export {
        narrative_id: String,
        format: String,
    },
    /// Execute a RAG question-answering query.
    AskLlm {
        question: String,
        narrative_id: Option<String>,
        mode: String,
        response_type: Option<String>,
        session_id: Option<String>,
        suggest: bool,
        /// Phase 3 — fuzzy override used during RAG score fusion (vector ×
        /// keyword × community-summary). Current MVP retrieval path
        /// continues to use its default fuse; this slot carries the user's
        /// selection so EXPLAIN reflects intent and future retrieval wires
        /// can honour it without a grammar/AST change.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Tune extraction prompts for a narrative.
    TunePrompts { narrative_id: String },
    /// EATH Phase 5 — submit a `SurrogateCalibration` async job. Async by
    /// nature: calibration is O(dataset).
    SubmitSurrogateCalibrationJob {
        narrative_id: String,
        model: String,
    },
    /// EATH Phase 5 — submit a `SurrogateGeneration` async job. Honors
    /// optional `params` override (skips the calibration load), `seed`
    /// for deterministic replay, `num_steps` (else engine default of
    /// 100), and `label_prefix` lineage tag (else engine default of
    /// `"synth"`). The latter two flow through the InferenceJob's
    /// `parameters` JSON blob.
    SubmitSurrogateGenerationJob {
        source_narrative_id: String,
        output_narrative_id: String,
        model: String,
        params: Option<serde_json::Value>,
        seed: Option<u64>,
        num_steps: Option<usize>,
        label_prefix: Option<String>,
    },
    /// EATH Phase 9 — submit a `SurrogateHybridGeneration` async job. Distinct
    /// step kind because the engine path multiplexes between sources rather
    /// than running a single EATH process. `components` carries the canonical
    /// JSON serialization of `Vec<HybridComponent>` (validated at the planner
    /// for sum-to-1.0 and non-empty); the engine re-validates.
    SubmitSurrogateHybridGenerationJob {
        output_narrative_id: String,
        components: serde_json::Value,
        seed: Option<u64>,
        num_steps: Option<usize>,
    },
    /// EATH Extension Phase 13c — submit a `SurrogateDualSignificance` async
    /// job. Wires the parsed `COMPUTE DUAL_SIGNIFICANCE` form into the
    /// engine. `models` is always non-None at this layer (planner expands
    /// the `Option<Vec<String>>` AST default into the canonical
    /// `["eath", "nudhy"]` pair when the MODELS clause is omitted).
    SubmitSurrogateDualSignificanceJob {
        narrative_id: String,
        metric: String,
        k_per_model: Option<u16>,
        models: Vec<String>,
    },
    /// EATH Extension Phase 15c — submit a `HypergraphReconstruction` job.
    /// `params_json` carries the canonical JSON serialization of
    /// `ReconstructionParams`; the engine deserializes it via serde and
    /// applies its own defaults for omitted fields. The TensaQL surface only
    /// exposes a small subset (observation source, max_order, lambda); for
    /// full control, callers should use POST /inference/hypergraph-reconstruction
    /// directly.
    SubmitHypergraphReconstructionJob {
        narrative_id: String,
        params_json: serde_json::Value,
        /// Phase 3 — fuzzy config slot.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// EATH Extension Phase 16c — synchronous opinion-dynamics simulation.
    /// Mirrors Phase 14 bistability's sync pattern (`compute_contagion_bistability`).
    /// `params_json` carries the partial JSON the planner assembled from the
    /// optional named clauses.
    RunOpinionDynamics {
        narrative_id: String,
        params_json: serde_json::Value,
        /// Phase 3 — fuzzy config slot.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// EATH Extension Phase 16c — synchronous opinion phase-transition sweep.
    RunOpinionPhaseTransition {
        narrative_id: String,
        c_start: f32,
        c_end: f32,
        c_steps: usize,
        /// Phase 3 — fuzzy config slot.
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Fuzzy Sprint Phase 6 — intermediate-quantifier evaluation. Runs
    /// synchronously inside the executor because the numeric work is an
    /// O(|domain|) scan.
    ///
    /// Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].
    RunQuantifier {
        quantifier: String,
        binding: Option<String>,
        type_name: String,
        where_clause: Option<WhereClause>,
        narrative_id: Option<String>,
        label: Option<String>,
    },
    /// Fuzzy Sprint Phase 7 — graded syllogism verification. Runs
    /// synchronously inside the executor; three quantifier evaluations
    /// at O(|domain|) each. Cites: [murinovanovak2014peterson].
    RunVerifySyllogism {
        major: String,
        minor: String,
        conclusion: String,
        narrative_id: String,
        threshold: Option<f64>,
        /// Phase 3 — fuzzy config slot (t-norm override).
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Fuzzy Sprint Phase 8 — build + persist a graded concept lattice.
    /// Synchronous; no job submission. Cites: [belohlavek2004fuzzyfca].
    RunFcaLattice {
        narrative_id: String,
        threshold: Option<usize>,
        attribute_allowlist: Option<Vec<String>>,
        entity_type: Option<String>,
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Fuzzy Sprint Phase 8 — retrieve one concept from a persisted
    /// lattice by index.
    RunFcaConcept {
        lattice_id: String,
        concept_idx: usize,
    },
    /// Fuzzy Sprint Phase 9 — Mamdani rule-system evaluation.
    /// Cites: [mamdani1975mamdani].
    RunEvaluateRules {
        narrative_id: String,
        entity_type: String,
        rule_ids: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
    /// Fuzzy Sprint Phase 10 — hybrid fuzzy-probability query. Runs
    /// synchronously inside the executor. The three payload strings
    /// are parsed at execute time (keeps the pest grammar small).
    /// Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
    RunFuzzyProbability {
        narrative_id: String,
        event_kind: String,
        event_ref: String,
        distribution: String,
        #[serde(default, skip_serializing_if = "FuzzyPlanConfig::is_empty")]
        fuzzy_config: FuzzyPlanConfig,
    },
}

/// A complete execution plan.
#[derive(Debug, Clone, Serialize)]
pub struct QueryPlan {
    pub class: QueryClass,
    pub steps: Vec<PlanStep>,
}

/// Create an execution plan from a parsed statement (query or mutation).
pub fn plan_statement(stmt: &TensaStatement) -> Result<QueryPlan> {
    match stmt {
        TensaStatement::Query(query) => plan_query(query),
        TensaStatement::Mutation(mutation) => Ok(QueryPlan {
            class: QueryClass::Mutation,
            steps: vec![PlanStep::Mutate {
                mutation: mutation.clone(),
            }],
        }),
        TensaStatement::Export {
            narrative_id,
            format,
        } => Ok(QueryPlan {
            class: QueryClass::Export,
            steps: vec![PlanStep::Export {
                narrative_id: narrative_id.clone(),
                format: format.clone(),
            }],
        }),
        TensaStatement::Tune(tune) => Ok(QueryPlan {
            class: QueryClass::Job,
            steps: vec![PlanStep::TunePrompts {
                narrative_id: tune.narrative_id.clone(),
            }],
        }),
        TensaStatement::CalibrateSurrogate {
            narrative_id,
            model,
        } => {
            validate_surrogate_model(model)?;
            Ok(QueryPlan {
                class: QueryClass::Job,
                steps: vec![PlanStep::SubmitSurrogateCalibrationJob {
                    narrative_id: narrative_id.clone(),
                    model: model.clone(),
                }],
            })
        }
        TensaStatement::GenerateNarrative {
            output_id,
            source_id,
            model,
            params,
            seed,
            num_steps,
            label_prefix,
        } => {
            validate_surrogate_model(model)?;
            // Plan-time check for "calibration must precede generation":
            // the planner has no store handle (deliberate — keeps the
            // planner pure), so we can't read `syn/p/{nid}/{model}` here
            // to confirm calibrated params exist. The
            // `SurrogateGenerationEngine` performs the same check at
            // execute time and returns a message guiding the caller to
            // calibrate or pass `params`. Documented in
            // docs/EATH_sprint.md Notes.
            Ok(QueryPlan {
                class: QueryClass::Job,
                steps: vec![PlanStep::SubmitSurrogateGenerationJob {
                    source_narrative_id: source_id.clone(),
                    output_narrative_id: output_id.clone(),
                    model: model.clone(),
                    params: params.clone(),
                    seed: *seed,
                    num_steps: *num_steps,
                    label_prefix: label_prefix.clone(),
                }],
            })
        }
        TensaStatement::GenerateHybridNarrative {
            output_id,
            components,
            seed,
            num_steps,
        } => {
            // Validate sum at the planner so callers get a synchronous
            // parse-time-ish error instead of a queued-then-failed job.
            let sum: f32 = components.iter().map(|c| c.weight).sum();
            if (sum - 1.0).abs() > crate::synth::hybrid::HYBRID_WEIGHT_TOLERANCE {
                return Err(crate::TensaError::ParseError(format!(
                    "GENERATE HYBRID weights must sum to 1.0 (got {sum})"
                )));
            }
            // Translate parser-side specs → synth-side HybridComponent
            // (defaults model='eath' since the grammar doesn't expose a
            // per-component model — Phase 9 ships EATH-only).
            let synth_components: Vec<crate::synth::hybrid::HybridComponent> = components
                .iter()
                .map(|c| crate::synth::hybrid::HybridComponent {
                    narrative_id: c.narrative_id.clone(),
                    model: "eath".into(),
                    weight: c.weight,
                })
                .collect();
            let components_json = serde_json::to_value(&synth_components).map_err(|e| {
                crate::TensaError::ParseError(format!("hybrid components serialize: {e}"))
            })?;
            Ok(QueryPlan {
                class: QueryClass::Job,
                steps: vec![PlanStep::SubmitSurrogateHybridGenerationJob {
                    output_narrative_id: output_id.clone(),
                    components: components_json,
                    seed: *seed,
                    num_steps: *num_steps,
                }],
            })
        }
        TensaStatement::InferHypergraphReconstruction {
            narrative_id,
            observation,
            max_order,
            lambda,
            fuzzy_config,
        } => {
            // Phase 3 — resolve fuzzy config up-front so unknown names fail
            // at plan time with the registry's "known: [...]" hint. The
            // reconstruction engine does not consume the config in Phase 3
            // (this MVP's algorithms ship with fixed semantics); the field
            // is carried through so EXPLAIN renders it correctly and
            // future phases can opt in without a grammar/AST change.
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            // Build a partial JSON params blob. The engine applies serde
            // defaults for any field we omit (max_order=3,
            // observation=ParticipationRate, lambda_l1=0.0 → auto, etc.).
            let mut params = serde_json::Map::new();
            if let Some(ord) = max_order {
                if !(2..=4).contains(ord) {
                    return Err(crate::TensaError::ParseError(format!(
                        "MAX_ORDER must be in 2..=4, got {ord}"
                    )));
                }
                params.insert("max_order".into(), serde_json::json!(ord));
            }
            if let Some(l) = lambda {
                if *l < 0.0 || !l.is_finite() {
                    return Err(crate::TensaError::ParseError(format!(
                        "LAMBDA must be finite and non-negative, got {l}"
                    )));
                }
                params.insert("lambda_l1".into(), serde_json::json!(l));
            }
            if let Some(src) = observation {
                // Map the lowercase grammar token to the engine's
                // ObservationSource enum variant. Unknown sources surface as
                // a parse-time error rather than a queued-then-failed job.
                let src_lower = src.to_ascii_lowercase();
                let mapped = match src_lower.as_str() {
                    "participation_rate" => serde_json::json!("ParticipationRate"),
                    "sentiment_mean" => serde_json::json!("SentimentMean"),
                    "belief_mass" => {
                        return Err(crate::TensaError::ParseError(
                            "USING OBSERVATION 'belief_mass' requires a `proposition` field; \
                             use POST /inference/hypergraph-reconstruction with an explicit \
                             params blob to set it"
                                .into(),
                        ));
                    }
                    "engagement" => serde_json::json!("Engagement"),
                    other => {
                        return Err(crate::TensaError::ParseError(format!(
                            "Unknown observation source '{other}'; expected one of: \
                             participation_rate, sentiment_mean, belief_mass, engagement"
                        )));
                    }
                };
                params.insert("observation".into(), mapped);
            }
            Ok(QueryPlan {
                class: QueryClass::Job,
                steps: vec![PlanStep::SubmitHypergraphReconstructionJob {
                    narrative_id: narrative_id.clone(),
                    params_json: serde_json::Value::Object(params),
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::InferOpinionDynamics {
            narrative_id,
            confidence_bound,
            variant,
            mu,
            initial,
            fuzzy_config,
        } => {
            // Phase 3 — resolve fuzzy config ahead of the synchronous
            // RunOpinionDynamics plan step. Carried into EXPLAIN via the
            // `fuzzy_config` slot on the plan step itself.
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            // Build a partial JSON params blob. Engine applies serde defaults
            // on every omitted field.
            let mut params = serde_json::Map::new();
            if let Some(c) = confidence_bound {
                if !c.is_finite() || *c <= 0.0 || *c >= 1.0 {
                    return Err(crate::TensaError::ParseError(format!(
                        "OPINION_DYNAMICS confidence_bound must be in (0, 1), got {c}"
                    )));
                }
                params.insert("confidence_bound".into(), serde_json::json!(c));
            }
            if let Some(m) = mu {
                if !m.is_finite() || *m <= 0.0 || *m > 1.0 {
                    return Err(crate::TensaError::ParseError(format!(
                        "OPINION_DYNAMICS mu must be in (0, 1], got {m}"
                    )));
                }
                params.insert("convergence_rate".into(), serde_json::json!(m));
            }
            if let Some(v) = variant {
                let v_lower = v.to_ascii_lowercase();
                let mapped = match v_lower.as_str() {
                    "pairwise" | "pairwise_within" => "pairwise_within",
                    "group_mean" | "groupmean" => "group_mean",
                    other => {
                        return Err(crate::TensaError::ParseError(format!(
                            "OPINION_DYNAMICS variant must be 'pairwise' or 'group_mean', got '{other}'"
                        )));
                    }
                };
                params.insert("model".into(), serde_json::json!(mapped));
            }
            if let Some(i) = initial {
                let i_lower = i.to_ascii_lowercase();
                let mapped = match i_lower.as_str() {
                    "uniform" => serde_json::json!({"kind": "uniform"}),
                    "gaussian" => {
                        serde_json::json!({"kind": "gaussian", "mean": 0.5, "std": 0.15})
                    }
                    "bimodal" => serde_json::json!({
                        "kind": "bimodal", "mode_a": 0.2, "mode_b": 0.8, "spread": 0.05
                    }),
                    other => {
                        return Err(crate::TensaError::ParseError(format!(
                            "OPINION_DYNAMICS initial must be 'uniform', 'gaussian', or 'bimodal', got '{other}'"
                        )));
                    }
                };
                params.insert("initial_opinion_distribution".into(), mapped);
            }
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunOpinionDynamics {
                    narrative_id: narrative_id.clone(),
                    params_json: serde_json::Value::Object(params),
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::Quantify {
            quantifier,
            binding,
            type_name,
            where_clause,
            narrative_id,
            label,
        } => {
            // Validate the quantifier name at plan time so unknown names
            // fail fast (before touching the hypergraph).
            crate::fuzzy::quantifier::quantifier_from_name(quantifier)?;
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunQuantifier {
                    quantifier: quantifier.clone(),
                    binding: binding.clone(),
                    type_name: type_name.clone(),
                    where_clause: where_clause.clone(),
                    narrative_id: narrative_id.clone(),
                    label: label.clone(),
                }],
            })
        }
        TensaStatement::VerifySyllogism {
            major,
            minor,
            conclusion,
            narrative_id,
            threshold,
            fuzzy_config,
        } => {
            // Resolve the t-norm name at plan time so unknown names fail
            // fast. The DSL bodies are parsed lazily at the executor —
            // grammar already enforced the single-quoted shape.
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            if narrative_id.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "VERIFY SYLLOGISM narrative_id is empty".into(),
                ));
            }
            if let Some(t) = threshold {
                if !t.is_finite() || *t < 0.0 || *t > 1.0 {
                    return Err(crate::TensaError::ParseError(format!(
                        "VERIFY SYLLOGISM THRESHOLD must be finite in [0,1]; got {t}"
                    )));
                }
            }
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunVerifySyllogism {
                    major: major.clone(),
                    minor: minor.clone(),
                    conclusion: conclusion.clone(),
                    narrative_id: narrative_id.clone(),
                    threshold: *threshold,
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::InferOpinionPhaseTransition {
            narrative_id,
            c_start,
            c_end,
            c_steps,
            fuzzy_config,
        } => {
            // Phase 3 — resolve fuzzy config so the planner fails fast on
            // unknown names. The phase-transition sweep does not consume
            // the config in Phase 3 (MVP algorithms use fixed semantics).
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            if *c_steps < 2 {
                return Err(crate::TensaError::ParseError(format!(
                    "OPINION_PHASE_TRANSITION c_steps must be >= 2, got {c_steps}"
                )));
            }
            if !c_start.is_finite() || !c_end.is_finite() {
                return Err(crate::TensaError::ParseError(
                    "OPINION_PHASE_TRANSITION c_start/c_end must be finite".into(),
                ));
            }
            if *c_start <= 0.0 || *c_end >= 1.0 || c_start >= c_end {
                return Err(crate::TensaError::ParseError(format!(
                    "OPINION_PHASE_TRANSITION require 0 < c_start < c_end < 1, got ({c_start}, {c_end})"
                )));
            }
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunOpinionPhaseTransition {
                    narrative_id: narrative_id.clone(),
                    c_start: *c_start,
                    c_end: *c_end,
                    c_steps: *c_steps,
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::FcaLattice {
            narrative_id,
            threshold,
            attribute_allowlist,
            entity_type,
            fuzzy_config,
        } => {
            if narrative_id.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "FCA LATTICE narrative_id is empty".into(),
                ));
            }
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunFcaLattice {
                    narrative_id: narrative_id.clone(),
                    threshold: *threshold,
                    attribute_allowlist: attribute_allowlist.clone(),
                    entity_type: entity_type.clone(),
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::FcaConcept {
            lattice_id,
            concept_idx,
        } => {
            if lattice_id.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "FCA CONCEPT lattice_id is empty".into(),
                ));
            }
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunFcaConcept {
                    lattice_id: lattice_id.clone(),
                    concept_idx: *concept_idx,
                }],
            })
        }
        TensaStatement::EvaluateRules {
            narrative_id,
            entity_type,
            rule_ids,
            fuzzy_config,
        } => {
            if narrative_id.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "EVALUATE RULES narrative_id is empty".into(),
                ));
            }
            if entity_type.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "EVALUATE RULES entity_type is empty".into(),
                ));
            }
            if let Some(ids) = rule_ids {
                if ids.is_empty() {
                    return Err(crate::TensaError::ParseError(
                        "EVALUATE RULES rule-id list is empty".into(),
                    ));
                }
            }
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunEvaluateRules {
                    narrative_id: narrative_id.clone(),
                    entity_type: entity_type.clone(),
                    rule_ids: rule_ids.clone(),
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::FuzzyProbability {
            narrative_id,
            event_kind,
            event_ref,
            distribution,
            fuzzy_config,
        } => {
            if narrative_id.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "INFER FUZZY_PROBABILITY narrative_id is empty".into(),
                ));
            }
            // Validate event_kind at plan time so unknown kinds fail
            // fast rather than at the executor's dispatch.
            match event_kind.to_ascii_lowercase().as_str() {
                "quantifier" | "mamdani_rule" | "custom" => {}
                other => {
                    return Err(crate::TensaError::ParseError(format!(
                        "INFER FUZZY_PROBABILITY unknown event_kind '{other}'; \
                         expected one of: quantifier, mamdani_rule, custom"
                    )));
                }
            }
            if event_ref.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "INFER FUZZY_PROBABILITY event_ref is empty".into(),
                ));
            }
            if distribution.trim().is_empty() {
                return Err(crate::TensaError::ParseError(
                    "INFER FUZZY_PROBABILITY distribution is empty".into(),
                ));
            }
            let resolved_fuzzy = resolve_fuzzy_config(fuzzy_config)?;
            Ok(QueryPlan {
                class: QueryClass::Instant,
                steps: vec![PlanStep::RunFuzzyProbability {
                    narrative_id: narrative_id.clone(),
                    event_kind: event_kind.clone(),
                    event_ref: event_ref.clone(),
                    distribution: distribution.clone(),
                    fuzzy_config: resolved_fuzzy,
                }],
            })
        }
        TensaStatement::ComputeDualSignificance {
            narrative_id,
            metric,
            k_per_model,
            models,
        } => {
            // Default models when the AST carries None (MODELS clause omitted).
            // Single source of truth: the engine const re-export.
            let models: Vec<String> = match models {
                Some(v) if !v.is_empty() => v.clone(),
                _ => crate::synth::dual_significance_engine::DEFAULT_MODELS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            };
            // Plan-time validation: every requested model must exist in the
            // registry. Fails fast with the registry's "did you mean" list.
            for m in &models {
                validate_surrogate_model(m)?;
            }
            Ok(QueryPlan {
                class: QueryClass::Job,
                steps: vec![PlanStep::SubmitSurrogateDualSignificanceJob {
                    narrative_id: narrative_id.clone(),
                    metric: metric.clone(),
                    k_per_model: *k_per_model,
                    models,
                }],
            })
        }
    }
}

/// Fuzzy Sprint Phase 3 — resolve a parser-side [`FuzzyConfig`] into the
/// planner-side [`FuzzyPlanConfig`].
///
/// Validation happens here (not in the parser) so the executor only ever
/// sees concrete t-norm / aggregator kinds:
///
/// * Unknown t-norm name → `TensaError::InvalidInput` via
///   `TNormRegistry::default().get(name)` — includes the "known: [...]" hint.
/// * Unknown aggregator variant is impossible at this layer (the parser
///   covers the six enumerated forms) but we still enumerate every variant
///   so new `AggregatorSpec` variants fail fast with an actionable message.
/// * OWA weights are validated to be non-empty and to sum to 1.0 ± 1e-9 so
///   callers can't silently get an OWA that's 10× too small.
/// * Choquet measure refs are NOT resolvable in Phase 3 — the `/fuzzy/measures`
///   CRUD surface lands in Phase 4. Until then the planner rejects any
///   `CHOQUET '<name>'` form with a clear error pointing at Phase 4.
///
/// Cites: [klement2000] [yager1988owa] [grabisch1996choquet].
pub(crate) fn resolve_fuzzy_config(cfg: &FuzzyConfig) -> Result<FuzzyPlanConfig> {
    let tnorm = match &cfg.tnorm {
        None => None,
        Some(name) => {
            let registry = crate::fuzzy::registry::TNormRegistry::default();
            Some(registry.get(name)?)
        }
    };
    let aggregator = match &cfg.aggregator {
        None => None,
        Some(spec) => Some(resolve_aggregator_spec(spec)?),
    };
    Ok(FuzzyPlanConfig { tnorm, aggregator })
}

/// Resolve an [`AggregatorSpec`] to a concrete [`AggregatorKind`]. OWA
/// weights are checked for non-emptiness and unit-sum; Choquet refs error
/// with a Phase 4 pointer.
fn resolve_aggregator_spec(
    spec: &AggregatorSpec,
) -> Result<crate::fuzzy::aggregation::AggregatorKind> {
    use crate::fuzzy::aggregation::AggregatorKind;
    match spec {
        AggregatorSpec::Mean => Ok(AggregatorKind::Mean),
        AggregatorSpec::Median => Ok(AggregatorKind::Median),
        AggregatorSpec::Owa(weights) => {
            if weights.is_empty() {
                return Err(crate::TensaError::InvalidInput(
                    "AGGREGATE OWA requires at least one weight".into(),
                ));
            }
            let sum: f64 = weights.iter().copied().sum();
            if (sum - 1.0).abs() > 1e-9 {
                return Err(crate::TensaError::InvalidInput(format!(
                    "AGGREGATE OWA weights must sum to 1.0 (got {sum}); \
                     use `crate::fuzzy::aggregation_owa::owa_normalize` to auto-scale"
                )));
            }
            Ok(AggregatorKind::Owa(weights.clone()))
        }
        AggregatorSpec::ChoquetByRef(name) => Err(crate::TensaError::InvalidInput(format!(
            "AGGREGATE CHOQUET '{name}' references a named fuzzy measure, \
             but the /fuzzy/measures CRUD surface is scheduled for Phase 4 of \
             the fuzzy sprint. Supply an ad-hoc measure via the synchronous \
             POST /fuzzy/aggregate endpoint once Phase 4 ships."
        ))),
        AggregatorSpec::TNormReduce(kind) => Ok(AggregatorKind::TNormReduce(*kind)),
        AggregatorSpec::TConormReduce(kind) => Ok(AggregatorKind::TConormReduce(*kind)),
    }
}

/// EATH Phase 5 — single-source-of-truth lookup against the default
/// `SurrogateRegistry` to error early on unknown model names. The
/// registry is constructed-once-per-call here intentionally — it's a
/// HashMap with only the built-in EATH model registered, so the cost is
/// trivial and the planner stays free of long-lived state.
fn validate_surrogate_model(model: &str) -> Result<()> {
    let registry = crate::synth::registry::SurrogateRegistry::default();
    if registry.get(model).is_err() {
        return Err(crate::TensaError::SynthFailure(format!(
            "unknown surrogate model '{}'; available: {:?}",
            model,
            registry.list()
        )));
    }
    Ok(())
}

/// Create a query plan from a parsed AST.
pub fn plan_query(query: &TensaQuery) -> Result<QueryPlan> {
    // Check if this is a discovery query
    if let Some(discover) = &query.discover_clause {
        return plan_discover_query(query, discover);
    }

    // Check if this is an inference query
    if let Some(infer) = &query.infer_clause {
        return plan_infer_query(query, infer);
    }

    // Check if this is an ASK (RAG) query
    if let Some(ask) = &query.ask_clause {
        return plan_ask_query(query, ask);
    }

    // Check if this is a path query
    if let Some(path) = &query.path_clause {
        return plan_path_query(query, path);
    }

    // Check if this is a flow query
    if let Some(flow) = &query.flow_clause {
        return plan_flow_query(query, flow);
    }

    plan_match_query(query)
}

fn plan_ask_query(query: &TensaQuery, ask: &AskClause) -> Result<QueryPlan> {
    let mode = match ask.mode {
        Some(crate::query::parser::RetrievalMode::Local) => "local",
        Some(crate::query::parser::RetrievalMode::Global) => "global",
        Some(crate::query::parser::RetrievalMode::Hybrid) => "hybrid",
        Some(crate::query::parser::RetrievalMode::Mix) => "mix",
        Some(crate::query::parser::RetrievalMode::Drift) => "drift",
        Some(crate::query::parser::RetrievalMode::Lazy) => "lazy",
        Some(crate::query::parser::RetrievalMode::Ppr) => "ppr",
        None => "hybrid", // default
    };

    let fuzzy = resolve_fuzzy_config(&query.fuzzy_config)?;
    let mut steps = Vec::new();
    steps.push(PlanStep::AskLlm {
        question: ask.question.clone(),
        narrative_id: ask.narrative_id.clone(),
        mode: mode.to_string(),
        response_type: ask.response_type.clone(),
        session_id: ask.session_id.clone(),
        suggest: ask.suggest,
        fuzzy_config: fuzzy,
    });
    steps.push(PlanStep::Project {
        return_clause: query.return_clause.clone(),
    });

    Ok(QueryPlan {
        class: QueryClass::Ask,
        steps,
    })
}

fn plan_discover_query(query: &TensaQuery, discover: &DiscoverClause) -> Result<QueryPlan> {
    let mut steps = Vec::new();

    let narrative_ids = query
        .across_clause
        .as_ref()
        .and_then(|a| a.narrative_ids.clone());

    let mut params = serde_json::Map::new();
    if let Some(ref ids) = narrative_ids {
        params.insert(
            "narrative_ids".into(),
            serde_json::Value::Array(
                ids.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
    }

    steps.push(PlanStep::SubmitDiscoverJob {
        discover_type: discover.discover_type.clone(),
        target_binding: discover.target_binding.clone(),
        target_type: discover.target_type.clone(),
        narrative_ids,
        parameters: serde_json::Value::Object(params),
    });

    steps.push(PlanStep::Project {
        return_clause: query.return_clause.clone(),
    });

    Ok(QueryPlan {
        class: QueryClass::Discovery,
        steps,
    })
}

fn plan_infer_query(query: &TensaQuery, infer: &InferClause) -> Result<QueryPlan> {
    let mut steps = Vec::new();

    // Build parameters from WHERE, ASSUMING, and UNDER clauses.
    //
    // Lift `narrative_id = "..."` / `target_id = "..."` equalities from WHERE
    // into job parameters. Narrative-scoped engines (ArcClassification,
    // PatternMining, CentralityAnalysis, StyleProfile, …) key on
    // `parameters.narrative_id` — without this lift, the engines fail
    // downstream with "missing narrative_id".
    let mut params = serde_json::Map::new();

    if let Some(where_clause) = &query.where_clause {
        for cond in where_clause.all_conditions() {
            if !matches!(cond.op, CompareOp::Eq) {
                continue;
            }
            let key = match cond
                .field
                .rsplit_once('.')
                .map(|(_, k)| k)
                .unwrap_or(&cond.field)
            {
                "narrative_id" => "narrative_id",
                "target_id" => "target_id",
                // `e.id = "uuid"` — lift as `target_id` so engines that operate
                // on a single entity (per-actor arc, motivation, etc.) can
                // pick it up via `job.target_id`.
                "id" => "target_id",
                _ => continue,
            };
            if let Some(val) = cond.value.to_json().as_str() {
                params
                    .entry(key.to_string())
                    .or_insert_with(|| serde_json::json!(val));
            }
        }
    }

    if let Some(assuming) = &query.assuming_clause {
        let assumptions: Vec<_> = assuming
            .assumptions
            .iter()
            .map(|a| {
                serde_json::json!({
                    "field": a.field,
                    "value": a.value.to_json(),
                })
            })
            .collect();
        params.insert("assumptions".into(), serde_json::Value::Array(assumptions));
    }

    if let Some(under) = &query.under_clause {
        for cond in &under.conditions {
            match cond {
                UnderCondition::Rationality(lambda) => {
                    params.insert("lambda".into(), serde_json::json!(lambda));
                }
                UnderCondition::Information(info) => {
                    params.insert("information".into(), serde_json::json!(info));
                }
            }
        }
    }

    // Phase 3 — resolve + attach fuzzy config so (a) EXPLAIN renders it,
    // (b) the engine dispatcher can read it via `parameters.fuzzy_config`
    // when it chooses to honour the user's selection.
    let fuzzy = resolve_fuzzy_config(&query.fuzzy_config)?;
    if !fuzzy.is_empty() {
        let fuzzy_json = serde_json::to_value(&fuzzy)
            .map_err(|e| crate::TensaError::Serialization(e.to_string()))?;
        params.insert("fuzzy_config".into(), fuzzy_json);
    }

    steps.push(PlanStep::SubmitInferenceJob {
        infer_type: infer.infer_type.clone(),
        target_binding: infer.target_binding.clone(),
        target_type: infer.target_type.clone(),
        parameters: serde_json::Value::Object(params),
        fuzzy_config: fuzzy,
    });

    steps.push(PlanStep::Project {
        return_clause: query.return_clause.clone(),
    });

    Ok(QueryPlan {
        class: QueryClass::Job,
        steps,
    })
}

fn plan_path_query(query: &TensaQuery, path: &PathClause) -> Result<QueryPlan> {
    let mut steps = Vec::new();

    steps.push(PlanStep::FindPath {
        start_binding: path.start_binding.clone(),
        end_binding: path.end_binding.clone(),
        rel_type: path.rel_type.clone(),
        mode: path.mode.clone(),
        restrictor: path.restrictor.clone(),
        min_depth: path.min_depth,
        max_depth: path.max_depth,
        top_k: path.top_k,
        weight_field: path.weight_field.clone(),
    });

    if let Some(ref wc) = query.where_clause {
        let fuzzy = resolve_fuzzy_config(&query.fuzzy_config)?;
        steps.push(PlanStep::FilterProperties {
            expr: wc.expr.clone(),
            fuzzy_config: fuzzy,
        });
    }

    steps.push(PlanStep::Project {
        return_clause: query.return_clause.clone(),
    });

    Ok(QueryPlan {
        class: QueryClass::Instant,
        steps,
    })
}

fn plan_flow_query(query: &TensaQuery, flow: &FlowClause) -> Result<QueryPlan> {
    let mut steps = Vec::new();

    steps.push(PlanStep::FindFlow {
        start_binding: flow.start_binding.clone(),
        end_binding: flow.end_binding.clone(),
        rel_type: flow.rel_type.clone(),
        flow_type: flow.flow_type.clone(),
    });

    steps.push(PlanStep::Project {
        return_clause: query.return_clause.clone(),
    });

    Ok(QueryPlan {
        class: QueryClass::Instant,
        steps,
    })
}

fn plan_match_query(query: &TensaQuery) -> Result<QueryPlan> {
    let class = QueryClass::Instant;
    let mut steps = Vec::new();

    // Phase 3 — resolve the fuzzy config once and clone into every plan
    // step that participates in score/confidence fusion. Errors surface
    // here (not at execute time) per the sprint contract.
    let fuzzy = resolve_fuzzy_config(&query.fuzzy_config)?;

    let match_clause = query
        .match_clause
        .as_ref()
        .ok_or_else(|| crate::TensaError::ParseError("Missing MATCH clause".into()))?;

    // Step 1: Scan steps for each node pattern
    for element in &match_clause.elements {
        match element {
            PatternElement::Node(node) => {
                let binding = node
                    .binding
                    .clone()
                    .unwrap_or_else(|| format!("_anon_{}", node.type_name));
                steps.push(PlanStep::ScanByType {
                    type_name: node.type_name.clone(),
                    binding,
                });
            }
            PatternElement::Edge(_) => {
                // Edges are handled below
            }
        }
    }

    // Step 2: Edge traversal steps
    let elements = &match_clause.elements;
    for (i, element) in elements.iter().enumerate() {
        if let PatternElement::Edge(edge) = element {
            let from_binding = if i > 0 {
                match &elements[i - 1] {
                    PatternElement::Node(n) => n
                        .binding
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", n.type_name)),
                    _ => "_unknown".to_string(),
                }
            } else {
                "_unknown".to_string()
            };
            let to_binding = if i + 1 < elements.len() {
                match &elements[i + 1] {
                    PatternElement::Node(n) => n
                        .binding
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", n.type_name)),
                    _ => "_unknown".to_string(),
                }
            } else {
                "_unknown".to_string()
            };
            // `<-[:REL]-` inverts the logical direction: the arrow points from
            // the right-hand node to the left-hand node. We normalize by
            // swapping from/to so the executor only sees forward edges.
            let (from, to) = if edge.reversed {
                (to_binding, from_binding)
            } else {
                (from_binding, to_binding)
            };
            steps.push(PlanStep::FollowEdge {
                from_binding: from,
                rel_type: edge.rel_type.clone(),
                to_binding: to,
                directed: edge.directed,
            });
        }
    }

    // Step 2b: Narrative filter from ACROSS NARRATIVES
    if let Some(across) = &query.across_clause {
        steps.push(PlanStep::FilterNarrative {
            narrative_ids: across.narrative_ids.clone(),
        });
    }

    // Step 3: Property filters from WHERE
    if let Some(wc) = &query.where_clause {
        steps.push(PlanStep::FilterProperties {
            expr: wc.expr.clone(),
            fuzzy_config: fuzzy.clone(),
        });
    }

    // Step 4: Temporal filter from AT
    if let Some(at) = &query.at_clause {
        steps.push(PlanStep::FilterTemporal {
            at_clause: at.clone(),
            fuzzy_config: fuzzy.clone(),
        });
    }

    // Step 4b: Vector similarity from NEAR
    if let Some(near) = &query.near_clause {
        steps.push(PlanStep::VectorNear {
            binding: near.binding.clone(),
            query_text: near.query_text.clone(),
            k: near.k,
            fuzzy_config: fuzzy.clone(),
        });
    }

    // Step 4c: Spatial proximity filter from SPATIAL
    if let Some(spatial) = &query.spatial_clause {
        steps.push(PlanStep::FilterSpatial {
            field: spatial.field.clone(),
            radius_km: spatial.radius_km,
            center_lat: spatial.center_lat,
            center_lon: spatial.center_lon,
        });
    }

    // Step 5: Aggregate (if GROUP BY or aggregate functions present)
    let has_aggregates = query
        .return_clause
        .expressions
        .iter()
        .any(|e| e.is_aggregate());
    if query.group_by.is_some() || has_aggregates {
        let group_by_fields = query
            .group_by
            .as_ref()
            .map(|g| g.fields.clone())
            .unwrap_or_default();
        let aggregates: Vec<(AggregateFunc, String)> = query
            .return_clause
            .expressions
            .iter()
            .filter_map(|e| match e {
                ReturnExpr::Aggregate { func, field } => Some((func.clone(), field.clone())),
                _ => None,
            })
            .collect();
        steps.push(PlanStep::Aggregate {
            group_by: group_by_fields,
            aggregates,
        });
    }

    // Step 6: Projection
    steps.push(PlanStep::Project {
        return_clause: query.return_clause.clone(),
    });

    Ok(QueryPlan { class, steps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parser::parse_query;

    #[test]
    fn test_plan_simple_scan() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Instant);
        assert!(matches!(
            &plan.steps[0],
            PlanStep::ScanByType { type_name, binding } if type_name == "Actor" && binding == "e"
        ));
    }

    #[test]
    fn test_plan_with_edge() {
        let q = parse_query("MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s").unwrap();
        let plan = plan_query(&q).unwrap();
        // ScanByType for Actor, ScanByType for Situation, FollowEdge, Project
        let scan_count = plan
            .steps
            .iter()
            .filter(|s| matches!(s, PlanStep::ScanByType { .. }))
            .count();
        assert_eq!(scan_count, 2);
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::FollowEdge { .. })));
    }

    #[test]
    fn test_plan_with_where() {
        let q = parse_query(r#"MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::FilterProperties { .. })));
    }

    #[test]
    fn test_plan_with_at() {
        let q = parse_query(r#"MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01" RETURN s"#)
            .unwrap();
        let plan = plan_query(&q).unwrap();
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::FilterTemporal { .. })));
    }

    #[test]
    fn test_plan_always_has_project() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        assert!(matches!(
            plan.steps.last().unwrap(),
            PlanStep::Project { .. }
        ));
    }

    #[test]
    fn test_plan_instant_classification() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Instant);
    }

    #[test]
    fn test_plan_full_query() {
        let q = parse_query(
            r#"MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) WHERE e.confidence > 0.5 AT s.temporal BEFORE "2025-06-01" RETURN e, s LIMIT 10"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        assert!(plan.steps.len() >= 5); // 2 scans + edge + filter + temporal + project
    }

    #[test]
    fn test_plan_preserves_limit() {
        let q = parse_query("MATCH (e:Actor) RETURN e LIMIT 5").unwrap();
        let plan = plan_query(&q).unwrap();
        if let PlanStep::Project { return_clause } = plan.steps.last().unwrap() {
            assert_eq!(return_clause.limit, Some(5));
        } else {
            panic!("Last step should be Project");
        }
    }

    #[test]
    fn test_plan_anonymous_binding() {
        let q = parse_query("MATCH (:Actor) RETURN *").unwrap();
        let plan = plan_query(&q).unwrap();
        if let PlanStep::ScanByType { binding, .. } = &plan.steps[0] {
            assert!(binding.starts_with("_anon_"));
        }
    }

    #[test]
    fn test_plan_order_by_preserved() {
        let q = parse_query("MATCH (e:Actor) RETURN e ORDER BY e.confidence DESC").unwrap();
        let plan = plan_query(&q).unwrap();
        if let PlanStep::Project { return_clause } = plan.steps.last().unwrap() {
            let ob = return_clause.order_by.as_ref().unwrap();
            assert_eq!(ob.field, "e.confidence");
            assert!(!ob.ascending);
        }
    }

    #[test]
    fn test_plan_infer_causes_is_job() {
        let q = parse_query("INFER CAUSES FOR s:Situation RETURN s").unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Job);
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::SubmitInferenceJob { .. })));
    }

    #[test]
    fn test_plan_infer_with_assuming_params() {
        let q = parse_query(
            r#"INFER COUNTERFACTUAL FOR s:Situation ASSUMING s.action = "cooperate" RETURN s"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Job);
        if let PlanStep::SubmitInferenceJob { parameters, .. } = &plan.steps[0] {
            assert!(parameters.get("assumptions").is_some());
        } else {
            panic!("Expected SubmitInferenceJob step");
        }
    }

    #[test]
    fn test_plan_infer_with_under_params() {
        let q = parse_query("INFER GAME FOR s:Scene UNDER RATIONALITY = 0.5 RETURN s").unwrap();
        let plan = plan_query(&q).unwrap();
        if let PlanStep::SubmitInferenceJob { parameters, .. } = &plan.steps[0] {
            assert_eq!(parameters.get("lambda").unwrap().as_f64().unwrap(), 0.5);
        } else {
            panic!("Expected SubmitInferenceJob step");
        }
    }

    #[test]
    fn test_plan_backward_compatibility() {
        let q = parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Instant);
        assert!(!plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::SubmitInferenceJob { .. })));
    }

    // ─── DISCOVER Plan Tests ─────────────────────────────────

    #[test]
    fn test_plan_discover_patterns() {
        let q = parse_query("DISCOVER PATTERNS RETURN *").unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Discovery);
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::SubmitDiscoverJob { .. })));
    }

    #[test]
    fn test_plan_discover_across_narratives() {
        let q = parse_query(r#"DISCOVER ARCS ACROSS NARRATIVES ("hamlet", "macbeth") RETURN *"#)
            .unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Discovery);
        if let PlanStep::SubmitDiscoverJob { narrative_ids, .. } = &plan.steps[0] {
            let ids = narrative_ids.as_ref().unwrap();
            assert_eq!(ids.len(), 2);
        } else {
            panic!("Expected SubmitDiscoverJob");
        }
    }

    #[test]
    fn test_plan_match_across_narratives() {
        let q = parse_query(r#"MATCH (e:Actor) ACROSS NARRATIVES ("hamlet") RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Instant);
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::FilterNarrative { .. })));
    }

    #[test]
    fn test_plan_match_across_all_narratives() {
        let q = parse_query("MATCH (e:Actor) ACROSS NARRATIVES RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        if let Some(PlanStep::FilterNarrative { narrative_ids }) = plan
            .steps
            .iter()
            .find(|s| matches!(s, PlanStep::FilterNarrative { .. }))
        {
            assert!(narrative_ids.is_none());
        } else {
            panic!("Expected FilterNarrative step");
        }
    }

    // ─── SPATIAL Planner Tests (Sprint P3.7) ────────────────────

    #[test]
    fn test_plan_spatial_step() {
        let q = parse_query(
            "MATCH (s:Situation) SPATIAL s.spatial WITHIN 10.0 KM OF (40.7, -74.0) RETURN s",
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::FilterSpatial { .. })));
    }

    #[test]
    fn test_plan_no_spatial_without_clause() {
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        assert!(!plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::FilterSpatial { .. })));
    }

    #[test]
    fn test_plan_export() {
        let stmt = parse_statement(r#"EXPORT NARRATIVE "hamlet" AS json"#).unwrap();
        let plan = plan_statement(&stmt).unwrap();
        assert_eq!(plan.class, QueryClass::Export);
        assert!(
            matches!(&plan.steps[0], PlanStep::Export { narrative_id, format } if narrative_id == "hamlet" && format == "json")
        );
    }

    // ─── ASK Planner Tests (Sprint RAG-2) ───────────────────

    #[test]
    fn test_plan_ask_simple() {
        let q = parse_query(r#"ASK "What happened?""#).unwrap();
        let plan = plan_query(&q).unwrap();
        assert_eq!(plan.class, QueryClass::Ask);
        assert!(plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::AskLlm { .. })));
    }

    #[test]
    fn test_plan_ask_with_mode() {
        let q = parse_query(r#"ASK "Who is the villain?" OVER "nar" MODE local"#).unwrap();
        let plan = plan_query(&q).unwrap();
        if let PlanStep::AskLlm {
            question,
            narrative_id,
            mode,
            ..
        } = &plan.steps[0]
        {
            assert_eq!(question, "Who is the villain?");
            assert_eq!(narrative_id.as_deref(), Some("nar"));
            assert_eq!(mode, "local");
        } else {
            panic!("Expected AskLlm step");
        }
    }

    #[test]
    fn test_plan_ask_default_mode() {
        let q = parse_query(r#"ASK "Question?""#).unwrap();
        let plan = plan_query(&q).unwrap();
        if let PlanStep::AskLlm { mode, .. } = &plan.steps[0] {
            assert_eq!(mode, "hybrid");
        } else {
            panic!("Expected AskLlm step");
        }
    }

    // ─── EATH Phase 5 — Surrogate planner tests ─────────────────

    #[test]
    fn test_planner_classifies_surrogate_jobs_as_async() {
        // Both verbs must classify as Job (async) because calibration is
        // O(dataset) and generation is variable-length forward simulation.
        let cal = parse_statement(r#"CALIBRATE SURROGATE USING 'eath' FOR "n1""#).unwrap();
        let cal_plan = plan_statement(&cal).unwrap();
        assert_eq!(cal_plan.class, QueryClass::Job);

        let gen = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" USING SURROGATE 'eath'"#,
        )
        .unwrap();
        let gen_plan = plan_statement(&gen).unwrap();
        assert_eq!(gen_plan.class, QueryClass::Job);
    }

    #[test]
    fn test_planner_emits_correct_plan_step_shape() {
        // Calibration plan step carries narrative_id + model.
        let cal = parse_statement(
            r#"CALIBRATE SURROGATE USING 'eath' FOR "narrative-a""#,
        )
        .unwrap();
        let cal_plan = plan_statement(&cal).unwrap();
        match &cal_plan.steps[0] {
            PlanStep::SubmitSurrogateCalibrationJob {
                narrative_id,
                model,
            } => {
                assert_eq!(narrative_id, "narrative-a");
                assert_eq!(model, "eath");
            }
            other => panic!("Expected SubmitSurrogateCalibrationJob, got: {other:?}"),
        }

        // Generation plan step carries every optional clause faithfully.
        let gen = parse_statement(
            r#"GENERATE NARRATIVE "out-1" LIKE "src-1" USING SURROGATE 'eath' PARAMS { "rho_low": 0.3 } SEED 99 STEPS 250 LABEL_PREFIX 'exp-a'"#,
        )
        .unwrap();
        let gen_plan = plan_statement(&gen).unwrap();
        match &gen_plan.steps[0] {
            PlanStep::SubmitSurrogateGenerationJob {
                source_narrative_id,
                output_narrative_id,
                model,
                params,
                seed,
                num_steps,
                label_prefix,
            } => {
                assert_eq!(source_narrative_id, "src-1");
                assert_eq!(output_narrative_id, "out-1");
                assert_eq!(model, "eath");
                assert!(params.is_some());
                assert_eq!(*seed, Some(99));
                assert_eq!(*num_steps, Some(250));
                assert_eq!(label_prefix.as_deref(), Some("exp-a"));
            }
            other => panic!("Expected SubmitSurrogateGenerationJob, got: {other:?}"),
        }
    }

    #[test]
    fn test_planner_errors_early_on_unknown_model_name() {
        // Both verbs MUST error at plan time when the model isn't in the
        // SurrogateRegistry — we don't want to waste a job submission on
        // a model we already know doesn't exist.
        let cal = parse_statement(r#"CALIBRATE SURROGATE USING 'nope' FOR "n1""#).unwrap();
        let cal_err = plan_statement(&cal).unwrap_err();
        let cal_msg = format!("{cal_err}");
        assert!(
            cal_msg.contains("unknown surrogate model") && cal_msg.contains("nope"),
            "expected 'unknown surrogate model nope' style error, got: {cal_msg}"
        );

        let gen = parse_statement(
            r#"GENERATE NARRATIVE "out" LIKE "src" USING SURROGATE 'nope'"#,
        )
        .unwrap();
        let gen_err = plan_statement(&gen).unwrap_err();
        let gen_msg = format!("{gen_err}");
        assert!(
            gen_msg.contains("unknown surrogate model"),
            "expected unknown-model error, got: {gen_msg}"
        );
    }
}
