use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::analysis::analysis_key;
use crate::error::{Result, TensaError};
#[allow(unused_imports)]
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::{keys, Hypergraph};
use crate::ingestion::embed::EmbeddingProvider;
use crate::ingestion::vector::VectorIndex;
use crate::narrative::registry::NarrativeRegistry;
use crate::query::parser::*;
use crate::query::planner::*;
use crate::temporal::index::IntervalTree;
use crate::temporal::interval::relation_between;
use crate::types::*;

/// A single result row: binding name -> JSON value.
pub type ResultRow = HashMap<String, serde_json::Value>;

/// Pseudo-binding name used to pass aggregate results from Aggregate to Project step.
const AGG_BINDING: &str = "_agg";

// Fuzzy boolean helpers for condition-expr evaluation (Phase 1 wiring).
//
// [`evaluate_expr`] short-circuits on booleans, which is the Gödel t-norm
// (`min`) / t-conorm (`max`) on `{0.0, 1.0}`. Planner / executor callers
// wanting a different family route through [`fuzzy_and`] / [`fuzzy_or`],
// thin re-exports of [`combine_tnorm`] / [`combine_tconorm`] named for
// the AND/OR role in the query surface. Cites: [klement2000].

/// Combine two graded truth values under the chosen t-norm (Gödel = `min`
/// preserves the existing short-circuit AND semantics).
pub use crate::fuzzy::tnorm::combine_tnorm as fuzzy_and;

/// Combine two graded truth values under the chosen t-conorm (Gödel = `max`
/// preserves the existing short-circuit OR semantics).
pub use crate::fuzzy::tnorm::combine_tconorm as fuzzy_or;

/// Fold a slice of graded truth values under a t-norm (neutral `1.0`).
pub use crate::fuzzy::tnorm::reduce_tnorm as reduce_fuzzy_and;

/// Fold a slice of graded truth values under a t-conorm (neutral `0.0`).
pub use crate::fuzzy::tnorm::reduce_tconorm as reduce_fuzzy_or;

/// Execute a query plan against the hypergraph.
///
/// For instant queries, executes scan/filter/project steps directly.
/// For job queries (INFER), returns a job submission descriptor that
/// the API layer uses to submit an async inference job.
/// If `explain` is true, returns the plan as JSON without executing.
pub fn execute(
    plan: &QueryPlan,
    hypergraph: &Hypergraph,
    interval_tree: &IntervalTree,
) -> Result<Vec<ResultRow>> {
    execute_with_options(plan, hypergraph, interval_tree, false, None, None, None)
}

/// Execute a query plan with EXPLAIN support (convenience wrapper without vector/embedder context).
pub fn execute_explained(
    plan: &QueryPlan,
    hypergraph: &Hypergraph,
    interval_tree: &IntervalTree,
    explain: bool,
) -> Result<Vec<ResultRow>> {
    execute_full(plan, hypergraph, interval_tree, explain, None, None)
}

/// Execute a query plan with full context (vector index + embedder for NEAR queries).
pub fn execute_full(
    plan: &QueryPlan,
    hypergraph: &Hypergraph,
    interval_tree: &IntervalTree,
    explain: bool,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
) -> Result<Vec<ResultRow>> {
    execute_with_options(
        plan,
        hypergraph,
        interval_tree,
        explain,
        vector_index,
        embedder,
        None,
    )
}

/// Execute a query plan with full context including RAG extractor.
pub fn execute_full_with_extractor(
    plan: &QueryPlan,
    hypergraph: &Hypergraph,
    interval_tree: &IntervalTree,
    explain: bool,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    extractor: Option<&dyn crate::ingestion::llm::NarrativeExtractor>,
) -> Result<Vec<ResultRow>> {
    execute_with_options(
        plan,
        hypergraph,
        interval_tree,
        explain,
        vector_index,
        embedder,
        extractor,
    )
}

/// Execute a query plan with a `JobQueue` handle so async-job plan steps
/// (currently the EATH Phase 5 surrogate steps) can submit to the queue
/// and return `{job_id, status, model}` in one call.
///
/// All other plan steps fall through to the standard execution path —
/// callers that don't need direct job submission should keep using
/// [`execute_full`] / [`execute_full_with_extractor`]. The
/// `inference::dispatch::infer_type_from_row` helpers also handle
/// surrogate descriptors so the embedded MCP backend and `POST /infer`
/// route automatically pick them up; this entry point is the
/// short-circuit for in-process callers (TensaQL → Studio chat,
/// dedicated `/synth/*` REST routes the next phase ships, tests).
pub fn execute_with_job_queue(
    plan: &QueryPlan,
    hypergraph: &Hypergraph,
    interval_tree: &IntervalTree,
    job_queue: &crate::inference::jobs::JobQueue,
) -> Result<Vec<ResultRow>> {
    // Surrogate plan steps live at index 0 of their plan (planner emits a
    // single PlanStep for each verb). Detect & short-circuit before
    // invoking the standard executor — saves the descriptor-row
    // round-trip and lets us return the friendlier `{job_id, status,
    // model}` shape.
    if let Some(step) = plan.steps.first() {
        match step {
            PlanStep::SubmitSurrogateCalibrationJob {
                narrative_id,
                model,
            } => {
                let job_id = submit_surrogate_calibration_job(
                    job_queue,
                    hypergraph,
                    narrative_id.clone(),
                    model.clone(),
                )?;
                let mut row = ResultRow::new();
                row.insert("job_id".into(), serde_json::json!(job_id));
                row.insert("status".into(), serde_json::json!("submitted"));
                row.insert("model".into(), serde_json::json!(model));
                return Ok(vec![row]);
            }
            PlanStep::SubmitSurrogateGenerationJob {
                source_narrative_id,
                output_narrative_id,
                model,
                params,
                seed,
                num_steps,
                label_prefix,
            } => {
                let job_id = submit_surrogate_generation_job(
                    job_queue,
                    hypergraph,
                    source_narrative_id.clone(),
                    output_narrative_id.clone(),
                    model.clone(),
                    params.clone(),
                    *seed,
                    *num_steps,
                    label_prefix.clone(),
                )?;
                let mut row = ResultRow::new();
                row.insert("job_id".into(), serde_json::json!(job_id));
                row.insert("status".into(), serde_json::json!("submitted"));
                row.insert("model".into(), serde_json::json!(model));
                return Ok(vec![row]);
            }
            // EATH Phase 9 — hybrid generation submission.
            PlanStep::SubmitSurrogateHybridGenerationJob {
                output_narrative_id,
                components,
                seed,
                num_steps,
            } => {
                let job_id = submit_surrogate_hybrid_generation_job(
                    job_queue,
                    hypergraph,
                    output_narrative_id.clone(),
                    components.clone(),
                    *seed,
                    *num_steps,
                )?;
                let mut row = ResultRow::new();
                row.insert("job_id".into(), serde_json::json!(job_id));
                row.insert("status".into(), serde_json::json!("submitted"));
                row.insert("model".into(), serde_json::json!("hybrid"));
                return Ok(vec![row]);
            }
            // EATH Extension Phase 13c — dual-null-model significance.
            PlanStep::SubmitSurrogateDualSignificanceJob {
                narrative_id,
                metric,
                k_per_model,
                models,
            } => {
                let job_id = submit_surrogate_dual_significance_job(
                    job_queue,
                    hypergraph,
                    narrative_id.clone(),
                    metric.clone(),
                    *k_per_model,
                    models.clone(),
                )?;
                let mut row = ResultRow::new();
                row.insert("job_id".into(), serde_json::json!(job_id));
                row.insert("status".into(), serde_json::json!("submitted"));
                row.insert("metric".into(), serde_json::json!(metric));
                row.insert("models".into(), serde_json::json!(models));
                return Ok(vec![row]);
            }
            // EATH Extension Phase 15c — SINDy hypergraph reconstruction.
            PlanStep::SubmitHypergraphReconstructionJob {
                narrative_id,
                params_json,
                ..
            } => {
                let job_id = submit_hypergraph_reconstruction_job(
                    job_queue,
                    hypergraph,
                    narrative_id.clone(),
                    params_json.clone(),
                )?;
                let mut row = ResultRow::new();
                row.insert("job_id".into(), serde_json::json!(job_id));
                row.insert("status".into(), serde_json::json!("submitted"));
                row.insert("narrative_id".into(), serde_json::json!(narrative_id));
                return Ok(vec![row]);
            }
            // EATH Extension Phase 16c — synchronous opinion-dynamics run.
            // Bypasses the job queue entirely: the engine runs in <1s for
            // MVP scales.
            PlanStep::RunOpinionDynamics {
                narrative_id,
                params_json,
                ..
            } => {
                return execute_opinion_dynamics_descriptor(
                    hypergraph,
                    narrative_id.clone(),
                    params_json.clone(),
                );
            }
            PlanStep::RunOpinionPhaseTransition {
                narrative_id,
                c_start,
                c_end,
                c_steps,
                ..
            } => {
                return execute_opinion_phase_transition_descriptor(
                    hypergraph,
                    narrative_id.clone(),
                    *c_start,
                    *c_end,
                    *c_steps,
                );
            }
            // Fuzzy Sprint Phase 6 — intermediate quantifier evaluation.
            // Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].
            PlanStep::RunQuantifier {
                quantifier,
                binding,
                type_name,
                where_clause,
                narrative_id,
                label,
            } => {
                return execute_quantifier_step(
                    hypergraph,
                    quantifier,
                    binding.as_deref(),
                    type_name,
                    where_clause.as_ref(),
                    narrative_id.as_deref(),
                    label.as_deref(),
                );
            }
            // Fuzzy Sprint Phase 7 — graded syllogism verification.
            // Cites: [murinovanovak2014peterson].
            PlanStep::RunVerifySyllogism {
                major,
                minor,
                conclusion,
                narrative_id,
                threshold,
                fuzzy_config,
            } => {
                return execute_verify_syllogism_step(
                    hypergraph,
                    major,
                    minor,
                    conclusion,
                    narrative_id,
                    *threshold,
                    fuzzy_config.tnorm,
                );
            }
            // Fuzzy Sprint Phase 8 — FCA lattice build + persist.
            // Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].
            PlanStep::RunFcaLattice {
                narrative_id,
                threshold,
                attribute_allowlist,
                entity_type,
                fuzzy_config,
            } => {
                return execute_fca_lattice_step(
                    hypergraph,
                    narrative_id,
                    *threshold,
                    attribute_allowlist.as_deref(),
                    entity_type.as_deref(),
                    fuzzy_config.tnorm,
                );
            }
            PlanStep::RunFcaConcept {
                lattice_id,
                concept_idx,
            } => {
                return execute_fca_concept_step(hypergraph, lattice_id, *concept_idx);
            }
            PlanStep::RunEvaluateRules {
                narrative_id,
                entity_type,
                rule_ids,
                ..
            } => {
                return execute_evaluate_rules_step(
                    hypergraph,
                    narrative_id,
                    entity_type,
                    rule_ids.as_deref(),
                );
            }
            PlanStep::RunFuzzyProbability {
                narrative_id,
                event_kind,
                event_ref,
                distribution,
                fuzzy_config,
            } => {
                return execute_fuzzy_probability_step(
                    hypergraph,
                    narrative_id,
                    event_kind,
                    event_ref,
                    distribution,
                    fuzzy_config.tnorm,
                );
            }
            _ => {}
        }
    }

    execute_with_options(plan, hypergraph, interval_tree, false, None, None, None)
}

/// Build the `InferenceJob` envelope for a calibration request and submit it
/// to the queue. Returns the (possibly-deduplicated) job id.
fn submit_surrogate_calibration_job(
    job_queue: &crate::inference::jobs::JobQueue,
    hypergraph: &Hypergraph,
    narrative_id: String,
    model: String,
) -> Result<String> {
    let job_type = crate::types::InferenceJobType::SurrogateCalibration {
        narrative_id,
        model,
    };
    let mut job = crate::inference::types::InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        target_id: Uuid::now_v7(),
        parameters: serde_json::json!({}),
        priority: crate::types::JobPriority::Normal,
        status: crate::types::JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    job.estimated_cost_ms = crate::inference::cost::estimate_cost(&job, hypergraph).unwrap_or(0);
    job_queue.submit(job)
}

/// Build the `InferenceJob` envelope for a generation request and submit it
/// to the queue. `num_steps` and `label_prefix` flow through the
/// `parameters` JSON blob (the engine reads them from there).
#[allow(clippy::too_many_arguments)]
fn submit_surrogate_generation_job(
    job_queue: &crate::inference::jobs::JobQueue,
    hypergraph: &Hypergraph,
    source_narrative_id: String,
    output_narrative_id: String,
    model: String,
    params_override: Option<serde_json::Value>,
    seed_override: Option<u64>,
    num_steps: Option<usize>,
    label_prefix: Option<String>,
) -> Result<String> {
    let job_type = crate::types::InferenceJobType::SurrogateGeneration {
        source_narrative_id: Some(source_narrative_id),
        output_narrative_id,
        model,
        params_override,
        seed_override,
    };

    // num_steps + label_prefix flow through the InferenceJob.parameters
    // blob — the engine reads them from there with documented defaults.
    let mut params_map = serde_json::Map::new();
    if let Some(n) = num_steps {
        params_map.insert("num_steps".into(), serde_json::json!(n));
    }
    if let Some(lp) = label_prefix {
        params_map.insert("label_prefix".into(), serde_json::json!(lp));
    }
    let parameters = if params_map.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::Value::Object(params_map)
    };

    let mut job = crate::inference::types::InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        target_id: Uuid::now_v7(),
        parameters,
        priority: crate::types::JobPriority::Normal,
        status: crate::types::JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    job.estimated_cost_ms = crate::inference::cost::estimate_cost(&job, hypergraph).unwrap_or(0);
    job_queue.submit(job)
}

/// EATH Extension Phase 13c — submit a `SurrogateDualSignificance` job.
/// Defaults `k_per_model` (when `None`) to the engine's `K_DEFAULT` so the
/// queue carries a concrete value the engine can clamp.
fn submit_surrogate_dual_significance_job(
    job_queue: &crate::inference::jobs::JobQueue,
    hypergraph: &Hypergraph,
    narrative_id: String,
    metric: String,
    k_per_model: Option<u16>,
    models: Vec<String>,
) -> Result<String> {
    let k_per_model = k_per_model
        .unwrap_or(crate::synth::significance::K_DEFAULT)
        .clamp(1, crate::synth::significance::K_MAX);
    let job_type = crate::types::InferenceJobType::SurrogateDualSignificance {
        narrative_id,
        metric,
        k_per_model,
        models,
    };
    let mut job = crate::inference::types::InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        target_id: Uuid::now_v7(),
        parameters: serde_json::json!({}),
        priority: crate::types::JobPriority::Normal,
        status: crate::types::JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    job.estimated_cost_ms = crate::inference::cost::estimate_cost(&job, hypergraph).unwrap_or(0);
    job_queue.submit(job)
}

/// Build the `InferenceJob` envelope for a hybrid generation request and
/// submit it to the queue. The components blob is JSON-validated at the
/// engine boundary; here we just thread it through.
fn submit_surrogate_hybrid_generation_job(
    job_queue: &crate::inference::jobs::JobQueue,
    hypergraph: &Hypergraph,
    output_narrative_id: String,
    components: serde_json::Value,
    seed_override: Option<u64>,
    num_steps: Option<usize>,
) -> Result<String> {
    let job_type = crate::types::InferenceJobType::SurrogateHybridGeneration {
        components,
        output_narrative_id,
        seed_override,
        num_steps,
    };
    let mut job = crate::inference::types::InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        target_id: Uuid::now_v7(),
        parameters: serde_json::json!({}),
        priority: crate::types::JobPriority::Normal,
        status: crate::types::JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    job.estimated_cost_ms = crate::inference::cost::estimate_cost(&job, hypergraph).unwrap_or(0);
    job_queue.submit(job)
}

/// EATH Extension Phase 16c — execute synchronous opinion-dynamics. Returns
/// the descriptor row containing the inline `OpinionDynamicsReport` JSON.
fn execute_opinion_dynamics_descriptor(
    hypergraph: &Hypergraph,
    narrative_id: String,
    params_json: serde_json::Value,
) -> Result<Vec<ResultRow>> {
    let params: crate::analysis::opinion_dynamics::OpinionDynamicsParams =
        if params_json.is_null() {
            crate::analysis::opinion_dynamics::OpinionDynamicsParams::default()
        } else {
            // Merge planner-emitted partial JSON over defaults.
            let defaults = serde_json::to_value(
                crate::analysis::opinion_dynamics::OpinionDynamicsParams::default(),
            )
            .map_err(|e| TensaError::Serialization(e.to_string()))?;
            let merged = merge_object_overrides(defaults, params_json);
            serde_json::from_value(merged).map_err(|e| {
                TensaError::QueryError(format!("invalid OpinionDynamicsParams: {e}"))
            })?
        };

    let report = crate::analysis::opinion_dynamics::simulate_opinion_dynamics(
        hypergraph,
        &narrative_id,
        &params,
    )?;
    let run_id = uuid::Uuid::now_v7();
    if let Err(e) = crate::analysis::opinion_dynamics::save_opinion_report(
        hypergraph.store(),
        &narrative_id,
        run_id,
        &report,
    ) {
        tracing::warn!(
            narrative_id = %narrative_id,
            "TensaQL OPINION_DYNAMICS: failed to persist report ({e})"
        );
    }
    let mut row = ResultRow::new();
    row.insert("narrative_id".into(), serde_json::json!(narrative_id));
    row.insert("run_id".into(), serde_json::json!(run_id));
    row.insert(
        "report".into(),
        serde_json::to_value(&report).map_err(|e| TensaError::Serialization(e.to_string()))?,
    );
    Ok(vec![row])
}

/// EATH Extension Phase 16c — execute synchronous opinion phase-transition
/// sweep. Returns a single descriptor row with the inline
/// `PhaseTransitionReport`.
fn execute_opinion_phase_transition_descriptor(
    hypergraph: &Hypergraph,
    narrative_id: String,
    c_start: f32,
    c_end: f32,
    c_steps: usize,
) -> Result<Vec<ResultRow>> {
    let base_params = crate::analysis::opinion_dynamics::OpinionDynamicsParams::default();
    let report = crate::analysis::opinion_dynamics::run_phase_transition_sweep(
        hypergraph,
        &narrative_id,
        (c_start, c_end, c_steps),
        &base_params,
    )?;
    let mut row = ResultRow::new();
    row.insert("narrative_id".into(), serde_json::json!(narrative_id));
    row.insert(
        "report".into(),
        serde_json::to_value(&report).map_err(|e| TensaError::Serialization(e.to_string()))?,
    );
    Ok(vec![row])
}

/// Fuzzy Sprint Phase 6 — execute a QUANTIFY plan step.
///
/// Resolves the entity- vs situation-domain split from `type_name`:
/// * `"Actor" | "Location" | "Artifact" | "Concept" | "Organization"` →
///   entity domain (restricted to that `EntityType`).
/// * `"Entity"` → all entity types in the narrative.
/// * `"Situation"` → situation domain.
///
/// Returns a single result row with one scalar column named `label`
/// (defaults to `"quantifier_result"`) carrying `Q(r) ∈ [0,1]`, plus
/// `_quantifier` + `_cardinality_ratio` diagnostic columns for tooling.
///
/// Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].
fn execute_quantifier_step(
    hypergraph: &Hypergraph,
    quantifier: &str,
    binding: Option<&str>,
    type_name: &str,
    where_clause: Option<&WhereClause>,
    narrative_id: Option<&str>,
    label: Option<&str>,
) -> Result<Vec<ResultRow>> {
    let q = crate::fuzzy::quantifier::quantifier_from_name(quantifier)?;
    let nid = narrative_id.ok_or_else(|| {
        TensaError::QueryError(
            "QUANTIFY requires FOR \"<narrative_id>\"; None domain makes the \
             ratio undefined"
                .into(),
        )
    })?;
    let bind_name = binding.unwrap_or("e").to_string();
    let label_col = label.unwrap_or("quantifier_result").to_string();

    // Resolve the domain: collect JSON-encoded records (Entity or Situation),
    // then fold the WHERE expression over each.
    let lower = type_name.to_ascii_lowercase();
    let domain_records: Vec<serde_json::Value> = match lower.as_str() {
        "situation" => hypergraph
            .list_situations_by_narrative(nid)?
            .iter()
            .map(|s| serde_json::to_value(s).unwrap_or(serde_json::Value::Null))
            .collect(),
        "entity" => hypergraph
            .list_entities_by_narrative(nid)?
            .iter()
            .map(|e| serde_json::to_value(e).unwrap_or(serde_json::Value::Null))
            .collect(),
        "actor" | "location" | "artifact" | "concept" | "organization" => {
            let et: EntityType = std::str::FromStr::from_str(type_name).map_err(|e| {
                TensaError::QueryError(format!("QUANTIFY unknown entity type: {e}"))
            })?;
            hypergraph
                .list_entities_by_narrative(nid)?
                .iter()
                .filter(|e| e.entity_type == et)
                .map(|e| serde_json::to_value(e).unwrap_or(serde_json::Value::Null))
                .collect()
        }
        other => {
            return Err(TensaError::QueryError(format!(
                "QUANTIFY pattern type '{other}' not supported; expected Actor/Location/\
                 Artifact/Concept/Organization/Entity/Situation"
            )));
        }
    };

    let n = domain_records.len();
    let mut sum = 0.0_f64;
    let mut cache = AnalysisCache::new();
    let mut co_graph_cache: Option<crate::analysis::graph_projection::CoGraph> = None;
    for rec in &domain_records {
        let passes = match where_clause {
            Some(w) => evaluate_expr(
                rec,
                &bind_name,
                &w.expr,
                hypergraph,
                &mut cache,
                &mut co_graph_cache,
            ),
            None => true,
        };
        // Phase 6 uses crisp WHERE: 1.0 if predicate holds, 0.0 otherwise.
        // Upgrading `evaluate_expr` to return graded [0,1] is a Phase 6.5
        // / downstream refactor — see docs/FUZZY_Sprint.md Phase 6 deferrals.
        if passes {
            sum += 1.0;
        }
    }
    let r = if n == 0 { 0.0 } else { sum / (n as f64) };
    let value = crate::fuzzy::quantifier::evaluate(q, r);

    let mut row = ResultRow::new();
    row.insert(label_col.clone(), serde_json::json!(value));
    row.insert("_quantifier".into(), serde_json::json!(q.name()));
    row.insert("_cardinality_ratio".into(), serde_json::json!(r));
    row.insert("_domain_size".into(), serde_json::json!(n));
    row.insert("_narrative_id".into(), serde_json::json!(nid));
    Ok(vec![row])
}

/// Fuzzy Sprint Phase 7 — execute a VERIFY SYLLOGISM statement.
/// Parses the three tiny-DSL strings, runs the verifier with the
/// configured t-norm + threshold, and returns a single descriptor row.
/// The result carries `{degree, figure, valid, threshold, fuzzy_config}`.
/// Cites: [murinovanovak2014peterson].
fn execute_verify_syllogism_step(
    hypergraph: &Hypergraph,
    major: &str,
    minor: &str,
    conclusion: &str,
    narrative_id: &str,
    threshold: Option<f64>,
    tnorm_override: Option<TNormKind>,
) -> Result<Vec<ResultRow>> {
    use crate::fuzzy::syllogism::{
        parse_statement, verify, Syllogism, TypePredicateResolver,
    };

    let major_stmt = parse_statement(major)?;
    let minor_stmt = parse_statement(minor)?;
    let conc_stmt = parse_statement(conclusion)?;

    let syl = Syllogism {
        major: major_stmt,
        minor: minor_stmt,
        conclusion: conc_stmt,
        figure_hint: None,
    };

    let t = tnorm_override.unwrap_or(TNormKind::Godel);
    let thr = threshold.unwrap_or(0.5);
    let gv = verify(
        hypergraph,
        narrative_id,
        &syl,
        t,
        thr,
        &TypePredicateResolver,
    )?;

    let mut row = ResultRow::new();
    row.insert("degree".into(), serde_json::json!(gv.degree));
    row.insert("figure".into(), serde_json::json!(gv.figure));
    row.insert("valid".into(), serde_json::json!(gv.valid));
    row.insert("threshold".into(), serde_json::json!(gv.threshold));
    row.insert(
        "fuzzy_config".into(),
        serde_json::json!({ "tnorm": t.name() }),
    );
    row.insert("_narrative_id".into(), serde_json::json!(narrative_id));
    Ok(vec![row])
}

/// Fuzzy Sprint Phase 8 — build + persist a concept lattice.
/// Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].
fn execute_fca_lattice_step(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    threshold: Option<usize>,
    attribute_allowlist: Option<&[String]>,
    entity_type: Option<&str>,
    tnorm_override: Option<TNormKind>,
) -> Result<Vec<ResultRow>> {
    use crate::fuzzy::fca::{
        build_lattice_with_threshold, FormalContext, FormalContextOptions,
    };
    use crate::fuzzy::fca_store::save_concept_lattice;

    let t = tnorm_override.unwrap_or(TNormKind::Godel);
    let et: Option<crate::types::EntityType> = match entity_type {
        None => None,
        Some(name) => Some(
            name.parse::<crate::types::EntityType>()
                .map_err(|e| TensaError::InvalidInput(format!(
                    "FCA LATTICE ENTITY_TYPE '{name}' unknown: {e}"
                )))?,
        ),
    };
    let opts = FormalContextOptions {
        entity_type_filter: et,
        attribute_allowlist: attribute_allowlist.map(|s| s.to_vec()),
        large_context: false,
    };
    let ctx = FormalContext::from_hypergraph(hypergraph, narrative_id, &opts)?;
    let mut lattice = build_lattice_with_threshold(&ctx, t, threshold.unwrap_or(0))?;
    lattice.narrative_id = narrative_id.to_string();
    save_concept_lattice(hypergraph.store(), narrative_id, &lattice)?;

    let mut row = ResultRow::new();
    row.insert("lattice_id".into(), serde_json::json!(lattice.id));
    row.insert(
        "num_concepts".into(),
        serde_json::json!(lattice.num_concepts()),
    );
    row.insert("num_objects".into(), serde_json::json!(lattice.objects.len()));
    row.insert(
        "num_attributes".into(),
        serde_json::json!(lattice.attributes.len()),
    );
    row.insert("tnorm".into(), serde_json::json!(t.name()));
    row.insert("_narrative_id".into(), serde_json::json!(narrative_id));
    Ok(vec![row])
}

/// Fuzzy Sprint Phase 8 — look up a single concept in a persisted
/// lattice by index. Scans every narrative's FCA prefix since the
/// query grammar only carries the lattice_id (no narrative_id).
fn execute_fca_concept_step(
    hypergraph: &Hypergraph,
    lattice_id: &str,
    concept_idx: usize,
) -> Result<Vec<ResultRow>> {
    use crate::fuzzy::fca::ConceptLattice;
    let target: Uuid = lattice_id
        .parse()
        .map_err(|e| TensaError::InvalidInput(format!("lattice_id not a UUID: {e}")))?;
    // Scan the fz/fca/ prefix for the matching lattice. Cheap: the
    // total number of persisted lattices per workspace is bounded by
    // explicit user requests.
    let pairs = hypergraph.store().prefix_scan(crate::fuzzy::FUZZY_FCA_PREFIX)?;
    let mut found: Option<ConceptLattice> = None;
    for (_, v) in pairs {
        match serde_json::from_slice::<ConceptLattice>(&v) {
            Ok(l) if l.id == target => {
                found = Some(l);
                break;
            }
            _ => continue,
        }
    }
    let lattice = found.ok_or_else(|| {
        TensaError::NotFound(format!("FCA lattice {lattice_id} not persisted"))
    })?;
    let concept = lattice.concept(concept_idx)?;
    let mut row = ResultRow::new();
    row.insert("lattice_id".into(), serde_json::json!(lattice.id));
    row.insert("concept_idx".into(), serde_json::json!(concept_idx));
    row.insert("extent".into(), serde_json::json!(concept.extent));
    row.insert("intent".into(), serde_json::json!(concept.intent));
    // Include readable labels so downstream callers don't need a
    // second lookup for display.
    let extent_labels: Vec<&str> = concept
        .extent
        .iter()
        .filter_map(|&i| lattice.objects.get(i).map(|o| o.label.as_str()))
        .collect();
    row.insert("extent_labels".into(), serde_json::json!(extent_labels));
    let intent_labels: Vec<(String, f64)> = concept
        .intent
        .iter()
        .filter_map(|&(j, mu)| lattice.attributes.get(j).map(|a| (a.clone(), mu)))
        .collect();
    row.insert("intent_labels".into(), serde_json::json!(intent_labels));
    Ok(vec![row])
}

/// Fuzzy Sprint Phase 9 — evaluate a Mamdani rule set against every
/// entity in `narrative_id` that matches `entity_type`. Emits one row
/// per entity (`entity_id`, `fired_rules`, `defuzzified_output`,
/// `defuzzification`).
/// Cites: [mamdani1975mamdani].
fn execute_evaluate_rules_step(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    entity_type: &str,
    rule_ids: Option<&[String]>,
) -> Result<Vec<ResultRow>> {
    let et: crate::types::EntityType = entity_type
        .parse::<crate::types::EntityType>()
        .map_err(|e| {
            TensaError::InvalidInput(format!(
                "EVALUATE RULES entity_type '{entity_type}' unknown: {e}"
            ))
        })?;
    let evs = crate::fuzzy::rules::evaluate_rules_over_narrative(
        hypergraph,
        narrative_id,
        rule_ids,
        Some(et),
    )?;
    let mut out = Vec::with_capacity(evs.len());
    for ev in evs {
        let mut row = ResultRow::new();
        row.insert("entity_id".into(), serde_json::json!(ev.entity_id));
        row.insert(
            "fired_rules".into(),
            serde_json::to_value(&ev.fired_rules)
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
        );
        row.insert(
            "defuzzified_output".into(),
            serde_json::json!(ev.defuzzified_output),
        );
        row.insert(
            "defuzzification".into(),
            serde_json::json!(ev.defuzzification),
        );
        row.insert("_narrative_id".into(), serde_json::json!(narrative_id));
        out.push(row);
    }
    Ok(out)
}

/// Fuzzy Sprint Phase 10 — execute an INFER FUZZY_PROBABILITY step.
/// Parses the three payload strings, dispatches through
/// [`crate::fuzzy::hybrid::fuzzy_probability`], and returns a single
/// descriptor row `{ value, event_kind, distribution_summary,
/// _narrative_id }`. Results are persisted to `fz/hybrid/{nid}/...`
/// so REST / MCP callers can retrieve the run later.
/// Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
fn execute_fuzzy_probability_step(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    event_kind: &str,
    event_ref: &str,
    distribution: &str,
    tnorm_override: Option<TNormKind>,
) -> Result<Vec<ResultRow>> {
    use crate::fuzzy::hybrid::{
        build_hybrid_report, fuzzy_probability, save_hybrid_result, FuzzyEvent,
        FuzzyEventPredicate, ProbDist,
    };

    // Resolve the event kind from the planner-validated string.
    let predicate_kind = match event_kind.to_ascii_lowercase().as_str() {
        "quantifier" => FuzzyEventPredicate::Quantifier,
        "mamdani_rule" => FuzzyEventPredicate::MamdaniRule,
        "custom" => FuzzyEventPredicate::Custom,
        other => {
            return Err(TensaError::InvalidInput(format!(
                "INFER FUZZY_PROBABILITY unknown event_kind '{other}'"
            )));
        }
    };

    let predicate_payload: serde_json::Value = serde_json::from_str(event_ref)
        .map_err(|e| TensaError::InvalidInput(format!("event_ref is not valid JSON: {e}")))?;
    let event = FuzzyEvent {
        predicate_kind,
        predicate_payload,
    };

    // Accept two distribution shapes:
    //   1. A literal JSON object `{ "outcomes": [["<uuid>", 0.5], ...] }`.
    //   2. The keyword `uniform` (require `outcomes` list on event spec).
    let dist: ProbDist = if distribution.eq_ignore_ascii_case("uniform") {
        // Uniform over every Actor entity in the narrative.
        let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
        if entities.is_empty() {
            return Err(TensaError::InvalidInput(
                "uniform distribution requested but narrative has no entities".into(),
            ));
        }
        let p = 1.0_f64 / entities.len() as f64;
        ProbDist::Discrete {
            outcomes: entities.into_iter().map(|e| (e.id, p)).collect(),
        }
    } else {
        serde_json::from_str::<ProbDist>(distribution).map_err(|e| {
            TensaError::InvalidInput(format!("distribution is not valid JSON: {e}"))
        })?
    };

    let tnorm = tnorm_override.unwrap_or(TNormKind::Godel);
    let value = fuzzy_probability(hypergraph, narrative_id, &event, &dist, tnorm)?;

    // Persist the descriptor so REST / MCP callers can read it.
    let report = build_hybrid_report(narrative_id, &event, &dist, tnorm, value);
    let query_id = report.query_id;
    if let Err(e) = save_hybrid_result(hypergraph.store(), narrative_id, &report) {
        tracing::warn!(
            narrative_id = %narrative_id,
            "failed to persist hybrid-probability report ({e}); returning inline anyway"
        );
    }

    let mut row = ResultRow::new();
    row.insert("value".into(), serde_json::json!(value));
    row.insert("event_kind".into(), serde_json::json!(report.event_kind));
    row.insert(
        "distribution_summary".into(),
        serde_json::json!(report.distribution_summary),
    );
    row.insert("query_id".into(), serde_json::json!(query_id));
    row.insert("_narrative_id".into(), serde_json::json!(narrative_id));
    Ok(vec![row])
}

/// Shallow merge: take every top-level key in `overrides` and replace the
/// corresponding key in `base`. Used to overlay planner-emitted partial
/// params blobs on engine defaults so callers can omit fields.
fn merge_object_overrides(
    mut base: serde_json::Value,
    overrides: serde_json::Value,
) -> serde_json::Value {
    if let (serde_json::Value::Object(bmap), serde_json::Value::Object(omap)) =
        (&mut base, overrides)
    {
        for (k, v) in omap {
            bmap.insert(k, v);
        }
    }
    base
}

/// EATH Extension Phase 15c — submit a `HypergraphReconstruction` job. The
/// `params_json` blob is the partial JSON the planner assembled from the
/// optional TensaQL clauses (USING OBSERVATION, MAX_ORDER, LAMBDA). The
/// engine applies serde defaults for omitted fields and then rebuilds a
/// concrete `ReconstructionParams` struct.
fn submit_hypergraph_reconstruction_job(
    job_queue: &crate::inference::jobs::JobQueue,
    hypergraph: &Hypergraph,
    narrative_id: String,
    params_json: serde_json::Value,
) -> Result<String> {
    let job_type = crate::types::InferenceJobType::HypergraphReconstruction {
        narrative_id: narrative_id.clone(),
        // Mirror the descriptor-row dispatch convention: keep the variant
        // payload sentinel and put the actual params blob on
        // `job.parameters` so the engine's two-slot resolver picks it up.
        params: serde_json::Value::Null,
    };
    let mut params_map = serde_json::Map::new();
    params_map.insert("narrative_id".into(), serde_json::json!(narrative_id));
    if let serde_json::Value::Object(obj) = params_json {
        for (k, v) in obj {
            params_map.insert(k, v);
        }
    }
    let mut job = crate::inference::types::InferenceJob {
        id: Uuid::now_v7().to_string(),
        job_type,
        target_id: Uuid::now_v7(),
        parameters: serde_json::Value::Object(params_map),
        priority: crate::types::JobPriority::Normal,
        status: crate::types::JobStatus::Pending,
        estimated_cost_ms: 0,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
        error: None,
    };
    job.estimated_cost_ms = crate::inference::cost::estimate_cost(&job, hypergraph).unwrap_or(0);
    job_queue.submit(job)
}

fn execute_with_options(
    plan: &QueryPlan,
    hypergraph: &Hypergraph,
    _interval_tree: &IntervalTree,
    explain: bool,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    extractor: Option<&dyn crate::ingestion::llm::NarrativeExtractor>,
) -> Result<Vec<ResultRow>> {
    if explain {
        let plan_json =
            serde_json::to_value(plan).map_err(|e| TensaError::Serialization(e.to_string()))?;
        let mut row = ResultRow::new();
        row.insert("plan".into(), plan_json);
        return Ok(vec![row]);
    }

    // Working set: binding -> list of JSON values
    let mut bindings: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

    // Edge pairs from FollowEdge: (from_binding, to_binding) -> Vec<(from_id, to_id)>
    // Used by the projection step to do proper joins instead of cross-products.
    let mut edge_pairs: HashMap<(String, String), Vec<(uuid::Uuid, uuid::Uuid)>> = HashMap::new();

    for step in &plan.steps {
        match step {
            PlanStep::ScanByType { type_name, binding } => {
                let values = scan_by_type(hypergraph, type_name)?;
                bindings.insert(binding.clone(), values);
            }
            PlanStep::FollowEdge {
                from_binding,
                rel_type,
                to_binding,
                ..
            } => {
                let pairs = follow_edge(
                    hypergraph,
                    &mut bindings,
                    from_binding,
                    rel_type,
                    to_binding,
                )?;
                edge_pairs.insert((from_binding.clone(), to_binding.clone()), pairs);
            }
            // Phase 3 — fuzzy_config slot is present on these plan steps but
            // the default boolean short-circuit fold in `filter_properties`
            // is bit-identical to Gödel T(a,b)=min on {0,1}. Downstream
            // graded-WHERE work will switch to `fuzzy_and` / `fuzzy_or` when
            // a non-Gödel selector arrives; for now the slot rides along
            // only to surface in EXPLAIN.
            PlanStep::FilterProperties { expr, .. } => {
                filter_properties(&mut bindings, expr, hypergraph)?;
                prune_edge_pairs_to_bindings(&mut bindings, &mut edge_pairs);
            }
            PlanStep::FilterTemporal { at_clause, .. } => {
                filter_temporal(&mut bindings, at_clause)?;
            }
            PlanStep::VectorNear {
                binding,
                query_text,
                k,
                ..
            } => {
                filter_vector_near(
                    &mut bindings,
                    binding,
                    query_text,
                    *k,
                    vector_index,
                    embedder,
                )?;
            }
            PlanStep::FilterSpatial {
                field,
                radius_km,
                center_lat,
                center_lon,
            } => {
                filter_spatial(&mut bindings, field, *radius_km, *center_lat, *center_lon)?;
            }
            PlanStep::SubmitInferenceJob {
                infer_type,
                target_binding,
                target_type,
                parameters,
                ..
            } => {
                // For JOB queries, return a descriptor row that the API layer
                // uses to submit the job to the inference queue. The fuzzy
                // config (if any) is already packed into `parameters` as
                // `parameters.fuzzy_config` so the dispatcher/engine can
                // consume it through the standard JSON surface.
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Job"));
                row.insert(
                    "_infer_type".into(),
                    serde_json::json!(format!("{:?}", infer_type)),
                );
                row.insert("_target_binding".into(), serde_json::json!(target_binding));
                row.insert("_target_type".into(), serde_json::json!(target_type));
                row.insert("_parameters".into(), parameters.clone());
                return Ok(vec![row]);
            }
            PlanStep::Aggregate {
                group_by,
                aggregates,
            } => {
                let agg_results = execute_aggregate(&bindings, group_by, aggregates)?;
                bindings.clear();
                let agg_values: Vec<serde_json::Value> = agg_results
                    .into_iter()
                    .map(|row| {
                        serde_json::to_value(row)
                            .map_err(|e| TensaError::Serialization(e.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                bindings.insert(AGG_BINDING.to_string(), agg_values);
            }
            PlanStep::FilterNarrative { narrative_ids } => {
                // Filter all bindings by narrative_id field
                for values in bindings.values_mut() {
                    values.retain(|val| {
                        let nid = val.get("narrative_id").and_then(|v| v.as_str());
                        match narrative_ids {
                            None => true, // ACROSS NARRATIVES without list = all narratives
                            Some(ids) => match nid {
                                Some(id) => ids.iter().any(|n| n == id),
                                None => false, // Items without narrative_id excluded when filtering
                            },
                        }
                    });
                }
            }
            PlanStep::SubmitDiscoverJob {
                discover_type,
                target_binding,
                target_type,
                narrative_ids,
                parameters,
            } => {
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Discovery"));
                row.insert(
                    "_discover_type".into(),
                    serde_json::json!(format!("{:?}", discover_type)),
                );
                if let Some(tb) = target_binding {
                    row.insert("_target_binding".into(), serde_json::json!(tb));
                }
                if let Some(tt) = target_type {
                    row.insert("_target_type".into(), serde_json::json!(tt));
                }
                if let Some(ids) = narrative_ids {
                    row.insert("_narrative_ids".into(), serde_json::json!(ids));
                }
                row.insert("_parameters".into(), parameters.clone());
                return Ok(vec![row]);
            }
            // EATH Phase 5 — surrogate calibration / generation. Mirror the
            // SubmitInferenceJob pattern: emit a descriptor row carrying
            // every field a downstream submitter (`inference::dispatch` →
            // `JobQueue.submit`) needs to build the right `InferenceJob`.
            // The dedicated `execute_with_job_queue` entry point in this
            // module short-circuits that round-trip and submits directly,
            // returning `{ job_id, status, model }`.
            PlanStep::SubmitSurrogateCalibrationJob {
                narrative_id,
                model,
            } => {
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Job"));
                row.insert(
                    "_synth_kind".into(),
                    serde_json::json!("SurrogateCalibration"),
                );
                row.insert("_synth_model".into(), serde_json::json!(model));
                row.insert(
                    "_synth_narrative_id".into(),
                    serde_json::json!(narrative_id),
                );
                return Ok(vec![row]);
            }
            PlanStep::SubmitSurrogateGenerationJob {
                source_narrative_id,
                output_narrative_id,
                model,
                params,
                seed,
                num_steps,
                label_prefix,
            } => {
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Job"));
                row.insert(
                    "_synth_kind".into(),
                    serde_json::json!("SurrogateGeneration"),
                );
                row.insert("_synth_model".into(), serde_json::json!(model));
                row.insert(
                    "_synth_source_narrative_id".into(),
                    serde_json::json!(source_narrative_id),
                );
                row.insert(
                    "_synth_output_narrative_id".into(),
                    serde_json::json!(output_narrative_id),
                );
                if let Some(p) = params {
                    row.insert("_synth_params_override".into(), p.clone());
                }
                if let Some(s) = seed {
                    row.insert("_synth_seed".into(), serde_json::json!(s));
                }
                if let Some(n) = num_steps {
                    row.insert("_synth_num_steps".into(), serde_json::json!(n));
                }
                if let Some(lp) = label_prefix {
                    row.insert("_synth_label_prefix".into(), serde_json::json!(lp));
                }
                return Ok(vec![row]);
            }
            PlanStep::SubmitSurrogateHybridGenerationJob {
                output_narrative_id,
                components,
                seed,
                num_steps,
            } => {
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Job"));
                row.insert(
                    "_synth_kind".into(),
                    serde_json::json!("SurrogateHybridGeneration"),
                );
                row.insert("_synth_model".into(), serde_json::json!("hybrid"));
                row.insert(
                    "_synth_output_narrative_id".into(),
                    serde_json::json!(output_narrative_id),
                );
                row.insert("_synth_hybrid_components".into(), components.clone());
                if let Some(s) = seed {
                    row.insert("_synth_seed".into(), serde_json::json!(s));
                }
                if let Some(n) = num_steps {
                    row.insert("_synth_num_steps".into(), serde_json::json!(n));
                }
                return Ok(vec![row]);
            }
            // EATH Extension Phase 13c — dual-null-model significance descriptor.
            // Mirrors the SurrogateHybridGeneration descriptor pattern: emit a
            // row carrying every field a downstream submitter needs to build
            // the right `InferenceJob`. The dedicated `execute_with_job_queue`
            // entry point above short-circuits this round-trip and submits
            // directly via `submit_surrogate_dual_significance_job`.
            PlanStep::SubmitSurrogateDualSignificanceJob {
                narrative_id,
                metric,
                k_per_model,
                models,
            } => {
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Job"));
                row.insert(
                    "_synth_kind".into(),
                    serde_json::json!("SurrogateDualSignificance"),
                );
                row.insert(
                    "_synth_narrative_id".into(),
                    serde_json::json!(narrative_id),
                );
                row.insert("_synth_metric".into(), serde_json::json!(metric));
                row.insert("_synth_models".into(), serde_json::json!(models));
                if let Some(k) = k_per_model {
                    row.insert("_synth_k_per_model".into(), serde_json::json!(k));
                }
                return Ok(vec![row]);
            }
            // EATH Extension Phase 15c — hypergraph reconstruction descriptor.
            // Routes through the same `_infer_type = "HypergraphReconstruction"`
            // path the dispatch.rs resolver already understands (Phase 15b).
            // The dedicated `execute_with_job_queue` entry point above
            // short-circuits this round-trip when a JobQueue is available.
            PlanStep::SubmitHypergraphReconstructionJob {
                narrative_id,
                params_json,
                fuzzy_config,
            } => {
                let mut row = ResultRow::new();
                row.insert("_job_class".into(), serde_json::json!("Job"));
                row.insert(
                    "_infer_type".into(),
                    serde_json::json!("HypergraphReconstruction"),
                );
                let mut params = serde_json::Map::new();
                params.insert("narrative_id".into(), serde_json::json!(narrative_id));
                if let serde_json::Value::Object(obj) = params_json {
                    for (k, v) in obj {
                        params.insert(k.clone(), v.clone());
                    }
                }
                if !fuzzy_config.is_empty() {
                    let fz = serde_json::to_value(fuzzy_config)
                        .map_err(|e| TensaError::Serialization(e.to_string()))?;
                    params.insert("fuzzy_config".into(), fz);
                }
                row.insert("_parameters".into(), serde_json::Value::Object(params));
                return Ok(vec![row]);
            }
            // EATH Extension Phase 16c — synchronous opinion-dynamics
            // descriptor. Always runs the engine inline because Phase 16b
            // benchmarks complete in milliseconds. No job queue needed.
            PlanStep::RunOpinionDynamics {
                narrative_id,
                params_json,
                ..
            } => {
                return execute_opinion_dynamics_descriptor(
                    hypergraph,
                    narrative_id.clone(),
                    params_json.clone(),
                );
            }
            PlanStep::RunOpinionPhaseTransition {
                narrative_id,
                c_start,
                c_end,
                c_steps,
                ..
            } => {
                return execute_opinion_phase_transition_descriptor(
                    hypergraph,
                    narrative_id.clone(),
                    *c_start,
                    *c_end,
                    *c_steps,
                );
            }
            PlanStep::RunQuantifier {
                quantifier,
                binding,
                type_name,
                where_clause,
                narrative_id,
                label,
            } => {
                return execute_quantifier_step(
                    hypergraph,
                    quantifier,
                    binding.as_deref(),
                    type_name,
                    where_clause.as_ref(),
                    narrative_id.as_deref(),
                    label.as_deref(),
                );
            }
            PlanStep::RunVerifySyllogism {
                major,
                minor,
                conclusion,
                narrative_id,
                threshold,
                fuzzy_config,
            } => {
                return execute_verify_syllogism_step(
                    hypergraph,
                    major,
                    minor,
                    conclusion,
                    narrative_id,
                    *threshold,
                    fuzzy_config.tnorm,
                );
            }
            PlanStep::RunFcaLattice {
                narrative_id,
                threshold,
                attribute_allowlist,
                entity_type,
                fuzzy_config,
            } => {
                return execute_fca_lattice_step(
                    hypergraph,
                    narrative_id,
                    *threshold,
                    attribute_allowlist.as_deref(),
                    entity_type.as_deref(),
                    fuzzy_config.tnorm,
                );
            }
            PlanStep::RunFcaConcept {
                lattice_id,
                concept_idx,
            } => {
                return execute_fca_concept_step(hypergraph, lattice_id, *concept_idx);
            }
            PlanStep::RunEvaluateRules {
                narrative_id,
                entity_type,
                rule_ids,
                ..
            } => {
                return execute_evaluate_rules_step(
                    hypergraph,
                    narrative_id,
                    entity_type,
                    rule_ids.as_deref(),
                );
            }
            PlanStep::RunFuzzyProbability {
                narrative_id,
                event_kind,
                event_ref,
                distribution,
                fuzzy_config,
            } => {
                return execute_fuzzy_probability_step(
                    hypergraph,
                    narrative_id,
                    event_kind,
                    event_ref,
                    distribution,
                    fuzzy_config.tnorm,
                );
            }
            PlanStep::AskLlm {
                question,
                narrative_id,
                mode,
                response_type,
                session_id,
                suggest,
                ..
            } => {
                let ext = extractor.ok_or_else(|| {
                    TensaError::QueryError("No LLM extractor configured for ASK queries".into())
                })?;
                let rag_mode = crate::query::rag_config::RetrievalMode::from_str_or_default(mode);
                let budget = crate::query::token_budget::TokenBudget::default();
                // Load session history if session_id specified
                let session_history = if let Some(sid) = session_id {
                    crate::query::session::load_session(hypergraph.store(), sid)?
                        .map(|s| crate::query::session::format_history(&s))
                } else {
                    None
                };
                let rag_answer = crate::query::rag::execute_ask(
                    question,
                    narrative_id.as_deref(),
                    &rag_mode,
                    &budget,
                    hypergraph,
                    vector_index,
                    embedder,
                    ext,
                    None, // no reranker in TensaQL path
                    response_type.as_deref(),
                    *suggest,
                    session_history.as_deref(),
                )?;
                // Save turn to session if session_id is set
                if let Some(sid) = session_id {
                    let _ = crate::query::session::append_turn(
                        hypergraph.store(),
                        sid,
                        question,
                        &rag_answer.answer,
                        5,
                    );
                }
                let mut row = ResultRow::new();
                row.insert(
                    "answer".into(),
                    serde_json::Value::String(rag_answer.answer),
                );
                row.insert(
                    "citations".into(),
                    serde_json::to_value(&rag_answer.citations)
                        .map_err(|e| TensaError::Serialization(e.to_string()))?,
                );
                row.insert("mode".into(), serde_json::Value::String(rag_answer.mode));
                row.insert(
                    "tokens_used".into(),
                    serde_json::json!(rag_answer.tokens_used),
                );
                if !rag_answer.suggestions.is_empty() {
                    row.insert(
                        "suggestions".into(),
                        serde_json::to_value(&rag_answer.suggestions)
                            .map_err(|e| TensaError::Serialization(e.to_string()))?,
                    );
                }
                return Ok(vec![row]);
            }
            PlanStep::TunePrompts { narrative_id } => {
                let ext = extractor.ok_or_else(|| {
                    TensaError::QueryError("No LLM extractor configured for TUNE PROMPTS".into())
                })?;
                let tuned = crate::ingestion::prompt_tuning::tune_prompts(
                    hypergraph.store(),
                    ext,
                    hypergraph,
                    narrative_id,
                )?;
                let mut row = ResultRow::new();
                row.insert(
                    "tuned_prompt".into(),
                    serde_json::to_value(&tuned)
                        .map_err(|e| TensaError::Serialization(e.to_string()))?,
                );
                return Ok(vec![row]);
            }
            PlanStep::Project { return_clause } => {
                // If path results exist, return them directly (with LIMIT)
                if let Some(path_values) = bindings.get("_paths") {
                    let mut rows: Vec<ResultRow> = path_values
                        .iter()
                        .filter_map(|v| {
                            v.as_object().map(|obj| {
                                obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                            })
                        })
                        .collect();
                    if let Some(limit) = return_clause.limit {
                        rows.truncate(limit);
                    }
                    return Ok(rows);
                }
                // If aggregate results exist, return them directly (with ORDER BY + LIMIT)
                if let Some(agg_values) = bindings.get(AGG_BINDING) {
                    let mut rows: Vec<ResultRow> = agg_values
                        .iter()
                        .filter_map(|v| {
                            v.as_object().map(|obj| {
                                obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                            })
                        })
                        .collect();
                    // Apply ORDER BY
                    if let Some(ob) = &return_clause.order_by {
                        let field = ob.field.clone();
                        let asc = ob.ascending;
                        rows.sort_by(|a, b| {
                            let va = a.get(&field);
                            let vb = b.get(&field);
                            let cmp = match (va, vb) {
                                (Some(a), Some(b)) => {
                                    if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                                    } else {
                                        std::cmp::Ordering::Equal
                                    }
                                }
                                _ => std::cmp::Ordering::Equal,
                            };
                            if asc {
                                cmp
                            } else {
                                cmp.reverse()
                            }
                        });
                    }
                    // Apply LIMIT
                    if let Some(limit) = return_clause.limit {
                        rows.truncate(limit);
                    }
                    return Ok(rows);
                }
                return project_results(&bindings, &edge_pairs, return_clause, hypergraph);
            }
            PlanStep::FindPath {
                start_binding,
                end_binding,
                rel_type,
                mode,
                restrictor,
                min_depth,
                max_depth,
                top_k,
                weight_field,
            } => {
                let paths = find_paths(
                    hypergraph,
                    &bindings,
                    start_binding,
                    end_binding,
                    rel_type,
                    mode,
                    restrictor,
                    *min_depth,
                    *max_depth,
                    *top_k,
                    weight_field.as_deref(),
                )?;
                bindings.insert("_paths".into(), paths);
            }
            PlanStep::FindFlow {
                start_binding,
                end_binding,
                rel_type,
                flow_type,
            } => {
                let results = execute_flow(
                    hypergraph,
                    &bindings,
                    start_binding,
                    end_binding,
                    rel_type,
                    flow_type,
                )?;
                bindings.insert("_paths".into(), results);
            }
            PlanStep::Mutate { mutation } => {
                return execute_mutation(mutation, hypergraph);
            }
            PlanStep::Export {
                narrative_id,
                format,
            } => {
                return execute_export(narrative_id, format, hypergraph);
            }
        }
    }

    Ok(vec![])
}

/// Find paths between situations using BFS (shortest) or DFS (all).
///
/// Path restrictors control traversal semantics:
/// - Walk: allow any repetition (default)
/// - Trail: no repeated edges
/// - Acyclic: no repeated nodes
/// - Simple: no repeated nodes except start=end
fn find_paths(
    hypergraph: &Hypergraph,
    bindings: &HashMap<String, Vec<serde_json::Value>>,
    start_binding: &str,
    end_binding: &str,
    rel_type: &str,
    mode: &PathMode,
    restrictor: &PathRestrictor,
    min_depth: usize,
    max_depth: usize,
    top_k: Option<usize>,
    _weight_field: Option<&str>,
) -> Result<Vec<serde_json::Value>> {
    use crate::analysis::pathfinding;

    let rel_upper = rel_type.to_uppercase();

    // LONGEST ACYCLIC on CAUSES → narrative diameter (DAG longest path)
    if *mode == PathMode::Longest && *restrictor == PathRestrictor::Acyclic && rel_upper == "CAUSES"
    {
        // Extract narrative_id from any bound situation
        let start_ids = collect_situation_ids(bindings, start_binding);
        let narrative_id = start_ids
            .first()
            .and_then(|sid| hypergraph.get_situation(sid).ok())
            .and_then(|s| s.narrative_id.clone());
        let nid = narrative_id.as_deref().unwrap_or("unknown");
        let dag = crate::analysis::graph_projection::build_causal_dag(hypergraph, nid)?;
        return match pathfinding::narrative_diameter(&dag)? {
            Some(sp) => {
                let path_uuids: Vec<String> = sp
                    .path
                    .iter()
                    .map(|&i| dag.situations[i].to_string())
                    .collect();
                Ok(vec![serde_json::json!({
                    "path": path_uuids,
                    "length": sp.path.len() - 1,
                    "total_weight": sp.total_weight,
                })])
            }
            None => Ok(vec![]),
        };
    }

    // PARTICIPATES edge queries → use CoGraph with Dijkstra/Yen's
    if rel_upper == "PARTICIPATES" {
        let entity_ids = collect_situation_ids(bindings, start_binding); // reuse UUID collector
        let end_entity_ids = collect_situation_ids(bindings, end_binding);
        if entity_ids.is_empty() || end_entity_ids.is_empty() {
            return Ok(vec![]);
        }

        // Determine narrative_id from first entity
        let nid = entity_ids
            .first()
            .and_then(|eid| hypergraph.get_entity(eid).ok())
            .and_then(|e| e.narrative_id.clone())
            .unwrap_or_default();
        let graph = crate::analysis::graph_projection::build_co_graph(hypergraph, &nid)?;

        let mut results = Vec::new();
        for &src in &entity_ids {
            let src_idx = graph.entities.iter().position(|&e| e == src);
            for &tgt in &end_entity_ids {
                let tgt_idx = graph.entities.iter().position(|&e| e == tgt);
                let (Some(si), Some(ti)) = (src_idx, tgt_idx) else {
                    continue;
                };
                if si == ti {
                    continue;
                }

                if let Some(k) = top_k {
                    // Yen's K-shortest
                    let paths = pathfinding::yen_k_shortest(&graph, si, ti, k);
                    for sp in paths {
                        let path_uuids: Vec<String> = sp
                            .path
                            .iter()
                            .map(|&i| graph.entities[i].to_string())
                            .collect();
                        results.push(serde_json::json!({
                            "path": path_uuids, "length": sp.path.len() - 1,
                            "total_weight": sp.total_weight,
                        }));
                    }
                } else {
                    // Dijkstra weighted shortest
                    if let Some(sp) = pathfinding::dijkstra(&graph, si, ti) {
                        let path_uuids: Vec<String> = sp
                            .path
                            .iter()
                            .map(|&i| graph.entities[i].to_string())
                            .collect();
                        results.push(serde_json::json!({
                            "path": path_uuids, "length": sp.path.len() - 1,
                            "total_weight": sp.total_weight,
                        }));
                    }
                }
            }
        }
        return Ok(results);
    }

    // CAUSES edge queries (original BFS-based path finding)
    if rel_upper != "CAUSES" {
        return Err(TensaError::QueryError(format!(
            "PATH queries support CAUSES and PARTICIPATES edges, got '{}'",
            rel_type
        )));
    }

    let start_ids = collect_situation_ids(bindings, start_binding);
    let end_ids = collect_situation_ids(bindings, end_binding);
    let mut result_paths = Vec::new();
    let max_paths = 1000;

    for start_id in &start_ids {
        for end_id in &end_ids {
            if start_id == end_id {
                continue;
            }
            match mode {
                PathMode::Shortest => {
                    if let Some(path) = bfs_shortest_path(hypergraph, start_id, end_id, max_depth)?
                    {
                        if path.len() >= min_depth + 1
                            && path_satisfies_restrictor(&path, restrictor)
                        {
                            result_paths.push(path_to_json(&path));
                        }
                    }
                }
                PathMode::All => {
                    let mut visited_nodes = std::collections::HashSet::new();
                    if *restrictor != PathRestrictor::Walk {
                        visited_nodes.insert(*start_id);
                    }
                    let mut all_paths = Vec::new();
                    dfs_all_paths(
                        hypergraph,
                        start_id,
                        end_id,
                        &mut visited_nodes,
                        &mut vec![*start_id],
                        min_depth,
                        max_depth,
                        &mut all_paths,
                        max_paths,
                    )?;
                    for path in all_paths {
                        if path_satisfies_restrictor(&path, restrictor) {
                            result_paths.push(path_to_json(&path));
                        }
                    }
                }
                PathMode::Longest => {
                    // Longest on CAUSES without ACYCLIC handled above
                    return Err(TensaError::QueryError(
                        "LONGEST mode on CAUSES requires ACYCLIC restrictor".into(),
                    ));
                }
            }
            if result_paths.len() >= max_paths {
                break;
            }
        }
        if result_paths.len() >= max_paths {
            break;
        }
    }

    Ok(result_paths)
}

/// Execute a MATCH FLOW query (max-flow or min-cut).
fn execute_flow(
    hypergraph: &Hypergraph,
    bindings: &HashMap<String, Vec<serde_json::Value>>,
    start_binding: &str,
    end_binding: &str,
    rel_type: &str,
    flow_type: &FlowType,
) -> Result<Vec<serde_json::Value>> {
    use crate::analysis::pathfinding;

    let src_ids = collect_situation_ids(bindings, start_binding);
    let tgt_ids = collect_situation_ids(bindings, end_binding);

    let src_id = src_ids
        .first()
        .ok_or_else(|| TensaError::QueryError("No source entity for FLOW".into()))?;
    let tgt_id = tgt_ids
        .first()
        .ok_or_else(|| TensaError::QueryError("No target entity for FLOW".into()))?;

    // Determine narrative from first entity
    let nid = if rel_type.to_uppercase() == "PARTICIPATES" {
        hypergraph
            .get_entity(src_id)
            .ok()
            .and_then(|e| e.narrative_id.clone())
            .unwrap_or_default()
    } else {
        hypergraph
            .get_situation(src_id)
            .ok()
            .and_then(|s| s.narrative_id.clone())
            .unwrap_or_default()
    };

    let graph = crate::analysis::graph_projection::build_co_graph(hypergraph, &nid)?;
    let src_idx = graph
        .entities
        .iter()
        .position(|&e| e == *src_id)
        .ok_or_else(|| TensaError::QueryError("Source not found in graph".into()))?;
    let tgt_idx = graph
        .entities
        .iter()
        .position(|&e| e == *tgt_id)
        .ok_or_else(|| TensaError::QueryError("Target not found in graph".into()))?;

    let (flow_val, cut_edges) = pathfinding::max_flow(&graph, src_idx, tgt_idx);

    let cut_json: Vec<serde_json::Value> = cut_edges
        .iter()
        .map(|&(a, b)| {
            serde_json::json!([graph.entities[a].to_string(), graph.entities[b].to_string()])
        })
        .collect();

    match flow_type {
        FlowType::Max => Ok(vec![
            serde_json::json!({"flow": flow_val, "cut_edges": cut_json}),
        ]),
        FlowType::MinCut => Ok(vec![
            serde_json::json!({"flow": flow_val, "cut_edges": cut_json}),
        ]),
    }
}

/// Check if a path satisfies the given GQL restrictor.
fn path_satisfies_restrictor(path: &[Uuid], restrictor: &PathRestrictor) -> bool {
    match restrictor {
        PathRestrictor::Walk => true, // no restrictions
        PathRestrictor::Trail => {
            // No repeated edges: check consecutive pairs are unique
            let edges: Vec<(Uuid, Uuid)> = path.windows(2).map(|w| (w[0], w[1])).collect();
            let unique: std::collections::HashSet<_> = edges.iter().collect();
            unique.len() == edges.len()
        }
        PathRestrictor::Acyclic => {
            // No repeated nodes
            let unique: std::collections::HashSet<_> = path.iter().collect();
            unique.len() == path.len()
        }
        PathRestrictor::Simple => {
            // No repeated nodes, except start may equal end
            if path.len() <= 1 {
                return true;
            }
            let interior = &path[1..];
            let unique: std::collections::HashSet<_> = interior.iter().collect();
            unique.len() == interior.len()
        }
    }
}

/// Collect situation UUIDs from a binding.
fn collect_situation_ids(
    bindings: &HashMap<String, Vec<serde_json::Value>>,
    binding: &str,
) -> Vec<Uuid> {
    bindings
        .get(binding)
        .map(|values| {
            values
                .iter()
                .filter_map(|v| {
                    v.get("id")
                        .and_then(|id| id.as_str())
                        .and_then(|s| Uuid::parse_str(s).ok())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// BFS shortest path between two situations via CAUSES edges.
fn bfs_shortest_path(
    hypergraph: &Hypergraph,
    start: &Uuid,
    end: &Uuid,
    max_depth: usize,
) -> Result<Option<Vec<Uuid>>> {
    use std::collections::{HashMap as StdMap, VecDeque};

    let mut queue = VecDeque::new();
    let mut visited: StdMap<Uuid, Uuid> = StdMap::new(); // child -> parent
    queue.push_back((*start, 0usize));
    visited.insert(*start, *start); // sentinel: start's parent is itself

    while let Some((current, depth)) = queue.pop_front() {
        if current == *end {
            // Reconstruct path
            let mut path = vec![current];
            let mut node = current;
            while visited[&node] != node {
                node = visited[&node];
                path.push(node);
            }
            path.reverse();
            return Ok(Some(path));
        }
        if depth >= max_depth {
            continue;
        }
        let consequences = hypergraph.get_consequences(&current)?;
        for link in consequences {
            // Skip soft-deleted or missing situations
            if hypergraph.get_situation(&link.to_situation).is_err() {
                continue;
            }
            if !visited.contains_key(&link.to_situation) {
                visited.insert(link.to_situation, current);
                queue.push_back((link.to_situation, depth + 1));
            }
        }
    }

    Ok(None) // not reachable
}

/// DFS all paths between two situations via CAUSES edges.
fn dfs_all_paths(
    hypergraph: &Hypergraph,
    current: &Uuid,
    end: &Uuid,
    visited: &mut std::collections::HashSet<Uuid>,
    current_path: &mut Vec<Uuid>,
    min_depth: usize,
    max_depth: usize,
    results: &mut Vec<Vec<Uuid>>,
    max_results: usize,
) -> Result<()> {
    if results.len() >= max_results {
        return Ok(());
    }
    if *current == *end {
        // path length = edges = nodes - 1
        if current_path.len() - 1 >= min_depth {
            results.push(current_path.clone());
        }
        return Ok(());
    }
    if current_path.len() - 1 >= max_depth {
        return Ok(());
    }
    let consequences = hypergraph.get_consequences(current)?;
    for link in consequences {
        // Skip soft-deleted situations
        if hypergraph
            .get_situation_include_deleted(&link.to_situation)
            .map(|s| s.deleted_at.is_some())
            .unwrap_or(true)
        {
            continue;
        }
        if !visited.contains(&link.to_situation) {
            visited.insert(link.to_situation);
            current_path.push(link.to_situation);
            dfs_all_paths(
                hypergraph,
                &link.to_situation,
                end,
                visited,
                current_path,
                min_depth,
                max_depth,
                results,
                max_results,
            )?;
            current_path.pop();
            visited.remove(&link.to_situation);
        }
    }
    Ok(())
}

/// Convert a path (Vec<Uuid>) to a JSON result row.
fn path_to_json(path: &[Uuid]) -> serde_json::Value {
    serde_json::json!({
        "path": path.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        "length": path.len() - 1,
    })
}

/// Execute an EXPORT statement, returning the exported content as a result row.
fn execute_export(
    narrative_id: &str,
    format: &str,
    hypergraph: &Hypergraph,
) -> Result<Vec<ResultRow>> {
    let fmt: crate::export::ExportFormat = format.parse()?;

    let output = crate::export::export_narrative(narrative_id, fmt, hypergraph, false)?;
    let body_str =
        String::from_utf8(output.body).unwrap_or_else(|_| "<binary content>".to_string());

    let mut row = ResultRow::new();
    row.insert(
        "export".to_string(),
        serde_json::json!({
            "narrative_id": narrative_id,
            "format": format,
            "content_type": output.content_type,
            "body": body_str,
        }),
    );
    Ok(vec![row])
}

/// Execute a DML mutation against the hypergraph.
///
/// Returns a single result row with the created/modified/deleted ID and status.
pub fn execute_mutation(
    mutation: &MutationStatement,
    hypergraph: &Hypergraph,
) -> Result<Vec<ResultRow>> {
    match mutation {
        MutationStatement::CreateNarrative {
            id,
            title,
            genre,
            tags,
        } => {
            let registry = NarrativeRegistry::new(hypergraph.store_arc());
            let narrative = crate::narrative::types::Narrative {
                id: id.clone(),
                title: title.clone().unwrap_or_else(|| id.clone()),
                genre: genre.clone(),
                tags: tags.clone(),
                source: None,
                project_id: None,
                description: None,
                authors: vec![],
                language: None,
                publication_date: None,
                cover_url: None,
                custom_properties: std::collections::HashMap::new(),
                entity_count: 0,
                situation_count: 0,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            let created_id = registry.create(narrative)?;
            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(created_id));
            row.insert("status".into(), serde_json::json!("created"));
            Ok(vec![row])
        }

        MutationStatement::CreateEntity {
            entity_type,
            properties,
            narrative_id,
            confidence,
        } => {
            let et = parse_entity_type(entity_type)?;

            // Build properties JSON from PropPair list
            let mut props = serde_json::Map::new();
            for pp in properties {
                props.insert(pp.key.clone(), pp.value.to_json());
            }

            let entity = Entity {
                id: Uuid::now_v7(),
                entity_type: et,
                properties: serde_json::Value::Object(props),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: confidence.map(|c| c as f32).unwrap_or(0.5),
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: narrative_id.clone(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            let id = hypergraph.create_entity(entity)?;
            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id.to_string()));
            row.insert("status".into(), serde_json::json!("created"));
            Ok(vec![row])
        }

        MutationStatement::CreateSituation {
            level,
            content,
            narrative_id,
            confidence,
        } => {
            let nl = parse_narrative_level(level)?;

            let raw_content: Vec<ContentBlock> =
                content.iter().map(|s| ContentBlock::text(s)).collect();

            let situation = Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: None,
                    end: None,
                    granularity: TimeGranularity::Approximate,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content,
                narrative_level: nl,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: confidence.map(|c| c as f32).unwrap_or(0.5),
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: narrative_id.clone(),
                source_chunk_id: None,
                source_span: None,
                synopsis: None,
                manuscript_order: None,
                parent_situation_id: None,
                label: None,
                status: None,
                keywords: vec![],
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            let id = hypergraph.create_situation(situation)?;
            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id.to_string()));
            row.insert("status".into(), serde_json::json!("created"));
            Ok(vec![row])
        }

        MutationStatement::DeleteEntity { id } => {
            let uuid = Uuid::parse_str(id)
                .map_err(|e| TensaError::QueryError(format!("Invalid UUID: {}", e)))?;
            hypergraph.delete_entity(&uuid)?;
            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id));
            row.insert("status".into(), serde_json::json!("deleted"));
            Ok(vec![row])
        }

        MutationStatement::DeleteSituation { id } => {
            let uuid = Uuid::parse_str(id)
                .map_err(|e| TensaError::QueryError(format!("Invalid UUID: {}", e)))?;
            hypergraph.delete_situation(&uuid)?;
            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id));
            row.insert("status".into(), serde_json::json!("deleted"));
            Ok(vec![row])
        }

        MutationStatement::DeleteNarrative { id } => {
            let registry = NarrativeRegistry::new(hypergraph.store_arc());
            registry.delete(id)?;
            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id));
            row.insert("status".into(), serde_json::json!("deleted"));
            Ok(vec![row])
        }

        MutationStatement::UpdateEntity { id, set_pairs } => {
            let uuid = Uuid::parse_str(id)
                .map_err(|e| TensaError::QueryError(format!("Invalid UUID: {}", e)))?;

            // Clone set_pairs for the closure
            let pairs = set_pairs.clone();
            hypergraph.update_entity(&uuid, |entity| {
                for (key, value) in &pairs {
                    let json_val = value.to_json();
                    match key.as_str() {
                        "confidence" => {
                            if let Some(f) = json_val.as_f64() {
                                entity.confidence = f as f32;
                            }
                        }
                        "maturity" => {
                            if let Some(s) = json_val.as_str() {
                                if let Ok(m) = serde_json::from_value(serde_json::json!(s)) {
                                    entity.maturity = m;
                                }
                            }
                        }
                        "narrative_id" => {
                            entity.narrative_id = json_val.as_str().map(|s| s.to_string());
                        }
                        _ => {
                            // Put into properties
                            if let Some(obj) = entity.properties.as_object_mut() {
                                obj.insert(key.clone(), json_val);
                            }
                        }
                    }
                }
            })?;

            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id));
            row.insert("status".into(), serde_json::json!("updated"));
            Ok(vec![row])
        }

        MutationStatement::UpdateNarrative { id, set_pairs } => {
            let registry = NarrativeRegistry::new(hypergraph.store_arc());
            let pairs = set_pairs.clone();
            registry.update(id, |narrative| {
                for (key, value) in &pairs {
                    let json_val = value.to_json();
                    match key.as_str() {
                        "title" => {
                            if let Some(s) = json_val.as_str() {
                                narrative.title = s.to_string();
                            }
                        }
                        "genre" => {
                            narrative.genre = json_val.as_str().map(|s| s.to_string());
                        }
                        _ => {}
                    }
                }
            })?;

            let mut row = ResultRow::new();
            row.insert("id".into(), serde_json::json!(id));
            row.insert("status".into(), serde_json::json!("updated"));
            Ok(vec![row])
        }

        MutationStatement::AddParticipant {
            entity_id,
            situation_id,
            role,
            action,
        } => {
            let eid = Uuid::parse_str(entity_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid entity UUID: {}", e)))?;
            let sid = Uuid::parse_str(situation_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid situation UUID: {}", e)))?;

            let parsed_role = parse_role(role);

            hypergraph.add_participant(Participation {
                entity_id: eid,
                situation_id: sid,
                role: parsed_role,
                info_set: None,
                action: action.clone(),
                payoff: None,
                seq: 0,
            })?;

            let mut row = ResultRow::new();
            row.insert("entity_id".into(), serde_json::json!(entity_id));
            row.insert("situation_id".into(), serde_json::json!(situation_id));
            row.insert("status".into(), serde_json::json!("created"));
            Ok(vec![row])
        }

        MutationStatement::RemoveParticipant {
            entity_id,
            situation_id,
        } => {
            let eid = Uuid::parse_str(entity_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid entity UUID: {}", e)))?;
            let sid = Uuid::parse_str(situation_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid situation UUID: {}", e)))?;

            hypergraph.remove_participant(&eid, &sid, None)?;

            let mut row = ResultRow::new();
            row.insert("entity_id".into(), serde_json::json!(entity_id));
            row.insert("situation_id".into(), serde_json::json!(situation_id));
            row.insert("status".into(), serde_json::json!("deleted"));
            Ok(vec![row])
        }

        MutationStatement::AddCause {
            from_id,
            to_id,
            causal_type,
            strength,
            mechanism,
        } => {
            let from = Uuid::parse_str(from_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid from UUID: {}", e)))?;
            let to = Uuid::parse_str(to_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid to UUID: {}", e)))?;

            let ct = causal_type
                .as_deref()
                .map(parse_causal_type)
                .transpose()?
                .unwrap_or(CausalType::Contributing);

            let link = CausalLink {
                from_situation: from,
                to_situation: to,
                mechanism: mechanism.clone(),
                strength: strength.map(|s| s as f32).unwrap_or(0.5),
                causal_type: ct,
                maturity: MaturityLevel::Candidate,
            };

            hypergraph.add_causal_link(link)?;

            let mut row = ResultRow::new();
            row.insert("from_id".into(), serde_json::json!(from_id));
            row.insert("to_id".into(), serde_json::json!(to_id));
            row.insert("status".into(), serde_json::json!("created"));
            Ok(vec![row])
        }

        MutationStatement::RemoveCause { from_id, to_id } => {
            let from = Uuid::parse_str(from_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid from UUID: {}", e)))?;
            let to = Uuid::parse_str(to_id)
                .map_err(|e| TensaError::QueryError(format!("Invalid to UUID: {}", e)))?;

            hypergraph.remove_causal_link(&from, &to)?;

            let mut row = ResultRow::new();
            row.insert("from_id".into(), serde_json::json!(from_id));
            row.insert("to_id".into(), serde_json::json!(to_id));
            row.insert("status".into(), serde_json::json!("deleted"));
            Ok(vec![row])
        }
    }
}

fn parse_entity_type(s: &str) -> Result<EntityType> {
    match s {
        "Actor" => Ok(EntityType::Actor),
        "Location" => Ok(EntityType::Location),
        "Artifact" => Ok(EntityType::Artifact),
        "Concept" => Ok(EntityType::Concept),
        "Organization" => Ok(EntityType::Organization),
        other => Err(TensaError::QueryError(format!(
            "Unknown entity type: {}",
            other
        ))),
    }
}

fn parse_narrative_level(s: &str) -> Result<NarrativeLevel> {
    match s {
        "Story" => Ok(NarrativeLevel::Story),
        "Arc" => Ok(NarrativeLevel::Arc),
        "Sequence" => Ok(NarrativeLevel::Sequence),
        "Scene" => Ok(NarrativeLevel::Scene),
        "Beat" => Ok(NarrativeLevel::Beat),
        "Event" => Ok(NarrativeLevel::Event),
        other => Err(TensaError::QueryError(format!(
            "Unknown narrative level: {}",
            other
        ))),
    }
}

fn parse_role(s: &str) -> Role {
    match s {
        "Protagonist" => Role::Protagonist,
        "Antagonist" => Role::Antagonist,
        "Witness" => Role::Witness,
        "Target" => Role::Target,
        "Instrument" => Role::Instrument,
        "Confidant" => Role::Confidant,
        "Informant" => Role::Informant,
        "Recipient" => Role::Recipient,
        "Bystander" => Role::Bystander,
        "SubjectOfDiscussion" => Role::SubjectOfDiscussion,
        other => Role::Custom(other.to_string()),
    }
}

fn parse_causal_type(s: &str) -> Result<CausalType> {
    match s {
        "Necessary" => Ok(CausalType::Necessary),
        "Sufficient" => Ok(CausalType::Sufficient),
        "Contributing" => Ok(CausalType::Contributing),
        "Enabling" => Ok(CausalType::Enabling),
        other => Err(TensaError::QueryError(format!(
            "Unknown causal type: {}",
            other
        ))),
    }
}

fn scan_by_type(hypergraph: &Hypergraph, type_name: &str) -> Result<Vec<serde_json::Value>> {
    // Check if it's an entity type
    let entity_type = match type_name {
        "Actor" => Some(EntityType::Actor),
        "Location" => Some(EntityType::Location),
        "Artifact" => Some(EntityType::Artifact),
        "Concept" => Some(EntityType::Concept),
        "Organization" => Some(EntityType::Organization),
        _ => None,
    };

    if let Some(et) = entity_type {
        let entities = hypergraph.list_entities_by_type(&et)?;
        return entities
            .into_iter()
            .map(|e| serde_json::to_value(e).map_err(|e| TensaError::Serialization(e.to_string())))
            .collect();
    }

    // Check if it's a situation (by narrative level or generic "Situation")
    if type_name == "Situation" {
        let pairs = hypergraph.store().prefix_scan(b"s/")?;
        return pairs
            .into_iter()
            .map(|(_k, v)| {
                let sit: Situation = serde_json::from_slice(&v)?;
                serde_json::to_value(sit).map_err(|e| TensaError::Serialization(e.to_string()))
            })
            .collect();
    }

    // Try matching narrative levels
    let level = match type_name {
        "Story" => Some(NarrativeLevel::Story),
        "Arc" => Some(NarrativeLevel::Arc),
        "Sequence" => Some(NarrativeLevel::Sequence),
        "Scene" => Some(NarrativeLevel::Scene),
        "Beat" => Some(NarrativeLevel::Beat),
        "Event" => Some(NarrativeLevel::Event),
        _ => None,
    };

    if let Some(lvl) = level {
        let situations = hypergraph.list_situations_by_level(lvl)?;
        return situations
            .into_iter()
            .map(|s| serde_json::to_value(s).map_err(|e| TensaError::Serialization(e.to_string())))
            .collect();
    }

    Err(TensaError::QueryError(format!(
        "Unknown type: {}",
        type_name
    )))
}

/// Build a set of UUIDs currently in a binding (for fast pair pruning).
fn binding_uuids(
    bindings: &HashMap<String, Vec<serde_json::Value>>,
    name: &str,
) -> std::collections::HashSet<uuid::Uuid> {
    bindings
        .get(name)
        .into_iter()
        .flatten()
        .filter_map(|v| v.get("id").and_then(|s| s.as_str()))
        .filter_map(|s| uuid::Uuid::parse_str(s).ok())
        .collect()
}

/// After filtering bindings, prune edge_pairs to only pairs where both endpoints survived,
/// then propagate the constraint back to the bindings — keep only entities/situations
/// that still have at least one surviving pair.
fn prune_edge_pairs_to_bindings(
    bindings: &mut HashMap<String, Vec<serde_json::Value>>,
    edge_pairs: &mut HashMap<(String, String), Vec<(uuid::Uuid, uuid::Uuid)>>,
) {
    for ((from_b, to_b), pairs) in edge_pairs.iter_mut() {
        let from_set = binding_uuids(bindings, from_b);
        let to_set = binding_uuids(bindings, to_b);
        pairs.retain(|(f, t)| from_set.contains(f) && to_set.contains(t));

        // Back-propagate: keep only bindings that participate in at least one surviving pair.
        let mut surviving_from: std::collections::HashSet<uuid::Uuid> =
            std::collections::HashSet::with_capacity(pairs.len());
        let mut surviving_to: std::collections::HashSet<uuid::Uuid> =
            std::collections::HashSet::with_capacity(pairs.len());
        for (f, t) in pairs.iter() {
            surviving_from.insert(*f);
            surviving_to.insert(*t);
        }

        let retain_by_id =
            |vals: &mut Vec<serde_json::Value>,
             surviving: &std::collections::HashSet<uuid::Uuid>| {
                vals.retain(|v| {
                    v.get("id")
                        .and_then(|s| s.as_str())
                        .and_then(|s| uuid::Uuid::parse_str(s).ok())
                        .map(|u| surviving.contains(&u))
                        .unwrap_or(false)
                });
            };
        if let Some(vals) = bindings.get_mut(from_b) {
            retain_by_id(vals, &surviving_from);
        }
        if let Some(vals) = bindings.get_mut(to_b) {
            retain_by_id(vals, &surviving_to);
        }
    }
}

fn follow_edge(
    hypergraph: &Hypergraph,
    bindings: &mut HashMap<String, Vec<serde_json::Value>>,
    from_binding: &str,
    rel_type: &str,
    to_binding: &str,
) -> Result<Vec<(uuid::Uuid, uuid::Uuid)>> {
    use std::collections::HashSet;
    let from_values = bindings.get(from_binding).cloned().unwrap_or_default();
    let to_values = bindings.get(to_binding).cloned().unwrap_or_default();

    let mut matched_from = Vec::new();
    let mut matched_to = Vec::new();
    let mut seen_from: HashSet<uuid::Uuid> = HashSet::new();
    let mut seen_to: HashSet<uuid::Uuid> = HashSet::new();
    let mut pairs: Vec<(uuid::Uuid, uuid::Uuid)> = Vec::new();

    // Parse to_values' UUIDs once up front — they're reused across the from_values loop.
    let to_parsed: Vec<(uuid::Uuid, &serde_json::Value)> = to_values
        .iter()
        .filter_map(|v| {
            v.get("id")
                .and_then(|s| s.as_str())
                .and_then(|s| uuid::Uuid::parse_str(s).ok())
                .map(|u| (u, v))
        })
        .collect();

    // Empty rel_type defaults to PARTICIPATES (backward compat for untyped edges).
    let is_causes = rel_type.eq_ignore_ascii_case("CAUSES");

    for from_val in &from_values {
        let from_uuid = match from_val
            .get("id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok())
        {
            Some(u) => u,
            None => continue,
        };

        let reachable_to_ids: HashSet<uuid::Uuid> = if is_causes {
            hypergraph
                .get_consequences(&from_uuid)
                .unwrap_or_default()
                .iter()
                .map(|link| link.to_situation)
                .collect()
        } else {
            hypergraph
                .get_situations_for_entity(&from_uuid)
                .unwrap_or_default()
                .iter()
                .map(|p| p.situation_id)
                .collect()
        };

        for (to_uuid, to_val) in &to_parsed {
            if reachable_to_ids.contains(to_uuid) {
                pairs.push((from_uuid, *to_uuid));
                if seen_from.insert(from_uuid) {
                    matched_from.push(from_val.clone());
                }
                if seen_to.insert(*to_uuid) {
                    matched_to.push((*to_val).clone());
                }
            }
        }
    }

    bindings.insert(from_binding.to_string(), matched_from);
    bindings.insert(to_binding.to_string(), matched_to);
    Ok(pairs)
}

fn filter_properties(
    bindings: &mut HashMap<String, Vec<serde_json::Value>>,
    expr: &ConditionExpr,
    hypergraph: &Hypergraph,
) -> Result<()> {
    let mut cache = AnalysisCache::new();
    let mut co_graph_cache: Option<crate::analysis::graph_projection::CoGraph> = None;
    let binding_names = collect_bindings_from_expr(expr);
    for binding_name in binding_names {
        if let Some(values) = bindings.get_mut(&binding_name) {
            values.retain(|val| {
                evaluate_expr(
                    val,
                    &binding_name,
                    expr,
                    hypergraph,
                    &mut cache,
                    &mut co_graph_cache,
                )
            });
        }
    }
    Ok(())
}

/// Extract the binding name (part before the first '.') from a dotted field path.
fn binding_name(field: &str) -> &str {
    field.split('.').next().unwrap_or("")
}

// ─── Temporal Filtering (AT clause) ─────────────────────────────

/// Parse an Allen relation name string into the enum variant.
fn parse_allen_relation(s: &str) -> Result<AllenRelation> {
    match s.to_uppercase().as_str() {
        "BEFORE" => Ok(AllenRelation::Before),
        "AFTER" => Ok(AllenRelation::After),
        "MEETS" => Ok(AllenRelation::Meets),
        "MEETS_INVERSE" | "MET_BY" | "METBY" => Ok(AllenRelation::MetBy),
        "OVERLAPS" => Ok(AllenRelation::Overlaps),
        "OVERLAPS_INVERSE" | "OVERLAPPED_BY" | "OVERLAPPEDBY" => {
            Ok(AllenRelation::OverlappedBy)
        }
        "DURING" | "WITHIN" => Ok(AllenRelation::During),
        "DURING_INVERSE" | "CONTAINS" => Ok(AllenRelation::Contains),
        "STARTS" => Ok(AllenRelation::Starts),
        "STARTS_INVERSE" | "STARTED_BY" | "STARTEDBY" => Ok(AllenRelation::StartedBy),
        "FINISHES" => Ok(AllenRelation::Finishes),
        "FINISHES_INVERSE" | "FINISHED_BY" | "FINISHEDBY" => Ok(AllenRelation::FinishedBy),
        "EQUALS" => Ok(AllenRelation::Equals),
        other => Err(TensaError::QueryError(format!(
            "Unknown Allen relation: {}",
            other
        ))),
    }
}

/// Parse a datetime string from a QueryValue.
fn parse_temporal_value(val: &QueryValue) -> Result<DateTime<Utc>> {
    match val {
        QueryValue::String(s) => {
            // Try RFC 3339 first (e.g., "2025-01-01T00:00:00Z")
            if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                return Ok(dt.with_timezone(&Utc));
            }
            // Try date-only (e.g., "2025-01-01") — treat as start-of-day UTC
            if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                let dt = nd
                    .and_hms_opt(0, 0, 0)
                    .ok_or_else(|| TensaError::QueryError("Invalid date".into()))?;
                return Ok(DateTime::from_naive_utc_and_offset(dt, Utc));
            }
            Err(TensaError::QueryError(format!(
                "Cannot parse datetime: '{}'. Use RFC 3339 (e.g., 2025-01-01T00:00:00Z) or date (2025-01-01)",
                s
            )))
        }
        _ => Err(TensaError::QueryError(
            "AT clause value must be a datetime string".into(),
        )),
    }
}

/// Filter bindings by Allen temporal relation against a reference datetime.
///
/// Fuzzy Sprint Phase 5 — when the AT clause carries a `fuzzy` tail
/// (`AS FUZZY <rel> THRESHOLD <t>`), switches from crisp equality to the
/// graded-Allen degree and keeps candidates whose degree meets the
/// threshold. When the tail is absent, the crisp path is bit-identical
/// to the pre-Phase-5 semantics.
fn filter_temporal(
    bindings: &mut HashMap<String, Vec<serde_json::Value>>,
    at_clause: &AtClause,
) -> Result<()> {
    let target_relation = parse_allen_relation(&at_clause.relation)?;
    let ref_dt = parse_temporal_value(&at_clause.value)?;

    // Build a reference interval (point in time: start == end)
    let ref_interval = AllenInterval {
        start: Some(ref_dt),
        end: Some(ref_dt),
        granularity: TimeGranularity::Exact,
        relations: vec![],
        fuzzy_endpoints: None,
    };

    let bname = binding_name(&at_clause.field);

    // Resolve fuzzy tail (if any) once so we don't re-parse per row.
    let fuzzy = match at_clause.fuzzy.as_ref() {
        Some(cfg) => {
            let rel = parse_allen_relation(&cfg.relation)?;
            Some((rel, cfg.threshold))
        }
        None => None,
    };

    if let Some(values) = bindings.get_mut(bname) {
        values.retain(|val| {
            // Extract temporal interval from the JSON value
            let start = val
                .get("temporal")
                .and_then(|t| t.get("start"))
                .and_then(|s| s.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));
            let end = val
                .get("temporal")
                .and_then(|t| t.get("end"))
                .and_then(|s| s.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            // Parse fuzzy_endpoints if present on the row.
            let fuzzy_endpoints = val
                .get("temporal")
                .and_then(|t| t.get("fuzzy_endpoints"))
                .and_then(|fe| {
                    serde_json::from_value::<Option<crate::fuzzy::allen::FuzzyEndpoints>>(
                        fe.clone(),
                    )
                    .ok()
                    .flatten()
                });

            // Skip items without any temporal information.
            if start.is_none() && end.is_none() && fuzzy_endpoints.is_none() {
                return false;
            }

            let item_interval = AllenInterval {
                start,
                end,
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints,
            };

            match &fuzzy {
                Some((fuzzy_rel, threshold)) => crate::fuzzy::allen::fuzzy_relation_holds(
                    &item_interval,
                    &ref_interval,
                    *fuzzy_rel,
                    *threshold,
                ),
                None => match relation_between(&item_interval, &ref_interval) {
                    Ok(actual) => actual == target_relation,
                    Err(_) => false,
                },
            }
        });
    }

    Ok(())
}

// ─── Vector Similarity (NEAR clause) ────────────────────────────

/// Filter bindings to the k nearest neighbors by embedding similarity.
fn filter_vector_near(
    bindings: &mut HashMap<String, Vec<serde_json::Value>>,
    binding: &str,
    query_text: &str,
    k: usize,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
) -> Result<()> {
    let vi = vector_index.ok_or_else(|| {
        TensaError::QueryError("NEAR queries require a vector index to be configured".into())
    })?;
    let emb = embedder.ok_or_else(|| {
        TensaError::QueryError("NEAR queries require an embedding provider to be configured".into())
    })?;

    let query_embedding = emb.embed_text(query_text)?;
    let results = vi.search(&query_embedding, k)?;

    // Collect matched IDs
    let matched_ids: HashSet<String> = results.iter().map(|r| r.id.to_string()).collect();

    if let Some(values) = bindings.get_mut(binding) {
        values.retain(|val| {
            val.get("id")
                .and_then(|v| v.as_str())
                .map(|id| matched_ids.contains(id))
                .unwrap_or(false)
        });
    }

    Ok(())
}

// ─── Spatial Filtering (SPATIAL clause) ─────────────────────────

/// Haversine distance between two lat/lon points in kilometers.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0; // Earth radius in km
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

/// Filter bindings by geospatial proximity.
fn filter_spatial(
    bindings: &mut HashMap<String, Vec<serde_json::Value>>,
    field: &str,
    radius_km: f64,
    center_lat: f64,
    center_lon: f64,
) -> Result<()> {
    let bname = binding_name(field);

    if let Some(values) = bindings.get_mut(bname) {
        values.retain(|val| {
            let lat = val
                .get("spatial")
                .and_then(|s| s.get("latitude"))
                .and_then(|v| v.as_f64());
            let lon = val
                .get("spatial")
                .and_then(|s| s.get("longitude"))
                .and_then(|v| v.as_f64());

            match (lat, lon) {
                (Some(lat), Some(lon)) => {
                    haversine_km(lat, lon, center_lat, center_lon) <= radius_km
                }
                _ => false, // no spatial data = excluded
            }
        });
    }

    Ok(())
}

/// Recursively evaluate a boolean expression tree against a JSON value.
fn evaluate_expr(
    val: &serde_json::Value,
    binding_name: &str,
    expr: &ConditionExpr,
    hypergraph: &Hypergraph,
    cache: &mut AnalysisCache,
    co_graph_cache: &mut Option<crate::analysis::graph_projection::CoGraph>,
) -> bool {
    match expr {
        ConditionExpr::Single(cond) => {
            // Graph function conditions: evaluate inline (co-graph shared across all rows)
            if let Some(gf) = &cond.graph_func {
                if gf.args.first().map(|a| a.as_str()) != Some(binding_name) {
                    return true;
                }
                let gf_val = evaluate_graph_func_return(gf, val, None, hypergraph, co_graph_cache);
                return evaluate_condition(&gf_val, &cond.op, &cond.value);
            }

            let parts: Vec<&str> = cond.field.splitn(2, '.').collect();
            if parts.len() < 2 || parts[0] != binding_name {
                return true;
            }
            let field_path = parts[1];

            if let Some(metric) = field_path.strip_prefix("an.") {
                let an_val = resolve_analysis_property(val, metric, hypergraph, cache);
                return evaluate_condition(&an_val, &cond.op, &cond.value);
            }

            let field_val = resolve_field(val, field_path);
            evaluate_condition(field_val, &cond.op, &cond.value)
        }
        ConditionExpr::And(children) => children
            .iter()
            .all(|c| evaluate_expr(val, binding_name, c, hypergraph, cache, co_graph_cache)),
        ConditionExpr::Or(children) => children
            .iter()
            .any(|c| evaluate_expr(val, binding_name, c, hypergraph, cache, co_graph_cache)),
    }
}

/// Collect all distinct binding names from a condition expression.
fn collect_bindings_from_expr(expr: &ConditionExpr) -> Vec<String> {
    let mut names = Vec::new();
    collect_bindings_recursive(expr, &mut names);
    names.sort();
    names.dedup();
    names
}

fn collect_bindings_recursive(expr: &ConditionExpr, out: &mut Vec<String>) {
    match expr {
        ConditionExpr::Single(cond) => {
            if let Some(binding) = cond.field.split('.').next() {
                out.push(binding.to_string());
            }
        }
        ConditionExpr::And(children) | ConditionExpr::Or(children) => {
            for child in children {
                collect_bindings_recursive(child, out);
            }
        }
    }
}

fn resolve_field<'a>(val: &'a serde_json::Value, path: &str) -> &'a serde_json::Value {
    static NULL: serde_json::Value = serde_json::Value::Null;
    path.split('.').fold(val, |current, part| {
        current
            .get(part)
            .or_else(|| current.get("properties").and_then(|p| p.get(part)))
            .unwrap_or(&NULL)
    })
}

/// Per-row cache for `an.*` KV lookups, keyed by (KV key bytes).
/// Avoids redundant deserialization when a query references multiple metrics
/// from the same KV entry (e.g. `e.an.betweenness, e.an.closeness`).
type AnalysisCache = HashMap<Vec<u8>, Option<serde_json::Value>>;

/// Resolve a virtual `an.*` analysis property from the KV store.
///
/// Given an entity/situation JSON value and a metric name (e.g. "pagerank",
/// "betweenness", "community_id"), looks up the pre-computed analysis result.
/// Returns `serde_json::Value::Null` if no result exists — this allows
/// WHERE clauses to simply filter out entities with no computed score.
fn resolve_analysis_property(
    val: &serde_json::Value,
    metric: &str,
    hypergraph: &Hypergraph,
    cache: &mut AnalysisCache,
) -> serde_json::Value {
    let entity_id = val.get("id").and_then(|v| v.as_str());
    let narrative_id = val.get("narrative_id").and_then(|v| v.as_str());

    let (entity_id, narrative_id) = match (entity_id, narrative_id) {
        (Some(eid), Some(nid)) => (eid, nid),
        _ => return serde_json::Value::Null,
    };

    // Map metric names to KV prefixes and extraction logic
    match metric {
        // Existing centrality metrics (stored as CentralityResult JSON)
        "betweenness" | "closeness" | "degree" | "community_id" => lookup_analysis_field(
            hypergraph,
            keys::ANALYSIS_CENTRALITY,
            narrative_id,
            entity_id,
            metric,
            cache,
        ),
        // Level 1 algorithm results (Sprint 1)
        "pagerank" => lookup_analysis_scalar(hypergraph, b"an/pr/", narrative_id, entity_id, cache),
        "eigenvector" => {
            lookup_analysis_scalar(hypergraph, b"an/ev_c/", narrative_id, entity_id, cache)
        }
        "harmonic" => lookup_analysis_scalar(hypergraph, b"an/hc/", narrative_id, entity_id, cache),
        "hub_score" => lookup_analysis_field(
            hypergraph,
            b"an/hits/",
            narrative_id,
            entity_id,
            "hub_score",
            cache,
        ),
        "authority_score" => lookup_analysis_field(
            hypergraph,
            b"an/hits/",
            narrative_id,
            entity_id,
            "authority_score",
            cache,
        ),
        "temporal_pagerank" => {
            lookup_analysis_scalar(hypergraph, b"an/tpr/", narrative_id, entity_id, cache)
        }
        "causal_influence" => {
            lookup_analysis_scalar(hypergraph, b"an/ci/", narrative_id, entity_id, cache)
        }
        "kcore" => lookup_analysis_scalar(hypergraph, b"an/kc/", narrative_id, entity_id, cache),
        "label" => lookup_analysis_scalar(hypergraph, b"an/lp/", narrative_id, entity_id, cache),
        "bottleneck_score" => {
            lookup_analysis_scalar(hypergraph, b"an/ib/", narrative_id, entity_id, cache)
        }
        "is_articulation_point" => lookup_analysis_field(
            hypergraph,
            b"an/tp/",
            narrative_id,
            entity_id,
            "is_articulation_point",
            cache,
        ),
        "is_bridge_endpoint" => lookup_analysis_field(
            hypergraph,
            b"an/tp/",
            narrative_id,
            entity_id,
            "is_bridge_endpoint",
            cache,
        ),
        // Per-actor Reagan arc classification (v0.74.1). Produced by
        // `INFER ARCS FOR e:Actor` and `INFER ACTOR_ARCS FOR n:Narrative`
        // and stored as the full `ArcClassification` JSON at `an/aa/`.
        "arc_type" => lookup_analysis_field(
            hypergraph,
            b"an/aa/",
            narrative_id,
            entity_id,
            "arc_type",
            cache,
        ),
        "arc_confidence" => lookup_analysis_field(
            hypergraph,
            b"an/aa/",
            narrative_id,
            entity_id,
            "confidence",
            cache,
        ),
        "arc_signal_quality" => lookup_analysis_field(
            hypergraph,
            b"an/aa/",
            narrative_id,
            entity_id,
            "signal_quality",
            cache,
        ),
        _ => serde_json::Value::Null,
    }
}

/// Look up a scalar analysis value stored directly as a JSON value at a KV prefix.
fn lookup_analysis_scalar(
    hypergraph: &Hypergraph,
    prefix: &[u8],
    narrative_id: &str,
    entity_id: &str,
    cache: &mut AnalysisCache,
) -> serde_json::Value {
    let key = analysis_key(prefix, &[narrative_id, entity_id]);
    if let Some(cached) = cache.get(&key) {
        return cached.clone().unwrap_or(serde_json::Value::Null);
    }
    let result = match hypergraph.store().get(&key) {
        Ok(Some(bytes)) => serde_json::from_slice(&bytes).ok(),
        _ => None,
    };
    let val = result.clone().unwrap_or(serde_json::Value::Null);
    cache.insert(key, result);
    val
}

/// Look up a specific field within a JSON object stored at a KV prefix.
fn lookup_analysis_field(
    hypergraph: &Hypergraph,
    prefix: &[u8],
    narrative_id: &str,
    entity_id: &str,
    field: &str,
    cache: &mut AnalysisCache,
) -> serde_json::Value {
    let key = analysis_key(prefix, &[narrative_id, entity_id]);
    if let Some(cached) = cache.get(&key) {
        return cached
            .as_ref()
            .and_then(|obj| obj.get(field).cloned())
            .unwrap_or(serde_json::Value::Null);
    }
    let result = match hypergraph.store().get(&key) {
        Ok(Some(bytes)) => serde_json::from_slice::<serde_json::Value>(&bytes).ok(),
        _ => None,
    };
    let val = result
        .as_ref()
        .and_then(|obj| obj.get(field).cloned())
        .unwrap_or(serde_json::Value::Null);
    cache.insert(key, result);
    val
}

/// Evaluate an inline graph function on entity JSON value(s).
///
/// Single-arg functions (triangles, clustering): use `val` as the entity.
/// Two-arg functions (jaccard, adamic_adar, etc.): `val` = first entity, `val2` = second.
/// Builds the co-graph lazily on first call and caches it.
fn evaluate_graph_func_return(
    gf: &GraphFunc,
    val: &serde_json::Value,
    val2: Option<&serde_json::Value>,
    hypergraph: &Hypergraph,
    co_graph_cache: &mut Option<crate::analysis::graph_projection::CoGraph>,
) -> serde_json::Value {
    // Extract entity UUID and narrative_id
    let eid = val.get("id").and_then(|v| v.as_str());
    let nid = val.get("narrative_id").and_then(|v| v.as_str());
    let (Some(eid), Some(nid)) = (eid, nid) else {
        return serde_json::Value::Null;
    };

    // Build co-graph lazily
    if co_graph_cache.is_none() {
        if let Ok(g) = crate::analysis::graph_projection::build_co_graph(hypergraph, nid) {
            *co_graph_cache = Some(g);
        } else {
            return serde_json::Value::Null;
        }
    }
    let graph = co_graph_cache.as_ref().unwrap();

    let a_idx = eid
        .parse::<uuid::Uuid>()
        .ok()
        .and_then(|uid| graph.entities.iter().position(|&e| e == uid));
    let Some(ai) = a_idx else {
        return serde_json::Value::Null;
    };

    match gf.name.as_str() {
        "triangles" => serde_json::json!(crate::analysis::topology::triangles(graph, ai)),
        "clustering" => {
            serde_json::json!(crate::analysis::topology::clustering_coefficient(graph, ai))
        }
        _ => {
            // Two-arg functions: resolve second entity
            let eid2 = val2.and_then(|v| v.get("id").and_then(|id| id.as_str()));
            let Some(eid2) = eid2 else {
                return serde_json::Value::Null;
            };
            let b_idx = eid2
                .parse::<uuid::Uuid>()
                .ok()
                .and_then(|uid| graph.entities.iter().position(|&e| e == uid));
            let Some(bi) = b_idx else {
                return serde_json::Value::Null;
            };

            match gf.name.as_str() {
                "common_neighbors" => serde_json::json!(
                    crate::analysis::link_prediction::common_neighbors(graph, ai, bi)
                ),
                "adamic_adar" => {
                    serde_json::json!(crate::analysis::link_prediction::adamic_adar(graph, ai, bi))
                }
                "preferential_attachment" => serde_json::json!(
                    crate::analysis::link_prediction::preferential_attachment(graph, ai, bi)
                ),
                "resource_allocation" => serde_json::json!(
                    crate::analysis::link_prediction::resource_allocation(graph, ai, bi)
                ),
                "jaccard" => serde_json::json!(crate::analysis::similarity::jaccard(graph, ai, bi)),
                "overlap" => serde_json::json!(crate::analysis::similarity::overlap(graph, ai, bi)),
                _ => serde_json::Value::Null,
            }
        }
    }
}

fn evaluate_condition(
    field_val: &serde_json::Value,
    op: &CompareOp,
    query_val: &QueryValue,
) -> bool {
    match op {
        CompareOp::Eq => values_equal(field_val, query_val),
        CompareOp::Ne => !values_equal(field_val, query_val),
        CompareOp::Gt => compare_values(field_val, query_val) == Some(std::cmp::Ordering::Greater),
        CompareOp::Lt => compare_values(field_val, query_val) == Some(std::cmp::Ordering::Less),
        CompareOp::Gte => {
            matches!(
                compare_values(field_val, query_val),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            )
        }
        CompareOp::Lte => {
            matches!(
                compare_values(field_val, query_val),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )
        }
        CompareOp::Contains => {
            if let (Some(s), QueryValue::String(q)) = (field_val.as_str(), query_val) {
                s.contains(q.as_str())
            } else {
                false
            }
        }
        CompareOp::In => false, // Stub for Phase 0
    }
}

fn values_equal(field_val: &serde_json::Value, query_val: &QueryValue) -> bool {
    match query_val {
        QueryValue::String(s) => field_val.as_str() == Some(s.as_str()),
        QueryValue::Integer(i) => field_val.as_i64() == Some(*i),
        QueryValue::Float(f) => field_val.as_f64() == Some(*f),
        QueryValue::Boolean(b) => field_val.as_bool() == Some(*b),
        QueryValue::Null => field_val.is_null(),
    }
}

fn compare_values(
    field_val: &serde_json::Value,
    query_val: &QueryValue,
) -> Option<std::cmp::Ordering> {
    match query_val {
        QueryValue::Integer(i) => field_val.as_i64().map(|fv| fv.cmp(i)),
        QueryValue::Float(f) => field_val.as_f64().and_then(|fv| fv.partial_cmp(f)),
        QueryValue::String(s) => field_val.as_str().map(|fv| fv.cmp(s.as_str())),
        _ => None,
    }
}

/// Execute aggregate functions over grouped bindings.
fn execute_aggregate(
    bindings: &HashMap<String, Vec<serde_json::Value>>,
    group_by: &[String],
    aggregates: &[(AggregateFunc, String)],
) -> Result<Vec<ResultRow>> {
    // Find the primary binding to aggregate over
    let primary_binding = if let Some(first_group) = group_by.first() {
        first_group.split('.').next().unwrap_or("_")
    } else if let Some((_, first_agg)) = aggregates.first() {
        if first_agg == "*" {
            bindings.keys().next().map(|s| s.as_str()).unwrap_or("_")
        } else {
            first_agg.split('.').next().unwrap_or("_")
        }
    } else {
        return Ok(vec![]);
    };

    let values = bindings.get(primary_binding).cloned().unwrap_or_default();

    if group_by.is_empty() {
        // No GROUP BY: single-group aggregation over all values
        let mut row = ResultRow::new();
        for (func, field) in aggregates {
            let key = format!("{}({})", agg_func_name(func), field);
            row.insert(
                key,
                compute_aggregate(func, field, &values, primary_binding),
            );
        }
        return Ok(vec![row]);
    }

    // Group values by group_by fields using BTreeMap for O(n log n) grouping
    let mut groups: BTreeMap<String, (Vec<serde_json::Value>, Vec<serde_json::Value>)> =
        BTreeMap::new();

    for val in &values {
        let group_key_vals: Vec<serde_json::Value> = group_by
            .iter()
            .map(|f| {
                let parts: Vec<&str> = f.splitn(2, '.').collect();
                if parts.len() == 2 {
                    resolve_field(val, parts[1]).clone()
                } else {
                    serde_json::Value::Null
                }
            })
            .collect();

        let key_str = serde_json::to_string(&group_key_vals).unwrap_or_default();
        let entry = groups
            .entry(key_str)
            .or_insert_with(|| (group_key_vals, Vec::new()));
        entry.1.push(val.clone());
    }

    // Build result rows
    let mut rows = Vec::new();
    for (group_key, group_values) in groups.values() {
        let mut row = ResultRow::new();
        for (i, field) in group_by.iter().enumerate() {
            row.insert(field.clone(), group_key[i].clone());
        }
        for (func, field) in aggregates {
            let key = format!("{}({})", agg_func_name(func), field);
            row.insert(
                key,
                compute_aggregate(func, field, group_values, primary_binding),
            );
        }
        rows.push(row);
    }

    Ok(rows)
}

fn agg_func_name(func: &AggregateFunc) -> &'static str {
    match func {
        AggregateFunc::Count => "COUNT",
        AggregateFunc::Sum => "SUM",
        AggregateFunc::Avg => "AVG",
        AggregateFunc::Min => "MIN",
        AggregateFunc::Max => "MAX",
    }
}

fn compute_aggregate(
    func: &AggregateFunc,
    field: &str,
    values: &[serde_json::Value],
    primary_binding: &str,
) -> serde_json::Value {
    match func {
        AggregateFunc::Count => {
            if field == "*" {
                serde_json::json!(values.len())
            } else {
                // Count non-null values of the field
                let field_path = extract_field_path(field, primary_binding);
                let count = values
                    .iter()
                    .filter(|v| !resolve_field(v, field_path).is_null())
                    .count();
                serde_json::json!(count)
            }
        }
        AggregateFunc::Sum => {
            let field_path = extract_field_path(field, primary_binding);
            let sum: f64 = values
                .iter()
                .filter_map(|v| resolve_field(v, field_path).as_f64())
                .sum();
            serde_json::json!(sum)
        }
        AggregateFunc::Avg => {
            let field_path = extract_field_path(field, primary_binding);
            let nums: Vec<f64> = values
                .iter()
                .filter_map(|v| resolve_field(v, field_path).as_f64())
                .collect();
            if nums.is_empty() {
                serde_json::Value::Null
            } else {
                let avg = nums.iter().sum::<f64>() / nums.len() as f64;
                serde_json::json!(avg)
            }
        }
        AggregateFunc::Min => {
            let field_path = extract_field_path(field, primary_binding);
            values
                .iter()
                .filter_map(|v| {
                    let fv = resolve_field(v, field_path);
                    fv.as_f64()
                })
                .fold(None, |acc: Option<f64>, x| {
                    Some(acc.map_or(x, |a| a.min(x)))
                })
                .map(|v| serde_json::json!(v))
                .unwrap_or(serde_json::Value::Null)
        }
        AggregateFunc::Max => {
            let field_path = extract_field_path(field, primary_binding);
            values
                .iter()
                .filter_map(|v| {
                    let fv = resolve_field(v, field_path);
                    fv.as_f64()
                })
                .fold(None, |acc: Option<f64>, x| {
                    Some(acc.map_or(x, |a| a.max(x)))
                })
                .map(|v| serde_json::json!(v))
                .unwrap_or(serde_json::Value::Null)
        }
    }
}

fn extract_field_path<'a>(field: &'a str, _primary_binding: &str) -> &'a str {
    let parts: Vec<&str> = field.splitn(2, '.').collect();
    if parts.len() == 2 {
        parts[1]
    } else {
        field
    }
}

/// Emit one field from `val` into `row`. Handles the `binding.field` projection
/// (including virtual `an.*` analysis properties) and the plain `binding` case.
fn emit_field(
    row: &mut ResultRow,
    binding: &str,
    field: Option<&str>,
    val: &serde_json::Value,
    hypergraph: &Hypergraph,
    cache: &mut AnalysisCache,
) {
    if let Some(f) = field {
        let resolved = if let Some(metric) = f.strip_prefix("an.") {
            resolve_analysis_property(val, metric, hypergraph, cache)
        } else {
            resolve_field(val, f).clone()
        };
        row.insert(format!("{}.{}", binding, f), resolved);
    } else {
        row.insert(binding.to_string(), val.clone());
    }
}

fn project_results(
    bindings: &HashMap<String, Vec<serde_json::Value>>,
    edge_pairs: &HashMap<(String, String), Vec<(uuid::Uuid, uuid::Uuid)>>,
    return_clause: &ReturnClause,
    hypergraph: &Hypergraph,
) -> Result<Vec<ResultRow>> {
    let mut rows = Vec::new();
    let mut cache = AnalysisCache::new();

    // Determine which bindings we're returning
    let mut return_bindings: Vec<&str> = Vec::new();
    let mut field_paths: Vec<(&str, Option<&str>)> = Vec::new();

    // Collect graph functions for evaluation during row production
    let mut graph_funcs: Vec<&GraphFunc> = Vec::new();
    for expr in &return_clause.expressions {
        if let ReturnExpr::GraphFunction(gf) = expr {
            graph_funcs.push(gf);
        }
    }

    for expr in &return_clause.expressions {
        let field_str = match expr {
            ReturnExpr::Field(s) => s.as_str(),
            ReturnExpr::Aggregate { .. } => continue,
            ReturnExpr::GraphFunction(_) => continue, // handled during row production
        };
        if field_str == "*" {
            for key in bindings.keys() {
                return_bindings.push(key);
                field_paths.push((key, None));
            }
        } else if field_str.contains('.') {
            let parts: Vec<&str> = field_str.splitn(2, '.').collect();
            field_paths.push((parts[0], Some(parts[1])));
            if !return_bindings.contains(&parts[0]) {
                return_bindings.push(parts[0]);
            }
        } else {
            return_bindings.push(field_str);
            field_paths.push((field_str, None));
        }
    }

    // If only one binding, produce one row per item
    // Lazy co-graph cache for graph function evaluation
    let mut co_graph_cache: Option<crate::analysis::graph_projection::CoGraph> = None;

    if return_bindings.len() == 1 || bindings.len() == 1 {
        let primary_binding = return_bindings.first().copied().unwrap_or("_");
        let values = bindings.get(primary_binding).cloned().unwrap_or_default();

        for val in &values {
            let mut row = ResultRow::new();
            for (binding, field) in &field_paths {
                if *binding == primary_binding {
                    emit_field(&mut row, binding, *field, val, hypergraph, &mut cache);
                }
            }
            // Evaluate graph functions for single-binding rows
            for gf in &graph_funcs {
                let gf_val =
                    evaluate_graph_func_return(gf, val, None, hypergraph, &mut co_graph_cache);
                let key = format!("{}({})", gf.name, gf.args.join(", "));
                row.insert(key, gf_val);
            }
            rows.push(row);
        }
    } else {
        // Multiple bindings: build an N-way join using edge_pairs whenever
        // possible. For 2 bindings this collapses to a plain inner join; for
        // 3+ bindings (e.g. T1-07 co-participation:
        // `(e:Actor)-[:PARTICIPATES]->(s:Situation)<-[:PARTICIPATES]-(d:Actor)`)
        // we chain edge_pairs by walking from one joined binding to the next,
        // producing assignments that agree on every shared node (the `s`
        // situation in the co-participation case). If no connecting edge
        // exists between a binding and the joined set, we fall back to a
        // cross-product over the remaining binding — same legacy behavior as
        // before, just generalized to N.
        //
        // Sort binding keys deterministically so result order doesn't depend
        // on HashMap iteration.
        let mut binding_keys: Vec<String> = bindings.keys().cloned().collect();
        binding_keys.sort();

        if binding_keys.len() == 1 {
            // Defensive: if only one binding remains, emit one row per value.
            let key = &binding_keys[0];
            let values = bindings.get(key).cloned().unwrap_or_default();
            for val in &values {
                let mut row = ResultRow::new();
                for (binding, field) in &field_paths {
                    if *binding == key.as_str() {
                        emit_field(&mut row, binding, *field, val, hypergraph, &mut cache);
                    }
                }
                rows.push(row);
            }
        } else if !binding_keys.is_empty() {
            // Build per-binding id → value maps once.
            let values_by_binding: HashMap<String, Vec<serde_json::Value>> = binding_keys
                .iter()
                .map(|k| (k.clone(), bindings.get(k).cloned().unwrap_or_default()))
                .collect();
            let id_maps: HashMap<String, HashMap<uuid::Uuid, serde_json::Value>> =
                values_by_binding
                    .iter()
                    .map(|(k, vals)| {
                        let m: HashMap<uuid::Uuid, serde_json::Value> = vals
                            .iter()
                            .filter_map(|v| {
                                v.get("id")
                                    .and_then(|s| s.as_str())
                                    .and_then(|s| uuid::Uuid::parse_str(s).ok())
                                    .map(|u| (u, v.clone()))
                            })
                            .collect();
                        (k.clone(), m)
                    })
                    .collect();

            // Helper: look up pairs between two bindings in either direction.
            // Returns (pairs, reverse_flag) where reverse_flag=true means the
            // stored pairs are (b,a) not (a,b).
            let find_edge_pairs =
                |a: &str, b: &str| -> Option<(&Vec<(uuid::Uuid, uuid::Uuid)>, bool)> {
                    if let Some(p) = edge_pairs.get(&(a.to_string(), b.to_string())) {
                        Some((p, false))
                    } else {
                        edge_pairs
                            .get(&(b.to_string(), a.to_string()))
                            .map(|p| (p, true))
                    }
                };

            // Seed the join with the first binding alphabetically.
            //
            // `partials` is a list of partial assignments: each entry maps a
            // subset of binding names to a concrete row value. We grow it one
            // binding at a time.
            type Assignment = HashMap<String, serde_json::Value>;
            let seed_key = binding_keys[0].clone();
            let seed_vals = values_by_binding
                .get(&seed_key)
                .cloned()
                .unwrap_or_default();
            let mut partials: Vec<Assignment> = seed_vals
                .into_iter()
                .map(|v| {
                    let mut a = HashMap::new();
                    a.insert(seed_key.clone(), v);
                    a
                })
                .collect();

            // For the remaining bindings, add them one at a time. Prefer a
            // binding that is connected via edge_pairs to at least one
            // already-joined binding (any such binding works — we pick the
            // first in sorted order). If none of the remaining bindings is
            // connected, we fall back to cross-product for the one we pick.
            let mut remaining: Vec<String> = binding_keys[1..].to_vec();
            while !remaining.is_empty() {
                // Try to pick a remaining binding that's edge-connected to a
                // joined one.
                let joined_keys: std::collections::HashSet<String> = partials
                    .first()
                    .map(|a| a.keys().cloned().collect())
                    .unwrap_or_default();
                let mut chosen_idx: Option<usize> = None;
                let mut chosen_link: Option<(String, Vec<(uuid::Uuid, uuid::Uuid)>, bool)> = None;
                for (idx, cand) in remaining.iter().enumerate() {
                    for j in &joined_keys {
                        if let Some((pairs, reverse)) = find_edge_pairs(cand, j) {
                            chosen_idx = Some(idx);
                            chosen_link = Some((j.clone(), pairs.clone(), reverse));
                            break;
                        }
                    }
                    if chosen_idx.is_some() {
                        break;
                    }
                }
                let (next_key, link) = match chosen_idx {
                    Some(i) => (remaining.remove(i), chosen_link),
                    None => (remaining.remove(0), None),
                };
                let next_vals = values_by_binding
                    .get(&next_key)
                    .cloned()
                    .unwrap_or_default();
                let next_id_map = id_maps.get(&next_key).cloned().unwrap_or_default();

                let mut new_partials: Vec<Assignment> = Vec::new();
                if let Some((other_binding, pairs, reverse)) = link {
                    // Build: other_id -> Vec<next_value>
                    // pairs are stored as (a,b); if reverse, swap so the first
                    // coord is `next_key` and the second is `other_binding`.
                    // Equivalently: if reverse=false, key (a,b) means a=cand=next_key,
                    //   b=j=other_binding. If reverse=true, key was stored as
                    //   (other_binding, next_key) so pairs are (other_id, next_id).
                    let mut by_other: HashMap<uuid::Uuid, Vec<uuid::Uuid>> = HashMap::new();
                    for (a, b) in &pairs {
                        let (next_id, other_id) = if reverse { (b, a) } else { (a, b) };
                        by_other.entry(*other_id).or_default().push(*next_id);
                    }
                    for partial in &partials {
                        let other_val = match partial.get(&other_binding) {
                            Some(v) => v,
                            None => continue,
                        };
                        let other_id = match other_val
                            .get("id")
                            .and_then(|s| s.as_str())
                            .and_then(|s| uuid::Uuid::parse_str(s).ok())
                        {
                            Some(u) => u,
                            None => continue,
                        };
                        let next_ids = match by_other.get(&other_id) {
                            Some(v) => v,
                            None => continue,
                        };
                        for nid in next_ids {
                            if let Some(nval) = next_id_map.get(nid) {
                                let mut a2 = partial.clone();
                                a2.insert(next_key.clone(), nval.clone());
                                new_partials.push(a2);
                            }
                        }
                    }
                } else {
                    // No connecting edge → cross-product.
                    for partial in &partials {
                        for nval in &next_vals {
                            let mut a2 = partial.clone();
                            a2.insert(next_key.clone(), nval.clone());
                            new_partials.push(a2);
                        }
                    }
                }
                partials = new_partials;
                // Early exit if the join has no rows.
                if partials.is_empty() {
                    break;
                }
            }

            // Materialize rows from partial assignments.
            for assignment in &partials {
                let mut row = ResultRow::new();
                for (binding, field) in &field_paths {
                    if let Some(val) = assignment.get(*binding) {
                        emit_field(&mut row, binding, *field, val, hypergraph, &mut cache);
                    }
                }
                rows.push(row);
            }
        }
    }

    // Apply ORDER BY
    if let Some(ob) = &return_clause.order_by {
        let field = ob.field.clone();
        let asc = ob.ascending;

        // Resolve an ordering value from a result row. Tries:
        // 1. Direct key lookup (e.g. "e.pagerank" if projected as "e.pagerank")
        // 2. Dotted fallback: split "binding.field" and resolve the nested field.
        let lookup = |row: &ResultRow, key: &str| -> Option<serde_json::Value> {
            if let Some(v) = row.get(key) {
                return Some(v.clone());
            }
            let parts: Vec<&str> = key.splitn(2, '.').collect();
            if parts.len() == 2 {
                if let Some(v) = row.get(parts[0]) {
                    return Some(resolve_field(v, parts[1]).clone());
                }
            }
            None
        };

        rows.sort_by(|a, b| {
            let va = lookup(a, &field);
            let vb = lookup(b, &field);
            let cmp = match (va, vb) {
                (Some(a), Some(b)) => {
                    if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                    } else if let (Some(sa), Some(sb)) = (a.as_str(), b.as_str()) {
                        sa.cmp(sb)
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };
            if asc {
                cmp
            } else {
                cmp.reverse()
            }
        });
    }

    // Safety cap: prevent unbounded result sets from exhausting memory
    const MAX_RESULT_SIZE: usize = 10_000;
    if rows.len() > MAX_RESULT_SIZE {
        tracing::warn!(
            "Query result truncated from {} to {} rows",
            rows.len(),
            MAX_RESULT_SIZE
        );
        rows.truncate(MAX_RESULT_SIZE);
    }

    // Apply user-specified LIMIT (after safety cap)
    if let Some(limit) = return_clause.limit {
        rows.truncate(limit);
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parser::{parse_query, parse_statement, TensaStatement};
    use crate::query::planner::plan_query;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn setup() -> (Hypergraph, IntervalTree) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let tree = IntervalTree::new();
        (hg, tree)
    }

    fn create_actor(hg: &Hypergraph, name: &str, confidence: f32) -> Uuid {
        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap()
    }

    fn create_situation_entry(hg: &Hypergraph) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_execute_match_entity_by_type() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.8);

        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_execute_match_entity_by_property() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.8);

        let q = parse_query(r#"MATCH (e:Actor) WHERE e.name = "Alice" RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_where_confidence_filter() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        let q = parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 2); // Alice and Charlie
    }

    #[test]
    fn test_execute_match_participation_pattern() {
        let (hg, tree) = setup();
        let eid = create_actor(&hg, "Alice", 0.9);
        let sid = create_situation_entry(&hg);
        hg.add_participant(Participation {
            entity_id: eid,
            situation_id: sid,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let q = parse_query("MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_reverse_edge_parses_and_plans_as_forward() {
        // `(s:Situation) <-[:PARTICIPATES]- (e:Actor)` must plan as
        // `FollowEdge { from: e, to: s }` — the reverse-arrow swap normalizes
        // every edge into a forward join, so the executor stays ignorant of
        // edge direction.
        let q = parse_query("MATCH (s:Situation)<-[p:PARTICIPATES]-(e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let follow = plan
            .steps
            .iter()
            .find_map(|s| match s {
                PlanStep::FollowEdge {
                    from_binding,
                    to_binding,
                    rel_type,
                    ..
                } => Some((from_binding.clone(), to_binding.clone(), rel_type.clone())),
                _ => None,
            })
            .expect("reverse edge must produce a FollowEdge step");
        assert_eq!(follow.0, "e");
        assert_eq!(follow.1, "s");
        assert_eq!(follow.2, "PARTICIPATES");
    }

    #[test]
    fn test_reverse_edge_executes_participation_match() {
        // Minimal end-to-end: one actor in one situation; the reverse-arrow form
        // `(s) <-[:PARTICIPATES]- (e)` returns the same rows as the forward form.
        let (hg, tree) = setup();
        let eid = create_actor(&hg, "Alice", 0.9);
        let sid = create_situation_entry(&hg);
        hg.add_participant(Participation {
            entity_id: eid,
            situation_id: sid,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let forward = execute(
            &plan_query(
                &parse_query("MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s")
                    .unwrap(),
            )
            .unwrap(),
            &hg,
            &tree,
        )
        .unwrap();
        let reverse = execute(
            &plan_query(
                &parse_query("MATCH (s:Situation)<-[p:PARTICIPATES]-(e:Actor) RETURN e, s")
                    .unwrap(),
            )
            .unwrap(),
            &hg,
            &tree,
        )
        .unwrap();
        assert_eq!(forward.len(), 1);
        assert_eq!(reverse.len(), 1);
    }

    #[test]
    fn test_three_binding_co_participation_join() {
        // T1-07 pattern: enumerate every pair (e, d) of actors that share a
        // situation via
        // `(e:Actor)-[:PARTICIPATES]->(s:Situation)<-[:PARTICIPATES]-(d:Actor)`.
        // Project all three bindings and verify the join is keyed on the
        // shared `s`, not a cross-product. The WHERE grammar only supports
        // field-to-literal comparisons, so `e != d` is not filtered at parse
        // time — the executor itself enumerates all (e, d) ordered pairs
        // (including self-pairs), and downstream consumers filter e = d out.
        // The critical property is that we're joined on `s`, so Bob never
        // co-appears with Carol.
        let (hg, tree) = setup();
        let alice = create_actor(&hg, "Alice", 0.9);
        let bob = create_actor(&hg, "Bob", 0.9);
        let carol = create_actor(&hg, "Carol", 0.9);
        let sit1 = create_situation_entry(&hg);
        let sit2 = create_situation_entry(&hg);

        // sit1: Alice + Bob. sit2: Alice + Carol. Bob and Carol never share.
        for (entity, situation) in [(alice, sit1), (bob, sit1), (alice, sit2), (carol, sit2)] {
            hg.add_participant(Participation {
                entity_id: entity,
                situation_id: situation,
                role: Role::Protagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let q = parse_query(
            "MATCH (e:Actor)-[p1:PARTICIPATES]->(s:Situation)<-[p2:PARTICIPATES]-(d:Actor) \
             RETURN e, s, d",
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();

        // Expected (e, d) ordered pairs sharing some situation s:
        //   sit1 participants × sit1 participants = {Alice, Bob}² = 4 rows
        //   sit2 participants × sit2 participants = {Alice, Carol}² = 4 rows
        // Total = 8 rows. A pure cross-product would be 3 * 2 * 3 = 18 (and
        // would include Bob co-appearing with Carol, which is wrong).
        assert_eq!(
            results.len(),
            8,
            "expected 8 co-participation rows (join keyed on s), got {}",
            results.len()
        );

        // Verify Bob and Carol never co-appear in any row — this is the
        // defining property of the three-way join.
        let bob_id = bob.to_string();
        let carol_id = carol.to_string();
        for row in &results {
            let e = row.get("e").expect("row missing e binding");
            let d = row.get("d").expect("row missing d binding");
            let s = row.get("s").expect("row missing s binding");
            let eid = e.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let did = d.get("id").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                !((eid == bob_id && did == carol_id) || (eid == carol_id && did == bob_id)),
                "Bob and Carol must never co-appear — they share no situation"
            );
            assert!(s.get("id").is_some(), "s binding must be a Situation");
        }
    }

    #[test]
    fn test_execute_return_limit() {
        let (hg, tree) = setup();
        for i in 0..10 {
            create_actor(&hg, &format!("Actor{}", i), 0.5);
        }

        let q = parse_query("MATCH (e:Actor) RETURN e LIMIT 3").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_execute_empty_result() {
        let (hg, tree) = setup();
        let q = parse_query("MATCH (e:Actor) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_execute_return_projection() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);

        let q = parse_query("MATCH (e:Actor) RETURN e.name").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains_key("e.name"));
    }

    #[test]
    fn test_execute_situation_scan() {
        let (hg, tree) = setup();
        create_situation_entry(&hg);
        create_situation_entry(&hg);

        let q = parse_query("MATCH (s:Situation) RETURN s").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_execute_scene_level_scan() {
        let (hg, tree) = setup();
        create_situation_entry(&hg); // Creates Scene-level situations

        let q = parse_query("MATCH (s:Scene) RETURN s").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_where_maturity_filter() {
        let (hg, tree) = setup();
        let id1 = create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.8);
        hg.set_entity_maturity(&id1, MaturityLevel::Validated, "admin", None)
            .unwrap();

        // Filter by maturity = "Validated" (as string comparison on JSON)
        // Maturity is serialized as a string enum variant in JSON
        let q = parse_query(r#"MATCH (e:Actor) WHERE e.maturity = "Validated" RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_combined_filters() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.9);

        let q = parse_query(
            r#"MATCH (e:Actor) WHERE e.confidence > 0.5 AND e.name = "Alice" RETURN e"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_unknown_type_error() {
        let (hg, tree) = setup();
        let q = parse_query("MATCH (e:UnknownType) RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let result = execute(&plan, &hg, &tree);
        assert!(result.is_err());
    }

    // ─── DML Mutation Execution Tests ───────────────────────

    #[test]
    fn test_execute_create_narrative() {
        let (hg, _tree) = setup();
        let stmt =
            parse_statement(r#"CREATE NARRATIVE "hamlet" TITLE "Hamlet" GENRE "tragedy""#).unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0]["id"], "hamlet");
            assert_eq!(results[0]["status"], "created");
        }
    }

    #[test]
    fn test_execute_create_entity() {
        let (hg, _tree) = setup();
        let stmt = parse_statement(
            r#"CREATE (e:Actor {name: "Alice", age: 30}) IN NARRATIVE "test" CONFIDENCE 0.9"#,
        )
        .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0]["status"], "created");
            let id_str = results[0]["id"].as_str().unwrap();
            let uuid = Uuid::parse_str(id_str).unwrap();

            // Verify entity was created
            let entity = hg.get_entity(&uuid).unwrap();
            assert_eq!(entity.entity_type, EntityType::Actor);
            assert_eq!(entity.properties["name"], "Alice");
            assert_eq!(entity.properties["age"], 30);
            assert_eq!(entity.narrative_id, Some("test".to_string()));
            assert!((entity.confidence - 0.9).abs() < 0.01);
        }
    }

    #[test]
    fn test_execute_create_situation() {
        let (hg, _tree) = setup();
        let stmt = parse_statement(
            r#"CREATE SITUATION AT Scene CONTENT "He entered the room" CONFIDENCE 0.75"#,
        )
        .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results[0]["status"], "created");
            let id_str = results[0]["id"].as_str().unwrap();
            let uuid = Uuid::parse_str(id_str).unwrap();

            let sit = hg.get_situation(&uuid).unwrap();
            assert_eq!(sit.narrative_level, NarrativeLevel::Scene);
            assert_eq!(sit.raw_content.len(), 1);
            assert_eq!(sit.raw_content[0].content, "He entered the room");
            assert_eq!(sit.extraction_method, ExtractionMethod::HumanEntered);
        }
    }

    #[test]
    fn test_execute_delete_entity() {
        let (hg, _tree) = setup();
        let id = create_actor(&hg, "ToDelete", 0.5);

        let stmt = parse_statement(&format!(r#"DELETE ENTITY "{}""#, id)).unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results[0]["status"], "deleted");
        }

        // Verify entity is gone
        assert!(hg.get_entity(&id).is_err());
    }

    #[test]
    fn test_execute_update_entity() {
        let (hg, _tree) = setup();
        let id = create_actor(&hg, "Original", 0.5);

        let stmt = parse_statement(&format!(
            r#"UPDATE ENTITY "{}" SET name = "Updated", confidence = 0.99"#,
            id
        ))
        .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results[0]["status"], "updated");
        }

        let entity = hg.get_entity(&id).unwrap();
        assert_eq!(entity.properties["name"], "Updated");
        assert!((entity.confidence - 0.99).abs() < 0.01);
    }

    #[test]
    fn test_execute_add_participant() {
        let (hg, _tree) = setup();
        let eid = create_actor(&hg, "Alice", 0.9);
        let sid = create_situation_entry(&hg);

        let stmt = parse_statement(&format!(
            r#"ADD PARTICIPANT "{}" TO SITUATION "{}" ROLE Protagonist ACTION "enters""#,
            eid, sid
        ))
        .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results[0]["status"], "created");
        }

        let participants = hg.get_participants_for_situation(&sid).unwrap();
        assert_eq!(participants.len(), 1);
        assert_eq!(participants[0].role, Role::Protagonist);
        assert_eq!(participants[0].action, Some("enters".to_string()));
    }

    #[test]
    fn test_execute_remove_participant() {
        let (hg, _tree) = setup();
        let eid = create_actor(&hg, "Alice", 0.9);
        let sid = create_situation_entry(&hg);
        hg.add_participant(Participation {
            entity_id: eid,
            situation_id: sid,
            role: Role::Protagonist,
            info_set: None,
            action: None,
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let stmt = parse_statement(&format!(
            r#"REMOVE PARTICIPANT "{}" FROM SITUATION "{}""#,
            eid, sid
        ))
        .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            execute_mutation(&m, &hg).unwrap();
        }

        let participants = hg.get_participants_for_situation(&sid).unwrap();
        assert!(participants.is_empty());
    }

    #[test]
    fn test_execute_add_and_remove_cause() {
        let (hg, _tree) = setup();
        let sid1 = create_situation_entry(&hg);
        let sid2 = create_situation_entry(&hg);

        // Add cause
        let stmt = parse_statement(&format!(
            r#"ADD CAUSE FROM "{}" TO "{}" TYPE Sufficient STRENGTH 0.9 MECHANISM "direct""#,
            sid1, sid2
        ))
        .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            let results = execute_mutation(&m, &hg).unwrap();
            assert_eq!(results[0]["status"], "created");
        }

        // Verify causal link exists
        let effects = hg.get_consequences(&sid1).unwrap();
        assert_eq!(effects.len(), 1);

        // Remove cause
        let stmt =
            parse_statement(&format!(r#"REMOVE CAUSE FROM "{}" TO "{}""#, sid1, sid2)).unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            execute_mutation(&m, &hg).unwrap();
        }

        let effects = hg.get_consequences(&sid1).unwrap();
        assert!(effects.is_empty());
    }

    #[test]
    fn test_execute_create_then_query() {
        // End-to-end: create entity via DML, query via MATCH
        let (hg, tree) = setup();

        // Create entity via DML
        let stmt =
            parse_statement(r#"CREATE (e:Actor {name: "Bob", role: "hero"}) CONFIDENCE 0.85"#)
                .unwrap();
        if let TensaStatement::Mutation(m) = stmt {
            execute_mutation(&m, &hg).unwrap();
        }

        // Query it back via MATCH
        let q = parse_query(r#"MATCH (e:Actor) WHERE e.name = "Bob" RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    // ─── EXPLAIN Tests ──────────────────────────────────────────

    #[test]
    fn test_explain_match_returns_plan() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);

        let q = parse_query("EXPLAIN MATCH (e:Actor) RETURN e").unwrap();
        assert!(q.explain);
        let plan = plan_query(&q).unwrap();
        let results = execute_explained(&plan, &hg, &tree, q.explain).unwrap();
        assert_eq!(results.len(), 1);
        let plan_val = &results[0]["plan"];
        assert!(plan_val.is_object());
        assert_eq!(plan_val["class"], "Instant");
        assert!(plan_val["steps"].is_array());
    }

    #[test]
    fn test_explain_with_where_shows_filter_step() {
        let (hg, tree) = setup();
        let q = parse_query("EXPLAIN MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute_explained(&plan, &hg, &tree, q.explain).unwrap();
        let steps = results[0]["plan"]["steps"].as_array().unwrap();
        assert!(steps.iter().any(|s| s["FilterProperties"].is_object()));
    }

    #[test]
    fn test_explain_with_edge_shows_follow_edge() {
        let (hg, tree) = setup();
        let q = parse_query("EXPLAIN MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation) RETURN e, s")
            .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute_explained(&plan, &hg, &tree, q.explain).unwrap();
        let steps = results[0]["plan"]["steps"].as_array().unwrap();
        assert!(steps.iter().any(|s| s["FollowEdge"].is_object()));
    }

    #[test]
    fn test_explain_infer_returns_job_class() {
        let (hg, tree) = setup();
        let q = parse_query("EXPLAIN INFER CAUSES FOR s:Situation RETURN s").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute_explained(&plan, &hg, &tree, q.explain).unwrap();
        assert_eq!(results[0]["plan"]["class"], "Job");
    }

    // ─── OR Condition Tests ─────────────────────────────────────

    #[test]
    fn test_execute_or_returns_union() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        // Should return Alice (>0.8) OR Bob (<0.4)
        let q =
            parse_query("MATCH (e:Actor) WHERE e.confidence > 0.8 OR e.confidence < 0.4 RETURN e")
                .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_execute_and_or_combined() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        // (confidence > 0.8 AND name = "Alice") OR confidence < 0.4
        // Should match Alice (both AND conditions) and Bob (< 0.4)
        let q = parse_query(
            r#"MATCH (e:Actor) WHERE e.confidence > 0.8 AND e.name = "Alice" OR e.confidence < 0.4 RETURN e"#,
        ).unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_execute_parenthesized_or() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        // (confidence > 0.8 OR confidence < 0.4) AND name = "Alice"
        // Only Alice matches (> 0.8) AND name "Alice"
        let q = parse_query(
            r#"MATCH (e:Actor) WHERE (e.confidence > 0.8 OR e.confidence < 0.4) AND e.name = "Alice" RETURN e"#,
        ).unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_or_no_matches() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.5);

        let q =
            parse_query("MATCH (e:Actor) WHERE e.confidence > 0.9 OR e.confidence < 0.1 RETURN e")
                .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_execute_single_condition_backward_compat() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);

        let q = parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
    }

    // ─── Aggregation Tests ──────────────────────────────────────

    #[test]
    fn test_execute_count_star() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        let q = parse_query("MATCH (e:Actor) RETURN COUNT(*)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["COUNT(*)"], 3);
    }

    #[test]
    fn test_execute_count_field() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);

        let q = parse_query("MATCH (e:Actor) RETURN COUNT(e.name)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["COUNT(e.name)"], 2);
    }

    #[test]
    fn test_execute_sum() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);

        let q = parse_query("MATCH (e:Actor) RETURN SUM(e.confidence)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        let sum = results[0]["SUM(e.confidence)"].as_f64().unwrap();
        assert!((sum - 1.2).abs() < 0.01);
    }

    #[test]
    fn test_execute_avg() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);

        let q = parse_query("MATCH (e:Actor) RETURN AVG(e.confidence)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        let avg = results[0]["AVG(e.confidence)"].as_f64().unwrap();
        assert!((avg - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_execute_min_max() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        let q = parse_query("MATCH (e:Actor) RETURN MIN(e.confidence), MAX(e.confidence)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        let min = results[0]["MIN(e.confidence)"].as_f64().unwrap();
        let max = results[0]["MAX(e.confidence)"].as_f64().unwrap();
        assert!((min - 0.3).abs() < 0.01);
        assert!((max - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_execute_group_by_type() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);

        // Create a location entity
        let loc = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Location,
            properties: serde_json::json!({"name": "Moscow"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(loc).unwrap();

        // Query: GROUP BY entity_type — should return 2 groups (Actor, Location)
        // We need to scan ALL entities, so use a broad type
        // Actually, ScanByType only returns one type. Let's test with Actor only.
        let q =
            parse_query("MATCH (e:Actor) GROUP BY e.entity_type RETURN e.entity_type, COUNT(*)")
                .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        // All actors are same type, so 1 group
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["COUNT(*)"], 2);
    }

    #[test]
    fn test_execute_agg_with_where() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);
        create_actor(&hg, "Bob", 0.3);
        create_actor(&hg, "Charlie", 0.7);

        let q = parse_query("MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN COUNT(*)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results[0]["COUNT(*)"], 2); // Alice and Charlie
    }

    #[test]
    fn test_execute_agg_empty_results() {
        let (hg, tree) = setup();

        let q = parse_query("MATCH (e:Actor) RETURN COUNT(*)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["COUNT(*)"], 0);
    }

    #[test]
    fn test_execute_agg_avg_empty() {
        let (hg, tree) = setup();

        let q = parse_query("MATCH (e:Actor) RETURN AVG(e.confidence)").unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0]["AVG(e.confidence)"].is_null());
    }

    // ─── Temporal Filtering Tests (F-QE1) ───────────────────────

    fn create_situation_at(hg: &Hypergraph, start: &str, end: &str) -> Uuid {
        let start_dt = chrono::DateTime::parse_from_rfc3339(start)
            .unwrap()
            .with_timezone(&Utc);
        let end_dt = chrono::DateTime::parse_from_rfc3339(end)
            .unwrap()
            .with_timezone(&Utc);
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(start_dt),
                end: Some(end_dt),
                granularity: TimeGranularity::Exact,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_filter_temporal_before() {
        let (hg, tree) = setup();
        // Situation in January 2024
        create_situation_at(&hg, "2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z");
        // Situation in June 2025
        create_situation_at(&hg, "2025-06-01T00:00:00Z", "2025-06-30T23:59:59Z");

        let q = parse_query(
            r#"MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01T00:00:00Z" RETURN s"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1); // only January 2024
    }

    #[test]
    fn test_filter_temporal_after() {
        let (hg, tree) = setup();
        create_situation_at(&hg, "2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z");
        create_situation_at(&hg, "2025-06-01T00:00:00Z", "2025-06-30T23:59:59Z");

        let q = parse_query(
            r#"MATCH (s:Situation) AT s.temporal AFTER "2025-01-01T00:00:00Z" RETURN s"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1); // only June 2025
    }

    #[test]
    fn test_filter_temporal_no_match() {
        let (hg, tree) = setup();
        create_situation_at(&hg, "2024-01-01T00:00:00Z", "2024-01-31T23:59:59Z");

        let q = parse_query(
            r#"MATCH (s:Situation) AT s.temporal AFTER "2030-01-01T00:00:00Z" RETURN s"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_filter_temporal_missing_interval() {
        let (hg, tree) = setup();
        // Create a situation with no temporal data (start/end = None)
        create_situation_entry(&hg);

        let q = parse_query(
            r#"MATCH (s:Situation) AT s.temporal BEFORE "2030-01-01T00:00:00Z" RETURN s"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        // Situations without temporal should be excluded by the filter
        // Note: create_situation_entry sets start/end to Utc::now() which IS before 2030
        // So this should still match
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_temporal_contains() {
        let (hg, tree) = setup();
        // A wide interval containing the reference point
        create_situation_at(&hg, "2024-01-01T00:00:00Z", "2025-12-31T23:59:59Z");
        // A narrow interval NOT containing the reference point
        create_situation_at(&hg, "2024-01-01T00:00:00Z", "2024-06-01T00:00:00Z");

        let q = parse_query(
            r#"MATCH (s:Situation) AT s.temporal CONTAINS "2025-06-15T00:00:00Z" RETURN s"#,
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1); // only the wide interval contains the point
    }

    // ─── Vector Near Tests (F-QE2) ──────────────────────────────

    #[test]
    fn test_vector_near_no_index_error() {
        let (hg, tree) = setup();
        create_actor(&hg, "Alice", 0.9);

        let q = parse_query(r#"MATCH (e:Actor) NEAR(e, "test", 5) RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        // Without vector_index/embedder, should error
        let result = execute_full(&plan, &hg, &tree, false, None, None);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("vector index"));
    }

    #[test]
    fn test_vector_near_basic() {
        use crate::ingestion::embed::HashEmbedding;
        use crate::ingestion::vector::VectorIndex;

        let (hg, tree) = setup();
        let id1 = create_actor(&hg, "Alice", 0.9);
        let id2 = create_actor(&hg, "Bob", 0.8);
        let _id3 = create_actor(&hg, "Charlie", 0.7);

        let embedder = HashEmbedding::new(64);
        let mut vi = VectorIndex::new(64);

        // Add embeddings for Alice and Bob
        let emb1 = embedder.embed_text("Alice hero").unwrap();
        vi.add(id1, &emb1).unwrap();
        let emb2 = embedder.embed_text("Bob villain").unwrap();
        vi.add(id2, &emb2).unwrap();

        let q = parse_query(r#"MATCH (e:Actor) NEAR(e, "Alice hero", 1) RETURN e"#).unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute_full(&plan, &hg, &tree, false, Some(&vi), Some(&embedder)).unwrap();
        // Should return only 1 result (k=1) and it should be Alice (exact match embedding)
        assert_eq!(results.len(), 1);
    }

    // ─── Spatial Filtering Tests (F-QE3) ────────────────────────

    fn create_spatial_situation(hg: &Hypergraph, lat: f64, lon: f64) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: Some(SpatialAnchor {
                latitude: Some(lat),
                longitude: Some(lon),
                precision: SpatialPrecision::Exact,
                location_entity: None,
                location_name: None,
                description: None,
            }),
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Spatial test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_spatial_filter_within() {
        let (hg, tree) = setup();
        // New York City area
        create_spatial_situation(&hg, 40.7128, -74.0060);
        // London
        create_spatial_situation(&hg, 51.5074, -0.1278);

        let q = parse_query(
            "MATCH (s:Situation) SPATIAL s.spatial WITHIN 100.0 KM OF (40.7, -74.0) RETURN s",
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 1); // only NYC within 100km
    }

    #[test]
    fn test_spatial_filter_outside() {
        let (hg, tree) = setup();
        // London
        create_spatial_situation(&hg, 51.5074, -0.1278);
        // Paris
        create_spatial_situation(&hg, 48.8566, 2.3522);

        let q = parse_query(
            "MATCH (s:Situation) SPATIAL s.spatial WITHIN 10.0 KM OF (40.7, -74.0) RETURN s",
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 0); // neither within 10km of NYC
    }

    #[test]
    fn test_spatial_missing_coords() {
        let (hg, tree) = setup();
        // Situation without spatial data
        create_situation_entry(&hg);

        let q = parse_query(
            "MATCH (s:Situation) SPATIAL s.spatial WITHIN 1000.0 KM OF (0.0, 0.0) RETURN s",
        )
        .unwrap();
        let plan = plan_query(&q).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        assert_eq!(results.len(), 0); // no spatial data = excluded
    }

    #[test]
    fn test_haversine_km_known_distance() {
        // NYC to London is approximately 5,570 km
        let dist = haversine_km(40.7128, -74.0060, 51.5074, -0.1278);
        assert!(dist > 5500.0 && dist < 5700.0);
    }

    // ─── PATH Query Tests ──────────────────────────────────────

    fn make_path_situation(hg: &Hypergraph) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("path test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_path_shortest_linear_chain() {
        let (hg, _tree) = setup();
        // Create chain: s1 -> s2 -> s3
        let s1 = make_path_situation(&hg);
        let s2 = make_path_situation(&hg);
        let s3 = make_path_situation(&hg);
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            mechanism: None,
            strength: 1.0,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            mechanism: None,
            strength: 1.0,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        // BFS shortest from s1 to s3
        let path = bfs_shortest_path(&hg, &s1, &s3, 10).unwrap();
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3); // s1, s2, s3
        assert_eq!(path[0], s1);
        assert_eq!(path[1], s2);
        assert_eq!(path[2], s3);
    }

    #[test]
    fn test_path_all_diamond_graph() {
        let (hg, _tree) = setup();
        // Create diamond: s1 -> s2 -> s4, s1 -> s3 -> s4
        let s1 = make_path_situation(&hg);
        let s2 = make_path_situation(&hg);
        let s3 = make_path_situation(&hg);
        let s4 = make_path_situation(&hg);
        for (from, to) in [(s1, s2), (s1, s3), (s2, s4), (s3, s4)] {
            hg.add_causal_link(CausalLink {
                from_situation: from,
                to_situation: to,
                mechanism: None,
                strength: 1.0,
                causal_type: CausalType::Contributing,
                maturity: MaturityLevel::Candidate,
            })
            .unwrap();
        }

        let mut visited = HashSet::new();
        visited.insert(s1);
        let mut results = Vec::new();
        dfs_all_paths(
            &hg,
            &s1,
            &s4,
            &mut visited,
            &mut vec![s1],
            1,
            10,
            &mut results,
            100,
        )
        .unwrap();
        assert_eq!(results.len(), 2); // two paths: s1->s2->s4 and s1->s3->s4
    }

    #[test]
    fn test_path_depth_limit_respected() {
        let (hg, _tree) = setup();
        // Chain of length 5: s1 -> s2 -> s3 -> s4 -> s5
        let sids: Vec<Uuid> = (0..5).map(|_| make_path_situation(&hg)).collect();
        for i in 0..4 {
            hg.add_causal_link(CausalLink {
                from_situation: sids[i],
                to_situation: sids[i + 1],
                mechanism: None,
                strength: 1.0,
                causal_type: CausalType::Contributing,
                maturity: MaturityLevel::Candidate,
            })
            .unwrap();
        }

        // max_depth=2 can't reach s5 from s1 (needs 4 hops)
        let path = bfs_shortest_path(&hg, &sids[0], &sids[4], 2).unwrap();
        assert!(path.is_none());

        // max_depth=4 can reach
        let path = bfs_shortest_path(&hg, &sids[0], &sids[4], 4).unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap().len(), 5);
    }

    #[test]
    fn test_path_unreachable_returns_empty() {
        let (hg, _tree) = setup();
        let s1 = make_path_situation(&hg);
        let s2 = make_path_situation(&hg);
        // No causal link between them
        let path = bfs_shortest_path(&hg, &s1, &s2, 10).unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_path_skips_soft_deleted() {
        let (hg, _tree) = setup();
        // Chain: s1 -> s2 -> s3, but s2 is soft-deleted
        let s1 = make_path_situation(&hg);
        let s2 = make_path_situation(&hg);
        let s3 = make_path_situation(&hg);
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            mechanism: None,
            strength: 1.0,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            mechanism: None,
            strength: 1.0,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        // Soft-delete s2
        hg.delete_situation(&s2).unwrap();

        // Path should not be found (s2 is skipped)
        let path = bfs_shortest_path(&hg, &s1, &s3, 10).unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_path_explain_returns_plan() {
        let (hg, tree) = setup();
        let q = parse_query(r#"EXPLAIN MATCH PATH SHORTEST (s1) -[:CAUSES*1..5]-> (s2) RETURN s1"#)
            .unwrap();
        assert!(q.explain);
        let plan = plan_query(&q).unwrap();
        let results = execute_explained(&plan, &hg, &tree, true).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains_key("plan"));
        let plan_json = &results[0]["plan"];
        let steps = plan_json.get("steps").unwrap().as_array().unwrap();
        let step0 = &steps[0];
        assert!(step0.get("FindPath").is_some());
    }

    // ─── Path restrictor tests ────────────────────────────

    #[test]
    fn test_path_restrictor_walk_allows_all() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        // Same node repeated — Walk allows this
        let path = vec![a, b, a];
        assert!(path_satisfies_restrictor(&path, &PathRestrictor::Walk));
    }

    #[test]
    fn test_path_restrictor_acyclic_rejects_repeated_node() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let path = vec![a, b, a]; // a repeats
        assert!(!path_satisfies_restrictor(&path, &PathRestrictor::Acyclic));
    }

    #[test]
    fn test_path_restrictor_acyclic_accepts_unique_nodes() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        let path = vec![a, b, c];
        assert!(path_satisfies_restrictor(&path, &PathRestrictor::Acyclic));
    }

    #[test]
    fn test_path_restrictor_trail_rejects_repeated_edge() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        // Edge a->b appears twice
        let path = vec![a, b, a, b];
        assert!(!path_satisfies_restrictor(&path, &PathRestrictor::Trail));
    }

    #[test]
    fn test_path_restrictor_trail_accepts_unique_edges() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        // Edges: a->b, b->c — all unique
        let path = vec![a, b, c];
        assert!(path_satisfies_restrictor(&path, &PathRestrictor::Trail));
    }

    #[test]
    fn test_path_restrictor_simple_allows_start_equals_end() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        // a -> b -> c -> a (start=end, simple cycle)
        let path = vec![a, b, c, a];
        assert!(path_satisfies_restrictor(&path, &PathRestrictor::Simple));
    }

    #[test]
    fn test_path_restrictor_simple_rejects_interior_repeat() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        // a -> b -> b -> c (b repeats in interior)
        let path = vec![a, b, b, c];
        assert!(!path_satisfies_restrictor(&path, &PathRestrictor::Simple));
    }

    // ─── Virtual Property Resolver (e.an.*) Tests ──────────

    #[test]
    fn test_an_field_resolver_pagerank() {
        let (hg, _tree) = setup();
        let nid = "an-pr-test";
        let entity = crate::types::Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Alice"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some(nid.to_string()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let eid = hg.create_entity(entity).unwrap();

        // Store a fake pagerank result at the expected KV prefix
        let key = format!("an/pr/{}/{}", nid, eid);
        let val = serde_json::to_vec(&serde_json::json!(0.42)).unwrap();
        hg.store().put(key.as_bytes(), &val).unwrap();

        // Serialize entity for resolve_analysis_property
        let entity_json = serde_json::to_value(hg.get_entity(&eid).unwrap()).unwrap();
        let mut cache = AnalysisCache::new();
        let result = resolve_analysis_property(&entity_json, "pagerank", &hg, &mut cache);
        assert_eq!(result, serde_json::json!(0.42));
    }

    #[test]
    fn test_an_field_resolver_missing_returns_null() {
        let (hg, _tree) = setup();
        let entity_json = serde_json::json!({
            "id": Uuid::now_v7().to_string(),
            "narrative_id": "nonexistent"
        });
        // No pagerank computed → should return null
        let mut cache = AnalysisCache::new();
        let result = resolve_analysis_property(&entity_json, "pagerank", &hg, &mut cache);
        assert!(result.is_null());
    }

    #[test]
    fn test_an_field_resolver_with_where() {
        let (hg, tree) = setup();
        let nid = "an-where-test";

        // Create two entities
        let mut eids = Vec::new();
        for name in &["High", "Low"] {
            let entity = crate::types::Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": name}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some(nid.to_string()),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            eids.push(hg.create_entity(entity).unwrap());
        }

        // Store centrality results — High has betweenness 0.9, Low has 0.1
        for (i, &eid) in eids.iter().enumerate() {
            let btwn = if i == 0 { 0.9 } else { 0.1 };
            let result = serde_json::json!({
                "entity_id": eid.to_string(),
                "betweenness": btwn,
                "closeness": 0.5,
                "degree": 0.5,
                "community_id": 0,
                "narrative_id": nid,
            });
            let key = format!("an/c/{}/{}", nid, eid);
            hg.store()
                .put(key.as_bytes(), &serde_json::to_vec(&result).unwrap())
                .unwrap();
        }

        // Query: WHERE e.an.betweenness > 0.5
        let query = format!(
            r#"MATCH (e:Actor) WHERE e.narrative_id = "{}" AND e.an.betweenness > 0.5 RETURN e"#,
            nid
        );
        let parsed = parse_query(&query).unwrap();
        let plan = plan_query(&parsed).unwrap();
        let results = execute(&plan, &hg, &tree).unwrap();
        // Only "High" should match
        assert_eq!(results.len(), 1);
    }

    // ─── EATH Phase 5 — Surrogate executor tests ────────────────

    /// Build a small narrative the calibrator can fit. Local copy of
    /// `engines_tests::build_basic_narrative` so this test module stays
    /// self-contained (cross-test-module imports get awkward).
    fn build_synth_basic_narrative(hg: &Hypergraph, narrative_id: &str) {
        let mut entity_ids = Vec::new();
        for i in 0..5 {
            let now = Utc::now();
            let e = Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("actor-{i}")}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 1.0,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some(narrative_id.into()),
                created_at: now,
                updated_at: now,
                deleted_at: None,
                transaction_time: None,
            };
            let id = e.id;
            hg.create_entity(e).unwrap();
            entity_ids.push(id);
        }
        use chrono::TimeZone;
        let epoch = chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        for t in 0..10i64 {
            let now = Utc::now();
            let start = epoch + chrono::Duration::seconds(t * 60);
            let s = Situation {
                id: Uuid::now_v7(),
                name: None,
                description: None,
                properties: serde_json::Value::Null,
                temporal: AllenInterval {
                    start: Some(start),
                    end: Some(start),
                    granularity: TimeGranularity::Exact,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![ContentBlock::text("phase 5 fixture")],
                narrative_level: NarrativeLevel::Event,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 1.0,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some(narrative_id.into()),
                source_chunk_id: None,
                source_span: None,
                synopsis: None,
                manuscript_order: None,
                parent_situation_id: None,
                label: None,
                status: None,
                keywords: vec![],
                created_at: now,
                updated_at: now,
                deleted_at: None,
                transaction_time: None,
            };
            let sid = s.id;
            hg.create_situation(s).unwrap();
            let group_size = 2 + (t as usize % 3);
            for k in 0..group_size {
                let eid = entity_ids[(t as usize + k) % entity_ids.len()];
                hg.add_participant(Participation {
                    entity_id: eid,
                    situation_id: sid,
                    role: Role::Bystander,
                    info_set: None,
                    action: None,
                    payoff: None,
                    seq: 0,
                })
                .unwrap();
            }
        }
    }

    #[test]
    fn test_executor_submits_calibration_returns_job_id() {
        // Set up a hypergraph backed by a memory store, share the store
        // with a JobQueue so plan + submit roundtrip uses the same KV
        // backing.
        let store: std::sync::Arc<dyn crate::store::KVStore> =
            std::sync::Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let tree = IntervalTree::new();
        let queue = crate::inference::jobs::JobQueue::new(store.clone());

        let stmt =
            parse_statement(r#"CALIBRATE SURROGATE USING 'eath' FOR "n-cal""#).unwrap();
        let plan = crate::query::planner::plan_statement(&stmt).unwrap();
        let results = execute_with_job_queue(&plan, &hg, &tree, &queue).unwrap();

        assert_eq!(results.len(), 1);
        let row = &results[0];
        assert_eq!(row["status"].as_str(), Some("submitted"));
        assert_eq!(row["model"].as_str(), Some("eath"));
        let job_id = row["job_id"].as_str().expect("job_id must be present");

        // Job persisted in the queue under that id.
        let job = queue.get_job(job_id).expect("job must be retrievable");
        match job.job_type {
            crate::types::InferenceJobType::SurrogateCalibration {
                narrative_id,
                model,
            } => {
                assert_eq!(narrative_id, "n-cal");
                assert_eq!(model, "eath");
            }
            other => panic!("expected SurrogateCalibration job type, got {other:?}"),
        }
    }

    #[test]
    fn test_executor_submits_generation_then_polls_to_completion() {
        // End-to-end flow: build a small narrative, calibrate via the
        // calibration engine directly (the worker pool isn't running in
        // this test), then submit a GENERATE NARRATIVE TensaQL statement
        // through the executor and verify the job lands and runs to
        // completion via direct engine execution.
        use crate::synth::engines::{SurrogateCalibrationEngine, SurrogateGenerationEngine};
        use crate::synth::registry::SurrogateRegistry;
        use crate::inference::InferenceEngine;

        let store: std::sync::Arc<dyn crate::store::KVStore> =
            std::sync::Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let tree = IntervalTree::new();
        let queue = crate::inference::jobs::JobQueue::new(store.clone());

        let nid = "src-narrative";
        build_synth_basic_narrative(&hg, nid);

        // Pre-calibrate so generation has params to load. Run the
        // calibration engine directly — we're not exercising calibration
        // in this test, just establishing preconditions for generation.
        let registry = std::sync::Arc::new(SurrogateRegistry::default());
        let cal_engine = SurrogateCalibrationEngine::new(registry.clone());
        let cal_job = crate::inference::types::InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: crate::types::InferenceJobType::SurrogateCalibration {
                narrative_id: nid.into(),
                model: "eath".into(),
            },
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({}),
            priority: crate::types::JobPriority::Normal,
            status: crate::types::JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        cal_engine.execute(&cal_job, &hg).expect("calibration must succeed");

        // Submit GENERATE through the TensaQL executor.
        let stmt = parse_statement(&format!(
            r#"GENERATE NARRATIVE "out-narrative" LIKE "{}" USING SURROGATE 'eath' SEED 42 STEPS 30"#,
            nid
        ))
        .unwrap();
        let plan = crate::query::planner::plan_statement(&stmt).unwrap();
        let results = execute_with_job_queue(&plan, &hg, &tree, &queue).unwrap();

        assert_eq!(results.len(), 1);
        let row = &results[0];
        assert_eq!(row["status"].as_str(), Some("submitted"));
        assert_eq!(row["model"].as_str(), Some("eath"));
        let job_id = row["job_id"].as_str().expect("job_id must be present").to_string();

        // Pull the job back out of the queue and run the generation
        // engine directly (no worker pool in tests). This proves the
        // executor built a properly-shaped InferenceJob.
        let job = queue.get_job(&job_id).expect("job must be persisted");
        match &job.job_type {
            crate::types::InferenceJobType::SurrogateGeneration {
                source_narrative_id,
                output_narrative_id,
                model,
                seed_override,
                ..
            } => {
                assert_eq!(source_narrative_id.as_deref(), Some(nid));
                assert_eq!(output_narrative_id, "out-narrative");
                assert_eq!(model, "eath");
                assert_eq!(*seed_override, Some(42));
            }
            other => panic!("expected SurrogateGeneration job type, got {other:?}"),
        }
        // STEPS clause flows through job.parameters.num_steps.
        assert_eq!(job.parameters["num_steps"].as_u64(), Some(30));

        let gen_engine = SurrogateGenerationEngine::new(registry);
        let result = gen_engine
            .execute(&job, &hg)
            .expect("generation must succeed end-to-end");
        assert_eq!(result.status, crate::types::JobStatus::Completed);
        assert_eq!(
            result.result["kind"].as_str(),
            Some("generation_done"),
            "generation engine must finish to a 'generation_done' result"
        );
        assert_eq!(
            result.result["output_narrative_id"].as_str(),
            Some("out-narrative")
        );
    }
}
