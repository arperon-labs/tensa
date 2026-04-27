//! `POST /fuzzy/rules`, `GET /fuzzy/rules/{nid}`,
//! `GET /fuzzy/rules/{nid}/{rule_id}`, `DELETE /fuzzy/rules/{nid}/{rule_id}`,
//! `POST /fuzzy/rules/{nid}/evaluate`.
//!
//! Fuzzy Sprint Phase 9 — Mamdani rule-system CRUD + single-entity
//! evaluation. Synchronous across the board: the heavy path is
//! `O(|rules| · |antecedent| + bins)` which fits comfortably in the
//! request path for any reasonable rule set.
//!
//! The `evaluate` handler accepts an optional `rule_ids` whitelist so
//! callers can exercise a slice of the rule catalogue without loading
//! every rule in the narrative. Unknown rule ids surface as
//! [`TensaError::NotFound`] → HTTP 404.
//!
//! Cites: [mamdani1975mamdani].

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::rules::{
    build_rule, delete_rule, evaluate_rules_against_entity, list_rules, load_rule_fn, save_rule,
    validate_rule, FuzzyCondition, FuzzyOutput, Rule, RuleSetEvaluation,
};
use crate::fuzzy::tnorm::TNormKind;

/// Body for `POST /fuzzy/rules`.
#[derive(Debug, Deserialize)]
pub struct CreateRuleBody {
    pub name: String,
    pub narrative_id: String,
    pub antecedent: Vec<FuzzyCondition>,
    pub consequent: FuzzyOutput,
    #[serde(default)]
    pub tnorm: Option<TNormKind>,
    #[serde(default)]
    pub enabled: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CreateRuleResponse {
    pub rule_id: Uuid,
    pub name: String,
    pub narrative_id: String,
}

/// POST /fuzzy/rules — create + persist a new Mamdani rule.
pub async fn create_rule(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateRuleBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput(
            "narrative_id is empty".into(),
        ))
        .into_response();
    }
    let mut rule = build_rule(
        body.name.clone(),
        body.narrative_id.clone(),
        body.antecedent,
        body.consequent,
    );
    if let Some(k) = body.tnorm {
        rule.tnorm = k;
    }
    if let Some(e) = body.enabled {
        rule.enabled = e;
    }
    if let Err(e) = validate_rule(&rule) {
        return error_response(e).into_response();
    }
    if let Err(e) = save_rule(state.hypergraph.store(), &body.narrative_id, &rule) {
        return error_response(e).into_response();
    }
    json_ok(&CreateRuleResponse {
        rule_id: rule.id,
        name: rule.name.clone(),
        narrative_id: rule.narrative_id.clone(),
    })
    .into_response()
}

/// GET /fuzzy/rules/{narrative_id} — list every rule for a narrative.
pub async fn list_rules_for_narrative(
    State(state): State<Arc<AppState>>,
    Path(nid): Path<String>,
) -> impl IntoResponse {
    match list_rules(state.hypergraph.store(), &nid) {
        Ok(rules) => json_ok(&rules).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /fuzzy/rules/{narrative_id}/{rule_id} — load one rule.
pub async fn get_rule(
    State(state): State<Arc<AppState>>,
    Path((nid, rule_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_rule_fn(state.hypergraph.store(), &nid, &rule_id) {
        Ok(Some(r)) => json_ok(&r).into_response(),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "Mamdani rule {rule_id} not persisted in narrative '{nid}'"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /fuzzy/rules/{narrative_id}/{rule_id}.
pub async fn delete_rule_endpoint(
    State(state): State<Arc<AppState>>,
    Path((nid, rule_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    if let Err(e) = delete_rule(state.hypergraph.store(), &nid, &rule_id) {
        return error_response(e).into_response();
    }
    json_ok(&serde_json::json!({"deleted": true, "rule_id": rule_id})).into_response()
}

/// Body for `POST /fuzzy/rules/{narrative_id}/evaluate`.
#[derive(Debug, Deserialize)]
pub struct EvaluateRulesBody {
    pub entity_id: Uuid,
    #[serde(default)]
    pub rule_ids: Option<Vec<String>>,
    /// Optional aggregator applied to firing strengths (the descriptor
    /// row's `firing_aggregate` field surfaces the fold).
    #[serde(default)]
    pub firing_aggregator: Option<AggregatorKind>,
}

/// POST /fuzzy/rules/{narrative_id}/evaluate — evaluate the rule set
/// against a single entity.
pub async fn evaluate_rules_endpoint(
    State(state): State<Arc<AppState>>,
    Path(nid): Path<String>,
    Json(body): Json<EvaluateRulesBody>,
) -> impl IntoResponse {
    let rule_ids_slice: Option<&[String]> = body.rule_ids.as_deref();

    let mut eval: RuleSetEvaluation = match evaluate_rules_against_entity(
        &state.hypergraph,
        &nid,
        rule_ids_slice,
        &body.entity_id,
    ) {
        Ok(ev) => ev,
        Err(e) => return error_response(e).into_response(),
    };

    // If the caller asked for an aggregator, compute it over the
    // firing strengths of the returned rules.
    if let Some(kind) = body.firing_aggregator {
        let xs: Vec<f64> = eval
            .fired_rules
            .iter()
            .map(|f| f.firing_strength)
            .collect();
        let agg = crate::fuzzy::aggregation::aggregator_for(kind);
        match agg.aggregate(&xs) {
            Ok(v) => eval.firing_aggregate = Some(v),
            Err(e) => {
                tracing::warn!("firing aggregator failed ({e}); omitting from response");
            }
        }
    }

    json_ok(&eval).into_response()
}

/// Convenience — builder constructor used by tests so the rule-creation
/// body can be exercised directly without going through axum.
pub fn rule_from_body(body: &CreateRuleBody) -> Rule {
    let mut rule = build_rule(
        body.name.clone(),
        body.narrative_id.clone(),
        body.antecedent.clone(),
        body.consequent.clone(),
    );
    if let Some(k) = body.tnorm {
        rule.tnorm = k;
    }
    if let Some(e) = body.enabled {
        rule.enabled = e;
    }
    rule
}
