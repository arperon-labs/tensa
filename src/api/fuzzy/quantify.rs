//! `POST /fuzzy/quantify` and `GET /fuzzy/quantify/{nid}/{predicate_hash}`.
//!
//! Fuzzy Sprint Phase 6 — synchronous intermediate-quantifier evaluation
//! over a narrative's entity or situation domain. Cached at
//! `fz/quant/{narrative_id}/{predicate_hash}`.
//!
//! The predicate is carried as a minimal string spec (`"confidence>0.7"`,
//! `"maturity=Confirmed"`, …) and evaluated by the handler into a graded
//! `μ_P(e) ∈ [0,1]`. For Phase 6 we ship crisp predicates only — the
//! graded upgrade is a Phase 6.5 follow-up.
//!
//! Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].

use std::str::FromStr;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::{Result, TensaError};
use crate::fuzzy::quantifier::{
    evaluate, load_quantifier_result, predicate_hash, quantifier_from_name,
    save_quantifier_result, QuantifierResult,
};
use crate::types::{Entity, EntityType, MaturityLevel};

/// Parse a free-form maturity name into [`MaturityLevel`]. Returns `None`
/// on unknown strings.
fn maturity_from_name(s: &str) -> Option<MaturityLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "candidate" => Some(MaturityLevel::Candidate),
        "reviewed" => Some(MaturityLevel::Reviewed),
        "validated" | "confirmed" | "verified" => Some(MaturityLevel::Validated),
        "groundtruth" | "ground_truth" | "ground-truth" => Some(MaturityLevel::GroundTruth),
        _ => None,
    }
}

/// Body for `POST /fuzzy/quantify`.
#[derive(Debug, Deserialize)]
pub struct QuantifyBody {
    pub narrative_id: String,
    /// Canonical quantifier name (`"most"`, `"many"`, `"almost_all"`,
    /// `"few"`). Case- and dash/underscore-insensitive at the resolver.
    pub quantifier: String,
    /// Optional entity-type restriction. `None` → scan all entity types.
    #[serde(default)]
    pub entity_type: Option<String>,
    /// Minimal predicate spec evaluated as a crisp filter. Supported forms:
    /// * `"confidence>0.7"` / `"confidence>=0.5"` / `"confidence<0.2"` /
    ///   `"confidence<=0.3"`
    /// * `"maturity=Confirmed"` / `"maturity>=Verified"`
    /// * `""` (empty) — every entity contributes `μ_P = 1.0`.
    #[serde(default)]
    pub r#where: Option<String>,
    /// Optional result label (echoed back in the response).
    #[serde(default)]
    pub label: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct QuantifyResponse {
    pub value: f64,
    pub quantifier_name: String,
    pub predicate_hash: String,
    pub narrative_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub domain_size: usize,
    pub cardinality_ratio: f64,
    pub cache_hit: bool,
}

/// Parse a minimal crisp predicate spec into a closure over [`Entity`].
/// Empty spec → the always-true predicate.
fn build_entity_predicate(spec: &str) -> Result<Box<dyn Fn(&Entity) -> f64 + Send + Sync>> {
    let s = spec.trim();
    if s.is_empty() {
        return Ok(Box::new(|_e: &Entity| 1.0));
    }

    // Parse `<field> <op> <value>` with `>=|<=|!=|=|>|<`.
    let (field, op, rhs) = split_comparison(s)?;
    let field = field.trim().to_string();
    let op = op.to_string();
    let rhs = rhs.trim().to_string();

    Ok(Box::new(move |e: &Entity| -> f64 {
        match field.as_str() {
            "confidence" => {
                let threshold: f64 = rhs.parse().unwrap_or(0.0);
                mu_bool(compare_f64(&op, e.confidence as f64, threshold))
            }
            "maturity" => match maturity_from_name(&rhs) {
                Some(level) => mu_bool(compare_ord(&op, &e.maturity, &level)),
                None => 0.0,
            },
            _ => evaluate_property_path(&e.properties, &field, &op, &rhs),
        }
    }))
}

/// Fuzzy-boolean μ from a Rust bool — 1.0 if true, 0.0 otherwise.
#[inline]
fn mu_bool(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Total-ordering comparison dispatch for any `Ord` lhs/rhs pair. Returns
/// `false` for unknown operators so callers map to μ = 0.
fn compare_ord<T: Ord>(op: &str, lhs: &T, rhs: &T) -> bool {
    match op {
        "=" => lhs == rhs,
        "!=" => lhs != rhs,
        ">=" => lhs >= rhs,
        "<=" => lhs <= rhs,
        ">" => lhs > rhs,
        "<" => lhs < rhs,
        _ => false,
    }
}

/// Walk a dotted JSON field path and compare the leaf value against `rhs`
/// using `op`. Returns 0.0 for missing paths, unsupported value types, or
/// unknown operators.
fn evaluate_property_path(
    props: &serde_json::Value,
    field: &str,
    op: &str,
    rhs: &str,
) -> f64 {
    let mut val = props;
    for p in field.split('.') {
        match val {
            serde_json::Value::Object(m) => match m.get(p) {
                Some(v) => val = v,
                None => return 0.0,
            },
            _ => return 0.0,
        }
    }
    match val {
        serde_json::Value::Number(n) => {
            let threshold: f64 = rhs.parse().unwrap_or(0.0);
            mu_bool(compare_f64(op, n.as_f64().unwrap_or(0.0), threshold))
        }
        serde_json::Value::String(s) => {
            let stripped = rhs.trim_matches('"');
            match op {
                "=" => mu_bool(s == stripped),
                "!=" => mu_bool(s != stripped),
                _ => 0.0,
            }
        }
        serde_json::Value::Bool(b) => {
            let want: bool = rhs.parse().unwrap_or(false);
            match op {
                "=" => mu_bool(*b == want),
                "!=" => mu_bool(*b != want),
                _ => 0.0,
            }
        }
        _ => 0.0,
    }
}

fn split_comparison(s: &str) -> Result<(&str, &str, &str)> {
    for op in &[">=", "<=", "!=", ">", "<", "="] {
        if let Some(idx) = s.find(op) {
            let (lhs, rest) = s.split_at(idx);
            let rhs = &rest[op.len()..];
            return Ok((lhs, op, rhs));
        }
    }
    Err(TensaError::InvalidInput(format!(
        "quantifier predicate missing comparator: '{s}'"
    )))
}

fn compare_f64(op: &str, lhs: f64, rhs: f64) -> bool {
    match op {
        ">=" => lhs >= rhs,
        "<=" => lhs <= rhs,
        ">" => lhs > rhs,
        "<" => lhs < rhs,
        "=" => (lhs - rhs).abs() < f64::EPSILON,
        "!=" => (lhs - rhs).abs() >= f64::EPSILON,
        _ => false,
    }
}

/// `POST /fuzzy/quantify` — compute + cache the quantifier scalar.
pub async fn quantify(
    State(state): State<Arc<AppState>>,
    Json(body): Json<QuantifyBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput("narrative_id is empty".into()))
            .into_response();
    }
    let quantifier = match quantifier_from_name(&body.quantifier) {
        Ok(q) => q,
        Err(e) => return error_response(e).into_response(),
    };
    let entity_type: Option<EntityType> = match body.entity_type.as_deref() {
        None | Some("") => None,
        Some(s) => match EntityType::from_str(s) {
            Ok(t) => Some(t),
            Err(e) => return error_response(e).into_response(),
        },
    };

    let predicate = match build_entity_predicate(body.r#where.as_deref().unwrap_or("")) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };

    // Domain scan.
    let hg = &state.hypergraph;
    let entities = match hg.list_entities_by_narrative(&body.narrative_id) {
        Ok(v) => v,
        Err(e) => return error_response(e).into_response(),
    };
    let domain: Vec<Entity> = match entity_type {
        Some(t) => entities.into_iter().filter(|e| e.entity_type == t).collect(),
        None => entities,
    };
    let n = domain.len();
    let mut sum = 0.0_f64;
    for e in &domain {
        sum += predicate(e).clamp(0.0, 1.0);
    }
    let r = if n == 0 { 0.0 } else { sum / (n as f64) };
    let value = evaluate(quantifier, r);

    let hash = predicate_hash(
        body.r#where.as_deref().unwrap_or(""),
        quantifier,
        body.entity_type.as_deref(),
    );
    let result = QuantifierResult {
        quantifier: quantifier.name().to_string(),
        value,
        label: body.label.clone(),
    };
    if let Err(e) =
        save_quantifier_result(hg.store(), &body.narrative_id, &hash, &result)
    {
        tracing::warn!(
            narrative_id = %body.narrative_id,
            predicate_hash = %hash,
            "failed to persist quantifier cache ({e}); returning inline anyway"
        );
    }

    json_ok(&QuantifyResponse {
        value,
        quantifier_name: quantifier.name().to_string(),
        predicate_hash: hash,
        narrative_id: body.narrative_id,
        label: body.label,
        domain_size: n,
        cardinality_ratio: r,
        cache_hit: false,
    })
}

/// `GET /fuzzy/quantify/{nid}/{predicate_hash}` — cache-only read.
pub async fn get_quantify(
    State(state): State<Arc<AppState>>,
    Path((nid, hash)): Path<(String, String)>,
) -> impl IntoResponse {
    match load_quantifier_result(state.hypergraph.store(), &nid, &hash) {
        Ok(Some(r)) => json_ok(&serde_json::json!({
            "value": r.value,
            "quantifier_name": r.quantifier,
            "predicate_hash": hash,
            "narrative_id": nid,
            "label": r.label,
            "cache_hit": true,
        })),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "no cached quantifier result at {hash}"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}
