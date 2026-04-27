//! `POST /fuzzy/syllogism/verify` and
//! `GET /fuzzy/syllogism/{nid}/{proof_id}`.
//!
//! Fuzzy Sprint Phase 7 — synchronous graded-syllogism verification.
//! The three premise / conclusion strings use the tiny DSL in
//! [`crate::fuzzy::syllogism::parse_statement`]; the handler parses them,
//! runs the verifier under the configured t-norm + threshold, persists
//! the proof at `fz/syllog/{nid}/{proof_id_v7_BE_16}`, and returns
//! `{proof_id, degree, figure, valid, threshold, fuzzy_config}`.
//!
//! Prototype status: see [`crate::fuzzy::syllogism`] module docs.
//!
//! Cites: [murinovanovak2014peterson].

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::registry::TNormRegistry;
use crate::fuzzy::syllogism::{
    load_syllogism_proof, parse_statement, save_syllogism_proof, verify, Syllogism,
    SyllogismProof, TypePredicateResolver,
};
use crate::fuzzy::tnorm::TNormKind;

/// Default threshold when the body omits it. Matches the Phase 7 spec
/// default at the executor layer.
const DEFAULT_THRESHOLD: f64 = 0.5;

/// Body for `POST /fuzzy/syllogism/verify`.
#[derive(Debug, Deserialize)]
pub struct VerifySyllogismBody {
    pub narrative_id: String,
    /// Tiny-DSL string for the major premise
    /// (e.g. `"ALL type:Actor IS type:Actor"`).
    pub major: String,
    pub minor: String,
    pub conclusion: String,
    /// Validity threshold; defaults to `0.5`. Clamped to `[0,1]`.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Registered t-norm name (`"godel"` / `"goguen"` / `"lukasiewicz"` /
    /// `"hamacher"`); defaults to `"godel"`.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional `"figure_hint"` override (`"I"` / `"II"` / ...).
    #[serde(default)]
    pub figure_hint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct VerifySyllogismResponse {
    pub proof_id: Uuid,
    pub narrative_id: String,
    pub degree: f64,
    pub figure: String,
    pub valid: bool,
    pub threshold: f64,
    pub tnorm: String,
}

/// `POST /fuzzy/syllogism/verify` — compute + persist a proof.
pub async fn verify_syllogism(
    State(state): State<Arc<AppState>>,
    Json(body): Json<VerifySyllogismBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput(
            "narrative_id is empty".into(),
        ))
        .into_response();
    }

    let major = match parse_statement(&body.major) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let minor = match parse_statement(&body.minor) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let conclusion = match parse_statement(&body.conclusion) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };

    let tnorm_name = body.tnorm.clone().unwrap_or_else(|| "godel".to_string());
    let tnorm_kind: TNormKind = match TNormRegistry::default().get(&tnorm_name) {
        Ok(k) => k,
        Err(e) => return error_response(e).into_response(),
    };

    let threshold = body.threshold.unwrap_or(DEFAULT_THRESHOLD);
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return error_response(TensaError::InvalidInput(format!(
            "threshold must be finite in [0,1]; got {threshold}"
        )))
        .into_response();
    }

    let syl = Syllogism {
        major,
        minor,
        conclusion,
        figure_hint: body.figure_hint.clone(),
    };
    let hg = &state.hypergraph;
    let gv = match verify(
        hg,
        &body.narrative_id,
        &syl,
        tnorm_kind,
        threshold,
        &TypePredicateResolver,
    ) {
        Ok(g) => g,
        Err(e) => return error_response(e).into_response(),
    };

    let proof = SyllogismProof::new(syl, gv.clone());
    if let Err(e) = save_syllogism_proof(hg.store(), &body.narrative_id, &proof) {
        tracing::warn!(
            narrative_id = %body.narrative_id,
            proof_id = %proof.id,
            "failed to persist syllogism proof ({e}); returning inline anyway"
        );
    }

    json_ok(&VerifySyllogismResponse {
        proof_id: proof.id,
        narrative_id: body.narrative_id,
        degree: gv.degree,
        figure: gv.figure,
        valid: gv.valid,
        threshold: gv.threshold,
        tnorm: tnorm_kind.name().to_string(),
    })
    .into_response()
}

/// `GET /fuzzy/syllogism/{nid}/{proof_id}` — fetch a persisted proof.
pub async fn get_syllogism_proof(
    State(state): State<Arc<AppState>>,
    Path((nid, proof_id)): Path<(String, Uuid)>,
) -> impl IntoResponse {
    match load_syllogism_proof(state.hypergraph.store(), &nid, &proof_id) {
        Ok(Some(p)) => json_ok(&p).into_response(),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "no syllogism proof at {proof_id} in narrative '{nid}'"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}
