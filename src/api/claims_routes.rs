//! Claims & fact-check API routes (Sprint D4).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::claims;
use crate::error::TensaError;

// ─── Request types ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateClaimRequest {
    pub text: String,
    #[serde(default)]
    pub narrative_id: Option<String>,
    #[serde(default)]
    pub source_situation_id: Option<String>,
    #[serde(default)]
    pub source_entity_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateFactCheckRequest {
    pub claim_id: String,
    pub verdict: String,
    pub source: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "default_lang")]
    pub language: String,
    #[serde(default)]
    pub explanation: Option<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

fn default_lang() -> String {
    "en".to_string()
}
fn default_confidence() -> f64 {
    0.9
}

#[derive(Debug, Deserialize)]
pub struct MatchClaimRequest {
    pub claim_id: String,
    #[serde(default = "default_min_sim")]
    pub min_similarity: f64,
}

fn default_min_sim() -> f64 {
    0.5
}

#[derive(Debug, Deserialize)]
pub struct ListClaimsQuery {
    #[serde(default)]
    pub narrative_id: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    100
}

// ─── Handlers ───────────────────────────────────────────────

/// POST /claims — detect claims in text and persist them.
pub async fn create_claims(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateClaimRequest>,
) -> Response {
    let sit_id = req
        .source_situation_id
        .as_deref()
        .and_then(|s| s.parse::<Uuid>().ok());
    let ent_id = req
        .source_entity_id
        .as_deref()
        .and_then(|s| s.parse::<Uuid>().ok());

    let detected = claims::detect_claims(&req.text, req.narrative_id.as_deref(), sit_id, ent_id);

    let hg = &state.hypergraph;
    let mut stored = Vec::new();
    for claim in &detected {
        if let Err(e) = claims::detection::store_claim(hg, claim) {
            return error_response(e).into_response();
        }
        stored.push(serde_json::to_value(claim).unwrap_or_default());
    }

    json_ok(&serde_json::json!({
        "claims_detected": stored.len(),
        "claims": stored,
    }))
}

/// GET /claims — list claims, optionally filtered by narrative.
pub async fn list_claims(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListClaimsQuery>,
) -> Response {
    match query.narrative_id.as_deref() {
        Some(nid) => match claims::detection::list_claims_for_narrative(&state.hypergraph, nid) {
            Ok(mut claims_list) => {
                claims_list.truncate(query.limit);
                json_ok(&claims_list)
            }
            Err(e) => error_response(e).into_response(),
        },
        None => {
            // Without narrative filter, list all claims (prefix scan)
            match claims::detection::list_claims_for_narrative(&state.hypergraph, "") {
                Ok(mut claims_list) => {
                    claims_list.truncate(query.limit);
                    json_ok(&claims_list)
                }
                Err(e) => error_response(e).into_response(),
            }
        }
    }
}

/// GET /claims/:id — get a claim by UUID.
pub async fn get_claim(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    match claims::detection::load_claim(&state.hypergraph, &uuid) {
        Ok(Some(claim)) => json_ok(&claim),
        Ok(None) => {
            error_response(TensaError::NotFound(format!("Claim {} not found", id))).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /claims/match — match a claim against known fact-checks.
pub async fn match_claim_route(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MatchClaimRequest>,
) -> Response {
    let uuid = match req.claim_id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    let claim = match claims::detection::load_claim(&state.hypergraph, &uuid) {
        Ok(Some(c)) => c,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!(
                "Claim {} not found",
                req.claim_id
            )))
            .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    match claims::match_claim(&state.hypergraph, &claim, req.min_similarity) {
        Ok(matches) => json_ok(&matches),
        Err(e) => error_response(e).into_response(),
    }
}

/// POST /fact-checks — ingest a fact-check for a claim.
pub async fn create_fact_check(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateFactCheckRequest>,
) -> Response {
    let claim_id = match req.claim_id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    let verdict = match req.verdict.parse::<claims::FactCheckVerdict>() {
        Ok(v) => v,
        Err(e) => return error_response(e).into_response(),
    };
    match claims::ingest_fact_check(
        &state.hypergraph,
        claim_id,
        verdict,
        &req.source,
        req.url.as_deref(),
        &req.language,
        req.explanation.as_deref(),
        req.confidence,
    ) {
        Ok(fc) => json_ok(&fc),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /claims/:id/origin — trace claim origin through the temporal chain.
pub async fn trace_claim_origin(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    let claim = match claims::detection::load_claim(&state.hypergraph, &uuid) {
        Ok(Some(c)) => c,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!("Claim {} not found", id)))
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };

    // Build origin trace from mutations
    let mutations = match claims::track_mutations(&state.hypergraph, &claim) {
        Ok(m) => m,
        Err(e) => return error_response(e).into_response(),
    };

    let mut chain: Vec<claims::ClaimAppearance> = Vec::new();
    chain.push(claims::ClaimAppearance {
        claim_id: claim.id,
        situation_id: claim.source_situation_id,
        entity_id: claim.source_entity_id,
        timestamp: Some(claim.created_at),
        similarity_to_original: 1.0,
    });

    for m in &mutations {
        if m.original_claim_id != claim.id {
            if let Ok(Some(orig)) =
                claims::detection::load_claim(&state.hypergraph, &m.original_claim_id)
            {
                chain.push(claims::ClaimAppearance {
                    claim_id: orig.id,
                    situation_id: orig.source_situation_id,
                    entity_id: orig.source_entity_id,
                    timestamp: Some(orig.created_at),
                    similarity_to_original: 1.0 - m.embedding_drift,
                });
            }
        }
    }

    // Sort by timestamp (earliest first)
    chain.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    let earliest = chain.first().cloned();

    let trace = claims::ClaimOriginTrace {
        claim_id: claim.id,
        chain,
        earliest,
    };

    json_ok(&trace)
}

/// GET /claims/:id/mutations — list mutation events for a claim.
pub async fn list_mutations(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    let claim = match claims::detection::load_claim(&state.hypergraph, &uuid) {
        Ok(Some(c)) => c,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!("Claim {} not found", id)))
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    match claims::track_mutations(&state.hypergraph, &claim) {
        Ok(mutations) => json_ok(&mutations),
        Err(e) => error_response(e).into_response(),
    }
}
