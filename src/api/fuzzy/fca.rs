//! `POST /fuzzy/fca/lattice`, `GET /fuzzy/fca/lattice/{lattice_id}`,
//! `GET /fuzzy/fca/lattices/{nid}`, `DELETE /fuzzy/fca/lattice/{lattice_id}`.
//!
//! Fuzzy Sprint Phase 8 — graded concept-lattice persistence over the
//! REST surface. The synchronous build path matches the TensaQL
//! `FCA LATTICE FOR ...` executor: `FormalContext::from_hypergraph` →
//! `build_lattice_with_threshold` → `save_concept_lattice`.
//!
//! DELETE scans every narrative's `fz/fca/` prefix because the endpoint
//! path carries only the lattice UUID (not the narrative_id). This is
//! cheap — the number of persisted lattices per workspace is bounded by
//! explicit user requests.
//!
//! Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::fca::{
    build_lattice_with_threshold, ConceptLattice, FormalContext, FormalContextOptions,
};
use crate::fuzzy::fca_store::{
    delete_concept_lattice, list_concept_lattices_for_narrative, save_concept_lattice,
    LatticeSummary,
};
use crate::fuzzy::registry::TNormRegistry;
use crate::fuzzy::tnorm::TNormKind;
use crate::types::EntityType;

/// Body for `POST /fuzzy/fca/lattice`.
#[derive(Debug, Deserialize)]
pub struct BuildLatticeBody {
    pub narrative_id: String,
    #[serde(default)]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub attribute_allowlist: Option<Vec<String>>,
    #[serde(default)]
    pub threshold: Option<usize>,
    /// Registered t-norm name (`"godel"` default, `"goguen"`,
    /// `"lukasiewicz"`, `"hamacher"`).
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Opt-in large-context mode (>soft cap, ≤hard cap). Logs a
    /// `warn!` when active.
    #[serde(default)]
    pub large_context: bool,
}

#[derive(Debug, Serialize)]
pub struct BuildLatticeResponse {
    pub lattice_id: Uuid,
    pub narrative_id: String,
    pub num_concepts: usize,
    pub num_objects: usize,
    pub num_attributes: usize,
    pub tnorm: String,
}

/// `POST /fuzzy/fca/lattice` — build + persist a lattice.
pub async fn build_lattice_endpoint(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BuildLatticeBody>,
) -> impl IntoResponse {
    if body.narrative_id.trim().is_empty() {
        return error_response(TensaError::InvalidInput(
            "narrative_id is empty".into(),
        ))
        .into_response();
    }
    let tnorm_name = body.tnorm.clone().unwrap_or_else(|| "godel".to_string());
    let tnorm_kind: TNormKind = match TNormRegistry::default().get(&tnorm_name) {
        Ok(k) => k,
        Err(e) => return error_response(e).into_response(),
    };
    let et: Option<EntityType> = match &body.entity_type {
        None => None,
        Some(name) => match name.parse::<EntityType>() {
            Ok(v) => Some(v),
            Err(e) => {
                return error_response(TensaError::InvalidInput(format!(
                    "entity_type '{name}' unknown: {e}"
                )))
                .into_response()
            }
        },
    };
    let opts = FormalContextOptions {
        entity_type_filter: et,
        attribute_allowlist: body.attribute_allowlist.clone(),
        large_context: body.large_context,
    };
    let hg = &state.hypergraph;
    let ctx = match FormalContext::from_hypergraph(hg, &body.narrative_id, &opts) {
        Ok(c) => c,
        Err(e) => return error_response(e).into_response(),
    };
    let mut lattice = match build_lattice_with_threshold(
        &ctx,
        tnorm_kind,
        body.threshold.unwrap_or(0),
    ) {
        Ok(l) => l,
        Err(e) => return error_response(e).into_response(),
    };
    lattice.narrative_id = body.narrative_id.clone();
    if let Err(e) = save_concept_lattice(hg.store(), &body.narrative_id, &lattice) {
        tracing::warn!(
            narrative_id = %body.narrative_id,
            lattice_id = %lattice.id,
            "failed to persist FCA lattice ({e}); returning inline"
        );
    }
    json_ok(&BuildLatticeResponse {
        lattice_id: lattice.id,
        narrative_id: lattice.narrative_id.clone(),
        num_concepts: lattice.num_concepts(),
        num_objects: lattice.objects.len(),
        num_attributes: lattice.attributes.len(),
        tnorm: tnorm_kind.name().to_string(),
    })
    .into_response()
}

/// `GET /fuzzy/fca/lattice/{lattice_id}` — load a persisted lattice
/// via workspace-wide scan (the URL does not carry the narrative_id).
pub async fn get_lattice_endpoint(
    State(state): State<Arc<AppState>>,
    Path(lattice_id): Path<Uuid>,
) -> impl IntoResponse {
    let pairs = match state
        .hypergraph
        .store()
        .prefix_scan(crate::fuzzy::FUZZY_FCA_PREFIX)
    {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    for (_, v) in pairs {
        match serde_json::from_slice::<ConceptLattice>(&v) {
            Ok(l) if l.id == lattice_id => return json_ok(&l).into_response(),
            _ => continue,
        }
    }
    error_response(TensaError::NotFound(format!(
        "FCA lattice {lattice_id} not persisted"
    )))
    .into_response()
}

/// `GET /fuzzy/fca/lattices/{nid}` — list every lattice summary for a
/// narrative, newest-first.
pub async fn list_lattices_endpoint(
    State(state): State<Arc<AppState>>,
    Path(nid): Path<String>,
) -> impl IntoResponse {
    match list_concept_lattices_for_narrative(state.hypergraph.store(), &nid) {
        Ok(lats) => {
            let out: Vec<LatticeSummary> =
                lats.iter().map(LatticeSummary::from_lattice).collect();
            json_ok(&out).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

/// `DELETE /fuzzy/fca/lattice/{lattice_id}` — find + drop the matching
/// lattice. Idempotent: non-existent ids return 204 (via `{ "deleted":
/// false }`).
pub async fn delete_lattice_endpoint(
    State(state): State<Arc<AppState>>,
    Path(lattice_id): Path<Uuid>,
) -> impl IntoResponse {
    let pairs = match state
        .hypergraph
        .store()
        .prefix_scan(crate::fuzzy::FUZZY_FCA_PREFIX)
    {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    for (_, v) in pairs {
        if let Ok(l) = serde_json::from_slice::<ConceptLattice>(&v) {
            if l.id == lattice_id {
                if let Err(e) = delete_concept_lattice(
                    state.hypergraph.store(),
                    &l.narrative_id,
                    &lattice_id,
                ) {
                    return error_response(e).into_response();
                }
                return json_ok(&serde_json::json!({"deleted": true})).into_response();
            }
        }
    }
    json_ok(&serde_json::json!({"deleted": false})).into_response()
}
