//! `POST /analysis/fuzzy-allen` and
//! `GET /analysis/fuzzy-allen/{nid}/{a_id}/{b_id}`.
//!
//! Fuzzy Sprint Phase 5. Returns the graded-Allen 13-vector between two
//! situations, cached at `fz/allen/{narrative_id}/{a_id_BE}/{b_id_BE}`.
//!
//! * `POST` computes on demand from the two situations' temporal intervals
//!   and caches the result.
//! * `GET` reads from the cache if present, otherwise recomputes
//!   (fall-through to the POST path so the two endpoints stay
//!   idempotent).
//!
//! Cites: [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen].

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::fuzzy::allen::{
    graded_relation, index_to_relation, load_fuzzy_allen, save_fuzzy_allen,
};

#[derive(Debug, Deserialize)]
pub struct FuzzyAllenBody {
    pub narrative_id: String,
    pub a_id: Uuid,
    pub b_id: Uuid,
}

/// Render a 13-vector as a list of `{name, degree}` objects in Allen's
/// canonical order so clients don't have to import the index mapping.
fn render_vector(v: &[f64; 13]) -> serde_json::Value {
    let rels = (0..13)
        .map(|i| {
            let rel = index_to_relation(i).expect("index in range");
            serde_json::json!({
                "name": format!("{:?}", rel),
                "degree": v[i],
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({ "relations": rels })
}

/// Compute + cache the graded-Allen 13-vector for a pair of situations.
pub async fn compute(
    State(state): State<Arc<AppState>>,
    Json(body): Json<FuzzyAllenBody>,
) -> impl IntoResponse {
    let hg = &state.hypergraph;
    let a = match hg.get_situation(&body.a_id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let b = match hg.get_situation(&body.b_id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let vector = graded_relation(&a.temporal, &b.temporal);
    if let Err(e) = save_fuzzy_allen(
        hg.store(),
        &body.narrative_id,
        &body.a_id,
        &body.b_id,
        &vector,
    ) {
        return error_response(e).into_response();
    }
    json_ok(&render_vector(&vector))
}

/// Read the cached graded-Allen 13-vector; recompute + cache if missing.
pub async fn get(
    State(state): State<Arc<AppState>>,
    Path((nid, a_id, b_id)): Path<(String, Uuid, Uuid)>,
) -> impl IntoResponse {
    let hg = &state.hypergraph;
    match load_fuzzy_allen(hg.store(), &nid, &a_id, &b_id) {
        Ok(Some(v)) => json_ok(&render_vector(&v)),
        Ok(None) => {
            // Fall through to recompute — same shape as POST.
            compute(
                State(state),
                Json(FuzzyAllenBody {
                    narrative_id: nid,
                    a_id,
                    b_id,
                }),
            )
            .await
            .into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}
