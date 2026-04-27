//! HTTP handlers for the disinfo CIB detection extension (Sprint D3).
//!
//! Endpoints:
//! - `POST /analysis/cib` — body `{narrative_id, cross_platform?, similarity_threshold?,
//!   alpha?, bootstrap_iter?, min_cluster_size?, seed?}`. Runs behavioral similarity
//!   network → label-propagation → density null calibration → flag clusters below α.
//!   Persists clusters + evidence, returns the full detection result.
//! - `GET  /analysis/cib/:narrative_id` — load the persisted cluster + evidence
//!   records for the narrative (no recomputation).
//! - `POST /analysis/superspreaders` — body `{narrative_id, method?, top_n?}`.
//!   Ranks actors by graph centrality (pagerank/eigenvector/harmonic).
//!
//! Cfg-gated behind `disinfo`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::analysis::cib::{
    detect_cib_clusters, detect_cross_platform_cib, load_cib_detection, rank_superspreaders,
    CibConfig, SuperspreaderMethod,
};
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

#[derive(Debug, Deserialize)]
pub struct CibDetectRequest {
    pub narrative_id: String,
    /// If `true`, only flag clusters spanning ≥ 2 platforms.
    #[serde(default)]
    pub cross_platform: bool,
    #[serde(default)]
    pub similarity_threshold: Option<f64>,
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub bootstrap_iter: Option<usize>,
    #[serde(default)]
    pub min_cluster_size: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
}

impl CibDetectRequest {
    fn to_config(&self) -> CibConfig {
        // Round-trip through JSON so the single authoritative parser in
        // `CibConfig::from_json` stays the only place config shape is read.
        CibConfig::from_json(&serde_json::json!({
            "similarity_threshold": self.similarity_threshold,
            "alpha": self.alpha,
            "bootstrap_iter": self.bootstrap_iter,
            "min_cluster_size": self.min_cluster_size,
            "seed": self.seed,
        }))
    }
}

/// `POST /analysis/cib` — run CIB detection, persist, return detection result.
pub async fn run_cib_detection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CibDetectRequest>,
) -> axum::response::Response {
    let config = req.to_config();
    let result = if req.cross_platform {
        detect_cross_platform_cib(&state.hypergraph, &req.narrative_id, &config)
    } else {
        detect_cib_clusters(&state.hypergraph, &req.narrative_id, &config)
    };
    match result {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /analysis/cib/:narrative_id` — load persisted CIB detection result.
pub async fn get_cib_detection(
    State(state): State<Arc<AppState>>,
    Path(narrative_id): Path<String>,
) -> axum::response::Response {
    match load_cib_detection(&state.hypergraph, &narrative_id) {
        Ok(Some(result)) => json_ok(&result),
        Ok(None) => error_response(TensaError::NotFound(format!(
            "no CIB clusters for narrative '{narrative_id}' — POST /analysis/cib to compute"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct SuperspreaderRequest {
    pub narrative_id: String,
    /// `"pagerank"` | `"eigenvector"` | `"harmonic"`. Defaults to `pagerank`.
    #[serde(default)]
    pub method: Option<String>,
    /// Number of actors to return. Defaults to 10.
    #[serde(default)]
    pub top_n: Option<usize>,
}

/// `POST /analysis/superspreaders` — rank actors by centrality, persist, return.
pub async fn run_superspreaders(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SuperspreaderRequest>,
) -> axum::response::Response {
    use std::str::FromStr;

    let method = match req.method.as_deref() {
        Some(s) => match SuperspreaderMethod::from_str(s) {
            Ok(m) => m,
            Err(e) => return error_response(e).into_response(),
        },
        None => SuperspreaderMethod::PageRank,
    };
    let top_n = req.top_n.unwrap_or(10);
    match rank_superspreaders(&state.hypergraph, &req.narrative_id, method, top_n) {
        Ok(r) => json_ok(&r),
        Err(e) => error_response(e).into_response(),
    }
}
