//! Analytics read-back endpoints.
//!
//! GET handlers that scan `an/*` KV prefixes and return the stored JSON as-is.
//! Complements the per-entity virtual-property path (`e.an.pagerank`, …) —
//! used when results are narrative-level blobs (NetInf, SIR, argumentation)
//! rather than per-entity scores. Write path is the matching inference engine
//! submitted via `POST /jobs`; these endpoints return 404 / empty when no
//! engine run has happened yet. See §5.15.1 of TENSA_REFERENCE.md.

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use axum::{
    extract::{Path, State},
    response::IntoResponse,
};
use serde_json::Value;
use std::sync::Arc;

/// Scan a KV prefix matching `{prefix}{narrative_id}/…` and return every value
/// parsed as JSON. Returns empty vec rather than 404 so callers can tell
/// "engine ran but no results" from "engine never ran".
fn scan_narrative_prefix(
    store: &dyn crate::store::KVStore,
    prefix: &[u8],
    narrative_id: &str,
) -> Result<Vec<Value>, TensaError> {
    let mut full = prefix.to_vec();
    full.extend_from_slice(narrative_id.as_bytes());
    full.push(b'/');
    let pairs = store.prefix_scan(&full)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        if let Ok(val) = serde_json::from_slice::<Value>(&v) {
            out.push(val);
        }
    }
    Ok(out)
}

/// Read a single blob at `{prefix}{narrative_id}` (no trailing slash segment).
/// Returns `None` when absent.
fn get_narrative_blob(
    store: &dyn crate::store::KVStore,
    prefix: &[u8],
    narrative_id: &str,
) -> Result<Option<Value>, TensaError> {
    let mut key = prefix.to_vec();
    key.extend_from_slice(narrative_id.as_bytes());
    match store.get(&key)? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes).ok()),
        None => Ok(None),
    }
}

fn blob_response(
    result: Result<Option<Value>, TensaError>,
    kind: &str,
) -> axum::response::Response {
    match result {
        Ok(Some(v)) => json_ok(&v).into_response(),
        Ok(None) => error_response(TensaError::QueryError(format!(
            "no {kind} result — run the corresponding inference job first"
        )))
        .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

fn list_response(result: Result<Vec<Value>, TensaError>, kind: &str) -> axum::response::Response {
    match result {
        Ok(list) => json_ok(&serde_json::json!({kind: list})).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /narratives/:id/contagion` — list all SIR cascade results.
pub async fn get_contagion(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    list_response(
        scan_narrative_prefix(state.hypergraph.store(), b"an/sir/", &id),
        "cascades",
    )
}

/// `GET /narratives/:id/netinf` — NetInf diffusion network blob.
pub async fn get_netinf(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    blob_response(
        get_narrative_blob(state.hypergraph.store(), b"an/ni/", &id),
        "netinf",
    )
}

/// `GET /narratives/:id/temporal-motifs` — motif census blob.
pub async fn get_temporal_motifs(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    blob_response(
        get_narrative_blob(state.hypergraph.store(), b"an/tm/", &id),
        "temporal_motifs",
    )
}

/// `GET /narratives/:id/faction-evolution` — faction evolution blob.
pub async fn get_faction_evolution(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    blob_response(
        get_narrative_blob(state.hypergraph.store(), b"an/fe/", &id),
        "faction_evolution",
    )
}

/// `GET /narratives/:id/temporal-rules` — Temporal ILP clauses.
pub async fn get_temporal_rules(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    blob_response(
        get_narrative_blob(state.hypergraph.store(), b"an/ilp/", &id),
        "temporal_rules",
    )
}

/// `GET /narratives/:id/mean-field` — MFG equilibria for every situation in
/// the narrative. MFG is keyed by `an/mfg/{situation_id}`, so we enumerate
/// situations first then look up each one.
pub async fn get_mean_field(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let store = state.hypergraph.store();
    let situations = match state.hypergraph.list_situations_by_narrative(&id) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let mut out = Vec::new();
    for s in situations {
        let mut key = b"an/mfg/".to_vec();
        key.extend_from_slice(s.id.to_string().as_bytes());
        match store.get(&key) {
            Ok(Some(bytes)) => {
                if let Ok(val) = serde_json::from_slice::<Value>(&bytes) {
                    out.push(serde_json::json!({"situation_id": s.id.to_string(), "result": val}));
                }
            }
            Ok(None) => {}
            Err(e) => return error_response(e).into_response(),
        }
    }
    json_ok(&serde_json::json!({"mean_field": out})).into_response()
}

/// `GET /narratives/:id/psl` — PSL rule scores.
pub async fn get_psl(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    blob_response(
        get_narrative_blob(state.hypergraph.store(), b"an/psl/", &id),
        "psl",
    )
}

/// `GET /narratives/:id/arguments` — Dung argumentation extensions.
pub async fn get_arguments(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    list_response(
        scan_narrative_prefix(state.hypergraph.store(), b"an/af/", &id),
        "arguments",
    )
}

/// `GET /narratives/:id/evidence` — Dempster-Shafer evidence fusions.
pub async fn get_evidence(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    list_response(
        scan_narrative_prefix(state.hypergraph.store(), b"an/ev/", &id),
        "evidence",
    )
}

/// `GET /narratives/:id/contentions` — narrative-wide contention list.
/// Complements the existing per-situation `GET /situations/:id/contentions`.
pub async fn get_narrative_contentions(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.hypergraph.list_contentions_for_narrative(&id) {
        Ok(links) => json_ok(&serde_json::json!({"contentions": links})).into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Name-keyed path / flow ──────────────────────────────────────

/// Resolve source/target endpoint names into `(CoGraph, src_idx, tgt_idx)`.
/// Both name-keyed handlers share this setup.
fn resolve_pair_in_cograph(
    state: &AppState,
    body: &Value,
) -> Result<(crate::analysis::graph_projection::CoGraph, usize, usize), TensaError> {
    let narrative_id = body
        .get("narrative_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let source_name = body
        .get("source_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let target_name = body
        .get("target_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if narrative_id.is_empty() || source_name.is_empty() || target_name.is_empty() {
        return Err(TensaError::QueryError(
            "narrative_id, source_name, target_name required".into(),
        ));
    }
    let find = |n: &str| -> Result<uuid::Uuid, TensaError> {
        state
            .hypergraph
            .find_entity_by_name(narrative_id, n)?
            .map(|e| e.id)
            .ok_or_else(|| {
                TensaError::QueryError(format!(
                    "no entity named '{n}' in narrative '{narrative_id}'"
                ))
            })
    };
    let src = find(source_name)?;
    let tgt = find(target_name)?;
    let graph = crate::analysis::graph_projection::build_co_graph(&state.hypergraph, narrative_id)?;
    let si = graph
        .entities
        .iter()
        .position(|&e| e == src)
        .ok_or_else(|| TensaError::QueryError("source not in narrative co-graph".into()))?;
    let ti = graph
        .entities
        .iter()
        .position(|&e| e == tgt)
        .ok_or_else(|| TensaError::QueryError("target not in narrative co-graph".into()))?;
    Ok((graph, si, ti))
}

/// Resolve a UUID string to its `properties.name` (for human-readable path rendering).
fn entity_display_name(state: &AppState, id_str: &str) -> String {
    id_str
        .parse::<uuid::Uuid>()
        .ok()
        .and_then(|u| state.hypergraph.get_entity(&u).ok())
        .and_then(|e| {
            e.properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| id_str.to_string())
}

fn render_path(
    state: &AppState,
    graph: &crate::analysis::graph_projection::CoGraph,
    sp: &crate::analysis::pathfinding::ShortestPath,
) -> Value {
    let ids: Vec<String> = sp
        .path
        .iter()
        .map(|&i| graph.entities[i].to_string())
        .collect();
    let names: Vec<String> = ids
        .iter()
        .map(|id| entity_display_name(state, id))
        .collect();
    serde_json::json!({
        "path": ids,
        "names": names,
        "length": sp.path.len() - 1,
        "total_weight": sp.total_weight,
    })
}

/// `POST /analysis/shortest-path-by-name` — name-keyed wrapper around
/// `/analysis/shortest-path`. Accepts `{narrative_id, source_name, target_name, k?}`.
pub async fn shortest_path_by_name(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<Value>,
) -> impl IntoResponse {
    let (graph, si, ti) = match resolve_pair_in_cograph(&state, &body) {
        Ok(x) => x,
        Err(e) => return error_response(e).into_response(),
    };
    if let Some(k) = body.get("k").and_then(|v| v.as_u64()).map(|v| v as usize) {
        let paths = crate::analysis::pathfinding::yen_k_shortest(&graph, si, ti, k);
        let results: Vec<Value> = paths
            .iter()
            .map(|sp| render_path(&state, &graph, sp))
            .collect();
        json_ok(&serde_json::json!({"paths": results})).into_response()
    } else {
        match crate::analysis::pathfinding::dijkstra(&graph, si, ti) {
            Some(sp) => json_ok(&render_path(&state, &graph, &sp)).into_response(),
            None => json_ok(&serde_json::json!({"path": null, "message": "No path found"}))
                .into_response(),
        }
    }
}

/// `POST /analysis/max-flow-by-name` — name-keyed wrapper around
/// `/analysis/max-flow`. Accepts `{narrative_id, source_name, target_name}`.
pub async fn max_flow_by_name(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<Value>,
) -> impl IntoResponse {
    let (graph, si, ti) = match resolve_pair_in_cograph(&state, &body) {
        Ok(x) => x,
        Err(e) => return error_response(e).into_response(),
    };
    let (flow, cut_edges) = crate::analysis::pathfinding::max_flow(&graph, si, ti);
    let cuts: Vec<Value> = cut_edges
        .iter()
        .map(|&(a, b)| {
            serde_json::json!([graph.entities[a].to_string(), graph.entities[b].to_string()])
        })
        .collect();
    json_ok(&serde_json::json!({"flow": flow, "cut_edges": cuts})).into_response()
}
