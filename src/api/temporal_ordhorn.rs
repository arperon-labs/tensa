//! `POST /temporal/ordhorn/closure` — synchronous van Beek path-
//! consistency closure on an Allen interval-algebra network.
//!
//! Phase 4 of the Graded Acceptability sprint. Mirrors the synchronous
//! pattern established by `POST /analysis/higher-order-contagion` (Fuzzy
//! Sprint Phase 7b): the request carries the full constraint network,
//! the handler runs [`crate::temporal::ordhorn::closure`], and the
//! response carries both the tightened network and a sound-but-only-
//! complete-for-ORD-Horn `satisfiable` flag.
//!
//! The handler does not touch the hypergraph store — closure is a pure
//! transformation on the supplied network. This keeps the endpoint
//! cacheable + idempotent + workspace-agnostic.
//!
//! Cites: [nebel1995ordhorn].

use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::{error_response, json_ok};
use crate::error::TensaError;
use crate::temporal::ordhorn::{closure, OrdHornNetwork};

/// Request body for `POST /temporal/ordhorn/closure`.
#[derive(Debug, Deserialize)]
pub struct OrdHornClosureRequest {
    pub network: OrdHornNetwork,
}

/// Response body for `POST /temporal/ordhorn/closure`.
#[derive(Debug, Serialize)]
pub struct OrdHornClosureResponse {
    pub closed_network: OrdHornNetwork,
    pub satisfiable: bool,
}

/// `POST /temporal/ordhorn/closure` — run path-consistency closure on
/// an Allen constraint network.
///
/// Failure modes:
/// * Closure exceeds its internal iteration cap → HTTP 400 via
///   `error_response` (`InvalidInput`). This indicates a bug or a
///   pathological input, not a normal unsatisfiable network.
pub async fn closure_handler(Json(req): Json<OrdHornClosureRequest>) -> impl IntoResponse {
    if req.network.n == 0 && !req.network.constraints.is_empty() {
        return error_response(TensaError::InvalidInput(
            "network has zero intervals but non-empty constraints".into(),
        ))
        .into_response();
    }

    let closed = match closure(&req.network) {
        Ok(c) => c,
        Err(e) => return error_response(e).into_response(),
    };
    // Satisfiability is decidable from the closure output: a non-empty
    // closure means no contradiction was found. This avoids running
    // closure twice.
    let satisfiable = closed.constraints.iter().all(|c| !c.relations.is_empty());

    let response = OrdHornClosureResponse {
        closed_network: closed,
        satisfiable,
    };
    json_ok(&response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::ordhorn::OrdHornConstraint;
    use crate::types::AllenRelation;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::util::ServiceExt;

    fn make_router() -> Router {
        Router::new().route("/temporal/ordhorn/closure", post(closure_handler))
    }

    async fn post_json(router: Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method("POST")
            .uri("/temporal/ordhorn/closure")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).expect("serialise body")))
            .expect("build request");
        let resp = router.oneshot(req).await.expect("oneshot");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("collect body");
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("response should be JSON");
        (status, value)
    }

    #[tokio::test]
    async fn t_rest_1_satisfiable_round_trip() {
        let net = OrdHornNetwork {
            n: 3,
            constraints: vec![
                OrdHornConstraint {
                    a: 0,
                    b: 1,
                    relations: vec![AllenRelation::Before],
                },
                OrdHornConstraint {
                    a: 1,
                    b: 2,
                    relations: vec![AllenRelation::Before],
                },
            ],
        };
        let body = serde_json::json!({ "network": net });

        let (status, value) = post_json(make_router(), body).await;
        assert_eq!(status, StatusCode::OK, "expected 200, got {}: {}", status, value);

        assert_eq!(
            value
                .get("satisfiable")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            true,
            "expected satisfiable=true, body={value}"
        );

        let closed_constraints = value
            .get("closed_network")
            .and_then(|n| n.get("constraints"))
            .and_then(|c| c.as_array())
            .expect("closed_network.constraints array missing");

        // After closure we expect (0, 2) tightened to {Before}.
        let has_propagated_before = closed_constraints.iter().any(|c| {
            let a = c.get("a").and_then(|v| v.as_u64());
            let b = c.get("b").and_then(|v| v.as_u64());
            let rels = c.get("relations").and_then(|v| v.as_array());
            matches!((a, b, rels), (Some(0), Some(2), Some(rs)) if rs.len() == 1
                && rs[0].as_str() == Some("Before"))
        });
        assert!(
            has_propagated_before,
            "expected (0, 2) -> [Before] after closure, got {:?}",
            closed_constraints
        );
    }

    #[tokio::test]
    async fn t_rest_2_unsatisfiable_round_trip() {
        let net = OrdHornNetwork {
            n: 3,
            constraints: vec![
                OrdHornConstraint {
                    a: 0,
                    b: 1,
                    relations: vec![AllenRelation::Before],
                },
                OrdHornConstraint {
                    a: 1,
                    b: 2,
                    relations: vec![AllenRelation::Before],
                },
                OrdHornConstraint {
                    a: 0,
                    b: 2,
                    relations: vec![AllenRelation::After],
                },
            ],
        };
        let body = serde_json::json!({ "network": net });

        let (status, value) = post_json(make_router(), body).await;
        assert_eq!(status, StatusCode::OK, "expected 200 even for unsat: {}", value);

        assert_eq!(
            value
                .get("satisfiable")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            false,
            "expected satisfiable=false, body={value}"
        );

        let closed_constraints = value
            .get("closed_network")
            .and_then(|n| n.get("constraints"))
            .and_then(|c| c.as_array())
            .expect("closed_network.constraints array missing");
        let has_empty = closed_constraints.iter().any(|c| {
            c.get("relations")
                .and_then(|v| v.as_array())
                .map(|arr| arr.is_empty())
                .unwrap_or(false)
        });
        assert!(
            has_empty,
            "expected at least one empty-relations constraint, got {:?}",
            closed_constraints
        );
    }
}
