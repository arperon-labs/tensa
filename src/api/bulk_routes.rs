//! Bulk operations for entities and situations (Sprint P3.6 — F-AP1).
//!
//! Provides endpoints for creating multiple entities or situations in a single
//! request, with per-item error reporting.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::server::AppState;
use crate::types::{Entity, Situation};

/// Maximum number of items allowed in a single bulk request.
const MAX_BULK_SIZE: usize = 1000;

/// Request body for bulk entity creation.
#[derive(Debug, Deserialize)]
pub struct BulkEntityRequest {
    pub entities: Vec<serde_json::Value>,
}

/// Request body for bulk situation creation.
#[derive(Debug, Deserialize)]
pub struct BulkSituationRequest {
    pub situations: Vec<serde_json::Value>,
}

/// Result for a single item in a bulk operation.
#[derive(Debug, Serialize)]
pub struct BulkItemResult {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Aggregate result for a bulk operation.
#[derive(Debug, Serialize)]
pub struct BulkResult {
    pub results: Vec<BulkItemResult>,
    pub succeeded: usize,
    pub failed: usize,
}

/// Generic bulk processor: deserialize each JSON value and apply `create_fn`.
fn process_bulk<T, F>(values: Vec<serde_json::Value>, create_fn: F) -> BulkResult
where
    T: serde::de::DeserializeOwned,
    F: Fn(T) -> crate::error::Result<Uuid>,
{
    let mut results = Vec::with_capacity(values.len());
    let mut succeeded = 0usize;
    let mut failed = 0usize;

    for (i, val) in values.into_iter().enumerate() {
        match serde_json::from_value::<T>(val) {
            Ok(item) => match create_fn(item) {
                Ok(id) => {
                    succeeded += 1;
                    results.push(BulkItemResult {
                        index: i,
                        id: Some(id),
                        error: None,
                    });
                }
                Err(e) => {
                    failed += 1;
                    results.push(BulkItemResult {
                        index: i,
                        id: None,
                        error: Some(e.to_string()),
                    });
                }
            },
            Err(e) => {
                failed += 1;
                results.push(BulkItemResult {
                    index: i,
                    id: None,
                    error: Some(format!("Deserialization error: {e}")),
                });
            }
        }
    }

    BulkResult {
        results,
        succeeded,
        failed,
    }
}

/// Process a batch of JSON values into entities.
pub fn process_bulk_entities(
    values: Vec<serde_json::Value>,
    hypergraph: &crate::hypergraph::Hypergraph,
) -> BulkResult {
    process_bulk::<Entity, _>(values, |e| hypergraph.create_entity(e))
}

/// Process a batch of JSON values into situations. Each value is fed through
/// [`crate::api::routes::fill_situation_defaults`] first, so callers may send
/// permissive partial DTOs (the same shape the single `POST /situations`
/// handler accepts).
pub fn process_bulk_situations(
    values: Vec<serde_json::Value>,
    hypergraph: &crate::hypergraph::Hypergraph,
) -> BulkResult {
    let filled: Vec<serde_json::Value> = values
        .into_iter()
        .map(|mut v| {
            crate::api::routes::fill_situation_defaults(&mut v);
            v
        })
        .collect();
    process_bulk::<Situation, _>(filled, |s| hypergraph.create_situation(s))
}

/// POST /entities/bulk — Create multiple entities in one request.
pub async fn bulk_create_entities(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BulkEntityRequest>,
) -> impl IntoResponse {
    if body.entities.len() > MAX_BULK_SIZE {
        return (
            StatusCode::BAD_REQUEST,
            Json(BulkResult {
                results: vec![],
                succeeded: 0,
                failed: body.entities.len(),
            }),
        )
            .into_response();
    }
    let result = process_bulk_entities(body.entities, &state.hypergraph);
    let status = if result.failed > 0 && result.succeeded == 0 {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::OK
    };
    (status, Json(result)).into_response()
}

/// POST /situations/bulk — Create multiple situations in one request.
pub async fn bulk_create_situations(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BulkSituationRequest>,
) -> impl IntoResponse {
    if body.situations.len() > MAX_BULK_SIZE {
        return (
            StatusCode::BAD_REQUEST,
            Json(BulkResult {
                results: vec![],
                succeeded: 0,
                failed: body.situations.len(),
            }),
        )
            .into_response();
    }
    let result = process_bulk_situations(body.situations, &state.hypergraph);
    let status = if result.failed > 0 && result.succeeded == 0 {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::OK
    };
    (status, Json(result)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::Hypergraph;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    fn make_entity_json(name: &str, entity_type: &str) -> serde_json::Value {
        serde_json::json!({
            "id": Uuid::now_v7(),
            "entity_type": entity_type,
            "properties": {"name": name},
            "maturity": "Candidate",
            "confidence": 0.9,
            "provenance": [],
            "created_at": Utc::now(),
            "updated_at": Utc::now(),
        })
    }

    fn make_situation_json(desc: &str) -> serde_json::Value {
        serde_json::json!({
            "id": Uuid::now_v7(),
            "temporal": {"granularity": "Unknown", "relations": []},
            "causes": [],
            "raw_content": [{"content_type": "Text", "content": desc}],
            "narrative_level": "Scene",
            "maturity": "Candidate",
            "confidence": 0.8,
            "extraction_method": "HumanEntered",
            "provenance": [],
            "created_at": Utc::now(),
            "updated_at": Utc::now(),
        })
    }

    #[test]
    fn test_bulk_create_entities_all_valid() {
        let hg = Hypergraph::new(test_store());
        let entities = vec![
            make_entity_json("Alice", "Actor"),
            make_entity_json("Berlin", "Location"),
            make_entity_json("ACME", "Organization"),
        ];
        let result = process_bulk_entities(entities, &hg);
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.failed, 0);
        assert!(result.results.iter().all(|r| r.id.is_some()));
    }

    #[test]
    fn test_bulk_create_situations_all_valid() {
        let hg = Hypergraph::new(test_store());
        let situations = vec![
            make_situation_json("Meeting"),
            make_situation_json("Departure"),
            make_situation_json("Arrival"),
        ];
        let result = process_bulk_situations(situations, &hg);
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_bulk_create_entities_partial_failure() {
        let hg = Hypergraph::new(test_store());
        let good = make_entity_json("Alice", "Actor");
        let bad = serde_json::json!({"not": "an entity"});
        let entities = vec![good, bad];
        let result = process_bulk_entities(entities, &hg);
        assert_eq!(result.succeeded, 1);
        assert_eq!(result.failed, 1);
        assert!(result.results[0].id.is_some());
        assert!(result.results[1].error.is_some());
    }

    #[test]
    fn test_bulk_create_entities_empty() {
        let hg = Hypergraph::new(test_store());
        let result = process_bulk_entities(vec![], &hg);
        assert_eq!(result.succeeded, 0);
        assert_eq!(result.failed, 0);
        assert!(result.results.is_empty());
    }

    #[test]
    fn test_bulk_create_situations_empty() {
        let hg = Hypergraph::new(test_store());
        let result = process_bulk_situations(vec![], &hg);
        assert_eq!(result.succeeded, 0);
        assert_eq!(result.failed, 0);
        assert!(result.results.is_empty());
    }

    #[test]
    fn test_bulk_result_counts() {
        let hg = Hypergraph::new(test_store());
        let entities = vec![
            make_entity_json("A", "Actor"),
            serde_json::json!({}),
            make_entity_json("B", "Actor"),
            serde_json::json!({"bad": true}),
        ];
        let result = process_bulk_entities(entities, &hg);
        assert_eq!(result.succeeded, 2);
        assert_eq!(result.failed, 2);
        assert_eq!(result.results.len(), 4);
    }

    #[test]
    fn test_bulk_create_entities_large_batch() {
        let hg = Hypergraph::new(test_store());
        let entities: Vec<_> = (0..100)
            .map(|i| make_entity_json(&format!("Entity_{i}"), "Actor"))
            .collect();
        let result = process_bulk_entities(entities, &hg);
        assert_eq!(result.succeeded, 100);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_bulk_create_situations_with_narrative_ids() {
        let hg = Hypergraph::new(test_store());
        let mut sit = make_situation_json("Test scene");
        sit["narrative_id"] = serde_json::json!("test-narrative");
        let result = process_bulk_situations(vec![sit], &hg);
        assert_eq!(result.succeeded, 1);
        let id = result.results[0].id.unwrap();
        let fetched = hg.get_situation(&id).unwrap();
        assert_eq!(fetched.narrative_id, Some("test-narrative".to_string()));
    }
}
