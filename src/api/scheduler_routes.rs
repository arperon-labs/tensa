//! Scheduler + Discovery + Sync + Reports + MCP Health API routes (Sprint D8).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;

// ─── Scheduler ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateTaskRequest {
    pub task_type: String,
    pub schedule: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub config: serde_json::Value,
}

fn default_true() -> bool {
    true
}

pub async fn list_tasks(State(state): State<Arc<AppState>>) -> Response {
    match crate::scheduler::engine::list_tasks(&state.hypergraph) {
        Ok(tasks) => json_ok(&tasks),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn create_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateTaskRequest>,
) -> Response {
    let task_type = match req
        .task_type
        .parse::<crate::scheduler::types::ScheduledTaskType>()
    {
        Ok(t) => t,
        Err(e) => return error_response(e).into_response(),
    };
    let schedule = match crate::scheduler::types::parse_schedule(&req.schedule) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };
    let task = crate::scheduler::types::ScheduledTask {
        id: Uuid::now_v7(),
        task_type,
        schedule,
        enabled: req.enabled,
        last_run: None,
        last_result: None,
        config: req.config,
        created_at: chrono::Utc::now(),
    };
    match crate::scheduler::engine::create_task(&state.hypergraph, &task) {
        Ok(id) => json_ok(&serde_json::json!({ "id": id.to_string(), "status": "created" })),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn delete_task(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    match crate::scheduler::engine::delete_task(&state.hypergraph, &uuid) {
        Ok(_) => json_ok(&serde_json::json!({ "status": "deleted" })),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn get_task_history(
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
    match crate::scheduler::engine::get_history(&state.hypergraph, &uuid, 50) {
        Ok(history) => json_ok(&history),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn run_task_now(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    match crate::scheduler::engine::get_task(&state.hypergraph, &uuid) {
        Ok(Some(_)) => {
            let result = crate::scheduler::types::TaskResult::Success {
                duration_ms: 0,
                summary: "Manual trigger -- task queued".to_string(),
            };
            let _ = crate::scheduler::engine::mark_task_run(&state.hypergraph, &uuid, result);
            json_ok(&serde_json::json!({ "status": "triggered", "task_id": id }))
        }
        Ok(None) => {
            error_response(TensaError::NotFound(format!("Task {} not found", id))).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Discovery ────────────────────────────────────────────────

pub async fn list_discovery_candidates(State(state): State<Arc<AppState>>) -> Response {
    match crate::ingestion::discovery::list_candidates(&state.hypergraph) {
        Ok(c) => json_ok(&c),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn approve_discovery(
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
    match crate::ingestion::discovery::approve_candidate(&state.hypergraph, &uuid) {
        Ok(_) => json_ok(&serde_json::json!({ "status": "approved" })),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn reject_discovery(
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
    match crate::ingestion::discovery::reject_candidate(&state.hypergraph, &uuid) {
        Ok(_) => json_ok(&serde_json::json!({ "status": "rejected" })),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct PolicyRequest {
    pub policy: String,
}

pub async fn get_discovery_policy(State(state): State<Arc<AppState>>) -> Response {
    match crate::ingestion::discovery::load_policy(&state.hypergraph) {
        Ok(p) => json_ok(&p),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn set_discovery_policy(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PolicyRequest>,
) -> Response {
    let policy = match req.policy.to_lowercase().as_str() {
        "manual" => crate::ingestion::discovery::DiscoveryPolicy::Manual,
        "auto_approve_co_amplification" | "autoapproveco" => {
            crate::ingestion::discovery::DiscoveryPolicy::AutoApproveCoAmplification
        }
        "auto_approve_all" | "autoapproveall" => {
            crate::ingestion::discovery::DiscoveryPolicy::AutoApproveAll
        }
        other => {
            return error_response(TensaError::InvalidQuery(format!(
                "Unknown policy: {}",
                other
            )))
            .into_response()
        }
    };
    match crate::ingestion::discovery::save_policy(&state.hypergraph, &policy) {
        Ok(_) => json_ok(&serde_json::json!({ "status": "updated" })),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Fact-Check Sync ──────────────────────────────────────────

pub async fn trigger_sync(State(state): State<Arc<AppState>>) -> Response {
    let sources = crate::claims::sync::default_sources();
    match crate::claims::sync::execute_sync(&state.hypergraph, &sources, &["en".to_string()]) {
        Ok(results) => json_ok(&results),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn sync_history(State(state): State<Arc<AppState>>) -> Response {
    match crate::claims::sync::list_sync_history(&state.hypergraph, 50) {
        Ok(history) => json_ok(&history),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn sync_sources(State(_state): State<Arc<AppState>>) -> Response {
    json_ok(&crate::claims::sync::default_sources())
}

// ─── Reports ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct GenerateReportRequest {
    #[serde(default = "default_hours")]
    pub hours: u64,
}

fn default_hours() -> u64 {
    24
}

pub async fn generate_report(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateReportRequest>,
) -> Response {
    let end = chrono::Utc::now();
    let start = end - chrono::Duration::hours(req.hours as i64);
    match crate::export::situation_report::generate_situation_report(&state.hypergraph, start, end)
    {
        Ok(report) => json_ok(&report),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn list_reports(State(state): State<Arc<AppState>>) -> Response {
    match crate::export::situation_report::list_reports(&state.hypergraph, 50) {
        Ok(reports) => json_ok(&reports),
        Err(e) => error_response(e).into_response(),
    }
}

pub async fn get_report(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(e) => {
            return error_response(TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
                .into_response()
        }
    };
    match crate::export::situation_report::load_report(&state.hypergraph, &uuid) {
        Ok(Some(r)) => json_ok(&r),
        Ok(None) => {
            error_response(TensaError::NotFound(format!("Report {} not found", id))).into_response()
        }
        Err(e) => error_response(e).into_response(),
    }
}

// ─── MCP Source Health ────────────────────────────────────────

pub async fn list_source_health(State(state): State<Arc<AppState>>) -> Response {
    match crate::disinfo::orchestrator::list_source_health(&state.hypergraph) {
        Ok(h) => json_ok(&h),
        Err(e) => error_response(e).into_response(),
    }
}
