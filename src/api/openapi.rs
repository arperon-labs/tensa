//! OpenAPI 3.0 specification generator for the TENSA REST API.
//!
//! Serves a complete OpenAPI spec at `GET /openapi.json` for AI agent discovery,
//! SDK generation, and documentation.

use std::sync::OnceLock;

use axum::response::IntoResponse;
use serde_json::{json, Map, Value};

static SPEC_JSON: OnceLock<Vec<u8>> = OnceLock::new();

/// GET /openapi.json — Return the OpenAPI 3.0 specification (cached after first call).
pub async fn openapi_json() -> impl IntoResponse {
    let bytes = SPEC_JSON.get_or_init(|| {
        serde_json::to_vec_pretty(&spec()).expect("static OpenAPI spec is valid JSON")
    });
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes.as_slice(),
    )
}

fn op(tag: &str, summary: &str, desc: &str) -> Value {
    json!({
        "tags": [tag],
        "summary": summary,
        "description": desc,
        "responses": {
            "200": { "description": "Success" },
            "400": { "description": "Bad request" },
            "404": { "description": "Not found" },
            "500": { "description": "Internal server error" }
        }
    })
}

fn op_body(tag: &str, summary: &str, desc: &str) -> Value {
    let mut v = op(tag, summary, desc);
    v["requestBody"] = json!({
        "required": true,
        "content": { "application/json": { "schema": { "type": "object" } } }
    });
    v
}

fn op_id(tag: &str, summary: &str, desc: &str) -> Value {
    let mut v = op(tag, summary, desc);
    v["parameters"] = json!([
        { "name": "id", "in": "path", "required": true, "schema": { "type": "string", "format": "uuid" } }
    ]);
    v
}

fn op_id_body(tag: &str, summary: &str, desc: &str) -> Value {
    let mut v = op_body(tag, summary, desc);
    v["parameters"] = json!([
        { "name": "id", "in": "path", "required": true, "schema": { "type": "string", "format": "uuid" } }
    ]);
    v
}

fn op_paginated(tag: &str, summary: &str, desc: &str) -> Value {
    let mut v = op(tag, summary, desc);
    v["parameters"] = json!([
        { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 50, "maximum": 1000 }, "description": "Max items per page" },
        { "name": "after", "in": "query", "schema": { "type": "string" }, "description": "Cursor: ID of last item from previous page" }
    ]);
    v
}

fn add(paths: &mut Map<String, Value>, path: &str, ops: Value) {
    paths.insert(path.to_string(), ops);
}

fn entity_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/entities",
        json!({
            "post": op_body("Entities", "Create entity", "Create an entity (Actor, Location, Artifact, Concept, Organization).")
        }),
    );
    add(
        paths,
        "/entities/{id}",
        json!({
            "get": op_id("Entities", "Get entity", "Retrieve entity by UUID."),
            "delete": op_id("Entities", "Delete entity", "Delete entity and its participations/state versions.")
        }),
    );
    add(
        paths,
        "/entities/merge",
        json!({
            "post": op_body("Entities", "Merge entities", "Merge two entities. Body: {keep_id, absorb_id}.")
        }),
    );
    add(
        paths,
        "/entities/{id}/split",
        json!({
            "post": op_id_body("Entities", "Split entity", "Split entity by moving situations to a new clone.")
        }),
    );
    add(
        paths,
        "/entities/bulk",
        json!({
            "post": op_body("Bulk", "Bulk create entities", "Create multiple entities. Returns per-item results.")
        }),
    );
}

fn situation_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/situations",
        json!({
            "post": op_body("Situations", "Create situation", "Create a situation (temporal hyperedge) with content.")
        }),
    );
    add(
        paths,
        "/situations/{id}",
        json!({
            "get": op_id("Situations", "Get situation", "Retrieve situation by UUID."),
            "delete": op_id("Situations", "Delete situation", "Delete situation and its participations/causal links.")
        }),
    );
    add(
        paths,
        "/situations/{id}/participants",
        json!({
            "get": op_id("Participations", "List participants", "List all entity participations in a situation.")
        }),
    );
    add(
        paths,
        "/situations/bulk",
        json!({
            "post": op_body("Bulk", "Bulk create situations", "Create multiple situations. Returns per-item results.")
        }),
    );
    add(
        paths,
        "/entities/{id}/situations",
        json!({
            "get": op_id("Participations", "List entity situations", "List all situations an entity participates in.")
        }),
    );
    add(
        paths,
        "/participations",
        json!({
            "post": op_body("Participations", "Add participant", "Link entity to situation with role (Protagonist, Antagonist, Witness, Target, Facilitator, Bystander).")
        }),
    );
}

fn query_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/query",
        json!({
            "post": op_body("Query", "Execute TensaQL",
                "Execute TensaQL query/mutation. Supports MATCH, INFER, DISCOVER, EXPLAIN, CREATE, DELETE, UPDATE, aggregation, OR conditions, AT temporal, NEAR vector, SPATIAL geospatial.")
        }),
    );
}

fn job_paths(paths: &mut Map<String, Value>) {
    let mut list_op = op("Jobs", "List jobs", "List inference jobs.");
    list_op["parameters"] = json!([
        { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 50 } },
        { "name": "target_id", "in": "query", "schema": { "type": "string", "format": "uuid" } }
    ]);
    add(
        paths,
        "/jobs",
        json!({
            "post": op_body("Jobs", "Submit inference job", "Submit async job. Types: Causes, Motivation, Game, Counterfactual, MissingEvent, AnomalyDetection."),
            "get": list_op
        }),
    );
    add(
        paths,
        "/jobs/{id}",
        json!({
            "get": op_id("Jobs", "Get job status", "Get job status (Pending/Running/Completed/Failed)."),
            "delete": op_id("Jobs", "Cancel job", "Cancel a pending inference job.")
        }),
    );
    add(
        paths,
        "/jobs/{id}/result",
        json!({
            "get": op_id("Jobs", "Get job result", "Get completed inference result.")
        }),
    );
}

fn narrative_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/narratives",
        json!({
            "post": op_body("Narratives", "Create narrative", "Create narrative metadata with id, title, genre, tags."),
            "get": op_paginated("Narratives", "List narratives", "List narratives with cursor-based pagination.")
        }),
    );
    let mut del = op_id(
        "Narratives",
        "Delete narrative",
        "Delete narrative. ?cascade=true deletes all contents.",
    );
    if let Some(params) = del.get_mut("parameters").and_then(|p| p.as_array_mut()) {
        params.push(json!({ "name": "cascade", "in": "query", "schema": { "type": "boolean" } }));
    }
    add(
        paths,
        "/narratives/{id}",
        json!({
            "get": op_id("Narratives", "Get narrative", "Retrieve narrative by ID."),
            "delete": del
        }),
    );
    add(
        paths,
        "/narratives/{id}/stats",
        json!({
            "get": op_id("Narratives", "Get narrative stats", "Per-narrative entity/situation counts and temporal span.")
        }),
    );
    let mut export = op_id(
        "Narratives",
        "Export narrative",
        "Export: csv, graphml, json, manuscript, report.",
    );
    if let Some(params) = export.get_mut("parameters").and_then(|p| p.as_array_mut()) {
        params.push(json!({
            "name": "format", "in": "query", "required": true,
            "schema": { "type": "string", "enum": ["csv", "graphml", "json", "manuscript", "report"] }
        }));
    }
    add(paths, "/narratives/{id}/export", json!({ "get": export }));
}

fn validation_paths(paths: &mut Map<String, Value>) {
    let mut list = op(
        "Validation",
        "List pending items",
        "List pending HITL validation items.",
    );
    list["parameters"] = json!([
        { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 50 } }
    ]);
    add(paths, "/validation-queue", json!({ "get": list }));
    add(
        paths,
        "/validation-queue/{id}",
        json!({
            "get": op_id("Validation", "Get queue item", "Get validation queue item by UUID.")
        }),
    );
    add(
        paths,
        "/validation-queue/{id}/approve",
        json!({
            "post": op_id_body("Validation", "Approve item", "Approve extraction. Body: {reviewer}.")
        }),
    );
    add(
        paths,
        "/validation-queue/{id}/reject",
        json!({
            "post": op_id_body("Validation", "Reject item", "Reject extraction. Body: {reviewer, notes}.")
        }),
    );
    add(
        paths,
        "/validation-queue/{id}/edit",
        json!({
            "post": op_id_body("Validation", "Edit and approve", "Edit data and approve. Body: {reviewer, edited_data}.")
        }),
    );
}

fn ingestion_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/ingest",
        json!({
            "post": op_body("Ingestion", "Ingest text", "Ingest raw text through LLM extraction pipeline.")
        }),
    );
    add(
        paths,
        "/import/json",
        json!({
            "post": op_body("Import", "Import structured JSON", "Import entities, situations, participations, causal links.")
        }),
    );
    add(
        paths,
        "/import/csv",
        json!({
            "post": op_body("Import", "Import CSV", "Import entities from CSV with column mapping.")
        }),
    );
}

fn source_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/sources",
        json!({
            "post": op_body("Sources", "Register source", "Register intelligence source with trust score and bias profile."),
            "get": op_paginated("Sources", "List sources", "List registered sources with pagination.")
        }),
    );
    add(
        paths,
        "/sources/{id}",
        json!({
            "get": op_id("Sources", "Get source", "Retrieve source by UUID."),
            "put": op_id_body("Sources", "Update source", "Update source properties."),
            "delete": op_id("Sources", "Delete source", "Delete source and its attributions.")
        }),
    );
    add(
        paths,
        "/sources/{source_id}/attributions",
        json!({
            "post": op_body("Sources", "Add attribution", "Link source to entity/situation claim."),
            "get": op("Sources", "List source attributions", "List all attributions from a source.")
        }),
    );
    add(
        paths,
        "/sources/{source_id}/attributions/{target_id}",
        json!({
            "delete": op("Sources", "Remove attribution", "Remove source-to-target attribution link.")
        }),
    );
    add(
        paths,
        "/entities/{id}/attributions",
        json!({
            "get": op_id("Sources", "Get entity attributions", "List all sources attributed to an entity.")
        }),
    );
    add(
        paths,
        "/situations/{id}/attributions",
        json!({
            "get": op_id("Sources", "Get situation attributions", "List all sources attributed to a situation.")
        }),
    );
    add(
        paths,
        "/contentions",
        json!({
            "post": op_body("Sources", "Add contention", "Create contention between contradictory claims.")
        }),
    );
    add(
        paths,
        "/contentions/resolve",
        json!({
            "post": op_body("Sources", "Resolve contention", "Resolve a contention.")
        }),
    );
    add(
        paths,
        "/situations/{id}/contentions",
        json!({
            "get": op_id("Sources", "List contentions", "List contentions involving a situation.")
        }),
    );
    add(
        paths,
        "/entities/{id}/recompute-confidence",
        json!({
            "post": op_id("Sources", "Recompute entity confidence", "Recompute confidence from source attributions.")
        }),
    );
    add(
        paths,
        "/situations/{id}/recompute-confidence",
        json!({
            "post": op_id("Sources", "Recompute situation confidence", "Recompute confidence from source attributions.")
        }),
    );
}

fn project_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/projects",
        json!({
            "post": op_body("Projects", "Create project", "Create project container."),
            "get": op_paginated("Projects", "List projects", "List projects with cursor-based pagination.")
        }),
    );
    let mut del = op_id(
        "Projects",
        "Delete project",
        "Delete project. ?cascade=true cascade-deletes contents.",
    );
    if let Some(params) = del.get_mut("parameters").and_then(|p| p.as_array_mut()) {
        params.push(json!({ "name": "cascade", "in": "query", "schema": { "type": "boolean" } }));
    }
    add(
        paths,
        "/projects/{id}",
        json!({
            "get": op_id("Projects", "Get project", "Retrieve project by ID."),
            "put": op_id_body("Projects", "Update project", "Update project title, description, tags."),
            "delete": del
        }),
    );
    add(
        paths,
        "/projects/{id}/narratives",
        json!({
            "get": op_id("Projects", "List project narratives", "List all narratives in a project.")
        }),
    );
}

fn settings_paths(paths: &mut Map<String, Value>) {
    add(
        paths,
        "/settings/llm",
        json!({
            "get": op("Settings", "Get LLM config", "Get current LLM provider (API keys redacted)."),
            "put": op_body("Settings", "Set LLM config", "Hot-swap LLM provider at runtime.")
        }),
    );
    let mut discover = op(
        "Settings",
        "Discover models",
        "Probe inference server for available models.",
    );
    discover["parameters"] = json!([
        { "name": "url", "in": "query", "required": true, "schema": { "type": "string" } }
    ]);
    add(paths, "/settings/models", json!({ "get": discover }));
    add(
        paths,
        "/settings/ingestion",
        json!({
            "get": op("Settings", "Get ingestion config", "Get ingestion pipeline configuration."),
            "put": op_body("Settings", "Set ingestion config", "Update ingestion pipeline config at runtime.")
        }),
    );
    add(
        paths,
        "/settings/embedding",
        json!({
            "get": op("Settings", "Get embedding info", "Get embedding provider status, dimension, provider name.")
        }),
    );
}

/// Build the full OpenAPI 3.0 specification as JSON.
pub fn spec() -> Value {
    let mut paths = Map::new();
    add(
        &mut paths,
        "/health",
        json!({ "get": op("Health", "Health check", "Returns {status: ok}.") }),
    );
    add(
        &mut paths,
        "/openapi.json",
        json!({ "get": op("Health", "OpenAPI spec", "This document.") }),
    );

    entity_paths(&mut paths);
    situation_paths(&mut paths);
    query_paths(&mut paths);
    job_paths(&mut paths);
    narrative_paths(&mut paths);
    validation_paths(&mut paths);
    ingestion_paths(&mut paths);
    source_paths(&mut paths);
    project_paths(&mut paths);
    settings_paths(&mut paths);

    json!({
        "openapi": "3.0.3",
        "info": {
            "title": "TENSA API",
            "version": "0.9.0",
            "description": "Temporal Narrative Tensor Architecture — multi-fidelity narrative storage, reasoning, and inference engine."
        },
        "servers": [{ "url": "/" }],
        "tags": tags(),
        "paths": Value::Object(paths)
    })
}

fn tags() -> Value {
    json!([
        { "name": "Health" },
        { "name": "Entities", "description": "Entity CRUD (Actor, Location, Artifact, Concept, Organization)" },
        { "name": "Situations", "description": "Situation CRUD (temporal hyperedges)" },
        { "name": "Participations", "description": "Entity-situation links with roles" },
        { "name": "Query", "description": "TensaQL execution (MATCH, INFER, DISCOVER, EXPLAIN)" },
        { "name": "Jobs", "description": "Async inference job management" },
        { "name": "Narratives", "description": "Narrative metadata and export" },
        { "name": "Validation", "description": "Human-in-the-loop review queue" },
        { "name": "Ingestion", "description": "Text/document ingestion via LLM pipeline" },
        { "name": "Import", "description": "Structured JSON/CSV import" },
        { "name": "Bulk", "description": "Bulk entity/situation creation" },
        { "name": "Sources", "description": "Source intelligence, attributions, contentions" },
        { "name": "Projects", "description": "Project containers for narratives" },
        { "name": "Settings", "description": "LLM, ingestion, and embedding configuration" }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_is_valid_openapi() {
        let s = spec();
        assert_eq!(s["openapi"], "3.0.3");
        assert_eq!(s["info"]["title"], "TENSA API");
        assert!(s["paths"].is_object());
    }

    #[test]
    fn test_spec_contains_key_paths() {
        let s = spec();
        let paths = s["paths"].as_object().unwrap();
        assert!(paths.contains_key("/entities"));
        assert!(paths.contains_key("/situations"));
        assert!(paths.contains_key("/narratives"));
        assert!(paths.contains_key("/query"));
        assert!(paths.contains_key("/jobs"));
        assert!(paths.contains_key("/sources"));
        assert!(paths.contains_key("/projects"));
        assert!(paths.contains_key("/health"));
        assert!(paths.contains_key("/openapi.json"));
    }

    #[test]
    fn test_spec_pagination_params() {
        let s = spec();
        let narratives_get = &s["paths"]["/narratives"]["get"];
        let params = narratives_get["parameters"].as_array().unwrap();
        let names: Vec<&str> = params.iter().filter_map(|p| p["name"].as_str()).collect();
        assert!(names.contains(&"limit"));
        assert!(names.contains(&"after"));
    }

    #[test]
    fn test_spec_query_endpoint_has_body() {
        let s = spec();
        let query_post = &s["paths"]["/query"]["post"];
        assert!(query_post["requestBody"].is_object());
    }
}
