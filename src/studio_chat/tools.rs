//! Chat toolset — small hand-rolled tool surface for Phase 3.
//!
//! Each tool has a name, JSON-schema-ish description, and a handler that
//! runs against the shared `AppState`. Read-only tools execute immediately.
//! Mutating tools run only after the confirmation gate returns `Approve`.
//!
//! Phase 3 keeps the surface intentionally narrow (~12 tools). Phase 4 will
//! supplement this with external MCP servers (`rmcp` client proxies).

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::api::server::AppState;
use crate::error::{Result, TensaError};
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::types::Narrative;
use crate::types::{ContentType, Entity, EntityType, MaturityLevel, NarrativeLevel, Situation};

/// Static metadata the LLM sees when deciding which tool to invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON schema-ish object describing accepted args. We store it as a
    /// `serde_json::Value` rather than a typed schema since each LLM
    /// provider serializes it slightly differently.
    pub args_schema: Value,
}

/// Render the entire toolset as a single Markdown block for inclusion in
/// the system prompt. Phase 3 uses prompt-encoded tool calls (JSON inside
/// a fenced block) so this doubles as the instructions.
pub fn tool_catalog_markdown(specs: &[ToolSpec]) -> String {
    let mut out = String::from(
        "## Tools\n\
         You can call the following tools by emitting a fenced code block with the\n\
         `tensa-tool` language tag. Example:\n\n\
         ```tensa-tool\n{\"tool\": \"list_narratives\", \"args\": {}}\n```\n\n\
         Emit **at most one** tool call per assistant turn. After the tool runs the\n\
         result is returned as a `[tool:<name>]` message in the next turn; at that\n\
         point either call another tool or produce the final answer.\n\n\
         Tools marked **Mutating** require the user to approve via a confirmation\n\
         card before they run — that delay is normal. Never call a mutating tool\n\
         without a clear user request.\n\n",
    );
    for spec in specs {
        let class = if super::confirm::classify(&spec.name) == super::confirm::ToolClass::Confirm {
            " · **Mutating**"
        } else {
            ""
        };
        out.push_str(&format!(
            "- `{}`{} — {}\n  Args: `{}`\n",
            spec.name,
            class,
            spec.description,
            serde_json::to_string(&spec.args_schema).unwrap_or_else(|_| "{}".into())
        ));
    }
    out
}

/// All tools supported by Phase 3.
pub fn catalog() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "list_narratives".into(),
            description: "List all narratives (id, title, counts). Use this to discover what data is available.".into(),
            args_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "get_narrative".into(),
            description: "Fetch a single narrative's full metadata by id.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["id"],
                "properties": { "id": { "type": "string" } }
            }),
        },
        ToolSpec {
            name: "list_entities".into(),
            description: "List entities, optionally filtered by narrative_id or entity_type. Default limit 25, max 200.".into(),
            args_schema: json!({
                "type": "object",
                "properties": {
                    "narrative_id": { "type": "string" },
                    "entity_type": { "type": "string", "enum": ["Actor","Location","Artifact","Concept","Organization"] },
                    "limit": { "type": "integer", "default": 25 }
                }
            }),
        },
        ToolSpec {
            name: "get_entity".into(),
            description: "Fetch a single entity's full record by UUID.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["id"],
                "properties": { "id": { "type": "string", "description": "entity UUID" } }
            }),
        },
        ToolSpec {
            name: "list_situations".into(),
            description: "List situations, optionally filtered by narrative_id. Default limit 25, max 200.".into(),
            args_schema: json!({
                "type": "object",
                "properties": {
                    "narrative_id": { "type": "string" },
                    "limit": { "type": "integer", "default": 25 }
                }
            }),
        },
        ToolSpec {
            name: "get_situation".into(),
            description: "Fetch a single situation's full record by UUID.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["id"],
                "properties": { "id": { "type": "string", "description": "situation UUID" } }
            }),
        },
        ToolSpec {
            name: "query_tensaql".into(),
            description: "Execute a read-only TensaQL MATCH query. Write-shape queries (INFER, DISCOVER, INGEST, EXPORT) are rejected.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["query"],
                "properties": { "query": { "type": "string" } }
            }),
        },
        ToolSpec {
            name: "navigate_to".into(),
            description: "Ask the Studio UI to navigate to a route (e.g. `/dossier/{uuid}`, `/graph`, `/timeline`). The user remains in the chat panel.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["route"],
                "properties": { "route": { "type": "string" } }
            }),
        },
        ToolSpec {
            name: "show_toast".into(),
            description: "Surface a short status message in the Studio toaster. Useful to highlight an action without bloating the chat stream.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": { "type": "string" },
                    "level": { "type": "string", "enum": ["info","success","warn","error"], "default": "info" }
                }
            }),
        },
        ToolSpec {
            name: "create_entity".into(),
            description: "Create a new entity. Mutating: requires user confirmation.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["entity_type", "properties"],
                "properties": {
                    "entity_type": { "type": "string", "enum": ["Actor","Location","Artifact","Concept","Organization"] },
                    "properties": { "type": "object" },
                    "narrative_id": { "type": "string" },
                    "confidence": { "type": "number", "default": 0.7 }
                }
            }),
        },
        ToolSpec {
            name: "create_situation".into(),
            description: "Create a minimal situation with optional start/end ISO-8601 times. Mutating: requires user confirmation.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["narrative_level", "summary"],
                "properties": {
                    "summary": { "type": "string", "description": "plain-text summary used as raw_content" },
                    "narrative_level": { "type": "string", "enum": ["Story","Arc","Sequence","Scene","Beat","Event"] },
                    "narrative_id": { "type": "string" },
                    "start": { "type": "string", "description": "RFC 3339 timestamp" },
                    "end": { "type": "string", "description": "RFC 3339 timestamp" },
                    "confidence": { "type": "number", "default": 0.7 }
                }
            }),
        },
        ToolSpec {
            name: "create_narrative".into(),
            description: "Create a new narrative container. Mutating: requires user confirmation.".into(),
            args_schema: json!({
                "type": "object",
                "required": ["id", "title"],
                "properties": {
                    "id": { "type": "string", "description": "url-safe slug" },
                    "title": { "type": "string" },
                    "genre": { "type": "string" },
                    "description": { "type": "string" }
                }
            }),
        },
    ]
}

/// Result shape returned by every tool. `display` is a short human summary
/// the UI can render in a `[tool:name]` message; `full` is the JSON the
/// LLM sees on the next turn.
#[derive(Debug, Clone, Serialize)]
pub struct ToolOutput {
    pub display: String,
    pub full: Value,
}

impl ToolOutput {
    fn ok(display: impl Into<String>, full: Value) -> Self {
        Self {
            display: display.into(),
            full,
        }
    }
}

/// Execute a tool by name against shared state. Async because some tools
/// (ask-RAG, queries hitting the inference engine, MCP proxies) need the
/// tokio runtime.
pub async fn dispatch(state: &Arc<AppState>, tool: &str, args: Value) -> Result<ToolOutput> {
    // Namespaced MCP tool: `{server_name}__{tool_name}` — dispatch to the
    // corresponding McpProxy. Catches this before the local match so a
    // malicious local tool can't shadow a proxy name.
    if let Some((server, sub)) = tool.split_once(super::mcp_proxy::NAMESPACE_SEP) {
        return dispatch_mcp_proxy(state, server, sub, args).await;
    }
    match tool {
        "list_narratives" => list_narratives(state),
        "get_narrative" => get_narrative(state, args),
        "list_entities" => list_entities(state, args),
        "get_entity" => get_entity(state, args),
        "list_situations" => list_situations(state, args),
        "get_situation" => get_situation(state, args),
        "query_tensaql" => query_tensaql(state, args),
        "navigate_to" => ui_navigate(args),
        "show_toast" => ui_toast(args),
        "create_entity" => create_entity(state, args),
        "create_situation" => create_situation(state, args),
        "create_narrative" => create_narrative(state, args),
        _ => Err(TensaError::InvalidInput(format!("unknown tool `{}`", tool))),
    }
}

async fn dispatch_mcp_proxy(
    state: &Arc<AppState>,
    server: &str,
    tool: &str,
    args: Value,
) -> Result<ToolOutput> {
    let proxy = state
        .chat_mcp_proxies
        .get(server)
        .await
        .ok_or_else(|| TensaError::InvalidInput(format!("unknown MCP server `{}`", server)))?;
    if !proxy.enabled() {
        return Err(TensaError::InvalidInput(format!(
            "MCP server `{}` is disabled",
            server
        )));
    }
    let output = proxy.call_tool(tool, args).await?;
    Ok(ToolOutput {
        display: format!("mcp `{}__{}` returned", server, tool),
        full: output,
    })
}

// ─── Argument helpers ───────────────────────────────────────

fn str_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(|v| v.as_str())
        .ok_or_else(|| TensaError::InvalidInput(format!("missing string arg `{}`", name)))
}

fn opt_str_arg<'a>(args: &'a Value, name: &str) -> Option<&'a str> {
    args.get(name)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
}

fn uuid_arg(args: &Value, name: &str) -> Result<Uuid> {
    let raw = str_arg(args, name)?;
    Uuid::parse_str(raw)
        .map_err(|e| TensaError::InvalidInput(format!("bad UUID in `{}`: {}", name, e)))
}

fn clamp_limit(args: &Value, default: usize, max: usize) -> usize {
    args.get("limit")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).min(max).max(1))
        .unwrap_or(default)
}

// ─── Read-only tool implementations ─────────────────────────

fn list_narratives(state: &Arc<AppState>) -> Result<ToolOutput> {
    let registry = NarrativeRegistry::new(state.hypergraph.store_arc());
    let all = registry.list(None, None)?;
    let brief: Vec<Value> = all
        .iter()
        .map(|n| {
            json!({
                "id": n.id,
                "title": n.title,
                "genre": n.genre,
                "entity_count": n.entity_count,
                "situation_count": n.situation_count,
            })
        })
        .collect();
    Ok(ToolOutput::ok(
        format!("{} narrative(s)", brief.len()),
        json!({ "narratives": brief }),
    ))
}

fn get_narrative(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let id = str_arg(&args, "id")?;
    let registry = NarrativeRegistry::new(state.hypergraph.store_arc());
    let n = registry.get(id)?;
    Ok(ToolOutput::ok(
        format!("narrative `{}`: {}", n.id, n.title),
        serde_json::to_value(&n)?,
    ))
}

fn list_entities(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let limit = clamp_limit(&args, 25, 200);
    let entities: Vec<Entity> = if let Some(nid) = opt_str_arg(&args, "narrative_id") {
        state.hypergraph.list_entities_by_narrative(nid)?
    } else if let Some(t) = opt_str_arg(&args, "entity_type") {
        let ty: EntityType = serde_json::from_value(json!(t))
            .map_err(|e| TensaError::InvalidInput(format!("bad entity_type: {}", e)))?;
        state.hypergraph.list_entities_by_type(&ty)?
    } else {
        state
            .hypergraph
            .list_entities_by_maturity(MaturityLevel::Candidate)?
    };
    let brief: Vec<Value> = entities
        .iter()
        .take(limit)
        .map(|e| {
            json!({
                "id": e.id,
                "entity_type": e.entity_type,
                "name": e.properties.get("name"),
                "confidence": e.confidence,
                "narrative_id": e.narrative_id,
            })
        })
        .collect();
    Ok(ToolOutput::ok(
        format!("{} entit(y|ies)", brief.len()),
        json!({ "entities": brief, "truncated": entities.len() > limit }),
    ))
}

fn get_entity(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let id = uuid_arg(&args, "id")?;
    let e = state.hypergraph.get_entity(&id)?;
    let label = e
        .properties
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("(unnamed)")
        .to_string();
    Ok(ToolOutput::ok(
        format!("{} {}", label, e.id),
        serde_json::to_value(&e)?,
    ))
}

fn list_situations(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let limit = clamp_limit(&args, 25, 200);
    let sits: Vec<Situation> = if let Some(nid) = opt_str_arg(&args, "narrative_id") {
        state.hypergraph.list_situations_by_narrative(nid)?
    } else {
        state
            .hypergraph
            .list_situations_by_maturity(MaturityLevel::Candidate)?
    };
    let brief: Vec<Value> = sits
        .iter()
        .take(limit)
        .map(|s| {
            let summary: String = s
                .raw_content
                .iter()
                .find(|b| matches!(b.content_type, ContentType::Text))
                .map(|b| b.content.clone())
                .unwrap_or_default();
            json!({
                "id": s.id,
                "narrative_level": s.narrative_level,
                "confidence": s.confidence,
                "narrative_id": s.narrative_id,
                "summary": summary.chars().take(160).collect::<String>(),
            })
        })
        .collect();
    Ok(ToolOutput::ok(
        format!("{} situation(s)", brief.len()),
        json!({ "situations": brief, "truncated": sits.len() > limit }),
    ))
}

fn get_situation(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let id = uuid_arg(&args, "id")?;
    let s = state.hypergraph.get_situation(&id)?;
    Ok(ToolOutput::ok(
        format!("situation {}", s.id),
        serde_json::to_value(&s)?,
    ))
}

fn query_tensaql(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    use crate::query::{executor, parser, planner};
    let q = str_arg(&args, "query")?.trim();
    // Reject mutating / expensive shapes. Parsing catches syntax errors too.
    let upper = q.to_ascii_uppercase();
    for banned in ["INFER ", "DISCOVER ", "EXPORT ", "INGEST ", "ASK ", "TUNE "] {
        if upper.contains(banned) {
            return Err(TensaError::InvalidInput(format!(
                "query_tensaql accepts MATCH only; use the dedicated tool for `{}`",
                banned.trim()
            )));
        }
    }
    let parsed = parser::parse_query(q)?;
    let plan = planner::plan_query(&parsed)?;
    let tree = state
        .interval_tree
        .read()
        .map_err(|_| TensaError::Internal("interval_tree lock poisoned".into()))?;
    let rows = executor::execute(&plan, &state.hypergraph, &tree)?;
    Ok(ToolOutput::ok(
        format!("{} row(s)", rows.len()),
        json!({ "rows": rows, "plan": plan }),
    ))
}

fn ui_navigate(args: Value) -> Result<ToolOutput> {
    let route = str_arg(&args, "route")?;
    if !route.starts_with('/') {
        return Err(TensaError::InvalidInput("route must begin with `/`".into()));
    }
    Ok(ToolOutput::ok(
        format!("navigating to {}", route),
        json!({ "ui_action": "navigate", "route": route }),
    ))
}

fn ui_toast(args: Value) -> Result<ToolOutput> {
    let message = str_arg(&args, "message")?;
    let level = opt_str_arg(&args, "level").unwrap_or("info");
    Ok(ToolOutput::ok(
        format!("toast: {}", message),
        json!({
            "ui_action": "toast",
            "message": message,
            "level": level,
        }),
    ))
}

// ─── Mutating tool implementations ──────────────────────────

fn create_entity(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    // Let serde do the heavy lifting: build JSON with the fields the caller
    // supplied + mandatory defaults and deserialize into Entity. Avoids
    // enumerating every struct field here (some are writer-only additions).
    let entity_type_str = str_arg(&args, "entity_type")?;
    let properties = args
        .get("properties")
        .cloned()
        .ok_or_else(|| TensaError::InvalidInput("missing `properties` object".into()))?;
    let confidence = args
        .get("confidence")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.7);
    let now = chrono::Utc::now();
    let mut raw = json!({
        "id": Uuid::now_v7().to_string(),
        "entity_type": entity_type_str,
        "properties": properties,
        "beliefs": null,
        "embedding": null,
        "maturity": "Candidate",
        "confidence": confidence,
        "provenance": [{
            "source_type": "chat",
            "description": "Created via Studio chat tool",
            "timestamp": now.to_rfc3339(),
        }],
        "extraction_method": "HumanEntered",
        "created_at": now.to_rfc3339(),
        "updated_at": now.to_rfc3339(),
    });
    if let Some(nid) = opt_str_arg(&args, "narrative_id") {
        raw["narrative_id"] = Value::String(nid.into());
    }
    let entity: Entity = serde_json::from_value(raw)
        .map_err(|e| TensaError::InvalidInput(format!("entity build failed: {}", e)))?;
    let id = state.hypergraph.create_entity(entity)?;
    Ok(ToolOutput::ok(
        format!("created entity {}", id),
        json!({ "id": id }),
    ))
}

fn create_situation(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let summary = str_arg(&args, "summary")?;
    let level_str = str_arg(&args, "narrative_level")?;
    let narrative_level: NarrativeLevel = serde_json::from_value(json!(level_str))
        .map_err(|e| TensaError::InvalidInput(format!("bad narrative_level: {}", e)))?;
    let confidence = args
        .get("confidence")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.7);
    let start = opt_str_arg(&args, "start");
    let end = opt_str_arg(&args, "end");
    let now = chrono::Utc::now();
    let mut raw = json!({
        "id": Uuid::now_v7().to_string(),
        "temporal": {
            "start": start,
            "end": end,
            "granularity": "Approximate",
            "relations": [],
        },
        "spatial": null,
        "game_structure": null,
        "causes": [],
        "deterministic": null,
        "probabilistic": null,
        "embedding": null,
        "raw_content": [{ "type": "text", "text": summary }],
        "narrative_level": narrative_level,
        "discourse": null,
        "maturity": "Candidate",
        "confidence": confidence,
        "extraction_method": "HumanEntered",
        "created_at": now.to_rfc3339(),
        "updated_at": now.to_rfc3339(),
    });
    if let Some(nid) = opt_str_arg(&args, "narrative_id") {
        raw["narrative_id"] = Value::String(nid.into());
    }
    let situation: Situation = serde_json::from_value(raw)
        .map_err(|e| TensaError::InvalidInput(format!("situation build failed: {}", e)))?;
    let id = state.hypergraph.create_situation(situation)?;
    Ok(ToolOutput::ok(
        format!("created situation {}", id),
        json!({ "id": id }),
    ))
}

fn create_narrative(state: &Arc<AppState>, args: Value) -> Result<ToolOutput> {
    let id = str_arg(&args, "id")?.to_string();
    let title = str_arg(&args, "title")?.to_string();
    let genre = opt_str_arg(&args, "genre").map(|s| s.to_string());
    let description = opt_str_arg(&args, "description").map(|s| s.to_string());
    let now = chrono::Utc::now();
    let n = Narrative {
        id: id.clone(),
        title,
        genre,
        tags: vec![],
        source: None,
        project_id: None,
        description,
        authors: vec![],
        language: None,
        publication_date: None,
        cover_url: None,
        custom_properties: Default::default(),
        entity_count: 0,
        situation_count: 0,
        created_at: now,
        updated_at: now,
    };
    let registry = NarrativeRegistry::new(state.hypergraph.store_arc());
    let new_id = registry.create(n)?;
    Ok(ToolOutput::ok(
        format!("created narrative `{}`", new_id),
        json!({ "id": new_id }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_nonempty_and_names_are_unique() {
        let specs = catalog();
        assert!(specs.len() >= 10);
        let mut seen = std::collections::HashSet::new();
        for s in &specs {
            assert!(
                seen.insert(s.name.clone()),
                "duplicate tool name: {}",
                s.name
            );
        }
    }

    #[test]
    fn tool_catalog_markdown_marks_mutating_tools() {
        let md = tool_catalog_markdown(&catalog());
        assert!(md.contains("list_narratives"));
        assert!(md.contains("**Mutating**"));
        assert!(md.contains("tensa-tool"));
    }

    #[test]
    fn uuid_arg_parses() {
        let uuid = Uuid::now_v7();
        let args = json!({ "id": uuid.to_string() });
        assert_eq!(uuid_arg(&args, "id").unwrap(), uuid);
    }

    #[test]
    fn uuid_arg_rejects_garbage() {
        let args = json!({ "id": "not-a-uuid" });
        assert!(matches!(
            uuid_arg(&args, "id").unwrap_err(),
            TensaError::InvalidInput(_)
        ));
    }

    #[test]
    fn clamp_limit_honors_cap() {
        assert_eq!(clamp_limit(&json!({}), 25, 200), 25);
        assert_eq!(clamp_limit(&json!({ "limit": 10 }), 25, 200), 10);
        assert_eq!(clamp_limit(&json!({ "limit": 500 }), 25, 200), 200);
        assert_eq!(clamp_limit(&json!({ "limit": 0 }), 25, 200), 1);
    }
}
