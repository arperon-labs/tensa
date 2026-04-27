//! Axum handlers for `/studio/chat/*`.
//!
//! Phase 1 shipped session CRUD + an SSE endpoint that echoed user messages.
//! Phase 2 (v0.61) swaps the echo loop for a real LLM turn, loading the
//! active skill bundles (studio-ui today), prior message history, and the
//! chat extractor resolved from state. Tool-calling + MCP proxying arrive in
//! later phases; the wire format already reserves space for them.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::ingestion::llm::ApiMessage;
use crate::ingestion::llm::NarrativeExtractor;

use super::skills::SkillRegistry;
use super::store::ChatStore;
use super::types::{
    ChatEvent, ChatTurnRequest, ContentPart, Message, MessageRole, Scope, SessionMeta,
    DEFAULT_USER, DEFAULT_WORKSPACE,
};

const WORKSPACE_HEADER: &str = "x-tensa-workspace";
const USER_HEADER: &str = "x-tensa-user";
const SSE_CHANNEL_CAPACITY: usize = 32;
/// Trim context to the last N turns before sending to the LLM. Large enough
/// to keep recent scroll-back but small enough that we don't blow past
/// provider token limits.
const MAX_HISTORY_TURNS: usize = 20;
/// Words to emit per SSE Token event when streaming a non-streaming LLM
/// response. Phase 3 switches to real provider streaming.
const TOKEN_WORDS_PER_EVENT: usize = 1;
/// Delay between simulated Token events — just enough to paint incrementally
/// without stalling short responses.
const TOKEN_DELAY: Duration = Duration::from_millis(18);
/// Maximum number of LLM round-trips per user turn. Prevents runaway
/// tool-call loops when the model refuses to produce a final answer.
const MAX_TOOL_ITERATIONS: u32 = 6;
/// How long we wait for a user confirmation before rejecting the tool.
const CONFIRM_TIMEOUT: Duration = Duration::from_secs(300);

// ─── Helpers ────────────────────────────────────────────────

fn scope_from_headers(headers: &HeaderMap) -> Scope {
    let workspace = headers
        .get(WORKSPACE_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_WORKSPACE)
        .to_string();
    let user = headers
        .get(USER_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_USER)
        .to_string();
    Scope::new(workspace, user)
}

fn chat_store(state: &Arc<AppState>) -> ChatStore {
    ChatStore::new(state.root_store.clone())
}

fn ensure_session(
    store: &ChatStore,
    scope: &Scope,
    session_id: Option<String>,
    first_user_msg: &str,
) -> Result<SessionMeta, TensaError> {
    if let Some(id) = session_id {
        if let Some(existing) = store.get_session(scope, &id)? {
            return Ok(existing);
        }
        // Explicit id that doesn't yet exist — create it.
        let meta = SessionMeta::new(scope, id, autogen_title(first_user_msg));
        store.create_session(scope, &meta)?;
        return Ok(meta);
    }
    let id = Uuid::now_v7().simple().to_string();
    let meta = SessionMeta::new(scope, id, autogen_title(first_user_msg));
    store.create_session(scope, &meta)?;
    Ok(meta)
}

fn autogen_title(first_user_msg: &str) -> String {
    let trimmed = first_user_msg.trim();
    if trimmed.is_empty() {
        return "New chat".to_string();
    }
    const LIMIT: usize = 60;
    if trimmed.chars().count() <= LIMIT {
        trimmed.to_string()
    } else {
        let truncated: String = trimmed.chars().take(LIMIT).collect();
        format!("{}…", truncated)
    }
}

fn to_sse_event(ev: &ChatEvent) -> Event {
    Event::default().event(ev.name()).data(ev.to_sse_data())
}

// ─── Session CRUD ───────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
pub struct CreateSessionBody {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub active_skills: Option<Vec<String>>,
    #[serde(default)]
    pub narrative_scope: Option<String>,
}

/// `POST /studio/chat/sessions` — Create an empty session.
///
/// Body (optional): `{ "title": "...", "active_skills": [...], "narrative_scope": "..." }`
pub async fn create_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Option<Json<CreateSessionBody>>,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let store = chat_store(&state);
    let id = Uuid::now_v7().simple().to_string();
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let title = body
        .title
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "New chat".to_string());
    let mut meta = SessionMeta::new(&scope, id, title);
    if let Some(skills) = body.active_skills {
        if !skills.is_empty() {
            meta.active_skills = skills;
        }
    }
    if let Some(narr) = body.narrative_scope.filter(|s| !s.is_empty()) {
        meta.narrative_scope = Some(narr);
    }
    match store.create_session(&scope, &meta) {
        Ok(_) => json_ok(&meta),
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Serialize)]
pub struct SessionDetail {
    pub meta: SessionMeta,
    pub messages: Vec<Message>,
}

/// `GET /studio/chat/sessions` — List sessions for the caller (scoped).
pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let store = chat_store(&state);
    match store.list_sessions(&scope) {
        Ok(list) => json_ok(&list),
        Err(e) => error_response(e).into_response(),
    }
}

/// `GET /studio/chat/sessions/:id` — Session + full message history.
pub async fn get_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let store = chat_store(&state);
    let meta = match store.get_session(&scope, &id) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("session '{}' not found", id)})),
            )
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    let messages = match store.list_messages(&scope, &id) {
        Ok(m) => m,
        Err(e) => return error_response(e).into_response(),
    };
    json_ok(&SessionDetail { meta, messages })
}

#[derive(Deserialize)]
pub struct PatchSession {
    pub title: Option<String>,
    pub archived: Option<bool>,
    pub active_skills: Option<Vec<String>>,
    pub narrative_scope: Option<Option<String>>,
    pub model_override: Option<Option<String>>,
}

/// `PATCH /studio/chat/sessions/:id` — Rename / archive / adjust skills.
pub async fn patch_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<PatchSession>,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let store = chat_store(&state);
    let mut meta = match store.get_session(&scope, &id) {
        Ok(Some(m)) => m,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("session '{}' not found", id)})),
            )
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    if let Some(t) = body.title {
        meta.title = t;
    }
    if let Some(a) = body.archived {
        meta.archived = a;
    }
    if let Some(s) = body.active_skills {
        meta.active_skills = s;
    }
    if let Some(n) = body.narrative_scope {
        meta.narrative_scope = n;
    }
    if let Some(m) = body.model_override {
        meta.model_override = m;
    }
    meta.updated_at = chrono::Utc::now();
    match store.update_session(&scope, &meta) {
        Ok(_) => json_ok(&meta),
        Err(e) => error_response(e).into_response(),
    }
}

/// `DELETE /studio/chat/sessions/:id` — Delete session + all its messages.
pub async fn delete_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let store = chat_store(&state);
    match store.delete_session(&scope, &id) {
        Ok(true) => json_ok(&json!({"deleted": id})),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("session '{}' not found", id)})),
        )
            .into_response(),
        Err(e) => error_response(e).into_response(),
    }
}

// ─── Confirmation + Stop (Phase 1 stubs) ────────────────────

#[derive(Debug, Deserialize)]
pub struct ConfirmBody {
    pub call_id: String,
    pub decision: String,
}

/// `POST /studio/chat/sessions/:id/confirm` — approve or reject a pending
/// mutating tool call that the harness is currently parked on.
pub async fn confirm_tool_call(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(session_id): Path<String>,
    Json(body): Json<ConfirmBody>,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let decision = match body.decision.as_str() {
        "approve" => super::confirm::ConfirmDecision::Approve,
        "reject" => super::confirm::ConfirmDecision::Reject,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!(
                        "decision must be \"approve\" or \"reject\", got `{}`",
                        other
                    ),
                })),
            )
                .into_response()
        }
    };
    let key = super::confirm::PendingKey::new(&scope, &session_id, &body.call_id);
    if state.chat_confirm_gate.settle(key, decision) {
        json_ok(&json!({ "status": "ok" }))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": "no pending confirmation for that call_id (already settled or unknown)",
            })),
        )
            .into_response()
    }
}

pub async fn stop_turn(Path(_id): Path<String>) -> axum::response::Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": "cancellation not yet wired to the harness",
        })),
    )
        .into_response()
}

// ─── MCP server CRUD (Phase 4) ──────────────────────────────

/// `GET /studio/chat/mcp-servers` — list configured servers + live status.
pub async fn list_mcp_servers(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let configs = super::mcp_proxy::load_persisted_mcp_servers(state.root_store.as_ref());
    let statuses = state.chat_mcp_proxies.statuses().await;
    let by_name: std::collections::HashMap<_, _> =
        statuses.into_iter().map(|s| (s.name.clone(), s)).collect();
    let out: Vec<serde_json::Value> = configs
        .iter()
        .map(|cfg| {
            let status = by_name.get(&cfg.name);
            serde_json::json!({
                "config": cfg,
                "running": status.map(|s| s.running).unwrap_or(false),
                "tool_count": status.map(|s| s.tool_count).unwrap_or(0),
                "last_error": status.and_then(|s| s.last_error.clone()),
            })
        })
        .collect();
    json_ok(&out)
}

#[derive(Debug, Deserialize)]
pub struct AddMcpServerBody {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    #[serde(default = "mcp_default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub description: Option<String>,
}

fn mcp_default_enabled() -> bool {
    true
}

/// `POST /studio/chat/mcp-servers` — add or update a server. Re-syncs the
/// proxy set; the server is spawned lazily on first tool call.
pub async fn upsert_mcp_server(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddMcpServerBody>,
) -> axum::response::Response {
    if body.name.is_empty() || body.name.contains('/') || body.name.contains(' ') {
        return error_response(TensaError::InvalidInput(
            "server name must be non-empty and contain no `/` or spaces".into(),
        ))
        .into_response();
    }
    if body.command.is_empty() {
        return error_response(TensaError::InvalidInput("command is required".into()))
            .into_response();
    }
    let cfg = super::mcp_proxy::McpServerConfig {
        name: body.name.clone(),
        command: body.command,
        args: body.args,
        env: body.env,
        enabled: body.enabled,
        description: body.description,
    };
    let mut list = super::mcp_proxy::load_persisted_mcp_servers(state.root_store.as_ref());
    if let Some(existing) = list.iter_mut().find(|c| c.name == body.name) {
        *existing = cfg.clone();
    } else {
        list.push(cfg.clone());
    }
    if let Err(e) = super::mcp_proxy::persist_mcp_servers(state.root_store.as_ref(), &list) {
        return error_response(e).into_response();
    }
    state.chat_mcp_proxies.sync(&list).await;
    json_ok(&cfg)
}

/// `DELETE /studio/chat/mcp-servers/:name` — remove a server and drop its proxy.
pub async fn delete_mcp_server(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> axum::response::Response {
    let mut list = super::mcp_proxy::load_persisted_mcp_servers(state.root_store.as_ref());
    let before = list.len();
    list.retain(|c| c.name != name);
    if list.len() == before {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("mcp server `{}` not found", name) })),
        )
            .into_response();
    }
    if let Err(e) = super::mcp_proxy::persist_mcp_servers(state.root_store.as_ref(), &list) {
        return error_response(e).into_response();
    }
    state.chat_mcp_proxies.sync(&list).await;
    json_ok(&json!({ "deleted": name }))
}

// ─── Skills listing ─────────────────────────────────────────

#[derive(Serialize)]
struct SkillInfo {
    name: String,
    description: String,
}

/// `GET /studio/chat/skills` — List bundled skills available to sessions.
pub async fn list_skills(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let skills: Vec<SkillInfo> = state
        .chat_skills
        .list()
        .iter()
        .map(|b| SkillInfo {
            name: b.name.clone(),
            description: b.description.clone(),
        })
        .collect();
    json_ok(&skills)
}

// ─── Chat turn (SSE) ────────────────────────────────────────

/// `POST /studio/chat` — Start or continue a conversation turn.
pub async fn chat_turn(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatTurnRequest>,
) -> axum::response::Response {
    let scope = scope_from_headers(&headers);
    let store = chat_store(&state);

    let session = match ensure_session(&store, &scope, req.session_id.clone(), &req.user_message) {
        Ok(s) => s,
        Err(e) => return error_response(e).into_response(),
    };

    let extractor = resolve_chat_extractor(&state);
    let skills = state.chat_skills.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(SSE_CHANNEL_CAPACITY);

    tokio::spawn(run_turn(
        state.clone(),
        scope,
        session,
        req,
        extractor,
        skills,
        tx,
    ));

    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

/// Resolution order: chat-specific LLM > inference/RAG LLM > ingestion LLM.
/// A poisoned lock is logged and treated as "no extractor here" so we fall
/// through to the next slot rather than hanging the turn.
fn resolve_chat_extractor(state: &AppState) -> Option<Arc<dyn NarrativeExtractor>> {
    fn read_slot(
        name: &str,
        slot: &std::sync::RwLock<Option<Arc<dyn NarrativeExtractor>>>,
    ) -> Option<Arc<dyn NarrativeExtractor>> {
        match slot.read() {
            Ok(guard) => guard.clone(),
            Err(e) => {
                tracing::warn!("{} extractor lock poisoned: {}", name, e);
                None
            }
        }
    }
    read_slot("chat", &state.chat_extractor)
        .or_else(|| read_slot("inference", &state.inference_extractor))
        .or_else(|| read_slot("ingestion", &state.extractor))
}

async fn run_turn(
    state: Arc<AppState>,
    scope: Scope,
    mut session: SessionMeta,
    req: ChatTurnRequest,
    extractor: Option<Arc<dyn NarrativeExtractor>>,
    skills: SkillRegistry,
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let store = ChatStore::new(state.root_store.clone());

    let _ = tx
        .send(Ok(to_sse_event(&ChatEvent::SessionStarted {
            session_id: session.id.clone(),
            title: session.title.clone(),
        })))
        .await;

    let prior = store.list_messages(&scope, &session.id).unwrap_or_default();

    let user_msg = Message::user_text(&session.id, &req.user_message);
    if let Err(e) = store.append_message(&scope, &user_msg) {
        send_error(&tx, "persist_user", &e.to_string()).await;
        return;
    }
    let _ = tx
        .send(Ok(to_sse_event(&ChatEvent::UserPersisted {
            message_id: user_msg.id.simple().to_string(),
        })))
        .await;

    let extractor = match extractor {
        Some(e) => e,
        None => {
            send_error(
                &tx,
                "no_llm",
                "No chat LLM configured. Set one at Settings → Chat LLM, or fall back to the ingestion LLM.",
            )
            .await;
            return;
        }
    };

    let tool_specs = super::tools::catalog();
    let external_tools = state.chat_mcp_proxies.collect_tools().await;
    let system_prompt = format!(
        "{}{}",
        build_system_prompt_with_tools(&skills, &session, &req, &tool_specs),
        render_external_tools(&external_tools)
    );

    // Running ApiMessage list the LLM sees. Starts with trimmed prior
    // history + the new user message; accumulates assistant + tool turns
    // as the harness iterates.
    let mut api_history = build_history(&prior, &req.user_message);
    // Track DB-persisted message count delta for session.msg_count bump.
    let mut msg_delta: u32 = 1; // user message already persisted

    let mut final_message_id: Option<String> = None;

    for iteration in 1..=MAX_TOOL_ITERATIONS {
        let reply = match call_llm(&extractor, &system_prompt, &api_history, &tx).await {
            Some(s) => s,
            None => return, // error already reported
        };
        let parsed = parse_tool_block(&reply);

        // Always stream the text that precedes the tool call (or the whole
        // reply if there was no call). This keeps the UI lively while the
        // harness possibly drills further.
        stream_tokens(&tx, &parsed.text_before).await;

        if parsed.tool_call.is_none() {
            let asst_msg = Message::assistant_text(&session.id, &reply);
            if let Err(e) = store.append_message(&scope, &asst_msg) {
                send_error(&tx, "persist_assistant", &e.to_string()).await;
                return;
            }
            msg_delta += 1;
            final_message_id = Some(asst_msg.id.simple().to_string());
            break;
        }

        let call = parsed.tool_call.expect("checked Some above");
        let call_id = Uuid::now_v7().simple().to_string();

        // Persist the assistant turn that requested the tool.
        let asst_text_part = if parsed.text_before.trim().is_empty() {
            None
        } else {
            Some(ContentPart::text(parsed.text_before.clone()))
        };
        let tool_call_part = ContentPart::ToolCall {
            id: call_id.clone(),
            name: call.name.clone(),
            args: call.args.clone(),
        };
        let asst_msg = Message::new(
            &session.id,
            MessageRole::Assistant,
            asst_text_part.into_iter().chain([tool_call_part]).collect(),
        );
        if let Err(e) = store.append_message(&scope, &asst_msg) {
            send_error(&tx, "persist_assistant", &e.to_string()).await;
            return;
        }
        msg_delta += 1;
        // Give the LLM the exact text it emitted so the next round has a
        // consistent reasoning trail.
        api_history.push(ApiMessage {
            role: "assistant".into(),
            content: reply.clone(),
        });

        let _ = tx
            .send(Ok(to_sse_event(&ChatEvent::ToolCall {
                id: call_id.clone(),
                name: call.name.clone(),
                args: call.args.clone(),
            })))
            .await;

        let decision = if super::confirm::classify(&call.name) == super::confirm::ToolClass::Confirm
        {
            await_confirmation(&state, &scope, &session.id, &call_id, &call, &tx).await
        } else {
            super::confirm::ConfirmDecision::Approve
        };

        let (ok, output) = if matches!(decision, super::confirm::ConfirmDecision::Reject) {
            (
                false,
                serde_json::json!({
                    "rejected": true,
                    "reason": "user declined the tool call",
                }),
            )
        } else {
            match super::tools::dispatch(&state, &call.name, call.args.clone()).await {
                Ok(out) => (true, out.full),
                Err(e) => (false, serde_json::json!({ "error": e.to_string() })),
            }
        };

        let result_part = ContentPart::ToolResult {
            call_id: call_id.clone(),
            ok,
            output: output.clone(),
        };
        let tool_msg = Message::new(&session.id, MessageRole::Tool, vec![result_part]);
        if let Err(e) = store.append_message(&scope, &tool_msg) {
            send_error(&tx, "persist_tool_result", &e.to_string()).await;
            return;
        }
        msg_delta += 1;

        let _ = tx
            .send(Ok(to_sse_event(&ChatEvent::ToolResult {
                call_id: call_id.clone(),
                ok,
                output_preview: truncate_for_preview(&output),
            })))
            .await;

        // Feed the tool result back to the LLM as a user-role marker.
        // Keeps the wire format simple across providers that don't share a
        // native tool-result role.
        api_history.push(ApiMessage {
            role: "user".into(),
            content: format!(
                "[tool_result name={} ok={}]\n{}",
                call.name,
                ok,
                serde_json::to_string(&output).unwrap_or_default()
            ),
        });

        if iteration == MAX_TOOL_ITERATIONS {
            send_error(
                &tx,
                "tool_loop_exhausted",
                "Reached the maximum number of tool iterations without a final answer.",
            )
            .await;
            return;
        }
    }

    session.updated_at = chrono::Utc::now();
    session.msg_count = session.msg_count.saturating_add(msg_delta);
    let _ = store.update_session(&scope, &session);

    if let Some(id) = final_message_id {
        let _ = tx
            .send(Ok(to_sse_event(&ChatEvent::Final { message_id: id })))
            .await;
    }
}

/// Run one LLM round-trip on the blocking pool. Returns None and emits an
/// error event on failure so the caller can `return` cleanly.
async fn call_llm(
    extractor: &Arc<dyn NarrativeExtractor>,
    system_prompt: &str,
    history: &[ApiMessage],
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) -> Option<String> {
    let extractor = extractor.clone();
    let system = system_prompt.to_string();
    let history: Vec<ApiMessage> = history.to_vec();

    let result = tokio::task::spawn_blocking(move || {
        if let Some(session_api) = extractor.as_session() {
            let mut msgs = Vec::with_capacity(history.len() + 1);
            msgs.push(ApiMessage {
                role: "system".into(),
                content: system,
            });
            msgs.extend(history);
            session_api.send_session_messages(&msgs)
        } else {
            let flat = flatten_history(&history);
            extractor.answer_question(&system, &flat)
        }
    })
    .await;

    match result {
        Ok(Ok(s)) => Some(s),
        Ok(Err(e)) => {
            send_error(tx, "llm", &e.to_string()).await;
            None
        }
        Err(e) => {
            send_error(tx, "llm_join", &e.to_string()).await;
            None
        }
    }
}

async fn await_confirmation(
    state: &Arc<AppState>,
    scope: &Scope,
    session_id: &str,
    call_id: &str,
    call: &ParsedToolCall,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) -> super::confirm::ConfirmDecision {
    let key = super::confirm::PendingKey::new(scope, session_id, call_id);
    let rx = state.chat_confirm_gate.register(key);
    let _ = tx
        .send(Ok(to_sse_event(&ChatEvent::AwaitingConfirm {
            call_id: call_id.to_string(),
            summary: format!("Run `{}` with the proposed arguments?", call.name),
            preview: call.args.clone(),
        })))
        .await;
    match tokio::time::timeout(CONFIRM_TIMEOUT, rx).await {
        Ok(Ok(decision)) => decision,
        _ => {
            // Timeout or receiver dropped — treat as rejection. Also clean
            // up in case the sender is still registered.
            let expired = super::confirm::PendingKey::new(scope, session_id, call_id);
            state.chat_confirm_gate.forget(&expired);
            super::confirm::ConfirmDecision::Reject
        }
    }
}

async fn stream_tokens(tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>, text: &str) {
    if text.is_empty() {
        return;
    }
    let mut drip = String::new();
    let mut emitted = 0usize;
    for word in text.split_inclusive(char::is_whitespace) {
        drip.push_str(word);
        emitted += 1;
        if emitted % TOKEN_WORDS_PER_EVENT == 0 {
            let _ = tx
                .send(Ok(to_sse_event(&ChatEvent::Token {
                    delta: std::mem::take(&mut drip),
                })))
                .await;
            tokio::time::sleep(TOKEN_DELAY).await;
        }
    }
    if !drip.is_empty() {
        let _ = tx
            .send(Ok(to_sse_event(&ChatEvent::Token { delta: drip })))
            .await;
    }
}

/// Keep the SSE `output_preview` field small — some tools return many KB of
/// rows that the UI neither needs nor can render usefully. The full output
/// is persisted in the `ToolResult` message for later inspection.
fn truncate_for_preview(v: &serde_json::Value) -> serde_json::Value {
    const MAX: usize = 1024;
    let s = serde_json::to_string(v).unwrap_or_default();
    if s.len() <= MAX {
        v.clone()
    } else {
        serde_json::json!({
            "truncated": true,
            "preview": s.chars().take(MAX).collect::<String>(),
        })
    }
}

// ─── Tool-block parser ──────────────────────────────────────

#[derive(Debug, Clone)]
struct ParsedToolCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Default)]
struct ParsedReply {
    text_before: String,
    tool_call: Option<ParsedToolCall>,
}

/// Extract a single `\`\`\`tensa-tool … \`\`\`` block from an LLM reply. We
/// intentionally keep this parser permissive so minor formatting drift (no
/// trailing newline before the fence, extra whitespace) doesn't abort the
/// turn. Only the FIRST block is honoured — the skill prompt tells the LLM
/// to emit at most one call per turn.
fn parse_tool_block(reply: &str) -> ParsedReply {
    const OPEN: &str = "```tensa-tool";
    const CLOSE: &str = "```";
    let Some(open_idx) = reply.find(OPEN) else {
        return ParsedReply {
            text_before: reply.to_string(),
            tool_call: None,
        };
    };
    let after_open = &reply[open_idx + OPEN.len()..];
    // Find the closing fence relative to after_open; ignore leading whitespace.
    let Some(close_rel) = after_open.find(CLOSE) else {
        return ParsedReply {
            text_before: reply.to_string(),
            tool_call: None,
        };
    };
    let body = after_open[..close_rel].trim();
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body) else {
        return ParsedReply {
            text_before: reply.to_string(),
            tool_call: None,
        };
    };
    let name = parsed
        .get("tool")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if name.is_empty() {
        return ParsedReply {
            text_before: reply.to_string(),
            tool_call: None,
        };
    }
    let args = parsed
        .get("args")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    // Drop trailing whitespace so we don't stream dangling blank lines
    // before the (invisible-to-user) tool fence.
    ParsedReply {
        text_before: reply[..open_idx].trim_end().to_string(),
        tool_call: Some(ParsedToolCall { name, args }),
    }
}

fn build_system_prompt_with_tools(
    skills: &SkillRegistry,
    session: &SessionMeta,
    req: &ChatTurnRequest,
    tools: &[super::tools::ToolSpec],
) -> String {
    let base = build_system_prompt(skills, session, req);
    format!("{}\n\n{}", base, super::tools::tool_catalog_markdown(tools))
}

/// Render the aggregated proxy tool catalog as a short Markdown section.
/// Empty when there are no external tools so we don't pollute the prompt.
fn render_external_tools(external: &[serde_json::Value]) -> String {
    if external.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n## External MCP tools (always require confirmation)\n\n");
    for t in external {
        let name = t
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("(unknown)");
        let desc = t
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("(no description)");
        out.push_str(&format!("- `{}` — {}\n", name, desc));
    }
    out
}

async fn send_error(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    code: &str,
    msg: &str,
) {
    let _ = tx
        .send(Ok(to_sse_event(&ChatEvent::Error {
            code: code.to_string(),
            msg: msg.to_string(),
        })))
        .await;
}

fn build_system_prompt(
    skills: &SkillRegistry,
    session: &SessionMeta,
    req: &ChatTurnRequest,
) -> String {
    let active = req
        .active_skills
        .as_ref()
        .filter(|v| !v.is_empty())
        .cloned()
        .unwrap_or_else(|| session.active_skills.clone());
    let mut out = skills.compose_system_prompt(&active);
    if let Some(n) = req
        .narrative_scope
        .as_deref()
        .or(session.narrative_scope.as_deref())
    {
        out.push_str(&format!(
            "\n## Current context\nActive narrative: `{}`\n",
            n
        ));
    }
    out
}

fn build_history(prior: &[Message], current_user: &str) -> Vec<ApiMessage> {
    let mut msgs: Vec<ApiMessage> = prior.iter().filter_map(message_to_api).collect();
    // Bound the prompt. We keep the *tail* (most recent) rather than head.
    if msgs.len() > MAX_HISTORY_TURNS {
        let skip = msgs.len() - MAX_HISTORY_TURNS;
        msgs.drain(..skip);
    }
    msgs.push(ApiMessage {
        role: "user".into(),
        content: current_user.to_string(),
    });
    msgs
}

fn message_to_api(m: &Message) -> Option<ApiMessage> {
    let role = match m.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => return None, // no tool turns to replay in Phase 2
        MessageRole::System => "system",
    };
    let content: String = m
        .content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if content.is_empty() {
        None
    } else {
        Some(ApiMessage {
            role: role.to_string(),
            content,
        })
    }
}

fn flatten_history(msgs: &[ApiMessage]) -> String {
    let mut out = String::new();
    for (i, m) in msgs.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        out.push_str(&format!("[{}] {}", m.role, m.content));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};

    #[test]
    fn scope_defaults_apply() {
        let headers = HeaderMap::new();
        let scope = scope_from_headers(&headers);
        assert_eq!(scope.workspace_id, "default");
        assert_eq!(scope.user_id, "local");
    }

    #[test]
    fn scope_reads_workspace_and_user_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(WORKSPACE_HEADER, HeaderValue::from_static("acme"));
        headers.insert(USER_HEADER, HeaderValue::from_static("alice"));
        let scope = scope_from_headers(&headers);
        assert_eq!(scope.workspace_id, "acme");
        assert_eq!(scope.user_id, "alice");
    }

    #[test]
    fn empty_header_falls_back_to_default() {
        let mut headers = HeaderMap::new();
        headers.insert(USER_HEADER, HeaderValue::from_static(""));
        let scope = scope_from_headers(&headers);
        assert_eq!(scope.user_id, "local");
    }

    #[test]
    fn autogen_title_truncates_long_messages() {
        let long = "x".repeat(200);
        let title = autogen_title(&long);
        assert!(title.ends_with('…'));
        // 60 chars + the ellipsis character
        assert_eq!(title.chars().count(), 61);
    }

    #[test]
    fn autogen_title_empty_message_becomes_new_chat() {
        assert_eq!(autogen_title("   "), "New chat");
    }

    #[test]
    fn chat_event_serializes_as_tagged_enum() {
        let ev = ChatEvent::SessionStarted {
            session_id: "s".into(),
            title: "t".into(),
        };
        let s = ev.to_sse_data();
        assert!(s.contains("\"event\":\"session_started\""));
        assert!(s.contains("\"session_id\":\"s\""));
    }

    #[test]
    fn message_to_api_concatenates_text_and_skips_tool_only() {
        let user = Message::user_text("s", "hello");
        let asst = Message::new(
            "s",
            MessageRole::Assistant,
            vec![
                ContentPart::text("thinking…"),
                ContentPart::ToolCall {
                    id: "c1".into(),
                    name: "list_narratives".into(),
                    args: serde_json::json!({}),
                },
            ],
        );
        let tool_only = Message::new(
            "s",
            MessageRole::Tool,
            vec![ContentPart::ToolResult {
                call_id: "c1".into(),
                ok: true,
                output: serde_json::json!([]),
            }],
        );
        assert!(message_to_api(&user).is_some());
        let api_asst = message_to_api(&asst).unwrap();
        assert_eq!(api_asst.role, "assistant");
        assert_eq!(api_asst.content, "thinking…");
        assert!(message_to_api(&tool_only).is_none());
    }

    #[test]
    fn build_history_trims_to_max_turns_and_appends_current() {
        let prior: Vec<Message> = (0..MAX_HISTORY_TURNS + 5)
            .map(|i| Message::user_text("s", format!("turn {}", i)))
            .collect();
        let h = build_history(&prior, "now");
        assert_eq!(h.len(), MAX_HISTORY_TURNS + 1);
        assert_eq!(h.last().unwrap().content, "now");
        assert_eq!(h.last().unwrap().role, "user");
        assert!(h[0].content.contains("turn 5"));
    }

    #[test]
    fn system_prompt_includes_narrative_scope_when_set() {
        let skills = SkillRegistry::default_bundled();
        let session = SessionMeta::new(&Scope::default_scope(), "s", "t");
        let req = ChatTurnRequest {
            session_id: None,
            user_message: String::new(),
            active_skills: None,
            narrative_scope: Some("harbor-case".into()),
            model_override: None,
        };
        let prompt = build_system_prompt(&skills, &session, &req);
        assert!(prompt.contains("Active narrative: `harbor-case`"));
    }

    #[test]
    fn flatten_history_interleaves_roles() {
        let msgs = vec![
            ApiMessage {
                role: "user".into(),
                content: "hi".into(),
            },
            ApiMessage {
                role: "assistant".into(),
                content: "hello".into(),
            },
        ];
        let s = flatten_history(&msgs);
        assert!(s.contains("[user] hi"));
        assert!(s.contains("[assistant] hello"));
    }

    #[test]
    fn parse_tool_block_extracts_name_and_args() {
        let reply =
            "Sure, let me look that up.\n\n```tensa-tool\n{\"tool\":\"list_narratives\",\"args\":{}}\n```\n";
        let parsed = parse_tool_block(reply);
        let call = parsed.tool_call.expect("tool call expected");
        assert_eq!(call.name, "list_narratives");
        assert!(parsed.text_before.contains("let me look"));
    }

    #[test]
    fn parse_tool_block_without_block_returns_plain_text() {
        let parsed = parse_tool_block("No tool needed — you have 3 narratives.");
        assert!(parsed.tool_call.is_none());
        assert!(parsed.text_before.contains("No tool needed"));
    }

    #[test]
    fn parse_tool_block_rejects_invalid_json() {
        let reply = "```tensa-tool\n{not json}\n```";
        let parsed = parse_tool_block(reply);
        assert!(parsed.tool_call.is_none());
    }

    #[test]
    fn parse_tool_block_extracts_nested_args() {
        let reply = "```tensa-tool\n{\"tool\":\"get_entity\",\"args\":{\"id\":\"e1\"}}\n```";
        let call = parse_tool_block(reply).tool_call.unwrap();
        assert_eq!(call.name, "get_entity");
        assert_eq!(call.args["id"], "e1");
    }

    #[test]
    fn truncate_for_preview_keeps_short_values() {
        let v = serde_json::json!({ "a": "b" });
        assert_eq!(truncate_for_preview(&v), v);
    }

    #[test]
    fn truncate_for_preview_shrinks_large_values() {
        let big: String = std::iter::repeat('x').take(4096).collect();
        let v = serde_json::json!({ "blob": big });
        let p = truncate_for_preview(&v);
        assert_eq!(p["truncated"], true);
    }
}
