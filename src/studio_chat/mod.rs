//! Studio integrated agent chat (v0.60, Phase 1 skeleton).
//!
//! Provides `/studio/chat` SSE endpoint and session storage for the Studio
//! right-side chat panel. Phase 1 ships the wire format and persistence only:
//! the endpoint echoes user messages back. LLM tool-calling, MCP proxying,
//! and skills arrive in later phases.
//!
//! Storage layout (KV keys on the root store, outside workspace prefix):
//!
//! ```text
//! chat/s/{ws}/{user}/{session_id}              SessionMeta
//! chat/m/{ws}/{user}/{session_id}/{msg_v7}     Message
//! ```
//!
//! Chat state is multi-tenant-ready from day one: sessions are always keyed
//! by `workspace_id` + `user_id`, with defaults `"default"` and `"local"`.

pub mod config;
pub mod confirm;
pub mod mcp_proxy;
pub mod routes;
pub mod skills;
pub mod store;
pub mod tools;
pub mod types;

pub use config::{load_persisted_chat_llm_config, CFG_CHAT_LLM_KEY};
pub use confirm::{classify as classify_tool, ConfirmDecision, ConfirmGate, PendingKey, ToolClass};
pub use mcp_proxy::{
    load_persisted_mcp_servers, persist_mcp_servers, McpProxy, McpProxySet, McpServerConfig,
    McpServerStatus, CFG_MCP_SERVERS_KEY, NAMESPACE_SEP,
};
pub use skills::{SkillBundle, SkillRegistry};
pub use store::ChatStore;
pub use tools::{
    catalog as tool_catalog, dispatch as tool_dispatch, tool_catalog_markdown, ToolSpec,
};
pub use types::{ChatEvent, ContentPart, Message, MessageRole, Scope, SessionMeta};
