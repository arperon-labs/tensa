//! Third-party MCP server proxy (Phase 4).
//!
//! Users register stdio MCP servers under `cfg/studio_chat_mcp_servers`.
//! When the Studio chat turn needs a tool we've never seen locally we
//! spawn the configured child process, perform the MCP handshake via
//! `rmcp`'s client transport, cache the tool list, and forward tool calls
//! through the proxy. Tools are namespaced as `{server_name}__{tool_name}`
//! so names don't collide with the local chat toolset.
//!
//! Phase 4 keeps supervision deliberately minimal: spawn on first demand,
//! hold the connection for the lifetime of the process, surface tool errors
//! back to the chat loop. Automatic restart and hot-reload land in Phase 5.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::Command;
use tokio::sync::{Mutex, OnceCell};

use crate::error::{Result, TensaError};
use crate::store::KVStore;

/// KV key for persisted MCP server configs.
pub const CFG_MCP_SERVERS_KEY: &[u8] = b"cfg/studio_chat_mcp_servers";
/// Separator between server name and tool name in the namespaced tool id.
pub const NAMESPACE_SEP: &str = "__";

/// Config entry for one third-party MCP server. The user pastes the
/// `command` / `args` they'd use in a Claude Desktop `mcp.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique, URL-safe name. Used as a namespace prefix on tools
    /// (`{name}__{tool}`).
    pub name: String,
    /// Executable to run (e.g. `npx`, `uvx`, absolute path).
    pub command: String,
    /// Arguments passed to `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables to inject.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Skip this server without deleting the config.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional human description for the settings UI.
    #[serde(default)]
    pub description: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Runtime status reported via `GET /studio/chat/mcp-servers`.
#[derive(Debug, Clone, Serialize)]
pub struct McpServerStatus {
    pub name: String,
    pub enabled: bool,
    pub running: bool,
    pub tool_count: usize,
    pub last_error: Option<String>,
}

pub fn load_persisted_mcp_servers(store: &dyn KVStore) -> Vec<McpServerConfig> {
    store
        .get(CFG_MCP_SERVERS_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub fn persist_mcp_servers(store: &dyn KVStore, servers: &[McpServerConfig]) -> Result<()> {
    let bytes =
        serde_json::to_vec(servers).map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(CFG_MCP_SERVERS_KEY, &bytes)?;
    Ok(())
}

// ─── Connection wrapper ─────────────────────────────────────

#[cfg(feature = "studio-chat")]
type RunningClient = rmcp::service::RunningService<rmcp::service::RoleClient, ()>;

/// One live connection to a third-party MCP server. Created lazily on first
/// use so an ill-configured server doesn't block startup.
pub struct McpProxy {
    config: McpServerConfig,
    #[cfg(feature = "studio-chat")]
    client: OnceCell<std::result::Result<Arc<RunningClient>, String>>,
    #[cfg(feature = "studio-chat")]
    tools: OnceCell<Vec<rmcp::model::Tool>>,
    last_error: Mutex<Option<String>>,
}

impl McpProxy {
    pub fn new(config: McpServerConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "studio-chat")]
            client: OnceCell::new(),
            #[cfg(feature = "studio-chat")]
            tools: OnceCell::new(),
            last_error: Mutex::new(None),
        }
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Current status snapshot for the Settings UI.
    pub async fn status(&self) -> McpServerStatus {
        #[cfg(feature = "studio-chat")]
        let (running, tool_count) = {
            let client_ready = self.client.get().map(|r| r.is_ok()).unwrap_or(false);
            let n = self.tools.get().map(|t| t.len()).unwrap_or(0);
            (client_ready, n)
        };
        #[cfg(not(feature = "studio-chat"))]
        let (running, tool_count) = (false, 0usize);
        McpServerStatus {
            name: self.config.name.clone(),
            enabled: self.config.enabled,
            running,
            tool_count,
            last_error: self.last_error.lock().await.clone(),
        }
    }

    #[cfg(feature = "studio-chat")]
    async fn client(&self) -> std::result::Result<Arc<RunningClient>, String> {
        self.client
            .get_or_init(|| async { spawn_client(&self.config).await.map(Arc::new) })
            .await
            .clone()
    }

    /// List the tools exposed by this server (cached after the first call).
    /// Returns a JSON array ready to stitch into the chat tool catalog.
    #[cfg(feature = "studio-chat")]
    pub async fn list_tools(&self) -> Result<Vec<Value>> {
        let client = self.client().await.map_err(|e| {
            self.record_error(&e);
            TensaError::Internal(format!(
                "mcp proxy `{}` spawn failed: {}",
                self.config.name, e
            ))
        })?;
        let tools = self
            .tools
            .get_or_try_init(|| async {
                client
                    .peer()
                    .list_all_tools()
                    .await
                    .map_err(|e| format!("list_tools: {}", e))
            })
            .await
            .map_err(|e| {
                self.record_error(e);
                TensaError::Internal(format!(
                    "mcp proxy `{}` list_tools failed",
                    self.config.name,
                ))
            })?;
        let out: Vec<Value> = tools
            .iter()
            .map(|t| {
                let desc = t
                    .description
                    .as_deref()
                    .unwrap_or("(no description)")
                    .to_string();
                serde_json::json!({
                    "name": format!("{}{}{}", self.config.name, NAMESPACE_SEP, t.name),
                    "description": desc,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        Ok(out)
    }

    #[cfg(not(feature = "studio-chat"))]
    pub async fn list_tools(&self) -> Result<Vec<Value>> {
        Err(TensaError::Internal("studio-chat feature disabled".into()))
    }

    /// Invoke a tool on the child server. `name` here is the unnamespaced
    /// tool name (the caller strips the `{server}__` prefix).
    #[cfg(feature = "studio-chat")]
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value> {
        use rmcp::model::CallToolRequestParams;
        let client = self.client().await.map_err(|e| {
            TensaError::Internal(format!(
                "mcp proxy `{}` spawn failed: {}",
                self.config.name, e
            ))
        })?;
        let arguments = match args {
            Value::Object(m) => Some(m),
            Value::Null => None,
            other => {
                return Err(TensaError::InvalidInput(format!(
                    "tool args must be a JSON object, got {:?}",
                    other
                )))
            }
        };
        let mut params = CallToolRequestParams::default();
        params.name = name.to_string().into();
        params.arguments = arguments;
        let result = client.peer().call_tool(params).await.map_err(|e| {
            TensaError::Internal(format!(
                "mcp tool `{}__{}` failed: {}",
                self.config.name, name, e
            ))
        })?;
        Ok(serde_json::json!({
            "is_error": result.is_error.unwrap_or(false),
            "content": result.content,
        }))
    }

    #[cfg(not(feature = "studio-chat"))]
    pub async fn call_tool(&self, _name: &str, _args: Value) -> Result<Value> {
        Err(TensaError::Internal("studio-chat feature disabled".into()))
    }
}

#[cfg(feature = "studio-chat")]
async fn spawn_client(cfg: &McpServerConfig) -> std::result::Result<RunningClient, String> {
    use rmcp::service::ServiceExt;
    use rmcp::transport::TokioChildProcess;

    let mut cmd = Command::new(&cfg.command);
    cmd.args(&cfg.args);
    for (k, v) in &cfg.env {
        cmd.env(k, v);
    }
    let transport =
        TokioChildProcess::new(cmd).map_err(|e| format!("spawn `{}`: {}", cfg.command, e))?;
    let running = ().serve(transport).await.map_err(|e| format!("mcp handshake: {}", e))?;
    Ok(running)
}

// ─── Set-level helpers ──────────────────────────────────────

/// Thread-safe registry of MCP proxies, one per configured server name.
/// `AppState.chat_mcp_proxies` holds an `Arc<McpProxySet>`.
#[derive(Default)]
pub struct McpProxySet {
    proxies: tokio::sync::RwLock<HashMap<String, Arc<McpProxy>>>,
}

impl McpProxySet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild the set from a fresh `Vec<McpServerConfig>`. Existing proxies
    /// whose config didn't change are kept alive (their OnceCells are
    /// retained); removed ones are dropped.
    pub async fn sync(&self, configs: &[McpServerConfig]) {
        let mut w = self.proxies.write().await;
        let keep: std::collections::HashSet<&String> = configs.iter().map(|c| &c.name).collect();
        w.retain(|name, _| keep.contains(name));
        for cfg in configs {
            w.entry(cfg.name.clone())
                .or_insert_with(|| Arc::new(McpProxy::new(cfg.clone())));
        }
    }

    pub async fn get(&self, name: &str) -> Option<Arc<McpProxy>> {
        self.proxies.read().await.get(name).cloned()
    }

    pub async fn statuses(&self) -> Vec<McpServerStatus> {
        let proxies: Vec<Arc<McpProxy>> = self.proxies.read().await.values().cloned().collect();
        let mut out = Vec::with_capacity(proxies.len());
        for p in proxies {
            out.push(p.status().await);
        }
        out
    }

    /// Enumerate every enabled proxy's tools (namespaced) for inclusion in
    /// the chat tool catalog. Errors from individual proxies degrade to an
    /// empty list for that server so one broken config doesn't break chat.
    pub async fn collect_tools(&self) -> Vec<Value> {
        let proxies: Vec<Arc<McpProxy>> = self
            .proxies
            .read()
            .await
            .values()
            .filter(|p| p.enabled())
            .cloned()
            .collect();
        let mut out = Vec::new();
        for p in proxies {
            match p.list_tools().await {
                Ok(tools) => out.extend(tools),
                Err(e) => tracing::warn!("mcp proxy `{}` tools failed: {}", p.name(), e),
            }
        }
        out
    }
}

impl McpProxy {
    /// Best-effort record of the last error for the Settings UI. Uses
    /// `try_lock` so we never wait on the mutex during the hot spawn path.
    fn record_error(&self, msg: impl Into<String>) {
        if let Ok(mut g) = self.last_error.try_lock() {
            *g = Some(msg.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    #[test]
    fn config_roundtrip() {
        let cfg = McpServerConfig {
            name: "fs".into(),
            command: "npx".into(),
            args: vec![
                "-y".into(),
                "@modelcontextprotocol/server-filesystem".into(),
            ],
            env: HashMap::new(),
            enabled: true,
            description: Some("Filesystem access".into()),
        };
        let s = serde_json::to_string(&cfg).unwrap();
        let parsed: McpServerConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.name, "fs");
        assert_eq!(parsed.args.len(), 2);
        assert!(parsed.enabled);
    }

    #[test]
    fn persist_roundtrip() {
        let store = MemoryStore::new();
        let list = vec![
            McpServerConfig {
                name: "a".into(),
                command: "x".into(),
                args: vec![],
                env: HashMap::new(),
                enabled: true,
                description: None,
            },
            McpServerConfig {
                name: "b".into(),
                command: "y".into(),
                args: vec![],
                env: HashMap::new(),
                enabled: false,
                description: None,
            },
        ];
        persist_mcp_servers(&store, &list).unwrap();
        let loaded = load_persisted_mcp_servers(&store);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "a");
        assert!(!loaded[1].enabled);
    }

    #[tokio::test]
    async fn proxy_set_sync_keeps_and_drops() {
        let set = McpProxySet::new();
        set.sync(&[McpServerConfig {
            name: "a".into(),
            command: "x".into(),
            args: vec![],
            env: HashMap::new(),
            enabled: true,
            description: None,
        }])
        .await;
        assert!(set.get("a").await.is_some());
        set.sync(&[]).await;
        assert!(set.get("a").await.is_none());
    }
}
