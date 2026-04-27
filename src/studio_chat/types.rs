//! Shared types for studio_chat.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const DEFAULT_WORKSPACE: &str = "default";
pub const DEFAULT_USER: &str = "local";

/// Workspace + user pair that scopes every chat KV access.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scope {
    pub workspace_id: String,
    pub user_id: String,
}

impl Scope {
    pub fn new(workspace_id: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            workspace_id: workspace_id.into(),
            user_id: user_id.into(),
        }
    }

    /// Scope with both defaults ("default"/"local").
    pub fn default_scope() -> Self {
        Self::new(DEFAULT_WORKSPACE, DEFAULT_USER)
    }
}

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
    System,
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
            MessageRole::System => write!(f, "system"),
        }
    }
}

/// One piece of content within a message. Phase 1 only populates `Text`;
/// tool variants are defined here so later phases don't need schema migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        call_id: String,
        ok: bool,
        output: serde_json::Value,
    },
    ConfirmRequest {
        call_id: String,
        summary: String,
        preview: serde_json::Value,
    },
}

impl ContentPart {
    pub fn text(s: impl Into<String>) -> Self {
        ContentPart::Text { text: s.into() }
    }
}

/// A single chat message, persisted at `chat/m/{ws}/{user}/{session_id}/{v7}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub session_id: String,
    pub role: MessageRole,
    pub content: Vec<ContentPart>,
    pub timestamp: DateTime<Utc>,
}

impl Message {
    pub fn new(
        session_id: impl Into<String>,
        role: MessageRole,
        content: Vec<ContentPart>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            session_id: session_id.into(),
            role,
            content,
            timestamp: Utc::now(),
        }
    }

    pub fn user_text(session_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::new(session_id, MessageRole::User, vec![ContentPart::text(text)])
    }

    pub fn assistant_text(session_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::new(
            session_id,
            MessageRole::Assistant,
            vec![ContentPart::text(text)],
        )
    }
}

/// Session metadata, persisted at `chat/s/{ws}/{user}/{session_id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub workspace_id: String,
    pub user_id: String,
    pub title: String,
    #[serde(default)]
    pub active_skills: Vec<String>,
    #[serde(default)]
    pub narrative_scope: Option<String>,
    #[serde(default)]
    pub model_override: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub msg_count: u32,
    #[serde(default)]
    pub archived: bool,
}

impl SessionMeta {
    pub fn new(scope: &Scope, id: impl Into<String>, title: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            workspace_id: scope.workspace_id.clone(),
            user_id: scope.user_id.clone(),
            title: title.into(),
            active_skills: vec!["studio-ui".into()],
            narrative_scope: None,
            model_override: None,
            created_at: now,
            updated_at: now,
            msg_count: 0,
            archived: false,
        }
    }
}

/// Request body for `POST /studio/chat`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatTurnRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    pub user_message: String,
    #[serde(default)]
    pub active_skills: Option<Vec<String>>,
    #[serde(default)]
    pub narrative_scope: Option<String>,
    #[serde(default)]
    pub model_override: Option<String>,
}

/// Events streamed back over SSE during a chat turn.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ChatEvent {
    SessionStarted {
        session_id: String,
        title: String,
    },
    UserPersisted {
        message_id: String,
    },
    Token {
        delta: String,
    },
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    AwaitingConfirm {
        call_id: String,
        summary: String,
        preview: serde_json::Value,
    },
    ToolResult {
        call_id: String,
        ok: bool,
        output_preview: serde_json::Value,
    },
    Final {
        message_id: String,
    },
    Error {
        code: String,
        msg: String,
    },
}

impl ChatEvent {
    /// Serialize for an SSE `data:` line. Errors collapse to an Error event payload.
    pub fn to_sse_data(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(
                "{{\"event\":\"error\",\"code\":\"serialize\",\"msg\":{:?}}}",
                e.to_string()
            )
        })
    }

    /// Short SSE event name (matches `event:` line in the stream).
    pub fn name(&self) -> &'static str {
        match self {
            ChatEvent::SessionStarted { .. } => "session_started",
            ChatEvent::UserPersisted { .. } => "user_persisted",
            ChatEvent::Token { .. } => "token",
            ChatEvent::ToolCall { .. } => "tool_call",
            ChatEvent::AwaitingConfirm { .. } => "awaiting_confirm",
            ChatEvent::ToolResult { .. } => "tool_result",
            ChatEvent::Final { .. } => "final",
            ChatEvent::Error { .. } => "error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_message_serialization() {
        let m = Message::user_text("s1", "hello");
        let s = serde_json::to_string(&m).unwrap();
        let parsed: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.session_id, "s1");
        assert_eq!(parsed.role, MessageRole::User);
        assert_eq!(parsed.content.len(), 1);
    }

    #[test]
    fn content_part_is_tagged() {
        let p = ContentPart::text("hi");
        let s = serde_json::to_string(&p).unwrap();
        assert!(s.contains("\"type\":\"text\""));
    }

    #[test]
    fn session_new_has_studio_ui_skill() {
        let scope = Scope::default_scope();
        let m = SessionMeta::new(&scope, "sid", "Untitled");
        assert_eq!(m.workspace_id, "default");
        assert_eq!(m.user_id, "local");
        assert_eq!(m.active_skills, vec!["studio-ui".to_string()]);
    }

    #[test]
    fn chat_event_sse_name() {
        let ev = ChatEvent::Token { delta: "x".into() };
        assert_eq!(ev.name(), "token");
        let s = ev.to_sse_data();
        assert!(s.contains("\"event\":\"token\""));
    }
}
