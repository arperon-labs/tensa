//! KV persistence for studio_chat sessions and messages.
//!
//! All keys live on the root KV store (not workspace-prefixed). Workspace and
//! user are part of the key itself so admin tooling can later enumerate across
//! workspaces without a WorkspaceStore wrapper.

use std::sync::Arc;

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;

use super::types::{Message, Scope, SessionMeta};

fn validate_segment(kind: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(TensaError::InvalidInput(format!("{} is required", kind)));
    }
    if value.len() > 64 {
        return Err(TensaError::InvalidInput(format!(
            "{} must be at most 64 chars",
            kind
        )));
    }
    if value.contains('/') {
        return Err(TensaError::InvalidInput(format!(
            "{} must not contain '/'",
            kind
        )));
    }
    Ok(())
}

fn validate_scope(scope: &Scope) -> Result<()> {
    validate_segment("workspace_id", &scope.workspace_id)?;
    validate_segment("user_id", &scope.user_id)?;
    Ok(())
}

fn session_key(scope: &Scope, session_id: &str) -> String {
    format!(
        "chat/s/{}/{}/{}",
        scope.workspace_id, scope.user_id, session_id
    )
}

fn session_prefix(scope: &Scope) -> String {
    format!("chat/s/{}/{}/", scope.workspace_id, scope.user_id)
}

fn message_key(scope: &Scope, session_id: &str, msg_id: &Uuid) -> String {
    // UUID is formatted as hex without hyphens so the v7 byte ordering is
    // preserved in lexicographic key order.
    format!(
        "chat/m/{}/{}/{}/{}",
        scope.workspace_id,
        scope.user_id,
        session_id,
        msg_id.simple()
    )
}

fn message_prefix(scope: &Scope, session_id: &str) -> String {
    format!(
        "chat/m/{}/{}/{}/",
        scope.workspace_id, scope.user_id, session_id
    )
}

/// Facade over the root KV store that handles chat session + message persistence.
pub struct ChatStore {
    store: Arc<dyn KVStore>,
}

impl ChatStore {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    pub fn create_session(&self, scope: &Scope, meta: &SessionMeta) -> Result<()> {
        validate_scope(scope)?;
        let key = session_key(scope, &meta.id);
        let val = serde_json::to_vec(meta).map_err(|e| TensaError::Serialization(e.to_string()))?;
        self.store.put(key.as_bytes(), &val)?;
        Ok(())
    }

    pub fn get_session(&self, scope: &Scope, session_id: &str) -> Result<Option<SessionMeta>> {
        validate_scope(scope)?;
        let key = session_key(scope, session_id);
        match self.store.get(key.as_bytes())? {
            Some(bytes) => {
                let meta: SessionMeta = serde_json::from_slice(&bytes)
                    .map_err(|e| TensaError::Serialization(e.to_string()))?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    pub fn update_session(&self, scope: &Scope, meta: &SessionMeta) -> Result<()> {
        // Same as create; create_session is idempotent.
        self.create_session(scope, meta)
    }

    pub fn list_sessions(&self, scope: &Scope) -> Result<Vec<SessionMeta>> {
        validate_scope(scope)?;
        let prefix = session_prefix(scope);
        let entries = self.store.prefix_scan(prefix.as_bytes())?;
        let mut sessions: Vec<SessionMeta> = entries
            .into_iter()
            .filter_map(|(_, v)| serde_json::from_slice::<SessionMeta>(&v).ok())
            .collect();
        // Newest first.
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    pub fn delete_session(&self, scope: &Scope, session_id: &str) -> Result<bool> {
        validate_scope(scope)?;
        let s_key = session_key(scope, session_id);
        let existed = self.store.get(s_key.as_bytes())?.is_some();
        if existed {
            self.store.delete(s_key.as_bytes())?;
        }
        // Delete all messages regardless (tolerate orphan messages if any).
        let m_prefix = message_prefix(scope, session_id);
        let entries = self.store.prefix_scan(m_prefix.as_bytes())?;
        for (k, _) in entries {
            let _ = self.store.delete(&k);
        }
        Ok(existed)
    }

    pub fn append_message(&self, scope: &Scope, msg: &Message) -> Result<()> {
        validate_scope(scope)?;
        let key = message_key(scope, &msg.session_id, &msg.id);
        let val = serde_json::to_vec(msg).map_err(|e| TensaError::Serialization(e.to_string()))?;
        self.store.put(key.as_bytes(), &val)?;
        Ok(())
    }

    pub fn list_messages(&self, scope: &Scope, session_id: &str) -> Result<Vec<Message>> {
        validate_scope(scope)?;
        let prefix = message_prefix(scope, session_id);
        let entries = self.store.prefix_scan(prefix.as_bytes())?;
        let mut messages: Vec<Message> = entries
            .into_iter()
            .filter_map(|(_, v)| serde_json::from_slice::<Message>(&v).ok())
            .collect();
        // prefix_scan already returns lexicographic order == v7 chronological.
        messages.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(messages)
    }

    /// Bump `updated_at` and `msg_count` on a session after a new message.
    pub fn bump_session(&self, scope: &Scope, session_id: &str, delta_msgs: u32) -> Result<()> {
        if let Some(mut meta) = self.get_session(scope, session_id)? {
            meta.updated_at = chrono::Utc::now();
            meta.msg_count = meta.msg_count.saturating_add(delta_msgs);
            self.update_session(scope, &meta)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{ContentPart, Message, MessageRole, SessionMeta};
    use super::*;
    use crate::store::memory::MemoryStore;

    fn mk_store() -> ChatStore {
        ChatStore::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn session_roundtrip() {
        let store = mk_store();
        let scope = Scope::default_scope();
        let meta = SessionMeta::new(&scope, "sid-1", "Title");
        store.create_session(&scope, &meta).unwrap();
        let loaded = store.get_session(&scope, "sid-1").unwrap().unwrap();
        assert_eq!(loaded.id, "sid-1");
        assert_eq!(loaded.title, "Title");
    }

    #[test]
    fn list_sessions_is_workspace_user_scoped() {
        let store = mk_store();
        let alice = Scope::new("default", "alice");
        let bob = Scope::new("default", "bob");
        store
            .create_session(&alice, &SessionMeta::new(&alice, "a1", "Alice 1"))
            .unwrap();
        store
            .create_session(&bob, &SessionMeta::new(&bob, "b1", "Bob 1"))
            .unwrap();

        let alice_list = store.list_sessions(&alice).unwrap();
        let bob_list = store.list_sessions(&bob).unwrap();
        assert_eq!(alice_list.len(), 1);
        assert_eq!(bob_list.len(), 1);
        assert_eq!(alice_list[0].id, "a1");
        assert_eq!(bob_list[0].id, "b1");
    }

    #[test]
    fn list_sessions_is_workspace_scoped() {
        let store = mk_store();
        let acme = Scope::new("acme", "local");
        let widgets = Scope::new("widgets", "local");
        store
            .create_session(&acme, &SessionMeta::new(&acme, "a1", "Acme 1"))
            .unwrap();
        store
            .create_session(&widgets, &SessionMeta::new(&widgets, "w1", "W 1"))
            .unwrap();
        assert_eq!(store.list_sessions(&acme).unwrap().len(), 1);
        assert_eq!(store.list_sessions(&widgets).unwrap().len(), 1);
    }

    #[test]
    fn append_and_list_messages_in_order() {
        let store = mk_store();
        let scope = Scope::default_scope();
        let meta = SessionMeta::new(&scope, "sid-m", "m");
        store.create_session(&scope, &meta).unwrap();
        let u = Message::user_text("sid-m", "first");
        let a = Message::assistant_text("sid-m", "second");
        store.append_message(&scope, &u).unwrap();
        store.append_message(&scope, &a).unwrap();
        let msgs = store.list_messages(&scope, "sid-m").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::User);
        assert_eq!(msgs[1].role, MessageRole::Assistant);
    }

    #[test]
    fn delete_session_removes_messages() {
        let store = mk_store();
        let scope = Scope::default_scope();
        let meta = SessionMeta::new(&scope, "sid-d", "d");
        store.create_session(&scope, &meta).unwrap();
        store
            .append_message(&scope, &Message::user_text("sid-d", "hi"))
            .unwrap();
        assert!(store.delete_session(&scope, "sid-d").unwrap());
        assert!(store.get_session(&scope, "sid-d").unwrap().is_none());
        assert_eq!(store.list_messages(&scope, "sid-d").unwrap().len(), 0);
    }

    #[test]
    fn bump_session_updates_counts() {
        let store = mk_store();
        let scope = Scope::default_scope();
        let meta = SessionMeta::new(&scope, "sid-b", "b");
        store.create_session(&scope, &meta).unwrap();
        store.bump_session(&scope, "sid-b", 2).unwrap();
        let loaded = store.get_session(&scope, "sid-b").unwrap().unwrap();
        assert_eq!(loaded.msg_count, 2);
    }

    #[test]
    fn rejects_scope_with_slash() {
        let store = mk_store();
        let bad = Scope::new("has/slash", "local");
        let err = store
            .create_session(&bad, &SessionMeta::new(&bad, "x", "x"))
            .unwrap_err();
        assert!(matches!(err, TensaError::InvalidInput(_)));
    }

    #[test]
    fn tool_and_text_parts_roundtrip() {
        let store = mk_store();
        let scope = Scope::default_scope();
        let meta = SessionMeta::new(&scope, "sid-c", "c");
        store.create_session(&scope, &meta).unwrap();
        let msg = Message::new(
            "sid-c",
            MessageRole::Assistant,
            vec![
                ContentPart::text("calling tool"),
                ContentPart::ToolCall {
                    id: "c1".into(),
                    name: "list_narratives".into(),
                    args: serde_json::json!({}),
                },
            ],
        );
        store.append_message(&scope, &msg).unwrap();
        let loaded = store.list_messages(&scope, "sid-c").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content.len(), 2);
    }
}
