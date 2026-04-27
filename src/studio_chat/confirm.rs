//! Confirmation gate for mutating tool calls.
//!
//! Phase 3 classifies each tool call as `Free` or `Confirm`. Free calls
//! execute immediately. `Confirm` calls park on a `oneshot` channel and
//! emit an `awaiting_confirm` SSE event; the harness resumes when the user
//! responds via `POST /studio/chat/sessions/:id/confirm`.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::oneshot;

use super::types::Scope;

/// Decision returned by the user for a pending confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmDecision {
    Approve,
    Reject,
}

/// Free = run immediately. Confirm = park until the user approves/rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    Free,
    Confirm,
}

#[derive(Debug)]
pub struct PendingKey {
    pub workspace_id: String,
    pub user_id: String,
    pub session_id: String,
    pub call_id: String,
}

impl PendingKey {
    pub fn new(scope: &Scope, session_id: &str, call_id: &str) -> Self {
        Self {
            workspace_id: scope.workspace_id.clone(),
            user_id: scope.user_id.clone(),
            session_id: session_id.to_string(),
            call_id: call_id.to_string(),
        }
    }

    fn tuple(&self) -> (String, String, String, String) {
        (
            self.workspace_id.clone(),
            self.user_id.clone(),
            self.session_id.clone(),
            self.call_id.clone(),
        )
    }
}

/// Routes pending confirmations to the harness future waiting on them.
/// A single `ConfirmGate` is shared across the whole server via AppState.
#[derive(Default)]
pub struct ConfirmGate {
    pending: Mutex<HashMap<(String, String, String, String), oneshot::Sender<ConfirmDecision>>>,
}

impl ConfirmGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pending confirmation and return the receiver. The harness
    /// awaits the returned `oneshot::Receiver`; the `/confirm` handler
    /// resolves it via `settle`.
    pub fn register(&self, key: PendingKey) -> oneshot::Receiver<ConfirmDecision> {
        let (tx, rx) = oneshot::channel();
        let mut guard = self.pending.lock().expect("ConfirmGate mutex poisoned");
        guard.insert(key.tuple(), tx);
        rx
    }

    /// Fulfill a pending confirmation. Returns `true` if a pending future
    /// was found and resolved; `false` if the call_id was unknown or the
    /// harness dropped its receiver first.
    pub fn settle(&self, key: PendingKey, decision: ConfirmDecision) -> bool {
        let mut guard = self.pending.lock().expect("ConfirmGate mutex poisoned");
        if let Some(tx) = guard.remove(&key.tuple()) {
            tx.send(decision).is_ok()
        } else {
            false
        }
    }

    /// Drop a pending entry without notifying (harness aborted, session
    /// deleted, etc). Safe to call on an absent key.
    pub fn forget(&self, key: &PendingKey) {
        let mut guard = self.pending.lock().expect("ConfirmGate mutex poisoned");
        guard.remove(&key.tuple());
    }
}

/// Classification table: tool names that need user confirmation before the
/// call runs. Everything not in the set is `Free`. Phase 3 keeps this list
/// explicit; Phase 5 exposes it in `ChatConfig` so users can whitelist
/// individual tools.
pub fn classify(tool_name: &str) -> ToolClass {
    match tool_name {
        // Read-only / navigation
        "list_narratives" | "get_narrative" | "list_entities" | "get_entity"
        | "list_situations" | "get_situation" | "query_tensaql" | "narrative_ask"
        | "navigate_to" | "show_toast" | "list_skills" => ToolClass::Free,
        // Mutating — explicit whitelist below, rest default to Confirm.
        _ => ToolClass::Confirm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(call: &str) -> PendingKey {
        PendingKey::new(&Scope::default_scope(), "sid", call)
    }

    #[tokio::test]
    async fn settle_resolves_pending_future() {
        let gate = ConfirmGate::new();
        let rx = gate.register(key("c1"));
        assert!(gate.settle(key("c1"), ConfirmDecision::Approve));
        assert_eq!(rx.await.unwrap(), ConfirmDecision::Approve);
    }

    #[tokio::test]
    async fn settle_unknown_returns_false() {
        let gate = ConfirmGate::new();
        assert!(!gate.settle(key("never"), ConfirmDecision::Reject));
    }

    #[test]
    fn forget_drops_without_error() {
        let gate = ConfirmGate::new();
        let _rx = gate.register(key("c2"));
        gate.forget(&key("c2"));
        gate.forget(&key("c2"));
    }

    #[test]
    fn classify_known_read_tools_are_free() {
        assert_eq!(classify("list_narratives"), ToolClass::Free);
        assert_eq!(classify("query_tensaql"), ToolClass::Free);
        assert_eq!(classify("navigate_to"), ToolClass::Free);
    }

    #[test]
    fn classify_unknown_defaults_to_confirm() {
        assert_eq!(classify("create_entity"), ToolClass::Confirm);
        assert_eq!(classify("ingest_text"), ToolClass::Confirm);
        assert_eq!(classify("some_future_mutation"), ToolClass::Confirm);
    }
}
