//! Session state for multi-turn RAG conversations.
//!
//! Stores conversation history (question/answer pairs) in KV at `sess/{session_id}`.
//! History is prepended to the RAG prompt for context-aware follow-ups.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::store::KVStore;

/// KV prefix for session state.
const SESSION_PREFIX: &str = "sess/";

/// Maximum number of history turns to keep (oldest are pruned).
const DEFAULT_MAX_TURNS: usize = 5;

/// A single conversation turn (question + answer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub question: String,
    pub answer: String,
    pub timestamp: DateTime<Utc>,
}

/// Session state for a multi-turn RAG conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub history: Vec<ConversationTurn>,
    pub last_accessed: DateTime<Utc>,
}

/// Load a session from KV. Returns None if not found.
pub fn load_session(store: &dyn KVStore, session_id: &str) -> Result<Option<SessionState>> {
    let key = format!("{}{}", SESSION_PREFIX, session_id);
    match store.get(key.as_bytes())? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Save a session to KV.
pub fn save_session(store: &dyn KVStore, session: &SessionState) -> Result<()> {
    let key = format!("{}{}", SESSION_PREFIX, session.session_id);
    let value = serde_json::to_vec(session)?;
    store.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Delete a session from KV.
pub fn delete_session(store: &dyn KVStore, session_id: &str) -> Result<bool> {
    let key = format!("{}{}", SESSION_PREFIX, session_id);
    if store.get(key.as_bytes())?.is_some() {
        store.delete(key.as_bytes())?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Append a turn to a session, creating it if needed. Trims to max_turns.
pub fn append_turn(
    store: &dyn KVStore,
    session_id: &str,
    question: &str,
    answer: &str,
    max_turns: usize,
) -> Result<SessionState> {
    let mut session = load_session(store, session_id)?.unwrap_or_else(|| SessionState {
        session_id: session_id.to_string(),
        history: vec![],
        last_accessed: Utc::now(),
    });

    session.history.push(ConversationTurn {
        question: question.to_string(),
        answer: answer.to_string(),
        timestamp: Utc::now(),
    });

    // Trim to max_turns (keep most recent)
    let max = if max_turns == 0 {
        DEFAULT_MAX_TURNS
    } else {
        max_turns
    };
    if session.history.len() > max {
        let start = session.history.len() - max;
        session.history = session.history[start..].to_vec();
    }

    session.last_accessed = Utc::now();
    save_session(store, &session)?;
    Ok(session)
}

/// Format conversation history as a prompt prefix.
pub fn format_history(session: &SessionState) -> String {
    if session.history.is_empty() {
        return String::new();
    }
    let mut prompt = String::from("## Previous Conversation\n\n");
    for turn in &session.history {
        prompt.push_str(&format!(
            "**Q:** {}\n**A:** {}\n\n",
            turn.question, turn.answer
        ));
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn make_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    #[test]
    fn test_session_roundtrip() {
        let store = make_store();
        let session = SessionState {
            session_id: "s1".to_string(),
            history: vec![ConversationTurn {
                question: "Who is Alice?".to_string(),
                answer: "She is the protagonist.".to_string(),
                timestamp: Utc::now(),
            }],
            last_accessed: Utc::now(),
        };
        save_session(store.as_ref(), &session).unwrap();
        let loaded = load_session(store.as_ref(), "s1").unwrap().unwrap();
        assert_eq!(loaded.session_id, "s1");
        assert_eq!(loaded.history.len(), 1);
        assert_eq!(loaded.history[0].question, "Who is Alice?");
    }

    #[test]
    fn test_session_not_found() {
        let store = make_store();
        assert!(load_session(store.as_ref(), "nonexistent")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_append_turn() {
        let store = make_store();
        append_turn(store.as_ref(), "s2", "Q1", "A1", 5).unwrap();
        append_turn(store.as_ref(), "s2", "Q2", "A2", 5).unwrap();
        let session = load_session(store.as_ref(), "s2").unwrap().unwrap();
        assert_eq!(session.history.len(), 2);
        assert_eq!(session.history[1].question, "Q2");
    }

    #[test]
    fn test_append_turn_trims() {
        let store = make_store();
        for i in 0..10 {
            append_turn(
                store.as_ref(),
                "s3",
                &format!("Q{}", i),
                &format!("A{}", i),
                3,
            )
            .unwrap();
        }
        let session = load_session(store.as_ref(), "s3").unwrap().unwrap();
        assert_eq!(session.history.len(), 3);
        assert_eq!(session.history[0].question, "Q7"); // Oldest kept
    }

    #[test]
    fn test_delete_session() {
        let store = make_store();
        append_turn(store.as_ref(), "s4", "Q", "A", 5).unwrap();
        assert!(delete_session(store.as_ref(), "s4").unwrap());
        assert!(!delete_session(store.as_ref(), "s4").unwrap());
        assert!(load_session(store.as_ref(), "s4").unwrap().is_none());
    }

    #[test]
    fn test_format_history() {
        let session = SessionState {
            session_id: "s5".to_string(),
            history: vec![ConversationTurn {
                question: "Who is Bob?".to_string(),
                answer: "He is a detective.".to_string(),
                timestamp: Utc::now(),
            }],
            last_accessed: Utc::now(),
        };
        let formatted = format_history(&session);
        assert!(formatted.contains("Previous Conversation"));
        assert!(formatted.contains("Who is Bob?"));
        assert!(formatted.contains("He is a detective."));
    }

    #[test]
    fn test_format_history_empty() {
        let session = SessionState {
            session_id: "s6".to_string(),
            history: vec![],
            last_accessed: Utc::now(),
        };
        assert!(format_history(&session).is_empty());
    }
}
