//! Persisted configuration for studio_chat.
//!
//! We intentionally reuse `api::server::LlmConfig` (same provider enum as
//! ingestion + inference) so users pick their chat provider from the same
//! dropdown shape. The stored key is independent, so chat can run on a
//! different model than ingestion / inference / RAG.

use crate::api::server::LlmConfig;
use crate::store::KVStore;

/// KV key for the chat-only LLM config. Independent of `cfg/llm` (ingestion)
/// and `cfg/inference_llm` (RAG/inference).
pub const CFG_CHAT_LLM_KEY: &[u8] = b"cfg/studio_chat_llm";

/// Load persisted chat LLM config, if any.
pub fn load_persisted_chat_llm_config(store: &dyn KVStore) -> Option<LlmConfig> {
    store
        .get(CFG_CHAT_LLM_KEY)
        .ok()
        .flatten()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}
