//! LLM response caching layer.
//!
//! Caches raw LLM responses keyed by the SHA-256 hash of the prompt pair
//! (system + user). Backed by the KV store under the `lc/` prefix.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::ingestion::chunker::TextChunk;
use crate::ingestion::extraction::{parse_llm_response, NarrativeExtraction};
use crate::ingestion::hex_encode;
use crate::ingestion::llm::{NarrativeExtractor, RawLlmExchange};
use crate::store::KVStore;

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cached entries.
    pub entries: usize,
    /// Total size of cached values in bytes.
    pub total_bytes: usize,
}

/// KV-backed LLM response cache.
///
/// Keys are `lc/{sha256_hex}` where the hash covers the concatenation
/// of the system prompt, a separator, and the user prompt.
pub struct LlmCache {
    store: Arc<dyn KVStore>,
}

const PREFIX: &[u8] = b"lc/";

impl LlmCache {
    /// Create a new cache backed by the given KV store.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Compute the KV key for a prompt pair.
    pub fn cache_key(system_prompt: &str, user_prompt: &str) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(system_prompt.as_bytes());
        hasher.update(b"\n---\n");
        hasher.update(user_prompt.as_bytes());
        let hash = hasher.finalize();
        let hex = hex_encode(&hash);
        let mut key = Vec::with_capacity(3 + hex.len());
        key.extend_from_slice(PREFIX);
        key.extend_from_slice(hex.as_bytes());
        key
    }

    /// Look up a cached response for the given prompt pair.
    pub fn get(&self, system: &str, user: &str) -> Result<Option<String>> {
        let key = Self::cache_key(system, user);
        match self.store.get(&key)? {
            Some(bytes) => {
                let s = String::from_utf8(bytes).map_err(|e| {
                    crate::error::TensaError::Store(format!("Invalid UTF-8 in cache: {}", e))
                })?;
                Ok(Some(s))
            }
            None => Ok(None),
        }
    }

    /// Store a response for the given prompt pair.
    pub fn put(&self, system: &str, user: &str, response: &str) -> Result<()> {
        let key = Self::cache_key(system, user);
        self.store.put(&key, response.as_bytes())?;
        Ok(())
    }

    /// Remove a cached response for the given prompt pair.
    pub fn invalidate(&self, system: &str, user: &str) -> Result<()> {
        let key = Self::cache_key(system, user);
        self.store.delete(&key)?;
        Ok(())
    }

    /// Remove all cached responses. Returns the number of entries cleared.
    pub fn clear(&self) -> Result<usize> {
        let entries = self.store.prefix_scan(PREFIX)?;
        let count = entries.len();
        for (key, _) in &entries {
            self.store.delete(key)?;
        }
        Ok(count)
    }

    /// Compute cache statistics (entry count and total bytes).
    pub fn stats(&self) -> Result<CacheStats> {
        let entries = self.store.prefix_scan(PREFIX)?;
        let total_bytes: usize = entries.iter().map(|(_, v)| v.len()).sum();
        Ok(CacheStats {
            entries: entries.len(),
            total_bytes,
        })
    }
}

/// A caching wrapper around any `NarrativeExtractor`.
///
/// On extraction, checks the LLM cache first. On miss, delegates to the
/// inner extractor and caches the raw response for future reuse.
pub struct CachedExtractor {
    inner: Arc<dyn NarrativeExtractor>,
    cache: LlmCache,
}

const CACHE_SYSTEM_MARKER: &str = "tensa:extract";

impl CachedExtractor {
    /// Create a new caching extractor wrapping `inner`.
    pub fn new(inner: Arc<dyn NarrativeExtractor>, cache: LlmCache) -> Self {
        Self { inner, cache }
    }

    fn user_prompt_key(chunk: &TextChunk, known_entities: &[String]) -> String {
        format!(
            "chunk:{}\nentities:{}",
            chunk.text,
            known_entities.join(",")
        )
    }
}

impl NarrativeExtractor for CachedExtractor {
    fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction> {
        self.extract_with_context(chunk, &[])
    }

    fn extract_with_context(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<NarrativeExtraction> {
        let (ext, _) = self.extract_with_logging(chunk, known_entities)?;
        Ok(ext)
    }

    fn extract_with_logging(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
        let user_key = Self::user_prompt_key(chunk, known_entities);

        // Check cache
        if let Some(cached_response) = self.cache.get(CACHE_SYSTEM_MARKER, &user_key)? {
            tracing::debug!("LLM cache hit for chunk (len={})", chunk.text.len());
            let extraction = parse_llm_response(&cached_response)?;
            return Ok((extraction, None));
        }

        // Cache miss — delegate to inner extractor
        tracing::debug!("LLM cache miss for chunk (len={})", chunk.text.len());
        let (extraction, exchange) = self.inner.extract_with_logging(chunk, known_entities)?;

        // Cache the raw response if we got an exchange
        if let Some(ref ex) = exchange {
            let _ = self
                .cache
                .put(CACHE_SYSTEM_MARKER, &user_key, &ex.raw_response);
        }

        Ok((extraction, exchange))
    }

    fn set_cancel_flag(&self, flag: Arc<AtomicBool>) {
        self.inner.set_cancel_flag(flag);
    }

    fn model_name(&self) -> Option<String> {
        self.inner.model_name()
    }

    fn answer_question(&self, system_prompt: &str, question: &str) -> crate::error::Result<String> {
        let cache_key = format!("ask:{}", question);
        // Check cache
        if let Some(cached) = self.cache.get(system_prompt, &cache_key)? {
            tracing::debug!("RAG cache hit for question (len={})", question.len());
            return Ok(cached);
        }
        // Cache miss — delegate
        let answer = self.inner.answer_question(system_prompt, question)?;
        let _ = self.cache.put(system_prompt, &cache_key, &answer);
        Ok(answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Mutex;

    #[test]
    fn test_cache_put_get() {
        let store = Arc::new(MemoryStore::new());
        let cache = LlmCache::new(store);
        cache.put("sys", "user", "response-text").unwrap();
        let result = cache.get("sys", "user").unwrap();
        assert_eq!(result, Some("response-text".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let store = Arc::new(MemoryStore::new());
        let cache = LlmCache::new(store);
        let result = cache.get("sys", "unknown").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_invalidate() {
        let store = Arc::new(MemoryStore::new());
        let cache = LlmCache::new(store);
        cache.put("sys", "user", "data").unwrap();
        assert!(cache.get("sys", "user").unwrap().is_some());
        cache.invalidate("sys", "user").unwrap();
        assert!(cache.get("sys", "user").unwrap().is_none());
    }

    #[test]
    fn test_cache_stats() {
        let store = Arc::new(MemoryStore::new());
        let cache = LlmCache::new(store);
        cache.put("sys", "a", "val1").unwrap();
        cache.put("sys", "b", "val2").unwrap();
        cache.put("sys", "c", "val3").unwrap();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entries, 3);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_cache_clear() {
        let store = Arc::new(MemoryStore::new());
        let cache = LlmCache::new(store);
        cache.put("sys", "a", "v1").unwrap();
        cache.put("sys", "b", "v2").unwrap();
        let count = cache.clear().unwrap();
        assert_eq!(count, 2);
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entries, 0);
    }

    /// Mock extractor that counts calls.
    struct MockExtractor {
        call_count: Mutex<usize>,
    }

    impl MockExtractor {
        fn new() -> Self {
            Self {
                call_count: Mutex::new(0),
            }
        }

        fn calls(&self) -> usize {
            *self.call_count.lock().unwrap()
        }
    }

    impl NarrativeExtractor for MockExtractor {
        fn extract_narrative(&self, _chunk: &TextChunk) -> Result<NarrativeExtraction> {
            Ok(NarrativeExtraction {
                entities: vec![],
                situations: vec![],
                participations: vec![],
                causal_links: vec![],
                temporal_relations: vec![],
            })
        }

        fn extract_with_logging(
            &self,
            chunk: &TextChunk,
            _known_entities: &[String],
        ) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
            *self.call_count.lock().unwrap() += 1;
            let extraction = NarrativeExtraction {
                entities: vec![],
                situations: vec![],
                participations: vec![],
                causal_links: vec![],
                temporal_relations: vec![],
            };
            let exchange = RawLlmExchange {
                system_prompt: "sys".into(),
                user_prompt: chunk.text.clone(),
                raw_response:
                    r#"{"entities":[],"situations":[],"participations":[],"causal_links":[]}"#
                        .into(),
                retry_prompt: None,
                retry_response: None,
                parse_error: None,
                duration_ms: 100,
                model: None,
                endpoint: None,
            };
            Ok((extraction, Some(exchange)))
        }

        fn set_cancel_flag(&self, _flag: Arc<AtomicBool>) {}
        fn model_name(&self) -> Option<String> {
            Some("mock".into())
        }
    }

    #[test]
    fn test_cached_extractor_cache_hit() {
        let store = Arc::new(MemoryStore::new());
        let cache = LlmCache::new(store.clone());
        let mock = Arc::new(MockExtractor::new());
        let cached = CachedExtractor::new(mock.clone() as Arc<dyn NarrativeExtractor>, cache);

        let chunk = TextChunk {
            chunk_id: 0,
            text: "Test text".into(),
            chapter: None,
            start_offset: 0,
            end_offset: 9,
            overlap_prefix: String::new(),
        };

        // First call — cache miss, delegates to mock
        let (_ext1, _) = cached.extract_with_logging(&chunk, &[]).unwrap();
        assert_eq!(mock.calls(), 1);

        // Second call — cache hit, mock not called again
        let (_ext2, _) = cached.extract_with_logging(&chunk, &[]).unwrap();
        assert_eq!(mock.calls(), 1); // still 1, not 2
    }
}
