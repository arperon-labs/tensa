//! LLM client for narrative extraction via the Anthropic Claude API.
//!
//! Provides a trait-based interface (`NarrativeExtractor`) so the pipeline
//! can work with both real and mock extractors.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::ingestion::chunker::TextChunk;
use crate::ingestion::extraction::{
    parse_enrichment_response, parse_llm_response, parse_reconciliation_response,
    repair_enrichment, ExtractionEnrichment, NarrativeExtraction, TemporalReconciliation,
};

/// Raw data from an LLM exchange, captured for debugging/logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawLlmExchange {
    /// System prompt sent to the LLM.
    pub system_prompt: String,
    /// User prompt sent to the LLM.
    pub user_prompt: String,
    /// Raw text response from the LLM.
    pub raw_response: String,
    /// Repair prompt sent on parse failure (if any).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_prompt: Option<String>,
    /// Retry response text (if repair was attempted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_response: Option<String>,
    /// First parse error (if parse failed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parse_error: Option<String>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Model name used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Endpoint URL called.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
}

/// Trait for extracting structured narrative data from text chunks.
pub trait NarrativeExtractor: Send + Sync {
    /// Extract entities, situations, participations, etc. from a text chunk.
    fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction>;

    /// Extract with context: pass known entity names from previous chunks
    /// for cross-chunk coherence. Default delegates to extract_narrative.
    fn extract_with_context(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<NarrativeExtraction> {
        // Default: ignore context, just extract
        let _ = known_entities;
        self.extract_narrative(chunk)
    }

    /// Extract with full logging output. Returns the extraction plus raw exchange data.
    /// Default delegates to extract_with_context and returns None for exchange.
    fn extract_with_logging(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
        let ext = self.extract_with_context(chunk, known_entities)?;
        Ok((ext, None))
    }

    /// Attach a cancellation flag so in-flight LLM requests can be aborted.
    /// Default no-op for test mocks.
    fn set_cancel_flag(&self, _flag: Arc<AtomicBool>) {}

    /// Return the model name being used for extraction (for progress display).
    fn model_name(&self) -> Option<String> {
        None
    }

    /// Cross-chunk temporal reconciliation: given situation summaries from all chunks,
    /// produce a global timeline with Allen relations and pinned dates.
    /// Default returns empty reconciliation (test mocks).
    fn reconcile_temporal(
        &self,
        _chunk_summaries: &[(usize, Vec<(String, Option<String>, Option<String>)>)],
    ) -> Result<TemporalReconciliation> {
        Ok(TemporalReconciliation::default())
    }

    /// Step 2: Enrich an existing extraction with deep annotations.
    /// Called in the same logical context as step 1. The LLM sees the step-1
    /// extraction summary + original text and adds beliefs, game structures,
    /// discourse, info_sets, and outcome models.
    /// Default returns empty enrichment (test mocks, NLP extractor).
    fn enrich_extraction(
        &self,
        _chunk: &TextChunk,
        _extraction: &NarrativeExtraction,
    ) -> Result<ExtractionEnrichment> {
        Ok(ExtractionEnrichment::default())
    }

    /// Answer a question given a system prompt with context.
    /// Used by the RAG pipeline. Default returns an error.
    fn answer_question(&self, _system_prompt: &str, _question: &str) -> Result<String> {
        Err(TensaError::LlmError(
            "answer_question not supported by this extractor".into(),
        ))
    }

    /// Batch-canonicalize raw place strings using narrative-setting context.
    ///
    /// Default impl reuses `answer_question` so every LLM client that already
    /// supports RAG also supports geocoding canonicalization without per-client
    /// wiring. Mocks / NLP extractors that don't override `answer_question`
    /// return an empty vec — geocoding then falls through to direct Nominatim.
    fn canonicalize_places(
        &self,
        setting: &crate::ingestion::extraction::NarrativeSettingHint,
        places: &[(String, String)],
    ) -> Result<Vec<crate::ingestion::extraction::PlaceCanonicalization>> {
        if places.is_empty() {
            return Ok(vec![]);
        }
        let user_prompt = build_canonicalize_places_prompt(setting, places);
        let raw = match self.answer_question(PLACE_CANONICALIZATION_PROMPT, &user_prompt) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    requested = places.len(),
                    "canonicalize_places: answer_question failed — falling back to direct Nominatim"
                );
                return Ok(vec![]);
            }
        };
        match crate::ingestion::extraction::parse_canonicalize_places_response(&raw) {
            Ok(rows) => {
                if rows.len() < places.len() {
                    let head: String = raw.chars().take(400).collect();
                    tracing::warn!(
                        count = rows.len(),
                        requested = places.len(),
                        head = %head,
                        "canonicalize_places: LLM returned fewer rows than requested"
                    );
                } else {
                    tracing::info!(
                        count = rows.len(),
                        requested = places.len(),
                        "canonicalize_places: parsed LLM response"
                    );
                }
                Ok(rows)
            }
            Err(_) => {
                // Parse error already logged inside parse_canonicalize_places_response.
                Ok(vec![])
            }
        }
    }

    /// Return self as a session-capable extractor, if multi-turn is supported.
    /// Default returns None. Override in OpenRouterClient and LocalLLMClient.
    fn as_session(&self) -> Option<&dyn SessionCapableExtractor> {
        None
    }
}

/// Trait for extractors that support multi-turn LLM sessions.
/// Only OpenRouterClient and LocalLLMClient implement this.
pub trait SessionCapableExtractor: NarrativeExtractor {
    /// Send a multi-turn conversation (system + user/assistant turns) and return the response.
    fn send_session_messages(&self, messages: &[ApiMessage]) -> Result<String>;
}

/// Poll a cancel flag every 250ms, returning when cancellation is requested.
async fn poll_cancel(flag: &AtomicBool) {
    loop {
        tokio::time::sleep(Duration::from_millis(250)).await;
        if flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

/// Race an async future against a cancellation flag.
/// Returns `Err(LlmError("Cancelled by user"))` if the flag is set before the future completes.
async fn with_cancel<T>(
    fut: impl std::future::Future<Output = T>,
    flag: &AtomicBool,
) -> std::result::Result<T, TensaError> {
    tokio::select! {
        result = fut => Ok(result),
        _ = poll_cancel(flag) => Err(TensaError::LlmError("Cancelled by user".into())),
    }
}

/// Claude API client with retry and rate limiting.
pub struct ClaudeClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_retries: u32,
    rate_limiter: RateLimiter,
    system_prompt: String,
    cancel_flag: Mutex<Arc<AtomicBool>>,
}

/// Simple rate limiter using a token bucket approach.
pub struct RateLimiter {
    requests_per_minute: u32,
    state: Mutex<RateLimiterState>,
}

struct RateLimiterState {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            state: Mutex::new(RateLimiterState {
                tokens: requests_per_minute as f64,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Try to acquire a token. Returns estimated wait time if rate limited.
    pub fn try_acquire(&self) -> std::result::Result<(), Duration> {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill);
        let refill = elapsed.as_secs_f64() * (self.requests_per_minute as f64 / 60.0);
        state.tokens = (state.tokens + refill).min(self.requests_per_minute as f64);
        state.last_refill = now;

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            Ok(())
        } else {
            let wait_secs = (1.0 - state.tokens) / (self.requests_per_minute as f64 / 60.0);
            Err(Duration::from_secs_f64(wait_secs))
        }
    }
}

#[derive(Clone, Serialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<ApiMessage>,
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentItem>,
}

#[derive(Deserialize)]
struct ContentItem {
    text: Option<String>,
}

impl ClaudeClient {
    /// Create a new Claude API client with default rate limit (40 RPM) and max retries (3).
    pub fn new(api_key: String, model: String) -> Self {
        Self::with_config(api_key, model, 40, 3)
    }

    /// Create a client with configurable rate limit and max retries.
    pub fn with_config(
        api_key: String,
        model: String,
        rate_limit_rpm: u32,
        max_retries: u32,
    ) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            api_key,
            model,
            max_retries,
            rate_limiter: RateLimiter::new(rate_limit_rpm),
            system_prompt: EXTRACTION_SYSTEM_PROMPT.to_string(),
            cancel_flag: Mutex::new(Arc::new(AtomicBool::new(false))),
        }
    }

    /// Set custom system prompt for extraction.
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Set custom rate limit.
    pub fn with_rate_limit(mut self, rpm: u32) -> Self {
        self.rate_limiter = RateLimiter::new(rpm);
        self
    }

    /// Send a message to the Claude API with retry logic.
    async fn send_message(&self, user_message: &str) -> Result<String> {
        let mut last_error = None;
        let flag = self.cancel_flag.lock().unwrap().clone();

        for attempt in 0..=self.max_retries {
            if flag.load(Ordering::Relaxed) {
                return Err(TensaError::LlmError("Cancelled by user".into()));
            }

            // Rate limiting
            if let Err(wait) = self.rate_limiter.try_acquire() {
                tokio::time::sleep(wait).await;
                if let Err(wait2) = self.rate_limiter.try_acquire() {
                    tokio::time::sleep(wait2).await;
                }
            }

            let request = ApiRequest {
                model: self.model.clone(),
                max_tokens: 16384,
                system: self.system_prompt.clone(),
                messages: vec![ApiMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                }],
            };

            let result = match with_cancel(
                self.client
                    .post("https://api.anthropic.com/v1/messages")
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .json(&request)
                    .send(),
                &flag,
            )
            .await
            {
                Err(cancel_err) => return Err(cancel_err),
                Ok(http_result) => http_result,
            };

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let body: ApiResponse = resp.json().await.map_err(|e| {
                            TensaError::LlmError(format!("Failed to parse response: {}", e))
                        })?;
                        let text = body
                            .content
                            .into_iter()
                            .filter_map(|c| c.text)
                            .collect::<Vec<_>>()
                            .join("");
                        return Ok(text);
                    } else if status.as_u16() == 429 {
                        let retry_secs = 2u64.pow(attempt);
                        last_error = Some(TensaError::LlmRateLimit {
                            retry_after_secs: retry_secs,
                        });
                        tokio::time::sleep(Duration::from_secs(retry_secs)).await;
                        continue;
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        last_error =
                            Some(TensaError::LlmError(format!("HTTP {}: {}", status, body)));
                        if status.is_server_error() {
                            let backoff = Duration::from_secs(2u64.pow(attempt));
                            tokio::time::sleep(backoff).await;
                            continue;
                        }
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(TensaError::LlmError(format!("Request failed: {}", e)));
                    let backoff = Duration::from_secs(2u64.pow(attempt));
                    tokio::time::sleep(backoff).await;
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| TensaError::LlmError("Unknown error".into())))
    }
}

impl ClaudeClient {
    /// Send a message to the Claude API with a custom system prompt.
    /// Used by the RAG pipeline to override the extraction system prompt.
    async fn send_message_with_system(&self, system: &str, user_message: &str) -> Result<String> {
        let flag = self.cancel_flag.lock().unwrap().clone();
        if let Err(wait) = self.rate_limiter.try_acquire() {
            tokio::time::sleep(wait).await;
            let _ = self.rate_limiter.try_acquire();
        }

        let request = ApiRequest {
            model: self.model.clone(),
            max_tokens: 16384,
            system: system.to_string(),
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: user_message.to_string(),
            }],
        };

        let result = match with_cancel(
            self.client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&request)
                .send(),
            &flag,
        )
        .await
        {
            Err(cancel_err) => return Err(cancel_err),
            Ok(http_result) => http_result,
        };

        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let body: ApiResponse = resp.json().await.map_err(|e| {
                        TensaError::LlmError(format!("Failed to parse response: {}", e))
                    })?;
                    Ok(body
                        .content
                        .into_iter()
                        .filter_map(|c| c.text)
                        .collect::<Vec<_>>()
                        .join(""))
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    Err(TensaError::LlmError(format!("HTTP {}: {}", status, body)))
                }
            }
            Err(e) => Err(TensaError::LlmError(format!("Request failed: {}", e))),
        }
    }

    fn extract_with_prompt(&self, user_message: String) -> Result<NarrativeExtraction> {
        let (ext, _) = self.extract_with_prompt_logged(user_message)?;
        Ok(ext)
    }

    fn extract_with_prompt_logged(
        &self,
        user_message: String,
    ) -> Result<(NarrativeExtraction, RawLlmExchange)> {
        let start = Instant::now();
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
        let response = rt
            .block_on(self.send_message(&user_message))
            .map_err(|e| TensaError::LlmError(format!("API call failed: {}", e)))?;

        let mut exchange = RawLlmExchange {
            system_prompt: self.system_prompt.clone(),
            user_prompt: user_message.clone(),
            raw_response: response.clone(),
            retry_prompt: None,
            retry_response: None,
            parse_error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            model: Some(self.model.clone()),
            endpoint: Some("https://api.anthropic.com/v1/messages".into()),
        };

        match parse_llm_response(&response) {
            Ok(ext) => {
                exchange.duration_ms = start.elapsed().as_millis() as u64;
                Ok((ext, exchange))
            }
            Err(first_err) => {
                exchange.parse_error = Some(format!("{}", first_err));
                let repair = format!(
                    "{}\n\nThe above JSON had a parse error: {}\nPlease return the COMPLETE corrected JSON.",
                    user_message, first_err
                );
                exchange.retry_prompt = Some(repair.clone());
                match rt.block_on(self.send_message(&repair)) {
                    Ok(retry) => {
                        exchange.retry_response = Some(retry.clone());
                        exchange.duration_ms = start.elapsed().as_millis() as u64;
                        match parse_llm_response(&retry) {
                            Ok(ext) => Ok((ext, exchange)),
                            Err(e) => {
                                exchange.duration_ms = start.elapsed().as_millis() as u64;
                                Err(TensaError::ExtractionError(format!(
                                    "Extraction failed after retry: {}",
                                    e
                                )))
                            }
                        }
                    }
                    Err(_) => {
                        exchange.duration_ms = start.elapsed().as_millis() as u64;
                        Err(first_err)
                    }
                }
            }
        }
    }
}

impl NarrativeExtractor for ClaudeClient {
    fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction> {
        self.extract_with_prompt(build_extraction_prompt(chunk))
    }

    fn extract_with_context(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<NarrativeExtraction> {
        let prompt = if known_entities.is_empty() {
            build_extraction_prompt(chunk)
        } else {
            build_extraction_prompt_with_context(chunk, Some(known_entities))
        };
        self.extract_with_prompt(prompt)
    }

    fn extract_with_logging(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
        let prompt = if known_entities.is_empty() {
            build_extraction_prompt(chunk)
        } else {
            build_extraction_prompt_with_context(chunk, Some(known_entities))
        };
        let (ext, exchange) = self.extract_with_prompt_logged(prompt)?;
        Ok((ext, Some(exchange)))
    }

    fn set_cancel_flag(&self, flag: Arc<AtomicBool>) {
        *self.cancel_flag.lock().unwrap() = flag;
    }

    fn model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn enrich_extraction(
        &self,
        chunk: &TextChunk,
        extraction: &NarrativeExtraction,
    ) -> Result<ExtractionEnrichment> {
        let prompt = build_enrichment_prompt(chunk, extraction);
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
        let raw = tokio::task::block_in_place(|| {
            rt.block_on(self.send_message_with_system(ENRICHMENT_SYSTEM_PROMPT, &prompt))
        })?;
        let mut enrichment = parse_enrichment_response(&raw)?;
        repair_enrichment(&mut enrichment);
        Ok(enrichment)
    }

    fn reconcile_temporal(
        &self,
        chunk_summaries: &[(usize, Vec<(String, Option<String>, Option<String>)>)],
    ) -> Result<TemporalReconciliation> {
        let prompt = build_reconciliation_prompt(chunk_summaries);
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
        let raw = tokio::task::block_in_place(|| {
            rt.block_on(self.send_message_with_system(TEMPORAL_RECONCILIATION_PROMPT, &prompt))
        })?;
        parse_reconciliation_response(&raw)
    }

    fn answer_question(&self, system_prompt: &str, question: &str) -> Result<String> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
        tokio::task::block_in_place(|| {
            rt.block_on(self.send_message_with_system(system_prompt, question))
        })
        .map_err(|e| TensaError::LlmError(format!("RAG answer failed: {}", e)))
    }
}

// ─── OpenRouter Client (OpenAI-compatible) ──────────────────

/// OpenRouter API request (OpenAI chat completions format).
#[derive(Serialize)]
struct OpenRouterRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ApiMessage>,
}

/// OpenRouter API response.
#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessage,
}

#[derive(Deserialize)]
struct OpenRouterMessage {
    content: Option<String>,
}

/// Parse an OpenRouter response, reading as text first for better error diagnostics.
/// Handles both success responses (with `choices`) and error responses (with `error`).
async fn parse_openrouter_response(resp: reqwest::Response) -> Result<String> {
    let raw = resp
        .text()
        .await
        .map_err(|e| TensaError::LlmError(format!("Failed to read response body: {}", e)))?;

    // Check for OpenRouter error envelope first (e.g. rate limits, overload, moderation)
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
        if let Some(err_obj) = val.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            let code = err_obj.get("code").and_then(|c| c.as_u64()).unwrap_or(0);
            tracing::warn!(
                code = code,
                message = %msg,
                "OpenRouter returned error in response body"
            );
            if code == 429 {
                return Err(TensaError::LlmRateLimit {
                    retry_after_secs: 5,
                });
            }
            return Err(TensaError::LlmError(format!(
                "OpenRouter error ({}): {}",
                code, msg
            )));
        }
    }

    let body: OpenRouterResponse = serde_json::from_str(&raw).map_err(|e| {
        let preview: String = raw.chars().take(500).collect();
        tracing::warn!(
            error = %e,
            body_len = raw.len(),
            body_preview = %preview,
            "OpenRouter response parse failed"
        );
        TensaError::LlmError(format!("Failed to parse response: {}", e))
    })?;
    Ok(body
        .choices
        .into_iter()
        .filter_map(|c| c.message.content)
        .collect::<Vec<_>>()
        .join(""))
}

/// LLM client that routes through OpenRouter (OpenAI-compatible API).
pub struct OpenRouterClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
    max_retries: u32,
    rate_limiter: RateLimiter,
    system_prompt: String,
    cancel_flag: Mutex<Arc<AtomicBool>>,
}

impl OpenRouterClient {
    /// Create a new OpenRouter client.
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            api_key,
            model,
            max_retries: 3,
            rate_limiter: RateLimiter::new(30),
            system_prompt: EXTRACTION_SYSTEM_PROMPT.to_string(),
            cancel_flag: Mutex::new(Arc::new(AtomicBool::new(false))),
        }
    }

    /// Override the system prompt for domain-specific extraction.
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Send a message via OpenRouter with retry logic.
    async fn send_message(&self, user_message: &str) -> Result<String> {
        let mut last_error = None;
        let flag = self.cancel_flag.lock().unwrap().clone();

        for attempt in 0..=self.max_retries {
            if flag.load(Ordering::Relaxed) {
                return Err(TensaError::LlmError("Cancelled by user".into()));
            }

            if let Err(wait) = self.rate_limiter.try_acquire() {
                tokio::time::sleep(wait).await;
                let _ = self.rate_limiter.try_acquire();
            }

            let request = OpenRouterRequest {
                model: self.model.clone(),
                max_tokens: 16384,
                messages: vec![
                    ApiMessage {
                        role: "system".to_string(),
                        content: self.system_prompt.clone(),
                    },
                    ApiMessage {
                        role: "user".to_string(),
                        content: user_message.to_string(),
                    },
                ],
            };

            let result = match with_cancel(
                self.client
                    .post("https://openrouter.ai/api/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json")
                    .header("HTTP-Referer", "https://tensa.dev")
                    .header("X-Title", "TENSA Studio")
                    .json(&request)
                    .send(),
                &flag,
            )
            .await
            {
                Err(cancel_err) => return Err(cancel_err),
                Ok(http_result) => http_result,
            };

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let text = parse_openrouter_response(resp).await?;
                        return Ok(text);
                    } else if status.as_u16() == 429 {
                        let retry_secs = 2u64.pow(attempt);
                        last_error = Some(TensaError::LlmRateLimit {
                            retry_after_secs: retry_secs,
                        });
                        tokio::time::sleep(Duration::from_secs(retry_secs)).await;
                        continue;
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        last_error =
                            Some(TensaError::LlmError(format!("HTTP {}: {}", status, body)));
                        if status.is_server_error() {
                            let backoff = Duration::from_secs(2u64.pow(attempt));
                            tokio::time::sleep(backoff).await;
                            continue;
                        }
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(TensaError::LlmError(format!("Request failed: {}", e)));
                    let backoff = Duration::from_secs(2u64.pow(attempt));
                    tokio::time::sleep(backoff).await;
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| TensaError::LlmError("Unknown error".into())))
    }

    /// Send a multi-turn conversation via OpenRouter with retry logic.
    async fn send_messages(&self, messages: Vec<ApiMessage>) -> Result<String> {
        let mut all_messages = vec![ApiMessage {
            role: "system".to_string(),
            content: self.system_prompt.clone(),
        }];
        all_messages.extend(messages);
        let request = OpenRouterRequest {
            model: self.model.clone(),
            max_tokens: 16384,
            messages: all_messages,
        };
        // Single attempt (retry is handled at the extraction level)
        let flag = self.cancel_flag.lock().unwrap().clone();
        let result = match with_cancel(
            self.client
                .post("https://openrouter.ai/api/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .header("HTTP-Referer", "https://tensa.dev")
                .header("X-Title", "TENSA Studio")
                .json(&request)
                .send(),
            &flag,
        )
        .await
        {
            Err(cancel_err) => return Err(cancel_err),
            Ok(http_result) => http_result,
        };
        match result {
            Ok(resp) if resp.status().is_success() => parse_openrouter_response(resp).await,
            Ok(resp) => {
                let body = resp.text().await.unwrap_or_default();
                Err(TensaError::LlmError(format!("HTTP error: {}", body)))
            }
            Err(e) => Err(TensaError::LlmError(format!("Request failed: {}", e))),
        }
    }

    /// Send pre-assembled messages without prepending self.system_prompt.
    /// Used by answer_question to provide a custom system prompt.
    async fn send_raw_messages(&self, messages: Vec<ApiMessage>) -> Result<String> {
        let msg_count = messages.len();
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        tracing::info!(
            model = %self.model,
            endpoint = "https://openrouter.ai/api/v1/chat/completions",
            messages = msg_count,
            total_chars = total_chars,
            "OpenRouter: sending raw session request"
        );
        let call_start = std::time::Instant::now();

        let request = OpenRouterRequest {
            model: self.model.clone(),
            max_tokens: 16384,
            messages,
        };
        let flag = self.cancel_flag.lock().unwrap().clone();
        let result = match with_cancel(
            self.client
                .post("https://openrouter.ai/api/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .header("HTTP-Referer", "https://tensa.dev")
                .header("X-Title", "TENSA Studio")
                .json(&request)
                .send(),
            &flag,
        )
        .await
        {
            Err(cancel_err) => {
                tracing::warn!(
                    model = %self.model,
                    elapsed_ms = call_start.elapsed().as_millis(),
                    "OpenRouter: request cancelled"
                );
                return Err(cancel_err);
            }
            Ok(http_result) => http_result,
        };
        match result {
            Ok(resp) if resp.status().is_success() => {
                let elapsed = call_start.elapsed().as_millis();
                let content = parse_openrouter_response(resp).await?;
                tracing::info!(
                    model = %self.model,
                    elapsed_ms = elapsed,
                    response_len = content.len(),
                    "OpenRouter: response received"
                );
                Ok(content)
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::error!(
                    model = %self.model,
                    status = %status,
                    elapsed_ms = call_start.elapsed().as_millis(),
                    body = %body,
                    "OpenRouter: HTTP error"
                );
                Err(TensaError::LlmError(format!("HTTP error: {}", body)))
            }
            Err(e) => {
                tracing::error!(
                    model = %self.model,
                    elapsed_ms = call_start.elapsed().as_millis(),
                    error = %e,
                    "OpenRouter: request failed"
                );
                Err(TensaError::LlmError(format!("Request failed: {}", e)))
            }
        }
    }

    /// Extract narrative from a pre-built prompt with retry-on-parse-failure.
    fn extract_with_prompt(&self, user_message: String) -> Result<NarrativeExtraction> {
        let (ext, _) = self.extract_with_prompt_logged(user_message)?;
        Ok(ext)
    }

    fn extract_with_prompt_logged(
        &self,
        user_message: String,
    ) -> Result<(NarrativeExtraction, RawLlmExchange)> {
        let start = Instant::now();
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;

        let response = rt
            .block_on(self.send_message(&user_message))
            .map_err(|e| TensaError::LlmError(format!("API call failed: {}", e)))?;

        let mut exchange = RawLlmExchange {
            system_prompt: self.system_prompt.clone(),
            user_prompt: user_message.clone(),
            raw_response: response.clone(),
            retry_prompt: None,
            retry_response: None,
            parse_error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            model: Some(self.model.clone()),
            endpoint: Some("https://openrouter.ai/api/v1/chat/completions".into()),
        };

        match parse_llm_response(&response) {
            Ok(ext) => {
                exchange.duration_ms = start.elapsed().as_millis() as u64;
                Ok((ext, exchange))
            }
            Err(first_err) => {
                exchange.parse_error = Some(format!("{}", first_err));
                let repair_prompt = format!(
                    "Your JSON response had a parse error:\n{}\n\n\
                     Please return the COMPLETE corrected JSON object. \
                     All fields (entities, situations, participations, causal_links, temporal_relations) \
                     must be arrays (use [] if empty). \
                     Return ONLY valid JSON.",
                    first_err
                );
                exchange.retry_prompt = Some(repair_prompt.clone());
                let messages = vec![
                    ApiMessage {
                        role: "user".to_string(),
                        content: user_message,
                    },
                    ApiMessage {
                        role: "assistant".to_string(),
                        content: response,
                    },
                    ApiMessage {
                        role: "user".to_string(),
                        content: repair_prompt,
                    },
                ];
                match rt.block_on(self.send_messages(messages)) {
                    Ok(retry_response) => {
                        exchange.retry_response = Some(retry_response.clone());
                        exchange.duration_ms = start.elapsed().as_millis() as u64;
                        match parse_llm_response(&retry_response) {
                            Ok(ext) => Ok((ext, exchange)),
                            Err(retry_err) => Err(TensaError::ExtractionError(format!(
                                "Extraction failed after retry: {}",
                                retry_err
                            ))),
                        }
                    }
                    Err(_) => {
                        exchange.duration_ms = start.elapsed().as_millis() as u64;
                        Err(first_err)
                    }
                }
            }
        }
    }
}

impl NarrativeExtractor for OpenRouterClient {
    fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction> {
        self.extract_with_prompt(build_extraction_prompt(chunk))
    }

    fn extract_with_context(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<NarrativeExtraction> {
        let prompt = if known_entities.is_empty() {
            build_extraction_prompt(chunk)
        } else {
            build_extraction_prompt_with_context(chunk, Some(known_entities))
        };
        self.extract_with_prompt(prompt)
    }

    fn extract_with_logging(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
        let prompt = if known_entities.is_empty() {
            build_extraction_prompt(chunk)
        } else {
            build_extraction_prompt_with_context(chunk, Some(known_entities))
        };
        let (ext, exchange) = self.extract_with_prompt_logged(prompt)?;
        Ok((ext, Some(exchange)))
    }

    fn set_cancel_flag(&self, flag: Arc<AtomicBool>) {
        *self.cancel_flag.lock().unwrap() = flag;
    }

    fn model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn enrich_extraction(
        &self,
        chunk: &TextChunk,
        extraction: &NarrativeExtraction,
    ) -> Result<ExtractionEnrichment> {
        self.enrich_extraction_impl(chunk, extraction)
    }

    fn reconcile_temporal(
        &self,
        chunk_summaries: &[(usize, Vec<(String, Option<String>, Option<String>)>)],
    ) -> Result<TemporalReconciliation> {
        self.reconcile_temporal_impl(chunk_summaries)
    }

    fn answer_question(&self, system_prompt: &str, question: &str) -> Result<String> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
        let messages = vec![
            ApiMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ApiMessage {
                role: "user".to_string(),
                content: question.to_string(),
            },
        ];
        tokio::task::block_in_place(|| rt.block_on(self.send_raw_messages(messages)))
            .map_err(|e| TensaError::LlmError(format!("RAG answer failed: {}", e)))
    }

    fn as_session(&self) -> Option<&dyn SessionCapableExtractor> {
        Some(self)
    }
}

// ─── Local LLM Client (OpenAI-compatible — vLLM, Ollama, LiteLLM, etc.) ──

/// LLM client for local/self-hosted endpoints that serve the OpenAI chat completions API.
///
/// Works with vLLM, Ollama (`/v1/chat/completions`), LiteLLM (`/chat/completions`),
/// text-generation-inference, or any server that implements the OpenAI-compatible format.
/// Auto-detects the correct endpoint path on first request.
pub struct LocalLLMClient {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    max_retries: u32,
    rate_limiter: RateLimiter,
    system_prompt: String,
    cancel_flag: Mutex<Arc<AtomicBool>>,
    /// Resolved endpoint path, detected on first request.
    /// `None` = not yet probed, `Some(path)` = confirmed working path.
    endpoint_path: Mutex<Option<String>>,
}

impl LocalLLMClient {
    /// Create a new local LLM client.
    ///
    /// `base_url` should be the root URL of the server (e.g. `http://192.168.1.50:8000`).
    /// The client will POST to `{base_url}/v1/chat/completions`.
    pub fn new(base_url: String, model: String) -> Self {
        // Strip trailing slash for consistency
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300)) // local models can be slow
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            base_url,
            model,
            api_key: None,
            max_retries: 3,
            rate_limiter: RateLimiter::new(120), // local GPU — higher default RPM
            system_prompt: EXTRACTION_SYSTEM_PROMPT.to_string(),
            cancel_flag: Mutex::new(Arc::new(AtomicBool::new(false))),
            endpoint_path: Mutex::new(None),
        }
    }

    /// Set an optional API key (some local servers require one).
    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Set custom rate limit.
    pub fn with_rate_limit(mut self, rpm: u32) -> Self {
        self.rate_limiter = RateLimiter::new(rpm);
        self
    }

    /// Set custom system prompt for extraction.
    pub fn with_system_prompt(mut self, prompt: String) -> Self {
        self.system_prompt = prompt;
        self
    }

    /// Send a multi-turn conversation to the local LLM with retry logic.
    async fn send_messages(&self, messages: Vec<ApiMessage>) -> Result<String> {
        let mut last_error = None;
        let flag = self.cancel_flag.lock().unwrap().clone();

        for attempt in 0..=self.max_retries {
            if flag.load(Ordering::Relaxed) {
                return Err(TensaError::LlmError("Cancelled by user".into()));
            }

            if let Err(wait) = self.rate_limiter.try_acquire() {
                tokio::time::sleep(wait).await;
                let _ = self.rate_limiter.try_acquire();
            }

            let mut all_messages = vec![ApiMessage {
                role: "system".to_string(),
                content: self.system_prompt.clone(),
            }];
            all_messages.extend(messages.clone());

            let request = OpenRouterRequest {
                model: self.model.clone(),
                max_tokens: 16384,
                messages: all_messages,
            };

            let cached_path = self.endpoint_path.lock().unwrap().clone();
            let paths_to_try: Vec<String> = match cached_path {
                Some(ref p) => vec![p.clone()],
                None => vec![
                    format!("{}/v1/chat/completions", self.base_url),
                    format!("{}/chat/completions", self.base_url),
                ],
            };

            let mut result = None;
            for endpoint in &paths_to_try {
                let mut req = self
                    .client
                    .post(endpoint)
                    .header("Content-Type", "application/json");

                if let Some(ref key) = self.api_key {
                    req = req.header("Authorization", format!("Bearer {}", key));
                }

                let r = match with_cancel(req.json(&request).send(), &flag).await {
                    Err(cancel_err) => return Err(cancel_err),
                    Ok(http_result) => http_result,
                };

                if let Ok(ref resp) = r {
                    if resp.status().as_u16() == 405 && paths_to_try.len() > 1 {
                        continue;
                    }
                }

                if cached_path.is_none() {
                    *self.endpoint_path.lock().unwrap() = Some(endpoint.clone());
                }
                result = Some(r);
                break;
            }
            let result = match result {
                Some(r) => r,
                None => {
                    last_error = Some(TensaError::LlmError(format!(
                        "All endpoint paths returned 405 for {}",
                        self.base_url
                    )));
                    break;
                }
            };

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let text = parse_openrouter_response(resp).await?;
                        return Ok(text);
                    } else if status.as_u16() == 429 {
                        let retry_secs = 2u64.pow(attempt);
                        last_error = Some(TensaError::LlmRateLimit {
                            retry_after_secs: retry_secs,
                        });
                        tokio::time::sleep(Duration::from_secs(retry_secs)).await;
                        continue;
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        last_error =
                            Some(TensaError::LlmError(format!("HTTP {}: {}", status, body)));
                        if status.is_server_error() {
                            let backoff = Duration::from_secs(2u64.pow(attempt));
                            tokio::time::sleep(backoff).await;
                            continue;
                        }
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(TensaError::LlmError(format!("Request failed: {}", e)));
                    let backoff = Duration::from_secs(2u64.pow(attempt));
                    tokio::time::sleep(backoff).await;
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| TensaError::LlmError("Unknown error".into())))
    }

    /// Send pre-assembled messages without prepending self.system_prompt.
    /// Used by answer_question to provide a custom system prompt.
    async fn send_raw_messages(&self, messages: Vec<ApiMessage>) -> Result<String> {
        let msg_count = messages.len();
        let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
        tracing::info!(
            model = %self.model,
            base_url = %self.base_url,
            messages = msg_count,
            total_chars = total_chars,
            "LocalLLM: sending raw session request"
        );
        let call_start = std::time::Instant::now();

        let flag = self.cancel_flag.lock().unwrap().clone();
        if let Err(wait) = self.rate_limiter.try_acquire() {
            tokio::time::sleep(wait).await;
            let _ = self.rate_limiter.try_acquire();
        }

        let request = OpenRouterRequest {
            model: self.model.clone(),
            max_tokens: 16384,
            messages,
        };

        let cached_path = self.endpoint_path.lock().unwrap().clone();
        let paths_to_try: Vec<String> = match cached_path {
            Some(ref p) => vec![p.clone()],
            None => vec![
                format!("{}/v1/chat/completions", self.base_url),
                format!("{}/chat/completions", self.base_url),
            ],
        };

        for endpoint in &paths_to_try {
            let mut req = self
                .client
                .post(endpoint)
                .header("Content-Type", "application/json");
            if let Some(ref key) = self.api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let result = match with_cancel(req.json(&request).send(), &flag).await {
                Err(cancel_err) => {
                    tracing::warn!(
                        model = %self.model,
                        elapsed_ms = call_start.elapsed().as_millis(),
                        "LocalLLM: request cancelled"
                    );
                    return Err(cancel_err);
                }
                Ok(http_result) => http_result,
            };
            if let Ok(ref resp) = result {
                if resp.status().as_u16() == 405 && paths_to_try.len() > 1 {
                    continue;
                }
            }
            if cached_path.is_none() {
                *self.endpoint_path.lock().unwrap() = Some(endpoint.clone());
            }
            return match result {
                Ok(resp) if resp.status().is_success() => {
                    let elapsed = call_start.elapsed().as_millis();
                    let content = parse_openrouter_response(resp).await?;
                    tracing::info!(
                        model = %self.model,
                        endpoint = %endpoint,
                        elapsed_ms = elapsed,
                        response_len = content.len(),
                        "LocalLLM: response received"
                    );
                    Ok(content)
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    tracing::error!(
                        model = %self.model,
                        endpoint = %endpoint,
                        status = %status,
                        elapsed_ms = call_start.elapsed().as_millis(),
                        body = %body,
                        "LocalLLM: HTTP error"
                    );
                    Err(TensaError::LlmError(format!("HTTP error: {}", body)))
                }
                Err(e) => {
                    tracing::error!(
                        model = %self.model,
                        endpoint = %endpoint,
                        elapsed_ms = call_start.elapsed().as_millis(),
                        error = %e,
                        "LocalLLM: request failed"
                    );
                    Err(TensaError::LlmError(format!("Request failed: {}", e)))
                }
            };
        }
        Err(TensaError::LlmError(format!(
            "All endpoint paths returned 405 for {}",
            self.base_url
        )))
    }

    /// Send a single message to the local LLM with retry logic.
    async fn send_message(&self, user_message: &str) -> Result<String> {
        let mut last_error = None;
        let flag = self.cancel_flag.lock().unwrap().clone();

        for attempt in 0..=self.max_retries {
            if flag.load(Ordering::Relaxed) {
                return Err(TensaError::LlmError("Cancelled by user".into()));
            }

            if let Err(wait) = self.rate_limiter.try_acquire() {
                tokio::time::sleep(wait).await;
                let _ = self.rate_limiter.try_acquire();
            }

            let request = OpenRouterRequest {
                model: self.model.clone(),
                max_tokens: 16384,
                messages: vec![
                    ApiMessage {
                        role: "system".to_string(),
                        content: self.system_prompt.clone(),
                    },
                    ApiMessage {
                        role: "user".to_string(),
                        content: user_message.to_string(),
                    },
                ],
            };

            // Auto-detect endpoint path on first request: try /v1/chat/completions,
            // fall back to /chat/completions (LiteLLM uses this).
            let cached_path = self.endpoint_path.lock().unwrap().clone();
            let paths_to_try: Vec<String> = match cached_path {
                Some(ref p) => vec![p.clone()],
                None => vec![
                    format!("{}/v1/chat/completions", self.base_url),
                    format!("{}/chat/completions", self.base_url),
                ],
            };

            let mut result = None;
            for endpoint in &paths_to_try {
                let mut req = self
                    .client
                    .post(endpoint)
                    .header("Content-Type", "application/json");

                if let Some(ref key) = self.api_key {
                    req = req.header("Authorization", format!("Bearer {}", key));
                }

                let r = match with_cancel(req.json(&request).send(), &flag).await {
                    Err(cancel_err) => return Err(cancel_err),
                    Ok(http_result) => http_result,
                };

                // If 405 Method Not Allowed, try next path
                if let Ok(ref resp) = r {
                    if resp.status().as_u16() == 405 && paths_to_try.len() > 1 {
                        continue;
                    }
                }

                // Cache the working path for subsequent requests
                if cached_path.is_none() {
                    *self.endpoint_path.lock().unwrap() = Some(endpoint.clone());
                }
                result = Some(r);
                break;
            }
            let result = match result {
                Some(r) => r,
                None => {
                    last_error = Some(TensaError::LlmError(format!(
                        "All endpoint paths returned 405 Method Not Allowed for {}",
                        self.base_url
                    )));
                    break;
                }
            };

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let text = parse_openrouter_response(resp).await?;
                        return Ok(text);
                    } else if status.as_u16() == 429 {
                        let retry_secs = 2u64.pow(attempt);
                        last_error = Some(TensaError::LlmRateLimit {
                            retry_after_secs: retry_secs,
                        });
                        tokio::time::sleep(Duration::from_secs(retry_secs)).await;
                        continue;
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        last_error =
                            Some(TensaError::LlmError(format!("HTTP {}: {}", status, body)));
                        if status.is_server_error() {
                            let backoff = Duration::from_secs(2u64.pow(attempt));
                            tokio::time::sleep(backoff).await;
                            continue;
                        }
                        break;
                    }
                }
                Err(e) => {
                    last_error = Some(TensaError::LlmError(format!("Request failed: {}", e)));
                    let backoff = Duration::from_secs(2u64.pow(attempt));
                    tokio::time::sleep(backoff).await;
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| TensaError::LlmError("Unknown error".into())))
    }

    /// Extract narrative from a pre-built prompt with retry-on-parse-failure.
    fn extract_with_prompt(&self, user_message: String) -> Result<NarrativeExtraction> {
        let (ext, _) = self.extract_with_prompt_logged(user_message)?;
        Ok(ext)
    }

    fn extract_with_prompt_logged(
        &self,
        user_message: String,
    ) -> Result<(NarrativeExtraction, RawLlmExchange)> {
        let start = Instant::now();
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;

        let response = rt
            .block_on(self.send_message(&user_message))
            .map_err(|e| TensaError::LlmError(format!("API call failed: {}", e)))?;

        let endpoint_url = {
            let path = self.endpoint_path.lock().unwrap();
            format!(
                "{}{}",
                self.base_url,
                path.as_deref().unwrap_or("/v1/chat/completions")
            )
        };

        let mut exchange = RawLlmExchange {
            system_prompt: self.system_prompt.clone(),
            user_prompt: user_message.clone(),
            raw_response: response.clone(),
            retry_prompt: None,
            retry_response: None,
            parse_error: None,
            duration_ms: start.elapsed().as_millis() as u64,
            model: Some(self.model.clone()),
            endpoint: Some(endpoint_url),
        };

        match parse_llm_response(&response) {
            Ok(ext) => {
                exchange.duration_ms = start.elapsed().as_millis() as u64;
                Ok((ext, exchange))
            }
            Err(first_err) => {
                exchange.parse_error = Some(format!("{}", first_err));
                let repair_prompt = format!(
                    "Your JSON response had a parse error:\n{}\n\n\
                     Please return the COMPLETE corrected JSON object. \
                     All fields (entities, situations, participations, causal_links, temporal_relations) \
                     must be arrays (use [] if empty). \
                     The aliases field must be an array of strings. \
                     Return ONLY valid JSON.",
                    first_err
                );
                exchange.retry_prompt = Some(repair_prompt.clone());
                let messages = vec![
                    ApiMessage {
                        role: "user".to_string(),
                        content: user_message,
                    },
                    ApiMessage {
                        role: "assistant".to_string(),
                        content: response,
                    },
                    ApiMessage {
                        role: "user".to_string(),
                        content: repair_prompt,
                    },
                ];
                match rt.block_on(self.send_messages(messages)) {
                    Ok(retry_response) => {
                        exchange.retry_response = Some(retry_response.clone());
                        exchange.duration_ms = start.elapsed().as_millis() as u64;
                        match parse_llm_response(&retry_response) {
                            Ok(ext) => Ok((ext, exchange)),
                            Err(retry_err) => Err(TensaError::ExtractionError(format!(
                                "Extraction failed after retry: {}",
                                retry_err
                            ))),
                        }
                    }
                    Err(_) => {
                        exchange.duration_ms = start.elapsed().as_millis() as u64;
                        Err(first_err)
                    }
                }
            }
        }
    }
}

impl NarrativeExtractor for LocalLLMClient {
    fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction> {
        self.extract_with_prompt(build_extraction_prompt(chunk))
    }

    fn extract_with_context(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<NarrativeExtraction> {
        let prompt = if known_entities.is_empty() {
            build_extraction_prompt(chunk)
        } else {
            build_extraction_prompt_with_context(chunk, Some(known_entities))
        };
        self.extract_with_prompt(prompt)
    }

    fn extract_with_logging(
        &self,
        chunk: &TextChunk,
        known_entities: &[String],
    ) -> Result<(NarrativeExtraction, Option<RawLlmExchange>)> {
        let prompt = if known_entities.is_empty() {
            build_extraction_prompt(chunk)
        } else {
            build_extraction_prompt_with_context(chunk, Some(known_entities))
        };
        let (ext, exchange) = self.extract_with_prompt_logged(prompt)?;
        Ok((ext, Some(exchange)))
    }

    fn set_cancel_flag(&self, flag: Arc<AtomicBool>) {
        *self.cancel_flag.lock().unwrap() = flag;
    }

    fn model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn enrich_extraction(
        &self,
        chunk: &TextChunk,
        extraction: &NarrativeExtraction,
    ) -> Result<ExtractionEnrichment> {
        self.enrich_extraction_impl(chunk, extraction)
    }

    fn reconcile_temporal(
        &self,
        chunk_summaries: &[(usize, Vec<(String, Option<String>, Option<String>)>)],
    ) -> Result<TemporalReconciliation> {
        self.reconcile_temporal_impl(chunk_summaries)
    }

    fn answer_question(&self, system_prompt: &str, question: &str) -> Result<String> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
        let messages = vec![
            ApiMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ApiMessage {
                role: "user".to_string(),
                content: question.to_string(),
            },
        ];
        tokio::task::block_in_place(|| rt.block_on(self.send_raw_messages(messages)))
            .map_err(|e| TensaError::LlmError(format!("RAG answer failed: {}", e)))
    }

    fn as_session(&self) -> Option<&dyn SessionCapableExtractor> {
        Some(self)
    }
}

// ─── ExplainProvider Implementations ────────────────────────

use crate::inference::explain::{build_explain_prompt, ExplainProvider};

macro_rules! impl_explain_provider {
    ($($t:ty),+) => {
        $(impl ExplainProvider for $t {
            fn explain(&self, result: &crate::types::InferenceResult) -> Result<String> {
                let prompt = build_explain_prompt(result);
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
                rt.block_on(self.send_message(&prompt))
                    .map_err(|e| TensaError::LlmError(format!("Explain API call failed: {}", e)))
            }
        })+
    };
}

impl_explain_provider!(ClaudeClient, OpenRouterClient, LocalLLMClient);

/// Implement `enrich_extraction` and `reconcile_temporal` for OpenAI-compatible clients
/// that have `send_raw_messages`. ClaudeClient uses `send_message_with_system` instead.
macro_rules! impl_enrichment_methods {
    ($($t:ty),+) => {
        $(
        impl $t {
            fn enrich_extraction_impl(
                &self,
                chunk: &TextChunk,
                extraction: &NarrativeExtraction,
            ) -> Result<ExtractionEnrichment> {
                let prompt = build_enrichment_prompt(chunk, extraction);
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
                let messages = vec![
                    ApiMessage { role: "system".to_string(), content: ENRICHMENT_SYSTEM_PROMPT.to_string() },
                    ApiMessage { role: "user".to_string(), content: prompt },
                ];
                let raw = tokio::task::block_in_place(|| rt.block_on(self.send_raw_messages(messages)))?;
                let mut enrichment = parse_enrichment_response(&raw)?;
                repair_enrichment(&mut enrichment);
                Ok(enrichment)
            }

            fn reconcile_temporal_impl(
                &self,
                chunk_summaries: &[(usize, Vec<(String, Option<String>, Option<String>)>)],
            ) -> Result<TemporalReconciliation> {
                let prompt = build_reconciliation_prompt(chunk_summaries);
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
                let messages = vec![
                    ApiMessage { role: "system".to_string(), content: TEMPORAL_RECONCILIATION_PROMPT.to_string() },
                    ApiMessage { role: "user".to_string(), content: prompt },
                ];
                let raw = tokio::task::block_in_place(|| rt.block_on(self.send_raw_messages(messages)))?;
                parse_reconciliation_response(&raw)
            }
        }
        )+
    };
}

impl_enrichment_methods!(OpenRouterClient, LocalLLMClient);

/// Implement SessionCapableExtractor for clients with send_messages().
macro_rules! impl_session_capable {
    ($($t:ty),+) => {
        $(
        impl SessionCapableExtractor for $t {
            fn send_session_messages(&self, messages: &[ApiMessage]) -> Result<String> {
                let model = self.model_name().unwrap_or_else(|| "unknown".into());
                tracing::debug!(
                    model = %model,
                    messages = messages.len(),
                    "SessionCapable: send_session_messages called"
                );
                let rt = tokio::runtime::Handle::try_current()
                    .map_err(|e| {
                        tracing::error!("SessionCapable: No tokio runtime — {}", e);
                        TensaError::LlmError("No tokio runtime available".into())
                    })?;
                // Use send_raw_messages to avoid prepending self.system_prompt —
                // session mode provides its own system prompt in the messages array.
                let result = tokio::task::block_in_place(|| rt.block_on(self.send_raw_messages(messages.to_vec())));
                if let Err(ref e) = result {
                    tracing::error!(model = %model, error = %e, "SessionCapable: session call failed");
                }
                result
            }
        }
        )+
    };
}

impl_session_capable!(OpenRouterClient, LocalLLMClient);

/// A no-op extractor used when only the processing pipeline (resolve/gate/commit)
/// is needed without any LLM calls (e.g. reprocessing stored extractions).
pub struct MockExtractor;

impl MockExtractor {
    /// Create an empty mock extractor.
    pub fn empty() -> Self {
        Self
    }
}

impl NarrativeExtractor for MockExtractor {
    fn extract_narrative(
        &self,
        _chunk: &TextChunk,
    ) -> crate::error::Result<crate::ingestion::extraction::NarrativeExtraction> {
        Ok(crate::ingestion::extraction::NarrativeExtraction {
            entities: vec![],
            situations: vec![],
            participations: vec![],
            causal_links: vec![],
            temporal_relations: vec![],
        })
    }
}

/// Build the user prompt for narrative extraction.
pub fn build_extraction_prompt(chunk: &TextChunk) -> String {
    build_extraction_prompt_with_context(chunk, None)
}

/// Build the user prompt with optional prior entity context.
/// When `known_entities` is provided, the LLM is told which characters/places
/// it has already seen, enabling consistent cross-chunk entity resolution.
pub fn build_extraction_prompt_with_context(
    chunk: &TextChunk,
    known_entities: Option<&[String]>,
) -> String {
    let mut prompt = String::new();

    // Prior entity context (for sequential/chapter mode)
    if let Some(entities) = known_entities {
        if !entities.is_empty() {
            prompt.push_str("[KNOWN ENTITIES FROM PREVIOUS CHAPTERS]\n");
            for name in entities {
                prompt.push_str(&format!("- {}\n", name));
            }
            prompt.push_str("Use these exact names when the same entity appears. Add new entities as needed.\n[END KNOWN ENTITIES]\n\n");
        }
    }

    if !chunk.overlap_prefix.is_empty() {
        prompt.push_str(&format!(
            "[CONTEXT FROM PREVIOUS CHUNK]\n{}\n[END CONTEXT]\n\n",
            chunk.overlap_prefix
        ));
    }
    if let Some(ch) = &chunk.chapter {
        prompt.push_str(&format!("[CHAPTER: {}]\n\n", ch));
    }
    prompt.push_str(&format!(
        "Extract all entities, situations, participations, causal links, and temporal relations from this text:\n\n{}",
        chunk.text
    ));
    prompt
}

/// System prompt for narrative extraction.
pub const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are a narrative extraction engine. Extract structured data from text as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- "aliases" MUST be an array of strings: ["name1", "name2"] or []
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of what happens", "temporal_marker": null, "location": null, "narrative_level": "Scene", "content_blocks": [{"content_type": "Text", "content": "Actual quote or close paraphrase from the source text (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "why A causes B", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = explicit, 0.7 = implied, 0.5 = inferred
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc. If you have N situations, valid values are 1 to N.

EXTRACTION GUIDELINES:
- Extract every named character, location, organization, and significant object
- Each distinct event or scene transition is a separate situation
- name: give each situation a concise name (5-8 words) identifying the event, e.g. "Bran discovers the Lannister secret", "Ned accepts Hand of the King"
- description: write a detailed 1-2 sentence summary, not a label. BAD: "fight". GOOD: "Oliver is beaten by Noah Claypole after Noah insults his mother". NEVER leave description empty.
- content_blocks: include actual quotes or close paraphrases from the source text. Use content_type "Dialogue" for speech, "Observation" for narrator description, "Text" for general prose. Each block should be 1-3 sentences. Include at least 1 block per situation.
- text_start / text_end: copy the first ~8-12 words and last ~8-12 words of this situation's prose EXACTLY from the chunk — verbatim, including punctuation and capitalization. These fingerprints are used to locate the situation's exact byte range in the source text. Do NOT paraphrase. If you are not 100% sure the excerpt is in the chunk word-for-word, OMIT the field (leave it out entirely) — the pipeline will fall back gracefully.
- List ALL participating entities for each situation — not just Actors:
  * Actor: use narrative roles (Protagonist, Antagonist, Witness, etc.)
  * Location: use role "Setting" for where the situation takes place
  * Artifact: use role "Instrument" for objects involved, or "Target" if the object is acted upon
  * Organization: use role appropriate to their involvement (e.g. Antagonist, Witness, Target)
  * Concept: use role "SubjectOfDiscussion" for themes, ideas, or abstract concepts referenced
- Add causal_links when one event leads to another. Format: {"from": situation_index, "to": situation_index, "causal_type": "...", "description": "why A causes B"} — indices are 1-based
- temporal_marker: any time reference ("Chapter 3", "three days later", "morning", etc.)
- If a field is unknown, use null for optional fields or [] for arrays

ANAPHORA & DESCRIPTIVE-REFERENCE RESOLUTION (CRITICAL):
- When text refers to a character by description rather than name ("a figure", "the stranger", "the voice on the phone", "the intruder", "a woman in red", "the killer", "someone", "a shape"), try to resolve the reference to a named entity from KNOWN ENTITIES or prior context, and add a Participation under that resolved name.
- Never leave a violent, pivotal, or action-driving event without a named agent as Protagonist/Antagonist. If you truly cannot identify the agent yet, use a placeholder entity like "Unknown Shooter" and set confidence to 0.4 so it can be merged later when the identity is revealed.
- Every killing, assault, theft, or confrontation scene needs THREE participations at minimum: the victim (Target), the actor (Antagonist or Protagonist depending on POV), and the means if a weapon or device is involved (Instrument). "The weapon" counts as a first-class participant, not flavor text.

NARRATOR & FIRST-PERSON RESOLUTION (CRITICAL):
- Identify the narrator early. If the text is first-person ("I did X", "I thought Y"), the "I" is a SPECIFIC named character — resolve it to that name and attribute the Participation to that name. NEVER create an unnamed "Narrator" entity.
- In multi-narrator books (epistolary novels, diary entries from multiple characters, multi-POV novels), the "I" can change between sections or chapters. Look at chapter headers, letter signatures, and diary date stamps to identify which character's voice you are currently in, and attribute first-person statements to that character only.
- Add the narrator as a Participation (role: Protagonist or Witness) in every situation they narrate — they were present for it by definition.
- When the narrator reports another character's dialogue or actions, that other character also gets a Participation for that situation.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array. If a location appears in a participation, it MUST first be in the entities array as entity_type "Location".
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

CHARACTER & ALIAS GUIDELINES:
- List ALL aliases for each character in the "aliases" array. Characters often have multiple names:
  * Formal vs informal: "Professor Abraham Van Helsing" → aliases: ["Van Helsing", "Professor"]
  * Married names: "Mina Murray" → aliases: ["Mina Harker", "Mrs. Harker"]
  * Titles: "Arthur Holmwood" → aliases: ["Lord Godalming"]
  * Epithets / monikers: "Count Dracula" → aliases: ["Dracula", "Vlad Dracula", "Count De Ville", "Nosferatu"]
- Include minor characters even if they appear only briefly

BIOGRAPHICAL FACTS (include in "properties" when stated or clearly implied):
- date_of_birth / date_of_death: ISO 8601 when possible ("1847-11-08"), otherwise a year or era as a string
- place_of_birth / place_of_death: free-form string, e.g. "Transylvania", "London"
- nationality, occupation, gender, title: short strings
- description: a single sentence capturing the character's role/identity (e.g. "Transylvanian vampire nobleman")
- Only include fields the source text actually supports. Do NOT invent. Leave the field out entirely if unsure.

TEMPORAL GUIDELINES:
- For diary/journal entries with explicit dates (e.g. "3 May", "25 September"), use the date as temporal_marker
- For letters and telegrams, extract the date from the header
- For relative time references ("three days later", "that evening"), note them as temporal_marker
- If a year is not stated but can be inferred from context, include it (e.g. "3 May" in a Victorian novel → "1893-05-03")
- IMPORTANT: Always produce temporal_relations linking situations in chronological order. At minimum, chain sequential situations: 1 Before 2, 2 Before 3, etc. Use Overlaps or During when events are concurrent.

CAUSAL LINK GUIDELINES:
- Connect situations that have cause-effect relationships
- At least consider: does situation A lead to or enable situation B?
- Common patterns: arrival → discovery, threat → escape, letter → action taken
- IMPORTANT: Always look for causal links. Most narratives have more cause-effect chains than are immediately obvious."#;

/// Return the system prompt for a given ingestion mode.
#[cfg(feature = "server")]
pub fn system_prompt_for_mode(mode: &crate::ingestion::config::IngestionMode) -> &'static str {
    use crate::ingestion::config::IngestionMode;
    match mode {
        IngestionMode::Novel => EXTRACTION_SYSTEM_PROMPT,
        IngestionMode::News => NEWS_EXTRACTION_PROMPT,
        IngestionMode::Intelligence => INTELLIGENCE_EXTRACTION_PROMPT,
        IngestionMode::Research => RESEARCH_EXTRACTION_PROMPT,
        IngestionMode::TemporalEvents => TEMPORAL_EVENTS_EXTRACTION_PROMPT,
        IngestionMode::Legal => LEGAL_EXTRACTION_PROMPT,
        IngestionMode::Financial => FINANCIAL_EXTRACTION_PROMPT,
        IngestionMode::Medical => MEDICAL_EXTRACTION_PROMPT,
        IngestionMode::Custom => EXTRACTION_SYSTEM_PROMPT, // fallback
    }
}

/// System prompt for news/journalism extraction.
pub const NEWS_EXTRACTION_PROMPT: &str = r#"You are a news extraction engine. Extract structured data from journalistic text as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of what happens", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Actual quote from source (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "why A causes B", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = explicitly stated, 0.7 = strongly implied, 0.5 = inferred from context
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

NEWS EXTRACTION GUIDELINES:
- name: give each situation a concise headline (5-8 words), e.g. "Mayor resigns amid corruption scandal"
- Focus on WHO did WHAT, WHEN, WHERE, and WHY
- Extract exact dates, times, and locations with high precision
- Every named person, organization, and place is an entity
- Each distinct event is a separate situation with precise temporal_marker
- Use direct quotes in content_blocks where available (content_type: "Dialogue")
- Source attribution: note which source/spokesperson said what in properties
- Confidence should reflect whether facts are confirmed, alleged, or speculated
- Extract organizations as entities (type: Organization) and link them to events
- For breaking news, emphasize temporal ordering and causal chains"#;

/// System prompt for intelligence/OSINT extraction.
pub const INTELLIGENCE_EXTRACTION_PROMPT: &str = r#"You are an intelligence extraction engine. Extract structured data from source material as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of what happens", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Key details from source (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "why A causes B", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = confirmed by multiple sources, 0.7 = single reliable source, 0.5 = unconfirmed, 0.3 = rumor/hearsay
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

INTELLIGENCE EXTRACTION GUIDELINES:
- name: give each situation a concise name (5-8 words), e.g. "Suspect meets handler at safehouse"
- Extract ALL named individuals with full names and known aliases
- Track entity relationships meticulously (who knows whom, who works with whom)
- Temporal precision is critical — exact dates, times, durations
- Location precision is critical — specific addresses, coordinates, areas
- Distinguish between confirmed facts and allegations/claims
- Properties should capture: nationality, affiliation, role, identifiers (phone, email, etc.)
- For organizations: structure, leadership, known members, operational areas
- For locations: type (residence, office, meeting point), geo coordinates if available
- Track communication patterns: who contacted whom, when, via what medium
- Financial transactions: amounts, currencies, accounts, intermediaries
- Use low confidence (0.3-0.5) for inferred connections not explicitly stated
- Causal links should capture operational sequences and decision chains"#;

/// System prompt for research/academic extraction.
pub const RESEARCH_EXTRACTION_PROMPT: &str = r#"You are a research extraction engine. Extract structured data from academic text as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of finding/method/claim", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Key text from source (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "why A causes B", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = statistically significant result, 0.7 = supported claim, 0.5 = hypothesis
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

RESEARCH EXTRACTION GUIDELINES:
- name: give each situation a concise name (5-8 words), e.g. "Model outperforms baseline on ImageNet"
- Authors are Actor entities; research groups/labs are Organization entities
- Concepts, theories, and methods are Concept entities
- Datasets, tools, and software are Artifact entities
- Each key finding, method step, or claim is a separate situation
- Use narrative_level "Arc" for overarching research questions, "Event" for specific findings
- Properties for Actors: affiliation, ORCID, h-index if mentioned
- Properties for Concepts: field, subfield, first_mentioned_year
- Track citation relationships: which findings support/contradict others
- Causal links should capture: methodology → finding, hypothesis → experiment → result
- content_blocks should quote key results, p-values, effect sizes where stated
- Use temporal_marker for publication dates, study periods, data collection dates"#;

/// System prompt for temporal events / timeline extraction.
pub const TEMPORAL_EVENTS_EXTRACTION_PROMPT: &str = r#"You are a temporal event extraction engine. Extract structured data from text describing events with dates and time intervals as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values
- TEMPORAL PRECISION IS THE PRIMARY GOAL — extract exact dates, years, and durations

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of the event", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Source text (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "why A causes B", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, MetBy, Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = dates explicitly stated, 0.7 = dates implied, 0.5 = dates inferred
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

TEMPORAL EVENTS GUIDELINES:
- Each distinct event with its own time span is a separate situation
- temporal_marker: extract the EXACT date or year range from the text. Examples:
  * "between year 1418 and year 1419" → temporal_marker: "1418-1419"
  * "from January 2020 to March 2021" → temporal_marker: "2020-01-01/2021-03-31"
  * "in 1776" → temporal_marker: "1776"
  * "3rd century BCE" → temporal_marker: "-300/-200"
- Named events, periods, eras, wars, reigns are entities (type: Concept or Organization)
- For each pair of events, determine the Allen temporal relation:
  * Before: A ends before B starts (gap between them)
  * Meets: A ends exactly when B starts (no gap, no overlap)
  * Overlaps: A starts before B, A ends during B
  * Starts: A and B start together, A ends first
  * During: A is entirely within B
  * Finishes: A starts after B, both end together
  * Equals: A and B have identical time spans
  * And their inverses: After, MetBy, OverlappedBy, StartedBy, Contains, FinishedBy
- ALWAYS produce temporal_relations between ALL pairs of situations. This is the most important output.
- When text says "between year X and year Y", treat X as start and Y as end
- Properties should capture: duration, periodization, era, calendar system"#;

/// System prompt for legal document extraction.
pub const LEGAL_EXTRACTION_PROMPT: &str = r#"You are a legal extraction engine. Extract structured data from legal documents as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of the legal event/clause", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Exact legal text (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "legal dependency", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = explicitly stated in document, 0.7 = legally implied, 0.5 = inferred from context
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

LEGAL EXTRACTION GUIDELINES:
- Parties: extract ALL named parties with full legal names and aliases (d/b/a, formerly known as)
- Properties for parties: role (plaintiff, defendant, guarantor, trustee), jurisdiction, registration number
- Each clause, obligation, right, or condition is a separate situation
- Use narrative_level "Arc" for contract sections, "Event" for specific clauses
- Dates are critical: effective dates, deadlines, filing dates, statute of limitations
- Causal links: condition → obligation, breach → remedy, filing → ruling
- content_blocks: quote exact legal language (critical for accuracy)
- Track monetary amounts, penalties, and consideration in properties
- Distinguish between obligations, permissions, and prohibitions in descriptions"#;

/// System prompt for financial document extraction.
pub const FINANCIAL_EXTRACTION_PROMPT: &str = r#"You are a financial extraction engine. Extract structured data from financial documents as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of the transaction/event", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Key financial details (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "financial dependency", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = audited/confirmed, 0.7 = reported, 0.5 = estimated, 0.3 = suspected
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

FINANCIAL EXTRACTION GUIDELINES:
- Entities: companies, individuals, accounts, funds, intermediaries
- Properties: registration numbers, account numbers, LEI codes, jurisdictions
- Each transaction, filing, disclosure, or material event is a separate situation
- Amounts: always include currency, exact figures, and direction (inflow/outflow)
- Track: sender → recipient, account → account, fund → investment
- Temporal precision: transaction dates, reporting periods, fiscal quarters
- Causal links: investment → return, violation → penalty, merger → restructuring
- Flag unusual patterns: round-number transactions, rapid fund movements, shell company chains
- For compliance: track regulatory requirements, deadlines, filing obligations"#;

/// System prompt for medical/clinical extraction.
pub const MEDICAL_EXTRACTION_PROMPT: &str = r#"You are a medical extraction engine. Extract structured data from clinical documents as JSON.

RULES:
- Return ONLY a JSON object, no other text, no markdown, no explanation
- Every field that holds a list MUST be a JSON array [], even if empty
- Do NOT invent new field names or enum values

JSON SCHEMA:
{
  "entities": [{"name": "...", "aliases": [], "entity_type": "Actor", "properties": {}, "confidence": 0.9}],
  "situations": [{"name": "Short name (5-8 words)", "description": "A detailed 1-2 sentence summary of the clinical event", "temporal_marker": null, "location": null, "narrative_level": "Event", "content_blocks": [{"content_type": "Text", "content": "Clinical details (1-3 sentences)"}], "confidence": 0.8, "text_start": "first 8-12 verbatim words of this situation's prose", "text_end": "last 8-12 verbatim words of this situation's prose"}],
  "participations": [{"entity_name": "...", "situation_index": 1, "role": "Protagonist", "action": "...", "confidence": 0.8}],
  "causal_links": [{"from": 1, "to": 2, "causal_type": "Contributing", "description": "clinical relationship", "confidence": 0.8}],
  "temporal_relations": [{"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.8}]
}

FIELD VALUES:
- entity_type: Actor, Location, Artifact, Concept, Organization
- narrative_level: Story, Arc, Sequence, Scene, Beat, Event
- role: Protagonist, Antagonist, Witness, Target, Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion
- relation (temporal): Before, After, Meets, Overlaps, During, Contains, Starts, Finishes, Equals
- causal_type: Necessary, Sufficient, Contributing, Enabling
- confidence: 1.0 = confirmed diagnosis, 0.7 = clinical assessment, 0.5 = differential diagnosis, 0.3 = rule-out
- situation_index: 1-BASED position in the situations array. First situation = 1, second = 2, etc.

CRITICAL CONSISTENCY RULES:
- Every entity_name in participations MUST exactly match a name (or alias) in the entities array.
- situation_index is 1-BASED (1 to N where N = number of situations). First situation = 1.
- Do NOT use locations, rooms, or places as entity_name unless you extracted them as entities first.

MEDICAL EXTRACTION GUIDELINES:
- Patients are Actor entities; providers/physicians are also Actor entities
- Conditions, diagnoses, and symptoms are Concept entities
- Medications, devices, and procedures are Artifact entities
- Facilities, departments are Location entities
- Each clinical event (presentation, diagnosis, treatment, outcome) is a separate situation
- Properties for patients: age, sex, relevant history, identifiers (de-identified)
- Properties for medications: dose, route, frequency, duration
- Track temporal sequence: symptom onset → presentation → diagnosis → treatment → outcome
- Causal links: condition → treatment, exposure → symptom, intervention → outcome
- Use narrative_level "Arc" for disease course, "Event" for specific encounters
- For lab results: include values, units, and reference ranges in properties
- Distinguish between confirmed diagnoses and differential/working diagnoses via confidence"#;

/// System prompt for step-2 enrichment (deep annotation pass).
pub const ENRICHMENT_SYSTEM_PROMPT: &str = r#"You are a narrative analysis engine. Given a text passage and its extracted structure (entities, situations, participations), produce deep annotations.

Return ONLY a JSON object with these arrays (all optional — include only what is present in the text):

{
  "entity_beliefs": [
    {"entity_name": "...", "beliefs": ["fact they believe"], "goals": ["what they want"], "misconceptions": ["what they're wrong about"], "confidence": 0.8}
  ],
  "game_structures": [
    {"situation_index": 1, "game_type": "Bargaining", "info_structure": "Incomplete", "description": "why this is a strategic interaction"}
  ],
  "discourse": [
    {"situation_index": 1, "order": "simultaneous", "duration": "scene", "focalization": "entity name", "voice": "heterodiegetic"}
  ],
  "info_sets": [
    {"entity_name": "...", "situation_index": 1, "knows_before": ["fact"], "learns": ["fact"], "reveals": ["fact"]}
  ],
  "extra_causal_links": [
    {"from_situation_index": 1, "to_situation_index": 2, "causal_type": "Contributing", "mechanism": "why A causes B", "strength": 0.7, "confidence": 0.8}
  ],
  "outcomes": [
    {"situation_index": 1, "deterministic": "what must follow", "alternatives": [{"description": "what else could happen", "probability": 0.3}]}
  ],
  "temporal_chain": [
    {"situation_a_index": 1, "situation_b_index": 2, "relation": "Before", "confidence": 0.9}
  ],
  "temporal_normalizations": [
    {"situation_index": 1, "normalized_date": "12069-01-01", "relative_description": null}
  ]
}

FIELD VALUES:
- game_type: PrisonersDilemma, Coordination, Signaling, Auction, Bargaining, ZeroSum, AsymmetricInformation
- info_structure: Complete, Incomplete, Imperfect, AsymmetricBecomingComplete
- order: analepsis (flashback), prolepsis (flash-forward), simultaneous
- duration: scene (real-time), summary (compressed), ellipsis (skipped), pause (narration stops action), stretch (slow-motion)
- voice: homodiegetic (narrator is a character), heterodiegetic (narrator is external)
- causal_type: Necessary, Sufficient, Contributing, Enabling
- relation (temporal): Before, After, Meets, MetBy, Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals
- situation_index: 1-BASED (first situation = 1)

GUIDELINES:
- entity_beliefs: what does each major actor BELIEVE, WANT, and MISUNDERSTAND at this point? Focus on beliefs that drive their actions or create dramatic irony.
- game_structures: any situation where actors make strategic choices with interdependent outcomes. Negotiations, confrontations, deceptions, alliances, betrayals.
- discourse: how is the narrative TOLD? POV shifts, time jumps, pacing changes. For first-person sections, set focalization to the narrator's entity name and voice to "homodiegetic". For third-person omniscient, use "heterodiegetic".
- info_sets: what does each actor KNOW going in, LEARN during, and REVEAL to others? This is critical for tracking information asymmetry.
- extra_causal_links: cause-effect chains between situations. "A happened BECAUSE of B". Be thorough — most narratives are denser causally than they appear. Specifically look for REVEALED MECHANISMS: when a later situation explains why an earlier situation happened (a reveal, a flashback explanation, a villain's confession, a diary entry clarifying a past event), add a CausalLink from the earlier situation to the later situation's explanation, or vice versa if the earlier one caused the later. Cross-situation causal links of this kind are the most commonly missed.
- outcomes: for pivotal situations, what are the possible outcomes and their likelihood?
- temporal_chain: IMPORTANT — produce Allen temporal relations between ALL pairs of situations in this chunk. Order every pair: is A Before B? During B? Overlaps B? Be thorough — this builds the timeline. At minimum, chain all situations in chronological order (1 Before 2, 2 Before 3, etc.).
- temporal_normalizations: for EVERY situation, try to normalize the temporal_marker to an absolute date if possible (even fictional dates like "12069-01-01" for "Year 12,069 GE"). If no absolute date, provide a relative_description ("3 days after situation 2", "same evening as situation 1"). This is critical for timeline construction.
- Only annotate what is actually present or strongly implied in the text. Use confidence to reflect certainty.

ANAPHORA SWEEP (do this explicitly):
- Walk each situation's raw text once more looking for descriptive references ("a figure", "the stranger", "the voice", "someone", "the killer", "a shape"). For each one you can now resolve from context — including context from OTHER situations in this chunk or earlier chunks — note the resolution in a belief/info_set entry so the pipeline can later add the missing participation. Example: if situation 3 says "a figure rose from the boot" and situation 1 established that the Electric Monk had climbed into the boot, add an info_set entry noting "Electric Monk is the figure in situation 3, based on situation 1"."#;

/// Build the enrichment prompt: step-1 extraction summary + original text.
pub fn build_enrichment_prompt(chunk: &TextChunk, extraction: &NarrativeExtraction) -> String {
    let mut prompt = String::new();

    // Summarize the step-1 extraction so the LLM can reference by name/index
    prompt.push_str("[EXTRACTED STRUCTURE]\n");

    prompt.push_str("Entities:\n");
    for (i, e) in extraction.entities.iter().enumerate() {
        prompt.push_str(&format!(
            "  {}. {} ({})\n",
            i + 1,
            e.name,
            format!("{:?}", e.entity_type)
        ));
    }

    prompt.push_str("Situations:\n");
    for (i, s) in extraction.situations.iter().enumerate() {
        let name = s.name.as_deref().unwrap_or(&s.description);
        let temporal = s
            .temporal_marker
            .as_deref()
            .map(|t| format!(" [time: {}]", t))
            .unwrap_or_default();
        let location = s
            .location
            .as_deref()
            .map(|l| format!(" [loc: {}]", l))
            .unwrap_or_default();
        prompt.push_str(&format!("  {}. {}{}{}\n", i + 1, name, temporal, location));
    }

    if !extraction.participations.is_empty() {
        prompt.push_str("Participations:\n");
        for p in &extraction.participations {
            prompt.push_str(&format!(
                "  {} → situation {} as {:?}",
                p.entity_name,
                p.situation_index + 1, // show as 1-based since prompt uses 1-based
                p.role
            ));
            if let Some(ref action) = p.action {
                prompt.push_str(&format!(" ({})", action));
            }
            prompt.push('\n');
        }
    }
    prompt.push_str("[END EXTRACTED STRUCTURE]\n\n");

    prompt.push_str("[ORIGINAL TEXT]\n");
    prompt.push_str(&chunk.text);
    prompt.push_str("\n[END TEXT]\n\n");

    prompt.push_str("Now produce deep annotations for the above structure. Focus on beliefs, strategic interactions, causal chains, information asymmetry, and narrative discourse. Return ONLY JSON.");
    prompt
}

/// System prompt for batch place-name canonicalization.
///
/// Disambiguates raw place strings extracted from a narrative ("Marseilles",
/// "Petersburg", "Cambridge") into modern canonical names + ISO country codes
/// using narrative-setting context. Output is JSON-only so the parser can take it directly.
pub const PLACE_CANONICALIZATION_PROMPT: &str = r#"You are a geographic disambiguation engine. Given a narrative's setting and a list of place strings as they appear in the text, return their modern canonical names and country codes so they can be geocoded.

CRITICAL RULES:
1. Output one row PER INPUT PLACE — do not skip rows you find easy or obvious. Skipping inflates work for the operator and breaks the join.
2. Use the modern canonical place name (e.g. "Marseille" not "Marseilles", "Saint Petersburg" not "Petrograd"). Real city/region names only — never invent fictional toponyms.
3. Country code must be ISO 3166-1 alpha-2, lowercase ("fr", "ru", "us").
4. Pick the place that fits the narrative setting. "Marseilles" in a 19th-century French novel is Marseille, France — NOT Marseilles, Illinois. "Yanina" in Dumas is Ioannina, Greece — NOT Yanina, Argentina.
5. Echo back the `uuid` and `raw_name` fields verbatim — they're the join key. Mismatched echoes are dropped silently.
6. ONLY skip a place when (a) it is unambiguously fictional with no real-world counterpart, or (b) it is a generic noun ("the village", "the chapel") with no proper name. When in doubt, return your best guess with a low `confidence` value rather than dropping it.
7. Return ONLY a JSON array — no prose, no markdown fences, no surrounding object.

Output format:
[
  {"uuid": "...", "raw_name": "Marseilles", "canonical_name": "Marseille", "country_code": "fr", "admin_region": "Provence-Alpes-Côte d'Azur", "confidence": 0.95},
  {"uuid": "...", "raw_name": "Petersburg", "canonical_name": "Saint Petersburg", "country_code": "ru", "confidence": 0.9},
  {"uuid": "...", "raw_name": "Yanina", "canonical_name": "Ioannina", "country_code": "gr", "confidence": 0.85}
]
"#;

/// Build the user-message body for `canonicalize_places`.
pub fn build_canonicalize_places_prompt(
    setting: &crate::ingestion::extraction::NarrativeSettingHint,
    places: &[(String, String)],
) -> String {
    let mut prompt = String::new();
    prompt.push_str("Narrative setting:\n");
    if setting.setting.is_empty() {
        prompt.push_str("(no setting context provided — disambiguate by best guess only if highly confident)\n\n");
    } else {
        prompt.push_str(&setting.setting);
        prompt.push_str("\n");
        if let Some(cc) = &setting.country_hint {
            prompt.push_str(&format!("Default country if otherwise ambiguous: {}\n", cc));
        }
        prompt.push_str("\n");
    }
    prompt.push_str("Places to canonicalize:\n[\n");
    for (uuid, raw) in places {
        let raw_escaped = raw.replace('\\', "\\\\").replace('"', "\\\"");
        prompt.push_str(&format!(
            "  {{\"uuid\": \"{}\", \"raw_name\": \"{}\"}},\n",
            uuid, raw_escaped
        ));
    }
    prompt.push_str("]\n\nReturn ONLY the JSON array described in the system prompt.");
    prompt
}

/// System prompt for cross-chunk temporal reconciliation.
pub const TEMPORAL_RECONCILIATION_PROMPT: &str = r#"You are a temporal analysis engine. Given a list of situations from a narrative with their temporal markers, establish a complete timeline.

Return ONLY a JSON object:
{
  "relations": [
    {"situation_a": "situation name/desc", "situation_b": "situation name/desc", "relation": "Before", "confidence": 0.9}
  ],
  "timeline": [
    {"situation": "situation name/desc", "date": "normalized date", "confidence": 0.8}
  ]
}

TEMPORAL RELATIONS: Before, After, Meets, MetBy, Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals

GUIDELINES:
- relations: establish Before/After ordering between situations from DIFFERENT chunks that relate to each other. Focus on key plot-advancing events, not every pair.
- timeline: for each situation where you can determine an absolute date (even fictional calendar systems), provide a normalized ISO-like date. For fictional calendars (e.g. "Galactic Era 12,069"), use a consistent mapping (e.g. GE years as-is: "12069-01-01").
- Use the temporal markers, chapter context, and narrative logic to infer ordering.
- When multiple situations share the same time reference, use "Equals" or "During".
- Confidence should reflect how certain the temporal relationship is: 1.0 = explicitly stated, 0.7 = strongly implied, 0.5 = inferred from narrative structure.
- REVEALED MECHANISMS (cross-chunk): when a situation in a later chunk EXPLAINS an earlier situation (a reveal, a confession, a flashback explanation, a witness statement about an earlier event), flag this with a temporal Before/After relation AND note in your reasoning that the later chunk explains the earlier. The pipeline uses these to add backfilled causal links.
- CLIMAX COVERAGE: look at the situations in the final 15-20% of chunks (highest chunk indices). If there are fewer situations there than in the middle chunks, the climax was under-extracted — note this as a warning in your timeline output so the pipeline can flag it."#;

/// Build the reconciliation prompt from all chunks' situation summaries.
pub fn build_reconciliation_prompt(
    chunk_summaries: &[(usize, Vec<(String, Option<String>, Option<String>)>)],
) -> String {
    let mut prompt = String::new();
    prompt.push_str("[NARRATIVE SITUATIONS BY CHUNK]\n\n");

    for (chunk_idx, situations) in chunk_summaries {
        prompt.push_str(&format!("Chunk {}:\n", chunk_idx + 1));
        for (i, (name, temporal, location)) in situations.iter().enumerate() {
            prompt.push_str(&format!("  {}. {}", i + 1, name));
            if let Some(t) = temporal {
                prompt.push_str(&format!(" [time: {}]", t));
            }
            if let Some(l) = location {
                prompt.push_str(&format!(" [loc: {}]", l));
            }
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    prompt.push_str("Establish temporal relationships between situations across chunks. Focus on key events and their chronological ordering. Pin dates where possible. Return ONLY JSON.");
    prompt
}

/// System prompt for entity coreference resolution.
// ─── SingleSession prompt builders ──────────────────────────

/// Build the system prompt for a SingleSession, wrapping the base extraction prompt
/// with instructions about chunk markers and multi-turn format.
pub fn build_session_system_prompt(base_system_prompt: &str) -> String {
    format!(
        "{}\n\n\
MULTI-CHUNK SESSION MODE:\n\
The user will send an entire document with chunk markers:\n\
  [=== CHUNK N: \"Title\" ===] ... [=== END CHUNK N ===]\n\
Acknowledge receipt briefly, then the user will ask you to extract from each chunk individually.\n\
When extracting a chunk, ONLY extract entities/situations/participations from text between the specified CHUNK markers.\n\
Use CONSISTENT entity names across all chunks — refer to previous extractions.\n\
Return ONLY the JSON extraction for the requested chunk, no other text.\n\
situation_index is 1-BASED within each chunk (first situation in the chunk = 1).\n\n\
CRITICAL SESSION ADVANTAGE — you can see the whole document, USE IT:\n\
- When a later chunk reveals the identity of a \"figure\"/\"stranger\"/\"voice\" mentioned in an earlier chunk, resolve the reference to the now-known entity. You already have the full document in context — use that knowledge to produce correctly-named participations from the first chunk onward.\n\
- When a later chunk explains WHY an earlier event happened (a reveal, a confession, a flashback), note this mentally and include cross-chunk causal_links during the final reconciliation step.\n\
- Identify the narrator of each section early. If the document has multiple first-person narrators (epistolary, multi-POV), track which character's voice is active in each chapter/section and attribute \"I\" statements to that specific character. Never create an unnamed \"Narrator\" entity.\n\
- Every killing/assault/confrontation scene needs three participations: Victim (Target), Actor (Antagonist or Protagonist), Weapon/Instrument if used.",
        base_system_prompt
    )
}

/// Build the per-chunk extraction prompt for SingleSession (references chunk by marker, no content).
pub fn build_session_chunk_prompt(
    chunk_index: usize,
    title: Option<&str>,
    accumulator_summary: &str,
) -> String {
    let mut prompt = String::new();
    if !accumulator_summary.is_empty() {
        prompt.push_str(accumulator_summary);
        prompt.push_str("\n\n");
    }
    match title {
        Some(t) => prompt.push_str(&format!(
            "Extract all entities, situations, participations, causal links, and temporal relations from CHUNK {}: \"{}\".\n\
             Return ONLY the JSON.",
            chunk_index + 1,
            t
        )),
        None => prompt.push_str(&format!(
            "Extract all entities, situations, participations, causal links, and temporal relations from CHUNK {}.\n\
             Return ONLY the JSON.",
            chunk_index + 1
        )),
    }
    prompt
}

/// Build the per-chunk enrichment prompt for SingleSession.
pub fn build_session_enrichment_prompt(chunk_index: usize, title: Option<&str>) -> String {
    format!(
        "Now produce enrichment annotations (entity_beliefs, game_structures, discourse, info_sets, \
         extra_causal_links, outcomes, temporal_chain, temporal_normalizations) for CHUNK {}{}.\n\
         You already extracted this chunk above. Return ONLY the enrichment JSON.",
        chunk_index + 1,
        title.map(|t| format!(": \"{}\"", t)).unwrap_or_default()
    )
}

/// Build the final in-session reconciliation prompt.
pub fn build_session_reconciliation_prompt(num_chunks: usize) -> String {
    format!(
        "You have now extracted all {} chunks of this document. Perform a final reconciliation across the whole narrative.\n\n\
         Return a JSON object with:\n\
         {{\n\
           \"entity_merges\": [{{\"canonical_name\": \"...\", \"duplicate_names\": [\"...\"]}}],\n\
           \"timeline\": [{{\"situation\": \"name\", \"date\": \"YYYY-MM-DD\", \"confidence\": 0.9}}],\n\
           \"confidence_adjustments\": [{{\"name\": \"...\", \"adjusted_confidence\": 0.9, \"reason\": \"...\"}}],\n\
           \"cross_chunk_causal_links\": [{{\"from_situation_index\": 1, \"to_situation_index\": 1, \"causal_type\": \"Contributing\", \"mechanism\": \"...\", \"strength\": 0.7, \"confidence\": 0.8}}],\n\
           \"anaphora_resolutions\": [{{\"situation_name\": \"name or 1-based global index\", \"descriptive_reference\": \"the figure | the stranger | ...\", \"resolved_entity\": \"CanonicalName\", \"role\": \"Antagonist|Instrument|Witness|...\", \"action\": \"what the entity did\", \"confidence\": 0.9}}],\n\
           \"coverage_warnings\": [\"free-text description of what seems under-extracted\"]\n\
         }}\n\n\
         THIS IS A SIX-STEP RECONCILIATION PASS. Do every step — it routinely catches 20-40% of participations and causal links that the per-chunk extractions missed.\n\n\
         1. Entity merges: identify entities that appear under different names across chunks (e.g. 'Svlad Cjelli' and 'Dirk Gently'). Produce entity_merges entries.\n\n\
         2. Anaphora / descriptive-reference sweep: walk all situations and find any that refer to a character by description rather than name ('a figure', 'the stranger', 'the killer', 'the voice', 'someone', 'the intruder'). For each one where you can now identify the referent from the full document, produce an anaphora_resolutions entry so the pipeline can add the missing Participation with the correct role. Pay special attention to: violent scenes without a named agent; dialogue with no identified speaker; any Scene situation that has zero participations in the extracted structure.\n\n\
         3. Revealed-mechanism backfill: for every reveal, confession, flashback, or explanation in the document, add a cross_chunk_causal_links entry from the explained situation to the explaining situation (or vice versa if the earlier event caused the later one). Cross-chunk causation is the single most commonly-missed kind of edge in single-pass extraction.\n\n\
         4. Protagonist coverage audit: for every main character, verify they have participations in MANY situations (at least 5 for a main character in a full-length book). If a main character has few participations, they were under-extracted — note it in coverage_warnings.\n\n\
         5. Climax coverage audit: compare the number of situations extracted from the final 15-20% of chunks to the number extracted from the middle third. If the ending has fewer, the climax was under-extracted — add a coverage_warning listing specific events (confrontations, reveals, transformations, consequences) you remember from the final chapters that should have been extracted as Scene situations.\n\n\
         6. Narrator consistency audit (for first-person / multi-POV books): verify that every first-person section's situations are attributed to the correct narrator for that section. In multi-narrator books the most common mistake is attributing later narrators' chapters to the first narrator. Note any mismatches in coverage_warnings.\n\n\
         7. Timeline: pin situations to absolute dates where possible (even fictional calendars).\n\n\
         8. Confidence adjustments: raise confidence for entities/situations confirmed across multiple chunks, lower for contradicted ones.\n\n\
         Return ONLY the JSON.",
        num_chunks
    )
}

pub const COREFERENCE_SYSTEM_PROMPT: &str = r#"You are an entity resolution engine. Given a list of entity names and their contexts, identify which names refer to the same real-world entity.

Return a JSON array of coreference groups:
[
  {
    "canonical_name": "the primary name",
    "aliases": ["other", "names"],
    "confidence": 0.0-1.0
  }
]

Rules:
- Group names that clearly refer to the same entity
- Consider nicknames, formal/informal names, titles, etc.
- Only merge when confident — when in doubt, keep separate
- Return ONLY the JSON array"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_acquire() {
        let rl = RateLimiter::new(60); // 1 per second
                                       // Should succeed for first request
        assert!(rl.try_acquire().is_ok());
    }

    #[test]
    fn test_rate_limiter_exhaustion() {
        let rl = RateLimiter::new(1); // Very low rate
                                      // First should succeed
        assert!(rl.try_acquire().is_ok());
        // Second should be rate limited
        assert!(rl.try_acquire().is_err());
    }

    #[test]
    fn test_build_extraction_prompt_basic() {
        let chunk = TextChunk {
            chunk_id: 0,
            text: "He entered the room.".to_string(),
            chapter: None,
            start_offset: 0,
            end_offset: 20,
            overlap_prefix: String::new(),
        };
        let prompt = build_extraction_prompt(&chunk);
        assert!(prompt.contains("He entered the room."));
        assert!(!prompt.contains("CONTEXT"));
    }

    #[test]
    fn test_build_extraction_prompt_with_context() {
        let chunk = TextChunk {
            chunk_id: 1,
            text: "He entered the room.".to_string(),
            chapter: Some("Chapter 1".to_string()),
            start_offset: 100,
            end_offset: 120,
            overlap_prefix: "Previous text here.".to_string(),
        };
        let prompt = build_extraction_prompt(&chunk);
        assert!(prompt.contains("CONTEXT FROM PREVIOUS CHUNK"));
        assert!(prompt.contains("Previous text here."));
        assert!(prompt.contains("CHAPTER: Chapter 1"));
        assert!(prompt.contains("He entered the room."));
    }
}
