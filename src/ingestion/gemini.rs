//! Google Gemini API client implementing NarrativeExtractor.
//!
//! Feature-gated behind `gemini`. Uses the Gemini REST API at
//! `generativelanguage.googleapis.com`.

#[cfg(feature = "gemini")]
mod inner {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use serde::{Deserialize, Serialize};

    use crate::error::{Result, TensaError};
    use crate::ingestion::chunker::TextChunk;
    use crate::ingestion::extraction::{parse_llm_response, NarrativeExtraction};
    use crate::ingestion::llm::{
        build_extraction_prompt, build_extraction_prompt_with_context, NarrativeExtractor,
        RawLlmExchange, EXTRACTION_SYSTEM_PROMPT,
    };

    /// Google Gemini API client.
    pub struct GeminiClient {
        api_key: String,
        model: String,
        client: reqwest::Client,
        system_prompt: String,
        cancel_flag: Mutex<Arc<AtomicBool>>,
    }

    // ─── Gemini API types ─────────────────────────────────────────

    #[derive(Serialize)]
    struct GeminiRequest {
        contents: Vec<GeminiContent>,
        #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
        system_instruction: Option<GeminiContent>,
    }

    #[derive(Serialize, Deserialize)]
    struct GeminiContent {
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        parts: Vec<GeminiPart>,
    }

    #[derive(Serialize, Deserialize)]
    struct GeminiPart {
        text: String,
    }

    #[derive(Deserialize)]
    struct GeminiResponse {
        candidates: Option<Vec<GeminiCandidate>>,
    }

    #[derive(Deserialize)]
    struct GeminiCandidate {
        content: Option<GeminiContent>,
    }

    impl GeminiClient {
        /// Create a new Gemini API client.
        pub fn new(api_key: String, model: String) -> Self {
            Self {
                client: reqwest::Client::builder()
                    .timeout(Duration::from_secs(120))
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new()),
                api_key,
                model,
                system_prompt: EXTRACTION_SYSTEM_PROMPT.to_string(),
                cancel_flag: Mutex::new(Arc::new(AtomicBool::new(false))),
            }
        }

        /// Send a message to the Gemini API.
        async fn send_message_async(&self, system: &str, user: &str) -> Result<String> {
            let flag = self.cancel_flag.lock().unwrap().clone();
            if flag.load(Ordering::Relaxed) {
                return Err(TensaError::LlmError("Cancelled by user".into()));
            }

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                self.model, self.api_key
            );

            let request = GeminiRequest {
                contents: vec![GeminiContent {
                    role: Some("user".into()),
                    parts: vec![GeminiPart {
                        text: user.to_string(),
                    }],
                }],
                system_instruction: Some(GeminiContent {
                    role: None,
                    parts: vec![GeminiPart {
                        text: system.to_string(),
                    }],
                }),
            };

            let resp = self
                .client
                .post(&url)
                .header("content-type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|e| TensaError::LlmError(format!("Gemini request failed: {}", e)))?;

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(TensaError::LlmError(format!(
                    "Gemini HTTP {}: {}",
                    status, body
                )));
            }

            let body: GeminiResponse = resp
                .json()
                .await
                .map_err(|e| TensaError::LlmError(format!("Gemini parse error: {}", e)))?;

            let text = body
                .candidates
                .and_then(|c| c.into_iter().next())
                .and_then(|c| c.content)
                .and_then(|c| c.parts.into_iter().next())
                .map(|p| p.text)
                .ok_or_else(|| TensaError::LlmError("Empty Gemini response".into()))?;

            Ok(text)
        }

        /// Synchronous extraction helper.
        fn extract_with_prompt(&self, prompt: String) -> Result<NarrativeExtraction> {
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
            let response = rt.block_on(self.send_message_async(&self.system_prompt, &prompt))?;
            parse_llm_response(&response)
        }

        /// Synchronous extraction with logging.
        fn extract_with_prompt_logged(
            &self,
            prompt: String,
        ) -> Result<(NarrativeExtraction, RawLlmExchange)> {
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
            let start = Instant::now();
            let response = rt.block_on(self.send_message_async(&self.system_prompt, &prompt))?;

            let mut exchange = RawLlmExchange {
                system_prompt: self.system_prompt.clone(),
                user_prompt: prompt.clone(),
                raw_response: response.clone(),
                retry_prompt: None,
                retry_response: None,
                parse_error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                model: Some(self.model.clone()),
                endpoint: Some(format!(
                    "generativelanguage.googleapis.com/v1beta/models/{}",
                    self.model
                )),
            };

            match parse_llm_response(&response) {
                Ok(ext) => Ok((ext, exchange)),
                Err(first_err) => {
                    exchange.parse_error = Some(format!("{}", first_err));
                    let repair = format!(
                        "{}\n\nThe above JSON had a parse error: {}\nPlease return the COMPLETE corrected JSON.",
                        prompt, first_err
                    );
                    exchange.retry_prompt = Some(repair.clone());
                    match rt.block_on(self.send_message_async(&self.system_prompt, &repair)) {
                        Ok(retry) => {
                            exchange.retry_response = Some(retry.clone());
                            exchange.duration_ms = start.elapsed().as_millis() as u64;
                            match parse_llm_response(&retry) {
                                Ok(ext) => Ok((ext, exchange)),
                                Err(e) => Err(TensaError::ExtractionError(format!(
                                    "Gemini extraction failed after retry: {}",
                                    e
                                ))),
                            }
                        }
                        Err(_) => Err(first_err),
                    }
                }
            }
        }
    }

    impl NarrativeExtractor for GeminiClient {
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

        fn answer_question(&self, system_prompt: &str, question: &str) -> Result<String> {
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
            rt.block_on(self.send_message_async(system_prompt, question))
                .map_err(|e| TensaError::LlmError(format!("Gemini RAG answer failed: {}", e)))
        }
    }
}

#[cfg(feature = "gemini")]
pub use inner::GeminiClient;

#[cfg(test)]
#[cfg(feature = "gemini")]
mod tests {
    use super::GeminiClient;
    use crate::ingestion::llm::NarrativeExtractor;

    #[test]
    fn test_gemini_client_creation() {
        let client = GeminiClient::new("test-key".into(), "gemini-2.0-flash".into());
        assert_eq!(client.model_name(), Some("gemini-2.0-flash".to_string()));
    }
}
