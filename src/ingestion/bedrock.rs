//! AWS Bedrock API client implementing NarrativeExtractor.
//!
//! Feature-gated behind `bedrock`. Uses the Bedrock InvokeModel API with
//! manual AWS SigV4 signing (via `hmac` + `sha2` crates).

#[cfg(feature = "bedrock")]
mod inner {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use hmac::{Hmac, Mac};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};

    use crate::error::{Result, TensaError};
    use crate::ingestion::chunker::TextChunk;
    use crate::ingestion::extraction::{parse_llm_response, NarrativeExtraction};
    use crate::ingestion::llm::{
        build_extraction_prompt, build_extraction_prompt_with_context, NarrativeExtractor,
        RawLlmExchange, EXTRACTION_SYSTEM_PROMPT,
    };

    type HmacSha256 = Hmac<Sha256>;

    /// AWS Bedrock API client for Claude models.
    pub struct BedrockClient {
        region: String,
        model_id: String,
        access_key_id: String,
        secret_access_key: String,
        client: reqwest::Client,
        system_prompt: String,
        cancel_flag: Mutex<Arc<AtomicBool>>,
    }

    // ─── Bedrock API types (Claude on Bedrock) ───────────────────

    #[derive(Serialize)]
    struct BedrockRequest {
        anthropic_version: String,
        max_tokens: u32,
        system: String,
        messages: Vec<BedrockMessage>,
    }

    #[derive(Serialize)]
    struct BedrockMessage {
        role: String,
        content: String,
    }

    #[derive(Deserialize)]
    struct BedrockResponse {
        content: Option<Vec<BedrockContentItem>>,
    }

    #[derive(Deserialize)]
    struct BedrockContentItem {
        text: Option<String>,
    }

    impl BedrockClient {
        /// Create a new Bedrock client.
        pub fn new(
            region: String,
            model_id: String,
            access_key_id: String,
            secret_access_key: String,
        ) -> Self {
            Self {
                client: reqwest::Client::builder()
                    .timeout(Duration::from_secs(120))
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new()),
                region,
                model_id,
                access_key_id,
                secret_access_key,
                system_prompt: EXTRACTION_SYSTEM_PROMPT.to_string(),
                cancel_flag: Mutex::new(Arc::new(AtomicBool::new(false))),
            }
        }

        /// Compute HMAC-SHA256.
        fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
            let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
            mac.update(data);
            mac.finalize().into_bytes().to_vec()
        }

        /// Compute SHA-256 hex digest.
        fn sha256_hex(data: &[u8]) -> String {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hex_encode(&hasher.finalize())
        }

        /// Build the SigV4 signing key.
        fn signing_key(&self, date_stamp: &str) -> Vec<u8> {
            let k_date = Self::hmac_sha256(
                format!("AWS4{}", self.secret_access_key).as_bytes(),
                date_stamp.as_bytes(),
            );
            let k_region = Self::hmac_sha256(&k_date, self.region.as_bytes());
            let k_service = Self::hmac_sha256(&k_region, b"bedrock");
            Self::hmac_sha256(&k_service, b"aws4_request")
        }

        /// Send a message to Bedrock (Claude on Bedrock format).
        async fn send_message_async(&self, system: &str, user: &str) -> Result<String> {
            let flag = self.cancel_flag.lock().unwrap().clone();
            if flag.load(Ordering::Relaxed) {
                return Err(TensaError::LlmError("Cancelled by user".into()));
            }

            let host = format!("bedrock-runtime.{}.amazonaws.com", self.region);
            let url = format!("https://{}/model/{}/invoke", host, self.model_id);

            let body = BedrockRequest {
                anthropic_version: "bedrock-2023-05-31".to_string(),
                max_tokens: 16384,
                system: system.to_string(),
                messages: vec![BedrockMessage {
                    role: "user".to_string(),
                    content: user.to_string(),
                }],
            };
            let body_bytes =
                serde_json::to_vec(&body).map_err(|e| TensaError::Serialization(e.to_string()))?;
            let payload_hash = Self::sha256_hex(&body_bytes);

            // SigV4 date strings
            let now = chrono::Utc::now();
            let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
            let date_stamp = now.format("%Y%m%d").to_string();

            let canonical_uri = format!("/model/{}/invoke", self.model_id);
            let canonical_querystring = "";
            let canonical_headers = format!(
                "content-type:application/json\nhost:{}\nx-amz-date:{}\n",
                host, amz_date
            );
            let signed_headers = "content-type;host;x-amz-date";

            let canonical_request = format!(
                "POST\n{}\n{}\n{}\n{}\n{}",
                canonical_uri,
                canonical_querystring,
                canonical_headers,
                signed_headers,
                payload_hash
            );

            let credential_scope = format!("{}/{}/bedrock/aws4_request", date_stamp, self.region);
            let string_to_sign = format!(
                "AWS4-HMAC-SHA256\n{}\n{}\n{}",
                amz_date,
                credential_scope,
                Self::sha256_hex(canonical_request.as_bytes())
            );

            let signing_key = self.signing_key(&date_stamp);
            let signature = hex_encode(&Self::hmac_sha256(&signing_key, string_to_sign.as_bytes()));

            let authorization = format!(
                "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
                self.access_key_id, credential_scope, signed_headers, signature
            );

            let resp = self
                .client
                .post(&url)
                .header("content-type", "application/json")
                .header("host", &host)
                .header("x-amz-date", &amz_date)
                .header("authorization", &authorization)
                .body(body_bytes)
                .send()
                .await
                .map_err(|e| TensaError::LlmError(format!("Bedrock request failed: {}", e)))?;

            let status = resp.status();
            if !status.is_success() {
                let body_text = resp.text().await.unwrap_or_default();
                return Err(TensaError::LlmError(format!(
                    "Bedrock HTTP {}: {}",
                    status, body_text
                )));
            }

            let response: BedrockResponse = resp
                .json()
                .await
                .map_err(|e| TensaError::LlmError(format!("Bedrock parse error: {}", e)))?;

            let text = response
                .content
                .and_then(|c| c.into_iter().next())
                .and_then(|item| item.text)
                .ok_or_else(|| TensaError::LlmError("Empty Bedrock response".into()))?;

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
                model: Some(self.model_id.clone()),
                endpoint: Some(format!(
                    "bedrock-runtime.{}.amazonaws.com/model/{}",
                    self.region, self.model_id
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
                                    "Bedrock extraction failed after retry: {}",
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

    impl NarrativeExtractor for BedrockClient {
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
            Some(self.model_id.clone())
        }

        fn answer_question(&self, system_prompt: &str, question: &str) -> Result<String> {
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| TensaError::LlmError("No tokio runtime available".into()))?;
            rt.block_on(self.send_message_async(system_prompt, question))
                .map_err(|e| TensaError::LlmError(format!("Bedrock RAG answer failed: {}", e)))
        }
    }

    /// Encode bytes as lowercase hex string.
    fn hex_encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(feature = "bedrock")]
pub use inner::BedrockClient;

#[cfg(test)]
#[cfg(feature = "bedrock")]
mod tests {
    use super::BedrockClient;

    #[test]
    fn test_bedrock_client_creation() {
        let client = BedrockClient::new(
            "us-east-1".into(),
            "anthropic.claude-3-sonnet-20240229-v1:0".into(),
            "AKIA_TEST".into(),
            "secret_test".into(),
        );
        // Verify it doesn't panic during construction
        use crate::ingestion::llm::NarrativeExtractor;
        assert_eq!(
            client.model_name(),
            Some("anthropic.claude-3-sonnet-20240229-v1:0".to_string())
        );
    }
}
