//! OpenAI / OpenRouter image-generation client.
//!
//! Two wire formats are supported because OpenRouter does NOT expose
//! `/v1/images/generations`. Image-output models on OpenRouter come back
//! through `/chat/completions` with `modalities: ["image"]`:
//!
//! ```json
//! POST {base_url}/chat/completions
//! { "model": "...", "messages": [{"role":"user","content":"prompt"}],
//!   "modalities": ["image"] }
//!
//! 200 -> { "choices": [{ "message": { "images": [
//!   { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }
//! ] } }] }
//! ```
//!
//! OpenAI direct (`gpt-image-1`, `dall-e-3`) and OpenAI-shape local servers
//! still use the standard `/v1/images/generations`:
//!
//! ```json
//! POST {base_url}/images/generations
//! { "model": "...", "prompt": "...", "n": 1, "response_format": "b64_json" }
//!
//! 200 -> { "data": [ { "b64_json": "..." } ] }
//! ```

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use reqwest::Client;
use serde::Deserialize;

use super::provider::{GeneratedImage, ImageGenerator};
use crate::error::{Result, TensaError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireFormat {
    /// `/chat/completions` with `modalities: ["image", "text"]`.
    OpenRouterChat,
    /// `/images/generations` with `response_format: "b64_json"`.
    OpenAIImages,
}

pub struct OpenRouterImageClient {
    api_key: String,
    model: String,
    base_url: String,
    /// Provider id stamped on the resulting `ImageRecord`.
    provider: String,
    wire: WireFormat,
    client: Client,
}

// ─── /images/generations response ────────────────────────────────

#[derive(Debug, Deserialize)]
struct ImagesResponse {
    #[serde(default)]
    data: Vec<ImageDataItem>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ImageDataItem {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

// ─── /chat/completions (OpenRouter image-output) response ────────
//
// OpenRouter returns the generated image in one of two shapes depending on
// the upstream model:
//
// 1. `message.images[]` — Gemini, FLUX, most image-capable models. Each
//    item carries `{type: "image_url", image_url: {url: "data:image/..."}}`.
//
// 2. `message.content[]` — multimodal-content style. Same item shape but
//    the image is mixed into the content array alongside text parts.
//
// The parser keeps the schema permissive (`serde_json::Value` for content,
// optional fields throughout) so a model that returns yet another shape
// fails with the raw response text in the error rather than a parse error.

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    images: Vec<ChatImage>,
    #[serde(default)]
    content: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ChatImage {
    #[serde(default)]
    image_url: Option<ChatImageUrl>,
}

#[derive(Debug, Deserialize)]
struct ChatImageUrl {
    #[serde(default)]
    url: Option<String>,
}

/// Walk a `message.content` value (string OR array of parts) looking for the
/// first `image_url.url` field. Returns `None` for plain text content.
fn extract_image_from_content(value: &serde_json::Value) -> Option<String> {
    let parts = value.as_array()?;
    for part in parts {
        if let Some(url) = part
            .get("image_url")
            .and_then(|iu| iu.get("url"))
            .and_then(|u| u.as_str())
        {
            return Some(url.to_string());
        }
        if let Some(url) = part.get("url").and_then(|u| u.as_str()) {
            if part
                .get("type")
                .and_then(|t| t.as_str())
                .map(|s| s.contains("image"))
                .unwrap_or(false)
            {
                return Some(url.to_string());
            }
        }
    }
    None
}

impl OpenRouterImageClient {
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url,
            provider: "openrouter".into(),
            wire: WireFormat::OpenRouterChat,
            client: Client::new(),
        }
    }

    pub fn new_openai(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            base_url: "https://api.openai.com/v1".into(),
            provider: "openai".into(),
            wire: WireFormat::OpenAIImages,
            client: Client::new(),
        }
    }

    pub fn new_local(base_url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.unwrap_or_default(),
            model,
            base_url,
            provider: "local".into(),
            wire: WireFormat::OpenAIImages,
            client: Client::new(),
        }
    }

    async fn generate_via_images(&self, prompt: &str) -> Result<GeneratedImage> {
        let url = format!("{}/images/generations", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        });
        let mut req = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("image-gen request failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(TensaError::Internal(format!(
                "image-gen returned {status}: {text}"
            )));
        }
        let parsed: ImagesResponse = resp
            .json()
            .await
            .map_err(|e| TensaError::Internal(format!("image-gen JSON parse: {e}")))?;
        if let Some(err) = parsed.error {
            return Err(TensaError::Internal(format!("image-gen error: {err}")));
        }
        let item = parsed.data.into_iter().next().ok_or_else(|| {
            TensaError::Internal("image-gen returned an empty `data` array".into())
        })?;
        let (bytes, mime) = if let Some(b64) = item.b64_json {
            (
                BASE64.decode(b64.as_bytes()).map_err(|e| {
                    TensaError::Internal(format!("image-gen base64 decode: {e}"))
                })?,
                "image/png".to_string(),
            )
        } else if let Some(remote) = item.url {
            let bytes = self
                .client
                .get(&remote)
                .send()
                .await
                .map_err(|e| TensaError::Internal(format!("image-gen url fetch: {e}")))?
                .bytes()
                .await
                .map_err(|e| TensaError::Internal(format!("image-gen url body: {e}")))?
                .to_vec();
            (bytes, "image/png".to_string())
        } else {
            return Err(TensaError::Internal(
                "image-gen response carried neither b64_json nor url".into(),
            ));
        };
        Ok(GeneratedImage {
            bytes,
            mime,
            provider: self.provider.clone(),
            model: self.model.clone(),
        })
    }

    async fn generate_via_chat(&self, prompt: &str) -> Result<GeneratedImage> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image"],
        });
        let mut req = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("image-gen request failed: {e}")))?;
        let status = resp.status();
        let raw_body = resp
            .text()
            .await
            .map_err(|e| TensaError::Internal(format!("image-gen body read: {e}")))?;
        if !status.is_success() {
            return Err(TensaError::Internal(format!(
                "image-gen returned {status}: {raw_body}"
            )));
        }
        let parsed: ChatResponse = serde_json::from_str(&raw_body).map_err(|e| {
            TensaError::Internal(format!(
                "image-gen JSON parse: {e}; body: {}",
                truncate(&raw_body, 800)
            ))
        })?;
        if let Some(err) = parsed.error {
            return Err(TensaError::Internal(format!("image-gen error: {err}")));
        }
        let mut raw_url: Option<String> = None;
        for choice in parsed.choices {
            if let Some(url) = choice
                .message
                .images
                .into_iter()
                .find_map(|img| img.image_url.and_then(|u| u.url))
            {
                raw_url = Some(url);
                break;
            }
            if let Some(content) = choice.message.content.as_ref() {
                if let Some(url) = extract_image_from_content(content) {
                    raw_url = Some(url);
                    break;
                }
            }
        }
        let raw_url = raw_url.ok_or_else(|| {
            TensaError::Internal(format!(
                "OpenRouter chat response carried no image. The model may not \
                 support image output, the request was filtered, or the response \
                 shape is unexpected. Raw response: {}",
                truncate(&raw_body, 1200)
            ))
        })?;
        let (bytes, mime) = decode_image_url(&self.client, &raw_url).await?;
        Ok(GeneratedImage {
            bytes,
            mime,
            provider: self.provider.clone(),
            model: self.model.clone(),
        })
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push_str("…");
        out
    }
}

#[async_trait]
impl ImageGenerator for OpenRouterImageClient {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn generate(&self, prompt: &str) -> Result<GeneratedImage> {
        match self.wire {
            WireFormat::OpenRouterChat => self.generate_via_chat(prompt).await,
            WireFormat::OpenAIImages => self.generate_via_images(prompt).await,
        }
    }
}

/// Decode either a `data:image/<x>;base64,...` URL or fetch a remote URL.
async fn decode_image_url(client: &Client, url: &str) -> Result<(Vec<u8>, String)> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (mime, payload) = rest.split_once(',').ok_or_else(|| {
            TensaError::Internal("malformed data URL — missing comma".into())
        })?;
        let (mime, encoding) = mime.split_once(';').unwrap_or((mime, "base64"));
        if !encoding.eq_ignore_ascii_case("base64") {
            return Err(TensaError::Internal(format!(
                "unsupported data-URL encoding: {encoding}"
            )));
        }
        let bytes = BASE64.decode(payload.as_bytes()).map_err(|e| {
            TensaError::Internal(format!("image-gen base64 decode: {e}"))
        })?;
        let mime = if mime.is_empty() {
            "image/png".to_string()
        } else {
            mime.to_string()
        };
        Ok((bytes, mime))
    } else {
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("image-gen url fetch: {e}")))?;
        let mime = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "image/png".to_string());
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| TensaError::Internal(format!("image-gen url body: {e}")))?
            .to_vec();
        Ok((bytes, mime))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn decode_image_url_handles_data_url() {
        let client = Client::new();
        let one_pixel_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        let (bytes, mime) = decode_image_url(&client, one_pixel_png).await.unwrap();
        assert_eq!(mime, "image/png");
        assert!(bytes.starts_with(&[0x89, 0x50, 0x4e, 0x47]));
    }

    #[tokio::test]
    async fn decode_image_url_rejects_non_base64_encoding() {
        let client = Client::new();
        let url = "data:image/png;utf8,abc";
        let err = decode_image_url(&client, url).await.unwrap_err();
        assert!(format!("{err}").contains("unsupported data-URL encoding"));
    }

    #[test]
    fn extract_image_from_content_finds_url_in_array() {
        let v = serde_json::json!([
            {"type": "text", "text": "here you go"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]);
        let url = extract_image_from_content(&v).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn extract_image_from_content_returns_none_for_text_only() {
        let v = serde_json::json!([{"type": "text", "text": "no image here"}]);
        assert!(extract_image_from_content(&v).is_none());
    }

    #[test]
    fn extract_image_from_content_handles_string_content() {
        let v = serde_json::Value::String("just text".into());
        assert!(extract_image_from_content(&v).is_none());
    }

    #[test]
    fn truncate_preserves_short_strings() {
        assert_eq!(truncate("hi", 10), "hi");
        assert_eq!(truncate(&"a".repeat(50), 10), format!("{}…", "a".repeat(10)));
    }
}
