//! `ImageGenerator` trait + provider dispatch from `ImageGenConfig`.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;

use super::openrouter::OpenRouterImageClient;
use super::types::ImageGenConfig;

/// Output of a single generation call.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub mime: String,
    pub provider: String,
    pub model: String,
}

/// Pluggable image-generation backend. All providers speak this trait so
/// the route handler is provider-agnostic.
#[async_trait]
pub trait ImageGenerator: Send + Sync {
    /// Provider id (`"openrouter"`, `"openai"`, `"local"`, `"comfyui"`).
    fn provider(&self) -> &str;
    /// Model id, for record-keeping.
    fn model(&self) -> &str;
    /// Generate a single image from `prompt`. Returns raw bytes + mime.
    async fn generate(&self, prompt: &str) -> Result<GeneratedImage>;
}

/// Build an `ImageGenerator` from persisted config. Returns `None` for
/// `ImageGenConfig::None` or for providers we don't have a runtime client for.
pub fn build_image_generator(config: &ImageGenConfig) -> Option<Arc<dyn ImageGenerator>> {
    match config {
        ImageGenConfig::OpenRouter { api_key, model } => {
            if api_key.is_empty() {
                return None;
            }
            Some(Arc::new(OpenRouterImageClient::new(
                api_key.clone(),
                model.clone(),
                "https://openrouter.ai/api/v1".into(),
            )))
        }
        ImageGenConfig::OpenAI { api_key, model } => {
            if api_key.is_empty() {
                return None;
            }
            Some(Arc::new(OpenRouterImageClient::new_openai(
                api_key.clone(),
                model.clone(),
            )))
        }
        ImageGenConfig::Local {
            base_url,
            model,
            api_key,
        } => Some(Arc::new(OpenRouterImageClient::new_local(
            base_url.clone(),
            model.clone(),
            api_key.clone(),
        ))),
        // ComfyUI client is not yet implemented; config is accepted and
        // persisted but `build_image_generator` returns `None` until the
        // client lands.
        ImageGenConfig::ComfyUI { .. } => None,
        ImageGenConfig::None => None,
    }
}
