//! Types for image storage and generation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Default styles offered by the Studio Generate-Image modal. Users can
/// type any free-form style — these are just the starter buttons.
pub const DEFAULT_IMAGE_STYLES: &[&str] = &[
    "Photorealistic",
    "Anime",
    "Pencil sketch",
    "Film noir",
    "Identikit",
    "Book illustration",
];

/// One image attached to an entity, persisted in the KV store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: Uuid,
    /// Entity that owns the image (typically an Actor).
    pub entity_id: Uuid,
    /// Narrative the image belongs to (mirrors the entity's narrative_id when
    /// available; lets us list images per-narrative for archive export).
    pub narrative_id: Option<String>,
    /// `image/png`, `image/jpeg`, `image/webp`, …
    pub mime: String,
    /// Byte length — kept on the metadata row so listing doesn't need to
    /// load the bytes blob.
    pub bytes_len: u64,
    /// Free-form caption shown under the thumbnail.
    pub caption: Option<String>,
    /// Source of the image: `upload`, `generated`, or a provider-specific tag.
    pub source: ImageSource,
    /// Original prompt for generated images.
    pub prompt: Option<String>,
    /// Style tag for generated images.
    pub style: Option<String>,
    /// Place hint included in the prompt (e.g. "early-19th-century Marseille").
    pub place: Option<String>,
    /// Era hint included in the prompt (e.g. "1815").
    pub era: Option<String>,
    /// Provider that generated the image (`openrouter`, `openai`, `comfyui`, …).
    pub provider: Option<String>,
    /// Model id (provider-specific).
    pub model: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// How the image arrived in TENSA.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageSource {
    #[serde(rename = "upload")]
    Upload,
    #[serde(rename = "generated")]
    Generated,
}

/// Persisted provider configuration (`cfg/image_gen`).
///
/// Mirrors the shape of `LlmConfig`: a tagged enum keyed on `provider`. We
/// keep it in its own module so the existing `LlmConfig` machinery (extractor
/// builders, fallback chains, etc.) doesn't get tangled with image plumbing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum ImageGenConfig {
    /// OpenRouter image-capable models (uses the OpenAI-style
    /// `/api/v1/images/generations` proxy).
    #[serde(rename = "openrouter")]
    OpenRouter { api_key: String, model: String },
    /// OpenAI direct (`/v1/images/generations`, e.g. `gpt-image-1`,
    /// `dall-e-3`). Same JSON shape as OpenRouter.
    #[serde(rename = "openai")]
    OpenAI { api_key: String, model: String },
    /// Local self-hosted endpoint speaking the OpenAI image-generation API
    /// shape (vLLM image, LiteLLM, koboldcpp, …).
    #[serde(rename = "local")]
    Local {
        base_url: String,
        model: String,
        api_key: Option<String>,
    },
    /// Local ComfyUI (`/prompt` workflow). The workflow JSON is supplied
    /// per-request — keep this scoped to "model id used in the workflow".
    #[serde(rename = "comfyui")]
    ComfyUI {
        base_url: String,
        /// Optional default workflow id; the front-end can override.
        workflow: Option<String>,
    },
    /// No image generator configured.
    #[serde(rename = "none")]
    None,
}

impl Default for ImageGenConfig {
    fn default() -> Self {
        Self::None
    }
}

impl ImageGenConfig {
    /// Redact secrets before sending the config to the Studio settings page.
    pub fn redact(&self) -> Self {
        match self {
            Self::OpenRouter { model, .. } => Self::OpenRouter {
                api_key: String::new(),
                model: model.clone(),
            },
            Self::OpenAI { model, .. } => Self::OpenAI {
                api_key: String::new(),
                model: model.clone(),
            },
            Self::Local {
                base_url,
                model,
                api_key,
            } => Self::Local {
                base_url: base_url.clone(),
                model: model.clone(),
                api_key: api_key.as_ref().map(|_| String::new()),
            },
            Self::ComfyUI { base_url, workflow } => Self::ComfyUI {
                base_url: base_url.clone(),
                workflow: workflow.clone(),
            },
            Self::None => Self::None,
        }
    }

    /// Preserve the existing API key when an incoming PUT carries an empty
    /// or omitted secret (Studio reads back `redact()`ed configs and re-saves
    /// them — without this, every Settings save would wipe the stored key).
    /// Mirrors `LlmConfig::merge_keys`.
    pub fn merge_keys(self, existing: &Self) -> Self {
        match (self, existing) {
            (
                Self::OpenRouter {
                    api_key,
                    model,
                },
                Self::OpenRouter {
                    api_key: prev_key,
                    ..
                },
            ) if api_key.is_empty() => Self::OpenRouter {
                api_key: prev_key.clone(),
                model,
            },
            (
                Self::OpenAI { api_key, model },
                Self::OpenAI {
                    api_key: prev_key, ..
                },
            ) if api_key.is_empty() => Self::OpenAI {
                api_key: prev_key.clone(),
                model,
            },
            (
                Self::Local {
                    base_url,
                    model,
                    api_key,
                },
                Self::Local {
                    api_key: prev_key, ..
                },
            ) if api_key.as_deref().map(|s| s.is_empty()).unwrap_or(true) => Self::Local {
                base_url,
                model,
                api_key: prev_key.clone(),
            },
            (incoming, _) => incoming,
        }
    }
}

/// Body for `POST /images/generate`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenRequest {
    pub entity_id: Uuid,
    pub prompt: String,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub place: Option<String>,
    #[serde(default)]
    pub era: Option<String>,
    /// Optional override of the persisted model.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional caption for the generated image.
    #[serde(default)]
    pub caption: Option<String>,
}

/// One of the configured starter styles, plus any user-typed string.
pub type ImageStyle = String;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_keys_preserves_existing_when_incoming_is_redacted() {
        let existing = ImageGenConfig::OpenRouter {
            api_key: "sk-or-real".into(),
            model: "old".into(),
        };
        let incoming = ImageGenConfig::OpenRouter {
            api_key: String::new(),
            model: "new".into(),
        };
        let merged = incoming.merge_keys(&existing);
        match merged {
            ImageGenConfig::OpenRouter { api_key, model } => {
                assert_eq!(api_key, "sk-or-real");
                assert_eq!(model, "new");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn merge_keys_takes_incoming_when_provider_changes() {
        let existing = ImageGenConfig::OpenRouter {
            api_key: "sk-or-real".into(),
            model: "old".into(),
        };
        let incoming = ImageGenConfig::OpenAI {
            api_key: "sk-new".into(),
            model: "gpt-image-1".into(),
        };
        match incoming.merge_keys(&existing) {
            ImageGenConfig::OpenAI { api_key, .. } => assert_eq!(api_key, "sk-new"),
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn redact_blanks_secrets() {
        let cfg = ImageGenConfig::OpenAI {
            api_key: "sk-real".into(),
            model: "gpt-image-1".into(),
        };
        match cfg.redact() {
            ImageGenConfig::OpenAI { api_key, model } => {
                assert!(api_key.is_empty());
                assert_eq!(model, "gpt-image-1");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }
}
