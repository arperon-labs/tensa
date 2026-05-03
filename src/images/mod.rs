//! Image storage + generation for actor portraits (and future entity media).
//!
//! Images are split across two KV prefixes so the bytes can be migrated to
//! disk later without touching metadata callers:
//!
//! * `img/m/{narrative_id}/{image_id_v7}` — `ImageRecord` metadata (JSON)
//! * `img/b/{image_id_v7}`                — raw bytes (no JSON wrap)
//!
//! Provider config persists at `cfg/image_gen` mirroring the LLM config slots.

pub mod openrouter;
pub mod provider;
pub mod storage;
pub mod types;

pub use provider::{build_image_generator, ImageGenerator};
pub use storage::{
    delete_image, image_bytes_key, image_index_key, image_meta_key, list_entity_images,
    list_narrative_images, load_image_by_id, load_image_bytes, load_image_meta,
    resolve_narrative_for_image, save_image,
};
pub use types::{
    ImageGenConfig, ImageGenRequest, ImageRecord, ImageStyle, DEFAULT_IMAGE_STYLES,
};

/// KV prefix for image metadata records.
pub const IMG_META_PREFIX: &[u8] = b"img/m/";
/// KV prefix for raw image bytes.
pub const IMG_BYTES_PREFIX: &[u8] = b"img/b/";
/// KV prefix for the global image-id → narrative-id index, so a fetch-by-id
/// can locate the full metadata row in O(1) without a full prefix scan.
pub const IMG_INDEX_PREFIX: &[u8] = b"img/i/";
