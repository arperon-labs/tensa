//! REST routes for image upload, retrieval, generation, and provider config.
//!
//! All file bytes flow as base64 inside JSON to avoid adding a multipart
//! parser dependency. The Studio side reads `<input type="file">` via
//! `FileReader.readAsDataURL` and sends `data: <base64>` in the body.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::api::settings_routes::{persist_config, CFG_IMAGE_GEN_KEY};
use crate::error::TensaError;
use crate::images::types::{ImageSource, DEFAULT_IMAGE_STYLES};
use crate::images::{
    build_image_generator, delete_image as kv_delete_image, list_entity_images, load_image_by_id,
    load_image_bytes, save_image, ImageGenConfig, ImageGenRequest, ImageRecord,
};

const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024; // 32 MB cap (covers small PDFs / docs)

// ─── Settings: GET / PUT /settings/image-gen ──────────────────

pub async fn get_image_gen_config(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cfg = state
        .image_gen_config
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();
    json_ok(&cfg.redact())
}

pub async fn put_image_gen_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ImageGenConfig>,
) -> impl IntoResponse {
    let merged = {
        let current = state
            .image_gen_config
            .read()
            .map(|g| g.clone())
            .unwrap_or_default();
        body.merge_keys(&current)
    };
    persist_config(state.root_store.as_ref(), CFG_IMAGE_GEN_KEY, &merged, "image-gen");

    let new_gen = build_image_generator(&merged);
    if let Ok(mut slot) = state.image_gen_config.write() {
        *slot = merged.clone();
    }
    if let Ok(mut slot) = state.image_generator.write() {
        *slot = new_gen;
    }
    json_ok(&merged.redact())
}

pub async fn get_image_styles() -> impl IntoResponse {
    let styles: Vec<&str> = DEFAULT_IMAGE_STYLES.to_vec();
    json_ok(&serde_json::json!({ "styles": styles }))
}

// ─── Upload: POST /entities/:id/images ────────────────────────

#[derive(Debug, Deserialize)]
pub struct ImageUploadRequest {
    /// `image/png`, `image/jpeg`, …
    pub mime: String,
    /// Base64-encoded bytes (no `data:` URI prefix).
    pub data: String,
    #[serde(default)]
    pub caption: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageUploadResponse {
    pub id: Uuid,
    pub bytes_len: u64,
    pub url: String,
}

pub async fn upload_entity_image(
    State(state): State<Arc<AppState>>,
    Path(entity_id): Path<Uuid>,
    Json(body): Json<ImageUploadRequest>,
) -> impl IntoResponse {
    // Strip a `data:image/png;base64,` prefix if the front-end forgot to.
    let raw = body
        .data
        .split_once(',')
        .map(|(_head, tail)| tail.trim())
        .unwrap_or_else(|| body.data.trim());
    let bytes = match BASE64.decode(raw.as_bytes()) {
        Ok(b) => b,
        Err(e) => {
            return error_response(TensaError::InvalidInput(format!(
                "image base64 decode failed: {e}"
            )))
            .into_response()
        }
    };
    if bytes.is_empty() {
        return error_response(TensaError::InvalidInput("empty image bytes".into()))
            .into_response();
    }
    if bytes.len() > MAX_IMAGE_BYTES {
        return error_response(TensaError::InvalidInput(format!(
            "image too large: {} bytes (max {})",
            bytes.len(),
            MAX_IMAGE_BYTES
        )))
        .into_response();
    }

    let entity = match state.hypergraph.get_entity(&entity_id) {
        Ok(e) => e,
        Err(e) => return error_response(e).into_response(),
    };
    let narrative_id = entity.narrative_id.clone();

    let record = ImageRecord {
        id: Uuid::now_v7(),
        entity_id,
        narrative_id,
        mime: body.mime.clone(),
        bytes_len: bytes.len() as u64,
        caption: body.caption.clone(),
        source: ImageSource::Upload,
        prompt: None,
        style: None,
        place: None,
        era: None,
        provider: None,
        model: None,
        created_at: Utc::now(),
    };

    if let Err(e) = save_image(state.root_store.as_ref(), &record, &bytes) {
        return error_response(e).into_response();
    }
    if let Err(e) = attach_image_to_entity(&state, &record) {
        // Don't fail the upload — the image bytes already landed.
        tracing::warn!("attach_image_to_entity: {e}");
    }

    let resp = ImageUploadResponse {
        id: record.id,
        bytes_len: record.bytes_len,
        url: format!("/images/{}", record.id),
    };
    json_ok(&resp)
}

// ─── List: GET /entities/:id/images ───────────────────────────

pub async fn list_entity_images_route(
    State(state): State<Arc<AppState>>,
    Path(entity_id): Path<Uuid>,
) -> impl IntoResponse {
    let entity = match state.hypergraph.get_entity(&entity_id) {
        Ok(e) => e,
        Err(e) => return error_response(e).into_response(),
    };
    let nid = entity.narrative_id.as_deref().unwrap_or("");
    match list_entity_images(state.root_store.as_ref(), nid, &entity_id) {
        Ok(records) => {
            let with_url: Vec<_> = records
                .into_iter()
                .map(|r| {
                    let url = format!("/images/{}", r.id);
                    ImageWithUrl { record: r, url }
                })
                .collect();
            json_ok(&with_url)
        }
        Err(e) => error_response(e).into_response(),
    }
}

#[derive(Debug, Serialize)]
struct ImageWithUrl {
    #[serde(flatten)]
    record: ImageRecord,
    url: String,
}

// ─── Fetch raw bytes: GET /images/:id ─────────────────────────

#[derive(Debug, Deserialize)]
pub struct ImageFetchQuery {
    /// Optional narrative_id to scope the metadata lookup. When omitted we
    /// just stream bytes — the bytes key is global to the store.
    #[serde(default)]
    pub narrative_id: Option<String>,
}

pub async fn fetch_image(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<Uuid>,
    Query(_q): Query<ImageFetchQuery>,
) -> impl IntoResponse {
    let bytes = match load_image_bytes(state.root_store.as_ref(), &image_id) {
        Ok(Some(b)) => b,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!("image {image_id}")))
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };

    let mime = load_image_by_id(state.root_store.as_ref(), &image_id)
        .ok()
        .flatten()
        .map(|r| r.mime)
        .unwrap_or_else(|| "image/png".into());

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&mime).unwrap_or_else(|_| HeaderValue::from_static("image/png")),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=300"),
    );

    (StatusCode::OK, headers, bytes).into_response()
}

// ─── Delete: DELETE /images/:id ───────────────────────────────

pub async fn delete_image_route(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<Uuid>,
) -> impl IntoResponse {
    let record = match load_image_by_id(state.root_store.as_ref(), &image_id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return error_response(TensaError::NotFound(format!("image {image_id}")))
                .into_response()
        }
        Err(e) => return error_response(e).into_response(),
    };
    let nid = record.narrative_id.as_deref().unwrap_or("");
    if let Err(e) = kv_delete_image(state.root_store.as_ref(), nid, &image_id) {
        return error_response(e).into_response();
    }
    if let Err(e) = detach_image_from_entity(&state, &record.entity_id, &image_id) {
        tracing::warn!("detach_image_from_entity: {e}");
    }
    json_ok(&serde_json::json!({"deleted": image_id.to_string()}))
}

// ─── Generate: POST /images/generate ──────────────────────────

pub async fn generate_image(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ImageGenRequest>,
) -> impl IntoResponse {
    let entity = match state.hypergraph.get_entity(&body.entity_id) {
        Ok(e) => e,
        Err(e) => return error_response(e).into_response(),
    };
    let narrative_id = entity.narrative_id.clone();

    let generator = match state.image_generator.read().ok().and_then(|g| g.clone()) {
        Some(g) => g,
        None => {
            return error_response(TensaError::InvalidInput(
                "no image generator configured (set provider via /settings/image-gen)".into(),
            ))
            .into_response()
        }
    };

    let img = match generator.generate(&body.prompt).await {
        Ok(img) => img,
        Err(e) => return error_response(e).into_response(),
    };

    let record = ImageRecord {
        id: Uuid::now_v7(),
        entity_id: body.entity_id,
        narrative_id,
        mime: img.mime.clone(),
        bytes_len: img.bytes.len() as u64,
        caption: body.caption,
        source: ImageSource::Generated,
        prompt: Some(body.prompt),
        style: body.style,
        place: body.place,
        era: body.era,
        provider: Some(img.provider),
        model: Some(body.model.unwrap_or(img.model)),
        created_at: Utc::now(),
    };
    if let Err(e) = save_image(state.root_store.as_ref(), &record, &img.bytes) {
        return error_response(e).into_response();
    }
    if let Err(e) = attach_image_to_entity(&state, &record) {
        tracing::warn!("attach_image_to_entity: {e}");
    }

    json_ok(&serde_json::json!({
        "id": record.id,
        "bytes_len": record.bytes_len,
        "url": format!("/images/{}", record.id),
        "provider": record.provider,
        "model": record.model,
    }))
}

// ─── Helpers: keep `entity.properties.media` in sync ──────────

fn attach_image_to_entity(state: &AppState, record: &ImageRecord) -> crate::error::Result<()> {
    state.hypergraph.update_entity_no_snapshot(&record.entity_id, |e| {
        let media = e
            .properties
            .as_object_mut()
            .map(|o| {
                o.entry("media")
                    .or_insert_with(|| serde_json::Value::Array(Vec::new()))
            });
        if let Some(serde_json::Value::Array(arr)) = media {
            arr.push(serde_json::json!({
                "id": record.id,
                "url": format!("/images/{}", record.id),
                "type": "image",
                "mime": record.mime,
                "caption": record.caption,
                "added_at": record.created_at,
                "source": serde_json::to_value(record.source).ok(),
                "prompt": record.prompt,
                "style": record.style,
                "place": record.place,
                "era": record.era,
                "provider": record.provider,
                "model": record.model,
            }));
        }
    })?;
    Ok(())
}

fn detach_image_from_entity(
    state: &AppState,
    entity_id: &Uuid,
    image_id: &Uuid,
) -> crate::error::Result<()> {
    state.hypergraph.update_entity_no_snapshot(entity_id, |e| {
        if let Some(arr) = e.properties.get_mut("media").and_then(|v| v.as_array_mut()) {
            arr.retain(|item| {
                item.get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s != image_id.to_string())
                    .unwrap_or(true)
            });
        }
    })?;
    Ok(())
}

