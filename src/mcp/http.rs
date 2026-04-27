//! HTTP backend — calls the TENSA REST API via reqwest.
//!
//! Each trait method maps to the corresponding REST endpoint.
//! The TENSA REST server must be running separately.

use serde_json::Value;

use crate::error::{Result, TensaError};
use crate::types::{EntityType, ALL_ENTITY_TYPES};

use super::backend::McpBackend;

/// Append `?tnorm=<kind>&aggregator=<kind>` to an existing REST path so
/// Phase 11's fuzzy opt-in rides through to the server unchanged. When
/// both are `None` the input path is returned verbatim (pre-sprint URL
/// bit-identity — the backward-compat test-assert contract).
///
/// Cites: [klement2000] for the t-norm/aggregator taxonomy these knobs
/// select among.
pub(crate) fn append_fuzzy_query(
    path: &str,
    tnorm: Option<&str>,
    aggregator: Option<&str>,
) -> String {
    let t = tnorm.map(str::trim).filter(|s| !s.is_empty());
    let a = aggregator.map(str::trim).filter(|s| !s.is_empty());
    if t.is_none() && a.is_none() {
        return path.to_string();
    }
    let sep = if path.contains('?') { '&' } else { '?' };
    let mut out = String::with_capacity(path.len() + 64);
    out.push_str(path);
    out.push(sep);
    let mut need_amp = false;
    if let Some(kind) = t {
        out.push_str("tnorm=");
        out.push_str(kind);
        need_amp = true;
    }
    if let Some(kind) = a {
        if need_amp {
            out.push('&');
        }
        out.push_str("aggregator=");
        out.push_str(kind);
    }
    out
}

/// Build the edit operation payload used by both `propose_edit` and
/// `estimate_edit_tokens`. `style_preset = Some("X")` → style transfer targeting
/// preset X; otherwise a free-form rewrite driven by `instruction`.
fn build_edit_op(instruction: &str, style_preset: Option<&str>) -> Value {
    match style_preset {
        Some(name) => serde_json::json!({
            "kind": "style_transfer",
            "target": { "kind": "preset", "name": name },
        }),
        None => serde_json::json!({
            "kind": "rewrite",
            "instruction": instruction,
        }),
    }
}

/// HTTP client backend that proxies MCP tool calls to the TENSA REST API.
#[derive(Clone)]
pub struct HttpBackend {
    client: reqwest::Client,
    base_url: String,
}

impl HttpBackend {
    /// Create a new HTTP backend pointing at the given TENSA server URL.
    pub fn new(base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn request_json(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value> {
        let mut req = self.client.request(method, self.url(path));
        if let Some(b) = body {
            req = req.json(b);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("HTTP request failed: {}", e)))?;

        let status = resp.status();
        if status == reqwest::StatusCode::NO_CONTENT {
            return Ok(serde_json::json!({"status": "ok"}));
        }

        // Buffer the raw body first so we can salvage non-JSON error payloads
        // (axum returns plain-text 422 bodies when JSON deserialization of the
        // request body fails — e.g. a client sending a nested object where the
        // schema expects a string). Without this, the caller saw the opaque
        // "Failed to parse response" error instead of the real field-level
        // complaint.
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| TensaError::Internal(format!("HTTP body read failed: {}", e)))?;

        if status.is_success() {
            return serde_json::from_slice::<Value>(&bytes)
                .map_err(|e| TensaError::Internal(format!("Failed to parse response: {}", e)));
        }

        // Error path. Try JSON-shaped {"error": "..."} first, then fall back
        // to plain-text bodies (axum validation errors, nginx pages, etc.).
        let text = String::from_utf8_lossy(&bytes);
        let parsed: Option<Value> = serde_json::from_slice(&bytes).ok();
        let msg = parsed
            .as_ref()
            .and_then(|v| v.get("error"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    "<empty response body>".to_string()
                } else {
                    trimmed.to_string()
                }
            });

        if status.is_client_error() {
            // 4xx → structured invalid-input so MCP clients surface the field.
            Err(TensaError::InvalidInput(format!("{}: {}", status, msg)))
        } else {
            Err(TensaError::Internal(format!(
                "API error ({}): {}",
                status, msg
            )))
        }
    }

    async fn get_json(&self, path: &str) -> Result<Value> {
        self.request_json(reqwest::Method::GET, path, None).await
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        self.request_json(reqwest::Method::POST, path, Some(body))
            .await
    }

    async fn put_json(&self, path: &str, body: &Value) -> Result<Value> {
        self.request_json(reqwest::Method::PUT, path, Some(body))
            .await
    }

    async fn delete(&self, path: &str) -> Result<Value> {
        let resp = self
            .client
            .delete(self.url(path))
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("HTTP request failed: {}", e)))?;

        let status = resp.status();
        if status.is_success() {
            Ok(serde_json::json!({"status": "ok"}))
        } else {
            let body: Value = resp.json().await.unwrap_or(serde_json::json!({}));
            let msg = body["error"]
                .as_str()
                .unwrap_or("Unknown error")
                .to_string();
            Err(TensaError::Internal(format!(
                "API error ({}): {}",
                status, msg
            )))
        }
    }
}

impl McpBackend for HttpBackend {
    async fn execute_query(&self, tensaql: &str) -> Result<Value> {
        self.post_json("/query", &serde_json::json!({"query": tensaql}))
            .await
    }

    async fn submit_inference_query(&self, tensaql: &str) -> Result<Value> {
        // `/infer` parses + plans + executes + submits to the inference queue so
        // the return shape matches the embedded backend
        // (`{job_id, status, message}`). Historically this hit `/query`, which
        // only ran the executor and returned the descriptor row without
        // submitting — the caller silently got back a plan and no job_id.
        self.post_json("/infer", &serde_json::json!({"query": tensaql}))
            .await
    }

    async fn create_entity(&self, data: Value) -> Result<Value> {
        self.post_json("/entities", &data).await
    }

    async fn get_entity(&self, id: &str) -> Result<Value> {
        self.get_json(&format!("/entities/{}", id)).await
    }

    async fn create_situation(&self, data: Value) -> Result<Value> {
        self.post_json("/situations", &data).await
    }

    async fn get_situation(&self, id: &str) -> Result<Value> {
        self.get_json(&format!("/situations/{}", id)).await
    }

    async fn add_participant(&self, data: Value) -> Result<Value> {
        self.post_json("/participations", &data).await
    }

    async fn ingest_text(
        &self,
        text: &str,
        narrative_id: &str,
        source: &str,
        auto_commit_threshold: Option<f32>,
        review_threshold: Option<f32>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "text": text,
            "narrative_id": narrative_id,
            "source_name": source,
            "auto_commit_threshold": auto_commit_threshold,
            "review_threshold": review_threshold,
        });
        self.post_json("/ingest", &body).await
    }

    async fn list_narratives(&self) -> Result<Value> {
        let resp = self.get_json("/narratives").await?;
        // Extract data array from paginated response wrapper
        Ok(resp.get("data").cloned().unwrap_or(resp))
    }

    async fn create_narrative(&self, data: Value) -> Result<Value> {
        self.post_json("/narratives", &data).await
    }

    async fn get_job_status(&self, job_id: &str) -> Result<Value> {
        self.get_json(&format!("/jobs/{}", job_id)).await
    }

    async fn get_job_result(&self, job_id: &str) -> Result<Value> {
        self.get_json(&format!("/jobs/{}/result", job_id)).await
    }

    async fn create_source(&self, data: Value) -> Result<Value> {
        self.post_json("/sources", &data).await
    }

    async fn get_source(&self, id: &str) -> Result<Value> {
        self.get_json(&format!("/sources/{}", id)).await
    }

    async fn list_sources(&self) -> Result<Value> {
        self.get_json("/sources").await
    }

    async fn add_attribution(&self, data: Value) -> Result<Value> {
        let source_id = data["source_id"]
            .as_str()
            .ok_or_else(|| TensaError::InvalidQuery("source_id required".into()))?;
        self.post_json(&format!("/sources/{}/attributions", source_id), &data)
            .await
    }

    async fn list_contentions(&self, situation_id: &str) -> Result<Value> {
        self.get_json(&format!("/situations/{}/contentions", situation_id))
            .await
    }

    async fn recompute_confidence(&self, id: &str) -> Result<Value> {
        // Try entity first, fall back to situation
        let result = self
            .post_json(
                &format!("/entities/{}/recompute-confidence", id),
                &serde_json::json!({}),
            )
            .await;
        if result.is_ok() {
            return result;
        }
        self.post_json(
            &format!("/situations/{}/recompute-confidence", id),
            &serde_json::json!({}),
        )
        .await
    }

    async fn review_queue(
        &self,
        action: &str,
        item_id: Option<&str>,
        reviewer: Option<&str>,
        notes: Option<&str>,
        edited_data: Option<Value>,
        limit: Option<usize>,
    ) -> Result<Value> {
        match action {
            "list" => {
                let limit = limit.unwrap_or(50);
                self.get_json(&format!("/validation-queue?limit={}", limit))
                    .await
            }
            "get" => {
                let id = item_id
                    .ok_or_else(|| TensaError::InvalidQuery("item_id required for 'get'".into()))?;
                self.get_json(&format!("/validation-queue/{}", id)).await
            }
            "approve" => {
                let id = item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'approve'".into())
                })?;
                let reviewer = reviewer.ok_or_else(|| {
                    TensaError::InvalidQuery("reviewer required for 'approve'".into())
                })?;
                self.post_json(
                    &format!("/validation-queue/{}/approve", id),
                    &serde_json::json!({"reviewer": reviewer}),
                )
                .await
            }
            "reject" => {
                let id = item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'reject'".into())
                })?;
                let reviewer = reviewer.ok_or_else(|| {
                    TensaError::InvalidQuery("reviewer required for 'reject'".into())
                })?;
                self.post_json(
                    &format!("/validation-queue/{}/reject", id),
                    &serde_json::json!({"reviewer": reviewer, "notes": notes}),
                )
                .await
            }
            "edit" => {
                let id = item_id.ok_or_else(|| {
                    TensaError::InvalidQuery("item_id required for 'edit'".into())
                })?;
                let reviewer = reviewer.ok_or_else(|| {
                    TensaError::InvalidQuery("reviewer required for 'edit'".into())
                })?;
                let data = edited_data.ok_or_else(|| {
                    TensaError::InvalidQuery("edited_data required for 'edit'".into())
                })?;
                self.post_json(
                    &format!("/validation-queue/{}/edit", id),
                    &serde_json::json!({"reviewer": reviewer, "edited_data": data}),
                )
                .await
            }
            other => Err(TensaError::InvalidQuery(format!(
                "Unknown review action: '{}'. Use list/get/approve/reject/edit.",
                other
            ))),
        }
    }

    async fn delete_entity(&self, id: &str) -> Result<Value> {
        self.delete(&format!("/entities/{}", id)).await
    }

    async fn delete_situation(&self, id: &str) -> Result<Value> {
        self.delete(&format!("/situations/{}", id)).await
    }

    async fn update_entity(&self, id: &str, updates: Value) -> Result<Value> {
        self.put_json(&format!("/entities/{}", id), &updates).await
    }

    async fn list_entities(
        &self,
        entity_type: Option<&str>,
        narrative_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Value> {
        // Validate and resolve entity types
        let types: Vec<String> = if let Some(et) = entity_type {
            let _: EntityType = et.parse()?; // validate
            vec![et.to_string()]
        } else {
            ALL_ENTITY_TYPES
                .iter()
                .map(|t| t.as_index_str().to_string())
                .collect()
        };

        let mut all_results = Vec::new();
        for t in &types {
            if let Some(lim) = limit {
                if all_results.len() >= lim {
                    break;
                }
            }
            let mut q = format!("MATCH (e:{}) ", t);
            if let Some(nar) = narrative_id {
                q.push_str(&format!(
                    r#"ACROSS NARRATIVES ("{}") "#,
                    nar.replace('"', "")
                ));
            }
            q.push_str("RETURN e");
            if let Some(lim) = limit {
                let remaining = lim.saturating_sub(all_results.len());
                q.push_str(&format!(" LIMIT {}", remaining));
            }
            let result = self
                .post_json("/query", &serde_json::json!({"query": q}))
                .await?;
            if let Some(arr) = result.as_array() {
                all_results.extend(arr.iter().cloned());
            }
        }
        if let Some(lim) = limit {
            all_results.truncate(lim);
        }
        Ok(serde_json::Value::Array(all_results))
    }

    async fn merge_entities(&self, keep_id: &str, absorb_id: &str) -> Result<Value> {
        self.post_json(
            "/entities/merge",
            &serde_json::json!({"keep_id": keep_id, "absorb_id": absorb_id}),
        )
        .await
    }

    async fn export_narrative(&self, narrative_id: &str, format: &str) -> Result<Value> {
        let resp = self
            .client
            .get(self.url(&format!(
                "/narratives/{}/export?format={}",
                narrative_id, format
            )))
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("HTTP request failed: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let body: Value = resp.json().await.unwrap_or(serde_json::json!({}));
            let msg = body["error"]
                .as_str()
                .unwrap_or("Unknown error")
                .to_string();
            return Err(TensaError::Internal(format!(
                "API error ({}): {}",
                status, msg
            )));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| TensaError::Internal(format!("Failed to read body: {}", e)))?;
        let body_str =
            String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| "<binary content>".to_string());

        Ok(serde_json::json!({
            "content_type": content_type,
            "body": body_str,
        }))
    }

    async fn get_narrative_stats(
        &self,
        narrative_id: &str,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
    ) -> Result<Value> {
        // Fuzzy-opt-in: when either knob is set, append to the query
        // string. Both-None preserves the pre-Phase-11 URL bit-identically.
        let path = format!("/narratives/{}/stats", narrative_id);
        let path = append_fuzzy_query(&path, tnorm, aggregator);
        self.get_json(&path).await
    }

    async fn search_entities(
        &self,
        query: &str,
        limit: Option<usize>,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
    ) -> Result<Value> {
        let lim = limit.unwrap_or(20);
        let escaped = query.replace('"', r#"\""#);
        let mut all_results = Vec::new();
        for et in &ALL_ENTITY_TYPES {
            if all_results.len() >= lim {
                break;
            }
            let remaining = lim - all_results.len();
            let q = format!(
                r#"MATCH (e:{}) WHERE e.properties CONTAINS "{}" RETURN e LIMIT {}"#,
                et.as_index_str(),
                escaped,
                remaining
            );
            // Fuzzy-opt-in: when knobs are set, append `?tnorm=&aggregator=`
            // to the POST path. REST `/query` currently ignores them but
            // the wire contract is now Phase-11-ready.
            let path = append_fuzzy_query("/query", tnorm, aggregator);
            let result = self
                .post_json(&path, &serde_json::json!({"query": q}))
                .await?;
            if let Some(arr) = result.as_array() {
                all_results.extend(arr.iter().cloned());
            }
        }
        all_results.truncate(lim);
        Ok(serde_json::Value::Array(all_results))
    }

    async fn ingest_url(
        &self,
        url: &str,
        narrative_id: &str,
        source_name: Option<&str>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "url": url,
            "narrative_id": narrative_id,
            "source_name": source_name.unwrap_or("mcp-url"),
        });
        self.post_json("/ingest/url", &body).await
    }

    async fn ingest_rss(
        &self,
        feed_url: &str,
        narrative_id: &str,
        max_items: Option<usize>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "feed_url": feed_url,
            "narrative_id": narrative_id,
            "max_items": max_items.unwrap_or(10),
        });
        self.post_json("/ingest/rss", &body).await
    }

    async fn get_actor_profile(&self, actor_id: &str) -> Result<Value> {
        let entity = self.get_json(&format!("/entities/{}", actor_id)).await?;
        let situations = self
            .get_json(&format!("/entities/{}/situations", actor_id))
            .await?;

        Ok(serde_json::json!({
            "entity": entity,
            "participations": situations,
            "state_history": [],
            "participation_count": situations.as_array().map(|a| a.len()).unwrap_or(0),
        }))
    }

    async fn split_entity(&self, entity_id: &str, situation_ids: &[String]) -> Result<Value> {
        let body = serde_json::json!({"situation_ids": situation_ids});
        self.post_json(&format!("/entities/{}/split", entity_id), &body)
            .await
    }

    async fn restore_entity(&self, id: &str) -> Result<Value> {
        self.post_json(&format!("/entities/{}/restore", id), &serde_json::json!({}))
            .await
    }

    async fn restore_situation(&self, id: &str) -> Result<Value> {
        self.post_json(
            &format!("/situations/{}/restore", id),
            &serde_json::json!({}),
        )
        .await
    }

    async fn create_project(&self, data: Value) -> Result<Value> {
        self.post_json("/projects", &data).await
    }

    async fn get_project(&self, id: &str) -> Result<Value> {
        self.get_json(&format!("/projects/{}", id)).await
    }

    async fn list_projects(&self, limit: Option<usize>) -> Result<Value> {
        let path = match limit {
            Some(lim) => format!("/projects?limit={}", lim),
            None => "/projects".to_string(),
        };
        let resp = self.get_json(&path).await?;
        // Extract data array from paginated response wrapper
        Ok(resp.get("data").cloned().unwrap_or(resp))
    }

    async fn update_project(&self, id: &str, updates: Value) -> Result<Value> {
        self.put_json(&format!("/projects/{}", id), &updates).await
    }

    async fn delete_project(&self, id: &str, cascade: bool) -> Result<Value> {
        let path = if cascade {
            format!("/projects/{}?cascade=true", id)
        } else {
            format!("/projects/{}", id)
        };
        self.delete(&path).await
    }

    async fn cache_stats(&self) -> Result<Value> {
        self.get_json("/cache/stats").await
    }

    async fn ask(
        &self,
        question: &str,
        narrative_id: Option<&str>,
        mode: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::json!({"question": question});
        if let Some(nid) = narrative_id {
            body["narrative_id"] = serde_json::Value::String(nid.to_string());
        }
        if let Some(m) = mode {
            body["mode"] = serde_json::Value::String(m.to_string());
        }
        self.post_json("/ask", &body).await
    }

    async fn tune_prompts(&self, narrative_id: &str) -> Result<Value> {
        let body = serde_json::json!({"narrative_id": narrative_id});
        self.post_json("/prompts/tune", &body).await
    }

    async fn community_hierarchy(&self, narrative_id: &str, level: Option<usize>) -> Result<Value> {
        let url = if let Some(lvl) = level {
            format!("/narratives/{}/communities?level={}", narrative_id, lvl)
        } else {
            format!("/narratives/{}/communities/hierarchy", narrative_id)
        };
        self.get_json(&url).await
    }

    async fn export_archive(&self, narrative_ids: Vec<String>) -> Result<Value> {
        self.post_json(
            "/export/archive",
            &serde_json::json!({"narrative_ids": narrative_ids}),
        )
        .await
    }

    async fn import_archive(&self, data_base64: &str) -> Result<Value> {
        // HTTP backend sends raw bytes, not base64
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_base64)
            .map_err(|e| TensaError::Internal(format!("Invalid base64: {e}")))?;
        // POST raw bytes to /import/archive
        let url = format!("{}/import/archive", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Content-Type", "application/octet-stream")
            .body(bytes)
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("HTTP error: {e}")))?;
        let body = resp
            .text()
            .await
            .map_err(|e| TensaError::Internal(format!("Response read error: {e}")))?;
        serde_json::from_str(&body)
            .map_err(|e| TensaError::Internal(format!("JSON parse error: {e}")))
    }

    async fn verify_authorship(&self, text_a: &str, text_b: &str) -> Result<Value> {
        self.post_json(
            "/style/verify",
            &serde_json::json!({"text_a": text_a, "text_b": text_b}),
        )
        .await
    }

    async fn diagnose_narrative(&self, narrative_id: &str, genre: Option<&str>) -> Result<Value> {
        let body = serde_json::json!({ "genre": genre });
        self.post_json(&format!("/narratives/{}/diagnose", narrative_id), &body)
            .await
    }

    async fn get_health_score(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/health-score", narrative_id))
            .await
    }

    async fn auto_repair(
        &self,
        narrative_id: &str,
        max_severity: Option<&str>,
        max_iterations: Option<usize>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "max_severity": max_severity,
            "max_iterations": max_iterations,
        });
        self.post_json(&format!("/narratives/{}/auto-repair", narrative_id), &body)
            .await
    }

    async fn compute_essentiality(&self, narrative_id: &str) -> Result<Value> {
        self.post_json(
            &format!("/narratives/{}/essentiality", narrative_id),
            &serde_json::Value::Null,
        )
        .await
    }

    async fn compress_narrative(
        &self,
        narrative_id: &str,
        preset: Option<&str>,
        target_chapters: Option<usize>,
        target_ratio: Option<f64>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "preset": preset,
            "config": { "target_chapters": target_chapters.unwrap_or(12), "target_ratio": target_ratio },
        });
        self.post_json(&format!("/narratives/{}/compress", narrative_id), &body)
            .await
    }

    async fn expand_narrative(&self, narrative_id: &str, target_chapters: usize) -> Result<Value> {
        let body = serde_json::json!({
            "preset": "novel",
            "target_chapters": target_chapters,
        });
        self.post_json(&format!("/narratives/{}/expand", narrative_id), &body)
            .await
    }

    async fn diff_narratives(&self, narrative_a: &str, narrative_b: &str) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_a": narrative_a,
            "narrative_b": narrative_b,
        });
        self.post_json("/narratives/diff", &body).await
    }

    // ─── Disinfo Sprint D1 ────────────────────────────────────

    #[cfg(feature = "disinfo")]
    async fn get_behavioral_fingerprint(&self, actor_id: &str, recompute: bool) -> Result<Value> {
        if recompute {
            self.post_json(
                &format!("/entities/{actor_id}/behavioral-fingerprint/compute"),
                &serde_json::Value::Null,
            )
            .await
        } else {
            self.get_json(&format!("/entities/{actor_id}/behavioral-fingerprint"))
                .await
        }
    }

    #[cfg(feature = "disinfo")]
    async fn get_disinfo_fingerprint(&self, narrative_id: &str, recompute: bool) -> Result<Value> {
        if recompute {
            self.post_json(
                &format!("/narratives/{narrative_id}/disinfo-fingerprint/compute"),
                &serde_json::Value::Null,
            )
            .await
        } else {
            self.get_json(&format!("/narratives/{narrative_id}/disinfo-fingerprint"))
                .await
        }
    }

    #[cfg(feature = "disinfo")]
    async fn compare_fingerprints(
        &self,
        kind: &str,
        task: Option<&str>,
        a_id: &str,
        b_id: &str,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "kind": kind,
            "task": task.unwrap_or("literary"),
            "a_id": a_id,
            "b_id": b_id,
        });
        self.post_json("/fingerprints/compare", &body).await
    }

    // ─── Spread Dynamics (Sprint D2) ───────────────────────────

    #[cfg(feature = "disinfo")]
    async fn estimate_r0_by_platform(
        &self,
        narrative_id: &str,
        fact: &str,
        about_entity: &str,
        narrative_kind: Option<&str>,
        beta_overrides: Option<std::collections::HashMap<String, f64>>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "fact": fact,
            "about_entity": about_entity,
            "narrative_kind": narrative_kind,
            "beta_overrides": beta_overrides,
        });
        self.post_json("/spread/r0", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn simulate_intervention(
        &self,
        narrative_id: &str,
        fact: &str,
        about_entity: &str,
        intervention: Value,
        beta_overrides: Option<std::collections::HashMap<String, f64>>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "fact": fact,
            "about_entity": about_entity,
            "intervention": intervention,
            "beta_overrides": beta_overrides,
        });
        self.post_json("/spread/intervention", &body).await
    }

    // ─── CIB Detection (Sprint D3) ───────────────────────────

    #[cfg(feature = "disinfo")]
    async fn detect_cib_cluster(
        &self,
        narrative_id: &str,
        cross_platform: bool,
        similarity_threshold: Option<f64>,
        alpha: Option<f64>,
        bootstrap_iter: Option<usize>,
        min_cluster_size: Option<usize>,
        seed: Option<u64>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "cross_platform": cross_platform,
            "similarity_threshold": similarity_threshold,
            "alpha": alpha,
            "bootstrap_iter": bootstrap_iter,
            "min_cluster_size": min_cluster_size,
            "seed": seed,
        });
        self.post_json("/analysis/cib", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn rank_superspreaders(
        &self,
        narrative_id: &str,
        method: Option<&str>,
        top_n: Option<usize>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "method": method,
            "top_n": top_n,
        });
        self.post_json("/analysis/superspreaders", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_claim(&self, text: &str, narrative_id: Option<&str>) -> Result<Value> {
        let body = serde_json::json!({ "text": text, "narrative_id": narrative_id });
        self.post_json("/claims", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_fact_check(
        &self,
        claim_id: &str,
        verdict: &str,
        source: &str,
        url: Option<&str>,
        language: &str,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "claim_id": claim_id,
            "verdict": verdict,
            "source": source,
            "url": url,
            "language": language,
        });
        self.post_json("/fact-checks", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn fetch_fact_checks(&self, claim_id: &str, min_similarity: f64) -> Result<Value> {
        let body = serde_json::json!({ "claim_id": claim_id, "min_similarity": min_similarity });
        self.post_json("/claims/match", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn trace_claim_origin(&self, claim_id: &str) -> Result<Value> {
        self.get_json(&format!("/claims/{}/origin", claim_id)).await
    }

    #[cfg(feature = "disinfo")]
    async fn classify_archetype(&self, actor_id: &str, force: bool) -> Result<Value> {
        let body = serde_json::json!({ "force": force });
        self.post_json(&format!("/actors/{}/archetype", actor_id), &body)
            .await
    }

    #[cfg(feature = "disinfo")]
    async fn assess_disinfo(&self, target_id: &str, signals: Value) -> Result<Value> {
        let body = serde_json::json!({ "target_id": target_id, "signals": signals });
        self.post_json("/analysis/disinfo-assess", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_post(
        &self,
        text: &str,
        actor_id: &str,
        narrative_id: &str,
        platform: Option<&str>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "text": text,
            "actor_id": actor_id,
            "narrative_id": narrative_id,
            "platform": platform,
        });
        self.post_json("/situations", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn ingest_actor(
        &self,
        name: &str,
        narrative_id: &str,
        platform: Option<&str>,
        properties: Option<Value>,
    ) -> Result<Value> {
        let mut props = properties.unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = props.as_object_mut() {
            obj.insert(
                "name".to_string(),
                serde_json::Value::String(name.to_string()),
            );
            if let Some(p) = platform {
                obj.insert(
                    "platform".to_string(),
                    serde_json::Value::String(p.to_string()),
                );
            }
        }
        let body = serde_json::json!({
            "entity_type": "Actor",
            "properties": props,
            "narrative_id": narrative_id,
            "confidence": 0.8,
            "maturity": "Candidate",
        });
        self.post_json("/entities", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn link_narrative(&self, entity_id: &str, narrative_id: &str) -> Result<Value> {
        let body = serde_json::json!({ "narrative_id": narrative_id });
        self.put_json(&format!("/entities/{}", entity_id), &body)
            .await
    }

    // ─── Multilingual & Export (Sprint D6) ───────────────────────

    #[cfg(feature = "disinfo")]
    async fn detect_language(&self, text: &str) -> Result<Value> {
        self.post_json("/lang/detect", &serde_json::json!({ "text": text }))
            .await
    }

    #[cfg(feature = "disinfo")]
    async fn export_misp_event(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/export?format=misp", narrative_id))
            .await
    }

    #[cfg(feature = "disinfo")]
    async fn export_maltego(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!(
            "/narratives/{}/export?format=maltego",
            narrative_id
        ))
        .await
    }

    #[cfg(feature = "disinfo")]
    async fn generate_disinfo_report(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!(
            "/narratives/{}/export?format=disinfo_report",
            narrative_id
        ))
        .await
    }

    // ─── Sprint D8: Scheduler + Reports + Health ─────────────────

    #[cfg(feature = "disinfo")]
    async fn list_scheduled_tasks(&self) -> Result<Value> {
        self.get_json("/scheduler/tasks").await
    }

    #[cfg(feature = "disinfo")]
    async fn create_scheduled_task(
        &self,
        task_type: &str,
        schedule: &str,
        config: Option<Value>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "task_type": task_type,
            "schedule": schedule,
            "config": config.unwrap_or(serde_json::Value::Null),
        });
        self.post_json("/scheduler/tasks", &body).await
    }

    #[cfg(feature = "disinfo")]
    async fn run_task_now(&self, task_id: &str) -> Result<Value> {
        self.post_json(
            &format!("/scheduler/tasks/{}/run-now", task_id),
            &serde_json::Value::Null,
        )
        .await
    }

    #[cfg(feature = "disinfo")]
    async fn list_discovery_candidates(&self) -> Result<Value> {
        self.get_json("/discovery/candidates").await
    }

    #[cfg(feature = "disinfo")]
    async fn sync_fact_checks(&self) -> Result<Value> {
        self.post_json("/claims/sync", &serde_json::Value::Null)
            .await
    }

    #[cfg(feature = "disinfo")]
    async fn generate_situation_report(&self, hours: u64) -> Result<Value> {
        let body = serde_json::json!({ "hours": hours });
        self.post_json("/reports/situation", &body).await
    }

    // ─── Adversarial Wargaming (Sprint D12) ──────────────────────

    async fn generate_adversary_policy(
        &self,
        narrative_id: &str,
        actor_id: Option<&str>,
        archetype: Option<&str>,
        lambda: Option<f64>,
        lambda_cap: Option<f64>,
        reward_weights: Option<Vec<f64>>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "actor_id": actor_id,
            "archetype": archetype,
            "lambda": lambda,
            "lambda_cap": lambda_cap,
            "reward_weights": reward_weights,
        });
        self.post_json("/adversarial/policy", &body).await
    }

    async fn configure_rationality(
        &self,
        model: &str,
        lambda: Option<f64>,
        lambda_cap: Option<f64>,
        tau: Option<f64>,
        feature_weights: Option<Vec<f64>>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "model": model,
            "lambda": lambda,
            "lambda_cap": lambda_cap,
            "tau": tau,
            "feature_weights": feature_weights,
        });
        self.post_json("/adversarial/rationality", &body).await
    }

    async fn create_wargame(
        &self,
        narrative_id: &str,
        max_turns: Option<usize>,
        time_step_minutes: Option<u64>,
        auto_red: Option<bool>,
        auto_blue: Option<bool>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "max_turns": max_turns,
            "time_step_minutes": time_step_minutes,
            "auto_red": auto_red,
            "auto_blue": auto_blue,
        });
        self.post_json("/wargame/sessions", &body).await
    }

    async fn submit_wargame_move(
        &self,
        session_id: &str,
        red_moves: Option<Value>,
        blue_moves: Option<Value>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "red_moves": red_moves,
            "blue_moves": blue_moves,
        });
        self.post_json(&format!("/wargame/sessions/{}/moves", session_id), &body)
            .await
    }

    async fn get_wargame_state(&self, session_id: &str) -> Result<Value> {
        self.get_json(&format!("/wargame/sessions/{}/state", session_id))
            .await
    }

    async fn auto_play_wargame(&self, session_id: &str, num_turns: usize) -> Result<Value> {
        let body = serde_json::json!({"num_turns": num_turns});
        self.post_json(
            &format!("/wargame/sessions/{}/auto-play", session_id),
            &body,
        )
        .await
    }

    // ─── Writer Workflows (Sprint W6) ────────────────────────────

    async fn get_narrative_plan(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/plan", narrative_id))
            .await
    }

    async fn upsert_narrative_plan(&self, narrative_id: &str, patch: Value) -> Result<Value> {
        // PUT = partial patch (matches the tool's documented "Fields omitted
        // are preserved" semantics). POST would be a full replace and would
        // require a complete `NarrativePlan` body including `narrative_id`,
        // which the MCP caller does not send.
        self.put_json(&format!("/narratives/{}/plan", narrative_id), &patch)
            .await
    }

    async fn get_writer_workspace(&self, narrative_id: &str) -> Result<Value> {
        let mut value = self
            .get_json(&format!("/narratives/{}/workspace", narrative_id))
            .await?;
        // Inject the REST base URL so callers don't have to probe ports to
        // find the Studio server (the skill doc still references :3000 even
        // when the user's TENSA_ADDR sets a different port).
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "api_base_url".to_string(),
                Value::String(self.base_url.clone()),
            );
        }
        Ok(value)
    }

    async fn run_workshop(
        &self,
        narrative_id: &str,
        tier: &str,
        focuses: Option<Vec<String>>,
        max_llm_calls: Option<u32>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "tier": tier,
            "focuses": focuses,
            "max_llm_calls": max_llm_calls,
        });
        self.post_json(&format!("/narratives/{}/workshop/run", narrative_id), &body)
            .await
    }

    async fn list_pinned_facts(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/pinned-facts", narrative_id))
            .await
    }

    async fn create_pinned_fact(&self, narrative_id: &str, fact: Value) -> Result<Value> {
        self.post_json(&format!("/narratives/{}/pinned-facts", narrative_id), &fact)
            .await
    }

    async fn check_continuity(&self, narrative_id: &str, prose: &str) -> Result<Value> {
        let body = serde_json::json!({ "prose": prose });
        self.post_json(
            &format!("/narratives/{}/continuity/check", narrative_id),
            &body,
        )
        .await
    }

    async fn list_narrative_revisions(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        let path = match limit {
            Some(n) => format!("/narratives/{}/revisions?limit={}", narrative_id, n),
            None => format!("/narratives/{}/revisions", narrative_id),
        };
        self.get_json(&path).await
    }

    async fn restore_narrative_revision(
        &self,
        narrative_id: &str,
        revision_id: &str,
        author: &str,
    ) -> Result<Value> {
        let body = serde_json::json!({ "author": author });
        self.post_json(
            &format!(
                "/narratives/{}/revisions/{}/restore",
                narrative_id, revision_id
            ),
            &body,
        )
        .await
    }

    async fn get_writer_cost_summary(
        &self,
        narrative_id: &str,
        window: Option<&str>,
    ) -> Result<Value> {
        let path = match window {
            Some(w) => format!(
                "/narratives/{}/cost-ledger/summary?window={}",
                narrative_id, w
            ),
            None => format!("/narratives/{}/cost-ledger/summary", narrative_id),
        };
        self.get_json(&path).await
    }

    async fn set_situation_content(
        &self,
        situation_id: &str,
        content: Value,
        status: Option<&str>,
    ) -> Result<Value> {
        use crate::types::ContentBlock;

        // The REST PUT handler expects raw_content as an array of
        // ContentBlock objects. Build from the typed struct so the shape
        // stays in lock-step with any future ContentBlock field additions.
        let blocks = match content {
            Value::String(s) => serde_json::to_value(vec![ContentBlock::text(&s)])
                .map_err(|e| TensaError::Serialization(e.to_string()))?,
            other => other,
        };
        let mut patch = serde_json::Map::new();
        patch.insert("raw_content".into(), blocks);
        if let Some(st) = status {
            patch.insert("status".into(), Value::String(st.to_string()));
        }
        self.put_json(
            &format!("/situations/{}", situation_id),
            &Value::Object(patch),
        )
        .await
    }

    async fn get_scene_context(
        &self,
        narrative_id: &str,
        situation_id: Option<&str>,
        pov_entity_id: Option<&str>,
        lookback_scenes: Option<usize>,
    ) -> Result<Value> {
        use crate::mcp::embedded_ext::{clamp_lookback, select_preceding_scenes};
        use crate::types::Situation;

        let lookback = clamp_lookback(lookback_scenes);

        // Four independent REST fetches run in parallel; try_join! cancels
        // the siblings on the first error so failure semantics match
        // sequential `?` — any one bad leg aborts the whole call.
        let plan_path = format!("/narratives/{}/plan", narrative_id);
        let pinned_path = format!("/narratives/{}/pinned-facts", narrative_id);
        let current_path = situation_id.map(|sid| format!("/situations/{}", sid));
        let plan_f = self.get_json(&plan_path);
        let pinned_f = self.get_json(&pinned_path);
        let current_f = async {
            match current_path.as_deref() {
                Some(p) => self.get_json(p).await.map(Some),
                None => Ok(None),
            }
        };
        let actor_profile_f = async {
            match pov_entity_id {
                Some(pid) => self.get_actor_profile(pid).await.map(Some),
                None => Ok(None),
            }
        };
        let (plan, pinned_facts, current_json, actor_profile) =
            tokio::try_join!(plan_f, pinned_f, current_f, actor_profile_f)?;

        let current_situation: Option<Situation> = current_json
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| TensaError::Internal(format!("Failed to parse current situation: {e}")))?;

        let preceding: Vec<Situation> = if let Some(ref current) = current_situation {
            // `/situations?narrative_id=` wraps the list in `{data, total}`.
            // We take the handler's clamp max (10000) to avoid silently
            // truncating a big novel's backlog before the filter runs.
            let resp = self
                .get_json(&format!(
                    "/situations?narrative_id={}&limit=10000",
                    narrative_id
                ))
                .await?;
            let data = resp
                .get("data")
                .cloned()
                .unwrap_or(Value::Array(Vec::new()));
            let all: Vec<Situation> = serde_json::from_value(data).map_err(|e| {
                TensaError::Internal(format!("Failed to parse situations list: {e}"))
            })?;
            select_preceding_scenes(all, current, lookback)
        } else {
            Vec::new()
        };

        let pov_profile = actor_profile.map(|profile| {
            let pov_pinned: Vec<Value> = pinned_facts
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|f| {
                    f.get("entity_id")
                        .and_then(|v| v.as_str())
                        .zip(pov_entity_id)
                        .map(|(a, b)| a == b)
                        .unwrap_or(false)
                })
                .collect();
            serde_json::json!({
                "profile": profile,
                "pinned_facts": pov_pinned,
            })
        });

        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "plan": plan,
            "pinned_facts": pinned_facts,
            "pov_profile": pov_profile,
            "current_situation": current_situation,
            "preceding_scenes": preceding,
            "effective_lookback": lookback,
        }))
    }

    #[cfg(feature = "generation")]
    async fn generate_chapter_with_fitness(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
        style_embedding_id: Option<&str>,
        target_fingerprint_source: Option<&str>,
        fitness_threshold: Option<f64>,
        max_retries: Option<usize>,
        temperature: Option<f64>,
    ) -> Result<Value> {
        // The HTTP backend proxies the job submission. The remote server
        // resolves `target_fingerprint_source` and constructs the StyleTarget;
        // we forward all knobs as job parameters via POST /jobs.
        let mut style = serde_json::Map::new();
        if let Some(v) = voice_description {
            style.insert("voice_description".into(), serde_json::json!(v));
        }
        if let Some(v) = style_embedding_id {
            style.insert("style_embedding_id".into(), serde_json::json!(v));
        }
        if let Some(v) = temperature {
            style.insert("temperature".into(), serde_json::json!(v));
        }
        if let Some(v) = max_retries {
            style.insert("max_retries_per_chapter".into(), serde_json::json!(v));
        }
        if let Some(v) = fitness_threshold {
            style.insert("fitness_threshold".into(), serde_json::json!(v));
        }
        if let Some(v) = target_fingerprint_source {
            style.insert("target_fingerprint_source".into(), serde_json::json!(v));
        }

        let body = serde_json::json!({
            "job_type": "ChapterGenerationFitness",
            "target_id": uuid::Uuid::now_v7(),
            "parameters": {
                "narrative_id": narrative_id,
                "chapter": chapter,
                "style": style,
            },
        });
        self.post_json("/jobs", &body).await
    }

    // ─── Sprint W14: narrative architecture ───────────────────────

    async fn detect_commitments(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/commitments", narrative_id))
            .await
    }

    async fn get_commitment_rhythm(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/commitment-rhythm", narrative_id))
            .await
    }

    async fn extract_fabula(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/fabula", narrative_id))
            .await
    }

    async fn extract_sjuzet(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/sjuzet", narrative_id))
            .await
    }

    async fn suggest_reordering(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/sjuzet/reorderings", narrative_id))
            .await
    }

    async fn compute_dramatic_irony(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/dramatic-irony", narrative_id))
            .await
    }

    async fn detect_focalization(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/focalization", narrative_id))
            .await
    }

    async fn detect_character_arc(
        &self,
        narrative_id: &str,
        character_id: Option<&str>,
    ) -> Result<Value> {
        let path = match character_id {
            Some(id) => format!(
                "/narratives/{}/character-arc?character_id={}",
                narrative_id, id
            ),
            None => format!("/narratives/{}/character-arc", narrative_id),
        };
        self.get_json(&path).await
    }

    async fn detect_subplots(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/subplots", narrative_id))
            .await
    }

    async fn classify_scene_sequel(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/scene-sequel", narrative_id))
            .await
    }

    #[cfg(feature = "generation")]
    async fn generate_narrative_plan(
        &self,
        premise: &str,
        genre: &str,
        chapter_count: usize,
        subplot_count: usize,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "premise": premise,
            "genre": genre,
            "chapter_count": chapter_count,
            "subplot_count": subplot_count,
        });
        self.post_json("/narratives/plan", &body).await
    }

    #[cfg(feature = "generation")]
    async fn materialize_plan(&self, plan_id: &str) -> Result<Value> {
        self.post_json(
            &format!("/plans/{}/materialize", plan_id),
            &serde_json::Value::Null,
        )
        .await
    }

    #[cfg(feature = "generation")]
    async fn validate_materialized_narrative(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!(
            "/narratives/{}/validate-materialized",
            narrative_id
        ))
        .await
    }

    #[cfg(feature = "generation")]
    async fn generate_chapter(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("chapter".into(), serde_json::json!(chapter));
        if let Some(v) = voice_description {
            body.insert("voice_description".into(), serde_json::json!(v));
        }
        self.post_json(
            &format!("/narratives/{}/generate-chapter", narrative_id),
            &serde_json::Value::Object(body),
        )
        .await
    }

    #[cfg(feature = "generation")]
    async fn generate_narrative(
        &self,
        narrative_id: &str,
        chapter_count: usize,
        voice_description: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("chapter_count".into(), serde_json::json!(chapter_count));
        if let Some(v) = voice_description {
            body.insert("voice_description".into(), serde_json::json!(v));
        }
        self.post_json(
            &format!("/narratives/{}/generate-narrative", narrative_id),
            &serde_json::Value::Object(body),
        )
        .await
    }

    // ─── Sprint W15: Writer MCP bridge ────────────────────────

    async fn create_annotation(
        &self,
        situation_id: &str,
        kind: &str,
        body: &str,
        span_start: usize,
        span_end: usize,
        source_id: Option<&str>,
        chunk_id: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        let mut payload = serde_json::Map::new();
        payload.insert("kind".into(), serde_json::json!(kind));
        payload.insert(
            "span".into(),
            serde_json::json!([span_start, span_end.max(span_start)]),
        );
        payload.insert("body".into(), serde_json::json!(body));
        if let Some(v) = source_id {
            payload.insert("source_id".into(), serde_json::json!(v));
        }
        if let Some(v) = chunk_id {
            payload.insert("chunk_id".into(), serde_json::json!(v));
        }
        if let Some(v) = author {
            payload.insert("author".into(), serde_json::json!(v));
        }
        self.post_json(
            &format!("/situations/{}/annotations", situation_id),
            &Value::Object(payload),
        )
        .await
    }

    async fn list_annotations(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> Result<Value> {
        if let Some(sid) = situation_id {
            return self
                .get_json(&format!("/situations/{}/annotations", sid))
                .await;
        }
        let nid = narrative_id.ok_or_else(|| {
            TensaError::InvalidQuery(
                "provide situation_id or narrative_id to list annotations".into(),
            )
        })?;
        // Server-side batch (v0.72): one prefix scan, one round-trip.
        // Prior releases (v0.70.0-v0.71.x) did N+1 HTTP calls here.
        self.get_json(&format!("/narratives/{}/annotations", nid))
            .await
    }

    async fn update_annotation(&self, annotation_id: &str, patch: Value) -> Result<Value> {
        self.put_json(&format!("/annotations/{}", annotation_id), &patch)
            .await
    }

    async fn delete_annotation(&self, annotation_id: &str) -> Result<Value> {
        self.delete(&format!("/annotations/{}", annotation_id))
            .await
    }

    async fn create_collection(
        &self,
        narrative_id: &str,
        name: &str,
        description: Option<&str>,
        query: Value,
    ) -> Result<Value> {
        let mut payload = serde_json::Map::new();
        payload.insert("name".into(), serde_json::json!(name));
        if let Some(d) = description {
            payload.insert("description".into(), serde_json::json!(d));
        }
        if !query.is_null() {
            payload.insert("query".into(), query);
        }
        self.post_json(
            &format!("/narratives/{}/collections", narrative_id),
            &Value::Object(payload),
        )
        .await
    }

    async fn list_collections(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/collections", narrative_id))
            .await
    }

    async fn get_collection(&self, collection_id: &str, resolve: bool) -> Result<Value> {
        let meta = self
            .get_json(&format!("/collections/{}", collection_id))
            .await?;
        if resolve {
            let res = self
                .get_json(&format!("/collections/{}/resolve", collection_id))
                .await?;
            return Ok(serde_json::json!({
                "collection": meta,
                "resolution": res,
            }));
        }
        Ok(meta)
    }

    async fn update_collection(&self, collection_id: &str, patch: Value) -> Result<Value> {
        self.put_json(&format!("/collections/{}", collection_id), &patch)
            .await
    }

    async fn delete_collection(&self, collection_id: &str) -> Result<Value> {
        self.delete(&format!("/collections/{}", collection_id))
            .await
    }

    async fn create_research_note(
        &self,
        narrative_id: &str,
        situation_id: &str,
        kind: &str,
        body: &str,
        source_id: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        let mut payload = serde_json::Map::new();
        payload.insert("narrative_id".into(), serde_json::json!(narrative_id));
        payload.insert("kind".into(), serde_json::json!(kind));
        payload.insert("body".into(), serde_json::json!(body));
        if let Some(v) = source_id {
            payload.insert("source_id".into(), serde_json::json!(v));
        }
        if let Some(v) = author {
            payload.insert("author".into(), serde_json::json!(v));
        }
        self.post_json(
            &format!("/situations/{}/research-notes", situation_id),
            &Value::Object(payload),
        )
        .await
    }

    async fn list_research_notes(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> Result<Value> {
        if let Some(sid) = situation_id {
            return self
                .get_json(&format!("/situations/{}/research-notes", sid))
                .await;
        }
        let nid = narrative_id.ok_or_else(|| {
            TensaError::InvalidQuery(
                "provide situation_id or narrative_id to list research notes".into(),
            )
        })?;
        self.get_json(&format!("/narratives/{}/research-notes", nid))
            .await
    }

    async fn get_research_note(&self, note_id: &str) -> Result<Value> {
        self.get_json(&format!("/research-notes/{}", note_id)).await
    }

    async fn update_research_note(&self, note_id: &str, patch: Value) -> Result<Value> {
        self.put_json(&format!("/research-notes/{}", note_id), &patch)
            .await
    }

    async fn delete_research_note(&self, note_id: &str) -> Result<Value> {
        self.delete(&format!("/research-notes/{}", note_id)).await
    }

    async fn promote_chunk_to_note(
        &self,
        narrative_id: &str,
        situation_id: &str,
        chunk_id: &str,
        body: &str,
        source_id: Option<&str>,
        kind: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        let mut payload = serde_json::Map::new();
        payload.insert("narrative_id".into(), serde_json::json!(narrative_id));
        payload.insert("chunk_id".into(), serde_json::json!(chunk_id));
        payload.insert("body".into(), serde_json::json!(body));
        if let Some(v) = source_id {
            payload.insert("source_id".into(), serde_json::json!(v));
        }
        if let Some(v) = kind {
            payload.insert("kind".into(), serde_json::json!(v));
        }
        if let Some(v) = author {
            payload.insert("author".into(), serde_json::json!(v));
        }
        self.post_json(
            &format!("/situations/{}/research-notes/from-chunk", situation_id),
            &Value::Object(payload),
        )
        .await
    }

    async fn propose_edit(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> Result<Value> {
        let op = build_edit_op(instruction, style_preset);
        self.post_json(&format!("/situations/{}/edit", situation_id), &op)
            .await
    }

    async fn apply_edit(&self, proposal: Value, author: Option<&str>) -> Result<Value> {
        let situation_id = proposal
            .get("situation_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("proposal.situation_id is required".into()))?
            .to_string();
        let mut body = serde_json::Map::new();
        body.insert("proposal".into(), proposal);
        if let Some(a) = author {
            body.insert("author".into(), serde_json::json!(a));
        }
        self.post_json(
            &format!("/situations/{}/edit/apply", situation_id),
            &Value::Object(body),
        )
        .await
    }

    async fn estimate_edit_tokens(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> Result<Value> {
        let op = build_edit_op(instruction, style_preset);
        // dry_run=true avoids the LLM call — returns the prompt + token estimate only.
        self.post_json(
            &format!("/situations/{}/edit?dry_run=true", situation_id),
            &op,
        )
        .await
    }

    async fn commit_narrative_revision(
        &self,
        narrative_id: &str,
        message: &str,
        author: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("message".into(), serde_json::json!(message));
        if let Some(a) = author {
            body.insert("author".into(), serde_json::json!(a));
        }
        self.post_json(
            &format!("/narratives/{}/revisions", narrative_id),
            &Value::Object(body),
        )
        .await
    }

    async fn diff_narrative_revisions(
        &self,
        narrative_id: &str,
        from_rev: &str,
        to_rev: &str,
    ) -> Result<Value> {
        self.get_json(&format!(
            "/narratives/{}/diff-revisions?from={}&to={}",
            narrative_id, from_rev, to_rev
        ))
        .await
    }

    async fn list_workshop_reports(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/workshop/reports", narrative_id))
            .await
    }

    async fn get_workshop_report(&self, report_id: &str) -> Result<Value> {
        self.get_json(&format!("/workshop/reports/{}", report_id))
            .await
    }

    async fn list_cost_ledger_entries(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        let path = match limit {
            Some(n) => format!("/narratives/{}/cost-ledger?limit={}", narrative_id, n),
            None => format!("/narratives/{}/cost-ledger", narrative_id),
        };
        self.get_json(&path).await
    }

    async fn list_compile_profiles(&self, narrative_id: &str) -> Result<Value> {
        self.get_json(&format!("/narratives/{}/compile-profiles", narrative_id))
            .await
    }

    async fn compile_narrative(
        &self,
        narrative_id: &str,
        format: &str,
        profile_id: Option<&str>,
    ) -> Result<Value> {
        let mut path = format!("/narratives/{}/compile?format={}", narrative_id, format);
        if let Some(pid) = profile_id {
            path.push_str(&format!("&profile_id={}", pid));
        }
        // The REST endpoint returns raw bytes; we wrap them into a JSON envelope
        // that matches the embedded-backend shape.
        let url = self.url(&path);
        let resp = self
            .client
            .post(url)
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("HTTP request failed: {}", e)))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(TensaError::Internal(format!(
                "compile failed ({}): {}",
                status, body
            )));
        }
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| TensaError::Internal(format!("compile body read failed: {}", e)))?;
        let (body_field, encoding) =
            if format.eq_ignore_ascii_case("markdown") || format.eq_ignore_ascii_case("md") {
                (
                    Value::String(
                        String::from_utf8(bytes.to_vec())
                            .unwrap_or_else(|_| "<invalid-utf8>".to_string()),
                    ),
                    "utf-8",
                )
            } else {
                use base64::Engine;
                (
                    Value::String(base64::engine::general_purpose::STANDARD.encode(&bytes)),
                    "base64",
                )
            };
        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "format": format,
            "content_type": content_type,
            "bytes": bytes.len(),
            "body": body_field,
            "encoding": encoding,
        }))
    }

    async fn upsert_compile_profile(
        &self,
        narrative_id: &str,
        profile_id: Option<&str>,
        patch: Value,
    ) -> Result<Value> {
        match profile_id {
            Some(id) => {
                self.put_json(&format!("/compile-profiles/{}", id), &patch)
                    .await
            }
            None => {
                self.post_json(
                    &format!("/narratives/{}/compile-profiles", narrative_id),
                    &patch,
                )
                .await
            }
        }
    }

    async fn list_narrative_templates(&self) -> Result<Value> {
        self.get_json("/narrative-templates").await
    }

    async fn instantiate_template(
        &self,
        template_id: &str,
        bindings: std::collections::HashMap<String, String>,
    ) -> Result<Value> {
        self.post_json(
            &format!("/narrative-templates/{}/instantiate", template_id),
            &serde_json::json!({ "bindings": bindings }),
        )
        .await
    }

    async fn extract_narrative_skeleton(&self, narrative_id: &str) -> Result<Value> {
        // No dedicated REST route; call the embedded-backend-equivalent via /query.
        // Fall back to a not-available error rather than inventing a nonexistent
        // endpoint — the MCP server should use EmbeddedBackend for this tool.
        Err(TensaError::Internal(format!(
            "extract_narrative_skeleton not available over HTTP (narrative={narrative_id}); use embedded backend"
        )))
    }

    async fn find_duplicate_candidates(
        &self,
        narrative_id: &str,
        threshold: Option<f64>,
        max_candidates: Option<usize>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        if let Some(t) = threshold {
            body.insert("threshold".into(), serde_json::json!(t));
        }
        if let Some(m) = max_candidates {
            body.insert("max_candidates".into(), serde_json::json!(m));
        }
        self.post_json(
            &format!("/narratives/{}/dedup-entities", narrative_id),
            &Value::Object(body),
        )
        .await
    }

    async fn suggest_narrative_fixes(&self, narrative_id: &str) -> Result<Value> {
        self.post_json(
            &format!("/narratives/{}/suggest-fixes", narrative_id),
            &Value::Null,
        )
        .await
    }

    async fn apply_narrative_fix(&self, narrative_id: &str, fix: Value) -> Result<Value> {
        self.post_json(&format!("/narratives/{}/apply-fix", narrative_id), &fix)
            .await
    }

    async fn apply_reorder(&self, narrative_id: &str, entries: Value) -> Result<Value> {
        self.post_json(
            &format!("/narratives/{}/reorder", narrative_id),
            &serde_json::json!({ "entries": entries }),
        )
        .await
    }

    // ─── EATH Phase 10 — Synthetic Hypergraph MCP tools ──────────

    async fn calibrate_surrogate(
        &self,
        narrative_id: &str,
        model: Option<&str>,
    ) -> Result<Value> {
        // Body matches the REST contract — `model` is optional; server defaults
        // to "eath" when absent. Send `{}` rather than `null` so axum's Json
        // extractor accepts it.
        let body = match model.filter(|m| !m.is_empty()) {
            Some(m) => serde_json::json!({"model": m}),
            None => serde_json::json!({}),
        };
        self.post_json(&format!("/synth/calibrate/{}", narrative_id), &body)
            .await
    }

    async fn generate_synthetic_narrative(
        &self,
        source_narrative_id: &str,
        output_narrative_id: &str,
        model: Option<&str>,
        params: Option<Value>,
        seed: Option<u64>,
        num_steps: Option<usize>,
        label_prefix: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        if !source_narrative_id.is_empty() {
            body.insert(
                "source_narrative_id".into(),
                Value::String(source_narrative_id.to_string()),
            );
        }
        body.insert(
            "output_narrative_id".into(),
            Value::String(output_narrative_id.to_string()),
        );
        if let Some(m) = model.filter(|m| !m.is_empty()) {
            body.insert("model".into(), Value::String(m.to_string()));
        }
        if let Some(p) = params {
            body.insert("params".into(), p);
        }
        if let Some(s) = seed {
            body.insert("seed".into(), serde_json::json!(s));
        }
        if let Some(n) = num_steps {
            body.insert("num_steps".into(), serde_json::json!(n));
        }
        if let Some(prefix) = label_prefix {
            if !prefix.is_empty() {
                body.insert("label_prefix".into(), Value::String(prefix.to_string()));
            }
        }
        self.post_json("/synth/generate", &Value::Object(body)).await
    }

    async fn generate_hybrid_narrative(
        &self,
        components: Value,
        output_narrative_id: &str,
        seed: Option<u64>,
        num_steps: Option<usize>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("components".into(), components);
        body.insert(
            "output_narrative_id".into(),
            Value::String(output_narrative_id.to_string()),
        );
        if let Some(s) = seed {
            body.insert("seed".into(), serde_json::json!(s));
        }
        if let Some(n) = num_steps {
            body.insert("num_steps".into(), serde_json::json!(n));
        }
        self.post_json("/synth/generate-hybrid", &Value::Object(body))
            .await
    }

    async fn list_synthetic_runs(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        let path = match limit {
            Some(n) => format!("/synth/runs/{}?limit={}", narrative_id, n),
            None => format!("/synth/runs/{}", narrative_id),
        };
        self.get_json(&path).await
    }

    async fn get_fidelity_report(
        &self,
        narrative_id: &str,
        run_id: &str,
    ) -> Result<Value> {
        self.get_json(&format!("/synth/fidelity/{}/{}", narrative_id, run_id))
            .await
    }

    async fn compute_pattern_significance(
        &self,
        narrative_id: &str,
        metric: &str,
        k: Option<u16>,
        model: Option<&str>,
        params_override: Option<Value>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("narrative_id".into(), Value::String(narrative_id.to_string()));
        body.insert("metric".into(), Value::String(metric.to_string()));
        if let Some(k) = k {
            body.insert("k".into(), serde_json::json!(k));
        }
        if let Some(m) = model.filter(|m| !m.is_empty()) {
            body.insert("model".into(), Value::String(m.to_string()));
        }
        if let Some(p) = params_override {
            body.insert("params_override".into(), p);
        }
        self.post_json("/synth/significance", &Value::Object(body))
            .await
    }

    async fn simulate_higher_order_contagion(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        model: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("narrative_id".into(), Value::String(narrative_id.to_string()));
        body.insert("params".into(), params);
        if let Some(k) = k {
            body.insert("k".into(), serde_json::json!(k));
        }
        if let Some(m) = model.filter(|m| !m.is_empty()) {
            body.insert("model".into(), Value::String(m.to_string()));
        }
        self.post_json("/synth/contagion-significance", &Value::Object(body))
            .await
    }

    async fn compute_dual_significance(
        &self,
        narrative_id: &str,
        metric: &str,
        k_per_model: Option<u16>,
        models: Option<Vec<String>>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("narrative_id".into(), Value::String(narrative_id.to_string()));
        body.insert("metric".into(), Value::String(metric.to_string()));
        if let Some(k) = k_per_model {
            body.insert("k_per_model".into(), serde_json::json!(k));
        }
        if let Some(ms) = models {
            body.insert("models".into(), serde_json::json!(ms));
        }
        self.post_json("/synth/dual-significance", &Value::Object(body))
            .await
    }

    async fn compute_bistability_significance(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        models: Option<Vec<String>>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("narrative_id".into(), Value::String(narrative_id.to_string()));
        body.insert("params".into(), params);
        if let Some(k) = k {
            body.insert("k".into(), serde_json::json!(k));
        }
        if let Some(ms) = models {
            body.insert("models".into(), serde_json::json!(ms));
        }
        self.post_json("/synth/bistability-significance", &Value::Object(body))
            .await
    }

    async fn reconstruct_hypergraph(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        if let Some(p) = params {
            body.insert("params".into(), p);
        }
        self.post_json("/inference/hypergraph-reconstruction", &Value::Object(body))
            .await
    }

    async fn simulate_opinion_dynamics(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        if let Some(p) = params {
            body.insert("params".into(), p);
        }
        self.post_json("/analysis/opinion-dynamics", &Value::Object(body))
            .await
    }

    async fn simulate_opinion_phase_transition(
        &self,
        narrative_id: &str,
        c_range: [Value; 3],
        base_params: Option<Value>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        body.insert("c_range".into(), Value::Array(c_range.to_vec()));
        if let Some(bp) = base_params {
            body.insert("base_params".into(), bp);
        }
        self.post_json(
            "/analysis/opinion-dynamics/phase-transition-sweep",
            &Value::Object(body),
        )
        .await
    }

    async fn fuzzy_probability(
        &self,
        narrative_id: &str,
        event: Value,
        distribution: Value,
        tnorm: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        body.insert("event".into(), event);
        body.insert("distribution".into(), distribution);
        if let Some(t) = tnorm {
            body.insert("tnorm".into(), Value::String(t.to_string()));
        }
        self.post_json("/fuzzy/hybrid/probability", &Value::Object(body))
            .await
    }

    // ─── Fuzzy Sprint Phase 11 — HTTP forwarders ─────────────────

    async fn fuzzy_list_tnorms(&self) -> Result<Value> {
        self.get_json("/fuzzy/tnorms").await
    }

    async fn fuzzy_list_aggregators(&self) -> Result<Value> {
        self.get_json("/fuzzy/aggregators").await
    }

    async fn fuzzy_get_config(&self) -> Result<Value> {
        self.get_json("/fuzzy/config").await
    }

    async fn fuzzy_set_config(
        &self,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
        measure: Option<Option<&str>>,
        reset: bool,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        if let Some(t) = tnorm {
            body.insert("tnorm".into(), Value::String(t.to_string()));
        }
        if let Some(a) = aggregator {
            body.insert("aggregator".into(), Value::String(a.to_string()));
        }
        match measure {
            Some(Some(m)) => {
                body.insert("measure".into(), Value::String(m.to_string()));
            }
            Some(None) => {
                body.insert("measure".into(), Value::Null);
            }
            None => {}
        }
        if reset {
            body.insert("reset".into(), Value::Bool(true));
        }
        self.put_json("/fuzzy/config", &Value::Object(body)).await
    }

    async fn fuzzy_create_measure(
        &self,
        name: &str,
        n: u8,
        values: Vec<f64>,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "name": name,
            "n": n,
            "values": values,
        });
        self.post_json("/fuzzy/measures", &body).await
    }

    async fn fuzzy_list_measures(&self) -> Result<Value> {
        self.get_json("/fuzzy/measures").await
    }

    async fn fuzzy_aggregate(
        &self,
        xs: Vec<f64>,
        aggregator: &str,
        tnorm: Option<&str>,
        measure: Option<&str>,
        owa_weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("xs".into(), serde_json::json!(xs));
        body.insert("aggregator".into(), Value::String(aggregator.to_string()));
        if let Some(t) = tnorm {
            body.insert("tnorm".into(), Value::String(t.to_string()));
        }
        if let Some(m) = measure {
            body.insert("measure".into(), Value::String(m.to_string()));
        }
        if let Some(w) = owa_weights {
            body.insert("owa_weights".into(), serde_json::json!(w));
        }
        if let Some(s) = seed {
            body.insert("seed".into(), serde_json::json!(s));
        }
        self.post_json("/fuzzy/aggregate", &Value::Object(body))
            .await
    }

    async fn fuzzy_allen_gradation(
        &self,
        narrative_id: &str,
        a_id: &str,
        b_id: &str,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "narrative_id": narrative_id,
            "a_id": a_id,
            "b_id": b_id,
        });
        self.post_json("/analysis/fuzzy-allen", &body).await
    }

    async fn fuzzy_quantify(
        &self,
        narrative_id: &str,
        quantifier: &str,
        entity_type: Option<&str>,
        where_spec: Option<&str>,
        label: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        body.insert("quantifier".into(), Value::String(quantifier.to_string()));
        if let Some(t) = entity_type {
            body.insert("entity_type".into(), Value::String(t.to_string()));
        }
        if let Some(w) = where_spec {
            body.insert("where".into(), Value::String(w.to_string()));
        }
        if let Some(l) = label {
            body.insert("label".into(), Value::String(l.to_string()));
        }
        self.post_json("/fuzzy/quantify", &Value::Object(body)).await
    }

    async fn fuzzy_verify_syllogism(
        &self,
        narrative_id: &str,
        major: &str,
        minor: &str,
        conclusion: &str,
        threshold: Option<f64>,
        tnorm: Option<&str>,
        figure_hint: Option<&str>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        body.insert("major".into(), Value::String(major.to_string()));
        body.insert("minor".into(), Value::String(minor.to_string()));
        body.insert("conclusion".into(), Value::String(conclusion.to_string()));
        if let Some(t) = threshold {
            body.insert("threshold".into(), serde_json::json!(t));
        }
        if let Some(t) = tnorm {
            body.insert("tnorm".into(), Value::String(t.to_string()));
        }
        if let Some(f) = figure_hint {
            body.insert("figure_hint".into(), Value::String(f.to_string()));
        }
        self.post_json("/fuzzy/syllogism/verify", &Value::Object(body))
            .await
    }

    async fn fuzzy_build_lattice(
        &self,
        narrative_id: &str,
        entity_type: Option<&str>,
        attribute_allowlist: Option<Vec<String>>,
        threshold: Option<usize>,
        tnorm: Option<&str>,
        large_context: bool,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        if let Some(et) = entity_type {
            body.insert("entity_type".into(), Value::String(et.to_string()));
        }
        if let Some(al) = attribute_allowlist {
            body.insert("attribute_allowlist".into(), serde_json::json!(al));
        }
        if let Some(th) = threshold {
            body.insert("threshold".into(), serde_json::json!(th));
        }
        if let Some(t) = tnorm {
            body.insert("tnorm".into(), Value::String(t.to_string()));
        }
        if large_context {
            body.insert("large_context".into(), Value::Bool(true));
        }
        self.post_json("/fuzzy/fca/lattice", &Value::Object(body))
            .await
    }

    async fn fuzzy_create_rule(
        &self,
        name: &str,
        narrative_id: &str,
        antecedent: Value,
        consequent: Value,
        tnorm: Option<&str>,
        enabled: Option<bool>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("name".into(), Value::String(name.to_string()));
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        body.insert("antecedent".into(), antecedent);
        body.insert("consequent".into(), consequent);
        if let Some(t) = tnorm {
            body.insert("tnorm".into(), Value::String(t.to_string()));
        }
        if let Some(e) = enabled {
            body.insert("enabled".into(), Value::Bool(e));
        }
        self.post_json("/fuzzy/rules", &Value::Object(body)).await
    }

    async fn fuzzy_evaluate_rules(
        &self,
        narrative_id: &str,
        entity_id: &str,
        rule_ids: Option<Vec<String>>,
        firing_aggregator: Option<crate::fuzzy::aggregation::AggregatorKind>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert("entity_id".into(), Value::String(entity_id.to_string()));
        if let Some(ids) = rule_ids {
            body.insert("rule_ids".into(), serde_json::json!(ids));
        }
        if let Some(k) = firing_aggregator {
            let v = serde_json::to_value(&k)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            body.insert("firing_aggregator".into(), v);
        }
        let path = format!("/fuzzy/rules/{}/evaluate", narrative_id);
        self.post_json(&path, &Value::Object(body)).await
    }

    // ─── Graded Acceptability Sprint Phase 5 — HTTP forwarders ───

    async fn argumentation_gradual(
        &self,
        narrative_id: &str,
        gradual_semantics: Value,
        tnorm: Option<Value>,
    ) -> Result<Value> {
        let mut body = serde_json::Map::new();
        body.insert(
            "narrative_id".into(),
            Value::String(narrative_id.to_string()),
        );
        body.insert("gradual_semantics".into(), gradual_semantics);
        if let Some(t) = tnorm {
            if !t.is_null() {
                body.insert("tnorm".into(), t);
            }
        }
        self.post_json("/analysis/argumentation/gradual", &Value::Object(body))
            .await
    }

    async fn fuzzy_learn_measure(
        &self,
        name: &str,
        n: u8,
        dataset: Vec<(Vec<f64>, u32)>,
        dataset_id: &str,
    ) -> Result<Value> {
        let body = serde_json::json!({
            "name": name,
            "n": n,
            "dataset": dataset,
            "dataset_id": dataset_id,
        });
        self.post_json("/fuzzy/measures/learn", &body).await
    }

    async fn fuzzy_get_measure_version(
        &self,
        name: &str,
        version: Option<u32>,
    ) -> Result<Value> {
        let path = match version {
            Some(v) => format!("/fuzzy/measures/{}?version={}", name, v),
            None => format!("/fuzzy/measures/{}", name),
        };
        self.get_json(&path).await
    }

    async fn fuzzy_list_measure_versions(&self, name: &str) -> Result<Value> {
        let path = format!("/fuzzy/measures/{}/versions", name);
        self.get_json(&path).await
    }

    async fn temporal_ordhorn_closure(&self, network: Value) -> Result<Value> {
        let body = serde_json::json!({"network": network});
        self.post_json("/temporal/ordhorn/closure", &body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_construction() {
        let backend = HttpBackend::new("http://localhost:3000/".to_string());
        assert_eq!(backend.url("/entities"), "http://localhost:3000/entities");
    }

    #[test]
    fn test_url_no_trailing_slash() {
        let backend = HttpBackend::new("http://localhost:3000".to_string());
        assert_eq!(backend.url("/entities"), "http://localhost:3000/entities");
    }

    /// Minimal one-shot HTTP responder used for HTTP-backend proxy tests.
    /// Captures the first request to `(path, body)` and replies with a
    /// canned JSON `{ "job_id": "test-job", "status": "Pending" }` 201
    /// response. Returns `(base_url, captured)` where `captured` resolves
    /// to the captured `(method, path, body_bytes)` tuple.
    async fn one_shot_capture() -> (
        String,
        tokio::sync::oneshot::Receiver<(String, String, Vec<u8>)>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base = format!("http://{}", addr);
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            // Read enough bytes to cover headers + small JSON body. 4 KiB is
            // plenty for our synthetic test bodies (largest is the contagion
            // params, well under 1 KiB).
            let mut buf = vec![0u8; 4096];
            let mut total = 0usize;
            // Read until \r\n\r\n is seen, then read content_length more.
            loop {
                let n = sock.read(&mut buf[total..]).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                total += n;
                if let Some(pos) = buf[..total]
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                {
                    let header_end = pos + 4;
                    // Owned copy of the header bytes so we can keep reading
                    // body bytes into `buf` without overlapping borrows.
                    let header_owned: String =
                        String::from_utf8_lossy(&buf[..header_end]).into_owned();
                    let mut content_len = 0usize;
                    for line in header_owned.split("\r\n") {
                        if let Some(rest) = line.strip_prefix("Content-Length: ") {
                            content_len = rest.parse().unwrap_or(0);
                        } else if let Some(rest) = line.strip_prefix("content-length: ") {
                            content_len = rest.parse().unwrap_or(0);
                        }
                    }
                    while total - header_end < content_len {
                        let n = sock.read(&mut buf[total..]).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        total += n;
                    }
                    let req_line = header_owned.lines().next().unwrap_or("");
                    let mut parts = req_line.split_whitespace();
                    let method = parts.next().unwrap_or("").to_string();
                    let path = parts.next().unwrap_or("").to_string();
                    let body = buf[header_end..header_end + content_len].to_vec();
                    let _ = tx.send((method, path, body));
                    break;
                }
            }
            let body = b"{\"job_id\":\"test-job\",\"status\":\"Pending\"}";
            let resp = format!(
                "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.write_all(body).await;
            let _ = sock.shutdown().await;
        });
        (base, rx)
    }

    #[tokio::test]
    async fn test_mcp_http_backend_proxies_calibrate_correctly() {
        // Verifies the HTTP backend hits POST /synth/calibrate/{nid} with the
        // correct body envelope. This is the EATH Phase 10 test T8.
        let (base, captured) = one_shot_capture().await;
        let backend = HttpBackend::new(base);

        let result = backend.calibrate_surrogate("hamlet", None).await.unwrap();
        assert_eq!(result["job_id"], "test-job");
        assert_eq!(result["status"], "Pending");

        let (method, path, body) =
            tokio::time::timeout(std::time::Duration::from_secs(2), captured)
                .await
                .expect("listener closed before request arrived")
                .expect("no request received");
        assert_eq!(method, "POST");
        assert_eq!(path, "/synth/calibrate/hamlet");
        // None → server-side default; we send `{}` so axum's Json extractor
        // accepts the body.
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed, serde_json::json!({}));
    }

    #[tokio::test]
    async fn test_mcp_http_backend_calibrate_includes_model() {
        let (base, captured) = one_shot_capture().await;
        let backend = HttpBackend::new(base);

        let _ = backend.calibrate_surrogate("hamlet", Some("eath")).await;
        let (_, path, body) =
            tokio::time::timeout(std::time::Duration::from_secs(2), captured)
                .await
                .expect("listener closed before request arrived")
                .expect("no request received");
        assert_eq!(path, "/synth/calibrate/hamlet");
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed, serde_json::json!({"model": "eath"}));
    }

    #[tokio::test]
    async fn test_mcp_http_backend_significance_routes_to_synth() {
        let (base, captured) = one_shot_capture().await;
        let backend = HttpBackend::new(base);

        let _ = backend
            .compute_pattern_significance("hamlet", "patterns", Some(50), None, None)
            .await;
        let (method, path, body) =
            tokio::time::timeout(std::time::Duration::from_secs(2), captured)
                .await
                .expect("listener closed before request arrived")
                .expect("no request received");
        assert_eq!(method, "POST");
        assert_eq!(path, "/synth/significance");
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["narrative_id"], "hamlet");
        assert_eq!(parsed["metric"], "patterns");
        assert_eq!(parsed["k"], 50);
    }
}
