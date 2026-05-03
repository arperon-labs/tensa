//! Extended EmbeddedBackend methods — new MCP tool backends + extracted helpers.
//!
//! Split from `embedded.rs` to keep files under 500 lines.

use serde_json::Value;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::export::{self, ExportFormat};
use crate::narrative::corpus::CorpusManager;
use crate::narrative::project::ProjectRegistry;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::types::Project;
use crate::types::{EntityType, ALL_ENTITY_TYPES};

use super::embedded::EmbeddedBackend;

/// Parse a UUID from a string, returning a TensaError on failure.
pub(crate) fn parse_uuid(id: &str) -> Result<Uuid> {
    id.parse()
        .map_err(|e| TensaError::InvalidQuery(format!("Invalid UUID: {}", e)))
}

/// Convert a value to JSON, mapping serde errors.
pub(crate) fn to_json<T: serde::Serialize>(val: T) -> Result<Value> {
    serde_json::to_value(val).map_err(|e| TensaError::Serialization(e.to_string()))
}

/// Parse a JSON value into `T`, mapping deserialization errors to
/// `TensaError::InvalidInput` with the target type labelled. serde_json
/// errors include the offending field path (`at line 1 column 42`,
/// `expected string, found null`, etc.) — prefixing them with the target
/// name surfaces enough context at the MCP tool boundary that callers can
/// locate the bad field without reading the Rust source.
pub(crate) fn parse_as<T: serde::de::DeserializeOwned>(value: Value, target: &str) -> Result<T> {
    serde_json::from_value(value)
        .map_err(|e| TensaError::InvalidInput(format!("invalid {target} payload: {e}")))
}

/// Canonical success envelope for delete / restore / no-content tool responses.
/// Keeps the `{"status":"ok","id":"..."}` shape consistent across embedded impls.
pub(crate) fn ok_id(id: impl AsRef<str>) -> Value {
    serde_json::json!({"status": "ok", "id": id.as_ref()})
}

/// Default and maximum `lookback_scenes` for `get_scene_context`. Keep the
/// default small so typical drafting loops don't bloat the prompt; cap at 10
/// to bound worst-case payload size for literary novels with long scenes.
pub(crate) const DEFAULT_LOOKBACK: usize = 2;
pub(crate) const MAX_LOOKBACK: usize = 10;

pub(crate) fn clamp_lookback(opt: Option<usize>) -> usize {
    opt.unwrap_or(DEFAULT_LOOKBACK).min(MAX_LOOKBACK)
}

/// Select the N scenes immediately preceding `current` in manuscript order,
/// using the canonical `manuscript_sort_key` for stable ordering. Drops
/// Story/Arc-level items so the bundle stays focused on scene-level prose.
/// Used by both the embedded and HTTP `get_scene_context` backends so they
/// can't drift on ordering semantics.
pub(crate) fn select_preceding_scenes(
    all: Vec<crate::types::Situation>,
    current: &crate::types::Situation,
    lookback: usize,
) -> Vec<crate::types::Situation> {
    use crate::types::NarrativeLevel;

    let mut scenes: Vec<_> = all
        .into_iter()
        .filter(|s| {
            s.id != current.id
                && !matches!(
                    s.narrative_level,
                    NarrativeLevel::Story | NarrativeLevel::Arc
                )
        })
        .collect();
    scenes.sort_by_key(crate::writer::scene::manuscript_sort_key);
    let cur_key = crate::writer::scene::manuscript_sort_key(current);
    scenes.retain(|s| crate::writer::scene::manuscript_sort_key(s) < cur_key);
    let skip = scenes.len().saturating_sub(lookback);
    scenes.into_iter().skip(skip).collect()
}

/// Deep-merge `patch` into `base`. For each key in `patch`:
/// - if both sides are JSON objects, recurse;
/// - otherwise the patch value replaces the base value wholesale.
///
/// Arrays are replaced rather than appended — the writer's PATCH semantics
/// treat `themes: ["a"]` as "use exactly these themes", not "add these".
pub(crate) fn deep_merge_plan_patch(base: &mut Value, patch: &Value) {
    let (Some(base_map), Some(patch_map)) = (base.as_object_mut(), patch.as_object()) else {
        return;
    };
    for (k, patch_v) in patch_map {
        match base_map.get_mut(k) {
            Some(base_v) if base_v.is_object() && patch_v.is_object() => {
                deep_merge_plan_patch(base_v, patch_v);
            }
            _ => {
                base_map.insert(k.clone(), patch_v.clone());
            }
        }
    }
}

impl EmbeddedBackend {
    /// Delete an entity by UUID (soft delete).
    pub(crate) async fn delete_entity_impl(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        self.hypergraph().delete_entity(&uuid)?;
        Ok(ok_id(id))
    }

    /// Delete a situation by UUID (soft delete).
    pub(crate) async fn delete_situation_impl(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        self.hypergraph().delete_situation(&uuid)?;
        Ok(ok_id(id))
    }

    /// Update entity properties by UUID.
    pub(crate) async fn update_entity_impl(&self, id: &str, updates: Value) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        if !updates.is_object() {
            return Err(TensaError::InvalidQuery(
                "updates must be a JSON object".into(),
            ));
        }

        let entity = self
            .hypergraph()
            .update_entity(&uuid, |e| e.apply_patch(&updates))?;

        to_json(entity)
    }

    /// Update situation properties by UUID. Mirrors `update_entity_impl`.
    pub(crate) async fn update_situation_impl(&self, id: &str, updates: Value) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        if !updates.is_object() {
            return Err(TensaError::InvalidQuery(
                "updates must be a JSON object".into(),
            ));
        }

        let situation = self
            .hypergraph()
            .update_situation(&uuid, |s| s.apply_patch(&updates))?;

        to_json(situation)
    }

    /// Sprint P4.2 retro-enrichment — update an existing participation in place.
    /// Looks up the row by `(situation_id, entity_id, seq)`, applies a JSON
    /// patch (info_set / payoff / action / role), and writes both forward and
    /// reverse keys via `Hypergraph::update_participation`.
    pub(crate) async fn update_participation_impl(
        &self,
        situation_id: &str,
        entity_id: &str,
        seq: u16,
        updates: Value,
    ) -> Result<Value> {
        let sit_uuid = parse_uuid(situation_id)?;
        let ent_uuid = parse_uuid(entity_id)?;
        if !updates.is_object() {
            return Err(TensaError::InvalidQuery(
                "updates must be a JSON object".into(),
            ));
        }
        let pairs = self
            .hypergraph()
            .get_participations_for_pair(&ent_uuid, &sit_uuid)?;
        let mut existing = pairs
            .into_iter()
            .find(|p| p.seq == seq)
            .ok_or_else(|| {
                TensaError::NotFound(format!(
                    "participation (situation={situation_id}, entity={entity_id}, seq={seq}) not found"
                ))
            })?;

        if let Some(v) = updates.get("info_set") {
            existing.info_set = if v.is_null() {
                None
            } else {
                serde_json::from_value(v.clone()).ok()
            };
        }
        if let Some(v) = updates.get("payoff") {
            existing.payoff = if v.is_null() { None } else { Some(v.clone()) };
        }
        if let Some(v) = updates.get("action") {
            existing.action = v.as_str().map(String::from);
        }
        if let Some(v) = updates.get("role") {
            if let Ok(r) = serde_json::from_value::<crate::types::Role>(v.clone()) {
                existing.role = r;
            }
        }

        self.hypergraph().update_participation(&existing)?;
        to_json(existing)
    }

    /// List entities, optionally filtered by type, narrative, and limit.
    pub(crate) async fn list_entities_impl(
        &self,
        entity_type: Option<&str>,
        narrative_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Value> {
        let mut entities = if let Some(nar_id) = narrative_id {
            let mut ents = self.hypergraph().list_entities_by_narrative(nar_id)?;
            if let Some(type_str) = entity_type {
                let et: EntityType = type_str.parse()?;
                ents.retain(|e| e.entity_type == et);
            }
            ents
        } else if let Some(type_str) = entity_type {
            let et: EntityType = type_str.parse()?;
            self.hypergraph().list_entities_by_type(&et)?
        } else {
            let mut all = Vec::new();
            for et in &ALL_ENTITY_TYPES {
                if limit.is_some_and(|lim| all.len() >= lim) {
                    break;
                }
                all.extend(self.hypergraph().list_entities_by_type(et)?);
            }
            all
        };

        if let Some(lim) = limit {
            entities.truncate(lim);
        }

        to_json(entities)
    }

    /// Merge two entities — absorb one into the other.
    pub(crate) async fn merge_entities_impl(
        &self,
        keep_id: &str,
        absorb_id: &str,
    ) -> Result<Value> {
        let keep_uuid = parse_uuid(keep_id)?;
        let absorb_uuid = parse_uuid(absorb_id)?;
        let merged = self.hypergraph().merge_entities(&keep_uuid, &absorb_uuid)?;
        to_json(merged)
    }

    /// Export a narrative in the specified format.
    pub(crate) async fn export_narrative_impl(
        &self,
        narrative_id: &str,
        format: &str,
    ) -> Result<Value> {
        let fmt: ExportFormat = format.parse()?;
        let hg = self.hypergraph();
        let output = export::export_narrative(narrative_id, fmt, hg, false)?;
        let body_str =
            String::from_utf8(output.body).unwrap_or_else(|_| "<binary content>".to_string());
        Ok(serde_json::json!({
            "content_type": output.content_type,
            "body": body_str,
        }))
    }

    /// Get per-narrative statistics.
    pub(crate) async fn get_narrative_stats_impl(&self, narrative_id: &str) -> Result<Value> {
        let corpus = CorpusManager::new(self.store_arc());
        let stats = corpus.compute_stats(narrative_id, self.hypergraph())?;
        to_json(stats)
    }

    /// Split an entity: clone it and move specified situation participations to the clone.
    pub(crate) async fn split_entity_impl(
        &self,
        entity_id: &str,
        situation_ids: &[String],
    ) -> Result<Value> {
        let entity_uuid = parse_uuid(entity_id)?;
        let sit_uuids: Vec<Uuid> = situation_ids
            .iter()
            .map(|s| parse_uuid(s))
            .collect::<Result<Vec<_>>>()?;
        let (_source, clone) = self.hypergraph().split_entity(&entity_uuid, sit_uuids)?;
        Ok(serde_json::json!({
            "status": "ok",
            "source_id": entity_id,
            "clone_id": clone.id.to_string(),
        }))
    }

    /// Restore a soft-deleted entity by clearing `deleted_at`.
    pub(crate) async fn restore_entity_impl(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        self.hypergraph().restore_entity(&uuid)?;
        Ok(ok_id(id))
    }

    /// Restore a soft-deleted situation by clearing `deleted_at`.
    pub(crate) async fn restore_situation_impl(&self, id: &str) -> Result<Value> {
        let uuid = parse_uuid(id)?;
        self.hypergraph().restore_situation(&uuid)?;
        Ok(ok_id(id))
    }

    /// Create a project container.
    pub(crate) async fn create_project_impl(&self, data: Value) -> Result<Value> {
        let project: Project = serde_json::from_value(data)?;
        let reg = ProjectRegistry::new(self.store_arc());
        let id = reg.create(project)?;
        Ok(serde_json::json!({"id": id}))
    }

    /// Get a project by its slug ID.
    pub(crate) async fn get_project_impl(&self, id: &str) -> Result<Value> {
        let reg = ProjectRegistry::new(self.store_arc());
        let project = reg.get(id)?;
        to_json(project)
    }

    /// List all projects, optionally limited.
    pub(crate) async fn list_projects_impl(&self, limit: Option<usize>) -> Result<Value> {
        let reg = ProjectRegistry::new(self.store_arc());
        let mut projects = reg.list()?;
        if let Some(lim) = limit {
            projects.truncate(lim);
        }
        to_json(projects)
    }

    /// Update a project by its slug ID.
    pub(crate) async fn update_project_impl(&self, id: &str, updates: Value) -> Result<Value> {
        let reg = ProjectRegistry::new(self.store_arc());
        let project = reg.update(id, |p| {
            if let Some(title) = updates.get("title").and_then(|v| v.as_str()) {
                p.title = title.to_string();
            }
            if let Some(desc) = updates.get("description") {
                p.description = desc.as_str().map(|s| s.to_string());
            }
            if let Some(tags) = updates.get("tags").and_then(|v| v.as_array()) {
                p.tags = tags
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
        })?;
        to_json(project)
    }

    /// Delete a project by its slug ID, optionally cascading.
    pub(crate) async fn delete_project_impl(&self, id: &str, cascade: bool) -> Result<Value> {
        let project_reg = ProjectRegistry::new(self.store_arc());

        if cascade {
            let narrative_reg = NarrativeRegistry::new(self.store_arc());
            let narrative_ids = project_reg.list_narrative_ids(id)?;

            let mut narratives_deleted = 0u64;
            for nid in &narrative_ids {
                let _ = narrative_reg.delete(nid);
                narratives_deleted += 1;
            }

            project_reg.delete(id)?;

            Ok(serde_json::json!({
                "deleted": true,
                "cascade": {
                    "narratives_deleted": narratives_deleted,
                }
            }))
        } else {
            project_reg.delete(id)?;
            Ok(ok_id(id))
        }
    }

    /// Search entities by text query across properties.
    pub(crate) async fn search_entities_impl(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        let limit = limit.unwrap_or(20);
        let query_lower = query.to_lowercase();

        let mut results = Vec::new();
        'outer: for et in &ALL_ENTITY_TYPES {
            if results.len() >= limit {
                break;
            }
            let entities = self.hypergraph().list_entities_by_type(et)?;
            for entity in entities {
                if results.len() >= limit {
                    break 'outer;
                }
                // Check property values directly to avoid full JSON serialization
                let matched = entity
                    .properties
                    .as_object()
                    .map(|obj| {
                        obj.values().any(|v| {
                            v.as_str()
                                .map(|s| s.to_lowercase().contains(&query_lower))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if matched {
                    results.push(entity);
                }
            }
        }

        to_json(results)
    }

    // ─── Writer Workflows (Sprint W6) ────────────────────────────

    pub(crate) async fn get_narrative_plan_impl(&self, narrative_id: &str) -> Result<Value> {
        let plan = crate::narrative::plan::get_plan(self.hypergraph().store(), narrative_id)?;
        match plan {
            Some(p) => to_json(p),
            None => Ok(Value::Null),
        }
    }

    pub(crate) async fn upsert_narrative_plan_impl(
        &self,
        narrative_id: &str,
        patch: Value,
    ) -> Result<Value> {
        use crate::narrative::plan as plan_store;
        use crate::types::NarrativePlan;

        if !patch.is_object() {
            return Err(TensaError::InvalidQuery(
                "plan patch must be a JSON object".into(),
            ));
        }
        let store = self.hypergraph().store();
        let existing = plan_store::get_plan(store, narrative_id)?;

        // Start from existing plan JSON (or a default one), merge in the patch,
        // then deserialize. upsert_plan sets created_at/updated_at itself.
        let now = chrono::Utc::now();
        let mut base = match existing {
            Some(p) => {
                serde_json::to_value(p).map_err(|e| TensaError::Serialization(e.to_string()))?
            }
            None => serde_json::json!({
                "narrative_id": narrative_id,
                "created_at": now,
                "updated_at": now,
            }),
        };
        // Deep-merge the patch into the existing plan. When both sides at a
        // key are JSON objects we recurse so that e.g.
        // `{"style": {"pov": "first"}}` only updates style.pov and leaves
        // tone/tense/voice untouched. Non-object values replace wholesale —
        // patching `themes: ["a"]` still replaces the array, matching the
        // tool's documented "fields omitted are preserved" semantics.
        deep_merge_plan_patch(&mut base, &patch);
        if let Some(bm) = base.as_object_mut() {
            bm.insert(
                "narrative_id".into(),
                Value::String(narrative_id.to_string()),
            );
        }
        let plan: NarrativePlan = serde_json::from_value(base)
            .map_err(|e| TensaError::InvalidInput(format!("plan patch failed validation: {e}")))?;
        let saved = plan_store::upsert_plan(store, plan)?;
        to_json(saved)
    }

    pub(crate) async fn get_writer_workspace_impl(&self, narrative_id: &str) -> Result<Value> {
        let summary =
            crate::narrative::workspace::get_workspace_summary(self.hypergraph(), narrative_id)?;
        to_json(summary)
    }

    pub(crate) async fn run_workshop_impl(
        &self,
        narrative_id: &str,
        tier: &str,
        focuses: Option<Vec<String>>,
        max_llm_calls: Option<u32>,
    ) -> Result<Value> {
        use crate::narrative::workshop::{
            run_workshop, WorkshopFocus, WorkshopRequest, WorkshopTier,
        };

        let tier_enum = match tier.to_lowercase().as_str() {
            "cheap" => WorkshopTier::Cheap,
            "standard" => WorkshopTier::Standard,
            "deep" => WorkshopTier::Deep,
            other => {
                return Err(TensaError::InvalidQuery(format!(
                    "unknown workshop tier '{other}' — use cheap | standard | deep"
                )))
            }
        };

        let focuses_parsed: Vec<WorkshopFocus> = match focuses {
            None => WorkshopFocus::all(),
            Some(list) => list
                .iter()
                .map(|s| match s.to_lowercase().as_str() {
                    "pacing" => Ok(WorkshopFocus::Pacing),
                    "continuity" => Ok(WorkshopFocus::Continuity),
                    "characterization" => Ok(WorkshopFocus::Characterization),
                    "prose" => Ok(WorkshopFocus::Prose),
                    "structure" => Ok(WorkshopFocus::Structure),
                    other => Err(TensaError::InvalidQuery(format!(
                        "unknown workshop focus '{other}'"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        };

        let registry = NarrativeRegistry::new(self.store_arc());
        // Best-effort LLM access: Cheap tier ignores None; Standard/Deep skip
        // LLM enrichment if the provider can't do session calls.
        let extractor_arc = self.extractor_opt();
        let extractor_opt = extractor_arc.as_ref().and_then(|arc| arc.as_session());

        let request = WorkshopRequest {
            narrative_id: narrative_id.to_string(),
            tier: tier_enum,
            focuses: focuses_parsed,
            max_llm_calls,
        };
        let report = run_workshop(self.hypergraph(), &registry, extractor_opt, request)?;
        to_json(report)
    }

    pub(crate) async fn list_pinned_facts_impl(&self, narrative_id: &str) -> Result<Value> {
        let facts = crate::narrative::continuity::list_pinned_facts(
            self.hypergraph().store(),
            narrative_id,
        )?;
        to_json(facts)
    }

    pub(crate) async fn create_pinned_fact_impl(
        &self,
        narrative_id: &str,
        fact: Value,
    ) -> Result<Value> {
        use crate::types::PinnedFact;
        let obj = fact
            .as_object()
            .ok_or_else(|| TensaError::InvalidQuery("fact must be a JSON object".into()))?;
        let key = obj
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("fact.key is required".into()))?
            .to_string();
        let value = obj
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InvalidQuery("fact.value is required".into()))?
            .to_string();
        let note = obj.get("note").and_then(|v| v.as_str()).map(String::from);
        let entity_id = obj
            .get("entity_id")
            .and_then(|v| v.as_str())
            .map(|s| s.parse::<Uuid>())
            .transpose()
            .map_err(|e| TensaError::InvalidQuery(format!("bad entity_id: {e}")))?;
        let now = chrono::Utc::now();
        let parsed = PinnedFact {
            id: Uuid::nil(),
            narrative_id: narrative_id.to_string(),
            entity_id,
            key,
            value,
            note,
            created_at: now,
            updated_at: now,
        };
        let saved =
            crate::narrative::continuity::create_pinned_fact(self.hypergraph().store(), parsed)?;
        to_json(saved)
    }

    pub(crate) async fn check_continuity_impl(
        &self,
        narrative_id: &str,
        prose: &str,
    ) -> Result<Value> {
        let warnings =
            crate::narrative::continuity::check_prose(self.hypergraph(), narrative_id, prose)?;
        to_json(warnings)
    }

    pub(crate) async fn list_narrative_revisions_impl(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        let summaries = crate::narrative::revision::list_revisions_tail(
            self.hypergraph().store(),
            narrative_id,
            limit,
        )?;
        to_json(summaries)
    }

    pub(crate) async fn restore_narrative_revision_impl(
        &self,
        narrative_id: &str,
        revision_id: &str,
        author: &str,
    ) -> Result<Value> {
        let rev_id = parse_uuid(revision_id)?;
        // Guard against mismatched narrative_id: refuse if the revision
        // belongs to a different narrative than the caller claimed.
        let rev = crate::narrative::revision::get_revision(self.hypergraph().store(), &rev_id)?;
        if rev.narrative_id != narrative_id {
            return Err(TensaError::InvalidQuery(format!(
                "revision {} belongs to narrative '{}' not '{}'",
                revision_id, rev.narrative_id, narrative_id
            )));
        }
        let registry = NarrativeRegistry::new(self.store_arc());
        let report = crate::narrative::revision::restore_revision(
            self.hypergraph(),
            &registry,
            &rev_id,
            Some(author.to_string()),
        )?;
        Ok(serde_json::json!({
            "restored_from": report.restored_from.to_string(),
            "auto_commit": report.auto_commit.map(|u| u.to_string()),
            "situations_restored": report.situations_restored,
            "entities_restored": report.entities_restored,
            "participations_restored": report.participations_restored,
            "user_arcs_restored": report.user_arcs_restored,
        }))
    }

    pub(crate) async fn get_writer_cost_summary_impl(
        &self,
        narrative_id: &str,
        window: Option<&str>,
    ) -> Result<Value> {
        let window_dur = window
            .map(crate::narrative::cost_ledger::parse_window)
            .unwrap_or(None);
        let summary = crate::narrative::cost_ledger::summary(
            self.hypergraph().store(),
            narrative_id,
            window_dur,
        )?;
        to_json(summary)
    }

    pub(crate) async fn set_situation_content_impl(
        &self,
        situation_id: &str,
        content: Value,
        status: Option<&str>,
    ) -> Result<Value> {
        use crate::types::ContentBlock;

        let uuid = parse_uuid(situation_id)?;
        let blocks: Vec<ContentBlock> = match content {
            Value::String(s) => vec![ContentBlock::text(&s)],
            Value::Array(_) => parse_as(content, "content blocks")?,
            _ => {
                return Err(TensaError::InvalidInput(
                    "content must be a string or array of content blocks".into(),
                ))
            }
        };
        // `validate_raw_content` lives in the server-feature module; skip the
        // structural validation when the server feature is off (mcp-only build)
        // — the hypergraph still rejects clearly-invalid content downstream.
        #[cfg(feature = "server")]
        crate::api::routes::validate_raw_content(&blocks)?;
        let status_owned = status.map(String::from);
        let updated = self.hypergraph().update_situation(&uuid, |s| {
            s.raw_content = blocks;
            if let Some(st) = status_owned {
                s.status = Some(st);
            }
        })?;
        to_json(updated)
    }

    pub(crate) async fn get_actor_profile_impl(&self, actor_id: &str) -> Result<Value> {
        let uuid = parse_uuid(actor_id)?;
        let hg = self.hypergraph();
        let entity = hg.get_entity(&uuid)?;
        let participations = hg.get_situations_for_entity(&uuid)?;
        let state_history = hg.get_state_history(&uuid)?;

        let mut situations = Vec::new();
        for p in &participations {
            if let Ok(sit) = hg.get_situation(&p.situation_id) {
                situations.push(serde_json::json!({
                    "situation": sit,
                    "role": p.role,
                    "action": p.action,
                }));
            }
        }

        Ok(serde_json::json!({
            "entity": entity,
            "participations": situations,
            "state_history": state_history,
            "participation_count": participations.len(),
        }))
    }

    pub(crate) async fn get_scene_context_impl(
        &self,
        narrative_id: &str,
        situation_id: Option<&str>,
        pov_entity_id: Option<&str>,
        lookback_scenes: Option<usize>,
    ) -> Result<Value> {
        let hg = self.hypergraph();
        let store = hg.store();
        let lookback = clamp_lookback(lookback_scenes);

        let plan = crate::narrative::plan::get_plan(store, narrative_id)?;
        let pinned_facts = crate::narrative::continuity::list_pinned_facts(store, narrative_id)?;

        let current_sit = situation_id
            .map(|sid| -> Result<_> { hg.get_situation(&parse_uuid(sid)?) })
            .transpose()?;

        let preceding = if let Some(ref current) = current_sit {
            let all = hg.list_situations_by_narrative(narrative_id)?;
            select_preceding_scenes(all, current, lookback)
        } else {
            Vec::new()
        };

        let pov_profile = if let Some(pid) = pov_entity_id {
            let profile = self.get_actor_profile_impl(pid).await?;
            let puuid = parse_uuid(pid)?;
            let pov_pinned: Vec<_> = pinned_facts
                .iter()
                .filter(|f| f.entity_id == Some(puuid))
                .cloned()
                .collect();
            Some(serde_json::json!({
                "profile": profile,
                "pinned_facts": pov_pinned,
            }))
        } else {
            None
        };

        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "plan": plan,
            "pinned_facts": pinned_facts,
            "pov_profile": pov_profile,
            "current_situation": current_sit,
            "preceding_scenes": preceding,
            "effective_lookback": lookback,
        }))
    }

    // ─── Sprint W14: narrative architecture backends ───────────────

    pub(crate) async fn detect_commitments_impl(&self, narrative_id: &str) -> Result<Value> {
        let cs =
            crate::narrative::commitments::detect_commitments(self.hypergraph(), narrative_id)?;
        to_json(cs)
    }

    pub(crate) async fn get_commitment_rhythm_impl(&self, narrative_id: &str) -> Result<Value> {
        let r =
            crate::narrative::commitments::compute_promise_rhythm(self.hypergraph(), narrative_id)?;
        to_json(r)
    }

    pub(crate) async fn extract_fabula_impl(&self, narrative_id: &str) -> Result<Value> {
        let f = crate::narrative::fabula_sjuzet::extract_fabula(self.hypergraph(), narrative_id)?;
        to_json(f)
    }

    pub(crate) async fn extract_sjuzet_impl(&self, narrative_id: &str) -> Result<Value> {
        let s = crate::narrative::fabula_sjuzet::extract_sjuzet(self.hypergraph(), narrative_id)?;
        to_json(s)
    }

    pub(crate) async fn suggest_reordering_impl(&self, narrative_id: &str) -> Result<Value> {
        let fabula =
            crate::narrative::fabula_sjuzet::extract_fabula(self.hypergraph(), narrative_id)?;
        let sjuzet =
            crate::narrative::fabula_sjuzet::extract_sjuzet(self.hypergraph(), narrative_id)?;
        let cs = crate::narrative::fabula_sjuzet::suggest_reordering(
            &fabula,
            &sjuzet,
            self.hypergraph(),
            narrative_id,
        )?;
        to_json(cs)
    }

    pub(crate) async fn compute_dramatic_irony_impl(&self, narrative_id: &str) -> Result<Value> {
        let m = crate::narrative::dramatic_irony::compute_dramatic_irony_map(
            self.hypergraph(),
            narrative_id,
        )?;
        to_json(m)
    }

    pub(crate) async fn detect_focalization_impl(&self, narrative_id: &str) -> Result<Value> {
        let ix = crate::narrative::dramatic_irony::compute_focalization_irony_interaction(
            self.hypergraph(),
            narrative_id,
        )?;
        to_json(ix)
    }

    pub(crate) async fn detect_character_arc_impl(
        &self,
        narrative_id: &str,
        character_id: Option<&str>,
    ) -> Result<Value> {
        match character_id {
            Some(id) => {
                let uuid = parse_uuid(id)?;
                let arc = crate::narrative::character_arcs::detect_character_arc(
                    self.hypergraph(),
                    narrative_id,
                    uuid,
                )?;
                to_json(arc)
            }
            None => {
                let arcs = crate::narrative::character_arcs::list_character_arcs(
                    self.hypergraph(),
                    narrative_id,
                )?;
                to_json(arcs)
            }
        }
    }

    pub(crate) async fn detect_subplots_impl(&self, narrative_id: &str) -> Result<Value> {
        let a = crate::narrative::subplots::detect_subplots(self.hypergraph(), narrative_id)?;
        to_json(a)
    }

    pub(crate) async fn classify_scene_sequel_impl(&self, narrative_id: &str) -> Result<Value> {
        let a =
            crate::narrative::scene_sequel::analyze_scene_sequel(self.hypergraph(), narrative_id)?;
        to_json(a)
    }

    #[cfg(feature = "generation")]
    pub(crate) async fn generate_narrative_plan_impl(
        &self,
        premise: &str,
        genre: &str,
        chapter_count: usize,
        subplot_count: usize,
    ) -> Result<Value> {
        use crate::generation::planner::{generate_plan, store_plan};
        use crate::generation::types::PlanConfig;

        let config = PlanConfig {
            genre: genre.to_string(),
            chapter_count,
            protagonist_count: 1,
            subplot_count,
            commitment_density: 0.5,
            premise: premise.to_string(),
            constraints: Vec::new(),
        };
        let plan = generate_plan(config)?;
        store_plan(self.hypergraph(), &plan)?;
        to_json(plan)
    }

    #[cfg(feature = "generation")]
    pub(crate) async fn materialize_plan_impl(&self, plan_id: &str) -> Result<Value> {
        use crate::generation::materializer::materialize_plan;
        use crate::generation::planner::load_plan;

        let uuid = parse_uuid(plan_id)?;
        let plan = load_plan(self.hypergraph(), &uuid)?
            .ok_or_else(|| TensaError::NotFound(format!("plan {plan_id} not found")))?;
        let report = materialize_plan(self.hypergraph(), &plan)?;
        to_json(report)
    }

    #[cfg(feature = "generation")]
    pub(crate) async fn validate_materialized_impl(&self, narrative_id: &str) -> Result<Value> {
        use crate::generation::materializer::validate_materialized;

        let issues = validate_materialized(self.hypergraph(), narrative_id)?;
        to_json(issues)
    }

    #[cfg(feature = "generation")]
    pub(crate) async fn generate_chapter_prep_impl(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
    ) -> Result<Value> {
        use crate::generation::engine::GenerationEngine;
        use crate::generation::types::StyleTarget;

        let style = StyleTarget {
            voice_description: voice_description.map(String::from),
            ..StyleTarget::default()
        };
        let engine = GenerationEngine::new(narrative_id.to_string());
        let (prompt, chapter_result) =
            engine.prepare_chapter(self.hypergraph(), chapter, &style, &[])?;
        Ok(serde_json::json!({
            "prompt": prompt,
            "chapter": chapter_result,
        }))
    }

    #[cfg(feature = "generation")]
    pub(crate) async fn generate_narrative_prep_impl(
        &self,
        narrative_id: &str,
        chapter_count: usize,
        voice_description: Option<&str>,
    ) -> Result<Value> {
        use crate::generation::engine::GenerationEngine;
        use crate::generation::types::StyleTarget;

        let style = StyleTarget {
            voice_description: voice_description.map(String::from),
            ..StyleTarget::default()
        };
        let engine = GenerationEngine::new(narrative_id.to_string());
        let result = engine.prepare_full_narrative(self.hypergraph(), &style, chapter_count)?;
        to_json(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::backend::McpBackend;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_backend() -> EmbeddedBackend {
        let store = Arc::new(MemoryStore::new());
        EmbeddedBackend::from_store(store)
    }

    fn make_entity_data(name: &str) -> Value {
        let now = chrono::Utc::now();
        serde_json::json!({
            "id": Uuid::now_v7(),
            "entity_type": "Actor",
            "properties": {"name": name},
            "beliefs": null,
            "embedding": null,
            "narrative_id": "test-narrative",
            "maturity": "Candidate",
            "confidence": 0.9,
            "provenance": [],
            "created_at": now,
            "updated_at": now
        })
    }

    #[tokio::test]
    async fn test_delete_entity() {
        let backend = test_backend();
        let data = make_entity_data("Delete Me");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        let del = backend.delete_entity_impl(&id).await.unwrap();
        assert_eq!(del["status"], "ok");
    }

    #[tokio::test]
    async fn test_delete_entity_not_found() {
        let backend = test_backend();
        let result = backend
            .delete_entity_impl(&Uuid::now_v7().to_string())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update_entity() {
        let backend = test_backend();
        let data = make_entity_data("Original");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        let updated = backend
            .update_entity_impl(
                &id,
                serde_json::json!({"properties": {"name": "Updated"}, "confidence": 0.95}),
            )
            .await
            .unwrap();

        assert_eq!(updated["properties"]["name"], "Updated");
        let conf = updated["confidence"].as_f64().unwrap();
        assert!((conf - 0.95).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_list_entities_empty() {
        let backend = test_backend();
        let result = backend.list_entities_impl(None, None, None).await.unwrap();
        assert!(result.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_list_entities_by_type() {
        let backend = test_backend();
        backend
            .create_entity(make_entity_data("Alice"))
            .await
            .unwrap();
        backend
            .create_entity(make_entity_data("Bob"))
            .await
            .unwrap();

        let result = backend
            .list_entities_impl(Some("Actor"), None, None)
            .await
            .unwrap();
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_list_entities_with_limit() {
        let backend = test_backend();
        backend.create_entity(make_entity_data("A")).await.unwrap();
        backend.create_entity(make_entity_data("B")).await.unwrap();
        backend.create_entity(make_entity_data("C")).await.unwrap();

        let result = backend
            .list_entities_impl(None, None, Some(2))
            .await
            .unwrap();
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_list_entities_by_narrative() {
        let backend = test_backend();
        backend
            .create_entity(make_entity_data("In Narrative"))
            .await
            .unwrap();

        let result = backend
            .list_entities_impl(None, Some("test-narrative"), None)
            .await
            .unwrap();
        assert_eq!(result.as_array().unwrap().len(), 1);

        let empty = backend
            .list_entities_impl(None, Some("nonexistent"), None)
            .await
            .unwrap();
        assert!(empty.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_merge_entities() {
        let backend = test_backend();
        let a = backend
            .create_entity(make_entity_data("Keep"))
            .await
            .unwrap();
        let b = backend
            .create_entity(make_entity_data("Absorb"))
            .await
            .unwrap();
        let keep_id = a["id"].as_str().unwrap().to_string();
        let absorb_id = b["id"].as_str().unwrap().to_string();

        let merged = backend
            .merge_entities_impl(&keep_id, &absorb_id)
            .await
            .unwrap();
        assert_eq!(merged["properties"]["name"], "Keep");
    }

    #[tokio::test]
    async fn test_narrative_stats_empty() {
        let backend = test_backend();
        let stats = backend
            .get_narrative_stats_impl("nonexistent")
            .await
            .unwrap();
        assert_eq!(stats["entity_count"], 0);
    }

    #[tokio::test]
    async fn test_search_entities() {
        let backend = test_backend();
        backend
            .create_entity(make_entity_data("Raskolnikov"))
            .await
            .unwrap();
        backend
            .create_entity(make_entity_data("Sonya"))
            .await
            .unwrap();

        let result = backend
            .search_entities_impl("raskolnikov", None)
            .await
            .unwrap();
        assert_eq!(result.as_array().unwrap().len(), 1);

        let all = backend.search_entities_impl("a", None).await.unwrap();
        assert_eq!(all.as_array().unwrap().len(), 2); // both names contain 'a'
    }

    #[tokio::test]
    async fn test_search_entities_with_limit() {
        let backend = test_backend();
        backend
            .create_entity(make_entity_data("Alpha"))
            .await
            .unwrap();
        backend
            .create_entity(make_entity_data("Bravo"))
            .await
            .unwrap();
        backend
            .create_entity(make_entity_data("Charlie"))
            .await
            .unwrap();

        let result = backend.search_entities_impl("a", Some(2)).await.unwrap();
        assert!(result.as_array().unwrap().len() <= 2);
    }

    #[test]
    fn test_parse_entity_type() {
        assert!("Actor".parse::<EntityType>().is_ok());
        assert!("actor".parse::<EntityType>().is_ok());
        assert!("Location".parse::<EntityType>().is_ok());
        assert!("invalid".parse::<EntityType>().is_err());
    }

    #[tokio::test]
    async fn test_create_and_get_project() {
        let backend = test_backend();
        let now = chrono::Utc::now();
        let data = serde_json::json!({
            "id": "test-proj",
            "title": "Test Project",
            "description": "A test project",
            "tags": [],
            "narrative_count": 0,
            "created_at": now,
            "updated_at": now,
        });
        let result = backend.create_project_impl(data).await.unwrap();
        assert_eq!(result["id"], "test-proj");

        let project = backend.get_project_impl("test-proj").await.unwrap();
        assert_eq!(project["title"], "Test Project");
        assert_eq!(project["description"], "A test project");
    }

    #[tokio::test]
    async fn test_list_projects() {
        let backend = test_backend();
        let now = chrono::Utc::now();
        for name in &["alpha", "beta", "gamma"] {
            let data = serde_json::json!({
                "id": name,
                "title": name,
                "description": null,
                "tags": [],
                "narrative_count": 0,
                "created_at": now,
                "updated_at": now,
            });
            backend.create_project_impl(data).await.unwrap();
        }

        let all = backend.list_projects_impl(None).await.unwrap();
        assert_eq!(all.as_array().unwrap().len(), 3);

        let limited = backend.list_projects_impl(Some(2)).await.unwrap();
        assert_eq!(limited.as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_update_project() {
        let backend = test_backend();
        let now = chrono::Utc::now();
        let data = serde_json::json!({
            "id": "upd-proj",
            "title": "Original",
            "description": null,
            "tags": [],
            "narrative_count": 0,
            "created_at": now,
            "updated_at": now,
        });
        backend.create_project_impl(data).await.unwrap();

        let updated = backend
            .update_project_impl(
                "upd-proj",
                serde_json::json!({"title": "Updated Title", "description": "New desc"}),
            )
            .await
            .unwrap();
        assert_eq!(updated["title"], "Updated Title");
        assert_eq!(updated["description"], "New desc");
    }

    #[tokio::test]
    async fn test_delete_project() {
        let backend = test_backend();
        let now = chrono::Utc::now();
        let data = serde_json::json!({
            "id": "del-proj",
            "title": "Delete Me",
            "description": null,
            "tags": [],
            "narrative_count": 0,
            "created_at": now,
            "updated_at": now,
        });
        backend.create_project_impl(data).await.unwrap();

        let result = backend
            .delete_project_impl("del-proj", false)
            .await
            .unwrap();
        assert_eq!(result["status"], "ok");

        // Should be gone
        assert!(backend.get_project_impl("del-proj").await.is_err());
    }

    #[tokio::test]
    async fn test_get_project_not_found() {
        let backend = test_backend();
        assert!(backend.get_project_impl("nonexistent").await.is_err());
    }

    #[test]
    fn test_parse_export_format() {
        assert!("csv".parse::<crate::export::ExportFormat>().is_ok());
        assert!("JSON".parse::<crate::export::ExportFormat>().is_ok());
        assert!("manuscript".parse::<crate::export::ExportFormat>().is_ok());
        assert!("report".parse::<crate::export::ExportFormat>().is_ok());
        assert!("invalid".parse::<crate::export::ExportFormat>().is_err());
    }

    #[tokio::test]
    async fn test_update_entity_properties_merge() {
        let backend = test_backend();
        let data = make_entity_data("Alice");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        // Update should merge properties, not replace them
        let updated = backend
            .update_entity_impl(&id, serde_json::json!({"properties": {"role": "analyst"}}))
            .await
            .unwrap();

        // Original "name" should still be there
        assert_eq!(updated["properties"]["name"], "Alice");
        // New "role" should be added
        assert_eq!(updated["properties"]["role"], "analyst");
    }

    #[tokio::test]
    async fn test_update_entity_narrative_id() {
        let backend = test_backend();
        let data = make_entity_data("Bob");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        let updated = backend
            .update_entity_impl(&id, serde_json::json!({"narrative_id": "new-narrative"}))
            .await
            .unwrap();

        assert_eq!(updated["narrative_id"], "new-narrative");
    }

    #[tokio::test]
    async fn test_update_entity_confidence_f32() {
        let backend = test_backend();
        let data = make_entity_data("Charlie");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        let updated = backend
            .update_entity_impl(&id, serde_json::json!({"confidence": 0.75}))
            .await
            .unwrap();

        // Verify the confidence was stored and returned correctly
        let conf = updated["confidence"].as_f64().unwrap();
        assert!((conf - 0.75).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_update_entity_no_fields() {
        let backend = test_backend();
        let data = make_entity_data("Diana");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        // Empty update should succeed (only updated_at changes)
        let updated = backend
            .update_entity_impl(&id, serde_json::json!({}))
            .await
            .unwrap();

        assert_eq!(updated["properties"]["name"], "Diana");
    }

    #[tokio::test]
    async fn test_update_entity_not_found() {
        let backend = test_backend();
        let result = backend
            .update_entity_impl(
                &Uuid::now_v7().to_string(),
                serde_json::json!({"confidence": 0.5}),
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_situation() {
        let backend = test_backend();
        let now = chrono::Utc::now();
        let sit_data = serde_json::json!({
            "id": Uuid::now_v7(),
            "temporal": {"start": null, "end": null, "granularity": "Approximate", "relations": []},
            "spatial": null, "game_structure": null, "causes": [],
            "deterministic": null, "probabilistic": null, "embedding": null,
            "raw_content": [{"content_type": "Text", "content": "Test situation", "source": null}],
            "narrative_level": "Scene", "narrative_id": null, "discourse": null,
            "maturity": "Candidate", "confidence": 0.7,
            "extraction_method": "HumanEntered",
            "created_at": now, "updated_at": now,
        });
        let result = backend.create_situation(sit_data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        let del = backend.delete_situation_impl(&id).await.unwrap();
        assert_eq!(del["status"], "ok");
    }

    #[tokio::test]
    async fn test_delete_situation_not_found() {
        let backend = test_backend();
        let result = backend
            .delete_situation_impl(&Uuid::now_v7().to_string())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_export_narrative_json() {
        let backend = test_backend();
        // Export an empty narrative — should succeed with empty data
        let result = backend
            .export_narrative_impl("nonexistent", "json")
            .await
            .unwrap();
        assert_eq!(result["content_type"], "application/json");
        assert!(result["body"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_export_narrative_invalid_format() {
        let backend = test_backend();
        let result = backend.export_narrative_impl("test", "xml").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_update_entity_invalid_updates() {
        let backend = test_backend();
        let data = make_entity_data("Test");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        let result = backend
            .update_entity_impl(&id, serde_json::json!("not an object"))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_split_entity_via_mcp() {
        let backend = test_backend();
        // Create entity
        let data = make_entity_data("Splittable");
        let result = backend.create_entity(data).await.unwrap();
        let entity_id = result["id"].as_str().unwrap().to_string();

        // Create a situation and link it
        let now = chrono::Utc::now();
        let sit_id = Uuid::now_v7();
        let sit_data = serde_json::json!({
            "id": sit_id,
            "temporal": {"start": null, "end": null, "granularity": "Approximate", "relations": []},
            "spatial": null, "game_structure": null, "causes": [],
            "deterministic": null, "probabilistic": null, "embedding": null,
            "raw_content": [{"content_type": "Text", "content": "Test situation", "source": null}],
            "narrative_level": "Scene", "narrative_id": "test-narrative", "discourse": null,
            "maturity": "Candidate", "confidence": 0.7,
            "extraction_method": "HumanEntered",
            "created_at": now, "updated_at": now,
        });
        backend.create_situation(sit_data).await.unwrap();

        let part_data = serde_json::json!({
            "entity_id": entity_id,
            "situation_id": sit_id,
            "role": "Protagonist",
            "info_set": null,
            "action": null,
            "payoff": null,
        });
        backend.add_participant(part_data).await.unwrap();

        let split = backend
            .split_entity_impl(&entity_id, &[sit_id.to_string()])
            .await
            .unwrap();
        assert_eq!(split["status"], "ok");
        assert!(split["clone_id"].as_str().is_some());
        assert_ne!(split["clone_id"].as_str().unwrap(), entity_id);
    }

    #[tokio::test]
    async fn test_restore_entity_via_mcp() {
        let backend = test_backend();
        let data = make_entity_data("Restorable");
        let result = backend.create_entity(data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        // Delete it first
        backend.delete_entity_impl(&id).await.unwrap();

        // Restore it
        let restored = backend.restore_entity_impl(&id).await.unwrap();
        assert_eq!(restored["status"], "ok");

        // Verify we can get it again
        let entity = backend.get_entity(&id).await.unwrap();
        assert_eq!(entity["properties"]["name"], "Restorable");
    }

    #[tokio::test]
    async fn test_restore_situation_via_mcp() {
        let backend = test_backend();
        let now = chrono::Utc::now();
        let sit_data = serde_json::json!({
            "id": Uuid::now_v7(),
            "temporal": {"start": null, "end": null, "granularity": "Approximate", "relations": []},
            "spatial": null, "game_structure": null, "causes": [],
            "deterministic": null, "probabilistic": null, "embedding": null,
            "raw_content": [{"content_type": "Text", "content": "Restorable situation", "source": null}],
            "narrative_level": "Scene", "narrative_id": null, "discourse": null,
            "maturity": "Candidate", "confidence": 0.7,
            "extraction_method": "HumanEntered",
            "created_at": now, "updated_at": now,
        });
        let result = backend.create_situation(sit_data).await.unwrap();
        let id = result["id"].as_str().unwrap().to_string();

        // Delete it first
        backend.delete_situation_impl(&id).await.unwrap();

        // Restore it
        let restored = backend.restore_situation_impl(&id).await.unwrap();
        assert_eq!(restored["status"], "ok");

        // Verify we can get it again
        let sit = backend.get_situation(&id).await.unwrap();
        assert!(sit["id"].as_str().is_some());
    }

    #[test]
    fn deep_merge_preserves_sibling_keys_in_nested_objects() {
        let mut base = serde_json::json!({
            "logline": "Old logline",
            "style": {"pov": "third", "tense": "past", "tone": ["dark"]},
            "themes": ["original"]
        });
        let patch = serde_json::json!({
            "style": {"pov": "first"},
            "themes": ["updated"]
        });
        deep_merge_plan_patch(&mut base, &patch);
        // Nested object — sibling keys preserved, patched key overridden.
        assert_eq!(base["style"]["pov"], "first");
        assert_eq!(base["style"]["tense"], "past");
        assert_eq!(base["style"]["tone"][0], "dark");
        // Array / scalar — replaced wholesale.
        assert_eq!(base["themes"][0], "updated");
        assert_eq!(base["logline"], "Old logline");
    }

    #[test]
    fn deep_merge_handles_missing_sides_gracefully() {
        let mut base = serde_json::json!({"a": 1});
        deep_merge_plan_patch(&mut base, &Value::Null);
        assert_eq!(base["a"], 1);
        let mut leaf = serde_json::json!(42);
        deep_merge_plan_patch(&mut leaf, &serde_json::json!({"a": 1}));
        assert_eq!(leaf, serde_json::json!(42));
    }

    // ─── Workflow 2: co-writing loop tools ───────────────────────

    fn make_scene_data(narrative: &str, order: u32, prose: &str) -> Value {
        let now = chrono::Utc::now();
        serde_json::json!({
            "id": Uuid::now_v7(),
            "temporal": {"start": null, "end": null, "granularity": "Approximate", "relations": []},
            "spatial": null, "game_structure": null, "causes": [],
            "deterministic": null, "probabilistic": null, "embedding": null,
            "raw_content": [{"content_type": "Text", "content": prose, "source": null}],
            "narrative_level": "Scene",
            "narrative_id": narrative,
            "manuscript_order": order,
            "discourse": null,
            "maturity": "Candidate", "confidence": 0.7,
            "extraction_method": "HumanEntered",
            "created_at": now, "updated_at": now,
        })
    }

    #[tokio::test]
    async fn test_set_situation_content_string_wraps_as_text_block() {
        let backend = test_backend();
        let sit = make_scene_data("book", 1, "stub");
        let created = backend.create_situation(sit).await.unwrap();
        let id = created["id"].as_str().unwrap().to_string();

        let new_prose = "The lighthouse blinked, and she ran.";
        let updated = backend
            .set_situation_content_impl(&id, Value::String(new_prose.into()), Some("first-draft"))
            .await
            .unwrap();

        assert_eq!(updated["raw_content"][0]["content_type"], "Text");
        assert_eq!(updated["raw_content"][0]["content"], new_prose);
        assert_eq!(updated["status"], "first-draft");
    }

    #[tokio::test]
    async fn test_set_situation_content_array_preserves_block_types() {
        let backend = test_backend();
        let sit = make_scene_data("book", 1, "stub");
        let id = backend.create_situation(sit).await.unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let blocks = serde_json::json!([
            {"content_type": "Text", "content": "She hesitated.", "source": null},
            {"content_type": "Dialogue", "content": "\"Not tonight,\" he said.", "source": null}
        ]);
        let updated = backend
            .set_situation_content_impl(&id, blocks, None)
            .await
            .unwrap();

        assert_eq!(updated["raw_content"][0]["content_type"], "Text");
        assert_eq!(updated["raw_content"][1]["content_type"], "Dialogue");
        assert_eq!(updated["status"], Value::Null);
    }

    #[tokio::test]
    async fn test_set_situation_content_rejects_empty_array() {
        let backend = test_backend();
        let sit = make_scene_data("book", 1, "stub");
        let id = backend.create_situation(sit).await.unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let err = backend
            .set_situation_content_impl(&id, serde_json::json!([]), None)
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_set_situation_content_rejects_empty_prose_block() {
        let backend = test_backend();
        let sit = make_scene_data("book", 1, "stub");
        let id = backend.create_situation(sit).await.unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let err = backend
            .set_situation_content_impl(
                &id,
                serde_json::json!([{"content_type": "Text", "content": "   ", "source": null}]),
                None,
            )
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_get_scene_context_returns_preceding_scenes_in_order() {
        let backend = test_backend();
        // Three scenes, the third one is the "current" scene. Expect 2
        // preceding scenes ordered by manuscript_order.
        let s1 = backend
            .create_situation(make_scene_data("book", 1, "Chapter 1 prose"))
            .await
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let s2 = backend
            .create_situation(make_scene_data("book", 2, "Chapter 2 prose"))
            .await
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        let s3 = backend
            .create_situation(make_scene_data("book", 3, "Chapter 3 stub"))
            .await
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let ctx = backend
            .get_scene_context_impl("book", Some(&s3), None, Some(5))
            .await
            .unwrap();

        let preceding = ctx["preceding_scenes"].as_array().unwrap();
        assert_eq!(preceding.len(), 2);
        assert_eq!(preceding[0]["id"], s1);
        assert_eq!(preceding[1]["id"], s2);
        assert_eq!(ctx["narrative_id"], "book");
        assert!(ctx["current_situation"]["id"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_get_scene_context_caps_lookback_at_ten() {
        let backend = test_backend();
        for i in 1..=15_u32 {
            backend
                .create_situation(make_scene_data("book", i, "scene"))
                .await
                .unwrap();
        }
        let last = backend
            .create_situation(make_scene_data("book", 99, "current"))
            .await
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();

        let ctx = backend
            .get_scene_context_impl("book", Some(&last), None, Some(50))
            .await
            .unwrap();
        assert_eq!(ctx["preceding_scenes"].as_array().unwrap().len(), 10);
        assert_eq!(ctx["effective_lookback"], 10);
    }
}
