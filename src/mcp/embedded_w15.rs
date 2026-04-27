//! Sprint W15 — EmbeddedBackend impls for the Writer MCP bridge.
//!
//! Annotations, collections, research notes, editing, revisions, workshops,
//! cost ledger, compile, templates, and secondary helpers (skeleton, dedup,
//! suggest/apply fixes, reorder). Each `*_impl` is a thin async wrapper over
//! an existing backend `pub fn` in `src/writer/`, `src/narrative/`, or
//! `src/export/`.

use std::collections::HashMap;

use serde_json::Value;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::narrative::registry::NarrativeRegistry;
use crate::writer::annotation::{self as ann, Annotation, AnnotationKind, AnnotationPatch};
use crate::writer::collection::{self as col, Collection, CollectionPatch, CollectionQuery};
use crate::writer::research::{
    self as rn, PromoteChunkRequest, ResearchNote, ResearchNoteKind, ResearchNotePatch,
};

use super::embedded::EmbeddedBackend;
use super::embedded_ext::{ok_id, parse_uuid, to_json};

fn parse_kind(s: &str) -> Result<AnnotationKind> {
    match s {
        "Comment" | "comment" => Ok(AnnotationKind::Comment),
        "Footnote" | "footnote" => Ok(AnnotationKind::Footnote),
        "Citation" | "citation" => Ok(AnnotationKind::Citation),
        other => Err(TensaError::InvalidQuery(format!(
            "unknown annotation kind '{other}' (expected Comment | Footnote | Citation)"
        ))),
    }
}

fn parse_note_kind(s: &str) -> Result<ResearchNoteKind> {
    match s {
        "Quote" | "quote" => Ok(ResearchNoteKind::Quote),
        "Clipping" | "clipping" => Ok(ResearchNoteKind::Clipping),
        "Link" | "link" => Ok(ResearchNoteKind::Link),
        "Note" | "note" => Ok(ResearchNoteKind::Note),
        other => Err(TensaError::InvalidQuery(format!(
            "unknown research-note kind '{other}' (expected Quote | Clipping | Link | Note)"
        ))),
    }
}

/// Assert that `situation_id` belongs to `narrative_id`. Used by research-note
/// and chunk-promotion flows that accept both IDs to keep the pin unambiguous.
fn check_situation_in_narrative(
    hg: &crate::hypergraph::Hypergraph,
    sid: &Uuid,
    narrative_id: &str,
) -> Result<()> {
    let sit = hg.get_situation(sid)?;
    if sit.narrative_id.as_deref() != Some(narrative_id) {
        return Err(TensaError::InvalidQuery(format!(
            "situation {sid} does not belong to narrative {narrative_id}"
        )));
    }
    Ok(())
}

fn parse_opt_uuid(s: Option<&str>) -> Result<Option<Uuid>> {
    match s {
        Some(v) => Ok(Some(parse_uuid(v)?)),
        None => Ok(None),
    }
}

impl EmbeddedBackend {
    // ─── Annotations ──────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn create_annotation_impl(
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
        let sid = parse_uuid(situation_id)?;
        // Reject unknown scenes early rather than creating a dangling index.
        self.hypergraph().get_situation(&sid)?;
        let end = span_end.max(span_start);
        let now = chrono::Utc::now();
        let a = Annotation {
            id: Uuid::nil(),
            situation_id: sid,
            kind: parse_kind(kind)?,
            span: (span_start, end),
            body: body.to_string(),
            source_id: parse_opt_uuid(source_id)?,
            chunk_id: parse_opt_uuid(chunk_id)?,
            author: author.map(String::from),
            detached: false,
            created_at: now,
            updated_at: now,
        };
        to_json(ann::create_annotation(self.hypergraph().store(), a)?)
    }

    pub(crate) async fn list_annotations_impl(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> Result<Value> {
        if let Some(sid) = situation_id {
            let uuid = parse_uuid(sid)?;
            return to_json(ann::list_annotations_for_situation(
                self.hypergraph().store(),
                &uuid,
            )?);
        }
        let nid = narrative_id.ok_or_else(|| {
            TensaError::InvalidQuery(
                "provide situation_id or narrative_id to list annotations".into(),
            )
        })?;
        // Scan all situations in the narrative, batch-fetch annotations.
        let situations = self.hypergraph().list_situations_by_narrative(nid)?;
        let scene_ids: std::collections::HashSet<Uuid> = situations.iter().map(|s| s.id).collect();
        let bucketed = ann::list_annotations_for_scenes(self.hypergraph().store(), &scene_ids)?;
        let mut flat: Vec<Annotation> = bucketed.into_values().flatten().collect();
        flat.sort_by(|a, b| {
            a.situation_id
                .cmp(&b.situation_id)
                .then(a.span.0.cmp(&b.span.0))
        });
        to_json(flat)
    }

    pub(crate) async fn update_annotation_impl(
        &self,
        annotation_id: &str,
        patch: Value,
    ) -> Result<Value> {
        let uuid = parse_uuid(annotation_id)?;
        let patch: AnnotationPatch = serde_json::from_value(patch)
            .map_err(|e| TensaError::InvalidQuery(format!("invalid annotation patch: {e}")))?;
        to_json(ann::update_annotation(
            self.hypergraph().store(),
            &uuid,
            patch,
        )?)
    }

    pub(crate) async fn delete_annotation_impl(&self, annotation_id: &str) -> Result<Value> {
        let uuid = parse_uuid(annotation_id)?;
        ann::delete_annotation(self.hypergraph().store(), &uuid)?;
        Ok(ok_id(annotation_id))
    }

    // ─── Collections ──────────────────────────────────────────

    pub(crate) async fn create_collection_impl(
        &self,
        narrative_id: &str,
        name: &str,
        description: Option<&str>,
        query: Value,
    ) -> Result<Value> {
        let q: CollectionQuery = if query.is_null() {
            CollectionQuery::default()
        } else {
            serde_json::from_value(query)
                .map_err(|e| TensaError::InvalidQuery(format!("invalid collection query: {e}")))?
        };
        let now = chrono::Utc::now();
        let c = Collection {
            id: Uuid::nil(),
            narrative_id: narrative_id.to_string(),
            name: name.to_string(),
            description: description.map(String::from),
            query: q,
            created_at: now,
            updated_at: now,
        };
        to_json(col::create_collection(self.hypergraph().store(), c)?)
    }

    pub(crate) async fn list_collections_impl(&self, narrative_id: &str) -> Result<Value> {
        to_json(col::list_collections_for_narrative(
            self.hypergraph().store(),
            narrative_id,
        )?)
    }

    pub(crate) async fn get_collection_impl(
        &self,
        collection_id: &str,
        resolve: bool,
    ) -> Result<Value> {
        let uuid = parse_uuid(collection_id)?;
        let c = col::get_collection(self.hypergraph().store(), &uuid)?;
        if resolve {
            let resolution = col::resolve_collection(self.hypergraph(), &c)?;
            return Ok(serde_json::json!({
                "collection": c,
                "resolution": resolution,
            }));
        }
        to_json(c)
    }

    pub(crate) async fn update_collection_impl(
        &self,
        collection_id: &str,
        patch: Value,
    ) -> Result<Value> {
        let uuid = parse_uuid(collection_id)?;
        let patch: CollectionPatch = serde_json::from_value(patch)
            .map_err(|e| TensaError::InvalidQuery(format!("invalid collection patch: {e}")))?;
        to_json(col::update_collection(
            self.hypergraph().store(),
            &uuid,
            patch,
        )?)
    }

    pub(crate) async fn delete_collection_impl(&self, collection_id: &str) -> Result<Value> {
        let uuid = parse_uuid(collection_id)?;
        col::delete_collection(self.hypergraph().store(), &uuid)?;
        Ok(ok_id(collection_id))
    }

    // ─── Research notes ───────────────────────────────────────

    pub(crate) async fn create_research_note_impl(
        &self,
        narrative_id: &str,
        situation_id: &str,
        kind: &str,
        body: &str,
        source_id: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        let sid = parse_uuid(situation_id)?;
        check_situation_in_narrative(self.hypergraph(), &sid, narrative_id)?;
        let now = chrono::Utc::now();
        let note = ResearchNote {
            id: Uuid::nil(),
            narrative_id: narrative_id.to_string(),
            situation_id: sid,
            kind: parse_note_kind(kind)?,
            body: body.to_string(),
            source_chunk_id: None,
            source_id: parse_opt_uuid(source_id)?,
            author: author.map(String::from),
            created_at: now,
            updated_at: now,
        };
        to_json(rn::create_research_note(self.hypergraph().store(), note)?)
    }

    pub(crate) async fn list_research_notes_impl(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> Result<Value> {
        if let Some(sid) = situation_id {
            let uuid = parse_uuid(sid)?;
            return to_json(rn::list_notes_for_situation(
                self.hypergraph().store(),
                &uuid,
            )?);
        }
        let nid = narrative_id.ok_or_else(|| {
            TensaError::InvalidQuery(
                "provide situation_id or narrative_id to list research notes".into(),
            )
        })?;
        to_json(rn::list_notes_for_narrative(
            self.hypergraph().store(),
            nid,
        )?)
    }

    pub(crate) async fn get_research_note_impl(&self, note_id: &str) -> Result<Value> {
        let uuid = parse_uuid(note_id)?;
        to_json(rn::get_research_note(self.hypergraph().store(), &uuid)?)
    }

    pub(crate) async fn update_research_note_impl(
        &self,
        note_id: &str,
        patch: Value,
    ) -> Result<Value> {
        let uuid = parse_uuid(note_id)?;
        let patch: ResearchNotePatch = serde_json::from_value(patch)
            .map_err(|e| TensaError::InvalidQuery(format!("invalid research-note patch: {e}")))?;
        to_json(rn::update_research_note(
            self.hypergraph().store(),
            &uuid,
            patch,
        )?)
    }

    pub(crate) async fn delete_research_note_impl(&self, note_id: &str) -> Result<Value> {
        let uuid = parse_uuid(note_id)?;
        rn::delete_research_note(self.hypergraph().store(), &uuid)?;
        Ok(ok_id(note_id))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn promote_chunk_to_note_impl(
        &self,
        narrative_id: &str,
        situation_id: &str,
        chunk_id: &str,
        body: &str,
        source_id: Option<&str>,
        kind: Option<&str>,
        author: Option<&str>,
    ) -> Result<Value> {
        let sid = parse_uuid(situation_id)?;
        let cid = parse_uuid(chunk_id)?;
        check_situation_in_narrative(self.hypergraph(), &sid, narrative_id)?;
        let req = PromoteChunkRequest {
            situation_id: sid,
            narrative_id: narrative_id.to_string(),
            chunk_id: cid,
            body: body.to_string(),
            source_id: parse_opt_uuid(source_id)?,
            kind: kind.map(parse_note_kind).transpose()?,
            author: author.map(String::from),
        };
        to_json(rn::promote_chunk_to_note(self.hypergraph().store(), req)?)
    }

    // ─── Editing engine ───────────────────────────────────────

    /// Resolve a writer `instruction` + optional `style_preset` into an
    /// `EditOperation`. Shared by propose / estimate so both pick the same
    /// operation kind.
    fn edit_operation(
        instruction: &str,
        style_preset: Option<&str>,
    ) -> crate::narrative::editing::EditOperation {
        use crate::narrative::editing::{EditOperation, StyleTarget};
        match style_preset {
            Some(name) => EditOperation::StyleTransfer {
                target: StyleTarget::Preset {
                    name: name.to_string(),
                },
            },
            None => EditOperation::Rewrite {
                instruction: instruction.to_string(),
            },
        }
    }

    pub(crate) async fn propose_edit_impl(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> Result<Value> {
        use crate::narrative::editing::propose_edit_for_situation;

        let sid = parse_uuid(situation_id)?;
        let extractor_arc = self.extractor_opt().ok_or_else(|| {
            TensaError::LlmError("no LLM extractor configured (see /settings/llm)".into())
        })?;
        let session = extractor_arc.as_session().ok_or_else(|| {
            TensaError::LlmError("active LLM provider does not support session-style calls".into())
        })?;
        let op = Self::edit_operation(instruction, style_preset);
        let registry = NarrativeRegistry::new(self.store_arc());
        let proposal =
            propose_edit_for_situation(self.hypergraph(), &registry, session, &sid, &op)?;
        to_json(proposal)
    }

    pub(crate) async fn apply_edit_impl(
        &self,
        proposal: Value,
        author: Option<&str>,
    ) -> Result<Value> {
        use crate::narrative::editing::{apply_edit, EditProposal};
        let proposal: EditProposal = serde_json::from_value(proposal)
            .map_err(|e| TensaError::InvalidQuery(format!("invalid EditProposal: {e}")))?;
        let registry = NarrativeRegistry::new(self.store_arc());
        let report = apply_edit(
            self.hypergraph(),
            &registry,
            proposal,
            author.map(String::from),
        )?;
        to_json(report)
    }

    pub(crate) async fn estimate_edit_tokens_impl(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> Result<Value> {
        use crate::narrative::editing::estimate_edit_tokens;
        use crate::narrative::revision::gather_snapshot;

        let sid = parse_uuid(situation_id)?;
        let situation = self.hypergraph().get_situation(&sid)?;
        let narrative_id = situation
            .narrative_id
            .as_deref()
            .ok_or_else(|| {
                TensaError::InvalidQuery("situation has no narrative_id; cannot estimate".into())
            })?
            .to_string();
        let registry = NarrativeRegistry::new(self.store_arc());
        let snapshot = gather_snapshot(self.hypergraph(), &registry, &narrative_id)?;
        let op = Self::edit_operation(instruction, style_preset);
        let estimate = estimate_edit_tokens(&situation, &snapshot, &op);
        to_json(estimate)
    }

    // ─── Revision completion ──────────────────────────────────

    pub(crate) async fn commit_narrative_revision_impl(
        &self,
        narrative_id: &str,
        message: &str,
        author: Option<&str>,
    ) -> Result<Value> {
        use crate::narrative::revision::{commit_narrative, summary_of, CommitOutcome};
        if message.trim().is_empty() {
            return Err(TensaError::InvalidQuery(
                "commit message is required".into(),
            ));
        }
        let registry = NarrativeRegistry::new(self.store_arc());
        let outcome = commit_narrative(
            self.hypergraph(),
            &registry,
            narrative_id,
            message.to_string(),
            author.map(String::from),
        )?;
        let (tag, rev) = match outcome {
            CommitOutcome::Committed(r) => ("committed", r),
            CommitOutcome::NoChange(r) => ("no_change", r),
        };
        Ok(serde_json::json!({
            "outcome": tag,
            "revision": summary_of(&rev),
        }))
    }

    pub(crate) async fn diff_narrative_revisions_impl(
        &self,
        narrative_id: &str,
        from_rev: &str,
        to_rev: &str,
    ) -> Result<Value> {
        use crate::narrative::revision::{diff_revisions, get_revision};
        let from = parse_uuid(from_rev)?;
        let to = parse_uuid(to_rev)?;
        let store = self.hypergraph().store();
        // Guard: both revisions must belong to the named narrative.
        for (label, id) in [("from", &from), ("to", &to)] {
            let rev = get_revision(store, id)?;
            if rev.narrative_id != narrative_id {
                return Err(TensaError::InvalidQuery(format!(
                    "{label} revision belongs to narrative '{}' not '{}'",
                    rev.narrative_id, narrative_id
                )));
            }
        }
        to_json(diff_revisions(store, &from, &to)?)
    }

    // ─── Workshop ─────────────────────────────────────────────

    pub(crate) async fn list_workshop_reports_impl(&self, narrative_id: &str) -> Result<Value> {
        to_json(crate::narrative::workshop::list_reports(
            self.hypergraph().store(),
            narrative_id,
        )?)
    }

    pub(crate) async fn get_workshop_report_impl(&self, report_id: &str) -> Result<Value> {
        let uuid = parse_uuid(report_id)?;
        to_json(crate::narrative::workshop::get_report(
            self.hypergraph().store(),
            &uuid,
        )?)
    }

    // ─── Cost ledger ──────────────────────────────────────────

    pub(crate) async fn list_cost_ledger_entries_impl(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> Result<Value> {
        let lim = limit.unwrap_or(50).clamp(1, 500);
        to_json(crate::narrative::cost_ledger::list(
            self.hypergraph().store(),
            narrative_id,
            lim,
        )?)
    }

    // ─── Compile ──────────────────────────────────────────────

    pub(crate) async fn list_compile_profiles_impl(&self, narrative_id: &str) -> Result<Value> {
        to_json(crate::export::compile::list_profiles_for_narrative(
            self.hypergraph().store(),
            narrative_id,
        )?)
    }

    pub(crate) async fn compile_narrative_impl(
        &self,
        narrative_id: &str,
        format: &str,
        profile_id: Option<&str>,
    ) -> Result<Value> {
        use crate::export::compile::{compile, get_profile, CompileFormat, CompileProfile};
        let fmt = match format.to_lowercase().as_str() {
            "markdown" | "md" => CompileFormat::Markdown,
            "epub" => CompileFormat::Epub,
            "docx" => CompileFormat::Docx,
            other => {
                return Err(TensaError::InvalidQuery(format!(
                    "unknown compile format '{other}' (expected markdown | epub | docx)"
                )))
            }
        };
        let data = crate::export::collect_narrative_data(narrative_id, self.hypergraph())?;
        let profile = match profile_id {
            Some(id) => get_profile(self.hypergraph().store(), &parse_uuid(id)?)?,
            None => CompileProfile {
                id: Uuid::nil(),
                narrative_id: narrative_id.to_string(),
                name: data.narrative_id.clone(),
                description: None,
                include_labels: vec![],
                exclude_labels: vec![],
                include_statuses: vec![],
                heading_templates: vec![],
                front_matter_md: None,
                back_matter_md: None,
                footnote_style: Default::default(),
                include_comments: false,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            },
        };
        let bytes = compile(self.hypergraph().store(), &data, &profile, fmt)?;
        let byte_len = bytes.len();
        let mime = match fmt {
            CompileFormat::Markdown => "text/markdown",
            CompileFormat::Epub => "application/epub+zip",
            CompileFormat::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
        };
        // Markdown is returned inline; epub/docx are base64-encoded so the
        // JSON payload stays intact.
        let (body_field, encoding): (Value, &str) = match fmt {
            CompileFormat::Markdown => (
                Value::String(
                    String::from_utf8(bytes).unwrap_or_else(|_| "<invalid-utf8>".to_string()),
                ),
                "utf-8",
            ),
            CompileFormat::Epub | CompileFormat::Docx => {
                use base64::Engine;
                (
                    Value::String(base64::engine::general_purpose::STANDARD.encode(&bytes)),
                    "base64",
                )
            }
        };
        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "format": format,
            "content_type": mime,
            "bytes": byte_len,
            "body": body_field,
            "encoding": encoding,
        }))
    }

    pub(crate) async fn upsert_compile_profile_impl(
        &self,
        narrative_id: &str,
        profile_id: Option<&str>,
        patch: Value,
    ) -> Result<Value> {
        use crate::export::compile::{
            create_profile, update_profile, CompileProfile, ProfilePatch,
        };
        match profile_id {
            Some(id) => {
                let uuid = parse_uuid(id)?;
                let patch: ProfilePatch = serde_json::from_value(patch).map_err(|e| {
                    TensaError::InvalidQuery(format!("invalid compile-profile patch: {e}"))
                })?;
                to_json(update_profile(self.hypergraph().store(), &uuid, patch)?)
            }
            None => {
                let mut p: CompileProfile = serde_json::from_value(patch).map_err(|e| {
                    TensaError::InvalidQuery(format!("invalid CompileProfile: {e}"))
                })?;
                p.id = Uuid::nil();
                p.narrative_id = narrative_id.to_string();
                p.created_at = chrono::Utc::now();
                p.updated_at = chrono::Utc::now();
                to_json(create_profile(self.hypergraph().store(), p)?)
            }
        }
    }

    // ─── Templates ────────────────────────────────────────────

    pub(crate) async fn list_narrative_templates_impl(&self) -> Result<Value> {
        use crate::narrative::templates::{builtin_templates, list_templates};
        let mut out = builtin_templates();
        let stored = list_templates(self.hypergraph())?;
        let existing: std::collections::HashSet<Uuid> = out.iter().map(|t| t.id).collect();
        for t in stored {
            if !existing.contains(&t.id) {
                out.push(t);
            }
        }
        to_json(out)
    }

    pub(crate) async fn instantiate_template_impl(
        &self,
        template_id: &str,
        bindings: HashMap<String, String>,
    ) -> Result<Value> {
        use crate::narrative::templates::{builtin_templates, instantiate_template, load_template};
        let uuid = parse_uuid(template_id)?;
        let tpl = match load_template(self.hypergraph(), &uuid)? {
            Some(t) => t,
            None => builtin_templates()
                .into_iter()
                .find(|t| t.id == uuid)
                .ok_or_else(|| TensaError::NotFound(format!("template {template_id} not found")))?,
        };
        let parsed_bindings = bindings
            .into_iter()
            .map(|(k, v)| Ok::<_, TensaError>((k, parse_uuid(&v)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        to_json(instantiate_template(&tpl, &parsed_bindings)?)
    }

    // ─── Secondary (skeleton / dedup / fixes / reorder) ───────

    pub(crate) async fn extract_narrative_skeleton_impl(
        &self,
        narrative_id: &str,
    ) -> Result<Value> {
        to_json(crate::narrative::skeleton::extract_skeleton(
            self.hypergraph(),
            narrative_id,
        )?)
    }

    pub(crate) async fn find_duplicate_candidates_impl(
        &self,
        narrative_id: &str,
        threshold: Option<f64>,
        max_candidates: Option<usize>,
    ) -> Result<Value> {
        let mut opts = crate::narrative::dedup::DedupOptions::default();
        if let Some(t) = threshold {
            opts.threshold = t as f32;
        }
        if let Some(m) = max_candidates {
            opts.max_candidates = m;
        }
        to_json(crate::narrative::dedup::find_duplicate_candidates(
            self.hypergraph(),
            narrative_id,
            &opts,
        )?)
    }

    pub(crate) async fn suggest_narrative_fixes_impl(&self, narrative_id: &str) -> Result<Value> {
        use crate::narrative::debug::{diagnose_narrative, load_diagnosis};
        use crate::narrative::debug_fixes::suggest_fixes;
        let diag = match load_diagnosis(self.hypergraph(), narrative_id)? {
            Some(d) => d,
            None => diagnose_narrative(self.hypergraph(), narrative_id)?,
        };
        let fixes = suggest_fixes(self.hypergraph(), &diag.pathologies)?;
        Ok(serde_json::json!({
            "narrative_id": narrative_id,
            "pathology_count": diag.pathologies.len(),
            "fixes": fixes,
        }))
    }

    pub(crate) async fn apply_narrative_fix_impl(
        &self,
        narrative_id: &str,
        fix: Value,
    ) -> Result<Value> {
        use crate::narrative::debug_fixes::{apply_fix, SuggestedFix};
        let fix: SuggestedFix = serde_json::from_value(fix)
            .map_err(|e| TensaError::InvalidQuery(format!("invalid SuggestedFix: {e}")))?;
        to_json(apply_fix(self.hypergraph(), narrative_id, &fix)?)
    }

    pub(crate) async fn apply_reorder_impl(
        &self,
        narrative_id: &str,
        entries: Value,
    ) -> Result<Value> {
        let entries: Vec<crate::writer::reorder::ReorderEntry> = serde_json::from_value(entries)
            .map_err(|e| TensaError::InvalidQuery(format!("invalid reorder entries: {e}")))?;
        if entries.is_empty() {
            return Err(TensaError::InvalidQuery(
                "reorder payload must contain at least one entry".into(),
            ));
        }
        to_json(crate::writer::reorder::apply_reorder(
            self.hypergraph(),
            narrative_id,
            &entries,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::backend::McpBackend;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Situation,
        TimeGranularity,
    };
    use std::sync::Arc;

    fn backend() -> EmbeddedBackend {
        let store = Arc::new(MemoryStore::new());
        EmbeddedBackend::from_store(store)
    }

    fn make_scene(narrative_id: &str) -> Situation {
        let now = chrono::Utc::now();
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some("scene".into()),
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("hello world")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative_id.into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[tokio::test]
    async fn test_annotation_crud_roundtrip() {
        let b = backend();
        let scene = make_scene("n1");
        let scene_id = scene.id;
        b.hypergraph().create_situation(scene).unwrap();

        let created = b
            .create_annotation(
                &scene_id.to_string(),
                "Comment",
                "nice line",
                0,
                5,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let ann_id = created["id"].as_str().unwrap().to_string();

        let listed = b
            .list_annotations(Some(&scene_id.to_string()), None)
            .await
            .unwrap();
        assert_eq!(listed.as_array().unwrap().len(), 1);

        let patched = b
            .update_annotation(&ann_id, serde_json::json!({"body": "edited"}))
            .await
            .unwrap();
        assert_eq!(patched["body"], "edited");

        b.delete_annotation(&ann_id).await.unwrap();
        let after = b
            .list_annotations(Some(&scene_id.to_string()), None)
            .await
            .unwrap();
        assert!(after.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_collection_crud_roundtrip() {
        let b = backend();
        let created = b
            .create_collection(
                "n1",
                "Needs revision",
                Some("scenes flagged for second pass"),
                serde_json::json!({}),
            )
            .await
            .unwrap();
        let cid = created["id"].as_str().unwrap().to_string();

        let listed = b.list_collections("n1").await.unwrap();
        assert_eq!(listed.as_array().unwrap().len(), 1);

        let got = b.get_collection(&cid, false).await.unwrap();
        assert_eq!(got["name"], "Needs revision");

        b.delete_collection(&cid).await.unwrap();
        assert!(b
            .list_collections("n1")
            .await
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn test_research_note_promote_chunk() {
        let b = backend();
        let scene = make_scene("n1");
        let scene_id = scene.id;
        b.hypergraph().create_situation(scene).unwrap();

        let chunk_id = Uuid::now_v7();
        let promoted = b
            .promote_chunk_to_note(
                "n1",
                &scene_id.to_string(),
                &chunk_id.to_string(),
                "Important quote.",
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(promoted["body"], "Important quote.");
        assert_eq!(promoted["source_chunk_id"], chunk_id.to_string());
        assert_eq!(promoted["kind"], "Quote");

        let listed = b
            .list_research_notes(Some(&scene_id.to_string()), None)
            .await
            .unwrap();
        assert_eq!(listed.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_list_narrative_templates_builtin() {
        let b = backend();
        let listed = b.list_narrative_templates().await.unwrap();
        // At least the three builtin templates.
        assert!(listed.as_array().unwrap().len() >= 3);
    }
}
