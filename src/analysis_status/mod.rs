//! Analysis-status registry — tracks which inference / LLM analyses have
//! been run for a narrative, by what source (server worker, skill, manual),
//! and whether the user has locked the result against bulk overwrite.
//!
//! KV layout:
//!   `as/{narrative_id}/{job_type}/{scope}` → JSON-encoded `AnalysisStatusEntry`
//!
//! The scope segment lets us track per-actor or per-arc analyses separately
//! from narrative-wide ones (e.g. an `ActorArcClassification` row per actor).
//! For narrative-wide analyses, scope is the literal string `"story"`.
//!
//! Two write paths:
//!   1. Automatic — `JobQueue::store_result` calls
//!      `mark_done_from_result` after a worker completes a job, producing a
//!      `Source::Auto` entry with `locked: false`.
//!   2. Skill / manual — the `tensa-narrative-llm` skill (or a human via
//!      Studio) POSTs to `/narratives/:id/analysis-status` with `Source::Skill`
//!      or `Source::Manual` and `locked: true` by default. Bulk-analysis runs
//!      check the `locked` flag and skip those entries unless `force=true`.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::store::KVStore;
use crate::types::{InferenceJobType, InferenceResult};

pub const PREFIX: &[u8] = b"as/";

/// Where this analysis came from. Drives default `locked` behavior and is
/// surfaced to the user in Studio so they can tell at a glance whether a
/// result is from the cheap server worker or from a stronger reasoning LLM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisSource {
    /// Written by the server's `WorkerPool` after a job completed.
    Auto,
    /// Written by an external skill / agent (typically a stronger LLM than
    /// the server's configured one). Defaults to `locked: true`.
    Skill,
    /// Entered or attested by a human in Studio. Defaults to `locked: true`.
    Manual,
}

impl AnalysisSource {
    pub fn default_locked(&self) -> bool {
        matches!(self, AnalysisSource::Skill | AnalysisSource::Manual)
    }
}

/// Reference to a write the analysis produced (annotation, pinned fact,
/// situation update, revision, etc.). Studio renders these as deep links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRef {
    pub kind: String,
    pub id: String,
}

/// One analysis-status row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatusEntry {
    pub narrative_id: String,
    pub job_type: InferenceJobType,
    /// `"story"` for narrative-wide; `"actor:<uuid>"` / `"arc:<uuid>"` /
    /// `"chapter:<index>"` for scoped analyses.
    pub scope: String,
    pub source: AnalysisSource,
    /// Skill name (e.g. `"tensa-narrative-llm"`) when source is `Skill`,
    /// or `None` for Auto / Manual.
    pub skill: Option<String>,
    /// Model identifier the analysis was produced with (best-effort).
    pub model: Option<String>,
    pub completed_at: DateTime<Utc>,
    /// When `true`, bulk-analysis runs MUST skip this entry unless the
    /// caller passes `force=true`. Defaults: Skill / Manual → true, Auto → false.
    pub locked: bool,
    /// One-sentence human-readable description.
    #[serde(default)]
    pub summary: Option<String>,
    /// Confidence the producer attached (0.0-1.0).
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Pointers to the actual writes.
    #[serde(default)]
    pub result_refs: Vec<ResultRef>,
}

fn entry_key(narrative_id: &str, jt: &InferenceJobType, scope: &str) -> Vec<u8> {
    let token = jt.variant_name();
    let mut k = Vec::with_capacity(PREFIX.len() + narrative_id.len() + token.len() + scope.len() + 3);
    k.extend_from_slice(PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(token.as_bytes());
    k.push(b'/');
    k.extend_from_slice(scope.as_bytes());
    k
}

fn narrative_prefix(narrative_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(PREFIX.len() + narrative_id.len() + 1);
    k.extend_from_slice(PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k
}

/// KV-backed analysis-status registry.
pub struct AnalysisStatusStore {
    store: Arc<dyn KVStore>,
}

impl AnalysisStatusStore {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Upsert one entry. Skill / Manual entries respect the caller's `locked`
    /// field; `Auto` entries that try to overwrite a `locked` row are NOT
    /// silently dropped here — that policy decision lives in the bulk-analysis
    /// caller. This function always writes.
    pub fn upsert(&self, entry: &AnalysisStatusEntry) -> Result<()> {
        let key = entry_key(&entry.narrative_id, &entry.job_type, &entry.scope);
        let bytes = serde_json::to_vec(entry)?;
        self.store.put(&key, &bytes)?;
        Ok(())
    }

    /// Get one entry. Returns `Ok(None)` if not present.
    pub fn get(
        &self,
        narrative_id: &str,
        jt: &InferenceJobType,
        scope: &str,
    ) -> Result<Option<AnalysisStatusEntry>> {
        let key = entry_key(narrative_id, jt, scope);
        match self.store.get(&key)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    /// All entries for a narrative.
    pub fn list_for_narrative(&self, narrative_id: &str) -> Result<Vec<AnalysisStatusEntry>> {
        let prefix = narrative_prefix(narrative_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut out = Vec::with_capacity(pairs.len());
        for (_k, v) in pairs {
            if let Ok(entry) = serde_json::from_slice::<AnalysisStatusEntry>(&v) {
                out.push(entry);
            }
        }
        Ok(out)
    }

    /// Toggle the `locked` flag without rewriting the rest of the entry.
    pub fn set_locked(
        &self,
        narrative_id: &str,
        jt: &InferenceJobType,
        scope: &str,
        locked: bool,
    ) -> Result<AnalysisStatusEntry> {
        let mut entry = self
            .get(narrative_id, jt, scope)?
            .ok_or_else(|| TensaError::NotFound(format!(
                "analysis-status entry {}/{:?}/{} not found",
                narrative_id, jt, scope
            )))?;
        entry.locked = locked;
        self.upsert(&entry)?;
        Ok(entry)
    }

    /// Delete one entry. Returns `true` if it existed.
    pub fn delete(
        &self,
        narrative_id: &str,
        jt: &InferenceJobType,
        scope: &str,
    ) -> Result<bool> {
        let key = entry_key(narrative_id, jt, scope);
        let existed = self.store.get(&key)?.is_some();
        if existed {
            self.store.delete(&key)?;
        }
        Ok(existed)
    }

    /// Decide whether a bulk-analysis run should submit a job for this
    /// (narrative, job_type, scope) tuple. Returns `true` to submit, `false`
    /// to skip. Locked entries always block unless `force` is set.
    pub fn should_run(
        &self,
        narrative_id: &str,
        jt: &InferenceJobType,
        scope: &str,
        force: bool,
    ) -> Result<bool> {
        if force {
            return Ok(true);
        }
        match self.get(narrative_id, jt, scope)? {
            Some(entry) if entry.locked => Ok(false),
            _ => Ok(true),
        }
    }
}

/// Best-effort: write an `Auto` entry for a freshly stored `InferenceResult`.
/// Failures are returned to the caller, which typically logs and continues
/// rather than aborting result storage. The `narrative_id` is supplied by
/// the worker (resolved from `result.target_id` or job parameters).
pub fn mark_done_from_result(
    store: &AnalysisStatusStore,
    narrative_id: &str,
    scope: &str,
    result: &InferenceResult,
) -> Result<()> {
    let entry = AnalysisStatusEntry {
        narrative_id: narrative_id.to_string(),
        job_type: result.job_type.clone(),
        scope: scope.to_string(),
        source: AnalysisSource::Auto,
        skill: None,
        model: None,
        completed_at: result.completed_at.unwrap_or_else(Utc::now),
        locked: false,
        summary: result.explanation.clone(),
        confidence: Some(result.confidence),
        result_refs: vec![ResultRef {
            kind: "inference_result".into(),
            id: result.job_id.clone(),
        }],
    };
    store.upsert(&entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::JobStatus;
    use uuid::Uuid;

    fn store() -> AnalysisStatusStore {
        AnalysisStatusStore::new(Arc::new(MemoryStore::new()))
    }

    fn entry(jt: InferenceJobType, locked: bool, source: AnalysisSource) -> AnalysisStatusEntry {
        AnalysisStatusEntry {
            narrative_id: "nar-1".into(),
            job_type: jt,
            scope: "story".into(),
            source,
            skill: None,
            model: None,
            completed_at: Utc::now(),
            locked,
            summary: None,
            confidence: None,
            result_refs: vec![],
        }
    }

    #[test]
    fn upsert_then_get_roundtrips() {
        let s = store();
        let e = entry(InferenceJobType::ArcClassification, true, AnalysisSource::Skill);
        s.upsert(&e).unwrap();
        let got = s
            .get("nar-1", &InferenceJobType::ArcClassification, "story")
            .unwrap()
            .unwrap();
        assert_eq!(got.narrative_id, "nar-1");
        assert!(got.locked);
        assert_eq!(got.source, AnalysisSource::Skill);
    }

    #[test]
    fn list_returns_all_entries_for_narrative() {
        let s = store();
        s.upsert(&entry(InferenceJobType::ArcClassification, true, AnalysisSource::Skill))
            .unwrap();
        s.upsert(&entry(InferenceJobType::CharacterArc, false, AnalysisSource::Auto))
            .unwrap();
        let mut other = entry(InferenceJobType::ArcClassification, false, AnalysisSource::Auto);
        other.narrative_id = "nar-2".into();
        s.upsert(&other).unwrap();

        let rows = s.list_for_narrative("nar-1").unwrap();
        assert_eq!(rows.len(), 2);
        let rows2 = s.list_for_narrative("nar-2").unwrap();
        assert_eq!(rows2.len(), 1);
    }

    #[test]
    fn set_locked_toggles_flag() {
        let s = store();
        s.upsert(&entry(InferenceJobType::CharacterArc, false, AnalysisSource::Auto))
            .unwrap();
        let updated = s
            .set_locked("nar-1", &InferenceJobType::CharacterArc, "story", true)
            .unwrap();
        assert!(updated.locked);
        let again = s
            .get("nar-1", &InferenceJobType::CharacterArc, "story")
            .unwrap()
            .unwrap();
        assert!(again.locked);
    }

    #[test]
    fn should_run_respects_lock_unless_forced() {
        let s = store();
        s.upsert(&entry(InferenceJobType::ArcClassification, true, AnalysisSource::Skill))
            .unwrap();
        // No row → run.
        assert!(s
            .should_run("nar-1", &InferenceJobType::CharacterArc, "story", false)
            .unwrap());
        // Locked row → skip.
        assert!(!s
            .should_run("nar-1", &InferenceJobType::ArcClassification, "story", false)
            .unwrap());
        // Locked row with force → run.
        assert!(s
            .should_run("nar-1", &InferenceJobType::ArcClassification, "story", true)
            .unwrap());
    }

    #[test]
    fn delete_removes_entry() {
        let s = store();
        s.upsert(&entry(InferenceJobType::ArcClassification, true, AnalysisSource::Skill))
            .unwrap();
        assert!(s
            .delete("nar-1", &InferenceJobType::ArcClassification, "story")
            .unwrap());
        assert!(s
            .get("nar-1", &InferenceJobType::ArcClassification, "story")
            .unwrap()
            .is_none());
    }

    #[test]
    fn mark_done_from_result_writes_auto_entry() {
        let s = store();
        let result = InferenceResult {
            job_id: "job-123".into(),
            job_type: InferenceJobType::ArcClassification,
            target_id: Uuid::now_v7(),
            result: serde_json::json!({"arc": "Cinderella"}),
            confidence: 0.82,
            explanation: Some("Detected upward fortune trajectory".into()),
            status: JobStatus::Completed,
            created_at: Utc::now(),
            completed_at: Some(Utc::now()),
        };
        mark_done_from_result(&s, "nar-1", "story", &result).unwrap();
        let row = s
            .get("nar-1", &InferenceJobType::ArcClassification, "story")
            .unwrap()
            .unwrap();
        assert_eq!(row.source, AnalysisSource::Auto);
        assert!(!row.locked);
        assert_eq!(row.confidence, Some(0.82));
        assert_eq!(row.result_refs.len(), 1);
        assert_eq!(row.result_refs[0].id, "job-123");
    }

    #[test]
    fn default_locked_per_source() {
        assert!(!AnalysisSource::Auto.default_locked());
        assert!(AnalysisSource::Skill.default_locked());
        assert!(AnalysisSource::Manual.default_locked());
    }
}
