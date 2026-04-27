//! Narrative plan — writer's canonical plot/style/length/setting document.
//!
//! One record per narrative, stored at `np/{narrative_id}`. Included in
//! revision snapshots so style/length/plot-beat edits roll back with the rest
//! of the narrative state. Consumed by the W1 generation, W2 edit, and W3
//! workshop engines as canonical writer intent.
//!
//! # Operations
//! - [`get_plan`] — read the plan (returns `None` if never created)
//! - [`upsert_plan`] — create or full-replace
//! - [`patch_plan`] — merge a partial update into an existing plan
//! - [`delete_plan`] — hard-delete

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::hypergraph::keys;
use crate::store::KVStore;
use crate::types::{LengthTargets, NarrativePlan, PlotBeat, SettingNotes, StyleTargets};

/// Read the plan for a narrative. Returns `Ok(None)` if the narrative exists
/// but has no plan yet — callers should treat this as "plan hasn't been set".
pub fn get_plan(store: &dyn KVStore, narrative_id: &str) -> Result<Option<NarrativePlan>> {
    match store.get(&keys::narrative_plan_key(narrative_id))? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Create or fully replace the plan. Stamps `updated_at`; sets `created_at` on
/// first write, preserves it across updates.
pub fn upsert_plan(store: &dyn KVStore, mut plan: NarrativePlan) -> Result<NarrativePlan> {
    if plan.narrative_id.trim().is_empty() {
        return Err(TensaError::InvalidQuery("narrative_id is required".into()));
    }
    let now = Utc::now();
    let key = keys::narrative_plan_key(&plan.narrative_id);
    if let Some(existing) = store.get(&key)? {
        let old: NarrativePlan = serde_json::from_slice(&existing)?;
        plan.created_at = old.created_at;
    } else {
        plan.created_at = now;
    }
    plan.updated_at = now;
    let bytes = serde_json::to_vec(&plan)?;
    store.put(&key, &bytes)?;
    Ok(plan)
}

/// Delete the plan for a narrative. Idempotent — missing plan is not an error.
pub fn delete_plan(store: &dyn KVStore, narrative_id: &str) -> Result<()> {
    store.delete(&keys::narrative_plan_key(narrative_id))
}

/// Partial update descriptor. Each field is `Some(_)` when the caller wants to
/// replace it (set to `Some(null)` to clear nullable fields). Nested structs
/// are replaced wholesale; use full-replace via [`upsert_plan`] when you need
/// fine-grained sub-field control.
#[derive(Debug, Default)]
pub struct PlanPatch {
    pub logline: Option<Option<String>>,
    pub synopsis: Option<Option<String>>,
    pub premise: Option<Option<String>>,
    pub themes: Option<Vec<String>>,
    pub central_conflict: Option<Option<String>>,
    pub plot_beats: Option<Vec<PlotBeat>>,
    pub style: Option<StyleTargets>,
    pub length: Option<LengthTargets>,
    pub setting: Option<SettingNotes>,
    pub notes: Option<String>,
    pub target_audience: Option<Option<String>>,
    pub comp_titles: Option<Vec<String>>,
    pub content_warnings: Option<Vec<String>>,
    pub custom: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Apply a partial update. Creates the plan if it doesn't exist yet.
pub fn patch_plan(
    store: &dyn KVStore,
    narrative_id: &str,
    patch: PlanPatch,
) -> Result<NarrativePlan> {
    let existing = get_plan(store, narrative_id)?;
    let mut plan = existing.unwrap_or_else(|| NarrativePlan {
        narrative_id: narrative_id.to_string(),
        ..Default::default()
    });

    if let Some(v) = patch.logline {
        plan.logline = v;
    }
    if let Some(v) = patch.synopsis {
        plan.synopsis = v;
    }
    if let Some(v) = patch.premise {
        plan.premise = v;
    }
    if let Some(v) = patch.themes {
        plan.themes = v;
    }
    if let Some(v) = patch.central_conflict {
        plan.central_conflict = v;
    }
    if let Some(v) = patch.plot_beats {
        plan.plot_beats = v;
    }
    if let Some(v) = patch.style {
        plan.style = v;
    }
    if let Some(v) = patch.length {
        plan.length = v;
    }
    if let Some(v) = patch.setting {
        plan.setting = v;
    }
    if let Some(v) = patch.notes {
        plan.notes = v;
    }
    if let Some(v) = patch.target_audience {
        plan.target_audience = v;
    }
    if let Some(v) = patch.comp_titles {
        plan.comp_titles = v;
    }
    if let Some(v) = patch.content_warnings {
        plan.content_warnings = v;
    }
    if let Some(v) = patch.custom {
        plan.custom = v;
    }

    upsert_plan(store, plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn store() -> MemoryStore {
        MemoryStore::new()
    }

    #[test]
    fn get_missing_returns_none() {
        let s = store();
        assert!(get_plan(&s, "absent").unwrap().is_none());
    }

    #[test]
    fn upsert_roundtrip() {
        let s = store();
        let plan = NarrativePlan {
            narrative_id: "draft".into(),
            logline: Some("A grifter learns honesty".into()),
            themes: vec!["identity".into()],
            ..Default::default()
        };
        let saved = upsert_plan(&s, plan).unwrap();
        assert_eq!(saved.logline.as_deref(), Some("A grifter learns honesty"));
        let got = get_plan(&s, "draft").unwrap().unwrap();
        assert_eq!(got.logline, saved.logline);
    }

    #[test]
    fn upsert_preserves_created_at_and_bumps_updated_at() {
        let s = store();
        let first = upsert_plan(
            &s,
            NarrativePlan {
                narrative_id: "d".into(),
                ..Default::default()
            },
        )
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = upsert_plan(
            &s,
            NarrativePlan {
                narrative_id: "d".into(),
                logline: Some("revised".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(first.created_at, second.created_at);
        assert!(second.updated_at > first.updated_at);
    }

    #[test]
    fn patch_creates_if_missing() {
        let s = store();
        let plan = patch_plan(
            &s,
            "new",
            PlanPatch {
                logline: Some(Some("x".into())),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(plan.narrative_id, "new");
        assert_eq!(plan.logline.as_deref(), Some("x"));
    }

    #[test]
    fn patch_distinguishes_clear_from_omit() {
        let s = store();
        upsert_plan(
            &s,
            NarrativePlan {
                narrative_id: "d".into(),
                logline: Some("keep".into()),
                premise: Some("also keep".into()),
                ..Default::default()
            },
        )
        .unwrap();

        // Clear logline explicitly, don't touch premise.
        let patched = patch_plan(
            &s,
            "d",
            PlanPatch {
                logline: Some(None),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(patched.logline.is_none(), "logline should be cleared");
        assert_eq!(
            patched.premise.as_deref(),
            Some("also keep"),
            "premise untouched"
        );
    }

    #[test]
    fn patch_merges_nested_length_wholesale() {
        let s = store();
        upsert_plan(
            &s,
            NarrativePlan {
                narrative_id: "d".into(),
                length: LengthTargets {
                    kind: Some("novel".into()),
                    target_words: Some(90_000),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

        let patched = patch_plan(
            &s,
            "d",
            PlanPatch {
                length: Some(LengthTargets {
                    target_words: Some(110_000),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
        // Patch replaces nested struct wholesale — kind is now None.
        assert_eq!(patched.length.target_words, Some(110_000));
        assert!(patched.length.kind.is_none());
    }

    #[test]
    fn delete_is_idempotent() {
        let s = store();
        delete_plan(&s, "nope").unwrap();
        upsert_plan(
            &s,
            NarrativePlan {
                narrative_id: "d".into(),
                ..Default::default()
            },
        )
        .unwrap();
        delete_plan(&s, "d").unwrap();
        assert!(get_plan(&s, "d").unwrap().is_none());
    }
}
