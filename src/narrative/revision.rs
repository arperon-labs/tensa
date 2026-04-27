//! Narrative revisions — linear version control for authored narrative state.
//!
//! A revision is an immutable snapshot of everything a writer cares about in a
//! narrative: situations, entities, participations, causal links, user arcs,
//! and the narrative metadata itself. Revisions chain via `parent_id` (v1 has
//! no branches) and are stored at `rv/r/{rev_uuid}` with a narrative index at
//! `rv/n/{narrative_id}/{rev_uuid_v7}` so prefix-scans yield chronological order.
//!
//! See also `src/hypergraph/state.rs` for per-entity StateVersion (a narrower
//! per-field snapshot used by the inference pipeline) — revisions are a
//! narrative-wide layer on top.
//!
//! # Operations
//! - [`commit_narrative`] — snapshot current state + link to previous HEAD
//! - [`list_revisions`] — chronological history for the history UI
//! - [`get_revision`] — full revision including snapshot
//! - [`restore_revision`] — replace current narrative state with a revision's
//!   snapshot. Auto-commits current state first so nothing is lost.
//! - [`diff_revisions`] — prose diff (unified line) + structural counts

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::narrative::plan;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::types::Narrative;
use crate::store::KVStore;
use crate::types::*;

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ─── Snapshot gathering ────────────────────────────────────────────

/// Read the current state of a narrative into a `NarrativeSnapshot`.
pub fn gather_snapshot(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    narrative_id: &str,
) -> Result<NarrativeSnapshot> {
    let meta = registry
        .get(narrative_id)
        .ok()
        .map(|n| serde_json::to_value(&n).unwrap_or(serde_json::Value::Null));

    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;

    let mut participations: Vec<Participation> = Vec::new();
    for s in &situations {
        let ps = hypergraph.get_participants_for_situation(&s.id)?;
        participations.extend(ps);
    }

    let causal_links = situations
        .iter()
        .flat_map(|s| s.causes.iter().cloned())
        .collect();

    let user_arcs = load_user_arcs(hypergraph.store(), narrative_id)?;
    let plan = plan::get_plan(hypergraph.store(), narrative_id)?;

    Ok(NarrativeSnapshot {
        narrative_metadata: meta,
        situations,
        entities,
        participations,
        causal_links,
        user_arcs,
        plan,
    })
}

fn load_user_arcs(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<UserArc>> {
    let prefix = keys::user_arc_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    let mut arcs: Vec<UserArc> = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        arcs.push(serde_json::from_slice(&value)?);
    }
    arcs.sort_by_key(|a| a.order);
    Ok(arcs)
}

/// Canonical SHA-256 of a snapshot. Used both for commit dedup (skip if the
/// snapshot is byte-identical to the current HEAD) and for quick equality
/// checks in the history UI.
pub fn snapshot_hash(snapshot: &NarrativeSnapshot) -> Result<String> {
    // `serde_json::to_vec` is deterministic for our structs (no HashMap fields
    // at the top level, vectors preserve order). If that ever changes, switch
    // to `serde_canonical_json` or similar.
    let bytes = serde_json::to_vec(snapshot)?;
    let digest = Sha256::digest(&bytes);
    Ok(hex_encode(&digest))
}

fn total_words(snapshot: &NarrativeSnapshot) -> usize {
    snapshot
        .situations
        .iter()
        .flat_map(|s| s.raw_content.iter())
        .map(|b| b.content.split_whitespace().count())
        .sum()
}

fn counts_of(snapshot: &NarrativeSnapshot) -> RevisionCounts {
    RevisionCounts {
        situations: snapshot.situations.len(),
        entities: snapshot.entities.len(),
        participations: snapshot.participations.len(),
        causal_links: snapshot.causal_links.len(),
        user_arcs: snapshot.user_arcs.len(),
        total_words: total_words(snapshot),
    }
}

// ─── Commit ────────────────────────────────────────────────────────

/// Returned by [`commit_narrative`] so callers can distinguish a real commit
/// from a dedup-no-op.
pub enum CommitOutcome {
    /// Snapshot hash differed from HEAD — new revision written.
    Committed(NarrativeRevision),
    /// Snapshot is identical to HEAD — nothing was written. The existing HEAD
    /// revision is returned so callers have something to link to.
    NoChange(NarrativeRevision),
}

/// Commit the current narrative state as a new revision. Returns
/// `CommitOutcome::NoChange` if the state is byte-identical to the current HEAD
/// (prevents duplicate commits on idle autosaves).
pub fn commit_narrative(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    narrative_id: &str,
    message: String,
    author: Option<String>,
) -> Result<CommitOutcome> {
    // Verify the narrative exists before committing — otherwise we'd happily
    // snapshot an empty state and create phantom revisions.
    registry.get(narrative_id)?;

    let snapshot = gather_snapshot(hypergraph, registry, narrative_id)?;
    let content_hash = snapshot_hash(&snapshot)?;

    let parent = head_revision(hypergraph.store(), narrative_id)?;
    if let Some(ref head) = parent {
        if head.content_hash == content_hash {
            return Ok(CommitOutcome::NoChange(head.clone()));
        }
    }

    let revision = NarrativeRevision {
        id: Uuid::now_v7(),
        narrative_id: narrative_id.to_string(),
        parent_id: parent.as_ref().map(|p| p.id),
        message,
        author,
        created_at: Utc::now(),
        content_hash,
        snapshot,
    };

    write_revision(hypergraph.store(), &revision)?;
    Ok(CommitOutcome::Committed(revision))
}

fn write_revision(store: &dyn KVStore, revision: &NarrativeRevision) -> Result<()> {
    let key = keys::revision_key(&revision.id);
    let bytes = serde_json::to_vec(revision)?;
    store.put(&key, &bytes)?;
    let idx_key = keys::revision_narrative_index_key(&revision.narrative_id, &revision.id);
    store.put(&idx_key, &[])?;
    Ok(())
}

// ─── Read ──────────────────────────────────────────────────────────

/// Get a single revision by id (including full snapshot).
pub fn get_revision(store: &dyn KVStore, id: &Uuid) -> Result<NarrativeRevision> {
    let key = keys::revision_key(id);
    match store.get(&key)? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::NotFound(format!("revision {}", id))),
    }
}

/// List all revisions for a narrative in chronological order (oldest first).
/// Returns summaries (no full snapshot) — use [`get_revision`] for the content.
pub fn list_revisions(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<RevisionSummary>> {
    list_revisions_tail(store, narrative_id, None)
}

/// Bounded variant — when `limit` is `Some(n)` returns only the `n` most
/// recent revisions, avoiding `N - n` wasted KV reads for the ones we'd
/// discard anyway. Used by the workspace dashboard.
pub fn list_revisions_tail(
    store: &dyn KVStore,
    narrative_id: &str,
    limit: Option<usize>,
) -> Result<Vec<RevisionSummary>> {
    let prefix = keys::revision_narrative_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    // v7 UUIDs sort chronologically; take the tail only when a limit is set.
    let start = match limit {
        Some(n) => pairs.len().saturating_sub(n),
        None => 0,
    };
    let mut summaries: Vec<RevisionSummary> = Vec::with_capacity(pairs.len() - start);
    for (key, _) in pairs.into_iter().skip(start) {
        if key.len() < 16 {
            continue;
        }
        let uuid_bytes = &key[key.len() - 16..];
        let mut arr = [0u8; 16];
        arr.copy_from_slice(uuid_bytes);
        let rev_id = Uuid::from_bytes(arr);
        let rev = get_revision(store, &rev_id)?;
        summaries.push(RevisionSummary {
            id: rev.id,
            narrative_id: rev.narrative_id,
            parent_id: rev.parent_id,
            message: rev.message,
            author: rev.author,
            created_at: rev.created_at,
            content_hash: rev.content_hash,
            counts: counts_of(&rev.snapshot),
        });
    }
    Ok(summaries)
}

fn head_revision(store: &dyn KVStore, narrative_id: &str) -> Result<Option<NarrativeRevision>> {
    let prefix = keys::revision_narrative_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    match pairs.last() {
        Some((key, _)) if key.len() >= 16 => {
            let uuid_bytes = &key[key.len() - 16..];
            let mut arr = [0u8; 16];
            arr.copy_from_slice(uuid_bytes);
            let rev_id = Uuid::from_bytes(arr);
            Ok(Some(get_revision(store, &rev_id)?))
        }
        _ => Ok(None),
    }
}

// ─── Restore ───────────────────────────────────────────────────────

#[derive(Debug)]
pub struct RestoreReport {
    /// The revision that is now HEAD after the restore.
    pub restored_from: Uuid,
    /// The auto-commit created to preserve pre-restore state (`None` if the
    /// working state was byte-identical to HEAD and no commit was written).
    pub auto_commit: Option<Uuid>,
    /// Elements re-created from the snapshot.
    pub situations_restored: usize,
    pub entities_restored: usize,
    pub participations_restored: usize,
    pub user_arcs_restored: usize,
}

/// Replace the narrative's current state with a revision's snapshot.
///
/// Before restoring, auto-commits the current state with message
/// `"auto-commit before restore to {rev_id}"` so the writer never loses work
/// to a restore click.
pub fn restore_revision(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    rev_id: &Uuid,
    author: Option<String>,
) -> Result<RestoreReport> {
    let revision = get_revision(hypergraph.store(), rev_id)?;
    let narrative_id = revision.narrative_id.clone();

    // Safety commit of current state.
    let auto = commit_narrative(
        hypergraph,
        registry,
        &narrative_id,
        format!("auto-commit before restore to {}", rev_id),
        author,
    )?;
    let auto_commit = match auto {
        CommitOutcome::Committed(r) => Some(r.id),
        CommitOutcome::NoChange(_) => None,
    };

    wipe_narrative_state(hypergraph, &narrative_id)?;
    apply_snapshot(hypergraph, registry, &narrative_id, &revision.snapshot)?;

    Ok(RestoreReport {
        restored_from: revision.id,
        auto_commit,
        situations_restored: revision.snapshot.situations.len(),
        entities_restored: revision.snapshot.entities.len(),
        participations_restored: revision.snapshot.participations.len(),
        user_arcs_restored: revision.snapshot.user_arcs.len(),
    })
}

fn wipe_narrative_state(hypergraph: &Hypergraph, narrative_id: &str) -> Result<()> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    for s in situations {
        hypergraph.hard_delete_situation(&s.id)?;
    }
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    for e in entities {
        hypergraph.hard_delete_entity(&e.id)?;
    }
    // Wipe user arcs (prefix scan; one transaction would be nicer but the
    // narrative is expected to be small).
    let arc_prefix = keys::user_arc_prefix(narrative_id);
    let arc_pairs = hypergraph.store().prefix_scan(&arc_prefix)?;
    for (key, _) in arc_pairs {
        hypergraph.store().delete(&key)?;
    }
    // Wipe the plan — it'll be re-applied from the snapshot if present.
    plan::delete_plan(hypergraph.store(), narrative_id)?;
    Ok(())
}

fn apply_snapshot(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    narrative_id: &str,
    snapshot: &NarrativeSnapshot,
) -> Result<()> {
    if let Some(meta_val) = &snapshot.narrative_metadata {
        if let Ok(meta) = serde_json::from_value::<Narrative>(meta_val.clone()) {
            // Keep the existing narrative record; just roll back the mutable
            // fields. If the narrative has been deleted since, recreate it.
            if registry.get(narrative_id).is_ok() {
                let tags = meta.tags.clone();
                let authors = meta.authors.clone();
                let custom_props = meta.custom_properties.clone();
                registry.update(narrative_id, |n| {
                    n.title = meta.title;
                    n.genre = meta.genre;
                    n.description = meta.description;
                    n.authors = authors;
                    n.language = meta.language;
                    n.publication_date = meta.publication_date;
                    n.cover_url = meta.cover_url;
                    n.tags = tags;
                    n.custom_properties = custom_props;
                })?;
            } else {
                registry.create(meta)?;
            }
        }
    }

    for entity in &snapshot.entities {
        hypergraph.create_entity(entity.clone())?;
    }
    for situation in &snapshot.situations {
        hypergraph.create_situation(situation.clone())?;
    }
    for p in &snapshot.participations {
        // `add_participant` auto-assigns seq; pre-restored seq values come
        // from the snapshot so we use a lower-level path to preserve them.
        // Use `add_participant` in v1 and accept seq renumbering.
        hypergraph.add_participant(p.clone())?;
    }
    for arc in &snapshot.user_arcs {
        let key = keys::user_arc_key(narrative_id, &arc.id);
        let bytes = serde_json::to_vec(arc)?;
        hypergraph.store().put(&key, &bytes)?;
    }
    if let Some(ref p) = snapshot.plan {
        let mut p = p.clone();
        p.narrative_id = narrative_id.to_string();
        plan::upsert_plan(hypergraph.store(), p)?;
    }

    Ok(())
}

// ─── Diff ──────────────────────────────────────────────────────────

/// Structural + textual diff between two revisions.
#[derive(Debug, Serialize, Deserialize)]
pub struct RevisionDiff {
    pub from: RevisionSummary,
    pub to: RevisionSummary,
    pub structure: StructuralDiff,
    /// Unified prose diff, chapter by chapter. Each entry is the reconstructed
    /// manuscript for one "section" (name-keyed situation) with diff hunks.
    pub prose: Vec<ProseDiffHunk>,
    /// Sprint W14 — per-scene compact summary (lines added/removed, word delta)
    /// computed from the same hunks. Makes large diffs scannable without having
    /// to unroll every line.
    #[serde(default)]
    pub scene_summaries: Vec<SceneDiffSummary>,
}

/// Per-scene diff summary (Sprint W14).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDiffSummary {
    pub situation_id: Option<Uuid>,
    pub header: String,
    pub lines_added: u32,
    pub lines_removed: u32,
    pub word_delta: i64,
    pub change_kind: SceneChangeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneChangeKind {
    Added,
    Removed,
    Modified,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StructuralDiff {
    pub situations_added: Vec<String>,
    pub situations_removed: Vec<String>,
    pub situations_changed: Vec<String>,
    pub entities_added: Vec<String>,
    pub entities_removed: Vec<String>,
    pub participations_delta: i64,
    pub causal_links_delta: i64,
    pub arcs_delta: i64,
    pub word_delta: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProseDiffHunk {
    pub situation_id: Option<Uuid>,
    /// Name shown in the diff header — e.g. "Chapter 1" or a situation's name.
    pub header: String,
    pub lines: Vec<DiffLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "text")]
pub enum DiffLine {
    Context(String),
    Added(String),
    Removed(String),
}

/// Compute a structural + prose diff between two revisions.
pub fn diff_revisions(store: &dyn KVStore, from: &Uuid, to: &Uuid) -> Result<RevisionDiff> {
    let a = get_revision(store, from)?;
    let b = get_revision(store, to)?;

    let structure = structural_diff(&a.snapshot, &b.snapshot);
    let prose = prose_diff(&a.snapshot, &b.snapshot);
    let scene_summaries = build_scene_summaries(&a.snapshot, &b.snapshot, &prose);

    Ok(RevisionDiff {
        from: RevisionSummary {
            id: a.id,
            narrative_id: a.narrative_id,
            parent_id: a.parent_id,
            message: a.message,
            author: a.author,
            created_at: a.created_at,
            content_hash: a.content_hash,
            counts: counts_of(&a.snapshot),
        },
        to: RevisionSummary {
            id: b.id,
            narrative_id: b.narrative_id,
            parent_id: b.parent_id,
            message: b.message,
            author: b.author,
            created_at: b.created_at,
            content_hash: b.content_hash,
            counts: counts_of(&b.snapshot),
        },
        structure,
        prose,
        scene_summaries,
    })
}

fn build_scene_summaries(
    a: &NarrativeSnapshot,
    b: &NarrativeSnapshot,
    hunks: &[ProseDiffHunk],
) -> Vec<SceneDiffSummary> {
    let a_sits: HashMap<Uuid, &Situation> = a.situations.iter().map(|s| (s.id, s)).collect();
    let b_sits: HashMap<Uuid, &Situation> = b.situations.iter().map(|s| (s.id, s)).collect();
    let mut out = Vec::with_capacity(hunks.len());
    for h in hunks {
        let id = match h.situation_id {
            Some(id) => id,
            None => continue,
        };
        let mut added_lines = 0u32;
        let mut removed_lines = 0u32;
        for line in &h.lines {
            match line {
                DiffLine::Added(_) => added_lines += 1,
                DiffLine::Removed(_) => removed_lines += 1,
                DiffLine::Context(_) => {}
            }
        }
        let a_wc = a_sits
            .get(&id)
            .map(|s| crate::writer::scene::word_count(s))
            .unwrap_or(0);
        let b_wc = b_sits
            .get(&id)
            .map(|s| crate::writer::scene::word_count(s))
            .unwrap_or(0);
        let word_delta = b_wc as i64 - a_wc as i64;
        let change_kind = if !a_sits.contains_key(&id) && b_sits.contains_key(&id) {
            SceneChangeKind::Added
        } else if a_sits.contains_key(&id) && !b_sits.contains_key(&id) {
            SceneChangeKind::Removed
        } else {
            SceneChangeKind::Modified
        };
        out.push(SceneDiffSummary {
            situation_id: Some(id),
            header: h.header.clone(),
            lines_added: added_lines,
            lines_removed: removed_lines,
            word_delta,
            change_kind,
        });
    }
    out
}

fn structural_diff(a: &NarrativeSnapshot, b: &NarrativeSnapshot) -> StructuralDiff {
    let a_sits: HashMap<Uuid, &Situation> = a.situations.iter().map(|s| (s.id, s)).collect();
    let b_sits: HashMap<Uuid, &Situation> = b.situations.iter().map(|s| (s.id, s)).collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    for (id, bs) in &b_sits {
        match a_sits.get(id) {
            None => added.push(bs.name.clone().unwrap_or_else(|| id.to_string())),
            Some(as_) => {
                if situation_text(as_) != situation_text(bs)
                    || as_.name != bs.name
                    || as_.description != bs.description
                    || as_.narrative_level != bs.narrative_level
                {
                    changed.push(bs.name.clone().unwrap_or_else(|| id.to_string()));
                }
            }
        }
    }
    for (id, as_) in &a_sits {
        if !b_sits.contains_key(id) {
            removed.push(as_.name.clone().unwrap_or_else(|| id.to_string()));
        }
    }

    let a_ents: HashSet<Uuid> = a.entities.iter().map(|e| e.id).collect();
    let b_ents: HashSet<Uuid> = b.entities.iter().map(|e| e.id).collect();
    let entities_added: Vec<String> = b
        .entities
        .iter()
        .filter(|e| !a_ents.contains(&e.id))
        .map(|e| entity_label(e))
        .collect();
    let entities_removed: Vec<String> = a
        .entities
        .iter()
        .filter(|e| !b_ents.contains(&e.id))
        .map(|e| entity_label(e))
        .collect();

    StructuralDiff {
        situations_added: added,
        situations_removed: removed,
        situations_changed: changed,
        entities_added,
        entities_removed,
        participations_delta: b.participations.len() as i64 - a.participations.len() as i64,
        causal_links_delta: b.causal_links.len() as i64 - a.causal_links.len() as i64,
        arcs_delta: b.user_arcs.len() as i64 - a.user_arcs.len() as i64,
        word_delta: total_words(b) as i64 - total_words(a) as i64,
    }
}

fn entity_label(e: &Entity) -> String {
    e.properties
        .get("name")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| e.id.to_string())
}

/// Line-level diff of each situation's prose (concatenated `raw_content`).
/// Situations are keyed by id so renames don't destroy the matching; the
/// header shows the `b` name (or the `a` name if the situation was removed).
fn prose_diff(a: &NarrativeSnapshot, b: &NarrativeSnapshot) -> Vec<ProseDiffHunk> {
    let a_sits: HashMap<Uuid, &Situation> = a.situations.iter().map(|s| (s.id, s)).collect();
    let b_sits: HashMap<Uuid, &Situation> = b.situations.iter().map(|s| (s.id, s)).collect();
    let mut all_ids: Vec<Uuid> = a_sits.keys().chain(b_sits.keys()).copied().collect();
    all_ids.sort_by_key(|id| {
        // Sort by temporal.start of whichever side has it (b takes precedence).
        let s = b_sits.get(id).or_else(|| a_sits.get(id));
        s.and_then(|s| s.temporal.start)
            .map(|t| t.timestamp_millis())
            .unwrap_or(i64::MAX)
    });
    all_ids.dedup();

    let mut hunks = Vec::new();
    for id in all_ids {
        let left = a_sits
            .get(&id)
            .map(|s| situation_text(s))
            .unwrap_or_default();
        let right = b_sits
            .get(&id)
            .map(|s| situation_text(s))
            .unwrap_or_default();
        if left == right {
            continue;
        }
        let header = b_sits
            .get(&id)
            .or_else(|| a_sits.get(&id))
            .and_then(|s| s.name.clone())
            .unwrap_or_else(|| format!("(untitled {})", id));
        let lines = unified_line_diff(&left, &right);
        if !lines.is_empty() {
            hunks.push(ProseDiffHunk {
                situation_id: Some(id),
                header,
                lines,
            });
        }
    }
    hunks
}

fn situation_text(s: &Situation) -> String {
    s.raw_content
        .iter()
        .map(|b| b.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Line-level LCS diff. Simple O(n*m) dynamic programming — plenty fast for
/// per-chapter prose (typically <1k lines per chapter). Shared with the edit
/// engine so preview diffs and history diffs render identically.
pub fn unified_line_diff(a: &str, b: &str) -> Vec<DiffLine> {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let n = a_lines.len();
    let m = b_lines.len();

    // Build LCS table
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if a_lines[i - 1] == b_lines[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Backtrack
    let mut out: Vec<DiffLine> = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        if a_lines[i - 1] == b_lines[j - 1] {
            out.push(DiffLine::Context(a_lines[i - 1].to_string()));
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            out.push(DiffLine::Removed(a_lines[i - 1].to_string()));
            i -= 1;
        } else {
            out.push(DiffLine::Added(b_lines[j - 1].to_string()));
            j -= 1;
        }
    }
    while i > 0 {
        out.push(DiffLine::Removed(a_lines[i - 1].to_string()));
        i -= 1;
    }
    while j > 0 {
        out.push(DiffLine::Added(b_lines[j - 1].to_string()));
        j -= 1;
    }
    out.reverse();
    out
}

// ─── Helper for route layer: build summary from revision ───────────

pub fn summary_of(rev: &NarrativeRevision) -> RevisionSummary {
    RevisionSummary {
        id: rev.id,
        narrative_id: rev.narrative_id.clone(),
        parent_id: rev.parent_id,
        message: rev.message.clone(),
        author: rev.author.clone(),
        created_at: rev.created_at,
        content_hash: rev.content_hash.clone(),
        counts: counts_of(&rev.snapshot),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc as StdArc;

    fn setup() -> (Hypergraph, NarrativeRegistry) {
        let store = StdArc::new(MemoryStore::new());
        let hypergraph = Hypergraph::new(store.clone());
        let registry = NarrativeRegistry::new(store);
        registry
            .create(Narrative {
                id: "draft".into(),
                title: "Draft".into(),
                genre: None,
                tags: vec![],
                description: None,
                authors: vec![],
                language: None,
                publication_date: None,
                cover_url: None,
                custom_properties: std::collections::HashMap::new(),
                entity_count: 0,
                situation_count: 0,
                source: None,
                project_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
            .unwrap();
        (hypergraph, registry)
    }

    fn mk_situation(name: &str, text: &str) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(name.into()),
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: None,
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock {
                content_type: ContentType::Text,
                content: text.into(),
                source: None,
            }],
            narrative_level: NarrativeLevel::Arc,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("draft".into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[test]
    fn empty_narrative_commits_empty_snapshot() {
        let (hg, reg) = setup();
        let out = commit_narrative(&hg, &reg, "draft", "init".into(), None).unwrap();
        let rev = match out {
            CommitOutcome::Committed(r) => r,
            CommitOutcome::NoChange(_) => panic!("expected a real commit"),
        };
        assert_eq!(rev.snapshot.situations.len(), 0);
        assert!(rev.parent_id.is_none());
    }

    #[test]
    fn commit_chain_parents_correctly() {
        let (hg, reg) = setup();
        let first = commit_for("first", &hg, &reg);
        hg.create_situation(mk_situation("Ch1", "opening line"))
            .unwrap();
        let second = commit_for("second", &hg, &reg);
        assert_eq!(second.parent_id, Some(first.id));
    }

    #[test]
    fn identical_state_deduped() {
        let (hg, reg) = setup();
        let _ = commit_for("first", &hg, &reg);
        let out = commit_narrative(&hg, &reg, "draft", "dup".into(), None).unwrap();
        assert!(matches!(out, CommitOutcome::NoChange(_)));
    }

    #[test]
    fn list_revisions_is_chronological() {
        let (hg, reg) = setup();
        let a = commit_for("a", &hg, &reg);
        hg.create_situation(mk_situation("Ch1", "text")).unwrap();
        let b = commit_for("b", &hg, &reg);
        let list = list_revisions(hg.store(), "draft").unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, a.id);
        assert_eq!(list[1].id, b.id);
    }

    #[test]
    fn restore_replaces_state_and_autocommits() {
        let (hg, reg) = setup();
        hg.create_situation(mk_situation("Ch1", "original"))
            .unwrap();
        let original = commit_for("original", &hg, &reg);

        // Mutate: add a new chapter.
        hg.create_situation(mk_situation("Ch2", "added later"))
            .unwrap();
        assert_eq!(hg.list_situations_by_narrative("draft").unwrap().len(), 2);

        let report = restore_revision(&hg, &reg, &original.id, None).unwrap();

        // Should be back to 1 situation.
        assert_eq!(hg.list_situations_by_narrative("draft").unwrap().len(), 1);
        // Auto-commit should have been created since state changed before restore.
        assert!(report.auto_commit.is_some());
        // History should now have: original, auto-commit, (restore itself does NOT commit).
        let list = list_revisions(hg.store(), "draft").unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn plan_snapshots_and_restores_with_revision() {
        use crate::narrative::plan;
        let (hg, reg) = setup();
        plan::upsert_plan(
            hg.store(),
            NarrativePlan {
                narrative_id: "draft".into(),
                logline: Some("v1 logline".into()),
                length: LengthTargets {
                    target_words: Some(80_000),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let first = commit_for("with v1 plan", &hg, &reg);
        assert_eq!(
            first
                .snapshot
                .plan
                .as_ref()
                .and_then(|p| p.logline.as_deref()),
            Some("v1 logline")
        );

        // Mutate plan.
        plan::upsert_plan(
            hg.store(),
            NarrativePlan {
                narrative_id: "draft".into(),
                logline: Some("v2 logline".into()),
                length: LengthTargets {
                    target_words: Some(120_000),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let _ = commit_for("with v2 plan", &hg, &reg);

        // Restore to first revision — plan should revert.
        restore_revision(&hg, &reg, &first.id, None).unwrap();
        let restored = plan::get_plan(hg.store(), "draft").unwrap().unwrap();
        assert_eq!(restored.logline.as_deref(), Some("v1 logline"));
        assert_eq!(restored.length.target_words, Some(80_000));
    }

    #[test]
    fn diff_reports_added_and_changed() {
        let (hg, reg) = setup();
        hg.create_situation(mk_situation("Ch1", "first line\nsecond line"))
            .unwrap();
        let a = commit_for("a", &hg, &reg);

        let sit_id_before = hg.list_situations_by_narrative("draft").unwrap()[0].id;
        let mut sit = hg.get_situation(&sit_id_before).unwrap();
        sit.raw_content[0].content = "first line\nmodified second\nadded third".into();
        sit.updated_at = Utc::now();
        let key = keys::situation_key(&sit.id);
        hg.store()
            .put(&key, &serde_json::to_vec(&sit).unwrap())
            .unwrap();

        hg.create_situation(mk_situation("Ch2", "brand new"))
            .unwrap();
        let b = commit_for("b", &hg, &reg);

        let d = diff_revisions(hg.store(), &a.id, &b.id).unwrap();
        assert_eq!(d.structure.situations_added.len(), 1);
        assert_eq!(d.structure.situations_changed.len(), 1);
        assert!(d.prose.iter().any(|h| h.header == "Ch1"));
        assert!(d.prose.iter().any(|h| h.header == "Ch2"));
        // The Ch1 hunk should have both an Added and Removed line for the edit.
        let ch1 = d.prose.iter().find(|h| h.header == "Ch1").unwrap();
        assert!(ch1.lines.iter().any(|l| matches!(l, DiffLine::Added(_))));
        assert!(ch1.lines.iter().any(|l| matches!(l, DiffLine::Removed(_))));
    }

    #[test]
    fn diff_scene_summaries_classify_and_count() {
        let (hg, reg) = setup();
        hg.create_situation(mk_situation("Ch1", "one two three"))
            .unwrap();
        let a = commit_for("a", &hg, &reg);

        // Modify Ch1; add a new Ch2.
        let sit_id = hg.list_situations_by_narrative("draft").unwrap()[0].id;
        let mut sit = hg.get_situation(&sit_id).unwrap();
        sit.raw_content[0].content = "one two three four five six".into();
        sit.updated_at = Utc::now();
        let key = keys::situation_key(&sit.id);
        hg.store()
            .put(&key, &serde_json::to_vec(&sit).unwrap())
            .unwrap();
        hg.create_situation(mk_situation("Ch2", "entirely new prose"))
            .unwrap();
        let b = commit_for("b", &hg, &reg);

        let d = diff_revisions(hg.store(), &a.id, &b.id).unwrap();
        assert!(!d.scene_summaries.is_empty());
        let ch1 = d
            .scene_summaries
            .iter()
            .find(|s| s.header == "Ch1")
            .expect("Ch1 summary");
        assert_eq!(ch1.change_kind, SceneChangeKind::Modified);
        assert!(ch1.word_delta > 0);
        let ch2 = d
            .scene_summaries
            .iter()
            .find(|s| s.header == "Ch2")
            .expect("Ch2 summary");
        assert_eq!(ch2.change_kind, SceneChangeKind::Added);
    }

    fn commit_for(msg: &str, hg: &Hypergraph, reg: &NarrativeRegistry) -> NarrativeRevision {
        let out = commit_narrative(hg, reg, "draft", msg.into(), None).unwrap();
        match out {
            CommitOutcome::Committed(r) => r,
            CommitOutcome::NoChange(_) => panic!("expected commit"),
        }
    }
}
