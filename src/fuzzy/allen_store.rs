//! Fuzzy Sprint Phase 5 — persistence, query helpers, and reconciler
//! heuristics that complement `src/fuzzy/allen.rs`. Split out to keep
//! `allen.rs` below the 500-line file cap.
//!
//! Cites: [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen].

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::Result;
use crate::fuzzy::allen::{
    fuzzy_relation_holds, graded_relation_value, FuzzyEndpoints, TrapezoidalFuzzy,
};
use crate::store::KVStore;
use crate::types::{AllenInterval, AllenRelation, TimeGranularity};

// ── KV persistence ──────────────────────────────────────────────────────────

/// Save a graded 13-vector at `fz/allen/{narrative_id}/{a_id}/{b_id}`.
pub fn save_fuzzy_allen(
    store: &dyn KVStore,
    narrative_id: &str,
    a_id: &Uuid,
    b_id: &Uuid,
    vector: &[f64; 13],
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_allen(narrative_id, a_id, b_id);
    let bytes = serde_json::to_vec(vector)?;
    store.put(&key, &bytes)
}

/// Load a cached 13-vector if present.
pub fn load_fuzzy_allen(
    store: &dyn KVStore,
    narrative_id: &str,
    a_id: &Uuid,
    b_id: &Uuid,
) -> Result<Option<[f64; 13]>> {
    let key = crate::fuzzy::key_fuzzy_allen(narrative_id, a_id, b_id);
    match store.get(&key)? {
        Some(bytes) => {
            let v: [f64; 13] = serde_json::from_slice(&bytes)?;
            Ok(Some(v))
        }
        None => Ok(None),
    }
}

/// Delete a cached 13-vector (idempotent).
pub fn delete_fuzzy_allen(
    store: &dyn KVStore,
    narrative_id: &str,
    a_id: &Uuid,
    b_id: &Uuid,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_allen(narrative_id, a_id, b_id);
    store.delete(&key)
}

/// Invalidate the cached pair (both directions).
pub fn invalidate_pair(
    store: &dyn KVStore,
    narrative_id: &str,
    a_id: &Uuid,
    b_id: &Uuid,
) -> Result<()> {
    delete_fuzzy_allen(store, narrative_id, a_id, b_id)?;
    delete_fuzzy_allen(store, narrative_id, b_id, a_id)
}

// ── Reconciler heuristic — keyword → trapezoid ──────────────────────────────

/// Pipeline-side heuristic: widen a crisp kernel into a ±10%
/// trapezoidal window when the free-text marker carries a fuzziness
/// cue. Returns `None` when no cue fires; callers keep the crisp
/// assignment unchanged in that case. Phase 5.5 may upgrade this to
/// LLM-extracted explicit bounds.
pub fn fuzzy_from_marker(
    kernel_start: DateTime<Utc>,
    kernel_end: DateTime<Utc>,
    description: &str,
) -> Option<FuzzyEndpoints> {
    const FUZZINESS_CUES: &[&str] = &[
        "shortly",
        "around",
        "about",
        "early",
        "late",
        "approximately",
    ];
    let haystack = description.to_lowercase();
    if !FUZZINESS_CUES.iter().any(|cue| haystack.contains(cue)) {
        return None;
    }
    let duration = kernel_end - kernel_start;
    let seed = if duration.num_nanoseconds().unwrap_or(0) == 0 {
        chrono::Duration::hours(1)
    } else {
        duration
    };
    let widen = seed / 10;
    let start_trapezoid = TrapezoidalFuzzy::new(
        kernel_start - widen,
        kernel_start,
        kernel_start,
        kernel_start + widen,
    )
    .ok()?;
    let end_trapezoid = TrapezoidalFuzzy::new(
        kernel_end - widen,
        kernel_end,
        kernel_end,
        kernel_end + widen,
    )
    .ok()?;
    Some(FuzzyEndpoints::from_pair(start_trapezoid, end_trapezoid))
}

// ── IntervalTree graded query ───────────────────────────────────────────────

/// Retrieve situations whose graded-Allen degree for `rel` against the
/// reference interval meets `threshold`. In-memory iteration; no KV
/// cache lookup. Caller can compose with [`load_fuzzy_allen`] if they
/// want the cached-first behaviour.
pub fn fuzzy_relation_query_situations(
    store: &dyn KVStore,
    narrative_id: &str,
    reference: &AllenInterval,
    rel: AllenRelation,
    threshold: f64,
) -> Result<Vec<Uuid>> {
    use crate::types::Situation;
    let pairs = store.prefix_scan(b"s/")?;
    let mut hits = Vec::new();
    for (_k, v) in pairs {
        let sit: Situation = match serde_json::from_slice(&v) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if sit.narrative_id.as_deref() != Some(narrative_id) {
            continue;
        }
        if sit.temporal.start.is_none()
            && sit.temporal.end.is_none()
            && !sit.temporal.has_fuzzy_endpoints()
        {
            continue;
        }
        if fuzzy_relation_holds(&sit.temporal, reference, rel, threshold) {
            hits.push(sit.id);
        }
    }
    Ok(hits)
}

/// Build a degenerate (crisp point-in-time) [`AllenInterval`] — used by
/// executor paths that need a reference against which to compute graded
/// relations.
pub fn crisp_reference_interval(t: DateTime<Utc>) -> AllenInterval {
    AllenInterval {
        start: Some(t),
        end: Some(t),
        granularity: TimeGranularity::Exact,
        relations: vec![],
        fuzzy_endpoints: None,
    }
}

/// Tiny helper shared between [`fuzzy_relation_query_situations`] and
/// downstream callers that want the bare degree without the threshold.
pub fn fuzzy_relation_degree(
    a: &AllenInterval,
    b: &AllenInterval,
    rel: AllenRelation,
) -> f64 {
    graded_relation_value(a, b, rel)
}
