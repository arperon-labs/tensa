//! Fuzzy Formal Concept Analysis — graded concept lattices over fuzzy
//! attribute relations. Phase 8 of the Fuzzy Sprint.
//!
//! Derives Bělohlávek-style fuzzy concept lattices from
//! Entity × graded-attribute contexts. The algorithm is:
//!
//! 1. [`FormalContext`] with objects = entities (optionally filtered
//!    by [`EntityType`] + narrative_id), attributes = property-keys +
//!    tags, graded incidence `I(e, a) = μ ∈ [0, 1]`.
//! 2. Galois closure under a configurable [`TNormKind`] using the
//!    residual implication (Gödel: `x → y = 1 if x ≤ y else y`; Goguen
//!    and Łukasiewicz have their own residua). A concept `(A, B)`
//!    satisfies `A↑ = B ∧ B↓ = A`.
//! 3. [`next_closure_enumerate`] — Ganter (1984) NextClosure, graded
//!    adaptation per Bělohlávek (2004). Deterministic; ≤64 objects.
//! 4. [`hasse_edges`] — transitive-reduction Hasse diagram edges.
//!
//! Perf caps: default 500×50, hard cap 2000×200 with explicit
//! `large_context: true`. Persistence in [`super::fca_store`].
//!
//! Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::types::{Entity, EntityType};

// ── Perf caps ────────────────────────────────────────────────────────────────

/// Default soft cap — allowed without opt-in.
pub const DEFAULT_MAX_OBJECTS: usize = 500;
pub const DEFAULT_MAX_ATTRIBUTES: usize = 50;
/// Hard cap — request bodies above this always fail, even with
/// `large_context = true`.
pub const HARD_MAX_OBJECTS: usize = 2000;
pub const HARD_MAX_ATTRIBUTES: usize = 200;
/// Numerical tolerance for graded concept-equality comparisons.
pub const EPSILON: f64 = 1e-9;

// ── FormalContext ───────────────────────────────────────────────────────────

/// A graded formal context `(O, A, I)` with dense `objects × attributes`
/// incidence values in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormalContext {
    pub objects: Vec<ContextObject>,
    pub attributes: Vec<String>,
    /// Dense matrix `[num_objects][num_attributes]`, clamped to `[0, 1]`
    /// at construction.
    pub incidence: Vec<Vec<f64>>,
}

/// Object metadata stored alongside each incidence row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextObject {
    pub id: Uuid,
    pub label: String,
}

impl FormalContext {
    /// Build a formal context directly from in-memory data. Incidence
    /// values are clamped to `[0, 1]` and `NaN` collapses to `0.0`.
    pub fn new(
        objects: Vec<ContextObject>,
        attributes: Vec<String>,
        incidence: Vec<Vec<f64>>,
    ) -> Result<Self> {
        if incidence.len() != objects.len() {
            return Err(TensaError::InvalidInput(format!(
                "FCA incidence row count {} != objects {}",
                incidence.len(),
                objects.len()
            )));
        }
        let ncols = attributes.len();
        let mut sanitized: Vec<Vec<f64>> = Vec::with_capacity(incidence.len());
        for (i, row) in incidence.into_iter().enumerate() {
            if row.len() != ncols {
                return Err(TensaError::InvalidInput(format!(
                    "FCA row {i} has {} cols, expected {ncols}",
                    row.len()
                )));
            }
            sanitized.push(row.into_iter().map(clamp01).collect());
        }
        Ok(Self { objects, attributes, incidence: sanitized })
    }

    #[inline]
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    #[inline]
    pub fn num_attributes(&self) -> usize {
        self.attributes.len()
    }

    /// Build a context from the hypergraph. `opts.attribute_allowlist`
    /// takes precedence over gathered attribute names; entity order is
    /// the sorted UUID order for reproducibility.
    pub fn from_hypergraph(
        hg: &Hypergraph,
        narrative_id: &str,
        opts: &FormalContextOptions,
    ) -> Result<Self> {
        let mut entities: Vec<Entity> = hg.list_entities_by_narrative(narrative_id)?;
        if let Some(et) = opts.entity_type_filter.as_ref() {
            entities.retain(|e| &e.entity_type == et);
        }
        entities.sort_by_key(|e| e.id);

        let attributes: Vec<String> = match &opts.attribute_allowlist {
            Some(list) => list.clone(),
            None => gather_attribute_names(&entities),
        };

        let nrows = entities.len();
        let ncols = attributes.len();
        enforce_perf_cap(nrows, ncols, opts.large_context)?;

        let mut incidence = vec![vec![0.0_f64; ncols]; nrows];
        let mut objects = Vec::with_capacity(nrows);
        for (i, e) in entities.iter().enumerate() {
            objects.push(ContextObject { id: e.id, label: entity_label(e) });
            for (j, attr) in attributes.iter().enumerate() {
                incidence[i][j] = clamp01(grade_property(&e.properties, attr));
            }
        }
        Ok(Self { objects, attributes, incidence })
    }

    /// Closure `B↓ = { e | ∀ a: B(a) → I(e, a) }`. Returns a sorted
    /// extent.
    fn intent_to_extent(&self, intent: &[f64], tnorm: TNormKind) -> Vec<usize> {
        (0..self.num_objects())
            .filter(|&i| {
                intent.iter().enumerate().all(|(j, &b_a)| {
                    residual_implication(tnorm, b_a, self.incidence[i][j]) >= 1.0 - EPSILON
                })
            })
            .collect()
    }

    /// Closure `A↑(a) = inf_{e ∈ A} I(e, a)`. Empty extent = top
    /// element = `1.0` for every attribute.
    fn extent_to_intent(&self, extent: &[usize]) -> Vec<f64> {
        let ncols = self.num_attributes();
        if extent.is_empty() {
            return vec![1.0; ncols];
        }
        let mut intent = vec![1.0_f64; ncols];
        for &i in extent {
            for (j, cell) in self.incidence[i].iter().enumerate() {
                if *cell < intent[j] {
                    intent[j] = *cell;
                }
            }
        }
        intent
    }
}

/// Options struct for [`FormalContext::from_hypergraph`].
#[derive(Debug, Clone, Default)]
pub struct FormalContextOptions {
    pub entity_type_filter: Option<EntityType>,
    pub attribute_allowlist: Option<Vec<String>>,
    /// Opt-in large-context mode (>soft cap but ≤hard cap).
    pub large_context: bool,
}

// ── Concept + ConceptLattice ────────────────────────────────────────────────

/// Graded formal concept. `extent` = sorted object indices into
/// [`FormalContext::objects`]; `intent` = sorted `(attr_idx, μ)` pairs
/// (zero-membership cells elided).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Concept {
    pub extent: Vec<usize>,
    pub intent: Vec<(usize, f64)>,
}

/// A full graded concept lattice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptLattice {
    pub id: Uuid,
    pub narrative_id: String,
    /// Echoed t-norm name (registry key).
    pub tnorm: String,
    pub attributes: Vec<String>,
    pub objects: Vec<ContextObject>,
    pub concepts: Vec<Concept>,
    /// Hasse edges `(parent_idx, child_idx)` with `parent`'s extent
    /// strictly containing `child`'s.
    pub order: Vec<(usize, usize)>,
    pub created_at: DateTime<Utc>,
}

impl ConceptLattice {
    #[inline]
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }

    /// `InvalidInput` when the index is out of range — the REST surface
    /// maps this to HTTP 400 cleanly.
    pub fn concept(&self, idx: usize) -> Result<&Concept> {
        self.concepts.get(idx).ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "concept index {idx} out of range (lattice has {} concepts)",
                self.concepts.len()
            ))
        })
    }
}

// ── Build + NextClosure enumeration ─────────────────────────────────────────

/// Build the full concept lattice for `ctx` under the configured
/// t-norm.
pub fn build_lattice(ctx: &FormalContext, tnorm: TNormKind) -> Result<ConceptLattice> {
    build_lattice_with_threshold(ctx, tnorm, 0)
}

/// Build a concept lattice, pruning concepts whose extent size is
/// strictly less than `min_extent` (0 = no pruning).
pub fn build_lattice_with_threshold(
    ctx: &FormalContext,
    tnorm: TNormKind,
    min_extent: usize,
) -> Result<ConceptLattice> {
    enforce_perf_cap(ctx.num_objects(), ctx.num_attributes(), false)?;
    let concepts_raw = next_closure_enumerate(ctx, tnorm)?;
    let concepts: Vec<Concept> = concepts_raw
        .into_iter()
        .filter(|c| c.extent.len() >= min_extent)
        .collect();
    let order = hasse_edges(&concepts);
    Ok(ConceptLattice {
        id: Uuid::now_v7(),
        narrative_id: String::new(), // caller fills this in
        tnorm: tnorm.name().to_string(),
        attributes: ctx.attributes.clone(),
        objects: ctx.objects.clone(),
        concepts,
        order,
        created_at: Utc::now(),
    })
}

/// Ganter's NextClosure (1984), graded adaptation per Bělohlávek 2004
/// §5 — iterate each object-subset (2^|O|), close under Galois, and
/// deduplicate by bitmask. Capped at 64 objects so the subset space
/// fits in a `u64` mask.
fn next_closure_enumerate(ctx: &FormalContext, tnorm: TNormKind) -> Result<Vec<Concept>> {
    let n = ctx.num_objects();
    if n == 0 {
        let intent = ctx.extent_to_intent(&[]);
        return Ok(vec![Concept {
            extent: Vec::new(),
            intent: sparse_intent(&intent),
        }]);
    }
    if n > 64 {
        return Err(TensaError::InvalidInput(format!(
            "FCA NextClosure enumeration capped at 64 objects (got {n}); \
             use a smaller entity_type_filter or attribute_allowlist"
        )));
    }

    let mut seen: HashMap<u64, Concept> = HashMap::new();
    for mask in 0..(1u64 << n) {
        let extent_candidate: Vec<usize> =
            (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();
        let intent = ctx.extent_to_intent(&extent_candidate);
        let closed_extent = ctx.intent_to_extent(&intent, tnorm);
        let key = mask_from_sorted(&closed_extent);
        if seen.contains_key(&key) {
            continue;
        }
        let closed_intent = ctx.extent_to_intent(&closed_extent);
        seen.insert(
            key,
            Concept {
                extent: closed_extent,
                intent: sparse_intent(&closed_intent),
            },
        );
    }
    // Parent-before-child ordering for hasse_edges: largest extent
    // first, then lexicographic.
    let mut concepts: Vec<Concept> = seen.into_values().collect();
    concepts.sort_by(|a, b| b.extent.len().cmp(&a.extent.len()).then(a.extent.cmp(&b.extent)));
    Ok(concepts)
}

/// Transitive-reduction Hasse edges over extent inclusion.
pub fn hasse_edges(concepts: &[Concept]) -> Vec<(usize, usize)> {
    let contains = |a: &[usize], b: &[usize]| a.len() > b.len() && subset(b, a);
    let mut edges = Vec::new();
    for (ci, c) in concepts.iter().enumerate() {
        for (pi, p) in concepts.iter().enumerate() {
            if pi == ci || !contains(&p.extent, &c.extent) {
                continue;
            }
            let intermediate = concepts.iter().enumerate().any(|(qi, q)| {
                qi != pi
                    && qi != ci
                    && contains(&p.extent, &q.extent)
                    && contains(&q.extent, &c.extent)
            });
            if !intermediate {
                edges.push((pi, ci));
            }
        }
    }
    edges.sort();
    edges.dedup();
    debug_assert!(edges.iter().all(|&(p, c)| p < concepts.len() && c < concepts.len()));
    edges
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Residual implication per t-norm. Hamacher residua have no closed
/// form for arbitrary λ — fall back to Gödel (see FUZZY_Sprint notes).
fn residual_implication(tnorm: TNormKind, x: f64, y: f64) -> f64 {
    let x = clamp01(x);
    let y = clamp01(y);
    match tnorm {
        TNormKind::Godel | TNormKind::Hamacher(_) => {
            if x <= y + EPSILON {
                1.0
            } else {
                y
            }
        }
        TNormKind::Goguen => {
            if x < EPSILON {
                1.0
            } else {
                (y / x).min(1.0)
            }
        }
        TNormKind::Lukasiewicz => (1.0 - x + y).clamp(0.0, 1.0),
    }
}

/// Defensive clamp to `[0, 1]`. NaN → 0.
#[inline]
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Grade membership of `attr` in an entity's `properties`. Accepts
/// `bool` / `number` / `string` direct hits, string-equal entries in
/// `properties.tags[]` (case-insensitive), and `properties.fuzzy_tags`
/// numeric entries. Everything else grades as `0.0`.
fn grade_property(props: &serde_json::Value, attr: &str) -> f64 {
    if let Some(v) = props.get(attr) {
        match v {
            serde_json::Value::Bool(b) => return if *b { 1.0 } else { 0.0 },
            serde_json::Value::Number(n) => {
                return n.as_f64().unwrap_or(0.0);
            }
            serde_json::Value::String(_) => return 1.0,
            _ => {}
        }
    }
    if let Some(tags) = props.get("tags").and_then(|v| v.as_array()) {
        if tags
            .iter()
            .filter_map(|t| t.as_str())
            .any(|s| s.eq_ignore_ascii_case(attr))
        {
            return 1.0;
        }
    }
    if let Some(fzt) = props.get("fuzzy_tags").and_then(|v| v.as_object()) {
        if let Some(v) = fzt.get(attr).and_then(|v| v.as_f64()) {
            return v;
        }
    }
    0.0
}

/// Union the property-keys, `tags[]` entries, and `fuzzy_tags` keys
/// across the entity slice into a stable alphabetically-sorted list.
/// `media` / `aliases` / `tags` / `fuzzy_tags` / `embedding` are
/// filtered from direct properties (their contents still contribute).
fn gather_attribute_names(entities: &[Entity]) -> Vec<String> {
    use std::collections::BTreeSet;
    const RESERVED: &[&str] = &["media", "aliases", "tags", "fuzzy_tags", "embedding"];
    let mut names: BTreeSet<String> = BTreeSet::new();
    for e in entities {
        if let Some(obj) = e.properties.as_object() {
            for k in obj.keys() {
                if !RESERVED.contains(&k.as_str()) {
                    names.insert(k.clone());
                }
            }
        }
        if let Some(tags) = e.properties.get("tags").and_then(|v| v.as_array()) {
            for t in tags.iter().filter_map(|v| v.as_str()) {
                names.insert(t.to_ascii_lowercase());
            }
        }
        if let Some(obj) = e
            .properties
            .get("fuzzy_tags")
            .and_then(|v| v.as_object())
        {
            for k in obj.keys() {
                names.insert(k.clone());
            }
        }
    }
    names.into_iter().collect()
}

/// Drop cells below EPSILON; return the sparse `(attr_idx, μ)` list.
fn sparse_intent(intent: &[f64]) -> Vec<(usize, f64)> {
    intent
        .iter()
        .enumerate()
        .filter_map(|(j, &v)| if v > EPSILON { Some((j, v)) } else { None })
        .collect()
}

/// Hard cap always fails; soft cap requires `large_context`; `warn!`
/// when active.
fn enforce_perf_cap(nrows: usize, ncols: usize, large_context: bool) -> Result<()> {
    if nrows > HARD_MAX_OBJECTS || ncols > HARD_MAX_ATTRIBUTES {
        return Err(TensaError::InvalidInput(format!(
            "FCA context {}x{} exceeds hard cap ({}x{}); \
             split the narrative or narrow attribute_allowlist",
            nrows, ncols, HARD_MAX_OBJECTS, HARD_MAX_ATTRIBUTES
        )));
    }
    let over_soft = nrows > DEFAULT_MAX_OBJECTS || ncols > DEFAULT_MAX_ATTRIBUTES;
    if over_soft && !large_context {
        return Err(TensaError::InvalidInput(format!(
            "FCA context {}x{} exceeds soft cap ({}x{}); \
             set `large_context: true` to opt in (worst-case O(2^min(|O|,|A|)) \
             concept blow-up)",
            nrows, ncols, DEFAULT_MAX_OBJECTS, DEFAULT_MAX_ATTRIBUTES
        )));
    }
    if over_soft {
        tracing::warn!(
            target: "tensa::fuzzy::fca",
            nrows, ncols,
            "FCA large_context mode — enumeration is worst-case O(2^min(|O|,|A|))"
        );
    }
    Ok(())
}

/// `properties.name` when present, else the first 8 chars of the UUID.
fn entity_label(e: &Entity) -> String {
    e.properties
        .get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| e.id.to_string().chars().take(8).collect())
}

/// Sorted `Vec<usize>` (indices < 64) → `u64` bitmask.
fn mask_from_sorted(extent: &[usize]) -> u64 {
    extent.iter().fold(0u64, |acc, &i| acc | (1u64 << i))
}

/// `a ⊆ b` for sorted `Vec<usize>`.
fn subset(a: &[usize], b: &[usize]) -> bool {
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j == b.len() || b[j] != x {
            return false;
        }
    }
    true
}

