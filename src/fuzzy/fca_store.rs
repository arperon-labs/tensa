//! Fuzzy FCA — KV persistence helpers.
//!
//! Split out of [`super::fca`] to keep both files under the 500-line
//! cap. The concept-lattice algorithms live in `fca.rs`; the KV round-
//! trip + `narrative_id`-scoped listing helpers live here.
//!
//! All lattice blobs persist at
//! `fz/fca/{narrative_id}/{lattice_id_v7_BE_BIN_16}` via
//! [`super::key_fuzzy_fca`].
//!
//! Cites: [belohlavek2004fuzzyfca] [kridlo2010fuzzyfca].

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::fca::{build_lattice, ConceptLattice, FormalContext, FormalContextOptions};
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;
use crate::types::EntityType;

/// Persist a concept lattice at `fz/fca/{narrative_id}/{lattice_id}`.
/// Caller must ensure `lattice.narrative_id` matches `narrative_id` —
/// this is the argument of record for the key builder, not the body
/// field.
pub fn save_concept_lattice(
    store: &dyn KVStore,
    narrative_id: &str,
    lattice: &ConceptLattice,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_fca(narrative_id, &lattice.id);
    let bytes =
        serde_json::to_vec(lattice).map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(&key, &bytes)
}

/// Load a persisted lattice, or `None` when the key does not exist.
pub fn load_concept_lattice(
    store: &dyn KVStore,
    narrative_id: &str,
    lattice_id: &Uuid,
) -> Result<Option<ConceptLattice>> {
    let key = crate::fuzzy::key_fuzzy_fca(narrative_id, lattice_id);
    match store.get(&key)? {
        Some(bytes) => {
            let lat: ConceptLattice = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(lat))
        }
        None => Ok(None),
    }
}

/// Delete a persisted lattice (idempotent).
pub fn delete_concept_lattice(
    store: &dyn KVStore,
    narrative_id: &str,
    lattice_id: &Uuid,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_fca(narrative_id, lattice_id);
    store.delete(&key)
}

/// List every persisted lattice for a narrative, newest-first.
/// Entries whose values fail to deserialize are skipped with a `warn!`.
pub fn list_concept_lattices_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<ConceptLattice>> {
    let mut prefix = crate::fuzzy::FUZZY_FCA_PREFIX.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let pairs = store.prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, v) in pairs {
        match serde_json::from_slice::<ConceptLattice>(&v) {
            Ok(l) => out.push(l),
            Err(e) => tracing::warn!(
                narrative_id = %narrative_id,
                "concept lattice deserialize failed ({e}); skipping"
            ),
        }
    }
    // Newest first: v7 UUIDs sort chronologically, so reverse.
    out.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(out)
}

/// Lightweight summary row for `GET /fuzzy/fca/lattices/{nid}`. Avoids
/// shipping every concept's extent/intent over the wire when callers
/// only need the table of known lattices.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LatticeSummary {
    pub lattice_id: Uuid,
    pub narrative_id: String,
    pub num_concepts: usize,
    pub num_objects: usize,
    pub num_attributes: usize,
    pub tnorm: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl LatticeSummary {
    /// Project a full lattice into a summary row. Pure — no I/O.
    pub fn from_lattice(l: &ConceptLattice) -> Self {
        Self {
            lattice_id: l.id,
            narrative_id: l.narrative_id.clone(),
            num_concepts: l.num_concepts(),
            num_objects: l.objects.len(),
            num_attributes: l.attributes.len(),
            tnorm: l.tnorm.clone(),
            created_at: l.created_at,
        }
    }
}

// ── Workflow wire: narrative-family lattice helper ──────────────────────────

/// Disinfo Ops narrative-family detection — thin wrapper that builds the
/// FCA lattice with narrative-tag attributes (no allowlist) and the
/// Gödel t-norm. Callers downstream interpret concepts as
/// narrative-family clusters. Returns an in-memory [`ConceptLattice`]
/// (the caller decides whether to persist).
pub fn narrative_family_lattice(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<ConceptLattice> {
    let ctx = FormalContext::from_hypergraph(
        hg,
        narrative_id,
        &FormalContextOptions::default(),
    )?;
    let mut lattice = build_lattice(&ctx, TNormKind::Godel)?;
    lattice.narrative_id = narrative_id.to_string();
    Ok(lattice)
}

/// Writer-archetype detection — build a character × trait-graded
/// concept lattice. Phase 8 uses the default entity-type filter
/// (Actor only). Workshop / writer integration is deferred (no
/// orderable step list at the REST surface yet — see FUZZY_Sprint.md
/// deferrals).
pub fn build_archetype_lattice(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<ConceptLattice> {
    let opts = FormalContextOptions {
        entity_type_filter: Some(EntityType::Actor),
        attribute_allowlist: None,
        large_context: false,
    };
    let ctx = FormalContext::from_hypergraph(hg, narrative_id, &opts)?;
    let mut lattice = build_lattice(&ctx, TNormKind::Godel)?;
    lattice.narrative_id = narrative_id.to_string();
    Ok(lattice)
}
