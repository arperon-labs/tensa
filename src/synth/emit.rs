//! Single source of truth for synthetic Entity / Situation / Participation
//! emission (EATH Phase 3+).
//!
//! Every synthetic record produced anywhere in TENSA MUST flow through the
//! `write_synthetic_*` helpers here. Hand-rolled construction bypasses the
//! provenance contract and breaks the load-bearing "no synthetic leak"
//! invariant tested in `invariant_tests.rs`.
//!
//! ## Provenance contract
//!
//! Each record carries:
//! - `synthetic: true` flag in its JSON sidecar (`Entity.properties` /
//!   `Situation.properties` / `Participation.info_set.knows_before`).
//! - `synth_run_id` linking back via `syn/seed/{run_id}` +
//!   `syn/r/{narrative}/{run_id}`.
//! - `extraction_method = ExtractionMethod::Synthetic { model, run_id }` on
//!   Entity + Situation (Participation has no such field; `info_set` sidecar).
//!
//! ## Role decision (closes Phase 1 deferral)
//!
//! Synthetic participations use `Role::Bystander`. Adding `Role::Observer`
//! whose sole purpose is "this row is synthetic" would force exhaustive-
//! match updates everywhere while providing no semantic value beyond what
//! [`is_synthetic_participation`] already gives downstream filters.
//!
//! ## Hypergraph thread safety
//!
//! All three writers go through [`Hypergraph::create_entity`],
//! [`Hypergraph::create_situation`], [`Hypergraph::add_participant`], which
//! serialize through the underlying [`KVStore`]. Safe under concurrent worker
//! access; Phase 4 relies on this.

use chrono::{DateTime, Utc};
use rand::RngCore;
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{
    AllenInterval, ContentBlock, Entity, EntityType, ExtractionMethod, InfoSet, KnowledgeFact,
    MaturityLevel, NarrativeLevel, Participation, Role, Situation, TimeGranularity,
};

/// Default confidence assigned to every synthetic record. Synthetic data is
/// suspect until proven otherwise — downstream consumers should treat it as
/// LOW-confidence by default.
pub const DEFAULT_SYNTHETIC_CONFIDENCE: f32 = 0.5;

/// JSON key carrying the synthetic flag on Entity and Situation `properties`.
const PROP_SYNTHETIC: &str = "synthetic";
/// JSON key carrying the originating run UUID (string form) on properties.
const PROP_RUN_ID: &str = "synth_run_id";
/// JSON key carrying the surrogate model name on properties.
const PROP_MODEL: &str = "synth_model";
/// JSON key carrying the per-step index on synthetic situations.
const PROP_STEP: &str = "synth_step";
/// JSON key carrying the per-entity emission index (0..n) on synthetic
/// entities. Used by EATH determinism + phase-distribution tests to map
/// emitted UUIDs back to per-entity parameter rows (e.g. `a_T[i]`).
const PROP_INDEX: &str = "synth_index";

/// Sentinel prefix on `KnowledgeFact.fact` that marks a participation as
/// synthetic. Format: `"synthetic|run_id={uuid}|model={name}"`.
const PARTICIPATION_SENTINEL_PREFIX: &str = "synthetic|run_id=";

// ── EmitContext ──────────────────────────────────────────────────────────────

/// Per-run emission context. Cheap to construct, cheap to clone — pass by
/// reference everywhere downstream of the run loop.
#[derive(Debug, Clone)]
pub struct EmitContext {
    /// v7 UUID minted once per run; embedded into every emitted record so
    /// downstream consumers can group-and-filter by run.
    pub run_id: Uuid,
    /// Output narrative ID — every record gets `narrative_id = Some(this)`.
    pub narrative_id: String,
    /// Maturity level for every record. Always `MaturityLevel::Candidate`
    /// for synthetic data — promotion is a deliberate human action.
    pub maturity: MaturityLevel,
    /// Confidence assigned to every record. Defaults to
    /// [`DEFAULT_SYNTHETIC_CONFIDENCE`]; surrogate impls may override (e.g.
    /// to 1.0 in deterministic toy-model runs that need tight comparison).
    pub confidence: f32,
    /// Prefix for entity labels — e.g. `"synth-actor-"` produces
    /// `"synth-actor-0"`, `"synth-actor-1"`, etc.
    pub label_prefix: String,
    /// Anchor for `temporal.start` on synthetic situations. Step `n` lands
    /// at `time_anchor + n * step_duration_seconds`.
    pub time_anchor: DateTime<Utc>,
    /// Seconds between successive steps. Combined with `time_anchor` to
    /// build the `AllenInterval` for each emitted situation.
    pub step_duration_seconds: i64,
    /// Surrogate-model name (`"eath"`, future: `"had"`, etc.) — written to
    /// every record's provenance metadata.
    pub model: String,
    /// Phase 13b — entity-reuse hook for configuration-style models (NuDHy)
    /// that preserve node identity instead of minting fresh entities.
    ///
    /// - `None` (default, EATH path): callers MUST invoke
    ///   [`write_synthetic_entities`] before [`write_synthetic_situation`];
    ///   the situation references freshly-minted entity UUIDs.
    /// - `Some(vec)`: callers SKIP [`write_synthetic_entities`] entirely and
    ///   pass pre-existing entity UUIDs (from the source narrative) directly
    ///   as the `members` slice. The vector is the canonical entity universe
    ///   for the run — primarily a documentation contract; the writers do
    ///   not iterate it (members come in via the situation call).
    ///
    /// Backward-compatible: `EmitContext::new` initialises this to `None`,
    /// preserving the EATH path bit-for-bit.
    pub reuse_entities: Option<Vec<Uuid>>,
}

impl EmitContext {
    /// Convenience constructor with [`DEFAULT_SYNTHETIC_CONFIDENCE`] and
    /// `MaturityLevel::Candidate`. Most callers want this. `reuse_entities`
    /// defaults to `None` (EATH-style fresh-entity path).
    pub fn new(
        run_id: Uuid,
        narrative_id: impl Into<String>,
        label_prefix: impl Into<String>,
        time_anchor: DateTime<Utc>,
        step_duration_seconds: i64,
        model: impl Into<String>,
    ) -> Self {
        Self {
            run_id,
            narrative_id: narrative_id.into(),
            maturity: MaturityLevel::Candidate,
            confidence: DEFAULT_SYNTHETIC_CONFIDENCE,
            label_prefix: label_prefix.into(),
            time_anchor,
            step_duration_seconds,
            model: model.into(),
            reuse_entities: None,
        }
    }

    fn extraction(&self) -> ExtractionMethod {
        ExtractionMethod::Synthetic {
            model: self.model.clone(),
            run_id: self.run_id,
        }
    }
}

// ── Writers ──────────────────────────────────────────────────────────────────

/// Mint a v4-style UUID from the supplied sub-RNG. Bits 6 + 8 are tweaked to
/// match RFC 4122 so `Uuid` validators accept them.
fn uuid_from_rng(rng: &mut ChaCha8Rng) -> Uuid {
    let mut bytes = [0u8; 16];
    rng.fill_bytes(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
    Uuid::from_bytes(bytes)
}

/// Emit `n` synthetic Actor entities, drawing UUIDs from `sub_rng`. Caller
/// owns the RNG so determinism stays the surrogate's responsibility.
pub fn write_synthetic_entities(
    ctx: &EmitContext,
    n: usize,
    sub_rng: &mut ChaCha8Rng,
    hypergraph: &Hypergraph,
) -> Result<Vec<Uuid>> {
    let now = Utc::now();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let id = uuid_from_rng(sub_rng);
        let label = format!("{}{}", ctx.label_prefix, i);
        let entity = Entity {
            id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({
                "name": label,
                PROP_SYNTHETIC: true,
                PROP_RUN_ID: ctx.run_id.to_string(),
                PROP_MODEL: ctx.model.clone(),
                PROP_INDEX: i,
            }),
            beliefs: None,
            embedding: None,
            maturity: ctx.maturity,
            confidence: ctx.confidence,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ctx.extraction()),
            narrative_id: Some(ctx.narrative_id.clone()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hypergraph.create_entity(entity)?;
        out.push(id);
    }
    Ok(out)
}

/// Emit one synthetic Situation at `step` and link `members` as participants
/// (every member tagged with [`Role::Bystander`] — see module-level doc for
/// the rationale closing the Phase 1 deferral).
pub fn write_synthetic_situation(
    ctx: &EmitContext,
    step: usize,
    members: &[Uuid],
    sub_rng: &mut ChaCha8Rng,
    hypergraph: &Hypergraph,
) -> Result<Uuid> {
    let sit_id = uuid_from_rng(sub_rng);
    let start = ctx.time_anchor
        + chrono::Duration::seconds(step as i64 * ctx.step_duration_seconds);
    let end = start + chrono::Duration::seconds(ctx.step_duration_seconds);
    let now = Utc::now();
    let situation = Situation {
        id: sit_id,
        name: None,
        description: None,
        properties: serde_json::json!({
            PROP_SYNTHETIC: true,
            PROP_RUN_ID: ctx.run_id.to_string(),
            PROP_MODEL: ctx.model.clone(),
            PROP_STEP: step,
        }),
        temporal: AllenInterval {
            start: Some(start),
            end: Some(end),
            granularity: TimeGranularity::Synthetic,
            relations: vec![],
            fuzzy_endpoints: None,
        },
        spatial: None,
        game_structure: None,
        causes: vec![],
        deterministic: None,
        probabilistic: None,
        embedding: None,
        raw_content: vec![ContentBlock::text(&format!(
            "Synthetic event at step {} (run {})",
            step, ctx.run_id
        ))],
        narrative_level: NarrativeLevel::Scene,
        discourse: None,
        maturity: ctx.maturity,
        confidence: ctx.confidence,
        confidence_breakdown: None,
        extraction_method: ctx.extraction(),
        provenance: vec![],
        narrative_id: Some(ctx.narrative_id.clone()),
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
    };
    hypergraph.create_situation(situation)?;

    for member in members {
        write_synthetic_participation(ctx, member, &sit_id, Role::Bystander, hypergraph)?;
    }
    Ok(sit_id)
}

/// Emit one synthetic Participation. The `info_set.knows_before` slot
/// carries the sentinel `KnowledgeFact` that downstream filters key on.
pub fn write_synthetic_participation(
    ctx: &EmitContext,
    entity_id: &Uuid,
    situation_id: &Uuid,
    role: Role,
    hypergraph: &Hypergraph,
) -> Result<()> {
    let participation = Participation {
        entity_id: *entity_id,
        situation_id: *situation_id,
        role,
        info_set: Some(InfoSet {
            knows_before: vec![KnowledgeFact {
                about_entity: Uuid::nil(),
                fact: format!(
                    "{}{}|model={}",
                    PARTICIPATION_SENTINEL_PREFIX, ctx.run_id, ctx.model
                ),
                confidence: 1.0,
            }],
            learns: vec![],
            reveals: vec![],
            beliefs_about_others: vec![],
        }),
        action: None,
        payoff: None,
        seq: 0, // overwritten by add_participant
    };
    hypergraph.add_participant(participation)?;
    Ok(())
}

// ── Filter predicates (downstream API helpers) ───────────────────────────────

/// True iff `entity.properties.synthetic == true`.
pub fn is_synthetic_entity(entity: &Entity) -> bool {
    entity
        .properties
        .get(PROP_SYNTHETIC)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// True iff `situation.properties.synthetic == true`.
pub fn is_synthetic_situation(situation: &Situation) -> bool {
    situation
        .properties
        .get(PROP_SYNTHETIC)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// True iff this Participation carries the synthetic sentinel in any of its
/// `info_set.knows_before` facts.
pub fn is_synthetic_participation(p: &Participation) -> bool {
    let Some(info) = &p.info_set else {
        return false;
    };
    info.knows_before
        .iter()
        .any(|f| f.fact.starts_with(PARTICIPATION_SENTINEL_PREFIX))
}

/// Filter a Vec of entities by the synthetic flag. Returns the input
/// unchanged when `include_synthetic` is true. Used by every aggregation
/// endpoint that respects `?include_synthetic=true`.
pub fn filter_synthetic_entities(items: Vec<Entity>, include_synthetic: bool) -> Vec<Entity> {
    if include_synthetic {
        items
    } else {
        items.into_iter().filter(|e| !is_synthetic_entity(e)).collect()
    }
}

/// Filter a Vec of situations by the synthetic flag. Mirror of
/// [`filter_synthetic_entities`].
pub fn filter_synthetic_situations(
    items: Vec<Situation>,
    include_synthetic: bool,
) -> Vec<Situation> {
    if include_synthetic {
        items
    } else {
        items
            .into_iter()
            .filter(|s| !is_synthetic_situation(s))
            .collect()
    }
}

// ── Run-id extraction (used by Phase 11 Studio + invariant tests) ────────────

/// Extract the run UUID from a synthetic entity's `properties.synth_run_id`.
/// Returns `None` if missing or unparseable — non-synthetic records always
/// return `None`.
pub fn entity_run_id(entity: &Entity) -> Option<Uuid> {
    entity
        .properties
        .get(PROP_RUN_ID)
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
}

/// Mirror of [`entity_run_id`] for situations.
pub fn situation_run_id(situation: &Situation) -> Option<Uuid> {
    situation
        .properties
        .get(PROP_RUN_ID)
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
}

/// Extract the run UUID from a synthetic participation's info_set sentinel.
/// Returns `None` for non-synthetic participations.
pub fn participation_run_id(p: &Participation) -> Option<Uuid> {
    let info = p.info_set.as_ref()?;
    for fact in &info.knows_before {
        if let Some(rest) = fact.fact.strip_prefix(PARTICIPATION_SENTINEL_PREFIX) {
            // rest = "{uuid}|model={name}"
            let id_part = rest.split('|').next()?;
            if let Ok(u) = Uuid::parse_str(id_part) {
                return Some(u);
            }
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────
//
// Extracted to `emit_tests.rs` so this file stays under the 500-line cap
// (Phase 13b added the `EmitContext::reuse_entities` field + a regression
// test for it).

#[cfg(test)]
#[path = "emit_tests.rs"]
mod emit_tests;
