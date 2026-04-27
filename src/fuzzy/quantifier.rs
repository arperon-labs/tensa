//! Fuzzy Sprint Phase 6 — Intermediate quantifiers (Novák–Murinová).
//!
//! First-class "most / many / few / almost all" over entity or situation
//! sets. Each quantifier is a non-decreasing (or, for `Few`, non-increasing)
//! ramp `Q: [0,1] → [0,1]` applied to a cardinality ratio `r = (Σ_e μ_P(e))
//! / N`. For crisp predicates `μ_P ∈ {0,1}` this reduces to the classical
//! cardinality-quantifier truth value; for graded predicates the ramp
//! interpolates smoothly.
//!
//! Phase 6 ships **monotonic type <1,1>** quantifiers only. Non-monotonic
//! quantifiers (e.g. "about half") are deferred — see the Phase 6.5 micro-
//! phase note in [`docs/FUZZY_Sprint.md`].
//!
//! ## Ramp specifications (hard-coded in Phase 6)
//!
//! * `Q_most`       : 0 below 0.3, linear 0.3 → 0.8, 1.0 at or above 0.8.
//! * `Q_many`       : 0 below 0.1, linear 0.1 → 0.5, 1.0 at or above 0.5.
//! * `Q_almost_all` : 0 below 0.7, linear 0.7 → 0.95, 1.0 at or above 0.95.
//! * `Q_few`        : defined as `1 - Q_many(r)` — "few" means "not many".
//!
//! Exposing these numbers as a calibration knob (`fuzzy_quantifier_calibration`
//! in `cfg/`) is a Phase 6.5 item per docs/FUZZY_Sprint.md.
//!
//! ## Evaluation semantics
//!
//! ```text
//!   r = (Σ_{e ∈ D} μ_P(e)) / |D|
//!   truth = Q(r)
//! ```
//!
//! Where `D` is the domain (entities of a given [`EntityType`] or situations
//! filtered by the caller's predicate). Empty `D` yields `r = 0`, and for
//! all four built-in quantifiers `Q(0) = 0`.
//!
//! Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;
use crate::types::{Entity, EntityType, Situation};

// ── Quantifier enum + ramp evaluation ───────────────────────────────────────

/// The four Phase 6 built-in monotonic quantifiers (Novák 2008 type <1,1>).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quantifier {
    /// "most": linear ramp 0.3 → 0.8, saturates at 1.0.
    Most,
    /// "many": linear ramp 0.1 → 0.5, saturates at 1.0.
    Many,
    /// "almost all": linear ramp 0.7 → 0.95, saturates at 1.0.
    AlmostAll,
    /// "few": the De Morgan dual `1 - Q_many(r)` — highest at low `r`.
    Few,
}

impl Quantifier {
    /// Canonical short name (lower-case snake) for registry + API surfaces.
    pub fn name(&self) -> &'static str {
        match self {
            Quantifier::Most => "most",
            Quantifier::Many => "many",
            Quantifier::AlmostAll => "almost_all",
            Quantifier::Few => "few",
        }
    }
}

/// Resolve a user-facing quantifier name (case- and dash/underscore-
/// insensitive) to a [`Quantifier`]. Returns
/// [`TensaError::InvalidInput`] for unknown names so API handlers can map
/// directly to HTTP 400.
pub fn quantifier_from_name(name: &str) -> Result<Quantifier> {
    let normalized: String = name
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c == '-' { '_' } else { c })
        .collect();
    match normalized.as_str() {
        "most" => Ok(Quantifier::Most),
        "many" => Ok(Quantifier::Many),
        "almost_all" | "almostall" => Ok(Quantifier::AlmostAll),
        "few" => Ok(Quantifier::Few),
        other => Err(TensaError::InvalidInput(format!(
            "unknown quantifier '{other}'; expected one of: most, many, almost_all, few"
        ))),
    }
}

/// Linear ramp helper — 0 below `lo`, 1 at or above `hi`, linear between.
/// Input is clamped to `[0,1]` before evaluation.
#[inline]
fn ramp(r: f64, lo: f64, hi: f64) -> f64 {
    let r = r.clamp(0.0, 1.0);
    if r <= lo {
        0.0
    } else if r >= hi {
        1.0
    } else {
        (r - lo) / (hi - lo)
    }
}

/// Evaluate `Q(r)` — the ramp semantics defined in the module docs.
///
/// Input is clamped to `[0,1]` before evaluation so callers don't have to
/// worry about numeric drift when constructing `r`.
pub fn evaluate(quantifier: Quantifier, r: f64) -> f64 {
    let r = r.clamp(0.0, 1.0);
    match quantifier {
        Quantifier::Most => ramp(r, 0.3, 0.8),
        Quantifier::Many => ramp(r, 0.1, 0.5),
        Quantifier::AlmostAll => ramp(r, 0.7, 0.95),
        // "few" = De Morgan dual of "many": the *lower* the ratio, the
        // more strongly the quantifier fires. Uses `r` directly (NOT
        // `1-r`) — `Q_few(r) = 1 - Q_many(r)` evaluated pointwise at r.
        Quantifier::Few => 1.0 - ramp(r, 0.1, 0.5),
    }
}

// ── Entity / situation evaluation ────────────────────────────────────────────

/// Evaluate `Q(r)` where `r = (Σ μ_P(e)) / |D|` over the entities of
/// `narrative_id`, optionally restricted to a single [`EntityType`].
///
/// `predicate` returns the graded membership `μ_P(e) ∈ [0,1]` per entity;
/// inputs outside the range are clamped to stay within the Zadeh domain.
/// Empty `D` → `r = 0` → `Q(0)` (0 for all four built-ins).
pub fn evaluate_over_entities<F>(
    hg: &Hypergraph,
    narrative_id: &str,
    entity_type: Option<EntityType>,
    predicate: F,
    quantifier: Quantifier,
) -> Result<f64>
where
    F: Fn(&Entity) -> f64,
{
    let entities = hg.list_entities_by_narrative(narrative_id)?;
    let domain: Vec<Entity> = match entity_type {
        Some(t) => entities.into_iter().filter(|e| e.entity_type == t).collect(),
        None => entities,
    };
    let n = domain.len();
    if n == 0 {
        return Ok(evaluate(quantifier, 0.0));
    }
    let mut sum = 0.0_f64;
    for e in &domain {
        sum += predicate(e).clamp(0.0, 1.0);
    }
    let r = sum / (n as f64);
    Ok(evaluate(quantifier, r))
}

/// Evaluate `Q(r)` where `r` is the normalised predicate sum over the
/// situations of `narrative_id`.
pub fn evaluate_over_situations<F>(
    hg: &Hypergraph,
    narrative_id: &str,
    predicate: F,
    quantifier: Quantifier,
) -> Result<f64>
where
    F: Fn(&Situation) -> f64,
{
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let n = situations.len();
    if n == 0 {
        return Ok(evaluate(quantifier, 0.0));
    }
    let mut sum = 0.0_f64;
    for s in &situations {
        sum += predicate(s).clamp(0.0, 1.0);
    }
    let r = sum / (n as f64);
    Ok(evaluate(quantifier, r))
}

// ── KV persistence at fz/quant/{narrative_id}/{predicate_hash} ──────────────
//
// The cache is best-effort: the value is a small scalar and the predicate
// hash is a caller-supplied deterministic hash of the predicate spec.
// A cache miss is always safe because the scalar can be recomputed.

/// Persisted shape for a quantifier evaluation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantifierResult {
    /// Canonical quantifier name.
    pub quantifier: String,
    /// Scalar truth value `Q(r) ∈ [0,1]`.
    pub value: f64,
    /// Optional human-readable label for provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Persist `result` at `fz/quant/{narrative_id}/{predicate_hash}`.
pub fn save_quantifier_result(
    store: &dyn KVStore,
    narrative_id: &str,
    predicate_hash: &str,
    result: &QuantifierResult,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_quant(narrative_id, predicate_hash);
    let bytes =
        serde_json::to_vec(result).map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(&key, &bytes)
}

/// Load a persisted quantifier result if present. Returns `Ok(None)` when
/// the key does not exist — callers recompute on miss.
pub fn load_quantifier_result(
    store: &dyn KVStore,
    narrative_id: &str,
    predicate_hash: &str,
) -> Result<Option<QuantifierResult>> {
    let key = crate::fuzzy::key_fuzzy_quant(narrative_id, predicate_hash);
    match store.get(&key)? {
        Some(bytes) => {
            let r: QuantifierResult = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(r))
        }
        None => Ok(None),
    }
}

/// Delete a persisted quantifier result (idempotent).
pub fn delete_quantifier_result(
    store: &dyn KVStore,
    narrative_id: &str,
    predicate_hash: &str,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_quant(narrative_id, predicate_hash);
    store.delete(&key)
}

/// List every persisted quantifier result for a narrative, returned as
/// `(predicate_hash, result)` pairs. Entries whose values fail to
/// deserialize are skipped silently with a `tracing::warn!`.
pub fn list_quantifier_results_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<(String, QuantifierResult)>> {
    let mut prefix = crate::fuzzy::FUZZY_QUANT_PREFIX.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let pairs = store.prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        let hash_bytes = &k[prefix.len()..];
        let hash = match std::str::from_utf8(hash_bytes) {
            Ok(s) => s.to_string(),
            Err(_) => {
                tracing::warn!(
                    narrative_id = %narrative_id,
                    "quantifier cache key has non-utf8 hash; skipping"
                );
                continue;
            }
        };
        match serde_json::from_slice::<QuantifierResult>(&v) {
            Ok(r) => out.push((hash, r)),
            Err(e) => {
                tracing::warn!(
                    narrative_id = %narrative_id,
                    predicate_hash = %hash,
                    "quantifier cache deserialize failed ({e}); skipping"
                );
            }
        }
    }
    Ok(out)
}

// ── Helpers for workflow wires (opinion-dynamics / disinfo) ────────────────

/// Apply a quantifier to the converged-fraction of an
/// [`OpinionDynamicsReport`] — "did *most* entities converge to
/// consensus?" rendered as a single scalar.
///
/// The convergence ratio used here is `num_converged / num_total` where
/// `num_converged` is the population of the largest cluster (classical
/// BCM consensus indicator) and `num_total` is the entity count carried
/// in `cluster_sizes`. Empty / zero-population runs produce `r = 0`.
pub fn quantify_converged(
    report: &crate::analysis::opinion_dynamics::OpinionDynamicsReport,
    quantifier: Quantifier,
) -> f64 {
    let total: usize = report.cluster_sizes.iter().sum();
    if total == 0 {
        return evaluate(quantifier, 0.0);
    }
    let largest: usize = report.cluster_sizes.iter().copied().max().unwrap_or(0);
    let r = largest as f64 / total as f64;
    evaluate(quantifier, r)
}

/// Helper used by the disinfo subsystem (when present) to fuse a
/// per-actor predicate into a single quantifier scalar — "do *most
/// actors* in the cluster exhibit X?". Re-exports [`evaluate_over_entities`]
/// with [`EntityType::Actor`] pre-bound.
pub fn quantify_over_actors<F>(
    hg: &Hypergraph,
    narrative_id: &str,
    predicate: F,
    quantifier: Quantifier,
) -> Result<f64>
where
    F: Fn(&Entity) -> f64,
{
    evaluate_over_entities(hg, narrative_id, Some(EntityType::Actor), predicate, quantifier)
}

// ── Declarative predicate spec for alerts + REST ────────────────────────────
//
// The REST / alerts surface carries the quantifier request as data rather
// than a closure — callers serialize a tiny spec and the handler resolves
// it into a predicate closure at evaluation time.

/// A minimal serde-carried quantifier condition body. Used by the REST
/// endpoint and by [`crate::analysis::alerts::AlertRule`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantifierCondition {
    /// Canonical predicate spec (e.g. `"confidence>0.7"` or a free-form
    /// tag the application layer resolves).
    pub predicate: String,
    /// Canonical quantifier name (`"most"`, `"many"`, ...).
    pub quantifier: String,
    /// Threshold that the evaluated `Q(r)` must meet or exceed for the
    /// rule to fire. Callers typically use `0.5` for "half-strength
    /// truth" — the exact calibration is application-specific.
    pub threshold: f64,
}

/// Compute a stable hex predicate-hash for cache keys from a REST body.
///
/// Hashes `{predicate, quantifier, entity_type?}` — so two calls that
/// differ only in label reuse the cache. Hash is SHA-256; 16 hex chars
/// of the digest are enough for collision-resistance at the per-narrative
/// cache scale and keep keys short.
pub fn predicate_hash(
    predicate: &str,
    quantifier: Quantifier,
    entity_type: Option<&str>,
) -> String {
    use std::fmt::Write;

    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(predicate.as_bytes());
    h.update(b"|");
    h.update(quantifier.name().as_bytes());
    h.update(b"|");
    if let Some(t) = entity_type {
        h.update(t.as_bytes());
    }
    let digest = h.finalize();
    let mut out = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
