//! Fuzzy-probabilistic hybrid inference — Phase 10 base case.
//!
//! Ships the Cao–Holčapek-Valášek 2026 FSTA base case. A fuzzy event
//! `E` is a graded predicate `μ_E : X → [0, 1]` over a finite domain
//! of entity UUIDs; a discrete probability mass function assigns
//! `P(e) ∈ [0, 1]` per outcome with `Σ P = 1`; the Sugeno-additive
//! fuzzy probability is
//!
//! ```text
//!   P_fuzzy(E) = Σ_{e ∈ outcomes} μ_E(e) · P(e)
//! ```
//!
//! which reduces bit-identically to the classical `P(A)` on crisp
//! indicator μ.
//!
//! Scope-capped by design — continuous distributions, modal-logic
//! embedding of Flaminio 2026, Fagin–Halpern multi-agent fuzzy
//! probability, and decision-theoretic query layers are all deferred
//! per [`docs/fuzzy_hybrid_algorithm.md`] §1.
//!
//! Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].

use std::collections::HashMap;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::rules::evaluate_rules_against_entity;
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;
use crate::types::{Entity, EntityType};

/// Tolerance used to validate that a discrete distribution sums to 1.
const PROB_SUM_TOLERANCE: f64 = 1e-9;

/// Which predicate surface the caller is using to grade each outcome.
///
/// Each variant reuses an existing Phase-6/9 surface rather than
/// introducing a third predicate language. `Custom` lets a caller
/// pre-compute memberships and ship them as data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzyEventPredicate {
    /// Phase 6 intermediate quantifier. Payload shape:
    /// `{ "quantifier": "<name>", "where": "<spec>", "entity_type"?: "<type>" }`.
    Quantifier,
    /// Phase 9 Mamdani rule. Payload shape:
    /// `{ "rule_id": "<uuid>", "narrative_id": "<nid>" }`.
    MamdaniRule,
    /// Pre-computed per-entity memberships. Payload shape:
    /// `{ "memberships": { "<uuid>": f64, ... } }`. Missing entries
    /// resolve to `0.0`.
    Custom,
}

/// A graded event carried on the REST / TensaQL wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyEvent {
    /// Which predicate surface to use when evaluating μ per outcome.
    pub predicate_kind: FuzzyEventPredicate,
    /// Free-form payload interpreted by the dispatcher — see the
    /// [`FuzzyEventPredicate`] variant docs for each shape.
    pub predicate_payload: serde_json::Value,
}

/// Discrete probability mass function over entity UUIDs.
///
/// Phase 10 ships a single variant. Continuous (`Gaussian`,
/// `KernelDensity`, …) is Phase 10.5 scope per the design doc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProbDist {
    /// Outcomes paired with probabilities. `Σ P ≈ 1 ± 1e-9`, each
    /// `P ∈ [0, 1]`, no duplicate UUIDs.
    Discrete { outcomes: Vec<(Uuid, f64)> },
}

impl ProbDist {
    /// Validate shape: `Σ P ≈ 1`, `P ∈ [0, 1]`, no duplicates.
    /// Empty domain is accepted so callers can short-circuit to 0 at
    /// the caller rather than surface a 400 on an empty narrative.
    pub fn validate(&self) -> Result<()> {
        match self {
            ProbDist::Discrete { outcomes } => {
                if outcomes.is_empty() {
                    return Ok(());
                }
                let mut seen: HashMap<Uuid, bool> = HashMap::with_capacity(outcomes.len());
                let mut sum = 0.0_f64;
                for (id, p) in outcomes {
                    if !p.is_finite() || *p < 0.0 || *p > 1.0 {
                        return Err(TensaError::InvalidInput(format!(
                            "probability for {id} out of range [0,1] or non-finite: {p}"
                        )));
                    }
                    if seen.insert(*id, true).is_some() {
                        return Err(TensaError::InvalidInput(format!(
                            "duplicate outcome UUID {id} in distribution"
                        )));
                    }
                    sum += *p;
                }
                if (sum - 1.0).abs() > PROB_SUM_TOLERANCE {
                    return Err(TensaError::InvalidInput(format!(
                        "discrete distribution must sum to 1.0 (±{PROB_SUM_TOLERANCE}); got {sum}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// Convenience: returns the outcomes slice if discrete.
    pub fn outcomes(&self) -> &[(Uuid, f64)] {
        match self {
            ProbDist::Discrete { outcomes } => outcomes,
        }
    }
}

// ── Persistent report ────────────────────────────────────────────────────────

/// Descriptor shape persisted at `fz/hybrid/{nid}/{query_id_BE_16}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridProbabilityReport {
    /// v7 UUID — chronological key ordering.
    pub query_id: Uuid,
    /// Narrative the event was evaluated against.
    pub narrative_id: String,
    /// Resulting Sugeno-additive scalar.
    pub value: f64,
    /// Echoed event kind for tooling.
    pub event_kind: String,
    /// Short distribution summary — `kind + num_outcomes`.
    pub distribution_summary: String,
    /// T-norm name (Phase 10 records the request even though the base
    /// case does not consume it; future composition phases will).
    pub tnorm: String,
    /// Created-at stamp.
    pub created_at: DateTime<Utc>,
}

// ── Core semantics ───────────────────────────────────────────────────────────

/// Evaluate `P_fuzzy(E) = Σ μ_E(e) · P(e)` in the Cao–Holčapek base case.
///
/// * `hg` — live hypergraph handle (entity lookups for Quantifier /
///   MamdaniRule predicates).
/// * `narrative_id` — narrative the outcomes belong to. Used when
///   loading entities for Quantifier / MamdaniRule evaluation.
/// * `event` — graded event spec; see [`FuzzyEventPredicate`].
/// * `distribution` — probability mass over outcome entity UUIDs.
/// * `tnorm` — accepted for forward-compat, ignored in Phase 10. Future
///   `P_fuzzy(E₁ ∧ E₂)` composition consumes it.
///
/// Returns a scalar in `[0, 1]`. Empty distributions short-circuit to
/// `0.0` (vacuous — nothing has any mass).
pub fn fuzzy_probability(
    hg: &Hypergraph,
    _narrative_id: &str,
    event: &FuzzyEvent,
    distribution: &ProbDist,
    _tnorm: TNormKind,
) -> Result<f64> {
    distribution.validate()?;
    let outcomes = distribution.outcomes();
    if outcomes.is_empty() {
        return Ok(0.0);
    }

    let dispatcher = MembershipDispatcher::prepare(event)?;
    let mut acc = 0.0_f64;
    for (entity_id, p) in outcomes {
        let m = dispatcher.mu(hg, entity_id)?.clamp(0.0, 1.0);
        acc += m * p;
    }
    Ok(acc.clamp(0.0, 1.0))
}

/// Pre-parsed predicate state — parses the payload once at the top of
/// the reduce and evaluates per-outcome in the reduce loop. Keeps the
/// cost proportional to `|outcomes|` rather than `|narrative|`.
enum MembershipDispatcher {
    Custom(HashMap<Uuid, f64>),
    Quantifier {
        where_spec: String,
        entity_type: Option<EntityType>,
    },
    MamdaniRule {
        rule_id: String,
        narrative_id: String,
    },
}

impl MembershipDispatcher {
    fn prepare(event: &FuzzyEvent) -> Result<Self> {
        match event.predicate_kind {
            FuzzyEventPredicate::Custom => {
                Ok(Self::Custom(parse_custom_memberships(&event.predicate_payload)?))
            }
            FuzzyEventPredicate::Quantifier => {
                let payload = &event.predicate_payload;
                let where_spec = payload
                    .get("where")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let entity_type = match payload.get("entity_type").and_then(|v| v.as_str()) {
                    None | Some("") => None,
                    Some(s) => Some(EntityType::from_str(s).map_err(|e| {
                        TensaError::InvalidInput(format!(
                            "Quantifier predicate entity_type '{s}': {e}"
                        ))
                    })?),
                };
                Ok(Self::Quantifier { where_spec, entity_type })
            }
            FuzzyEventPredicate::MamdaniRule => {
                let payload = &event.predicate_payload;
                let rule_id = payload
                    .get("rule_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TensaError::InvalidInput(
                            "MamdaniRule predicate payload missing `rule_id`".into(),
                        )
                    })?
                    .to_string();
                Uuid::parse_str(&rule_id).map_err(|e| {
                    TensaError::InvalidInput(format!(
                        "MamdaniRule predicate rule_id '{rule_id}' is not a UUID: {e}"
                    ))
                })?;
                let narrative_id = payload
                    .get("narrative_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        TensaError::InvalidInput(
                            "MamdaniRule predicate payload missing `narrative_id`".into(),
                        )
                    })?
                    .to_string();
                Ok(Self::MamdaniRule { rule_id, narrative_id })
            }
        }
    }

    fn mu(&self, hg: &Hypergraph, id: &Uuid) -> Result<f64> {
        match self {
            Self::Custom(map) => Ok(map.get(id).copied().unwrap_or(0.0)),
            Self::Quantifier { where_spec, entity_type } => {
                let entity = match hg.get_entity(id) {
                    Ok(e) => e,
                    Err(_) => return Ok(0.0),
                };
                if let Some(t) = entity_type {
                    if entity.entity_type != *t {
                        return Ok(0.0);
                    }
                }
                Ok(if crisp_where_match(&entity, where_spec) {
                    1.0
                } else {
                    0.0
                })
            }
            Self::MamdaniRule { rule_id, narrative_id } => {
                // Outcome entity may live in the query's narrative or
                // in the rule's narrative; the rule-eval helper needs
                // the rule's narrative for the KV lookup.
                let rule_ids = [rule_id.clone()];
                let eval =
                    match evaluate_rules_against_entity(hg, narrative_id, Some(&rule_ids), id) {
                        Ok(e) => e,
                        Err(_) => return Ok(0.0),
                    };
                Ok(eval
                    .fired_rules
                    .iter()
                    .map(|f| f.firing_strength)
                    .fold(0.0_f64, f64::max)
                    .clamp(0.0, 1.0))
            }
        }
    }
}

fn parse_custom_memberships(payload: &serde_json::Value) -> Result<HashMap<Uuid, f64>> {
    let map = payload.get("memberships").ok_or_else(|| {
        TensaError::InvalidInput("Custom predicate payload missing `memberships` map".into())
    })?;
    let obj = map.as_object().ok_or_else(|| {
        TensaError::InvalidInput(
            "Custom predicate `memberships` must be a JSON object keyed by UUID".into(),
        )
    })?;
    let mut out = HashMap::with_capacity(obj.len());
    for (k, v) in obj {
        let id = Uuid::parse_str(k).map_err(|e| {
            TensaError::InvalidInput(format!(
                "Custom predicate memberships key '{k}' is not a UUID: {e}"
            ))
        })?;
        let mu = v.as_f64().ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "Custom predicate memberships['{k}'] must be a number"
            ))
        })?;
        out.insert(id, mu.clamp(0.0, 1.0));
    }
    Ok(out)
}

/// Tiny crisp comparator used for Quantifier-predicate WHERE evaluation.
/// Reuses the same `<field> <op> <value>` mini-language as the Phase 6
/// quantify endpoint. Empty spec → always true (every entity
/// contributes μ = 1.0).
fn crisp_where_match(entity: &Entity, spec: &str) -> bool {
    let s = spec.trim();
    if s.is_empty() {
        return true;
    }
    let (field, op, rhs) = match split_comparison(s) {
        Some(t) => t,
        None => return false,
    };
    let field = field.trim();
    let rhs = rhs.trim();
    match field {
        "confidence" => {
            let threshold: f64 = rhs.parse().unwrap_or(0.0);
            compare_f64(op, entity.confidence as f64, threshold)
        }
        _ => property_path_matches(&entity.properties, field, op, rhs),
    }
}

fn split_comparison(s: &str) -> Option<(&str, &str, &str)> {
    for op in &[">=", "<=", "!=", ">", "<", "="] {
        if let Some(idx) = s.find(op) {
            let (lhs, rest) = s.split_at(idx);
            let rhs = &rest[op.len()..];
            return Some((lhs, op, rhs));
        }
    }
    None
}

fn compare_f64(op: &str, lhs: f64, rhs: f64) -> bool {
    match op {
        ">=" => lhs >= rhs,
        "<=" => lhs <= rhs,
        ">" => lhs > rhs,
        "<" => lhs < rhs,
        "=" => (lhs - rhs).abs() < f64::EPSILON,
        "!=" => (lhs - rhs).abs() >= f64::EPSILON,
        _ => false,
    }
}

fn property_path_matches(props: &serde_json::Value, field: &str, op: &str, rhs: &str) -> bool {
    let mut cur = props;
    for seg in field.split('.') {
        match cur {
            serde_json::Value::Object(o) => match o.get(seg) {
                Some(v) => cur = v,
                None => return false,
            },
            _ => return false,
        }
    }
    match cur {
        serde_json::Value::Number(n) => {
            let v = n.as_f64().unwrap_or(0.0);
            let threshold: f64 = rhs.parse().unwrap_or(0.0);
            compare_f64(op, v, threshold)
        }
        serde_json::Value::String(s) => {
            let stripped = rhs.trim_matches('"');
            match op {
                "=" => s == stripped,
                "!=" => s != stripped,
                _ => false,
            }
        }
        serde_json::Value::Bool(b) => match (op, rhs.parse::<bool>()) {
            ("=", Ok(want)) => *b == want,
            ("!=", Ok(want)) => *b != want,
            _ => false,
        },
        _ => false,
    }
}

// ── KV persistence ───────────────────────────────────────────────────────────

/// Build a [`HybridProbabilityReport`] record from a computed scalar
/// and the incoming request shape.
pub fn build_hybrid_report(
    narrative_id: &str,
    event: &FuzzyEvent,
    distribution: &ProbDist,
    tnorm: TNormKind,
    value: f64,
) -> HybridProbabilityReport {
    let event_kind = match event.predicate_kind {
        FuzzyEventPredicate::Quantifier => "quantifier",
        FuzzyEventPredicate::MamdaniRule => "mamdani_rule",
        FuzzyEventPredicate::Custom => "custom",
    }
    .to_string();
    let distribution_summary = match distribution {
        ProbDist::Discrete { outcomes } => format!("discrete:{}", outcomes.len()),
    };
    HybridProbabilityReport {
        query_id: Uuid::now_v7(),
        narrative_id: narrative_id.to_string(),
        value,
        event_kind,
        distribution_summary,
        tnorm: tnorm.name().to_string(),
        created_at: Utc::now(),
    }
}

/// Persist a hybrid report at `fz/hybrid/{nid}/{query_id_BE_16}`.
pub fn save_hybrid_result(
    store: &dyn KVStore,
    narrative_id: &str,
    report: &HybridProbabilityReport,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_hybrid(narrative_id, &report.query_id);
    let bytes = serde_json::to_vec(report).map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(&key, &bytes)
}

/// Load a hybrid report, `Ok(None)` when absent.
pub fn load_hybrid_result(
    store: &dyn KVStore,
    narrative_id: &str,
    query_id: &Uuid,
) -> Result<Option<HybridProbabilityReport>> {
    let key = crate::fuzzy::key_fuzzy_hybrid(narrative_id, query_id);
    match store.get(&key)? {
        Some(bytes) => {
            let r: HybridProbabilityReport = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(r))
        }
        None => Ok(None),
    }
}

/// Delete a hybrid report (idempotent — absent keys return `Ok(())`).
pub fn delete_hybrid_result(
    store: &dyn KVStore,
    narrative_id: &str,
    query_id: &Uuid,
) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_hybrid(narrative_id, query_id);
    store.delete(&key)
}

/// Scan every hybrid report for a narrative, newest-first (v7 UUIDs
/// sort chronologically; we reverse). Malformed rows are skipped with
/// a `warn!` so one corrupt blob doesn't poison the list.
pub fn list_hybrid_results_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<HybridProbabilityReport>> {
    let mut prefix = crate::fuzzy::FUZZY_HYBRID_PREFIX.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let pairs = store.prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, v) in pairs {
        match serde_json::from_slice::<HybridProbabilityReport>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!(
                narrative_id = %narrative_id,
                "hybrid-probability record deserialize failed ({e}); skipping"
            ),
        }
    }
    out.sort_by(|a, b| b.query_id.cmp(&a.query_id));
    Ok(out)
}
