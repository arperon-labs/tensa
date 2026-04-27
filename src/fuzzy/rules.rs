//! Mamdani-style fuzzy rule systems — "IF premises THEN consequent"
//! with rule-strength computed under a configurable t-norm.
//!
//! Phase 9. Type definitions live in [`super::rules_types`], the
//! evaluation pipeline in [`super::rules_eval`], and KV persistence in
//! [`super::rules_store`]. This module re-exports the public surface
//! and houses the high-level helper functions that thread the three
//! layers together for the REST / TensaQL / workflow wires.
//!
//! ### Reference example
//!
//! ```ignore
//! use tensa::fuzzy::rules::*;
//! let rule = build_rule(
//!     "elevated-disinfo-risk",
//!     "narrative-42",
//!     vec![
//!         FuzzyCondition {
//!             variable_path: "entity.properties.inflammatory_score".into(),
//!             membership: MembershipFunction::Trapezoidal { a: 0.5, b: 0.7, c: 0.9, d: 1.0 },
//!             linguistic_term: "high".into(),
//!         },
//!     ],
//!     FuzzyOutput {
//!         variable: "disinfo_risk".into(),
//!         membership: MembershipFunction::Gaussian { mean: 0.8, sigma: 0.1 },
//!         linguistic_term: "elevated".into(),
//!     },
//! );
//! ```
//!
//! Cites: [mamdani1975mamdani].

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::rules_store as store_ops;
use crate::fuzzy::rules_types::{MamdaniRule, MamdaniRuleSet};
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;

pub use crate::fuzzy::rules_eval::{
    aggregate_consequents, build_rule, centroid_defuzz, defuzzify, evaluate_rule_set,
    evaluate_single_rule, firing_strength, fuzzify_condition, mean_of_maxima_defuzz,
    resolve_variable, validate_rule,
};
pub use crate::fuzzy::rules_store::{
    delete_rule, find_rule_by_id_anywhere as find_rule_any_narrative,
    list_rules_for_narrative as list_rules, load_rule as load_rule_fn, save_rule,
};
pub use crate::fuzzy::rules_types::{
    Defuzzification, FiredRule, FuzzyCondition, FuzzyOutput, MamdaniRule as Rule,
    MamdaniRuleSet as RuleSet, MembershipFunction, RuleSetEvaluation,
    RuleSetEvaluation as Evaluation, DEFAULT_DEFUZZ_BINS,
};

// ── High-level entry points used by the REST + TensaQL surfaces ─────────────

/// Build a rule set from a list of `rule_ids` (scoped to `narrative_id`)
/// and evaluate it against one entity. `rule_ids = None` loads every
/// rule for the narrative (enabled + disabled); disabled rules are
/// filtered out inside [`evaluate_rule_set`].
pub fn evaluate_rules_against_entity(
    hg: &Hypergraph,
    narrative_id: &str,
    rule_ids: Option<&[String]>,
    entity_id: &Uuid,
) -> Result<RuleSetEvaluation> {
    let store = hg.store();
    let entity = hg.get_entity(entity_id)?;
    let rules = collect_rules(store, narrative_id, rule_ids)?;
    let set = MamdaniRuleSet::new(rules);
    evaluate_rule_set(&set, &entity)
}

/// Evaluate every entity in a narrative against a shared rule set —
/// used by the TensaQL `EVALUATE RULES FOR "<nid>" AGAINST (e:Actor)`
/// verb.
pub fn evaluate_rules_over_narrative(
    hg: &Hypergraph,
    narrative_id: &str,
    rule_ids: Option<&[String]>,
    entity_type_filter: Option<crate::types::EntityType>,
) -> Result<Vec<RuleSetEvaluation>> {
    let store = hg.store();
    let rules = collect_rules(store, narrative_id, rule_ids)?;
    if rules.is_empty() {
        return Ok(Vec::new());
    }
    let set = MamdaniRuleSet::new(rules);
    let mut entities = hg.list_entities_by_narrative(narrative_id)?;
    if let Some(et) = entity_type_filter {
        entities.retain(|e| e.entity_type == et);
    }
    let mut out = Vec::with_capacity(entities.len());
    for e in &entities {
        let ev = evaluate_rule_set(&set, e)?;
        out.push(ev);
    }
    Ok(out)
}

fn collect_rules(
    store: &dyn KVStore,
    narrative_id: &str,
    rule_ids: Option<&[String]>,
) -> Result<Vec<MamdaniRule>> {
    let Some(ids) = rule_ids else {
        return store_ops::list_rules_for_narrative(store, narrative_id);
    };
    let mut out = Vec::with_capacity(ids.len());
    for id_str in ids {
        let uuid: Uuid = id_str.parse().map_err(|e| {
            TensaError::InvalidInput(format!("rule_id '{id_str}' not a UUID: {e}"))
        })?;
        // Prefer narrative-scoped load; fall back to a cross-narrative scan
        // so analysts can reuse a rule authored against another narrative
        // without duplicating it.
        let rule = match store_ops::load_rule(store, narrative_id, &uuid)? {
            Some(r) => r,
            None => store_ops::find_rule_by_id_anywhere(store, &uuid)?
                .ok_or_else(|| TensaError::NotFound(format!("Mamdani rule {uuid} not persisted")))?,
        };
        out.push(rule);
    }
    Ok(out)
}

/// Dispatch helper used by the `InferenceJobType::FuzzyRuleEvaluate`
/// engine. Phase 9 covers the "entity target required" path; narrative-
/// scoped sweep falls through to [`evaluate_rules_over_narrative`] when
/// `entity_id = None`.
pub fn dispatch_fuzzy_rule_evaluate(
    hg: &Hypergraph,
    narrative_id: &str,
    rule_id: &str,
    entity_id: Option<Uuid>,
) -> Result<serde_json::Value> {
    let rule_ids: Vec<String> = vec![rule_id.to_string()];
    match entity_id {
        Some(eid) => {
            let ev = evaluate_rules_against_entity(hg, narrative_id, Some(&rule_ids), &eid)?;
            serde_json::to_value(ev).map_err(|e| TensaError::Serialization(e.to_string()))
        }
        None => {
            let evs =
                evaluate_rules_over_narrative(hg, narrative_id, Some(&rule_ids), None)?;
            serde_json::to_value(evs).map_err(|e| TensaError::Serialization(e.to_string()))
        }
    }
}
