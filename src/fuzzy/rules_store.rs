//! Mamdani rule — KV persistence helpers.
//!
//! All rules persist at
//! `fz/rules/{narrative_id_utf8}/{rule_id_v7_BE_BIN_16}` via
//! [`super::key_fuzzy_rule`]. Split from [`super::rules`] for the
//! 500-line cap. Phase 9.
//!
//! Cites: [mamdani1975mamdani].

use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::rules_types::MamdaniRule;
use crate::store::KVStore;

/// Persist a rule at `fz/rules/{narrative_id}/{rule_id}`. `rule.narrative_id`
/// is the source of record — callers pass the id explicitly so the
/// builder and the field cannot drift.
pub fn save_rule(store: &dyn KVStore, narrative_id: &str, rule: &MamdaniRule) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_rule(narrative_id, &rule.id);
    let bytes = serde_json::to_vec(rule).map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(&key, &bytes)
}

/// Load a persisted rule, or `None` when the key does not exist.
pub fn load_rule(
    store: &dyn KVStore,
    narrative_id: &str,
    rule_id: &Uuid,
) -> Result<Option<MamdaniRule>> {
    let key = crate::fuzzy::key_fuzzy_rule(narrative_id, rule_id);
    match store.get(&key)? {
        Some(bytes) => {
            let rule: MamdaniRule = serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?;
            Ok(Some(rule))
        }
        None => Ok(None),
    }
}

/// Delete a persisted rule (idempotent — absent keys return `Ok(())`).
pub fn delete_rule(store: &dyn KVStore, narrative_id: &str, rule_id: &Uuid) -> Result<()> {
    let key = crate::fuzzy::key_fuzzy_rule(narrative_id, rule_id);
    store.delete(&key)
}

/// Scan every persisted rule for a narrative, newest-first (v7 UUIDs
/// sort chronologically, so we reverse the natural order). Malformed
/// rows are skipped with a `warn!`.
pub fn list_rules_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<MamdaniRule>> {
    let mut prefix = crate::fuzzy::FUZZY_RULE_PREFIX.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let pairs = store.prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, v) in pairs {
        match serde_json::from_slice::<MamdaniRule>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!(
                narrative_id = %narrative_id,
                "Mamdani rule deserialize failed ({e}); skipping"
            ),
        }
    }
    out.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(out)
}

/// Scan every rule across every narrative. Used by the alerts engine
/// when a rule is referenced by id without narrative scope. Linear in
/// the total number of persisted rules; bounded by explicit user
/// curation so the scan is cheap in practice.
pub fn find_rule_by_id_anywhere(
    store: &dyn KVStore,
    rule_id: &Uuid,
) -> Result<Option<MamdaniRule>> {
    let pairs = store.prefix_scan(crate::fuzzy::FUZZY_RULE_PREFIX)?;
    for (_, v) in pairs {
        if let Ok(r) = serde_json::from_slice::<MamdaniRule>(&v) {
            if r.id == *rule_id {
                return Ok(Some(r));
            }
        }
    }
    Ok(None)
}
