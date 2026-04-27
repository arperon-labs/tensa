//! Canonical hashing for surrogate-run reproducibility + significance dedup.
//!
//! Two hashes both backed by `sha2::Sha256` (already in tree — see Cargo.toml
//! audit, `blake3` would have added a new dep for negligible benefit at
//! Phase 0 sizes).
//!
//! ## Canonical-JSON contract (LOAD-BEARING — never change silently)
//!
//! 1. Object keys sorted lexicographically at every nesting level.
//! 2. No insignificant whitespace.
//! 3. Numbers in shortest round-trippable form (delegated to `serde_json`).
//! 4. Strings UTF-8. **NFC normalization is deferred** — the
//!    `unicode-normalization` crate is not in tree, and Phase 0 narrative ids
//!    are ASCII slugs in practice. Tracked under "Deferred Items" in the
//!    Phase 0 report; revisit when an actual non-ASCII narrative id surfaces.
//!
//! Bumping the contract requires a sweep: any KV record keyed by one of these
//! hashes (significance dedup, run signatures) becomes orphaned.

use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::hypergraph::Hypergraph;

use super::types::SurrogateParams;

/// Canonical-JSON hash of `params` after JSON round-trip — sort keys at every
/// level, strip whitespace, lowercase hex output. Two `SurrogateParams`
/// constructed in any order with equal values produce the same hash.
pub fn canonical_params_hash(params: &SurrogateParams) -> String {
    let value = match serde_json::to_value(params) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    let canonical = canonicalize_value(&value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    hex_lower(&hasher.finalize())
}

/// Canonical hash of a narrative's *current state*: sorted entity ids,
/// sorted situation ids, sorted participation tuples, plus the most recent
/// `updated_at` timestamp across all of them. Used by Phase 11 "Reproduce
/// this run" to detect drift since a prior run was recorded.
pub fn canonical_narrative_state_hash(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<String> {
    let mut entity_ids: Vec<String> = hypergraph
        .list_entities_by_narrative(narrative_id)?
        .into_iter()
        .map(|e| e.id.to_string())
        .collect();
    entity_ids.sort_unstable();

    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let mut situation_ids: Vec<String> =
        situations.iter().map(|s| s.id.to_string()).collect();
    situation_ids.sort_unstable();

    // Participation tuples: `(entity_id, situation_id, seq)` rendered as
    // strings so they sort lexicographically and there's no ambiguity about
    // numeric vs string comparison.
    let mut participation_tuples: Vec<String> = Vec::new();
    for sit in &situations {
        let parts = hypergraph.get_participants_for_situation(&sit.id)?;
        for p in parts {
            participation_tuples
                .push(format!("{}|{}|{}", p.entity_id, p.situation_id, p.seq));
        }
    }
    participation_tuples.sort_unstable();

    // Most recent update across entities + situations. Missing values count
    // as epoch so an empty narrative still hashes consistently.
    let mut last_modified_max: i64 = 0;
    for e in hypergraph.list_entities_by_narrative(narrative_id)? {
        last_modified_max = last_modified_max.max(e.updated_at.timestamp_millis());
    }
    for s in &situations {
        last_modified_max = last_modified_max.max(s.updated_at.timestamp_millis());
    }

    let payload = serde_json::json!({
        "entity_ids": entity_ids,
        "situation_ids": situation_ids,
        "participations": participation_tuples,
        "last_modified_max": last_modified_max,
    });
    let canonical = canonicalize_value(&payload);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(hex_lower(&hasher.finalize()))
}

// ── canonical-JSON helpers ───────────────────────────────────────────────────

fn canonicalize_value(v: &serde_json::Value) -> String {
    let mut buf = String::new();
    write_canonical(&mut buf, v);
    buf
}

fn write_canonical(out: &mut String, v: &serde_json::Value) {
    use serde_json::Value;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k.as_str()).unwrap_or_default());
                out.push(':');
                if let Some(val) = map.get(*k) {
                    write_canonical(out, val);
                }
            }
            out.push('}');
        }
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::Hypergraph;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn test_canonical_hash_stable_across_key_order() {
        // Two params built in different field-insertion orders should hash
        // identically (canonical-JSON sorts keys at every nesting level).
        let p1 = SurrogateParams {
            model: "eath".into(),
            params_json: serde_json::json!({
                "alpha": 1,
                "beta": [1, 2, 3],
                "nested": { "x": 10, "y": 20 },
            }),
            seed: 42,
            num_steps: 100,
            label_prefix: "synth".into(),
        };
        let p2 = SurrogateParams {
            model: "eath".into(),
            // Same values, different child-object key order:
            params_json: serde_json::json!({
                "nested": { "y": 20, "x": 10 },
                "beta": [1, 2, 3],
                "alpha": 1,
            }),
            seed: 42,
            num_steps: 100,
            label_prefix: "synth".into(),
        };
        let h1 = canonical_params_hash(&p1);
        let h2 = canonical_params_hash(&p2);
        assert_eq!(h1, h2, "key-order changes must not change the hash");
        assert_eq!(h1.len(), 64, "sha256 hex digest is 64 chars");
    }

    #[test]
    fn test_canonical_hash_changes_on_value_change() {
        let p1 = SurrogateParams {
            model: "eath".into(),
            params_json: serde_json::json!({"a": 1}),
            seed: 42,
            num_steps: 100,
            label_prefix: "synth".into(),
        };
        let mut p2 = p1.clone();
        p2.params_json = serde_json::json!({"a": 2});
        assert_ne!(canonical_params_hash(&p1), canonical_params_hash(&p2));
    }

    #[test]
    fn test_narrative_state_hash_changes_when_entity_added() {
        let store: Arc<dyn crate::store::KVStore> = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let nid = "test-narr";

        let h_empty = canonical_narrative_state_hash(&hg, nid).unwrap();

        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Alice"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some(nid.into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap();

        let h_after = canonical_narrative_state_hash(&hg, nid).unwrap();
        assert_ne!(
            h_empty, h_after,
            "adding an entity must change the state hash"
        );
    }
}
