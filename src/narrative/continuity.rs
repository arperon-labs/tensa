//! Continuity — pinned facts + pre-apply warning generation (Sprint W4, v0.49.3).
//!
//! The writer marks certain property values as canonical via [`PinnedFact`]s
//! (e.g. "Alice's age is 23"). Before a W1 generation or W2 edit proposal is
//! applied, [`check_generation_proposal`] / [`check_edit_proposal`] compare
//! it against the pinned facts and surface [`ContinuityWarning`]s.
//!
//! # Simplify
//! - Pinned facts are a plain KV table at `pf/{narrative_id}/{fact_id}`; no
//!   secondary index (small cardinality per narrative).
//! - Continuity checks are deterministic and fast: property-value comparisons
//!   for entity-scoped facts; substring scan of proposed prose for narrative-
//!   wide facts. No LLM calls.
//! - Warnings are advisory, not blocking — the apply path ignores them; the
//!   Studio surfaces them so the writer can cancel or override.

use std::collections::HashMap;

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::narrative::editing::EditProposal;
use crate::narrative::generation::GenerationProposal;
use crate::store::KVStore;
use crate::types::*;

// ─── Pinned fact storage ──────────────────────────────────────────

pub fn list_pinned_facts(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<PinnedFact>> {
    let prefix = keys::pinned_fact_narrative_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    let mut out: Vec<PinnedFact> = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        out.push(serde_json::from_slice(&value)?);
    }
    // Deterministic ordering: by entity_id then key.
    out.sort_by(|a, b| {
        a.entity_id
            .cmp(&b.entity_id)
            .then_with(|| a.key.cmp(&b.key))
    });
    Ok(out)
}

pub fn get_pinned_fact(
    store: &dyn KVStore,
    narrative_id: &str,
    fact_id: &Uuid,
) -> Result<PinnedFact> {
    let key = keys::pinned_fact_key(narrative_id, fact_id);
    match store.get(&key)? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::NotFound(format!("pinned fact {}", fact_id))),
    }
}

/// Create a new pinned fact. Generates a fresh v7 UUID; `id`/`created_at`/
/// `updated_at` on the input are ignored.
pub fn create_pinned_fact(store: &dyn KVStore, mut fact: PinnedFact) -> Result<PinnedFact> {
    validate_fact(&fact)?;
    let now = Utc::now();
    fact.id = Uuid::now_v7();
    fact.created_at = now;
    fact.updated_at = now;
    let key = keys::pinned_fact_key(&fact.narrative_id, &fact.id);
    let bytes = serde_json::to_vec(&fact)?;
    store.put(&key, &bytes)?;
    Ok(fact)
}

pub fn update_pinned_fact(
    store: &dyn KVStore,
    narrative_id: &str,
    fact_id: &Uuid,
    patch: PinnedFactPatch,
) -> Result<PinnedFact> {
    let mut fact = get_pinned_fact(store, narrative_id, fact_id)?;
    if let Some(k) = patch.key {
        if k.trim().is_empty() {
            return Err(TensaError::InvalidQuery(
                "pinned fact key cannot be empty".into(),
            ));
        }
        fact.key = k;
    }
    if let Some(v) = patch.value {
        if v.trim().is_empty() {
            return Err(TensaError::InvalidQuery(
                "pinned fact value cannot be empty".into(),
            ));
        }
        fact.value = v;
    }
    if let Some(note) = patch.note {
        fact.note = note;
    }
    if let Some(entity_id) = patch.entity_id {
        fact.entity_id = entity_id;
    }
    fact.updated_at = Utc::now();
    let key = keys::pinned_fact_key(narrative_id, fact_id);
    let bytes = serde_json::to_vec(&fact)?;
    store.put(&key, &bytes)?;
    Ok(fact)
}

pub fn delete_pinned_fact(store: &dyn KVStore, narrative_id: &str, fact_id: &Uuid) -> Result<()> {
    let key = keys::pinned_fact_key(narrative_id, fact_id);
    store.delete(&key)
}

#[derive(Debug, Default)]
pub struct PinnedFactPatch {
    pub key: Option<String>,
    pub value: Option<String>,
    pub note: Option<Option<String>>,
    pub entity_id: Option<Option<Uuid>>,
}

fn validate_fact(f: &PinnedFact) -> Result<()> {
    if f.narrative_id.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "pinned fact narrative_id is required".into(),
        ));
    }
    if f.key.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "pinned fact key is required".into(),
        ));
    }
    if f.value.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "pinned fact value is required".into(),
        ));
    }
    Ok(())
}

// ─── Continuity checks ────────────────────────────────────────────

/// Check a generation proposal against the current narrative's pinned facts
/// and existing cast. Returns advisory warnings; never errors on content.
pub fn check_generation_proposal(
    hypergraph: &Hypergraph,
    proposal: &GenerationProposal,
) -> Result<Vec<ContinuityWarning>> {
    let facts = list_pinned_facts(hypergraph.store(), &proposal.narrative_id)?;
    let existing = hypergraph.list_entities_by_narrative(&proposal.narrative_id)?;
    // HashMap so resolution is O(1) per proposed entity instead of O(existing).
    let name_index: HashMap<String, &Entity> = existing
        .iter()
        .filter_map(|e| {
            e.properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(|n| (n.trim().to_lowercase(), e))
        })
        .collect();
    let mut warnings: Vec<ContinuityWarning> = Vec::new();

    for proposed in &proposal.entities {
        let proposed_name = proposed
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let Some(proposed_name) = proposed_name else {
            continue;
        };

        let target = proposed_name.trim().to_lowercase();
        let Some(&existing_entity) = name_index.get(&target) else {
            continue;
        };

        // For each pinned fact scoped to that entity, compare the proposed
        // value against the pinned one.
        for fact in facts
            .iter()
            .filter(|f| f.entity_id == Some(existing_entity.id))
        {
            if let Some(proposed_value) = proposed
                .properties
                .get(&fact.key)
                .and_then(|v| property_value_to_string(v))
            {
                if !values_compatible(&fact.value, &proposed_value) {
                    warnings.push(ContinuityWarning {
                        severity: ContinuityWarningSeverity::Conflict,
                        headline: format!(
                            "{} conflicts with pinned fact",
                            display_entity(existing_entity)
                        ),
                        detail: format!(
                            "Proposed {} = \"{}\" conflicts with pinned fact {} = \"{}\".",
                            fact.key, proposed_value, fact.key, fact.value
                        ),
                        pinned_fact_id: Some(fact.id),
                        entity_id: Some(existing_entity.id),
                        proposed_value: Some(proposed_value),
                    });
                }
            }
        }
    }

    // Narrative-wide facts: scan the proposed situation descriptions/hints
    // for explicit contradictions (substring match "age: 45" when pinned is
    // "age: 23" etc.). Kept intentionally narrow — v1 only flags obvious
    // key:value phrasings to avoid false positives on prose.
    let narrative_wide: Vec<&PinnedFact> = facts.iter().filter(|f| f.entity_id.is_none()).collect();
    if !narrative_wide.is_empty() {
        let mut prose = String::new();
        for s in &proposal.situations {
            if let Some(d) = &s.description {
                prose.push_str(d);
                prose.push('\n');
            }
            if let Some(h) = &s.raw_content_hint {
                prose.push_str(h);
                prose.push('\n');
            }
        }
        warnings.extend(scan_prose_for_fact_conflicts(&prose, &narrative_wide));
    }

    Ok(warnings)
}

/// Check an edit proposal. For v0.49.3 we scan the proposed prose for
/// explicit `key: value` phrasings that conflict with pinned facts —
/// intentionally narrow to avoid false positives.
pub fn check_edit_proposal(
    hypergraph: &Hypergraph,
    proposal: &EditProposal,
) -> Result<Vec<ContinuityWarning>> {
    let prose = proposal
        .proposed
        .iter()
        .map(|b| b.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    check_prose(hypergraph, &proposal.narrative_id, &prose)
}

/// Scan a raw prose string against a narrative's pinned facts. Narrow helper
/// used by MCP / REST callers that don't want to fabricate an EditProposal.
pub fn check_prose(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    prose: &str,
) -> Result<Vec<ContinuityWarning>> {
    let facts = list_pinned_facts(hypergraph.store(), narrative_id)?;
    if facts.is_empty() {
        return Ok(Vec::new());
    }
    let fact_refs: Vec<&PinnedFact> = facts.iter().collect();
    Ok(scan_prose_for_fact_conflicts(prose, &fact_refs))
}

/// Narrow-scope scanner: looks for `"key: candidate"` or `"key=candidate"`
/// phrasings in prose. When the candidate doesn't match the pinned value,
/// emits an advisory warning. False positives are cheap here (advisory
/// severity, writer can dismiss); false negatives are expected (the scanner
/// can't parse natural language).
fn scan_prose_for_fact_conflicts<'a>(
    prose: &str,
    facts: &[&'a PinnedFact],
) -> Vec<ContinuityWarning> {
    let mut out = Vec::new();
    let lower = prose.to_lowercase();
    for fact in facts {
        let key_lc = fact.key.to_lowercase();
        // Look for "<key>: <something>" or "<key>=<something>" anywhere.
        // Bare ":" is intentionally omitted — it's a prefix of ": " and would
        // double-fire on every "key: value" match.
        for sep in &[": ", "="] {
            let needle = format!("{}{}", key_lc, sep);
            let mut pos = 0;
            while let Some(idx) = lower[pos..].find(&needle) {
                let after = pos + idx + needle.len();
                // Read up to whitespace/punct to get the candidate.
                let candidate: String = lower[after..]
                    .chars()
                    .take_while(|c| !c.is_whitespace() && *c != ',' && *c != '.' && *c != ';')
                    .collect();
                if !candidate.is_empty() && !values_compatible(&fact.value, &candidate) {
                    out.push(ContinuityWarning {
                        severity: ContinuityWarningSeverity::Advisory,
                        headline: format!(
                            "Prose mentions {} conflicting with pinned fact",
                            fact.key
                        ),
                        detail: format!(
                            "Proposed prose contains \"{}{}{}\" but pinned fact says {} = \"{}\".",
                            fact.key, sep, candidate, fact.key, fact.value
                        ),
                        pinned_fact_id: Some(fact.id),
                        entity_id: fact.entity_id,
                        proposed_value: Some(candidate),
                    });
                    // Only flag the first occurrence per key+sep to avoid spam.
                    break;
                }
                pos = after;
            }
        }
    }
    out
}

fn values_compatible(pinned: &str, candidate: &str) -> bool {
    let p = pinned.trim().to_lowercase();
    let c = candidate
        .trim()
        .trim_matches(|ch| matches!(ch, '"' | '\'' | ',' | '.'))
        .to_lowercase();
    if p == c {
        return true;
    }
    // Numeric tolerance: allow within ±1 on integers.
    if let (Ok(pn), Ok(cn)) = (p.parse::<i64>(), c.parse::<i64>()) {
        return (pn - cn).abs() <= 1;
    }
    // Token containment: pinned "john smith" matches candidate "john".
    if p.split_whitespace().any(|tok| tok == c) {
        return true;
    }
    false
}

fn property_value_to_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => Some(v.to_string()),
    }
}

fn display_entity(e: &Entity) -> String {
    e.properties
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| e.id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn setup() -> (Hypergraph, Uuid) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let reg = crate::narrative::registry::NarrativeRegistry::new(store);
        reg.create(Narrative {
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

        let alice_id = Uuid::now_v7();
        hg.create_entity(Entity {
            id: alice_id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Alice", "age": "23"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::HumanEntered),
            narrative_id: Some("draft".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap();
        (hg, alice_id)
    }

    #[test]
    fn create_list_get_delete_roundtrip() {
        let (hg, alice) = setup();
        let fact = create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::nil(),
                narrative_id: "draft".into(),
                entity_id: Some(alice),
                key: "age".into(),
                value: "23".into(),
                note: Some("at story start".into()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();
        assert_ne!(fact.id, Uuid::nil());

        let list = list_pinned_facts(hg.store(), "draft").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].key, "age");

        let fetched = get_pinned_fact(hg.store(), "draft", &fact.id).unwrap();
        assert_eq!(fetched.value, "23");

        delete_pinned_fact(hg.store(), "draft", &fact.id).unwrap();
        assert!(list_pinned_facts(hg.store(), "draft").unwrap().is_empty());
    }

    #[test]
    fn empty_validation_errors() {
        let (hg, _) = setup();
        let err = create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::nil(),
                narrative_id: "draft".into(),
                entity_id: None,
                key: "".into(),
                value: "x".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        );
        assert!(err.is_err());
    }

    #[test]
    fn update_changes_fields_and_bumps_updated_at() {
        let (hg, alice) = setup();
        let fact = create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::nil(),
                narrative_id: "draft".into(),
                entity_id: Some(alice),
                key: "age".into(),
                value: "23".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let updated = update_pinned_fact(
            hg.store(),
            "draft",
            &fact.id,
            PinnedFactPatch {
                value: Some("24".into()),
                note: Some(Some("revised".into())),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(updated.value, "24");
        assert_eq!(updated.note.as_deref(), Some("revised"));
        assert!(updated.updated_at > fact.updated_at);
    }

    #[test]
    fn generation_proposal_conflict_with_pinned_fact() {
        use crate::narrative::generation::{GenerationProposal, ProposedEntity};
        let (hg, alice) = setup();
        create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::nil(),
                narrative_id: "draft".into(),
                entity_id: Some(alice),
                key: "age".into(),
                value: "23".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();

        let proposal = GenerationProposal {
            narrative_id: "draft".into(),
            kind: "character: Alice".into(),
            situations: vec![],
            entities: vec![ProposedEntity {
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "Alice", "age": "45"}),
            }],
            participations: vec![],
            commit_message: None,
            rationale: None,
        };
        let warnings = check_generation_proposal(&hg, &proposal).unwrap();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, ContinuityWarningSeverity::Conflict);
    }

    #[test]
    fn numeric_tolerance_plus_minus_one() {
        let (hg, alice) = setup();
        create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::nil(),
                narrative_id: "draft".into(),
                entity_id: Some(alice),
                key: "age".into(),
                value: "23".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();

        // +1 year is tolerated (could be same-year-ish).
        use crate::narrative::generation::{GenerationProposal, ProposedEntity};
        let proposal = GenerationProposal {
            narrative_id: "draft".into(),
            kind: "character: Alice".into(),
            situations: vec![],
            entities: vec![ProposedEntity {
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": "Alice", "age": "24"}),
            }],
            participations: vec![],
            commit_message: None,
            rationale: None,
        };
        assert!(check_generation_proposal(&hg, &proposal)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn edit_proposal_prose_scan_flags_conflicting_key_value() {
        use crate::narrative::editing::{EditOperation, EditProposal};
        use crate::narrative::revision::DiffLine;

        let (hg, alice) = setup();
        create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::nil(),
                narrative_id: "draft".into(),
                entity_id: Some(alice),
                key: "age".into(),
                value: "23".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();

        let proposal = EditProposal {
            narrative_id: "draft".into(),
            situation_id: Uuid::now_v7(),
            operation: EditOperation::Rewrite {
                instruction: "test".into(),
            },
            original: vec![],
            proposed: vec![ContentBlock {
                content_type: ContentType::Text,
                content: "Alice was a grifter. age: 45, she thought.".into(),
                source: None,
            }],
            original_word_count: 0,
            proposed_word_count: 9,
            diff: Vec::<DiffLine>::new(),
            rationale: None,
        };
        let warnings = check_edit_proposal(&hg, &proposal).unwrap();
        assert!(
            !warnings.is_empty(),
            "should flag prose 'age: 45' vs pinned 'age: 23'"
        );
        assert_eq!(warnings[0].severity, ContinuityWarningSeverity::Advisory);
    }
}
