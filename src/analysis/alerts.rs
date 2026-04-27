//! Alert rules engine for OSINT source reliability monitoring.
//!
//! Alert rules fire reactively when source trust changes, confidence drops,
//! new contentions are detected, or DST conflict exceeds thresholds.
//! Rules are stored in KV and checked from hooks in hypergraph operations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::keys;
use crate::store::KVStore;
use crate::types::EntityType;

// ─── Alert Rule Types ──────────────────────────────────────

/// A persistent alert rule that monitors for specific conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub narrative_id: Option<String>,
    pub rule_type: AlertRuleType,
    pub threshold: f64,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    /// Phase 2 fuzzy wiring — when Some, multi-signal alert rules fold
    /// their per-signal evaluations through this aggregator before
    /// comparing against `threshold`. `None` (serde default) preserves
    /// the pre-sprint single-signal threshold logic.
    ///
    /// Cites: [yager1988owa] [grabisch1996choquet].
    #[serde(default)]
    pub aggregator: Option<crate::fuzzy::aggregation::AggregatorKind>,
    /// Phase 6 fuzzy wiring — when Some, the rule body additionally checks
    /// a quantifier condition (e.g. "fire when *most* actors have
    /// confidence below X"). `None` (serde default) preserves the pre-
    /// sprint single-signal threshold logic.
    ///
    /// Cites: [novak2008quantifiers] [murinovanovak2013syllogisms].
    #[serde(default)]
    pub quantifier_condition: Option<crate::fuzzy::quantifier::QuantifierCondition>,
    /// Phase 9 fuzzy wiring — when Some, references a persisted Mamdani
    /// rule (`fz/rules/.../<rule_id>`). The alert evaluator loads the
    /// rule set and uses its firing-strength output as the gate signal
    /// (fires when firing-strength > `threshold`). `None` (serde
    /// default) preserves the pre-Phase-9 numeric-threshold path.
    ///
    /// Cites: [mamdani1975mamdani].
    #[serde(default)]
    pub mamdani_rule_id: Option<String>,
}

/// The condition type that triggers an alert.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertRuleType {
    /// Source trust_score drops below threshold.
    TrustDrop { source_id: Option<Uuid> },
    /// Entity/situation confidence drops below threshold.
    ConfidenceDrop { entity_type: Option<EntityType> },
    /// Any new unresolved contention is created.
    NewContention,
    /// DST evidence combination conflict K exceeds threshold.
    HighConflict,
    /// New data matches a stored pattern.
    PatternMatch { pattern_id: String },
}

/// Severity levels for alert events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// An alert event generated when a rule fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub id: String,
    pub rule_id: String,
    pub narrative_id: Option<String>,
    pub event_type: String,
    pub description: String,
    pub target_ids: Vec<Uuid>,
    pub severity: AlertSeverity,
    pub acknowledged: bool,
    pub created_at: DateTime<Utc>,
}

// ─── KV Persistence ────────────────────────────────────────

fn rule_key(rule_id: &str) -> Vec<u8> {
    let mut key = keys::ALERT_RULE.to_vec();
    key.extend_from_slice(rule_id.as_bytes());
    key
}

fn event_key(event_id: &str) -> Vec<u8> {
    let mut key = keys::ALERT_EVENT.to_vec();
    key.extend_from_slice(event_id.as_bytes());
    key
}

/// Store an alert rule.
pub fn store_rule(store: &dyn KVStore, rule: &AlertRule) -> Result<()> {
    let key = rule_key(&rule.id);
    let value = serde_json::to_vec(rule)?;
    store.put(&key, &value)?;
    Ok(())
}

/// Load an alert rule by ID.
pub fn load_rule(store: &dyn KVStore, rule_id: &str) -> Result<Option<AlertRule>> {
    let key = rule_key(rule_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List all alert rules.
pub fn list_rules(store: &dyn KVStore) -> Result<Vec<AlertRule>> {
    let entries = store.prefix_scan(keys::ALERT_RULE)?;
    let mut rules = Vec::new();
    for (_, value) in entries {
        if let Ok(rule) = serde_json::from_slice::<AlertRule>(&value) {
            rules.push(rule);
        }
    }
    Ok(rules)
}

/// Delete an alert rule.
pub fn delete_rule(store: &dyn KVStore, rule_id: &str) -> Result<()> {
    let key = rule_key(rule_id);
    store.delete(&key)?;
    Ok(())
}

/// Store an alert event.
pub fn store_event(store: &dyn KVStore, event: &AlertEvent) -> Result<()> {
    let key = event_key(&event.id);
    let value = serde_json::to_vec(event)?;
    store.put(&key, &value)?;
    Ok(())
}

/// List alert events (most recent first, paginated).
pub fn list_events(store: &dyn KVStore, limit: usize) -> Result<Vec<AlertEvent>> {
    let entries = store.prefix_scan(keys::ALERT_EVENT)?;
    let mut events: Vec<AlertEvent> = entries
        .iter()
        .filter_map(|(_, v)| serde_json::from_slice(v).ok())
        .collect();
    events.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    events.truncate(limit);
    Ok(events)
}

/// Count unacknowledged events.
pub fn count_unread(store: &dyn KVStore) -> Result<usize> {
    let entries = store.prefix_scan(keys::ALERT_EVENT)?;
    let count = entries
        .iter()
        .filter_map(|(_, v)| serde_json::from_slice::<AlertEvent>(v).ok())
        .filter(|e| !e.acknowledged)
        .count();
    Ok(count)
}

/// Acknowledge an alert event.
pub fn acknowledge_event(store: &dyn KVStore, event_id: &str) -> Result<()> {
    let key = event_key(event_id);
    match store.get(&key)? {
        Some(bytes) => {
            let mut event: AlertEvent = serde_json::from_slice(&bytes)?;
            event.acknowledged = true;
            store.put(&key, &serde_json::to_vec(&event)?)?;
            Ok(())
        }
        None => Ok(()),
    }
}

// ─── Alert Checking (Reactive Hooks) ───────────────────────

/// Check all alert rules against a trust score change.
/// Called from `propagate_source_trust_change`.
pub fn check_trust_drop(
    store: &dyn KVStore,
    source_id: Uuid,
    new_trust: f32,
    narrative_id: Option<&str>,
) -> Result<Vec<AlertEvent>> {
    let rules = list_rules(store)?;
    let mut events = Vec::new();

    for rule in &rules {
        if !rule.enabled {
            continue;
        }
        if let AlertRuleType::TrustDrop {
            source_id: filter_src,
        } = &rule.rule_type
        {
            // Check if rule applies to this source (or all sources)
            if let Some(filter) = filter_src {
                if *filter != source_id {
                    continue;
                }
            }

            if (new_trust as f64) < rule.threshold {
                let severity = if new_trust < 0.2 {
                    AlertSeverity::Critical
                } else if new_trust < 0.5 {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Info
                };

                let event = AlertEvent {
                    id: Uuid::now_v7().to_string(),
                    rule_id: rule.id.clone(),
                    narrative_id: narrative_id.map(String::from),
                    event_type: "trust_drop".into(),
                    description: format!(
                        "Source trust dropped to {:.2} (threshold: {:.2})",
                        new_trust, rule.threshold
                    ),
                    target_ids: vec![source_id],
                    severity,
                    acknowledged: false,
                    created_at: Utc::now(),
                };
                store_event(store, &event)?;
                events.push(event);
            }
        }
    }

    Ok(events)
}

/// Check all alert rules against a confidence change.
pub fn check_confidence_drop(
    store: &dyn KVStore,
    target_id: Uuid,
    new_confidence: f32,
    entity_type: Option<&EntityType>,
    narrative_id: Option<&str>,
) -> Result<Vec<AlertEvent>> {
    let rules = list_rules(store)?;
    let mut events = Vec::new();

    for rule in &rules {
        if !rule.enabled {
            continue;
        }
        if let AlertRuleType::ConfidenceDrop {
            entity_type: filter_type,
        } = &rule.rule_type
        {
            // Check entity type filter
            if let (Some(filter), Some(actual)) = (filter_type, entity_type) {
                if filter != actual {
                    continue;
                }
            }

            if (new_confidence as f64) < rule.threshold {
                let severity = if new_confidence < 0.3 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                };

                let event = AlertEvent {
                    id: Uuid::now_v7().to_string(),
                    rule_id: rule.id.clone(),
                    narrative_id: narrative_id.map(String::from),
                    event_type: "confidence_drop".into(),
                    description: format!(
                        "Confidence dropped to {:.2} (threshold: {:.2})",
                        new_confidence, rule.threshold
                    ),
                    target_ids: vec![target_id],
                    severity,
                    acknowledged: false,
                    created_at: Utc::now(),
                };
                store_event(store, &event)?;
                events.push(event);
            }
        }
    }

    Ok(events)
}

/// Check for NewContention alert rules.
pub fn check_new_contention(
    store: &dyn KVStore,
    situation_a: Uuid,
    situation_b: Uuid,
    narrative_id: Option<&str>,
) -> Result<Vec<AlertEvent>> {
    let rules = list_rules(store)?;
    let mut events = Vec::new();

    for rule in &rules {
        if !rule.enabled {
            continue;
        }
        if rule.rule_type == AlertRuleType::NewContention {
            let event = AlertEvent {
                id: Uuid::now_v7().to_string(),
                rule_id: rule.id.clone(),
                narrative_id: narrative_id.map(String::from),
                event_type: "new_contention".into(),
                description: "New unresolved contention detected between situations".into(),
                target_ids: vec![situation_a, situation_b],
                severity: AlertSeverity::Warning,
                acknowledged: false,
                created_at: Utc::now(),
            };
            store_event(store, &event)?;
            events.push(event);
        }
    }

    Ok(events)
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    fn make_rule(rule_type: AlertRuleType, threshold: f64) -> AlertRule {
        AlertRule {
            id: Uuid::now_v7().to_string(),
            narrative_id: None,
            rule_type,
            threshold,
            enabled: true,
            created_at: Utc::now(),
            aggregator: None,
            quantifier_condition: None,
            mamdani_rule_id: None,
        }
    }

    #[test]
    fn test_alert_rule_mamdani_rule_id_roundtrip() {
        // Phase 9 added the `mamdani_rule_id` slot as #[serde(default)]
        // so legacy payloads (no field) still deserialize cleanly.
        let mut rule = make_rule(
            AlertRuleType::ConfidenceDrop { entity_type: None },
            0.5,
        );
        rule.mamdani_rule_id = Some(
            "01930000-0000-7000-8000-000000000001".into(),
        );
        let json = serde_json::to_string(&rule).unwrap();
        let parsed: AlertRule = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.mamdani_rule_id.as_deref(),
            Some("01930000-0000-7000-8000-000000000001")
        );
        // Legacy payload (no field) → deserialize clean, default None.
        let legacy = r#"{
            "id": "r9",
            "narrative_id": null,
            "rule_type": {"TrustDrop": {"source_id": null}},
            "threshold": 0.5,
            "enabled": true,
            "created_at": "2026-01-01T00:00:00Z"
        }"#;
        let parsed: AlertRule = serde_json::from_str(legacy).expect("legacy");
        assert!(parsed.mamdani_rule_id.is_none());
    }

    #[test]
    fn test_alert_rule_aggregator_defaults_to_none() {
        // Phase 2 added the aggregator slot as #[serde(default)] so
        // persisted rules from earlier versions deserialize cleanly and
        // default to the pre-sprint single-threshold semantics.
        let rule = make_rule(AlertRuleType::TrustDrop { source_id: None }, 0.5);
        assert!(rule.aggregator.is_none());

        // Round-trip via serde — legacy payload without the field should
        // still deserialize.
        let legacy = r#"{
            "id": "r1",
            "narrative_id": null,
            "rule_type": {"TrustDrop": {"source_id": null}},
            "threshold": 0.5,
            "enabled": true,
            "created_at": "2026-01-01T00:00:00Z"
        }"#;
        let parsed: AlertRule = serde_json::from_str(legacy).expect("legacy deserialize");
        assert!(parsed.aggregator.is_none());
    }

    #[test]
    fn test_create_alert_rule() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::TrustDrop { source_id: None }, 0.5);
        store_rule(store.as_ref(), &rule).unwrap();

        let loaded = load_rule(store.as_ref(), &rule.id).unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().threshold, 0.5);
    }

    #[test]
    fn test_trust_drop_triggers_alert() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::TrustDrop { source_id: None }, 0.5);
        store_rule(store.as_ref(), &rule).unwrap();

        let source_id = Uuid::now_v7();
        let events = check_trust_drop(store.as_ref(), source_id, 0.3, None).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "trust_drop");
        assert_eq!(events[0].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_confidence_drop_triggers_alert() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::ConfidenceDrop { entity_type: None }, 0.4);
        store_rule(store.as_ref(), &rule).unwrap();

        let target_id = Uuid::now_v7();
        let events = check_confidence_drop(store.as_ref(), target_id, 0.2, None, None).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "confidence_drop");
        assert_eq!(events[0].severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_new_contention_triggers_alert() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::NewContention, 0.0);
        store_rule(store.as_ref(), &rule).unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let events = check_new_contention(store.as_ref(), a, b, None).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "new_contention");
        assert_eq!(events[0].target_ids.len(), 2);
    }

    #[test]
    fn test_alert_severity_levels() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::TrustDrop { source_id: None }, 0.8);
        store_rule(store.as_ref(), &rule).unwrap();

        // Critical: trust < 0.2
        let events = check_trust_drop(store.as_ref(), Uuid::now_v7(), 0.1, None).unwrap();
        assert_eq!(events[0].severity, AlertSeverity::Critical);

        // Warning: trust 0.2-0.5
        let events = check_trust_drop(store.as_ref(), Uuid::now_v7(), 0.35, None).unwrap();
        assert_eq!(events[0].severity, AlertSeverity::Warning);

        // Info: trust 0.5-threshold
        let events = check_trust_drop(store.as_ref(), Uuid::now_v7(), 0.7, None).unwrap();
        assert_eq!(events[0].severity, AlertSeverity::Info);
    }

    #[test]
    fn test_alert_acknowledgement() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::NewContention, 0.0);
        store_rule(store.as_ref(), &rule).unwrap();

        let events =
            check_new_contention(store.as_ref(), Uuid::now_v7(), Uuid::now_v7(), None).unwrap();
        assert!(!events[0].acknowledged);

        acknowledge_event(store.as_ref(), &events[0].id).unwrap();

        let all = list_events(store.as_ref(), 10).unwrap();
        assert!(all[0].acknowledged);
    }

    #[test]
    fn test_disabled_rule_no_alert() {
        let store = test_store();
        let mut rule = make_rule(AlertRuleType::TrustDrop { source_id: None }, 0.5);
        rule.enabled = false;
        store_rule(store.as_ref(), &rule).unwrap();

        let events = check_trust_drop(store.as_ref(), Uuid::now_v7(), 0.1, None).unwrap();
        assert!(events.is_empty(), "Disabled rule should not trigger");
    }

    #[test]
    fn test_alert_event_persistence() {
        let store = test_store();
        let rule = make_rule(AlertRuleType::TrustDrop { source_id: None }, 0.9);
        store_rule(store.as_ref(), &rule).unwrap();

        // Generate 3 events
        for _ in 0..3 {
            check_trust_drop(store.as_ref(), Uuid::now_v7(), 0.5, None).unwrap();
        }

        let all = list_events(store.as_ref(), 100).unwrap();
        assert_eq!(all.len(), 3);

        let unread = count_unread(store.as_ref()).unwrap();
        assert_eq!(unread, 3);

        // Acknowledge one
        acknowledge_event(store.as_ref(), &all[0].id).unwrap();
        let unread = count_unread(store.as_ref()).unwrap();
        assert_eq!(unread, 2);
    }
}
