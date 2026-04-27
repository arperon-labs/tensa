//! Adversarial STIX 2.1 export — extends narrative STIX bundles with
//! DISARM attack-pattern objects and wargame session data.
//!
//! Maps DISARM techniques to STIX attack-patterns with external references
//! to the DISARM framework. Maps CIB clusters to STIX intrusion-sets.
//! Compatible with EU FIMI-ISAC, OpenCTI, and MISP.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::error::Result;
use crate::export::stix::deterministic_uuid;

use super::disarm::{DisarmAnnotation, DisarmCountermeasure, DisarmTechnique};
use super::session::WargameSession;

/// Extended STIX bundle with DISARM attack-patterns.
#[derive(Debug, Serialize, Deserialize)]
pub struct AdversarialStixBundle {
    #[serde(rename = "type")]
    pub bundle_type: String,
    pub id: String,
    pub objects: Vec<Value>,
}

/// Export a wargame session as a STIX 2.1 bundle with DISARM annotations.
pub fn export_wargame_stix(session: &WargameSession) -> Result<String> {
    let mut objects: Vec<Value> = Vec::new();
    let now = Utc::now().to_rfc3339();

    // Campaign SDO for the wargame narrative
    let campaign_id = format!(
        "campaign--{}",
        deterministic_uuid(format!("wargame:{}", session.narrative_id).as_bytes())
    );
    objects.push(json!({
        "type": "campaign",
        "spec_version": "2.1",
        "id": &campaign_id,
        "created": &now,
        "modified": &now,
        "name": format!("Wargame: {}", session.narrative_id),
        "description": format!(
            "Adversarial wargame simulation. {} turns completed. Status: {:?}",
            session.state.turn_number, session.status
        ),
    }));

    // Attack-patterns from DISARM techniques used in the session
    let mut seen_techniques = std::collections::HashSet::new();
    let mut seen_countermeasures = std::collections::HashSet::new();

    for log_entry in &session.state.move_log {
        // We don't have the full WargameAction in MoveLogEntry, but we can
        // reconstruct DISARM annotations from the ActionType
        let annotation = action_type_to_disarm_annotation(&log_entry.action_type);

        if let Some(ref tech) = annotation.technique {
            let tech_id_str = tech.id().to_string();
            if seen_techniques.insert(tech_id_str.clone()) {
                objects.push(technique_to_attack_pattern(tech, &now));

                // Relationship: campaign uses attack-pattern
                let ap_id = format!(
                    "attack-pattern--{}",
                    deterministic_uuid(format!("disarm:{}", tech.id()).as_bytes())
                );
                objects.push(json!({
                    "type": "relationship",
                    "spec_version": "2.1",
                    "id": format!("relationship--{}", Uuid::now_v7()),
                    "relationship_type": "uses",
                    "source_ref": &campaign_id,
                    "target_ref": &ap_id,
                    "created": &now,
                    "modified": &now,
                }));
            }
        }

        if let Some(ref cm) = annotation.countermeasure {
            let cm_id_str = cm.id().to_string();
            if seen_countermeasures.insert(cm_id_str) {
                objects.push(countermeasure_to_course_of_action(cm, &now));
            }
        }
    }

    // Threat actors from red team actors
    for actor in session.state.actors.values() {
        if actor.team == super::types::Team::Red {
            let actor_id = format!(
                "threat-actor--{}",
                deterministic_uuid(actor.entity_id.as_bytes())
            );
            objects.push(json!({
                "type": "threat-actor",
                "spec_version": "2.1",
                "id": &actor_id,
                "created": &now,
                "modified": &now,
                "name": &actor.name,
                "threat_actor_types": ["unknown"],
                "roles": ["agent"],
            }));

            // Relationship: threat-actor attributed-to campaign
            objects.push(json!({
                "type": "relationship",
                "spec_version": "2.1",
                "id": format!("relationship--{}", Uuid::now_v7()),
                "relationship_type": "attributed-to",
                "source_ref": &actor_id,
                "target_ref": &campaign_id,
                "created": &now,
                "modified": &now,
            }));
        }
    }

    let bundle = AdversarialStixBundle {
        bundle_type: "bundle".into(),
        id: format!("bundle--{}", Uuid::now_v7()),
        objects,
    };

    Ok(serde_json::to_string_pretty(&bundle)?)
}

/// Convert a DISARM technique to a STIX attack-pattern SDO.
fn technique_to_attack_pattern(tech: &DisarmTechnique, created: &str) -> Value {
    let tactic = tech.tactic();
    json!({
        "type": "attack-pattern",
        "spec_version": "2.1",
        "id": format!(
            "attack-pattern--{}",
            deterministic_uuid(format!("disarm:{}", tech.id()).as_bytes())
        ),
        "created": created,
        "modified": created,
        "name": tech.id(),
        "description": format!("DISARM technique {} under tactic {} ({})", tech.id(), tactic.id(), tactic.label()),
        "kill_chain_phases": [{
            "kill_chain_name": "disarm",
            "phase_name": tactic.id(),
        }],
        "external_references": [{
            "source_name": "DISARM",
            "external_id": tech.id(),
            "url": format!("https://github.com/DISARMFoundation/DISARMframeworks/blob/main/generated_pages/techniques/{}.md", tech.id()),
        }],
    })
}

/// Convert a DISARM countermeasure to a STIX course-of-action SDO.
fn countermeasure_to_course_of_action(cm: &DisarmCountermeasure, created: &str) -> Value {
    json!({
        "type": "course-of-action",
        "spec_version": "2.1",
        "id": format!(
            "course-of-action--{}",
            deterministic_uuid(format!("disarm-cm:{}", cm.id()).as_bytes())
        ),
        "created": created,
        "modified": created,
        "name": cm.id(),
        "description": format!("DISARM countermeasure {}", cm.id()),
        "external_references": [{
            "source_name": "DISARM",
            "external_id": cm.id(),
        }],
    })
}

/// Map ActionType (from move log) back to a DISARM annotation.
/// Delegates beta values to the canonical DisarmTechnique/DisarmCountermeasure methods.
fn action_type_to_disarm_annotation(action_type: &super::types::ActionType) -> DisarmAnnotation {
    use super::types::ActionType;

    // Helper: build annotation from a technique
    fn from_tech(tech: DisarmTechnique) -> DisarmAnnotation {
        DisarmAnnotation {
            tactic: Some(tech.tactic()),
            beta_effect: tech.beta_multiplier(),
            technique: Some(tech),
            countermeasure: None,
        }
    }
    // Helper: build annotation from a countermeasure
    fn from_cm(cm: DisarmCountermeasure) -> DisarmAnnotation {
        DisarmAnnotation {
            tactic: None,
            beta_effect: cm.beta_reduction(),
            technique: None,
            countermeasure: Some(cm),
        }
    }

    match action_type {
        ActionType::Post => from_tech(DisarmTechnique::PostContent),
        ActionType::Amplify => from_tech(DisarmTechnique::AmplifyNarratives),
        ActionType::Coordinate => from_tech(DisarmTechnique::CoordinatedInauthenticBehavior),
        ActionType::CreateAccount => from_tech(DisarmTechnique::BulkAccountCreation),
        ActionType::Reply => from_tech(DisarmTechnique::CommentOrReply),
        ActionType::Debunk => from_cm(DisarmCountermeasure::Debunk),
        ActionType::Prebunk => from_cm(DisarmCountermeasure::Prebunk),
        ActionType::TakeDown => from_cm(DisarmCountermeasure::PlatformAccountRemoval),
        ActionType::Observe => DisarmAnnotation {
            technique: None,
            countermeasure: None,
            tactic: None,
            beta_effect: 1.0,
        },
    }
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adversarial::session::*;
    use crate::adversarial::sim_state::*;
    use crate::adversarial::types::*;
    use crate::adversarial::wargame::*;
    use crate::store::memory::MemoryStore;
    use crate::types::EntityType;
    use std::sync::Arc;

    fn make_test_session() -> WargameSession {
        let store = Arc::new(MemoryStore::new());
        let hg = crate::Hypergraph::new(store);

        let red_id = hg
            .create_entity(crate::types::Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: json!({"name": "RedBot"}),
                beliefs: None,
                embedding: None,
                maturity: crate::types::MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("stix-test".to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();

        let config = WargameConfig {
            max_turns: 5,
            auto_red: true,
            auto_blue: false,
            ..Default::default()
        };
        let mut session = WargameSession::create(&hg, "stix-test", config).unwrap();
        session.state.assign_team(&red_id, Team::Red).unwrap();
        session.state.seed_narrative("stix-test", "twitter", 10.0);

        // Play a couple turns to generate move log
        session.auto_play(2).unwrap();
        session
    }

    #[test]
    fn test_export_wargame_stix_produces_valid_bundle() {
        let session = make_test_session();
        let stix_json = export_wargame_stix(&session).unwrap();

        let bundle: AdversarialStixBundle = serde_json::from_str(&stix_json).unwrap();
        assert_eq!(bundle.bundle_type, "bundle");
        assert!(!bundle.objects.is_empty());

        // Should have at least: campaign + attack-patterns + threat-actor + relationships
        let types: Vec<&str> = bundle
            .objects
            .iter()
            .filter_map(|o| o.get("type").and_then(|t| t.as_str()))
            .collect();
        assert!(types.contains(&"campaign"));
        assert!(types.contains(&"threat-actor"));
    }

    #[test]
    fn test_disarm_attack_patterns_have_external_refs() {
        let session = make_test_session();
        let stix_json = export_wargame_stix(&session).unwrap();
        let bundle: AdversarialStixBundle = serde_json::from_str(&stix_json).unwrap();

        let attack_patterns: Vec<&Value> = bundle
            .objects
            .iter()
            .filter(|o| o.get("type").and_then(|t| t.as_str()) == Some("attack-pattern"))
            .collect();

        for ap in &attack_patterns {
            let refs = ap.get("external_references").and_then(|r| r.as_array());
            assert!(
                refs.is_some(),
                "attack-pattern should have external_references"
            );
            let first = &refs.unwrap()[0];
            assert_eq!(first["source_name"], "DISARM");
        }
    }

    #[test]
    fn test_deterministic_uuid_consistency() {
        let a = deterministic_uuid(b"test-data");
        let b = deterministic_uuid(b"test-data");
        assert_eq!(a, b, "same input should produce same UUID");

        let c = deterministic_uuid(b"different-data");
        assert_ne!(a, c, "different input should produce different UUID");
    }

    #[test]
    fn test_countermeasure_course_of_action() {
        let now = Utc::now().to_rfc3339();
        let coa = countermeasure_to_course_of_action(&DisarmCountermeasure::Debunk, &now);
        assert_eq!(coa["type"], "course-of-action");
        assert!(coa["id"]
            .as_str()
            .unwrap()
            .starts_with("course-of-action--"));
        assert_eq!(coa["name"], "C00011");
    }
}
