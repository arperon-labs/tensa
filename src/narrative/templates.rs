//! Situation template library — reusable narrative building blocks.
//!
//! Extract common narrative patterns as reusable templates with typed
//! slots. Instantiate templates by binding entities to slots.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// A slot in a situation template (to be filled by a specific entity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSlot {
    pub slot_id: String,
    pub role: String,
    pub entity_type: String,
    pub description: String,
}

/// A situation within a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSituation {
    pub label: String,
    pub participant_slots: Vec<String>,
    pub narrative_level: String,
    pub description: String,
    /// Relative position within the template (0.0 = first, 1.0 = last).
    pub relative_position: f64,
    /// Causal predecessor labels within this template.
    pub after: Vec<String>,
}

/// A reusable narrative building block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationTemplate {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    /// Named slots that must be filled with entities.
    pub slots: Vec<TemplateSlot>,
    /// Situations in this template.
    pub situations: Vec<TemplateSituation>,
    /// Tags for discovery.
    pub tags: Vec<String>,
    /// Source narrative this was extracted from (if any).
    pub extracted_from: Option<String>,
}

/// Bindings from template slots to entity UUIDs.
pub type SlotBindings = HashMap<String, Uuid>;

/// Result of instantiating a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateInstantiation {
    pub template_id: Uuid,
    pub template_name: String,
    pub bindings: SlotBindings,
    /// The planned situations generated from the template.
    pub situations: Vec<InstantiatedSituation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstantiatedSituation {
    pub label: String,
    pub participants: Vec<Uuid>,
    pub description: String,
}

// ─── KV ─────────────────────────────────────────────────────

fn template_key(id: &Uuid) -> Vec<u8> {
    format!("ntpl/{}", id).into_bytes()
}

pub fn store_template(hg: &Hypergraph, tpl: &SituationTemplate) -> Result<()> {
    let key = template_key(&tpl.id);
    let val = serde_json::to_vec(tpl)?;
    hg.store().put(&key, &val)
}

pub fn load_template(hg: &Hypergraph, id: &Uuid) -> Result<Option<SituationTemplate>> {
    let key = template_key(id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn list_templates(hg: &Hypergraph) -> Result<Vec<SituationTemplate>> {
    let prefix = b"ntpl/";
    let items = hg.store().prefix_scan(prefix)?;
    let mut out = Vec::new();
    for (_k, v) in items {
        if let Ok(tpl) = serde_json::from_slice::<SituationTemplate>(&v) {
            out.push(tpl);
        }
    }
    Ok(out)
}

pub fn delete_template(hg: &Hypergraph, id: &Uuid) -> Result<()> {
    let key = template_key(id);
    hg.store().delete(&key)
}

// ─── Built-in Templates ─────────────────────────────────────

/// Create the built-in template library.
pub fn builtin_templates() -> Vec<SituationTemplate> {
    vec![
        SituationTemplate {
            id: Uuid::from_u128(0x0001_0001_0001_0001_0001_0001_0001_0001),
            name: "The Mentor's Death".into(),
            description: "Mentor transfers knowledge to protagonist then dies, triggering the dark night.".into(),
            slots: vec![
                TemplateSlot { slot_id: "mentor".into(), role: "Mentor".into(), entity_type: "Actor".into(), description: "The guiding figure".into() },
                TemplateSlot { slot_id: "protagonist".into(), role: "Protagonist".into(), entity_type: "Actor".into(), description: "The hero".into() },
            ],
            situations: vec![
                TemplateSituation { label: "final_lesson".into(), participant_slots: vec!["mentor".into(), "protagonist".into()], narrative_level: "Scene".into(), description: "Mentor shares crucial knowledge with protagonist".into(), relative_position: 0.0, after: vec![] },
                TemplateSituation { label: "mentors_death".into(), participant_slots: vec!["mentor".into(), "protagonist".into()], narrative_level: "Scene".into(), description: "Mentor dies or departs permanently".into(), relative_position: 0.5, after: vec!["final_lesson".into()] },
                TemplateSituation { label: "dark_night".into(), participant_slots: vec!["protagonist".into()], narrative_level: "Scene".into(), description: "Protagonist processes loss, reaches lowest point".into(), relative_position: 1.0, after: vec!["mentors_death".into()] },
            ],
            tags: vec!["arc".into(), "loss".into(), "transformation".into()],
            extracted_from: None,
        },
        SituationTemplate {
            id: Uuid::from_u128(0x0001_0001_0001_0001_0001_0001_0001_0002),
            name: "The False Victory".into(),
            description: "Protagonist achieves stated want, valence peaks, then reversal reveals the need.".into(),
            slots: vec![
                TemplateSlot { slot_id: "protagonist".into(), role: "Protagonist".into(), entity_type: "Actor".into(), description: "The hero".into() },
                TemplateSlot { slot_id: "antagonist".into(), role: "Antagonist".into(), entity_type: "Actor".into(), description: "The opposition".into() },
            ],
            situations: vec![
                TemplateSituation { label: "apparent_victory".into(), participant_slots: vec!["protagonist".into()], narrative_level: "Scene".into(), description: "Protagonist achieves their stated want".into(), relative_position: 0.0, after: vec![] },
                TemplateSituation { label: "celebration".into(), participant_slots: vec!["protagonist".into()], narrative_level: "Beat".into(), description: "Brief moment of triumph".into(), relative_position: 0.3, after: vec!["apparent_victory".into()] },
                TemplateSituation { label: "reversal".into(), participant_slots: vec!["protagonist".into(), "antagonist".into()], narrative_level: "Scene".into(), description: "Twist reveals the victory was hollow — the real problem is the unaddressed need".into(), relative_position: 1.0, after: vec!["celebration".into()] },
            ],
            tags: vec!["arc".into(), "reversal".into(), "midpoint".into()],
            extracted_from: None,
        },
        SituationTemplate {
            id: Uuid::from_u128(0x0001_0001_0001_0001_0001_0001_0001_0003),
            name: "The Information Marketplace".into(),
            description: "Multi-entity information asymmetry scene: each participant holds partial knowledge.".into(),
            slots: vec![
                TemplateSlot { slot_id: "broker".into(), role: "Broker".into(), entity_type: "Actor".into(), description: "Controls information flow".into() },
                TemplateSlot { slot_id: "seeker".into(), role: "Seeker".into(), entity_type: "Actor".into(), description: "Needs information".into() },
                TemplateSlot { slot_id: "holder".into(), role: "Holder".into(), entity_type: "Actor".into(), description: "Has the information".into() },
            ],
            situations: vec![
                TemplateSituation { label: "approach".into(), participant_slots: vec!["seeker".into(), "broker".into()], narrative_level: "Scene".into(), description: "Seeker approaches broker for information".into(), relative_position: 0.0, after: vec![] },
                TemplateSituation { label: "negotiation".into(), participant_slots: vec!["seeker".into(), "broker".into(), "holder".into()], narrative_level: "Scene".into(), description: "Three-way negotiation with asymmetric knowledge".into(), relative_position: 0.5, after: vec!["approach".into()] },
                TemplateSituation { label: "exchange".into(), participant_slots: vec!["seeker".into(), "holder".into()], narrative_level: "Scene".into(), description: "Information changes hands, power dynamics shift".into(), relative_position: 1.0, after: vec!["negotiation".into()] },
            ],
            tags: vec!["game_theory".into(), "information".into(), "negotiation".into()],
            extracted_from: None,
        },
    ]
}

// ─── Instantiation ──────────────────────────────────────────

/// Instantiate a template with entity bindings.
pub fn instantiate_template(
    template: &SituationTemplate,
    bindings: &SlotBindings,
) -> Result<TemplateInstantiation> {
    // Validate all required slots are bound
    for slot in &template.slots {
        if !bindings.contains_key(&slot.slot_id) {
            return Err(TensaError::InvalidQuery(format!(
                "template slot '{}' ({}) is not bound",
                slot.slot_id, slot.description
            )));
        }
    }

    let situations: Vec<InstantiatedSituation> = template
        .situations
        .iter()
        .map(|ts| {
            let participants: Vec<Uuid> = ts
                .participant_slots
                .iter()
                .filter_map(|slot_id| bindings.get(slot_id).copied())
                .collect();

            InstantiatedSituation {
                label: ts.label.clone(),
                participants,
                description: ts.description.clone(),
            }
        })
        .collect();

    Ok(TemplateInstantiation {
        template_id: template.id,
        template_name: template.name.clone(),
        bindings: bindings.clone(),
        situations,
    })
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_builtin_templates() {
        let templates = builtin_templates();
        assert_eq!(templates.len(), 3);
        assert_eq!(templates[0].name, "The Mentor's Death");
        assert_eq!(templates[1].name, "The False Victory");
        assert_eq!(templates[2].name, "The Information Marketplace");
    }

    #[test]
    fn test_instantiate_template() {
        let templates = builtin_templates();
        let mentor_death = &templates[0];

        let mut bindings = SlotBindings::new();
        bindings.insert("mentor".into(), Uuid::now_v7());
        bindings.insert("protagonist".into(), Uuid::now_v7());

        let result = instantiate_template(mentor_death, &bindings).unwrap();
        assert_eq!(result.situations.len(), 3);
        assert_eq!(result.situations[0].label, "final_lesson");
        assert_eq!(result.situations[0].participants.len(), 2);
        assert_eq!(result.situations[2].label, "dark_night");
        assert_eq!(result.situations[2].participants.len(), 1); // Only protagonist
    }

    #[test]
    fn test_instantiate_missing_slot() {
        let templates = builtin_templates();
        let mentor_death = &templates[0];

        let mut bindings = SlotBindings::new();
        bindings.insert("mentor".into(), Uuid::now_v7());
        // Missing "protagonist" slot

        let result = instantiate_template(mentor_death, &bindings);
        assert!(result.is_err());
    }

    #[test]
    fn test_template_kv_crud() {
        let hg = test_hg();
        let templates = builtin_templates();

        for tpl in &templates {
            store_template(&hg, tpl).unwrap();
        }

        let loaded = list_templates(&hg).unwrap();
        assert_eq!(loaded.len(), 3);

        delete_template(&hg, &templates[0].id).unwrap();
        let after_delete = list_templates(&hg).unwrap();
        assert_eq!(after_delete.len(), 2);
    }
}
