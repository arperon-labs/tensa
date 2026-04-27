//! Hypergraph materialization (Sprint D9.6, Stage 2).
//!
//! Writes a `NarrativePlan` into the TENSA hypergraph as real entities,
//! situations, Allen relations, causal edges, commitments, knowledge states,
//! and subplot tags. After materialization, the narrative is fully queryable.

use std::collections::HashMap;

use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::commitments::{self, CommitmentStatus, NarrativeCommitment};
use crate::narrative::registry::NarrativeRegistry;
use crate::types::*;

use super::types::*;

/// Result of materialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaterializationReport {
    pub narrative_id: String,
    pub entities_created: usize,
    pub situations_created: usize,
    pub participations_created: usize,
    pub causal_links_created: usize,
    pub commitments_created: usize,
    pub facts_stored: usize,
}

/// Materialize a narrative plan into the TENSA hypergraph.
///
/// Creates real Entity, Situation, Participation, CausalLink, Commitment,
/// and Fact records. Returns the narrative_id used.
pub fn materialize_plan(
    hg: &Hypergraph,
    plan: &super::types::NarrativePlan,
) -> Result<MaterializationReport> {
    let nid = &plan.narrative_id;

    // Create narrative metadata via registry
    let registry = NarrativeRegistry::new(hg.store_arc());
    let narrative = crate::narrative::types::Narrative {
        id: nid.clone(),
        title: format!("Generated: {}", plan.config.premise),
        genre: Some(plan.config.genre.clone()),
        tags: vec!["generated".into()],
        source: None,
        project_id: None,
        description: Some(plan.config.premise.clone()),
        authors: vec!["tensa-generator".into()],
        language: Some("en".into()),
        publication_date: None,
        cover_url: None,
        custom_properties: HashMap::new(),
        entity_count: plan.entities.len(),
        situation_count: plan.fabula.len(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    registry.create(narrative)?;

    let mut entity_id_map: HashMap<Uuid, Uuid> = HashMap::new();
    let mut entities_created = 0;
    let mut situations_created = 0;
    let mut participations_created = 0;
    let mut causal_links_created = 0;

    // ── Stage 1: Materialize entities ───────────────────────

    for pe in &plan.entities {
        let mut props = serde_json::json!({
            "name": pe.name,
            "role": pe.role,
            "want": pe.want,
            "need": pe.need,
        });
        if let Some(lie) = &pe.lie {
            props["lie"] = serde_json::json!(lie);
        }
        if let Some(truth) = &pe.truth {
            props["truth"] = serde_json::json!(truth);
        }
        props["arc_type"] = serde_json::to_value(&pe.arc_type)?;
        props["initial_knowledge"] = serde_json::to_value(&pe.initial_knowledge)?;
        props["relationships"] = serde_json::to_value(&pe.relationships)?;

        let entity_type = match pe.entity_type.as_str() {
            "Location" => EntityType::Location,
            "Artifact" => EntityType::Artifact,
            "Concept" => EntityType::Concept,
            "Organization" => EntityType::Organization,
            _ => EntityType::Actor,
        };

        let entity = Entity {
            id: pe.id,
            entity_type,
            properties: props,
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Validated,
            confidence: 1.0, // Author-asserted
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::StructuredImport),
            narrative_id: Some(nid.clone()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };

        let real_id = hg.create_entity(entity)?;
        entity_id_map.insert(pe.id, real_id);
        entities_created += 1;

        // Store initial motivation vector
        if !pe.initial_motivation.is_empty() {
            let mot_key = format!("irl/{}/t0", real_id).into_bytes();
            let mot_val = serde_json::to_vec(&pe.initial_motivation)?;
            hg.store().put(&mot_key, &mot_val)?;
        }
    }

    // ── Stage 2: Materialize situations ─────────────────────

    let base_time = Utc::now();
    let mut situation_id_map: HashMap<Uuid, Uuid> = HashMap::new();
    let mut prev_sit_id: Option<Uuid> = None;

    for ps in &plan.fabula {
        let start = base_time + Duration::hours(ps.chapter as i64 * 2);
        let end = start + Duration::hours(1);

        // Causal links are written to the c/ / cr/ KV prefixes below via
        // add_causal_link so Workshop's detectors and the graph-projection
        // layer can see them. The inline Situation.causes vec is left empty —
        // keeping both in sync would require a re-read after every mutation
        // and is not used by any reader.

        // Add game structure if specified
        let game_structure = ps.game_type.as_ref().map(|gt| GameStructure {
            game_type: match gt {
                GameType::Cooperative => GameClassification::Coordination,
                GameType::Adversarial => GameClassification::ZeroSum,
                GameType::InfoAsymmetry => GameClassification::AsymmetricInformation,
                GameType::Mixed => GameClassification::Custom("Mixed".into()),
            },
            info_structure: InfoStructureType::Complete,
            description: None,
            maturity: MaturityLevel::Validated,
        });

        let situation = Situation {
            id: ps.id,
            properties: serde_json::Value::Null,
            name: Some(format!(
                "Chapter {} — {}",
                ps.chapter + 1,
                &ps.summary[..ps.summary.len().min(50)]
            )),
            description: Some(ps.summary.clone()),
            temporal: crate::types::AllenInterval {
                start: Some(start),
                end: Some(end),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure,
            causes: Vec::new(),
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(&ps.summary)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Validated,
            confidence: 1.0,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::StructuredImport,
            provenance: vec![],
            narrative_id: Some(nid.clone()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };

        let real_sid = hg.create_situation(situation)?;
        situation_id_map.insert(ps.id, real_sid);
        situations_created += 1;

        // Write the sequential predecessor edge now that both endpoints exist.
        if let Some(prev_id) = prev_sit_id {
            if crate::narrative::causal_helpers::add_sequential_link(
                hg,
                prev_id,
                real_sid,
                crate::narrative::causal_helpers::MECHANISM_SEQUENTIAL,
                0.8,
                MaturityLevel::Validated,
            )? {
                causal_links_created += 1;
            }
        }

        // Add participations
        for (seq, pid) in ps.participants.iter().enumerate() {
            let real_eid = entity_id_map.get(pid).copied().unwrap_or(*pid);

            // Determine role from entity design
            let role = plan
                .entities
                .iter()
                .find(|e| e.id == *pid)
                .map(|e| e.role.clone())
                .unwrap_or(Role::Bystander);

            // Build info_set from knowledge transitions
            let info_set = ps
                .knowledge_transitions
                .iter()
                .find(|kt| kt.entity_id == *pid)
                .map(|kt| InfoSet {
                    knows_before: Vec::new(),
                    learns: kt
                        .learns
                        .iter()
                        .map(|f| KnowledgeFact {
                            about_entity: *pid,
                            fact: f.clone(),
                            confidence: 1.0,
                        })
                        .collect(),
                    reveals: Vec::new(),
                    beliefs_about_others: Vec::new(),
                });

            hg.add_participant(Participation {
                entity_id: real_eid,
                situation_id: real_sid,
                role,
                info_set,
                action: None,
                payoff: None,
                seq: seq as u16,
            })?;
            participations_created += 1;
        }

        // Store scene type and narration mode as metadata
        let meta_key = format!("gm/{}/{}", nid, real_sid).into_bytes();
        let meta = serde_json::json!({
            "scene_type": ps.scene_type,
            "narration_mode": ps.narration_mode,
            "emotional_valence": ps.emotional_valence,
            "chapter": ps.chapter,
        });
        hg.store().put(&meta_key, &serde_json::to_vec(&meta)?)?;

        prev_sit_id = Some(real_sid);
    }

    // ── Stage 3: Materialize commitments ────────────────────

    let mut commitments_created = 0;
    for pc in &plan.commitments {
        let setup_sit = plan
            .fabula
            .iter()
            .find(|s| s.chapter == pc.setup_chapter)
            .map(|s| situation_id_map.get(&s.id).copied().unwrap_or(s.id))
            .unwrap_or(Uuid::nil());

        let payoff_sit = plan
            .fabula
            .iter()
            .find(|s| s.chapter == pc.payoff_chapter)
            .map(|s| situation_id_map.get(&s.id).copied().unwrap_or(s.id));

        let commitment = NarrativeCommitment {
            id: Uuid::now_v7(),
            narrative_id: nid.clone(),
            commitment_type: pc.commitment_type.clone(),
            setup_event: setup_sit,
            setup_chapter: pc.setup_chapter,
            setup_salience: 2.0,
            status: CommitmentStatus::Planted,
            payoff_event: payoff_sit,
            payoff_chapter: Some(pc.payoff_chapter),
            payoff_distance: Some(pc.payoff_chapter - pc.setup_chapter),
            causal_chain: Vec::new(),
            description: format!("Planned: {}", pc.element),
            tracked_element: pc.element.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        commitments::store_commitment(hg, &commitment)?;
        commitments_created += 1;
    }

    // ── Stage 4: Materialize facts ──────────────────────────

    let mut facts_stored = 0;
    for pf in &plan.facts {
        let key = fact_key(nid, &pf.id);
        let val = serde_json::to_vec(pf)?;
        hg.store().put(&key, &val)?;
        facts_stored += 1;
    }

    // ── Stage 5: Store motivation waypoints ─────────────────

    for arc in &plan.character_arcs {
        let real_eid = entity_id_map
            .get(&arc.character_id)
            .copied()
            .unwrap_or(arc.character_id);
        for (chapter, motivation) in &arc.motivation_waypoints {
            let key = format!("irl/{}/t{}", real_eid, chapter).into_bytes();
            let val = serde_json::to_vec(motivation)?;
            hg.store().put(&key, &val)?;
        }
    }

    Ok(MaterializationReport {
        narrative_id: nid.clone(),
        entities_created,
        situations_created,
        participations_created,
        causal_links_created,
        commitments_created,
        facts_stored,
    })
}

/// Validate a materialized narrative for consistency.
pub fn validate_materialized(hg: &Hypergraph, narrative_id: &str) -> Result<Vec<ConsistencyIssue>> {
    let mut issues = Vec::new();
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let sit_map: std::collections::HashMap<Uuid, &Situation> =
        situations.iter().map(|s| (s.id, s)).collect();

    // Temporal consistency: no effect before cause
    for sit in &situations {
        for cause in &sit.causes {
            if let Some(cause_sit) = sit_map.get(&cause.from_situation) {
                if let (Some(cause_end), Some(effect_start)) =
                    (cause_sit.temporal.end, sit.temporal.start)
                {
                    if cause_end > effect_start {
                        issues.push(ConsistencyIssue {
                            severity: IssueSeverity::Error,
                            category: IssueCategory::TemporalInconsistency,
                            description: format!(
                                "Cause ends after effect starts: {} → {}",
                                cause.from_situation, sit.id
                            ),
                            target_id: Some(sit.id),
                            chapter: None,
                        });
                    }
                }
            }
        }
    }

    // Commitment reachability: every commitment has a setup situation that exists
    let commitments = crate::narrative::commitments::list_commitments(hg, narrative_id)?;
    let sit_ids: std::collections::HashSet<Uuid> = situations.iter().map(|s| s.id).collect();
    for c in &commitments {
        if c.setup_event != Uuid::nil() && !sit_ids.contains(&c.setup_event) {
            issues.push(ConsistencyIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::CommitmentOrphaned,
                description: format!(
                    "Commitment '{}' setup event {} not found in narrative",
                    c.tracked_element, c.setup_event
                ),
                target_id: Some(c.setup_event),
                chapter: Some(c.setup_chapter),
            });
        }
    }

    Ok(issues)
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::planner;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_materialize_basic() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "A detective investigates a murder".into(),
            chapter_count: 5,
            protagonist_count: 1,
            subplot_count: 1,
            commitment_density: 0.4,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        let report = materialize_plan(&hg, &plan).unwrap();

        assert_eq!(report.narrative_id, plan.narrative_id);
        assert!(report.entities_created > 0);
        assert_eq!(report.situations_created, 5);
        assert!(report.participations_created > 0);

        // Verify entities are queryable
        let entities = hg.list_entities_by_narrative(&plan.narrative_id).unwrap();
        assert_eq!(entities.len(), report.entities_created);

        // Verify situations are queryable
        let situations = hg.list_situations_by_narrative(&plan.narrative_id).unwrap();
        assert_eq!(situations.len(), 5);
    }

    #[test]
    fn test_materialize_creates_commitments() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "A story with commitments".into(),
            chapter_count: 10,
            commitment_density: 0.5,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        let report = materialize_plan(&hg, &plan).unwrap();

        assert!(report.commitments_created > 0);

        // Commitments should be queryable
        let commitments = commitments::list_commitments(&hg, &plan.narrative_id).unwrap();
        assert_eq!(commitments.len(), report.commitments_created);
    }

    #[test]
    fn test_materialize_creates_causal_links() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "Sequential story".into(),
            chapter_count: 4,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        let report = materialize_plan(&hg, &plan).unwrap();

        // Should have n-1 causal links (sequential chain)
        assert_eq!(report.causal_links_created, 3);
    }

    #[test]
    fn test_validate_materialized() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "Validation test".into(),
            chapter_count: 5,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        materialize_plan(&hg, &plan).unwrap();

        let issues = validate_materialized(&hg, &plan.narrative_id).unwrap();
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Materialized plan should have no temporal errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_materialize_stores_facts() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "Facts test".into(),
            chapter_count: 3,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        let report = materialize_plan(&hg, &plan).unwrap();

        assert!(report.facts_stored > 0);

        // Verify facts are readable
        let key = fact_key(&plan.narrative_id, "premise");
        let val = hg.store().get(&key).unwrap();
        assert!(val.is_some(), "Premise fact should be stored");
    }
}
