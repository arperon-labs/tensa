use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::store::TxnOp;
use crate::types::*;
use chrono::Utc;
use uuid::Uuid;

impl Hypergraph {
    /// Set maturity on an entity. Only forward transitions allowed.
    /// Atomically writes updated entity + validation log entry.
    pub fn set_entity_maturity(
        &self,
        entity_id: &Uuid,
        new_maturity: MaturityLevel,
        reviewer: &str,
        notes: Option<String>,
    ) -> Result<Entity> {
        let mut entity = self.get_entity(entity_id)?;
        if new_maturity <= entity.maturity {
            return Err(TensaError::InvalidMaturityTransition {
                from: entity.maturity,
                to: new_maturity,
            });
        }

        let old_maturity = entity.maturity;
        entity.maturity = new_maturity;
        entity.updated_at = Utc::now();

        let log_entry = ValidationLogEntry {
            target_id: *entity_id,
            reviewer: reviewer.to_string(),
            old_maturity,
            new_maturity,
            notes,
            timestamp: entity.updated_at,
        };

        let entity_bytes = serde_json::to_vec(&entity)?;
        let log_bytes = serde_json::to_vec(&log_entry)?;

        self.store.transaction(vec![
            TxnOp::Put(keys::entity_key(entity_id), entity_bytes),
            TxnOp::Put(
                keys::validation_log_key(&log_entry.timestamp, entity_id),
                log_bytes,
            ),
        ])?;

        Ok(entity)
    }

    /// Set maturity on a situation. Only forward transitions allowed.
    /// Atomically writes updated situation + validation log entry.
    pub fn set_situation_maturity(
        &self,
        situation_id: &Uuid,
        new_maturity: MaturityLevel,
        reviewer: &str,
        notes: Option<String>,
    ) -> Result<Situation> {
        let mut situation = self.get_situation(situation_id)?;
        if new_maturity <= situation.maturity {
            return Err(TensaError::InvalidMaturityTransition {
                from: situation.maturity,
                to: new_maturity,
            });
        }

        let old_maturity = situation.maturity;
        situation.maturity = new_maturity;
        situation.updated_at = Utc::now();

        let log_entry = ValidationLogEntry {
            target_id: *situation_id,
            reviewer: reviewer.to_string(),
            old_maturity,
            new_maturity,
            notes,
            timestamp: situation.updated_at,
        };

        let sit_bytes = serde_json::to_vec(&situation)?;
        let log_bytes = serde_json::to_vec(&log_entry)?;

        self.store.transaction(vec![
            TxnOp::Put(keys::situation_key(situation_id), sit_bytes),
            TxnOp::Put(
                keys::validation_log_key(&log_entry.timestamp, situation_id),
                log_bytes,
            ),
        ])?;

        Ok(situation)
    }

    /// Promote entity maturity by one level.
    pub fn promote_entity_maturity(
        &self,
        entity_id: &Uuid,
        reviewer: &str,
        notes: Option<String>,
    ) -> Result<Entity> {
        let entity = self.get_entity(entity_id)?;
        let next = next_maturity(entity.maturity)?;
        self.set_entity_maturity(entity_id, next, reviewer, notes)
    }

    /// Promote situation maturity by one level.
    pub fn promote_situation_maturity(
        &self,
        situation_id: &Uuid,
        reviewer: &str,
        notes: Option<String>,
    ) -> Result<Situation> {
        let situation = self.get_situation(situation_id)?;
        let next = next_maturity(situation.maturity)?;
        self.set_situation_maturity(situation_id, next, reviewer, notes)
    }

    /// Get validation log entries for a specific target (entity or situation).
    pub fn get_validation_log(&self, target_id: &Uuid) -> Result<Vec<ValidationLogEntry>> {
        let prefix = keys::validation_log_prefix();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let entry: ValidationLogEntry = serde_json::from_slice(&value)?;
            if entry.target_id == *target_id {
                result.push(entry);
            }
        }
        Ok(result)
    }
}

/// Get the next maturity level. Returns error if already at max.
fn next_maturity(current: MaturityLevel) -> Result<MaturityLevel> {
    match current {
        MaturityLevel::Candidate => Ok(MaturityLevel::Reviewed),
        MaturityLevel::Reviewed => Ok(MaturityLevel::Validated),
        MaturityLevel::Validated => Ok(MaturityLevel::GroundTruth),
        MaturityLevel::GroundTruth => Err(TensaError::InvalidMaturityTransition {
            from: MaturityLevel::GroundTruth,
            to: MaturityLevel::GroundTruth,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    fn setup_entity(hg: &Hypergraph) -> Uuid {
        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Test"}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap()
    }

    fn setup_situation(hg: &Hypergraph) -> Uuid {
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: Some(Utc::now()),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("Test")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: None,
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
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_set_maturity_candidate_to_reviewed() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        let entity = hg
            .set_entity_maturity(&eid, MaturityLevel::Reviewed, "alice", None)
            .unwrap();
        assert_eq!(entity.maturity, MaturityLevel::Reviewed);
        // Verify persisted
        assert_eq!(
            hg.get_entity(&eid).unwrap().maturity,
            MaturityLevel::Reviewed
        );
    }

    #[test]
    fn test_set_maturity_reviewed_to_validated() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(&eid, MaturityLevel::Reviewed, "alice", None)
            .unwrap();
        let entity = hg
            .set_entity_maturity(&eid, MaturityLevel::Validated, "bob", None)
            .unwrap();
        assert_eq!(entity.maturity, MaturityLevel::Validated);
    }

    #[test]
    fn test_set_maturity_skip_levels_allowed() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        // Skip from Candidate directly to Validated
        let entity = hg
            .set_entity_maturity(&eid, MaturityLevel::Validated, "admin", None)
            .unwrap();
        assert_eq!(entity.maturity, MaturityLevel::Validated);
    }

    #[test]
    fn test_set_maturity_backward_rejected() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(&eid, MaturityLevel::Validated, "alice", None)
            .unwrap();
        let result = hg.set_entity_maturity(&eid, MaturityLevel::Reviewed, "bob", None);
        assert!(matches!(
            result,
            Err(TensaError::InvalidMaturityTransition { .. })
        ));
    }

    #[test]
    fn test_validation_log_created() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(
            &eid,
            MaturityLevel::Reviewed,
            "alice",
            Some("Looks good".to_string()),
        )
        .unwrap();

        let log = hg.get_validation_log(&eid).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].old_maturity, MaturityLevel::Candidate);
        assert_eq!(log[0].new_maturity, MaturityLevel::Reviewed);
    }

    #[test]
    fn test_validation_log_records_reviewer() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(&eid, MaturityLevel::Reviewed, "alice", None)
            .unwrap();

        let log = hg.get_validation_log(&eid).unwrap();
        assert_eq!(log[0].reviewer, "alice");
    }

    #[test]
    fn test_validation_log_notes() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(
            &eid,
            MaturityLevel::Reviewed,
            "alice",
            Some("Verified against source material".to_string()),
        )
        .unwrap();

        let log = hg.get_validation_log(&eid).unwrap();
        assert_eq!(
            log[0].notes.as_deref(),
            Some("Verified against source material")
        );
    }

    #[test]
    fn test_promote_entity_maturity() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        let entity = hg.promote_entity_maturity(&eid, "alice", None).unwrap();
        assert_eq!(entity.maturity, MaturityLevel::Reviewed);
    }

    #[test]
    fn test_promote_at_max_fails() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(&eid, MaturityLevel::GroundTruth, "admin", None)
            .unwrap();
        let result = hg.promote_entity_maturity(&eid, "alice", None);
        assert!(matches!(
            result,
            Err(TensaError::InvalidMaturityTransition { .. })
        ));
    }

    #[test]
    fn test_set_situation_maturity() {
        let hg = Hypergraph::new(test_store());
        let sid = setup_situation(&hg);
        let sit = hg
            .set_situation_maturity(&sid, MaturityLevel::Reviewed, "alice", None)
            .unwrap();
        assert_eq!(sit.maturity, MaturityLevel::Reviewed);
    }

    #[test]
    fn test_promote_situation_maturity() {
        let hg = Hypergraph::new(test_store());
        let sid = setup_situation(&hg);
        let sit = hg.promote_situation_maturity(&sid, "alice", None).unwrap();
        assert_eq!(sit.maturity, MaturityLevel::Reviewed);
    }

    #[test]
    fn test_multiple_validation_log_entries() {
        let hg = Hypergraph::new(test_store());
        let eid = setup_entity(&hg);
        hg.set_entity_maturity(&eid, MaturityLevel::Reviewed, "alice", None)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        hg.set_entity_maturity(&eid, MaturityLevel::Validated, "bob", None)
            .unwrap();

        let log = hg.get_validation_log(&eid).unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].reviewer, "alice");
        assert_eq!(log[1].reviewer, "bob");
    }
}
