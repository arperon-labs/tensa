use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::types::*;
use chrono::Utc;
use uuid::Uuid;

impl Hypergraph {
    /// Write secondary index entries for a situation.
    fn write_situation_indexes(&self, situation: &Situation) -> Result<()> {
        // Situation level index: sl/{level}/{uuid} → empty
        let level_key = keys::situation_level_index_key(
            situation.narrative_level.as_index_str(),
            &situation.id,
        );
        self.store.put(&level_key, &[])?;
        // Situation narrative index: sn/{narrative_id}/{uuid} → empty
        if let Some(ref nid) = situation.narrative_id {
            let nar_key = keys::situation_narrative_index_key(nid, &situation.id);
            self.store.put(&nar_key, &[])?;
        }
        Ok(())
    }

    /// Remove secondary index entries for a situation.
    fn remove_situation_indexes(&self, situation: &Situation) -> Result<()> {
        let level_key = keys::situation_level_index_key(
            situation.narrative_level.as_index_str(),
            &situation.id,
        );
        self.store.delete(&level_key)?;
        if let Some(ref nid) = situation.narrative_id {
            let nar_key = keys::situation_narrative_index_key(nid, &situation.id);
            self.store.delete(&nar_key)?;
        }
        Ok(())
    }

    /// Create a new situation and store it at `s/{id}`.
    /// Returns the situation's UUID.
    pub fn create_situation(&self, situation: Situation) -> Result<Uuid> {
        let id = situation.id;
        let bytes = serde_json::to_vec(&situation)?;
        self.store.put(&keys::situation_key(&id), &bytes)?;
        self.write_situation_indexes(&situation)?;
        Ok(id)
    }

    /// Get a situation by ID. Returns `SituationNotFound` if missing or soft-deleted.
    pub fn get_situation(&self, id: &Uuid) -> Result<Situation> {
        match self.store.get(&keys::situation_key(id))? {
            Some(bytes) => {
                let situation: Situation = serde_json::from_slice(&bytes)?;
                if situation.deleted_at.is_some() {
                    Err(TensaError::SituationNotFound(*id))
                } else {
                    Ok(situation)
                }
            }
            None => Err(TensaError::SituationNotFound(*id)),
        }
    }

    /// Get a situation by ID, including soft-deleted situations.
    pub fn get_situation_include_deleted(&self, id: &Uuid) -> Result<Situation> {
        match self.store.get(&keys::situation_key(id))? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Err(TensaError::SituationNotFound(*id)),
        }
    }

    /// Update a situation by applying a closure, then re-serializing.
    /// Automatically updates `updated_at`. Returns the updated situation.
    pub fn update_situation(
        &self,
        id: &Uuid,
        updater: impl FnOnce(&mut Situation),
    ) -> Result<Situation> {
        let old = self.get_situation(id)?;
        let mut situation = old.clone();
        updater(&mut situation);
        situation.updated_at = Utc::now();
        let bytes = serde_json::to_vec(&situation)?;
        self.store.put(&keys::situation_key(id), &bytes)?;
        // Update secondary indexes if indexed fields changed
        if old.narrative_level != situation.narrative_level
            || old.narrative_id != situation.narrative_id
        {
            self.remove_situation_indexes(&old)?;
            self.write_situation_indexes(&situation)?;
        }
        Ok(situation)
    }

    /// Soft-delete a situation by setting `deleted_at`. Returns `SituationNotFound` if missing.
    pub fn delete_situation(&self, id: &Uuid) -> Result<()> {
        let situation = self.get_situation(id)?;
        let mut deleted = situation;
        deleted.deleted_at = Some(Utc::now());
        deleted.updated_at = Utc::now();
        let bytes = serde_json::to_vec(&deleted)?;
        self.store.put(&keys::situation_key(id), &bytes)?;
        self.remove_situation_indexes(&deleted)?;
        Ok(())
    }

    /// Restore a soft-deleted situation by clearing `deleted_at`.
    pub fn restore_situation(&self, id: &Uuid) -> Result<Situation> {
        let situation = self.get_situation_include_deleted(id)?;
        if situation.deleted_at.is_none() {
            return Err(TensaError::QueryError(format!(
                "Situation {} is not deleted",
                id
            )));
        }
        let mut restored = situation;
        restored.deleted_at = None;
        restored.updated_at = Utc::now();
        let bytes = serde_json::to_vec(&restored)?;
        self.store.put(&keys::situation_key(id), &bytes)?;
        self.write_situation_indexes(&restored)?;
        Ok(restored)
    }

    /// Permanently remove a situation from storage (hard delete).
    pub fn hard_delete_situation(&self, id: &Uuid) -> Result<()> {
        let situation = self.get_situation_include_deleted(id)?;
        if situation.deleted_at.is_none() {
            self.remove_situation_indexes(&situation)?;
        }
        self.store.delete(&keys::situation_key(id))
    }

    /// List all situations at a given narrative level using the `sl/` secondary index.
    pub fn list_situations_by_level(&self, level: NarrativeLevel) -> Result<Vec<Situation>> {
        let prefix = keys::situation_level_index_prefix(level.as_index_str());
        let ids = self.ids_from_index(&prefix)?;
        Ok(ids
            .into_iter()
            .filter_map(|id| self.get_situation(&id).ok())
            .collect())
    }

    /// List all situations belonging to a specific narrative using the `sn/` secondary index.
    pub fn list_situations_by_narrative(&self, narrative_id: &str) -> Result<Vec<Situation>> {
        let prefix = keys::situation_narrative_index_prefix(narrative_id);
        let ids = self.ids_from_index(&prefix)?;
        Ok(ids
            .into_iter()
            .filter_map(|id| self.get_situation(&id).ok())
            .collect())
    }

    /// List all situations at or above a given maturity level.
    /// Still uses full scan — maturity is ordinal and changes frequently.
    pub fn list_situations_by_maturity(
        &self,
        min_maturity: MaturityLevel,
    ) -> Result<Vec<Situation>> {
        let pairs = self.store.prefix_scan(keys::SITUATION)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let situation: Situation = serde_json::from_slice(&value)?;
            if situation.deleted_at.is_none() && situation.maturity >= min_maturity {
                result.push(situation);
            }
        }
        Ok(result)
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

    fn test_situation(level: NarrativeLevel) -> Situation {
        Situation {
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
            raw_content: vec![ContentBlock::text("Test situation")],
            narrative_level: level,
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
        }
    }

    #[test]
    fn test_create_situation_minimal() {
        let hg = Hypergraph::new(test_store());
        let sit = test_situation(NarrativeLevel::Scene);
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        assert_eq!(retrieved.narrative_level, NarrativeLevel::Scene);
    }

    #[test]
    fn test_create_situation_full() {
        let hg = Hypergraph::new(test_store());
        let mut sit = test_situation(NarrativeLevel::Scene);
        sit.spatial = Some(SpatialAnchor {
            latitude: Some(59.93),
            longitude: Some(30.32),
            precision: SpatialPrecision::Area,
            location_entity: None,
            location_name: Some("St. Petersburg".to_string()),
            description: Some("St. Petersburg".to_string()),
            geo_provenance: Some(crate::types::GeoProvenance::Source),
        });
        sit.game_structure = Some(GameStructure {
            game_type: GameClassification::AsymmetricInformation,
            info_structure: InfoStructureType::Incomplete,
            description: Some("Interrogation".to_string()),
            maturity: MaturityLevel::Candidate,
        });
        sit.discourse = Some(DiscourseAnnotation {
            order: Some("prolepsis".to_string()),
            duration: Some("scene".to_string()),
            focalization: None,
            voice: Some("homodiegetic".to_string()),
        });
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        assert!(retrieved.spatial.is_some());
        assert!(retrieved.game_structure.is_some());
        assert!(retrieved.discourse.is_some());
    }

    #[test]
    fn test_get_situation_not_found() {
        let hg = Hypergraph::new(test_store());
        let result = hg.get_situation(&Uuid::now_v7());
        assert!(matches!(result, Err(TensaError::SituationNotFound(_))));
    }

    #[test]
    fn test_update_situation_add_content() {
        let hg = Hypergraph::new(test_store());
        let sit = test_situation(NarrativeLevel::Scene);
        let id = hg.create_situation(sit).unwrap();
        let updated = hg
            .update_situation(&id, |s| {
                s.raw_content.push(ContentBlock::text("Additional content"));
            })
            .unwrap();
        assert_eq!(updated.raw_content.len(), 2);
    }

    #[test]
    fn test_update_situation_game_structure() {
        let hg = Hypergraph::new(test_store());
        let sit = test_situation(NarrativeLevel::Scene);
        let id = hg.create_situation(sit).unwrap();
        let updated = hg
            .update_situation(&id, |s| {
                s.game_structure = Some(GameStructure {
                    game_type: GameClassification::PrisonersDilemma,
                    info_structure: InfoStructureType::Complete,
                    description: None,
                    maturity: MaturityLevel::Candidate,
                });
            })
            .unwrap();
        assert!(updated.game_structure.is_some());
    }

    #[test]
    fn test_delete_situation() {
        let hg = Hypergraph::new(test_store());
        let sit = test_situation(NarrativeLevel::Scene);
        let id = hg.create_situation(sit).unwrap();
        hg.delete_situation(&id).unwrap();
        assert!(matches!(
            hg.get_situation(&id),
            Err(TensaError::SituationNotFound(_))
        ));
    }

    #[test]
    fn test_delete_situation_not_found() {
        let hg = Hypergraph::new(test_store());
        assert!(matches!(
            hg.delete_situation(&Uuid::now_v7()),
            Err(TensaError::SituationNotFound(_))
        ));
    }

    #[test]
    fn test_list_situations_by_level() {
        let hg = Hypergraph::new(test_store());
        hg.create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.create_situation(test_situation(NarrativeLevel::Arc))
            .unwrap();

        let scenes = hg.list_situations_by_level(NarrativeLevel::Scene).unwrap();
        assert_eq!(scenes.len(), 2);
        let arcs = hg.list_situations_by_level(NarrativeLevel::Arc).unwrap();
        assert_eq!(arcs.len(), 1);
    }

    #[test]
    fn test_list_situations_by_maturity() {
        let hg = Hypergraph::new(test_store());
        let mut s1 = test_situation(NarrativeLevel::Scene);
        s1.maturity = MaturityLevel::Candidate;
        let mut s2 = test_situation(NarrativeLevel::Scene);
        s2.maturity = MaturityLevel::Validated;
        let mut s3 = test_situation(NarrativeLevel::Scene);
        s3.maturity = MaturityLevel::GroundTruth;

        hg.create_situation(s1).unwrap();
        hg.create_situation(s2).unwrap();
        hg.create_situation(s3).unwrap();

        let validated_plus = hg
            .list_situations_by_maturity(MaturityLevel::Validated)
            .unwrap();
        assert_eq!(validated_plus.len(), 2);
    }

    #[test]
    fn test_situation_serialization_roundtrip() {
        let hg = Hypergraph::new(test_store());
        let mut sit = test_situation(NarrativeLevel::Beat);
        sit.confidence = 0.42;
        sit.deterministic = Some(serde_json::json!({"outcome": "success"}));
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        assert_eq!(retrieved.confidence, 0.42);
        assert_eq!(retrieved.deterministic.unwrap()["outcome"], "success");
    }

    #[test]
    fn test_situation_with_temporal() {
        let hg = Hypergraph::new(test_store());
        let mut sit = test_situation(NarrativeLevel::Event);
        sit.temporal = AllenInterval {
            start: Some(Utc::now()),
            end: Some(Utc::now()),
            granularity: TimeGranularity::Exact,
            relations: vec![],
            fuzzy_endpoints: None,
        };
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        assert_eq!(retrieved.temporal.granularity, TimeGranularity::Exact);
    }

    #[test]
    fn test_situation_with_spatial() {
        let hg = Hypergraph::new(test_store());
        let mut sit = test_situation(NarrativeLevel::Scene);
        sit.spatial = Some(SpatialAnchor {
            latitude: Some(55.75),
            longitude: Some(37.62),
            precision: SpatialPrecision::Exact,
            location_entity: None,
            location_name: Some("Moscow".to_string()),
            description: Some("Moscow".to_string()),
            geo_provenance: Some(crate::types::GeoProvenance::Source),
        });
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        let spatial = retrieved.spatial.unwrap();
        assert_eq!(spatial.latitude, Some(55.75));
    }

    #[test]
    fn test_situation_with_raw_content() {
        let hg = Hypergraph::new(test_store());
        let mut sit = test_situation(NarrativeLevel::Scene);
        sit.raw_content = vec![
            ContentBlock::text("He entered the room..."),
            ContentBlock {
                content_type: ContentType::Dialogue,
                content: "I confess!".to_string(),
                source: None,
            },
        ];
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        assert_eq!(retrieved.raw_content.len(), 2);
        assert_eq!(retrieved.raw_content[1].content_type, ContentType::Dialogue);
    }

    #[test]
    fn test_situation_extraction_method_tracking() {
        let hg = Hypergraph::new(test_store());
        let mut sit = test_situation(NarrativeLevel::Scene);
        sit.extraction_method = ExtractionMethod::LlmParsed;
        let id = hg.create_situation(sit).unwrap();
        let retrieved = hg.get_situation(&id).unwrap();
        assert_eq!(retrieved.extraction_method, ExtractionMethod::LlmParsed);
    }

    #[test]
    fn test_list_situations_empty() {
        let hg = Hypergraph::new(test_store());
        let scenes = hg.list_situations_by_level(NarrativeLevel::Scene).unwrap();
        assert!(scenes.is_empty());
    }

    #[test]
    fn test_update_situation_updates_timestamp() {
        let hg = Hypergraph::new(test_store());
        let sit = test_situation(NarrativeLevel::Scene);
        let original_time = sit.updated_at;
        let id = hg.create_situation(sit).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let updated = hg
            .update_situation(&id, |s| {
                s.confidence = 0.5;
            })
            .unwrap();
        assert!(updated.updated_at > original_time);
    }

    // ─── Soft Delete Tests ─────────────────────────────────────

    #[test]
    fn test_soft_delete_situation_hides_from_get() {
        let hg = Hypergraph::new(test_store());
        let id = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.delete_situation(&id).unwrap();
        assert!(matches!(
            hg.get_situation(&id),
            Err(TensaError::SituationNotFound(_))
        ));
    }

    #[test]
    fn test_soft_delete_situation_visible_with_include_deleted() {
        let hg = Hypergraph::new(test_store());
        let id = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.delete_situation(&id).unwrap();
        let sit = hg.get_situation_include_deleted(&id).unwrap();
        assert!(sit.deleted_at.is_some());
    }

    #[test]
    fn test_restore_situation() {
        let hg = Hypergraph::new(test_store());
        let id = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.delete_situation(&id).unwrap();
        let restored = hg.restore_situation(&id).unwrap();
        assert!(restored.deleted_at.is_none());
        assert!(hg.get_situation(&id).is_ok());
    }

    #[test]
    fn test_restore_non_deleted_situation_errors() {
        let hg = Hypergraph::new(test_store());
        let id = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        assert!(hg.restore_situation(&id).is_err());
    }

    #[test]
    fn test_soft_delete_excludes_from_list_by_level() {
        let hg = Hypergraph::new(test_store());
        let id1 = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.delete_situation(&id1).unwrap();
        let scenes = hg.list_situations_by_level(NarrativeLevel::Scene).unwrap();
        assert_eq!(scenes.len(), 1);
    }

    #[test]
    fn test_soft_delete_excludes_from_list_by_maturity() {
        let hg = Hypergraph::new(test_store());
        let id = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.delete_situation(&id).unwrap();
        let all = hg
            .list_situations_by_maturity(MaturityLevel::Candidate)
            .unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_hard_delete_situation() {
        let hg = Hypergraph::new(test_store());
        let id = hg
            .create_situation(test_situation(NarrativeLevel::Scene))
            .unwrap();
        hg.delete_situation(&id).unwrap();
        hg.hard_delete_situation(&id).unwrap();
        assert!(hg.get_situation_include_deleted(&id).is_err());
    }
}
