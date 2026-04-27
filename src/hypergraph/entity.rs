use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::store::TxnOp;
use crate::types::*;
use chrono::Utc;
use uuid::Uuid;

impl Hypergraph {
    /// Write secondary index entries for an entity.
    fn write_entity_indexes(&self, entity: &Entity) -> Result<()> {
        // Entity type index: et/{type}/{uuid} → empty
        let type_key = keys::entity_type_index_key(entity.entity_type.as_index_str(), &entity.id);
        self.store.put(&type_key, &[])?;
        // Entity narrative index: en/{narrative_id}/{uuid} → empty
        if let Some(ref nid) = entity.narrative_id {
            let nar_key = keys::entity_narrative_index_key(nid, &entity.id);
            self.store.put(&nar_key, &[])?;
        }
        Ok(())
    }

    /// Remove secondary index entries for an entity.
    fn remove_entity_indexes(&self, entity: &Entity) -> Result<()> {
        let type_key = keys::entity_type_index_key(entity.entity_type.as_index_str(), &entity.id);
        self.store.delete(&type_key)?;
        if let Some(ref nid) = entity.narrative_id {
            let nar_key = keys::entity_narrative_index_key(nid, &entity.id);
            self.store.delete(&nar_key)?;
        }
        Ok(())
    }

    /// Create a new entity and store it at `e/{id}`.
    /// Returns the entity's UUID.
    pub fn create_entity(&self, entity: Entity) -> Result<Uuid> {
        let id = entity.id;
        let mut entity = entity;
        entity.transaction_time = Some(Utc::now());
        let bytes = serde_json::to_vec(&entity)?;
        self.store.put(&keys::entity_key(&id), &bytes)?;
        self.write_entity_indexes(&entity)?;
        // Store initial bitemporal version
        let _ = crate::hypergraph::bitemporal::store_bitemporal_snapshot(self.store(), &entity);
        Ok(id)
    }

    /// Get an entity by ID. Returns `EntityNotFound` if missing or soft-deleted.
    pub fn get_entity(&self, id: &Uuid) -> Result<Entity> {
        match self.store.get(&keys::entity_key(id))? {
            Some(bytes) => {
                let entity: Entity = serde_json::from_slice(&bytes)?;
                if entity.deleted_at.is_some() {
                    Err(TensaError::EntityNotFound(*id))
                } else {
                    Ok(entity)
                }
            }
            None => Err(TensaError::EntityNotFound(*id)),
        }
    }

    /// Get an entity by ID, including soft-deleted entities.
    pub fn get_entity_include_deleted(&self, id: &Uuid) -> Result<Entity> {
        match self.store.get(&keys::entity_key(id))? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Err(TensaError::EntityNotFound(*id)),
        }
    }

    /// Internal update helper. When `snapshot` is true, creates a `StateVersion`
    /// capturing the entity's pre-update state before applying the closure.
    fn update_entity_inner(
        &self,
        id: &Uuid,
        updater: impl FnOnce(&mut Entity),
        snapshot: bool,
    ) -> Result<Entity> {
        let old = self.get_entity(id)?;

        if snapshot {
            let snapshot_id = Uuid::now_v7();
            let sv = StateVersion {
                entity_id: *id,
                situation_id: snapshot_id,
                properties: old.properties.clone(),
                beliefs: old.beliefs.clone(),
                embedding: old.embedding.clone(),
                timestamp: old.updated_at,
            };
            self.create_state_version(sv)?;
        }

        // Store bitemporal snapshot of the old version before overwriting
        if snapshot {
            let _ = crate::hypergraph::bitemporal::store_bitemporal_snapshot(self.store(), &old);
        }

        let mut entity = old.clone();
        updater(&mut entity);
        entity.updated_at = Utc::now();
        entity.transaction_time = Some(Utc::now());
        let bytes = serde_json::to_vec(&entity)?;
        self.store.put(&keys::entity_key(id), &bytes)?;
        // Update secondary indexes if indexed fields changed
        if old.entity_type != entity.entity_type || old.narrative_id != entity.narrative_id {
            self.remove_entity_indexes(&old)?;
            self.write_entity_indexes(&entity)?;
        }
        Ok(entity)
    }

    /// Update an entity by applying a closure, then re-serializing.
    /// Automatically creates a `StateVersion` snapshot of the pre-update state
    /// and updates `updated_at`. Returns the updated entity.
    pub fn update_entity(&self, id: &Uuid, updater: impl FnOnce(&mut Entity)) -> Result<Entity> {
        self.update_entity_inner(id, updater, true)
    }

    /// Update an entity without creating a state snapshot.
    /// Use this for derived/automated updates (e.g., confidence recomputation)
    /// where snapshotting would add noise to the state history.
    pub fn update_entity_no_snapshot(
        &self,
        id: &Uuid,
        updater: impl FnOnce(&mut Entity),
    ) -> Result<Entity> {
        self.update_entity_inner(id, updater, false)
    }

    /// Soft-delete an entity by setting `deleted_at`. Returns `EntityNotFound` if missing.
    pub fn delete_entity(&self, id: &Uuid) -> Result<()> {
        let entity = self.get_entity(id)?;
        let mut deleted = entity;
        deleted.deleted_at = Some(Utc::now());
        deleted.updated_at = Utc::now();
        let bytes = serde_json::to_vec(&deleted)?;
        self.store.put(&keys::entity_key(id), &bytes)?;
        // Remove from secondary indexes so soft-deleted entities don't appear in list queries
        self.remove_entity_indexes(&deleted)?;
        Ok(())
    }

    /// Restore a soft-deleted entity by clearing `deleted_at`.
    pub fn restore_entity(&self, id: &Uuid) -> Result<Entity> {
        let entity = self.get_entity_include_deleted(id)?;
        if entity.deleted_at.is_none() {
            return Err(TensaError::QueryError(format!(
                "Entity {} is not deleted",
                id
            )));
        }
        let mut restored = entity;
        restored.deleted_at = None;
        restored.updated_at = Utc::now();
        let bytes = serde_json::to_vec(&restored)?;
        self.store.put(&keys::entity_key(id), &bytes)?;
        self.write_entity_indexes(&restored)?;
        Ok(restored)
    }

    /// Permanently remove an entity from storage (hard delete).
    pub fn hard_delete_entity(&self, id: &Uuid) -> Result<()> {
        let entity = self.get_entity_include_deleted(id)?;
        if entity.deleted_at.is_none() {
            self.remove_entity_indexes(&entity)?;
        }
        self.store.delete(&keys::entity_key(id))
    }

    /// List all entities of a given type using the `et/` secondary index.
    pub fn list_entities_by_type(&self, entity_type: &EntityType) -> Result<Vec<Entity>> {
        let prefix = keys::entity_type_index_prefix(entity_type.as_index_str());
        let ids = self.ids_from_index(&prefix)?;
        Ok(ids
            .into_iter()
            .filter_map(|id| self.get_entity(&id).ok())
            .collect())
    }

    /// List all entities belonging to a specific narrative using the `en/` secondary index.
    pub fn list_entities_by_narrative(&self, narrative_id: &str) -> Result<Vec<Entity>> {
        let prefix = keys::entity_narrative_index_prefix(narrative_id);
        let ids = self.ids_from_index(&prefix)?;
        Ok(ids
            .into_iter()
            .filter_map(|id| self.get_entity(&id).ok())
            .collect())
    }

    /// Find the first entity in a narrative whose name / slug / alias matches
    /// `needle`. Diacritic-insensitive, case-insensitive; separators (space,
    /// hyphen, underscore) normalized to `-`. Returns `Ok(None)` on no match.
    pub fn find_entity_by_name(&self, narrative_id: &str, needle: &str) -> Result<Option<Entity>> {
        use crate::text_util::normalize_slug;
        let target = normalize_slug(needle);
        for entity in self.list_entities_by_narrative(narrative_id)? {
            let props = &entity.properties;
            let name_hit = ["name", "slug", "id"].iter().any(|f| {
                props
                    .get(*f)
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| normalize_slug(s) == target)
            });
            if name_hit {
                return Ok(Some(entity));
            }
            let alias_hit = props
                .get("aliases")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|a| a.as_str())
                .any(|s| normalize_slug(s) == target);
            if alias_hit {
                return Ok(Some(entity));
            }
        }
        Ok(None)
    }

    /// List all entities at or above a given maturity level.
    /// Still uses full scan — maturity is ordinal and changes frequently.
    pub fn list_entities_by_maturity(&self, min_maturity: MaturityLevel) -> Result<Vec<Entity>> {
        let pairs = self.store.prefix_scan(keys::ENTITY)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let entity: Entity = serde_json::from_slice(&value)?;
            if entity.deleted_at.is_none() && entity.maturity >= min_maturity {
                result.push(entity);
            }
        }
        Ok(result)
    }

    /// Transfer state versions from one entity to another.
    /// If `situation_ids` is `None`, transfers all state versions.
    /// If `Some(ids)`, transfers only those for the specified situations.
    fn transfer_state_versions(
        &self,
        from_id: &Uuid,
        to_id: &Uuid,
        situation_ids: Option<&[Uuid]>,
    ) -> Result<()> {
        match situation_ids {
            None => {
                // Transfer all: prefix scan sv/{from}/
                let sv_prefix = keys::state_version_prefix(from_id);
                let sv_pairs = self.store.prefix_scan(&sv_prefix)?;
                for (old_key, value) in sv_pairs {
                    let mut sv: StateVersion = serde_json::from_slice(&value)?;
                    sv.entity_id = *to_id;
                    let new_key = keys::state_version_key(to_id, &sv.situation_id);
                    let new_bytes = serde_json::to_vec(&sv)?;
                    self.store.transaction(vec![
                        TxnOp::Put(new_key, new_bytes),
                        TxnOp::Delete(old_key),
                    ])?;
                }
            }
            Some(ids) => {
                for sit_id in ids {
                    let sv_key = keys::state_version_key(from_id, sit_id);
                    if let Some(sv_bytes) = self.store.get(&sv_key)? {
                        let mut sv: StateVersion = serde_json::from_slice(&sv_bytes)?;
                        sv.entity_id = *to_id;
                        let new_sv_key = keys::state_version_key(to_id, sit_id);
                        let new_sv_bytes = serde_json::to_vec(&sv)?;
                        self.store.transaction(vec![
                            TxnOp::Put(new_sv_key, new_sv_bytes),
                            TxnOp::Delete(sv_key),
                        ])?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Merge two entities: keep `keep_id`, absorb `absorb_id`.
    /// Re-points all participations, state versions, and source attributions.
    /// Merges properties (keep takes precedence), provenance, and confidence.
    /// Deletes the absorbed entity afterward.
    pub fn merge_entities(&self, keep_id: &Uuid, absorb_id: &Uuid) -> Result<Entity> {
        if keep_id == absorb_id {
            return Err(TensaError::QueryError(
                "Cannot merge an entity with itself".into(),
            ));
        }
        let keep = self.get_entity(keep_id)?;
        let absorb = self.get_entity(absorb_id)?;

        // 1. Re-point participations: scan p/{absorb}/
        let absorb_participations = self.get_situations_for_entity(absorb_id)?;
        for mut part in absorb_participations {
            // Remove the absorbed participation by its exact seq
            self.remove_participant(absorb_id, &part.situation_id, Some(part.seq))?;
            // Re-point to keep entity and add (seq auto-assigned)
            part.entity_id = *keep_id;
            self.add_participant(part)?;
        }

        // 2. Re-point state versions: scan sv/{absorb}/
        self.transfer_state_versions(absorb_id, keep_id, None)?;

        // 3. Re-point source attributions atomically: scan sar/{absorb}/
        let sar_prefix = keys::source_attribution_reverse_prefix(absorb_id);
        let sar_pairs = self.store.prefix_scan(&sar_prefix)?;
        for (old_rev_key, value) in sar_pairs {
            let attr: serde_json::Value = serde_json::from_slice(&value)?;
            if let Some(source_id_str) = attr.get("source_id").and_then(|v| v.as_str()) {
                if let Ok(source_id) = Uuid::parse_str(source_id_str) {
                    let old_fwd_key = keys::source_attribution_key(&source_id, absorb_id);
                    let new_fwd_key = keys::source_attribution_key(&source_id, keep_id);
                    let new_rev_key = keys::source_attribution_reverse_key(keep_id, &source_id);
                    self.store.transaction(vec![
                        TxnOp::Delete(old_fwd_key),
                        TxnOp::Delete(old_rev_key.clone()),
                        TxnOp::Put(new_fwd_key, value.clone()),
                        TxnOp::Put(new_rev_key, value),
                    ])?;
                    continue; // old_rev_key already deleted in transaction
                }
            }
            self.store.delete(&old_rev_key)?;
        }

        // 4. Merge properties: deep-merge (keep takes precedence)
        let mut merged_props = absorb.properties.clone();
        if let (Some(merged_obj), Some(keep_obj)) =
            (merged_props.as_object_mut(), keep.properties.as_object())
        {
            for (k, v) in keep_obj {
                merged_obj.insert(k.clone(), v.clone());
            }
        }

        // 5. Update keep entity
        let result = self.update_entity(keep_id, |entity| {
            entity.properties = merged_props.clone();
            entity.confidence = entity.confidence.max(absorb.confidence);
            // Merge provenance arrays
            entity.provenance.extend(absorb.provenance.clone());
        })?;

        // 6. Delete absorbed entity
        self.delete_entity(absorb_id)?;

        Ok(result)
    }

    /// Split an entity: clone it and move specified situations to the clone.
    /// Returns `(original, clone)` — the clone gets a new UUID.
    pub fn split_entity(
        &self,
        source_id: &Uuid,
        situation_ids: Vec<Uuid>,
    ) -> Result<(Entity, Entity)> {
        if situation_ids.is_empty() {
            return Err(TensaError::QueryError(
                "Cannot split with empty situation list".into(),
            ));
        }

        let source = self.get_entity(source_id)?;

        // 1. Create clone with new UUID
        let clone_id = Uuid::now_v7();
        let clone = Entity {
            id: clone_id,
            entity_type: source.entity_type.clone(),
            properties: source.properties.clone(),
            beliefs: source.beliefs.clone(),
            embedding: source.embedding.clone(),
            maturity: source.maturity,
            confidence: source.confidence,
            confidence_breakdown: source.confidence_breakdown.clone(),
            provenance: source.provenance.clone(),
            extraction_method: source.extraction_method,
            narrative_id: source.narrative_id.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        self.create_entity(clone)?;

        // 2. Move specified participations to clone
        for sit_id in &situation_ids {
            let parts = self.get_participations_for_pair(source_id, sit_id)?;
            for mut part in parts {
                self.remove_participant(source_id, sit_id, Some(part.seq))?;
                part.entity_id = clone_id;
                self.add_participant(part)?;
            }
        }

        // 3. Move state versions for split situations
        self.transfer_state_versions(source_id, &clone_id, Some(&situation_ids))?;

        let updated_source = self.get_entity(source_id)?;
        let updated_clone = self.get_entity(&clone_id)?;
        Ok((updated_source, updated_clone))
    }
}

#[cfg(test)]
#[path = "entity_tests.rs"]
mod tests;
