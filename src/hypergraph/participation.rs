use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::store::TxnOp;
use crate::types::*;
use uuid::Uuid;

impl Hypergraph {
    /// Add a participant to a situation.
    /// Validates both entity and situation exist.
    /// Supports multi-role: the same entity can participate multiple times
    /// in the same situation with different roles. The `seq` field is
    /// auto-assigned based on existing entries for the (entity, situation) pair.
    /// Writes dual index: `p/{entity}/{situation}/{seq}` and `ps/{situation}/{entity}/{seq}`.
    pub fn add_participant(&self, mut participation: Participation) -> Result<()> {
        // Validate entity exists
        self.get_entity(&participation.entity_id)?;
        // Validate situation exists
        self.get_situation(&participation.situation_id)?;

        // Count existing entries for this pair to assign seq
        let pair_prefix =
            keys::participation_pair_prefix(&participation.entity_id, &participation.situation_id);
        let existing = self.store.prefix_scan(&pair_prefix)?;
        participation.seq = existing.len() as u16;

        let forward_key = keys::participation_seq_key(
            &participation.entity_id,
            &participation.situation_id,
            participation.seq,
        );
        let reverse_key = keys::participation_reverse_seq_key(
            &participation.situation_id,
            &participation.entity_id,
            participation.seq,
        );
        let bytes = serde_json::to_vec(&participation)?;

        self.store.transaction(vec![
            TxnOp::Put(forward_key, bytes.clone()),
            TxnOp::Put(reverse_key, bytes),
        ])
    }

    /// Remove participant(s). If `seq` is `Some(n)`, deletes only that specific
    /// entry. If `seq` is `None`, deletes ALL entries for the (entity, situation) pair.
    pub fn remove_participant(
        &self,
        entity_id: &Uuid,
        situation_id: &Uuid,
        seq: Option<u16>,
    ) -> Result<()> {
        match seq {
            Some(n) => {
                let forward_key = keys::participation_seq_key(entity_id, situation_id, n);
                if self.store.get(&forward_key)?.is_none() {
                    return Err(TensaError::ParticipationNotFound {
                        entity_id: *entity_id,
                        situation_id: *situation_id,
                    });
                }
                let reverse_key = keys::participation_reverse_seq_key(situation_id, entity_id, n);
                self.store
                    .transaction(vec![TxnOp::Delete(forward_key), TxnOp::Delete(reverse_key)])
            }
            None => {
                let pair_prefix = keys::participation_pair_prefix(entity_id, situation_id);
                let entries = self.store.prefix_scan(&pair_prefix)?;
                if entries.is_empty() {
                    return Err(TensaError::ParticipationNotFound {
                        entity_id: *entity_id,
                        situation_id: *situation_id,
                    });
                }
                let mut ops = Vec::new();
                for (fwd_key, value) in &entries {
                    let p: Participation = serde_json::from_slice(value)?;
                    ops.push(TxnOp::Delete(fwd_key.clone()));
                    ops.push(TxnOp::Delete(keys::participation_reverse_seq_key(
                        situation_id,
                        entity_id,
                        p.seq,
                    )));
                }
                self.store.transaction(ops)
            }
        }
    }

    /// Scan a KV prefix and deserialize all values as Participation records.
    fn scan_participations(&self, prefix: &[u8]) -> Result<Vec<Participation>> {
        self.store
            .prefix_scan(prefix)?
            .into_iter()
            .map(|(_k, v)| serde_json::from_slice(&v).map_err(Into::into))
            .collect()
    }

    /// Update a participation in-place by rewriting both forward and reverse keys.
    pub fn update_participation(&self, participation: &Participation) -> Result<()> {
        let forward_key = keys::participation_seq_key(
            &participation.entity_id,
            &participation.situation_id,
            participation.seq,
        );
        let reverse_key = keys::participation_reverse_seq_key(
            &participation.situation_id,
            &participation.entity_id,
            participation.seq,
        );
        let bytes = serde_json::to_vec(participation)?;
        self.store.transaction(vec![
            TxnOp::Put(forward_key, bytes.clone()),
            TxnOp::Put(reverse_key, bytes),
        ])
    }

    /// Get all participations for a specific (entity, situation) pair.
    pub fn get_participations_for_pair(
        &self,
        entity_id: &Uuid,
        situation_id: &Uuid,
    ) -> Result<Vec<Participation>> {
        self.scan_participations(&keys::participation_pair_prefix(entity_id, situation_id))
    }

    /// Get all participants in a situation (reverse index scan on `ps/{situation}/`).
    pub fn get_participants_for_situation(
        &self,
        situation_id: &Uuid,
    ) -> Result<Vec<Participation>> {
        self.scan_participations(&keys::participation_prefix_for_situation(situation_id))
    }

    /// Get all situations an entity participates in (forward index scan on `p/{entity}/`).
    pub fn get_situations_for_entity(&self, entity_id: &Uuid) -> Result<Vec<Participation>> {
        self.scan_participations(&keys::participation_prefix_for_entity(entity_id))
    }

    /// Filter participations by role (in-memory filter, no KV access).
    pub fn filter_by_role(participations: &[Participation], role: &Role) -> Vec<Participation> {
        participations
            .iter()
            .filter(|p| &p.role == role)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
#[path = "participation_tests.rs"]
mod tests;
