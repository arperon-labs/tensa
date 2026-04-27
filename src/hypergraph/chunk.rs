//! Chunk storage CRUD operations.
//!
//! Persists `ChunkRecord` values produced during text ingestion, enabling
//! provenance tracing, re-analysis, and honest stylometry from original text.

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::store::TxnOp;
use crate::types::ChunkRecord;
use uuid::Uuid;

/// Parse a UUID from 16 index-value bytes, returning None on malformed data.
fn uuid_from_value(bytes: &[u8]) -> Option<Uuid> {
    let arr: [u8; 16] = bytes.try_into().ok()?;
    Some(Uuid::from_bytes(arr))
}

impl Hypergraph {
    /// Store a chunk record with all secondary indexes atomically.
    ///
    /// Indexes written:
    /// - `ch/r/{id}` — primary record
    /// - `ch/j/{job_id}/{chunk_index}` — job lookup
    /// - `ch/n/{narrative_id}/{chunk_index}` — narrative lookup (if narrative_id set)
    /// - `ch/h/{narrative_id}/{hash}` — hash dedup (if narrative_id set)
    pub fn store_chunk(&self, chunk: &ChunkRecord) -> Result<Uuid> {
        let id = chunk.id;
        let bytes = serde_json::to_vec(chunk)?;
        let id_bytes = id.as_bytes().to_vec();

        let mut ops = vec![
            TxnOp::Put(keys::chunk_record_key(&id), bytes),
            TxnOp::Put(
                keys::chunk_job_key(&chunk.job_id, chunk.chunk_index),
                id_bytes.clone(),
            ),
        ];

        if let Some(ref nid) = chunk.narrative_id {
            ops.push(TxnOp::Put(
                keys::chunk_narrative_key(nid, chunk.chunk_index),
                id_bytes.clone(),
            ));
            ops.push(TxnOp::Put(
                keys::chunk_hash_dedup_key(nid, &chunk.content_hash),
                id_bytes,
            ));
        }

        self.store.transaction(ops)?;
        Ok(id)
    }

    /// Get a chunk by its UUID.
    pub fn get_chunk(&self, id: &Uuid) -> Result<ChunkRecord> {
        match self.store.get(&keys::chunk_record_key(id))? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Err(TensaError::ChunkNotFound(*id)),
        }
    }

    /// Load chunk records from a prefix-scan index where values are UUID bytes.
    fn load_chunks_from_index(&self, prefix: &[u8]) -> Result<Vec<ChunkRecord>> {
        let entries = self.store.prefix_scan(prefix)?;
        let mut chunks = Vec::with_capacity(entries.len());
        for (_key, uuid_bytes) in entries {
            if let Some(id) = uuid_from_value(&uuid_bytes) {
                if let Some(record_bytes) = self.store.get(&keys::chunk_record_key(&id))? {
                    chunks.push(serde_json::from_slice(&record_bytes)?);
                }
            }
        }
        Ok(chunks)
    }

    /// List all chunks for a given ingestion job, ordered by chunk_index.
    pub fn list_chunks_by_job(&self, job_id: &str) -> Result<Vec<ChunkRecord>> {
        self.load_chunks_from_index(&keys::chunk_job_prefix(job_id))
    }

    /// List all chunks for a narrative, ordered by chunk_index.
    pub fn list_chunks_by_narrative(&self, narrative_id: &str) -> Result<Vec<ChunkRecord>> {
        self.load_chunks_from_index(&keys::chunk_narrative_prefix(narrative_id))
    }

    /// Check if a chunk with the given content hash exists for a narrative.
    /// Returns the chunk UUID if found.
    pub fn chunk_exists_by_hash(&self, narrative_id: &str, hash: &str) -> Result<Option<Uuid>> {
        let key = keys::chunk_hash_dedup_key(narrative_id, hash);
        match self.store.get(&key)? {
            Some(uuid_bytes) => Ok(uuid_from_value(&uuid_bytes)),
            None => Ok(None),
        }
    }

    /// Delete all chunks for a given ingestion job. Returns the number deleted.
    pub fn delete_chunks_by_job(&self, job_id: &str) -> Result<usize> {
        let prefix = keys::chunk_job_prefix(job_id);
        let entries = self.store.prefix_scan(&prefix)?;
        let mut count = 0;

        for (job_key, uuid_bytes) in entries {
            let id = match uuid_from_value(&uuid_bytes) {
                Some(id) => id,
                None => continue,
            };

            // Load record to get narrative_id and hash for index cleanup
            if let Some(record_bytes) = self.store.get(&keys::chunk_record_key(&id))? {
                if let Ok(chunk) = serde_json::from_slice::<ChunkRecord>(&record_bytes) {
                    if let Some(ref nid) = chunk.narrative_id {
                        self.store
                            .delete(&keys::chunk_narrative_key(nid, chunk.chunk_index))?;
                        self.store
                            .delete(&keys::chunk_hash_dedup_key(nid, &chunk.content_hash))?;
                    }
                }
            }

            self.store.delete(&keys::chunk_record_key(&id))?;
            self.store.delete(&job_key)?;
            count += 1;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;
    use std::sync::Arc;

    fn make_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_chunk(job_id: &str, narrative_id: Option<&str>, index: u32) -> ChunkRecord {
        ChunkRecord {
            id: Uuid::now_v7(),
            job_id: job_id.to_string(),
            narrative_id: narrative_id.map(|s| s.to_string()),
            chunk_index: index,
            text: format!("Chunk {} text content for testing.", index),
            byte_range: (index as usize * 100, (index as usize + 1) * 100),
            overlap_bytes: if index == 0 { 0 } else { 20 },
            chapter: None,
            content_hash: format!("hash_{}", index),
            embedding: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_store_and_get_chunk() {
        let hg = make_hg();
        let chunk = make_chunk("job-1", Some("nar-1"), 0);
        let id = chunk.id;

        let stored_id = hg.store_chunk(&chunk).unwrap();
        assert_eq!(stored_id, id);

        let retrieved = hg.get_chunk(&id).unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.job_id, "job-1");
        assert_eq!(retrieved.chunk_index, 0);
        assert_eq!(retrieved.text, "Chunk 0 text content for testing.");
    }

    #[test]
    fn test_get_chunk_not_found() {
        let hg = make_hg();
        let result = hg.get_chunk(&Uuid::now_v7());
        assert!(result.is_err());
    }

    #[test]
    fn test_list_chunks_by_job() {
        let hg = make_hg();
        let c0 = make_chunk("job-1", Some("nar-1"), 0);
        let c1 = make_chunk("job-1", Some("nar-1"), 1);
        let c2 = make_chunk("job-1", Some("nar-1"), 2);
        let other = make_chunk("job-2", Some("nar-1"), 0);

        hg.store_chunk(&c0).unwrap();
        hg.store_chunk(&c1).unwrap();
        hg.store_chunk(&c2).unwrap();
        hg.store_chunk(&other).unwrap();

        let chunks = hg.list_chunks_by_job("job-1").unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[2].chunk_index, 2);

        let other_chunks = hg.list_chunks_by_job("job-2").unwrap();
        assert_eq!(other_chunks.len(), 1);
    }

    #[test]
    fn test_list_chunks_by_narrative() {
        let hg = make_hg();
        let c0 = make_chunk("job-1", Some("nar-A"), 0);
        let c1 = make_chunk("job-1", Some("nar-A"), 1);
        let c2 = make_chunk("job-2", Some("nar-B"), 0);

        hg.store_chunk(&c0).unwrap();
        hg.store_chunk(&c1).unwrap();
        hg.store_chunk(&c2).unwrap();

        let chunks = hg.list_chunks_by_narrative("nar-A").unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);

        let chunks_b = hg.list_chunks_by_narrative("nar-B").unwrap();
        assert_eq!(chunks_b.len(), 1);
    }

    #[test]
    fn test_chunk_exists_by_hash() {
        let hg = make_hg();
        let chunk = make_chunk("job-1", Some("nar-1"), 0);
        let id = chunk.id;

        assert!(hg
            .chunk_exists_by_hash("nar-1", "hash_0")
            .unwrap()
            .is_none());

        hg.store_chunk(&chunk).unwrap();

        let found = hg.chunk_exists_by_hash("nar-1", "hash_0").unwrap();
        assert_eq!(found, Some(id));

        assert!(hg
            .chunk_exists_by_hash("nar-1", "hash_999")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_chunk_without_narrative_id() {
        let hg = make_hg();
        let chunk = make_chunk("job-1", None, 0);
        let id = chunk.id;

        hg.store_chunk(&chunk).unwrap();

        let retrieved = hg.get_chunk(&id).unwrap();
        assert_eq!(retrieved.id, id);
        assert!(retrieved.narrative_id.is_none());

        let by_job = hg.list_chunks_by_job("job-1").unwrap();
        assert_eq!(by_job.len(), 1);

        assert!(hg.chunk_exists_by_hash("any", "hash_0").unwrap().is_none());
    }

    #[test]
    fn test_delete_chunks_by_job() {
        let hg = make_hg();
        let c0 = make_chunk("job-del", Some("nar-del"), 0);
        let c1 = make_chunk("job-del", Some("nar-del"), 1);
        let id0 = c0.id;

        hg.store_chunk(&c0).unwrap();
        hg.store_chunk(&c1).unwrap();

        assert_eq!(hg.list_chunks_by_job("job-del").unwrap().len(), 2);

        let deleted = hg.delete_chunks_by_job("job-del").unwrap();
        assert_eq!(deleted, 2);

        assert!(hg.get_chunk(&id0).is_err());
        assert!(hg.list_chunks_by_job("job-del").unwrap().is_empty());
        assert!(hg.list_chunks_by_narrative("nar-del").unwrap().is_empty());
        assert!(hg
            .chunk_exists_by_hash("nar-del", "hash_0")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_chunk_ordering_preserved() {
        let hg = make_hg();
        let c2 = make_chunk("job-ord", Some("nar-ord"), 2);
        let c0 = make_chunk("job-ord", Some("nar-ord"), 0);
        let c1 = make_chunk("job-ord", Some("nar-ord"), 1);

        hg.store_chunk(&c2).unwrap();
        hg.store_chunk(&c0).unwrap();
        hg.store_chunk(&c1).unwrap();

        let chunks = hg.list_chunks_by_job("job-ord").unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[2].chunk_index, 2);
    }
}
