pub mod foundationdb;
pub mod memory;
#[cfg(feature = "rocksdb")]
pub mod rocks;
pub mod workspace;

use crate::error::Result;

/// Operation within a transaction
#[derive(Debug, Clone)]
pub enum TxnOp {
    Put(Vec<u8>, Vec<u8>),
    Delete(Vec<u8>),
}

/// Abstract key-value store interface.
///
/// All hypergraph operations go through this trait.
/// Implementations: MemoryStore (tests), RocksDBStore (production),
/// FoundationDBStore (future distributed).
///
/// Keys and values are raw bytes. The hypergraph layer handles
/// serialization (bincode) and key encoding (prefix scheme).
pub trait KVStore: Send + Sync {
    /// Get a value by key. Returns None if key doesn't exist.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Store a key-value pair. Overwrites if key exists.
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;

    /// Delete a key. No-op if key doesn't exist.
    fn delete(&self, key: &[u8]) -> Result<()>;

    /// Range scan: returns all KV pairs where start <= key < end.
    /// Results ordered by key.
    fn range(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Prefix scan: returns all KV pairs where key starts with prefix.
    /// Results ordered by key.
    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Execute a batch of operations atomically.
    /// Either all succeed or none do.
    fn transaction(&self, ops: Vec<TxnOp>) -> Result<()>;

    /// Batch put for performance. Default implementation uses transaction.
    fn batch_put(&self, pairs: Vec<(&[u8], &[u8])>) -> Result<()> {
        let ops = pairs
            .into_iter()
            .map(|(k, v)| TxnOp::Put(k.to_vec(), v.to_vec()))
            .collect();
        self.transaction(ops)
    }
}

/// Decode a list of records from a secondary index of the shape
/// `prefix/{uuid_bytes}` — where the key payload is empty and the actual record
/// lives under some other primary key. Given a `load` closure that fetches the
/// record by uuid, this walks the index, skips malformed entries, and collects
/// whatever `load` succeeds on.
///
/// Used by every writer-facing module that maintains a narrative- or
/// scene-scoped index of uuid-keyed records (research notes, annotations,
/// collections, compile profiles).
pub fn scan_uuid_index<T, F>(store: &dyn KVStore, prefix: &[u8], mut load: F) -> Result<Vec<T>>
where
    F: FnMut(&uuid::Uuid) -> Result<T>,
{
    let entries = store.prefix_scan(prefix)?;
    let mut out = Vec::with_capacity(entries.len());
    for (key, _) in entries {
        if key.len() < prefix.len() + 16 {
            continue;
        }
        let id_bytes: [u8; 16] = match key[prefix.len()..prefix.len() + 16].try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };
        let id = uuid::Uuid::from_bytes(id_bytes);
        if let Ok(rec) = load(&id) {
            out.push(rec);
        }
    }
    Ok(out)
}
