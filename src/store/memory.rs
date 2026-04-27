use std::collections::BTreeMap;
use std::sync::RwLock;

use crate::error::{Result, TensaError};
use crate::store::{KVStore, TxnOp};

/// In-memory KVStore implementation using BTreeMap.
/// Used for unit tests — no I/O, no dependencies, instant.
///
/// Thread-safe via RwLock. Not meant for production use.
pub struct MemoryStore {
    data: RwLock<BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl KVStore for MemoryStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let data = self
            .data
            .read()
            .map_err(|e| TensaError::Store(format!("Lock poisoned: {e}")))?;
        Ok(data.get(key).cloned())
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|e| TensaError::Store(format!("Lock poisoned: {e}")))?;
        data.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|e| TensaError::Store(format!("Lock poisoned: {e}")))?;
        data.remove(key);
        Ok(())
    }

    fn range(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let data = self
            .data
            .read()
            .map_err(|e| TensaError::Store(format!("Lock poisoned: {e}")))?;
        let results = data
            .range(start.to_vec()..end.to_vec())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(results)
    }

    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let data = self
            .data
            .read()
            .map_err(|e| TensaError::Store(format!("Lock poisoned: {e}")))?;

        // Compute the end key for prefix scan:
        // increment the last byte of prefix, handling overflow
        let mut end = prefix.to_vec();
        let mut found_end = false;
        for i in (0..end.len()).rev() {
            if end[i] < 0xFF {
                end[i] += 1;
                end.truncate(i + 1);
                found_end = true;
                break;
            }
        }

        let results = if found_end {
            data.range(prefix.to_vec()..end)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        } else {
            // prefix is all 0xFF bytes — scan to end
            data.range(prefix.to_vec()..)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        Ok(results)
    }

    fn transaction(&self, ops: Vec<TxnOp>) -> Result<()> {
        let mut data = self
            .data
            .write()
            .map_err(|e| TensaError::Store(format!("Lock poisoned: {e}")))?;

        for op in ops {
            match op {
                TxnOp::Put(k, v) => {
                    data.insert(k, v);
                }
                TxnOp::Delete(k) => {
                    data.remove(&k);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_put_and_get() {
        let store = MemoryStore::new();
        store.put(b"key1", b"value1").unwrap();
        let result = store.get(b"key1").unwrap();
        assert_eq!(result, Some(b"value1".to_vec()));
    }

    #[test]
    fn test_memory_get_nonexistent_returns_none() {
        let store = MemoryStore::new();
        let result = store.get(b"nonexistent").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_memory_delete() {
        let store = MemoryStore::new();
        store.put(b"key1", b"value1").unwrap();
        store.delete(b"key1").unwrap();
        let result = store.get(b"key1").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_memory_delete_nonexistent() {
        let store = MemoryStore::new();
        // Should not error
        store.delete(b"nonexistent").unwrap();
    }

    #[test]
    fn test_memory_range_inclusive() {
        let store = MemoryStore::new();
        store.put(b"a", b"1").unwrap();
        store.put(b"b", b"2").unwrap();
        store.put(b"c", b"3").unwrap();
        store.put(b"d", b"4").unwrap();

        let results = store.range(b"b", b"d").unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"b".to_vec());
        assert_eq!(results[1].0, b"c".to_vec());
    }

    #[test]
    fn test_memory_range_empty() {
        let store = MemoryStore::new();
        store.put(b"a", b"1").unwrap();
        let results = store.range(b"x", b"z").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_prefix_scan() {
        let store = MemoryStore::new();
        store.put(b"e/aaa", b"entity_a").unwrap();
        store.put(b"e/bbb", b"entity_b").unwrap();
        store.put(b"s/ccc", b"situation_c").unwrap();

        let results = store.prefix_scan(b"e/").unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"e/aaa".to_vec());
        assert_eq!(results[1].0, b"e/bbb".to_vec());
    }

    #[test]
    fn test_memory_prefix_scan_no_match() {
        let store = MemoryStore::new();
        store.put(b"e/aaa", b"entity_a").unwrap();
        let results = store.prefix_scan(b"x/").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_memory_transaction_commit() {
        let store = MemoryStore::new();
        store
            .transaction(vec![
                TxnOp::Put(b"k1".to_vec(), b"v1".to_vec()),
                TxnOp::Put(b"k2".to_vec(), b"v2".to_vec()),
            ])
            .unwrap();

        assert_eq!(store.get(b"k1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_memory_transaction_with_delete() {
        let store = MemoryStore::new();
        store.put(b"k1", b"v1").unwrap();
        store
            .transaction(vec![
                TxnOp::Put(b"k2".to_vec(), b"v2".to_vec()),
                TxnOp::Delete(b"k1".to_vec()),
            ])
            .unwrap();

        assert_eq!(store.get(b"k1").unwrap(), None);
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_memory_batch_put() {
        let store = MemoryStore::new();
        store
            .batch_put(vec![(b"k1", b"v1"), (b"k2", b"v2"), (b"k3", b"v3")])
            .unwrap();

        assert_eq!(store.get(b"k1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(store.get(b"k3").unwrap(), Some(b"v3".to_vec()));
    }

    #[test]
    fn test_memory_overwrite() {
        let store = MemoryStore::new();
        store.put(b"key", b"old").unwrap();
        store.put(b"key", b"new").unwrap();
        assert_eq!(store.get(b"key").unwrap(), Some(b"new".to_vec()));
    }
}
