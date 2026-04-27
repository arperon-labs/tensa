#![cfg(feature = "rocksdb")]

use rocksdb::{Options, WriteBatch, DB};
use std::path::Path;

use crate::error::{Result, TensaError};
use crate::store::{KVStore, TxnOp};

/// RocksDB-backed KVStore implementation.
/// Production storage for Phase 0-3.
pub struct RocksDBStore {
    db: DB,
}

impl RocksDBStore {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let db = DB::open(&opts, path).map_err(|e| TensaError::Store(e.to_string()))?;
        Ok(Self { db })
    }
}

impl KVStore for RocksDBStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db
            .get(key)
            .map_err(|e| TensaError::Store(e.to_string()))
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db
            .put(key, value)
            .map_err(|e| TensaError::Store(e.to_string()))
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.db
            .delete(key)
            .map_err(|e| TensaError::Store(e.to_string()))
    }

    fn range(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut results = Vec::new();
        let iter = self.db.iterator(rocksdb::IteratorMode::From(
            start,
            rocksdb::Direction::Forward,
        ));
        for item in iter {
            let (k, v) = item.map_err(|e| TensaError::Store(e.to_string()))?;
            if k.as_ref() >= end {
                break;
            }
            results.push((k.to_vec(), v.to_vec()));
        }
        Ok(results)
    }

    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut results = Vec::new();
        let iter = self.db.iterator(rocksdb::IteratorMode::From(
            prefix,
            rocksdb::Direction::Forward,
        ));
        for item in iter {
            let (k, v) = item.map_err(|e| TensaError::Store(e.to_string()))?;
            if !k.starts_with(prefix) {
                break;
            }
            results.push((k.to_vec(), v.to_vec()));
        }
        Ok(results)
    }

    fn transaction(&self, ops: Vec<TxnOp>) -> Result<()> {
        let mut batch = WriteBatch::default();
        for op in ops {
            match op {
                TxnOp::Put(k, v) => batch.put(&k, &v),
                TxnOp::Delete(k) => batch.delete(&k),
            }
        }
        self.db
            .write(batch)
            .map_err(|e| TensaError::Store(e.to_string()))
    }
}
