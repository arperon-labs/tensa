use std::sync::Arc;

use crate::error::Result;
use crate::store::{KVStore, TxnOp};

/// A KV store wrapper that transparently prefixes all keys with a workspace namespace.
///
/// This enables multi-tenant isolation: data written through one `WorkspaceStore`
/// is invisible when accessed through a different workspace.  The underlying
/// physical store is shared; only the key prefix differs.
///
/// The prefix format is `w/{workspace_id}/`, so a key `e/{uuid}` becomes
/// `w/my-workspace/e/{uuid}` in the backing store.
pub struct WorkspaceStore {
    inner: Arc<dyn KVStore>,
    prefix: Vec<u8>,
}

impl WorkspaceStore {
    /// Create a new workspace-scoped store.
    ///
    /// The `workspace_id` is embedded in the key prefix `w/{workspace_id}/`.
    /// All reads and writes through this store are transparently namespaced.
    pub fn new(inner: Arc<dyn KVStore>, workspace_id: &str) -> Self {
        let prefix = format!("w/{}/", workspace_id).into_bytes();
        Self { inner, prefix }
    }

    /// Return the workspace identifier extracted from the prefix.
    pub fn workspace_id(&self) -> &str {
        // prefix is "w/{id}/" — strip bookends
        let s = std::str::from_utf8(&self.prefix).unwrap_or("");
        s.strip_prefix("w/")
            .and_then(|s| s.strip_suffix("/"))
            .unwrap_or("unknown")
    }

    /// Build the prefixed key by prepending the workspace namespace.
    fn prefixed_key(&self, key: &[u8]) -> Vec<u8> {
        let mut pk = Vec::with_capacity(self.prefix.len() + key.len());
        pk.extend_from_slice(&self.prefix);
        pk.extend_from_slice(key);
        pk
    }

    /// Strip the workspace prefix from a returned key, if present.
    fn strip_prefix(&self, key: Vec<u8>) -> Vec<u8> {
        if key.starts_with(&self.prefix) {
            key[self.prefix.len()..].to_vec()
        } else {
            key
        }
    }
}

impl KVStore for WorkspaceStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.inner.get(&self.prefixed_key(key))
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.inner.put(&self.prefixed_key(key), value)
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.inner.delete(&self.prefixed_key(key))
    }

    fn range(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let pstart = self.prefixed_key(start);
        let pend = self.prefixed_key(end);
        let results = self.inner.range(&pstart, &pend)?;
        Ok(results
            .into_iter()
            .map(|(k, v)| (self.strip_prefix(k), v))
            .collect())
    }

    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let ppfx = self.prefixed_key(prefix);
        let results = self.inner.prefix_scan(&ppfx)?;
        Ok(results
            .into_iter()
            .map(|(k, v)| (self.strip_prefix(k), v))
            .collect())
    }

    fn transaction(&self, ops: Vec<TxnOp>) -> Result<()> {
        let prefixed_ops: Vec<TxnOp> = ops
            .into_iter()
            .map(|op| match op {
                TxnOp::Put(k, v) => TxnOp::Put(self.prefixed_key(&k), v),
                TxnOp::Delete(k) => TxnOp::Delete(self.prefixed_key(&k)),
            })
            .collect();
        self.inner.transaction(prefixed_ops)
    }

    fn batch_put(&self, pairs: Vec<(&[u8], &[u8])>) -> Result<()> {
        let owned: Vec<(Vec<u8>, Vec<u8>)> = pairs
            .into_iter()
            .map(|(k, v)| (self.prefixed_key(k), v.to_vec()))
            .collect();
        let refs: Vec<(&[u8], &[u8])> = owned
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        self.inner.batch_put(refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    #[test]
    fn test_workspace_put_get() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root.clone(), "alpha");

        ws.put(b"e/123", b"entity_data").unwrap();
        let val = ws.get(b"e/123").unwrap();
        assert_eq!(val, Some(b"entity_data".to_vec()));

        // Data is stored prefixed in the root store
        let raw = root.get(b"w/alpha/e/123").unwrap();
        assert_eq!(raw, Some(b"entity_data".to_vec()));
    }

    #[test]
    fn test_workspace_isolation() {
        let root = Arc::new(MemoryStore::new());
        let ws_a = WorkspaceStore::new(root.clone(), "alpha");
        let ws_b = WorkspaceStore::new(root.clone(), "beta");

        ws_a.put(b"e/1", b"alice").unwrap();
        ws_b.put(b"e/1", b"bob").unwrap();

        assert_eq!(ws_a.get(b"e/1").unwrap(), Some(b"alice".to_vec()));
        assert_eq!(ws_b.get(b"e/1").unwrap(), Some(b"bob".to_vec()));

        // Each workspace cannot see the other's data
        let a_entities = ws_a.prefix_scan(b"e/").unwrap();
        assert_eq!(a_entities.len(), 1);
        assert_eq!(a_entities[0].1, b"alice".to_vec());

        let b_entities = ws_b.prefix_scan(b"e/").unwrap();
        assert_eq!(b_entities.len(), 1);
        assert_eq!(b_entities[0].1, b"bob".to_vec());
    }

    #[test]
    fn test_workspace_prefix_scan_strips_keys() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root.clone(), "proj1");

        ws.put(b"e/aaa", b"v1").unwrap();
        ws.put(b"e/bbb", b"v2").unwrap();
        ws.put(b"s/ccc", b"v3").unwrap();

        let entities = ws.prefix_scan(b"e/").unwrap();
        assert_eq!(entities.len(), 2);
        // Keys should be stripped — no workspace prefix visible
        assert_eq!(entities[0].0, b"e/aaa".to_vec());
        assert_eq!(entities[1].0, b"e/bbb".to_vec());
    }

    #[test]
    fn test_workspace_range_strips_keys() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root.clone(), "proj1");

        ws.put(b"e/a", b"1").unwrap();
        ws.put(b"e/b", b"2").unwrap();
        ws.put(b"e/c", b"3").unwrap();
        ws.put(b"e/d", b"4").unwrap();

        let results = ws.range(b"e/b", b"e/d").unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, b"e/b".to_vec());
        assert_eq!(results[1].0, b"e/c".to_vec());
    }

    #[test]
    fn test_workspace_delete() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root.clone(), "proj1");

        ws.put(b"e/1", b"data").unwrap();
        assert!(ws.get(b"e/1").unwrap().is_some());

        ws.delete(b"e/1").unwrap();
        assert!(ws.get(b"e/1").unwrap().is_none());

        // Also gone from root
        assert!(root.get(b"w/proj1/e/1").unwrap().is_none());
    }

    #[test]
    fn test_workspace_transaction() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root.clone(), "txn-test");

        ws.transaction(vec![
            TxnOp::Put(b"k1".to_vec(), b"v1".to_vec()),
            TxnOp::Put(b"k2".to_vec(), b"v2".to_vec()),
        ])
        .unwrap();

        assert_eq!(ws.get(b"k1").unwrap(), Some(b"v1".to_vec()));
        assert_eq!(ws.get(b"k2").unwrap(), Some(b"v2".to_vec()));

        // Verify prefixed in root
        assert!(root.get(b"w/txn-test/k1").unwrap().is_some());

        // Transaction with delete
        ws.transaction(vec![TxnOp::Delete(b"k1".to_vec())]).unwrap();
        assert!(ws.get(b"k1").unwrap().is_none());
    }

    #[test]
    fn test_workspace_batch_put() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root.clone(), "batch");

        ws.batch_put(vec![
            (b"a".as_ref(), b"1".as_ref()),
            (b"b".as_ref(), b"2".as_ref()),
            (b"c".as_ref(), b"3".as_ref()),
        ])
        .unwrap();

        assert_eq!(ws.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(ws.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(ws.get(b"c").unwrap(), Some(b"3".to_vec()));

        // Root store has prefixed keys
        assert_eq!(root.get(b"w/batch/a").unwrap(), Some(b"1".to_vec()));
    }

    #[test]
    fn test_workspace_id_accessor() {
        let root = Arc::new(MemoryStore::new());
        let ws = WorkspaceStore::new(root, "my-workspace");
        assert_eq!(ws.workspace_id(), "my-workspace");
    }

    #[test]
    fn test_workspace_hypergraph_integration() {
        // Demonstrate that Hypergraph works seamlessly with WorkspaceStore
        let root = Arc::new(MemoryStore::new());
        let ws_a: Arc<dyn KVStore> = Arc::new(WorkspaceStore::new(root.clone(), "narrative-a"));
        let ws_b: Arc<dyn KVStore> = Arc::new(WorkspaceStore::new(root.clone(), "narrative-b"));

        let hg_a = crate::Hypergraph::new(ws_a);
        let hg_b = crate::Hypergraph::new(ws_b);

        // Create an entity in workspace A
        let entity_a = crate::types::Entity {
            id: uuid::Uuid::now_v7(),
            entity_type: crate::types::EntityType::Actor,
            properties: serde_json::json!({"name": "Alice"}),
            beliefs: None,
            embedding: None,
            narrative_id: None,
            maturity: crate::types::MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: None,
            provenance: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let id_a = hg_a.create_entity(entity_a).unwrap();

        // Workspace B cannot see it
        assert!(hg_b.get_entity(&id_a).is_err());

        // Workspace A can
        let fetched = hg_a.get_entity(&id_a).unwrap();
        assert_eq!(
            fetched.properties.get("name").and_then(|v| v.as_str()),
            Some("Alice")
        );
    }

    #[test]
    fn test_workspace_meta_serialization() {
        // WorkspaceMeta round-trip (mirrors the API struct)
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct WorkspaceMeta {
            id: String,
            name: String,
            created_at: chrono::DateTime<chrono::Utc>,
        }

        let meta = WorkspaceMeta {
            id: "test-ws".into(),
            name: "Test Workspace".into(),
            created_at: chrono::Utc::now(),
        };

        let bytes = serde_json::to_vec(&meta).unwrap();
        let decoded: WorkspaceMeta = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.id, meta.id);
        assert_eq!(decoded.name, meta.name);
    }
}
