//! Project registry — KV-backed CRUD for project metadata.
//!
//! Projects are stored at `pj/{project_id}` in the KV store.
//! A secondary index `pn/{project_id}/{narrative_id}` tracks
//! which narratives belong to each project.

use std::sync::Arc;

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::hypergraph::keys;
use crate::store::KVStore;

use super::types::Project;

/// Registry for managing project metadata in the KV store.
pub struct ProjectRegistry {
    store: Arc<dyn KVStore>,
}

impl ProjectRegistry {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Create a new project. Returns error if the ID is taken.
    pub fn create(&self, project: Project) -> Result<String> {
        let key = keys::project_key(&project.id);
        if self.store.get(&key)?.is_some() {
            return Err(TensaError::Internal(format!(
                "Project already exists: {}",
                project.id
            )));
        }
        let bytes = serde_json::to_vec(&project)?;
        self.store.put(&key, &bytes)?;
        Ok(project.id)
    }

    /// Get a project by ID.
    pub fn get(&self, id: &str) -> Result<Project> {
        let key = keys::project_key(id);
        match self.store.get(&key)? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Err(TensaError::NotFound(format!("Project not found: {}", id))),
        }
    }

    /// Update a project by applying a closure. Automatically updates `updated_at`.
    pub fn update(&self, id: &str, updater: impl FnOnce(&mut Project)) -> Result<Project> {
        let mut project = self.get(id)?;
        updater(&mut project);
        project.updated_at = Utc::now();
        let key = keys::project_key(id);
        let bytes = serde_json::to_vec(&project)?;
        self.store.put(&key, &bytes)?;
        Ok(project)
    }

    /// Delete a project by ID.
    pub fn delete(&self, id: &str) -> Result<()> {
        self.get(id)?; // verify existence
                       // Remove project-narrative index entries
        let prefix = keys::project_narrative_index_prefix(id);
        let pairs = self.store.prefix_scan(&prefix)?;
        for (key, _) in pairs {
            self.store.delete(&key)?;
        }
        self.store.delete(&keys::project_key(id))
    }

    /// List all projects.
    pub fn list(&self) -> Result<Vec<Project>> {
        let prefix = keys::project_prefix();
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            // Skip project-narrative index keys (pn/ would be separate prefix)
            if let Ok(p) = serde_json::from_slice::<Project>(&value) {
                result.push(p);
            }
        }
        Ok(result)
    }

    /// List projects with cursor-based pagination.
    pub fn list_paginated(
        &self,
        limit: usize,
        after: Option<&str>,
    ) -> Result<(Vec<Project>, Option<String>)> {
        let prefix = keys::project_prefix();
        let start = match after {
            Some(cursor) => {
                let mut k = keys::project_key(cursor);
                k.push(0);
                k
            }
            None => prefix.clone(),
        };
        let mut end = prefix;
        end.push(0xFF);

        let pairs = self.store.range(&start, &end)?;
        let mut result = Vec::with_capacity(limit + 1);
        for (_key, value) in pairs.iter().take(limit + 1) {
            if let Ok(project) = serde_json::from_slice::<Project>(value) {
                result.push(project);
            }
        }

        let next_cursor = if result.len() > limit {
            result.pop();
            result.last().map(|p| p.id.clone())
        } else {
            None
        };

        Ok((result, next_cursor))
    }

    /// Add a narrative to a project (writes index key).
    pub fn add_narrative(&self, project_id: &str, narrative_id: &str) -> Result<()> {
        self.get(project_id)?; // verify project exists
        let key = keys::project_narrative_index_key(project_id, narrative_id);
        self.store.put(&key, &[])?;
        Ok(())
    }

    /// Remove a narrative from a project (deletes index key).
    pub fn remove_narrative(&self, project_id: &str, narrative_id: &str) -> Result<()> {
        let key = keys::project_narrative_index_key(project_id, narrative_id);
        self.store.delete(&key)
    }

    /// List all narrative IDs belonging to a project.
    pub fn list_narrative_ids(&self, project_id: &str) -> Result<Vec<String>> {
        let prefix = keys::project_narrative_index_prefix(project_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let prefix_len = prefix.len();
        let mut ids = Vec::new();
        for (key, _) in pairs {
            if key.len() > prefix_len {
                if let Ok(s) = std::str::from_utf8(&key[prefix_len..]) {
                    ids.push(s.to_string());
                }
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn setup() -> ProjectRegistry {
        ProjectRegistry::new(Arc::new(MemoryStore::new()))
    }

    fn make_project(id: &str, title: &str) -> Project {
        Project {
            id: id.into(),
            title: title.into(),
            description: None,
            tags: vec![],
            narrative_count: 0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn test_project_crud() {
        let reg = setup();
        reg.create(make_project("geo", "Geopolitics")).unwrap();

        let p = reg.get("geo").unwrap();
        assert_eq!(p.title, "Geopolitics");

        reg.update("geo", |p| p.description = Some("World events".into()))
            .unwrap();
        assert_eq!(
            reg.get("geo").unwrap().description,
            Some("World events".into())
        );

        assert_eq!(reg.list().unwrap().len(), 1);

        reg.delete("geo").unwrap();
        assert!(reg.get("geo").is_err());
    }

    #[test]
    fn test_project_duplicate_rejected() {
        let reg = setup();
        reg.create(make_project("a", "A")).unwrap();
        assert!(reg.create(make_project("a", "A2")).is_err());
    }

    #[test]
    fn test_project_narrative_index() {
        let reg = setup();
        reg.create(make_project("geo", "Geopolitics")).unwrap();

        reg.add_narrative("geo", "ukraine").unwrap();
        reg.add_narrative("geo", "middle-east").unwrap();

        let nids = reg.list_narrative_ids("geo").unwrap();
        assert_eq!(nids.len(), 2);
        assert!(nids.contains(&"ukraine".to_string()));
        assert!(nids.contains(&"middle-east".to_string()));

        reg.remove_narrative("geo", "ukraine").unwrap();
        assert_eq!(reg.list_narrative_ids("geo").unwrap().len(), 1);
    }

    #[test]
    fn test_project_delete_cleans_index() {
        let reg = setup();
        reg.create(make_project("x", "X")).unwrap();
        reg.add_narrative("x", "n1").unwrap();
        reg.add_narrative("x", "n2").unwrap();
        reg.delete("x").unwrap();
        // Index should be cleaned up (no orphaned pn/ keys)
    }
}
