//! Taxonomy registry — predefined + custom tag/genre categories.
//!
//! Provides builtin genre and content type constants plus a KV-backed
//! registry for user-defined custom entries at `tx/{category}/{value}`.

use std::sync::Arc;

use crate::error::{Result, TensaError};
use crate::hypergraph::keys;
use crate::store::KVStore;

use super::types::TaxonomyEntry;

/// Built-in genre values shipped with TENSA.
pub const BUILTIN_GENRES: &[(&str, &str)] = &[
    ("novel", "Novel"),
    ("novella", "Novella"),
    ("short-story", "Short Story"),
    ("investigation", "Investigation"),
    ("geopolitical", "Geopolitical Analysis"),
    ("intelligence-report", "Intelligence Report"),
    ("news-article", "News Article"),
    ("academic-paper", "Academic Paper"),
    ("biography", "Biography"),
    ("memoir", "Memoir"),
    ("essay", "Essay"),
    ("technical", "Technical Document"),
    ("legal", "Legal Document"),
    ("financial", "Financial Report"),
    ("social-media", "Social Media Thread"),
    ("transcript", "Transcript"),
    ("other", "Other"),
];

/// Built-in content type values shipped with TENSA.
pub const BUILTIN_CONTENT_TYPES: &[(&str, &str)] = &[
    ("fiction", "Fiction"),
    ("non-fiction", "Non-Fiction"),
    ("mixed", "Mixed"),
    ("primary-source", "Primary Source"),
    ("secondary-source", "Secondary Source"),
    ("opinion", "Opinion/Editorial"),
    ("analysis", "Analysis"),
    ("raw-data", "Raw Data"),
];

/// Built-in participation role values shipped with TENSA. Mirrors the Rust `Role` enum.
pub const BUILTIN_ROLES: &[(&str, &str)] = &[
    ("protagonist", "Protagonist"),
    ("antagonist", "Antagonist"),
    ("witness", "Witness"),
    ("target", "Target"),
    ("instrument", "Instrument"),
    ("confidant", "Confidant"),
    ("informant", "Informant"),
    ("recipient", "Recipient"),
    ("bystander", "Bystander"),
    ("subject-of-discussion", "Subject of Discussion"),
];

/// Returns the builtin array for a known category, or empty slice.
fn builtins_for(category: &str) -> &'static [(&'static str, &'static str)] {
    match category {
        "genre" => BUILTIN_GENRES,
        "content_type" => BUILTIN_CONTENT_TYPES,
        "role" => BUILTIN_ROLES,
        _ => &[],
    }
}

/// KV-backed taxonomy registry with builtin + custom entries.
pub struct TaxonomyRegistry {
    store: Arc<dyn KVStore>,
}

impl TaxonomyRegistry {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// List all entries for a category (builtins first, then custom).
    pub fn list(&self, category: &str) -> Result<Vec<TaxonomyEntry>> {
        let builtins = builtins_for(category);
        let mut entries: Vec<TaxonomyEntry> = builtins
            .iter()
            .map(|(value, label)| TaxonomyEntry {
                category: category.to_string(),
                value: value.to_string(),
                label: label.to_string(),
                description: None,
                is_builtin: true,
            })
            .collect();

        // Append custom entries from KV
        let prefix = keys::taxonomy_prefix(category);
        let pairs = self.store.prefix_scan(&prefix)?;
        for (_key, value) in pairs {
            let entry: TaxonomyEntry = serde_json::from_slice(&value)?;
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Add a custom taxonomy entry. Rejects if it shadows a builtin value.
    pub fn add(&self, entry: TaxonomyEntry) -> Result<()> {
        let builtins = builtins_for(&entry.category);
        if builtins.iter().any(|(v, _)| *v == entry.value) {
            return Err(TensaError::TaxonomyEntryExists(entry.category, entry.value));
        }
        let key = keys::taxonomy_key(&entry.category, &entry.value);
        if self.store.get(&key)?.is_some() {
            return Err(TensaError::TaxonomyEntryExists(entry.category, entry.value));
        }
        let bytes = serde_json::to_vec(&entry)?;
        self.store.put(&key, &bytes)
    }

    /// Remove a custom taxonomy entry. Cannot remove builtins.
    pub fn remove(&self, category: &str, value: &str) -> Result<()> {
        let builtins = builtins_for(category);
        if builtins.iter().any(|(v, _)| *v == value) {
            return Err(TensaError::TaxonomyBuiltinRemoval(
                category.to_string(),
                value.to_string(),
            ));
        }
        let key = keys::taxonomy_key(category, value);
        self.store.delete(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    #[test]
    fn test_list_builtin_genres() {
        let reg = TaxonomyRegistry::new(test_store());
        let genres = reg.list("genre").unwrap();
        assert_eq!(genres.len(), BUILTIN_GENRES.len());
        assert!(genres[0].is_builtin);
        assert_eq!(genres[0].value, "novel");
        assert_eq!(genres[0].label, "Novel");
    }

    #[test]
    fn test_list_builtin_content_types() {
        let reg = TaxonomyRegistry::new(test_store());
        let types = reg.list("content_type").unwrap();
        assert_eq!(types.len(), BUILTIN_CONTENT_TYPES.len());
        assert!(types[0].is_builtin);
    }

    #[test]
    fn test_list_unknown_category_empty() {
        let reg = TaxonomyRegistry::new(test_store());
        let entries = reg.list("unknown_category").unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_add_custom_entry() {
        let reg = TaxonomyRegistry::new(test_store());
        let entry = TaxonomyEntry {
            category: "genre".to_string(),
            value: "graphic-novel".to_string(),
            label: "Graphic Novel".to_string(),
            description: Some("Visual narrative format".to_string()),
            is_builtin: false,
        };
        reg.add(entry).unwrap();
        let genres = reg.list("genre").unwrap();
        assert_eq!(genres.len(), BUILTIN_GENRES.len() + 1);
        let custom = genres.last().unwrap();
        assert_eq!(custom.value, "graphic-novel");
        assert!(!custom.is_builtin);
    }

    #[test]
    fn test_add_duplicate_custom_entry() {
        let reg = TaxonomyRegistry::new(test_store());
        let entry = TaxonomyEntry {
            category: "genre".to_string(),
            value: "podcast".to_string(),
            label: "Podcast".to_string(),
            description: None,
            is_builtin: false,
        };
        reg.add(entry.clone()).unwrap();
        let result = reg.add(entry);
        assert!(matches!(result, Err(TensaError::TaxonomyEntryExists(_, _))));
    }

    #[test]
    fn test_add_shadowing_builtin_rejected() {
        let reg = TaxonomyRegistry::new(test_store());
        let entry = TaxonomyEntry {
            category: "genre".to_string(),
            value: "novel".to_string(), // builtin
            label: "Custom Novel".to_string(),
            description: None,
            is_builtin: false,
        };
        let result = reg.add(entry);
        assert!(matches!(result, Err(TensaError::TaxonomyEntryExists(_, _))));
    }

    #[test]
    fn test_remove_custom_entry() {
        let reg = TaxonomyRegistry::new(test_store());
        let entry = TaxonomyEntry {
            category: "genre".to_string(),
            value: "podcast".to_string(),
            label: "Podcast".to_string(),
            description: None,
            is_builtin: false,
        };
        reg.add(entry).unwrap();
        assert_eq!(reg.list("genre").unwrap().len(), BUILTIN_GENRES.len() + 1);
        reg.remove("genre", "podcast").unwrap();
        assert_eq!(reg.list("genre").unwrap().len(), BUILTIN_GENRES.len());
    }

    #[test]
    fn test_remove_builtin_rejected() {
        let reg = TaxonomyRegistry::new(test_store());
        let result = reg.remove("genre", "novel");
        assert!(matches!(
            result,
            Err(TensaError::TaxonomyBuiltinRemoval(_, _))
        ));
    }

    #[test]
    fn test_custom_entries_in_new_category() {
        let reg = TaxonomyRegistry::new(test_store());
        let entry = TaxonomyEntry {
            category: "era".to_string(),
            value: "modern".to_string(),
            label: "Modern Era".to_string(),
            description: None,
            is_builtin: false,
        };
        reg.add(entry).unwrap();
        let eras = reg.list("era").unwrap();
        assert_eq!(eras.len(), 1);
        assert_eq!(eras[0].value, "modern");
    }
}
