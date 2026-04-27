//! Entity resolution and coreference detection.
//!
//! Merges extracted entities that refer to the same real-world entity
//! using a cascading matcher pipeline:
//! - Pass 1: Exact alias match (normalized, case-insensitive)
//! - Pass 2: Fuzzy string match (Jaro-Winkler similarity)
//! - Pass 3: Embedding cosine similarity (if embedder provided)
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! The "best match" reductions on Passes 2 and 3 (`if sim > best_sim`)
//! are the **Gödel t-conorm** (`max`) in disguise — the cascade keeps
//! the strongest candidate and discards the rest. Phase 1 exposes
//! [`EntityResolver::resolve_with_tconorm`] for callers that want to
//! override the default. The existing [`EntityResolver::resolve`] stays
//! bit-identical by threading `TNormKind::Godel` internally. Non-Gödel
//! t-conorms over per-candidate similarity scores are experimental — the
//! Pass-2 / Pass-3 loops combine scores from a single candidate against
//! multiple aliases, not scores across candidates, so a Goguen or
//! Łukasiewicz override will affect the intra-candidate fold, leaving
//! the inter-candidate selection logic untouched.
//!
//! Cites: [klement2000].

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::fuzzy::tnorm::{reduce_tconorm, TNormKind};
use crate::ingestion::embed::{cosine_similarity, EmbeddingProvider};
use crate::ingestion::extraction::ExtractedEntity;
use crate::types::{Entity, EntityType};

/// Result of resolving an extracted entity.
#[derive(Debug, Clone)]
pub enum ResolveResult {
    /// Matched to an existing entity.
    Existing(Uuid),
    /// No match found; this is a new entity.
    New,
}

/// Configuration for the entity resolution cascade.
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Enable exact alias matching (Pass 1). Default: true.
    pub alias_enabled: bool,
    /// Enable fuzzy string matching (Pass 2). Default: true.
    pub fuzzy_enabled: bool,
    /// Enable embedding similarity matching (Pass 3). Default: true.
    pub embedding_enabled: bool,
    /// Jaro-Winkler similarity threshold for fuzzy matching. Default: 0.88.
    pub fuzzy_threshold: f64,
    /// Cosine similarity threshold for embedding matching. Default: 0.85.
    pub embedding_threshold: f64,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            alias_enabled: true,
            fuzzy_enabled: true,
            embedding_enabled: true,
            fuzzy_threshold: 0.88,
            embedding_threshold: 0.85,
        }
    }
}

/// Tracks known entities for coreference resolution.
pub struct EntityResolver {
    known_entities: HashMap<Uuid, EntityRecord>,
    alias_index: HashMap<String, Uuid>,
    config: ResolverConfig,
    /// Backward-compat: direct embedding threshold access.
    pub embedding_threshold: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EntityRecord {
    id: Uuid,
    canonical_name: String,
    aliases: HashSet<String>,
    entity_type: EntityType,
    embedding: Option<Vec<f32>>,
}

impl Default for EntityResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityResolver {
    /// Create a new empty resolver with default config.
    pub fn new() -> Self {
        let config = ResolverConfig::default();
        let threshold = config.embedding_threshold;
        Self {
            known_entities: HashMap::new(),
            alias_index: HashMap::new(),
            config,
            embedding_threshold: threshold,
        }
    }

    /// Create a new resolver with a custom embedding similarity threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        let mut config = ResolverConfig::default();
        config.embedding_threshold = threshold;
        Self {
            known_entities: HashMap::new(),
            alias_index: HashMap::new(),
            config,
            embedding_threshold: threshold,
        }
    }

    /// Create a new resolver with full custom config.
    pub fn with_config(config: ResolverConfig) -> Self {
        let threshold = config.embedding_threshold;
        Self {
            known_entities: HashMap::new(),
            alias_index: HashMap::new(),
            config,
            embedding_threshold: threshold,
        }
    }

    /// Bootstrap the resolver from existing entities in the store.
    ///
    /// Extracts `name` and optional `aliases` from each entity's `properties`
    /// and registers them. Entities with no usable name (unnamed, numeric-only,
    /// empty) are skipped so a raw UUID doesn't get indexed as an alias.
    /// Returns the number of entities actually registered.
    pub fn bootstrap_from_entities(&mut self, entities: &[Entity]) -> usize {
        let mut count = 0;
        for ent in entities {
            if ent.deleted_at.is_some() {
                continue;
            }
            let name = match ent.properties.get("name").and_then(|v| v.as_str()) {
                Some(s) if !s.trim().is_empty() => s.to_string(),
                _ => continue,
            };
            let aliases: Vec<String> = ent
                .properties
                .get("aliases")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .filter(|s| !s.trim().is_empty())
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default();
            self.register(
                ent.id,
                &name,
                &aliases,
                ent.entity_type.clone(),
                ent.embedding.clone(),
            );
            count += 1;
        }
        count
    }

    /// Register a known entity with its name and aliases.
    pub fn register(
        &mut self,
        id: Uuid,
        name: &str,
        aliases: &[String],
        entity_type: EntityType,
        embedding: Option<Vec<f32>>,
    ) {
        let normalized = normalize_name(name);
        self.alias_index.insert(normalized.clone(), id);

        let mut alias_set = HashSet::new();
        alias_set.insert(normalized);
        for alias in aliases {
            let norm = normalize_name(alias);
            self.alias_index.insert(norm.clone(), id);
            alias_set.insert(norm);
        }

        self.known_entities.insert(
            id,
            EntityRecord {
                id,
                canonical_name: name.to_string(),
                aliases: alias_set,
                entity_type,
                embedding,
            },
        );
    }

    /// Extend an already-registered entity with additional aliases discovered
    /// on a later mention. Returns the set of aliases that were newly added
    /// (normalized form), so callers can persist only genuinely new ones.
    /// Unknown ids are silently ignored.
    pub fn extend_known_aliases(&mut self, id: Uuid, new_aliases: &[String]) -> Vec<String> {
        let mut added = Vec::new();
        if let Some(record) = self.known_entities.get_mut(&id) {
            for alias in new_aliases {
                let norm = normalize_name(alias);
                if norm.is_empty() {
                    continue;
                }
                if record.aliases.insert(norm.clone()) {
                    self.alias_index.insert(norm.clone(), id);
                    added.push(norm);
                }
            }
        }
        added
    }

    /// Resolve an extracted entity against known entities.
    ///
    /// Pass 1: Exact alias match (normalized).
    /// Pass 2: Fuzzy string match (Jaro-Winkler, same entity type).
    /// Pass 3: Embedding cosine similarity (if embedder provided, same entity type).
    ///
    /// Default t-conorm is Gödel (`max`), which is the pre-sprint
    /// behaviour. Use [`Self::resolve_with_tconorm`] to override the
    /// per-candidate alias-score fold (experimental).
    pub fn resolve(
        &self,
        extracted: &ExtractedEntity,
        embedder: Option<&dyn EmbeddingProvider>,
    ) -> ResolveResult {
        self.resolve_with_tconorm(extracted, embedder, TNormKind::Godel)
    }

    /// Variant of [`Self::resolve`] that folds per-candidate alias scores
    /// under an explicit t-conorm. The inter-candidate selection remains
    /// Gödel (`argmax`) — this parameter only affects the intra-candidate
    /// fold across `(canonical_name, alias_1, alias_2, …)` similarity
    /// scores.
    pub fn resolve_with_tconorm(
        &self,
        extracted: &ExtractedEntity,
        embedder: Option<&dyn EmbeddingProvider>,
        alias_fold: TNormKind,
    ) -> ResolveResult {
        // Pass 1: Exact alias match
        if self.config.alias_enabled {
            let normalized = normalize_name(&extracted.name);
            if let Some(&id) = self.alias_index.get(&normalized) {
                return ResolveResult::Existing(id);
            }
            for alias in &extracted.aliases {
                let norm = normalize_name(alias);
                if let Some(&id) = self.alias_index.get(&norm) {
                    return ResolveResult::Existing(id);
                }
            }
        }

        // Pass 2: Fuzzy string match (Jaro-Winkler)
        if self.config.fuzzy_enabled {
            let query_norm = normalize_name(&extracted.name);
            let mut best_match: Option<(Uuid, f64)> = None;

            for record in self.known_entities.values() {
                // Only match same entity type for fuzzy matching
                if record.entity_type != extracted.entity_type {
                    continue;
                }
                // Collect all per-alias similarities then fold under the
                // chosen t-conorm. Default Gödel reproduces the original
                // "take the max" behaviour.
                let canonical_norm = normalize_name(&record.canonical_name);
                let mut sims: Vec<f64> =
                    Vec::with_capacity(1 + record.aliases.len());
                sims.push(strsim::jaro_winkler(&query_norm, &canonical_norm));
                for alias in &record.aliases {
                    sims.push(strsim::jaro_winkler(&query_norm, alias));
                }
                let best_sim = reduce_tconorm(alias_fold, &sims);

                if best_sim >= self.config.fuzzy_threshold
                    && best_match.map(|(_, prev)| best_sim > prev).unwrap_or(true)
                {
                    best_match = Some((record.id, best_sim));
                }
            }

            if let Some((id, _)) = best_match {
                return ResolveResult::Existing(id);
            }
        }

        // Pass 3: Embedding similarity — single cosine per candidate so
        // the t-conorm choice is moot here (one-element fold == identity).
        if self.config.embedding_enabled {
            if let Some(embedder) = embedder {
                if let Ok(query_emb) = embedder.embed_text(&extracted.name) {
                    let mut best_match: Option<(Uuid, f32)> = None;
                    for record in self.known_entities.values() {
                        if record.entity_type != extracted.entity_type {
                            continue;
                        }
                        if let Some(ref emb) = record.embedding {
                            let sim = cosine_similarity(&query_emb, emb);
                            if sim > self.config.embedding_threshold as f32
                                && best_match.map(|(_, p)| sim > p).unwrap_or(true)
                            {
                                best_match = Some((record.id, sim));
                            }
                        }
                    }
                    if let Some((id, _)) = best_match {
                        return ResolveResult::Existing(id);
                    }
                }
            }
        }

        ResolveResult::New
    }

    /// Number of known entities.
    pub fn len(&self) -> usize {
        self.known_entities.len()
    }

    /// Whether the resolver has any entities.
    pub fn is_empty(&self) -> bool {
        self.known_entities.is_empty()
    }
}

/// Normalize an entity name for matching: lowercase, strip articles, trim.
/// Used by both the resolver's alias index and the ingestion pipeline's
/// property-merge dedup, so the two stay consistent ("The Count" and "Count"
/// collapse to the same key in both).
pub(crate) fn normalize_name(name: &str) -> String {
    let lower = name.trim().to_lowercase();
    // Strip common articles
    let stripped = lower
        .strip_prefix("the ")
        .or_else(|| lower.strip_prefix("a "))
        .or_else(|| lower.strip_prefix("an "))
        .unwrap_or(&lower);
    stripped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::embed::HashEmbedding;
    use chrono::Utc;

    fn make_extracted(name: &str, aliases: Vec<&str>) -> ExtractedEntity {
        ExtractedEntity {
            name: name.to_string(),
            aliases: aliases.into_iter().map(String::from).collect(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({}),
            confidence: 0.9,
        }
    }

    #[test]
    fn test_resolve_exact_match() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(
            id,
            "Raskolnikov",
            &["Rodion".to_string(), "Rodya".to_string()],
            EntityType::Actor,
            None,
        );

        let extracted = make_extracted("Raskolnikov", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected existing match"),
        }
    }

    #[test]
    fn test_resolve_alias_match() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(
            id,
            "Raskolnikov",
            &["Rodion".to_string(), "Rodya".to_string()],
            EntityType::Actor,
            None,
        );

        let extracted = make_extracted("Rodya", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected alias match"),
        }
    }

    #[test]
    fn test_resolve_case_insensitive() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(id, "Raskolnikov", &[], EntityType::Actor, None);

        let extracted = make_extracted("raskolnikov", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected case-insensitive match"),
        }
    }

    #[test]
    fn test_resolve_new_entity() {
        let resolver = EntityResolver::new();
        let extracted = make_extracted("NewCharacter", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::New => {} // Expected
            ResolveResult::Existing(_) => panic!("Expected new entity"),
        }
    }

    #[test]
    fn test_resolve_via_extracted_alias() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(id, "Sonya", &[], EntityType::Actor, None);

        let extracted = make_extracted("Sophia", vec!["Sonya"]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected alias match from extracted aliases"),
        }
    }

    #[test]
    fn test_resolve_strips_articles() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(id, "Old Woman", &[], EntityType::Actor, None);

        let extracted = make_extracted("The Old Woman", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected article-stripped match"),
        }
    }

    #[test]
    fn test_resolve_embedding_similarity() {
        let embedder = HashEmbedding::new(64);
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        let emb = embedder.embed_text("Raskolnikov").unwrap();
        resolver.register(id, "Raskolnikov", &[], EntityType::Actor, Some(emb));

        // Same text should produce identical embedding → similarity = 1.0
        let extracted = make_extracted("Raskolnikov", vec![]);
        match resolver.resolve(&extracted, Some(&embedder)) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected embedding match"),
        }
    }

    #[test]
    fn test_resolve_different_type_no_match() {
        let embedder = HashEmbedding::new(64);
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        let emb = embedder.embed_text("Petersburg").unwrap();
        resolver.register(id, "Petersburg", &[], EntityType::Location, Some(emb));

        // Same name but different entity type
        let mut extracted = make_extracted("Petersburg", vec![]);
        extracted.entity_type = EntityType::Actor;
        // Alias match still works regardless of type (Pass 1)
        match resolver.resolve(&extracted, Some(&embedder)) {
            ResolveResult::Existing(_) => {} // Alias match wins
            ResolveResult::New => panic!("Expected alias match"),
        }
    }

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("  The King  "), "king");
        assert_eq!(normalize_name("A Dog"), "dog");
        assert_eq!(normalize_name("An Apple"), "apple");
        assert_eq!(normalize_name("Raskolnikov"), "raskolnikov");
    }

    // ─── Fuzzy matching tests (Sprint 3) ────────────────────

    #[test]
    fn test_fuzzy_match_typo() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(id, "Raskolnikov", &[], EntityType::Actor, None);

        // Single-char typo: "Raskolnikoff" is close to "Raskolnikov"
        let extracted = make_extracted("Raskolnikoff", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected fuzzy match for typo"),
        }
    }

    #[test]
    fn test_fuzzy_match_transliteration() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(id, "Dostoyevsky", &[], EntityType::Actor, None);

        // Alternate transliteration
        let extracted = make_extracted("Dostoevsky", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected fuzzy match for transliteration"),
        }
    }

    #[test]
    fn test_fuzzy_no_cross_type_match() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(id, "Springfield", &[], EntityType::Location, None);

        // Disable alias matching to test fuzzy isolation
        let config = ResolverConfig {
            alias_enabled: false,
            fuzzy_enabled: true,
            embedding_enabled: false,
            fuzzy_threshold: 0.88,
            embedding_threshold: 0.85,
        };
        let mut resolver2 = EntityResolver::with_config(config);
        resolver2.register(id, "Springfield", &[], EntityType::Location, None);

        let mut extracted = make_extracted("Springfeld", vec![]);
        extracted.entity_type = EntityType::Actor;
        // Fuzzy match should NOT fire across entity types
        match resolver2.resolve(&extracted, None) {
            ResolveResult::New => {} // Expected: different entity type
            ResolveResult::Existing(_) => panic!("Should not fuzzy-match across types"),
        }
    }

    #[test]
    fn test_fuzzy_too_different() {
        let mut resolver = EntityResolver::new();
        resolver.register(Uuid::now_v7(), "Raskolnikov", &[], EntityType::Actor, None);

        // Completely different name shouldn't fuzzy match
        let extracted = make_extracted("Porfiry", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::New => {} // Expected
            ResolveResult::Existing(_) => panic!("Should not fuzzy-match very different names"),
        }
    }

    #[test]
    fn test_fuzzy_disabled_config() {
        let config = ResolverConfig {
            alias_enabled: true,
            fuzzy_enabled: false, // disabled
            embedding_enabled: false,
            fuzzy_threshold: 0.88,
            embedding_threshold: 0.85,
        };
        let mut resolver = EntityResolver::with_config(config);
        resolver.register(Uuid::now_v7(), "Raskolnikov", &[], EntityType::Actor, None);

        // Without fuzzy, typo should NOT match (no exact alias)
        let extracted = make_extracted("Raskolnikoff", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::New => {} // Expected: fuzzy disabled
            ResolveResult::Existing(_) => panic!("Fuzzy is disabled, should not match"),
        }
    }

    #[test]
    fn test_fuzzy_custom_threshold() {
        let config = ResolverConfig {
            alias_enabled: false, // disable alias to force fuzzy path
            fuzzy_enabled: true,
            embedding_enabled: false,
            fuzzy_threshold: 0.99, // very strict
            embedding_threshold: 0.85,
        };
        let mut resolver = EntityResolver::with_config(config);
        let id = Uuid::now_v7();
        resolver.register(id, "Raskolnikov", &[], EntityType::Actor, None);

        // With 0.99 threshold, slight typo won't meet it
        let extracted = make_extracted("Raskolnikoff", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::New => {} // Expected: threshold too strict
            ResolveResult::Existing(_) => panic!("Threshold too strict for this typo"),
        }
    }

    #[test]
    fn test_resolver_config_default() {
        let config = ResolverConfig::default();
        assert!(config.alias_enabled);
        assert!(config.fuzzy_enabled);
        assert!(config.embedding_enabled);
        assert!((config.fuzzy_threshold - 0.88).abs() < f64::EPSILON);
        assert!((config.embedding_threshold - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fuzzy_match_against_alias() {
        let mut resolver = EntityResolver::new();
        let id = Uuid::now_v7();
        resolver.register(
            id,
            "Saint Petersburg",
            &["St. Petersburg".to_string()],
            EntityType::Location,
            None,
        );

        // "St Petersburg" (without dot) should fuzzy-match "St. Petersburg"
        let mut extracted = make_extracted("St Petersburg", vec![]);
        extracted.entity_type = EntityType::Location;
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(found_id) => assert_eq!(found_id, id),
            ResolveResult::New => panic!("Expected fuzzy match against alias"),
        }
    }

    // ─── Bootstrap tests ────────────────────────────────────

    fn make_entity(name: &str, aliases: Vec<&str>, kind: EntityType) -> Entity {
        use crate::types::MaturityLevel;
        Entity {
            id: Uuid::now_v7(),
            entity_type: kind,
            properties: serde_json::json!({
                "name": name,
                "aliases": aliases,
            }),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: Some("test".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[test]
    fn test_bootstrap_matches_previously_ingested_name() {
        let ana = make_entity("Ana Stojanović", vec![], EntityType::Actor);
        let ana_id = ana.id;
        let mut resolver = EntityResolver::new();
        let loaded = resolver.bootstrap_from_entities(&[ana]);
        assert_eq!(loaded, 1);

        let extracted = make_extracted("Ana Stojanović", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(id) => assert_eq!(id, ana_id),
            ResolveResult::New => panic!("bootstrap should dedupe cross-run ingests"),
        }
    }

    #[test]
    fn test_bootstrap_skips_unnamed_and_deleted() {
        use crate::types::MaturityLevel;
        let unnamed = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        let mut deleted = make_entity("Deleted", vec![], EntityType::Actor);
        deleted.deleted_at = Some(Utc::now());
        let named = make_entity("Kept", vec![], EntityType::Actor);

        let mut resolver = EntityResolver::new();
        let loaded = resolver.bootstrap_from_entities(&[unnamed, deleted, named]);
        assert_eq!(loaded, 1, "only the named, non-deleted entity should count");
    }

    #[test]
    fn test_bootstrap_loads_aliases() {
        let sonya = make_entity("Sonya", vec!["Sophia"], EntityType::Actor);
        let sonya_id = sonya.id;
        let mut resolver = EntityResolver::new();
        resolver.bootstrap_from_entities(&[sonya]);

        // Resolve by alias should still work
        let extracted = make_extracted("Sophia", vec![]);
        match resolver.resolve(&extracted, None) {
            ResolveResult::Existing(id) => assert_eq!(id, sonya_id),
            ResolveResult::New => panic!("alias from properties should be registered"),
        }
    }
}
