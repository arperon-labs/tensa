//! Entity deduplication within a narrative.
//!
//! Proposes candidate merges between entities in the same narrative using
//! a composite score of name similarity, alias overlap, and participation
//! (co-occurrence) overlap. Returns a list for human review — never merges
//! automatically.
//!
//! Typical use: after merging narrative A into narrative B via
//! `NarrativeRegistry::merge_narratives`, call this to find duplicate
//! actors ("Alex Murdaugh" vs "Alexander Murdaugh") that the merge
//! couldn't collapse on its own.

use std::collections::HashSet;
use std::hash::Hash;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{Entity, EntityType};

/// A proposed merge between two entities in the same narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupCandidate {
    /// Suggested entity to keep (the one with more participations, or
    /// higher confidence as tiebreaker). The UI may override this.
    pub keep_id: Uuid,
    /// Suggested entity to absorb into `keep_id`.
    pub absorb_id: Uuid,
    /// Canonical name of `keep_id` at the time the candidate was proposed.
    pub keep_name: String,
    /// Canonical name of `absorb_id` at the time the candidate was proposed.
    pub absorb_name: String,
    /// Entity type (both entities have the same type).
    pub entity_type: EntityType,
    /// Composite score in [0.0, 1.0]. Higher = more likely duplicate.
    pub score: f32,
    /// Individual signal scores for transparency.
    pub breakdown: DedupBreakdown,
    /// Number of situations `keep_id` participates in.
    pub keep_participation_count: usize,
    /// Number of situations `absorb_id` participates in.
    pub absorb_participation_count: usize,
}

/// Decomposed dedup score showing each signal's contribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupBreakdown {
    /// Jaro-Winkler similarity between normalized canonical names, 0.0-1.0.
    pub name_similarity: f32,
    /// Size of the alias set intersection divided by the size of the smaller set, 0.0-1.0.
    /// Canonical names are included in each entity's alias set for this computation.
    pub alias_overlap: f32,
    /// Size of the participation (situation) intersection divided by the size of the smaller set, 0.0-1.0.
    /// 0.0 when either entity has no participations.
    pub participation_overlap: f32,
}

/// Options for a dedup scan.
#[derive(Debug, Clone, Deserialize)]
pub struct DedupOptions {
    /// Minimum composite score for a pair to be reported. Default 0.7.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Maximum number of candidates to return. Default 200.
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,
    /// When true, only consider entities of these types. Empty = all types.
    #[serde(default)]
    pub entity_types: Vec<EntityType>,
}

fn default_threshold() -> f32 {
    0.7
}

fn default_max_candidates() -> usize {
    200
}

impl Default for DedupOptions {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            max_candidates: default_max_candidates(),
            entity_types: Vec::new(),
        }
    }
}

/// Scan a narrative for duplicate entity pairs.
///
/// Buckets entities by type first so pairs are localized (Actors never
/// match Locations) and O(n²) cost is Σ kᵢ²/2 over type groups instead
/// of the full n². Within each bucket scores every pair; returns
/// candidates with score >= threshold, sorted by score descending.
pub fn find_duplicate_candidates(
    hg: &Hypergraph,
    narrative_id: &str,
    opts: &DedupOptions,
) -> Result<Vec<DedupCandidate>> {
    let all_entities = hg.list_entities_by_narrative(narrative_id)?;

    // Filter by type if requested, then bucket by entity_type so we only
    // ever pair entities of the same kind.
    let mut buckets: std::collections::HashMap<EntityType, Vec<Entity>> =
        std::collections::HashMap::new();
    for e in all_entities {
        if !opts.entity_types.is_empty() && !opts.entity_types.contains(&e.entity_type) {
            continue;
        }
        buckets.entry(e.entity_type.clone()).or_default().push(e);
    }

    let mut candidates: Vec<DedupCandidate> = Vec::new();

    for (_type, entities) in buckets {
        let features = compute_features(hg, &entities);
        score_pairs(&features, opts.threshold, &mut candidates);
    }

    // Sort by score descending, then truncate to max_candidates
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(opts.max_candidates);

    Ok(candidates)
}

/// Pre-computed per-entity features for pair scoring. The `norm_chars`
/// field avoids re-allocating `Vec<char>` inside Jaro-Winkler on every
/// pair comparison — this is the dominant cost on large narratives.
struct EntityFeatures<'a> {
    entity: &'a Entity,
    norm_name: String,
    norm_chars: Vec<char>,
    alias_set: HashSet<String>,
    situations: HashSet<Uuid>,
}

impl<'a> EntityFeatures<'a> {
    fn participation_count(&self) -> usize {
        self.situations.len()
    }

    fn name(&self) -> &str {
        self.entity
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("(unnamed)")
    }
}

fn compute_features<'a>(hg: &Hypergraph, entities: &'a [Entity]) -> Vec<EntityFeatures<'a>> {
    let mut features: Vec<EntityFeatures> = Vec::with_capacity(entities.len());
    for e in entities {
        let name = e
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let norm_name = normalize_name(name);
        let norm_chars: Vec<char> = norm_name.chars().collect();

        let mut alias_set: HashSet<String> = HashSet::new();
        alias_set.insert(norm_name.clone());
        if let Some(aliases) = e.properties.get("aliases").and_then(|v| v.as_array()) {
            for a in aliases {
                if let Some(s) = a.as_str() {
                    alias_set.insert(normalize_name(s));
                }
            }
        }

        let parts = hg.get_situations_for_entity(&e.id).unwrap_or_default();
        let situations: HashSet<Uuid> = parts.into_iter().map(|p| p.situation_id).collect();

        features.push(EntityFeatures {
            entity: e,
            norm_name,
            norm_chars,
            alias_set,
            situations,
        });
    }
    features
}

fn score_pairs(features: &[EntityFeatures], threshold: f32, candidates: &mut Vec<DedupCandidate>) {
    for i in 0..features.len() {
        for j in (i + 1)..features.len() {
            let a = &features[i];
            let b = &features[j];

            if a.norm_name.is_empty() || b.norm_name.is_empty() {
                continue;
            }

            let name_sim = jaro_winkler(&a.norm_chars, &b.norm_chars);
            let alias_ovl = set_overlap(&a.alias_set, &b.alias_set);
            let part_ovl = if a.situations.is_empty() || b.situations.is_empty() {
                0.0
            } else {
                set_overlap(&a.situations, &b.situations)
            };

            let score = composite_score(name_sim, alias_ovl, part_ovl);

            if score < threshold {
                continue;
            }

            // Pick keep vs absorb: more participations wins, break ties by
            // confidence, break further ties by UUID ordering so the result
            // is deterministic.
            let (keep, absorb) = if a.participation_count() > b.participation_count() {
                (a, b)
            } else if a.participation_count() < b.participation_count() {
                (b, a)
            } else if a.entity.confidence >= b.entity.confidence {
                (a, b)
            } else {
                (b, a)
            };

            candidates.push(DedupCandidate {
                keep_id: keep.entity.id,
                absorb_id: absorb.entity.id,
                keep_name: keep.name().to_string(),
                absorb_name: absorb.name().to_string(),
                entity_type: a.entity.entity_type.clone(),
                score,
                breakdown: DedupBreakdown {
                    name_similarity: name_sim,
                    alias_overlap: alias_ovl,
                    participation_overlap: part_ovl,
                },
                keep_participation_count: keep.participation_count(),
                absorb_participation_count: absorb.participation_count(),
            });
        }
    }
}

/// Composite dedup score, in [0.0, 1.0].
///
/// Two sub-scores are computed and the larger wins:
///
/// 1. **Context-weighted name score**: `name_sim * (0.8 + 0.2 * max(alias, part))`.
///    Strong name similarity alone is enough to flag a candidate; supporting
///    context (shared aliases or shared situations) lifts the score further.
///
/// 2. **Alias-dominant score**: `0.5 + 0.4 * alias_overlap` (0.0 if no overlap).
///    High alias overlap alone is a strong signal even when canonical names
///    differ — catches cases like A.name="Reg" with alias "Professor Chronotis"
///    vs B.name="Professor Chronotis" with alias "Reg".
fn composite_score(name_sim: f32, alias_overlap: f32, participation_overlap: f32) -> f32 {
    let context_boost = 0.8 + 0.2 * alias_overlap.max(participation_overlap);
    let context_weighted = name_sim * context_boost;
    let alias_only = if alias_overlap > 0.0 {
        0.5 + 0.4 * alias_overlap
    } else {
        0.0
    };
    context_weighted.max(alias_only)
}

/// Normalize an entity name for comparison: lowercase, trim, strip articles,
/// and collapse repeated whitespace to a single space. Matches the logic in
/// `ingestion::resolve::normalize_name` but is reimplemented here to avoid
/// a cross-module dependency on the ingestion layer.
fn normalize_name(name: &str) -> String {
    let lower: String = name
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let stripped = lower
        .strip_prefix("the ")
        .or_else(|| lower.strip_prefix("a "))
        .or_else(|| lower.strip_prefix("an "))
        .unwrap_or(&lower);
    stripped.to_string()
}

/// Jaro-Winkler similarity in [0.0, 1.0]. Higher = more similar.
///
/// Takes char slices so callers can pre-compute `Vec<char>` once per
/// entity and pass references during pair comparison — avoids O(n²)
/// `Vec<char>` allocations on large narratives.
fn jaro_winkler(a: &[char], b: &[char]) -> f32 {
    let jaro = jaro_similarity(a, b);
    if jaro < 0.7 {
        return jaro;
    }

    // Winkler boost: count common prefix (max 4 chars)
    let prefix_len = a
        .iter()
        .zip(b.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count();

    jaro + (prefix_len as f32) * 0.1 * (1.0 - jaro)
}

/// Jaro similarity in [0.0, 1.0]. Building block for Jaro-Winkler.
fn jaro_similarity(a: &[char], b: &[char]) -> f32 {
    if a == b {
        return 1.0;
    }
    let a_len = a.len();
    let b_len = b.len();
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    let match_window = (a_len.max(b_len) / 2).saturating_sub(1).max(1);

    let mut a_matches = vec![false; a_len];
    let mut b_matches = vec![false; b_len];

    let mut matches = 0usize;
    for i in 0..a_len {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(b_len);
        for j in start..end {
            if b_matches[j] {
                continue;
            }
            if a[i] != b[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions (half the number of out-of-order matching pairs)
    let mut k = 0usize;
    let mut transpositions = 0usize;
    for i in 0..a_len {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a[i] != b[k] {
            transpositions += 1;
        }
        k += 1;
    }
    let transpositions = transpositions / 2;

    let m = matches as f32;
    ((m / a_len as f32) + (m / b_len as f32) + ((m - transpositions as f32) / m)) / 3.0
}

/// Compute |A ∩ B| / min(|A|, |B|). Returns 0.0 if either set is empty.
fn set_overlap<T: Eq + Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f32 {
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if smaller.is_empty() {
        return 0.0;
    }
    let intersection_size = smaller.iter().filter(|x| larger.contains(*x)).count();
    intersection_size as f32 / smaller.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn make_actor(name: &str, aliases: &[&str], narrative_id: &str) -> Entity {
        Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
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
            extraction_method: Some(ExtractionMethod::LlmParsed),
            narrative_id: Some(narrative_id.to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn make_situation(narrative_id: &str) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: None,
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: Some(narrative_id.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    #[test]
    fn test_normalize_name_strips_articles_and_collapses_whitespace() {
        assert_eq!(normalize_name("The   Murdaugh  Family"), "murdaugh family");
        assert_eq!(normalize_name("A  Defendant"), "defendant");
        assert_eq!(normalize_name("Alex Murdaugh"), "alex murdaugh");
    }

    fn jw(a: &str, b: &str) -> f32 {
        let av: Vec<char> = a.chars().collect();
        let bv: Vec<char> = b.chars().collect();
        jaro_winkler(&av, &bv)
    }

    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jw("alex", "alex") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_composite_score_name_dominant() {
        // Near-identical names, no context → should cross 0.7 threshold
        let s = composite_score(0.94, 0.0, 0.0);
        assert!(s >= 0.7, "expected >= 0.7, got {}", s);
    }

    #[test]
    fn test_composite_score_alias_dominant() {
        // Different names but full alias overlap → alias_only rule fires
        let s = composite_score(0.3, 1.0, 0.0);
        assert!(s >= 0.85, "expected >= 0.85, got {}", s);
    }

    #[test]
    fn test_composite_score_unrelated_names_not_saved_by_context() {
        // Completely different names, even with full participation overlap,
        // should NOT cross threshold — prevents "everyone who appears in the
        // same scene is the same person" false positives.
        let s = composite_score(0.2, 0.0, 1.0);
        assert!(s < 0.5, "expected < 0.5, got {}", s);
    }

    #[test]
    fn test_jaro_winkler_similar_names() {
        // "alex murdaugh" vs "alexander murdaugh" — should score quite high
        let s = jw("alex murdaugh", "alexander murdaugh");
        assert!(s > 0.85, "expected > 0.85, got {}", s);
    }

    #[test]
    fn test_jaro_winkler_unrelated() {
        let s = jw("fagin", "oliver");
        assert!(s < 0.5, "expected < 0.5, got {}", s);
    }

    #[test]
    fn test_set_overlap_empty() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = ["x".to_string()].into_iter().collect();
        assert_eq!(set_overlap(&a, &b), 0.0);
    }

    #[test]
    fn test_set_overlap_full_subset() {
        let a: HashSet<String> = ["x".to_string()].into_iter().collect();
        let b: HashSet<String> = ["x".to_string(), "y".to_string()].into_iter().collect();
        assert_eq!(set_overlap(&a, &b), 1.0); // |A ∩ B| / min(|A|, |B|) = 1/1
    }

    #[test]
    fn test_find_candidates_no_duplicates() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let narrative = "test-narrative";

        let e1 = make_actor("Alice", &[], narrative);
        let e2 = make_actor("Bob", &[], narrative);
        hg.create_entity(e1).unwrap();
        hg.create_entity(e2).unwrap();

        let candidates =
            find_duplicate_candidates(&hg, narrative, &DedupOptions::default()).unwrap();
        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_find_candidates_obvious_duplicate() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let narrative = "case-narrative";

        let alex1 = make_actor("Alex Murdaugh", &["Murdaugh"], narrative);
        let alex2 = make_actor("Alexander Murdaugh", &["the defendant"], narrative);
        hg.create_entity(alex1).unwrap();
        hg.create_entity(alex2).unwrap();

        let candidates =
            find_duplicate_candidates(&hg, narrative, &DedupOptions::default()).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].score >= 0.7);
        assert!(candidates[0].breakdown.name_similarity > 0.8);
    }

    #[test]
    fn test_find_candidates_participation_boost() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let narrative = "case-narrative";

        // Moderately similar names AND 100% participation overlap. The
        // context_boost term multiplies name_sim by up to 1.0 instead of the
        // 0.8 default, so a borderline name pair gets lifted above threshold
        // when their situations fully overlap. Conversely, truly unrelated
        // names ("D Smith" vs "Detective Smith") should NOT be saved by
        // participation overlap alone — see test_composite_score_unrelated.
        let e1 = make_actor("Maggie Murdaugh", &[], narrative);
        let e2 = make_actor("Margaret Murdaugh", &[], narrative);
        let e1_id = e1.id;
        let e2_id = e2.id;
        hg.create_entity(e1).unwrap();
        hg.create_entity(e2).unwrap();

        // Create 3 situations that both participate in
        for _ in 0..3 {
            let s = make_situation(narrative);
            let sid = s.id;
            hg.create_situation(s).unwrap();
            hg.add_participant(Participation {
                entity_id: e1_id,
                situation_id: sid,
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
            hg.add_participant(Participation {
                entity_id: e2_id,
                situation_id: sid,
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let candidates =
            find_duplicate_candidates(&hg, narrative, &DedupOptions::default()).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].breakdown.participation_overlap - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_find_candidates_threshold_filter() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let narrative = "case-narrative";

        let e1 = make_actor("Alice Johnson", &[], narrative);
        let e2 = make_actor("Bob Williams", &[], narrative);
        hg.create_entity(e1).unwrap();
        hg.create_entity(e2).unwrap();

        // Default threshold 0.7 should filter these out
        let candidates =
            find_duplicate_candidates(&hg, narrative, &DedupOptions::default()).unwrap();
        assert_eq!(candidates.len(), 0);

        // Low threshold should include them
        let loose = DedupOptions {
            threshold: 0.0,
            ..DedupOptions::default()
        };
        let candidates = find_duplicate_candidates(&hg, narrative, &loose).unwrap();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_find_candidates_type_filter() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let narrative = "test";

        let actor1 = make_actor("Alex", &[], narrative);
        let actor2 = make_actor("Alex", &[], narrative);
        let mut location = make_actor("Alex", &[], narrative);
        location.entity_type = EntityType::Location;

        hg.create_entity(actor1).unwrap();
        hg.create_entity(actor2).unwrap();
        hg.create_entity(location).unwrap();

        // Even though the location is named "Alex", we should not get a
        // cross-type candidate (actor ↔ location).
        let candidates =
            find_duplicate_candidates(&hg, narrative, &DedupOptions::default()).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].entity_type, EntityType::Actor);
    }

    #[test]
    fn test_find_candidates_respects_narrative_boundary() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let a = make_actor("Alex", &[], "narrative-a");
        let b = make_actor("Alex", &[], "narrative-b");
        hg.create_entity(a).unwrap();
        hg.create_entity(b).unwrap();

        // Dedup within narrative-a should see exactly one entity → no pairs
        let cands_a =
            find_duplicate_candidates(&hg, "narrative-a", &DedupOptions::default()).unwrap();
        assert_eq!(cands_a.len(), 0);
    }

    #[test]
    fn test_keep_is_entity_with_more_participations() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let narrative = "test";

        let sparse = make_actor("Alex Murdaugh", &[], narrative);
        let rich = make_actor("Alexander Murdaugh", &[], narrative);
        let sparse_id = sparse.id;
        let rich_id = rich.id;
        hg.create_entity(sparse).unwrap();
        hg.create_entity(rich).unwrap();

        // Give `rich` 3 participations, `sparse` 0
        for _ in 0..3 {
            let s = make_situation(narrative);
            let sid = s.id;
            hg.create_situation(s).unwrap();
            hg.add_participant(Participation {
                entity_id: rich_id,
                situation_id: sid,
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let candidates =
            find_duplicate_candidates(&hg, narrative, &DedupOptions::default()).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].keep_id, rich_id);
        assert_eq!(candidates[0].absorb_id, sparse_id);
    }
}
