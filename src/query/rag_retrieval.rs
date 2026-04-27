//! RAG retrieval helpers — local, global, and context assembly.
//!
//! Provides retrieval functions that assemble context from the hypergraph,
//! vector index, and community summaries for use in RAG queries.

use crate::analysis::community::CommunitySummary;
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::ingestion::embed::EmbeddingProvider;
use crate::ingestion::vector::VectorIndex;
use crate::query::bm25::BM25Index;
use crate::query::rag_config::RetrievalMode;
use crate::query::reranker::Reranker;
use crate::query::token_budget::{ItemCategory, ScoredItem, TokenBudget};
use crate::types::{Entity, Situation};

/// Assembled context from retrieval.
#[derive(Debug, Default)]
pub struct RagContext {
    /// Entities with relevance scores.
    pub entities: Vec<(Entity, f32)>,
    /// Situations with relevance scores.
    pub situations: Vec<(Situation, f32)>,
    /// Community summaries with relevance scores.
    pub community_summaries: Vec<(CommunitySummary, f32)>,
}

/// Summarize an entity as a text string for the RAG prompt.
pub fn summarize_entity(entity: &Entity) -> String {
    let name = entity
        .properties
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("(unnamed)");
    let mut summary = format!(
        "[Entity: {} | Type: {:?} | Confidence: {:.2}",
        name, entity.entity_type, entity.confidence
    );
    // Add key properties (skip name since already shown)
    if let Some(obj) = entity.properties.as_object() {
        let extras: Vec<String> = obj
            .iter()
            .filter(|(k, _)| *k != "name")
            .take(5)
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect();
        if !extras.is_empty() {
            summary.push_str(" | ");
            summary.push_str(&extras.join(", "));
        }
    }
    if let Some(nid) = &entity.narrative_id {
        summary.push_str(&format!(" | narrative: {}", nid));
    }
    summary.push(']');
    summary
}

/// Summarize a situation as a text string for the RAG prompt.
pub fn summarize_situation(situation: &Situation) -> String {
    let content_text: String = situation
        .raw_content
        .iter()
        .map(|cb| cb.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let truncated = crate::export::truncate_with_ellipsis(&content_text, 500);
    let mut summary = format!(
        "[Situation: {:?} | Confidence: {:.2}",
        situation.narrative_level, situation.confidence,
    );
    if let Some(start) = &situation.temporal.start {
        summary.push_str(&format!(" | time: {}", start.format("%Y-%m-%d")));
    }
    if let Some(nid) = &situation.narrative_id {
        summary.push_str(&format!(" | narrative: {}", nid));
    }
    summary.push_str(&format!("]\n{}", truncated));
    summary
}

/// Summarize a community summary as a text string for the RAG prompt.
pub fn summarize_community(cs: &CommunitySummary) -> String {
    format!(
        "[Community {} | {} entities: {}]\n{}\nThemes: {}",
        cs.community_id,
        cs.entity_count,
        cs.entity_names
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
        cs.summary,
        cs.key_themes.join(", "),
    )
}

// ─── Retrieval Helpers ──────────────────────────────────────────

/// Local retrieval with optional BM25 hybrid search.
fn retrieve_local_hybrid(
    question: &str,
    narrative_id: Option<&str>,
    hypergraph: &Hypergraph,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    bm25_index: Option<&BM25Index>,
    hybrid_alpha: f32,
) -> Result<(Vec<(Entity, f32)>, Vec<(Situation, f32)>)> {
    let mut entities: Vec<(Entity, f32)> = Vec::new();
    let mut situations: Vec<(Situation, f32)> = Vec::new();

    let has_vector = vector_index.is_some() && embedder.is_some();

    if has_vector {
        let emb = embedder.expect("checked above");
        let vi = vector_index.expect("checked above");

        let query_embedding = emb.embed_text(question)?;
        let top_k = 200;
        let vector_results = vi.search(&query_embedding, top_k)?;

        // If BM25 is available, merge with vector results
        let results: Vec<(uuid::Uuid, f32)> = if let Some(bm25) = bm25_index {
            let bm25_results = bm25.search(question, top_k);
            let vector_pairs: Vec<(uuid::Uuid, f32)> =
                vector_results.iter().map(|r| (r.id, r.score)).collect();
            crate::query::bm25::merge_hybrid_results(&vector_pairs, &bm25_results, hybrid_alpha)
        } else {
            vector_results.iter().map(|r| (r.id, r.score)).collect()
        };

        for (result_id, result_score) in &results {
            if let Ok(entity) = hypergraph.get_entity(result_id) {
                if let Some(nid) = narrative_id {
                    if entity.narrative_id.as_deref() != Some(nid) {
                        continue;
                    }
                }
                entities.push((entity, *result_score));
            } else if let Ok(situation) = hypergraph.get_situation(result_id) {
                if let Some(nid) = narrative_id {
                    if situation.narrative_id.as_deref() != Some(nid) {
                        continue;
                    }
                }
                situations.push((situation, *result_score));
            }
        }

        // Boost scores by entity confidence + participation degree (a proxy
        // for centrality). This prevents peripheral entities from dominating
        // the result set on abstract queries like "who are the main characters"
        // where raw vector similarity is nearly flat across short entity
        // summaries. Formula: score * (0.5 + 0.5*confidence) * (1 + ln(1+degree) * 0.15)
        for (entity, score) in entities.iter_mut() {
            let degree = hypergraph
                .get_situations_for_entity(&entity.id)
                .map(|ps| ps.len())
                .unwrap_or(0);
            let conf_boost = 0.5 + 0.5 * entity.confidence;
            let degree_boost = 1.0 + (degree as f32 + 1.0).ln() * 0.15;
            *score *= conf_boost * degree_boost;
        }

        // Supplement the vector-search pool with the top entities by
        // confidence * centrality pulled directly from the hypergraph.
        // This is essential when:
        // - embeddings are degenerate (hash-based or missing for central entities)
        // - the query is an abstract overview ("main characters", "key events")
        //   where vector similarity to short name-only summaries is nearly flat
        // - top_k vector results don't include the true protagonists
        // We union by entity_id and keep the max score for duplicates.
        let existing_ids: std::collections::HashSet<uuid::Uuid> =
            entities.iter().map(|(e, _)| e.id).collect();
        let all_in_narrative = if let Some(nid) = narrative_id {
            hypergraph
                .list_entities_by_narrative(nid)
                .unwrap_or_default()
        } else {
            let mut all = Vec::new();
            for et in &[
                crate::types::EntityType::Actor,
                crate::types::EntityType::Location,
                crate::types::EntityType::Artifact,
                crate::types::EntityType::Concept,
                crate::types::EntityType::Organization,
            ] {
                all.extend(hypergraph.list_entities_by_type(et).unwrap_or_default());
            }
            all
        };
        let mut by_centrality: Vec<(Entity, f32)> = all_in_narrative
            .into_iter()
            .filter(|e| !existing_ids.contains(&e.id))
            .map(|e| {
                let degree = hypergraph
                    .get_situations_for_entity(&e.id)
                    .map(|ps| ps.len())
                    .unwrap_or(0);
                // Pure confidence * log-degree signal, normalized to roughly
                // the same range as boosted vector scores (0-2).
                let score = e.confidence * (1.0 + (degree as f32 + 1.0).ln() * 0.3);
                (e, score)
            })
            .collect();
        by_centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Take the top 20 central entities that weren't already in vector results
        entities.extend(by_centrality.into_iter().take(20));

        // Final ranking pass over the merged pool
        entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Also boost situations by confidence (no degree proxy available, but
        // keep the same shape so aggregate queries benefit).
        for (situation, score) in situations.iter_mut() {
            let conf_boost = 0.5 + 0.5 * situation.confidence;
            *score *= conf_boost;
        }

        // Expand situations via participation: for each top-ranked entity,
        // pull its participation situations directly from the hypergraph and
        // mix them into the candidate pool. This is critical for questions
        // that name an entity (e.g. "who killed Gordon Way") — the entity
        // gets retrieved but the specific situations describing its fate
        // often don't rank high enough in vector search to be pulled on their
        // own, yet they sit one hop away in the participation graph. Without
        // this expansion, the LLM sees Gordon Way the entity but not the
        // shooting situation, and answers "context does not specify".
        //
        // Done AFTER the entity candidate pool is assembled so that both
        // vector-matched entities and centrality-supplemented entities
        // contribute their situations.
        let existing_sit_ids: std::collections::HashSet<uuid::Uuid> =
            situations.iter().map(|(s, _)| s.id).collect();
        let mut expanded_sit_ids = existing_sit_ids.clone();
        // Use top 15 entities by current score — enough to cover the question's
        // focal entities without exploding the pool.
        let top_entity_snapshot: Vec<(uuid::Uuid, f32)> = {
            let mut sorted = entities.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted
                .into_iter()
                .take(15)
                .map(|(e, s)| (e.id, s))
                .collect()
        };
        for (eid, entity_score) in &top_entity_snapshot {
            if let Ok(parts) = hypergraph.get_situations_for_entity(eid) {
                // Pull up to 10 participations per entity; we'll trim later.
                for p in parts.into_iter().take(10) {
                    if !expanded_sit_ids.insert(p.situation_id) {
                        continue;
                    }
                    if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
                        if let Some(nid) = narrative_id {
                            if sit.narrative_id.as_deref() != Some(nid) {
                                continue;
                            }
                        }
                        // Derived score: propagate some of the parent entity's
                        // score but below a direct vector hit, and apply the
                        // same confidence boost.
                        let conf_boost = 0.5 + 0.5 * sit.confidence;
                        let derived = 0.7 * entity_score * conf_boost;
                        situations.push((sit, derived));
                    }
                }
            }
        }

        situations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Cap at a reasonable size for downstream budget allocation.
        // Situations cap raised to 80 so the expanded pool has room; the
        // token budget will trim further as needed.
        entities.truncate(50);
        situations.truncate(80);
    }

    // Fall back to hypergraph listing when vector search returned nothing
    // (e.g. empty in-memory index after server restart)
    if entities.is_empty() && situations.is_empty() {
        let q_lower = question.to_lowercase();
        let q_words: Vec<&str> = q_lower.split_whitespace().filter(|w| w.len() > 2).collect();

        let all_entities = if let Some(nid) = narrative_id {
            hypergraph.list_entities_by_narrative(nid)?
        } else {
            let mut all = Vec::new();
            for et in &[
                crate::types::EntityType::Actor,
                crate::types::EntityType::Location,
                crate::types::EntityType::Artifact,
                crate::types::EntityType::Concept,
                crate::types::EntityType::Organization,
            ] {
                all.extend(hypergraph.list_entities_by_type(et)?);
            }
            all
        };

        // Score entities by name overlap with the question, then boost by
        // confidence and participation degree so main characters surface
        // even when the question doesn't name anyone (e.g. "main characters").
        let mut scored: Vec<(Entity, f32)> = all_entities
            .into_iter()
            .map(|e| {
                let name = e
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                let name_words: Vec<&str> = name.split_whitespace().collect();
                let overlap = q_words
                    .iter()
                    .filter(|w| {
                        name_words
                            .iter()
                            .any(|nw| nw.contains(*w) || w.contains(nw))
                    })
                    .count() as f32;
                let base = if overlap > 0.0 {
                    0.5 + overlap * 0.2
                } else {
                    0.1
                };
                let degree = hypergraph
                    .get_situations_for_entity(&e.id)
                    .map(|ps| ps.len())
                    .unwrap_or(0);
                let conf_boost = 0.5 + 0.5 * e.confidence;
                let degree_boost = 1.0 + (degree as f32 + 1.0).ln() * 0.15;
                let score = base * conf_boost * degree_boost;
                (e, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entities = scored.into_iter().take(50).collect();

        // For matching entities, also pull their situations
        let top_entity_ids: Vec<uuid::Uuid> = entities
            .iter()
            .filter(|(_, s)| *s > 0.3)
            .map(|(e, _)| e.id)
            .take(10)
            .collect();
        for eid in &top_entity_ids {
            if let Ok(parts) = hypergraph.get_situations_for_entity(eid) {
                for p in parts.into_iter().take(15) {
                    if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
                        if narrative_id.is_none() || sit.narrative_id.as_deref() == narrative_id {
                            let conf_boost = 0.5 + 0.5 * sit.confidence;
                            situations.push((sit, 0.6 * conf_boost));
                        }
                    }
                }
            }
        }

        // Deduplicate situations by ID
        let mut seen = std::collections::HashSet::new();
        situations.retain(|(s, _)| seen.insert(s.id));

        // Cap situations
        situations.truncate(50);
    }

    Ok((entities, situations))
}

/// Retrieve community summaries for a narrative from KV.
/// Returns summaries with a default relevance score of 0.7.
fn retrieve_global(
    narrative_id: Option<&str>,
    store: &dyn crate::store::KVStore,
) -> Vec<(CommunitySummary, f32)> {
    let nid = match narrative_id {
        Some(n) => n,
        None => return Vec::new(),
    };
    match crate::analysis::community::list_summaries(store, nid) {
        Ok(summaries) => summaries.into_iter().map(|s| (s, 0.7)).collect(),
        Err(_) => Vec::new(),
    }
}

/// Assemble context for a RAG query from the hypergraph and vector index.
///
/// Embeds the question, searches the vector index, loads matching
/// entities/situations, filters by narrative, and allocates within budget.
/// Falls back to hypergraph listing when vector index or embedder is unavailable.
///
/// If a `reranker` is provided, retrieved items are reranked by relevance
/// to the question before budget allocation.
///
/// Mode behavior:
/// - **Local**: entities + situations only (entity-focused)
/// - **Global**: community summaries first, falls back to Local if none exist
/// - **Hybrid**: combines Local entities/situations with Global community summaries
/// - **Mix**: keyword-driven — high-level keywords query Global, low-level query Local
pub fn assemble_context(
    question: &str,
    narrative_id: Option<&str>,
    mode: &RetrievalMode,
    hypergraph: &Hypergraph,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    budget: &TokenBudget,
    reranker: Option<&dyn Reranker>,
) -> Result<RagContext> {
    assemble_context_hybrid(
        question,
        narrative_id,
        mode,
        hypergraph,
        vector_index,
        embedder,
        budget,
        reranker,
        None,
        0.7,
    )
}

/// Assemble context with optional BM25 hybrid search.
pub fn assemble_context_hybrid(
    question: &str,
    narrative_id: Option<&str>,
    mode: &RetrievalMode,
    hypergraph: &Hypergraph,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    budget: &TokenBudget,
    reranker: Option<&dyn Reranker>,
    bm25_index: Option<&BM25Index>,
    hybrid_alpha: f32,
) -> Result<RagContext> {
    let mut entities: Vec<(Entity, f32)> = Vec::new();
    let mut situations: Vec<(Situation, f32)> = Vec::new();
    let mut community_summaries: Vec<(CommunitySummary, f32)> = Vec::new();

    match mode {
        RetrievalMode::Local => {
            let (ents, sits) = retrieve_local_hybrid(
                question,
                narrative_id,
                hypergraph,
                vector_index,
                embedder,
                bm25_index,
                hybrid_alpha,
            )?;
            entities = ents;
            situations = sits;
        }
        RetrievalMode::Global => {
            // Try community summaries first
            community_summaries = retrieve_global(narrative_id, hypergraph.store());
            if community_summaries.is_empty() {
                // Fall back to Local when no summaries exist
                let (ents, sits) = retrieve_local_hybrid(
                    question,
                    narrative_id,
                    hypergraph,
                    vector_index,
                    embedder,
                    bm25_index,
                    hybrid_alpha,
                )?;
                entities = ents;
                situations = sits;
            }
        }
        RetrievalMode::Hybrid => {
            let (ents, sits) = retrieve_local_hybrid(
                question,
                narrative_id,
                hypergraph,
                vector_index,
                embedder,
                bm25_index,
                hybrid_alpha,
            )?;
            entities = ents;
            situations = sits;
            community_summaries = retrieve_global(narrative_id, hypergraph.store());
        }
        RetrievalMode::Drift => {
            // DRIFT: Dynamic Reasoning with Flexible Traversal
            // Three-phase adaptive retrieval traversing the community hierarchy.

            // Load the community hierarchy for this narrative
            let hierarchy = if let Some(nid) = narrative_id {
                crate::analysis::community::get_hierarchy(hypergraph.store(), nid)
                    .unwrap_or_default()
            } else {
                std::collections::HashMap::new()
            };

            if hierarchy.is_empty() {
                // No community hierarchy — fall back to Hybrid
                let (ents, sits) = retrieve_local_hybrid(
                    question,
                    narrative_id,
                    hypergraph,
                    vector_index,
                    embedder,
                    bm25_index,
                    hybrid_alpha,
                )?;
                entities = ents;
                situations = sits;
                community_summaries = retrieve_global(narrative_id, hypergraph.store());
            } else {
                // ── Phase 1: Community Primer ──
                // Start at coarsest level, score by keyword relevance
                let max_level = *hierarchy.keys().max().unwrap_or(&0);
                let coarsest = hierarchy.get(&max_level).cloned().unwrap_or_default();

                let keywords = crate::query::keywords::extract_keywords_heuristic(question);
                let q_words: Vec<String> = keywords
                    .high_level
                    .iter()
                    .chain(keywords.low_level.iter())
                    .map(|w| w.to_lowercase())
                    .collect();

                let max_entity_count = coarsest
                    .iter()
                    .map(|cs| cs.entity_count)
                    .max()
                    .unwrap_or(1)
                    .max(1);

                const DRIFT_K_PRIMER: usize = 5;
                const DRIFT_K_DRILL: usize = 3;

                let mut scored_top: Vec<(CommunitySummary, f32)> = coarsest
                    .into_iter()
                    .map(|cs| {
                        let text = format!(
                            "{} {}",
                            cs.summary.to_lowercase(),
                            cs.key_themes.join(" ").to_lowercase()
                        );
                        let keyword_overlap =
                            q_words.iter().filter(|w| text.contains(w.as_str())).count() as f32;
                        let entity_norm = ((1 + cs.entity_count) as f32).ln()
                            / ((1 + max_entity_count) as f32).ln();
                        let score = 0.4 * keyword_overlap + 0.3 * entity_norm + 0.3;
                        (cs, score)
                    })
                    .collect();
                scored_top
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_primer: Vec<CommunitySummary> = scored_top
                    .iter()
                    .take(DRIFT_K_PRIMER)
                    .map(|(cs, _)| cs.clone())
                    .collect();

                // Add Phase 1 summaries to results
                for (cs, score) in scored_top.iter().take(DRIFT_K_PRIMER) {
                    community_summaries.push((cs.clone(), *score));
                }

                // ── Phase 2: Follow-Up Drill-Down ──
                // For each selected top community, explore its children
                // Emphasize keywords NOT covered by the parent's themes
                let mut drill_summaries: Vec<(CommunitySummary, f32)> = Vec::new();

                for parent in &top_primer {
                    let covered: std::collections::HashSet<String> = parent
                        .key_themes
                        .iter()
                        .flat_map(|t| {
                            t.to_lowercase()
                                .split_whitespace()
                                .map(String::from)
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    // Gap keywords: those not covered by parent themes
                    let gap_words: Vec<&String> = q_words
                        .iter()
                        .filter(|w| !covered.contains(w.as_str()))
                        .collect();

                    // Find children: use child_community_ids, or scan lower level
                    let mut children: Vec<CommunitySummary> = Vec::new();
                    if !parent.child_community_ids.is_empty() {
                        // Look up children from the next lower level
                        if max_level > 0 {
                            if let Some(lower) = hierarchy.get(&(max_level - 1)) {
                                for child in lower {
                                    if parent.child_community_ids.contains(&child.community_id) {
                                        children.push(child.clone());
                                    }
                                }
                            }
                        }
                    } else if max_level > 0 {
                        // Fallback: scan lower level for children matching parent_community_id
                        if let Some(lower) = hierarchy.get(&(max_level - 1)) {
                            for child in lower {
                                if child.parent_community_id == Some(parent.community_id) {
                                    children.push(child.clone());
                                }
                            }
                        }
                    }

                    // Score children by gap keyword overlap
                    let mut scored_children: Vec<(CommunitySummary, f32)> = children
                        .into_iter()
                        .map(|cs| {
                            let text = format!(
                                "{} {}",
                                cs.summary.to_lowercase(),
                                cs.key_themes.join(" ").to_lowercase()
                            );
                            let gap_overlap = gap_words
                                .iter()
                                .filter(|w| text.contains(w.as_str()))
                                .count() as f32;
                            let all_overlap =
                                q_words.iter().filter(|w| text.contains(w.as_str())).count() as f32;
                            let score = 0.5 * gap_overlap + 0.3 * all_overlap + 0.2;
                            (cs, score)
                        })
                        .collect();
                    scored_children
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    drill_summaries.extend(scored_children.into_iter().take(DRIFT_K_DRILL));
                }

                // Add Phase 2 summaries
                for (cs, score) in &drill_summaries {
                    community_summaries.push((cs.clone(), *score));
                }

                // ── Phase 3: Leaf Entity Retrieval ──
                // Collect entity IDs from all selected communities
                let target_entity_ids: std::collections::HashSet<uuid::Uuid> = community_summaries
                    .iter()
                    .flat_map(|(cs, _)| cs.entity_ids.iter().copied())
                    .collect();

                for eid in &target_entity_ids {
                    if let Ok(entity) = hypergraph.get_entity(eid) {
                        let degree = hypergraph
                            .get_situations_for_entity(&entity.id)
                            .map(|ps| ps.len())
                            .unwrap_or(0);
                        let conf_boost = 0.5 + 0.5 * entity.confidence;
                        let degree_boost = 1.0 + (degree as f32 + 1.0).ln() * 0.15;
                        let score = 0.8 * conf_boost * degree_boost;
                        entities.push((entity, score));
                    }
                }
                entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Retrieve situations for top entities
                let top_entity_ids: Vec<uuid::Uuid> =
                    entities.iter().take(20).map(|(e, _)| e.id).collect();
                for eid in &top_entity_ids {
                    if let Ok(participations) = hypergraph.get_situations_for_entity(eid) {
                        for p in participations.iter().take(5) {
                            if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
                                let conf_boost = 0.5 + 0.5 * sit.confidence;
                                situations.push((sit, 0.6 * conf_boost));
                            }
                        }
                    }
                }
                // Deduplicate situations
                let mut seen = std::collections::HashSet::new();
                situations.retain(|(s, _)| seen.insert(s.id));
            }
        }
        RetrievalMode::Mix => {
            // Use heuristic keyword extraction to split
            let keywords = crate::query::keywords::extract_keywords_heuristic(question);

            // If we have high-level keywords, pull community summaries
            if !keywords.high_level.is_empty() {
                community_summaries = retrieve_global(narrative_id, hypergraph.store());
            }

            // Always pull local context for low-level (specific) keywords
            let (ents, sits) = retrieve_local_hybrid(
                question,
                narrative_id,
                hypergraph,
                vector_index,
                embedder,
                bm25_index,
                hybrid_alpha,
            )?;
            entities = ents;
            situations = sits;
        }
        RetrievalMode::Lazy => {
            // LazyGraphRAG: no pre-computation. Vector search → local subgraph → on-demand summary.
            // Step 1: Local retrieval (same as Local mode)
            let (ents, sits) = retrieve_local_hybrid(
                question,
                narrative_id,
                hypergraph,
                vector_index,
                embedder,
                bm25_index,
                hybrid_alpha,
            )?;
            entities = ents;
            situations = sits;

            // Step 2: On-demand mini-community detection from retrieved entities
            if let Some(nid) = narrative_id {
                if let Ok(graph) =
                    crate::analysis::graph_projection::build_co_graph(hypergraph, nid)
                {
                    // Find entities in the co-graph and detect their local community via label propagation
                    let entity_ids: std::collections::HashSet<uuid::Uuid> =
                        entities.iter().map(|(e, _)| e.id).collect();
                    let prize_indices: Vec<usize> = graph
                        .entities
                        .iter()
                        .enumerate()
                        .filter(|(_, eid)| entity_ids.contains(eid))
                        .map(|(i, _)| i)
                        .collect();

                    if prize_indices.len() >= 2 {
                        // Build PCST subgraph connecting the retrieved entities
                        let subgraph_nodes = crate::analysis::pathfinding::pcst_approximation(
                            &graph,
                            &prize_indices,
                        );
                        // Boost entities in the connected subgraph
                        for (entity, score) in &mut entities {
                            if let Some(idx) = graph.entities.iter().position(|&e| e == entity.id) {
                                if subgraph_nodes.contains(&idx) {
                                    *score *= 1.3; // boost connected entities
                                }
                            }
                        }
                    }
                }
            }
        }
        RetrievalMode::Ppr => {
            // Personalized PageRank: seed from vector-search-relevant entities.
            // Step 1: Get seed entities via vector search
            let (seed_ents, _) = retrieve_local_hybrid(
                question,
                narrative_id,
                hypergraph,
                vector_index,
                embedder,
                bm25_index,
                hybrid_alpha,
            )?;

            // Step 2: Run PPR on co-graph seeded from these entities
            if let Some(nid) = narrative_id {
                if let Ok(graph) =
                    crate::analysis::graph_projection::build_co_graph(hypergraph, nid)
                {
                    let n = graph.entities.len();
                    if n > 0 {
                        // Build seed weight vector: mass on retrieved entities
                        let mut seed_weights = vec![0.0_f64; n];
                        let seed_ids: std::collections::HashSet<uuid::Uuid> =
                            seed_ents.iter().map(|(e, _)| e.id).collect();
                        let mut seed_count = 0;
                        for (i, eid) in graph.entities.iter().enumerate() {
                            if seed_ids.contains(eid) {
                                seed_weights[i] = 1.0;
                                seed_count += 1;
                            }
                        }
                        if seed_count > 0 {
                            // Normalize
                            for w in &mut seed_weights {
                                *w /= seed_count as f64;
                            }
                            let ppr = crate::analysis::graph_centrality::personalized_pagerank(
                                &graph,
                                &seed_weights,
                                0.15,
                            );

                            // Rank entities by PPR score, take top 50
                            let mut scored: Vec<(usize, f64)> = ppr
                                .iter()
                                .enumerate()
                                .filter(|(_, &s)| s > 0.0)
                                .map(|(i, &s)| (i, s))
                                .collect();
                            scored.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            scored.truncate(50);

                            // Fetch entities
                            for (idx, score) in scored {
                                if let Ok(entity) = hypergraph.get_entity(&graph.entities[idx]) {
                                    entities.push((entity, score as f32));
                                }
                            }
                        }
                    }
                }
            }

            // If PPR didn't produce results, fall back to the seed entities
            if entities.is_empty() {
                entities = seed_ents;
            }

            // Get situations for PPR-selected entities (top 5 per entity)
            for (entity, _) in &entities {
                let participations = hypergraph
                    .get_situations_for_entity(&entity.id)
                    .unwrap_or_default();
                for p in participations.iter().take(5) {
                    if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
                        if !situations.iter().any(|(s, _)| s.id == sit.id) {
                            situations.push((sit, 0.5));
                        }
                    }
                }
            }
        }
    }

    // Apply reranker if provided: reorder entities and situations by relevance
    if let Some(rr) = reranker {
        // Build text representations for all retrieved items
        let mut texts: Vec<String> = Vec::new();
        let mut item_source: Vec<(&str, usize)> = Vec::new(); // ("entity"|"situation", index)

        for (entity, _) in &entities {
            texts.push(summarize_entity(entity));
            item_source.push(("entity", item_source.len()));
        }
        let entity_count = entities.len();
        for (situation, _) in &situations {
            texts.push(summarize_situation(situation));
            item_source.push(("situation", item_source.len()));
        }

        if !texts.is_empty() {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let ranked = rr.rerank(question, &text_refs)?;

            // Rebuild entities and situations in reranked order with reranker scores
            let mut new_entities: Vec<(Entity, f32)> = Vec::new();
            let mut new_situations: Vec<(Situation, f32)> = Vec::new();

            for (orig_idx, score) in ranked {
                if orig_idx < entity_count {
                    let (entity, _old_score) = entities[orig_idx].clone();
                    new_entities.push((entity, score));
                } else {
                    let sit_idx = orig_idx - entity_count;
                    let (situation, _old_score) = situations[sit_idx].clone();
                    new_situations.push((situation, score));
                }
            }

            entities = new_entities;
            situations = new_situations;
        }
    }

    // Deduplicate entities and situations by id before budget allocation.
    // This is critical: the upstream pipeline (vector search + centrality
    // supplement + participation expansion) can introduce duplicates, and
    // without this dedup the budget allocator would consume slots on
    // identical items and the rebuild step would surface duplicate citations.
    {
        let mut seen = std::collections::HashSet::new();
        entities.retain(|(e, _)| seen.insert(e.id));
    }
    {
        let mut seen = std::collections::HashSet::new();
        situations.retain(|(s, _)| {
            // Drop situations with zero useful content — they waste budget
            // on a placeholder like "[Situation: Scene | Confidence: X.XX]"
            // and confuse the LLM. A situation with no raw_content is not
            // retrievable information.
            let has_content = s.raw_content.iter().any(|cb| !cb.content.trim().is_empty());
            has_content && seen.insert(s.id)
        });
    }

    // Convert to ScoredItems keyed by original index so we can rebuild
    // without text matching (which collapses distinct items with identical
    // summaries onto the first match).
    //
    // We append a unique "#<idx>" suffix to each content string before
    // passing to the allocator, then strip it back off during rebuild.
    // The suffix is never shown to the LLM because rebuild looks up the
    // original pair by index, not by the tagged content.
    let mut scored_items: Vec<ScoredItem> = Vec::new();

    for (idx, (entity, score)) in entities.iter().enumerate() {
        let text = summarize_entity(entity);
        let tokens = TokenBudget::estimate_tokens(&text);
        scored_items.push(ScoredItem {
            category: ItemCategory::Entity,
            content: format!("e#{}\u{0}{}", idx, text),
            score: *score,
            token_estimate: tokens,
        });
    }

    for (idx, (situation, score)) in situations.iter().enumerate() {
        let text = summarize_situation(situation);
        let tokens = TokenBudget::estimate_tokens(&text);
        scored_items.push(ScoredItem {
            category: ItemCategory::Situation,
            content: format!("s#{}\u{0}{}", idx, text),
            score: *score,
            token_estimate: tokens,
        });
    }

    for (idx, (cs, score)) in community_summaries.iter().enumerate() {
        let text = summarize_community(cs);
        let tokens = TokenBudget::estimate_tokens(&text);
        scored_items.push(ScoredItem {
            category: ItemCategory::Community,
            content: format!("c#{}\u{0}{}", idx, text),
            score: *score,
            token_estimate: tokens,
        });
    }

    let allocated = budget.allocate(&scored_items);

    // Rebuild the context by parsing the index prefix from each allocated
    // item's content. This is unambiguous even when multiple items produce
    // identical summary text, so duplicates cannot leak through.
    let mut ctx = RagContext::default();
    for item in &allocated {
        let (prefix, rest) = match item.content.split_once('\u{0}') {
            Some(parts) => parts,
            None => continue,
        };
        let _ = rest; // unused — we look up by index, not text
        let (kind, idx_str) = match prefix.split_once('#') {
            Some(parts) => parts,
            None => continue,
        };
        let idx: usize = match idx_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        match (kind, item.category) {
            ("e", ItemCategory::Entity) => {
                if let Some(pair) = entities.get(idx) {
                    ctx.entities.push(pair.clone());
                }
            }
            ("s", ItemCategory::Situation) => {
                if let Some(pair) = situations.get(idx) {
                    ctx.situations.push(pair.clone());
                }
            }
            ("c", ItemCategory::Community) => {
                if let Some(pair) = community_summaries.get(idx) {
                    ctx.community_summaries.push(pair.clone());
                }
            }
            _ => {}
        }
    }

    Ok(ctx)
}

/// Build the RAG system prompt from a question and assembled context.
pub fn build_rag_prompt(question: &str, context: &RagContext) -> String {
    let mut prompt = String::from(
        "You are a narrative intelligence analyst. Answer the question based ONLY on the following context. \
         Cite specific entities and situations by name.\n\n"
    );

    if !context.community_summaries.is_empty() {
        prompt.push_str("## Community Overviews\n");
        for (summary, score) in &context.community_summaries {
            prompt.push_str(&format!(
                "### Community {} (relevance: {:.2})\n{}\nEntities: {}\nThemes: {}\n\n",
                summary.community_id,
                score,
                summary.summary,
                summary.entity_names.join(", "),
                summary.key_themes.join(", "),
            ));
        }
    }

    if !context.entities.is_empty() {
        prompt.push_str("## Entities\n");
        for (entity, _score) in &context.entities {
            prompt.push_str(&summarize_entity(entity));
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    if !context.situations.is_empty() {
        prompt.push_str("## Situations\n");
        for (situation, _score) in &context.situations {
            prompt.push_str(&summarize_situation(situation));
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    if context.entities.is_empty()
        && context.situations.is_empty()
        && context.community_summaries.is_empty()
    {
        prompt.push_str("(No relevant context found in the knowledge base.)\n\n");
    }

    prompt.push_str("## Question\n");
    prompt.push_str(question);

    prompt
}
