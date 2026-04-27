//! RAG (Retrieval-Augmented Generation) query engine.
//!
//! Assembles context from the hypergraph and vector index, builds a
//! system prompt, calls the LLM via `NarrativeExtractor::answer_question`,
//! and returns a structured answer with citations.
//!
//! Supports four retrieval modes:
//! - **Local**: entity-focused retrieval via vector search
//! - **Global**: community-summary-focused retrieval
//! - **Hybrid**: combines Local + Global context
//! - **Mix**: keyword-driven split (HL → Global, LL → Local)

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::ingestion::embed::EmbeddingProvider;
use crate::ingestion::llm::NarrativeExtractor;
use crate::ingestion::vector::VectorIndex;
use crate::query::rag_config::RetrievalMode;
use crate::query::reranker::Reranker;
use crate::query::token_budget::TokenBudget;

// Re-export retrieval types and functions used by other modules.
pub use crate::query::rag_retrieval::{
    assemble_context, build_rag_prompt, summarize_community, summarize_entity, summarize_situation,
    RagContext,
};

/// A citation linking an answer to source data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Entity UUID if this citation refers to an entity.
    pub entity_id: Option<Uuid>,
    /// Situation UUID if this citation refers to a situation.
    pub situation_id: Option<Uuid>,
    /// Chunk UUID if this citation refers to a raw chunk.
    pub chunk_id: Option<Uuid>,
    /// Excerpt from the source data used as context.
    pub excerpt: String,
    /// Relevance score (0.0-1.0).
    pub score: f32,
}

/// Complete RAG answer with citations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagAnswer {
    /// The generated answer text.
    pub answer: String,
    /// Citations linking the answer to source data.
    pub citations: Vec<Citation>,
    /// Retrieval mode used.
    pub mode: String,
    /// Estimated tokens used in the context.
    pub tokens_used: usize,
    /// Follow-up question suggestions (populated when suggest=true).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
    /// Debug trace of the RAG pipeline (populated when debug=true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<RagDebugTrace>,
}

/// Debug trace showing how the RAG pipeline assembled its answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagDebugTrace {
    /// How entities/situations were retrieved.
    pub retrieval_method: String,
    /// Entities retrieved with names and scores (before budget trim).
    pub retrieved_entities: Vec<DebugEntity>,
    /// Situations retrieved with excerpts and scores.
    pub retrieved_situations: Vec<DebugSituation>,
    /// Community summaries used (count + theme keywords).
    pub community_summaries_used: usize,
    /// The full system prompt sent to the LLM.
    pub system_prompt: String,
    /// Token budget limits applied.
    pub budget_limits: String,
}

/// A retrieved entity in the debug trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugEntity {
    pub id: Uuid,
    pub name: String,
    pub entity_type: String,
    pub score: f32,
}

/// A retrieved situation in the debug trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSituation {
    pub id: Uuid,
    pub excerpt: String,
    pub score: f32,
}

/// Execute a full RAG pipeline: assemble context, build prompt, call LLM, return answer.
///
/// Pass `reranker: None` for default ordering, or provide a `Reranker`
/// implementation to reorder retrieved items by relevance before LLM call.
/// Pass `response_type` to instruct the LLM on desired output format.
/// Pass `suggest: true` to generate follow-up question suggestions.
pub fn execute_ask(
    question: &str,
    narrative_id: Option<&str>,
    mode: &RetrievalMode,
    budget: &TokenBudget,
    hypergraph: &Hypergraph,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    extractor: &dyn NarrativeExtractor,
    reranker: Option<&dyn Reranker>,
    response_type: Option<&str>,
    suggest: bool,
    session_history: Option<&str>,
) -> Result<RagAnswer> {
    execute_ask_inner(
        question,
        narrative_id,
        mode,
        budget,
        hypergraph,
        vector_index,
        embedder,
        extractor,
        reranker,
        response_type,
        suggest,
        session_history,
        false,
    )
}

/// Execute ASK with optional debug trace.
pub fn execute_ask_debug(
    question: &str,
    narrative_id: Option<&str>,
    mode: &RetrievalMode,
    budget: &TokenBudget,
    hypergraph: &Hypergraph,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    extractor: &dyn NarrativeExtractor,
    reranker: Option<&dyn Reranker>,
    response_type: Option<&str>,
    suggest: bool,
    session_history: Option<&str>,
) -> Result<RagAnswer> {
    execute_ask_inner(
        question,
        narrative_id,
        mode,
        budget,
        hypergraph,
        vector_index,
        embedder,
        extractor,
        reranker,
        response_type,
        suggest,
        session_history,
        true,
    )
}

fn execute_ask_inner(
    question: &str,
    narrative_id: Option<&str>,
    mode: &RetrievalMode,
    budget: &TokenBudget,
    hypergraph: &Hypergraph,
    vector_index: Option<&VectorIndex>,
    embedder: Option<&dyn EmbeddingProvider>,
    extractor: &dyn NarrativeExtractor,
    reranker: Option<&dyn Reranker>,
    response_type: Option<&str>,
    suggest: bool,
    session_history: Option<&str>,
    debug: bool,
) -> Result<RagAnswer> {
    // Detect retrieval method before context assembly
    let has_vector = vector_index.is_some() && embedder.is_some();
    let vi_empty = vector_index.map(|vi| vi.is_empty()).unwrap_or(true);
    let retrieval_method = if has_vector && !vi_empty {
        "vector_search"
    } else {
        "hypergraph_fallback"
    };

    let context = assemble_context(
        question,
        narrative_id,
        mode,
        hypergraph,
        vector_index,
        embedder,
        budget,
        reranker,
    )?;

    let mut system_prompt = build_rag_prompt(question, &context);

    // Prepend conversation history if session context provided
    if let Some(history) = session_history {
        if !history.is_empty() {
            system_prompt = format!("{}\n{}", history, system_prompt);
        }
    }

    // Append response type instruction if specified
    if let Some(rt) = response_type {
        system_prompt.push_str(&format!("\n\nIMPORTANT: Format your response as: {}", rt));
    }

    // Estimate tokens used
    let tokens_used = TokenBudget::estimate_tokens(&system_prompt);

    // Call LLM
    let answer = extractor.answer_question(&system_prompt, question)?;

    // Generate follow-up suggestions if requested
    let suggestions = if suggest {
        generate_suggestions(extractor, question, &answer)
    } else {
        vec![]
    };

    // Build citations from context
    let mut citations = Vec::new();
    for (entity, score) in &context.entities {
        let name = entity
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("(unnamed)");
        citations.push(Citation {
            entity_id: Some(entity.id),
            situation_id: None,
            chunk_id: None,
            excerpt: format!("{} ({:?})", name, entity.entity_type),
            score: *score,
        });
    }
    for (situation, score) in &context.situations {
        let excerpt: String = situation
            .raw_content
            .iter()
            .map(|cb| cb.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let truncated = crate::export::truncate_with_ellipsis(&excerpt, 200);
        citations.push(Citation {
            entity_id: None,
            situation_id: Some(situation.id),
            chunk_id: situation.source_chunk_id,
            excerpt: truncated,
            score: *score,
        });
    }

    let mode_str = match mode {
        RetrievalMode::Local => "local",
        RetrievalMode::Global => "global",
        RetrievalMode::Hybrid => "hybrid",
        RetrievalMode::Mix => "mix",
        RetrievalMode::Drift => "drift",
        RetrievalMode::Lazy => "lazy",
        RetrievalMode::Ppr => "ppr",
    };

    let debug_trace = if debug {
        Some(RagDebugTrace {
            retrieval_method: retrieval_method.to_string(),
            retrieved_entities: context
                .entities
                .iter()
                .map(|(e, score)| DebugEntity {
                    id: e.id,
                    name: e
                        .properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unnamed)")
                        .to_string(),
                    entity_type: format!("{:?}", e.entity_type),
                    score: *score,
                })
                .collect(),
            retrieved_situations: context
                .situations
                .iter()
                .map(|(s, score)| {
                    let excerpt: String = s
                        .raw_content
                        .iter()
                        .map(|cb| cb.content.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    DebugSituation {
                        id: s.id,
                        excerpt: crate::export::truncate_with_ellipsis(&excerpt, 150),
                        score: *score,
                    }
                })
                .collect(),
            community_summaries_used: context.community_summaries.len(),
            system_prompt: system_prompt.clone(),
            budget_limits: format!(
                "entity={}, situation={}, chunk={}, community={}, total={}",
                budget.entity_tokens,
                budget.situation_tokens,
                budget.chunk_tokens,
                budget.community_tokens,
                budget.total_tokens,
            ),
        })
    } else {
        None
    };

    Ok(RagAnswer {
        answer,
        citations,
        mode: mode_str.to_string(),
        tokens_used,
        suggestions,
        debug: debug_trace,
    })
}

/// Generate follow-up question suggestions from the LLM.
fn generate_suggestions(
    extractor: &dyn NarrativeExtractor,
    question: &str,
    answer: &str,
) -> Vec<String> {
    // Truncate answer to avoid inflating the suggestion LLM call.
    let answer_excerpt = crate::export::truncate_with_ellipsis(answer, 400);
    let suggest_prompt = format!(
        "Given this question and answer exchange, generate 3-5 relevant follow-up questions.\n\n\
         Question: {}\n\nAnswer (excerpt): {}\n\n\
         Return ONLY a JSON array of strings, e.g. [\"question1\", \"question2\", \"question3\"]. \
         No other text.",
        question, answer_excerpt
    );
    match extractor.answer_question(
        "You are a question suggestion engine. Return only a JSON array of follow-up question strings.",
        &suggest_prompt,
    ) {
        Ok(response) => parse_suggestions(&response),
        Err(_) => vec![],
    }
}

/// Parse suggestion response into a Vec of strings.
fn parse_suggestions(response: &str) -> Vec<String> {
    let cleaned = crate::ingestion::extraction::extract_json_from_response(response);
    serde_json::from_str::<Vec<String>>(&cleaned).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::community::CommunitySummary;
    use crate::ingestion::llm::MockExtractor;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;

    fn make_entity(name: &str, entity_type: EntityType, narrative_id: Option<&str>) -> Entity {
        Entity {
            id: Uuid::now_v7(),
            entity_type,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.85,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: narrative_id.map(|s| s.to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn make_situation(text: &str, narrative_id: Option<&str>) -> Situation {
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
            raw_content: vec![ContentBlock::text(text)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.75,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: narrative_id.map(|s| s.to_string()),
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

    fn make_community_summary(narrative_id: &str, community_id: usize) -> CommunitySummary {
        CommunitySummary {
            narrative_id: narrative_id.to_string(),
            community_id,
            summary: format!("Community {} overview.", community_id),
            entity_ids: vec![Uuid::nil()],
            entity_names: vec!["TestEntity".to_string()],
            key_themes: vec!["theme-a".to_string()],
            entity_count: 1,
            generated_at: Utc::now(),
            model: None,
            level: 0,
            parent_community_id: None,
            child_community_ids: vec![],
        }
    }

    #[test]
    fn test_build_rag_prompt_with_entities() {
        let entity = make_entity("Alice", EntityType::Actor, Some("test-narrative"));
        let context = RagContext {
            entities: vec![(entity, 0.9)],
            situations: vec![],
            community_summaries: vec![],
        };
        let prompt = build_rag_prompt("Who is Alice?", &context);
        assert!(prompt.contains("## Entities"));
        assert!(prompt.contains("Alice"));
        assert!(prompt.contains("## Question"));
        assert!(prompt.contains("Who is Alice?"));
    }

    #[test]
    fn test_build_rag_prompt_with_situations() {
        let situation = make_situation("The meeting took place at dawn.", Some("test"));
        let context = RagContext {
            entities: vec![],
            situations: vec![(situation, 0.8)],
            community_summaries: vec![],
        };
        let prompt = build_rag_prompt("What happened at dawn?", &context);
        assert!(prompt.contains("## Situations"));
        assert!(prompt.contains("meeting took place at dawn"));
        assert!(prompt.contains("What happened at dawn?"));
    }

    #[test]
    fn test_build_rag_prompt_empty_context() {
        let context = RagContext::default();
        let prompt = build_rag_prompt("Is anything here?", &context);
        assert!(prompt.contains("No relevant context found"));
        assert!(prompt.contains("Is anything here?"));
    }

    #[test]
    fn test_build_rag_prompt_with_communities() {
        let cs = make_community_summary("nar-1", 0);
        let context = RagContext {
            entities: vec![],
            situations: vec![],
            community_summaries: vec![(cs, 0.7)],
        };
        let prompt = build_rag_prompt("What groups exist?", &context);
        assert!(prompt.contains("## Community Overviews"));
        assert!(prompt.contains("Community 0"));
        assert!(prompt.contains("relevance: 0.70"));
        assert!(prompt.contains("theme-a"));
        assert!(!prompt.contains("No relevant context found"));
    }

    #[test]
    fn test_assemble_context_no_vector_index() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Bob", EntityType::Actor, None);
        hg.create_entity(entity).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "Who is Bob?",
            None,
            &RetrievalMode::Hybrid,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Should have found the entity via fallback listing
        assert!(!ctx.entities.is_empty());
    }

    #[test]
    fn test_assemble_context_narrative_filter_fallback() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let e1 = make_entity("Carol", EntityType::Actor, Some("narrative-a"));
        let e2 = make_entity("Dave", EntityType::Actor, Some("narrative-b"));
        hg.create_entity(e1).unwrap();
        hg.create_entity(e2).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "Who is Carol?",
            Some("narrative-a"),
            &RetrievalMode::Hybrid,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Only narrative-a entities
        assert!(ctx
            .entities
            .iter()
            .all(|(e, _)| e.narrative_id.as_deref() == Some("narrative-a")));
    }

    #[test]
    fn test_execute_ask_no_extractor_returns_error() {
        // MockExtractor.answer_question returns LlmError by default
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let extractor = MockExtractor;
        let budget = TokenBudget::default();

        let result = execute_ask(
            "test question",
            None,
            &RetrievalMode::Hybrid,
            &budget,
            &hg,
            None,
            None,
            &extractor,
            None,
            None,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_ask_with_mock() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Eve", EntityType::Actor, None);
        hg.create_entity(entity).unwrap();

        let extractor = AnswerableMockExtractor;
        let budget = TokenBudget::default();

        let result = execute_ask(
            "Tell me about Eve",
            None,
            &RetrievalMode::Hybrid,
            &budget,
            &hg,
            None,
            None,
            &extractor,
            None,
            None,
            false,
            None,
        )
        .unwrap();

        assert!(!result.answer.is_empty());
        assert_eq!(result.mode, "hybrid");
        assert!(!result.citations.is_empty());
    }

    #[test]
    fn test_summarize_entity_format() {
        let entity = make_entity("Test Actor", EntityType::Actor, Some("nar-1"));
        let summary = summarize_entity(&entity);
        assert!(summary.contains("Test Actor"));
        assert!(summary.contains("Actor"));
        assert!(summary.contains("0.85"));
    }

    #[test]
    fn test_summarize_situation_format() {
        let situation = make_situation("Something happened here.", Some("nar-1"));
        let summary = summarize_situation(&situation);
        assert!(summary.contains("Something happened here."));
        assert!(summary.contains("Scene"));
    }

    #[test]
    fn test_rag_answer_serialization() {
        let answer = RagAnswer {
            answer: "The answer is 42.".into(),
            citations: vec![Citation {
                entity_id: Some(Uuid::nil()),
                situation_id: None,
                chunk_id: None,
                excerpt: "test".into(),
                score: 0.9,
            }],
            mode: "hybrid".into(),
            tokens_used: 100,
            suggestions: vec![],
            debug: None,
        };
        let json = serde_json::to_string(&answer).unwrap();
        let parsed: RagAnswer = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.answer, "The answer is 42.");
        assert_eq!(parsed.citations.len(), 1);
    }

    #[test]
    fn test_mode_filtering_local_only_entities() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Frank", EntityType::Actor, None);
        hg.create_entity(entity).unwrap();
        let situation = make_situation("A scene occurred.", None);
        hg.create_situation(situation).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "Who is Frank?",
            None,
            &RetrievalMode::Local,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Local mode: entities + situations, no community summaries
        assert!(!ctx.entities.is_empty());
        assert!(ctx.community_summaries.is_empty());
    }

    #[test]
    fn test_mode_filtering_global_no_summaries_fallback() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Grace", EntityType::Actor, None);
        hg.create_entity(entity).unwrap();
        let situation = make_situation("Global event happened.", None);
        hg.create_situation(situation).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "What happened?",
            None,
            &RetrievalMode::Global,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Global mode with no community summaries: falls back to local retrieval
        assert!(ctx.community_summaries.is_empty());
        // Should have entities or situations from fallback
        assert!(!ctx.entities.is_empty() || !ctx.situations.is_empty());
    }

    #[test]
    fn test_mode_global_with_summaries() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());

        // Store a community summary
        let cs = make_community_summary("nar-g", 0);
        crate::analysis::community::store_summary(store.as_ref(), &cs).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "What groups exist?",
            Some("nar-g"),
            &RetrievalMode::Global,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Global mode with summaries: should return community summaries, no local
        assert!(!ctx.community_summaries.is_empty());
        assert!(ctx.entities.is_empty());
    }

    #[test]
    fn test_mode_hybrid_includes_both() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());

        let entity = make_entity("Hybrid Actor", EntityType::Actor, Some("nar-h"));
        hg.create_entity(entity).unwrap();

        let cs = make_community_summary("nar-h", 0);
        crate::analysis::community::store_summary(store.as_ref(), &cs).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "Tell me everything",
            Some("nar-h"),
            &RetrievalMode::Hybrid,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Hybrid: should have both entities and community summaries
        assert!(!ctx.entities.is_empty());
        assert!(!ctx.community_summaries.is_empty());
    }

    #[test]
    fn test_mode_drift_with_communities() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());

        let entity = make_entity("Drift Actor", EntityType::Actor, Some("nar-d"));
        let eid = entity.id;
        hg.create_entity(entity).unwrap();

        // Store a community summary that references the entity
        let mut cs = make_community_summary("nar-d", 0);
        cs.entity_ids = vec![eid];
        cs.summary = "A group of actors involved in the main plot.".to_string();
        cs.key_themes = vec!["plot".to_string(), "conflict".to_string()];
        crate::analysis::community::store_summary(store.as_ref(), &cs).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "What is the main plot conflict?",
            Some("nar-d"),
            &RetrievalMode::Drift,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // DRIFT should retrieve community summaries and drill down to entities
        assert!(!ctx.community_summaries.is_empty());
        assert!(!ctx.entities.is_empty());
    }

    #[test]
    fn test_mode_drift_no_communities_fallback() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Fallback Actor", EntityType::Actor, None);
        hg.create_entity(entity).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "What happened?",
            None,
            &RetrievalMode::Drift,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // No communities → falls back to local retrieval
        assert!(ctx.community_summaries.is_empty());
        assert!(!ctx.entities.is_empty());
    }

    #[test]
    fn test_mode_mix_retrieves_both() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());

        let entity = make_entity("Mix Actor", EntityType::Actor, Some("nar-m"));
        hg.create_entity(entity).unwrap();

        let cs = make_community_summary("nar-m", 0);
        crate::analysis::community::store_summary(store.as_ref(), &cs).unwrap();

        let budget = TokenBudget::default();
        let ctx = assemble_context(
            "How did the political movement evolve?",
            Some("nar-m"),
            &RetrievalMode::Mix,
            &hg,
            None,
            None,
            &budget,
            None,
        )
        .unwrap();

        // Mix mode with high-level keywords ("political", "movement", "evolve")
        // should retrieve community summaries + local entities
        assert!(!ctx.entities.is_empty());
        assert!(!ctx.community_summaries.is_empty());
    }

    #[test]
    fn test_rag_context_with_communities() {
        let cs = make_community_summary("nar-ctx", 0);
        let entity = make_entity("CtxActor", EntityType::Actor, Some("nar-ctx"));
        let ctx = RagContext {
            entities: vec![(entity, 0.8)],
            situations: vec![],
            community_summaries: vec![(cs, 0.7)],
        };
        let prompt = build_rag_prompt("test?", &ctx);
        assert!(prompt.contains("Community Overviews"));
        assert!(prompt.contains("Entities"));
        assert!(!prompt.contains("No relevant context found"));
    }

    #[test]
    fn test_citation_from_entity() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Helen", EntityType::Actor, None);
        let eid = entity.id;
        hg.create_entity(entity).unwrap();

        let extractor = AnswerableMockExtractor;
        let budget = TokenBudget::default();
        let result = execute_ask(
            "Tell me about Helen",
            None,
            &RetrievalMode::Local,
            &budget,
            &hg,
            None,
            None,
            &extractor,
            None,
            None,
            false,
            None,
        )
        .unwrap();

        assert!(result.citations.iter().any(|c| c.entity_id == Some(eid)));
    }

    /// A mock extractor that supports answer_question.
    struct AnswerableMockExtractor;

    impl NarrativeExtractor for AnswerableMockExtractor {
        fn extract_narrative(
            &self,
            _chunk: &crate::ingestion::chunker::TextChunk,
        ) -> Result<crate::ingestion::extraction::NarrativeExtraction> {
            Ok(crate::ingestion::extraction::NarrativeExtraction {
                entities: vec![],
                situations: vec![],
                participations: vec![],
                causal_links: vec![],
                temporal_relations: vec![],
            })
        }

        fn answer_question(&self, _system_prompt: &str, question: &str) -> Result<String> {
            Ok(format!(
                "Based on the available context, here is the answer to: {}",
                question
            ))
        }
    }

    #[test]
    fn test_execute_ask_with_reranker() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        // Create two entities — reranker should reorder them
        let e1 = make_entity("Alpha", EntityType::Actor, None);
        let e2 = make_entity("Beta", EntityType::Actor, None);
        hg.create_entity(e1).unwrap();
        hg.create_entity(e2).unwrap();

        let extractor = AnswerableMockExtractor;
        let budget = TokenBudget::default();
        let reranker = crate::query::reranker::TermOverlapReranker::new();

        let result = execute_ask(
            "Tell me about Beta actor",
            None,
            &RetrievalMode::Local,
            &budget,
            &hg,
            None,
            None,
            &extractor,
            Some(&reranker),
            None,
            false,
            None,
        )
        .unwrap();

        assert!(!result.answer.is_empty());
        assert!(!result.citations.is_empty());
    }

    #[test]
    fn test_execute_ask_no_reranker_backward_compat() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);
        let entity = make_entity("Zeta", EntityType::Actor, None);
        hg.create_entity(entity).unwrap();

        let extractor = AnswerableMockExtractor;
        let budget = TokenBudget::default();

        // None reranker should work exactly as before
        let result = execute_ask(
            "Tell me about Zeta",
            None,
            &RetrievalMode::Hybrid,
            &budget,
            &hg,
            None,
            None,
            &extractor,
            None,
            None,
            false,
            None,
        )
        .unwrap();

        assert!(!result.answer.is_empty());
        assert_eq!(result.mode, "hybrid");
    }
}
