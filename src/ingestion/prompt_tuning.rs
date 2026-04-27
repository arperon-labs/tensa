//! Prompt auto-tuning for domain-adapted extraction.
//!
//! Samples chunks from a narrative, sends them to the LLM with a meta-prompt
//! asking for domain-specific extraction guidelines, and stores the result
//! in KV at `pt/{narrative_id}` for use during ingestion.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::llm::NarrativeExtractor;
use crate::store::KVStore;

/// KV prefix for tuned prompts.
const TUNED_PROMPT_PREFIX: &str = "pt/";

/// Default maximum number of chunks to sample for prompt tuning.
pub const DEFAULT_MAX_SAMPLE_CHUNKS: usize = 5;

/// Default maximum character length per sample chunk.
pub const DEFAULT_SAMPLE_CHAR_LIMIT: usize = 500;

/// A domain-adapted extraction prompt generated from narrative data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunedPrompt {
    /// Narrative this prompt was tuned for.
    pub narrative_id: String,
    /// The generated domain-specific extraction instructions.
    pub prompt_text: String,
    /// Short description of the domain/genre detected.
    pub domain_description: String,
    /// Recommended entity types for this domain.
    pub entity_types: Vec<String>,
    /// When the prompt was generated.
    pub generated_at: DateTime<Utc>,
    /// Model used for generation.
    pub model: Option<String>,
}

/// Store a tuned prompt in KV.
pub fn store_tuned_prompt(store: &dyn KVStore, prompt: &TunedPrompt) -> Result<()> {
    let key = format!("{}{}", TUNED_PROMPT_PREFIX, prompt.narrative_id);
    let value = serde_json::to_vec(prompt)?;
    store.put(key.as_bytes(), &value)?;
    Ok(())
}

/// Get a tuned prompt for a narrative.
pub fn get_tuned_prompt(store: &dyn KVStore, narrative_id: &str) -> Result<Option<TunedPrompt>> {
    let key = format!("{}{}", TUNED_PROMPT_PREFIX, narrative_id);
    match store.get(key.as_bytes())? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List all tuned prompts.
pub fn list_tuned_prompts(store: &dyn KVStore) -> Result<Vec<TunedPrompt>> {
    let pairs = store.prefix_scan(TUNED_PROMPT_PREFIX.as_bytes())?;
    let mut prompts = Vec::with_capacity(pairs.len());
    for (_key, value) in pairs {
        prompts.push(serde_json::from_slice(&value)?);
    }
    Ok(prompts)
}

/// Delete a tuned prompt for a narrative.
pub fn delete_tuned_prompt(store: &dyn KVStore, narrative_id: &str) -> Result<bool> {
    let key = format!("{}{}", TUNED_PROMPT_PREFIX, narrative_id);
    let existed = store.get(key.as_bytes())?.is_some();
    if existed {
        store.transaction(vec![crate::store::TxnOp::Delete(key.into_bytes())])?;
    }
    Ok(existed)
}

/// Meta-prompt sent to the LLM to generate domain-adapted extraction guidelines.
const TUNING_META_PROMPT: &str = r#"You are a prompt-tuning engine. Analyze the following text samples from a narrative corpus and generate domain-specific extraction guidelines.

Your output MUST be a JSON object with these fields:
{
  "domain_description": "A 1-2 sentence description of the domain/genre (e.g. 'Victorian Gothic horror novel', 'Cold War intelligence reports')",
  "entity_types": ["list", "of", "recommended", "entity", "types"],
  "extraction_guidelines": "Detailed extraction instructions tailored to this domain. Include: (1) what kinds of entities are most important, (2) what narrative patterns to look for, (3) domain-specific terminology to watch for, (4) how to handle ambiguity typical of this genre, (5) what confidence levels are appropriate for different types of claims in this domain."
}

TEXT SAMPLES:"#;

/// Run prompt tuning for a narrative by sampling chunks and generating domain-adapted guidelines.
/// Run prompt tuning with default settings.
pub fn tune_prompts(
    store: &dyn KVStore,
    extractor: &dyn NarrativeExtractor,
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<TunedPrompt> {
    tune_prompts_with_config(
        store,
        extractor,
        hypergraph,
        narrative_id,
        DEFAULT_MAX_SAMPLE_CHUNKS,
        DEFAULT_SAMPLE_CHAR_LIMIT,
    )
}

/// Run prompt tuning with configurable sample count and char limit.
pub fn tune_prompts_with_config(
    store: &dyn KVStore,
    extractor: &dyn NarrativeExtractor,
    hypergraph: &Hypergraph,
    narrative_id: &str,
    max_sample_chunks: usize,
    sample_char_limit: usize,
) -> Result<TunedPrompt> {
    // Sample chunks from the narrative
    let chunks = hypergraph.list_chunks_by_narrative(narrative_id)?;
    if chunks.is_empty() {
        return Err(TensaError::NotFound(format!(
            "No chunks found for narrative '{}'. Ingest text before tuning prompts.",
            narrative_id
        )));
    }

    // Sample evenly across the corpus
    let sample_count = chunks.len().min(max_sample_chunks);
    let step = if chunks.len() <= sample_count {
        1
    } else {
        chunks.len() / sample_count
    };
    let sampled: Vec<&str> = chunks
        .iter()
        .step_by(step)
        .take(sample_count)
        .map(|c| c.text.as_str())
        .collect();

    // Build the meta-prompt with samples
    let mut prompt = TUNING_META_PROMPT.to_string();
    for (i, sample) in sampled.iter().enumerate() {
        // Truncate each sample to stay within token limits
        let truncated = if sample.len() > sample_char_limit {
            &sample[..sample_char_limit]
        } else {
            sample
        };
        prompt.push_str(&format!("\n\n--- Sample {} ---\n{}", i + 1, truncated));
    }

    // Call LLM
    let response = extractor.answer_question(
        &prompt,
        "Generate domain-specific extraction guidelines based on these samples.",
    )?;

    // Parse response
    let (domain_description, entity_types, extraction_guidelines) =
        parse_tuning_response(&response);

    let tuned = TunedPrompt {
        narrative_id: narrative_id.to_string(),
        prompt_text: extraction_guidelines,
        domain_description,
        entity_types,
        generated_at: Utc::now(),
        model: extractor.model_name(),
    };

    // Store in KV
    store_tuned_prompt(store, &tuned)?;

    Ok(tuned)
}

/// Parse the LLM tuning response, with lenient fallback.
fn parse_tuning_response(response: &str) -> (String, Vec<String>, String) {
    // Reuse the project-wide JSON extraction utility (handles thinking tags, fences, etc.)
    let json_str = crate::ingestion::extraction::extract_json_from_response(response);
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
        let domain = val
            .get("domain_description")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown domain")
            .to_string();
        let entity_types = val
            .get("entity_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let guidelines = val
            .get("extraction_guidelines")
            .and_then(|v| v.as_str())
            .unwrap_or(response)
            .to_string();
        (domain, entity_types, guidelines)
    } else {
        // Fallback: use the raw response as guidelines
        ("Unknown domain".to_string(), vec![], response.to_string())
    }
}

/// Build the extraction system prompt, prepending domain-specific tuning if available.
pub fn build_tuned_system_prompt(base_prompt: &str, tuned: Option<&TunedPrompt>) -> String {
    match tuned {
        Some(t) => {
            format!(
                "DOMAIN CONTEXT: {}\n\nDOMAIN-SPECIFIC GUIDELINES:\n{}\n\n{}",
                t.domain_description, t.prompt_text, base_prompt
            )
        }
        None => base_prompt.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn make_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    #[test]
    fn test_tuned_prompt_kv_roundtrip() {
        let store = make_store();
        let prompt = TunedPrompt {
            narrative_id: "test-narrative".to_string(),
            prompt_text: "Focus on character motivations and power dynamics.".to_string(),
            domain_description: "Political thriller novel".to_string(),
            entity_types: vec!["Actor".into(), "Organization".into(), "Location".into()],
            generated_at: Utc::now(),
            model: Some("test-model".to_string()),
        };

        store_tuned_prompt(store.as_ref(), &prompt).unwrap();
        let loaded = get_tuned_prompt(store.as_ref(), "test-narrative")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.narrative_id, "test-narrative");
        assert_eq!(loaded.domain_description, "Political thriller novel");
        assert_eq!(loaded.entity_types.len(), 3);
    }

    #[test]
    fn test_tuned_prompt_not_found() {
        let store = make_store();
        let result = get_tuned_prompt(store.as_ref(), "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_list_tuned_prompts() {
        let store = make_store();
        for id in &["alpha", "beta", "gamma"] {
            let prompt = TunedPrompt {
                narrative_id: id.to_string(),
                prompt_text: format!("Guidelines for {}", id),
                domain_description: format!("Domain {}", id),
                entity_types: vec![],
                generated_at: Utc::now(),
                model: None,
            };
            store_tuned_prompt(store.as_ref(), &prompt).unwrap();
        }
        let all = list_tuned_prompts(store.as_ref()).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_delete_tuned_prompt() {
        let store = make_store();
        let prompt = TunedPrompt {
            narrative_id: "to-delete".to_string(),
            prompt_text: "test".to_string(),
            domain_description: "test".to_string(),
            entity_types: vec![],
            generated_at: Utc::now(),
            model: None,
        };
        store_tuned_prompt(store.as_ref(), &prompt).unwrap();
        assert!(delete_tuned_prompt(store.as_ref(), "to-delete").unwrap());
        assert!(!delete_tuned_prompt(store.as_ref(), "to-delete").unwrap());
        assert!(get_tuned_prompt(store.as_ref(), "to-delete")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_parse_tuning_response_json() {
        let response = r#"```json
{
  "domain_description": "Victorian Gothic novel",
  "entity_types": ["Actor", "Location", "Artifact"],
  "extraction_guidelines": "Focus on atmospheric descriptions and character psychology."
}
```"#;
        let (domain, types, guidelines) = parse_tuning_response(response);
        assert_eq!(domain, "Victorian Gothic novel");
        assert_eq!(types, vec!["Actor", "Location", "Artifact"]);
        assert!(guidelines.contains("atmospheric"));
    }

    #[test]
    fn test_parse_tuning_response_fallback() {
        let response = "Just some plain text guidelines without JSON";
        let (domain, types, guidelines) = parse_tuning_response(response);
        assert_eq!(domain, "Unknown domain");
        assert!(types.is_empty());
        assert_eq!(guidelines, response);
    }

    #[test]
    fn test_build_tuned_system_prompt_with_tuning() {
        let tuned = TunedPrompt {
            narrative_id: "test".to_string(),
            prompt_text: "Watch for spy terminology.".to_string(),
            domain_description: "Cold War espionage".to_string(),
            entity_types: vec![],
            generated_at: Utc::now(),
            model: None,
        };
        let result = build_tuned_system_prompt("BASE PROMPT", Some(&tuned));
        assert!(result.starts_with("DOMAIN CONTEXT: Cold War espionage"));
        assert!(result.contains("Watch for spy terminology."));
        assert!(result.ends_with("BASE PROMPT"));
    }

    #[test]
    fn test_build_tuned_system_prompt_without_tuning() {
        let result = build_tuned_system_prompt("BASE PROMPT", None);
        assert_eq!(result, "BASE PROMPT");
    }
}
