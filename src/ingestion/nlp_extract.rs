//! NLP-based fast extraction without LLM calls.
//!
//! Uses regex-based NER and sentence co-occurrence to extract entities
//! and situations from text. Much cheaper than LLM extraction, suitable
//! for bulk ingestion or initial exploration passes.

use crate::error::Result;
use crate::ingestion::chunker::TextChunk;
use crate::ingestion::extraction::{
    ExtractedEntity, ExtractedParticipation, ExtractedSituation, NarrativeExtraction,
};
use crate::ingestion::llm::NarrativeExtractor;
use crate::types::{ContentBlock, EntityType, NarrativeLevel, Role};

/// Regex-based fast extractor implementing `NarrativeExtractor`.
///
/// Extracts entities by recognizing capitalized multi-word sequences (actors),
/// organization suffixes (Inc, Corp, Ltd), and location patterns.
/// Creates situations from sentences and links co-occurring entities.
pub struct NlpFastExtractor {
    /// Maximum situations to extract per chunk. Default: 20.
    pub max_situations_per_chunk: usize,
}

impl Default for NlpFastExtractor {
    fn default() -> Self {
        Self {
            max_situations_per_chunk: 20,
        }
    }
}

impl NlpFastExtractor {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::default()
    }
}

impl NarrativeExtractor for NlpFastExtractor {
    fn extract_narrative(&self, chunk: &TextChunk) -> Result<NarrativeExtraction> {
        self.extract_with_context(chunk, &[])
    }

    fn extract_with_context(
        &self,
        chunk: &TextChunk,
        _known_entities: &[String],
    ) -> Result<NarrativeExtraction> {
        let text = &chunk.text;
        let sentences = split_sentences(text);
        let mut entities: Vec<ExtractedEntity> = Vec::new();
        let mut situations: Vec<ExtractedSituation> = Vec::new();
        let mut participations: Vec<ExtractedParticipation> = Vec::new();
        let mut entity_names_seen: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Extract entities from the full text
        let entity_candidates = extract_entity_candidates(text);
        for (name, entity_type) in &entity_candidates {
            if entity_names_seen.contains(name) {
                continue;
            }
            entity_names_seen.insert(name.clone());
            entities.push(ExtractedEntity {
                name: name.clone(),
                aliases: vec![],
                entity_type: entity_type.clone(),
                properties: serde_json::json!({}),
                confidence: 0.5,
            });
        }

        // Create a situation per sentence with entity co-occurrence
        for (_sent_i, sentence) in sentences.iter().enumerate() {
            let trimmed = sentence.trim();
            if trimmed.len() < 10 {
                continue;
            }

            // Find which entities appear in this sentence
            let sentence_entities: Vec<&str> = entity_candidates
                .iter()
                .filter(|(name, _)| sentence.contains(name.as_str()))
                .map(|(name, _)| name.as_str())
                .collect();

            if sentence_entities.is_empty() {
                continue;
            }

            let situation_idx = situations.len();
            situations.push(ExtractedSituation {
                name: None,
                description: if trimmed.len() > 150 {
                    format!("{}...", &trimmed[..150])
                } else {
                    trimmed.to_string()
                },
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Scene,
                content_blocks: vec![ContentBlock::text(trimmed)],
                confidence: 0.5,
                text_start: None,
                text_end: None,
            });

            // Create participation for each entity in the sentence
            for entity_name in sentence_entities {
                participations.push(ExtractedParticipation {
                    entity_name: entity_name.to_string(),
                    situation_index: situation_idx,
                    role: Role::Protagonist,
                    action: None,
                    confidence: 0.5,
                });
            }

            // Limit situations per chunk
            if situations.len() >= self.max_situations_per_chunk {
                break;
            }
        }

        Ok(NarrativeExtraction {
            entities,
            situations,
            participations,
            causal_links: vec![],
            temporal_relations: vec![],
        })
    }

    fn model_name(&self) -> Option<String> {
        Some("nlp-fast".to_string())
    }
}

/// Split text into sentences (simple heuristic).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

/// Organization suffixes for entity type detection.
const ORG_SUFFIXES: &[&str] = &[
    "Inc",
    "Corp",
    "Ltd",
    "LLC",
    "Co",
    "Agency",
    "Bureau",
    "Institute",
    "Foundation",
    "Association",
    "Department",
    "Ministry",
    "Committee",
];

/// Extract entity candidates from text using capitalization and pattern heuristics.
fn extract_entity_candidates(text: &str) -> Vec<(String, EntityType)> {
    let mut candidates: Vec<(String, EntityType)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Find capitalized multi-word sequences (potential actors/locations/orgs)
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
        if word.len() >= 2
            && word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        {
            // Start of a potential entity name
            let mut name_parts = vec![word.to_string()];
            let mut j = i + 1;
            while j < words.len() {
                let next = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                if next.len() >= 2
                    && next
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    name_parts.push(next.to_string());
                    j += 1;
                } else {
                    break;
                }
            }

            // Skip single common words that are just sentence starters
            let name = name_parts.join(" ");
            if name_parts.len() >= 2 || !is_common_word(&name) {
                if !seen.contains(&name) {
                    let entity_type = classify_entity_type(&name);
                    seen.insert(name.clone());
                    candidates.push((name, entity_type));
                }
            }

            i = j;
        } else {
            i += 1;
        }
    }

    candidates
}

/// Classify entity type based on name patterns.
fn classify_entity_type(name: &str) -> EntityType {
    for suffix in ORG_SUFFIXES {
        if name.ends_with(suffix) {
            return EntityType::Organization;
        }
    }
    EntityType::Actor
}

/// Check if a word is a common English word (not an entity).
fn is_common_word(word: &str) -> bool {
    matches!(
        word.to_lowercase().as_str(),
        "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "there"
            | "here"
            | "then"
            | "when"
            | "where"
            | "what"
            | "which"
            | "who"
            | "how"
            | "but"
            | "and"
            | "for"
            | "not"
            | "yet"
            | "nor"
            | "its"
            | "his"
            | "her"
            | "our"
            | "their"
            | "some"
            | "any"
            | "all"
            | "each"
            | "every"
            | "both"
            | "few"
            | "more"
            | "most"
            | "other"
            | "another"
            | "such"
            | "many"
            | "much"
            | "after"
            | "before"
            | "during"
            | "since"
            | "until"
            | "however"
            | "therefore"
            | "meanwhile"
            | "furthermore"
            | "moreover"
            | "nevertheless"
            | "indeed"
            | "perhaps"
            | "chapter"
            | "part"
            | "book"
            | "section"
            | "volume"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(text: &str) -> TextChunk {
        TextChunk {
            chunk_id: 0,
            text: text.to_string(),
            chapter: None,
            start_offset: 0,
            end_offset: text.len(),
            overlap_prefix: String::new(),
        }
    }

    #[test]
    fn test_extract_capitalized_names() {
        let candidates = extract_entity_candidates(
            "John Smith met with Alice Johnson at the Central Park today. They discussed the project.",
        );
        let names: Vec<&str> = candidates.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"John Smith"),
            "Missing John Smith in {:?}",
            names
        );
        assert!(
            names.contains(&"Alice Johnson"),
            "Missing Alice Johnson in {:?}",
            names
        );
        assert!(
            names.contains(&"Central Park"),
            "Missing Central Park in {:?}",
            names
        );
    }

    #[test]
    fn test_classify_organization() {
        assert_eq!(classify_entity_type("Acme Corp"), EntityType::Organization);
        assert_eq!(classify_entity_type("John Smith"), EntityType::Actor);
    }

    #[test]
    fn test_skip_common_words() {
        assert!(is_common_word("The"));
        assert!(is_common_word("However"));
        assert!(!is_common_word("Alice"));
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. How are you? Fine!");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "Fine!");
    }

    #[test]
    fn test_nlp_fast_extractor() {
        let extractor = NlpFastExtractor::new();
        let chunk = make_chunk(
            "Robert Parker arrived at the Grand Hotel in London. \
             Sarah Mitchell was already waiting in the lobby. \
             Robert Parker greeted Sarah Mitchell warmly.",
        );
        let result = extractor.extract_narrative(&chunk).unwrap();

        assert!(!result.entities.is_empty());
        assert!(!result.situations.is_empty());
        assert!(!result.participations.is_empty());

        // Check entity names
        let names: Vec<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Robert Parker"));
        assert!(names.contains(&"Sarah Mitchell"));

        // All confidence should be 0.5 (inferred)
        for e in &result.entities {
            assert!((e.confidence - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_nlp_fast_cooccurrence() {
        let extractor = NlpFastExtractor::new();
        let chunk =
            make_chunk("Alice Brown and Bob Carter met at the cafe. They discussed the plan.");
        let result = extractor.extract_narrative(&chunk).unwrap();

        // Should create situations linking entities that appear in the same sentence
        assert!(!result.situations.is_empty());
        // First situation should have both Alice and Bob as participants
        let first_sit_parts: Vec<&str> = result
            .participations
            .iter()
            .filter(|p| p.situation_index == 0)
            .map(|p| p.entity_name.as_str())
            .collect();
        assert!(first_sit_parts.contains(&"Alice Brown"));
        assert!(first_sit_parts.contains(&"Bob Carter"));
    }

    #[test]
    fn test_nlp_fast_empty_text() {
        let extractor = NlpFastExtractor::new();
        let chunk = make_chunk("");
        let result = extractor.extract_narrative(&chunk).unwrap();
        assert!(result.entities.is_empty());
        assert!(result.situations.is_empty());
    }

    #[test]
    fn test_nlp_fast_model_name() {
        let extractor = NlpFastExtractor::new();
        assert_eq!(extractor.model_name(), Some("nlp-fast".to_string()));
    }
}
