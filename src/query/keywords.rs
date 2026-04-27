//! Keyword extraction from natural language questions.
//!
//! Provides both LLM-powered and heuristic-based extraction of
//! high-level (concepts, themes) and low-level (names, dates, places)
//! keywords from user questions for use in RAG retrieval.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ingestion::llm::NarrativeExtractor;

/// Keywords extracted from a natural language question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedKeywords {
    /// High-level concepts, themes, abstract ideas.
    pub high_level: Vec<String>,
    /// Specific names, dates, places, facts.
    pub low_level: Vec<String>,
}

/// Stop words to filter out during heuristic extraction.
const STOP_WORDS: &[&str] = &[
    "the", "what", "who", "how", "why", "when", "where", "did", "does", "was", "were", "are", "is",
    "has", "had", "have", "and", "but", "for", "with", "about", "from", "that", "this", "which",
    "there", "been", "being", "can", "could", "would", "should", "will", "shall", "may", "might",
    "not", "its", "his", "her", "their", "our", "your", "any", "all", "each", "some", "than",
    "then", "also", "just", "into", "over",
];

/// Extract keywords from a question using an LLM.
///
/// Sends a structured prompt to the extractor and parses the JSON response.
/// Falls back to heuristic extraction if the LLM response cannot be parsed.
pub fn extract_keywords_llm(
    question: &str,
    extractor: &dyn NarrativeExtractor,
) -> Result<ExtractedKeywords> {
    let system = "You are a keyword extraction assistant. Given a question, extract:\n\
        1. High-level keywords: abstract concepts, themes, broad topics\n\
        2. Low-level keywords: specific names, dates, places, concrete facts\n\
        Respond ONLY with JSON: {\"high_level\": [...], \"low_level\": [...]}";

    let response = extractor.answer_question(system, question)?;

    let cleaned = crate::ingestion::extraction::extract_json_from_response(&response);
    if let Ok(kw) = serde_json::from_str::<ExtractedKeywords>(&cleaned) {
        return Ok(kw);
    }

    // Fall back to heuristic
    Ok(extract_keywords_heuristic(question))
}

/// Extract keywords using simple heuristics (no LLM needed).
///
/// Splits the question into words, filters stop words, and classifies
/// capitalized words as low-level (proper nouns) and others as high-level.
pub fn extract_keywords_heuristic(question: &str) -> ExtractedKeywords {
    let mut high_level = Vec::new();
    let mut low_level = Vec::new();

    for word in question.split_whitespace() {
        // Clean punctuation from edges
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
            .collect();

        if clean.len() <= 2 {
            continue;
        }

        let lower = clean.to_lowercase();
        if STOP_WORDS.contains(&lower.as_str()) {
            continue;
        }

        // Capitalized words (potential proper nouns) go to low-level
        let is_capitalized = word
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        // Skip if first word of question (often capitalized naturally)
        let is_first_word = question.trim_start().starts_with(word);

        if is_capitalized && !is_first_word {
            low_level.push(lower);
        } else {
            high_level.push(lower);
        }
    }

    ExtractedKeywords {
        high_level,
        low_level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_proper_nouns() {
        let kw = extract_keywords_heuristic("What happened to John in London?");
        assert!(kw.low_level.contains(&"john".to_string()));
        assert!(kw.low_level.contains(&"london".to_string()));
    }

    #[test]
    fn test_heuristic_stop_words_filtered() {
        let kw = extract_keywords_heuristic("What is the relationship between them?");
        assert!(!kw.high_level.contains(&"what".to_string()));
        assert!(!kw.high_level.contains(&"the".to_string()));
        assert!(!kw.high_level.contains(&"is".to_string()));
    }

    #[test]
    fn test_heuristic_empty() {
        let kw = extract_keywords_heuristic("");
        assert!(kw.high_level.is_empty());
        assert!(kw.low_level.is_empty());
    }

    #[test]
    fn test_heuristic_all_stop_words() {
        let kw = extract_keywords_heuristic("What is the");
        assert!(kw.high_level.is_empty());
        assert!(kw.low_level.is_empty());
    }

    #[test]
    fn test_heuristic_mixed() {
        let kw =
            extract_keywords_heuristic("How did Alice influence the political movement in Paris?");
        assert!(kw.low_level.contains(&"alice".to_string()));
        assert!(kw.low_level.contains(&"paris".to_string()));
        assert!(kw
            .high_level
            .iter()
            .any(|w| w == "political" || w == "influence" || w == "movement"));
    }

    #[test]
    fn test_extracted_keywords_serialization() {
        let kw = ExtractedKeywords {
            high_level: vec!["conflict".into(), "power".into()],
            low_level: vec!["napoleon".into(), "waterloo".into()],
        };
        let json = serde_json::to_string(&kw).unwrap();
        let parsed: ExtractedKeywords = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.high_level.len(), 2);
        assert_eq!(parsed.low_level.len(), 2);
    }

    #[test]
    fn test_heuristic_punctuation_handling() {
        let kw = extract_keywords_heuristic("What about Caesar's assassination?");
        // Should handle possessives and punctuation
        assert!(
            kw.low_level.contains(&"caesar's".to_string())
                || kw.high_level.iter().any(|w| w.contains("caesar"))
        );
    }
}
