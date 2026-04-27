//! Token budget system for RAG context assembly.
//!
//! Allocates a fixed token budget across content categories (entities, situations,
//! chunks, communities), selecting the highest-scored items that fit within both
//! per-category and total limits.

use serde::{Deserialize, Serialize};

/// Category of content item for budget allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ItemCategory {
    /// Entity descriptions and metadata.
    Entity,
    /// Situation summaries and raw content.
    Situation,
    /// Raw text chunks from ingestion.
    Chunk,
    /// Community/cluster summaries.
    Community,
}

/// A content item scored by relevance for budget allocation.
#[derive(Debug, Clone)]
pub struct ScoredItem {
    /// Which budget category this item draws from.
    pub category: ItemCategory,
    /// The textual content of this item.
    pub content: String,
    /// Relevance score (higher = more relevant).
    pub score: f32,
    /// Estimated token count for this item.
    pub token_estimate: usize,
}

/// Token budget configuration for RAG context assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Maximum tokens allocated to entity descriptions.
    pub entity_tokens: usize,
    /// Maximum tokens allocated to situation summaries.
    pub situation_tokens: usize,
    /// Maximum tokens allocated to raw text chunks.
    pub chunk_tokens: usize,
    /// Maximum tokens allocated to community summaries.
    pub community_tokens: usize,
    /// Hard cap on total tokens across all categories.
    pub total_tokens: usize,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            entity_tokens: 4000,
            situation_tokens: 4000,
            chunk_tokens: 4000,
            community_tokens: 2000,
            total_tokens: 16000,
        }
    }
}

impl TokenBudget {
    /// Estimate token count from text using the chars/4 heuristic.
    pub fn estimate_tokens(text: &str) -> usize {
        (text.len() + 3) / 4 // ceiling division
    }

    /// Allocate items within budget, returning those that fit.
    ///
    /// Items are selected in descending score order. Each item must fit within
    /// both its per-category limit and the remaining total budget. Items that
    /// exceed either limit are skipped (not truncated).
    pub fn allocate(&self, items: &[ScoredItem]) -> Vec<ScoredItem> {
        let mut sorted: Vec<_> = items.to_vec();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result = Vec::new();
        let mut entity_used = 0usize;
        let mut situation_used = 0usize;
        let mut chunk_used = 0usize;
        let mut community_used = 0usize;
        let mut total_used = 0usize;

        for item in &sorted {
            let tokens = item.token_estimate;
            if total_used + tokens > self.total_tokens {
                continue;
            }

            let (used, limit) = match item.category {
                ItemCategory::Entity => (&mut entity_used, self.entity_tokens),
                ItemCategory::Situation => (&mut situation_used, self.situation_tokens),
                ItemCategory::Chunk => (&mut chunk_used, self.chunk_tokens),
                ItemCategory::Community => (&mut community_used, self.community_tokens),
            };

            if *used + tokens > limit {
                continue;
            }

            *used += tokens;
            total_used += tokens;
            result.push(item.clone());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(cat: ItemCategory, content: &str, score: f32, tokens: usize) -> ScoredItem {
        ScoredItem {
            category: cat,
            content: content.to_string(),
            score,
            token_estimate: tokens,
        }
    }

    #[test]
    fn test_allocate_fits_within_total() {
        let budget = TokenBudget {
            total_tokens: 100,
            entity_tokens: 100,
            situation_tokens: 100,
            chunk_tokens: 100,
            community_tokens: 100,
        };
        let items = vec![
            item(ItemCategory::Entity, "a", 1.0, 50),
            item(ItemCategory::Entity, "b", 0.9, 50),
            item(ItemCategory::Entity, "c", 0.8, 50), // exceeds total
        ];
        let result = budget.allocate(&items);
        assert_eq!(result.len(), 2);
        let total: usize = result.iter().map(|i| i.token_estimate).sum();
        assert!(total <= 100);
    }

    #[test]
    fn test_allocate_respects_category_limit() {
        let budget = TokenBudget {
            total_tokens: 10000,
            entity_tokens: 80,
            situation_tokens: 10000,
            chunk_tokens: 10000,
            community_tokens: 10000,
        };
        let items = vec![
            item(ItemCategory::Entity, "a", 1.0, 50),
            item(ItemCategory::Entity, "b", 0.9, 50), // exceeds entity_tokens
            item(ItemCategory::Situation, "c", 0.8, 50),
        ];
        let result = budget.allocate(&items);
        let entity_count = result
            .iter()
            .filter(|i| i.category == ItemCategory::Entity)
            .count();
        assert_eq!(entity_count, 1);
        assert_eq!(result.len(), 2); // 1 entity + 1 situation
    }

    #[test]
    fn test_allocate_by_score_order() {
        let budget = TokenBudget {
            total_tokens: 100,
            entity_tokens: 100,
            ..TokenBudget::default()
        };
        let items = vec![
            item(ItemCategory::Entity, "low", 0.1, 60),
            item(ItemCategory::Entity, "high", 0.9, 60),
        ];
        let result = budget.allocate(&items);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "high");
    }

    #[test]
    fn test_allocate_empty_input() {
        let budget = TokenBudget::default();
        let result = budget.allocate(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(TokenBudget::estimate_tokens(""), 0);
        assert_eq!(TokenBudget::estimate_tokens("a"), 1);
        assert_eq!(TokenBudget::estimate_tokens("abcd"), 1);
        assert_eq!(TokenBudget::estimate_tokens("abcde"), 2);
        // 12 chars -> 3 tokens
        assert_eq!(TokenBudget::estimate_tokens("abcdefghijkl"), 3);
    }

    #[test]
    fn test_allocate_mixed_categories() {
        let budget = TokenBudget {
            entity_tokens: 100,
            situation_tokens: 100,
            chunk_tokens: 100,
            community_tokens: 100,
            total_tokens: 300,
        };
        let items = vec![
            item(ItemCategory::Entity, "e1", 1.0, 80),
            item(ItemCategory::Situation, "s1", 0.9, 80),
            item(ItemCategory::Chunk, "c1", 0.8, 80),
            item(ItemCategory::Community, "co1", 0.7, 80), // exceeds total (240+80=320)
            item(ItemCategory::Entity, "e2", 0.6, 30),     // exceeds entity limit (80+30=110)
        ];
        let result = budget.allocate(&items);
        assert_eq!(result.len(), 3);
        let total: usize = result.iter().map(|i| i.token_estimate).sum();
        assert!(total <= 300);
        // Verify categories: entity, situation, chunk
        assert!(result.iter().any(|i| i.category == ItemCategory::Entity));
        assert!(result.iter().any(|i| i.category == ItemCategory::Situation));
        assert!(result.iter().any(|i| i.category == ItemCategory::Chunk));
    }
}
