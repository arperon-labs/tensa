//! Ranking metrics: MRR, Hits@k, NDCG.
//!
//! Used for temporal knowledge graph link prediction (ICEWS, GDELT).

use serde::{Deserialize, Serialize};

/// Standard ranking metrics for link prediction benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingMetrics {
    pub mrr: f64,
    pub hits_at_1: f64,
    pub hits_at_3: f64,
    pub hits_at_10: f64,
}

impl RankingMetrics {
    /// Compute ranking metrics from a list of 1-based ranks.
    ///
    /// Each rank represents the position of the correct entity in the
    /// sorted candidate list for one test query. Rank 1 = correct entity
    /// was top-ranked.
    pub fn from_ranks(ranks: &[usize]) -> Self {
        if ranks.is_empty() {
            return Self {
                mrr: 0.0,
                hits_at_1: 0.0,
                hits_at_3: 0.0,
                hits_at_10: 0.0,
            };
        }
        let n = ranks.len() as f64;
        let mrr = ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n;
        let hits_at_1 = ranks.iter().filter(|&&r| r <= 1).count() as f64 / n;
        let hits_at_3 = ranks.iter().filter(|&&r| r <= 3).count() as f64 / n;
        let hits_at_10 = ranks.iter().filter(|&&r| r <= 10).count() as f64 / n;
        Self {
            mrr,
            hits_at_1,
            hits_at_3,
            hits_at_10,
        }
    }

    /// Convert to a serde_json::Value for embedding in DatasetReport.
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "mrr": self.mrr,
            "hits_at_1": self.hits_at_1,
            "hits_at_3": self.hits_at_3,
            "hits_at_10": self.hits_at_10,
        })
    }
}

/// Streaming accumulator for ranking metrics (avoids storing all ranks).
pub struct RankingAccumulator {
    sum_reciprocal: f64,
    hits_1: usize,
    hits_3: usize,
    hits_10: usize,
    count: usize,
}

impl RankingAccumulator {
    pub fn new() -> Self {
        Self {
            sum_reciprocal: 0.0,
            hits_1: 0,
            hits_3: 0,
            hits_10: 0,
            count: 0,
        }
    }

    /// Record a single 1-based rank.
    pub fn add(&mut self, rank: usize) {
        self.sum_reciprocal += 1.0 / rank as f64;
        if rank <= 1 {
            self.hits_1 += 1;
        }
        if rank <= 3 {
            self.hits_3 += 1;
        }
        if rank <= 10 {
            self.hits_10 += 1;
        }
        self.count += 1;
    }

    /// Finalize into RankingMetrics.
    pub fn finalize(&self) -> RankingMetrics {
        if self.count == 0 {
            return RankingMetrics {
                mrr: 0.0,
                hits_at_1: 0.0,
                hits_at_3: 0.0,
                hits_at_10: 0.0,
            };
        }
        let n = self.count as f64;
        RankingMetrics {
            mrr: self.sum_reciprocal / n,
            hits_at_1: self.hits_1 as f64 / n,
            hits_at_3: self.hits_3 as f64 / n,
            hits_at_10: self.hits_10 as f64 / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_ranks() {
        let metrics = RankingMetrics::from_ranks(&[1, 1, 1, 1]);
        assert!((metrics.mrr - 1.0).abs() < 1e-9);
        assert!((metrics.hits_at_1 - 1.0).abs() < 1e-9);
        assert!((metrics.hits_at_3 - 1.0).abs() < 1e-9);
        assert!((metrics.hits_at_10 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mixed_ranks() {
        // ranks: 1, 2, 5, 20
        let metrics = RankingMetrics::from_ranks(&[1, 2, 5, 20]);
        // MRR = (1/1 + 1/2 + 1/5 + 1/20) / 4 = (1.0 + 0.5 + 0.2 + 0.05) / 4 = 0.4375
        assert!((metrics.mrr - 0.4375).abs() < 1e-9);
        // Hits@1: only rank 1 → 1/4 = 0.25
        assert!((metrics.hits_at_1 - 0.25).abs() < 1e-9);
        // Hits@3: ranks 1, 2 → 2/4 = 0.5
        assert!((metrics.hits_at_3 - 0.5).abs() < 1e-9);
        // Hits@10: ranks 1, 2, 5 → 3/4 = 0.75
        assert!((metrics.hits_at_10 - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_empty_ranks() {
        let metrics = RankingMetrics::from_ranks(&[]);
        assert!((metrics.mrr).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_matches_batch() {
        let ranks = vec![1, 3, 7, 15, 2];
        let batch = RankingMetrics::from_ranks(&ranks);
        let mut acc = RankingAccumulator::new();
        for &r in &ranks {
            acc.add(r);
        }
        let streamed = acc.finalize();
        assert!((batch.mrr - streamed.mrr).abs() < 1e-9);
        assert!((batch.hits_at_1 - streamed.hits_at_1).abs() < 1e-9);
        assert!((batch.hits_at_3 - streamed.hits_at_3).abs() < 1e-9);
        assert!((batch.hits_at_10 - streamed.hits_at_10).abs() < 1e-9);
    }
}
