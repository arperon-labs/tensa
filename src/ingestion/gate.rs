//! Confidence gating for extracted narrative data.
//!
//! Routes extracted items based on confidence scores:
//! - ≥ auto_commit_threshold → auto-commit as Candidate
//! - ≥ review_threshold → queue for human review
//! - < review_threshold → reject (log only)

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Confidence gate configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceGate {
    /// Minimum confidence for auto-commit (default: 0.8).
    pub auto_commit_threshold: f32,
    /// Minimum confidence for review queue (default: 0.3).
    pub review_threshold: f32,
}

impl Default for ConfidenceGate {
    fn default() -> Self {
        Self {
            auto_commit_threshold: 0.8,
            review_threshold: 0.3,
        }
    }
}

/// Decision made by the confidence gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Confidence high enough for auto-commit.
    AutoCommit,
    /// Confidence requires human review.
    QueueForReview,
    /// Confidence too low; reject.
    Reject,
}

/// Report of gating results for an extraction batch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateReport {
    pub auto_committed: usize,
    pub queued: usize,
    pub rejected: usize,
    pub entity_ids: Vec<Uuid>,
    pub situation_ids: Vec<Uuid>,
}

impl ConfidenceGate {
    /// Create a gate with custom thresholds.
    pub fn new(auto_commit_threshold: f32, review_threshold: f32) -> Self {
        Self {
            auto_commit_threshold,
            review_threshold,
        }
    }

    /// Decide what to do with an item based on its confidence.
    pub fn decide(&self, confidence: f32) -> GateDecision {
        if confidence >= self.auto_commit_threshold {
            GateDecision::AutoCommit
        } else if confidence >= self.review_threshold {
            GateDecision::QueueForReview
        } else {
            GateDecision::Reject
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_auto_commit_high_confidence() {
        let gate = ConfidenceGate::default();
        assert_eq!(gate.decide(0.95), GateDecision::AutoCommit);
        assert_eq!(gate.decide(0.8), GateDecision::AutoCommit);
    }

    #[test]
    fn test_gate_queue_medium_confidence() {
        let gate = ConfidenceGate::default();
        assert_eq!(gate.decide(0.5), GateDecision::QueueForReview);
        assert_eq!(gate.decide(0.3), GateDecision::QueueForReview);
        assert_eq!(gate.decide(0.79), GateDecision::QueueForReview);
    }

    #[test]
    fn test_gate_reject_low_confidence() {
        let gate = ConfidenceGate::default();
        assert_eq!(gate.decide(0.29), GateDecision::Reject);
        assert_eq!(gate.decide(0.1), GateDecision::Reject);
        assert_eq!(gate.decide(0.0), GateDecision::Reject);
    }

    #[test]
    fn test_gate_boundary_at_thresholds() {
        let gate = ConfidenceGate::new(0.8, 0.3);
        assert_eq!(gate.decide(0.8), GateDecision::AutoCommit);
        assert_eq!(gate.decide(0.3), GateDecision::QueueForReview);
    }

    #[test]
    fn test_gate_custom_thresholds() {
        let gate = ConfidenceGate::new(0.9, 0.5);
        assert_eq!(gate.decide(0.85), GateDecision::QueueForReview);
        assert_eq!(gate.decide(0.95), GateDecision::AutoCommit);
        assert_eq!(gate.decide(0.4), GateDecision::Reject);
    }

    #[test]
    fn test_gate_report_defaults() {
        let report = GateReport::default();
        assert_eq!(report.auto_committed, 0);
        assert_eq!(report.queued, 0);
        assert_eq!(report.rejected, 0);
    }
}
