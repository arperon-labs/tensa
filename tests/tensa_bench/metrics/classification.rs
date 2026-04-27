//! Multi-class classification metrics: per-class P/R/F1, macro/micro averages.
//!
//! Used for MAVEN-ERE event relation extraction evaluation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Accumulates per-class confusion matrix counts.
pub struct ConfusionMatrix {
    /// For each class label: (true_positives, false_positives, false_negatives)
    classes: HashMap<String, (usize, usize, usize)>,
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
        }
    }

    /// Record a single prediction.
    ///
    /// - `predicted`: the label the system predicted (or None if it didn't predict this instance)
    /// - `gold`: the ground truth label
    pub fn add(&mut self, predicted: Option<&str>, gold: &str) {
        match predicted {
            Some(pred) if pred == gold => {
                // True positive for this class
                self.classes.entry(gold.to_string()).or_insert((0, 0, 0)).0 += 1;
            }
            Some(pred) => {
                // False positive for predicted class
                self.classes.entry(pred.to_string()).or_insert((0, 0, 0)).1 += 1;
                // False negative for gold class
                self.classes.entry(gold.to_string()).or_insert((0, 0, 0)).2 += 1;
            }
            None => {
                // False negative for gold class (system missed it)
                self.classes.entry(gold.to_string()).or_insert((0, 0, 0)).2 += 1;
            }
        }
    }

    /// Record a batch of (predicted_label, gold_label) pairs.
    /// Both predicted and gold are present.
    pub fn add_pair(&mut self, predicted: &str, gold: &str) {
        self.add(Some(predicted), gold);
    }

    /// Per-class precision, recall, F1.
    pub fn per_class(&self) -> HashMap<String, ClassMetrics> {
        self.classes
            .iter()
            .map(|(label, &(tp, fp, fneg))| {
                let precision = if tp + fp > 0 {
                    tp as f64 / (tp + fp) as f64
                } else {
                    0.0
                };
                let recall = if tp + fneg > 0 {
                    tp as f64 / (tp + fneg) as f64
                } else {
                    0.0
                };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };
                (
                    label.clone(),
                    ClassMetrics {
                        precision,
                        recall,
                        f1,
                        support: tp + fneg,
                    },
                )
            })
            .collect()
    }

    /// Macro-averaged F1: unweighted mean of per-class F1.
    pub fn macro_f1(&self) -> f64 {
        let per_class = self.per_class();
        if per_class.is_empty() {
            return 0.0;
        }
        let sum: f64 = per_class.values().map(|m| m.f1).sum();
        sum / per_class.len() as f64
    }

    /// Micro-averaged F1: computed from global TP/FP/FN counts.
    pub fn micro_f1(&self) -> f64 {
        let (tp, fp, fneg): (usize, usize, usize) =
            self.classes.values().fold((0, 0, 0), |acc, &(t, f, n)| {
                (acc.0 + t, acc.1 + f, acc.2 + n)
            });
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fneg > 0 {
            tp as f64 / (tp + fneg) as f64
        } else {
            0.0
        };
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    /// Build a summary suitable for DatasetReport.metrics.
    pub fn to_summary(&self) -> ClassificationSummary {
        ClassificationSummary {
            macro_f1: self.macro_f1(),
            micro_f1: self.micro_f1(),
            per_class: self.per_class(),
        }
    }
}

/// Metrics for a single class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Aggregated classification summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationSummary {
    pub macro_f1: f64,
    pub micro_f1: f64,
    pub per_class: HashMap<String, ClassMetrics>,
}

impl ClassificationSummary {
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_classification() {
        let mut cm = ConfusionMatrix::new();
        cm.add_pair("CAUSE", "CAUSE");
        cm.add_pair("CAUSE", "CAUSE");
        cm.add_pair("BEFORE", "BEFORE");

        let per_class = cm.per_class();
        assert!((per_class["CAUSE"].f1 - 1.0).abs() < 1e-9);
        assert!((per_class["BEFORE"].f1 - 1.0).abs() < 1e-9);
        assert!((cm.macro_f1() - 1.0).abs() < 1e-9);
        assert!((cm.micro_f1() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_all_wrong() {
        let mut cm = ConfusionMatrix::new();
        cm.add_pair("CAUSE", "BEFORE");
        cm.add_pair("BEFORE", "CAUSE");

        assert!((cm.macro_f1()).abs() < 1e-9);
        assert!((cm.micro_f1()).abs() < 1e-9);
    }

    #[test]
    fn test_mixed() {
        let mut cm = ConfusionMatrix::new();
        // CAUSE: 2 TP, 1 FP (predicted CAUSE but was BEFORE), 0 FN
        cm.add_pair("CAUSE", "CAUSE");
        cm.add_pair("CAUSE", "CAUSE");
        cm.add_pair("CAUSE", "BEFORE"); // FP for CAUSE, FN for BEFORE
                                        // BEFORE: 1 TP, 0 FP, 1 FN (the misclassified one above)
        cm.add_pair("BEFORE", "BEFORE");

        let per_class = cm.per_class();
        // CAUSE: P=2/3, R=2/2=1.0, F1=2*(2/3)*1/(2/3+1)=0.8
        assert!((per_class["CAUSE"].precision - 2.0 / 3.0).abs() < 0.01);
        assert!((per_class["CAUSE"].recall - 1.0).abs() < 0.01);
        // BEFORE: P=1/1=1.0, R=1/2=0.5, F1=2*1*0.5/1.5=0.667
        assert!((per_class["BEFORE"].precision - 1.0).abs() < 0.01);
        assert!((per_class["BEFORE"].recall - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_missing_predictions() {
        let mut cm = ConfusionMatrix::new();
        cm.add(None, "CAUSE"); // System missed it entirely
        cm.add(Some("CAUSE"), "CAUSE"); // Got this one

        let per_class = cm.per_class();
        // CAUSE: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1=0.667
        assert!((per_class["CAUSE"].recall - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty() {
        let cm = ConfusionMatrix::new();
        assert!((cm.macro_f1()).abs() < 1e-9);
        assert!((cm.micro_f1()).abs() < 1e-9);
    }
}
