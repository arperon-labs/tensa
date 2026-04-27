//! Published TKG link prediction baselines.
//!
//! Numbers from: TComplEx (Lacroix+ 2020), RE-NET (Jin+ 2020),
//! TANGO (Han+ 2021), TimeTraveler (Sun+ 2021).

use crate::tensa_bench::benchmarks_base::BaselineComparison;
use crate::tensa_bench::metrics::ranking::RankingMetrics;

/// Published ICEWS14 baselines (filtered setting, time-aware).
pub fn icews14_baselines() -> Vec<(&'static str, RankingMetrics)> {
    vec![
        (
            "TComplEx",
            RankingMetrics {
                mrr: 0.560,
                hits_at_1: 0.470,
                hits_at_3: 0.610,
                hits_at_10: 0.730,
            },
        ),
        (
            "RE-NET",
            RankingMetrics {
                mrr: 0.457,
                hits_at_1: 0.360,
                hits_at_3: 0.510,
                hits_at_10: 0.650,
            },
        ),
        (
            "TANGO",
            RankingMetrics {
                mrr: 0.584,
                hits_at_1: 0.498,
                hits_at_3: 0.638,
                hits_at_10: 0.750,
            },
        ),
        (
            "TimeTraveler",
            RankingMetrics {
                mrr: 0.594,
                hits_at_1: 0.507,
                hits_at_3: 0.648,
                hits_at_10: 0.760,
            },
        ),
    ]
}

/// Published ICEWS18 baselines.
pub fn icews18_baselines() -> Vec<(&'static str, RankingMetrics)> {
    vec![
        (
            "TComplEx",
            RankingMetrics {
                mrr: 0.470,
                hits_at_1: 0.370,
                hits_at_3: 0.520,
                hits_at_10: 0.660,
            },
        ),
        (
            "RE-NET",
            RankingMetrics {
                mrr: 0.429,
                hits_at_1: 0.335,
                hits_at_3: 0.480,
                hits_at_10: 0.620,
            },
        ),
        (
            "TANGO",
            RankingMetrics {
                mrr: 0.490,
                hits_at_1: 0.396,
                hits_at_3: 0.542,
                hits_at_10: 0.678,
            },
        ),
    ]
}

/// Build BaselineComparison entries for a given dataset.
pub fn build_tkg_comparisons(
    tensa_metrics: &RankingMetrics,
    baselines: &[(&str, RankingMetrics)],
) -> Vec<BaselineComparison> {
    let mut comparisons = Vec::new();
    for (name, baseline) in baselines {
        comparisons.push(BaselineComparison {
            method: name.to_string(),
            metric: "MRR".to_string(),
            baseline_value: baseline.mrr,
            tensa_value: tensa_metrics.mrr,
            delta: tensa_metrics.mrr - baseline.mrr,
        });
        comparisons.push(BaselineComparison {
            method: name.to_string(),
            metric: "Hits@1".to_string(),
            baseline_value: baseline.hits_at_1,
            tensa_value: tensa_metrics.hits_at_1,
            delta: tensa_metrics.hits_at_1 - baseline.hits_at_1,
        });
        comparisons.push(BaselineComparison {
            method: name.to_string(),
            metric: "Hits@10".to_string(),
            baseline_value: baseline.hits_at_10,
            tensa_value: tensa_metrics.hits_at_10,
            delta: tensa_metrics.hits_at_10 - baseline.hits_at_10,
        });
    }
    comparisons
}
