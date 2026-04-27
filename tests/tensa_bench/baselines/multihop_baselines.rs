//! Published multi-hop QA baselines (HotpotQA distractor setting).
//!
//! Numbers from: Microsoft GraphRAG (Edge+ 2024), LightRAG (Guo+ 2024),
//! HippoRAG (Yu+ 2024), vanilla RAG baselines.

use crate::tensa_bench::benchmarks_base::BaselineComparison;

/// Published HotpotQA distractor baselines (EM, F1).
pub fn hotpotqa_baselines() -> Vec<(&'static str, f64, f64)> {
    vec![
        // (method, EM, F1)
        ("GraphRAG (Global)", 0.38, 0.51),
        ("GraphRAG (Local)", 0.42, 0.55),
        ("LightRAG", 0.35, 0.48),
        ("HippoRAG", 0.42, 0.55),
        ("Vanilla RAG", 0.30, 0.42),
    ]
}

/// Published 2WikiMultiHopQA baselines.
pub fn wikimultihop_baselines() -> Vec<(&'static str, f64, f64)> {
    vec![
        ("GraphRAG (Global)", 0.35, 0.48),
        ("GraphRAG (Local)", 0.39, 0.52),
        ("LightRAG", 0.33, 0.46),
        ("HippoRAG", 0.40, 0.53),
        ("Vanilla RAG", 0.28, 0.40),
    ]
}

/// Build BaselineComparison entries for multi-hop QA.
pub fn build_multihop_comparisons(
    tensa_em: f64,
    tensa_f1: f64,
    baselines: &[(&str, f64, f64)],
) -> Vec<BaselineComparison> {
    let mut comparisons = Vec::new();
    for &(name, baseline_em, baseline_f1) in baselines {
        comparisons.push(BaselineComparison {
            method: name.to_string(),
            metric: "EM".to_string(),
            baseline_value: baseline_em,
            tensa_value: tensa_em,
            delta: tensa_em - baseline_em,
        });
        comparisons.push(BaselineComparison {
            method: name.to_string(),
            metric: "F1".to_string(),
            baseline_value: baseline_f1,
            tensa_value: tensa_f1,
            delta: tensa_f1 - baseline_f1,
        });
    }
    comparisons
}
