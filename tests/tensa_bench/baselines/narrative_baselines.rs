//! Published narrative understanding baselines.
//!
//! ROCStories: GPT-4 (OpenAI 2023), fine-tuned models (Mostafazadeh+ 2016).
//! NarrativeQA: GPT-4 (OpenAI 2023), various reader models (Kočiský+ 2018).
//! MAVEN-ERE: EEQA, DEGREE, TagPrime (Wang+ 2022).

use crate::tensa_bench::benchmarks_base::BaselineComparison;

/// Published ROCStories / Story Cloze Test baselines (accuracy).
pub fn rocstories_baselines() -> Vec<(&'static str, f64)> {
    vec![
        ("GPT-4", 0.97),
        ("GPT-3.5", 0.92),
        ("BERT-large (fine-tuned)", 0.88),
        ("Random", 0.50),
    ]
}

/// Published NarrativeQA baselines (ROUGE-L on summary setting).
pub fn narrativeqa_baselines() -> Vec<(&'static str, f64)> {
    vec![
        ("GPT-4", 0.64),
        ("GPT-3.5", 0.52),
        ("BiDAF", 0.36),
        ("Retrieval + Reader", 0.45),
    ]
}

/// Published MAVEN-ERE baselines (micro F1).
pub fn maven_ere_baselines() -> Vec<(&'static str, f64)> {
    vec![
        ("EEQA", 0.48),
        ("DEGREE", 0.51),
        ("TagPrime", 0.53),
        ("OneIE", 0.46),
    ]
}

/// Build BaselineComparison entries for ROCStories (accuracy).
pub fn build_rocstories_comparisons(
    tensa_accuracy: f64,
    baselines: &[(&str, f64)],
) -> Vec<BaselineComparison> {
    baselines
        .iter()
        .map(|&(name, baseline_acc)| BaselineComparison {
            method: name.to_string(),
            metric: "Accuracy".to_string(),
            baseline_value: baseline_acc,
            tensa_value: tensa_accuracy,
            delta: tensa_accuracy - baseline_acc,
        })
        .collect()
}

/// Build BaselineComparison entries for NarrativeQA (ROUGE-L).
pub fn build_narrativeqa_comparisons(
    tensa_rouge_l: f64,
    baselines: &[(&str, f64)],
) -> Vec<BaselineComparison> {
    baselines
        .iter()
        .map(|&(name, baseline_rl)| BaselineComparison {
            method: name.to_string(),
            metric: "ROUGE-L".to_string(),
            baseline_value: baseline_rl,
            tensa_value: tensa_rouge_l,
            delta: tensa_rouge_l - baseline_rl,
        })
        .collect()
}

/// Build BaselineComparison entries for MAVEN-ERE (micro F1).
pub fn build_maven_comparisons(
    tensa_micro_f1: f64,
    baselines: &[(&str, f64)],
) -> Vec<BaselineComparison> {
    baselines
        .iter()
        .map(|&(name, baseline_f1)| BaselineComparison {
            method: name.to_string(),
            metric: "Micro-F1".to_string(),
            baseline_value: baseline_f1,
            tensa_value: tensa_micro_f1,
            delta: tensa_micro_f1 - baseline_f1,
        })
        .collect()
}
