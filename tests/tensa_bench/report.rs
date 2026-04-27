//! Report builder: aggregates benchmark results into JSON, Markdown, and LaTeX.

use crate::tensa_bench::{DatasetReport, DomainReport, TensaBenchReport};

/// Builder for constructing a TensaBenchReport incrementally.
pub struct ReportBuilder {
    version: String,
    domains: Vec<DomainReport>,
    start_time: std::time::Instant,
}

impl ReportBuilder {
    pub fn new(version: &str) -> Self {
        Self {
            version: version.to_string(),
            domains: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Add a completed domain report.
    pub fn add_domain(&mut self, domain: DomainReport) {
        self.domains.push(domain);
    }

    /// Build the final report.
    pub fn build(self) -> TensaBenchReport {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        TensaBenchReport {
            suite: "tensa-bench".to_string(),
            version: self.version,
            timestamp: chrono::Utc::now().to_rfc3339(),
            domains: self.domains,
            total_duration_sec: elapsed,
        }
    }
}

/// Generate a LaTeX table from benchmark results.
///
/// Produces a `\begin{tabular}` environment suitable for paper appendix.
pub fn to_latex_table(report: &TensaBenchReport) -> String {
    let mut latex = String::new();

    for domain in &report.domains {
        latex.push_str(&format!("% Domain: {}\n", domain.domain));

        match domain.domain.as_str() {
            "tkg" => latex.push_str(&tkg_latex_table(&domain.datasets)),
            "multihop" => latex.push_str(&multihop_latex_table(&domain.datasets)),
            "narrative" => latex.push_str(&narrative_latex_table(&domain.datasets)),
            _ => {}
        }
        latex.push('\n');
    }

    latex
}

fn tkg_latex_table(datasets: &[DatasetReport]) -> String {
    let mut s = String::new();
    s.push_str("\\begin{table}[h]\n");
    s.push_str("\\centering\n");
    s.push_str("\\caption{Temporal Knowledge Graph Link Prediction}\n");
    s.push_str("\\begin{tabular}{lcccc}\n");
    s.push_str("\\toprule\n");
    s.push_str("Method & MRR & Hits@1 & Hits@3 & Hits@10 \\\\\n");
    s.push_str("\\midrule\n");

    for ds in datasets {
        // Add baseline rows first
        for b in &ds.baselines {
            if b.metric == "MRR" {
                // Find matching Hits@1 and Hits@10 baselines
                let h1 = ds
                    .baselines
                    .iter()
                    .find(|bb| bb.method == b.method && bb.metric == "Hits@1")
                    .map(|bb| bb.baseline_value)
                    .unwrap_or(0.0);
                let h10 = ds
                    .baselines
                    .iter()
                    .find(|bb| bb.method == b.method && bb.metric == "Hits@10")
                    .map(|bb| bb.baseline_value)
                    .unwrap_or(0.0);
                s.push_str(&format!(
                    "{} & {:.3} & {:.3} & -- & {:.3} \\\\\n",
                    b.method, b.baseline_value, h1, h10
                ));
            }
        }

        // Add TENSA row
        if let Some(metrics) = ds.metrics.as_object() {
            s.push_str("\\midrule\n");
            s.push_str(&format!(
                "\\textbf{{TENSA}} & \\textbf{{{:.3}}} & \\textbf{{{:.3}}} & \\textbf{{{:.3}}} & \\textbf{{{:.3}}} \\\\\n",
                metrics.get("mrr").and_then(|v| v.as_f64()).unwrap_or(0.0),
                metrics.get("hits_at_1").and_then(|v| v.as_f64()).unwrap_or(0.0),
                metrics.get("hits_at_3").and_then(|v| v.as_f64()).unwrap_or(0.0),
                metrics.get("hits_at_10").and_then(|v| v.as_f64()).unwrap_or(0.0),
            ));
        }
    }

    s.push_str("\\bottomrule\n");
    s.push_str("\\end{tabular}\n");
    s.push_str("\\end{table}\n");
    s
}

fn multihop_latex_table(datasets: &[DatasetReport]) -> String {
    let mut s = String::new();
    s.push_str("\\begin{table}[h]\n");
    s.push_str("\\centering\n");
    s.push_str("\\caption{Multi-hop Question Answering}\n");
    s.push_str("\\begin{tabular}{lcc}\n");
    s.push_str("\\toprule\n");
    s.push_str("Method & EM & F1 \\\\\n");
    s.push_str("\\midrule\n");

    for ds in datasets {
        for b in &ds.baselines {
            if b.metric == "EM" {
                let f1 = ds
                    .baselines
                    .iter()
                    .find(|bb| bb.method == b.method && bb.metric == "F1")
                    .map(|bb| bb.baseline_value)
                    .unwrap_or(0.0);
                s.push_str(&format!(
                    "{} & {:.3} & {:.3} \\\\\n",
                    b.method, b.baseline_value, f1
                ));
            }
        }

        if let Some(metrics) = ds.metrics.as_object() {
            s.push_str("\\midrule\n");
            s.push_str(&format!(
                "\\textbf{{TENSA}} & \\textbf{{{:.3}}} & \\textbf{{{:.3}}} \\\\\n",
                metrics
                    .get("exact_match")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                metrics
                    .get("token_f1")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            ));
        }
    }

    s.push_str("\\bottomrule\n");
    s.push_str("\\end{tabular}\n");
    s.push_str("\\end{table}\n");
    s
}

fn narrative_latex_table(datasets: &[DatasetReport]) -> String {
    let mut s = String::new();
    s.push_str("\\begin{table}[h]\n");
    s.push_str("\\centering\n");
    s.push_str("\\caption{Narrative Understanding}\n");
    s.push_str("\\begin{tabular}{llcc}\n");
    s.push_str("\\toprule\n");
    s.push_str("Dataset & Method & Primary Metric & Value \\\\\n");
    s.push_str("\\midrule\n");

    for ds in datasets {
        for b in &ds.baselines {
            s.push_str(&format!(
                "{} & {} & {} & {:.3} \\\\\n",
                ds.name, b.method, b.metric, b.baseline_value
            ));
        }
        // TENSA row
        let primary_value = if ds.name.contains("ROC") || ds.name.contains("Story") {
            ds.metrics
                .get("accuracy")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        } else if ds.name.contains("MAVEN") {
            ds.metrics
                .get("micro_f1")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        } else {
            ds.metrics
                .get("rouge_l")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };
        let primary_name = if ds.name.contains("ROC") {
            "Accuracy"
        } else if ds.name.contains("MAVEN") {
            "Micro-F1"
        } else {
            "ROUGE-L"
        };
        s.push_str(&format!(
            "{} & \\textbf{{TENSA}} & {} & \\textbf{{{:.3}}} \\\\\n",
            ds.name, primary_name, primary_value
        ));
        s.push_str("\\midrule\n");
    }

    s.push_str("\\bottomrule\n");
    s.push_str("\\end{tabular}\n");
    s.push_str("\\end{table}\n");
    s
}

/// Write report to a JSON file.
pub fn write_json(report: &TensaBenchReport, path: &std::path::Path) -> Result<(), String> {
    let json = report.to_json();
    std::fs::write(path, json).map_err(|e| format!("Failed to write report: {}", e))
}

/// Print a comprehensive Markdown report to stderr.
pub fn print_full_report(report: &TensaBenchReport) {
    eprintln!("# TENSA-BENCH Results");
    eprintln!();
    eprintln!(
        "Version: {} | Date: {} | Duration: {:.1}s",
        report.version, report.timestamp, report.total_duration_sec
    );
    eprintln!();

    for domain in &report.domains {
        eprintln!("## Domain: {}", domain.domain);
        eprintln!();
        for ds in &domain.datasets {
            ds.print_markdown();
        }
    }

    report.print_summary();
}
