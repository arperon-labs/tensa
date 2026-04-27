//! Reusable comparison harness for benchmarks (EATH Phase 8).
//!
//! Two snapshots of a hypergraph (a "baseline" and a "treatment") get reduced
//! to a fixed scalar metric set, the deltas are computed, and the result is
//! rendered as a side-by-side Markdown table. Phase 9 (adversarial wargame)
//! reuses this to produce "intervention vs no-intervention on identical
//! synthetic substrate" reports — the harness is intentionally narrative-id
//! agnostic so the same instance can compare two runs from different sources.
//!
//! ## Metric set
//!
//! [`MetricKind`] enumerates every comparison the harness knows how to compute.
//! New metrics are added by extending the enum + the `compute_one` match arm.
//! Each metric reduces to a single `f64` so the side-by-side table is
//! homogeneous; structural metrics (e.g. centrality vectors) are summarized to
//! the top-actor scalar so we don't paper over a per-entity diff in a
//! single-row report.
//!
//! ## Rendering contract
//!
//! [`ComparisonHarness::render_markdown`] always emits the exact 5-column
//! table `| Metric | Baseline | Treatment | Δ | % |`. The percent column
//! handles the zero-baseline edge case explicitly (renders `n/a` instead of
//! `inf` or `nan`) — see the unit test
//! `test_comparison_harness_handles_zero_baseline_gracefully`.

use std::collections::HashMap;

use tensa::analysis::graph_centrality::compute_pagerank;
use tensa::analysis::graph_projection::build_co_graph;
use tensa::error::Result;
use tensa::hypergraph::Hypergraph;

/// One scalar comparison the harness knows how to compute.
///
/// Each variant maps to a deterministic, side-effect-free reduction over a
/// `(Hypergraph, narrative_id)` pair. Variants are ordered by cost (cheapest
/// first) — `ComparisonHarness::compare` runs them in declaration order so a
/// caller that fails midway can still inspect the cheap rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// Number of entities in the narrative.
    EntityCount,
    /// Number of situations in the narrative.
    SituationCount,
    /// Number of (entity, situation) participation rows in the narrative.
    ParticipationCount,
    /// Average number of participants per situation. 0.0 when there are no
    /// situations (avoids div-by-zero).
    AvgGroupSize,
    /// Top entity's PageRank score (max over the entity vector). 0.0 when the
    /// narrative is empty. Summarized to a scalar so the side-by-side table
    /// stays homogeneous; full per-entity diff is out of scope for this
    /// summary view.
    PagerankTopActor,
}

impl MetricKind {
    /// Stable identifier used in the rendered Markdown table's `Metric` column.
    /// MUST be lower_snake_case so downstream test assertions can grep for it.
    pub fn label(&self) -> &'static str {
        match self {
            MetricKind::EntityCount => "entity_count",
            MetricKind::SituationCount => "situation_count",
            MetricKind::ParticipationCount => "participation_count",
            MetricKind::AvgGroupSize => "avg_group_size",
            MetricKind::PagerankTopActor => "pagerank_top_actor",
        }
    }

    /// Returns every variant in canonical declaration order. Used by the
    /// `default()` constructor to build a "compare everything" harness.
    pub fn all() -> Vec<MetricKind> {
        vec![
            MetricKind::EntityCount,
            MetricKind::SituationCount,
            MetricKind::ParticipationCount,
            MetricKind::AvgGroupSize,
            MetricKind::PagerankTopActor,
        ]
    }
}

/// One row in the rendered comparison table.
#[derive(Debug, Clone)]
pub struct ComparisonRow {
    /// Metric label (e.g. `"pagerank_top_actor"`).
    pub metric: String,
    pub baseline: f64,
    pub treatment: f64,
    /// `treatment - baseline`. Sign-preserving so callers can sort by impact.
    pub delta: f64,
    /// `100.0 * delta / baseline` when `baseline != 0`. `None` for the
    /// zero-baseline edge case (rendered as `n/a` in the table).
    pub pct_change: Option<f64>,
}

/// Provenance tag for a comparison result — distinguishes empirical
/// narratives from synthetic / hybrid runs so reviewers can tell at a glance
/// whether they're looking at real-world data vs an EATH surrogate vs a
/// hybrid mix. EATH Phase 9 addition.
///
/// `#[allow(dead_code)]`: the `Real` and `Synthetic` variants are part of
/// the public API surface even though only `Hybrid` is exercised in this
/// crate's unit tests. Downstream consumers (Phase 9 wargame intervention
/// reports, future Phase 12.5 fidelity benchmarks) populate the other two.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceTag {
    /// Real (empirical) narrative — no synthetic generation in the pipeline.
    Real,
    /// Single-source synthetic narrative produced by `model` calibrated on
    /// `source` with `seed` for the deterministic stream.
    Synthetic {
        model: String,
        source: String,
        seed: u64,
    },
    /// Hybrid (mixture-distribution) narrative — `components` carries the
    /// `(narrative_id, weight)` pairs that produced this run, in the order
    /// the hybrid generator received them.
    Hybrid {
        components: Vec<(String, f32)>,
        seed: u64,
    },
}

/// Result of one [`ComparisonHarness::compare`] call.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub baseline_label: String,
    pub treatment_label: String,
    pub rows: Vec<ComparisonRow>,
    /// Optional provenance tag — when present, [`ComparisonHarness::render_markdown`]
    /// emits a "Provenance: ..." line above the table so reviewers can tell
    /// at a glance whether a comparison ran on empirical or synthetic data.
    /// EATH Phase 9 addition.
    pub provenance: Option<ProvenanceTag>,
    /// Optional substrate hash — typically the canonical narrative state hash
    /// (see `crate::synth::hashing::canonical_narrative_state_hash`). Phase 9
    /// wargame intervention reports use this to prove that the baseline and
    /// treatment ran on the SAME substrate.
    pub substrate_hash: Option<String>,
}

/// Configurable comparison harness — pick the metric set up front, then call
/// `compare` for each (baseline, treatment) pair.
///
/// Cloning is cheap (only `Vec<MetricKind>` inside) so the harness can be
/// shared across many comparison rounds in a benchmark loop.
#[derive(Debug, Clone)]
pub struct ComparisonHarness {
    metrics: Vec<MetricKind>,
}

impl Default for ComparisonHarness {
    /// Default harness compares every metric in [`MetricKind::all`]. Phase 9's
    /// wargame intervention reports will likely override this with a curated
    /// subset (peak prevalence, R₀, narrative diameter).
    fn default() -> Self {
        Self {
            metrics: MetricKind::all(),
        }
    }
}

impl ComparisonHarness {
    /// Build a harness with the explicit metric set. Empty `metrics` produces
    /// a no-op harness whose `compare` returns an empty rows vector — useful
    /// in tests but never useful in production.
    pub fn new(metrics: Vec<MetricKind>) -> Self {
        Self { metrics }
    }

    /// Return a reference to the active metric set. Used by the
    /// "metric_set_is_configurable" unit test.
    pub fn metrics(&self) -> &[MetricKind] {
        &self.metrics
    }

    /// Compute every configured metric on both snapshots and produce a
    /// `ComparisonResult`. Failures during metric computation propagate as
    /// `TensaError` — the caller decides whether to swallow or rethrow.
    pub fn compare(
        &self,
        baseline: &Hypergraph,
        baseline_narrative_id: &str,
        treatment: &Hypergraph,
        treatment_narrative_id: &str,
    ) -> Result<ComparisonResult> {
        let mut rows = Vec::with_capacity(self.metrics.len());
        // Cheap caches keyed by (graph identity, narrative_id) would be ideal
        // but Hypergraph doesn't expose an identity handle; in practice each
        // metric does its own work so duplication is acceptable for the
        // 5-row default set.
        let mut baseline_cache: HashMap<&'static str, f64> = HashMap::new();
        let mut treatment_cache: HashMap<&'static str, f64> = HashMap::new();

        for &metric in &self.metrics {
            let label = metric.label();
            let baseline_value = match baseline_cache.get(label) {
                Some(v) => *v,
                None => {
                    let v = compute_one(baseline, baseline_narrative_id, metric)?;
                    baseline_cache.insert(label, v);
                    v
                }
            };
            let treatment_value = match treatment_cache.get(label) {
                Some(v) => *v,
                None => {
                    let v = compute_one(treatment, treatment_narrative_id, metric)?;
                    treatment_cache.insert(label, v);
                    v
                }
            };
            let delta = treatment_value - baseline_value;
            let pct_change = if baseline_value.abs() < 1e-12 {
                None
            } else {
                Some(100.0 * delta / baseline_value)
            };
            rows.push(ComparisonRow {
                metric: label.to_string(),
                baseline: baseline_value,
                treatment: treatment_value,
                delta,
                pct_change,
            });
        }

        Ok(ComparisonResult {
            baseline_label: baseline_narrative_id.to_string(),
            treatment_label: treatment_narrative_id.to_string(),
            rows,
            provenance: None,
            substrate_hash: None,
        })
    }

    /// Render `result` as the canonical 5-column Markdown table. See the
    /// module-level docs for the exact format. When `result.provenance` is
    /// present, emits a "Provenance:" line before the table; when
    /// `substrate_hash` is present, emits a "Substrate:" line. Both lines
    /// support EATH Phase 9 wargame intervention reports.
    pub fn render_markdown(&self, result: &ComparisonResult) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "### Comparison: {} vs {}\n\n",
            result.baseline_label, result.treatment_label
        ));
        if let Some(prov) = &result.provenance {
            out.push_str(&format!("Provenance: {}\n\n", render_provenance(prov)));
        }
        if let Some(hash) = &result.substrate_hash {
            out.push_str(&format!("Substrate: `{hash}`\n\n"));
        }
        out.push_str("| Metric | Baseline | Treatment | Δ | % |\n");
        out.push_str("|---|---|---|---|---|\n");
        for row in &result.rows {
            let pct = match row.pct_change {
                Some(p) => format!("{p:+.1}%"),
                None => "n/a".to_string(),
            };
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                row.metric,
                fmt_value(row.baseline),
                fmt_value(row.treatment),
                fmt_signed(row.delta),
                pct,
            ));
        }
        out
    }
}

// ── Internal: provenance + per-metric reducers ───────────────────────────────

fn render_provenance(p: &ProvenanceTag) -> String {
    match p {
        ProvenanceTag::Real => "real (empirical)".into(),
        ProvenanceTag::Synthetic { model, source, seed } => {
            format!("synthetic (model={model}, source={source}, seed={seed})")
        }
        ProvenanceTag::Hybrid { components, seed } => {
            let parts: Vec<String> = components
                .iter()
                .map(|(nid, w)| format!("{nid}@{w:.2}"))
                .collect();
            format!("hybrid (components=[{}], seed={seed})", parts.join(", "))
        }
    }
}



fn compute_one(hypergraph: &Hypergraph, narrative_id: &str, metric: MetricKind) -> Result<f64> {
    match metric {
        MetricKind::EntityCount => Ok(hypergraph
            .list_entities_by_narrative(narrative_id)?
            .len() as f64),
        MetricKind::SituationCount => Ok(hypergraph
            .list_situations_by_narrative(narrative_id)?
            .len() as f64),
        MetricKind::ParticipationCount => {
            let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
            let mut total: usize = 0;
            for s in &situations {
                total += hypergraph.get_participants_for_situation(&s.id)?.len();
            }
            Ok(total as f64)
        }
        MetricKind::AvgGroupSize => {
            let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
            if situations.is_empty() {
                return Ok(0.0);
            }
            let mut total: usize = 0;
            for s in &situations {
                total += hypergraph.get_participants_for_situation(&s.id)?.len();
            }
            Ok(total as f64 / situations.len() as f64)
        }
        MetricKind::PagerankTopActor => {
            let graph = build_co_graph(hypergraph, narrative_id)?;
            if graph.adj.is_empty() {
                return Ok(0.0);
            }
            let scores = compute_pagerank(&graph);
            Ok(scores.into_iter().fold(0.0_f64, f64::max))
        }
    }
}

fn fmt_value(v: f64) -> String {
    // Integer-valued metrics (counts) render as integers for readability;
    // fractional metrics keep 4 decimals.
    if (v.fract().abs()) < 1e-9 && v.abs() < 1e15 {
        format!("{:.0}", v)
    } else {
        format!("{:.4}", v)
    }
}

fn fmt_signed(v: f64) -> String {
    if (v.fract().abs()) < 1e-9 && v.abs() < 1e15 {
        format!("{:+.0}", v)
    } else {
        format!("{:+.4}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tensa::store::memory::MemoryStore;
    use tensa::types::*;

    /// Build a tiny narrative with `n` actors all participating in one
    /// situation. Just enough for the harness to compute every metric on.
    fn build_tiny_narrative(narrative_id: &str, n: usize) -> Arc<Hypergraph> {
        let store: Arc<dyn tensa::store::KVStore> = Arc::new(MemoryStore::new());
        let hg = Arc::new(Hypergraph::new(store));
        let now = chrono::Utc::now();

        let mut entity_ids = Vec::with_capacity(n);
        for i in 0..n {
            let entity = Entity {
                id: uuid::Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("actor-{i}")}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 1.0,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some(narrative_id.into()),
                created_at: now,
                updated_at: now,
                deleted_at: None,
                transaction_time: None,
            };
            let id = entity.id;
            hg.create_entity(entity).unwrap();
            entity_ids.push(id);
        }

        let sit = Situation {
            id: uuid::Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(now),
                end: Some(now),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("comparison harness fixture")],
            narrative_level: NarrativeLevel::Event,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative_id.into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        let sid = sit.id;
        hg.create_situation(sit).unwrap();
        for &eid in &entity_ids {
            hg.add_participant(Participation {
                entity_id: eid,
                situation_id: sid,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
        hg
    }

    #[test]
    fn test_comparison_harness_renders_markdown_table_with_pct_change() {
        let baseline = build_tiny_narrative("baseline-narr", 2);
        let treatment = build_tiny_narrative("treatment-narr", 4);

        let harness = ComparisonHarness::default();
        let result = harness
            .compare(&baseline, "baseline-narr", &treatment, "treatment-narr")
            .expect("compare should succeed");

        // Treatment has 2x entities, so entity_count delta should be +2 and
        // pct_change should be +100%.
        let entity_row = result
            .rows
            .iter()
            .find(|r| r.metric == "entity_count")
            .expect("entity_count row");
        assert_eq!(entity_row.baseline, 2.0);
        assert_eq!(entity_row.treatment, 4.0);
        assert_eq!(entity_row.delta, 2.0);
        assert_eq!(entity_row.pct_change, Some(100.0));

        let md = harness.render_markdown(&result);
        // Header row + separator + 5 metric rows.
        assert!(md.contains("| Metric | Baseline | Treatment | Δ | % |"));
        assert!(md.contains("|---|---|---|---|---|"));
        assert!(md.contains("entity_count"));
        assert!(md.contains("+100.0%"));
    }

    #[test]
    fn test_comparison_harness_handles_zero_baseline_gracefully() {
        // Empty baseline narrative ⇒ entity_count = 0 ⇒ pct_change must be
        // None / "n/a" (no div-by-zero panic, no inf/nan in output).
        let store: Arc<dyn tensa::store::KVStore> = Arc::new(MemoryStore::new());
        let baseline = Arc::new(Hypergraph::new(store));
        let treatment = build_tiny_narrative("treatment-zero-baseline", 3);

        let harness = ComparisonHarness::new(vec![MetricKind::EntityCount]);
        let result = harness
            .compare(&baseline, "missing-narr", &treatment, "treatment-zero-baseline")
            .expect("compare should succeed");

        let row = &result.rows[0];
        assert_eq!(row.baseline, 0.0);
        assert_eq!(row.treatment, 3.0);
        assert_eq!(row.delta, 3.0);
        assert!(row.pct_change.is_none(), "zero baseline must yield None");

        let md = harness.render_markdown(&result);
        assert!(md.contains("n/a"), "Markdown must render 'n/a' for None pct_change");
        assert!(!md.contains("inf"));
        assert!(!md.contains("NaN"));
    }

    #[test]
    fn test_comparison_harness_renders_provenance_and_substrate() {
        let baseline = build_tiny_narrative("base", 2);
        let treatment = build_tiny_narrative("treat", 4);

        let harness = ComparisonHarness::new(vec![MetricKind::EntityCount]);
        let mut result = harness
            .compare(&baseline, "base", &treatment, "treat")
            .expect("compare should succeed");

        // Phase 9: caller stamps provenance + substrate before render.
        result.provenance = Some(ProvenanceTag::Hybrid {
            components: vec![("src-a".into(), 0.7), ("src-b".into(), 0.3)],
            seed: 42,
        });
        result.substrate_hash = Some("abc123def456".into());

        let md = harness.render_markdown(&result);
        assert!(
            md.contains("Provenance: hybrid"),
            "Provenance line should appear: {md}"
        );
        assert!(md.contains("src-a@0.70"));
        assert!(md.contains("src-b@0.30"));
        assert!(md.contains("seed=42"));
        assert!(md.contains("Substrate: `abc123def456`"));
    }

    #[test]
    fn test_comparison_harness_metric_set_is_configurable() {
        // Constructor accepts a custom subset; only those metrics appear in
        // the result. Order is preserved.
        let custom = vec![MetricKind::ParticipationCount, MetricKind::AvgGroupSize];
        let harness = ComparisonHarness::new(custom.clone());
        assert_eq!(harness.metrics(), custom.as_slice());

        let baseline = build_tiny_narrative("a", 3);
        let treatment = build_tiny_narrative("b", 5);

        let result = harness
            .compare(&baseline, "a", &treatment, "b")
            .expect("compare should succeed");
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].metric, "participation_count");
        assert_eq!(result.rows[1].metric, "avg_group_size");

        // Empty subset: harness becomes a no-op (rows vector is empty).
        let empty_harness = ComparisonHarness::new(vec![]);
        let empty_result = empty_harness
            .compare(&baseline, "a", &treatment, "b")
            .expect("empty compare should succeed");
        assert!(empty_result.rows.is_empty());
    }
}
