//! Fuzzy Sprint Phase 10 + 11 — embedded MCP impls for the fuzzy surface.
//!
//! Phase 10 landed one tool (`fuzzy_probability`); Phase 11 adds the
//! remaining 13 fuzzy MCP tools. This file hosts tools 1–7
//! (`fuzzy_list_tnorms` / `fuzzy_list_aggregators` / `fuzzy_get_config` /
//! `fuzzy_set_config` / `fuzzy_create_measure` / `fuzzy_list_measures` /
//! `fuzzy_aggregate`) + Phase 10's `fuzzy_probability`; the remaining
//! tools (allen / quantify / syllogism / fca / rules) live in the sibling
//! [`super::embedded_fuzzy_ext`] module so the 500-line file cap holds.
//!
//! All implementations call directly into [`crate::fuzzy`] /
//! [`crate::api::fuzzy`] helpers so the MCP path is bit-identical to the
//! REST / TensaQL paths — every knob (`tnorm`, `aggregator`, `measure`)
//! round-trips through the same registry validation.
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
//!        [flaminio2026fsta] [faginhalpern1994fuzzyprob].

use serde_json::Value;

use crate::api::fuzzy::measure::StoredMeasure;
use crate::api::fuzzy::{
    load_workspace_config, save_workspace_config, FuzzyWorkspaceConfig, DEFAULT_TNORM,
    FUZZY_MEASURE_PREFIX,
};
use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::{
    Aggregator, MeanAggregator, MedianAggregator, OwaAggregator, TConormReduceAggregator,
    TNormReduceAggregator,
};
use crate::fuzzy::aggregation_choquet::{choquet, EXACT_N_CAP};
use crate::fuzzy::aggregation_learn::{default_measure_version, MeasureProvenance};
use crate::fuzzy::aggregation_measure::new_monotone;
use crate::fuzzy::hybrid::{
    build_hybrid_report, fuzzy_probability as run_fuzzy_probability, save_hybrid_result, FuzzyEvent,
    ProbDist,
};
use crate::fuzzy::registry::{AggregatorRegistry, TNormRegistry};
use crate::fuzzy::tnorm::TNormKind;

use super::embedded::EmbeddedBackend;

/// Build the KV key for a named measure: `fz/tn/measures/{name}`. Mirrors
/// `crate::api::fuzzy::measure::measure_key` (private in the REST module)
/// so embedded measure CRUD can reuse the same wire format.
fn measure_key(name: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(FUZZY_MEASURE_PREFIX.len() + name.len());
    k.extend_from_slice(FUZZY_MEASURE_PREFIX);
    k.extend_from_slice(name.as_bytes());
    k
}

impl EmbeddedBackend {
    // ─── Phase 10: fuzzy_probability ─────────────────────────────

    /// Implementation of the `fuzzy_probability` MCP tool. Mirrors the
    /// `POST /fuzzy/hybrid/probability` REST envelope —
    /// `{ value, event_kind, distribution_summary, query_id,
    /// narrative_id, tnorm }`.
    pub(crate) async fn fuzzy_probability_impl(
        &self,
        narrative_id: &str,
        event: Value,
        distribution: Value,
        tnorm: Option<&str>,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let tnorm_kind: TNormKind = match tnorm {
            None | Some("") => TNormKind::Godel,
            Some(name) => TNormRegistry::default().get(name)?,
        };
        let event: FuzzyEvent = serde_json::from_value(event)
            .map_err(|e| TensaError::InvalidInput(format!("invalid event payload: {e}")))?;
        let distribution: ProbDist = serde_json::from_value(distribution)
            .map_err(|e| TensaError::InvalidInput(format!("invalid distribution payload: {e}")))?;

        let hg = self.hypergraph();
        let value = run_fuzzy_probability(hg, narrative_id, &event, &distribution, tnorm_kind)?;
        let report = build_hybrid_report(narrative_id, &event, &distribution, tnorm_kind, value);
        let query_id = report.query_id;
        if let Err(e) = save_hybrid_result(hg.store(), narrative_id, &report) {
            tracing::warn!(
                narrative_id = %narrative_id,
                "MCP fuzzy_probability: persist failed: {e}"
            );
        }
        Ok(serde_json::json!({
            "value": value,
            "event_kind": report.event_kind,
            "distribution_summary": report.distribution_summary,
            "query_id": query_id,
            "narrative_id": narrative_id,
            "tnorm": report.tnorm,
        }))
    }

    // ─── Phase 11 Tool 1: fuzzy_list_tnorms ─────────────────────

    /// Enumerate registered t-norm families. Mirrors `GET /fuzzy/tnorms`.
    /// Returns `{ tnorms: [...] }` where each entry carries
    /// `{ name, description, formula, tconorm_formula, citation }`.
    pub(crate) async fn fuzzy_list_tnorms_impl(&self) -> Result<Value> {
        let reg = TNormRegistry::default();
        let mut names = reg.list();
        names.sort();
        let tnorms: Vec<Value> = names
            .iter()
            .filter_map(|n| describe_tnorm(n).map(|d| d.to_json()))
            .collect();
        Ok(serde_json::json!({"tnorms": tnorms}))
    }

    // ─── Phase 11 Tool 2: fuzzy_list_aggregators ─────────────────

    /// Enumerate registered aggregators. Mirrors `GET /fuzzy/aggregators`.
    pub(crate) async fn fuzzy_list_aggregators_impl(&self) -> Result<Value> {
        let reg = AggregatorRegistry::default();
        let mut names = reg.list();
        names.sort();
        let aggregators: Vec<Value> = names
            .iter()
            .filter_map(|n| describe_aggregator(n).map(|d| d.to_json()))
            .collect();
        Ok(serde_json::json!({"aggregators": aggregators}))
    }

    // ─── Phase 11 Tool 3: fuzzy_get_config ──────────────────────

    /// Load the persisted workspace fuzzy config. Mirrors
    /// `GET /fuzzy/config`; absent / corrupt records fall back to the
    /// Gödel / Mean factory default (same as the REST endpoint).
    pub(crate) async fn fuzzy_get_config_impl(&self) -> Result<Value> {
        let cfg = load_workspace_config(self.hypergraph().store());
        serde_json::to_value(&cfg).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    // ─── Phase 11 Tool 4: fuzzy_set_config ──────────────────────

    /// Update the workspace fuzzy config. Mirrors `PUT /fuzzy/config`.
    /// `reset = true` short-circuits to the Gödel / Mean default.
    /// Unknown t-norm / aggregator names surface as `InvalidInput` (HTTP
    /// 400 on the REST surface).
    pub(crate) async fn fuzzy_set_config_impl(
        &self,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
        measure: Option<Option<&str>>,
        reset: bool,
    ) -> Result<Value> {
        let store = self.hypergraph().store();
        if reset {
            let defaults = FuzzyWorkspaceConfig::default();
            save_workspace_config(store, &defaults)?;
            return serde_json::to_value(&defaults)
                .map_err(|e| TensaError::Serialization(e.to_string()));
        }
        let mut cfg = load_workspace_config(store);
        if let Some(t) = tnorm {
            cfg.tnorm = t.to_string();
        }
        if let Some(a) = aggregator {
            cfg.aggregator = a.to_string();
        }
        if let Some(m) = measure {
            cfg.measure = m.map(|s| s.to_string());
        }
        if cfg.tnorm.trim().is_empty() || cfg.aggregator.trim().is_empty() {
            return Err(TensaError::InvalidInput(
                "fuzzy config must carry non-empty tnorm and aggregator fields".into(),
            ));
        }
        save_workspace_config(store, &cfg)?;
        serde_json::to_value(&cfg).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    // ─── Phase 11 Tool 5: fuzzy_create_measure ──────────────────

    /// Validate + persist a named monotone fuzzy measure. Mirrors
    /// `POST /fuzzy/measures`. Rejects non-monotone measures with a
    /// message containing the literal word "monotonicity" (Phase 4
    /// test-assert contract).
    pub(crate) async fn fuzzy_create_measure_impl(
        &self,
        name: &str,
        n: u8,
        values: Vec<f64>,
    ) -> Result<Value> {
        let name = name.trim();
        if name.is_empty() {
            return Err(TensaError::InvalidInput(
                "measure name must be non-empty".into(),
            ));
        }
        if name.contains('/') || name.contains('\n') || name.contains(' ') {
            return Err(TensaError::InvalidInput(format!(
                "measure name '{name}' contains invalid characters ('/', whitespace, or newlines)"
            )));
        }
        let measure = new_monotone(n, values)?;
        let stored = StoredMeasure {
            name: name.to_string(),
            measure,
            version: default_measure_version(),
            provenance: MeasureProvenance::Manual,
        };
        let bytes = serde_json::to_vec(&stored)
            .map_err(|e| TensaError::Serialization(e.to_string()))?;
        self.hypergraph().store().put(&measure_key(name), &bytes)?;
        serde_json::to_value(&stored).map_err(|e| TensaError::Serialization(e.to_string()))
    }

    // ─── Phase 11 Tool 6: fuzzy_list_measures ───────────────────

    /// Enumerate persisted fuzzy measures. Mirrors `GET /fuzzy/measures`.
    pub(crate) async fn fuzzy_list_measures_impl(&self) -> Result<Value> {
        let pairs = self.hypergraph().store().prefix_scan(FUZZY_MEASURE_PREFIX)?;
        let measures: Vec<StoredMeasure> = pairs
            .into_iter()
            .filter_map(|(_, v)| serde_json::from_slice::<StoredMeasure>(&v).ok())
            .collect();
        Ok(serde_json::json!({"measures": measures}))
    }

    // ─── Phase 11 Tool 7: fuzzy_aggregate ───────────────────────

    /// One-shot aggregation over a caller-supplied vector. Mirrors
    /// `POST /fuzzy/aggregate`. Enforces `|xs| ≤ 1000` (above this the
    /// REST cap routes through the worker pool) and per-aggregator shape
    /// constraints (Choquet n ≤ 16, OWA weight-length = |xs|, etc.).
    pub(crate) async fn fuzzy_aggregate_impl(
        &self,
        xs: Vec<f64>,
        aggregator: &str,
        tnorm: Option<&str>,
        measure: Option<&str>,
        owa_weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> Result<Value> {
        const MAX_AGGREGATE_LEN: usize = 1000;
        if xs.is_empty() {
            return Err(TensaError::InvalidInput("aggregate: xs must be non-empty".into()));
        }
        if xs.len() > MAX_AGGREGATE_LEN {
            return Err(TensaError::InvalidInput(format!(
                "aggregate: |xs|={} exceeds the synchronous cap of {MAX_AGGREGATE_LEN}",
                xs.len()
            )));
        }
        for (i, v) in xs.iter().enumerate() {
            if !v.is_finite() {
                return Err(TensaError::InvalidInput(format!(
                    "aggregate: xs[{i}] is not finite ({v})"
                )));
            }
        }
        let agg_name = aggregator.trim().to_lowercase();
        let registry = AggregatorRegistry::default();
        if registry.get(&agg_name).is_err() {
            return Err(TensaError::InvalidInput(format!(
                "aggregate: unknown aggregator '{agg_name}'"
            )));
        }
        let (value, tnorm_name, measure_name, std_err) = match agg_name.as_str() {
            "mean" => (MeanAggregator.aggregate(&xs)?, None, None, None),
            "median" => (MedianAggregator.aggregate(&xs)?, None, None, None),
            "owa" => {
                let weights = owa_weights.ok_or_else(|| {
                    TensaError::InvalidInput(
                        "aggregate: owa requires 'owa_weights' of length |xs|".into(),
                    )
                })?;
                (OwaAggregator { weights }.aggregate(&xs)?, None, None, None)
            }
            "choquet" => {
                let mname = measure.ok_or_else(|| {
                    TensaError::InvalidInput(
                        "aggregate: choquet requires a 'measure' reference".into(),
                    )
                })?;
                let stored = self
                    .hypergraph()
                    .store()
                    .get(&measure_key(mname))?
                    .ok_or_else(|| {
                        TensaError::InvalidInput(format!(
                            "aggregate: no measure named '{mname}' — create one via fuzzy_create_measure"
                        ))
                    })?;
                let stored: StoredMeasure = serde_json::from_slice(&stored)
                    .map_err(|e| TensaError::Serialization(e.to_string()))?;
                if xs.len() != stored.measure.n as usize {
                    return Err(TensaError::InvalidInput(format!(
                        "aggregate: choquet requires |xs|={} to equal measure.n={}",
                        xs.len(),
                        stored.measure.n
                    )));
                }
                if xs.len() > 16 {
                    return Err(TensaError::InvalidInput(format!(
                        "aggregate: choquet |xs|={} exceeds the cap of 16; exact path caps at n ≤ {EXACT_N_CAP}",
                        xs.len()
                    )));
                }
                let r = choquet(&xs, &stored.measure, seed.unwrap_or(0))?;
                (r.value, None, Some(mname.to_string()), r.std_err)
            }
            "tnorm_reduce" => {
                let (kind, name) = resolve_tnorm(tnorm)?;
                let v = TNormReduceAggregator { kind }.aggregate(&xs).unwrap_or(0.0);
                (v, Some(name), None, None)
            }
            "tconorm_reduce" => {
                let (kind, name) = resolve_tnorm(tnorm)?;
                let v = TConormReduceAggregator { kind }.aggregate(&xs).unwrap_or(0.0);
                (v, Some(name), None, None)
            }
            other => {
                return Err(TensaError::InvalidInput(format!(
                    "aggregate: aggregator '{other}' is registered but not reachable"
                )))
            }
        };
        let mut out = serde_json::Map::new();
        out.insert("value".into(), serde_json::json!(value));
        out.insert("aggregator_name".into(), Value::String(agg_name));
        if let Some(t) = tnorm_name {
            out.insert("tnorm_name".into(), Value::String(t));
        }
        if let Some(m) = measure_name {
            out.insert("measure_name".into(), Value::String(m));
        }
        if let Some(s) = std_err {
            out.insert("std_err".into(), serde_json::json!(s));
        }
        Ok(Value::Object(out))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn resolve_tnorm(name: Option<&str>) -> Result<(TNormKind, String)> {
    let n = name
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_TNORM)
        .to_lowercase();
    let kind = TNormRegistry::default().get(&n)?;
    Ok((kind, n))
}

/// Minimal t-norm descriptor, mirrors `crate::api::fuzzy::tnorm::TNormInfo`.
/// Re-declared here because the REST module is gated behind `server`.
struct TNormDescriptor {
    name: &'static str,
    description: &'static str,
    formula: &'static str,
    tconorm_formula: &'static str,
    citation: &'static str,
}

impl TNormDescriptor {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "formula": self.formula,
            "tconorm_formula": self.tconorm_formula,
            "citation": self.citation,
        })
    }
}

fn describe_tnorm(name: &str) -> Option<TNormDescriptor> {
    match name {
        "godel" => Some(TNormDescriptor {
            name: "godel",
            description: "Gödel (minimum) t-norm — the maximal t-norm.",
            formula: "T(a, b) = min(a, b)",
            tconorm_formula: "S(a, b) = max(a, b)",
            citation: "klement2000",
        }),
        "goguen" => Some(TNormDescriptor {
            name: "goguen",
            description: "Goguen (product) t-norm — matches probabilistic AND.",
            formula: "T(a, b) = a * b",
            tconorm_formula: "S(a, b) = a + b - a*b",
            citation: "klement2000",
        }),
        "lukasiewicz" => Some(TNormDescriptor {
            name: "lukasiewicz",
            description: "Łukasiewicz (bounded-difference) t-norm.",
            formula: "T(a, b) = max(0, a + b - 1)",
            tconorm_formula: "S(a, b) = min(1, a + b)",
            citation: "klement2000",
        }),
        "hamacher" => Some(TNormDescriptor {
            name: "hamacher",
            description:
                "Hamacher family parameterised by λ ≥ 0. λ = 1 ≡ Goguen; λ = 0 = Hamacher product.",
            formula: "T(a, b) = ab / (λ + (1 - λ)(a + b - ab))",
            tconorm_formula: "S(a, b) = 1 - T(1 - a, 1 - b)",
            citation: "klement2000",
        }),
        _ => None,
    }
}

struct AggregatorDescriptor {
    name: &'static str,
    description: &'static str,
    formula: &'static str,
    required_params: &'static [&'static str],
    citation: &'static str,
}

impl AggregatorDescriptor {
    fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "formula": self.formula,
            "required_params": self.required_params,
            "citation": self.citation,
        })
    }
}

fn describe_aggregator(name: &str) -> Option<AggregatorDescriptor> {
    match name {
        "mean" => Some(AggregatorDescriptor {
            name: "mean",
            description: "Arithmetic mean (xs treated as a uniform sample).",
            formula: "A(xs) = (Σ xs) / n",
            required_params: &[],
            citation: "klement2000",
        }),
        "median" => Some(AggregatorDescriptor {
            name: "median",
            description: "Sample median — midpoint of the middle value(s).",
            formula: "A(xs) = median(xs)",
            required_params: &[],
            citation: "klement2000",
        }),
        "owa" => Some(AggregatorDescriptor {
            name: "owa",
            description: "Yager OWA — sort descending, then dot-product with weights.",
            formula: "A(xs) = Σ w_i * x_(i)",
            required_params: &["owa_weights"],
            citation: "yager1988owa",
        }),
        "choquet" => Some(AggregatorDescriptor {
            name: "choquet",
            description:
                "Choquet integral over a fuzzy measure. Exact for n ≤ 10; MC otherwise.",
            formula: "A(xs; μ) = Σ (x_(i) - x_(i-1)) * μ(A_i)",
            required_params: &["measure"],
            citation: "grabisch1996choquet",
        }),
        "tnorm_reduce" => Some(AggregatorDescriptor {
            name: "tnorm_reduce",
            description: "Left-fold under a t-norm (logical conjunction).",
            formula: "A(xs) = T(T(...T(x_1, x_2)..., x_{n-1}), x_n)",
            required_params: &["tnorm"],
            citation: "klement2000",
        }),
        "tconorm_reduce" => Some(AggregatorDescriptor {
            name: "tconorm_reduce",
            description: "Left-fold under a t-conorm (logical disjunction).",
            formula: "A(xs) = S(S(...S(x_1, x_2)..., x_{n-1}), x_n)",
            required_params: &["tnorm"],
            citation: "klement2000",
        }),
        _ => None,
    }
}

// Tests live in [`super::embedded_fuzzy_tests`] to keep this file under
// the 500-line cap.
