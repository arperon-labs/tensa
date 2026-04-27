//! Fuzzy Sprint Phase 11 — embedded MCP impls for the "higher-layer"
//! fuzzy tools: graded Allen, intermediate quantifiers, graded syllogisms,
//! fuzzy FCA lattices, Mamdani rule CRUD + evaluation.
//!
//! Split from [`super::embedded_fuzzy`] to keep the 500-line file cap.
//! Each `*_impl` mirrors the corresponding REST endpoint contract so the
//! MCP path is bit-identical to the TensaQL / REST path.
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
//!        [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen]
//!        [novak2008quantifiers] [murinovanovak2014peterson]
//!        [belohlavek2004fuzzyfca] [mamdani1975mamdani].

use std::str::FromStr;

use serde_json::Value;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::allen::{graded_relation, index_to_relation, save_fuzzy_allen};
use crate::fuzzy::fca::{
    build_lattice_with_threshold, FormalContext, FormalContextOptions,
};
use crate::fuzzy::fca_store::save_concept_lattice;
use crate::fuzzy::quantifier::{
    evaluate as eval_quantifier, predicate_hash, quantifier_from_name, save_quantifier_result,
    QuantifierResult,
};
use crate::fuzzy::registry::TNormRegistry;
use crate::fuzzy::rules::{
    build_rule, evaluate_rules_against_entity, save_rule, validate_rule, FuzzyCondition,
    FuzzyOutput, RuleSetEvaluation,
};
use crate::fuzzy::syllogism::{
    parse_statement, save_syllogism_proof, verify as verify_syllogism_core, Syllogism,
    SyllogismProof, TypePredicateResolver,
};
use crate::fuzzy::tnorm::TNormKind;
use crate::types::{Entity, EntityType, MaturityLevel};

use super::embedded::EmbeddedBackend;

/// Default validity threshold for syllogism verification (matches the
/// Phase 7 REST handler default).
const DEFAULT_SYLLOGISM_THRESHOLD: f64 = 0.5;

impl EmbeddedBackend {
    // ─── Phase 11 Tool 8: fuzzy_allen_gradation ─────────────────

    /// Compute + cache the 13-dim graded Allen relation vector between
    /// two situations. Mirrors `POST /analysis/fuzzy-allen` — returns
    /// `{ relations: [{name, degree}, ...] }` in Allen's canonical order.
    pub(crate) async fn fuzzy_allen_gradation_impl(
        &self,
        narrative_id: &str,
        a_id: &str,
        b_id: &str,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let a_uuid: Uuid = a_id.parse().map_err(|e| {
            TensaError::InvalidInput(format!("a_id '{a_id}' not a UUID: {e}"))
        })?;
        let b_uuid: Uuid = b_id.parse().map_err(|e| {
            TensaError::InvalidInput(format!("b_id '{b_id}' not a UUID: {e}"))
        })?;
        let hg = self.hypergraph();
        let a = hg.get_situation(&a_uuid)?;
        let b = hg.get_situation(&b_uuid)?;
        let vector = graded_relation(&a.temporal, &b.temporal);
        save_fuzzy_allen(hg.store(), narrative_id, &a_uuid, &b_uuid, &vector)?;
        let rels: Vec<Value> = (0..13)
            .map(|i| {
                let rel = index_to_relation(i).expect("index in range");
                serde_json::json!({"name": format!("{:?}", rel), "degree": vector[i]})
            })
            .collect();
        Ok(serde_json::json!({"relations": rels}))
    }

    // ─── Phase 11 Tool 9: fuzzy_quantify ────────────────────────

    /// Evaluate an intermediate quantifier (`most` / `many` / `few` /
    /// `almost_all`) over a narrative's entity domain. Mirrors
    /// `POST /fuzzy/quantify` — crisp predicate only (Phase 6 scope).
    pub(crate) async fn fuzzy_quantify_impl(
        &self,
        narrative_id: &str,
        quantifier: &str,
        entity_type: Option<&str>,
        where_spec: Option<&str>,
        label: Option<&str>,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let q = quantifier_from_name(quantifier)?;
        let et: Option<EntityType> = match entity_type {
            None | Some("") => None,
            Some(s) => Some(EntityType::from_str(s)?),
        };
        let hg = self.hypergraph();
        let entities = hg.list_entities_by_narrative(narrative_id)?;
        let domain: Vec<Entity> = match et {
            Some(t) => entities.into_iter().filter(|e| e.entity_type == t).collect(),
            None => entities,
        };
        let n = domain.len();
        let spec = where_spec.unwrap_or("").trim();
        let mut sum = 0.0_f64;
        for e in &domain {
            sum += eval_entity_predicate(e, spec).clamp(0.0, 1.0);
        }
        let r = if n == 0 { 0.0 } else { sum / (n as f64) };
        let value = eval_quantifier(q, r);
        let hash = predicate_hash(spec, q, entity_type);
        let result = QuantifierResult {
            quantifier: q.name().to_string(),
            value,
            label: label.map(|s| s.to_string()),
        };
        if let Err(e) = save_quantifier_result(hg.store(), narrative_id, &hash, &result) {
            tracing::warn!(
                narrative_id = %narrative_id,
                predicate_hash = %hash,
                "MCP fuzzy_quantify: persist failed: {e}"
            );
        }
        let mut out = serde_json::json!({
            "value": value,
            "quantifier_name": q.name(),
            "predicate_hash": hash,
            "narrative_id": narrative_id,
            "domain_size": n,
            "cardinality_ratio": r,
            "cache_hit": false,
        });
        if let Some(l) = label {
            out["label"] = Value::String(l.to_string());
        }
        Ok(out)
    }

    // ─── Phase 11 Tool 10: fuzzy_verify_syllogism ───────────────

    /// Verify a graded Peterson syllogism. Mirrors
    /// `POST /fuzzy/syllogism/verify` — parses the three tiny-DSL
    /// statements, runs `verify()` under the configured t-norm +
    /// threshold, persists the proof, returns
    /// `{ proof_id, narrative_id, degree, figure, valid, threshold, tnorm }`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn fuzzy_verify_syllogism_impl(
        &self,
        narrative_id: &str,
        major: &str,
        minor: &str,
        conclusion: &str,
        threshold: Option<f64>,
        tnorm: Option<&str>,
        figure_hint: Option<&str>,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let major = parse_statement(major)?;
        let minor = parse_statement(minor)?;
        let conclusion = parse_statement(conclusion)?;
        let tnorm_name = tnorm
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("godel")
            .to_string();
        let tnorm_kind: TNormKind = TNormRegistry::default().get(&tnorm_name)?;
        let threshold = threshold.unwrap_or(DEFAULT_SYLLOGISM_THRESHOLD);
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(TensaError::InvalidInput(format!(
                "threshold must be finite in [0,1]; got {threshold}"
            )));
        }
        let syl = Syllogism {
            major,
            minor,
            conclusion,
            figure_hint: figure_hint.map(|s| s.to_string()),
        };
        let hg = self.hypergraph();
        let gv = verify_syllogism_core(
            hg,
            narrative_id,
            &syl,
            tnorm_kind,
            threshold,
            &TypePredicateResolver,
        )?;
        let proof = SyllogismProof::new(syl, gv.clone());
        if let Err(e) = save_syllogism_proof(hg.store(), narrative_id, &proof) {
            tracing::warn!(
                narrative_id = %narrative_id,
                proof_id = %proof.id,
                "MCP fuzzy_verify_syllogism: persist failed: {e}"
            );
        }
        Ok(serde_json::json!({
            "proof_id": proof.id,
            "narrative_id": narrative_id,
            "degree": gv.degree,
            "figure": gv.figure,
            "valid": gv.valid,
            "threshold": gv.threshold,
            "tnorm": tnorm_kind.name(),
        }))
    }

    // ─── Phase 11 Tool 11: fuzzy_build_lattice ──────────────────

    /// Build + persist a fuzzy concept lattice for a narrative. Mirrors
    /// `POST /fuzzy/fca/lattice`.
    pub(crate) async fn fuzzy_build_lattice_impl(
        &self,
        narrative_id: &str,
        entity_type: Option<&str>,
        attribute_allowlist: Option<Vec<String>>,
        threshold: Option<usize>,
        tnorm: Option<&str>,
        large_context: bool,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let tnorm_name = tnorm
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .unwrap_or("godel")
            .to_string();
        let tnorm_kind: TNormKind = TNormRegistry::default().get(&tnorm_name)?;
        let et: Option<EntityType> = match entity_type {
            None | Some("") => None,
            Some(s) => Some(EntityType::from_str(s).map_err(|e| {
                TensaError::InvalidInput(format!("entity_type '{s}' unknown: {e}"))
            })?),
        };
        let opts = FormalContextOptions {
            entity_type_filter: et,
            attribute_allowlist,
            large_context,
        };
        let hg = self.hypergraph();
        let ctx = FormalContext::from_hypergraph(hg, narrative_id, &opts)?;
        let mut lattice =
            build_lattice_with_threshold(&ctx, tnorm_kind, threshold.unwrap_or(0))?;
        lattice.narrative_id = narrative_id.to_string();
        if let Err(e) = save_concept_lattice(hg.store(), narrative_id, &lattice) {
            tracing::warn!(
                narrative_id = %narrative_id,
                lattice_id = %lattice.id,
                "MCP fuzzy_build_lattice: persist failed: {e}"
            );
        }
        Ok(serde_json::json!({
            "lattice_id": lattice.id,
            "narrative_id": lattice.narrative_id,
            "num_concepts": lattice.num_concepts(),
            "num_objects": lattice.objects.len(),
            "num_attributes": lattice.attributes.len(),
            "tnorm": tnorm_kind.name(),
        }))
    }

    // ─── Phase 11 Tool 12: fuzzy_create_rule ────────────────────

    /// Create + persist a Mamdani fuzzy rule. Mirrors `POST /fuzzy/rules`.
    /// `antecedent` / `consequent` are JSON values deserialized into
    /// `FuzzyCondition` / `FuzzyOutput`; invalid shapes surface as
    /// `InvalidInput` (HTTP 400 on REST).
    pub(crate) async fn fuzzy_create_rule_impl(
        &self,
        name: &str,
        narrative_id: &str,
        antecedent: Value,
        consequent: Value,
        tnorm: Option<&str>,
        enabled: Option<bool>,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        if name.trim().is_empty() {
            return Err(TensaError::InvalidInput("rule name is empty".into()));
        }
        let antecedent: Vec<FuzzyCondition> = serde_json::from_value(antecedent)
            .map_err(|e| TensaError::InvalidInput(format!("invalid antecedent: {e}")))?;
        let consequent: FuzzyOutput = serde_json::from_value(consequent)
            .map_err(|e| TensaError::InvalidInput(format!("invalid consequent: {e}")))?;
        let mut rule = build_rule(
            name.to_string(),
            narrative_id.to_string(),
            antecedent,
            consequent,
        );
        if let Some(t) = tnorm {
            let t = t.trim();
            if !t.is_empty() {
                rule.tnorm = TNormRegistry::default().get(t)?;
            }
        }
        if let Some(e) = enabled {
            rule.enabled = e;
        }
        validate_rule(&rule)?;
        save_rule(self.hypergraph().store(), narrative_id, &rule)?;
        Ok(serde_json::json!({
            "rule_id": rule.id,
            "name": rule.name,
            "narrative_id": rule.narrative_id,
        }))
    }

    // ─── Phase 11 Tool 13: fuzzy_evaluate_rules ─────────────────

    /// Evaluate the rule set for a narrative against a single entity.
    /// Mirrors `POST /fuzzy/rules/{nid}/evaluate`.
    pub(crate) async fn fuzzy_evaluate_rules_impl(
        &self,
        narrative_id: &str,
        entity_id: &str,
        rule_ids: Option<Vec<String>>,
        firing_aggregator: Option<AggregatorKind>,
    ) -> Result<Value> {
        if narrative_id.trim().is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let entity_uuid: Uuid = entity_id.parse().map_err(|e| {
            TensaError::InvalidInput(format!("entity_id '{entity_id}' not a UUID: {e}"))
        })?;
        let rule_ids_slice: Option<&[String]> = rule_ids.as_deref();
        let mut eval: RuleSetEvaluation = evaluate_rules_against_entity(
            self.hypergraph(),
            narrative_id,
            rule_ids_slice,
            &entity_uuid,
        )?;
        if let Some(kind) = firing_aggregator {
            let xs: Vec<f64> = eval.fired_rules.iter().map(|f| f.firing_strength).collect();
            let agg = crate::fuzzy::aggregation::aggregator_for(kind);
            match agg.aggregate(&xs) {
                Ok(v) => eval.firing_aggregate = Some(v),
                Err(e) => {
                    tracing::warn!("MCP fuzzy_evaluate_rules: firing aggregator failed ({e})");
                }
            }
        }
        serde_json::to_value(&eval).map_err(|e| TensaError::Serialization(e.to_string()))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────
//
// Tiny crisp-predicate evaluator. Mirrors `crate::api::fuzzy::quantify::
// build_entity_predicate` semantics without pulling the server-gated
// module into the default build.
//
// Supported forms:
//   * ""                            → 1.0 (always-true)
//   * "confidence>0.5"              → 1.0 iff e.confidence > 0.5
//   * "maturity=Confirmed"          → 1.0 iff e.maturity ≥ Validated
//   * "prop.path = \"value\""       → 1.0 iff walk(e.properties, path) == value

fn eval_entity_predicate(e: &Entity, spec: &str) -> f64 {
    let s = spec.trim();
    if s.is_empty() {
        return 1.0;
    }
    let Some((field, op, rhs)) = split_comparison(s) else {
        return 0.0;
    };
    let field = field.trim();
    let rhs = rhs.trim();
    match field {
        "confidence" => {
            let threshold: f64 = rhs.parse().unwrap_or(0.0);
            bool_to_mu(compare_f64(op, e.confidence as f64, threshold))
        }
        "maturity" => match maturity_from_name(rhs) {
            Some(level) => bool_to_mu(compare_ord(op, &e.maturity, &level)),
            None => 0.0,
        },
        _ => eval_property_path(&e.properties, field, op, rhs),
    }
}

fn maturity_from_name(s: &str) -> Option<MaturityLevel> {
    match s.trim().to_ascii_lowercase().as_str() {
        "candidate" => Some(MaturityLevel::Candidate),
        "reviewed" => Some(MaturityLevel::Reviewed),
        "validated" | "confirmed" | "verified" => Some(MaturityLevel::Validated),
        "groundtruth" | "ground_truth" | "ground-truth" => Some(MaturityLevel::GroundTruth),
        _ => None,
    }
}

fn split_comparison(s: &str) -> Option<(&str, &str, &str)> {
    for op in &[">=", "<=", "!=", ">", "<", "="] {
        if let Some(idx) = s.find(op) {
            let (lhs, rest) = s.split_at(idx);
            return Some((lhs, op, &rest[op.len()..]));
        }
    }
    None
}

fn compare_f64(op: &str, lhs: f64, rhs: f64) -> bool {
    match op {
        ">=" => lhs >= rhs,
        "<=" => lhs <= rhs,
        ">" => lhs > rhs,
        "<" => lhs < rhs,
        "=" => (lhs - rhs).abs() < f64::EPSILON,
        "!=" => (lhs - rhs).abs() >= f64::EPSILON,
        _ => false,
    }
}

fn compare_ord<T: Ord>(op: &str, lhs: &T, rhs: &T) -> bool {
    match op {
        "=" => lhs == rhs,
        "!=" => lhs != rhs,
        ">=" => lhs >= rhs,
        "<=" => lhs <= rhs,
        ">" => lhs > rhs,
        "<" => lhs < rhs,
        _ => false,
    }
}

fn bool_to_mu(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

fn eval_property_path(props: &serde_json::Value, field: &str, op: &str, rhs: &str) -> f64 {
    let mut val = props;
    for p in field.split('.') {
        match val {
            serde_json::Value::Object(m) => match m.get(p) {
                Some(v) => val = v,
                None => return 0.0,
            },
            _ => return 0.0,
        }
    }
    match val {
        serde_json::Value::Number(n) => {
            let threshold: f64 = rhs.parse().unwrap_or(0.0);
            bool_to_mu(compare_f64(op, n.as_f64().unwrap_or(0.0), threshold))
        }
        serde_json::Value::String(s) => {
            let stripped = rhs.trim_matches('"');
            match op {
                "=" => bool_to_mu(s == stripped),
                "!=" => bool_to_mu(s != stripped),
                _ => 0.0,
            }
        }
        serde_json::Value::Bool(b) => {
            let want: bool = rhs.parse().unwrap_or(false);
            match op {
                "=" => bool_to_mu(*b == want),
                "!=" => bool_to_mu(*b != want),
                _ => 0.0,
            }
        }
        _ => 0.0,
    }
}
