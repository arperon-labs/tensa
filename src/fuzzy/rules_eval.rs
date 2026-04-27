//! Mamdani-style fuzzy rule system — evaluation pipeline.
//!
//! Phase 9. Split from [`super::rules`] / [`super::rules_types`] to keep
//! each file under the 500-line cap.
//!
//! ### Evaluation pipeline
//!
//! For each rule in the set:
//! 1. **Fuzzification** — resolve every antecedent's `variable_path`
//!    against the entity and compute μ via the membership function.
//!    Missing / non-numeric paths yield μ = 0.
//! 2. **Firing strength** — fold the per-antecedent μ slice under the
//!    rule's t-norm (`reduce_tnorm`). Empty antecedent = neutral = 1.0.
//! 3. **Aggregation** — union of scaled consequent memberships across
//!    fired rules: `μ_agg(y) = max_k min(firing_k, μ_consequent_k(y))`.
//!    This is the classical Mamdani max-min inference shape; the
//!    per-rule cap (`min`) matches Mamdani's original 1975 paper.
//! 4. **Defuzzification** — Centroid (centre of area) or Mean-of-Maxima
//!    over the discretised output fuzzy set.
//!
//! Cites: [mamdani1975mamdani].

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::aggregator_for;
use crate::fuzzy::rules_types::{
    Defuzzification, FiredRule, FuzzyCondition, FuzzyOutput, MamdaniRule, MamdaniRuleSet,
    RuleSetEvaluation, DEFAULT_DEFUZZ_BINS,
};
use crate::fuzzy::tnorm::reduce_tnorm;
use crate::types::Entity;

// ── Variable resolution ──────────────────────────────────────────────────────

/// Resolve a `variable_path` against an entity and return its numeric
/// value (as `f64`), or `None` when the path does not resolve to a
/// number / bool / string-indicator.
///
/// Resolution rules:
/// * `"entity.confidence"` → `entity.confidence as f64`.
/// * `"entity.entity_type"` → `1.0` always (indicator of presence). A
///   future extension can route through a user-supplied enum lookup;
///   the current use-case is "resolve the variable to compute
///   membership over" and the entity-type membership function would
///   be domain-specific (see Phase 9 deferrals).
/// * `"entity.properties.<dot.path>"` → descent through JSON
///   properties. `entity.properties.` prefix is optional — a bare
///   `"score"` resolves to `entity.properties.score`.
/// * Numeric → value. Boolean → 1.0 / 0.0. String → 1.0 when present
///   (indicator), 0.0 when missing.
pub fn resolve_variable(entity: &Entity, path: &str) -> Option<f64> {
    let path = path.trim();
    if path.is_empty() {
        return None;
    }

    if path == "entity.confidence" {
        return Some(entity.confidence as f64);
    }
    if path == "entity.entity_type" {
        return Some(1.0);
    }

    // Strip optional "entity.properties." / "entity." / "properties." prefix.
    let mut rest = path;
    for prefix in ["entity.properties.", "entity.", "properties."] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped;
            break;
        }
    }
    if rest.is_empty() {
        return None;
    }

    let mut current = &entity.properties;
    for segment in rest.split('.') {
        match current {
            serde_json::Value::Object(obj) => match obj.get(segment) {
                Some(v) => current = v,
                None => return None,
            },
            _ => return None,
        }
    }

    match current {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        serde_json::Value::String(_) => Some(1.0),
        _ => None,
    }
}

// ── Firing strength ──────────────────────────────────────────────────────────

/// Fuzzify one condition: resolve the variable, compute μ.
pub fn fuzzify_condition(entity: &Entity, cond: &FuzzyCondition) -> f64 {
    match resolve_variable(entity, &cond.variable_path) {
        Some(x) => cond.membership.membership(x),
        None => 0.0,
    }
}

/// Compute a rule's firing strength: reduce the per-antecedent μ slice
/// under the rule's t-norm. Empty antecedents fire at neutral = 1.0
/// (this matches the "rule with no preconditions always fires"
/// semantics of classical production systems).
pub fn firing_strength(entity: &Entity, rule: &MamdaniRule) -> (f64, Vec<f64>) {
    let per_antecedent: Vec<f64> = rule
        .antecedent
        .iter()
        .map(|c| fuzzify_condition(entity, c))
        .collect();
    let strength = reduce_tnorm(rule.tnorm, &per_antecedent);
    (strength, per_antecedent)
}

// ── Aggregation + defuzzification ────────────────────────────────────────────

/// Build a single unified discretised output fuzzy set from the rules'
/// individual consequents, scaled by their firing strengths.
///
/// All consequents in the rule set are assumed to target the same
/// output variable (callers that want to defuzzify multiple output
/// variables should partition their rules into disjoint rule sets).
/// The support interval is the union of every consequent's support.
///
/// Returns `(xs, mus)` with `xs.len() == mus.len() == bins`.
pub fn aggregate_consequents(
    fired: &[(f64, &FuzzyOutput)],
    bins: usize,
) -> (Vec<f64>, Vec<f64>) {
    if fired.is_empty() || bins == 0 {
        return (Vec::new(), Vec::new());
    }
    // Union of supports.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (_, out) in fired {
        let (l, h) = out.membership.support();
        if l < lo {
            lo = l;
        }
        if h > hi {
            hi = h;
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return (Vec::new(), Vec::new());
    }
    let step = (hi - lo) / (bins as f64);
    let mut xs = Vec::with_capacity(bins);
    let mut mus = Vec::with_capacity(bins);
    for i in 0..bins {
        let x = lo + (i as f64 + 0.5) * step;
        let mu_out = fired.iter().fold(0.0_f64, |acc, (strength, out)| {
            let mu = out.membership.membership(x).min(*strength);
            acc.max(mu)
        });
        xs.push(x);
        mus.push(mu_out);
    }
    (xs, mus)
}

/// Centre-of-area over the aggregated discretised fuzzy set. Returns
/// `None` when the total mass is zero (no rule fired with non-trivial
/// strength).
pub fn centroid_defuzz(xs: &[f64], mus: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.len() != mus.len() {
        return None;
    }
    let total: f64 = mus.iter().sum();
    if total <= 1e-12 {
        return None;
    }
    let numerator: f64 = xs.iter().zip(mus).map(|(x, m)| x * m).sum();
    Some(numerator / total)
}

/// Mean-of-maxima — mean x over the bins where aggregated μ is maximal.
pub fn mean_of_maxima_defuzz(xs: &[f64], mus: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.len() != mus.len() {
        return None;
    }
    let max_mu = mus.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_mu <= 1e-12 {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for (x, m) in xs.iter().zip(mus) {
        if (max_mu - *m).abs() <= 1e-9 {
            sum += x;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Dispatch defuzzification by strategy.
pub fn defuzzify(strategy: Defuzzification, xs: &[f64], mus: &[f64]) -> Option<f64> {
    match strategy {
        Defuzzification::Centroid => centroid_defuzz(xs, mus),
        Defuzzification::MeanOfMaxima => mean_of_maxima_defuzz(xs, mus),
    }
}

// ── Rule-set evaluation entry point ──────────────────────────────────────────

/// Evaluate a rule set against one entity.
///
/// All rules must be scoped to the entity (the caller is responsible
/// for filtering by `narrative_id` upstream). Disabled rules are
/// skipped. Returns a [`RuleSetEvaluation`] with the per-rule firing
/// strengths and the optional defuzzified output.
pub fn evaluate_rule_set(
    rule_set: &MamdaniRuleSet,
    entity: &Entity,
) -> Result<RuleSetEvaluation> {
    let mut fired: Vec<FiredRule> = Vec::with_capacity(rule_set.rules.len());
    let mut scaled: Vec<(f64, &FuzzyOutput)> = Vec::with_capacity(rule_set.rules.len());
    for rule in &rule_set.rules {
        if !rule.enabled {
            continue;
        }
        let (strength, per_antecedent) = firing_strength(entity, rule);
        fired.push(FiredRule {
            rule_id: rule.id,
            rule_name: rule.name.clone(),
            firing_strength: strength,
            per_antecedent_mu: per_antecedent,
        });
        if strength > 0.0 {
            scaled.push((strength, &rule.consequent));
        }
    }
    let (xs, mus) = aggregate_consequents(&scaled, DEFAULT_DEFUZZ_BINS);
    let defuzzified = defuzzify(rule_set.defuzzification, &xs, &mus);

    let firing_aggregate = match rule_set.firing_aggregator.clone() {
        None => None,
        Some(kind) => {
            let xs: Vec<f64> = fired.iter().map(|f| f.firing_strength).collect();
            let agg = aggregator_for(kind);
            match agg.aggregate(&xs) {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!(
                        "firing_aggregator failed ({e}); continuing without summary scalar"
                    );
                    None
                }
            }
        }
    };

    Ok(RuleSetEvaluation {
        entity_id: entity.id,
        fired_rules: fired,
        defuzzified_output: defuzzified,
        defuzzification: rule_set.defuzzification,
        firing_aggregate,
    })
}

/// Evaluate a single rule against one entity. Convenience wrapper used
/// by ad-hoc pipelines (alerts, ingestion post-pass) that want a
/// single-rule firing strength without building a rule set.
pub fn evaluate_single_rule(rule: &MamdaniRule, entity: &Entity) -> RuleSetEvaluation {
    let set = MamdaniRuleSet::new(vec![rule.clone()]);
    match evaluate_rule_set(&set, entity) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("single-rule eval failed ({e})");
            RuleSetEvaluation {
                entity_id: entity.id,
                fired_rules: Vec::new(),
                defuzzified_output: None,
                defuzzification: Defuzzification::default(),
                firing_aggregate: None,
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a default Gödel-tnorm rule from raw components. Thin
/// convenience used by tests + REST handlers.
pub fn build_rule(
    name: impl Into<String>,
    narrative_id: impl Into<String>,
    antecedent: Vec<FuzzyCondition>,
    consequent: FuzzyOutput,
) -> MamdaniRule {
    MamdaniRule {
        id: uuid::Uuid::now_v7(),
        name: name.into(),
        narrative_id: narrative_id.into(),
        antecedent,
        consequent,
        tnorm: crate::fuzzy::tnorm::TNormKind::Godel,
        created_at: chrono::Utc::now(),
        enabled: true,
    }
}

/// Validate a rule's shape — empty-name or empty-narrative rejections
/// surfaced as [`TensaError::InvalidInput`] so REST handlers can
/// respond with HTTP 400.
pub fn validate_rule(rule: &MamdaniRule) -> Result<()> {
    if rule.name.trim().is_empty() {
        return Err(TensaError::InvalidInput("rule name is empty".into()));
    }
    if rule.narrative_id.trim().is_empty() {
        return Err(TensaError::InvalidInput("narrative_id is empty".into()));
    }
    if rule.antecedent.is_empty() {
        return Err(TensaError::InvalidInput(
            "rule antecedent is empty — at least one condition required".into(),
        ));
    }
    for (i, cond) in rule.antecedent.iter().enumerate() {
        if cond.variable_path.trim().is_empty() {
            return Err(TensaError::InvalidInput(format!(
                "antecedent[{i}] variable_path is empty"
            )));
        }
    }
    if rule.consequent.variable.trim().is_empty() {
        return Err(TensaError::InvalidInput(
            "consequent variable is empty".into(),
        ));
    }
    Ok(())
}
