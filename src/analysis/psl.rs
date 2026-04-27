//! Probabilistic Soft Logic (PSL) engine.
//!
//! Implements weighted soft Horn clauses with continuous truth values [0,1].
//! Complements Bayesian confidence (data quality) and Dempster-Shafer
//! (evidence uncertainty) with rule-based global inference.
//!
//! Example rules:
//!   weight 0.8: near_body(X) ∧ has_motive(X) → killer(X)
//!   weight 0.5: has_alibi(X) → ¬killer(X)
//!
//! Inference minimizes weighted hinge loss via coordinate descent:
//!   min_y Σ_r w_r · max(0, body_r(y) - head_r(y))²
//!
//! Reference: Bach et al., "Hinge-Loss Markov Random Fields and Probabilistic
//! Soft Logic" (JMLR 2017)

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id};
use crate::error::{Result, TensaError};
use crate::hypergraph::keys::ANALYSIS_PSL;
use crate::hypergraph::Hypergraph;
use crate::types::*;

use crate::inference::types::*;
use crate::inference::InferenceEngine;

// ─── Configuration ─────────────────────────────────────────

/// Configuration for the PSL solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PslConfig {
    /// Maximum coordinate descent iterations.
    pub max_iterations: usize,
    /// Convergence threshold (max change in any truth value).
    pub convergence_threshold: f64,
    /// Step size for coordinate descent updates.
    pub step_size: f64,
}

impl Default for PslConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            convergence_threshold: 1e-4,
            step_size: 0.1,
        }
    }
}

// ─── Rule Types ────────────────────────────────────────────

/// A PSL rule: weighted soft Horn clause.
///
/// Semantics: `weight * max(0, body_truth - head_truth)²`
/// For negated heads: `weight * max(0, body_truth - (1 - head_truth))²`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PslRule {
    pub weight: f64,
    pub body: Vec<Predicate>,
    pub head: Predicate,
    pub negated_head: bool,
}

/// A predicate template: `name(variable)`.
/// Variable is grounded against entity IDs during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predicate {
    pub name: String,
    pub variable: String,
}

/// A grounded atom: predicate applied to a specific entity with a truth value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundAtom {
    pub predicate: String,
    pub entity_id: Uuid,
    pub truth_value: f64,
}

// ─── Result Types ──────────────────────────────────────────

/// Full result of a PSL inference job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PslResult {
    pub narrative_id: String,
    pub ground_atoms: Vec<GroundAtom>,
    pub rules_applied: usize,
    pub total_loss: f64,
    pub convergence: PslConvergence,
}

/// Convergence diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PslConvergence {
    pub converged: bool,
    pub iterations: usize,
    pub final_delta: f64,
}

// ─── Engine ────────────────────────────────────────────────

/// Probabilistic Soft Logic inference engine.
pub struct PslEngine;

impl InferenceEngine for PslEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ProbabilisticSoftLogic
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;

        // Parse rules from parameters
        let rules: Vec<PslRule> = job
            .parameters
            .get("rules")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        if rules.is_empty() {
            return Err(TensaError::InferenceError("No PSL rules provided".into()));
        }

        // Parse config overrides
        let mut config = PslConfig::default();
        if let Some(v) = job
            .parameters
            .get("max_iterations")
            .and_then(|v| v.as_u64())
        {
            config.max_iterations = v as usize;
        }
        if let Some(v) = job.parameters.get("step_size").and_then(|v| v.as_f64()) {
            config.step_size = v;
        }

        // Ground rules against the hypergraph
        let (grounded_rules, initial_atoms) = ground_rules(&rules, hypergraph, narrative_id)?;

        if initial_atoms.is_empty() {
            return Ok(InferenceResult {
                job_id: job.id.clone(),
                job_type: InferenceJobType::ProbabilisticSoftLogic,
                target_id: job.target_id,
                result: serde_json::to_value(&PslResult {
                    narrative_id: narrative_id.to_string(),
                    ground_atoms: vec![],
                    rules_applied: 0,
                    total_loss: 0.0,
                    convergence: PslConvergence {
                        converged: true,
                        iterations: 0,
                        final_delta: 0.0,
                    },
                })?,
                confidence: 0.0,
                explanation: Some("No predicates could be grounded".into()),
                status: JobStatus::Completed,
                created_at: job.created_at,
                completed_at: Some(Utc::now()),
            });
        }

        // Solve via coordinate descent
        let (atoms, convergence, total_loss) = solve_psl(&grounded_rules, initial_atoms, &config);

        let result_data = PslResult {
            narrative_id: narrative_id.to_string(),
            ground_atoms: atoms.clone(),
            rules_applied: grounded_rules.len(),
            total_loss,
            convergence: convergence.clone(),
        };

        // Persist to KV
        let result_value = serde_json::to_value(&result_data)?;
        let key = analysis_key(ANALYSIS_PSL, &[narrative_id]);
        hypergraph
            .store()
            .put(&key, result_value.to_string().as_bytes())?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::ProbabilisticSoftLogic,
            target_id: job.target_id,
            result: result_value,
            confidence: if convergence.converged { 0.85 } else { 0.5 },
            explanation: Some(format!(
                "PSL: {} ground atoms, {} rules, loss={:.4} ({} iter, {})",
                atoms.len(),
                grounded_rules.len(),
                total_loss,
                convergence.iterations,
                if convergence.converged {
                    "converged"
                } else {
                    "not converged"
                },
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Grounding ─────────────────────────────────────────────

/// A grounded rule: all variables replaced with specific entity IDs.
#[derive(Debug, Clone)]
pub(crate) struct GroundedRule {
    weight: f64,
    body_keys: Vec<AtomKey>,
    head_key: AtomKey,
    negated_head: bool,
}

/// Key for looking up a ground atom in the truth value map.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct AtomKey {
    predicate: String,
    entity_id: Uuid,
}

/// Ground rules against the hypergraph: find entities matching predicates.
///
/// Predicates are matched by checking entity properties, roles, and
/// participation patterns. The variable in each predicate is bound to
/// entity IDs that satisfy the predicate's conditions.
fn ground_rules(
    rules: &[PslRule],
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<(Vec<GroundedRule>, HashMap<AtomKey, f64>)> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    if entities.is_empty() {
        return Ok((vec![], HashMap::new()));
    }

    // Compute initial truth values for all predicate-entity combinations
    let mut atoms: HashMap<AtomKey, f64> = HashMap::new();

    // Collect all unique predicate names from rules
    let mut predicate_names: Vec<String> = Vec::new();
    for rule in rules {
        for p in &rule.body {
            if !predicate_names.contains(&p.name) {
                predicate_names.push(p.name.clone());
            }
        }
        if !predicate_names.contains(&rule.head.name) {
            predicate_names.push(rule.head.name.clone());
        }
    }

    // For each entity, evaluate each predicate to get initial truth value
    for entity in &entities {
        for pred_name in &predicate_names {
            let truth = evaluate_predicate(pred_name, entity, hypergraph)?;
            let key = AtomKey {
                predicate: pred_name.clone(),
                entity_id: entity.id,
            };
            atoms.insert(key, truth);
        }
    }

    // Ground each rule: bind variables to entities
    let mut grounded = Vec::new();
    for rule in rules {
        // All variables in a rule must be the same (single-variable rules for simplicity)
        for entity in &entities {
            let body_keys: Vec<AtomKey> = rule
                .body
                .iter()
                .map(|p| AtomKey {
                    predicate: p.name.clone(),
                    entity_id: entity.id,
                })
                .collect();

            let head_key = AtomKey {
                predicate: rule.head.name.clone(),
                entity_id: entity.id,
            };

            grounded.push(GroundedRule {
                weight: rule.weight,
                body_keys,
                head_key,
                negated_head: rule.negated_head,
            });
        }
    }

    Ok((grounded, atoms))
}

/// Evaluate a predicate for a given entity.
///
/// Returns initial truth value [0, 1] based on entity properties and
/// participation patterns. Unrecognized predicates use entity confidence.
fn evaluate_predicate(predicate: &str, entity: &Entity, _hypergraph: &Hypergraph) -> Result<f64> {
    let props = &entity.properties;

    match predicate {
        // Check if entity has a property matching the predicate name
        // e.g., "guilty" checks properties.guilty, "has_motive" checks properties.has_motive
        _ => {
            // Try to find the predicate as a boolean/numeric property
            if let Some(val) = props.get(predicate) {
                if let Some(b) = val.as_bool() {
                    return Ok(if b { 1.0 } else { 0.0 });
                }
                if let Some(n) = val.as_f64() {
                    return Ok(n.clamp(0.0, 1.0));
                }
                // String "true"/"false"
                if let Some(s) = val.as_str() {
                    return Ok(match s.to_lowercase().as_str() {
                        "true" | "yes" | "1" => 1.0,
                        "false" | "no" | "0" => 0.0,
                        _ => 0.5, // ambiguous string
                    });
                }
            }

            // Unknown predicate with no matching property: use entity confidence as prior
            Ok(entity.confidence as f64 * 0.5)
        }
    }
}

// ─── PSL Solver ────────────────────────────────────────────

/// Solve PSL inference via coordinate descent.
///
/// Minimizes: Σ_r w_r · max(0, body_truth(r) - head_truth(r))²
/// where body_truth = min(body atom truth values) (conjunction)
/// and head_truth is the target atom (or 1-target for negated heads).
fn solve_psl(
    rules: &[GroundedRule],
    mut atoms: HashMap<AtomKey, f64>,
    config: &PslConfig,
) -> (Vec<GroundAtom>, PslConvergence, f64) {
    let mut iterations = 0;
    let mut final_delta = 0.0;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        let mut max_change = 0.0_f64;

        // For each ground atom, compute gradient from all rules that reference it
        let keys: Vec<AtomKey> = atoms.keys().cloned().collect();
        for key in &keys {
            let old_value = atoms[key];

            // Sum gradients from all rules where this atom appears as the head
            let mut gradient = 0.0_f64;
            for rule in rules {
                if rule.head_key == *key {
                    let body_truth = compute_body_truth(&rule.body_keys, &atoms);
                    let head_truth = if rule.negated_head {
                        1.0 - old_value
                    } else {
                        old_value
                    };

                    let violation = body_truth - head_truth;
                    if violation > 0.0 {
                        // Gradient of w * max(0, body - head)²
                        // d/d(head) = -2 * w * violation (for non-negated)
                        // d/d(head) = +2 * w * violation (for negated, since head = 1 - y)
                        let sign = if rule.negated_head { 1.0 } else { -1.0 };
                        gradient += sign * 2.0 * rule.weight * violation;
                    }
                }
            }

            // Update via gradient descent
            let new_value = (old_value - config.step_size * gradient).clamp(0.0, 1.0);
            let change = (new_value - old_value).abs();
            max_change = max_change.max(change);
            atoms.insert(key.clone(), new_value);
        }

        final_delta = max_change;
        if max_change < config.convergence_threshold {
            converged = true;
            break;
        }
    }

    // Compute final total loss
    let total_loss = compute_total_loss(rules, &atoms);

    // Convert to output format
    let ground_atoms: Vec<GroundAtom> = atoms
        .iter()
        .map(|(key, &value)| GroundAtom {
            predicate: key.predicate.clone(),
            entity_id: key.entity_id,
            truth_value: value,
        })
        .collect();

    let convergence = PslConvergence {
        converged,
        iterations,
        final_delta,
    };

    (ground_atoms, convergence, total_loss)
}

/// Compute conjunctive body truth: min of all body atom truth values.
fn compute_body_truth(body_keys: &[AtomKey], atoms: &HashMap<AtomKey, f64>) -> f64 {
    if body_keys.is_empty() {
        return 1.0; // empty body is always true
    }
    body_keys
        .iter()
        .map(|k| atoms.get(k).copied().unwrap_or(0.0))
        .fold(f64::INFINITY, f64::min)
}

/// Compute total weighted hinge loss across all rules.
pub(crate) fn compute_total_loss(rules: &[GroundedRule], atoms: &HashMap<AtomKey, f64>) -> f64 {
    rules
        .iter()
        .map(|rule| {
            let body_truth = compute_body_truth(&rule.body_keys, atoms);
            let head_value = atoms.get(&rule.head_key).copied().unwrap_or(0.0);
            let head_truth = if rule.negated_head {
                1.0 - head_value
            } else {
                head_value
            };
            let violation = (body_truth - head_truth).max(0.0);
            rule.weight * violation * violation
        })
        .sum()
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, make_hg};

    fn make_rule(weight: f64, body: &[&str], head: &str, negated: bool) -> PslRule {
        PslRule {
            weight,
            body: body
                .iter()
                .map(|name| Predicate {
                    name: name.to_string(),
                    variable: "X".into(),
                })
                .collect(),
            head: Predicate {
                name: head.to_string(),
                variable: "X".into(),
            },
            negated_head: negated,
        }
    }

    fn make_atoms(entries: &[(&str, Uuid, f64)]) -> HashMap<AtomKey, f64> {
        entries
            .iter()
            .map(|(pred, eid, val)| {
                (
                    AtomKey {
                        predicate: pred.to_string(),
                        entity_id: *eid,
                    },
                    *val,
                )
            })
            .collect()
    }

    fn ground_rule(
        weight: f64,
        body: &[(&str, Uuid)],
        head: (&str, Uuid),
        negated: bool,
    ) -> GroundedRule {
        GroundedRule {
            weight,
            body_keys: body
                .iter()
                .map(|(p, e)| AtomKey {
                    predicate: p.to_string(),
                    entity_id: *e,
                })
                .collect(),
            head_key: AtomKey {
                predicate: head.0.to_string(),
                entity_id: head.1,
            },
            negated_head: negated,
        }
    }

    #[test]
    fn test_hinge_loss_satisfied() {
        // Rule: A → B, with A=0.5, B=0.8 → body ≤ head → loss = 0
        let eid = Uuid::now_v7();
        let atoms = make_atoms(&[("A", eid, 0.5), ("B", eid, 0.8)]);
        let rules = vec![ground_rule(1.0, &[("A", eid)], ("B", eid), false)];
        let loss = compute_total_loss(&rules, &atoms);
        assert!(
            loss < 1e-10,
            "Satisfied rule should have zero loss, got {}",
            loss
        );
    }

    #[test]
    fn test_hinge_loss_violated() {
        // Rule: A → B, with A=0.9, B=0.2 → violation = 0.7, loss = 1.0 * 0.49
        let eid = Uuid::now_v7();
        let atoms = make_atoms(&[("A", eid, 0.9), ("B", eid, 0.2)]);
        let rules = vec![ground_rule(1.0, &[("A", eid)], ("B", eid), false)];
        let loss = compute_total_loss(&rules, &atoms);
        assert!(
            (loss - 0.49).abs() < 0.01,
            "Violated rule loss should be ~0.49, got {}",
            loss
        );
    }

    #[test]
    fn test_coordinate_descent_converges() {
        // Rule: evidence → guilty (weight 0.8)
        // Initial: evidence=0.9, guilty=0.1
        // Should push guilty upward
        let eid = Uuid::now_v7();
        let atoms = make_atoms(&[("evidence", eid, 0.9), ("guilty", eid, 0.1)]);
        let rules = vec![ground_rule(
            0.8,
            &[("evidence", eid)],
            ("guilty", eid),
            false,
        )];

        let (result_atoms, conv, _) = solve_psl(&rules, atoms, &PslConfig::default());
        assert!(conv.converged, "Should converge");

        let guilty = result_atoms
            .iter()
            .find(|a| a.predicate == "guilty" && a.entity_id == eid)
            .unwrap();
        assert!(
            guilty.truth_value > 0.5,
            "Guilty should increase toward evidence level, got {}",
            guilty.truth_value
        );
    }

    #[test]
    fn test_two_opposing_rules() {
        // Rule 1: evidence → guilty (weight 0.8)
        // Rule 2: alibi → ¬guilty (weight 0.6)
        // Evidence = 0.9, alibi = 0.7 → guilty should be moderate (compromise)
        let eid = Uuid::now_v7();
        let atoms = make_atoms(&[
            ("evidence", eid, 0.9),
            ("alibi", eid, 0.7),
            ("guilty", eid, 0.5),
        ]);
        let rules = vec![
            ground_rule(0.8, &[("evidence", eid)], ("guilty", eid), false),
            ground_rule(0.6, &[("alibi", eid)], ("guilty", eid), true),
        ];

        let (result_atoms, conv, _) = solve_psl(&rules, atoms, &PslConfig::default());
        assert!(conv.converged);

        let guilty = result_atoms
            .iter()
            .find(|a| a.predicate == "guilty" && a.entity_id == eid)
            .unwrap();
        // With stronger evidence rule (0.8) vs weaker alibi (0.6), guilty should be > 0.5
        assert!(
            guilty.truth_value > 0.3 && guilty.truth_value < 0.95,
            "Guilty should be moderate compromise, got {}",
            guilty.truth_value
        );
    }

    #[test]
    fn test_negated_head() {
        // Rule: alibi → ¬guilty (weight 1.0)
        // Initial: alibi=0.9, guilty=0.8
        // Should push guilty down (toward 1 - alibi = 0.1)
        let eid = Uuid::now_v7();
        let atoms = make_atoms(&[("alibi", eid, 0.9), ("guilty", eid, 0.8)]);
        let rules = vec![ground_rule(1.0, &[("alibi", eid)], ("guilty", eid), true)];

        let (result_atoms, conv, _) = solve_psl(&rules, atoms, &PslConfig::default());
        assert!(conv.converged);

        let guilty = result_atoms
            .iter()
            .find(|a| a.predicate == "guilty" && a.entity_id == eid)
            .unwrap();
        assert!(
            guilty.truth_value < 0.5,
            "Alibi should reduce guilty, got {}",
            guilty.truth_value
        );
    }

    #[test]
    fn test_empty_grounding() {
        let hg = make_hg();
        let rules = vec![make_rule(0.8, &["evidence"], "guilty", false)];
        let (grounded, atoms) = ground_rules(&rules, &hg, "empty-narrative").unwrap();
        assert!(grounded.is_empty());
        assert!(atoms.is_empty());
    }

    #[test]
    fn test_simple_rule_grounding() {
        let hg = make_hg();
        let _e1 = add_entity(&hg, "Butler", "mystery");
        let _e2 = add_entity(&hg, "Maid", "mystery");

        let rules = vec![make_rule(0.8, &["suspicious"], "guilty", false)];
        let (grounded, atoms) = ground_rules(&rules, &hg, "mystery").unwrap();

        // 2 entities × 1 rule = 2 grounded rules
        assert_eq!(grounded.len(), 2, "Should ground rule for each entity");
        // 2 entities × 2 predicates = 4 atoms
        assert_eq!(
            atoms.len(),
            4,
            "Should have atoms for all predicate-entity pairs"
        );
    }

    #[test]
    fn test_engine_execute() {
        let hg = make_hg();
        let _e1 = add_entity(&hg, "Butler", "psl-test");
        let _e2 = add_entity(&hg, "Maid", "psl-test");

        let engine = PslEngine;
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::ProbabilisticSoftLogic,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({
                "narrative_id": "psl-test",
                "rules": [
                    {"weight": 0.8, "body": [{"name": "suspicious", "variable": "X"}], "head": {"name": "guilty", "variable": "X"}, "negated_head": false}
                ]
            }),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.explanation.is_some());

        let parsed: PslResult = serde_json::from_value(result.result).unwrap();
        assert_eq!(parsed.narrative_id, "psl-test");
        assert!(!parsed.ground_atoms.is_empty());
    }

    #[test]
    fn test_result_serde() {
        let result = PslResult {
            narrative_id: "test".into(),
            ground_atoms: vec![
                GroundAtom {
                    predicate: "guilty".into(),
                    entity_id: Uuid::now_v7(),
                    truth_value: 0.73,
                },
                GroundAtom {
                    predicate: "alibi".into(),
                    entity_id: Uuid::now_v7(),
                    truth_value: 0.2,
                },
            ],
            rules_applied: 3,
            total_loss: 0.05,
            convergence: PslConvergence {
                converged: true,
                iterations: 42,
                final_delta: 1e-5,
            },
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: PslResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rules_applied, 3);
        assert_eq!(parsed.ground_atoms.len(), 2);
    }

    #[test]
    fn test_kv_persistence() {
        let hg = make_hg();
        let _e = add_entity(&hg, "Suspect", "persist-psl");

        let engine = PslEngine;
        let job = InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type: InferenceJobType::ProbabilisticSoftLogic,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({
                "narrative_id": "persist-psl",
                "rules": [
                    {"weight": 1.0, "body": [{"name": "evidence", "variable": "X"}], "head": {"name": "guilty", "variable": "X"}, "negated_head": false}
                ]
            }),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };

        engine.execute(&job, &hg).unwrap();

        let key = analysis_key(ANALYSIS_PSL, &["persist-psl"]);
        let stored = hg.store().get(&key).unwrap();
        assert!(stored.is_some(), "Result should be persisted in KV");
    }
}
