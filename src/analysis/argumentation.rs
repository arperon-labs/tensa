//! Dung's abstract argumentation frameworks.
//!
//! Computes grounded, preferred, and stable extensions to determine
//! which claims survive when claims attack each other.
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! This module stores a single per-argument `confidence` float and does
//! not arithmetically compose arguments; the Phase 0 audit flagged no
//! confidence-combination call sites here. Future graded-attack /
//! graded-defence extensions should thread `fuzzy::tnorm::TNormKind`
//! with a default-Gödel hook so existing numerics stay bit-identical.
//!
//! Cites: [klement2000].

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::argumentation_gradual::{
    run_gradual_argumentation, GradualResult, GradualSemanticsKind,
};
use crate::analysis::{analysis_key, extract_narrative_id};
use crate::error::Result;
use crate::fuzzy::tnorm::TNormKind;
use crate::hypergraph::{keys, Hypergraph};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

// ─── Data Structures ────────────────────────────────────────

/// An argument in the framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    pub id: Uuid,
    pub label: String,
    pub source_id: Option<Uuid>,
    pub confidence: f32,
}

/// An abstract argumentation framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentationFramework {
    pub arguments: Vec<Argument>,
    pub attacks: Vec<(usize, usize)>, // (attacker_idx, target_idx)
}

/// Label for an argument under a given semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArgumentLabel {
    In,
    Out,
    Undec,
}

/// Result of argumentation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentationResult {
    pub narrative_id: String,
    pub framework: ArgumentationFramework,
    pub grounded: Vec<(Uuid, ArgumentLabel)>,
    pub preferred_extensions: Vec<Vec<Uuid>>,
    pub stable_extensions: Vec<Vec<Uuid>>,
    /// Optional gradual-semantics evaluation; serde-skipped when `None`
    /// so existing KV blobs round-trip unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradual: Option<GradualResult>,
}

// ─── Framework Construction ─────────────────────────────────

/// Build an argumentation framework from contention links in a narrative.
pub fn build_framework(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<ArgumentationFramework> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let situation_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();

    // Collect all contentions.
    let mut all_contentions = Vec::new();
    let mut argument_ids: Vec<Uuid> = Vec::new();
    let mut id_set: HashSet<Uuid> = HashSet::new();

    for sid in &situation_ids {
        let contentions = hypergraph.get_contentions_for_situation(sid)?;
        for c in contentions {
            if !c.resolved {
                // Both situations become arguments.
                for &id in &[c.situation_a, c.situation_b] {
                    if id_set.insert(id) {
                        argument_ids.push(id);
                    }
                }
                all_contentions.push(c);
            }
        }
    }

    // Build arguments.
    let mut id_to_idx: HashMap<Uuid, usize> = HashMap::new();
    let mut arguments = Vec::new();
    for (i, &id) in argument_ids.iter().enumerate() {
        id_to_idx.insert(id, i);
        let (label, confidence) = match hypergraph.get_situation(&id) {
            Ok(sit) => (
                sit.raw_content
                    .first()
                    .map(|c| c.content.clone())
                    .unwrap_or_else(|| format!("situation-{}", id)),
                sit.confidence,
            ),
            Err(_) => (format!("situation-{}", id), 0.5),
        };
        arguments.push(Argument {
            id,
            label,
            source_id: None,
            confidence,
        });
    }

    // Build attacks. CONTRADICTS/CHALLENGES → bidirectional attacks.
    let mut attacks = Vec::new();
    for c in &all_contentions {
        if let (Some(&a), Some(&b)) = (id_to_idx.get(&c.situation_a), id_to_idx.get(&c.situation_b))
        {
            attacks.push((a, b));
            attacks.push((b, a));
        }
    }

    // Deduplicate attacks.
    attacks.sort();
    attacks.dedup();

    Ok(ArgumentationFramework { arguments, attacks })
}

/// Build an argumentation framework from explicit arguments and attacks.
pub fn from_explicit(
    arguments: Vec<Argument>,
    attacks: Vec<(usize, usize)>,
) -> ArgumentationFramework {
    ArgumentationFramework { arguments, attacks }
}

// ─── Grounded Extension ─────────────────────────────────────

/// Compute the grounded extension (unique, most conservative).
/// Iteratively labels unattacked arguments as IN, arguments attacked
/// only by OUT arguments as IN, and everything else as UNDEC.
pub fn grounded_extension(framework: &ArgumentationFramework) -> Vec<ArgumentLabel> {
    let n = framework.arguments.len();
    if n == 0 {
        return vec![];
    }

    let mut labels = vec![ArgumentLabel::Undec; n];

    // Build attack sets.
    let mut attackers: Vec<Vec<usize>> = vec![vec![]; n];
    for &(a, t) in &framework.attacks {
        attackers[t].push(a);
    }

    // Check for self-attacks first.
    for &(a, t) in &framework.attacks {
        if a == t {
            labels[a] = ArgumentLabel::Out;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;

        for i in 0..n {
            if labels[i] != ArgumentLabel::Undec {
                continue;
            }

            // Check if all attackers are OUT.
            let all_attackers_out = attackers[i]
                .iter()
                .all(|&a| labels[a] == ArgumentLabel::Out);

            if all_attackers_out {
                labels[i] = ArgumentLabel::In;
                changed = true;

                // Mark everything attacked by i as OUT.
                for &(a, t) in &framework.attacks {
                    if a == i && labels[t] == ArgumentLabel::Undec {
                        labels[t] = ArgumentLabel::Out;
                        changed = true;
                    }
                }
            }
        }
    }

    labels
}

// ─── Preferred Extensions ───────────────────────────────────

/// Compute all preferred extensions (maximal admissible sets).
/// Default search space limit for preferred extension enumeration.
pub const DEFAULT_PREFERRED_EXTENSION_CAP: usize = 10_000;

pub fn preferred_extensions(framework: &ArgumentationFramework) -> Vec<BTreeSet<usize>> {
    preferred_extensions_with_cap(framework, DEFAULT_PREFERRED_EXTENSION_CAP)
}

/// Compute preferred extensions with a configurable search space cap.
pub fn preferred_extensions_with_cap(
    framework: &ArgumentationFramework,
    search_cap: usize,
) -> Vec<BTreeSet<usize>> {
    let n = framework.arguments.len();
    if n == 0 {
        return vec![BTreeSet::new()];
    }

    // Build attack structures.
    let mut attacks_from: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut attacked_by: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(a, t) in &framework.attacks {
        attacks_from[a].insert(t);
        attacked_by[t].insert(a);
    }

    // Find all admissible sets via BFS over subsets (practical for small frameworks).
    let mut admissible: Vec<BTreeSet<usize>> = vec![BTreeSet::new()]; // empty set is always admissible.

    // Generate candidate sets by trying to extend admissible sets.
    let mut queue: VecDeque<BTreeSet<usize>> = VecDeque::new();
    queue.push_back(BTreeSet::new());
    let mut seen: HashSet<BTreeSet<usize>> = HashSet::new();
    seen.insert(BTreeSet::new());

    while let Some(current) = queue.pop_front() {
        for i in 0..n {
            if current.contains(&i) {
                continue;
            }

            let mut candidate = current.clone();
            candidate.insert(i);

            if seen.contains(&candidate) {
                continue;
            }
            seen.insert(candidate.clone());

            if is_admissible(&candidate, &attacks_from, &attacked_by) {
                admissible.push(candidate.clone());
                if candidate.len() < n {
                    queue.push_back(candidate);
                }
            }
        }

        // Limit search space for large frameworks.
        if seen.len() > search_cap {
            break;
        }
    }

    // Filter to maximal admissible sets.
    let mut preferred: Vec<BTreeSet<usize>> = Vec::new();
    for set in &admissible {
        let is_maximal = !admissible
            .iter()
            .any(|other| other != set && set.is_subset(other));
        if is_maximal {
            preferred.push(set.clone());
        }
    }

    if preferred.is_empty() {
        preferred.push(BTreeSet::new());
    }

    preferred
}

/// Check if a set is admissible: conflict-free and defends all members.
fn is_admissible(
    set: &BTreeSet<usize>,
    attacks_from: &[HashSet<usize>],
    attacked_by: &[HashSet<usize>],
) -> bool {
    // Conflict-free: no internal attacks.
    for &a in set {
        for &b in set {
            if attacks_from[a].contains(&b) {
                return false;
            }
        }
    }

    // Defends all members: every attacker of a member is counter-attacked.
    for &member in set {
        for &attacker in &attacked_by[member] {
            if set.contains(&attacker) {
                continue; // can't happen if conflict-free, but safe
            }
            // Some member of set must attack the attacker.
            let defended = set
                .iter()
                .any(|&defender| attacks_from[defender].contains(&attacker));
            if !defended {
                return false;
            }
        }
    }

    true
}

// ─── Stable Extensions ─────────────────────────────────────

/// Compute stable extensions: conflict-free sets that attack every non-member.
/// Accepts pre-computed preferred extensions to avoid redundant computation.
pub fn stable_extensions_from(
    framework: &ArgumentationFramework,
    preferred: &[BTreeSet<usize>],
) -> Vec<BTreeSet<usize>> {
    let n = framework.arguments.len();
    if n == 0 {
        return vec![BTreeSet::new()];
    }

    let mut attacks_from: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for &(a, t) in &framework.attacks {
        attacks_from[a].insert(t);
    }

    let mut stable = Vec::new();

    for ext in preferred {
        let mut attacks_all = true;
        for i in 0..n {
            if ext.contains(&i) {
                continue;
            }
            let attacked = ext.iter().any(|&member| attacks_from[member].contains(&i));
            if !attacked {
                attacks_all = false;
                break;
            }
        }
        if attacks_all {
            stable.push(ext.clone());
        }
    }

    stable
}

/// Convenience wrapper that computes preferred extensions first.
pub fn stable_extensions(framework: &ArgumentationFramework) -> Vec<BTreeSet<usize>> {
    let preferred = preferred_extensions(framework);
    stable_extensions_from(framework, &preferred)
}

// ─── Main Entry Point ───────────────────────────────────────

/// Run argumentation analysis on a narrative — legacy crisp entry point.
/// Equivalent to [`run_argumentation_with_gradual`] with both optionals `None`.
pub fn run_argumentation(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<ArgumentationResult> {
    run_argumentation_with_gradual(hypergraph, narrative_id, None, None)
}

/// Run argumentation analysis on a narrative, optionally augmenting the
/// crisp grounded/preferred/stable result with a gradual-semantics
/// evaluation under the supplied t-norm. `gradual_semantics = None`
/// skips the gradual pass; `tnorm = None` uses Gödel (the canonical
/// paper formula). Both `None` is bit-identical to legacy behaviour.
pub fn run_argumentation_with_gradual(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    gradual_semantics: Option<GradualSemanticsKind>,
    tnorm: Option<TNormKind>,
) -> Result<ArgumentationResult> {
    let framework = build_framework(hypergraph, narrative_id)?;
    run_on_framework_with_gradual(hypergraph, narrative_id, &framework, gradual_semantics, tnorm)
}

/// Legacy crisp entry point on a pre-built framework.
pub fn run_on_framework(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    framework: &ArgumentationFramework,
) -> Result<ArgumentationResult> {
    run_on_framework_with_gradual(hypergraph, narrative_id, framework, None, None)
}

/// Run argumentation analysis on a pre-built framework with optional
/// gradual-semantics extension. See [`run_argumentation_with_gradual`].
pub fn run_on_framework_with_gradual(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    framework: &ArgumentationFramework,
    gradual_semantics: Option<GradualSemanticsKind>,
    tnorm: Option<TNormKind>,
) -> Result<ArgumentationResult> {
    let grounded_labels = grounded_extension(framework);
    let preferred = preferred_extensions(framework);
    let stable = stable_extensions_from(framework, &preferred);

    let grounded: Vec<(Uuid, ArgumentLabel)> = framework
        .arguments
        .iter()
        .enumerate()
        .map(|(i, arg)| (arg.id, grounded_labels[i]))
        .collect();

    let to_uuids = |sets: &[BTreeSet<usize>]| -> Vec<Vec<Uuid>> {
        sets.iter()
            .map(|ext| ext.iter().map(|&i| framework.arguments[i].id).collect())
            .collect()
    };
    let preferred_ext = to_uuids(&preferred);
    let stable_ext = to_uuids(&stable);

    let gradual = gradual_semantics
        .map(|kind| run_gradual_argumentation(framework, &kind, tnorm))
        .transpose()?;

    let result = ArgumentationResult {
        narrative_id: narrative_id.to_string(),
        framework: framework.clone(),
        grounded,
        preferred_extensions: preferred_ext,
        stable_extensions: stable_ext,
        gradual,
    };

    // Store per-argument results.
    for (i, arg) in framework.arguments.iter().enumerate() {
        let key = analysis_key(
            keys::ANALYSIS_ARGUMENTATION,
            &[narrative_id, &arg.id.to_string()],
        );
        let data = serde_json::json!({
            "argument_id": arg.id,
            "label": arg.label,
            "grounded_status": grounded_labels[i],
            "preferred_memberships": preferred.iter()
                .enumerate()
                .filter(|(_, ext)| ext.contains(&i))
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>(),
        });
        let bytes = serde_json::to_vec(&data)?;
        hypergraph.store().put(&key, &bytes)?;
    }

    Ok(result)
}

// ─── InferenceEngine ────────────────────────────────────────

pub struct ArgumentationEngine;

impl InferenceEngine for ArgumentationEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ArgumentationAnalysis
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(2000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        // Both fields absent = legacy behaviour bit-identical.
        let param = |k: &str| -> Option<serde_json::Value> { job.parameters.get(k).cloned() };
        let gradual_semantics: Option<GradualSemanticsKind> =
            param("gradual_semantics").and_then(|v| serde_json::from_value(v).ok());
        let tnorm: Option<TNormKind> =
            param("tnorm").and_then(|v| serde_json::from_value(v).ok());
        let result =
            run_argumentation_with_gradual(hypergraph, narrative_id, gradual_semantics, tnorm)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::ArgumentationAnalysis,
            target_id: job.target_id,
            result: serde_json::to_value(&result)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Argumentation: {} arguments, {} preferred extensions",
                result.framework.arguments.len(),
                result.preferred_extensions.len()
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
#[path = "argumentation_tests.rs"]
mod tests;
