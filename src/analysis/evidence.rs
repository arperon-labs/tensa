//! Dempster-Shafer evidence theory for uncertainty quantification.
//!
//! Provides mass functions, Dempster's rule of combination, and
//! belief/plausibility interval computation. Additive to the existing
//! confidence model — does not replace single-float confidence.
//!
//! ## Fuzzy-logic wiring (Phase 1)
//!
//! Dempster's and Yager's combination rules both multiply per-focal-element
//! masses (`ma * mb`) which is the **Goguen t-norm** (product). The Phase 1
//! fuzzy wiring exposes this choice explicitly via
//! [`combine_with_tnorm`] and [`combine_yager_with_tnorm`] while the
//! existing `combine` / `combine_yager` helpers stay bit-identical to the
//! pre-sprint numerics by defaulting to `TNormKind::Goguen`. Additive
//! aggregation on the Dirichlet path (`alpha_k = bel_k * num_sources + 1`)
//! is **NOT a t-norm** — the Phase 2 aggregator selector owns it.
//!
//! Cites: [klement2000].

use std::collections::{BTreeSet, HashMap};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::{analysis_key, extract_narrative_id};
use crate::error::{Result, TensaError};
use crate::fuzzy::tnorm::{combine_tnorm, TNormKind};
use crate::hypergraph::{keys, Hypergraph};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

// ─── Data Structures ────────────────────────────────────────

/// A mass assignment over a power set (frame of discernment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassFunction {
    /// Universe of hypotheses.
    pub frame: Vec<String>,
    /// Subset → mass. Key is sorted set of indices into `frame`,
    /// serialized as a sorted Vec for JSON compatibility.
    #[serde(
        serialize_with = "serialize_masses",
        deserialize_with = "deserialize_masses"
    )]
    pub masses: HashMap<BTreeSet<usize>, f64>,
}

fn serialize_masses<S>(
    masses: &HashMap<BTreeSet<usize>, f64>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(masses.len()))?;
    for (k, v) in masses {
        let entry: (Vec<usize>, f64) = (k.iter().copied().collect(), *v);
        seq.serialize_element(&entry)?;
    }
    seq.end()
}

fn deserialize_masses<'de, D>(
    deserializer: D,
) -> std::result::Result<HashMap<BTreeSet<usize>, f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let entries: Vec<(Vec<usize>, f64)> = serde::Deserialize::deserialize(deserializer)?;
    Ok(entries
        .into_iter()
        .map(|(k, v)| (k.into_iter().collect(), v))
        .collect())
}

/// Belief and plausibility for a hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefPlausibility {
    pub hypothesis: BTreeSet<usize>,
    pub belief: f64,
    pub plausibility: f64,
    pub uncertainty: f64,
}

/// Result of evidence combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceResult {
    pub frame: Vec<String>,
    pub combined_mass: MassFunction,
    pub belief_plausibility: Vec<BeliefPlausibility>,
    pub conflict: f64,
}

impl MassFunction {
    /// Create a new mass function with validation.
    pub fn new(frame: Vec<String>, masses: HashMap<BTreeSet<usize>, f64>) -> Result<Self> {
        // Validate no empty set mass.
        if masses.contains_key(&BTreeSet::new()) {
            return Err(TensaError::InferenceError(
                "Mass of empty set must be 0".into(),
            ));
        }

        // Validate all indices are within frame.
        for subset in masses.keys() {
            for &idx in subset {
                if idx >= frame.len() {
                    return Err(TensaError::InferenceError(format!(
                        "Index {} out of frame bounds (size {})",
                        idx,
                        frame.len()
                    )));
                }
            }
        }

        // Validate masses sum to ~1.0.
        let total: f64 = masses.values().sum();
        if (total - 1.0).abs() > 0.01 {
            return Err(TensaError::InferenceError(format!(
                "Masses must sum to 1.0, got {}",
                total
            )));
        }

        Ok(Self { frame, masses })
    }

    /// Create a vacuous mass function (all mass on the full frame).
    pub fn vacuous(frame: Vec<String>) -> Self {
        let n = frame.len();
        let full_set: BTreeSet<usize> = (0..n).collect();
        let mut masses = HashMap::new();
        masses.insert(full_set, 1.0);
        Self { frame, masses }
    }

    /// Create a categorical mass function (all mass on a single hypothesis).
    pub fn categorical(frame: Vec<String>, hypothesis_idx: usize) -> Result<Self> {
        if hypothesis_idx >= frame.len() {
            return Err(TensaError::InferenceError(
                "Hypothesis index out of bounds".into(),
            ));
        }
        let mut masses = HashMap::new();
        masses.insert(BTreeSet::from([hypothesis_idx]), 1.0);
        Ok(Self { frame, masses })
    }
}

// ─── Dempster's Rule of Combination ─────────────────────────

/// Combine two mass functions using Dempster's rule.
///
/// Returns the combined mass function and the conflict measure K.
/// Delegates to [`combine_with_tnorm`] using `TNormKind::Goguen` (product),
/// which matches the pre-sprint Dempster numerics bit-for-bit.
pub fn combine(m1: &MassFunction, m2: &MassFunction) -> Result<(MassFunction, f64)> {
    combine_with_tnorm(m1, m2, TNormKind::Goguen)
}

/// Combine two mass functions using Dempster's rule under an explicit
/// t-norm choice. `TNormKind::Goguen` (product) reproduces the classical
/// Dempster–Shafer combination; other kinds produce variant mass folds
/// that are useful for sensitivity studies.
pub fn combine_with_tnorm(
    m1: &MassFunction,
    m2: &MassFunction,
    tnorm: TNormKind,
) -> Result<(MassFunction, f64)> {
    if m1.frame != m2.frame {
        return Err(TensaError::InferenceError(
            "Cannot combine mass functions with different frames".into(),
        ));
    }

    let frame = m1.frame.clone();
    let mut combined: HashMap<BTreeSet<usize>, f64> = HashMap::new();
    let mut conflict = 0.0;

    for (a, &ma) in &m1.masses {
        for (b, &mb) in &m2.masses {
            let intersection: BTreeSet<usize> = a.intersection(b).copied().collect();
            let product = combine_tnorm(tnorm, ma, mb);

            if intersection.is_empty() {
                conflict += product;
            } else {
                *combined.entry(intersection).or_insert(0.0) += product;
            }
        }
    }

    // Normalize by 1/(1-K).
    if (1.0 - conflict).abs() < 1e-10 {
        return Err(TensaError::InferenceError(
            "Total conflict (K=1): sources completely contradict".into(),
        ));
    }

    let normalizer = 1.0 / (1.0 - conflict);
    for mass in combined.values_mut() {
        *mass *= normalizer;
    }

    Ok((
        MassFunction {
            frame,
            masses: combined,
        },
        conflict,
    ))
}

/// Combination rule selection for evidence combination.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
pub enum CombinationRule {
    /// Standard Dempster's rule: normalize conflict away.
    #[default]
    Dempster,
    /// Yager's rule: reallocate conflict mass to the universal set Θ.
    /// More conservative than Dempster — high conflict increases ignorance
    /// rather than being normalized away.
    Yager,
}

/// Combine two mass functions using Yager's rule.
///
/// Instead of normalizing conflict mass K away (Dempster), Yager assigns
/// the conflict mass to the full frame (universal set Θ), representing
/// increased ignorance when sources disagree.
pub fn combine_yager(m1: &MassFunction, m2: &MassFunction) -> Result<(MassFunction, f64)> {
    combine_yager_with_tnorm(m1, m2, TNormKind::Goguen)
}

/// Yager's rule under an explicit t-norm choice (default `Goguen` preserves
/// pre-sprint numerics).
pub fn combine_yager_with_tnorm(
    m1: &MassFunction,
    m2: &MassFunction,
    tnorm: TNormKind,
) -> Result<(MassFunction, f64)> {
    if m1.frame != m2.frame {
        return Err(TensaError::InferenceError(
            "Cannot combine mass functions with different frames".into(),
        ));
    }

    let frame = m1.frame.clone();
    let n = frame.len();
    let full_set: BTreeSet<usize> = (0..n).collect();
    let mut combined: HashMap<BTreeSet<usize>, f64> = HashMap::new();
    let mut conflict = 0.0;

    for (a, &ma) in &m1.masses {
        for (b, &mb) in &m2.masses {
            let intersection: BTreeSet<usize> = a.intersection(b).copied().collect();
            let product = combine_tnorm(tnorm, ma, mb);

            if intersection.is_empty() {
                conflict += product;
            } else {
                *combined.entry(intersection).or_insert(0.0) += product;
            }
        }
    }

    // Yager: assign conflict mass to the universal set (Θ)
    *combined.entry(full_set).or_insert(0.0) += conflict;

    Ok((
        MassFunction {
            frame,
            masses: combined,
        },
        conflict,
    ))
}

/// Combine two mass functions using the selected combination rule.
pub fn combine_with_rule(
    m1: &MassFunction,
    m2: &MassFunction,
    rule: CombinationRule,
) -> Result<(MassFunction, f64)> {
    match rule {
        CombinationRule::Dempster => combine(m1, m2),
        CombinationRule::Yager => combine_yager(m1, m2),
    }
}

/// Combine multiple mass functions (Dempster's rule is associative).
pub fn combine_multiple(mass_functions: &[MassFunction]) -> Result<(MassFunction, f64)> {
    if mass_functions.is_empty() {
        return Err(TensaError::InferenceError(
            "Need at least one mass function".into(),
        ));
    }
    if mass_functions.len() == 1 {
        return Ok((mass_functions[0].clone(), 0.0));
    }

    let mut result = mass_functions[0].clone();
    let mut total_conflict = 0.0;

    for mf in &mass_functions[1..] {
        let (combined, k) = combine(&result, mf)?;
        result = combined;
        total_conflict = 1.0 - (1.0 - total_conflict) * (1.0 - k);
    }

    Ok((result, total_conflict))
}

// ─── Dirichlet / Evidential DL Confidence ──────────────────

/// Dirichlet-based uncertainty decomposition (Evidential Deep Learning).
///
/// Separates "I don't know" (epistemic, reducible with more data) from
/// "it's genuinely ambiguous" (aleatoric, irreducible). Computed from a
/// Dempster-Shafer mass function via `alpha_k = evidence_for_k + 1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirichletConfidence {
    /// Dirichlet concentration parameters (one per hypothesis).
    pub alpha: Vec<f64>,
    /// Sum of evidence across all hypotheses.
    pub total_evidence: f64,
    /// Epistemic uncertainty: K / S. High when total evidence is low.
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty: normalized entropy of the expected distribution.
    /// High when evidence is spread evenly across hypotheses.
    pub aleatoric_uncertainty: f64,
}

/// Compute Dirichlet confidence parameters from a Dempster-Shafer mass function.
///
/// Maps singleton beliefs to evidence via `evidence_k = Bel({k}) * num_sources`,
/// then computes `alpha_k = evidence_k + 1` (Dirichlet prior of 1 per class).
///
/// See also [`dirichlet_from_mass_with_aggregator`] for the Phase 2
/// aggregator-selectable variant that lets callers swap the
/// multiplicative `Bel({k}) * num_sources` accumulation for an
/// `AggregatorKind` fold over per-source mass contributions.
///
/// Cites: [yager1988owa] [grabisch1996choquet].
pub fn dirichlet_from_mass(mass: &MassFunction, num_sources: usize) -> DirichletConfidence {
    let k = mass.frame.len();
    if k == 0 {
        return DirichletConfidence {
            alpha: vec![],
            total_evidence: 0.0,
            epistemic_uncertainty: 1.0,
            aleatoric_uncertainty: 0.0,
        };
    }

    let scale = num_sources.max(1) as f64;
    let mut alpha = Vec::with_capacity(k);
    let mut total_evidence = 0.0;

    for i in 0..k {
        let bel_k = belief(mass, &BTreeSet::from([i]));
        let evidence_k = bel_k * scale;
        total_evidence += evidence_k;
        alpha.push(evidence_k + 1.0);
    }

    let s: f64 = alpha.iter().sum();

    // Epistemic uncertainty: K / S (ranges from K/(K+E) to 1.0 when E=0)
    let epistemic = k as f64 / s;

    // Aleatoric uncertainty: normalized Shannon entropy of expected distribution
    let aleatoric = if k > 1 {
        let mut entropy = 0.0;
        for &a in &alpha {
            let p = a / s;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy / (k as f64).ln()
    } else {
        0.0
    };

    DirichletConfidence {
        alpha,
        total_evidence,
        epistemic_uncertainty: epistemic,
        aleatoric_uncertainty: aleatoric,
    }
}

/// Phase 2 variant: aggregator-selectable Dirichlet construction.
///
/// The classical `dirichlet_from_mass` accumulates evidence via
/// `evidence_k = Bel({k}) * num_sources`. When `agg = AggregatorKind::Mean`
/// this function is bit-identical to that path (arithmetic mean × num_sources
/// reduces to Bel({k}) × num_sources when all mass is carried by a single
/// aggregate call). Any other aggregator (OWA, Choquet, TNormReduce, etc.)
/// rewires the fold — the belief contribution for hypothesis `k` is folded
/// through the aggregator before scaling by `num_sources`.
///
/// Cites: [yager1988owa] [grabisch1996choquet].
pub fn dirichlet_from_mass_with_aggregator(
    mass: &MassFunction,
    num_sources: usize,
    agg: &crate::fuzzy::aggregation::AggregatorKind,
) -> Result<DirichletConfidence> {
    let k = mass.frame.len();
    if k == 0 {
        return Ok(DirichletConfidence {
            alpha: vec![],
            total_evidence: 0.0,
            epistemic_uncertainty: 1.0,
            aleatoric_uncertainty: 0.0,
        });
    }
    let scale = num_sources.max(1) as f64;
    let aggregator = crate::fuzzy::aggregation::aggregator_for(agg.clone());

    let mut alpha = Vec::with_capacity(k);
    let mut total_evidence = 0.0;

    for i in 0..k {
        // Collect per-subset masses contributing to the singleton belief and
        // aggregate them under the caller-selected operator.
        let contribs: Vec<f64> = mass
            .masses
            .iter()
            .filter(|(subset, _)| subset.is_subset(&BTreeSet::from([i])))
            .map(|(_, &m)| m)
            .collect();
        let bel_k = if contribs.is_empty() {
            0.0
        } else {
            aggregator.aggregate(&contribs)?
        };
        let evidence_k = bel_k * scale;
        total_evidence += evidence_k;
        alpha.push(evidence_k + 1.0);
    }

    let s: f64 = alpha.iter().sum();
    let epistemic = k as f64 / s;
    let aleatoric = if k > 1 {
        let mut entropy = 0.0;
        for &a in &alpha {
            let p = a / s;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy / (k as f64).ln()
    } else {
        0.0
    };

    Ok(DirichletConfidence {
        alpha,
        total_evidence,
        epistemic_uncertainty: epistemic,
        aleatoric_uncertainty: aleatoric,
    })
}

// ─── Belief and Plausibility ────────────────────────────────

/// Compute belief for a hypothesis: Bel(A) = Σ m(B) for all B ⊆ A.
pub fn belief(mass: &MassFunction, hypothesis: &BTreeSet<usize>) -> f64 {
    mass.masses
        .iter()
        .filter(|(subset, _)| subset.is_subset(hypothesis))
        .map(|(_, &m)| m)
        .sum()
}

/// Compute plausibility for a hypothesis: Pl(A) = Σ m(B) for all B where B ∩ A ≠ ∅.
pub fn plausibility(mass: &MassFunction, hypothesis: &BTreeSet<usize>) -> f64 {
    mass.masses
        .iter()
        .filter(|(subset, _)| !subset.is_disjoint(hypothesis))
        .map(|(_, &m)| m)
        .sum()
}

/// Compute belief and plausibility for all singleton hypotheses.
pub fn compute_intervals(mass: &MassFunction) -> Vec<BeliefPlausibility> {
    let n = mass.frame.len();
    let mut results = Vec::new();

    for i in 0..n {
        let hyp = BTreeSet::from([i]);
        let bel = belief(mass, &hyp);
        let pl = plausibility(mass, &hyp);
        results.push(BeliefPlausibility {
            hypothesis: hyp,
            belief: bel,
            plausibility: pl,
            uncertainty: pl - bel,
        });
    }

    results
}

// ─── Main Entry Point ───────────────────────────────────────

/// Build a vacuous (no-information) evidence result for a given frame.
fn vacuous_result(frame: Vec<String>) -> EvidenceResult {
    let vacuous = MassFunction::vacuous(frame.clone());
    let intervals = compute_intervals(&vacuous);
    EvidenceResult {
        frame,
        combined_mass: vacuous,
        belief_plausibility: intervals,
        conflict: 0.0,
    }
}

/// Run evidence combination from source attributions for a hypothesis frame.
pub fn run_evidence(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    frame: Vec<String>,
    target_id: Uuid,
) -> Result<EvidenceResult> {
    let attributions = hypergraph.get_attributions_for_target(&target_id)?;

    if attributions.is_empty() {
        return Ok(vacuous_result(frame));
    }

    // Build mass functions from source trust scores.
    let mut mass_functions = Vec::new();
    for attr in &attributions {
        if let Ok(source) = hypergraph.get_source(&attr.source_id) {
            let trust = source.trust_score.clamp(0.0, 0.99) as f64;

            // Create a mass function: trust goes to a specific hypothesis
            // (based on excerpt/claim), remainder goes to full frame.
            // Without detailed claim parsing, distribute trust uniformly
            // across all singletons.
            let n = frame.len();
            if n == 0 {
                continue;
            }

            let full_set: BTreeSet<usize> = (0..n).collect();
            let mut masses = HashMap::new();

            // Check if this attribution has a claim matching a frame element
            let claimed_idx = attr
                .claim
                .as_ref()
                .and_then(|claim| frame.iter().position(|h| h == claim));

            if let Some(idx) = claimed_idx {
                // Concentrate mass on the claimed hypothesis
                masses.insert(BTreeSet::from([idx]), trust);
            } else {
                // No claim or claim doesn't match frame — distribute uniformly
                let per_hyp = trust / n as f64;
                for i in 0..n {
                    *masses.entry(BTreeSet::from([i])).or_insert(0.0) += per_hyp;
                }
            }

            *masses.entry(full_set).or_insert(0.0) += 1.0 - trust;

            mass_functions.push(MassFunction {
                frame: frame.clone(),
                masses,
            });
        }
    }

    if mass_functions.is_empty() {
        return Ok(vacuous_result(frame));
    }

    let (combined, conflict) = combine_multiple(&mass_functions)?;
    let intervals = compute_intervals(&combined);

    // Store result.
    let key = analysis_key(
        keys::ANALYSIS_EVIDENCE,
        &[narrative_id, &target_id.to_string()],
    );
    let result = EvidenceResult {
        frame: frame.clone(),
        combined_mass: combined,
        belief_plausibility: intervals,
        conflict,
    };
    let bytes = serde_json::to_vec(&result)?;
    hypergraph.store().put(&key, &bytes)?;

    Ok(result)
}

// ─── InferenceEngine ────────────────────────────────────────

pub struct EvidenceEngine;

impl InferenceEngine for EvidenceEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::EvidenceCombination
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(1000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;

        let frame: Vec<String> = job
            .parameters
            .get("frame")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .ok_or_else(|| TensaError::InferenceError("missing frame parameter".into()))?;

        let result = run_evidence(hypergraph, narrative_id, frame, job.target_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::EvidenceCombination,
            target_id: job.target_id,
            result: serde_json::to_value(&result)?,
            confidence: 1.0,
            explanation: Some(format!(
                "Evidence combination: conflict={:.3}",
                result.conflict
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
#[path = "evidence_tests.rs"]
mod tests;
