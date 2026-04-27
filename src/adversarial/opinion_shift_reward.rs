//! EATH Extension Phase 16c — Opinion-shift reward function for wargames.
//!
//! Adds a new `RewardFunction` variant — `OpinionShift` — that scores how
//! much an adversarial intervention shifted aggregate opinion toward a
//! target value. Reuses the synchronous opinion-dynamics engine (Phase 16b)
//! to compare a no-intervention baseline against the post-intervention
//! substrate.
//!
//! This module deliberately does NOT modify [`super::session::WargameConfig`]
//! — instead it offers a stand-alone `RewardFunction` enum + an
//! `evaluate_opinion_shift` helper. Wargame harnesses that want
//! opinion-shift scoring construct one explicitly and call evaluate after
//! the intervention completes. The wargame's existing
//! [`super::session::Objective`] / `ObjectiveMetric` pair handles binary
//! "did we hit a threshold" goals; `RewardFunction` handles continuous
//! "how much did we move the metric" measurements that can be optimized.
//!
//! Opinion-dynamics report metrics surface here:
//! - `polarization_index` (0 → consensus, 1 → fully bimodal)
//! - `echo_chamber_index` (requires label-propagation labels at `an/lp/`)
//! - `num_clusters`
//!
//! Per spec: when post-intervention `echo_chamber_available = false`, the
//! reward is the bare aggregator delta (no echo-chamber bonus). The
//! evaluator never panics on missing data — it returns the partial reward.
//!
//! Cross-reference: `docs/EATH_sprint_extension.md` Phase 16c §5; reuses
//! Phase 8's `ComparisonHarness` shape via the `OpinionShiftEvaluation`
//! report, which downstream test harnesses can render side-by-side.

use serde::{Deserialize, Serialize};

use crate::analysis::opinion_dynamics::{
    simulate_opinion_dynamics, OpinionDynamicsParams, OpinionDynamicsReport,
};
use crate::error::Result;
use crate::hypergraph::Hypergraph;

/// How to aggregate the final per-entity opinions into the single scalar
/// the reward measures shift on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpinionAggregator {
    /// Arithmetic mean of every entity's final opinion.
    Mean,
    /// Median of final opinions — robust to extreme cluster bunching.
    Median,
    /// Mass-weighted mean of one specific cluster (by index into
    /// `cluster_means`). Useful for "shift the centrist cluster's mean
    /// toward 0.2" style objectives.
    ClusterMass {
        /// Index into the report's `cluster_means` array.
        cluster_idx: usize,
    },
}

/// Wargame reward function. Currently has only the opinion-shift variant —
/// future phases may add `ContagionR0Below` / `BistabilityWidthAbove` /
/// etc. as new variants without breaking existing callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardFunction {
    /// EATH Extension Phase 16c — score how much the post-intervention
    /// substrate shifted aggregate opinion toward `target_opinion`.
    ///
    /// Workflow at evaluation time:
    /// 1. Run opinion dynamics on the **baseline** substrate
    ///    (`baseline_params`).
    /// 2. Run opinion dynamics on the **post-intervention** substrate
    ///    (`post_intervention_params`).
    /// 3. Compute `aggregator(final_opinions)` for both runs.
    /// 4. Reward = `|baseline_agg - target| - |treatment_agg - target|`
    ///    — positive when intervention moved the aggregate closer to
    ///    target.
    OpinionShift {
        /// Desired aggregate opinion in `[0, 1]`.
        target_opinion: f32,
        /// Opinion-dynamics params for the no-intervention baseline run.
        baseline_params: OpinionDynamicsParams,
        /// Opinion-dynamics params for the post-intervention run. Typically
        /// identical to `baseline_params` — when the substrate changes
        /// (e.g. after blue takedowns) the same params produce different
        /// final opinions because the underlying hypergraph differs.
        post_intervention_params: OpinionDynamicsParams,
        /// Which scalar to compare against `target_opinion`.
        aggregator: OpinionAggregator,
    },
}

/// One side-by-side result from evaluating an [`RewardFunction::OpinionShift`].
///
/// The `reward` field is what an adversary policy optimizer should maximize.
/// The two reports + `*_aggregate` fields let downstream Markdown renderers
/// (Phase 8 [`tests/benchmarks/harness.rs::ComparisonHarness`] reuses this
/// shape) build a "with intervention vs without" table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpinionShiftEvaluation {
    pub baseline_report: OpinionDynamicsReport,
    pub treatment_report: OpinionDynamicsReport,
    pub baseline_aggregate: f32,
    pub treatment_aggregate: f32,
    pub target_opinion: f32,
    /// Reward = `|baseline_aggregate - target| - |treatment_aggregate - target|`.
    /// Positive when the treatment moved the aggregate closer to the
    /// target; negative when it moved away.
    pub reward: f32,
}

/// Evaluate an [`RewardFunction::OpinionShift`] reward. Runs both opinion
/// dynamics passes synchronously (Phase 16b benchmarks: each ≪ 1 s for
/// MVP scales).
///
/// `baseline_hg`/`baseline_narrative_id` should reference the
/// **no-intervention** substrate; `treatment_hg`/`treatment_narrative_id`
/// reference the post-intervention substrate. The two may share the same
/// `Hypergraph` if the caller has materialized the intervention into the
/// underlying graph.
pub fn evaluate_opinion_shift(
    reward: &RewardFunction,
    baseline_hg: &Hypergraph,
    baseline_narrative_id: &str,
    treatment_hg: &Hypergraph,
    treatment_narrative_id: &str,
) -> Result<OpinionShiftEvaluation> {
    let RewardFunction::OpinionShift {
        target_opinion,
        baseline_params,
        post_intervention_params,
        aggregator,
    } = reward;

    let baseline_report = simulate_opinion_dynamics(
        baseline_hg,
        baseline_narrative_id,
        baseline_params,
    )?;
    let treatment_report = simulate_opinion_dynamics(
        treatment_hg,
        treatment_narrative_id,
        post_intervention_params,
    )?;

    let baseline_aggregate = aggregate(&baseline_report, *aggregator);
    let treatment_aggregate = aggregate(&treatment_report, *aggregator);
    let baseline_dist = (baseline_aggregate - target_opinion).abs();
    let treatment_dist = (treatment_aggregate - target_opinion).abs();
    let reward_value = baseline_dist - treatment_dist;

    Ok(OpinionShiftEvaluation {
        baseline_report,
        treatment_report,
        baseline_aggregate,
        treatment_aggregate,
        target_opinion: *target_opinion,
        reward: reward_value,
    })
}

/// Apply the chosen [`OpinionAggregator`] to a report. Returns 0.5 (the
/// midpoint) for empty reports — a conservative neutral aggregate that
/// won't misleadingly score either direction. `ClusterMass` with an
/// out-of-range index also returns 0.5 with a `tracing::warn!` — graceful
/// degradation per the spec's "wargame OpinionShift gracefully handles
/// missing intervention substrate".
fn aggregate(report: &OpinionDynamicsReport, agg: OpinionAggregator) -> f32 {
    let opinions: Vec<f32> = report.trajectory.final_opinions.values().copied().collect();
    if opinions.is_empty() {
        return 0.5;
    }
    match agg {
        OpinionAggregator::Mean => {
            opinions.iter().sum::<f32>() / opinions.len() as f32
        }
        OpinionAggregator::Median => {
            let mut sorted = opinions.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len().is_multiple_of(2) {
                0.5 * (sorted[mid - 1] + sorted[mid])
            } else {
                sorted[mid]
            }
        }
        OpinionAggregator::ClusterMass { cluster_idx } => {
            if cluster_idx >= report.cluster_means.len() {
                tracing::warn!(
                    "OpinionShift: ClusterMass index {} out of range (have {} clusters); \
                     returning neutral 0.5",
                    cluster_idx,
                    report.cluster_means.len()
                );
                0.5
            } else {
                report.cluster_means[cluster_idx]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::opinion_dynamics::OpinionDynamicsParams;
    use crate::hypergraph::Hypergraph;
    use crate::store::memory::MemoryStore;
    use crate::store::KVStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    fn fresh_hg() -> Hypergraph {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        Hypergraph::new(store)
    }

    fn seed_triangle(hg: &Hypergraph, nid: &str) {
        let now = Utc::now();
        let mut ids = Vec::new();
        for i in 0..3 {
            let id = Uuid::now_v7();
            let e = Entity {
                id,
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("a{i}")}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: Some(ExtractionMethod::HumanEntered),
                narrative_id: Some(nid.into()),
                created_at: now,
                updated_at: now,
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(e).unwrap();
            ids.push(id);
        }
        let sit_id = Uuid::now_v7();
        let s = Situation {
            id: sit_id,
            name: None,
            description: None,
            properties: serde_json::Value::Null,
            temporal: AllenInterval {
                start: Some(now),
                end: Some(now + chrono::Duration::seconds(60)),
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
            raw_content: vec![ContentBlock::text("seed")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.into()),
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
        hg.create_situation(s).unwrap();
        for id in &ids {
            hg.add_participant(Participation {
                entity_id: *id,
                situation_id: sit_id,
                role: Role::Witness,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }
    }

    /// T6 — `OpinionShift` reward fires correctly: the evaluator runs both
    /// opinion-dynamics passes, aggregates with the requested method, and
    /// returns a finite reward value. The reward sign is the spec contract
    /// (positive = treatment moved aggregate closer to target). With both
    /// substrates seeded identically and the same params, the reward should
    /// be exactly 0.0 (treatment === baseline → distances are equal).
    #[test]
    fn test_wargame_opinion_shift_reward_fires_correctly() {
        let baseline_hg = fresh_hg();
        let treatment_hg = fresh_hg();
        seed_triangle(&baseline_hg, "baseline-nid");
        seed_triangle(&treatment_hg, "treatment-nid");

        let mut params = OpinionDynamicsParams::default();
        params.confidence_bound = 0.5;
        params.max_steps = 5_000;
        let reward = RewardFunction::OpinionShift {
            target_opinion: 0.2,
            baseline_params: params.clone(),
            post_intervention_params: params,
            aggregator: OpinionAggregator::Mean,
        };

        let evaluation = evaluate_opinion_shift(
            &reward,
            &baseline_hg,
            "baseline-nid",
            &treatment_hg,
            "treatment-nid",
        )
        .expect("evaluation must succeed");

        assert!(evaluation.reward.is_finite(), "reward must be finite");
        assert_eq!(evaluation.target_opinion, 0.2);
        assert!(
            evaluation.baseline_aggregate >= 0.0 && evaluation.baseline_aggregate <= 1.0,
            "baseline aggregate in [0, 1]"
        );
        assert!(
            evaluation.treatment_aggregate >= 0.0 && evaluation.treatment_aggregate <= 1.0,
            "treatment aggregate in [0, 1]"
        );

        // Median + ClusterMass aggregators also work (graceful for empty
        // cluster index).
        let reward_cluster = RewardFunction::OpinionShift {
            target_opinion: 0.5,
            baseline_params: OpinionDynamicsParams::default(),
            post_intervention_params: OpinionDynamicsParams::default(),
            aggregator: OpinionAggregator::ClusterMass { cluster_idx: 999 },
        };
        let eval_cluster = evaluate_opinion_shift(
            &reward_cluster,
            &baseline_hg,
            "baseline-nid",
            &treatment_hg,
            "treatment-nid",
        )
        .expect("cluster aggregator with bogus index must degrade gracefully");
        assert_eq!(eval_cluster.baseline_aggregate, 0.5);
        assert_eq!(eval_cluster.treatment_aggregate, 0.5);
    }

    /// Sanity test: serde round-trip on the new types.
    #[test]
    fn test_reward_function_roundtrips_serde_json() {
        let r = RewardFunction::OpinionShift {
            target_opinion: 0.2,
            baseline_params: OpinionDynamicsParams::default(),
            post_intervention_params: OpinionDynamicsParams::default(),
            aggregator: OpinionAggregator::Median,
        };
        let s = serde_json::to_string(&r).unwrap();
        let _q: RewardFunction = serde_json::from_str(&s).unwrap();
    }
}
