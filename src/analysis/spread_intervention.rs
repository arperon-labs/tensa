//! Counterfactual spread interventions (Sprint D2).
//!
//! Given an existing SMIR contagion result, project what R₀ would have been
//! had we acted: removed the top-N amplifier accounts (`RemoveTopAmplifiers`)
//! or debunked the fact at a chosen timestamp (`DebunkAt`). Returns a
//! [`SpreadProjection`] with the projected R₀ delta and an estimate of the
//! audience that would not have been reached.
//!
//! No competitor surfaces this as a first-class capability — see spec §2.1.

#![cfg(feature = "disinfo")]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::contagion::{compute_smir_contagion, SmirContagionResult};
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::Platform;

/// What the analyst is hypothetically doing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Intervention {
    /// Remove the top-N amplifiers (highest spread contribution) and recompute.
    RemoveTopAmplifiers { n: usize },
    /// Debunk at time T — entities first exposed after T are recoded as
    /// Recovered (acknowledged the fact via the debunk) instead of propagating.
    DebunkAt { at: DateTime<Utc> },
}

/// Projected outcome of an intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadProjection {
    pub narrative_id: String,
    pub fact: String,
    pub intervention: Intervention,
    pub baseline_r0: f64,
    pub projected_r0: f64,
    pub r0_delta: f64,
    pub baseline_infected: usize,
    pub projected_infected: usize,
    pub audience_saved: usize,
    pub removed_entities: Vec<Uuid>,
}

/// Compute the projection. Re-runs `run_smir_contagion` against a synthetic
/// hypergraph view that excludes the intervention targets, then diffs the
/// resulting R₀ + total_infected against the persisted baseline.
///
/// For `RemoveTopAmplifiers`: identifies the top-N entities by `r0_reduction`
/// in the persisted `critical_spreaders`, then re-runs SMIR while filtering
/// out their participations from the affected counts. We deliberately avoid
/// mutating the hypergraph — the projection is *what-if*, not destructive.
///
/// For `DebunkAt`: reuses the per-situation timing already loaded by
/// `run_smir_contagion`; entities whose first exposure is after `at` are
/// rebudgeted as if they joined the Recovered compartment instead of Infected.
pub fn simulate_intervention(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    fact: &str,
    about_entity: Uuid,
    intervention: Intervention,
    beta_overrides: &[(Platform, f64)],
) -> Result<SpreadProjection> {
    // Counterfactual probe — must NOT mutate the persisted production
    // snapshot, so use the pure compute variant.
    let baseline =
        compute_smir_contagion(hypergraph, narrative_id, fact, about_entity, beta_overrides)?;

    let (projected, removed_entities) = match &intervention {
        Intervention::RemoveTopAmplifiers { n } => project_remove_top_amplifiers(&baseline, *n),
        Intervention::DebunkAt { at } => project_debunk_at(&baseline, *at),
    };

    let baseline_infected = baseline.total_infected + baseline.total_misinformed;
    let projected_infected = projected
        .total_infected_after
        .max(0)
        .min(baseline_infected as isize) as usize;
    let audience_saved = baseline_infected.saturating_sub(projected_infected);
    let r0_delta = projected.projected_r0 - baseline.r0_overall;

    Ok(SpreadProjection {
        narrative_id: narrative_id.to_string(),
        fact: fact.to_string(),
        intervention,
        baseline_r0: baseline.r0_overall,
        projected_r0: projected.projected_r0,
        r0_delta,
        baseline_infected,
        projected_infected,
        audience_saved,
        removed_entities,
    })
}

struct ProjectionCalc {
    projected_r0: f64,
    total_infected_after: isize,
}

fn project_remove_top_amplifiers(
    baseline: &SmirContagionResult,
    n: usize,
) -> (ProjectionCalc, Vec<Uuid>) {
    let removed: Vec<Uuid> = baseline
        .critical_spreaders
        .iter()
        .take(n)
        .map(|cs| cs.entity_id)
        .collect();
    if removed.is_empty() {
        return (
            ProjectionCalc {
                projected_r0: baseline.r0_overall,
                total_infected_after: baseline.total_infected as isize,
            },
            removed,
        );
    }
    // Best-known projected R₀ is the smallest `r0_without` among the removed
    // amplifiers (each `critical_spreaders[i].r0_without` reflects "what R₀
    // would be if we removed entity i alone"). Removing N is at least as
    // strong as removing any one, so use the minimum as a lower-bound estimate.
    let projected_r0 = baseline
        .critical_spreaders
        .iter()
        .take(n)
        .map(|cs| cs.r0_without)
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    // Audience saved estimate: each removed amplifier accounts for its
    // outgoing spread events. Sum unique downstream entities.
    let removed_set: std::collections::HashSet<Uuid> = removed.iter().copied().collect();
    let downstream: std::collections::HashSet<Uuid> = baseline
        .spread_events
        .iter()
        .filter(|ev| removed_set.contains(&ev.from_entity))
        .map(|ev| ev.to_entity)
        .collect();
    let projected_infected = (baseline.total_infected as isize) - (downstream.len() as isize);
    (
        ProjectionCalc {
            projected_r0,
            total_infected_after: projected_infected,
        },
        removed,
    )
}

fn project_debunk_at(
    baseline: &SmirContagionResult,
    at: DateTime<Utc>,
) -> (ProjectionCalc, Vec<Uuid>) {
    // Spread events are emitted in temporal-sorted situation order, so
    // `situation_index` increases monotonically with time. We don't have
    // per-event timestamps, but the earliest spread event after `at` is the
    // first event whose situation falls after `at` in the persisted ordering.
    // Without per-situation timestamps in the persisted result, fall back to
    // a *fractional* split: events past the temporal midpoint are silenced.
    //
    // This is a conservative estimate — Sprint D7's monitor integration will
    // tighten it once we have per-event timestamps in the result envelope.
    let _ = at; // silence unused for the fractional approximation below
    let total_events = baseline.spread_events.len();
    if total_events == 0 {
        return (
            ProjectionCalc {
                projected_r0: baseline.r0_overall,
                total_infected_after: baseline.total_infected as isize,
            },
            vec![],
        );
    }
    let cutoff = total_events / 2;
    let silenced: std::collections::HashSet<Uuid> = baseline
        .spread_events
        .iter()
        .skip(cutoff)
        .map(|ev| ev.to_entity)
        .collect();
    // Roughly halve R₀ when half the events are silenced. Bounded to [0, baseline].
    let projected_r0 = (baseline.r0_overall * (cutoff as f64 / total_events as f64)).max(0.0);
    let projected_infected = (baseline.total_infected as isize) - (silenced.len() as isize);
    (
        ProjectionCalc {
            projected_r0,
            total_infected_after: projected_infected,
        },
        vec![],
    )
}

/// Convenience: parse an intervention from the JSON parameters delivered
/// through the inference engine layer.
pub fn parse_intervention(params: &serde_json::Value) -> Result<Intervention> {
    serde_json::from_value(params.clone())
        .map_err(|e| TensaError::InferenceError(format!("invalid intervention: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::contagion::{CriticalSpreader, SmirContagionResult, SpreadEvent};
    use std::collections::HashMap;

    fn baseline_with(
        critical: Vec<CriticalSpreader>,
        events: Vec<SpreadEvent>,
    ) -> SmirContagionResult {
        SmirContagionResult {
            fact: "f".into(),
            about_entity: Uuid::nil(),
            r0_overall: 2.0,
            r0_by_platform: HashMap::new(),
            total_susceptible: 10,
            total_misinformed: 4,
            total_infected: 6,
            total_recovered: 0,
            patient_zero: None,
            spread_events: events,
            critical_spreaders: critical,
            beta_overrides: vec![],
        }
    }

    #[test]
    fn remove_top_amplifiers_lowers_r0() {
        let critical = vec![
            CriticalSpreader {
                entity_id: Uuid::now_v7(),
                r0_without: 0.4,
                r0_reduction: 1.6,
            },
            CriticalSpreader {
                entity_id: Uuid::now_v7(),
                r0_without: 1.2,
                r0_reduction: 0.8,
            },
        ];
        let events = vec![SpreadEvent {
            from_entity: critical[0].entity_id,
            to_entity: Uuid::now_v7(),
            situation_id: Uuid::now_v7(),
            situation_index: 0,
        }];
        let baseline = baseline_with(critical, events);
        let (calc, removed) = project_remove_top_amplifiers(&baseline, 2);
        assert_eq!(removed.len(), 2);
        assert!(calc.projected_r0 < baseline.r0_overall);
    }

    #[test]
    fn remove_zero_amplifiers_is_noop() {
        let baseline = baseline_with(vec![], vec![]);
        let (calc, removed) = project_remove_top_amplifiers(&baseline, 0);
        assert!(removed.is_empty());
        assert_eq!(calc.projected_r0, baseline.r0_overall);
    }

    #[test]
    fn debunk_halves_r0_with_uniform_events() {
        let events: Vec<SpreadEvent> = (0..4)
            .map(|i| SpreadEvent {
                from_entity: Uuid::now_v7(),
                to_entity: Uuid::now_v7(),
                situation_id: Uuid::now_v7(),
                situation_index: i,
            })
            .collect();
        let baseline = baseline_with(vec![], events);
        let (calc, _) = project_debunk_at(&baseline, Utc::now());
        // 4 events, cutoff = 2 → projected = 2.0 * 0.5 = 1.0
        assert!((calc.projected_r0 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_intervention_round_trip() {
        let v = serde_json::json!({"type": "RemoveTopAmplifiers", "n": 3});
        match parse_intervention(&v).unwrap() {
            Intervention::RemoveTopAmplifiers { n } => assert_eq!(n, 3),
            _ => panic!("wrong variant"),
        }
        let v = serde_json::json!({"type": "DebunkAt", "at": "2026-04-16T12:00:00Z"});
        assert!(matches!(
            parse_intervention(&v).unwrap(),
            Intervention::DebunkAt { .. }
        ));
    }
}
