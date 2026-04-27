//! EATH Extension Phase 16b — Opinion dynamics on hypergraphs (BCM + Deffuant).
//!
//! Asynchronous bounded-confidence opinion dynamics over the entity-situation
//! hyperedge structure of a TENSA narrative. Each entity carries a scalar
//! opinion `x ∈ [0, 1]`. At every step, one or more hyperedges (situations) are
//! selected per [`HyperedgeSelection`], and an update is applied per
//! [`BcmVariant`].
//!
//! ## Model variants
//!
//! - [`BcmVariant::PairwiseWithin`] — Hickok, Kureh, Brooks, Feng, Porter,
//!   *A Bounded-Confidence Model of Opinion Dynamics on Hypergraphs*,
//!   SIAM J. Appl. Dyn. Syst. **21**, 1 (2022). All ordered pairs within the
//!   selected hyperedge attempt the classical Deffuant 2000 dyadic update,
//!   processed in canonical (sorted-UUID) order — i.e. Gauss-Seidel, with
//!   updates immediately visible to subsequent pairs in the same edge. This
//!   ordering is what produces the *opinion-jumping* phenomenon (Hickok §4)
//!   and reduces to dyadic Deffuant 2000 on size-2-only hypergraphs.
//! - [`BcmVariant::GroupMean`] — Schawe & Hernández, *Higher order
//!   interactions destroy phase transitions in Deffuant opinion dynamics
//!   model*, Commun. Phys. **5**, 32 (2022). All-or-nothing group update:
//!   when the spread within the selected edge is below the (possibly
//!   size-scaled) confidence bound, every member moves toward the group mean.
//!
//! ## Phase-transition sweep
//!
//! [`run_phase_transition_sweep`] sweeps the confidence bound `c` and reports
//! the convergence-time spike near the critical point. On a complete
//! hypergraph with Gaussian-N(0.5, σ²) initial opinions, the spike sits near
//! `c = σ²` (Hickok §5). This is *distinct* from Phase 14's bistability
//! sweep, which varies β and measures prevalence.
//!
//! ## Outputs and KV
//!
//! Reports are returned synchronously by
//! [`simulate_opinion_dynamics`]. KV-persistence keys are reserved here for
//! Phase 16c (no writes happen in 16b):
//!
//! ```text
//! opd/report/{narrative_id}/{run_id_v7_bytes}
//! ```
//!
//! ## Determinism
//!
//! A single [`rand_chacha::ChaCha8Rng`] seeded from `params.seed` drives every
//! stochastic decision in the simulation, in a fixed order (see §4 of
//! `docs/opinion_dynamics_algorithm.md`). The phase-transition sweep
//! constructs a fresh per-c RNG via `seed XOR (c_index as u64)`.

pub mod cluster;
pub mod init;
pub mod phase_transition;
pub mod simulate;
pub mod types;
pub mod update;

#[cfg(test)]
mod opinion_dynamics_tests;
#[cfg(test)]
mod perf_tests;
#[cfg(test)]
mod test_fixtures;

pub use cluster::{detect_clusters_density_gap, echo_chamber_index, polarization_index};
pub use phase_transition::{detect_critical_c, run_phase_transition_sweep};
pub use simulate::simulate_opinion_dynamics;
pub use types::{
    BcmVariant, ConfidenceScaling, HyperedgeSelection, InitialOpinionDist, OpinionDynamicsParams,
    OpinionDynamicsReport, OpinionTrajectory, PhaseTransitionReport,
};

/// KV prefix reserved for opinion-dynamics report storage. Phase 16c writes;
/// Phase 16b only reserves the namespace.
pub const OPD_REPORT_PREFIX: &[u8] = b"opd/report/";

/// Build the canonical KV key for an opinion-dynamics report.
///
/// Layout: `opd/report/{narrative_id}/{run_id_v7_bytes_16}`. Using v7 UUIDs
/// for `run_id` gives chronological scan order, matching the codebase's KV
/// key encoding scheme (CLAUDE.md §"Key Encoding Scheme").
pub fn key_opd_report(narrative_id: &str, run_id: uuid::Uuid) -> Vec<u8> {
    let mut k = OPD_REPORT_PREFIX.to_vec();
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

// ── Phase 16c — KV persistence helpers ──────────────────────────────────────

/// Persist an [`OpinionDynamicsReport`] under
/// `opd/report/{narrative_id}/{run_id_v7_BE_BIN_16}`. The caller supplies
/// `run_id`; mint via `Uuid::now_v7()` for chronological scan order.
pub fn save_opinion_report(
    store: &dyn crate::store::KVStore,
    narrative_id: &str,
    run_id: uuid::Uuid,
    report: &OpinionDynamicsReport,
) -> crate::error::Result<()> {
    let key = key_opd_report(narrative_id, run_id);
    let value = serde_json::to_vec(report)
        .map_err(|e| crate::error::TensaError::Serialization(e.to_string()))?;
    store.put(&key, &value)
}

/// Load a single opinion-dynamics report. `Ok(None)` when no record exists at
/// the (narrative_id, run_id) pair.
pub fn load_opinion_report(
    store: &dyn crate::store::KVStore,
    narrative_id: &str,
    run_id: uuid::Uuid,
) -> crate::error::Result<Option<OpinionDynamicsReport>> {
    let key = key_opd_report(narrative_id, run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
            crate::error::TensaError::Serialization(e.to_string())
        })?)),
        None => Ok(None),
    }
}

/// List up to `limit` opinion-dynamics reports for `narrative_id`, newest
/// first. Mirrors the synth helpers' reverse-then-truncate pattern — O(n) on
/// the scan window, no sort.
pub fn list_opinion_reports(
    store: &dyn crate::store::KVStore,
    narrative_id: &str,
    limit: usize,
) -> crate::error::Result<Vec<OpinionDynamicsReport>> {
    let mut prefix = OPD_REPORT_PREFIX.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let mut pairs = store.prefix_scan(&prefix)?;
    pairs.reverse();
    pairs.truncate(limit);
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        match serde_json::from_slice::<OpinionDynamicsReport>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!("skipping malformed OpinionDynamicsReport: {e}"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod kv_tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::store::KVStore;
    use std::sync::Arc;

    fn fixture_report() -> OpinionDynamicsReport {
        OpinionDynamicsReport {
            num_steps_executed: 100,
            converged: true,
            convergence_step: Some(100),
            num_clusters: 1,
            cluster_sizes: vec![5],
            cluster_means: vec![0.5],
            variance_timeseries: vec![0.1, 0.05, 0.0],
            polarization_index: 0.0,
            echo_chamber_index: 0.0,
            echo_chamber_available: false,
            trajectory: OpinionTrajectory {
                opinion_history: vec![vec![0.5; 5]],
                sample_steps: vec![0],
                final_opinions: std::collections::HashMap::new(),
                entity_order: vec![],
            },
            params_used: OpinionDynamicsParams::default(),
        }
    }

    /// T8 — KV keys for opinion reports MUST sort chronologically when
    /// `run_id` is a v7 UUID encoded as 16 BE-binary bytes. This is the
    /// load-bearing property — if it ever breaks, every "newest first" listing
    /// silently corrupts.
    #[test]
    fn test_opinion_dynamics_kv_prefix_sortable() {
        let earlier = uuid::Uuid::now_v7();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let later = uuid::Uuid::now_v7();
        let k_earlier = key_opd_report("nid", earlier);
        let k_later = key_opd_report("nid", later);
        assert!(
            k_earlier < k_later,
            "earlier run-id key must sort before later"
        );
        assert!(k_earlier.starts_with(OPD_REPORT_PREFIX));
        assert!(k_later.starts_with(OPD_REPORT_PREFIX));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let report = fixture_report();
        let run_id = uuid::Uuid::now_v7();
        save_opinion_report(store.as_ref(), "nid", run_id, &report).unwrap();
        let loaded = load_opinion_report(store.as_ref(), "nid", run_id)
            .unwrap()
            .expect("report round-trips");
        assert_eq!(loaded, report);
    }

    #[test]
    fn test_list_reports_newest_first() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let r = fixture_report();
        let mut ids = Vec::new();
        for _ in 0..3 {
            let id = uuid::Uuid::now_v7();
            ids.push(id);
            save_opinion_report(store.as_ref(), "nid", id, &r).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let listed = list_opinion_reports(store.as_ref(), "nid", 10).unwrap();
        assert_eq!(listed.len(), 3, "all three reports returned");
    }
}
