//! Surrogate generation models for temporal hypergraphs (EATH sprint).
//!
//! This module is a **layer parallel to `analysis/` and `inference/`**: it
//! depends only on `hypergraph/`, `store/`, `error/`, and `types/`, and is
//! never depended on by anything below itself. Synthetic-narrative generation
//! is a generative task fundamentally different from inference (which reads
//! the graph) or analysis (which scores existing structure).
//!
//! ## Why a trait, not a struct
//!
//! The first concrete model is **EATH** (Effective Active Temporal Hypergraph)
//! from Mancastroppa, Cencetti, Barrat — see arXiv:2507.01124v2. EATH treats
//! actors as active/inactive "stations" with phase-distributed activations and
//! group-recruitment dynamics, producing realistic burstiness and group-size
//! distributions on temporal hypergraphs.
//!
//! Future surrogate families (Iacopini-style hyperedge configuration,
//! HAD-style hyperedge-aware degeneracy, narrative-conditioned diffusion
//! models, etc.) plug in via the [`SurrogateModel`] trait and the
//! [`SurrogateRegistry`]; the TensaQL grammar and REST API treat the model
//! name as a string, never a hardcoded keyword.
//!
//! ## KV layout
//!
//! Six KV prefixes under `syn/`. **Run UUIDs are encoded in 16-byte
//! big-endian binary** (NOT hex strings). UUIDv7 is time-ordered, so
//! big-endian binary keys make `prefix_scan` return runs in chronological
//! order (oldest first); "newest first" listings reverse the scan output —
//! O(n) on the scan window, never O(n log n) sort.
//!
//! ```text
//!   syn/p/{narrative_id_utf8}/{model_utf8}              → SurrogateParams
//!   syn/r/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}     → SurrogateRunSummary
//!   syn/seed/{run_id_v7_BE_BIN_16}                      → ReproducibilityBlob
//!   syn/lineage/{narrative_id_utf8}/{run_id_v7_BE_BIN_16} → lineage marker
//!   syn/fidelity/{narrative_id_utf8}/{run_id_v7_BE_BIN_16} → FidelityReport (Phase 2.5)
//!   syn/sig/{narrative_id_utf8}/{metric_utf8}/{run_id_v7_BE_BIN_16} → significance result (Phase 7)
//!   syn/recon/{output_narrative_id_utf8}/{job_id_utf8}/{situation_id_v7_BE_BIN_16} → reconstructed situation (Phase 15c)
//! ```
//!
//! Always go through the `key_synth_*` helpers — they are the single source
//! of truth for the encoding contract above.

pub mod activity;
pub mod bistability_significance_engine;
pub mod bistability_significance_samples;
#[cfg(test)]
pub mod bistability_tests;
pub mod calibrate;
mod calibrate_fitters;
pub mod dual_significance_engine;
pub mod eath;
pub mod eath_recruit;
pub mod emit;
pub mod engines;
pub mod fidelity;
mod fidelity_metrics;
pub mod fidelity_pipeline;
pub mod hashing;
pub mod hybrid;
#[cfg(test)]
pub mod invariant_tests;
pub mod memory;
pub mod nudhy;
pub mod nudhy_surrogate;
pub mod opinion_significance_engine;
pub mod registry;
pub mod significance;
pub mod surrogate;
pub mod types;

use chrono::Utc;
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::store::KVStore;

pub use bistability_significance_engine::SurrogateBistabilitySignificanceEngine;
pub use dual_significance_engine::SurrogateDualSignificanceEngine;
pub use engines::SurrogateContagionSignificanceEngine;
pub use opinion_significance_engine::SurrogateOpinionSignificanceEngine;
pub use registry::SurrogateRegistry;
pub use significance::{
    list_significance_reports, load_significance_report, save_significance_report,
    SignificanceMetric, SignificanceReport, SurrogateSignificanceEngine, SyntheticDistribution,
};
pub use surrogate::SurrogateModel;
pub use types::{
    BistabilitySignificance, BistabilitySignificanceReport, CombinedSignificance,
    DualSignificanceReport, EathParams, OpinionSignificanceReport, ReproducibilityBlob, RunKind,
    SingleModelBistabilityNull, SingleModelOpinionNull, SingleModelSignificance, SurrogateParams,
    SurrogateRunSummary,
};

// ── KV prefix constants (single source of truth) ─────────────────────────────

pub const SYNTH_PARAMS_PREFIX: &[u8] = b"syn/p/";
pub const SYNTH_RESULT_PREFIX: &[u8] = b"syn/r/";
pub const SYNTH_SEED_PREFIX: &[u8] = b"syn/seed/";
pub const SYNTH_LINEAGE_PREFIX: &[u8] = b"syn/lineage/";
pub const SYNTH_FIDELITY_PREFIX: &[u8] = b"syn/fidelity/";
pub const SYNTH_SIG_PREFIX: &[u8] = b"syn/sig/";
/// Phase 13c — dual-null-model significance reports. Disjoint from
/// `SYNTH_SIG_PREFIX` so a `prefix_scan` on the latter never accidentally
/// returns a dual report. Schema:
/// `syn/dual_sig/{narrative_id}/{metric}/{run_id_v7_BE_BIN_16}`.
pub const SYN_DUAL_SIG_PREFIX: &[u8] = b"syn/dual_sig/";

/// Phase 14 — bistability-significance reports. Disjoint from every other
/// `syn/*` slice. Schema:
/// `syn/bistability/{narrative_id}/{run_id_v7_BE_BIN_16}`.
pub const SYN_BISTABILITY_PREFIX: &[u8] = b"syn/bistability/";

/// EATH Extension Phase 16c — opinion-dynamics-significance reports.
/// Disjoint from every other `syn/*` slice so opinion null-model runs never
/// collide with the contagion / bistability / structural-pattern significance
/// indexes. Schema:
/// `syn/opinion_sig/{narrative_id}/{run_id_v7_BE_BIN_16}`.
pub const SYN_OPINION_SIG_PREFIX: &[u8] = b"syn/opinion_sig/";

/// EATH Extension Phase 15c — materialized hypergraph-reconstruction
/// outputs. Disjoint from every other `syn/*` slice so reconstruction
/// records never collide with EATH/NuDHy generation runs. Schema:
/// `syn/recon/{output_narrative_id_utf8}/{job_id_utf8}/{situation_id_v7_BE_BIN_16}`.
///
/// Materialization is opt-in via
/// `POST /inference/hypergraph-reconstruction/{job_id}/materialize`. Each
/// situation key persists a `ReconstructedSituationRef` linking the
/// materialized Situation back to its source narrative + job (also encoded
/// in `Situation.extraction_method = ExtractionMethod::Reconstructed`).
pub const SYN_RECON_PREFIX: &[u8] = b"syn/recon/";

/// User-editable fidelity thresholds live in the existing `cfg/` namespace
/// (alongside `cfg/llm`, `cfg/rag`, etc.) — NOT under `syn/*`. Phase 2.5
/// `fidelity` module owns the persist/load helpers; this constant is
/// re-exported here so callers see all KV prefixes in one place.
pub use fidelity::SYNTH_FIDELITY_THRESHOLDS_PREFIX;

// ── Key builders (NEVER assemble keys manually) ──────────────────────────────

/// `syn/p/{narrative_id}/{model}` — calibrated params for one (narrative, model).
pub fn key_synth_params(narrative_id: &str, model: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(SYNTH_PARAMS_PREFIX.len() + narrative_id.len() + model.len() + 1);
    k.extend_from_slice(SYNTH_PARAMS_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(model.as_bytes());
    k
}

/// `syn/r/{narrative_id}/{run_id_v7_BE_BIN_16}` — one run summary record.
pub fn key_synth_run(narrative_id: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(SYNTH_RESULT_PREFIX.len() + narrative_id.len() + 17);
    k.extend_from_slice(SYNTH_RESULT_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/seed/{run_id_v7_BE_BIN_16}` — full ReproducibilityBlob.
pub fn key_synth_seed(run_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(SYNTH_SEED_PREFIX.len() + 16);
    k.extend_from_slice(SYNTH_SEED_PREFIX);
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/lineage/{narrative_id}/{run_id_v7_BE_BIN_16}` — lineage marker (value = empty).
pub fn key_synth_lineage(narrative_id: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k =
        Vec::with_capacity(SYNTH_LINEAGE_PREFIX.len() + narrative_id.len() + 17);
    k.extend_from_slice(SYNTH_LINEAGE_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/fidelity/{narrative_id}/{run_id_v7_BE_BIN_16}` — fidelity report (Phase 2.5).
pub fn key_synth_fidelity(narrative_id: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k =
        Vec::with_capacity(SYNTH_FIDELITY_PREFIX.len() + narrative_id.len() + 17);
    k.extend_from_slice(SYNTH_FIDELITY_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/sig/{narrative_id}/{metric}/{run_id_v7_BE_BIN_16}` — Phase 7 significance row.
pub fn key_synth_sig(narrative_id: &str, metric: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(
        SYNTH_SIG_PREFIX.len() + narrative_id.len() + metric.len() + 18,
    );
    k.extend_from_slice(SYNTH_SIG_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(metric.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/bistability/{narrative_id}/{run_id_v7_BE_BIN_16}` — Phase 14
/// bistability-significance report. Note this slice is keyed only by
/// `narrative_id` (no `metric` segment) because bistability is a single
/// observation per (narrative, run) — width and gap are inside the report.
pub fn key_synth_bistability(narrative_id: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(SYN_BISTABILITY_PREFIX.len() + narrative_id.len() + 17);
    k.extend_from_slice(SYN_BISTABILITY_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/opinion_sig/{narrative_id}/{run_id_v7_BE_BIN_16}` — Phase 16c
/// opinion-dynamics-significance report. Single observation per
/// (narrative, run) — three scalars (num_clusters, polarization,
/// echo_chamber) folded into the report.
pub fn key_synth_opinion_sig(narrative_id: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(SYN_OPINION_SIG_PREFIX.len() + narrative_id.len() + 17);
    k.extend_from_slice(SYN_OPINION_SIG_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// `syn/recon/{output_narrative_id}/{job_id_utf8}/{situation_id_v7_BE_BIN_16}` —
/// Phase 15c materialized reconstruction situation. `job_id` is the SHA-style
/// UUID string of the reconstruction `InferenceJob`; the situation UUID is
/// stored as the canonical 16-byte big-endian binary suffix so per-job scans
/// land in chronological order.
pub fn key_synth_recon_situation(
    output_narrative_id: &str,
    job_id: &str,
    situation_id: &Uuid,
) -> Vec<u8> {
    let mut k = Vec::with_capacity(
        SYN_RECON_PREFIX.len() + output_narrative_id.len() + job_id.len() + 18,
    );
    k.extend_from_slice(SYN_RECON_PREFIX);
    k.extend_from_slice(output_narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(job_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(situation_id.as_bytes());
    k
}

/// `syn/dual_sig/{narrative_id}/{metric}/{run_id_v7_BE_BIN_16}` — Phase 13c
/// dual-null-model significance report. Mirrors [`key_synth_sig`] structurally
/// but uses a disjoint prefix.
pub fn key_synth_dual_sig(narrative_id: &str, metric: &str, run_id: &Uuid) -> Vec<u8> {
    let mut k = Vec::with_capacity(
        SYN_DUAL_SIG_PREFIX.len() + narrative_id.len() + metric.len() + 18,
    );
    k.extend_from_slice(SYN_DUAL_SIG_PREFIX);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k.extend_from_slice(metric.as_bytes());
    k.push(b'/');
    k.extend_from_slice(run_id.as_bytes());
    k
}

/// Build a prefix-scan boundary `<prefix>{narrative_id}/`. Used by helpers
/// that scan a per-narrative slice of one of the `syn/*` indexes.
pub(crate) fn narrative_scan_prefix(prefix: &[u8], narrative_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(prefix.len() + narrative_id.len() + 1);
    k.extend_from_slice(prefix);
    k.extend_from_slice(narrative_id.as_bytes());
    k.push(b'/');
    k
}

// ── Persistence helpers ──────────────────────────────────────────────────────

/// Write a `ReproducibilityBlob` so the run can be replayed later.
pub fn store_reproducibility_blob(
    store: &dyn KVStore,
    blob: &ReproducibilityBlob,
) -> Result<()> {
    let key = key_synth_seed(&blob.run_id);
    let value = serde_json::to_vec(blob)?;
    store.put(&key, &value)
}

/// Load a previously-stored `ReproducibilityBlob` by run id.
pub fn load_reproducibility_blob(
    store: &dyn KVStore,
    run_id: &Uuid,
) -> Result<Option<ReproducibilityBlob>> {
    let key = key_synth_seed(run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// Mark `run_id` as part of `narrative_id`'s synthetic lineage.
/// Value is empty: lineage is a presence-only index.
pub fn record_lineage_run(
    store: &dyn KVStore,
    narrative_id: &str,
    run_id: &Uuid,
) -> Result<()> {
    let key = key_synth_lineage(narrative_id, run_id);
    store.put(&key, &[])
}

/// List every run id under `narrative_id`'s lineage index, oldest first
/// (UUIDv7 + big-endian binary suffix → chronological scan order).
pub fn list_lineage_runs(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<Uuid>> {
    let prefix = narrative_scan_prefix(SYNTH_LINEAGE_PREFIX, narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (k, _v) in pairs {
        if k.len() < prefix.len() + 16 {
            continue;
        }
        let suffix = &k[prefix.len()..];
        let bytes: [u8; 16] = match suffix.try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };
        out.push(Uuid::from_bytes(bytes));
    }
    Ok(out)
}

// ── Phase 13c — dual-significance persistence helpers ───────────────────────

/// Persist a [`types::DualSignificanceReport`] at
/// `syn/dual_sig/{narrative_id}/{metric}/{run_id_BE}`.
pub fn save_dual_significance_report(
    store: &dyn KVStore,
    report: &types::DualSignificanceReport,
) -> Result<()> {
    let key = key_synth_dual_sig(&report.narrative_id, &report.metric, &report.run_id);
    let value = serde_json::to_vec(report)?;
    store.put(&key, &value)
}

/// Load a single dual-significance report. `Ok(None)` when no record exists at
/// the (narrative, metric, run_id) triple.
pub fn load_dual_significance_report(
    store: &dyn KVStore,
    narrative_id: &str,
    metric: &str,
    run_id: &Uuid,
) -> Result<Option<types::DualSignificanceReport>> {
    let key = key_synth_dual_sig(narrative_id, metric, run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List up to `limit` dual reports for `(narrative_id, metric)`, newest first.
/// Mirrors [`significance::list_significance_reports`]'s reverse-then-truncate
/// pattern — O(n) on the scan window, no sort.
pub fn list_dual_significance_reports(
    store: &dyn KVStore,
    narrative_id: &str,
    metric: &str,
    limit: usize,
) -> Result<Vec<types::DualSignificanceReport>> {
    let mut prefix = narrative_scan_prefix(SYN_DUAL_SIG_PREFIX, narrative_id);
    prefix.extend_from_slice(metric.as_bytes());
    prefix.push(b'/');
    let mut pairs = store.prefix_scan(&prefix)?;
    pairs.reverse();
    pairs.truncate(limit);
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        match serde_json::from_slice::<types::DualSignificanceReport>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!("skipping malformed DualSignificanceReport: {e}"),
        }
    }
    Ok(out)
}

// ── Phase 14 — bistability-significance persistence helpers ─────────────────

/// Persist a [`types::BistabilitySignificanceReport`] at
/// `syn/bistability/{narrative_id}/{run_id_BE}`.
pub fn save_bistability_significance_report(
    store: &dyn KVStore,
    report: &types::BistabilitySignificanceReport,
) -> Result<()> {
    let key = key_synth_bistability(&report.narrative_id, &report.run_id);
    let value = serde_json::to_vec(report)?;
    store.put(&key, &value)
}

/// Load a single bistability-significance report. `Ok(None)` when no record
/// exists at the (narrative, run_id) pair.
pub fn load_bistability_significance_report(
    store: &dyn KVStore,
    narrative_id: &str,
    run_id: &Uuid,
) -> Result<Option<types::BistabilitySignificanceReport>> {
    let key = key_synth_bistability(narrative_id, run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List up to `limit` bistability-significance reports for `narrative_id`,
/// newest first.
pub fn list_bistability_significance_reports(
    store: &dyn KVStore,
    narrative_id: &str,
    limit: usize,
) -> Result<Vec<types::BistabilitySignificanceReport>> {
    let prefix = narrative_scan_prefix(SYN_BISTABILITY_PREFIX, narrative_id);
    let mut pairs = store.prefix_scan(&prefix)?;
    pairs.reverse();
    pairs.truncate(limit);
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        match serde_json::from_slice::<types::BistabilitySignificanceReport>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!(
                "skipping malformed BistabilitySignificanceReport: {e}"
            ),
        }
    }
    Ok(out)
}

// ── Phase 16c — Opinion-dynamics-significance persistence helpers ───────────

/// Persist an [`types::OpinionSignificanceReport`] at
/// `syn/opinion_sig/{narrative_id}/{run_id_BE}`.
pub fn save_opinion_significance_report(
    store: &dyn KVStore,
    report: &types::OpinionSignificanceReport,
) -> Result<()> {
    let key = key_synth_opinion_sig(&report.narrative_id, &report.run_id);
    let value = serde_json::to_vec(report)?;
    store.put(&key, &value)
}

/// Load a single opinion-dynamics-significance report. `Ok(None)` when no
/// record exists at the (narrative, run_id) pair.
pub fn load_opinion_significance_report(
    store: &dyn KVStore,
    narrative_id: &str,
    run_id: &Uuid,
) -> Result<Option<types::OpinionSignificanceReport>> {
    let key = key_synth_opinion_sig(narrative_id, run_id);
    match store.get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

/// List up to `limit` opinion-dynamics-significance reports for
/// `narrative_id`, newest first.
pub fn list_opinion_significance_reports(
    store: &dyn KVStore,
    narrative_id: &str,
    limit: usize,
) -> Result<Vec<types::OpinionSignificanceReport>> {
    let prefix = narrative_scan_prefix(SYN_OPINION_SIG_PREFIX, narrative_id);
    let mut pairs = store.prefix_scan(&prefix)?;
    pairs.reverse();
    pairs.truncate(limit);
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        match serde_json::from_slice::<types::OpinionSignificanceReport>(&v) {
            Ok(r) => out.push(r),
            Err(e) => tracing::warn!("skipping malformed OpinionSignificanceReport: {e}"),
        }
    }
    Ok(out)
}

/// List the most recent `limit` runs for `narrative_id`. Reverses the
/// chronological prefix scan in-place — O(n) on the scan window, no sort.
pub fn list_runs_newest_first(
    store: &dyn KVStore,
    narrative_id: &str,
    limit: usize,
) -> Result<Vec<SurrogateRunSummary>> {
    let prefix = narrative_scan_prefix(SYNTH_RESULT_PREFIX, narrative_id);
    let mut pairs = store.prefix_scan(&prefix)?;
    pairs.reverse();
    pairs.truncate(limit);
    let mut out = Vec::with_capacity(pairs.len());
    for (_k, v) in pairs {
        match serde_json::from_slice::<SurrogateRunSummary>(&v) {
            Ok(s) => out.push(s),
            Err(e) => tracing::warn!("skipping malformed SurrogateRunSummary: {e}"),
        }
    }
    Ok(out)
}

// ── Run-id minting (centralized so every caller agrees on the contract) ──────

/// Mint a new run id. UUIDv7, time-ordered, suitable for chronological prefix
/// scans when encoded as 16-byte big-endian binary in keys.
pub fn new_run_id() -> Uuid {
    Uuid::now_v7()
}

/// Build a `ReproducibilityBlob` from the components callers usually have.
/// `git_sha` is best-effort: callers who don't know the SHA pass `None`.
pub fn build_reproducibility_blob(
    run_id: Uuid,
    params: SurrogateParams,
    source_state_hash: Option<String>,
) -> ReproducibilityBlob {
    let model = params.model.clone();
    let seed = params.seed;
    ReproducibilityBlob {
        run_id,
        model,
        params_full: params,
        seed,
        git_sha: option_env!("TENSA_GIT_HASH").map(str::to_owned),
        tensa_version: env!("CARGO_PKG_VERSION").to_string(),
        captured_at: Utc::now(),
        source_state_hash,
    }
}

// ── Generic helper: feature-gated SynthFailure constructor ───────────────────

/// Convenience for stub bodies that haven't been implemented yet.
///
/// Phase 1 replaced the EATH stubs but kept this helper for the calibrate /
/// emit / fidelity-report stubs Phase 2 / 2.5 / 3 still need to land. The
/// `#[allow(dead_code)]` covers the brief window where no callers exist.
#[allow(dead_code)]
pub(crate) fn not_yet_implemented(what: &str) -> TensaError {
    TensaError::SynthFailure(format!("{what}: not yet implemented in this phase"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    /// Verifies that two run UUIDs minted 1ms apart sort chronologically when
    /// encoded as 16-byte BE binary keys. This is the load-bearing property —
    /// if it ever breaks, every "newest first" scan silently corrupts.
    #[test]
    fn test_run_id_keys_sort_chronologically() {
        let earlier = Uuid::now_v7();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let later = Uuid::now_v7();

        let k_earlier = key_synth_run("nid", &earlier);
        let k_later = key_synth_run("nid", &later);
        assert!(
            k_earlier < k_later,
            "earlier run-id key must sort before later (got {:?} vs {:?})",
            k_earlier,
            k_later
        );

        let s_earlier = key_synth_seed(&earlier);
        let s_later = key_synth_seed(&later);
        assert!(s_earlier < s_later);
    }

    #[test]
    fn test_key_builders_produce_disjoint_prefixes() {
        let id = Uuid::now_v7();
        let kp = key_synth_params("n", "eath");
        let kr = key_synth_run("n", &id);
        let ks = key_synth_seed(&id);
        let kl = key_synth_lineage("n", &id);
        let kf = key_synth_fidelity("n", &id);
        let ksig = key_synth_sig("n", "motifs", &id);
        let kdsig = key_synth_dual_sig("n", "motifs", &id);
        let kbist = key_synth_bistability("n", &id);
        let all = [&kp, &kr, &ks, &kl, &kf, &ksig, &kdsig, &kbist];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert!(
                    !a.starts_with(b) && !b.starts_with(a),
                    "key prefixes must be disjoint"
                );
            }
        }
    }

    #[test]
    fn test_record_and_list_lineage_runs() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let nid = "narr";
        let r1 = Uuid::now_v7();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let r2 = Uuid::now_v7();
        record_lineage_run(store.as_ref(), nid, &r1).unwrap();
        record_lineage_run(store.as_ref(), nid, &r2).unwrap();
        let listed = list_lineage_runs(store.as_ref(), nid).unwrap();
        assert_eq!(listed, vec![r1, r2], "lineage should list oldest-first");
    }

    #[test]
    fn test_round_trip_reproducibility_blob() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let run_id = Uuid::now_v7();
        let params = SurrogateParams {
            model: "eath".into(),
            params_json: serde_json::json!({"a_t_distribution": [0.1, 0.2]}),
            seed: 42,
            num_steps: 100,
            label_prefix: "synth".into(),
        };
        let blob = build_reproducibility_blob(run_id, params.clone(), Some("abc123".into()));
        store_reproducibility_blob(store.as_ref(), &blob).unwrap();
        let loaded = load_reproducibility_blob(store.as_ref(), &run_id)
            .unwrap()
            .expect("blob round-trips");
        assert_eq!(loaded.run_id, run_id);
        assert_eq!(loaded.params_full.seed, 42);
        assert_eq!(loaded.source_state_hash.as_deref(), Some("abc123"));
    }

    #[test]
    fn test_list_runs_newest_first_is_reverse_chronological() {
        use crate::synth::types::RunKind;

        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let nid = "narr";

        let mut ids = Vec::new();
        for _ in 0..3 {
            let id = Uuid::now_v7();
            ids.push(id);
            let summary = SurrogateRunSummary {
                run_id: id,
                model: "eath".into(),
                params_hash: "deadbeef".into(),
                source_narrative_id: Some(nid.into()),
                source_state_hash: None,
                output_narrative_id: format!("synth-{}", id),
                num_entities: 0,
                num_situations: 0,
                num_participations: 0,
                started_at: Utc::now(),
                finished_at: Utc::now(),
                duration_ms: 0,
                kind: RunKind::Generation,
            };
            let key = key_synth_run(nid, &id);
            store
                .put(&key, &serde_json::to_vec(&summary).unwrap())
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let listed = list_runs_newest_first(store.as_ref(), nid, 10).unwrap();
        let listed_ids: Vec<Uuid> = listed.iter().map(|s| s.run_id).collect();
        let mut expected = ids.clone();
        expected.reverse();
        assert_eq!(listed_ids, expected);

        let limited = list_runs_newest_first(store.as_ref(), nid, 2).unwrap();
        assert_eq!(limited.len(), 2);
        assert_eq!(limited[0].run_id, ids[2]);
    }
}
