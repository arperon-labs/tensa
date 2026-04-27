//! REST API module for the fuzzy-logic subsystem (Phase 4).
//!
//! Mirrors [`crate::api::synth`] / [`crate::api::analysis`] layout: one
//! sub-module per endpoint family, shared helpers live here.
//!
//! ```text
//! GET    /fuzzy/tnorms                        → tnorm::list_tnorms
//! GET    /fuzzy/tnorms/{kind}                 → tnorm::get_tnorm
//! GET    /fuzzy/aggregators                   → aggregator::list_aggregators
//! GET    /fuzzy/aggregators/{kind}            → aggregator::get_aggregator
//! POST   /fuzzy/measures                      → measure::create_measure
//! GET    /fuzzy/measures                      → measure::list_measures
//! POST   /fuzzy/measures/learn                → learn::learn_measure_handler
//! GET    /fuzzy/measures/{name}/versions      → measure::list_versions       (Phase 3)
//! GET    /fuzzy/measures/{name}[?version=N]   → measure::get_measure         (Phase 3)
//! DELETE /fuzzy/measures/{name}[?version=N]   → measure::delete_measure      (Phase 3)
//! GET    /fuzzy/config                        → config::get_config
//! PUT    /fuzzy/config                        → config::put_config
//! POST   /fuzzy/aggregate                     → aggregate::aggregate
//! ```
//!
//! ## Persistence
//!
//! Phase 4 introduces **one** new KV prefix slot: fuzzy measures live under
//! `fz/tn/measures/{name}` (namespaced under the existing `fz/tn/` prefix
//! so the `fuzzy/` key-builder family stays canonical). `cfg/fuzzy` holds
//! the workspace default `{ tnorm, aggregator }` config.
//!
//! ## Per-endpoint opt-in
//!
//! [`parse_fuzzy_config`] consumes `?tnorm=<kind>&aggregator=<kind>` query
//! parameters on any endpoint wired to respect the override. When both
//! params are absent the function returns `Ok(None)` and the caller
//! preserves its pre-sprint behaviour bit-identically (this is the
//! backward-compat contract asserted by Phase 1 regressions).
//!
//! Unknown kinds surface as [`TensaError::InvalidInput`] so endpoints map
//! them to HTTP 400 via [`crate::api::routes::error_response`].
//!
//! Cites: [klement2000] [yager1988owa] [grabisch1996choquet].

pub mod aggregate;
pub mod aggregator;
pub mod config;
pub mod fca;
pub mod hybrid;
pub mod learn;
pub mod measure;
pub mod quantify;
pub mod rules;
pub mod syllogism;
pub mod tnorm;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::AggregatorKind;
use crate::fuzzy::registry::{AggregatorRegistry, TNormRegistry};
use crate::fuzzy::tnorm::TNormKind;

// ── Shared constants ─────────────────────────────────────────────────────────

/// Default t-norm name when no config has been persisted. Matches the
/// Phase 1 "site-default for best-match confidence aggregation" wiring.
pub const DEFAULT_TNORM: &str = "godel";

/// Default aggregator name. Matches the arithmetic-mean site default in
/// the Phase 1 confidence-breakdown composite path.
pub const DEFAULT_AGGREGATOR: &str = "mean";

/// KV prefix for persisted fuzzy measures (Phase 4). Lives under the
/// `fz/tn/` slice so the fuzzy prefix family stays contiguous — see
/// [`crate::fuzzy::FUZZY_TN_PREFIX`].
pub const FUZZY_MEASURE_PREFIX: &[u8] = b"fz/tn/measures/";

/// KV key for the per-workspace default fuzzy config (t-norm + aggregator).
pub const CFG_FUZZY_KEY: &[u8] = b"cfg/fuzzy";

/// Current format version for `cfg/fuzzy` persistence. Bumped when the
/// stored shape changes so old blobs can still load.
pub const FUZZY_CONFIG_VERSION: u32 = 1;

// ── Persistent workspace config ──────────────────────────────────────────────

/// Workspace-wide default t-norm + aggregator selection. Persists at
/// `cfg/fuzzy`. Every field is `#[serde(default)]` so older blobs load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyWorkspaceConfig {
    /// Registry name (`"godel"` / `"goguen"` / `"lukasiewicz"` / `"hamacher"`).
    pub tnorm: String,
    /// Registry name (`"mean"` / `"median"` / ...).
    pub aggregator: String,
    /// Optional named measure that CHOQUET defaults to when no measure is
    /// supplied on a request. Resolved at use time, not on put.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure: Option<String>,
    /// Format version — lets future edits migrate existing blobs.
    #[serde(default = "default_cfg_version")]
    pub version: u32,
}

fn default_cfg_version() -> u32 {
    FUZZY_CONFIG_VERSION
}

impl Default for FuzzyWorkspaceConfig {
    fn default() -> Self {
        Self {
            tnorm: DEFAULT_TNORM.to_string(),
            aggregator: DEFAULT_AGGREGATOR.to_string(),
            measure: None,
            version: FUZZY_CONFIG_VERSION,
        }
    }
}

/// Load the persisted fuzzy config from the workspace KV store, if any.
/// Corrupt or absent records fall back to the Godel/Mean default so the
/// API never 500's on a malformed blob.
pub fn load_workspace_config(store: &dyn crate::store::KVStore) -> FuzzyWorkspaceConfig {
    match store.get(CFG_FUZZY_KEY) {
        Ok(Some(bytes)) => match serde_json::from_slice::<FuzzyWorkspaceConfig>(&bytes) {
            Ok(cfg) => cfg,
            Err(e) => {
                tracing::warn!(
                    "cfg/fuzzy deserialize failed ({e}); falling back to Godel/Mean default"
                );
                FuzzyWorkspaceConfig::default()
            }
        },
        _ => FuzzyWorkspaceConfig::default(),
    }
}

/// Persist a fuzzy config to the workspace KV store. Validates every field
/// against the Phase 0 registries before writing.
pub fn save_workspace_config(
    store: &dyn crate::store::KVStore,
    cfg: &FuzzyWorkspaceConfig,
) -> Result<()> {
    // Validate via the same registries the request-time path uses —
    // unknown kinds get a clean 400 instead of the caller discovering the
    // problem the first time they `GET`.
    let _ = TNormRegistry::default().get(&cfg.tnorm)?;
    let _ = AggregatorRegistry::default().get(&cfg.aggregator)?;
    let bytes = serde_json::to_vec(cfg).map_err(|e| TensaError::Serialization(e.to_string()))?;
    store.put(CFG_FUZZY_KEY, &bytes)?;
    Ok(())
}

// ── Per-endpoint opt-in query-string parser ─────────────────────────────────

/// Request-level override of the site default. Carries the resolved
/// `TNormKind` / `AggregatorKind` so downstream services don't have to
/// touch the registry a second time.
///
/// `None` on either field = "use site default" (Phase 1/2 wiring).
///
/// Phase 0 of the Graded Acceptability sprint added the `measure_id` +
/// `measure_version` slots so every aggregation envelope can echo a
/// learned-measure reference back to the caller. Both default to `None`
/// — symmetric defaults remain bit-identical on the wire.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FuzzyConfig {
    /// Echoed t-norm name (canonical registry key). Present iff the
    /// request supplied `?tnorm=<kind>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tnorm: Option<String>,
    /// Echoed aggregator name. Present iff `?aggregator=<kind>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregator: Option<String>,
    /// Echoed name of a learned Choquet measure. Phase 2 populates this
    /// when an aggregation routes through a learned measure; symmetric
    /// defaults leave it `None` so envelopes stay bit-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure_id: Option<String>,
    /// Echoed version stamp of the learned measure. Pairs with
    /// `measure_id` — `None` iff `measure_id` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure_version: Option<u32>,
    /// Resolved t-norm kind (not serialised to wire — deliberate; the
    /// caller cares about the name, the engine cares about the kind).
    #[serde(skip)]
    pub tnorm_kind: Option<TNormKind>,
    /// Resolved aggregator kind (not serialised to wire).
    #[serde(skip)]
    pub aggregator_kind: Option<AggregatorKind>,
}

impl FuzzyConfig {
    /// `true` when no override was supplied (no t-norm, no aggregator,
    /// no learned-measure reference).
    pub fn is_empty(&self) -> bool {
        self.tnorm.is_none()
            && self.aggregator.is_none()
            && self.measure_id.is_none()
            && self.measure_version.is_none()
    }
}

/// Parse `?tnorm=<kind>&aggregator=<kind>` from a flat query-string map.
///
/// * Returns `Ok(None)` when both keys are absent (preserves backward-compat —
///   the caller takes the pre-sprint no-op path bit-identically).
/// * Returns `Ok(Some(cfg))` when at least one key is present.
/// * Returns `Err(TensaError::InvalidInput)` when a key resolves to an
///   unknown kind — the handler maps this to HTTP 400.
pub fn parse_fuzzy_config(params: &HashMap<String, String>) -> Result<Option<FuzzyConfig>> {
    let tnorm_raw = params.get("tnorm").map(|s| s.trim().to_string());
    let agg_raw = params.get("aggregator").map(|s| s.trim().to_string());
    let measure_raw = params.get("measure").map(|s| s.trim().to_string());
    let measure_version_raw = params
        .get("measure_version")
        .map(|s| s.trim().to_string());

    if tnorm_raw.is_none()
        && agg_raw.is_none()
        && measure_raw.is_none()
        && measure_version_raw.is_none()
    {
        return Ok(None);
    }

    tracing::debug!(
        target: "tensa::fuzzy",
        "loading fuzzy config for request (tnorm={:?}, aggregator={:?}, measure={:?}, measure_version={:?})",
        tnorm_raw,
        agg_raw,
        measure_raw,
        measure_version_raw
    );

    let mut cfg = FuzzyConfig::default();

    if let Some(name) = tnorm_raw {
        let kind = TNormRegistry::default().get(&name)?;
        cfg.tnorm = Some(name);
        cfg.tnorm_kind = Some(kind);
    }

    if let Some(name) = agg_raw {
        let kind = AggregatorRegistry::default().get(&name)?;
        cfg.aggregator = Some(name);
        cfg.aggregator_kind = Some(kind);
    }

    if let Some(name) = measure_raw {
        if !name.is_empty() {
            cfg.measure_id = Some(name);
        }
    }

    if let Some(raw) = measure_version_raw {
        if !raw.is_empty() {
            let parsed = raw.parse::<u32>().map_err(|e| {
                TensaError::InvalidInput(format!(
                    "invalid measure_version: {raw} ({e})"
                ))
            })?;
            cfg.measure_version = Some(parsed);
        }
    }

    Ok(Some(cfg))
}

/// Convenience wrapper that exposes [`parse_fuzzy_config`] as a
/// `Option<FuzzyConfig>`; unknown kinds surface as `Err`. Kept alongside
/// the core parser so call sites can inline `parse_fuzzy_config_opt(&q)?`
/// rather than the boolean-shaped helper they would otherwise write.
pub fn parse_fuzzy_config_opt(params: &HashMap<String, String>) -> Result<Option<FuzzyConfig>> {
    parse_fuzzy_config(params)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "../fuzzy_tests.rs"]
mod fuzzy_tests;

/// Phase 3 of the Graded Acceptability sprint — version-aware measure
/// CRUD tests + synchronous gradual-argumentation endpoint tests.
#[cfg(test)]
#[path = "learn_tests.rs"]
mod learn_tests;
