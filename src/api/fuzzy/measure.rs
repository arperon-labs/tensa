//! Fuzzy-measure CRUD handlers. This is the only Phase 4 surface that
//! writes persistent state (under [`super::FUZZY_MEASURE_PREFIX`]).
//!
//! A fuzzy measure is a monotone set function `μ : 2^N → [0, 1]` with
//! `μ(∅) = 0` and `μ(N) = 1`. The constructor [`crate::fuzzy::
//! aggregation_measure::new_monotone`] enforces the monotonicity
//! invariant; non-monotone measures get rejected with a `400` whose
//! body mentions the word "monotonicity" (asserted by the Phase 4 test
//! suite).
//!
//! ## Versioning (Graded Acceptability Sprint Phase 2 + Phase 3)
//!
//! Each measure name has:
//! * a versionless **latest pointer** at `fz/tn/measures/{name}` —
//!   updated on every write so unversioned reads always return the most
//!   recent fit (this is the legacy CRUD behaviour, preserved
//!   bit-identically).
//! * one or more **history slices** at `fz/tn/measures/{name}/v{N}` —
//!   written by [`super::learn::learn_measure_handler`] so paper-figure
//!   callers can replay a specific fit.
//!
//! Phase 3 adds:
//! * `GET /fuzzy/measures/{name}/versions` → list all known versions.
//! * `GET /fuzzy/measures/{name}?version=N` → fetch a specific version.
//! * `DELETE /fuzzy/measures/{name}?version=N` → delete just the
//!   versioned slice (leaves the latest pointer alone).
//!
//! When `?version=N` is omitted, the handlers preserve the pre-Phase-3
//! behaviour bit-identically (read latest / delete latest +
//! versionless pointer).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::fuzzy::aggregation::FuzzyMeasure;
use crate::fuzzy::aggregation_learn::{default_measure_version, MeasureProvenance};
use crate::fuzzy::aggregation_measure::new_monotone;

/// Persistence envelope — carries the canonical name alongside the
/// measure so list-by-prefix scans don't have to re-derive it.
///
/// Phase 0 of the Graded Acceptability sprint added `version` +
/// `provenance` (both serde-defaulted) so pre-sprint records load
/// unchanged. Phase 2 populates them on `POST /fuzzy/measures/learn`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMeasure {
    pub name: String,
    pub measure: FuzzyMeasure,
    /// Schema version of this stored record. Defaults to `1` on legacy
    /// records that pre-date the Graded sprint.
    #[serde(default = "default_measure_version")]
    pub version: u32,
    /// How the measure came to exist. Defaults to
    /// [`MeasureProvenance::Manual`] on legacy records.
    #[serde(default)]
    pub provenance: MeasureProvenance,
}

/// Build the full KV key for the *latest pointer* of a named measure:
/// `fz/tn/measures/{name}`. Use [`versioned_measure_key`] for a
/// specific historical slice.
pub(crate) fn measure_key(name: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(super::FUZZY_MEASURE_PREFIX.len() + name.len());
    k.extend_from_slice(super::FUZZY_MEASURE_PREFIX);
    k.extend_from_slice(name.as_bytes());
    k
}

/// Build the KV key for a specific version slice of a named measure:
/// `fz/tn/measures/{name}/v{version}`. Single source of truth shared
/// with [`super::learn`] so the format never drifts.
pub(crate) fn versioned_measure_key(name: &str, version: u32) -> Vec<u8> {
    let suffix = format!("/v{version}");
    let mut k = Vec::with_capacity(super::FUZZY_MEASURE_PREFIX.len() + name.len() + suffix.len());
    k.extend_from_slice(super::FUZZY_MEASURE_PREFIX);
    k.extend_from_slice(name.as_bytes());
    k.extend_from_slice(suffix.as_bytes());
    k
}

/// Optional `?version=N` query string accepted by the version-aware
/// GET / DELETE measure handlers.
///
/// Phase 3 keeps the legacy unversioned shape backward-compatible by
/// making `version` optional — when absent, the handlers behave
/// bit-identically to the pre-Phase-3 code path.
#[derive(Debug, Default, Deserialize)]
pub struct MeasureVersionQuery {
    #[serde(default)]
    pub version: Option<u32>,
}

/// Body for `POST /fuzzy/measures`.
#[derive(Debug, Deserialize)]
pub struct CreateMeasureBody {
    pub name: String,
    pub n: u8,
    pub values: Vec<f64>,
}

/// POST /fuzzy/measures — validate + persist. Rejects non-monotone
/// measures with a `400` whose message contains "monotonicity".
pub async fn create_measure(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateMeasureBody>,
) -> impl IntoResponse {
    let name = body.name.trim();
    if name.is_empty() {
        return error_response(TensaError::InvalidInput(
            "measure name must be non-empty".into(),
        ))
        .into_response();
    }
    // Names are used as KV key tails — reject any character that could
    // collide with the prefix-scan boundary or the HTTP route pattern.
    if name.contains('/') || name.contains('\n') || name.contains(' ') {
        return error_response(TensaError::InvalidInput(format!(
            "measure name '{name}' contains invalid characters ('/', whitespace, or newlines)"
        )))
        .into_response();
    }

    // `new_monotone` enforces length + endpoints + [0,1] range AND
    // monotonicity. Every failure path surfaces InvalidInput with a
    // descriptive message — including the literal word "monotonicity"
    // on the monotone-violation path, which Phase 4 tests assert on.
    let measure = match new_monotone(body.n, body.values) {
        Ok(m) => m,
        Err(e) => return error_response(e).into_response(),
    };

    let stored = StoredMeasure {
        name: name.to_string(),
        measure,
        version: default_measure_version(),
        provenance: MeasureProvenance::Manual,
    };
    let bytes = match serde_json::to_vec(&stored) {
        Ok(b) => b,
        Err(e) => return error_response(TensaError::Serialization(e.to_string())).into_response(),
    };
    let key = measure_key(name);
    if let Err(e) = state.hypergraph.store().put(&key, &bytes) {
        return error_response(e).into_response();
    }

    (StatusCode::CREATED, Json(serde_json::to_value(&stored).unwrap_or_default()))
        .into_response()
}

/// GET /fuzzy/measures — list all persisted measures (latest pointer
/// only — version slices live under `{name}/v{N}` keys and are
/// filtered out so the list stays one row per unique name).
pub async fn list_measures(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let store = state.hypergraph.store();
    let pairs = match store.prefix_scan(super::FUZZY_MEASURE_PREFIX) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };
    let measures: Vec<StoredMeasure> = pairs
        .into_iter()
        // Skip versioned-history slices (`{name}/v{N}` — distinguishable
        // by the `/v` infix in the key tail). Latest pointers never
        // contain `/`.
        .filter(|(k, _)| !key_is_versioned_slice(k))
        .filter_map(|(_k, v)| serde_json::from_slice::<StoredMeasure>(&v).ok())
        .collect();
    json_ok(&serde_json::json!({"measures": measures}))
}

/// GET /fuzzy/measures/{name} — version-aware.
///
/// * No query string → returns the latest pointer (legacy behaviour).
/// * `?version=N` → returns the versioned slice; missing → HTTP 404
///   with body `{"error": "measure '{name}' version {N} not found"}`.
pub async fn get_measure(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<MeasureVersionQuery>,
) -> impl IntoResponse {
    let key = match q.version {
        Some(v) => versioned_measure_key(&name, v),
        None => measure_key(&name),
    };
    match state.hypergraph.store().get(&key) {
        Ok(Some(bytes)) => match serde_json::from_slice::<StoredMeasure>(&bytes) {
            Ok(stored) => json_ok(&stored),
            Err(e) => {
                error_response(TensaError::Serialization(e.to_string())).into_response()
            }
        },
        Ok(None) => not_found_response(&name, q.version),
        Err(e) => error_response(e).into_response(),
    }
}

/// DELETE /fuzzy/measures/{name} — version-aware.
///
/// * No query string → deletes the latest pointer; legacy idempotent
///   behaviour preserved.
/// * `?version=N` → deletes only the versioned slice; the latest
///   pointer is left intact. Missing → HTTP 404.
pub async fn delete_measure(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(q): Query<MeasureVersionQuery>,
) -> impl IntoResponse {
    let store = state.hypergraph.store();
    match q.version {
        Some(v) => {
            // Versioned-slice delete: 404 if absent (so callers can
            // distinguish "already gone" from "never existed").
            let key = versioned_measure_key(&name, v);
            match store.get(&key) {
                Ok(Some(_)) => match store.delete(&key) {
                    Ok(()) => StatusCode::NO_CONTENT.into_response(),
                    Err(e) => error_response(e).into_response(),
                },
                Ok(None) => not_found_response(&name, Some(v)),
                Err(e) => error_response(e).into_response(),
            }
        }
        None => {
            // Legacy: idempotent latest-pointer delete.
            let key = measure_key(&name);
            match store.delete(&key) {
                Ok(()) => StatusCode::NO_CONTENT.into_response(),
                Err(e) => error_response(e).into_response(),
            }
        }
    }
}

/// GET /fuzzy/measures/{name}/versions — list all known version numbers
/// for a measure name, sorted ascending. Returns `{"versions": []}`
/// when the name has no versioned slices (e.g. it was created via
/// `POST /fuzzy/measures` instead of `POST /fuzzy/measures/learn`).
pub async fn list_versions(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let store = state.hypergraph.store();
    // Prefix-scan `fz/tn/measures/{name}/v` so we only see the version
    // slices for THIS exact name (not, e.g., `{name}_extra`). The
    // trailing `/v` discriminator is the same suffix written by
    // [`versioned_measure_key`].
    let mut prefix =
        Vec::with_capacity(super::FUZZY_MEASURE_PREFIX.len() + name.len() + "/v".len());
    prefix.extend_from_slice(super::FUZZY_MEASURE_PREFIX);
    prefix.extend_from_slice(name.as_bytes());
    prefix.extend_from_slice(b"/v");

    let pairs = match store.prefix_scan(&prefix) {
        Ok(p) => p,
        Err(e) => return error_response(e).into_response(),
    };

    let mut versions: Vec<u32> = pairs
        .iter()
        .filter_map(|(k, _v)| parse_version_from_key(k, &prefix))
        .collect();
    versions.sort_unstable();
    json_ok(&serde_json::json!({"name": name, "versions": versions}))
}

// ── Internal helpers ────────────────────────────────────────────────

/// `true` when the KV key represents a versioned-history slice
/// (`{name}/v{N}`) rather than a latest pointer (`{name}`). Used by
/// [`list_measures`] so the list view stays one row per unique name.
fn key_is_versioned_slice(key: &[u8]) -> bool {
    // The latest-pointer keys do not contain `/v` past the prefix;
    // versioned slices always do. Strip the prefix and look for `/v`.
    if !key.starts_with(super::FUZZY_MEASURE_PREFIX) {
        return false;
    }
    let tail = &key[super::FUZZY_MEASURE_PREFIX.len()..];
    // Find the LAST `/v` in the tail — names cannot contain `/` (validated
    // at create / learn time), so any `/v` in the tail is the version
    // discriminator. Followed by digits = versioned slice.
    if let Some(pos) = find_subsequence(tail, b"/v") {
        let after = &tail[pos + 2..];
        !after.is_empty() && after.iter().all(|b| b.is_ascii_digit())
    } else {
        false
    }
}

/// Extract `N` from a key shaped like `fz/tn/measures/{name}/v{N}`,
/// where `prefix == fz/tn/measures/{name}/v`.
fn parse_version_from_key(key: &[u8], prefix: &[u8]) -> Option<u32> {
    if !key.starts_with(prefix) {
        return None;
    }
    let tail = &key[prefix.len()..];
    if tail.is_empty() || !tail.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    std::str::from_utf8(tail).ok().and_then(|s| s.parse().ok())
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn not_found_response(name: &str, version: Option<u32>) -> axum::response::Response {
    let body = match version {
        Some(v) => serde_json::json!({
            "error": format!("measure '{name}' version {v} not found")
        }),
        None => serde_json::json!({
            "error": format!("no measure named '{name}'")
        }),
    };
    (StatusCode::NOT_FOUND, Json(body)).into_response()
}

/// Helper exposed to [`super::aggregate`] so POST /fuzzy/aggregate can
/// resolve a `measure=<name>` reference without duplicating KV wiring.
pub fn load_measure(
    store: &dyn crate::store::KVStore,
    name: &str,
) -> Result<Option<FuzzyMeasure>, TensaError> {
    match store.get(&measure_key(name)) {
        Ok(Some(bytes)) => match serde_json::from_slice::<StoredMeasure>(&bytes) {
            Ok(stored) => Ok(Some(stored.measure)),
            Err(e) => Err(TensaError::Serialization(e.to_string())),
        },
        Ok(None) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Phase-3 sibling of [`load_measure`] that resolves a specific
/// version. Returns `Ok(None)` when the slice is absent — the caller
/// chooses whether that's a 404 or a fallback to the latest pointer.
pub fn load_measure_version(
    store: &dyn crate::store::KVStore,
    name: &str,
    version: u32,
) -> Result<Option<FuzzyMeasure>, TensaError> {
    match store.get(&versioned_measure_key(name, version)) {
        Ok(Some(bytes)) => match serde_json::from_slice::<StoredMeasure>(&bytes) {
            Ok(stored) => Ok(Some(stored.measure)),
            Err(e) => Err(TensaError::Serialization(e.to_string())),
        },
        Ok(None) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 0 backward-compat invariant: pre-sprint `StoredMeasure` JSON
    /// (no `version`, no `provenance` fields) MUST deserialise into a
    /// modern record with `version = 1` and `provenance = Manual`.
    #[test]
    fn legacy_stored_measure_json_loads_with_defaults() {
        let legacy = serde_json::json!({
            "name": "legacy",
            "measure": {
                "n": 2,
                "values": [0.0, 0.5, 0.5, 1.0]
            }
        });
        let bytes = serde_json::to_vec(&legacy).expect("serialise");
        let stored: StoredMeasure =
            serde_json::from_slice(&bytes).expect("legacy JSON must deserialise");
        assert_eq!(stored.name, "legacy");
        assert_eq!(stored.version, 1);
        assert!(matches!(stored.provenance, MeasureProvenance::Manual));
        assert_eq!(stored.measure.n, 2);
    }

    /// Round-trip a fully populated modern record so the new fields
    /// survive serialisation.
    #[test]
    fn modern_stored_measure_round_trips_through_json() {
        let measure = new_monotone(2, vec![0.0, 0.5, 0.5, 1.0]).expect("monotone");
        let original = StoredMeasure {
            name: "modern".into(),
            measure,
            version: 7,
            provenance: MeasureProvenance::Symmetric {
                kind: "additive".into(),
            },
        };
        let bytes = serde_json::to_vec(&original).expect("serialise");
        let back: StoredMeasure = serde_json::from_slice(&bytes).expect("deserialise");
        assert_eq!(back.name, "modern");
        assert_eq!(back.version, 7);
        match back.provenance {
            MeasureProvenance::Symmetric { kind } => assert_eq!(kind, "additive"),
            _ => panic!("provenance variant must round-trip"),
        }
    }

    #[test]
    fn versioned_key_format_is_stable() {
        // Single-source-of-truth shared with [`super::learn`]; if this
        // assertion changes the learn handler MUST change in lockstep
        // (or vice versa).
        let k = versioned_measure_key("alpha", 7);
        assert_eq!(k, b"fz/tn/measures/alpha/v7");
    }

    #[test]
    fn key_is_versioned_slice_distinguishes_latest_from_history() {
        assert!(!key_is_versioned_slice(b"fz/tn/measures/alpha"));
        assert!(key_is_versioned_slice(b"fz/tn/measures/alpha/v1"));
        assert!(key_is_versioned_slice(b"fz/tn/measures/alpha/v42"));
        // `/v` followed by non-digits is not a version slice.
        assert!(!key_is_versioned_slice(b"fz/tn/measures/alpha/vX"));
    }

    #[test]
    fn parse_version_from_key_handles_padding_and_garbage() {
        let prefix = b"fz/tn/measures/alpha/v";
        assert_eq!(
            parse_version_from_key(b"fz/tn/measures/alpha/v3", prefix),
            Some(3)
        );
        assert_eq!(
            parse_version_from_key(b"fz/tn/measures/alpha/v123", prefix),
            Some(123)
        );
        assert_eq!(parse_version_from_key(b"fz/tn/measures/beta/v3", prefix), None);
    }

    /// The `measure_version_query` defaults to `version = None` so an
    /// absent query string takes the legacy code path.
    #[test]
    fn measure_version_query_defaults_to_none() {
        let q = MeasureVersionQuery::default();
        assert!(q.version.is_none());
    }

}
