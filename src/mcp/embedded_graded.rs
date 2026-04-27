//! Graded Acceptability Sprint Phase 5 — embedded MCP impls for the five
//! graded-surface tools shipped on top of Phases 3 + 4 of the sprint.
//!
//! Five tools:
//! 1. `argumentation_gradual`         — POST /analysis/argumentation/gradual
//! 2. `fuzzy_learn_measure`           — POST /fuzzy/measures/learn
//! 3. `fuzzy_get_measure_version`     — GET  /fuzzy/measures/{name}?version=N
//! 4. `fuzzy_list_measure_versions`   — GET  /fuzzy/measures/{name}/versions
//! 5. `temporal_ordhorn_closure`      — POST /temporal/ordhorn/closure
//!
//! Each `*_impl` mirrors the corresponding REST handler contract so the
//! MCP path stays bit-identical to the REST path. Helpers from
//! [`crate::api::fuzzy::measure`] (`measure_key`, `versioned_measure_key`,
//! `StoredMeasure`) are reused as-is so the wire format never drifts.
//!
//! Cites: [amgoud2013ranking] [besnard2001hcategoriser] [amgoud2017weighted]
//!        [grabisch1996choquet] [bustince2016choquet] [nebel1995ordhorn].

use serde_json::Value;

use crate::analysis::argumentation::run_argumentation_with_gradual;
use crate::analysis::argumentation_gradual::GradualSemanticsKind;
use crate::api::fuzzy::measure::{
    measure_key, versioned_measure_key, StoredMeasure,
};
use crate::api::fuzzy::FUZZY_MEASURE_PREFIX;
use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation_learn::{
    learn_choquet_measure, LearnedChoquetMeasureOutput, MeasureProvenance,
};
use crate::fuzzy::tnorm::TNormKind;
use crate::temporal::ordhorn::{closure as ordhorn_closure, OrdHornNetwork};

use super::embedded::EmbeddedBackend;

impl EmbeddedBackend {
    // ─── Tool 1: argumentation_gradual ───────────────────────────

    /// Run gradual / ranking-based argumentation synchronously. Mirrors
    /// `POST /analysis/argumentation/gradual`. Returns
    /// `{ narrative_id, gradual: {acceptability, iterations, converged},
    /// iterations, converged }`.
    ///
    /// `gradual_semantics` is parsed opaquely from JSON because
    /// [`GradualSemanticsKind`] does not derive `JsonSchema`. `tnorm` is
    /// likewise parsed opaquely; default `None` reproduces canonical
    /// Gödel formulas bit-identically.
    pub(crate) async fn argumentation_gradual_impl(
        &self,
        narrative_id: &str,
        gradual_semantics: Value,
        tnorm: Option<Value>,
    ) -> Result<Value> {
        let nid = narrative_id.trim();
        if nid.is_empty() {
            return Err(TensaError::InvalidInput("narrative_id is empty".into()));
        }
        let kind: GradualSemanticsKind = serde_json::from_value(gradual_semantics)
            .map_err(|e| {
                TensaError::InvalidInput(format!(
                    "invalid gradual_semantics payload: {e}"
                ))
            })?;
        let tnorm_kind: Option<TNormKind> = match tnorm {
            None => None,
            Some(v) if v.is_null() => None,
            Some(v) => Some(serde_json::from_value(v).map_err(|e| {
                TensaError::InvalidInput(format!("invalid tnorm payload: {e}"))
            })?),
        };
        let result =
            run_argumentation_with_gradual(self.hypergraph(), nid, Some(kind), tnorm_kind)?;
        let gradual = result.gradual.ok_or_else(|| {
            TensaError::Internal(
                "argumentation engine returned no gradual result for a gradual request"
                    .into(),
            )
        })?;
        Ok(serde_json::json!({
            "narrative_id": nid,
            "iterations": gradual.iterations,
            "converged": gradual.converged,
            "gradual": gradual,
        }))
    }

    // ─── Tool 2: fuzzy_learn_measure ─────────────────────────────

    /// Fit + persist a Choquet measure from a `(input_vec, rank)`
    /// dataset. Mirrors `POST /fuzzy/measures/learn`. Returns the
    /// `{ name, version, n, provenance, train_auc, test_auc }` summary.
    ///
    /// Re-training under an existing name auto-increments the version;
    /// both the versionless latest pointer and the versioned history
    /// slice are written.
    pub(crate) async fn fuzzy_learn_measure_impl(
        &self,
        name: &str,
        n: u8,
        dataset: Vec<(Vec<f64>, u32)>,
        dataset_id: &str,
    ) -> Result<Value> {
        let name = name.trim();
        if name.is_empty() {
            return Err(TensaError::InvalidInput(
                "measure name must be non-empty".into(),
            ));
        }
        if name.contains('/') || name.contains('\n') || name.contains(' ') {
            return Err(TensaError::InvalidInput(format!(
                "measure name '{name}' contains invalid characters ('/', whitespace, or newlines)"
            )));
        }

        let LearnedChoquetMeasureOutput {
            mut measure,
            provenance,
            train_auc,
            test_auc,
        } = learn_choquet_measure(n, &dataset, dataset_id)?;

        let store = self.hypergraph().store();
        let prev_version: Option<u32> = match store.get(&measure_key(name))? {
            Some(bytes) => match serde_json::from_slice::<StoredMeasure>(&bytes) {
                Ok(prev) => Some(prev.version),
                Err(e) => return Err(TensaError::Serialization(e.to_string())),
            },
            None => None,
        };
        let next_version = prev_version.map(|v| v.saturating_add(1)).unwrap_or(1);

        // Stamp the inline measure so downstream callers see provenance
        // even when only the FuzzyMeasure escapes the StoredMeasure
        // envelope.
        measure.measure_id = Some(name.to_string());
        measure.measure_version = Some(next_version);

        let stored = StoredMeasure {
            name: name.to_string(),
            measure,
            version: next_version,
            provenance: MeasureProvenance::Learned(provenance.clone()),
        };
        let bytes = serde_json::to_vec(&stored)
            .map_err(|e| TensaError::Serialization(e.to_string()))?;

        // Write versionless (latest pointer) AND versioned (history) keys
        // — same dual-write contract enforced by the REST handler.
        store.put(&measure_key(name), &bytes)?;
        store.put(&versioned_measure_key(name, next_version), &bytes)?;

        Ok(serde_json::json!({
            "name": name,
            "version": next_version,
            "n": n,
            "provenance": provenance,
            "train_auc": train_auc,
            "test_auc": test_auc,
        }))
    }

    // ─── Tool 3: fuzzy_get_measure_version ───────────────────────

    /// Version-aware fetch. Mirrors `GET /fuzzy/measures/{name}` with
    /// optional `?version=N`. Absent `version` returns the latest
    /// pointer (legacy behaviour); present but missing → `Internal`
    /// error so MCP callers see a clear message rather than a silent
    /// `null`.
    pub(crate) async fn fuzzy_get_measure_version_impl(
        &self,
        name: &str,
        version: Option<u32>,
    ) -> Result<Value> {
        let key = match version {
            Some(v) => versioned_measure_key(name, v),
            None => measure_key(name),
        };
        match self.hypergraph().store().get(&key)? {
            Some(bytes) => {
                let stored: StoredMeasure = serde_json::from_slice(&bytes)
                    .map_err(|e| TensaError::Serialization(e.to_string()))?;
                serde_json::to_value(&stored)
                    .map_err(|e| TensaError::Serialization(e.to_string()))
            }
            None => Err(TensaError::Internal(match version {
                Some(v) => format!("measure '{name}' version {v} not found"),
                None => format!("no measure named '{name}'"),
            })),
        }
    }

    // ─── Tool 4: fuzzy_list_measure_versions ─────────────────────

    /// Enumerate every persisted version of a named measure. Mirrors
    /// `GET /fuzzy/measures/{name}/versions`. Returns
    /// `{ name, versions: [u32] }` sorted ascending; missing-name case
    /// yields an empty list (same behaviour as the REST handler so
    /// callers can poll without a 404 path).
    pub(crate) async fn fuzzy_list_measure_versions_impl(
        &self,
        name: &str,
    ) -> Result<Value> {
        let mut prefix = Vec::with_capacity(
            FUZZY_MEASURE_PREFIX.len() + name.len() + b"/v".len(),
        );
        prefix.extend_from_slice(FUZZY_MEASURE_PREFIX);
        prefix.extend_from_slice(name.as_bytes());
        prefix.extend_from_slice(b"/v");

        let pairs = self.hypergraph().store().prefix_scan(&prefix)?;
        let mut versions: Vec<u32> = pairs
            .iter()
            .filter_map(|(k, _)| parse_version_from_key(k, &prefix))
            .collect();
        versions.sort_unstable();
        Ok(serde_json::json!({"name": name, "versions": versions}))
    }

    // ─── Tool 5: temporal_ordhorn_closure ────────────────────────

    /// Run path-consistency closure on an Allen interval-algebra
    /// network. Mirrors `POST /temporal/ordhorn/closure`. Pure
    /// transformation — never touches the hypergraph store.
    ///
    /// Returns `{ closed_network, satisfiable }`. `satisfiable` is
    /// derived from the closure output (no second pass through the
    /// algorithm) — `false` iff at least one constraint cell is empty.
    pub(crate) async fn temporal_ordhorn_closure_impl(
        &self,
        network: Value,
    ) -> Result<Value> {
        let net: OrdHornNetwork = serde_json::from_value(network).map_err(|e| {
            TensaError::InvalidInput(format!("invalid OrdHornNetwork payload: {e}"))
        })?;
        if net.n == 0 && !net.constraints.is_empty() {
            return Err(TensaError::InvalidInput(
                "network has zero intervals but non-empty constraints".into(),
            ));
        }
        let closed = ordhorn_closure(&net)?;
        let satisfiable = closed.constraints.iter().all(|c| !c.relations.is_empty());
        Ok(serde_json::json!({
            "closed_network": closed,
            "satisfiable": satisfiable,
        }))
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Extract `N` from a key shaped like `fz/tn/measures/{name}/v{N}` where
/// `prefix == fz/tn/measures/{name}/v`. Mirrors
/// `crate::api::fuzzy::measure::parse_version_from_key` (private there).
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

// Tests live in the sibling [`super::embedded_graded_tests`] module to
// keep this file under the 500-line cap.
