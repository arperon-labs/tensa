//! Name → kind registries for t-norms and aggregators.
//!
//! The TensaQL grammar and REST / MCP surfaces accept the operator name
//! as a **string parameter**, never a hardcoded keyword. New families
//! register here without a parser or API migration — callers only need
//! to add a `TNormKind` / `AggregatorKind` variant and wire it in
//! [`crate::fuzzy::tnorm`] / [`crate::fuzzy::aggregation`] respectively,
//! then register the name in [`TNormRegistry::default`] /
//! [`AggregatorRegistry::default`].
//!
//! Cites: [klement2000].

use std::collections::BTreeMap;

use crate::error::{Result, TensaError};
use crate::fuzzy::aggregation::{AggregatorKind, FuzzyMeasure};
use crate::fuzzy::tnorm::TNormKind;

/// Default Hamacher lambda (Phase 0). Phase 1 adds a `get_with_param`
/// helper for callers that need a non-default λ without mutating the
/// registry.
pub const DEFAULT_HAMACHER_LAMBDA: f64 = 0.5;

// ── TNormRegistry ─────────────────────────────────────────────────────────────

/// Name-keyed registry for t-norm kinds.
///
/// Phase 0 ships four built-in families: `godel`, `goguen`, `lukasiewicz`,
/// `hamacher` (with λ = [`DEFAULT_HAMACHER_LAMBDA`]).
pub struct TNormRegistry {
    kinds: BTreeMap<&'static str, TNormKind>,
}

impl TNormRegistry {
    /// Empty registry — caller must populate manually. Most callers should
    /// use [`TNormRegistry::default`] instead.
    pub fn new() -> Self {
        Self {
            kinds: BTreeMap::new(),
        }
    }

    /// Register a named t-norm kind. Last-write-wins if the name already
    /// exists — mirrors the synth registry's contract.
    pub fn register(&mut self, name: &'static str, kind: TNormKind) {
        self.kinds.insert(name, kind);
    }

    /// Look up a registered t-norm kind by name.
    ///
    /// Returns [`TensaError::InvalidInput`] if the name is unknown. Phase 1
    /// and later phases rely on this error path to reject unknown string
    /// parameters coming from TensaQL / REST without panicking.
    pub fn get(&self, name: &str) -> Result<TNormKind> {
        self.kinds.get(name).copied().ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "unknown t-norm kind '{}' — known: {:?}",
                name,
                self.list()
            ))
        })
    }

    /// Sorted list of registered names (for REST /fuzzy/tnorms etc.).
    pub fn list(&self) -> Vec<&'static str> {
        self.kinds.keys().copied().collect()
    }
}

impl Default for TNormRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register("godel", TNormKind::Godel);
        reg.register("goguen", TNormKind::Goguen);
        reg.register("lukasiewicz", TNormKind::Lukasiewicz);
        reg.register("hamacher", TNormKind::Hamacher(DEFAULT_HAMACHER_LAMBDA));
        reg
    }
}

// ── AggregatorRegistry ────────────────────────────────────────────────────────

/// Name-keyed registry for aggregator kinds.
///
/// Phase 0 ships six built-in aggregators: `mean`, `median`, `owa`
/// (empty weights — caller supplies), `choquet` (trivial n=1 measure —
/// caller supplies the real one), `tnorm_reduce` (default Gödel),
/// `tconorm_reduce` (default Gödel).
pub struct AggregatorRegistry {
    kinds: BTreeMap<&'static str, AggregatorKind>,
}

impl AggregatorRegistry {
    /// Empty registry — most callers want [`AggregatorRegistry::default`].
    pub fn new() -> Self {
        Self {
            kinds: BTreeMap::new(),
        }
    }

    /// Register a named aggregator kind. Last-write-wins on collision.
    pub fn register(&mut self, name: &'static str, kind: AggregatorKind) {
        self.kinds.insert(name, kind);
    }

    /// Look up a registered aggregator kind by name, returning a **clone** —
    /// aggregator kinds may carry non-Copy payloads (OWA weights, Choquet
    /// measure). Callers that want to mutate the template should clone
    /// again into their own scope.
    pub fn get(&self, name: &str) -> Result<AggregatorKind> {
        self.kinds.get(name).cloned().ok_or_else(|| {
            TensaError::InvalidInput(format!(
                "unknown aggregator kind '{}' — known: {:?}",
                name,
                self.list()
            ))
        })
    }

    /// Sorted list of registered names.
    pub fn list(&self) -> Vec<&'static str> {
        self.kinds.keys().copied().collect()
    }
}

impl Default for AggregatorRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register("mean", AggregatorKind::Mean);
        reg.register("median", AggregatorKind::Median);
        reg.register("owa", AggregatorKind::Owa(Vec::new()));
        // Trivial 1-element measure — caller supplies the real one before
        // dispatching a Choquet integral computation. Phase 2 wires the
        // full constructor path through the aggregate endpoint.
        let trivial = FuzzyMeasure::trivial(1).expect("n=1 trivial measure is well-formed");
        reg.register("choquet", AggregatorKind::Choquet(trivial));
        reg.register("tnorm_reduce", AggregatorKind::TNormReduce(TNormKind::Godel));
        reg.register(
            "tconorm_reduce",
            AggregatorKind::TConormReduce(TNormKind::Godel),
        );
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tnorm_registry_get_hamacher_uses_default_lambda() {
        let reg = TNormRegistry::default();
        let got = reg.get("hamacher").unwrap();
        if let TNormKind::Hamacher(l) = got {
            assert!((l - DEFAULT_HAMACHER_LAMBDA).abs() < f64::EPSILON);
        } else {
            panic!("expected Hamacher variant");
        }
    }

    #[test]
    fn test_tnorm_registry_register_replaces() {
        let mut reg = TNormRegistry::new();
        reg.register("foo", TNormKind::Godel);
        reg.register("foo", TNormKind::Lukasiewicz);
        assert!(matches!(reg.get("foo").unwrap(), TNormKind::Lukasiewicz));
    }

    #[test]
    fn test_aggregator_registry_owa_default_is_empty() {
        // Default registry's OWA entry has no weights — callers must
        // supply them. Aggregators that validate input (Phase 2) reject
        // mismatched lengths loudly.
        let reg = AggregatorRegistry::default();
        match reg.get("owa").unwrap() {
            AggregatorKind::Owa(w) => assert!(w.is_empty()),
            other => panic!("expected Owa, got {:?}", other),
        }
    }

    #[test]
    fn test_aggregator_registry_choquet_default_is_trivial() {
        let reg = AggregatorRegistry::default();
        match reg.get("choquet").unwrap() {
            AggregatorKind::Choquet(m) => {
                assert_eq!(m.n, 1);
                assert_eq!(m.values.len(), 2);
            }
            other => panic!("expected Choquet, got {:?}", other),
        }
    }
}
