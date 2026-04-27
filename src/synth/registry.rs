//! Surrogate-model registry — keyed by the model's `name()` string so the
//! TensaQL grammar and REST API can refer to models by name without the
//! Rust type-system needing to know about every plug-in.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, TensaError};

use super::eath::EathSurrogate;
use super::nudhy_surrogate::NudhySurrogate;
use super::surrogate::SurrogateModel;

/// Process-wide registry of available surrogate models.
///
/// Construct one via [`SurrogateRegistry::default`] (which auto-registers
/// the built-in EATH model) and stash it in `AppState` / `EmbeddedBackend`.
/// Custom registries can `register` additional models in tests or for
/// experimental third-party plug-ins.
pub struct SurrogateRegistry {
    models: HashMap<&'static str, Arc<dyn SurrogateModel>>,
}

impl SurrogateRegistry {
    /// Empty registry — caller must `register` at least one model.
    pub fn empty() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register a surrogate model under its `name()`. Replaces any previous
    /// entry with the same name (last-registered wins) — this is intentional
    /// so tests can inject mocks without renaming.
    pub fn register(&mut self, model: Arc<dyn SurrogateModel>) {
        self.models.insert(model.name(), model);
    }

    /// Look a model up by name. Errors with `SynthFailure` when unknown so
    /// the caller can surface "did you mean X?" suggestions if they want.
    pub fn get(&self, model: &str) -> Result<Arc<dyn SurrogateModel>> {
        self.models.get(model).cloned().ok_or_else(|| {
            TensaError::SynthFailure(format!(
                "unknown surrogate model '{model}' (registered: {})",
                self.list().join(", ")
            ))
        })
    }

    /// Names of every registered model, sorted for stable output.
    pub fn list(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.models.keys().copied().collect();
        names.sort_unstable();
        names
    }

    /// Number of registered models — handy for dashboards / health checks.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry has any models registered.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl Default for SurrogateRegistry {
    /// The default registry has EATH + NuDHy pre-registered. Future built-in
    /// models (HAD, bistable contagion, etc.) get added here.
    fn default() -> Self {
        let mut r = Self::empty();
        r.register(Arc::new(EathSurrogate));
        r.register(Arc::new(NudhySurrogate));
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_registry_includes_eath_and_nudhy() {
        let r = SurrogateRegistry::default();
        let names = r.list();
        assert!(names.contains(&"eath"), "default registry must register EATH");
        assert!(names.contains(&"nudhy"), "default registry must register NuDHy (Phase 13b)");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_get_unknown_model_errors() {
        let r = SurrogateRegistry::default();
        // `Arc<dyn SurrogateModel>` doesn't implement Debug, so unwrap_err()
        // can't print it. Match instead.
        let e = match r.get("nonexistent") {
            Ok(_) => panic!("expected error for unknown model"),
            Err(e) => e,
        };
        let msg = format!("{e}");
        assert!(msg.contains("unknown surrogate model"));
        assert!(msg.contains("eath"), "error should list registered models");
    }

    #[test]
    fn test_get_returns_eath_by_name() {
        let r = SurrogateRegistry::default();
        let eath = match r.get("eath") {
            Ok(m) => m,
            Err(_) => panic!("eath should be registered by default"),
        };
        assert_eq!(eath.name(), "eath");
    }

    /// Phase 13b T7: registry lookup returns the correct NuDHy impl.
    #[test]
    fn test_nudhy_registry_lookup_returns_correct_impl() {
        let r = SurrogateRegistry::default();
        assert!(r.list().contains(&"nudhy"));
        assert_eq!(r.list().len(), 2); // eath + nudhy
        let m = match r.get("nudhy") {
            Ok(m) => m,
            Err(_) => panic!("nudhy must be registered"),
        };
        assert_eq!(m.name(), "nudhy");
        assert_eq!(m.version(), "v1.0");
        // fidelity_metrics returns the 3 design-doc names.
        let metrics = m.fidelity_metrics();
        assert_eq!(metrics.len(), 3);
        assert!(metrics.contains(&"degree_sequence_preservation"));
        assert!(metrics.contains(&"edge_size_sequence_preservation"));
        assert!(metrics.contains(&"entity_pair_overlap_divergence"));
    }
}
