//! `/synth/models` handler — enumerates registered surrogate models.

use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::routes::json_ok;
use crate::api::server::AppState;

#[derive(serde::Serialize)]
struct ModelDescriptor {
    name: &'static str,
    version: &'static str,
    fidelity_metrics: Vec<&'static str>,
}

/// GET /synth/models
///
/// Lists every registered surrogate model with its name, version, and
/// fidelity-metric catalog. Used by the Studio model picker.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let registry = &state.synth_registry;
    let mut out = Vec::with_capacity(registry.len());
    for name in registry.list() {
        // `get` only fails when the name isn't registered; `list()` only
        // returns names the registry knows. Belt-and-braces: skip on error.
        let model = match registry.get(name) {
            Ok(m) => m,
            Err(_) => continue,
        };
        out.push(ModelDescriptor {
            name: model.name(),
            version: model.version(),
            fidelity_metrics: model.fidelity_metrics(),
        });
    }
    json_ok(&out)
}
