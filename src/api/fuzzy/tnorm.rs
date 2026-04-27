//! `GET /fuzzy/tnorms` handlers — list registered t-norm families and
//! look up individual entries by name. Read-only; backed by the process-
//! global [`crate::fuzzy::registry::TNormRegistry::default`].

use axum::extract::Path;
use axum::response::IntoResponse;
use serde::Serialize;

use crate::api::routes::{error_response, json_ok};
use crate::fuzzy::registry::TNormRegistry;
use crate::fuzzy::tnorm::TNormKind;

/// Read-only description of a registered t-norm. The formula field is a
/// plain-ASCII rendering so clients can surface it without re-rendering
/// LaTeX or Unicode math.
#[derive(Debug, Serialize)]
pub struct TNormInfo {
    /// Registry key — `"godel"` / `"goguen"` / `"lukasiewicz"` / `"hamacher"`.
    pub name: &'static str,
    /// One-line English-language description of the family.
    pub description: &'static str,
    /// Canonical formula `T(a, b) = ...` in plain ASCII.
    pub formula: &'static str,
    /// Dual t-conorm formula `S(a, b) = ...`.
    pub tconorm_formula: &'static str,
    /// BibKey of the defining reference — matches `docs/refs.bib`.
    pub citation: &'static str,
}

fn describe(name: &str) -> Option<TNormInfo> {
    match name {
        "godel" => Some(TNormInfo {
            name: "godel",
            description: "Gödel (minimum) t-norm — the maximal t-norm.",
            formula: "T(a, b) = min(a, b)",
            tconorm_formula: "S(a, b) = max(a, b)",
            citation: "klement2000",
        }),
        "goguen" => Some(TNormInfo {
            name: "goguen",
            description: "Goguen (product) t-norm — matches probabilistic AND.",
            formula: "T(a, b) = a * b",
            tconorm_formula: "S(a, b) = a + b - a*b",
            citation: "klement2000",
        }),
        "lukasiewicz" => Some(TNormInfo {
            name: "lukasiewicz",
            description: "Łukasiewicz (bounded-difference) t-norm.",
            formula: "T(a, b) = max(0, a + b - 1)",
            tconorm_formula: "S(a, b) = min(1, a + b)",
            citation: "klement2000",
        }),
        "hamacher" => Some(TNormInfo {
            name: "hamacher",
            description:
                "Hamacher family parameterised by λ ≥ 0. λ = 1 ≡ Goguen; λ = 0 = Hamacher product.",
            formula: "T(a, b) = ab / (λ + (1 - λ)(a + b - ab))",
            tconorm_formula: "S(a, b) = 1 - T(1 - a, 1 - b)",
            citation: "klement2000",
        }),
        _ => None,
    }
}

/// GET /fuzzy/tnorms — enumerate registered t-norm families.
pub async fn list_tnorms() -> impl IntoResponse {
    let reg = TNormRegistry::default();
    let mut names = reg.list();
    names.sort();
    let infos: Vec<TNormInfo> = names.iter().filter_map(|n| describe(n)).collect();
    json_ok(&serde_json::json!({"tnorms": infos}))
}

/// GET /fuzzy/tnorms/{kind} — single-entry lookup.
pub async fn get_tnorm(Path(kind): Path<String>) -> impl IntoResponse {
    let reg = TNormRegistry::default();
    let resolved = match reg.get(&kind) {
        Ok(k) => k,
        Err(e) => return error_response(e).into_response(),
    };
    // Map resolved kind back to the canonical name in case the caller used
    // a stale alias — today every registry name round-trips, but keep the
    // indirection so future aliasing stays correct.
    let canonical = match resolved {
        TNormKind::Godel => "godel",
        TNormKind::Goguen => "goguen",
        TNormKind::Lukasiewicz => "lukasiewicz",
        TNormKind::Hamacher(_) => "hamacher",
    };
    match describe(canonical) {
        Some(info) => json_ok(&info),
        None => error_response(crate::error::TensaError::InvalidInput(format!(
            "no descriptor for t-norm '{canonical}'"
        )))
        .into_response(),
    }
}
