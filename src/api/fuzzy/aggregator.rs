//! `GET /fuzzy/aggregators` handlers — list and describe the six built-in
//! aggregator families. Read-only.

use axum::extract::Path;
use axum::response::IntoResponse;
use serde::Serialize;

use crate::api::routes::{error_response, json_ok};
use crate::fuzzy::registry::AggregatorRegistry;

#[derive(Debug, Serialize)]
pub struct AggregatorInfo {
    pub name: &'static str,
    pub description: &'static str,
    /// One-line mathematical shape, plain ASCII.
    pub formula: &'static str,
    /// Required extras when calling `POST /fuzzy/aggregate`.
    pub required_params: &'static [&'static str],
    pub citation: &'static str,
}

fn describe(name: &str) -> Option<AggregatorInfo> {
    match name {
        "mean" => Some(AggregatorInfo {
            name: "mean",
            description: "Arithmetic mean (xs treated as a uniform sample).",
            formula: "A(xs) = (Σ xs) / n",
            required_params: &[],
            citation: "klement2000",
        }),
        "median" => Some(AggregatorInfo {
            name: "median",
            description: "Sample median — midpoint of the middle value(s).",
            formula: "A(xs) = median(xs)",
            required_params: &[],
            citation: "klement2000",
        }),
        "owa" => Some(AggregatorInfo {
            name: "owa",
            description: "Yager Ordered Weighted Averaging — sort descending, then dot-product.",
            formula: "A(xs) = Σ w_i * x_(i)   (x_(i) is the i-th largest)",
            required_params: &["owa_weights"],
            citation: "yager1988owa",
        }),
        "choquet" => Some(AggregatorInfo {
            name: "choquet",
            description:
                "Choquet integral over a fuzzy measure. Exact path for n ≤ 10; MC otherwise.",
            formula: "A(xs; μ) = Σ (x_(i) - x_(i-1)) * μ(A_i)",
            required_params: &["measure"],
            citation: "grabisch1996choquet",
        }),
        "tnorm_reduce" => Some(AggregatorInfo {
            name: "tnorm_reduce",
            description: "Left-fold under a t-norm (logical conjunction).",
            formula: "A(xs) = T(T(...T(x_1, x_2)..., x_{n-1}), x_n)",
            required_params: &["tnorm"],
            citation: "klement2000",
        }),
        "tconorm_reduce" => Some(AggregatorInfo {
            name: "tconorm_reduce",
            description: "Left-fold under a t-conorm (logical disjunction).",
            formula: "A(xs) = S(S(...S(x_1, x_2)..., x_{n-1}), x_n)",
            required_params: &["tnorm"],
            citation: "klement2000",
        }),
        _ => None,
    }
}

/// GET /fuzzy/aggregators — enumerate registered aggregators.
pub async fn list_aggregators() -> impl IntoResponse {
    let reg = AggregatorRegistry::default();
    let mut names = reg.list();
    names.sort();
    let infos: Vec<AggregatorInfo> = names.iter().filter_map(|n| describe(n)).collect();
    json_ok(&serde_json::json!({"aggregators": infos}))
}

/// GET /fuzzy/aggregators/{kind}
pub async fn get_aggregator(Path(kind): Path<String>) -> impl IntoResponse {
    let reg = AggregatorRegistry::default();
    if reg.get(&kind).is_err() {
        return error_response(crate::error::TensaError::InvalidInput(format!(
            "unknown aggregator kind '{kind}'"
        )))
        .into_response();
    }
    match describe(&kind) {
        Some(info) => json_ok(&info),
        None => error_response(crate::error::TensaError::InvalidInput(format!(
            "no descriptor for aggregator '{kind}'"
        )))
        .into_response(),
    }
}
