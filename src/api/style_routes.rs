//! HTTP handlers for narrative style/fingerprint endpoints.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::analysis::style_profile::{
    compute_radar, compute_style_profile, detect_style_anomalies, fingerprint_similarity,
    NarrativeFingerprint, NarrativeStyleProfile, WeightedSimilarityConfig,
};
use crate::analysis::stylometry::{compute_prose_features, ProseStyleFeatures};
use crate::analysis::stylometry_stats::{
    bootstrap_pair_similarity_ci, detect_style_anomalies_calibrated, DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_ITER,
};
use crate::api::routes::{error_response, json_ok};
use crate::api::server::AppState;
use crate::error::TensaError;
use crate::hypergraph::keys;

// ─── Generic KV Helpers ────────────────────────────────────

fn kv_load<T: DeserializeOwned>(
    state: &AppState,
    prefix: &[u8],
    id: &str,
) -> crate::error::Result<Option<T>> {
    let key = crate::analysis::analysis_key(prefix, &[id]);
    match state.hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).map_err(|e| TensaError::Serialization(e.to_string()))?,
        )),
        None => Ok(None),
    }
}

fn kv_store<T: Serialize>(
    state: &AppState,
    prefix: &[u8],
    id: &str,
    val: &T,
) -> crate::error::Result<()> {
    let key = crate::analysis::analysis_key(prefix, &[id]);
    let value = serde_json::to_vec(val).map_err(|e| TensaError::Serialization(e.to_string()))?;
    state.hypergraph.store().put(&key, &value)
}

/// Collect raw text from situations (fallback when chunks are not available).
fn text_from_situations(situations: &[crate::types::Situation]) -> String {
    situations
        .iter()
        .flat_map(|s| s.raw_content.iter())
        .map(|cb| cb.content.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Reconstruct original text from stored chunks (preferred for honest stylometry).
/// Strips overlap from consecutive chunks to avoid double-counting.
fn text_from_chunks(chunks: &[crate::types::ChunkRecord]) -> String {
    let capacity: usize = chunks.iter().map(|c| c.text.len()).sum();
    let mut raw = String::with_capacity(capacity);
    for (i, chunk) in chunks.iter().enumerate() {
        raw.push_str(chunk.text_without_overlap(i));
    }
    // Collapse runs of 3+ newlines into double newlines (paragraph breaks)
    // Handles bad epub conversions with excessive blank lines
    let mut text = String::with_capacity(raw.len());
    let mut newline_run = 0u32;
    for ch in raw.chars() {
        if ch == '\n' {
            newline_run += 1;
            if newline_run <= 2 {
                text.push(ch);
            }
        } else {
            newline_run = 0;
            text.push(ch);
        }
    }
    text
}

/// Compute and store both profile and fingerprint, returning the fingerprint.
fn compute_and_store(state: &AppState, id: &str) -> crate::error::Result<NarrativeFingerprint> {
    // Prefer original chunk text for prose features (honest stylometry)
    let chunks = state
        .hypergraph
        .list_chunks_by_narrative(id)
        .unwrap_or_default();
    let text = if !chunks.is_empty() {
        text_from_chunks(&chunks)
    } else {
        let situations = state
            .hypergraph
            .list_situations_by_narrative(id)
            .unwrap_or_default();
        text_from_situations(&situations)
    };
    let prose = compute_prose_features(&text);
    let profile = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute_style_profile(id, &state.hypergraph)
    })) {
        Ok(result) => result?,
        Err(panic_info) => {
            let msg = panic_info
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic_info.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!("Style profile panicked for {}: {}", id, msg);
            return Err(crate::error::TensaError::Internal(format!(
                "Style profile computation panicked: {}",
                msg
            )));
        }
    };
    kv_store(state, keys::ANALYSIS_STYLE_PROFILE, id, &profile)?;

    let fp = NarrativeFingerprint {
        narrative_id: id.to_string(),
        computed_at: chrono::Utc::now(),
        prose,
        structure: profile,
    };
    kv_store(state, keys::ANALYSIS_FINGERPRINT, id, &fp)?;
    Ok(fp)
}

// ─── Endpoints ──────────────────────────────────────────────

/// POST /narratives/:id/style — Compute and store style profile.
pub async fn compute_narrative_style(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match compute_and_store(&state, &id) {
        Ok(fp) => json_ok(&fp.structure),
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/style — Get stored style profile (compute on-the-fly if absent).
pub async fn get_narrative_style(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match kv_load::<NarrativeStyleProfile>(&state, keys::ANALYSIS_STYLE_PROFILE, &id) {
        Ok(Some(profile)) => json_ok(&profile),
        Ok(None) => match compute_and_store(&state, &id) {
            Ok(fp) => json_ok(&fp.structure),
            Err(e) => error_response(e).into_response(),
        },
        Err(e) => error_response(e).into_response(),
    }
}

/// GET /narratives/:id/fingerprint — Get combined fingerprint (prose + structure).
pub async fn get_narrative_fingerprint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match kv_load::<NarrativeFingerprint>(&state, keys::ANALYSIS_FINGERPRINT, &id) {
        Ok(Some(fp)) => json_ok(&fp),
        Ok(None) => match compute_and_store(&state, &id) {
            Ok(fp) => json_ok(&fp),
            Err(e) => error_response(e).into_response(),
        },
        Err(e) => error_response(e).into_response(),
    }
}

/// Request body for POST /style/compare.
#[derive(Deserialize)]
pub struct CompareRequest {
    pub narrative_a: String,
    pub narrative_b: String,
}

/// Load or compute a fingerprint.
fn ensure_fingerprint(state: &AppState, id: &str) -> crate::error::Result<NarrativeFingerprint> {
    match kv_load::<NarrativeFingerprint>(state, keys::ANALYSIS_FINGERPRINT, id)? {
        Some(fp) => Ok(fp),
        None => compute_and_store(state, id),
    }
}

/// Query parameters for style comparison.
#[derive(Deserialize, Default)]
pub struct CompareParams {
    /// Return a bootstrap CI instead of a point estimate when `true`.
    #[serde(default)]
    pub ci: Option<bool>,
    /// Significance level for the CI (default: 0.05 → 95% CI).
    #[serde(default)]
    pub alpha: Option<f32>,
    /// Bootstrap iterations (default: 500 — lower than anomaly default to keep latency reasonable).
    #[serde(default)]
    pub n_iter: Option<usize>,
    /// RNG seed for reproducibility (default: 0xC0FFEE).
    #[serde(default)]
    pub seed: Option<u64>,
}

/// POST /style/compare — Compare two narrative fingerprints.
///
/// With `?ci=true`, returns a `StyleSimilarityCi` including percentile confidence
/// intervals on the overall and prose-layer similarity (structural layers are
/// reported as point estimates).
pub async fn compare_styles(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CompareParams>,
    Json(body): Json<CompareRequest>,
) -> impl IntoResponse {
    let fp_a = match ensure_fingerprint(&state, &body.narrative_a) {
        Ok(fp) => fp,
        Err(e) => return error_response(e).into_response(),
    };
    let fp_b = match ensure_fingerprint(&state, &body.narrative_b) {
        Ok(fp) => fp,
        Err(e) => return error_response(e).into_response(),
    };

    if params.ci.unwrap_or(false) {
        let chunks_a = state
            .hypergraph
            .list_chunks_by_narrative(&body.narrative_a)
            .unwrap_or_default();
        let chunks_b = state
            .hypergraph
            .list_chunks_by_narrative(&body.narrative_b)
            .unwrap_or_default();
        let chunk_features_a: Vec<ProseStyleFeatures> = chunks_a
            .iter()
            .map(|c| compute_prose_features(&c.text))
            .collect();
        let chunk_features_b: Vec<ProseStyleFeatures> = chunks_b
            .iter()
            .map(|c| compute_prose_features(&c.text))
            .collect();
        let cfg = WeightedSimilarityConfig::load_or_default(state.hypergraph.store());
        let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
        let n_iter = params.n_iter.unwrap_or(500);
        let seed = params.seed.unwrap_or(0xC0FFEE);
        let ci = bootstrap_pair_similarity_ci(
            &fp_a,
            &fp_b,
            &chunk_features_a,
            &chunk_features_b,
            &cfg,
            alpha,
            n_iter,
            seed,
        );
        return json_ok(&ci);
    }

    let similarity = fingerprint_similarity(&fp_a, &fp_b);
    json_ok(&similarity)
}

/// Query parameters for anomaly detection.
#[derive(Deserialize)]
pub struct AnomalyParams {
    /// Legacy threshold gate (0.7 default). Ignored in calibrated mode.
    pub threshold: Option<f32>,
    /// Detection mode. `"calibrated"` (aka `"pvalue"`, `"bootstrap"`) enables
    /// bootstrap-calibrated p-values. Default (or `"threshold"`) preserves the
    /// legacy similarity-threshold behavior.
    pub mode: Option<String>,
    /// Significance level for calibrated mode (default: 0.05).
    pub alpha: Option<f32>,
    /// Bootstrap iterations for calibrated mode (default: 1000).
    pub n_iter: Option<usize>,
    /// RNG seed for reproducibility (default: 0xC0FFEE).
    pub seed: Option<u64>,
}

/// GET /narratives/:id/style/anomalies — Detect per-chunk style anomalies.
///
/// When stored chunks are available, computes per-chunk prose features from original
/// text (true authorship signal). Falls back to per-situation analysis otherwise.
///
/// With `?mode=calibrated`, returns a `Vec<AnomalyPValue>` with bootstrap-derived
/// empirical p-values instead of a similarity-threshold gate.
pub async fn get_style_anomalies(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<AnomalyParams>,
) -> impl IntoResponse {
    let mode = params.mode.as_deref().unwrap_or("threshold");
    let calibrated = matches!(mode, "calibrated" | "pvalue" | "bootstrap");

    // Prefer per-chunk analysis when chunks are available
    let stored_chunks = state
        .hypergraph
        .list_chunks_by_narrative(&id)
        .unwrap_or_default();

    let (global, per_unit) = if !stored_chunks.is_empty() {
        // Original text chunks — honest authorship analysis
        let per_unit: Vec<ProseStyleFeatures> = stored_chunks
            .iter()
            .map(|c| compute_prose_features(&c.text))
            .collect();
        let all_text = text_from_chunks(&stored_chunks);
        let global = compute_prose_features(&all_text);
        (global, per_unit)
    } else {
        // Fallback: per-situation analysis from LLM-extracted content
        let situations = match state.hypergraph.list_situations_by_narrative(&id) {
            Ok(s) => s,
            Err(e) => return error_response(e).into_response(),
        };
        let per_unit: Vec<ProseStyleFeatures> = situations
            .iter()
            .map(|s| {
                let text: String = s
                    .raw_content
                    .iter()
                    .map(|cb| cb.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                compute_prose_features(&text)
            })
            .collect();
        let all_text = text_from_situations(&situations);
        let global = compute_prose_features(&all_text);
        (global, per_unit)
    };

    if calibrated {
        let cfg = WeightedSimilarityConfig::load_or_default(state.hypergraph.store());
        let alpha = params.alpha.unwrap_or(DEFAULT_ALPHA);
        let n_iter = params.n_iter.unwrap_or(DEFAULT_BOOTSTRAP_ITER);
        let seed = params.seed.unwrap_or(0xC0FFEE);
        let results = detect_style_anomalies_calibrated(&per_unit, alpha, n_iter, &cfg, seed);
        return json_ok(&results);
    }

    let threshold = params.threshold.unwrap_or(0.7);
    let anomalies = detect_style_anomalies(&global, &per_unit, threshold);
    json_ok(&anomalies)
}

/// GET /narratives/:id/style/radar — Get radar chart data.
pub async fn get_style_radar(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let profile = match kv_load::<NarrativeStyleProfile>(&state, keys::ANALYSIS_STYLE_PROFILE, &id)
    {
        Ok(Some(p)) => p,
        Ok(None) => match compute_and_store(&state, &id) {
            Ok(fp) => fp.structure,
            Err(e) => return error_response(e).into_response(),
        },
        Err(e) => return error_response(e).into_response(),
    };
    let radar = compute_radar(&profile);
    json_ok(&radar)
}

// ─── PAN@CLEF Authorship Verification ───────────────────────

use crate::analysis::pan_verification::{
    evaluate as pan_evaluate, score_pairs, verify_texts, PanMetrics, VerificationConfig,
    VerificationPair, VerificationScore,
};

/// Request body for POST /style/verify.
#[derive(Deserialize)]
pub struct VerifyRequest {
    pub text_a: String,
    pub text_b: String,
    #[serde(default)]
    pub config: Option<VerificationConfig>,
}

/// Response body for POST /style/verify.
#[derive(Serialize)]
pub struct VerifyResponse {
    pub score: f32,
    pub decision: Option<bool>,
    pub same_author_probability: f32,
}

/// POST /style/verify — Score a single authorship-verification pair.
///
/// Returns the calibrated same-author probability in `[0, 1]` and a decision
/// (`Some(bool)` outside the uncertainty band, `None` inside it).
pub async fn verify_authorship(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<VerifyRequest>,
) -> impl IntoResponse {
    let cfg = body.config.unwrap_or_default();
    let (score, decision) = verify_texts(&body.text_a, &body.text_b, &cfg);
    json_ok(&VerifyResponse {
        score,
        decision,
        same_author_probability: score,
    })
}

/// Request body for POST /style/pan/evaluate.
///
/// Either provide `pairs` inline (each with optional `same_author`) or
/// `dataset_path` (server-side file path to a PAN JSONL file). `truth_path`
/// is optional and used when the pairs file has no labels.
#[derive(Deserialize)]
pub struct PanEvaluateRequest {
    #[serde(default)]
    pub pairs: Option<Vec<VerificationPair>>,
    #[serde(default)]
    pub dataset_path: Option<String>,
    #[serde(default)]
    pub truth_path: Option<String>,
    #[serde(default)]
    pub config: Option<VerificationConfig>,
}

/// Response body for POST /style/pan/evaluate.
#[derive(Serialize)]
pub struct PanEvaluateResponse {
    pub metrics: PanMetrics,
    pub scores: Vec<VerificationScore>,
}

/// POST /style/pan/evaluate — Run PAN@CLEF evaluation on a set of pairs.
pub async fn pan_evaluate_handler(
    State(_state): State<Arc<AppState>>,
    Json(body): Json<PanEvaluateRequest>,
) -> impl IntoResponse {
    // Resolve pairs: inline takes priority, otherwise load from path.
    let mut pairs = match (&body.pairs, &body.dataset_path) {
        (Some(p), _) => p.clone(),
        (None, Some(path)) => match crate::analysis::pan_loader::load_pan_jsonl(path) {
            Ok(p) => p,
            Err(e) => return error_response(e).into_response(),
        },
        (None, None) => {
            return error_response(crate::error::TensaError::InvalidQuery(
                "PAN evaluate: supply either `pairs` or `dataset_path`".into(),
            ))
            .into_response();
        }
    };

    // Optional truth merge.
    if let Some(tp) = &body.truth_path {
        match crate::analysis::pan_loader::load_pan_truth(tp) {
            Ok(truth) => crate::analysis::pan_loader::apply_truth(&mut pairs, &truth),
            Err(e) => return error_response(e).into_response(),
        }
    }

    // Ensure labels are present for evaluation.
    let labels: Vec<bool> = match pairs
        .iter()
        .map(|p| p.same_author)
        .collect::<Option<Vec<_>>>()
    {
        Some(v) => v,
        None => {
            return error_response(crate::error::TensaError::InvalidQuery(
                "PAN evaluate: every pair must have `same_author` (inline or via truth_path)"
                    .into(),
            ))
            .into_response();
        }
    };

    let cfg = body.config.unwrap_or_default();
    let scores = score_pairs(&pairs, &cfg);
    let metrics = pan_evaluate(&scores, &labels);
    json_ok(&PanEvaluateResponse { metrics, scores })
}

/// GET /settings/style-weights — Get the persisted weighted-similarity config.
pub async fn get_style_weights(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match kv_load::<WeightedSimilarityConfig>(&state, keys::ANALYSIS_STYLE_WEIGHTS, "default") {
        Ok(Some(cfg)) => json_ok(&cfg),
        Ok(None) => json_ok(&WeightedSimilarityConfig::default()),
        Err(e) => error_response(e).into_response(),
    }
}

/// PUT /settings/style-weights — Persist a weighted-similarity config.
pub async fn put_style_weights(
    State(state): State<Arc<AppState>>,
    Json(cfg): Json<WeightedSimilarityConfig>,
) -> impl IntoResponse {
    match kv_store(&state, keys::ANALYSIS_STYLE_WEIGHTS, "default", &cfg) {
        Ok(()) => json_ok(&cfg),
        Err(e) => error_response(e).into_response(),
    }
}
