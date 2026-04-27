//! Arc classification using Reagan et al. 6-arc taxonomy.
//!
//! Classifies narrative arcs by computing a "fortune trajectory"
//! from situation payoffs and sentiment, then fitting to one of
//! six canonical arc shapes via correlation.
//!
//! Sentiment scoring supports two backends:
//! - **Keyword** (default): fast, no dependencies, uses configurable word lists
//! - **ONNX ML model** (feature-gated `embedding`): higher accuracy via
//!   a pre-trained sentiment classifier (e.g., distilbert-sst2)

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::Role;

// ─── Sentiment Scorer Trait ─────────────────────────────────

/// Trait for scoring sentiment of text passages.
/// Implementations return a score where positive = good fortune, negative = bad fortune.
pub trait SentimentScorer: Send + Sync {
    /// Score the sentiment of a text passage. Returns a value roughly in [-3, +3].
    fn score(&self, text: &str) -> f64;
    /// Name of the scorer for logging.
    fn name(&self) -> &str;
}

/// Keyword-based sentiment scorer (default). Uses configurable word lists.
pub struct KeywordSentimentScorer {
    pub positive_words: Vec<String>,
    pub negative_words: Vec<String>,
}

impl KeywordSentimentScorer {
    pub fn from_config(config: &ArcConfig) -> Self {
        Self {
            positive_words: config.positive_words.clone(),
            negative_words: config.negative_words.clone(),
        }
    }
}

impl SentimentScorer for KeywordSentimentScorer {
    fn score(&self, text: &str) -> f64 {
        count_sentiment_with_words(text, &self.positive_words, &self.negative_words)
    }
    fn name(&self) -> &str {
        "keyword"
    }
}

/// ONNX-based ML sentiment scorer (feature-gated behind `embedding`).
///
/// Wraps an ONNX session for a binary sentiment classifier (e.g., distilbert-sst2).
/// The model should output a single logit score per input where positive = positive sentiment.
#[cfg(feature = "embedding")]
pub struct OnnxSentimentScorer {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
}

#[cfg(feature = "embedding")]
impl OnnxSentimentScorer {
    /// Load sentiment model from a directory containing `model.onnx` and `tokenizer.json`.
    pub fn from_directory(dir: &str) -> Result<Self> {
        use crate::error::TensaError;
        use ort::session::builder::GraphOptimizationLevel;

        let dir_path = std::path::Path::new(dir);
        let model_path = dir_path.join("model.onnx");
        let tokenizer_path = dir_path.join("tokenizer.json");

        if !model_path.exists() {
            return Err(TensaError::EmbeddingError(format!(
                "Sentiment model not found: {}",
                model_path.display()
            )));
        }

        let model_str = model_path
            .to_str()
            .ok_or_else(|| TensaError::EmbeddingError("Non-UTF-8 path".into()))?;
        let tokenizer_str = tokenizer_path
            .to_str()
            .ok_or_else(|| TensaError::EmbeddingError("Non-UTF-8 path".into()))?;

        let session = ort::session::Session::builder()
            .map_err(|e| TensaError::EmbeddingError(format!("Session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TensaError::EmbeddingError(format!("Optimization: {e}")))?
            .commit_from_file(model_str)
            .map_err(|e| TensaError::EmbeddingError(format!("Model load: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_str)
            .map_err(|e| TensaError::EmbeddingError(format!("Tokenizer load: {e}")))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            max_length: 128,
        })
    }
}

#[cfg(feature = "embedding")]
impl SentimentScorer for OnnxSentimentScorer {
    fn score(&self, text: &str) -> f64 {
        use ndarray::Array2;
        use ort::value::Tensor;

        let encoding = match self.tokenizer.encode(text, true) {
            Ok(e) => e,
            Err(_) => return 0.0,
        };

        let ids = encoding.get_ids();
        let attn = encoding.get_attention_mask();
        let len = ids.len().min(self.max_length);

        let ids_i64: Vec<i64> = ids[..len].iter().map(|&x| x as i64).collect();
        let attn_i64: Vec<i64> = attn[..len].iter().map(|&x| x as i64).collect();

        let ids_array = match Array2::from_shape_vec((1, len), ids_i64) {
            Ok(a) => a,
            Err(_) => return 0.0,
        };
        let attn_array = match Array2::from_shape_vec((1, len), attn_i64) {
            Ok(a) => a,
            Err(_) => return 0.0,
        };

        let ids_tensor = match Tensor::from_array(ids_array) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };
        let attn_tensor = match Tensor::from_array(attn_array) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        let mut session = match self.session.lock() {
            Ok(s) => s,
            Err(_) => return 0.0,
        };

        let outputs = match session.run(ort::inputs![ids_tensor, attn_tensor]) {
            Ok(o) => o,
            Err(_) => return 0.0,
        };

        // Extract logits: typically [1, 2] for binary sentiment (neg, pos)
        if let Some((_name, output)) = outputs.iter().next() {
            if let Ok(tensor) = output.try_extract_tensor::<f32>() {
                let values: Vec<f32> = tensor.1.iter().copied().collect();
                if values.len() >= 2 {
                    // Return positive - negative logit as sentiment score
                    return (values[1] - values[0]) as f64;
                } else if let Some(&v) = values.first() {
                    return v as f64;
                }
            }
        }
        0.0
    }

    fn name(&self) -> &str {
        "onnx-sentiment"
    }
}

// ─── Config ─────────────────────────────────────────────────

/// Configuration for arc classification parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcConfig {
    /// Moving average smoothing window size. Default: 3.
    #[serde(default = "default_smoothing_window")]
    pub smoothing_window: usize,
    /// Minimum Pearson correlation for confident classification. Default: 0.3.
    #[serde(default = "default_correlation_threshold")]
    pub correlation_threshold: f64,
    /// Positive sentiment keywords.
    #[serde(default = "default_positive_words")]
    pub positive_words: Vec<String>,
    /// Negative sentiment keywords.
    #[serde(default = "default_negative_words")]
    pub negative_words: Vec<String>,
}

fn default_smoothing_window() -> usize {
    3
}
fn default_correlation_threshold() -> f64 {
    0.3
}
fn default_positive_words() -> Vec<String> {
    // VADER-inspired positive lexicon. Keeps short, high-signal word stems
    // so substring matching (`text.contains(word)`) hits morphological
    // variants (`hope`/`hoped`/`hopeful`). A prior 12-word list produced
    // flat trajectories on long narratives; this is wide enough that most
    // chapters score non-zero.
    [
        "love",
        "loved",
        "loving",
        "loves",
        "lovely",
        "joy",
        "joyful",
        "happy",
        "happiness",
        "hope",
        "hoped",
        "hopeful",
        "delight",
        "elated",
        "ecstatic",
        "blissful",
        "success",
        "successful",
        "succeed",
        "won",
        "win",
        "wins",
        "winning",
        "victory",
        "victorious",
        "triumph",
        "triumphant",
        "prevail",
        "peace",
        "peaceful",
        "calm",
        "serene",
        "courage",
        "courageous",
        "brave",
        "bravery",
        "valor",
        "heroic",
        "freedom",
        "free",
        "freed",
        "liberated",
        "trust",
        "trusted",
        "faithful",
        "loyal",
        "loyalty",
        "save",
        "saved",
        "saves",
        "rescue",
        "rescued",
        "protect",
        "protected",
        "safe",
        "safety",
        "heal",
        "healed",
        "healing",
        "cured",
        "recover",
        "recovered",
        "kind",
        "kindness",
        "gentle",
        "tender",
        "warm",
        "warmth",
        "friend",
        "friendship",
        "ally",
        "united",
        "united",
        "laugh",
        "laughter",
        "smile",
        "smiled",
        "cheer",
        "cheerful",
        "bright",
        "brilliant",
        "wise",
        "wisdom",
        "clever",
        "thrive",
        "thrived",
        "flourish",
        "flourished",
        "bloom",
        "bloomed",
        "prosper",
        "prospered",
        "wealth",
        "rich",
        "riches",
        "fortune",
        "blessed",
        "blessing",
        "proud",
        "pride",
        "honor",
        "honored",
        "noble",
        "beautiful",
        "beauty",
        "gift",
        "gifted",
        "talent",
        "talented",
        "gratitude",
        "grateful",
        "thankful",
        "content",
        "satisfied",
        "embrace",
        "embraced",
        "reunion",
        "reunited",
        "wedding",
        "married",
        "birth",
        "born",
        "revive",
        "revived",
        "forgive",
        "forgiven",
        "pardon",
        "redeemed",
        "redemption",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}
fn default_negative_words() -> Vec<String> {
    [
        "death",
        "died",
        "dying",
        "dead",
        "kill",
        "killed",
        "killing",
        "murder",
        "murdered",
        "slain",
        "slay",
        "betray",
        "betrayed",
        "betrayal",
        "treason",
        "traitor",
        "fail",
        "failed",
        "failure",
        "loss",
        "lost",
        "lose",
        "defeat",
        "defeated",
        "ruined",
        "ruin",
        "pain",
        "painful",
        "hurt",
        "hurts",
        "wound",
        "wounded",
        "bleed",
        "bled",
        "bleeding",
        "fear",
        "feared",
        "afraid",
        "terror",
        "terrified",
        "dread",
        "horror",
        "horrified",
        "guilt",
        "guilty",
        "shame",
        "shamed",
        "humiliate",
        "humiliated",
        "disgrace",
        "despair",
        "despaired",
        "hopeless",
        "helpless",
        "misery",
        "miserable",
        "wretched",
        "prison",
        "imprisoned",
        "captive",
        "bound",
        "chained",
        "punish",
        "punished",
        "punishment",
        "sentence",
        "condemned",
        "sin",
        "sinful",
        "evil",
        "wicked",
        "cruel",
        "cruelty",
        "vile",
        "vicious",
        "bitter",
        "bitterness",
        "hatred",
        "hated",
        "hate",
        "hates",
        "angry",
        "anger",
        "rage",
        "furious",
        "wrath",
        "enraged",
        "broken",
        "break",
        "breaks",
        "shattered",
        "fall",
        "fell",
        "fallen",
        "falling",
        "sick",
        "sickness",
        "disease",
        "plague",
        "ill",
        "illness",
        "weak",
        "weakness",
        "collapse",
        "collapsed",
        "crumble",
        "crumbled",
        "destroy",
        "destroyed",
        "destruction",
        "war",
        "battle",
        "fought",
        "fight",
        "fights",
        "fighting",
        "blood",
        "bloody",
        "flee",
        "fled",
        "escape",
        "trapped",
        "tragedy",
        "tragic",
        "cry",
        "cried",
        "weep",
        "wept",
        "tears",
        "sorrow",
        "grief",
        "grieving",
        "mourn",
        "mourning",
        "lonely",
        "alone",
        "abandon",
        "abandoned",
        "exile",
        "exiled",
        "banish",
        "banished",
        "starve",
        "starved",
        "hunger",
        "cold",
        "dark",
        "darkness",
        "doom",
        "doomed",
        "curse",
        "cursed",
        "ominous",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl Default for ArcConfig {
    fn default() -> Self {
        Self {
            smoothing_window: default_smoothing_window(),
            correlation_threshold: default_correlation_threshold(),
            positive_words: default_positive_words(),
            negative_words: default_negative_words(),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────

/// The six canonical arc types from Reagan et al.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArcType {
    RagsToRiches, // monotonic rise
    RichesToRags, // monotonic fall
    ManInAHole,   // fall then rise (V-shape)
    Icarus,       // rise then fall (inverted V)
    Cinderella,   // rise-fall-rise (W-shape)
    Oedipus,      // fall-rise-fall (M-shape)
}

/// Result of classifying a narrative's arc.
///
/// `all_correlations` reports the Pearson correlation against every
/// Reagan template so callers can see how distinguishable the best-fit
/// is from the runners-up. `signal_quality` is the delta between the
/// top-1 and top-2 correlations — anything under ~0.1 means the classifier
/// is guessing rather than genuinely matching a shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcClassification {
    pub narrative_id: String,
    pub arc_type: ArcType,
    pub confidence: f32,
    pub sentiment_trajectory: Vec<f64>,
    pub key_turning_points: Vec<TurningPoint>,
    #[serde(default)]
    pub all_correlations: Vec<(ArcType, f64)>,
    #[serde(default)]
    pub signal_quality: f64,
    #[serde(default)]
    pub scorer: String,
}

/// A turning point in the fortune trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurningPoint {
    pub situation_id: Uuid,
    pub position: f64,     // 0.0-1.0 normalized position
    pub direction: String, // "rise" or "fall"
    pub magnitude: f64,
}

// ─── Arc Classification ──────────────────────────────────────

/// Classify the arc of a narrative using default config.
///
/// Auto-detects an ONNX sentiment model from `TENSA_SENTIMENT_MODEL`
/// (feature `embedding` only) and falls back to the keyword scorer
/// otherwise.
pub fn classify_arc(narrative_id: &str, hypergraph: &Hypergraph) -> Result<ArcClassification> {
    let config = ArcConfig::default();
    #[cfg(feature = "embedding")]
    if let Some(onnx) = try_load_onnx_sentiment_scorer() {
        return classify_arc_with_scorer(narrative_id, hypergraph, &config, &*onnx);
    }
    let scorer = KeywordSentimentScorer::from_config(&config);
    classify_arc_with_scorer(narrative_id, hypergraph, &config, &scorer)
}

/// Classify the arc of a narrative with custom config (keyword scorer).
pub fn classify_arc_with_config(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    config: &ArcConfig,
) -> Result<ArcClassification> {
    let scorer = KeywordSentimentScorer::from_config(config);
    classify_arc_with_scorer(narrative_id, hypergraph, config, &scorer)
}

/// Classify the arc of a narrative with a caller-supplied sentiment scorer.
pub fn classify_arc_with_scorer(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    config: &ArcConfig,
    scorer: &dyn SentimentScorer,
) -> Result<ArcClassification> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    if situations.is_empty() {
        return Ok(empty_classification(narrative_id, scorer.name()));
    }

    let mut sorted = situations.clone();
    sorted.sort_by(|a, b| {
        let a_time = a.temporal.start.unwrap_or_default();
        let b_time = b.temporal.start.unwrap_or_default();
        a_time.cmp(&b_time)
    });

    let trajectory = compute_fortune_trajectory_with_scorer(&sorted, hypergraph, config, scorer);

    if trajectory.is_empty() {
        return Ok(empty_classification(narrative_id, scorer.name()));
    }

    let smoothed = smooth_trajectory(&trajectory, config.smoothing_window);

    // Variance check — if the trajectory is effectively flat, Pearson against
    // every template will collapse to ~0 and the "best fit" is noise. Log a
    // warning so callers know the classifier isn't discriminating.
    let variance = {
        let mean = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
        smoothed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / smoothed.len() as f64
    };
    if variance < 1e-6 {
        tracing::warn!(
            narrative_id = %narrative_id,
            scorer = scorer.name(),
            variance = variance,
            "fortune trajectory is flat — arc classification will be degenerate. \
             Either the narrative is too short, the sentiment scorer matched no words, \
             or situations have empty content."
        );
    }

    let (arc_type, confidence, correlations, signal_quality) =
        fit_to_templates_full(&smoothed, config.correlation_threshold);

    let turning_points = detect_turning_points(&smoothed, &sorted);

    Ok(ArcClassification {
        narrative_id: narrative_id.to_string(),
        arc_type,
        confidence,
        sentiment_trajectory: smoothed,
        key_turning_points: turning_points,
        all_correlations: correlations,
        signal_quality,
        scorer: scorer.name().to_string(),
    })
}

fn empty_classification(narrative_id: &str, scorer_name: &str) -> ArcClassification {
    ArcClassification {
        narrative_id: narrative_id.to_string(),
        arc_type: ArcType::RagsToRiches,
        confidence: 0.0,
        sentiment_trajectory: vec![],
        key_turning_points: vec![],
        all_correlations: vec![],
        signal_quality: 0.0,
        scorer: scorer_name.to_string(),
    }
}

/// Try to load an ONNX sentiment model from `TENSA_SENTIMENT_MODEL` or the
/// default `models/sentiment/` directory. Returns `None` on any failure —
/// callers should fall back to the keyword scorer.
#[cfg(feature = "embedding")]
fn try_load_onnx_sentiment_scorer() -> Option<Box<dyn SentimentScorer>> {
    let path = std::env::var("TENSA_SENTIMENT_MODEL").ok().or_else(|| {
        let default = "models/sentiment";
        if std::path::Path::new(default).join("model.onnx").exists() {
            Some(default.into())
        } else {
            None
        }
    })?;
    match OnnxSentimentScorer::from_directory(&path) {
        Ok(s) => {
            tracing::info!(path = %path, "loaded ONNX sentiment scorer for arc classification");
            Some(Box::new(s))
        }
        Err(e) => {
            tracing::warn!(
                path = %path,
                error = %e,
                "failed to load ONNX sentiment model \u{2014} falling back to keyword scorer"
            );
            None
        }
    }
}

/// Compute fortune trajectory using a pluggable sentiment scorer.
///
/// Scores all narratively-bearing text on each situation: `name`,
/// `description`, `synopsis`, `label`, and each `raw_content` block.
/// Prior versions only scored `raw_content`, so situations ingested via
/// structured import (which populate `description` / `synopsis` but not
/// `raw_content`) produced flat trajectories.
pub fn compute_fortune_trajectory_with_scorer(
    situations: &[crate::types::Situation],
    hypergraph: &Hypergraph,
    _config: &ArcConfig,
    scorer: &dyn SentimentScorer,
) -> Vec<f64> {
    let mut trajectory = Vec::with_capacity(situations.len());

    for sit in situations {
        let mut score = 0.0;

        if let Some(name) = &sit.name {
            score += scorer.score(&name.to_lowercase());
        }
        if let Some(desc) = &sit.description {
            score += scorer.score(&desc.to_lowercase());
        }
        if let Some(syn) = &sit.synopsis {
            score += scorer.score(&syn.to_lowercase());
        }
        if let Some(label) = &sit.label {
            score += scorer.score(&label.to_lowercase());
        }
        for block in &sit.raw_content {
            score += scorer.score(&block.content.to_lowercase());
        }

        // Payoffs from participations. Kept bounded so a single
        // numerically-large payoff doesn't swamp the sentiment channel.
        if let Ok(participants) = hypergraph.get_participants_for_situation(&sit.id) {
            for p in &participants {
                if p.role == Role::Protagonist {
                    if let Some(val) = p.payoff.as_ref().and_then(|v| v.as_f64()) {
                        score += val.clamp(-3.0, 3.0);
                    }
                }
                if p.role == Role::Antagonist {
                    if let Some(val) = p.payoff.as_ref().and_then(|v| v.as_f64()) {
                        score -= val.clamp(-3.0, 3.0) * 0.5;
                    }
                }
            }
        }

        score += (sit.confidence as f64 - 0.5) * 0.1;

        trajectory.push(score);
    }

    trajectory
}

/// Keyword sentiment scoring with configurable word lists.
fn count_sentiment_with_words(text: &str, positive: &[String], negative: &[String]) -> f64 {
    let mut score = 0.0;
    for word in positive {
        if text.contains(word.as_str()) {
            score += 1.0;
        }
    }
    for word in negative {
        if text.contains(word.as_str()) {
            score -= 1.0;
        }
    }
    score
}

/// Apply moving average smoothing.
fn smooth_trajectory(trajectory: &[f64], window: usize) -> Vec<f64> {
    if trajectory.len() <= window {
        return trajectory.to_vec();
    }
    let half = window / 2;
    let mut smoothed = Vec::with_capacity(trajectory.len());
    for i in 0..trajectory.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(trajectory.len());
        let avg: f64 = trajectory[start..end].iter().sum::<f64>() / (end - start) as f64;
        smoothed.push(avg);
    }
    smoothed
}

/// Fit + report all correlations and a "signal quality" score.
///
/// `signal_quality` is `top1 - top2`: it's how much daylight the winning
/// template has over the runner-up. A value under ~0.1 means the classifier
/// is effectively guessing. `confidence` is kept as the historical
/// `best_corr * 0.8` when above threshold, or the 0.1 floor otherwise —
/// callers that want the real correlation should read `all_correlations`.
fn fit_to_templates_full(
    trajectory: &[f64],
    correlation_threshold: f64,
) -> (ArcType, f32, Vec<(ArcType, f64)>, f64) {
    let n = trajectory.len();
    if n < 2 {
        return (ArcType::RagsToRiches, 0.1, vec![], 0.0);
    }

    let templates = [
        (ArcType::RagsToRiches, generate_template(n, &[1.0])),
        (ArcType::RichesToRags, generate_template(n, &[-1.0])),
        (ArcType::ManInAHole, generate_template(n, &[-1.0, 1.0])),
        (ArcType::Icarus, generate_template(n, &[1.0, -1.0])),
        (ArcType::Cinderella, generate_template(n, &[1.0, -1.0, 1.0])),
        (ArcType::Oedipus, generate_template(n, &[-1.0, 1.0, -1.0])),
    ];

    let mut correlations: Vec<(ArcType, f64)> = templates
        .iter()
        .map(|(arc, tpl)| (arc.clone(), pearson_correlation(trajectory, tpl)))
        .collect();
    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let best_type = correlations[0].0.clone();
    let best_corr = correlations[0].1;
    let runner_up = correlations.get(1).map(|(_, c)| *c).unwrap_or(-1.0);
    let signal_quality = (best_corr - runner_up).max(0.0);

    let confidence = if best_corr > correlation_threshold {
        (best_corr * 0.8) as f32
    } else {
        0.1
    };

    (
        best_type,
        confidence.min(0.95),
        correlations,
        signal_quality,
    )
}

/// Generate a template trajectory with given segments.
fn generate_template(n: usize, segments: &[f64]) -> Vec<f64> {
    let seg_len = n / segments.len();
    let mut template = Vec::with_capacity(n);

    for (seg_idx, &direction) in segments.iter().enumerate() {
        let seg_start = seg_idx * seg_len;
        let seg_end = if seg_idx == segments.len() - 1 {
            n
        } else {
            (seg_idx + 1) * seg_len
        };
        let actual_len = seg_end - seg_start;

        for i in 0..actual_len {
            let t = i as f64 / actual_len.max(1) as f64;
            let base: f64 = if seg_idx > 0 {
                // Start from where previous segment ended
                let prev_end: f64 = segments[..seg_idx].iter().sum();
                prev_end
            } else {
                0.0
            };
            template.push(base + direction * t);
        }
    }
    template
}

/// Compute Pearson correlation between two sequences.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Detect turning points (local minima and maxima).
fn detect_turning_points(
    trajectory: &[f64],
    situations: &[crate::types::Situation],
) -> Vec<TurningPoint> {
    let mut points = Vec::new();
    let n = trajectory.len();
    if n < 3 {
        return points;
    }

    for i in 1..n - 1 {
        let prev = trajectory[i - 1];
        let curr = trajectory[i];
        let next = trajectory[i + 1];

        if curr > prev && curr > next {
            // Local maximum → start of fall
            points.push(TurningPoint {
                situation_id: situations[i].id,
                position: i as f64 / (n - 1) as f64,
                direction: "fall".to_string(),
                magnitude: (curr - next).abs(),
            });
        } else if curr < prev && curr < next {
            // Local minimum → start of rise
            points.push(TurningPoint {
                situation_id: situations[i].id,
                position: i as f64 / (n - 1) as f64,
                direction: "rise".to_string(),
                magnitude: (next - curr).abs(),
            });
        }
    }
    points
}

// ─── InferenceEngine Implementation ─────────────────────────

use crate::analysis::extract_narrative_id;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::{InferenceJobType, InferenceResult, JobStatus};

/// Arc classification engine for the inference job queue.
pub struct ArcEngine;

impl InferenceEngine for ArcEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ArcClassification
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000) // 3 seconds estimate
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;

        let classification = classify_arc(narrative_id, hypergraph)?;

        let explanation = if classification.signal_quality < 0.1 {
            format!(
                "Arc classified as {:?} (confidence={:.3}, scorer={}, signal_quality={:.3}). \
                 Signal quality is low — all Reagan templates correlate weakly with the fortune \
                 trajectory, so the best-fit is effectively a guess. Check `all_correlations`.",
                classification.arc_type,
                classification.confidence,
                classification.scorer,
                classification.signal_quality
            )
        } else {
            format!(
                "Arc classified as {:?} (confidence={:.3}, scorer={}, signal_quality={:.3})",
                classification.arc_type,
                classification.confidence,
                classification.scorer,
                classification.signal_quality
            )
        };

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::ArcClassification,
            target_id: job.target_id,
            result: serde_json::to_value(&classification)?,
            confidence: classification.confidence,
            explanation: Some(explanation),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(chrono::Utc::now()),
        })
    }
}

// ─── Per-actor classification (v0.74.1) ─────────────────────
//
// Narrative-level Reagan classification flattens every actor's fate into one
// fortune line. For multi-POV or adversarial work that signal is useless —
// Ned rises while Littlefinger rises while Cersei falls and the average is
// flat. Per-actor classification runs the same 6-template fit on each
// Actor's situation slice instead.

/// Classify the Reagan arc for a single actor within a narrative.
///
/// Scores only situations the actor participates in, weighted by the actor's
/// role in each (Protagonist = +1.0, Antagonist = -0.5, others = +0.3). This
/// is the symmetric extension of `classify_arc` from narrative scope down to
/// entity scope.
pub fn classify_arc_for_actor(
    actor_id: &Uuid,
    narrative_id: &str,
    hypergraph: &Hypergraph,
) -> Result<ArcClassification> {
    let config = ArcConfig::default();
    #[cfg(feature = "embedding")]
    if let Some(onnx) = try_load_onnx_sentiment_scorer() {
        return classify_arc_for_actor_with_scorer(
            actor_id,
            narrative_id,
            hypergraph,
            &config,
            &*onnx,
        );
    }
    let scorer = KeywordSentimentScorer::from_config(&config);
    classify_arc_for_actor_with_scorer(actor_id, narrative_id, hypergraph, &config, &scorer)
}

/// Per-actor variant with an injectable scorer — used by tests and callers
/// that want to swap in a custom sentiment backend.
pub fn classify_arc_for_actor_with_scorer(
    actor_id: &Uuid,
    narrative_id: &str,
    hypergraph: &Hypergraph,
    config: &ArcConfig,
    scorer: &dyn SentimentScorer,
) -> Result<ArcClassification> {
    let mut pairs = Vec::new();
    for p in hypergraph.get_situations_for_entity(actor_id)? {
        let sit = match hypergraph.get_situation(&p.situation_id) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if sit.narrative_id.as_deref() != Some(narrative_id) {
            continue;
        }
        pairs.push((sit, p.role));
    }
    Ok(classify_from_pairs(
        actor_id,
        narrative_id,
        pairs,
        config,
        scorer,
    ))
}

/// Shared core: given chronological `(Situation, Role)` pairs, produce the
/// ArcClassification. Used by both the single-actor path and the batch path
/// (`classify_arcs_per_actor`) so they can't drift.
fn classify_from_pairs(
    actor_id: &Uuid,
    narrative_id: &str,
    mut pairs: Vec<(crate::types::Situation, Role)>,
    config: &ArcConfig,
    scorer: &dyn SentimentScorer,
) -> ArcClassification {
    if pairs.is_empty() {
        return empty_classification(narrative_id, scorer.name());
    }
    pairs.sort_by_key(|(s, _)| s.temporal.start.unwrap_or_default());

    let trajectory = compute_actor_fortune_trajectory(&pairs, scorer);
    let smoothed = smooth_trajectory(&trajectory, config.smoothing_window);
    let variance = smoothed_variance(&smoothed);
    if variance < 1e-6 {
        tracing::warn!(
            narrative_id = %narrative_id,
            actor_id = %actor_id,
            scorer = scorer.name(),
            variance = variance,
            "actor fortune trajectory is flat \u{2014} arc classification will be degenerate"
        );
    }

    let (arc_type, confidence, correlations, signal_quality) =
        fit_to_templates_full(&smoothed, config.correlation_threshold);
    let sits: Vec<crate::types::Situation> = pairs.into_iter().map(|(s, _)| s).collect();
    let turning_points = detect_turning_points(&smoothed, &sits);

    ArcClassification {
        narrative_id: narrative_id.to_string(),
        arc_type,
        confidence,
        sentiment_trajectory: smoothed,
        key_turning_points: turning_points,
        all_correlations: correlations,
        signal_quality,
        scorer: scorer.name().to_string(),
    }
}

fn smoothed_variance(smoothed: &[f64]) -> f64 {
    if smoothed.is_empty() {
        return 0.0;
    }
    let mean = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
    smoothed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / smoothed.len() as f64
}

/// Classify every Actor in a narrative that participates in at least one
/// situation. Returns one `(actor_id, ArcClassification)` per actor,
/// ordered by signal_quality descending so callers can triage by confidence.
///
/// Loads the narrative's situations once and builds an actor → pairs index
/// from participation records, avoiding the N+1 `get_situation` scan that
/// naive per-actor iteration would produce on large narratives. Loads the
/// ONNX sentiment model (if configured) once and shares it across actors.
pub fn classify_arcs_per_actor(
    narrative_id: &str,
    hypergraph: &Hypergraph,
) -> Result<Vec<(Uuid, ArcClassification)>> {
    let config = ArcConfig::default();
    #[cfg(feature = "embedding")]
    if let Some(onnx) = try_load_onnx_sentiment_scorer() {
        return classify_arcs_per_actor_with_scorer(narrative_id, hypergraph, &config, &*onnx);
    }
    let scorer = KeywordSentimentScorer::from_config(&config);
    classify_arcs_per_actor_with_scorer(narrative_id, hypergraph, &config, &scorer)
}

/// Batch variant taking a caller-owned scorer — lets the ONNX model load once
/// per narrative rather than once per actor.
pub fn classify_arcs_per_actor_with_scorer(
    narrative_id: &str,
    hypergraph: &Hypergraph,
    config: &ArcConfig,
    scorer: &dyn SentimentScorer,
) -> Result<Vec<(Uuid, ArcClassification)>> {
    use std::collections::HashMap;

    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let sit_by_id: HashMap<Uuid, crate::types::Situation> =
        situations.into_iter().map(|s| (s.id, s)).collect();

    let actors: Vec<_> = hypergraph
        .list_entities_by_narrative(narrative_id)?
        .into_iter()
        .filter(|e| e.entity_type == crate::types::EntityType::Actor)
        .collect();

    let mut out = Vec::with_capacity(actors.len());
    for actor in actors {
        let mut pairs = Vec::new();
        for p in hypergraph.get_situations_for_entity(&actor.id)? {
            if let Some(sit) = sit_by_id.get(&p.situation_id) {
                pairs.push((sit.clone(), p.role));
            }
        }
        if pairs.is_empty() {
            continue;
        }
        let arc = classify_from_pairs(&actor.id, narrative_id, pairs, config, scorer);
        out.push((actor.id, arc));
    }
    out.sort_by(|a, b| {
        b.1.signal_quality
            .partial_cmp(&a.1.signal_quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// Per-actor trajectory — role-weighted per situation.
/// A scene of violence is a fall for the victim, a rise for the aggressor,
/// and mostly informational for a witness.
fn compute_actor_fortune_trajectory(
    pairs: &[(crate::types::Situation, Role)],
    scorer: &dyn SentimentScorer,
) -> Vec<f64> {
    pairs
        .iter()
        .map(|(sit, role)| {
            let mut s = 0.0;
            if let Some(name) = &sit.name {
                s += scorer.score(&name.to_lowercase());
            }
            if let Some(desc) = &sit.description {
                s += scorer.score(&desc.to_lowercase());
            }
            if let Some(syn) = &sit.synopsis {
                s += scorer.score(&syn.to_lowercase());
            }
            if let Some(label) = &sit.label {
                s += scorer.score(&label.to_lowercase());
            }
            for block in &sit.raw_content {
                s += scorer.score(&block.content.to_lowercase());
            }
            s * role_weight(role) + (sit.confidence as f64 - 0.5) * 0.1
        })
        .collect()
}

fn role_weight(role: &Role) -> f64 {
    match role {
        Role::Protagonist => 1.0,
        Role::Antagonist => -0.5,
        Role::Target => -0.8,
        Role::Witness | Role::Bystander => 0.3,
        Role::Confidant | Role::Recipient => 0.7,
        Role::Instrument | Role::Informant | Role::Facilitator => 0.4,
        Role::SubjectOfDiscussion => 0.2,
        Role::Custom(_) => 0.5,
    }
}

/// Per-actor arc classification engine for the inference job queue.
///
/// Accepts either of two parameter shapes:
/// - `{narrative_id, actor_id}` — classify a single actor, write result at
///   `an/aa/{narrative_id}/{actor_id}`.
/// - `{narrative_id}` — classify every Actor in the narrative, write one
///   row per actor at `an/aa/{narrative_id}/{actor_id}`.
pub struct ActorArcEngine;

impl InferenceEngine for ActorArcEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ActorArcClassification
    }

    fn estimate_cost(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
        let nid = extract_narrative_id(job)?;
        let actor_count = hypergraph
            .list_entities_by_narrative(nid)
            .map(|v| {
                v.iter()
                    .filter(|e| e.entity_type == crate::types::EntityType::Actor)
                    .count() as u64
            })
            .unwrap_or(1);
        // 500ms per actor + constant startup (first-call ONNX model load amortizes here)
        Ok((actor_count.max(1) * 500).max(2000))
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let actor_id = resolve_actor_id(job);

        let results: Vec<(Uuid, ArcClassification)> = if let Some(aid) = actor_id {
            vec![(aid, classify_arc_for_actor(&aid, narrative_id, hypergraph)?)]
        } else {
            classify_arcs_per_actor(narrative_id, hypergraph)?
        };

        // Persist the full ArcClassification at `an/aa/{nid}/{actor_id}`.
        // The virtual-property resolver reads `arc_type`, `confidence`, and
        // `signal_quality` by name — these field names are stable across the
        // ArcClassification struct, so no trimmed shadow struct is needed.
        let mut max_signal = f64::NEG_INFINITY;
        let mut best_idx: Option<usize> = None;
        for (i, (aid, arc)) in results.iter().enumerate() {
            let key = crate::analysis::analysis_key(b"an/aa/", &[narrative_id, &aid.to_string()]);
            hypergraph.store().put(&key, &serde_json::to_vec(arc)?)?;
            if arc.signal_quality > max_signal {
                max_signal = arc.signal_quality;
                best_idx = Some(i);
            }
        }

        let summary: Vec<_> = results
            .iter()
            .map(|(aid, arc)| {
                serde_json::json!({
                    "actor_id": aid.to_string(),
                    "arc_type": arc.arc_type,
                    "confidence": arc.confidence,
                    "signal_quality": arc.signal_quality,
                    "scorer": arc.scorer,
                    "all_correlations": arc.all_correlations,
                })
            })
            .collect();

        let best = best_idx.map(|i| &results[i]);
        let confidence = best.map(|(_, a)| a.confidence).unwrap_or(0.0);
        let explanation = match best {
            Some((_, a)) => format!(
                "Classified {} actor arc(s). Top: {:?} (signal_quality={:.3}, scorer={})",
                results.len(),
                a.arc_type,
                a.signal_quality,
                a.scorer
            ),
            None => format!(
                "No Actor entities with situations found in narrative `{}`",
                narrative_id
            ),
        };

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::ActorArcClassification,
            target_id: job.target_id,
            result: serde_json::json!({
                "narrative_id": narrative_id,
                "actors": summary,
            }),
            confidence,
            explanation: Some(explanation),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(chrono::Utc::now()),
        })
    }
}

/// Single-actor mode triggers on either an explicit `actor_id` param
/// (MCP/REST callers) or a real UUID in `target_id` (which is what the
/// planner lifts from `INFER ARCS FOR e:Actor WHERE e.id = "..."`).
fn resolve_actor_id(job: &InferenceJob) -> Option<Uuid> {
    let get_uuid = |key: &str| -> Option<Uuid> {
        job.parameters
            .get(key)
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::try_parse(s).ok())
    };
    get_uuid("actor_id").or_else(|| get_uuid("target_id"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::{Duration, Utc};
    use std::sync::Arc;

    fn setup() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_sit(hg: &Hypergraph, nid: &str, hours: i64, content: &str) -> Uuid {
        let base = Utc::now();
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(base + Duration::hours(hours)),
                end: Some(base + Duration::hours(hours + 1)),
                granularity: TimeGranularity::Approximate,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(content)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(nid.to_string()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_situation(sit).unwrap()
    }

    #[test]
    fn test_classify_empty_narrative() {
        let hg = setup();
        let result = classify_arc("empty", &hg).unwrap();
        assert_eq!(result.confidence, 0.0);
        assert!(result.sentiment_trajectory.is_empty());
    }

    #[test]
    fn test_classify_rising_arc() {
        let hg = setup();
        // Create situations with increasingly positive content
        make_sit(&hg, "test", 0, "pain and despair");
        make_sit(&hg, "test", 1, "some hope appears");
        make_sit(&hg, "test", 2, "hope and courage");
        make_sit(&hg, "test", 3, "joy and triumph");
        make_sit(&hg, "test", 4, "victory and freedom");

        let result = classify_arc("test", &hg).unwrap();
        assert_eq!(result.arc_type, ArcType::RagsToRiches);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_fit_to_templates_discriminates() {
        // Feed the classifier a trajectory that IS literally the Cinderella
        // template — the best fit should be Cinderella with signal_quality
        // clearly above the 0.1 "noise floor."
        let cinderella_tpl = generate_template(12, &[1.0, -1.0, 1.0]);
        let (arc, confidence, correlations, signal_quality) =
            fit_to_templates_full(&cinderella_tpl, 0.3);
        assert_eq!(arc, ArcType::Cinderella);
        assert!(
            confidence > 0.5,
            "confidence {} should be high on a literal template match",
            confidence
        );
        assert!(
            signal_quality > 0.1,
            "signal_quality {:.3} should exceed 0.1 \u{2014} classifier not discriminating",
            signal_quality
        );
        assert_eq!(correlations[0].0, ArcType::Cinderella);
        assert!(correlations[0].1 > correlations[1].1);
    }

    #[test]
    fn test_fit_to_templates_flat_yields_low_signal() {
        // A constant trajectory has zero variance → Pearson undefined for
        // every template, all correlations near zero, signal_quality near zero.
        let flat = vec![1.0; 10];
        let (_, _, _correlations, signal_quality) = fit_to_templates_full(&flat, 0.3);
        assert!(
            signal_quality < 0.05,
            "flat trajectory signal_quality {:.3} should be near zero",
            signal_quality
        );
    }

    #[test]
    fn test_classify_falling_arc() {
        let hg = setup();
        make_sit(&hg, "test", 0, "love and joy and hope");
        make_sit(&hg, "test", 1, "some worry");
        make_sit(&hg, "test", 2, "fear grows");
        make_sit(&hg, "test", 3, "betrayal and shame");
        make_sit(&hg, "test", 4, "death and despair");

        let result = classify_arc("test", &hg).unwrap();
        assert_eq!(result.arc_type, ArcType::RichesToRags);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_pearson_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = pearson_correlation(&x, &x);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pearson_opposite() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_smooth_trajectory() {
        let t = vec![0.0, 10.0, 0.0, 10.0, 0.0];
        let smoothed = smooth_trajectory(&t, 3);
        assert_eq!(smoothed.len(), 5);
        // Middle values should be smoothed
        assert!(smoothed[1] > 0.0 && smoothed[1] < 10.0);
    }

    #[test]
    fn test_turning_points_v_shape() {
        let trajectory = vec![1.0, 0.5, 0.0, 0.5, 1.0];
        let situations: Vec<Situation> = (0..5)
            .map(|_| Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(Utc::now()),
                    end: Some(Utc::now()),
                    granularity: TimeGranularity::Approximate,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.8,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: None,
                source_chunk_id: None,
                source_span: None,
                synopsis: None,
                manuscript_order: None,
                parent_situation_id: None,
                label: None,
                status: None,
                keywords: vec![],
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .collect();

        let points = detect_turning_points(&trajectory, &situations);
        assert!(!points.is_empty());
        assert_eq!(points[0].direction, "rise"); // minimum → start of rise
    }

    #[test]
    fn test_count_sentiment() {
        let config = ArcConfig::default();
        let score =
            |text| count_sentiment_with_words(text, &config.positive_words, &config.negative_words);
        assert!(score("love and joy") > 0.0);
        assert!(score("death and despair") < 0.0);
        assert_eq!(score("neutral text"), 0.0);
    }

    #[test]
    fn test_generate_template_rise() {
        let t = generate_template(10, &[1.0]);
        assert_eq!(t.len(), 10);
        assert!(t[9] > t[0]);
    }

    #[test]
    fn test_generate_template_v_shape() {
        let t = generate_template(10, &[-1.0, 1.0]);
        assert_eq!(t.len(), 10);
        // First half should decrease, second half increase
        assert!(t[4] < t[0]);
    }

    #[test]
    fn test_arc_serialization() {
        let arc = ArcClassification {
            narrative_id: "test".to_string(),
            arc_type: ArcType::ManInAHole,
            confidence: 0.7,
            sentiment_trajectory: vec![1.0, -1.0, 1.0],
            key_turning_points: vec![],
            all_correlations: vec![],
            signal_quality: 0.0,
            scorer: "keyword".to_string(),
        };
        let json = serde_json::to_vec(&arc).unwrap();
        let decoded: ArcClassification = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded.arc_type, ArcType::ManInAHole);
        assert_eq!(decoded.confidence, 0.7);
    }

    #[test]
    fn test_arc_engine_execute() {
        let hg = setup();
        make_sit(&hg, "arc-eng", 0, "pain and despair");
        make_sit(&hg, "arc-eng", 1, "hope and joy");
        make_sit(&hg, "arc-eng", 2, "triumph and victory");

        let engine = ArcEngine;
        assert_eq!(engine.job_type(), InferenceJobType::ArcClassification);

        let job = crate::inference::types::InferenceJob {
            id: "arc-test".to_string(),
            job_type: InferenceJobType::ArcClassification,
            target_id: uuid::Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": "arc-eng"}),
            priority: crate::types::JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);
        assert!(result.completed_at.is_some());
        assert!(result.result.get("arc_type").is_some());
    }

    // ─── Per-actor arc classification tests (v0.74.1) ──────

    /// Create an Actor entity in a narrative and return its id.
    fn make_actor(hg: &Hypergraph, nid: &str, name: &str) -> Uuid {
        let e = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            narrative_id: Some(nid.to_string()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::HumanEntered),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(e).unwrap()
    }

    fn add_participation(hg: &Hypergraph, actor: Uuid, sit: Uuid, role: Role) {
        hg.add_participant(Participation {
            entity_id: actor,
            situation_id: sit,
            seq: 0,
            role,
            info_set: None,
            action: None,
            payoff: None,
        })
        .unwrap();
    }

    #[test]
    fn test_classify_arc_for_actor_no_participations() {
        let hg = setup();
        let actor = make_actor(&hg, "multi", "Ghost");
        let result = classify_arc_for_actor(&actor, "multi", &hg).unwrap();
        assert!(result.sentiment_trajectory.is_empty());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_per_actor_separates_opposing_arcs() {
        // The payoff case: a narrative whose flattened trajectory is flat
        // because two actors' arcs cancel out, but per-actor classification
        // correctly identifies one rising and one falling actor.
        let hg = setup();
        let nid = "dual";
        let hero = make_actor(&hg, nid, "Hero");
        let villain = make_actor(&hg, nid, "Villain");

        // Per-actor role-weighting inverts antagonist sentiment, so for the
        // villain to read as "rising" (his fortunes going up as he wins) we
        // need NEGATIVE sentiment on his protagonist-role scenes (his own
        // triumph reads as fall for others), then POSITIVE sentiment later
        // (describing loss). Easier: just tag the villain Protagonist of his
        // own scenes and write the text from his POV.
        for (h, content) in [
            (0, "defeat and despair"),
            (1, "shame and loss"),
            (2, "pain and fear"),
            (3, "courage stirs"),
            (4, "hope returns"),
            (5, "triumph and victory"),
        ] {
            let s = make_sit(&hg, nid, h, content);
            add_participation(&hg, hero, s, Role::Protagonist);
        }
        for (h, content) in [
            (6, "triumph and victory"),
            (7, "love and joy"),
            (8, "peace and freedom"),
            (9, "fear and pain"),
            (10, "shame and defeat"),
            (11, "death and despair"),
        ] {
            let s = make_sit(&hg, nid, h, content);
            add_participation(&hg, villain, s, Role::Protagonist);
        }

        let results = classify_arcs_per_actor(nid, &hg).unwrap();
        assert_eq!(results.len(), 2, "both actors should produce trajectories");

        let hero_arc = results.iter().find(|(id, _)| *id == hero).unwrap();
        let villain_arc = results.iter().find(|(id, _)| *id == villain).unwrap();

        assert_eq!(
            hero_arc.1.arc_type,
            ArcType::RagsToRiches,
            "hero scenes go despair → victory; expected RagsToRiches, got {:?} \
             (signal_quality={:.3})",
            hero_arc.1.arc_type,
            hero_arc.1.signal_quality
        );
        assert_eq!(
            villain_arc.1.arc_type,
            ArcType::RichesToRags,
            "villain scenes go triumph → despair; expected RichesToRags"
        );
        // Per-actor trajectories should have real signal, not noise floor.
        assert!(
            hero_arc.1.signal_quality > 0.1,
            "hero signal_quality={:.3} should be well above floor",
            hero_arc.1.signal_quality
        );
    }

    #[test]
    fn test_actor_arc_engine_persists_to_kv() {
        // ActorArcEngine.execute must write a row at `an/aa/{nid}/{actor_id}`
        // so the virtual-property resolver can answer `e.an.arc_type`.
        let hg = setup();
        let nid = "persist";
        let a = make_actor(&hg, nid, "A");
        let s1 = make_sit(&hg, nid, 0, "despair and loss");
        let s2 = make_sit(&hg, nid, 1, "pain and fear");
        let s3 = make_sit(&hg, nid, 2, "hope returns");
        let s4 = make_sit(&hg, nid, 3, "triumph and joy");
        for s in [s1, s2, s3, s4] {
            add_participation(&hg, a, s, Role::Protagonist);
        }

        let engine = ActorArcEngine;
        let job = InferenceJob {
            id: "per-actor-kv".into(),
            job_type: InferenceJobType::ActorArcClassification,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({"narrative_id": nid}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let result = engine.execute(&job, &hg).unwrap();
        assert_eq!(result.status, JobStatus::Completed);

        let key = crate::analysis::analysis_key(b"an/aa/", &[nid, &a.to_string()]);
        let stored = hg.store().get(&key).unwrap();
        assert!(stored.is_some(), "per-actor arc result should be persisted at an/aa/");
        let parsed: serde_json::Value =
            serde_json::from_slice(&stored.unwrap()).unwrap();
        assert!(parsed.get("arc_type").is_some());
        assert!(parsed.get("signal_quality").is_some());
    }

    #[test]
    fn test_actor_arc_engine_single_actor_mode_via_target_id() {
        // `INFER ARCS FOR e:Actor WHERE e.id = "..."` dispatches with target_id
        // set. The engine must run only for that one actor.
        let hg = setup();
        let nid = "single";
        let a = make_actor(&hg, nid, "A");
        let b = make_actor(&hg, nid, "B");
        let s = make_sit(&hg, nid, 0, "joy");
        add_participation(&hg, a, s, Role::Protagonist);
        add_participation(&hg, b, s, Role::Protagonist);

        let engine = ActorArcEngine;
        let job = InferenceJob {
            id: "single-actor".into(),
            job_type: InferenceJobType::ActorArcClassification,
            target_id: a,
            parameters: serde_json::json!({
                "narrative_id": nid,
                "target_id": a.to_string(),
            }),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let result = engine.execute(&job, &hg).unwrap();
        let actors = result.result.get("actors").and_then(|v| v.as_array()).unwrap();
        assert_eq!(actors.len(), 1, "single-actor mode should yield exactly one row");
        assert_eq!(
            actors[0].get("actor_id").and_then(|v| v.as_str()),
            Some(a.to_string().as_str())
        );
    }
}
