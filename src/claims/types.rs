//! Claim pipeline types (Sprint D4).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A verifiable claim extracted from narrative content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub id: Uuid,
    /// The claim text in normalized form.
    pub text: String,
    /// Original verbatim text before normalization.
    pub original_text: String,
    /// Language code (ISO 639-1).
    pub language: String,
    /// Source situation UUID where the claim was extracted from.
    pub source_situation_id: Option<Uuid>,
    /// Source entity (actor) who made the claim.
    pub source_entity_id: Option<Uuid>,
    /// Narrative this claim belongs to.
    pub narrative_id: Option<String>,
    /// Embedding vector for similarity matching.
    pub embedding: Option<Vec<f64>>,
    /// Confidence that this is a verifiable claim (0.0-1.0).
    pub confidence: f64,
    /// Claim category for prioritization.
    pub category: ClaimCategory,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Category of a claim for triage/prioritization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimCategory {
    /// Contains specific numbers or statistics.
    Numerical,
    /// Direct or indirect quote attribution.
    Quote,
    /// Asserts a cause-effect relationship.
    Causal,
    /// Compares two or more things.
    Comparison,
    /// Predicts a future event.
    Predictive,
    /// General factual assertion.
    Factual,
    /// Could not be categorized.
    Unknown,
}

/// A fact-check verdict for a claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheck {
    pub id: Uuid,
    /// UUID of the claim being checked.
    pub claim_id: Uuid,
    /// Verdict from the fact-checker.
    pub verdict: FactCheckVerdict,
    /// Source organization or fact-checker name.
    pub source: String,
    /// URL to the fact-check article.
    pub url: Option<String>,
    /// Language of the fact-check.
    pub language: String,
    /// Explanation or summary of the fact-check.
    pub explanation: Option<String>,
    /// Confidence in this fact-check (0.0-1.0).
    pub confidence: f64,
    pub created_at: DateTime<Utc>,
}

/// Possible verdicts from a fact-check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactCheckVerdict {
    True,
    False,
    Misleading,
    PartiallyTrue,
    Unverifiable,
    Satire,
    OutOfContext,
}

impl std::str::FromStr for FactCheckVerdict {
    type Err = crate::TensaError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "true" => Ok(Self::True),
            "false" => Ok(Self::False),
            "misleading" => Ok(Self::Misleading),
            "partially_true" | "partiallytrue" | "partially true" => Ok(Self::PartiallyTrue),
            "unverifiable" => Ok(Self::Unverifiable),
            "satire" => Ok(Self::Satire),
            "out_of_context" | "outofcontext" | "out of context" => Ok(Self::OutOfContext),
            other => Err(crate::TensaError::InvalidQuery(format!(
                "Unknown verdict: {}",
                other
            ))),
        }
    }
}

/// A match between a claim and an existing fact-check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedFactCheck {
    pub claim_id: Uuid,
    pub fact_check_id: Uuid,
    /// Semantic similarity score between the claim and the fact-checked claim.
    pub similarity: f64,
    /// Match method used.
    pub method: MatchMethod,
    pub matched_at: DateTime<Utc>,
}

/// Method used to match a claim to a fact-check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchMethod {
    /// Exact or near-exact text match.
    ExactText,
    /// Embedding cosine similarity.
    Embedding,
    /// Keyword overlap with reranking.
    KeywordRerank,
}

/// A mutation event tracking how a claim changed over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationEvent {
    pub id: Uuid,
    /// The original claim UUID.
    pub original_claim_id: Uuid,
    /// The mutated claim UUID.
    pub mutated_claim_id: Uuid,
    /// Embedding distance between original and mutated versions.
    pub embedding_drift: f64,
    /// Text edit distance (normalized Levenshtein).
    pub text_distance: f64,
    /// What changed.
    pub mutation_type: MutationType,
    pub detected_at: DateTime<Utc>,
}

/// Type of mutation observed in a claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationType {
    /// Minor wording changes, same meaning.
    Paraphrase,
    /// Key details added or removed.
    DetailShift,
    /// Numbers or statistics changed.
    NumericalShift,
    /// Attribution changed (who said it).
    AttributionShift,
    /// Context or framing changed.
    ContextShift,
    /// Major semantic change.
    SemanticDrift,
}

/// Result of a claim origin trace -- walking back through the hypergraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimOriginTrace {
    pub claim_id: Uuid,
    /// Chain of claim appearances from most recent to earliest.
    pub chain: Vec<ClaimAppearance>,
    /// The earliest detected appearance.
    pub earliest: Option<ClaimAppearance>,
}

/// A single appearance of a claim in the temporal chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimAppearance {
    pub claim_id: Uuid,
    pub situation_id: Option<Uuid>,
    pub entity_id: Option<Uuid>,
    pub timestamp: Option<DateTime<Utc>>,
    /// Confidence that this is the same claim (cosine similarity).
    pub similarity_to_original: f64,
}
