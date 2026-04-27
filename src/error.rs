use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum TensaError {
    // Store errors
    #[error("Storage error: {0}")]
    Store(String),

    #[error("Key not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Transaction failed: {0}")]
    Transaction(String),

    // Hypergraph errors
    #[error("Entity not found: {0}")]
    EntityNotFound(Uuid),

    #[error("Situation not found: {0}")]
    SituationNotFound(Uuid),

    #[error("Participation already exists: entity {entity_id} in situation {situation_id}")]
    ParticipationExists { entity_id: Uuid, situation_id: Uuid },

    #[error("Participation not found: entity {entity_id} in situation {situation_id}")]
    ParticipationNotFound { entity_id: Uuid, situation_id: Uuid },

    #[error("Causal link would create cycle: {from} -> {to}")]
    CausalCycle { from: Uuid, to: Uuid },

    #[error("Invalid maturity transition: {from:?} -> {to:?}")]
    InvalidMaturityTransition {
        from: crate::types::MaturityLevel,
        to: crate::types::MaturityLevel,
    },

    // Temporal errors
    #[error("Invalid temporal interval: {0}")]
    InvalidInterval(String),

    #[error("Temporal relation inconsistency: {0}")]
    TemporalInconsistency(String),

    // Query errors
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Query execution error: {0}")]
    QueryError(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    // Inference errors
    #[error("Job not found: {0}")]
    JobNotFound(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    // Narrative errors (Phase 3)
    #[error("Narrative not found: {0}")]
    NarrativeNotFound(String),

    #[error("Narrative already exists: {0}")]
    NarrativeExists(String),

    // Source intelligence errors (Phase 4)
    #[error("Source not found: {0}")]
    SourceNotFound(Uuid),

    #[error("Source already exists: {0}")]
    SourceExists(Uuid),

    #[error("Source attribution already exists: source {source_id} -> {target_id}")]
    AttributionExists { source_id: Uuid, target_id: Uuid },

    #[error("Contention link already exists: {situation_a} <-> {situation_b}")]
    ContentionExists {
        situation_a: Uuid,
        situation_b: Uuid,
    },

    #[error("Contention link not found: {situation_a} <-> {situation_b}")]
    ContentionNotFound {
        situation_a: Uuid,
        situation_b: Uuid,
    },

    // Taxonomy errors
    #[error("Taxonomy entry already exists: {0}/{1}")]
    TaxonomyEntryExists(String, String),

    #[error("Cannot remove builtin taxonomy entry: {0}/{1}")]
    TaxonomyBuiltinRemoval(String, String),

    // Narrative merge errors
    #[error("Narrative merge error: {0}")]
    NarrativeMergeError(String),

    // Chunk errors
    #[error("Chunk not found: {0}")]
    ChunkNotFound(Uuid),

    // Ingestion errors (Phase 1)
    #[error("LLM API error: {0}")]
    LlmError(String),

    #[error("LLM rate limit exceeded, retry after {retry_after_secs}s")]
    LlmRateLimit { retry_after_secs: u64 },

    #[error("Extraction parse error: {0}")]
    ExtractionError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Ingestion error: {0}")]
    IngestionError(String),

    #[error("Validation queue error: {0}")]
    ValidationQueueError(String),

    // Import/Export errors (Sprint P3.6)
    #[error("Import error: {0}")]
    ImportError(String),

    #[error("Export error: {0}")]
    ExportError(String),

    #[error("Document parse error: {0}")]
    DocParseError(String),

    // Synthetic generation (EATH sprint)
    #[error("Synthetic generation failure: {0}")]
    SynthFailure(String),

    // Generic
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, TensaError>;

// Conversion from bincode errors
impl From<bincode::Error> for TensaError {
    fn from(e: bincode::Error) -> Self {
        TensaError::Serialization(e.to_string())
    }
}

// Conversion from serde_json errors
impl From<serde_json::Error> for TensaError {
    fn from(e: serde_json::Error) -> Self {
        TensaError::Serialization(e.to_string())
    }
}
