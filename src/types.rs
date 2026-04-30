use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

use crate::source::ConfidenceBreakdown;

/// `skip_serializing_if` predicate for `serde_json::Value` fields that default
/// to null — keeps the on-disk representation small for records that don't use
/// the `properties` slot (e.g. non-synthetic Situations).
fn is_null_value(v: &serde_json::Value) -> bool {
    v.is_null()
}

/// Deserialize a field that may be a string or an array of strings.
/// Arrays are joined with "; ". Null/missing produces None.
fn deserialize_string_or_array<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de;

    struct StringOrArray;

    impl<'de> de::Visitor<'de> for StringOrArray {
        type Value = Option<String>;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string, an array of strings, or null")
        }

        fn visit_none<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
            Ok(Some(v.to_owned()))
        }

        fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
            Ok(Some(v))
        }

        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut parts = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                parts.push(s);
            }
            if parts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(parts.join("; ")))
            }
        }
    }

    deserializer.deserialize_any(StringOrArray)
}

// ─── Entity Types ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Actor,
    Location,
    Artifact,
    Concept,
    Organization,
}

/// All entity type variants, for iteration.
pub const ALL_ENTITY_TYPES: [EntityType; 5] = [
    EntityType::Actor,
    EntityType::Location,
    EntityType::Artifact,
    EntityType::Concept,
    EntityType::Organization,
];

impl EntityType {
    /// Stable string tag used for secondary index keys.
    pub fn as_index_str(&self) -> &'static str {
        match self {
            Self::Actor => "Actor",
            Self::Location => "Location",
            Self::Artifact => "Artifact",
            Self::Concept => "Concept",
            Self::Organization => "Organization",
        }
    }
}

impl std::str::FromStr for EntityType {
    type Err = crate::error::TensaError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s.to_lowercase().as_str() {
            "actor" => Ok(Self::Actor),
            "location" => Ok(Self::Location),
            "artifact" => Ok(Self::Artifact),
            "concept" => Ok(Self::Concept),
            "organization" => Ok(Self::Organization),
            other => Err(crate::error::TensaError::InvalidQuery(format!(
                "Unknown entity type: '{}'. Use Actor, Location, Artifact, Concept, or Organization.",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub entity_type: EntityType,
    pub properties: serde_json::Value,
    pub beliefs: Option<serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub maturity: MaturityLevel,
    pub confidence: f32,
    #[serde(default)]
    pub confidence_breakdown: Option<ConfidenceBreakdown>,
    pub provenance: Vec<SourceReference>,
    #[serde(default)]
    pub extraction_method: Option<ExtractionMethod>,
    #[serde(default)]
    pub narrative_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub deleted_at: Option<DateTime<Utc>>,
    /// Transaction time: when this version of the entity was recorded in the system.
    /// Enables bi-temporal queries: "what did we know about X at time T?"
    /// `None` means "current" (pre-bi-temporal data).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction_time: Option<DateTime<Utc>>,
}

impl Entity {
    /// Apply a JSON patch to mutable fields (properties, confidence, narrative_id).
    pub fn apply_patch(&mut self, updates: &serde_json::Value) {
        if let Some(obj) = updates.get("properties").and_then(|v| v.as_object()) {
            if let Some(existing) = self.properties.as_object_mut() {
                for (k, v) in obj {
                    existing.insert(k.clone(), v.clone());
                }
            }
        }
        if let Some(confidence) = updates.get("confidence").and_then(|v| v.as_f64()) {
            self.confidence = confidence as f32;
        }
        if let Some(narrative_id) = updates.get("narrative_id") {
            self.narrative_id = narrative_id.as_str().map(String::from);
        }
        self.updated_at = chrono::Utc::now();
    }
}

// ─── Situation Types ──────────────────────────────────────────

/// Position of a situation in the original source text.
/// Computed during ingestion from chunk metadata + LLM array position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    /// Which chunk this situation came from (0-based).
    pub chunk_index: u32,
    /// Byte offset of the chunk's start in the full source text.
    pub byte_offset_start: usize,
    /// Byte offset of the chunk's end in the full source text.
    pub byte_offset_end: usize,
    /// Situation's index within its chunk (0-based, based on LLM array order).
    pub local_index: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Situation {
    pub id: Uuid,
    /// Short descriptive name (5-8 words), e.g. "Bran's fall from the tower".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 1-2 sentence summary of what happens in this situation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Free-form structured metadata. Used by the synthetic-generation pipeline
    /// (EATH Phase 0+) to mark synthetic situations with `{synthetic: true,
    /// synth_run_id, synth_model, synth_step}`. Backward-compatible via
    /// `#[serde(default)]` so existing on-disk situations deserialize cleanly
    /// (older records simply get `Value::Null`).
    #[serde(default, skip_serializing_if = "is_null_value")]
    pub properties: serde_json::Value,
    pub temporal: AllenInterval,
    pub spatial: Option<SpatialAnchor>,
    pub game_structure: Option<GameStructure>,
    pub causes: Vec<CausalLink>,
    pub deterministic: Option<serde_json::Value>,
    pub probabilistic: Option<serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub raw_content: Vec<ContentBlock>,
    pub narrative_level: NarrativeLevel,
    pub discourse: Option<DiscourseAnnotation>,
    pub maturity: MaturityLevel,
    pub confidence: f32,
    #[serde(default)]
    pub confidence_breakdown: Option<ConfidenceBreakdown>,
    pub extraction_method: ExtractionMethod,
    #[serde(default)]
    pub provenance: Vec<SourceReference>,
    #[serde(default)]
    pub narrative_id: Option<String>,
    /// UUID of the source chunk that produced this situation during ingestion.
    #[serde(default)]
    pub source_chunk_id: Option<Uuid>,
    /// Approximate position of this situation in the original source text.
    #[serde(default)]
    pub source_span: Option<SourceSpan>,
    /// Writer-authored one-line card summary (for corkboard / outliner views).
    /// Distinct from `description` (extracted at ingest) and `name` (short title).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synopsis: Option<String>,
    /// Writer-curated order for manuscript compile (binder order).
    /// Separate from `temporal.start` so that narrated order can diverge from chronology.
    /// When present, manuscript export sorts by this field; when absent, falls back to time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manuscript_order: Option<u32>,
    /// Parent scene id for binder hierarchy (Part → Chapter → Scene).
    /// Cycles are rejected by `update_situation`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_situation_id: Option<Uuid>,
    /// Free-text writer label for colour-coding (e.g. "Draft", "Revise", "Cut").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Free-text workflow status (e.g. "first-draft", "revised", "final").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Writer keywords / tags (free-text, used by Outliner filter + Collections).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub deleted_at: Option<DateTime<Utc>>,
    /// Transaction time: when this version of the situation was recorded in the system.
    /// `None` means "current" (pre-bi-temporal data).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction_time: Option<DateTime<Utc>>,
}

// ─── Participation ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participation {
    pub entity_id: Uuid,
    pub situation_id: Uuid,
    pub role: Role,
    pub info_set: Option<InfoSet>,
    /// Action description. Accepts both a single string and an array of strings
    /// (joined with "; ") for compatibility with externally-created archives.
    #[serde(default, deserialize_with = "deserialize_string_or_array")]
    pub action: Option<String>,
    pub payoff: Option<serde_json::Value>,
    /// Sequence number for multi-role support. Auto-assigned by `add_participant`.
    #[serde(default)]
    pub seq: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    Protagonist,
    Antagonist,
    Witness,
    Target,
    Instrument,
    Confidant,
    Informant,
    Recipient,
    Bystander,
    SubjectOfDiscussion, // entity is discussed but may not be present
    Facilitator, // enables protagonist's action without being its agent (gatekeepers, keepers, intermediaries)
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoSet {
    pub knows_before: Vec<KnowledgeFact>,
    pub learns: Vec<KnowledgeFact>,
    pub reveals: Vec<KnowledgeFact>,
    /// Depth-2 recursive beliefs: what this entity thinks others know.
    #[serde(default)]
    pub beliefs_about_others: Vec<RecursiveBelief>,
}

/// A recursive belief: what entity A thinks entity B knows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveBelief {
    /// The entity this belief is about.
    pub about_entity: Uuid,
    /// What A thinks this entity knows.
    pub believed_knowledge: Vec<KnowledgeFact>,
    /// Confidence in this belief model.
    pub confidence: f32,
    /// Situation ID when this belief was last updated.
    pub last_updated_at: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeFact {
    pub about_entity: Uuid,
    pub fact: String,
    pub confidence: f32,
}

// ─── State Versioning ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVersion {
    pub entity_id: Uuid,
    pub situation_id: Uuid,
    pub properties: serde_json::Value,
    pub beliefs: Option<serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub timestamp: DateTime<Utc>,
}

// ─── Temporal ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllenInterval {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    pub granularity: TimeGranularity,
    pub relations: Vec<AllenRelationTo>,
    /// Fuzzy Sprint Phase 5 — trapezoidal fuzzy endpoints for situations whose
    /// temporal boundaries are fuzzy sets ("early 2024", "around that time",
    /// "shortly after") rather than crisp timestamps. `#[serde(default)]` keeps
    /// every existing serialized situation deserializing unchanged; when
    /// absent, `graded_relation` falls back to the crisp Allen path
    /// (bit-identical to the pre-Phase-5 semantics).
    ///
    /// Cites: [duboisprade1989fuzzyallen] [schockaert2008fuzzyallen].
    #[serde(default)]
    pub fuzzy_endpoints: Option<crate::fuzzy::allen::FuzzyEndpoints>,
}

impl AllenInterval {
    /// Fuzzy Sprint Phase 5 — true iff this interval carries trapezoidal fuzzy
    /// endpoints that graded-Allen consumers should use instead of the crisp
    /// `start` / `end` timestamps.
    pub fn has_fuzzy_endpoints(&self) -> bool {
        self.fuzzy_endpoints.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllenRelationTo {
    pub target_situation: Uuid,
    pub relation: AllenRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AllenRelation {
    Before,
    After,
    Meets,
    MetBy,
    Overlaps,
    OverlappedBy,
    During,
    Contains,
    Starts,
    StartedBy,
    Finishes,
    FinishedBy,
    Equals,
}

impl std::str::FromStr for AllenRelation {
    type Err = crate::error::TensaError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s {
            "Before" => Ok(Self::Before),
            "After" => Ok(Self::After),
            "Meets" => Ok(Self::Meets),
            "MetBy" => Ok(Self::MetBy),
            "Overlaps" => Ok(Self::Overlaps),
            "OverlappedBy" => Ok(Self::OverlappedBy),
            "During" => Ok(Self::During),
            "Contains" => Ok(Self::Contains),
            "Starts" => Ok(Self::Starts),
            "StartedBy" => Ok(Self::StartedBy),
            "Finishes" => Ok(Self::Finishes),
            "FinishedBy" => Ok(Self::FinishedBy),
            "Equals" => Ok(Self::Equals),
            other => Err(crate::error::TensaError::InvalidQuery(format!(
                "Unknown Allen relation: '{}'",
                other
            ))),
        }
    }
}

impl AllenRelation {
    /// Return the inverse relation
    pub fn inverse(&self) -> Self {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
            Self::Meets => Self::MetBy,
            Self::MetBy => Self::Meets,
            Self::Overlaps => Self::OverlappedBy,
            Self::OverlappedBy => Self::Overlaps,
            Self::During => Self::Contains,
            Self::Contains => Self::During,
            Self::Starts => Self::StartedBy,
            Self::StartedBy => Self::Starts,
            Self::Finishes => Self::FinishedBy,
            Self::FinishedBy => Self::Finishes,
            Self::Equals => Self::Equals,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeGranularity {
    Exact,
    Day,
    Approximate,
    Unknown,
    /// Marker for synthetically-generated situations (EATH Phase 0+).
    /// Synthetic times are arbitrary step counters from the generator, not
    /// real wall-clock observations — downstream consumers (e.g. the manuscript
    /// exporter) should treat them as relative ordering, not absolute time.
    Synthetic,
}

// ─── Spatial ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialAnchor {
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub precision: SpatialPrecision,
    pub location_entity: Option<Uuid>,
    /// Free-text location name (e.g. "London", "Reg's rooms, Second Court").
    /// Accepted by archive import as an alternative to lat/long coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location_name: Option<String>,
    pub description: Option<String>,
}

impl Default for SpatialAnchor {
    fn default() -> Self {
        Self {
            latitude: None,
            longitude: None,
            precision: SpatialPrecision::Approximate,
            location_entity: None,
            location_name: None,
            description: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialPrecision {
    Exact,
    Area,
    Region,
    Approximate,
    Unknown,
}

// ─── Causality ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    pub from_situation: Uuid,
    pub to_situation: Uuid,
    pub mechanism: Option<String>,
    pub strength: f32,
    pub causal_type: CausalType,
    pub maturity: MaturityLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalType {
    Necessary,
    Sufficient,
    Contributing,
    Enabling,
}

// ─── Game Theory ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStructure {
    pub game_type: GameClassification,
    pub info_structure: InfoStructureType,
    pub description: Option<String>,
    pub maturity: MaturityLevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameClassification {
    PrisonersDilemma,
    Coordination,
    Signaling,
    Auction,
    Bargaining,
    ZeroSum,
    AsymmetricInformation,
    Custom(String),
}

impl std::str::FromStr for GameClassification {
    type Err = crate::error::TensaError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s {
            "PrisonersDilemma" => Ok(Self::PrisonersDilemma),
            "Coordination" => Ok(Self::Coordination),
            "Signaling" => Ok(Self::Signaling),
            "Auction" => Ok(Self::Auction),
            "Bargaining" => Ok(Self::Bargaining),
            "ZeroSum" => Ok(Self::ZeroSum),
            "AsymmetricInformation" => Ok(Self::AsymmetricInformation),
            other => Ok(Self::Custom(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InfoStructureType {
    Complete,
    Incomplete,
    Imperfect,
    AsymmetricBecomingComplete,
    Custom(String),
}

impl std::str::FromStr for InfoStructureType {
    type Err = crate::error::TensaError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s {
            "Complete" => Ok(Self::Complete),
            "Incomplete" => Ok(Self::Incomplete),
            "Imperfect" => Ok(Self::Imperfect),
            "AsymmetricBecomingComplete" => Ok(Self::AsymmetricBecomingComplete),
            other => Ok(Self::Custom(other.to_string())),
        }
    }
}

// ─── Narrative ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NarrativeLevel {
    Story,
    Arc,
    Sequence,
    Scene,
    Beat,
    Event,
}

impl NarrativeLevel {
    /// Stable string tag used for secondary index keys.
    pub fn as_index_str(&self) -> &'static str {
        match self {
            Self::Story => "Story",
            Self::Arc => "Arc",
            Self::Sequence => "Sequence",
            Self::Scene => "Scene",
            Self::Beat => "Beat",
            Self::Event => "Event",
        }
    }

    /// Numeric ordinal for feature extraction (coarser → finer granularity).
    pub fn ordinal(&self) -> usize {
        match self {
            Self::Story => 0,
            Self::Arc => 1,
            Self::Sequence => 2,
            Self::Scene => 3,
            Self::Beat => 4,
            Self::Event => 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseAnnotation {
    pub order: Option<String>,      // analepsis, prolepsis, simultaneous
    pub duration: Option<String>,   // scene, summary, ellipsis, pause, stretch
    pub focalization: Option<Uuid>, // whose perspective
    pub voice: Option<String>,      // homodiegetic, heterodiegetic
}

// ─── Content ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    pub content_type: ContentType,
    pub content: String,
    pub source: Option<SourceReference>,
}

impl ContentBlock {
    pub fn text(s: &str) -> Self {
        Self {
            content_type: ContentType::Text,
            content: s.to_string(),
            source: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Text,
    Dialogue,
    Observation,
    Document,
    MediaRef,
}

// ─── Data Quality ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaturityLevel {
    Candidate = 0,
    Reviewed = 1,
    Validated = 2,
    GroundTruth = 3,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionMethod {
    LlmParsed,
    HumanEntered,
    StructuredImport,
    Sensor,
    Simulated,
    /// Generated by a registered surrogate model (EATH Phase 0+). Carries the
    /// model name (`"eath"`, future: `"had"`, `"hyperedge-config"`, ...) and
    /// the run UUID so consumers can filter / cross-reference with
    /// `syn/r/{narrative}/{run_id}` records and `ReproducibilityBlob`s.
    Synthetic { model: String, run_id: Uuid },
    /// Inferred via SINDy hypergraph reconstruction (EATH Extension Phase 15).
    /// Carries the source narrative the time-series was observed on plus the
    /// reconstruction job id so consumers can cross-reference with
    /// `ir/{job_id}` and the run's `ReconstructionResult` blob.
    Reconstructed {
        source_narrative_id: String,
        job_id: String,
    },
}

// ─── Provenance ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    pub source_type: String,
    pub source_id: Option<String>,
    pub description: Option<String>,
    pub timestamp: DateTime<Utc>,
    /// Optional link to a registered Source record.
    #[serde(default)]
    pub registered_source: Option<Uuid>,
}

// ─── Inference ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub job_id: String,
    pub job_type: InferenceJobType,
    pub target_id: Uuid,
    pub result: serde_json::Value,
    pub confidence: f32,
    #[serde(default)]
    pub explanation: Option<String>,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceJobType {
    MotivationInference,
    CausalDiscovery,
    Counterfactual,
    GameClassification,
    MissingLinks,
    AnomalyDetection,
    // Phase 3: Cross-narrative
    PatternMining,
    ArcClassification,
    /// Per-actor Reagan arc classification — one ArcClassification per Actor
    /// entity in a narrative, keyed by actor_id. Addresses the narrative-level
    /// averaging blind spot in multi-protagonist stories (v0.74.1).
    ActorArcClassification,
    MissingEventPrediction,
    // Phase 4: Analysis
    CentralityAnalysis,
    EntropyAnalysis,
    BeliefModeling,
    EvidenceCombination,
    ArgumentationAnalysis,
    ContagionAnalysis,
    // Narrative Fingerprint (Stylometry + Style Profile)
    StyleProfile,
    StyleComparison,
    StyleAnomaly,
    // PAN@CLEF authorship verification (v0.28)
    AuthorshipVerification,
    // Temporal Correlation Graph anomaly detection
    TCGAnomaly,
    // Hawkes process event prediction
    NextEvent,
    // Temporal Inductive Logic Programming
    TemporalILP,
    // Mean Field Game Theory
    MeanFieldGame,
    // Probabilistic Soft Logic
    ProbabilisticSoftLogic,
    // Trajectory embedding (TGN-style temporal entity embeddings)
    TrajectoryEmbedding,
    // Narrative simulation (generative agent forward-play)
    NarrativeSimulation,
    // Sprint 1: Core graph centrality algorithms
    PageRank,
    EigenvectorCentrality,
    HarmonicCentrality,
    HITS,
    // Sprint 2: Topology & community
    Topology,
    LabelPropagation,
    KCore,
    // Sprint 7: Narrative-native algorithms
    TemporalPageRank,
    CausalInfluence,
    InfoBottleneck,
    Assortativity,
    // Sprint 8: Temporal patterns & faction evolution
    TemporalMotifs,
    FactionEvolution,
    // Sprint 11: Graph embeddings & network inference
    FastRP,
    Node2Vec,
    NetworkInference,
    // Disinfo Sprint D1: dual fingerprints
    BehavioralFingerprint,
    DisinfoFingerprint,
    // Disinfo Sprint D2: spread dynamics
    /// SMIR + per-platform R₀ + cross-platform jumps + velocity-monitor alert.
    SpreadVelocity,
    /// Counterfactual intervention projection (RemoveTopAmplifiers / DebunkAt).
    SpreadIntervention,
    // Disinfo Sprint D3: CIB detection
    /// Coordinated Inauthentic Behavior cluster detection (behavioral similarity
    /// network → community detection → calibrated p-value density threshold).
    CibDetection,
    /// Superspreader ranking via graph centrality (pagerank/betweenness/eigenvector).
    Superspreaders,
    // Disinfo Sprint D4: claims & fact-check pipeline
    /// Trace a claim back to its earliest appearance in the temporal chain.
    ClaimOrigin,
    /// Match claims in a narrative against known fact-checks.
    ClaimMatch,
    // Disinfo Sprint D5: archetypes + fusion
    /// Actor archetype classification via behavioral fingerprint template matching.
    ArchetypeClassification,
    /// Dempster-Shafer fusion of multiple disinfo signals.
    DisinfoAssessment,
    // Sprint D9: Narrative architecture & generative engine
    /// Detect narrative commitments (Chekhov's guns, foreshadowing, etc.)
    CommitmentDetection,
    /// Extract chronological fabula ordering from Allen constraints.
    FabulaExtraction,
    /// Extract discourse-order sjužet from chapter structure.
    SjuzetExtraction,
    /// Compute dramatic irony map (reader vs character knowledge gaps).
    DramaticIrony,
    /// Detect focalization (POV) segments across narrative.
    Focalization,
    /// Detect character arc type and transformation trajectory.
    CharacterArc,
    /// Detect subplots via community detection on situation graph.
    SubplotDetection,
    /// Classify scene-sequel rhythm (Swain/Bickham).
    SceneSequel,
    /// Suggest sjužet reorderings for dramatic effect.
    SjuzetReordering,
    /// Community summarization (LLM-driven naming + theming of detected
    /// communities). The actual generation runs via the
    /// `/narratives/:id/communities/summarize` endpoint, but the registry
    /// row records that the analysis was attested.
    CommunitySummary,
    /// Structural narrative-linter pass (orphans, knowledge gaps, pacing,
    /// commitment failures, etc.). Runs via the `diagnose_narrative` MCP
    /// tool / `/narratives/:id/diagnose` endpoint.
    NarrativeDiagnose,
    // Sprint D12: Adversarial narrative wargaming
    /// Generate adversary policy from IRL reward weights + SUQR rationality.
    AdversaryPolicy,
    /// Cognitive Hierarchy level-k best response.
    CognitiveHierarchy,
    /// Run wargame simulation (turn-based red/blue).
    WargameSimulation,
    /// Compute psychological reward fingerprint for a narrative.
    RewardFingerprint,
    /// Generate reward-aware counter-narratives.
    CounterNarrative,
    /// Run retrodiction against historical campaign data.
    Retrodiction,
    /// Generate a chapter via the fitness-driven loop: prompt with SE + target
    /// fingerprint conditioning, score generated prose, revise toward target,
    /// return best-of-N attempts.
    ChapterGenerationFitness,
    // ── EATH Synthetic Generation Sprint (Phase 0+) ───────────────────────
    // These four variants are added in one bundled exhaustive-match sweep so
    // later phases (1 → 7b) don't trigger four separate sweeps. Engines for
    // SurrogateSignificance + SurrogateContagionSignificance are stubbed
    // (return `TensaError::SynthFailure("not yet implemented in this phase")`)
    // until Phases 7 and 7b respectively.
    /// Calibrate a registered surrogate model (e.g. EATH) against a real
    /// narrative — produces `SurrogateParams` persisted at `syn/p/{nid}/{model}`.
    SurrogateCalibration {
        narrative_id: String,
        model: String,
    },
    /// Generate a synthetic narrative via a registered surrogate model.
    /// `params_override` lets callers swap fitted params for hand-tuned ones
    /// without re-calibrating; `seed_override` enables deterministic replay.
    SurrogateGeneration {
        source_narrative_id: Option<String>,
        output_narrative_id: String,
        model: String,
        params_override: Option<serde_json::Value>,
        seed_override: Option<u64>,
    },
    /// Statistical significance of a metric (motifs, communities, patterns)
    /// vs `k` synthetic null-model runs. Phase 7 implements the engine.
    SurrogateSignificance {
        narrative_id: String,
        metric_kind: String,
        k: u16,
        model: String,
    },
    /// Significance of contagion-spread metrics vs synthetic null-model runs.
    /// Phase 7b implements the engine.
    SurrogateContagionSignificance {
        narrative_id: String,
        k: u16,
        model: String,
        contagion_params: serde_json::Value,
    },
    /// Generate a synthetic narrative as a weighted **mixture** of `n` calibrated
    /// surrogate processes (EATH Phase 9 hybrid hypergraphs). Distinct from
    /// [`InferenceJobType::SurrogateGeneration`] — the engine multiplexes between
    /// per-source recruitment helpers rather than running ONE EATH process.
    /// `components` carries an array of
    /// `{ narrative_id, model, weight }` objects whose weights must sum to 1.0
    /// within `1e-6` (validated at engine execution). Stored as
    /// `serde_json::Value` because `crate::synth::hybrid::HybridComponent`
    /// lives in a layer above `types`, and adding a leaf dependency the
    /// other direction would invert the architectural rule.
    /// See [`crate::synth::hybrid`].
    SurrogateHybridGeneration {
        components: serde_json::Value,
        output_narrative_id: String,
        seed_override: Option<u64>,
        num_steps: Option<usize>,
    },
    /// EATH Extension Phase 13c — dual-null-model significance.
    ///
    /// Runs the Phase 7 K-loop ONCE per requested null model and reports the
    /// per-model + combined significance verdict. `models` defaults to
    /// `["eath", "nudhy"]` at the API/grammar layer; the engine validates
    /// each name against the registry and rejects unknowns. `metric` is a
    /// string for grammar agnosticism — engine maps it via
    /// [`crate::synth::significance::SignificanceMetric::parse`].
    SurrogateDualSignificance {
        narrative_id: String,
        metric: String,
        k_per_model: u16,
        models: Vec<String>,
    },
    /// EATH Extension Phase 14 — bistability/hysteresis significance.
    ///
    /// Runs a forward-backward β-sweep on the source narrative AND on K
    /// surrogate samples (per requested null model), then reports per-model
    /// quantiles for the bistable_interval width and max_hysteresis_gap.
    /// `params` carries a serialized
    /// [`crate::analysis::contagion_bistability::BistabilitySweepParams`]
    /// blob; `models` defaults to `["eath", "nudhy"]` at the API/grammar
    /// layer.
    SurrogateBistabilitySignificance {
        narrative_id: String,
        params: serde_json::Value,
        k: u16,
        models: Vec<String>,
    },
    /// EATH Extension Phase 15 — SINDy hypergraph reconstruction from dynamics.
    ///
    /// Infers latent hyperedges from per-entity time-series observations.
    /// `params` carries a serialized
    /// [`crate::inference::hypergraph_reconstruction::ReconstructionParams`]
    /// blob; the engine deserializes it on execution.
    ///
    /// Reference: Delabays, De Pasquale, Dörfler, Zhang — Nat. Commun. 16,
    /// 2691 (2025), arXiv:2402.00078.
    HypergraphReconstruction {
        narrative_id: String,
        params: serde_json::Value,
    },
    /// EATH Extension Phase 16c — opinion-dynamics-significance against
    /// K surrogate runs from each requested null model. Compares observed
    /// num_clusters / polarization_index / echo_chamber_index of one
    /// opinion-dynamics simulation against per-model surrogate distributions.
    ///
    /// `params` carries a serialized
    /// [`crate::analysis::opinion_dynamics::OpinionDynamicsParams`] blob;
    /// engine reuses Phase 13c's per-model `std::thread::scope` parallelism.
    /// `models` defaults to `["eath", "nudhy"]` at the API/grammar layer.
    /// Reference: Hickok et al. SIAM J. Appl. Dyn. Syst. 21:1 (2022).
    SurrogateOpinionSignificance {
        narrative_id: String,
        params: serde_json::Value,
        k: u16,
        models: Vec<String>,
    },
    // ── Fuzzy Logic Sprint (Phase 0+) ───────────────────────────────────────
    // Seven variants added in one exhaustive-match sweep so later phases
    // (1 → 10) don't trigger seven separate sweeps through the cost
    // estimator + downstream analysis sites. Engines for all seven return
    // `TensaError::InvalidInput("not yet implemented in this phase")` until
    // the owning phase (1 → 10) replaces them. `config` / `params` fields
    // carry `serde_json::Value` so the variant shapes can stabilise ahead
    // of the concrete Phase 2 / 9 / 10 payload types landing.
    /// Aggregate confidence values over a target entity or situation under a
    /// specified fuzzy configuration (t-norm + aggregator). Phase 2 ships
    /// the engine.
    FuzzyAggregate {
        narrative_id: String,
        target_id: Uuid,
        config: serde_json::Value,
    },
    /// Compute the graded 13-vector of Allen relations for a situation pair.
    /// Phase 5 ships the engine. Cites: [duboisprade1989fuzzyallen]
    /// [schockaert2008fuzzyallen].
    FuzzyAllenGradation {
        narrative_id: String,
        situation_pair: (Uuid, Uuid),
    },
    /// Evaluate an intermediate quantifier (Novák–Murinová) against a
    /// fuzzy predicate on a narrative's entities / situations. Phase 6
    /// ships the engine. Cites: [novak2008quantifiers].
    FuzzyQuantifierEvaluate {
        narrative_id: String,
        predicate: String,
        quantifier: String,
    },
    /// Verify a graded syllogism from a set of premises under Peterson's
    /// extended square of opposition. Phase 7 ships the engine.
    FuzzySyllogismVerify {
        narrative_id: String,
        premises: Vec<String>,
        conclusion: String,
    },
    /// Compute the fuzzy formal-concept lattice for a narrative's
    /// entity/attribute matrix. Phase 8 ships the engine.
    /// Cites: [belohlavek2004fuzzyfca].
    FuzzyFcaLattice {
        narrative_id: String,
        attribute_filter: Option<Vec<String>>,
    },
    /// Evaluate a Mamdani fuzzy rule against an optional target entity.
    /// Phase 9 ships the engine. Cites: [mamdani1975mamdani].
    FuzzyRuleEvaluate {
        narrative_id: String,
        rule_id: String,
        entity_id: Option<Uuid>,
    },
    /// Hybrid fuzzy-probabilistic inference — Flaminio-style probability
    /// of a fuzzy event under a discrete distribution. Phase 10 ships
    /// the engine. Cites: [flaminio2026fsta].
    FuzzyHybridInference {
        narrative_id: String,
        config: serde_json::Value,
    },
}

impl InferenceJobType {
    /// Stable variant name (e.g. `"ArcClassification"`, `"SurrogateCalibration"`)
    /// regardless of whether the variant carries a payload. Used as a KV-key
    /// segment by the analysis-status registry so that re-running a payload
    /// variant for the same narrative writes to the same row.
    pub fn variant_name(&self) -> &'static str {
        // serde-derived JSON gives `"Foo"` for unit variants and `{"Foo": ...}` for
        // payload-carrying ones. We normalize to the variant name in both cases.
        // Deriving a static slice via `serde_json` would be a fresh allocation, so
        // we use a hand-rolled discriminant table — keep alphabetized for diffing.
        use InferenceJobType::*;
        match self {
            MotivationInference => "MotivationInference",
            CausalDiscovery => "CausalDiscovery",
            Counterfactual => "Counterfactual",
            GameClassification => "GameClassification",
            MissingLinks => "MissingLinks",
            AnomalyDetection => "AnomalyDetection",
            PatternMining => "PatternMining",
            ArcClassification => "ArcClassification",
            ActorArcClassification => "ActorArcClassification",
            MissingEventPrediction => "MissingEventPrediction",
            CentralityAnalysis => "CentralityAnalysis",
            EntropyAnalysis => "EntropyAnalysis",
            BeliefModeling => "BeliefModeling",
            EvidenceCombination => "EvidenceCombination",
            ArgumentationAnalysis => "ArgumentationAnalysis",
            ContagionAnalysis => "ContagionAnalysis",
            StyleProfile => "StyleProfile",
            StyleComparison => "StyleComparison",
            StyleAnomaly => "StyleAnomaly",
            AuthorshipVerification => "AuthorshipVerification",
            TCGAnomaly => "TCGAnomaly",
            NextEvent => "NextEvent",
            TemporalILP => "TemporalILP",
            MeanFieldGame => "MeanFieldGame",
            ProbabilisticSoftLogic => "ProbabilisticSoftLogic",
            TrajectoryEmbedding => "TrajectoryEmbedding",
            NarrativeSimulation => "NarrativeSimulation",
            PageRank => "PageRank",
            EigenvectorCentrality => "EigenvectorCentrality",
            HarmonicCentrality => "HarmonicCentrality",
            HITS => "HITS",
            Topology => "Topology",
            LabelPropagation => "LabelPropagation",
            KCore => "KCore",
            TemporalPageRank => "TemporalPageRank",
            CausalInfluence => "CausalInfluence",
            InfoBottleneck => "InfoBottleneck",
            Assortativity => "Assortativity",
            TemporalMotifs => "TemporalMotifs",
            FactionEvolution => "FactionEvolution",
            FastRP => "FastRP",
            Node2Vec => "Node2Vec",
            NetworkInference => "NetworkInference",
            BehavioralFingerprint => "BehavioralFingerprint",
            DisinfoFingerprint => "DisinfoFingerprint",
            SpreadVelocity => "SpreadVelocity",
            SpreadIntervention => "SpreadIntervention",
            CibDetection => "CibDetection",
            Superspreaders => "Superspreaders",
            ClaimOrigin => "ClaimOrigin",
            ClaimMatch => "ClaimMatch",
            ArchetypeClassification => "ArchetypeClassification",
            DisinfoAssessment => "DisinfoAssessment",
            CommitmentDetection => "CommitmentDetection",
            FabulaExtraction => "FabulaExtraction",
            SjuzetExtraction => "SjuzetExtraction",
            DramaticIrony => "DramaticIrony",
            Focalization => "Focalization",
            CharacterArc => "CharacterArc",
            SubplotDetection => "SubplotDetection",
            SceneSequel => "SceneSequel",
            SjuzetReordering => "SjuzetReordering",
            CommunitySummary => "CommunitySummary",
            NarrativeDiagnose => "NarrativeDiagnose",
            AdversaryPolicy => "AdversaryPolicy",
            CognitiveHierarchy => "CognitiveHierarchy",
            WargameSimulation => "WargameSimulation",
            RewardFingerprint => "RewardFingerprint",
            CounterNarrative => "CounterNarrative",
            Retrodiction => "Retrodiction",
            ChapterGenerationFitness => "ChapterGenerationFitness",
            SurrogateCalibration { .. } => "SurrogateCalibration",
            SurrogateGeneration { .. } => "SurrogateGeneration",
            SurrogateSignificance { .. } => "SurrogateSignificance",
            SurrogateContagionSignificance { .. } => "SurrogateContagionSignificance",
            SurrogateHybridGeneration { .. } => "SurrogateHybridGeneration",
            SurrogateDualSignificance { .. } => "SurrogateDualSignificance",
            SurrogateBistabilitySignificance { .. } => "SurrogateBistabilitySignificance",
            SurrogateOpinionSignificance { .. } => "SurrogateOpinionSignificance",
            HypergraphReconstruction { .. } => "HypergraphReconstruction",
            FuzzyAggregate { .. } => "FuzzyAggregate",
            FuzzyAllenGradation { .. } => "FuzzyAllenGradation",
            FuzzyQuantifierEvaluate { .. } => "FuzzyQuantifierEvaluate",
            FuzzySyllogismVerify { .. } => "FuzzySyllogismVerify",
            FuzzyFcaLattice { .. } => "FuzzyFcaLattice",
            FuzzyRuleEvaluate { .. } => "FuzzyRuleEvaluate",
            FuzzyHybridInference { .. } => "FuzzyHybridInference",
        }
    }
}

impl std::fmt::Display for InferenceJobType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.variant_name())
    }
}

impl std::str::FromStr for InferenceJobType {
    type Err = crate::error::TensaError;

    /// Parse a path-segment / variant name back into an `InferenceJobType`.
    /// Only unit variants are constructible this way — payload-carrying
    /// variants need their data and must be constructed directly.
    fn from_str(s: &str) -> crate::error::Result<Self> {
        serde_json::from_value(serde_json::Value::String(s.to_string())).map_err(|e| {
            crate::error::TensaError::InvalidInput(format!(
                "unknown InferenceJobType '{}': {} (payload variants cannot be parsed from a name alone)",
                s, e
            ))
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Priority level for inference jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobPriority {
    High = 0,
    Normal = 1,
    Low = 2,
}

// ─── Validation ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationLogEntry {
    pub target_id: Uuid,
    pub reviewer: String,
    pub old_maturity: MaturityLevel,
    pub new_maturity: MaturityLevel,
    pub notes: Option<String>,
    pub timestamp: DateTime<Utc>,
}

// ─── Platform & Engagement (Disinfo Sprint D1) ───────────────

/// Social-media / web platform that an entity (account) or situation (post)
/// originated on. Used by the disinfo extension for platform-aware spread
/// dynamics, behavioral fingerprinting, and CIB cross-platform detection.
///
/// Stored as serde-tagged string in entity/situation properties or as
/// part of `EngagementMetrics`. The `Other` variant carries an arbitrary
/// platform name for fediverse instances or new platforms not yet enumerated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Twitter,
    Telegram,
    Bluesky,
    Reddit,
    Facebook,
    Instagram,
    TikTok,
    YouTube,
    Mastodon,
    VKontakte,
    Rss,
    Web,
    Other(String),
}

impl Platform {
    /// Stable string tag — used as KV index segment and in JSON serialization.
    pub fn as_index_str(&self) -> &str {
        match self {
            Self::Twitter => "twitter",
            Self::Telegram => "telegram",
            Self::Bluesky => "bluesky",
            Self::Reddit => "reddit",
            Self::Facebook => "facebook",
            Self::Instagram => "instagram",
            Self::TikTok => "tiktok",
            Self::YouTube => "youtube",
            Self::Mastodon => "mastodon",
            Self::VKontakte => "vkontakte",
            Self::Rss => "rss",
            Self::Web => "web",
            Self::Other(name) => name.as_str(),
        }
    }
}

impl std::str::FromStr for Platform {
    type Err = crate::error::TensaError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s.to_lowercase().as_str() {
            "twitter" | "x" => Ok(Self::Twitter),
            "telegram" => Ok(Self::Telegram),
            "bluesky" | "bsky" => Ok(Self::Bluesky),
            "reddit" => Ok(Self::Reddit),
            "facebook" | "fb" | "meta" => Ok(Self::Facebook),
            "instagram" | "ig" => Ok(Self::Instagram),
            "tiktok" => Ok(Self::TikTok),
            "youtube" | "yt" => Ok(Self::YouTube),
            "mastodon" => Ok(Self::Mastodon),
            "vkontakte" | "vk" => Ok(Self::VKontakte),
            "rss" | "atom" => Ok(Self::Rss),
            "web" | "html" => Ok(Self::Web),
            other => Ok(Self::Other(other.to_string())),
        }
    }
}

/// Per-post engagement metrics captured at ingestion time.
///
/// Used by the disinfo extension for behavioral fingerprinting (engagement
/// ratio, response latency) and spread dynamics (R₀ amplification by
/// engagement-weighted reach). Optional fields default to `None` when the
/// platform does not surface that signal or it has not been fetched.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngagementMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub likes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shares: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replies: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub views: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saves: Option<u64>,
    /// Snapshot timestamp — when these metrics were observed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
}

impl EngagementMetrics {
    /// Total interactions (likes + shares + replies). Treats absent fields as 0.
    pub fn total_interactions(&self) -> u64 {
        self.likes.unwrap_or(0) + self.shares.unwrap_or(0) + self.replies.unwrap_or(0)
    }

    /// Share-to-like ratio in `[0, 1+)`. Returns `None` when there are no likes.
    pub fn share_ratio(&self) -> Option<f64> {
        let likes = self.likes.unwrap_or(0);
        if likes == 0 {
            None
        } else {
            Some(self.shares.unwrap_or(0) as f64 / likes as f64)
        }
    }
}

// ─── Chunk Storage ───────────────────────────────────────────

/// A persisted text chunk from the ingestion pipeline.
///
/// Chunks are the atomic units of text that the LLM processes during ingestion.
/// Storing them enables re-analysis, provenance tracing, and honest stylometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: Uuid,
    /// Ingestion job that created this chunk (matches `IngestionJob.id`).
    pub job_id: String,
    pub narrative_id: Option<String>,
    pub chunk_index: u32,
    pub text: String,
    /// Byte range `(start, end)` in the original document.
    pub byte_range: (usize, usize),
    /// Bytes of overlap with the previous chunk.
    pub overlap_bytes: usize,
    pub chapter: Option<String>,
    /// SHA-256 content hash for dedup detection.
    pub content_hash: String,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
}

impl ChunkRecord {
    /// Return the chunk text with overlap stripped for chunks after the first.
    /// `index` is the chunk's position in the sequence (0-based).
    pub fn text_without_overlap(&self, index: usize) -> &str {
        if index == 0 {
            &self.text
        } else {
            let mut skip = self.overlap_bytes.min(self.text.len());
            // Ensure we don't slice in the middle of a multi-byte UTF-8 character
            while skip < self.text.len() && !self.text.is_char_boundary(skip) {
                skip += 1;
            }
            &self.text[skip..]
        }
    }
}

// ─── Storywriting: User-defined narrative arcs ────────────────────────

/// A user-defined narrative arc scaffold: a named emotional/structural shape
/// (rising, falling, tragedy, rags-to-riches, custom, ...) spanning an ordered
/// subset of a narrative's situations. Distinct from `narrative::arc::NarrativeArc`,
/// which is a *classification result* computed from situation payoffs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserArc {
    pub id: Uuid,
    pub narrative_id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Arc shape tag — "rising", "falling", "tragedy", "rags-to-riches", "custom", etc.
    pub arc_type: String,
    /// Situations belonging to this arc, in writer-intended order.
    pub situation_ids: Vec<Uuid>,
    /// Display order among arcs in the narrative.
    pub order: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ─── Narrative Plan ───────────────────────────────────────────────────
//
// The writer's living document for a narrative: plot scaffolding, style
// targets, length targets, setting notes, audience. Separate from the
// bibliographic `Narrative` record so it can be large (synopsis + notes)
// without bloating every narrative list query. Consumed by W1 generation,
// W2 edit, and W3 workshop engines as canonical writer intent.

/// Stylistic targets for generation / edit operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StyleTargets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pov: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tense: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tone: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reading_level: Option<String>,
    /// Style influences / comps — "like McCarthy's Blood Meridian".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub influences: Vec<String>,
    /// Words or tropes the writer wants avoided.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub avoid: Vec<String>,
}

/// Length / structural targets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LengthTargets {
    /// "novel" | "novella" | "short-story" | "screenplay" | "serialized" | custom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_words: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_words: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_words: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_chapters: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_scenes_per_chapter: Option<u32>,
}

/// Setting / world notes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SettingNotes {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_period: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<String>,
    /// Freeform markdown — world-building rules, magic systems, tech levels, etc.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub world_notes: String,
    /// Freeform markdown — research links, historical references, domain notes.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub research_notes: String,
}

/// One story-structure beat (Inciting Incident, Midpoint, Dark Night, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotBeat {
    pub label: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_chapter: Option<u32>,
}

/// The writer's canonical plan for a narrative. One per narrative id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativePlan {
    pub narrative_id: String,

    // Plot & premise
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synopsis: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub premise: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub themes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub central_conflict: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plot_beats: Vec<PlotBeat>,

    // Style & structural targets
    #[serde(default)]
    pub style: StyleTargets,
    #[serde(default)]
    pub length: LengthTargets,
    #[serde(default)]
    pub setting: SettingNotes,

    // Freeform markdown notes (TODOs, open questions, inspiration).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,

    // Audience & positioning
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_audience: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comp_titles: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_warnings: Vec<String>,

    /// Escape hatch for fields outside the structured schema. Values may be
    /// any JSON — strings, numbers, booleans, arrays, or nested objects.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub custom: std::collections::HashMap<String, serde_json::Value>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ─── Cost Ledger (v0.49.4) ────────────────────────────────────────────
//
// Append-only table of token costs per AI operation. Every generation/edit/
// workshop LLM call writes one entry so writers can see cumulative spend.
// Token counts are the char/4 estimate used across generation / editing /
// workshop — real counts live on the provider side. Stored at
// `cl/{narrative_id}/{entry_uuid_bytes}`.

/// Which end-user operation produced the cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostOperation {
    Generation,
    Edit,
    Workshop,
}

/// One LLM call's cost record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostLedgerEntry {
    pub id: Uuid,
    pub narrative_id: String,
    pub operation: CostOperation,
    /// Operation-specific sub-kind (`"outline"`, `"character"`, `"scenes"` for
    /// generation; `"rewrite"`, `"tighten"`, ... for edit; `"standard"` for
    /// workshop Standard-tier enrichment).
    pub kind: String,
    pub prompt_tokens: u32,
    pub response_tokens: u32,
    /// Active model string at the time of the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Whether the response came from the LlmCache (free re-run).
    #[serde(default)]
    pub cache_hit: bool,
    /// Whether the underlying LLM call succeeded. Failures still record so
    /// the writer knows about wasted prompt tokens.
    #[serde(default = "default_true")]
    pub success: bool,
    /// Wall-clock duration, milliseconds.
    pub duration_ms: u64,
    pub created_at: DateTime<Utc>,
    /// Optional structured metadata. Used by multi-step operations (e.g. the
    /// fitness loop) to record per-iteration data without inflating `kind`
    /// cardinality so aggregation queries stay clean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

/// Aggregate over a time window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostSummary {
    pub narrative_id: String,
    /// Window label used ("7d", "30d", "all").
    pub window: String,
    pub total_calls: u32,
    pub cache_hits: u32,
    pub total_prompt_tokens: u64,
    pub total_response_tokens: u64,
    pub total_duration_ms: u64,
    /// Breakdown by operation.
    pub by_operation: Vec<CostOperationSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostOperationSummary {
    pub operation: CostOperation,
    pub calls: u32,
    pub cache_hits: u32,
    pub prompt_tokens: u64,
    pub response_tokens: u64,
}

// ─── Pinned Facts (v0.49.3) ───────────────────────────────────────────
//
// Writer's canonical facts that generation and edit operations must respect.
// Typical uses: "Alice's age is 23 at story start", "Elder Wand cannot be
// won in a duel where the holder is disarmed by the wielder", etc.
// Stored at `pf/{narrative_id}/{fact_id}` and consumed by W4 continuity checks.

/// A single canonical fact. Optionally scoped to an entity; otherwise it's
/// a narrative-wide fact (e.g. "All magic requires belief in it").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnedFact {
    pub id: Uuid,
    pub narrative_id: String,
    /// Entity this fact attaches to, if any. `None` = narrative-wide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<Uuid>,
    /// Key within the entity's properties, or a freeform label for
    /// narrative-wide facts. E.g. `"age"`, `"hair_color"`, `"magic_rule"`.
    pub key: String,
    /// Canonical value. Stored as string for simplicity; LLM prompts
    /// include it verbatim.
    pub value: String,
    /// Optional writer-supplied note (e.g. rationale, exception).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Continuity warning surfaced before applying a proposal. Non-blocking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityWarning {
    pub severity: ContinuityWarningSeverity,
    /// Short human-readable summary ("Alice age differs from pinned fact").
    pub headline: String,
    /// Detailed explanation.
    pub detail: String,
    /// The pinned fact id this warning references, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_fact_id: Option<Uuid>,
    /// The entity id this warning is about, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<Uuid>,
    /// What's proposed that triggered the warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposed_value: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityWarningSeverity {
    /// Direct conflict with a pinned fact.
    Conflict,
    /// Potential issue; informational.
    Advisory,
}

// ─── Narrative Revisions ──────────────────────────────────────────────
//
// Linear version-control over a narrative's authored state. A revision is an
// immutable snapshot of everything a writer cares about — situations, entities,
// participations, causal links, and user arcs — tagged with an author + message
// like a git commit. See `src/narrative/revision.rs` for the commit/restore/diff
// operations.

/// Full authored state of a single narrative at one point in time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativeSnapshot {
    pub narrative_metadata: Option<serde_json::Value>,
    pub situations: Vec<Situation>,
    pub entities: Vec<Entity>,
    pub participations: Vec<Participation>,
    pub causal_links: Vec<CausalLink>,
    pub user_arcs: Vec<UserArc>,
    /// Writer's plan (v0.48.2). `None` if the narrative has no plan set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<NarrativePlan>,
}

/// A named, immutable snapshot of a narrative, akin to a git commit.
/// Revisions form a linear chain via `parent_id`; v1 has no branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeRevision {
    pub id: Uuid,
    pub narrative_id: String,
    /// Parent revision id; `None` for the first commit on a narrative.
    pub parent_id: Option<Uuid>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    pub created_at: DateTime<Utc>,
    /// SHA-256 of the canonical-JSON-serialized `snapshot`. Used for dedup
    /// (identical state → same hash → skip the commit).
    pub content_hash: String,
    pub snapshot: NarrativeSnapshot,
}

/// Lightweight revision descriptor — everything except the full snapshot,
/// suitable for listing history without shipping megabytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevisionSummary {
    pub id: Uuid,
    pub narrative_id: String,
    pub parent_id: Option<Uuid>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    pub created_at: DateTime<Utc>,
    pub content_hash: String,
    /// Quick stats for the history UI.
    pub counts: RevisionCounts,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RevisionCounts {
    pub situations: usize,
    pub entities: usize,
    pub participations: usize,
    pub causal_links: usize,
    pub user_arcs: usize,
    pub total_words: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin index key strings — changing these would silently orphan on-disk index entries.
    #[test]
    fn test_entity_type_index_str_stability() {
        assert_eq!(EntityType::Actor.as_index_str(), "Actor");
        assert_eq!(EntityType::Location.as_index_str(), "Location");
        assert_eq!(EntityType::Artifact.as_index_str(), "Artifact");
        assert_eq!(EntityType::Concept.as_index_str(), "Concept");
        assert_eq!(EntityType::Organization.as_index_str(), "Organization");
    }

    #[test]
    fn test_narrative_level_index_str_stability() {
        assert_eq!(NarrativeLevel::Story.as_index_str(), "Story");
        assert_eq!(NarrativeLevel::Arc.as_index_str(), "Arc");
        assert_eq!(NarrativeLevel::Sequence.as_index_str(), "Sequence");
        assert_eq!(NarrativeLevel::Scene.as_index_str(), "Scene");
        assert_eq!(NarrativeLevel::Beat.as_index_str(), "Beat");
        assert_eq!(NarrativeLevel::Event.as_index_str(), "Event");
    }

    #[test]
    fn test_platform_index_str_stability() {
        // Pin platform tags — these become KV index segments.
        assert_eq!(Platform::Twitter.as_index_str(), "twitter");
        assert_eq!(Platform::Telegram.as_index_str(), "telegram");
        assert_eq!(Platform::Bluesky.as_index_str(), "bluesky");
        assert_eq!(Platform::Reddit.as_index_str(), "reddit");
        assert_eq!(Platform::Facebook.as_index_str(), "facebook");
        assert_eq!(Platform::Instagram.as_index_str(), "instagram");
        assert_eq!(Platform::TikTok.as_index_str(), "tiktok");
        assert_eq!(Platform::YouTube.as_index_str(), "youtube");
        assert_eq!(Platform::Mastodon.as_index_str(), "mastodon");
        assert_eq!(Platform::VKontakte.as_index_str(), "vkontakte");
        assert_eq!(Platform::Rss.as_index_str(), "rss");
        assert_eq!(Platform::Web.as_index_str(), "web");
        assert_eq!(Platform::Other("threads".into()).as_index_str(), "threads");
    }

    #[test]
    fn test_platform_from_str_aliases() {
        use std::str::FromStr;
        assert_eq!(Platform::from_str("X").unwrap(), Platform::Twitter);
        assert_eq!(Platform::from_str("twitter").unwrap(), Platform::Twitter);
        assert_eq!(Platform::from_str("BSKY").unwrap(), Platform::Bluesky);
        assert_eq!(Platform::from_str("Meta").unwrap(), Platform::Facebook);
        assert_eq!(Platform::from_str("Atom").unwrap(), Platform::Rss);
        // Unknown → Other(name)
        match Platform::from_str("threads").unwrap() {
            Platform::Other(name) => assert_eq!(name, "threads"),
            _ => panic!("expected Other"),
        }
    }

    #[test]
    fn test_engagement_metrics_helpers() {
        let m = EngagementMetrics {
            likes: Some(100),
            shares: Some(40),
            replies: Some(10),
            views: Some(5000),
            saves: None,
            timestamp: None,
        };
        assert_eq!(m.total_interactions(), 150);
        assert!((m.share_ratio().unwrap() - 0.4).abs() < 1e-9);

        let zero = EngagementMetrics::default();
        assert_eq!(zero.total_interactions(), 0);
        assert!(zero.share_ratio().is_none());
    }
}
