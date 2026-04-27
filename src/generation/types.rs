//! Shared types for the narrative generation pipeline (Sprint D9.6–D9.7).

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::style_profile::NarrativeFingerprint;
use crate::error::TensaError;
use crate::narrative::character_arcs::ArcType;
use crate::narrative::commitments::CommitmentType;
use crate::narrative::fabula_sjuzet::NarrationMode;
use crate::narrative::scene_sequel::SceneType;
use crate::narrative::subplots::SubplotRelation;

// ─── Fact System ────────────────────────────────────────────

/// A narrative fact ID (string key for knowledge tracking).
pub type FactId = String;

/// A planned fact in the narrative universe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedFact {
    pub id: FactId,
    pub description: String,
    /// Entities who know this fact at narrative start.
    pub known_by: Vec<Uuid>,
    /// Situation where this fact is first revealed to the reader.
    pub revealed_in: Option<Uuid>,
    /// Whether this fact is actually true. False = planned deception.
    pub is_true: bool,
}

/// A knowledge transition within a single situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeTransition {
    pub entity_id: Uuid,
    /// Facts this entity learns in this situation.
    pub learns: Vec<FactId>,
    /// Facts this entity loses/forgets/is deceived about.
    pub loses: Vec<FactId>,
    /// Who told them (if applicable).
    pub source: Option<Uuid>,
}

// ─── Plan Types ─────────────────────────────────────────────

/// A planned entity in the narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedEntity {
    pub id: Uuid,
    pub name: String,
    pub entity_type: String,
    pub role: crate::types::Role,
    pub arc_type: ArcType,
    /// Conscious desire.
    pub want: String,
    /// Unconscious necessity.
    pub need: String,
    /// False belief.
    pub lie: Option<String>,
    /// Reality to be discovered.
    pub truth: Option<String>,
    /// Facts known at narrative start.
    pub initial_knowledge: HashSet<FactId>,
    /// Initial MaxEnt IRL motivation prior.
    pub initial_motivation: Vec<f64>,
    /// Relationships to other entities: (entity_id, relationship_type).
    pub relationships: Vec<(Uuid, String)>,
}

/// Game type for a situation (cooperative, adversarial, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameType {
    Cooperative,
    Adversarial,
    InfoAsymmetry,
    Mixed,
}

/// A planned situation in the fabula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSituation {
    pub id: Uuid,
    pub chapter: usize,
    pub summary: String,
    pub participants: Vec<Uuid>,
    pub scene_type: SceneType,
    pub causal_predecessors: Vec<Uuid>,
    pub commitments_planted: Vec<String>,
    pub commitments_fulfilled: Vec<String>,
    pub narration_mode: NarrationMode,
    pub emotional_valence: f64,
    pub game_type: Option<GameType>,
    pub knowledge_transitions: Vec<KnowledgeTransition>,
}

/// A planned commitment (setup-payoff pair).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedCommitment {
    pub id: String,
    pub commitment_type: CommitmentType,
    pub element: String,
    pub setup_chapter: usize,
    pub payoff_chapter: usize,
    pub intermediate_progress_chapters: Vec<usize>,
}

/// A planned subplot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSubplot {
    pub label: String,
    pub chapters_active: Vec<usize>,
    pub relation_to_main: SubplotRelation,
    pub convergence_chapter: Option<usize>,
    pub characters: Vec<Uuid>,
}

/// A planned character arc with waypoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedCharacterArc {
    pub character_id: Uuid,
    pub arc_type: ArcType,
    pub midpoint_chapter: usize,
    pub dark_night_chapter: usize,
    pub resolution_chapter: usize,
    /// (chapter, target motivation vector) waypoints.
    pub motivation_waypoints: Vec<(usize, Vec<f64>)>,
}

/// User-specified plan constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanConstraint {
    pub description: String,
    pub constraint_type: ConstraintType,
}

/// Types of plan constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// "Character X must survive"
    CharacterSurvival { character: String },
    /// "Set in 1920s Prague"
    Setting { description: String },
    /// "Must include a twist in act 3"
    PlotPoint { description: String },
    /// "Avoid graphic violence"
    ContentRestriction { description: String },
    /// Freeform
    Custom { description: String },
}

/// Configuration for plan generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanConfig {
    pub genre: String,
    pub chapter_count: usize,
    pub protagonist_count: usize,
    pub subplot_count: usize,
    pub commitment_density: f64,
    /// One-sentence story premise from user.
    pub premise: String,
    /// User-specified constraints.
    pub constraints: Vec<PlanConstraint>,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            genre: "literary fiction".into(),
            chapter_count: 12,
            protagonist_count: 1,
            subplot_count: 2,
            commitment_density: 0.5,
            premise: String::new(),
            constraints: Vec::new(),
        }
    }
}

/// A complete narrative plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativePlan {
    pub id: Uuid,
    /// The narrative_id that will be used when materialized.
    pub narrative_id: String,
    pub config: PlanConfig,
    pub entities: Vec<PlannedEntity>,
    pub facts: Vec<PlannedFact>,
    pub fabula: Vec<PlannedSituation>,
    pub commitments: Vec<PlannedCommitment>,
    pub subplots: Vec<PlannedSubplot>,
    pub character_arcs: Vec<PlannedCharacterArc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ─── Validation Types ───────────────────────────────────────

/// A consistency issue found during plan/materialized narrative validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub description: String,
    /// The entity or situation involved.
    pub target_id: Option<Uuid>,
    pub chapter: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    TemporalInconsistency,
    CausalInconsistency,
    KnowledgeViolation,
    CommitmentOrphaned,
    ArcIncomplete,
    FactContradiction,
    RationalityViolation,
}

// ─── Generation Types ───────────────────────────────────────

/// Validated similarity threshold in 0.0..=1.0.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct Threshold(f64);

impl Threshold {
    /// Construct, returning `InvalidInput` on out-of-range.
    pub fn new(v: f64) -> Result<Self, TensaError> {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(TensaError::InvalidInput(format!(
                "threshold must be in 0.0..=1.0, got {v}"
            )));
        }
        Ok(Self(v))
    }

    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Threshold {
    type Error = TensaError;
    fn try_from(v: f64) -> Result<Self, Self::Error> {
        Self::new(v)
    }
}

impl Default for Threshold {
    // TODO(calibration): tune via held-out same-author baseline; see
    // TENSA_REFERENCE generation/fitness section.
    fn default() -> Self {
        Self(0.80)
    }
}

/// Output from a `ChapterGenerator`, with token counts for cost-ledger accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedText {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Style target for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTarget {
    /// Natural language voice description as fallback.
    pub voice_description: Option<String>,
    /// Style embedding ID (if available).
    pub style_embedding_id: Option<Uuid>,
    /// LoRA adapter ID (if available).
    pub lora_id: Option<Uuid>,
    /// Generation temperature.
    pub temperature: f64,
    /// Maximum retries per chapter.
    pub max_retries_per_chapter: usize,
    /// Target fingerprint that drives the fitness loop's accept/revise decision.
    /// When `None`, the loop is a single-shot generation.
    #[serde(default)]
    pub target_fingerprint: Option<NarrativeFingerprint>,
    /// Minimum prose-similarity score to accept an attempt.
    #[serde(default)]
    pub fitness_threshold: Threshold,
}

impl Default for StyleTarget {
    fn default() -> Self {
        Self {
            voice_description: None,
            style_embedding_id: None,
            lora_id: None,
            temperature: 0.7,
            max_retries_per_chapter: 3,
            target_fingerprint: None,
            fitness_threshold: Threshold::default(),
        }
    }
}

/// A generated chapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedChapter {
    pub chapter_number: usize,
    pub text: String,
    pub attempts: usize,
    pub style_adherence: f64,
    /// Commitments fulfilled in this chapter.
    pub commitment_fulfillments: Vec<String>,
    /// New entities the LLM improvised (not in the plan).
    pub entities_improvised: Vec<Uuid>,
}

/// Full generation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub narrative_id: String,
    pub chapters: Vec<GeneratedChapter>,
    pub generation_log: Vec<GenerationLogEntry>,
    pub total_attempts: usize,
    pub unfired_commitments: Vec<String>,
    pub knowledge_violations: Vec<String>,
}

/// Log entry for a generation attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationLogEntry {
    pub chapter: usize,
    pub attempt: usize,
    pub accepted: bool,
    pub deviations: Vec<String>,
    pub corrective_constraints: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Prompt for a single situation within a chapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationPrompt {
    pub situation_id: Uuid,
    pub summary: String,
    /// Per-participant knowledge and motivation context.
    pub character_contexts: Vec<CharacterContext>,
    pub scene_type_instruction: String,
    pub narration_mode_instruction: String,
    pub commitment_instructions: Vec<String>,
    pub dramatic_irony_instructions: Vec<String>,
    pub causal_context: String,
}

/// Character context extracted from the hypergraph for prompt construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterContext {
    pub entity_id: Uuid,
    pub name: String,
    /// What they know at this point.
    pub knows: Vec<String>,
    /// What they believe but is false.
    pub false_beliefs: Vec<String>,
    /// Current motivation description.
    pub motivation: String,
    /// Relationships to other participants.
    pub relationships: Vec<String>,
    /// Arc phase: pre-midpoint, dark-night, resolution.
    pub arc_phase: String,
}

/// Full generation prompt for a chapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPrompt {
    pub system_prompt: String,
    pub chapter_context: String,
    pub situation_specs: Vec<SituationPrompt>,
    /// Corrective constraints from previous failed attempts.
    pub constraints: Vec<String>,
}

// ─── KV ─────────────────────────────────────────────────────

/// Plan KV prefix.
pub fn plan_key(plan_id: &Uuid) -> Vec<u8> {
    format!("gp/{}", plan_id).into_bytes()
}

/// Fact KV prefix.
pub fn fact_key(narrative_id: &str, fact_id: &str) -> Vec<u8> {
    format!("fact/{}/{}", narrative_id, fact_id).into_bytes()
}

/// Generated chapter KV prefix.
pub fn chapter_text_key(narrative_id: &str, chapter: usize) -> Vec<u8> {
    format!("text/{}/chapter_{:04}", narrative_id, chapter).into_bytes()
}
