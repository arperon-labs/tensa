//! Request types for MCP tool parameters.
//!
//! All structs derive `Deserialize` + `JsonSchema` for automatic
//! MCP tool parameter schema generation via rmcp macros.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Parameters for the `query` tool — execute a TensaQL MATCH query.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct QueryRequest {
    /// TensaQL query string, e.g. `MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e LIMIT 10`
    pub tensaql: String,
}

/// Parameters for the `infer` tool — submit a TensaQL INFER or DISCOVER query.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct InferRequest {
    /// TensaQL INFER or DISCOVER query string, e.g.
    /// `INFER CAUSES FOR s:Situation RETURN s` or
    /// `DISCOVER PATTERNS ACROSS NARRATIVES RETURN *`
    pub tensaql: String,
}

/// Parameters for the `ingest_text` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestTextRequest {
    /// Raw narrative text to extract entities and situations from.
    pub text: String,
    /// Narrative ID to associate extracted data with (e.g. "crime-and-punishment").
    pub narrative_id: String,
    /// Source name for provenance tracking.
    #[serde(default = "default_source_name")]
    pub source_name: String,
    /// Confidence threshold for auto-commit (default 0.8).
    pub auto_commit_threshold: Option<f32>,
    /// Confidence threshold for review queue (default 0.3).
    pub review_threshold: Option<f32>,
    /// Extraction mode preset: novel, news, intelligence, research, temporal_events, legal, financial, medical.
    /// Selects a domain-specific extraction prompt. If not set, uses the global ingestion mode.
    pub ingestion_mode: Option<String>,
}

fn default_source_name() -> String {
    "mcp-upload".to_string()
}

/// Parameters for the `create_entity` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateEntityRequest {
    /// Entity type: Actor, Location, Artifact, Concept, or Organization.
    pub entity_type: String,
    /// JSON properties for the entity (e.g. `{"name": "Raskolnikov", "age": 23}`).
    pub properties: serde_json::Value,
    /// Optional narrative ID to associate this entity with.
    pub narrative_id: Option<String>,
    /// Confidence score (0.0 to 1.0, default 0.5).
    pub confidence: Option<f64>,
    /// Optional beliefs JSON — the entity's epistemic state.
    /// Shape depends on downstream inference engines (belief modeling,
    /// Dempster-Shafer evidence, recursive theory-of-mind). Pass
    /// through opaquely.
    #[serde(default)]
    pub beliefs: Option<serde_json::Value>,
}

/// Parameters for the `create_situation` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateSituationRequest {
    /// Short title for the scene/chapter/event (shows in the Studio outline
    /// and manuscript header). Skip at your peril: untitled scenes render as
    /// "(untitled)" everywhere.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional long-form description / synopsis. Distinct from `raw_content`
    /// which holds the prose itself.
    #[serde(default)]
    pub description: Option<String>,
    /// Raw text content describing the situation/event.
    pub raw_content: String,
    /// Optional ISO 8601 start time (e.g. "2024-01-15T10:00:00Z").
    pub start: Option<String>,
    /// Optional ISO 8601 end time.
    pub end: Option<String>,
    /// Narrative level: Story, Arc, Sequence, Scene, Beat, or Event (default: Scene).
    pub narrative_level: Option<String>,
    /// Optional narrative ID to associate this situation with.
    pub narrative_id: Option<String>,
    /// Confidence score (0.0 to 1.0, default 0.5).
    pub confidence: Option<f64>,
    /// Optional narratology metadata. Set POV at creation time —
    /// back-filling focalization scene-by-scene later is manual work.
    pub discourse: Option<DiscourseInput>,
    /// Optional SpatialData JSON (place name, lat/lng, polygon, etc.).
    /// Pass through opaquely; the server validates.
    #[serde(default)]
    pub spatial: Option<serde_json::Value>,
    /// Optional GameStructure JSON for scenes that model strategic
    /// interaction. Pass through opaquely.
    #[serde(default)]
    pub game_structure: Option<serde_json::Value>,
    /// Writer-curated manuscript order (binder position).
    /// Distinct from `temporal.start` so narrated order can diverge from
    /// chronology. Set on Scene-level situations for deterministic
    /// `get_scene_context` / manuscript export ordering.
    #[serde(default)]
    pub manuscript_order: Option<u32>,
    /// Parent scene UUID for binder hierarchy (Part → Chapter → Scene).
    /// Server rejects cycles and depth >MAX_PARENT_DEPTH.
    #[serde(default)]
    pub parent_situation_id: Option<String>,
}

/// Narratology fields for `create_situation`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiscourseInput {
    /// Temporal order of narration relative to fabula time:
    /// "analepsis" (flashback), "prolepsis" (flash-forward), "simultaneous".
    #[serde(default)]
    pub order: Option<String>,
    /// Duration of the telling: "scene", "summary", "ellipsis", "pause", "stretch".
    #[serde(default)]
    pub duration: Option<String>,
    /// UUID of the entity whose perspective focalizes this scene.
    #[serde(default)]
    pub focalization: Option<String>,
    /// Voice: "homodiegetic" (narrator is a character) or "heterodiegetic" (narrator is outside).
    #[serde(default)]
    pub voice: Option<String>,
}

/// Parameters for the `add_participant` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddParticipantRequest {
    /// UUID of the entity participating.
    pub entity_id: String,
    /// UUID of the situation.
    pub situation_id: String,
    /// Role. Either a built-in string (Protagonist, Antagonist, Witness, Target,
    /// Instrument, Confidant, Informant, Recipient, Bystander, SubjectOfDiscussion,
    /// Facilitator) or `{"Custom": "role-name"}` for arbitrary roles. Accepts
    /// arbitrary JSON so both payload shapes work.
    pub role: serde_json::Value,
    /// Optional action description (e.g. "confesses").
    #[serde(default)]
    pub action: Option<String>,
    /// Optional InfoSet: `{knows_before: [...], learns: [...], reveals: [...]}`
    /// — drives belief propagation and dramatic-irony analysis.
    #[serde(default)]
    pub info_set: Option<serde_json::Value>,
    /// Optional game-theoretic payoff. Accepts any JSON — scalar, object, or
    /// array — per the `Participation.payoff` schema.
    #[serde(default)]
    pub payoff: Option<serde_json::Value>,
}

/// Parameters for tools that take a single ID.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetByIdRequest {
    /// UUID of the entity or situation.
    pub id: String,
}

/// Parameters for the `create_narrative` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateNarrativeRequest {
    /// Human-readable slug ID (e.g. "crime-and-punishment").
    pub narrative_id: String,
    /// Display title.
    pub title: String,
    /// Optional genre tag (e.g. "novel", "investigation", "geopolitical").
    pub genre: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Optional tags for filtering.
    pub tags: Option<Vec<String>>,
    /// Author(s) of the source material.
    #[serde(default)]
    pub authors: Option<Vec<String>>,
    /// ISO 639-1 language code (e.g. "en", "ru", "de").
    #[serde(default)]
    pub language: Option<String>,
    /// Publication date of the source material (ISO 8601 RFC 3339).
    #[serde(default)]
    pub publication_date: Option<String>,
    /// URL to a cover image.
    #[serde(default)]
    pub cover_url: Option<String>,
    /// Optional parent project ID.
    #[serde(default)]
    pub project_id: Option<String>,
    /// Arbitrary user-defined key-value metadata. Values may be any JSON —
    /// strings, numbers, booleans, arrays, or nested objects.
    #[serde(default)]
    pub custom_properties: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Parameters for the `job_status` and `job_result` tools.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobIdRequest {
    /// The inference job ID returned by the `infer` tool.
    pub job_id: String,
}

/// Parameters for the `create_source` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateSourceRequest {
    /// Human-readable name for the source (e.g. "Reuters", "Twitter/OSINT").
    pub name: String,
    /// Source type: NewsOutlet, GovernmentAgency, AcademicInstitution,
    /// SocialMedia, Sensor, StructuredApi, HumanAnalyst, OsintTool, or custom string.
    pub source_type: String,
    /// URL of the source (e.g. "https://reuters.com").
    pub url: Option<String>,
    /// Description of the source.
    pub description: Option<String>,
    /// Initial trust score (0.0 to 1.0, default 0.5).
    pub trust_score: Option<f64>,
    /// Known biases as free-text labels (e.g. ["pro-government", "sensationalist"]).
    pub known_biases: Option<Vec<String>>,
    /// Optional tags for filtering.
    pub tags: Option<Vec<String>>,
}

/// Parameters for the `get_source` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSourceRequest {
    /// UUID of the source to retrieve.
    pub id: String,
}

/// Parameters for the `add_attribution` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddAttributionRequest {
    /// UUID of the registered source.
    pub source_id: String,
    /// UUID of the entity or situation being attributed.
    pub target_id: String,
    /// Whether the target is an "Entity" or "Situation".
    pub target_kind: String,
    /// URL of the specific article or data point.
    pub original_url: Option<String>,
    /// Relevant excerpt from the source material.
    pub excerpt: Option<String>,
    /// How well the extraction parsed this source (0.0–1.0, default 0.8).
    pub extraction_confidence: Option<f64>,
}

/// Parameters for the `list_contentions` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListContentionsRequest {
    /// UUID of a situation to find contentions for.
    pub situation_id: String,
}

/// Parameters for the `recompute_confidence` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecomputeConfidenceRequest {
    /// UUID of the entity or situation to recompute confidence for.
    pub id: String,
}

/// Parameters for the `review_queue` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReviewQueueRequest {
    /// Action to perform: "list", "get", "approve", "reject", or "edit".
    pub action: String,
    /// Item UUID (required for get/approve/reject/edit).
    pub item_id: Option<String>,
    /// Reviewer name (required for approve/reject/edit).
    pub reviewer: Option<String>,
    /// Optional rejection notes.
    pub notes: Option<String>,
    /// Edited data JSON (required for "edit" action).
    pub edited_data: Option<serde_json::Value>,
    /// Max items to return for "list" action (default 50).
    pub limit: Option<usize>,
}

/// Parameters for the `simulate_counterfactual` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimulateCounterfactualRequest {
    /// UUID of the situation to apply the counterfactual intervention to.
    pub situation_id: String,
    /// The field to intervene on (e.g. "action", "payoff", "outcome").
    pub intervention_target: String,
    /// The new value to assume for the intervention target.
    pub new_value: String,
}

/// Parameters for the `get_actor_profile` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetActorProfileRequest {
    /// UUID of the actor entity to retrieve a comprehensive profile for.
    pub actor_id: String,
}

/// Parameters for the `find_cross_narrative_patterns` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindCrossNarrativePatternsRequest {
    /// List of narrative IDs to compare for structural pattern discovery.
    pub narrative_ids: Vec<String>,
}

/// Parameters for the `delete_entity` and `delete_situation` tools.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteByIdRequest {
    /// UUID of the entity or situation to delete.
    pub id: String,
}

/// Parameters for the `update_entity` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateEntityRequest {
    /// UUID of the entity to update.
    pub id: String,
    /// JSON object with fields to update. Supports: `properties` (merged into existing),
    /// `confidence` (number), `narrative_id` (string or null).
    pub updates: serde_json::Value,
}

/// Parameters for the `update_situation` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateSituationRequest {
    /// UUID of the situation to update.
    pub id: String,
    /// JSON object with fields to update. Supports: `properties` (merged into existing),
    /// `name`, `description`, `confidence` (0..1), `narrative_id`, `synopsis`, `label`,
    /// `status`, `keywords` (array of strings), and Sprint P4.2 enrichment slots
    /// `game_structure`, `deterministic`, `probabilistic`, `temporal` (each accepts
    /// either an object to set or `null` to clear).
    pub updates: serde_json::Value,
}

/// Parameters for the `update_participation` tool (Sprint P4.2 retro-enrichment).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateParticipationRequest {
    /// UUID of the situation the participation belongs to.
    pub situation_id: String,
    /// UUID of the entity participating in the situation.
    pub entity_id: String,
    /// Sequence number for multi-role participations (default 0).
    #[serde(default)]
    pub seq: u16,
    /// JSON object with fields to update. Supports: `info_set` (object with
    /// `knows_before`, `learns`, `reveals`, `beliefs_about_others`), `payoff`
    /// (any JSON value), `action` (string or null), `role` (one of Protagonist
    /// / Antagonist / Witness / Target / Instrument / Confidant / Informant /
    /// Recipient / Bystander / SubjectOfDiscussion). Use `null` on info_set
    /// or payoff to clear.
    pub updates: serde_json::Value,
}

/// Parameters for the `list_entities` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListEntitiesRequest {
    /// Filter by entity type: Actor, Location, Artifact, Concept, or Organization.
    pub entity_type: Option<String>,
    /// Filter by narrative ID.
    pub narrative_id: Option<String>,
    /// Maximum number of entities to return (default: all).
    pub limit: Option<usize>,
}

/// Parameters for the `merge_entities` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MergeEntitiesRequest {
    /// UUID of the entity to keep (survives the merge).
    pub keep_id: String,
    /// UUID of the entity to absorb (deleted after merge).
    pub absorb_id: String,
}

/// Parameters for the `export_narrative` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportNarrativeRequest {
    /// Narrative ID to export.
    pub narrative_id: String,
    /// Export format: csv, graphml, json, manuscript, or report.
    pub format: String,
}

/// Parameters for the `get_narrative_stats` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetNarrativeStatsRequest {
    /// Narrative ID to compute statistics for.
    pub narrative_id: String,
    /// Optional fuzzy t-norm override (Phase 11). When set, threads
    /// through to the REST endpoint as `?tnorm=<kind>` and the response
    /// envelope carries a `fuzzy_config` echo. `None` preserves pre-
    /// sprint URL / body shape bit-identically.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional fuzzy aggregator override (Phase 11).
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// Parameters for the `search_entities` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchEntitiesRequest {
    /// Text query to search for across entity properties.
    pub query: String,
    /// Maximum number of results to return (default: 20).
    pub limit: Option<usize>,
    /// Optional fuzzy t-norm override (Phase 11). Threaded through as
    /// `?tnorm=<kind>` on HTTP; `None` preserves pre-sprint URL shape.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional fuzzy aggregator override (Phase 11).
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// Parameters for the `ingest_url` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestUrlRequest {
    /// URL to fetch and ingest.
    pub url: String,
    /// Narrative ID to associate extracted data with.
    pub narrative_id: String,
    /// Source name for provenance tracking.
    pub source_name: Option<String>,
}

/// Parameters for the `ingest_rss` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestRssRequest {
    /// RSS/Atom feed URL to fetch.
    pub feed_url: String,
    /// Narrative ID to associate extracted data with.
    pub narrative_id: String,
    /// Maximum number of feed items to ingest (default: 10).
    pub max_items: Option<usize>,
}

/// Parameters for the `split_entity` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SplitEntityRequest {
    /// UUID of the entity to split.
    pub entity_id: String,
    /// Situation UUIDs to move to the new (cloned) entity.
    pub situation_ids: Vec<String>,
}

/// Parameters for the `restore_entity` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RestoreEntityRequest {
    /// UUID of the soft-deleted entity to restore.
    pub id: String,
}

/// Parameters for the `restore_situation` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RestoreSituationRequest {
    /// UUID of the soft-deleted situation to restore.
    pub id: String,
}

/// Parameters for the `create_project` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateProjectRequest {
    /// Human-readable slug ID for the project (e.g. "geopolitics").
    pub name: String,
    /// Optional display title (defaults to name if not provided).
    pub title: Option<String>,
    /// Optional description of the project.
    pub description: Option<String>,
}

/// Parameters for the `get_project` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetProjectRequest {
    /// Project ID (slug) to retrieve.
    pub id: String,
}

/// Parameters for the `list_projects` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListProjectsRequest {
    /// Maximum number of projects to return.
    pub limit: Option<usize>,
}

/// Parameters for the `update_project` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateProjectRequest {
    /// Project ID (slug) to update.
    pub id: String,
    /// New title (if provided).
    pub title: Option<String>,
    /// New description (if provided, use null to clear).
    pub description: Option<String>,
}

/// Parameters for the `delete_project` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteProjectRequest {
    /// Project ID (slug) to delete.
    pub id: String,
    /// If true, also cascade-delete all narratives and their entities/situations.
    pub cascade: Option<bool>,
}

/// Parameters for `run_full_analysis` — headless equivalent of Studio's
/// "Run Full Analysis" button. Submits every algorithmic inference job for
/// a narrative, skipping rows whose analysis-status entry is locked unless
/// `force = true`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunFullAnalysisRequest {
    /// Narrative ID (slug or UUID).
    pub narrative_id: String,
    /// Optional tier subset for the HTTP backend (`foundational`, `structural`,
    /// `per_actor`, `temporal`, `advanced`). Ignored by the embedded backend,
    /// which always submits its curated flat list.
    #[serde(default)]
    pub tiers: Option<Vec<String>>,
    /// When `true`, locked Skill / Manual rows are re-run and overwritten.
    /// Defaults to `false`.
    #[serde(default)]
    pub force: bool,
}

/// Parameters for `backfill_embeddings` — generates embeddings for entities
/// and situations missing them. Requires an embedding provider configured
/// server-side.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BackfillEmbeddingsRequest {
    /// Optional narrative ID to scope the backfill. When omitted, sweeps all
    /// `Candidate`-maturity rows.
    #[serde(default)]
    pub narrative_id: Option<String>,
    /// When `true`, re-embeds rows that already have an embedding.
    #[serde(default)]
    pub force: bool,
}

/// Parameters for the `ask` tool — RAG question answering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AskRequest {
    /// The natural language question to answer.
    pub question: String,
    /// Optional narrative ID to scope the query.
    pub narrative_id: Option<String>,
    /// Retrieval mode: "local", "global", "hybrid", or "mix". Defaults to "hybrid".
    pub mode: Option<String>,
    /// Desired response format (e.g. "brief summary", "bullet points", "detailed report").
    pub response_type: Option<String>,
    /// Whether to generate follow-up question suggestions.
    #[serde(default)]
    pub suggest: bool,
    /// Optional fuzzy t-norm override (Phase 11). Threaded through as
    /// `?tnorm=<kind>` on HTTP. `None` preserves pre-sprint URL shape.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional fuzzy aggregator override (Phase 11).
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// Parameters for the `tune_prompts` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TunePromptsRequest {
    /// Narrative ID to generate domain-adapted extraction prompts for.
    pub narrative_id: String,
}

/// Parameters for the `community_hierarchy` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CommunityHierarchyRequest {
    /// Narrative ID to get community hierarchy for.
    pub narrative_id: String,
    /// Optional level filter (0 = leaf, higher = coarser). Omit for all levels.
    pub level: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportArchiveRequest {
    /// List of narrative IDs to export.
    pub narrative_ids: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportArchiveRequest {
    /// Base64-encoded .tensa archive data.
    pub data: String,
}

/// Parameters for the `verify_authorship` tool (PAN@CLEF authorship verification).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VerifyAuthorshipRequest {
    /// First text to compare.
    pub text_a: String,
    /// Second text to compare.
    pub text_b: String,
}

// ─── Narrative Debugger (Sprint D10) ─────────────────────

/// Parameters for the `diagnose_narrative` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiagnoseNarrativeRequest {
    pub narrative_id: String,
    /// Optional genre preset: thriller | literary_fiction | epic_fantasy | mystery | generic.
    pub genre: Option<String>,
}

/// Parameters for the `get_health_score` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetHealthScoreRequest {
    pub narrative_id: String,
}

/// Parameters for the `auto_repair_narrative` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AutoRepairRequest {
    pub narrative_id: String,
    /// Max severity to address: error | warning | info (default: warning).
    pub max_severity: Option<String>,
    /// Bound on the diagnose→fix loop (default 5, clamped to 1..=20).
    pub max_iterations: Option<usize>,
}

// ─── Narrative Adaptation (Sprint D11) ────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeEssentialityRequest {
    pub narrative_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompressNarrativeRequest {
    pub narrative_id: String,
    /// Preset: novella | short_story | screenplay_outline.
    pub preset: Option<String>,
    pub target_chapters: Option<usize>,
    /// Alternative to target_chapters: a fraction of original (0.4 = 40%).
    pub target_ratio: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExpandNarrativeRequest {
    pub narrative_id: String,
    pub target_chapters: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffNarrativesRequest {
    pub narrative_a: String,
    pub narrative_b: String,
}

// ─── Disinfo Fingerprint Tools (Sprint D1) ─────────────────

/// Parameters for the `get_behavioral_fingerprint` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetBehavioralFingerprintRequest {
    /// Actor (entity) UUID.
    pub actor_id: String,
    /// If true, recompute even when a cached fingerprint exists. Defaults to false.
    #[serde(default)]
    pub recompute: bool,
    /// Optional fuzzy t-norm override (Phase 11). Threaded through as
    /// `?tnorm=<kind>` on HTTP; `None` preserves pre-sprint URL shape.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional fuzzy aggregator override (Phase 11).
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// Parameters for the `get_disinfo_fingerprint` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetDisinfoFingerprintRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// If true, recompute even when a cached fingerprint exists. Defaults to false.
    #[serde(default)]
    pub recompute: bool,
    /// Optional fuzzy t-norm override (Phase 11). Threaded through as
    /// `?tnorm=<kind>` on HTTP; `None` preserves pre-sprint URL shape.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional fuzzy aggregator override (Phase 11).
    #[serde(default)]
    pub aggregator: Option<String>,
}

/// Parameters for the `compare_fingerprints` tool.
///
/// Returns per-axis distances, weighted composite, p-value, and 95% CI for
/// either two behavioral or two disinfo fingerprints.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareFingerprintsRequest {
    /// Comparison kind: `"behavioral"` or `"disinfo"`. Narrative content
    /// comparisons go via the existing `verify_authorship` tool.
    pub kind: String,
    /// Task weighting profile: `"literary"` (default), `"cib"`, or `"factory"`.
    #[serde(default)]
    pub task: Option<String>,
    /// Identifier of the first fingerprint:
    /// - For `behavioral`: actor UUID.
    /// - For `disinfo`: narrative ID.
    pub a_id: String,
    /// Identifier of the second fingerprint (same kind as `a_id`).
    pub b_id: String,
}

// ─── Spread Dynamics Tools (Sprint D2) ─────────────────────

/// Parameters for the `estimate_r0_by_platform` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EstimateR0Request {
    pub narrative_id: String,
    /// The KnowledgeFact text being modeled (e.g. "Election was rigged").
    pub fact: String,
    /// UUID of the entity the fact is about.
    pub about_entity: String,
    /// Bucket name for baseline lookup (default `"default"`).
    #[serde(default)]
    pub narrative_kind: Option<String>,
    /// Optional per-platform β overrides (`{ "twitter": 0.5 }`). Defaults
    /// come from `PlatformBeta::default_for`.
    #[serde(default)]
    pub beta_overrides: Option<std::collections::HashMap<String, f64>>,
}

/// Parameters for the `simulate_intervention` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimulateInterventionRequest {
    pub narrative_id: String,
    pub fact: String,
    pub about_entity: String,
    /// Intervention spec: either `{type: "RemoveTopAmplifiers", n: 3}` or
    /// `{type: "DebunkAt", at: "2026-04-16T12:00:00Z"}`.
    pub intervention: serde_json::Value,
    #[serde(default)]
    pub beta_overrides: Option<std::collections::HashMap<String, f64>>,
}

// ─── CIB Detection Tools (Sprint D3) ─────────────────────

/// Parameters for the `detect_cib_cluster` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectCibClusterRequest {
    pub narrative_id: String,
    /// If true, only flag clusters spanning ≥ 2 distinct platforms (factory-task
    /// detection). Defaults to false.
    #[serde(default)]
    pub cross_platform: bool,
    /// Minimum axis-composite similarity (0..1) to draw an edge. Default 0.7.
    #[serde(default)]
    pub similarity_threshold: Option<f64>,
    /// Right-tail p-value cutoff for flagging a cluster. Default 0.01.
    #[serde(default)]
    pub alpha: Option<f64>,
    /// Bootstrap iterations to calibrate the density null. Default 500.
    #[serde(default)]
    pub bootstrap_iter: Option<usize>,
    /// Minimum members required to consider a community. Default 3.
    #[serde(default)]
    pub min_cluster_size: Option<usize>,
    /// Seed for the deterministic bootstrap RNG.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Parameters for the `rank_superspreaders` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RankSuperspreadersRequest {
    pub narrative_id: String,
    /// Centrality method: `"pagerank"` (default), `"eigenvector"`, or
    /// `"harmonic"` (betweenness-like). Case-insensitive.
    #[serde(default)]
    pub method: Option<String>,
    /// Number of top actors to return. Default 10.
    #[serde(default)]
    pub top_n: Option<usize>,
}

// ─── Claims & Fact-Check Tools (Sprint D4) ──────────────────

/// Parameters for the `ingest_claim` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestClaimRequest {
    /// Raw text to detect verifiable claims in.
    pub text: String,
    /// Optional narrative ID to associate claims with.
    #[serde(default)]
    pub narrative_id: Option<String>,
}

/// Parameters for the `ingest_fact_check` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestFactCheckRequest {
    /// UUID of the claim being fact-checked.
    pub claim_id: String,
    /// Verdict: `"true"`, `"false"`, `"misleading"`, `"partially_true"`,
    /// `"unverifiable"`, `"satire"`, `"out_of_context"`.
    pub verdict: String,
    /// Fact-checking organization or source name.
    pub source: String,
    /// URL to the fact-check article.
    #[serde(default)]
    pub url: Option<String>,
    /// Language of the fact-check (ISO 639-1). Default `"en"`.
    #[serde(default = "default_en")]
    pub language: String,
}

#[cfg(feature = "disinfo")]
fn default_en() -> String {
    "en".to_string()
}

/// Parameters for the `fetch_fact_checks` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FetchFactChecksRequest {
    /// UUID of the claim to find matching fact-checks for.
    pub claim_id: String,
    /// Minimum similarity threshold. Default 0.5.
    #[serde(default = "default_min_sim")]
    pub min_similarity: f64,
}

#[cfg(feature = "disinfo")]
fn default_min_sim() -> f64 {
    0.5
}

/// Parameters for the `trace_claim_origin` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TraceClaimOriginRequest {
    /// UUID of the claim to trace.
    pub claim_id: String,
}

/// Parameters for the `classify_archetype` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClassifyArchetypeRequest {
    /// UUID of the actor entity to classify.
    pub actor_id: String,
    /// Force recompute even if a cached result exists.
    #[serde(default)]
    pub force: bool,
}

/// Parameters for the `assess_disinfo` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssessDisinfoRequest {
    /// ID of the target entity or narrative being assessed.
    pub target_id: String,
    /// Array of disinfo signal objects to fuse via Dempster-Shafer combination.
    /// Each signal has: source (string), mass_true, mass_false, mass_misleading, mass_uncertain (all f64).
    pub signals: serde_json::Value,
}

/// Parameters for the `ingest_post` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestPostRequest {
    /// Text content of the social media post.
    pub text: String,
    /// UUID of the actor who made the post.
    pub actor_id: String,
    /// Narrative ID to associate the post with.
    pub narrative_id: String,
    /// Platform name (e.g. "twitter", "telegram", "facebook").
    #[serde(default)]
    pub platform: Option<String>,
}

/// Parameters for the `ingest_actor` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestActorRequest {
    /// Display name of the actor.
    pub name: String,
    /// Narrative ID to associate the actor with.
    pub narrative_id: String,
    /// Platform the actor operates on (e.g. "twitter", "telegram").
    #[serde(default)]
    pub platform: Option<String>,
    /// Additional properties for the actor entity (JSON object).
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
}

/// Parameters for the `link_narrative` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LinkNarrativeRequest {
    /// UUID of the entity to link.
    pub entity_id: String,
    /// Narrative ID to set on the entity.
    pub narrative_id: String,
}

// ─── Multilingual & Export (Sprint D6) ──────────────────────

/// Parameters for the `detect_language` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectLanguageRequest {
    /// Text to detect the language of.
    pub text: String,
}

/// Parameters for the `export_misp_event` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportMispRequest {
    /// Narrative ID to export as a MISP event.
    pub narrative_id: String,
}

/// Parameters for the `export_maltego` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportMaltegoRequest {
    /// Narrative ID to export as Maltego transform results.
    pub narrative_id: String,
}

/// Parameters for the `generate_report` tool (disinfo narrative report).
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateReportRequest {
    /// Narrative ID to generate a disinfo analysis report for.
    pub narrative_id: String,
}

// ─── Sprint D8: Scheduler + Reports + Health ──────────────────

/// Parameters for the `create_scheduled_task` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateScheduledTaskRequest {
    /// Task type: cib_scan, source_discovery, fact_check_sync,
    /// report_generation, mcp_poll, fingerprint_refresh, velocity_baseline_update.
    pub task_type: String,
    /// Schedule interval (e.g. "30m", "6h", "1d", "3600s").
    pub schedule: String,
    /// Optional configuration JSON for the task.
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

/// Parameters for the `run_task_now` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunTaskNowRequest {
    /// UUID of the scheduled task to trigger immediately.
    pub task_id: String,
}

/// Parameters for the `generate_situation_report` tool.
#[cfg(feature = "disinfo")]
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateSituationReportRequest {
    /// Time window in hours (default 24). Report covers the last N hours.
    #[serde(default = "default_report_hours")]
    pub hours: u64,
}

#[cfg(feature = "disinfo")]
fn default_report_hours() -> u64 {
    24
}

// ─── Generation (Sprint D9) ─────────────────────────────────

/// Parameters for detecting narrative commitments (Chekhov's guns, foreshadowing).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectCommitmentsRequest {
    /// Narrative ID to analyze.
    pub narrative_id: String,
}

/// Parameters for getting commitment promise rhythm.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetCommitmentRhythmRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for extracting fabula (chronological order).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractFabulaRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for extracting sjužet (discourse order).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractSjuzetRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for suggesting sjužet reorderings.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SuggestReorderingRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for computing dramatic irony map.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeDramaticIronyRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for detecting focalization.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectFocalizationRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for detecting a character arc.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectCharacterArcRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Character entity ID (optional — if omitted, detects all character arcs).
    pub character_id: Option<String>,
}

/// Parameters for detecting subplots.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectSubplotsRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for classifying scene-sequel rhythm.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClassifySceneSequelRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for generating a narrative plan.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateNarrativePlanRequest {
    /// One-sentence story premise.
    pub premise: String,
    /// Genre (e.g., "mystery", "literary fiction", "thriller").
    #[serde(default = "default_genre")]
    pub genre: String,
    /// Number of chapters.
    #[serde(default = "default_chapters")]
    pub chapter_count: usize,
    /// Number of subplots.
    #[serde(default = "default_subplots")]
    pub subplot_count: usize,
}

fn default_genre() -> String {
    "literary fiction".into()
}
fn default_chapters() -> usize {
    12
}
fn default_subplots() -> usize {
    2
}

/// Parameters for materializing a plan into the hypergraph.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MaterializePlanRequest {
    /// Plan ID (UUID).
    pub plan_id: String,
}

/// Parameters for validating a materialized narrative.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ValidateMaterializedRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for generating a chapter.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateChapterRequest {
    /// Narrative ID (must be materialized).
    pub narrative_id: String,
    /// Chapter number (0-indexed).
    pub chapter: usize,
    /// Natural language voice/style description (optional).
    pub voice_description: Option<String>,
}

/// Parameters for generating a full narrative.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateNarrativeRequest {
    /// Narrative ID (must be materialized).
    pub narrative_id: String,
    /// Number of chapters to generate.
    pub chapter_count: usize,
    /// Natural language voice/style description (optional).
    pub voice_description: Option<String>,
}

/// Parameters for the `generate_chapter_with_fitness` tool — submits a
/// `ChapterGenerationFitness` inference job that runs the LLM-driven
/// SE→fitness retry loop.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateChapterWithFitnessRequest {
    /// Narrative ID (must be materialized).
    pub narrative_id: String,
    /// Chapter number (0-indexed).
    pub chapter: usize,
    /// Natural language voice/style description (optional).
    pub voice_description: Option<String>,
    /// Style embedding ID to steer generation (optional, UUID string).
    pub style_embedding_id: Option<String>,
    /// Narrative ID whose fingerprint becomes the fitness target (optional).
    /// When provided, the engine builds the target fingerprint via
    /// `analysis::style_profile::build_fingerprint` before submitting the job.
    pub target_fingerprint_source: Option<String>,
    /// Minimum prose-similarity score required to accept an attempt (0.0..=1.0).
    pub fitness_threshold: Option<f64>,
    /// Maximum retries per chapter (defaults to the StyleTarget default).
    pub max_retries: Option<usize>,
    /// Generation temperature (defaults to the StyleTarget default).
    pub temperature: Option<f64>,
}

// ─── Sprint D12: Adversarial Narrative Wargaming ─────────────

/// Parameters for generating an adversary policy.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateAdversaryPolicyRequest {
    /// Entity ID of the actor (UUID), or omit to use archetype defaults.
    pub actor_id: Option<String>,
    /// Narrative ID for context.
    pub narrative_id: String,
    /// Adversarial archetype (e.g. "state_actor", "troll_farm"). Optional.
    pub archetype: Option<String>,
    /// Rationality parameter lambda (default 1.0).
    pub lambda: Option<f64>,
    /// Maximum lambda cap (default 4.6).
    pub lambda_cap: Option<f64>,
    /// IRL reward weights (default: uniform). Length must match feature dimensions.
    pub reward_weights: Option<Vec<f64>>,
}

/// Parameters for configuring rationality parameters.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConfigureRationalityRequest {
    /// Rationality model: "qre", "suqr", or "cognitive_hierarchy".
    pub model: String,
    /// Lambda (rationality) parameter.
    pub lambda: Option<f64>,
    /// Lambda cap to prevent superhuman play.
    pub lambda_cap: Option<f64>,
    /// Tau parameter (for cognitive hierarchy, default 1.5).
    pub tau: Option<f64>,
    /// Feature weights for SUQR (optional).
    pub feature_weights: Option<Vec<f64>>,
}

/// Parameters for creating a wargame session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateWargameRequest {
    /// Narrative ID to fork the simulation from.
    pub narrative_id: String,
    /// Maximum turns (default 20).
    pub max_turns: Option<usize>,
    /// Time step per turn in minutes (default 60).
    pub time_step_minutes: Option<u64>,
    /// Auto-play red team (default true).
    pub auto_red: Option<bool>,
    /// Auto-play blue team (default false).
    pub auto_blue: Option<bool>,
}

/// Parameters for submitting wargame moves.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SubmitWargameMoveRequest {
    /// Session ID.
    pub session_id: String,
    /// JSON array of red team moves.
    pub red_moves: Option<serde_json::Value>,
    /// JSON array of blue team moves.
    pub blue_moves: Option<serde_json::Value>,
}

/// Parameters for getting wargame state.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetWargameStateRequest {
    /// Session ID.
    pub session_id: String,
}

/// Parameters for auto-playing a wargame.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AutoPlayWargameRequest {
    /// Session ID.
    pub session_id: String,
    /// Number of turns to auto-play (default 5).
    pub num_turns: Option<usize>,
}

// ─── Sprint W6: Writer Workflows ─────────────────────────────

/// Parameters for narrative-ID-only writer tools.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriterNarrativeRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

/// Parameters for upserting a narrative plan.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpsertPlanRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Partial plan patch. See `upsert_narrative_plan` tool description for the
    /// exact field shapes (plot_beats, style, length, setting, custom, etc.).
    pub patch: serde_json::Value,
}

/// Parameters for running a workshop pass.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunWorkshopRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Tier: "cheap" (no LLM), "standard" (LLM review), or "deep" (asynchronous).
    pub tier: String,
    /// Focus areas: pacing, continuity, characterization, prose, structure.
    /// Defaults to all five if omitted.
    pub focuses: Option<Vec<String>>,
    /// Cap on LLM-enriched findings per focus in Standard tier.
    pub max_llm_calls: Option<u32>,
}

/// Parameters for creating a pinned continuity fact.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreatePinnedFactRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Fact object. Required fields: `key` (string, e.g. "hair_color"),
    /// `value` (string, e.g. "blond"). Optional: `note` (string),
    /// `entity_id` (UUID string — scopes the fact to a cast member;
    /// omit for narrative-wide canon).
    pub fact: serde_json::Value,
}

/// Parameters for checking prose against pinned facts.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CheckContinuityRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Prose to scan.
    pub prose: String,
}

/// Parameters for listing narrative revisions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListRevisionsRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Max number of most-recent revisions to return.
    pub limit: Option<usize>,
}

/// Parameters for restoring a narrative revision.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RestoreRevisionRequest {
    /// Narrative ID. Validated against the revision — must match.
    pub narrative_id: String,
    /// Revision UUID to restore from.
    pub revision_id: String,
    /// Author attribution for the safety auto-commit created before restore.
    pub author: String,
}

/// Parameters for the writer cost ledger summary.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CostSummaryRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Time window label: "24h", "7d", "30d", or "all" (default "all").
    pub window: Option<String>,
}

/// Parameters for `set_situation_content` — the assistant-as-author prose
/// write-back tool used by Workflow 2 (co-writing loop).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetSituationContentRequest {
    /// Situation UUID to write prose into.
    pub situation_id: String,
    /// Either a plain string (wrapped as one Text content block) OR an
    /// array of content blocks `[{content_type: "Text"|"Dialogue"|
    /// "Observation"|"Document"|"MediaRef", content: string}]`. The new
    /// blocks replace the situation's existing `raw_content` wholesale —
    /// call `commit_narrative_revision` first if you want to preserve
    /// the prior prose in revision history.
    pub content: serde_json::Value,
    /// Optional workflow status string to stamp on the situation
    /// (e.g. "first-draft", "revised"). Leaves status unchanged if omitted.
    #[serde(default)]
    pub status: Option<String>,
}

/// Parameters for `get_scene_context` — packaged context bundle for
/// scene-scoped drafting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSceneContextRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Optional situation UUID being drafted. When provided, the response
    /// includes the situation's current state and the N scenes immediately
    /// preceding it in manuscript order (fallback: temporal order).
    #[serde(default)]
    pub situation_id: Option<String>,
    /// Optional POV entity UUID. When provided, the response includes the
    /// entity's properties, beliefs, pinned facts, and recent participations.
    #[serde(default)]
    pub pov_entity_id: Option<String>,
    /// How many preceding scenes to include (default 2, max 10).
    #[serde(default)]
    pub lookback_scenes: Option<usize>,
}

// ─── Sprint W15: Writer MCP bridge ────────────────────────

// --- Annotations ----------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateAnnotationRequest {
    /// UUID of the situation (scene) this annotation anchors to.
    pub situation_id: String,
    /// Kind: "Comment" | "Footnote" | "Citation".
    pub kind: String,
    /// Byte-span start index into the scene's concatenated prose (default 0).
    #[serde(default)]
    pub span_start: usize,
    /// Byte-span end index (exclusive). Must be >= span_start (default = span_start).
    #[serde(default)]
    pub span_end: usize,
    /// Annotation body text. Required for Comment/Footnote; may be empty for Citation.
    #[serde(default)]
    pub body: String,
    /// Optional source UUID (attaches a Citation to a registered source).
    #[serde(default)]
    pub source_id: Option<String>,
    /// Optional chunk UUID (citation-to-specific-chunk back-link).
    #[serde(default)]
    pub chunk_id: Option<String>,
    /// Optional author attribution for the annotation.
    #[serde(default)]
    pub author: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListAnnotationsRequest {
    /// Scope: provide exactly one of `situation_id` (scene-scoped) or
    /// `narrative_id` (flatten every annotation under the narrative).
    #[serde(default)]
    pub situation_id: Option<String>,
    #[serde(default)]
    pub narrative_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateAnnotationRequest {
    /// Annotation UUID to patch.
    pub annotation_id: String,
    /// Arbitrary patch object; recognised keys: body, span (\[start,end\] tuple),
    /// source_id, chunk_id, detached.
    pub patch: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteAnnotationRequest {
    /// Annotation UUID to delete.
    pub annotation_id: String,
}

// --- Collections ----------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateCollectionRequest {
    /// Narrative the collection belongs to.
    pub narrative_id: String,
    /// Human-friendly collection name (non-empty).
    pub name: String,
    /// Optional description shown in Studio's binder.
    #[serde(default)]
    pub description: Option<String>,
    /// Structured filter — labels, statuses, keywords, text substring, bounds.
    /// Accepts the same shape as `CollectionQuery` (see
    /// [src/writer/collection.rs]). Empty / missing = match all.
    #[serde(default)]
    pub query: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListCollectionsRequest {
    /// Narrative to list collections for.
    pub narrative_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetCollectionRequest {
    /// Collection UUID.
    pub collection_id: String,
    /// If true, include the resolved matching situation IDs in the response.
    #[serde(default)]
    pub resolve: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateCollectionRequest {
    /// Collection UUID to patch.
    pub collection_id: String,
    /// Patch object — recognised keys: name, description, query.
    pub patch: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteCollectionRequest {
    /// Collection UUID.
    pub collection_id: String,
}

// --- Research notes ------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateResearchNoteRequest {
    /// Narrative ID this note belongs to.
    pub narrative_id: String,
    /// Situation (scene) this note is pinned to.
    pub situation_id: String,
    /// Kind: "Quote" | "Clipping" | "Link" | "Note" (default "Note").
    #[serde(default = "default_research_kind")]
    pub kind: String,
    /// Body of the note (non-empty after trim).
    pub body: String,
    /// Optional registered source UUID.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Optional author attribution.
    #[serde(default)]
    pub author: Option<String>,
}

fn default_research_kind() -> String {
    "Note".into()
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListResearchNotesRequest {
    /// Scope: provide exactly one of `situation_id` or `narrative_id`.
    #[serde(default)]
    pub situation_id: Option<String>,
    #[serde(default)]
    pub narrative_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetResearchNoteRequest {
    /// Note UUID.
    pub note_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateResearchNoteRequest {
    /// Note UUID to patch.
    pub note_id: String,
    /// Patch — keys: kind, body, author, source_id.
    pub patch: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteResearchNoteRequest {
    /// Note UUID.
    pub note_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PromoteChunkToNoteRequest {
    /// Narrative containing the scene + chunk.
    pub narrative_id: String,
    /// Scene to attach the promoted note to.
    pub situation_id: String,
    /// Chunk UUID to promote.
    pub chunk_id: String,
    /// Verbatim quote / excerpt extracted from the chunk.
    pub body: String,
    /// Optional registered source UUID.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Optional kind override (default "Quote").
    #[serde(default)]
    pub kind: Option<String>,
    /// Optional author.
    #[serde(default)]
    pub author: Option<String>,
}

// --- Editing engine ------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProposeEditRequest {
    /// Situation (scene) to edit.
    pub situation_id: String,
    /// Freeform writer instruction describing the rewrite.
    pub instruction: String,
    /// Optional named style preset (minimal | lyrical | punchy | formal |
    /// interior | cinematic) — when set, the edit operation is `StyleTransfer`
    /// with a preset; when missing, it's a `Rewrite` with `instruction`.
    #[serde(default)]
    pub style_preset: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApplyEditRequest {
    /// Full `EditProposal` object (as returned by `propose_edit`).
    pub proposal: serde_json::Value,
    /// Optional author attribution for the revision commit.
    #[serde(default)]
    pub author: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EstimateEditTokensRequest {
    /// Scene UUID.
    pub situation_id: String,
    /// Instruction the edit prompt will carry.
    pub instruction: String,
    /// Optional style preset (same semantics as `propose_edit`).
    #[serde(default)]
    pub style_preset: Option<String>,
}

// --- Revision completion ------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CommitRevisionRequest {
    /// Narrative to commit.
    pub narrative_id: String,
    /// Commit message (required, non-empty).
    pub message: String,
    /// Optional author attribution.
    #[serde(default)]
    pub author: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffRevisionsRequest {
    /// Narrative ID (validated against both revisions).
    pub narrative_id: String,
    /// Earlier revision UUID.
    pub from_rev: String,
    /// Later revision UUID.
    pub to_rev: String,
}

// --- Workshop completion ------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListWorkshopReportsRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetWorkshopReportRequest {
    /// Report UUID.
    pub report_id: String,
}

// --- Cost ledger ledger-of-records --------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListCostLedgerRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Max entries (default 50, max 500).
    #[serde(default)]
    pub limit: Option<usize>,
}

// --- Compile -------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListCompileProfilesRequest {
    /// Narrative ID.
    pub narrative_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompileNarrativeRequest {
    /// Narrative ID to compile.
    pub narrative_id: String,
    /// Output format: "markdown" | "epub" | "docx".
    #[serde(default = "default_compile_format")]
    pub format: String,
    /// Optional profile UUID. If missing, a default profile is synthesised.
    #[serde(default)]
    pub profile_id: Option<String>,
}

fn default_compile_format() -> String {
    "markdown".into()
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpsertCompileProfileRequest {
    /// Narrative ID the profile belongs to.
    pub narrative_id: String,
    /// If set, update the existing profile; if omitted, create a new one.
    #[serde(default)]
    pub profile_id: Option<String>,
    /// Full or partial profile object — `CompileProfile` on create,
    /// `ProfilePatch` on update. On create, `name` is required.
    pub patch: serde_json::Value,
}

// --- Templates ----------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct InstantiateTemplateRequest {
    /// Template UUID (builtin or stored).
    pub template_id: String,
    /// Map of `slot_id -> entity_uuid`.
    pub bindings: std::collections::HashMap<String, String>,
}

// --- Secondary (skeleton / dedup / debug-fixes / reorder) ---

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindDuplicatesRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Optional similarity threshold (0.0–1.0, default 0.7).
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Optional max number of candidate pairs (default 200).
    #[serde(default)]
    pub max_candidates: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApplyNarrativeFixRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Full `SuggestedFix` object from `suggest_narrative_fixes`.
    pub fix: serde_json::Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApplyReorderRequest {
    /// Narrative ID.
    pub narrative_id: String,
    /// Array of `{situation_id, parent_id?}` entries in the new desired order.
    pub entries: serde_json::Value,
}

// --- EATH Phase 10 — Synthetic Hypergraph MCP tools ----------
//
// Wire-shape mirrors of `synth::hybrid::HybridComponent` and
// `analysis::higher_order_contagion::HigherOrderSirParams`. We re-define them
// here (instead of deriving JsonSchema on the engine types) because the
// `schemars` dep is gated behind the `mcp` feature — adding it to the synth
// and analysis modules would leak the gate into engine code. Round-tripping
// happens via `serde_json::to_value` in the handlers.

/// One mixture component for `generate_hybrid_narrative`. Mirrors
/// [`crate::synth::hybrid::HybridComponent`] field-for-field.
///
/// Re-defined here (rather than re-exported) because the engine type lives in
/// a layer that doesn't depend on `schemars`. `Serialize` is derived so the
/// handler can `to_value(&req.components)` and pass the JSON straight to the
/// engine — round-trips losslessly because field names match.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct HybridComponentRequest {
    /// Source narrative whose calibrated params drive this component.
    pub narrative_id: String,
    /// Surrogate model name (typically `"eath"`).
    pub model: String,
    /// Mixture weight in `[0, 1]`. Sum across components must equal 1.0
    /// within `1e-6` tolerance.
    pub weight: f32,
}

/// Parameters for the `calibrate_surrogate` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CalibrateSurrogateRequest {
    /// Source narrative to calibrate the surrogate model against.
    pub narrative_id: String,
    /// Surrogate model name. Defaults to `"eath"`.
    #[serde(default)]
    pub model: Option<String>,
}

/// Parameters for the `generate_synthetic_narrative` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateSyntheticNarrativeRequest {
    /// Source narrative whose calibrated params drive generation. Required
    /// unless inline `params` is supplied.
    pub source_narrative_id: String,
    /// Where to write the synthetic narrative.
    pub output_narrative_id: String,
    /// Surrogate model name. Defaults to `"eath"`.
    #[serde(default)]
    pub model: Option<String>,
    /// Inline param override — wins over loaded params when present.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Deterministic-replay seed. Engine generates one when absent.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Number of generation steps. Engine defaults to 100 when absent.
    #[serde(default)]
    pub num_steps: Option<usize>,
    /// Synthetic-narrative label prefix. Engine defaults to `"synth"` when absent.
    #[serde(default)]
    pub label_prefix: Option<String>,
}

/// Parameters for the `generate_hybrid_narrative` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateHybridNarrativeRequest {
    /// Mixture components — each carries `(narrative_id, model, weight)`.
    /// Weights must sum to 1.0 within `1e-6`.
    pub components: Vec<HybridComponentRequest>,
    /// Where to write the synthetic output.
    pub output_narrative_id: String,
    /// Deterministic-replay seed. Engine generates one when absent.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Number of generation steps. Engine defaults to 100 when absent.
    #[serde(default)]
    pub num_steps: Option<usize>,
}

/// Parameters for the `list_synthetic_runs` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListSyntheticRunsRequest {
    /// Source narrative whose runs to list.
    pub narrative_id: String,
    /// Page size. Defaults to 50, clamped to [1, 1000].
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Parameters for the `get_fidelity_report` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFidelityReportRequest {
    /// Source narrative the report belongs to.
    pub narrative_id: String,
    /// Run UUID to fetch the fidelity report for. Required — fidelity is
    /// keyed `(narrative_id, run_id)`.
    pub run_id: String,
}

/// Parameters for the `compute_pattern_significance` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputePatternSignificanceRequest {
    /// Source narrative being tested.
    pub narrative_id: String,
    /// Metric to test: `temporal_motifs` | `communities` | `patterns`.
    pub metric: String,
    /// Number of synthetic samples. Defaults to 100, clamped to [1, 1000].
    #[serde(default)]
    pub k: Option<u16>,
    /// Surrogate model. Defaults to `"eath"`.
    #[serde(default)]
    pub model: Option<String>,
    /// Inline EathParams override. When present, the engine skips
    /// `load_params` and uses this directly.
    #[serde(default)]
    pub params_override: Option<serde_json::Value>,
}

/// Parameters for the `simulate_higher_order_contagion` tool. Mirrors
/// `POST /synth/contagion-significance` — runs K synthetic samples comparing
/// a real narrative's higher-order SIR contagion outcome against the
/// EATH null distribution.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimulateHigherOrderContagionRequest {
    /// Source narrative being tested.
    pub narrative_id: String,
    /// `HigherOrderSirParams` blob (per-size beta, gamma, threshold rule,
    /// seed strategy, max_steps, rng_seed). Pass through opaquely; the engine
    /// deserializes via `serde_json::from_value`.
    pub params: serde_json::Value,
    /// Number of synthetic samples. Defaults to 100, clamped to [1, 1000].
    #[serde(default)]
    pub k: Option<u16>,
    /// Surrogate model. Defaults to `"eath"`.
    #[serde(default)]
    pub model: Option<String>,
}

/// Parameters for the `compute_dual_significance` tool (Phase 13c). Submits
/// a `SurrogateDualSignificance` job that runs K samples per requested null
/// model and reports per-model + combined verdicts.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeDualSignificanceRequest {
    /// Source narrative being tested.
    pub narrative_id: String,
    /// Metric to test: `temporal_motifs` | `communities` | `patterns`.
    /// Contagion is rejected — use `simulate_higher_order_contagion` instead.
    pub metric: String,
    /// Per-model number of synthetic samples. Defaults to 100, clamped at
    /// 1000 per model.
    #[serde(default)]
    pub k_per_model: Option<u16>,
    /// Surrogate models to compare against. Defaults to `["eath", "nudhy"]`
    /// when omitted or empty.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// Parameters for the `reconstruct_hypergraph` tool (EATH Extension Phase 15c).
/// Submits a `HypergraphReconstruction` job that recovers latent
/// hyperedges from observed entity time-series via the THIS / SINDy method
/// (Delabays et al., Nat. Commun. 16:2691, 2025).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReconstructHypergraphRequest {
    /// Source narrative whose entity time-series to reconstruct from. Required.
    pub narrative_id: String,
    /// Optional partial `ReconstructionParams` blob — every field has a
    /// `serde(default)`, so callers can omit this entirely to get the
    /// engine's recommended defaults (max_order=3,
    /// observation=ParticipationRate, lambda auto-selected).
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

/// Parameters for the `simulate_opinion_dynamics` tool
/// (EATH Extension Phase 16c). Synchronously runs one bounded-confidence
/// opinion-dynamics simulation on the source narrative's entity-situation
/// hypergraph. Returns an inline `OpinionDynamicsReport`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimulateOpinionDynamicsRequest {
    /// Source narrative whose entities + situations form the hypergraph. Required.
    pub narrative_id: String,
    /// Optional partial `OpinionDynamicsParams` blob — every field has a
    /// `serde(default)`, so callers can omit this entirely to get the
    /// documented defaults (PairwiseWithin, c=0.3, μ=0.5, etc.).
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

/// Parameters for the `simulate_opinion_phase_transition` tool
/// (EATH Extension Phase 16c). Sweeps the confidence bound `c` over
/// `c_range = [start, end, num_points]` and reports per-c convergence times +
/// the inferred critical-c spike.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimulateOpinionPhaseTransitionRequest {
    /// Source narrative. Required.
    pub narrative_id: String,
    /// `[c_start, c_end, num_points]` — `num_points >= 2`,
    /// `0 < c_start < c_end < 1`.
    pub c_range: [serde_json::Value; 3],
    /// Optional base `OpinionDynamicsParams` blob.
    #[serde(default)]
    pub base_params: Option<serde_json::Value>,
}

/// Parameters for the `compute_bistability_significance` tool (Phase 14).
/// Submits a `SurrogateBistabilitySignificance` job that runs a
/// forward-backward β-sweep on the source narrative AND on K surrogate
/// samples per requested null model, then reports per-model quantiles for
/// `bistable_interval` width and `max_hysteresis_gap`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeBistabilitySignificanceRequest {
    /// Source narrative being tested.
    pub narrative_id: String,
    /// `BistabilitySweepParams` blob — see
    /// [`crate::analysis::contagion_bistability::BistabilitySweepParams`].
    pub params: serde_json::Value,
    /// Number of synthetic samples per model. Defaults to 50 (smaller than
    /// the structural-significance default because each sample runs an
    /// entire sweep). Clamped at 500.
    #[serde(default)]
    pub k: Option<u16>,
    /// Surrogate models to compare against. Defaults to `["eath", "nudhy"]`
    /// when omitted or empty.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// Parameters for the `fuzzy_probability` tool (Fuzzy Sprint Phase 10).
/// Synchronously computes the Cao–Holčapek base-case fuzzy probability
/// `P_fuzzy(E) = Σ μ_E(e) · P(e)` on a discrete distribution over
/// entity UUIDs. Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyProbabilityRequest {
    /// Narrative the event and outcomes belong to.
    pub narrative_id: String,
    /// FuzzyEvent `{predicate_kind, predicate_payload}`. See
    /// [`crate::fuzzy::hybrid::FuzzyEventPredicate`] for payload shape
    /// per kind. `predicate_kind ∈ {quantifier, mamdani_rule, custom}`.
    pub event: serde_json::Value,
    /// Discrete ProbDist shape: `{ "kind": "discrete", "outcomes":
    /// [["<uuid>", <f64>], ...] }`. Must sum to 1.0 within 1e-9;
    /// P ∈ [0, 1]; no duplicate UUIDs.
    pub distribution: serde_json::Value,
    /// Optional t-norm override name (e.g. `"godel"`). Phase 10 accepts
    /// but does not consume it — future composition phases will.
    #[serde(default)]
    pub tnorm: Option<String>,
}

// ─── Fuzzy Sprint Phase 11 — MCP request types ──────────────────────────
//
// Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
//        [duboisprade1989fuzzyallen] [novak2008quantifiers]
//        [murinovanovak2014peterson] [belohlavek2004fuzzyfca]
//        [mamdani1975mamdani].

/// Parameters for the `fuzzy_list_tnorms` tool — mirrors `GET /fuzzy/tnorms`.
/// Takes no args; the empty schema is still required by rmcp's Parameters
/// wrapper.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct FuzzyListTnormsRequest {}

/// Parameters for the `fuzzy_list_aggregators` tool.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct FuzzyListAggregatorsRequest {}

/// Parameters for the `fuzzy_get_config` tool — no args.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct FuzzyGetConfigRequest {}

/// Parameters for the `fuzzy_set_config` tool (`PUT /fuzzy/config`).
/// `reset = true` short-circuits to the Gödel / Mean factory default.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzySetConfigRequest {
    /// Canonical t-norm name (`"godel"` / `"goguen"` / `"lukasiewicz"` /
    /// `"hamacher"`). Omit to leave unchanged.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Canonical aggregator name (`"mean"` / `"median"` / `"owa"` /
    /// `"choquet"` / `"tnorm_reduce"` / `"tconorm_reduce"`).
    #[serde(default)]
    pub aggregator: Option<String>,
    /// Named fuzzy measure reference; `None` leaves unchanged, explicit
    /// `null` clears the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measure: Option<Option<String>>,
    /// Reset to the Gödel / Mean factory default.
    #[serde(default)]
    pub reset: bool,
}

/// Parameters for the `fuzzy_create_measure` tool (`POST /fuzzy/measures`).
/// Validation enforces `μ(∅)=0`, `μ(N)=1`, and monotonicity; rejected
/// measures surface messages containing the literal word "monotonicity".
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyCreateMeasureRequest {
    /// Canonical name (used as the KV key tail). No `/`, whitespace, or
    /// newlines.
    pub name: String,
    /// Cardinality of the underlying universe (must equal
    /// `|values| = 2^n`). Phase 4 caps `n ≤ 16` (exact Choquet at 10,
    /// Monte-Carlo above).
    pub n: u8,
    /// Measure values in subset-bitmask order — `values[0]` is `μ(∅)`,
    /// `values[2^n - 1]` is `μ(N)`.
    pub values: Vec<f64>,
}

/// Parameters for the `fuzzy_list_measures` tool — no args.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct FuzzyListMeasuresRequest {}

/// Parameters for the `fuzzy_aggregate` tool (`POST /fuzzy/aggregate`).
/// One-shot aggregation over a caller-supplied vector. Caps `|xs| ≤ 1000`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyAggregateRequest {
    /// Input vector; every element must be finite. Empty vectors are
    /// rejected.
    pub xs: Vec<f64>,
    /// Canonical aggregator name — one of `mean` / `median` / `owa` /
    /// `choquet` / `tnorm_reduce` / `tconorm_reduce`.
    pub aggregator: String,
    /// Optional t-norm name (required by `tnorm_reduce` / `tconorm_reduce`).
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Named measure reference (required by `choquet`).
    #[serde(default)]
    pub measure: Option<String>,
    /// Explicit OWA weights. Required by `owa`; must have length `|xs|`.
    #[serde(default)]
    pub owa_weights: Option<Vec<f64>>,
    /// Optional RNG seed for the Choquet Monte-Carlo path.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Parameters for the `fuzzy_allen_gradation` tool
/// (`POST /analysis/fuzzy-allen`). Computes the 13-dim graded Allen
/// relation vector between two situations.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyAllenGradationRequest {
    /// Narrative the two situations belong to.
    pub narrative_id: String,
    /// UUID of the first situation.
    pub a_id: String,
    /// UUID of the second situation.
    pub b_id: String,
}

/// Parameters for the `fuzzy_quantify` tool (`POST /fuzzy/quantify`).
/// Evaluates an intermediate quantifier (Novák-Murinová) over a
/// narrative's entity domain.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyQuantifyRequest {
    /// Source narrative — the entity domain to quantify over.
    pub narrative_id: String,
    /// Quantifier name: `most` / `many` / `almost_all` / `few`.
    pub quantifier: String,
    /// Optional entity-type filter (`Actor` / `Location` / ...).
    #[serde(default)]
    pub entity_type: Option<String>,
    /// Crisp predicate spec (Phase 6 scope) — `"confidence>0.7"`,
    /// `"maturity=Confirmed"`, or a dotted property path. Empty string
    /// maps every entity to μ=1.
    #[serde(default)]
    pub r#where: Option<String>,
    /// Optional echo label.
    #[serde(default)]
    pub label: Option<String>,
}

/// Parameters for the `fuzzy_verify_syllogism` tool
/// (`POST /fuzzy/syllogism/verify`).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyVerifySyllogismRequest {
    /// Narrative against which the syllogism's predicates resolve.
    pub narrative_id: String,
    /// Tiny-DSL string for the major premise
    /// (e.g. `"ALL type:Actor IS type:Actor"`).
    pub major: String,
    /// Tiny-DSL string for the minor premise.
    pub minor: String,
    /// Tiny-DSL string for the conclusion.
    pub conclusion: String,
    /// Optional validity threshold, default `0.5`, must be in `[0, 1]`.
    #[serde(default)]
    pub threshold: Option<f64>,
    /// Optional t-norm name override. Defaults to `"godel"`.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional figure hint (`"I"` / `"II"` / `"III"` / `"IV"`).
    #[serde(default)]
    pub figure_hint: Option<String>,
}

/// Parameters for the `fuzzy_build_lattice` tool
/// (`POST /fuzzy/fca/lattice`). Builds + persists a graded concept
/// lattice via Bělohlávek Galois closure under the configured t-norm.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyBuildLatticeRequest {
    /// Source narrative.
    pub narrative_id: String,
    /// Optional entity-type filter.
    #[serde(default)]
    pub entity_type: Option<String>,
    /// Optional attribute allowlist — restricts the formal context's
    /// attribute axis to the named properties.
    #[serde(default)]
    pub attribute_allowlist: Option<Vec<String>>,
    /// Optional concept-extent-size prune threshold.
    #[serde(default)]
    pub threshold: Option<usize>,
    /// Optional t-norm name (default `"godel"`).
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Opt-in large-context mode (> soft cap, ≤ hard cap).
    #[serde(default)]
    pub large_context: bool,
}

/// Parameters for the `fuzzy_create_rule` tool (`POST /fuzzy/rules`).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyCreateRuleRequest {
    /// Human-readable name for the rule.
    pub name: String,
    /// Narrative the rule is scoped to.
    pub narrative_id: String,
    /// Antecedent — `[{ variable_path, membership, linguistic_term }, ...]`.
    /// See [`crate::fuzzy::rules_types::FuzzyCondition`].
    pub antecedent: serde_json::Value,
    /// Consequent — `{ variable, membership, linguistic_term }`.
    /// See [`crate::fuzzy::rules_types::FuzzyOutput`].
    pub consequent: serde_json::Value,
    /// Optional t-norm name override for rule-strength folding.
    #[serde(default)]
    pub tnorm: Option<String>,
    /// Optional enabled flag (default `true`).
    #[serde(default)]
    pub enabled: Option<bool>,
}

/// Parameters for the `fuzzy_evaluate_rules` tool
/// (`POST /fuzzy/rules/{nid}/evaluate`).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyEvaluateRulesRequest {
    /// Narrative containing the rule set.
    pub narrative_id: String,
    /// UUID of the entity to evaluate against.
    pub entity_id: String,
    /// Optional whitelist of rule UUIDs (strings) — omit to evaluate
    /// every enabled rule in the narrative.
    #[serde(default)]
    pub rule_ids: Option<Vec<String>>,
    /// Optional aggregator applied to firing strengths. Shape mirrors
    /// [`crate::fuzzy::aggregation::AggregatorKind`]'s externally-tagged
    /// JSON — e.g. `"Mean"`, `"Median"`, `{"Owa": [...]}`, `{"Choquet":
    /// {...}}`, `{"TNormReduce": {"kind": "godel"}}`. Passed through
    /// opaquely because the tagged-union enum doesn't implement
    /// `JsonSchema` (would require a derive on a downstream crate).
    #[serde(default)]
    pub firing_aggregator: Option<serde_json::Value>,
}

// ─── Graded Acceptability Sprint Phase 5 — MCP request types ────────────
//
// Five new tools wrap REST endpoints shipped in Phases 3 + 4:
//   * `argumentation_gradual`     → POST /analysis/argumentation/gradual
//   * `fuzzy_learn_measure`       → POST /fuzzy/measures/learn
//   * `fuzzy_get_measure_version` → GET  /fuzzy/measures/{name}?version=N
//   * `fuzzy_list_measure_versions` → GET /fuzzy/measures/{name}/versions
//   * `temporal_ordhorn_closure`  → POST /temporal/ordhorn/closure
//
// `GradualSemanticsKind`, `OrdHornNetwork`, and `TNormKind` do not derive
// `JsonSchema`, so we accept them as opaque `serde_json::Value` blobs and
// validate inside the embedded impl. Same pattern as
// `FuzzyEvaluateRulesRequest::firing_aggregator`.
//
// Cites: [amgoud2013ranking] [grabisch1996choquet] [nebel1995ordhorn].

/// Parameters for the `argumentation_gradual` tool.
/// Mirrors `POST /analysis/argumentation/gradual`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArgumentationGradualRequest {
    /// Source narrative whose contention edges form the framework.
    pub narrative_id: String,
    /// Gradual-semantics variant. Externally-tagged JSON matching
    /// [`crate::analysis::argumentation_gradual::GradualSemanticsKind`]:
    /// `"HCategoriser"`, `"MaxBased"`, `"CardBased"`, or
    /// `{"WeightedHCategoriser": {"weights": [..]}}`. Passed through
    /// opaquely because the enum does not derive `JsonSchema`.
    pub gradual_semantics: serde_json::Value,
    /// Optional t-norm override blob matching
    /// [`crate::fuzzy::tnorm::TNormKind`]'s externally-tagged JSON
    /// (`{"kind":"godel"}` / `{"kind":"hamacher","param":1.5}` / …).
    /// Default `None` reproduces the canonical Gödel formulas
    /// bit-identically.
    #[serde(default)]
    pub tnorm: Option<serde_json::Value>,
}

/// Parameters for the `fuzzy_learn_measure` tool.
/// Mirrors `POST /fuzzy/measures/learn`. Persists a Choquet capacity
/// fitted from a ranking-supervised dataset; auto-increments version on
/// retrain under an existing name.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyLearnMeasureRequest {
    /// Persistence name. Rejects `/`, whitespace, and newlines.
    pub name: String,
    /// Universe size. Capped at 6 by the in-tree PGD path; larger
    /// universes return InvalidInput with a pointer to k-additive.
    pub n: u8,
    /// `(input_vec, rank)` pairs. Lower rank = more strongly coordinated.
    pub dataset: Vec<(Vec<f64>, u32)>,
    /// Caller-supplied dataset identifier — drives the deterministic
    /// 50 / 50 train / test split seed.
    pub dataset_id: String,
}

/// Parameters for the `fuzzy_get_measure_version` tool.
/// Mirrors `GET /fuzzy/measures/{name}?version=N`. Omitting `version`
/// returns the latest pointer (legacy behaviour).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyGetMeasureVersionRequest {
    /// Measure name (KV key tail).
    pub name: String,
    /// Optional version slice. Absent → latest pointer.
    #[serde(default)]
    pub version: Option<u32>,
}

/// Parameters for the `fuzzy_list_measure_versions` tool.
/// Mirrors `GET /fuzzy/measures/{name}/versions`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FuzzyListMeasureVersionsRequest {
    /// Measure name (KV key tail).
    pub name: String,
}

/// Parameters for the `temporal_ordhorn_closure` tool.
/// Mirrors `POST /temporal/ordhorn/closure`. Pure transformation —
/// the handler never touches the hypergraph store.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TemporalOrdHornClosureRequest {
    /// Constraint network blob matching
    /// [`crate::temporal::ordhorn::OrdHornNetwork`]'s JSON shape:
    /// `{"n": <usize>, "constraints": [{"a": <usize>, "b": <usize>,
    /// "relations": ["Before", "Meets", ...]}]}`. Passed through
    /// opaquely — the type doesn't derive `JsonSchema` (and importing
    /// `schemars` into the temporal crate isn't worth a one-tool
    /// surface).
    pub network: serde_json::Value,
}
