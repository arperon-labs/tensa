//! LLM response types and parsing for narrative extraction.
//!
//! Defines the intermediate types for data extracted by the LLM,
//! plus parsing and validation logic.

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{Result, TensaError};
use crate::types::*;

/// Parse a role name (case-insensitive) to a `Role`, mapping unknown
/// variants to `Custom(String)`. Used by both the serde deserializer and
/// the pipeline when applying anaphora resolutions.
pub fn parse_role_lenient(s: &str) -> Role {
    match s {
        "Protagonist" => Role::Protagonist,
        "Antagonist" => Role::Antagonist,
        "Witness" => Role::Witness,
        "Target" => Role::Target,
        "Instrument" => Role::Instrument,
        "Confidant" => Role::Confidant,
        "Informant" => Role::Informant,
        "Recipient" => Role::Recipient,
        "Bystander" => Role::Bystander,
        "SubjectOfDiscussion" => Role::SubjectOfDiscussion,
        other => Role::Custom(other.to_string()),
    }
}

/// Deserialize a Role, mapping unknown variants to Custom(String).
fn deserialize_role_lenient<'de, D>(deserializer: D) -> std::result::Result<Role, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(parse_role_lenient(&s))
}

/// Deserialize content_blocks that may be strings or ContentBlock objects.
fn deserialize_content_blocks_lenient<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<ContentBlock>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(ContentBlock::text(&s)),
            serde_json::Value::Object(_) => serde_json::from_value(v).ok(),
            _ => None,
        })
        .collect())
}

/// Complete extraction from a single text chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeExtraction {
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    #[serde(default)]
    pub situations: Vec<ExtractedSituation>,
    #[serde(default)]
    pub participations: Vec<ExtractedParticipation>,
    #[serde(default)]
    pub causal_links: Vec<ExtractedCausalLink>,
    #[serde(default)]
    pub temporal_relations: Vec<ExtractedTemporalRelation>,
}

/// An entity extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(deserialize_with = "deserialize_entity_type_lenient")]
    pub entity_type: EntityType,
    #[serde(default = "default_properties")]
    pub properties: serde_json::Value,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_properties() -> serde_json::Value {
    serde_json::json!({})
}

fn default_confidence() -> f32 {
    0.5
}

/// A situation (event/scene) extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedSituation {
    /// Short descriptive name (5-8 words).
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: String,
    pub temporal_marker: Option<String>,
    pub location: Option<String>,
    #[serde(
        default = "default_narrative_level",
        deserialize_with = "deserialize_narrative_level_lenient"
    )]
    pub narrative_level: NarrativeLevel,
    #[serde(default, deserialize_with = "deserialize_content_blocks_lenient")]
    pub content_blocks: Vec<ContentBlock>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Verbatim opening fragment (~8–12 words) used to locate the situation's
    /// exact byte span inside its source chunk. Optional — when absent or
    /// unmatched, the pipeline falls back to the whole-chunk span.
    #[serde(default)]
    pub text_start: Option<String>,
    /// Verbatim closing fragment (~8–12 words) pairing with `text_start`.
    #[serde(default)]
    pub text_end: Option<String>,
}

fn deserialize_entity_type_lenient<'de, D>(
    deserializer: D,
) -> std::result::Result<EntityType, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "Actor" | "Person" | "Character" => Ok(EntityType::Actor),
        "Location" | "Place" | "Geography" => Ok(EntityType::Location),
        "Artifact" | "Object" | "Item" | "Document" => Ok(EntityType::Artifact),
        "Concept" | "Idea" | "Theme" | "Abstract" => Ok(EntityType::Concept),
        "Organization" | "Group" | "Faction" | "Institution" => Ok(EntityType::Organization),
        _ => Ok(EntityType::Concept), // fallback for LLM-invented types
    }
}

fn default_narrative_level() -> NarrativeLevel {
    NarrativeLevel::Event
}

fn deserialize_narrative_level_lenient<'de, D>(
    deserializer: D,
) -> std::result::Result<NarrativeLevel, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "Story" => Ok(NarrativeLevel::Story),
        "Arc" => Ok(NarrativeLevel::Arc),
        "Sequence" => Ok(NarrativeLevel::Sequence),
        "Scene" => Ok(NarrativeLevel::Scene),
        "Beat" => Ok(NarrativeLevel::Beat),
        "Event" => Ok(NarrativeLevel::Event),
        _ => Ok(NarrativeLevel::Scene), // fallback for LLM-invented levels
    }
}

/// A participation link extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedParticipation {
    #[serde(default)]
    pub entity_name: String,
    #[serde(default)]
    pub situation_index: usize,
    #[serde(deserialize_with = "deserialize_role_lenient")]
    pub role: Role,
    pub action: Option<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// A causal link extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedCausalLink {
    #[serde(default, alias = "from", alias = "source_index")]
    pub from_situation_index: usize,
    #[serde(default, alias = "to", alias = "target_index")]
    pub to_situation_index: usize,
    pub mechanism: Option<String>,
    #[serde(
        default = "default_causal_type",
        deserialize_with = "deserialize_causal_type_lenient"
    )]
    pub causal_type: CausalType,
    #[serde(default = "default_confidence")]
    pub strength: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_causal_type() -> CausalType {
    CausalType::Contributing
}

/// Lenient deserializer for CausalType — maps LLM-invented variants to Contributing.
fn deserialize_causal_type_lenient<'de, D>(
    deserializer: D,
) -> std::result::Result<CausalType, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "Necessary" => Ok(CausalType::Necessary),
        "Sufficient" => Ok(CausalType::Sufficient),
        "Contributing" => Ok(CausalType::Contributing),
        "Enabling" => Ok(CausalType::Enabling),
        _ => Ok(CausalType::Contributing), // fallback for LLM-invented types like "Temporal"
    }
}

fn deserialize_allen_relation_lenient<'de, D>(
    deserializer: D,
) -> std::result::Result<AllenRelation, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "Before" => Ok(AllenRelation::Before),
        "After" => Ok(AllenRelation::After),
        "Meets" => Ok(AllenRelation::Meets),
        "MetBy" => Ok(AllenRelation::MetBy),
        "Overlaps" => Ok(AllenRelation::Overlaps),
        "OverlappedBy" => Ok(AllenRelation::OverlappedBy),
        "During" => Ok(AllenRelation::During),
        "Contains" => Ok(AllenRelation::Contains),
        "Starts" => Ok(AllenRelation::Starts),
        "StartedBy" => Ok(AllenRelation::StartedBy),
        "Finishes" => Ok(AllenRelation::Finishes),
        "FinishedBy" => Ok(AllenRelation::FinishedBy),
        "Equals" => Ok(AllenRelation::Equals),
        _ => Ok(AllenRelation::Before), // fallback
    }
}

/// A temporal relation extracted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTemporalRelation {
    #[serde(default, alias = "from", alias = "first")]
    pub situation_a_index: usize,
    #[serde(default, alias = "to", alias = "second")]
    pub situation_b_index: usize,
    #[serde(deserialize_with = "deserialize_allen_relation_lenient")]
    pub relation: AllenRelation,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

// ─── Step 2: Enrichment types ────────────────────────────────

/// Enrichment data for a single chunk, produced by the second LLM pass.
/// The LLM sees the step-1 extraction + original text and adds deep annotations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionEnrichment {
    /// Per-entity beliefs/goals/knowledge at the time of this chunk.
    #[serde(default)]
    pub entity_beliefs: Vec<EntityBeliefEnrichment>,
    /// Per-situation game-theoretic classification.
    #[serde(default)]
    pub game_structures: Vec<SituationGameEnrichment>,
    /// Per-situation discourse/narratology annotation.
    #[serde(default)]
    pub discourse: Vec<SituationDiscourseEnrichment>,
    /// Per-participation knowledge (what actor knows before/learns/reveals).
    #[serde(default)]
    pub info_sets: Vec<ParticipationInfoEnrichment>,
    /// Additional causal links the first pass missed.
    #[serde(default)]
    pub extra_causal_links: Vec<ExtractedCausalLink>,
    /// Per-situation outcome modeling.
    #[serde(default)]
    pub outcomes: Vec<SituationOutcomeEnrichment>,
    /// Complete temporal chain: Allen relations between ALL situation pairs in this chunk.
    #[serde(default)]
    pub temporal_chain: Vec<ExtractedTemporalRelation>,
    /// Normalized temporal markers (resolved dates where possible).
    #[serde(default)]
    pub temporal_normalizations: Vec<TemporalNormalization>,
}

/// What an entity believes/knows/wants at the time of this chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityBeliefEnrichment {
    pub entity_name: String,
    /// Key beliefs this entity holds.
    #[serde(default)]
    pub beliefs: Vec<String>,
    /// Goals or motivations driving their actions.
    #[serde(default)]
    pub goals: Vec<String>,
    /// What they are mistaken about or don't know.
    #[serde(default)]
    pub misconceptions: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// Game-theoretic structure for a situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationGameEnrichment {
    /// 1-based situation index.
    #[serde(default)]
    pub situation_index: usize,
    /// Game classification (e.g. "Bargaining", "Signaling", "ZeroSum").
    pub game_type: String,
    /// Information structure (e.g. "Incomplete", "AsymmetricBecomingComplete").
    pub info_structure: String,
    /// Free-text description of the strategic interaction.
    pub description: Option<String>,
}

/// Discourse/narratology annotation for a situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationDiscourseEnrichment {
    /// 1-based situation index.
    #[serde(default)]
    pub situation_index: usize,
    /// Temporal order relative to story time: analepsis, prolepsis, simultaneous.
    pub order: Option<String>,
    /// Duration mode: scene, summary, ellipsis, pause, stretch.
    pub duration: Option<String>,
    /// Whose perspective: entity name (resolved to UUID later).
    pub focalization: Option<String>,
    /// Narrative voice: homodiegetic, heterodiegetic.
    pub voice: Option<String>,
}

/// What an actor knows before/learns during/reveals in a situation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationInfoEnrichment {
    pub entity_name: String,
    /// 1-based situation index.
    #[serde(default)]
    pub situation_index: usize,
    /// Facts the entity knows going into this situation.
    #[serde(default)]
    pub knows_before: Vec<String>,
    /// Facts the entity learns during this situation.
    #[serde(default)]
    pub learns: Vec<String>,
    /// Facts the entity reveals to others during this situation.
    #[serde(default)]
    pub reveals: Vec<String>,
}

/// Outcome modeling for a situation (what must/might happen).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationOutcomeEnrichment {
    /// 1-based situation index.
    #[serde(default)]
    pub situation_index: usize,
    /// What necessarily follows from this situation.
    pub deterministic: Option<String>,
    /// Alternative outcomes and rough likelihoods.
    #[serde(default)]
    pub alternatives: Vec<AlternativeOutcome>,
}

/// An alternative outcome with an estimated probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeOutcome {
    pub description: String,
    pub probability: f32,
}

/// Normalized temporal marker for a situation (from enrichment step 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalNormalization {
    /// 1-based situation index.
    #[serde(default)]
    pub situation_index: usize,
    /// Normalized date/time if determinable (ISO 8601, or fictional-but-parseable).
    pub normalized_date: Option<String>,
    /// Relative temporal description if no absolute date ("three days after situation 2").
    pub relative_description: Option<String>,
}

/// Cross-chunk temporal reconciliation result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalReconciliation {
    /// Ordered pairs of (situation_name, situation_name, Allen relation).
    #[serde(default)]
    pub relations: Vec<ReconciledTemporalRelation>,
    /// Global timeline events with normalized dates.
    #[serde(default)]
    pub timeline: Vec<TimelineEvent>,
}

/// A temporal relation between two named situations (cross-chunk).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciledTemporalRelation {
    /// Name or description of situation A.
    pub situation_a: String,
    /// Name or description of situation B.
    pub situation_b: String,
    /// Allen temporal relation from A to B.
    pub relation: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// A pinned event in the global timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Name or description of the situation.
    pub situation: String,
    /// Normalized date/time (best effort, may be fictional).
    pub date: String,
    /// How confident we are in this dating.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// Parse a temporal reconciliation response.
pub fn parse_reconciliation_response(raw: &str) -> Result<TemporalReconciliation> {
    let json_str = extract_json_from_response(raw);
    if let Ok(rec) = serde_json::from_str::<TemporalReconciliation>(&json_str) {
        return Ok(rec);
    }
    let normalized = normalize_llm_json(&json_str);
    if let Ok(rec) = serde_json::from_str::<TemporalReconciliation>(&normalized) {
        return Ok(rec);
    }
    let fixed = fix_truncated_json(&normalized);
    serde_json::from_str(&fixed).map_err(|e| {
        TensaError::ExtractionError(format!("Failed to parse reconciliation JSON: {}", e))
    })
}

// ─── Session Reconciliation types ────────────────────────────

/// Full reconciliation from a SingleSession final turn.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionReconciliation {
    #[serde(default)]
    pub entity_merges: Vec<EntityMerge>,
    #[serde(default)]
    pub timeline: Vec<TimelineEvent>,
    #[serde(default)]
    pub confidence_adjustments: Vec<ConfidenceAdjustment>,
    #[serde(default)]
    pub cross_chunk_causal_links: Vec<ExtractedCausalLink>,
    /// Descriptive-reference resolutions: "a figure" → a named entity.
    /// Applied as additional Participations during the pipeline's
    /// reconciliation-consumption step, fixing participation gaps where
    /// the initial per-chunk extraction could not resolve an anaphor.
    #[serde(default)]
    pub anaphora_resolutions: Vec<AnaphoraResolution>,
    /// Free-text warnings about coverage gaps (e.g. under-extracted climax,
    /// main character with few participations). Stored on the job report
    /// so reviewers can act on them; not consumed automatically.
    #[serde(default)]
    pub coverage_warnings: Vec<String>,
}

/// A descriptive-reference ("a figure", "the stranger") resolved to a
/// named entity during session reconciliation. Applied as a new
/// Participation in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnaphoraResolution {
    /// The situation where the descriptive reference appeared. Matched by
    /// name (case-insensitive substring) against extracted situations.
    pub situation_name: String,
    /// The original descriptive phrase from the text (for logging only).
    #[serde(default)]
    pub descriptive_reference: String,
    /// The canonical entity name that the reference resolves to.
    pub resolved_entity: String,
    /// Role for the new Participation.
    #[serde(default)]
    pub role: Option<String>,
    /// Action text for the new Participation.
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

/// Two entity names that should be merged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMerge {
    pub canonical_name: String,
    #[serde(default)]
    pub duplicate_names: Vec<String>,
}

/// A confidence score adjustment with reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAdjustment {
    pub name: String,
    pub adjusted_confidence: f32,
    pub reason: String,
}

/// Narrative-setting hint passed to `canonicalize_places` so the LLM can
/// disambiguate ambiguous place names ("Marseilles" → Marseille, FR vs.
/// Marseilles, IL) without a per-place LLM call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NarrativeSettingHint {
    /// Free-form description of the narrative's setting — country, era, region.
    /// e.g. "Early-19th-century France and Italy (Mediterranean coast)".
    pub setting: String,
    /// Optional ISO 3166-1 alpha-2 country code if the narrative is single-country.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country_hint: Option<String>,
}

/// One row of the LLM's place-canonicalization response.
/// `uuid` is opaque — the geocoder uses it to thread the result back to the
/// originating Situation or Location entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceCanonicalization {
    /// Echoed back verbatim from the request (entity / situation UUID as string).
    pub uuid: String,
    /// Echoed back verbatim from the request — the raw place string.
    pub raw_name: String,
    /// Modern canonical place name suitable for Nominatim ("Marseille").
    pub canonical_name: String,
    /// ISO 3166-1 alpha-2 country code, lowercase ("fr").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country_code: Option<String>,
    /// Optional admin region (state / department / oblast) for further disambiguation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admin_region: Option<String>,
    /// LLM self-reported confidence in the canonicalization (0..=1).
    #[serde(default = "default_canon_confidence")]
    pub confidence: f32,
}

fn default_canon_confidence() -> f32 {
    0.7
}

/// Extract a JSON array (`[...]`) from a raw LLM response — handles markdown
/// fences (```json ... ```), `<think>` blocks, and surrounding prose. Mirrors
/// `extract_json_from_response` but for top-level arrays, which `find('{')` /
/// `rfind('}')` would otherwise mangle.
pub fn extract_json_array_from_response(raw: &str) -> String {
    let cleaned = strip_thinking_tags(raw);
    let trimmed = cleaned.trim();

    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if end >= start {
                return trimmed[start..=end].to_string();
            }
        }
    }
    trimmed.to_string()
}

/// Parse the LLM's canonicalize_places response into `PlaceCanonicalization` rows.
/// On parse failure logs the raw response head + serde error via `tracing::warn!`
/// so silent fall-through to the legacy `Geocoded` path is debuggable.
pub fn parse_canonicalize_places_response(raw: &str) -> Result<Vec<PlaceCanonicalization>> {
    let json_str = extract_json_array_from_response(raw);
    match serde_json::from_str::<Vec<PlaceCanonicalization>>(&json_str) {
        Ok(rows) => return Ok(rows),
        Err(e) => {
            let normalized = normalize_llm_json(&json_str);
            if let Ok(rows) = serde_json::from_str::<Vec<PlaceCanonicalization>>(&normalized) {
                return Ok(rows);
            }
            let fixed = fix_truncated_json(&normalized);
            match serde_json::from_str::<Vec<PlaceCanonicalization>>(&fixed) {
                Ok(rows) => return Ok(rows),
                Err(e2) => {
                    tracing::warn!(
                        first_error = %e,
                        final_error = %e2,
                        head = %json_str.chars().take(400).collect::<String>(),
                        "canonicalize_places parse failed — falling back to direct Nominatim"
                    );
                    Err(TensaError::ExtractionError(format!(
                        "Failed to parse canonicalize_places JSON: {}",
                        e2
                    )))
                }
            }
        }
    }
}

/// Parse a session reconciliation response.
pub fn parse_session_reconciliation_response(raw: &str) -> Result<SessionReconciliation> {
    let json_str = extract_json_from_response(raw);
    if let Ok(rec) = serde_json::from_str::<SessionReconciliation>(&json_str) {
        return Ok(rec);
    }
    let normalized = normalize_llm_json(&json_str);
    if let Ok(rec) = serde_json::from_str::<SessionReconciliation>(&normalized) {
        return Ok(rec);
    }
    let fixed = fix_truncated_json(&normalized);
    serde_json::from_str(&fixed).map_err(|e| {
        TensaError::ExtractionError(format!(
            "Failed to parse session reconciliation JSON: {}",
            e
        ))
    })
}

/// Parse an LLM enrichment response.
pub fn parse_enrichment_response(raw: &str) -> Result<ExtractionEnrichment> {
    let json_str = extract_json_from_response(raw);
    if let Ok(enr) = serde_json::from_str::<ExtractionEnrichment>(&json_str) {
        return Ok(enr);
    }
    let normalized = normalize_llm_json(&json_str);
    if let Ok(enr) = serde_json::from_str::<ExtractionEnrichment>(&normalized) {
        return Ok(enr);
    }
    let fixed = fix_truncated_json(&normalized);
    serde_json::from_str(&fixed)
        .map_err(|e| TensaError::ExtractionError(format!("Failed to parse enrichment JSON: {}", e)))
}

/// Convert 1-based situation indices to 0-based in enrichment data.
pub fn repair_enrichment(enrichment: &mut ExtractionEnrichment) {
    let convert = |idx: &mut usize| {
        if *idx >= 1 {
            *idx -= 1;
        }
    };
    for g in &mut enrichment.game_structures {
        convert(&mut g.situation_index);
    }
    for d in &mut enrichment.discourse {
        convert(&mut d.situation_index);
    }
    for i in &mut enrichment.info_sets {
        convert(&mut i.situation_index);
    }
    for o in &mut enrichment.outcomes {
        convert(&mut o.situation_index);
    }
    for c in &mut enrichment.extra_causal_links {
        if c.from_situation_index >= 1 {
            c.from_situation_index -= 1;
        }
        if c.to_situation_index >= 1 {
            c.to_situation_index -= 1;
        }
    }
    for t in &mut enrichment.temporal_chain {
        convert(&mut t.situation_a_index);
        convert(&mut t.situation_b_index);
    }
    for tn in &mut enrichment.temporal_normalizations {
        convert(&mut tn.situation_index);
    }
}

/// Warning produced during extraction validation.
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
}

/// Parse an LLM response (possibly wrapped in markdown) into a NarrativeExtraction.
pub fn parse_llm_response(raw: &str) -> Result<NarrativeExtraction> {
    let json_str = extract_json_from_response(raw);

    // Try parsing as-is first
    if let Ok(ext) = serde_json::from_str::<NarrativeExtraction>(&json_str) {
        return Ok(ext);
    }

    // Stage 1: Normalize common LLM JSON mistakes before retrying
    let normalized = normalize_llm_json(&json_str);
    if let Ok(ext) = serde_json::from_str::<NarrativeExtraction>(&normalized) {
        return Ok(ext);
    }

    // Stage 2: Fix truncated JSON by closing open brackets/braces and unclosed strings
    let fixed = fix_truncated_json(&normalized);
    if let Ok(ext) = serde_json::from_str::<NarrativeExtraction>(&fixed) {
        return Ok(ext);
    }

    // Stage 3: Salvage via serde_json::Value — parse what we can, fill missing arrays
    if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&fixed) {
        // Ensure required array fields exist (may be truncated away)
        if let Some(obj) = val.as_object_mut() {
            for key in &[
                "entities",
                "situations",
                "participations",
                "causal_links",
                "temporal_relations",
            ] {
                if !obj.contains_key(*key) {
                    obj.insert(key.to_string(), serde_json::Value::Array(vec![]));
                }
            }
        }
        if let Ok(ext) = serde_json::from_value::<NarrativeExtraction>(val) {
            tracing::warn!("Salvaged partial extraction from truncated JSON");
            return Ok(ext);
        }
    }

    serde_json::from_str(&fixed)
        .map_err(|e| TensaError::ExtractionError(format!("Failed to parse extraction JSON: {}", e)))
}

/// Return the parse error message without failing — used for retry prompts.
pub fn parse_llm_response_error(raw: &str) -> Option<String> {
    match parse_llm_response(raw) {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    }
}

/// Normalize common LLM JSON mistakes:
/// - String values where arrays are expected (e.g. `"aliases": "Bob"` → `"aliases": ["Bob"]`)
/// - Trailing commas before closing brackets
/// - Single quotes → double quotes (outside existing double-quoted strings)
fn normalize_llm_json(json: &str) -> String {
    // Parse as generic Value first to apply structural fixes
    let mut s = json.to_string();

    // Fix trailing commas: ,] or ,}
    // Simple approach: regex-like manual replacement
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b',' {
            // Look ahead past whitespace for ] or }
            let mut j = i + 1;
            while j < bytes.len()
                && (bytes[j] == b' ' || bytes[j] == b'\n' || bytes[j] == b'\r' || bytes[j] == b'\t')
            {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b']' || bytes[j] == b'}') {
                // Skip the trailing comma
                i += 1;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    s = String::from_utf8(out).unwrap_or(s);

    // Try to parse as serde_json::Value and fix string-where-array-expected
    if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&s) {
        fix_string_to_array(&mut val, &["aliases", "content_blocks"]);
        // Also fix top-level fields that should be arrays
        fix_field_to_array(
            &mut val,
            &[
                "entities",
                "situations",
                "participations",
                "causal_links",
                "temporal_relations",
            ],
        );
        if let Ok(fixed) = serde_json::to_string(&val) {
            return fixed;
        }
    }

    s
}

/// If a field that should be an array is a string, wrap it in an array.
fn fix_string_to_array(val: &mut serde_json::Value, field_names: &[&str]) {
    match val {
        serde_json::Value::Object(map) => {
            for name in field_names {
                if let Some(v) = map.get_mut(*name) {
                    if v.is_string() {
                        *v = serde_json::Value::Array(vec![v.clone()]);
                    }
                }
            }
            // Recurse into all values
            for v in map.values_mut() {
                fix_string_to_array(v, field_names);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                fix_string_to_array(v, field_names);
            }
        }
        _ => {}
    }
}

/// If a top-level field that should be an array is missing or not an array, fix it.
fn fix_field_to_array(val: &mut serde_json::Value, field_names: &[&str]) {
    if let serde_json::Value::Object(map) = val {
        for name in field_names {
            match map.get(*name) {
                None => {
                    map.insert(name.to_string(), serde_json::Value::Array(vec![]));
                }
                Some(v) if !v.is_array() => {
                    // Wrap non-array in array (e.g. single object instead of array of objects)
                    let v = v.clone();
                    map.insert(name.to_string(), serde_json::Value::Array(vec![v]));
                }
                _ => {}
            }
        }
    }
}

/// Fix truncated JSON by closing open brackets/braces.
fn fix_truncated_json(json: &str) -> String {
    let mut fixed = json.trim_end().to_string();
    if fixed.ends_with(',') {
        fixed.pop();
    }

    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;
    for ch in fixed.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    // Close unclosed string literal (truncated mid-value)
    if in_string {
        fixed.push('"');
    }
    // Remove trailing incomplete key-value (e.g. `"key": "val` → we just closed the string,
    // but the parent object/array may still need a comma removed)
    let trimmed = fixed.trim_end();
    if trimmed.ends_with(',') {
        fixed = trimmed.trim_end_matches(',').to_string();
    }
    while let Some(closer) = stack.pop() {
        fixed.push(closer);
    }
    fixed
}

/// Strip `<think>...</think>` blocks emitted by reasoning models (Qwen3, DeepSeek, etc.).
/// Single-pass, single-allocation. Handles multiple blocks and unclosed trailing tags.
pub(crate) fn strip_thinking_tags(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    loop {
        match rest.find("<think>") {
            None => {
                out.push_str(rest);
                break;
            }
            Some(start) => {
                out.push_str(&rest[..start]);
                rest = &rest[start + "<think>".len()..];
                match rest.find("</think>") {
                    Some(end) => rest = &rest[end + "</think>".len()..],
                    None => break, // unclosed — discard remainder
                }
            }
        }
    }
    out
}

/// Extract JSON from an LLM response that may be wrapped in markdown code blocks
/// or prefixed with thinking tags (e.g. Qwen3's `<think>...</think>`).
pub fn extract_json_from_response(raw: &str) -> String {
    // Strip <think>...</think> blocks (Qwen3, DeepSeek, etc.)
    let cleaned = strip_thinking_tags(raw);
    let trimmed = cleaned.trim();

    // Try to extract from ```json ... ``` blocks
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }

    // Try to extract from ``` ... ``` blocks
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }

    // Try to find JSON object directly
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_string();
        }
    }

    trimmed.to_string()
}

/// Repair common LLM extraction errors in-place, returning warnings for what was fixed.
///
/// Fixes applied:
/// 1. **1-based → 0-based index conversion** — the prompt tells the LLM to use 1-based
///    indices (natural for LLMs). This pass converts them to 0-based for internal use.
///    Index 0 is treated as already-converted or an error and left as-is.
/// 2. **Empty entity names** — participations with blank entity_name are dropped.
/// 3. **Out-of-range causal links / temporal relations** — entries with uncorrectable
///    indices are removed.
///
/// Note: orphan entity names (participations referencing entities not in the array) are
/// handled by the pipeline's `process_extraction`, which auto-creates missing entities
/// through the normal gate + resolver path.
pub fn repair_extraction(extraction: &mut NarrativeExtraction) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    let num_situations = extraction.situations.len();

    // --- Pass 1: Convert 1-based indices to 0-based ---
    // The prompt tells the LLM to use 1-based situation numbering (1..N) which is natural
    // for language models. We convert to 0-based here. An index of 0 from the LLM is
    // ambiguous (could be 0-based holdover or an error) — we leave it as-is since
    // situation 0 is valid either way.
    {
        let convert = |idx: &mut usize| {
            if *idx >= 1 {
                *idx -= 1;
            }
        };
        for p in &mut extraction.participations {
            convert(&mut p.situation_index);
        }
        for c in &mut extraction.causal_links {
            convert(&mut c.from_situation_index);
            convert(&mut c.to_situation_index);
        }
        for t in &mut extraction.temporal_relations {
            convert(&mut t.situation_a_index);
            convert(&mut t.situation_b_index);
        }
    }

    // --- Pass 3: Remove participations with empty entity_name ---
    let before_len = extraction.participations.len();
    extraction
        .participations
        .retain(|p| !p.entity_name.trim().is_empty());
    let removed = before_len - extraction.participations.len();
    if removed > 0 {
        warnings.push(ValidationWarning {
            message: format!(
                "Removed {} participation(s) with empty entity_name",
                removed
            ),
        });
    }

    // --- Pass 4: Remove causal links with still-out-of-range indices ---
    if num_situations > 0 {
        let before_len = extraction.causal_links.len();
        extraction.causal_links.retain(|c| {
            c.from_situation_index < num_situations && c.to_situation_index < num_situations
        });
        let removed = before_len - extraction.causal_links.len();
        if removed > 0 {
            warnings.push(ValidationWarning {
                message: format!(
                    "Removed {} causal link(s) with out-of-range indices",
                    removed
                ),
            });
        }
    }

    // --- Pass 5: Remove temporal relations with still-out-of-range indices ---
    if num_situations > 0 {
        let before_len = extraction.temporal_relations.len();
        extraction.temporal_relations.retain(|t| {
            t.situation_a_index < num_situations && t.situation_b_index < num_situations
        });
        let removed = before_len - extraction.temporal_relations.len();
        if removed > 0 {
            warnings.push(ValidationWarning {
                message: format!(
                    "Removed {} temporal relation(s) with out-of-range indices",
                    removed
                ),
            });
        }
    }

    // --- Pass 6: Remove participations with still-out-of-range situation indices ---
    if num_situations > 0 {
        let before_len = extraction.participations.len();
        extraction
            .participations
            .retain(|p| p.situation_index < num_situations);
        let removed = before_len - extraction.participations.len();
        if removed > 0 {
            warnings.push(ValidationWarning {
                message: format!(
                    "Removed {} participation(s) with out-of-range situation_index",
                    removed
                ),
            });
        }
    }

    warnings
}

/// Guess entity type from the participation role.
/// Locations typically appear with role "Setting" or Custom("Setting").
pub fn guess_entity_type_from_role(role: &Role) -> EntityType {
    match role {
        Role::Custom(s) if s.eq_ignore_ascii_case("Setting") => EntityType::Location,
        Role::Instrument => EntityType::Artifact,
        Role::Custom(s) if s.eq_ignore_ascii_case("Location") => EntityType::Location,
        _ => EntityType::Location, // default: most orphan names are locations
    }
}

/// Validate extraction for internal consistency.
pub fn validate_extraction(extraction: &NarrativeExtraction) -> Vec<ValidationWarning> {
    let mut warnings = Vec::new();
    let num_situations = extraction.situations.len();

    // Check entity names are non-empty
    for (i, e) in extraction.entities.iter().enumerate() {
        if e.name.trim().is_empty() {
            warnings.push(ValidationWarning {
                message: format!("Entity {} has empty name", i),
            });
        }
        if !(0.0..=1.0).contains(&e.confidence) {
            warnings.push(ValidationWarning {
                message: format!("Entity {} confidence {} out of [0,1]", i, e.confidence),
            });
        }
    }

    // Check situation descriptions
    for (i, s) in extraction.situations.iter().enumerate() {
        if s.description.trim().is_empty() {
            warnings.push(ValidationWarning {
                message: format!("Situation {} has empty description", i),
            });
        }
        if !(0.0..=1.0).contains(&s.confidence) {
            warnings.push(ValidationWarning {
                message: format!("Situation {} confidence {} out of [0,1]", i, s.confidence),
            });
        }
    }

    // Check participation indices
    for (i, p) in extraction.participations.iter().enumerate() {
        if p.situation_index >= num_situations {
            warnings.push(ValidationWarning {
                message: format!(
                    "Participation {} situation_index {} out of range (max {})",
                    i,
                    p.situation_index,
                    num_situations.saturating_sub(1)
                ),
            });
        }
        // entity_name should match an extracted entity (case-insensitive)
        let pname = p.entity_name.to_lowercase();
        if !extraction.entities.iter().any(|e| {
            e.name.to_lowercase() == pname || e.aliases.iter().any(|a| a.to_lowercase() == pname)
        }) {
            warnings.push(ValidationWarning {
                message: format!(
                    "Participation {} entity_name '{}' not found in extracted entities",
                    i, p.entity_name
                ),
            });
        }
    }

    // Check causal link indices
    for (i, c) in extraction.causal_links.iter().enumerate() {
        if c.from_situation_index >= num_situations {
            warnings.push(ValidationWarning {
                message: format!(
                    "CausalLink {} from_situation_index {} out of range",
                    i, c.from_situation_index
                ),
            });
        }
        if c.to_situation_index >= num_situations {
            warnings.push(ValidationWarning {
                message: format!(
                    "CausalLink {} to_situation_index {} out of range",
                    i, c.to_situation_index
                ),
            });
        }
    }

    // Check temporal relation indices
    for (i, t) in extraction.temporal_relations.iter().enumerate() {
        if t.situation_a_index >= num_situations || t.situation_b_index >= num_situations {
            warnings.push(ValidationWarning {
                message: format!("TemporalRelation {} has out-of-range situation index", i),
            });
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_canon_array_with_extra_fields() {
        // Real-world shape returned by grok-4.1-fast: array with extra fields
        // (`type`, `geo_display_name`, `latitude`…) beyond our schema. Should
        // parse cleanly because serde ignores unknown fields by default — and
        // the array extractor must handle the leading `[` (the legacy
        // `extract_json_from_response` dropped the outer brackets).
        let raw = r#"[
          {"uuid":"abc1","raw_name":"Yanina","canonical_name":"Janina",
           "type":"Location","geo_display_name":"Ioannina, Greece",
           "latitude":39.66,"longitude":20.85},
          {"uuid":"abc2","raw_name":"Catalans village","canonical_name":"Village des Catalans",
           "country_code":"fr","admin_region":"Provence-Alpes-Côte d'Azur"}
        ]"#;
        let rows = parse_canonicalize_places_response(raw).expect("must parse");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].uuid, "abc1");
        assert_eq!(rows[0].canonical_name, "Janina");
        assert_eq!(rows[0].country_code, None);
        assert_eq!(rows[1].country_code.as_deref(), Some("fr"));
    }

    #[test]
    fn test_parse_canon_array_inside_markdown_fence() {
        let raw = "Sure, here you go:\n\n```json\n[\n  {\"uuid\":\"x\",\"raw_name\":\"Marseilles\",\"canonical_name\":\"Marseille\",\"country_code\":\"fr\"}\n]\n```\n";
        let rows = parse_canonicalize_places_response(raw).expect("must parse from fence");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].canonical_name, "Marseille");
        assert_eq!(rows[0].country_code.as_deref(), Some("fr"));
    }

    #[test]
    fn test_parse_canon_empty_array() {
        let rows = parse_canonicalize_places_response("[]").expect("must parse");
        assert_eq!(rows.len(), 0);
    }

    fn sample_extraction_json() -> &'static str {
        r#"{
            "entities": [
                {
                    "name": "Raskolnikov",
                    "aliases": ["Rodion", "Rodya"],
                    "entity_type": "Actor",
                    "properties": {"age": 23},
                    "confidence": 0.95
                },
                {
                    "name": "Sonya",
                    "aliases": [],
                    "entity_type": "Actor",
                    "properties": {},
                    "confidence": 0.9
                }
            ],
            "situations": [
                {
                    "description": "Raskolnikov confesses to Sonya",
                    "temporal_marker": "evening",
                    "location": "Sonya's room",
                    "narrative_level": "Scene",
                    "content_blocks": [],
                    "confidence": 0.85
                }
            ],
            "participations": [
                {
                    "entity_name": "Raskolnikov",
                    "situation_index": 0,
                    "role": "Protagonist",
                    "action": "confesses",
                    "confidence": 0.9
                },
                {
                    "entity_name": "Sonya",
                    "situation_index": 0,
                    "role": "Witness",
                    "action": "listens",
                    "confidence": 0.88
                }
            ],
            "causal_links": [],
            "temporal_relations": []
        }"#
    }

    #[test]
    fn test_parse_valid_extraction_json() {
        let result = parse_llm_response(sample_extraction_json());
        assert!(result.is_ok());
        let ext = result.unwrap();
        assert_eq!(ext.entities.len(), 2);
        assert_eq!(ext.situations.len(), 1);
        assert_eq!(ext.participations.len(), 2);
        assert_eq!(ext.entities[0].name, "Raskolnikov");
        assert_eq!(ext.entities[0].aliases, vec!["Rodion", "Rodya"]);
    }

    #[test]
    fn test_parse_extraction_from_markdown_codeblock() {
        let wrapped = format!(
            "Here's the extraction:\n```json\n{}\n```\nDone.",
            sample_extraction_json()
        );
        let result = parse_llm_response(&wrapped);
        assert!(result.is_ok());
        let ext = result.unwrap();
        assert_eq!(ext.entities.len(), 2);
    }

    #[test]
    fn test_parse_extraction_from_bare_codeblock() {
        let wrapped = format!("```\n{}\n```", sample_extraction_json());
        let result = parse_llm_response(&wrapped);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_extraction_with_surrounding_text() {
        let wrapped = format!(
            "Some preamble text. {} And trailing text.",
            sample_extraction_json()
        );
        let result = parse_llm_response(&wrapped);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_invalid_json_error() {
        let result = parse_llm_response("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_fields_defaults() {
        let json = r#"{
            "entities": [{"name": "Test", "entity_type": "Actor"}],
            "situations": [],
            "participations": [],
            "causal_links": [],
            "temporal_relations": []
        }"#;
        let result = parse_llm_response(json).unwrap();
        assert_eq!(result.entities[0].confidence, 0.5); // default
        assert_eq!(result.entities[0].aliases.len(), 0); // default empty
    }

    #[test]
    fn test_validate_extraction_valid() {
        let ext = parse_llm_response(sample_extraction_json()).unwrap();
        let warnings = validate_extraction(&ext);
        assert!(
            warnings.is_empty(),
            "Got warnings: {:?}",
            warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_validate_extraction_out_of_range_index() {
        let mut ext = parse_llm_response(sample_extraction_json()).unwrap();
        ext.participations[0].situation_index = 99;
        let warnings = validate_extraction(&ext);
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("out of range")));
    }

    #[test]
    fn test_validate_extraction_invalid_confidence() {
        let mut ext = parse_llm_response(sample_extraction_json()).unwrap();
        ext.entities[0].confidence = 1.5;
        let warnings = validate_extraction(&ext);
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("confidence")));
    }

    #[test]
    fn test_validate_extraction_unknown_entity_name() {
        let mut ext = parse_llm_response(sample_extraction_json()).unwrap();
        ext.participations[0].entity_name = "NonExistent".to_string();
        let warnings = validate_extraction(&ext);
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("not found")));
    }

    #[test]
    fn test_parse_with_thinking_tags() {
        let wrapped = format!(
            "<think>\nLet me analyze this text carefully...\nI see two characters.\n</think>\n{}",
            sample_extraction_json()
        );
        let result = parse_llm_response(&wrapped);
        assert!(result.is_ok());
        let ext = result.unwrap();
        assert_eq!(ext.entities.len(), 2);
    }

    #[test]
    fn test_parse_with_thinking_tags_and_codeblock() {
        let wrapped = format!(
            "<think>\nAnalyzing the passage...\n</think>\n```json\n{}\n```",
            sample_extraction_json()
        );
        let result = parse_llm_response(&wrapped);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_unclosed_thinking_tag() {
        // When thinking fills the entire token budget, closing tag may be missing
        let wrapped = format!(
            "{}\n<think>\nThis trailing think should be stripped",
            sample_extraction_json()
        );
        let result = parse_llm_response(&wrapped);
        assert!(result.is_ok());
    }

    #[test]
    fn test_strip_thinking_only() {
        let result = parse_llm_response("<think>\nJust thinking, no JSON\n</think>");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_causal_link_out_of_range() {
        let mut ext = parse_llm_response(sample_extraction_json()).unwrap();
        ext.causal_links.push(ExtractedCausalLink {
            from_situation_index: 0,
            to_situation_index: 50,
            mechanism: None,
            causal_type: CausalType::Contributing,
            strength: 0.5,
            confidence: 0.8,
        });
        let warnings = validate_extraction(&ext);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_parse_string_as_array() {
        // LLM returns "aliases": "Old Lady" instead of ["Old Lady"]
        let json = r#"{
            "entities": [{"name": "Test", "aliases": "Old Lady", "entity_type": "Actor"}],
            "situations": [],
            "participations": []
        }"#;
        let result = parse_llm_response(json).unwrap();
        assert_eq!(result.entities[0].aliases, vec!["Old Lady"]);
    }

    #[test]
    fn test_parse_missing_optional_arrays() {
        // LLM omits causal_links and temporal_relations entirely
        let json = r#"{
            "entities": [{"name": "Bob", "entity_type": "Actor"}],
            "situations": [{"description": "Something happened"}]
        }"#;
        let result = parse_llm_response(json).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.situations.len(), 1);
        assert!(result.causal_links.is_empty());
        assert!(result.temporal_relations.is_empty());
        assert!(result.participations.is_empty());
    }

    #[test]
    fn test_parse_trailing_comma() {
        let json = r#"{
            "entities": [{"name": "Test", "entity_type": "Actor",}],
            "situations": [],
        }"#;
        let result = parse_llm_response(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_unknown_entity_type() {
        let json = r#"{
            "entities": [{"name": "London", "entity_type": "Place"}],
            "situations": []
        }"#;
        let result = parse_llm_response(json).unwrap();
        assert_eq!(result.entities[0].entity_type, EntityType::Location);
    }

    #[test]
    fn test_repair_leaves_orphan_entities_for_pipeline() {
        // repair_extraction no longer auto-creates missing entities —
        // that's handled by the pipeline's process_extraction (upsert).
        // Repair only converts indices and cleans up empty names.
        let mut ext = NarrativeExtraction {
            entities: vec![ExtractedEntity {
                name: "Alice".into(),
                aliases: vec![],
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.9,
            }],
            situations: vec![ExtractedSituation {
                name: Some("Test".into()),
                description: "Something happens".into(),
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Scene,
                content_blocks: vec![],
                confidence: 0.8,
                text_start: None,
                text_end: None,
            }],
            participations: vec![
                ExtractedParticipation {
                    entity_name: "Alice".into(),
                    situation_index: 1, // 1-based → 0
                    role: Role::Protagonist,
                    action: None,
                    confidence: 0.9,
                },
                ExtractedParticipation {
                    entity_name: "Palace".into(),
                    situation_index: 1, // 1-based → 0
                    role: Role::Custom("Setting".into()),
                    action: None,
                    confidence: 0.7,
                },
            ],
            causal_links: vec![],
            temporal_relations: vec![],
        };
        repair_extraction(&mut ext);
        // entities array unchanged — pipeline handles orphan creation
        assert_eq!(ext.entities.len(), 1);
        // indices converted: 1 → 0
        assert_eq!(ext.participations[0].situation_index, 0);
        assert_eq!(ext.participations[1].situation_index, 0);
        // "Palace" participation is kept — pipeline will create it
        assert_eq!(ext.participations.len(), 2);
    }

    #[test]
    fn test_repair_converts_1based_to_0based() {
        // LLM outputs 1-based: situation_index=2 means the 2nd situation (0-based index 1)
        let mut ext = NarrativeExtraction {
            entities: vec![ExtractedEntity {
                name: "Bob".into(),
                aliases: vec![],
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.9,
            }],
            situations: vec![
                ExtractedSituation {
                    name: None,
                    description: "Sit 1".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Event,
                    content_blocks: vec![],
                    confidence: 0.8,
                    text_start: None,
                    text_end: None,
                },
                ExtractedSituation {
                    name: None,
                    description: "Sit 2".into(),
                    temporal_marker: None,
                    location: None,
                    narrative_level: NarrativeLevel::Event,
                    content_blocks: vec![],
                    confidence: 0.8,
                    text_start: None,
                    text_end: None,
                },
            ],
            participations: vec![
                ExtractedParticipation {
                    entity_name: "Bob".into(),
                    situation_index: 1, // 1-based → becomes 0
                    role: Role::Protagonist,
                    action: None,
                    confidence: 0.9,
                },
                ExtractedParticipation {
                    entity_name: "Bob".into(),
                    situation_index: 2, // 1-based → becomes 1
                    role: Role::Witness,
                    action: None,
                    confidence: 0.9,
                },
            ],
            causal_links: vec![ExtractedCausalLink {
                from_situation_index: 1,
                to_situation_index: 2,
                mechanism: None,
                causal_type: CausalType::Contributing,
                strength: 0.5,
                confidence: 0.8,
            }],
            temporal_relations: vec![],
        };
        repair_extraction(&mut ext);
        assert_eq!(ext.participations[0].situation_index, 0); // 1 → 0
        assert_eq!(ext.participations[1].situation_index, 1); // 2 → 1
        assert_eq!(ext.causal_links[0].from_situation_index, 0);
        assert_eq!(ext.causal_links[0].to_situation_index, 1);
        // Both should be valid after conversion
        let val_warnings = validate_extraction(&ext);
        assert!(!val_warnings
            .iter()
            .any(|w| w.message.contains("out of range")));
    }

    #[test]
    fn test_repair_removes_empty_entity_name() {
        let mut ext = NarrativeExtraction {
            entities: vec![],
            situations: vec![ExtractedSituation {
                name: None,
                description: "Something".into(),
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Event,
                content_blocks: vec![],
                confidence: 0.8,
                text_start: None,
                text_end: None,
            }],
            participations: vec![ExtractedParticipation {
                entity_name: "".into(),
                situation_index: 0,
                role: Role::Witness,
                action: None,
                confidence: 0.5,
            }],
            causal_links: vec![],
            temporal_relations: vec![],
        };
        let warnings = repair_extraction(&mut ext);
        assert!(ext.participations.is_empty());
        assert!(warnings
            .iter()
            .any(|w| w.message.contains("empty entity_name")));
    }

    #[test]
    fn test_repair_removes_far_out_of_range_indices() {
        // 1 situation: valid 1-based index is 1 only (→ 0-based: 0)
        let mut ext = NarrativeExtraction {
            entities: vec![ExtractedEntity {
                name: "X".into(),
                aliases: vec![],
                entity_type: EntityType::Actor,
                properties: serde_json::json!({}),
                confidence: 0.9,
            }],
            situations: vec![ExtractedSituation {
                name: None,
                description: "Only situation".into(),
                temporal_marker: None,
                location: None,
                narrative_level: NarrativeLevel::Event,
                content_blocks: vec![],
                confidence: 0.8,
                text_start: None,
                text_end: None,
            }],
            participations: vec![ExtractedParticipation {
                entity_name: "X".into(),
                situation_index: 5, // 1-based 5 → 0-based 4, still out of range
                role: Role::Protagonist,
                action: None,
                confidence: 0.9,
            }],
            causal_links: vec![ExtractedCausalLink {
                from_situation_index: 1, // valid (→ 0)
                to_situation_index: 10,  // 1-based 10 → 0-based 9, out of range
                mechanism: None,
                causal_type: CausalType::Contributing,
                strength: 0.5,
                confidence: 0.7,
            }],
            temporal_relations: vec![],
        };
        let warnings = repair_extraction(&mut ext);
        // participation 0-based 4 still out of range → removed
        assert!(ext.participations.is_empty());
        // causal link to_index 9 out of range → removed
        assert!(ext.causal_links.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("out-of-range")));
    }

    #[test]
    fn test_parse_session_reconciliation() {
        let json = r#"{
            "entity_merges": [
                {"canonical_name": "Hari Seldon", "duplicate_names": ["Seldon", "Dr. Seldon"]}
            ],
            "timeline": [
                {"situation": "Trial of Seldon", "date": "12069-01-15", "confidence": 0.9}
            ],
            "confidence_adjustments": [
                {"name": "Gaal Dornick", "adjusted_confidence": 0.95, "reason": "Appears in 5 chunks"}
            ],
            "cross_chunk_causal_links": []
        }"#;
        let result = parse_session_reconciliation_response(json).unwrap();
        assert_eq!(result.entity_merges.len(), 1);
        assert_eq!(result.entity_merges[0].canonical_name, "Hari Seldon");
        assert_eq!(result.timeline.len(), 1);
        assert_eq!(result.confidence_adjustments.len(), 1);
        assert_eq!(result.confidence_adjustments[0].adjusted_confidence, 0.95);
    }

    #[test]
    fn test_parse_session_reconciliation_empty() {
        let json = r#"{}"#;
        let result = parse_session_reconciliation_response(json).unwrap();
        assert!(result.entity_merges.is_empty());
        assert!(result.timeline.is_empty());
    }
}
