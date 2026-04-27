//! Generation Engine — Sprint W1 (v0.49.0).
//!
//! AI-assisted scaffolding for writers: outline → chapters → scenes,
//! characters, places, arcs. Every generation produces a [`GenerationProposal`]
//! — a plain data value the writer can inspect before writing anything to the
//! store. Applying a proposal is a commit via the revision system, so undo is
//! free and diff/rollback come for free too.
//!
//! # Invariants
//! - **No destructive AI.** `generate` only returns proposals.
//! - **Single LLM primitive.** All prompts go through `send_session_messages`
//!   on the caller's `SessionCapableExtractor`.
//! - **Dry-run honored.** [`build_prompt`] + [`estimate_tokens`] let the caller
//!   preview cost without an LLM call.
//! - **Context is the snapshot.** Prompts read `NarrativeSnapshot` (which
//!   includes the plan from v0.48.2) — no bespoke context loaders.

use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::llm::{ApiMessage, SessionCapableExtractor};
use crate::ingestion::llm_cache::LlmCache;
use crate::narrative::causal_helpers;
use crate::narrative::cost_ledger;
use crate::narrative::plan as plan_store;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::revision::{commit_narrative, gather_snapshot, CommitOutcome};
use crate::narrative::writer_common::{approx_tokens, truncate_utf8, write_plan_section};
use crate::types::*;

// ─── Public request / response types ──────────────────────────────

/// What the writer wants generated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GenerationRequest {
    /// Top-level chapter outline for the whole narrative.
    Outline {
        /// Optional premise override. If omitted, uses the narrative's plan.
        #[serde(default)]
        premise: Option<String>,
        /// Number of chapters to produce.
        num_chapters: u32,
        /// Optional tone hint ("grim", "comic", ...). Falls back to plan.style.tone.
        #[serde(default)]
        tone_hint: Option<String>,
        /// POV strategy for the whole outline. When supplied, each generated
        /// chapter receives a `pov_entity_name` (or is explicitly omniscient).
        /// None leaves POV unassigned — downstream tools can fill it later.
        #[serde(default)]
        pov_hint: Option<PovHint>,
    },
    /// A single character profile.
    Character {
        /// Seed — at least one of name / role / one-liner should be set.
        #[serde(default)]
        seed: CharacterSeed,
    },
    /// Scene sketches inside an existing chapter situation.
    Scenes {
        chapter_situation_id: Uuid,
        count: u32,
        #[serde(default)]
        constraints: SceneConstraints,
        /// POV character for the whole chapter. If omitted the scene generator
        /// reads the chapter's existing `discourse.focalization` from the
        /// snapshot and carries it forward. If both are absent, scenes inherit
        /// no POV and the writer sets it later.
        #[serde(default)]
        pov_entity_name: Option<String>,
    },
}

/// Narrative POV strategy for an outline-level generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PovHint {
    /// Single-narrator novel — every chapter is told from this character's POV.
    Single { entity_name: String },
    /// Multi-POV rotation (e.g. Dracula, A Song of Ice and Fire). The LLM is
    /// asked to distribute chapters across the listed narrators in the order
    /// that best serves the premise.
    Rotating { entity_names: Vec<String> },
    /// Third-person omniscient / heterodiegetic voice throughout.
    Omniscient,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharacterSeed {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub one_liner: Option<String>,
    /// Preferred entity type. Defaults to Actor.
    #[serde(default)]
    pub entity_type: Option<EntityType>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneConstraints {
    /// Guidance the LLM should honour ("end on a revelation", "keep it under 800 words of prose").
    #[serde(default)]
    pub instruction: Option<String>,
    /// Entity ids to include as participants.
    #[serde(default)]
    pub must_include_entities: Vec<Uuid>,
}

/// Draft snapshot delta the writer can accept, edit, or discard.
/// Carries just enough to reconstruct what will be written on apply.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationProposal {
    pub narrative_id: String,
    /// Human-readable name of what was generated ("15-chapter outline", "character: Alice").
    pub kind: String,
    /// Chapters / situations to create.
    pub situations: Vec<ProposedSituation>,
    /// Entities to create.
    pub entities: Vec<ProposedEntity>,
    /// Participations linking proposed situations to proposed or existing entities.
    pub participations: Vec<ProposedParticipation>,
    /// Commit message to use on apply. Defaults to `"AI-generated: {kind}"`.
    #[serde(default)]
    pub commit_message: Option<String>,
    /// Raw LLM rationale (for UI).
    #[serde(default)]
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedSituation {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub narrative_level: NarrativeLevel,
    /// Days from "now" at which this situation starts. Keeps the LLM from
    /// hallucinating specific dates.
    pub temporal_day_offset: f64,
    /// Optional duration in days (start+duration → end).
    #[serde(default)]
    pub temporal_duration_days: Option<f64>,
    #[serde(default)]
    pub raw_content_hint: Option<String>,
    /// Name of the narrating/focalizing Actor entity. Resolved to a UUID at
    /// apply time and stored on `Situation.discourse.focalization`. None means
    /// no explicit POV (omniscient or unspecified).
    #[serde(default)]
    pub pov_entity_name: Option<String>,
    /// "homodiegetic" (narrator is a character in the story) or
    /// "heterodiegetic" (narrator is outside it). Stored on
    /// `Situation.discourse.voice`. When `pov_entity_name` is set, voice
    /// defaults to "homodiegetic"; when omniscient, defaults to "heterodiegetic".
    #[serde(default)]
    pub voice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedEntity {
    pub entity_type: EntityType,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedParticipation {
    /// Either an existing entity id, or an index into `entities` (prefix "new:N").
    pub entity_ref: String,
    /// Index into `situations` (prefix "new:N") or an existing situation id.
    pub situation_ref: String,
    pub role: Role,
    #[serde(default)]
    pub action: Option<String>,
}

/// Rough cost estimate — token counts only. Currency conversion is the
/// caller's job (it depends on the active model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationEstimate {
    pub prompt_tokens: u32,
    pub expected_response_tokens: u32,
}

/// Assembled prompt for dry-run display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPrompt {
    pub system: String,
    pub user: String,
    pub estimate: GenerationEstimate,
}

// ─── Prompt assembly ──────────────────────────────────────────────

/// Build the system + user prompt for a request, given the current snapshot.
/// Pure function — safe to call at dry-run time.
pub fn build_prompt(request: &GenerationRequest, snapshot: &NarrativeSnapshot) -> GenerationPrompt {
    let (system, user, expected_response_tokens) = match request {
        GenerationRequest::Outline {
            premise,
            num_chapters,
            tone_hint,
            pov_hint,
        } => {
            let system = SYSTEM_OUTLINE.to_string();
            let user = build_outline_user_prompt(
                snapshot,
                premise.as_deref(),
                *num_chapters,
                tone_hint.as_deref(),
                pov_hint.as_ref(),
            );
            let expected = (num_chapters * 80).clamp(200, 4000);
            (system, user, expected)
        }
        GenerationRequest::Character { seed } => {
            let system = SYSTEM_CHARACTER.to_string();
            let user = build_character_user_prompt(snapshot, seed);
            (system, user, 600)
        }
        GenerationRequest::Scenes {
            chapter_situation_id,
            count,
            constraints,
            pov_entity_name,
        } => {
            let system = SYSTEM_SCENES.to_string();
            let user = build_scenes_user_prompt(
                snapshot,
                chapter_situation_id,
                *count,
                constraints,
                pov_entity_name.as_deref(),
            );
            let expected = (count * 120).clamp(200, 3000);
            (system, user, expected)
        }
    };
    let prompt_tokens = approx_tokens(&system) + approx_tokens(&user);
    GenerationPrompt {
        system,
        user,
        estimate: GenerationEstimate {
            prompt_tokens,
            expected_response_tokens,
        },
    }
}

/// Cheap cost estimator — just tokens. No LLM call.
pub fn estimate_tokens(
    request: &GenerationRequest,
    snapshot: &NarrativeSnapshot,
) -> GenerationEstimate {
    build_prompt(request, snapshot).estimate
}

const SYSTEM_OUTLINE: &str = "You are a story outliner. You produce chapter-by-chapter outlines for novels. \
Return ONLY a JSON object with the shape: \
{\"chapters\": [{\"name\": str, \"summary\": str, \"day_offset\": float, \"duration_days\": float|null, \"pov_character\": str|null, \"voice\": \"homodiegetic\"|\"heterodiegetic\"|null}], \"rationale\": str}. \
Each chapter's day_offset is days from the narrative start (0 for the first chapter), forming a strictly increasing sequence. \
Names are short (under 60 chars). Summaries are 1–3 sentences. No prose drafts — just structural outline. \
POV RULES: if the user specifies a single narrator, set pov_character to that name on every chapter and voice=\"homodiegetic\". \
If the user supplies multiple narrators for rotation, distribute them across chapters in whatever order best serves the premise (a typical pattern alternates, but you may cluster if the plot demands it), set pov_character to the narrator's exact name for each chapter, voice=\"homodiegetic\". \
If the user requests omniscient voice, set pov_character=null and voice=\"heterodiegetic\". \
If no POV guidance is provided, set pov_character=null and voice=null.";

const SYSTEM_CHARACTER: &str = "You are a character designer. You produce a single character profile consistent with the narrative's premise, style, and existing cast. \
Return ONLY a JSON object with the shape: \
{\"name\": str, \"entity_type\": \"Actor\"|\"Organization\", \"description\": str, \"role_in_story\": str, \"motivations\": [str], \"voice_notes\": str, \"quirks\": [str], \"rationale\": str}. \
description is 1–3 sentences. role_in_story is 3–8 words.";

const SYSTEM_SCENES: &str = "You are a scene sketcher. For a given chapter you produce scene sketches that advance the chapter's goal. \
Return ONLY a JSON object with the shape: \
{\"scenes\": [{\"name\": str, \"description\": str, \"day_offset\": float, \"duration_days\": float|null, \"pov_character\": str|null, \"voice\": \"homodiegetic\"|\"heterodiegetic\"|null, \"participants\": [{\"name\": str, \"role\": \"Protagonist\"|\"Antagonist\"|\"Witness\"|\"Target\"|\"Bystander\"|\"Confidant\"|\"Informant\"|\"Recipient\"|\"Instrument\"|\"SubjectOfDiscussion\", \"action\": str|null}]}], \"rationale\": str}. \
day_offset is relative to the chapter start (0 for the first scene). Names under 60 chars, descriptions 1–2 sentences. \
Participants should be named characters that exist in the narrative when possible, with a role and a short action phrase (\"confesses\", \"eavesdrops\"). \
POV RULES: if a chapter POV character is supplied, propagate that exact name to every scene's pov_character and set voice=\"homodiegetic\". \
If the chapter is explicitly omniscient, set pov_character=null and voice=\"heterodiegetic\". If the chapter has no POV guidance, set both to null — do not invent one.";

fn write_cast_context(out: &mut String, snapshot: &NarrativeSnapshot, cap: usize) {
    if snapshot.entities.is_empty() {
        return;
    }
    out.push_str("[existing cast]\n");
    for e in snapshot.entities.iter().take(cap) {
        let name = e
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let desc = e
            .properties
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        out.push_str(&format!("- {:?} «{}»: {}\n", e.entity_type, name, desc));
    }
    if snapshot.entities.len() > cap {
        out.push_str(&format!("… and {} more\n", snapshot.entities.len() - cap));
    }
    out.push('\n');
}

fn write_chapter_list(out: &mut String, snapshot: &NarrativeSnapshot, cap: usize) {
    let mut chapters: Vec<&Situation> = snapshot
        .situations
        .iter()
        .filter(|s| {
            matches!(
                s.narrative_level,
                NarrativeLevel::Arc | NarrativeLevel::Sequence | NarrativeLevel::Story
            )
        })
        .collect();
    chapters.sort_by_key(|s| s.temporal.start);
    if chapters.is_empty() {
        return;
    }
    out.push_str("[existing chapters]\n");
    for c in chapters.iter().take(cap) {
        let name = c.name.as_deref().unwrap_or("(untitled)");
        let summary = c.description.as_deref().unwrap_or("");
        out.push_str(&format!("- {} — {}\n", name, summary));
    }
    if chapters.len() > cap {
        out.push_str(&format!("… and {} more\n", chapters.len() - cap));
    }
    out.push('\n');
}

fn build_outline_user_prompt(
    snapshot: &NarrativeSnapshot,
    premise_override: Option<&str>,
    num_chapters: u32,
    tone_hint: Option<&str>,
    pov_hint: Option<&PovHint>,
) -> String {
    let mut out = String::new();
    write_plan_section(&mut out, snapshot, true);
    if let Some(p) = premise_override {
        out.push_str("[premise override]\n");
        out.push_str(p);
        out.push_str("\n\n");
    }
    if let Some(t) = tone_hint {
        out.push_str(&format!("[tone hint]\n{}\n\n", t));
    }
    if let Some(hint) = pov_hint {
        out.push_str("[POV strategy]\n");
        match hint {
            PovHint::Single { entity_name } => {
                out.push_str(&format!(
                    "Single-narrator novel. Every chapter is told from {}'s POV. \
                     Set pov_character=\"{}\" and voice=\"homodiegetic\" on all chapters.\n\n",
                    entity_name, entity_name,
                ));
            }
            PovHint::Rotating { entity_names } => {
                if entity_names.is_empty() {
                    out.push_str("Multi-POV novel. Narrators not listed yet — leave pov_character=null and voice=null.\n\n");
                } else {
                    out.push_str(&format!(
                        "Multi-POV rotation. Distribute chapters across these narrators: {}. \
                         Assign each chapter a single POV. Set pov_character to the narrator's exact \
                         name and voice=\"homodiegetic\". You may cluster (e.g. 2-3 chapters in one \
                         narrator's voice before switching) when the plot demands it.\n\n",
                        entity_names.join(", "),
                    ));
                }
            }
            PovHint::Omniscient => {
                out.push_str(
                    "Third-person omniscient voice throughout. Set pov_character=null and \
                     voice=\"heterodiegetic\" on every chapter.\n\n",
                );
            }
        }
    }
    write_cast_context(&mut out, snapshot, 20);
    write_chapter_list(&mut out, snapshot, 30);
    out.push_str(&format!("Generate a {}-chapter outline. Chapters should progress in time (strictly increasing day_offset).", num_chapters));
    out
}

fn build_character_user_prompt(snapshot: &NarrativeSnapshot, seed: &CharacterSeed) -> String {
    let mut out = String::new();
    write_plan_section(&mut out, snapshot, true);
    write_cast_context(&mut out, snapshot, 30);
    out.push_str("[seed]\n");
    if let Some(n) = &seed.name {
        out.push_str(&format!("Name: {}\n", n));
    }
    if let Some(r) = &seed.role {
        out.push_str(&format!("Role: {}\n", r));
    }
    if let Some(o) = &seed.one_liner {
        out.push_str(&format!("One-liner: {}\n", o));
    }
    if seed.name.is_none() && seed.role.is_none() && seed.one_liner.is_none() {
        out.push_str("(no seed — pick a character the story needs next)\n");
    }
    out.push('\n');
    out.push_str("Produce one character profile that fits the narrative.");
    out
}

fn build_scenes_user_prompt(
    snapshot: &NarrativeSnapshot,
    chapter_id: &Uuid,
    count: u32,
    constraints: &SceneConstraints,
    pov_override: Option<&str>,
) -> String {
    let mut out = String::new();
    write_plan_section(&mut out, snapshot, true);

    let chapter = snapshot.situations.iter().find(|s| s.id == *chapter_id);
    let inherited_pov = chapter.and_then(|c| c.discourse.as_ref()).and_then(|d| {
        d.focalization.and_then(|fid| {
            snapshot
                .entities
                .iter()
                .find(|e| e.id == fid)
                .and_then(|e| {
                    e.properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                })
        })
    });
    let pov_to_use = pov_override.map(str::to_string).or(inherited_pov);
    let chapter_voice = chapter
        .and_then(|c| c.discourse.as_ref())
        .and_then(|d| d.voice.clone());
    if let Some(c) = chapter {
        out.push_str("[chapter]\n");
        out.push_str(&format!(
            "Name: {}\n",
            c.name.as_deref().unwrap_or("(untitled)")
        ));
        if let Some(d) = &c.description {
            out.push_str(&format!("Summary: {}\n", d));
        }
        out.push('\n');
    } else {
        out.push_str(
            "[chapter]\n(id not found in snapshot — use the existing chapter list below)\n\n",
        );
    }

    if let Some(pov) = &pov_to_use {
        let voice = chapter_voice.as_deref().unwrap_or("homodiegetic");
        out.push_str(&format!(
            "[POV strategy]\nChapter POV: {}. Every scene is focalized through this character. \
             Set pov_character=\"{}\" and voice=\"{}\" on all scenes.\n\n",
            pov, pov, voice,
        ));
    } else if matches!(chapter_voice.as_deref(), Some("heterodiegetic")) {
        out.push_str(
            "[POV strategy]\nChapter is third-person omniscient. \
             Set pov_character=null and voice=\"heterodiegetic\" on all scenes.\n\n",
        );
    }

    write_cast_context(&mut out, snapshot, 20);
    write_chapter_list(&mut out, snapshot, 15);

    if let Some(i) = &constraints.instruction {
        out.push_str(&format!("[instruction]\n{}\n\n", i));
    }
    if !constraints.must_include_entities.is_empty() {
        let names: Vec<String> = constraints
            .must_include_entities
            .iter()
            .filter_map(|id| {
                snapshot.entities.iter().find(|e| e.id == *id).map(|e| {
                    e.properties
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string()
                })
            })
            .collect();
        out.push_str(&format!("[must include]\n{}\n\n", names.join(", ")));
    }
    out.push_str(&format!(
        "Produce {} scene sketches. Scenes progress in time relative to chapter start.",
        count
    ));
    out
}

// ─── LLM call + response parsing ──────────────────────────────────

/// Send the prompt to the extractor and parse the response into a proposal.
pub fn generate(
    extractor: &dyn SessionCapableExtractor,
    snapshot: &NarrativeSnapshot,
    narrative_id: &str,
    request: &GenerationRequest,
) -> Result<GenerationProposal> {
    let prompt = build_prompt(request, snapshot);
    let messages = vec![
        ApiMessage {
            role: "system".into(),
            content: prompt.system,
        },
        ApiMessage {
            role: "user".into(),
            content: prompt.user,
        },
    ];
    let response = extractor.send_session_messages(&messages)?;
    parse_response(request, narrative_id, &response)
}

fn parse_response(
    request: &GenerationRequest,
    narrative_id: &str,
    raw: &str,
) -> Result<GenerationProposal> {
    let json = extract_json_object(raw)?;
    match request {
        GenerationRequest::Outline { .. } => parse_outline(narrative_id, &json),
        GenerationRequest::Character { .. } => parse_character(narrative_id, &json),
        GenerationRequest::Scenes {
            chapter_situation_id,
            ..
        } => parse_scenes(narrative_id, &json, chapter_situation_id),
    }
}

/// Extract the first JSON object from a possibly chatty LLM response. Handles
/// ```json ... ``` code fences and leading/trailing commentary. Shared with
/// the edit engine.
pub fn extract_json_object(raw: &str) -> Result<serde_json::Value> {
    // Strip common code-fence wrappers.
    let trimmed = raw.trim();
    let body = if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        let after = after.strip_prefix("json").unwrap_or(after);
        let after = after.trim_start_matches('\n');
        if let Some(end) = after.find("```") {
            &after[..end]
        } else {
            after
        }
    } else {
        trimmed
    };

    // Find the first `{` and the matching closing `}`.
    let bytes = body.as_bytes();
    let start = body
        .find('{')
        .ok_or_else(|| TensaError::ExtractionError("no JSON object found in response".into()))?;
    let mut depth: i32 = 0;
    let mut end: Option<usize> = None;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match b {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or_else(|| TensaError::ExtractionError("unterminated JSON object".into()))?;
    let slice = &body[start..end];
    serde_json::from_str(slice).map_err(|e| {
        TensaError::ExtractionError(format!(
            "invalid JSON: {} (slice was: {})",
            e,
            truncate_utf8(slice, 200)
        ))
    })
}

fn parse_outline(narrative_id: &str, v: &serde_json::Value) -> Result<GenerationProposal> {
    let chapters = v
        .get("chapters")
        .and_then(|c| c.as_array())
        .ok_or_else(|| {
            TensaError::ExtractionError("outline response missing 'chapters' array".into())
        })?;
    let rationale = v
        .get("rationale")
        .and_then(|r| r.as_str())
        .map(String::from);
    let mut situations = Vec::with_capacity(chapters.len());
    for (i, ch) in chapters.iter().enumerate() {
        let name = ch
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| TensaError::ExtractionError(format!("chapter {} missing name", i)))?
            .to_string();
        let summary = ch.get("summary").and_then(|s| s.as_str()).map(String::from);
        let day_offset = ch
            .get("day_offset")
            .and_then(|d| d.as_f64())
            .unwrap_or(i as f64 * 7.0);
        let duration = ch.get("duration_days").and_then(|d| d.as_f64());
        let (pov, voice) = read_pov_fields(ch);
        situations.push(ProposedSituation {
            name,
            description: summary,
            narrative_level: NarrativeLevel::Arc,
            temporal_day_offset: day_offset,
            temporal_duration_days: duration,
            raw_content_hint: None,
            pov_entity_name: pov,
            voice,
        });
    }
    Ok(GenerationProposal {
        narrative_id: narrative_id.to_string(),
        kind: format!("outline ({} chapters)", situations.len()),
        situations,
        entities: Vec::new(),
        participations: Vec::new(),
        commit_message: None,
        rationale,
    })
}

/// Extract `pov_character` and `voice` from a chapter/scene JSON object.
/// Returns (pov_entity_name, voice). Empty/null strings → `None`. When
/// `pov_character` is present but `voice` is null/missing, default the
/// voice to `"homodiegetic"` (a narrator is a character in the story by
/// default). When `pov_character` is null and `voice` is `"heterodiegetic"`,
/// that represents explicit omniscient voice — keep voice, leave pov None.
fn read_pov_fields(v: &serde_json::Value) -> (Option<String>, Option<String>) {
    let pov = v
        .get("pov_character")
        .and_then(|p| p.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "null")
        .map(String::from);
    let voice = v
        .get("voice")
        .and_then(|p| p.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "null")
        .map(String::from);
    let voice_final = voice.or_else(|| pov.as_ref().map(|_| "homodiegetic".to_string()));
    (pov, voice_final)
}

fn parse_character(narrative_id: &str, v: &serde_json::Value) -> Result<GenerationProposal> {
    let name = v
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| TensaError::ExtractionError("character response missing 'name'".into()))?
        .to_string();
    let entity_type = v
        .get("entity_type")
        .and_then(|t| t.as_str())
        .map(|t| match t {
            "Organization" => EntityType::Organization,
            _ => EntityType::Actor,
        })
        .unwrap_or(EntityType::Actor);
    let description = v
        .get("description")
        .and_then(|d| d.as_str())
        .map(String::from);
    let role_in_story = v
        .get("role_in_story")
        .and_then(|r| r.as_str())
        .map(String::from);
    let motivations: Vec<String> = v
        .get("motivations")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let voice_notes = v
        .get("voice_notes")
        .and_then(|vn| vn.as_str())
        .map(String::from);
    let quirks: Vec<String> = v
        .get("quirks")
        .and_then(|q| q.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let rationale = v
        .get("rationale")
        .and_then(|r| r.as_str())
        .map(String::from);

    let mut props = serde_json::Map::new();
    props.insert("name".into(), serde_json::Value::String(name.clone()));
    if let Some(d) = description {
        props.insert("description".into(), serde_json::Value::String(d));
    }
    if let Some(r) = role_in_story {
        props.insert("role_in_story".into(), serde_json::Value::String(r));
    }
    if !motivations.is_empty() {
        props.insert(
            "motivations".into(),
            serde_json::Value::Array(
                motivations
                    .into_iter()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(v) = voice_notes {
        props.insert("voice_notes".into(), serde_json::Value::String(v));
    }
    if !quirks.is_empty() {
        props.insert(
            "quirks".into(),
            serde_json::Value::Array(quirks.into_iter().map(serde_json::Value::String).collect()),
        );
    }

    Ok(GenerationProposal {
        narrative_id: narrative_id.to_string(),
        kind: format!("character: {}", name),
        situations: Vec::new(),
        entities: vec![ProposedEntity {
            entity_type,
            properties: serde_json::Value::Object(props),
        }],
        participations: Vec::new(),
        commit_message: None,
        rationale,
    })
}

fn parse_scenes(
    narrative_id: &str,
    v: &serde_json::Value,
    chapter_id: &Uuid,
) -> Result<GenerationProposal> {
    let scenes = v.get("scenes").and_then(|s| s.as_array()).ok_or_else(|| {
        TensaError::ExtractionError("scenes response missing 'scenes' array".into())
    })?;
    let rationale = v
        .get("rationale")
        .and_then(|r| r.as_str())
        .map(String::from);
    let mut situations = Vec::with_capacity(scenes.len());
    let mut participations: Vec<ProposedParticipation> = Vec::new();
    for (i, sc) in scenes.iter().enumerate() {
        let name = sc
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| TensaError::ExtractionError(format!("scene {} missing name", i)))?
            .to_string();
        let description = sc
            .get("description")
            .and_then(|d| d.as_str())
            .map(String::from);
        let day_offset = sc
            .get("day_offset")
            .and_then(|d| d.as_f64())
            .unwrap_or(i as f64 * 0.2);
        let duration = sc.get("duration_days").and_then(|d| d.as_f64());
        let (pov, voice) = read_pov_fields(sc);
        situations.push(ProposedSituation {
            name,
            description,
            narrative_level: NarrativeLevel::Scene,
            temporal_day_offset: day_offset,
            temporal_duration_days: duration,
            raw_content_hint: None,
            pov_entity_name: pov,
            voice,
        });
        if let Some(parts) = sc.get("participants").and_then(|p| p.as_array()) {
            for p in parts {
                let name = match p.get("name").and_then(|n| n.as_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                let role_str = p
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("Bystander");
                let role = match role_str {
                    "Protagonist" => Role::Protagonist,
                    "Antagonist" => Role::Antagonist,
                    "Witness" => Role::Witness,
                    "Target" => Role::Target,
                    "Bystander" => Role::Bystander,
                    "Confidant" => Role::Confidant,
                    "Informant" => Role::Informant,
                    "Recipient" => Role::Recipient,
                    "Instrument" => Role::Instrument,
                    "SubjectOfDiscussion" => Role::SubjectOfDiscussion,
                    other => Role::Custom(other.to_string()),
                };
                let action = p.get("action").and_then(|a| a.as_str()).map(String::from);
                participations.push(ProposedParticipation {
                    entity_ref: format!("name:{}", name),
                    situation_ref: format!("new:{}", i),
                    role,
                    action,
                });
            }
        }
    }
    // Every scene becomes a child of the chapter — encode a structural note so
    // apply() can attach them causally or temporally to the chapter. We don't
    // have parent_situation_id in the model, so we record the chapter id here
    // so the apply path can use it if needed.
    let _ = chapter_id; // not persisted; caller-known via GenerationRequest::Scenes
    Ok(GenerationProposal {
        narrative_id: narrative_id.to_string(),
        kind: format!("scenes ({})", situations.len()),
        situations,
        entities: Vec::new(),
        participations,
        commit_message: None,
        rationale,
    })
}

// ─── Apply ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedReport {
    pub revision_id: Uuid,
    pub entities_created: usize,
    pub situations_created: usize,
    pub participations_created: usize,
    #[serde(default)]
    pub causal_links_created: usize,
}

/// Apply a proposal: write entities + situations + participations, then commit
/// a revision.
///
/// Caller is responsible for resolving `entity_ref: "name:..."` to existing
/// entities (not implemented here; keeping the apply path small). For v0.49.0
/// we match on `properties.name` case-insensitively among existing entities.
pub fn apply_proposal(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    proposal: GenerationProposal,
    author: Option<String>,
) -> Result<AppliedReport> {
    let narrative_id = proposal.narrative_id.clone();

    let now = Utc::now();

    // 1. Create entities first so participations can reference them.
    let mut created_entity_ids: Vec<Uuid> = Vec::with_capacity(proposal.entities.len());
    for proposed in &proposal.entities {
        let id = Uuid::now_v7();
        let entity = Entity {
            id,
            entity_type: proposed.entity_type.clone(),
            properties: proposed.properties.clone(),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: Vec::new(),
            extraction_method: Some(ExtractionMethod::LlmParsed),
            narrative_id: Some(narrative_id.clone()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hypergraph.create_entity(entity)?;
        created_entity_ids.push(id);
    }

    // Build a name→id lookup from existing entities for participation resolution.
    // HashMap so resolution is O(1) per participation rather than O(N).
    let existing = hypergraph.list_entities_by_narrative(&narrative_id)?;
    let name_to_id: std::collections::HashMap<String, Uuid> = existing
        .iter()
        .filter_map(|e| {
            e.properties
                .get("name")
                .and_then(|v| v.as_str())
                .map(|n| (n.trim().to_lowercase(), e.id))
        })
        .collect();
    let lookup_name =
        |name: &str| -> Option<Uuid> { name_to_id.get(&name.trim().to_lowercase()).copied() };

    // 2. Create situations.
    // Pair id+level so the downstream adjacency linker can skip cross-level
    // neighbours without maintaining two parallel vecs.
    let mut created_situations: Vec<(Uuid, NarrativeLevel)> =
        Vec::with_capacity(proposal.situations.len());
    for proposed in &proposal.situations {
        let id = Uuid::now_v7();
        let start =
            now + Duration::milliseconds((proposed.temporal_day_offset * 86_400_000.0) as i64);
        let end = proposed
            .temporal_duration_days
            .map(|d| start + Duration::milliseconds((d * 86_400_000.0) as i64));
        let raw_content = proposed
            .raw_content_hint
            .as_ref()
            .map(|c| {
                vec![ContentBlock {
                    content_type: ContentType::Text,
                    content: c.clone(),
                    source: None,
                }]
            })
            .unwrap_or_default();
        // Resolve POV name → entity UUID. Name matches are case-insensitive
        // against both existing entities and entities created earlier in this
        // proposal. Unresolved names do not fail — the situation is still
        // created, with voice set (if the LLM supplied one) and focalization
        // left null; the writer can wire it up later in the Manuscript UI.
        let focalization_id = proposed.pov_entity_name.as_deref().and_then(lookup_name);
        let discourse = if proposed.pov_entity_name.is_some() || proposed.voice.is_some() {
            Some(DiscourseAnnotation {
                order: None,
                duration: None,
                focalization: focalization_id,
                voice: proposed.voice.clone(),
            })
        } else {
            None
        };
        let situation = Situation {
            id,
            properties: serde_json::Value::Null,
            name: Some(proposed.name.clone()),
            description: proposed.description.clone(),
            temporal: AllenInterval {
                start: Some(start),
                end,
                granularity: TimeGranularity::Approximate,
                relations: Vec::new(),
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: Vec::new(),
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content,
            narrative_level: proposed.narrative_level,
            discourse,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: Vec::new(),
            narrative_id: Some(narrative_id.clone()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: None,
            parent_situation_id: None,
            label: None,
            status: None,
            keywords: vec![],
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hypergraph.create_situation(situation)?;
        created_situations.push((id, proposed.narrative_level));
    }

    // 2b. Weak sequential Enabling links between adjacent same-level
    // situations so Workshop's causal analyses have something to read.
    // Cross-level neighbours (Arc→Scene) are skipped to avoid hopping the
    // hierarchy.
    let mut causal_links_created = 0usize;
    for pair in created_situations.windows(2) {
        if pair[0].1 != pair[1].1 {
            continue;
        }
        if causal_helpers::add_sequential_link(
            hypergraph,
            pair[0].0,
            pair[1].0,
            causal_helpers::MECHANISM_SEQUENTIAL,
            0.5,
            MaturityLevel::Candidate,
        )? {
            causal_links_created += 1;
        }
    }

    // 3. Participations.
    let created_situation_ids: Vec<Uuid> = created_situations.iter().map(|(id, _)| *id).collect();
    let mut participations_written = 0usize;
    for p in &proposal.participations {
        let sit_id = resolve_situation_ref(&p.situation_ref, &created_situation_ids)?;
        let entity_id = match resolve_entity_ref(&p.entity_ref, &created_entity_ids) {
            Ok(id) => id,
            Err(_) => {
                // Try name match against existing entities.
                let name = p.entity_ref.strip_prefix("name:").unwrap_or(&p.entity_ref);
                match lookup_name(name) {
                    Some(id) => id,
                    None => {
                        // Skip participations we can't resolve — we don't want
                        // to block the whole proposal on one missing name.
                        continue;
                    }
                }
            }
        };
        hypergraph.add_participant(Participation {
            entity_id,
            situation_id: sit_id,
            role: p.role.clone(),
            info_set: None,
            action: p.action.clone(),
            payoff: None,
            seq: 0,
        })?;
        participations_written += 1;
    }

    // 4. Commit a revision — this is the writer's undo button.
    let message = proposal
        .commit_message
        .unwrap_or_else(|| format!("AI-generated: {}", proposal.kind));
    let outcome = commit_narrative(hypergraph, registry, &narrative_id, message, author)?;
    let revision_id = match outcome {
        CommitOutcome::Committed(r) => r.id,
        CommitOutcome::NoChange(r) => r.id,
    };

    Ok(AppliedReport {
        revision_id,
        entities_created: proposal.entities.len(),
        situations_created: proposal.situations.len(),
        participations_created: participations_written,
        causal_links_created,
    })
}

fn resolve_situation_ref(r: &str, created: &[Uuid]) -> Result<Uuid> {
    if let Some(rest) = r.strip_prefix("new:") {
        let idx: usize = rest.parse().map_err(|_| {
            TensaError::InvalidQuery(format!("bad situation_ref '{}': expected new:N or UUID", r))
        })?;
        created.get(idx).copied().ok_or_else(|| {
            TensaError::InvalidQuery(format!("situation_ref new:{} out of range", idx))
        })
    } else {
        Uuid::parse_str(r).map_err(|_| {
            TensaError::InvalidQuery(format!("bad situation_ref '{}': expected new:N or UUID", r))
        })
    }
}

fn resolve_entity_ref(r: &str, created: &[Uuid]) -> Result<Uuid> {
    if let Some(rest) = r.strip_prefix("new:") {
        let idx: usize = rest
            .parse()
            .map_err(|_| TensaError::InvalidQuery(format!("bad entity_ref '{}'", r)))?;
        created
            .get(idx)
            .copied()
            .ok_or_else(|| TensaError::InvalidQuery(format!("entity_ref new:{} out of range", idx)))
    } else if let Some(_) = r.strip_prefix("name:") {
        // Caller resolves against existing cast.
        Err(TensaError::NotFound(format!(
            "entity_ref {} needs name lookup",
            r
        )))
    } else {
        Uuid::parse_str(r).map_err(|_| TensaError::InvalidQuery(format!("bad entity_ref '{}'", r)))
    }
}

/// Convenience: gather snapshot + generate + return proposal. For the API layer.
pub fn generate_for_narrative(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    extractor: &dyn SessionCapableExtractor,
    cache: Option<&LlmCache>,
    narrative_id: &str,
    request: &GenerationRequest,
) -> Result<GenerationProposal> {
    // Ensure narrative exists (clear error message if not).
    registry.get(narrative_id)?;
    let _ = plan_store::get_plan(hypergraph.store(), narrative_id)?; // just validate KV reachable
    let snapshot = gather_snapshot(hypergraph, registry, narrative_id)?;

    let prompt = build_prompt(request, &snapshot);
    let ledger_kind = match request {
        GenerationRequest::Outline { .. } => "outline",
        GenerationRequest::Character { .. } => "character",
        GenerationRequest::Scenes { .. } => "scenes",
    };
    let model = extractor.model_name();
    let start = std::time::Instant::now();

    // Cache check first — zero-cost return if the same prompt ran recently.
    if let Some(c) = cache {
        if let Ok(Some(cached)) = c.get(&prompt.system, &prompt.user) {
            cost_ledger::record(
                hypergraph.store(),
                narrative_id,
                CostOperation::Generation,
                ledger_kind,
                prompt.estimate.prompt_tokens,
                0,
                model.clone(),
                true,
                true,
                start.elapsed().as_millis() as u64,
                None,
            );
            return parse_response(request, narrative_id, &cached);
        }
    }

    let messages = vec![
        ApiMessage {
            role: "system".into(),
            content: prompt.system.clone(),
        },
        ApiMessage {
            role: "user".into(),
            content: prompt.user.clone(),
        },
    ];
    let result = extractor.send_session_messages(&messages);
    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let response_tokens = approx_tokens(&response);
            if let Some(c) = cache {
                let _ = c.put(&prompt.system, &prompt.user, &response);
            }
            cost_ledger::record(
                hypergraph.store(),
                narrative_id,
                CostOperation::Generation,
                ledger_kind,
                prompt.estimate.prompt_tokens,
                response_tokens,
                model,
                false,
                true,
                duration_ms,
                None,
            );
            parse_response(request, narrative_id, &response)
        }
        Err(e) => {
            cost_ledger::record(
                hypergraph.store(),
                narrative_id,
                CostOperation::Generation,
                ledger_kind,
                prompt.estimate.prompt_tokens,
                0,
                model,
                false,
                false,
                duration_ms,
                None,
            );
            Err(e)
        }
    }
}

/// Convenience: gather snapshot + build prompt. Dry-run path.
pub fn build_prompt_for_narrative(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    narrative_id: &str,
    request: &GenerationRequest,
) -> Result<GenerationPrompt> {
    registry.get(narrative_id)?;
    let snapshot = gather_snapshot(hypergraph, registry, narrative_id)?;
    Ok(build_prompt(request, &snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::llm::NarrativeExtractor;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// Mock extractor that returns a canned response.
    struct MockExtractor {
        response: String,
    }

    impl NarrativeExtractor for MockExtractor {
        fn extract_narrative(
            &self,
            _chunk: &crate::ingestion::chunker::TextChunk,
        ) -> Result<crate::ingestion::extraction::NarrativeExtraction> {
            unimplemented!()
        }
        fn set_cancel_flag(&self, _flag: Arc<AtomicBool>) {}
        fn as_session(&self) -> Option<&dyn SessionCapableExtractor> {
            Some(self)
        }
    }

    impl SessionCapableExtractor for MockExtractor {
        fn send_session_messages(&self, _messages: &[ApiMessage]) -> Result<String> {
            Ok(self.response.clone())
        }
    }

    fn setup() -> (Hypergraph, NarrativeRegistry) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let reg = NarrativeRegistry::new(store);
        reg.create(Narrative {
            id: "draft".into(),
            title: "Draft".into(),
            genre: Some("novel".into()),
            tags: vec![],
            description: None,
            authors: vec![],
            language: None,
            publication_date: None,
            cover_url: None,
            custom_properties: std::collections::HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            source: None,
            project_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
        .unwrap();
        (hg, reg)
    }

    #[test]
    fn prompt_includes_plan_when_present() {
        use crate::narrative::plan;
        let (hg, reg) = setup();
        plan::upsert_plan(
            hg.store(),
            NarrativePlan {
                narrative_id: "draft".into(),
                logline: Some("A grifter learns honesty".into()),
                themes: vec!["identity".into()],
                style: StyleTargets {
                    pov: Some("3rd_limited".into()),
                    tense: Some("past".into()),
                    tone: vec!["grim".into()],
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let prompt = build_prompt(
            &GenerationRequest::Outline {
                premise: None,
                num_chapters: 5,
                tone_hint: None,
                pov_hint: None,
            },
            &snapshot,
        );
        assert!(prompt.user.contains("A grifter learns honesty"));
        assert!(prompt.user.contains("identity"));
        assert!(prompt.user.contains("POV=3rd_limited"));
    }

    #[test]
    fn estimate_is_fast_and_no_llm_needed() {
        let (hg, reg) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let est = estimate_tokens(
            &GenerationRequest::Outline {
                premise: Some("A premise.".into()),
                num_chapters: 10,
                tone_hint: None,
                pov_hint: None,
            },
            &snapshot,
        );
        assert!(est.prompt_tokens > 0);
        assert!(est.expected_response_tokens >= 200);
    }

    #[test]
    fn outline_response_parses_into_proposal() {
        let mock = MockExtractor {
            response: r#"{
              "chapters": [
                {"name":"Ch1","summary":"Alice meets Bob","day_offset":0,"duration_days":1},
                {"name":"Ch2","summary":"Bob betrays Alice","day_offset":7,"duration_days":1}
              ],
              "rationale":"classic setup-betrayal"
            }"#
            .into(),
        };
        let (hg, reg) = setup();
        let proposal = generate_for_narrative(
            &hg,
            &reg,
            &mock,
            None,
            "draft",
            &GenerationRequest::Outline {
                premise: Some("Test".into()),
                num_chapters: 2,
                tone_hint: None,
                pov_hint: None,
            },
        )
        .unwrap();
        assert_eq!(proposal.situations.len(), 2);
        assert_eq!(proposal.situations[0].name, "Ch1");
        assert!(matches!(
            proposal.situations[0].narrative_level,
            NarrativeLevel::Arc
        ));
        assert_eq!(
            proposal.rationale.as_deref(),
            Some("classic setup-betrayal")
        );
    }

    #[test]
    fn character_response_parses() {
        let mock = MockExtractor {
            response: r#"```json
            {
              "name":"Alice",
              "entity_type":"Actor",
              "description":"A grifter with a conscience.",
              "role_in_story":"protagonist and narrator",
              "motivations":["redemption","debt repayment"],
              "voice_notes":"terse, ironic",
              "quirks":["plays chess"],
              "rationale":"fills the protagonist slot"
            }
            ```"#
                .into(),
        };
        let (hg, reg) = setup();
        let proposal = generate_for_narrative(
            &hg,
            &reg,
            &mock,
            None,
            "draft",
            &GenerationRequest::Character {
                seed: CharacterSeed::default(),
            },
        )
        .unwrap();
        assert_eq!(proposal.entities.len(), 1);
        let name = proposal.entities[0]
            .properties
            .get("name")
            .and_then(|v| v.as_str());
        assert_eq!(name, Some("Alice"));
    }

    #[test]
    fn apply_proposal_writes_and_commits() {
        let (hg, reg) = setup();
        let proposal = GenerationProposal {
            narrative_id: "draft".into(),
            kind: "outline (2 chapters)".into(),
            situations: vec![
                ProposedSituation {
                    name: "Ch1".into(),
                    description: Some("opening".into()),
                    narrative_level: NarrativeLevel::Arc,
                    temporal_day_offset: 0.0,
                    temporal_duration_days: Some(1.0),
                    raw_content_hint: None,
                    pov_entity_name: None,
                    voice: None,
                },
                ProposedSituation {
                    name: "Ch2".into(),
                    description: Some("rising action".into()),
                    narrative_level: NarrativeLevel::Arc,
                    temporal_day_offset: 7.0,
                    temporal_duration_days: Some(1.0),
                    raw_content_hint: None,
                    pov_entity_name: None,
                    voice: None,
                },
            ],
            entities: vec![],
            participations: vec![],
            commit_message: None,
            rationale: None,
        };
        let report = apply_proposal(&hg, &reg, proposal, Some("tester".into())).unwrap();
        assert_eq!(report.situations_created, 2);
        let sits = hg.list_situations_by_narrative("draft").unwrap();
        assert_eq!(sits.len(), 2);
        // Adjacent same-level situations get a weak sequential causal edge so
        // Workshop's causal-orphan detector has something to read.
        assert_eq!(report.causal_links_created, 1);
        let antecedents = hg.get_antecedents(&sits[1].id).unwrap();
        let has_link = antecedents
            .iter()
            .chain(hg.get_antecedents(&sits[0].id).unwrap().iter())
            .any(|l| l.mechanism.as_deref() == Some(causal_helpers::MECHANISM_SEQUENTIAL));
        assert!(has_link, "expected a sequential causal link");
    }

    #[test]
    fn apply_proposal_skips_cross_level_adjacency() {
        let (hg, reg) = setup();
        let proposal = GenerationProposal {
            narrative_id: "draft".into(),
            kind: "mixed".into(),
            situations: vec![
                ProposedSituation {
                    name: "Arc".into(),
                    description: None,
                    narrative_level: NarrativeLevel::Arc,
                    temporal_day_offset: 0.0,
                    temporal_duration_days: None,
                    raw_content_hint: None,
                    pov_entity_name: None,
                    voice: None,
                },
                ProposedSituation {
                    name: "Scene".into(),
                    description: None,
                    narrative_level: NarrativeLevel::Scene,
                    temporal_day_offset: 1.0,
                    temporal_duration_days: None,
                    raw_content_hint: None,
                    pov_entity_name: None,
                    voice: None,
                },
            ],
            entities: vec![],
            participations: vec![],
            commit_message: None,
            rationale: None,
        };
        let report = apply_proposal(&hg, &reg, proposal, None).unwrap();
        // Cross-level adjacency (Arc → Scene) should NOT auto-link: we'd risk
        // spurious edges that hop the hierarchy.
        assert_eq!(report.causal_links_created, 0);
    }

    #[test]
    fn extract_json_handles_fences_and_chatter() {
        let raw = "Sure, here you go:\n```json\n{\"chapters\":[]}\n```\nlet me know!";
        let parsed = extract_json_object(raw).unwrap();
        assert!(parsed.get("chapters").is_some());
    }

    #[test]
    fn scenes_response_parses_participants() {
        let mock = MockExtractor {
            response: r#"{
              "scenes":[
                {"name":"S1","description":"Alice enters","day_offset":0,
                 "participants":[{"name":"Alice","role":"Protagonist","action":"enters"}]}
              ],
              "rationale":"intro scene"
            }"#
            .into(),
        };
        let (hg, reg) = setup();
        // Create a chapter first.
        let chapter_id = Uuid::now_v7();
        hg.create_situation(Situation {
            id: chapter_id,
            properties: serde_json::Value::Null,
            name: Some("Chapter 1".into()),
            description: Some("opening".into()),
            temporal: AllenInterval {
                start: Some(Utc::now()),
                end: None,
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
            narrative_level: NarrativeLevel::Arc,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 1.0,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("draft".into()),
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
        .unwrap();

        let proposal = generate_for_narrative(
            &hg,
            &reg,
            &mock,
            None,
            "draft",
            &GenerationRequest::Scenes {
                chapter_situation_id: chapter_id,
                count: 1,
                constraints: SceneConstraints::default(),
                pov_entity_name: None,
            },
        )
        .unwrap();
        assert_eq!(proposal.situations.len(), 1);
        assert_eq!(proposal.participations.len(), 1);
        assert_eq!(proposal.participations[0].entity_ref, "name:Alice");
    }
}
