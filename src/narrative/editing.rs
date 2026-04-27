//! Edit Engine — Sprint W2 (v0.49.1).
//!
//! AI-assisted rewrites of existing prose. Every edit is a *proposal*:
//! [`propose_edit`] returns original + proposed blocks with a pre-computed
//! line diff the UI can render; [`apply_edit`] writes the new raw_content
//! into the situation and commits a revision. No destructive edits.
//!
//! # Invariants
//! - **Reuses the LLM primitive.** `SessionCapableExtractor::send_session_messages`.
//! - **Reuses the snapshot.** Context (plan, cast, neighbouring chapters) is
//!   gathered via `gather_snapshot`.
//! - **Reuses the diff renderer.** `unified_line_diff` from `revision.rs`
//!   produces the same `DiffLine` shape History renders — the Studio uses one
//!   component for both.
//! - **Reuses the revision system.** Applying an edit = `update_situation` +
//!   `commit_narrative`. Undo is the previous revision.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::llm::{ApiMessage, SessionCapableExtractor};
use crate::narrative::generation::GenerationEstimate;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::revision::{
    commit_narrative, gather_snapshot, unified_line_diff, CommitOutcome, DiffLine,
};
use crate::narrative::writer_common::{
    approx_tokens, blocks_to_labeled, count_words_blocks, write_plan_section,
};
use crate::types::*;

// ─── Public types ─────────────────────────────────────────────────

/// One of the five v0.49.1 edit operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditOperation {
    /// Freeform rewrite per a writer-supplied instruction.
    Rewrite { instruction: String },
    /// Tighten to a fraction of the current word count. 0.7 = aim for 70%.
    Tighten {
        target_ratio: f32,
        #[serde(default)]
        instruction: Option<String>,
    },
    /// Expand by a multiplier. 1.5 = aim for 150% of current.
    Expand {
        target_multiplier: f32,
        #[serde(default)]
        instruction: Option<String>,
    },
    /// Transfer to a named / custom / plan-derived style.
    StyleTransfer { target: StyleTarget },
    /// Rewrite dialogue in the voice of an existing character. The LLM sees
    /// that character's existing dialogue from the narrative as a voice
    /// sample.
    DialoguePass {
        entity_id: Uuid,
        #[serde(default)]
        instruction: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StyleTarget {
    /// A named preset — see [`style_preset_instruction`].
    Preset { name: String },
    /// Freeform writer-supplied style instruction.
    Custom { prompt: String },
    /// Use the narrative's `plan.style` (pov / tense / tone / voice / influences / avoid).
    FromPlan,
}

/// Built-in style presets. Simple instructions to nudge the LLM in a direction;
/// they are NOT voice-stealing — they don't attempt to reproduce a specific
/// author's prose.
pub fn style_preset_instruction(name: &str) -> Option<&'static str> {
    Some(match name {
        "minimal" => "Write in a minimalist register: short declarative sentences, plain vocabulary, concrete nouns over abstract ones. Cut filler. Let silence carry weight.",
        "lyrical" => "Write in a lyrical register: longer rhythmic sentences, selective imagery, attention to sound. No purple prose.",
        "punchy" => "Write with punchy momentum: vary sentence length aggressively, lean into active verbs, cut qualifiers (very, quite, rather, somewhat). No adverb pileups.",
        "formal" => "Write in a formal register: full syntax, restrained vocabulary, measured pacing. Avoid contractions and slang unless a character's voice requires them.",
        "interior" => "Stay close to the POV character's interior: register shifts with their mood, narration slips into free indirect discourse where useful.",
        "cinematic" => "Write cinematically: external, visible, hearable. Minimize interior access. Let action and dialogue carry the scene.",
        _ => return None,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditProposal {
    pub narrative_id: String,
    pub situation_id: Uuid,
    pub operation: EditOperation,
    pub original: Vec<ContentBlock>,
    pub proposed: Vec<ContentBlock>,
    pub original_word_count: usize,
    pub proposed_word_count: usize,
    /// Pre-computed line diff for the UI. Same shape History uses.
    pub diff: Vec<DiffLine>,
    #[serde(default)]
    pub rationale: Option<String>,
}

/// Assembled prompt for dry-run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditPrompt {
    pub system: String,
    pub user: String,
    pub estimate: GenerationEstimate,
}

/// Applied report — returned from [`apply_edit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedEditReport {
    pub revision_id: Uuid,
    pub situation_id: Uuid,
    pub words_before: usize,
    pub words_after: usize,
}

// ─── Prompt assembly ──────────────────────────────────────────────

pub fn build_prompt_for_edit(
    situation: &Situation,
    snapshot: &NarrativeSnapshot,
    operation: &EditOperation,
) -> EditPrompt {
    let system = match operation {
        EditOperation::Rewrite { .. } => SYSTEM_REWRITE,
        EditOperation::Tighten { .. } => SYSTEM_TIGHTEN,
        EditOperation::Expand { .. } => SYSTEM_EXPAND,
        EditOperation::StyleTransfer { .. } => SYSTEM_STYLE,
        EditOperation::DialoguePass { .. } => SYSTEM_DIALOGUE,
    }
    .to_string();

    let mut user = String::new();
    write_plan_section(&mut user, snapshot, false);
    write_neighbors(&mut user, snapshot, &situation.id, 2);

    user.push_str("[target situation]\n");
    if let Some(name) = &situation.name {
        user.push_str(&format!("Name: {}\n", name));
    }
    if let Some(desc) = &situation.description {
        user.push_str(&format!("Summary: {}\n", desc));
    }
    user.push('\n');

    user.push_str("[current blocks]\n");
    user.push_str(&blocks_to_labeled(&situation.raw_content));
    user.push('\n');

    let words = count_words_blocks(&situation.raw_content);
    user.push_str(&format!("[current word count]\n{}\n\n", words));

    // Operation-specific suffix.
    match operation {
        EditOperation::Rewrite { instruction } => {
            user.push_str("[rewrite instruction]\n");
            user.push_str(instruction);
            user.push('\n');
        }
        EditOperation::Tighten {
            target_ratio,
            instruction,
        } => {
            let target_words = ((words as f32) * target_ratio).round() as u32;
            user.push_str(&format!(
                "[tighten instruction]\nAim for around {} words (current {}). Preserve the scene's beats; cut filler, redundancy, adverb pileups, throat-clearing.",
                target_words, words
            ));
            if let Some(extra) = instruction {
                user.push_str("\nAdditional: ");
                user.push_str(extra);
            }
            user.push('\n');
        }
        EditOperation::Expand {
            target_multiplier,
            instruction,
        } => {
            let target_words = ((words as f32) * target_multiplier).round() as u32;
            user.push_str(&format!(
                "[expand instruction]\nAim for around {} words (current {}). Add texture, sensory detail, beats that deepen what's already there. Do not invent new plot.",
                target_words, words
            ));
            if let Some(extra) = instruction {
                user.push_str("\nAdditional: ");
                user.push_str(extra);
            }
            user.push('\n');
        }
        EditOperation::StyleTransfer { target } => {
            user.push_str("[style target]\n");
            match target {
                StyleTarget::Preset { name } => {
                    user.push_str(&format!("Preset: {}\n", name));
                    if let Some(ins) = style_preset_instruction(name) {
                        user.push_str(ins);
                        user.push('\n');
                    }
                }
                StyleTarget::Custom { prompt } => {
                    user.push_str(prompt);
                    user.push('\n');
                }
                StyleTarget::FromPlan => {
                    user.push_str("Match the narrative's plan.style exactly (see [narrative plan] above). If the plan's style is sparse, preserve existing style and do no harm.\n");
                }
            }
        }
        EditOperation::DialoguePass {
            entity_id,
            instruction,
        } => {
            user.push_str("[dialogue voice sample]\n");
            let character = snapshot.entities.iter().find(|e| e.id == *entity_id);
            if let Some(c) = character {
                let name = c
                    .properties
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("the character");
                user.push_str(&format!("Character: {}\n", name));
                if let Some(voice) = c.properties.get("voice_notes").and_then(|v| v.as_str()) {
                    user.push_str(&format!("Voice notes: {}\n", voice));
                }
                // Gather up to ~20 lines of this entity's existing dialogue from
                // other situations so the LLM has a voice fingerprint.
                let samples = collect_dialogue_samples(snapshot, entity_id, 20);
                if !samples.is_empty() {
                    user.push_str("Existing dialogue samples:\n");
                    for s in &samples {
                        user.push_str(&format!("  \u{2022} {}\n", s));
                    }
                }
            } else {
                user.push_str("(character not found in snapshot; fall back to the situation's existing dialogue voice)\n");
            }
            user.push_str("\nRewrite every Dialogue block in this character's voice. Preserve the dialogue's meaning and beat. Leave Text / Observation / Document blocks alone unless adjustment is required to accommodate the new dialogue.");
            if let Some(extra) = instruction {
                user.push_str("\nAdditional: ");
                user.push_str(extra);
            }
            user.push('\n');
        }
    }

    user.push_str("\n[output]\nReturn ONLY a JSON object with the shape: \
{\"blocks\":[{\"content_type\":\"Text\"|\"Dialogue\"|\"Observation\"|\"Document\",\"content\":str}], \
\"rationale\":str}. Preserve the same number and kind of blocks when possible; splitting or merging is allowed if the instruction requires it.");

    let prompt_tokens = approx_tokens(&system) + approx_tokens(&user);
    let expected_response_tokens = (words as u32 * 2).clamp(200, 6000);
    EditPrompt {
        system,
        user,
        estimate: GenerationEstimate {
            prompt_tokens,
            expected_response_tokens,
        },
    }
}

pub fn estimate_edit_tokens(
    situation: &Situation,
    snapshot: &NarrativeSnapshot,
    operation: &EditOperation,
) -> GenerationEstimate {
    build_prompt_for_edit(situation, snapshot, operation).estimate
}

const SYSTEM_REWRITE: &str = "You are a prose editor. You rewrite a scene's content blocks to satisfy a writer-supplied instruction. \
Preserve the scene's meaning and participants unless the instruction explicitly changes them. \
Return the rewritten blocks in the same JSON shape as the input, with a short rationale.";

const SYSTEM_TIGHTEN: &str = "You are a prose editor specialising in cutting. \
Reduce word count to the target while preserving every beat, meaningful image, and character line. \
Cut: filler phrases, adverb pileups, redundant gestures, clause repetitions, weak verbs with modifiers (was walking \u{2192} walked). \
Return the tightened blocks in the same JSON shape as the input.";

const SYSTEM_EXPAND: &str = "You are a prose editor specialising in enlargement. \
Grow the scene to the target word count by adding sensory detail, physical beats, small interior moments \u{2014} never new plot. \
Preserve the sequence of events exactly. Return the expanded blocks in the same JSON shape.";

const SYSTEM_STYLE: &str = "You are a prose editor specialising in register. \
Rewrite the scene to match the given style target. Do not change who does what. \
Return the restyled blocks in the same JSON shape.";

const SYSTEM_DIALOGUE: &str = "You are a prose editor specialising in character voice. \
Rewrite every Dialogue block so the named character speaks in their characteristic register, vocabulary, cadence. \
Preserve the meaning and beat of each line. Return the updated blocks in the same JSON shape.";

fn write_neighbors(
    out: &mut String,
    snapshot: &NarrativeSnapshot,
    target_id: &Uuid,
    radius: usize,
) {
    // Pull the N preceding + N following situations in temporal order for context.
    let mut sorted: Vec<&Situation> = snapshot.situations.iter().collect();
    sorted.sort_by_key(|s| s.temporal.start);
    let idx = sorted.iter().position(|s| s.id == *target_id);
    let Some(idx) = idx else { return };
    let lo = idx.saturating_sub(radius);
    let hi = (idx + radius + 1).min(sorted.len());

    out.push_str("[neighbours]\n");
    for (i, s) in sorted[lo..hi].iter().enumerate() {
        if lo + i == idx {
            out.push_str(&format!(
                "  >> {} (target)\n",
                s.name.as_deref().unwrap_or("(untitled)")
            ));
        } else {
            let summary = s.description.as_deref().unwrap_or("");
            out.push_str(&format!(
                "  {} \u{2014} {}\n",
                s.name.as_deref().unwrap_or("(untitled)"),
                summary
            ));
        }
    }
    out.push('\n');
}

fn collect_dialogue_samples(
    snapshot: &NarrativeSnapshot,
    entity_id: &Uuid,
    cap: usize,
) -> Vec<String> {
    // Situations this entity participates in, filtered to dialogue blocks of
    // the situation. This is a rough voice sample — a real DialoguePass would
    // use sentence-level attribution, but for v1 any existing dialogue in
    // scenes where this character participates is a reasonable starting point.
    let sit_ids: std::collections::HashSet<Uuid> = snapshot
        .participations
        .iter()
        .filter(|p| p.entity_id == *entity_id)
        .map(|p| p.situation_id)
        .collect();
    let mut out: Vec<String> = Vec::new();
    for s in snapshot
        .situations
        .iter()
        .filter(|s| sit_ids.contains(&s.id))
    {
        for b in &s.raw_content {
            if matches!(b.content_type, ContentType::Dialogue) {
                let line = b.content.trim();
                if !line.is_empty() {
                    out.push(line.to_string());
                    if out.len() >= cap {
                        return out;
                    }
                }
            }
        }
    }
    out
}

// ─── Propose ──────────────────────────────────────────────────────

pub fn propose_edit(
    extractor: &dyn SessionCapableExtractor,
    snapshot: &NarrativeSnapshot,
    situation_id: &Uuid,
    operation: &EditOperation,
) -> Result<EditProposal> {
    let situation = snapshot
        .situations
        .iter()
        .find(|s| s.id == *situation_id)
        .ok_or_else(|| TensaError::SituationNotFound(*situation_id))?
        .clone();

    let prompt = build_prompt_for_edit(&situation, snapshot, operation);
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
    let (proposed, rationale) = parse_edit_response(&response, &situation.raw_content)?;

    let original = situation.raw_content.clone();
    let diff = unified_line_diff(&blocks_to_prose(&original), &blocks_to_prose(&proposed));

    Ok(EditProposal {
        narrative_id: situation.narrative_id.clone().unwrap_or_default(),
        situation_id: situation.id,
        operation: operation.clone(),
        original_word_count: count_words_blocks(&situation.raw_content),
        proposed_word_count: count_words_blocks(&proposed),
        original,
        proposed,
        diff,
        rationale,
    })
}

fn blocks_to_prose(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(|b| b.content.trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_edit_response(
    raw: &str,
    original: &[ContentBlock],
) -> Result<(Vec<ContentBlock>, Option<String>)> {
    // Reuse the fence-tolerant JSON extractor from the generation module.
    let json = crate::narrative::generation::extract_json_object(raw)?;
    let blocks_v = json
        .get("blocks")
        .and_then(|b| b.as_array())
        .ok_or_else(|| TensaError::ExtractionError("edit response missing 'blocks'".into()))?;
    let mut blocks: Vec<ContentBlock> = Vec::with_capacity(blocks_v.len());
    for (i, b) in blocks_v.iter().enumerate() {
        let ct = b
            .get("content_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Text");
        let content_type = match ct {
            "Dialogue" => ContentType::Dialogue,
            "Observation" => ContentType::Observation,
            "Document" => ContentType::Document,
            "MediaRef" => ContentType::MediaRef,
            _ => ContentType::Text,
        };
        let content = b
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::ExtractionError(format!("block {} missing content", i)))?
            .to_string();
        blocks.push(ContentBlock {
            content_type,
            content,
            source: None,
        });
    }
    // Fall back: if the LLM returned nothing usable, keep the original. Better
    // to no-op than to truncate the writer's scene.
    if blocks.is_empty() {
        blocks = original.to_vec();
    }
    let rationale = json
        .get("rationale")
        .and_then(|v| v.as_str())
        .map(String::from);
    Ok((blocks, rationale))
}

// ─── Apply ────────────────────────────────────────────────────────

pub fn apply_edit(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    proposal: EditProposal,
    author: Option<String>,
) -> Result<AppliedEditReport> {
    let EditProposal {
        narrative_id,
        situation_id,
        operation,
        original_word_count,
        proposed_word_count,
        proposed,
        ..
    } = proposal;

    // Rewrite raw_content on the situation.
    hypergraph.update_situation(&situation_id, |s| {
        s.raw_content = proposed.clone();
        s.updated_at = Utc::now();
    })?;

    let message = edit_commit_message(&operation, original_word_count, proposed_word_count);
    let outcome = commit_narrative(hypergraph, registry, &narrative_id, message, author)?;
    let revision_id = match outcome {
        CommitOutcome::Committed(r) => r.id,
        CommitOutcome::NoChange(r) => r.id,
    };

    Ok(AppliedEditReport {
        revision_id,
        situation_id,
        words_before: original_word_count,
        words_after: proposed_word_count,
    })
}

fn edit_commit_message(op: &EditOperation, before: usize, after: usize) -> String {
    let head = match op {
        EditOperation::Rewrite { .. } => "AI edit: rewrite".to_string(),
        EditOperation::Tighten { target_ratio, .. } => {
            format!(
                "AI edit: tighten to ~{}%",
                (target_ratio * 100.0).round() as u32
            )
        }
        EditOperation::Expand {
            target_multiplier, ..
        } => {
            format!(
                "AI edit: expand to ~{}%",
                (target_multiplier * 100.0).round() as u32
            )
        }
        EditOperation::StyleTransfer { target } => match target {
            StyleTarget::Preset { name } => format!("AI edit: style={}", name),
            StyleTarget::Custom { .. } => "AI edit: style=custom".to_string(),
            StyleTarget::FromPlan => "AI edit: style=from-plan".to_string(),
        },
        EditOperation::DialoguePass { .. } => "AI edit: dialogue pass".to_string(),
    };
    format!("{} ({}\u{2192}{} words)", head, before, after)
}

// ─── Convenience wrappers for the API layer ───────────────────────

pub fn propose_edit_for_situation(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    extractor: &dyn SessionCapableExtractor,
    situation_id: &Uuid,
    operation: &EditOperation,
) -> Result<EditProposal> {
    let situation = hypergraph.get_situation(situation_id)?;
    let narrative_id = situation
        .narrative_id
        .as_deref()
        .ok_or_else(|| {
            TensaError::InvalidQuery("situation has no narrative_id; cannot edit".into())
        })?
        .to_string();
    let snapshot = gather_snapshot(hypergraph, registry, &narrative_id)?;

    let kind = edit_kind_label(operation);
    let prompt = build_prompt_for_edit(&situation, &snapshot, operation);
    let model = extractor.model_name();
    let start = std::time::Instant::now();
    let result = propose_edit(extractor, &snapshot, situation_id, operation);
    let duration_ms = start.elapsed().as_millis() as u64;

    let (response_tokens, success) = match &result {
        Ok(p) => (
            p.proposed
                .iter()
                .map(|b| approx_tokens(&b.content))
                .sum::<u32>(),
            true,
        ),
        Err(_) => (0, false),
    };
    crate::narrative::cost_ledger::record(
        hypergraph.store(),
        &narrative_id,
        CostOperation::Edit,
        kind,
        prompt.estimate.prompt_tokens,
        response_tokens,
        model,
        false,
        success,
        duration_ms,
        None,
    );
    result
}

fn edit_kind_label(op: &EditOperation) -> &'static str {
    match op {
        EditOperation::Rewrite { .. } => "rewrite",
        EditOperation::Tighten { .. } => "tighten",
        EditOperation::Expand { .. } => "expand",
        EditOperation::StyleTransfer { .. } => "style_transfer",
        EditOperation::DialoguePass { .. } => "dialogue_pass",
    }
}

pub fn build_prompt_for_situation(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    situation_id: &Uuid,
    operation: &EditOperation,
) -> Result<EditPrompt> {
    let situation = hypergraph.get_situation(situation_id)?;
    let narrative_id = situation
        .narrative_id
        .as_deref()
        .ok_or_else(|| TensaError::InvalidQuery("situation has no narrative_id".into()))?
        .to_string();
    let snapshot = gather_snapshot(hypergraph, registry, &narrative_id)?;
    Ok(build_prompt_for_edit(&situation, &snapshot, operation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::llm::NarrativeExtractor;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    struct MockExtractor(String);
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
            Ok(self.0.clone())
        }
    }

    fn setup() -> (Hypergraph, NarrativeRegistry, Uuid) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let reg = NarrativeRegistry::new(store);
        reg.create(Narrative {
            id: "draft".into(),
            title: "Draft".into(),
            genre: None,
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
        let sit_id = Uuid::now_v7();
        hg.create_situation(Situation {
            id: sit_id,
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
            raw_content: vec![
                ContentBlock {
                    content_type: ContentType::Text,
                    content: "She stood at the window for a long time.".into(),
                    source: None,
                },
                ContentBlock {
                    content_type: ContentType::Dialogue,
                    content: "\"I don't know how to begin,\" she said.".into(),
                    source: None,
                },
            ],
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
        (hg, reg, sit_id)
    }

    #[test]
    fn rewrite_prompt_includes_blocks_and_instruction() {
        let (hg, reg, sit_id) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let situation = hg.get_situation(&sit_id).unwrap();
        let op = EditOperation::Rewrite {
            instruction: "Cut filler; tighten the opening.".into(),
        };
        let p = build_prompt_for_edit(&situation, &snapshot, &op);
        assert!(p.user.contains("[current blocks]"));
        assert!(p.user.contains("She stood at the window"));
        assert!(p.user.contains("Cut filler"));
    }

    #[test]
    fn tighten_prompt_computes_target_words() {
        let (hg, reg, sit_id) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let situation = hg.get_situation(&sit_id).unwrap();
        let op = EditOperation::Tighten {
            target_ratio: 0.5,
            instruction: None,
        };
        let p = build_prompt_for_edit(&situation, &snapshot, &op);
        // Situation has ~15 words; target at 0.5 is ~8.
        assert!(p.user.contains("Aim for around"));
    }

    #[test]
    fn estimate_is_llm_free() {
        let (hg, reg, sit_id) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let situation = hg.get_situation(&sit_id).unwrap();
        let est = estimate_edit_tokens(
            &situation,
            &snapshot,
            &EditOperation::Tighten {
                target_ratio: 0.7,
                instruction: None,
            },
        );
        assert!(est.prompt_tokens > 0);
        assert!(est.expected_response_tokens >= 200);
    }

    #[test]
    fn propose_parses_and_builds_diff() {
        let mock = MockExtractor(
            r#"{"blocks":[
              {"content_type":"Text","content":"She stood, silent."},
              {"content_type":"Dialogue","content":"\"I can't.\""}
            ],"rationale":"tighter opening"}"#
                .into(),
        );
        let (hg, reg, sit_id) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let proposal = propose_edit(
            &mock,
            &snapshot,
            &sit_id,
            &EditOperation::Tighten {
                target_ratio: 0.3,
                instruction: None,
            },
        )
        .unwrap();
        assert_eq!(proposal.proposed.len(), 2);
        assert!(proposal.proposed_word_count < proposal.original_word_count);
        assert!(!proposal.diff.is_empty());
        // Diff should contain at least one Added (new text) and Removed (old).
        let has_added = proposal
            .diff
            .iter()
            .any(|l| matches!(l, DiffLine::Added(_)));
        let has_removed = proposal
            .diff
            .iter()
            .any(|l| matches!(l, DiffLine::Removed(_)));
        assert!(has_added && has_removed);
    }

    #[test]
    fn apply_writes_and_commits_revision() {
        let mock = MockExtractor(
            r#"{"blocks":[{"content_type":"Text","content":"Silent."}],"rationale":"brutal cut"}"#
                .into(),
        );
        let (hg, reg, sit_id) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let proposal = propose_edit(
            &mock,
            &snapshot,
            &sit_id,
            &EditOperation::Tighten {
                target_ratio: 0.1,
                instruction: None,
            },
        )
        .unwrap();
        let report = apply_edit(&hg, &reg, proposal, Some("tester".into())).unwrap();
        assert_eq!(report.situation_id, sit_id);
        let after = hg.get_situation(&sit_id).unwrap();
        assert_eq!(after.raw_content.len(), 1);
        assert_eq!(after.raw_content[0].content, "Silent.");
        // Revision recorded.
        let revs = crate::narrative::revision::list_revisions(hg.store(), "draft").unwrap();
        assert_eq!(revs.len(), 1);
        assert!(revs[0].message.starts_with("AI edit: tighten"));
    }

    #[test]
    fn style_preset_lookup_is_stable() {
        assert!(style_preset_instruction("minimal").is_some());
        assert!(style_preset_instruction("doesnotexist").is_none());
    }

    #[test]
    fn empty_response_keeps_original_blocks() {
        let mock = MockExtractor(r#"{"blocks":[],"rationale":"empty"}"#.into());
        let (hg, reg, sit_id) = setup();
        let snapshot = gather_snapshot(&hg, &reg, "draft").unwrap();
        let proposal = propose_edit(
            &mock,
            &snapshot,
            &sit_id,
            &EditOperation::Rewrite {
                instruction: "test".into(),
            },
        )
        .unwrap();
        assert_eq!(proposal.proposed.len(), proposal.original.len());
    }
}
