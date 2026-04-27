//! Prompt construction from hypergraph queries (Sprint D9.7).
//!
//! For each situation in a chapter, queries the materialized hypergraph to
//! extract character knowledge states, motivation, relationships, causal
//! context, commitment status, and dramatic irony — then renders these into
//! a structured LLM prompt.

use crate::analysis::stylometry::prose_features_to_directives;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::style::embedding::{load_embedding, StyleEmbedding, StyleEmbeddingSource};

use super::types::*;

/// Build a generation prompt for a specific chapter by querying the hypergraph.
pub fn build_chapter_prompt(
    hg: &Hypergraph,
    narrative_id: &str,
    chapter: usize,
    style: &StyleTarget,
    preceding_summaries: &[String],
) -> Result<GenerationPrompt> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    let mut sorted_sits: Vec<_> = situations.iter().collect();
    sorted_sits.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));

    // Find situations for this chapter
    let chapter_situations: Vec<_> = sorted_sits
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx == chapter)
        .map(|(_, s)| *s)
        .collect();

    if chapter_situations.is_empty() {
        return Err(TensaError::NotFound(format!(
            "no situations for chapter {}",
            chapter
        )));
    }

    // Build style system prompt
    let system_prompt = build_style_prompt(hg, style)?;

    // Build chapter context from preceding summaries
    let chapter_context = if preceding_summaries.is_empty() {
        "This is the opening chapter.".to_string()
    } else {
        let recent: Vec<_> = preceding_summaries.iter().rev().take(3).rev().collect();
        format!(
            "Previous chapters summary:\n{}",
            recent
                .iter()
                .enumerate()
                .map(|(i, s)| format!("Chapter {}: {}", chapter - recent.len() + i, s))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    // Build situation specs
    let mut situation_specs = Vec::new();
    for sit in &chapter_situations {
        let spec = build_situation_prompt(hg, narrative_id, sit, chapter)?;
        situation_specs.push(spec);
    }

    Ok(GenerationPrompt {
        system_prompt,
        chapter_context,
        situation_specs,
        constraints: Vec::new(),
    })
}

/// Build style instructions from a StyleTarget.
///
/// Order: base persona → voice_description → SE-derived voice (if a stored
/// `style_embedding_id` resolves) → fingerprint directives (if a target
/// fingerprint is set) → execution instructions.
///
/// SE rendering uses the [`StyleEmbeddingSource`] enum, never the opaque
/// vector — there is no decoder. A missing/deleted SE record logs a warning
/// and skips the SE fragment instead of failing generation.
fn build_style_prompt(hg: &Hypergraph, style: &StyleTarget) -> Result<String> {
    let mut parts = Vec::new();

    parts.push(
        "You are a skilled novelist. Write prose for the following chapter specification."
            .to_string(),
    );

    if let Some(voice) = &style.voice_description {
        parts.push(format!("Voice and style: {}", voice));
    }

    if let Some(emb_id) = style.style_embedding_id {
        match load_embedding(hg, &emb_id)? {
            Some(emb) => parts.push(render_style_embedding(hg, &emb)),
            None => tracing::warn!(
                "style_embedding_id {} not found; continuing without SE conditioning",
                emb_id
            ),
        }
    }

    if let Some(fp) = &style.target_fingerprint {
        let directives = prose_features_to_directives(&fp.prose);
        if !directives.is_empty() {
            parts.push(format!("Style targets:\n  - {}", directives.join("\n  - ")));
        }
    }

    parts.push(format!(
        "Write naturally and engagingly. Temperature: {:.1}. \
         Follow the scene specifications precisely — each character's knowledge, \
         motivation, and relationships are exact. Characters must not reference \
         information they don't have.",
        style.temperature
    ));

    Ok(parts.join("\n\n"))
}

/// Render a `StyleEmbedding` into a natural-language voice instruction.
///
/// The vector is opaque — this only consumes the [`StyleEmbeddingSource`]
/// metadata, resolving author UUIDs to entity names where possible.
fn render_style_embedding(hg: &Hypergraph, emb: &StyleEmbedding) -> String {
    match &emb.source {
        StyleEmbeddingSource::SingleAuthor { author_id } => {
            let name = author_name(hg, author_id);
            format!("Write in the voice of {}.", name)
        }
        StyleEmbeddingSource::Blended { sources } => {
            if sources.is_empty() {
                return "Blend voices: (no sources specified).".into();
            }
            let mut fragments: Vec<String> = sources
                .iter()
                .map(|(id, weight)| {
                    let pct = (weight * 100.0).round() as u32;
                    format!("{}% {}", pct, author_name(hg, id))
                })
                .collect();
            // Stable order: by descending weight (already user-supplied; preserve as-is).
            // Single-element fragments get a special rendering for naturalness.
            if fragments.len() == 1 {
                format!(
                    "Write predominantly in the voice of {}.",
                    fragments.remove(0)
                )
            } else {
                format!("Blend voices: {}.", fragments.join(", "))
            }
        }
        StyleEmbeddingSource::GenreComposite { genre, .. } => {
            format!("Write in the {} tradition.", genre)
        }
        StyleEmbeddingSource::Custom { label } => format!("Style: {}.", label),
    }
}

fn author_name(hg: &Hypergraph, id: &uuid::Uuid) -> String {
    match hg.get_entity(id) {
        Ok(entity) => entity
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| short_uuid_label(id)),
        Err(_) => short_uuid_label(id),
    }
}

fn short_uuid_label(id: &uuid::Uuid) -> String {
    let s = id.to_string();
    let head = s.split('-').next().unwrap_or(&s);
    format!("author {}", head)
}

/// Build a prompt specification for a single situation.
fn build_situation_prompt(
    hg: &Hypergraph,
    narrative_id: &str,
    sit: &crate::types::Situation,
    chapter: usize,
) -> Result<SituationPrompt> {
    let participants = hg.get_participants_for_situation(&sit.id)?;

    // Build character contexts
    let mut character_contexts = Vec::new();
    for p in &participants {
        let entity = hg.get_entity(&p.entity_id)?;
        let name = entity
            .properties
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string();

        // Extract knowledge from info_set
        let knows: Vec<String> = p
            .info_set
            .as_ref()
            .map(|i| {
                let mut all: Vec<String> = i.knows_before.iter().map(|f| f.fact.clone()).collect();
                all.extend(i.learns.iter().map(|f| f.fact.clone()));
                all
            })
            .unwrap_or_default();

        // Extract false beliefs from entity properties
        let false_beliefs: Vec<String> = entity
            .properties
            .get("lie")
            .and_then(|v| v.as_str())
            .map(|l| vec![l.to_string()])
            .unwrap_or_default();

        // Motivation from stored IRL vectors
        let mot_key = format!("irl/{}/t{}", p.entity_id, chapter).into_bytes();
        let motivation = match hg.store().get(&mot_key)? {
            Some(v) => {
                let vec: Vec<f64> = serde_json::from_slice(&v).unwrap_or_default();
                format!("Motivation vector: {:?}", vec)
            }
            None => "Motivation: (not specified)".into(),
        };

        // Relationships
        let relationships: Vec<String> = entity
            .properties
            .get("relationships")
            .and_then(|v| serde_json::from_value::<Vec<(String, String)>>(v.clone()).ok())
            .unwrap_or_default()
            .iter()
            .map(|(id, rel)| format!("{}: {}", rel, id))
            .collect();

        // Arc phase
        let arc_phase = entity
            .properties
            .get("arc_type")
            .and_then(|v| v.as_str())
            .unwrap_or("flat")
            .to_string();

        character_contexts.push(CharacterContext {
            entity_id: p.entity_id,
            name,
            knows,
            false_beliefs,
            motivation,
            relationships,
            arc_phase,
        });
    }

    // Scene type instruction
    let meta_key = format!("gm/{}/{}", narrative_id, sit.id).into_bytes();
    let scene_type_instruction = match hg.store().get(&meta_key)? {
        Some(v) => {
            let meta: serde_json::Value = serde_json::from_slice(&v).unwrap_or_default();
            format!(
                "Scene type: {}. Narration mode: {}. Emotional valence: {}.",
                meta.get("scene_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("scene"),
                meta.get("narration_mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("scene"),
                meta.get("emotional_valence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5),
            )
        }
        None => "Scene type: standard scene.".into(),
    };

    // Causal context
    let causal_context = if sit.causes.is_empty() {
        "This is the opening situation.".into()
    } else {
        let cause_names: Vec<String> = sit
            .causes
            .iter()
            .filter_map(|c| hg.get_situation(&c.from_situation).ok())
            .filter_map(|s| s.name.clone())
            .collect();
        format!("This follows from: {}", cause_names.join(", "))
    };

    // Commitment instructions from metadata
    let commitment_instructions = Vec::new(); // Populated by caller from plan

    // Dramatic irony
    let dramatic_irony_instructions = Vec::new(); // Populated by caller

    Ok(SituationPrompt {
        situation_id: sit.id,
        summary: sit
            .description
            .clone()
            .or(sit.name.clone())
            .unwrap_or_default(),
        character_contexts,
        scene_type_instruction,
        narration_mode_instruction: String::new(),
        commitment_instructions,
        dramatic_irony_instructions,
        causal_context,
    })
}

/// Render the system half of a `GenerationPrompt` for LLM submission.
///
/// Two-message form (system + user) is the natural shape for chat completion
/// APIs and is what [`ChapterGenerator`] expects.
pub fn render_system(prompt: &GenerationPrompt) -> String {
    prompt.system_prompt.clone()
}

/// Render the user half of a `GenerationPrompt`: chapter context, every
/// situation spec, optional corrective constraints, and the trailing
/// instructions block.
pub fn render_user(prompt: &GenerationPrompt) -> String {
    let mut parts = Vec::new();

    parts.push(format!("=== CONTEXT ===\n{}", prompt.chapter_context));

    for (i, spec) in prompt.situation_specs.iter().enumerate() {
        parts.push(format!("\n=== SITUATION {} ===", i + 1));
        parts.push(format!("Summary: {}", spec.summary));
        parts.push(format!("Scene: {}", spec.scene_type_instruction));
        parts.push(format!("Causality: {}", spec.causal_context));

        for ctx in &spec.character_contexts {
            parts.push(format!(
                "\nCharacter: {} (arc: {})\n  Knows: {:?}\n  False beliefs: {:?}\n  {}\n  Relationships: {:?}",
                ctx.name, ctx.arc_phase,
                ctx.knows, ctx.false_beliefs,
                ctx.motivation, ctx.relationships
            ));
        }

        if !spec.commitment_instructions.is_empty() {
            parts.push(format!(
                "Commitments: {}",
                spec.commitment_instructions.join("; ")
            ));
        }
        if !spec.dramatic_irony_instructions.is_empty() {
            parts.push(format!(
                "Dramatic irony: {}",
                spec.dramatic_irony_instructions.join("; ")
            ));
        }
    }

    if !prompt.constraints.is_empty() {
        parts.push(format!(
            "\n=== CORRECTIVE CONSTRAINTS ===\n{}",
            prompt.constraints.join("\n")
        ));
    }

    parts.push(
        "\n=== INSTRUCTIONS ===\nWrite the chapter prose now. Follow all specifications precisely."
            .to_string(),
    );

    parts.join("\n")
}

/// Backwards-compatible single-string rendering. Prefer [`render_system`] +
/// [`render_user`] for two-message LLM APIs.
pub fn render_prompt(prompt: &GenerationPrompt) -> String {
    format!(
        "=== SYSTEM ===\n{}\n{}",
        render_system(prompt),
        render_user(prompt)
    )
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{materializer, planner};
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;
    use uuid::Uuid;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_build_chapter_prompt() {
        let hg = test_hg();
        let config = PlanConfig {
            premise: "A detective investigates a murder".into(),
            chapter_count: 3,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        materializer::materialize_plan(&hg, &plan).unwrap();

        let style = StyleTarget::default();
        let prompt = build_chapter_prompt(&hg, &plan.narrative_id, 0, &style, &[]).unwrap();

        assert!(!prompt.system_prompt.is_empty());
        assert!(!prompt.situation_specs.is_empty());
        // Should have character contexts for participants
        let spec = &prompt.situation_specs[0];
        assert!(!spec.character_contexts.is_empty());
    }

    #[test]
    fn test_render_prompt() {
        let prompt = GenerationPrompt {
            system_prompt: "Write in the style of Hemingway.".into(),
            chapter_context: "Previously: Alice met Bob.".into(),
            situation_specs: vec![SituationPrompt {
                situation_id: Uuid::nil(),
                summary: "The confrontation in the library.".into(),
                character_contexts: vec![CharacterContext {
                    entity_id: Uuid::nil(),
                    name: "Alice".into(),
                    knows: vec!["the secret".into()],
                    false_beliefs: vec![],
                    motivation: "Seeks truth".into(),
                    relationships: vec!["adversary: Bob".into()],
                    arc_phase: "pre-midpoint".into(),
                }],
                scene_type_instruction: "Action scene with conflict.".into(),
                narration_mode_instruction: String::new(),
                commitment_instructions: vec!["Plant the revolver.".into()],
                dramatic_irony_instructions: vec!["Reader knows Bob is lying.".into()],
                causal_context: "Follows from: the discovery.".into(),
            }],
            constraints: vec![],
        };

        let rendered = render_prompt(&prompt);
        assert!(rendered.contains("Hemingway"));
        assert!(rendered.contains("Alice"));
        assert!(rendered.contains("the secret"));
        assert!(rendered.contains("Plant the revolver"));
        assert!(rendered.contains("Bob is lying"));
    }

    #[test]
    fn test_style_prompt_includes_voice() {
        let hg = test_hg();
        let style = StyleTarget {
            voice_description: Some("Sparse, declarative sentences. No adverbs.".into()),
            ..StyleTarget::default()
        };
        let prompt = build_style_prompt(&hg, &style).unwrap();
        assert!(prompt.contains("Sparse, declarative"));
    }

    fn make_author_entity(hg: &Hypergraph, name: &str) -> Uuid {
        use crate::types::{Entity, EntityType, MaturityLevel};
        let id = Uuid::now_v7();
        let entity = Entity {
            id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            maturity: MaturityLevel::Validated,
            confidence: 1.0,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            narrative_id: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap()
    }

    fn make_se_record(
        hg: &Hypergraph,
        source: crate::style::embedding::StyleEmbeddingSource,
    ) -> Uuid {
        use crate::style::embedding::{store_embedding, StyleEmbedding};
        let emb = StyleEmbedding {
            id: Uuid::now_v7(),
            vector: vec![0.0; 4],
            source,
            base_model: "test-model".into(),
            training_corpus_size: 100,
            created_at: chrono::Utc::now(),
        };
        let id = emb.id;
        store_embedding(hg, &emb).unwrap();
        id
    }

    #[test]
    fn test_style_prompt_renders_se_single_author() {
        use crate::style::embedding::StyleEmbeddingSource;
        let hg = test_hg();
        let author_id = make_author_entity(&hg, "Hemingway");
        let emb_id = make_se_record(&hg, StyleEmbeddingSource::SingleAuthor { author_id });
        let style = StyleTarget {
            style_embedding_id: Some(emb_id),
            ..StyleTarget::default()
        };
        let prompt = build_style_prompt(&hg, &style).unwrap();
        assert!(
            prompt.contains("voice of Hemingway"),
            "expected voice fragment, got: {prompt}"
        );
    }

    #[test]
    fn test_style_prompt_renders_se_blended() {
        use crate::style::embedding::StyleEmbeddingSource;
        let hg = test_hg();
        let a = make_author_entity(&hg, "Murakami");
        let b = make_author_entity(&hg, "Borges");
        let emb_id = make_se_record(
            &hg,
            StyleEmbeddingSource::Blended {
                sources: vec![(a, 0.6), (b, 0.4)],
            },
        );
        let style = StyleTarget {
            style_embedding_id: Some(emb_id),
            ..StyleTarget::default()
        };
        let prompt = build_style_prompt(&hg, &style).unwrap();
        assert!(prompt.contains("Murakami"), "missing author A: {prompt}");
        assert!(prompt.contains("Borges"), "missing author B: {prompt}");
        assert!(prompt.contains("60%"), "missing weight A: {prompt}");
        assert!(prompt.contains("40%"), "missing weight B: {prompt}");
    }

    #[test]
    fn test_style_prompt_handles_missing_se_gracefully() {
        let hg = test_hg();
        let nonexistent = Uuid::now_v7();
        let style = StyleTarget {
            style_embedding_id: Some(nonexistent),
            voice_description: Some("fallback voice".into()),
            ..StyleTarget::default()
        };
        let prompt = build_style_prompt(&hg, &style).unwrap();
        // Should still build a prompt; falls back silently to voice_description.
        assert!(prompt.contains("fallback voice"));
        // No "voice of" SE fragment because the SE record was missing.
        assert!(!prompt.contains("voice of"));
    }
}
