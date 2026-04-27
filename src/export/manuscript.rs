//! Manuscript export — reconstructs narrative prose from raw content blocks.
//!
//! Sorts situations temporally, groups them by narrative level into chapters,
//! and renders each `ContentBlock` as formatted Markdown.

use crate::error::Result;
use crate::export::{situation_preview, sort_situations_by_time, NarrativeExport};
use crate::types::{ContentType, NarrativeLevel, Situation};

/// A chapter grouping for manuscript rendering.
struct Chapter<'a> {
    title: String,
    situations: Vec<&'a Situation>,
}

/// Export narrative data as a Markdown manuscript.
pub fn export_manuscript(data: &NarrativeExport) -> Result<String> {
    let sorted = sort_situations_by_time(&data.situations);
    let chapters = group_into_chapters(&sorted);

    let mut out = String::with_capacity(sorted.len() * 200);

    // Title from narrative_id
    out.push_str(&format!("# {}\n\n", titlecase(&data.narrative_id)));

    if chapters.is_empty() {
        out.push_str("*No situations in this narrative.*\n");
        return Ok(out);
    }

    for (i, chapter) in chapters.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("## {}\n\n", chapter.title));

        for (j, sit) in chapter.situations.iter().enumerate() {
            // Scene-level sub-headings within a chapter
            if sit.narrative_level == NarrativeLevel::Scene {
                let desc = first_content_preview(sit);
                out.push_str(&format!("### {}\n\n", desc));
            }

            // Separator between beats/events (not before the first)
            if j > 0
                && matches!(
                    sit.narrative_level,
                    NarrativeLevel::Beat | NarrativeLevel::Event
                )
            {
                out.push_str("---\n\n");
            }

            // Render all content blocks
            for block in &sit.raw_content {
                out.push_str(&render_content_block(block));
                out.push_str("\n\n");
            }
        }
    }

    Ok(out.trim_end().to_string() + "\n")
}

/// Group sorted situations into chapters based on narrative level.
/// Arc and Sequence boundaries start new chapters. Story-level situations
/// form their own chapter. Scene/Beat/Event are grouped under the current chapter.
fn group_into_chapters<'a>(situations: &[&'a Situation]) -> Vec<Chapter<'a>> {
    let mut chapters: Vec<Chapter<'a>> = Vec::new();
    let mut chapter_counter = 0usize;

    for &sit in situations {
        match sit.narrative_level {
            NarrativeLevel::Story | NarrativeLevel::Arc | NarrativeLevel::Sequence => {
                chapter_counter += 1;
                let title = chapter_title(sit, chapter_counter);
                chapters.push(Chapter {
                    title,
                    situations: vec![sit],
                });
            }
            NarrativeLevel::Scene | NarrativeLevel::Beat | NarrativeLevel::Event => {
                if chapters.is_empty() {
                    chapter_counter += 1;
                    chapters.push(Chapter {
                        title: format!("Chapter {}", chapter_counter),
                        situations: Vec::new(),
                    });
                }
                if let Some(chapter) = chapters.last_mut() {
                    chapter.situations.push(sit);
                }
            }
        }
    }

    chapters
}

/// Derive a chapter title from a situation's content or level.
fn chapter_title(sit: &Situation, counter: usize) -> String {
    let preview = first_content_preview(sit);
    if preview.len() > 3 {
        preview
    } else {
        format!("Chapter {}", counter)
    }
}

/// Extract a short preview from the first content block of a situation.
fn first_content_preview(sit: &Situation) -> String {
    situation_preview(sit, 57, String::new)
}

/// Render a single ContentBlock to Markdown.
fn render_content_block(block: &crate::types::ContentBlock) -> String {
    let content = block.content.trim();
    match block.content_type {
        ContentType::Text => content.to_string(),
        ContentType::Dialogue => content
            .lines()
            .map(|line| format!("> {}", line))
            .collect::<Vec<_>>()
            .join("\n"),
        ContentType::Observation => format!("*{}*", content),
        ContentType::Document => format!("```\n{}\n```", content),
        ContentType::MediaRef => format!("![media]({})", content),
    }
}

/// Export original source text from stored chunks as a Markdown manuscript.
///
/// Groups chunks by chapter, renders text in order, strips overlap.
pub fn export_manuscript_from_chunks(chunks: &[crate::types::ChunkRecord]) -> Result<String> {
    if chunks.is_empty() {
        return Ok("*No source text available.*\n".to_string());
    }

    let mut out = String::with_capacity(chunks.len() * 500);

    // Title from narrative_id (if available)
    if let Some(ref nid) = chunks[0].narrative_id {
        out.push_str(&format!("# {} (Source)\n\n", titlecase(nid)));
    } else {
        out.push_str("# Source Text\n\n");
    }

    let mut current_chapter: Option<&str> = None;
    for (i, chunk) in chunks.iter().enumerate() {
        // Chapter headings from chunk metadata
        if let Some(ref ch) = chunk.chapter {
            if current_chapter != Some(ch.as_str()) {
                if i > 0 {
                    out.push('\n');
                }
                out.push_str(&format!("## {}\n\n", ch));
                current_chapter = Some(ch.as_str());
            }
        }

        // Strip overlap from consecutive chunks
        let text = chunk.text_without_overlap(i);

        out.push_str(text.trim());
        out.push_str("\n\n");
    }

    Ok(out.trim_end().to_string() + "\n")
}

/// Convert a kebab-case or snake_case narrative id to title case.
fn titlecase(s: &str) -> String {
    s.split(['-', '_'])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    format!("{}{}", upper, chars.as_str())
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::{TimeZone, Utc};
    use uuid::Uuid;

    fn make_situation(
        level: NarrativeLevel,
        content_blocks: Vec<ContentBlock>,
        start: Option<chrono::DateTime<Utc>>,
    ) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start,
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
            raw_content: content_blocks,
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: Some("test".into()),
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
        }
    }

    fn make_export(situations: Vec<Situation>) -> NarrativeExport {
        NarrativeExport {
            narrative_id: "test-narrative".into(),
            entities: vec![],
            situations,
            participations: vec![],
            causal_links: vec![],
        }
    }

    #[test]
    fn test_manuscript_empty_narrative() {
        let data = make_export(vec![]);
        let md = export_manuscript(&data).unwrap();
        assert!(md.contains("# Test Narrative"));
        assert!(md.contains("No situations"));
    }

    #[test]
    fn test_manuscript_single_scene() {
        let sit = make_situation(
            NarrativeLevel::Scene,
            vec![ContentBlock::text("The rain fell softly.")],
            None,
        );
        let data = make_export(vec![sit]);
        let md = export_manuscript(&data).unwrap();
        assert!(md.contains("The rain fell softly."));
    }

    #[test]
    fn test_manuscript_dialogue_formatting() {
        let sit = make_situation(
            NarrativeLevel::Scene,
            vec![ContentBlock {
                content_type: ContentType::Dialogue,
                content: "Hello there.\nGeneral Kenobi.".into(),
                source: None,
            }],
            None,
        );
        let data = make_export(vec![sit]);
        let md = export_manuscript(&data).unwrap();
        assert!(md.contains("> Hello there."));
        assert!(md.contains("> General Kenobi."));
    }

    #[test]
    fn test_manuscript_temporal_ordering() {
        let t1 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let sit_late = make_situation(
            NarrativeLevel::Beat,
            vec![ContentBlock::text("Second event")],
            Some(t2),
        );
        let sit_early = make_situation(
            NarrativeLevel::Beat,
            vec![ContentBlock::text("First event")],
            Some(t1),
        );
        // Insert in reverse order
        let data = make_export(vec![sit_late, sit_early]);
        let md = export_manuscript(&data).unwrap();
        let pos_first = md.find("First event").unwrap();
        let pos_second = md.find("Second event").unwrap();
        assert!(
            pos_first < pos_second,
            "First event should appear before second"
        );
    }

    #[test]
    fn test_manuscript_chapter_grouping() {
        let t1 = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2024, 2, 1, 0, 0, 0).unwrap();
        let t3 = Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap();

        let arc1 = make_situation(
            NarrativeLevel::Arc,
            vec![ContentBlock::text("The Beginning")],
            Some(t1),
        );
        let scene1 = make_situation(
            NarrativeLevel::Scene,
            vec![ContentBlock::text("Opening scene")],
            Some(t2),
        );
        let arc2 = make_situation(
            NarrativeLevel::Arc,
            vec![ContentBlock::text("The Middle")],
            Some(t3),
        );
        let data = make_export(vec![arc1, scene1, arc2]);
        let md = export_manuscript(&data).unwrap();
        assert!(md.contains("## The Beginning"));
        assert!(md.contains("## The Middle"));
    }

    #[test]
    fn test_manuscript_mixed_content_types() {
        let sit = make_situation(
            NarrativeLevel::Scene,
            vec![
                ContentBlock::text("Narration here."),
                ContentBlock {
                    content_type: ContentType::Dialogue,
                    content: "Spoken words.".into(),
                    source: None,
                },
                ContentBlock {
                    content_type: ContentType::Observation,
                    content: "A quiet observation.".into(),
                    source: None,
                },
            ],
            None,
        );
        let data = make_export(vec![sit]);
        let md = export_manuscript(&data).unwrap();
        assert!(md.contains("Narration here."));
        assert!(md.contains("> Spoken words."));
        assert!(md.contains("*A quiet observation.*"));
    }

    #[test]
    fn test_manuscript_no_temporal_data() {
        let sit1 = make_situation(
            NarrativeLevel::Beat,
            vec![ContentBlock::text("No time A")],
            None,
        );
        let sit2 = make_situation(
            NarrativeLevel::Beat,
            vec![ContentBlock::text("No time B")],
            None,
        );
        let data = make_export(vec![sit1, sit2]);
        let md = export_manuscript(&data).unwrap();
        // Both should render even without temporal data
        assert!(md.contains("No time A"));
        assert!(md.contains("No time B"));
    }

    #[test]
    fn test_manuscript_multiple_content_blocks() {
        let sit = make_situation(
            NarrativeLevel::Event,
            vec![
                ContentBlock::text("Paragraph one."),
                ContentBlock::text("Paragraph two."),
                ContentBlock {
                    content_type: ContentType::Document,
                    content: "Evidence doc".into(),
                    source: None,
                },
            ],
            None,
        );
        let data = make_export(vec![sit]);
        let md = export_manuscript(&data).unwrap();
        assert!(md.contains("Paragraph one."));
        assert!(md.contains("Paragraph two."));
        assert!(md.contains("```\nEvidence doc\n```"));
    }
}
