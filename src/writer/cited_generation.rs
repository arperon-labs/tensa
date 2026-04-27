//! Sprint W11 — source-backed generation + contention-aware drafts.
//!
//! Closes the loop between research (W9) + fact-check (W10) + generation.
//! When the writer asks TENSA to draft prose, we:
//!   1. Assemble the scene research context.
//!   2. Flag any open contentions on participants so the writer picks a stance.
//!   3. Inject research + stance into the generation prompt, instructing the
//!      model to emit `[[cite: source_id]]` inline markers.
//!   4. Parse the generated text into `GeneratedSpan { text, citations }`.
//!   5. Run the fact-check engine over the result as a hallucination guard.
//!
//! This module does *not* perform the LLM call itself — existing generation
//! plumbing in `narrative::generation` stays in charge. The helpers here
//! produce the prompt addendum, parse citations out of the LLM response, and
//! run the post-commit guard.

use std::collections::HashSet;

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use uuid::Uuid;

use crate::error::Result;
use crate::writer::factcheck::{run_factcheck, FactCheckReport, FactCheckTier, VerdictStatus};
use crate::writer::research::{build_scene_research_context, SceneResearchContext};
use crate::Hypergraph;

/// Writer's chosen stance for a pending contention. Carried in the
/// generation request so the prompt can pick a side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionStance {
    pub contention_id: Uuid,
    /// The source the writer chose to believe.
    pub chosen_source_id: Option<Uuid>,
    /// Optional free-text note — e.g. "treating the Telegraph account as canonical".
    #[serde(default)]
    pub note: Option<String>,
}

/// An unresolved contention surfaced to the caller before generation runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingContention {
    pub situation_a: Uuid,
    pub situation_b: Uuid,
    pub contention_type: String,
    pub description: Option<String>,
}

/// Request wrapper; carries generation config + stance resolutions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitedGenerationRequest {
    pub situation_id: Uuid,
    #[serde(default)]
    pub require_citations: bool,
    /// Optional resolutions for open contentions on this scene. If absent,
    /// `build_research_prompt` returns `pending_contentions` and the caller
    /// should halt until the writer supplies stances.
    #[serde(default)]
    pub stances: Vec<ContentionStance>,
}

/// Response from `build_research_prompt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchPromptAddendum {
    pub situation_id: Uuid,
    /// Markdown-ready block to splice into the generation system prompt.
    pub system_addendum: String,
    /// Contentions the writer must take a stance on before generating.
    /// If non-empty and `require_stance_resolution` is true, callers should
    /// return it to the UI rather than proceed to the LLM.
    pub pending_contentions: Vec<PendingContention>,
}

/// A parsed generated prose span with extracted citation markers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSpan {
    pub text: String,
    /// Byte offsets in the cleaned (marker-stripped) text.
    pub span: (usize, usize),
    pub citations: Vec<Uuid>,
}

/// Result of parsing an LLM response into cited spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedCitedText {
    /// Text with `[[cite:...]]` markers stripped.
    pub clean_text: String,
    /// One entry per sentence/paragraph with the citations that applied to it.
    pub spans: Vec<GeneratedSpan>,
    /// All unique source ids cited across the output.
    pub unique_citations: Vec<Uuid>,
}

/// Sprint W11 step 1 — fold research context into a prompt addendum.
///
/// The addendum is markdown, suitable for concatenation into an existing
/// generation system prompt. When `require_stance_resolution` is true and
/// open contentions touch the scene, `pending_contentions` is populated and
/// the caller should surface the stance picker before calling the LLM.
pub fn build_research_prompt(
    hypergraph: &Hypergraph,
    req: &CitedGenerationRequest,
) -> Result<ResearchPromptAddendum> {
    let ctx = build_scene_research_context(hypergraph, &req.situation_id)?;
    let mut pending = Vec::new();
    for c in &ctx.contentions {
        if c.resolved {
            continue;
        }
        let already_resolved = req.stances.iter().any(|s| s.contention_id == c.situation_a);
        if !already_resolved {
            pending.push(PendingContention {
                situation_a: c.situation_a,
                situation_b: c.situation_b,
                contention_type: format!("{:?}", c.contention_type),
                description: c.description.clone(),
            });
        }
    }

    let system_addendum = render_addendum(&ctx, req.require_citations, &req.stances);

    Ok(ResearchPromptAddendum {
        situation_id: req.situation_id,
        system_addendum,
        pending_contentions: pending,
    })
}

fn render_addendum(
    ctx: &SceneResearchContext,
    require_citations: bool,
    stances: &[ContentionStance],
) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("\n\n### Research context for this scene\n");
    out.push_str("You MUST treat the following as authoritative:\n\n");

    if !ctx.pinned_facts.is_empty() {
        out.push_str("**Pinned facts (do not contradict):**\n");
        for f in &ctx.pinned_facts {
            out.push_str(&format!("- {}: {}", f.key, f.value));
            if let Some(n) = &f.note {
                out.push_str(&format!(" ({})", n));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    let all_sources: Vec<_> = ctx
        .scene_sources
        .iter()
        .chain(ctx.participant_sources.iter().map(|p| &p.attribution))
        .collect();
    if !all_sources.is_empty() {
        out.push_str("**Sources (cite when stating a factual claim):**\n");
        let mut seen: HashSet<Uuid> = HashSet::new();
        for a in &all_sources {
            if !seen.insert(a.source_id) {
                continue;
            }
            out.push_str(&format!("- source_id={}", a.source_id));
            if let Some(e) = &a.excerpt {
                let trimmed = if e.len() > 180 { &e[..180] } else { e };
                out.push_str(&format!(" — \"{}\"", trimmed));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    if !ctx.notes.is_empty() {
        out.push_str("**Writer research notes:**\n");
        for n in &ctx.notes {
            let snippet = if n.body.len() > 200 {
                &n.body[..200]
            } else {
                &n.body
            };
            out.push_str(&format!("- [{:?}] {}\n", n.kind, snippet));
        }
        out.push('\n');
    }

    if !stances.is_empty() {
        out.push_str("**Writer-chosen stance on contentions:**\n");
        for s in stances {
            match s.chosen_source_id {
                Some(id) => out.push_str(&format!(
                    "- Contention {}: treat source {} as canonical.",
                    s.contention_id, id
                )),
                None => out.push_str(&format!(
                    "- Contention {}: writer declines to resolve; keep prose neutral.",
                    s.contention_id
                )),
            }
            if let Some(note) = &s.note {
                out.push_str(&format!(" ({})", note));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    if require_citations {
        out.push_str(
            "**Citation requirement:** every factual statement that derives from a \
             listed source must be followed by an inline marker in the form \
             `[[cite: <source_id>]]`. If no listed source supports a specific claim, \
             rephrase it as hedged speculation (e.g. \"it is said that…\") or omit it.\n",
        );
    }
    out
}

/// Parse `[[cite: uuid]]` markers out of an LLM output string.
///
/// The parser produces one `GeneratedSpan` per sentence or newline-delimited
/// block and attaches to it every citation marker that appeared *within that
/// span*. Markers are then stripped, leaving `clean_text` as publishable prose.
pub fn parse_cited_text(raw: &str) -> ParsedCitedText {
    static CITE_RE: OnceLock<Regex> = OnceLock::new();
    let re = CITE_RE.get_or_init(|| Regex::new(r"\[\[cite:\s*([0-9a-fA-F\-]{36})\s*\]\]").unwrap());

    let mut clean = String::with_capacity(raw.len());
    let mut spans: Vec<GeneratedSpan> = Vec::new();
    let mut all_citations: HashSet<Uuid> = HashSet::new();

    // Split by sentence terminators / paragraph breaks. Preserve delimiters.
    let blocks = crate::narrative::writer_common::split_sentences(raw);
    for raw_block in blocks {
        let mut block_citations: Vec<Uuid> = Vec::new();
        let mut block_clean = String::with_capacity(raw_block.len());
        let mut cursor = 0;
        for caps in re.captures_iter(raw_block) {
            let m = caps.get(0).unwrap();
            block_clean.push_str(&raw_block[cursor..m.start()]);
            if let Ok(id) = Uuid::parse_str(&caps[1]) {
                block_citations.push(id);
                all_citations.insert(id);
            }
            cursor = m.end();
        }
        block_clean.push_str(&raw_block[cursor..]);
        let block_clean_trimmed = block_clean.trim();
        if block_clean_trimmed.is_empty() {
            // keep any trailing whitespace for flow, but don't record a span
            clean.push_str(&block_clean);
            continue;
        }
        let start = clean.len();
        clean.push_str(&block_clean);
        let end = clean.len();
        spans.push(GeneratedSpan {
            text: block_clean_trimmed.to_string(),
            span: (start, end),
            citations: block_citations,
        });
    }

    ParsedCitedText {
        clean_text: clean,
        spans,
        unique_citations: all_citations.into_iter().collect(),
    }
}

/// Sprint W11 step 5 — hallucination guard. Run the W10 fact-checker over
/// the generated clean text and return any `Contradicted` verdicts. Callers
/// should either block the commit or require the writer to acknowledge each
/// contradicted claim.
pub fn hallucination_guard(
    hypergraph: &Hypergraph,
    situation_id: &Uuid,
    clean_text: &str,
) -> Result<HallucinationReport> {
    let report = run_factcheck(
        hypergraph,
        situation_id,
        clean_text,
        FactCheckTier::Standard,
    )?;
    let blocking: Vec<_> = report
        .verdicts
        .iter()
        .filter(|v| v.status == VerdictStatus::Contradicted)
        .cloned()
        .collect();
    Ok(HallucinationReport {
        blocking,
        full: report,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationReport {
    pub blocking: Vec<crate::writer::factcheck::ClaimVerdict>,
    pub full: FactCheckReport,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, PinnedFact,
        Situation, TimeGranularity,
    };
    use chrono::Utc;
    use std::sync::Arc;

    fn setup() -> (Hypergraph, Uuid, String) {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let sit = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text("body")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some("n1".into()),
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
        };
        let id = sit.id;
        hg.create_situation(sit).unwrap();
        (hg, id, "n1".into())
    }

    #[test]
    fn test_parse_citations_extracts_and_strips_markers() {
        let raw = "The king died in 1847[[cite: 00000000-0000-0000-0000-000000000001]]. \
                   His cousin succeeded[[cite: 00000000-0000-0000-0000-000000000002]].";
        let parsed = parse_cited_text(raw);
        assert!(!parsed.clean_text.contains("[[cite"));
        assert_eq!(parsed.unique_citations.len(), 2);
        assert_eq!(parsed.spans.len(), 2);
        assert_eq!(parsed.spans[0].citations.len(), 1);
    }

    #[test]
    fn test_parse_citations_handles_no_markers() {
        let parsed = parse_cited_text("Plain prose with nothing cited.");
        assert!(parsed.unique_citations.is_empty());
        assert_eq!(parsed.spans.len(), 1);
        assert_eq!(parsed.clean_text, "Plain prose with nothing cited.");
    }

    #[test]
    fn test_build_research_prompt_surfaces_no_pending_when_clear() {
        let (hg, sid, _) = setup();
        let add = build_research_prompt(
            &hg,
            &CitedGenerationRequest {
                situation_id: sid,
                require_citations: true,
                stances: vec![],
            },
        )
        .unwrap();
        assert!(add.pending_contentions.is_empty());
        assert!(add.system_addendum.contains("Citation requirement"));
    }

    #[test]
    fn test_prompt_includes_pinned_facts() {
        let (hg, sid, nid) = setup();
        crate::narrative::continuity::create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::now_v7(),
                narrative_id: nid,
                entity_id: None,
                key: "weather".into(),
                value: "raining".into(),
                note: Some("established in prologue".into()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();
        let add = build_research_prompt(
            &hg,
            &CitedGenerationRequest {
                situation_id: sid,
                require_citations: false,
                stances: vec![],
            },
        )
        .unwrap();
        assert!(add.system_addendum.contains("weather: raining"));
        assert!(add.system_addendum.contains("established in prologue"));
    }

    #[test]
    fn test_hallucination_guard_blocks_contradicted() {
        let (hg, sid, nid) = setup();
        crate::narrative::continuity::create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::now_v7(),
                narrative_id: nid,
                entity_id: None,
                key: "soldiers".into(),
                value: "500".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();
        let report = hallucination_guard(&hg, &sid, "Only 200 soldiers remained.").unwrap();
        assert!(!report.blocking.is_empty());
    }
}
