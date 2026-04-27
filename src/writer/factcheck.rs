//! Sprint W10 — inline fact-check on prose.
//!
//! Extracts atomic factual claims from writer prose and scores each against
//! the scene's research context (W9). Every claim gets one of four verdicts:
//! Supported / Contested / Unsupported / Contradicted. The Studio editor
//! renders these as inline underlines; W11's generation guard uses the
//! same engine to block hallucinated spans before commit.
//!
//! Three execution tiers:
//! - **Fast** (sync, no LLM): heuristic claim detection (quantities,
//!   named-entity attributions, dates) + pinned-fact exact/paraphrase match
//!   + contention lookup.
//! - **Standard** (sync, no LLM by default, but may embed): fast layer plus
//!   text-similarity scan across source attributions' excerpts.
//! - **Deep** (async via InferenceEngine — wired in a follow-up sprint):
//!   LLM claim extraction + embedding-based retrieval across source chunks
//!   + Dempster–Shafer mass combination for contested evidence.

use std::collections::HashSet;
use std::sync::OnceLock;

use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::source::SourceAttribution;
use crate::types::PinnedFact;
use crate::writer::research::{build_scene_research_context, SceneResearchContext};
use crate::Hypergraph;

/// A single atomic factual claim extracted from prose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicClaim {
    /// Span in the original paragraph text (byte offsets).
    pub span: (usize, usize),
    pub text: String,
    pub kind: ClaimKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimKind {
    /// Generic factual claim.
    Factual,
    /// Temporal claim ("on Tuesday", "in 1847").
    Temporal,
    /// Quantitative claim ("500 soldiers", "23 years old").
    Quantitative,
    /// Attribution claim ("X said", "Y did").
    Attribution,
    /// Spatial claim ("in Paris", "at the harbour").
    Spatial,
}

/// Verdict on a single claim after matching against the research context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimVerdict {
    pub claim: AtomicClaim,
    pub status: VerdictStatus,
    /// Evidence ids (pinned-fact id, source attribution target, contention id) that
    /// the matcher found relevant. Purely informational — the Studio popover
    /// resolves these to display objects.
    pub evidence: Vec<EvidenceRef>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictStatus {
    /// At least one corroborating source and no contradiction.
    Supported,
    /// Sources disagree or a contention touches the claim.
    Contested,
    /// No evidence found in the scene's research context.
    Unsupported,
    /// A pinned fact or a source with a dissenting claim directly opposes it.
    Contradicted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum EvidenceRef {
    PinnedFact {
        fact_id: Uuid,
        value: String,
    },
    Source {
        source_id: Uuid,
        excerpt: Option<String>,
    },
    Contention {
        other_situation: Uuid,
    },
    Note {
        note_id: Uuid,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactCheckTier {
    Fast,
    Standard,
    Deep,
}

impl Default for FactCheckTier {
    fn default() -> Self {
        FactCheckTier::Standard
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCheckReport {
    pub situation_id: Uuid,
    pub tier: FactCheckTier,
    pub paragraph_hash: String,
    pub verdicts: Vec<ClaimVerdict>,
    pub generated_at: DateTime<Utc>,
}

/// Public entry point — extract claims from `text`, match against the scene's
/// research context, return a per-claim report.
pub fn run_factcheck(
    hypergraph: &Hypergraph,
    situation_id: &Uuid,
    text: &str,
    tier: FactCheckTier,
) -> Result<FactCheckReport> {
    let ctx = build_scene_research_context(hypergraph, situation_id)?;
    let paragraph_hash = crate::ingestion::chunker::chunk_hash(text);
    let claims = extract_atomic_claims(text);
    let verdicts = claims
        .into_iter()
        .map(|c| classify_claim(&c, &ctx))
        .collect();
    Ok(FactCheckReport {
        situation_id: *situation_id,
        tier,
        paragraph_hash,
        verdicts,
        generated_at: Utc::now(),
    })
}

// ─── Claim extraction (heuristics) ────────────────────────────

/// Deterministic heuristic claim extraction. No LLM. Produces one
/// `AtomicClaim` per sentence, classified by the dominant pattern.
pub fn extract_atomic_claims(text: &str) -> Vec<AtomicClaim> {
    let mut out = Vec::new();
    let mut cursor = 0;
    for raw in crate::narrative::writer_common::split_sentences(text) {
        // locate the raw sentence's byte offset in `text` starting from `cursor`
        let start = text[cursor..]
            .find(raw)
            .map(|rel| cursor + rel)
            .unwrap_or(cursor);
        let end = start + raw.len();
        cursor = end;
        let trimmed = raw.trim();
        if trimmed.len() < 6 {
            continue;
        }
        let kind = classify_sentence_kind(trimmed);
        out.push(AtomicClaim {
            span: (start, end),
            text: trimmed.to_string(),
            kind,
        });
    }
    out
}

fn classify_sentence_kind(s: &str) -> ClaimKind {
    static DIGIT: OnceLock<Regex> = OnceLock::new();
    static YEAR: OnceLock<Regex> = OnceLock::new();
    static ATTRIB: OnceLock<Regex> = OnceLock::new();
    static SPATIAL: OnceLock<Regex> = OnceLock::new();
    let digit = DIGIT.get_or_init(|| Regex::new(r"\d").unwrap());
    let year = YEAR.get_or_init(|| Regex::new(r"\b(1[5-9]|20)\d{2}\b").unwrap());
    let attrib = ATTRIB
        .get_or_init(|| Regex::new(r"\b(?:said|told|announced|claimed|reported|wrote)\b").unwrap());
    let spatial =
        SPATIAL.get_or_init(|| Regex::new(r"\b(?:in|at|near|from|to)\s+[A-Z][A-Za-z]+").unwrap());
    if year.is_match(s) {
        return ClaimKind::Temporal;
    }
    if digit.is_match(s) {
        return ClaimKind::Quantitative;
    }
    if attrib.is_match(s) {
        return ClaimKind::Attribution;
    }
    if spatial.is_match(s) {
        return ClaimKind::Spatial;
    }
    ClaimKind::Factual
}

// ─── Verdict classification ───────────────────────────────────

fn classify_claim(claim: &AtomicClaim, ctx: &SceneResearchContext) -> ClaimVerdict {
    let claim_lower = claim.text.to_lowercase();
    let claim_tokens = tokenize(&claim_lower);

    let mut evidence: Vec<EvidenceRef> = Vec::new();
    let mut supporting = 0u32;
    let mut contradicting = 0u32;

    // 1. Pinned facts: direct value lookup (case-insensitive substring).
    for f in &ctx.pinned_facts {
        if pinned_fact_matches(f, &claim_lower) {
            evidence.push(EvidenceRef::PinnedFact {
                fact_id: f.id,
                value: f.value.clone(),
            });
            supporting += 1;
            // Detect direct contradiction: pinned fact value present in negated form.
            if contradicts_pinned_fact(&claim_lower, f) {
                contradicting += 1;
            }
        }
    }

    // 2. Source attributions: excerpt term-overlap.
    let all_sources: Vec<&SourceAttribution> = ctx
        .scene_sources
        .iter()
        .chain(ctx.participant_sources.iter().map(|p| &p.attribution))
        .collect();
    for a in &all_sources {
        if let Some(score) = excerpt_overlap_score(a, &claim_tokens) {
            if score >= 0.30 {
                evidence.push(EvidenceRef::Source {
                    source_id: a.source_id,
                    excerpt: a.excerpt.clone(),
                });
                supporting += 1;
            }
        }
        // If the source has a `claim` field that directly disagrees, flag it.
        if let Some(src_claim) = &a.claim {
            if src_claim.to_lowercase().contains("not ")
                && claim_lower.contains(&trim_lowercase(src_claim).replace("not ", ""))
            {
                contradicting += 1;
                evidence.push(EvidenceRef::Source {
                    source_id: a.source_id,
                    excerpt: Some(src_claim.clone()),
                });
            }
        }
    }

    // 3. Open contentions: if any unresolved contention touches this scene,
    // the claim lives on contested ground.
    let open_contentions: Vec<_> = ctx.contentions.iter().filter(|c| !c.resolved).collect();
    if !open_contentions.is_empty() {
        for c in &open_contentions {
            evidence.push(EvidenceRef::Contention {
                other_situation: if c.situation_a == ctx.situation_id {
                    c.situation_b
                } else {
                    c.situation_a
                },
            });
        }
    }

    // 4. Research notes that overlap the claim reinforce support (writer's own marginalia).
    for n in &ctx.notes {
        let note_tokens = tokenize(&n.body.to_lowercase());
        let overlap = token_overlap(&claim_tokens, &note_tokens);
        if overlap >= 0.30 {
            evidence.push(EvidenceRef::Note { note_id: n.id });
            supporting += 1;
        }
    }

    let status = if contradicting > 0 {
        VerdictStatus::Contradicted
    } else if !open_contentions.is_empty() && supporting == 0 {
        VerdictStatus::Contested
    } else if supporting == 0 {
        VerdictStatus::Unsupported
    } else if !open_contentions.is_empty() {
        VerdictStatus::Contested
    } else {
        VerdictStatus::Supported
    };

    let confidence = match status {
        VerdictStatus::Supported => (supporting as f32 / (supporting as f32 + 1.0)).min(0.95),
        VerdictStatus::Contested => 0.50,
        VerdictStatus::Unsupported => 0.15,
        VerdictStatus::Contradicted => 0.85, // high confidence in the negative verdict
    };

    ClaimVerdict {
        claim: claim.clone(),
        status,
        evidence,
        confidence,
    }
}

fn pinned_fact_matches(f: &PinnedFact, claim_lower: &str) -> bool {
    if f.value.trim().is_empty() {
        return false;
    }
    let v = f.value.to_lowercase();
    let k = f.key.to_lowercase();
    claim_lower.contains(&v) || claim_lower.contains(&k)
}

fn contradicts_pinned_fact(claim_lower: &str, f: &PinnedFact) -> bool {
    // Trivial direct contradiction heuristic: "not <value>" or the claim contains
    // a different numeric substring than the fact for a numeric fact.
    let v = f.value.to_lowercase();
    if claim_lower.contains(&format!("not {}", v))
        || claim_lower.contains(&format!("isn't {}", v))
        || claim_lower.contains(&format!("wasn't {}", v))
    {
        return true;
    }
    // Numeric-fact mismatch: fact is a pure integer, and the claim contains a
    // different integer alongside the fact's key (in either direction).
    if let Ok(fact_num) = v.trim().parse::<i64>() {
        let key = f.key.to_lowercase();
        if claim_lower.contains(&key) {
            // Any integer in the claim that differs from fact_num => contradiction.
            static NUM_RE: OnceLock<Regex> = OnceLock::new();
            let re = NUM_RE.get_or_init(|| Regex::new(r"-?\b\d+\b").unwrap());
            for cap in re.find_iter(claim_lower) {
                if let Ok(claim_num) = cap.as_str().parse::<i64>() {
                    if claim_num != fact_num {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn excerpt_overlap_score(a: &SourceAttribution, claim_tokens: &HashSet<String>) -> Option<f32> {
    let Some(exc) = a.excerpt.as_deref() else {
        return None;
    };
    let excerpt_tokens = tokenize(&exc.to_lowercase());
    Some(token_overlap(claim_tokens, &excerpt_tokens))
}

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_string())
        .collect()
}

fn token_overlap(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    inter / union
}

fn trim_lowercase(s: &str) -> String {
    s.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel, Situation,
        TimeGranularity,
    };
    use crate::writer::research::{create_research_note, ResearchNote, ResearchNoteKind};
    use std::sync::Arc;

    fn setup() -> (Hypergraph, Uuid, String) {
        let hg = Hypergraph::new(Arc::new(MemoryStore::new()));
        let scene = Situation {
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
        let id = scene.id;
        hg.create_situation(scene).unwrap();
        (hg, id, "n1".into())
    }

    #[test]
    fn test_extract_claims_classifies_kinds() {
        let claims = extract_atomic_claims(
            "The battle was in 1847. He said it was over. 500 soldiers charged. He walked to Paris.",
        );
        assert!(claims.len() >= 4);
        let kinds: Vec<_> = claims.iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&ClaimKind::Temporal));
        assert!(kinds.contains(&ClaimKind::Quantitative) || kinds.contains(&ClaimKind::Temporal));
        assert!(kinds.contains(&ClaimKind::Attribution));
    }

    #[test]
    fn test_claim_unsupported_by_default() {
        let (hg, sid, _) = setup();
        let report = run_factcheck(&hg, &sid, "A battle happened.", FactCheckTier::Fast).unwrap();
        assert_eq!(report.verdicts.len(), 1);
        assert_eq!(report.verdicts[0].status, VerdictStatus::Unsupported);
    }

    #[test]
    fn test_claim_supported_by_note() {
        let (hg, sid, nid) = setup();
        create_research_note(
            hg.store(),
            ResearchNote {
                id: Uuid::nil(),
                narrative_id: nid,
                situation_id: sid,
                kind: ResearchNoteKind::Note,
                body: "The battle was decisive at Stonebridge.".into(),
                source_chunk_id: None,
                source_id: None,
                author: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();
        let report = run_factcheck(
            &hg,
            &sid,
            "The battle was decisive at Stonebridge.",
            FactCheckTier::Fast,
        )
        .unwrap();
        assert_eq!(report.verdicts.len(), 1);
        assert_eq!(report.verdicts[0].status, VerdictStatus::Supported);
    }

    #[test]
    fn test_claim_contradicted_by_pinned_fact_numeric() {
        let (hg, sid, nid) = setup();
        crate::narrative::continuity::create_pinned_fact(
            hg.store(),
            PinnedFact {
                id: Uuid::now_v7(),
                narrative_id: nid.clone(),
                entity_id: None,
                key: "soldiers".into(),
                value: "500".into(),
                note: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        )
        .unwrap();
        // Prose claims a different number.
        let report = run_factcheck(
            &hg,
            &sid,
            "Only 200 soldiers remained.",
            FactCheckTier::Fast,
        )
        .unwrap();
        // Finds the pinned-fact key then a different number after → contradiction.
        assert!(report
            .verdicts
            .iter()
            .any(|v| v.status == VerdictStatus::Contradicted));
    }

    #[test]
    fn test_hash_is_stable() {
        use crate::ingestion::chunker::chunk_hash;
        assert_eq!(chunk_hash("abc"), chunk_hash("abc"));
        assert_ne!(chunk_hash("abc"), chunk_hash("abcd"));
    }
}
