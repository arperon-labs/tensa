//! Workshop — tiered manuscript analysis (Sprint W3, v0.49.2).
//!
//! One entry point produces a useful critique at the tier the writer chose
//! and priced up-front. Focuses are orthogonal (multi-select); tiers are
//! ordered from free (deterministic only) to expensive (full-LLM).
//!
//! # Tier semantics
//! - **Cheap** — deterministic only. Delegates to the existing `debug`
//!   pathology detector and a handful of simple per-chapter metrics. Zero
//!   LLM calls, runs in milliseconds.
//! - **Standard** — Cheap + selective LLM judgment on the highest-severity
//!   findings. Bounded by `max_llm_calls` (default 8) so cost is predictable.
//! - **Deep** — full-book LLM critique. **Not implemented in v0.49.2**;
//!   requesting it returns a "deferred" report with the Cheap-tier findings
//!   so the writer isn't empty-handed.
//!
//! # Simplify
//! - No new analysis algorithms in this module. Everything is orchestration
//!   over `debug::diagnose_narrative`, `export::manuscript` prose reconstruction,
//!   and the existing snapshot / LLM primitives.
//! - Findings persist at `wr/r/{report_id}` with narrative index at
//!   `wr/n/{narrative_id}/{report_id}` so the History-style "past reports"
//!   list is a cheap prefix scan.
//! - Suggested edits are stored as serialized [`EditOperation`] JSON so
//!   clicking "Apply suggestion" in the Studio simply pipes it to the W2
//!   EditPanel — no new edit path.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::ingestion::llm::{ApiMessage, SessionCapableExtractor};
use crate::narrative::debug::{
    diagnose_narrative, NarrativePathology, PathologyKind, PathologySeverity,
};
use crate::narrative::editing::{EditOperation, StyleTarget};
use crate::narrative::generation::extract_json_object;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::revision::gather_snapshot;
use crate::narrative::writer_common::{approx_tokens, count_words_blocks};
use crate::store::KVStore;
use crate::types::*;

// ─── Public types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkshopTier {
    Cheap,
    Standard,
    Deep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkshopFocus {
    Pacing,
    Continuity,
    Characterization,
    Prose,
    Structure,
}

impl WorkshopFocus {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Pacing,
            Self::Continuity,
            Self::Characterization,
            Self::Prose,
            Self::Structure,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    High,
    Warning,
    Info,
    Note,
}

impl From<PathologySeverity> for FindingSeverity {
    fn from(p: PathologySeverity) -> Self {
        match p {
            PathologySeverity::Error => FindingSeverity::High,
            PathologySeverity::Warning => FindingSeverity::Warning,
            PathologySeverity::Info => FindingSeverity::Info,
            PathologySeverity::Note => FindingSeverity::Note,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedEdit {
    pub situation_id: Uuid,
    pub label: String,
    /// Serialized `EditOperation` — the Studio feeds this straight into the
    /// W2 EditPanel when the writer clicks "Apply suggestion".
    pub operation: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub id: Uuid,
    pub severity: FindingSeverity,
    pub focus: WorkshopFocus,
    #[serde(default)]
    pub chapter_id: Option<Uuid>,
    #[serde(default)]
    pub chapter_name: Option<String>,
    /// Short one-line summary (plays role of a PR title).
    pub headline: String,
    /// Detailed evidence / explanation.
    pub evidence: String,
    /// Human-readable next step. May be set even when there's no structured edit.
    #[serde(default)]
    pub suggestion: Option<String>,
    /// Optional one-click edit that plugs into the W2 edit engine.
    #[serde(default)]
    pub suggested_edit: Option<SuggestedEdit>,
    /// LLM review (Standard tier). `None` on Cheap tier.
    #[serde(default)]
    pub llm_review: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostRecord {
    pub prompt_tokens: u32,
    pub response_tokens: u32,
    pub llm_calls: u32,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkshopRequest {
    #[serde(default)]
    pub narrative_id: String,
    pub tier: WorkshopTier,
    pub focuses: Vec<WorkshopFocus>,
    /// Cap on LLM-enriched findings per focus in Standard tier.
    #[serde(default)]
    pub max_llm_calls: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkshopReport {
    pub id: Uuid,
    pub narrative_id: String,
    pub tier: WorkshopTier,
    pub focuses: Vec<WorkshopFocus>,
    pub findings: Vec<Finding>,
    pub cost: CostRecord,
    pub deferred: bool,
    pub created_at: DateTime<Utc>,
}

/// Lightweight list entry for the reports sidebar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub id: Uuid,
    pub narrative_id: String,
    pub tier: WorkshopTier,
    pub focuses: Vec<WorkshopFocus>,
    pub finding_counts: FindingCounts,
    pub created_at: DateTime<Utc>,
    pub deferred: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FindingCounts {
    pub high: u32,
    pub warning: u32,
    pub info: u32,
    pub note: u32,
    pub total: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkshopEstimate {
    pub tier: WorkshopTier,
    pub focuses: Vec<WorkshopFocus>,
    pub expected_llm_calls: u32,
    pub expected_prompt_tokens: u32,
    pub expected_response_tokens: u32,
    pub estimated_duration_seconds: u32,
}

// ─── Estimate ─────────────────────────────────────────────────────

const STANDARD_DEFAULT_MAX_LLM_CALLS: u32 = 8;
const STANDARD_AVG_PROMPT_TOKENS: u32 = 1_200;
const STANDARD_AVG_RESPONSE_TOKENS: u32 = 400;

pub fn estimate_cost(request: &WorkshopRequest) -> WorkshopEstimate {
    let focus_count = request.focuses.len().max(1) as u32;
    match request.tier {
        WorkshopTier::Cheap => WorkshopEstimate {
            tier: request.tier,
            focuses: request.focuses.clone(),
            expected_llm_calls: 0,
            expected_prompt_tokens: 0,
            expected_response_tokens: 0,
            estimated_duration_seconds: 1,
        },
        WorkshopTier::Standard => {
            let per_focus = request
                .max_llm_calls
                .unwrap_or(STANDARD_DEFAULT_MAX_LLM_CALLS);
            let total_calls = per_focus * focus_count;
            WorkshopEstimate {
                tier: request.tier,
                focuses: request.focuses.clone(),
                expected_llm_calls: total_calls,
                expected_prompt_tokens: total_calls * STANDARD_AVG_PROMPT_TOKENS,
                expected_response_tokens: total_calls * STANDARD_AVG_RESPONSE_TOKENS,
                estimated_duration_seconds: (total_calls * 15).min(600),
            }
        }
        WorkshopTier::Deep => WorkshopEstimate {
            tier: request.tier,
            focuses: request.focuses.clone(),
            // Advertised estimate so writers know what they'd be paying for
            // even though the actual call returns a Deferred report for v1.
            expected_llm_calls: 30 + focus_count * 10,
            expected_prompt_tokens: 200_000,
            expected_response_tokens: 60_000,
            estimated_duration_seconds: 300,
        },
    }
}

// ─── Run ──────────────────────────────────────────────────────────

/// Run a workshop analysis. The extractor is required only for Standard tier;
/// pass `None` for Cheap-only runs.
pub fn run_workshop(
    hypergraph: &Hypergraph,
    registry: &NarrativeRegistry,
    extractor: Option<&dyn SessionCapableExtractor>,
    request: WorkshopRequest,
) -> Result<WorkshopReport> {
    registry.get(&request.narrative_id)?;

    let start = std::time::Instant::now();
    let deferred = matches!(request.tier, WorkshopTier::Deep);

    let mut findings = gather_cheap_findings(hypergraph, &request)?;

    let mut cost = CostRecord::default();
    let mut model: Option<String> = None;

    if matches!(request.tier, WorkshopTier::Standard) {
        if let Some(extractor) = extractor {
            let snapshot = gather_snapshot(hypergraph, registry, &request.narrative_id)?;
            let budget = request
                .max_llm_calls
                .unwrap_or(STANDARD_DEFAULT_MAX_LLM_CALLS);
            model = extractor.model_name();
            enrich_findings_with_llm(
                extractor,
                &snapshot,
                &mut findings,
                &request.focuses,
                budget,
                &mut cost,
            )?;
        }
    }

    sort_findings(&mut findings);

    cost.duration_ms = start.elapsed().as_millis() as u64;

    // One ledger entry per workshop run. Per-call granularity would be
    // noisy — the workshop report itself already breaks down findings.
    if cost.llm_calls > 0 {
        crate::narrative::cost_ledger::record(
            hypergraph.store(),
            &request.narrative_id,
            CostOperation::Workshop,
            format!("{:?}", request.tier).to_lowercase(),
            cost.prompt_tokens,
            cost.response_tokens,
            model,
            false,
            true,
            cost.duration_ms,
            None,
        );
    }

    let report = WorkshopReport {
        id: Uuid::now_v7(),
        narrative_id: request.narrative_id.clone(),
        tier: request.tier,
        focuses: request.focuses.clone(),
        findings,
        cost,
        deferred,
        created_at: Utc::now(),
    };

    save_report(hypergraph.store(), &report)?;
    Ok(report)
}

fn sort_findings(findings: &mut [Finding]) {
    findings.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| a.chapter_name.cmp(&b.chapter_name))
    });
}

// ─── Cheap tier — orchestration over existing detectors ───────────

fn gather_cheap_findings(hg: &Hypergraph, request: &WorkshopRequest) -> Result<Vec<Finding>> {
    let focus_set: HashSet<WorkshopFocus> = request.focuses.iter().copied().collect();
    let mut findings: Vec<Finding> = Vec::new();

    // Structure-wide diagnose covers most focuses. Run once and demux.
    let diag = diagnose_narrative(hg, &request.narrative_id)?;
    let situations = hg.list_situations_by_narrative(&request.narrative_id)?;
    let chapter_name_lookup = build_chapter_name_lookup(&situations);

    for p in &diag.pathologies {
        let focus = focus_for_pathology(&p.kind);
        let requested =
            focus_set.contains(&focus) || (focus_set.contains(&WorkshopFocus::Structure));
        if !requested {
            continue;
        }

        let chapter_id = p.location.situation.or(p.target_id);
        let chapter_name = chapter_id.and_then(|id| chapter_name_lookup.get(&id).cloned());

        findings.push(Finding {
            id: Uuid::now_v7(),
            severity: p.severity.into(),
            focus,
            chapter_id,
            chapter_name,
            headline: format_pathology_headline(p),
            evidence: p.description.clone(),
            suggestion: p.suggestion.clone(),
            suggested_edit: suggest_edit_for_pathology(p),
            llm_review: None,
        });
    }

    if focus_set.contains(&WorkshopFocus::Prose) || focus_set.contains(&WorkshopFocus::Pacing) {
        findings.extend(chapter_metric_findings(&situations, &focus_set));
    }

    if focus_set.contains(&WorkshopFocus::Pacing) || focus_set.contains(&WorkshopFocus::Structure) {
        if let Some(plan_finding) = plan_length_finding(hg, &request.narrative_id, &situations)? {
            findings.push(plan_finding);
        }
    }

    Ok(findings)
}

fn focus_for_pathology(kind: &PathologyKind) -> WorkshopFocus {
    match kind {
        PathologyKind::PacingArrhythmia
        | PathologyKind::NarrationModeMonotony
        | PathologyKind::SubplotStarvation
        | PathologyKind::SubplotOrphan => WorkshopFocus::Pacing,
        PathologyKind::OrphanedSetup
        | PathologyKind::UnseededPayoff
        | PathologyKind::PrematurePayoff
        | PathologyKind::PromiseOverload
        | PathologyKind::PromiseDesert
        | PathologyKind::ImpossibleKnowledge
        | PathologyKind::ForgottenKnowledge
        | PathologyKind::IronyCollapse
        | PathologyKind::LeakyFocalization
        | PathologyKind::CausalOrphan
        | PathologyKind::CausalContradiction
        | PathologyKind::CausalIsland
        | PathologyKind::TemporalImpossibility
        | PathologyKind::AnachronismRisk => WorkshopFocus::Continuity,
        PathologyKind::MotivationDiscontinuity
        | PathologyKind::ArcAbandonment
        | PathologyKind::FlatProtagonist
        | PathologyKind::MotivationImplausibility => WorkshopFocus::Characterization,
    }
}

fn format_pathology_headline(p: &NarrativePathology) -> String {
    format!("{:?}", p.kind)
        .chars()
        .enumerate()
        .flat_map(|(i, c)| {
            if i > 0 && c.is_uppercase() {
                vec![' ', c.to_ascii_lowercase()]
            } else if i == 0 {
                vec![c]
            } else {
                vec![c.to_ascii_lowercase()]
            }
        })
        .collect()
}

fn suggest_edit_for_pathology(p: &NarrativePathology) -> Option<SuggestedEdit> {
    // v1: map a handful of kinds to concrete EditOperations. Rest are
    // "suggestion text only" for the writer to act on.
    let sit_id = p.location.situation.or(p.target_id)?;
    match p.kind {
        PathologyKind::PacingArrhythmia => Some(SuggestedEdit {
            situation_id: sit_id,
            label: "Tighten this chapter to 75%".into(),
            operation: serde_json::to_value(EditOperation::Tighten {
                target_ratio: 0.75,
                instruction: p.suggestion.clone(),
            })
            .ok()?,
        }),
        PathologyKind::SubplotStarvation => Some(SuggestedEdit {
            situation_id: sit_id,
            label: "Expand this chapter to 120%".into(),
            operation: serde_json::to_value(EditOperation::Expand {
                target_multiplier: 1.2,
                instruction: p.suggestion.clone(),
            })
            .ok()?,
        }),
        PathologyKind::NarrationModeMonotony => Some(SuggestedEdit {
            situation_id: sit_id,
            label: "Vary register (cinematic style transfer)".into(),
            operation: serde_json::to_value(EditOperation::StyleTransfer {
                target: StyleTarget::Preset {
                    name: "cinematic".into(),
                },
            })
            .ok()?,
        }),
        _ => None,
    }
}

fn build_chapter_name_lookup(situations: &[Situation]) -> std::collections::HashMap<Uuid, String> {
    situations
        .iter()
        .map(|s| {
            let name = s.name.clone().unwrap_or_else(|| "(untitled)".into());
            (s.id, name)
        })
        .collect()
}

/// Per-chapter sentence-length outliers + empty-chapter checks. Very cheap.
fn chapter_metric_findings(
    situations: &[Situation],
    focus_set: &HashSet<WorkshopFocus>,
) -> Vec<Finding> {
    let mut out = Vec::new();
    // Chapter-level scope only.
    let chapters: Vec<&Situation> = situations
        .iter()
        .filter(|s| {
            matches!(
                s.narrative_level,
                NarrativeLevel::Arc | NarrativeLevel::Sequence | NarrativeLevel::Story
            )
        })
        .collect();

    if chapters.is_empty() {
        return out;
    }

    // Compute per-chapter means once; reuse for both the baseline stats and
    // the outlier loop below. Previously computed twice per chapter.
    let chapter_means: Vec<(usize, f64)> = chapters
        .iter()
        .enumerate()
        .map(|(i, c)| (i, chapter_mean_sentence_len(c)))
        .collect();
    let samples: Vec<f64> = chapter_means
        .iter()
        .map(|(_, m)| *m)
        .filter(|m| *m > 0.0)
        .collect();
    let (baseline_mean, baseline_std) = if samples.is_empty() {
        (0.0, 0.0)
    } else {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        (mean, variance.sqrt())
    };

    for (i, c) in chapters.iter().enumerate() {
        let words = count_words_blocks(&c.raw_content);
        let name = c.name.clone().unwrap_or_else(|| "(untitled)".into());

        // Very-short chapter — usually a stub.
        if words < 150 && focus_set.contains(&WorkshopFocus::Pacing) {
            let sit_id = c.id;
            out.push(Finding {
                id: Uuid::now_v7(),
                severity: FindingSeverity::Info,
                focus: WorkshopFocus::Pacing,
                chapter_id: Some(c.id),
                chapter_name: Some(name.clone()),
                headline: "Very short chapter".into(),
                evidence: format!("Chapter \u{201C}{}\u{201D} has only {} words. Likely a stub that needs expansion.", name, words),
                suggestion: Some("Expand or merge with a neighbour.".into()),
                suggested_edit: Some(SuggestedEdit {
                    situation_id: sit_id,
                    label: "Expand to 200%".into(),
                    operation: serde_json::to_value(EditOperation::Expand {
                        target_multiplier: 2.0,
                        instruction: None,
                    }).ok().unwrap_or_default(),
                }),
                llm_review: None,
            });
        }

        // Very-long chapter — flag for review.
        if words > 10_000 && focus_set.contains(&WorkshopFocus::Pacing) {
            let sit_id = c.id;
            out.push(Finding {
                id: Uuid::now_v7(),
                severity: FindingSeverity::Warning,
                focus: WorkshopFocus::Pacing,
                chapter_id: Some(c.id),
                chapter_name: Some(name.clone()),
                headline: "Very long chapter".into(),
                evidence: format!(
                    "Chapter \u{201C}{}\u{201D} has {} words. Consider splitting or tightening.",
                    name, words
                ),
                suggestion: Some("Split into two chapters, or tighten with an AI edit.".into()),
                suggested_edit: Some(SuggestedEdit {
                    situation_id: sit_id,
                    label: "Tighten to 65%".into(),
                    operation: serde_json::to_value(EditOperation::Tighten {
                        target_ratio: 0.65,
                        instruction: None,
                    })
                    .ok()
                    .unwrap_or_default(),
                }),
                llm_review: None,
            });
        }

        // Sentence-length outliers (Prose focus).
        if focus_set.contains(&WorkshopFocus::Prose) {
            let mean = chapter_means[i].1;
            if mean > 0.0 && baseline_std > 0.0 {
                let z = (mean - baseline_mean).abs() / baseline_std;
                if z > 2.0 {
                    let direction = if mean > baseline_mean {
                        "longer"
                    } else {
                        "shorter"
                    };
                    out.push(Finding {
                        id: Uuid::now_v7(),
                        severity: FindingSeverity::Info,
                        focus: WorkshopFocus::Prose,
                        chapter_id: Some(c.id),
                        chapter_name: Some(name.clone()),
                        headline: format!("Sentence length outlier ({})", direction),
                        evidence: format!(
                            "Chapter \u{201C}{}\u{201D}: mean sentence length {:.1} words vs corpus mean {:.1} \u{00b1} {:.1} (z \u{2248} {:.1}).",
                            name, mean, baseline_mean, baseline_std, z
                        ),
                        suggestion: Some(format!("Review for monotonous rhythm — chapter runs {} than corpus average.", direction)),
                        suggested_edit: Some(SuggestedEdit {
                            situation_id: c.id,
                            label: if mean > baseline_mean { "Apply punchy style".into() } else { "Apply lyrical style".into() },
                            operation: serde_json::to_value(EditOperation::StyleTransfer {
                                target: StyleTarget::Preset {
                                    name: if mean > baseline_mean { "punchy".into() } else { "lyrical".into() },
                                },
                            }).ok().unwrap_or_default(),
                        }),
                        llm_review: None,
                    });
                }
            }
        }
    }

    out
}

fn chapter_mean_sentence_len(s: &Situation) -> f64 {
    let text: String = s
        .raw_content
        .iter()
        .map(|b| b.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let sentences: Vec<&str> = text
        .split(|c: char| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();
    if sentences.is_empty() {
        return 0.0;
    }
    let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();
    total_words as f64 / sentences.len() as f64
}

/// Flag when current word count is far from plan.length.target_words.
fn plan_length_finding(
    hg: &Hypergraph,
    narrative_id: &str,
    situations: &[Situation],
) -> Result<Option<Finding>> {
    let plan = match crate::narrative::plan::get_plan(hg.store(), narrative_id)? {
        Some(p) => p,
        None => return Ok(None),
    };
    let target = match plan.length.target_words {
        Some(t) => t as f64,
        None => return Ok(None),
    };
    let current: usize = situations
        .iter()
        .map(|s| count_words_blocks(&s.raw_content))
        .sum();
    let ratio = (current as f64) / target;

    let (severity, direction) = if ratio < 0.6 {
        (FindingSeverity::Warning, "far below")
    } else if ratio < 0.85 {
        (FindingSeverity::Info, "below")
    } else if ratio > 1.4 {
        (FindingSeverity::Warning, "far above")
    } else if ratio > 1.15 {
        (FindingSeverity::Info, "above")
    } else {
        return Ok(None);
    };

    Ok(Some(Finding {
        id: Uuid::now_v7(),
        severity,
        focus: WorkshopFocus::Pacing,
        chapter_id: None,
        chapter_name: None,
        headline: format!("Manuscript length {} plan target", direction),
        evidence: format!(
            "Plan target: {} words. Current total: {} words ({:.0}% of target).",
            target as u32,
            current,
            ratio * 100.0
        ),
        suggestion: Some("Either update the plan or adjust scope in the manuscript.".into()),
        suggested_edit: None,
        llm_review: None,
    }))
}

// ─── Standard tier — LLM enrichment of top findings ──────────────

const LLM_SYSTEM_CRITIC: &str = "You are a manuscript editor giving a concise second opinion on a structural finding. \
Return ONLY a JSON object: {\"review\": str, \"severity_adjustment\": \"up\"|\"down\"|\"keep\"}. \
The review is 1\u{2013}2 sentences, specific, actionable, no fluff. \
Use severity_adjustment=up when the finding is more serious than the summary suggests; down when it's a false alarm; keep otherwise.";

fn enrich_findings_with_llm(
    extractor: &dyn SessionCapableExtractor,
    snapshot: &NarrativeSnapshot,
    findings: &mut [Finding],
    focuses: &[WorkshopFocus],
    budget_per_focus: u32,
    cost: &mut CostRecord,
) -> Result<()> {
    // Select up to `budget_per_focus` from each focus, highest severity first.
    let mut picked: Vec<usize> = Vec::new();
    for focus in focuses {
        let mut candidates: Vec<usize> = findings
            .iter()
            .enumerate()
            .filter(|(_, f)| f.focus == *focus)
            .map(|(i, _)| i)
            .collect();
        candidates.sort_by_key(|&i| findings[i].severity);
        picked.extend(candidates.into_iter().take(budget_per_focus as usize));
    }

    for idx in picked {
        let finding = &findings[idx];
        let prompt = build_finding_prompt(finding, snapshot);
        cost.prompt_tokens = cost.prompt_tokens.saturating_add(approx_tokens(&prompt));
        cost.llm_calls += 1;

        let messages = vec![
            ApiMessage {
                role: "system".into(),
                content: LLM_SYSTEM_CRITIC.into(),
            },
            ApiMessage {
                role: "user".into(),
                content: prompt,
            },
        ];
        let response = match extractor.send_session_messages(&messages) {
            Ok(r) => r,
            Err(_) => continue,
        };
        cost.response_tokens = cost
            .response_tokens
            .saturating_add(approx_tokens(&response));

        if let Ok(v) = extract_json_object(&response) {
            let review = v.get("review").and_then(|x| x.as_str()).map(String::from);
            let severity_adj = v
                .get("severity_adjustment")
                .and_then(|x| x.as_str())
                .unwrap_or("keep");
            findings[idx].llm_review = review;
            findings[idx].severity = adjust_severity(findings[idx].severity, severity_adj);
        }
    }
    Ok(())
}

fn adjust_severity(current: FindingSeverity, direction: &str) -> FindingSeverity {
    match (current, direction) {
        (FindingSeverity::Note, "up") => FindingSeverity::Info,
        (FindingSeverity::Info, "up") => FindingSeverity::Warning,
        (FindingSeverity::Warning, "up") => FindingSeverity::High,
        (FindingSeverity::High, "down") => FindingSeverity::Warning,
        (FindingSeverity::Warning, "down") => FindingSeverity::Info,
        (FindingSeverity::Info, "down") => FindingSeverity::Note,
        _ => current,
    }
}

fn build_finding_prompt(finding: &Finding, snapshot: &NarrativeSnapshot) -> String {
    let mut out = String::new();
    if let Some(plan) = snapshot.plan.as_ref() {
        if let Some(l) = &plan.logline {
            out.push_str(&format!("Logline: {}\n", l));
        }
        let s = &plan.style;
        if let Some(pov) = &s.pov {
            out.push_str(&format!("POV: {}\n", pov));
        }
        if !s.tone.is_empty() {
            out.push_str(&format!("Tone: {}\n", s.tone.join("/")));
        }
    }
    out.push_str(&format!(
        "\nFocus: {:?}\nHeadline: {}\nEvidence: {}\n",
        finding.focus, finding.headline, finding.evidence
    ));
    if let Some(name) = &finding.chapter_name {
        out.push_str(&format!("Chapter: {}\n", name));
    }
    if let Some(chapter_id) = finding.chapter_id {
        if let Some(chapter) = snapshot.situations.iter().find(|s| s.id == chapter_id) {
            out.push_str("\n[chapter excerpt]\n");
            // Cap the excerpt at 2000 bytes. Check BEFORE pushing so a single
            // long block can't balloon the prompt.
            const EXCERPT_CAP: usize = 2000;
            let mut excerpt = String::new();
            for b in &chapter.raw_content {
                if excerpt.len() >= EXCERPT_CAP {
                    break;
                }
                let trimmed = b.content.trim();
                let remaining = EXCERPT_CAP - excerpt.len();
                if trimmed.len() > remaining {
                    let mut boundary = remaining;
                    while boundary > 0 && !trimmed.is_char_boundary(boundary) {
                        boundary -= 1;
                    }
                    excerpt.push_str(&trimmed[..boundary]);
                    break;
                }
                excerpt.push_str(trimmed);
                excerpt.push('\n');
            }
            out.push_str(&excerpt);
        }
    }
    out
}

// ─── Persistence ──────────────────────────────────────────────────

fn save_report(store: &dyn KVStore, report: &WorkshopReport) -> Result<()> {
    let key = keys::workshop_report_key(&report.id);
    let bytes = serde_json::to_vec(report)?;
    store.put(&key, &bytes)?;
    // Store the summary as the index value so `list_reports` doesn't need a
    // second per-report fetch (previously an N+1 pattern).
    let summary = ReportSummary {
        id: report.id,
        narrative_id: report.narrative_id.clone(),
        tier: report.tier,
        focuses: report.focuses.clone(),
        finding_counts: count_findings(&report.findings),
        created_at: report.created_at,
        deferred: report.deferred,
    };
    let idx = keys::workshop_report_narrative_index_key(&report.narrative_id, &report.id);
    let summary_bytes = serde_json::to_vec(&summary)?;
    store.put(&idx, &summary_bytes)?;
    Ok(())
}

pub fn get_report(store: &dyn KVStore, id: &Uuid) -> Result<WorkshopReport> {
    let key = keys::workshop_report_key(id);
    match store.get(&key)? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::NotFound(format!("workshop report {}", id))),
    }
}

pub fn list_reports(store: &dyn KVStore, narrative_id: &str) -> Result<Vec<ReportSummary>> {
    let prefix = keys::workshop_report_narrative_prefix(narrative_id);
    let pairs = store.prefix_scan(&prefix)?;
    let min_len = prefix.len() + 16;
    let mut out: Vec<ReportSummary> = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        // Deserialize the summary directly from the index value. No second
        // KV round-trip needed.
        if key.len() < min_len {
            continue;
        }
        match serde_json::from_slice::<ReportSummary>(&value) {
            Ok(summary) => out.push(summary),
            Err(_) => {
                // Legacy entries (pre-refactor) stored empty values; fall back
                // to fetching the full record by UUID.
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&key[key.len() - 16..]);
                let id = Uuid::from_bytes(arr);
                if let Ok(report) = get_report(store, &id) {
                    out.push(ReportSummary {
                        id: report.id,
                        narrative_id: report.narrative_id,
                        tier: report.tier,
                        focuses: report.focuses,
                        finding_counts: count_findings(&report.findings),
                        created_at: report.created_at,
                        deferred: report.deferred,
                    });
                }
            }
        }
    }
    Ok(out)
}

fn count_findings(findings: &[Finding]) -> FindingCounts {
    let mut c = FindingCounts::default();
    for f in findings {
        c.total += 1;
        match f.severity {
            FindingSeverity::High => c.high += 1,
            FindingSeverity::Warning => c.warning += 1,
            FindingSeverity::Info => c.info += 1,
            FindingSeverity::Note => c.note += 1,
        }
    }
    c
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
        fn send_session_messages(&self, _m: &[ApiMessage]) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    fn setup_with_chapters(chapters: &[(&str, usize)]) -> (Hypergraph, NarrativeRegistry) {
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
        let mut when = Utc::now();
        for (name, words) in chapters {
            let text = "word ".repeat(*words);
            hg.create_situation(Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: Some((*name).into()),
                description: Some(format!("{} (test)", name)),
                temporal: AllenInterval {
                    start: Some(when),
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
                raw_content: vec![ContentBlock {
                    content_type: ContentType::Text,
                    content: text.trim_end().to_string(),
                    source: None,
                }],
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
                created_at: when,
                updated_at: when,
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap();
            when = when + chrono::Duration::days(1);
        }
        (hg, reg)
    }

    #[test]
    fn cheap_tier_flags_very_short_chapter() {
        let (hg, reg) = setup_with_chapters(&[
            ("Ch1", 50), // too short
            ("Ch2", 800),
            ("Ch3", 900),
        ]);
        let report = run_workshop(
            &hg,
            &reg,
            None,
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Cheap,
                focuses: vec![WorkshopFocus::Pacing],
                max_llm_calls: None,
            },
        )
        .unwrap();
        assert!(report
            .findings
            .iter()
            .any(|f| f.headline == "Very short chapter"));
        assert_eq!(report.cost.llm_calls, 0);
    }

    #[test]
    fn cheap_tier_flags_very_long_chapter() {
        let (hg, reg) = setup_with_chapters(&[("Ch1", 15_000)]);
        let report = run_workshop(
            &hg,
            &reg,
            None,
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Cheap,
                focuses: vec![WorkshopFocus::Pacing],
                max_llm_calls: None,
            },
        )
        .unwrap();
        assert!(report
            .findings
            .iter()
            .any(|f| f.headline == "Very long chapter"));
        // Suggested edit should be a Tighten operation.
        let finding = report
            .findings
            .iter()
            .find(|f| f.headline == "Very long chapter")
            .unwrap();
        let op = finding.suggested_edit.as_ref().unwrap();
        assert!(op.label.contains("Tighten"));
    }

    #[test]
    fn cheap_tier_flags_plan_length_mismatch() {
        use crate::narrative::plan;
        let (hg, reg) = setup_with_chapters(&[("Ch1", 800), ("Ch2", 800)]);
        plan::upsert_plan(
            hg.store(),
            NarrativePlan {
                narrative_id: "draft".into(),
                length: LengthTargets {
                    target_words: Some(80_000),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let report = run_workshop(
            &hg,
            &reg,
            None,
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Cheap,
                focuses: vec![WorkshopFocus::Pacing],
                max_llm_calls: None,
            },
        )
        .unwrap();
        // Far below target (1600 / 80k) — should be Warning severity.
        let f = report
            .findings
            .iter()
            .find(|f| f.headline.contains("Manuscript length"))
            .unwrap();
        assert!(matches!(f.severity, FindingSeverity::Warning));
    }

    #[test]
    fn estimate_is_free_for_cheap() {
        let est = estimate_cost(&WorkshopRequest {
            narrative_id: "x".into(),
            tier: WorkshopTier::Cheap,
            focuses: vec![WorkshopFocus::Pacing, WorkshopFocus::Prose],
            max_llm_calls: None,
        });
        assert_eq!(est.expected_llm_calls, 0);
        assert_eq!(est.expected_prompt_tokens, 0);
    }

    #[test]
    fn estimate_scales_with_focus_count_for_standard() {
        let one = estimate_cost(&WorkshopRequest {
            narrative_id: "x".into(),
            tier: WorkshopTier::Standard,
            focuses: vec![WorkshopFocus::Pacing],
            max_llm_calls: None,
        });
        let three = estimate_cost(&WorkshopRequest {
            narrative_id: "x".into(),
            tier: WorkshopTier::Standard,
            focuses: vec![
                WorkshopFocus::Pacing,
                WorkshopFocus::Prose,
                WorkshopFocus::Continuity,
            ],
            max_llm_calls: None,
        });
        assert_eq!(three.expected_llm_calls, one.expected_llm_calls * 3);
    }

    #[test]
    fn standard_tier_enriches_findings_with_llm() {
        let (hg, reg) = setup_with_chapters(&[("Ch1", 60)]);
        let mock = MockExtractor(
            r#"{"review":"Short but might be a deliberate stylistic choice; check adjacency.","severity_adjustment":"down"}"#.into(),
        );
        let report = run_workshop(
            &hg,
            &reg,
            Some(&mock),
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Standard,
                focuses: vec![WorkshopFocus::Pacing],
                max_llm_calls: Some(4),
            },
        )
        .unwrap();
        assert!(report.cost.llm_calls >= 1);
        let enriched = report
            .findings
            .iter()
            .find(|f| f.llm_review.is_some())
            .expect("at least one finding should be enriched");
        assert!(enriched.llm_review.as_ref().unwrap().contains("deliberate"));
    }

    #[test]
    fn deep_tier_is_deferred_but_returns_cheap_findings() {
        let (hg, reg) = setup_with_chapters(&[("Ch1", 50)]);
        let report = run_workshop(
            &hg,
            &reg,
            None,
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Deep,
                focuses: vec![WorkshopFocus::Pacing],
                max_llm_calls: None,
            },
        )
        .unwrap();
        assert!(report.deferred);
        assert!(
            !report.findings.is_empty(),
            "should still return cheap findings"
        );
    }

    #[test]
    fn reports_persist_and_list() {
        let (hg, reg) = setup_with_chapters(&[("Ch1", 100)]);
        let r1 = run_workshop(
            &hg,
            &reg,
            None,
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Cheap,
                focuses: vec![WorkshopFocus::Pacing],
                max_llm_calls: None,
            },
        )
        .unwrap();
        let r2 = run_workshop(
            &hg,
            &reg,
            None,
            WorkshopRequest {
                narrative_id: "draft".into(),
                tier: WorkshopTier::Cheap,
                focuses: vec![WorkshopFocus::Structure],
                max_llm_calls: None,
            },
        )
        .unwrap();
        let list = list_reports(hg.store(), "draft").unwrap();
        assert_eq!(list.len(), 2);
        let ids: HashSet<Uuid> = list.iter().map(|r| r.id).collect();
        assert!(ids.contains(&r1.id) && ids.contains(&r2.id));

        let fetched = get_report(hg.store(), &r1.id).unwrap();
        assert_eq!(fetched.id, r1.id);
    }
}
