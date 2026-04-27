//! Ingestion pipeline configuration types.
//!
//! Supports single-pass (legacy) and multi-pass (Pass 1 extraction + Pass 2
//! reconciliation + Pass 3 algorithmic merge) pipeline modes.

use serde::{Deserialize, Serialize};

use crate::api::server::LlmConfig;

/// Pipeline execution mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PipelineMode {
    /// Original single-pass: chunk → extract → resolve → gate → commit.
    #[default]
    Single,
    /// Multi-pass: Pass 1 (fast extraction) → Pass 2 (reasoning reconciliation) → Pass 3 (merge).
    Multi,
    /// Single-session: entire text sent once, then per-chunk extraction in same LLM conversation.
    /// Requires a large-context LLM (128k+ tokens) and a session-capable extractor (OpenRouter/Local).
    SingleSession,
}

/// Full ingestion pipeline configuration, exposed via the settings API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Pipeline mode: single-pass or multi-pass.
    pub mode: PipelineMode,

    // ─── Pass 1 (Extraction) ─────────────────────────────────
    /// LLM config for Pass 1 (fast model, e.g. qwen3.5:4b).
    pub pass1: LlmConfig,

    /// Number of chunks to extract in parallel (GPU concurrency).
    #[serde(default = "default_pass1_concurrency")]
    pub pass1_concurrency: usize,

    // ─── Pass 2 (Reconciliation) ─────────────────────────────
    /// LLM config for Pass 2 (reasoning model, e.g. qwen3.5:27b).
    /// Only used in multi-pass mode.
    pub pass2: LlmConfig,

    /// Number of chunks per reconciliation window.
    #[serde(default = "default_pass2_window_size")]
    pub pass2_window_size: usize,

    /// Overlap chunks between adjacent reconciliation windows.
    #[serde(default = "default_pass2_window_overlap")]
    pub pass2_window_overlap: usize,

    // ─── Chunking ────────────────────────────────────────────
    /// Chunking strategy: "fixed_size" (default) or "chapter" (for large-context models).
    #[serde(default)]
    pub chunk_strategy: crate::ingestion::chunker::ChunkStrategy,

    /// Maximum tokens per chunk.
    #[serde(default = "default_chunk_max_tokens")]
    pub chunk_max_tokens: usize,

    /// Overlap tokens between adjacent chunks.
    #[serde(default = "default_chunk_overlap_tokens")]
    pub chunk_overlap_tokens: usize,

    /// Model context window size in tokens (used to auto-select chunking strategy).
    /// 0 = unknown (use defaults).
    #[serde(default)]
    pub context_window_tokens: usize,

    /// Optional custom regex for chapter detection (overrides built-in heuristics).
    /// E.g. `(?mi)^([IVXLCDM]+)$` for bare Roman numerals.
    #[serde(default)]
    pub chapter_regex: Option<String>,

    /// Strip publisher boilerplate (Standard Ebooks colophon, Gutenberg headers, etc.)
    /// before chunking. Enabled by default.
    #[serde(default = "default_strip_boilerplate")]
    pub strip_boilerplate: bool,

    // ─── Confidence Gating ───────────────────────────────────
    /// Confidence threshold for auto-commit (>= this → commit directly).
    #[serde(default = "default_auto_commit_threshold")]
    pub auto_commit_threshold: f32,

    /// Confidence threshold for review queue (>= this → queue for HITL).
    #[serde(default = "default_review_threshold")]
    pub review_threshold: f32,

    /// Enable step-2 enrichment pass (beliefs, game structures, discourse, info sets).
    /// Doubles LLM calls per chunk but produces much richer data.
    #[serde(default)]
    pub enrich: bool,

    /// Content domain mode — selects a domain-specific extraction prompt.
    /// Each mode focuses the LLM on domain-relevant entities, relations, and temporal patterns.
    #[serde(default)]
    pub ingestion_mode: IngestionMode,

    /// Maximum context window tokens for SingleSession mode.
    /// 0 = use `context_window_tokens` as the limit.
    #[serde(default)]
    pub session_max_context_tokens: usize,

    /// Fuzzy Sprint Phase 9 — when Some, references a persisted Mamdani
    /// rule (`fz/rules/{nid}/{rule_id}`). After each chunk commits its
    /// fresh entities, the pipeline runs the referenced rule against
    /// every newly-ingested entity and attaches the firing strength +
    /// linguistic-term label to `properties.mamdani` for downstream
    /// ranking / review-queue prioritisation. `None` (serde default)
    /// preserves the pre-Phase-9 pipeline bit-identically.
    ///
    /// Cites: [mamdani1975mamdani].
    #[serde(default)]
    pub post_ingest_mamdani_rule_id: Option<String>,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            mode: PipelineMode::default(),
            pass1: LlmConfig::None,
            pass1_concurrency: default_pass1_concurrency(),
            pass2: LlmConfig::None,
            pass2_window_size: default_pass2_window_size(),
            pass2_window_overlap: default_pass2_window_overlap(),
            chunk_strategy: Default::default(),
            chunk_max_tokens: default_chunk_max_tokens(),
            chunk_overlap_tokens: default_chunk_overlap_tokens(),
            context_window_tokens: 0,
            chapter_regex: None,
            strip_boilerplate: default_strip_boilerplate(),
            auto_commit_threshold: default_auto_commit_threshold(),
            review_threshold: default_review_threshold(),
            enrich: true,
            session_max_context_tokens: 0,
            ingestion_mode: IngestionMode::default(),
            post_ingest_mamdani_rule_id: None,
        }
    }
}

impl IngestionConfig {
    /// Return a copy with API keys redacted for safe frontend serialization.
    pub fn redacted(&self) -> Self {
        Self {
            pass1: self.pass1.redacted(),
            pass2: self.pass2.redacted(),
            ..self.clone()
        }
    }

    /// Return a copy with API keys replaced by hints (first 4 chars + "...").
    pub fn redacted_hint(&self) -> Self {
        Self {
            pass1: self.pass1.redacted_hint(),
            pass2: self.pass2.redacted_hint(),
            ..self.clone()
        }
    }

    /// Build a ChunkerConfig from the ingestion settings.
    ///
    /// When `context_window_tokens` is set and `chunk_strategy` is `Chapter`,
    /// automatically raises `max_tokens` so chapters stay whole (up to 10% of
    /// the context window). This prevents absurd sub-chunking when the model
    /// can handle large inputs (e.g. 2M tokens → 200k max per chapter).
    pub fn chunker_config(&self) -> crate::ingestion::chunker::ChunkerConfig {
        let mut max_tokens = self.chunk_max_tokens;

        if self.context_window_tokens > 0
            && self.chunk_strategy == crate::ingestion::chunker::ChunkStrategy::Chapter
            && self.chunk_max_tokens == default_chunk_max_tokens()
        {
            // Auto-tune: allow each chapter up to 10% of context window.
            // For 2M tokens → 200k per chunk, 128k → 12.8k per chunk.
            max_tokens = self.context_window_tokens / 10;
        }

        crate::ingestion::chunker::ChunkerConfig {
            max_tokens,
            overlap_tokens: self.chunk_overlap_tokens,
            chapter_regex: self.chapter_regex.clone(),
            strategy: self.chunk_strategy.clone(),
        }
    }
}

fn default_pass1_concurrency() -> usize {
    4
}
fn default_pass2_window_size() -> usize {
    20
}
fn default_pass2_window_overlap() -> usize {
    3
}
fn default_chunk_max_tokens() -> usize {
    2000
}
fn default_chunk_overlap_tokens() -> usize {
    200
}
fn default_strip_boilerplate() -> bool {
    true
}
fn default_auto_commit_threshold() -> f32 {
    0.8
}
fn default_review_threshold() -> f32 {
    0.3
}
fn default_max_retries() -> u8 {
    3
}

// ─── Ingestion Templates ────────────────────────────────────────

/// Content domain mode — affects default system prompt and extraction behavior.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IngestionMode {
    /// Optimized for fiction: character-focused, chapter-aware, dialogue extraction.
    Novel,
    /// Optimized for journalism: date-focused, source attribution, fact extraction.
    News,
    /// Optimized for intelligence reports: entity relationships, confidence, source reliability.
    Intelligence,
    /// Optimized for academic papers: citations, methodology, findings, concepts.
    Research,
    /// Optimized for event databases, timelines, historical records: precise date/time extraction, Allen relations.
    TemporalEvents,
    /// Optimized for contracts, court documents, regulations: parties, obligations, dates, clauses.
    Legal,
    /// Optimized for transactions, filings, financial reports: entities, amounts, dates, flows.
    Financial,
    /// Optimized for clinical records, case studies, medical literature: patients, conditions, treatments, outcomes.
    Medical,
    /// User-defined prompt, no mode-specific defaults.
    Custom,
}

impl Default for IngestionMode {
    fn default() -> Self {
        Self::Novel
    }
}

impl IngestionMode {
    /// All modes exposed to users (`Custom` is not a preset — it represents user-defined prompts).
    pub const PRESETS: &'static [IngestionMode] = &[
        IngestionMode::Novel,
        IngestionMode::News,
        IngestionMode::Intelligence,
        IngestionMode::Research,
        IngestionMode::TemporalEvents,
        IngestionMode::Legal,
        IngestionMode::Financial,
        IngestionMode::Medical,
    ];

    /// Stable snake_case identifier matching `#[serde(rename_all = "snake_case")]`.
    pub fn id_str(&self) -> &'static str {
        match self {
            IngestionMode::Novel => "novel",
            IngestionMode::News => "news",
            IngestionMode::Intelligence => "intelligence",
            IngestionMode::Research => "research",
            IngestionMode::TemporalEvents => "temporal_events",
            IngestionMode::Legal => "legal",
            IngestionMode::Financial => "financial",
            IngestionMode::Medical => "medical",
            IngestionMode::Custom => "custom",
        }
    }

    /// Human-readable description for UI presentation.
    pub fn description(&self) -> &'static str {
        match self {
            IngestionMode::Novel => "Fiction, novels, stories — character-focused, chapter-aware, dialogue extraction",
            IngestionMode::News => "Journalism, news articles — date-focused, source attribution, fact extraction",
            IngestionMode::Intelligence => "OSINT, intelligence reports — entity relationships, confidence, source reliability",
            IngestionMode::Research => "Academic papers, studies — citations, methodology, findings, concepts",
            IngestionMode::TemporalEvents => "Timelines, event databases, historical records — precise date/time, Allen temporal relations",
            IngestionMode::Legal => "Contracts, court documents, regulations — parties, obligations, dates, clauses",
            IngestionMode::Financial => "Transactions, filings, financial reports — entities, amounts, flows, compliance",
            IngestionMode::Medical => "Clinical records, case studies — patients, conditions, treatments, outcomes",
            IngestionMode::Custom => "Default extraction with user-defined prompt tuning",
        }
    }
}

/// Role of a single pass within a template.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PassRole {
    /// Primary extraction pass.
    Extraction,
    /// Reconciliation/verification pass.
    Reconciliation,
    /// Custom/user-defined pass role.
    Custom,
}

impl Default for PassRole {
    fn default() -> Self {
        Self::Extraction
    }
}

/// Configuration for a single pass within a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplatePassConfig {
    /// LLM provider config for this pass.
    pub llm: LlmConfig,
    /// Parallel extraction concurrency.
    #[serde(default = "default_pass1_concurrency")]
    pub concurrency: usize,
    /// Custom system prompt override (None = use mode default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Role of this pass in the pipeline.
    #[serde(default)]
    pub role: PassRole,
    /// Window size for reconciliation passes.
    #[serde(default = "default_pass2_window_size")]
    pub window_size: usize,
    /// Overlap for reconciliation windows.
    #[serde(default = "default_pass2_window_overlap")]
    pub window_overlap: usize,
}

/// Chunking configuration within a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateChunkConfig {
    #[serde(default)]
    pub strategy: crate::ingestion::chunker::ChunkStrategy,
    #[serde(default = "default_chunk_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_chunk_overlap_tokens")]
    pub overlap_tokens: usize,
    #[serde(default)]
    pub context_window_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chapter_regex: Option<String>,
}

impl Default for TemplateChunkConfig {
    fn default() -> Self {
        Self {
            strategy: Default::default(),
            max_tokens: default_chunk_max_tokens(),
            overlap_tokens: default_chunk_overlap_tokens(),
            context_window_tokens: 0,
            chapter_regex: None,
        }
    }
}

/// Confidence gating configuration within a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateGatingConfig {
    #[serde(default = "default_auto_commit_threshold")]
    pub auto_commit_threshold: f32,
    #[serde(default = "default_review_threshold")]
    pub review_threshold: f32,
}

impl Default for TemplateGatingConfig {
    fn default() -> Self {
        Self {
            auto_commit_threshold: default_auto_commit_threshold(),
            review_threshold: default_review_threshold(),
        }
    }
}

/// A named, reusable ingestion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionTemplate {
    /// Unique template ID (e.g. "novel-default").
    pub id: String,
    /// Human-readable name (e.g. "Novel / Literature").
    pub name: String,
    /// Content domain mode — affects default prompts.
    #[serde(default)]
    pub mode: IngestionMode,
    /// Ordered list of extraction passes. At least 1 required.
    pub passes: Vec<TemplatePassConfig>,
    /// Chunking configuration.
    #[serde(default)]
    pub chunking: TemplateChunkConfig,
    /// Confidence gating configuration.
    #[serde(default)]
    pub gating: TemplateGatingConfig,
    /// Maximum retries per chunk on transient errors.
    #[serde(default = "default_max_retries")]
    pub max_retries: u8,
    /// Strip publisher boilerplate before chunking.
    #[serde(default = "default_strip_boilerplate")]
    pub strip_boilerplate: bool,
    /// Whether this is a built-in template (cannot be deleted).
    #[serde(default)]
    pub builtin: bool,
    /// Use SingleSession mode (entire text in one LLM context, per-chunk extraction turns).
    #[serde(default)]
    pub single_session: bool,
}

impl IngestionTemplate {
    /// Return a copy with API keys redacted.
    pub fn redacted(&self) -> Self {
        Self {
            passes: self
                .passes
                .iter()
                .map(|p| TemplatePassConfig {
                    llm: p.llm.redacted(),
                    ..p.clone()
                })
                .collect(),
            ..self.clone()
        }
    }

    /// Convert template to a legacy IngestionConfig for backward compatibility.
    pub fn to_ingestion_config(&self) -> IngestionConfig {
        let pass1 = self
            .passes
            .first()
            .map(|p| p.llm.clone())
            .unwrap_or(LlmConfig::None);
        let pass1_concurrency = self
            .passes
            .first()
            .map(|p| p.concurrency)
            .unwrap_or(default_pass1_concurrency());
        let pass2 = self
            .passes
            .get(1)
            .map(|p| p.llm.clone())
            .unwrap_or(LlmConfig::None);
        let pass2_window_size = self
            .passes
            .get(1)
            .map(|p| p.window_size)
            .unwrap_or(default_pass2_window_size());
        let pass2_window_overlap = self
            .passes
            .get(1)
            .map(|p| p.window_overlap)
            .unwrap_or(default_pass2_window_overlap());
        let mode = if self.single_session {
            PipelineMode::SingleSession
        } else if self.passes.len() > 1 {
            PipelineMode::Multi
        } else {
            PipelineMode::Single
        };

        IngestionConfig {
            mode,
            pass1,
            pass1_concurrency,
            pass2,
            pass2_window_size,
            pass2_window_overlap,
            chunk_strategy: self.chunking.strategy.clone(),
            chunk_max_tokens: self.chunking.max_tokens,
            chunk_overlap_tokens: self.chunking.overlap_tokens,
            context_window_tokens: self.chunking.context_window_tokens,
            chapter_regex: self.chunking.chapter_regex.clone(),
            strip_boilerplate: self.strip_boilerplate,
            auto_commit_threshold: self.gating.auto_commit_threshold,
            review_threshold: self.gating.review_threshold,
            enrich: true,
            session_max_context_tokens: self.chunking.context_window_tokens,
            ingestion_mode: self.mode.clone(),
            post_ingest_mamdani_rule_id: None,
        }
    }
}

/// Built-in template definitions.
pub fn builtin_templates() -> Vec<IngestionTemplate> {
    vec![
        IngestionTemplate {
            id: "novel".into(),
            name: "Novel / Literature".into(),
            mode: IngestionMode::Novel,
            passes: vec![TemplatePassConfig {
                llm: LlmConfig::None,
                concurrency: default_pass1_concurrency(),
                system_prompt: None,
                role: PassRole::Extraction,
                window_size: default_pass2_window_size(),
                window_overlap: default_pass2_window_overlap(),
            }],
            chunking: TemplateChunkConfig {
                strategy: crate::ingestion::chunker::ChunkStrategy::Chapter,
                context_window_tokens: 2_000_000,
                ..Default::default()
            },
            gating: Default::default(),
            max_retries: 3,
            strip_boilerplate: true,
            builtin: true,
            single_session: false,
        },
        IngestionTemplate {
            id: "news".into(),
            name: "Current Events / News".into(),
            mode: IngestionMode::News,
            passes: vec![TemplatePassConfig {
                llm: LlmConfig::None,
                concurrency: default_pass1_concurrency(),
                system_prompt: None,
                role: PassRole::Extraction,
                window_size: default_pass2_window_size(),
                window_overlap: default_pass2_window_overlap(),
            }],
            chunking: TemplateChunkConfig::default(),
            gating: Default::default(),
            max_retries: 3,
            strip_boilerplate: false,
            builtin: true,
            single_session: false,
        },
        IngestionTemplate {
            id: "intelligence".into(),
            name: "Intelligence / OSINT".into(),
            mode: IngestionMode::Intelligence,
            passes: vec![TemplatePassConfig {
                llm: LlmConfig::None,
                concurrency: default_pass1_concurrency(),
                system_prompt: None,
                role: PassRole::Extraction,
                window_size: default_pass2_window_size(),
                window_overlap: default_pass2_window_overlap(),
            }],
            chunking: TemplateChunkConfig::default(),
            gating: TemplateGatingConfig {
                auto_commit_threshold: 0.85,
                review_threshold: 0.4,
            },
            max_retries: 3,
            strip_boilerplate: false,
            builtin: true,
            single_session: false,
        },
        IngestionTemplate {
            id: "research".into(),
            name: "Research / Academic".into(),
            mode: IngestionMode::Research,
            passes: vec![TemplatePassConfig {
                llm: LlmConfig::None,
                concurrency: default_pass1_concurrency(),
                system_prompt: None,
                role: PassRole::Extraction,
                window_size: default_pass2_window_size(),
                window_overlap: default_pass2_window_overlap(),
            }],
            chunking: TemplateChunkConfig::default(),
            gating: Default::default(),
            max_retries: 3,
            strip_boilerplate: false,
            builtin: true,
            single_session: false,
        },
        single_pass_builtin(
            "temporal_events",
            "Temporal Events / Timelines",
            IngestionMode::TemporalEvents,
            TemplateGatingConfig::default(),
        ),
        single_pass_builtin(
            "legal",
            "Legal / Contracts",
            IngestionMode::Legal,
            TemplateGatingConfig {
                auto_commit_threshold: 0.85,
                review_threshold: 0.4,
            },
        ),
        single_pass_builtin(
            "financial",
            "Financial / Transactions",
            IngestionMode::Financial,
            TemplateGatingConfig {
                auto_commit_threshold: 0.9,
                review_threshold: 0.5,
            },
        ),
        single_pass_builtin(
            "medical",
            "Medical / Clinical",
            IngestionMode::Medical,
            TemplateGatingConfig {
                auto_commit_threshold: 0.85,
                review_threshold: 0.4,
            },
        ),
    ]
}

/// Helper: a single-extraction-pass builtin template with default chunking and no second pass.
fn single_pass_builtin(
    id: &str,
    name: &str,
    mode: IngestionMode,
    gating: TemplateGatingConfig,
) -> IngestionTemplate {
    IngestionTemplate {
        id: id.into(),
        name: name.into(),
        mode,
        passes: vec![TemplatePassConfig {
            llm: LlmConfig::None,
            concurrency: default_pass1_concurrency(),
            system_prompt: None,
            role: PassRole::Extraction,
            window_size: default_pass2_window_size(),
            window_overlap: default_pass2_window_overlap(),
        }],
        chunking: TemplateChunkConfig::default(),
        gating,
        max_retries: 3,
        strip_boilerplate: false,
        builtin: true,
        single_session: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = IngestionConfig::default();
        assert_eq!(cfg.mode, PipelineMode::Single);
        assert_eq!(cfg.pass1_concurrency, 4);
        assert_eq!(cfg.pass2_window_size, 20);
        assert_eq!(cfg.pass2_window_overlap, 3);
        assert_eq!(cfg.chunk_max_tokens, 2000);
        assert_eq!(cfg.chunk_overlap_tokens, 200);
        assert!((cfg.auto_commit_threshold - 0.8).abs() < f32::EPSILON);
        assert!((cfg.review_threshold - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = IngestionConfig {
            mode: PipelineMode::Multi,
            pass1: LlmConfig::Local {
                base_url: "http://localhost:11434".into(),
                model: "qwen3.5:4b".into(),
                api_key: None,
            },
            pass2: LlmConfig::Local {
                base_url: "http://localhost:11434".into(),
                model: "qwen3.5:27b".into(),
                api_key: None,
            },
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: IngestionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.mode, PipelineMode::Multi);
        assert_eq!(parsed.pass1_concurrency, 4);
    }

    #[test]
    fn test_redacted_masks_keys() {
        let cfg = IngestionConfig {
            pass1: LlmConfig::Local {
                base_url: "http://localhost:11434".into(),
                model: "qwen3.5:4b".into(),
                api_key: Some("secret-key".into()),
            },
            pass2: LlmConfig::Anthropic {
                api_key: "sk-ant-secret".into(),
                model: "claude-sonnet-4-20250514".into(),
            },
            ..Default::default()
        };
        let redacted = cfg.redacted();
        match &redacted.pass1 {
            LlmConfig::Local { api_key, .. } => assert_eq!(api_key.as_deref(), Some("***")),
            _ => panic!("Expected Local"),
        }
        match &redacted.pass2 {
            LlmConfig::Anthropic { api_key, .. } => assert_eq!(api_key, "***"),
            _ => panic!("Expected Anthropic"),
        }
    }
}
