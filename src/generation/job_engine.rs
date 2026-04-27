//! Inference-job adapter for the SE→fitness chapter generation loop.
//!
//! Wraps [`crate::generation::engine::GenerationEngine::generate_with_fitness`]
//! so it can be invoked through the standard async job pipeline
//! (`POST /jobs` → polling → `GET /jobs/:id/result`). The engine carries an
//! optional [`NarrativeExtractor`] resolved at startup; if unconfigured, jobs
//! fail with a clear "no LLM" error rather than a stack trace.

use std::sync::Arc;

use chrono::Utc;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::ingestion::llm::{ApiMessage, NarrativeExtractor};
use crate::narrative::writer_common::approx_tokens;
use crate::types::{InferenceJobType, JobStatus};
use crate::InferenceResult;

use super::engine::{ChapterGenerator, GenerationEngine};
use super::types::{GeneratedText, StyleTarget};

/// Adapter that lets a `NarrativeExtractor` satisfy [`ChapterGenerator`].
struct ExtractorChapterGenerator {
    extractor: Arc<dyn NarrativeExtractor>,
}

impl ChapterGenerator for ExtractorChapterGenerator {
    fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        _temperature: f64,
    ) -> Result<GeneratedText> {
        // Prefer multi-turn for providers that support it; the system + user
        // pair maps cleanly. Fall back to single-shot answer_question() for
        // providers without session support (Anthropic Messages API today).
        let text = if let Some(session) = self.extractor.as_session() {
            session.send_session_messages(&[
                ApiMessage {
                    role: "system".into(),
                    content: system_prompt.to_string(),
                },
                ApiMessage {
                    role: "user".into(),
                    content: user_prompt.to_string(),
                },
            ])?
        } else {
            self.extractor.answer_question(system_prompt, user_prompt)?
        };

        Ok(GeneratedText {
            prompt_tokens: approx_tokens(system_prompt) + approx_tokens(user_prompt),
            completion_tokens: approx_tokens(&text),
            text,
        })
    }
}

pub struct ChapterGenerationFitnessEngine {
    extractor: Option<Arc<dyn NarrativeExtractor>>,
}

impl ChapterGenerationFitnessEngine {
    pub fn new(extractor: Option<Arc<dyn NarrativeExtractor>>) -> Self {
        Self { extractor }
    }
}

impl InferenceEngine for ChapterGenerationFitnessEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ChapterGenerationFitness
    }

    fn estimate_cost(&self, job: &InferenceJob, hg: &Hypergraph) -> Result<u64> {
        crate::inference::cost::estimate_cost(job, hg)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let extractor = self.extractor.as_ref().ok_or_else(|| {
            TensaError::InferenceError(
                "Chapter generation requires a configured LLM. Set up a provider via /settings/llm or /settings/inference-llm.".into(),
            )
        })?;

        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InferenceError("narrative_id required".into()))?;

        let chapter = job
            .parameters
            .get("chapter")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| TensaError::InferenceError("chapter (usize) required".into()))?
            as usize;

        let style: StyleTarget = match job.parameters.get("style") {
            Some(v) => serde_json::from_value(v.clone())
                .map_err(|e| TensaError::InferenceError(format!("invalid style: {e}")))?,
            None => StyleTarget::default(),
        };

        let preceding_summaries: Vec<String> = job
            .parameters
            .get("preceding_summaries")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let engine = GenerationEngine::new(narrative_id.to_string());
        let generator = ExtractorChapterGenerator {
            extractor: extractor.clone(),
        };
        let (chapter_result, log) = engine.generate_with_fitness(
            hypergraph,
            chapter,
            &style,
            &generator,
            &preceding_summaries,
        )?;

        let attempts = chapter_result.attempts;
        let adherence = chapter_result.style_adherence;
        let result_json = serde_json::json!({
            "chapter": chapter_result,
            "log": log,
        });

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::ChapterGenerationFitness,
            target_id: job.target_id,
            result: result_json,
            confidence: adherence as f32,
            explanation: Some(format!(
                "Fitness loop ran {} attempt(s); best style adherence {:.3}",
                attempts, adherence
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn engine_without_llm_returns_clear_error() {
        let engine = ChapterGenerationFitnessEngine::new(None);
        let store = Arc::new(crate::store::memory::MemoryStore::new());
        let hg = Hypergraph::new(store);
        let job = InferenceJob {
            id: "test-job".into(),
            job_type: InferenceJobType::ChapterGenerationFitness,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({
                "narrative_id": "test",
                "chapter": 0,
            }),
            priority: crate::types::JobPriority::Normal,
            status: crate::types::JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        };
        let err = engine.execute(&job, &hg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("LLM"),
            "expected LLM-related error, got: {msg}"
        );
    }
}
