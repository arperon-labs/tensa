//! Text generation engine with fingerprint fitness loop (Sprint D9.7).
//!
//! Generates prose by querying the materialized hypergraph for each situation's
//! specification, evaluates against a target fingerprint, and re-ingests
//! generated text back into the hypergraph.
//!
//! Pipeline per chapter: Query → Prompt → Generate → Evaluate → Re-ingest → Next.

use chrono::Utc;

use crate::analysis::style_profile::prose_similarity;
use crate::analysis::stylometry::{compute_prose_features, prose_delta_to_constraints};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::narrative::cost_ledger;
use crate::types::CostOperation;

use super::prompt_builder;
use super::types::*;

/// Stable cost-ledger `kind` for fitness-loop iterations. Keeping the
/// cardinality bounded means existing summary/aggregation queries stay clean —
/// per-iteration data goes into the new `metadata` field instead.
pub const FITNESS_LEDGER_KIND: &str = "chapter_gen_fitness";

/// Trait implemented by anything that can produce chapter prose given a
/// system + user prompt and a temperature. Kept narrow on purpose so the
/// engine stays LLM-agnostic and tests can wire a mock without dragging
/// any real LLM client into the fixture.
pub trait ChapterGenerator: Send + Sync {
    fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        temperature: f64,
    ) -> Result<GeneratedText>;
}

/// The generation engine.
pub struct GenerationEngine {
    narrative_id: String,
}

impl GenerationEngine {
    /// Create a new engine for a materialized narrative.
    pub fn new(narrative_id: String) -> Self {
        Self { narrative_id }
    }

    /// Generate a single chapter.
    ///
    /// In the full pipeline this would:
    /// 1. Query the hypergraph for situation specs
    /// 2. Build a prompt via `prompt_builder`
    /// 3. Send to LLM and receive prose
    /// 4. Evaluate fingerprint fitness
    /// 5. Re-ingest accepted text back into the hypergraph
    ///
    /// Without an LLM client, this returns the structured prompt and a
    /// placeholder indicating generation is ready.
    pub fn prepare_chapter(
        &self,
        hg: &Hypergraph,
        chapter: usize,
        style: &StyleTarget,
        preceding_summaries: &[String],
    ) -> Result<(GenerationPrompt, GeneratedChapter)> {
        let prompt = prompt_builder::build_chapter_prompt(
            hg,
            &self.narrative_id,
            chapter,
            style,
            preceding_summaries,
        )?;

        // In production, this would send the prompt to an LLM.
        // For now, create a placeholder chapter.
        let rendered = prompt_builder::render_prompt(&prompt);

        let chapter_result = GeneratedChapter {
            chapter_number: chapter,
            text: format!(
                "[Generation ready — prompt built with {} situation specs, {} characters]\n\n{}",
                prompt.situation_specs.len(),
                prompt
                    .situation_specs
                    .iter()
                    .map(|s| s.character_contexts.len())
                    .sum::<usize>(),
                &rendered[..rendered.len().min(500)]
            ),
            attempts: 0,
            style_adherence: 0.0,
            commitment_fulfillments: Vec::new(),
            entities_improvised: Vec::new(),
        };

        Ok((prompt, chapter_result))
    }

    /// Re-ingest a generated (or edited) chapter's text back into the hypergraph.
    ///
    /// Stores the text at `text/{narrative_id}/chapter_{n}` and updates the
    /// situation's raw_content with the generated prose.
    pub fn store_chapter_text(&self, hg: &Hypergraph, chapter: usize, text: &str) -> Result<()> {
        let key = chapter_text_key(&self.narrative_id, chapter);
        hg.store().put(&key, text.as_bytes())
    }

    /// Load previously generated chapter text.
    pub fn load_chapter_text(&self, hg: &Hypergraph, chapter: usize) -> Result<Option<String>> {
        let key = chapter_text_key(&self.narrative_id, chapter);
        match hg.store().get(&key)? {
            Some(v) => Ok(Some(String::from_utf8_lossy(&v).into_owned())),
            None => Ok(None),
        }
    }

    /// Prepare the full narrative generation (all chapters sequentially).
    pub fn prepare_full_narrative(
        &self,
        hg: &Hypergraph,
        style: &StyleTarget,
        chapter_count: usize,
    ) -> Result<GenerationResult> {
        let mut chapters = Vec::new();
        let mut log = Vec::new();
        let mut preceding_summaries: Vec<String> = Vec::new();

        for ch in 0..chapter_count {
            let (prompt, chapter_result) =
                self.prepare_chapter(hg, ch, style, &preceding_summaries)?;

            log.push(GenerationLogEntry {
                chapter: ch,
                attempt: 1,
                accepted: true,
                deviations: Vec::new(),
                corrective_constraints: Vec::new(),
                timestamp: Utc::now(),
            });

            // Use the situation summary as the preceding summary for next chapter
            let summary = prompt
                .situation_specs
                .first()
                .map(|s| s.summary.clone())
                .unwrap_or_default();
            preceding_summaries.push(summary);

            chapters.push(chapter_result);
        }

        Ok(GenerationResult {
            narrative_id: self.narrative_id.clone(),
            chapters,
            generation_log: log,
            total_attempts: chapter_count,
            unfired_commitments: Vec::new(),
            knowledge_violations: Vec::new(),
        })
    }

    /// Regenerate a specific chapter with additional constraints.
    ///
    /// Must un-ingest the previous version first (remove improvised entities,
    /// revert knowledge state changes).
    pub fn prepare_regeneration(
        &self,
        hg: &Hypergraph,
        chapter: usize,
        style: &StyleTarget,
        additional_constraints: Vec<String>,
        preceding_summaries: &[String],
    ) -> Result<(GenerationPrompt, GeneratedChapter)> {
        // Remove previous chapter text
        let key = chapter_text_key(&self.narrative_id, chapter);
        let _ = hg.store().delete(&key);

        // Build prompt with additional constraints
        let mut prompt = prompt_builder::build_chapter_prompt(
            hg,
            &self.narrative_id,
            chapter,
            style,
            preceding_summaries,
        )?;
        prompt.constraints = additional_constraints;

        let rendered = prompt_builder::render_prompt(&prompt);
        let chapter_result = GeneratedChapter {
            chapter_number: chapter,
            text: format!(
                "[Regeneration ready — {} constraints applied]\n\n{}",
                prompt.constraints.len(),
                &rendered[..rendered.len().min(500)]
            ),
            attempts: 0,
            style_adherence: 0.0,
            commitment_fulfillments: Vec::new(),
            entities_improvised: Vec::new(),
        };

        Ok((prompt, chapter_result))
    }

    /// Handle author editing a chapter: re-ingest the edited text.
    pub fn continue_from_edit(
        &self,
        hg: &Hypergraph,
        chapter: usize,
        edited_text: &str,
    ) -> Result<()> {
        self.store_chapter_text(hg, chapter, edited_text)
    }

    /// The fitness-driven loop. Generates a chapter, scores its prose against
    /// `style.target_fingerprint.prose`, and revises iteratively (up to
    /// `style.max_retries_per_chapter` attempts). Returns the **best** attempt
    /// across iterations — never the last by definition, since the LLM can
    /// overcorrect. When `style.target_fingerprint` is `None` this collapses
    /// to a single-shot generation with `style_adherence = 1.0`.
    ///
    /// Each iteration writes one cost-ledger entry with stable
    /// `kind = FITNESS_LEDGER_KIND`; iteration index and score live in the
    /// entry's `metadata` field so aggregate queries on `kind` stay clean.
    pub fn generate_with_fitness(
        &self,
        hg: &Hypergraph,
        chapter: usize,
        style: &StyleTarget,
        generator: &dyn ChapterGenerator,
        preceding_summaries: &[String],
    ) -> Result<(GeneratedChapter, Vec<GenerationLogEntry>)> {
        let max_attempts = style.max_retries_per_chapter.max(1);
        let threshold = style.fitness_threshold.into_inner();

        // Build the prompt scaffold once: situation specs, character contexts,
        // chapter context, system prompt — none of these change across
        // iterations. Only `constraints` and the optional prior-attempt
        // prepend differ, so per-iteration cost stays constant rather than
        // re-scanning the hypergraph each round.
        let base_prompt = prompt_builder::build_chapter_prompt(
            hg,
            &self.narrative_id,
            chapter,
            style,
            preceding_summaries,
        )?;
        let system = prompt_builder::render_system(&base_prompt);

        let mut log: Vec<GenerationLogEntry> = Vec::new();
        let mut prior_attempt: Option<GeneratedText> = None;
        let mut best: Option<(GeneratedText, f64)> = None;
        let mut next_constraints: Vec<String> = Vec::new();

        for attempt in 0..max_attempts {
            let mut iter_prompt = base_prompt.clone();
            iter_prompt.constraints = next_constraints.clone();

            let mut user = prompt_builder::render_user(&iter_prompt);
            if let Some(prev) = &prior_attempt {
                user = format!(
                    "Revise the following draft to address the issues below:\n\n{}\n\n{}",
                    prev.text, user
                );
            }

            let start = std::time::Instant::now();
            let generated = generator.generate(&system, &user, style.temperature)?;
            let duration_ms = start.elapsed().as_millis() as u64;

            // Single-shot path: no target fingerprint → accept immediately.
            if style.target_fingerprint.is_none() {
                self.write_ledger_entry(hg, &generated, duration_ms, attempt, 1.0);
                log.push(GenerationLogEntry {
                    chapter,
                    attempt,
                    accepted: true,
                    deviations: Vec::new(),
                    corrective_constraints: Vec::new(),
                    timestamp: Utc::now(),
                });
                return Ok((self.build_chapter(chapter, generated.text, 1, 1.0), log));
            }

            // Scoring path. Unwrap is safe — None case returned above.
            let target = style.target_fingerprint.as_ref().unwrap();
            let actual_prose = compute_prose_features(&generated.text);
            let score = prose_similarity(&target.prose, &actual_prose) as f64;
            let deviations = prose_delta_to_constraints(&target.prose, &actual_prose);
            let accepted = score >= threshold;

            self.write_ledger_entry(hg, &generated, duration_ms, attempt, score);

            // Constraints fed to the *next* iteration mirror this iteration's
            // deviations; the log keeps them semantically separate so callers
            // can audit the loop.
            let next = deviations.clone();
            log.push(GenerationLogEntry {
                chapter,
                attempt,
                accepted,
                deviations,
                corrective_constraints: if accepted { Vec::new() } else { next.clone() },
                timestamp: Utc::now(),
            });

            // Best-so-far tracking. Strict greater-than only — ties don't
            // churn the prior_attempt context.
            if best.as_ref().is_none_or(|(_, s)| score > *s) {
                best = Some((generated.clone(), score));
            }

            if accepted {
                break;
            }

            prior_attempt = Some(generated);
            next_constraints = next;
        }

        // best is always Some — the loop ran at least once.
        let (best_text, best_score) = best.expect("fitness loop ran but produced no best");
        Ok((
            self.build_chapter(chapter, best_text.text, log.len(), best_score),
            log,
        ))
    }

    fn write_ledger_entry(
        &self,
        hg: &Hypergraph,
        generated: &GeneratedText,
        duration_ms: u64,
        attempt: usize,
        score: f64,
    ) {
        cost_ledger::record(
            hg.store(),
            &self.narrative_id,
            CostOperation::Generation,
            FITNESS_LEDGER_KIND,
            generated.prompt_tokens,
            generated.completion_tokens,
            None,
            false,
            true,
            duration_ms,
            Some(serde_json::json!({"iteration": attempt, "score": score})),
        );
    }

    fn build_chapter(
        &self,
        chapter: usize,
        text: String,
        attempts: usize,
        adherence: f64,
    ) -> GeneratedChapter {
        let _ = self;
        GeneratedChapter {
            chapter_number: chapter,
            text,
            attempts,
            style_adherence: adherence,
            commitment_fulfillments: Vec::new(),
            entities_improvised: Vec::new(),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{materializer, planner};
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn materialize_test_narrative(hg: &Hypergraph) -> (String, usize) {
        let config = PlanConfig {
            premise: "A detective investigates".into(),
            chapter_count: 3,
            ..PlanConfig::default()
        };
        let plan = planner::generate_plan(config).unwrap();
        let report = materializer::materialize_plan(hg, &plan).unwrap();
        (report.narrative_id, 3)
    }

    #[test]
    fn test_prepare_chapter() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);

        let engine = GenerationEngine::new(nid);
        let style = StyleTarget::default();
        let (prompt, chapter) = engine.prepare_chapter(&hg, 0, &style, &[]).unwrap();

        assert!(!prompt.system_prompt.is_empty());
        assert!(!prompt.situation_specs.is_empty());
        assert_eq!(chapter.chapter_number, 0);
    }

    #[test]
    fn test_prepare_full_narrative() {
        let hg = test_hg();
        let (nid, ch_count) = materialize_test_narrative(&hg);

        let engine = GenerationEngine::new(nid);
        let style = StyleTarget::default();
        let result = engine
            .prepare_full_narrative(&hg, &style, ch_count)
            .unwrap();

        assert_eq!(result.chapters.len(), 3);
        assert_eq!(result.generation_log.len(), 3);
    }

    #[test]
    fn test_store_and_load_chapter_text() {
        let hg = test_hg();
        let engine = GenerationEngine::new("test-narrative".into());

        engine
            .store_chapter_text(&hg, 0, "It was a dark and stormy night.")
            .unwrap();
        let loaded = engine.load_chapter_text(&hg, 0).unwrap().unwrap();
        assert_eq!(loaded, "It was a dark and stormy night.");

        // Non-existent chapter returns None
        assert!(engine.load_chapter_text(&hg, 99).unwrap().is_none());
    }

    #[test]
    fn test_continue_from_edit() {
        let hg = test_hg();
        let engine = GenerationEngine::new("edit-test".into());

        // Store original
        engine.store_chapter_text(&hg, 0, "Original text").unwrap();

        // Edit
        engine
            .continue_from_edit(&hg, 0, "Edited text by author")
            .unwrap();

        let loaded = engine.load_chapter_text(&hg, 0).unwrap().unwrap();
        assert_eq!(loaded, "Edited text by author");
    }

    #[test]
    fn test_prepare_regeneration() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);

        let engine = GenerationEngine::new(nid);
        let style = StyleTarget::default();

        let (prompt, _) = engine
            .prepare_regeneration(
                &hg,
                0,
                &style,
                vec!["More dialogue, less description".into()],
                &[],
            )
            .unwrap();

        assert_eq!(prompt.constraints.len(), 1);
        assert!(prompt.constraints[0].contains("dialogue"));
    }

    // ─── Fitness Loop Tests ──────────────────────────────────

    use crate::analysis::style_profile::NarrativeFingerprint;
    use crate::analysis::stylometry::{compute_prose_features, ProseStyleFeatures};
    use std::cell::RefCell;
    use std::sync::Mutex;

    /// Mock generator: returns a sequence of texts (one per call), recording
    /// every prompt it received for assertions.
    struct ScriptedGenerator {
        outputs: Mutex<Vec<&'static str>>,
        captured: Mutex<Vec<(String, String)>>,
    }
    impl ScriptedGenerator {
        fn new(outputs: Vec<&'static str>) -> Self {
            Self {
                outputs: Mutex::new(outputs),
                captured: Mutex::new(Vec::new()),
            }
        }
    }
    impl ChapterGenerator for ScriptedGenerator {
        fn generate(&self, system: &str, user: &str, _temp: f64) -> Result<GeneratedText> {
            self.captured
                .lock()
                .unwrap()
                .push((system.to_string(), user.to_string()));
            let mut outs = self.outputs.lock().unwrap();
            let text = if outs.is_empty() {
                "fallback text"
            } else {
                outs.remove(0)
            };
            Ok(GeneratedText {
                text: text.to_string(),
                prompt_tokens: 100,
                completion_tokens: 50,
            })
        }
    }

    fn fingerprint_with_prose(prose: ProseStyleFeatures) -> NarrativeFingerprint {
        use crate::analysis::style_profile::NarrativeStyleProfile;
        NarrativeFingerprint {
            narrative_id: "test-target".into(),
            computed_at: Utc::now(),
            prose,
            structure: NarrativeStyleProfile {
                narrative_id: "test-target".into(),
                computed_at: Utc::now(),
                ..Default::default()
            },
        }
    }

    /// Short, punchy text — high dialogue, short sentences.
    const SHORT_TEXT: &str =
        "\"Stop.\" He froze. The door creaked. \"Who's there?\" \"Me.\" Silence stretched.";
    /// Long, dense text — narration-heavy, long sentences.
    const LONG_TEXT: &str = "The labyrinthine corridors of the ancient mansion stretched before him in an unbroken procession of marble and shadow, each successive doorway promising revelation yet delivering only further mystery. He walked slowly, methodically, considering every architectural flourish.";

    #[test]
    fn generate_with_fitness_no_target_is_single_shot() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);
        let engine = GenerationEngine::new(nid);
        let gen = ScriptedGenerator::new(vec!["one and done"]);

        let style = StyleTarget {
            target_fingerprint: None,
            max_retries_per_chapter: 5,
            ..StyleTarget::default()
        };
        let (chapter, log) = engine
            .generate_with_fitness(&hg, 0, &style, &gen, &[])
            .unwrap();
        assert_eq!(chapter.attempts, 1);
        assert_eq!(chapter.style_adherence, 1.0);
        assert_eq!(chapter.text, "one and done");
        assert_eq!(log.len(), 1);
        assert_eq!(gen.captured.lock().unwrap().len(), 1);
    }

    #[test]
    fn generate_with_fitness_converges() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);
        let engine = GenerationEngine::new(nid);
        // Target = fingerprint of SHORT_TEXT itself; first attempt is far,
        // second attempt is the target text → score should jump above any
        // reasonable threshold.
        let target = fingerprint_with_prose(compute_prose_features(SHORT_TEXT));
        let gen = ScriptedGenerator::new(vec![LONG_TEXT, SHORT_TEXT]);

        let style = StyleTarget {
            target_fingerprint: Some(target),
            fitness_threshold: Threshold::new(0.99).unwrap(),
            max_retries_per_chapter: 3,
            ..StyleTarget::default()
        };
        let (chapter, log) = engine
            .generate_with_fitness(&hg, 0, &style, &gen, &[])
            .unwrap();
        // Iteration 1 hits the target exactly → accepted, loop exits.
        assert_eq!(chapter.text, SHORT_TEXT);
        assert!(chapter.style_adherence > 0.99);
        assert_eq!(log.len(), 2);
        assert!(log[1].accepted);
        assert!(!log[0].accepted);
    }

    #[test]
    fn generate_with_fitness_returns_best_not_last() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);
        let engine = GenerationEngine::new(nid);
        // Target = a perturbed SHORT_TEXT fingerprint so iteration-0 SHORT
        // scores close-but-not-perfect (under any plausible threshold), and
        // LONG scores far. Outputs in order: SHORT (close), LONG (far),
        // LONG (far). Best must be the iteration-0 SHORT text even though
        // the loop continues to a worse final attempt.
        let mut target_prose = compute_prose_features(SHORT_TEXT);
        target_prose.dialogue_ratio = 0.95; // SHORT_TEXT actual is much lower
        target_prose.avg_sentence_length = 3.0;
        let target = fingerprint_with_prose(target_prose);
        let gen = ScriptedGenerator::new(vec![SHORT_TEXT, LONG_TEXT, LONG_TEXT]);

        let style = StyleTarget {
            target_fingerprint: Some(target),
            // High enough that no attempt accepts → loop runs all 3.
            fitness_threshold: Threshold::new(0.99).unwrap(),
            max_retries_per_chapter: 3,
            ..StyleTarget::default()
        };
        let (chapter, log) = engine
            .generate_with_fitness(&hg, 0, &style, &gen, &[])
            .unwrap();
        assert_eq!(log.len(), 3, "loop should run all 3 attempts");
        assert!(
            log.iter().all(|e| !e.accepted),
            "no attempt should meet the high threshold"
        );
        assert_eq!(
            chapter.text, SHORT_TEXT,
            "best attempt was iteration 0, must be returned"
        );
    }

    #[test]
    fn generate_with_fitness_caps_at_max_retries() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);
        let engine = GenerationEngine::new(nid);
        let target = fingerprint_with_prose(compute_prose_features(SHORT_TEXT));
        let gen = ScriptedGenerator::new(vec![LONG_TEXT, LONG_TEXT, LONG_TEXT, LONG_TEXT]);

        let style = StyleTarget {
            target_fingerprint: Some(target),
            fitness_threshold: Threshold::new(0.99).unwrap(),
            max_retries_per_chapter: 3,
            ..StyleTarget::default()
        };
        let (chapter, log) = engine
            .generate_with_fitness(&hg, 0, &style, &gen, &[])
            .unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(chapter.attempts, 3);
        assert!(log.iter().all(|e| !e.accepted));
        assert_eq!(gen.captured.lock().unwrap().len(), 3);
    }

    #[test]
    fn generate_with_fitness_revision_includes_prior_text() {
        let hg = test_hg();
        let (nid, _) = materialize_test_narrative(&hg);
        let engine = GenerationEngine::new(nid);
        let target = fingerprint_with_prose(compute_prose_features(SHORT_TEXT));
        // Two distinguishable outputs so we can find the first inside the
        // second iteration's user prompt.
        let _: RefCell<()> = RefCell::new(());
        let gen = ScriptedGenerator::new(vec!["first iteration text marker", LONG_TEXT]);

        let style = StyleTarget {
            target_fingerprint: Some(target),
            fitness_threshold: Threshold::new(0.99).unwrap(),
            max_retries_per_chapter: 2,
            ..StyleTarget::default()
        };
        let _ = engine
            .generate_with_fitness(&hg, 0, &style, &gen, &[])
            .unwrap();
        let captured = gen.captured.lock().unwrap();
        assert_eq!(captured.len(), 2);
        let second_user_prompt = &captured[1].1;
        assert!(
            second_user_prompt.contains("first iteration text marker"),
            "iteration 1 user prompt missing prior attempt text:\n{second_user_prompt}"
        );
        assert!(
            second_user_prompt.starts_with("Revise the following draft"),
            "iteration 1 should lead with the revise framing"
        );
    }
}
