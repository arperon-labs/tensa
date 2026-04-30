//! McpBackend trait — abstraction over embedded and HTTP modes.
//!
//! All methods accept and return `serde_json::Value` to keep
//! the trait surface minimal. The embedded backend deserializes
//! internally; the HTTP backend forwards JSON directly.

use crate::error::Result;
use serde_json::Value;

/// Backend abstraction for MCP tool handlers.
///
/// Two implementations:
/// - `EmbeddedBackend`: Direct library access via Hypergraph + stores
/// - `HttpBackend`: HTTP client calling the TENSA REST API
pub trait McpBackend: Send + Sync {
    /// Execute a TensaQL MATCH query (instant results).
    fn execute_query(
        &self,
        tensaql: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit a TensaQL INFER or DISCOVER query (returns job descriptor).
    fn submit_inference_query(
        &self,
        tensaql: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create an entity from JSON data.
    fn create_entity(&self, data: Value)
        -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get an entity by UUID string.
    fn get_entity(&self, id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create a situation from JSON data.
    fn create_situation(
        &self,
        data: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get a situation by UUID string.
    fn get_situation(&self, id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Add a participant (entity → situation link) from JSON data.
    fn add_participant(
        &self,
        data: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ingest raw text through the LLM extraction pipeline.
    fn ingest_text(
        &self,
        text: &str,
        narrative_id: &str,
        source: &str,
        auto_commit_threshold: Option<f32>,
        review_threshold: Option<f32>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List all narrative metadata.
    fn list_narratives(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create narrative metadata from JSON data.
    fn create_narrative(
        &self,
        data: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get the status of an inference job.
    fn get_job_status(
        &self,
        job_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get the result of a completed inference job.
    fn get_job_result(
        &self,
        job_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create a registered source with trust and bias metadata.
    fn create_source(&self, data: Value)
        -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get a source by UUID string.
    fn get_source(&self, id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List all registered sources.
    fn list_sources(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Add a source attribution linking a source to an entity or situation.
    fn add_attribution(
        &self,
        data: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List contentions involving a situation.
    fn list_contentions(
        &self,
        situation_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Recompute confidence breakdown for an entity or situation.
    fn recompute_confidence(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Manage the validation queue (list/get/approve/reject/edit).
    fn review_queue(
        &self,
        action: &str,
        item_id: Option<&str>,
        reviewer: Option<&str>,
        notes: Option<&str>,
        edited_data: Option<Value>,
        limit: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get a comprehensive actor profile: entity data, participations, state history.
    fn get_actor_profile(
        &self,
        actor_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Delete an entity by UUID (soft delete).
    fn delete_entity(&self, id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Delete a situation by UUID (soft delete).
    fn delete_situation(&self, id: &str)
        -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Update entity properties by UUID.
    fn update_entity(
        &self,
        id: &str,
        updates: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List entities, optionally filtered by type, narrative, and limit.
    fn list_entities(
        &self,
        entity_type: Option<&str>,
        narrative_id: Option<&str>,
        limit: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Merge two entities — absorb one into the other.
    fn merge_entities(
        &self,
        keep_id: &str,
        absorb_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Export a narrative in the specified format.
    fn export_narrative(
        &self,
        narrative_id: &str,
        format: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get per-narrative statistics (entity/situation/participation counts).
    ///
    /// Phase 11 fuzzy-surface extension: optional `tnorm` + `aggregator`
    /// ride through to the REST endpoint as query-string params
    /// (`?tnorm=<kind>&aggregator=<kind>`). When both are `None` the URL
    /// is bit-identical to the pre-sprint shape.
    fn get_narrative_stats(
        &self,
        narrative_id: &str,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Search entities by text query across properties.
    ///
    /// Phase 11 fuzzy-surface extension: optional `tnorm` + `aggregator`
    /// forwarded to the internal TensaQL routes. `None` path preserves
    /// pre-sprint URL shape bit-identically.
    fn search_entities(
        &self,
        query: &str,
        limit: Option<usize>,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ingest text from a URL (fetch, strip HTML, run pipeline).
    fn ingest_url(
        &self,
        url: &str,
        narrative_id: &str,
        source_name: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ingest items from an RSS/Atom feed.
    fn ingest_rss(
        &self,
        feed_url: &str,
        narrative_id: &str,
        max_items: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Split an entity: clone it and move specified situation participations to the clone.
    fn split_entity(
        &self,
        entity_id: &str,
        situation_ids: &[String],
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Restore a soft-deleted entity by clearing its deleted_at timestamp.
    fn restore_entity(&self, id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Restore a soft-deleted situation by clearing its deleted_at timestamp.
    fn restore_situation(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create a project container.
    fn create_project(
        &self,
        data: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get a project by its slug ID.
    fn get_project(&self, id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List all projects, optionally limited.
    fn list_projects(
        &self,
        limit: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Update a project by its slug ID.
    fn update_project(
        &self,
        id: &str,
        updates: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Delete a project by its slug ID, optionally cascading.
    fn delete_project(
        &self,
        id: &str,
        cascade: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get LLM response cache statistics (entries and total bytes).
    fn cache_stats(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Run the per-narrative bulk-analysis tier suite (the headless equivalent
    /// of Studio's "Run Full Analysis" button). Submits every algorithmic
    /// inference job in the requested tiers, skipping any rows whose
    /// analysis-status entry is `locked: true` unless `force = true`.
    /// Returns `{ submitted, skipped_locked, entities, situations, actors }`.
    fn run_full_analysis(
        &self,
        narrative_id: &str,
        tiers: Option<Vec<String>>,
        force: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Generate embeddings for entities + situations missing them. Optional
    /// `narrative_id` scopes to one narrative (otherwise sweeps all `Candidate`
    /// rows). `force = true` re-embeds rows that already have an embedding.
    /// Requires an embedding provider to be configured server-side.
    fn backfill_embeddings(
        &self,
        narrative_id: Option<&str>,
        force: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ask a natural language question with RAG (Retrieval-Augmented Generation).
    fn ask(
        &self,
        question: &str,
        narrative_id: Option<&str>,
        mode: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Tune extraction prompts for a narrative by sampling its chunks and generating
    /// domain-adapted guidelines via LLM.
    fn tune_prompts(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get community hierarchy for a narrative, optionally filtered by level.
    fn community_hierarchy(
        &self,
        narrative_id: &str,
        level: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Export one or more narratives as a .tensa archive (returns base64-encoded bytes).
    fn export_archive(
        &self,
        narrative_ids: Vec<String>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Import a .tensa archive from base64-encoded bytes.
    fn import_archive(
        &self,
        data_base64: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Verify whether two texts were likely written by the same author
    /// (PAN@CLEF authorship verification). Returns `{score, decision}`.
    fn verify_authorship(
        &self,
        text_a: &str,
        text_b: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Narrative Debugger (Sprint D10) ─────────────────────

    /// Run full structural diagnosis on a narrative.
    fn diagnose_narrative(
        &self,
        narrative_id: &str,
        genre: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get narrative health score + summary counts.
    fn get_health_score(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Run auto-repair loop (diagnose + apply safe fixes).
    fn auto_repair(
        &self,
        narrative_id: &str,
        max_severity: Option<&str>,
        max_iterations: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Narrative Adaptation (Sprint D11) ────────────────

    /// Compute essentiality scores for all situations, entities, and subplots.
    fn compute_essentiality(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Produce a compression plan (preset: novella | short_story | screenplay_outline).
    fn compress_narrative(
        &self,
        narrative_id: &str,
        preset: Option<&str>,
        target_chapters: Option<usize>,
        target_ratio: Option<f64>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Produce an expansion plan.
    fn expand_narrative(
        &self,
        narrative_id: &str,
        target_chapters: usize,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Diff two narratives structurally.
    fn diff_narratives(
        &self,
        narrative_a: &str,
        narrative_b: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Disinfo Fingerprints (Sprint D1) ─────────────────────

    /// Load (or compute) the behavioral fingerprint for an actor.
    #[cfg(feature = "disinfo")]
    fn get_behavioral_fingerprint(
        &self,
        actor_id: &str,
        recompute: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Load (or compute) the disinformation fingerprint for a narrative.
    #[cfg(feature = "disinfo")]
    fn get_disinfo_fingerprint(
        &self,
        narrative_id: &str,
        recompute: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Compare two behavioral or disinfo fingerprints. Returns composite
    /// distance, per-axis distances, p-value, 95% CI, and same-source verdict.
    #[cfg(feature = "disinfo")]
    fn compare_fingerprints(
        &self,
        kind: &str,
        task: Option<&str>,
        a_id: &str,
        b_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Spread Dynamics (Sprint D2) ─────────────────────────

    /// Run SMIR + per-platform R₀ + cross-platform jumps + velocity-monitor
    /// alert check. Returns the aggregated payload.
    #[cfg(feature = "disinfo")]
    fn estimate_r0_by_platform(
        &self,
        narrative_id: &str,
        fact: &str,
        about_entity: &str,
        narrative_kind: Option<&str>,
        beta_overrides: Option<std::collections::HashMap<String, f64>>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Counterfactual intervention projection (RemoveTopAmplifiers / DebunkAt).
    #[cfg(feature = "disinfo")]
    fn simulate_intervention(
        &self,
        narrative_id: &str,
        fact: &str,
        about_entity: &str,
        intervention: Value,
        beta_overrides: Option<std::collections::HashMap<String, f64>>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── CIB Detection (Sprint D3) ───────────────────────────

    /// Run Coordinated Inauthentic Behavior detection on a narrative.
    /// Returns clusters, evidence, network stats.
    #[cfg(feature = "disinfo")]
    fn detect_cib_cluster(
        &self,
        narrative_id: &str,
        cross_platform: bool,
        similarity_threshold: Option<f64>,
        alpha: Option<f64>,
        bootstrap_iter: Option<usize>,
        min_cluster_size: Option<usize>,
        seed: Option<u64>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Rank top-N actors by graph centrality. `method` is
    /// `"pagerank" | "eigenvector" | "harmonic"`.
    #[cfg(feature = "disinfo")]
    fn rank_superspreaders(
        &self,
        narrative_id: &str,
        method: Option<&str>,
        top_n: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Claims & Fact-Checks (Sprint D4) ────────────────────

    /// Detect claims in text and persist them.
    #[cfg(feature = "disinfo")]
    fn ingest_claim(
        &self,
        text: &str,
        narrative_id: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ingest a fact-check for a claim, creating an argumentation attack.
    #[cfg(feature = "disinfo")]
    fn ingest_fact_check(
        &self,
        claim_id: &str,
        verdict: &str,
        source: &str,
        url: Option<&str>,
        language: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Match claims against known fact-checks.
    #[cfg(feature = "disinfo")]
    fn fetch_fact_checks(
        &self,
        claim_id: &str,
        min_similarity: f64,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Trace a claim back to its earliest appearance.
    #[cfg(feature = "disinfo")]
    fn trace_claim_origin(
        &self,
        claim_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Classify an actor into adversarial archetypes.
    #[cfg(feature = "disinfo")]
    fn classify_archetype(
        &self,
        actor_id: &str,
        force: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Fuse disinfo signals via Dempster-Shafer combination.
    #[cfg(feature = "disinfo")]
    fn assess_disinfo(
        &self,
        target_id: &str,
        signals: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ingest a social media post as a situation with platform metadata.
    #[cfg(feature = "disinfo")]
    fn ingest_post(
        &self,
        text: &str,
        actor_id: &str,
        narrative_id: &str,
        platform: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Ingest an actor entity with disinfo-relevant properties.
    #[cfg(feature = "disinfo")]
    fn ingest_actor(
        &self,
        name: &str,
        narrative_id: &str,
        platform: Option<&str>,
        properties: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Link an entity to a narrative by setting its narrative_id.
    #[cfg(feature = "disinfo")]
    fn link_narrative(
        &self,
        entity_id: &str,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Multilingual & Export (Sprint D6) ────────────────────

    /// Detect the language of a text snippet using Unicode script heuristics.
    #[cfg(feature = "disinfo")]
    fn detect_language(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Export a narrative as a MISP event JSON.
    #[cfg(feature = "disinfo")]
    fn export_misp_event(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Export narrative entities as Maltego transform results.
    #[cfg(feature = "disinfo")]
    fn export_maltego(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Generate a comprehensive disinfo analysis Markdown report.
    #[cfg(feature = "disinfo")]
    fn generate_disinfo_report(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Scheduler + Reports + Health (Sprint D8) ────────────────

    /// List all scheduled tasks.
    #[cfg(feature = "disinfo")]
    fn list_scheduled_tasks(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create a scheduled task with type and schedule.
    #[cfg(feature = "disinfo")]
    fn create_scheduled_task(
        &self,
        task_type: &str,
        schedule: &str,
        config: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Trigger immediate execution of a scheduled task.
    #[cfg(feature = "disinfo")]
    fn run_task_now(
        &self,
        task_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List discovery candidates (potential new sources).
    #[cfg(feature = "disinfo")]
    fn list_discovery_candidates(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Trigger fact-check database sync.
    #[cfg(feature = "disinfo")]
    fn sync_fact_checks(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Generate a situation report for a time window.
    #[cfg(feature = "disinfo")]
    fn generate_situation_report(
        &self,
        hours: u64,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Adversarial Wargaming (Sprint D12) ──────────────────────

    /// Generate an adversary policy for an actor or archetype.
    fn generate_adversary_policy(
        &self,
        narrative_id: &str,
        actor_id: Option<&str>,
        archetype: Option<&str>,
        lambda: Option<f64>,
        lambda_cap: Option<f64>,
        reward_weights: Option<Vec<f64>>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Configure rationality model parameters.
    fn configure_rationality(
        &self,
        model: &str,
        lambda: Option<f64>,
        lambda_cap: Option<f64>,
        tau: Option<f64>,
        feature_weights: Option<Vec<f64>>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create a wargame session.
    fn create_wargame(
        &self,
        narrative_id: &str,
        max_turns: Option<usize>,
        time_step_minutes: Option<u64>,
        auto_red: Option<bool>,
        auto_blue: Option<bool>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit moves for one wargame turn.
    fn submit_wargame_move(
        &self,
        session_id: &str,
        red_moves: Option<Value>,
        blue_moves: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get the current state of a wargame session.
    fn get_wargame_state(
        &self,
        session_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Auto-play N turns of a wargame session.
    fn auto_play_wargame(
        &self,
        session_id: &str,
        num_turns: usize,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Writer Workflows (Sprint W6) ────────────────────────────

    /// Get the narrative plan for a narrative (returns `null` if not set).
    fn get_narrative_plan(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Upsert a narrative plan. `patch` is a full or partial plan object.
    fn upsert_narrative_plan(
        &self,
        narrative_id: &str,
        patch: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get the writer workspace summary (counts, plan status, next-step suggestions).
    fn get_writer_workspace(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Run a workshop pass (cheap | standard | deep) over the narrative.
    fn run_workshop(
        &self,
        narrative_id: &str,
        tier: &str,
        focuses: Option<Vec<String>>,
        max_llm_calls: Option<u32>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List pinned facts for a narrative (continuity anchors).
    fn list_pinned_facts(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create a pinned continuity fact. `fact` should contain label + value + optional details.
    fn create_pinned_fact(
        &self,
        narrative_id: &str,
        fact: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Check a block of prose against pinned facts; returns continuity warnings.
    fn check_continuity(
        &self,
        narrative_id: &str,
        prose: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List narrative revisions (git-like history) newest-first.
    fn list_narrative_revisions(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Restore a narrative to a previous revision. Auto-commits current state
    /// first as a safety net, then replays the target revision's snapshot.
    fn restore_narrative_revision(
        &self,
        narrative_id: &str,
        revision_id: &str,
        author: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Summarize writer cost ledger (tokens, LLM calls) for a narrative over a time window.
    fn get_writer_cost_summary(
        &self,
        narrative_id: &str,
        window: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Replace a situation's `raw_content` with new prose. Accepts either a
    /// plain string (wrapped as one Text block) or a pre-shaped array of
    /// `ContentBlock` JSON objects. Optional `status` stamps the writer
    /// workflow status in the same call. Used by the Workflow 2 co-writing
    /// loop to write assistant-authored prose back into the hypergraph.
    fn set_situation_content(
        &self,
        situation_id: &str,
        content: Value,
        status: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Package per-scene drafting context in a single round-trip: plan
    /// summary, pinned facts, POV character profile (if requested), current
    /// situation, and N preceding scenes. Reduces the per-scene MCP call
    /// count from ~5 to 1 during assistant-as-author drafting.
    fn get_scene_context(
        &self,
        narrative_id: &str,
        situation_id: Option<&str>,
        pov_entity_id: Option<&str>,
        lookback_scenes: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit a `ChapterGenerationFitness` inference job that runs the
    /// LLM-driven SE→fitness retry loop. Returns `{job_id, status}`; the
    /// caller polls `job_status` / `job_result` to retrieve the result.
    #[cfg(feature = "generation")]
    #[allow(clippy::too_many_arguments)]
    fn generate_chapter_with_fitness(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
        style_embedding_id: Option<&str>,
        target_fingerprint_source: Option<&str>,
        fitness_threshold: Option<f64>,
        max_retries: Option<usize>,
        temperature: Option<f64>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Narrative Architecture (Sprint D9 / W14) ────────────────

    /// Detect narrative commitments (Chekhov's guns, foreshadowing, dramatic questions).
    fn detect_commitments(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Per-chapter promise rhythm (tension curve + fulfillment ratio).
    fn get_commitment_rhythm(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Extract fabula (chronological event ordering) from Allen constraints.
    fn extract_fabula(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Extract sjužet (discourse/telling order) from the chapter structure.
    fn extract_sjuzet(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Suggest alternative sjužet orderings that maximize dramatic effect.
    fn suggest_reordering(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Compute the dramatic irony map (reader vs. character knowledge gaps).
    fn compute_dramatic_irony(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Detect focalization × irony interactions across the narrative.
    fn detect_focalization(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Detect character arc(s). If `character_id` is `None`, returns all stored arcs.
    fn detect_character_arc(
        &self,
        narrative_id: &str,
        character_id: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Detect subplots via community detection on the situation graph.
    fn detect_subplots(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Classify scene-sequel rhythm (Swain/Bickham) for the narrative.
    fn classify_scene_sequel(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Generate a formal narrative plan from a premise. Returns the stored plan.
    #[cfg(feature = "generation")]
    fn generate_narrative_plan(
        &self,
        premise: &str,
        genre: &str,
        chapter_count: usize,
        subplot_count: usize,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Materialize a stored plan into the hypergraph.
    #[cfg(feature = "generation")]
    fn materialize_plan(
        &self,
        plan_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Validate a materialized narrative's temporal / causal / commitment consistency.
    #[cfg(feature = "generation")]
    fn validate_materialized_narrative(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Prepare a chapter generation prompt (no LLM call). Returns `{prompt, chapter}`.
    #[cfg(feature = "generation")]
    fn generate_chapter(
        &self,
        narrative_id: &str,
        chapter: usize,
        voice_description: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Prepare full narrative generation (all chapters sequentially, no LLM call).
    #[cfg(feature = "generation")]
    fn generate_narrative(
        &self,
        narrative_id: &str,
        chapter_count: usize,
        voice_description: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Sprint W15: Writer MCP bridge ────────────────────────

    // Annotations
    #[allow(clippy::too_many_arguments)]
    fn create_annotation(
        &self,
        situation_id: &str,
        kind: &str,
        body: &str,
        span_start: usize,
        span_end: usize,
        source_id: Option<&str>,
        chunk_id: Option<&str>,
        author: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn list_annotations(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn update_annotation(
        &self,
        annotation_id: &str,
        patch: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn delete_annotation(
        &self,
        annotation_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Collections
    fn create_collection(
        &self,
        narrative_id: &str,
        name: &str,
        description: Option<&str>,
        query: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn list_collections(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn get_collection(
        &self,
        collection_id: &str,
        resolve: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn update_collection(
        &self,
        collection_id: &str,
        patch: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn delete_collection(
        &self,
        collection_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Research notes
    #[allow(clippy::too_many_arguments)]
    fn create_research_note(
        &self,
        narrative_id: &str,
        situation_id: &str,
        kind: &str,
        body: &str,
        source_id: Option<&str>,
        author: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn list_research_notes(
        &self,
        situation_id: Option<&str>,
        narrative_id: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn get_research_note(
        &self,
        note_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn update_research_note(
        &self,
        note_id: &str,
        patch: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn delete_research_note(
        &self,
        note_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    #[allow(clippy::too_many_arguments)]
    fn promote_chunk_to_note(
        &self,
        narrative_id: &str,
        situation_id: &str,
        chunk_id: &str,
        body: &str,
        source_id: Option<&str>,
        kind: Option<&str>,
        author: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Editing engine
    fn propose_edit(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn apply_edit(
        &self,
        proposal: Value,
        author: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn estimate_edit_tokens(
        &self,
        situation_id: &str,
        instruction: &str,
        style_preset: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Revision completion
    fn commit_narrative_revision(
        &self,
        narrative_id: &str,
        message: &str,
        author: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn diff_narrative_revisions(
        &self,
        narrative_id: &str,
        from_rev: &str,
        to_rev: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Workshop
    fn list_workshop_reports(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn get_workshop_report(
        &self,
        report_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Cost ledger
    fn list_cost_ledger_entries(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Compile
    fn list_compile_profiles(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn compile_narrative(
        &self,
        narrative_id: &str,
        format: &str,
        profile_id: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn upsert_compile_profile(
        &self,
        narrative_id: &str,
        profile_id: Option<&str>,
        patch: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Templates
    fn list_narrative_templates(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn instantiate_template(
        &self,
        template_id: &str,
        bindings: std::collections::HashMap<String, String>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // Secondary: skeleton, dedup, debug-fixes, reorder
    fn extract_narrative_skeleton(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn find_duplicate_candidates(
        &self,
        narrative_id: &str,
        threshold: Option<f64>,
        max_candidates: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn suggest_narrative_fixes(
        &self,
        narrative_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn apply_narrative_fix(
        &self,
        narrative_id: &str,
        fix: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    fn apply_reorder(
        &self,
        narrative_id: &str,
        entries: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── EATH Phase 10 — Synthetic Hypergraph MCP tools ──────────

    /// Submit a `SurrogateCalibration` job. Returns `{ job_id, status }`.
    fn calibrate_surrogate(
        &self,
        narrative_id: &str,
        model: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit a `SurrogateGeneration` job. Returns `{ job_id, status }`.
    #[allow(clippy::too_many_arguments)]
    fn generate_synthetic_narrative(
        &self,
        source_narrative_id: &str,
        output_narrative_id: &str,
        model: Option<&str>,
        params: Option<Value>,
        seed: Option<u64>,
        num_steps: Option<usize>,
        label_prefix: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit a `SurrogateHybridGeneration` job. Returns `{ job_id, status }`.
    fn generate_hybrid_narrative(
        &self,
        components: Value,
        output_narrative_id: &str,
        seed: Option<u64>,
        num_steps: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List synthetic runs for a narrative, newest first.
    fn list_synthetic_runs(
        &self,
        narrative_id: &str,
        limit: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get a fidelity report for a (narrative, run_id) pair, or 404.
    fn get_fidelity_report(
        &self,
        narrative_id: &str,
        run_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit a `SurrogateSignificance` job. Returns `{ job_id, status }`.
    fn compute_pattern_significance(
        &self,
        narrative_id: &str,
        metric: &str,
        k: Option<u16>,
        model: Option<&str>,
        params_override: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Submit a `SurrogateContagionSignificance` job. Returns `{ job_id, status }`.
    fn simulate_higher_order_contagion(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        model: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// EATH Extension Phase 13c — submit a `SurrogateDualSignificance` job.
    /// Defaults `models` to `["eath", "nudhy"]` when omitted. Returns
    /// `{ job_id, status }`.
    fn compute_dual_significance(
        &self,
        narrative_id: &str,
        metric: &str,
        k_per_model: Option<u16>,
        models: Option<Vec<String>>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// EATH Extension Phase 14 — submit a `SurrogateBistabilitySignificance`
    /// job. Defaults `models` to `["eath", "nudhy"]` when omitted. Returns
    /// `{ job_id, status }`.
    fn compute_bistability_significance(
        &self,
        narrative_id: &str,
        params: Value,
        k: Option<u16>,
        models: Option<Vec<String>>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// EATH Extension Phase 15c — submit a `HypergraphReconstruction` job.
    /// `params` is an optional partial `ReconstructionParams` blob; the
    /// engine applies serde defaults for any omitted field. Returns
    /// `{ job_id, status }`.
    fn reconstruct_hypergraph(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// EATH Extension Phase 16c — synchronously run one opinion-dynamics
    /// simulation. `params` is an optional `OpinionDynamicsParams` blob;
    /// engine defaults applied when omitted. Returns an inline
    /// `{ run_id, report }` envelope (the report is the full
    /// `OpinionDynamicsReport`).
    fn simulate_opinion_dynamics(
        &self,
        narrative_id: &str,
        params: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// EATH Extension Phase 16c — sweep the confidence bound `c` and
    /// return the per-c convergence times plus the inferred critical-c
    /// spike. `c_range = [start, end, num_points]`. Returns the inline
    /// `PhaseTransitionReport`.
    fn simulate_opinion_phase_transition(
        &self,
        narrative_id: &str,
        c_range: [Value; 3],
        base_params: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Fuzzy Sprint Phase 10 — synchronous hybrid fuzzy-probability
    /// query. Evaluates `P_fuzzy(E) = Σ μ_E(e) · P(e)` on the
    /// Cao–Holčapek base case and persists at
    /// `fz/hybrid/{nid}/{query_id_BE_16}`. Returns
    /// `{ value, event_kind, distribution_summary, query_id, tnorm }`.
    /// Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
    fn fuzzy_probability(
        &self,
        narrative_id: &str,
        event: Value,
        distribution: Value,
        tnorm: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Fuzzy Sprint Phase 11 — 13 new fuzzy MCP tools ──────────
    //
    // Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
    //        [duboisprade1989fuzzyallen] [novak2008quantifiers]
    //        [murinovanovak2014peterson] [belohlavek2004fuzzyfca]
    //        [mamdani1975mamdani].

    /// List registered t-norm families (`GET /fuzzy/tnorms`).
    fn fuzzy_list_tnorms(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List registered aggregators (`GET /fuzzy/aggregators`).
    fn fuzzy_list_aggregators(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Get the per-workspace default fuzzy config (`GET /fuzzy/config`).
    fn fuzzy_get_config(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Update the per-workspace default fuzzy config (`PUT /fuzzy/config`).
    /// `reset = true` short-circuits to Gödel / Mean. The `measure`
    /// parameter is `None` → leave unchanged, `Some(None)` → clear,
    /// `Some(Some(name))` → set.
    fn fuzzy_set_config(
        &self,
        tnorm: Option<&str>,
        aggregator: Option<&str>,
        measure: Option<Option<&str>>,
        reset: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Persist a named monotone fuzzy measure (`POST /fuzzy/measures`).
    fn fuzzy_create_measure(
        &self,
        name: &str,
        n: u8,
        values: Vec<f64>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// List persisted fuzzy measures (`GET /fuzzy/measures`).
    fn fuzzy_list_measures(&self) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// One-shot aggregation over `xs` (`POST /fuzzy/aggregate`).
    fn fuzzy_aggregate(
        &self,
        xs: Vec<f64>,
        aggregator: &str,
        tnorm: Option<&str>,
        measure: Option<&str>,
        owa_weights: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Compute + cache the 13-dim graded Allen relation vector between
    /// two situations (`POST /analysis/fuzzy-allen`).
    fn fuzzy_allen_gradation(
        &self,
        narrative_id: &str,
        a_id: &str,
        b_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Evaluate an intermediate quantifier over a narrative's entities
    /// (`POST /fuzzy/quantify`).
    fn fuzzy_quantify(
        &self,
        narrative_id: &str,
        quantifier: &str,
        entity_type: Option<&str>,
        where_spec: Option<&str>,
        label: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Verify a graded Peterson syllogism (`POST /fuzzy/syllogism/verify`).
    #[allow(clippy::too_many_arguments)]
    fn fuzzy_verify_syllogism(
        &self,
        narrative_id: &str,
        major: &str,
        minor: &str,
        conclusion: &str,
        threshold: Option<f64>,
        tnorm: Option<&str>,
        figure_hint: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Build a fuzzy concept lattice for a narrative (`POST /fuzzy/fca/lattice`).
    fn fuzzy_build_lattice(
        &self,
        narrative_id: &str,
        entity_type: Option<&str>,
        attribute_allowlist: Option<Vec<String>>,
        threshold: Option<usize>,
        tnorm: Option<&str>,
        large_context: bool,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Create + persist a Mamdani fuzzy rule (`POST /fuzzy/rules`).
    fn fuzzy_create_rule(
        &self,
        name: &str,
        narrative_id: &str,
        antecedent: Value,
        consequent: Value,
        tnorm: Option<&str>,
        enabled: Option<bool>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Evaluate a rule set against one entity
    /// (`POST /fuzzy/rules/{nid}/evaluate`).
    fn fuzzy_evaluate_rules(
        &self,
        narrative_id: &str,
        entity_id: &str,
        rule_ids: Option<Vec<String>>,
        firing_aggregator: Option<crate::fuzzy::aggregation::AggregatorKind>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    // ─── Graded Acceptability Sprint Phase 5 — 5 new MCP tools ───
    //
    // Cites: [amgoud2013ranking] [grabisch1996choquet] [nebel1995ordhorn].

    /// Run gradual / ranking-based argumentation synchronously
    /// (`POST /analysis/argumentation/gradual`). `gradual_semantics` and
    /// `tnorm` are passed as opaque JSON because their tagged-union
    /// enums do not derive `JsonSchema`.
    fn argumentation_gradual(
        &self,
        narrative_id: &str,
        gradual_semantics: Value,
        tnorm: Option<Value>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Fit + persist a Choquet measure from a `(input_vec, rank)`
    /// dataset (`POST /fuzzy/measures/learn`). Auto-increments the
    /// version on retrain under an existing name.
    fn fuzzy_learn_measure(
        &self,
        name: &str,
        n: u8,
        dataset: Vec<(Vec<f64>, u32)>,
        dataset_id: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Version-aware fetch for a named measure
    /// (`GET /fuzzy/measures/{name}?version=N`). Absent `version`
    /// returns the latest pointer (legacy behaviour).
    fn fuzzy_get_measure_version(
        &self,
        name: &str,
        version: Option<u32>,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Enumerate every persisted version of a named measure
    /// (`GET /fuzzy/measures/{name}/versions`). Returns
    /// `{ name, versions: [u32] }` sorted ascending.
    fn fuzzy_list_measure_versions(
        &self,
        name: &str,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;

    /// Run path-consistency closure on an Allen interval-algebra
    /// network (`POST /temporal/ordhorn/closure`). `network` is passed
    /// as opaque JSON because [`crate::temporal::ordhorn::OrdHornNetwork`]
    /// does not derive `JsonSchema`.
    fn temporal_ordhorn_closure(
        &self,
        network: Value,
    ) -> impl std::future::Future<Output = Result<Value>> + Send;
}
