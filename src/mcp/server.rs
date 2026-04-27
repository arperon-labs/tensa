//! MCP server implementation with tool handlers.
//!
//! Uses rmcp macros (`#[tool_router]`, `#[tool]`, `#[tool_handler]`)
//! to expose 30 tools that delegate to the McpBackend trait.

use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::{tool, tool_handler, tool_router, ServerHandler};

use super::backend::McpBackend;
use super::error::error_result;
use super::types::*;

/// Wrap a backend Result into a CallToolResult with pretty-printed JSON.
fn wrap(result: crate::error::Result<serde_json::Value>) -> Result<CallToolResult, ErrorData> {
    match result {
        Ok(value) => Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&value).unwrap_or_else(|_| value.to_string()),
        )])),
        Err(e) => Ok(error_result(e)),
    }
}

/// TENSA MCP server — exposes hypergraph operations as MCP tools.
#[derive(Clone)]
pub struct TensaMcp<B: McpBackend + Clone + 'static> {
    tool_router: ToolRouter<Self>,
    backend: Arc<B>,
}

#[tool_router]
impl<B: McpBackend + Clone + 'static> TensaMcp<B> {
    /// Create a new TensaMcp server with the given backend.
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            backend,
        }
    }

    #[tool(
        name = "query",
        description = "Execute a TensaQL query or DML mutation. Queries: MATCH (graph patterns), INFER (async jobs), DISCOVER (cross-narrative). Mutations: CREATE (entity/narrative/situation), DELETE, UPDATE SET, ADD/REMOVE PARTICIPANT, ADD/REMOVE CAUSE. Example query: MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e. Example mutation: CREATE (e:Actor {name: \"Alice\"}) IN NARRATIVE \"hamlet\" CONFIDENCE 0.9"
    )]
    async fn query(
        &self,
        Parameters(req): Parameters<QueryRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.execute_query(&req.tensaql).await)
    }

    #[tool(
        name = "infer",
        description = "Submit a TensaQL INFER or DISCOVER query as an async inference job. INFER types: CAUSES, MOTIVATION, GAME, COUNTERFACTUAL, MISSING, ANOMALIES, CENTRALITY, ENTROPY, BELIEFS, EVIDENCE, ARGUMENTS, CONTAGION, TEMPORAL_RULES, MEAN_FIELD, PSL, TRAJECTORY, SIMULATE. DISCOVER: PATTERNS, ARCS, MISSING across narratives. Returns a job ID to poll with job_status/job_result."
    )]
    async fn infer(
        &self,
        Parameters(req): Parameters<InferRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.submit_inference_query(&req.tensaql).await)
    }

    #[tool(
        name = "ingest_text",
        description = "Ingest raw narrative text through the LLM extraction pipeline. Extracts entities, situations, and relationships using Claude. High-confidence extractions are auto-committed; lower-confidence ones go to the validation queue for human review."
    )]
    async fn ingest_text(
        &self,
        Parameters(req): Parameters<IngestTextRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_text(
                    &req.text,
                    &req.narrative_id,
                    &req.source_name,
                    req.auto_commit_threshold,
                    req.review_threshold,
                )
                .await,
        )
    }

    #[tool(
        name = "create_entity",
        description = "Create an entity in the hypergraph. Required: `entity_type` (Actor|Location|Artifact|Concept|Organization), `properties` (JSON object — e.g. {name, age}). Optional: `narrative_id` (string), `confidence` (0.0..1.0, default 0.5), `beliefs` (JSON object — entity's epistemic state; shape depends on downstream inference engines). The server sets `id` (UUID v7), `maturity` (starts Candidate), and timestamps; do not send them. Maturity is promoted later via validation — the real enum is Candidate/Reviewed/Validated/GroundTruth."
    )]
    async fn create_entity(
        &self,
        Parameters(req): Parameters<CreateEntityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = chrono::Utc::now();
        let data = serde_json::json!({
            "id": uuid::Uuid::now_v7(),
            "entity_type": req.entity_type,
            "properties": req.properties,
            "beliefs": req.beliefs,
            "embedding": null,
            "narrative_id": req.narrative_id,
            "maturity": "Candidate",
            "confidence": req.confidence.unwrap_or(0.5),
            "provenance": [],
            "created_at": now,
            "updated_at": now,
        });
        wrap(self.backend.create_entity(data).await)
    }

    #[tool(
        name = "get_entity",
        description = "Get an entity by its UUID. Returns the full entity record including type, properties, beliefs, maturity level, confidence score, and provenance."
    )]
    async fn get_entity(
        &self,
        Parameters(req): Parameters<GetByIdRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_entity(&req.id).await)
    }

    #[tool(
        name = "create_situation",
        description = "Create a situation (event/scene) in the hypergraph. Required: `raw_content` (string). Strongly recommended: `name` (string) — the chapter/scene title; without it the Studio outline shows \"(untitled)\". Optional: `description` (string long-form synopsis), `start`/`end` (ISO 8601 RFC 3339), `narrative_level` (Story|Arc|Sequence|Scene|Beat|Event, default Scene), `narrative_id`, `confidence` (0..1, default 0.5), `manuscript_order` (u32 — writer-curated binder position; set on Scene-level situations for deterministic ordering), `parent_situation_id` (UUID — binder hierarchy, e.g. Chapter Arc → Scene). Optional structure passthroughs: \
            \n  `discourse` — narratology: {order? (\"analepsis\"|\"prolepsis\"|\"simultaneous\"), duration? (\"scene\"|\"summary\"|\"ellipsis\"|\"pause\"|\"stretch\"), focalization? (entity UUID), voice? (\"homodiegetic\"|\"heterodiegetic\")}. Set at creation — back-filling POV per scene later is manual. \
            \n  `spatial` — SpatialData JSON (place name, lat/lng, polygon, etc.). \
            \n  `game_structure` — GameStructure JSON for scenes with strategic interaction. \
            \n  Server sets `id`, `maturity` (Candidate), `extraction_method` (HumanEntered), and timestamps."
    )]
    async fn create_situation(
        &self,
        Parameters(req): Parameters<CreateSituationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = chrono::Utc::now();
        let start = req.start.as_ref().and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Utc))
        });
        let end = req.end.as_ref().and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Utc))
        });
        let level = req.narrative_level.as_deref().unwrap_or("Scene");

        let discourse_json = req.discourse.as_ref().map(|d| {
            serde_json::json!({
                "order": d.order,
                "duration": d.duration,
                "focalization": d.focalization,
                "voice": d.voice,
            })
        });

        let data = serde_json::json!({
            "id": uuid::Uuid::now_v7(),
            "name": req.name,
            "description": req.description,
            "temporal": {
                "start": start,
                "end": end,
                "granularity": "Approximate",
                "relations": [],
            },
            "spatial": req.spatial,
            "game_structure": req.game_structure,
            "causes": [],
            "deterministic": null,
            "probabilistic": null,
            "embedding": null,
            "raw_content": [{
                "content_type": "Text",
                "content": req.raw_content,
                "source": null,
            }],
            "narrative_level": level,
            "narrative_id": req.narrative_id,
            "discourse": discourse_json,
            "manuscript_order": req.manuscript_order,
            "parent_situation_id": req.parent_situation_id,
            "maturity": "Candidate",
            "confidence": req.confidence.unwrap_or(0.5),
            "extraction_method": "HumanEntered",
            "created_at": now,
            "updated_at": now,
        });
        wrap(self.backend.create_situation(data).await)
    }

    #[tool(
        name = "get_situation",
        description = "Get a situation (event/scene) by its UUID. Returns the full record including temporal interval, spatial data, game structure, raw content, narrative level, and confidence."
    )]
    async fn get_situation(
        &self,
        Parameters(req): Parameters<GetByIdRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_situation(&req.id).await)
    }

    #[tool(
        name = "add_participant",
        description = "Link an entity to a situation as a participant. Required: `entity_id` (UUID), `situation_id` (UUID), `role`. `role` may be one of the built-in strings (Protagonist|Antagonist|Witness|Target|Instrument|Confidant|Informant|Recipient|Bystander|SubjectOfDiscussion|Facilitator) OR a `{\"Custom\": \"role-name\"}` JSON object for arbitrary roles. Optional: `action` (string, e.g. \"confesses\"), `info_set` (JSON: {knows_before, learns, reveals} — drives belief and dramatic-irony reasoning), `payoff` (number, game-theoretic scalar)."
    )]
    async fn add_participant(
        &self,
        Parameters(req): Parameters<AddParticipantRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let data = serde_json::json!({
            "entity_id": req.entity_id,
            "situation_id": req.situation_id,
            "role": req.role,
            "info_set": req.info_set,
            "action": req.action,
            "payoff": req.payoff,
        });
        wrap(self.backend.add_participant(data).await)
    }

    #[tool(
        name = "list_narratives",
        description = "List all narrative metadata. Each narrative is a named collection of entities and situations (e.g. 'crime-and-punishment'). Returns IDs, titles, genres, tags, and entity/situation counts."
    )]
    async fn list_narratives(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_narratives().await)
    }

    #[tool(
        name = "create_narrative",
        description = "Create narrative metadata — a named collection for grouping entities and situations. Required: `narrative_id` (slug, e.g. 'hamlet'), `title`. Optional: `genre`, `description`, `tags` (array<string>), `authors` (array<string>), `language` (ISO 639-1, e.g. 'en'), `publication_date` (ISO 8601 RFC 3339), `cover_url`, `project_id` (parent project slug), `custom_properties` (arbitrary JSON object — values may be strings, numbers, booleans, arrays, or nested objects). Server sets counts and timestamps."
    )]
    async fn create_narrative(
        &self,
        Parameters(req): Parameters<CreateNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = chrono::Utc::now();
        let pub_date = req.publication_date.as_ref().and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Utc))
        });
        let data = serde_json::json!({
            "id": req.narrative_id,
            "title": req.title,
            "genre": req.genre,
            "tags": req.tags.unwrap_or_default(),
            "source": null,
            "description": req.description,
            "authors": req.authors.unwrap_or_default(),
            "language": req.language,
            "publication_date": pub_date,
            "cover_url": req.cover_url,
            "project_id": req.project_id,
            "custom_properties": req.custom_properties.unwrap_or_default(),
            "entity_count": 0,
            "situation_count": 0,
            "created_at": now,
            "updated_at": now,
        });
        wrap(self.backend.create_narrative(data).await)
    }

    #[tool(
        name = "job_status",
        description = "Check the status of an async inference job by its ID. Returns job type, status (Pending/Running/Completed/Failed), timestamps, and error details if failed."
    )]
    async fn job_status(
        &self,
        Parameters(req): Parameters<JobIdRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_job_status(&req.job_id).await)
    }

    #[tool(
        name = "job_result",
        description = "Get the result of a completed inference job. Returns the inference output (causal graph, game structure, motivation profile, etc.) with confidence scores."
    )]
    async fn job_result(
        &self,
        Parameters(req): Parameters<JobIdRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_job_result(&req.job_id).await)
    }

    #[tool(
        name = "create_source",
        description = "Register an intelligence source with trust score, bias profile, and metadata. Source types: NewsOutlet, GovernmentAgency, AcademicInstitution, SocialMedia, Sensor, StructuredApi, HumanAnalyst, OsintTool. Returns the source UUID."
    )]
    async fn create_source(
        &self,
        Parameters(req): Parameters<CreateSourceRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = chrono::Utc::now();
        let data = serde_json::json!({
            "id": uuid::Uuid::now_v7(),
            "name": req.name,
            "source_type": req.source_type,
            "url": req.url,
            "description": req.description,
            "trust_score": req.trust_score.unwrap_or(0.5),
            "bias_profile": {
                "known_biases": req.known_biases.unwrap_or_default(),
                "political_lean": null,
                "sensationalism": null,
                "notes": null,
            },
            "track_record": {
                "claims_made": 0,
                "claims_corroborated": 0,
                "claims_contradicted": 0,
                "last_evaluated": null,
            },
            "tags": req.tags.unwrap_or_default(),
            "created_at": now,
            "updated_at": now,
        });
        wrap(self.backend.create_source(data).await)
    }

    #[tool(
        name = "get_source",
        description = "Get a registered intelligence source by UUID. Returns trust score, bias profile, track record, and metadata."
    )]
    async fn get_source(
        &self,
        Parameters(req): Parameters<GetSourceRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_source(&req.id).await)
    }

    #[tool(
        name = "add_attribution",
        description = "Link a registered source to an entity or situation claim. Tracks which source reported what, with extraction confidence and optional excerpt. Enables multi-source corroboration scoring."
    )]
    async fn add_attribution(
        &self,
        Parameters(req): Parameters<AddAttributionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let data = serde_json::json!({
            "source_id": req.source_id,
            "target_id": req.target_id,
            "target_kind": req.target_kind,
            "retrieved_at": chrono::Utc::now(),
            "original_url": req.original_url,
            "excerpt": req.excerpt,
            "extraction_confidence": req.extraction_confidence.unwrap_or(0.8),
        });
        wrap(self.backend.add_attribution(data).await)
    }

    #[tool(
        name = "list_contentions",
        description = "List all contention links for a situation — contradictions, numerical disagreements, temporal disagreements, or omission biases between competing source claims."
    )]
    async fn list_contentions(
        &self,
        Parameters(req): Parameters<ListContentionsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_contentions(&req.situation_id).await)
    }

    #[tool(
        name = "recompute_confidence",
        description = "Recompute the confidence breakdown for an entity or situation based on current source attributions. Returns extraction, source credibility, corroboration, and recency scores plus a weighted composite."
    )]
    async fn recompute_confidence(
        &self,
        Parameters(req): Parameters<RecomputeConfidenceRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.recompute_confidence(&req.id).await)
    }

    #[tool(
        name = "review_queue",
        description = "Manage the validation queue for LLM extraction review. Actions: 'list' (pending items), 'get' (by item_id), 'approve' (requires reviewer), 'reject' (requires reviewer, optional notes), 'edit' (requires reviewer + edited_data JSON)."
    )]
    async fn review_queue(
        &self,
        Parameters(req): Parameters<ReviewQueueRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .review_queue(
                    &req.action,
                    req.item_id.as_deref(),
                    req.reviewer.as_deref(),
                    req.notes.as_deref(),
                    req.edited_data,
                    req.limit,
                )
                .await,
        )
    }

    #[tool(
        name = "simulate_counterfactual",
        description = "Simulate a counterfactual: 'What if this had been different?' Provide a situation UUID, the field to change (e.g. 'action', 'payoff'), and the new value. Returns a job ID — poll with job_status/job_result for cascading narrative probabilities."
    )]
    async fn simulate_counterfactual(
        &self,
        Parameters(req): Parameters<SimulateCounterfactualRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Validate intervention_target is a safe identifier (alphanumeric + underscore)
        if !req
            .intervention_target
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            return Ok(CallToolResult::error(vec![Content::text(
                "Error: intervention_target must be alphanumeric (e.g. 'action', 'payoff')"
                    .to_string(),
            )]));
        }
        let escaped_value = req.new_value.replace('"', r#"\""#);
        let tensaql = format!(
            r#"INFER COUNTERFACTUAL FOR s:Situation ASSUMING s.{} = "{}" RETURN s"#,
            req.intervention_target, escaped_value
        );
        wrap(
            self.backend
                .submit_inference_query(&tensaql)
                .await
                .map(|mut value| {
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("situation_id".into(), serde_json::json!(req.situation_id));
                    }
                    value
                }),
        )
    }

    #[tool(
        name = "get_actor_profile",
        description = "Get a comprehensive actor dossier: entity properties, all situation participations with roles and actions, state version history showing how the actor changed over time, and participation count. Use to understand character/suspect behavior and trajectory."
    )]
    async fn get_actor_profile(
        &self,
        Parameters(req): Parameters<GetActorProfileRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_actor_profile(&req.actor_id).await)
    }

    #[tool(
        name = "find_cross_narrative_patterns",
        description = "Discover structural patterns shared across narratives using graph kernel analysis. Provide narrative IDs to compare. Returns a job ID — poll with job_status/job_result for recurring motifs (e.g. 'betrayal dynamic', 'escalation pattern') and similarity scores."
    )]
    async fn find_cross_narrative_patterns(
        &self,
        Parameters(req): Parameters<FindCrossNarrativePatternsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let ids_str = req
            .narrative_ids
            .iter()
            .map(|id| format!(r#""{}""#, id.replace('"', "")))
            .collect::<Vec<_>>()
            .join(", ");
        let tensaql = format!(
            r#"DISCOVER PATTERNS ACROSS NARRATIVES ({}) RETURN *"#,
            ids_str
        );
        wrap(self.backend.submit_inference_query(&tensaql).await)
    }

    #[tool(
        name = "delete_entity",
        description = "Soft-delete an entity by UUID. The entity is marked as deleted but can be restored. All participations and state versions are preserved."
    )]
    async fn delete_entity(
        &self,
        Parameters(req): Parameters<DeleteByIdRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.delete_entity(&req.id).await)
    }

    #[tool(
        name = "delete_situation",
        description = "Soft-delete a situation by UUID. The situation is marked as deleted but can be restored. Participations and causal links are preserved."
    )]
    async fn delete_situation(
        &self,
        Parameters(req): Parameters<DeleteByIdRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.delete_situation(&req.id).await)
    }

    #[tool(
        name = "update_entity",
        description = "Update an entity by UUID. Provide a JSON object with fields to update: 'properties' (merged into existing), 'confidence' (0-1), 'narrative_id'. Only specified fields are changed."
    )]
    async fn update_entity(
        &self,
        Parameters(req): Parameters<UpdateEntityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.update_entity(&req.id, req.updates).await)
    }

    #[tool(
        name = "list_entities",
        description = "List entities in the hypergraph. Filter by entity_type (Actor/Location/Artifact/Concept/Organization), narrative_id, and limit. Returns entity records with properties, confidence, and maturity."
    )]
    async fn list_entities(
        &self,
        Parameters(req): Parameters<ListEntitiesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .list_entities(
                    req.entity_type.as_deref(),
                    req.narrative_id.as_deref(),
                    req.limit,
                )
                .await,
        )
    }

    #[tool(
        name = "list_sources",
        description = "List all registered intelligence sources with trust scores, bias profiles, and track records. Use to review source credibility before analysis."
    )]
    async fn list_sources(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_sources().await)
    }

    #[tool(
        name = "merge_entities",
        description = "Merge two entities discovered to be the same. The 'keep' entity survives; the 'absorb' entity's participations, state versions, and causal links are transferred to keep, then absorb is deleted."
    )]
    async fn merge_entities(
        &self,
        Parameters(req): Parameters<MergeEntitiesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .merge_entities(&req.keep_id, &req.absorb_id)
                .await,
        )
    }

    #[tool(
        name = "export_narrative",
        description = "Export a narrative in a standard format. Formats: 'csv' (entities + situations as rows), 'graphml' (nodes + edges for Gephi), 'json' (full dump), 'manuscript' (temporal prose as Markdown), 'report' (analytical Markdown with timeline, profiles, relationships)."
    )]
    async fn export_narrative(
        &self,
        Parameters(req): Parameters<ExportNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .export_narrative(&req.narrative_id, &req.format)
                .await,
        )
    }

    #[tool(
        name = "get_narrative_stats",
        description = "Get statistics for a narrative: entity count, situation count, participation count, causal link count, temporal span, and narrative level distribution."
    )]
    async fn get_narrative_stats(
        &self,
        Parameters(req): Parameters<GetNarrativeStatsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_narrative_stats(
                    &req.narrative_id,
                    req.tnorm.as_deref(),
                    req.aggregator.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "search_entities",
        description = "Search for entities by text query. Searches across entity property values (names, descriptions, etc.). Returns matching entities ranked by relevance, limited to 20 results by default."
    )]
    async fn search_entities(
        &self,
        Parameters(req): Parameters<SearchEntitiesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .search_entities(
                    &req.query,
                    req.limit,
                    req.tnorm.as_deref(),
                    req.aggregator.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "ingest_url",
        description = "Fetch a URL, strip HTML, and ingest the text through the LLM extraction pipeline. Extracts entities, situations, and relationships from web content."
    )]
    async fn ingest_url(
        &self,
        Parameters(req): Parameters<IngestUrlRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_url(&req.url, &req.narrative_id, req.source_name.as_deref())
                .await,
        )
    }

    #[tool(
        name = "ingest_rss",
        description = "Fetch an RSS/Atom feed and ingest each item through the LLM extraction pipeline. Requires the web-ingest feature to be enabled."
    )]
    async fn ingest_rss(
        &self,
        Parameters(req): Parameters<IngestRssRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_rss(&req.feed_url, &req.narrative_id, req.max_items)
                .await,
        )
    }

    #[tool(
        name = "split_entity",
        description = "Split an entity into two. Creates a clone and moves the specified situation participations to the new entity. Returns the new entity's UUID."
    )]
    async fn split_entity(
        &self,
        Parameters(req): Parameters<SplitEntityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .split_entity(&req.entity_id, &req.situation_ids)
                .await,
        )
    }

    #[tool(
        name = "restore_entity",
        description = "Restore a soft-deleted entity by clearing its deleted_at timestamp."
    )]
    async fn restore_entity(
        &self,
        Parameters(req): Parameters<RestoreEntityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.restore_entity(&req.id).await)
    }

    #[tool(
        name = "restore_situation",
        description = "Restore a soft-deleted situation by clearing its deleted_at timestamp."
    )]
    async fn restore_situation(
        &self,
        Parameters(req): Parameters<RestoreSituationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.restore_situation(&req.id).await)
    }

    #[tool(
        name = "create_project",
        description = "Create a project container for grouping narratives. Provide a slug ID (name), optional title, and description. Returns the project ID."
    )]
    async fn create_project(
        &self,
        Parameters(req): Parameters<CreateProjectRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let now = chrono::Utc::now();
        let data = serde_json::json!({
            "id": req.name,
            "title": req.title.unwrap_or_else(|| req.name.clone()),
            "description": req.description,
            "tags": [],
            "narrative_count": 0,
            "created_at": now,
            "updated_at": now,
        });
        wrap(self.backend.create_project(data).await)
    }

    #[tool(
        name = "get_project",
        description = "Get a project by its slug ID. Returns the project metadata including title, description, tags, and narrative count."
    )]
    async fn get_project(
        &self,
        Parameters(req): Parameters<GetProjectRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_project(&req.id).await)
    }

    #[tool(
        name = "list_projects",
        description = "List all project containers. Optionally limit the number of results."
    )]
    async fn list_projects(
        &self,
        Parameters(req): Parameters<ListProjectsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_projects(req.limit).await)
    }

    #[tool(
        name = "update_project",
        description = "Update a project's title, description, or tags. Only specified fields are changed."
    )]
    async fn update_project(
        &self,
        Parameters(req): Parameters<UpdateProjectRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut updates = serde_json::Map::new();
        if let Some(title) = req.title {
            updates.insert("title".into(), serde_json::Value::String(title));
        }
        if let Some(desc) = req.description {
            updates.insert("description".into(), serde_json::Value::String(desc));
        }
        wrap(
            self.backend
                .update_project(&req.id, serde_json::Value::Object(updates))
                .await,
        )
    }

    #[tool(
        name = "delete_project",
        description = "Delete a project by its slug ID. With cascade=true, also deletes all narratives in the project."
    )]
    async fn delete_project(
        &self,
        Parameters(req): Parameters<DeleteProjectRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .delete_project(&req.id, req.cascade.unwrap_or(false))
                .await,
        )
    }

    #[tool(
        name = "cache_stats",
        description = "Get LLM response cache statistics (entries, total bytes)."
    )]
    async fn cache_stats(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.cache_stats().await)
    }

    #[tool(
        name = "ask",
        description = "Ask a natural language question about narrative data. Uses RAG (Retrieval-Augmented Generation) to find relevant context and generate an answer with citations. Optionally scope to a narrative and choose retrieval mode (local/global/hybrid/mix). Set response_type for custom formatting and suggest=true for follow-up questions."
    )]
    async fn ask(
        &self,
        Parameters(req): Parameters<AskRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ask(
                    &req.question,
                    req.narrative_id.as_deref(),
                    req.mode.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "tune_prompts",
        description = "Generate domain-adapted extraction prompts for a narrative by sampling its ingested chunks and using the LLM to produce extraction guidelines. The tuned prompts are stored and automatically used during future ingestion for this narrative."
    )]
    async fn tune_prompts(
        &self,
        Parameters(req): Parameters<TunePromptsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.tune_prompts(&req.narrative_id).await)
    }

    #[tool(
        name = "community_hierarchy",
        description = "Get the hierarchical community structure for a narrative. Communities are detected using the Leiden algorithm (guarantees connected communities) and organized into levels: level 0 is the most granular, higher levels are coarser. Optionally filter by level."
    )]
    async fn community_hierarchy(
        &self,
        Parameters(req): Parameters<CommunityHierarchyRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .community_hierarchy(&req.narrative_id, req.level)
                .await,
        )
    }

    #[tool(
        name = "export_archive",
        description = "Export one or more narratives as a lossless .tensa archive (ZIP with JSON files). Returns base64-encoded archive data. Use for backup, transfer between Tensa instances, or offline analysis."
    )]
    async fn export_archive(
        &self,
        Parameters(req): Parameters<ExportArchiveRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.export_archive(req.narrative_ids).await)
    }

    #[tool(
        name = "import_archive",
        description = "Import a .tensa archive from base64-encoded data. Creates narratives, entities, situations, and all associated data. Returns an import report with counts and any warnings."
    )]
    async fn import_archive(
        &self,
        Parameters(req): Parameters<ImportArchiveRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.import_archive(&req.data).await)
    }

    #[tool(
        name = "verify_authorship",
        description = "PAN@CLEF-style authorship verification: score whether two texts were written by the same author. Returns a probability in [0, 1] plus a same/different/abstain decision. Uses Burrows-Cosine on z-scored function words plus scalar feature differences."
    )]
    async fn verify_authorship(
        &self,
        Parameters(req): Parameters<VerifyAuthorshipRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .verify_authorship(&req.text_a, &req.text_b)
                .await,
        )
    }

    // ─── Narrative Debugger (Sprint D10) ─────────────────────

    #[tool(
        name = "diagnose_narrative",
        description = "Run structural 'linter' on a narrative — detects orphaned setups, impossible knowledge, causal orphans/contradictions, motivation discontinuities, irony collapses, arc abandonment, pacing arrhythmia, and 10+ more pathology types. Returns a DiagnosisResult with pathologies, severities, and a health score (0–1). Optional `genre`: thriller | literary_fiction | epic_fantasy | mystery | generic."
    )]
    async fn diagnose_narrative(
        &self,
        Parameters(req): Parameters<DiagnoseNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .diagnose_narrative(&req.narrative_id, req.genre.as_deref())
                .await,
        )
    }

    #[tool(
        name = "get_health_score",
        description = "Fast narrative health check: returns health_score (0–1), pathology counts by severity, and the worst chapter. A score ≥ 0.9 is clean, 0.7–0.9 has warnings, < 0.5 has structural errors."
    )]
    async fn get_health_score(
        &self,
        Parameters(req): Parameters<GetHealthScoreRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_health_score(&req.narrative_id).await)
    }

    #[tool(
        name = "auto_repair_narrative",
        description = "Iteratively diagnose and apply safe structural fixes (e.g. mark orphaned commitments as RedHerring). Conservative: applies only confidence ≥ 0.5 fixes with no side effects. `max_severity` caps which pathologies to address: error | warning | info. `max_iterations` (default 5) bounds the loop."
    )]
    async fn auto_repair_narrative(
        &self,
        Parameters(req): Parameters<AutoRepairRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .auto_repair(
                    &req.narrative_id,
                    req.max_severity.as_deref(),
                    req.max_iterations,
                )
                .await,
        )
    }

    // ─── Narrative Adaptation (Sprint D11) ─────────────────

    #[tool(
        name = "compute_essentiality",
        description = "Score narrative elements (situations, entities, subplots) 0.0–1.0 by structural essentiality. Weighted combination of causal criticality, commitment load, knowledge-gate role, arc anchoring, and dramatic irony contribution. Low scores are candidates for compression; high scores are anchors that must be preserved."
    )]
    async fn compute_essentiality(
        &self,
        Parameters(req): Parameters<ComputeEssentialityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.compute_essentiality(&req.narrative_id).await)
    }

    #[tool(
        name = "compress_narrative",
        description = "Produce a compression plan — a list of subplots/situations to remove and characters to merge — that shrinks the narrative to a target length while preserving the causal critical path. Presets: novella (~40%), short_story (~15%), screenplay_outline (~50 scenes). Plan-only: does not mutate the hypergraph."
    )]
    async fn compress_narrative(
        &self,
        Parameters(req): Parameters<CompressNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .compress_narrative(
                    &req.narrative_id,
                    req.preset.as_deref(),
                    req.target_chapters,
                    req.target_ratio,
                )
                .await,
        )
    }

    #[tool(
        name = "expand_narrative",
        description = "Produce an expansion plan — list of arcs to deepen, commitments to extend with progress events, scenes to expand, subplots to add — bringing the narrative to target_chapters. Plan-only: LLM-assisted content generation happens downstream."
    )]
    async fn expand_narrative_tool(
        &self,
        Parameters(req): Parameters<ExpandNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .expand_narrative(&req.narrative_id, req.target_chapters)
                .await,
        )
    }

    #[tool(
        name = "diff_narratives",
        description = "Structural diff between two narratives: entity adds/removes/modifications, situation changes, broken/new causal links, commitment status changes, arc type changes, pacing delta, composite structural distance."
    )]
    async fn diff_narratives(
        &self,
        Parameters(req): Parameters<DiffNarrativesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .diff_narratives(&req.narrative_a, &req.narrative_b)
                .await,
        )
    }

    // ─── Disinfo Sprint D1: dual fingerprint tools ─────────────

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "get_behavioral_fingerprint",
        description = "Disinfo extension: get the 10-axis BehavioralFingerprint for an actor (account/entity). Axes: posting_cadence_regularity, sleep_pattern_presence, engagement_ratio, account_maturity, platform_diversity, content_originality, response_latency, hashtag_concentration, network_insularity, temporal_coordination. NaN axes await later sprints. Set recompute=true to force a fresh computation."
    )]
    async fn get_behavioral_fingerprint(
        &self,
        Parameters(req): Parameters<GetBehavioralFingerprintRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_behavioral_fingerprint(&req.actor_id, req.recompute)
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "get_disinfo_fingerprint",
        description = "Disinfo extension: get the 12-axis DisinformationFingerprint for a narrative. Axes: virality_velocity, cross_platform_jump_rate, linguistic_variance, bot_amplification_ratio, emotional_loading, source_diversity, coordination_score, claim_mutation_rate, counter_narrative_resistance, evidential_uncertainty, temporal_anomaly, authority_exploitation. NaN axes await later sprints. Set recompute=true to force a fresh computation."
    )]
    async fn get_disinfo_fingerprint(
        &self,
        Parameters(req): Parameters<GetDisinfoFingerprintRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_disinfo_fingerprint(&req.narrative_id, req.recompute)
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "compare_fingerprints",
        description = "Disinfo extension: compare two behavioral or two disinfo fingerprints using per-layer distance kernels (Burrows-Cosine, Jensen-Shannon, Hamming) with task-specific weighting. kind: 'behavioral' or 'disinfo'. task: 'literary' (default) | 'cib' | 'factory'. Returns composite_distance, per-axis distances, p_value, 95% confidence_interval, same_source_verdict, and top anomaly axes."
    )]
    async fn compare_fingerprints(
        &self,
        Parameters(req): Parameters<CompareFingerprintsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .compare_fingerprints(&req.kind, req.task.as_deref(), &req.a_id, &req.b_id)
                .await,
        )
    }

    // ─── Sprint D2: Spread Dynamics ───────────────────────────

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "estimate_r0_by_platform",
        description = "Disinfo Sprint D2: run SMIR contagion + per-platform R₀ + cross-platform jump detection + velocity-monitor anomaly check for a fact within a narrative. β defaults are platform-tuned (Twitter ≈ 0.45, Telegram ≈ 0.30, Bluesky ≈ 0.20, ...) and overridable per call. Returns the SMIR result, list of cross-platform jumps, and any velocity alerts that fired."
    )]
    async fn estimate_r0_by_platform(
        &self,
        Parameters(req): Parameters<EstimateR0Request>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .estimate_r0_by_platform(
                    &req.narrative_id,
                    &req.fact,
                    &req.about_entity,
                    req.narrative_kind.as_deref(),
                    req.beta_overrides,
                )
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "simulate_intervention",
        description = "Disinfo Sprint D2: counterfactual spread projection. intervention: {type: 'RemoveTopAmplifiers', n: <usize>} or {type: 'DebunkAt', at: <RFC3339 timestamp>}. Returns baseline_r0, projected_r0, r0_delta, audience_saved, removed_entities."
    )]
    async fn simulate_intervention(
        &self,
        Parameters(req): Parameters<SimulateInterventionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .simulate_intervention(
                    &req.narrative_id,
                    &req.fact,
                    &req.about_entity,
                    req.intervention,
                    req.beta_overrides,
                )
                .await,
        )
    }

    // ─── Sprint D3: CIB Detection ────────────────────────────

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "detect_cib_cluster",
        description = "Disinfo Sprint D3: detect Coordinated Inauthentic Behavior clusters in a narrative. Builds a behavioral-similarity network over actors, runs label-propagation community detection, and flags communities whose induced-subgraph density falls in the right tail of a bootstrap null (calibrated p-value below alpha). Persists CibCluster + CibEvidence records. Set cross_platform=true to filter to clusters spanning ≥ 2 platforms (factory-task detection)."
    )]
    async fn detect_cib_cluster(
        &self,
        Parameters(req): Parameters<DetectCibClusterRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .detect_cib_cluster(
                    &req.narrative_id,
                    req.cross_platform,
                    req.similarity_threshold,
                    req.alpha,
                    req.bootstrap_iter,
                    req.min_cluster_size,
                    req.seed,
                )
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "rank_superspreaders",
        description = "Disinfo Sprint D3: rank the top-N actors in a narrative by graph centrality on the co-participation network. method: 'pagerank' (default) | 'eigenvector' | 'harmonic'. Persists the ranking at cib/s/{narrative_id}."
    )]
    async fn rank_superspreaders(
        &self,
        Parameters(req): Parameters<RankSuperspreadersRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .rank_superspreaders(&req.narrative_id, req.method.as_deref(), req.top_n)
                .await,
        )
    }

    // ─── Claims & Fact-Check (Sprint D4) ─────────────────────

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "ingest_claim",
        description = "Disinfo Sprint D4: detect verifiable claims in text using regex-based heuristics. Returns detected claims with categories (numerical, quote, causal, comparison, predictive, factual) and confidence scores. Claims are persisted at cl/ KV prefix."
    )]
    async fn ingest_claim(
        &self,
        Parameters(req): Parameters<IngestClaimRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_claim(&req.text, req.narrative_id.as_deref())
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "ingest_fact_check",
        description = "Disinfo Sprint D4: ingest a fact-check verdict for an existing claim. Persists at fc/ KV prefix. Verdicts: true, false, misleading, partially_true, unverifiable, satire, out_of_context."
    )]
    async fn ingest_fact_check(
        &self,
        Parameters(req): Parameters<IngestFactCheckRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_fact_check(
                    &req.claim_id,
                    &req.verdict,
                    &req.source,
                    req.url.as_deref(),
                    &req.language,
                )
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "fetch_fact_checks",
        description = "Disinfo Sprint D4: match a claim against known fact-checks using text similarity and embedding cosine similarity. Returns matches sorted by similarity score."
    )]
    async fn fetch_fact_checks(
        &self,
        Parameters(req): Parameters<FetchFactChecksRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fetch_fact_checks(&req.claim_id, req.min_similarity)
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "trace_claim_origin",
        description = "Disinfo Sprint D4: trace a claim back through the hypergraph to its earliest appearance. Returns a temporal chain of claim appearances with similarity scores."
    )]
    async fn trace_claim_origin(
        &self,
        Parameters(req): Parameters<TraceClaimOriginRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.trace_claim_origin(&req.claim_id).await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "classify_archetype",
        description = "Sprint D5: classify an actor into adversarial archetypes (StateActor, OrganicConspiracist, CommercialTrollFarm, Hacktivist, UsefulIdiot, HybridActor). Returns a probability distribution across all archetypes."
    )]
    async fn classify_archetype(
        &self,
        Parameters(req): Parameters<ClassifyArchetypeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .classify_archetype(&req.actor_id, req.force)
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "assess_disinfo",
        description = "Sprint D5: fuse multiple disinfo signals via Dempster-Shafer evidence combination. Returns belief/plausibility intervals for True/False/Misleading hypotheses, conflict measure, and overall verdict."
    )]
    async fn assess_disinfo(
        &self,
        Parameters(req): Parameters<AssessDisinfoRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .assess_disinfo(&req.target_id, req.signals)
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "ingest_post",
        description = "Sprint D5: ingest a social media post as a situation linked to an actor. Creates a situation with the post text and adds a participation linking the actor."
    )]
    async fn ingest_post(
        &self,
        Parameters(req): Parameters<IngestPostRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_post(
                    &req.text,
                    &req.actor_id,
                    &req.narrative_id,
                    req.platform.as_deref(),
                )
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "ingest_actor",
        description = "Sprint D5: create an Actor entity with disinfo-relevant properties (platform, name). Returns the new entity UUID."
    )]
    async fn ingest_actor(
        &self,
        Parameters(req): Parameters<IngestActorRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .ingest_actor(
                    &req.name,
                    &req.narrative_id,
                    req.platform.as_deref(),
                    req.properties,
                )
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "link_narrative",
        description = "Sprint D5: set the narrative_id on an existing entity, linking it to a narrative for cross-narrative analysis."
    )]
    async fn link_narrative(
        &self,
        Parameters(req): Parameters<LinkNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .link_narrative(&req.entity_id, &req.narrative_id)
                .await,
        )
    }

    // ─── Sprint D6: Multilingual & Export ────────────────────────

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "detect_language",
        description = "Sprint D6: detect the language of text using Unicode script heuristics. Returns ISO 639-1 code and confidence score."
    )]
    async fn detect_language(
        &self,
        Parameters(req): Parameters<DetectLanguageRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.detect_language(&req.text).await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "export_misp_event",
        description = "Sprint D6: export a narrative as a MISP event with attributes mapped from TENSA entities and situations."
    )]
    async fn export_misp_event(
        &self,
        Parameters(req): Parameters<ExportMispRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.export_misp_event(&req.narrative_id).await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "export_maltego",
        description = "Sprint D6: export narrative entities as Maltego-compatible transform results for graph visualization."
    )]
    async fn export_maltego(
        &self,
        Parameters(req): Parameters<ExportMaltegoRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.export_maltego(&req.narrative_id).await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "generate_report",
        description = "Sprint D6: generate a comprehensive Markdown disinfo analysis report for a narrative including actors, timeline, network, and causal chains."
    )]
    async fn generate_report(
        &self,
        Parameters(req): Parameters<GenerateReportRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_disinfo_report(&req.narrative_id)
                .await,
        )
    }

    // ─── Sprint D8: Scheduler + Reports + Health ────────────────

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "list_scheduled_tasks",
        description = "Sprint D8: list all scheduled analysis tasks (CIB scans, MCP polls, fact-check syncs, report generation, etc.)."
    )]
    async fn list_scheduled_tasks(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_scheduled_tasks().await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "create_scheduled_task",
        description = "Sprint D8: create a new scheduled task. Types: cib_scan, source_discovery, fact_check_sync, report_generation, mcp_poll, fingerprint_refresh, velocity_baseline_update. Schedule: e.g. '30m', '6h', '1d'."
    )]
    async fn create_scheduled_task(
        &self,
        Parameters(req): Parameters<CreateScheduledTaskRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .create_scheduled_task(&req.task_type, &req.schedule, req.config)
                .await,
        )
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "run_task_now",
        description = "Sprint D8: trigger immediate execution of a scheduled task by its UUID."
    )]
    async fn run_task_now(
        &self,
        Parameters(req): Parameters<RunTaskNowRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.run_task_now(&req.task_id).await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "list_discovery_candidates",
        description = "Sprint D8: list discovered source candidates (channels, accounts, URLs found through link shares, CIB clusters, and mentions)."
    )]
    async fn list_discovery_candidates(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_discovery_candidates().await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "sync_fact_checks",
        description = "Sprint D8: trigger fact-check database sync against configured sources (Google Fact Check Tools, ClaimsKG, RSS feeds)."
    )]
    async fn sync_fact_checks(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.sync_fact_checks().await)
    }

    #[cfg(feature = "disinfo")]
    #[tool(
        name = "generate_situation_report",
        description = "Sprint D8: generate a periodic situation report aggregating narratives tracked, CIB clusters, velocity alerts, claims, and new sources for the specified time window (default 24 hours)."
    )]
    async fn generate_situation_report(
        &self,
        Parameters(req): Parameters<GenerateSituationReportRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.generate_situation_report(req.hours).await)
    }

    // ─── Sprint D9: Narrative Architecture ──────────────────

    #[tool(
        name = "detect_commitments",
        description = "D9.1: Detect narrative commitments (Chekhov's guns, foreshadowing, dramatic questions) in a narrative. Returns setup-payoff pairs and promise rhythm."
    )]
    async fn detect_commitments(
        &self,
        Parameters(req): Parameters<DetectCommitmentsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.detect_commitments(&req.narrative_id).await)
    }

    #[tool(
        name = "get_commitment_rhythm",
        description = "D9.1: Get the promise rhythm (per-chapter tension curve, fulfillment ratio) for a narrative."
    )]
    async fn get_commitment_rhythm(
        &self,
        Parameters(req): Parameters<GetCommitmentRhythmRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_commitment_rhythm(&req.narrative_id).await)
    }

    #[tool(
        name = "extract_fabula",
        description = "D9.2: Extract the fabula (chronological event ordering) from a narrative's Allen constraints."
    )]
    async fn extract_fabula(
        &self,
        Parameters(req): Parameters<ExtractFabulaRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.extract_fabula(&req.narrative_id).await)
    }

    #[tool(
        name = "extract_sjuzet",
        description = "D9.2: Extract the sjužet (discourse/telling order) from a narrative's chapter structure."
    )]
    async fn extract_sjuzet(
        &self,
        Parameters(req): Parameters<ExtractSjuzetRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.extract_sjuzet(&req.narrative_id).await)
    }

    #[tool(
        name = "suggest_reordering",
        description = "D9.2: Suggest alternative sjužet orderings that maximize dramatic irony or commitment tension."
    )]
    async fn suggest_reordering(
        &self,
        Parameters(req): Parameters<SuggestReorderingRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.suggest_reordering(&req.narrative_id).await)
    }

    #[tool(
        name = "compute_dramatic_irony",
        description = "D9.3: Compute the dramatic irony map (reader vs character knowledge gaps) for a narrative."
    )]
    async fn compute_dramatic_irony(
        &self,
        Parameters(req): Parameters<ComputeDramaticIronyRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.compute_dramatic_irony(&req.narrative_id).await)
    }

    #[tool(
        name = "detect_focalization",
        description = "D9.3: Detect focalization (POV) segments and switch patterns across a narrative."
    )]
    async fn detect_focalization(
        &self,
        Parameters(req): Parameters<DetectFocalizationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.detect_focalization(&req.narrative_id).await)
    }

    #[tool(
        name = "detect_character_arc",
        description = "D9.4: Detect character arc type (positive change, flat, negative corruption, etc.) and transformation trajectory."
    )]
    async fn detect_character_arc(
        &self,
        Parameters(req): Parameters<DetectCharacterArcRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .detect_character_arc(&req.narrative_id, req.character_id.as_deref())
                .await,
        )
    }

    #[tool(
        name = "detect_subplots",
        description = "D9.4: Detect subplots via community detection on the situation interaction graph."
    )]
    async fn detect_subplots(
        &self,
        Parameters(req): Parameters<DetectSubplotsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.detect_subplots(&req.narrative_id).await)
    }

    #[tool(
        name = "classify_scene_sequel",
        description = "D9.4: Classify scene-sequel rhythm (Swain/Bickham) and compute pacing score."
    )]
    async fn classify_scene_sequel(
        &self,
        Parameters(req): Parameters<ClassifySceneSequelRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.classify_scene_sequel(&req.narrative_id).await)
    }

    #[cfg(feature = "generation")]
    #[tool(
        name = "generate_narrative_plan",
        description = "D9.6: Generate a formal narrative plan (entities, situations, commitments, subplots, arcs) from a premise."
    )]
    async fn generate_narrative_plan(
        &self,
        Parameters(req): Parameters<GenerateNarrativePlanRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_narrative_plan(
                    &req.premise,
                    &req.genre,
                    req.chapter_count,
                    req.subplot_count,
                )
                .await,
        )
    }

    #[cfg(feature = "generation")]
    #[tool(
        name = "materialize_plan",
        description = "D9.6: Materialize a narrative plan into the TENSA hypergraph as real entities, situations, and relationships."
    )]
    async fn materialize_plan(
        &self,
        Parameters(req): Parameters<MaterializePlanRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.materialize_plan(&req.plan_id).await)
    }

    #[cfg(feature = "generation")]
    #[tool(
        name = "validate_materialized_narrative",
        description = "D9.6: Validate a materialized narrative for temporal, causal, knowledge, and commitment consistency."
    )]
    async fn validate_materialized_narrative(
        &self,
        Parameters(req): Parameters<ValidateMaterializedRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .validate_materialized_narrative(&req.narrative_id)
                .await,
        )
    }

    #[cfg(feature = "generation")]
    #[tool(
        name = "generate_chapter",
        description = "D9.7: Prepare chapter generation by querying the materialized hypergraph for situation specs."
    )]
    async fn generate_chapter(
        &self,
        Parameters(req): Parameters<GenerateChapterRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_chapter(
                    &req.narrative_id,
                    req.chapter,
                    req.voice_description.as_deref(),
                )
                .await,
        )
    }

    #[cfg(feature = "generation")]
    #[tool(
        name = "generate_narrative",
        description = "D9.7: Prepare full narrative generation from a materialized hypergraph with style targeting."
    )]
    async fn generate_narrative(
        &self,
        Parameters(req): Parameters<GenerateNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_narrative(
                    &req.narrative_id,
                    req.chapter_count,
                    req.voice_description.as_deref(),
                )
                .await,
        )
    }

    #[cfg(feature = "generation")]
    #[tool(
        name = "generate_chapter_with_fitness",
        description = "Submit an LLM-driven chapter generation job that retries until a fitness threshold is met. Returns {job_id, status}; poll job_status / job_result to retrieve the generated chapter and the fitness loop log. The optional target_fingerprint_source narrative ID supplies the style target the loop is fitted against."
    )]
    async fn generate_chapter_with_fitness(
        &self,
        Parameters(req): Parameters<GenerateChapterWithFitnessRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_chapter_with_fitness(
                    &req.narrative_id,
                    req.chapter,
                    req.voice_description.as_deref(),
                    req.style_embedding_id.as_deref(),
                    req.target_fingerprint_source.as_deref(),
                    req.fitness_threshold,
                    req.max_retries,
                    req.temperature,
                )
                .await,
        )
    }

    // ─── Sprint D12: Adversarial Narrative Wargaming ─────────────

    #[tool(
        name = "generate_adversary_policy",
        description = "Generate an adversary policy: given an actor ID or archetype, produce a ranked list of adversary actions using SUQR bounded rationality over IRL reward weights. Returns actions sorted by expected reward, subject to operational constraints (budget, platforms, working hours, opsec). Stores result at adv/policy/{narrative_id}/{entity_id}."
    )]
    async fn generate_adversary_policy(
        &self,
        Parameters(req): Parameters<GenerateAdversaryPolicyRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_adversary_policy(
                    &req.narrative_id,
                    req.actor_id.as_deref(),
                    req.archetype.as_deref(),
                    req.lambda,
                    req.lambda_cap,
                    req.reward_weights,
                )
                .await,
        )
    }

    #[tool(
        name = "configure_rationality",
        description = "Configure the rationality model for game-theoretic analysis. Models: 'qre' (Quantal Response Equilibrium), 'suqr' (Subjective Utility QR with feature weights), 'cognitive_hierarchy' (Poisson CH with tau parameter). Returns the configured model parameters."
    )]
    async fn configure_rationality(
        &self,
        Parameters(req): Parameters<ConfigureRationalityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .configure_rationality(
                    &req.model,
                    req.lambda,
                    req.lambda_cap,
                    req.tau,
                    req.feature_weights,
                )
                .await,
        )
    }

    #[tool(
        name = "create_wargame",
        description = "Create a new wargame session: fork a narrative into a mutable simulation with red/blue teams, SMIR compartments, and configurable objectives. Returns a session_id for subsequent moves."
    )]
    async fn create_wargame(
        &self,
        Parameters(req): Parameters<CreateWargameRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .create_wargame(
                    &req.narrative_id,
                    req.max_turns,
                    req.time_step_minutes,
                    req.auto_red,
                    req.auto_blue,
                )
                .await,
        )
    }

    #[tool(
        name = "submit_wargame_move",
        description = "Submit red and/or blue team moves for one wargame turn. Validates moves, applies effects to SMIR compartments, advances time, and evaluates objectives. Returns turn result with R₀, misinformed counts, and objectives met."
    )]
    async fn submit_wargame_move(
        &self,
        Parameters(req): Parameters<SubmitWargameMoveRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .submit_wargame_move(&req.session_id, req.red_moves, req.blue_moves)
                .await,
        )
    }

    #[tool(
        name = "get_wargame_state",
        description = "Get the current state summary of a wargame session: turn number, R₀, misinformed/susceptible totals, objectives met, and move count."
    )]
    async fn get_wargame_state(
        &self,
        Parameters(req): Parameters<GetWargameStateRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_wargame_state(&req.session_id).await)
    }

    #[tool(
        name = "auto_play_wargame",
        description = "Auto-play N turns of a wargame session using heuristic AI for auto-controlled teams. Returns an array of turn results showing state evolution."
    )]
    async fn auto_play_wargame(
        &self,
        Parameters(req): Parameters<AutoPlayWargameRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .auto_play_wargame(&req.session_id, req.num_turns.unwrap_or(5))
                .await,
        )
    }

    // ─── Sprint W6: Writer Workflows ─────────────────────────────

    #[tool(
        name = "get_narrative_plan",
        description = "Get the narrative plan (logline, synopsis, plot beats, style targets, length targets, setting notes, themes, comp titles, target audience, custom fields) for a narrative. Returns null if no plan is set. Use this to understand the writer's intent before generating or editing."
    )]
    async fn get_narrative_plan(
        &self,
        Parameters(req): Parameters<WriterNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_narrative_plan(&req.narrative_id).await)
    }

    #[tool(
        name = "upsert_narrative_plan",
        description = "Create or update a narrative plan (PATCH semantics — fields omitted are preserved). `patch` may contain any subset of: \
            \n  logline (string), synopsis (string), premise (string), \
            \n  themes (array<string>), central_conflict (string), \
            \n  plot_beats: array of {label: string (required), description: string (required), target_chapter?: number}, \
            \n  style: {pov?: string, tense?: string, tone?: array<string>, voice?: string, reading_level?: string, influences?: array<string>, avoid?: array<string>}, \
            \n  length: {kind?: string (\"novel\"|\"novella\"|\"short-story\"|\"screenplay\"|\"serialized\"), target_words?: number, min_words?: number, max_words?: number, target_chapters?: number, target_scenes_per_chapter?: number}, \
            \n  setting: {time_period?: string, locations?: array<string>, world_notes?: string (markdown), research_notes?: string (markdown)}, \
            \n  notes (string, markdown), target_audience (string), comp_titles (array<string>), content_warnings (array<string>), \
            \n  custom: object — freeform key-value escape hatch; values may be strings, numbers, booleans, arrays, or nested objects. \
            \n Returns the full updated NarrativePlan."
    )]
    async fn upsert_narrative_plan(
        &self,
        Parameters(req): Parameters<UpsertPlanRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .upsert_narrative_plan(&req.narrative_id, req.patch)
                .await,
        )
    }

    #[tool(
        name = "get_writer_workspace",
        description = "Get the writer workspace dashboard for a narrative: counts (situations, entities, arcs, pinned_facts, total_words), plan status, last revision, and up to 4 ranked next-step suggestions (generate_outline, run_workshop, pin_facts, review_findings, commit_revision). Call this first to orient before a writing session."
    )]
    async fn get_writer_workspace(
        &self,
        Parameters(req): Parameters<WriterNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_writer_workspace(&req.narrative_id).await)
    }

    #[tool(
        name = "run_workshop",
        description = "Run a tiered workshop analysis pass on a narrative. Tiers: 'cheap' (instant, deterministic findings only — pacing anomalies, continuity gaps, stub characters, structural holes); 'standard' (cheap + LLM-enriched reviews of the top findings per focus); 'deep' (async job; returns immediately with a deferred flag — poll via `list_workshop_reports` / `get_workshop_report`). Focuses: pacing, continuity, characterization, prose, structure. Returns WorkshopReport with findings (severity, headline, evidence, suggestion, optional one-click suggested_edit) and cost breakdown."
    )]
    async fn run_workshop(
        &self,
        Parameters(req): Parameters<RunWorkshopRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .run_workshop(&req.narrative_id, &req.tier, req.focuses, req.max_llm_calls)
                .await,
        )
    }

    #[tool(
        name = "list_pinned_facts",
        description = "List all pinned continuity facts for a narrative. Pinned facts are canonical key/value pairs (e.g. 'Hair_color: blond', 'Ship_name: Revenant') that the continuity checker compares proposed prose against."
    )]
    async fn list_pinned_facts(
        &self,
        Parameters(req): Parameters<WriterNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_pinned_facts(&req.narrative_id).await)
    }

    #[tool(
        name = "create_pinned_fact",
        description = "Pin a canonical fact for continuity checking. `fact` must include: key (string, e.g. 'Hair_color'), value (string, e.g. 'blond'). Optional: note (string, free-form context), entity_id (UUID — links the fact to a specific cast member). Pinned facts are cheap to create — prefer over-pinning to under-pinning."
    )]
    async fn create_pinned_fact(
        &self,
        Parameters(req): Parameters<CreatePinnedFactRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .create_pinned_fact(&req.narrative_id, req.fact)
                .await,
        )
    }

    #[tool(
        name = "check_continuity",
        description = "Scan a block of prose for conflicts with pinned facts. Returns an array of ContinuityWarning objects with `severity` (either `conflict` — direct contradiction with a pinned fact — or `advisory` — potential issue worth reviewing), `headline`, `detail`, `pinned_fact_id`, `proposed_value`. Use before committing generated prose to catch canon drift. Deterministic — no LLM cost."
    )]
    async fn check_continuity(
        &self,
        Parameters(req): Parameters<CheckContinuityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .check_continuity(&req.narrative_id, &req.prose)
                .await,
        )
    }

    #[tool(
        name = "list_narrative_revisions",
        description = "List git-like revisions for a narrative, newest first. Each summary has id, parent_id, message, author, created_at, content_hash, and counts (situations/entities/participations/arcs). Use `limit` to bound the response for large histories."
    )]
    async fn list_narrative_revisions(
        &self,
        Parameters(req): Parameters<ListRevisionsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .list_narrative_revisions(&req.narrative_id, req.limit)
                .await,
        )
    }

    #[tool(
        name = "restore_narrative_revision",
        description = "Restore the narrative to a previous revision. Required params: `narrative_id`, `revision_id`, and `author` (string — attribution for the automatic safety commit created before the restore). Auto-commits current state first (safety net), then replays the target revision's snapshot. Returns restored_from, auto_commit (id of safety commit or null if no-op), and counts restored."
    )]
    async fn restore_narrative_revision(
        &self,
        Parameters(req): Parameters<RestoreRevisionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .restore_narrative_revision(&req.narrative_id, &req.revision_id, &req.author)
                .await,
        )
    }

    #[tool(
        name = "get_writer_cost_summary",
        description = "Aggregate the writer cost ledger for a narrative over a time window. `window` format: '24h', '7d', '30d', or 'all' (default 'all'). Returns total_calls, cache_hits, total_prompt_tokens, total_response_tokens, total_duration_ms, and a per-operation breakdown (Generation, Edit, Workshop, Continuity)."
    )]
    async fn get_writer_cost_summary(
        &self,
        Parameters(req): Parameters<CostSummaryRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_writer_cost_summary(&req.narrative_id, req.window.as_deref())
                .await,
        )
    }

    #[tool(
        name = "set_situation_content",
        description = "Replace a situation's prose content. Primary tool for the assistant-as-author co-writing loop (Workflow 2). `content` accepts either a plain string (wrapped as one Text block) or a JSON array of content blocks `[{content_type, content}]` where `content_type` is Text|Dialogue|Observation|Document|MediaRef. The new content replaces `raw_content` wholesale — call `commit_narrative_revision` first to preserve the prior prose in history. Optional `status` stamps the writer workflow status (e.g. 'first-draft', 'revised'). Run `check_continuity` on the new prose before committing."
    )]
    async fn set_situation_content(
        &self,
        Parameters(req): Parameters<SetSituationContentRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .set_situation_content(&req.situation_id, req.content, req.status.as_deref())
                .await,
        )
    }

    #[tool(
        name = "get_scene_context",
        description = "Return a packaged per-scene context bundle in one call: plan (logline, synopsis, active plot_beats, style, length), pinned facts, optional POV character profile (entity + pinned facts + recent participations), the current situation (if `situation_id` provided), and the N preceding scenes in manuscript order (fallback: temporal start). Reduces per-scene MCP round-trips from ~5 to 1 during drafting. `lookback_scenes` defaults to 2, max 10."
    )]
    async fn get_scene_context(
        &self,
        Parameters(req): Parameters<GetSceneContextRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_scene_context(
                    &req.narrative_id,
                    req.situation_id.as_deref(),
                    req.pov_entity_id.as_deref(),
                    req.lookback_scenes,
                )
                .await,
        )
    }

    // ─── Sprint W15: Writer MCP bridge ────────────────────────

    #[tool(
        name = "create_annotation",
        description = "Create an inline annotation (Comment, Footnote, or Citation) anchored to a byte-span of a scene's concatenated prose. `kind` is one of Comment | Footnote | Citation. span_start/span_end are byte offsets into the scene's Text+Dialogue blocks joined by newlines (detached=false on creation). Comments never ship in compile output; footnotes and citations render via the active compile profile."
    )]
    async fn create_annotation(
        &self,
        Parameters(req): Parameters<CreateAnnotationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .create_annotation(
                    &req.situation_id,
                    &req.kind,
                    &req.body,
                    req.span_start,
                    req.span_end,
                    req.source_id.as_deref(),
                    req.chunk_id.as_deref(),
                    req.author.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "list_annotations",
        description = "List annotations by scope. Provide exactly one of `situation_id` (scene-scoped) or `narrative_id` (every annotation across the narrative, sorted by situation then span)."
    )]
    async fn list_annotations(
        &self,
        Parameters(req): Parameters<ListAnnotationsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .list_annotations(req.situation_id.as_deref(), req.narrative_id.as_deref())
                .await,
        )
    }

    #[tool(
        name = "update_annotation",
        description = "Patch an annotation by id. `patch` accepts: body (string), span ([start,end] tuple), source_id (UUID or null), chunk_id (UUID or null), detached (bool). Updating span resets the detached flag."
    )]
    async fn update_annotation(
        &self,
        Parameters(req): Parameters<UpdateAnnotationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .update_annotation(&req.annotation_id, req.patch)
                .await,
        )
    }

    #[tool(
        name = "delete_annotation",
        description = "Delete an annotation permanently by id. Returns {status: ok, id}."
    )]
    async fn delete_annotation(
        &self,
        Parameters(req): Parameters<DeleteAnnotationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.delete_annotation(&req.annotation_id).await)
    }

    #[tool(
        name = "create_collection",
        description = "Create a Collection (saved search) for a narrative. `query` is a structured filter object with optional fields: labels (array<string>, match-in), statuses (array<string>), keywords_any (array<string>, intersection), text (substring over name+synopsis+description), min_order/max_order (manuscript_order bounds), min_words/max_words (word-count bounds). Empty / missing = match all."
    )]
    async fn create_collection(
        &self,
        Parameters(req): Parameters<CreateCollectionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .create_collection(
                    &req.narrative_id,
                    &req.name,
                    req.description.as_deref(),
                    req.query,
                )
                .await,
        )
    }

    #[tool(
        name = "list_collections",
        description = "List all Collections for a narrative, sorted by name."
    )]
    async fn list_collections(
        &self,
        Parameters(req): Parameters<ListCollectionsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_collections(&req.narrative_id).await)
    }

    #[tool(
        name = "get_collection",
        description = "Get a Collection by id. When `resolve=true`, the response also includes `resolution` with the current matching situation UUIDs and count."
    )]
    async fn get_collection(
        &self,
        Parameters(req): Parameters<GetCollectionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_collection(&req.collection_id, req.resolve)
                .await,
        )
    }

    #[tool(
        name = "update_collection",
        description = "Patch a Collection. `patch` accepts: name (string), description (string or null), query (full CollectionQuery object)."
    )]
    async fn update_collection(
        &self,
        Parameters(req): Parameters<UpdateCollectionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .update_collection(&req.collection_id, req.patch)
                .await,
        )
    }

    #[tool(name = "delete_collection", description = "Delete a Collection by id.")]
    async fn delete_collection(
        &self,
        Parameters(req): Parameters<DeleteCollectionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.delete_collection(&req.collection_id).await)
    }

    #[tool(
        name = "create_research_note",
        description = "Pin a research note to a scene. Kinds: Quote | Clipping | Link | Note (default Note). Body is required and non-empty. Research notes are the writer's margin-of-page workspace, distinct from registered Sources — lightweight, per-scene, freeform."
    )]
    async fn create_research_note(
        &self,
        Parameters(req): Parameters<CreateResearchNoteRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .create_research_note(
                    &req.narrative_id,
                    &req.situation_id,
                    &req.kind,
                    &req.body,
                    req.source_id.as_deref(),
                    req.author.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "list_research_notes",
        description = "List research notes by scope. Provide exactly one of `situation_id` (scene-scoped) or `narrative_id` (all notes in the narrative). Sorted by created_at."
    )]
    async fn list_research_notes(
        &self,
        Parameters(req): Parameters<ListResearchNotesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .list_research_notes(req.situation_id.as_deref(), req.narrative_id.as_deref())
                .await,
        )
    }

    #[tool(name = "get_research_note", description = "Get a research note by id.")]
    async fn get_research_note(
        &self,
        Parameters(req): Parameters<GetResearchNoteRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_research_note(&req.note_id).await)
    }

    #[tool(
        name = "update_research_note",
        description = "Patch a research note. Recognised keys: kind (Quote|Clipping|Link|Note), body (non-empty string), author (string or null), source_id (UUID or null)."
    )]
    async fn update_research_note(
        &self,
        Parameters(req): Parameters<UpdateResearchNoteRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .update_research_note(&req.note_id, req.patch)
                .await,
        )
    }

    #[tool(name = "delete_research_note", description = "Delete a research note.")]
    async fn delete_research_note(
        &self,
        Parameters(req): Parameters<DeleteResearchNoteRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.delete_research_note(&req.note_id).await)
    }

    #[tool(
        name = "promote_chunk_to_note",
        description = "Promote a ChunkStore chunk (source-derived text) to a scene-scoped research note. Sets `source_chunk_id` back-link so the note can navigate back to the origin chunk. Default kind is Quote."
    )]
    async fn promote_chunk_to_note(
        &self,
        Parameters(req): Parameters<PromoteChunkToNoteRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .promote_chunk_to_note(
                    &req.narrative_id,
                    &req.situation_id,
                    &req.chunk_id,
                    &req.body,
                    req.source_id.as_deref(),
                    req.kind.as_deref(),
                    req.author.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "propose_edit",
        description = "Propose an LLM-generated edit to a scene. When `style_preset` is set (minimal | lyrical | punchy | formal | interior | cinematic) the operation is StyleTransfer; otherwise it's a freeform Rewrite using `instruction`. Returns an EditProposal with original/proposed blocks, word-count delta, unified line diff, and optional rationale. Requires a session-capable LLM (OpenRouter / Local)."
    )]
    async fn propose_edit(
        &self,
        Parameters(req): Parameters<ProposeEditRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .propose_edit(
                    &req.situation_id,
                    &req.instruction,
                    req.style_preset.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "apply_edit",
        description = "Write a proposed edit into the hypergraph and commit a revision. `proposal` must be a full EditProposal (round-tripped from `propose_edit`). Returns AppliedEditReport with revision_id and words_before/words_after."
    )]
    async fn apply_edit(
        &self,
        Parameters(req): Parameters<ApplyEditRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .apply_edit(req.proposal, req.author.as_deref())
                .await,
        )
    }

    #[tool(
        name = "estimate_edit_tokens",
        description = "Estimate prompt_tokens + expected_response_tokens for a prospective scene edit, without calling the LLM. Cheap pre-flight before `propose_edit`."
    )]
    async fn estimate_edit_tokens(
        &self,
        Parameters(req): Parameters<EstimateEditTokensRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .estimate_edit_tokens(
                    &req.situation_id,
                    &req.instruction,
                    req.style_preset.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "commit_narrative_revision",
        description = "Commit the current narrative state as a new revision (git-like). No-op if the snapshot is byte-identical to HEAD; in that case `outcome=no_change` and the existing HEAD summary is returned."
    )]
    async fn commit_narrative_revision(
        &self,
        Parameters(req): Parameters<CommitRevisionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .commit_narrative_revision(&req.narrative_id, &req.message, req.author.as_deref())
                .await,
        )
    }

    #[tool(
        name = "diff_narrative_revisions",
        description = "Diff two revisions of a narrative. Returns structural delta (situations/entities/participations added/modified/removed), prose hunks (unified line diff per scene), and per-scene summaries (word delta, change kind). Both revisions are validated to belong to `narrative_id`."
    )]
    async fn diff_narrative_revisions(
        &self,
        Parameters(req): Parameters<DiffRevisionsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .diff_narrative_revisions(&req.narrative_id, &req.from_rev, &req.to_rev)
                .await,
        )
    }

    #[tool(
        name = "list_workshop_reports",
        description = "List past workshop reports for a narrative (newest first), summarising the tier, focuses, finding counts, and cost per run."
    )]
    async fn list_workshop_reports(
        &self,
        Parameters(req): Parameters<ListWorkshopReportsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_workshop_reports(&req.narrative_id).await)
    }

    #[tool(
        name = "get_workshop_report",
        description = "Fetch a specific workshop report by id, including all findings with severities, evidence, suggestions, and any attached one-click suggested_edit objects."
    )]
    async fn get_workshop_report(
        &self,
        Parameters(req): Parameters<GetWorkshopReportRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.get_workshop_report(&req.report_id).await)
    }

    #[tool(
        name = "list_cost_ledger_entries",
        description = "List the raw cost-ledger entries (one per LLM call) for a narrative, newest first, bounded by `limit` (default 50, max 500). Each entry has operation, kind, prompt_tokens, response_tokens, model, cache_hit, success, duration_ms, timestamp. Use `get_writer_cost_summary` for rolled-up totals."
    )]
    async fn list_cost_ledger_entries(
        &self,
        Parameters(req): Parameters<ListCostLedgerRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .list_cost_ledger_entries(&req.narrative_id, req.limit)
                .await,
        )
    }

    #[tool(
        name = "list_compile_profiles",
        description = "List all saved compile profiles for a narrative. Each profile captures the include/exclude rules, heading templates, front/back matter, footnote style, and compile-comments flag."
    )]
    async fn list_compile_profiles(
        &self,
        Parameters(req): Parameters<ListCompileProfilesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_compile_profiles(&req.narrative_id).await)
    }

    #[tool(
        name = "compile_narrative",
        description = "Compile a narrative through the given profile (or a synthesised default if profile_id is omitted). Formats: markdown (default, UTF-8 body) | epub | docx (base64-encoded body). Returns {narrative_id, format, content_type, bytes, body, encoding}."
    )]
    async fn compile_narrative(
        &self,
        Parameters(req): Parameters<CompileNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .compile_narrative(&req.narrative_id, &req.format, req.profile_id.as_deref())
                .await,
        )
    }

    #[tool(
        name = "upsert_compile_profile",
        description = "Create a new compile profile (omit profile_id; `patch` must include `name`) or patch an existing one (provide profile_id; `patch` keys: name, description, include_labels, exclude_labels, include_statuses, heading_templates, front_matter_md, back_matter_md, footnote_style, include_comments)."
    )]
    async fn upsert_compile_profile(
        &self,
        Parameters(req): Parameters<UpsertCompileProfileRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .upsert_compile_profile(&req.narrative_id, req.profile_id.as_deref(), req.patch)
                .await,
        )
    }

    #[tool(
        name = "list_narrative_templates",
        description = "List reusable narrative scaffolding templates — the three builtins (The Mentor's Death, The False Victory, The Information Marketplace) plus any user-stored templates. Each template exposes typed slots the caller binds to entities via `instantiate_template`."
    )]
    async fn list_narrative_templates(&self) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.list_narrative_templates().await)
    }

    #[tool(
        name = "instantiate_template",
        description = "Bind a template's slots to specific entity UUIDs and return the planned situations. Does NOT write to the hypergraph — the caller reviews the InstantiatedSituation list and creates situations + participations separately. `bindings` is a map of `slot_id -> entity_uuid`; every declared slot must be bound."
    )]
    async fn instantiate_template(
        &self,
        Parameters(req): Parameters<InstantiateTemplateRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .instantiate_template(&req.template_id, req.bindings)
                .await,
        )
    }

    #[tool(
        name = "extract_narrative_skeleton",
        description = "Extract a compact structural skeleton (entity slots, situation slots, commitment positions) for a narrative. Primarily used as input to cross-narrative similarity. Embedded backend only — HTTP backend returns an error."
    )]
    async fn extract_narrative_skeleton(
        &self,
        Parameters(req): Parameters<WriterNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .extract_narrative_skeleton(&req.narrative_id)
                .await,
        )
    }

    #[tool(
        name = "find_duplicate_candidates",
        description = "Propose entity merge candidates for a narrative. Does NOT actually merge — returns a list of candidate pairs with a similarity score so the caller (or UI) can accept/reject. Default threshold 0.7, max 200 candidates. Use `merge_entities` to commit an accepted pair."
    )]
    async fn find_duplicate_candidates(
        &self,
        Parameters(req): Parameters<FindDuplicatesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .find_duplicate_candidates(&req.narrative_id, req.threshold, req.max_candidates)
                .await,
        )
    }

    #[tool(
        name = "suggest_narrative_fixes",
        description = "Suggest structural fixes for the most recent diagnosis of a narrative (diagnoses first if none stored). Returns fixes keyed to NarrativePathology instances: mark-red-herring, add-commitment, move-situation, fulfill-commitment, etc. Pass the chosen SuggestedFix to `apply_narrative_fix`."
    )]
    async fn suggest_narrative_fixes(
        &self,
        Parameters(req): Parameters<WriterNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .suggest_narrative_fixes(&req.narrative_id)
                .await,
        )
    }

    #[tool(
        name = "apply_narrative_fix",
        description = "Apply a single SuggestedFix produced by `suggest_narrative_fixes`. Each fix type mutates the hypergraph deterministically (e.g. mark a commitment as RedHerringResolved). Returns the FixResult with applied changes."
    )]
    async fn apply_narrative_fix(
        &self,
        Parameters(req): Parameters<ApplyNarrativeFixRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .apply_narrative_fix(&req.narrative_id, req.fix)
                .await,
        )
    }

    #[tool(
        name = "apply_reorder",
        description = "Batch-reorder scenes inside a narrative. `entries` is an array of {situation_id, parent_id?} in the new desired order. Writes manuscript_order + parent_situation_id atomically per-situation, densifying positions to 1000, 2000, 3000, … so drag-inserts don't cascade. Max 10,000 entries. Returns {applied_count, conflicts}."
    )]
    async fn apply_reorder(
        &self,
        Parameters(req): Parameters<ApplyReorderRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .apply_reorder(&req.narrative_id, req.entries)
                .await,
        )
    }

    // ─── EATH Phase 10 — Synthetic Hypergraph MCP tools ──────────

    #[tool(
        name = "calibrate_surrogate",
        description = "Submit a SurrogateCalibration job that fits the named surrogate model (default \"eath\") to the source narrative. Returns { job_id, status }. Poll with `job_status` / `job_result`. The calibrated params drive `generate_synthetic_narrative` and the K-loop in `compute_pattern_significance`."
    )]
    async fn calibrate_surrogate(
        &self,
        Parameters(req): Parameters<CalibrateSurrogateRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .calibrate_surrogate(&req.narrative_id, req.model.as_deref())
                .await,
        )
    }

    #[tool(
        name = "generate_synthetic_narrative",
        description = "Submit a SurrogateGeneration job. Generates a synthetic narrative using calibrated params from `source_narrative_id` (or inline `params` for source-less generation), writing into `output_narrative_id`. Optional `seed` enables deterministic replay; `num_steps` (default 100) controls trajectory length; `label_prefix` (default \"synth\") controls the name pattern of emitted entities. Returns { job_id, status }."
    )]
    async fn generate_synthetic_narrative(
        &self,
        Parameters(req): Parameters<GenerateSyntheticNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .generate_synthetic_narrative(
                    &req.source_narrative_id,
                    &req.output_narrative_id,
                    req.model.as_deref(),
                    req.params,
                    req.seed,
                    req.num_steps,
                    req.label_prefix.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "generate_hybrid_narrative",
        description = "Submit a SurrogateHybridGeneration job. Generates a synthetic narrative as a weighted **mixture** of N calibrated surrogate processes — at each step, sample a source from the multinomial defined by `components[i].weight` and recruit one hyperedge using that source's params. Weights MUST sum to 1.0 within 1e-6. Returns { job_id, status }."
    )]
    async fn generate_hybrid_narrative(
        &self,
        Parameters(req): Parameters<GenerateHybridNarrativeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Convert the JsonSchema mirror struct into the engine's serde-compatible
        // shape — same field names, so to_value passes through unchanged.
        let components_json = match serde_json::to_value(&req.components) {
            Ok(v) => v,
            Err(e) => {
                return Ok(error_result(crate::error::TensaError::InvalidInput(
                    format!("invalid components: {e}"),
                )));
            }
        };
        wrap(
            self.backend
                .generate_hybrid_narrative(
                    components_json,
                    &req.output_narrative_id,
                    req.seed,
                    req.num_steps,
                )
                .await,
        )
    }

    #[tool(
        name = "list_synthetic_runs",
        description = "List previously-submitted synthetic generation runs for a narrative, newest first. Each entry is a SurrogateRunSummary with run_id, model, kind (Calibration/Generation/Hybrid), output_narrative_id, source_narrative_id, params_hash, started_at/finished_at, duration_ms, and emit counters. `limit` defaults to 50, max 1000."
    )]
    async fn list_synthetic_runs(
        &self,
        Parameters(req): Parameters<ListSyntheticRunsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .list_synthetic_runs(&req.narrative_id, req.limit)
                .await,
        )
    }

    #[tool(
        name = "get_fidelity_report",
        description = "Fetch the FidelityReport for a calibrated run — per-metric KS / Spearman / MAE statistics, overall_score, passed flag, and thresholds_provenance (Default | UserOverride | StudyCalibrated). Both `narrative_id` and `run_id` are required — fidelity is keyed by the (narrative, run) pair. Returns 404-equivalent error if no report exists."
    )]
    async fn get_fidelity_report(
        &self,
        Parameters(req): Parameters<GetFidelityReportRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .get_fidelity_report(&req.narrative_id, &req.run_id)
                .await,
        )
    }

    #[tool(
        name = "compute_pattern_significance",
        description = "Submit a SurrogateSignificance job comparing the source narrative against the EATH null distribution for a structural metric. `metric` must be one of: `temporal_motifs`, `communities`, `patterns`. (For contagion, use `simulate_higher_order_contagion`.) `k` is the number of synthetic samples (default 100, max 1000). Returns { job_id, status }; the completed report carries per-element z-scores and one-tailed empirical p-values."
    )]
    async fn compute_pattern_significance(
        &self,
        Parameters(req): Parameters<ComputePatternSignificanceRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .compute_pattern_significance(
                    &req.narrative_id,
                    &req.metric,
                    req.k,
                    req.model.as_deref(),
                    req.params_override,
                )
                .await,
        )
    }

    #[tool(
        name = "simulate_higher_order_contagion",
        description = "Submit a SurrogateContagionSignificance job. Runs K higher-order SIR simulations on the source narrative AND on K EATH null samples, then reports per-element z-scores and p-values. `params` is a HigherOrderSirParams blob (per-size beta, gamma, threshold rule, seed strategy, max_steps, rng_seed). `k` defaults to 100, capped at 1000. Returns { job_id, status }."
    )]
    async fn simulate_higher_order_contagion(
        &self,
        Parameters(req): Parameters<SimulateHigherOrderContagionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .simulate_higher_order_contagion(
                    &req.narrative_id,
                    req.params,
                    req.k,
                    req.model.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "compute_dual_significance",
        description = "EATH Extension Phase 13c — submit a SurrogateDualSignificance job that runs the K-loop ONCE per requested null model and reports per-model + combined significance verdicts. `metric` must be one of: `temporal_motifs`, `communities`, `patterns`. `k_per_model` defaults to 100 per model, capped at 1000. `models` defaults to `[\"eath\", \"nudhy\"]` when omitted — the canonical dual-null pair. Returns { job_id, status }; the completed report carries per-model {z_score, p_value, samples_used, starvations} rows AND the AND-reduced combined verdict (significant_vs_all_at_p05 / significant_vs_all_at_p01)."
    )]
    async fn compute_dual_significance(
        &self,
        Parameters(req): Parameters<ComputeDualSignificanceRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .compute_dual_significance(
                    &req.narrative_id,
                    &req.metric,
                    req.k_per_model,
                    req.models,
                )
                .await,
        )
    }

    #[tool(
        name = "compute_bistability_significance",
        description = "EATH Extension Phase 14 — submit a SurrogateBistabilitySignificance job. Runs a forward-backward β-sweep on the source narrative AND on K surrogate samples per requested null model, then reports per-model quantiles for the bistable_interval width and max_hysteresis_gap. `params` is a BistabilitySweepParams blob (beta_0_range, beta_scaling, gamma, threshold, initial_prevalence_low/high, steady_state_steps, replicates_per_beta). `k` defaults to 50 per model, capped at 500. `models` defaults to [\"eath\", \"nudhy\"]. Returns { job_id, status }; the completed report carries per-model quantile rows AND the AND-reduced combined verdict (source_bistable_wider_than_all_at_p05 / p01)."
    )]
    async fn compute_bistability_significance(
        &self,
        Parameters(req): Parameters<ComputeBistabilitySignificanceRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .compute_bistability_significance(
                    &req.narrative_id,
                    req.params,
                    req.k,
                    req.models,
                )
                .await,
        )
    }

    #[tool(
        name = "reconstruct_hypergraph",
        description = "EATH Extension Phase 15c — submit a HypergraphReconstruction job that recovers latent hyperedges from observed entity time-series via the THIS / SINDy method (Delabays et al., Nat. Commun. 16:2691, 2025; arXiv:2402.00078). `narrative_id` is the source narrative whose entity time-series to reconstruct from. `params` is an optional partial ReconstructionParams blob — every field has a serde default, so callers can omit the field entirely to get the engine's recommended defaults (max_order=3, observation=ParticipationRate, lambda auto-selected via the λ_max heuristic, bootstrap_k=10). Returns { job_id, status: \"Pending\" }. Poll the result via GET /inference/hypergraph-reconstruction/{job_id} or the standard /jobs/{id}/result envelope. Inferred hyperedges land in the result's `inferred_edges` field with a bootstrap confidence score; per architect §13.7 of docs/synth_reconstruction_algorithm.md, filter by `confidence > 0.7` rather than `weight > ε` to avoid Taylor-expansion masking artifacts. Use POST /inference/hypergraph-reconstruction/{job_id}/materialize (with opt_in:true) to commit the high-confidence edges as Situations under ExtractionMethod::Reconstructed."
    )]
    async fn reconstruct_hypergraph(
        &self,
        Parameters(req): Parameters<ReconstructHypergraphRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .reconstruct_hypergraph(&req.narrative_id, req.params)
                .await,
        )
    }

    #[tool(
        name = "simulate_opinion_dynamics",
        description = "EATH Extension Phase 16c — synchronously run one bounded-confidence opinion-dynamics simulation on the source narrative's entity-situation hypergraph. Implements the Hickok et al. 2022 PairwiseWithin variant (default) and the Schawe-Hernández 2022 GroupMean variant of BCM-on-hypergraphs (Deffuant 2000 dyadic update lifted to higher-order edges). `narrative_id` is the source narrative. `params` is an optional partial OpinionDynamicsParams blob — every field has a serde(default), so callers can omit it entirely to get the documented defaults (model=PairwiseWithin, confidence_bound=0.3, convergence_rate=0.5, hyperedge_selection=UniformRandom, initial_opinion_distribution=Uniform, convergence_tol=1e-4, convergence_window=100, max_steps=100k, seed=42). Returns inline { run_id, report } where report is the full OpinionDynamicsReport (num_clusters, polarization_index, echo_chamber_index, cluster_sizes, cluster_means, trajectory). Echo-chamber index requires precomputed label_propagation labels at an/lp/{narrative_id}/{entity_id}; when missing, echo_chamber_available=false (no error — the analyst sees the missing-data signal). Each run persists at opd/report/{narrative_id}/{run_id_v7}."
    )]
    async fn simulate_opinion_dynamics(
        &self,
        Parameters(req): Parameters<SimulateOpinionDynamicsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .simulate_opinion_dynamics(&req.narrative_id, req.params)
                .await,
        )
    }

    #[tool(
        name = "simulate_opinion_phase_transition",
        description = "EATH Extension Phase 16c — sweep the confidence bound `c` and report the per-c convergence times plus the inferred critical-c spike. On a complete hypergraph with N(0.5, σ²) initial opinions, convergence time spikes near c = σ² (Hickok §5). DISTINCT from Phase 14 bistability: that sweeps β and measures prevalence; this sweeps c and measures convergence time. `narrative_id` is the source narrative. `c_range` is `[c_start, c_end, num_points]` with `0 < c_start < c_end < 1` and `num_points >= 2`. `base_params` is an optional OpinionDynamicsParams blob applied to every per-c run (with `confidence_bound` overridden by the sweep). Returns inline PhaseTransitionReport { c_values, convergence_times, critical_c_estimate, initial_variance, spike_threshold }. Synchronous — no job queue."
    )]
    async fn simulate_opinion_phase_transition(
        &self,
        Parameters(req): Parameters<SimulateOpinionPhaseTransitionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .simulate_opinion_phase_transition(
                    &req.narrative_id,
                    req.c_range,
                    req.base_params,
                )
                .await,
        )
    }

    // ─── Fuzzy Sprint Phase 10 — hybrid fuzzy-probability MCP tool ──────
    //
    // Only this one tool lands in Phase 10 — Phase 11 adds the other 13
    // fuzzy MCP tools. Cites: [flaminio2026fsta] [faginhalpern1994fuzzyprob].
    #[tool(
        name = "fuzzy_probability",
        description = "Compute the Cao–Holčapek base-case fuzzy probability P_fuzzy(E) = Σ μ_E(e) · P(e) on a discrete distribution over entity UUIDs. `event` is a FuzzyEvent { predicate_kind: 'quantifier' | 'mamdani_rule' | 'custom', predicate_payload: {...} }. `distribution` is a ProbDist { kind: 'discrete', outcomes: [['<uuid>', p], ...] } with Σ P = 1.0 ± 1e-9. Runs synchronously, persists the result at fz/hybrid/{nid}/{query_id_BE_16}, and returns { value, event_kind, distribution_summary, query_id, narrative_id, tnorm }. `tnorm` is accepted for forward-compat (future composition phases consume it) but the Phase 10 base case ignores the override."
    )]
    async fn fuzzy_probability(
        &self,
        Parameters(req): Parameters<FuzzyProbabilityRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_probability(
                    &req.narrative_id,
                    req.event,
                    req.distribution,
                    req.tnorm.as_deref(),
                )
                .await,
        )
    }

    // ─── Fuzzy Sprint Phase 11 — 13 new fuzzy MCP tools ────────────
    //
    // Cites: [klement2000] [yager1988owa] [grabisch1996choquet]
    //        [duboisprade1989fuzzyallen] [novak2008quantifiers]
    //        [murinovanovak2014peterson] [belohlavek2004fuzzyfca]
    //        [mamdani1975mamdani].

    #[tool(
        name = "fuzzy_list_tnorms",
        description = "List the registered t-norm families (Gödel / Goguen / Łukasiewicz / Hamacher). Returns `{tnorms: [{name, description, formula, tconorm_formula, citation}]}`. Mirrors `GET /fuzzy/tnorms`."
    )]
    async fn fuzzy_list_tnorms(
        &self,
        Parameters(_): Parameters<FuzzyListTnormsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.fuzzy_list_tnorms().await)
    }

    #[tool(
        name = "fuzzy_list_aggregators",
        description = "List the registered aggregators (mean, median, owa, choquet, tnorm_reduce, tconorm_reduce). Returns `{aggregators: [{name, description, formula, required_params, citation}]}`. Mirrors `GET /fuzzy/aggregators`."
    )]
    async fn fuzzy_list_aggregators(
        &self,
        Parameters(_): Parameters<FuzzyListAggregatorsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.fuzzy_list_aggregators().await)
    }

    #[tool(
        name = "fuzzy_get_config",
        description = "Load the per-workspace default fuzzy config `{tnorm, aggregator, measure?, version}`. Falls back to the Gödel/Mean factory default when no config has been persisted. Mirrors `GET /fuzzy/config`."
    )]
    async fn fuzzy_get_config(
        &self,
        Parameters(_): Parameters<FuzzyGetConfigRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.fuzzy_get_config().await)
    }

    #[tool(
        name = "fuzzy_set_config",
        description = "Update the per-workspace default fuzzy config. Pass `reset: true` to restore Gödel/Mean factory defaults. Unknown t-norm / aggregator names surface as `InvalidInput` (HTTP 400 on REST). Mirrors `PUT /fuzzy/config`."
    )]
    async fn fuzzy_set_config(
        &self,
        Parameters(req): Parameters<FuzzySetConfigRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Convert nested Option<Option<String>> into the Option<Option<&str>>
        // shape the backend expects: outer None = leave unchanged, inner
        // None = explicitly clear.
        let measure: Option<Option<&str>> = req.measure.as_ref().map(|o| o.as_deref());
        wrap(
            self.backend
                .fuzzy_set_config(
                    req.tnorm.as_deref(),
                    req.aggregator.as_deref(),
                    measure,
                    req.reset,
                )
                .await,
        )
    }

    #[tool(
        name = "fuzzy_create_measure",
        description = "Persist a named monotone fuzzy measure μ : 2^N → [0,1]. Enforces μ(∅)=0, μ(N)=1, and monotonicity (rejections mention the literal word 'monotonicity'). `values` is in subset-bitmask order, `|values| = 2^n`. n ≤ 16. Mirrors `POST /fuzzy/measures`."
    )]
    async fn fuzzy_create_measure(
        &self,
        Parameters(req): Parameters<FuzzyCreateMeasureRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_create_measure(&req.name, req.n, req.values)
                .await,
        )
    }

    #[tool(
        name = "fuzzy_list_measures",
        description = "List every persisted fuzzy measure. Returns `{measures: [{name, measure}]}`. Mirrors `GET /fuzzy/measures`."
    )]
    async fn fuzzy_list_measures(
        &self,
        Parameters(_): Parameters<FuzzyListMeasuresRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.fuzzy_list_measures().await)
    }

    #[tool(
        name = "fuzzy_aggregate",
        description = "One-shot aggregation over a caller-supplied vector `xs` under the named aggregator (mean/median/owa/choquet/tnorm_reduce/tconorm_reduce). OWA requires `owa_weights` (length = |xs|, sum = 1 ± 1e-9). Choquet requires a `measure` reference and caps |xs| ≤ 16. Synchronous cap |xs| ≤ 1000. Mirrors `POST /fuzzy/aggregate`."
    )]
    async fn fuzzy_aggregate(
        &self,
        Parameters(req): Parameters<FuzzyAggregateRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_aggregate(
                    req.xs,
                    &req.aggregator,
                    req.tnorm.as_deref(),
                    req.measure.as_deref(),
                    req.owa_weights,
                    req.seed,
                )
                .await,
        )
    }

    #[tool(
        name = "fuzzy_allen_gradation",
        description = "Compute + cache the 13-dim graded Allen relation vector between two situations (Dubois-Prade fuzzy interval algebra). Returns `{relations: [{name, degree}]}` in Allen's canonical order. Mirrors `POST /analysis/fuzzy-allen`."
    )]
    async fn fuzzy_allen_gradation(
        &self,
        Parameters(req): Parameters<FuzzyAllenGradationRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_allen_gradation(&req.narrative_id, &req.a_id, &req.b_id)
                .await,
        )
    }

    #[tool(
        name = "fuzzy_quantify",
        description = "Evaluate a Novák-Murinová intermediate quantifier (`most`, `many`, `almost_all`, `few`) over a narrative's entity domain. Phase 6 scope is crisp predicates only: `confidence>0.7`, `maturity=Confirmed`, or a dotted property path. Empty `where` → every entity contributes μ=1. Mirrors `POST /fuzzy/quantify`."
    )]
    async fn fuzzy_quantify(
        &self,
        Parameters(req): Parameters<FuzzyQuantifyRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_quantify(
                    &req.narrative_id,
                    &req.quantifier,
                    req.entity_type.as_deref(),
                    req.r#where.as_deref(),
                    req.label.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "fuzzy_verify_syllogism",
        description = "Verify a graded Peterson syllogism (Murinová-Novák 2014 prototype). Statements use the tiny DSL `<QUANT> <subj> IS <obj>` — quantifiers are the Phase 6 names, subj/obj resolve via `type:Actor` / `entity` / `*` predicates. Persists a proof + returns `{proof_id, degree, figure, valid, threshold, tnorm}`. Mirrors `POST /fuzzy/syllogism/verify`."
    )]
    async fn fuzzy_verify_syllogism(
        &self,
        Parameters(req): Parameters<FuzzyVerifySyllogismRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_verify_syllogism(
                    &req.narrative_id,
                    &req.major,
                    &req.minor,
                    &req.conclusion,
                    req.threshold,
                    req.tnorm.as_deref(),
                    req.figure_hint.as_deref(),
                )
                .await,
        )
    }

    #[tool(
        name = "fuzzy_build_lattice",
        description = "Build + persist a graded concept lattice (Bělohlávek fuzzy FCA) for a narrative. `attribute_allowlist` restricts the formal context's attribute axis. `threshold` prunes concepts with |extent| below the given size. Soft cap 500×50; `large_context: true` to cross it (hard cap 2000×200). Mirrors `POST /fuzzy/fca/lattice`."
    )]
    async fn fuzzy_build_lattice(
        &self,
        Parameters(req): Parameters<FuzzyBuildLatticeRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_build_lattice(
                    &req.narrative_id,
                    req.entity_type.as_deref(),
                    req.attribute_allowlist,
                    req.threshold,
                    req.tnorm.as_deref(),
                    req.large_context,
                )
                .await,
        )
    }

    #[tool(
        name = "fuzzy_create_rule",
        description = "Create + persist a Mamdani fuzzy rule. `antecedent` is a list of `{variable_path, membership, linguistic_term}`; `consequent` is a single `{variable, membership, linguistic_term}`. Membership functions: Triangular/Trapezoidal/Gaussian. Mirrors `POST /fuzzy/rules`."
    )]
    async fn fuzzy_create_rule(
        &self,
        Parameters(req): Parameters<FuzzyCreateRuleRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_create_rule(
                    &req.name,
                    &req.narrative_id,
                    req.antecedent,
                    req.consequent,
                    req.tnorm.as_deref(),
                    req.enabled,
                )
                .await,
        )
    }

    // ─── Graded Acceptability Sprint Phase 5 — 5 new MCP tools ────
    //
    // Cites: [amgoud2013ranking] [besnard2001hcategoriser] [amgoud2017weighted]
    //        [grabisch1996choquet] [bustince2016choquet] [nebel1995ordhorn].

    #[tool(
        name = "argumentation_gradual",
        description = "Run gradual / ranking-based argumentation semantics synchronously over a narrative's contention framework. Mirrors `POST /analysis/argumentation/gradual`. `gradual_semantics` is the externally-tagged JSON shape: `\"HCategoriser\"` / `\"MaxBased\"` / `\"CardBased\"` / `{\"WeightedHCategoriser\":{\"weights\":[..]}}`. Optional `tnorm` is the t-norm-coupling override for the influence step (`{\"kind\":\"godel\"}` / `{\"kind\":\"hamacher\",\"param\":1.5}` / …); default `None` reproduces the canonical Gödel formulas bit-identically. Returns `{narrative_id, gradual:{acceptability, iterations, converged}, iterations, converged}`. Cites: [amgoud2013ranking] [besnard2001hcategoriser] [amgoud2017weighted]."
    )]
    async fn argumentation_gradual(
        &self,
        Parameters(req): Parameters<ArgumentationGradualRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .argumentation_gradual(&req.narrative_id, req.gradual_semantics, req.tnorm)
                .await,
        )
    }

    #[tool(
        name = "fuzzy_learn_measure",
        description = "Fit + persist a Choquet fuzzy measure from a `(input_vec, rank)` ranking-supervised dataset via in-tree pure-Rust PGD. Mirrors `POST /fuzzy/measures/learn`. Caps `n ≤ 6` (k-additive specialisation required for larger universes). Re-training under an existing name auto-increments the version; both the versionless latest pointer and the versioned history slice are written. Returns `{name, version, n, provenance, train_auc, test_auc}`. Cites: [grabisch1996choquet] [bustince2016choquet]."
    )]
    async fn fuzzy_learn_measure(
        &self,
        Parameters(req): Parameters<FuzzyLearnMeasureRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_learn_measure(&req.name, req.n, req.dataset, &req.dataset_id)
                .await,
        )
    }

    #[tool(
        name = "fuzzy_get_measure_version",
        description = "Version-aware fetch for a named Choquet measure. Mirrors `GET /fuzzy/measures/{name}?version=N`. Absent `version` returns the latest pointer (Phase 2 legacy behaviour); explicit version returns the historical slice or surfaces a `not found` error when the slice is missing."
    )]
    async fn fuzzy_get_measure_version(
        &self,
        Parameters(req): Parameters<FuzzyGetMeasureVersionRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(
            self.backend
                .fuzzy_get_measure_version(&req.name, req.version)
                .await,
        )
    }

    #[tool(
        name = "fuzzy_list_measure_versions",
        description = "Enumerate every persisted version of a named Choquet measure. Mirrors `GET /fuzzy/measures/{name}/versions`. Returns `{name, versions: [u32]}` sorted ascending — empty list when the name has no versioned slices (e.g. measure was created via `POST /fuzzy/measures` instead of `POST /fuzzy/measures/learn`)."
    )]
    async fn fuzzy_list_measure_versions(
        &self,
        Parameters(req): Parameters<FuzzyListMeasureVersionsRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.fuzzy_list_measure_versions(&req.name).await)
    }

    #[tool(
        name = "temporal_ordhorn_closure",
        description = "Run van Beek path-consistency closure on an Allen interval-algebra constraint network (Nebel-Bürckert 1995). Mirrors `POST /temporal/ordhorn/closure`. `network` is the externally-tagged JSON shape: `{\"n\":<usize>,\"constraints\":[{\"a\":<usize>,\"b\":<usize>,\"relations\":[\"Before\",\"Meets\",...]}]}`. Sound for any Allen network (empty cell ⇒ unsatisfiable); complete only for ORD-Horn networks. Returns `{closed_network, satisfiable}`. Cites: [nebel1995ordhorn]."
    )]
    async fn temporal_ordhorn_closure(
        &self,
        Parameters(req): Parameters<TemporalOrdHornClosureRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        wrap(self.backend.temporal_ordhorn_closure(req.network).await)
    }

    #[tool(
        name = "fuzzy_evaluate_rules",
        description = "Evaluate the rule set scoped to a narrative against one entity. Returns `{entity_id, fired_rules, defuzzified_output, defuzzification, firing_aggregate?}`. Omit `rule_ids` to evaluate every enabled rule; optional `firing_aggregator` collapses fired-rule strengths into a scalar. Mirrors `POST /fuzzy/rules/{nid}/evaluate`."
    )]
    async fn fuzzy_evaluate_rules(
        &self,
        Parameters(req): Parameters<FuzzyEvaluateRulesRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        // Deserialize the opaque firing_aggregator Value into the tagged-
        // union enum. Invalid shapes surface as InvalidInput via the wrap()
        // path — callers see a clean 400-equivalent error.
        let firing_aggregator = match req.firing_aggregator {
            Some(v) if !v.is_null() => {
                match serde_json::from_value::<crate::fuzzy::aggregation::AggregatorKind>(v) {
                    Ok(k) => Some(k),
                    Err(e) => {
                        return Ok(error_result(
                            crate::error::TensaError::InvalidInput(format!(
                                "invalid firing_aggregator shape: {e}"
                            )),
                        ))
                    }
                }
            }
            _ => None,
        };
        wrap(
            self.backend
                .fuzzy_evaluate_rules(
                    &req.narrative_id,
                    &req.entity_id,
                    req.rule_ids,
                    firing_aggregator,
                )
                .await,
        )
    }
}

#[tool_handler]
impl<B: McpBackend + Clone + 'static> ServerHandler for TensaMcp<B> {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "TENSA — Temporal Narrative Tensor Architecture. \
            A multi-fidelity narrative storage, reasoning, and inference engine. \
            Use the 'query' tool with TensaQL for graph pattern matching, \
            temporal reasoning, and vector search. Use 'infer' for causal, \
            game-theoretic, and motivation analysis. Use CRUD tools to \
            create/read entities and situations. Use 'ingest_text' to extract \
            narrative structure from raw text via LLM. \
            For novel-writing workflows: call 'get_writer_workspace' first to \
            orient, 'get_narrative_plan' / 'upsert_narrative_plan' to manage \
            intent, 'list_pinned_facts' / 'create_pinned_fact' for continuity \
            canon, 'check_continuity' to validate generated prose, \
            'run_workshop' for tiered critique, and 'list_narrative_revisions' / \
            'restore_narrative_revision' for git-like history.",
            )
            .with_server_info(Implementation::from_build_env())
    }
}
