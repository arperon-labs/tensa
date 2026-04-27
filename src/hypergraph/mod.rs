pub mod bitemporal;
pub mod causal;
pub mod chunk;
pub mod entity;
pub mod maturity;
pub mod participation;
pub mod situation;
pub mod source;
pub mod state;

use std::sync::Arc;

use crate::store::KVStore;

/// Key prefix constants for the KV encoding scheme.
pub mod keys {
    use uuid::Uuid;

    pub const ENTITY: &[u8] = b"e/";
    pub const SITUATION: &[u8] = b"s/";
    pub const PARTICIPATION: &[u8] = b"p/";
    pub const PARTICIPATION_REVERSE: &[u8] = b"ps/";
    pub const CAUSAL: &[u8] = b"c/";
    pub const CAUSAL_REVERSE: &[u8] = b"cr/";
    pub const STATE_VERSION: &[u8] = b"sv/";
    pub const INFERENCE_RESULT: &[u8] = b"ir/";
    pub const VALIDATION_LOG: &[u8] = b"v/";
    pub const META: &[u8] = b"meta/";
    pub const VALIDATION_QUEUE: &[u8] = b"vq/";
    pub const ENTITY_ALIAS: &[u8] = b"ea/";
    pub const CHUNK: &[u8] = b"ch/";
    pub const INFERENCE_JOB: &[u8] = b"ij/";
    pub const INFERENCE_QUEUE: &[u8] = b"ij/q/";
    pub const INFERENCE_TARGET: &[u8] = b"ij/t/";
    // Phase 3: Narrative/corpus/pattern
    pub const NARRATIVE: &[u8] = b"nr/";
    pub const CORPUS: &[u8] = b"cp/";
    pub const PATTERN: &[u8] = b"pm/";
    pub const PATTERN_MATCH: &[u8] = b"pm/m/";
    // Storywriting: user-defined narrative arcs (ua/{narrative_id}/{arc_uuid})
    pub const USER_ARC: &[u8] = b"ua/";
    // Narrative revisions (rv/r/{rev_uuid}) + narrative index (rv/n/{narrative_id}/{rev_uuid_v7})
    pub const REVISION: &[u8] = b"rv/r/";
    pub const REVISION_NARRATIVE_IDX: &[u8] = b"rv/n/";
    // Narrative plan — writer's canonical plot/style/length/setting doc (np/{narrative_id})
    pub const NARRATIVE_PLAN: &[u8] = b"np/";
    // Workshop reports — tiered analysis output (wr/r/{report_uuid}, wr/n/{narrative_id}/{report_uuid_v7})
    pub const WORKSHOP_REPORT: &[u8] = b"wr/r/";
    pub const WORKSHOP_REPORT_NARRATIVE_IDX: &[u8] = b"wr/n/";
    // Pinned facts — W4 continuity (pf/{narrative_id}/{fact_uuid})
    pub const PINNED_FACT: &[u8] = b"pf/";
    // Cost ledger — W5 AI operation cost records (cl/{narrative_id}/{entry_uuid_v7})
    pub const COST_LEDGER: &[u8] = b"cl/";

    // Projects (container above narratives)
    pub const PROJECT: &[u8] = b"pj/";
    pub const PROJECT_NARRATIVE_IDX: &[u8] = b"pn/";

    // Source intelligence (Phase 4)
    pub const SOURCE: &[u8] = b"src/";
    pub const SOURCE_ATTRIBUTION: &[u8] = b"sa/";
    pub const SOURCE_ATTRIBUTION_REV: &[u8] = b"sar/";
    pub const CONTENTION: &[u8] = b"ct/";
    pub const CONTENTION_REVERSE: &[u8] = b"ctr/";

    // Taxonomy (user-defined tags/genres)
    pub const TAXONOMY: &[u8] = b"tx/";

    // Analysis results
    pub const ANALYSIS_CENTRALITY: &[u8] = b"an/c/";
    pub const ANALYSIS_ENTROPY: &[u8] = b"an/e/";
    pub const ANALYSIS_MUTUAL_INFO: &[u8] = b"an/mi/";
    pub const ANALYSIS_BELIEF: &[u8] = b"an/b/";
    pub const ANALYSIS_EVIDENCE: &[u8] = b"an/ev/";
    pub const ANALYSIS_ARGUMENTATION: &[u8] = b"an/af/";
    pub const ANALYSIS_CONTAGION: &[u8] = b"an/sir/";
    pub const ANALYSIS_STYLE_PROFILE: &[u8] = b"an/ns/";
    pub const ANALYSIS_FINGERPRINT: &[u8] = b"an/nf/";
    /// PAN-learned / user-tuned weighted similarity config (v0.28).
    pub const ANALYSIS_STYLE_WEIGHTS: &[u8] = b"an/nw/";
    pub const ANALYSIS_TEMPORAL_ILP: &[u8] = b"an/ilp/";
    pub const ANALYSIS_MEAN_FIELD: &[u8] = b"an/mfg/";
    pub const ANALYSIS_PSL: &[u8] = b"an/psl/";
    pub const ANALYSIS_TRAJECTORY: &[u8] = b"an/traj/";
    pub const ANALYSIS_SIMULATION: &[u8] = b"an/sim/";

    // Disinfo extension (Sprint D1)
    /// BehavioralFingerprint per actor: `bf/{actor_uuid}`.
    pub const BEHAVIORAL_FINGERPRINT: &[u8] = b"bf/";
    /// DisinformationFingerprint per narrative: `df/{narrative_id}`.
    pub const DISINFO_FINGERPRINT: &[u8] = b"df/";

    // Disinfo extension (Sprint D2 — spread dynamics)
    /// Per-platform R₀ snapshots: `sp/r0/{narrative_id}`.
    pub const SPREAD_R0: &[u8] = b"sp/r0/";
    /// Cross-platform jump events: `sp/jump/{narrative_id}/{ts_be}`.
    pub const SPREAD_JUMP: &[u8] = b"sp/jump/";
    /// Spread velocity baselines: `vm/baseline/{platform}/{narrative_kind}`.
    pub const VELOCITY_BASELINE: &[u8] = b"vm/baseline/";
    /// Spread velocity alerts: `vm/alert/{ts_be}/{narrative_id}`.
    pub const VELOCITY_ALERT: &[u8] = b"vm/alert/";

    // Disinfo extension (Sprint D3 — CIB detection)
    /// CIB cluster detections per narrative: `cib/c/{narrative_id}/{cluster_id}`.
    pub const CIB_CLUSTER: &[u8] = b"cib/c/";
    /// CIB evidence records: `cib/e/{narrative_id}/{cluster_id}`.
    pub const CIB_EVIDENCE: &[u8] = b"cib/e/";
    /// Superspreader rankings per narrative: `cib/s/{narrative_id}`.
    pub const CIB_SUPERSPREADERS: &[u8] = b"cib/s/";

    // Disinfo extension (Sprint D4 — claims & fact-checks)
    /// Claims: `cl/{claim_uuid}`.
    pub const CLAIM: &[u8] = b"cl/";
    /// Claim normalization: `cl/n/{claim_uuid}`.
    pub const CLAIM_NORMALIZED: &[u8] = b"cl/n/";
    /// Fact-checks: `fc/{fact_check_uuid}`.
    pub const FACT_CHECK: &[u8] = b"fc/";
    /// Claim mutation events: `cl/mut/{original_uuid}/{mutated_uuid}`.
    pub const CLAIM_MUTATION: &[u8] = b"cl/mut/";
    /// Claim-to-fact-check matches: `cl/m/{claim_uuid}/{fact_check_uuid}`.
    pub const CLAIM_MATCH: &[u8] = b"cl/m/";

    // Disinfo extension (Sprint D5 — archetypes + fusion)
    /// Actor archetype distributions: `arch/{actor_uuid}`.
    pub const ARCHETYPE: &[u8] = b"arch/";
    /// Disinfo assessments: `da/{target_id}`.
    pub const DISINFO_ASSESSMENT: &[u8] = b"da/";

    // Disinfo extension (Sprint D6 — MCP orchestrator)
    /// MCP audit log: `mcp/audit/{timestamp_be}`.
    pub const MCP_AUDIT: &[u8] = b"mcp/audit/";

    // Disinfo extension (Sprint D7 — integration gaps)
    /// Monitor subscriptions: `mon/{subscription_uuid}`.
    pub const MONITOR_SUBSCRIPTION: &[u8] = b"mon/";
    /// Monitor alerts: `mon/alert/{alert_uuid}`.
    pub const MONITOR_ALERT: &[u8] = b"mon/alert/";
    /// Content factory detections: `cib/f/{narrative_id}/{factory_id}`.
    pub const CIB_FACTORY: &[u8] = b"cib/f/";

    // Disinfo extension (Sprint D8 — scheduler)
    /// Scheduled tasks: `sched/{task_uuid}`.
    pub const SCHEDULED_TASK: &[u8] = b"sched/";
    /// Task execution history: `sched/hist/{task_uuid}/{timestamp}`.
    pub const TASK_HISTORY: &[u8] = b"sched/hist/";
    /// Task execution locks: `sched/lock/{task_uuid}`.
    pub const TASK_LOCK: &[u8] = b"sched/lock/";

    // Disinfo extension (Sprint D8.2–D8.4 — scheduler subsystems)
    /// CIB scan deltas: `cib/delta/{timestamp}`.
    pub const CIB_DELTA: &[u8] = b"cib/delta/";
    /// Source discovery candidates: `disc/{candidate_uuid}`.
    pub const DISCOVERY: &[u8] = b"disc/";
    /// Approved discovery sources: `disc/approved/{candidate_uuid}`.
    pub const DISCOVERY_APPROVED: &[u8] = b"disc/approved/";
    /// Fact-check sync results: `fc/sync/{timestamp}`.
    pub const FACT_CHECK_SYNC: &[u8] = b"fc/sync/";

    // Alert system
    pub const ALERT_RULE: &[u8] = b"alert/r/";
    pub const ALERT_EVENT: &[u8] = b"alert/e/";

    // Secondary indexes
    pub const ENTITY_TYPE_IDX: &[u8] = b"et/";
    pub const ENTITY_NARRATIVE_IDX: &[u8] = b"en/";
    pub const SITUATION_LEVEL_IDX: &[u8] = b"sl/";
    pub const SITUATION_NARRATIVE_IDX: &[u8] = b"sn/";

    /// Build a key by concatenating prefix + uuid bytes
    pub fn entity_key(id: &Uuid) -> Vec<u8> {
        let mut key = ENTITY.to_vec();
        key.extend_from_slice(id.as_bytes());
        key
    }

    pub fn situation_key(id: &Uuid) -> Vec<u8> {
        let mut key = SITUATION.to_vec();
        key.extend_from_slice(id.as_bytes());
        key
    }

    pub fn participation_key(entity_id: &Uuid, situation_id: &Uuid) -> Vec<u8> {
        let mut key = PARTICIPATION.to_vec();
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(situation_id.as_bytes());
        key
    }

    pub fn participation_reverse_key(situation_id: &Uuid, entity_id: &Uuid) -> Vec<u8> {
        let mut key = PARTICIPATION_REVERSE.to_vec();
        key.extend_from_slice(situation_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(entity_id.as_bytes());
        key
    }

    pub fn participation_prefix_for_entity(entity_id: &Uuid) -> Vec<u8> {
        let mut key = PARTICIPATION.to_vec();
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key
    }

    pub fn participation_prefix_for_situation(situation_id: &Uuid) -> Vec<u8> {
        let mut key = PARTICIPATION_REVERSE.to_vec();
        key.extend_from_slice(situation_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Key for a specific participation with sequence number.
    pub fn participation_seq_key(entity_id: &Uuid, situation_id: &Uuid, seq: u16) -> Vec<u8> {
        let mut key = PARTICIPATION.to_vec();
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(situation_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(&seq.to_be_bytes());
        key
    }

    /// Reverse key for a specific participation with sequence number.
    pub fn participation_reverse_seq_key(
        situation_id: &Uuid,
        entity_id: &Uuid,
        seq: u16,
    ) -> Vec<u8> {
        let mut key = PARTICIPATION_REVERSE.to_vec();
        key.extend_from_slice(situation_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(&seq.to_be_bytes());
        key
    }

    /// Prefix for all participations of an entity in a specific situation.
    pub fn participation_pair_prefix(entity_id: &Uuid, situation_id: &Uuid) -> Vec<u8> {
        let mut key = PARTICIPATION.to_vec();
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(situation_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Reverse prefix for all participations of an entity in a specific situation.
    pub fn participation_reverse_pair_prefix(situation_id: &Uuid, entity_id: &Uuid) -> Vec<u8> {
        let mut key = PARTICIPATION_REVERSE.to_vec();
        key.extend_from_slice(situation_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key
    }

    pub fn state_version_key(entity_id: &Uuid, situation_id: &Uuid) -> Vec<u8> {
        let mut key = STATE_VERSION.to_vec();
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(situation_id.as_bytes());
        key
    }

    pub fn state_version_prefix(entity_id: &Uuid) -> Vec<u8> {
        let mut key = STATE_VERSION.to_vec();
        key.extend_from_slice(entity_id.as_bytes());
        key.push(b'/');
        key
    }

    pub fn causal_key(from_id: &Uuid, to_id: &Uuid) -> Vec<u8> {
        let mut key = CAUSAL.to_vec();
        key.extend_from_slice(from_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(to_id.as_bytes());
        key
    }

    pub fn causal_prefix(from_id: &Uuid) -> Vec<u8> {
        let mut key = CAUSAL.to_vec();
        key.extend_from_slice(from_id.as_bytes());
        key.push(b'/');
        key
    }

    pub fn causal_reverse_key(to_id: &Uuid, from_id: &Uuid) -> Vec<u8> {
        let mut key = CAUSAL_REVERSE.to_vec();
        key.extend_from_slice(to_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(from_id.as_bytes());
        key
    }

    pub fn causal_reverse_prefix(to_id: &Uuid) -> Vec<u8> {
        let mut key = CAUSAL_REVERSE.to_vec();
        key.extend_from_slice(to_id.as_bytes());
        key.push(b'/');
        key
    }

    pub fn validation_log_key(
        timestamp: &chrono::DateTime<chrono::Utc>,
        target_id: &Uuid,
    ) -> Vec<u8> {
        let mut key = VALIDATION_LOG.to_vec();
        key.extend_from_slice(&timestamp.timestamp_millis().to_be_bytes());
        key.push(b'/');
        key.extend_from_slice(target_id.as_bytes());
        key
    }

    pub fn validation_log_prefix() -> Vec<u8> {
        VALIDATION_LOG.to_vec()
    }

    /// Build a key for a narrative record: `nr/{narrative_id}`.
    pub fn narrative_key(narrative_id: &str) -> Vec<u8> {
        let mut key = NARRATIVE.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key
    }

    /// Build a key prefix for listing all narratives.
    pub fn narrative_prefix() -> Vec<u8> {
        NARRATIVE.to_vec()
    }

    /// Build a key for a project record: `pj/{project_id}`.
    pub fn project_key(project_id: &str) -> Vec<u8> {
        let mut key = PROJECT.to_vec();
        key.extend_from_slice(project_id.as_bytes());
        key
    }

    /// Prefix for listing all projects.
    pub fn project_prefix() -> Vec<u8> {
        PROJECT.to_vec()
    }

    /// Build project-narrative index key: `pn/{project_id}/{narrative_id}`.
    pub fn project_narrative_index_key(project_id: &str, narrative_id: &str) -> Vec<u8> {
        let mut key = PROJECT_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(project_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(narrative_id.as_bytes());
        key
    }

    /// Prefix for scanning all narratives in a project.
    pub fn project_narrative_index_prefix(project_id: &str) -> Vec<u8> {
        let mut key = PROJECT_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(project_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a key for a corpus split: `cp/split/{split_name}`.
    pub fn corpus_split_key(split_name: &str) -> Vec<u8> {
        let mut key = CORPUS.to_vec();
        key.extend_from_slice(b"split/");
        key.extend_from_slice(split_name.as_bytes());
        key
    }

    /// Build a key for a narrative pattern: `pm/{pattern_id}`.
    pub fn pattern_key(pattern_id: &str) -> Vec<u8> {
        let mut key = PATTERN.to_vec();
        key.extend_from_slice(pattern_id.as_bytes());
        key
    }

    /// Build a key for a pattern match: `pm/m/{narrative_id}/{pattern_id}`.
    pub fn pattern_match_key(narrative_id: &str, pattern_id: &str) -> Vec<u8> {
        let mut key = PATTERN_MATCH.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(pattern_id.as_bytes());
        key
    }

    /// Build a key for a user-defined narrative arc: `ua/{narrative_id}/{arc_uuid_bytes}`.
    pub fn user_arc_key(narrative_id: &str, arc_id: &Uuid) -> Vec<u8> {
        let mut key = USER_ARC.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(arc_id.as_bytes());
        key
    }

    /// Prefix for scanning all user arcs of a narrative.
    pub fn user_arc_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = USER_ARC.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a key for a narrative revision record: `rv/r/{rev_uuid_bytes}`.
    pub fn revision_key(rev_id: &Uuid) -> Vec<u8> {
        let mut key = REVISION.to_vec();
        key.extend_from_slice(rev_id.as_bytes());
        key
    }

    /// Narrative→revision index entry: `rv/n/{narrative_id}/{rev_uuid_bytes}`.
    /// Rev IDs are v7 UUIDs, so prefix-scan yields chronological order.
    pub fn revision_narrative_index_key(narrative_id: &str, rev_id: &Uuid) -> Vec<u8> {
        let mut key = REVISION_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(rev_id.as_bytes());
        key
    }

    /// Prefix for scanning all revisions of a narrative.
    pub fn revision_narrative_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = REVISION_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a key for a narrative plan: `np/{narrative_id}`. One plan per narrative.
    pub fn narrative_plan_key(narrative_id: &str) -> Vec<u8> {
        let mut key = NARRATIVE_PLAN.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key
    }

    /// Build a key for a workshop report: `wr/r/{report_uuid_bytes}`.
    pub fn workshop_report_key(report_id: &Uuid) -> Vec<u8> {
        let mut key = WORKSHOP_REPORT.to_vec();
        key.extend_from_slice(report_id.as_bytes());
        key
    }

    /// Narrative→report index entry: `wr/n/{narrative_id}/{report_uuid_bytes}`.
    /// UUID v7 keys sort chronologically under prefix scan.
    pub fn workshop_report_narrative_index_key(narrative_id: &str, report_id: &Uuid) -> Vec<u8> {
        let mut key = WORKSHOP_REPORT_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(report_id.as_bytes());
        key
    }

    /// Prefix for scanning all workshop reports of a narrative.
    pub fn workshop_report_narrative_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = WORKSHOP_REPORT_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a key for a pinned fact: `pf/{narrative_id}/{fact_uuid_bytes}`.
    pub fn pinned_fact_key(narrative_id: &str, fact_id: &Uuid) -> Vec<u8> {
        let mut key = PINNED_FACT.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(fact_id.as_bytes());
        key
    }

    /// Prefix for scanning all pinned facts of a narrative.
    pub fn pinned_fact_narrative_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = PINNED_FACT.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a key for a cost ledger entry: `cl/{narrative_id}/{entry_uuid_bytes}`.
    /// Entry UUIDs are v7 so prefix-scan yields chronological order.
    pub fn cost_ledger_key(narrative_id: &str, entry_id: &Uuid) -> Vec<u8> {
        let mut key = COST_LEDGER.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(entry_id.as_bytes());
        key
    }

    /// Prefix for scanning all cost ledger entries for a narrative.
    pub fn cost_ledger_narrative_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = COST_LEDGER.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    // ─── Source Intelligence Key Helpers ────────────────────────

    /// Build a key for a source record: `src/{uuid}`.
    pub fn source_key(id: &Uuid) -> Vec<u8> {
        let mut key = SOURCE.to_vec();
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Prefix for listing all sources.
    pub fn source_prefix() -> Vec<u8> {
        SOURCE.to_vec()
    }

    /// Build a source attribution key: `sa/{source_uuid}/{target_uuid}`.
    pub fn source_attribution_key(source_id: &Uuid, target_id: &Uuid) -> Vec<u8> {
        let mut key = SOURCE_ATTRIBUTION.to_vec();
        key.extend_from_slice(source_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(target_id.as_bytes());
        key
    }

    /// Prefix for all attributions from a given source.
    pub fn source_attribution_prefix(source_id: &Uuid) -> Vec<u8> {
        let mut key = SOURCE_ATTRIBUTION.to_vec();
        key.extend_from_slice(source_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a reverse attribution key: `sar/{target_uuid}/{source_uuid}`.
    pub fn source_attribution_reverse_key(target_id: &Uuid, source_id: &Uuid) -> Vec<u8> {
        let mut key = SOURCE_ATTRIBUTION_REV.to_vec();
        key.extend_from_slice(target_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(source_id.as_bytes());
        key
    }

    /// Prefix for all sources attributing a given target.
    pub fn source_attribution_reverse_prefix(target_id: &Uuid) -> Vec<u8> {
        let mut key = SOURCE_ATTRIBUTION_REV.to_vec();
        key.extend_from_slice(target_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a contention key: `ct/{situation_a}/{situation_b}`.
    pub fn contention_key(a: &Uuid, b: &Uuid) -> Vec<u8> {
        let mut key = CONTENTION.to_vec();
        key.extend_from_slice(a.as_bytes());
        key.push(b'/');
        key.extend_from_slice(b.as_bytes());
        key
    }

    /// Prefix for all contentions involving situation_a.
    pub fn contention_prefix(a: &Uuid) -> Vec<u8> {
        let mut key = CONTENTION.to_vec();
        key.extend_from_slice(a.as_bytes());
        key.push(b'/');
        key
    }

    /// Build a reverse contention key: `ctr/{situation_b}/{situation_a}`.
    pub fn contention_reverse_key(b: &Uuid, a: &Uuid) -> Vec<u8> {
        let mut key = CONTENTION_REVERSE.to_vec();
        key.extend_from_slice(b.as_bytes());
        key.push(b'/');
        key.extend_from_slice(a.as_bytes());
        key
    }

    /// Prefix for reverse contention lookups.
    pub fn contention_reverse_prefix(b: &Uuid) -> Vec<u8> {
        let mut key = CONTENTION_REVERSE.to_vec();
        key.extend_from_slice(b.as_bytes());
        key.push(b'/');
        key
    }

    // ─── Taxonomy Keys ─────────────────────────────────────────

    /// Build a taxonomy entry key: `tx/{category}/{value}`.
    pub fn taxonomy_key(category: &str, value: &str) -> Vec<u8> {
        let mut key = TAXONOMY.to_vec();
        key.extend_from_slice(category.as_bytes());
        key.push(b'/');
        key.extend_from_slice(value.as_bytes());
        key
    }

    /// Prefix for all taxonomy entries in a category: `tx/{category}/`.
    pub fn taxonomy_prefix(category: &str) -> Vec<u8> {
        let mut key = TAXONOMY.to_vec();
        key.extend_from_slice(category.as_bytes());
        key.push(b'/');
        key
    }

    // ─── Chunk Storage Keys ────────────────────────────────────

    // Sub-prefixes under ch/:
    pub const CHUNK_RECORD: &[u8] = b"ch/r/";
    pub const CHUNK_JOB_IDX: &[u8] = b"ch/j/";
    pub const CHUNK_NARRATIVE_IDX: &[u8] = b"ch/n/";
    pub const CHUNK_HASH_IDX: &[u8] = b"ch/h/";

    /// Primary chunk record key: `ch/r/{uuid_bytes}`.
    pub fn chunk_record_key(id: &Uuid) -> Vec<u8> {
        let mut key = CHUNK_RECORD.to_vec();
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Job index key: `ch/j/{job_id}/{chunk_index_be32}`.
    pub fn chunk_job_key(job_id: &str, chunk_index: u32) -> Vec<u8> {
        let mut key = CHUNK_JOB_IDX.to_vec();
        key.extend_from_slice(job_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(&chunk_index.to_be_bytes());
        key
    }

    /// Prefix for all chunks in a job: `ch/j/{job_id}/`.
    pub fn chunk_job_prefix(job_id: &str) -> Vec<u8> {
        let mut key = CHUNK_JOB_IDX.to_vec();
        key.extend_from_slice(job_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Narrative index key: `ch/n/{narrative_id}/{chunk_index_be32}`.
    pub fn chunk_narrative_key(narrative_id: &str, chunk_index: u32) -> Vec<u8> {
        let mut key = CHUNK_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(&chunk_index.to_be_bytes());
        key
    }

    /// Prefix for all chunks in a narrative: `ch/n/{narrative_id}/`.
    pub fn chunk_narrative_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = CHUNK_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Hash dedup key: `ch/h/{narrative_id}/{hash}`.
    pub fn chunk_hash_dedup_key(narrative_id: &str, hash: &str) -> Vec<u8> {
        let mut key = CHUNK_HASH_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(hash.as_bytes());
        key
    }

    // Deprecated: old format `ch/{narrative_id}/{hash}` — kept for backward compat.

    /// Build a chunk hash key: `ch/{narrative_id}/{hash}`.
    #[deprecated(note = "Use chunk_hash_dedup_key instead")]
    pub fn chunk_hash_key(narrative_id: &str, hash: &str) -> Vec<u8> {
        let mut key = CHUNK.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(hash.as_bytes());
        key
    }

    /// Prefix for all chunk hashes in a narrative: `ch/{narrative_id}/`.
    #[deprecated(note = "Use chunk_narrative_prefix instead")]
    pub fn chunk_hash_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = CHUNK.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    // ─── Secondary Index Helpers ─────────────────────────────────

    /// Build entity-type index key: `et/{type_str}/{uuid}`.
    pub fn entity_type_index_key(entity_type: &str, id: &Uuid) -> Vec<u8> {
        let mut key = ENTITY_TYPE_IDX.to_vec();
        key.extend_from_slice(entity_type.as_bytes());
        key.push(b'/');
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Prefix for scanning all entities of a given type.
    pub fn entity_type_index_prefix(entity_type: &str) -> Vec<u8> {
        let mut key = ENTITY_TYPE_IDX.to_vec();
        key.extend_from_slice(entity_type.as_bytes());
        key.push(b'/');
        key
    }

    /// Build entity-narrative index key: `en/{narrative_id}/{uuid}`.
    pub fn entity_narrative_index_key(narrative_id: &str, id: &Uuid) -> Vec<u8> {
        let mut key = ENTITY_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Prefix for scanning all entities in a narrative.
    pub fn entity_narrative_index_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = ENTITY_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    /// Build situation-level index key: `sl/{level_str}/{uuid}`.
    pub fn situation_level_index_key(level: &str, id: &Uuid) -> Vec<u8> {
        let mut key = SITUATION_LEVEL_IDX.to_vec();
        key.extend_from_slice(level.as_bytes());
        key.push(b'/');
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Prefix for scanning all situations at a given level.
    pub fn situation_level_index_prefix(level: &str) -> Vec<u8> {
        let mut key = SITUATION_LEVEL_IDX.to_vec();
        key.extend_from_slice(level.as_bytes());
        key.push(b'/');
        key
    }

    /// Build situation-narrative index key: `sn/{narrative_id}/{uuid}`.
    pub fn situation_narrative_index_key(narrative_id: &str, id: &Uuid) -> Vec<u8> {
        let mut key = SITUATION_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Prefix for scanning all situations in a narrative.
    pub fn situation_narrative_index_prefix(narrative_id: &str) -> Vec<u8> {
        let mut key = SITUATION_NARRATIVE_IDX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key
    }

    // Disinfo fingerprint keys are built via `crate::analysis::analysis_key`
    // — see `src/disinfo/fingerprints.rs` (`behavioral_key` / `disinfo_key`).
}

/// The Hypergraph engine. Central API for all entity, situation,
/// participation, state, and causal operations.
///
/// Generic over the KVStore implementation — works with MemoryStore
/// in tests and RocksDBStore in production.
pub struct Hypergraph {
    store: Arc<dyn KVStore>,
}

impl Hypergraph {
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self { store }
    }

    /// Get a reference to the underlying store (for testing)
    pub fn store(&self) -> &dyn KVStore {
        self.store.as_ref()
    }

    /// Get a clone of the underlying store Arc (for constructing other KV-backed services).
    pub fn store_arc(&self) -> Arc<dyn KVStore> {
        Arc::clone(&self.store)
    }

    /// Extract UUIDs from a secondary index prefix scan.
    /// Index keys end with 16 UUID bytes; this helper collects them.
    pub(crate) fn ids_from_index(&self, prefix: &[u8]) -> crate::error::Result<Vec<uuid::Uuid>> {
        let pairs = self.store.prefix_scan(prefix)?;
        let mut ids = Vec::with_capacity(pairs.len());
        for (key, _) in pairs {
            if key.len() >= 16 {
                if let Ok(id) = uuid::Uuid::from_slice(&key[key.len() - 16..]) {
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }
}
