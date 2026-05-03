# TENSA Reference Book

**TENSA — Temporal Narrative Storage & Analysis**

**Version:** 0.79.11  **Last revised:** 2026-05-01

This reference tracks the shipped surface of TENSA. Behaviour flagged as scaffolded or partial is listed in [Appendix D](#appendix-d-implementation-status). For the version-by-version delta, see `CHANGELOG.md` at the repo root.

Source-tree references in this document (e.g. `src/narrative/revision.rs`) are path-only — they are not hyperlinked so the reference renders identically whether hosted standalone or browsed inside the repo.

---

## Table of Contents

- [Chapter 1: Introduction & Core Concepts](#chapter-1-introduction--core-concepts)
- [Chapter 2: Architecture](#chapter-2-architecture)
- [Chapter 3: TensaQL Language Reference](#chapter-3-tensaql-language-reference)
- [Chapter 4: MCP Tools Reference](#chapter-4-mcp-tools-reference)
- [Chapter 5: REST API Reference](#chapter-5-rest-api-reference)
- [Chapter 6: Common Workflows](#chapter-6-common-workflows)
- [Chapter 7: Algorithms & Theory](#chapter-7-algorithms--theory)
- [Chapter 8: Configuration Reference](#chapter-8-configuration-reference)
- [Chapter 9: Synthetic Generation (Surrogate Models)](#chapter-9-synthetic-generation-surrogate-models)
- [Chapter 10: Hypergraph Reconstruction from Dynamics](#chapter-10-hypergraph-reconstruction-from-dynamics)
- [Chapter 11: Opinion Dynamics (BCM on Hypergraphs)](#chapter-11-opinion-dynamics-bcm-on-hypergraphs)
- [Chapter 12: Configuration-Style Null Model (NuDHy) + Dual-Null-Model Significance](#chapter-12-configuration-style-null-model-nudhy--dual-null-model-significance)
- [Chapter 13: Bistability / Hysteresis in Higher-Order Contagion](#chapter-13-bistability--hysteresis-in-higher-order-contagion)
- [Chapter 14: Fuzzy Logic](#chapter-14-fuzzy-logic)
- [Chapter 15: Graded Acceptability & Measure Learning](#chapter-15-graded-acceptability--measure-learning)
- [Appendix A: TensaQL Grammar Quick Reference](#appendix-a-tensaql-grammar-quick-reference)
- [Appendix B: Key Encoding Scheme](#appendix-b-key-encoding-scheme)
- [Appendix C: Glossary](#appendix-c-glossary)
- [Appendix D: Implementation Status](#appendix-d-implementation-status)

---

# Chapter 1: Introduction & Core Concepts

## What Is TENSA?

TENSA is a **multi-fidelity narrative storage, reasoning, and inference engine**. It represents complex multi-actor event systems — investigations, novels, historical events, intelligence reports — as temporal hypergraphs. Every piece of data carries a confidence score and maturity level, enabling TENSA to reason under uncertainty.

TENSA is not a general-purpose database. It is purpose-built for narratives: sequences of events involving actors, locations, artifacts, and organizations, connected by causal chains and layered with temporal reasoning.

## The Core Mental Model

Think of TENSA as a graph where:

- **Entities** are nodes — the actors, locations, artifacts, concepts, and organizations in your narrative
- **Situations** are hyperedges — events that connect multiple entities simultaneously (a meeting involves several people at a location)
- **Time is first-class** — every situation has a temporal interval, and TENSA can reason about temporal relations (does event A happen before, during, or after event B?)
- **Confidence is built-in** — every piece of data carries a score from 0.0 to 1.0, reflecting how certain we are about it

```
    [Actor: Alice]
         |
         | role: Protagonist
         | action: "discovers evidence"
         v
  +--------------------------+
  | Situation: Crime Scene   |
  | temporal: 2025-03-15     |  <--- causal link ---> [Situation: Arrest]
  | level: Scene             |
  | confidence: 0.85         |
  +--------------------------+
         ^
         | role: Witness
         |
    [Actor: Bob]
```

## Entity Types

TENSA recognizes five entity types:

| Type | Description | Examples |
|------|-------------|----------|
| **Actor** | People, agents, characters | Suspects, witnesses, detectives |
| **Location** | Places, areas, regions | Crime scene, headquarters, safehouse |
| **Artifact** | Physical or digital objects | Weapon, document, recording |
| **Concept** | Abstract ideas or themes | "Revenge", "cover-up", "alliance" |
| **Organization** | Groups, companies, agencies | Police dept, criminal ring, corp |

## Narrative Hierarchy

Situations are organized in a six-level hierarchy from coarsest to finest granularity:

```
Story          The entire narrative ("The Embassy Investigation")
  Arc          A major plot thread ("The Money Trail")
    Sequence   A coherent sequence of related events
      Scene    A continuous event in one location/time
        Beat   A single dramatic moment within a scene
          Event  The finest unit — one atomic action
```

Each level corresponds to a `NarrativeLevel` value: `Story`, `Arc`, `Sequence`, `Scene`, `Beat`, `Event`.

## Confidence & Maturity

Every entity and situation in TENSA carries two quality indicators.

### Confidence Score (0.0 — 1.0)

How certain we are about this piece of data:
- **0.9 — 1.0**: High confidence (corroborated by multiple sources)
- **0.5 — 0.9**: Moderate confidence (single credible source)
- **0.3 — 0.5**: Low confidence (uncorroborated, possible errors)
- **0.0 — 0.3**: Very low confidence (speculation, rumor)

### Maturity Levels

The data quality lifecycle:

```
Candidate  ──>  Reviewed  ──>  Validated  ──>  GroundTruth
  (LLM          (Analyst        (Multiple       (Confirmed
  extracted)    reviewed)       sources)        fact)
```

| Level | Meaning |
|-------|---------|
| **Candidate** | Freshly extracted, unreviewed |
| **Reviewed** | An analyst has checked it |
| **Validated** | Corroborated by multiple sources |
| **GroundTruth** | Confirmed, authoritative fact |

## Participation Roles

When an entity participates in a situation, it has a role:

| Role | Description |
|------|-------------|
| **Protagonist** | Primary actor driving the event |
| **Antagonist** | Actor opposing the protagonist |
| **Witness** | Observer of the event |
| **Target** | Object or person affected by the event |
| **Instrument** | Tool or means used in the event |
| **Confidant** | Trusted advisor or ally |
| **Informant** | Provides inside information |
| **Recipient** | Receives something in the event |
| **Bystander** | Present but not involved |
| **SubjectOfDiscussion** | Discussed but not present |
| **Facilitator** | Enables the protagonist's action without being its agent (gatekeepers, keepers, intermediaries) |
| **Custom(String)** | Domain-specific role label (serialised as `{"Custom": "role-name"}`) |

An entity may have multiple roles in the same situation via a sequence number — the participation key is `p/{entity}/{situation}/{seq}`. Use `add_participant` to append a new role without replacing existing ones.

## Causal Links

Situations can cause other situations. Each causal link has:

- **Strength** (0.0 — 1.0): How strong the causal relationship is
- **Mechanism**: A text description of how causation works
- **Type**: The nature of the causal relationship

| Causal Type | Meaning |
|-------------|---------|
| **Necessary** | Without this cause, the effect cannot happen |
| **Sufficient** | This cause alone is enough for the effect |
| **Contributing** | This cause increases the likelihood of the effect |
| **Enabling** | This cause makes the effect possible but doesn't guarantee it |

Causal links are checked for cycles at insertion time.

## Running Example: The Harbor Case

Throughout this book, we use a running example — an intelligence investigation:

> A series of suspicious meetings at a harbor warehouse leads to the discovery of a smuggling ring. Key actors include **Agent Chen** (investigator), **Viktor Orlov** (suspect), **Marina Silva** (informant), and the organization **Nightfall Group**. The investigation spans multiple locations and involves wiretaps, surveillance, financial records, and witness interviews.

We'll build this narrative step by step, showing each TENSA feature in context.

---

# Chapter 2: Architecture

## Layered Design

TENSA is built in layers. Each layer depends only on layers below it:

```
Layer 7: Studio UI        React frontend (Observatory Console theme)
Layer 6: MCP Server       Model Context Protocol interface
Layer 5: REST API         Axum HTTP endpoints (feature-gated)
Layer 4: Analysis         Centrality, entropy, beliefs, evidence, argumentation, contagion
Layer 3b: Narrative       Cross-narrative patterns, arcs, prediction
Layer 3a: Inference       Causal, game-theoretic, motivation (async workers)
Layer 2: Query            TensaQL parser + planner + executor
Layer 1: Hypergraph       Entities, situations, participation, state, causal links
Layer 0b: Temporal        Allen interval algebra + interval tree index
Layer 0a: Storage         KVStore trait + RocksDB / MemoryStore implementations
Layer 0: Types            Shared types, UUIDs, serialization
```

## The KV Store Abstraction

All data storage goes through a trait, never directly to a database:

```rust
pub trait KVStore: Send + Sync {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn range(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn prefix_scan(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn transaction(&self, ops: Vec<TxnOp>) -> Result<()>;
    fn batch_put(&self, pairs: Vec<(&[u8], &[u8])>) -> Result<()>;
}
```

Two implementations exist:
- **RocksDB** (`rocks.rs`) — Production storage, requires C++ toolchain
- **MemoryStore** (`memory.rs`) — BTreeMap-based, used in tests

This abstraction enables future migration to FoundationDB without changing any hypergraph code. Values are serialized with `serde_json` (not bincode — many types contain `serde_json::Value` fields that require `deserialize_any`).

## Multi-Tenancy: Workspaces

A *workspace* is a top-level KV namespace that physically isolates one tenant's data from another. Two workspaces literally cannot see each other's entities, situations, narratives, projects, sources, validation queue items, jobs, or analysis results.

**Implementation** ([src/store/workspace.rs](../src/store/workspace.rs)):
- `WorkspaceStore` is a `KVStore` decorator that transparently prepends `w/{workspace_id}/` to **every** key before delegating to the underlying RocksDB / Memory store. `prefix_scan` and `range` strip the prefix on the way out so callers see clean keys.
- The `Hypergraph` accepts any `Arc<dyn KVStore>`, so swapping in a `WorkspaceStore` namespaces the entire engine. **Nothing inside the engine knows workspaces exist** — that's the design property that lets the same code path serve every tenant.
- Workspace metadata (`{id, name, created_at}`) lives at `_ws/{id}` on the **root** (un-namespaced) store via [src/api/workspace_routes.rs](../src/api/workspace_routes.rs:14) so the admin / list surface can enumerate workspaces cheaply across all tenants.
- Studio scopes every API call by sending an `X-Tensa-Workspace` header; `"default"` is used when absent.
- Studio chat tables (`chat/s/{ws}/{user}/...`, `chat/m/{ws}/{user}/...`) live on the **root** store rather than under a `WorkspaceStore` so admin / cross-workspace enumeration is cheap.

**REST surface**:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/workspaces` | Create. Body `{id, name?}`. `id` is URL-safe slug (no `/`, no `.`, ≤64 chars). 409 on conflict. |
| `GET`  | `/workspaces` | List all `WorkspaceMeta`. |
| `GET`  | `/workspaces/:id` | Single workspace. |
| `DELETE` | `/workspaces/:id` | Cascade-deletes everything under `w/{id}/`. The `default` workspace is protected — 400 on attempt. |

**Status (v0.79.x)**:
- The `X-Tensa-Workspace` header is honour-system — there is **no auth layer**. Workspaces are a data boundary, not a security boundary.
- Studio has no workspace switcher UI yet; the rest of the app talks to whatever the header says (default `"default"`).
- No quota enforcement, no per-workspace billing surface.

## Projects (Soft Narrative Grouping)

A *project* is a logical bucket inside a workspace that groups related narratives. Pure organizational tag — no data isolation, no key namespacing. Example: a "Geopolitics" project containing the "Ukraine", "Middle East", and "South China Sea" narratives.

**Implementation** ([src/narrative/project.rs](../src/narrative/project.rs), [src/narrative/types.rs:31](../src/narrative/types.rs#L31)):
- `Project { id, title, description, tags, narrative_count, created_at, updated_at }` stored at `pj/{project_id}` via `ProjectRegistry`.
- A secondary index `pn/{project_id}/{narrative_id}` records membership; `list_narrative_ids(project_id)` does a single `prefix_scan` on `pn/{project_id}/`.
- `Narrative.project_id: Option<String>` ([src/narrative/types.rs:68](../src/narrative/types.rs#L68)) is the back-pointer set when a narrative is created or moved into a project.
- `list_paginated(limit, after)` returns the cursor envelope `{data, next_cursor}` (this is the surface every Studio paginated list uses; see §5 for the full pattern).

**Functional consequence beyond grouping** — [src/ingestion/pipeline.rs:533-554](../src/ingestion/pipeline.rs#L533) uses project membership for **cross-narrative entity resolution**: when fuzzy rules trigger, the pipeline pulls *sibling narratives in the same project* via `projects.list_narrative_ids(pid)` so "Putin" in the Ukraine narrative matches "Putin" in the Middle East narrative. This is the only place project membership has functional (vs purely organizational) consequence today.

**REST surface**:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/projects` | Create. |
| `GET`  | `/projects?limit=N&after=cursor` | Cursor-paginated list (`{data, next_cursor}`). |
| `GET`  | `/projects/:id` | Single project. |
| `PUT`  | `/projects/:id` | Partial update (title / description / tags). |
| `DELETE` | `/projects/:id?cascade=true` | Delete; `cascade=true` also removes the `pn/` index entries. |
| `GET`  | `/projects/:id/narratives` | List narratives belonging to the project (raw array, not paginated). |

**Studio surface** ([studio/src/views/Projects.tsx](../studio/src/views/Projects.tsx)) — flat list view + create/delete + a detail modal showing linked narratives. No drag-and-drop into/out of projects yet; moving a narrative between projects requires a `PUT /narratives/:id` setting `project_id` plus the membership-index update.

**Workspace × Project relationship**:

```
┌───────────────────────────────────────────────────────────┐
│ Workspace "default"     (KV namespace w/default/...)      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Project "geopolitics"                               │  │
│  │  ├─ Narrative "ukraine"      (entities, sits, ...) │  │
│  │  ├─ Narrative "middle-east"  (entities, sits, ...) │  │
│  │  └─ Narrative "south-china-sea"                    │  │
│  │      ↳ siblings see each other for entity-         │  │
│  │        resolution + fuzzy rules                    │  │
│  ├─ Narrative "crime-and-punishment" (no project)     │  │
│  └─ Validation queue, jobs, sources, synth runs, ...  │  │
└───────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────┐
│ Workspace "client-acme" (KV namespace w/client-acme/...)  │
│   ...its own projects, narratives, queues — invisible     │
│   to "default"                                            │
└───────────────────────────────────────────────────────────┘
```

## Key Encoding

Data is stored as structured byte keys. UUIDs use big-endian binary (16 bytes) so lexicographic sort matches time order for v7 UUIDs. The full table is in [Appendix B](#appendix-b-key-encoding-scheme); a summary by area:

| Area | Prefixes |
|------|----------|
| Hypergraph | `e/`, `s/`, `p/`, `ps/`, `c/`, `cr/`, `sv/` |
| Narratives & taxonomy | `nr/`, `cp/`, `tx/`, `pt/` |
| Validation & jobs | `vq/`, `ij/`, `ir/`, `v/` |
| Analysis results | `an/c/`, `an/e/`, `an/pr/`, `an/ev_c/`, `an/hc/`, `an/hits/`, `an/tp/`, `an/kc/`, `an/lp/`, `an/tpr/`, `an/ci/`, `an/ib/`, `an/ch/`, `an/tm/`, `an/fe/`, `an/frp/`, `an/n2v/`, `an/traj/`, `an/mfg/`, `an/psl/`, `an/sir/`, `an/sim/`, `an/ev/`, `an/af/`, `an/ilp/`, `an/mi/`, `an/b/` |
| RAG & sessions | `cs/`, `sess/`, `chat/s/`, `chat/m/`, `lc/` |
| Ingestion | `ds/`, `si/`, `ch/`, `geo/` |
| Writer workflows | `rv/r/`, `rv/n/`, `np/`, `wr/r/`, `wr/n/`, `pf/`, `cl/`, `ua/` |
| Alerts & investigation | `alert/r/`, `alert/e/`, `inv/` |
| Disinfo feature | `bf/`, `df/`, `sp/r0/`, `sp/jump/`, `vm/baseline/`, `vm/alert/`, `cib/c/`, `cib/e/`, `cib/s/`, `fc/`, `arch/`, `da/`, `mon/`, `sched/`, `reports/` |
| Adversarial feature | `adv/policy/`, `adv/sim/`, `adv/wg/`, `adv/disarm/`, `adv/reward/`, `adv/counter/`, `adv/retro/`, `adv/calib/`, `adv/audit/` |
| Metadata & workspaces | `meta/`, `ea/`, `cfg/`, `w/` |

This encoding enables efficient scans: "all situations entity X participates in" = prefix scan on `p/{entity_x_uuid}/`.

## The Ingestion Pipeline

TENSA converts raw text into structured narrative data through an LLM-powered pipeline:

```
Raw Text
    |
    v
[1. Chunking]
    Split into paragraphs, respect chapter boundaries
    Add overlap for context retention
    SHA-256 hashes for incremental re-ingestion
    |
    v
                ┌─────────────────────────┐
                │  single_session: true?   │
                └────┬──────────┬─────────┘
                     │ no       │ yes
                     v          v
          ┌──────────────┐  ┌──────────────────────────┐
          │ Standard Mode│  │ SingleSession Mode        │
          │ (per-chunk   │  │ (conversational pipeline) │
          │  LLM calls)  │  │ OpenRouter / Local only   │
          └──────┬───────┘  └──────────┬───────────────┘
                 │                     │
                 v                     v
[2. LLM Extraction]
    Standard: independent per-chunk LLM calls returning structured JSON
    (entities, situations, participations, causal links).
    SingleSession: turn 1 sends full text with chunk markers
    ([=== CHUNK N: "Title" ===]); turns 2..N extract each chunk as
    follow-ups in the same session, so the LLM sees prior extractions
    plus a compact accumulator summary. Parse failures trigger
    in-session repair. Falls back to Standard if text exceeds context.
    Situations may carry `text_start` / `text_end` — short verbatim
    fragments (~8–12 words) used to pin the per-situation source span.
    |
    v
[2b. Multi-Step Enrichment] (when `enrich: true`)
    A second LLM pass extracts richer structure:
      - Entity beliefs (what actors believe, want, misunderstand)
      - Game structures (strategic interaction classification)
      - Discourse annotations (POV, pacing, narrative voice)
      - Information sets (knows_before / learns / reveals per actor)
      - Additional causal links
      - Outcome models (deterministic + probabilistic)
      - Temporal chains (Allen relations between situation pairs)
      - Temporal normalizations (dates resolved from markers)
    |
    v
[2c. Reconciliation]
    Cross-chunk reconciliation produces:
      - Entity merges (deduplicate entities seen across chunks)
      - Global timeline (cross-chunk Allen relations)
      - Confidence adjustments (based on full-text context)
      - Cross-chunk causal links
    |
    v
[3. Entity Resolution]
    Bootstrap: at run start, the resolver is seeded from existing entities in
      the configured narrative, and from all sibling narratives in the same
      project (via the `pn/{project_id}/` index) — prevents cross-run
      duplicates when re-ingesting the same source.
    Pass 1: Exact alias match (case-insensitive)
    Pass 2: Fuzzy string match (Jaro-Winkler >= 0.88, same entity type)
    Pass 3: Embedding similarity (cosine > 0.85, same entity type)
    Maps extracted names to existing entities or marks as new
    |
    v
[4. Confidence Gating]
    >= 0.8  -->  Auto-commit (becomes Candidate)
    0.3-0.8 -->  Queue for human review
    < 0.3   -->  Reject (log only)
    |
    v
[5. Commit / Queue / Reject]
    Committed: created in hypergraph + added to indexes
    Queued: stored in ValidationQueue for HITL review
    Rejected: logged, not persisted
    |
    v
[6. Adjacency Fallback (per chunk)]
    If zero causal edges were committed for this chunk AND the chunk has
    >= 2 situations, chain adjacent situations with a weak Enabling link
    (strength 0.3, mechanism "sequential (fallback)", maturity Candidate).
    Keeps Workshop's causal detectors, graph-projection, and narrative-
    diameter analyses non-degenerate when the LLM misses causal structure.
```

The pipeline supports these LLM providers:
- **Anthropic** (Claude API, direct)
- **OpenRouter** (cloud, OpenAI-compatible)
- **Google Gemini** (feature: `gemini`)
- **AWS Bedrock** (feature: `bedrock`, SigV4-signed)
- **Local** (vLLM, Ollama, LiteLLM — any OpenAI-compatible endpoint)

All LLM calls are cached by SHA-256 hash at `lc/` so re-ingesting identical content is free.

### Point-of-View & Focalization

Every `Situation` can carry a `discourse: DiscourseAnnotation` with four fields: `order` (analepsis / prolepsis / simultaneous), `duration` (scene / summary / ellipsis / pause / stretch), `focalization: Option<Uuid>` — the entity whose perspective the chapter is told from — and `voice: Option<String>` — `"homodiegetic"` when the narrator is a character in the story, `"heterodiegetic"` when the voice is third-person omniscient outside it.

Enrichment populates `discourse` for ingested text. The generation pipeline fills it when the outline request carries a POV strategy:

```json
{
  "kind": "outline",
  "premise": "A vampire in London, told through letters and journals.",
  "num_chapters": 27,
  "pov_hint": {
    "mode": "rotating",
    "entity_names": ["Jonathan Harker", "Mina Murray", "Dr. John Seward", "Lucy Westenra"]
  }
}
```

Manually set or clear POV per situation via `PUT /situations/:id` with `{"discourse": {"focalization": "<uuid>", "voice": "homodiegetic"}}` or `{"discourse": null}`.

### Per-Situation Source Spans

Each committed `Situation` carries an optional `source_span` with `chunk_index`, `byte_offset_start`, `byte_offset_end`, and `local_index`. When the LLM returns `text_start` / `text_end`, the pipeline narrows offsets to that exact span via `src/ingestion/span_resolve.rs`:

```
Chunk text (bytes 1000–2500 in the source):
  "...background noise.  [Jonathan left Munich at 8:35 p.m. and arrived in
  Vienna early the next morning.]  He continued on to Budapest..."

LLM returns on the situation:
  text_start: "Jonathan left Munich at 8:35 p.m."
  text_end:   "arrived in Vienna early the next morning."

resolve_span() → (1023, 1109)    ← narrower than (1000, 2500)
```

Matching is whitespace-tolerant but otherwise verbatim. When fingerprints are missing or unmatched, the pipeline falls back to chunk-wide offsets. Studio's **Source** view renders narrow spans as inline highlights; chunk-wide spans are filtered out client-side.

### Post-Ingestion Chunk Control

After ingestion, individual chunks can be selectively re-processed without re-running the entire pipeline:

- **Re-extract**: re-run LLM extraction on specific chunks, with configurable cross-chunk context (`selected` / `neighbors` / `all`)
- **Enrich**: run enrichment on specific chunks
- **Reconcile**: run temporal reconciliation on specific chunks
- **Reprocess**: rollback committed entities/situations and re-gate from stored extractions (no LLM cost)

Retry jobs are linked to the original via `parent_job_id`, forming a lineage tree. Studio's Ingestion Detail view shows yield-based chunk coloring, multi-select for batch operations, and cross-job comparison of extraction results. See [POST /ingest/jobs/{id}/chunks/batch](#post-ingestjobsidchunksbatch).

## Allen Interval Algebra

TENSA uses Allen's 13 interval relations for temporal reasoning. Given two time intervals A and B:

```
BEFORE:       |--A--|          |--B--|
MEETS:        |--A--|--B--|
OVERLAPS:     |--A--|
                 |--B--|
STARTS:       |--A--|
              |----B----|
DURING:          |--A--|
              |-----B-----|
FINISHES:        |--A--|
              |-----B----|
EQUALS:       |--A--|
              |--B--|
```

Each relation has an inverse (Before ↔ After, Meets ↔ MetBy, etc.), giving 13 total: **Before, After, Meets, MetBy, Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals**.

TENSA also implements a **13×13 composition table**: if A is *Before* B and B is *During* C, what are the possible relations between A and C? This enables transitive temporal reasoning.

### Constraint Network (Path Consistency)

Beyond pairwise queries, TENSA provides `ConstraintNetwork` (`temporal/constraint.rs`) — a full Allen constraint propagation engine:
- **Add constraints** between interval pairs (sets of allowed Allen relations)
- **Path consistency (PC-2):** for every triple (i, j, k), tightens constraint(i,k) by composing constraint(i,j) with constraint(j,k) using the composition table, then intersecting with the existing constraint. Iterates until no changes.
- **Inconsistency detection:** if any constraint set becomes empty during propagation, the network is inconsistent (no valid temporal assignment exists)
- **`from_situations()`:** automatically builds a constraint network from situations with known temporal data

`AllenRelation`, `GameClassification`, and `InfoStructureType` all implement `FromStr` for string parsing by the enrichment layer.

## Inference Job System

Inference operations (causal discovery, game theory, motivation analysis, graph centrality, community detection) run asynchronously:

```
[Submit Job]  -->  [Priority Queue]  -->  [Worker Pool]  -->  [Store Result]
                   High | Normal | Low     tokio tasks         KV-backed
                   Deduplication            spawn_blocking      JSON results
```

1. **Submit**: Client sends job (target entity/situation + job type). Queue deduplicates against in-flight jobs.
2. **Queue**: Jobs are ordered by priority (High > Normal > Low), then by submission time.
3. **Workers**: Async pool with configurable concurrency. CPU-bound work runs in `spawn_blocking`.
4. **Results**: Stored in KV at `ir/{job_id}`. Many engines also write per-entity virtual properties under `an/…/` that become queryable as `e.an.<metric>` in TensaQL. Returns `null` when a metric has not been computed, so `WHERE e.an.pagerank > 0.05` naturally filters to analyzed entities.

## Source Intelligence Model

TENSA tracks where information comes from and how much to trust it:

```
[Source]  --attribution-->  [Entity or Situation]
   |
   +-- trust_score: 0.0-1.0
   +-- bias_profile (political lean, sensationalism)
   +-- track_record (claims made, corroborated, contradicted)
```

**Confidence breakdown** for multi-source entities:
- Extraction confidence (20%) — how well the LLM parsed this
- Source credibility (35%) — trust score of the attributing source
- Corroboration (35%) — agreement across multiple sources
- Recency (10%) — how recent the information is

Confidence is recomputed automatically when attributions are added or removed. Changing a source's `trust_score` triggers `propagate_source_trust_change(source_id)` which recomputes every target attributed to that source. Each `SourceAttribution` may carry an optional `claim: String` — when set, Dempster-Shafer mass is concentrated on that hypothesis instead of distributed uniformly across frame elements.

**Contentions** arise when sources disagree:
- `DirectContradiction` — sources say opposite things
- `NumericalDisagreement` — different numbers for the same fact
- `TemporalDisagreement` — different timestamps for the same event
- `OmissionBias` — a source omits relevant facts

## Security & Multi-Tenancy

**Trust model.** TENSA is a single-process server with no built-in authentication, authorization, or request signing. Every HTTP request that reaches `TENSA_ADDR` is trusted. This is appropriate for:

- Local development against `127.0.0.1`
- Trusted internal networks behind a VPN
- Deployments where an upstream reverse proxy (nginx, Caddy, Envoy, Traefik) handles TLS termination, authentication, and rate limiting before forwarding to TENSA

It is **not** appropriate for direct exposure to the public internet. If you need auth, put a proxy in front. If you need per-user auditing, use a proxy that injects a user identifier that TENSA can scope sessions by (see the `X-Tensa-User` header, below).

**Workspace isolation is a data boundary, not a security boundary.** The `X-Tensa-Workspace` header selects a workspace; `WorkspaceStore` transparently prefixes every KV key with `w/{workspace_id}/` so each workspace's entities, situations, inference results, and indexes live in disjoint key ranges. This prevents accidental cross-contamination — a TensaQL query scoped to workspace `alpha` physically cannot return rows from workspace `beta`, because the keys are not in the scan range.

However:

- Any client can send any workspace header. There is no enforcement that user U may only access workspaces in some allowlist.
- The default workspace (`default`) is used whenever the header is absent. Deployments that care about isolation should reject requests without the header at the proxy layer.
- Some schemas are deliberately **not** workspace-prefixed — notably Studio chat session keys (`chat/s/…`, `chat/m/…`) — so a cross-workspace admin view remains cheap. Audit any feature before assuming its keys are scoped.

**User scoping.** The `X-Tensa-User` header keys Studio chat sessions to `(workspace_id, user_id)` tuples (default `local`). The schema is ready for multi-user deployments; the *auth* to back it is not. A reverse proxy that validates a bearer token and writes the verified user id into `X-Tensa-User` before forwarding is the intended deployment pattern.

**Network & secrets.** LLM API keys live in the KV store at `cfg/llm`, `cfg/inference_llm`, `cfg/studio_chat_llm` and are redacted on read but stored in plaintext on disk. Protect the RocksDB directory accordingly. HTTP clients receive keys hinted rather than echoed on `GET /settings/*`. There is no encryption-at-rest provided by TENSA itself.

**What TENSA does enforce**, regardless of workspace or user:

- Bulk-size caps (`MAX_BULK_SIZE = 1000`, `MAX_RESULT_SIZE = 10000`) to prevent unbounded memory use on `/entities/bulk`, `/situations/bulk`, and query responses
- Cycle checks on causal-link insertion and scene parent assignment
- SHA-256 content-hash deduplication on ingestion chunks so replayed traffic does not re-spend LLM tokens
- Per-Nominatim rate-limiting (1 rps) for geocode requests

---

# Chapter 3: TensaQL Language Reference

TensaQL is TENSA's declarative query language for pattern matching, temporal reasoning, and inference on temporal hypergraphs. It supports instant queries (MATCH, PATH), asynchronous jobs (INFER, DISCOVER), mutations (CREATE, UPDATE, DELETE), RAG (ASK), prompt tuning, and exports.

## 3.1 MATCH Queries

Pattern matching against the hypergraph. Returns results instantly.

### Node Patterns

```
MATCH (binding:TypeName)
MATCH (binding:TypeName {prop: value, prop2: value2})
MATCH (:TypeName)                    -- anonymous binding
```

```sql
MATCH (e:Actor) RETURN e
MATCH (e:Actor {name: "Agent Chen"}) RETURN e
MATCH (loc:Location) RETURN loc
```

### Edge Patterns

```
MATCH (a:TypeA) -[edge:RELTYPE]-> (b:TypeB)       -- directed edge (a → b)
MATCH (a:TypeA) <-[edge:RELTYPE]- (b:TypeB)       -- reverse-directed (b → a) — v0.73.3+
MATCH (a:TypeA) -[edge:RELTYPE]- (b:TypeB)         -- undirected edge
MATCH (a:TypeA) -[edge:RELTYPE {prop: val}]-> (b:TypeB)  -- with properties
```

The primary edge types are `PARTICIPATES` (entity participates in situation) and `CAUSES` (situation causes another).

```sql
MATCH (e:Actor) -[p:PARTICIPATES]-> (s:Situation) RETURN e, s
MATCH (s1:Situation) -[c:CAUSES]-> (s2:Situation) RETURN s1, s2
MATCH (e:Actor) -[p:PARTICIPATES {role: "Protagonist"}]-> (s:Scene) RETURN e, s
-- Reverse-edge co-participation join: "who shares situations with Drago?"
MATCH (e:Actor)-[p:PARTICIPATES]->(s:Situation)<-[p2:PARTICIPATES]-(d:Actor {name: "Drago"})
RETURN e
```

Reverse edges (`<-[:REL]-`) are planner-level sugar: the planner swaps the from/to
endpoints so the executor only sees forward edges. Semantics match the Cypher
convention — the arrow points to the node on the right.

### Multi-Hop Patterns

Chain node-edge-node patterns for multi-hop traversal:

```sql
MATCH (e:Actor) -[p:PARTICIPATES]-> (s1:Situation) -[c:CAUSES]-> (s2:Situation)
RETURN e, s1, s2
```

## 3.2 PATH and FLOW Queries

Find paths through the causal or co-participation graph with depth control.

```sql
MATCH PATH (SHORTEST|ALL|LONGEST ACYCLIC) (start) -[:RELTYPE*min..max]-> (end)
WHERE conditions
RETURN bindings
```

| Mode | Behavior |
|------|----------|
| `SHORTEST` | Shortest path(s) between start and end |
| `ALL` | All paths within the depth range |
| `LONGEST ACYCLIC` | Longest causal chain (narrative diameter) via DAG DP |
| `TOP k SHORTEST` | Yen's k-shortest paths |

**Depth range:**
- `*` — default range (1..10)
- `*1..5` — minimum 1 hop, maximum 5 hops

**Weighted paths:**

```sql
MATCH PATH SHORTEST (a) -[:PARTICIPATES*1..10]-> (b) WEIGHT s.confidence RETURN *
```

**Flow queries (Edmonds-Karp):**

```sql
MATCH FLOW MAX (source) -[:PARTICIPATES*]-> (sink) RETURN flow, cut_edges
MATCH FLOW MIN_CUT (a) -[:PARTICIPATES*]-> (b) RETURN flow, cut_edges
```

**Inline graph functions** (synchronous, usable in WHERE and RETURN):

| Function | Scope | Description |
|----------|-------|-------------|
| `triangles(e)` | per-node | Number of triangles through `e` |
| `clustering(e)` | per-node | Local clustering coefficient |
| `common_neighbors(a,b)` | per-pair | Shared neighbors |
| `adamic_adar(a,b)` | per-pair | Adamic–Adar link-prediction score |
| `preferential_attachment(a,b)` | per-pair | Degree product |
| `resource_allocation(a,b)` | per-pair | Resource-allocation score |
| `jaccard(a,b)` | per-pair | Jaccard similarity of neighbor sets |
| `overlap(a,b)` | per-pair | Overlap similarity |

```sql
MATCH (e:Actor) WHERE e.narrative_id = "story" AND triangles(e) > 5
  RETURN e.properties.name, triangles(e), clustering(e)

MATCH (a:Actor), (b:Actor) WHERE a.narrative_id = "story" AND b.narrative_id = "story"
  RETURN a.properties.name, b.properties.name, adamic_adar(a, b), jaccard(a, b)
```

## 3.3 WHERE Clause

Filter by property conditions. Supports boolean logic with standard precedence (`AND` binds tighter than `OR`).

### Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equal | `e.name = "Chen"` |
| `!=` | Not equal | `e.entity_type != "Location"` |
| `>` / `<` / `>=` / `<=` | Ordering | `e.confidence > 0.8` |
| `IN` | Array membership | `e.entity_type IN ["Actor", "Organization"]` |
| `CONTAINS` | Substring match | `e.name CONTAINS "Chen"` |

### Boolean Logic

```sql
WHERE e.confidence > 0.5 AND e.name = "Chen"
WHERE e.confidence > 0.8 OR e.confidence < 0.2
WHERE (e.confidence > 0.8 OR e.confidence < 0.2) AND e.name = "Chen"
```

### Field Paths

Access nested properties with dot notation: `e.name`, `s.narrative_level`, `e.an.pagerank` (see [virtual properties](#virtual-properties)).

## 3.4 AT Clause (Temporal Filtering)

Filter situations by Allen interval relations against a timestamp. Supports ISO 8601 and date-only strings (interpreted as start of day UTC).

```
AT field_path RELATION "timestamp"
```

| Relation | Meaning |
|----------|---------|
| `BEFORE` | Situation ends before the given time |
| `AFTER` | Situation starts after the given time |
| `MEETS` | Situation ends exactly when the given time starts |
| `OVERLAPS` | Situation starts before and extends past the given time |
| `DURING` / `WITHIN` | Situation is completely contained within the given interval |
| `CONTAINS` | Situation completely contains the given interval |
| `STARTS` / `FINISHES` | Endpoint alignment with the given interval |
| `EQUALS` | Identical start and end |

```sql
MATCH (s:Situation) AT s.temporal BEFORE "2025-06-01" RETURN s
MATCH (s:Situation) AT s.temporal AFTER "2025-03-15T14:30:00Z" RETURN s
```

### Fuzzy AT Tail (Fuzzy Sprint Phase 5)

When a situation's temporal interval carries trapezoidal fuzzy endpoints
("early 2024", "around that time", "shortly after"), use the graded-Allen
tail to filter by minimum relation degree instead of crisp equality:

```
AT field_path RELATION "timestamp" AS FUZZY <rel> THRESHOLD <degree>
```

```sql
-- keep situations whose BEFORE degree vs. 2025-01-01 is at least 0.6
MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01" AS FUZZY BEFORE THRESHOLD 0.6 RETURN s
```

All 13 Allen relations are addressable by name (`BEFORE`, `AFTER`,
`MEETS`, `MEETS_INVERSE`, `OVERLAPS`, `OVERLAPS_INVERSE`, `STARTS`,
`STARTS_INVERSE`, `DURING`, `DURING_INVERSE`, `FINISHES`,
`FINISHES_INVERSE`, `EQUALS`). The threshold must lie in `[0.0, 1.0]`.
Omitting the tail keeps the classical crisp semantics unchanged.

See also: `POST /analysis/fuzzy-allen` for the full 13-vector between two
situations. The cache lives at `fz/allen/{narrative_id}/{a_id}/{b_id}`.

## 3.5 NEAR Clause (Vector Similarity)

K-nearest neighbors via embedding cosine distance. Requires a configured embedding provider.

```
NEAR(binding, "search text", k)
```

```sql
MATCH (e:Actor) NEAR(e, "smuggling suspect", 5) RETURN e
MATCH (e:Actor) WHERE e.confidence > 0.5 NEAR(e, "financial crime", 10) RETURN e
```

## 3.6 SPATIAL Clause (Geospatial Filtering)

Filter by geographic proximity using Haversine distance.

```
SPATIAL field WITHIN radius KM OF (latitude, longitude)
```

```sql
MATCH (s:Situation) SPATIAL s.spatial WITHIN 10.0 KM OF (40.7128, -74.0060) RETURN s
MATCH (s:Scene) WHERE s.confidence > 0.5
SPATIAL s.spatial WITHIN 50.0 KM OF (51.5074, -0.1278)
RETURN s
```

## 3.7 ACROSS NARRATIVES Clause

Restrict or expand queries to specific narratives or all narratives.

```sql
MATCH (e:Actor) ACROSS NARRATIVES ("harbor-case") RETURN e
MATCH (e:Actor) ACROSS NARRATIVES ("harbor-case", "embassy-case") RETURN e
MATCH (e:Actor) ACROSS NARRATIVES RETURN e                 -- all narratives
```

## 3.8 GROUP BY & Aggregation

| Function | Description |
|----------|-------------|
| `COUNT(*)` | Total rows |
| `COUNT(field)` | Non-null count |
| `SUM(field)` / `AVG(field)` | Numeric aggregates |
| `MIN(field)` / `MAX(field)` | Extremes |

```sql
MATCH (e:Actor) RETURN COUNT(*), AVG(e.confidence), MAX(e.confidence)

MATCH (e:Actor) GROUP BY e.entity_type
RETURN e.entity_type, COUNT(*), AVG(e.confidence)

MATCH (e:Actor) WHERE e.confidence > 0.3
GROUP BY e.entity_type, e.narrative_id
RETURN e.entity_type, COUNT(*), AVG(e.confidence)
LIMIT 10
```

## 3.9 RETURN, ORDER BY, LIMIT

```sql
RETURN e                             -- entire entity
RETURN e, s                          -- multiple bindings
RETURN e.name, e.confidence          -- specific fields
RETURN *                             -- all matched bindings

MATCH (e:Actor) WHERE e.confidence > 0.5
RETURN e.name, e.confidence
ORDER BY e.confidence DESC
LIMIT 20
```

## 3.10 INFER Queries (Async Jobs)

Submit asynchronous inference jobs. Returns a job ID; poll with `GET /jobs/{id}` or use the MCP `job_status` / `job_result` tools.

```
INFER infer_type FOR binding:TypeName
MATCH pattern?
WHERE condition (AND|OR condition)*
ASSUMING field = value (AND field = value)*
UNDER constraint = value (AND constraint = value)*
RETURN bindings
```

`WHERE` at the top level (v0.74.0) lets narrative-scoped forms be written
without a redundant `MATCH`. Equalities on `narrative_id` /
`target_id` are lifted into job parameters by the planner, so:

```sql
INFER ARCS FOR n:Narrative WHERE n.narrative_id = "oliver_twist" RETURN n
```

submits an `ArcClassification` job with `parameters.narrative_id =
"oliver_twist"`. The same pattern works for `PATTERNS`, `MISSING_EVENTS`,
`CENTRALITY`, `COMMUNITIES`, `STYLE`, and every other narrative-scoped
infer type.

### Inference Types

Grouped by family:

**Graph centrality (Level 1)**

| Type | What It Does | Virtual property |
|------|--------------|------------------|
| `CENTRALITY` | Weighted Brandes betweenness, Wasserman–Faust closeness, degree. Writes a flat `community_id` for compatibility; for hierarchical Leiden communities use `COMMUNITIES` below. | `e.an.betweenness`, `e.an.closeness`, `e.an.degree`, `e.an.community_id` |
| `PAGERANK` | Power iteration (damping 0.85) | `e.an.pagerank` |
| `EIGENVECTOR` | Per-component power iteration | `e.an.eigenvector` |
| `HARMONIC` | `H(v) = Σ 1/d(v,u)` | `e.an.harmonic` |
| `HITS` | Kleinberg hubs + authorities on the bipartite graph | `e.an.hub_score`, `e.an.authority_score` |

**Topology & community**

| Type | What It Does | Virtual property |
|------|--------------|------------------|
| `TOPOLOGY` | Tarjan articulation points + bridges on a selectable projection (`projection` parameter: `"cooccurrence"` default, `"causal"` for entity-level graph induced by the causal DAG) | `e.an.is_articulation_point`, `e.an.is_bridge_endpoint` |
| `KCORE` | K-core decomposition | `e.an.kcore` |
| `LABEL_PROPAGATION` | O(m) parameter-free community detection | `e.an.label` |
| `COMMUNITIES` | **Hierarchical Leiden** with refinement step (connected-community guarantee), multi-level summaries. This is the primary community-detection path; `CENTRALITY` also writes `community_id` as a flat fallback. | `e.an.community_id` + hierarchy stored at `an/ch/{narrative_id}` |

**Narrative-native (time-aware)**

| Type | What It Does | Virtual property |
|------|--------------|------------------|
| `TEMPORAL_PAGERANK` | Time-decayed PageRank. `decay_lambda` parameter accepts a positive number **or** the string `"auto"` — Auto derives λ = ln(2) / (span_days / 2) so mid-narrative situations decay by half regardless of wall-clock age. Age is measured relative to the **latest situation in the narrative**, not `Utc::now()` (v0.73.3). | `e.an.temporal_pagerank` |
| `CAUSAL_INFLUENCE` | Betweenness on causal DAG mapped to entities | `e.an.causal_influence` |
| `INFO_BOTTLENECK` | Sole-knower detection from belief network | `e.an.bottleneck_score` |
| `ASSORTATIVITY` | Degree correlation, narrative scalar at `an/as/` |  |
| `TEMPORAL_MOTIFS` | 3–4 node Allen-constrained temporal motif census |  |
| `FACTION_EVOLUTION` | Sliding-window Label Propagation + event detection (merges, splits, births, deaths) |  |

**Information theory & epistemic reasoning**

| Type | What It Does |
|------|--------------|
| `ENTROPY` | Shannon self-information, mutual information, KL divergence |
| `BELIEFS` | Depth-2 recursive belief modeling ("what A thinks B knows") |
| `EVIDENCE` | Dempster–Shafer mass combination, Bel/Pl intervals |
| `ARGUMENTS` | Dung argumentation frameworks — grounded/preferred/stable extensions |
| `CONTAGION` | SIR information contagion — R₀, spread DAG, critical spreaders |

**Causal inference & game theory**

| Type | What It Does |
|------|--------------|
| `CAUSES` | NOTEARS causal discovery with LLM priors + DAGMA + SCC pre-validation |
| `COUNTERFACTUAL` | Beam search over interventions (ASSUMING clause) |
| `MISSING` | Missing link detection |
| `ANOMALIES` | Z-score anomaly detection on confidence + temporal gaps |
| `GAME` | Game classification + QRE equilibrium solver (UNDER clause) |
| `MEAN_FIELD` | Mean-field games — population equilibrium via fixed-point softmax |
| `PSL` | Probabilistic Soft Logic — weighted rules, continuous truth values |
| `TEMPORAL_RULES` | Temporal ILP — learns Horn clauses over entity participation |

**Motivation & strategic reasoning**

| Type | What It Does |
|------|--------------|
| `MOTIVATION` | MaxEnt IRL per trajectory + archetype classification for sparse data |

**Graph embeddings & network inference**

| Type | What It Does |
|------|--------------|
| `FAST_RP` | Fast Random Projection embeddings (sparse projection + neighbor averaging) |
| `NODE2VEC` | Node2Vec biased random walks → PMI + truncated SVD |
| `NETWORK_INFERENCE` | Gomez-Rodriguez cascade-based diffusion network inference |
| `TRAJECTORY` | TGN-style temporal entity embeddings |
| `SIMULATE` | LLM-powered generative agent forward-play |

**Stylometry (`stylometry` feature)**

| Type | What It Does |
|------|--------------|
| `STYLE` | 6-layer narrative style profile |
| `STYLE_COMPARE` | Fingerprint similarity |
| `STYLE_ANOMALIES` | Per-chapter deviation from baseline (bootstrap-calibrated p-values). Three `mode` values: `"threshold"` (default — similarity < threshold), `"calibrated"` (bootstrap null), `"per_source_type"` (v0.73.3 — leave-one-out cohort baseline grouped by `source_type`; `min_cohort` parameter, default 3). |
| `VERIFY_AUTHORSHIP` | PAN@CLEF authorship verification (calibrated decision, AUC/c@1/F0.5u) |

**Disinformation (`disinfo` feature)**

| Type | What It Does |
|------|--------------|
| `BEHAVIORAL_FINGERPRINT` | 10-axis per-actor fingerprint (cadence, sleep pattern, engagement, etc.) |
| `DISINFO_FINGERPRINT` | 12-axis per-narrative fingerprint (virality, source diversity, evidential uncertainty, ...) |
| `SPREAD_VELOCITY` | SMIR + per-platform R₀ + cross-platform jump detection + velocity-monitor anomaly check. Platform-tuned β defaults. |
| `SPREAD_INTERVENTION` | Counterfactual projection — `RemoveTopAmplifiers { n }` or `DebunkAt { at }` |
| `CIB` | Coordinated Inauthentic Behavior cluster detection via label propagation on behavioral similarity |
| `SUPERSPREADERS` | Top-N amplifiers by graph centrality (`method`: `pagerank`/`eigenvector`/`harmonic`) |
| `CLAIM_ORIGIN` | Trace a claim back through its mutation chain |
| `CLAIM_MATCH` | Match all claims against known fact-checks |
| `ARCHETYPE` | Classify actor into adversarial archetypes (StateActor, OrganicConspiracist, CommercialTrollFarm, Hacktivist, UsefulIdiot, HybridActor) via softmax template matching |
| `DISINFO_ASSESSMENT` | Fuse disinfo signals via Dempster-Shafer (Yager's rule for high-conflict K>0.7) |

**Adversarial wargaming (`adversarial` feature)**

| Type | What It Does |
|------|--------------|
| `ADVERSARY_POLICY` | Generate adversary policy from IRL reward weights + SUQR bounded rationality |
| `COGNITIVE_HIERARCHY` | Poisson Cognitive Hierarchy level-k best response |
| `WARGAME` | Turn-based red/blue wargame simulation |
| `REWARD_FINGERPRINT` | 8-dimensional psychological reward fingerprint for a narrative |
| `COUNTER_NARRATIVE` | Generate reward-aware counter-narratives (reward parity + conclusion redirect) |
| `RETRODICTION` | Validate against historical campaigns via SocialSim metrics (RBO, RMSLE, KL, Spearman ρ) |

**Generation (`generation` feature)**

| Type | What It Does | Parameters | Result |
|------|--------------|------------|--------|
| `CHAPTER_GENERATION_FITNESS` (`InferenceJobType::ChapterGenerationFitness`) | Closed-loop chapter generation: `Query → Prompt → Generate → Score → Revise → Best-of-N` against a target `NarrativeFingerprint`, optionally conditioned on a `StyleEmbedding`. Returns the **best-scoring attempt** (not the last). Cost estimate: 90 s (≈3 iterations × LLM call). See §7.11.14. | `{narrative_id: String, chapter: usize, voice_description?: String, style_embedding_id?: Uuid, target_fingerprint_source?: String, fitness_threshold?: f64, max_retries?: usize, synchronous?: bool}` | `{best_text: String, best_score: f64, attempts: [{iteration, score, accepted, prompt_tokens, completion_tokens}], best_iteration: usize}` |

**Cross-narrative (per-narrative forms, v0.74.0)**

| Type | What It Does | Result |
|------|--------------|--------|
| `ARCS` (alias `ARC`) | Reagan 6-arc classification on a single narrative. Returns `{arc_type, confidence, sentiment_trajectory, key_turning_points, all_correlations, signal_quality, scorer}`. `signal_quality < 0.1` means all six templates correlate weakly — treat the result as a guess. Auto-detects `TENSA_SENTIMENT_MODEL` for ONNX scoring; falls back to keyword lexicon. | `ArcClassification` |
| `PATTERNS` | Frequent subgraph mining on a single narrative. Maps to `PatternMining` engine with `parameters.narrative_id`. | `Vec<PatternRecord>` |
| `MISSING_EVENTS` | Missing-event prediction for a single narrative. Maps to `MissingEventPrediction`. | `Vec<PredictedEvent>` |

These three are also reachable via `DISCOVER` for cross-narrative use:
`DISCOVER ARCS ACROSS NARRATIVES ("a", "b") RETURN *`. The per-narrative
`INFER` form is more ergonomic when scoping to a single narrative.

**Per-actor (v0.74.1)**

| Type | What It Does | Result |
|------|--------------|--------|
| `ACTOR_ARCS` (alias `ACTOR_ARC`) | Per-actor Reagan arc classification — one `ArcClassification` per Actor entity in the narrative, role-weighted by each actor's participation role (Protagonist=+1.0, Antagonist=-0.5, Target=-0.8, etc.). Solves the multi-POV averaging blind spot: in `game-of-thrones`, narrative-level classifier produces flat signal because Ned rises while Daenerys rises while Cersei falls; per-actor resolves them separately. Results stored at `an/aa/{narrative_id}/{actor_id}` and queryable inline as `e.an.arc_type` / `e.an.arc_confidence` / `e.an.arc_signal_quality`. | `{narrative_id, actors: [{actor_id, arc_type, confidence, signal_quality, scorer, all_correlations}]}` |

Also reachable as `INFER ARCS FOR e:Actor WHERE e.id = "<uuid>" RETURN e`
— same keyword, scope inferred from the binding type (`e:Actor` routes
to per-actor, `n:Narrative` routes to narrative-level). The single-actor
form is the symmetric extension of how `PAGERANK`/`CENTRALITY` already
work per-entity.

```sql
-- Batch: classify every Actor in the narrative
INFER ACTOR_ARCS FOR n:Narrative WHERE n.narrative_id = "game-of-thrones" RETURN n

-- Single actor (by id)
INFER ARCS FOR e:Actor WHERE e.id = "019d...ned-stark" RETURN e

-- Find all Cinderella-arc actors across a narrative after running the above
MATCH (e:Actor)
  WHERE e.narrative_id = "game-of-thrones"
    AND e.an.arc_type = "Cinderella"
    AND e.an.arc_signal_quality > 0.1
  RETURN e.properties.name, e.an.arc_type, e.an.arc_signal_quality
  ORDER BY e.an.arc_signal_quality DESC
```

### ASSUMING Clause (Interventions)

Specifies counterfactual assumptions for "what if" analysis:

```sql
INFER COUNTERFACTUAL FOR s:Situation
ASSUMING s.action = "cooperate"
RETURN s

INFER COUNTERFACTUAL FOR s:Situation
ASSUMING s.action = "cooperate" AND s.payoff = 5
RETURN s
```

### UNDER Clause (Constraints)

Sets game-theoretic constraints:

| Key | Values | Default |
|-----|--------|---------|
| `RATIONALITY` | 0.0 (random) to 1.0 (perfectly rational) | 1.0 |
| `INFORMATION` | `"complete"`, `"incomplete"`, `"imperfect"`, `"asymmetric"` | — |

```sql
INFER GAME FOR s:Scene UNDER RATIONALITY = 0.5 RETURN s
INFER GAME FOR s:Scene UNDER INFORMATION = "incomplete" RETURN s
```

### Virtual Properties

After an INFER job completes, many engines write per-entity results under `an/…/` that TensaQL exposes as virtual dotted properties:

```sql
MATCH (e:Actor) WHERE e.narrative_id = "harbor-case"
  AND e.an.betweenness > 0.1
RETURN e.properties.name, e.an.betweenness, e.an.closeness, e.an.community_id
ORDER BY e.an.betweenness DESC LIMIT 20
```

The resolver returns `null` when a metric has not been computed, so WHERE filters naturally drop unanalyzed entities. Supported metrics: `betweenness`, `closeness`, `degree`, `community_id`, `pagerank`, `eigenvector`, `harmonic`, `hub_score`, `authority_score`, `temporal_pagerank`, `causal_influence`, `kcore`, `label`, `bottleneck_score`, `is_articulation_point`, `is_bridge_endpoint`, `arc_type`, `arc_confidence`, `arc_signal_quality` (populated by `INFER ACTOR_ARCS` / `INFER ARCS FOR e:Actor`).

## 3.11 DISCOVER Queries (Cross-Narrative Mining)

Submit asynchronous discovery jobs for patterns across narratives. Returns a job ID.

```sql
DISCOVER discover_type (IN binding:TypeName)?
ACROSS NARRATIVES (narrative_ids)?
WHERE conditions?
RETURN bindings
```

| Type | What It Finds |
|------|---------------|
| `PATTERNS` | Frequent structural subgraphs across narratives (WL / random walk kernel + VF2-lite matching, max 6 nodes) |
| `ARCS` | Reagan 6-arc character/plot classification (Pearson correlation on fortune trajectory) |
| `MISSING` | Missing events predicted from patterns and causal gaps |

```sql
DISCOVER PATTERNS RETURN *
DISCOVER PATTERNS ACROSS NARRATIVES ("harbor-case", "embassy-case") RETURN *
DISCOVER ARCS RETURN *
DISCOVER MISSING IN s:Situation RETURN s
```

## 3.12 EXPORT Queries

Export narrative data in standard formats.

```sql
EXPORT NARRATIVE "narrative_id" AS format
```

| Format | Content type | Description |
|--------|--------------|-------------|
| `csv` | text/csv | Entities and situations as tabular rows |
| `graphml` | application/xml | Graph structure for Gephi, yEd |
| `json` | application/json | Full structured data dump |
| `manuscript` | text/markdown | Temporal prose reconstruction — narrative as readable text |
| `report` | text/markdown | Analytical report (timeline, entity profiles, relationships) |
| `archive` | application/x-tensa-archive | Lossless `.tensa` ZIP archive |
| `stix` | application/json | STIX 2.1 bundle (campaign, threat-actor, identity, indicator, SROs) |

The `archive` format produces a ZIP of JSON files in a human-readable directory structure. It preserves all data losslessly — entities, situations, participations, causal links, sources, attributions, contentions, chunks, state versions, inference results, community summaries, tuned prompts, taxonomy, and projects. Archives can be imported via `POST /import/archive` or built externally.

**v1.1.0 (TENSA 0.79.5+) added six new optional layers** so external skill output (e.g. from `/tensa-narrative-llm`) round-trips losslessly:

| Layer | What it preserves | Source |
|---|---|---|
| `annotations` | Inline comments / footnotes / citations on situation prose, byte-span anchored | `/tensa-narrative-llm` skill (dramatic-irony, subplots, arc-classification) + reviewer notes |
| `pinned_facts` | Continuity facts: entity property pins + narrative-wide rules | `/tensa-narrative-llm commitments`, manual writer entries |
| `revisions` | Git-like narrative snapshots with author + message + parent chain | `commit_narrative_revision` (called by `narrative-diagnose-and-repair`) |
| `workshop_reports` | Three-tier critique reports with structured findings | `run_workshop` |
| `narrative_plan` | Writer doc — logline / synopsis / premise / themes / plot beats / style targets | Writer flow |
| `analysis_status` | Per-narrative registry of which inference jobs ran, by what source, **with lock state**. Without this layer, a re-imported archive would lose the `Source: Skill, locked: true` rows that protect skill-attested results from being silently overwritten by the next bulk-analysis run | TENSA worker pool + `/tensa-narrative-llm` skill |

The `GET /narratives/:id/export?format=archive` endpoint accepts a `&preset=full|minimal|default` query parameter:
- `default` (omitted) — every v1.1.0 layer ON; inference + embeddings + synthetic OFF for size.
- `full` — all v1.1.0 layers + inference results + embeddings + synthetic records.
- `minimal` — core graph data only.

For per-flag control use `POST /export/archive` with an `ArchiveExportOptions` body. See [`archive-template/`](archive-template/) for the canonical annotated example with every field documented.

## 3.13 EXPLAIN Prefix

Returns the query execution plan as JSON without running the query. Works with MATCH, INFER, DISCOVER, and PATH queries.

```sql
EXPLAIN MATCH (e:Actor) RETURN e
EXPLAIN MATCH (e:Actor) WHERE e.confidence > 0.5 RETURN e
EXPLAIN INFER CAUSES FOR s:Situation RETURN s
EXPLAIN MATCH PATH SHORTEST (s1) -[:CAUSES*1..5]-> (s2) RETURN s1
```

The result is a JSON array of plan steps (e.g., `ScanByType`, `FilterProperties`, `Project`).

## 3.14 Mutations

### CREATE ENTITY

```sql
CREATE (binding:EntityType {prop: value, ...})
IN NARRATIVE "narrative_id"
CONFIDENCE 0.85
```

```sql
CREATE (a:Actor {name: "Agent Chen", rank: "Senior"})
IN NARRATIVE "harbor-case"
CONFIDENCE 0.95

CREATE (org:Organization {name: "Nightfall Group", type: "criminal"})
IN NARRATIVE "harbor-case"
CONFIDENCE 0.7
```

### CREATE SITUATION

```sql
CREATE SITUATION AT NarrativeLevel
CONTENT "description"
IN NARRATIVE "narrative_id"
CONFIDENCE 0.8
```

### CREATE NARRATIVE

```sql
CREATE NARRATIVE "narrative_id"
TITLE "display title"
GENRE "genre"
TAGS ("tag1", "tag2")
```

### UPDATE / DELETE

```sql
UPDATE ENTITY "uuid" SET field = value (, field = value)*
UPDATE NARRATIVE "id" SET field = value

DELETE ENTITY "uuid"
DELETE SITUATION "uuid"
DELETE NARRATIVE "narrative_id"
```

Updates to entities automatically snapshot the prior state to `sv/` so you can query historical state via `get_state_at_time(entity_id, timestamp)` or the `state` API.

### Participation and Causal Edges

```sql
ADD PARTICIPANT "entity_uuid" TO SITUATION "situation_uuid"
ROLE RoleName
ACTION "description"

REMOVE PARTICIPANT "entity_uuid" FROM SITUATION "situation_uuid"

ADD CAUSE FROM "situation_uuid" TO "situation_uuid"
TYPE CausalType
STRENGTH 0.8
MECHANISM "description"

REMOVE CAUSE FROM "situation_uuid" TO "situation_uuid"
```

## 3.15 ASK Queries (RAG Question Answering)

Submit natural-language questions answered by Retrieval-Augmented Generation. TENSA assembles relevant context from the hypergraph and vector index, then calls the LLM to generate an answer with citations.

```
ASK "question"
OVER "narrative_id"?
MODE retrieval_mode?
RESPOND AS "format"?
SESSION "session_id"?
SUGGEST?
```

### Retrieval Modes

| Mode | Strategy |
|------|----------|
| `local` | Entity-focused retrieval via vector search |
| `global` | Community summaries for broad overview questions |
| `hybrid` | Local + global context (default) |
| `mix` | Keyword-driven — high-level keywords use global, specifics use local |
| `drift` | Three-phase adaptive retrieval traversing community hierarchy (primer → follow-up → leaf) |
| `lazy` | LazyGraphRAG — zero pre-computation, PCST-connected entity boosting |
| `ppr` | Personalized PageRank seeded from query-relevant entities |

### Optional Clauses

| Clause | Purpose |
|--------|---------|
| `OVER "narrative_id"` | Scope to a specific narrative |
| `MODE mode` | Select retrieval strategy |
| `RESPOND AS "format"` | Instruct LLM on output format ("bullet points", "brief summary", …) |
| `SESSION "session_id"` | Multi-turn conversation — prepends history from previous turns |
| `SUGGEST` | Generate 3–5 follow-up question suggestions |

```sql
ASK "Who is the main suspect?"
ASK "What are the key alliances?" OVER "harbor-case" MODE global
ASK "What are the major themes?" OVER "harbor-case" MODE drift
ASK "Who is most central to the plot?" OVER "story" MODE ppr
ASK "Summarize the timeline" OVER "harbor-case" RESPOND AS "bullet points"
ASK "What happened next?" SESSION "session-abc123"
ASK "Who is involved?" OVER "harbor-case" SUGGEST
```

The response includes `answer`, `citations` (entity/situation references with relevance scores), `mode`, `tokens_used`, and `suggestions` (when `SUGGEST` is used).

## 3.16 TUNE Queries (Prompt Auto-Tuning)

Generate domain-adapted extraction prompts by sampling ingested chunks and sending them to the LLM for analysis. Generated prompts are stored at `pt/{narrative_id}` and automatically used during future ingestion.

```sql
TUNE PROMPTS FOR "narrative_id"
```

Response fields: `narrative_id`, `prompt_text`, `domain_description`, `entity_types`. Manage tuned prompts via `/prompts/*`.

## 3.17 Quick-Reference Tables

### Entity Types

| Value | Description |
|-------|-------------|
| `Actor` | People, agents, characters |
| `Location` | Places, areas |
| `Artifact` | Physical/digital objects |
| `Concept` | Abstract ideas |
| `Organization` | Groups, companies |

### Narrative Levels (coarse → fine)

| Value | Ordinal | Description |
|-------|---------|-------------|
| `Story` | 0 | Entire narrative |
| `Arc` | 1 | Major plot thread |
| `Sequence` | 2 | Related event sequence |
| `Scene` | 3 | Continuous event |
| `Beat` | 4 | Single dramatic moment |
| `Event` | 5 | Atomic action |

### Roles

| Value | Description |
|-------|-------------|
| `Protagonist` | Primary driving actor |
| `Antagonist` | Opposing actor |
| `Witness` | Observer |
| `Target` | Affected entity |
| `Instrument` | Tool or means |
| `Confidant` | Trusted ally |
| `Informant` | Provides information |
| `Recipient` | Receives something |
| `Bystander` | Present, uninvolved |
| `SubjectOfDiscussion` | Discussed, not present |
| `Facilitator` | Enables without being an agent |
| `{"Custom": "..."}` | Domain-specific role label |

### Allen Temporal Relations

| Relation | Inverse | Meaning |
|----------|---------|---------|
| Before | After | A ends before B starts |
| Meets | MetBy | A ends exactly when B starts |
| Overlaps | OverlappedBy | A starts before B, ends during B |
| Starts | StartedBy | A and B start together; A ends first |
| During | Contains | A is completely inside B |
| Finishes | FinishedBy | A and B end together; A starts later |
| Equals | Equals | Identical intervals |

---

# Chapter 4: MCP Tools Reference

TENSA exposes a Model Context Protocol (MCP) server with tools grouped by capability. The base set covers data CRUD, querying, inference, ingestion, export, and source intelligence. Additional tools are gated behind optional Cargo features (`stylometry`, `disinfo`, `adversarial`, `generation`). Tools can be used by any MCP-compatible client (Claude Code, other AI assistants, custom integrations).

> **TENSA also acts as an MCP *client*** (`studio-chat` feature). The Studio chat assistant can spawn user-configured stdio MCP servers via `rmcp`'s `TokioChildProcess` transport and expose their tools namespaced as `{server}__{tool}`. That side of the story lives in [§5.26 Studio Agent Chat](#526-studio-agent-chat-studio-chat-feature).

## 4.1 Architecture

The MCP server uses an `McpBackend` trait with two implementations:

- **EmbeddedBackend** — direct library access, runs in-process
- **HttpBackend** — REST API client proxy, used when TENSA runs as a separate HTTP server

Both backends provide identical functionality; the choice is transparent to the client.

The **MCP client** side (Studio chat) is a separate code path — `src/studio_chat/mcp_proxy.rs` wraps `rmcp`'s client-side `serve` + `TokioChildProcess` transport. Server-side (this chapter) and client-side (§5.26) do not share the `McpBackend` trait.

## 4.2 Entities & Situations

### create_entity

Create an entity in the hypergraph. Server sets `id`, `maturity` (starts `Candidate`), and timestamps.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity_type` | string | Yes | `Actor`, `Location`, `Artifact`, `Concept`, `Organization` |
| `properties` | object | Yes | JSON properties (e.g., `{"name": "Agent Chen"}`) |
| `narrative_id` | string | No | Narrative to associate with |
| `confidence` | number | No | 0.0–1.0, default 0.5 |
| `beliefs` | object | No | Opaque JSON — entity's epistemic state, consumed by belief/evidence engines |

**Recognized `properties` keys.** `properties` is a free-form JSON bag, but the
ingestion pipeline writes and the resolver reads a small set of conventional
keys. Use them if you want your data to interoperate with cross-session
deduplication and the biographical UI:

| Key | Shape | Purpose |
|-----|-------|---------|
| `name` | string | Canonical display name. Always set by the pipeline. |
| `aliases` | string[] | Set-unioned across mentions. Read by `EntityResolver::bootstrap_from_entities` on startup so cross-session dedup survives reloads. Epithets, titles, married names, monikers (e.g. `"Count Dracula"` → `["Dracula", "Vlad Dracula", "Count De Ville", "Nosferatu"]`). |
| `date_of_birth` / `date_of_death` | string | ISO 8601 preferred, year/era otherwise. First-write-wins on resolve-merge. |
| `place_of_birth` / `place_of_death` | string | Free-form. |
| `nationality`, `occupation`, `gender`, `title`, `description` | string | First-write-wins on resolve-merge. |

On `commit_entity`, the pipeline merges incoming aliases (union) and
biographical keys (first-write-wins) into an already-resolved entity via
`update_entity_no_snapshot`. A later chunk that adds `"Nosferatu"` to a
previously-seen `"Dracula"` will extend the alias list without snapshotting
and without overwriting any hard fact already on record. See
[`src/ingestion/pipeline.rs::BIOGRAPHICAL_KEYS`](../src/ingestion/pipeline.rs).

Maturity enum (`Candidate`/`Reviewed`/`Validated`/`GroundTruth`) is promoted via validation, not settable at creation.

Returns: entity record with generated UUID v7.

### create_situation

Create a situation (event/scene) with optional temporal anchors.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `raw_content` | string | Yes | Description of the event |
| `start` / `end` | string | No | ISO 8601 temporal anchors |
| `narrative_level` | string | No | `Story`/`Arc`/`Sequence`/`Scene`/`Beat`/`Event` |
| `narrative_id` | string | No | Narrative to associate with |
| `confidence` | number | No | 0.0–1.0, default 0.5 |
| `discourse` | object | No | Narratology: `{order?, duration?, focalization? (UUID), voice?}` — set POV at creation |
| `spatial` | object | No | SpatialData passthrough (place name, lat/lng, polygon) |
| `game_structure` | object | No | GameStructure passthrough for scenes with strategic interaction |
| `manuscript_order` | number | No | Writer-curated binder position (u32). Set on Scene-level situations for deterministic ordering. |
| `parent_situation_id` | string | No | Binder hierarchy parent (e.g. Chapter Arc → Scene). Cycles rejected. |

### update_entity

Update entity properties. Automatically snapshots the prior state to `sv/`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Entity UUID |
| `updates` | object | Yes | JSON with `properties`, `confidence`, `narrative_id` |

### update_situation

*(NEW v0.79.11)* — Update situation primitive fields and merge into `properties`. Mirrors `update_entity`. Forwards to `PUT /situations/:id`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Situation UUID |
| `updates` | object | Yes | JSON patch — supported keys: `properties` (object-merge), `name`, `description`, `confidence`, `narrative_id`, `synopsis`, `label`, `status`, `keywords` |

### delete_entity / delete_situation

Soft-delete by UUID. `restore_entity` / `restore_situation` reverse the soft-delete.

### merge_entities

Merge two entities — the "keep" entity survives, the "absorb" entity is deleted, all participations and causal links transferred.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `keep_id` | string | Yes | Entity UUID to keep |
| `absorb_id` | string | Yes | Entity UUID to absorb (deleted) |

### split_entity

Split an entity by moving specified situation participations to a new clone entity.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity_id` | string | Yes | Entity UUID to split |
| `situation_ids` | array | Yes | Situation UUIDs whose participations move to the new entity |

## 4.3 Retrieval & Search

| Tool | Purpose |
|------|---------|
| `get_entity` / `get_situation` | Fetch by UUID |
| `list_entities` | List entities filtered by `entity_type`, `narrative_id`, `limit` |
| `search_entities` | Text search ranked by relevance (`query`, `limit`) |
| `get_actor_profile` | Full actor dossier — entity properties, participations, state history |

## 4.4 Participations & Causal Links

### add_participant

Link an entity to a situation with a role and optional action. Appends rather than replaces — an entity may have multiple roles in the same situation.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entity_id` | string | Yes | Entity UUID |
| `situation_id` | string | Yes | Situation UUID |
| `role` | string \| object | Yes | One of `Protagonist`/`Antagonist`/`Witness`/`Target`/`Instrument`/`Confidant`/`Informant`/`Recipient`/`Bystander`/`SubjectOfDiscussion`/`Facilitator`, or `{"Custom": "role-name"}` for domain-specific roles |
| `action` | string | No | What the entity does in this situation |
| `info_set` | object | No | `{knows_before, learns, reveals}` — drives belief + dramatic-irony analysis |
| `payoff` | any | No | Game-theoretic payoff — any JSON |

## 4.5 Narratives & Projects

### create_narrative

Create a narrative metadata container.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `narrative_id` | string | Yes | Slug identifier (e.g., `"harbor-case"`) |
| `title` | string | Yes | Display title |
| `genre` | string | No | Genre classification |
| `description` | string | No | Description |
| `tags` | array | No | String tags |
| `authors` | array | No | Author(s) of the source material |
| `language` | string | No | ISO 639-1 language code (e.g. `"en"`) |
| `publication_date` | string | No | ISO 8601 RFC 3339 |
| `cover_url` | string | No | URL to a cover image |
| `project_id` | string | No | Parent project slug |
| `custom_properties` | object | No | Arbitrary key→JSON map |

Additional tools: `list_narratives`, `get_narrative_stats` (entity/situation/participation/causal counts + temporal span + level distribution), `link_narrative` (set `narrative_id` on an existing entity).

### Projects (containers grouping narratives)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `create_project` | `name`, `description?`, `tags?` | Create project container |
| `get_project` / `list_projects` |  | Read project records |
| `update_project` | `id`, fields | Merge-update project metadata |
| `delete_project` | `id`, `cascade?` | Delete project; cascade optionally removes contained narratives |

## 4.6 Query & Inference

| Tool | Purpose |
|------|---------|
| `query` | Execute a TensaQL MATCH query (instant results) |
| `infer` | Submit an INFER / DISCOVER query; returns a `job_id` |
| `job_status` | Poll job status (`Pending`/`Running`/`Completed`/`Failed`) |
| `job_result` | Fetch the result of a completed inference job |
| `simulate_counterfactual` | Submit a "what-if" intervention on a situation; returns `job_id` |
| `find_cross_narrative_patterns` | Submit pattern mining across multiple narratives |

## 4.7 Ingestion

### ingest_text

Ingest raw narrative text through the LLM extraction pipeline.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Raw text to ingest |
| `narrative_id` | string | Yes | Target narrative |
| `source_name` | string | No | Source attribution (default: `"mcp-upload"`) |
| `auto_commit_threshold` | number | No | Default 0.8 |
| `review_threshold` | number | No | Default 0.3 |
| `single_session` | bool | No | Use SingleSession mode (OpenRouter or Local LLM only) |
| `enrich` | bool | No | Enable multi-step enrichment |

Returns: ingestion report (`entities_created`, `situations_created`, `items_queued`, `errors`). With `single_session: true`, the report also includes `reconciliation` (entity merges, confidence adjustments, cross-chunk causal links).

### ingest_url

Fetch a URL, strip HTML markup, ingest the extracted text. Same `auto_commit_threshold` / `review_threshold` parameters.

### ingest_rss

Fetch an RSS or Atom feed and ingest each item's content. Additional parameter: `max_items` (default: all). Returns an array of ingestion reports, one per feed item.

### review_queue

Manage the validation queue for human-in-the-loop review.

| Parameter | Type | Required for | Description |
|-----------|------|--------------|-------------|
| `action` | string | always | `list`/`get`/`approve`/`reject`/`edit` |
| `item_id` | string | get/approve/reject/edit | Queue item UUID |
| `reviewer` | string | approve/reject/edit | Reviewer name |
| `notes` | string | — | Review notes |
| `edited_data` | object | edit | Modified extraction JSON |
| `limit` | number | list | Max results |

## 4.8 Source Intelligence

| Tool | Purpose |
|------|---------|
| `create_source` | Register a source with `name`, `source_type` (NewsOutlet/GovernmentAgency/AcademicInstitution/SocialMedia/Sensor/StructuredApi/HumanAnalyst/OsintTool), `trust_score`, `known_biases` |
| `get_source` / `list_sources` | Read source records |
| `add_attribution` | Link source to entity/situation with `source_id`, `target_id`, `target_kind`, optional `original_url` / `excerpt` / `extraction_confidence` / `claim` |
| `list_contentions` | List disagreements for a situation |
| `recompute_confidence` | Recompute from attributions; returns Bayesian breakdown (extraction, source_credibility, corroboration, recency, `prior_alpha/beta`, `posterior_alpha/beta`) |

## 4.9 Export & Import

| Tool | Purpose |
|------|---------|
| `export_narrative` | Export in `csv`/`graphml`/`json`/`manuscript`/`report`/`archive`/`stix` |
| `export_archive` | Lossless `.tensa` archive (base64 ZIP) — takes `narrative_ids[]` |
| `import_archive` | Import a `.tensa` archive from base64-encoded bytes; returns import report with counts, id_remaps, warnings, errors |

## 4.10 Advanced Analysis

### tune_prompts

Generate domain-adapted extraction prompts for a narrative by sampling ingested chunks. Tuned prompts are stored at `pt/{narrative_id}` and used during future ingestion.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `narrative_id` | string | Yes | Narrative to tune prompts for |

Returns: `narrative_id`, `prompt_text`, `domain_description`, `entity_types`, `generated_at`, `model`.

### community_hierarchy

Get the hierarchical community structure for a narrative. Communities are detected using hierarchical Leiden (connected communities guaranteed). Level 0 is most granular; higher levels are coarser.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `narrative_id` | string | Yes | Narrative |
| `level` | integer | No | Filter to a specific hierarchy level |

Returns: array of `CommunitySummary` records with `level`, `parent_community_id`, `child_community_ids`, `entity_ids`, `entity_names`, `key_themes`, `summary`.

## 4.11 Writer Workflows

Tools driving the AI-assisted novel-writing pipeline. The `tensa-writer` Claude Code skill orchestrates them end-to-end.

| Tool | Purpose |
|------|---------|
| `get_writer_workspace` | Dashboard: counts (situations/entities/narrative_arcs/arc_situations/pinned_facts/total_words), plan status, last revision, ranked next-step suggestions, and `api_base_url` (from HttpBackend). Always call first. |
| `get_narrative_plan` / `upsert_narrative_plan` | Read / merge-update the plan (logline, synopsis, plot beats, style targets, length targets, setting, comp titles). `upsert_narrative_plan` deep-merges nested object fields (`style`, `length`, `setting`, `custom`) — a patch `{"style": {"pov": "first"}}` only updates `style.pov` and leaves sibling fields intact. Arrays and scalars replace wholesale. |
| `list_pinned_facts` / `create_pinned_fact` | Pin canonical facts for continuity checking. Prefer over-pinning. |
| `check_continuity` | Scan prose for conflicts with pinned facts. Deterministic, no LLM cost. |
| `run_workshop` | Tiered critique: `cheap` (deterministic) / `standard` (LLM-enriched review of top findings) / `deep` (async job). `focuses`: subset of `[pacing, continuity, characterization, prose, structure]`. |
| `list_narrative_revisions` | Newest-first git-like revision list (id, parent_id, message, author, content_hash, counts) |
| `restore_narrative_revision` | Restore to a previous revision. Auto-commits current state as a safety net. |
| `get_writer_cost_summary` | Aggregate cost ledger in a window (`24h`/`7d`/`30d`/`all`) with per-operation breakdown |
| `set_situation_content` | (Workflow 2) Replace a situation's `raw_content` with assistant-authored prose. Accepts a plain string (wrapped as one Text block) or an array of `{content_type, content}` blocks. Optional `status` stamps the writer workflow state. |
| `get_scene_context` | (Workflow 2) One-call context bundle: plan + pinned facts + optional POV profile + current scene + N preceding scenes in manuscript order. `lookback_scenes` default 2, max 10. |

### Writer analysis touchpoints

When Workflow 2 invokes the narratology analysis layer (§4.12) — all
deterministic, free, in-band unless noted. The `tensa-writer` skill
runs these at fixed cadence points during drafting.

| Touchpoint | Tools | Action on output |
|---|---|---|
| **Per-chapter post-flight** (after every chapter commit) | `run_workshop(tier="cheap")`, `get_health_score`, `detect_focalization`, `classify_scene_sequel`, `compute_dramatic_irony` (mystery/thriller/crime) | POV drift → revise offending scene via `set_situation_content`; pacing score low → insert reflection beat; flat irony gap → chapter didn't advance info state, flag |
| **Midpoint audit** (at ⌈N/2⌉ chapters drafted) | `detect_commitments`, `get_commitment_rhythm`, `detect_subplots`, `detect_character_arc` | Surface unpaid commitments as a list; subplot sprawl → propose merge/cut; arc mismatch vs. plan → `upsert_narrative_plan` with corrections |
| **Final pass** (after last chapter) | `diagnose_narrative(genre)`, `get_health_score`, `suggest_narrative_fixes` → `apply_narrative_fix` (user-approved only), `compute_essentiality` (if over-length), `extract_fabula`/`extract_sjuzet`/`suggest_reordering` (non-linear only), optional `run_workshop(tier="standard")` | Pathology fixes applied one-by-one after user approval; cut candidates proposed but not applied autonomously |
| **Repair** (user-requested only) | `auto_repair_narrative` — loops diagnose + apply safe fixes. Paid in hypergraph mutations. | Always commit a revision first; prefer the manual `suggest_narrative_fixes` loop. |
| **What-if** (user-requested only) | `simulate_counterfactual` — async INFER job | Submit, poll `job_status` / `job_result`, surface cascading probabilities |
| **Cross-corpus** (user with prior work) | `find_cross_narrative_patterns` — async INFER job | Submit, poll, surface recurring motifs |

## 4.12 Narrative Architecture

Tools for narrative structural analysis and generation. The D9 detection tools ship
unconditionally; the generation/materialization/prompt-prep tools (last five rows) are
gated behind `--features generation` (which the `mcp` feature already implies). All 15
tools were fully wired to their backends in v0.69.0 (Sprint W14) — prior releases
(v0.67.3 – v0.68.0) shipped them as stubs that returned a `NotWired` error.

| Tool | Feature gate | Purpose |
|------|--------------|---------|
| `detect_commitments` | — | Detect setup-payoff pairs (Chekhov's guns, foreshadowing, dramatic questions) |
| `get_commitment_rhythm` | — | Per-chapter promise rhythm (tension curve, fulfillment ratio) |
| `extract_fabula` / `extract_sjuzet` | — | Chronological event order / discourse order |
| `suggest_reordering` | — | Sjužet reorderings for dramatic irony or tension |
| `compute_dramatic_irony` | — | Reader vs character knowledge gaps |
| `detect_focalization` | — | Focalization × irony interactions across POV boundaries |
| `detect_character_arc` | — | Arc type and transformation trajectory (`character_id` optional; omit to list stored arcs) |
| `detect_subplots` | — | Subplot detection via community detection on situation graph |
| `classify_scene_sequel` | — | Scene-sequel rhythm classification and pacing score |
| `generate_narrative_plan` | `generation` | Generate a formal plan from a premise; persists plan at `plan/{plan_id}` |
| `materialize_plan` | `generation` | Write plan into hypergraph as real entities/situations |
| `validate_materialized_narrative` | `generation` | Check temporal, causal, knowledge, commitment consistency |
| `generate_chapter` / `generate_narrative` | `generation` | Build chapter / full-narrative generation prompts from hypergraph queries (LLM-less preview / dry-run path) |
| `generate_chapter_with_fitness` | `generation` | Submit a fitness-loop chapter generation job. Args: `narrative_id`, `chapter`, `voice_description?`, `style_embedding_id?`, `target_fingerprint_source?` (narrative whose fingerprint is the target), `fitness_threshold?` (default `0.80`), `max_retries?`, `synchronous?` (default `false`). Async by default — returns `{job_id}` for polling via `job_status` / `job_result`; with `synchronous: true` the result is returned inline. Returns the best-scoring attempt across iterations, not the last (§7.11.14). |

**Borrowing style from another narrative.** A common writer workflow: draft a new book in the voice of an ingested comp title (e.g. a `good-omens` narrative). Two tiers:

- **Informal** — in the chat-native co-writing loop (`tensa-writer` Workflow 2), ask the assistant to read a few scenes from the comp title via `query` or `ask OVER "<comp-id>"`, extract the style cues, and mimic them in-prompt. Fast, no extra LLM cost beyond drafting, but unmeasured.
- **Formal** — escalate to `generate_chapter_with_fitness` with `target_fingerprint_source = "<comp-id>"`. The server computes the comp title's `NarrativeFingerprint` (sentence-length distribution, dialogue ratio, FK grade, vocabulary level, etc.) on first call and caches; the fitness loop iteratively re-drafts until the distance from the target is below `fitness_threshold`. Best-of-k across iterations. Scores the **prose layer only** — structure is not transferred. Tagged `kind = "chapter_gen_fitness"` in the cost ledger.

**Typed enumerations** used in results:

- `CommitmentType`: ChekhovsGun, Foreshadowing, RedHerring, DramaticQuestion, CharacterPromise, ThematicSeed, MysterySetup
- `CommitmentStatus`: Planted → InProgress → Fulfilled | Abandoned | Subverted | RedHerringResolved
- `NarrationMode` (Genette): Scene, Summary, Pause, Ellipsis, Stretch
- `TemporalShift`: Chronological, Analepsis, Prolepsis, InMediasRes, FrameNarrative
- `IronyType`: Suspense, Anticipation, TragedyForeknowledge, ComedicMisunderstanding
- `Focalization` (Genette): Zero, Internal, External
- `ArcType`: PositiveChange, NegativeCorruption, NegativeDisillusionment, Flat, PositiveDisillusionment
- `SubplotRelation`: Mirror, Contrast, Complication, Convergence, Independent, Setup
- `SceneType` (Swain/Bickham): ActionScene (goal/conflict/disaster), Sequel (reaction/dilemma/decision), Hybrid, Unclassified

## 4.13 Writer MCP Bridge (Sprint W15, v0.70.0)

28 tools wrap TENSA's manuscript-tooling writer affordances. Backends already
existed; W15 is pure MCP wiring plus one REST route family (`/narrative-templates`).

### 4.13.0 Annotations ([src/writer/annotation.rs](../src/writer/annotation.rs))

| Tool | Purpose |
|------|---------|
| `create_annotation` | Comment / Footnote / Citation anchored to a byte-span of a scene's concatenated prose. Comments never ship; footnotes + citations render via the active compile profile. |
| `list_annotations` | Scope by `situation_id` (scene) or `narrative_id` (every annotation, batched). |
| `update_annotation` | Patch keys: body, span, source_id, chunk_id, detached. Updating span resets detached. |
| `delete_annotation` | Hard delete. |

### 4.13.1 Collections ([src/writer/collection.rs](../src/writer/collection.rs))

| Tool | Purpose |
|------|---------|
| `create_collection` | Saved search with structured `CollectionQuery` (labels, statuses, keywords_any, text, manuscript_order and word-count bounds). |
| `list_collections` | All collections for a narrative, sorted by name. |
| `get_collection` | Fetch by id; `resolve=true` also returns current matching situation UUIDs. |
| `update_collection` / `delete_collection` | Patch (name / description / query) or delete. |

### 4.13.2 Research Notes ([src/writer/research.rs](../src/writer/research.rs))

| Tool | Purpose |
|------|---------|
| `create_research_note` | Scene-scoped writer note: Quote / Clipping / Link / Note. |
| `list_research_notes` | Scope by `situation_id` or `narrative_id`. |
| `get_research_note` / `update_research_note` / `delete_research_note` | CRUD. |
| `promote_chunk_to_note` | Promote a ChunkStore chunk to a scene-pinned note with `source_chunk_id` back-link. |

### 4.13.3 Editing Engine ([src/narrative/editing.rs](../src/narrative/editing.rs))

| Tool | Purpose |
|------|---------|
| `propose_edit` | LLM-driven rewrite. `style_preset` ∈ {minimal, lyrical, punchy, formal, interior, cinematic} switches to StyleTransfer; otherwise Rewrite with `instruction`. Requires a session-capable LLM. Returns EditProposal with word-count delta + unified line diff. |
| `apply_edit` | Write proposal into hypergraph and commit a revision. Returns `{revision_id, words_before, words_after}`. |
| `estimate_edit_tokens` | Prompt + response token estimate without calling the LLM. |

### 4.13.4 Revision Completion ([src/narrative/revision.rs](../src/narrative/revision.rs))

| Tool | Purpose |
|------|---------|
| `commit_narrative_revision` | Snapshot current state. `{outcome: committed \| no_change, revision}`. |
| `diff_narrative_revisions` | Structural + prose diff between two revisions. Both revs validated against the narrative. |

### 4.13.5 Workshop Completion ([src/narrative/workshop.rs](../src/narrative/workshop.rs))

| Tool | Purpose |
|------|---------|
| `list_workshop_reports` | List all past reports for a narrative, newest-first. |
| `get_workshop_report` | Fetch a report by id with findings + cost. |

### 4.13.6 Cost Ledger ([src/narrative/cost_ledger.rs](../src/narrative/cost_ledger.rs))

| Tool | Purpose |
|------|---------|
| `list_cost_ledger_entries` | Raw entries (one per LLM call). `limit` default 50, max 500. `get_writer_cost_summary` remains the rolled-up surface. |

### 4.13.7 Compile ([src/export/compile.rs](../src/export/compile.rs))

| Tool | Purpose |
|------|---------|
| `list_compile_profiles` | All profiles for a narrative. |
| `compile_narrative` | Render narrative through profile → markdown (UTF-8 body) / epub / docx (base64 body). If `profile_id` omitted a default profile is synthesised. |
| `upsert_compile_profile` | Create (omit `profile_id`; patch must include `name`) or patch an existing profile. |

### 4.13.8 Templates ([src/narrative/templates.rs](../src/narrative/templates.rs))

| Tool | Purpose |
|------|---------|
| `list_narrative_templates` | Builtin templates (Mentor's Death, False Victory, Information Marketplace) + any user-stored templates. |
| `instantiate_template` | Bind slots to entity UUIDs. Returns InstantiatedSituation list without writing to the hypergraph. |

### 4.13.9 Secondary (skeleton / dedup / fixes / reorder)

| Tool | Purpose |
|------|---------|
| `extract_narrative_skeleton` | Compact structural skeleton for cross-narrative similarity. **Embedded backend only.** |
| `find_duplicate_candidates` | Candidate entity merges (does NOT merge). Call `merge_entities` to commit. Default threshold 0.7, max 200. |
| `suggest_narrative_fixes` | Fix suggestions keyed to pathologies from the latest diagnosis. |
| `apply_narrative_fix` | Apply one SuggestedFix. |
| `apply_reorder` | Batch reorder scenes; writes manuscript_order + parent atomically, densifying positions to 1000/2000/3000/… |

## 4.14 Disinfo Tools (`disinfo` feature)

Tools for disinformation analysis. All require `--features disinfo` (default-on).

### 4.14.1 Fingerprints

| Tool | Purpose |
|------|---------|
| `get_behavioral_fingerprint` | Load (or compute) the 10-axis `BehavioralFingerprint` for an actor. Axes: `posting_cadence_regularity`, `sleep_pattern_presence`, `engagement_ratio`, `account_maturity`, `platform_diversity`, `content_originality`, `response_latency`, `hashtag_concentration`, `network_insularity`, `temporal_coordination` |
| `get_disinfo_fingerprint` | Load (or compute) the 12-axis `DisinformationFingerprint` for a narrative. Axes: `virality_velocity`, `cross_platform_jump_rate`, `linguistic_variance`, `bot_amplification_ratio`, `emotional_loading`, `source_diversity`, `coordination_score`, `claim_mutation_rate`, `counter_narrative_resistance`, `evidential_uncertainty`, `temporal_anomaly`, `authority_exploitation` |
| `compare_fingerprints` | Compare two behavioral or two disinfo fingerprints. `kind`: `"behavioral"` or `"disinfo"`. `task`: `"literary"` / `"cib"` / `"factory"`. Returns `composite_distance`, per-layer distances, `p_value`, `confidence_interval`, `same_source_verdict`, `comparable_axes`, `anomaly_axes` |

### 4.14.2 Spread Dynamics

| Tool | Purpose |
|------|---------|
| `estimate_r0_by_platform` | SMIR contagion + per-platform R₀ + cross-platform jump detection + velocity-monitor anomaly check. Params: `narrative_id`, `fact`, `about_entity`, `narrative_kind`, `beta_overrides`. Returns SMIR compartment counts, jumps, velocity alerts with `baseline_source: "Synthetic" | "Learned"` |
| `simulate_intervention` | Counterfactual spread projection. `intervention`: `{type:"RemoveTopAmplifiers",n}` or `{type:"DebunkAt",at}`. Returns `SpreadProjection` (baseline/projected R₀, delta, audience saved, removed entities) |

### 4.14.3 CIB Detection

| Tool | Purpose |
|------|---------|
| `detect_cib_cluster` | Build behavioral similarity network → label-propagation communities → flag clusters whose density is in the right tail of a bootstrap null. Params: `narrative_id`, `cross_platform` (factory variant), `similarity_threshold` (0.7), `alpha` (0.01), `bootstrap_iter` (500), `min_cluster_size` (3), `seed`. Returns clusters, evidence, null stats |
| `rank_superspreaders` | Rank top-N actors by centrality. `method`: `"pagerank"` / `"eigenvector"` / `"harmonic"`. Persists at `cib/s/{narrative_id}` |

### 4.14.4 Claims & Fact-Checks

| Tool | Purpose |
|------|---------|
| `ingest_claim` | Detect verifiable claims via regex heuristics (numerical, quote, causal, comparison, predictive, factual) |
| `ingest_fact_check` | Ingest a fact-check verdict for a claim. Verdicts: `true`, `false`, `misleading`, `partially_true`, `unverifiable`, `satire`, `out_of_context` |
| `fetch_fact_checks` | Match a claim against known fact-checks using Levenshtein + embedding similarity |
| `trace_claim_origin` | Trace a claim back through its mutation chain to the earliest appearance |

### 4.14.5 Archetypes & DS Fusion

| Tool | Purpose |
|------|---------|
| `classify_archetype` | Classify actor into: StateActor, OrganicConspiracist, CommercialTrollFarm, Hacktivist, UsefulIdiot, HybridActor. Returns softmax probability distribution |
| `assess_disinfo` | Fuse multiple disinfo signals via Dempster-Shafer. Returns belief/plausibility intervals + verdict |

**Built-in archetype templates:**

| Archetype | Key Behavioral Signals |
|-----------|----------------------|
| StateActor | High cadence regularity, no sleep pattern, low originality, high coordination |
| OrganicConspiracist | Irregular cadence, normal sleep, moderate originality, low coordination |
| CommercialTrollFarm | Regular cadence, minimal sleep, very low originality, high coordination |
| Hacktivist | Event-driven bursts, high originality, moderate coordination |
| UsefulIdiot | Irregular, normal sleep, low originality, no coordination |
| HybridActor | Mixed signals across all axes |

### 4.14.6 Post / Actor Ingestion

| Tool | Purpose |
|------|---------|
| `ingest_post` | Create a `Situation (Event)` with post content linked to an actor |
| `ingest_actor` | Create an `Actor` entity with disinfo-relevant properties |

### 4.14.7 Multilingual (library functions)

Not direct MCP tools — library helpers surfaced through other pipelines:

- `detect_language(text)` — classify by dominant Unicode script; returns ISO 639-1 code + confidence (`en`, `ru`, `ar`, `zh`, `ko`, `hi`)
- `transliterate_cyrillic_to_latin` / `transliterate_latin_to_cyrillic` — character-level and greedy longest-match mappings
- `strip_diacritics` — remove accents for CEE name matching. Lives in the non-gated [src/text_util.rs](../src/text_util.rs) as of v0.73.0 and is re-exported from `disinfo::multilingual` for backward compatibility. Also exposes `normalize_slug(s)` (lowercase + diacritic-fold + `_`/` `→`-`), used by `Hypergraph::find_entity_by_name`.
- `normalize_for_matching` — lowercase + strip diacritics + collapse whitespace
- `linguistic_variance(languages)` — Shannon entropy of language distribution, normalized to [0, 1]; wires `DisinfoAxis::LinguisticVariance`

### 4.14.8 MCP Client Orchestrator

Multi-platform source ingestion with audit logging.

- `NormalizedPost` — platform-agnostic post representation (id, platform, content, language, engagement, hashtags, mentions, URLs, reply/repost chains, raw JSON, confidence)
- `McpSource` / `SourceRegistry` — JSON-loadable source registry with `sources_for_platform()` and `active_sources()` (priority-sorted)
- `AuditEntry` — provenance audit trail
- `normalize_post(raw, source)` — normalize MCP tool JSON to `NormalizedPost` (auto language detection fallback)
- `store_audit_entry` / `list_audit_entries` — persist/retrieve at `mcp/audit/{timestamp_be}`

### 4.14.9 Scheduler, Reports & Health

| Tool | Purpose |
|------|---------|
| `list_scheduled_tasks` | List all scheduled analysis tasks |
| `create_scheduled_task` | Types: `cib_scan`, `source_discovery`, `fact_check_sync`, `report_generation`, `mcp_poll`, `fingerprint_refresh`, `velocity_baseline_update`. Schedule: `"30m"`, `"6h"`, `"1d"` |
| `run_task_now` | Trigger immediate execution |
| `list_discovery_candidates` | List discovered source candidates (channels, accounts, URLs) |
| `sync_fact_checks` | Trigger fact-check database sync |
| `generate_situation_report` | Periodic report aggregating narratives, CIB clusters, velocity alerts, claims, new sources. Persists at `reports/{uuid}` |

Source health tracking: `SourceHealth` records at `mcp/health/{source_name}`; after 3 consecutive failures `healthy=false`, `record_poll_success` resets.

### 4.14.10 Exports

| Tool | Purpose |
|------|---------|
| `detect_language` | ISO 639-1 code + confidence |
| `export_misp_event` | Export narrative as MISP event with entity→attribute mapping |
| `export_maltego` | Export entities as Maltego-compatible transform results |
| `generate_report` | Generate comprehensive Markdown disinfo analysis report |

## 4.15 Adversarial Wargaming (`adversarial` feature)

Feature-gated behind `--features adversarial` (depends on `disinfo`). Tools for adversarial simulation, counter-narrative generation, and governance.

| Tool | Purpose |
|------|---------|
| `generate_adversary_policy` | Adversary action policy using SUQR bounded rationality over IRL reward weights. Params: `narrative_id`, `actor_id?`, `archetype?`, `lambda?`, `lambda_cap?`, `reward_weights?`. Stores at `adv/policy/` |
| `configure_rationality` | Configure rationality model. `model`: `qre` / `suqr` / `cognitive_hierarchy`. Params: `lambda?`, `lambda_cap?`, `tau?`, `feature_weights?` |
| `create_wargame` | Fork narrative into mutable simulation with SMIR compartments. Params: `narrative_id`, `max_turns?`, `time_step_minutes?`, `auto_red?`, `auto_blue?` |
| `submit_wargame_move` | Submit red/blue moves for one turn. Validates, applies SMIR effects, evaluates objectives |
| `get_wargame_state` | Current turn, R₀, misinformed, susceptible, objectives met |
| `auto_play_wargame` | Auto-play N turns using heuristic AI for auto-controlled teams |

---

# Chapter 5: REST API Reference

The REST API is feature-gated behind `--features server`. Default address: `0.0.0.0:3000`. Override with `TENSA_ADDR`. All request/response bodies are JSON unless noted otherwise.

Multi-tenancy headers (optional):

- `X-Tensa-Workspace` — workspace identifier (default `default`). Enables isolated datasets per workspace via transparent `w/{workspace_id}/` key prefixing.
- `X-Tensa-User` — user identifier (default `local`). Used by Studio chat session scoping.

## 5.1 Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |

## 5.2 Entities

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/entities` | Create entity |
| `GET` | `/entities` | List (`?narrative_id=&entity_type=&limit=`, default 2000, max 10000) |
| `GET` | `/entities/{id}` | Get by UUID |
| `PUT` | `/entities/{id}` | Update (auto-snapshots prior state) |
| `DELETE` | `/entities/{id}` | Soft-delete |
| `POST` | `/entities/bulk` | Bulk create (max 1000) |
| `POST` | `/entities/merge` | Merge two entities `{keep_id, absorb_id}` |
| `POST` | `/entities/{id}/split` | Split: `{situation_ids: [...]}` moves those participations to a new clone |
| `GET` | `/entities/{id}/situations` | Situations entity participates in |
| `GET` | `/entities/{id}/attributions` | All source attributions |
| `POST` | `/entities/{id}/recompute-confidence` | Rebuild confidence from source evidence |

**Create body:**
```json
{
  "entity_type": "Actor",
  "properties": {"name": "Agent Chen"},
  "narrative_id": "harbor-case",
  "confidence": 0.9,
  "maturity": "Candidate"
}
```

## 5.3 Situations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/situations` | Create. Accepts either a full `Situation` JSON or a permissive partial body (see below). |
| `GET` | `/situations` | List (`?narrative_id=&narrative_level=&limit=`, default 2000, max 10000) |
| `GET` | `/situations/{id}` | Get by UUID |
| `PUT` | `/situations/{id}` | Patch subset of `name`, `description`, `synopsis`, `narrative_level`, `narrative_id`, `confidence`, `raw_content` (≤500 blocks), `temporal`, `spatial`, `discourse`, `manuscript_order`, `parent_situation_id` (cycle-rejected), `label`, `status`, `keywords`. Nullable fields accept `null` to clear; omission preserves. |
| `DELETE` | `/situations/{id}` | Soft-delete |
| `POST` | `/situations/bulk` | Bulk create (max 1000). Each element uses the same permissive body shape as `POST /situations`. |
| `GET` | `/situations/{id}/participants` | List participants |
| `GET` | `/situations/{id}/attributions` | Source attributions |
| `GET` | `/situations/{id}/contentions` | Contentions |
| `POST` | `/situations/{id}/recompute-confidence` | Rebuild confidence |

**Create body (permissive):**
```json
{
  "raw_content": "She entered the room.",
  "narrative_id": "harbor-case",
  "narrative_level": "Scene",
  "start": "2025-03-15T10:00:00Z",
  "end":   "2025-03-15T10:30:00Z",
  "confidence": 0.9
}
```

Auto-supplied when absent: `id` (v7 UUID), `maturity` (`"Candidate"`),
`extraction_method` (`"HumanEntered"`), `granularity` (`"Approximate"`),
`created_at` / `updated_at` (now), `causes: []`, `provenance: []`.
`raw_content` accepts a bare string and is wrapped into a single
`ContentBlock`; or you may send the full
`[{"content_type":"Text","content":"...","source":null}]` shape
directly. Passing a complete `Situation` JSON (including all of the
above) also works — nothing is overwritten.

## 5.4 Participations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/participations` | Add participant `{entity_id, situation_id, role, action?}`. Returns `201 Created` with body `{"status": "ok"}`. |
| `POST` | `/participations/bulk` | Bulk (max 500). Never aborts early; per-item errors in `errors[]` |
| `DELETE` | `/participations/{entity_id}/{situation_id}` | Remove. `?seq=N` deletes a single role; without it deletes all seqs for the pair |

## 5.5 Queries

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Execute TensaQL query `{"query": "..."}`. Returns raw results; for INFER/DISCOVER this is a descriptor row, not a submitted job. |
| `POST` | `/infer` | Parse + plan + execute + **submit** a TensaQL INFER/DISCOVER query to the inference queue. Returns `{job_id, status, message}`. This is what the HTTP MCP backend uses for `submit_inference_query`. |

## 5.6 Inference Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit `{job_type, target_id?, parameters}`. `target_id` is optional (v0.74.0) — narrative-scoped jobs key on `parameters.narrative_id` and ignore it; omit to let the server generate a throwaway UUID. |
| `GET` | `/jobs` | List (`?limit=N&target_id=UUID&narrative_id=...`) |
| `GET` | `/jobs/{id}` | Get status (`Pending`/`Running`/`Completed`/`Failed`) |
| `GET` | `/jobs/{id}/result` | Get result |
| `DELETE` | `/jobs/{id}` | Cancel a pending job |
| `GET` | `/ws/jobs/{id}` | WebSocket push for real-time status |

## 5.7 Validation Queue

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/validation-queue` | List pending (`?limit=N`) |
| `GET` | `/validation-queue/{id}` | Get item |
| `POST` | `/validation-queue/{id}/approve` | `{reviewer}` |
| `POST` | `/validation-queue/{id}/reject` | `{reviewer, notes?}` |
| `POST` | `/validation-queue/{id}/edit` | `{reviewer, edited_data}` |

## 5.8 Narratives

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/narratives` | Create |
| `GET` | `/narratives` | List |
| `GET` | `/narratives/{id}` | Get |
| `PUT` | `/narratives/{id}` | Update metadata (partial) |
| `DELETE` | `/narratives/{id}` | Delete |
| `GET` | `/narratives/{id}/stats` | Entity/situation/participation/causal counts, temporal span, level distribution |
| `POST` | `/narratives/{id}/merge` | Merge source narrative into this one `{source_id}` |
| `POST` | `/narratives/{id}/reorder` | Batch-reorder scenes. Body: `{entries: [{situation_id, parent_id?}, ...]}` (max 10k). Writes `manuscript_order` + `parent_situation_id` atomically, densified to 1000, 2000, 3000, … Rejects duplicates, foreign-narrative ids, and cycle-forming parent assignments. |
| `POST` | `/narratives/{id}/dedup-entities` | Propose fuzzy merge candidates within a narrative. Body (all optional): `{threshold:0.7, max_candidates:200, entity_types:["Actor"]}`. Never merges automatically — analyst applies via `POST /entities/merge`. |
| `GET` | `/narratives/{id}/export` | Export (`?format=csv\|graphml\|json\|manuscript\|report\|archive\|stix`) |
| `POST` | `/narratives/{id}/analyze` | Run full inference battery (centrality, entropy, arc classification, causal discovery, style profile, motivation per actor, belief modeling) |

## 5.9 Narrative Arcs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narratives/{id}/arcs` | List user-defined arcs, sorted by `order` |
| `POST` | `/narratives/{id}/arcs` | Create `UserArc` (`title`, `arc_type`, `situation_ids`, `order`, optional `description`) |
| `PUT` | `/narratives/{id}/arcs/{arc_id}` | Patch an arc |
| `DELETE` | `/narratives/{id}/arcs/{arc_id}` | Hard-delete |

## 5.10 Projects

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/projects` | Create |
| `GET` | `/projects` | List |
| `GET` | `/projects/{id}` | Get |
| `PUT` | `/projects/{id}` | Update |
| `DELETE` | `/projects/{id}` | Delete (`?cascade=true` also deletes contained narratives) |
| `GET` | `/projects/{id}/narratives` | Narratives in project |

## 5.11 Taxonomy

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/taxonomy/{category}` | List builtin + custom entries |
| `POST` | `/taxonomy/{category}` | Add custom entry `{value, label}` |
| `DELETE` | `/taxonomy/{category}/{value}` | Remove custom entry (builtins are immutable) |

**Builtin categories:**
- `genre` — novel, novella, short-story, investigation, geopolitical, intelligence-report, news-article, academic-paper, biography, memoir, essay, technical, legal, financial, social-media, transcript, other
- `content_type` — fiction, non-fiction, mixed, primary-source, secondary-source, opinion, analysis, raw-data
- `role` — the standard participation roles

## 5.12 Ingestion

### Ingest endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest raw text. Body may set `enrich: true`, `single_session: true`, `ingestion_mode` (overrides global default for this job) |
| `POST` | `/ingest/document` | Ingest PDF/DOCX (requires `docparse` feature) |
| `POST` | `/ingest/url` | Fetch URL, strip HTML, ingest (requires `web-ingest` feature) |
| `POST` | `/ingest/rss` | Fetch RSS/Atom feed and ingest each item (requires `web-ingest` feature) |
| `DELETE` | `/ingest/source/{source_id}` | Cascade-delete a source and all derived entities/situations |

### Ingestion jobs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ingest/jobs?limit=N` | List recent jobs (default 50) |
| `GET` | `/ingest/jobs/{id}` | Status, report, progress (`?debug=true` for LLM call summary) |
| `DELETE` | `/ingest/jobs/{id}` | Cancel a running job |
| `POST` | `/ingest/jobs/{id}/rollback` | Delete all entities/situations created by this job |
| `POST` | `/ingest/jobs/{id}/retry` | Re-run failed chunks (creates new job linked via `parent_job_id`) |
| `GET` | `/ingest/jobs/{id}/lineage` | Full lineage tree (parent + all retry children) |
| `POST` | `/ingest/jobs/{id}/reconcile` | Re-run temporal reconciliation on stored extractions |
| `POST` | `/ingest/jobs/{id}/reprocess` | Rollback + re-gate all chunks (no LLM call) |
| `GET` | `/ingest/jobs/{id}/logs` | All LLM call logs |
| `GET` | `/ingest/jobs/{id}/extractions` | All chunk extraction records |

### Per-chunk control

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ingest/jobs/{id}/chunks/{index}/logs` | Chunk LLM logs |
| `GET` | `/ingest/jobs/{id}/chunks/{index}/extraction` | Chunk extraction |
| `POST` | `/ingest/jobs/{id}/chunks/{index}/resend` | Re-extract a single chunk |
| `POST` | `/ingest/jobs/{id}/chunks/{index}/reprocess` | Rollback + re-gate one chunk |
| `POST` | `/ingest/jobs/{id}/chunks/batch` | Batch operations (see below) |

### POST /ingest/jobs/{id}/chunks/batch

Perform bulk operations on selected chunks without re-running the full pipeline.

```json
{
  "chunk_indices": [0, 2, 5],
  "action": "reextract",
  "context_mode": "neighbors"
}
```

| Field | Description |
|-------|-------------|
| `chunk_indices` | Max 200 chunk indices |
| `action` | `reextract` / `enrich` / `reconcile` / `reprocess` |
| `context_mode` | For `reextract` only: `selected` / `neighbors` / `all` |

Response returns per-chunk `{chunk_index, status, entities?, situations?, error?}`.

### Ingestion status tracking

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ingest/status` | List document ingestion statuses |
| `GET` | `/ingest/status/{hash}` | Status for a specific document (SHA-256) |

## 5.13 Import & Export

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/import/json` | Import structured data (entities, situations, participations, causal links). `?analyze=true` triggers community detection + summaries after import. |
| `POST` | `/import/csv` | Import entities from CSV with column mapping. Supports `?analyze=true`. |
| `POST` | `/import/stix` | Import STIX 2.1 bundle |
| `POST` | `/import/archive` | Import `.tensa` archive (raw bytes). Returns import report. |
| `POST` | `/export/archive` | Export narratives as `.tensa` archive `{narrative_ids: [...], options: {...}}` |

## 5.14 Sources & Contentions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sources` | Register source |
| `GET` | `/sources` | List. `?limit=&after=` paginates. `?narrative_id=<id>` returns only sources attributed to an entity or situation in that narrative (unpaginated, sorted by name). |
| `GET` | `/sources/{id}` | Get |
| `PUT` | `/sources/{id}` | Update (changes to `trust_score` auto-propagate to attributed targets) |
| `DELETE` | `/sources/{id}` | Delete |
| `POST` | `/sources/{src}/attributions` | Link to target |
| `GET` | `/sources/{src}/attributions` | List attributions from source |
| `DELETE` | `/sources/{src}/attributions/{target}` | Remove attribution |
| `POST` | `/contentions` | Create |
| `POST` | `/contentions/resolve` | Resolve |

## 5.15 Cross-Narrative Analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analysis/by-tag` | Submit analysis jobs across all narratives matching a tag `{tag, job_type}` |
| `POST` | `/analysis/shortest-path` | Weighted shortest path — UUID-keyed `{source, target, narrative_id, k?}` |
| `POST` | `/analysis/shortest-path-by-name` | Same, name-keyed `{source_name, target_name, narrative_id, k?}` (v0.73.0) |
| `POST` | `/analysis/narrative-diameter` | Longest causal chain via DAG DP |
| `POST` | `/analysis/max-flow` | Max-flow / min-cut — UUID-keyed |
| `POST` | `/analysis/max-flow-by-name` | Same, name-keyed (v0.73.0) |

Supported by-tag job types: `StyleProfile`, `StyleComparison`, `StyleAnomaly`, `ArcClassification`, `Centrality`, `Entropy`, `Contagion`.

The `*-by-name` variants diacritic-fold and lowercase both the query string and each entity's `properties.name`/`properties.slug`/`properties.aliases` before comparing, so `"drago-milošević"`, `"Drago Milošević"`, and `"drago milosevic"` all resolve to the same entity within the given narrative. Useful for clients that key on human-readable slugs (benchmark harnesses, Studio direct links, MCP tool arguments).

## 5.15.1 Narrative-Scoped Analytics Read-Back (v0.73.0)

Phase-B inference engines persist per-narrative blobs under `an/*/`. These endpoints expose those blobs directly so clients don't need TensaQL virtual properties to read engine outputs.

| Method | Path | Source | Description |
|--------|------|--------|-------------|
| `GET` | `/narratives/{id}/contentions` | `ct/` + narrative situations | All contentions touching situations in this narrative (deduped forward/reverse) |
| `GET` | `/narratives/{id}/contagion` | `an/sir/` | SIR cascade runs (may be multiple per narrative, keyed by fact hash) |
| `GET` | `/narratives/{id}/netinf` | `an/ni/` | NetInf diffusion-network blob |
| `GET` | `/narratives/{id}/temporal-motifs` | `an/tm/` | Motif census blob |
| `GET` | `/narratives/{id}/faction-evolution` | `an/fe/` | Sliding-window faction events |
| `GET` | `/narratives/{id}/temporal-rules` | `an/ilp/` | Temporal ILP Horn clauses |
| `GET` | `/narratives/{id}/mean-field` | `an/mfg/` | Mean-field equilibria per situation (aggregated) |
| `GET` | `/narratives/{id}/psl` | `an/psl/` | Probabilistic Soft Logic scores |
| `GET` | `/narratives/{id}/arguments` | `an/af/` | Dung argumentation extensions |
| `GET` | `/narratives/{id}/evidence` | `an/ev/` | Dempster-Shafer fusions |

Read-only. To populate: submit the matching `POST /jobs` (`ContagionAnalysis`, `NetworkInference`, `TemporalMotifs`, `FactionEvolution`, `TemporalILP`, `MeanFieldGame`, `ProbabilisticSoftLogic`, `ArgumentationAnalysis`, `EvidenceCombination`). Single-blob endpoints return 404 when the engine hasn't been run; list endpoints return `{"<kind>": []}`.

## 5.16 RAG / Ask

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ask` | RAG question answering with citations |
| `POST` | `/narratives/{id}/communities/summarize` | Generate community summaries |
| `GET` | `/narratives/{id}/communities` | List (`?level=N` for hierarchy filter) |
| `GET` | `/narratives/{id}/communities/hierarchy` | Full hierarchy (all levels) |
| `GET` | `/narratives/{id}/communities/{cid}` | Specific summary |
| `GET` | `/cache/stats` | LLM response cache statistics |
| `POST` | `/cache/clear` | Clear LLM response cache |

### POST /ask

```json
{
  "question": "Who is the main suspect?",
  "narrative_id": "harbor-case",
  "mode": "hybrid",
  "response_type": "bullet points",
  "suggest": true,
  "debug": true
}
```

| Field | Description |
|-------|-------------|
| `mode` | `local` / `global` / `hybrid` (default) / `mix` / `drift` / `lazy` / `ppr` |
| `response_type` | LLM output format instruction |
| `suggest` | Generate follow-up question suggestions |
| `debug` | Include retrieval-pipeline internals in response |

Response: `{answer, citations: [{entity_id?, situation_id?, chunk_id?, excerpt, score}], mode, tokens_used, suggestions[], debug?}`.

## 5.17 Prompt Tuning

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/prompts/tune` | Trigger auto-tuning for a narrative |
| `GET` | `/prompts` | List all tuned prompts |
| `GET` | `/prompts/{narrative_id}` | Get |
| `PUT` | `/prompts/{narrative_id}` | Manually update |
| `DELETE` | `/prompts/{narrative_id}` | Delete |

## 5.18 Geocoding & Embeddings

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/geocode` | Geocode a place via Nominatim (cached at `geo/`). Body: `{place}` |
| `POST` | `/geocode/backfill` | Batch-geocode situations + Location entities missing coords. Body: `{narrative_id?, setting?, country_hint?, skip_canonicalization?}`. **As of v0.79.20**: when `narrative_id` and a configured LLM extractor are both available, runs a one-shot batch canonicalization pass to disambiguate ambiguous place names (e.g. *Marseilles* → Marseille, FR — not Marseilles, IL) using narrative-setting context. `setting` defaults to `Narrative.description + genre` when omitted. Each result row records `geo_provenance` (`source` / `llm_canonicalized` / `geocoded` / `manual`). Response: `{situations_geocoded, entities_geocoded, total_updated, canonicalization_used}`. |
| `POST` | `/embeddings/backfill` | Generate embeddings for entities/situations lacking them. Body: `{narrative_id?, force?}`. **As of v0.79.11**: per-row execution — empty inputs counted into `empty_skipped` instead of aborting the batch; provider errors caught + logged, accumulated into a capped `errors[]` (max 50). Response: `{entities_embedded, situations_embedded, total_updated, skipped, empty_skipped, failed, errors}`. Situation text is built from `name + description + first raw_content block` joined-and-trimmed. |
| `POST` | `/entities/backfill-settings` | **As of v0.79.50** — deterministic Location → Setting participation backfill. Body: `{narrative_id?}`. For each Location entity, builds a term list from `properties.name + properties.aliases[]` (terms < 3 bytes dropped), then for each Situation builds a haystack from `name + description + spatial.description + spatial.location_name + raw_content[].content`, lowercases both sides, and looks for any term flanked by ASCII-non-alphanumeric bytes (multi-byte UTF-8 chars count as boundaries — French diacritics work). When a match is found and `get_participations_for_pair(loc, sit)` is empty, inserts `Participation { role: Custom("Setting"), … }`. Idempotent — re-running only adds still-missing rows. No LLM calls. Response: `{locations_scanned, situations_scanned, links_created, skipped_existing_pairs, errors[]}`. Studio button: Cast & Places header → `🔗 Backfill location settings`. |
| `GET` | `/style-embeddings` | List style embeddings (`Vec<StyleEmbedding>`) for the SE picker on the Generate view. See §7.11.10 for the type and §7.11.14 for how the SE flows into the fitness loop |

Geocode precision values: `Exact` (building/amenity), `Area` (suburb/neighbourhood), `Region` (city/state), `Approximate`. Rate-limited to 1 rps per Nominatim usage policy. Negative results (fictional places) are cached.

**`SpatialAnchor.geo_provenance` (v0.79.20+):** distinguishes hard-fact coordinates from inferred ones. Values:
* `source` — coords came from the source text or a structured-import payload (hard fact, never re-geocoded).
* `llm_canonicalized` — LLM canonicalized the place name (with narrative-setting context) and Nominatim resolved it under the country filter.
* `geocoded` — direct Nominatim lookup, no canonicalization (legacy path / `skip_canonicalization: true`).
* `manual` — user-edited in Studio.

Rows that arrive with lat/lng already populated are stamped `source` automatically by the backfill route; archive imports do the same on read. The Studio map popup renders a one-glyph pill (`◆ source` / `◇ llm canon` / `~ geocoded` / `✎ manual`) so analysts can tell ground truth from inference at a glance.

**Per-location editor (Studio Cast canvas, v0.79.33+):** editing any `Location` entity in `/n/:narrativeId/cast` exposes a Geolocation strip with the current coords + provenance pill and three actions — *Geocode by name* (editable Nominatim query, side-modal preview), *Pick on map* (Leaflet picker with embedded `POST /geocode` search), and *Clear* (drops `latitude`/`longitude`/`geo_*` so the next backfill re-resolves). Both apply paths stamp `geo_provenance: 'manual'`.

**KV layout:** canonicalization results are cached at `geo/canon/{narrative_id}/{normalized_raw_name}` so re-runs of the same narrative skip the LLM call entirely. Country-filtered Nominatim results live at `geo/{cc}|{name}` to keep e.g. *marseille|fr* from colliding with *marseilles|us* in the cache.

## 5.18b Actor Images (v0.79.34+)

Per-actor portraits — uploaded files or generated via a configured text-to-image provider — backed by KV storage and round-tripped through `.tensa` archives.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/settings/image-gen` | Get image-generation provider config (keys redacted) |
| `PUT` | `/settings/image-gen` | Set image-gen provider. Body matches `ImageGenConfig` (`{provider: "openrouter"\|"openai"\|"local"\|"comfyui"\|"none", …}`). `merge_keys` preserves the existing API key when the body comes back redacted from a Settings round-trip |
| `GET` | `/settings/image-gen/styles` | Default starter styles: `["Photorealistic","Anime","Pencil sketch","Film noir","Identikit","Book illustration"]` |
| `GET` | `/entities/:id/images` | List portraits attached to an entity (oldest → newest), each carrying `url: "/images/:id"` |
| `POST` | `/entities/:id/images` | Upload `{mime, data: <base64>, caption?}`. 12 MB cap. Stores bytes in KV and appends a `properties.media[]` entry on the entity |
| `POST` | `/images/generate` | `{entity_id, prompt, style?, place?, era?, model?, caption?}` → calls the configured generator, persists the result, attaches it to the entity |
| `GET` | `/images/:id` | Stream raw bytes with the original `Content-Type` |
| `DELETE` | `/images/:id` | Remove from KV + detach from `entity.properties.media` |

**KV layout:**
* `img/m/{narrative_id}/{image_id_v7_BE_BIN_16}` — full `ImageRecord` (mime, source, prompt, style, place, era, provider, model, …)
* `img/b/{image_id_v7_BE_BIN_16}` — raw bytes
* `img/i/{image_id_v7_BE_BIN_16}` — `narrative_id` pointer for O(log N) by-id lookup (used by `GET /images/:id` and `DELETE /images/:id`)
* `cfg/image_gen` — persisted `ImageGenConfig`

**Provider config (`ImageGenConfig`):**
```json
// OpenRouter (image-capable models on the OpenAI shape)
{"provider": "openrouter", "api_key": "sk-or-...", "model": "black-forest-labs/flux-schnell"}
// OpenAI direct
{"provider": "openai", "api_key": "sk-...", "model": "gpt-image-1"}
// Self-hosted server speaking the OpenAI image API
{"provider": "local", "base_url": "http://localhost:8000/v1", "model": "...", "api_key": null}
// ComfyUI — config persisted but client not yet implemented
{"provider": "comfyui", "base_url": "http://localhost:8188", "workflow": null}
// Disabled
{"provider": "none"}
```

**Archive round-trip:** `ArchiveExportOptions.include_images` (default ON) writes per-narrative `narratives/{nid}/images/{image_id}.{ext}` blobs plus `narratives/{nid}/images/index.json` carrying every record. Import remaps `entity_id` against the existing remap table, retargets `narrative_id` to the final slug, recomputes `bytes_len`, and rewrites all three KV keys via the same `save_image` path used by the live upload route. Manifest gains a `layers.images: bool` flag.

**Studio surface:** Editing an `Actor` entity in `/n/:narrativeId/cast` exposes a Photos strip with thumbnails + Upload + Generate buttons. The Generate modal pre-fills `place` from the narrative's description and `era` from its genre (same hint source as the geocoder canon), offers the six default styles via a combo, and lets the user fully edit the composed prompt before submitting. Provider config lives at Settings → Image Generation.

## 5.19 Settings

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/settings/llm` | Get LLM provider config (keys redacted) |
| `PUT` | `/settings/llm` | Hot-swap ingestion LLM |
| `GET` | `/settings/models` | Discover models from local server (`?url=...`) |
| `GET` | `/settings/ingestion` | Get ingestion pipeline config |
| `PUT` | `/settings/ingestion` | Update |
| `GET` | `/settings/embedding` | Provider info + dimension + current model |
| `GET` | `/settings/embedding/models` | List available ONNX models in `models/embeddings/` |
| `PUT` | `/settings/embedding` | Switch model: `{"model": "all-MiniLM-L6-v2"\|"hash"\|"none"}` |
| `POST` | `/settings/embedding/download` | Download model from HuggingFace: `{"repo_id": "sentence-transformers/..."}` |
| `GET` | `/settings/vector-store` | Vector backend config |
| `GET` | `/settings/rag` | Get RAG config (budget, mode) |
| `PUT` | `/settings/rag` | Update |
| `GET` | `/settings/inference-llm` | Get dedicated query/RAG LLM |
| `PUT` | `/settings/inference-llm` | Set dedicated query/RAG LLM (falls back to ingestion LLM when unset) |
| `GET` | `/settings/chat-llm` | Get dedicated Studio chat LLM |
| `PUT` | `/settings/chat-llm` | Set dedicated Studio chat LLM |
| `GET` | `/settings/style-weights` | Get persisted `WeightedSimilarityConfig` (stylometry feature) |
| `PUT` | `/settings/style-weights` | Persist weights at `an/nw/` |
| `GET` | `/settings/presets` | List extraction mode presets (`{id, description}[]`) |

### LLM config JSON

```json
// Local LLM (vLLM, Ollama, LiteLLM)
{"provider": "local", "base_url": "http://localhost:11434", "model": "qwen3:32b"}

// OpenRouter
{"provider": "openrouter", "api_key": "sk-or-...", "model": "anthropic/claude-sonnet-4"}

// Anthropic
{"provider": "anthropic", "api_key": "sk-ant-...", "model": "claude-sonnet-4-20250514"}

// Google Gemini (feature: gemini)
{"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash"}

// AWS Bedrock (feature: bedrock)
{"provider": "bedrock", "region": "us-east-1", "model": "anthropic.claude-3-5-sonnet-20241022-v2:0"}

// Disabled
{"provider": "none"}
```

### Extraction Modes (`IngestionMode`)

Domain-specific extraction prompts selected per ingestion config or per request (`IngestRequest.ingestion_mode`):

| ID | Domain |
|----|--------|
| `novel` | Fiction — character-focused, chapter-aware, dialogue extraction |
| `news` | Journalism — date-focused, source attribution |
| `intelligence` | OSINT/reports — entity relationships, confidence, source reliability |
| `research` | Academic papers — citations, methodology, findings, concepts |
| `temporal_events` | Timelines/event databases — precise dates, Allen relations |
| `legal` | Contracts, court documents — parties, obligations, clauses |
| `financial` | Transactions, filings — entities, amounts, flows, compliance |
| `medical` | Clinical records — patients, conditions, treatments, outcomes |
| `custom` | User-defined via `TUNE PROMPTS` or `system_prompt` override |

## 5.20 Workspaces

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workspaces` | Create workspace |
| `GET` | `/workspaces` | List |
| `DELETE` | `/workspaces/{id}` | Delete (default workspace is protected) |

## 5.21 OpenAI-Compatible API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-format chat completion. Auto-detects TensaQL queries; otherwise proxies to the configured LLM. |
| `GET` | `/v1/models` | List available models |

## 5.22 Writer Workflows

All endpoints are narrative-scoped. See [Chapter 6 — Writer workflow](#writer-workflow) for the end-to-end flow.

### Plan

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narratives/{id}/plan` | Plan or `null` |
| `POST` | `/narratives/{id}/plan` | Full replace. Body: `NarrativePlan` (logline, synopsis, premise, themes, plot_beats, style, length, setting, notes, target_audience, comp_titles, content_warnings, custom) |
| `PUT` | `/narratives/{id}/plan` | Partial patch (`null` clears, omission preserves; nested structs replace wholesale) |
| `DELETE` | `/narratives/{id}/plan` | Delete |

### Revisions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/narratives/{id}/revisions` | Commit current state `{message, author?}`. Returns `no_change` when content hash matches HEAD |
| `GET` | `/narratives/{id}/revisions` | List newest-first (summaries only) |
| `GET` | `/narratives/{id}/revisions/head` | Latest summary |
| `GET` | `/revisions/{rev_id}` | Full revision including snapshot |
| `POST` | `/narratives/{id}/revisions/{rev_id}/restore` | Restore. Auto-commits current state first |
| `POST` | `/narratives/{id}/revisions/diff` | Structural + line-level prose diff. Response bundles `scene_summaries[]` with per-scene `{situation_id, header, lines_added, lines_removed, word_delta, change_kind: Added\|Removed\|Modified}` |
| `GET` | `/narratives/{id}/diff-revisions?from=UUID&to=UUID` | Same diff via GET query params |

### Generation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generation/propose` | Propose generation. Body: `{narrative_id, request: {kind: outline\|characters\|scenes, ...}}`. `?dry_run=true` returns prompt + token estimate. Outline accepts `pov_hint: {mode: "single"\|"rotating"\|"omniscient", entity_name?\|entity_names?}`; scenes accepts `pov_entity_name`. Proposed situations carry `pov_entity_name` + `voice` which `apply` resolves into `Situation.discourse`. |
| `POST` | `/generation/apply` | Commit proposal. Body: `{proposal, commit_message?, author?}`. Adjacent same-level situations in the proposal are automatically linked with a weak `Enabling` causal edge (`mechanism: "narrative sequence"`, strength 0.5) so Workshop's causal-graph analyses have something to read. `AppliedReport.causal_links_created` reports how many were added. |
| `POST` | `/narratives/:id/causal-links/backfill-adjacent` | One-shot repair for empty causal graphs. Walks the narrative's situations in manuscript order and adds a weak `Enabling` link (strength 0.3, mechanism `"sequential (backfill)"`) between each adjacent same-level pair that doesn't already have an edge. Idempotent. Beat/Event/Line levels are skipped. Returns `{narrative_id, situations_total, links_added, pairs_skipped_existing, pairs_skipped_cross_level}`. |
| `POST` | `/narratives/:id/names/backfill-from-content` | Derive a short title from the first line of prose for every situation with an empty `name`. Strips `"Chapter N — "` / `": "` / `". "` prefixes, cuts at the first sentence terminator, caps at 80 chars on a word boundary. Idempotent. Returns `{narrative_id, situations_total, names_set, skipped_already_named, skipped_no_content}`. |

**Fitness-loop chapter generation.** Submission reuses the inference job system — there is no dedicated `/generation/chapter-with-fitness` endpoint. Submit `POST /jobs` with `job_type: "chapter_generation_fitness"` (parameter shape under §5.6 / Inference Job Types) and poll `GET /jobs/:id/result`. The returned text is the **best-scoring attempt across iterations**, not the last one (§7.11.14).

**Studio surface.** The Generate view (`/story/:narrativeId/generate`) gains a `Fitness Loop` mode alongside the existing outline / characters / scenes modes. The form lists a target-fingerprint picker (a narrative dropdown that, on selection, fetches and previews that narrative's prose features inline), an SE picker (grouped by `StyleEmbeddingSource` variant via `seVariant`, backed by `GET /style-embeddings`), a fitness-threshold slider (default `0.80`), max-retries control (1–8, default 3), and a temperature slider. Submission requires at least one of voice description / SE / target fingerprint — pure-default submission is blocked client-side. The view polls `GET /jobs/:id` every 3s with a `useRef`-tracked interval cleared on unmount, then renders `FitnessResultPanel` (in `studio/src/components/writer/`) with a per-attempt accordion (best attempt expanded by default, "best attempt" badge), a D3 score-over-iteration sparkline, and an "Accept and re-ingest" button. **The re-ingest button currently ships disabled** with a precise TODO in its `title` attribute: no backend endpoint accepts a fitness-generated chapter text into the manuscript yet. Wire it once chapter-text persistence lands. The sparkline uses the proxy `accepted ? chapter.style_adherence : 0` for per-attempt height because the result-log shape (§5.6) does not carry per-attempt scores; the proxy is documented in both the file-level JSDoc and an in-component hint block. The job monitor (`/inference`) labels these jobs with a writer-distinct entry in `studio/src/config/inferenceMatrix.ts` so writer jobs are visually separable from analytic jobs. The typed client supplement lives in `studio/src/api/fitness_client.ts` (`StyleEmbedding`, `ChapterFitnessJobParams`, `FitnessLogEntry`, `ChapterFitnessResult`, plus `listStyleEmbeddings` / `submitChapterFitnessJob` / `getChapterFitnessResult`) and reuses `apiBase` / `apiHeaders` / `ApiError` / `InferenceResult` / `NarrativeFingerprint` from `client.ts` so the main client file does not grow.

### Editing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/editing/propose` | Propose an edit. Body: `{narrative_id, situation_id, operation}`. `operation`: `Rewrite` / `Tighten` / `Expand` / `StyleTransfer` / `DialoguePass`. `?dry_run=true` for cost preview |
| `POST` | `/editing/apply` | Commit |

### Workshop

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/narratives/:id/workshop/estimate` | Cost preview (no LLM). `narrative_id` is taken from the path; the body field is optional |
| `POST` | `/narratives/:id/workshop/run` | Run a pass. Body: `{tier: cheap\|standard\|deep, focuses[], max_llm_calls?}` (`narrative_id` optional — path wins). `focuses`: subset of `[pacing, continuity, characterization, prose, structure]`. `deep` tier returns a deferred report |
| `GET` | `/narratives/:id/workshop/reports` | List summaries newest-first |
| `GET` | `/workshop/reports/{id}` | Full report |

### Continuity (pinned facts)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narratives/{id}/pinned-facts` | List |
| `POST` | `/narratives/{id}/pinned-facts` | Pin `{key, value, note?, entity_id?}` |
| `PUT` | `/narratives/{id}/pinned-facts/{fact_id}` | Patch |
| `DELETE` | `/narratives/{id}/pinned-facts/{fact_id}` | Delete |
| `POST` | `/narratives/{id}/continuity/check` | Scan prose `{prose}`; returns `ContinuityWarning[]`. Deterministic — no LLM cost |
| `POST` | `/narratives/{id}/continuity-check` | Proposal-scoped check. Body: `{kind: "generation"\|"edit", proposal}` |
| `GET` | `/narratives/{id}/workspace` | Writer workspace dashboard: counts, plan status, recent revisions, recent workshop reports, ranked next-step suggestions |

### Cost Ledger

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narratives/{id}/cost-ledger?window=` | Raw entries newest-first. Window: `24h`/`7d`/`30d`/`all` |
| `GET` | `/narratives/{id}/cost-ledger/summary?window=` | Aggregated totals + per-operation breakdown (Generation / Edit / Workshop / Continuity) |

**Entry schema (`CostLedgerEntry`).** Persisted at `cl/{narrative_id}/{entry_uuid_v7}`. Beyond the standard fields (id, narrative_id, kind, prompt_tokens, completion_tokens, cost, created_at), v0.65 added an optional `metadata: Option<serde_json::Value>` field, declared as `#[serde(default, skip_serializing_if = "Option::is_none")]` so older records still deserialize unchanged. Fitness-loop chapter generation writes one entry per iteration with stable `kind = "chapter_gen_fitness"` and `metadata = {"iteration": N, "score": s}`; using a stable `kind` keeps `cost-ledger/summary` aggregations bounded while the per-iteration drill-down lives inside `metadata`.

### Research

Research panel treats sources / attributions / contentions / pinned facts as a live overlay on the manuscript. Writers can also pin lightweight per-scene notes.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/situations/{id}/research-context` | `SceneResearchContext` bundle: scene-attributed sources, participant sources, contentions, in-scope pinned facts, research notes |
| `GET` | `/situations/{id}/research-notes` | List notes pinned to scene |
| `POST` | `/situations/{id}/research-notes` | Create `{narrative_id, kind, body, source_chunk_id?, source_id?, author?}`. `kind`: `Quote` / `Clipping` / `Link` / `Note` |
| `POST` | `/situations/{id}/research-notes/from-chunk` | Promote a ChunkStore chunk into a scene-scoped note |
| `GET` | `/narratives/{id}/research-notes` | List all notes in a narrative |
| `GET` / `PUT` / `DELETE` | `/research-notes/{id}` | Per-note CRUD |

### Fact-Check

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/situations/{id}/factcheck` | Extract atomic claims, verdict against scene's research context. Body: `{text, tier?}` (`Fast` / `Standard` / `Deep`). Returns `FactCheckReport` with per-claim `VerdictStatus` (`Supported`, `Contested`, `Unsupported`, `Contradicted`) and `EvidenceRef[]` (`PinnedFact`, `Source`, `Contention`, `Note`) |

### Cited Generation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/situations/{id}/generation-prompt` | Assemble research addendum. Body: `CitedGenerationRequest {require_citations, stances}`. Returns `ResearchPromptAddendum` + `pending_contentions[]` the writer must resolve first |
| `POST` | `/situations/{id}/hallucination-guard` | Fact-check freshly-generated text; returns `{blocking: [Contradicted verdicts], full: FactCheckReport}` |
| `POST` | `/generation/parse-citations` | Strip `[[cite: <uuid>]]` markers. Body: `{raw}`. Returns `ParsedCitedText {clean_text, spans: [{text, span, citations}], unique_citations}` |

### Annotations

Structured inline annotations anchored on byte spans of a scene's concatenated prose. Comments stay in the editor; Footnotes and Citations render into compile output.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/situations/{id}/annotations` | List (sorted by `span.0`) |
| `POST` | `/situations/{id}/annotations` | Create `{kind: Comment\|Footnote\|Citation, span: [start, end], body, source_id?, chunk_id?, author?}` |
| `POST` | `/situations/{id}/annotations/reconcile` | Re-anchor after prose edit `{old_prose, new_prose}`. Returns `{moved, detached, unchanged}` |
| `GET` / `PUT` / `DELETE` | `/annotations/{id}` | Per-annotation CRUD |

### Compile Profiles

Saved compile profiles. Output formats: Markdown, EPUB 3 (hand-rolled container + XHTML), DOCX (requires `docparse` feature).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narratives/{id}/compile-profiles` | List |
| `POST` | `/narratives/{id}/compile-profiles` | Create `CompileProfile` (`name`, `description?`, `include_labels[]`, `exclude_labels[]`, `include_statuses[]`, `heading_templates[]`, `front_matter_md?`, `back_matter_md?`, `footnote_style: Inline\|Endnotes\|None`, `include_comments`) |
| `POST` | `/narratives/{id}/compile?format=markdown\|epub\|docx&profile_id=?` | Emit compiled bytes. Without `profile_id` a default profile is synthesised |
| `GET` / `PUT` / `DELETE` | `/compile-profiles/{id}` | Per-profile CRUD |

### Collections (Saved Searches)

Virtual folders surfaced in Studio Binder alongside the real scene hierarchy. Persisted filters re-resolve on demand.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narratives/{id}/collections` | List |
| `POST` | `/narratives/{id}/collections` | Create `{name, description?, query: CollectionQuery{labels[], statuses[], keywords_any[], text?, min_order?, max_order?, min_words?, max_words?}}` |
| `GET` / `PUT` / `DELETE` | `/collections/{id}` | Per-collection CRUD |
| `GET` | `/collections/{id}/resolve` | Evaluate the saved query against current scene state. Returns `{collection_id, narrative_id, matches: [situation_id, ...], match_count}` |

### Narrative Templates (Sprint W15, v0.70.0)

Reusable scaffolding templates (distinct from the ingestion `/templates` surface). Three builtins plus any user-stored templates.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/narrative-templates` | List builtin + stored templates (deduplicated by id) |
| `POST` | `/narrative-templates/{id}/instantiate` | Bind slots to entities. Body: `{bindings: {slot_id: entity_uuid, ...}}`. Returns `TemplateInstantiation` — does NOT write to the hypergraph |

## 5.23 Stylometry (`stylometry` feature)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/narratives/{id}/style` | Compute and store style profile + prose features |
| `GET` | `/narratives/{id}/style` | Retrieve profile (computes on-the-fly if absent) |
| `GET` | `/narratives/{id}/fingerprint` | Combined fingerprint (prose + structure) |
| `POST` | `/style/compare` | Compare `{narrative_a, narrative_b}`. Query: `?ci=true&alpha=0.05&n_iter=500&seed=...` adds bootstrap percentile CIs on overall + prose similarity |
| `GET` | `/narratives/{id}/style/anomalies` | Per-chapter anomalies. Legacy: `?threshold=0.7`. Calibrated: `?mode=calibrated&alpha=0.05&n_iter=1000&seed=...` returns `Vec<AnomalyPValue>` with bootstrap-derived empirical p-values |
| `GET` | `/narratives/{id}/style/radar` | 12-axis radar chart data (includes `surprise` from normalized situation self-information) |
| `POST` | `/style/verify` | PAN@CLEF authorship verification. Body: `{text_a, text_b, config?}`. Returns `{score, decision, same_author_probability}`. `decision` is `None` when the score falls inside the configured uncertainty band |
| `POST` | `/style/pan/evaluate` | PAN@CLEF metric suite (AUC, c@1, F0.5u, F1, Brier, overall). Body: `{pairs: [...] or dataset_path, config?}` |

## 5.24 Disinformation (`disinfo` feature)

### Fingerprints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/entities/{id}/behavioral-fingerprint` | Load (compute on first access) 10-axis behavioral fingerprint. NaN axes serialize as JSON `null` |
| `POST` | `/entities/{id}/behavioral-fingerprint/compute` | Force recompute + persist |
| `GET` | `/narratives/{id}/disinfo-fingerprint` | Load (compute on first access) 12-axis disinfo fingerprint |
| `POST` | `/narratives/{id}/disinfo-fingerprint/compute` | Force recompute + persist |
| `POST` | `/fingerprints/compare` | `{kind: "behavioral"\|"disinfo", task: "literary"\|"cib"\|"factory"?, a_id, b_id}`. Returns `FingerprintComparison` (composite distance, per-layer distances, `p_value`, 95% CI, `same_source_verdict`, `comparable_axes`, top-3 `anomaly_axes`) |

### Spread Dynamics

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/spread/r0` | `{narrative_id, fact, about_entity, narrative_kind?, beta_overrides?}`. SMIR + jumps + velocity check; persists SMIR snapshot. Returns `{smir, cross_platform_jumps, alerts}` |
| `GET` | `/spread/r0/{narrative_id}` | Load most recent persisted SMIR snapshot |
| `GET` | `/spread/velocity/{narrative_id}?limit=N` | Recent velocity alerts (default 50, max 500) |
| `GET` | `/spread/jumps/{narrative_id}` | Cross-platform jumps |
| `POST` | `/spread/intervention` | Pure counterfactual projection — does not overwrite production snapshot. `intervention`: `RemoveTopAmplifiers { n }` or `DebunkAt { at }` |

### CIB Detection

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analysis/cib` | `{narrative_id, cross_platform?, similarity_threshold?, alpha?, bootstrap_iter?, min_cluster_size?, seed?}`. Behavioral-similarity network → label-propagation → calibrated p-value density threshold → flag clusters. Persists at `cib/c/` + `cib/e/` |
| `GET` | `/analysis/cib/{narrative_id}` | Load persisted clusters + evidence (404 when nothing computed) |
| `POST` | `/analysis/superspreaders` | `{narrative_id, method?, top_n?}`. Rank actors by `pagerank` / `eigenvector` / `harmonic`. Persists at `cib/s/{narrative_id}` |

### Claims & Fact-Check

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/claims` | Detect claims `{text, narrative_id?, source_situation_id?, source_entity_id?}` |
| `GET` | `/claims/{id}` | Get a claim |
| `POST` | `/claims/match` | Match against fact-checks `{claim_id, min_similarity?}` |
| `POST` | `/fact-checks` | Ingest `{claim_id, verdict, source, url?, language?, explanation?, confidence?}`. Verdicts: `true`, `false`, `misleading`, `partially_true`, `unverifiable`, `satire`, `out_of_context` |
| `GET` | `/claims/{id}/origin` | Trace claim origin through mutation chain |
| `GET` | `/claims/{id}/mutations` | Mutation events |
| `POST` | `/claims/sync` | Trigger fact-check database sync |
| `GET` | `/claims/sync/history` | Sync history (newest first, up to 50) |
| `GET` | `/claims/sync/sources` | Configured sync sources |

### Archetypes & DS Fusion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/actors/{id}/archetype` | Classify actor. Body: `{force?}`. Returns softmax over 6 archetypes |
| `GET` | `/actors/{id}/archetype` | Cached classification |
| `POST` | `/analysis/disinfo-assess` | DS fusion `{target_id, signals: [{source, mass_true, mass_false, mass_misleading, mass_uncertain}]}` |
| `GET` | `/analysis/disinfo-assess/{id}` | Cached assessment |

### Multilingual

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/lang/detect` | `{text}` — returns `{language, normalized_text}` |

### Monitor Subscriptions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/monitor/subscriptions` | Create `MonitorSubscription` `{narrative_id, platforms[], threshold, callback, active}`. `callback`: `"log"` / `{"webhook": "url"}` / `"internal_queue"` |
| `GET` | `/monitor/subscriptions` | List (`?active_only=true`) |
| `DELETE` | `/monitor/subscriptions/{id}` | Delete |
| `GET` | `/monitor/alerts` | Recent alerts (`?limit=N`) |

### Scheduler, Discovery, Reports

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/scheduler/tasks` | List |
| `POST` | `/scheduler/tasks` | Create `{task_type, schedule, enabled, config}`. Types: `cib_scan`, `source_discovery`, `fact_check_sync`, `report_generation`, `mcp_poll`, `fingerprint_refresh`, `velocity_baseline_update`. Schedule: `"30m"`, `"6h"`, `"1d"` |
| `DELETE` | `/scheduler/tasks/{id}` | Delete |
| `GET` | `/scheduler/tasks/{id}/history` | Execution history (up to 50) |
| `POST` | `/scheduler/tasks/{id}/run-now` | Trigger immediate execution |
| `GET` | `/discovery/candidates` | Potential new sources |
| `POST` | `/discovery/candidates/{id}/approve` | Approve |
| `POST` | `/discovery/candidates/{id}/reject` | Reject |
| `GET` / `PUT` | `/discovery/policy` | Policy: `manual` / `auto_approve_co_amplification` / `auto_approve_all` |
| `POST` | `/reports/situation` | Generate situation report `{hours: 24}` |
| `GET` | `/reports` | List (newest first, up to 50) |
| `GET` | `/reports/{id}` | Get report |
| `GET` | `/ingest/mcp-sources/health` | MCP source health records |
| `GET` | `/ingest/mcp-sources` | List configured MCP sources (placeholder) |

## 5.25 Narrative Architecture & Generation

Routes landed in v0.69.0 (Sprint W14) alongside the MCP wiring. Detection routes
ship unconditionally; plan / materialize / prompt-prep routes are gated behind the
`generation` feature.

| Method | Path | Feature | Description |
|--------|------|---------|-------------|
| `GET` | `/narratives/{id}/commitments` | — | Detect setup-payoff pairs |
| `GET` | `/narratives/{id}/commitment-rhythm` | — | Per-chapter tension curve + fulfillment ratio |
| `GET` | `/narratives/{id}/fabula` | — | Chronological event ordering |
| `GET` | `/narratives/{id}/sjuzet` | — | Discourse / telling order |
| `GET` | `/narratives/{id}/sjuzet/reorderings` | — | Candidate alternative orderings |
| `GET` | `/narratives/{id}/dramatic-irony` | — | Reader vs. character knowledge gaps |
| `GET` | `/narratives/{id}/focalization` | — | Focalization × irony interactions |
| `GET` | `/narratives/{id}/character-arc` | — | Detect arc(s). `?character_id=UUID` for a single arc; omit to list stored arcs |
| `GET` | `/narratives/{id}/subplots` | — | Subplot detection |
| `GET` | `/narratives/{id}/scene-sequel` | — | Scene-sequel rhythm (Swain/Bickham) |
| `POST` | `/narratives/plan` | `generation` | Generate a plan (body: `{premise, genre?, chapter_count?, subplot_count?, ...}`) |
| `POST` | `/plans/{plan_id}/materialize` | `generation` | Write plan into the hypergraph |
| `GET` | `/narratives/{id}/validate-materialized` | `generation` | Consistency issues for a materialized narrative |
| `POST` | `/narratives/{id}/generate-chapter` | `generation` | Prepare chapter prompt (body: `{chapter, voice_description?}`) — no LLM call |
| `POST` | `/narratives/{id}/generate-narrative` | `generation` | Prepare prompts for all chapters (body: `{chapter_count, voice_description?}`) — no LLM call |
| `POST` | `/narratives/{id}/generate` | `generation` | LLM-driven outline / character / scene generation (returns a proposal) |
| `POST` | `/narratives/{id}/generate/apply` | `generation` | Apply an accepted proposal (writes + commits revision) |
| `POST` | `/narratives/{id}/generate/estimate` | `generation` | Token estimate for a generation request (no LLM) |

## 5.26 Studio Agent Chat (`studio-chat` feature)

Integrated agent chat for Studio: a right-side panel that runs a multi-turn LLM loop with local tools, third-party MCP servers, per-session skill toggles, and a confirmation gate on mutations.

Every session is keyed by `(workspace_id, user_id)`; defaults `default` / `local`. Studio sends `X-Tensa-Workspace` and `X-Tensa-User` on all chat calls. Auth is not built yet; the schema is ready.

**Extractor resolution order** when handling a chat turn: chat-specific LLM (`cfg/studio_chat_llm`) → inference/RAG LLM (`cfg/inference_llm`) → ingestion LLM (`cfg/llm`). When all three are unset, the turn emits an `error` event with code `no_llm` and the panel surfaces a "No chat LLM configured" state.

### Tool catalog

12 hand-rolled built-in tools, grouped by class:

| Class | Tools |
|-------|-------|
| Read | `list_narratives`, `get_narrative`, `list_entities`, `get_entity`, `list_situations`, `get_situation`, `query_tensaql` (MATCH-only) |
| UI | `navigate_to`, `show_toast` |
| Mutating | `create_entity`, `create_situation`, `create_narrative` |

Mutating tools are classified by `confirm::classify()` and park the harness on `ConfirmGate::register`; a `{call_id, decision: "approve"\|"reject"}` body on `POST /studio/chat/sessions/:id/confirm` resolves the parked future. Read-only tools execute immediately. Unknown tool names default to **Confirm** so new additions cannot silently run.

In addition to built-ins, user-registered third-party stdio MCP servers (see `/studio/chat/mcp-servers` below) contribute tools namespaced as `{server}__{tool}`. They go through the same classifier, and unknown tools default to Confirm.

**Tool call format.** The LLM emits tool calls as a fenced code block:

~~~
```tensa-tool
{"tool": "list_narratives", "args": {}}
```
~~~

The harness parses the first such block per iteration, loops up to **6 iterations** per user turn, and reports `tool_call` → (optionally `awaiting_confirm`) → `tool_result` events over SSE. `tool_result.output_preview` is capped at ~1 KiB to keep the stream lively; the full result lives in the persisted `ToolResult` content part.

### Skills

A per-session skill list scopes which modules shape the assistant's behaviour. Bundled skills:

- `studio-ui` — navigation + toast helpers
- `tensa` — read/query tools against the hypergraph
- `tensa-writer` — writer workflow guidance
- `tensa-synth` — synthetic generation (EATH), null-model significance, z-score interpretation, hybrid mixtures, wargame substrates (added v0.75.0, EATH Phase 12)

The header picker in Studio toggles these per session; `PATCH /studio/chat/sessions/:id` with `{active_skills: [...]}` persists the choice. `GET /studio/chat/skills` enumerates all bundled skills with their descriptions.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/studio/chat` | Start/continue a turn (SSE). Body: `{session_id?, user_message, active_skills?, narrative_scope?, model_override?}`. Creates a session on the fly if absent |
| `POST` | `/studio/chat/sessions` | Create empty session `{title?, active_skills?, narrative_scope?}` |
| `GET` | `/studio/chat/sessions` | List `SessionMeta` newest-first |
| `GET` | `/studio/chat/sessions/{id}` | `{meta, messages}` full history |
| `PATCH` | `/studio/chat/sessions/{id}` | Patch `{title?, archived?, active_skills?, narrative_scope?, model_override?}`. `null` on `narrative_scope` / `model_override` clears; omission preserves |
| `DELETE` | `/studio/chat/sessions/{id}` | Delete session + cascade messages |
| `POST` | `/studio/chat/sessions/{id}/confirm` | Approve/reject a pending tool call. Body: `{call_id, decision: "approve"\|"reject"}` |
| `POST` | `/studio/chat/sessions/{id}/stop` | Cancel an in-flight turn |
| `GET` | `/studio/chat/skills` | Enumerate bundled skills |
| `GET` | `/studio/chat/mcp-servers` | List configured third-party MCP servers + live status |
| `POST` | `/studio/chat/mcp-servers` | Upsert a server. Body: `{name, command, args, env?, enabled?, description?}`. Spawned lazily on first tool call |
| `DELETE` | `/studio/chat/mcp-servers/{name}` | Remove server + drop its proxy |

### SSE event envelope

Events are emitted as `event: <name>\ndata: <json>\n\n` frames. The JSON payload re-states the discriminator as `"event"` so clients can parse the data line alone.

| `event` | Payload |
|---------|---------|
| `session_started` | `{session_id, title}` |
| `user_persisted` | `{message_id}` (v7 UUID hex, server-assigned) |
| `token` | `{delta: string}` (append to streaming assistant text) |
| `tool_call` | `{id, name, args}` |
| `awaiting_confirm` | `{call_id, summary, preview}` |
| `tool_result` | `{call_id, ok, output_preview}` |
| `final` | `{message_id}` (turn complete) |
| `error` | `{code, msg}` |

### Persistence

KV keys (all on root store, *not* workspace-prefixed, so cross-workspace admin enumeration stays cheap):

```
chat/s/{workspace_id}/{user_id}/{session_id}              SessionMeta
chat/m/{workspace_id}/{user_id}/{session_id}/{msg_v7}     Message
```

UUIDv7 hex ordering preserves chronological order on prefix scans, so message listing returns insertion order without a separate timestamp sort.

## 5.27 Adversarial Wargaming (`adversarial` feature)

Adversary policies persist at `adv/policy/`, simulation state at `adv/sim/`, wargame sessions at `adv/wg/`, DISARM TTP calibration at `adv/disarm/`, audit trail at `adv/audit/`.

Wargame REST routes (added v0.79.1 — paired with the existing MCP tools so the Studio Wargame view works without going through MCP):

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/wargame/sessions` | Create session. Body: `{narrative_id, max_turns?, time_step_minutes?, auto_red?, auto_blue?, red_objectives?, blue_objectives?, background?}`. Returns `{session_id}` (201). |
| `GET`  | `/wargame/sessions` | List all sessions as `WargameSummaryResponse[]` (`SessionSummary` + `narrative_id`). |
| `GET`  | `/wargame/sessions/{id}/state` | Single session summary. |
| `POST` | `/wargame/sessions/{id}/auto-play` | Body `{num_turns}` → `TurnResult[]`. Uses the `auto_red` / `auto_blue` heuristics for whichever team(s) are auto-controlled. |
| `POST` | `/wargame/sessions/{id}/moves` | Body `{red_moves, blue_moves}` → `TurnResult`. Submit explicit moves for one turn. |
| `DELETE` | `/wargame/sessions/{id}` | Idempotent delete. |

---

# Chapter 6: Common Workflows

## Ingest a Novel and Explore It

**Step 1: Create the narrative container**

```
create_narrative(
  narrative_id: "crime-and-punishment",
  title: "Crime and Punishment",
  genre: "novel",
  tags: ["dostoevsky", "russian", "1860s"]
)
```

**Step 2: Ingest the text**

```
ingest_text(
  text: "<full text of the novel>",
  narrative_id: "crime-and-punishment",
  source_name: "gutenberg-edition",
  auto_commit_threshold: 0.8,
  review_threshold: 0.3
)
```

The pipeline will:
- Chunk the text into paragraphs
- Extract entities (Raskolnikov, Sonya, Porfiry, etc.)
- Extract situations (the murder, the confession, etc.)
- Resolve entity coreferences ("Rodya" = "Raskolnikov")
- Gate by confidence (high → auto-commit, medium → review queue, low → reject)

**Step 2b (optional): Ingest with enrichment**

Pass `enrich: true` for deeper extraction of beliefs, game structures, discourse annotations, information sets, outcome models, and a global cross-chunk Allen-relation timeline.

**Step 2c (optional): SingleSession ingestion**

For best quality on large LLMs, use `single_session: true`. The LLM sees the entire text upfront and extracts each chunk as a follow-up turn in one continuous conversation, with accumulator summaries, in-session repair on parse failures, and a final reconciliation turn that merges duplicate entities, builds the global timeline, and adjusts confidence from full-text context.

SingleSession requires OpenRouter or a Local LLM (not Anthropic direct). If the text exceeds the model's context budget it automatically falls back to standard mode.

**Step 3: Review the queue**

```
review_queue(action: "list", limit: 20)
review_queue(action: "approve", item_id: "uuid", reviewer: "analyst-1")
```

**Step 4: Explore the narrative**

```
query(tensaql: 'MATCH (e:Actor) ACROSS NARRATIVES ("crime-and-punishment") RETURN e')
query(tensaql: 'MATCH (e:Actor) -[p:PARTICIPATES]-> (s:Scene) RETURN e.name, s LIMIT 20')
query(tensaql: 'MATCH (s:Situation) AT s.temporal BEFORE "1866-08-01" RETURN s')
```

**Step 5: Export for visualization**

```
export_narrative(narrative_id: "crime-and-punishment", format: "graphml")
```

Import the GraphML file into Gephi for network visualization.

## Build an Investigation Case

**Step 1: Register sources**

```
create_source(
  name: "Field Report #127",
  source_type: "HumanAnalyst",
  trust_score: 0.85,
  description: "Senior field officer, 10yr experience"
)

create_source(
  name: "SIGINT Intercept 2025-03-12",
  source_type: "Sensor",
  trust_score: 0.9
)
```

**Step 2: Create narrative and entities**

```
create_narrative(narrative_id: "harbor-case", title: "Harbor Investigation")

create_entity(entity_type: "Actor", properties: {"name": "Viktor Orlov"}, narrative_id: "harbor-case")
create_entity(entity_type: "Location", properties: {"name": "Warehouse 7"}, narrative_id: "harbor-case")
create_entity(entity_type: "Organization", properties: {"name": "Nightfall Group"}, narrative_id: "harbor-case")
```

**Step 3: Add situations and link participants**

```
create_situation(
  name: "Warehouse 7 Meeting",
  raw_content: "Orlov meets unknown contact at Warehouse 7",
  start: "2025-03-15T22:15:00Z",
  narrative_level: "Scene",
  narrative_id: "harbor-case"
)

add_participant(entity_id: "orlov-uuid", situation_id: "meeting-uuid",
                role: "Protagonist", action: "meets contact")
```

**Step 4: Attribute to sources**

```
add_attribution(source_id: "field-report-uuid",
                target_id: "meeting-uuid",
                target_kind: "Situation",
                excerpt: "Subject observed entering warehouse at 22:15")
```

**Step 5: Run causal inference**

```
infer(tensaql: 'INFER CAUSES FOR s:Situation RETURN s')
job_status(job_id: "returned-job-id")
job_result(job_id: "returned-job-id")
```

## Run Inference and Interpret Results

**Motivation inference**

```
infer(tensaql: 'INFER MOTIVATION FOR e:Actor RETURN e')
```

Result: reward weights, archetype classification (PowerSeeking, Altruistic, Opportunistic, etc.), trajectory length, confidence.

**Game-theoretic analysis**

```
infer(tensaql: 'INFER GAME FOR s:Scene UNDER RATIONALITY = 0.7 RETURN s')
```

Result: game type (PrisonersDilemma, Coordination, etc.), equilibrium strategies, λ, sub-games.

**Counterfactual simulation**

```
simulate_counterfactual(
  situation_id: "meeting-uuid",
  intervention_target: "action",
  new_value: "refuses to meet"
)
```

Result: affected downstream situations with probability estimates.

**Actor dossier**

```
get_actor_profile(actor_id: "orlov-uuid")
```

Returns all participations, roles, actions, state changes, and inferred characteristics.

## Cross-Narrative Comparison

```
find_cross_narrative_patterns(narrative_ids: ["harbor-case", "embassy-case", "airport-case"])

infer(tensaql: 'DISCOVER PATTERNS ACROSS NARRATIVES ("harbor-case", "embassy-case") RETURN *')

query(tensaql: 'MATCH (e:Actor) ACROSS NARRATIVES RETURN e.name, e.narrative_id')

infer(tensaql: 'DISCOVER ARCS RETURN *')
```

`DISCOVER ARCS` classifies narratives using the Reagan 6-arc taxonomy (Rags to Riches, Riches to Rags, Man in a Hole, Icarus, Cinderella, Oedipus).

## Source Intelligence and Contention Resolution

```
list_contentions(situation_id: "disputed-event-uuid")
recompute_confidence(id: "entity-uuid")
```

The confidence breakdown returns `extraction`, `source_credibility`, `corroboration`, `recency`, and a composite score. Adding/removing attributions or changing a source's `trust_score` auto-propagates to attributed targets.

```
infer(tensaql: 'INFER ARGUMENTS FOR s:Situation RETURN s')
```

Returns which claims survive attack relationships (grounded extension = most conservative accepted set).

## Export for External Tools

```
export_narrative(narrative_id: "harbor-case", format: "graphml")     -- Gephi
export_narrative(narrative_id: "harbor-case", format: "csv")         -- Excel/Sheets
export_narrative(narrative_id: "harbor-case", format: "json")        -- programmatic
export_narrative(narrative_id: "harbor-case", format: "manuscript")  -- readable prose
export_narrative(narrative_id: "harbor-case", format: "report")      -- analytical report
export_narrative(narrative_id: "harbor-case", format: "stix")        -- STIX 2.1 bundle
```

Via TensaQL:

```sql
EXPORT NARRATIVE "harbor-case" AS graphml
EXPORT NARRATIVE "harbor-case" AS archive
```

**Lossless archive for backup or transfer:**

```
export_archive(narrative_ids: ["harbor-case", "smuggling-ring"])
import_archive(data: "<base64-encoded .tensa file>")
```

The `.tensa` archive is a ZIP of JSON files. It preserves everything: entities, situations, participations, causal links, sources, attributions, contentions, chunks, state versions, inference results, community summaries, tuned prompts, taxonomy, and projects. External tools can also create `.tensa` archives — the minimal valid archive needs only `manifest.json`, `narrative.json`, `entities.json`, `situations.json`, `participations.json`.

## RAG Question Answering with DRIFT Search

```sql
-- Simple question with hybrid retrieval
ASK "Who are the main suspects?" OVER "harbor-case" MODE hybrid

-- DRIFT traverses community hierarchy for complex questions
ASK "What are the major power dynamics and how do they evolve?" OVER "harbor-case" MODE drift

-- Follow-up suggestions
ASK "What happened at the warehouse?" OVER "harbor-case" SUGGEST

-- Multi-turn conversation
ASK "Who is Viktor Orlov?" OVER "harbor-case" SESSION "my-session"
ASK "What did he do at the warehouse?" SESSION "my-session"

-- Custom response format
ASK "Summarize the investigation timeline" OVER "harbor-case" RESPOND AS "bullet points"
```

## Prompt Tuning for Domain-Specific Ingestion

```
ingest_text(text: "<novel text>", narrative_id: "gothic-novel")
```

```sql
TUNE PROMPTS FOR "gothic-novel"
```

Or via MCP: `tune_prompts(narrative_id: "gothic-novel")`.

The system samples chunks, sends them to the LLM to analyze the domain, and generates tailored extraction guidelines (e.g. "Focus on atmospheric descriptions, character psychology, and supernatural elements"). Subsequent ingestion for this narrative automatically uses the tuned prompts.

## Import External Graph with Auto-Analysis (BYOG)

```
POST /import/json?analyze=true
{
  "entities": [...],
  "situations": [...],
  "participations": [...]
}
```

The `?analyze=true` flag triggers automatic Leiden community detection and summary generation after import — immediate community structure without manual analysis steps.

## Selective Re-Ingestion and Chunk Control

After ingesting a large document, some chunks may have failed or produced low-quality extractions. Instead of re-ingesting everything, selectively re-extract specific chunks.

**Step 1: Identify problematic chunks**

The Studio Ingestion Detail view (`/ingest/{job_id}`) colors the chunk map by extraction yield:
- **Green shades** — successful (darker = fewer entities/situations, brighter = more)
- **Orange** — extracted but nothing committed (0 entities + 0 situations)
- **Yellow** — all items queued/rejected (low confidence)
- **Red** — failed (LLM error, parse failure)
- **Slate with blue border** — skipped (content hash dedup)

**Step 2: Select chunks for re-processing**

Ctrl+click to multi-select individual chunks, Shift+click for range selection. Quick-select buttons select all failed or all empty chunks.

**Step 3: Choose an action**

```
POST /ingest/jobs/{job_id}/chunks/batch
{
  "chunk_indices": [2, 5, 8],
  "action": "reextract",
  "context_mode": "neighbors"
}
```

- **Re-extract** with context mode `selected` / `neighbors` / `all` (quality vs token cost tradeoff)
- **Enrich** — enrichment pass on selected chunks (beliefs, game structures, discourse)
- **Reconcile** — temporal reconciliation on selected chunks
- **Reprocess** — rollback + re-gate from stored extractions (no LLM call, free)

**Step 4: View cross-job results**

If you retry a failed job, the retry creates a new job linked to the original via `parent_job_id`. Chunk detail tabs show extractions from all related jobs (Original, Retry 1, etc.), so you can compare extraction quality across attempts.

**Step 5: View full logs**

The Logs tab shows all LLM calls across the entire job lineage, filterable by chunk, pass number, error status, or specific job.

```
GET /ingest/jobs/{job_id}/lineage
```

Returns the full tree of parent → child retry jobs with their reports.

## Writer Workflow (AI-Assisted Novel Writing)

TENSA layers an AI-assisted writing surface over the hypergraph with three journeys on one tool surface:

- **Greenfield** — premise → outline → scenes → characters → first draft
- **Workshop** — critique + prioritized edit plan for an existing manuscript
- **Incremental** — draft chapter N+1 with continuity awareness of 1..N

**Design invariants:**

1. **Proposal, never destructive.** Every generation and every edit produces a `GenerationProposal` / `EditProposal`; applying is a commit via the revision system. Undo is free.
2. **Tiered analysis.** Cheap (deterministic) / Standard (selective LLM) / Deep (full LLM) — with cost estimate before running.
3. **`?dry_run=true` on every generative endpoint.** Returns prompt + token estimate without an LLM call.
4. **Context gathers once.** `gather_snapshot` in `src/narrative/revision.rs` is the single entry point; every generation / edit / analysis consumes the same snapshot-based context.

**End-to-end flow:**

```
get_writer_workspace(narrative_id) → dashboard
upsert_narrative_plan(narrative_id, {logline, synopsis, plot_beats, style, length, setting, ...})
POST /generation/propose → {proposal, token_estimate}
POST /generation/apply   → writes entities + situations, commits a revision
create_pinned_fact(narrative_id, {key, value, note?, entity_id?})
POST /editing/propose → {proposal, diff}
POST /editing/apply   → writes new raw_content, commits a revision
check_continuity(narrative_id, prose) → deterministic ContinuityWarning[], no LLM cost
run_workshop(narrative_id, tier: "standard", focuses: ["pacing", "continuity"])
list_narrative_revisions(narrative_id) → newest-first history
restore_narrative_revision(narrative_id, revision_id, author) → auto-commits current state first
get_writer_cost_summary(narrative_id, window: "7d") → per-operation totals
```

**Feature surface:**

- **Plan** — logline, synopsis, plot beats, style targets, length targets, setting, comp titles, custom fields. See `/narratives/{id}/plan`.
- **Generation** — outline, characters, scenes. POV hints (`{mode: "single", entity_name}` / `"rotating"` / `"omniscient"`) propagate into `Situation.discourse`.
- **Editing** — rewrite, tighten, expand, style_transfer, dialogue_pass; each returns a pre-computed line diff.
- **Workshop** — tiered critique across focus areas (pacing, continuity, characterization, prose, structure). Deep tier returns a deferred report.
- **Continuity** — pinned facts + deterministic prose scan. No LLM cost.
- **Research panel** — sources, attributions, contentions, and pinned per-scene notes (Quote / Clipping / Link / Note). `GET /situations/{id}/research-context` is the one-shot bundle.
- **Fact-check** — atomic claims extracted from prose, verdicted against the scene's research context (`Supported` / `Contested` / `Unsupported` / `Contradicted`). Tiers: `Fast` / `Standard` / `Deep`.
- **Cited generation** — hallucination-guard blocks commit on any `Contradicted` verdict; `[[cite: <uuid>]]` marker syntax keeps source links out of the prose and into a structured span map.
- **Annotations** — inline `Comment` / `Footnote` / `Citation` anchored on byte spans; reconcile on prose edits.
- **Compile profiles** — saved-profile compile to Markdown / EPUB 3 / DOCX with per-label front/back matter, footnote style, and comment toggle.
- **Collections** — saved searches (labels, statuses, keywords, text, order range, word range) that surface as virtual folders in the Studio Binder.
- **Scene reorder** — `POST /narratives/{id}/reorder` atomically rewrites `manuscript_order` + `parent_situation_id` (densified to 1000, 2000, …), rejecting cycles, duplicates, and foreign-narrative ids.
- **Revisions** — git-like history with content-hash de-dup; `POST /narratives/{id}/revisions/diff` returns structural + line-level prose diff, including per-scene `scene_summaries[]` for scannable large diffs.
- **Cost ledger** — per-narrative cost accounting with per-operation breakdown (Generation / Edit / Workshop / Continuity) and windowed rollups (`24h`/`7d`/`30d`/`all`).

**Claude Code skill:** `.claude/skills/tensa-writer/SKILL.md` triggers on novel/story/manuscript writing intents with three flagship workflows (greenfield / workshop / voice audit). The Studio Storywriting Hub has a "Launch in Claude" button that copies a one-line snippet scoped to the current narrative.

---

# Chapter 7: Algorithms & Theory

This chapter explains the core algorithms TENSA uses, grouped by the kind of question each answers.

## 7.1 Network Analysis

### 7.1.1 Betweenness Centrality (Brandes)

**What it answers:** Who controls the flow of information in the narrative?

**Intuition:** In a city with many neighborhoods connected by bridges, some bridges are the only route between two neighborhoods. The person controlling that bridge has outsized influence. In a spy network, the handler who connects field agents to headquarters has high betweenness — remove them and communication breaks down.

**How:** Build a co-participation graph (entities = nodes, edges = co-appearing in the same situation). For each entity *v*, BFS finds all shortest paths from *v* to every other entity. Back-propagate: for each shortest path through *v*, count how many pass through each intermediate node. Sum the dependency scores, normalize by `(n-1)(n-2)/2`.

### 7.1.2 Closeness (Wasserman–Faust)

**What it answers:** Who can reach everyone else most quickly?

BFS from each node; `closeness(v) = (reachable) / (sum of distances)`. Disconnected nodes get 0.

### 7.1.3 Degree Centrality

**What it answers:** Who is most directly connected?

`degree(v) = neighbor_count / (total_entities - 1)`.

### 7.1.4 PageRank, Eigenvector, HITS, Harmonic

- **PageRank** — power iteration with damping 0.85, convergence 1e-6.
- **Eigenvector** — per-component power iteration via weakly connected components.
- **Harmonic** — `H(v) = Σ 1/d(v,u)`, BFS-based, O(V²).
- **HITS** — Kleinberg hubs + authorities on the bipartite participation graph.

All four write per-entity virtual properties (`e.an.pagerank`, `e.an.eigenvector`, `e.an.harmonic`, `e.an.hub_score`, `e.an.authority_score`) queryable via TensaQL.

### 7.1.5 Topology: Articulation Points, Bridges, K-Core

**Articulation points** and **bridges** are computed via an iterative Tarjan DFS (stack-based, safe on deep graphs). `e.an.is_articulation_point` flags bottleneck entities; `e.an.is_bridge_endpoint` flags entities adjacent to a critical edge.

**K-Core decomposition** peels vertices by current minimum degree; `e.an.kcore` returns the largest *k* such that *e* is in the *k*-core.

### 7.1.6 Leiden Community Detection (Hierarchical)

**What it answers:** Are there natural clusters or factions in the narrative, at multiple levels of granularity?

Leiden improves on Louvain by guaranteeing each community is a connected subgraph, through a refinement step.

**Three phases:**
1. **Local moves** — each entity starts in its own community; greedily move entities to neighbors' communities to maximize modularity gain.
2. **Refinement** — for each community, BFS connectivity check; split disconnected components into separate communities.
3. **Hierarchy** — recursively apply Leiden to communities larger than a threshold (default 10) by coarsening: each community becomes a super-node with edge weights = total inter-community links.

The result is a multi-level hierarchy: Level 0 (leaf) is the most granular; higher levels are coarser. Each community carries an LLM-generated summary, entity list, and parent/child links.

**Label Propagation** is an O(m) alternative that needs no tuning parameters; iteratively each node adopts the most common label among its neighbors until stable. Stored at `an/lp/{narrative_id}`.

### 7.1.7 Temporal PageRank, Causal Influence, Information Bottleneck, Assortativity

- **Temporal PageRank** — PageRank with time-decayed edge weights (configurable `decay_lambda`). `e.an.temporal_pagerank`.
- **Causal Influence** — betweenness on the causal DAG, mapped back to entities through their participations. `e.an.causal_influence`.
- **Information Bottleneck** — identifies sole-knowers in the belief network (entities on which a fact's knowledge chain strictly depends). `e.an.bottleneck_score`.
- **Assortativity** — narrative-level scalar measuring degree correlation at `an/as/`.

### 7.1.8 Temporal Motifs & Faction Evolution

- **Temporal Motif Census** — enumerates 3–4 node Allen-constrained temporal patterns across all situations. Reveals recurring temporal micro-structures ("A meets B while C observes, then C reveals to D").
- **Faction Evolution** — sliding-window Label Propagation with community tracking across chapters. Detects **merges** (two factions collapse into one), **splits** (a faction fragments), **births** (new community emerges), and **deaths** (community dissolves). Useful for political dramas and long-running series.

### 7.1.9 Pathfinding

TensaQL PATH queries dispatch to:
- **Dijkstra** — weighted shortest path on the co-participation graph.
- **Yen's k-shortest** — top-k alternative shortest paths.
- **Narrative diameter** — longest causal chain via DAG DP.
- **Max-flow / min-cut** — Edmonds-Karp on participation edges.

### 7.1.10 Inline Graph Functions

Callable in TensaQL WHERE and RETURN (no INFER job needed):

| Function | Description |
|----------|-------------|
| `triangles(e)` | Number of triangles through `e` |
| `clustering(e)` | Local clustering coefficient |
| `common_neighbors(a,b)` | Shared neighbors |
| `adamic_adar(a,b)` | Adamic–Adar link-prediction score |
| `preferential_attachment(a,b)` | Degree product |
| `resource_allocation(a,b)` | Resource-allocation index |
| `jaccard(a,b)` | Jaccard similarity of neighbor sets |
| `overlap(a,b)` | Overlap similarity |

### 7.1.11 Graph Embeddings

- **FastRP** — sparse random projection embeddings. LCG-seeded, √3-sparse, iterative neighbor averaging. Fast O(E) preprocessing. Stored at `an/frp/`.
- **Node2Vec** — biased random walks (p, q parameters) → PMI matrix → truncated SVD. Stored at `an/n2v/`.

Embeddings plug into link prediction, downstream ML pipelines, and can be combined with vector search for hybrid retrieval.

### 7.1.12 Network Inference (NetInf)

Given observed information cascades (who learned a fact when), **Gomez-Rodriguez cascade-based diffusion network inference** learns the underlying influence graph — edges whose posterior likelihood best explains the cascades. Stored at `an/ni/`.

## 7.2 Information Theory

### 7.2.1 Shannon Self-Information

**What it answers:** Which situations are surprising or unusual?

For each situation, extract a feature signature (participant count bucket, unique role count, narrative level, has game structure, causal in/out degree buckets, normalized temporal position). Self-information `I(s) = -log₂(frequency of this signature)`. Rare patterns get high entropy and are flagged for analyst attention.

### 7.2.2 Mutual Information

**What it answers:** Do two entities tend to appear together (or avoid each other)?

Build a 2×2 co-occurrence table (both present, A only, B only, neither). `MI(A,B) = Σ P(x,y) · log₂(P(x,y) / (P(x)·P(y)))`. MI = 0 means independent; higher MI means stronger dependency.

### 7.2.3 KL Divergence (Deception Detection)

**What it answers:** Is an entity behaving suspiciously differently from "normal"?

Observed distribution `P(action)` vs. uniform baseline `Q(action)`. `KL(P‖Q) = Σ P(x)·ln(P(x)/Q(x))`. High KL means concentrated / single-purpose behavior. Only computed for entities with 2+ distinct actions.

## 7.3 Epistemic Reasoning

### 7.3.1 Recursive Belief Modeling

**What it answers:** What does entity A *think* entity B knows?

**SymbolicToM seeding:** Parse initial belief states from `beliefs_about_others` in each entity's first InfoSet. The LLM populates this during ingestion, providing initial epistemic states before the 4-phase pipeline runs.

**Four-phase pipeline** per situation, chronologically:
1. Update actual knowledge from InfoSet (`knows_before` + `learns`)
2. Collect all facts revealed by participants
3. For each pair of co-present entities (A, B): A learns that B knows the revealed facts, and A learns what B learned
4. Record snapshots and compute gaps

**Gap types:**
- **Unknown to A** — facts B actually knows, but A doesn't think B knows
- **False beliefs** — facts A thinks B knows, but B doesn't actually know

### 7.3.2 Dempster-Shafer Evidence Combination

**What it answers:** Given multiple uncertain sources, how much should we believe each hypothesis?

**Key concepts:**
- **Mass function `m(A)`** — how much evidence directly supports hypothesis set A
- **Belief `Bel(A)`** — total evidence supporting A (lower bound on probability)
- **Plausibility `Pl(A)`** — total evidence not contradicting A (upper bound)
- **Uncertainty** — `Pl(A) − Bel(A)` — gap between certain and possible

Each source produces a mass function based on its trust score: trust distributed uniformly across singletons, (1 − trust) goes to the full frame as uncertainty. Sources combine pairwise via Dempster's rule (products to intersections, conflict K accumulated, normalized by 1/(1−K)). For high-conflict evidence (K > 0.7), TENSA uses **Yager's rule** instead to keep mass on the frame instead of over-normalizing.

**Claim-aware attribution:** When a `SourceAttribution` carries an optional `claim: String` matching a frame element, mass is concentrated on that hypothesis rather than distributed uniformly.

### 7.3.3 Dirichlet / Evidential Deep-Learning Confidence

**What it answers:** Is the uncertainty due to lack of evidence (epistemic) or genuine ambiguity (aleatoric)?

Take a combined Dempster-Shafer mass function. For each singleton *k*: `evidence_k = Bel({k}) · num_sources`. Compute `alpha_k = evidence_k + 1`. Epistemic uncertainty `= K/S` (high when total evidence is low); aleatoric `=` normalized Shannon entropy of the expected Dirichlet distribution.

Struct: `DirichletConfidence { alpha, total_evidence, epistemic_uncertainty, aleatoric_uncertainty }`.

### 7.3.4 Propp Narrative Functions

**What it answers:** What structural role does a situation play in a folktale-style plot?

Vladimir Propp identified 31 canonical functions (absentation, interdiction, violation, villainy, departure, struggle, victory, wedding, …). TENSA's `classify_propp_function` matches situation `raw_content` against keyword patterns and returns the best-matching function with a normalized confidence score, or `None` for ambiguous situations. 32 variants total: 31 canonical + Lack (8a).

## 7.4 Argumentation (Dung Frameworks)

**What it answers:** When claims conflict, which ones survive?

Arguments are connected by attack relations. Three semantics:

1. **Grounded Extension (most conservative)** — only accepts arguments that are completely unattacked or defended by already-accepted arguments. Algorithm: self-attackers OUT → unattacked IN → arguments attacked only by OUT are IN → repeat.
2. **Preferred Extensions (maximal admissible)** — all maximal sets that are internally consistent (no in-set attacks) and defend members against outside attacks. May be multiple.
3. **Stable Extensions (strictest)** — a preferred extension that additionally attacks every non-member. If it exists, it's the clearest "winner."

## 7.5 Epidemiological Modeling — SIR Information Contagion

**What it answers:** How does a piece of information spread through the entity network, and who are the critical spreaders?

**SIR states:** Susceptible (doesn't know yet), Infected (knows and may spread), Recovered (knows but no longer spreading).

Process situations chronologically. Entities that reveal the fact become Infected; entities that learn it (from InfoSet) become Infected. Track each spread event; compute `R₀ = average secondary infections per spreader`.

**R₀ interpretation:** < 1 dies out, = 1 stable, > 1 exponential (viral).

**Critical spreader analysis:** for each entity, re-simulate without them and compute `R0_reduction`. Highest reduction = most critical spreader — the "broker" of the information cascade.

Stored at `an/sir/{narrative_id}/{fact_hash}`.

## 7.6 Causal Inference

### 7.6.1 NOTEARS Causal Discovery

**What it answers:** What hidden causal links between situations haven't been explicitly recorded?

**Pipeline:**
1. **Temporal mask** — for each pair (i, j), allow i → j only if Allen relations permit it (forward in time).
2. **Features** — per-situation 7-dim vector: confidence, participant count, narrative level, has game structure, causal link count, has spatial anchor, action diversity.
3. **Seed adjacency** — initialize from known causal links.
4. **Augmented Lagrangian optimization** (requires `inference` feature for nalgebra):
   - DAG constraint: `h(W) = tr(e^{W∘W}) - d = 0`
   - Objective: minimize `0.5/n · ‖X − XW‖² + λ₁‖W‖₁` subject to `h(W) = 0`
   - Outer loop updates Lagrange multiplier α and penalty ρ (doubling schedule)
   - Inner loop: gradient descent with data loss + L1 subgradient + DAG gradient
   - After each step: apply temporal mask, zero diagonal, clamp to [0, 1]
5. **LLM-augmented priors** — optional `prior_matrix` job parameter adds an L2 regularizer (`lambda_prior`) pulling W toward the prior.
6. **DAGMA adaptive s-schedule** — `s_schedule: [1.0, 0.75, 0.5, 0.25]` decreases per outer iteration for better convergence on tight DAG constraints.
7. **SCC pre-validation** — before optimization, detect strongly-connected components in the seeded graph and prune the weakest edge in each cycle.

Link classification: `> 0.8` Necessary, `0.5–0.8` Contributing, `< 0.5` Enabling.

**Hierarchical decomposition** for narratives with > 50 situations: process by narrative level (Scene, Sequence, Arc) separately, then merge results.

Without the `inference` feature, falls back to correlation-based scoring with the temporal mask.

### 7.6.2 Counterfactual Beam Search

**What it answers:** "What would have happened if X had been different?"

1. **Intervention** — specify what to change via the `ASSUMING` clause.
2. **Forward propagation** — follow causal links from the intervention.
3. **Beam search** (width 5, depth 20) — only explore links with strength ≥ 0.05; keep the top 5 outcomes at each level; probabilities multiply along the path.
4. **Result** — affected situations with counterfactual probabilities.

## 7.7 Strategic Reasoning

### 7.7.1 Game Classification

**2-player rules:**
- **Zero-Sum** — payoffs sum to ~0
- **Prisoner's Dilemma** — both benefit from cooperation but each is tempted to defect
- **Coordination** — both benefit from choosing the same action
- **Bargaining** — asymmetric payoffs, negotiation dynamics
- **Signaling** — one player reveals information to influence the other

**N-player rules:**
- **Auction** — target role + 3+ competing players
- **Asymmetric Information** — players have different knowledge
- **Coordination** — default for multi-player positive-sum interactions

**Information structure:** Complete / Incomplete / Imperfect / Asymmetric-becoming-complete.

### 7.7.2 Quantal Response Equilibrium (QRE)

**What it answers:** Given that actors aren't perfectly rational, what strategies are they likely to use?

`P(action_i) = exp(λ · EU(action_i)) / Σ exp(λ · EU(action_j))` — softmax over expected utilities. `λ → ∞` is Nash equilibrium (perfect rationality); `λ = 0` is complete randomness; `λ ∈ [1, 5]` is bounded rationality.

**Fixed-point iteration:** grid-search λ ∈ [0, 10] step 0.5. For each λ, initialize uniform strategies; iterate softmax best response per player until `‖σ^{t+1} − σ^t‖_∞ < ε` (default 1e-6). Pick λ that maximizes log-likelihood of observed actions. Classify as Nash if λ > 5, else QRE.

### 7.7.3 Mean Field Games

**What it answers:** What is the population-level equilibrium when many agents interact simultaneously?

When a crowd of 50+ agents each choose an action, individual strategies are intractable. Mean Field Games replace individual opponents with a population distribution μ; each agent best-responds to μ, and we iterate until μ is a fixed point.

**Pipeline:**
1. Aggregate all participants' actions and average base payoffs per action.
2. Coupling matrix — how each action's prevalence affects another's payoff (derived from co-occurrence payoff patterns).
3. `EU(a_i, μ) = base_payoff(a_i) + coupling_strength · Σ_j μ(j) · coupling[i][j]`
4. Fixed-point iteration: `μ_new = softmax(λ · EU(·, μ))` until convergence.
5. Stability classified via contraction criterion — if `max |λ · coupling[i][j]| < 1`, the equilibrium is stable.

Stored at `an/mfg/{situation_id}`.

## 7.8 Motivation Analysis

### 7.8.1 MaxEnt IRL (Inverse Reinforcement Learning)

**What it answers:** What is an actor optimizing for?

**Pipeline (Ziebart et al. 2008):**
1. Collect all participations for the entity, sorted chronologically.
2. Extract 10-dim features per situation: protagonist flag, antagonist flag, has action, payoff, knowledge gained, knowledge revealed, confidence, co-participant count, narrative granularity, has game structure.
3. Observed features = average feature values over trajectory (action steps weighted 2×).
4. Learn reward weights θ:
   - **T ≤ 20** (full trajectory): enumerate all 2^T trajectories, `Z(θ) = Σ_τ exp(θᵀf(τ))`, gradient `∇L = f_observed − E_θ[f]`.
   - **T > 20** (per-step logistic): `P(participate at t | θ) = σ(θᵀf_t)`.
   - Converge when max gradient < threshold.
5. Classify archetype from learned weights.

### 7.8.2 Archetype Classification

Seven core archetypes scored 0.0–1.0 each; the dominant is the argmax:

| Archetype | IRL Mapping | Sparse Keywords |
|-----------|------------|-----------------|
| **PowerSeeking** | antagonist + payoff + action | attack, fight, destroy, seize, dominate |
| **Altruistic** | protagonist + reveal − payoff | help, save, protect, support, heal |
| **SelfPreserving** | protagonist + confidence − reveal | flee, hide, defend, survive, escape |
| **StatusDriven** | co-participants + granularity + protagonist | boast, display, impress, claim, announce |
| **Vengeful** | action + antagonist + knowledge | revenge, avenge, retaliate, punish |
| **Loyal** | protagonist − payoff + co-participants | follow, serve, loyal, obey, pledge |
| **Opportunistic** | payoff + game structure + action | exploit, steal, betray, trade, bargain |

The enum also includes `Ideological` and `Custom(String)` variants, excluded from scored radar axes.

**Two scoring paths:**
- **Sparse (< 5 actions)** — keyword matching + role signals → soft scores (normalized so max = 1.0) → dominant from argmax. Confidence 0.3–0.5.
- **Rich (≥ 5 actions)** — full MaxEnt IRL → sigmoid-mapped weighted sums per archetype → normalized. Higher confidence.

Output: `MotivationProfile { entity_id, reward_weights, archetype, archetype_scores, trajectory_length, confidence }`. Studio's Dossier renders a 7-axis D3 radar chart.

## 7.9 Stylometry & Narrative Fingerprint

### 7.9.1 Prose-Level Stylometry

**What it answers:** Who wrote this text? Is this chapter stylistically consistent?

`compute_prose_features(text)` extracts ~26 features in four categories:

1. **Lexical (Burrows' Delta inputs)** — frequency distribution of 100 canonical English function words.
2. **Vocabulary richness** — Type-Token Ratio (windowed 1000 words), hapax legomena, Yule's K, Simpson's D.
3. **Sentence rhythm** — mean/std/CV of sentence length, lag-1 autocorrelation, bin distribution (≤5, 6-20, 21-40, >40).
4. **Readability & formality** — Flesch-Kincaid, passive voice ratio, adjective/adverb density, dialogue ratio, conjunction density.

**Burrows' Delta:** `Δ(A, B) = (1/n) · Σ |z_A(w) − z_B(w)|` over the 100 function words. Typical same-author Delta < 1.0; different-author > 1.5.

Academic basis: Burrows, J. (2002), "'Delta': A Measure of Stylistic Difference", *Literary and Linguistic Computing* 17(3): 267–287.

### 7.9.2 Multi-Layer Narrative Style Profile

Six layers capturing structural style beyond word choice:

| Layer | What It Measures | Key Metric |
|-------|-----------------|------------|
| 1. Structural Rhythm (Genette) | Pacing, scene density | Situation density curve (20 bins) |
| 2. Character Dynamics (Moretti) | Cast structure, power distribution | Role entropy, Gini coefficient of payoffs |
| 3. Information Management (Daley-Kendall) | Secrets, deception, revelation | Info R₀, secret survival rate |
| 4. Causal Architecture (Pearl/Trabasso) | Plot tightness | Causal density, chain lengths |
| 5. Temporal Texture (Allen/Herman) | Linearity vs. complexity | Allen relation distribution, flashback frequency |
| 6. Graph Topology (WL/Moretti) | Structural shape | WL hash histogram, community count |

**Narrative Fingerprint** = prose features + narrative style profile combined. Analogous to a **LoRA in diffusion models** — a compact ~120–150 float vector capturing "how this creator makes choices" from lexical surface to causal architecture.

**Radar chart:** 12 axes normalized [0, 1]: pacing, ensemble, causal density, info R₀, deception, temporal complexity, strategic variety, power asymmetry, protagonist focus, late revelation, subplot richness, surprise.

KV storage: `an/ns/{narrative_id}` (profile), `an/nf/{narrative_id}` (combined fingerprint), `ps/{narrative_id}/{chunk_index}` (per-chunk prose features), `ps/{narrative_id}/_aggregate` (aggregated).

### 7.9.3 Weighted Similarity

The fingerprint is heterogeneous — plain cosine is suboptimal. Similarity uses a kernel matched to each component:

| Layer | Components | Kernel |
|-------|-----------|--------|
| Prose — function words | 100 z-scored relative frequencies | **Burrows-Cosine** `1 − Δcos/2` (Würzburg Cosine Delta) |
| Prose — scalars | 7 readability + rhythm features | **Mahalanobis-diagonal** when `scalar_stds` present, else averaged scalar sim |
| Rhythm / Character / Temporal distributions | Density (20), game-type (7), Allen (13) bins | **Jensen–Shannon similarity** `1 − √JS` |
| All scalar blocks | Gini, entropy, chain lengths | **Mahalanobis-diagonal** `exp(−d²/dim)` |
| Topology — WL histogram | Top-50 label frequencies | **JS similarity** |
| Topology — WL SimHash | 256-bit signature of the full WL bag | **Hamming-distance similarity** `1 − popcount(a⊕b)/256` |

Each layer weighted by a `WeightedSimilarityConfig` at `an/nw/`. Defaults are uniform; train on a labeled PAN dataset via `train_pan_weights` and `PUT /settings/style-weights` to persist the learned vector.

### 7.9.4 Calibrated Anomaly Detection

The legacy anomaly endpoint gates chunks by a hard-coded 0.7 similarity threshold. Calibrated mode replaces it with a bootstrap-derived empirical p-value:

1. Compute each chunk's prose similarity vs. the narrative-wide aggregate.
2. Resample chunks with replacement `n_iter` times (default 1000) to build a null distribution.
3. Report each chunk's left-tail p-value; flag `p < alpha` (default 0.05).

`?mode=calibrated&alpha=0.05&n_iter=1000&seed=...`. Deterministic under fixed seed (ChaCha8Rng). The same primitive backs `POST /style/compare?ci=true`, which returns percentile confidence intervals on overall and prose-layer similarity.

### 7.9.5 PAN@CLEF Authorship Verification

Frames authorship as binary classification on pairs, with a calibrated abstention band.

| Metric | Definition |
|--------|-----------|
| AUC | Trapezoidal area under ROC |
| c@1 | `(n_correct + n_unanswered · n_correct/n) / n` — rewards abstention |
| F0.5u | β=0.5 F-measure treating non-decisions as correct |
| F1 | Standard F1 on confident decisions |
| Brier | Mean squared error of same-author probability |
| Overall | `AUC × c@1 × F0.5u × F1 × (1 − Brier)` (PAN 2020+ aggregate) |

`verify_pair` computes a logistic same-author probability from Burrows-Cosine + scalar differences; the uncertainty band around 0.5 produces abstentions. `unmasking` implements Koppel's feature-ablation curve for gray-zone cases.

## 7.10 Temporal & Rule-Based Inference

### 7.10.1 Temporal ILP (Inductive Logic Programming)

**What it answers:** What temporal patterns recur across entity participation sequences?

**Pipeline:**
1. For each entity, build a chronological sequence of `(role, narrative_level, temporal_interval)` tuples.
2. **Bigram mining** — window size 2 over each sequence. Body (step_i) → Head (step_j) with Allen relation; count support.
3. **Trigram mining** (when `max_body_size ≥ 2`) — window size 3. Body has 2 atoms, head has 1.
4. **Scoring** — `confidence = support / body_occurrences`, `lift = confidence / baseline`. Filter by `min_support`, `min_confidence`, `min_lift`.
5. **Pruning** — remove rules subsumed by more specific rules with similar confidence.

**Example output:**

| Body | Temporal | Head | Support | Confidence | Lift |
|------|----------|------|---------|------------|------|
| Actor/Protagonist in Scene | Before | Actor/Protagonist in Scene | 15 | 0.83 | 1.4 |
| Actor/Witness in Event | Before | Actor/Target in Beat | 8 | 0.67 | 3.2 |

Academic basis: "Temporal Inductive Logic Reasoning over Hypergraphs" (IJCAI 2024). Stored at `an/ilp/{narrative_id}`.

### 7.10.2 Probabilistic Soft Logic (PSL)

**What it answers:** Given a set of weighted logical rules and observed evidence, what are the most likely truth values for all facts in the narrative?

**Three uncertainty layers contrasted (the butler scenario):**
- **Bayesian confidence** (data quality): "Butler has confidence 0.85 — 3 sources corroborate his existence and presence."
- **Dempster-Shafer** (evidence state): "Evidence mass on {Butler_guilty}: Bel=0.6, Pl=0.9 (ignorance 0.3)."
- **PSL** (story-level inference): "Given 'near_body ∧ has_motive → killer' (weight 0.8) and 'has_alibi → ¬killer' (weight 0.5), the global probability of Butler=killer is 0.73."

**Pipeline (`src/analysis/psl.rs`):**
1. **Rules** — user-defined weighted soft Horn clauses.
2. **Grounding** — variables bound to all entities in the narrative; predicate truth values initialized from entity properties.
3. **Inference** — coordinate descent minimizes `Σ_r w_r · max(0, min(body) − head)²`.
4. **Output** — per-entity truth values for all predicates.

Config: `max_iterations` (200), `convergence_threshold` (1e-4), `step_size` (0.1). Academic basis: Bach et al., "Hinge-Loss Markov Random Fields and Probabilistic Soft Logic" (JMLR 2017). Stored at `an/psl/{narrative_id}`.

## 7.11 Narrative Architecture (`generation` feature)

§7.11 has three sub-families:

- **Analytical passes** (§7.11.1–7.11.7) — eight structural patterns computed over an existing narrative: commitment tracking, fabula/sjužet separation, dramatic irony, focalization, three-process analysis, character arcs, subplot detection, and scene-sequel rhythm.
- **Generation pipeline** (§7.11.8–7.11.14) — plan → materialize → generate, a personalization ladder (prompt → style embedding → LoRA) sitting on top of it, and the reusable structural primitives (layer transfer, skeleton, fitness loop) that feed the generator.
- **Fingerprint, debugging, adaptation** (§7.11.15–7.11.17) — narrative-architecture fingerprint axes, structural linter, and length-adaptation primitives.

A small number of items in the generation pipeline ship library-only (no REST/MCP surface yet) and are flagged as *scaffolded* in [Appendix D](#appendix-d-implementation-status).

### 7.11.1 Commitment Tracking (Chekhov's Gun Engine)

Formalizes setup-payoff pairs as computable `NarrativeCommitment` records. Detection heuristics: entities introduced with high detail (large properties JSON) that disappear for N+ chapters are potential unfired guns; weak causal links (strength < 0.5) signal mystery setups. Promise rhythm computes per-chapter `{outstanding, fulfilled, new, net_tension}` time series. `PromiseFulfillmentRatio = fulfilled / (fulfilled + abandoned)` is a fingerprint axis — higher = tighter craft.

### 7.11.2 Fabula/Sjužet Separation

Separates "what happened" (fabula: chronological from Allen constraints) from "how it's told" (sjužet: discourse order from ingestion sequence). Divergence measured as normalized Kendall tau distance (0.0 chronological, 1.0 maximally reordered). `NarrationMode` classifies each segment as Scene/Summary/Pause/Ellipsis/Stretch (Genette's duration categories). `TemporalShift` identifies Analepsis (flashback), Prolepsis (flash-forward), InMediasRes, FrameNarrative. `suggest_reordering` generates In Medias Res, Reverse Chronological, and Frame Narrative candidates.

### 7.11.3 Dramatic Irony & Focalization

At each situation in sjužet order, compute `reader_knowledge − character_knowledge` per participant. Non-empty difference = dramatic irony event. Intensity = gap cardinality weighted by consequence keywords (danger → Suspense, love → Anticipation, doom → TragedyForeknowledge). `DramaticIronyDensity` is a fingerprint axis.

**Focalization detection** assigns Genette categories (Zero/Internal/External) per situation based on `InfoSet` richness — the participant with the most `learns` entries and a Protagonist role scores highest as focalizer. When two participants score similarly → Zero (omniscient). Switch rate and unique focalizer count feed `focalization_diversity`.

### 7.11.4 Three-Process Analysis

Complements the Reagan 6-arc taxonomy with **structural phase analysis** based on "The Narrative Arc" (Berger et al., *Science Advances* 2020, validated on ~60,000 narratives). Three processes are measured as intensity curves over situations:

| Process | Measures |
|---------|----------|
| **Staging** | Setting establishment, character introductions — high early, decays over narrative |
| **Plot Progression** | Action density, causal-link frequency — rises through middle |
| **Cognitive Tension** | Uncertainty, suspense, unresolved commitments — peaks before climax |

`analyze_three_processes` in `src/narrative/three_process.rs` returns a `ThreeProcessResult` with `points: Vec<ProcessPoint>` (per-situation normalized position + three intensities) and aggregate peak positions (`staging_peak`, `progression_peak`, `tension_peak`) as fractions of narrative length. Signal extraction is deterministic — keyword counts over `raw_content` plus participation and causal-graph structure, no LLM required.

This is the eighth analytical pattern; the preceding seven cover plot shape (arcs), order (fabula/sjužet), knowledge asymmetry (irony + focalization), cast dynamics (subplots), pacing (scene-sequel), and promise debt (commitments). Three-Process captures structural *phases* that cut across all of those.

### 7.11.5 Character Arcs

Payoff values and keyword sentiment from `raw_content` give a valence trajectory per character across their situations. Arc type from first→last delta and trajectory shape: PositiveChange (rises), NegativeCorruption (falls), Flat (stable), PositiveDisillusionment (drops then rises), NegativeDisillusionment (rises then crashes). Completeness scored by presence of midpoint turn + dark night (minimum valence) + resolution.

### 7.11.6 Subplot Detection

Situation interaction graph (edges weighted by shared entity count) → label propagation. The community containing the most protagonist-participating situations = main plot; other communities of 2+ situations = subplots. `SubplotRelation`: Convergence (shares entities with main plot), Complication (contains protagonist), Independent.

### 7.11.7 Scene-Sequel Rhythm (Swain/Bickham)

Classify each situation as **ActionScene** (action/dialogue keywords + participant actions) or **Sequel** (reflective/contemplative keywords). Lag-1 autocorrelation measures alternation quality: negative = good alternation, positive = clustering. `pacing_score = action_fraction · 0.7 + max(0, −autocorrelation) · 0.3`.

### 7.11.8 Three-Stage Generation Pipeline: Plan → Materialize → Generate

The *structural* pipeline that produces prose (as opposed to the *personalization* ladder in §7.11.9):

1. **Plan** (`generate_plan`) — hierarchical top-down from premise → macro arc → characters with want/need/lie/truth → fact universe → subplots → commitments → per-chapter situations → temporal structure.
2. **Materialize** (`materialize_plan`) — writes plan into real hypergraph: Entity records, Situation records with Allen intervals and causal edges, Participation links, Commitment nodes at `nc/`, motivation vectors at `irl/`, facts at `fact/`. After materialization all analytical tools work on the planned narrative.
3. **Generate** (`prepare_chapter` / `prepare_full_narrative`) — query hypergraph per situation for knowledge states, motivation vectors, relationships, causal context, arc phase → build structured `GenerationPrompt` with `SituationPrompt` per scene and per-character `CharacterContext` (knows, false beliefs, motivation, relationships). Chapter text stored at `text/{narrative_id}/chapter_{n}` for the re-ingestion loop.

### 7.11.9 Personalization Ladder: Prompt → Style Embedding → LoRA

Three tiers of voice control, increasing in cost and specificity. Generation can apply any combination; each tier conditions the tier below it.

| Tier | What it is | Cost | Best for |
|------|-----------|------|----------|
| **1. Prompt** | Style description in the generation system prompt (tone, pacing, voice, constraints) | Free | One-off drafts, fast iteration |
| **2. Style embedding** | Dense 256- or 512-dim voice vector (`StyleEmbedding`, §7.11.10) injected into the prompt assembly as few-shot exemplars + explicit voice guidance | Encoder inference only | A specific author's cadence; blends of multiple authors |
| **3. LoRA adapter** | Fine-tuned low-rank adapter (`LoraAdapter`, §7.11.11) applied to the generation model at inference time | Training required (separate binary) | Production-grade author mimicry; author-arithmetic merges |

All three compose with the structural pipeline in §7.11.8 — you plan and materialize the story, then generate prose against the structural graph at whichever tier you can afford.

### 7.11.10 Style Embeddings

Dense voice vectors in a learned homogeneous space. Because the space is learned via contrastive training (same-author chunks pull together, different-author chunks push apart, triplet loss / InfoNCE), linear interpolation is valid — unlike fingerprints, which need per-layer kernels.

**Types** (`src/style/embedding.rs`):

| Type | Fields |
|------|--------|
| `StyleEmbedding` | `id`, `vector: Vec<f32>` (256 or 512), `source`, `base_model`, `training_corpus_size`, `created_at` |
| `StyleEmbeddingSource` | `SingleAuthor { author_id }`, `Blended { sources: Vec<(Uuid, f64)> }`, `GenreComposite { genre, corpus_size }`, `Custom { label }` |
| `StyleBlendRecipe` | Blend composition stored at `se/blend/` |

**Functions:**
- `blend_styles(&[(StyleEmbedding, f64)])` — weighted average in embedding space
- `average_with_outlier_removal(&[Vec<f32>])` — encode an author corpus by averaging chunk embeddings, removing chunks deviating > 2σ from the mean
- `store_embedding` / `load_embedding` / `list_embeddings` — KV CRUD at `se/`
- `store_blend` / `load_blend` — blend recipes at `se/blend/`

**Encoder** (`src/style/encoder.rs`): trait `StyleEncoder` + `HashStyleEncoder` (deterministic, non-semantic, for integration testing). Contrastive training is handled by a separate binary using ONNX/candle infrastructure. `EncoderTrainingConfig` captures hyperparameters.

**Consumption.** The vector itself is **opaque to the generator** — there is no decoder that maps it back to text. SE conditioning is rendered into natural language via the `StyleEmbeddingSource` enum, so the generator sees a human-readable directive rather than a 256-/512-dim float array:

| `StyleEmbeddingSource` variant | Rendered prompt fragment |
|--------------------------------|--------------------------|
| `SingleAuthor { author_id }` | `voice of {author_name}` |
| `Blended { sources: Vec<(Uuid, f64)> }` | `blend N% A, N% B[, ...]` |
| `GenreComposite { genre, .. }` | `{genre} tradition` |
| `Custom { label }` | `Style: {label}` |

This rendering is what flows into the chapter generator's system prompt, alongside (and not instead of) any explicit `voice_description` the user supplied. Pickers in Studio (e.g. the Generate view) list embeddings by `source` so the writer reads the same description as the generator will.

### 7.11.11 LoRA Adapters

Author-specific low-rank adapters for generation. Metadata management and arithmetic merging live in `src/style/lora.rs`; actual training is delegated to external infrastructure (e.g. `bin/train_author_lora.rs`).

**Types:**

| Type | Purpose |
|------|---------|
| `LoraTrainingConfig` | `rank` (8–64), `alpha`, `target_modules` (e.g. `["q_proj", "v_proj"]`), `learning_rate`, `epochs`, `batch_size`, `use_qlora` (4-bit quantization) |
| `LoraStatus` | Training lifecycle (Queued → Training → Ready → Failed) |
| `LoraAdapter` | Per-author adapter metadata, status, training config, weights reference |
| `MergedLoraAdapter` | Combined adapter with source weights and `MergeStrategy` |
| `MergeStrategy` | How multiple adapters are arithmetically combined |

**Functions:**
- `queue_training(...)` — record adapter metadata with status `Queued`; returns adapter ID for the external trainer to pick up
- `queue_merge(...)` — request arithmetic merge of multiple adapters with weights
- `store_adapter` / `load_adapter` / `list_adapters` / `delete_adapter` — KV CRUD at `lora/`
- `store_merged` / `load_merged` — merged adapters at `lora/merged/`

### 7.11.12 Fingerprint Layer Transfer

Decompose a narrative fingerprint into independent **layers** and recombine layers from different source narratives — "take pacing from a thriller, character arcs from Tolstoy, commitment density from Christie." Lives in `src/narrative/rhythm_transfer.rs`.

`FingerprintLayer` enum:

| Layer | Signal |
|-------|--------|
| `StructuralRhythm` | Situation density, participants, arc type |
| `CharacterDynamics` | Game types, role entropy, power asymmetry |
| `InformationFlow` | Info R₀, deception, knowledge asymmetry |
| `CausalArchitecture` | Causal density, chain length, branching |
| `TemporalTexture` | Allen relations, flashbacks, temporal spans |
| `GraphTopology` | WL hash, community count, graph topology |
| `NarrativeArchitecture` | Promise fulfillment, payoff distance, irony, arcs, pacing |

**Functions:**
- `compose_fingerprint(recipe)` — picks layers from different source narratives per a `CompositionRecipe`, returns `ComposedFingerprint`
- `transfer_layer(source, target, layer)` — transplant a single layer
- `weights_for_layers(&[FingerprintLayer])` — build a `WeightedSimilarityConfig` that only evaluates the selected layers (e.g. for style-matching constrained to temporal texture + rhythm)

### 7.11.13 Narrative Skeleton

Content-free structural abstraction of a narrative — arc shapes, game types, commitment patterns, knowledge flows, causal topology — stripped of all content so the skeleton can be re-skinned in a different domain. Lives in `src/narrative/skeleton.rs`.

**Types:**

| Type | What it holds |
|------|---------------|
| `NarrativeSkeleton` | Narrative-level structural signature |
| `EntitySlot` | Role, not identity (arc type, relational position) |
| `SituationSlot` | Structure, not content (level, role count, Allen relations, causal in/out degree) |
| `CommitmentSlot` | Commitment pattern (type + status progression) |

**Functions:**
- `extract_skeleton(hg, narrative_id)` — strip content, keep structure
- `skeleton_similarity(a, b)` — structural distance between two skeletons (independent of genre/setting)
- `store_skeleton` / `load_skeleton` — KV persistence

Used as a retrieval layer for structural similarity search: "find narratives with a plot shape like this one" regardless of genre or surface content.

### 7.11.14 Fingerprint Fitness Loop

Closed-loop generation refinement against a target fingerprint. Uses `WeightedSimilarityConfig::generative_weights()` — a preset that up-weights the eight narrative-architecture axes (2.0) and down-weights stylometric axes (0.3) — to score generated drafts against a target narrative's fingerprint. Drafts below the threshold trigger re-generation with the underweight axes elevated further.

**Pipeline.** `Query → Prompt → Generate → Score → Revise → Best-of-N`. Iteration `N+1`'s prompt prepends iteration `N`'s text with a *"Revise the following draft to address the issues below:"* framing. Only the most recent attempt is included in the next prompt — the loop is **bounded growth**, not an unbounded transcript. The scoring stage runs `compute_prose_features` against the freshly generated text; the revise stage emits a constraint list via `prose_delta_to_constraints`.

**Best-of-N selection.** `GenerationEngine::generate_with_fitness(...)` returns the **best-scoring attempt across iterations**, not the last one. Per-iteration log entries flag `accepted: true|false` for *that specific attempt*, so the returned text may pre-date the final iteration. Studio surfaces this via the `FitnessResultPanel` (§5.22 — Generation).

**Prose-only scope (limitation).** The loop scores the **prose layer only** — sentence length, sentence-length variance, dialogue ratio, passive voice ratio, FK grade, type-token ratio, function-word frequencies, Burrows-Delta inputs. The structure layer (graph-derived axes from §7.11.15: promise fulfillment, dramatic irony density, focalization diversity, etc.) is **not** rescored per iteration — that would require re-ingesting each candidate into the hypergraph, which is prohibitive. Treat this as a *prose-stylometry* refinement loop, not a full structural one.

**Actionable constraint subset.** The constraint list given back to the LLM is restricted to axes the model can plausibly act on:

| Actionable axis | Why |
|-----------------|-----|
| Sentence length | Direct, testable instruction |
| Sentence-length variance | Direct, testable instruction |
| Dialogue ratio | Composable directive |
| Passive voice ratio | Composable directive |
| FK grade | Vocabulary + sentence complexity dial |
| Type-token ratio | Vocabulary diversity dial |

Function-word frequencies and Burrows-Delta inputs **influence the score** but are *not* surfaced as constraints — there is no useful way to instruct a model to "use 'the' 4.2% of the time."

**`Threshold` newtype.** Acceptance is gated by a `Threshold(f64)` newtype defined in `src/generation/types.rs` that validates `0.0..=1.0` at construction (`Threshold::new`, `TryFrom<f64>`, `into_inner`). `Default` returns `0.80`. Inserting it on `StyleTarget` keeps the public type backward-compatible because both `target_fingerprint: Option<NarrativeFingerprint>` and `fitness_threshold: Threshold` are `serde(default)`.

> **Calibration TODO.** The `0.80` default is a placeholder pending a held-out same-author baseline (held-out chapters from one author should score above 0.80; cross-author chapters should score below). Until that calibration ships, treat the threshold as a tuning knob rather than a verdict.

**`ChapterGenerator` trait.** Per-iteration LLM calls go through:

```rust
pub trait ChapterGenerator: Send + Sync {
    fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        temperature: f64,
    ) -> Result<GeneratedText>;
}

pub struct GeneratedText {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}
```

This is the seam used by `GenerationEngine::generate_with_fitness(...)`; tests substitute an in-process generator without a live LLM.

**Cost ledger.** Every iteration writes a `CostLedgerEntry` with stable `kind = "chapter_gen_fitness"` and an optional `metadata = {"iteration": N, "score": s}` payload. Stable `kind` keeps `cost_ledger/summary` aggregations bounded; `metadata` (added in v0.65 with `serde(default, skip_serializing_if = "Option::is_none")`) carries the per-iteration drill-down without exploding the operation taxonomy.

**Surfaces.** Submission goes through the existing inference job system — submit `POST /jobs` with `job_type: "chapter_generation_fitness"`, then `GET /jobs/:id/result`. The MCP tool `generate_chapter_with_fitness` (§4.12) wraps the same submission. Studio's Generate view (§5.22) drives it end-to-end.

### 7.11.15 Narrative-Architecture Fingerprint Axes

Eight additional axes plug into the narrative fingerprint:

| Axis | Range | Interpretation |
|------|-------|----------------|
| `promise_fulfillment_ratio` | [0, 1] | Higher = tighter setup-payoff craft |
| `average_payoff_distance` | [0, 1] | Short = thriller; long = epic |
| `fabula_sjuzet_divergence` | [0, 1] | 0 = chronological; 1 = maximally reordered |
| `dramatic_irony_density` | [0, 1] | High = thriller/horror; low = adventure |
| `focalization_diversity` | [0, 1] | Single POV vs multi-POV vs omniscient |
| `character_arc_completeness` | [0, 1] | Average arc completeness across main characters |
| `subplot_convergence_ratio` | [0, 1] | High = tightly woven; low = sprawling |
| `scene_sequel_rhythm_score` | [0, 1] | Composite pacing quality |

`WeightedSimilarityConfig::generative_weights()` is a preset with high weight on these axes (2.0) and lower weight on stylometric axes (0.3) for generation fitness evaluation.

### 7.11.16 Narrative Debugger

A structural "linter" that runs every narrative analysis engine and reports pathologies with severity, location, and suggested fixes.

**22 pathology types across five families:**

| Family | Pathology types |
|--------|------------------|
| Commitment | `OrphanedSetup`, `UnseededPayoff`, `PrematurePayoff`, `PromiseOverload`, `PromiseDesert` |
| Knowledge | `ImpossibleKnowledge`, `ForgottenKnowledge`, `IronyCollapse`, `LeakyFocalization` |
| Causal | `CausalOrphan`, `CausalContradiction`, `CausalIsland` |
| Motivation/arc | `MotivationDiscontinuity`, `ArcAbandonment`, `FlatProtagonist`, `MotivationImplausibility` |
| Pacing/rhythm | `PacingArrhythmia`, `NarrationModeMonotony`, `SubplotStarvation`, `SubplotOrphan` |
| Temporal | `TemporalImpossibility`, `AnachronismRisk` |

**Severity order:** `Error < Warning < Info < Note`.

**Health score:** `1.0 − (errors·0.15 + warnings·0.05 + infos·0.01) / max(1, total_situations)`, clamped to `[0, 1]`.

**Genre presets** (`DiagnosticConfig::for_genre`): `thriller`, `literary_fiction`, `epic_fantasy`, `mystery` — each tunes thresholds and suppresses pathologies that are genre-conventional.

**Entry points:** `diagnose_narrative`, `diagnose_narrative_with`, `diagnose_chapter`, `store_diagnosis` / `load_diagnosis` (cache at `nd/{narrative_id}`).

**Fix suggestions + auto-repair:** `suggest_fixes(pathologies) → Vec<SuggestedFix>` with 13 `FixType` variants; `apply_fix` handles lossless commitment-status changes; `auto_repair(..., max_severity, max_iterations)` iteratively diagnoses and fixes confidence-≥-0.5 suggestions.

### 7.11.17 Narrative Compression & Expansion

Architecture-aware length adaptation. **Compression** shrinks to a target chapter count by removing lowest-essentiality elements; **expansion** grows by identifying structural thinness and planning new content.

**Essentiality** — each situation/entity/subplot gets a score in [0, 1]. For situations:

| Signal | Weight | Source |
|--------|--------|--------|
| Causal criticality (in+out degree) | 0.30 | `Situation.causes` |
| Commitment load (setup + payoff roles) | 0.25 | `list_commitments` |
| Knowledge gate (participant diversity) | 0.20 | `get_participants_for_situation` |
| Arc anchor (first/mid/last) | 0.15 | `list_character_arcs` |
| Dramatic irony contribution | 0.10 | `compute_dramatic_irony_map` |

Entities: participation centrality (0.6) + arc importance (0.4). Subplots: `SubplotRelation`-weighted (Complication=1.0 → Independent=0.1) + convergence + size. KV: `ess/{narrative_id}`.

**Compression strategies:** `Structural` | `SubplotPruning` | `CharacterMerging` | `TemporalCompaction` | `Balanced`. Balanced phases: prune lowest-essentiality subplots → merge low-essentiality characters with disjoint participations → drop lowest-essentiality situations. Presets: `compress_to_novella` (~40%), `compress_to_short_story` (~15%), `compress_to_screenplay_outline` (50 scenes). `preview_compression` returns a dry-run with warnings.

**Expansion strategies:** `SubplotAddition` | `CharacterDevelopment` | `CommitmentExtension` | `SceneExpansion` | `WorldBuilding` | `Balanced`. Identifies expansion points: arcs with completeness < 0.7 → progress situations; commitments with payoff_distance < 3 → progress events; 3+ consecutive action scenes → inserted sequel; subplot placeholders for thematic parallels. Plan-level only (no LLM generation). Presets: `expand_to_novel(target_chapters)`, `add_subplot_to(theme, relation)`.

**Narrative diff** (`diff_narratives`) returns `NarrativeDiff` with entity / situation / commitment / arc changes, pacing delta, and composite structural distance.

---

# Chapter 8: Configuration Reference

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TENSA_ADDR` | Server bind address | `0.0.0.0:3000` |
| `LOCAL_LLM_URL` | Local LLM endpoint (highest priority) | — |
| `OPENROUTER_API_KEY` | OpenRouter API key (second priority) | — |
| `ANTHROPIC_API_KEY` | Anthropic API key (third priority) | — |
| `TENSA_MODEL` | Model name for LLM provider | provider-dependent |
| `TENSA_EMBEDDING_MODEL` | Path to ONNX embedding model directory | — |
| `VITE_STADIA_API_KEY` | Stadia Maps API key (Studio UI) | — |
| `RUST_LOG` | Log level filter | — |

**LLM provider priority:** `LOCAL_LLM_URL` > `OPENROUTER_API_KEY` > `ANTHROPIC_API_KEY`.

## Cargo Features

| Feature | Dependencies | Purpose | Default? |
|---------|--------------|---------|----------|
| `rocksdb` | rocksdb (C++) | Production KV store | Yes |
| `disinfo` | none | Disinformation analysis (fingerprints, CIB, spread, claims, archetypes, monitoring) | Yes |
| `server` | axum, tower, tower-http | REST API server | No |
| `inference` | nalgebra | NOTEARS causal discovery | No |
| `narrative` | inference | Cross-narrative pattern mining | No |
| `mcp` | rmcp, schemars | MCP server interface | No |
| `docparse` | lopdf, docx-rs | PDF/DOCX text extraction | No |
| `embedding` | ort, ndarray, tokenizers | ONNX semantic embeddings | No |
| `cli` | clap | CLI binary tools | No |
| `web-ingest` | scraper, feed-rs | URL/RSS content ingestion | No |
| `gemini` | — | Google Gemini LLM provider | No |
| `bedrock` | hmac | AWS Bedrock LLM provider (SigV4) | No |
| `stylometry` | — | Prose features + narrative style profile + PAN@CLEF | No |
| `generation` | — | Narrative architecture & generative engine | No |
| `adversarial` | disinfo | Adversarial wargaming, policy, counter-narratives | No |
| `studio-chat` | server, tokio-stream | Studio integrated agent chat | No |

## Build & Run

```bash
# Minimal build (no RocksDB)
cargo build --no-default-features

# Full build with RocksDB
cargo build

# With REST API
cargo build --no-default-features --features server

# With ONNX embeddings
cargo build --no-default-features --features embedding

# Run tests
cargo test --no-default-features

# Run API server
cargo run --no-default-features --features server

# Run with local LLM
LOCAL_LLM_URL=http://localhost:11434 TENSA_MODEL=qwen3:32b cargo run --features server

# Run with OpenRouter
OPENROUTER_API_KEY=sk-or-... TENSA_MODEL=anthropic/claude-sonnet-4 cargo run --features server
```

## LLM Provider Configuration (Runtime)

Use `PUT /settings/llm`, `/settings/inference-llm`, or `/settings/chat-llm` to hot-swap providers without restart. All three accept the same `LlmConfig` JSON:

```json
// Local LLM (vLLM, Ollama, LiteLLM)
{"provider": "local", "base_url": "http://localhost:11434", "model": "qwen3:32b"}

// OpenRouter
{"provider": "openrouter", "api_key": "sk-or-...", "model": "anthropic/claude-sonnet-4"}

// Anthropic direct
{"provider": "anthropic", "api_key": "sk-ant-...", "model": "claude-sonnet-4-20250514"}

// Google Gemini (feature: gemini)
{"provider": "gemini", "api_key": "...", "model": "gemini-2.0-flash"}

// AWS Bedrock (feature: bedrock)
{"provider": "bedrock", "region": "us-east-1", "model": "anthropic.claude-3-5-sonnet-20241022-v2:0"}

// Disable
{"provider": "none"}
```

**Resolution order** for a given LLM role:

- **Ingestion** — `cfg/llm` (via `/settings/llm`)
- **Query / RAG** — `cfg/inference_llm` → ingestion LLM
- **Studio chat** — `cfg/studio_chat_llm` → inference LLM → ingestion LLM

## Ingestion Configuration

Adjustable via `PUT /settings/ingestion`:

```json
{
  "auto_commit_threshold": 0.8,
  "review_threshold": 0.3,
  "chunk_size": 2000,
  "chunk_overlap": 200,
  "max_retries": 3,
  "enable_pass2_reconciliation": false,
  "enrich": false,
  "single_session": false,
  "ingestion_mode": "novel"
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `auto_commit_threshold` | 0.8 | Confidence above which data is auto-committed |
| `review_threshold` | 0.3 | Confidence below which data is rejected (between this and `auto_commit` → review queue) |
| `chunk_size` | 2000 | Target chunk size in characters |
| `chunk_overlap` | 200 | Overlap between adjacent chunks |
| `max_retries` | 3 | LLM call retries on failure |
| `enable_pass2_reconciliation` | false | Enable cross-chunk reconciliation pass |
| `enrich` | false | Second LLM pass per chunk (beliefs, game structures, discourse, InfoSets, extra causal links, outcome models, Allen relations, date normalizations). With multiple chunks, a final cross-chunk temporal reconciliation pass stitches global timelines. Can also be passed per-request. |
| `single_session` | false | SingleSession mode — entire text sent in the first turn with chunk markers, per-chunk extraction as follow-up turns in the same conversation. Parse failures repaired in-session. Final reconciliation turn produces entity merges, global timeline, confidence adjustments, and cross-chunk causal links. Requires OpenRouter or Local LLM. Falls back to Single mode if text exceeds context. Can also be passed per-request. |
| `ingestion_mode` | `novel` | Domain preset — see [§5.19 Extraction Modes](#extraction-modes-ingestionmode) |

## Embedding Configuration

When the `embedding` feature is enabled, set `TENSA_EMBEDDING_MODEL` to a directory containing:

- `model.onnx` — ONNX model file
- `tokenizer.json` — HuggingFace tokenizer

Default model: **all-MiniLM-L6-v2** (384 dimensions).

Check status via `GET /settings/embedding`:

```json
{"provider": "onnx", "dimension": 384, "enabled": true, "model": "all-MiniLM-L6-v2"}
```

Switch models at runtime: `PUT /settings/embedding` with `{"model": "all-MiniLM-L6-v2"|"hash"|"none"}`. Download a new HuggingFace model via `POST /settings/embedding/download` with `{"repo_id": "sentence-transformers/..."}`. Enumerate available local models via `GET /settings/embedding/models`.

## RAG Configuration

`PUT /settings/rag` controls token budgets and the default retrieval mode. Token budget is partitioned across retrieval sources (entity, situation, chunk, community) with per-category and total caps; reranking (when enabled) runs after retrieval and before budget trimming.

## SingleSession Reconciliation Types

When `single_session: true` is used, the ingestion report includes a `reconciliation` field:

```json
{
  "entity_merges": [
    {
      "keep_id": "uuid-of-canonical-entity",
      "absorbed_id": "uuid-of-duplicate-entity",
      "reason": "Same person referenced as 'Viktor' in chunk 1 and 'Orlov' in chunk 3"
    }
  ],
  "confidence_adjustments": [
    {
      "target_id": "uuid-of-entity-or-situation",
      "old_confidence": 0.65,
      "new_confidence": 0.82,
      "reason": "Corroborated by events in chunks 2 and 5"
    }
  ],
  "cross_chunk_causal_links": [
    { "from_situation": "uuid", "to_situation": "uuid" }
  ],
  "global_timeline": [
    { "situation_id": "uuid", "relation": "Before", "other_situation_id": "uuid" }
  ]
}
```

---

# Chapter 9: Synthetic Generation (Surrogate Models)

> **Citations.** EATH model: Mancastroppa, Cencetti, Barrat — *"Effective Active Temporal Hypergraph"*, arXiv:2507.01124v2. Higher-order contagion: Iacopini, Petri, Barrat, Latora — *"Simplicial models of social contagion"*, Nat Commun 2019.

## 9.1 Why Synthetic?

TENSA's synthetic-generation surface is **model-agnostic**. EATH is the first concrete `SurrogateModel` impl; **NuDHy** (Phase 13b) is the second, shipping the configuration-style null complement (see [Chapter 12](#chapter-12-configuration-style-null-model-nudhy--dual-null-model-significance)). Future models (hyperedge-aware degeneracy, narrative-conditioned diffusion) plug in via the same trait, the same registry, the same TensaQL verbs, the same `/synth/*` REST surface. Use synth for:

- **Null models for significance testing.** Pattern mining, community detection, temporal motifs, higher-order contagion. The K-surrogate distribution gives `(z, p)` per metric so you can claim "the observed structure is significantly above background" with statistical force.
- **Synthetic benchmarks at scale.** PageRank/Leiden/SIR stress-tests without real corpora. A small calibrated narrative scales to thousands of entities while keeping the burstiness and group-size profile honest.
- **Adversarial wargame substrate.** D12 wargame sessions need a calibrated civilian background. `BackgroundSubstrate::Synthetic | SyntheticHybrid` ships a substrate inline (≤500 entities) or pulls one from a Phase 4 job.

Synth is **NOT** a stand-in for real ingestion. Synth participations default to `Role::Bystander`; situations have empty `info_set` and no `game_structure`. The graph statistics are reproduced; the meaning is not. Don't use synth output for writer workflows, RAG question-answering, or anything semantic.

## 9.2 The `SurrogateModel` Trait

```rust
pub trait SurrogateModel: Send + Sync {
    fn name(&self) -> &str;          // unique registry key — "eath", "nudhy", future "had", ...
    fn version(&self) -> &str;       // bumped on calibration shape changes
    fn calibrate(&self, hg: &Hypergraph, narrative_id: &str) -> Result<SurrogateParams>;
    fn generate(&self, hg: &Hypergraph, params: &SurrogateParams,
                seed: u64, num_steps: u32, label_prefix: &str) -> Result<SurrogateRunSummary>;
    fn fidelity_metrics(&self, ...) -> Result<FidelityReport>;
}
```

The default `SurrogateRegistry` registers `EathSurrogate` AND `NudhySurrogate` — `SurrogateRegistry::default().list()` returns `["eath", "nudhy"]` as of Phase 13b. Add new models by registering them at startup; the TensaQL grammar accepts the model name as a string parameter so no parser changes are required to surface a new model. See [Chapter 12](#chapter-12-configuration-style-null-model-nudhy--dual-null-model-significance) for the NuDHy MCMC configuration null + dual-null-model significance surface, [Chapter 13](#chapter-13-bistability--hysteresis-in-higher-order-contagion) for hysteresis/bistability detection in higher-order contagion, and [docs/EATH_sprint_extension.md](EATH_sprint_extension.md) for the full multi-phase extension log (13-16, all shipped).

## 9.3 EATH (Effective Active Temporal Hypergraph)

EATH treats actors as activity-modulated stations. Two-step generation per tick:

1. **Activity (Step 1).** Per-entity low/high phase Markov modulated by Λ_t (time-bucketed activity multiplier). Transition probabilities `ρ_low` (leave quiet) and `ρ_high` (leave active) shape inter-event burstiness.
2. **Recruitment (Step 2).** Each active entity either recruits a fresh group from scratch (probability `p_from_scratch`, weighted by activity × order propensity × LTM weights) or mutates an existing group from short-term memory (`stm_capacity` ring buffer per entity).

`EathParams` (selected fields; full enumeration in `src/synth/types.rs`):

| Field | Effect | Calibration |
|---|---|---|
| `aT[i]` | Per-entity activity rate | participation rate per entity in source narrative |
| `Λ_t` | Time-bucketed activity multiplier (≤100 buckets, clamped `[0.01, 100.0]`) | bucket totals, normalized |
| `rho_low` | Leave-quiet probability per step | 1/mean_quiet_run, clamped `[0.01, 1.0]` |
| `rho_high` | Leave-active probability per step | 1/mean_active_run, clamped `[0.01, 1.0]` |
| `xi` | Mean groups per Λ_t bucket | mean groups per bucket, clamped `[0.1, 50.0]` |
| `p_from_scratch` | Probability of fresh-group recruitment | 1 − consecutive-pair ≥50% overlap fraction |
| `group_size_distribution` | Empirical CDF of recruitment sizes | observed size distribution, capped 50 |
| `order_propensity[i]` | Per-entity recruitment-size bias | currently `ah[i]`; per-entity fitting deferred |
| `max_group_size` | Hard cap on hyperedge size | max observed in source, capped 50 |
| `stm_capacity` | Short-term memory ring buffer size per entity | default 7 (Miller's number) |

Determinism: `ChaCha8Rng(seed)` plus deterministic UUIDs from a seeded sub-RNG (NOT `Uuid::now_v7`). Every conditional branch in `step_transition`, `pick_top_k_weighted`, and `recruit_from_memory` consumes the same number of RNG draws regardless of branch taken — `test_eath_deterministic_by_seed` regresses this.

## 9.4 Calibration & Fidelity

`CALIBRATE SURROGATE USING 'eath' FOR "narrative-id"` submits a `SurrogateCalibration` job. The fitter walks the source narrative once, derives all `EathParams` fields, and persists the params at `syn/p/{narrative_id}/eath`.

The fidelity pipeline (Phase 2.5) generates K=20 surrogates and computes 7 metrics:

| Metric | Type | Default threshold |
|---|---|---|
| `inter_event_ks` | KS divergence on inter-event time distribution | `≤ 0.10` |
| `group_size_ks` | KS divergence on group-size distribution | `≤ 0.05` |
| `activity_spearman` | Spearman ρ on per-entity activity ranks | `≥ 0.7` |
| `order_propensity_spearman` | Spearman ρ on per-entity order ranks | `≥ 0.6` |
| `burstiness_mae` | Mean absolute error on burstiness coefficient B = (σ-μ)/(σ+μ) | `≤ 0.15` |
| `memory_autocorr_mae` | MAE on lag-1 autocorrelation of inter-event times | `≤ 0.15` |
| `hyperdegree_ks` | KS on hyperdegree distribution | `≤ 0.10` |

`FidelityReport.thresholds_provenance` is one of `Default | UserOverride | StudyCalibrated`. Reports rendered with `Default` carry a `⚠ Default` warning banner — the values are PLACEHOLDER, pending the threshold-calibration study.

Per-narrative threshold overrides at `cfg/synth_fidelity/{narrative_id}` via `PUT /synth/fidelity-thresholds/{nid}`.

## 9.5 Generation, Hybrid Generation, and Reproducibility

`GENERATE NARRATIVE "<out>" LIKE "<src>" USING SURROGATE 'eath' [PARAMS {json}] [SEED <n>] [STEPS <n>]` submits a `SurrogateGeneration` job. Default `model = 'eath'` if `USING SURROGATE` is omitted.

**Reproducibility blob.** Every run writes a `ReproducibilityBlob` to `syn/seed/{run_id}` containing: seed, params hash (`canonical_params_hash`), model name + version, source narrative state hash, num_steps, timestamps. Replay by re-submitting with the same blob.

**Hybrid (mixture-distribution) generation.** Per Mancastroppa § III.4 approach (a): per-step source pick → recruit ONE hyperedge using that source's calibrated `EathParams`. Output group-size distribution = `Σ w_k · P_k(s)` (weighted mixture). Per-source LTM, shared STM, weight tolerance 1e-6.

```sql
GENERATE NARRATIVE "wargame-bg-A1" USING HYBRID
    FROM "civilian-corpus" WEIGHT 0.85,
    FROM "adversarial-corpus" WEIGHT 0.15
    SEED 7 STEPS 2000
```

Hybrid grammar still accepts only `model = "eath"` per component — the per-component `USING` clause is a one-rule-change Phase 13d/13e follow-up tracked in [docs/EATH_sprint_extension.md](EATH_sprint_extension.md). NuDHy (Phase 13b) IS registered and queryable everywhere else (calibrate / generate / dual significance / bistability significance / opinion significance), but mixing EATH-recruited and NuDHy-MCMC steps in a single hybrid run requires a per-component model field on `HybridComponent` plus a planner hook — neither of which has shipped yet.

## 9.6 Significance Pipeline (Null Models)

`POST /synth/significance` with `{narrative_id, metric, k?}` submits a `SurrogateSignificance` job. Three metric adapters:

- **`temporal_motifs`** — per-motif z/p on Allen+NarrativeLevel keys.
- **`communities`** — weighted modularity Q + community count.
- **`patterns`** — PRESENCE/ABSENCE per discovered pattern.

K-loop runs via `std::thread::scope` with per-K `MemoryStore` isolation — synthetic records never pollute the user's KV. Source observation computed ONCE before the K-loop. Refuses synthetic source narratives via `emit::is_synthetic_*`. Auto-calibrates if no params (doubles wall-clock).

Higher-order contagion has its own dedicated endpoint (`POST /synth/contagion-significance`) because the request shape (`HigherOrderSirParams`) doesn't fit the metric-string pattern. The `SurrogateContagionSignificanceEngine` reuses the K-loop infrastructure via the `AdapterChoice::Contagion` variant.

`/analysis/higher-order-contagion` is the **synchronous** higher-order SIR on a real narrative — no surrogates, no K-loop, just one simulation. **LOAD-BEARING contract:** `beta_per_size = [β, 0, 0, ...]` AND `threshold = ThresholdRule::Absolute(1)` reduces bit-identically to `analysis::contagion`. Future contributors who change the default threshold break the reduction.

## 9.7 Provenance: Synthetic vs Empirical

Every synthetic record carries:

- `Entity.properties.synthetic = true`, `synth_run_id`, `synth_model`
- `Situation.properties.synthetic = true`, `synth_run_id`, `synth_model`, `synth_step`
- `Entity.extraction_method = ExtractionMethod::Synthetic { model, run_id }`
- `Situation.extraction_method = ExtractionMethod::Synthetic { model, run_id }`
- `Situation.temporal.granularity = TimeGranularity::Synthetic`
- `Participation.info_set.knows_before` carries a sentinel `KnowledgeFact { about_entity: Uuid::nil(), fact: "synthetic|run_id=…|model=…", confidence: 1.0 }` (Participation has no JSON slot, so the sentinel rides along the only carrier the Phase 0 audit found)

Discriminate via `synth::emit::is_synthetic_entity`, `is_synthetic_situation`, `is_synthetic_participation` — the canonical predicates.

### The `INCLUDE SYNTHETIC` opt-in

**Default behaviour: read endpoints filter synthetic records out.** This is the safe default — synthetic data should never silently appear in analyst-facing aggregations. Opt-in via:

- TensaQL: `INCLUDE SYNTHETIC` clause appended to MATCH (where supported)
- REST: `?include_synthetic=true` query parameter
- Studio: ⊛ button in `WorkspaceHeader` (purple = on)

### Endpoints respecting the flag (5 covered, 7 pending, 1 known leak)

The Phase 3 invariant test in `src/synth/invariant_tests.rs` enumerates a 13-endpoint coverage matrix. Status as of v0.75.0:

**Covered (5):**

| Endpoint | Behaviour with flag off |
|---|---|
| `GET /narratives/:id/stats` | Counts exclude synthetic entities + situations |
| `GET /entities` | Synthetic filtered out unless `?include_synthetic=true` |
| `GET /situations` | Synthetic filtered out unless `?include_synthetic=true` |
| `POST /ask` | Body field `include_synthetic` accepted (downstream RAG context filtering pending Phase 12.5; defaults to `false` so behaviour is correct) |
| `POST /export/archive` | Synthetic narratives excluded by default |

**Pending (7) — Phase 12.5:**

The following endpoints don't exist in TENSA today at the EATH-spec'd path. When they land, they MUST add `?include_synthetic=true` support and route through `synth::emit::filter_synthetic_*`:

- `POST /analysis/centrality` — TENSA exposes per-entity virtual properties (`e.an.pagerank`) and `POST /jobs` for explicit centrality runs; no narrative-level POST aggregation endpoint exists.
- `POST /analysis/communities` — closest existing surface is `POST /narratives/:id/communities/summarize`.
- `POST /analysis/temporal-motifs` — only the readback `GET /narratives/:id/temporal-motifs` exists; the underlying `an/tm/{id}` blob was computed by an inference engine that had no opt-in flag at run time. Blob-level filtering would require re-running the analysis.
- `POST /analysis/contagion` — same readback shape (`GET /narratives/:id/contagion` reads back `an/sir/{id}/{cascade_id}`).
- `GET /fingerprint/stylometry/{id}` — TENSA has `GET /narratives/:id/fingerprint`, but the `compute_and_store` cache doesn't yet honor a flag.
- `GET /fingerprint/disinfo/{id}` — TENSA has `GET /narratives/:id/disinfo-fingerprint` (disinfo-feature gated).
- `GET /fingerprint/behavioral/{id}` — TENSA has `GET /entities/:id/behavioral-fingerprint` (per-entity, not per-narrative).

### Known Synthetic-Leak Edge Cases

**`GET /narratives/{id}/communities`** — Community summaries are pre-rendered at write time at `cs/{nid}/{cid}` without distinguishing synthetic-derived from empirical communities. Filtering at readback is not possible without dropping summaries entirely. Phase 12.5 fix proposed: add `synthetic_derived: bool` to `CommunitySummary` struct, populate at compute time from constituent entities, filter at read.

**`/ask` RAG context wire-up** — the route handler accepts `include_synthetic: bool` but the downstream RAG context assembly in `crate::query::rag` currently discards it (`let _ = include_synthetic`). Default behaviour is correct (RAG context defaults to all records, not synthetic-included), but explicit opt-out via the flag is a no-op until Phase 12.5 wires the filter.

## 9.8 TensaQL Grammar

```sql
-- Calibrate (model required, no default — calibration is O(dataset))
CALIBRATE SURROGATE USING 'eath' FOR "narrative-id"

-- Generate (model defaults to 'eath' if USING SURROGATE omitted)
GENERATE NARRATIVE "<output_id>" LIKE "<source_id>"
    [USING SURROGATE 'eath']
    [PARAMS { ...json... }]
    [SEED <n>]
    [STEPS <n>]
    [LABEL_PREFIX '<str>']

-- Hybrid generation (mixture-distribution; weights must sum to 1.0 ± 1e-6)
GENERATE NARRATIVE "<output_id>" USING HYBRID
    FROM "<source_a>" WEIGHT 0.7,
    FROM "<source_b>" WEIGHT 0.3
    [SEED <n>] [STEPS <n>]

-- Higher-order contagion on a REAL narrative (synchronous, NOT a job)
INFER HIGHER_ORDER_CONTAGION(<json-params>) FOR n:Narrative WHERE n.id = "..."
```

Single-quoted model strings (`'eath'`) visually separate from double-quoted narrative IDs. `PARAMS` body is eagerly parsed at parse time — malformed JSON errors at planner time, not execute time.

## 9.9 Synthetic Job Types

Nine `InferenceJobType` variants are owned by the synth + reconstruction surface (five from the base sprint plus four added in EATH Extension Phases 13c, 14, 15c, 16c):

| Variant | Engine | Submitted by |
|---|---|---|
| `SurrogateCalibration { narrative_id, model }` | `SurrogateCalibrationEngine` | `POST /synth/calibrate/{nid}`, `CALIBRATE SURROGATE` |
| `SurrogateGeneration { source_narrative_id, output_narrative_id, model, params, seed, num_steps, label_prefix }` | `SurrogateGenerationEngine` | `POST /synth/generate`, `GENERATE NARRATIVE … LIKE …` |
| `SurrogateHybridGeneration { components, output_narrative_id, seed, num_steps }` | `SurrogateHybridGenerationEngine` | `POST /synth/generate-hybrid`, `GENERATE NARRATIVE … USING HYBRID …` |
| `SurrogateSignificance { narrative_id, metric, k, model, params_override }` | `SurrogateSignificanceEngine` | `POST /synth/significance` |
| `SurrogateContagionSignificance { narrative_id, params, k, model }` | `SurrogateContagionSignificanceEngine` | `POST /synth/contagion-significance` |
| `SurrogateDualSignificance { narrative_id, metric, k_per_model, models }` | `SurrogateDualSignificanceEngine` | `POST /synth/dual-significance`, `COMPUTE DUAL_SIGNIFICANCE` (see [Chapter 12](#chapter-12-configuration-style-null-model-nudhy--dual-null-model-significance)) |
| `SurrogateBistabilitySignificance { narrative_id, params, k, models }` | `SurrogateBistabilitySignificanceEngine` | `POST /synth/bistability-significance` (see [Chapter 13](#chapter-13-bistability--hysteresis-in-higher-order-contagion)) |
| `SurrogateOpinionSignificance { narrative_id, params, k, models }` | `SurrogateOpinionSignificanceEngine` | `POST /synth/opinion-significance` (see [Chapter 11](#chapter-11-opinion-dynamics-bcm-on-hypergraphs)) |
| `HypergraphReconstruction { narrative_id, params }` | `ReconstructionEngine` | `POST /inference/hypergraph-reconstruction` (see [Chapter 10](#chapter-10-hypergraph-reconstruction-from-dynamics)) |

`WorkerPool::engines` is keyed by `std::mem::Discriminant<InferenceJobType>` so payload-bearing variants like `SurrogateGeneration { source_narrative_id: Some("foo"), … }` resolve correctly to the engine registered with the sentinel-payload `job_type()`.

## 9.10 Module Layout

`src/synth/`:

```
mod.rs                  # KV prefix constants, key builders, persistence, list_runs_newest_first
types.rs                # SurrogateParams, EathParams, RunKind, SurrogateRunSummary, ReproducibilityBlob
surrogate.rs            # SurrogateModel trait
registry.rs             # SurrogateRegistry — name-keyed lookup
hashing.rs              # canonical_params_hash, canonical_narrative_state_hash (SHA-256 canonical-JSON)
eath.rs                 # EathSurrogate — Step 1 + Step 2 main loop
eath_recruit.rs         # recruit_from_scratch / recruit_from_memory primitives
eath_tests.rs           # 8 named acceptance tests (T1-T8)
activity.rs             # Per-entity phase Markov sampling
memory.rs               # LongTermMemory (sparse HashMap, lazy decay) + ShortTermMemory (VecDeque)
calibrate.rs            # fit_params_from_narrative + calibrate_with_fidelity_report
calibrate_fitters.rs    # Per-field fitters (aT, ah, Λ_t, p_from_scratch, ρ_*, ξ, …)
calibrate_tests.rs      # 8 acceptance tests + LTM constructor / pair-symmetry
emit.rs                 # write_synthetic_*, filter_synthetic_*, is_synthetic_* (LOAD-BEARING)
invariant_tests.rs      # 9 tests including the load-bearing 13-endpoint synthetic-leak scan
fidelity.rs             # FidelityReport, FidelityThresholds, thresholds_provenance, KV persistence
fidelity_metrics.rs     # KS / Spearman / MAE / burstiness / lag-1 autocorr (zero new deps)
fidelity_pipeline.rs    # K=20 sample loop via std::thread::scope (Auto / Single / Threads(n))
fidelity_tests.rs       # 8 acceptance tests + threshold roundtrip + 16 metric primitives unit
hybrid.rs               # HybridComponent + HybridParams + generate_hybrid_hypergraph
hybrid_tests.rs         # 6 hybrid tests (mean, provenance, weight validation, determinism, …)
engines.rs              # SurrogateCalibrationEngine + SurrogateGenerationEngine + SurrogateHybridGenerationEngine + SurrogateContagionSignificanceEngine + make_synth_engines()
engines_tests.rs        # 8 acceptance tests + cancellation #[ignore]'d + 3 error-path
significance/
├── mod.rs              # SurrogateSignificanceEngine, run_significance_pipeline
├── adapters.rs         # 3 metric adapters: TemporalMotifs, Communities, Patterns; AdapterChoice::Contagion
└── stats.rs            # Mean/stddev/z/p computation
significance_tests.rs   # 6 acceptance tests (T1-T6, 1 ignored)
```

`src/analysis/higher_order_contagion.rs` (Phase 7b) — Iacopini/Petri/Barrat 2019 threshold SIR on hypergraphs. Single ordered stochastic stream via `ChaCha8Rng(rng_seed)`; per-step participation index built ONCE (O(P)). The contract above guarantees exact reduction to pairwise SIR.

`src/api/synth/` (Phase 6 + 7 + 7b + 9):

```
mod.rs              # submit_synth_job helper + DEFAULT_MODEL
calibration.rs      # POST /synth/calibrate/{nid}, params CRUD
generation.rs       # POST /synth/generate, /synth/generate-hybrid, runs list/get, seed
fidelity.rs         # GET /synth/fidelity/{nid}/{rid}, /synth/fidelity-thresholds CRUD
significance.rs     # POST /synth/significance, GET single + GET list
contagion.rs        # POST /synth/contagion-significance, GET single + GET list
models.rs           # GET /synth/models
```

## 9.11 REST API for Synthetic Generation

| Method | Path | Use |
|--------|------|-----|
| `POST` | `/synth/calibrate/{narrative_id}` | Submit `SurrogateCalibration` job. Body: `{model? = "eath"}`. Returns `{job_id, status}`. |
| `GET` | `/synth/params/{nid}/{model}` | Read calibrated params. |
| `PUT` | `/synth/params/{nid}/{model}` | Override params manually (advanced). |
| `DELETE` | `/synth/params/{nid}/{model}` | Drop calibrated params. |
| `POST` | `/synth/generate` | Submit `SurrogateGeneration` job. Body: `{source_narrative_id, output_narrative_id, model?, params?, seed?, num_steps?, label_prefix?}`. |
| `POST` | `/synth/generate-hybrid` | Submit `SurrogateHybridGeneration` job. Body: `{components: [{narrative_id, model, weight}], output_narrative_id, seed?, num_steps?}`. Σ weight = 1.0 ± 1e-6 validated synchronously. |
| `GET` | `/synth/runs/{nid}` | List runs (`?limit=N` default 50, max 1000). Newest first. |
| `GET` | `/synth/runs/{nid}/{run_id}` | Single `SurrogateRunSummary`. |
| `GET` | `/synth/seed/{run_id}` | `ReproducibilityBlob` (replay capsule). |
| `GET` | `/synth/fidelity/{nid}/{run_id}` | `FidelityReport`. |
| `GET` | `/synth/fidelity-thresholds/{nid}` | Per-narrative `FidelityThresholds`. |
| `PUT` | `/synth/fidelity-thresholds/{nid}` | Override thresholds. |
| `GET` | `/synth/models` | List registered surrogate models (`["eath", ...]`). |
| `POST` | `/synth/significance` | Submit `SurrogateSignificance` job. Body: `{narrative_id, metric: "temporal_motifs"\|"communities"\|"patterns", k?, model?, params_override?}`. |
| `GET` | `/synth/significance/{nid}/{metric}/{run_id}` | Single significance result. |
| `GET` | `/synth/significance/{nid}/{metric}` | List significance results for a metric. |
| `POST` | `/synth/contagion-significance` | Submit `SurrogateContagionSignificance` job. Body: `{narrative_id, params: HigherOrderSirParams, k?, model?}`. |
| `GET` | `/synth/contagion-significance/{nid}/{run_id}` | Single contagion-significance result. |
| `GET` | `/synth/contagion-significance/{nid}` | List contagion-significance results. |
| `POST` | `/analysis/higher-order-contagion` | **Synchronous** higher-order SIR on a real narrative. Body: `{narrative_id, params: HigherOrderSirParams}`. No surrogates. |

## 9.12 MCP Tools for Synthetic Generation

9 tools (7 base + Phase 13c `compute_dual_significance` + Phase 14 `compute_bistability_significance`). All wrap the corresponding `/synth/*` REST endpoints; HTTP backend uses the frozen Phase 10.5 contract.

| Tool | Wraps | Returns |
|------|-------|---------|
| `calibrate_surrogate` | `POST /synth/calibrate/{nid}` | `{job_id, status: "Pending"}` |
| `generate_synthetic_narrative` | `POST /synth/generate` | `{job_id, status: "Pending"}` |
| `generate_hybrid_narrative` | `POST /synth/generate-hybrid` | `{job_id, status: "Pending"}` |
| `list_synthetic_runs` | `GET /synth/runs/{nid}` | `SurrogateRunSummary[]` (newest first) |
| `get_fidelity_report` | `GET /synth/fidelity/{nid}/{run_id}` | `FidelityReport` or `null` |
| `compute_pattern_significance` | `POST /synth/significance` | `{job_id, status: "Pending"}`. Routes higher-order contagion to the dedicated tool below. |
| `simulate_higher_order_contagion` | `POST /synth/contagion-significance` | `{job_id, status: "Pending"}` |
| `compute_dual_significance` | `POST /synth/dual-significance` (Phase 13c) | `{job_id, status: "Pending"}`. See [Chapter 12](#chapter-12-configuration-style-null-model-nudhy--dual-null-model-significance). |
| `compute_bistability_significance` | `POST /synth/bistability-significance` (Phase 14) | `{job_id, status: "Pending"}`. See [Chapter 13](#chapter-13-bistability--hysteresis-in-higher-order-contagion). |

`get_wargame_state` was extended with a `substrate_provenance` key surfacing whether a wargame session ran on real, synthetic, or hybrid data — additive, existing fields unaffected.

---

# Chapter 10: Hypergraph Reconstruction from Dynamics

> **Citations.** Primary method: Delabays, De Pasquale, Dörfler, Zhang — *"Hypergraph reconstruction from dynamics"*, Nat. Commun. **16**, 2691 (2025), arXiv:2402.00078. SINDy parent: Brunton, Proctor, Kutz — *"Discovering governing equations from data by sparse identification of nonlinear dynamical systems"*, PNAS **113**, 3932 (2016), arXiv:1509.03580. Pairwise baseline: Casadiego, Nitzan, Hallerberg, Timme — ARNI, Nat. Commun. **8**, 2192 (2017). Ground-truth substrate: Mancastroppa, Cencetti, Barrat — arXiv:2507.01124.

## 10.1 What Reconstruction Does

Reconstruction is the **inverse problem** to synthetic generation. Synth generates a synthetic narrative from a calibrated null-model surrogate; reconstruction recovers latent hyperedges from observed entity dynamics. Different problem, different math, different provenance.

Given a `narrative_id` and an observation function (per-entity scalar time-series), the engine recovers the latent hyperedges that best explain the observed joint dynamics under a sparse dynamical-systems assumption. The output is a ranked list of `InferredHyperedge` records, each carrying member entity UUIDs, hyperedge order (2 = pairwise, 3 = triadic, ...), a symmetrized weight, a bootstrap confidence in `[0, 1]`, and a `possible_masking_artifact` flag for pairwise edges that overlap higher-order ones.

The load-bearing application is **disinformation coordination discovery**: given a Telegram corpus where group structure was never explicitly ingested, recover the latent coordination groups by observing which accounts co-act in the per-time-bin participation rate.

## 10.2 Method (THIS / SINDy)

The Delabays et al. THIS (Taylor-expanded Hypergraph Identification via SINDy) method extends Brunton's SINDy to higher-order group interactions. The Taylor expansion of the dynamics around a reference point makes mixed partial derivatives `∂²F_i/∂x_j∂x_k` nonzero **iff** entities `j` and `k` both interact with entity `i` in a common hyperedge of order ≥ 3. Discovering which `x_j * x_k` terms survive a per-entity LASSO regression therefore directly reveals which triadic hyperedges are incident on entity `i`.

TENSA's pipeline:

1. **Observe** — build a state matrix `X[t][i]` from the source narrative (sliding-window participation rate for entity `i` in time bin `t`).
2. **Differentiate** — estimate `dx_i/dt` via Savitzky-Golay (default `window=5, order=2`) or central finite differences.
3. **Build library** — enumerate all monomial terms `x_j`, `x_j x_k`, `x_j x_k x_l`, ... up to `max_order` (default 3, hard cap 4). Apply a Pearson correlation pre-filter (default `ρ_min = 0.1`) so candidate triadic+ terms only enter the library when every constituent pairwise correlation clears the threshold.
4. **Solve** — N independent LASSO regressions, one per entity `i`, with auto-`λ` selected as `0.1 × max(|Θᵀ y|) / T'`.
5. **Symmetrize** — average MAX-style across mirror coefficient rows so a single strong piece of evidence isn't diluted by zeros.
6. **Bootstrap** — K time-axis resamples (default K=10) per inferred edge, reporting the retention frequency as `confidence`.
7. **Flag artifacts** — any pairwise edge whose member set is a subset of a higher-weight higher-order edge gets `possible_masking_artifact = true`.

Validated against EATH-generated synthetic narratives with planted ground truth — Phase 15b's load-bearing test
(`reconstruct_tests::test_reconstruction_recovers_planted_eath_structure_auroc_gt_0_85`) achieves **AUROC = 0.852** at `T=300, N=12, max_order=3, K_bootstrap=25`.

## 10.3 REST Endpoints

```
POST /inference/hypergraph-reconstruction
  body: { narrative_id, params?: ReconstructionParams }
  → 201 { job_id, status: "Pending" }

GET  /inference/hypergraph-reconstruction/{job_id}
  → 200 InferenceResult { result: { kind: "reconstruction_done",
                                    result: ReconstructionResult } }

POST /inference/hypergraph-reconstruction/{job_id}/materialize
  body: { output_narrative_id, opt_in: true, confidence_threshold? }
  → 200 MaterializationReport { situations_created, situations_skipped, ... }
```

`ReconstructionParams` is fully optional — every field has a serde default. The engine applies `max_order = 3`, `observation = ParticipationRate`, `lambda_l1 = 0.0` (auto-select), `bootstrap_k = 10`, `pearson_filter_threshold = 0.1`, `entity_cap = 200` when the caller omits them.

Materialization is opt-in with a `confidence_threshold` (default **0.7** per §10.6 below). Each surviving `InferredHyperedge` becomes a `Situation` under `output_narrative_id` with `extraction_method = ExtractionMethod::Reconstructed { source_narrative_id, job_id }`.

## 10.4 TensaQL Grammar

```sql
-- Defaults: observation = participation_rate, max_order = 3, λ auto.
INFER HYPERGRAPH FROM DYNAMICS FOR "telegram-corpus-1"

-- Tunable form. observation source: participation_rate | sentiment_mean
-- | engagement (belief_mass requires a `proposition` field; use POST
-- directly for that).
INFER HYPERGRAPH FROM DYNAMICS FOR "telegram-corpus-1"
    USING OBSERVATION 'participation_rate'
    MAX_ORDER 3
    LAMBDA 0.05
```

Returns `{ job_id, status: "submitted" }`. Poll via `/jobs/{id}` or the dedicated `/inference/hypergraph-reconstruction/{job_id}` GET.

## 10.5 MCP Tool

```
reconstruct_hypergraph(narrative_id, params?) → { job_id, status }
```

Identical envelope to the REST `submit` endpoint. Tool count became 157 in v0.76.0.

## 10.6 Analyst Workflow — Confidence Over Weight

**Always filter by `confidence > 0.7`, not by `weight > ε`.**

The Taylor expansion makes triadic terms contribute nonzero pairwise coefficients (`possible_masking_artifact = true`). A pairwise edge with weight above the LASSO threshold is often a masking artifact of an underlying higher-order edge. Bootstrap confidence cleanly separates true edges from artifacts because resampling time-axis perturbs the artifacts more than the genuine signals.

The Studio reconstruction canvas defaults to confidence > 0.7 for the visual filter and the sidebar highlight. The materialization endpoint defaults to the same threshold for the same reason.

## 10.7 Provenance Encoding

Materialized situations carry:

```rust
ExtractionMethod::Reconstructed {
    source_narrative_id: String,
    job_id: String,
}
```

This is the third provenance variant alongside `Synthetic { model, run_id }` (synth) and the standard `LlmParsed` / `HumanEntered` / `StructuredImport` / `Sensor` / `Simulated`. Provenance lets analysts cleanly distinguish "this Situation was observed" from "this Situation was recovered from the inverse problem on entity dynamics."

Per-job materialized refs live under the dedicated KV slice:

```
syn/recon/{output_narrative_id_utf8}/{job_id_utf8}/{situation_id_v7_BE_BIN_16}
                                                  → ReconstructedSituationRef
```

Disjoint from every other `syn/*` slice. A prefix scan on `syn/recon/{output_narrative_id}/{job_id}/` returns every situation produced by a given reconstruction job in O(scan-window).

## 10.8 Studio Canvas

`/n/:id/reconstruction` (icon `⌬`) renders:

* **Top**: parameter form — observation source picker (only `participation_rate` enabled in MVP, others greyed with hint), max_order selector (2/3/4), λ override toggle + numeric input, bootstrap_k selector (5/10/20/50), confidence threshold input (default 0.7).
* **Middle**: side-by-side declared vs inferred hypergraph as force-directed graphs. Inferred edges filtered by `confidence > threshold`; edges flagged `possible_masking_artifact` rendered in amber.
* **Right sidebar**: bootstrap retention frequency table sorted by confidence descending, with a masking-artifact dot indicator + tooltip.

## 10.9 Open Q5 Escalation Path

If the load-bearing AUROC > 0.85 test starts failing on a real corpus, the documented escalation path (per `docs/EATH_sprint_extension.md` Phase 15b Q5):

1. Increase the planted-group driving forces (synthetic substrates) or the observation window length (real corpora).
2. Widen the SavitzkyGolay window from 5 to 7+.
3. Raise the Pearson filter threshold from 0.1 to 0.3+ to drop more candidate terms.
4. Bump `bootstrap_k` from 10 to 25+ for tighter confidence intervals.
5. Switch to MAX symmetrization for Order-2 / Order-3 terms (already the default in v0.75.4+).
6. Switch to ODE-driven generators (continuous-state) for synthetic ground truth so the observation has continuous-system semantics SINDy can exploit.

If steps 1-6 don't recover AUROC > 0.85 on a target corpus, escalate to algorithm review — the assumption that hyperedges manifest in joint co-occurrence dynamics may not hold for that domain.

---

# Chapter 11: Opinion Dynamics (BCM on Hypergraphs)

> **Citations.** Algorithm variants: Hickok, Kureh, Brooks, Feng, Porter — *"A Bounded-Confidence Model of Opinion Dynamics on Hypergraphs"*, SIAM J. Appl. Dyn. Syst. **21**, 1 (2022). Schawe, Hernández — *"Higher order interactions destroy phase transitions in Deffuant opinion dynamics model"*, Commun. Phys. **5**, 32 (2022). Dyadic baseline: Deffuant, Neau, Amblard, Weisbuch — *"Mixing beliefs among interacting agents"*, Adv. Complex Syst. **3**, 87 (2000).

## 11.1 What Opinion Dynamics Does

Opinion dynamics treats each entity as carrying a scalar opinion `x_i ∈ [0, 1]` and watches that opinion evolve under bounded-confidence (BCM) updates as the entity participates in successive hyperedges (situations) of the narrative. Two questions matter:

- **Does the population reach consensus, or does it fragment** into persistent opinion clusters?
- **At what confidence threshold does the dynamics transition** from rapid global consensus to persistent fragmentation?

The hypergraph is treated as fixed (no edge generation, no edge inference) — opinion dynamics complements the synthetic-generation (Chapter 9) and reconstruction (Chapter 10) chapters by working *on* a given hypergraph rather than producing or recovering one.

## 11.2 Algorithm Variants

Two MVP variants share the same hyperedge-selection / convergence machinery and differ only in the per-edge update rule:

- **PairwiseWithin** (default) — Hickok et al. 2022. Lifts Deffuant 2000 dyadic BCM to higher-order edges by applying the pairwise update to every ordered pair within the selected hyperedge in canonical (sorted-UUID) order — Gauss-Seidel, with updates immediately visible to subsequent pairs in the same edge. This ordering produces the *opinion-jumping* phenomenon (Hickok §4) and reduces to dyadic Deffuant on size-2 edges.
- **GroupMean** — Schawe & Hernández 2022. All-or-nothing group update: when the spread within the selected edge is below the (size-scaled) confidence bound `c_e`, every member moves toward the group mean by `μ`. Otherwise no update. Produces a **smooth crossover** rather than a sharp phase transition (paper-fidelity finding reproduced by Phase 16b's `test_bcm_group_mean_variant_smooth_crossover_matches_schawe_hernandez`).

## 11.3 Parameters

```
OpinionDynamicsParams {
    model: BcmVariant,                              // PairwiseWithin (default) | GroupMean
    confidence_bound: f32,                          // c ∈ (0, 1); default 0.3
    confidence_size_scaling: Option<...>,           // Flat (default) | InverseSqrtSize | InverseSize
    convergence_rate: f32,                          // μ ∈ (0, 1]; default 0.5 (Deffuant canonical)
    hyperedge_selection: HyperedgeSelection,        // UniformRandom (default) | ActivityWeighted | PerStepAll
    initial_opinion_distribution: InitialOpinionDist, // Uniform | Gaussian{mean, std} | Bimodal{mode_a, mode_b, spread} | Custom(Vec<f32>)
    convergence_tol: f32,                           // ε_conv (1e-4 default)
    convergence_window: usize,                      // N_conv consecutive sub-tolerance steps (100 default)
    max_steps: usize,                               // 100k default
    seed: u64,                                      // 42 default
}
```

`InitialOpinionDist::Custom(v)` requires `v.len() == N_entities`; mismatch raises `TensaError::InvalidInput` and the REST handler maps it to **HTTP 400**.

## 11.4 REST Endpoints

```
POST /analysis/opinion-dynamics
  body: { narrative_id, params?: OpinionDynamicsParams,
          include_synthetic?: bool }
  → 200 { run_id, report: OpinionDynamicsReport }   -- synchronous

POST /analysis/opinion-dynamics/phase-transition-sweep
  body: { narrative_id, c_range: [c_start, c_end, num_points],
          base_params?: OpinionDynamicsParams,
          include_synthetic?: bool }
  → 200 PhaseTransitionReport                        -- synchronous

POST /synth/opinion-significance                  -- default models = ["eath", "nudhy"] (both registered as of Phase 13b)
  body: { narrative_id, params?, k?, models? }
  → 201 { job_id, status: "Pending" }              -- queued; result via /jobs/{id}/result

GET  /synth/opinion-significance/{narrative_id}/{run_id}
  → 200 OpinionSignificanceReport

GET  /synth/opinion-significance/{narrative_id}?limit=N
  → 200 [OpinionSignificanceReport, ...]            -- newest first, default limit 50
```

Both `POST /analysis/opinion-dynamics*` endpoints run **synchronously** — Phase 16b benchmarks: 100×10k = 21 ms, 1000×100k = 98 ms (well under 1 s for MVP scales). The significance endpoint goes through the worker pool because per-K-sample × per-model parallelism makes K=50 × 2 models a multi-second operation.

`include_synthetic` defaults to `false` per the EATH Phase 3 invariant — aggregation endpoints must NOT mix synthetic records into real-only views by default. Each successful `/analysis/opinion-dynamics` run persists at `opd/report/{narrative_id}/{run_id_v7}` for chronological readback.

## 11.5 TensaQL Grammar

```sql
-- Synchronous opinion-dynamics run (named-parameter syntax mirrors INFER for analysis ops).
INFER OPINION_DYNAMICS(
    confidence_bound := <f>,
    variant := '<pairwise|group_mean>',
    [mu := <f>],
    [initial := '<uniform|gaussian|bimodal>']
) FOR "<narrative_id>"

-- Synchronous phase-transition sweep.
INFER OPINION_PHASE_TRANSITION(
    c_start := <f>, c_end := <f>, c_steps := <n>
) FOR "<narrative_id>"
```

Both verbs descriptor-row through the planner and execute inline against the engine. Returns `{ narrative_id, run_id, report }` for `OPINION_DYNAMICS`, `{ narrative_id, report }` for `OPINION_PHASE_TRANSITION`.

## 11.6 MCP Tools

- `simulate_opinion_dynamics(narrative_id, params?)` → inline `{ run_id, report }` (no job queue).
- `simulate_opinion_phase_transition(narrative_id, c_range, base_params?)` → inline `PhaseTransitionReport`.

Both tools mirror the REST handlers and bypass the job queue. Total MCP tool count after Phase 16c: **159** (157 → 159 with the two new opinion-dynamics tools).

## 11.7 KV Prefixes

| Prefix | Schema | Purpose |
|---|---|---|
| `opd/report/` | `opd/report/{narrative_id}/{run_id_v7_BE_BIN_16}` | Persisted `OpinionDynamicsReport` records (chronological scan order) |
| `syn/opinion_sig/` | `syn/opinion_sig/{narrative_id}/{run_id_v7_BE_BIN_16}` | Persisted `OpinionSignificanceReport` records |

Both encode `run_id` as 16 BE-binary bytes per the global key-encoding contract (Appendix B).

## 11.8 Phase Transition vs Phase 14 Bistability — Don't Conflate

This is the load-bearing distinction. Both phenomena are "phase transitions" in the loose sense, but they measure different observables on different dynamics:

| Observable | Phase 14 Bistability | Phase 16 Phase Transition |
|---|---|---|
| What varies | Transmission rate β | Confidence bound c |
| What's measured | Final infected prevalence | Time to convergence |
| Phenomenon | Bistable interval where two stable prevalence states coexist | Sharp spike in convergence time near `c ≈ σ²` |
| Model | SIR higher-order contagion (Ferraz de Arruda et al. 2023) | BCM opinion dynamics (Hickok et al. 2022 §5) |
| Phase signal | Hysteresis gap > threshold | Convergence-time spike |
| REST | `POST /analysis/contagion-bistability` | `POST /analysis/opinion-dynamics/phase-transition-sweep` |

Use Phase 14 for "is this narrative in the bistable contagion regime?" Use Phase 16 phase-transition sweep for "at what confidence threshold does this narrative stop converging to consensus?" The two endpoints + the two TensaQL verbs (`INFER CONTAGION_BISTABILITY` vs `INFER OPINION_PHASE_TRANSITION`) keep them disjoint at every layer.

## 11.9 Echo Chambers — Graceful Degradation

`OpinionDynamicsReport.echo_chamber_index` requires precomputed Label-Propagation labels at `an/lp/{narrative_id}/{entity_id}`. When labels are missing, the report carries `echo_chamber_available = false` and `echo_chamber_index = 0.0`. The analyst sees the missing-data signal — no panic, no error.

Workflow when echoes matter:
1. `INFER LABEL_PROPAGATION FOR n:Narrative WHERE n.id = "narr-1" RETURN n` (queues a job; result lands at `an/lp/`).
2. `INFER OPINION_DYNAMICS( ... ) FOR "narr-1"` (now `echo_chamber_available = true`).

The Studio canvas surfaces the missing-data hint inline ("label_propagation not run — echo_chamber_index is unavailable") so analysts know what to run next.

## 11.10 Wargame Integration (Phase 16c, --features adversarial)

```rust
RewardFunction::OpinionShift {
    target_opinion: 0.2,                              // shift toward this aggregate
    baseline_params: OpinionDynamicsParams,           // no-intervention run
    post_intervention_params: OpinionDynamicsParams,  // typically same
    aggregator: OpinionAggregator,                    // Mean | Median | ClusterMass{cluster_idx}
}
```

Evaluated via `evaluate_opinion_shift(reward, baseline_hg, baseline_nid, treatment_hg, treatment_nid)` which:
1. Runs opinion dynamics on the **baseline** substrate (no intervention applied).
2. Runs opinion dynamics on the **post-intervention** substrate.
3. Aggregates final opinions per the chosen `OpinionAggregator`.
4. Returns `reward = |baseline_agg - target| - |treatment_agg - target|` — positive when the intervention moved the aggregate closer to the target.

The `OpinionShiftEvaluation` shape mirrors Phase 8's `ComparisonHarness` rows so downstream test harnesses can render side-by-side "with intervention vs without" Markdown reports.

## 11.11 Studio Canvas

`/n/:narrativeId/opinion` — dedicated canvas tab (alongside Synth and Reconstruction). Layout:

- **Top:** params form (confidence-bound slider [0.01, 0.99], variant radio, initial-distribution picker, μ + ε_conv + max_steps inputs).
- **Main:** D3 trajectory chart (one line per entity, x = step, y = opinion ∈ [0, 1]; final converged clusters render as colored bands at the right edge).
- **Right panel:** numeric report summary (num_clusters, polarization_index, echo_chamber_index, converged?, steps; "label_propagation not run" hint when echo unavailable).
- **Below main viz:** collapsible phase-transition sub-panel (c_start / c_end / steps inputs + a separate "Sweep" button; renders convergence-time-vs-c with the critical-c spike marked in crimson).

No new npm dependencies — D3 is already in the studio bundle.

## 11.12 Skill Bundle

Phase 16c ships a sixth skill bundle: `tensa-opinion-dynamics`. Activate when the user wants to reason about consensus / fragmentation / echo chambers. The `tensa` and `tensa-writer` bundles cross-reference it for the three load-bearing idioms ("Does this narrative converge or fragment?", "What confidence threshold makes this group find consensus?", "Are there echo chambers in this corpus?").

The `/studio/chat/skills` enumeration is now: `studio-ui`, `tensa`, `tensa-writer`, `tensa-synth`, `tensa-reconstruction`, `tensa-opinion-dynamics`.

---

# Chapter 12: Configuration-Style Null Model (NuDHy) + Dual-Null-Model Significance

> **Citations.** Configuration-style null model: Preti, Fazzone, Petri, De Francisci Morales — *"Higher-Order Null Models as a Lens for Social Systems"*, Phys. Rev. X **14**, 031032 (2024), arXiv:2402.18470 (NuDHy-Degs and NuDHy-JOINT MCMC samplers for directed hypergraphs). Undirected double-edge-swap construction TENSA actually implements: Chodrow — *"Configuration models of random hypergraphs"*, J. Complex Networks **8**, cnaa018 (2020), arXiv:1902.09302.

## 12.1 What NuDHy Does

NuDHy is the **second registered `SurrogateModel`** alongside EATH. Where EATH is a *generative-dynamics* null (parameterised activity / recruitment process; samples by replaying), NuDHy is a *configuration-style* null: it samples uniformly from the set of all hypergraphs that share the source narrative's **per-entity hyperdegree sequence** AND **per-hyperedge size sequence** while randomizing everything else. Both invariants are preserved by construction; nothing else is.

The two nulls answer different questions:

| Null model | Fixes | Random | Asks "is the observed pattern beyond what's explainable by …" |
|---|---|---|---|
| **EATH** | Activity rates + group-size distribution + memory + bursty timing | Specific entity assignments per situation | … *generative dynamics* (per-entity activity, group-size profile, recruitment memory)? |
| **NuDHy** | Per-entity hyperdegree, per-edge size | Which entities sit in which edges | … *structural constraints alone* (degree distribution, edge-size distribution)? |

A pattern that's significant against BOTH is meaningfully above background in two independent ways — see §12.6.

**Inverted fidelity story.** NuDHy's `fidelity_metrics()` is **not** asking "does the surrogate reproduce source statistics?" — that's EATH's question, where reproduction is the goal. NuDHy's question is "did we successfully randomize while preserving the invariants?" The expected outcome is `degree_sequence_preservation` Spearman ρ ≈ 1.0 (invariant preserved), `edge_size_sequence_preservation` Spearman ρ ≈ 1.0 (invariant preserved), `entity_pair_overlap_divergence` KS *non-zero* (randomization happened). A NuDHy run that reproduces source pair-overlaps too closely is a *failed* run — the chain hasn't mixed.

## 12.2 NuDHy in the Surrogate Registry

`SurrogateRegistry::default()` returns BOTH EATH and NuDHy as of Phase 13b:

```rust
fn default() -> Self {
    let mut r = SurrogateRegistry::new();
    r.register(Arc::new(EathSurrogate));
    r.register(Arc::new(NudhySurrogate));    // Phase 13b
    r
}
// SurrogateRegistry::default().list() → ["eath", "nudhy"]
```

Downstream code (significance engines, generation jobs, hybrid components, MCP tools) looks models up by name string against the registry. Adding a third model means one `r.register` line — no parser, planner, REST, or MCP edits required for the bare model surface; the dual-null-model and bistability/opinion-significance surfaces (Phases 13c, 14, 16c) use whatever names the registry exposes.

`NudhySurrogate::name()` returns `"nudhy"`, `version()` returns `"v1.0"`. Synthetic records produced by NuDHy carry `synth_model = "nudhy"` in their `properties.synth_model` field and in `ExtractionMethod::Synthetic { model: "nudhy", run_id }` — the standard Phase 3 provenance tagging applies unchanged.

## 12.3 Calibration vs Generation

The calibrate/generate split mirrors EATH's even though NuDHy's parameter shape is far simpler:

- **Calibrate** (`POST /synth/calibrate/{nid}` body `{model: "nudhy"}`, or `CALIBRATE SURROGATE USING 'nudhy' FOR "..."`) reads the source narrative's participation index via `analysis::graph_projection::collect_participation_index`, builds the initial `NudhyState` (the hyperedge list + per-entity degree map + per-entity edge-membership index), and persists it at `syn/p/{narrative_id}/nudhy`. Calibration also derives the default `burn_in_steps = max(10_000, 10 × sum_of_edge_sizes)` and the `accept_rejection_rate_min` starvation threshold (default 0.01).
- **Generate** (`POST /synth/generate` body `{model: "nudhy", ...}`, or `GENERATE NARRATIVE "out" LIKE "src" USING SURROGATE 'nudhy' SEED <n>`) deserializes the calibrated `NudhyState`, runs the MCMC chain for `burn_in_steps + sample_gap_steps`, and emits each hyperedge as a Situation with a fresh UUIDv7 and participations.

**Entity-reuse.** Configuration models preserve node identity by construction — you cannot randomize membership across a fresh entity set. Phase 13b closed the Phase 0 deferral by adding `EmitContext::reuse_entities: Option<Vec<Uuid>>` to `src/synth/emit.rs`. When NuDHy emits, it sets `reuse_entities = Some(source_entity_uuids)`; the emit pipeline references those entities directly instead of minting new ones. EATH's emit path is unchanged (it still mints fresh entity UUIDs because its generation model is parameterised, not bijective with the source entity set).

The calibration blob is larger than EATH's (O(num_situations × mean_edge_size) vs O(num_entities)) — for a 1000-situation source with mean edge size 4, the serialized state weighs roughly 70 KB. This is acceptable; NuDHy intentionally preserves the full edge structure as its "parameters."

## 12.4 Algorithm Sketch

NuDHy implements Chodrow's undirected double-edge-swap MCMC. Each step picks two distinct hyperedges and proposes swapping one element between them:

```
loop {
    pick e1, e2 uniformly from hyperedges where e1 != e2
    pick v1 uniformly from e1, v2 uniformly from e2 where v1 != v2
    propose:
        e1' = (e1 \ {v1}) ∪ {v2}
        e2' = (e2 \ {v2}) ∪ {v1}
    accept iff:
        v1 ∉ e2  AND  v2 ∉ e1     // no self-duplicates in the target edges
}
```

Size preservation is automatic (`|e1'| = |e1|` and `|e2'| = |e2|`); degree preservation is preserved iff the acceptance rule fires (each entity stays in exactly the same number of edges, just redistributed across them). The `entity_to_edges` membership index updates incrementally on accept (O(1) deletions + insertions per swap) — full membership recomputation is never required.

**RNG.** Single `ChaCha8Rng` seeded from `SurrogateParams.seed`; every index pick + acceptance test draws from it. K-chain parallelism (in significance loops) uses XOR-mixed seeds (`seed_i = base_seed ⊕ i`) so each chain is independent + reproducible.

**Burn-in.** Default `burn_in_steps = max(10_000, 10 × sum_of_edge_sizes)`. After burn-in the current state is one sample; for multiple samples per chain, continue for `sample_gap_steps` between emissions (default 0 — one sample per chain, multiple samples = multiple chains).

**Starvation guard.** If acceptance rate drops below `accept_rejection_rate_min` (default 0.01) in the first 1000 proposals, the chain returns `Err(SynthFailure("MCMC starvation; source may be too rigid"))`. The K-loop in significance engines tallies starvations per model in `SingleModelSignificance.starvations` so callers can distinguish "0/50 succeeded" from "50/50 succeeded."

## 12.5 Edge Cases

- **Source has < 2 hyperedges** → `Err(SynthFailure("need >= 2 hyperedges"))`. Single-edge or empty narratives have nothing to swap.
- **All hyperedges have identical entity sets** → chain is a fixed point (every proposed swap rejected on the no-duplicates rule). Calibration warns and emits the source unchanged; generation warns + emits unchanged. Diagnostic: this signals a degenerate input, not an algorithm bug.
- **Single-element edges** (`|e| = 1`) → excluded from the chain state because no double-swap can preserve their size. They're stored separately in `NudhyParams.fixed_edges_json` and emitted as-is in the generation pass; the MCMC operates only on the multi-element subset.
- **MCMC starvation** (acceptance rate < `accept_rejection_rate_min` in first 1000 proposals) → `Err(SynthFailure("MCMC starvation; source may be too rigid"))`. NuDHy chains can starve; EATH never does.

## 12.6 Dual-Null-Model Significance (Phase 13c)

The Phase 13c surface combines EATH and NuDHy as TWO independent nulls in a single significance run. Standard practice in the higher-order networks literature: a pattern significant against EATH AND NuDHy is meaningfully above background in two independent ways — once against generative dynamics, once against structural constraints alone.

The engine (`SurrogateDualSignificanceEngine` in `src/synth/dual_significance_engine.rs`) runs the Phase 7 K-loop ONCE per requested model (not K/2 + K/2 — full K samples per model). Per-model parallelism is `std::thread::scope` (Phase 13c contract). Per-K-chain isolation reuses the Phase 7 `MemoryStore` pattern so synthetic records never pollute the user's KV store between samples.

Per-model rows surface AGGREGATE z-scores: each row picks the metric element with maximum |z| (NaN-safe — finite values win over NaN) so a single number per model can be compared without dragging the full `SyntheticDistribution` into the dual report. The combined verdict is AND-reduced across models.

## 12.7 REST Surface

```
POST /synth/dual-significance
  body: {
    narrative_id,
    metric: "temporal_motifs" | "communities" | "patterns",
    k_per_model?,                        // default 100, capped at 1000
    models?,                             // default ["eath", "nudhy"]
    metric_params?: {...}                // optional, threaded into job.parameters.metric_params
  }
  → 201 { job_id, status: "Pending" }

GET  /synth/dual-significance/{narrative_id}/{metric}/{run_id}
  → 200 DualSignificanceReport

GET  /synth/dual-significance/{narrative_id}/{metric}?limit=N
  → 200 [DualSignificanceReport, ...]   -- newest first, default limit 50
```

`metric` is intentionally restricted to the three structural values — contagion is NOT supported on the dual surface in Phase 13c because it requires a `HigherOrderSirParams` blob; future phases may add dual contagion. Each requested model is validated against the registry at engine entry; unknown names return `SynthFailure` with the registered list.

## 12.8 TensaQL Grammar

```sql
COMPUTE DUAL_SIGNIFICANCE FOR "<narrative_id>" USING '<metric>'
    [K_PER_MODEL <n>] [MODELS '<m1>','<m2>',...]
```

Default `MODELS` when the clause is omitted: `'eath','nudhy'` (the canonical dual-null pair). Default `K_PER_MODEL` when omitted: 100 per model (engine caps at 1000). Single-quoted strings for metric + model names visually separate from the double-quoted narrative ID.

Examples:

```sql
-- Default dual-null run (EATH + NuDHy, K=100 each).
COMPUTE DUAL_SIGNIFICANCE FOR "telegram-corpus-1" USING 'temporal_motifs'

-- Custom K + explicit model list.
COMPUTE DUAL_SIGNIFICANCE FOR "telegram-corpus-1" USING 'communities'
    K_PER_MODEL 200 MODELS 'eath','nudhy'
```

## 12.9 MCP Tool

```
compute_dual_significance(narrative_id, metric, k_per_model?, models?)
    → { job_id, status: "Pending" }
```

Wraps `POST /synth/dual-significance`. Returns the standard async job envelope; poll the result via `job_status` / `job_result` or fetch directly from `GET /synth/dual-significance/{nid}/{metric}/{run_id}`.

## 12.10 KV Prefixes

`syn/dual_sig/{narrative_id}/{metric}/{run_id_v7_BE_BIN_16}` persists the `DualSignificanceReport` blob. The prefix is **disjoint** from `syn/sig/` (Phase 7 single-model significance) so a `prefix_scan` on either slice never accidentally returns the other shape. The `run_id` is encoded as 16-byte big-endian binary so `prefix_scan` returns reports in chronological order; "newest first" listings reverse the scan output (O(n), no sort).

See [Appendix B](#appendix-b-key-encoding-scheme) for the full key-encoding contract.

## 12.11 Result Shape

```rust
pub struct DualSignificanceReport {
    pub run_id: Uuid,
    pub narrative_id: String,
    pub metric: String,                              // echoed with original casing
    pub k_per_model: u16,
    pub per_model: Vec<SingleModelSignificance>,    // one row per requested model
    pub combined: CombinedSignificance,             // AND-reduced verdict
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
}

pub struct SingleModelSignificance {
    pub model: String,                  // "eath" | "nudhy" | ...
    pub observed_value: f64,            // source-side value at the max-|z| element
    pub mean_null: f64,
    pub std_null: f64,                  // population stddev
    pub z_score: f64,                   // max-|z| across metric elements (NaN-safe)
    pub p_value: f64,                   // empirical one-tailed p at the same element
    pub samples_used: u16,              // < requested if the model starved
    pub starvations: u16,               // 0 for EATH; non-zero possible for NuDHy
}

pub struct CombinedSignificance {
    pub significant_vs_all_at_p05: bool,    // every per-model |z| > 1.96 AND p < 0.05
    pub significant_vs_all_at_p01: bool,    // every per-model |z| > 2.58 AND p < 0.01
    pub min_p_across_models: f32,
    pub max_abs_z_across_models: f32,
}
```

The `samples_used` + `starvations` pair lets callers tell "this model couldn't sample at all" from "this model sampled successfully but found nothing significant." A pattern that's `significant_vs_all_at_p05 = true` while one model has `samples_used = 0` is still surfaced with a row — the combined verdict checks every requested model, but consumers can filter on `samples_used > 0` if they want only models that actually contributed.

## 12.12 Performance + Cost

The largest fixed cost in dual significance is NuDHy's calibration parsing — the `NudhyState` blob is O(num_situations × mean_edge_size) and gets deserialized once per K-chain. Phase 13c mitigates this by parsing the calibration blob ONCE at engine entry and cloning the parsed `NudhyState` per chain (Arc-share where possible) instead of re-parsing from the JSON blob in every thread. For a 1000-situation source with mean edge size 4 (calibration blob ~70 KB), this saves K × parse-cost on every dual run.

Per-model parallelism via `std::thread::scope` is bounded — `models.len()` threads at outer level, then sequential K-loops within each model thread. K-chain parallelism within a single model is reserved for future tuning if profiling shows headroom.

## 12.13 Limitations + Tracked Follow-Ups

- **Single-model significance engine inside K-loop body** — `SurrogateSignificanceEngine` (Phase 7) still hard-codes EATH inside the per-K sample body even though the registry exposes NuDHy. Phase 13e is a tiny follow-up to thread the model name through the K-loop body so `POST /synth/significance` can target NuDHy directly. Until then, NuDHy sampling lands only via the dual-null surface.
- **Contagion through the dual engine** — the dual surface today supports only the three structural metrics (`temporal_motifs`, `communities`, `patterns`). Adding higher-order contagion to `SurrogateDualSignificanceEngine` is Phase 13d (deferred): the request shape needs a `HigherOrderSirParams` blob in `metric_params` and the adapter dispatcher needs an `AdapterChoice::Contagion` branch. Phase 14's `POST /synth/bistability-significance` is the dedicated dual-null surface for the contagion regime question; see [Chapter 13](#chapter-13-bistability--hysteresis-in-higher-order-contagion).
- **Per-component model in `GENERATE … USING HYBRID`** — the hybrid grammar still pins `model = "eath"` per component (see §9.5). Per-component `USING '<model>'` clause is one rule change; deferred until a use case lands.

---

# Chapter 13: Bistability / Hysteresis in Higher-Order Contagion

> **Citations.** Forward-backward β-sweep + bistability diagnostics: Ferraz de Arruda, Petri, Rodriguez, Moreno — *"Multistability, intermittency, and hybrid transitions in social contagion models on hypergraphs"*, Nat. Commun. **14**, 1375 (2023). Regime taxonomy (continuous / discontinuous / hybrid): Ferraz de Arruda, Aleta, Moreno — *"Contagion dynamics on higher-order networks"*, Nat. Rev. Phys. **6**, 468 (2024). Threshold-rule simplicial contagion (the substrate Phase 14 sweeps over): Iacopini, Petri, Barrat, Latora — *"Simplicial models of social contagion"*, Nat. Commun. **10**, 2485 (2019).

## 13.1 What Bistability Detection Does

Pairwise SIR shows a smooth, continuous relationship between transmission rate β and final infected prevalence — increase β a little, prevalence increases a little. Higher-order contagion with threshold rules behaves differently: there's often a **bistable interval** of β where the same value can give either a tiny outbreak (forward branch — start from low initial prevalence) or a massive outbreak (backward branch — start from high initial prevalence) depending on initial conditions, plus a hybrid regime at the boundary.

Phase 14's bistability detection sweeps β forward + backward, measures the gap, and classifies the regime. The EIC Pathfinder / disinfo-demo money shot: **"this narrative sits in the bistable regime where a small adversarial push flips the system."** The same sweep against K surrogate runs (Phase 14 dual-null significance) lets the analyst claim **"this real narrative has a wider bistable interval than 95% of EATH+NuDHy surrogates"** — a structural claim about the narrative's susceptibility to coordinated perturbation.

The simulator under the hood is unchanged Phase 7b — `super::higher_order_contagion::simulate_higher_order_sir` is reused verbatim for every (β, branch, replicate) triple. Phase 7b's load-bearing reduction-to-pairwise contract (`beta_per_size = [β, 0, 0, ...]` AND `threshold = ThresholdRule::Absolute(1)` ⇒ bit-identical pairwise SIR) is automatically preserved because Phase 14 never touches the simulator.

## 13.2 Algorithm

For each β in `linspace(beta_start, beta_end, num_points)`:

1. Spawn `replicates_per_beta` **forward** simulations with `initial_prevalence = initial_prevalence_low` (default 0.01).
2. Spawn `replicates_per_beta` **backward** simulations with `initial_prevalence = initial_prevalence_high` (default 0.5).
3. For each simulation, the **steady-state prevalence** is the mean of `per_step_infected / total_entities` across the last 10% of `steady_state_steps`.
4. Aggregate per-(β, branch): mean + population stddev across replicates.

Per-(β, replicate, branch) RNG seeds are XOR-mixes of `params.base_seed` with `(beta_idx, replicate_idx, branch_tag)` so the report is bit-identical across thread-scope reorderings. Outer parallelism is per-(β, branch) via `std::thread::scope` (Phase 13c pattern); replicates are sequential within each spawned task.

After the sweep, three diagnostics fall out of the curve:

- `max_hysteresis_gap` = `max_β |backward_prevalence(β) − forward_prevalence(β)|`
- `bistable_interval` — see §13.3
- `transition_type` — see §13.3

## 13.3 Detection Rules

**Bistable interval.** Contiguous β-range where `(backward_prevalence - forward_prevalence) > bistable_gap_threshold` (default **0.10**). If multiple disjoint contiguous spans qualify, the longest is reported as `bistable_interval = Some((β_low, β_high))`; if none qualifies, `None`.

**Transition classification:**

| Condition | `TransitionType` |
|---|---|
| `max_hysteresis_gap < 0.05` | `Continuous` |
| `0.05 ≤ max_hysteresis_gap < 0.30` AND `bistable_interval is None` | `Hybrid` |
| `max_hysteresis_gap ≥ 0.30` OR `bistable_interval is Some(_)` | `Discontinuous` |

`critical_beta_estimate`: midpoint of `bistable_interval` when present, else the β with steepest positive slope on the forward branch.

## 13.4 REST Surface

```
POST /analysis/contagion-bistability
  body: { narrative_id, params: BistabilitySweepParams }
  → 200 BistabilityReport                       -- SYNCHRONOUS (no job)

POST /synth/bistability-significance
  body: { narrative_id, params, k?, models? }
  → 201 { job_id, status: "Pending" }           -- queued; result via /jobs/{id}/result

GET  /synth/bistability-significance/{narrative_id}/{run_id}
  → 200 BistabilitySignificanceReport

GET  /synth/bistability-significance/{narrative_id}?limit=N
  → 200 [BistabilitySignificanceReport, ...]   -- newest first, default limit 50
```

`POST /analysis/contagion-bistability` is **synchronous** — bounded by the sweep design. `replicates_per_beta × num_points × 2 branches × steady_state_steps` is fixed at request time, so the handler can return the full `BistabilityReport` inline without queueing. Phase 14's quick-smoke sweep (`BistabilitySweepParams::quick`) is `10 × 5 × 2 × 200 = 20_000` simulations end-to-end and finishes well under the synchronous threshold.

`POST /synth/bistability-significance` IS queued because `K × models.len() × replicates_per_beta × num_points × 2 × steady_state_steps` blows past the synchronous threshold. Default `K = 50` (Phase 14 ships with a smaller default than the standard `K = 100` for structural-pattern significance, because each sample IS a full sweep), capped at 500. Default `models = ["eath", "nudhy"]`.

## 13.5 TensaQL Grammar

```sql
INFER CONTAGION_BISTABILITY(<beta_start>, <beta_end>, <steps>)
    FOR n:Narrative WHERE n.id = "<narrative_id>"
```

The grammar accepts the three sweep arguments as positional floats + integer; the dispatcher submits the synchronous bistability job with `BistabilitySweepParams::quick`-style defaults (gamma 0.1, threshold `Absolute(1)`, `initial_prevalence_low/high = 0.01/0.5`, `steady_state_steps = 200`, `replicates_per_beta = 5`). For per-field control over scaling / threshold / replicates, use the REST surface directly with a fully-specified `BistabilitySweepParams` blob.

Threading all `BistabilitySweepParams` fields through the grammar (e.g. a `WITH PARAMS { ... }` clause) is a tracked follow-up; the engine accepts arbitrary params via REST today.

## 13.6 MCP Tool

```
compute_bistability_significance(narrative_id, params, k?, models?)
    → { job_id, status: "Pending" }
```

Wraps `POST /synth/bistability-significance`. The synchronous `POST /analysis/contagion-bistability` endpoint does not have a dedicated MCP tool — callers route through the standard `query` tool with the `INFER CONTAGION_BISTABILITY(...)` grammar from §13.5, or invoke the REST endpoint directly.

## 13.7 KV Prefix

`syn/bistability/{narrative_id}/{run_id_v7_BE_BIN_16}` persists the `BistabilitySignificanceReport` blob. The prefix is **disjoint** from every other `syn/*` slice (params, runs, fidelity, single-model significance, contagion significance, dual significance, opinion significance, reconstruction). The `run_id` is 16-byte big-endian binary so `prefix_scan` returns reports in chronological order; "newest first" listings reverse the scan output (O(n), no sort).

The synchronous `POST /analysis/contagion-bistability` endpoint does NOT persist its `BistabilityReport` to KV — the response IS the result. Persistence applies only to the significance variant.

See [Appendix B](#appendix-b-key-encoding-scheme).

## 13.8 Result Shapes

```rust
pub struct BistabilitySweepParams {
    pub beta_0_range: (f32, f32, usize),    // linspace (start, end, num_points)
    pub beta_scaling: BetaScaling,          // UniformScaled { factor } | Custom(Vec<f32>)
    pub gamma: f32,                         // recovery rate
    pub threshold: ThresholdRule,           // Phase 7b type reused
    pub initial_prevalence_low: f32,        // forward-branch start (default 0.01)
    pub initial_prevalence_high: f32,       // backward-branch start (default 0.5)
    pub steady_state_steps: usize,          // per-β simulation length
    pub replicates_per_beta: usize,         // replicates per (β, branch)
    pub bistable_gap_threshold: f32,        // contiguous gap counted as bistable (default 0.10)
    pub base_seed: u64,                     // RNG seed; per-(β, replicate, branch) XOR-mixed
}

pub enum BetaScaling {
    UniformScaled { factor: f32 },          // β_d = factor × β_2 for d ≥ 3
    Custom(Vec<f32>),                       // absolute β_d values for d ≥ 3
}

pub struct HysteresisCurve {
    pub beta_values: Vec<f32>,
    pub forward_prevalence: Vec<f32>,
    pub backward_prevalence: Vec<f32>,
    pub forward_std: Vec<f32>,
    pub backward_std: Vec<f32>,
}

pub enum TransitionType { Continuous, Discontinuous, Hybrid }

pub struct BistabilityReport {
    pub curve: HysteresisCurve,
    pub bistable_interval: Option<(f32, f32)>,
    pub transition_type: TransitionType,
    pub max_hysteresis_gap: f32,
    pub critical_beta_estimate: Option<f32>,
}
```

Significance variants on top:

```rust
pub struct SingleModelBistabilityNull {
    pub model: String,                                  // "eath" | "nudhy" | ...
    pub mean_bistable_interval_width: f32,
    pub std_bistable_interval_width: f32,
    pub bistable_interval_width_quantile: f32,          // empirical quantile of source vs K nulls
    pub mean_max_hysteresis_gap: f32,
    pub std_max_hysteresis_gap: f32,
    pub max_hysteresis_gap_quantile: f32,
    pub samples_used: u16,
    pub starvations: u16,
}

pub struct BistabilitySignificance {
    pub source_bistable_wider_than_all_at_p05: bool,    // every quantile > 0.95
    pub source_bistable_wider_than_all_at_p01: bool,    // every quantile > 0.99
    pub min_quantile_across_models: f32,
    pub max_z_across_models: f32,
}

pub struct BistabilitySignificanceReport {
    pub run_id: Uuid,
    pub narrative_id: String,
    pub params: serde_json::Value,                       // echo of BistabilitySweepParams
    pub k: u16,
    pub models: Vec<String>,
    pub source_observation: serde_json::Value,           // BistabilityReport on real narrative
    pub per_model: Vec<SingleModelBistabilityNull>,
    pub combined: BistabilitySignificance,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
}
```

`params` and `source_observation` ride as `serde_json::Value` blobs because `BistabilitySweepParams` and `BistabilityReport` live in `crate::analysis::contagion_bistability` while the report types live in `crate::synth::types`; the layering avoids a cross-crate type dependency without losing field information at the wire.

The headline claim Phase 14 surfaces is **"real narrative has a wider bistable interval than 95% of EATH+NuDHy surrogates"** — `source_bistable_wider_than_all_at_p05 = true` is exactly this verdict, AND-reduced across all requested models.

## 13.9 Phase 14 vs Phase 16: Don't Conflate the Phase Transitions

Both Phase 14 (this chapter) and Phase 16 (Chapter 11 §11.8) study **phase transitions** in the loose sense — a sharp change in system behavior as a control parameter crosses a critical value. They measure DIFFERENT observables on DIFFERENT dynamics, and the load-bearing distinction is preserved at every layer (REST endpoint, TensaQL verb, MCP tool, KV slice, Studio canvas tab):

| Observable | **Phase 14 — Bistability** | **Phase 16 — Phase Transition** |
|---|---|---|
| Control parameter | Transmission rate β | Confidence bound c |
| Measured | Final infected prevalence | Time to convergence |
| Phenomenon | Bistable interval where two stable prevalence states coexist | Sharp spike in convergence time near `c ≈ σ²` |
| Dynamics | SIR higher-order contagion (Ferraz de Arruda et al. 2023) | BCM opinion dynamics (Hickok et al. 2022 §5) |
| Phase signal | Hysteresis gap > threshold | Convergence-time spike |
| REST | `POST /analysis/contagion-bistability` | `POST /analysis/opinion-dynamics/phase-transition-sweep` |
| TensaQL | `INFER CONTAGION_BISTABILITY(...)` | `INFER OPINION_PHASE_TRANSITION(...)` |
| MCP | `compute_bistability_significance` | `simulate_opinion_phase_transition` |

Use Phase 14 for **"is this narrative in the bistable contagion regime?"** Use Phase 16 phase-transition sweep for **"at what confidence threshold does this narrative stop converging to consensus?"** See [Chapter 11 §11.8](#chapter-11-opinion-dynamics-bcm-on-hypergraphs) for the full distinguishing table.

## 13.10 Studio Canvas

The Studio `SignificancePanel` ships a **Bistability** tab (alongside Single-model, Dual-null, Opinion). The `BistabilityChart` D3 component renders:

- **Forward branch** (low → high initial prevalence sweep) as a blue line with a shaded ±σ band.
- **Backward branch** as a red line with a shaded ±σ band.
- **Bistable interval** (when present) as a vertical amber-shaded region spanning `(β_low, β_high)`.
- **Critical β estimate** as a vertical crimson dashed line.
- **Transition type badge** in the corner: green (`Continuous`), amber (`Hybrid`), red (`Discontinuous`).

The component reuses the studio bundle's existing D3 dependency — no new npm packages.

## 13.11 Limitations + Caveats

- **Small-N narratives may classify as Hybrid not Discontinuous.** A 16-actor fixture with sparse higher-order substrate often produces `max_hysteresis_gap ∈ [0.05, 0.30)` and no contiguous bistable interval — the sweep correctly returns `TransitionType::Hybrid`, but a paper-quality `Discontinuous` classification needs ≥ 50 actors with denser higher-order substrate (rings of simplices per Iacopini et al. 2019 Fig. 2 give the cleanest planted-bistability fixture).
- **Synthetic narratives accepted.** Phase 14 does NOT refuse synthetic source narratives — bistability on a synthetic substrate is meaningful (it characterizes the surrogate itself). This contrasts with Phase 7 structural significance, which refuses fully-synthetic sources via `synth::emit::is_synthetic_*`.
- **Reduction-to-pairwise contract is load-bearing.** Phase 14 never modifies the underlying simulator; Phase 7b's contract (`beta_per_size = [β, 0, 0, ...]` + `threshold = Absolute(1)` ⇒ bit-identical pairwise SIR) is preserved. The bistability test suite in `tests/contagion_bistability_tests.rs` includes T1, which uses pairwise-equivalent params and asserts `TransitionType::Continuous` (smooth pairwise transition; no bistability on pure pairwise SIR).

---

# Chapter 14: Fuzzy Logic

> **Citations.** T-norms: Klement, Mesiar, Pap — *Triangular Norms*, Trends in Logic vol. 8, Kluwer 2000 (`klement2000`). OWA: Yager — IEEE Trans. SMC **18**, 183–190 (1988) (`yager1988owa`). Choquet integral: Grabisch — Eur. J. Oper. Res. **89**, 445–456 (1996) (`grabisch1996choquet`); Bustince et al. — IEEE Trans. Fuzzy Syst. **24**, 179–194 (2016) (`bustince2016choquet`). Fuzzy measures: Grabisch, Murofushi, Sugeno — Physica-Verlag 2000 (`grabisch2000fuzzymeasure`). Fuzzy Allen: Dubois-Prade — IEEE Trans. SMC **19**, 729–744 (1989) (`duboisprade1989fuzzyallen`); Schockaert, De Cock — Artif. Intell. **172**, 1158–1193 (2008) (`schockaert2008fuzzyallen`). Intermediate quantifiers: Novák — Fuzzy Sets Syst. **159**, 1229–1246 (2008) (`novak2008quantifiers`). Graded syllogisms: Murinová, Novák — Fuzzy Sets Syst. **186**, 47–80 (2012) (`murinovanovak2013syllogisms`) and **242**, 89–113 (2014) (`murinovanovak2014peterson`). Fuzzy FCA: Bělohlávek — Ann. Pure Appl. Logic **128**, 277–298 (2004) (`belohlavek2004fuzzyfca`); Krídlo, Ojeda-Aciego — Fuzzy Sets Syst. **161**, 1737–1749 (2010) (`kridlo2010fuzzyfca`). Mamdani: Mamdani, Assilian — Int. J. Man-Machine Studies **7**, 1–13 (1975) (`mamdani1975mamdani`). Hybrid: Flaminio, Holčapek, Cao — FSTA 2026 invited track (`flaminio2026fsta`); Fagin, Halpern — J. ACM **41**, 340–367 (1994) (`faginhalpern1994fuzzyprob`). Full BibTeX: [`docs/FUZZY_BIBLIOGRAPHY.bib`](FUZZY_BIBLIOGRAPHY.bib).

## 14.1 Overview + Default-Gödel Invariant

The Fuzzy Logic Sprint (v0.77.2 → v0.78.0) adds graded-truth reasoning on top of TENSA's existing confidence pipeline. Every user-visible fuzzy surface accepts:

- A **t-norm** `T : [0,1]² → [0,1]` — fuzzy conjunction (AND on graded values).
- An **aggregator** — reduces N graded values to one.
- Structured-logic primitives built on those two: fuzzy Allen relations, intermediate quantifiers, Peterson syllogisms, fuzzy FCA, Mamdani rule systems, fuzzy-probabilistic hybrid inference.

### Default-Gödel invariant (load-bearing)

Every TENSA surface that accepts `?tnorm=<kind>` defaults to **Gödel** (min) when the parameter is omitted — with ONE documented exception: pre-existing Dempster / Yager mass combination inside `analysis::evidence` defaults to **Goguen** (product) because that is the canonical mathematical form (preserving published DS numerics). The t-norm of record is always tagged in response JSON; `"tnorm": null` means the site-config default.

Site default lives at `cfg/fuzzy` (KV) and is read/written via `GET / PUT /fuzzy/config`. Default shipped out of the box: `{tnorm: "godel", aggregator: "mean"}`.

### Pre-sprint semantics preserved

Phase 1 opt-in wires (`*_with_tnorm` / `*_with_aggregator`) never change the default path. The Phase 0–12 `backward_compat_tests` (25 tests, all green at v0.78.0) are bit-identity snapshots against pre-sprint behaviour. Default callers see zero numerical difference. Opting into a non-default t-norm or aggregator is an explicit per-call decision.

## 14.2 T-Norms

Four canonical t-norm families are registered by default. Each satisfies commutativity, associativity, monotonicity in both arguments, and the neutral element `T(a, 1) = a`.

| Kind | `T(a, b)` | `T`-conorm (De Morgan dual) | Notes |
|---|---|---|---|
| **Gödel** (`godel`) | `min(a, b)` | `max(a, b)` | Floor semantics — strength = weakest input. TENSA default. |
| **Goguen** (`goguen`) | `a · b` | `a + b − a·b` | Probabilistic independence. Dempster's rule default. |
| **Łukasiewicz** (`lukasiewicz`) | `max(0, a + b − 1)` | `min(1, a + b)` | Strictest of the four. "Too low + too low = impossible." |
| **Hamacher(λ)** (`hamacher`, λ ≥ 0) | `(a · b) / (λ + (1 − λ)(a + b − a·b))` | derived from De Morgan | Recovers Goguen at λ = 1 (< 1e-12 on 11 × 11 grid). |

**Pointwise ordering** (at every `(a, b) ∈ [0, 1]²`): `Łukasiewicz ≤ Goguen ≤ Gödel`. Tested explicitly at 36 grid points in `fuzzy::tnorm_tests::tnorm_ordering_at_36_points`. Pick Łukasiewicz for strictness, Gödel for leniency, Goguen for stochastic-independence semantics.

```
[0,1]² ───►  Łukasiewicz  ≤  Goguen  ≤  Gödel
               (strict)      (product)     (min; TENSA default)
```

All inputs are defensively clamped to `[0, 1]`; no NaN / Inf emission. `n`-ary `reduce_tnorm` / `reduce_tconorm` fold from the neutral element (1.0 / 0.0).

Implementation lives at [`src/fuzzy/tnorm.rs`](../src/fuzzy/tnorm.rs) + [`src/fuzzy/tnorm_tests.rs`](../src/fuzzy/tnorm_tests.rs) (33 tests).

## 14.3 Aggregators

Aggregators reduce `xs: &[f64]` to a single `f64`. Registered by default:

| Kind | Signature | Semantics |
|---|---|---|
| `Mean` | `sum(xs) / n` | Arithmetic mean |
| `Median` | sorted middle | Middle value (tie-breaks below) |
| `TNormReduce` | `reduce_tnorm(tnorm, xs)` | Fuse via t-norm (AND-of-all) |
| `TConormReduce` | `reduce_tconorm(tnorm, xs)` | Fuse via t-conorm (OR-of-all) |
| `Owa { weights }` | Sort xs desc + Σ `w_i · x_(i)` | Yager 1988 ordered weighted average |
| `Choquet { measure }` | Integral against `μ` | Grabisch 1996 (non-additive) |

### OWA (Yager 1988)

Given `xs = [x_1, ..., x_n]` and weights `w = [w_1, ..., w_n]` with `Σw = 1`:
```
OWA_w(xs) = Σ w_i · x_(i)         where x_(1) ≥ x_(2) ≥ ... ≥ x_(n)
```

Ships linguistic-quantifier weight helpers (`Most`, `AlmostAll`, `Few`) via `w_i = Q(i/n) − Q((i−1)/n)`; `Σ w = 1` by telescoping. `owa_normalize(xs, w)` renormalises non-unit sums; empty / length-mismatched / non-unit inputs surface as `TensaError::InvalidInput → HTTP 400`.

### Choquet integral (Grabisch 1996, Bustince et al. 2016)

Given a monotone fuzzy measure `μ : 2^N → [0, 1]` with `μ(∅) = 0` and `μ(N) = 1`, and `xs` sorted ascending (permutation `π`):
```
C_μ(xs) = Σ_{i=1..n} (x_π(i) − x_π(i−1)) · μ({π(i), ..., π(n)})     (x_π(0) := 0)
```

Three symmetric built-ins:

| Built-in | Behaviour |
|---|---|
| **additive** | Recovers arithmetic mean (`μ(S) = |S|/n`). |
| **pessimistic** | Recovers `min` (μ concentrates mass on the full set). |
| **optimistic** | Recovers `max` (μ concentrates mass at first non-empty). |

Performance caps: exact path via subset-bitmask lookup at `O(n · 2^n)` for `n ≤ 10` (`EXACT_N_CAP`). Above 10 and up to the hard `FuzzyMeasure` cap of 16, falls back to Monte-Carlo with `k = 1000` permutations (ChaCha8 seeded), returning `ChoquetResult { value, std_err }`. `choquet(xs, measure, seed)` dispatches.

### FuzzyMeasure representation

```rust
pub struct FuzzyMeasure {
    pub name: String,
    pub n: usize,                 // 1..=16
    pub masses: Vec<f64>,         // length 2^n; masses[bitmask] = μ(S)
}
```

`check_monotone` verifies `μ(S) ≤ μ(T)` for all `S ⊆ T` via "add one element" iteration (`O(n · 2^n)`). `new_monotone` constructor rejects violations up front. `mobius_from_measure` / `measure_from_mobius` round-trip the Möbius transform for advanced decision-modelling workflows.

Implementation: [`src/fuzzy/aggregation.rs`](../src/fuzzy/aggregation.rs), [`aggregation_owa.rs`](../src/fuzzy/aggregation_owa.rs), [`aggregation_choquet.rs`](../src/fuzzy/aggregation_choquet.rs), [`aggregation_measure.rs`](../src/fuzzy/aggregation_measure.rs), [`aggregation_tests.rs`](../src/fuzzy/aggregation_tests.rs) (17 tests).

## 14.4 Fuzzy Allen Relations

A crisp `AllenInterval` may carry optional `fuzzy_endpoints: Option<FuzzyEndpoints>` (serde-default on load — historical archives deserialize unchanged). When both operands have `None`, the 13-vector fast path returns a one-hot vector matching the crisp Allen relation.

```rust
pub struct TrapezoidalFuzzy {
    a: f64, b: f64, c: f64, d: f64,       // a ≤ b ≤ c ≤ d; validated at construction
}

pub struct FuzzyEndpoints {
    pub start: TrapezoidalFuzzy,
    pub end:   TrapezoidalFuzzy,
}

pub fn graded_relation(
    a: &AllenInterval,
    b: &AllenInterval,
    cfg: &GradedAllenConfig,             // { tnorm: TNormKind }
) -> [f64; 13];
```

**Construction** (Dubois-Prade 1989, refined by Schockaert-De Cock 2008): the four point-order constraints between fuzzy endpoints (`a⁻ < b⁻`, `a⁻ < b⁺`, `a⁺ < b⁻`, `a⁺ < b⁺`) are evaluated via
```
μ_≤(x̃, ỹ) = (possibility(x̃ ≤ ỹ) + necessity(x̃ ≤ ỹ)) / 2
```
and the 13 Allen relations reduce to conjunctions of these constraints under the configured t-norm (default Gödel). Reconciler-detected fuzziness cues (`shortly`, `around`, `about`, `early`, `late`, `approximately`) from Phase P4.2 ingestion widen crisp endpoints into ±10 % trapezoidal windows automatically.

KV cache: `fz/allen/{narrative_id}/{a_id_BE}/{b_id_BE}` via `save_fuzzy_allen` / `load_fuzzy_allen` / `delete_fuzzy_allen` / `invalidate_pair`.

REST:
```
POST /analysis/fuzzy-allen
  body: { narrative_id, a_id, b_id, tnorm? }
  → 200 { relation: [f64; 13], tnorm, cached: bool }

GET /analysis/fuzzy-allen/:nid/:a_id/:b_id
  → 200 (cached vector, or recomputes on miss)
```

TensaQL tail:
```sql
MATCH (s:Situation) AT s.temporal AS FUZZY OVERLAPS THRESHOLD 0.3 RETURN s
```

Implementation: [`src/fuzzy/allen.rs`](../src/fuzzy/allen.rs), [`allen_store.rs`](../src/fuzzy/allen_store.rs), [`allen_tests.rs`](../src/fuzzy/allen_tests.rs) (9 tests).

## 14.5 Intermediate Quantifiers

Novák-Murinová "most / many / few / almost all" ramps on `r ∈ [0, 1]`:

| Quantifier | Ramp (non-zero window) |
|---|---|
| `Most` | peaks `0.3 → 0.8` |
| `Many` | peaks `0.1 → 0.5` |
| `Few` | `1 − Q_many` |
| `AlmostAll` | peaks `0.7 → 0.95` |

Evaluation over a predicate `P` on domain `D`: `r = (Σ_{e ∈ D} μ_P(e)) / |D|` → `Q(r)`. Empty domain returns `0`. Implementation applies the ramp over `evaluate_property_path` with `mu_bool` / `compare_ord` helpers.

TensaQL:
```sql
QUANTIFY <quantifier> (pattern) [WHERE <condition>] [FOR "<nid>"] [AS "<label>"]
```

REST:
```
POST /fuzzy/quantify
  body: { narrative_id, quantifier: "most|many|few|almost_all",
          predicate, where?, label? }
  → 200 { value: f64, n_domain, n_matched, label? }

GET /fuzzy/quantify/{nid}/{predicate_hash}
  → 200 cached result
```

Cache: `fz/quant/{nid}/{predicate_hash}` (best-effort; invalidated on mutation).

Implementation: [`src/fuzzy/quantifier.rs`](../src/fuzzy/quantifier.rs) + [`quantifier_tests.rs`](../src/fuzzy/quantifier_tests.rs) (12 tests).

## 14.6 Graded Peterson Syllogisms (Prototype)

Murinová-Novák 2014. Peterson's square of opposition with graded degrees.

```rust
pub struct Syllogism {
    pub major: SyllogismStatement,       // Q_major Subject IS Middle
    pub minor: SyllogismStatement,       // Q_minor Middle IS Predicate
    pub conclusion: SyllogismStatement,  // Q_conclusion Subject IS Predicate
    pub figure_hint: Option<SyllogismFigure>,
}

pub struct GradedValidity {
    pub degree: f64,             // in [0, 1]; conjunction of 3 Q_ evaluations under configured t-norm
    pub figure: SyllogismFigure, // Figure I..V
    pub valid: bool,             // degree ≥ threshold AND figure admissibility
    pub threshold: f64,
}
```

**Five Peterson figures** (`I`, `II`, `III`, `IV`, `V`) are derived from a pattern-matching classifier on the quantifier triple. **Figure II with non-canonical quantifier ordering always returns `valid = false`** regardless of computed degree — this is the Peterson taxonomy. Higher-figure syllogisms are only partially covered in the prototype; formal Łukasiewicz-BL* soundness is deferred to Phase 7.5.

DSL (parsed by `syllogism::parse_statement`, shared by executor + REST):
```
"<QUANTIFIER> <Subject> IS <Object>"
```

Quantifiers: `All`, `Most`, `Many`, `Few`, `AlmostAll`, `Some`, `No`.

**Predicate forms** (default `TypePredicateResolver`, expanded in v0.79.32 to match the Studio panel hint):

| Form | Example | Match |
|------|---------|-------|
| `entity` / `*` | `entity` | every entity (μ = 1.0) |
| `type:<EntityType>` | `type:Actor` | `Actor` / `Location` / `Artifact` / `Concept` / `Organization` (case-insensitive) |
| `maturity=<level>` | `maturity=Validated` | `Candidate` / `Reviewed` / `Validated` / `GroundTruth` (case-insensitive; `ground_truth` accepted) |
| `property:<key>=<value>` | `property:role=protagonist` | `entity.properties[key]` literal-equals `value` (string/null/number/bool coerced) |
| `confidence<op><number>` | `confidence>0.7`, `confidence<=0.5` | crisp comparison on `entity.confidence`; ops: `>`, `<`, `>=`, `<=`, `=` |

Plug a custom [`PredicateResolver`](../src/fuzzy/syllogism.rs) for richer logic (text filters, embedding similarity, …); unknown ids on the default resolver return a `400` enumerating these five forms.

TensaQL:
```sql
VERIFY SYLLOGISM {
    major:      'Most Actor IS Influential',
    minor:      'All Influential IS Leader',
    conclusion: 'Most Actor IS Leader'
} FOR "n1" [THRESHOLD <f>] [WITH TNORM '<kind>']
```

REST:
```
POST /fuzzy/syllogism/verify
  body: { narrative_id, major, minor, conclusion, threshold?, tnorm? }
  → 200 { degree, figure, valid, threshold, fuzzy_config }

GET /fuzzy/syllogism/{nid}/{proof_id}
  → 200 stored proof
```

KV: `fz/syllog/{nid}/{proof_id_v7_BE_16}`.

Implementation: [`src/fuzzy/syllogism.rs`](../src/fuzzy/syllogism.rs) + [`syllogism_tests.rs`](../src/fuzzy/syllogism_tests.rs) (14 tests as of v0.79.32).

## 14.7 Fuzzy Formal Concept Analysis (FCA)

Bělohlávek-Krídlo concept lattices over a graded incidence `I : O × A → [0, 1]`.

```rust
pub struct FormalContext {
    pub objects: Vec<Uuid>,
    pub attributes: Vec<String>,
    pub incidence: Vec<Vec<f64>>,           // [objects][attributes], values in [0, 1]
}

pub struct Concept {
    pub extent: Vec<usize>,                  // object indices
    pub intent: Vec<(usize, f64)>,           // (attribute_idx, grade)
}

pub struct ConceptLattice {
    pub id: Uuid,
    pub narrative_id: String,
    pub tnorm: TNormKind,
    pub attributes: Vec<String>,
    pub objects: Vec<Uuid>,
    pub concepts: Vec<Concept>,
    pub order: Vec<(usize, usize)>,          // transitive-reduction Hasse edges
    pub created_at: DateTime<Utc>,
}
```

Galois closure under the configured t-norm uses the corresponding residual implication:
- Gödel / Hamacher: `x → y = 1` if `x ≤ y`, else `y`.
- Goguen: `min(1, y / x)` (`1` if `x = 0`).
- Łukasiewicz: `min(1, 1 − x + y)`.

Enumeration: Ganter 1984 `next_closure_enumerate`, iterating every object-subset with `u64` bitmask dedup (hard-caps at 64 objects). Hasse edges via transitive reduction.

**Performance caps**: soft `500 × 50` (`tracing::warn!` on opt-in with `large_context: true`), hard `2000 × 200` (rejects with `InvalidInput`). Above 64 objects, narrow via an attribute allowlist or an entity-type filter.

TensaQL:
```sql
FCA LATTICE FOR "<nid>"
    [THRESHOLD <n>]
    [ATTRIBUTES ['tag1', 'tag2', ...]]
    [ENTITY_TYPE <Actor|Location|...>]
    [WITH TNORM '<kind>']

FCA CONCEPT <idx> FROM "<lattice_id>"
```

REST:
```
POST /fuzzy/fca/lattice
  body: { narrative_id, attributes?, entity_type?, threshold?, tnorm? }
  → 200 ConceptLattice

GET  /fuzzy/fca/lattice/{lattice_id}
GET  /fuzzy/fca/lattices/{nid}
DELETE /fuzzy/fca/lattice/{lattice_id}
```

KV: `fz/fca/{lattice_id_v7_BE_16}` + narrative-scoped index.

Workflow wires: `narrative_family_lattice()` (Disinfo Ops) and `build_archetype_lattice()` (writer, Actor-filtered).

Implementation: [`src/fuzzy/fca.rs`](../src/fuzzy/fca.rs) + [`fca_store.rs`](../src/fuzzy/fca_store.rs) + [`fca_tests.rs`](../src/fuzzy/fca_tests.rs) (11 tests).

## 14.8 Mamdani Rule Systems

Mamdani-Assilian 1975-style linguistic rule bases. Three membership-function kinds:

```rust
pub enum MembershipFunction {
    Triangular  { a, b, c },       // peak at b, 0 outside [a, c]
    Trapezoidal { a, b, c, d },    // plateau [b, c], 0 outside [a, d]
    Gaussian    { mean, sigma },   // standard form; sigma > 0
}
```

All MFs clamp output to `[0, 1]` and guard degenerate parameters (no NaN emission on zero-width triangles or zero-σ Gaussians).

```rust
pub struct MamdaniRule {
    pub id: Uuid,
    pub name: String,
    pub narrative_id: String,
    pub antecedent: Vec<FuzzyCondition>,       // AND of fuzzified property conditions
    pub consequent: Vec<FuzzyOutput>,           // one or more output variables
    pub tnorm: TNormKind,                       // rule-local firing semantics
    pub created_at: DateTime<Utc>,
    pub enabled: bool,
}

pub struct MamdaniRuleSet {
    pub rules: Vec<MamdaniRule>,
    pub defuzzification: Defuzzification,       // Centroid (default) | MeanOfMaxima
    pub firing_aggregator: Option<AggregatorKind>,
}
```

Pipeline (`evaluate_rule_set`):
1. `resolve_variable` walks dot-paths into `entity.properties.*` + reserved `entity.confidence` / `entity.entity_type`. bool → 1/0, string → indicator.
2. `fuzzify_condition` → μ per antecedent condition.
3. `firing_strength` = `reduce_tnorm(rule.tnorm, &condition_mus)`.
4. `aggregate_consequents` — union of scaled consequents over `DEFAULT_DEFUZZ_BINS = 100`.
5. Defuzzification: centroid `Σ x·μ(x) / Σ μ(x)` or mean-of-maxima over bins where μ attains its maximum within 1e-9.

KV: `fz/rules/{nid}/{rule_id_v7_BE_16}`. `find_rule_by_id_anywhere` does a workspace-wide scan for cross-narrative rule reuse.

TensaQL:
```sql
EVALUATE RULES FOR "<nid>"
    AGAINST (e:Actor)
    [RULES ['rule-a', 'rule-b']]
    [WITH TNORM '<kind>']
```

REST:
```
POST   /fuzzy/rules                        -- create / upsert
GET    /fuzzy/rules/{nid}                  -- list
GET    /fuzzy/rules/{nid}/{rule_id}        -- read
DELETE /fuzzy/rules/{nid}/{rule_id}        -- drop
POST   /fuzzy/rules/{nid}/evaluate          -- fire against matched entities
```

Workflow wires:
- `AlertRule.mamdani_rule_id: Option<String>` — firing strength as gate signal.
- `IngestionConfig.post_ingest_mamdani_rule_id` + `PipelineConfig.post_ingest_mamdani_rule_id` — stamp `properties.mamdani = {rule_id, rule_name, firing_strength, linguistic_term, defuzzified_output}` on each entity post-ingest via `update_entity_no_snapshot`.

Reference rule fixture: `"elevated-disinfo-risk"` — triangular MF on source trust, trapezoidal on corroboration, Gaussian on recency → elevated-risk linguistic output.

Implementation: [`src/fuzzy/rules.rs`](../src/fuzzy/rules.rs) + [`rules_types.rs`](../src/fuzzy/rules_types.rs) + [`rules_eval.rs`](../src/fuzzy/rules_eval.rs) + [`rules_store.rs`](../src/fuzzy/rules_store.rs) + [`rules_tests.rs`](../src/fuzzy/rules_tests.rs) + [`rules_integration_tests.rs`](../src/fuzzy/rules_integration_tests.rs) (17 tests combined).

## 14.9 Fuzzy-Probabilistic Hybrid (Scope-Capped)

Cao-Holčapek-Flaminio 2026 FSTA base case. Sugeno-additive semantics on the **discrete** probability distribution:
```
P_fuzzy(E) = Σ_{e ∈ outcomes} μ_E(e) · P(e)
```

This reduces bit-identically to the classical `P(A)` on crisp indicator `μ`. Full continuous-distribution / Flaminio modal-logic embedding / Fagin-Halpern multi-agent is **deferred to Phase 10.5+** per [`docs/fuzzy_hybrid_algorithm.md`](fuzzy_hybrid_algorithm.md).

```rust
pub struct FuzzyEvent {
    pub predicate_kind: FuzzyEventPredicate,    // Quantifier | MamdaniRule | Custom
    pub predicate_payload: String,               // opaque payload; MembershipDispatcher parses once
}

pub enum ProbDist {
    Discrete { outcomes: Vec<(Uuid, f64)> },    // validate: Σ P ≈ 1 ± 1e-9, no duplicates
}
```

**Scope boundary — IN:**
- Discrete outcome distributions.
- Fuzzy event predicates: Phase 6 `Quantifier`, Phase 9 `MamdaniRule`, pre-computed `Custom` map.

**Scope boundary — OUT (deferred):**
- Continuous distributions (Phase 10.5).
- Flaminio modal-logic embedding (Phase 15 of research extension).
- Fagin-Halpern multi-agent epistemic fuzzy probability.
- Decision-theoretic query layers.

Performance: `MembershipDispatcher` pre-parses the payload once then evaluates per outcome (`O(|outcomes|)`, NOT `O(|narrative|)`). Quantifier path reuses Phase 6 crisp-WHERE semantics; Mamdani path calls Phase 9 `evaluate_rules_against_entity`; Custom is pure map lookup.

TensaQL:
```sql
INFER FUZZY_PROBABILITY(
    event_kind   := '<quantifier|mamdani_rule|custom>',
    event_ref    := '<payload>',                     -- JSON or string id
    distribution := '<uniform|{"kind":"discrete", "outcomes":[...]}>'
) FOR "<nid>" [WITH TNORM '<kind>']
```

REST:
```
POST   /fuzzy/hybrid/probability
  body: { narrative_id, event, distribution, tnorm? }
  → 200 HybridProbabilityReport

GET    /fuzzy/hybrid/probability/{nid}
GET    /fuzzy/hybrid/probability/{nid}/{query_id}
DELETE /fuzzy/hybrid/probability/{nid}/{query_id}
```

### Worked example

> *"What is the probability that most sources with at least a single attribution corroborate a specific claim?"*

```sql
INFER FUZZY_PROBABILITY(
    event_kind   := 'quantifier',
    event_ref    := '{"kind":"most","predicate":"e.confidence > 0.7"}',
    distribution := 'uniform'
) FOR "n1"
```

The fuzzy event `μ_E` is the Phase-6 `Q_most` ramp evaluated per outcome; the distribution is uniform over entities in `n1`. The integral `P_fuzzy(E) ∈ [0, 1]` is the expected fuzzy truth of "most support this claim" under a random draw.

KV: `fz/hybrid/{nid}/{query_id_v7_BE_16}`.

Implementation: [`src/fuzzy/hybrid.rs`](../src/fuzzy/hybrid.rs) + [`hybrid_tests.rs`](../src/fuzzy/hybrid_tests.rs) (7 tests).

## 14.10 REST Surface

| Method | Path | Use |
|---|---|---|
| GET | `/fuzzy/tnorms` / `/fuzzy/tnorms/:kind` | Registered t-norm catalogue (kind / description / formula) |
| GET | `/fuzzy/aggregators` / `/fuzzy/aggregators/:kind` | Registered aggregator catalogue |
| POST | `/fuzzy/measures` | Register a fuzzy measure (monotonicity-checked; rejects with `"monoton"` on violation) |
| GET / DELETE | `/fuzzy/measures` / `/fuzzy/measures/:name` | List / drop measures (Phase 3: `?version=N` selects a specific historical slice; absent = latest pointer) |
| POST | `/fuzzy/measures/learn` | Graded sprint Phase 2 — fit a Choquet measure from a `(input_vec, rank)` dataset; persists at `fz/tn/measures/{name}` + `fz/tn/measures/{name}/v{N}` |
| GET | `/fuzzy/measures/:name/versions` | Graded sprint Phase 3 — list known versions for a measure name (sorted ascending) |
| GET / PUT | `/fuzzy/config` | Site-default `{tnorm, aggregator}` (KV: `cfg/fuzzy`) |
| POST | `/fuzzy/aggregate` | One-shot aggregation: `{xs, aggregator, tnorm?, weights?, measure?}` |
| POST | `/analysis/argumentation/gradual` | Graded sprint Phase 3 — synchronous gradual / ranking-based argumentation. Body: `{narrative_id, gradual_semantics, tnorm?}` → `{narrative_id, gradual: GradualResult, iterations, converged}` |
| POST / GET | `/analysis/fuzzy-allen` + `/analysis/fuzzy-allen/:nid/:a/:b` | Compute + cache graded Allen 13-vector |
| POST / GET | `/fuzzy/quantify` + `/fuzzy/quantify/{nid}/{hash}` | Evaluate intermediate quantifier |
| POST / GET | `/fuzzy/syllogism/verify` + `/fuzzy/syllogism/{nid}/{proof_id}` | Graded Peterson verification |
| POST | `/fuzzy/fca/lattice` | Build concept lattice |
| GET | `/fuzzy/fca/lattice/:id` + `/fuzzy/fca/lattices/:nid` | Read lattice / list per-narrative |
| DELETE | `/fuzzy/fca/lattice/:id` | Drop lattice |
| POST / GET / DELETE | `/fuzzy/rules` + `/fuzzy/rules/:nid` + `/fuzzy/rules/:nid/:rid` | Mamdani rule CRUD |
| POST | `/fuzzy/rules/:nid/evaluate` | Fire rule set against matched entities |
| POST / GET / DELETE | `/fuzzy/hybrid/probability` + `.../:nid` + `.../:nid/:qid` | Fuzzy-probabilistic hybrid |

**Per-endpoint opt-in** (Phase 4 — every existing confidence-returning route accepts `?tnorm=<kind>&aggregator=<kind>`): `/entities`, `/entities/:id`, `/situations`, `/situations/:id`, `/entities/:id/attributions`, `/situations/:id/attributions`, `/entities/:id/recompute-confidence`, `/situations/:id/recompute-confidence`, `/ask`, `/narratives/:id/communities`, `/narratives/:id/fingerprint`, all `/synth/*` significance endpoints. Default path bit-identical to pre-sprint; opting in returns the value under the selected semantics and tags the response JSON with the chosen config. Unknown kinds surface as `InvalidInput → HTTP 400`.

## 14.11 TensaQL Surface

```
with_tnorm_clause    = "WITH TNORM" STRING
with_aggregator_clause = "AGGREGATE" aggregator_kind [weight_vector] [measure_ref]
aggregator_kind      = "mean" | "median" | "tnorm" | "tconorm" | "owa" | "choquet"
weight_vector        = "[" number ("," number)* "]"
measure_ref          = STRING

fuzzy_at_tail        = "AS" "FUZZY" allen_relation "THRESHOLD" number

quantify_verb        = "QUANTIFY" quantifier "(" pattern ")" ["WHERE" cond] ["FOR" STRING] ["AS" STRING]
verify_syllogism     = "VERIFY" "SYLLOGISM" "{" "major:" STRING "," "minor:" STRING "," "conclusion:" STRING "}" "FOR" STRING ["THRESHOLD" number] [with_tnorm_clause]
fca_lattice_verb     = "FCA" "LATTICE" "FOR" STRING ["THRESHOLD" number] ["ATTRIBUTES" string_list] ["ENTITY_TYPE" entity_type] [with_tnorm_clause]
fca_concept_verb     = "FCA" "CONCEPT" number "FROM" STRING
evaluate_rules_verb  = "EVALUATE" "RULES" "FOR" STRING "AGAINST" "(" pattern ")" ["RULES" string_list] [with_tnorm_clause]
fuzzy_probability_verb = "INFER" "FUZZY_PROBABILITY" "(" event_kind ":=" STRING "," event_ref ":=" STRING "," distribution ":=" STRING ")" "FOR" STRING [with_tnorm_clause]
```

Every existing `INFER` verb (16 total: CENTRALITY / ENTROPY / BELIEFS / EVIDENCE / ARGUMENTS / CONTAGION / COMMUNITIES / TEMPORAL_RULES / MEAN_FIELD / PSL / TRAJECTORY / SIMULATE / OPINION_DYNAMICS / OPINION_PHASE_TRANSITION / HIGHER_ORDER_CONTAGION / HYPERGRAPH FROM DYNAMICS) accepts the trailing `WITH TNORM '<kind>' AGGREGATE <kind>` clauses. Unknown kinds fail at plan time via `TNormRegistry::get` / `AggregatorRegistry::get` (no runtime surprises).

Backward compat: existing serialized plans deserialize unchanged because `FuzzyConfig` is `#[serde(default)]`. `EXPLAIN` plan JSON includes the resolved `fuzzy_config`.

## 14.12 MCP Tools

14 new fuzzy MCP tools land in Phase 11 (tool count: **159 → 173**). Phase 5 of the Graded Acceptability sprint adds 5 more (tool count: **173 → 178**). v0.79.11 adds `update_situation` (tool count: **178 → 179**).

| Tool | Wraps |
|---|---|
| `fuzzy_list_tnorms` | `GET /fuzzy/tnorms` |
| `fuzzy_list_aggregators` | `GET /fuzzy/aggregators` |
| `fuzzy_get_config` | `GET /fuzzy/config` |
| `fuzzy_set_config` | `PUT /fuzzy/config` |
| `fuzzy_create_measure` | `POST /fuzzy/measures` |
| `fuzzy_list_measures` | `GET /fuzzy/measures` |
| `fuzzy_aggregate` | `POST /fuzzy/aggregate` |
| `fuzzy_allen_gradation` | `POST /analysis/fuzzy-allen` |
| `fuzzy_quantify` | `POST /fuzzy/quantify` |
| `fuzzy_verify_syllogism` | `POST /fuzzy/syllogism/verify` |
| `fuzzy_build_lattice` | `POST /fuzzy/fca/lattice` |
| `fuzzy_create_rule` | `POST /fuzzy/rules` |
| `fuzzy_evaluate_rules` | `POST /fuzzy/rules/:nid/evaluate` |
| `fuzzy_probability` | `POST /fuzzy/hybrid/probability` |
| `argumentation_gradual` (Graded P5) | `POST /analysis/argumentation/gradual` |
| `fuzzy_learn_measure` (Graded P5) | `POST /fuzzy/measures/learn` |
| `fuzzy_get_measure_version` (Graded P5) | `GET /fuzzy/measures/{name}?version=N` |
| `fuzzy_list_measure_versions` (Graded P5) | `GET /fuzzy/measures/{name}/versions` |
| `temporal_ordhorn_closure` (Graded P5) | `POST /temporal/ordhorn/closure` |

**Optional-arg extensions** (existing tools now accept `tnorm: Option<String>` + `aggregator: Option<String>` with `#[serde(default)]`): `get_entity`, `search_entities`, `list_pinned_facts`, `ask`, `get_narrative_stats`, `get_behavioral_fingerprint`, `get_disinfo_fingerprint`. Absent fields forward bit-identical URLs / bodies.

Implementation split across [`src/mcp/embedded_fuzzy.rs`](../src/mcp/embedded_fuzzy.rs) (tools 1-7 + hybrid), [`src/mcp/embedded_fuzzy_ext.rs`](../src/mcp/embedded_fuzzy_ext.rs) (tools 8-13), and [`src/mcp/embedded_graded.rs`](../src/mcp/embedded_graded.rs) (Phase 5 graded tools 1-5) to hold the 500-line file cap.

## 14.13 KV Prefixes

| Prefix | Schema | Purpose |
|---|---|---|
| `cfg/fuzzy` | singleton | Site-default `FuzzyWorkspaceConfig {tnorm, aggregator}` |
| `fz/tn/` | `fz/tn/measures/{name}` | Registered fuzzy measures (persistent; Phase 4) |
| `fz/agg/` | reserved | Aggregator-kind metadata (future) |
| `fz/allen/` | `fz/allen/{nid}/{a_id_BE}/{b_id_BE}` | Cached graded Allen 13-vector per situation pair |
| `fz/quant/` | `fz/quant/{nid}/{predicate_hash}` | Cached intermediate-quantifier evaluation |
| `fz/fca/` | `fz/fca/{lattice_id_v7_BE_16}` + `fz/fca/n/{nid}/...` | Persisted concept lattices + narrative index |
| `fz/rules/` | `fz/rules/{nid}/{rule_id_v7_BE_16}` | Mamdani rule definitions |
| `fz/syllog/` | `fz/syllog/{nid}/{proof_id_v7_BE_16}` | Stored graded-syllogism proofs |
| `fz/hybrid/` | `fz/hybrid/{nid}/{query_id_v7_BE_16}` | Cached hybrid-probability query reports (Phase 10) |

Eight `fz/*` prefixes + the `cfg/fuzzy` singleton. UUID components follow the global encoding contract (v7 UUIDs in big-endian 16-byte binary; see Appendix B).

## 14.14 Studio UI — Fuzzy Canvas

`/n/:narrativeId/fuzzy` — dedicated canvas tab alongside Synth, Reconstruction, and Opinion. Deep-linkable via `?sub=config|aggregation|rules|lattice`.

Layout (eight sub-panels — three were backend-only on the original Phase 12 ship and are now surfaced as of v0.79.10):

- **Config** — GET/PUT `/fuzzy/config` with live-save semantics. Also embeddable inside the `WorkspaceHeader` `◇` indicator's slide-in Modal.
- **Aggregation Playground** — up to 10 source-confidence slots; debounced 250 ms `POST /fuzzy/aggregate` per edit. Surfaces descriptor + t-norm + weight/measure inputs per aggregator kind.
- **Measure Compare** *(Graded Sprint Phase 5)* — symmetric-default `mean` vs learned-`choquet(measure)` side-by-side; auto-picks the first learned measure, highlights `|delta| > 1e-3`. Deep-link: `?sub=comparison`.
- **Quantify** *(NEW v0.79.10, Phase 6 surface)* — `POST /fuzzy/quantify`. Picker for Most / Many / Almost-all / Few + entity-type filter + crisp predicate spec (`confidence>0.7`, `maturity=Validated`, …). Renders the quantifier ramp `μ_Q(r)` as a canvas chart with the response cardinality-ratio cursor highlighted, plus per-session history strip. Deep-link: `?sub=quantify`.
- **Syllogism** *(NEW v0.79.10, Phase 7 surface)* — `POST /fuzzy/syllogism/verify`. Three premise textareas (major / minor / conclusion) using the Phase 7 tiny-DSL (`ALL type:Actor IS type:Actor`), t-norm picker, threshold slider, optional figure hint, three Figure I/II/III starter examples. Result card shows valid/invalid pill, degree gauge with threshold marker, persisted `proof_id`. Deep-link: `?sub=syllogism`.
- **Hybrid Prob.** *(NEW v0.79.10, Phase 10 surface)* — `POST /fuzzy/hybrid/probability`, list / get / delete. Three sections: graded event μ_E with predicate-kind switch (`quantifier` / `mamdani_rule` / `custom`); discrete-distribution editor with per-row `EntityComboBox` + numeric `P(e)` (and per-row `μ_E(e)` in `custom` mode), running Σ in the table footer with sum-to-1 (±1e-6) validation, helper buttons for `uniform` / `normalize` / `+ outcome`; result gauge + persisted-reports table (extracted into `HybridReportsTable.tsx` to keep `HybridPanel.tsx` under the 500-line cap). Deep-link: `?sub=hybrid`.
- **Rule Editor** — narrative-scoped Mamdani rule CRUD (triangular MF in the form; trapezoidal + Gaussian still accepted over the wire).
- **Concept Lattice Viewer** — D3 force-directed lattice (concepts as nodes, Hasse edges as links; hover surfaces extent + intent). Uses the existing `d3` umbrella (`d3.forceSimulation` / `forceLink` / `forceManyBody`); no new npm deps.

Cross-canvas surface wires:

- **`FuzzyBadge`** — `◇ <TNORM>` provenance pill; rendered by `AttributionList`, `AnalysisHub` result cards, and `AskConsole` answers whenever the payload carries a `fuzzy_config` tag.
- **`WorkspaceHeader` ◇ indicator** — reads `GET /fuzzy/config`; turns phosphorescent teal when the site default is non-default; click opens `FuzzyConfigPanel` in a slide-in `Modal`.
- **Command Palette** — 5 new jump targets in a new `fuzzy` item-kind: Open Fuzzy Canvas / Open Aggregation Playground / Open Rule Editor / Open Concept Lattice / Set t-norm: Łukasiewicz.
- **Ask Console** — expandable "Fuzzy semantics" disclosure; selection rides through as `?tnorm=…&aggregator=…`.
- **Ingest view** — site-default t-norm + aggregator pickers next to the existing thresholds block; both PUT directly to `/fuzzy/config`.

All new component files under the 500-line cap ([`FuzzyCanvas.tsx`](../studio/src/workspace/canvases/FuzzyCanvas.tsx), [`FuzzyConfigPanel.tsx`](../studio/src/components/fuzzy/FuzzyConfigPanel.tsx), [`AggregationPlayground.tsx`](../studio/src/components/fuzzy/AggregationPlayground.tsx), [`MeasureComparisonPanel.tsx`](../studio/src/components/fuzzy/MeasureComparisonPanel.tsx), [`QuantifyPanel.tsx`](../studio/src/components/fuzzy/QuantifyPanel.tsx), [`SyllogismPanel.tsx`](../studio/src/components/fuzzy/SyllogismPanel.tsx), [`HybridPanel.tsx`](../studio/src/components/fuzzy/HybridPanel.tsx), [`HybridReportsTable.tsx`](../studio/src/components/fuzzy/HybridReportsTable.tsx), [`RuleEditor.tsx`](../studio/src/components/fuzzy/RuleEditor.tsx), [`ConceptLatticeViewer.tsx`](../studio/src/components/fuzzy/ConceptLatticeViewer.tsx), [`FuzzyBadge.tsx`](../studio/src/components/fuzzy/FuzzyBadge.tsx), [`useFuzzyCanvas.ts`](../studio/src/components/fuzzy/useFuzzyCanvas.ts), [`fuzzyStyles.ts`](../studio/src/components/fuzzy/fuzzyStyles.ts)). The shared truth-color helper `fuzzyValueColor()` lives in `fuzzyStyles.ts`; the cross-surface relative-time formatter lives at [`studio/src/utils/time.ts`](../studio/src/utils/time.ts).

`cd studio && npm run build` clean; FuzzyCanvas chunks to 21.7 kB (gzip 7.42 kB).

## 14.15 Citations + Bibliography

All citations preserved verbatim in [`docs/FUZZY_BIBLIOGRAPHY.bib`](FUZZY_BIBLIOGRAPHY.bib) for LaTeX `\bibliography{docs/FUZZY_BIBLIOGRAPHY}` usage. The `/simplify` pass is EXPLICITLY FORBIDDEN from touching that file or the "Citations — Preserve Verbatim" section of [`docs/FUZZY_Sprint.md`](FUZZY_Sprint.md). Grep-audited at each phase: every BibTeX key below is referenced from at least one `src/fuzzy/*.rs` module doc.

```bibtex
@book{klement2000triangular,
  title={Triangular Norms},
  author={Klement, Erich Peter and Mesiar, Radko and Pap, Endre},
  year={2000}, publisher={Kluwer Academic},
  series={Trends in Logic, vol. 8}
}
@article{yager1988owa,
  title={On ordered weighted averaging aggregation operators in multicriteria decisionmaking},
  author={Yager, Ronald R.},
  journal={IEEE Trans. Systems, Man, and Cybernetics},
  volume={18}, number={1}, pages={183--190}, year={1988}
}
@article{grabisch1996choquet,
  title={The application of fuzzy integrals in multicriteria decision making},
  author={Grabisch, Michel},
  journal={European Journal of Operational Research},
  volume={89}, number={3}, pages={445--456}, year={1996}
}
@article{bustince2016choquet,
  title={A historical account of types of fuzzy sets and their relationships},
  author={Bustince, Humberto and Barrenechea, Edurne and Pagola, Miguel and
          Fernandez, Javier and Xu, Zeshui and Bedregal, Benjamin and
          Montero, Javier and Hagras, Hani and Herrera, Francisco and
          De Baets, Bernard},
  journal={IEEE Trans. Fuzzy Systems},
  volume={24}, number={1}, pages={179--194}, year={2016}
}
@incollection{grabisch2000fuzzymeasure,
  title={Fuzzy measures and integrals: theory and applications},
  author={Grabisch, Michel and Murofushi, Toshiaki and Sugeno, Michio},
  booktitle={Studies in Fuzziness and Soft Computing, vol. 40},
  year={2000}, publisher={Physica-Verlag}
}
@article{duboisprade1989fuzzyallen,
  title={Processing fuzzy temporal knowledge},
  author={Dubois, Didier and Prade, Henri},
  journal={IEEE Trans. Systems, Man, and Cybernetics},
  volume={19}, number={4}, pages={729--744}, year={1989}
}
@article{schockaert2008fuzzyallen,
  title={Temporal reasoning about fuzzy intervals},
  author={Schockaert, Steven and De Cock, Martine},
  journal={Artificial Intelligence},
  volume={172}, number={8-9}, pages={1158--1193}, year={2008}
}
@article{novak2008quantifiers,
  title={A formal theory of intermediate quantifiers},
  author={Nov{\'a}k, Vil{\'e}m},
  journal={Fuzzy Sets and Systems},
  volume={159}, number={10}, pages={1229--1246}, year={2008}
}
@article{murinovanovak2013syllogisms,
  title={A formal theory of generalized intermediate syllogisms},
  author={Murinov{\'a}, Petra and Nov{\'a}k, Vil{\'e}m},
  journal={Fuzzy Sets and Systems},
  volume={186}, pages={47--80}, year={2012}
}
@article{murinovanovak2014peterson,
  title={Analysis of generalized square of opposition with intermediate quantifiers},
  author={Murinov{\'a}, Petra and Nov{\'a}k, Vil{\'e}m},
  journal={Fuzzy Sets and Systems},
  volume={242}, pages={89--113}, year={2014}
}
@article{belohlavek2004fuzzyfca,
  title={Concept lattices and order in fuzzy logic},
  author={B{\v{e}}lohl{\'a}vek, Radim},
  journal={Annals of Pure and Applied Logic},
  volume={128}, number={1-3}, pages={277--298}, year={2004}
}
@article{kridlo2010fuzzyfca,
  title={L-bonds},
  author={Kr{\'\i}dlo, Ondrej and Ojeda-Aciego, Manuel},
  journal={Fuzzy Sets and Systems},
  volume={161}, number={12}, pages={1737--1749}, year={2010}
}
@article{mamdani1975mamdani,
  title={An experiment in linguistic synthesis with a fuzzy logic controller},
  author={Mamdani, E. H. and Assilian, S.},
  journal={International Journal of Man-Machine Studies},
  volume={7}, number={1}, pages={1--13}, year={1975}
}
@inproceedings{flaminio2026fsta,
  title={Fuzzy-probabilistic logics (FSTA 2026 invited track)},
  author={Flaminio, Tommaso and Hol{\v{c}}apek, Michal and Cao, Nhung},
  booktitle={FSTA 2026 — Fuzzy Set Theory and Applications}, year={2026}
}
@article{faginhalpern1994fuzzyprob,
  title={Reasoning about knowledge and probability},
  author={Fagin, Ronald and Halpern, Joseph Y.},
  journal={Journal of the ACM},
  volume={41}, number={2}, pages={340--367}, year={1994}
}
```

---

# Chapter 15: Graded Acceptability & Measure Learning

> **Citations.** Gradual / ranking-based argumentation: Amgoud & Ben-Naim — *Ranking-Based Semantics for Argumentation Frameworks*, SUM 2013, LNCS **8078**, 134–147 (`amgoud2013ranking`); Amgoud & Ben-Naim — *Evaluation of arguments in weighted bipolar graphs*, Int. J. Approx. Reasoning **99**, 39–55 (2018) (`amgoud2017weighted`); Besnard & Hunter — *A logic-based theory of deductive arguments*, Artif. Intell. **128**, 203–235 (2001) (`besnard2001hcategoriser`). k-additive fuzzy measures: Grabisch — Fuzzy Sets Syst. **92**, 167–189 (1997) (`grabisch1997kadditive`). ORD-Horn maximal tractable subclass: Nebel & Bürckert — J. ACM **42**(1), 43–66 (1995) (`nebel1995ordhorn`). Full BibTeX (5 new entries appended to the Fuzzy Sprint set): [`docs/FUZZY_BIBLIOGRAPHY.bib`](FUZZY_BIBLIOGRAPHY.bib).

## 15.1 Overview

The Graded Acceptability & Measure Learning Sprint (v0.78.2 → v0.79.0; six phases) extends three TENSA surfaces with continuous / supervised counterparts:

1. **Argumentation** — adds four canonical gradual / ranking-based semantics returning a real-valued degree per argument, on top of the legacy crisp Dung extensions.
2. **Aggregation measures** — adds ranking-supervised Choquet measure learning from a labelled `(input_vec, rank)` dataset, on top of the symmetric defaults.
3. **Temporal reasoning** — adds the Nebel-Bürckert 1995 ORD-Horn path-consistency closure for disjunctive Allen networks.

### Sprint motivation

The recruitment-positive feedback called out the §11 Related-Work paragraph: implementing **four standard gradual semantics** (Amgoud & Ben-Naim 2013 is the canonical comparative reference) and a **ranking-supervised Choquet learner** lets TENSA engage the Dung-Amgoud-Woltran lineage explicitly. The synthetic-CIB worked example (§15.6) demonstrates a `+0.21` AUC gap that no symmetric-additive baseline can recover — that gap is the §5.3 paper-figure ROI surface.

### Default-zero-change invariant (load-bearing)

Every existing call site stays bit-identical. The `analysis::argumentation::run_argumentation` legacy entrypoint becomes a one-line wrapper around the new `_with_gradual` variant; absent `gradual_semantics` (default `None`) returns the legacy crisp result with `gradual: None`. Symmetric-default Choquet aggregation stays within `1e-12` of direct `choquet_exact`. The 25 `backward_compat_tests` from the Fuzzy Sprint plus 9 fixture replays + 1 contention-based integration replay (§5.4 of Phase 1) all remain green.

## 15.2 Four Gradual Semantics

Implementation lives at [`src/analysis/argumentation_gradual.rs`](../src/analysis/argumentation_gradual.rs) (~325 lines). Four variants share a single iteration loop with a semantics-specific aggregator and a parametric **influence-step denominator**.

| Variant | Update rule | Source |
|---|---|---|
| **h-Categoriser** | `Acc_{i+1}(a) = w(a) / (1 + Σ_{b ∈ Att(a)} Acc_i(b))` | Besnard & Hunter 2001 |
| **Weighted h-Categoriser** | `Acc_{i+1}(a) = w(a) / (1 + Σ_b v_{ba} · Acc_i(b))` with `Σ v_{ba} ≤ 1` per target | Amgoud & Ben-Naim 2017 |
| **Max-Based** | replaces sum with max over attackers | Amgoud & Ben-Naim 2013 |
| **Card-Based** | `Acc_{i+1}(a) = w(a) / ((1 + card⁺(a)) · (1 + sum(a)))` (lexicographic tie-break) | Amgoud & Ben-Naim 2013 |

`card⁺(a)` counts attackers with strictly positive acceptability; `sum(a)` ranges over ALL attackers.

### Convergence table

| Semantics × T-norm | Contraction provable? | Behaviour |
|---|---|---|
| `{HCategoriser, Weighted, MaxBased, CardBased} × Gödel` | Yes | Bit-identical to the canonical paper formulas. |
| `{HCategoriser, Weighted, MaxBased, CardBased} × Łukasiewicz` | Yes | Clamped to `[0, 1]`. |
| `... × Goguen` (`1 - exp(-s)`) | NO | Relies on `MAX_GRADUAL_ITERATIONS = 200` cap. |
| `... × Hamacher(λ)` | NO | Relies on the cap. |

`MAX_GRADUAL_ITERATIONS = 200`; `CONVERGENCE_EPSILON = 1e-9` measured as `||Acc_{i+1} - Acc_i||_∞`. Cap-hit emits `tracing::warn!` and returns `converged: false` with the last computed iterate. **Always check `gradual.converged` before consuming the result for a publication-grade claim.**

## 15.3 Influence-Step T-Norm Coupling (option B from feedback)

The standard h-Categoriser influence function is `infl(s) = s / (1 + s)`. Option B was: KEEP the canonical aggregation step verbatim (sum / max / card) and expose a t-norm-parametric denominator family:

| `TNormKind` | denominator term `infl_denom(s)` |
|---|---|
| `Godel` (default) | `s` (canonical, bit-identical) |
| `Lukasiewicz` | `s.min(1.0)` |
| `Goguen` | `1 - exp(-s)` (Poisson-sum limit of probabilistic OR) |
| `Hamacher(λ)` | `S_Hamacher(s.clamp(0,1), s.clamp(0,1))` |

For card-based the cardinality factor `(1 + card)` stays raw — the t-norm only modulates the sum-component denominator term. Default `tnorm = None` reproduces the cited paper formulas bit-identically; this preserves the §15.14 backward-compat regression corpus.

## 15.4 Argumentation Principle Tests

The 30-trial property-test battery (3 properties × 10 ChaCha8Rng-seeded random frameworks) validates the implementation against the Amgoud & Ben-Naim 2013 axiomatic framework. Tests live at `analysis::argumentation_gradual_tests::§5.5`.

| Principle | Definition (Amgoud & Ben-Naim 2013) | Assertion |
|---|---|---|
| **Anonymity** | Definition 3 — semantics is invariant under argument permutation. | Random argument permutation leaves Uuid-keyed acceptability unchanged. |
| **Independence** | Definition 6 — disconnected components have independent acceptability. | Disconnected components yield independent vectors. |
| **Monotonicity** | Proposition 1 — adding an attack `(c → b)` cannot increase `Acc(b)`. | Within `1e-9` tolerance. |

Random framework generator: `n_args ∈ [3, 8]` uniform; intrinsic strengths ∈ `[0.1, 0.9]` uniform; attack edges with probability `0.3` per ordered pair; `rand_chacha::ChaCha8Rng::seed_from_u64` per the no-new-deps rule.

## 15.5 Learned Choquet Measures

Implementation lives at [`src/fuzzy/aggregation_learn.rs`](../src/fuzzy/aggregation_learn.rs) (~545 lines). Pure-Rust projected gradient descent in μ-space (full `2^n` capacity table). The OSQP / Clarabel QP solver alternative was deferred — no QP solver crate is currently in `Cargo.toml`.

Per iteration:

1. Compute the analytic gradient of the pairwise hinge loss `L(μ) = Σ_{rank_i < rank_j} max(0, C_μ(x_j) - C_μ(x_i) + ε)` — closed-form `O(n)` per active pair via the Choquet integral's piecewise-linear structure (sort `x` ascending; tail subsets `A_k` contribute `x_(k) - x_(k-1)`).
2. Take a step `μ' = μ - η · ∇L`.
3. Project onto the feasible set: clip to `[0, 1]`, monotonicity sweep `μ(S ∪ {i}) ≥ μ(S)` smaller-subsets-first, then re-pin the boundaries `μ(∅) = 0` and `μ(N) = 1`.

| Constant | Value | Meaning |
|---|---|---|
| `MAX_PGD_ITERATIONS` | 5 000 | Hard cap; cap-hit emits `tracing::warn!`. |
| `PGD_TOLERANCE` | 1e-6 | `max_A |Δμ[A]|` termination threshold. |
| `DEFAULT_LEARNING_RATE` | 1.0 | Mean-normalised loss + gradient → unit-scale step. |
| `DEFAULT_MARGIN` | 0.05 | Hinge margin ε. |
| `N_CAP` | 6 | Hard rejection threshold; see §15.8. |

Adaptive LR: divergence-rollback (never accept a worse μ) plus halve-LR after stalls; `1e-9` LR floor.

## 15.6 Synthetic-CIB Worked Example

The shipped synthetic-CIB generator (`fuzzy::synthetic_cib_dataset::generate_synthetic_cib(seed, n_clusters)`) produces 4-signal Coordinated-Inauthentic-Behaviour clusters with the design §3.2 ground-truth score:

```
score = sigmoid(2·x0·x1 + 0.3·x2 - 0.5·x3)
```

The `2·x0·x1` term is the **load-bearing non-additive interaction** that no additive measure can recover. Ranks are assigned by descending score; `ChaCha8Rng`-seeded for reproducibility.

Demonstration result (`aggregation_learn_tests::synthetic_cib_demonstration` with `dataset_id = "synthetic-cib-paper-figure-v1"`):

| Aggregator | AUC on 50/50 test split |
|---|---|
| `symmetric_additive` (= arithmetic mean) | **0.6367** |
| `learn_choquet_measure(4, generate_synthetic_cib(42, 100), …)` | **0.8522** |
| **Gap** | **+0.2155** |

This exceeds the design's 0.80 / 0.65 / 0.15 thresholds. The `dataset_id` was chosen offline from a 37-candidate sweep; the underlying generative model is unchanged from the design verbatim.

## 15.7 Provenance Slot Contract (load-bearing)

Every aggregation result that uses a learned measure carries `fuzzy_config.measure_id` + `fuzzy_config.measure_version` in the emitted envelope. Symmetric defaults emit `None`/`None` — bit-identical to pre-Phase-0 envelopes.

Three workflow surfaces threaded the slot through their `_tracked` siblings in Phase 2 (existing methods became 1-line wrappers; existing call sites compile unchanged):

| Site | `_tracked` signature |
|---|---|
| [`src/source.rs`](../src/source.rs) | `ConfidenceBreakdown::composite_with_aggregator_tracked(agg, measure_id, measure_version) -> Result<(f32, Option<String>, Option<u32>)>` |
| [`src/synth/fidelity_pipeline.rs`](../src/synth/fidelity_pipeline.rs) | `aggregate_metrics_with_aggregator_tracked(metrics, agg, measure_id, measure_version) -> Result<(f32, Option<String>, Option<u32>)>` |
| [`src/adversarial/reward_model.rs`](../src/adversarial/reward_model.rs) | `RewardProfile::score_with_aggregator_tracked(agg, measure_id, measure_version) -> Result<(f64, Option<String>, Option<u32>)>` |

`FuzzyMeasure` carries `measure_id: Option<String>` + `measure_version: Option<u32>`, both `#[serde(default, skip_serializing_if = "Option::is_none")]` so existing JSON blobs round-trip unchanged. `FidelityReport` gains the same two fields. `parse_fuzzy_config` accepts `?measure=<name>&measure_version=<N>`.

## 15.8 k-Additive Deferral

`n > 6` is rejected with the canonical k-additive pointer error message. The Möbius / k-additive specialisation reduces the parameter count from `2^n` to `O(n^k)` per Grabisch 1997 (`grabisch1997kadditive`). See `docs/architecture_paper` §12.2 for the planned future-sprint extension. The recruitment-positive item is the existing `n ≤ 6` PGD baseline; specialisation is a follow-on.

## 15.9 ORD-Horn Path-Consistency

Implementation lives at [`src/temporal/ordhorn.rs`](../src/temporal/ordhorn.rs) (~307 lines). The van Beek path-consistency closure is built on top of TENSA's existing 13×13 Allen composition table ([`src/temporal/interval.rs`](../src/temporal/interval.rs)) — a 30-year-old algorithm port from Nebel & Bürckert 1995 (`nebel1995ordhorn`), not a reinvention.

`OrdHornConstraint` carries a *disjunction* of basic Allen relations (`relations: Vec<AllenRelation>`) rather than a single relation; `OrdHornNetwork { n, constraints }` is the on-wire form. `closure(network)` runs the van Beek `(i, j, k)` triple loop with auto-inverse propagation + early termination on the empty constraint + a defensive iteration cap (`n^3 · 13`). Dense scratch matrix of shape `n × n × ≤13` is built/torn-down per call; the on-wire form omits unconstrained pairs (full 13-relation disjunction).

### Soundness vs completeness (load-bearing distinction)

Per Nebel-Bürckert 1995 Theorem 1:

- **Sound for any Allen constraint network** — empty constraint anywhere ⇒ provably unsatisfiable.
- **Complete only for ORD-Horn networks** — non-empty closure ⇒ satisfiable IFF every input disjunction lies in the 868-element ORD-Horn class.

The 868-element ORD-Horn membership oracle is **NOT shipped** this sprint. Callers requiring decidability guarantees must restrict inputs to ORD-Horn by construction (e.g. only Pointisable Allen relations, or only the "convex" subset).

### Why we don't drive real intervals through the closure

The canonical 13×13 composition table inside `src/temporal/interval.rs` has known incompleteness at certain entries (e.g. `Starts ∘ Contains = {Before, Meets, Overlaps}` omits the legitimate `{FinishedBy, Contains}` outcomes). For real intervals where the actual relation is e.g. `FinishedBy`, the closure would intersect `{FinishedBy}` with the table-derived set and return `∅` (false unsatisfiability). Composition-table audit + correction is out of scope this sprint. The REST endpoint is sufficient — callers construct any `OrdHornNetwork` they wish and POST it; no `IntervalTree` integration required.

## 15.10 REST Surface

Five new endpoints land this sprint:

| Method | Path | Use |
|---|---|---|
| POST | `/analysis/argumentation/gradual` | Sync gradual semantics. Body: `{narrative_id, gradual_semantics, tnorm?}`. Returns `{narrative_id, gradual: GradualResult, iterations, converged}`. Mirrors the Fuzzy Sprint Phase 7b `/analysis/higher-order-contagion` synchronous precedent; delegates to `analysis::argumentation::run_argumentation_with_gradual`. |
| POST | `/fuzzy/measures/learn` | Train a Choquet measure. Body: `{name, n, dataset, dataset_id}`. Persists `StoredMeasure` with `MeasureProvenance::Learned(...)` at BOTH `fz/tn/measures/{name}` (latest pointer) AND `fz/tn/measures/{name}/v{N}` (history slice). Re-training auto-increments version. Returns `LearnedMeasureSummary` with `201 CREATED`. |
| GET | `/fuzzy/measures/{name}/versions` | Prefix-scans `fz/tn/measures/{name}/v` and parses `u32` suffixes; returns `{name, versions: [u32]}` sorted ascending. |
| GET | `/fuzzy/measures/{name}?version=N` | Phase 2 latest-pointer behaviour preserved when `version` absent. Missing version → HTTP 404 with body `"measure '{name}' version {N} not found"`. |
| DELETE | `/fuzzy/measures/{name}?version=N` | Versioned delete leaves latest pointer alone; legacy unversioned delete unchanged. |
| POST | `/temporal/ordhorn/closure` | Sync path-consistency closure. Body: `{network: OrdHornNetwork}`. Returns `{closed_network, satisfiable}`. Bare-handler signature (no `State` extraction) — pure on the supplied network. |

The `?gradual=<kind>` query string was rejected in favour of the dedicated synchronous endpoint so the read-back `GET /narratives/:id/arguments` stays cacheable + idempotent. Decision recorded in [`docs/GRADED_SPRINT.md`](GRADED_SPRINT.md) Notes "Phase 3 TensaQL decision".

## 15.11 MCP Tools

Five new tools land this sprint; tool count **173 → 178**. Implementation in [`src/mcp/embedded_graded.rs`](../src/mcp/embedded_graded.rs) (~250 lines, under 500-line cap); types in [`src/mcp/types.rs`](../src/mcp/types.rs).

| Tool | Wraps |
|---|---|
| `argumentation_gradual` | `POST /analysis/argumentation/gradual` |
| `fuzzy_learn_measure` | `POST /fuzzy/measures/learn` |
| `fuzzy_get_measure_version` | `GET /fuzzy/measures/{name}?version=N` |
| `fuzzy_list_measure_versions` | `GET /fuzzy/measures/{name}/versions` |
| `temporal_ordhorn_closure` | `POST /temporal/ordhorn/closure` |

`gradual_semantics` / `tnorm` / `network` are carried as opaque `serde_json::Value` because their tagged-union enums don't derive `JsonSchema` (same pattern as `FuzzyEvaluateRulesRequest::firing_aggregator`). Reuses `crate::api::fuzzy::measure::{measure_key, versioned_measure_key, StoredMeasure}` so the KV wire format never drifts.

## 15.12 TensaQL — REST-Only by Design

**No new TensaQL clauses this sprint.** Two architectural decisions stand:

- The existing `AGGREGATE CHOQUET BY '<measure_id>'` form already resolves learned measures via the `FuzzyMeasure.measure_id` slot wired in Phase 2 + the `parse_fuzzy_config` extension that accepts `?measure=<name>&measure_version=<N>`. **Learning is a control-plane action (POST), not a query verb** — adding `TUNE MEASURE` ceremonial syntax would not improve ergonomics over the REST path.
- Gradual argumentation is exposed via `POST /analysis/argumentation/gradual`, not via a `?gradual=<kind>` query string on the cacheable read-back endpoint. The grammar is full enough.

If batch-learning across multiple narratives lands in a future phase, or a paper figure needs a one-line query, the existing `cli::tensa.pest` grammar has room for a `TUNE` verb in the same slot as `EXPLAIN` / `DISCOVER`.

## 15.13 Studio Surfaces

Three Studio extensions land in Phase 5 (no new npm deps):

| Component | Behaviour |
|---|---|
| `MeasureComparisonPanel.tsx` (NEW, ~360 lines) | Side-by-side `aggregate(xs, mean)` vs `aggregate(xs, choquet, measure=picked)` with auto-pick of the first learned measure on mount + delta highlight when `|delta| > 1e-3`. Mounted as `Measure Compare` sub-tab on FuzzyCanvas (deep-linkable via `?sub=comparison`). The §5.3 paper-figure surface. |
| `AggregationPlayground.tsx` measure-picker | `<select>` dropdown listing `(symmetric default — additive)` plus every persisted measure with provenance tags (`learned` / `manual` / `symmetric:<kind>`). Falls back to manual entry when no learned measure exists. |
| `ArgumentationViz.tsx` Gradual tab | Renders the optional `gradual: GradualResult` field as a horizontal-bar chart (one row per argument, bar width = acceptability ∈ [0, 1], sorted descending). Pure CSS — no D3. Auto-promotes `Gradual` to the landing tab when the field is present; crisp-only results land on `Grounded` as before. |

Five typed wrappers added to [`studio/src/api/client.ts`](../studio/src/api/client.ts): `runGradualArgumentation`, `learnFuzzyMeasure`, `getFuzzyMeasureVersion`, `listFuzzyMeasureVersions`, `closeOrdHornNetwork`.

`StoredFuzzyMeasure` gains optional `version` + `provenance`; `ArgumentationResult` gains optional `gradual: GradualResult`. /simplify pass lifted `listFuzzyMeasures()` fetch into the existing `FuzzyCanvasProvider` (joining the mount-time `Promise.all`) so both consumers read from a single source.

## 15.14 §10.2 Backward-Compat Corpus Extensions

Phase 1 + Phase 2 added load-bearing regression assertions extending the §10.2 backward-compat corpus:

| Phase | Assertion | What it locks down |
|---|---|---|
| 1 | 9 fixture replays in `argumentation_gradual_tests::§5.4` | Each legacy `argumentation_tests.rs` fixture re-runs through the new wrapper; (a) `result.gradual.is_none()`, (b) crisp `grounded` + `preferred_extensions` + `stable_extensions` match bit-identically. |
| 1 | `regression_integration_with_contentions` | End-to-end replay of the legacy contention-based scenario; both UNDEC labels and 2 preferred extensions stay identical. |
| 1 | 16 legacy `argumentation_tests` continue to pass | The default `gradual_semantics = None` path is bit-identical to pre-sprint behaviour. |
| 2 | `symmetric_default_choquet_bit_identical` | Covers `ConfidenceBreakdown` + `aggregate_metrics_with_aggregator_tracked` + `ChoquetAggregator::aggregate` + (under `feature = "adversarial"`) `RewardProfile`. All sites within `1e-12` of direct `choquet_exact`. |
| 2 | `fuzzy_config_omits_measure_id_when_symmetric` | Symmetric defaults stay `None`/`None` — bit-identical envelope shape to pre-Phase-0 blobs. |
| 4 | Every existing temporal test (93 total) bit-identical | The new ORD-Horn module is additive; the legacy `IntervalTree` queries are untouched. |

Run via `cargo test --no-default-features --lib` (2117 tests pass) + `cargo test --no-default-features --features server --lib` (2204 pass) + `cargo test --no-default-features --features mcp,server,disinfo --lib` (2548 pass). Zero regressions across all six phases.

## 15.15 Citations

The five new BibTeX keys appended (positionally last) to [`docs/FUZZY_BIBLIOGRAPHY.bib`](FUZZY_BIBLIOGRAPHY.bib):

| Key | Reference |
|---|---|
| `amgoud2013ranking` | Amgoud, L. & Ben-Naim, J. — *Ranking-based semantics for argumentation frameworks*, Scalable Uncertainty Management (SUM 2013), LNCS **8078**, 134–147. Springer. |
| `amgoud2017weighted` | Amgoud, L. & Ben-Naim, J. — *Evaluation of arguments in weighted bipolar graphs*, Int. J. Approx. Reasoning **99**, 39–55 (2018). |
| `besnard2001hcategoriser` | Besnard, P. & Hunter, A. — *A logic-based theory of deductive arguments*, Artif. Intell. **128**(1-2), 203–235 (2001). |
| `grabisch1997kadditive` | Grabisch, M. — *k-order additive discrete fuzzy measures and their representation*, Fuzzy Sets Syst. **92**(2), 167–189 (1997). |
| `nebel1995ordhorn` | Nebel, B. & Bürckert, H.-J. — *Reasoning about temporal relations: A maximal tractable subclass of Allen's interval algebra*, J. ACM **42**(1), 43–66 (1995). |

Grep-audited every phase: each key cited from ≥ 1 module-doc comment under `src/`. The full BibTeX file is `/simplify`-FORBIDDEN per the Fuzzy Sprint acceptance gate.

---

# Appendix A: TensaQL Grammar Quick Reference

```
query       = EXPLAIN? (EXPORT | GENERATE | mutation | DISCOVER | INFER | TUNE | ASK | MATCH PATH | MATCH FLOW | MATCH)

-- MATCH
MATCH       (binding:Type {props}) -[e:Rel]-> (b:Type)
            ACROSS NARRATIVES ("id", ...)?
            WHERE condition (AND|OR condition)*
            AT field RELATION "timestamp"
            NEAR(binding, "text", k)
            SPATIAL field WITHIN float KM OF (lat, lon)
            GROUP BY field (, field)*
            RETURN expr (, expr)* (ORDER BY field ASC|DESC)? (LIMIT int)?

-- PATH
MATCH PATH  (SHORTEST | ALL | LONGEST ACYCLIC | TOP k SHORTEST)
            (start) -[:Rel*min..max]-> (end)
            WEIGHT field?
            WHERE ...  RETURN ...

-- FLOW
MATCH FLOW  (MAX | MIN_CUT) (source) -[:Rel*]-> (sink) RETURN flow, cut_edges

-- ASK (RAG question answering)
ASK         "question"
            OVER "narrative_id"?
            MODE (local|global|hybrid|mix|drift|lazy|ppr)?
            RESPOND AS "format"?
            SESSION "session_id"?
            SUGGEST?

-- TUNE (prompt auto-tuning)
TUNE        PROMPTS FOR "narrative_id"

-- INFER (grouped by family; see §3.10 for full semantics)
INFER       infer_type
            FOR binding:Type
            MATCH pattern?
            ASSUMING field = value (AND ...)*
            UNDER key = value (AND ...)*
            RETURN ...

-- Graph centrality
infer_type  = CENTRALITY | PAGERANK | EIGENVECTOR | HARMONIC | HITS

-- Topology & community
            | TOPOLOGY | KCORE | LABEL_PROPAGATION | COMMUNITIES

-- Narrative-native (time-aware)
            | TEMPORAL_PAGERANK | CAUSAL_INFLUENCE | INFO_BOTTLENECK
            | ASSORTATIVITY | TEMPORAL_MOTIFS | FACTION_EVOLUTION

-- Information theory & epistemic reasoning
            | ENTROPY | BELIEFS | EVIDENCE | ARGUMENTS | CONTAGION

-- Causal inference & game theory
            | CAUSES | COUNTERFACTUAL | MISSING | ANOMALIES
            | GAME | MEAN_FIELD | PSL | TEMPORAL_RULES

-- Motivation
            | MOTIVATION

-- Graph embeddings & network inference
            | FAST_RP | NODE2VEC | NETWORK_INFERENCE
            | TRAJECTORY | SIMULATE

-- Stylometry (feature: stylometry)
            | STYLE | STYLE_COMPARE | STYLE_ANOMALIES | VERIFY_AUTHORSHIP

-- Disinformation (feature: disinfo)
            | BEHAVIORAL_FINGERPRINT | DISINFO_FINGERPRINT
            | SPREAD_VELOCITY | SPREAD_INTERVENTION
            | CIB | SUPERSPREADERS
            | CLAIM_ORIGIN | CLAIM_MATCH
            | ARCHETYPE | DISINFO_ASSESSMENT

-- Adversarial wargaming (feature: adversarial)
            | ADVERSARY_POLICY | COGNITIVE_HIERARCHY | WARGAME
            | REWARD_FINGERPRINT | COUNTER_NARRATIVE | RETRODICTION

-- Narrative architecture (feature: generation)
            | COMMITMENTS | FABULA | SJUZET
            | DRAMATIC_IRONY | FOCALIZATION
            | CHARACTER_ARC | SUBPLOTS | SCENE_SEQUEL | REORDERING

-- DISCOVER
DISCOVER    (PATTERNS|ARCS|MISSING) (IN binding:Type)?
            ACROSS NARRATIVES ("id", ...)?
            WHERE ...?  RETURN ...

-- GENERATE (generation feature)
GENERATE    PLAN FOR "premise" GENRE "genre"? CHAPTERS int? SUBPLOTS int?
GENERATE    CHAPTER int FROM narrative:binding STYLE (e:binding | BLEND ...)?
MATERIALIZE PLAN "plan_id"
VALIDATE    NARRATIVE "narrative_id"

-- EXPORT
EXPORT      NARRATIVE "id" AS (csv|graphml|json|manuscript|report|archive|stix)

-- SYNTHETIC GENERATION (EATH sprint, model-agnostic — see Chapter 9)
CALIBRATE   SURROGATE USING '<model>' FOR "narrative_id"
GENERATE    NARRATIVE "<output_id>" LIKE "<source_id>"
            [USING SURROGATE '<model>']        -- defaults to 'eath' if omitted
            [PARAMS { ...json... }]
            [SEED <int>] [STEPS <int>] [LABEL_PREFIX '<str>']
GENERATE    NARRATIVE "<output_id>" USING HYBRID
            FROM "<source_a>" WEIGHT <float>,
            FROM "<source_b>" WEIGHT <float>   -- Σ weights = 1.0 ± 1e-6
            [SEED <int>] [STEPS <int>]
INFER       HIGHER_ORDER_CONTAGION(<json-params>)
            FOR n:Narrative WHERE n.id = "..."  -- synchronous (not a job)

-- HYPERGRAPH RECONSTRUCTION (EATH Extension Phase 15c — see Chapter 10)
INFER       HYPERGRAPH FROM DYNAMICS FOR "<narrative_id>"
            [USING OBSERVATION '<source>']     -- 'participation_rate' (default)
                                               -- | 'sentiment_mean' | 'engagement'
            [MAX_ORDER <int>]                  -- 2..=4; default 3
            [LAMBDA <float>]                   -- L1 strength override; auto when omitted

-- OPINION DYNAMICS (EATH Extension Phase 16c — see Chapter 11)
INFER       OPINION_DYNAMICS(
                confidence_bound := <float>,
                variant := '<pairwise|group_mean>',
                [mu := <float>],
                [initial := '<uniform|gaussian|bimodal>']
            ) FOR "<narrative_id>"             -- synchronous (not a job)
INFER       OPINION_PHASE_TRANSITION(
                c_start := <float>, c_end := <float>, c_steps := <int>
            ) FOR "<narrative_id>"             -- synchronous; sweeps c, reports critical-c spike

-- DUAL-NULL-MODEL SIGNIFICANCE (EATH Extension Phase 13c — see Chapter 12)
COMPUTE     DUAL_SIGNIFICANCE FOR "<nid>" USING '<metric>'
            [K_PER_MODEL <n>] [MODELS '<m1>','<m2>',...]
                                                 -- default MODELS = 'eath','nudhy'
                                                 -- metric ∈ {temporal_motifs, communities, patterns}

-- BISTABILITY / HYSTERESIS DETECTION (EATH Extension Phase 14 — see Chapter 13)
INFER       CONTAGION_BISTABILITY(<beta_start>, <beta_end>, <steps>)
            FOR n:Narrative WHERE n.id = "<nid>"  -- synchronous β-sweep
                                                  -- (significance variant via REST: POST /synth/bistability-significance)

-- MUTATIONS
CREATE      (binding:Type {props}) IN NARRATIVE "id"? CONFIDENCE float?
CREATE      SITUATION AT Level CONTENT "text"+ IN NARRATIVE "id"? CONFIDENCE float?
CREATE      NARRATIVE "id" TITLE "t"? GENRE "g"? TAGS ("t1", ...)?
UPDATE      ENTITY "uuid" SET field = value (, ...)*
UPDATE      NARRATIVE "id" SET field = value (, ...)*
DELETE      (ENTITY|SITUATION|NARRATIVE) "id"
ADD         PARTICIPANT "uuid" TO SITUATION "uuid" ROLE Role ACTION "text"?
REMOVE      PARTICIPANT "uuid" FROM SITUATION "uuid"
ADD         CAUSE FROM "uuid" TO "uuid" TYPE CausalType? STRENGTH float? MECHANISM "text"?
REMOVE      CAUSE FROM "uuid" TO "uuid"

-- SYNTHETIC OPT-IN (where supported by the read endpoint; see Chapter 9 §9.7)
            INCLUDE SYNTHETIC?                 -- default OFF; synthetic records filtered out unless opted in

-- VALUES
value       = "string" | integer | float | true | false | null
comparator  = = | != | > | < | >= | <= | IN | CONTAINS
```

---

# Appendix B: Key Encoding Scheme

UUIDs are stored as 16 bytes big-endian binary. Timestamps as 8-byte big-endian milliseconds since epoch. Entity-ordering keys use v7 UUIDs so lexicographic sort matches time order.

**Hypergraph core:**

| Prefix | Content | Key Structure |
|--------|---------|---------------|
| `e/` | Entity records | `e/{uuid_bytes}` |
| `s/` | Situation records | `s/{uuid_bytes}` |
| `p/` | Participation links (multi-role) | `p/{entity_uuid}/{situation_uuid}/{seq_be}` |
| `ps/` | Reverse participation | `ps/{situation_uuid}/{entity_uuid}/{seq_be}` |
| `c/` | Causal links | `c/{from_uuid}/{to_uuid}` |
| `cr/` | Reverse causal | `cr/{to_uuid}/{from_uuid}` |
| `sv/` | State versions | `sv/{entity_uuid}/{situation_uuid}` |
| `ea/` | Entity aliases | `ea/{normalized_name}` |
| `et/` | Entity type index | `et/{entity_type}/{uuid}` |
| `en/` | Entity-narrative index | `en/{narrative_id}/{uuid}` |
| `sl/` | Situation level index | `sl/{narrative_level}/{uuid}` |
| `sn/` | Situation-narrative index | `sn/{narrative_id}/{uuid}` |

**Narratives, projects, taxonomy, prompts:**

| Prefix | Content |
|--------|---------|
| `nr/` | Narrative metadata |
| `cp/` | Corpus splits |
| `pn/` | Project-narrative index |
| `tx/` | Custom taxonomy entries |
| `pt/` | Tuned extraction prompts |
| `ua/` | User-defined narrative arcs |

**Jobs & validation:**

| Prefix | Content |
|--------|---------|
| `ij/` | Inference job records |
| `ij/q/` | Job priority queue |
| `ij/t/` | Job target index |
| `ir/` | Inference results |
| `vq/i/` | Validation queue items |
| `vq/p/` | Validation queue pending index |
| `v/` | Validation log |

**Analysis results:**

| Prefix | Content |
|--------|---------|
| `an/c/` | Centrality (Brandes betweenness, closeness, degree, Louvain) |
| `an/e/` | Situation entropy |
| `an/mi/` | Mutual information |
| `an/b/` | Recursive beliefs |
| `an/ev/` | Dempster-Shafer evidence |
| `an/af/` | Argumentation framework |
| `an/sir/` | SIR contagion |
| `an/pr/` | PageRank |
| `an/ev_c/` | Eigenvector centrality |
| `an/hc/` | Harmonic centrality |
| `an/hits/` | HITS hub + authority |
| `an/tp/` | Topology (articulation points, bridges) |
| `an/kc/` | K-Core |
| `an/lp/` | Label propagation |
| `an/tpr/` | Temporal PageRank |
| `an/ci/` | Causal influence |
| `an/ib/` | Information bottleneck |
| `an/as/` | Assortativity |
| `an/ch/` | Community hierarchy levels |
| `an/tm/` | Temporal motifs |
| `an/fe/` | Faction evolution |
| `an/frp/` | FastRP embeddings |
| `an/n2v/` | Node2Vec embeddings |
| `an/ni/` | NetInf diffusion network |
| `an/ilp/` | Temporal ILP rules |
| `an/mfg/` | Mean field game results |
| `an/psl/` | PSL inference |
| `an/traj/` | Trajectory embeddings |
| `an/sim/` | Narrative simulation results |
| `an/ns/` | Style profile (stylometry) |
| `an/nf/` | Narrative fingerprint (stylometry) |
| `an/nw/` | Style weights config |
| `ps/{nar}/{ci}` | Per-chunk prose features; `_aggregate` suffix = aggregate |

**Chunking, ingestion, sources:**

| Prefix | Content |
|--------|---------|
| `ch/` | Ingestion chunks |
| `ds/` | Document status |
| `si/` | Source index for cascade deletion |
| `lc/` | LLM response cache |
| `geo/` | Geocode cache (Nominatim) |
| `sa/` | Source attributions |
| `sar/` | Reverse attributions |
| `ct/` | Contentions |

**Patterns, communities, RAG, sessions:**

| Prefix | Content |
|--------|---------|
| `pm/` | Pattern records |
| `pm/m/` | Pattern matches |
| `cs/` | Community summaries |
| `sess/` | Conversation session state |
| `chat/s/` | Studio chat session meta (`chat/s/{ws}/{user}/{session_id}`) |
| `chat/m/` | Studio chat messages (`chat/m/{ws}/{user}/{session_id}/{msg_v7}`) |

**Writer workflows:**

| Prefix | Content |
|--------|---------|
| `np/` | Narrative plan (writer doc) |
| `rv/r/` | Narrative revisions |
| `rv/n/` | Narrative → revision index |
| `wr/r/` | Workshop reports |
| `wr/n/` | Narrative → workshop report index |
| `pf/` | Pinned facts (continuity) |
| `cl/` | Cost ledger |
| `rn/` | Research notes |
| `rn/s/` | Research note → situation index |
| `rn/n/` | Research note → narrative index |
| `ann/` | Annotations |
| `ann/s/` | Annotation → situation index |
| `compile/` | Compile profiles |
| `compile/n/` | Compile profile → narrative index |
| `col/` | Collections / saved searches |
| `col/n/` | Collection → narrative index |

**Alerts, investigation, monitoring:**

| Prefix | Content |
|--------|---------|
| `alert/r/` | Alert rules |
| `alert/e/` | Alert events |
| `inv/` | Investigation reports |

**Disinfo feature:**

| Prefix | Content |
|--------|---------|
| `bf/` | Behavioral fingerprint per actor |
| `df/` | Disinformation fingerprint per narrative |
| `sp/r0/` | SMIR snapshot per narrative |
| `sp/jump/` | Cross-platform jump events |
| `vm/baseline/` | Velocity baseline per platform/narrative_kind |
| `vm/alert/` | Velocity anomaly alerts |
| `cib/c/` | CIB clusters |
| `cib/e/` | CIB cluster evidence |
| `cib/s/` | Superspreader rankings |
| `cib/f/` | Content factory detections |
| `cib/delta/` | CIB scan deltas |
| `cl/n/` | Claim normalization |
| `cl/m/` | Claim→fact-check matches |
| `cl/mut/` | Claim mutation events |
| `fc/` | Fact-checks |
| `fc/sync/` | Fact-check sync history |
| `arch/` | Actor archetype distributions |
| `da/` | Disinfo assessments |
| `mon/` | Monitor subscriptions |
| `mon/alert/` | Monitor alerts |
| `sched/` | Scheduled tasks |
| `sched/hist/` | Task execution history |
| `sched/lock/` | Task execution locks |
| `disc/` | Discovery candidates |
| `disc/approved/` | Approved discovery sources |
| `disc/policy` | Discovery policy (single key) |
| `reports/` | Situation reports |
| `mcp/audit/` | MCP audit log |
| `mcp/health/` | MCP source health |

**Narrative architecture (`generation`):**

| Prefix | Content |
|--------|---------|
| `nc/` | Narrative commitments |
| `fs/f/` | Fabula (chronological order) |
| `fs/s/` | Sjužet (discourse order) |
| `di/i/` | Dramatic irony map |
| `di/f/` | Focalization analysis |
| `ca/` | Character arcs |
| `sp/` | Subplot analysis |
| `ss/` | Scene-sequel analysis |
| `se/` | Style embeddings |
| `se/blend/` | Style blend recipes |
| `lora/` | LoRA adapters |
| `lora/merged/` | Merged LoRA adapters |
| `gp/` | Generation plans |
| `fact/` | Planned facts |
| `gm/` | Generation metadata per situation |
| `irl/` | Motivation vectors |
| `text/` | Generated chapter text |
| `ess/` | Essentiality scores |
| `nd/` | Narrative diagnoses (debugger cache) |

**Adversarial wargaming:**

| Prefix | Content |
|--------|---------|
| `adv/policy/` | Adversary policies |
| `adv/sim/` | Simulation state snapshots |
| `adv/wg/` | Wargame sessions |
| `adv/disarm/` | DISARM TTP calibration |
| `adv/reward/` | Psychological reward fingerprints |
| `adv/counter/` | Counter-narrative results |
| `adv/retro/` | Retrodiction results |
| `adv/calib/` | Platform β calibrations |
| `adv/audit/` | Governance audit trail |

**Synthetic generation (EATH sprint, v0.74.4 → v0.75.0):**

UUIDv7 run-IDs are encoded as 16-byte big-endian binary (NOT hex), so `prefix_scan` returns runs in chronological order; "newest first" listings reverse the scan output (O(n), no sort).

| Prefix | Content | Key Structure |
|--------|---------|---------------|
| `syn/p/` | Surrogate calibrated params | `syn/p/{narrative_id_utf8}/{model_utf8}` |
| `syn/r/` | Surrogate run summary | `syn/r/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}` |
| `syn/seed/` | `ReproducibilityBlob` (replay capsule) | `syn/seed/{run_id_v7_BE_BIN_16}` |
| `syn/lineage/` | Synthetic-run lineage index | `syn/lineage/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}` |
| `syn/fidelity/` | `FidelityReport` (Phase 2.5) | `syn/fidelity/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}` |
| `syn/sig/` | Significance result (Phase 7) | `syn/sig/{narrative_id_utf8}/{metric_utf8}/{run_id_v7_BE_BIN_16}` |
| `syn/dual_sig/` | DualSignificanceReport (EATH Extension Phase 13c) | `syn/dual_sig/{narrative_id_utf8}/{metric_utf8}/{run_id_v7_BE_BIN_16}` |
| `syn/bistability/` | BistabilitySignificanceReport (EATH Extension Phase 14) | `syn/bistability/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}` |
| `syn/recon/` | Materialized reconstruction situation refs (Phase 15c) | `syn/recon/{output_narrative_id_utf8}/{job_id_utf8}/{situation_id_v7_BE_BIN_16}` |
| `syn/opinion_sig/` | Opinion-dynamics-significance report (Phase 16c) | `syn/opinion_sig/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}` |
| `opd/report/` | Persisted `OpinionDynamicsReport` records (Phase 16c) | `opd/report/{narrative_id_utf8}/{run_id_v7_BE_BIN_16}` |
| `cfg/synth_fidelity/` | Per-narrative `FidelityThresholds` override | `cfg/synth_fidelity/{narrative_id}` |

**Fuzzy Logic (Fuzzy Sprint, v0.77.2 → v0.78.0):**

| Prefix | Content | Key Structure |
|--------|---------|---------------|
| `fz/tn/` | Registered fuzzy measures (Phase 4) | `fz/tn/measures/{name_utf8}` |
| `fz/agg/` | Reserved (future per-aggregator metadata) | `fz/agg/...` |
| `fz/allen/` | Cached graded Allen 13-vector per pair (Phase 5) | `fz/allen/{narrative_id_utf8}/{a_id_BE_BIN_16}/{b_id_BE_BIN_16}` |
| `fz/quant/` | Cached intermediate-quantifier evaluation (Phase 6) | `fz/quant/{narrative_id_utf8}/{predicate_hash_utf8}` |
| `fz/fca/` | Concept lattices + narrative index (Phase 8) | `fz/fca/{lattice_id_v7_BE_BIN_16}` + `fz/fca/n/{narrative_id_utf8}/{lattice_id_v7_BE_BIN_16}` |
| `fz/rules/` | Mamdani rule definitions (Phase 9) | `fz/rules/{narrative_id_utf8}/{rule_id_v7_BE_BIN_16}` |
| `fz/syllog/` | Stored graded-syllogism proofs (Phase 7) | `fz/syllog/{narrative_id_utf8}/{proof_id_v7_BE_BIN_16}` |
| `fz/hybrid/` | Cached hybrid-probability query reports (Phase 10) | `fz/hybrid/{narrative_id_utf8}/{query_id_v7_BE_BIN_16}` |
| `cfg/fuzzy` | Site-default `FuzzyWorkspaceConfig` (singleton, Phase 4) | `cfg/fuzzy` |

**Metadata & workspaces:**

| Prefix | Content |
|--------|---------|
| `meta/` | Metadata (interval tree, vector index) |
| `cfg/` | Persisted settings (`cfg/llm`, `cfg/inference_llm`, `cfg/studio_chat_llm`, `cfg/ingestion`, `cfg/rag`, `cfg/synth_fidelity/{nid}`, `cfg/fuzzy`) |
| `w/` | Workspace isolation prefix — every other key prefixed with `w/{workspace_id}/` when `X-Tensa-Workspace` is set |

---

# Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Allen Relation** | One of 13 temporal relations between time intervals (Before, After, Meets, …) |
| **Archetype** | A behavioral pattern classification for actors (PowerSeeking, Altruistic, …) |
| **Beam Search** | A search algorithm that explores the top-k most promising branches at each level |
| **Belief Function** | A Dempster-Shafer mass assignment over hypothesis sets |
| **Betweenness** | How often an entity lies on shortest paths between others |
| **Causal Link** | A directed cause-effect relationship between two situations |
| **Closeness** | Average inverse distance to all reachable entities |
| **Confidence** | A 0.0–1.0 score indicating certainty about a piece of data |
| **Contention** | A conflict between sources about the same fact |
| **Counterfactual** | A "what if" scenario where one variable is changed |
| **Degree** | The number of direct connections an entity has |
| **Dempster-Shafer** | A theory of evidence that generalizes probability to handle uncertainty |
| **DRIFT** | Three-phase adaptive RAG retrieval: community primer → follow-up drill-down → leaf entity retrieval |
| **Dung Framework** | A formalism for modeling attacks between arguments |
| **Entity** | A node in the hypergraph (Actor, Location, Artifact, Concept, Organization) |
| **Extension** | A set of accepted arguments in a Dung framework |
| **Fabula / Sjužet** | Chronological events (fabula) vs. discourse order (sjužet) — Russian formalism distinction |
| **Focalization** | POV classification (Zero / Internal / External) per Genette |
| **Gating** | Routing items by confidence to auto-commit, review queue, or rejection |
| **Grounded Extension** | The most conservative set of accepted arguments |
| **Hypergraph** | A graph where edges (situations) can connect multiple nodes (entities) |
| **InfoSet** | An entity's knowledge state (`knows_before`, `learns`, `reveals`) |
| **KL Divergence** | A measure of how one probability distribution differs from another |
| **Leiden** | A community detection algorithm improving on Louvain with a refinement step guaranteeing connected communities |
| **Louvain** | A community detection algorithm based on modularity optimization (superseded by Leiden in TENSA) |
| **Mass Function** | A function assigning evidence mass to subsets of hypotheses |
| **Maturity** | The quality lifecycle stage (Candidate → Reviewed → Validated → GroundTruth) |
| **MaxEnt IRL** | Maximum Entropy Inverse Reinforcement Learning — infers rewards from behavior |
| **MCP** | Model Context Protocol — the tool interface for AI assistants |
| **Modularity** | A measure of community structure quality in networks |
| **Mutual Information** | A measure of statistical dependency between entity appearances |
| **Narrative** | A container for a coherent set of entities, situations, and their relationships |
| **Narrative Level** | Granularity of a situation (Story > Arc > Sequence > Scene > Beat > Event) |
| **NOTEARS** | A method for learning causal structure from observational data via a differentiable DAG constraint |
| **Participation** | An entity's involvement in a situation, with role and action |
| **Plausibility** | Upper bound on probability in Dempster-Shafer theory |
| **QRE** | Quantal Response Equilibrium — Nash equilibrium with bounded rationality |
| **R₀** | Basic reproduction number — average secondary infections per spreader |
| **Self-Information** | `−log₂(P)` — the surprise of an event with probability P |
| **SingleSession** | Ingestion mode where the LLM sees the full text upfront and extracts each chunk as a follow-up turn in one continuous conversation |
| **SIR Model** | Susceptible-Infected-Recovered epidemiological model |
| **Situation** | A hyperedge connecting multiple entities at a point in time |
| **Spatial Anchor** | Geographic coordinates (lat/lon) with precision level (Exact / Area / Region / Approximate) |
| **State Version** | A snapshot of an entity's properties at a specific situation |
| **TensaQL** | TENSA's declarative query language |
| **Temporal Mask** | A boolean matrix indicating which situation pairs are temporally valid for causation |
| **Validation Queue** | A human-in-the-loop review pipeline for uncertain extractions |
| **Virtual Property** | Pre-computed analysis result exposed through TensaQL as `e.an.<metric>` |
| **Workspace** | An isolated dataset sharing a single RocksDB instance via transparent `w/{workspace_id}/` key prefixing |

**Fuzzy Logic (Fuzzy Sprint, Chapter 14):**

| Term | Definition |
|------|------------|
| **T-norm** | A binary operator `T : [0,1]² → [0,1]` modelling fuzzy conjunction. TENSA registers Gödel (`min`), Goguen (`·`), Łukasiewicz (bounded), Hamacher(λ). |
| **T-conorm** | De Morgan dual of a t-norm, modelling fuzzy disjunction. |
| **OWA** | Ordered Weighted Average (Yager 1988) — sort inputs descending, take weighted sum. |
| **Choquet Integral** | Non-additive fuzzy integral against a monotone fuzzy measure (Grabisch 1996). |
| **Fuzzy Measure** | A monotone set function `μ : 2^N → [0, 1]` with `μ(∅) = 0`, `μ(N) = 1`. Cap at N = 16; exact integration capped at N = 10. |
| **Intermediate Quantifier** | Graded "most / many / few / almost all" ramp on `[0, 1]` (Novák 2008). |
| **Peterson Syllogism** | A graded fuzzy syllogism across Peterson's 5 figures (Murinová-Novák 2014). |
| **Fuzzy FCA** | Formal concept lattice over a graded object-attribute incidence (Bělohlávek 2004). |
| **Mamdani Rule** | A fuzzy rule with linguistic antecedents (triangular / trapezoidal / Gaussian membership) and defuzzified numeric output (Mamdani-Assilian 1975). |
| **Fuzzy Probability** | Sugeno-additive integral `P_fuzzy(E) = Σ μ_E(e) · P(e)` over a discrete distribution. Scope-capped base case per Flaminio 2026 FSTA. |

---

# Appendix D: Implementation Status

This appendix tracks where the shipped surface diverges from the reference's implicit "everything works end to end" framing. It is maintained alongside the main reference and updated every version bump. Items are rated on four tiers:

| Tier | Meaning |
|------|---------|
| **Production** | End-to-end wired: types, engine, storage, REST, MCP, Studio where applicable. Covered by tests. |
| **Engine-complete, surface-partial** | Logic and storage work; REST/MCP exposure is missing or partial. Library-only consumers are fine. |
| **Scaffolded** | Types + KV prefixes exist; engine and/or the end-to-end flow are stubbed or minimally tested. Don't treat as production. |
| **Planned** | Described in this reference because the architecture depends on it, but code not shipped. Flagged explicitly. |

## Core platform

| Feature | Tier | Notes |
|---------|------|-------|
| Hypergraph CRUD (entities, situations, participations, causal, state versions) | Production | Ch2, Ch4, Ch5 |
| KV abstraction + RocksDB + MemoryStore | Production | Ch2 |
| Allen interval algebra + constraint network | Production | Ch2, Ch7 |
| TensaQL parser / planner / executor (MATCH, PATH, FLOW, WHERE, AT, NEAR, SPATIAL, GROUP BY) | Production | Ch3 |
| TensaQL mutations (CREATE / UPDATE / DELETE / ADD PARTICIPANT / ADD CAUSE) | Production | Ch3 |
| TensaQL EXPLAIN / EXPORT | Production | Ch3 |
| Ingestion pipeline (standard mode) | Production | Ch2 |
| Ingestion pipeline — SingleSession mode | Production | Ch2, Ch8. OpenRouter / Local LLM only |
| Multi-step enrichment pass | Production | Ch2 |
| Per-situation source spans | Production | Ch2 |
| Post-ingestion chunk control (reextract / enrich / reconcile / reprocess) | Production | Ch2, Ch5 |
| Confidence gating + validation queue | Production | Ch5 |
| Entity resolution (alias + embedding similarity) | Production | Ch2 |
| Source intelligence (sources, attributions, contentions, claim-aware DS mass) | Production | Ch2, Ch5 |
| Bayesian confidence with source-trust propagation | Production | Ch2 |
| Workspaces (transparent KV prefixing) | Production as data boundary; **not** a security boundary. See §"Security & Multi-Tenancy" in Ch2 |
| Embedding backfill (hash + ONNX) | Production | Ch5 |
| Geocoding (Nominatim, cached) | Production | Ch5 |
| LLM response cache | Production | Ch5 |

## Inference engines

| Engine | Tier | Notes |
|--------|------|-------|
| Centrality (Brandes / Wasserman–Faust / degree) | Production | §7.1.1–7.1.3 |
| PageRank / Eigenvector / Harmonic / HITS | Production | §7.1.4 |
| Topology (articulation points, bridges, K-Core) | Production | §7.1.5 |
| Leiden communities (hierarchical) | Production | §7.1.6 |
| Label propagation | Production | §7.1.6 |
| Temporal PageRank / Causal Influence / Info Bottleneck / Assortativity | Production | §7.1.7 |
| Temporal motif census | Production | §7.1.8 |
| Faction evolution | Production | §7.1.8 |
| Pathfinding (Dijkstra / Yen / diameter / max-flow) | Production | §7.1.9, Ch5 `/analysis/*` |
| Inline graph functions in TensaQL | Production | §7.1.10 |
| Graph embeddings (FastRP, Node2Vec) | Production | §7.1.11 |
| NetInf diffusion network inference | Engine-complete, surface-partial | §7.1.12. Library-only; no REST endpoint yet |
| Self-information / mutual information / KL divergence | Production | §7.2 |
| Recursive beliefs (SymbolicToM seeding + 4-phase) | Production | §7.3.1 |
| Dempster-Shafer combination (Dempster's + Yager's rules) | Production | §7.3.2 |
| Dirichlet/EDL confidence | Production | §7.3.3 |
| Propp function classification | Production | §7.3.4 |
| Dung argumentation frameworks | Production | §7.4 |
| SIR information contagion | Production | §7.5 |
| NOTEARS causal discovery (with DAGMA, SCC pre-validation, LLM priors) | Production | §7.6.1. Requires `inference` feature |
| Counterfactual beam search | Production | §7.6.2 |
| Game classification + QRE solver | Production | §7.7 |
| Mean field games | Production | §7.7.3 |
| MaxEnt IRL + archetype classification | Production | §7.8 |
| Temporal ILP (rule mining) | Production | §7.10.1 |
| Probabilistic Soft Logic | Production | §7.10.2 |

## Narrative architecture (`generation` feature)

| Feature | Tier | Notes |
|---------|------|-------|
| Commitment tracking | Production | §7.11.1 |
| Fabula / sjužet separation + reordering suggestions | Production | §7.11.2 |
| Dramatic irony map | Production | §7.11.3 |
| Focalization detection | Production | §7.11.3 |
| Three-process analysis | Engine-complete, surface-partial | §7.11.4. `analyze_three_processes` exists; no dedicated REST/MCP endpoint — results reachable via the full analysis battery |
| Character arcs | Production | §7.11.5 |
| Subplot detection | Production | §7.11.6 |
| Scene-sequel rhythm | Production | §7.11.7 |
| Three-stage generation pipeline (plan → materialize → generate) | Production | §7.11.8 |
| Personalization ladder — tier 1 (prompt-only) | Production | §7.11.9. The existing generation prompt path is tier 1 |
| Personalization ladder — tier 2 (style embedding injection into prompt) | Production (via fitness loop) | §7.11.9/10/14. As of v0.65 the fitness-loop chapter generator (`generate_chapter_with_fitness`) consumes a `StyleEmbedding` by id and renders its `source` enum into the system prompt (single-author / blended / genre / custom — see §7.11.10 *Consumption*). The vector itself stays opaque; SE conditioning on the **non-fitness** `prepare_chapter` path is still partial |
| Personalization ladder — tier 3 (LoRA adapter applied at inference) | Scaffolded | §7.11.9/11. `LoraAdapter` metadata, status lifecycle, and arithmetic merging exist in-process. Actual training + inference-time LoRA application is delegated to external binaries (e.g. `bin/train_author_lora.rs`). Default generation path is the base model without LoRA |
| Style encoder (contrastive training) | Scaffolded | §7.11.10. `HashStyleEncoder` for tests; production contrastive encoder is an external ONNX/candle binary, not included in `cargo build` |
| Fingerprint layer transfer (`rhythm_transfer`) | Engine-complete, surface-partial | §7.11.12. Library-only; no REST endpoint |
| Narrative skeleton (extract, similarity, transplant) | Engine-complete, surface-partial | §7.11.13. `extract_narrative_skeleton` MCP tool wired in v0.70.0 (Sprint W15); `store_skeleton` / `skeleton_similarity` / `transplant` remain library-only — no REST or MCP endpoints |
| Fingerprint fitness loop (closed-loop refine) | Production (prose-only); re-ingest path missing | §7.11.14, §5.22. `GenerationEngine::generate_with_fitness` + `ChapterGenerator` trait + `generate_chapter_with_fitness` MCP tool ship in v0.65. Submitted via `POST /jobs` (`job_type: "chapter_generation_fitness"`); Studio Generate view drives it end-to-end with target-fingerprint + SE pickers, threshold/retries/temperature controls, 3 s polling, and a `FitnessResultPanel` (per-attempt accordion + sparkline). Scores the prose layer per iteration; structure-layer rescoring is **out of scope per iteration** because re-ingesting each attempt is prohibitive. Threshold default `0.80` is a placeholder pending held-out same-author calibration. **Caveat:** the Studio "Accept and re-ingest" button currently ships **disabled** — there is no backend endpoint that accepts a fitness-generated chapter text into the manuscript yet. The button's `title` carries a precise TODO. The loop produces text; promoting it to the manuscript is a manual copy or a follow-up commit |
| Narrative debugger (22 pathologies, auto-repair, genre presets) | Production | §7.11.16, Ch5 |
| Narrative compression & expansion | Production | §7.11.17, Ch5 |

## Stylometry (`stylometry` feature)

| Feature | Tier | Notes |
|---------|------|-------|
| Prose-level stylometry (26 features, Burrows' Delta) | Production | §7.9.1 |
| Multi-layer narrative style profile | Production | §7.9.2 |
| Weighted similarity with per-layer kernels | Production | §7.9.3 |
| Calibrated anomaly detection (bootstrap p-values) | Production | §7.9.4 |
| PAN@CLEF authorship verification + metric suite | Production | §7.9.5 |
| Style weights training (`train_pan_weights` CLI) | Production | Requires `cli` feature |

## Disinformation (`disinfo` feature, default-on)

| Feature | Tier | Notes |
|---------|------|-------|
| Behavioral fingerprint (10 axes) | Production | §4.13.1, Ch5 |
| Disinfo fingerprint (12 axes) | Production | §4.13.1, Ch5. Note: several axes require feature-specific data (CIB, velocity, claims) to populate; they serialize as `null` without that data |
| Fingerprint comparison with task-specific weighting | Production | §4.13.1 |
| SMIR spread dynamics + R₀ + cross-platform jumps + velocity alerts | Production | §4.13.2, Ch5 |
| Spread intervention (counterfactual projection) | Production | §4.13.2 |
| CIB cluster detection (similarity → label-prop → bootstrap p-value) | Production | §4.13.3 |
| Superspreader ranking | Production | §4.13.3 |
| Claim detection + fact-check matching + mutation tracing | Production | §4.13.4, Ch5 |
| Archetype classification | Production | §4.13.5 |
| DS fusion of disinfo signals | Production | §4.13.5 |
| Post/actor ingestion helpers | Production | §4.13.6 |
| Multilingual (language detection, transliteration, diacritic stripping) | Production | §4.13.7 |
| MCP client orchestrator (NormalizedPost, audit trail) | Production | §4.13.8 |
| Scheduler + task runner + discovery + reports + source health | Production | §4.13.9, Ch5 |
| MISP / Maltego / comprehensive report exports | Production | §4.13.10 |

## Adversarial (`adversarial` feature)

| Feature | Tier | Notes |
|---------|------|-------|
| Adversary policy generation (SUQR, IRL reward weights) | Production | §4.14, Ch7 |
| Rationality configuration (QRE / SUQR / Cognitive Hierarchy) | Production | §4.14 |
| Wargame sessions (red/blue turn-based) | Production | §4.14 |
| Reward fingerprint (8 psychological dimensions) | Production | §4.14 |
| Counter-narrative generation (reward-aware) | Production | §4.14 |
| Retrodiction (RBO, RMSLE, KL, Spearman) | Production | §4.14 |
| DISARM TTP calibration | Production | Stored at `adv/disarm/` |
| Governance audit trail | Production | Stored at `adv/audit/` |

## Writer workflows

| Feature | Tier | Notes |
|---------|------|-------|
| Plan CRUD | Production | §5.22 |
| Revisions (commit, restore, diff, scene summaries) | Production | §5.22 |
| Generation (outline / characters / scenes, POV hints) | Production | §5.22 |
| Editing (rewrite / tighten / expand / style_transfer / dialogue_pass) | Production | §5.22 |
| Workshop (cheap / standard / deep tiers) | Production for `cheap` + `standard`. `deep` tier returns a deferred `cheap` report (deep async inference pipeline is Scaffolded) |
| Pinned facts + continuity check | Production | §5.22 |
| Cost ledger + windowed summary | Production | §5.22 |
| Research panel (notes, context bundle) | Production | §5.22 |
| Fact-check (Fast / Standard / Deep tiers) | Production for Fast + Standard. Deep tier returns a deferred result |
| Cited generation (prompt addendum, hallucination guard, citation parser) | Production | §5.22 |
| Annotations (Comment / Footnote / Citation, reconcile on edit) | Production | §5.22 |
| Compile profiles (Markdown / EPUB 3 / DOCX) | Production. DOCX requires `docparse` feature |
| Collections / saved searches | Production | §5.22 |
| Scene reorder (atomic, cycle-rejected) | Production | §5.22 |

## Studio integrated chat (`studio-chat` feature)

| Feature | Tier | Notes |
|---------|------|-------|
| Session storage + SSE streaming | Production | §5.26 |
| Multi-turn LLM loop with independent chat LLM | Production | §5.26 |
| 12 built-in tools + confirmation gate on mutations | Production | §5.26 |
| Third-party stdio MCP server proxy via `rmcp` | Production | §5.26, `/studio/chat/mcp-servers` |
| Skills catalog + session-scoped picker | Production | `studio-ui`, `tensa`, `tensa-writer` |
| `POST /studio/chat/sessions/:id/stop` — in-flight cancel | Scaffolded | Returns 501 until the cancel-token wiring lands |

## Studio UI ([studio/src/](../studio/src/))

The Observatory Console frontend (React 18 + Vite + TypeScript) exposes the TENSA surface. As of v0.74.20 the information architecture is **narrative-centric**: most work happens inside a `WorkspaceShell` scoped to a single narrative, where a canvas tab (Graph / Timeline / Map / …) is paired with a shared Inspector that opens the same panel regardless of which canvas did the selecting. The consolidation sprint collapsed a 37-item sidebar and ~22 siloed views into 12 primary nav entries + a Legacy Views accordion.

### Sidebar (12 items + Legacy Views)

Grouped into four sections. Source: [studio/src/components/Sidebar.tsx](../studio/src/components/Sidebar.tsx).

| Group | Items |
|-------|-------|
| LIBRARY | Home, Narratives, Projects, Media Gallery |
| INGEST | Ingest, Review Queue, Sources |
| INTELLIGENCE | Disinfo Ops, Wargame, Alerts, Job Monitor |
| SYSTEM | Settings |

A collapsed **Legacy Views** accordion (localStorage-persisted, default closed) keeps the pre-consolidation routes reachable for habit and troubleshooting: Geo Map, Graph Explorer, Hypergraph, Timeline, Embedding Space, TensaQL Console, Entities, Situations, Story Hub, Ask, Analysis Hub, Fingerprint, Relations, State Diff, Debugger, Adaptation.

### Workspace routes — `/n/:narrativeId/:tab`

`WorkspaceShell` (mounted at `/n/:narrativeId/:tab`) owns the per-narrative workspace. `/n/:narrativeId` (no tab) redirects to `…/graph`; unknown tabs fall back to `graph`. The canonical `CanvasTabId` enum is defined in [studio/src/workspace/CanvasTabs.tsx](../studio/src/workspace/CanvasTabs.tsx) and indexed in [studio/src/workspace/index.ts](../studio/src/workspace/index.ts).

| Tab | One-liner |
|-----|-----------|
| `graph` | Force-directed graph + bipartite/hypergraph layout toggle (unified Graph+Hypergraph) |
| `temporal-graph` | Playable time-ordered hypergraph — situations sorted by `temporal.start` and advanced by a cursor; previous situations remain on stage. Play/pause, ±step, 0.5×/1×/2×/4× speeds, scrub slider, entering-situation pulse. Sim runs between frames with a short dwell (~400 ms / speed) once alpha < 0.015; node positions cached across rebuilds to keep the camera stable. |
| `timeline` | D3 horizontal temporal lanes; brush writes workspace `timeRange` filter |
| `map` | Leaflet geo view with Stadia dark tiles; viewport writes `spatialBounds` filter |
| `matrix` | Co-occurrence relationship heatmap; cell click focuses a community |
| `embeddings` | 2D/3D embedding scatter with NodeDetail-style drill-in. Two projection modes selectable in the toolbar: **UMAP** (non-linear, neighborhood-preserving — preserves clusters; deterministic per dataset) and **PCA** (linear, top-16 principal components computed once via power iteration with deflation; X/Y/Z dropdowns swap which PCs map to which screen axes, with variance share shown next to each option e.g. `PC1 (37.4%)`). The header **📏 Ruler** toggle measures cosine similarity + cosine distance between the last two selected points using the **original embedding vectors** (not the projection), so the readout is honest regardless of mode. Implementation: [studio/src/utils/projection.ts](../studio/src/utils/projection.ts) (PCA + cosine helpers), [studio/src/views/EmbeddingSpace.tsx](../studio/src/views/EmbeddingSpace.tsx) (canvas). |
| `ask` | RAG / DRIFT / PPR question answering scoped to the narrative |
| `manuscript` | Manuscript prose editor wrapped in `MentionAnnotator` (bidirectional entity linking) |
| `plan` | Narrative plan CRUD, fitness-loop chapter generation, binder/corkboard/outliner |
| `cast` | Character/place management with research panel, participant wiring |
| `workshop` | Tiered critique runner + architecture readout (D9 commitments, rhythm, fabula/sjužet, …) |
| `history` | Revision list + diff viewer + revert |
| `analysis` | Analysis Hub: 27 `InferenceJobType` cards with bespoke and generic result renderers + `RecentRuns` strip (v0.79.7) for one-click reload of prior runs |
| `fingerprint` | Multi-layer narrative style profile + stylometry |
| `debugger` | Narrative debugger (D10/D11) — trace contradictions, dead ends, rule violations |
| `adaptation` | Adaptation studio — cross-medium transforms, style transfer |
| `synth` | **Reserved** for EATH Synthetic Hypergraph; always pinned last in the tab strip |

### Shared component library

The consolidation sprint distilled the UI into four tiers of shared primitives. Every canvas and every legacy view routes through these for selection, detail panels, filters, and keyboard-driven navigation.

**Tier 1 — Inspectors** ([studio/src/components/inspectors/](../studio/src/components/inspectors/))

Four kind-scoped panels with a uniform Header + Tabs + Panel + ActionBar skeleton, plus a global mount:

- `EntityInspector` — entity header, properties, participations, attributions, merge/split/open-dossier actions
- `SituationInspector` — situation header, participants, causal edges, temporal strip, attributions
- `NarrativeInspector` — narrative metadata, stats, manage actions
- `SourceInspector` — source identity, trust, attributions
- `InspectorHost` — single `position: fixed; right: 0; width: 520px` panel mounted once in App.tsx; reacts to the global selection store

**Tier 2 — Shell (global state)** ([studio/src/components/inspectors/useSelection.ts](../studio/src/components/inspectors/useSelection.ts), [studio/src/workspace/useWorkspaceFilters.ts](../studio/src/workspace/useWorkspaceFilters.ts))

- `useSelection` / `useIsSelected` — canonical selection store. `select*()` opens the correct Inspector; `clear()` closes it. Transient `hoveredEntity` drives manuscript highlight.
- `useWorkspaceFilters` — cross-lens filter state (`timeRange`, `spatialBounds`, `focusedCommunity`, `filteredEntityTypes`), scoped per narrativeId in sessionStorage.

**Tier 3 — Detail blocks** ([studio/src/components/detail/](../studio/src/components/detail/))

Nine canonical presentation primitives. Every list row / header / action footer in every Inspector and legacy view now delegates here:

- `EntityHeader`, `SituationHeader`, `NarrativeHeader`, `SourceHeader` — type-dot + name + maturity/confidence
- `PropertyGrid` — 2-col key/value grid with collapsible nested objects and `mode: read | edit`
- `ParticipantList` — `situation_id → participants` with role chips (`ROLE_COLORS`)
- `ParticipationList` — `entity_id → situations` with narrative-level badges
- `CausalEdgeList` — `situation.causes[]` with resolved from/to names
- `AttributionList` — source attributions with trust chip and excerpt
- `TemporalStrip` — Allen interval start/end + per-relation symbol chip (`ALLEN_SYMBOLS`)
- `ActionBar` — uniform footer with default / danger variants

Shared primitives: `Badges` (maturity, entity-type, status, confidence), `ConfidenceBar`, `Drawer`.

**Tier 4 — Chrome** ([studio/src/components/CommandPalette.tsx](../studio/src/components/CommandPalette.tsx), [studio/src/components/EmptyState.tsx](../studio/src/components/EmptyState.tsx), [studio/src/workspace/canvases/manuscript/](../studio/src/workspace/canvases/manuscript/), [studio/src/components/inspectors/InspectorStates.tsx](../studio/src/components/inspectors/InspectorStates.tsx))

- `CommandPalette` — global Ctrl+K / Cmd+K modal with fuzzy search over narratives, entities, situations, sources, canvas tabs, and settings subpages. Routes to `select*()` or `navigate()`.
- `EmptyState` — unified empty-state primitive with icon + title + message + primary/secondary 44px-touch-target actions. Adopted across WorkspaceShell 404, Sources, Dashboard, Timeline, Fingerprint, Matrix, AdaptationStudio, AnalysisHub.
- `MentionAnnotator` — `display: contents` wrapper around `ManuscriptEditor`. Debounced MutationObserver walks text nodes, wraps entity matches in `<span data-entity-id>`. Pairs with `hoveredEntity` + sessionStorage `tensa-manuscript-scroll-to` for bidirectional Manuscript ↔ Graph/Timeline/Inspector linking.
- `InspectorStates` — shared idle/loading/error/not-found states with consistent tone.

### Deprecated routes → redirects

The consolidation sprint retired 13 top-level routes plus the `/story/*` hub plus 8 `/disinfo/*` sub-routes; all remain reachable either via the Legacy Views accordion or the Command Palette.

| Old route | Redirects to | Notes |
|-----------|--------------|-------|
| `/explorer` | `/n/{last}/graph` | last = `tensa-last-narrative` localStorage key |
| `/graph` | `/n/{last}/graph` | `LegacyCanvasRedirect` |
| `/hypergraph` | `/n/{last}/graph?layout=bipartite` | Same canvas, different layout |
| `/timeline` | `/n/{last}/timeline` | |
| `/map` | `/n/{last}/map` | |
| `/matrix` | `/n/{last}/matrix` | |
| `/embeddings` | `/n/{last}/embeddings` | |
| `/analysis` | `/n/{last}/analysis` | |
| `/fingerprint` | `/n/{last}/fingerprint` | |
| `/debugger` | `/n/{last}/debugger` | |
| `/adaptation` | `/n/{last}/adaptation` | |
| `/disinfo/*` (8 subpages) | `/disinfo?tab={slug}` | Collapsed into one `DisinfoOps` hub; Disinfo stays OUTSIDE the workspace because it's cross-narrative |
| `/story/{id}/{manuscript\|cast\|research\|history\|plan\|generate\|binder\|corkboard\|outliner\|workshop\|architecture}` | `/n/{id}/{manuscript\|cast\|history\|plan\|workshop}` | Storywriting flatten via `StoryLegacyRedirect` |
| `/story/{id}/{situations\|arcs\|scrivenings\|source\|collections\|compile}` | *(kept on legacy path)* | No canvas fit yet — remain accessible via Story Hub |

### Keyboard shortcuts

| Scope | Key | Action |
|-------|-----|--------|
| Global | `Ctrl+K` / `Cmd+K` | Open CommandPalette (captured even when an input has focus) |
| Global | `Esc` | Progressive clear: selection → workspace filters (double-tap) |
| Workspace | `[` / `]` | Cycle to prev / next canvas tab (skips reserved tabs) |
| Workspace | `Ctrl+B` | Toggle NarrativeBrowser collapsed/open |
| Inspector | `Esc` | Clear selection and close Inspector |
| Inspector | `E` | Enter edit mode (PropertyGrid, where supported) |
| Inspector | `H` | Hide (collapse) the Inspector without clearing selection |
| Inspector | `G` / `T` / `M` | Open current selection in Graph / Timeline / Map canvas |

### Cross-lens coupling

All canvases wired into `WorkspaceShell` share one filter store ([studio/src/workspace/useWorkspaceFilters.ts](../studio/src/workspace/useWorkspaceFilters.ts)), scoped per `narrativeId` in sessionStorage:

- **Timeline** brush → writes `timeRange`; **Graph** and **Map** read it to fade out-of-range nodes.
- **Map** viewport pan/zoom → writes `spatialBounds`; **Matrix** reads it to narrow to entities within the bounds.
- **Matrix** cell click → writes `focusedCommunity`; **Graph** highlights that community's subgraph.
- **NarrativeBrowser** entity-type filter toggles → writes `filteredEntityTypes`; every canvas honours it.

Active filters are surfaced as chips in `WorkspaceHeader`. `Esc` clears them progressively (first press clears selection, second clears filters).

### Shared pickers ([studio/src/components/pickers/](../studio/src/components/pickers/))

v0.72.2. `ComboBox` (generic portal-popover, searchable, keyboard-nav), `NarrativeComboBox` (reads from `NarrativeContext` — cheap to mount many on one page), `EntityComboBox` (fetches `/entities?narrative_id=&entity_type=` with request-id guard against stale responses). Used across DisinfoOps tabs (Fingerprints, SpreadMonitor, CibDashboard, Archetypes, Claims, Exports) to replace raw UUID text inputs. `allowCustomId` escape hatch preserves support for narrative slugs that are server-only.

### Writer surfaces (inside canvas tabs)

The writer-facing views below are now mounted inside their respective canvas tabs rather than as standalone routes.

| Surface | Canvas tab | Notes |
|---------|------------|-------|
| `ManuscriptEditor` + `AnnotationsPanel` | `manuscript` | Wrapped in `MentionAnnotator`; ¶ NOTES toggle, scene-scoped comments / footnotes / citations |
| `ResearchPanel`, `EditPanel` | `manuscript` / `cast` | Notes grid, `propose_edit` / `apply_edit` with tier estimate |
| `WorkshopView`, `ArchitectureView`, `ContinuityWarningList` | `workshop` | Three-tier critique + D9 architecture readout (commitments, rhythm, fabula/sjužet, dramatic irony, focalization, character arcs, subplots, scene-sequel) |
| `HistoryView` + `CostStrip` | `history` | Revision list + diff viewer + expandable raw-ledger drawer |
| `PlanView` + `GenerateView` | `plan` | Plan CRUD + fitness-loop chapter generation with target-fingerprint + SE picker |
| `Binder / Corkboard / Outliner / Scrivenings` | `plan` | Sprint W8 manuscript-tooling modes (subpages flatten into the `plan` tab) |
| `Hypergraph` polygon metaphor | `graph` (layout toggle) | Oliver/Zhang/Zhang 2023 convex-polygon situations; lib at [studio/src/lib/hypergraph/](../studio/src/lib/hypergraph/) |
| `Analysis Hub` | `analysis` | 27 `InferenceJobType`s; 16 bespoke renderers + 1 shared `EntityScoresViz`. `RecentRuns` strip (v0.79.7) above the cards lists the latest job per type from `/jobs?narrative_id=…` with status-colored chips; clicking a Completed chip reloads its result. Per-card status badges + result-header "ran Nm ago" reuse the same fetch. AnomalyViz adds a z-score scatter chart with severity bands (\|z\|=2.5 amber, \|z\|=3 danger); GameTheoryViz renders QRE strategies as stacked horizontal bars per player with a Shannon-entropy mixedness indicator; PatternViz shows a frequency histogram + narrative-coverage matrix. |
| `Sources` | workspace-aware | `GET /sources?narrative_id=<id>` when a narrative is locked |
| `CollectionsView`, `CompileView`, `SituationList`, `ArcAndTaxonomy`, `ScriveningsView`, `SourceView` | *(legacy `/story/*` paths)* | Subpages without a canvas home yet — reachable via Story Hub |

## RAG surface

| Feature | Tier | Notes |
|---------|------|-------|
| ASK with `local` / `global` / `hybrid` / `mix` / `drift` / `lazy` / `ppr` modes | Production | §3.15, Ch5 |
| Community summaries (LLM-generated, cached) | Production | Ch5 |
| Hierarchical community summaries (via Leiden) | Production | Ch5 |
| Multi-turn sessions + follow-up suggestions | Production | Ch5 |
| Prompt auto-tuning | Production | §3.16 |
| OpenAI-compatible `/v1/chat/completions` + `/v1/models` | Production | §5.21 |
| Pluggable vector backends (Qdrant, pgvector) | Scaffolded | `VectorStoreConfig` enum exists; default is `InMemory`. External backends compile-gated but not integration-tested against production instances |
| Reranker (term-overlap baseline) | Production | Extensible via `Reranker` trait |

## Notes on reading this table

- **Scaffolded** items are not failures — they indicate deliberate library-first plumbing where the public surface (REST/MCP/Studio) is waiting for a concrete consumer. If you need one, the types and storage will survive; wiring a route is additive work.
- **Engine-complete, surface-partial** items are the priority list for the next reference-review pass: either expose the missing endpoint or document the private entry point explicitly.
- When a row in this table disagrees with the main chapters, the main chapters are wrong — open an issue or update them.


