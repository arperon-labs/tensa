---
name: tensa
description: TensaQL grammar and idioms for querying the temporal hypergraph — activate when the user wants to retrieve, filter, or analyse narrative data.
---

# TensaQL Assistant

Turn on this skill when the user is writing or interpreting TensaQL.
Prefer calling the `query_tensaql` tool over explaining syntax — it
executes the query and returns live rows. Restrict to MATCH-shape
queries; INFER / DISCOVER / INGEST / EXPORT / ASK / TUNE are rejected
by the tool for safety.

## Core shapes

```tensaql
-- Node patterns with type + label
MATCH (e:Actor)
MATCH (s:Situation)

-- Filters
WHERE e.confidence > 0.8
WHERE e.properties.name = "Alice"
WHERE e.narrative_id = "crime-and-punishment"

-- Edge patterns
MATCH (a:Actor)-[:PARTICIPATES]->(s:Situation)
MATCH (s1:Situation)-[:CAUSES]->(s2:Situation)

-- Multi-hop
MATCH (a:Actor)-[:PARTICIPATES]->(s:Situation)-[:CAUSES*1..3]->(s2:Situation)

-- Temporal (Allen relations)
MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01"
MATCH (s:Situation) AT s.temporal DURING "2024-06..2024-09"
MATCH (s:Situation) AT s.temporal MEETS another

-- Vector similarity
MATCH (e:Actor) NEAR(e, "villain antagonist", 5)

-- Spatial
MATCH (s:Situation) SPATIAL s.spatial WITHIN 10.0 KM OF (40.71, -74.01)

-- Aggregation
MATCH (e:Actor) WHERE e.confidence > 0.5
RETURN e.entity_type, COUNT(*), AVG(e.confidence)
GROUP BY e.entity_type
ORDER BY COUNT(*) DESC LIMIT 10

-- Boolean WHERE with parens
MATCH (e:Actor) WHERE (e.confidence > 0.8 OR e.maturity = "Validated") AND e.narrative_id = "h"
```

## Typical intents → queries

| Intent | TensaQL |
|---|---|
| "Who are the main actors in narrative N?" | `MATCH (e:Actor) WHERE e.narrative_id = "N" RETURN e ORDER BY e.confidence DESC LIMIT 20` |
| "What happened around date D?" | `MATCH (s:Situation) AT s.temporal OVERLAPS "D..D+1day" RETURN s` |
| "Who appears in this scene?" | `MATCH (e)-[:PARTICIPATES]->(s) WHERE s.id = "<uuid>" RETURN e` |
| "Longest causal chain in narrative N" | `MATCH PATH LONGEST ACYCLIC (s1)-[:CAUSES*]->(s2) WHERE s1.narrative_id = "N" RETURN PATH` |
| "Entities similar to X" | `MATCH (e:Actor) NEAR(e, "<free text describing X>", 10) RETURN e` |

## Output conventions

- Return **entities or situations**, not raw ids, so the user sees names + types.
- Default `LIMIT 10` unless the user asks for more.
- When the user asks qualitatively ("who is Alice?"), prefer `query_tensaql`
  to fetch the actual record first, then answer from the data.

## When NOT to use TensaQL

- "What's a narrative?" → conceptual answer (no query).
- "Create a new character." → use the mutating `create_entity` tool (Phase 3 gate).
- "Explain this scene." → use `get_situation` + prose summary.
- "Summarise the whole story." → use the `ask` (RAG) tool — richer than a MATCH scan.

## Synthetic Generation

TENSA can generate synthetic narratives from a `SurrogateModel` (currently
EATH — Effective Active Temporal Hypergraph from Mancastroppa, Cencetti,
Barrat, arXiv:2507.01124v2). Use synth for:

- **Null models for significance testing** — pattern mining, community
  detection, temporal motifs, higher-order contagion. A z-score ≥ 1.96
  (|p| ≤ 0.05) on the source-vs-K-surrogates distribution means the
  observed structure is genuinely above background.
- **Synthetic benchmarks at scale** — PageRank/Leiden/SIR stress-tests
  without real corpora. EATH calibrated on a small narrative scales to
  thousands of entities with controlled burstiness + group-size
  distributions.
- **Adversarial wargame substrate** — D12 sessions on calibrated civilian
  backgrounds. `WargameConfig.background = Synthetic | SyntheticHybrid`
  spins up the substrate at session creation.

### TensaQL grammar

```sql
-- Calibrate (model required, no default — calibration is O(dataset))
CALIBRATE SURROGATE USING 'eath' FOR "narrative-id"

-- Generate (model defaults to 'eath' if USING SURROGATE omitted)
GENERATE NARRATIVE "output-id" LIKE "source-id" USING SURROGATE 'eath'
    [PARAMS { ...json... }] [SEED <n>] [STEPS <n>] [LABEL_PREFIX '<str>']

-- Hybrid generation (mixture-distribution; weights must sum to 1.0 ± 1e-6)
GENERATE NARRATIVE "output-id" USING HYBRID
    FROM "source-a" WEIGHT 0.7,
    FROM "source-b" WEIGHT 0.3
    [SEED <n>] [STEPS <n>]

-- Higher-order contagion on real narrative (synchronous, NOT a job)
INFER HIGHER_ORDER_CONTAGION(<json-params>) FOR n:Narrative WHERE n.id = "..."
```

### Common idioms

**Calibrate then generate:**
```sql
CALIBRATE SURROGATE USING 'eath' FOR "harbor-case"
-- wait for job; then:
GENERATE NARRATIVE "harbor-case-synth-1" LIKE "harbor-case"
    USING SURROGATE 'eath' SEED 42 STEPS 1000
```

**Reproducible run with seed override:**
```sql
GENERATE NARRATIVE "harbor-case-replay-7" LIKE "harbor-case"
    USING SURROGATE 'eath'
    PARAMS {"rho_low": 0.05, "rho_high": 0.5, "xi": 3.0}
    SEED 12345 STEPS 500 LABEL_PREFIX 'replay-7'
```

**Significance test for pattern mining (REST):**
```
POST /synth/significance
  { "narrative_id": "harbor-case", "metric": "patterns", "k": 100 }
-- Returns job_id; poll, then read individual rows: |z| ≥ 1.96 ⇒ significant.
```

**Wargame substrate from a hybrid (mixture):**
```sql
GENERATE NARRATIVE "wargame-bg-A1" USING HYBRID
    FROM "civilian-corpus-baseline" WEIGHT 0.85,
    FROM "adversarial-corpus-2024" WEIGHT 0.15
    SEED 7 STEPS 2000
-- then create wargame session with background = ExistingNarrative("wargame-bg-A1")
```

### When NOT to use synth output

- **Not a stand-in for real ingestion when you need rich semantics.**
  Synth participations default to `Role::Bystander`; situations have
  empty `info_set` and no `game_structure`. The narrative shape
  (group-size distribution, burstiness, temporal correlations) is
  reproduced; the meaning is not.
- **Not for writer workflows.** Workshop, pinned facts, plans, prose
  generation will technically run on synthetic narratives but produce
  nonsense — the underlying scenes are blank.
- **Not for RAG question-answering.** The `/ask` endpoint defaults to
  excluding synthetic records; if you opt in via `?include_synthetic=true`
  the LLM will narrate synthetic scenes as if real, producing fiction.

Read the `tensa-synth` skill bundle for parameter intuition (`rho_low`,
`rho_high`, `xi`, `p_from_scratch`) and the full /synth/* REST surface.

## Hypergraph Reconstruction (EATH Extension Phase 15c)

INVERSE problem to synth: given an observed narrative whose entity
time-series you can sample, recover the latent hyperedges that best
explain the joint dynamics. SINDy-based; cite Delabays, De Pasquale,
Dörfler, Zhang — Nat. Commun. 16:2691 (2025), arXiv:2402.00078.

```sql
-- Defaults: observation = participation_rate, max_order = 3, λ auto.
INFER HYPERGRAPH FROM DYNAMICS FOR "telegram-corpus-1"

-- Tunable form. observation source must be one of: participation_rate,
-- sentiment_mean, engagement (belief_mass requires explicit JSON params
-- so use POST /inference/hypergraph-reconstruction directly).
INFER HYPERGRAPH FROM DYNAMICS FOR "telegram-corpus-1"
    USING OBSERVATION 'participation_rate'
    MAX_ORDER 3
    LAMBDA 0.05
```

### Common idioms

**"How do I find hidden coordination in my disinfo corpus?"**
```sql
INFER HYPERGRAPH FROM DYNAMICS FOR "disinfo-tg-q1-2026"
-- → returns job_id; poll GET /inference/hypergraph-reconstruction/{job_id}
-- Inferred edges with confidence > 0.7 are candidate coordination groups.
-- Cross-reference against declared groups; novel edges go to analyst review.
```

**"Materialize the high-confidence inferred edges as Situations":**
```
POST /inference/hypergraph-reconstruction/{job_id}/materialize
  { "output_narrative_id": "disinfo-tg-q1-2026-recon",
    "opt_in": true,
    "confidence_threshold": 0.7 }
-- Each survivor becomes a Situation with
-- ExtractionMethod::Reconstructed { source_narrative_id, job_id }.
```

**"Compare reconstruction confidence vs ingested ground truth":**
```sql
-- Reconstruction lands at narrative "X-recon"; declared coordination
-- groups (e.g. ingested from DISARM) at "X". Cross-reference via
-- shared entity membership.
MATCH (s:Situation) WHERE s.narrative_id = "X-recon" AND s.confidence > 0.7 RETURN s
```

### When to filter on confidence vs weight

Per architect §13.7 of `docs/synth_reconstruction_algorithm.md`: ALWAYS
filter on `confidence > 0.7`, NOT on weight. The Taylor expansion makes
triadic terms contribute nonzero pairwise coefficients (`possible_masking_artifact = true`)
that pass the weight threshold but rarely pass the bootstrap-stability
threshold. Confidence is the right axis.

## Opinion Dynamics (BCM on hypergraphs — Phase 16c)

Use opinion dynamics when the user wants to reason about whether
agents in a narrative reach consensus or fragment, or to score how
much an intervention shifts aggregate opinion.

```sql
-- Synchronous (sub-second at MVP scales).
INFER OPINION_DYNAMICS(
    confidence_bound := 0.3,        -- c ∈ (0, 1)
    variant := 'pairwise',          -- 'pairwise' (Hickok 2022) | 'group_mean' (Schawe-Hernández 2022)
    mu := 0.5,                       -- optional convergence rate
    initial := 'uniform'             -- 'uniform' | 'gaussian' | 'bimodal'
) FOR "narr-1"

-- Phase-transition sweep: locate the critical c where convergence
-- time spikes (Hickok §5; on a complete hypergraph with N(0.5, σ²)
-- initial opinions, the spike sits near c = σ²).
INFER OPINION_PHASE_TRANSITION(
    c_start := 0.05, c_end := 0.5, c_steps := 10
) FOR "narr-1"
```

### Three load-bearing idioms

**"Does this narrative converge or fragment?":**
```sql
INFER OPINION_DYNAMICS( confidence_bound := 0.3, variant := 'pairwise' )
    FOR "narr-1"
-- → report.num_clusters == 1 ⇒ converges; > 1 ⇒ fragments.
```

**"What confidence threshold makes this group find consensus?":**
```sql
INFER OPINION_PHASE_TRANSITION( c_start := 0.05, c_end := 0.5, c_steps := 10 )
    FOR "narr-1"
-- → critical_c_estimate is the threshold above which convergence is fast.
```

**"Are there echo chambers in this corpus?":**
```sql
-- Requires Label Propagation labels first (an/lp/{nid}/{eid}).
INFER LABEL_PROPAGATION FOR n:Narrative WHERE n.id = "narr-1" RETURN n
INFER OPINION_DYNAMICS( confidence_bound := 0.3, variant := 'pairwise' )
    FOR "narr-1"
-- → report.echo_chamber_index ∈ [0, 1]; closer to 1 ⇒ strong alignment
--   between final opinion clusters and pre-existing community labels.
--   echo_chamber_available = false ⇒ run Label Propagation first.
```

### Phase 16 phase transition vs Phase 14 bistability

Don't conflate. They measure different observables on different
phenomena:

| Observable | Phase 14 Bistability | Phase 16 Phase Transition |
|---|---|---|
| What varies | Transmission rate β | Confidence bound c |
| What's measured | Final infected prevalence | Time to convergence |
| Phenomenon | Bistable interval (hysteresis) | Convergence-time spike |
| Model | SIR higher-order contagion | BCM opinion dynamics |
| Source | Ferraz de Arruda et al. 2023 | Hickok et al. 2022 §5 |

For deeper opinion-dynamics workflows, switch to the
`tensa-opinion-dynamics` skill bundle.

## Fuzzy Logic (Fuzzy Sprint, v0.78.0)

TensaQL now supports graded-truth reasoning via fuzzy t-norms,
aggregators, fuzzy Allen relations, intermediate quantifiers, graded
Peterson syllogisms, fuzzy FCA, Mamdani rules, and fuzzy-probabilistic
hybrid inference. Full chapter: `docs/TENSA_REFERENCE.md` "Fuzzy Logic".

### Cheat sheet

```sql
-- Switch the t-norm for a condition fusion
MATCH (e:Actor) WHERE e.confidence > 0.7 AND e.maturity = "Validated"
RETURN e WITH TNORM 'lukasiewicz' AGGREGATE owa

-- Fuzzy Allen relation with threshold
MATCH (s:Situation) AT s.temporal AS FUZZY OVERLAPS THRESHOLD 0.3 RETURN s

-- "Do most actors in narrative n1 have high confidence?"
QUANTIFY most (e:Actor) WHERE e.confidence > 0.7 FOR "n1"

-- Graded Peterson syllogism
VERIFY SYLLOGISM {
    major: 'Most Actor IS Influential',
    minor: 'All Influential IS Leader',
    conclusion: 'Most Actor IS Leader'
} FOR "n1"

-- Mamdani rule evaluation + FCA lattice + fuzzy-probabilistic hybrid
EVALUATE RULES FOR "n1" AGAINST (e:Actor) RULES ['elevated-disinfo-risk']
FCA LATTICE FOR "n1" ATTRIBUTES ['partisan','trusted'] ENTITY_TYPE Actor
INFER FUZZY_PROBABILITY(event_kind := 'quantifier',
    event_ref := '{"kind":"most","predicate":"e.confidence > 0.7"}',
    distribution := 'uniform') FOR "n1"
```

Every existing confidence-returning REST endpoint gains an optional
`?tnorm=<kind>&aggregator=<kind>` query string; default path is
bit-identical to pre-sprint TENSA (Gödel / Mean unless the site config
is overridden at `/fuzzy/config`).

For the full aggregator catalogue, Choquet-measure registration,
OWA linguistic-quantifier weights, and the 14 new MCP tools, switch
to the `tensa-fuzzy` skill bundle.

## Gradual Argumentation (Graded Sprint, v0.79.0)

Beyond the legacy crisp Dung extensions (grounded / preferred /
stable), TENSA now exposes four canonical gradual / ranking-based
argumentation semantics: **h-Categoriser** (Besnard & Hunter 2001),
**Weighted h-Categoriser** (Amgoud & Ben-Naim 2017), **Max-Based** +
**Card-Based** (Amgoud & Ben-Naim 2013). These return a real-valued
acceptability degree per argument (in `[0, 1]`) instead of a label.

```
POST /analysis/argumentation/gradual
{"narrative_id": "case-alpha",
 "gradual_semantics": "HCategoriser",
 "tnorm": null}
→ {"gradual": {"acceptability": {<uuid>: 0.42, ...},
               "iterations": 17, "converged": true}, ...}
```

The default `tnorm = null` reproduces the canonical paper formulas
bit-identically. Switching to `Goguen` / `Hamacher` makes the influence
step non-contracting; ALWAYS check `gradual.converged` before using
the result.

For the full convergence table, principle tests, and t-norm coupling
semantics, switch to the `tensa-graded` skill bundle.

## Learned Measures (Graded Sprint, v0.79.0)

Beyond the symmetric Choquet defaults (`additive`, `pessimistic`,
`optimistic`), TENSA now supports **ranking-supervised Choquet measure
learning** from a labelled `(input_vec, rank)` dataset via projected
gradient descent. Use this when sources non-additively interact —
the synthetic-CIB worked example shows a `+0.21` AUC gap (learned
0.85 vs additive 0.64) on coordinated cluster detection.

```
POST /fuzzy/measures/learn
{"name": "telegram-cib-2026-q1", "n": 4,
 "dataset": [{"input_vec": [0.7, 0.6, 0.4, 0.2], "rank": 0}, ...],
 "dataset_id": "tg-cib-q1-v1"}
→ 201 CREATED  {"name": "...", "version": 1, "train_auc": 0.87, "test_auc": 0.85}
```

`n` is hard-capped at 6; k-additive specialisation deferred per
Grabisch 1997. Versioned history is preserved at
`fz/tn/measures/{name}/v{N}`; the unversioned key points at the
latest. Once registered, learned measures resolve into TensaQL via
the existing `AGGREGATE CHOQUET BY '<measure_id>'` form — **no new
TensaQL clauses this sprint**.

For the full provenance contract, version-aware CRUD, ORD-Horn
path-consistency closure, and the 5 new MCP tools, switch to the
`tensa-graded` skill bundle.
