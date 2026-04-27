# TENSA Functional Test Protocol Plan

> **STATUS: ON HOLD** — Pre-audit found 6 material discrepancies between paper claims and
> actual code implementations. These must be resolved before functional tests can be written
> against stable algorithms. See `docs/DISCREPANCIES.md` for the full list and remediation
> plan.

## Overview

**Purpose:** Verify correctness of algorithms, mathematical properties, and logical
functionality via professional test protocols with comparative hypotheses against baselines
(LLM-in-context, LightRAG, GraphRAG, CoT prompting).

**Evaluation Framework** (from 2nd paper outline):
- **TCS** (Temporal Consistency Score): correct temporal orderings / total temporal claims
- **CVI** (Counterfactual Validity Index): correct intervention target + chain propagation + final state
- **IIS** (Information Isolation Score): 1 - information leakage rate (global-to-local knowledge bleed)

**Test Data:** 5 full-length narratives in `tests/functional/test_data/` as `.tensa` archives
(pre-enriched, no LLM at test time). Primary narrative: "The Gull's Last Crossing" (murder
mystery) with 12 temporal traps, 9 causal traps, 7 game-theoretic traps formally specified
in `PAPER/paper_2/stories/outlines/01_mystery_outline.md`.

---

## Protocol Format

Every test follows this structure:

```
### XX-NN: Title
**Description:** What capability is being tested.
**Hypothesis:** Why TENSA should produce correct results where baselines fail.
  Cites specific TENSA mechanism and specific baseline limitation.
**Preconditions:** Required data setup.
**Procedure:** Step-by-step test method.
**Expected Outcome:** Concrete assertion with numeric thresholds.
**Metric:** TCS | CVI | IIS | Correctness
**Module:** Rust module under test.
```

---

## Batch 1: Algorithm Unit Tests (82 tests, no story data)

Pure algorithmic correctness. Small hand-built fixtures via Hypergraph API. Temp RocksDB.

### Category Hypothesis Themes

| Category | # | Core Hypothesis |
|----------|---|----------------|
| Temporal (Allen) | 6 | Deterministic 13x13 composition table vs LLM hallucinated compositions on 3+ hop chains |
| Causal (NOTEARS) | 5 | Temporal mask enforces DAG acyclicity vs LLMs producing backward-in-time links |
| Network (centrality, Leiden) | 7 | Leiden refinement guarantees connected communities vs Louvain disconnected clusters |
| Information theory | 6 | Deterministic log2/contingency tables vs LLM inability to compute joint probabilities |
| Epistemic (beliefs, DS) | 6 | Per-entity knowledge set tracking vs LLM global knowledge leakage |
| Argumentation (Dung) | 4 | Unique fixed-point grounded extension vs LLM inconsistent extensions |
| Contagion (SIR) | 4 | R0 from explicit spread DAG vs LLM intuitive spread estimation |
| Game theory (QRE) | 6 | Exact softmax probabilities summing to 1.0 vs LLM approximate distributions |
| Motivation (IRL) | 5 | Gradient-descent feature weights vs LLM keyword-based archetype assignment |
| Cross-narrative (WL, arcs) | 7 | Permutation-invariant graph comparison vs LLM surface text similarity |
| Stylometry (Delta) | 4 | 100 function-word z-scores vs LLM content-based style judgment |
| Spatial (Haversine) | 2 | Exact spherical distance computation |
| Vector similarity | 4 | Deterministic cosine similarity |
| TensaQL query engine | 10 | Grammar-driven parser/executor correctness |
| Anomaly detection | 2 | Statistical z-score > 2.0 vs LLM narrative-salience flagging |
| Entity operations | 4 | Transactional participation transfer and state versioning |

See `TEST_PLAN.md` for individual test descriptions (T-01 through EO-04).

---

## Batch 2: Mystery Narrative Tests (34 tests, requires 01_mystery.tensa)

Cross-referenced to trap IDs from `PAPER/paper_2/stories/outlines/01_mystery_outline.md`.

### 2A: Temporal Trap Tests (12 tests, TCS metric)

| ID | Trap | Probe | TENSA Mechanism | Baseline Failure |
|----|------|-------|-----------------|-----------------|
| MT-01 | T1 | "On what day/time was the body discovered?" | Allen index point query -> D 06:50 | LLM infers D-1 from Ch.1 lantern context |
| MT-02 | T2 | "Did Calloway arrive before or after Prue's photo?" | Allen BEFORE: arrival D-2 vs photo D-1 | LLM loses temporal order across chapters |
| MT-03 | T3 | "When did Rosalind know vs when she disclosed?" | Belief model timestamps vs disclosure timestamp | LLM conflates knowing with acting |
| MT-04 | T4 | "What time did Bram's boat return?" | Situation temporal.start = D-1 16:20 | LLM retrieves Bram's claim, not photo evidence |
| MT-05 | T5 | "Where was Rosalind at 22:40 on D-1?" | get_state_at_time -> tower stairs | LLM retrieves Ch.3 partial flashback: "room" |
| MT-06 | T6 | "Was Calloway asleep at 22:35?" | Contention: claim vs Oake's log, argumentation resolves | LLM takes Ch.6 statement as authoritative |
| MT-07 | T7 | "Did Vesper expect her interview to happen?" | InfoSet(Vesper, D-1 21:00) contains appointment | LLM misses exculpatory timing |
| MT-08 | T8 | "Envelope chain of custody?" | Causal chain traversal: Quay -> Cormac -> killer | Retrieval scatters across 3 chapters |
| MT-09 | T9 | "When did Edmund first read the envelope?" | Temporal query on knowledge-acquisition situation | LLM conflates reader-learns with character-learns |
| MT-10 | T10 | "Which day is 'second night of the blow'?" | Allen DURING: storm D-1 situation | LLM parses relative to discourse position |
| MT-11 | T11 | "Was Bram on island at D-1 16:20?" | Contention: deposition vs photo evidence | LLM defaults to Bram's deposition |
| MT-12 | T12 | "When did Voss learn Bram returned?" | Belief model: knowledge acquisition timestamp | LLM conflates strategic timing with ignorance |

### 2B: Causal-Counterfactual Trap Tests (9 tests, CVI metric)

| ID | Trap | Probe | TENSA Mechanism | Baseline Failure |
|----|------|-------|-----------------|-----------------|
| MC-01 | C1 | "If envelope water-damaged, would inheritance be voided?" | Conditional edge: legibility=TRUE required | LLM retrieves chain without checking condition |
| MC-02 | C2 | "If no sedative, would Quay be on stairs at 22:55?" | Counterfactual: sedative causally independent of tower visit | LLM attributes causal weight to temporal proximity |
| MC-03 | C3 | "If no power failure, would murder occur?" | Enabling condition vs sufficient cause distinction | LLM conflates enabling with sufficient |
| MC-04 | C4 | "Was envelope handover caused by the argument?" | Confounded: both downstream of D-2 safe tampering | LLM infers causation from temporal correlation |
| MC-05 | C5 | "If storm broke earlier, would Bram be on island?" | Conditional: storm_window_gives_cover=TRUE | Retrieval cannot reason about absent conditions |
| MC-06 | C6 | "If Rosalind spoke at 23:00, would Edmund be caught?" | 10-min window: murder 22:55, planting 23:05 | LLM cannot simulate temporal intervention |
| MC-07 | C7 | "Did Vesper's arrival prompt the envelope?" | Confounded: both caused by Quay's press letter | LLM sees narrative proximity as causation |
| MC-08 | C8 | "If Calloway actually asleep, is case provable?" | Non-load-bearing evidence identification | LLM overweights dramatic evidence |
| MC-09 | C9 | "If press letter intercepted, would Quay be killed?" | Conditional: signal_received_by_Edmund=TRUE | Retrieval has no conditional edge evaluation |

### 2C: Game-Theoretic Trap Tests (7 tests, IIS metric)

| ID | Trap | Probe | TENSA Mechanism | Baseline Failure |
|----|------|-------|-----------------|-----------------|
| MG-01 | G1 | "What does Edmund believe Voss knows about envelope?" | InfoSet partitioning at Ch.4 | LLM has no per-actor knowledge partition |
| MG-02 | G2 | "Edmund's rational action re: Calloway's sleep claim?" | Deceptive signal analysis under incomplete info | LLM answers from omniscient perspective |
| MG-03 | G3 | "Rosalind's best response given p(Voss suspects Edmund)?" | QRE threshold computation: p* ~ 0.6 | LLM cannot compute from payoff structure |
| MG-04 | G4 | "What should Bram say about D-1 whereabouts?" | InfoSet(Bram): believes photo unknown to Voss | Retrieval returns global facts |
| MG-05 | G5 | "Under what conditions do Oake and Cormac both disclose?" | Coordination game: sequential disclosure dominance | LLM describes but cannot compute equilibrium |
| MG-06 | G6 | "Rational interpretation of Edmund's volunteered statement?" | Negative information content -> guilt signal | LLM treats volunteered info as helpful |
| MG-07 | G7 | "What does Edmund believe Voss believes about Calloway?" | Depth-2: beliefs[Edmund][Voss] re Calloway | LLM cannot track 2nd-order beliefs |

### 2D: Full Narrative Analysis Tests (6 tests, Correctness metric)

| ID | Analysis | Expected Assertion |
|----|----------|-------------------|
| MN-01 | Centrality | Voss, Quay highest; Slane lowest |
| MN-02 | Community detection | >=2 communities (investigation vs suspects) |
| MN-03 | Arc classification | Man-in-a-Hole or Oedipus arc |
| MN-04 | Causal chain discovery | Forgery -> investigation -> confrontation -> murder |
| MN-05 | Style anomaly | Flag chapters with shifted narrative voice (if any) |
| MN-06 | Contagion R0 | Edmund identified as critical spreader for envelope knowledge |

---

## Execution Strategy

### Phase 1: Fix Discrepancies
Resolve all items in `docs/DISCREPANCIES.md` before writing any test code.

### Phase 2: Build Test Fixtures
1. Run `/build-tensa-archive` with `enrich: true` for each of the 5 narratives
2. Commit `.tensa` files to `tests/functional/test_data/`
3. Create `mystery_ground_truth.json` hand-annotated answer key

### Phase 3: Implement Batch 1 (82 algorithm unit tests)
Rust `#[test]` functions, temp RocksDB, no LLM, no server.

### Phase 4: Implement Batch 2 (34 mystery narrative tests)
Load `01_mystery.tensa` into temp RocksDB. Run analysis pipelines.
Compare against ground truth from outline traps.

---

## Test Count Summary

| Batch | Category | Count |
|-------|----------|-------|
| 1 | Algorithm unit tests | 82 |
| 2A | Temporal traps (TCS) | 12 |
| 2B | Causal traps (CVI) | 9 |
| 2C | Game-theoretic traps (IIS) | 7 |
| 2D | Full narrative analysis | 6 |
| **Total** | | **116** |
