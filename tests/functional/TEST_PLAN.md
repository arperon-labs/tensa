# TENSA Functional Test Plan

**Purpose:** Verify correctness of algorithms, mathematical properties, and logical functionality.
Separate from e2e/integration tests (which test API wiring and CRUD). These tests validate
that TENSA's analytical engines produce *correct results* against known ground truth.

**Test Narrative:** "The Gull's Last Crossing" (murder mystery) + 4 companion narratives.
Full-length versions (~170-210KB each) in `tests/functional/test_data/`. These provide rich
ground truth for temporal, causal, game-theoretic, and cross-narrative testing.

---

## Test Data Strategy

### Primary Narrative: "The Gull's Last Crossing" (`01_mystery.md`)

Scottish island murder mystery. DI Mairead Voss investigates retired magistrate Halvor Quay's
death at a lighthouse during a storm. 9 characters, tight temporal window, provable causal chains.

**Ground truth entities (9):**
| Entity | Type | Notes |
|--------|------|-------|
| DI Mairead Voss | Actor | Investigator / Protagonist |
| Halvor Quay | Actor | Victim (deceased magistrate) |
| Edmund Thornwick | Actor | Innkeeper, suspect (inheritance motive) |
| Rosalind Thornwick | Actor | Edmund's wife, suspect (secret anxiety) |
| Dr. Oake | Actor | Physician on island |
| Prue Ashenden | Actor | Naturalist, witness |
| Vesper Linn | Actor | Journalist, has outside knowledge of 2004 case |
| Hollis Calloway | Actor | Former DCI, key suspect (caught in lie) |
| Cormac Slane | Actor | Lighthouse keeper, discovered body |

**Ground truth locations:** Lighthouse, inn/pub, harbour, Quay's room, store cupboard, spiral staircase landing

**Ground truth temporal window:**
- Storm begins: evening
- Power failure: ~22:15
- Murder window: 22:15 - ~23:00 (30-45 min)
- Body found: dawn (~06:00 next day)

**Ground truth causal chains:**
1. Thornwick estate forgery (2004) -> Quay investigates -> confrontation -> murder
2. Storm + isolation -> secrecy advantageous -> evidence destruction attempt
3. Hollis's secret boat return -> unexplained presence -> alibi problems

**Ground truth information asymmetries:**
- Edmund knows about inheritance threat (others don't)
- Rosalind was at store cupboard near tower (lies about it)
- Calloway claims asleep from 22:00 but was awake at 22:35 (caught in contradiction)
- Vesper knows about 2004 estate case (outside knowledge)
- Hollis returned by boat (photograph evidence)

### Cross-Narrative Corpus (5 total):
| # | File | Genre | Key Features |
|---|------|-------|-------------|
| 1 | `01_mystery.md` | Murder mystery | Temporal reasoning, causal discovery, deception |
| 2 | `02_political_thriller.md` | Political thriller | Power dynamics, alliance networks, betrayal |
| 3 | `03_hard_scifi.md` | Hard sci-fi | Technical causal chains, disaster timeline |
| 4 | `04_historical_espionage.md` | WWII espionage | Double agents, information asymmetry |
| 5 | `05_corporate_intrigue.md` | Corporate intrigue | Financial motives, insider trading |

### Data Preparation

1. **Build `.tensa` archives** using `/build-tensa-archive` skill with `enrich: true` for each
   narrative. This produces enriched archives with beliefs, game structures, temporal relations,
   and causal links already populated — no LLM needed at test time.
2. Store both `.md` source files (for reference) and `.tensa` archives (actual test fixtures)
   in `tests/functional/test_data/`.
3. Create a hand-annotated ground truth JSON for the mystery (entities, situations, temporal
   bounds, causal links, participations) that serves as the "answer key".
4. **Use RocksDB** (temp directories via `tempfile` crate) for all functional tests. These
   tests validate algorithm correctness on real narrative-scale data — production storage
   behavior matters more than speed here. Archives are loaded via the standard import path.

```
tests/functional/test_data/
├── 01_mystery.md                  # Source text (reference)
├── 01_mystery.tensa               # Enriched archive (test fixture)
├── 02_political_thriller.md
├── 02_political_thriller.tensa
├── 03_hard_scifi.md
├── 03_hard_scifi.tensa
├── 04_historical_espionage.md
├── 04_historical_espionage.tensa
├── 05_corporate_intrigue.md
├── 05_corporate_intrigue.tensa
└── mystery_ground_truth.json      # Hand-annotated answer key
```

---

## Category 1: Temporal Reasoning (Allen Interval Algebra)

### T-01: Allen relation computation (13 relations)
**Claim:** TENSA correctly computes all 13 Allen relations between two intervals.
**Method:** Create 13 interval pairs that exemplify each relation (Before, After, Meets, MetBy,
Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals).
Verify `classify()` returns the correct relation for each pair.
**Module:** `temporal::interval`

### T-02: Allen composition table (13x13 = 169 entries)
**Claim:** The composition of two Allen relations yields the correct set of possible relations per
the standard composition table from Allen (1983).
**Method:** For each of the 169 (R1, R2) pairs, verify `compose(R1, R2)` matches the known
composition table. Use reference table from Allen's original paper.
**Module:** `temporal::interval`

### T-03: Interval tree point queries
**Claim:** Point queries on the interval tree return all intervals containing that point.
**Method:** Insert 20 intervals with known overlaps. Query specific points and verify the returned
set matches expectations (manually computed).
**Module:** `temporal::index`

### T-04: Interval tree Allen queries
**Claim:** Allen-relation queries on the interval tree return all intervals satisfying the given
relation with a reference interval.
**Method:** Insert intervals from the mystery timeline (storm, murder window, interviews). Query:
"All situations DURING the storm" — verify correct subset returned. "All situations BEFORE
the body discovery" — verify temporal ordering.
**Module:** `temporal::index`

### T-05: AT clause execution in TensaQL
**Claim:** `MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01" RETURN s` correctly filters.
**Method:** Create situations with known timestamps. Run AT queries with each of the supported
relations (BEFORE, AFTER, MEETS, OVERLAPS, DURING, CONTAINS, STARTS, FINISHES, EQUALS).
Verify result sets match expected.
**Module:** `query::executor` (FilterTemporal plan step)

### T-06: Temporal interval persistence
**Claim:** IntervalTree survives save/load to KV store without data loss.
**Method:** Build tree with 50 intervals, save to MemoryStore, create new tree, load from store,
verify identical query results.
**Module:** `temporal::index`

---

## Category 2: Causal Inference

### C-01: Causal link cycle detection
**Claim:** Adding a causal link that would create a cycle is rejected.
**Method:** Create chain A -> B -> C. Attempt C -> A. Verify error returned.
**Module:** `hypergraph::causal`

### C-02: NOTEARS causal discovery — known structure recovery
**Claim:** NOTEARS discovers hidden causal links from situation features.
**Method:** Create 10 situations with known causal structure (A -> B -> C, D -> C, E -> F).
Set features (confidence, participant count, etc.) to correlate along causal paths.
Run NOTEARS. Verify discovered links are a superset of the ground truth and no
backward-in-time links exist.
**Module:** `inference::causal` (CausalEngine)

### C-03: NOTEARS temporal mask
**Claim:** NOTEARS only discovers forward-in-time causal links.
**Method:** Create situations where temporal order is reversed from feature correlation.
Verify NOTEARS respects temporal mask and does not produce anachronistic links.
**Module:** `inference::causal`

### C-04: NOTEARS link strength classification
**Claim:** Links > 0.8 classified as Necessary, 0.5-0.8 as Contributing, < 0.5 as Enabling.
**Method:** Run NOTEARS on crafted data, verify classification thresholds match.
**Module:** `inference::causal`

### C-05: Counterfactual beam search
**Claim:** Given an intervention, forward-propagation through causal DAG produces affected
situations with accumulated probability.
**Method:** Build causal DAG (5 nodes, 4 links with known strengths). Intervene on root node.
Verify affected nodes and P(outcome) values match hand-computed expectations.
Beam width = 5, verify pruning behavior.
**Module:** `inference::causal` (counterfactual)

### C-06: Causal chain from mystery ground truth
**Claim:** After ingesting the mystery, causal discovery finds the forgery -> investigation ->
confrontation -> murder chain.
**Method:** Ingest mystery narrative. Run INFER CAUSES. Check that the main causal chain
appears in discovered links (may require minimum confidence threshold).
**Module:** Full pipeline integration

---

## Category 3: Network Analysis (Centrality & Community)

### N-01: Betweenness centrality — star graph
**Claim:** In a star graph, the center node has betweenness = 1.0, leaves have 0.0.
**Method:** Create star topology: center entity participates in all situations, each leaf
in only one with center. Run centrality. Verify center = 1.0, leaves = 0.0.
**Module:** `analysis::centrality`

### N-02: Betweenness centrality — line graph
**Claim:** In a line A-B-C-D-E, the middle node C has highest betweenness.
**Method:** Create chain topology. Run centrality. Verify C > B = D > A = E.
**Module:** `analysis::centrality`

### N-03: Closeness centrality — connected graph
**Claim:** Central node in connected graph has highest closeness score.
**Method:** Create small connected graph (6 nodes). Compute by hand. Verify engine matches.
**Module:** `analysis::centrality`

### N-04: Degree centrality — normalization
**Claim:** Degree(v) = neighbors / (total_entities - 1), range [0, 1].
**Method:** Create graph with known degree distribution. Verify formula.
**Module:** `analysis::centrality`

### N-05: Leiden community detection — basic clustering
**Claim:** Leiden finds meaningful communities in a graph with clear cluster structure.
**Method:** Create two densely connected clusters with one bridge edge. Verify Leiden
separates them into two communities. Verify communities are internally connected.
**Module:** `analysis::centrality` (leiden)

### N-06: Leiden guarantees connected communities
**Claim:** Unlike Louvain, Leiden refinement step guarantees all members of a community
are reachable from each other within the community.
**Method:** Create graph where naive modularity optimization would group disconnected
nodes. Verify Leiden splits them. Compare with what Louvain would produce.
**Module:** `analysis::centrality`

### N-07: Hierarchical Leiden — multi-level
**Claim:** Recursive Leiden produces hierarchy with parent/child links.
**Method:** Create 3-tier community structure (3 macro-communities, each with 2 sub-clusters).
Run hierarchical Leiden. Verify at least 2 levels and correct nesting.
**Module:** `analysis::centrality` (hierarchical_leiden), `analysis::community`

### N-08: Centrality on mystery narrative
**Claim:** After ingesting the mystery, DI Voss and Quay should have high centrality
(they connect to most characters). Cormac Slane (peripheral) should have low.
**Method:** Ingest mystery. Run centrality. Verify ranking order.
**Module:** Full pipeline integration

---

## Category 4: Information Theory

### I-01: Self-information — uniform vs skewed distribution
**Claim:** Rare situations have higher self-information than common ones.
**Method:** Create 20 situations: 15 with identical signature, 5 with unique signatures.
Verify the 5 unique have higher self-information than the 15 common.
**Module:** `analysis::entropy`

### I-02: Self-information formula correctness
**Claim:** I(s) = -log2(frequency) where frequency = count(matching_signature) / total.
**Method:** Create known distribution. Hand-compute expected values. Verify match.
**Module:** `analysis::entropy`

### I-03: Mutual information — correlated entities
**Claim:** Entities that always appear together have positive MI. Independent entities have MI near 0.
**Method:** Create 10 situations. A and B always co-occur (5 situations). C appears independently.
MI(A,B) should be positive. MI(A,C) should be near 0.
**Module:** `analysis::entropy`

### I-04: Mutual information formula
**Claim:** MI(A,B) = sum P(x,y) * log2(P(x,y) / (P(x)*P(y))) over the 2x2 contingency table.
**Method:** Hand-compute from known co-occurrence. Verify engine matches to 4 decimal places.
**Module:** `analysis::entropy`

### I-05: KL divergence — uniform baseline
**Claim:** Entity with uniform action distribution has KL = 0. Concentrated has high KL.
**Method:** Entity X: actions [a,b,c,d,e] equally. Entity Y: all action "a". Verify KL(X) ~ 0,
KL(Y) >> 0.
**Module:** `analysis::entropy`

### I-06: KL divergence skip for sparse data
**Claim:** KL divergence only computed for entities with >= 2 distinct actions.
**Method:** Create entity with 1 action repeated. Verify no KL result or explicitly skipped.
**Module:** `analysis::entropy`

---

## Category 5: Epistemic Reasoning

### E-01: Recursive belief modeling — basic knowledge tracking
**Claim:** After two entities co-participate in a situation with InfoSets, the engine correctly
tracks what each knows.
**Method:** Create 3 situations with InfoSets:
- Sit1: A reveals fact-X to B (both present)
- Sit2: B reveals fact-Y to C (A absent)
- Sit3: All three present, C reveals fact-Z
Verify: A knows {X, Z}, B knows {X, Y, Z}, C knows {Y, Z}.
A believes B knows {X, Z} (misses Y because A wasn't present for Sit2).
**Module:** `analysis::beliefs`

### E-02: Belief gaps — false beliefs
**Claim:** Engine detects when A thinks B knows something B actually doesn't.
**Method:** Construct scenario where A was told "B knows X" but B never learned X.
Verify gap type "false belief" flagged.
**Module:** `analysis::beliefs`

### E-03: Belief gaps — unknown to A
**Claim:** Engine detects facts B knows that A doesn't know B knows.
**Method:** B learns fact in a situation A is absent from. Verify gap flagged.
**Module:** `analysis::beliefs`

### E-04: Dempster-Shafer — two agreeing sources
**Claim:** Two sources both supporting hypothesis H produce Bel(H) > either individually.
**Method:** Source1: m({H}) = 0.6, m(Theta) = 0.4. Source2: m({H}) = 0.7, m(Theta) = 0.3.
Hand-compute combined. Verify Bel(H) matches and Pl(H) >= Bel(H).
**Module:** `analysis::evidence`

### E-05: Dempster-Shafer — conflicting sources
**Claim:** Two sources supporting contradictory hypotheses produce high conflict K.
**Method:** Source1: m({H1}) = 0.9. Source2: m({H2}) = 0.9. Verify K is high (> 0.8).
Verify Bel/Pl intervals are wide (high uncertainty).
**Module:** `analysis::evidence`

### E-06: Dempster-Shafer — claim-aware mass concentration
**Claim:** When a source has a claim matching a hypothesis frame element, mass concentrates
on that hypothesis instead of distributing uniformly.
**Method:** Create source with explicit claim. Verify mass function reflects claim.
**Module:** `analysis::evidence`

### E-07: Mystery information asymmetry test
**Claim:** After ingesting the mystery with InfoSets, the belief engine correctly identifies
that Calloway knows something about the murder that Voss doesn't (yet).
**Method:** Ingest mystery with enrichment. Run beliefs analysis. Verify asymmetries.
**Module:** Full pipeline integration

---

## Category 6: Argumentation

### A-01: Grounded extension — unattacked argument wins
**Claim:** An argument with no attackers is always in the grounded extension.
**Method:** Create framework: A attacks B, C is unattacked. Verify grounded = {C} (or {A,C}
depending on whether A is self-defending).
**Module:** `analysis::argumentation`

### A-02: Grounded extension — self-attacking argument excluded
**Claim:** Self-attacking arguments are OUT in grounded semantics.
**Method:** Create A attacks A. Verify A is not in grounded extension.
**Module:** `analysis::argumentation`

### A-03: Preferred extensions — multiple valid sets
**Claim:** Preferred semantics can yield multiple maximal admissible sets.
**Method:** Create odd cycle (A attacks B, B attacks C, C attacks A). Verify multiple
preferred extensions returned ({A}, {B}, {C}).
**Module:** `analysis::argumentation`

### A-04: Stable extension — existence and uniqueness
**Claim:** Stable extension attacks all non-members. May not exist for all frameworks.
**Method:** Test with framework that has stable extension and one that doesn't.
Verify correct presence/absence.
**Module:** `analysis::argumentation`

### A-05: Contention resolution via argumentation
**Claim:** Conflicting source claims can be modeled as arguments and resolved.
**Method:** Create two sources with contradictory claims. Model as arguments with
mutual attacks. Run grounded/preferred. Verify resolution matches expectation
(e.g., higher-trust source wins if unattacked).
**Module:** `analysis::argumentation` + source intelligence

---

## Category 7: Contagion (SIR Model)

### S-01: Basic SIR state transitions
**Claim:** Entities transition S -> I -> R correctly based on InfoSet reveals/learns.
**Method:** 3 entities, 3 situations. A reveals fact in Sit1 (A: I). B learns from A in Sit2
(B: I). A does not reveal in Sit3 (A: R). Verify state at each step.
**Module:** `analysis::contagion`

### S-02: R0 computation
**Claim:** R0 = average secondary infections per spreader.
**Method:** Create scenario: A infects {B, C}. B infects {D}. C infects nobody.
R0 = (2 + 1 + 0) / 3 = 1.0. Verify.
**Module:** `analysis::contagion`

### S-03: Critical spreader identification
**Claim:** Removing the entity with highest R0 reduction is the "critical spreader."
**Method:** Star topology: A spreads to B, C, D, E (all via A). Remove A: R0 drops to 0.
Remove B: R0 drops slightly. Verify A identified as critical.
**Module:** `analysis::contagion`

### S-04: R0 interpretation thresholds
**Claim:** R0 < 1 = dies out, R0 = 1 = stable, R0 > 1 = viral.
**Method:** Create 3 scenarios with known R0 values. Verify classification.
**Module:** `analysis::contagion`

---

## Category 8: Game Theory

### G-01: Game classification — zero-sum detection
**Claim:** When payoffs sum to ~0, the game is classified as ZeroSum.
**Method:** Create situation with 2 participants, payoffs (3, -3). Verify ZeroSum classification.
**Module:** `inference::game`

### G-02: Game classification — prisoner's dilemma
**Claim:** Classic PD payoff structure detected.
**Method:** Create situation with payoff matrix matching PD (T > R > P > S, T+S < 2R).
Verify PrisonersDilemma classification.
**Module:** `inference::game`

### G-03: Game classification — N-player types
**Claim:** 3+ player games classified as Auction, AsymmetricInformation, or Coordination.
**Method:** Create situations with 4 participants, various payoff structures. Verify.
**Module:** `inference::game`

### G-04: QRE solver — extreme rationality
**Claim:** As lambda -> infinity, QRE converges to Nash equilibrium (deterministic best response).
**Method:** Create 2-player game with dominant strategy. Set high lambda. Verify QRE
probabilities concentrate on dominant strategy.
**Module:** `inference::game`

### G-05: QRE solver — zero rationality
**Claim:** At lambda = 0, QRE produces uniform distribution over actions.
**Method:** Create game, set lambda = 0. Verify all actions equally probable.
**Module:** `inference::game`

### G-06: QRE lambda estimation from observed play
**Claim:** Grid search finds lambda that maximizes log-likelihood against observed actions.
**Method:** Create game with known lambda (e.g., 3.0). Generate observed actions from QRE
at that lambda. Run estimator. Verify recovered lambda is close to 3.0.
**Module:** `inference::game`

---

## Category 9: Motivation Analysis

### M-01: MaxEnt IRL — reward weight learning
**Claim:** Given an entity's trajectory, MaxEnt IRL learns feature weights that explain behavior.
**Method:** Create entity with clear pattern (always protagonist, high payoff situations).
Run IRL. Verify protagonist-weight and payoff-weight are dominant.
**Module:** `inference::motivation`

### M-02: Archetype classification — PowerSeeking
**Claim:** Entity with high antagonist + high payoff weights classified as PowerSeeking.
**Method:** Create entity trajectory: always antagonist role, high payoffs. Verify archetype.
**Module:** `inference::motivation`

### M-03: Archetype classification — Altruistic
**Claim:** Entity with high reveal + low knowledge weight classified as Altruistic.
**Method:** Create entity that always reveals info, gains little. Verify archetype.
**Module:** `inference::motivation`

### M-04: Archetype classification — sparse data fallback
**Claim:** With < 5 actions, classification falls back to keyword matching with lower confidence.
**Method:** Create entity with 2 actions containing "revenge" keyword. Verify Vengeful archetype
with confidence 0.3-0.5.
**Module:** `inference::motivation`

### M-05: Archetype classification — all 7 types
**Claim:** Each of the 7 archetypes (PowerSeeking, Altruistic, Ideological, Opportunistic,
SelfPreserving, Loyal, Vengeful) can be triggered by appropriate inputs.
**Method:** Create 7 entities, each with trajectory targeting one archetype. Verify all 7.
**Module:** `inference::motivation`

---

## Category 10: Cross-Narrative Analysis

### X-01: WL subtree kernel — identical graphs
**Claim:** Two identical narrative graphs have kernel similarity = 1.0.
**Method:** Create two narratives with identical structure. Compute WL kernel. Verify = 1.0.
**Module:** `narrative::similarity`

### X-02: WL subtree kernel — completely different graphs
**Claim:** Two unrelated narrative graphs have low kernel similarity.
**Method:** Create two narratives with completely different structures. Verify similarity < 0.3.
**Module:** `narrative::similarity`

### X-03: Reagan 6-arc classification
**Claim:** Fortune trajectories are classified into one of 6 Reagan arcs via Pearson correlation.
**Method:** Create narrative with known fortune trajectory (e.g., Rags-to-Riches = monotonically
increasing). Verify classification matches expected arc.
**Module:** `narrative::arc`

### X-04: All 6 Reagan arcs distinguishable
**Claim:** Each of the 6 arcs (Rags-to-Riches, Riches-to-Rags, Man-in-a-Hole, Icarus,
Cinderella, Oedipus) has a distinct template that can be triggered.
**Method:** Create 6 narratives each with fortune trajectory matching one arc. Verify all 6
correctly classified.
**Module:** `narrative::arc`

### X-05: Frequent subgraph mining
**Claim:** Recurring structural patterns across narratives are discovered.
**Method:** Create 3 narratives each containing a triangle pattern (A-B-C all co-participating).
Run pattern mining. Verify triangle discovered as frequent pattern.
**Module:** `narrative::pattern`

### X-06: VF2-lite matching (max 6 nodes)
**Claim:** Pattern matching finds occurrences of a pattern graph within a larger narrative.
**Method:** Define 4-node pattern. Create narrative with 2 instances of that pattern embedded
in a larger graph. Verify both found.
**Module:** `narrative::pattern`

### X-07: Missing event prediction — causal gap detection
**Claim:** When a causal chain has a gap (A -> ? -> C), the engine predicts the missing event.
**Method:** Create narrative where pattern mining shows A-B-C chains in other narratives.
Create test narrative with only A-C. Verify B predicted as missing.
**Module:** `narrative::prediction`

### X-08: Cross-narrative comparison with 5 test narratives
**Claim:** Pattern mining across the 5 test narratives finds shared structural patterns
(e.g., "investigation-discovery-confrontation" sequences).
**Method:** Ingest all 5 narratives. Run DISCOVER PATTERNS ACROSS NARRATIVES. Examine results
for meaningful patterns.
**Module:** Full pipeline integration

---

## Category 11: Stylometry & Narrative Fingerprint

### F-01: Burrows' Delta — same text = 0.0
**Claim:** Delta of a text compared to itself is 0.0.
**Method:** Compute Delta(text, text). Verify exactly 0.0.
**Module:** `analysis::style_profile` (prose features)

### F-02: Burrows' Delta — different authors > 1.5
**Claim:** Texts by different authors typically have Delta > 1.5.
**Method:** Use two narratives from different genres in test data. Verify Delta > 1.5.
**Module:** `analysis::style_profile`

### F-03: Type-Token Ratio computation
**Claim:** TTR is computed correctly with windowed approach at 1000 words.
**Method:** Create text with known vocabulary. Hand-compute TTR. Verify match.
**Module:** `analysis::style_profile`

### F-04: Sentence length statistics
**Claim:** Mean, std, CV, and lag-1 autocorrelation of sentence lengths computed correctly.
**Method:** Create text with known sentence lengths. Verify statistics.
**Module:** `analysis::style_profile`

### F-05: Style profile 6-layer computation
**Claim:** The 6-layer style profile (structural rhythm, character dynamics, information
management, causal architecture, temporal texture, graph topology) produces non-zero
values for a narrative with sufficient data.
**Method:** Ingest a narrative. Run style profile. Verify all 6 layers present with values.
**Module:** `analysis::style_profile`, `narrative::*`

### F-06: Style similarity — cosine similarity
**Claim:** Two similar narratives have higher style similarity than two dissimilar ones.
**Method:** Compare mystery vs. espionage (both investigation-heavy) and mystery vs. sci-fi.
Verify mystery-espionage > mystery-scifi.
**Module:** `analysis::style_profile`

### F-07: Style anomaly detection — chapter deviation
**Claim:** A chapter with different style (e.g., inserted from another author) is flagged.
**Method:** Create narrative where chapter 3 is replaced with text from a different genre.
Run anomaly detection. Verify chapter 3 flagged with score < 0.7.
**Module:** `analysis::style_profile`

### F-08: Radar chart — 12-axis normalized values
**Claim:** Radar chart produces 12 normalized [0,1] values covering pacing, ensemble,
causal density, info R0, deception, temporal complexity, strategic variety,
power asymmetry, protagonist focus, late revelation, subplot richness, surprise.
**Method:** Ingest narrative. Compute radar. Verify all 12 axes present and in [0,1].
**Module:** `analysis::style_profile`

---

## Category 12: Spatial Reasoning

### SP-01: Haversine distance computation
**Claim:** SPATIAL clause correctly computes Haversine distance in kilometers.
**Method:** Create two situations with known lat/lng. Known distance (e.g., NYC to LA = 3944 km).
Query `SPATIAL WITHIN 4000 KM OF (NYC)` should include LA. `WITHIN 3000 KM` should not.
**Module:** `query::executor` (FilterSpatial)

### SP-02: Haversine edge cases
**Claim:** Haversine works at poles, across antimeridian, and at equator.
**Method:** Test with coordinates at (90,0), (-90,0), (0,179), (0,-179). Verify distances.
**Module:** `query::executor`

---

## Category 13: Vector Similarity (NEAR clause)

### V-01: Cosine similarity — identical vectors
**Claim:** Cosine similarity of identical vectors = 1.0.
**Method:** Create two identical embeddings. Verify cosine_similarity = 1.0.
**Module:** `ingestion::embed`

### V-02: Cosine similarity — orthogonal vectors
**Claim:** Cosine similarity of orthogonal vectors = 0.0.
**Method:** Create [1,0,0] and [0,1,0]. Verify cosine_similarity = 0.0.
**Module:** `ingestion::embed`

### V-03: NEAR clause k-nearest neighbors
**Claim:** NEAR returns exactly k closest entities by cosine distance.
**Method:** Create 10 entities with known embeddings. Query NEAR with k=3. Verify top-3 by
hand-computed distances.
**Module:** `query::executor` (VectorNear)

### V-04: HashEmbedding determinism
**Claim:** HashEmbedding produces identical embeddings for identical inputs.
**Method:** Embed same text twice. Verify byte-identical results.
**Module:** `ingestion::embed`

---

## Category 14: TensaQL Query Engine

### Q-01: MATCH with property filter
**Claim:** `WHERE e.confidence > 0.5` correctly filters.
**Method:** Create entities with confidence 0.3, 0.5, 0.7, 0.9. Verify only 0.7 and 0.9 returned.
**Module:** `query::executor`

### Q-02: OR conditions with precedence
**Claim:** `(A OR B) AND C` evaluates correctly with AND binding tighter in absence of parens,
and parens overriding.
**Method:** Test: `A OR B AND C` should mean `A OR (B AND C)`.
Test: `(A OR B) AND C` should mean `(A OR B) AND C`. Verify different results.
**Module:** `query::parser`, `query::executor`

### Q-03: Aggregation — COUNT, SUM, AVG, MIN, MAX
**Claim:** Aggregate functions produce correct values.
**Method:** Create 10 entities with known confidence values. Run COUNT(*), SUM(confidence),
AVG(confidence), MIN(confidence), MAX(confidence). Verify against hand computation.
**Module:** `query::executor`

### Q-04: GROUP BY
**Claim:** GROUP BY correctly partitions results and applies aggregates per group.
**Method:** Create entities: 3 Actors, 2 Locations, 1 Artifact. Run
`GROUP BY e.entity_type RETURN e.entity_type, COUNT(*)`. Verify counts.
**Module:** `query::executor`

### Q-05: PATH queries — shortest path
**Claim:** `MATCH PATH SHORTEST (a)-[*]->(b)` finds the shortest causal path.
**Method:** Create diamond DAG (A->B, A->C, B->D, C->D). Verify shortest path A->D is
length 2 (either via B or C, not A->B->D->... which doesn't exist).
**Module:** `query::executor`

### Q-06: PATH queries — depth limit
**Claim:** `PATH (a)-[*1..3]->(b)` only finds paths within depth 1-3.
**Method:** Create chain A->B->C->D->E. Query *1..2 from A. Verify B and C reachable, not D or E.
**Module:** `query::executor`

### Q-07: EXPLAIN returns plan without executing
**Claim:** EXPLAIN prefix returns query plan JSON without side effects.
**Method:** Run EXPLAIN MATCH. Verify JSON plan returned and no state changed.
**Module:** `query::planner`

### Q-08: LIMIT and ORDER BY
**Claim:** ORDER BY sorts correctly, LIMIT caps result count.
**Method:** Create 10 entities. `ORDER BY e.confidence DESC LIMIT 3`. Verify top 3 by confidence.
**Module:** `query::executor`

### Q-09: IN operator
**Claim:** `WHERE e.entity_type IN ["Actor", "Location"]` returns only those types.
**Method:** Create entities of all 5 types. Filter with IN. Verify only Actor and Location returned.
**Module:** `query::executor`

### Q-10: CONTAINS operator on string fields
**Claim:** `WHERE e.name CONTAINS "Thorn"` matches "Edmund Thornwick".
**Method:** Create entities with various names. Test CONTAINS substring matching.
**Module:** `query::executor`

---

## Category 15: Anomaly Detection

### AD-01: Z-score anomaly on confidence
**Claim:** Situations with confidence significantly deviating from mean are flagged as anomalies.
**Method:** Create 20 situations: 19 with confidence 0.8 +/- 0.05, one with 0.1.
Run anomaly detection. Verify the 0.1 outlier flagged.
**Module:** `analysis::anomaly`

### AD-02: Z-score anomaly on temporal gaps
**Claim:** Unusually large temporal gaps between consecutive situations are flagged.
**Method:** Create situations at regular intervals except one large gap. Verify gap flagged.
**Module:** `analysis::anomaly`

---

## Category 16: Entity Operations

### EO-01: Entity merge — participation transfer
**Claim:** Merging entity B into A transfers all of B's participations to A.
**Method:** B participates in situations {S1, S2}. Merge B into A. Verify A now participates
in those situations and B is deleted.
**Module:** `hypergraph::entity` (merge_entities)

### EO-02: Entity split — situation subset transfer
**Claim:** Splitting entity A by situation_ids creates clone with those participations.
**Method:** A participates in {S1, S2, S3, S4}. Split with {S3, S4}. Verify original A
has {S1, S2}, clone has {S3, S4}.
**Module:** `hypergraph::entity` (split_entity)

### EO-03: Multi-role participation
**Claim:** Entity can have multiple roles in the same situation via seq field.
**Method:** Add entity to situation as Protagonist (seq=0) and as Witness (seq=1).
Verify both participations retrievable.
**Module:** `hypergraph::participation`

### EO-04: Auto-snapshot on entity update
**Claim:** `update_entity` creates StateVersion snapshot of pre-update state.
**Method:** Create entity. Update it. Verify StateVersion exists with old values.
**Module:** `hypergraph::entity`, `hypergraph::state`

---

## Category 17: Confidence & Source Intelligence

### SI-01: Bayesian confidence recomputation from sources
**Claim:** Entity confidence is recomputed from source trust scores after attribution changes.
**Method:** Create entity with confidence 0.5. Add high-trust source (0.9). Recompute.
Verify confidence increased. Add low-trust source (0.2). Recompute. Verify adjustment.
**Module:** `api::source_routes` (recompute_confidence)

### SI-02: Source trust propagation
**Claim:** Updating a source's trust_score propagates to all entities/situations it attributes.
**Method:** Source S attributes to entities {E1, E2, E3}. Change S.trust_score from 0.9 to 0.3.
Verify E1, E2, E3 confidence all decreased.
**Module:** `hypergraph` (propagate_source_trust_change)

### SI-03: Contention creation and resolution
**Claim:** Conflicting claims from different sources are tracked as contentions and resolvable.
**Method:** Source A claims "X happened at 10pm". Source B claims "X happened at 11pm".
Create contention. Resolve in favor of A. Verify resolution recorded.
**Module:** `api::source_routes`

---

## Category 18: Export Formats

### EX-01: CSV export — correct headers and row counts
**Claim:** CSV export contains entities and situations with expected columns.
**Method:** Ingest narrative. Export as CSV. Parse CSV. Verify headers, row count matches entity
+ situation count.
**Module:** `export::csv_export`

### EX-02: GraphML export — valid XML
**Claim:** GraphML export is well-formed XML with nodes and edges.
**Method:** Export narrative as GraphML. Parse as XML. Verify node/edge counts match.
**Module:** `export::graphml`

### EX-03: Archive round-trip (.tensa)
**Claim:** Export as .tensa archive and re-import produces identical data.
**Method:** Create narrative with entities, situations, participations, causal links.
Export as archive. Delete narrative. Import archive. Verify all data restored.
**Module:** `export::archive`, `ingestion::archive`

### EX-04: Manuscript export — temporal ordering
**Claim:** Manuscript export reconstructs narrative in temporal order.
**Method:** Create situations with known temporal order. Export as manuscript. Verify
situations appear in chronological order in the output markdown.
**Module:** `export::manuscript`

### EX-05: Report export — structure
**Claim:** Report export contains timeline table, entity profiles, co-occurrence matrix.
**Method:** Ingest narrative. Export as report. Verify sections present.
**Module:** `export::report`

---

## Execution Strategy

### Phase 1: Build Test Fixtures
1. Run `/build-tensa-archive` with `enrich: true` for each of the 5 narratives
2. Commit the `.tensa` files to `tests/functional/test_data/`
3. Create `mystery_ground_truth.json` with hand-annotated expected values:
   - All entities with expected types
   - All situations with expected temporal bounds
   - All causal links with expected strengths
   - All participations with expected roles
   - Expected centrality ranking
   - Expected community structure
   - Expected arc classification
   This becomes the "answer key" for all narrative-based tests.

### Phase 2: Algorithm Unit Tests (no server needed)
Tests: T-01 through T-06, C-01 through C-05, N-01 through N-07, I-01 through I-06,
E-01 through E-06, A-01 through A-04, S-01 through S-04, G-01 through G-06,
M-01 through M-05, X-01 through X-07, F-01 through F-04, SP-01 through SP-02,
V-01 through V-04, Q-01 through Q-10, AD-01 through AD-02, EO-01 through EO-04

**Approach:** Rust `#[test]` functions using temp RocksDB. No LLM. No server.
Small-scale tests build fixtures programmatically via Hypergraph API.
Narrative-scale tests load `.tensa` archives via `import_archive()` into temp RocksDB.

### Phase 3: Narrative Integration Tests (server + test data)
Tests: C-06, N-08, E-07, X-08, F-05 through F-08, SI-01 through SI-03,
EX-01 through EX-05

**Approach:** Load `.tensa` archives into a running server instance. Run analysis and
compare against ground truth. No LLM needed — archives are pre-enriched.

---

## Test Count Summary

| Category | Phase 2 (Unit) | Phase 3 (Integration) | Total |
|----------|---------------|----------------------|-------|
| Temporal Reasoning | 6 | 0 | 6 |
| Causal Inference | 5 | 1 | 6 |
| Network Analysis | 7 | 1 | 8 |
| Information Theory | 6 | 0 | 6 |
| Epistemic Reasoning | 6 | 1 | 7 |
| Argumentation | 4 | 1 | 5 |
| Contagion (SIR) | 4 | 0 | 4 |
| Game Theory | 6 | 0 | 6 |
| Motivation Analysis | 5 | 0 | 5 |
| Cross-Narrative | 7 | 1 | 8 |
| Stylometry/Fingerprint | 4 | 4 | 8 |
| Spatial Reasoning | 2 | 0 | 2 |
| Vector Similarity | 4 | 0 | 4 |
| TensaQL Query Engine | 10 | 0 | 10 |
| Anomaly Detection | 2 | 0 | 2 |
| Entity Operations | 4 | 0 | 4 |
| Confidence/Sources | 0 | 3 | 3 |
| Export Formats | 0 | 5 | 5 |
| **TOTAL** | **82** | **17** | **99** |
