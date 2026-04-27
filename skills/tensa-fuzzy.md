---
name: tensa-fuzzy
description: Fuzzy logic surface over TENSA — t-norms, aggregators, fuzzy Allen relations, intermediate quantifiers, graded Peterson syllogisms, fuzzy FCA, Mamdani rules, and fuzzy-probabilistic hybrid inference. Activate when the user wants to combine graded confidences, aggregate source support under different semantics, verify a graded syllogism, or build a concept lattice.
---

# TENSA Fuzzy Logic Assistant

Activate this skill when the user wants to:

- **Combine graded confidences under an explicit logical semantics**
  (Gödel / Goguen / Łukasiewicz / Hamacher) rather than the default
  site wiring.
- **Aggregate N source confidences** under OWA (linguistic-quantifier
  weighted) or Choquet integral (source-interaction-aware) aggregation.
- **Ask "do most sources support X?"** — fuzzy intermediate quantifiers
  (Novák-Murinová) replace ad-hoc thresholds.
- **Verify a graded syllogism** over a narrative's entity types (Peterson's
  five figures; "Most A are B; all B are C").
- **Build a concept lattice** of entities vs. tags (fuzzy Formal Concept
  Analysis; Bělohlávek / Krídlo-Ojeda).
- **Evaluate Mamdani-style fuzzy rules** against an entity (triangular /
  trapezoidal / Gaussian membership functions; centroid / mean-of-maxima
  defuzzification).
- **Compute `P_fuzzy(E)`** — the fuzzy probability of a fuzzy event under
  a discrete distribution (Sugeno-additive base case, scope-capped per
  `docs/fuzzy_hybrid_algorithm.md`).
- **Switch the t-norm on any analysis / RAG / reranker / contagion /
  opinion-dynamics endpoint** via the opt-in `?tnorm=<kind>&aggregator=<kind>`
  query string. The default path stays bit-identical to pre-sprint TENSA.

The sprint ships on top of TENSA v0.78.0. Full chapter reference is
`docs/TENSA_REFERENCE.md` "Fuzzy Logic".

## Key concepts

### The four canonical t-norms

A **t-norm** `T : [0,1]² → [0,1]` models fuzzy conjunction (logical AND
on graded truth values). All four satisfy commutativity, associativity,
monotonicity, and `T(a,1) = a`.

| T-norm | `T(a, b)` | Use when |
|---|---|---|
| **Gödel** (default) | `min(a, b)` | The overall strength is only as strong as the weakest input. Safest choice. |
| **Goguen** (product) | `a · b` | Sources are probabilistically independent (Dempster's rule of combination uses this). |
| **Łukasiewicz** | `max(0, a + b − 1)` | You want "too low + too low = impossible" strictness. More punitive than Gödel at low inputs. |
| **Hamacher(λ)** | `(a · b) / (λ + (1−λ)(a + b − a · b))`, λ ≥ 0 | Need a tunable knob between product (λ=1) and drastic (λ→0). |

Ordering at every point: `Łukasiewicz ≤ Goguen ≤ Gödel`. Choose
Łukasiewicz for strictness, Gödel for leniency, Goguen for independence.

**De Morgan duals** (t-conorms) compute fuzzy OR via
`S(a, b) = 1 − T(1−a, 1−b)` — picked automatically on the same knob.

### Default-Gödel invariant

Every TENSA surface that accepts `?tnorm=<kind>` defaults to Gödel
when the parameter is omitted — with ONE exception: pre-existing
Dempster / Yager mass combination in `analysis::evidence` defaults to
Goguen (product) because that is the canonical mathematical form. The
t-norm of record is tagged in every response JSON; if the user sees
`"tnorm": null` they are looking at the site default.

### Aggregators

Aggregators take N fuzzy values and reduce them to one:

| Kind | What it does |
|---|---|
| `Mean` | Arithmetic mean |
| `Median` | Middle value |
| `TNormReduce` | Fold via the selected t-norm (AND of all) |
| `TConormReduce` | Fold via the t-conorm (OR of all) |
| `Owa` (Yager 1988) | Sort xs descending, weighted sum. `weights: [w_1, ..., w_n]` with `Σw = 1`. |
| `Choquet` (Grabisch 1996) | Integral against a fuzzy measure μ; captures source interaction. |

**OWA linguistic-quantifier weights** ship for free: `Most` (the middle
80% dominate), `AlmostAll` (the top few), `Few` (the bottom few).
Helper: `w_i = Q(i/n) - Q((i-1)/n)`.

**Choquet with a symmetric-additive measure recovers the arithmetic
mean**. Symmetric-pessimistic recovers `min`; symmetric-optimistic
recovers `max`. Use non-symmetric (standard) measures when sources
corroborate or mask each other — e.g. two independent cable-news
mirrors count as one when μ(both) ≈ μ(either).

### Measure cap

Choquet's exact path uses a `2^n`-byte lookup table. Hard cap at
`n = 16`; exact computation caps at `n = 10`. Above 10 the engine
switches to Monte-Carlo (k = 1000 permutations, deterministic seed)
and returns `ChoquetResult { value, std_err }`.

## TensaQL cheat sheet

All new clauses accept optional `WITH TNORM '<kind>'` and/or
`AGGREGATE <kind>` tails. Where both apply, t-norm fuses condition
groups; aggregator fuses evaluation results.

```sql
-- T-norm + aggregator on a regular query
MATCH (e:Actor)
WHERE (e.confidence > 0.7 AND e.maturity = "Validated")
   OR e.narrative_id = "n1"
RETURN e
WITH TNORM 'lukasiewicz' AGGREGATE mean

-- Fuzzy Allen relation tail
MATCH (s:Situation)
AT s.temporal AS FUZZY OVERLAPS THRESHOLD 0.3
RETURN s

-- Intermediate quantifier (do MOST actors in n1 have high confidence?)
QUANTIFY most (e:Actor) WHERE e.confidence > 0.7 FOR "n1" AS "partisanship"

-- Graded Peterson syllogism verification
VERIFY SYLLOGISM {
    major:      'Most Actor IS Influential',
    minor:      'All Influential IS Leader',
    conclusion: 'Most Actor IS Leader'
} FOR "n1" THRESHOLD 0.7 WITH TNORM 'godel'

-- Fuzzy FCA — build a concept lattice of Actors vs tags
FCA LATTICE FOR "n1"
    ATTRIBUTES ['partisan', 'trusted', 'disputed']
    ENTITY_TYPE Actor
    WITH TNORM 'godel'

FCA CONCEPT 3 FROM "<lattice_id>"

-- Mamdani rule evaluation (triangular/trapezoidal/Gaussian MF)
EVALUATE RULES FOR "n1"
    AGAINST (e:Actor)
    RULES ['elevated-disinfo-risk']
    WITH TNORM 'godel'

-- Fuzzy probability of a fuzzy event under a discrete distribution
INFER FUZZY_PROBABILITY(
    event_kind := 'quantifier',
    event_ref  := '{"kind":"most","predicate":"e.confidence > 0.7"}',
    distribution := 'uniform'
) FOR "n1" WITH TNORM 'godel'
```

Opt-in per-endpoint query string for every existing confidence-returning
route:

```
GET  /entities?tnorm=lukasiewicz&aggregator=owa
GET  /entities/{id}?tnorm=godel
POST /ask       body: { ..., tnorm: "lukasiewicz", aggregator: "owa" }
POST /analysis/opinion-dynamics     ?tnorm=<kind>     — consensus under alternate semantics
POST /analysis/higher-order-contagion ?tnorm=<kind>   — fuzzy threshold rule
```

## REST quick-reference

| Method | Path | Use |
|---|---|---|
| GET | `/fuzzy/tnorms` / `/fuzzy/tnorms/:kind` | Registered t-norm catalogue |
| GET | `/fuzzy/aggregators` / `/fuzzy/aggregators/:kind` | Registered aggregator catalogue |
| POST | `/fuzzy/measures` | Register a fuzzy measure (monotonicity-checked) |
| GET / DELETE | `/fuzzy/measures` / `/fuzzy/measures/:name` | List / drop |
| GET / PUT | `/fuzzy/config` | Site-default t-norm + aggregator |
| POST | `/fuzzy/aggregate` | One-shot aggregation: `{xs, aggregator, tnorm?, weights?, measure?}` |
| POST / GET | `/analysis/fuzzy-allen` | Compute + cache graded Allen 13-vector |
| POST / GET / DELETE | `/fuzzy/quantify` | Evaluate intermediate quantifier over WHERE |
| POST / GET | `/fuzzy/syllogism/verify` + `/fuzzy/syllogism/:nid/:proof_id` | Graded Peterson verification |
| POST | `/fuzzy/fca/lattice` | Build concept lattice |
| GET / DELETE | `/fuzzy/fca/lattice/:id` + `/fuzzy/fca/lattices/:nid` | List / inspect / drop |
| POST / GET / DELETE | `/fuzzy/rules` + `/fuzzy/rules/:nid` + `/fuzzy/rules/:nid/:rid` | Mamdani rule CRUD |
| POST | `/fuzzy/rules/:nid/evaluate` | Fire rule set against matched entities |
| POST / GET / DELETE | `/fuzzy/hybrid/probability` + `/fuzzy/hybrid/probability/:nid` + `.../:nid/:qid` | Fuzzy-probabilistic hybrid |

## MCP tools (14 fuzzy tools)

| Tool | Wraps |
|---|---|
| `fuzzy_list_tnorms` | `GET /fuzzy/tnorms` |
| `fuzzy_list_aggregators` | `GET /fuzzy/aggregators` |
| `fuzzy_get_config` / `fuzzy_set_config` | `GET / PUT /fuzzy/config` |
| `fuzzy_create_measure` / `fuzzy_list_measures` | `POST / GET /fuzzy/measures` |
| `fuzzy_aggregate` | `POST /fuzzy/aggregate` — the one-shot calculator |
| `fuzzy_allen_gradation` | `POST /analysis/fuzzy-allen` |
| `fuzzy_quantify` | `POST /fuzzy/quantify` |
| `fuzzy_verify_syllogism` | `POST /fuzzy/syllogism/verify` |
| `fuzzy_build_lattice` | `POST /fuzzy/fca/lattice` |
| `fuzzy_create_rule` / `fuzzy_evaluate_rules` | `POST /fuzzy/rules` + `/evaluate` |
| `fuzzy_probability` | `POST /fuzzy/hybrid/probability` |

Existing tools that return confidence now accept optional `tnorm` +
`aggregator` args (`get_entity`, `search_entities`, `list_pinned_facts`,
`ask`, `get_narrative_stats`, `get_behavioral_fingerprint`,
`get_disinfo_fingerprint`).

## Three load-bearing idioms

Learn these; they're the most common user intents.

### 1. "What's the aggregated confidence of this claim under Gödel vs Łukasiewicz?"

```sql
-- Gödel (default) — weakest input wins
MATCH (e:Actor) WHERE e.id = "<uuid>" RETURN e WITH TNORM 'godel'

-- Łukasiewicz — "too low + too low = impossible"
MATCH (e:Actor) WHERE e.id = "<uuid>" RETURN e WITH TNORM 'lukasiewicz'
```

Or via REST for the confidence-rollup endpoints:

```
POST /entities/{id}/recompute-confidence?tnorm=godel
POST /entities/{id}/recompute-confidence?tnorm=lukasiewicz
```

The response JSON carries `fuzzy_config: { tnorm, aggregator }` so the
user sees the semantics of record. Difference between answers is
your signal — if Gödel and Łukasiewicz agree, the claim's support is
uniform; if they diverge, at least one source is very weak.

### 2. "Do most sources support the claim?"

```sql
QUANTIFY most (e:Actor) WHERE e.confidence > 0.7 FOR "n1" AS "majority_support"
```

Or one-shot via MCP:

```
fuzzy_quantify(narrative_id="n1", quantifier="most",
               predicate="e.confidence > 0.7")
```

Returns a scalar in `[0, 1]` — the ramp evaluation of the quantifier
"most" against the fraction of matching entities. `Q_most` ramps up
over `[0.3, 0.8]`, so:
- `r = 0.2` → `0.0` (clearly *not* most)
- `r = 0.55` → `0.5` (borderline)
- `r = 0.9` → `1.0` (clearly most)

Don't confuse this with a simple majority (> 50%). The ramp is
deliberately conservative; Novák-Murinová calibrate it so "most" does
not collapse to "just above half".

### 3. "Does the syllogism 'Most A are B; all B are C' conclude?"

```sql
VERIFY SYLLOGISM {
    major:      'Most Actor IS Influential',
    minor:      'All Influential IS Leader',
    conclusion: 'Most Actor IS Leader'
} FOR "n1"
```

Returns:

```json
{
    "degree": 0.73,
    "figure": "FigureI",
    "valid": true,
    "threshold": 0.5
}
```

Peterson's five figures (Figure I through Figure V) determine which
quantifier-triples are syntactically admissible. **Figure II with
non-canonical quantifier ordering always reports `valid = false`
regardless of the computed degree** — this is the Peterson taxonomy
talking, not the fuzzy math. When the user gets an unexpected `false`,
check the figure hint in the response.

The DSL is exact: `"<QUANTIFIER> <SUBJECT> IS <OBJECT>"`. Quantifiers:
`All`, `Most`, `Many`, `Few`, `AlmostAll`, `Some`, `No`. Subjects
and objects resolve via `type:<EntityType>` or literal entity tags.

## When NOT to use fuzzy logic

- **Don't apply a non-default t-norm to the Dempster-Shafer evidence
  mass combination unless you understand the consequences.** DS defaults
  to Goguen because that's the canonical mathematical form. Switching
  to Gödel changes the output distribution and invalidates comparison
  with any cited DS reference.
- **Don't use Choquet with symmetric-additive measures.** It recovers
  the arithmetic mean exactly. Use `AggregatorKind::Mean` — it's
  cheaper and clearer.
- **Don't expect the fuzzy-probability surface to handle continuous
  distributions.** Phase 10 is scope-capped at discrete distributions;
  continuous is deferred to Phase 10.5.
- **Don't expect formal Łukasiewicz-BL* soundness from the syllogism
  verifier.** It's a prototype; Phase 7.5 owns the full formal layer.
  Use the degree as a heuristic, not as a logical proof.
- **Don't switch t-norms on a calibrated fidelity report without
  re-calibrating.** `FidelityThresholds` are fit under the default
  wiring; a non-default t-norm produces metric values outside the
  threshold envelope.

## Paper pointers (full bibliography: `docs/FUZZY_BIBLIOGRAPHY.bib`)

- T-norms — Klement, Mesiar, Pap 2000 (`klement2000`).
- OWA — Yager 1988 (`yager1988owa`).
- Choquet integral — Grabisch 1996 (`grabisch1996choquet`);
  Bustince et al. 2016 (`bustince2016choquet`).
- Fuzzy measures — Grabisch, Murofushi, Sugeno 2000 (`grabisch2000fuzzymeasure`).
- Fuzzy Allen — Dubois-Prade 1989 (`duboisprade1989fuzzyallen`);
  Schockaert-De Cock 2008 (`schockaert2008fuzzyallen`).
- Intermediate quantifiers — Novák 2008 (`novak2008quantifiers`).
- Peterson's graded syllogisms — Murinová-Novák 2012/2014
  (`murinovanovak2013syllogisms`, `murinovanovak2014peterson`).
- Fuzzy FCA — Bělohlávek 2004 (`belohlavek2004fuzzyfca`);
  Krídlo-Ojeda-Aciego 2010 (`kridlo2010fuzzyfca`).
- Mamdani rules — Mamdani-Assilian 1975 (`mamdani1975mamdani`).
- Fuzzy-probabilistic logics — Flaminio-Holčapek-Cao 2026
  (`flaminio2026fsta`); Fagin-Halpern 1994 (`faginhalpern1994fuzzyprob`).

See `docs/TENSA_REFERENCE.md` "Fuzzy Logic" chapter for the full spec,
defaults, and surface taxonomy.
