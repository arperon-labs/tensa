---
name: tensa-graded
description: Graded acceptability + measure learning over TENSA — gradual / ranking-based argumentation semantics (h-Categoriser, weighted h-Categoriser, max-based, card-based), supervised Choquet measure learning from labelled clusters, and ORD-Horn path-consistency closure for temporal constraint networks. Activate when the user asks about argumentation strength, ranking-based reasoning, learned aggregation measures, or temporal-network satisfiability.
---

# TENSA Graded Acceptability Assistant

Activate this skill when the user wants to:

- **Score argument strength on a continuum** rather than the crisp Dung
  In/Out/Undec labels — gradual / ranking-based semantics return a
  degree in `[0, 1]` per argument.
- **Train a fuzzy aggregation measure from data** — when the symmetric
  defaults (additive / pessimistic / optimistic) miss source-source
  interaction (e.g. coordinated inauthentic behaviour clusters where
  `μ({A, B}) ≠ μ(A) + μ(B)`).
- **Decide whether a disjunctive Allen network is satisfiable** — the
  Nebel-Bürckert 1995 ORD-Horn closure runs path-consistency in `O(n³)`,
  sound for any Allen network, complete on the ORD-Horn subfragment.

The sprint ships on top of TENSA v0.78.7. Full chapter reference:
`docs/TENSA_REFERENCE.md` Chapter 15 "Graded Acceptability & Measure
Learning". Bibliography: `docs/FUZZY_BIBLIOGRAPHY.bib` (5 new entries
land at the end: `amgoud2013ranking`, `amgoud2017weighted`,
`besnard2001hcategoriser`, `grabisch1997kadditive`, `nebel1995ordhorn`).

## Gradual Argumentation

The legacy `analysis::argumentation` module returns Dung's three crisp
labels. The new `analysis::argumentation_gradual` module returns a
real-valued **acceptability degree** per argument, computed by iterating
a contracting update map until `||Acc_{i+1} - Acc_i||_∞ < 1e-9` (or
`MAX_GRADUAL_ITERATIONS = 200` is hit).

### When to pick which semantics

| Variant | Formula | When to pick |
|---|---|---|
| **h-Categoriser** | `Acc_{i+1}(a) = w(a) / (1 + Σ Acc_i(b))` | Default. Smooth, contracting under any bounded attack count. Strongest published precedent (Besnard & Hunter 2001). |
| **Weighted h-Categoriser** | `Acc_{i+1}(a) = w(a) / (1 + Σ v_{ba} · Acc_i(b))` | When attacks have non-uniform strength (e.g. derived from source trust). Construction asserts `Σ_b v_{ba} ≤ 1` per target — Amgoud & Ben-Naim 2017 §3 Remark 1. |
| **Max-Based** | replaces sum with max over attackers | When a single overwhelming attacker should silence the argument regardless of how many weaker ones pile on. |
| **Card-Based** | `Acc_{i+1}(a) = w(a) / ((1 + card⁺) · (1 + sum))` | When you want **both** "how many attackers" (cardinality) and "how strong" (sum) to count. Lexicographic tie-break. Reaches a fixed point in `O(|A|)` rounds (Amgoud & Ben-Naim 2013 §4.3 Proposition 3). |

### Convergence guarantees

| Semantics × T-norm | Contraction provable? |
|---|---|
| `{HCategoriser, Weighted, MaxBased, CardBased} × Gödel` | Yes (canonical form). |
| `{HCategoriser, Weighted, MaxBased, CardBased} × Łukasiewicz` | Yes (clamped to `[0, 1]`). |
| `... × Goguen` (`1 - exp(-s)`) | NOT guaranteed; relies on the cap. |
| `... × Hamacher(λ)` | NOT guaranteed; relies on the cap. |

Non-contracting cells return `converged: false` + `tracing::warn!` —
the result reflects the last computed iterate. Always check
`gradual.converged` before consuming the acceptability map for a
publication-grade claim.

### Influence-step t-norm coupling (option B)

The standard h-Categoriser influence function is `infl(s) = s / (1 + s)`.
Option B from feedback was: KEEP the canonical aggregation step
verbatim (sum / max / card) and expose a t-norm-parametric
**influence-step denominator family**:

| `TNormKind` | denominator term `infl_denom(s)` |
|---|---|
| `Godel` (default) | `s` (canonical form, bit-identical to the cited papers) |
| `Lukasiewicz` | `s.min(1.0)` |
| `Goguen` | `1 - exp(-s)` (Poisson-sum limit of probabilistic OR) |
| `Hamacher(λ)` | `S_Hamacher(s.clamp(0,1), s.clamp(0,1))` |

For card-based the `(1 + card)` factor stays raw — the t-norm only
modulates the sum-component denominator term. Default `tnorm = None`
reproduces the cited paper formulas bit-identically.

### Load-bearing idiom: "Which arguments survive under h-Categoriser semantics?"

```
POST /analysis/argumentation/gradual
{
    "narrative_id": "case-alpha",
    "gradual_semantics": "HCategoriser",
    "tnorm": null
}
→ {"narrative_id": "...", "gradual": {"acceptability": {<uuid>: 0.42, ...},
                                       "iterations": 17,
                                       "converged": true},
   "iterations": 17, "converged": true}
```

A degree `≥ 0.5` is the conventional "accepted" cutoff but is
**convention only** — Amgoud & Ben-Naim 2013 leave the cutoff to the
caller. For binary decisions on top of degrees, prefer
`max-based + threshold` over `card-based + threshold` because max-based
is more robust to attacker count manipulation.

### Argumentation principle tests

The implementation passes the three Amgoud & Ben-Naim 2013 principle
property tests (anonymity, independence, monotonicity) on 30
ChaCha8Rng-seeded random frameworks (3 properties × 10 frameworks).
See `analysis::argumentation_gradual_tests::§5.5`.

## Learned Measures

The legacy `fuzzy::aggregation::Choquet` aggregator accepts a fuzzy
measure `μ` directly. The shipped symmetric defaults (`additive`,
`pessimistic`, `optimistic`) cover the *symmetric* corner — they treat
every source identically. **Use a learned measure when sources
non-additively interact**: two cable-news mirrors that count as one,
or a coordination cluster where `μ({A, B}) > μ(A) + μ(B)`.

### How to label a dataset for ranking-supervised induction

Inputs:

- `n` — number of signals (sources). Hard cap `n ≤ 6` — k-additive
  specialisation per `[grabisch1997kadditive]` extends to `n > 6` in a
  future sprint.
- `dataset` — `[(input_vec: Vec<f64>, rank: usize)]`. Lower rank means
  "more strongly coordinated"; ties on the rank field are allowed.
- `dataset_id` — string used to seed the deterministic 50/50 train/test
  split via `ChaCha8Rng`.

Output: a `FuzzyMeasure` with stamped `measure_id` + `measure_version`.
Train/test AUCs are returned for sanity-check.

### Synthetic-CIB worked example

The shipped synthetic-CIB generator
(`fuzzy::synthetic_cib_dataset::generate_synthetic_cib(seed, n_clusters)`)
produces 4-signal CIB clusters with the ground-truth score
`sigmoid(2·x0·x1 + 0.3·x2 - 0.5·x3)` and rank assignment by descending
score. The `2·x0·x1` term is the load-bearing non-additive interaction
that no additive measure can recover.

Demonstration result (see `aggregation_learn_tests::synthetic_cib_demonstration`):

| Aggregator | AUC on test split |
|---|---|
| `symmetric_additive` (= arithmetic mean) | **0.6367** |
| `learn_choquet_measure(4, generate_synthetic_cib(42, 100), …)` | **0.8522** |
| Gap | **+0.2155** |

That 0.21 AUC gap is the §5.3 paper-figure surface — recruitment-
positive feedback flagged this as the single most visible win of the
sprint.

### Load-bearing idiom: "Train a coordination measure from these labelled clusters."

```
POST /fuzzy/measures/learn
{
    "name": "telegram-cib-2026-q1",
    "n": 4,
    "dataset": [{"input_vec": [0.7, 0.6, 0.4, 0.2], "rank": 0}, ...],
    "dataset_id": "tg-cib-q1-v1"
}
→ 201 CREATED
  {"name": "telegram-cib-2026-q1",
   "version": 1,
   "n": 4,
   "provenance": {"Learned": {"dataset_id": "tg-cib-q1-v1", ...}},
   "train_auc": 0.873, "test_auc": 0.851}
```

Re-train under the same name → version auto-increments. Versioned
history is preserved; the unversioned `fz/tn/measures/{name}` key
points at the latest. Use `?version=N` on GET / DELETE for explicit
version pinning.

### k-additive deferral

`n > 6` is rejected with the canonical k-additive pointer error.
k-additive specialisation reduces the parameter count from `2^n` to
`O(n^k)` (Grabisch 1997 — `[grabisch1997kadditive]`). See
`docs/architecture_paper` §12.2 for the planned extension; the
recruitment-positive item is the existing `n ≤ 6` PGD baseline.

### Provenance contract (LOAD-BEARING)

Every aggregation result that uses a learned measure carries
`fuzzy_config.measure_id` + `fuzzy_config.measure_version` in the
emitted envelope. Symmetric defaults emit `None`/`None` —
**bit-identical to pre-Phase-0 envelopes**. Three workflow surfaces
threaded the slot through their `_tracked` siblings in Phase 2:
`ConfidenceBreakdown::composite_with_aggregator_tracked`,
`aggregate_metrics_with_aggregator_tracked`,
`RewardProfile::score_with_aggregator_tracked`. Use these whenever the
caller cares about audit + reproducibility.

## ORD-Horn

Allen's interval algebra has 13 basic relations and `2^13 = 8192`
disjunctive subsets. The full algebra's satisfiability problem is
NP-complete (Vilain & Kautz 1986). **ORD-Horn is the maximal tractable
subclass** — the 868-element subset for which path-consistency is
both sound and complete (Nebel & Bürckert 1995, JACM 42:1, 43–66).

### When to call POST /temporal/ordhorn/closure

When you have a disjunctive Allen constraint network and you want a
**polynomial-time satisfiability decision** (rather than the legacy
crisp `IntervalTree` queries that only handle individual relations).

Example: a deposition timeline where one witness says "the call was
**before or while** the meeting" and another says "the meeting was
**after** the call". The path-consistency closure tightens the
disjunction to its strongest consistent subset.

```
POST /temporal/ordhorn/closure
{
    "network": {
        "n": 3,
        "constraints": [
            {"a": 0, "b": 1, "relations": ["Before", "Overlaps"]},
            {"a": 1, "b": 2, "relations": ["Before"]}
        ]
    }
}
→ {"closed_network": {<dense propagated form>},
   "satisfiable": true}
```

### Soundness vs completeness distinction

This is the load-bearing caveat (Nebel-Bürckert Theorem 1):

- **Sound for any Allen constraint network** — if the closure produces
  an empty constraint at any cell, the network is provably
  unsatisfiable.
- **Complete only for ORD-Horn networks** — if every constraint's
  disjunction lies inside the 868-element ORD-Horn class, then a
  non-empty closure proves satisfiability.

For general Allen networks the closure may report "satisfiable" when
the network is actually unsatisfiable — additional backtracking search
is required. **The 868-element ORD-Horn membership oracle is NOT
shipped this sprint** — callers that need decidability guarantees must
restrict their inputs to ORD-Horn by construction (Pointisable Allen
relations, or only the "convex" subset).

### Why we don't drive real intervals through the closure

The canonical 13×13 composition table inside
[`src/temporal/interval.rs`](../src/temporal/interval.rs) has known
incompleteness at certain entries (e.g. `Starts ∘ Contains = {Before,
Meets, Overlaps}` does not include the legitimate `{FinishedBy,
Contains}` outcomes). For real intervals where the actual relation is
e.g. `FinishedBy`, the closure would intersect `{FinishedBy}` with the
table-derived `{Before, Meets, Overlaps}` and return `∅` (false
unsatisfiability). Composition-table audit + correction is out of
scope for this sprint. **The REST endpoint is sufficient** — callers
construct any `OrdHornNetwork` they wish and POST it; no `IntervalTree`
integration required.

### Load-bearing idiom: "Is this temporal constraint network satisfiable under ORD-Horn?"

```
POST /temporal/ordhorn/closure
{"network": {"n": 3,
             "constraints": [
                 {"a": 0, "b": 1, "relations": ["Before"]},
                 {"a": 1, "b": 2, "relations": ["Before"]},
                 {"a": 0, "b": 2, "relations": ["After"]}
             ]}}
→ {"closed_network": {...with empty cell...}, "satisfiable": false}
```

The Before-Before-After cycle is unsatisfiable; the closure correctly
empties at least one cell.

## Cheat sheet

### REST endpoints (5 new)

| Method | Path | Use |
|---|---|---|
| POST | `/analysis/argumentation/gradual` | Sync gradual semantics: `{narrative_id, gradual_semantics, tnorm?}` → `GradualResult`. |
| POST | `/fuzzy/measures/learn` | Train a Choquet measure from a labelled dataset; returns `{name, version, train_auc, test_auc}`. |
| GET | `/fuzzy/measures/{name}/versions` | List versions for a measure (sorted ascending). |
| GET | `/fuzzy/measures/{name}?version=N` | Get a specific version (omit `version` for latest pointer). |
| DELETE | `/fuzzy/measures/{name}?version=N` | Delete a specific version (latest pointer untouched). |
| POST | `/temporal/ordhorn/closure` | Sync van Beek path-consistency closure on `OrdHornNetwork`. |

### MCP tools (5 new — count 173 → 178)

| Tool | Wraps |
|---|---|
| `argumentation_gradual` | `POST /analysis/argumentation/gradual` |
| `fuzzy_learn_measure` | `POST /fuzzy/measures/learn` |
| `fuzzy_get_measure_version` | `GET /fuzzy/measures/{name}?version=N` |
| `fuzzy_list_measure_versions` | `GET /fuzzy/measures/{name}/versions` |
| `temporal_ordhorn_closure` | `POST /temporal/ordhorn/closure` |

### TensaQL — REST-only by design

**No new TensaQL clauses this sprint.** The existing
`AGGREGATE CHOQUET BY '<measure_id>'` form already resolves learned
measures via the `FuzzyMeasure.measure_id` slot wired in Phase 2 + the
`parse_fuzzy_config` extension that accepts
`?measure=<name>&measure_version=<N>`. Learning is a control-plane
action (POST), not a query verb. Gradual semantics is a synchronous
endpoint (mirrors the Fuzzy Sprint Phase 7b `/analysis/higher-order-
contagion` precedent) — adding a `?gradual=<kind>` to the cacheable
read-back `GET /narratives/:id/arguments` was rejected because the
read-back must stay idempotent + cacheable.

## Three load-bearing idioms (verbatim)

Learn these — they're the most common user intents.

1. **"Which arguments survive under h-Categoriser semantics?"** —
   `POST /analysis/argumentation/gradual` with
   `gradual_semantics: "HCategoriser"`. Convention: degree `≥ 0.5` ⇒
   accepted; check `converged` before consuming.

2. **"Train a coordination measure from these labelled clusters."** —
   `POST /fuzzy/measures/learn` with `n ≤ 6` and a labelled
   `(input_vec, rank)` dataset. The synthetic-CIB demo lands at
   `learned_auc=0.85`, `additive_auc=0.64`, `gap=0.21` — that gap is
   what learned measures buy you.

3. **"Is this temporal constraint network satisfiable under ORD-Horn?"** —
   `POST /temporal/ordhorn/closure` with an `OrdHornNetwork`. Empty
   cell ⇒ provably unsatisfiable. Non-empty ⇒ satisfiable IFF every
   input disjunction was in the 868-element ORD-Horn class (oracle not
   shipped).

## When NOT to use this skill

- **Don't use gradual semantics for a binary "accept/reject" decision
  without thinking about the cutoff.** The default `≥ 0.5` rule is
  convention only; for a publication-grade claim, pick a cutoff
  derived from domain calibration and document it.
- **Don't use a learned measure on a different signal universe than
  the one it was trained on.** `n` is fixed at training time; signal
  ordering matters (the `μ` lookup table is keyed on subsets of
  `{0, ..., n-1}`).
- **Don't trust ORD-Horn closure satisfiability for non-ORD-Horn
  inputs without backtracking search.** Sound but incomplete.
- **Don't replace the legacy crisp `analysis::argumentation` engine.**
  Dung extensions (grounded / preferred / stable) and gradual
  semantics answer different questions. Gradual semantics returns a
  degree per argument; Dung returns a labelling (sets of accepted
  arguments). Use both; they corroborate.
- **Don't switch t-norms on the influence step without understanding
  the convergence trade-off.** Goguen + Hamacher cells rely on the
  `MAX_GRADUAL_ITERATIONS = 200` cap; consult `gradual.converged`.

## Citations (full BibTeX: `docs/FUZZY_BIBLIOGRAPHY.bib`)

- Gradual / ranking-based semantics — Amgoud & Ben-Naim 2013
  (`amgoud2013ranking`); Amgoud & Ben-Naim 2017
  (`amgoud2017weighted`); Besnard & Hunter 2001
  (`besnard2001hcategoriser`).
- k-additive fuzzy measures — Grabisch 1997 (`grabisch1997kadditive`).
- ORD-Horn maximal tractable subclass — Nebel & Bürckert 1995
  (`nebel1995ordhorn`).

See `docs/TENSA_REFERENCE.md` Chapter 15 "Graded Acceptability &
Measure Learning" for the full spec, REST + MCP surface tables, and
backward-compat regression assertions.
